#!/usr/bin/env python3
"""
Re-embed all chunks using OpenAI text-embedding-3-small and create a new LanceDB database.

Reads chunks from data/chunks.json, embeds them via OpenAI API in batches,
saves embeddings to data/embeddings_openai.npy, and creates a new LanceDB
database at data/bjj_search_db_openai/.

Usage:
    python reembed_openai.py
"""

import json
import time
from pathlib import Path

import numpy as np
import lancedb
import pyarrow as pa
from openai import OpenAI
from dotenv import load_dotenv


CHUNKS_PATH = Path("data/chunks.json")
EMBEDDINGS_PATH = Path("data/embeddings_openai.npy")
DB_PATH = Path("data/bjj_search_db_openai")
MODEL = "text-embedding-3-small"
DIMENSIONS = 512
BATCH_SIZE = 500  # ~250K tokens per batch (OpenAI max 300K tokens per request)


def embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using OpenAI API."""
    response = client.embeddings.create(
        model=MODEL,
        input=texts,
        dimensions=DIMENSIONS,
    )
    return [item.embedding for item in response.data]


def main():
    load_dotenv()
    client = OpenAI()

    # Load chunks
    print(f"Loading chunks from {CHUNKS_PATH}...")
    with open(CHUNKS_PATH) as f:
        chunks = json.load(f)
    print(f"  {len(chunks)} chunks loaded")

    # Extract texts for embedding
    texts = [chunk["text"] for chunk in chunks]

    # Embed in batches
    all_embeddings = []
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"Embedding {len(texts)} chunks in {total_batches} batches (batch_size={BATCH_SIZE})...")

    for i in range(0, len(texts), BATCH_SIZE):
        batch_num = i // BATCH_SIZE + 1
        batch = texts[i:i + BATCH_SIZE]
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} texts)...", end=" ", flush=True)

        t0 = time.time()
        embeddings = embed_batch(client, batch)
        elapsed = time.time() - t0

        all_embeddings.extend(embeddings)
        print(f"done in {elapsed:.1f}s")

    # Convert to numpy array and save
    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    print(f"\nEmbeddings shape: {embeddings_array.shape}")

    print(f"Saving embeddings to {EMBEDDINGS_PATH}...")
    np.save(EMBEDDINGS_PATH, embeddings_array)

    # Create LanceDB database
    print(f"\nCreating LanceDB database at {DB_PATH}...")
    DB_PATH.mkdir(parents=True, exist_ok=True)

    db = lancedb.connect(str(DB_PATH))

    # Build records matching existing schema
    records = []
    for chunk, embedding in zip(chunks, all_embeddings):
        records.append({
            "vector": embedding,
            "text": chunk["text"],
            "video_file": chunk.get("video_file", ""),
            "video_title": chunk.get("video_title", ""),
            "start_time": float(chunk.get("start_time", 0)),
            "end_time": float(chunk.get("end_time", 0)),
            "chunk_index": int(chunk.get("chunk_index", 0)),
            "source_transcript": chunk.get("source_transcript", ""),
        })

    # Define schema to match existing DB
    schema = pa.schema([
        pa.field("vector", pa.list_(pa.float32(), DIMENSIONS)),
        pa.field("text", pa.string()),
        pa.field("video_file", pa.string()),
        pa.field("video_title", pa.string()),
        pa.field("start_time", pa.float64()),
        pa.field("end_time", pa.float64()),
        pa.field("chunk_index", pa.int64()),
        pa.field("source_transcript", pa.string()),
    ])

    # Drop existing table if present, then create
    try:
        db.drop_table("transcripts")
    except Exception:
        pass

    print(f"Writing {len(records)} records to 'transcripts' table...")
    table = db.create_table("transcripts", data=records, schema=schema)
    print(f"  Table created with {table.count_rows()} rows")

    print("\nDone! New database ready at:", DB_PATH)


if __name__ == "__main__":
    main()
