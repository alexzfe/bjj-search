#!/usr/bin/env python3
"""
Re-chunk and Embed BJJ Transcripts (v2)

Single-pass script that:
1. Re-chunks all transcripts at ~800 tokens with sentence boundaries
2. Prepends video context (instructor + title) to each chunk for embedding
3. Embeds with OpenAI text-embedding-3-small (512 dims)
4. Builds a new LanceDB at data/bjj_search_db_v2/

Usage:
    python rechunk_and_embed.py
"""

import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import lancedb
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# =============================================================================
# Config
# =============================================================================

TRANSCRIPTS_DIR = Path("./data/transcripts")
CHUNKS_OUTPUT = Path("./data/chunks_v2.json")
EMBEDDINGS_OUTPUT = Path("./data/embeddings_v2.npy")
DB_PATH = Path("./data/bjj_search_db_v2")

TARGET_TOKENS = 800
OVERLAP_TOKENS = 100
CHARS_PER_TOKEN = 4  # approximate

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 512
EMBED_BATCH_SIZE = 300  # 300 * ~800 tokens = ~240K tokens per batch, safe under 300K


# =============================================================================
# Step 1: Sentence Splitting & Chunking
# =============================================================================

def simple_sentence_split(text: str) -> list[str]:
    """Split text on sentence boundaries ('. ' or '? ' or '! ' followed by uppercase)."""
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(text: str, target_tokens: int = TARGET_TOKENS, overlap_tokens: int = OVERLAP_TOKENS) -> list[str]:
    """Chunk text at sentence boundaries with overlap."""
    sentences = simple_sentence_split(text)
    target_chars = target_tokens * CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * CHARS_PER_TOKEN

    chunks = []
    current_chunk = []
    current_length = 0

    for sent in sentences:
        sent_length = len(sent)

        if current_length + sent_length > target_chars and current_chunk:
            chunks.append(" ".join(current_chunk))

            # Keep overlap from end of current chunk
            overlap_sents = []
            overlap_len = 0
            for s in reversed(current_chunk):
                if overlap_len + len(s) <= overlap_chars:
                    overlap_sents.insert(0, s)
                    overlap_len += len(s)
                else:
                    break

            current_chunk = overlap_sents
            current_length = overlap_len

        current_chunk.append(sent)
        current_length += sent_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def build_segment_text_and_times(transcript: dict) -> list[dict]:
    """Build list of segments with their text and timestamps."""
    segments = transcript.get("segments", [])
    result = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if text:
            result.append({
                "text": text,
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
            })
    return result


def chunk_with_segment_timestamps(
    segments: list[dict],
    target_chars: int = TARGET_TOKENS * CHARS_PER_TOKEN,
    overlap_chars: int = OVERLAP_TOKENS * CHARS_PER_TOKEN,
) -> list[dict]:
    """
    Chunk by accumulating segments until target size, tracking timestamps directly.
    Returns list of {text, start_time, end_time}.
    """
    if not segments:
        return []

    # First, join all text for sentence splitting while tracking char-to-segment mapping
    full_text = " ".join(s["text"] for s in segments)

    # Build a char offset -> segment index mapping
    char_to_seg = []
    offset = 0
    for i, seg in enumerate(segments):
        seg_len = len(seg["text"])
        for _ in range(seg_len):
            char_to_seg.append(i)
        char_to_seg.append(i)  # for the joining space
        offset += seg_len + 1

    # Sentence-split the full text
    sentences = simple_sentence_split(full_text)

    # Now chunk sentences, and find timestamps by locating each chunk in full_text
    chunks = []
    current_sents = []
    current_length = 0

    for sent in sentences:
        sent_length = len(sent)

        if current_length + sent_length > target_chars and current_sents:
            chunk_text = " ".join(current_sents)

            # Find where this chunk starts and ends in full_text
            chunk_start_pos = full_text.find(current_sents[0][:50])
            chunk_end_pos = full_text.find(current_sents[-1][-50:])
            if chunk_end_pos >= 0:
                chunk_end_pos += len(current_sents[-1][-50:])

            # Map to segment timestamps
            if chunk_start_pos >= 0 and chunk_start_pos < len(char_to_seg):
                start_seg_idx = char_to_seg[min(chunk_start_pos, len(char_to_seg) - 1)]
                start_time = segments[start_seg_idx]["start"]
            else:
                start_time = 0.0

            if chunk_end_pos >= 0 and chunk_end_pos < len(char_to_seg):
                end_seg_idx = char_to_seg[min(chunk_end_pos, len(char_to_seg) - 1)]
                end_time = segments[end_seg_idx]["end"]
            else:
                end_time = segments[-1]["end"] if segments else 0.0

            chunks.append({
                "text": chunk_text,
                "start_time": start_time,
                "end_time": end_time,
            })

            # Overlap
            overlap_sents = []
            overlap_len = 0
            for s in reversed(current_sents):
                if overlap_len + len(s) <= overlap_chars:
                    overlap_sents.insert(0, s)
                    overlap_len += len(s)
                else:
                    break

            current_sents = overlap_sents
            current_length = overlap_len

        current_sents.append(sent)
        current_length += sent_length

    # Last chunk
    if current_sents:
        chunk_text = " ".join(current_sents)
        chunk_start_pos = full_text.find(current_sents[0][:50])
        chunk_end_pos = full_text.rfind(current_sents[-1][-50:])
        if chunk_end_pos >= 0:
            chunk_end_pos += len(current_sents[-1][-50:])

        if chunk_start_pos >= 0 and chunk_start_pos < len(char_to_seg):
            start_time = segments[char_to_seg[min(chunk_start_pos, len(char_to_seg) - 1)]]["start"]
        else:
            start_time = 0.0

        if chunk_end_pos >= 0 and chunk_end_pos < len(char_to_seg):
            end_time = segments[char_to_seg[min(chunk_end_pos, len(char_to_seg) - 1)]]["end"]
        else:
            end_time = segments[-1]["end"] if segments else 0.0

        chunks.append({
            "text": chunk_text,
            "start_time": start_time,
            "end_time": end_time,
        })

    return chunks


# =============================================================================
# Step 1 (main): Process all transcripts into chunks
# =============================================================================

def process_all_transcripts() -> list[dict]:
    """Read all transcript JSONs and chunk them."""
    transcript_files = sorted(TRANSCRIPTS_DIR.rglob("*.json"))
    print(f"Found {len(transcript_files)} transcript files\n")

    all_chunks = []
    files_processed = 0
    files_failed = 0

    for i, path in enumerate(transcript_files):
        if (i + 1) % 100 == 0:
            print(f"  Chunking {i + 1}/{len(transcript_files)}...")

        try:
            with open(path, "r") as f:
                transcript = json.load(f)

            segments = build_segment_text_and_times(transcript)
            if not segments:
                continue

            # Extract metadata from path
            rel_path = path.relative_to(TRANSCRIPTS_DIR)
            instructor = rel_path.parts[0] if rel_path.parts else "Unknown"
            video_title = path.stem  # filename without .json
            video_file = str(rel_path)  # e.g. 'John Danaher/Go Further Faster/...'

            chunked = chunk_with_segment_timestamps(segments)

            for ci, chunk in enumerate(chunked):
                # Step 2: build embedding_text with video context prepended
                embedding_text = f"{instructor} - {video_title}: {chunk['text']}"

                all_chunks.append({
                    "text": chunk["text"],
                    "embedding_text": embedding_text,
                    "video_file": video_file,
                    "video_title": video_title,
                    "start_time": chunk["start_time"],
                    "end_time": chunk["end_time"],
                    "chunk_index": ci,
                    "instructor": instructor,
                })

            files_processed += 1

        except Exception as e:
            print(f"  ERROR: {path}: {e}")
            files_failed += 1

    print(f"\nChunking complete: {files_processed} files -> {len(all_chunks)} chunks")
    if files_failed:
        print(f"  ({files_failed} files failed)")

    return all_chunks


# =============================================================================
# Step 3: Embed with OpenAI
# =============================================================================

def embed_chunks(chunks: list[dict]) -> np.ndarray:
    """Embed all chunks using OpenAI text-embedding-3-small in batches of 300."""
    client = OpenAI()
    texts = [c["embedding_text"] for c in chunks]
    total_batches = (len(texts) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE

    all_embeddings = []

    for batch_num in range(total_batches):
        start = batch_num * EMBED_BATCH_SIZE
        end = min(start + EMBED_BATCH_SIZE, len(texts))
        batch = texts[start:end]

        print(f"  Embedding batch {batch_num + 1}/{total_batches} ({len(batch)} texts)...")
        t0 = time.time()

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
            dimensions=EMBEDDING_DIM,
        )

        batch_embeddings = [np.array(d.embedding, dtype=np.float32) for d in response.data]
        all_embeddings.extend(batch_embeddings)

        elapsed = time.time() - t0
        print(f"    Done in {elapsed:.1f}s")

    return np.array(all_embeddings, dtype=np.float32)


# =============================================================================
# Step 4: Build LanceDB
# =============================================================================

def build_lancedb(chunks: list[dict], embeddings: np.ndarray):
    """Create LanceDB v2 database from chunks + embeddings."""
    print(f"\nBuilding LanceDB at {DB_PATH}...")

    # Attach vectors to chunk records
    records = []
    for chunk, emb in zip(chunks, embeddings):
        records.append({
            "vector": emb.tolist(),
            "text": chunk["text"],
            "embedding_text": chunk["embedding_text"],
            "video_file": chunk["video_file"],
            "video_title": chunk["video_title"],
            "start_time": float(chunk["start_time"]),
            "end_time": float(chunk["end_time"]),
            "chunk_index": int(chunk["chunk_index"]),
            "instructor": chunk["instructor"],
        })

    db = lancedb.connect(str(DB_PATH))

    # Drop existing table if present
    try:
        db.drop_table("transcripts")
    except Exception:
        pass

    table = db.create_table("transcripts", records)
    print(f"  Created table with {len(records)} rows")

    # Create FTS index on text
    print("  Creating FTS index...")
    table.create_fts_index("text")
    print("  FTS index ready")

    return table


# =============================================================================
# Step 5: Stats
# =============================================================================

def print_stats(chunks: list[dict], embeddings: np.ndarray, num_transcripts: int):
    """Print summary statistics."""
    total_chunks = len(chunks)
    token_counts = [len(c["text"].split()) for c in chunks]
    avg_tokens = sum(token_counts) / total_chunks if total_chunks else 0
    char_counts = [len(c["text"]) for c in chunks]
    avg_chars = sum(char_counts) / total_chunks if total_chunks else 0

    # Estimate embedding cost: text-embedding-3-small is $0.02 per 1M tokens
    total_embed_chars = sum(len(c["embedding_text"]) for c in chunks)
    total_embed_tokens_est = total_embed_chars / CHARS_PER_TOKEN
    cost_estimate = (total_embed_tokens_est / 1_000_000) * 0.02

    # DB size on disk
    db_size_bytes = sum(f.stat().st_size for f in DB_PATH.rglob("*") if f.is_file())
    db_size_mb = db_size_bytes / (1024 * 1024)

    print("\n" + "=" * 60)
    print("STATS")
    print("=" * 60)
    print(f"  Transcripts processed:  {num_transcripts}")
    print(f"  Total chunks:           {total_chunks}")
    print(f"  Avg tokens/chunk:       {avg_tokens:.0f} (word-split approx)")
    print(f"  Avg chars/chunk:        {avg_chars:.0f}")
    print(f"  Embedding dimensions:   {EMBEDDING_DIM}")
    print(f"  Embeddings shape:       {embeddings.shape}")
    print(f"  Embedding cost est:     ${cost_estimate:.4f}")
    print(f"  DB size:                {db_size_mb:.1f} MB")
    print(f"  Chunks saved to:        {CHUNKS_OUTPUT}")
    print(f"  Embeddings saved to:    {EMBEDDINGS_OUTPUT}")
    print(f"  DB path:                {DB_PATH}")
    print("=" * 60)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("BJJ Transcript Re-chunk & Embed (v2)")
    print(f"  Target: ~{TARGET_TOKENS} tokens/chunk, {OVERLAP_TOKENS} token overlap")
    print(f"  Embedding: {EMBEDDING_MODEL} @ {EMBEDDING_DIM} dims")
    print("=" * 60 + "\n")

    # Step 1 + 2: chunk all transcripts with context prepended
    print("STEP 1/2: Chunking transcripts with video context...")
    chunks = process_all_transcripts()
    if not chunks:
        print("No chunks produced. Exiting.")
        return

    num_transcripts = len(set(c["video_file"] for c in chunks))

    # Save chunks
    print(f"\nSaving {len(chunks)} chunks to {CHUNKS_OUTPUT}...")
    with open(CHUNKS_OUTPUT, "w") as f:
        # Save without embedding_text bloating the file (it can be reconstructed)
        json.dump(chunks, f, indent=2)

    # Step 3: embed
    print(f"\nSTEP 3: Embedding {len(chunks)} chunks with OpenAI...")
    embeddings = embed_chunks(chunks)

    # Save embeddings
    print(f"Saving embeddings to {EMBEDDINGS_OUTPUT}...")
    np.save(EMBEDDINGS_OUTPUT, embeddings)

    # Step 4: build DB
    print("\nSTEP 4: Building LanceDB...")
    build_lancedb(chunks, embeddings)

    # Step 5: stats
    print_stats(chunks, embeddings, num_transcripts)

    print("\nDone!")


if __name__ == "__main__":
    main()
