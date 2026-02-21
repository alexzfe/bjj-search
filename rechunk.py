#!/usr/bin/env python3
"""
Re-chunking Script for BJJ Transcripts

Creates larger chunks (800-1000 tokens) with sentence boundaries
for better context preservation.

Usage:
    python rechunk.py --target-tokens 800    # Re-chunk all transcripts
    python rechunk.py --dry-run              # Preview without writing
    python rechunk.py --rebuild-db           # Re-chunk and rebuild vector DB

Requirements:
    pip install spacy sentence-transformers lancedb numpy
    python -m spacy download en_core_web_sm
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict
import hashlib

import numpy as np


def load_spacy():
    """Load spaCy model for sentence segmentation."""
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy model...")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
        import spacy
        return spacy.load("en_core_web_sm")


def simple_sentence_split(text: str) -> list[str]:
    """Fallback sentence splitting without spaCy."""
    # Split on sentence-ending punctuation followed by space and capital
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_with_sentences(
    text: str,
    target_tokens: int = 800,
    overlap_tokens: int = 100,
    nlp=None
) -> list[str]:
    """
    Chunk text at sentence boundaries.

    Args:
        text: Full transcript text
        target_tokens: Target chunk size in tokens (~4 chars/token)
        overlap_tokens: Overlap between chunks
        nlp: spaCy model (optional, falls back to regex)

    Returns:
        List of chunk texts
    """
    # Get sentences
    if nlp:
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
    else:
        sentences = simple_sentence_split(text)

    # Approximate chars per token
    target_chars = target_tokens * 4
    overlap_chars = overlap_tokens * 4

    chunks = []
    current_chunk = []
    current_length = 0

    for sent in sentences:
        sent_length = len(sent)

        # Check if adding this sentence exceeds target
        if current_length + sent_length > target_chars and current_chunk:
            # Save current chunk
            chunks.append(" ".join(current_chunk))

            # Calculate overlap: keep last N chars worth of sentences
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

    # Don't forget last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def extract_text_from_transcript(transcript: dict) -> tuple[str, list[dict]]:
    """
    Extract full text and word timestamps from transcript JSON.

    Returns:
        (full_text, words_list)
    """
    segments = transcript.get("segments", [])

    full_text_parts = []
    all_words = []

    for segment in segments:
        text = segment.get("text", "")
        full_text_parts.append(text)

        words = segment.get("words", [])
        all_words.extend(words)

    full_text = " ".join(full_text_parts)
    return full_text, all_words


def map_timestamps_to_chunk(chunk_text: str, words: list[dict], start_search_idx: int = 0) -> tuple[float, float, int]:
    """
    Find start and end timestamps for a chunk by matching words.

    Returns:
        (start_time, end_time, next_search_idx)
    """
    chunk_words = chunk_text.lower().split()
    if not chunk_words or not words:
        return 0.0, 0.0, start_search_idx

    start_time = None
    end_time = None
    word_idx = start_search_idx

    # Find first matching word for start time
    for cw in chunk_words[:10]:  # Check first 10 words
        while word_idx < len(words):
            transcript_word = words[word_idx].get("word", "").strip().lower()
            # Remove punctuation for comparison
            transcript_word_clean = re.sub(r'[^\w]', '', transcript_word)
            cw_clean = re.sub(r'[^\w]', '', cw)

            if transcript_word_clean == cw_clean or cw_clean in transcript_word_clean:
                if start_time is None:
                    start_time = words[word_idx].get("start", 0)
                end_time = words[word_idx].get("end", 0)
                word_idx += 1
                break
            word_idx += 1
        else:
            break

    # Continue to find end time
    for cw in chunk_words[10:]:
        while word_idx < len(words):
            transcript_word = words[word_idx].get("word", "").strip().lower()
            transcript_word_clean = re.sub(r'[^\w]', '', transcript_word)
            cw_clean = re.sub(r'[^\w]', '', cw)

            if transcript_word_clean == cw_clean or cw_clean in transcript_word_clean:
                end_time = words[word_idx].get("end", end_time)
                word_idx += 1
                break
            word_idx += 1

    return start_time or 0.0, end_time or 0.0, word_idx


def process_transcript(
    transcript_path: Path,
    target_tokens: int = 800,
    overlap_tokens: int = 100,
    nlp=None
) -> list[dict]:
    """
    Process a single transcript file into chunks.

    Returns:
        List of chunk dicts with text, timestamps, and metadata
    """
    with open(transcript_path, 'r') as f:
        transcript = json.load(f)

    full_text, words = extract_text_from_transcript(transcript)

    if not full_text.strip():
        return []

    # Chunk the text
    chunk_texts = chunk_with_sentences(
        full_text,
        target_tokens=target_tokens,
        overlap_tokens=overlap_tokens,
        nlp=nlp
    )

    # Build chunks with metadata
    chunks = []
    word_idx = 0
    video_file = transcript.get("source_file", str(transcript_path))

    # Extract instructor from path (first directory component after transcripts/)
    try:
        rel_path = transcript_path.relative_to(transcript_path.parents[3])  # Relative to transcripts/
        instructor = rel_path.parts[0] if rel_path.parts else "Unknown"
    except (ValueError, IndexError):
        instructor = "Unknown"

    for chunk_text in chunk_texts:
        start_time, end_time, word_idx = map_timestamps_to_chunk(chunk_text, words, word_idx)

        chunk_id = hashlib.md5(f"{video_file}:{start_time}:{chunk_text[:50]}".encode()).hexdigest()[:12]

        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text,
            "video_file": video_file,
            "start_time": start_time,
            "end_time": end_time,
            "instructor": instructor,
            "token_count": len(chunk_text.split()),  # Approximate
        })

    return chunks


def process_all_transcripts(
    transcripts_dir: Path,
    output_path: Path,
    target_tokens: int = 800,
    overlap_tokens: int = 100,
    dry_run: bool = False
) -> list[dict]:
    """
    Process all transcript files in directory.
    """
    print("Loading spaCy model...")
    try:
        nlp = load_spacy()
    except Exception as e:
        print(f"Warning: Could not load spaCy ({e}), using fallback sentence splitting")
        nlp = None

    # Find all JSON files
    transcript_files = list(transcripts_dir.rglob("*.json"))
    print(f"Found {len(transcript_files)} transcript files")

    all_chunks = []
    stats = defaultdict(int)

    for i, transcript_path in enumerate(transcript_files):
        if (i + 1) % 50 == 0:
            print(f"Processing {i + 1}/{len(transcript_files)}...")

        try:
            chunks = process_transcript(
                transcript_path,
                target_tokens=target_tokens,
                overlap_tokens=overlap_tokens,
                nlp=nlp
            )
            all_chunks.extend(chunks)
            stats["files_processed"] += 1
            stats["chunks_created"] += len(chunks)
        except Exception as e:
            print(f"Error processing {transcript_path}: {e}")
            stats["files_failed"] += 1

    print(f"\nProcessing complete:")
    print(f"  Files processed: {stats['files_processed']}")
    print(f"  Files failed: {stats['files_failed']}")
    print(f"  Total chunks: {stats['chunks_created']}")

    if all_chunks:
        avg_tokens = sum(c["token_count"] for c in all_chunks) / len(all_chunks)
        print(f"  Avg tokens/chunk: {avg_tokens:.0f}")

    if not dry_run:
        print(f"\nSaving chunks to {output_path}...")
        with open(output_path, 'w') as f:
            json.dump(all_chunks, f, indent=2)
        print("Done!")

    return all_chunks


def rebuild_vector_db(
    chunks: list[dict],
    db_path: Path,
    embedding_dim: int = 512
):
    """
    Rebuild LanceDB with new chunks.
    """
    import lancedb
    from sentence_transformers import SentenceTransformer
    import torch

    print("\nLoading embedding model...")
    embedder = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True
    )
    embedder = embedder.half().to("cuda")
    embedder.max_seq_length = 8192

    print(f"Embedding {len(chunks)} chunks...")

    # Embed in batches
    batch_size = 32
    all_embeddings = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = ["search_document: " + c["text"] for c in batch]

        if (i // batch_size + 1) % 10 == 0:
            print(f"  Batch {i // batch_size + 1}/{(len(chunks) + batch_size - 1) // batch_size}")

        with torch.no_grad():
            embeddings = embedder.encode(
                texts,
                convert_to_tensor=True,
                device="cuda",
                normalize_embeddings=True
            )
            if embedding_dim < embeddings.shape[1]:
                embeddings = embeddings[:, :embedding_dim]
            all_embeddings.append(embeddings.cpu().numpy())

    embeddings_array = np.vstack(all_embeddings)
    print(f"Embeddings shape: {embeddings_array.shape}")

    # Add embeddings to chunks
    for chunk, emb in zip(chunks, embeddings_array):
        chunk["vector"] = emb.tolist()

    # Create LanceDB table
    print(f"\nCreating LanceDB at {db_path}...")
    db = lancedb.connect(str(db_path))

    # Drop existing table if exists
    try:
        db.drop_table("transcripts")
    except Exception:
        pass

    # Create new table
    table = db.create_table("transcripts", chunks)
    print(f"Created table with {len(chunks)} rows")

    # Create FTS index
    print("Creating FTS index...")
    table.create_fts_index("text")
    print("Done!")

    return table


def main():
    parser = argparse.ArgumentParser(description="Re-chunk BJJ transcripts")
    parser.add_argument(
        "--transcripts-dir",
        default="./data/transcripts",
        help="Directory containing transcript JSON files"
    )
    parser.add_argument(
        "--output",
        default="./data/chunks_v2.json",
        help="Output path for chunked data"
    )
    parser.add_argument(
        "--db-path",
        default="./data/bjj_search_db_v2",
        help="Path for new LanceDB database"
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=800,
        help="Target tokens per chunk (default: 800)"
    )
    parser.add_argument(
        "--overlap-tokens",
        type=int,
        default=100,
        help="Overlap tokens between chunks (default: 100)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process but don't write output"
    )
    parser.add_argument(
        "--rebuild-db",
        action="store_true",
        help="Also rebuild vector database after chunking"
    )

    args = parser.parse_args()

    transcripts_dir = Path(args.transcripts_dir)
    if not transcripts_dir.exists():
        print(f"Error: Transcripts directory not found: {transcripts_dir}")
        return

    chunks = process_all_transcripts(
        transcripts_dir=transcripts_dir,
        output_path=Path(args.output),
        target_tokens=args.target_tokens,
        overlap_tokens=args.overlap_tokens,
        dry_run=args.dry_run
    )

    if args.rebuild_db and not args.dry_run:
        rebuild_vector_db(
            chunks=chunks,
            db_path=Path(args.db_path)
        )


if __name__ == "__main__":
    main()
