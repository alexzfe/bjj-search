#!/usr/bin/env python3
"""
BJJ Transcript Search v2 - Improved RAG with Heavy Reranking

Improvements over v1:
- Intent classification (EXECUTE/ESCAPE/DEFENSE/COUNTER)
- HyDE (Hypothetical Document Embeddings) for better retrieval
- Multi-pass retrieval with RRF fusion
- BGE cross-encoder reranking for semantic discrimination
- Timestamp deduplication and MMR diversity
- LLM relevance scoring before synthesis
- Answer-first synthesis with source attribution
- Dynamic context expansion

Usage:
    python bjj_rag_v2.py                           # Interactive mode
    python bjj_rag_v2.py --query "triangle escape" # Single query

Requirements:
    pip install lancedb sentence-transformers FlagEmbedding ollama numpy
"""

import os
import json
import hashlib
import argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

import numpy as np
import lancedb
from dotenv import load_dotenv

from llm import LLMClient


# =============================================================================
# Intent Classification
# =============================================================================

INTENT_CLASSIFICATION_PROMPT = """Analyze this BJJ technique query and classify the user's intent.

Query: "{query}"

Intent categories:
- EXECUTE: User wants to learn how to perform, apply, or finish a technique
- ESCAPE: User wants to learn how to get out when caught in a position or submission
- DEFENSE: User wants to learn how to prevent or stop a technique from being applied
- COUNTER: User wants to learn how to reverse a situation into their own attack
- CONCEPT: User wants to understand principles, theory, or strategy (not a specific technique)
- DRILL: User wants training exercises or repetition methods

Respond with ONLY the category name, nothing else."""

INTENT_DESCRIPTIONS = {
    "EXECUTE": "learn how to perform and finish this technique",
    "ESCAPE": "learn how to get out when caught in this position/submission",
    "DEFENSE": "learn how to prevent this technique from being applied",
    "COUNTER": "learn how to reverse this into their own attack",
    "CONCEPT": "understand principles, theory, or strategy",
    "DRILL": "find training exercises or repetition methods"
}

INTENT_EXPANSIONS = {
    "ESCAPE": "escape get out defense survive when caught in",
    "EXECUTE": "setup finish attack submission apply technique",
    "DEFENSE": "prevent stop block deny shut down",
    "COUNTER": "counter reverse sweep attack from bottom",
    "CONCEPT": "theory principle strategy philosophy approach",
    "DRILL": "drill exercise repetition training practice"
}


def classify_intent(query: str, llm: LLMClient) -> str:
    """Classify query intent using LLM."""
    response = llm.generate(
        prompt=INTENT_CLASSIFICATION_PROMPT.format(query=query),
        temperature=0.1,
        max_tokens=10,
    )

    intent = response.strip().upper()
    valid_intents = {"EXECUTE", "ESCAPE", "DEFENSE", "COUNTER", "CONCEPT", "DRILL"}
    return intent if intent in valid_intents else "EXECUTE"


# =============================================================================
# HyDE (Hypothetical Document Embeddings)
# =============================================================================

HYDE_PROMPT = """You are an experienced BJJ instructor. Write a brief instructional passage
(3-4 sentences) that would directly answer this question from a student:

Question: {query}
Intent: {intent}

Write as if explaining to a student on the mat. Include specific details about:
- Grips and hand placement
- Body positioning and weight distribution
- The sequence of movements
- Common mistakes to avoid

Passage:"""


def generate_hypothetical_document(query: str, intent: str, llm: LLMClient) -> str:
    """Generate a hypothetical ideal answer for HyDE retrieval."""
    response = llm.generate(
        prompt=HYDE_PROMPT.format(query=query, intent=intent),
        temperature=0.7,
        max_tokens=200,
    )
    return response.strip()


# =============================================================================
# Reciprocal Rank Fusion
# =============================================================================

def reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """
    Merge multiple ranked lists using RRF.

    RRF score = Σ 1/(k + rank_i) across all lists where document appears.
    k=60 is the standard constant that balances early vs late ranks.
    """
    rrf_scores = defaultdict(float)
    doc_lookup = {}

    for results in result_lists:
        for rank, doc in enumerate(results, start=1):
            # Create unique ID from video_file + start_time
            doc_id = f"{doc.get('video_file', '')}:{doc.get('start_time', 0)}"
            rrf_scores[doc_id] += 1.0 / (k + rank)
            doc_lookup[doc_id] = doc

    # Sort by RRF score
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    results = []
    for doc_id in sorted_ids:
        doc = doc_lookup[doc_id].copy()
        doc["rrf_score"] = rrf_scores[doc_id]
        results.append(doc)

    return results


# =============================================================================
# Timestamp Deduplication
# =============================================================================

def dedupe_by_timestamp(
    chunks: list[dict],
    window_seconds: float = 45.0,
    score_key: str = "rerank_score"
) -> list[dict]:
    """
    Group overlapping chunks by video and time window, keep best per group.

    45 seconds captures most single-technique segments while allowing distinct
    techniques in the same video to remain separate.
    """
    # Group by video first
    video_groups = defaultdict(list)
    for chunk in chunks:
        video_groups[chunk.get('video_file', 'unknown')].append(chunk)

    deduped = []

    for video_file, video_chunks in video_groups.items():
        # Sort by start time within each video
        video_chunks.sort(key=lambda x: x.get('start_time', 0))

        if not video_chunks:
            continue

        # Cluster overlapping chunks
        clusters = []
        current_cluster = [video_chunks[0]]

        for chunk in video_chunks[1:]:
            cluster_end = max(c.get('end_time', c.get('start_time', 0)) for c in current_cluster)

            # If this chunk starts within window of cluster's end, merge it
            if chunk.get('start_time', 0) <= cluster_end + window_seconds:
                current_cluster.append(chunk)
            else:
                # Start new cluster
                clusters.append(current_cluster)
                current_cluster = [chunk]

        clusters.append(current_cluster)  # Don't forget last cluster

        # Select highest-scoring chunk from each cluster
        for cluster in clusters:
            best = max(cluster, key=lambda x: x.get(score_key, 0))
            best["cluster_size"] = len(cluster)
            deduped.append(best)

    return sorted(deduped, key=lambda x: x.get(score_key, 0), reverse=True)


# =============================================================================
# MMR (Maximal Marginal Relevance)
# =============================================================================

def mmr_select(
    query_embedding: np.ndarray,
    candidates: list[dict],
    embeddings: np.ndarray,
    k: int = 12,
    lambda_mult: float = 0.5
) -> list[dict]:
    """
    Maximal Marginal Relevance selection.

    MMR = λ * similarity(doc, query) - (1-λ) * max(similarity(doc, selected))

    lambda_mult controls the relevance/diversity tradeoff:
    - 1.0 = pure relevance (equivalent to top-k)
    - 0.0 = pure diversity (maximally different documents)
    - 0.5 = balanced (good for seeing different instructors' takes)
    """
    if len(candidates) == 0:
        return []

    # Normalize for cosine similarity
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
    cand_norms = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)

    # Query-document similarities
    query_sims = np.dot(cand_norms, query_norm)

    selected_indices = []
    selected_embeddings = []

    for _ in range(min(k, len(candidates))):
        if not selected_indices:
            # First selection: pure relevance
            best_idx = int(np.argmax(query_sims))
        else:
            # Compute MMR scores
            selected_arr = np.array(selected_embeddings)

            # Max similarity to any selected document
            diversity_penalties = np.max(np.dot(cand_norms, selected_arr.T), axis=1)

            # MMR score
            mmr_scores = lambda_mult * query_sims - (1 - lambda_mult) * diversity_penalties

            # Mask already selected
            for idx in selected_indices:
                mmr_scores[idx] = -np.inf

            best_idx = int(np.argmax(mmr_scores))

        selected_indices.append(best_idx)
        selected_embeddings.append(cand_norms[best_idx])

    return [candidates[i] for i in selected_indices]


# =============================================================================
# Instructor Diversity
# =============================================================================

def enforce_instructor_diversity(
    chunks: list[dict],
    max_per_instructor: int = 3,
    score_key: str = "rerank_score"
) -> list[dict]:
    """
    Ensure no single instructor dominates results.
    Max 3 allows for setup, execution, and troubleshooting from each voice.
    """
    instructor_counts = defaultdict(int)
    selected = []

    # Process in score order
    for chunk in sorted(chunks, key=lambda x: x.get(score_key, 0), reverse=True):
        # Extract instructor from video path (first directory component)
        video_path = chunk.get('video_file', '')
        instructor = Path(video_path).parts[0] if video_path else 'unknown'

        if instructor_counts[instructor] < max_per_instructor:
            selected.append(chunk)
            instructor_counts[instructor] += 1

    return selected


# =============================================================================
# LLM Relevance Scoring
# =============================================================================

RELEVANCE_SCORING_PROMPT = """You are strictly evaluating search results for a BJJ technique query. Be HARSH - only high scores for exact matches.

Query: {query}
User Intent: {intent} (the user wants to {intent_description})

CRITICAL SCORING RULES:
- 9-10: ONLY if content directly addresses the EXACT technique AND matches the user's intent perfectly
- 7-8: Addresses the technique with closely related intent (escape vs defense is close)
- 5-6: Related but not what user asked for
- 0-4: WRONG INTENT - give these low scores! Examples:
  * User wants ESCAPE but content shows how to ATTACK/EXECUTE → score 0-3
  * User wants EXECUTE but content shows how to DEFEND/ESCAPE → score 0-3
  * Content about a different technique entirely → score 0-2

INTENT MATCHING IS CRITICAL:
- "triangle escape" query → content about setting up triangles = score 0-2
- "armbar from mount" query → content about escaping armbars = score 0-2
- "heel hook defense" query → content about finishing heel hooks = score 0-2

Be strict! When in doubt, score LOWER. It's better to exclude marginally relevant content than include wrong-intent content.

Sources to evaluate:
{numbered_sources}

Respond in this exact JSON format:
[
  {{"source_num": 1, "score": 9, "reason": "Directly shows triangle escape sequence"}},
  {{"source_num": 2, "score": 2, "reason": "Shows triangle SETUP - wrong intent"}}
]

JSON response:"""


def score_relevance(
    query: str,
    intent: str,
    chunks: list[dict],
    llm: LLMClient = None,
    min_score: int = 6
) -> list[dict]:
    """
    Use LLM to score each chunk's relevance, filter low scores.
    Stricter filtering (min_score=6) to reduce wrong-intent results.
    """
    if not chunks:
        return []

    # Format chunks for prompt (truncate to avoid token limits)
    numbered_sources = "\n\n".join([
        f"Source {i+1}:\n{chunk['text'][:600]}..."
        for i, chunk in enumerate(chunks[:15])  # Max 15 sources
    ])

    prompt = RELEVANCE_SCORING_PROMPT.format(
        query=query,
        intent=intent,
        intent_description=INTENT_DESCRIPTIONS.get(intent, "understand this technique"),
        numbered_sources=numbered_sources
    )

    response = llm.generate(
        prompt=prompt,
        temperature=0.1,
        max_tokens=600,
        json_output=True,
    )

    try:
        scores = json.loads(response)

        # Attach scores to chunks
        for score_entry in scores:
            idx = score_entry["source_num"] - 1
            if 0 <= idx < len(chunks):
                chunks[idx]["llm_relevance_score"] = score_entry["score"]
                chunks[idx]["relevance_reason"] = score_entry.get("reason", "")

        # Filter by minimum score
        filtered = [c for c in chunks if c.get("llm_relevance_score", 0) >= min_score]

        # Sort by LLM score
        return sorted(filtered, key=lambda x: x.get("llm_relevance_score", 0), reverse=True)

    except (json.JSONDecodeError, KeyError, TypeError):
        # Fallback: return original chunks if JSON parsing fails
        return chunks


# =============================================================================
# Answer-First Synthesis
# =============================================================================

SYNTHESIS_PROMPT = """You are an expert BJJ assistant helping practitioners find techniques in their video library.

QUERY: {query}
INTENT: {intent} (user wants to {intent_description})

RELEVANT SOURCES:
{formatted_chunks}

RESPONSE STRUCTURE (follow exactly):

## Answer
A synthesized response (2-4 paragraphs) that directly addresses the query, combining insights from the most relevant sources. Include specific technical details: grips, positioning, weight distribution, common mistakes. Reference sources inline like [Source 1] when citing specific points.

## Key Timestamps
For each highly relevant source, provide:
- **[Video Title @ MM:SS]**: One sentence describing what's taught at this timestamp

## Sources Used
Bulleted list of all sources referenced, with relevance scores.

RULES:
- Only use information from the provided sources
- If sources show different approaches, explain each
- If sources don't adequately answer the query, say so explicitly
- Be specific about techniques - vague summaries aren't helpful
- Skip sources that aren't actually relevant to the query intent"""


def format_chunks_for_synthesis(chunks: list[dict]) -> str:
    """Format chunks with clear source attribution."""
    formatted = []
    for i, chunk in enumerate(chunks, 1):
        video_path = Path(chunk.get("video_file", f"Video {i}"))
        # Build a readable title from path components
        if len(video_path.parts) >= 2:
            video_title = f"{video_path.parts[0]}/{'/'.join(video_path.parts[-2:])}"
        else:
            video_title = video_path.stem
        video_title = video_title.replace('.json', '')

        timestamp = format_timestamp(chunk.get("start_time", 0))
        relevance = chunk.get("llm_relevance_score", "N/A")

        formatted.append(f"""
---
SOURCE {i}: {video_title} @ {timestamp}
Relevance Score: {relevance}/10

{chunk['text']}
---""")

    return "\n".join(formatted)


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS or MM:SS format."""
    h, remainder = divmod(int(seconds), 3600)
    m, s = divmod(remainder, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def synthesize_answer(
    query: str,
    intent: str,
    chunks: list[dict],
    llm: LLMClient = None,
) -> str:
    """Generate answer-first synthesis with source attribution."""
    if not chunks:
        return "No relevant content found in the transcript database for this query."

    formatted_chunks = format_chunks_for_synthesis(chunks)

    prompt = SYNTHESIS_PROMPT.format(
        query=query,
        intent=intent,
        intent_description=INTENT_DESCRIPTIONS.get(intent, "understand this technique"),
        formatted_chunks=formatted_chunks
    )

    return llm.generate(
        prompt=prompt,
        temperature=0.3,
        max_tokens=1200,
    )


# =============================================================================
# Dynamic Context Expansion
# =============================================================================

def expand_context_dynamic(
    matched_chunk: dict,
    all_chunks_by_video: dict[str, list[dict]],
    window: int = 2,
    max_chars: int = 6000
) -> str:
    """
    Expand matched chunk with surrounding context from same video.
    Runtime solution without re-indexing.
    """
    video_file = matched_chunk.get("video_file", "")
    video_chunks = all_chunks_by_video.get(video_file, [])

    if not video_chunks:
        return matched_chunk.get("text", "")

    # Sort by start time
    video_chunks = sorted(video_chunks, key=lambda x: x.get("start_time", 0))

    # Find matched chunk position
    matched_time = matched_chunk.get("start_time", 0)
    idx = None
    for i, c in enumerate(video_chunks):
        if abs(c.get("start_time", 0) - matched_time) < 1:
            idx = i
            break

    if idx is None:
        return matched_chunk.get("text", "")

    # Expand window
    start_idx = max(0, idx - window)
    end_idx = min(len(video_chunks), idx + window + 1)

    # Concatenate, respecting char limit
    expanded_text = ""
    for c in video_chunks[start_idx:end_idx]:
        candidate = expanded_text + " " + c.get("text", "") if expanded_text else c.get("text", "")
        if len(candidate) <= max_chars:
            expanded_text = candidate
        else:
            break

    return expanded_text.strip()


# =============================================================================
# Main RAG Class
# =============================================================================

class BJJSearchRAGv2:
    """
    Improved RAG system with heavy reranking and semantic discrimination.

    Pipeline:
    1. Intent classification
    2. HyDE generation
    3. Multi-pass retrieval (direct + expanded + HyDE)
    4. RRF fusion
    5. Cross-encoder reranking (BGE)
    6. Timestamp deduplication
    7. MMR diversity selection
    8. LLM relevance scoring
    9. Context expansion
    10. Answer-first synthesis
    """

    def __init__(
        self,
        db_path: str,
        llm_model: str = "llama3.1:8b",
        embedding_dim: int = 512,
        top_k: int = 12,
        use_reranker: bool = True,
        profile: str = "laptop",
    ):
        self.llm_model = llm_model
        self.embedding_dim = embedding_dim
        self.top_k = top_k
        self.profile = profile

        # LLM client (dispatches to Ollama or OpenAI based on profile)
        self.llm = LLMClient(profile=profile, ollama_model=llm_model)

        # Load embedding model
        if profile == "homeserver":
            # OpenAI embeddings via LLMClient - no local model needed
            print("Using OpenAI embeddings (text-embedding-3-small)")
            self.embedder = None
        else:
            import torch
            from sentence_transformers import SentenceTransformer
            print("Loading embedding model...")
            self.embedder = SentenceTransformer(
                "nomic-ai/nomic-embed-text-v1.5",
                trust_remote_code=True
            )
            self.embedder = self.embedder.half().to("cuda")
            self._embed_device = "cuda"
            self.embedder.max_seq_length = 8192

        # Load reranker if enabled (laptop only)
        self.reranker = None
        if profile == "homeserver":
            self.use_reranker = False
        else:
            self.use_reranker = use_reranker
            if use_reranker:
                print("Loading BGE reranker...")
                try:
                    from FlagEmbedding import FlagReranker
                    self.reranker = FlagReranker(
                        "BAAI/bge-reranker-v2-m3",
                        use_fp16=True,
                        device="cuda"
                    )
                except ImportError:
                    print("Warning: FlagEmbedding not installed. Reranking disabled.")
                    print("Install with: pip install FlagEmbedding")
                    self.use_reranker = False

        # Connect to database
        print(f"Connecting to database at {db_path}...")
        self.db = lancedb.connect(db_path)
        self.table = self.db.open_table("transcripts")

        # Try to create FTS index if it doesn't exist
        self._ensure_fts_index()

        print("Ready!\n")

    def _ensure_fts_index(self):
        """Create FTS index if not exists."""
        try:
            self.table.create_fts_index("text", replace=False)
            print("FTS index ready")
        except Exception:
            pass  # Index likely already exists

    def _embed_query(self, text: str, is_document: bool = False) -> np.ndarray:
        """Embed text with appropriate prefix."""
        if self.profile == "homeserver":
            return self.llm.embed(text, dimensions=self.embedding_dim)

        import torch
        prefix = "search_document: " if is_document else "search_query: "
        prefixed = prefix + text

        with torch.no_grad():
            emb = self.embedder.encode(
                [prefixed],
                convert_to_tensor=True,
                device=self._embed_device,
                normalize_embeddings=True
            )
            if self.embedding_dim < emb.shape[1]:
                emb = emb[:, :self.embedding_dim]

        return emb.cpu().numpy()[0]

    def _vector_search(self, embedding: np.ndarray, limit: int = 150) -> list[dict]:
        """Pure vector search."""
        results = self.table.search(embedding).limit(limit).to_pandas()
        return results.to_dict('records')

    def _hybrid_search(self, query: str, limit: int = 150) -> list[dict]:
        """Hybrid search combining vector + full-text."""
        query_vec = self._embed_query(query)

        try:
            # Try hybrid search
            results = (
                self.table
                .search(query_vec, query_type="hybrid")
                .limit(limit)
                .to_pandas()
            )
        except Exception:
            # Fall back to pure vector search
            results = self.table.search(query_vec).limit(limit).to_pandas()

        return results.to_dict('records')

    def _rerank(self, query: str, documents: list[dict], top_k: int = 50) -> list[dict]:
        """Rerank documents with BGE cross-encoder."""
        if not self.reranker or not documents:
            return documents[:top_k]

        pairs = [[query, doc["text"]] for doc in documents]
        scores = self.reranker.compute_score(pairs, normalize=True)

        # Handle both single score and list of scores
        if isinstance(scores, (int, float)):
            scores = [scores]

        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        ranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:top_k]

    def search(self, query: str, verbose: bool = False) -> tuple[str, list[dict]]:
        """
        Full RAG pipeline with all improvements.

        Returns:
            Tuple of (synthesis_text, final_chunks)
        """
        # Stage 1: Intent classification
        if verbose:
            print("Classifying intent...")
        intent = classify_intent(query, self.llm)
        if verbose:
            print(f"  Intent: {intent}")

        # Stage 2: Multi-pass retrieval
        if verbose:
            print("Retrieving candidates...")

        # Pass 1: Direct hybrid search
        direct_results = self._hybrid_search(query, limit=150)

        # Pass 2: Intent-expanded query
        expansion = INTENT_EXPANSIONS.get(intent, "")
        expanded_query = f"{query} {expansion}"
        expanded_results = self._hybrid_search(expanded_query, limit=150)

        # Pass 3: HyDE
        if verbose:
            print("Generating hypothetical document...")
        hyde_doc = generate_hypothetical_document(query, intent, self.llm)
        hyde_embedding = self._embed_query(hyde_doc, is_document=True)
        hyde_results = self._vector_search(hyde_embedding, limit=150)

        # Stage 3: RRF fusion
        if verbose:
            print("Fusing results...")
        merged = reciprocal_rank_fusion([direct_results, expanded_results, hyde_results])
        if verbose:
            print(f"  Merged {len(merged)} unique chunks")

        # Stage 4: Cross-encoder reranking
        if self.use_reranker and verbose:
            print("Reranking with BGE...")

        # Augment query with intent for better discrimination
        intent_context = {
            "ESCAPE": "how to escape from and get out of",
            "EXECUTE": "how to apply, finish, and submit with",
            "DEFENSE": "how to prevent, stop, and defend against",
            "COUNTER": "how to counter and reverse into attack from"
        }
        augmented_query = f"{intent_context.get(intent, '')} {query}"
        reranked = self._rerank(augmented_query, merged[:150], top_k=50)

        # Stage 5: Timestamp deduplication
        if verbose:
            print("Deduplicating...")
        deduped = dedupe_by_timestamp(reranked, window_seconds=45.0, score_key="rerank_score")

        # Stage 6: Instructor diversity
        diverse = enforce_instructor_diversity(deduped, max_per_instructor=4, score_key="rerank_score")

        # Stage 7: MMR selection for semantic diversity
        if verbose:
            print("Applying MMR diversity...")
        query_embedding = self._embed_query(query)

        # Get embeddings for MMR candidates
        candidate_embeddings = []
        for doc in diverse[:30]:
            emb = self._embed_query(doc["text"], is_document=True)
            candidate_embeddings.append(emb)

        if candidate_embeddings:
            candidate_embeddings = np.array(candidate_embeddings)
            mmr_selected = mmr_select(
                query_embedding=query_embedding,
                candidates=diverse[:30],
                embeddings=candidate_embeddings,
                k=self.top_k + 3,  # Get a few extra for LLM filtering
                lambda_mult=0.6  # Slightly favor relevance over diversity
            )
        else:
            mmr_selected = diverse[:self.top_k + 3]

        # Stage 8: LLM relevance scoring
        if verbose:
            print("Scoring relevance...")
        scored = score_relevance(
            query=query,
            intent=intent,
            chunks=mmr_selected,
            llm=self.llm,
            min_score=6,  # Stricter threshold to reduce wrong-intent results
        )

        final_chunks = scored[:self.top_k]
        if verbose:
            print(f"  {len(final_chunks)} chunks passed relevance filter")

        # Stage 9: Synthesis
        if verbose:
            print("Synthesizing answer...")
        synthesis = synthesize_answer(
            query=query,
            intent=intent,
            chunks=final_chunks,
            llm=self.llm,
        )

        return synthesis, final_chunks

    def interactive(self):
        """Run interactive search loop."""
        print("\n" + "=" * 60)
        print("BJJ Transcript Search v2")
        print("=" * 60)
        print(f"Model: {self.llm_model} | Reranker: {'BGE' if self.use_reranker else 'disabled'}")
        print("Type your question and press Enter. Type 'quit' to exit.\n")

        while True:
            try:
                query = input("\nSearch: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if query.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            if len(query) < 3:
                print("Query too short, please enter at least 3 characters.")
                continue

            print("\nProcessing...\n")

            try:
                synthesis, chunks = self.search(query, verbose=True)
                print("\n" + "=" * 60)
                print(synthesis)
                print("\n" + "-" * 60)
                print(f"Final chunks: {len(chunks)} | "
                      f"Videos: {len(set(c.get('video_file', '') for c in chunks))}")
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="BJJ Transcript Search v2")
    parser.add_argument(
        "--db",
        default=None,
        help="Path to LanceDB database (default: profile-dependent)"
    )
    parser.add_argument(
        "--model",
        default="llama3.1:8b",
        help="Ollama model for synthesis"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Number of final chunks"
    )
    parser.add_argument(
        "--query",
        help="Single query (non-interactive mode)"
    )
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="Disable BGE reranker"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show pipeline stages"
    )
    parser.add_argument(
        "--profile",
        choices=["laptop", "homeserver"],
        default=os.environ.get("PROFILE", "laptop"),
        help="Hardware profile (default: from PROFILE env var, fallback: laptop)"
    )

    args = parser.parse_args()

    if args.db is None:
        args.db = "./data/bjj_search_db_openai" if args.profile == "homeserver" else "./data/bjj_search_db"

    rag = BJJSearchRAGv2(
        db_path=args.db,
        llm_model=args.model,
        top_k=args.top_k,
        use_reranker=not args.no_reranker,
        profile=args.profile,
    )

    if args.query:
        synthesis, _ = rag.search(args.query, verbose=args.verbose)
        print(synthesis)
    else:
        rag.interactive()


if __name__ == "__main__":
    main()
