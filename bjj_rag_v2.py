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
    max_per_instructor: int = 8,
    score_key: str = "rerank_score"
) -> list[dict]:
    """
    Ensure no single instructor dominates results.
    Max 8 allows broad coverage from each instructor for source-first retrieval.
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

RELEVANCE_SCORING_PROMPT = """You are evaluating search results for a BJJ technique query.

Query: {query}
User Intent: {intent} (the user wants to {intent_description})

SCORING RULES (0-10):
- 8-10: Directly addresses the technique AND matches the user's intent
- 6-7: Related technique or closely related intent (e.g., escape vs defense)
- 4-5: Same position family but different focus
- 1-3: WRONG INTENT (e.g., content about attacking when user wants to escape)
- 0: Completely unrelated

Pay attention to INTENT: content about executing a triangle is NOT relevant if the user wants to escape a triangle.

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
    min_score: int = 4,
    batch_size: int = 15
) -> list[dict]:
    """
    Use LLM to score each chunk's relevance in batches, filter low scores.
    Filters out wrong-intent results (min_score=4).
    """
    if not chunks:
        return []

    # Score in batches to handle large candidate pools
    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start:batch_start + batch_size]

        numbered_sources = "\n\n".join([
            f"Source {i+1}:\n{chunk['text'][:400]}..."
            for i, chunk in enumerate(batch)
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
            max_tokens=1500,
            json_output=True,
        )

        try:
            parsed = json.loads(response)

            # Handle both bare list and wrapped {"results": [...]} formats
            if isinstance(parsed, list):
                scores = parsed
            elif isinstance(parsed, dict):
                scores = parsed.get("results", parsed.get("scores", parsed.get("data", [])))
                if not isinstance(scores, list):
                    scores = list(parsed.values())[0] if parsed else []
            else:
                scores = []

            # Attach scores to batch chunks
            for score_entry in scores:
                idx = score_entry["source_num"] - 1
                if 0 <= idx < len(batch):
                    batch[idx]["llm_relevance_score"] = score_entry["score"]
                    batch[idx]["relevance_reason"] = score_entry.get("reason", "")

        except (json.JSONDecodeError, KeyError, TypeError):
            pass  # Batch failed, chunks keep default score of 0

    # Filter by minimum score
    filtered = [c for c in chunks if c.get("llm_relevance_score", 0) >= min_score]

    # Sort by LLM score
    return sorted(filtered, key=lambda x: x.get("llm_relevance_score", 0), reverse=True)


# =============================================================================
# Brief Summary Synthesis
# =============================================================================

SYNTHESIS_PROMPT = """You are summarizing BJJ video search results. Be EXTREMELY brief.

QUERY: {query}
INTENT: {intent}

INSTRUCTORS AND THEIR TOPICS:
{instructor_summary}

Write 2-3 sentences MAX summarizing what the sources cover. Rules:
- ONLY mention themes/techniques that appear across MULTIPLE sources
- Name specific instructors only if they have a distinctive approach
- Do NOT teach the technique — just describe what coverage exists
- Do NOT list every source — summarize the overall picture

Brief summary:"""


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS or MM:SS format."""
    h, remainder = divmod(int(seconds), 3600)
    m, s = divmod(remainder, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def extract_instructor(video_path: str) -> str:
    """Extract instructor name from video file path (first directory component)."""
    parts = Path(video_path).parts
    return parts[0] if parts else 'Unknown'


def group_results_by_instructor(chunks: list[dict]) -> dict[str, list[dict]]:
    """
    Group chunks by instructor, formatting each as a source result.

    Returns dict like:
    {
        'John Danaher': [
            {'video_title': '...', 'timestamp': '2:05', 'end_time': '3:30',
             'text': '...first 200 chars...', 'relevance_score': 8},
            ...
        ],
    }
    """
    grouped = defaultdict(list)

    for chunk in chunks:
        video_path = chunk.get('video_file', '')
        instructor = chunk.get('instructor') or extract_instructor(video_path)

        # Use video_title field if available, otherwise parse from path
        video_title = chunk.get('video_title')
        if not video_title:
            path = Path(video_path)
            video_title = path.stem.replace('.json', '').replace('.opus', '')

        start_time = chunk.get('start_time', 0)
        end_time = chunk.get('end_time', start_time)

        grouped[instructor].append({
            'video_title': video_title,
            'timestamp': format_timestamp(start_time),
            'end_time': format_timestamp(end_time),
            'text': chunk.get('text', '')[:500],
            'relevance_score': chunk.get('llm_relevance_score', 0),
        })

    # Sort each instructor's results by relevance score descending
    for instructor in grouped:
        grouped[instructor].sort(key=lambda x: x['relevance_score'], reverse=True)

    # Sort instructors by their best result's score
    sorted_grouped = dict(sorted(
        grouped.items(),
        key=lambda item: max(r['relevance_score'] for r in item[1]) if item[1] else 0,
        reverse=True,
    ))

    return sorted_grouped


def synthesize_brief_summary(
    query: str,
    intent: str,
    grouped_results: dict[str, list[dict]],
    llm: LLMClient = None,
) -> str:
    """Generate a 2-3 sentence brief summary of what the sources cover."""
    if not grouped_results:
        return "No relevant content found in the transcript database for this query."

    # Build a compact instructor summary for the prompt
    lines = []
    for instructor, results in grouped_results.items():
        snippets = [r['text'][:100] for r in results[:3]]
        lines.append(f"- {instructor} ({len(results)} clips): {' | '.join(snippets)}")

    instructor_summary = "\n".join(lines)

    prompt = SYNTHESIS_PROMPT.format(
        query=query,
        intent=intent,
        instructor_summary=instructor_summary,
    )

    return llm.generate(
        prompt=prompt,
        temperature=0.3,
        max_tokens=200,
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
    Source-first RAG system for BJJ instructional video transcripts.

    Pipeline:
    1. Intent classification
    2. HyDE generation
    3. Multi-pass retrieval (direct + expanded + HyDE)
    4. RRF fusion
    5. Cross-encoder reranking (BGE)
    6. Timestamp deduplication
    7. MMR diversity selection
    8. LLM relevance scoring
    9. Group by instructor
    10. Brief summary synthesis
    """

    def __init__(
        self,
        db_path: str,
        llm_model: str = "llama3.1:8b",
        embedding_dim: int = 512,
        top_k: int = 25,
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

    def search(self, query: str, verbose: bool = False) -> tuple[str, dict[str, list[dict]]]:
        """
        Full RAG pipeline optimized for source-first retrieval.

        Returns:
            Tuple of (brief_summary, grouped_results) where grouped_results
            is a dict keyed by instructor name.
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

        # Stage 6: Instructor diversity (max 8 per instructor for broader coverage)
        diverse = enforce_instructor_diversity(deduped, max_per_instructor=8, score_key="rerank_score")

        # Stage 7: MMR selection for semantic diversity (laptop only — too many API calls for homeserver)
        if self.profile == "laptop":
            if verbose:
                print("Applying MMR diversity...")
            query_embedding = self._embed_query(query)

            mmr_pool = diverse[:40]
            candidate_embeddings = []
            for doc in mmr_pool:
                emb = self._embed_query(doc["text"], is_document=True)
                candidate_embeddings.append(emb)

            if candidate_embeddings:
                candidate_embeddings = np.array(candidate_embeddings)
                mmr_selected = mmr_select(
                    query_embedding=query_embedding,
                    candidates=mmr_pool,
                    embeddings=candidate_embeddings,
                    k=self.top_k * 3,
                    lambda_mult=0.6
                )
            else:
                mmr_selected = diverse[:self.top_k * 3]
        else:
            # Homeserver: skip MMR, pass diverse pool directly to scoring
            if verbose:
                print("Skipping MMR (API mode)...")
            mmr_selected = diverse[:self.top_k * 3]

        # Stage 8: LLM relevance scoring
        if verbose:
            print("Scoring relevance...")
        scored = score_relevance(
            query=query,
            intent=intent,
            chunks=mmr_selected,
            llm=self.llm,
            min_score=4,
        )

        final_chunks = scored[:self.top_k]
        if verbose:
            print(f"  {len(final_chunks)} chunks passed relevance filter")

        # Stage 9: Group by instructor
        grouped_results = group_results_by_instructor(final_chunks)

        # Stage 10: Brief summary synthesis
        if verbose:
            print("Generating brief summary...")
        summary = synthesize_brief_summary(
            query=query,
            intent=intent,
            grouped_results=grouped_results,
            llm=self.llm,
        )

        return summary, grouped_results

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
                summary, grouped = self.search(query, verbose=True)
                print("\n" + "=" * 60)
                print(summary)
                print("\n" + "-" * 60)
                total_results = sum(len(v) for v in grouped.values())
                print(f"Results: {total_results} clips from {len(grouped)} instructors\n")
                for instructor, results in grouped.items():
                    print(f"\n  {instructor} ({len(results)} clips):")
                    for r in results:
                        print(f"    [{r['relevance_score']}/10] {r['video_title']} @ {r['timestamp']}-{r['end_time']}")
                        print(f"           {r['text'][:100]}...")
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
        default=25,
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
        args.db = "./data/bjj_search_db_v2" if args.profile == "homeserver" else "./data/bjj_search_db"

    rag = BJJSearchRAGv2(
        db_path=args.db,
        llm_model=args.model,
        top_k=args.top_k,
        use_reranker=not args.no_reranker,
        profile=args.profile,
    )

    if args.query:
        summary, grouped = rag.search(args.query, verbose=args.verbose)
        print(summary)
        for instructor, results in grouped.items():
            print(f"\n{instructor} ({len(results)} clips):")
            for r in results:
                print(f"  [{r['relevance_score']}/10] {r['video_title']} @ {r['timestamp']}-{r['end_time']}")
    else:
        rag.interactive()


if __name__ == "__main__":
    main()
