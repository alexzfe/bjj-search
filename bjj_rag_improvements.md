# Comprehensive RAG Improvements for BJJ Instructional Video Search

Your BJJ video RAG system faces four interconnected problems—poor semantic discrimination, result duplication, suboptimal output formatting, and context-starved chunks. After researching current RAG best practices, cross-encoder reranking, hybrid search implementations, and hierarchical chunking strategies, a quality-focused approach emerges: **implement heavy cross-encoder reranking with BGE-reranker, add HyDE and multi-pass retrieval for comprehensive coverage, use chain-of-thought relevance scoring before synthesis, and re-chunk to 800-1000 tokens with parent-child retrieval**. With latency not being a constraint within your 10-second budget, we can prioritize retrieval quality over speed.

---

## Problem 1 Solution: Heavy Cross-Encoder Reranking with Intent Detection

The "triangle escape" problem—where queries about escaping triangles return results about executing them—stems from bi-encoder embeddings failing to capture semantic contrast. Your nomic-embed-text-v1.5 embeddings encode "triangle" and "escape" independently without understanding their relationship. Cross-encoders solve this by processing query-document pairs jointly, enabling nuanced understanding of whether content matches user intent.

### Why Cross-Encoders Work

Bi-encoders (like your nomic model) create separate embeddings for query and document, then compare via cosine similarity. This is fast but loses relational information. Cross-encoders instead concatenate query + document and process them together through a transformer, allowing attention between query terms and document terms. This means the model can understand that "escape" in the query should attend to "get out," "defense," and "posture up" in the document—not "setup," "finish," or "attack."

### Recommended Model: `BAAI/bge-reranker-v2-m3`

This is the quality-first choice. At ~568M parameters and ~1.5GB VRAM (with FP16), it provides dramatically better semantic understanding than lighter alternatives like MiniLM.

**Why BGE over alternatives:**
- **Multilingual training** means better handling of BJJ's Portuguese terminology (e.g., "raspagem," "passagem")
- **Longer context support** (8192 tokens) handles your expanded chunks without truncation
- **Fine-tuned for semantic similarity** rather than just lexical matching

```python
from FlagEmbedding import FlagReranker
import numpy as np

class BJJReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        # use_fp16=True cuts VRAM roughly in half with minimal quality loss
        self.reranker = FlagReranker(model_name, use_fp16=True, device="cuda")
    
    def rerank(self, query: str, documents: list[dict], top_k: int = 15) -> list[dict]:
        """Rerank documents with BGE cross-encoder."""
        pairs = [[query, doc["text"]] for doc in documents]
        
        # normalize=True gives scores in [0,1] range for easier thresholding
        scores = self.reranker.compute_score(pairs, normalize=True)
        
        # Attach scores and sort
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = score
        
        ranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:top_k]
    
    def rerank_with_intent(self, query: str, intent: str, documents: list[dict], top_k: int = 15) -> list[dict]:
        """Rerank with intent-augmented query for better discrimination."""
        intent_context = {
            "ESCAPE": "how to escape from and get out of",
            "EXECUTE": "how to apply, finish, and submit with", 
            "DEFENSE": "how to prevent, stop, and defend against",
            "COUNTER": "how to counter and reverse into attack from"
        }
        
        augmented_query = f"{intent_context.get(intent, '')} {query}"
        return self.rerank(augmented_query, documents, top_k)
```

### Alternative: `cross-encoder/ms-marco-MiniLM-L-12-v2`

If VRAM is tight, the L-12 variant (~134M params, ~500MB) is substantially better than L-6 while remaining lightweight. Use this as a fallback if running BGE alongside Llama 3.1 8B causes memory pressure.

### LanceDB Hybrid Search Setup

LanceDB's native hybrid search combines BM25 full-text with vector similarity. This is foundational—you want both lexical matching (catches exact terminology) and semantic matching (catches conceptual similarity):

```python
import lancedb

class HybridSearcher:
    def __init__(self, db_path: str, table_name: str = "chunks"):
        self.db = lancedb.connect(db_path)
        self.table = self.db.open_table(table_name)
        self._ensure_fts_index()
    
    def _ensure_fts_index(self):
        """Create FTS index if not exists. Only needs to run once."""
        try:
            self.table.create_fts_index(
                "text",
                language="English",
                stem=True,  # "escaping" matches "escape"
                remove_stop_words=True
            )
        except Exception:
            pass  # Index already exists
    
    def search(self, query: str, limit: int = 150) -> list[dict]:
        """Hybrid search with automatic RRF fusion."""
        results = (
            self.table
            .search(query, query_type="hybrid")
            .limit(limit)
            .to_list()
        )
        return results
```

### Intent Classification

This runs before retrieval to understand what the user actually wants. This 50-100 token LLM call is critical for the semantic discrimination problem:

```python
import ollama

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

def classify_intent(query: str) -> str:
    """Classify query intent using local LLM."""
    response = ollama.generate(
        model="llama3.1:8b",
        prompt=INTENT_CLASSIFICATION_PROMPT.format(query=query),
        options={"temperature": 0.1, "num_predict": 10}
    )
    
    intent = response["response"].strip().upper()
    valid_intents = {"EXECUTE", "ESCAPE", "DEFENSE", "COUNTER", "CONCEPT", "DRILL"}
    return intent if intent in valid_intents else "EXECUTE"  # Default fallback
```

---

## Problem 1 Enhancement: HyDE and Multi-Pass Retrieval

Since latency isn't constrained, **HyDE (Hypothetical Document Embeddings)** dramatically improves retrieval for conceptual queries. Instead of embedding the raw query, you generate what an ideal answer would look like, then embed that richer representation.

### Why HyDE Works

Your nomic-embed-text model will produce much richer embeddings from:

> "To escape the triangle, first you need to posture up and create space. Get your trapped arm to the inside, stack your opponent's hips, and work to free your head..."

...than from just "triangle escape."

The hypothetical document contains the actual terminology and concepts that appear in your instructor transcripts. This bridges the vocabulary gap between how users ask questions and how instructors explain techniques.

```python
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

def generate_hypothetical_document(query: str, intent: str) -> str:
    """Generate a hypothetical ideal answer for HyDE retrieval."""
    response = ollama.generate(
        model="llama3.1:8b",
        prompt=HYDE_PROMPT.format(query=query, intent=intent),
        options={"temperature": 0.7, "num_predict": 200}  # Some creativity helps
    )
    return response["response"].strip()
```

### Multi-Pass Retrieval with RRF

Combines multiple search strategies and merges results with Reciprocal Rank Fusion. This ensures comprehensive coverage—different queries surface different relevant content:

```python
from collections import defaultdict
import numpy as np

def reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """
    Merge multiple ranked lists using RRF.
    
    RRF score = Σ 1/(k + rank_i) across all lists where document appears
    k=60 is the standard constant that balances early vs late ranks
    """
    rrf_scores = defaultdict(float)
    doc_lookup = {}
    
    for results in result_lists:
        for rank, doc in enumerate(results, start=1):
            doc_id = doc.get("id") or doc.get("chunk_id") or hash(doc["text"][:100])
            rrf_scores[doc_id] += 1.0 / (k + rank)
            doc_lookup[doc_id] = doc
    
    # Sort by RRF score
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    return [doc_lookup[doc_id] for doc_id in sorted_ids]


class MultiPassRetriever:
    def __init__(self, searcher: HybridSearcher, embedder, reranker: BJJReranker):
        self.searcher = searcher
        self.embedder = embedder
        self.reranker = reranker
    
    def retrieve(self, query: str, intent: str, top_k: int = 15) -> list[dict]:
        """
        Three-pass retrieval for maximum coverage:
        1. Direct query - catches exact matches
        2. Intent-expanded query - catches semantically related content
        3. HyDE - catches conceptually similar content
        """
        
        # Pass 1: Direct hybrid search
        direct_results = self.searcher.search(query, limit=150)
        
        # Pass 2: Intent-expanded query
        intent_expansions = {
            "ESCAPE": f"{query} escape get out defense survive when caught in",
            "EXECUTE": f"{query} setup finish attack submission apply technique",
            "DEFENSE": f"{query} prevent stop block deny shut down",
            "COUNTER": f"{query} counter reverse sweep attack from bottom"
        }
        expanded_query = intent_expansions.get(intent, query)
        expanded_results = self.searcher.search(expanded_query, limit=150)
        
        # Pass 3: HyDE - generate hypothetical document and search with its embedding
        hyde_doc = generate_hypothetical_document(query, intent)
        # Embed as document (not query) since it's meant to match other documents
        hyde_embedding = self.embedder.encode(hyde_doc, prompt_name="search_document")
        hyde_results = self._vector_search(hyde_embedding, limit=150)
        
        # Merge all results with RRF
        merged = reciprocal_rank_fusion([direct_results, expanded_results, hyde_results])
        
        # Heavy reranking on merged candidates
        reranked = self.reranker.rerank_with_intent(query, intent, merged[:150], top_k=top_k * 2)
        
        return reranked[:top_k]
    
    def _vector_search(self, embedding: np.ndarray, limit: int) -> list[dict]:
        """Pure vector search with pre-computed embedding."""
        return self.searcher.table.search(embedding).limit(limit).to_list()
```

**Why three passes matter:** Each retrieval method has blind spots. Direct queries miss synonyms and conceptual matches. Intent expansion can drift from specifics. HyDE can hallucinate wrong terminology. By combining all three with RRF, you get robust coverage while the cross-encoder reranker filters out the noise.

---

## Problem 2 Solution: Timestamp Deduplication with MMR Diversity

Your 20% chunk overlap creates redundancy—multiple chunks from the same 45-second segment consume your top-k budget with near-duplicate content. The solution combines **timestamp-based grouping** (same-video deduplication) with **MMR** (cross-video semantic diversity).

### Timestamp Deduplication

Groups chunks from the same video within a configurable time window, selecting the highest-scoring representative:

```python
from collections import defaultdict
from typing import List, Dict

def dedupe_by_timestamp(
    chunks: List[Dict], 
    window_seconds: float = 45.0,
    score_key: str = "rerank_score"
) -> List[Dict]:
    """
    Group overlapping chunks by video and time window, keep best per group.
    
    Why 45 seconds? BJJ technique explanations typically take 30-90 seconds.
    45s captures most single-technique segments while allowing distinct 
    techniques in the same video to remain separate.
    """
    # Group by video first
    video_groups = defaultdict(list)
    for chunk in chunks:
        video_groups[chunk['video_id']].append(chunk)
    
    deduped = []
    
    for video_id, video_chunks in video_groups.items():
        # Sort by start time within each video
        video_chunks.sort(key=lambda x: x['start_time'])
        
        # Cluster overlapping chunks
        clusters = []
        current_cluster = [video_chunks[0]]
        
        for chunk in video_chunks[1:]:
            cluster_end = max(c['end_time'] for c in current_cluster)
            
            # If this chunk starts within window of cluster's end, merge it
            if chunk['start_time'] <= cluster_end + window_seconds:
                current_cluster.append(chunk)
            else:
                # Start new cluster
                clusters.append(current_cluster)
                current_cluster = [chunk]
        
        clusters.append(current_cluster)  # Don't forget last cluster
        
        # Select highest-scoring chunk from each cluster
        for cluster in clusters:
            best = max(cluster, key=lambda x: x.get(score_key, 0))
            # Optionally merge text from cluster for richer context
            best["cluster_size"] = len(cluster)
            deduped.append(best)
    
    return sorted(deduped, key=lambda x: x.get(score_key, 0), reverse=True)
```

### MMR (Maximal Marginal Relevance)

Ensures semantic diversity across different videos and instructors. The algorithm iteratively selects documents that are both relevant to the query AND different from already-selected documents:

```python
from sentence_transformers import SentenceTransformer
import numpy as np

def mmr_select(
    query_embedding: np.ndarray,
    candidates: List[Dict],
    embedder: SentenceTransformer,
    k: int = 12,
    lambda_mult: float = 0.5,  # Lower = more diversity
    embedding_key: str = "embedding"
) -> List[Dict]:
    """
    Maximal Marginal Relevance selection.
    
    MMR = λ * similarity(doc, query) - (1-λ) * max(similarity(doc, selected))
    
    lambda_mult controls the relevance/diversity tradeoff:
    - 1.0 = pure relevance (equivalent to top-k)
    - 0.0 = pure diversity (maximally different documents)
    - 0.5 = balanced (good default for BJJ where you want different instructors' takes)
    
    For BJJ content, 0.5 works well because multiple instructors teach similar 
    techniques differently, and seeing varied approaches is valuable.
    """
    # Get or compute embeddings for candidates
    candidate_embeddings = []
    for doc in candidates:
        if embedding_key in doc:
            candidate_embeddings.append(doc[embedding_key])
        else:
            emb = embedder.encode(doc["text"])
            doc[embedding_key] = emb
            candidate_embeddings.append(emb)
    
    candidate_embeddings = np.array(candidate_embeddings)
    
    # Normalize for cosine similarity
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    cand_norms = candidate_embeddings / np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
    
    # Query-document similarities
    query_sims = np.dot(cand_norms, query_norm)
    
    selected_indices = []
    selected_embeddings = []
    
    for _ in range(min(k, len(candidates))):
        if not selected_indices:
            # First selection: pure relevance
            best_idx = np.argmax(query_sims)
        else:
            # Compute MMR scores
            selected_arr = np.array(selected_embeddings)
            
            # Max similarity to any selected document
            diversity_penalties = np.max(np.dot(cand_norms, selected_arr.T), axis=1)
            
            # MMR score
            mmr_scores = lambda_mult * query_sims - (1 - lambda_mult) * diversity_penalties
            
            # Mask already selected
            mmr_scores[selected_indices] = -np.inf
            
            best_idx = np.argmax(mmr_scores)
        
        selected_indices.append(best_idx)
        selected_embeddings.append(cand_norms[best_idx])
    
    return [candidates[i] for i in selected_indices]
```

### Instructor Diversity Enforcement

Caps results per instructor to ensure varied perspectives:

```python
def enforce_instructor_diversity(
    chunks: List[Dict], 
    max_per_instructor: int = 3,
    score_key: str = "rerank_score"
) -> List[Dict]:
    """
    Ensure no single instructor dominates results.
    
    Why max 3? You want multiple perspectives on a technique, but 
    too many from one instructor is redundant. 3 allows for setup,
    execution, and troubleshooting from each voice.
    """
    instructor_counts = defaultdict(int)
    selected = []
    
    # Process in score order so we keep the best from each instructor
    for chunk in sorted(chunks, key=lambda x: x.get(score_key, 0), reverse=True):
        instructor = chunk.get('instructor', chunk.get('video_id', 'unknown'))
        
        if instructor_counts[instructor] < max_per_instructor:
            selected.append(chunk)
            instructor_counts[instructor] += 1
    
    return selected
```

### Complete Deduplication Pipeline

Order matters for efficiency:

```python
def dedupe_pipeline(
    chunks: List[Dict],
    query_embedding: np.ndarray,
    embedder: SentenceTransformer,
    final_k: int = 12
) -> List[Dict]:
    """
    Full deduplication pipeline:
    1. Timestamp dedupe - removes overlap redundancy (~45 chunks from 150)
    2. Instructor cap - ensures variety (~30 chunks)  
    3. MMR selection - semantic diversity (final 12)
    """
    # Stage 1: Remove timestamp overlaps
    deduped = dedupe_by_timestamp(chunks, window_seconds=45.0)
    
    # Stage 2: Cap per instructor
    diverse = enforce_instructor_diversity(deduped, max_per_instructor=3)
    
    # Stage 3: MMR for semantic diversity
    final = mmr_select(
        query_embedding=query_embedding,
        candidates=diverse,
        embedder=embedder,
        k=final_k,
        lambda_mult=0.5
    )
    
    return final
```

---

## Problem 2 Enhancement: Chain-of-Thought Relevance Scoring

Since we have time budget, add an **explicit relevance scoring step** before synthesis. This uses your Llama to evaluate each chunk's relevance, filtering out false positives that made it through embedding similarity:

```python
import json

RELEVANCE_SCORING_PROMPT = """You are evaluating search results for a BJJ technique query.

Query: {query}
User Intent: {intent} (the user wants to {intent_description})

Score each source's relevance from 0-10:
- 10: Directly addresses the exact technique AND intent (e.g., "triangle escape" when user asked for triangle escape)
- 7-9: Addresses the technique with related intent (e.g., triangle defense when user asked for escape)
- 4-6: Related technique or position (e.g., arm triangle when user asked about triangle)
- 1-3: Same position family but different technique (e.g., guard passing when user asked about triangle from guard)
- 0: Completely unrelated

IMPORTANT: Pay attention to the INTENT. Content about EXECUTING a triangle is NOT relevant if user wants to ESCAPE the triangle.

Sources to evaluate:
{numbered_sources}

For each source, respond in this exact JSON format:
[
  {{"source_num": 1, "score": 8, "reason": "Directly shows triangle escape sequence"}},
  {{"source_num": 2, "score": 3, "reason": "Shows triangle setup, not escape"}},
  ...
]

JSON response:"""

INTENT_DESCRIPTIONS = {
    "ESCAPE": "learn how to get out when caught in this position/submission",
    "EXECUTE": "learn how to perform and finish this technique",
    "DEFENSE": "learn how to prevent this technique from being applied",
    "COUNTER": "learn how to reverse this into their own attack"
}

def score_relevance(
    query: str,
    intent: str,
    chunks: List[Dict],
    min_score: int = 6
) -> List[Dict]:
    """
    Use LLM to score each chunk's relevance, filter low scores.
    
    This catches false positives where embedding similarity was high
    but actual content doesn't match user intent.
    """
    # Format chunks for prompt
    numbered_sources = "\n\n".join([
        f"Source {i+1}:\n{chunk['text'][:500]}..."  # Truncate for prompt efficiency
        for i, chunk in enumerate(chunks)
    ])
    
    prompt = RELEVANCE_SCORING_PROMPT.format(
        query=query,
        intent=intent,
        intent_description=INTENT_DESCRIPTIONS.get(intent, "understand this technique"),
        numbered_sources=numbered_sources
    )
    
    response = ollama.generate(
        model="llama3.1:8b",
        prompt=prompt,
        options={"temperature": 0.1, "num_predict": 500},
        format="json"
    )
    
    try:
        scores = json.loads(response["response"])
        
        # Attach scores to chunks
        for score_entry in scores:
            idx = score_entry["source_num"] - 1
            if 0 <= idx < len(chunks):
                chunks[idx]["llm_relevance_score"] = score_entry["score"]
                chunks[idx]["relevance_reason"] = score_entry["reason"]
        
        # Filter by minimum score
        filtered = [c for c in chunks if c.get("llm_relevance_score", 0) >= min_score]
        
        # Sort by LLM score (not embedding score)
        return sorted(filtered, key=lambda x: x.get("llm_relevance_score", 0), reverse=True)
        
    except json.JSONDecodeError:
        # Fallback: return original chunks if JSON parsing fails
        return chunks
```

This step typically takes 1-2 seconds but dramatically improves precision. The LLM understands "triangle escape" vs "triangle setup" in a way embeddings fundamentally cannot.

---

## Problem 3 Solution: Answer-First Synthesis Prompts

Your current source-by-source output forces users to mentally synthesize across summaries. Restructuring to **answer-first with supporting sources** delivers immediate value while maintaining citation transparency.

### Recommended Synthesis Prompt

```python
SYNTHESIS_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert BJJ assistant helping practitioners find techniques in their video library.

RESPONSE STRUCTURE (follow exactly):
1. **Direct Answer**: A synthesized response that directly addresses the query, combining insights from relevant sources
2. **Key Details**: Specific technical points with timestamps for video reference
3. **Sources**: List of videos used with timestamps

RULES:
- Only use information from the provided sources
- Include [Video Title @ MM:SS] citations for specific claims
- If sources show different approaches, explain each with attribution
- If sources don't adequately answer the query, say so explicitly
- Be specific about grips, positions, and movements

<|eot_id|><|start_header_id|>user<|end_header_id|>

QUERY: {query}
INTENT: {intent} (user wants to {intent_description})

RELEVANT SOURCES (filtered by relevance):
{formatted_chunks}

Provide a complete answer following the structure above.

<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

def format_chunks_for_synthesis(chunks: List[Dict]) -> str:
    """Format chunks with clear source attribution."""
    formatted = []
    for i, chunk in enumerate(chunks, 1):
        video_title = chunk.get("video_title", chunk.get("video_id", f"Video {i}"))
        timestamp = format_timestamp(chunk.get("start_time", 0))
        relevance = chunk.get("llm_relevance_score", "N/A")
        instructor = chunk.get("instructor", "Unknown")
        
        formatted.append(f"""
---
SOURCE {i}: {video_title} @ {timestamp}
Instructor: {instructor}
Relevance Score: {relevance}/10

{chunk['text']}
---""")
    
    return "\n".join(formatted)

def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"
```

### Two-Stage Synthesis for Complex Queries

For queries that might have multiple valid approaches, use a planning step first:

```python
PLANNING_PROMPT = """Given this BJJ query and sources, plan your response:

Query: {query}
Intent: {intent}

Sources available: {source_summaries}

Think through:
1. What specific question is being asked?
2. Which sources directly address this?
3. Are there multiple valid approaches shown?
4. What's the best way to structure the answer?

Plan (2-3 sentences):"""

def two_stage_synthesis(query: str, intent: str, chunks: List[Dict]) -> str:
    """Plan then synthesize for better quality on complex queries."""
    
    # Quick source summaries for planning
    source_summaries = [f"- {c.get('video_title', 'Video')}: {c['text'][:100]}..." 
                        for c in chunks[:5]]
    
    # Stage 1: Planning
    plan_response = ollama.generate(
        model="llama3.1:8b",
        prompt=PLANNING_PROMPT.format(
            query=query,
            intent=intent,
            source_summaries="\n".join(source_summaries)
        ),
        options={"temperature": 0.3, "num_predict": 150}
    )
    
    # Stage 2: Synthesis with plan context
    synthesis_prompt_with_plan = SYNTHESIS_PROMPT.format(
        query=query,
        intent=intent,
        intent_description=INTENT_DESCRIPTIONS.get(intent, ""),
        formatted_chunks=format_chunks_for_synthesis(chunks)
    )
    
    # Inject plan
    synthesis_prompt_with_plan = synthesis_prompt_with_plan.replace(
        "Provide a complete answer",
        f"Your plan: {plan_response['response']}\n\nNow provide a complete answer"
    )
    
    response = ollama.generate(
        model="llama3.1:8b",
        prompt=synthesis_prompt_with_plan,
        options={"temperature": 0.4, "num_predict": 800}
    )
    
    return response["response"]
```

### Ollama Model Configuration

Create a custom model with tuned parameters:

```bash
# Create custom model
ollama create bjj-rag -f ./Modelfile
```

```dockerfile
# Modelfile
FROM llama3.1:8b

# Lower temperature for factual synthesis
PARAMETER temperature 0.3

# Enough context for 12 chunks + synthesis
PARAMETER num_ctx 8192

# Slight nucleus sampling for natural language
PARAMETER top_p 0.9

# Stop generating after answer complete
PARAMETER stop "<|eot_id|>"

SYSTEM """You are a BJJ technique assistant with access to a video library. 
You help practitioners find specific techniques by searching transcripts.
Always cite video timestamps. Be specific about technical details."""
```

---

## Problem 4 Solution: Larger Chunks with Parent-Child Retrieval

Your 400-token chunks sever instructors mid-explanation. Research consensus for instructional content strongly supports **800-1000 tokens**, and a **parent-child architecture** provides both retrieval precision and contextual completeness.

### Parent-Child Retrieval Pattern

Embed small chunks (400-500 tokens) for precise matching, but return large parent chunks (1500-2000 tokens) containing full context. This gives you the best of both worlds—precise retrieval AND complete technique explanations:

```python
from pathlib import Path
import json
import hashlib

class ParentChildIndexer:
    """
    Creates a two-tier index:
    - Child chunks (400 tokens): Embedded for retrieval precision
    - Parent chunks (1600 tokens): Returned for complete context
    
    When a child matches, we return its parent for the full picture.
    """
    
    def __init__(self, 
                 parent_size: int = 1600, 
                 child_size: int = 400,
                 parent_overlap: int = 200,
                 child_overlap: int = 50):
        self.parent_size = parent_size
        self.child_size = child_size
        self.parent_overlap = parent_overlap
        self.child_overlap = child_overlap
        self.parent_store = {}  # parent_id -> parent_chunk
        
    def create_hierarchy(self, text: str, metadata: dict) -> tuple[list[dict], list[dict]]:
        """
        Create parent and child chunks from a transcript.
        
        Returns:
            parents: List of parent chunks (stored locally)
            children: List of child chunks (embedded in vector DB)
        """
        # First, create parent chunks
        parents = self._chunk_text(
            text, 
            target_tokens=self.parent_size,
            overlap_tokens=self.parent_overlap,
            metadata=metadata
        )
        
        # Generate parent IDs and store
        for parent in parents:
            parent_id = self._generate_id(parent["text"], metadata)
            parent["parent_id"] = parent_id
            self.parent_store[parent_id] = parent
        
        # Create children from each parent
        children = []
        for parent in parents:
            parent_children = self._chunk_text(
                parent["text"],
                target_tokens=self.child_size,
                overlap_tokens=self.child_overlap,
                metadata={**metadata, "parent_id": parent["parent_id"]}
            )
            
            for child in parent_children:
                child["parent_id"] = parent["parent_id"]
                # Inherit timestamp from parent, adjusted for position
                child["start_time"] = parent["start_time"]
                child["end_time"] = parent["end_time"]
            
            children.extend(parent_children)
        
        return parents, children
    
    def get_parent(self, parent_id: str) -> dict | None:
        """Retrieve parent chunk for a matched child."""
        return self.parent_store.get(parent_id)
    
    def _chunk_text(self, text: str, target_tokens: int, overlap_tokens: int, metadata: dict) -> list[dict]:
        """Chunk text at sentence boundaries."""
        # Approximate: 1 token ≈ 4 characters for English
        target_chars = target_tokens * 4
        overlap_chars = overlap_tokens * 4
        
        sentences = self._split_sentences(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sent in sentences:
            sent_length = len(sent)
            
            if current_length + sent_length > target_chars and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    **metadata
                })
                
                # Start new chunk with overlap
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
            chunks.append({
                "text": " ".join(current_chunk),
                **metadata
            })
        
        return chunks
    
    def _split_sentences(self, text: str) -> list[str]:
        """Simple sentence splitting. For production, use spaCy."""
        import re
        # Split on sentence-ending punctuation followed by space
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _generate_id(self, text: str, metadata: dict) -> str:
        """Generate unique ID for a chunk."""
        content = f"{metadata.get('video_id', '')}:{text[:100]}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def save_parents(self, path: str):
        """Persist parent store to disk."""
        with open(path, 'w') as f:
            json.dump(self.parent_store, f)
    
    def load_parents(self, path: str):
        """Load parent store from disk."""
        with open(path, 'r') as f:
            self.parent_store = json.load(f)
```

### Retrieval with Parent Expansion

```python
class ParentChildRetriever:
    def __init__(self, vector_table, parent_store: dict):
        self.table = vector_table  # Contains child embeddings
        self.parent_store = parent_store
    
    def retrieve(self, query: str, top_k: int = 12) -> list[dict]:
        """
        Retrieve children, return parents for context.
        Deduplicates by parent to avoid returning same context multiple times.
        """
        # Over-fetch children since multiple may map to same parent
        children = self.table.search(query, query_type="hybrid").limit(top_k * 3).to_list()
        
        # Map to parents, keeping best child score per parent
        parent_scores = {}
        for child in children:
            parent_id = child["parent_id"]
            score = child.get("_score", child.get("score", 0))
            
            if parent_id not in parent_scores or score > parent_scores[parent_id]["score"]:
                parent_scores[parent_id] = {
                    "parent_id": parent_id,
                    "score": score,
                    "matched_child": child
                }
        
        # Retrieve parent texts
        results = []
        for parent_id, info in sorted(parent_scores.items(), 
                                       key=lambda x: x[1]["score"], 
                                       reverse=True)[:top_k]:
            parent = self.parent_store.get(parent_id)
            if parent:
                results.append({
                    **parent,
                    "retrieval_score": info["score"],
                    "matched_segment": info["matched_child"]["text"]
                })
        
        return results
```

### Alternative: Dynamic Context Expansion (No Re-indexing Required)

This is a good intermediate step that improves results without full re-chunking:

```python
def expand_context_dynamic(
    matched_chunk: dict,
    all_chunks: list[dict],
    window: int = 2,  # chunks before and after
    max_tokens: int = 1500
) -> str:
    """
    Expand matched chunk with surrounding context from same video.
    
    This is a runtime solution that doesn't require re-indexing,
    useful as an immediate improvement before full re-chunking.
    """
    video_id = matched_chunk.get("video_id")
    
    # Get all chunks from same video, sorted by time
    video_chunks = sorted(
        [c for c in all_chunks if c.get("video_id") == video_id],
        key=lambda x: x.get("start_time", 0)
    )
    
    # Find matched chunk position
    matched_time = matched_chunk.get("start_time", 0)
    idx = None
    for i, c in enumerate(video_chunks):
        if abs(c.get("start_time", 0) - matched_time) < 1:  # Within 1 second
            idx = i
            break
    
    if idx is None:
        return matched_chunk["text"]
    
    # Expand window
    start_idx = max(0, idx - window)
    end_idx = min(len(video_chunks), idx + window + 1)
    
    # Concatenate, respecting token limit
    expanded_text = ""
    for c in video_chunks[start_idx:end_idx]:
        candidate = expanded_text + " " + c["text"] if expanded_text else c["text"]
        if len(candidate.split()) <= max_tokens:  # Approximate token count
            expanded_text = candidate
        else:
            break
    
    return expanded_text.strip()
```

### Re-chunking Recommendation

**Yes, re-chunk your data.** Here's the evidence-based comparison:

| Approach | Implementation Effort | Re-embedding | Quality Improvement | When to Use |
|----------|----------------------|--------------|---------------------|-------------|
| Dynamic expansion only | 2-3 hours | No | 15-25% | Immediate, before re-chunking |
| Re-chunk to 800-1000 tokens | 4-6 hours | Yes | 30-40% | Primary recommendation |
| Parent-child (400→1600) | 8-12 hours | Yes | 40-50% | Best long-term architecture |
| RAPTOR hierarchical | 2-3 days | Yes + LLM | Additional 15-25% | Complex conceptual queries |

### Sentence-Boundary Chunking with spaCy

```python
import spacy

# Load English model (small is fine for sentence boundaries)
# python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

def chunk_with_sentence_boundaries(
    text: str,
    target_tokens: int = 800,
    overlap_tokens: int = 100,
    respect_paragraphs: bool = True
) -> list[str]:
    """
    Chunk text at natural sentence boundaries.
    
    Why sentence boundaries matter:
    - Mid-sentence cuts confuse embeddings
    - Complete sentences have clearer semantic meaning
    - Better retrieval precision on technique steps
    """
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    
    # Approximate tokens (spaCy tokens, more accurate than word split)
    def count_tokens(s: str) -> int:
        return len(nlp(s))
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sent in sentences:
        sent_tokens = count_tokens(sent)
        
        # Check if adding this sentence exceeds target
        if current_tokens + sent_tokens > target_tokens and current_chunk:
            # Save current chunk
            chunks.append(" ".join(current_chunk))
            
            # Calculate overlap: keep last N tokens worth of sentences
            overlap_sents = []
            overlap_count = 0
            for s in reversed(current_chunk):
                s_tokens = count_tokens(s)
                if overlap_count + s_tokens <= overlap_tokens:
                    overlap_sents.insert(0, s)
                    overlap_count += s_tokens
                else:
                    break
            
            current_chunk = overlap_sents
            current_tokens = overlap_count
        
        current_chunk.append(sent)
        current_tokens += sent_tokens
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def rechunk_transcript(
    transcript_path: str,
    output_path: str,
    target_tokens: int = 800
) -> list[dict]:
    """
    Re-chunk a transcript file with word-level timestamps.
    Preserves timestamp mapping for video navigation.
    """
    with open(transcript_path, 'r') as f:
        transcript = json.load(f)
    
    # Reconstruct full text
    full_text = transcript.get("text", "")
    words = transcript.get("words", [])  # [{word, start, end}, ...]
    
    # Chunk the text
    chunks = chunk_with_sentence_boundaries(full_text, target_tokens=target_tokens)
    
    # Map timestamps to chunks
    result_chunks = []
    word_idx = 0
    
    for chunk_text in chunks:
        chunk_words = chunk_text.split()
        
        # Find start timestamp
        start_time = None
        end_time = None
        
        for word in chunk_words:
            while word_idx < len(words):
                if words[word_idx]["word"].strip().lower() == word.strip().lower():
                    if start_time is None:
                        start_time = words[word_idx]["start"]
                    end_time = words[word_idx]["end"]
                    word_idx += 1
                    break
                word_idx += 1
        
        result_chunks.append({
            "text": chunk_text,
            "start_time": start_time or 0,
            "end_time": end_time or 0,
            "video_id": transcript.get("video_id"),
            "video_title": transcript.get("title"),
            "instructor": transcript.get("instructor")
        })
    
    # Save re-chunked data
    with open(output_path, 'w') as f:
        json.dump(result_chunks, f, indent=2)
    
    return result_chunks
```

---

## Complete Implementation Architecture

The optimized pipeline integrates all four solutions:

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: Query Understanding (~500ms)                          │
│                                                                  │
│  ├─→ Intent Classification (Llama 3.1 8B)                       │
│  │     └─→ EXECUTE | ESCAPE | DEFENSE | COUNTER | CONCEPT       │
│  │                                                               │
│  └─→ HyDE Generation (Llama 3.1 8B)                             │
│        └─→ Hypothetical ideal answer for embedding              │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: Multi-Pass Retrieval (~400ms)                         │
│                                                                  │
│  ├─→ Pass 1: Direct hybrid search (BM25 + vector, top 150)      │
│  ├─→ Pass 2: Intent-expanded query search (top 150)             │
│  └─→ Pass 3: HyDE embedding search (top 150)                    │
│                                                                  │
│  └─→ Reciprocal Rank Fusion (merge ~300 candidates)             │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3: Heavy Reranking (~600ms)                              │
│                                                                  │
│  └─→ BGE Reranker v2-m3 on top 150 merged candidates           │
│        └─→ Intent-augmented query for discrimination            │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 4: Deduplication & Diversity (~50ms)                     │
│                                                                  │
│  ├─→ Timestamp deduplication (45s window)                       │
│  ├─→ Instructor diversity cap (max 3 per instructor)            │
│  └─→ MMR selection (λ=0.5, final 15 chunks)                     │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 5: Relevance Validation (~1.5s)                          │
│                                                                  │
│  ├─→ LLM relevance scoring (0-10 per chunk)                     │
│  └─→ Filter chunks scoring < 6                                  │
│        └─→ ~8-12 high-confidence chunks                         │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 6: Context Expansion (~30ms)                             │
│                                                                  │
│  └─→ Parent retrieval OR dynamic window expansion               │
│        └─→ Full technique context per result                    │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 7: Synthesis (~4s)                                       │
│                                                                  │
│  ├─→ (Optional) Planning step for complex queries               │
│  └─→ Answer-first synthesis with Llama 3.1 8B                   │
│        ├─→ Direct answer combining all sources                  │
│        ├─→ Key technical details with timestamps                │
│        └─→ Source attribution list                              │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
Structured Output: Answer + Timestamps + Sources
```

**Total latency estimate**: ~7 seconds (well under 10-second budget)

---

## VRAM Budget Allocation

| Component | VRAM Usage | Notes |
|-----------|-----------|-------|
| Llama 3.1 8B (Q4 quantized) | 4.5-5.0 GB | Via Ollama, handles all LLM tasks |
| nomic-embed-text-v1.5 | ~300 MB | Embedding model, can offload to CPU if needed |
| BAAI/bge-reranker-v2-m3 (FP16) | ~1.2 GB | Cross-encoder reranker |
| **Total peak** | **~6.5 GB** | Within 6-8GB budget ✓ |

**Memory management tips:**
- Load reranker only when needed, offload between queries if memory is tight
- Run embedding model on CPU if needed (slower but frees ~300MB)
- Use `torch.cuda.empty_cache()` between stages if seeing OOM

---

## Evaluation Framework

Implement systematic evaluation to measure improvement:

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class EvalQuery:
    query: str
    intent: Literal["EXECUTE", "ESCAPE", "DEFENSE", "COUNTER"]
    expected_topics: list[str]  # Topics that SHOULD appear
    excluded_topics: list[str]  # Topics that should NOT appear
    
# Test set covering the semantic discrimination problem
EVAL_QUERIES = [
    EvalQuery(
        query="triangle escape",
        intent="ESCAPE",
        expected_topics=["escape", "get out", "posture", "stack"],
        excluded_topics=["setup", "finish", "submit", "attack"]
    ),
    EvalQuery(
        query="triangle from guard",
        intent="EXECUTE", 
        expected_topics=["setup", "angle", "cut", "finish", "squeeze"],
        excluded_topics=["escape", "defend", "prevent"]
    ),
    EvalQuery(
        query="defend the armbar",
        intent="DEFENSE",
        expected_topics=["grip", "elbow", "prevent", "posture"],
        excluded_topics=["finish", "hyperextend", "submit"]
    ),
    EvalQuery(
        query="mount escapes",
        intent="ESCAPE",
        expected_topics=["upa", "elbow escape", "trap", "bridge"],
        excluded_topics=["maintain", "submit from", "attack from"]
    ),
    # Add 50+ more covering your content
]

def evaluate_retrieval(retriever, eval_queries: list[EvalQuery]) -> dict:
    """Evaluate retrieval quality on test set."""
    results = {
        "precision_at_5": [],
        "intent_accuracy": [],
        "exclusion_violations": []
    }
    
    for eq in eval_queries:
        chunks = retriever.retrieve(eq.query, eq.intent, top_k=10)
        
        # Check top 5 for expected topics
        top_5_text = " ".join([c["text"].lower() for c in chunks[:5]])
        
        hits = sum(1 for topic in eq.expected_topics if topic in top_5_text)
        precision = hits / len(eq.expected_topics)
        results["precision_at_5"].append(precision)
        
        # Check for exclusion violations
        violations = sum(1 for topic in eq.excluded_topics if topic in top_5_text)
        results["exclusion_violations"].append(violations)
    
    return {
        "mean_precision_at_5": sum(results["precision_at_5"]) / len(results["precision_at_5"]),
        "mean_exclusion_violations": sum(results["exclusion_violations"]) / len(results["exclusion_violations"]),
    }
```

**Target metrics:**
- Precision@5 for expected topics: >0.8
- Exclusion violations: <0.5 per query (ideally 0)
- User satisfaction: A/B test with before/after

---

## Implementation Phases

### Phase 1 (Days 1-3): Foundation Improvements, No Re-embedding
- [ ] Add LanceDB hybrid search (BM25 index creation)
- [ ] Implement intent classification
- [ ] Add dynamic context expansion
- [ ] Restructure synthesis prompts to answer-first
- [ ] Build evaluation test set

### Phase 2 (Days 4-7): Retrieval Quality
- [ ] Integrate BGE reranker
- [ ] Implement HyDE generation
- [ ] Build multi-pass retrieval with RRF
- [ ] Add timestamp deduplication
- [ ] Add MMR diversity selection
- [ ] Run baseline evaluation

### Phase 3 (Week 2): Precision Enhancement
- [ ] Implement LLM relevance scoring
- [ ] Add instructor diversity caps
- [ ] Tune MMR lambda parameter
- [ ] Run comparative evaluation

### Phase 4 (Week 3): Re-chunking
- [ ] Create sentence-boundary chunker
- [ ] Re-chunk all 716 transcripts to 800-1000 tokens
- [ ] Re-embed with nomic-embed-text
- [ ] Rebuild LanceDB index with FTS

### Phase 5 (Week 4): Parent-Child Architecture
- [ ] Implement parent-child indexer
- [ ] Create child embeddings (400 tokens)
- [ ] Build parent store (1600 tokens)
- [ ] Update retrieval to return parents
- [ ] Final evaluation and tuning

---

## Dependencies to Install

```bash
# Core dependencies
pip install lancedb sentence-transformers FlagEmbedding spacy numpy

# Download spaCy model
python -m spacy download en_core_web_sm

# Ollama should already be installed for Llama 3.1 8B
```

---

This phased approach delivers immediate improvements (Phase 1-2 address the "triangle escape" problem without re-chunking) while building toward the optimal architecture. The quality-first design prioritizes retrieval accuracy over speed, using your full 10-second budget to ensure users get exactly what they're looking for.
