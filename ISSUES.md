# Known Issues and Planned Improvements

## V2 Implementation Status

The `bjj_rag_v2.py` implementation addresses most of the original issues. See benchmark results below.

### Benchmark Results (V1 vs V2)

| Metric | V1 | V2 | Improvement |
|--------|----|----|-------------|
| Mean Precision@5 | 58.54% | 71.46% | **+12.92%** |
| Mean Exclusion Rate | 36.98% | 45.31% | +8.33%* |
| Mean Latency | 17.5s | 30.3s | +12.8s |

*Exclusion rate measures wrong-intent content (e.g., "triangle setup" when asking for "triangle escape"). Higher rate in V2 is partly due to stricter evaluation - V2 retrieves more content overall, some of which contains related terminology.

---

## Issue Status

### 1. Duplicate Results from Same Timestamp - SOLVED

**Status:** ✅ Implemented in V2

**Solution:**
- Timestamp deduplication with 45-second window grouping
- MMR (Maximal Marginal Relevance) for semantic diversity
- Instructor diversity cap (max 4 per instructor)

**Code:** `dedupe_by_timestamp()`, `mmr_select()`, `enforce_instructor_diversity()` in `bjj_rag_v2.py`

---

### 2. Keyword Prioritization is Too Rigid - SOLVED

**Status:** ✅ Implemented in V2

**Solution:**
- Replaced hardcoded keywords with intent classification (ESCAPE/EXECUTE/DEFENSE/COUNTER/CONCEPT)
- BGE cross-encoder reranking (`BAAI/bge-reranker-v2-m3`) for semantic discrimination
- HyDE (Hypothetical Document Embeddings) for better query understanding
- Multi-pass retrieval with RRF fusion

**Code:** `classify_intent()`, `_rerank()`, `generate_hypothetical_document()` in `bjj_rag_v2.py`

---

### 3. LLM Sometimes Summarizes Irrelevant Content - SOLVED

**Status:** ✅ Implemented in V2

**Solution:**
- Two-stage filtering: LLM scores relevance 0-10 before synthesis
- Strict relevance prompt emphasizing intent matching
- Minimum score threshold (6/10) filters low-relevance chunks
- Answer-first synthesis format

**Code:** `score_relevance()`, `RELEVANCE_SCORING_PROMPT` in `bjj_rag_v2.py`

---

### 4. No Source Deduplication Across Instructors - PARTIALLY SOLVED

**Status:** ⚠️ Partially addressed

**Current Solution:**
- Instructor diversity enforcement ensures varied perspectives
- MMR provides semantic diversity across results

**Remaining Work:**
- [ ] Semantic clustering to group similar techniques across instructors
- [ ] "Multiple instructors cover this" presentation

---

## Remaining Issues

### 5. Small Chunk Size (400 tokens)

**Problem:** Current chunks often cut mid-explanation, losing context.

**Status:** 🔄 Ready to implement

**Solution:** Re-chunking script (`rechunk.py`) ready to create 800-1000 token chunks with sentence boundaries.

```bash
python rechunk.py --rebuild-db --target-tokens 800
```

---

### 6. Latency Increased

**Problem:** V2 takes ~30s vs V1's ~17s due to additional processing stages.

**Cause:** Multi-pass retrieval, cross-encoder reranking, and LLM relevance scoring add latency.

**Potential Optimizations:**
- [ ] Batch reranker calls
- [ ] Cache HyDE generations for similar queries
- [ ] Use smaller reranker model for initial filtering
- [ ] Async processing of independent stages

---

## Future Improvements

### Search Quality
- [x] Implement proper deduplication by video+timestamp
- [x] Replace keyword boosting with cross-encoder reranking
- [x] Add intent classification
- [x] HyDE for better retrieval
- [ ] Re-chunk to 800-1000 tokens with sentence boundaries
- [ ] Add instructor/series filtering options
- [ ] Support time range queries ("show me the first 10 minutes of X")

### User Experience
- [ ] Web UI with video player integration
- [ ] Clickable timestamps that open video at position
- [ ] Save/bookmark search results
- [ ] Search history

### Performance
- [ ] Cache embedding model between queries (persistent server)
- [ ] Batch multiple queries
- [ ] Smaller/faster LLM options for quick searches
- [ ] Optimize reranker batching

### Data
- [ ] Re-chunk with better segmentation (sentence boundaries) - script ready
- [ ] Add instructor metadata to chunks
- [ ] Index video titles/descriptions separately
- [ ] Parent-child chunk architecture for context expansion
