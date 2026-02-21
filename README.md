# BJJ Transcript Search

Local RAG (Retrieval-Augmented Generation) system for searching 564 hours of BJJ instructional video transcripts with LLM-powered summaries.

## Overview

This system allows you to search through transcribed BJJ instructional videos using natural language queries. It combines vector similarity search with a local LLM to provide contextual summaries of relevant content, including timestamps for easy video navigation.

**Two versions available:**
- `bjj_rag.py` - Original simple implementation
- `bjj_rag_v2.py` - **Improved version** with better relevance, deduplication, and answer-first output

**Data:**
- 716 transcribed videos
- 19,515 searchable chunks (400 tokens each, 20% overlap)
- Word-level timestamps preserved

**Stack:**
- **Embeddings:** nomic-embed-text-v1.5 (512-dim Matryoshka)
- **Vector DB:** LanceDB (local, file-based)
- **LLM:** Llama 3.1 8B via Ollama
- **Reranker (v2):** BGE-reranker-v2-m3 for semantic discrimination
- **GPU:** ~6.5GB VRAM total (RTX 4070 compatible)

## Directory Structure

```
~/projects/bjj-search/
├── app.py                  # Gradio web interface
├── bjj_rag.py              # Original search script
├── bjj_rag_v2.py           # Improved search with heavy reranking
├── evaluate.py             # Evaluation framework
├── rechunk.py              # Re-chunking script for larger chunks
├── README.md               # This file
├── ISSUES.md               # Known issues and improvements
├── bjj_rag_improvements.md # Detailed improvement research
└── data/
    ├── bjj_search_db/      # LanceDB vector database (87 MB)
    ├── transcripts/        # Raw JSON transcripts (533 MB)
    ├── chunks.json         # Pre-processed chunks (46 MB)
    └── embeddings.npy      # Pre-computed vectors (39 MB)
```

## Requirements

### System
- NVIDIA GPU with 8GB+ VRAM
- CUDA drivers installed
- ~1GB disk space for models

### Python Dependencies

For v1 (original):
```bash
pip install lancedb sentence-transformers ollama einops
```

For v2 (improved) - adds reranker and spaCy:
```bash
pip install lancedb sentence-transformers ollama einops FlagEmbedding spacy
python -m spacy download en_core_web_sm
```

### Ollama
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the LLM model
sudo ollama pull llama3.1:8b

# Verify it's running
ollama list
```

## Usage

### Web Interface (Recommended)

The easiest way to use the search:

```bash
# Install Gradio
pip install gradio

# Start the web interface
python app.py
```

Then open http://localhost:7860 in your browser.

**Options:**
```bash
python app.py --port 8080        # Custom port
python app.py --share            # Create public link
python app.py --no-reranker      # Faster, less accurate
```

### V2 Command Line

The improved version with better semantic discrimination:

```bash
# Interactive mode
python bjj_rag_v2.py

# Single query with verbose output
python bjj_rag_v2.py --query "triangle escape" -v

# Without reranker (faster but less accurate)
python bjj_rag_v2.py --query "mount escape" --no-reranker
```

**V2 Improvements:**
- Intent classification (ESCAPE vs EXECUTE vs DEFENSE)
- HyDE for better retrieval
- BGE cross-encoder reranking
- Timestamp deduplication (no more duplicate results)
- MMR diversity (varied sources)
- LLM relevance scoring
- Answer-first output format

### V1 (Original)

```bash
# Interactive mode
python bjj_rag.py

# Single query
python bjj_rag.py --query "armbar from mount"
```

### Common Options
```bash
python bjj_rag_v2.py --help

Options:
  --db PATH       Path to LanceDB database (default: ./data/bjj_search_db)
  --model NAME    Ollama model for synthesis (default: llama3.1:8b)
  --top-k N       Number of final chunks (default: 12 for v2, 15 for v1)
  --query TEXT    Single query (non-interactive mode)
  --no-reranker   Disable BGE reranker (v2 only)
  -v, --verbose   Show pipeline stages (v2 only)
```

### Examples
```bash
# Triangle escape - v2 will correctly find escapes, not setups
python bjj_rag_v2.py --query "triangle escape" -v

# Compare results between v1 and v2
python bjj_rag.py --query "heel hook defense"
python bjj_rag_v2.py --query "heel hook defense"
```

## Output Format

Results include:
- **Source path:** Instructor/Series/Video name
- **Timestamp:** MM:SS or HH:MM:SS format
- **Summary:** LLM-generated description of what's taught

Example:
```
**[John Danaher/New Wave/Mount Attacks/Vol 4 @ 2:05]**
John Danaher discusses the armbar from mount, highlighting its limitations
and risks. He emphasizes that attempting an armbar from mount can lead to
losing top position if it fails.
```

## How It Works

### V1 Pipeline
1. **Query Embedding:** Query → 512-dim vector via nomic-embed-text-v1.5
2. **Vector Search:** LanceDB finds similar chunks
3. **Keyword Boosting:** Hardcoded BJJ keywords boost matching results
4. **LLM Synthesis:** Per-source summaries

### V2 Pipeline (8 stages, ~30 seconds)
1. **Intent Classification:** LLM categorizes query as ESCAPE/EXECUTE/DEFENSE/COUNTER/CONCEPT
2. **Multi-Pass Retrieval:**
   - Direct hybrid search (BM25 + vector)
   - Intent-expanded query search
   - HyDE (hypothetical document) search
3. **RRF Fusion:** Merge 300+ candidates via Reciprocal Rank Fusion
4. **Cross-Encoder Reranking:** BGE reranker scores query-document pairs
5. **Deduplication:** Remove timestamp overlaps (45s window)
6. **Instructor Diversity:** Cap results per instructor (max 4)
7. **MMR Diversity:** Balance relevance vs variety (λ=0.6)
8. **LLM Relevance Scoring:** Filter chunks scoring <6/10
9. **Answer-First Synthesis:** Direct answer, then timestamps and sources

## Benchmark Results

V2 shows significant improvement in retrieval precision:

| Metric | V1 | V2 | Change |
|--------|----|----|--------|
| **Mean Precision@5** | 58.54% | 71.46% | **+12.92%** |
| Mean Exclusion Rate | 36.98% | 45.31% | +8.33% |
| Mean Latency | 17.5s | 30.3s | +12.8s |

**Key improvements:**
- "triangle escape" correctly finds escape content, not triangle setups
- "half guard sweeps" precision improved from 25% to 75%
- "pressure passing philosophy" and "guard retention principles" hit 100% precision

See `ISSUES.md` for detailed analysis.

## Evaluation

Run the evaluation suite:

```bash
# Evaluate v2 only
python evaluate.py --system v2 -v

# Compare both systems
python evaluate.py --compare -v
```

Metrics measured:
- **Precision@5:** Expected topics in top 5 results
- **Exclusion Rate:** Wrong-intent results (e.g., triangle attack when asking for escape)
- **Latency:** Query processing time

## Re-chunking

Create larger chunks (800-1000 tokens) for better context:

```bash
# Preview what would be created
python rechunk.py --dry-run --target-tokens 800

# Re-chunk and rebuild database
python rechunk.py --rebuild-db --target-tokens 800
```

This creates a new database at `data/bjj_search_db_v2/`.

## Transcript Format

Raw transcripts in `data/transcripts/` are JSON files with:
```json
{
  "source_file": "Instructor/Series/Video.opus",
  "duration": 3600.5,
  "language": "en",
  "segments": [
    {
      "start": 13.52,
      "end": 42.18,
      "text": "What's up guys...",
      "words": [
        {"word": "What's", "start": 13.52, "end": 14.0},
        ...
      ]
    }
  ]
}
```

## Troubleshooting

### Ollama EOF errors
```bash
sudo systemctl restart ollama
sudo ollama pull llama3.1:8b
```

### Missing einops
```bash
pip install einops
```

### CUDA out of memory
Try a smaller model or disable the reranker:
```bash
# Use smaller LLM
sudo ollama pull llama3.2:3b
python bjj_rag_v2.py --model llama3.2:3b

# Or disable reranker (saves ~1.2GB VRAM)
python bjj_rag_v2.py --no-reranker
```

### V2 CUDA memory issues
V2 loads multiple models (~6.5GB total). If you get OOM errors:
```bash
# Clear GPU cache before running
python -c "import torch; torch.cuda.empty_cache()"

# Or run without reranker
python bjj_rag_v2.py --no-reranker --query "your query"
```

### Slow first query
The embedding model loads on first query (~5-10 seconds). V2 also loads the BGE reranker (~3 seconds). Subsequent queries are faster.

### FlagEmbedding installation issues
If lxml fails to build:
```bash
# Fedora/RHEL
sudo dnf install -y libxml2-devel libxslt-devel

# Ubuntu/Debian
sudo apt install -y libxml2-dev libxslt-dev

# Then retry
pip install FlagEmbedding
```

## Data Pipeline (Reference)

This documents how the data was created (for reproducibility):

1. **Transcription:** faster-whisper large-v3 on Crusoe Cloud 4x L40S GPU
   - 716 files, 564 hours total
   - Word-level timestamps via VAD + DTW
   - BJJ terminology prompt for accuracy

2. **Chunking:** 400 tokens with 20% overlap
   - Preserves timestamp mapping
   - 19,515 total chunks

3. **Embedding:** nomic-embed-text-v1.5
   - 512-dim Matryoshka truncation
   - Document prefix: "search_document: "
   - Query prefix: "search_query: "

4. **Indexing:** LanceDB with full-text search index
