#!/usr/bin/env python3
"""
BJJ Transcript Search - Fully Local RAG with Ollama LLM Synthesis

Usage:
    python bjj_rag.py                           # Interactive mode
    python bjj_rag.py --query "kimura setup"    # Single query
    python bjj_rag.py --top-k 10                # More results

Requirements:
    pip install lancedb sentence-transformers ollama
    ollama pull llama3.1:8b
"""

import ollama
from sentence_transformers import SentenceTransformer
import lancedb
from pathlib import Path
import torch
import argparse


class BJJSearchRAG:
    """
    Local RAG system for BJJ instructional video transcripts.

    Architecture:
    1. Query embedding via nomic-embed-text-v1.5 (GPU)
    2. Vector similarity search via LanceDB
    3. LLM synthesis via Ollama (GPU)

    Total VRAM usage: ~6GB (fits RTX 4070 8GB comfortably)
    Query latency: ~3-5 seconds including synthesis
    """

    def __init__(
        self,
        db_path: str,
        llm_model: str = "llama3.1:8b",
        embedding_dim: int = 512,
        top_k: int = 15
    ):
        """
        Initialize the RAG system.

        Args:
            db_path: Path to LanceDB database directory
            llm_model: Ollama model name for synthesis
            embedding_dim: Matryoshka dimension (512 recommended)
            top_k: Default number of chunks to retrieve
        """
        self.top_k = top_k
        self.llm_model = llm_model
        self.embedding_dim = embedding_dim

        # Load embedding model (~1GB VRAM)
        print("Loading embedding model...")
        self.embedder = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5",
            trust_remote_code=True
        )
        self.embedder = self.embedder.half().to("cuda")
        self.embedder.max_seq_length = 8192

        # Connect to vector database
        print(f"Connecting to database at {db_path}...")
        self.db = lancedb.connect(db_path)
        self.table = self.db.open_table("transcripts")

        # Verify Ollama is running
        try:
            ollama.list()
            print(f"Ollama connected, using model: {llm_model}")
        except Exception as e:
            raise RuntimeError(
                f"Ollama not running. Start with: ollama serve\n"
                f"Then pull model: ollama pull {llm_model}"
            ) from e

    def _format_timestamp(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS or MM:SS format."""
        h, remainder = divmod(int(seconds), 3600)
        m, s = divmod(remainder, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

    def _embed_query(self, query: str):
        """
        Embed query using nomic's asymmetric search prefix.

        Nomic uses different prefixes for queries vs documents:
        - Documents: "search_document: {text}"
        - Queries: "search_query: {text}"

        This asymmetric approach improves retrieval quality.
        """
        prefixed = "search_query: " + query
        with torch.no_grad():
            emb = self.embedder.encode(
                [prefixed],
                convert_to_tensor=True,
                device="cuda",
                normalize_embeddings=True
            )
            # Matryoshka truncation to configured dimension
            if self.embedding_dim < emb.shape[1]:
                emb = emb[:, :self.embedding_dim]
        return emb.cpu().numpy()[0]

    def _retrieve(self, query: str, top_k: int = None) -> list[dict]:
        """
        Retrieve relevant chunks via hybrid search (vector + full-text).

        Returns list of dicts with: video_file, start_time, end_time, text
        """
        k = top_k or self.top_k
        query_vec = self._embed_query(query)

        # Hybrid search: combine vector similarity with full-text keyword matching
        # This ensures important keywords (like "escape") are weighted properly
        results = (
            self.table
            .search(query_vec)
            .limit(k * 2)  # Get more candidates
            .to_pandas()
        )

        # Boost results that contain query keywords
        query_words = set(query.lower().split())
        important_words = {'escape', 'defense', 'counter', 'setup', 'entry', 'finish', 'attack', 'sweep', 'pass', 'guard', 'mount', 'back', 'side'}
        key_terms = query_words & important_words

        if key_terms:
            def keyword_score(text):
                text_lower = text.lower()
                return sum(1 for term in key_terms if term in text_lower)

            results['keyword_score'] = results['text'].apply(keyword_score)
            # Sort by keyword matches first, then by vector similarity (lower _distance = better)
            results = results.sort_values(
                by=['keyword_score', '_distance'],
                ascending=[False, True]
            )

        return results.head(k).to_dict('records')

    def _synthesize(self, query: str, chunks: list[dict]) -> str:
        """
        Use local LLM to summarize what each chunk says.

        The prompt instructs the model to:
        1. Summarize the content of each chunk
        2. Explain how it relates to the user's query
        3. Skip chunks that aren't actually relevant
        """
        if not chunks:
            return "No relevant content found in the transcript database."

        # Build context from retrieved chunks
        context_blocks = []
        for i, chunk in enumerate(chunks):
            # Include folder path for context (e.g., "Instructor/Series/Video")
            video_path = Path(chunk["video_file"])
            video_name = f"{video_path.parent}/{video_path.stem}" if video_path.parent != Path('.') else video_path.stem
            timestamp = self._format_timestamp(chunk["start_time"])
            context_blocks.append(
                f"[Source {i+1}: {video_name} @ {timestamp}]\n"
                f"{chunk['text']}"
            )

        context = "\n\n".join(context_blocks)

        # Synthesis prompt - focuses on summarizing chunk content
        prompt = f"""You are analyzing BJJ (Brazilian Jiu-Jitsu) instructional video transcripts to help a practitioner find relevant techniques.

USER'S QUESTION: {query}

RETRIEVED TRANSCRIPT EXCERPTS:
{context}

YOUR TASK:
For each source that is relevant to the user's question, provide:
1. The source reference (video name @ timestamp)
2. A concise summary of what the instructor is teaching in that segment
3. How this content addresses the user's question

FORMAT YOUR RESPONSE LIKE THIS:
**[Video-Name @ MM:SS]**
Summary of what this segment covers and how it helps answer the question.

GUIDELINES:
- Summarize what the transcript SAYS, don't add information that isn't there
- Focus on specific techniques, setups, grips, or concepts mentioned
- Skip sources that aren't actually relevant to the question
- Keep each summary to 2-3 sentences maximum
- Use BJJ terminology accurately"""

        response = ollama.chat(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.3,  # Lower = more factual/consistent
                "num_predict": 1024,  # Max tokens in response
            }
        )

        return response["message"]["content"]

    def search(self, query: str, top_k: int = None) -> tuple[str, list[dict]]:
        """
        Full RAG pipeline: retrieve chunks and synthesize summary.

        Args:
            query: Natural language search query
            top_k: Number of chunks to retrieve (default: self.top_k)

        Returns:
            Tuple of (synthesis_text, raw_chunks)
        """
        chunks = self._retrieve(query, top_k)
        synthesis = self._synthesize(query, chunks)
        return synthesis, chunks

    def interactive(self):
        """Run interactive search loop in terminal."""
        print("\n" + "=" * 60)
        print("BJJ Transcript Search")
        print("=" * 60)
        print(f"Model: {self.llm_model} | Chunks per query: {self.top_k}")
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

            print("\nSearching and synthesizing...\n")

            try:
                synthesis, chunks = self.search(query)
                print(synthesis)
                print("\n" + "-" * 60)
                print(f"Retrieved {len(chunks)} chunks | "
                      f"Videos: {len(set(c['video_file'] for c in chunks))}")
            except Exception as e:
                print(f"Error: {e}")


def main():
    """Entry point for the search application."""
    parser = argparse.ArgumentParser(description="BJJ Transcript Search")
    parser.add_argument(
        "--db",
        default="./data/bjj_search_db",
        help="Path to LanceDB database"
    )
    parser.add_argument(
        "--model",
        default="llama3.1:8b",
        help="Ollama model for synthesis"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Number of chunks to retrieve"
    )
    parser.add_argument(
        "--query",
        help="Single query (non-interactive mode)"
    )

    args = parser.parse_args()

    rag = BJJSearchRAG(
        db_path=args.db,
        llm_model=args.model,
        top_k=args.top_k
    )

    if args.query:
        # Single query mode
        synthesis, _ = rag.search(args.query)
        print(synthesis)
    else:
        # Interactive mode
        rag.interactive()


if __name__ == "__main__":
    main()
