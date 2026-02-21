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

import os
import argparse
from pathlib import Path
from collections import defaultdict

import lancedb
from dotenv import load_dotenv

from llm import LLMClient


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
        top_k: int = 25,
        profile: str = "laptop",
    ):
        """
        Initialize the RAG system.

        Args:
            db_path: Path to LanceDB database directory
            llm_model: Ollama model name for synthesis
            embedding_dim: Matryoshka dimension (512 recommended)
            top_k: Default number of chunks to retrieve
            profile: Hardware profile (laptop or homeserver)
        """
        self.top_k = top_k
        self.llm_model = llm_model
        self.embedding_dim = embedding_dim
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

        # Connect to vector database
        print(f"Connecting to database at {db_path}...")
        self.db = lancedb.connect(db_path)
        self.table = self.db.open_table("transcripts")

    def _format_timestamp(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS or MM:SS format."""
        h, remainder = divmod(int(seconds), 3600)
        m, s = divmod(remainder, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

    def _embed_query(self, query: str):
        """
        Embed query using nomic's asymmetric search prefix (laptop)
        or OpenAI text-embedding-3-small (homeserver).
        """
        if self.profile == "homeserver":
            return self.llm.embed(query, dimensions=self.embedding_dim)

        import torch
        prefixed = "search_query: " + query
        with torch.no_grad():
            emb = self.embedder.encode(
                [prefixed],
                convert_to_tensor=True,
                device=self._embed_device,
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

    def _group_by_instructor(self, chunks: list[dict]) -> dict[str, list[dict]]:
        """Group chunks by instructor (first directory component of video_file)."""
        grouped = defaultdict(list)

        for chunk in chunks:
            video_path = chunk.get('video_file', '')
            parts = Path(video_path).parts
            instructor = parts[0] if parts else 'Unknown'

            path = Path(video_path)
            video_title = path.stem.replace('.json', '').replace('.opus', '')

            start_time = chunk.get('start_time', 0)
            end_time = chunk.get('end_time', start_time)

            grouped[instructor].append({
                'video_title': video_title,
                'timestamp': self._format_timestamp(start_time),
                'end_time': self._format_timestamp(end_time),
                'text': chunk.get('text', '')[:200],
                'relevance_score': 0,
            })

        return dict(sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True))

    def _synthesize_brief(self, query: str, grouped: dict[str, list[dict]]) -> str:
        """Generate a 2-3 sentence brief summary of what the sources cover."""
        if not grouped:
            return "No relevant content found in the transcript database."

        lines = []
        for instructor, results in grouped.items():
            snippets = [r['text'][:100] for r in results[:3]]
            lines.append(f"- {instructor} ({len(results)} clips): {' | '.join(snippets)}")

        instructor_summary = "\n".join(lines)

        prompt = f"""You are summarizing BJJ video search results. Be EXTREMELY brief.

QUERY: {query}

INSTRUCTORS AND THEIR TOPICS:
{instructor_summary}

Write 2-3 sentences MAX summarizing what the sources cover. Rules:
- ONLY mention themes/techniques that appear across MULTIPLE sources
- Name specific instructors only if they have a distinctive approach
- Do NOT teach the technique — just describe what coverage exists
- Do NOT list every source — summarize the overall picture

Brief summary:"""

        return self.llm.generate(
            prompt=prompt,
            temperature=0.3,
            max_tokens=200,
        )

    def search(self, query: str, top_k: int = None) -> tuple[str, dict[str, list[dict]]]:
        """
        Full RAG pipeline: retrieve chunks, group by instructor, brief summary.

        Returns:
            Tuple of (brief_summary, grouped_results)
        """
        chunks = self._retrieve(query, top_k)
        grouped = self._group_by_instructor(chunks)
        summary = self._synthesize_brief(query, grouped)
        return summary, grouped

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

            print("\nSearching...\n")

            try:
                summary, grouped = self.search(query)
                print(summary)
                print("\n" + "-" * 60)
                total_results = sum(len(v) for v in grouped.values())
                print(f"Results: {total_results} clips from {len(grouped)} instructors\n")
                for instructor, results in grouped.items():
                    print(f"\n  {instructor} ({len(results)} clips):")
                    for r in results:
                        print(f"    {r['video_title']} @ {r['timestamp']}-{r['end_time']}")
                        print(f"           {r['text'][:100]}...")
            except Exception as e:
                print(f"Error: {e}")


def main():
    """Entry point for the search application."""
    load_dotenv()

    parser = argparse.ArgumentParser(description="BJJ Transcript Search")
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
        help="Number of chunks to retrieve"
    )
    parser.add_argument(
        "--query",
        help="Single query (non-interactive mode)"
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

    rag = BJJSearchRAG(
        db_path=args.db,
        llm_model=args.model,
        top_k=args.top_k,
        profile=args.profile,
    )

    if args.query:
        # Single query mode
        summary, grouped = rag.search(args.query)
        print(summary)
        for instructor, results in grouped.items():
            print(f"\n{instructor} ({len(results)} clips):")
            for r in results:
                print(f"  {r['video_title']} @ {r['timestamp']}-{r['end_time']}")
    else:
        # Interactive mode
        rag.interactive()


if __name__ == "__main__":
    main()
