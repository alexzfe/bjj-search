#!/usr/bin/env python3
"""
BJJ Transcript Search - Gradio Web Interface

A simple web UI for searching BJJ instructional video transcripts.

Usage:
    python app.py                    # Start on default port 7860
    python app.py --port 8080        # Custom port
    python app.py --share            # Create public link

Requirements:
    pip install gradio
"""

import os
import argparse

import gradio as gr
from pathlib import Path
from dotenv import load_dotenv

# Import the v2 RAG system
from bjj_rag_v2 import (
    BJJSearchRAGv2,
    classify_intent,
    INTENT_DESCRIPTIONS
)


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS or MM:SS format."""
    h, remainder = divmod(int(seconds), 3600)
    m, s = divmod(remainder, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def format_sources_table(chunks: list[dict]) -> str:
    """Format chunks as a markdown table."""
    if not chunks:
        return "No sources found."

    rows = []
    for i, chunk in enumerate(chunks, 1):
        video_path = Path(chunk.get("video_file", f"Video {i}"))

        # Build readable title
        if len(video_path.parts) >= 2:
            video_title = f"{video_path.parts[0]} / {video_path.parts[-1]}"
        else:
            video_title = video_path.stem
        video_title = video_title.replace('.json', '').replace('.opus', '')

        # Format time range
        start_time = chunk.get("start_time")
        end_time = chunk.get("end_time")

        if start_time is not None and end_time is not None:
            start_str = format_timestamp(start_time)
            end_str = format_timestamp(end_time)
            timestamp = f"{start_str} - {end_str}"
        elif start_time is not None:
            timestamp = format_timestamp(start_time)
        else:
            timestamp = "N/A"

        relevance = chunk.get("llm_relevance_score", "N/A")

        rows.append(f"| {i} | {video_title} | {timestamp} | {relevance}/10 |")

    header = "| # | Video | Time Range | Relevance |\n|---|-------|------------|-----------|"
    return header + "\n" + "\n".join(rows)


def search(query: str, rag: BJJSearchRAGv2) -> tuple[str, str, str, str]:
    """
    Perform search and return formatted results.

    Returns:
        (intent, answer, sources_table, raw_chunks_json)
    """
    if not query or len(query.strip()) < 3:
        return "N/A", "Please enter a query with at least 3 characters.", "", ""

    try:
        # Get intent
        intent = classify_intent(query, rag.llm)
        intent_desc = INTENT_DESCRIPTIONS.get(intent, "understand this technique")
        intent_display = f"**{intent}** - {intent_desc}"

        # Run search
        synthesis, chunks = rag.search(query)

        # Format sources
        sources_table = format_sources_table(chunks)

        # Format raw chunks for expandable section
        raw_chunks = ""
        for i, chunk in enumerate(chunks, 1):
            video_path = Path(chunk.get("video_file", ""))
            start_time = chunk.get("start_time")
            end_time = chunk.get("end_time")

            if start_time is not None and end_time is not None:
                time_str = f"{format_timestamp(start_time)} - {format_timestamp(end_time)}"
            elif start_time is not None:
                time_str = format_timestamp(start_time)
            else:
                time_str = "N/A"

            raw_chunks += f"### Source {i}: {video_path.name} @ {time_str}\n\n"
            raw_chunks += f"**Relevance:** {chunk.get('llm_relevance_score', 'N/A')}/10\n\n"
            raw_chunks += f"{chunk.get('text', '')}\n\n---\n\n"

        return intent_display, synthesis, sources_table, raw_chunks

    except Exception as e:
        return "Error", f"Search failed: {str(e)}", "", ""


def create_app(rag: BJJSearchRAGv2) -> gr.Blocks:
    """Create the Gradio interface."""

    with gr.Blocks(title="BJJ Transcript Search") as app:

        gr.Markdown(
            """
            # BJJ Transcript Search

            Search through 564 hours of BJJ instructional video transcripts with AI-powered summaries.

            **Tips:**
            - Be specific: "triangle escape" works better than just "triangle"
            - Include intent: "how to escape mount" vs "mount attacks"
            - Try technique + position: "kimura from closed guard"
            """,
            elem_classes=["header"]
        )

        with gr.Row():
            with gr.Column(scale=4):
                query_input = gr.Textbox(
                    label="Search Query",
                    placeholder="e.g., triangle escape, armbar from mount, heel hook defense...",
                    lines=1,
                    max_lines=1
                )
            with gr.Column(scale=1):
                search_btn = gr.Button("Search", variant="primary", size="lg")

        with gr.Row():
            intent_output = gr.Markdown(label="Detected Intent")

        with gr.Tabs():
            with gr.TabItem("Answer"):
                answer_output = gr.Markdown(label="Synthesized Answer")

            with gr.TabItem("Sources"):
                sources_output = gr.Markdown(label="Source Videos")

            with gr.TabItem("Raw Transcripts"):
                raw_output = gr.Markdown(label="Full Transcript Excerpts")

        # Example queries
        gr.Examples(
            examples=[
                ["triangle escape"],
                ["armbar from mount"],
                ["heel hook defense"],
                ["half guard sweeps"],
                ["back escape when they have hooks"],
                ["closed guard attacks"],
                ["pressure passing philosophy"],
                ["kimura from side control"],
            ],
            inputs=query_input,
            label="Example Queries"
        )

        # Wire up the search
        def do_search(query):
            return search(query, rag)

        search_btn.click(
            fn=do_search,
            inputs=[query_input],
            outputs=[intent_output, answer_output, sources_output, raw_output]
        )

        query_input.submit(
            fn=do_search,
            inputs=[query_input],
            outputs=[intent_output, answer_output, sources_output, raw_output]
        )

        gr.Markdown(
            """
            ---
            **About:** This tool uses local AI models to search through BJJ instructional transcripts.
            No data is sent to external servers. Powered by Llama 3.1 8B and BGE Reranker.
            """
        )

    return app


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="BJJ Search Web Interface")
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
        "--port",
        type=int,
        default=7860,
        help="Port to run the server on"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public link"
    )
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="Disable BGE reranker (faster, less accurate)"
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

    print("Initializing BJJ Search...")
    rag = BJJSearchRAGv2(
        db_path=args.db,
        llm_model=args.model,
        use_reranker=not args.no_reranker,
        profile=args.profile,
    )

    print("Starting web interface...")
    app = create_app(rag)

    app.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
        theme=gr.themes.Soft()
    )


if __name__ == "__main__":
    main()
