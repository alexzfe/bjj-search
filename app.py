#!/usr/bin/env python3
"""
BJJ Transcript Search - Gradio Web Interface

Source-first search: find WHICH VIDEOS to watch at WHAT TIMESTAMPS.

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


def format_grouped_results(grouped: dict[str, list[dict]]) -> str:
    """Format instructor-grouped results as markdown with headers."""
    if not grouped:
        return "No results found."

    sections = []
    for instructor, results in grouped.items():
        lines = [f"### {instructor} ({len(results)} clips)\n"]
        lines.append("| Video | Timestamps | Score | Excerpt |")
        lines.append("|-------|-----------|-------|---------|")

        for r in results:
            title = r['video_title']
            time_range = f"{r['timestamp']} - {r['end_time']}"
            score = r['relevance_score']
            excerpt = r['text'][:150].replace('\n', ' ').replace('|', '\\|')
            if len(r['text']) > 150:
                excerpt += "..."
            lines.append(f"| {title} | {time_range} | {score}/10 | {excerpt} |")

        sections.append("\n".join(lines))

    return "\n\n---\n\n".join(sections)


def search(query: str, rag: BJJSearchRAGv2) -> tuple[str, str, str]:
    """
    Perform search and return formatted results.

    Returns:
        (intent, summary, grouped_results)
    """
    if not query or len(query.strip()) < 3:
        return "N/A", "Please enter a query with at least 3 characters.", ""

    try:
        # Get intent
        intent = classify_intent(query, rag.llm)
        intent_desc = INTENT_DESCRIPTIONS.get(intent, "understand this technique")
        intent_display = f"**{intent}** - {intent_desc}"

        # Run search
        summary, grouped = rag.search(query)

        # Format grouped results
        results_md = format_grouped_results(grouped)

        total = sum(len(v) for v in grouped.values())
        results_header = f"**{total} results from {len(grouped)} instructors**\n\n"

        return intent_display, summary, results_header + results_md

    except Exception as e:
        return "Error", f"Search failed: {str(e)}", ""


def create_app(rag: BJJSearchRAGv2) -> gr.Blocks:
    """Create the Gradio interface."""

    with gr.Blocks(title="BJJ Transcript Search") as app:

        gr.Markdown(
            """
            # BJJ Transcript Search

            Find which videos to watch and at what timestamps. Search 564 hours of BJJ instructionals.

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

        summary_output = gr.Markdown(label="Summary")
        results_output = gr.Markdown(label="Results by Instructor")

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
            outputs=[intent_output, summary_output, results_output]
        )

        query_input.submit(
            fn=do_search,
            inputs=[query_input],
            outputs=[intent_output, summary_output, results_output]
        )

        gr.Markdown(
            """
            ---
            **About:** Source-first search through BJJ instructional transcripts.
            Results grouped by instructor — find the right video to watch.
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
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True,
        theme=gr.themes.Soft()
    )


if __name__ == "__main__":
    main()
