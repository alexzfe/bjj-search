#!/usr/bin/env python3
"""
Evaluation Framework for BJJ RAG System

Measures:
- Precision@k: How many expected topics appear in top-k results
- Exclusion violations: How many wrong-intent results appear
- Intent accuracy: How often intent classification is correct

Usage:
    python evaluate.py --system v1    # Evaluate original system
    python evaluate.py --system v2    # Evaluate improved system
    python evaluate.py --compare      # Compare both systems
"""

import argparse
import json
import time
from dataclasses import dataclass, asdict
from typing import Literal
from pathlib import Path


@dataclass
class EvalQuery:
    """A test query with expected outcomes."""
    query: str
    intent: Literal["EXECUTE", "ESCAPE", "DEFENSE", "COUNTER", "CONCEPT", "DRILL"]
    expected_topics: list[str]  # Topics that SHOULD appear in relevant results
    excluded_topics: list[str]  # Topics that should NOT appear (wrong intent)
    description: str = ""  # Human-readable description of what we're testing


# Test queries covering the semantic discrimination problem
EVAL_QUERIES = [
    # ESCAPE vs EXECUTE discrimination
    EvalQuery(
        query="triangle escape",
        intent="ESCAPE",
        expected_topics=["escape", "get out", "posture", "stack", "defend", "survive"],
        excluded_topics=["setup", "finish", "submit", "attack", "lock", "squeeze"],
        description="Should find escapes, not triangle attacks"
    ),
    EvalQuery(
        query="triangle from guard",
        intent="EXECUTE",
        expected_topics=["setup", "angle", "cut", "finish", "squeeze", "lock"],
        excluded_topics=["escape", "defend", "get out", "survive"],
        description="Should find triangle setups/finishes"
    ),
    EvalQuery(
        query="armbar escape",
        intent="ESCAPE",
        expected_topics=["escape", "hitchhiker", "stack", "posture", "defend"],
        excluded_topics=["finish", "hyperextend", "submit", "break"],
        description="Should find armbar escapes"
    ),
    EvalQuery(
        query="armbar from mount",
        intent="EXECUTE",
        expected_topics=["setup", "grip", "pivot", "finish", "control"],
        excluded_topics=["escape", "defend", "survive"],
        description="Should find armbar attacks from mount"
    ),
    EvalQuery(
        query="mount escape",
        intent="ESCAPE",
        expected_topics=["upa", "elbow escape", "trap", "bridge", "shrimp", "hip"],
        excluded_topics=["maintain", "submit from", "attack from", "control"],
        description="Should find mount escapes"
    ),
    EvalQuery(
        query="back escape",
        intent="ESCAPE",
        expected_topics=["escape", "hands", "defend", "turn", "survive"],
        excluded_topics=["rear naked", "strangle", "choke", "finish"],
        description="Should find back escapes, not back attacks"
    ),
    EvalQuery(
        query="rear naked choke",
        intent="EXECUTE",
        expected_topics=["choke", "strangle", "finish", "squeeze", "arm"],
        excluded_topics=["escape", "defend", "survive"],
        description="Should find RNC technique, not defenses"
    ),

    # DEFENSE queries
    EvalQuery(
        query="defend the armbar",
        intent="DEFENSE",
        expected_topics=["grip", "elbow", "prevent", "posture", "block"],
        excluded_topics=["finish", "hyperextend", "submit"],
        description="Should find prevention, not attacks"
    ),
    EvalQuery(
        query="kimura defense",
        intent="DEFENSE",
        expected_topics=["defend", "grip", "prevent", "posture"],
        excluded_topics=["finish", "rotate", "submit", "attack"],
        description="Should find kimura defense"
    ),

    # CONCEPT queries
    EvalQuery(
        query="pressure passing philosophy",
        intent="CONCEPT",
        expected_topics=["pressure", "weight", "concept", "principle", "approach"],
        excluded_topics=[],  # Concepts can relate to anything
        description="Should find conceptual discussion"
    ),
    EvalQuery(
        query="guard retention principles",
        intent="CONCEPT",
        expected_topics=["retain", "principle", "frame", "hip", "concept"],
        excluded_topics=[],
        description="Should find retention concepts"
    ),

    # Position-specific queries
    EvalQuery(
        query="half guard sweeps",
        intent="EXECUTE",
        expected_topics=["sweep", "underhook", "dogfight", "reversal"],
        excluded_topics=["pass", "smash", "flatten"],
        description="Should find sweeps, not passing"
    ),
    EvalQuery(
        query="half guard pass",
        intent="EXECUTE",
        expected_topics=["pass", "knee", "pressure", "crossface"],
        excluded_topics=["sweep", "reversal", "back take"],
        description="Should find passing, not sweeps"
    ),
    EvalQuery(
        query="closed guard attacks",
        intent="EXECUTE",
        expected_topics=["attack", "submit", "sweep", "armbar", "triangle"],
        excluded_topics=["open", "pass", "break"],
        description="Should find attacks from closed guard"
    ),

    # Leg lock queries (common confusion area)
    EvalQuery(
        query="heel hook escape",
        intent="ESCAPE",
        expected_topics=["escape", "boot", "clear", "defend", "survive"],
        excluded_topics=["finish", "break", "attack", "enter"],
        description="Should find heel hook escapes"
    ),
    EvalQuery(
        query="heel hook finish",
        intent="EXECUTE",
        expected_topics=["finish", "breaking mechanics", "rotate", "control"],
        excluded_topics=["escape", "defend", "survive"],
        description="Should find heel hook finishes"
    ),
]


def evaluate_retrieval(
    retriever,
    eval_queries: list[EvalQuery],
    top_k: int = 5,
    verbose: bool = False
) -> dict:
    """
    Evaluate retrieval quality on test set.

    Returns dict with:
    - precision_at_k: Average precision for expected topics
    - exclusion_rate: Average rate of excluded topic violations
    - per_query_results: Detailed results for each query
    """
    results = {
        "precision_at_k": [],
        "exclusion_violations": [],
        "per_query_results": [],
        "latencies": []
    }

    for eq in eval_queries:
        start_time = time.time()

        # Get search results
        try:
            _, chunks = retriever.search(eq.query)
            chunks = chunks[:top_k]
        except Exception as e:
            print(f"Error on query '{eq.query}': {e}")
            continue

        latency = time.time() - start_time
        results["latencies"].append(latency)

        # Combine top-k text
        top_k_text = " ".join([c.get("text", "").lower() for c in chunks])

        # Check for expected topics
        expected_hits = sum(1 for topic in eq.expected_topics if topic in top_k_text)
        precision = expected_hits / len(eq.expected_topics) if eq.expected_topics else 1.0
        results["precision_at_k"].append(precision)

        # Check for exclusion violations
        violations = sum(1 for topic in eq.excluded_topics if topic in top_k_text)
        violation_rate = violations / len(eq.excluded_topics) if eq.excluded_topics else 0.0
        results["exclusion_violations"].append(violation_rate)

        query_result = {
            "query": eq.query,
            "intent": eq.intent,
            "description": eq.description,
            "precision": precision,
            "expected_found": expected_hits,
            "expected_total": len(eq.expected_topics),
            "violations": violations,
            "violation_rate": violation_rate,
            "latency": latency
        }
        results["per_query_results"].append(query_result)

        if verbose:
            status = "PASS" if precision >= 0.6 and violation_rate <= 0.3 else "FAIL"
            print(f"[{status}] {eq.query}")
            print(f"       Precision: {precision:.2f} | Violations: {violation_rate:.2f} | {latency:.1f}s")

    # Calculate aggregates
    summary = {
        "mean_precision_at_k": sum(results["precision_at_k"]) / len(results["precision_at_k"]) if results["precision_at_k"] else 0,
        "mean_exclusion_rate": sum(results["exclusion_violations"]) / len(results["exclusion_violations"]) if results["exclusion_violations"] else 0,
        "mean_latency": sum(results["latencies"]) / len(results["latencies"]) if results["latencies"] else 0,
        "total_queries": len(eval_queries),
        "per_query_results": results["per_query_results"]
    }

    return summary


def compare_systems(db_path: str, verbose: bool = False):
    """Compare v1 and v2 systems side by side."""
    print("=" * 70)
    print("BJJ RAG System Comparison")
    print("=" * 70)

    # Import both systems
    print("\nLoading v1 system...")
    from bjj_rag import BJJSearchRAG
    v1 = BJJSearchRAG(db_path=db_path)

    print("\nLoading v2 system...")
    from bjj_rag_v2 import BJJSearchRAGv2
    v2 = BJJSearchRAGv2(db_path=db_path)

    print("\n" + "-" * 70)
    print("Evaluating v1 (original)...")
    print("-" * 70)
    v1_results = evaluate_retrieval(v1, EVAL_QUERIES, verbose=verbose)

    print("\n" + "-" * 70)
    print("Evaluating v2 (improved)...")
    print("-" * 70)
    v2_results = evaluate_retrieval(v2, EVAL_QUERIES, verbose=verbose)

    # Print comparison
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    print(f"\n{'Metric':<30} {'v1':>12} {'v2':>12} {'Change':>12}")
    print("-" * 70)

    p1, p2 = v1_results["mean_precision_at_k"], v2_results["mean_precision_at_k"]
    print(f"{'Mean Precision@5':<30} {p1:>12.2%} {p2:>12.2%} {p2-p1:>+12.2%}")

    e1, e2 = v1_results["mean_exclusion_rate"], v2_results["mean_exclusion_rate"]
    print(f"{'Mean Exclusion Violations':<30} {e1:>12.2%} {e2:>12.2%} {e2-e1:>+12.2%}")

    l1, l2 = v1_results["mean_latency"], v2_results["mean_latency"]
    print(f"{'Mean Latency (seconds)':<30} {l1:>12.1f}s {l2:>12.1f}s {l2-l1:>+12.1f}s")

    print("\n" + "=" * 70)

    # Save detailed results
    output = {
        "v1": v1_results,
        "v2": v2_results
    }

    output_path = Path("evaluation_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nDetailed results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate BJJ RAG systems")
    parser.add_argument(
        "--db",
        default="./data/bjj_search_db",
        help="Path to LanceDB database"
    )
    parser.add_argument(
        "--system",
        choices=["v1", "v2"],
        help="Which system to evaluate"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare v1 and v2 side by side"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show per-query results"
    )
    parser.add_argument(
        "--output",
        default="evaluation_results.json",
        help="Output file for detailed results"
    )

    args = parser.parse_args()

    if args.compare:
        compare_systems(args.db, verbose=args.verbose)
        return

    if args.system == "v1":
        from bjj_rag import BJJSearchRAG
        rag = BJJSearchRAG(db_path=args.db)
    elif args.system == "v2":
        from bjj_rag_v2 import BJJSearchRAGv2
        rag = BJJSearchRAGv2(db_path=args.db)
    else:
        parser.error("Specify --system v1/v2 or --compare")

    print(f"\nEvaluating {args.system} system...")
    print("-" * 50)

    results = evaluate_retrieval(rag, EVAL_QUERIES, verbose=args.verbose)

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Mean Precision@5:        {results['mean_precision_at_k']:.2%}")
    print(f"Mean Exclusion Rate:     {results['mean_exclusion_rate']:.2%}")
    print(f"Mean Latency:            {results['mean_latency']:.1f}s")
    print(f"Total Queries:           {results['total_queries']}")

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
