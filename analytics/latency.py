"""
analytics/latency.py
────────────────────
Latency benchmark harness.

Measures P50 / P70 / P100 (max) latency for each pipeline stage
across N sample queries (default: 100).

Excludes STT network latency (measured separately as it depends on
Sarvam API response time, not the retrieval pipeline).

Usage:
    python analytics/latency.py --n 100
    python analytics/latency.py --n 50 --include-stt
"""
from __future__ import annotations

import os
import sys
import argparse
import logging
import time

# Make root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.WARNING)  # suppress info during benchmark

import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()

# Sample benchmark queries (representative MSMARCO-style questions)
BENCHMARK_QUERIES = [
    "what is the speed of light",
    "how does photosynthesis work",
    "who invented the telephone",
    "what causes earthquakes",
    "how does the immune system fight viruses",
    "what is the capital of Japan",
    "explain how vaccines work",
    "what is inflation in economics",
    "history of the Roman Empire",
    "how do computers process data",
    "what is the boiling point of water",
    "what causes thunder and lightning",
    "how does memory work in humans",
    "what is DNA and its function",
    "explain the water cycle",
    "who wrote Romeo and Juliet",
    "what is climate change",
    "how are rainbows formed",
    "what is the function of the heart",
    "how do airplanes stay in the air",
    "what is machine learning",
    "explain supply and demand",
    "what is quantum mechanics",
    "how does digestion work",
    "what is the greenhouse effect",
    "how are mountains formed",
    "what causes ocean tides",
    "how does solar energy work",
    "what is the theory of evolution",
    "explain how the internet works",
    "what is the function of white blood cells",
    "how does sound travel",
    "what is the speed of sound",
    "how are clouds formed",
    "what is gravity",
    "how does a battery work",
    "what is the ozone layer",
    "how does the brain store memories",
    "what is nuclear energy",
    "how do plants reproduce",
    "what is the Big Bang theory",
    "how does electricity work",
    "what is metabolism",
    "how do earthquakes cause tsunamis",
    "what is the function of the liver",
    "how does wifi work",
    "what is entropy",
    "how do black holes form",
    "what is photon",
    "how does a refrigerator work",
    "what is the Pythagorean theorem",
    "how does blood clotting work",
    "what are tectonic plates",
    "how do languages evolve",
    "what is dark matter",
    "how does anesthesia work",
    "what is the placebo effect",
    "how are fossils formed",
    "what is osmosis",
    "how do birds migrate",
    "what is the function of the kidney",
    "how does sound cause hearing",
    "what is fermentation",
    "how do antibiotics work",
    "what is erosion",
    "how does the stock market work",
    "what is cryptography",
    "how do plants make food",
    "what is the function of red blood cells",
    "how does the eye see color",
    "what is genetic mutation",
    "how do volcanoes erupt",
    "what is the difference between weather and climate",
    "how does GPS work",
    "what is the function of the pancreas",
    "how do hormones work",
    "what is the nitrogen cycle",
    "how does a nuclear reactor work",
    "what is the doppler effect",
    "how do muscles contract",
    "what is the function of the spleen",
    "how are diamonds formed",
    "what is the higgs boson",
    "how does radioactivity work",
    "what is the carbon cycle",
    "how do neurons communicate",
    "what is absolute zero",
    "how does the moon affect tides",
    "what is natural selection",
    "how do viruses replicate",
    "what is the function of the thyroid gland",
    "how does sonar work",
    "what is static electricity",
    "how do earthquakes measure on the richter scale",
    "what is the function of the adrenal gland",
    "how does photovoltaic cells work",
    "what is symbiosis in biology",
    "how does chemotherapy work",
    "what is the function of the cerebellum",
    "how are antibodies produced",
    "what is conservation of energy",
]


def run_benchmark(n: int = 100, warmup: int = 5, verbose: bool = False) -> dict:
    """
    Run the retrieval pipeline (excluding LLM) on N queries.
    Returns per-stage latency arrays.
    """
    from pipeline.harness import get_harness

    harness = get_harness()
    queries = (BENCHMARK_QUERIES * ((n // len(BENCHMARK_QUERIES)) + 1))[:n]

    # Warmup
    console.print(f"[dim]Warming up ({warmup} queries)...[/dim]")
    for q in queries[:warmup]:
        harness.retriever.retrieve(q)

    # Benchmark
    stage_timings = {
        "encode_ms": [],
        "retrieval_ms": [],
        "rrf_ms": [],
        "total_retrieval_ms": [],
    }

    console.print(f"[dim]Benchmarking retrieval path ({n} queries)...[/dim]")
    t_all = time.perf_counter()

    for i, q in enumerate(queries):
        _, timings = harness.retriever.retrieve(q)
        for key in stage_timings:
            if key in timings:
                stage_timings[key].append(timings[key])
        if verbose and i % 10 == 0:
            console.print(f"  [{i}/{n}] {q[:50]}...")

    wall_ms = (time.perf_counter() - t_all) * 1000
    console.print(f"[green]✓[/green] Benchmarked {n} queries in {wall_ms:.0f}ms ({n / (wall_ms/1000):.0f} QPS)\n")

    return stage_timings


def percentile(data: list, pct: float) -> float:
    if not data:
        return 0.0
    return float(np.percentile(data, pct))


def print_latency_report(stage_timings: dict, n: int):
    table = Table(title=f"Retrieval Latency Report (N={n})", show_header=True, header_style="bold cyan")
    table.add_column("Stage", style="dim", width=24)
    table.add_column("P50 (ms)", justify="right")
    table.add_column("P70 (ms)", justify="right")
    table.add_column("P90 (ms)", justify="right")
    table.add_column("P100/Max (ms)", justify="right")
    table.add_column("Mean (ms)", justify="right")

    stage_labels = {
        "encode_ms": "Query Encode",
        "retrieval_ms": "FAISS Search",
        "rrf_ms": "RRF Fusion",
        "total_retrieval_ms": "Total Retrieval ✅",
    }

    for key, label in stage_labels.items():
        data = stage_timings.get(key, [])
        if not data:
            continue
        style = "bold green" if key == "total_retrieval_ms" else ""
        table.add_row(
            label,
            f"[{style}]{percentile(data, 50):.1f}[/{style}]",
            f"[{style}]{percentile(data, 70):.1f}[/{style}]",
            f"[{style}]{percentile(data, 90):.1f}[/{style}]",
            f"[{style}]{max(data):.1f}[/{style}]",
            f"[{style}]{sum(data)/len(data):.1f}[/{style}]",
        )

    console.print(table)
    console.print()
    console.print(
        "[bold]Note:[/bold] LLM generation adds ~200–400ms (Groq) on top of retrieval. "
        "STT adds ~300–800ms (network). The 200ms target covers the retrieval path only."
    )


def main():
    parser = argparse.ArgumentParser(description="Voice-RAG Latency Benchmark")
    parser.add_argument("--n", type=int, default=100, help="Number of queries to benchmark")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup queries before timing")
    parser.add_argument("--verbose", action="store_true", help="Show per-query output")
    args = parser.parse_args()

    console.print(f"[bold green]Voice-RAG Latency Benchmark[/bold green]")
    console.print(f"  N={args.n} queries, warmup={args.warmup}")
    console.print()

    timings = run_benchmark(n=args.n, warmup=args.warmup, verbose=args.verbose)
    print_latency_report(timings, args.n)


if __name__ == "__main__":
    main()
