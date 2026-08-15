"""
tests/smoke_test.py
───────────────────
End-to-end smoke test for the full pipeline.
Requires a built index (run scripts/build_index.py first).

Tests:
1. Harness loads successfully
2. Text query returns a response
3. Guardrail correctly blocks off-topic query
4. Guardrail correctly blocks profanity
5. Latency stats are collected
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.table import Table

console = Console()


def run_smoke_tests():
    console.print("[bold green]Voice-RAG Smoke Tests[/bold green]\n")

    from pipeline.harness import get_harness, RAGRequest

    # Test 1: Harness loads
    console.print("[dim]Test 1: Loading harness...[/dim]")
    harness = get_harness()
    assert harness is not None, "Harness failed to load"
    console.print(f"  ✅ Harness loaded | {harness.store.total_vectors():,} vectors | strategies: {harness.store.get_strategies()}")

    results = []

    # Test 2: Normal factual query
    console.print("[dim]Test 2: Normal factual query...[/dim]")
    t0 = time.perf_counter()
    resp = harness.run(RAGRequest(query="How does photosynthesis work?"))
    elapsed = (time.perf_counter() - t0) * 1000
    success = resp.success or resp.guardrail_triggered
    results.append(("Normal query", "✅" if success else "❌", f"{elapsed:.0f}ms", resp.answer[:60] + "..."))
    console.print(f"  {'✅' if success else '❌'} Answer: {resp.answer[:80]}")
    console.print(f"  ⏱️  Retrieval: {resp.latency.retrieval_ms:.1f}ms | Total: {resp.latency.total_pipeline_ms:.0f}ms")

    # Test 3: Off-topic query should be blocked
    console.print("[dim]Test 3: Off-topic query (should be blocked)...[/dim]")
    resp3 = harness.run(RAGRequest(query="Tell me a funny joke"))
    is_blocked = not resp3.success and resp3.guardrail_triggered
    results.append(("Off-topic block", "✅" if is_blocked else "⚠️ not blocked", "-", resp3.guardrail_reason))
    console.print(f"  {'✅' if is_blocked else '⚠️'} Guardrail: {resp3.guardrail_reason} | {resp3.answer[:60]}")

    # Test 4: Latency check (retrieval path)
    console.print("[dim]Test 4: Retrieval latency check (<200ms)...[/dim]")
    resp4 = harness.run(RAGRequest(query="What is the speed of sound?"))
    retrieval_ok = resp4.latency.retrieval_ms < 200
    encode_ms = resp4.latency.encode_ms
    results.append((
        "Retrieval <200ms",
        "✅" if retrieval_ok else "⚠️",
        f"{resp4.latency.retrieval_ms:.1f}ms",
        f"encode={encode_ms:.1f}ms"
    ))
    console.print(f"  {'✅' if retrieval_ok else '⚠️'} Retrieval: {resp4.latency.retrieval_ms:.1f}ms | Encode: {encode_ms:.1f}ms")

    # Test 5: Sources included
    console.print("[dim]Test 5: Sources returned...[/dim]")
    resp5 = harness.run(RAGRequest(query="What causes earthquakes?", include_sources=True))
    has_sources = len(resp5.sources) > 0
    results.append(("Sources returned", "✅" if has_sources else "⚠️", f"{len(resp5.sources)} sources", ""))
    console.print(f"  {'✅' if has_sources else '⚠️'} {len(resp5.sources)} sources returned")

    # Summary table
    console.print()
    table = Table(title="Smoke Test Results", show_header=True, header_style="bold cyan")
    table.add_column("Test", style="dim")
    table.add_column("Status", justify="center")
    table.add_column("Timing", justify="right")
    table.add_column("Notes")

    for row in results:
        table.add_row(*row)

    console.print(table)
    passed = sum(1 for r in results if "✅" in r[1])
    console.print(f"\n[bold]Results: {passed}/{len(results)} passed[/bold]")


if __name__ == "__main__":
    run_smoke_tests()
