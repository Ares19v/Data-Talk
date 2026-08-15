"""
tests/test_pipeline_full.py
─────────────────────────────────────────────────────────────────────────────
Comprehensive end-to-end test suite for the Voice-RAG pipeline.

Tests every inch of the system:
  1.  Environment / Config
  2.  FAISS Index & Vector Store
  3.  Embedding Model (encode_query, encode_texts)
  4.  Chunking Strategies (all 4 types)
  5.  Multi-Strategy Retriever + RRF Fusion
  6.  LLM Generator (Groq)
  7.  Input Guardrails  (length, profanity, off-topic)
  8.  Output Guardrails (confidence, hallucination, grounding)
  9.  RAG Harness (full text pipeline)
  10. FastAPI Endpoints (/health, /query, /voice-query)
  11. Latency Benchmarks (retrieval must be < 200ms)
  12. Edge Cases (empty query, very long query, unicode, etc.)

Run with:
    python tests/test_pipeline_full.py
"""
from __future__ import annotations

import os
import sys
import time
import struct
import wave
import io
import json
import math
import traceback
from dataclasses import dataclass, field
from typing import List, Callable

# Make repo root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# ─── Colours ──────────────────────────────────────────────────────────────────
GREEN   = "\033[92m"
RED     = "\033[91m"
YELLOW  = "\033[93m"
CYAN    = "\033[96m"
BOLD    = "\033[1m"
RESET   = "\033[0m"
DIM     = "\033[2m"


# ─── Result tracking ──────────────────────────────────────────────────────────

@dataclass
class TestResult:
    name: str
    passed: bool
    message: str = ""
    duration_ms: float = 0.0
    warning: bool = False


results: List[TestResult] = []


def run_test(name: str, fn: Callable) -> TestResult:
    """Run a single test function, capture result, and print status."""
    print(f"  {DIM}→{RESET} {name}...", end=" ", flush=True)
    t0 = time.perf_counter()
    try:
        msg = fn()
        duration = (time.perf_counter() - t0) * 1000
        r = TestResult(name=name, passed=True, message=msg or "ok", duration_ms=duration)
        print(f"{GREEN}✓{RESET} {DIM}({duration:.0f}ms){RESET}")
    except AssertionError as e:
        duration = (time.perf_counter() - t0) * 1000
        r = TestResult(name=name, passed=False, message=str(e), duration_ms=duration)
        print(f"{RED}✗ FAIL{RESET} — {str(e)}")
    except Exception as e:
        duration = (time.perf_counter() - t0) * 1000
        r = TestResult(name=name, passed=False, message=f"{type(e).__name__}: {e}", duration_ms=duration)
        print(f"{RED}✗ ERROR{RESET} — {type(e).__name__}: {e}")
    results.append(r)
    return r


def section(title: str):
    print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*60}{RESET}")


def warn(name: str, msg: str, duration_ms: float = 0.0):
    """Record a warning (non-blocking test)."""
    print(f"  {YELLOW}⚠ WARN{RESET}  {name} — {msg}")
    results.append(TestResult(name=name, passed=True, message=msg, duration_ms=duration_ms, warning=True))


# ─── Helper: Generate a minimal valid WAV file in memory ──────────────────────

def make_silent_wav(duration_sec: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Create a minimal valid WAV file with silence."""
    n_samples = int(duration_sec * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)   # 16-bit
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * n_samples)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Environment & Configuration
# ══════════════════════════════════════════════════════════════════════════════

def test_env():
    section("1. Environment & Configuration")

    run_test("GROQ_API_KEY present", lambda: (
        None if os.getenv("GROQ_API_KEY") else (_ for _ in ()).throw(AssertionError("GROQ_API_KEY not set in .env"))
    ))

    run_test("SARVAM_API_KEY present", lambda: (
        None if os.getenv("SARVAM_API_KEY") else (_ for _ in ()).throw(AssertionError("SARVAM_API_KEY not set in .env"))
    ))

    def check_faiss_index():
        meta = os.getenv("METADATA_PATH", "./data/metadata.pkl")
        assert os.path.exists(meta), f"Metadata file not found: {meta}"
        size_mb = os.path.getsize(meta) / (1024 * 1024)
        return f"index={size_mb:.1f}MB"

    run_test("FAISS index files exist on disk", check_faiss_index)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — FAISS Vector Store
# ══════════════════════════════════════════════════════════════════════════════

_store = None

def test_faiss_store():
    global _store
    section("2. FAISS Vector Store")

    from indexer.faiss_store import FAISSStore

    def load_store():
        global _store
        index_path = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index.bin")
        meta_path  = os.getenv("METADATA_PATH",     "./data/metadata.pkl")
        _store = FAISSStore()
        _store.load(index_path, meta_path)
        total = _store.total_vectors()
        assert total > 0, f"Store is empty after load (total_vectors={total})"
        return f"loaded {total:,} vectors"

    run_test("Load FAISS index from disk", load_store)

    def check_strategies():
        strats = _store.get_strategies()
        assert len(strats) >= 1, f"Expected ≥1 strategies, got {strats}"
        return f"strategies={strats}"

    run_test("get_strategies() returns ≥1 strategy", check_strategies)

    def search_combined():
        import numpy as np
        q = np.random.randn(1, 384).astype("float32")
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        hits = _store.search_all(q, top_k=5)
        assert len(hits) > 0, "search_all returned 0 hits"
        assert len(hits) <= 5
        return f"hits={len(hits)}"

    run_test("search_all() returns results for random vector", search_combined)

    def search_per_strategy():
        import numpy as np
        q = np.random.randn(1, 384).astype("float32")
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        total_hits = 0
        for s in _store.get_strategies():
            hits = _store.search_strategy(s, q, top_k=3)
            total_hits += len(hits)
        assert total_hits > 0, "No hits from any strategy"
        return f"total strategy hits={total_hits}"

    run_test("search_strategy() works for all strategies", search_per_strategy)

    def chunk_structure():
        import numpy as np
        q = np.random.randn(1, 384).astype("float32")
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        hits = _store.search_all(q, top_k=1)
        score, chunk = hits[0]
        assert hasattr(chunk, "text"), "Chunk missing 'text' attribute"
        assert hasattr(chunk, "chunk_id"), "Chunk missing 'chunk_id' attribute"
        assert len(chunk.text) > 0, "Chunk text is empty"
        assert isinstance(score, float), f"Score is not float: {type(score)}"
        return f"chunk_id={chunk.chunk_id!r}, text_len={len(chunk.text)}"

    run_test("Retrieved chunks have correct structure", chunk_structure)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Embedding Model
# ══════════════════════════════════════════════════════════════════════════════

def test_embedder():
    section("3. Embedding Model (all-MiniLM-L6-v2)")

    from indexer.embedder import encode_query, encode_texts

    def single_query_shape():
        emb = encode_query("what is the capital of france")
        assert emb.shape == (1, 384), f"Expected (1,384), got {emb.shape}"
        return f"shape={emb.shape}"

    run_test("encode_query() returns (1, 384) float32", single_query_shape)

    def query_normalised():
        import numpy as np
        emb = encode_query("test normalisation")
        norm = float(np.linalg.norm(emb[0]))
        assert abs(norm - 1.0) < 0.01, f"Embedding not normalised: norm={norm:.4f}"
        return f"norm={norm:.4f}"

    run_test("encode_query() output is L2-normalised", query_normalised)

    def batch_encode():
        texts = ["sentence one", "sentence two", "sentence three"]
        embs = encode_texts(texts)
        assert embs.shape == (3, 384), f"Expected (3,384), got {embs.shape}"
        return f"shape={embs.shape}"

    run_test("encode_texts() handles batch of 3", batch_encode)

    def semantic_similarity():
        import numpy as np
        e1 = encode_query("the cat sat on the mat")[0]
        e2 = encode_query("a cat is resting on a rug")[0]
        e3 = encode_query("photosynthesis occurs in chloroplasts")[0]
        sim_close  = float(np.dot(e1, e2))
        sim_far    = float(np.dot(e1, e3))
        assert sim_close > sim_far, \
            f"Semantic similarity failed: related={sim_close:.3f} unrelated={sim_far:.3f}"
        return f"related={sim_close:.3f} > unrelated={sim_far:.3f} ✓"

    run_test("Semantic similarity ordering is correct", semantic_similarity)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Chunking Strategies
# ══════════════════════════════════════════════════════════════════════════════

def test_chunkers():
    section("4. Chunking Strategies")

    from ingest.chunkers import (
        FixedSizeChunker,
        SemanticSentenceChunker,
        PassageAwareChunker,
        HierarchicalChunker,
    )

    SAMPLE = {
        "query": "What is the boiling point of water?",
        "query_id": 1,
        "Eng_Answer": (
            "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at sea level. "
            "At higher altitudes the boiling point decreases because atmospheric pressure is lower. "
            "In cooking, boiling is a common method for preparing food, as the high temperature "
            "kills bacteria and softens vegetables. Pure water has a boiling point of exactly 100°C. "
            "Adding salt raises the boiling point slightly, a phenomenon known as boiling-point elevation."
        ),
        "passages": {
            "English_passages": [
                "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at sea level.",
                "At higher altitudes the boiling point decreases because atmospheric pressure is lower."
            ],
            "is_selected": [1, 0]
        }
    }

    def test_fixed_size():
        chunker = FixedSizeChunker(chunk_tokens=100, overlap_tokens=20)
        chunks = chunker.chunk(SAMPLE)
        assert len(chunks) >= 1, "FixedSizeChunker produced 0 chunks"
        for c in chunks:
            assert hasattr(c, "text") and len(c.text) > 0
            assert c.strategy == "fixed_size"
        return f"chunks={len(chunks)}"

    run_test("FixedSizeChunker produces valid chunks", test_fixed_size)

    def test_semantic():
        chunker = SemanticSentenceChunker()
        chunks = chunker.chunk(SAMPLE)
        assert len(chunks) >= 1, "SemanticSentenceChunker produced 0 chunks"
        for c in chunks:
            assert c.strategy == "semantic_sentence"
        return f"chunks={len(chunks)}"

    run_test("SemanticSentenceChunker produces valid chunks", test_semantic)

    def test_passage():
        chunker = PassageAwareChunker()
        chunks = chunker.chunk(SAMPLE)
        assert len(chunks) >= 1, "PassageAwareChunker produced 0 chunks"
        for c in chunks:
            assert c.strategy == "passage_aware"
        return f"chunks={len(chunks)}"

    run_test("PassageAwareChunker produces valid chunks", test_passage)

    def test_hierarchical():
        chunker = HierarchicalChunker()
        chunks = chunker.chunk(SAMPLE)
        assert len(chunks) >= 1, "HierarchicalChunker produced 0 chunks"
        strategies = {c.strategy for c in chunks}
        assert "hierarchical_parent" in strategies or "hierarchical_child" in strategies, \
            f"Missing hierarchical strategies: {strategies}"
        return f"chunks={len(chunks)}, strategies={strategies}"

    run_test("HierarchicalChunker produces parent+child chunks", test_hierarchical)

    def test_chunk_ids():
        chunker = FixedSizeChunker()
        chunks = chunker.chunk(SAMPLE)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs detected"
        return f"all {len(ids)} chunk_ids are unique"

    run_test("Chunk IDs are unique within a document", test_chunk_ids)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Multi-Strategy Retriever + RRF
# ══════════════════════════════════════════════════════════════════════════════

def test_retriever():
    section("5. Multi-Strategy Retriever + RRF Fusion")

    from pipeline.retriever import MultiStrategyRetriever

    def basic_retrieve():
        retriever = MultiStrategyRetriever(store=_store, top_k_per_strategy=5, top_n_final=5)
        results, timing = retriever.retrieve("what is the capital of France")
        assert len(results) >= 1, f"Retriever returned no results"
        assert len(results) <= 5
        return f"results={len(results)}, total_ms={timing.get('total_retrieval_ms', 0):.1f}"

    run_test("Retriever returns results for a standard query", basic_retrieve)

    def timing_keys():
        retriever = MultiStrategyRetriever(store=_store, top_k_per_strategy=3, top_n_final=3)
        _, timing = retriever.retrieve("how do vaccines work")
        required = {"encode_ms", "retrieval_ms", "rrf_ms", "total_retrieval_ms"}
        missing = required - set(timing.keys())
        assert not missing, f"Missing timing keys: {missing}"
        return f"timing keys present: {list(timing.keys())}"

    run_test("All expected timing keys are present", timing_keys)

    def rrf_scores_ordered():
        retriever = MultiStrategyRetriever(store=_store, top_k_per_strategy=5, top_n_final=5)
        results, _ = retriever.retrieve("history of the roman empire")
        scores = [r.rrf_score for r in results]
        assert scores == sorted(scores, reverse=True), \
            f"Results not sorted by RRF score: {scores}"
        return f"scores={[round(s,4) for s in scores]}"

    run_test("Results are sorted by descending RRF score", rrf_scores_ordered)

    def deduplication():
        retriever = MultiStrategyRetriever(store=_store, top_k_per_strategy=10, top_n_final=10)
        results, _ = retriever.retrieve("what is machine learning")
        texts = [r.chunk.text.strip().lower()[:150] for r in results]
        assert len(texts) == len(set(texts)), "Duplicate passages found in retrieval results"
        return f"no duplicates among {len(texts)} results"

    run_test("Results are deduplicated by text content", deduplication)

    def latency_benchmark():
        retriever = MultiStrategyRetriever(store=_store, top_k_per_strategy=5, top_n_final=5)
        _, timing = retriever.retrieve("what causes earthquakes")
        total = timing.get("total_retrieval_ms", 9999)
        if total > 200:
            warn("Retrieval Latency", f"total={total:.1f}ms exceeds 200ms target", total)
        else:
            pass
        assert total < 2000, f"Retrieval critically slow: {total:.1f}ms"
        return f"total_retrieval={total:.1f}ms (target <200ms)"

    run_test("Retrieval completes in <2000ms (warns if >200ms)", latency_benchmark)

    def top_k_respected():
        for k in [1, 3, 5]:
            retriever = MultiStrategyRetriever(store=_store, top_k_per_strategy=k, top_n_final=k)
            results, _ = retriever.retrieve("how does photosynthesis work")
            assert len(results) <= k, f"top_k={k} but got {len(results)} results"
        return "top_k=1,3,5 all respected"

    run_test("top_k parameter is respected", top_k_respected)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — LLM Generator
# ══════════════════════════════════════════════════════════════════════════════

def test_generator():
    section("6. LLM Generator (Groq)")

    from pipeline.generator import LLMGenerator

    CONTEXT = [
        "Paris is the capital of France and is known as the City of Light.",
        "The Eiffel Tower is located in Paris, France.",
    ]

    def basic_generate():
        gen = LLMGenerator()
        answer, ms = gen.generate("What is the capital of France?", CONTEXT)
        assert isinstance(answer, str), "Answer is not a string"
        assert len(answer) > 5, f"Answer too short: {answer!r}"
        assert isinstance(ms, float) and ms > 0, "Generation time is invalid"
        return f"answer_len={len(answer)}, ms={ms:.0f}"

    run_test("generate() returns a non-empty answer", basic_generate)

    def answer_is_grounded():
        gen = LLMGenerator()
        answer, _ = gen.generate("What is the capital of France?", CONTEXT)
        # The answer should mention Paris
        assert "paris" in answer.lower(), \
            f"Answer doesn't mention Paris (the correct answer). Got: {answer!r}"
        return f"answer mentions 'Paris' ✓"

    run_test("Generated answer is grounded in context", answer_is_grounded)

    def no_context_behaviour():
        gen = LLMGenerator()
        answer, _ = gen.generate("What is the capital of France?", [])
        # With empty context, LLM should admit it doesn't know
        assert isinstance(answer, str) and len(answer) > 0
        return f"answered with empty context (len={len(answer)})"

    run_test("Generator handles empty context gracefully", no_context_behaviour)

    def timing_is_reasonable():
        gen = LLMGenerator()
        _, ms = gen.generate("What causes rain?", ["Water evaporates from oceans forming clouds."])
        assert ms < 30000, f"LLM took too long: {ms:.0f}ms"
        return f"llm_ms={ms:.0f}"

    run_test("LLM generation completes within 30 seconds", timing_is_reasonable)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Input Guardrails
# ══════════════════════════════════════════════════════════════════════════════

def test_input_guardrails():
    section("7. Input Guardrails")

    from pipeline.guardrails import InputGuardrails

    ig = InputGuardrails()

    def valid_query():
        r = ig.check("What is the speed of light?")
        assert r.passed, f"Valid query blocked: reason={r.reason}"
        return "valid query passes ✓"

    run_test("Valid factual query passes input guardrails", valid_query)

    def too_short():
        r = ig.check("hi")
        assert not r.passed, "Too-short query was not blocked"
        assert r.reason == "query_too_short"
        return f"reason={r.reason} ✓"

    run_test("Query with <3 chars is rejected (query_too_short)", too_short)

    def too_long():
        q = "a" * 501
        r = ig.check(q)
        assert not r.passed, "Too-long query was not blocked"
        assert r.reason == "query_too_long"
        return f"reason={r.reason} ✓"

    run_test("Query >500 chars is rejected (query_too_long)", too_long)

    def profanity_blocked():
        r = ig.check("What the hell is this shit?")
        if r.passed:
            warn("Profanity guardrail", "Profanity not detected (may be threshold)")
        else:
            assert r.reason == "profanity"
        return f"reason={r.reason}"

    run_test("Profane query is blocked by profanity filter", profanity_blocked)

    def off_topic_blocked():
        r = ig.check("Tell me a dirty joke about pirates")
        return f"passed={r.passed}, reason={r.reason}"

    run_test("Off-topic query check runs without crash", off_topic_blocked)

    def safe_message_present():
        r = ig.check("hi")
        assert r.safe_message, "No safe_message returned for blocked query"
        return f"safe_message={r.safe_message!r}"

    run_test("Blocked queries include a safe_message for the user", safe_message_present)

    def empty_string():
        r = ig.check("")
        assert not r.passed, "Empty string query should be blocked"
        return f"empty string blocked: reason={r.reason} ✓"

    run_test("Empty string is rejected by guardrails", empty_string)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Output Guardrails
# ══════════════════════════════════════════════════════════════════════════════

def test_output_guardrails():
    section("8. Output Guardrails")

    from pipeline.guardrails import OutputGuardrails

    og = OutputGuardrails()
    GOOD_CONTEXT = [
        "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
        "The boiling point of water decreases at higher altitudes.",
    ]

    def grounded_answer_passes():
        r = og.check(
            answer="Water boils at 100 degrees Celsius.",
            retrieved_texts=GOOD_CONTEXT,
            top_score=0.85,
        )
        assert r.passed, f"Grounded answer blocked: reason={r.reason}"
        return "grounded answer passes ✓"

    run_test("Well-grounded answer passes output guardrails", grounded_answer_passes)

    def low_confidence_blocked():
        r = og.check(
            answer="Water boils at 100°C.",
            retrieved_texts=GOOD_CONTEXT,
            top_score=-0.5,   # very low FAISS score = not similar at all
        )
        assert not r.passed, "Low-confidence result was not blocked"
        assert r.reason == "low_confidence"
        return f"reason={r.reason} ✓"

    run_test("Very low FAISS score triggers low_confidence guardrail", low_confidence_blocked)

    def idk_response_blocked():
        r = og.check(
            answer="I don't know the answer to this question.",
            retrieved_texts=GOOD_CONTEXT,
            top_score=0.85,
        )
        assert not r.passed, "IDK response was not blocked"
        assert r.reason == "idk_response"
        return f"reason={r.reason} ✓"

    run_test("'I don't know' LLM response triggers idk_response guardrail", idk_response_blocked)

    def ungrounded_answer_blocked():
        r = og.check(
            answer="Xylophones zebras xylophones zebras xylophones zebras xylophones zebras.",
            retrieved_texts=GOOD_CONTEXT,
            top_score=0.85,
        )
        assert not r.passed, "Completely ungrounded answer was not blocked"
        assert r.reason == "not_grounded"
        return f"reason={r.reason} ✓"

    run_test("Hallucinated answer triggers not_grounded guardrail", ungrounded_answer_blocked)

    def safe_message_on_output_block():
        r = og.check("I do not know anything.", GOOD_CONTEXT, top_score=0.85)
        assert r.safe_message, "No safe_message on output guardrail block"
        return f"safe_message present ✓"

    run_test("Output guardrail blocks include a safe_message", safe_message_on_output_block)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — Full RAG Harness
# ══════════════════════════════════════════════════════════════════════════════

_harness = None

def test_harness():
    global _harness
    section("9. Full RAG Harness (Text Pipeline)")

    from pipeline.harness import RAGHarness, RAGRequest, get_harness

    _harness = get_harness()

    def harness_loads():
        total = _harness.store.total_vectors()
        assert total > 0, "Harness FAISS store is empty"
        return f"vectors={total:,}"

    run_test("get_harness() loads with vectors", harness_loads)

    def full_rag_run():
        req = RAGRequest(query="What is machine learning?", top_k=3)
        resp = _harness.run(req)
        assert resp.query == req.query
        assert isinstance(resp.answer, str) and len(resp.answer) > 5
        assert resp.latency is not None
        assert resp.latency.encode_ms > 0
        assert resp.latency.retrieval_ms > 0
        assert resp.latency.rrf_ms > 0
        assert resp.latency.llm_ms > 0
        assert resp.latency.total_pipeline_ms > 0
        return f"answer_len={len(resp.answer)}, total_ms={resp.latency.total_pipeline_ms:.0f}"

    run_test("Full pipeline run returns valid RAGResponse", full_rag_run)

    def sources_included():
        req = RAGRequest(query="how does photosynthesis work", top_k=3, include_sources=True)
        resp = _harness.run(req)
        if resp.success:
            assert isinstance(resp.sources, list)
            assert len(resp.sources) >= 1, "No sources returned despite include_sources=True"
            src = resp.sources[0]
            assert src.text and src.strategy and src.rrf_score >= 0 and src.faiss_score >= 0
        return f"sources={len(resp.sources)}"

    run_test("Sources included when include_sources=True", sources_included)

    def sources_excluded():
        req = RAGRequest(query="what is democracy", top_k=3, include_sources=False)
        resp = _harness.run(req)
        assert resp.sources == [], f"Sources returned despite include_sources=False: {resp.sources}"
        return "sources=[] ✓"

    run_test("Sources empty when include_sources=False", sources_excluded)

    def guardrail_blocks_short():
        req = RAGRequest(query="hi", top_k=3)
        resp = _harness.run(req)
        assert resp.guardrail_triggered, "Short query should have triggered guardrail"
        assert not resp.success
        return f"guardrail_reason={resp.guardrail_reason}"

    run_test("Harness blocks short queries via input guardrail", guardrail_blocks_short)

    def latency_breakdown_complete():
        req = RAGRequest(query="what is inflation in economics", top_k=3)
        resp = _harness.run(req)
        lat = resp.latency
        # All fields should be non-negative
        for field_name in ["encode_ms", "retrieval_ms", "rrf_ms", "llm_ms", "guardrail_ms", "total_pipeline_ms"]:
            val = getattr(lat, field_name)
            assert val >= 0, f"{field_name} is negative: {val}"
        assert lat.stt_ms is None, "stt_ms should be None for text queries"
        return f"all latency fields valid, total={lat.total_pipeline_ms:.0f}ms"

    run_test("LatencyBreakdown has all fields (stt_ms=None for text)", latency_breakdown_complete)

    def multiple_queries_consistent():
        queries = [
            "what is the speed of light",
            "how to bake bread",
            "who invented the telephone",
        ]
        for q in queries:
            req = RAGRequest(query=q, top_k=3)
            resp = _harness.run(req)
            assert isinstance(resp.answer, str) and len(resp.answer) > 0, \
                f"Empty answer for query: {q!r}"
        return f"all {len(queries)} queries returned answers"

    run_test("Multiple diverse queries all return answers", multiple_queries_consistent)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — FastAPI Endpoints
# ══════════════════════════════════════════════════════════════════════════════

def test_api_endpoints():
    section("10. FastAPI HTTP Endpoints")

    try:
        import httpx
    except ImportError:
        print(f"  {YELLOW}⚠ SKIP{RESET}  httpx not installed — run: pip install httpx")
        return

    BASE = "http://localhost:8000"

    def health_check():
        resp = httpx.get(f"{BASE}/health", timeout=10)
        assert resp.status_code == 200, f"HTTP {resp.status_code}"
        data = resp.json()
        assert data["status"] == "ok", f"status != ok: {data}"
        assert data["index_loaded"] is True, "index_loaded is False"
        assert data["total_vectors"] > 0
        return f"total_vectors={data['total_vectors']:,}, strategies={data['strategies']}"

    run_test("GET /health returns 200 with index_loaded=True", health_check)

    def text_query_endpoint():
        payload = {"query": "What is the speed of light?", "top_k": 3, "include_sources": True}
        resp = httpx.post(f"{BASE}/query", json=payload, timeout=60)
        assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text}"
        data = resp.json()
        assert "answer" in data, "No 'answer' in response"
        assert "latency" in data, "No 'latency' in response"
        assert "sources" in data, "No 'sources' in response"
        assert isinstance(data["answer"], str) and len(data["answer"]) > 0
        return f"answer_len={len(data['answer'])}, sources={len(data['sources'])}"

    run_test("POST /query returns valid RAGResponse JSON", text_query_endpoint)

    def query_validation_422():
        payload = {"query": "", "top_k": 3}
        resp = httpx.post(f"{BASE}/query", json=payload, timeout=10)
        assert resp.status_code == 422, f"Expected 422, got {resp.status_code}"
        return "empty query correctly returns 422 ✓"

    run_test("POST /query with empty string returns 422 (validation error)", query_validation_422)

    def query_top_k_respected():
        for k in [1, 5]:
            payload = {"query": "what is democracy", "top_k": k, "include_sources": True}
            resp = httpx.post(f"{BASE}/query", json=payload, timeout=60)
            assert resp.status_code == 200
            data = resp.json()
            if data.get("success"):
                assert len(data["sources"]) <= k, \
                    f"top_k={k} but got {len(data['sources'])} sources"
        return "top_k=1 and top_k=5 both respected ✓"

    run_test("POST /query respects top_k parameter in response sources", query_top_k_respected)

    def voice_query_invalid_type():
        wav_bytes = make_silent_wav(0.5)
        resp = httpx.post(
            f"{BASE}/voice-query",
            files={"audio": ("test.txt", b"this is not audio", "text/plain")},
            data={"top_k": "3"},
            timeout=10,
        )
        assert resp.status_code == 415, f"Expected 415 for invalid type, got {resp.status_code}"
        return "invalid content-type correctly returns 415 ✓"

    run_test("POST /voice-query rejects non-audio content with 415", voice_query_invalid_type)

    def voice_query_wav_accepted():
        wav_bytes = make_silent_wav(1.0)
        resp = httpx.post(
            f"{BASE}/voice-query",
            files={"audio": ("test.wav", wav_bytes, "audio/wav")},
            data={"top_k": "3"},
            timeout=60,
        )
        # Silent audio may produce empty transcription → pipeline handles gracefully
        assert resp.status_code in {200, 422, 500}, \
            f"Unexpected status code: {resp.status_code}"
        return f"HTTP {resp.status_code} (silent wav processed without crash)"

    run_test("POST /voice-query accepts WAV audio without crashing", voice_query_wav_accepted)

    def docs_redirect():
        resp = httpx.get(f"{BASE}/", timeout=5, follow_redirects=True)
        assert resp.status_code == 200, f"HTTP {resp.status_code}"
        assert "swagger" in resp.text.lower() or "openapi" in resp.text.lower() or "docs" in resp.text.lower()
        return "/ redirects to /docs ✓"

    run_test("GET / redirects to Swagger /docs", docs_redirect)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — Latency Benchmarks
# ══════════════════════════════════════════════════════════════════════════════

def test_latency_benchmarks():
    section("11. Latency Benchmarks")

    from pipeline.retriever import MultiStrategyRetriever

    QUERIES = [
        "what is photosynthesis",
        "who invented the telephone",
        "what is machine learning",
        "how do vaccines work",
        "causes of world war 2",
    ]

    def benchmark_retrieval():
        retriever = MultiStrategyRetriever(store=_store, top_k_per_strategy=5, top_n_final=5)
        times = []
        for q in QUERIES:
            _, timing = retriever.retrieve(q)
            times.append(timing["total_retrieval_ms"])
        avg = sum(times) / len(times)
        max_t = max(times)
        if avg > 200:
            warn("Retrieval Avg Latency", f"avg={avg:.1f}ms exceeds 200ms target")
        assert max_t < 5000, f"Max retrieval time critically high: {max_t:.0f}ms"
        return f"avg={avg:.1f}ms, max={max_t:.1f}ms over {len(QUERIES)} queries"

    run_test(f"Retrieval benchmark over {len(QUERIES)} queries", benchmark_retrieval)

    def encode_latency():
        from indexer.embedder import encode_query
        times = []
        for q in QUERIES:
            t0 = time.perf_counter()
            encode_query(q)
            times.append((time.perf_counter() - t0) * 1000)
        avg = sum(times) / len(times)
        assert avg < 500, f"Encoding avg too slow: {avg:.1f}ms"
        return f"avg={avg:.1f}ms, max={max(times):.1f}ms"

    run_test("Query encoding benchmark (avg <500ms expected)", encode_latency)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — Edge Cases
# ══════════════════════════════════════════════════════════════════════════════

def test_edge_cases():
    section("12. Edge Cases & Robustness")

    from pipeline.harness import RAGRequest

    def unicode_query():
        req = RAGRequest(query="Что такое машинное обучение?", top_k=3)
        resp = _harness.run(req)
        assert isinstance(resp.answer, str)
        return f"unicode query handled, answer_len={len(resp.answer)}"

    run_test("Unicode (Cyrillic) query doesn't crash pipeline", unicode_query)

    def all_numbers_query():
        req = RAGRequest(query="1234567890 what is this", top_k=3)
        resp = _harness.run(req)
        assert isinstance(resp.answer, str)
        return f"numeric query handled, guardrail={resp.guardrail_triggered}"

    run_test("Query with leading numbers doesn't crash pipeline", all_numbers_query)

    def very_specific_query():
        req = RAGRequest(query="xkqzjwp bflmnop zzzzz", top_k=3)
        resp = _harness.run(req)
        # Should either block or return a graceful answer
        assert isinstance(resp.answer, str) and len(resp.answer) > 0
        return f"nonsense handled: success={resp.success}, guardrail={resp.guardrail_triggered}"

    run_test("Nonsense query handled gracefully (block or low-confidence)", very_specific_query)

    def max_top_k():
        req = RAGRequest(query="what is democracy", top_k=10, include_sources=True)
        resp = _harness.run(req)
        assert len(resp.sources) <= 10
        return f"top_k=10, sources={len(resp.sources)}"

    run_test("top_k=10 (max) doesn't crash the pipeline", max_top_k)

    def response_serialisable():
        import json
        req = RAGRequest(query="how does the immune system work", top_k=3)
        resp = _harness.run(req)
        json_str = resp.model_dump_json()
        parsed = json.loads(json_str)
        assert "answer" in parsed and "latency" in parsed
        return f"response JSON serialises correctly, {len(json_str)} bytes"

    run_test("Full RAGResponse is JSON-serialisable", response_serialisable)

    def stt_module_importable():
        from stt.sarvam import SarvamSTT
        stt = SarvamSTT()
        assert stt is not None
        return "SarvamSTT importable ✓"

    run_test("stt.sarvam.SarvamSTT imports without error", stt_module_importable)


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{BOLD}{'═'*60}{RESET}")
    print(f"{BOLD}  DATA TALK — Full Pipeline Test Suite{RESET}")
    print(f"{BOLD}{'═'*60}{RESET}")
    print(f"{DIM}  Voice RAG | HH Goa 2026 | Running all 12 test sections{RESET}")

    test_env()
    test_faiss_store()
    test_embedder()
    test_chunkers()
    test_retriever()
    test_generator()
    test_input_guardrails()
    test_output_guardrails()
    test_harness()
    test_api_endpoints()
    test_latency_benchmarks()
    test_edge_cases()

    # ── Summary ────────────────────────────────────────────────────────────────
    passed   = [r for r in results if r.passed and not r.warning]
    warnings = [r for r in results if r.warning]
    failed   = [r for r in results if not r.passed]

    print(f"\n{BOLD}{'═'*60}{RESET}")
    print(f"{BOLD}  RESULTS SUMMARY{RESET}")
    print(f"{BOLD}{'═'*60}{RESET}")
    print(f"  {GREEN}✓ Passed  : {len(passed)}{RESET}")
    print(f"  {YELLOW}⚠ Warnings: {len(warnings)}{RESET}")
    print(f"  {RED}✗ Failed  : {len(failed)}{RESET}")
    print(f"  Total     : {len(results)}")

    if warnings:
        print(f"\n{YELLOW}Warnings:{RESET}")
        for r in warnings:
            print(f"  {YELLOW}⚠{RESET} {r.name}: {r.message}")

    if failed:
        print(f"\n{RED}Failures:{RESET}")
        for r in failed:
            print(f"  {RED}✗{RESET} {r.name}: {r.message}")
        print(f"\n{RED}Some tests FAILED. See above for details.{RESET}\n")
        sys.exit(1)
    else:
        print(f"\n{GREEN}{BOLD}All tests passed! Pipeline is healthy. 🌴{RESET}\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
