"""
pipeline/harness.py
───────────────────
RAG pipeline orchestration harness.

Responsibilities:
- Structured Pydantic input/output models
- Retry logic (tenacity) for LLM and STT calls
- Per-stage timing collection
- Error recovery + graceful degradation
- Wires together: InputGuardrail → Retriever → LLM → OutputGuardrail
"""
from __future__ import annotations

import os
import time
import logging
from typing import List, Optional

from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from indexer.faiss_store import FAISSStore
from pipeline.guardrails import InputGuardrails, OutputGuardrails
from pipeline.retriever import MultiStrategyRetriever, RetrievalResult
from pipeline.generator import LLMGenerator

logger = logging.getLogger(__name__)


# ── Pydantic models ───────────────────────────────────────────────────────────

class RAGRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="User's question text")
    language_hint: Optional[str] = Field(None, description="ISO language code hint e.g. 'en', 'hi'")
    top_k: int = Field(5, ge=1, le=20, description="Number of context chunks to retrieve")
    include_sources: bool = Field(True, description="Include source passages in response")


class SourcePassage(BaseModel):
    text: str
    strategy: str
    rrf_score: float
    faiss_score: float
    query_id: int
    is_selected: int


class LatencyBreakdown(BaseModel):
    stt_ms: Optional[float] = None
    encode_ms: float
    retrieval_ms: float
    rrf_ms: float
    llm_ms: float
    guardrail_ms: float
    total_pipeline_ms: float


class RAGResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourcePassage] = []
    latency: LatencyBreakdown
    guardrail_triggered: bool = False
    guardrail_reason: str = ""
    success: bool = True


# ── Harness ───────────────────────────────────────────────────────────────────

class RAGHarness:
    """
    Main orchestrator for the Voice-RAG pipeline.
    """

    def __init__(
        self,
        store: FAISSStore,
        top_k: int = 5,
        max_llm_retries: int = 3,
    ):
        self.store = store
        self.top_k = top_k
        self.max_llm_retries = max_llm_retries

        self.input_guardrails = InputGuardrails()
        self.output_guardrails = OutputGuardrails()
        self.retriever = MultiStrategyRetriever(
            store=store,
            top_k_per_strategy=top_k,
            top_n_final=top_k,
        )
        self.generator = LLMGenerator()

        logger.info("RAGHarness initialized")

    def run(self, request: RAGRequest) -> RAGResponse:
        """
        Execute the full RAG pipeline for a text query.
        """
        total_start = time.perf_counter()
        timings: dict = {}

        # ── Step 1: Input Guardrails ─────────────────────────────────────
        t = time.perf_counter()
        input_check = self.input_guardrails.check(request.query)
        timings["guardrail_input_ms"] = (time.perf_counter() - t) * 1000

        if not input_check.passed:
            logger.info(f"Input guardrail blocked: reason={input_check.reason}")
            return _blocked_response(
                query=request.query,
                answer=input_check.safe_message,
                reason=input_check.reason,
                timings=timings,
                total_start=total_start,
            )

        # ── Step 2: Retrieval (encode + FAISS + RRF) ─────────────────────
        try:
            results, retrieval_timings = self.retriever.retrieve(request.query)
            timings.update(retrieval_timings)
        except Exception as e:
            logger.error(f"Retrieval failed: {e}", exc_info=True)
            return _error_response(request.query, "Retrieval failed. Please try again.", total_start, timings)

        if not results:
            return _blocked_response(
                query=request.query,
                answer="I couldn't find relevant information in my knowledge base.",
                reason="no_results",
                timings=timings,
                total_start=total_start,
            )

        top_score = results[0].faiss_score
        retrieved_texts = [r.chunk.text for r in results]

        # ── Step 3: LLM Generation (with retries) ────────────────────────
        t = time.perf_counter()
        try:
            answer, llm_ms = self._generate_with_retry(request.query, retrieved_texts)
        except Exception as e:
            logger.error(f"LLM generation failed after retries: {e}", exc_info=True)
            return _error_response(request.query, "Answer generation failed. Please try again.", total_start, timings)
        timings["llm_ms"] = llm_ms

        # ── Step 4: Output Guardrails ────────────────────────────────────
        t = time.perf_counter()
        output_check = self.output_guardrails.check(
            answer=answer,
            retrieved_texts=retrieved_texts,
            top_score=top_score,
        )
        timings["guardrail_output_ms"] = (time.perf_counter() - t) * 1000

        if not output_check.passed:
            logger.info(f"Output guardrail blocked: reason={output_check.reason}")
            return _blocked_response(
                query=request.query,
                answer=output_check.safe_message,
                reason=output_check.reason,
                timings=timings,
                total_start=total_start,
            )

        # ── Step 5: Assemble Response ─────────────────────────────────────
        total_ms = (time.perf_counter() - total_start) * 1000
        guardrail_ms = timings.get("guardrail_input_ms", 0) + timings.get("guardrail_output_ms", 0)

        sources = []
        if request.include_sources:
            for r in results:
                sources.append(SourcePassage(
                    text=r.chunk.text[:500],
                    strategy=r.strategy,
                    rrf_score=round(r.rrf_score, 4),
                    faiss_score=round(r.faiss_score, 4),
                    query_id=r.chunk.query_id,
                    is_selected=r.chunk.is_selected,
                ))

        return RAGResponse(
            query=request.query,
            answer=answer,
            sources=sources,
            latency=LatencyBreakdown(
                encode_ms=round(timings.get("encode_ms", 0), 2),
                retrieval_ms=round(timings.get("retrieval_ms", 0), 2),
                rrf_ms=round(timings.get("rrf_ms", 0), 2),
                llm_ms=round(llm_ms, 2),
                guardrail_ms=round(guardrail_ms, 2),
                total_pipeline_ms=round(total_ms, 2),
            ),
            success=True,
        )

    def run_with_stt(
        self,
        audio_bytes: bytes,
        language_hint: Optional[str] = None,
        **kwargs,
    ) -> RAGResponse:
        """
        Full voice pipeline: STT → query → RAG pipeline.
        Returns response with STT latency included in latency breakdown.
        """
        from stt.sarvam import SarvamSTT
        stt = SarvamSTT()

        t = time.perf_counter()
        query = stt.transcribe_bytes(audio_bytes, language_code=language_hint)
        stt_ms = (time.perf_counter() - t) * 1000

        if not query.strip():
            return _error_response("", "Could not transcribe audio. Please speak clearly and try again.", time.perf_counter(), {})

        request = RAGRequest(query=query, language_hint=language_hint, **kwargs)
        response = self.run(request)
        response.query = query
        if response.latency:
            response.latency.stt_ms = round(stt_ms, 2)
            response.latency.total_pipeline_ms = round(
                response.latency.total_pipeline_ms + stt_ms, 2
            )
        return response

    # ── Internal helpers ──────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.05, min=0.05, max=1.0),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _generate_with_retry(self, query: str, retrieved_texts: list) -> tuple:
        return self.generator.generate(query, retrieved_texts)


# ── Helper constructors ───────────────────────────────────────────────────────

def _blank_latency(timings: dict, total_start: float) -> LatencyBreakdown:
    total_ms = (time.perf_counter() - total_start) * 1000
    return LatencyBreakdown(
        encode_ms=round(timings.get("encode_ms", 0), 2),
        retrieval_ms=round(timings.get("retrieval_ms", 0), 2),
        rrf_ms=round(timings.get("rrf_ms", 0), 2),
        llm_ms=0.0,
        guardrail_ms=round(timings.get("guardrail_input_ms", 0), 2),
        total_pipeline_ms=round(total_ms, 2),
    )


def _blocked_response(query, answer, reason, timings, total_start) -> RAGResponse:
    return RAGResponse(
        query=query,
        answer=answer,
        sources=[],
        latency=_blank_latency(timings, total_start),
        guardrail_triggered=True,
        guardrail_reason=reason,
        success=False,
    )


def _error_response(query, answer, total_start, timings) -> RAGResponse:
    return RAGResponse(
        query=query,
        answer=answer,
        sources=[],
        latency=_blank_latency(timings, total_start),
        success=False,
    )


# ── Singleton store loader ────────────────────────────────────────────────────

_harness_singleton: RAGHarness | None = None


def get_harness() -> RAGHarness:
    """Load (or return cached) the RAGHarness with a pre-built FAISS index."""
    global _harness_singleton
    if _harness_singleton is None:
        index_path = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index.bin")
        metadata_path = os.getenv("METADATA_PATH", "./data/metadata.pkl")

        store = FAISSStore()
        if os.path.exists(metadata_path):
            logger.info(f"Loading pre-built FAISS index from {metadata_path}")
            store.load(index_path, metadata_path)
        else:
            logger.warning(
                f"No pre-built index found at {metadata_path}. "
                "Run `python scripts/build_index.py` first, or the system will "
                "answer with no context."
            )

        _harness_singleton = RAGHarness(store=store)
    return _harness_singleton
