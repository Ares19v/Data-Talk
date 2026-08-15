"""
pipeline/retriever.py
─────────────────────
Multi-strategy retrieval with Reciprocal Rank Fusion (RRF).

For each query:
1. Encode the query → 384-dim embedding
2. Search each strategy's FAISS partition (top-k each)
3. Also search the combined index (top-k)
4. Merge all result lists using RRF
5. Deduplicate by text content
6. Return top-N final chunks with scores
"""
from __future__ import annotations

import time
import logging
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from ingest.chunkers import Chunk
from indexer.embedder import encode_query
from indexer.faiss_store import FAISSStore

logger = logging.getLogger(__name__)

# RRF constant (k=60 is standard; lower k boosts top-ranked docs more)
RRF_K = 60


class RetrievalResult:
    """Holds a chunk and its final RRF score for downstream use."""
    __slots__ = ("chunk", "rrf_score", "faiss_score", "strategy")

    def __init__(self, chunk: Chunk, rrf_score: float, faiss_score: float, strategy: str):
        self.chunk = chunk
        self.rrf_score = rrf_score
        self.faiss_score = faiss_score
        self.strategy = strategy


class MultiStrategyRetriever:
    """
    Retrieves relevant chunks using all available FAISS index partitions,
    fuses their rankings with RRF, and returns deduplicated top results.
    """

    def __init__(
        self,
        store: FAISSStore,
        top_k_per_strategy: int = 5,
        top_n_final: int = 5,
    ):
        self.store = store
        self.top_k_per_strategy = top_k_per_strategy
        self.top_n_final = top_n_final

    def retrieve(self, query: str) -> Tuple[List[RetrievalResult], Dict[str, float]]:
        """
        Retrieve top-N chunks for a query string.

        Returns:
            results: List of RetrievalResult sorted best-first
            timing:  Dict of stage timings in milliseconds
        """
        timing: Dict[str, float] = {}

        # Step 1: Encode query
        t0 = time.perf_counter()
        q_emb = encode_query(query)
        timing["encode_ms"] = (time.perf_counter() - t0) * 1000

        # Step 2: Search all strategy partitions
        t1 = time.perf_counter()
        strategy_results: Dict[str, List[Tuple[float, Chunk]]] = {}

        for strategy in self.store.get_strategies():
            hits = self.store.search_strategy(strategy, q_emb, top_k=self.top_k_per_strategy)
            if hits:
                strategy_results[strategy] = hits

        # Also search the combined index
        combined_hits = self.store.search_all(q_emb, top_k=self.top_k_per_strategy * 2)
        if combined_hits:
            strategy_results["combined"] = combined_hits

        timing["retrieval_ms"] = (time.perf_counter() - t1) * 1000

        # Step 3: RRF fusion
        t2 = time.perf_counter()
        fused = _reciprocal_rank_fusion(strategy_results, k=RRF_K)
        timing["rrf_ms"] = (time.perf_counter() - t2) * 1000

        # Step 4: Deduplicate + build final result list
        seen_texts: set = set()
        results: List[RetrievalResult] = []

        for chunk_id, (rrf_score, faiss_score, chunk, strategy) in sorted(
            fused.items(), key=lambda x: x[1][0], reverse=True
        ):
            # Deduplicate by text prefix
            dedup_key = chunk.text.strip().lower()[:150]
            if dedup_key in seen_texts:
                continue
            seen_texts.add(dedup_key)

            results.append(RetrievalResult(
                chunk=chunk,
                rrf_score=rrf_score,
                faiss_score=faiss_score,
                strategy=strategy,
            ))
            if len(results) >= self.top_n_final:
                break

        timing["total_retrieval_ms"] = timing["encode_ms"] + timing["retrieval_ms"] + timing["rrf_ms"]

        logger.debug(
            f"Retrieval: encode={timing['encode_ms']:.1f}ms "
            f"faiss={timing['retrieval_ms']:.1f}ms "
            f"rrf={timing['rrf_ms']:.1f}ms "
            f"total={timing['total_retrieval_ms']:.1f}ms "
            f"results={len(results)}"
        )

        return results, timing


def _reciprocal_rank_fusion(
    strategy_results: Dict[str, List[Tuple[float, Chunk]]],
    k: int = 60,
) -> Dict[str, Tuple[float, float, Chunk, str]]:
    """
    Merge ranked result lists from multiple strategies using RRF.

    RRF score for document d:
        RRF(d) = Σ_strategy  1 / (k + rank_of_d_in_strategy)

    Returns:
        dict mapping chunk_id → (rrf_score, best_faiss_score, chunk, strategy)
    """
    scores: Dict[str, float] = defaultdict(float)
    meta: Dict[str, Tuple[float, Chunk, str]] = {}

    for strategy, hits in strategy_results.items():
        for rank, (faiss_score, chunk) in enumerate(hits):
            # Use text prefix as dedup key for RRF
            cid = chunk.chunk_id or chunk.text.strip().lower()[:100]
            rrf_contribution = 1.0 / (k + rank + 1)
            scores[cid] += rrf_contribution

            # Keep best FAISS score across strategies
            if cid not in meta or faiss_score > meta[cid][0]:
                meta[cid] = (faiss_score, chunk, strategy)

    # Merge
    result = {}
    for cid, rrf_score in scores.items():
        if cid in meta:
            faiss_score, chunk, strategy = meta[cid]
            result[cid] = (rrf_score, faiss_score, chunk, strategy)

    return result
