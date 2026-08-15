"""
indexer/faiss_store.py
──────────────────────
Builds and queries FAISS indices for the RAG pipeline.

Index strategy:
- If N > 10_000 vectors → IndexIVFFlat (nlist=256, nprobe=32) — fast ANN
- If N <= 10_000 vectors → IndexFlatIP — exact, still <5ms for small sets
- Embeddings are L2-normalized so IP == cosine similarity

Each strategy gets its own index partition stored in a dict:
  {strategy_name: (faiss_index, [Chunk])}
"""
from __future__ import annotations

import os
import pickle
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np

from ingest.chunkers import Chunk
from indexer.embedder import encode_texts, EMBEDDING_DIM

logger = logging.getLogger(__name__)

# Minimum vectors before switching to IVF (avoid training on tiny sets)
IVF_MIN_VECTORS = 10_000
IVF_NLIST = 256
IVF_NPROBE = 32


# ── Index store container ─────────────────────────────────────────────────────

class FAISSStore:
    """
    Maintains per-strategy FAISS indices. At query time, each strategy's
    index is searched independently and results are merged by the retriever.
    """

    def __init__(self):
        # strategy_name → (faiss.Index, list[Chunk])
        self._indices: Dict[str, Tuple[faiss.Index, List[Chunk]]] = {}
        # flat combined index for cross-strategy search
        self._all_index: faiss.Index | None = None
        self._all_chunks: List[Chunk] = []

    # ── Build ──────────────────────────────────────────────────────────────

    def build_from_chunks(self, chunks: List[Chunk], batch_size: int = 1024) -> None:
        """
        Build per-strategy indices from a flat list of Chunk objects.
        Also builds a combined 'all' index for cross-strategy queries.
        """
        # Group chunks by strategy
        by_strategy: Dict[str, List[Chunk]] = {}
        for chunk in chunks:
            by_strategy.setdefault(chunk.strategy, []).append(chunk)

        all_texts = []
        all_chunk_refs: List[Chunk] = []

        for strategy, strategy_chunks in by_strategy.items():
            logger.info(f"Building index for strategy={strategy}, n={len(strategy_chunks)}")
            texts = [c.text for c in strategy_chunks]
            embeddings = _encode_in_batches(texts, batch_size)

            index = _build_index(embeddings)
            self._indices[strategy] = (index, strategy_chunks)

            all_texts.extend(texts)
            all_chunk_refs.extend(strategy_chunks)

        # Combined index
        if all_chunk_refs:
            logger.info(f"Building combined index, n={len(all_chunk_refs)}")
            all_embeddings = _encode_in_batches(all_texts, batch_size)
            self._all_index = _build_index(all_embeddings)
            self._all_chunks = all_chunk_refs

    # ── Search ─────────────────────────────────────────────────────────────

    def search_strategy(
        self,
        strategy: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[float, Chunk]]:
        """
        Search a single strategy's index.
        Returns list of (score, Chunk) sorted best-first.
        Inner-product on L2-normalized vectors = cosine similarity.
        """
        if strategy not in self._indices:
            return []
        index, chunks = self._indices[strategy]
        if index.ntotal == 0:
            return []
        k = min(top_k, index.ntotal)
        scores, indices = index.search(query_embedding, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(chunks):
                results.append((float(score), chunks[idx]))
        return results

    def search_all(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> List[Tuple[float, Chunk]]:
        """Search the combined index across all strategies."""
        if self._all_index is None or self._all_index.ntotal == 0:
            return []
        k = min(top_k, self._all_index.ntotal)
        scores, indices = self._all_index.search(query_embedding, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self._all_chunks):
                results.append((float(score), self._all_chunks[idx]))
        return results

    def get_strategies(self) -> List[str]:
        return list(self._indices.keys())

    def total_vectors(self) -> int:
        return self._all_index.ntotal if self._all_index else 0

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, index_path: str, metadata_path: str) -> None:
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)

        # Save per-strategy indices
        strategy_index_data = {}
        for strategy, (index, chunks) in self._indices.items():
            vec = faiss.serialize_index(index)
            strategy_index_data[strategy] = {"index_bytes": vec, "chunks": chunks}

        # Save combined index
        combined = None
        if self._all_index:
            combined = faiss.serialize_index(self._all_index)

        with open(metadata_path, "wb") as f:
            pickle.dump({
                "strategy_data": strategy_index_data,
                "combined_bytes": combined,
                "all_chunks": self._all_chunks,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"Saved FAISS store → {metadata_path}")

    def load(self, index_path: str, metadata_path: str) -> None:
        with open(metadata_path, "rb") as f:
            data = pickle.load(f)

        self._indices = {}
        for strategy, sd in data["strategy_data"].items():
            index = faiss.deserialize_index(sd["index_bytes"])
            # set nprobe for IVF indices
            if hasattr(index, "nprobe"):
                index.nprobe = IVF_NPROBE
            self._indices[strategy] = (index, sd["chunks"])

        if data.get("combined_bytes") is not None:
            self._all_index = faiss.deserialize_index(data["combined_bytes"])
            if hasattr(self._all_index, "nprobe"):
                self._all_index.nprobe = IVF_NPROBE
        self._all_chunks = data.get("all_chunks", [])
        logger.info(f"Loaded FAISS store — {self.total_vectors()} total vectors")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _encode_in_batches(texts: List[str], batch_size: int) -> np.ndarray:
    t0 = time.perf_counter()
    embeddings = encode_texts(texts, batch_size=batch_size)
    elapsed = time.perf_counter() - t0
    logger.info(f"Encoded {len(texts)} texts in {elapsed:.2f}s ({len(texts)/elapsed:.0f} texts/s)")
    return embeddings


def _build_index(embeddings: np.ndarray) -> faiss.Index:
    n, d = embeddings.shape
    assert d == EMBEDDING_DIM, f"Expected dim={EMBEDDING_DIM}, got {d}"

    if n >= IVF_MIN_VECTORS:
        # IVFFlat: approximate, fast for large collections
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, IVF_NLIST, faiss.METRIC_INNER_PRODUCT)
        # Train on a sample or all vectors
        train_size = min(n, IVF_NLIST * 40)
        train_data = embeddings[:train_size]
        index.train(train_data)
        index.nprobe = IVF_NPROBE
        index.add(embeddings)
        logger.debug(f"Built IVFFlat index: n={n}, nlist={IVF_NLIST}, nprobe={IVF_NPROBE}")
    else:
        # FlatIP: exact search, fine for <10K vectors
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)
        logger.debug(f"Built FlatIP index: n={n}")

    return index
