"""
indexer/embedder.py
───────────────────
Singleton sentence-transformer encoder using all-MiniLM-L6-v2.
- 384-dim embeddings
- ~5ms per query on CPU (warmed up)
- Thread-safe encode method
"""
from __future__ import annotations

import logging
import numpy as np
from functools import lru_cache
from typing import List, Union

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


@lru_cache(maxsize=1)
def get_encoder() -> SentenceTransformer:
    """Load and warm up the encoder (cached singleton)."""
    logger.info(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    # Warmup to pre-JIT compile
    model.encode(["warmup sentence"], show_progress_bar=False, normalize_embeddings=True)
    logger.info("Embedding model ready.")
    return model


def encode_texts(texts: List[str], batch_size: int = 256) -> np.ndarray:
    """
    Encode a list of strings into L2-normalized float32 embeddings.
    Returns shape (N, 384).
    """
    model = get_encoder()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 500,
        normalize_embeddings=True,   # cosine similarity = dot product
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def encode_query(query: str) -> np.ndarray:
    """
    Encode a single query string. Returns shape (1, 384).
    Optimized path — bypasses batch overhead.
    """
    model = get_encoder()
    emb = model.encode(
        [query],
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return emb.astype(np.float32)
