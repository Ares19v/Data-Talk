"""
ingest/chunkers.py
──────────────────
Four chunking strategies applied to MSMARCO-XI records:

1. FixedSizeChunker      — fixed token window + overlap (baseline recall)
2. SemanticSentenceChunker — NLTK sentence split, merged by cosine sim boundary
3. PassageAwareChunker   — uses the dataset's own passage arrays (ground-truth)
4. HierarchicalChunker   — parent (full answer) + child (sentence) with parent_id

All return a list of Chunk dataclasses with unified metadata.
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np
import nltk

logger = logging.getLogger(__name__)

# ── ensure NLTK tokenizer data is available ───────────────────────────────────
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


# ── Chunk dataclass ───────────────────────────────────────────────────────────

@dataclass
class Chunk:
    text: str
    strategy: str                    # which chunker produced this
    query_id: int = -1
    source_lang: str = "en"
    query_type: str = "unknown"
    is_selected: int = 0             # 1 if this is a ground-truth passage
    passage_idx: int = -1
    parent_id: str | None = None     # used by hierarchical strategy
    chunk_id: str = ""               # unique identifier set by indexer
    metadata: dict = field(default_factory=dict)


# ── Utility helpers ───────────────────────────────────────────────────────────

def _approximate_token_count(text: str) -> int:
    """Very fast approximate token count (whitespace split * 1.3 factor)."""
    return max(1, int(len(text.split()) * 1.3))


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using NLTK."""
    try:
        sents = nltk.sent_tokenize(text)
    except Exception:
        # fallback: split on period/newline
        sents = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sents if s.strip()]


# ── Strategy 1: Fixed-size with overlap ──────────────────────────────────────

class FixedSizeChunker:
    """
    Splits text into windows of `chunk_tokens` approximate tokens with
    `overlap_tokens` overlap between consecutive chunks.
    """
    def __init__(self, chunk_tokens: int = 256, overlap_tokens: int = 64):
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, record: dict) -> List[Chunk]:
        text = record.get("Eng_Answer") or record.get("Answer") or ""
        if not text:
            return []

        words = text.split()
        # approximate: 1.3 words ≈ 1 token
        win = max(1, int(self.chunk_tokens / 1.3))
        step = max(1, int((self.chunk_tokens - self.overlap_tokens) / 1.3))

        chunks = []
        i = 0
        while i < len(words):
            window = words[i: i + win]
            chunk_text = " ".join(window)
            if chunk_text.strip():
                chunks.append(Chunk(
                    text=chunk_text,
                    strategy="fixed_size",
                    query_id=record.get("query_id", -1),
                    source_lang=record.get("source_lang", "en"),
                    query_type=record.get("query_type", "unknown"),
                    metadata={"window_start": i, "window_end": i + len(window)},
                ))
            i += step
        return chunks


# ── Strategy 2: Sentence-boundary semantic chunking ───────────────────────────

class SemanticSentenceChunker:
    """
    Splits text into sentences, then greedily merges consecutive sentences
    until the merged block exceeds `max_tokens`. A new chunk starts when
    the running token count would exceed the limit, ensuring each chunk
    stays semantically coherent at sentence boundaries.
    """
    def __init__(self, max_tokens: int = 192, min_tokens: int = 30):
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens

    def chunk(self, record: dict) -> List[Chunk]:
        text = record.get("Eng_Answer") or record.get("Answer") or ""
        if not text:
            return []

        sentences = _split_sentences(text)
        if not sentences:
            return []

        chunks: List[Chunk] = []
        current_sents: List[str] = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = _approximate_token_count(sent)
            if current_tokens + sent_tokens > self.max_tokens and current_sents:
                chunk_text = " ".join(current_sents)
                if _approximate_token_count(chunk_text) >= self.min_tokens:
                    chunks.append(Chunk(
                        text=chunk_text,
                        strategy="semantic_sentence",
                        query_id=record.get("query_id", -1),
                        source_lang=record.get("source_lang", "en"),
                        query_type=record.get("query_type", "unknown"),
                        metadata={"sentence_count": len(current_sents)},
                    ))
                current_sents = [sent]
                current_tokens = sent_tokens
            else:
                current_sents.append(sent)
                current_tokens += sent_tokens

        # flush remainder
        if current_sents:
            chunk_text = " ".join(current_sents)
            if chunk_text.strip():
                chunks.append(Chunk(
                    text=chunk_text,
                    strategy="semantic_sentence",
                    query_id=record.get("query_id", -1),
                    source_lang=record.get("source_lang", "en"),
                    query_type=record.get("query_type", "unknown"),
                    metadata={"sentence_count": len(current_sents)},
                ))
        return chunks


# ── Strategy 3: Passage-aware chunking ───────────────────────────────────────

class PassageAwareChunker:
    """
    Uses the dataset's built-in passages structure. Each English passage in
    `passages.English_passages` becomes its own chunk, tagged with is_selected
    and passage index. This gives the model access to the original retrieved
    passages, including the ground-truth selected ones.
    """
    def chunk(self, record: dict) -> List[Chunk]:
        passages_obj = record.get("passages") or {}
        english_passages = passages_obj.get("English_passages") or []
        is_selected_flags = passages_obj.get("is_selected") or []

        chunks = []
        for idx, passage in enumerate(english_passages):
            if not passage or not passage.strip():
                continue
            selected = int(is_selected_flags[idx]) if idx < len(is_selected_flags) else 0
            chunks.append(Chunk(
                text=passage.strip(),
                strategy="passage_aware",
                query_id=record.get("query_id", -1),
                source_lang=record.get("source_lang", "en"),
                query_type=record.get("query_type", "unknown"),
                is_selected=selected,
                passage_idx=idx,
                metadata={"is_ground_truth": selected == 1},
            ))
        return chunks


# ── Strategy 4: Hierarchical parent-child chunking ────────────────────────────

class HierarchicalChunker:
    """
    Creates two levels:
    - Parent chunk: the full English answer (for long-answer synthesis)
    - Child chunks: individual sentences of the answer, each carrying a
      parent_id so the retriever can fetch the parent context when a
      child is matched.
    """
    def chunk(self, record: dict) -> List[Chunk]:
        answer = record.get("Eng_Answer") or record.get("Answer") or ""
        if not answer or not answer.strip():
            return []

        q_id = record.get("query_id", -1)
        parent_id = f"hier_parent_{q_id}"

        # Parent
        parent_chunk = Chunk(
            text=answer.strip(),
            strategy="hierarchical_parent",
            query_id=q_id,
            source_lang=record.get("source_lang", "en"),
            query_type=record.get("query_type", "unknown"),
            chunk_id=parent_id,
            metadata={"level": "parent"},
        )

        # Children (sentences)
        sentences = _split_sentences(answer)
        child_chunks = []
        for i, sent in enumerate(sentences):
            if sent.strip():
                child_chunks.append(Chunk(
                    text=sent.strip(),
                    strategy="hierarchical_child",
                    query_id=q_id,
                    source_lang=record.get("source_lang", "en"),
                    query_type=record.get("query_type", "unknown"),
                    parent_id=parent_id,
                    metadata={"level": "child", "sentence_index": i},
                ))

        return [parent_chunk] + child_chunks


# ── Multi-strategy dispatcher ─────────────────────────────────────────────────

class MultiStrategyChunker:
    """
    Applies all four chunking strategies to a record and returns the
    combined list of chunks.
    """
    def __init__(self):
        self.strategies = [
            FixedSizeChunker(chunk_tokens=256, overlap_tokens=64),
            SemanticSentenceChunker(max_tokens=192, min_tokens=30),
            PassageAwareChunker(),
            HierarchicalChunker(),
        ]

    def chunk(self, record: dict) -> List[Chunk]:
        all_chunks: List[Chunk] = []
        for strategy in self.strategies:
            try:
                all_chunks.extend(strategy.chunk(record))
            except Exception as e:
                logger.warning(f"Chunker {strategy.__class__.__name__} failed: {e}")
        return all_chunks
