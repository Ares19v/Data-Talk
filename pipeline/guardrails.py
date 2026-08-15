"""
pipeline/guardrails.py
──────────────────────
Input and output guardrails for the RAG pipeline.

Input guardrails:
  1. Profanity/unsafe content filter
  2. Off-topic detection via cosine distance to domain centroid
  3. Query length validation

Output guardrails:
  4. Grounding check — answer must have ROUGE-1 recall ≥ threshold vs retrieved context
  5. Confidence gate — if FAISS distance > threshold, refuse to answer
  6. Hallucination heuristic — detect "I don't know" responses when context exists
"""
from __future__ import annotations

import os
import logging
import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Lazy imports (avoid slowing startup) ─────────────────────────────────────
_profanity_filter = None
_rouge_scorer = None


def _get_profanity():
    global _profanity_filter
    if _profanity_filter is None:
        from better_profanity import profanity
        profanity.load_censor_words()
        _profanity_filter = profanity
    return _profanity_filter


def _get_rouge():
    global _rouge_scorer
    if _rouge_scorer is None:
        from rouge_score import rouge_scorer
        _rouge_scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=False)
    return _rouge_scorer


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class GuardrailResult:
    passed: bool
    reason: str = ""
    safe_message: str = ""


# ── Domain centroid (precomputed or lazily computed) ──────────────────────────

_domain_centroid: Optional[np.ndarray] = None

# Representative MSMARCO-style queries for centroid calculation
_DOMAIN_SEEDS = [
    "what is the capital of france",
    "how does photosynthesis work",
    "what causes earthquakes",
    "history of the roman empire",
    "how to treat a cold",
    "what is machine learning",
    "how does the immune system work",
    "who invented the telephone",
    "what is inflation",
    "how do vaccines work",
    "what is the speed of light",
    "how to bake bread",
    "what is democracy",
    "causes of world war 2",
    "how does a computer processor work",
]


def _get_domain_centroid() -> np.ndarray:
    global _domain_centroid
    if _domain_centroid is None:
        from indexer.embedder import encode_texts
        embeddings = encode_texts(_DOMAIN_SEEDS)
        _domain_centroid = embeddings.mean(axis=0)
        _domain_centroid /= np.linalg.norm(_domain_centroid) + 1e-10
    return _domain_centroid


# ── Input guardrails ──────────────────────────────────────────────────────────

class InputGuardrails:
    def __init__(
        self,
        off_topic_threshold: float | None = None,
        max_query_length: int = 500,
        min_query_length: int = 3,
    ):
        self.off_topic_threshold = float(
            off_topic_threshold or os.getenv("OFF_TOPIC_THRESHOLD", "0.15")
        )
        self.max_query_length = max_query_length
        self.min_query_length = min_query_length

    def check(self, query: str) -> GuardrailResult:
        """Run all input checks. Returns GuardrailResult."""

        # 1. Length check
        stripped = query.strip()
        if len(stripped) < self.min_query_length:
            return GuardrailResult(
                passed=False,
                reason="query_too_short",
                safe_message="Please ask a complete question.",
            )
        if len(stripped) > self.max_query_length:
            return GuardrailResult(
                passed=False,
                reason="query_too_long",
                safe_message="Your question is too long. Please keep it under 500 characters.",
            )

        # 2. Profanity / unsafe content
        pf = _get_profanity()
        if pf.contains_profanity(stripped):
            return GuardrailResult(
                passed=False,
                reason="profanity",
                safe_message="I'm not able to process that request. Please rephrase your question.",
            )

        # 3. Off-topic detection
        off_topic_result = self._check_off_topic(stripped)
        if not off_topic_result.passed:
            return off_topic_result

        return GuardrailResult(passed=True, reason="ok")

    def _check_off_topic(self, query: str) -> GuardrailResult:
        """
        Check if the query is semantically related to the MSMARCO domain
        by computing cosine similarity to the domain centroid.
        """
        try:
            from indexer.embedder import encode_query
            q_emb = encode_query(query)[0]
            centroid = _get_domain_centroid()
            # Both are L2-normalized so dot product = cosine similarity
            similarity = float(np.dot(q_emb, centroid))

            logger.debug(f"Off-topic similarity={similarity:.3f} threshold={self.off_topic_threshold}")

            if similarity < self.off_topic_threshold:
                return GuardrailResult(
                    passed=False,
                    reason="off_topic",
                    safe_message=(
                        "I can only answer factual questions from my knowledge base. "
                        "Please ask a different question."
                    ),
                )
        except Exception as e:
            logger.warning(f"Off-topic check failed (allowing through): {e}")

        return GuardrailResult(passed=True, reason="ok")


# ── Output guardrails ─────────────────────────────────────────────────────────

class OutputGuardrails:
    def __init__(
        self,
        rouge_threshold: float | None = None,
        distance_threshold: float | None = None,
    ):
        self.rouge_threshold = float(
            rouge_threshold or os.getenv("GROUNDING_ROUGE_THRESHOLD", "0.10")
        )
        self.distance_threshold = float(
            distance_threshold or os.getenv("CONFIDENCE_DISTANCE_THRESHOLD", "1.60")
        )
        # Patterns indicating the LLM admits it doesn't know
        self._idk_patterns = re.compile(
            r"(i don'?t know|i do not know|i am not sure|i cannot|"
            r"no information|not found|cannot find|unable to answer|"
            r"the (context|passage|document) does not|"
            r"not mentioned|no relevant)",
            re.IGNORECASE,
        )

    def check(
        self,
        answer: str,
        retrieved_texts: List[str],
        top_score: float,
    ) -> GuardrailResult:
        """
        Validate the generated answer against retrieved context.

        Args:
            answer: LLM-generated answer
            retrieved_texts: List of retrieved passage texts
            top_score: Top FAISS inner-product score (higher = more similar)
        """
        # 1. Confidence gate — if retrieved context is too dissimilar, bail
        # Note: With L2-normalized IP, score ≥ 0 means some similarity;
        # we use a low threshold to catch truly unrelated retrievals
        if top_score < (1.0 - self.distance_threshold / 2):
            return GuardrailResult(
                passed=False,
                reason="low_confidence",
                safe_message=(
                    "I couldn't find reliable information in my knowledge base "
                    "to answer your question."
                ),
            )

        # 2. Hallucination heuristic — LLM said it doesn't know
        if self._idk_patterns.search(answer):
            return GuardrailResult(
                passed=False,
                reason="idk_response",
                safe_message=(
                    "I found some context but couldn't formulate a confident answer. "
                    "Please try rephrasing your question."
                ),
            )

        # 3. Grounding check via ROUGE-1 recall
        if retrieved_texts:
            combined_context = " ".join(retrieved_texts[:5])
            scorer = _get_rouge()
            try:
                scores = scorer.score(answer.lower(), combined_context.lower())
                rouge1_recall = scores["rouge1"].recall
                logger.debug(f"ROUGE-1 recall={rouge1_recall:.3f} threshold={self.rouge_threshold}")
                if rouge1_recall < self.rouge_threshold:
                    return GuardrailResult(
                        passed=False,
                        reason="not_grounded",
                        safe_message=(
                            "My answer doesn't seem well-supported by the retrieved documents. "
                            "Please try a more specific question."
                        ),
                    )
            except Exception as e:
                logger.warning(f"ROUGE grounding check failed (skipping): {e}")

        return GuardrailResult(passed=True, reason="ok")
