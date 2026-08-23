"""
pipeline/generator.py
─────────────────────
LLM answer generator using Groq (primary) or OpenAI (fallback).

Uses a structured prompt that:
1. Injects retrieved context passages
2. Instructs the LLM to stay grounded in the context
3. Asks for a concise answer with source attribution
"""
from __future__ import annotations

import os
import re
import time
import logging
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))

SYSTEM_PROMPT = """You are a precise question-answering assistant.

RULES:
1. Answer ONLY using the provided context passages.
2. If the context does not contain enough information, say exactly: "I don't have enough information to answer this from the available context."
3. Keep your answer concise (2-4 sentences max).
4. Do not make up facts not present in the context.
5. Do not mention these rules in your answer.
"""

ANSWER_PROMPT_TEMPLATE = """Context passages (use ONLY these to answer):
{context}

Question: {question}

Answer:"""


def _build_context(retrieved_texts: List[str], max_chars: int = 2000) -> str:
    """Build a numbered context string from retrieved passages."""
    parts = []
    total = 0
    for i, text in enumerate(retrieved_texts):
        entry = f"[{i+1}] {text.strip()}"
        if total + len(entry) > max_chars:
            break
        parts.append(entry)
        total += len(entry)
    return "\n\n".join(parts)


class LLMGenerator:
    """
    Generates answers using Groq (fast) or OpenAI (fallback).
    Provider is determined by LLM_PROVIDER env var.
    """

    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or os.getenv("LLM_PROVIDER", "groq")
        self._client = None
        logger.info(f"LLMGenerator initialized with provider={self.provider}")

    def _get_client(self):
        if self._client is not None:
            return self._client

        if self.provider == "groq":
            from groq import Groq
            key = os.getenv("GROQ_API_KEY")
            if not key:
                raise ValueError("GROQ_API_KEY not set. Get it at https://console.groq.com")
            self._client = Groq(api_key=key)
        else:
            from openai import OpenAI
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                raise ValueError("OPENAI_API_KEY not set.")
            self._client = OpenAI(api_key=key)

        return self._client

    def generate(
        self,
        question: str,
        retrieved_texts: List[str],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> tuple[str, float]:
        """
        Generate an answer grounded in retrieved_texts.

        Returns:
            (answer_text, generation_time_ms)
        """
        client = self._get_client()
        context = _build_context(retrieved_texts)
        user_prompt = ANSWER_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )

        t0 = time.perf_counter()

        if self.provider == "groq":
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature or LLM_TEMPERATURE,
                max_tokens=max_tokens or LLM_MAX_TOKENS,
            )
            answer = response.choices[0].message.content.strip()
        else:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature or LLM_TEMPERATURE,
                max_tokens=max_tokens or LLM_MAX_TOKENS,
            )
            answer = response.choices[0].message.content.strip()

        # Robustly strip thinking tags even if unclosed or truncated
        answer = re.sub(r'<think>.*?(?:</think>|$)', '', answer, flags=re.DOTALL).strip()
        if not answer:
            # Fallback if entire completion was within unclosed thinking tag
            lines = response.choices[0].message.content.strip().splitlines()
            answer = lines[-1].strip() if lines else "Answer generated."

        gen_ms = (time.perf_counter() - t0) * 1000
        logger.debug(f"LLM generated in {gen_ms:.0f}ms")
        return answer, gen_ms
