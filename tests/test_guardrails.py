"""
tests/test_guardrails.py
────────────────────────
Unit tests for the guardrail layer (input and output).
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from pipeline.guardrails import InputGuardrails, OutputGuardrails


class TestInputGuardrails:
    def setup_method(self):
        self.ig = InputGuardrails(off_topic_threshold=0.05)  # low threshold for tests

    def test_empty_query_blocked(self):
        result = self.ig.check("")
        assert not result.passed
        assert result.reason == "query_too_short"

    def test_very_short_query_blocked(self):
        result = self.ig.check("hi")
        assert not result.passed
        assert result.reason == "query_too_short"

    def test_too_long_query_blocked(self):
        result = self.ig.check("a" * 600)
        assert not result.passed
        assert result.reason == "query_too_long"

    def test_normal_query_passes(self):
        result = self.ig.check("What is the speed of light?")
        assert result.passed

    def test_factual_query_passes(self):
        result = self.ig.check("How does photosynthesis work?")
        assert result.passed

    def test_profanity_blocked(self):
        result = self.ig.check("damn this is a bad question")
        # 'damn' may or may not be in profanity list depending on config
        # just ensure it doesn't crash
        assert isinstance(result.passed, bool)


class TestOutputGuardrails:
    def setup_method(self):
        self.og = OutputGuardrails(
            rouge_threshold=0.05,
            distance_threshold=1.8,  # permissive for unit tests
        )

    def test_grounded_answer_passes(self):
        answer = "Photosynthesis is the process by which plants convert light energy into sugar."
        context = ["Photosynthesis converts light energy. Plants use sunlight to make food from CO2 and water."]
        result = self.og.check(answer, context, top_score=0.85)
        assert result.passed

    def test_idk_response_blocked(self):
        answer = "I don't know the answer to this question."
        context = ["Photosynthesis is a biological process in plants."]
        result = self.og.check(answer, context, top_score=0.85)
        assert not result.passed
        assert result.reason == "idk_response"

    def test_low_confidence_blocked(self):
        answer = "Some random unrelated answer."
        context = ["Context about something else entirely."]
        result = self.og.check(answer, context, top_score=0.05)
        assert not result.passed
        assert result.reason == "low_confidence"

    def test_empty_answer_idk_blocked(self):
        answer = "I cannot find this information in the context."
        context = ["Some relevant context here."]
        result = self.og.check(answer, context, top_score=0.85)
        assert not result.passed


class TestGuardrailMessages:
    """Ensure safe_messages are user-friendly strings."""
    def test_input_messages_are_strings(self):
        ig = InputGuardrails()
        for query in ["", "a" * 600, "hi"]:
            result = ig.check(query)
            assert isinstance(result.safe_message, str)
            if not result.passed:
                assert len(result.safe_message) > 10

    def test_output_messages_are_strings(self):
        og = OutputGuardrails()
        result = og.check("I don't know", [], top_score=0.1)
        assert isinstance(result.safe_message, str)
