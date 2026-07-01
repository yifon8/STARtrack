"""
Integration tests for the UI ↔ eval.py boundary in ui.py.

Covers the two functions added/changed since the original test_eval.py:
  - _run_meta_eval_safe(history, assessment) -> str
  - _handle_run_eval(user_id) -> str

These sit between run_meta_eval() (already tested in test_eval.py) and the
Gradio component. They own formatting, non-blocking error handling, and the
"load latest attempt from history" logic.

Run with:
    pytest tests/test_eval_ui.py -v

No API key or network access required — run_meta_eval is mocked throughout.
"""

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ui.py imports gradio at module level; stub it before importing so tests
# run without a running Gradio server or display.
import unittest.mock as _mock
sys.modules.setdefault("gradio", _mock.MagicMock())

import skills.ui as ui_mod  # noqa: E402 — must come after the gradio stub
from skills.ui import _run_meta_eval_safe, _handle_run_eval  # noqa: E402


# ------------------------------------------------------------------
# Shared fixtures
# ------------------------------------------------------------------

@pytest.fixture
def weak_assessment():
    return {
        "question_id": "question_1",
        "user_id": "user_a",
        "attempt_number": 1,
        "transcript": "I worked with a coworker once.",
        "scores": {
            "star_structure": 1, "specificity": 1, "relevance": 1,
            "confidence_language": 1, "conciseness": 1, "overall_score": 5,
        },
        "strengths": ["Mentioned a coworker."],
        "gaps": ["No specifics."],
        "one_specific_improvement": "Be more specific.",
    }


@pytest.fixture
def strong_assessment():
    return {
        "question_id": "question_1",
        "user_id": "user_a",
        "attempt_number": 2,
        "transcript": "I convinced James in two weeks; adoption cut time by 30%.",
        "scores": {
            "star_structure": 3, "specificity": 3, "relevance": 3,
            "confidence_language": 3, "conciseness": 3, "overall_score": 15,
        },
        "strengths": ["Quantified result.", "Named James."],
        "gaps": [],
        "one_specific_improvement": "Trim the opening clause to reach the action faster.",
    }


@pytest.fixture
def sample_history(weak_assessment, strong_assessment):
    return [weak_assessment, strong_assessment]


def _verdict(accuracy, actionability, flagged=False):
    """Build a minimal run_meta_eval() verdict dict."""
    return {
        "accuracy_score": accuracy,
        "accuracy_reason": f"Accuracy reason at {accuracy}.",
        "actionability_score": actionability,
        "actionability_reason": f"Actionability reason at {actionability}.",
        "meta_score": round((accuracy + actionability) / 2, 3),
        "flagged": flagged,
        "judge_raw": {},
    }


# ------------------------------------------------------------------
# _run_meta_eval_safe: output format
# ------------------------------------------------------------------

class TestRunMetaEvalSafeFormat:

    def test_ok_verdict_contains_status_accuracy_actionable_judge_score(
        self, sample_history, weak_assessment
    ):
        verdict = _verdict(0.9, 0.8)
        with patch.object(ui_mod, "run_meta_eval", return_value={1: verdict}):
            output = _run_meta_eval_safe(sample_history, weak_assessment)

        assert "STATUS" in output
        assert "✅ OK" in output
        assert "ACCURACY" in output
        assert "ACTIONABLE" in output
        assert "JUDGE SCORE" in output

    def test_flagged_verdict_shows_flagged_status(
        self, sample_history, weak_assessment
    ):
        verdict = _verdict(0.3, 0.2, flagged=True)
        with patch.object(ui_mod, "run_meta_eval", return_value={1: verdict}):
            output = _run_meta_eval_safe(sample_history, weak_assessment)

        assert "⚠️ FLAGGED" in output
        assert "✅ OK" not in output

    def test_scores_formatted_as_x_slash_1(
        self, sample_history, weak_assessment
    ):
        verdict = _verdict(0.75, 0.60)
        with patch.object(ui_mod, "run_meta_eval", return_value={1: verdict}):
            output = _run_meta_eval_safe(sample_history, weak_assessment)

        assert "0.75/1.00" in output
        assert "0.60/1.00" in output

    def test_full_reason_text_not_truncated(
        self, sample_history, weak_assessment
    ):
        long_reason = "A" * 120   # longer than the old 80-char truncation limit
        verdict = {
            "accuracy_score": 0.8,
            "accuracy_reason": long_reason,
            "actionability_score": 0.8,
            "actionability_reason": "Short reason.",
            "meta_score": 0.8,
            "flagged": False,
            "judge_raw": {},
        }
        with patch.object(ui_mod, "run_meta_eval", return_value={1: verdict}):
            output = _run_meta_eval_safe(sample_history, weak_assessment)

        assert long_reason in output          # full text present
        assert "…" not in output             # no ellipsis truncation


# ------------------------------------------------------------------
# _run_meta_eval_safe: non-blocking error handling
# ------------------------------------------------------------------

class TestRunMetaEvalSafeErrors:

    def test_run_meta_eval_exception_returns_string_not_raise(
        self, sample_history, weak_assessment
    ):
        with patch.object(
            ui_mod, "run_meta_eval", side_effect=RuntimeError("API down")
        ):
            output = _run_meta_eval_safe(sample_history, weak_assessment)

        assert isinstance(output, str)
        assert "unavailable" in output.lower()
        assert "API down" in output

    def test_verdict_missing_for_attempt_returns_string_not_raise(
        self, sample_history, weak_assessment
    ):
        # run_meta_eval returns a dict but the attempt_number key is absent
        with patch.object(ui_mod, "run_meta_eval", return_value={}):
            output = _run_meta_eval_safe(sample_history, weak_assessment)

        assert isinstance(output, str)
        assert "no result" in output.lower()

    def test_missing_api_key_is_caught_not_propagated(
        self, monkeypatch, sample_history, weak_assessment
    ):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with patch.object(
            ui_mod, "run_meta_eval",
            side_effect=RuntimeError("GOOGLE_API_KEY is not set")
        ):
            output = _run_meta_eval_safe(sample_history, weak_assessment)

        assert isinstance(output, str)   # never raises; UI always gets a string


# ------------------------------------------------------------------
# _handle_run_eval: loads latest attempt from history
# ------------------------------------------------------------------

class TestHandleRunEval:

    def test_returns_string_for_user_with_history(
        self, sample_history, strong_assessment
    ):
        # strong_assessment is attempt 2, the last one in sample_history
        verdict = _verdict(0.9, 0.85)
        with patch.object(ui_mod, "_load_history", return_value=sample_history), \
             patch.object(ui_mod, "run_meta_eval", return_value={2: verdict}):
            output = _handle_run_eval("user_a")

        assert isinstance(output, str)
        assert "ACCURACY" in output

    def test_judges_the_last_attempt_not_the_first(self, sample_history):
        # history has attempt 1 and 2; judge should be called with attempt 2
        called_with = {}

        def capture_call(history, assessments):
            called_with["attempt_number"] = assessments[0]["attempt_number"]
            return {assessments[0]["attempt_number"]: _verdict(0.8, 0.8)}

        with patch.object(ui_mod, "_load_history", return_value=sample_history), \
             patch.object(ui_mod, "run_meta_eval", side_effect=capture_call):
            _handle_run_eval("user_a")

        assert called_with["attempt_number"] == 2  # last in history, not first

    def test_no_history_returns_instructional_message(self):
        with patch.object(ui_mod, "_load_history", return_value=[]):
            output = _handle_run_eval("user_a")

        assert isinstance(output, str)
        assert "submit" in output.lower()   # tells user what to do next
        # run_meta_eval should NOT have been called