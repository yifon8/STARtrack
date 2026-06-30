"""
tests for skills/eval.py — run_meta_eval()

Run with:
    cd startrack_test && pytest tests/test_eval.py -v

All tests mock google.genai.Client so they run with NO real API key and
NO network access. They verify the *contract* of run_meta_eval, not the
actual judging quality of the real model (that needs a live eval set —
see the "live smoke test" at the bottom, skipped by default).
"""

import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from skills import eval as eval_mod  # noqa: E402
from skills.eval import run_meta_eval  # noqa: E402


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture(autouse=True)
def fake_api_key(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key-for-tests")


@pytest.fixture
def sample_history():
    return [
        {
            "question_id": "question_1",
            "user_id": "user_a",
            "attempt_number": 1,
            "transcript": "I worked with James on a project once.",
            "scores": {
                "star_structure": 1, "specificity": 1, "relevance": 2,
                "confidence_language": 1, "conciseness": 2, "overall_score": 7,
            },
        },
        {
            "question_id": "question_1",
            "user_id": "user_a",
            "attempt_number": 2,
            "transcript": (
                "I convinced James to adopt our API design within two weeks "
                "by preparing a comparison doc; adoption cut integration "
                "time by 30%."
            ),
            "scores": {
                "star_structure": 3, "specificity": 3, "relevance": 3,
                "confidence_language": 2, "conciseness": 3, "overall_score": 14,
            },
        },
    ]


@pytest.fixture
def sample_assessments(sample_history):
    """assess_answer()-shaped outputs matching sample_history attempts."""
    return [
        {
            "question_id": "question_1",
            "user_id": "user_a",
            "attempt_number": 1,
            "transcript": sample_history[0]["transcript"],
            "scores": sample_history[0]["scores"],
            "strengths": ["Mentioned a named stakeholder (James)."],
            "gaps": ["No measurable result.", "Vague on what was done."],
            "one_specific_improvement": "Be more specific next time.",
        },
        {
            "question_id": "question_1",
            "user_id": "user_a",
            "attempt_number": 2,
            "transcript": sample_history[1]["transcript"],
            "scores": sample_history[1]["scores"],
            "strengths": ["Quantified result (30% reduction).", "Named James."],
            "gaps": ["Slight hedging in delivery."],
            "one_specific_improvement": (
                "Replace 'I think it helped' with a direct statement of the "
                "30% integration-time reduction to sound more assertive."
            ),
        },
    ]


def _mock_genai_client(judge_responses: list[dict]):
    """Build a MagicMock standing in for genai.Client whose
    .models.generate_content() returns the given judge JSON payloads
    in sequence (one per call)."""
    mock_client = MagicMock()
    mock_responses = [
        SimpleNamespace(text=json.dumps(payload)) for payload in judge_responses
    ]
    mock_client.models.generate_content.side_effect = mock_responses
    return mock_client


# ------------------------------------------------------------------
# 1. Happy path — well-matched judge scores
# ------------------------------------------------------------------

def test_happy_path_returns_keyed_by_attempt_number(sample_history, sample_assessments):
    judge_payloads = [
        {
            "accuracy_score": 0.9,
            "accuracy_reason": "Scores align closely with transcript quality.",
            "actionability_score": 0.2,
            "actionability_reason": "Improvement is generic, not transcript-specific.",
        },
        {
            "accuracy_score": 0.95,
            "accuracy_reason": "Scores correctly reflect a strong STAR answer.",
            "actionability_score": 0.9,
            "actionability_reason": "Improvement references the exact hedging phrase to fix.",
        },
    ]

    with patch.object(eval_mod, "_get_client", return_value=_mock_genai_client(judge_payloads)):
        result = run_meta_eval(sample_history, sample_assessments)

    assert set(result.keys()) == {1, 2}

    a1 = result[1]
    assert a1["attempt_number"] == 1
    assert a1["accuracy_score"] == 0.9
    assert a1["actionability_score"] == 0.2
    assert a1["meta_score"] == pytest.approx(0.55, abs=1e-3)
    assert a1["flagged"] is True   # below 0.60 threshold -> generic improvement caught

    a2 = result[2]
    assert a2["meta_score"] == pytest.approx(0.925, abs=1e-3)
    assert a2["flagged"] is False


# ------------------------------------------------------------------
# 2. Score clamping — judge returns out-of-range values
# ------------------------------------------------------------------

def test_judge_scores_are_clamped_to_0_1(sample_history, sample_assessments):
    judge_payloads = [
        {
            "accuracy_score": 1.4,        # out of range, should clamp to 1.0
            "accuracy_reason": "x",
            "actionability_score": -0.3,  # out of range, should clamp to 0.0
            "actionability_reason": "y",
        },
        {
            "accuracy_score": 0.5,
            "accuracy_reason": "x",
            "actionability_score": 0.5,
            "actionability_reason": "y",
        },
    ]
    with patch.object(eval_mod, "_get_client", return_value=_mock_genai_client(judge_payloads)):
        result = run_meta_eval(sample_history, sample_assessments)

    assert result[1]["accuracy_score"] == 1.0
    assert result[1]["actionability_score"] == 0.0


# ------------------------------------------------------------------
# 3. Missing GOOGLE_API_KEY -> clear error, not a cryptic failure
# ------------------------------------------------------------------

def test_missing_api_key_raises_runtime_error(monkeypatch, sample_history, sample_assessments):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        run_meta_eval(sample_history, sample_assessments)


# ------------------------------------------------------------------
# 4. Empty assessments list -> ValueError, not a silent empty dict
# ------------------------------------------------------------------

def test_empty_assessments_raises_value_error(sample_history):
    with pytest.raises(ValueError, match="empty"):
        run_meta_eval(sample_history, [])


# ------------------------------------------------------------------
# 5. Assessment with no matching transcript anywhere -> ValueError
# ------------------------------------------------------------------

def test_no_transcript_found_raises_value_error(sample_history):
    orphan_assessment = [{
        "question_id": "question_1",
        "user_id": "user_a",
        "attempt_number": 99,   # not in history, no transcript embedded
        "scores": {"overall_score": 5},
        "one_specific_improvement": "n/a",
    }]
    with patch.object(eval_mod, "_get_client", return_value=_mock_genai_client([])):
        with pytest.raises(ValueError, match="no transcript found"):
            run_meta_eval(sample_history, orphan_assessment)


# ------------------------------------------------------------------
# 6. Falls back to transcript embedded in assessment if missing from history
# ------------------------------------------------------------------

def test_falls_back_to_assessment_transcript_when_not_in_history():
    assessment = [{
        "attempt_number": 1,
        "transcript": "Some standalone transcript not in history.",
        "scores": {"overall_score": 9},
        "one_specific_improvement": "Quantify the outcome.",
    }]
    judge_payloads = [{
        "accuracy_score": 0.7, "accuracy_reason": "ok",
        "actionability_score": 0.7, "actionability_reason": "ok",
    }]
    with patch.object(eval_mod, "_get_client", return_value=_mock_genai_client(judge_payloads)):
        result = run_meta_eval(history=[], assessments=assessment)

    assert result[1]["meta_score"] == 0.7


# ------------------------------------------------------------------
# 7. Malformed judge JSON for one attempt doesn't kill the whole batch
# ------------------------------------------------------------------

def test_one_bad_judge_response_does_not_abort_batch(sample_history, sample_assessments):
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = [
        SimpleNamespace(text="not valid json {{{"),   # attempt 1 fails
        SimpleNamespace(text=json.dumps({              # attempt 2 succeeds
            "accuracy_score": 0.8, "accuracy_reason": "fine",
            "actionability_score": 0.8, "actionability_reason": "fine",
        })),
    ]
    with patch.object(eval_mod, "_get_client", return_value=mock_client):
        result = run_meta_eval(sample_history, sample_assessments)

    assert result[1]["flagged"] is True
    assert result[1]["meta_score"] == 0.0
    assert "Judge call failed" in result[1]["accuracy_reason"]

    assert result[2]["flagged"] is False
    assert result[2]["meta_score"] == 0.8


# ------------------------------------------------------------------
# 8. before_tool_call / after_tool_call hooks actually fire
# ------------------------------------------------------------------

def test_trajectory_hooks_are_invoked(sample_history, sample_assessments):
    before_calls = []
    after_calls = []

    def fake_before(tool_name, kwargs):
        before_calls.append((tool_name, kwargs))

    def fake_after(tool_name, kwargs, result, elapsed_ms):
        after_calls.append((tool_name, result, elapsed_ms))

    judge_payloads = [
        {"accuracy_score": 0.5, "accuracy_reason": "x",
         "actionability_score": 0.5, "actionability_reason": "y"},
        {"accuracy_score": 0.5, "accuracy_reason": "x",
         "actionability_score": 0.5, "actionability_reason": "y"},
    ]

    with patch.object(eval_mod, "_get_client", return_value=_mock_genai_client(judge_payloads)), \
         patch.object(eval_mod, "before_tool_call", side_effect=fake_before), \
         patch.object(eval_mod, "after_tool_call", side_effect=fake_after):
        run_meta_eval(sample_history, sample_assessments)

    # one before/after pair per attempt judged
    assert len(before_calls) == 2
    assert len(after_calls) == 2
    assert before_calls[0][0] == "judge_single_attempt"
    assert after_calls[0][2] >= 0  # elapsed_ms is non-negative


# ------------------------------------------------------------------
# 9. Matching is by attempt_number, not list order
# ------------------------------------------------------------------

def test_matches_by_attempt_number_not_list_position(sample_history):
    # assessments deliberately out of order / reversed vs history
    assessments_reversed = [
        {
            "attempt_number": 2,
            "transcript": sample_history[1]["transcript"],
            "scores": sample_history[1]["scores"],
            "one_specific_improvement": "second",
        },
        {
            "attempt_number": 1,
            "transcript": sample_history[0]["transcript"],
            "scores": sample_history[0]["scores"],
            "one_specific_improvement": "first",
        },
    ]
    judge_payloads = [
        {"accuracy_score": 0.3, "accuracy_reason": "for attempt 2",
         "actionability_score": 0.3, "actionability_reason": "for attempt 2"},
        {"accuracy_score": 0.9, "accuracy_reason": "for attempt 1",
         "actionability_score": 0.9, "actionability_reason": "for attempt 1"},
    ]
    with patch.object(eval_mod, "_get_client", return_value=_mock_genai_client(judge_payloads)):
        result = run_meta_eval(sample_history, assessments_reversed)

    # result keyed correctly regardless of input order
    assert result[2]["accuracy_score"] == 0.3
    assert result[1]["accuracy_score"] == 0.9


# ------------------------------------------------------------------
# 10. (Optional) Live smoke test — only runs if you export a real key
#     and pass --run-live. Skipped by default so CI never needs network.
# ------------------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("RUN_LIVE_EVAL_TEST"),
    reason="Set RUN_LIVE_EVAL_TEST=1 and a real GOOGLE_API_KEY to hit the real model.",
)
def test_live_smoke_real_model(sample_history, sample_assessments):
    """Hits the actual gemini-3.1-flash judge model. Costs tokens — opt-in only."""
    result = run_meta_eval(sample_history, sample_assessments)
    assert set(result.keys()) == {1, 2}
    for verdict in result.values():
        assert 0.0 <= verdict["accuracy_score"] <= 1.0
        assert 0.0 <= verdict["actionability_score"] <= 1.0
        # the weak attempt 1 ("Be more specific next time.") should plausibly
        # score lower on actionability than attempt 2's specific rewrite
    assert result[1]["actionability_score"] < result[2]["actionability_score"]