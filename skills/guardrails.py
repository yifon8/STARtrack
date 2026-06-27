"""
Skills: semantic_gate, validate_session
Owner: P2
Two guardrails that run on every attempt before data is persisted.
semantic_gate confirms the transcript is a plausible behavioral interview answer.
validate_session confirms overall_score equals the sum of the five dimension scores.
"""

from typing import Optional


def semantic_gate(
    transcript: str,
    question_id: str = "question_1",
    model: str = "gemini-2.5-flash-lite",
) -> dict:
    """Check whether a transcript is a plausible behavioral interview answer.

    Uses an LLM call to detect garbage input, prompt injection attempts, or
    text that is clearly unrelated to the active question before full assessment.

    Args:
        transcript: Raw candidate answer text to evaluate.
        question_id: Active question identifier (used to anchor the relevance check).
        model: Model to use for the semantic check.

    Returns:
        dict with keys: passed (bool), reason (str).
    """
    pass


def validate_session(session: dict) -> dict:
    """Verify that overall_score equals the sum of the five dimension scores.

    Args:
        session: Session dict conforming to schema/session.json,
                 after assess_answer() has populated the scores field.

    Returns:
        dict with keys: valid (bool), reason (str), expected_overall (int).
    """
    pass
