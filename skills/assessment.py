"""
Skill: assess_answer
Owner: P2
Scores a transcript against the five-dimension STAR rubric defined in AGENTS.md.
Returns scores, strengths, gaps, and one_specific_improvement.
"""

from typing import Optional


def assess_answer(
    transcript: str,
    question_id: str = "question_1",
    user_id: Optional[str] = None,
    attempt_number: Optional[int] = None,
) -> dict:
    """Score a candidate's answer transcript against the rubric.

    Args:
        transcript: Raw answer text (typed or transcribed).
        question_id: Which question is being answered (default: question_1).
        user_id: Progression track identifier (e.g. user_a).
        attempt_number: Position within the user's progression track (1–5).

    Returns:
        dict matching the session.json schema:
          scores (star_structure, specificity, relevance, confidence_language,
                  conciseness, overall_score), strengths, gaps,
          one_specific_improvement.
    """
    pass
