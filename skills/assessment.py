"""
Skill: assess_answer
Owner: P2
Scores a transcript against the five-dimension STAR rubric defined in AGENTS.md.
Returns scores, strengths, gaps, and one_specific_improvement.
"""

import json
import os
from typing import Optional

from google import genai
from google.genai import types as genai_types

DIMENSIONS = [
    "star_structure",
    "specificity",
    "relevance",
    "confidence_language",
    "conciseness",
]

# Question text keyed by question_id. Extend this as new questions are added.
_QUESTION_TEXT = {
    "question_1": "Tell me about a time you had to influence someone without authority.",
}

# Rubric + calibration anchors, copied from AGENTS.md so this Skill is
# self-contained and stays in sync with the source of truth.
_RUBRIC = """You are an expert behavioral interview coach. Score the candidate's answer
transcript against the following five-dimension STAR rubric. Each dimension
is scored from 0 to 3.

star_structure:
  0 - No recognizable structure
  1 - Situation only, or a vague sequence of events
  2 - Situation + Action present, but Result is weak or missing
  3 - Clear Situation / Task / Action / Result, with a distinct Result

specificity:
  0 - No concrete details
  1 - Vague claims, no names or numbers
  2 - Some specifics but incomplete
  3 - Named people, quantified outcomes, concrete timeline

relevance:
  0 - Does not address the question
  1 - Tangentially related
  2 - Mostly on-topic with minor drift
  3 - Directly and fully addresses the question

confidence_language:
  0 - Excessive hedging, filler words throughout
  1 - Frequent hedging or passive voice
  2 - Occasional hedging, mostly direct
  3 - Assertive, direct, active voice throughout

conciseness (estimate spoken length from the transcript at ~130 words/minute):
  0 - Under 30 seconds or over 4 minutes
  1 - 30-75 seconds or 3-4 minutes
  2 - 75-90 seconds or 2.5-3 minutes
  3 - 90 seconds to 2.5 minutes (sweet spot)

Calibration anchors for specificity:
  "I worked on a project once" -> 0
  "I worked with a senior engineer named James" -> 1
  "I convinced James to adopt our API design by preparing a comparison doc" -> 2
  "I convinced James within two weeks; adoption cut integration time by 30%" -> 3

Calibration anchors for conciseness (spoken length):
  Under 30 seconds -> 0 (no room for full STAR)
  30-75 seconds -> 1 (STAR possible but Action always thin)
  75-90 seconds or 2.5-3 minutes -> 2 (close but slightly off)
  90 seconds to 2.5 minutes -> 3 (sweet spot - complete STAR with details)
"""

# JSON schema the model is constrained to. Keeping this here (rather than
# letting the model free-write) is what makes downstream parsing reliable.
_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "object",
            "properties": {
                "star_structure": {"type": "integer"},
                "specificity": {"type": "integer"},
                "relevance": {"type": "integer"},
                "confidence_language": {"type": "integer"},
                "conciseness": {"type": "integer"},
            },
            "required": [
                "star_structure",
                "specificity",
                "relevance",
                "confidence_language",
                "conciseness",
            ],
        },
        "strengths": {"type": "array", "items": {"type": "string"}},
        "gaps": {"type": "array", "items": {"type": "string"}},
        "one_specific_improvement": {"type": "string"},
    },
    "required": ["scores", "strengths", "gaps", "one_specific_improvement"],
}


def _build_prompt(transcript: str, question_id: str) -> str:
    question_text = _QUESTION_TEXT.get(
        question_id, "the active behavioral interview question"
    )
    return f"""{_RUBRIC}
The interview question being answered is:
"{question_text}"

Candidate's transcript:
\"\"\"
{transcript}
\"\"\"

Score the transcript against all five dimensions above.

Provide:
- strengths: 1 to 3 specific things the candidate did well. Reference actual
  content from the transcript (names, numbers, phrasing) -- not generic praise.
- gaps: 1 to 3 specific things missing or weak. Reference actual content.
- one_specific_improvement: one concrete, actionable instruction the candidate
  should apply on their next attempt. It must be specific to this transcript,
  not generic interview advice.

Respond with JSON only, matching the required schema exactly.
"""


def assess_answer(
    transcript: str,
    question_id: str = "question_1",
    user_id: Optional[str] = None,
    attempt_number: Optional[int] = None,
    model: str = "gemini-3.1-flash-lite",
) -> dict:
    """Score a candidate's answer transcript against the rubric.

    Args:
        transcript: Raw answer text (typed or transcribed).
        question_id: Which question is being answered (default: question_1).
        user_id: Progression track identifier (e.g. user_a).
        attempt_number: Position within the user's progression track (1-5).
        model: Model to use for scoring.

    Returns:
        dict matching the session.json schema:
          scores (star_structure, specificity, relevance, confidence_language,
                  conciseness, overall_score), strengths, gaps,
          one_specific_improvement. Also echoes back question_id, user_id,
          attempt_number, and transcript so the caller can pass the result
          straight into validate_session() / save_session() without
          re-assembling the record.

    Raises:
        ValueError: if transcript is empty/blank.
        RuntimeError: if the model response can't be parsed as valid JSON.
    """
    if not transcript or not transcript.strip():
        raise ValueError("transcript must be a non-empty string")

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    client = genai.Client(api_key=api_key)
    prompt = _build_prompt(transcript, question_id)

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_RESPONSE_SCHEMA,
            temperature=0.2,
        ),
    )

    try:
        result = json.loads(response.text)
    except (json.JSONDecodeError, TypeError) as e:
        raise RuntimeError(f"assess_answer: model did not return valid JSON: {e}")

    raw_scores = result.get("scores", {})

    # Clamp every dimension to the valid 0-3 range, then derive overall_score
    # ourselves rather than trusting the model's arithmetic. validate_session()
    # double-checks this later, but there's no reason to ship a known-wrong sum.
    scores = {}
    for dim in DIMENSIONS:
        value = int(raw_scores.get(dim, 0))
        scores[dim] = max(0, min(3, value))
    scores["overall_score"] = sum(scores[dim] for dim in DIMENSIONS)

    return {
        "question_id": question_id,
        "user_id": user_id,
        "attempt_number": attempt_number,
        "transcript": transcript,
        "scores": scores,
        "strengths": list(result.get("strengths", []))[:3],
        "gaps": list(result.get("gaps", []))[:3],
        "one_specific_improvement": result.get("one_specific_improvement", ""),
    }
