"""
Skills: save_session, analyze_progression
Owner: P3
Persists attempt records to history/{user_id}.jsonl and generates a
narrative summary of improvement trends across a user's attempt series.
"""

import json
import os
from datetime import date
from pathlib import Path
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

_REQUIRED_SESSION_FIELDS = [
    "question_id",
    "user_id",
    "attempt_number",
    "scores",
    "strengths",
    "gaps",
    "one_specific_improvement",
    "transcript",
]


def save_session(session: dict, history_dir: str = "history") -> bool:
    """Append a validated session record to history/{user_id}.jsonl.

    Args:
        session: A complete session dict conforming to schema/session.json.
                 Must pass validate_session() before calling this function.
        history_dir: Directory containing per-user .jsonl history files.
                      Defaults to "history" (relative to cwd), matching the
                      convention used throughout this project.

    Returns:
        True if the record was written successfully, False otherwise.

    Raises:
        ValueError: if session is missing required fields, missing a
                    rubric dimension, or overall_score doesn't match the
                    sum of the five dimensions.
    """
    missing = [
        field
        for field in _REQUIRED_SESSION_FIELDS
        if session.get(field) is None
    ]
    if missing:
        raise ValueError(f"save_session: session missing required fields: {missing}")

    scores = session["scores"]
    missing_dims = [dim for dim in DIMENSIONS if dim not in scores]
    if missing_dims:
        raise ValueError(f"save_session: scores missing dimensions: {missing_dims}")
    if "overall_score" not in scores:
        raise ValueError("save_session: scores missing 'overall_score'")

    expected_overall = sum(scores[dim] for dim in DIMENSIONS)
    if scores["overall_score"] != expected_overall:
        raise ValueError(
            f"save_session: overall_score ({scores['overall_score']}) != "
            f"sum of dimensions ({expected_overall}). "
            "Run validate_session() before save_session()."
        )

    record = {
        "question_id": session["question_id"],
        "user_id": session["user_id"],
        "attempt_number": session["attempt_number"],
        "date": session.get("date") or date.today().isoformat(),
        "source_file": session.get("source_file"),
        "transcription_method": session.get("transcription_method") or "text_upload",
        "transcript": session["transcript"],
        "scores": scores,
        "strengths": session["strengths"],
        "gaps": session["gaps"],
        "one_specific_improvement": session["one_specific_improvement"],
    }

    try:
        history_path = Path(history_dir)
        history_path.mkdir(parents=True, exist_ok=True)
        file_path = history_path / f"{session['user_id']}.jsonl"

        # Read existing records (if any) so we can de-duplicate on
        # (question_id, user_id, attempt_number) -- saving the same attempt
        # twice overwrites rather than appends a duplicate.
        existing_records = []
        if file_path.exists():
            with file_path.open("r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        key = (record["question_id"], record["user_id"], record["attempt_number"])
        existing_records = [
            r
            for r in existing_records
            if (r.get("question_id"), r.get("user_id"), r.get("attempt_number")) != key
        ]
        existing_records.append(record)

        with file_path.open("w") as f:
            for r in existing_records:
                f.write(json.dumps(r) + "\n")

        return True
    except OSError:
        return False


def _load_history(
    user_id: str, question_id: str, history_dir: str
) -> list[dict]:
    """Load and sort all session records for a user, filtered by question_id."""
    file_path = Path(history_dir) / f"{user_id}.jsonl"
    if not file_path.exists():
        return []

    records = []
    with file_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("question_id") == question_id:
                records.append(record)

    records.sort(key=lambda r: r.get("attempt_number", 0))
    return records


def _compute_dimension_trends(records: list[dict]) -> dict[str, str]:
    """Deterministically classify each dimension's trend from first to last
    attempt. Kept arithmetic (not LLM-derived) so it's always self-consistent
    with the raw scores, the same way assess_answer() computes overall_score
    itself rather than trusting the model.
    """
    trends = {}
    first, last = records[0]["scores"], records[-1]["scores"]
    for dim in DIMENSIONS:
        delta = last[dim] - first[dim]
        if delta > 0:
            trends[dim] = "improving"
        elif delta < 0:
            trends[dim] = "declining"
        else:
            trends[dim] = "plateauing"
    return trends


def _compute_score_history(records: list[dict]) -> list[dict]:
    """Raw overall_score per attempt, in order. This exists specifically so
    the agent has a concrete, real number to quote when referencing a prior
    attempt's score in its response -- without it, there is no grounded
    field for "your first attempt scored X" to come from, and the LLM will
    invent a plausible-sounding number instead.
    """
    return [
        {"attempt_number": r["attempt_number"], "overall_score": r["scores"]["overall_score"]}
        for r in records
    ]


_NARRATIVE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "trend": {"type": "string"},
        "persistent_gaps": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "trend", "persistent_gaps"],
}


def _build_narrative_prompt(records: list[dict], dimension_trends: dict) -> str:
    attempts_summary = []
    for r in records:
        attempts_summary.append(
            {
                "attempt_number": r["attempt_number"],
                "scores": r["scores"],
                "strengths": r["strengths"],
                "gaps": r["gaps"],
                "one_specific_improvement": r["one_specific_improvement"],
            }
        )

    return f"""You are an expert interview coach reviewing a candidate's progression
across multiple attempts at the same behavioral interview question.

Per-dimension score trends (computed from first attempt to most recent,
already calculated -- do not recompute, just use these as ground truth):
{json.dumps(dimension_trends, indent=2)}

Full attempt history, in order:
{json.dumps(attempts_summary, indent=2)}

Based on this history:
- summary: a 2-4 sentence narrative describing how the candidate has
  progressed overall (reference specific attempts/scores where useful).
- trend: a single overall directional label such as "improving",
  "plateauing", "uneven", or "declining" -- describing the candidate's
  trajectory as a whole, not per-dimension.
- persistent_gaps: 1-3 weaknesses that recur across multiple attempts
  (semantically, even if worded differently each time) -- not just a
  restatement of the most recent attempt's gaps.

Respond with JSON only, matching the required schema exactly.
"""


def analyze_progression(
    user_id: str,
    question_id: str = "question_1",
    history_dir: str = "history",
    model: str = "gemini-3.1-flash-lite",
) -> dict:
    """Generate a narrative analysis across all recorded attempts for a user.

    Reads history/{user_id}.jsonl and computes trend direction, persistent
    gaps, and dimension-level observations across attempts 1-N.

    Args:
        user_id: Progression track identifier (e.g. user_c).
        question_id: Filters records to a single question set.
        history_dir: Path to the directory containing .jsonl history files.
        model: Model to use for narrative generation.

    Returns:
        dict with keys: summary (str), trend (str), persistent_gaps (list[str]),
        dimension_trends (dict[str, str]), attempt_count (int).
    """
    records = _load_history(user_id, question_id, history_dir)
    attempt_count = len(records)

    if attempt_count < 2:
        # No LLM call needed -- there's nothing to compare yet. Returning a
        # clear, structured "not enough data" result rather than raising
        # keeps this safe to call after attempt 1, per AGENTS.md's flow
        # (analyze_progression is only meant to run from attempt 2 onward,
        # but a defensive caller should still get a sane response).
        return {
            "summary": (
                "Only one attempt on record; trend analysis requires at "
                "least two attempts."
                if attempt_count == 1
                else "No attempts on record for this user and question."
            ),
            "trend": "insufficient_data",
            "persistent_gaps": [],
            "dimension_trends": {},
            "score_history": _compute_score_history(records),
            "attempt_count": attempt_count,
        }

    dimension_trends = _compute_dimension_trends(records)

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    client = genai.Client(api_key=api_key)
    prompt = _build_narrative_prompt(records, dimension_trends)

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_NARRATIVE_RESPONSE_SCHEMA,
            temperature=0.2,
        ),
    )

    try:
        result = json.loads(response.text)
    except (json.JSONDecodeError, TypeError) as e:
        raise RuntimeError(f"analyze_progression: model did not return valid JSON: {e}")

    return {
        "summary": result.get("summary", ""),
        "trend": result.get("trend", "uneven"),
        "persistent_gaps": list(result.get("persistent_gaps", []))[:3],
        "dimension_trends": dimension_trends,
        "score_history": _compute_score_history(records),
        "attempt_count": attempt_count,
    }
