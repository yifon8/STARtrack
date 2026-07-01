"""
Skill: run_meta_eval
Owner: P3
LLM-as-judge evaluation of assessment quality.

Scores each assessment on two meta-dimensions:
  - accuracy:      Did the dimension scores match the transcript's actual quality?
  - actionability: Is one_specific_improvement concrete and attempt-specific?

Implements before_tool_call / after_tool_call hooks for trajectory logging.
Model: gemini-3.1-flash-lite

Entry point:
    run_meta_eval(history, assessments) -> dict

    history:     list[dict]  — session records (same shape as history/*.jsonl)
    assessments: list[dict]  — assess_answer() outputs, one per attempt

Returns a dict keyed by attempt_number, each value containing:
    {
        "attempt_number":   int,
        "accuracy_score":   float,   # 0.0 – 1.0
        "accuracy_reason":  str,
        "actionability_score":  float,   # 0.0 – 1.0
        "actionability_reason": str,
        "meta_score":       float,   # average of accuracy + actionability
        "flagged":          bool,    # True if meta_score < FLAG_THRESHOLD
        "judge_raw":        dict,    # full judge JSON for debugging
    }

Trajectory hooks surface as module-level callables so the ADK agent (or
tests) can monkey-patch them:
    before_tool_call(tool_name, kwargs)  -> None
    after_tool_call(tool_name, kwargs, result, elapsed_ms) -> None
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

from google import genai
from google.genai import types as genai_types

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

_MODEL = "gemini-3.5-flash"

# meta_score below this threshold gets flagged for human review
_FLAG_THRESHOLD = 0.60

_DIMENSIONS = [
    "star_structure",
    "specificity",
    "relevance",
    "confidence_language",
    "conciseness",
]

# ------------------------------------------------------------------
# Trajectory hooks (before / after tool call)
#
# These are module-level so they can be replaced at runtime:
#   import skills.eval as ev
#   ev.before_tool_call = my_logger
# ------------------------------------------------------------------

def before_tool_call(tool_name: str, kwargs: dict) -> None:
    """Called immediately before any internal LLM call in this Skill.

    Args:
        tool_name: Logical name of the operation (e.g. "judge_single_attempt").
        kwargs:    The arguments being passed to the LLM call.
    """
    logger.debug("[TRAJECTORY] before_tool_call | tool=%s | kwargs_keys=%s",
                 tool_name, list(kwargs.keys()))


def after_tool_call(
    tool_name: str,
    kwargs: dict,
    result: Any,
    elapsed_ms: float,
) -> None:
    """Called immediately after any internal LLM call in this Skill.

    Args:
        tool_name:  Logical name of the operation.
        kwargs:     The arguments that were passed to the LLM call.
        result:     Whatever the LLM call returned (parsed dict or raw str).
        elapsed_ms: Wall-clock time the call took, in milliseconds.
    """
    logger.debug(
        "[TRAJECTORY] after_tool_call | tool=%s | elapsed_ms=%.1f | "
        "result_keys=%s",
        tool_name,
        elapsed_ms,
        list(result.keys()) if isinstance(result, dict) else type(result).__name__,
    )


# ------------------------------------------------------------------
# Judge prompt + schema
# ------------------------------------------------------------------

_RUBRIC_SUMMARY = """\
The 5-dimension STAR rubric (each 0-3):
  star_structure      – clarity of Situation/Task/Action/Result
  specificity         – named people, quantified outcomes, concrete timeline
  relevance           – directly addresses "influence without authority"
  confidence_language – assertive, active voice; minimal hedging/fillers
  conciseness         – spoken length 90 s – 2.5 min scores 3; outside range penalised

overall_score = sum of all five dimensions (max 15).
"""

_JUDGE_PROMPT_TEMPLATE = """\
You are an expert meta-evaluator for a behavioral interview coaching system.

## Your task
Evaluate the QUALITY of an automated assessment that was produced for a
candidate's answer to this interview question:

  "Tell me about a time you had to influence someone without authority."

You will be given:
  1. The candidate's transcript (what they actually said / wrote).
  2. The automated assessment that was generated for it.

You must rate the assessment on two dimensions (each 0.0 – 1.0):

### accuracy
Did the automated dimension scores reflect the actual quality of the transcript?
  1.0 – Every dimension score is exactly what a calibrated human judge would give.
  0.75 – At most one dimension is off by 1 point; no score is off by 2+.
  0.50 – Two dimensions off by 1, OR one dimension off by 2.
  0.25 – Multiple dimensions badly wrong, or systematic over/under-scoring.
  0.0  – Scores bear no relation to the transcript content.

### actionability
Is the one_specific_improvement concrete and tailored to THIS transcript?
  1.0 – Names a specific gap from this transcript and gives a concrete fix
         (e.g. "Add the exact percentage result you mentioned knowing but
         did not state aloud.").
  0.75 – Mostly specific; could be slightly more concrete or transcript-anchored.
  0.50 – Half generic advice, half specific; usable but not ideal.
  0.25 – Mostly generic ("be more specific") with token reference to transcript.
  0.0  – Pure generic coaching tip; could apply to any candidate.

## Rubric reference
{rubric}

## Assessment to judge
Attempt number: {attempt_number}

### Transcript
\"\"\"
{transcript}
\"\"\"

### Automated assessment output
```json
{assessment_json}
```

Respond ONLY with valid JSON matching this exact schema — no preamble, no
markdown fences:
{{
  "accuracy_score":        <float 0.0-1.0>,
  "accuracy_reason":       <one sentence explaining the accuracy rating>,
  "actionability_score":   <float 0.0-1.0>,
  "actionability_reason":  <one sentence explaining the actionability rating>
}}
"""

_JUDGE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "accuracy_score":        {"type": "number"},
        "accuracy_reason":       {"type": "string"},
        "actionability_score":   {"type": "number"},
        "actionability_reason":  {"type": "string"},
    },
    "required": [
        "accuracy_score",
        "accuracy_reason",
        "actionability_score",
        "actionability_reason",
    ],
}


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _get_client() -> genai.Client:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return genai.Client(api_key=api_key)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))



def _judge_single_attempt(
    client: genai.Client,
    transcript: str,
    assessment: dict,
    attempt_number: int,
) -> dict:
    """Call the judge model for one attempt and return the parsed verdict."""
    prompt = _JUDGE_PROMPT_TEMPLATE.format(
        rubric=_RUBRIC_SUMMARY,
        attempt_number=attempt_number,
        transcript=transcript,
        assessment_json=json.dumps(assessment, indent=2),
    )

    call_kwargs = {
        "model": _MODEL,
        "prompt_length": len(prompt),
        "attempt_number": attempt_number,
    }

    # --- before hook ---
    before_tool_call("judge_single_attempt", call_kwargs)

    t0 = time.perf_counter()

    response = client.models.generate_content(
        model=_MODEL,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_JUDGE_RESPONSE_SCHEMA,
            temperature=0.1,   # low temp for consistent judging
        ),
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    try:
        judge_raw = json.loads(response.text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError(
            f"run_meta_eval: judge model returned non-JSON for attempt "
            f"{attempt_number}: {exc}\nRaw: {response.text!r}"
        ) from exc

    # --- after hook ---
    after_tool_call("judge_single_attempt", call_kwargs, judge_raw, elapsed_ms)

    return judge_raw


def _build_attempt_result(
    attempt_number: int,
    judge_raw: dict,
) -> dict:
    """Combine raw judge output into a normalised per-attempt result dict."""
    accuracy_score      = _clamp(judge_raw.get("accuracy_score", 0.0))
    actionability_score = _clamp(judge_raw.get("actionability_score", 0.0))
    meta_score          = (accuracy_score + actionability_score) / 2.0

    return {
        "attempt_number":        attempt_number,
        "accuracy_score":        round(accuracy_score, 3),
        "accuracy_reason":       judge_raw.get("accuracy_reason", ""),
        "actionability_score":   round(actionability_score, 3),
        "actionability_reason":  judge_raw.get("actionability_reason", ""),
        "meta_score":            round(meta_score, 3),
        "flagged":               meta_score < _FLAG_THRESHOLD,
        "judge_raw":             judge_raw,
    }


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def run_meta_eval(
    history: list[dict],
    assessments: list[dict],
    question_id: str = "question_1",
    model: str = _MODEL,
) -> dict:
    """LLM-as-judge layer scoring the quality of the assessment Skill's outputs.

    Implements trajectory inspection hooks (before_tool_call / after_tool_call).

    Args:
        history:     List of session dicts for this user (all saved attempts).
                     Used to retrieve the original transcript for each attempt.
                     Each record must contain at minimum:
                         attempt_number (int), transcript (str), scores (dict).
        assessments: List of assess_answer() output dicts to be judged.
                     Each must contain at minimum:
                         attempt_number (int), scores (dict),
                         one_specific_improvement (str).
                     assessments and history do NOT need to be the same length
                     or sorted; this function matches on attempt_number.
        question_id: Active question identifier (informational; not used in
                     prompts for v1.0 since only question_1 exists).
        model:       Override the judge model (default: gemini-3.1-flash-lite).

    Returns:
        dict keyed by attempt_number (int), each value containing:
            attempt_number       (int)
            accuracy_score       (float, 0.0-1.0)
            accuracy_reason      (str)
            actionability_score  (float, 0.0-1.0)
            actionability_reason (str)
            meta_score           (float, 0.0-1.0, average of the two)
            flagged              (bool, True if meta_score < 0.60)
            judge_raw            (dict, full judge JSON for debugging)

    Raises:
        ValueError:  If an assessment has no matching transcript in history,
                     or if assessments list is empty.
        RuntimeError: If GOOGLE_API_KEY is missing or the judge model fails.

    Example:
        >>> results = run_meta_eval(history, assessments)
        >>> for attempt_num, verdict in results.items():
        ...     print(attempt_num, verdict["meta_score"], verdict["flagged"])
    """
    if not assessments:
        raise ValueError("run_meta_eval: assessments list is empty.")

    # Validate model override propagates to the LLM call
    global _MODEL
    _MODEL = model

    client = _get_client()

    # Index history by attempt_number for O(1) lookup
    history_index: dict[int, dict] = {
        rec["attempt_number"]: rec for rec in history
    }

    results: dict[int, dict] = {}

    for assessment in assessments:
        attempt_number = assessment.get("attempt_number")
        if attempt_number is None:
            logger.warning(
                "run_meta_eval: skipping assessment with no attempt_number: %s",
                assessment,
            )
            continue

        # Prefer transcript from history (the ground-truth saved record);
        # fall back to transcript embedded in the assessment dict itself.
        history_record = history_index.get(attempt_number)
        if history_record:
            transcript = history_record.get("transcript", "")
        else:
            transcript = assessment.get("transcript", "")

        if not transcript:
            raise ValueError(
                f"run_meta_eval: no transcript found for attempt_number="
                f"{attempt_number}. Ensure the attempt is saved to history "
                f"before calling run_meta_eval, or include 'transcript' in "
                f"the assessment dict."
            )

        try:
            judge_raw = _judge_single_attempt(
                client=client,
                transcript=transcript,
                assessment=assessment,
                attempt_number=attempt_number,
            )
        except RuntimeError as exc:
            # Log but continue so one bad call doesn't abort the whole batch.
            logger.error(
                "run_meta_eval: judge call failed for attempt %d: %s",
                attempt_number,
                exc,
            )
            results[attempt_number] = {
                "attempt_number":        attempt_number,
                "accuracy_score":        0.0,
                "accuracy_reason":       f"Judge call failed: {exc}",
                "actionability_score":   0.0,
                "actionability_reason":  f"Judge call failed: {exc}",
                "meta_score":            0.0,
                "flagged":               True,
                "judge_raw":             {},
            }
            continue

        results[attempt_number] = _build_attempt_result(attempt_number, judge_raw)

    return results