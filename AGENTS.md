# Interview Practice Coach — AGENTS.md

This is the static context file for the Interview Practice Coach agent.
It is the source of truth for agent behavior, rubric definitions, Skill inventory,
and data contracts. Load this file at agent startup. Do not modify during a session.

---

## Agent Purpose

The Interview Practice Coach helps job seekers improve their behavioral interview
answers over time. It assesses individual answers against a structured rubric,
saves attempt history, and analyzes progression across a series of attempts to
identify improvement trends and persistent gaps.

---

## The Interview Question

### question_id: question_1

> "Tell me about a time you had to influence someone without authority."

This is the only active question for v1.0. All attempt data in `question_1/` is
anchored to this question. Do not change the question mid-series — doing so
invalidates progression comparisons.

**What a strong answer looks like:**
A strong answer identifies a specific situation where the candidate had no formal
power over the person they needed to influence. It explains what was at stake
(Task), describes the specific influencing actions taken (Action), and closes
with a concrete measurable outcome (Result). Ideal spoken length: 90 seconds to 2.5 minutes.

**This question tests:** stakeholder management, communication, emotional
intelligence, and the ability to drive outcomes without positional authority.

---

## Rubric Definitions

All answers are scored on five dimensions, each from 0 to 3.
`overall_score` must always equal the sum of all five (maximum 15).

| Dimension | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **star_structure** | No recognizable structure | Situation only or vague sequence | Situation + Action present, Result weak or missing | Clear S/T/A/R with distinct Result |
| **specificity** | No concrete details | Vague claims, no names or numbers | Some specifics but incomplete | Named people, quantified outcomes, concrete timeline |
| **relevance** | Does not address the question | Tangentially related | Mostly on-topic with minor drift | Directly and fully addresses the question |
| **confidence_language** | Excessive hedging, filler words throughout | Frequent hedging or passive voice | Occasional hedging, mostly direct | Assertive, direct, active voice throughout |
| **conciseness** | Under 30 seconds or over 4 minutes | 30–75 seconds or 3–4 minutes | 75–90 seconds or 2.5–3 minutes | 90 seconds – 2.5 minutes (sweet spot) |

**Score anchors for calibration:**

*Specificity:*
- "I worked on a project once" → specificity: 0
- "I worked with a senior engineer named James" → specificity: 1
- "I convinced James to adopt our API design by preparing a comparison doc" → specificity: 2
- "I convinced James within two weeks; adoption cut integration time by 30%" → specificity: 3

*Conciseness (spoken length):*
- Under 30 seconds → conciseness: 0 (no room for full STAR)
- 30–75 seconds → conciseness: 1 (STAR possible but Action always thin)
- 75–90 seconds or 2.5–3 minutes → conciseness: 2 (close but slightly off)
- 90 seconds – 2.5 minutes → conciseness: 3 (sweet spot — complete STAR with details)

---

## Data Contract

All attempt data conforms to `schema/session.json`.
Read that file for field definitions, types, and constraints.
Never add fields to a session record without updating `schema/session.json` first.

### Key fields (v1.0):

| Field | Type | Notes |
|---|---|---|
| `question_id` | string | e.g. `question_1` — identifies the question |
| `user_id` | string | e.g. `user_a`, `user_b`, `user_c` — identifies the user or track |
| `attempt_number` | int | 1 to 5 per user — resets independently for each user_id |
| `date` | string | YYYY-MM-DD |
| `source_file` | string | relative path to .txt or audio input file |
| `transcription_method` | string | `text_upload` \| `gemini_audio` \| `whisper` \| `null` |
| `transcript` | string | raw answer text |
| `scores` | object | five dimensions + overall_score (sum, 0–15) |
| `strengths` | list | 1–3 specific strengths |
| `gaps` | list | 1–3 specific gaps |
| `one_specific_improvement` | string | single actionable instruction for next attempt |

**attempt_number is not globally unique.** Uniqueness comes from the combination
of `question_id` + `user_id` + `attempt_number` together.

---

## Users / Progression Tracks (v1.0)

Each user has exactly 5 attempts against question_id: question_1.

| user_id | Progression shape | Primary use |
|---|---|---|
| `user_a` | Steady improvement across all dimensions | Baseline happy path |
| `user_b` | Plateau attempts 1–3, breakthrough attempts 4–5 | Non-linear growth story |
| `user_c` | Uneven — specificity up, confidence dips mid-series | Primary demo track |

> **Assessment bias guard:** The "Progression shape" and "Primary use" columns above are
> documentation for human developers only. When running `assess_answer()` or any scoring
> function, ignore this table entirely. Score each transcript solely on its own content
> against the rubric — do not adjust scores or expectations based on a user's known
> progression shape or their designated role in the test suite.

Source text files: `question_1/{user_id}/attempt_N.txt`
Processed history: `history/{user_id}.jsonl`

---

## Folder Structure

```
/
├── AGENTS.md                              ← this file
├── schema/session.json                    ← data contract
├── question_1/                              ← question_id: question_1 source inputs
│   ├── user_a/
│   │   ├── attempt_1.txt ... attempt_5.txt
│   ├── user_b/
│   │   ├── attempt_1.txt ... attempt_5.txt
│   └── user_c/
│       ├── attempt_1.txt ... attempt_5.txt
├── history/
│   ├── user_a.jsonl                   ← processed attempt records
│   ├── user_b.jsonl
│   └── user_c.jsonl
├── db/sessions.db                         ← SQLite (Phase 4, auto-created)
├── skills/
│   ├── assessment.py                      ← assess_answer()
│   ├── progression.py                     ← save_session(), analyze_progression()
│   ├── scoresheet.py                      ← generate_scoresheet()
│   ├── guardrails.py                      ← semantic_gate(), validate_session()
│   ├── eval.py                            ← run_meta_eval()
│   └── ui.py                              ← Gradio interface, reset_progress()
├── docs/
│   ├── PROMPTS.md
│   └── architecture.md
├── tests/integration_test.py
├── outputs/                               ← generated PDFs land here
├── notebooks/capstone.ipynb
├── agent.py                               ← ADK agent definition
└── requirements.txt
```

---

## Skill Inventory

| Skill | File | Owner | Purpose |
|---|---|---|---|
| `assess_answer` | `skills/assessment.py` | P2 | Score a transcript against the rubric |
| `save_session` | `skills/progression.py` | P3 | Persist an attempt record to history |
| `analyze_progression` | `skills/progression.py` | P3 | Generate narrative from attempt history |
| `generate_scoresheet` | `skills/scoresheet.py` | P4 | Produce downloadable PDF scoresheet |
| `semantic_gate` | `skills/guardrails.py` | P2 | Validate transcript before assessment |
| `validate_session` | `skills/guardrails.py` | P2 | Confirm score integrity before saving |
| `run_meta_eval` | `skills/eval.py` | P3 | LLM-as-judge on assessment quality |
| `reset_progress` | `skills/ui.py` | P4 | Clear a user's attempts for fresh start |

---

## Context Engineering Map

| Context Type | Source | How loaded |
|---|---|---|
| Instructions | This file (AGENTS.md) | Static — loaded at agent startup |
| Knowledge | Rubric table + score anchors above | Static — part of this file |
| Memory | `history/{user_id}.jsonl` | Dynamic — loaded per user on demand |
| Examples | Few-shot transcripts in `skills/assessment.py` | Static — embedded in Skill prompt |
| Tools | Skills listed above | Dynamic — registered at agent init |
| Guardrails | `skills/guardrails.py` | Dynamic — called via before_tool hook |

---

## Guardrails

Two guardrails run on every attempt before data is persisted:

1. **Semantic gate** (`semantic_gate`) — confirms the transcript is a plausible
   behavioral interview answer before running full assessment. Blocks garbage
   input and prompt injection attempts.

2. **Score validation** (`validate_session`) — confirms `overall_score` equals
   the sum of the five dimension scores. Rejects any session record with
   inconsistent scores.

---

## Agent Flow (single attempt)

```
user submits attempt_N text
        ↓
semantic_gate(transcript)         ← guardrail: is this a real answer?
        ↓
assess_answer(transcript)         ← returns scores + strengths + gaps
        ↓
validate_session(session)         ← guardrail: does overall_score = sum?
        ↓
save_session(session)             ← appends to history/{user_id}.jsonl
        ↓
        ├── [attempt_number == 1]
        │     analyze_progression() NOT called (no history to compare)
        │     generate_scoresheet(history, narrative=None)
        │       → PDF: radar chart + scores + strengths + gaps
        │       → NO line chart, NO progression narrative
        │
        └── [attempt_number >= 2]
              analyze_progression(history)
                → generates narrative across all attempts so far
              generate_scoresheet(history, narrative)
                → PDF: radar chart + line chart + scores table
                       + progression narrative
        ↓
return (both paths):
  - text feedback: scores + strengths + gaps + one_specific_improvement
  - PDF path: downloadable scoresheet for this attempt
```

**Every attempt produces both text feedback and a downloadable PDF.**
Attempt 1 PDF contains current session scores only.
Attempt 2–5 PDFs add the progression line chart and narrative.

---

## Response Grounding Rules

These rules exist because tool outputs are the only source of truth. The
final response must be fully derived from what the tools actually returned
this turn -- never from what the flow above says *should* have happened.

1. **Never state that a file was created unless `generate_scoresheet()` was
   actually called this turn and returned a path.** Quote that exact
   returned string. Do not guess, construct, or assume a filename.

2. **Never state that a session was saved unless `save_session()` was
   actually called this turn and returned `true`.** If it returned `false`
   or raised an error, tell the user the save failed -- do not claim success.

3. **If any tool call returns `null`/`None`, or raises an error, stop and
   tell the user something went wrong instead of continuing the flow or
   describing an outcome as if it succeeded.** A missing result is not a
   silent success.

4. **When `analyze_progression()` was called and returned real data
   (`attempt_number >= 2`), the final response MUST explicitly reference
   it** -- mention the previous attempt's score, the `trend` value, and at
   least one item from `persistent_gaps` if present. A response for
   attempt 2+ that only describes the current attempt, with no comparison
   to prior attempts, is incomplete and not acceptable. **Any specific
   numeric score for a prior attempt MUST be copied exactly from
   `analyze_progression()`'s `score_history` field (a list of
   attempt_number / overall_score pairs) -- never estimated,
   rounded, or recalled from memory of the conversation.** If
   `score_history` does not contain a given attempt number, do not state a
   score for it.

5. **Every number, score, strength, gap, and improvement stated in the
   response must come directly from the dict returned by `assess_answer()`
   (or `analyze_progression()` for trend data) this turn.** Do not
   paraphrase rubric descriptions from this document as if they were the
   actual scoring result.

---

## Roadmap (out of scope for v1.0)

- Audio input and transcription (Gemini Audio or Whisper)
- Attempts beyond 5 per user
- series_2 / additional question support
- Real user authentication replacing user_a/b/c placeholders
- CI/CD pipeline for automated eval regression testing
- Cloud Run deployment via agents-cli

---

## Version

| Field | Value |
|---|---|
| Schema version | 1.0 |
| ADK version | 2.3.0 |
| Model (assessment + progression) | gemini-3.1-flash-lite |
| Model (semantic gate) | gemini-2.5-flash-lite |
| Max attempts per user | 5 |
| Active question | question_1 |
| Last updated | 2026-06-27 |
| Maintained by | Capstone Team |
