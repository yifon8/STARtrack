# STARtrack — Architecture

## Overview

STARtrack is a single-agent system built on Google ADK 2.x.
The agent loads `AGENTS.md` as its static instruction context at startup
and delegates all work to a set of registered Skills.

## Agent Flow

```
user submits attempt text
        ↓
semantic_gate(transcript)         ← guardrail
        ↓
assess_answer(transcript)         ← rubric scoring
        ↓
validate_session(session)         ← guardrail
        ↓
save_session(session)             ← persistence
        ↓
[attempt >= 2] analyze_progression(history)
        ↓
generate_scoresheet(history, narrative)
        ↓
return text feedback + PDF path
```

## Context Engineering

| Context Type | Source | How loaded |
|---|---|---|
| Instructions | AGENTS.md | Static — loaded at agent startup |
| Knowledge | Rubric table + anchors | Static — part of AGENTS.md |
| Memory | history/{user_id}.jsonl | Dynamic — loaded per user on demand |
| Examples | Few-shot transcripts | Static — embedded in assessment.py |
| Tools | Skills | Dynamic — registered at agent init |
| Guardrails | guardrails.py | Dynamic — called via before_tool hook |

## Data Flow

```
question_1/{user_id}/attempt_N.txt
        ↓  (text_upload)
transcript (str)
        ↓  assess_answer()
session dict (schema/session.json)
        ↓  save_session()
history/{user_id}.jsonl
        ↓  analyze_progression()
narrative dict
        ↓  generate_scoresheet()
outputs/{user_id}_attempt_N.pdf
```

## Storage

- **history/**: append-only JSONL files, one per user
- **db/sessions.db**: SQLite (Phase 4, auto-created by progression.py)
- **outputs/**: generated PDF scoresheets (gitignored)
