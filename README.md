# STARtrack — Interview Practice Coach

A behavioral interview coaching agent built with Google ADK.
Scores answers against a five-dimension STAR rubric, tracks progression
across five attempts, and generates a downloadable PDF scoresheet per session.

---

## Quick Start

```bash
git clone <repo> && cd STARtrack
cp .env.example .env          # add your GOOGLE_API_KEY
pip install -r requirements.txt
python agent.py               # CLI mode
# OR
adk web .                     # ADK browser UI on http://localhost:8000
# OR
python run_ui.py              # Customized Gradio UI on http://localhost:7860
```

---

## Project Structure

See `AGENTS.md` for the full folder layout, Skill inventory, rubric definitions,
and data contract.

---

## Active Question

`question_1` — *"Tell me about a time you had to influence someone without authority."*

---

## Users / Progression Tracks

| user_id | Description |
|---------|-------------|
| `user_a` | Steady improvement across all dimensions |
| `user_b` | Plateau attempts 1–3, breakthrough attempts 4–5 |
| `user_c` | Uneven — specificity up, confidence dips mid-series |

---

## Skills

| Skill | File | Purpose |
|-------|------|---------|
| `assess_answer` | `skills/assessment.py` | Score transcript against rubric |
| `save_session` | `skills/progression.py` | Persist attempt to history |
| `analyze_progression` | `skills/progression.py` | Narrative across attempts |
| `generate_scoresheet` | `skills/scoresheet.py` | PDF scoresheet |
| `semantic_gate` | `skills/guardrails.py` | Validate input before assessment |
| `validate_session` | `skills/guardrails.py` | Confirm score integrity |
| `run_meta_eval` | `skills/eval.py` | LLM-as-judge on assessment quality |
| `reset_progress` | `skills/ui.py` | Clear user history |

---

## Environment

- Python 3.10+
- Google ADK 2.3.0
- Model (assessment + progression): `gemini-3.1-flash-lite`
- Model (semantic gate): `gemini-2.5-flash-lite`

---

## License

For capstone / educational use only.
