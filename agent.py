"""
STARtrack Interview Practice Coach — ADK Agent entry point.
Run with:  python agent.py   OR   adk web .
Requires GOOGLE_API_KEY in .env (copy .env.example to get started).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from google import adk

try:
    # Works when loaded as a package submodule, e.g. by `adk web .`
    from .skills.assessment import assess_answer
    from .skills.progression import save_session, analyze_progression
    from .skills.scoresheet import generate_scoresheet
    from .skills.guardrails import semantic_gate, validate_session
    from .skills.eval import run_meta_eval
    from .skills.ui import reset_progress
except ImportError:
    # Works when run directly, e.g. `python agent.py` from inside STARtrack/
    from skills.assessment import assess_answer
    from skills.progression import save_session, analyze_progression
    from skills.scoresheet import generate_scoresheet
    from skills.guardrails import semantic_gate, validate_session
    from skills.eval import run_meta_eval
    from skills.ui import reset_progress

# Load static context from AGENTS.md at startup.
# Resolved relative to this file (not the current working directory) so it
# works no matter where `adk web` / `python agent.py` is launched from.
#
# AGENTS.md contains literal "{user_id}"-style text in folder-path examples
# (e.g. history/{user_id}.jsonl). ADK's instruction field treats curly
# braces as session-state template placeholders to substitute. Doubling the
# braces ({{ }}) is the officially documented escape, but it doesn't
# actually work in ADK 2.3.0 — its regex (`{+[^{}]*}+`) still matches
# {{user_id}} and tries to substitute it, raising
# KeyError: Context variable not found. (Tracked upstream:
# https://github.com/google/adk-python/issues/3527)
#
# Workaround: strip curly braces entirely from the instruction text so
# ADK's templating never triggers. This only affects the illustrative
# folder-path examples in AGENTS.md, not the rubric or logic.
_AGENTS_MD = (Path(__file__).parent / "AGENTS.md").read_text()
_AGENTS_MD = _AGENTS_MD.replace("{", "[").replace("}", "]")

root_agent = adk.Agent(
    name="interview_practice_coach",
    model="gemini-3.1-flash-lite",
    description="Interview Practice Coach — scores behavioral interview answers and tracks progression.",
    instruction=_AGENTS_MD,
    tools=[
        semantic_gate,
        assess_answer,
        validate_session,
        save_session,
        analyze_progression,
        generate_scoresheet,
        run_meta_eval,
        reset_progress,
    ],
)

if __name__ == "__main__":
    adk.Runner(root_agent).run_cli()
