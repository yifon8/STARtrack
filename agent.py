"""
STARtrack Interview Practice Coach — ADK Agent entry point.
Run with:  python agent.py   OR   adk web .
Requires GOOGLE_API_KEY in .env (copy .env.example to get started).
"""

import os
from dotenv import load_dotenv

load_dotenv()

from google import adk

from skills.assessment import assess_answer
from skills.progression import save_session, analyze_progression
from skills.scoresheet import generate_scoresheet
from skills.guardrails import semantic_gate, validate_session
from skills.eval import run_meta_eval
from skills.ui import reset_progress

# Load static context from AGENTS.md at startup
_AGENTS_MD = open("AGENTS.md", encoding="utf-8").read()

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
