"""
Skills: Gradio UI, reset_progress
Owner: P4
Gradio web interface for the Interview Practice Coach.
Exposes the full agent flow as a chat-style UI on http://localhost:7860.
reset_progress clears a user's attempt history for a fresh start.
"""

from pathlib import Path
from typing import Optional


def reset_progress(
    user_id: str,
    question_id: str = "question_1",
    history_dir: str = "history",
    confirm: bool = False,
) -> dict:
    """Delete all recorded attempts for a user so they can restart from attempt 1.

    Removes the user's .jsonl history file. Does not delete source .txt files.
    Requires confirm=True to execute — a safety guard against accidental resets.

    Args:
        user_id: Progression track identifier to reset (e.g. user_c).
        question_id: Question set scope (reserved for future multi-question support).
        history_dir: Directory containing .jsonl history files.
        confirm: Must be True for the deletion to proceed.

    Returns:
        dict with keys: success (bool), message (str), deleted_file (str | None).
    """
    pass


def build_ui():
    """Construct and return the Gradio Blocks interface.

    Wires up the full agent flow:
      semantic_gate → assess_answer → validate_session → save_session
      → [analyze_progression if attempt >= 2] → generate_scoresheet

    Returns:
        gr.Blocks: The assembled Gradio demo object. Call .launch() to serve.
    """
    pass


if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
