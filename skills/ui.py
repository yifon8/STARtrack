"""
Skills: Gradio UI, reset_progress
Owner: P4
Gradio web interface for the Interview Practice Coach.
Exposes the full agent flow as a chat-style UI on http://localhost:7860.
reset_progress clears a user's attempt history for a fresh start.
"""

import json
from pathlib import Path
from typing import Optional

import gradio as gr

from skills.assessment import assess_answer
from skills.guardrails import semantic_gate, validate_session
from skills.progression import save_session, analyze_progression, _load_history
from skills.scoresheet import generate_scoresheet


USERS = ["user_a", "user_b", "user_c"]
QUESTION_ID = "question_1"
QUESTION_TEXT = "Tell me about a time you had to influence someone without authority."


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
    if not confirm:
        return {
            "success": False,
            "message": "Reset not confirmed. Pass confirm=True to proceed.",
            "deleted_file": None,
        }

    history_path = Path(history_dir) / f"{user_id}.jsonl"
    if not history_path.exists():
        return {
            "success": False,
            "message": f"No history file found for {user_id}.",
            "deleted_file": None,
        }

    history_path.unlink()
    return {
        "success": True,
        "message": f"History cleared for {user_id}.",
        "deleted_file": str(history_path),
    }


def _next_attempt_number(user_id: str) -> int:
    records = _load_history(user_id, QUESTION_ID, "history")
    return len(records) + 1


def _run_pipeline(user_id: str, transcript: str):
    """Run the full pipeline and return (scores_json, narrative_text, pdf_path, status_msg)."""
    gate = semantic_gate(transcript, QUESTION_ID)
    if not gate["passed"]:
        return None, None, None, f"Blocked by guardrail: {gate['reason']}"

    attempt_number = _next_attempt_number(user_id)
    if attempt_number > 5:
        return None, None, None, "Maximum of 5 attempts reached for this user."

    assessment = assess_answer(transcript, QUESTION_ID, user_id, attempt_number)

    validation = validate_session(assessment)
    if not validation["valid"]:
        return None, None, None, f"Validation error: {validation['reason']}"

    save_session(assessment)

    history = _load_history(user_id, QUESTION_ID, "history")

    narrative = None
    narrative_text = "First attempt — complete a second attempt to see progression analysis."
    if attempt_number >= 2:
        narrative = analyze_progression(user_id, QUESTION_ID)
        narrative_text = (
            f"Trend: {narrative['trend']}\n\n"
            f"{narrative['summary']}"
        )
        if narrative.get("persistent_gaps"):
            narrative_text += "\n\nPersistent gaps:\n" + "\n".join(
                f"• {g}" for g in narrative["persistent_gaps"]
            )

    pdf_path = generate_scoresheet(history, narrative, user_id, attempt_number)

    scores_display = json.dumps(assessment["scores"], indent=2)

    return scores_display, narrative_text, pdf_path, f"Attempt {attempt_number} scored successfully."


def _handle_text(user_id: str, text: str):
    if not text or not text.strip():
        return None, None, None, "Please enter your answer text."
    return _run_pipeline(user_id, text.strip())


def _handle_file(user_id: str, file_obj):
    if file_obj is None:
        return None, None, None, "Please upload a .txt file."
    try:
        transcript = Path(file_obj.name).read_text(encoding="utf-8").strip()
    except Exception as e:
        return None, None, None, f"Could not read file: {e}"
    if not transcript:
        return None, None, None, "Uploaded file is empty."
    return _run_pipeline(user_id, transcript)


def _handle_reset(user_id: str):
    result = reset_progress(user_id, confirm=True)
    # Also delete any PDFs for this user
    for pdf in Path("outputs").glob(f"{user_id}_attempt_*.pdf"):
        pdf.unlink(missing_ok=True)
    return result["message"]


def _make_user_tab(user_id: str):
    with gr.Tab(user_id):
        gr.Markdown(f"### Question\n{QUESTION_TEXT}")

        with gr.Row():
            with gr.Column():
                answer_text = gr.Textbox(
                    label="Your answer (paste or type)",
                    lines=8,
                    placeholder="Describe a situation where you influenced someone without authority...",
                )
                submit_text_btn = gr.Button("Submit answer", variant="primary")

                gr.Markdown("**— or —**")

                upload_file = gr.File(label="Upload .txt file", file_types=[".txt"])
                submit_file_btn = gr.Button("Submit file", variant="secondary")

            with gr.Column():
                status_box = gr.Textbox(label="Status", interactive=False)
                scores_box = gr.Code(label="Scores (JSON)", language="json")
                narrative_box = gr.Textbox(label="Progression narrative", lines=8, interactive=False)
                pdf_output = gr.File(label="Download scoresheet PDF")

        submit_text_btn.click(
            fn=lambda text: _handle_text(user_id, text),
            inputs=[answer_text],
            outputs=[scores_box, narrative_box, pdf_output, status_box],
        )

        submit_file_btn.click(
            fn=lambda f: _handle_file(user_id, f),
            inputs=[upload_file],
            outputs=[scores_box, narrative_box, pdf_output, status_box],
        )

        gr.Markdown("---")
        gr.Markdown("#### Reset progress")
        reset_btn = gr.Button(f"Clear all history for {user_id}", variant="stop")
        reset_status = gr.Textbox(label="Reset status", interactive=False)

        reset_btn.click(
            fn=lambda: _handle_reset(user_id),
            inputs=[],
            outputs=[reset_status],
        )


def build_ui():
    """Construct and return the Gradio Blocks interface.

    Wires up the full agent flow:
      semantic_gate → assess_answer → validate_session → save_session
      → [analyze_progression if attempt >= 2] → generate_scoresheet

    Returns:
        gr.Blocks: The assembled Gradio demo object. Call .launch() to serve.
    """
    with gr.Blocks(title="STARtrack Interview Practice Coach") as demo:
        gr.Markdown("# STARtrack — Interview Practice Coach")
        gr.Markdown(
            "Practice behavioral interview answers and track your progression across up to 5 attempts."
        )

        with gr.Tabs():
            for user_id in USERS:
                _make_user_tab(user_id)

        gr.Markdown("---")
        gr.Markdown(
            "**Bulk reset** — select a user and click Reset below to clear their history and PDFs."
        )
        with gr.Row():
            reset_user_dropdown = gr.Dropdown(choices=USERS, label="Select user", value=USERS[0])
            bulk_reset_btn = gr.Button("Reset selected user", variant="stop")
        bulk_reset_status = gr.Textbox(label="Reset status", interactive=False)

        bulk_reset_btn.click(
            fn=_handle_reset,
            inputs=[reset_user_dropdown],
            outputs=[bulk_reset_status],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
