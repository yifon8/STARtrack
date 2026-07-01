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
from dotenv import load_dotenv

load_dotenv()

import gradio as gr

from skills.assessment import assess_answer
from skills.guardrails import semantic_gate, validate_session
from skills.progression import save_session, analyze_progression, _load_history
from skills.scoresheet import generate_scoresheet
from skills.eval import run_meta_eval


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
    """Run the full pipeline and return (scores_json, narrative_text, pdf_path, status_msg, meta_eval_text)."""
    gate = semantic_gate(transcript, QUESTION_ID, user_id=user_id)
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

def _handle_run_eval(user_id: str) -> str:
    """Load the most recent saved assessment for this user and run the
    LLM-as-judge meta-evaluation. Called only when the user clicks
    'Run Eval' — never triggered automatically by submit.
 
    Returns a formatted string for display in meta_eval_box."""
    history = _load_history(user_id, QUESTION_ID, "history")
    if not history:
        return "No attempts on record for this user — submit an answer first."
 
    # Use the most recent attempt as the assessment to judge.
    latest = history[-1]
    return _run_meta_eval_safe(history, latest)


def _run_meta_eval_safe(history: list[dict], assessment: dict) -> str:
    """Call run_meta_eval() for the just-saved assessment and format a short
    display string. Never raises — any failure becomes a status note instead,
    so the eval layer can never block the main scoring pipeline."""
    try:
        verdicts = run_meta_eval(history, [assessment])
    except Exception as e:
        return f"Meta-eval unavailable: {e}"
    verdict = verdicts.get(assessment["attempt_number"])
    if not verdict:
        return "Meta-eval returned no result for this attempt."
    flag = "⚠️ FLAGGED" if verdict["flagged"] else "✅ OK"
 
    return (
        f"STATUS       {flag}\n"
        f"─────────────────────────────────────────\n"
        f"ACCURACY     {verdict['accuracy_score']:.2f}/1.00\n"
        f"             {verdict['accuracy_reason']}\n"
        f"\n"
        f"ACTIONABLE   {verdict['actionability_score']:.2f}/1.00\n"
        f"             {verdict['actionability_reason']}\n"
        f"\n"
        f"JUDGE SCORE  {verdict['meta_score']:.2f}/1.00"
    )
    

def _handle_text(user_id: str, text: str):
    if not text or not text.strip():
        return None, None, None, "Please enter your answer text.", None
    return _run_pipeline(user_id, text.strip())


def _handle_file(user_id: str, file_obj):
    if file_obj is None:
        return None, None, None, "Please upload a .txt file.", None
    try:
        transcript = Path(file_obj.name).read_text(encoding="utf-8").strip()
    except Exception as e:
        return None, None, None, f"Could not read file: {e}", None
    if not transcript:
        return None, None, None, "Uploaded file is empty.", None
    return _run_pipeline(user_id, transcript)


def _handle_reset(user_id: str):
    result = reset_progress(user_id, confirm=True)
    # Also delete any PDFs for this user
    for pdf in Path("outputs").glob(f"{user_id}_attempt_*.pdf"):
        pdf.unlink(missing_ok=True)
    return result["message"]


def _make_user_tab(user_id: str):
    with gr.Tab(user_id):
        gr.HTML(
            f"""<div style="background:#E8EEF4; border-left:4px solid #2C5F8A;
                            padding:14px 18px; border-radius:4px; margin-bottom:8px;">
                <div style="font-size:1rem; font-weight:700; color:#2C5F8A;
                            text-transform:uppercase; letter-spacing:0.05em; margin-bottom:6px;">
                    Interview Question
                </div>
                <div style="font-size:1.2rem; font-weight:500; color:#1a1a1a; line-height:1.5;">
                    {QUESTION_TEXT}
                </div>
            </div>"""
        )

        with gr.Row():
            with gr.Column():
                answer_text = gr.Textbox(
                    label="Your answer (paste, type, or upload a .txt file below)",
                    lines=10,
                    placeholder="Describe a situation where you influenced someone without authority...",
                    elem_classes=["answer-box"],
                )
                submit_text_btn = gr.Button("Submit answer", variant="primary")

                gr.Markdown("**— or upload a .txt file —**")

                upload_file = gr.File(label="Upload .txt file", file_types=[".txt"])

            with gr.Column():
                status_box = gr.Textbox(label="Status", interactive=False)
                scores_box = gr.Code(label="Scores (JSON)", language="json")
                narrative_box = gr.Textbox(label="Progression narrative", lines=8, interactive=False)
                pdf_output = gr.File(label="Download scoresheet PDF")

                gr.Markdown("---")
                run_eval_btn = gr.Button("▶ Run Eval", variant="secondary")
                meta_eval_box = gr.Textbox(
                    label="Meta-eval — LLM-as-judge verdict on the latest assessment",
                    lines=6,
                    max_lines=6,
                    interactive=False,
                    placeholder="Click 'Run Eval' after submitting an answer to judge the quality of this assessment.",
                    elem_id="meta-eval-box",
                )

        def _load_file_into_box(file_obj):
            if file_obj is None:
                return gr.update()
            try:
                return gr.update(value=Path(file_obj.name).read_text(encoding="utf-8").strip())
            except Exception:
                return gr.update()

        upload_file.change(
            fn=_load_file_into_box,
            inputs=[upload_file],
            outputs=[answer_text],
        )

        submit_text_btn.click(
            fn=lambda text: _handle_text(user_id, text),
            inputs=[answer_text],
            outputs=[scores_box, narrative_box, pdf_output, status_box],
        )

        run_eval_btn.click(
            fn=lambda: _handle_run_eval(user_id),
            inputs=[],
            outputs=[meta_eval_box],
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
    """Construct and return the Gradio Blocks interface."""
    answer_css = """
    .answer-box textarea {
        font-size: 1.05rem !important;
    }
    #meta-eval-box textarea {
        min-height: 130px !important;
        max-height: 220px !important;
        overflow-y: auto !important;
        resize: none !important;
        font-family: monospace;
        font-size: 0.85rem !important;
        line-height: 1.6 !important;
    }
    """
    with gr.Blocks(title="STARtrack Interview Practice Coach", css=answer_css) as demo:
        gr.Markdown("# STARtrack — Interview Practice Coach")
        gr.Markdown(
            "Practice behavioral interview answers and track your progression across up to 5 attempts."
        )

        with gr.Tabs():
            for user_id in USERS:
                _make_user_tab(user_id)


    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
