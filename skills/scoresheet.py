"""
Skill: generate_scoresheet
Owner: P4
Produces a downloadable PDF scoresheet for a completed attempt.
Attempt 1 PDFs include a radar chart and current scores only.
Attempt 2–5 PDFs add a progression line chart and narrative summary.
Uses ReportLab for PDF generation and Matplotlib (Agg backend) for charts.
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path
from typing import Optional


def generate_scoresheet(
    history: list[dict],
    narrative: Optional[dict],
    user_id: str,
    attempt_number: int,
    output_dir: str = "outputs",
) -> str:
    """Render a PDF scoresheet for the current attempt and return its file path.

    For attempt_number == 1: includes radar chart + scores + strengths + gaps.
    For attempt_number >= 2: also includes a line chart and progression narrative.

    Args:
        history: List of session dicts for this user (all attempts so far).
        narrative: Output of analyze_progression(), or None for attempt 1.
        user_id: Progression track identifier used in the output filename.
        attempt_number: Current attempt number (1–5).
        output_dir: Directory where the PDF will be saved.

    Returns:
        Relative path to the generated PDF file (e.g. outputs/user_c_attempt_3.pdf).
    """
    pass
