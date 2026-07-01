"""
Skill: generate_scoresheet
Owner: P4
Produces a downloadable PDF scoresheet for a completed attempt.
Attempt 1 PDFs include a radar chart and current scores only.
Attempt 2-5 PDFs add a progression line chart and narrative summary.
Uses ReportLab for PDF generation and Matplotlib (Agg backend) for charts.
"""

import matplotlib

matplotlib.use("Agg")

import io
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    ListFlowable,
    ListItem,
)

DIMENSIONS = [
    "star_structure",
    "specificity",
    "relevance",
    "confidence_language",
    "conciseness",
]

DIMENSION_LABELS = {
    "star_structure": "STAR Structure",
    "specificity": "Specificity",
    "relevance": "Relevance",
    "confidence_language": "Confidence / Language",
    "conciseness": "Conciseness",
}


def _find_attempt(history: list[dict], attempt_number: int) -> dict:
    for record in history:
        if record.get("attempt_number") == attempt_number:
            return record
    raise ValueError(
        f"generate_scoresheet: no record for attempt_number={attempt_number} "
        f"found in history (got attempts: "
        f"{[r.get('attempt_number') for r in history]})"
    )


def render_radar_chart(scores: dict) -> io.BytesIO:
    """Render a 5-axis radar chart of the current attempt's dimension scores."""
    labels = [DIMENSION_LABELS[d] for d in DIMENSIONS]
    values = [scores[d] for d in DIMENSIONS]

    angles = np.linspace(0, 2 * np.pi, len(DIMENSIONS), endpoint=False).tolist()
    values_closed = values + values[:1]
    angles_closed = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True))
    ax.plot(angles_closed, values_closed, color="#2C5F8A", linewidth=2)
    ax.fill(angles_closed, values_closed, color="#2C5F8A", alpha=0.25)
    ax.set_ylim(0, 3)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title("Current Attempt — Dimension Scores", fontsize=11, pad=20)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _render_progression_line_chart(history: list[dict]) -> io.BytesIO:
    """Render overall_score across all attempts so far."""
    sorted_history = sorted(history, key=lambda r: r["attempt_number"])
    attempt_numbers = [r["attempt_number"] for r in sorted_history]
    overall_scores = [r["scores"]["overall_score"] for r in sorted_history]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(attempt_numbers, overall_scores, marker="o", color="#2C5F8A", linewidth=2)
    ax.set_xlabel("Attempt")
    ax.set_ylabel("Overall Score (0-15)")
    ax.set_title("Progression Across Attempts", fontsize=11)
    ax.set_ylim(0, 15)
    ax.set_xticks(attempt_numbers)
    ax.grid(True, alpha=0.3)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


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
        attempt_number: Current attempt number (1-5).
        output_dir: Directory where the PDF will be saved.

    Returns:
        Relative path to the generated PDF file (e.g. outputs/user_c_attempt_3.pdf).
    """
    attempt = _find_attempt(history, attempt_number)
    scores = attempt["scores"]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    file_path = out_path / f"{user_id}_attempt_{attempt_number}.pdf"

    styles = getSampleStyleSheet()
    h1 = styles["Heading1"]
    h2 = styles["Heading2"]
    body = styles["BodyText"]
    bold = ParagraphStyle("bold", parent=body, fontName="Helvetica-Bold")

    elements = []
    elements.append(Paragraph("STARtrack — Interview Scoresheet", h1))
    elements.append(
        Paragraph(
            f"User: {user_id} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"Attempt: {attempt_number} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"Date: {attempt.get('date', '')}",
            body,
        )
    )
    elements.append(Spacer(1, 0.2 * inch))

    # --- Scores table ---
    table_data = [["Dimension", "Score (0-3)"]]
    for dim in DIMENSIONS:
        table_data.append([DIMENSION_LABELS[dim], str(scores[dim])])
    table_data.append(["Overall Score", f"{scores['overall_score']} / 15"])

    score_table = Table(table_data, colWidths=[3.5 * inch, 1.5 * inch])
    score_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C5F8A")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
                ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#E8EEF4")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN", (1, 0), (1, -1), "CENTER"),
            ]
        )
    )
    elements.append(score_table)
    elements.append(Spacer(1, 0.25 * inch))

    # --- Radar chart (current attempt) ---
    radar_buf = render_radar_chart(scores)
    elements.append(Image(radar_buf, width=3.5 * inch, height=3.5 * inch))
    elements.append(Spacer(1, 0.2 * inch))

    # --- Strengths / Gaps / Improvement ---
    elements.append(Paragraph("Strengths", h2))
    elements.append(
        ListFlowable(
            [ListItem(Paragraph(s, body)) for s in attempt.get("strengths", [])],
            bulletType="bullet",
        )
    )
    elements.append(Spacer(1, 0.15 * inch))

    elements.append(Paragraph("Gaps", h2))
    elements.append(
        ListFlowable(
            [ListItem(Paragraph(g, body)) for g in attempt.get("gaps", [])],
            bulletType="bullet",
        )
    )
    elements.append(Spacer(1, 0.15 * inch))

    elements.append(Paragraph("One Specific Improvement", h2))
    elements.append(Paragraph(attempt.get("one_specific_improvement", ""), bold))
    elements.append(Spacer(1, 0.25 * inch))

    # --- Progression section (attempt 2+) ---
    if attempt_number >= 2 and narrative:
        elements.append(Paragraph("Progression Across Attempts", h2))
        elements.append(Paragraph(f"Trend: {narrative.get('trend', 'n/a')}", bold))
        elements.append(Spacer(1, 0.1 * inch))
        elements.append(Paragraph(narrative.get("summary", ""), body))
        elements.append(Spacer(1, 0.15 * inch))

        line_buf = _render_progression_line_chart(history)
        elements.append(Image(line_buf, width=5.5 * inch, height=2.75 * inch))
        elements.append(Spacer(1, 0.15 * inch))

        persistent_gaps = narrative.get("persistent_gaps", [])
        if persistent_gaps:
            elements.append(Paragraph("Persistent Gaps", h2))
            elements.append(
                ListFlowable(
                    [ListItem(Paragraph(g, body)) for g in persistent_gaps],
                    bulletType="bullet",
                )
            )

    doc = SimpleDocTemplate(str(file_path), pagesize=letter)
    doc.build(elements)

    return str(file_path)
