"""
pdf_export.py

Generates a downloadable PDF health summary from any UC's results,
using fpdf2. Per midsem VIVA commitment (item 3iii). Purely additive --
takes already-computed results as plain data, does not touch any
retrieval, matching, or urgency logic.
"""

from fpdf import FPDF
from datetime import datetime


def _clean(text: str) -> str:
    """
    fpdf2's default core fonts (Helvetica) only support Latin-1
    characters and will crash on things like em-dashes (--) or curly
    quotes, which can appear in LLM-generated text. Replaces common
    problem characters with safe equivalents rather than requiring a
    full Unicode font.
    """
    if not text:
        return ""
    replacements = {
        "\u2014": "-", "\u2013": "-",  # em dash, en dash
        "\u2018": "'", "\u2019": "'",  # curly single quotes
        "\u201c": '"', "\u201d": '"',  # curly double quotes
        "\u2026": "...",  # ellipsis
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text.encode("latin-1", "ignore").decode("latin-1")

TEAL = (8, 145, 178)
TEAL_LIGHT = (224, 242, 246)
DARK = (26, 32, 44)
GREY = (100, 110, 120)
GREY_LIGHT = (245, 246, 248)

URGENCY_COLORS = {
    "Self-Care at Home": (34, 139, 84),
    "See a Doctor Soon": (200, 140, 20),
    "See a Doctor Today": (210, 90, 20),
    "Go to Emergency": (190, 30, 30),
}


class HealthSummaryPDF(FPDF):
    def header(self):
        self.set_fill_color(*TEAL)
        self.rect(0, 0, 210, 28, "F")
        self.set_xy(12, 8)
        self.set_font("Helvetica", "B", 20)
        self.set_text_color(255, 255, 255)
        self.cell(0, 8, "Dr. Friend", ln=True)
        self.set_x(12)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(224, 242, 246)
        self.cell(0, 6, "AI-Powered Healthcare Guidance - Health Summary", ln=True)
        self.ln(10)

    def footer(self):
        self.set_y(-18)
        self.set_draw_color(220, 220, 220)
        self.line(12, self.get_y(), 198, self.get_y())
        self.set_y(-14)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*GREY)
        self.cell(0, 5, "Dr. Friend is a healthcare guidance assistant, not a replacement for professional medical advice.", ln=True, align="C")
        self.set_font("Helvetica", "", 8)
        self.cell(0, 5, f"Page {self.page_no()}", align="C")

    def section_title(self, text):
        self.set_fill_color(*TEAL_LIGHT)
        self.set_text_color(*TEAL)
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 9, "  " + text, ln=True, fill=True)
        self.ln(2)

    def body_text(self, text, size=10.5, color=DARK):
        self.set_font("Helvetica", "", size)
        self.set_text_color(*color)
        self.set_x(self.l_margin)
        self.multi_cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)


def generate_health_summary_pdf(
    condition: str,
    urgency_level: str,
    urgency_reasoning: str,
    guidance_text: str,
    sources: list = None,
    pathway: str = "",
    matched_criteria: list = None,
) -> bytes:
    """
    Builds a styled, readable PDF summarizing a single Dr. Friend
    consultation. Returns raw PDF bytes, suitable for
    st.download_button()'s `data` parameter.
    """
    condition = _clean(condition)
    urgency_level = _clean(urgency_level)
    urgency_reasoning = _clean(urgency_reasoning)
    guidance_text = _clean(guidance_text)
    pathway = _clean(pathway)
    sources = [_clean(s) for s in sources] if sources else sources
    matched_criteria = [_clean(c) for c in matched_criteria] if matched_criteria else matched_criteria

    pdf = HealthSummaryPDF()
    pdf.set_auto_page_break(auto=True, margin=22)
    pdf.add_page()

    pdf.set_fill_color(*GREY_LIGHT)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*GREY)
    generated_on = datetime.now().strftime("%d %b %Y, %I:%M %p")
    meta = f"Generated: {generated_on}" + (f"      |      Pathway: {pathway}" if pathway else "")
    pdf.cell(0, 8, "  " + meta, ln=True, fill=True)
    pdf.ln(4)

    pdf.section_title("What Might Be Going On")
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(*DARK)
    pdf.cell(0, 9, condition, ln=True)
    pdf.ln(3)

    pdf.section_title("Urgency Assessment")
    u_color = URGENCY_COLORS.get(urgency_level, (90, 90, 90))
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(*u_color)
    pdf.cell(0, 8, urgency_level, ln=True)

    box_y_start = pdf.get_y()
    pdf.set_fill_color(250, 245, 235)
    pdf.set_draw_color(*u_color)
    pdf.set_line_width(0.4)
    box_x = 12
    box_w = 186
    pdf.set_xy(box_x, box_y_start)
    pdf.set_font("Helvetica", "", 10.5)
    pdf.set_text_color(*DARK)
    lines = pdf.multi_cell(box_w - 8, 6, urgency_reasoning, dry_run=True, output="LINES")
    box_h = max(12, len(lines) * 6 + 6)
    pdf.rect(box_x, box_y_start, box_w, box_h, style="DF")
    pdf.set_xy(box_x + 4, box_y_start + 3)
    pdf.multi_cell(box_w - 8, 6, urgency_reasoning)
    pdf.set_y(box_y_start + box_h + 4)

    if matched_criteria:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*DARK)
        pdf.cell(0, 6, "Why this urgency level was flagged:", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(*GREY)
        for criterion in matched_criteria:
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 5.5, f"  -  {criterion}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    pdf.ln(2)

    pdf.section_title("Guidance")
    pdf.body_text(guidance_text)
    pdf.ln(2)

    if sources:
        pdf.section_title("Sources Consulted")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(*GREY)
        for source in sources:
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 5.5, f"  -  {source}", new_x="LMARGIN", new_y="NEXT")

    return bytes(pdf.output())