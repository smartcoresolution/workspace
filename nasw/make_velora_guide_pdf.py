from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "VELORA_GPU_학습_운영_반출_가이드.md"
OUTPUT = ROOT / "VELORA_GPU_학습_운영_반출_가이드.pdf"
FONT_REGULAR = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"
FONT_MONO = "/usr/share/fonts/truetype/nanum/NanumGothicCoding.ttf"


def register_fonts():
    pdfmetrics.registerFont(TTFont("NanumGothic", FONT_REGULAR))
    pdfmetrics.registerFont(TTFont("NanumGothicBold", FONT_BOLD))
    pdfmetrics.registerFont(TTFont("NanumMono", FONT_MONO))


def make_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="KTitle",
            fontName="NanumGothicBold",
            fontSize=20,
            leading=28,
            textColor=HexColor("#1f2937"),
            spaceAfter=10,
            alignment=TA_LEFT,
        )
    )
    styles.add(
        ParagraphStyle(
            name="KH2",
            fontName="NanumGothicBold",
            fontSize=13,
            leading=19,
            textColor=HexColor("#111827"),
            spaceBefore=9,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="KBody",
            fontName="NanumGothic",
            fontSize=9.5,
            leading=15,
            textColor=HexColor("#111827"),
            spaceAfter=5,
            wordWrap="CJK",
        )
    )
    styles.add(
        ParagraphStyle(
            name="KCode",
            fontName="NanumMono",
            fontSize=7.5,
            leading=10,
            leftIndent=4,
            rightIndent=4,
            backColor=HexColor("#f3f4f6"),
            borderColor=HexColor("#e5e7eb"),
            borderWidth=0.5,
            borderPadding=5,
            spaceBefore=3,
            spaceAfter=7,
        )
    )
    return styles


def escape(text):
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def build_story(markdown, styles):
    story = []
    code_lines = []
    in_code = False

    def flush_code():
        nonlocal code_lines
        if code_lines:
            story.append(Preformatted("\n".join(code_lines), styles["KCode"], maxLineLength=96))
            code_lines = []

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        if line.startswith("```"):
            if in_code:
                flush_code()
                in_code = False
            else:
                in_code = True
            continue

        if in_code:
            code_lines.append(line)
            continue

        if not line:
            story.append(Spacer(1, 2.5 * mm))
            continue

        if line.startswith("# "):
            story.append(Paragraph(escape(line[2:]), styles["KTitle"]))
            story.append(Paragraph("폐쇄망 GPU 서버 운영 기준", styles["KBody"]))
            story.append(Spacer(1, 5 * mm))
            continue

        if line.startswith("## "):
            title = line[3:]
            if title.startswith("7. ") or title.startswith("12. "):
                story.append(PageBreak())
            story.append(Paragraph(escape(title), styles["KH2"]))
            continue

        if line.startswith("- "):
            story.append(Paragraph("• " + escape(line[2:]), styles["KBody"]))
            continue

        story.append(Paragraph(escape(line), styles["KBody"]))

    flush_code()
    return story


def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("NanumGothic", 8)
    canvas.setFillColor(HexColor("#6b7280"))
    canvas.drawRightString(200 * mm, 10 * mm, f"{doc.page}")
    canvas.restoreState()


def main():
    register_fonts()
    styles = make_styles()
    markdown = SOURCE.read_text(encoding="utf-8")
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=17 * mm,
        rightMargin=17 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        title="VELORA GPU 학습 운영 및 반출 가이드",
        author="VELORA",
    )
    doc.build(build_story(markdown, styles), onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(OUTPUT)


if __name__ == "__main__":
    main()
