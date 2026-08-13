from __future__ import annotations

import shutil
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Iterable, Sequence

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output" / "pdf"
DOCS_DIR = ROOT / "docs" / "pdf"

NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#246B9E")
TEAL = colors.HexColor("#247A73")
GREEN = colors.HexColor("#2F7D5C")
AMBER = colors.HexColor("#B8791D")
RED = colors.HexColor("#A94442")
INK = colors.HexColor("#1F2933")
MUTED = colors.HexColor("#5F6B78")
LINE = colors.HexColor("#D5DEE7")
PANEL = colors.HexColor("#F4F7FA")
PALE_BLUE = colors.HexColor("#EAF3F8")
PALE_GREEN = colors.HexColor("#EAF5F1")
PALE_AMBER = colors.HexColor("#FFF4E2")
PALE_RED = colors.HexColor("#FBECEC")

_REGULAR_CANDIDATES = (
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
)
_BOLD_CANDIDATES = (
    Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
)


def _first_existing(paths: Iterable[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("A Unicode-capable PDF font could not be found")


def register_fonts() -> None:
    if "AORegular" in pdfmetrics.getRegisteredFontNames():
        return
    pdfmetrics.registerFont(TTFont("AORegular", str(_first_existing(_REGULAR_CANDIDATES))))
    pdfmetrics.registerFont(TTFont("AOBold", str(_first_existing(_BOLD_CANDIDATES))))
    pdfmetrics.registerFontFamily("AO", normal="AORegular", bold="AOBold")


def styles() -> dict[str, ParagraphStyle]:
    register_fonts()
    base = getSampleStyleSheet()
    return {
        "Title": ParagraphStyle(
            "AO_Title",
            parent=base["Title"],
            fontName="AOBold",
            fontSize=24,
            leading=29,
            textColor=NAVY,
            alignment=TA_CENTER,
            spaceAfter=7,
        ),
        "Subtitle": ParagraphStyle(
            "AO_Subtitle",
            parent=base["Normal"],
            fontName="AORegular",
            fontSize=13.5,
            leading=18,
            textColor=BLUE,
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
        "Cover": ParagraphStyle(
            "AO_Cover",
            parent=base["BodyText"],
            fontName="AORegular",
            fontSize=9.3,
            leading=13.5,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceAfter=6,
        ),
        "H1": ParagraphStyle(
            "AO_H1",
            parent=base["Heading1"],
            fontName="AOBold",
            fontSize=14.2,
            leading=18,
            textColor=NAVY,
            spaceBefore=10,
            spaceAfter=7,
            keepWithNext=True,
        ),
        "H2": ParagraphStyle(
            "AO_H2",
            parent=base["Heading2"],
            fontName="AOBold",
            fontSize=11.1,
            leading=14,
            textColor=TEAL,
            spaceBefore=7,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "Body": ParagraphStyle(
            "AO_Body",
            parent=base["BodyText"],
            fontName="AORegular",
            fontSize=8.8,
            leading=12.6,
            textColor=INK,
            spaceAfter=6,
        ),
        "Bullet": ParagraphStyle(
            "AO_Bullet",
            parent=base["BodyText"],
            fontName="AORegular",
            fontSize=8.6,
            leading=12.2,
            leftIndent=13,
            firstLineIndent=-8,
            bulletIndent=3,
            textColor=INK,
            spaceAfter=3,
        ),
        "Formula": ParagraphStyle(
            "AO_Formula",
            parent=base["Code"],
            fontName="AORegular",
            fontSize=8.1,
            leading=11.5,
            leftIndent=8,
            rightIndent=8,
            textColor=NAVY,
            backColor=PANEL,
            borderColor=LINE,
            borderWidth=0.45,
            borderPadding=6,
            spaceBefore=3,
            spaceAfter=5,
        ),
        "Small": ParagraphStyle(
            "AO_Small",
            parent=base["BodyText"],
            fontName="AORegular",
            fontSize=7.2,
            leading=9.4,
            textColor=INK,
        ),
        "SmallBold": ParagraphStyle(
            "AO_SmallBold",
            parent=base["BodyText"],
            fontName="AOBold",
            fontSize=7.3,
            leading=9.5,
            textColor=NAVY,
        ),
        "CalloutTitle": ParagraphStyle(
            "AO_CalloutTitle",
            parent=base["BodyText"],
            fontName="AOBold",
            fontSize=9.0,
            leading=11,
            textColor=NAVY,
            spaceAfter=3,
        ),
    }


def para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(text).replace("\n", "<br/>"), style)


def h1(text: str, s: dict[str, ParagraphStyle]) -> Paragraph:
    return para(text, s["H1"])


def h2(text: str, s: dict[str, ParagraphStyle]) -> Paragraph:
    return para(text, s["H2"])


def body(text: str, s: dict[str, ParagraphStyle]) -> Paragraph:
    return para(text, s["Body"])


def bullets(items: Sequence[str], s: dict[str, ParagraphStyle]) -> list[Paragraph]:
    return [Paragraph(f"• {escape(item)}", s["Bullet"]) for item in items]


def formula(lines: Sequence[str] | str, s: dict[str, ParagraphStyle]) -> Paragraph:
    values = [lines] if isinstance(lines, str) else list(lines)
    return Paragraph("<br/>".join(escape(str(line)) for line in values), s["Formula"])


def table(
    rows: Sequence[Sequence[object]],
    widths: Sequence[float],
    s: dict[str, ParagraphStyle],
    *,
    header: bool = True,
    repeat_header: bool = True,
) -> Table:
    wrapped: list[list[Paragraph]] = []
    for row_index, row in enumerate(rows):
        style = s["SmallBold"] if header and row_index == 0 else s["Small"]
        wrapped.append([para(str(value), style) for value in row])
    result = Table(
        wrapped,
        colWidths=list(widths),
        repeatRows=1 if header and repeat_header else 0,
        hAlign="LEFT",
    )
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4.5),
        ("TOPPADDING", (0, 0), (-1, -1), 4.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.2),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5EEF5")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ]
        )
    else:
        commands.append(("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]))
        commands.append(("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF1F6")))
    result.setStyle(TableStyle(commands))
    return result


def callout(
    title: str,
    text: str,
    s: dict[str, ParagraphStyle],
    *,
    tone: str = "blue",
) -> Table:
    palette = {
        "blue": (PALE_BLUE, BLUE),
        "green": (PALE_GREEN, GREEN),
        "amber": (PALE_AMBER, AMBER),
        "red": (PALE_RED, RED),
    }
    background, border = palette[tone]
    result = Table(
        [[para(title, s["CalloutTitle"])], [body(text, s)]],
        colWidths=[16.45 * cm],
        hAlign="LEFT",
    )
    result.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 0.7, border),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return result


def keep(items: Sequence[object]) -> KeepTogether:
    return KeepTogether(list(items))


def page_break() -> PageBreak:
    return PageBreak()


@dataclass(frozen=True)
class PdfSpec:
    filename: str
    title: str
    subtitle: str
    version: str
    document_date: str
    subject: str


def cover(
    spec: PdfSpec,
    metadata: Sequence[Sequence[object]],
    s: dict[str, ParagraphStyle],
    *,
    summary_title: str,
    summary: str,
) -> list[object]:
    return [
        Spacer(1, 1.0 * cm),
        para(spec.title, s["Title"]),
        para(spec.subtitle, s["Subtitle"]),
        para("AkılOyunu.com | UEFA Kulüp Turnuvaları", s["Cover"]),
        Spacer(1, 0.35 * cm),
        table(metadata, [5.1 * cm, 11.35 * cm], s, header=False),
        Spacer(1, 0.4 * cm),
        callout(summary_title, summary, s, tone="blue"),
        page_break(),
    ]


def build_pdf(spec: PdfSpec, story: Sequence[object]) -> tuple[Path, Path]:
    register_fonts()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / spec.filename
    docs_path = DOCS_DIR / spec.filename

    doc = BaseDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=1.35 * cm,
        rightMargin=1.35 * cm,
        topMargin=1.35 * cm,
        bottomMargin=1.30 * cm,
        title=f"{spec.title} - {spec.subtitle}",
        author="Akıl Oyunu",
        subject=spec.subject,
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")

    def footer(canvas, current_doc) -> None:
        canvas.saveState()
        canvas.setStrokeColor(LINE)
        canvas.setLineWidth(0.4)
        canvas.line(current_doc.leftMargin, 0.95 * cm, A4[0] - current_doc.rightMargin, 0.95 * cm)
        canvas.setFont("AORegular", 7.0)
        canvas.setFillColor(MUTED)
        canvas.drawString(current_doc.leftMargin, 0.58 * cm, spec.version)
        canvas.drawCentredString(A4[0] / 2, 0.58 * cm, spec.document_date)
        canvas.drawRightString(A4[0] - current_doc.rightMargin, 0.58 * cm, f"Sayfa {current_doc.page}")
        canvas.restoreState()

    doc.addPageTemplates([PageTemplate(id="normal", frames=[frame], onPage=footer)])
    doc.build(list(story))
    shutil.copyfile(output_path, docs_path)
    return output_path, docs_path
