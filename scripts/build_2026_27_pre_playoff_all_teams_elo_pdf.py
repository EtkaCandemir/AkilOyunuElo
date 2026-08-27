from __future__ import annotations

import argparse
import html
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate, Frame, LongTable, PageTemplate, Paragraph, TableStyle


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "output" / "season_2026_27_preproduction" / "team_rating_phase_summary.csv"
DEFAULT_OUTPUT = ROOT / "output" / "pdf" / "AO_2026_27_Playoff_Oncesi_Tum_Takimlar_Elo_Siralamasi.pdf"

INK = colors.HexColor("#172A3A")
TEAL = colors.HexColor("#087E8B")
PALE = colors.HexColor("#F3F6F8")
MID = colors.HexColor("#D5DFE5")
MUTED = colors.HexColor("#60717D")
WHITE = colors.white


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the complete 2026/27 AO Elo ranking before qualifying play-offs"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def register_fonts() -> tuple[str, str]:
    candidates = [
        (
            Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        ),
        (
            Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        ),
    ]
    for regular_path, bold_path in candidates:
        if regular_path.exists() and bold_path.exists():
            pdfmetrics.registerFont(TTFont("AOAllRegular", str(regular_path)))
            pdfmetrics.registerFont(TTFont("AOAllBold", str(bold_path)))
            return "AOAllRegular", "AOAllBold"
    return "Helvetica", "Helvetica-Bold"


def load_ratings(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    ratings = pd.read_csv(path)
    required = {
        "team_id",
        "team_name",
        "country_code",
        "pre_playoff_live_elo",
        "pre_playoff_rank",
        "main_entry_reset_applied_at_this_cutoff",
        "non_match_main_entry_adjustment",
    }
    missing = required.difference(ratings.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if len(ratings) != 237 or ratings["team_id"].nunique() != 237:
        raise ValueError("Expected exactly 237 unique 2026/27 teams")
    expected_ranks = list(range(1, 238))
    actual_ranks = sorted(ratings["pre_playoff_rank"].astype(int).tolist())
    if actual_ranks != expected_ranks:
        raise ValueError("Pre-play-off ranks must cover every integer from 1 to 237")
    if ratings["pre_playoff_live_elo"].isna().any():
        raise ValueError("Every team must have a finite pre-play-off Elo")
    if not pd.to_numeric(ratings["pre_playoff_live_elo"], errors="coerce").map(pd.notna).all():
        raise ValueError("Every team must have a numeric pre-play-off Elo")
    if ratings["main_entry_reset_applied_at_this_cutoff"].astype(bool).any():
        raise ValueError("Unexpected main-entry reset in the pre-play-off snapshot")
    if not (ratings["non_match_main_entry_adjustment"].abs() < 1e-12).all():
        raise ValueError("Unexpected non-match rating adjustment in the snapshot")
    return ratings.sort_values("pre_playoff_rank", kind="mergesort").reset_index(drop=True)


def styles(regular: str, bold: str) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName=bold,
            fontSize=18,
            leading=22,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=3 * mm,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["BodyText"],
            fontName=regular,
            fontSize=8.5,
            leading=11.5,
            textColor=MUTED,
            spaceAfter=4 * mm,
        ),
        "head": ParagraphStyle(
            "Head",
            parent=base["BodyText"],
            fontName=bold,
            fontSize=7.2,
            leading=8.5,
            textColor=WHITE,
            alignment=TA_CENTER,
        ),
        "cell": ParagraphStyle(
            "Cell",
            parent=base["BodyText"],
            fontName=regular,
            fontSize=7.3,
            leading=8.7,
            textColor=INK,
        ),
        "team": ParagraphStyle(
            "Team",
            parent=base["BodyText"],
            fontName=bold,
            fontSize=7.3,
            leading=8.7,
            textColor=INK,
        ),
        "elo": ParagraphStyle(
            "Elo",
            parent=base["BodyText"],
            fontName=bold,
            fontSize=7.4,
            leading=8.7,
            textColor=TEAL,
            alignment=TA_CENTER,
        ),
    }


def paragraph(value: object, style: ParagraphStyle) -> Paragraph:
    return Paragraph(html.escape(str(value), quote=False), style)


def page_frame(regular: str, bold: str):
    def draw(canvas, document):
        width, height = A4
        canvas.saveState()
        canvas.setStrokeColor(MID)
        canvas.line(14 * mm, 13 * mm, width - 14 * mm, 13 * mm)
        canvas.setFont(regular, 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(14 * mm, 9 * mm, "2026/27 | Q1-Q3 sonrası | Play-off maçları hariç")
        canvas.drawRightString(width - 14 * mm, 9 * mm, f"Sayfa {document.page}")
        if document.page > 1:
            canvas.setFont(bold, 7.4)
            canvas.setFillColor(INK)
            canvas.drawString(14 * mm, height - 10 * mm, "AkılOyunu.com | Tüm takımlar AO Elo sıralaması")
        canvas.restoreState()

    return draw


def build_pdf(input_path: Path, output_path: Path) -> Path:
    ratings = load_ratings(input_path)
    regular, bold = register_fonts()
    report_styles = styles(regular, bold)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    document = BaseDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=15 * mm,
        bottomMargin=16 * mm,
        title="AO European Elo 2026/27 Play-off Öncesi Tüm Takımlar Sıralaması",
        author="AkılOyunu.com",
        subject="237 takımın Q1-Q3 sonuçları sonrası ve play-off maçları öncesi AO Elo sıralaması",
    )
    frame = Frame(document.leftMargin, document.bottomMargin, document.width, document.height, id="main")
    document.addPageTemplates(PageTemplate(id="ranking", frames=[frame], onPage=page_frame(regular, bold)))

    rows: list[list[object]] = [
        [
            paragraph("Sıra", report_styles["head"]),
            paragraph("Takım", report_styles["head"]),
            paragraph("Ülke", report_styles["head"]),
            paragraph("AO Elo", report_styles["head"]),
        ]
    ]
    for row in ratings.itertuples(index=False):
        rows.append(
            [
                paragraph(int(row.pre_playoff_rank), report_styles["cell"]),
                paragraph(row.team_name, report_styles["team"]),
                paragraph(row.country_code, report_styles["cell"]),
                paragraph(f"{row.pre_playoff_live_elo:.1f}", report_styles["elo"]),
            ]
        )

    ranking = LongTable(
        rows,
        colWidths=[16 * mm, 111 * mm, 20 * mm, 31 * mm],
        repeatRows=1,
    )
    ranking.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), INK),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("GRID", (0, 0), (-1, -1), 0.35, MID),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PALE]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 1), (0, -1), "CENTER"),
                ("ALIGN", (2, 1), (-1, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 2.6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6),
            ]
        )
    )

    story: list[object] = [
        Paragraph("2026/27 Play-off Öncesi Tüm Takımlar AO Elo Sıralaması", report_styles["title"]),
        Paragraph(
            "237 takımın tamamı. Q1, Q2 ve Q3 sonuçları AO Live Elo'ya işlenmiştir; qualifying play-off maçları dahil değildir. "
            "Takımlar güncel Elo değerine göre 1'den 237'ye sıralanmıştır.",
            report_styles["subtitle"],
        ),
        ranking,
    ]
    document.build(story)
    return output_path


def main() -> None:
    args = parse_args()
    print(build_pdf(args.input.resolve(), args.output.resolve()))


if __name__ == "__main__":
    main()
