from __future__ import annotations

import argparse
import html
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    LongTable,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "season_2026_27_preproduction"
OUTPUT_ROOT = ROOT / "output" / "season_2026_27_preproduction"
DEFAULT_OUTPUT = ROOT / "output" / "pdf" / "AO_2026_27_Playoff_Oncesi_Elo_Siralamasi.pdf"

INK = colors.HexColor("#172A3A")
BLUE = colors.HexColor("#2D5F8B")
TEAL = colors.HexColor("#087E8B")
GOLD = colors.HexColor("#D39A22")
GREEN = colors.HexColor("#2C7A4B")
RED = colors.HexColor("#B84A4A")
PALE = colors.HexColor("#F3F6F8")
MID = colors.HexColor("#D5DFE5")
MUTED = colors.HexColor("#60717D")
WHITE = colors.white

ROUND_KEYS = {
    "1st Qualifying Round": "Q1",
    "2nd Qualifying Round": "Q2",
    "3rd Qualifying Round": "Q3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the 2026/27 AO Elo ranking immediately before qualifying play-offs"
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def esc(value: object) -> str:
    return html.escape(str(value), quote=False)


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
            pdfmetrics.registerFont(TTFont("AORegular", str(regular_path)))
            pdfmetrics.registerFont(TTFont("AOBold", str(bold_path)))
            return "AORegular", "AOBold"
    return "Helvetica", "Helvetica-Bold"


def build_styles(regular: str, bold: str) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "cover_title": ParagraphStyle(
            "CoverTitle",
            parent=base["Title"],
            fontName=bold,
            fontSize=24,
            leading=29,
            textColor=WHITE,
            alignment=TA_LEFT,
            spaceAfter=8,
        ),
        "cover_subtitle": ParagraphStyle(
            "CoverSubtitle",
            parent=base["Normal"],
            fontName=regular,
            fontSize=11.5,
            leading=16,
            textColor=colors.HexColor("#D9EBEF"),
            spaceAfter=4,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName=bold,
            fontSize=16,
            leading=20,
            textColor=INK,
            spaceBefore=2,
            spaceAfter=7,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName=bold,
            fontSize=11,
            leading=14,
            textColor=TEAL,
            spaceBefore=7,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName=regular,
            fontSize=8.6,
            leading=12.2,
            textColor=INK,
            spaceAfter=5,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName=regular,
            fontSize=7.2,
            leading=9.5,
            textColor=INK,
            spaceAfter=3,
        ),
        "callout": ParagraphStyle(
            "Callout",
            parent=base["BodyText"],
            fontName=bold,
            fontSize=9.0,
            leading=12.5,
            textColor=INK,
            borderColor=TEAL,
            borderWidth=1,
            borderPadding=7,
            backColor=colors.HexColor("#EAF5F5"),
            spaceBefore=4,
            spaceAfter=7,
        ),
        "table": ParagraphStyle(
            "Table",
            parent=base["BodyText"],
            fontName=regular,
            fontSize=6.3,
            leading=7.7,
            textColor=INK,
        ),
        "table_bold": ParagraphStyle(
            "TableBold",
            parent=base["BodyText"],
            fontName=bold,
            fontSize=6.3,
            leading=7.7,
            textColor=INK,
        ),
        "table_head": ParagraphStyle(
            "TableHead",
            parent=base["BodyText"],
            fontName=bold,
            fontSize=6.1,
            leading=7.4,
            textColor=WHITE,
            alignment=TA_CENTER,
        ),
        "result": ParagraphStyle(
            "Result",
            parent=base["BodyText"],
            fontName=regular,
            fontSize=6.0,
            leading=7.5,
            textColor=INK,
        ),
    }


def p(text: object, style: ParagraphStyle) -> Paragraph:
    return Paragraph(esc(text), style)


def page_frame(regular: str, bold: str, page_size: tuple[float, float], cutoff_label: str):
    def draw(canvas, document):
        width, height = page_size
        canvas.saveState()
        if document.page > 1:
            canvas.setFont(bold, 7.4)
            canvas.setFillColor(INK)
            canvas.drawString(12 * mm, height - 9.5 * mm, "AkılOyunu.com | 2026/27 play-off öncesi AO Elo")
        canvas.setStrokeColor(MID)
        canvas.line(12 * mm, 11.5 * mm, width - 12 * mm, 11.5 * mm)
        canvas.setFont(regular, 6.8)
        canvas.setFillColor(MUTED)
        canvas.drawString(12 * mm, 7.8 * mm, f"Q1-Q3 tamamlandı | Kesim: {cutoff_label} | Play-off sonuçları hariç")
        canvas.drawRightString(width - 12 * mm, 7.8 * mm, f"Sayfa {document.page}")
        canvas.restoreState()

    return draw


def load_data(data_root: Path, output_root: Path) -> dict[str, object]:
    paths = {
        "ratings": output_root / "playoff_pre_match_team_ratings.csv",
        "universe": output_root / "team_rating_phase_summary.csv",
        "matches": data_root / "matches_completed.csv",
        "fixtures": data_root / "fixtures_upcoming.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing report inputs: {missing}")

    ratings = pd.read_csv(paths["ratings"])
    universe = pd.read_csv(paths["universe"])
    matches = pd.read_csv(paths["matches"])
    fixtures = pd.read_csv(paths["fixtures"])

    if len(ratings) != 86 or ratings["team_id"].nunique() != 86:
        raise ValueError("Expected exactly 86 unique qualifying play-off participants")
    if len(universe) != 237 or universe["team_id"].nunique() != 237:
        raise ValueError("Expected exactly 237 unique 2026/27 AO teams")
    if len(matches) != 342 or matches["match_id"].nunique() != 342:
        raise ValueError("Expected exactly 342 unique completed Q1-Q3 matches")
    if not ratings["upcoming_playoff_participant"].astype(bool).all():
        raise ValueError("Ratings file contains a non-play-off participant")

    fixture_team_competition: dict[str, set[str]] = defaultdict(set)
    for row in fixtures.itertuples(index=False):
        fixture_team_competition[str(row.home_team_id)].add(str(row.competition))
        fixture_team_competition[str(row.away_team_id)].add(str(row.competition))
    if set(fixture_team_competition) != set(ratings["team_id"].astype(str)):
        raise ValueError("Play-off fixture participants do not match the rating snapshot")
    if any(len(values) != 1 for values in fixture_team_competition.values()):
        raise ValueError("A play-off participant is mapped to multiple competitions")

    ratings = ratings.sort_values(
        ["pre_playoff_live_elo", "team_id"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    ratings["playoff_rank"] = ratings.index + 1
    ratings["playoff_competition"] = ratings["team_id"].astype(str).map(
        lambda team_id: next(iter(fixture_team_competition[team_id]))
    )

    cutoff_utc = pd.to_datetime(matches["kickoff_utc"], utc=True).max().to_pydatetime()
    cutoff_istanbul = cutoff_utc.astimezone(ZoneInfo("Europe/Istanbul"))

    return {
        "ratings": ratings,
        "universe": universe,
        "matches": matches.sort_values(["kickoff_utc", "match_id"], kind="mergesort"),
        "fixtures": fixtures,
        "cutoff_utc": cutoff_utc,
        "cutoff_istanbul": cutoff_istanbul,
    }


def Turkish_date(value: datetime) -> str:
    months = {
        1: "Ocak", 2: "Şubat", 3: "Mart", 4: "Nisan", 5: "Mayıs", 6: "Haziran",
        7: "Temmuz", 8: "Ağustos", 9: "Eylül", 10: "Ekim", 11: "Kasım", 12: "Aralık",
    }
    return f"{value.day} {months[value.month]} {value.year} {value:%H:%M} TSİ"


def format_score_event(row: object, team_id: str) -> str:
    home = str(row.home_team_id) == team_id
    opponent = row.away_team_name if home else row.home_team_name
    goals_for = int(row.home_goals if home else row.away_goals)
    goals_against = int(row.away_goals if home else row.home_goals)
    venue = "İ" if home else "D"
    text = f"{goals_for}-{goals_against} {opponent} ({venue})"
    if bool(row.decided_on_penalties):
        home_pen = row.home_penalty_goals
        away_pen = row.away_penalty_goals
        if pd.notna(home_pen) and pd.notna(away_pen):
            pen_for = int(home_pen if home else away_pen)
            pen_against = int(away_pen if home else home_pen)
            text += f" [pen. {pen_for}-{pen_against}]"
        else:
            text += " [pen.]"
    return text


def round_details(matches: pd.DataFrame, ratings: pd.DataFrame) -> dict[tuple[str, str], dict[str, object]]:
    details: dict[tuple[str, str], dict[str, object]] = {}
    rating_by_team = ratings.set_index("team_id")
    playoff_ids = set(ratings["team_id"].astype(str))
    appearances: dict[tuple[str, str], list[object]] = defaultdict(list)

    for row in matches.itertuples(index=False):
        round_key = ROUND_KEYS.get(str(row.round))
        if round_key is None:
            raise ValueError(f"Unsupported completed round: {row.round}")
        for team_id in (str(row.home_team_id), str(row.away_team_id)):
            if team_id in playoff_ids:
                appearances[(team_id, round_key)].append(row)

    for team_id in playoff_ids:
        for round_key in ("Q1", "Q2", "Q3"):
            rows = appearances.get((team_id, round_key), [])
            wins = draws = losses = goals_for = goals_against = 0
            competitions: list[str] = []
            score_events: list[str] = []
            for row in rows:
                home = str(row.home_team_id) == team_id
                gf = int(row.home_goals if home else row.away_goals)
                ga = int(row.away_goals if home else row.home_goals)
                goals_for += gf
                goals_against += ga
                wins += int(gf > ga)
                draws += int(gf == ga)
                losses += int(gf < ga)
                competitions.append(str(row.competition))
                score_events.append(format_score_event(row, team_id))
            summary_row = rating_by_team.loc[team_id]
            end_elo = float(summary_row[f"{round_key.lower()}_end_live_elo"])
            details[(team_id, round_key)] = {
                "matches": len(rows),
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "goals_for": goals_for,
                "goals_against": goals_against,
                "competitions": "/".join(dict.fromkeys(competitions)),
                "scores": score_events,
                "end_elo": end_elo,
            }
    return details


def phase_elo(value: float, matches: int) -> str:
    return "-" if matches == 0 else f"{value:.1f}"


def delta_text(value: float) -> str:
    return f"{value:+.1f}"


def rank_delta_text(value: float) -> str:
    return f"{int(value):+d}"


def build_pdf(data_root: Path, output_root: Path, output_path: Path) -> Path:
    data = load_data(data_root, output_root)
    ratings = data["ratings"]
    matches = data["matches"]
    assert isinstance(ratings, pd.DataFrame)
    assert isinstance(matches, pd.DataFrame)
    details = round_details(matches, ratings)

    regular, bold = register_fonts()
    styles = build_styles(regular, bold)
    page_size = landscape(A4)
    cutoff_label = Turkish_date(data["cutoff_istanbul"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = BaseDocTemplate(
        str(output_path),
        pagesize=page_size,
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=16 * mm,
        bottomMargin=15 * mm,
        title="AO European Elo 2026/27 Play-off Öncesi Sıralama",
        author="AkılOyunu.com",
        subject="Q1-Q3 sonuçları sonrası 86 play-off takımının AO Live Elo sıralaması",
    )
    frame = Frame(document.leftMargin, document.bottomMargin, document.width, document.height, id="main")
    document.addPageTemplates(
        PageTemplate(
            id="report",
            frames=[frame],
            onPage=page_frame(regular, bold, page_size, cutoff_label),
        )
    )

    story: list[object] = []
    story.extend(cover_section(data, ratings, matches, styles))
    story.extend(method_section(styles))
    story.extend(ranking_section(ratings, details, styles))
    story.extend(results_section(ratings, details, styles))
    story.extend(audit_section(data, ratings, matches, details, styles))
    document.build(story)
    return output_path


def cover_section(
    data: dict[str, object], ratings: pd.DataFrame, matches: pd.DataFrame, styles: dict[str, ParagraphStyle]
) -> list[object]:
    cutoff_label = Turkish_date(data["cutoff_istanbul"])
    band = Table(
        [
            [Paragraph("AO European Elo", styles["cover_title"])],
            [Paragraph("2026/27 Play-off Öncesi Güncel Elo Sıralaması", styles["cover_subtitle"])],
            [Paragraph("Q1, Q2 ve Q3 sonuçları işlendi - play-off maçları henüz dahil edilmedi", styles["cover_subtitle"])],
            [Paragraph(f"86 play-off takımı  |  342 tamamlanmış eleme maçı  |  {cutoff_label}", styles["cover_subtitle"])],
        ],
        colWidths=[257 * mm],
        rowHeights=[29 * mm, 14 * mm, 13 * mm, 13 * mm],
    )
    band.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 2), INK),
                ("BACKGROUND", (0, 3), (-1, 3), TEAL),
                ("LEFTPADDING", (0, 0), (-1, -1), 10 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 3 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3 * mm),
            ]
        )
    )

    q_counts = matches.assign(round_key=matches["round"].map(ROUND_KEYS)).groupby("round_key").size()
    comp_counts = ratings.groupby("playoff_competition").size()
    cards = Table(
        [
            [p("Q1 maçları", styles["small"]), p("Q2 maçları", styles["small"]), p("Q3 maçları", styles["small"]), p("Play-off takımları", styles["small"])],
            [p(str(int(q_counts.get("Q1", 0))), styles["h1"]), p(str(int(q_counts.get("Q2", 0))), styles["h1"]), p(str(int(q_counts.get("Q3", 0))), styles["h1"]), p(str(len(ratings)), styles["h1"])],
            [p("UCL / UEL / UECL", styles["small"]), p("UCL / UEL / UECL", styles["small"]), p("UCL / UEL / UECL", styles["small"]), p(f"{int(comp_counts.get('UCL', 0))} / {int(comp_counts.get('UEL', 0))} / {int(comp_counts.get('UECL', 0))}", styles["small"])],
        ],
        colWidths=[62.5 * mm] * 4,
    )
    cards.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE),
                ("BOX", (0, 0), (-1, -1), 0.7, MID),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, MID),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )

    return [
        band,
        Spacer(1, 9 * mm),
        cards,
        Spacer(1, 8 * mm),
        Paragraph(
            "Bu rapor, play-off eşleşmelerinde yer alan güncel 86 takımı kendi aralarında sıralar. "
            "<b>Genel sıra</b> ise aynı anda 2026/27 AO sezon havuzundaki 237 takımın tamamına göre hesaplanmıştır.",
            styles["callout"],
        ),
        Paragraph(
            "Güncel Elo, AO First Elo üzerine Q1-Q3 maçlarının kronolojik ve sıfır-toplamlı Power Elo güncellemelerinin "
            "eklenmiş halidir. Play-off karşılaşmalarının hiçbir sonucu veya maç sonrası bilgisi bu kesime girmemiştir.",
            styles["body"],
        ),
        PageBreak(),
    ]


def method_section(styles: dict[str, ParagraphStyle]) -> list[object]:
    params = [
        [p("Bileşen", styles["table_head"]), p("Aktif değer", styles["table_head"]), p("Bu rapordaki işlev", styles["table_head"])],
        [p("AO First Elo", styles["table_bold"]), p("Exposure cap 0.65", styles["table"]), p("Her takımın 2026/27 sezon başlangıç gücü", styles["table"])],
        [p("Temel K", styles["table_bold"]), p("103.980986", styles["table"]), p("Ana turnuva için referans öğrenme hızı", styles["table"])],
        [p("Q1 efektif K", styles["table_bold"]), p("0.20 × K", styles["table"]), p("Q1 maç değişiminin kontrollü ağırlığı", styles["table"])],
        [p("Q2 efektif K", styles["table_bold"]), p("0.275 × K", styles["table"]), p("Q2 maç değişiminin kontrollü ağırlığı", styles["table"])],
        [p("Q3 efektif K", styles["table_bold"]), p("0.35 × K", styles["table"]), p("Q3 maç değişiminin kontrollü ağırlığı", styles["table"])],
        [p("Play-off efektif K", styles["table_bold"]), p("0.425 × K", styles["table"]), p("Bu raporda henüz uygulanmadı", styles["table"])],
        [p("Gol farkı", styles["table_bold"]), p("alpha 0.15, tau 300, cap 4", styles["table"]), p("Büyük galibiyetlere kontrollü ek kanıt", styles["table"])],
        [p("xG", styles["table_bold"]), p("ratio 0.30, scale 1.25", styles["table"]), p("Uygun iki taraflı veri varsa; yoksa gol farkı fallback", styles["table"])],
        [p("Ana aşama geçişi", styles["table_bold"]), p("Reset yok", styles["table"]), p("Maç olmadan puan düşürülmez veya artırılmaz", styles["table"])],
    ]
    table = Table(params, colWidths=[46 * mm, 62 * mm, 145 * mm], repeatRows=1)
    table.setStyle(common_table_style())
    return [
        Paragraph("1. Hesap Sözleşmesi", styles["h1"]),
        Paragraph(
            "Eleme turu ağırlığı, turun sonunda toplu bir kesinti olarak uygulanmaz. Her maçın Power Elo değişimi, "
            "ilgili turun efektif K çarpanıyla doğrudan hesaplanır. Bu nedenle kullanıcıya gösterilen Elo tur geçişinde "
            "yapay biçimde geriye düşmez.",
            styles["body"],
        ),
        table,
        Spacer(1, 4 * mm),
        Paragraph("Sıralama alanları", styles["h2"]),
        Paragraph(
            "<b>PO sıra:</b> yalnız bu rapordaki 86 play-off takımı arasındaki sıra. "
            "<b>Genel sıra:</b> 237 takımlık 2026/27 AO havuzundaki sıra. "
            "<b>Sıra Δ:</b> AO First genel sırası eksi play-off öncesi genel sıra; pozitif değer yükseliştir. "
            "Q1/Q2/Q3 Elo kolonları yalnız takım ilgili turda maç oynadıysa gösterilir.",
            styles["body"],
        ),
        Paragraph(
            "Not: Production ML + Domestic Poisson katmanı maç olasılığı üretir; AO Live Elo state'ine geri beslenmez. "
            "Bu yüzden bu sıralama yalnız rating motorunun sonuçlarını gösterir.",
            styles["callout"],
        ),
        PageBreak(),
    ]


def ranking_section(
    ratings: pd.DataFrame,
    details: dict[tuple[str, str], dict[str, object]],
    styles: dict[str, ParagraphStyle],
) -> list[object]:
    header = [
        "PO sıra", "Genel", "Takım", "Ülke", "PO", "AO First", "Q1 Elo", "Q2 Elo", "Q3 Elo",
        "Güncel Elo", "Elo Δ", "Sıra Δ",
    ]
    rows: list[list[object]] = [[p(value, styles["table_head"]) for value in header]]
    for row in ratings.itertuples(index=False):
        team_id = str(row.team_id)
        rows.append(
            [
                p(int(row.playoff_rank), styles["table_bold"]),
                p(int(row.pre_playoff_rank), styles["table"]),
                p(row.team_name, styles["table_bold"]),
                p(row.country_code, styles["table"]),
                p(row.playoff_competition, styles["table_bold"]),
                p(f"{row.ao_first_elo:.1f}", styles["table"]),
                p(phase_elo(row.q1_end_live_elo, int(row.q1_matches)), styles["table"]),
                p(phase_elo(row.q2_end_live_elo, int(row.q2_matches)), styles["table"]),
                p(phase_elo(row.q3_end_live_elo, int(row.q3_matches)), styles["table"]),
                p(f"{row.pre_playoff_live_elo:.1f}", styles["table_bold"]),
                p(delta_text(row.pre_playoff_live_elo_change), styles["table"]),
                p(rank_delta_text(row.pre_playoff_rank_change), styles["table"]),
            ]
        )

    table = LongTable(
        rows,
        colWidths=[12 * mm, 13 * mm, 50 * mm, 14 * mm, 14 * mm, 22 * mm, 20 * mm, 20 * mm, 20 * mm, 24 * mm, 17 * mm, 17 * mm],
        repeatRows=1,
    )
    style = common_table_style()
    style.add("ALIGN", (0, 1), (1, -1), "CENTER")
    style.add("ALIGN", (3, 1), (-1, -1), "CENTER")
    for index, row in enumerate(ratings.itertuples(index=False), start=1):
        if row.pre_playoff_live_elo_change > 0.0005:
            style.add("TEXTCOLOR", (10, index), (10, index), GREEN)
        elif row.pre_playoff_live_elo_change < -0.0005:
            style.add("TEXTCOLOR", (10, index), (10, index), RED)
    table.setStyle(style)
    return [
        Paragraph("2. Play-off Öncesi Güncel Elo Sıralaması", styles["h1"]),
        Paragraph(
            "Takımlar play-off öncesi AO Live Elo'ya göre azalan sıradadır. Q turu kolonu '-' ise takım o turda maç oynamamıştır.",
            styles["small"],
        ),
        table,
        PageBreak(),
    ]


def result_cell(detail: dict[str, object], styles: dict[str, ParagraphStyle]) -> Paragraph:
    if int(detail["matches"]) == 0:
        return Paragraph("<font color='#60717D'>Oynamadı</font>", styles["result"])
    scores = "<br/>".join(esc(value) for value in detail["scores"])
    line = (
        f"<b>{esc(detail['competitions'])} | {detail['wins']}G-{detail['draws']}B-{detail['losses']}M | "
        f"{detail['goals_for']}:{detail['goals_against']}</b><br/>{scores}<br/>"
        f"Tur sonu Elo: <b>{float(detail['end_elo']):.1f}</b>"
    )
    return Paragraph(line, styles["result"])


def results_section(
    ratings: pd.DataFrame,
    details: dict[tuple[str, str], dict[str, object]],
    styles: dict[str, ParagraphStyle],
) -> list[object]:
    header = ["PO sıra", "Takım", "Q1 sonuçları", "Q2 sonuçları", "Q3 sonuçları", "Güncel Elo"]
    rows: list[list[object]] = [[p(value, styles["table_head"]) for value in header]]
    for row in ratings.itertuples(index=False):
        team_id = str(row.team_id)
        rows.append(
            [
                p(int(row.playoff_rank), styles["table_bold"]),
                Paragraph(
                    f"{esc(row.team_name)}<br/><font color='#60717D'>{esc(row.country_code)} | "
                    f"{esc(row.playoff_competition)}</font>",
                    styles["table_bold"],
                ),
                result_cell(details[(team_id, "Q1")], styles),
                result_cell(details[(team_id, "Q2")], styles),
                result_cell(details[(team_id, "Q3")], styles),
                Paragraph(
                    f"{row.pre_playoff_live_elo:.1f}<br/>{esc(delta_text(row.pre_playoff_live_elo_change))}",
                    styles["table_bold"],
                ),
            ]
        )

    table = LongTable(
        rows,
        colWidths=[14 * mm, 43 * mm, 62 * mm, 62 * mm, 62 * mm, 20 * mm],
        repeatRows=1,
        splitByRow=1,
    )
    style = common_table_style()
    style.add("ALIGN", (0, 1), (0, -1), "CENTER")
    style.add("ALIGN", (5, 1), (5, -1), "CENTER")
    table.setStyle(style)
    return [
        Paragraph("3. Takım Bazında Q1-Q3 Sonuç ve Elo İzi", styles["h1"]),
        Paragraph(
            "Skorlar takım perspektifindedir. (İ) iç saha, (D) deplasman anlamına gelir. Penaltı atışları saha skoruna eklenmez; "
            "[pen.] etiketi yalnız eşleşmenin penaltılarla sonuçlandığını belirtir.",
            styles["small"],
        ),
        table,
        PageBreak(),
    ]


def audit_section(
    data: dict[str, object],
    ratings: pd.DataFrame,
    matches: pd.DataFrame,
    details: dict[tuple[str, str], dict[str, object]],
    styles: dict[str, ParagraphStyle],
) -> list[object]:
    changed = ratings[ratings["pre_playoff_live_elo_change"].abs() > 1e-9]
    top_up = ratings.nlargest(8, "pre_playoff_live_elo_change")
    top_down = ratings.nsmallest(8, "pre_playoff_live_elo_change")

    movers = [[p("En çok yükselen", styles["table_head"]), p("Δ", styles["table_head"]), p("En çok düşen", styles["table_head"]), p("Δ", styles["table_head"])]]
    for (_, up), (_, down) in zip(top_up.iterrows(), top_down.iterrows()):
        movers.append([
            p(up.team_name, styles["table_bold"]), p(delta_text(up.pre_playoff_live_elo_change), styles["table"]),
            p(down.team_name, styles["table_bold"]), p(delta_text(down.pre_playoff_live_elo_change), styles["table"]),
        ])
    mover_table = Table(movers, colWidths=[90 * mm, 25 * mm, 90 * mm, 25 * mm], repeatRows=1)
    mover_table.setStyle(common_table_style())

    checks = [
        ("Play-off takım kimliği", len(ratings) == ratings["team_id"].nunique() == 86),
        ("237 takımlık genel sıra aralığı", ratings["pre_playoff_rank"].between(1, 237).all()),
        ("Q1-Q3 maç kimliği benzersizliği", len(matches) == matches["match_id"].nunique() == 342),
        ("Play-off sonucu rapora girmedi", set(matches["round"]) == set(ROUND_KEYS)),
        ("Maç dışı ana aşama düzeltmesi yok", (ratings["non_match_main_entry_adjustment"].abs() < 1e-12).all()),
        ("Main-entry reset uygulanmadı", (~ratings["main_entry_reset_applied_at_this_cutoff"].astype(bool)).all()),
    ]
    audit_rows = [[p("Kontrol", styles["table_head"]), p("Sonuç", styles["table_head"])]]
    for label, passed in checks:
        audit_rows.append([p(label, styles["table"]), p("GEÇTİ" if passed else "KALDI", styles["table_bold"])])
    audit_table = Table(audit_rows, colWidths=[190 * mm, 40 * mm], repeatRows=1)
    audit_style = common_table_style()
    for index, (_, passed) in enumerate(checks, start=1):
        audit_style.add("TEXTCOLOR", (1, index), (1, index), GREEN if passed else RED)
    audit_table.setStyle(audit_style)

    cutoff_label = Turkish_date(data["cutoff_istanbul"])
    return [
        Paragraph("4. Özet ve Veri Denetimi", styles["h1"]),
        Paragraph(
            f"86 play-off takımının {len(changed)} tanesi Q1-Q3 döneminde en az bir maç oynadığı için başlangıç Elo'sundan ayrıştı. "
            f"Ortalama mutlak değişim {ratings['pre_playoff_live_elo_change'].abs().mean():.1f} Elo, "
            f"en yüksek artış {ratings['pre_playoff_live_elo_change'].max():+.1f}, en yüksek düşüş "
            f"{ratings['pre_playoff_live_elo_change'].min():+.1f} Elo oldu.",
            styles["body"],
        ),
        mover_table,
        Spacer(1, 5 * mm),
        Paragraph("Otomatik kontroller", styles["h2"]),
        audit_table,
        Spacer(1, 5 * mm),
        Paragraph("Kaynak ve kapsam", styles["h2"]),
        Paragraph(
            f"Veri kesimi: <b>{cutoff_label}</b>.<br/>"
            "Rating snapshot: <b>playoff_pre_match_team_ratings.csv</b>.<br/>"
            "Q1-Q3 sonuçları: <b>matches_completed.csv</b>. Play-off eşleşme doğrulaması: "
            "<b>fixtures_upcoming.csv</b>.<br/>"
            "Production revision: <b>2026-08-20-european-exposure-cap-065</b>.",
            styles["small"],
        ),
        Spacer(1, 3 * mm),
        Paragraph(
            "Bu belge bir tahmin tablosu değildir. Play-off başlamadan hemen önceki rating snapshot'ını ve bu snapshot'a ulaşan "
            "tamamlanmış Q1-Q3 saha sonuçlarını gösterir.",
            styles["callout"],
        ),
    ]


def common_table_style() -> TableStyle:
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), INK),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
            ("GRID", (0, 0), (-1, -1), 0.35, MID),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PALE]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3.2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
        ]
    )


def main() -> None:
    args = parse_args()
    path = build_pdf(args.data_root.resolve(), args.output_root.resolve(), args.output.resolve())
    print(path)


if __name__ == "__main__":
    main()
