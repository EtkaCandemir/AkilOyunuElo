from __future__ import annotations

import math
from html import escape
from pathlib import Path

import pandas as pd
from reportlab.graphics.shapes import Circle, Drawing, Line, Rect, String
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
OUTPUT_ROOT = ROOT / "output" / "pilot_20_teams"
PDF_PATH = OUTPUT_ROOT / "AO_European_Elo_20_Takim_Pilot_Raporu.pdf"
FONT_REGULAR = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
DOCUMENT_DATE = "23 Temmuz 2026"


def main() -> None:
    teams = pd.read_csv(OUTPUT_ROOT / "team_start_end_summary.csv")
    matches = pd.read_csv(OUTPUT_ROOT / "match_updates_detailed.csv")
    scenarios = pd.read_csv(OUTPUT_ROOT / "scenario_summary.csv")
    competitions = pd.read_csv(OUTPUT_ROOT / "competition_summary.csv")
    parameters = pd.read_csv(OUTPUT_ROOT / "model_parameters.csv")
    _validate_sources(teams, matches, competitions)
    register_fonts()
    build_pdf(teams, matches, scenarios, competitions, parameters)
    print(f"PDF written: {PDF_PATH}")


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("PilotRegular", FONT_REGULAR))
    pdfmetrics.registerFont(TTFont("PilotBold", FONT_BOLD))
    pdfmetrics.registerFontFamily(
        "Pilot",
        normal="PilotRegular",
        bold="PilotBold",
    )


def build_pdf(
    teams: pd.DataFrame,
    matches: pd.DataFrame,
    scenarios: pd.DataFrame,
    competitions: pd.DataFrame,
    parameters: pd.DataFrame,
) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=1.25 * cm,
        rightMargin=1.25 * cm,
        topMargin=1.25 * cm,
        bottomMargin=1.25 * cm,
        title="AO European Elo - 20 Takımlı Sentetik Pilot Raporu",
        author="Akıl Oyunu",
        subject="Kontrollü gol farkı production model pilotu",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates(
        [PageTemplate(id="pilot", frames=[frame], onPage=footer)]
    )
    doc.build(story(teams, matches, scenarios, competitions, parameters))


def story(
    teams: pd.DataFrame,
    matches: pd.DataFrame,
    scenarios: pd.DataFrame,
    competitions: pd.DataFrame,
    parameters: pd.DataFrame,
) -> list:
    s = styles()
    biggest_gain = teams.sort_values("total_elo_change", ascending=False).iloc[0]
    biggest_loss = teams.sort_values("total_elo_change").iloc[0]
    biggest_gd = teams.loc[
        teams["goal_difference_net_effect_vs_classic"].abs().idxmax()
    ]
    strongest_multiplier = matches.sort_values(
        "goal_multiplier",
        ascending=False,
    ).iloc[0]
    parameter_map = dict(
        zip(parameters["parameter"].astype(str), parameters["value"])
    )

    out: list = [
        Spacer(1, 1.1 * cm),
        paragraph("AO European Elo", s["Title"]),
        paragraph("20 Takımlı Sentetik Production Pilotu", s["Subtitle"]),
        paragraph(
            "Başlangıç Elo, maç güncellemeleri, kontrollü gol farkı ve final Elo karşılaştırması",
            s["CoverLine"],
        ),
        Spacer(1, 0.45 * cm),
        table(
            [
                ["Belge tarihi", DOCUMENT_DATE],
                ["Takım / maç", "20 sentetik takım / 33 exact-UTC maç"],
                ["Turnuva dağılımı", "8 UCL / 6 UEL / 6 UECL"],
                ["Başlangıç Elo bandı", "940 - 1940"],
                ["Production config", str(parameter_map["config_id"])],
                ["Veri statüsü", "Tamamen sentetik; gerçek takım gücü iddiası yoktur"],
            ],
            [5.0 * cm, 11.6 * cm],
            s,
            header=False,
        ),
        callout(
            "Pilotun amacı",
            "Modelin başlangıç rating'inden canlı rating'e nasıl geçtiğini; beraberlik, tek fark, "
            "2/3/4/5 fark, favori galibiyeti, sürpriz sonuç, nötr final ve penaltı beraberliğinde "
            "nasıl davrandığını kontrollü biçimde göstermektir.",
            s,
        ),
        PageBreak(),
    ]

    out += heading("1. Yönetici Özeti", s)
    out += metric_grid(
        [
            ("En büyük kazanç", f"{biggest_gain.team_name}\n{biggest_gain.total_elo_change:+.3f}"),
            ("En büyük kayıp", f"{biggest_loss.team_name}\n{biggest_loss.total_elo_change:+.3f}"),
            ("En yüksek M_GD", f"{strongest_multiplier.goal_multiplier:.6f}\n{strongest_multiplier.match_id}"),
            ("Maks. GD net etkisi", f"{biggest_gd.goal_difference_net_effect_vs_classic:+.3f}\n{biggest_gd.team_name}"),
        ],
        s,
    )
    out += bullets(
        [
            "Toplam başlangıç Elo ile toplam final Elo aynıdır; Power güncellemeleri sıfır toplamlıdır.",
            "UCL, UEL ve UECL etiketleri normal maç K değerini değiştirmez; fark rakip rating'lerinden doğar.",
            "Beraberlik, tek farklı sonuç ve penaltı saha beraberliğinde M_GD tam 1.00 kalır.",
            "Dört üzerindeki gol farkı büyümez; beş farklı sonuç da formülde GD=4 kullanır.",
            "Progression bonus, Achievement Reserve, season carry ve competition K kapalıdır.",
            "Gol farkı açık production ile kapalı karşı-olgusal replay arasındaki en büyük takım farkı "
            f"{teams['goal_difference_net_effect_vs_classic'].abs().max():.3f} Elo'dur.",
        ],
        s,
    )
    out += heading("2. Aktif Model Katsayıları", s)
    parameter_rows = [
        ["Parametre", "Aktif değer", "Anlamı"],
        ["Scale", f"{float(parameter_map['elo_scale']):.6f}", "Rating farkını beklentiye dönüştürür"],
        ["Home advantage", f"{float(parameter_map['home_advantage']):.6f}", "Nötr olmayan maçta D'ye eklenir"],
        ["K", f"{float(parameter_map['k_factor']):.6f}", "Temel öğrenme hızı"],
        ["alpha", f"{float(parameter_map['goal_alpha']):.2f}", "Gol farkı bonus şiddeti"],
        ["tau", f"{float(parameter_map['goal_tau']):.0f}", "Güç farkı büyüdükçe sönümleme"],
        ["GD cap", f"{int(float(parameter_map['goal_difference_cap']))}", "4 ve üzeri aynı GD sinyali"],
        ["Carry", f"{float(parameter_map['power_carry']):.2f}", "Kapalı"],
        ["Progression", "0", "Kapalı"],
        ["Reserve", f"{float(parameter_map['achievement_reserve_base']):.0f}", "Kapalı"],
    ]
    out.append(table(parameter_rows, [4.3 * cm, 3.2 * cm, 9.1 * cm], s))
    out += formulas(
        [
            "D = Home Live Elo - Away Live Elo + H",
            "E_home = 1 / (1 + 10 ^ (-D / Scale))",
            "M_GD = 1 + 0.10 x ln(min(GD, 4)) x exp(-abs(D) / 300)",
            "Delta = 103.980986 x (S_home - E_home) x M_GD",
            "Home New = Home Old + Delta; Away New = Away Old - Delta",
        ],
        s,
    )
    out.append(PageBreak())

    out += heading("3. Gol Farkı Çarpanı Nasıl Değişir?", s)
    multiplier_rows = [["GD", "D=0", "|D|=300", "|D|=600", "Yorum"]]
    for gd in (0, 1, 2, 3, 4, 5):
        multiplier_rows.append(
            [
                str(gd),
                f"{theoretical_multiplier(gd, 0):.6f}",
                f"{theoretical_multiplier(gd, 300):.6f}",
                f"{theoretical_multiplier(gd, 600):.6f}",
                (
                    "Beraberlik"
                    if gd == 0
                    else "Tek fark; bonus yok"
                    if gd == 1
                    else "Cap sonrası GD=4" if gd >= 5 else "Azalan getirili bonus"
                ),
            ]
        )
    out.append(
        table(
            multiplier_rows,
            [1.5 * cm, 3.0 * cm, 3.0 * cm, 3.0 * cm, 6.1 * cm],
            s,
        )
    )
    out.append(
        callout(
            "Ana okuma",
            "Aynı 4-0 skoru yakın güçteki iki takım arasında D=0 iken M_GD=1.138629 üretir. "
            "|D|=600 olduğunda aynı skorun çarpanı 1.018761'e düşer. Böylece büyük favorinin "
            "beklenen farklı galibiyeti rating'i aşırı büyütmez.",
            s,
        )
    )
    out += heading("4. Başlangıç Elo Dağılımı", s)
    for chunk in chunks(
        teams.sort_values("start_rank"),
        10,
    ):
        rows = [["Sıra", "Takım", "Kulvar", "Bant", "Başlangıç"]]
        for row in chunk.itertuples(index=False):
            rows.append(
                [
                    str(row.start_rank),
                    row.team_name,
                    row.competition_track,
                    row.strength_band,
                    f"{row.start_elo:.3f}",
                ]
            )
        out.append(table(rows, [1.5 * cm, 6.0 * cm, 2.2 * cm, 3.2 * cm, 3.7 * cm], s))
    out.append(PageBreak())

    out += heading("5. Maç Bazında Elo Güncellemeleri", s)
    out += bullets(
        [
            "Klasik Delta aynı pre-match state üzerinde K x (S-E) değeridir.",
            "GD Ek, production delta ile aynı-state klasik delta arasındaki farktır.",
            "Final Delta production modelinin ev sahibi açısından gerçek Power değişimidir.",
            "Negatif Final Delta deplasman takımının aynı büyüklükte Elo kazandığını gösterir.",
        ],
        s,
    )
    match_chunks = chunks(matches, 17)
    for index, chunk in enumerate(match_chunks):
        rows = [
            [
                "ID",
                "Kulvar",
                "Maç",
                "Skor",
                "D",
                "E",
                "GD / M",
                "Klasik / Ek / Final Δ",
            ]
        ]
        for row in chunk.itertuples(index=False):
            rows.append(
                [
                    row.match_id.replace("pilot20-", ""),
                    row.competition,
                    f"{row.home_team_name} - {row.away_team_name}",
                    f"{row.home_goals}-{row.away_goals}",
                    f"{row.effective_rating_difference:+.1f}",
                    f"{row.expected_home_score:.3f}",
                    f"{row.goal_difference} / {row.goal_multiplier:.4f}",
                    (
                        f"{row.classic_delta_same_state:+.2f} / "
                        f"{row.gd_extra_delta_same_state:+.2f} / "
                        f"{row.power_delta:+.2f}"
                    ),
                ]
            )
        out.append(
            table(
                rows,
                [
                    0.8 * cm,
                    1.3 * cm,
                    5.0 * cm,
                    1.2 * cm,
                    1.4 * cm,
                    1.3 * cm,
                    2.0 * cm,
                    3.6 * cm,
                ],
                s,
                compact=True,
            )
        )
        if index < len(match_chunks) - 1:
            out.append(PageBreak())
            out += heading("5. Maç Bazında Elo Güncellemeleri - Devam", s)
    out.append(PageBreak())

    out += heading("6. Başlangıç ve Bitiş Elo Karşılaştırması", s)
    out.append(rating_dumbbell(teams.sort_values("final_rank").head(10), "Final sırası 1-10"))
    out.append(rating_dumbbell(teams.sort_values("final_rank").tail(10), "Final sırası 11-20"))
    out.append(PageBreak())

    team_chunks = chunks(teams.sort_values("final_rank"), 20)
    for index, chunk in enumerate(team_chunks):
        out += heading(
            "7. Takım Bazında Detay"
            if index == 0
            else "7. Takım Bazında Detay - Devam",
            s,
        )
        rows = [
            [
                "Final",
                "Takım",
                "Kulvar",
                "Başlangıç",
                "Bitiş",
                "Değişim",
                "Sıra +/-",
                "GD net",
                "G-B-M",
            ]
        ]
        for row in chunk.itertuples(index=False):
            rows.append(
                [
                    str(row.final_rank),
                    row.team_name,
                    row.competition_track,
                    f"{row.start_elo:.1f}",
                    f"{row.final_live_elo:.1f}",
                    f"{row.total_elo_change:+.1f}",
                    f"{row.rank_change:+d}",
                    f"{row.goal_difference_net_effect_vs_classic:+.2f}",
                    f"{row.wins}-{row.draws}-{row.losses}",
                ]
            )
        out.append(
            table(
                rows,
                [
                    1.2 * cm,
                    4.2 * cm,
                    1.5 * cm,
                    2.0 * cm,
                    2.0 * cm,
                    2.0 * cm,
                    1.5 * cm,
                    1.8 * cm,
                    1.7 * cm,
                ],
                s,
                compact=True,
            )
        )
    out.append(PageBreak())

    out += heading("8. Senaryo ve Turnuva Kontrolleri", s)
    scenario_rows = [
        ["Senaryo", "Maç", "Ort. |D|", "Ort. GD", "Ort. M", "Maks. M", "Ort. |GD Ek|"]
    ]
    for row in scenarios.head(14).itertuples(index=False):
        scenario_rows.append(
            [
                row.scenario_group.replace("_", " "),
                str(row.matches),
                f"{row.average_abs_D:.1f}",
                f"{row.average_goal_difference:.2f}",
                f"{row.average_multiplier:.4f}",
                f"{row.maximum_multiplier:.4f}",
                f"{row.average_abs_gd_extra:.3f}",
            ]
        )
    out.append(
        table(
            scenario_rows,
            [5.1 * cm, 1.2 * cm, 2.0 * cm, 1.7 * cm, 2.0 * cm, 2.0 * cm, 2.6 * cm],
            s,
            compact=True,
        )
    )
    competition_rows = [
        ["Kulvar", "Takım", "Maç", "Başlangıç toplamı", "Final toplamı", "Korunum hatası", "Maks. M"]
    ]
    for row in competitions.itertuples(index=False):
        competition_rows.append(
            [
                row.competition,
                str(row.teams),
                str(row.matches),
                f"{row.start_elo_sum:.3f}",
                f"{row.final_elo_sum:.3f}",
                f"{row.elo_conservation_error:.3e}",
                f"{row.maximum_multiplier:.6f}",
            ]
        )
    out.append(
        table(
            competition_rows,
            [2.0 * cm, 1.5 * cm, 1.5 * cm, 3.1 * cm, 3.1 * cm, 3.1 * cm, 2.6 * cm],
            s,
            compact=True,
        )
    )
    out += heading("9. Model Davranışı ve Karar", s)
    out += bullets(
        [
            f"Northbridge FC üç galibiyetle {teams.loc[teams.team_name.eq('Northbridge FC'),'total_elo_change'].iloc[0]:+.3f} Elo kazanıp 4. sıradan 2. sıraya çıktı.",
            f"Island Rovers üç baskın sürpriz galibiyetle {biggest_gain.total_elo_change:+.3f} Elo ve dört sıra kazandı.",
            f"Granite Town bir beraberlik ve üç yenilgiyle {biggest_loss.total_elo_change:+.3f} Elo kaybetti.",
            "Apex City başlangıçta 1.940 iken iki galibiyet, bir yenilgi ve penaltı beraberliği sonunda "
            f"{teams.loc[teams.team_name.eq('Apex City'),'final_live_elo'].iloc[0]:.3f} seviyesinde kaldı.",
            "Gol farkı katmanı yönü değiştirmez; klasik sürpriz sinyalini yalnızca sınırlı biçimde büyütür.",
            "En yüksek çarpan yakın güçteki Island Rovers - Granite Town 4-0 maçında oluştu; "
            "ağır favorilerin 4/5 farklı galibiyetlerinde sönümleme nedeniyle çarpan çok daha düşüktür.",
            "Penaltıyla biten UCL finalinde saha skoru 1-1, S=0.5, GD=0 ve M_GD=1 olarak işlendi; "
            "tur veya kupa bonusu eklenmedi.",
            "Bu pilot matematik davranışını doğrular; gerçek dünya tahmin doğruluğu kanıtı değildir.",
        ],
        s,
    )
    out.append(
        callout(
            "Pilot kararı",
            "Production yapılandırması tasarlandığı gibi çalışmıştır: aktif tek ek katman kontrollü "
            "gol farkıdır. Sıfır toplam korunmuş, cap ve favori sönümlemesi görünür olmuş, progression "
            "ve reserve her maçta sıfır kalmıştır. Gerçek veri için aynı audit kolonlarıyla prospective "
            "izleme sürdürülmelidir.",
            s,
        )
    )
    return out


def theoretical_multiplier(goal_difference: int, absolute_d: float) -> float:
    if goal_difference <= 1:
        return 1.0
    return 1.0 + 0.10 * math.log(min(goal_difference, 4)) * math.exp(
        -absolute_d / 300.0
    )


def rating_dumbbell(frame: pd.DataFrame, title: str) -> Drawing:
    width = 16.6 * cm
    row_height = 0.48 * cm
    height = 0.75 * cm + len(frame) * row_height
    drawing = Drawing(width, height)
    drawing.add(
        String(
            0,
            height - 0.22 * cm,
            title,
            fontName="PilotBold",
            fontSize=9,
            fillColor=colors.HexColor("#173A5E"),
        )
    )
    x0 = 5.0 * cm
    x1 = 16.2 * cm
    minimum = 800.0
    maximum = 2000.0

    def x_position(value: float) -> float:
        return x0 + (float(value) - minimum) / (maximum - minimum) * (x1 - x0)

    for index, row in enumerate(frame.itertuples(index=False)):
        y = height - 0.62 * cm - index * row_height
        start_x = x_position(row.start_elo)
        final_x = x_position(row.final_live_elo)
        drawing.add(
            String(
                0,
                y - 2,
                f"{row.final_rank:>2}. {row.team_name}",
                fontName="PilotRegular",
                fontSize=7.2,
                fillColor=colors.HexColor("#263747"),
            )
        )
        drawing.add(
            Line(
                min(start_x, final_x),
                y,
                max(start_x, final_x),
                y,
                strokeColor=colors.HexColor("#91A4B7"),
                strokeWidth=1.4,
            )
        )
        drawing.add(
            Circle(
                start_x,
                y,
                2.6,
                fillColor=colors.white,
                strokeColor=colors.HexColor("#64788C"),
                strokeWidth=1,
            )
        )
        final_color = (
            colors.HexColor("#16836B")
            if row.total_elo_change >= 0
            else colors.HexColor("#C84A4A")
        )
        drawing.add(
            Circle(
                final_x,
                y,
                3.2,
                fillColor=final_color,
                strokeColor=final_color,
            )
        )
        drawing.add(
            String(
                x1 + 0.1 * cm,
                y - 2,
                f"{row.start_elo:.0f}->{row.final_live_elo:.0f}",
                fontName="PilotRegular",
                fontSize=6.8,
                fillColor=colors.HexColor("#263747"),
            )
        )
    drawing.add(
        Rect(
            x0,
            0.05 * cm,
            x1 - x0,
            0.02 * cm,
            fillColor=colors.HexColor("#D5DEE7"),
            strokeColor=None,
        )
    )
    return drawing


def chunks(frame: pd.DataFrame, size: int) -> list[pd.DataFrame]:
    return [frame.iloc[start : start + size] for start in range(0, len(frame), size)]


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D2DBE4"))
    canvas.setLineWidth(0.4)
    canvas.line(doc.leftMargin, 0.9 * cm, A4[0] - doc.rightMargin, 0.9 * cm)
    canvas.setFont("PilotRegular", 7)
    canvas.setFillColor(colors.HexColor("#5B6978"))
    canvas.drawString(doc.leftMargin, 0.55 * cm, "AO European Elo - 20 Takımlı Sentetik Pilot")
    canvas.drawCentredString(A4[0] / 2, 0.55 * cm, DOCUMENT_DATE)
    canvas.drawRightString(A4[0] - doc.rightMargin, 0.55 * cm, f"Sayfa {doc.page}")
    canvas.restoreState()


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "Title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="PilotBold",
            fontSize=24,
            leading=29,
            textColor=colors.HexColor("#153554"),
            alignment=TA_CENTER,
            spaceAfter=5,
        ),
        "Subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Heading2"],
            fontName="PilotBold",
            fontSize=15,
            leading=19,
            textColor=colors.HexColor("#315A7C"),
            alignment=TA_CENTER,
            spaceAfter=7,
        ),
        "CoverLine": ParagraphStyle(
            "CoverLine",
            parent=base["BodyText"],
            fontName="PilotRegular",
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#556575"),
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
        "H1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName="PilotBold",
            fontSize=14,
            leading=17,
            textColor=colors.HexColor("#173A5E"),
            spaceBefore=4,
            spaceAfter=7,
        ),
        "Body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="PilotRegular",
            fontSize=8.2,
            leading=11.2,
            textColor=colors.HexColor("#263747"),
            alignment=TA_LEFT,
            spaceAfter=4,
        ),
        "Bullet": ParagraphStyle(
            "Bullet",
            parent=base["BodyText"],
            fontName="PilotRegular",
            fontSize=8.1,
            leading=10.9,
            leftIndent=8,
            firstLineIndent=-6,
            textColor=colors.HexColor("#263747"),
            spaceAfter=3,
        ),
        "Formula": ParagraphStyle(
            "Formula",
            parent=base["BodyText"],
            fontName="PilotRegular",
            fontSize=8.1,
            leading=10.7,
            leftIndent=7,
            rightIndent=7,
            borderWidth=0.4,
            borderColor=colors.HexColor("#B9C9D8"),
            backgroundColor=colors.HexColor("#F2F6FA"),
            borderPadding=4,
            spaceAfter=3,
            textColor=colors.HexColor("#173A5E"),
        ),
        "TableHeader": ParagraphStyle(
            "TableHeader",
            parent=base["BodyText"],
            fontName="PilotBold",
            fontSize=7.2,
            leading=8.5,
            textColor=colors.white,
        ),
        "TableCell": ParagraphStyle(
            "TableCell",
            parent=base["BodyText"],
            fontName="PilotRegular",
            fontSize=7.0,
            leading=8.3,
            textColor=colors.HexColor("#253545"),
        ),
        "TableHeaderCompact": ParagraphStyle(
            "TableHeaderCompact",
            parent=base["BodyText"],
            fontName="PilotBold",
            fontSize=6.4,
            leading=7.3,
            textColor=colors.white,
        ),
        "TableCellCompact": ParagraphStyle(
            "TableCellCompact",
            parent=base["BodyText"],
            fontName="PilotRegular",
            fontSize=6.25,
            leading=7.25,
            textColor=colors.HexColor("#253545"),
        ),
        "CalloutTitle": ParagraphStyle(
            "CalloutTitle",
            parent=base["BodyText"],
            fontName="PilotBold",
            fontSize=8.5,
            leading=10.5,
            textColor=colors.HexColor("#173A5E"),
            spaceAfter=3,
        ),
        "CalloutBody": ParagraphStyle(
            "CalloutBody",
            parent=base["BodyText"],
            fontName="PilotRegular",
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#263747"),
        ),
        "MetricLabel": ParagraphStyle(
            "MetricLabel",
            parent=base["BodyText"],
            fontName="PilotRegular",
            fontSize=7.2,
            leading=8.5,
            textColor=colors.HexColor("#617285"),
            alignment=TA_CENTER,
        ),
        "MetricValue": ParagraphStyle(
            "MetricValue",
            parent=base["BodyText"],
            fontName="PilotBold",
            fontSize=10,
            leading=12,
            textColor=colors.HexColor("#173A5E"),
            alignment=TA_CENTER,
        ),
    }


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(str(text)).replace("\n", "<br/>"), style)


def heading(text: str, s: dict[str, ParagraphStyle]) -> list:
    return [paragraph(text, s["H1"])]


def bullets(items: list[str], s: dict[str, ParagraphStyle]) -> list:
    return [paragraph(f"- {item}", s["Bullet"]) for item in items]


def formulas(items: list[str], s: dict[str, ParagraphStyle]) -> list:
    return [paragraph(item, s["Formula"]) for item in items]


def callout(title: str, body: str, s: dict[str, ParagraphStyle]) -> KeepTogether:
    box = Table(
        [
            [
                [
                    paragraph(title, s["CalloutTitle"]),
                    paragraph(body, s["CalloutBody"]),
                ]
            ]
        ],
        colWidths=[16.6 * cm],
    )
    box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EAF2F8")),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#A8BED1")),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return KeepTogether([box, Spacer(1, 0.15 * cm)])


def metric_grid(
    metrics: list[tuple[str, str]],
    s: dict[str, ParagraphStyle],
) -> list:
    cells = []
    for label, value in metrics:
        cells.append(
            [
                paragraph(label, s["MetricLabel"]),
                paragraph(value, s["MetricValue"]),
            ]
        )
    grid = Table(
        [[cells[0], cells[1], cells[2], cells[3]]],
        colWidths=[4.05 * cm] * 4,
    )
    grid.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F5F8FB")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#C6D3DF")),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D5DEE6")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return [grid, Spacer(1, 0.2 * cm)]


def table(
    rows: list[list[str]],
    widths: list[float],
    s: dict[str, ParagraphStyle],
    *,
    header: bool = True,
    compact: bool = False,
) -> Table:
    converted = []
    for row_index, row in enumerate(rows):
        if compact:
            style = (
                s["TableHeaderCompact"]
                if header and row_index == 0
                else s["TableCellCompact"]
            )
        else:
            style = (
                s["TableHeader"]
                if header and row_index == 0
                else s["TableCell"]
            )
        converted.append([paragraph(value, style) for value in row])
    result = Table(
        converted,
        colWidths=widths,
        repeatRows=1 if header else 0,
        hAlign="LEFT",
    )
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#C9D4DF")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4 if compact else 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4 if compact else 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3 if compact else 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3 if compact else 4),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#24527A")),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#F6F8FB")],
                ),
            ]
        )
    else:
        commands.append(
            (
                "ROWBACKGROUNDS",
                (0, 0),
                (-1, -1),
                [colors.white, colors.HexColor("#F6F8FB")],
            )
        )
    result.setStyle(TableStyle(commands))
    result.spaceAfter = 8
    return result


def _validate_sources(
    teams: pd.DataFrame,
    matches: pd.DataFrame,
    competitions: pd.DataFrame,
) -> None:
    if len(teams) != 20 or len(matches) != 33:
        raise ValueError("PDF source must contain 20 teams and 33 matches")
    if matches["total_power_zero_sum_error"].max() > 1e-9:
        raise ValueError("PDF source failed zero-sum validation")
    if competitions["elo_conservation_error"].max() > 1e-9:
        raise ValueError("Competition Elo conservation failed")


if __name__ == "__main__":
    main()
