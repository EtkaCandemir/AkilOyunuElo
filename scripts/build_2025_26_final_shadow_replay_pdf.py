from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

import pandas as pd
from reportlab.graphics.shapes import Circle, Drawing, Line, Rect, String
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
    KeepTogether,
    LongTable,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "output" / "final_shadow_replay_2025_26"
DEFAULT_PDF = ROOT / "output" / "pdf" / "AO_2025_26_Final_ve_Shadow_Replay_Raporu.pdf"

INK = colors.HexColor("#172A3A")
BLUE = colors.HexColor("#2D5F8B")
TEAL = colors.HexColor("#087E8B")
GOLD = colors.HexColor("#D39A22")
CORAL = colors.HexColor("#C95D63")
OLIVE = colors.HexColor("#708238")
PALE = colors.HexColor("#F2F5F7")
MID = colors.HexColor("#D6E0E5")
MUTED = colors.HexColor("#61727E")
WHITE = colors.white


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the 2025/26 final/shadow replay PDF")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_PDF)
    args = parser.parse_args()
    path = build_pdf(args.input_root.resolve(), args.output.resolve())
    print(path)


def build_pdf(input_root: Path, output_path: Path) -> Path:
    data = load_report_data(input_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    regular, bold = register_fonts()
    styles = build_styles(regular, bold)
    page_size = landscape(A4)
    document = BaseDocTemplate(
        str(output_path),
        pagesize=page_size,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=17 * mm,
        bottomMargin=15 * mm,
        title="AO European Elo 2025/26 Final Model ve Shadow Replay",
        author="AkılOyunu.com",
        subject="AO First Elo, sezon sonu Power Elo ve shadow karşılaştırması",
    )
    frame = Frame(
        document.leftMargin,
        document.bottomMargin,
        document.width,
        document.height,
        id="report",
    )
    document.addPageTemplates(
        PageTemplate(id="main", frames=[frame], onPage=page_frame(regular, bold, page_size))
    )
    story: list[object] = []
    story.extend(cover(data, styles))
    story.extend(technical_summary(data, styles))
    story.extend(model_contract(data, styles))
    story.extend(main_results(data, styles))
    story.extend(surprise_results(data, styles))
    story.extend(progression_results(data, styles))
    story.extend(combined_results(data, styles))
    story.extend(segment_results(data, styles))
    story.extend(limitations(data, styles))
    story.extend(full_team_appendix(data, styles))
    document.build(story)
    return output_path


def load_report_data(root: Path) -> dict[str, object]:
    required = {
        "manifest": "replay_manifest.json",
        "ratings": "initial_and_final_ratings.csv",
        "models": "model_comparison.csv",
        "competitions": "competition_summary.csv",
        "ranking": "same_season_ranking_diagnostic.csv",
        "shadows": "shadow_effect_summary.csv",
        "bonus": "bonus_events.csv",
        "audits": "identity_and_invariant_audit.csv",
    }
    missing = [filename for filename in required.values() if not (root / filename).exists()]
    if missing:
        raise FileNotFoundError(f"Replay output is incomplete: {missing}")
    data: dict[str, object] = {
        "manifest": json.loads((root / required["manifest"]).read_text(encoding="utf-8"))
    }
    for key, filename in required.items():
        if key != "manifest":
            data[key] = pd.read_csv(root / filename)
    return data


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
            "CoverTitle", parent=base["Title"], fontName=bold, fontSize=25,
            leading=30, textColor=WHITE, alignment=TA_LEFT, spaceAfter=8,
        ),
        "cover_subtitle": ParagraphStyle(
            "CoverSubtitle", parent=base["Normal"], fontName=regular, fontSize=12,
            leading=17, textColor=colors.HexColor("#DCECEF"), spaceAfter=5,
        ),
        "h1": ParagraphStyle(
            "H1", parent=base["Heading1"], fontName=bold, fontSize=17,
            leading=21, textColor=INK, spaceBefore=2, spaceAfter=7,
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Heading2"], fontName=bold, fontSize=11.5,
            leading=14, textColor=TEAL, spaceBefore=7, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "Body", parent=base["BodyText"], fontName=regular, fontSize=8.7,
            leading=12.5, textColor=INK, spaceAfter=5,
        ),
        "small": ParagraphStyle(
            "Small", parent=base["BodyText"], fontName=regular, fontSize=7.2,
            leading=9.8, textColor=INK, spaceAfter=3,
        ),
        "callout": ParagraphStyle(
            "Callout", parent=base["BodyText"], fontName=bold, fontSize=9.2,
            leading=13, textColor=INK, borderColor=TEAL, borderWidth=1,
            borderPadding=7, backColor=colors.HexColor("#E8F4F4"),
            spaceBefore=4, spaceAfter=7,
        ),
        "warning": ParagraphStyle(
            "Warning", parent=base["BodyText"], fontName=bold, fontSize=8.8,
            leading=12.5, textColor=INK, borderColor=GOLD, borderWidth=1,
            borderPadding=7, backColor=colors.HexColor("#FFF7E5"),
            spaceBefore=4, spaceAfter=7,
        ),
        "table": ParagraphStyle(
            "Table", parent=base["BodyText"], fontName=regular, fontSize=6.4,
            leading=8.0, textColor=INK,
        ),
        "table_small": ParagraphStyle(
            "TableSmall", parent=base["BodyText"], fontName=regular, fontSize=5.7,
            leading=7.0, textColor=INK,
        ),
        "table_head": ParagraphStyle(
            "TableHead", parent=base["BodyText"], fontName=bold, fontSize=6.2,
            leading=7.8, textColor=WHITE, alignment=TA_CENTER,
        ),
    }


def page_frame(regular: str, bold: str, page_size: tuple[float, float]):
    def draw(canvas, document):
        width, height = page_size
        canvas.saveState()
        if document.page > 1:
            canvas.setFont(bold, 7.5)
            canvas.setFillColor(INK)
            canvas.drawString(14 * mm, height - 10 * mm, "AkılOyunu.com | AO European Elo teknik replay")
        canvas.setStrokeColor(MID)
        canvas.line(14 * mm, 12 * mm, width - 14 * mm, 12 * mm)
        canvas.setFont(regular, 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(14 * mm, 8 * mm, "2025/26 UCL, UEL ve UECL | 961 maç | 236 kalıcı club_id")
        canvas.drawRightString(width - 14 * mm, 8 * mm, f"Sayfa {document.page}")
        canvas.restoreState()
    return draw


def cover(data: dict[str, object], styles: dict[str, ParagraphStyle]) -> list[object]:
    manifest = data["manifest"]
    assert isinstance(manifest, dict)
    band = Table(
        [
            [Paragraph("AO European Elo", styles["cover_title"])],
            [Paragraph("2025/26 Final Model ve Shadow Replay", styles["cover_subtitle"])],
            [Paragraph("Başlangıç AO First Elo - sezon sonu Power Elo - iki bağımsız shadow senaryosu", styles["cover_subtitle"])],
            [Paragraph("961 gerçek maç  |  236 kulüp  |  606 doğrulanmış xG satırı", styles["cover_subtitle"])],
        ],
        colWidths=[257 * mm],
        rowHeights=[30 * mm, 14 * mm, 14 * mm, 13 * mm],
    )
    band.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 2), INK),
                ("BACKGROUND", (0, 3), (-1, 3), TEAL),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 14 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10 * mm),
            ]
        )
    )
    return [
        Spacer(1, 21 * mm),
        band,
        Spacer(1, 15 * mm),
        Paragraph(
            "Sonuç: Ana final-aday model değiştirilmeden 2025/26 sezonu kronolojik olarak yeniden işlendi. "
            "Domestic Surprise ve aşama bonusu yalnız ayrı shadow state'lerinde çalıştırıldı; ana modelin "
            "başlangıç veya maç sonrası ratinglerine hiçbir veri geri beslemesi yapılmadı.",
            styles["callout"],
        ),
        two_column_key_values(
            [
                ("Kanıt sınıfı", manifest["evidence_class"]),
                ("Kimlik sözleşmesi", manifest["identity_contract"]),
                ("Ana model değişikliği", "Yok"),
                ("Shadow production etkisi", "Yok"),
                ("xG eksik maç davranışı", "Gol farkı katmanına dönüş"),
                ("Karar sınırı", "Bu replay tek başına PROMOTE üretmez"),
            ],
            styles,
        ),
        PageBreak(),
    ]


def technical_summary(data: dict[str, object], styles: dict[str, ParagraphStyle]) -> list[object]:
    models = data["models"]
    ratings = data["ratings"]
    shadows = data["shadows"]
    assert isinstance(models, pd.DataFrame)
    assert isinstance(ratings, pd.DataFrame)
    assert isinstance(shadows, pd.DataFrame)
    main = models.loc[models["model_arm"].eq("MAIN_FINAL_CANDIDATE")].iloc[0]
    surprise = shadows.loc[shadows["model_arm"].eq("DOMESTIC_SURPRISE_SHADOW")].iloc[0]
    progression = shadows.loc[shadows["model_arm"].eq("PROGRESSION_12_8_4_SHADOW")].iloc[0]
    top = ratings.nsmallest(3, "main_end_rank")
    names = ", ".join(
        f"{row.team_name} ({row.main_end_live_rating:.1f})" for row in top.itertuples()
    )
    return [
        Paragraph("1. Teknik Özet", styles["h1"]),
        metric_strip(
            [
                ("Maç", "961", "tam sezon"),
                ("Kulüp", "236", "kalıcı kimlik"),
                ("xG", "606", "%63.1 kapsama"),
                ("Brier", f"{main.brier_1x2:.6f}", "ana model"),
                ("Log-loss", f"{main.log_loss_1x2:.6f}", "ana model"),
            ],
            styles,
        ),
        Spacer(1, 4 * mm),
        Paragraph("Ana modelde sezon sonu liderleri", styles["h2"]),
        Paragraph(
            f"Sezon sonunda genel rating sıralamasının ilk üçü {names}. Başlangıç-sezon sonu "
            f"Spearman korelasyonu {main.start_end_spearman:.3f}; yani maçlar sıralamayı anlamlı biçimde "
            "hareket ettirirken başlangıç gücü tamamen silinmiyor.",
            styles["body"],
        ),
        Paragraph("Shadow sinyalleri küçük ama yön olarak olumlu", styles["h2"]),
        Paragraph(
            f"Domestic Surprise bu tek sezonda Brier'i {abs(surprise.brier_delta_vs_main):.6f}, "
            f"log-loss'u {abs(surprise.log_loss_delta_vs_main):.6f} azalttı. Progression shadow "
            f"için azalışlar sırasıyla {abs(progression.brier_delta_vs_main):.6f} ve "
            f"{abs(progression.log_loss_delta_vs_main):.6f}. Etkiler pozitif yönde, fakat pratik büyüklükleri "
            "küçük ve bu veri bağımsız holdout değil.",
            styles["body"],
        ),
        Paragraph(
            "Model kararı: Ana model aynen korunur. İki özellik shadow olarak izlenmeye devam eder. "
            "Bu raporda görülen küçük loss iyileşmeleri, geçmiş walk-forward kararlarını tek başına tersine çevirmez.",
            styles["warning"],
        ),
        Paragraph("Kimlik ve invariant sonucu", styles["h2"]),
        Paragraph(
            "236 yerel team_id, 236 UEFA team_id ve 236 kalıcı club_id bire bir eşleşti. Dört kolun "
            "tamamında 961 benzersiz maç bir kez işlendi; maç içi Power Elo sıfır toplamı korundu. "
            "Ana kol, daha önce üretilmiş final-aday replay ile 1e-8 toleransında aynı sonucu verdi.",
            styles["body"],
        ),
        PageBreak(),
    ]


def model_contract(data: dict[str, object], styles: dict[str, ParagraphStyle]) -> list[object]:
    manifest = data["manifest"]
    assert isinstance(manifest, dict)
    final = manifest["final_candidate"]
    surprise = manifest["domestic_surprise_shadow"]
    progression = manifest["progression_shadow"]
    return [
        Paragraph("2. Sabit Model Sözleşmesi", styles["h1"]),
        Paragraph("Ana canlı Elo güncellemesi", styles["h2"]),
        Paragraph(
            "Önce maç öncesi rating farkı hesaplanır: D = R_home - R_away + H. Nötr sahada H=0. "
            "Beklenen skor E = 1 / (1 + 10^(-D/Scale)). Klasik residual, kontrollü gol farkı ve "
            "varsa iki taraflı xG performans sinyali ile ayarlanır. Kazanan takımın toplam maç kazancı "
            "her koşulda pozitif kalır ve iki takım arasındaki Power Elo değişimi sıfır toplamlıdır.",
            styles["body"],
        ),
        key_value_table(
            [
                ("Scale", f"{final['elo_scale']:.6f}"),
                ("Ev sahibi avantajı H", f"{final['home_advantage']:.6f}"),
                ("K", f"{final['k_factor']:.6f}"),
                ("Beraberlik modeli", f"draw_at_even={final['draw_at_even']:.2f}, shape={final['draw_shape']:.2f}"),
                ("Gol farkı", f"alpha={final['goal_alpha']:.2f}, tau={final['goal_tau']:.0f}, cap={final['goal_difference_cap']}"),
                ("xG", f"ratio={final['xg_ratio']:.2f}, scale={final['xg_scale']:.2f}, winner floor={final['minimum_winner_gain_ratio']:.2f}"),
                ("xG kapsamı", "606/961; eksikse gol farkı ve klasik sonuç çekirdeği"),
                ("Carry / reserve", "Kapalı; bu replay'de ana modele aşama bonusu eklenmedi"),
            ],
            styles,
            widths=(60 * mm, 190 * mm),
        ),
        Paragraph("Domestic Surprise shadow", styles["h2"]),
        Paragraph(
            "Takımın mevcut lig bitiriş skoru, yalnız geçmiş iki sezonun güvenilir ortalamasıyla karşılaştırılır. "
            "Pozitif sürpriz ağırlığı 0.40, negatif sürpriz ağırlığı 0.20 ve mutlak başlangıç rating "
            "düzeltmesi 75 puanla sınırlıdır. En az iki geçmiş sezon yoksa düzeltme sıfırdır. Seçili aday "
            f"geçmiş backtestte {surprise['decision']} aldığı için production değildir.",
            styles["body"],
        ),
        Paragraph("Progression 12/8/4 shadow", styles["h2"]),
        Paragraph(
            "Knockout play-off, Son 16, çeyrek final, yarı final ve final tamamlandığında turu geçen takıma "
            f"UCL +{progression['ucl_increment']:.0f}, UEL +{progression['uel_increment']:.0f}, "
            f"UECL +{progression['uecl_increment']:.0f} eklenir. Sezonluk tavanlar 60/40/20'dir. "
            "Kaybedene kesinti yapılmaz; bu nedenle bonus non-zero-sum'dır ve Power Elo'dan ayrı denetlenir.",
            styles["body"],
        ),
        PageBreak(),
    ]


def main_results(data: dict[str, object], styles: dict[str, ParagraphStyle]) -> list[object]:
    ratings = data["ratings"]
    assert isinstance(ratings, pd.DataFrame)
    top = ratings.nsmallest(9, "main_end_rank")
    risers = ratings.nlargest(10, "main_rank_gain")
    fallers = ratings.nsmallest(10, "main_rank_gain")
    table_rows = [["Sıra", "Takım", "Turnuva", "Başlangıç", "Bitiş", "Elo değişimi", "Sıra değişimi"]]
    for row in top.itertuples():
        table_rows.append(
            [
                f"{int(row.main_end_rank)}",
                row.team_name,
                row.participating_competitions,
                f"{row.main_start_rating:.1f}",
                f"{row.main_end_live_rating:.1f}",
                signed(row.main_live_change, 1),
                signed(row.main_rank_gain, 0),
            ]
        )
    return [
        Paragraph("3. Ana Modelde Başlangıçtan Sezon Sonuna", styles["h1"]),
        Paragraph(
            "Aşağıdaki dağılım başlangıç AO First Elo ile 961 maç sonrasındaki Power Elo ilişkisini gösterir. "
            "500-2000 aralığı bir hard cap değildir; bu nedenle güçlü sezon geçiren takımlar 2000 üzerine "
            "çıkabilir. Arsenal'in 2331.3 ile ilk sıraya gelmesi bu canlı hareketin sonucudur.",
            styles["body"],
        ),
        start_final_scatter(ratings),
        Spacer(1, 3 * mm),
        Paragraph("Sezon sonu ilk 9", styles["h2"]),
        styled_table(table_rows, styles, [14 * mm, 61 * mm, 22 * mm, 30 * mm, 30 * mm, 31 * mm, 31 * mm]),
        PageBreak(),
        Paragraph("4. Maçların Sıralamaya Etkisi", styles["h1"]),
        Paragraph(
            "Sıra değişimi başlangıç sırası eksi sezon sonu sırası olarak tanımlıdır; pozitif değer yükselişi "
            "gösterir. AEK Athens 65, Universitatea Craiova 62 sıra yükselirken RFS Riga 66 ve The New Saints "
            "62 sıra geriledi. Bu hareketler yalnız rakip ve skor akışından değil, rakiplerin sonraki rating "
            "hareketlerinden de etkilenir.",
            styles["body"],
        ),
        rank_mover_chart(risers, fallers),
        PageBreak(),
        Paragraph("Yükselen ve Düşen Takımların Detayı", styles["h1"]),
        Paragraph(
            "Aşağıdaki iki tablo, grafikteki en büyük on pozitif ve on negatif sıra hareketini başlangıç, "
            "final ve mutlak Elo değişimiyle birlikte verir.",
            styles["body"],
        ),
        two_mover_tables(risers, fallers, styles),
        PageBreak(),
    ]


def surprise_results(data: dict[str, object], styles: dict[str, ParagraphStyle]) -> list[object]:
    ratings = data["ratings"]
    shadows = data["shadows"]
    assert isinstance(ratings, pd.DataFrame)
    assert isinstance(shadows, pd.DataFrame)
    summary = shadows.loc[shadows["model_arm"].eq("DOMESTIC_SURPRISE_SHADOW")].iloc[0]
    positive = ratings.nlargest(10, "ao_first_elo_adjustment")
    negative = ratings.nsmallest(10, "ao_first_elo_adjustment")
    rows = [["Takım", "Başlangıç düzeltmesi", "Sezon sonu farkı", "Sıra farkı"]]
    for row in pd.concat([positive.head(5), negative.head(5)]).itertuples():
        rows.append(
            [
                row.team_name,
                signed(row.ao_first_elo_adjustment, 1),
                signed(row.surprise_final_delta_vs_main, 1),
                signed(row.surprise_rank_change_vs_main, 0),
            ]
        )
    return [
        Paragraph("5. Domestic Surprise Shadow", styles["h1"]),
        Paragraph(
            f"236 takımın 193'ünde başlangıç düzeltmesi oluştu; ortalama mutlak değişim "
            f"{summary.mean_abs_initial_delta:.2f} Elo. Sezon sonunda ağ etkisi nedeniyle 233 takımın ratingi "
            f"ana koldan ayrıştı ve ortalama mutlak fark {summary.mean_abs_final_delta_vs_main:.2f} oldu. "
            "Bir takımın doğrudan başlangıç düzeltmesi küçük olsa bile, farklı rakip beklentileri tüm maç "
            "ağı üzerinden başka takımlara taşınabilir.",
            styles["body"],
        ),
        shadow_boxplot(ratings),
        Spacer(1, 3 * mm),
        Paragraph("En belirgin başlangıç sürprizleri", styles["h2"]),
        styled_table(rows, styles, [84 * mm, 50 * mm, 50 * mm, 42 * mm]),
        Paragraph(
            "Önemli yorum: Nottingham Forest ve Iberia 1999 başlangıçta +75 cap'ine ulaştı. Ancak aynı +75, "
            "sezon sonunda sırasıyla yaklaşık +27.1 ve +58.8 fark yarattı. Başlangıç müdahalesinin nihai "
            "etkisi fikstür, rakip güçleri ve maç sırasına bağlıdır.",
            styles["warning"],
        ),
        PageBreak(),
    ]


def progression_results(data: dict[str, object], styles: dict[str, ParagraphStyle]) -> list[object]:
    bonus = data["bonus"]
    ratings = data["ratings"]
    shadows = data["shadows"]
    assert isinstance(bonus, pd.DataFrame)
    assert isinstance(ratings, pd.DataFrame)
    assert isinstance(shadows, pd.DataFrame)
    progression_bonus = bonus.loc[bonus["model_arm"].eq("PROGRESSION_12_8_4_SHADOW")]
    team_bonus = (
        progression_bonus.groupby(["winner_club_id", "winner_team_name"], as_index=False)
        .agg(events=("tie_id", "nunique"), bonus=("applied_bonus", "sum"))
        .sort_values(["bonus", "winner_team_name"], ascending=[False, True])
    )
    team_bonus = team_bonus.merge(
        ratings[["club_id", "progression_final_delta_vs_main", "progression_rank_change_vs_main"]],
        left_on="winner_club_id",
        right_on="club_id",
        validate="one_to_one",
    )
    summary = shadows.loc[shadows["model_arm"].eq("PROGRESSION_12_8_4_SHADOW")].iloc[0]
    rows = [["Takım", "Olay", "Brüt bonus", "Final farkı", "Sıra farkı"]]
    for row in team_bonus.head(8).itertuples():
        rows.append(
            [
                row.winner_team_name,
                f"{int(row.events)}",
                f"+{row.bonus:.0f}",
                signed(row.progression_final_delta_vs_main, 1),
                signed(row.progression_rank_change_vs_main, 0),
            ]
        )
    return [
        Paragraph("6. Progression 12/8/4 Shadow", styles["h1"]),
        Paragraph(
            f"Her progression kolunda 69 uygun tur sonucu bir kez işlendi. Toplam {summary.total_progression_bonus:.0f} "
            f"Elo bonusu 48 takıma dağıldı; maksimum takım bonusu 60 oldu. Bonus maçtan sonra eklendiği için "
            "bir sonraki maçın beklenen skorunu etkiler. Bu geri besleme nedeniyle sezon sonu shadow farkı, "
            "brüt bonusla bire bir aynı olmak zorunda değildir.",
            styles["body"],
        ),
        bonus_leader_chart(team_bonus.head(15)),
        Spacer(1, 3 * mm),
        Paragraph("En yüksek bonus alan takımlar", styles["h2"]),
        styled_table(rows, styles, [79 * mm, 25 * mm, 38 * mm, 43 * mm, 38 * mm]),
        Paragraph(
            "Örnek: PSG brüt +60 alırken sezon sonu ana modele göre farkı +56.1'dir. Arsenal brüt +36 "
            "almasına rağmen final farkı +38.9'dur. Farkın yönü sonraki maçlarda değişen beklenen skor ve "
            "dolayısıyla değişen Power Elo residualından gelir.",
            styles["callout"],
        ),
        PageBreak(),
    ]


def combined_results(data: dict[str, object], styles: dict[str, ParagraphStyle]) -> list[object]:
    models = data["models"]
    shadows = data["shadows"]
    assert isinstance(models, pd.DataFrame)
    assert isinstance(shadows, pd.DataFrame)
    rows = [["Kol", "Brier", "Ana fark", "Log-loss", "Ana fark", "Doğruluk", "Rating std"]]
    label = {
        "MAIN_FINAL_CANDIDATE": "Ana model",
        "DOMESTIC_SURPRISE_SHADOW": "Surprise",
        "PROGRESSION_12_8_4_SHADOW": "Progression",
        "SURPRISE_PLUS_PROGRESSION_SHADOW": "Birleşik",
    }
    for row in models.itertuples():
        rows.append(
            [
                label[row.model_arm],
                f"{row.brier_1x2:.6f}",
                signed(row.brier_1x2_delta_vs_main, 6),
                f"{row.log_loss_1x2:.6f}",
                signed(row.log_loss_1x2_delta_vs_main, 6),
                f"{row.accuracy_1x2:.3f}",
                f"{row.final_rating_std:.1f}",
            ]
        )
    return [
        Paragraph("7. Shadow Kolları Ana Sonucu Nasıl Değiştirdi?", styles["h1"]),
        Paragraph(
            "İki shadow birlikte kullanıldığında Brier 0.000852 ve log-loss 0.001290 azalıyor. Bu, dört "
            "kol arasındaki en düşük loss değeridir; ancak doğruluk 0.0062 düşüyor ve pooled aynı-sezon "
            "ranking metriği de çok küçük ölçüde geriliyor. Sonuç tek yönlü bir zafer değil, karışık bir sinyaldir.",
            styles["body"],
        ),
        loss_delta_chart(models),
        Spacer(1, 4 * mm),
        styled_table(rows, styles, [50 * mm, 32 * mm, 34 * mm, 35 * mm, 34 * mm, 30 * mm, 30 * mm]),
        Paragraph("Model kararı", styles["h2"]),
        decision_table(styles),
        PageBreak(),
    ]


def segment_results(data: dict[str, object], styles: dict[str, ParagraphStyle]) -> list[object]:
    competitions = data["competitions"]
    ranking = data["ranking"]
    assert isinstance(competitions, pd.DataFrame)
    assert isinstance(ranking, pd.DataFrame)
    rows = [["Kol", "Turnuva", "Maç", "Brier", "Brier farkı", "Log-loss", "Log farkı"]]
    label = {
        "MAIN_FINAL_CANDIDATE": "Ana",
        "DOMESTIC_SURPRISE_SHADOW": "Surprise",
        "PROGRESSION_12_8_4_SHADOW": "Progression",
        "SURPRISE_PLUS_PROGRESSION_SHADOW": "Birleşik",
    }
    for row in competitions.sort_values(["competition", "model_arm"]).itertuples():
        rows.append(
            [
                label[row.model_arm],
                row.competition,
                f"{int(row.matches)}",
                f"{row.brier_1x2:.6f}",
                signed(row.brier_1x2_delta_vs_main, 6),
                f"{row.log_loss_1x2:.6f}",
                signed(row.log_loss_1x2_delta_vs_main, 6),
            ]
        )
    pooled = ranking.loc[ranking["competition"].eq("ALL")].copy()
    rank_rows = [["Kol", "Spearman", "Ana fark", "Pairwise", "Ana fark"]]
    for row in pooled.itertuples():
        rank_rows.append(
            [
                label[row.model_arm],
                f"{row.ranking_score:.6f}",
                signed(row.ranking_score_delta_vs_main, 6),
                f"{row.pairwise_accuracy:.6f}",
                signed(row.pairwise_accuracy_delta_vs_main, 6),
            ]
        )
    return [
        Paragraph("8. Turnuva ve Aynı-Sezon Sıralama Diagnostiği", styles["h1"]),
        Paragraph(
            "Surprise shadow UCL, UEL ve UECL'nin üçünde de Brier ve log-loss'u küçük ölçüde düşürdü; "
            "en büyük loss kazancı UEL segmentinde görüldü. Progression'ın belirgin yönü UCL'de, etkisi "
            "UEL ve UECL'de sıfıra yakın. Bu segmentler katsayı kararı için tek başına yeterli değildir.",
            styles["body"],
        ),
        styled_table(rows, styles, [48 * mm, 24 * mm, 21 * mm, 35 * mm, 38 * mm, 35 * mm, 38 * mm]),
        Spacer(1, 6 * mm),
        Paragraph("Pooled aynı-sezon schedule-adjusted performans", styles["h2"]),
        Paragraph(
            "Bu metrik ileri dönem sıralama testi değildir. Sezon sonu rating ile aynı sezondaki rakip "
            "ayarlı performansın ilişkisini gösterir. Ana model pooled Spearman 0.6912 ve pairwise 0.7627 "
            "üretti; shadow farkları binde birin altında kaldı.",
            styles["body"],
        ),
        styled_table(rank_rows, styles, [65 * mm, 45 * mm, 45 * mm, 45 * mm, 45 * mm]),
        PageBreak(),
    ]


def limitations(data: dict[str, object], styles: dict[str, ParagraphStyle]) -> list[object]:
    audits = data["audits"]
    assert isinstance(audits, pd.DataFrame)
    return [
        Paragraph("9. Sağlamlık, Sınırlar ve Kullanım Kararı", styles["h1"]),
        Paragraph("Geçen kontroller", styles["h2"]),
        Paragraph(
            f"Toplam {len(audits)} kimlik ve invariant kontrolünün tamamı geçti. Kontroller; maç ve takım "
            "sayılarını, kalıcı kimlik benzersizliğini, xG kapsamını, her kolun 961 maçı işlemesini, winner "
            "direction guard'ını, Power Elo sıfır toplamını, progression event sayısını ve cap'leri kapsar.",
            styles["body"],
        ),
        bullet_table(
            [
                ("Counterfactual sınırı", "2025/26 verisi model geliştirme sürecinde görüldü; sonuç bağımsız OOS değildir."),
                ("xG kapsamı", "606 maçta doğrulanmış iki taraflı xG var; 355 maçta xG bonusu sıfır ve GD-only fallback kullanılır."),
                ("Progression muhasebesi", "Bonus non-zero-sum'dır; Power Elo korunumundan ayrı olarak total bonus ve cap denetlenir."),
                ("Surprise etkisi", "Başlangıç müdahalesi fikstür ağına yayıldığı için final farkı doğrudan adjustment ile aynı değildir."),
                ("Ranking tanımı", "PDF'deki genel sıra 236 takımın tek havuz sırasıdır; turnuva segmentleri ayrıca raporlanır."),
                ("500-2000 bandı", "Referans bandıdır, hard cap değildir; canlı rating 500 altına veya 2000 üstüne çıkabilir."),
            ],
            styles,
        ),
        Paragraph("Önerilen kullanım", styles["h2"]),
        Paragraph(
            "Kullanıcıya gösterilecek tek ana değer `main_end_live_rating` olmalıdır. Surprise ve progression "
            "kolonları ürün arayüzüne taşınmamalı; analiz ve izleme tablolarında shadow karşılaştırması olarak "
            "kalmalıdır. Sezon başlangıcı için `main_start_rating`, sezon sonu için `main_end_live_rating` ve "
            "hareket için `main_live_change` birlikte saklanmalıdır.",
            styles["callout"],
        ),
        Paragraph("Sonraki kanıt adımı", styles["h2"]),
        Paragraph(
            "2026/27 sezonu başlamadan parametreler, kimlik registry'si ve maç öncesi log sözleşmesi dondurulmalı. "
            "Prospective sezonda aynı dört kol sonuç görülmeden paralel yürütülmeli; shadow statüsü ancak fold ve "
            "turnuva guardrail'leri birlikte geçerse yeniden değerlendirilmelidir.",
            styles["body"],
        ),
        PageBreak(),
    ]


def full_team_appendix(data: dict[str, object], styles: dict[str, ParagraphStyle]) -> list[object]:
    ratings = data["ratings"]
    assert isinstance(ratings, pd.DataFrame)
    rows: list[list[object]] = [
        [
            "Final sıra",
            "Takım",
            "Ülke",
            "Kupa",
            "Başlangıç",
            "Ana final",
            "Sezon Δ",
            "Sıra Δ",
            "Surprise final Δ",
            "Progression final Δ",
            "Birleşik final Δ",
            "club_id",
        ]
    ]
    for row in ratings.sort_values(["main_end_rank", "club_id"]).itertuples():
        rows.append(
            [
                f"{int(row.main_end_rank)}",
                row.team_name,
                row.country_code,
                row.participating_competitions,
                f"{row.main_start_rating:.1f}",
                f"{row.main_end_live_rating:.1f}",
                signed(row.main_live_change, 1),
                signed(row.main_rank_gain, 0),
                signed(row.surprise_final_delta_vs_main, 1),
                signed(row.progression_final_delta_vs_main, 1),
                signed(row.combined_final_delta_vs_main, 1),
                row.club_id,
            ]
        )
    table = styled_table(
        rows,
        styles,
        [14 * mm, 50 * mm, 14 * mm, 16 * mm, 22 * mm, 22 * mm, 20 * mm, 18 * mm, 25 * mm, 28 * mm, 25 * mm, 33 * mm],
        long=True,
        small=True,
    )
    return [
        Paragraph("Ek A. 236 Takımın Başlangıç ve Bitiş Elo Karşılaştırması", styles["h1"]),
        Paragraph(
            "Sıralama ana final-aday modelin sezon sonu Power Elo değerine göredir. Shadow delta kolonları "
            "ilgili kolun final ratingi eksi ana final ratingidir; kullanıcıya gösterilecek ikinci bir Elo değildir.",
            styles["small"],
        ),
        table,
    ]


def start_final_scatter(ratings: pd.DataFrame) -> Drawing:
    width, height = 250 * mm, 88 * mm
    drawing = Drawing(width, height)
    left, right, bottom, top = 17 * mm, 8 * mm, 13 * mm, 8 * mm
    plot_w, plot_h = width - left - right, height - bottom - top
    x = ratings["main_start_rating"].astype(float)
    y = ratings["main_end_live_rating"].astype(float)
    low = math.floor(min(x.min(), y.min()) / 100) * 100
    high = math.ceil(max(x.max(), y.max()) / 100) * 100
    scale_x = lambda value: left + (value - low) / (high - low) * plot_w
    scale_y = lambda value: bottom + (value - low) / (high - low) * plot_h
    for tick in range(int(low), int(high) + 1, 300):
        px, py = scale_x(tick), scale_y(tick)
        drawing.add(Line(px, bottom, px, bottom + plot_h, strokeColor=MID, strokeWidth=0.4))
        drawing.add(Line(left, py, left + plot_w, py, strokeColor=MID, strokeWidth=0.4))
        drawing.add(String(px, bottom - 4 * mm, str(tick), fontName="AORegular", fontSize=6.5, textAnchor="middle", fillColor=MUTED))
        drawing.add(String(left - 2 * mm, py - 2, str(tick), fontName="AORegular", fontSize=6.5, textAnchor="end", fillColor=MUTED))
    drawing.add(Line(scale_x(low), scale_y(low), scale_x(high), scale_y(high), strokeColor=INK, strokeWidth=0.8, strokeDashArray=[3, 3]))
    palette = {"UCL": (GOLD, "circle"), "UEL": (TEAL, "square"), "UECL": (CORAL, "triangle")}
    for row in ratings.itertuples():
        competition = str(row.participating_competitions).split("/")[0]
        color, marker = palette.get(competition, (BLUE, "circle"))
        px, py = scale_x(float(row.main_start_rating)), scale_y(float(row.main_end_live_rating))
        if marker == "square":
            drawing.add(Rect(px - 1.5, py - 1.5, 3, 3, fillColor=color, strokeColor=None, fillOpacity=0.65))
        else:
            drawing.add(Circle(px, py, 1.7, fillColor=color, strokeColor=None, fillOpacity=0.65))
    label_positions = {
        "Arsenal": (-4 * mm, 4 * mm, "end"),
        "Bayern München": (4 * mm, -1 * mm, "start"),
        "Paris Saint-Germain": (4 * mm, -6 * mm, "start"),
        "Aston Villa": (4 * mm, 3 * mm, "start"),
    }
    for row in ratings.nsmallest(4, "main_end_rank").itertuples():
        dx, dy, anchor = label_positions[row.team_name]
        drawing.add(
            String(
                scale_x(row.main_start_rating) + dx,
                scale_y(row.main_end_live_rating) + dy,
                row.team_name,
                fontName="AORegular",
                fontSize=5.8,
                textAnchor=anchor,
                fillColor=INK,
            )
        )
    drawing.add(String(width / 2, 2 * mm, "Başlangıç AO First Elo", fontName="AOBold", fontSize=7.5, textAnchor="middle", fillColor=INK))
    drawing.add(String(2 * mm, height / 2, "Final Power Elo", fontName="AOBold", fontSize=7.5, fillColor=INK, angle=90, textAnchor="middle"))
    legend_x = width - 54 * mm
    for index, (name, (color, _)) in enumerate(palette.items()):
        x0 = legend_x + index * 17 * mm
        drawing.add(Circle(x0, height - 4 * mm, 2, fillColor=color, strokeColor=None))
        drawing.add(String(x0 + 3 * mm, height - 5.5 * mm, name, fontName="AORegular", fontSize=7, fillColor=INK))
    return drawing


def rank_mover_chart(risers: pd.DataFrame, fallers: pd.DataFrame) -> Drawing:
    values = pd.concat([fallers.sort_values("main_rank_gain"), risers.sort_values("main_rank_gain")])
    width, height = 250 * mm, 92 * mm
    drawing = Drawing(width, height)
    label_w, center, right = 49 * mm, 142 * mm, 8 * mm
    bar_w = width - label_w - right
    max_abs = max(abs(values["main_rank_gain"].min()), abs(values["main_rank_gain"].max()))
    zero_x = label_w + bar_w * 0.50
    row_h = (height - 8 * mm) / len(values)
    drawing.add(Line(zero_x, 3 * mm, zero_x, height - 3 * mm, strokeColor=INK, strokeWidth=0.8))
    for index, row in enumerate(values.itertuples()):
        y = height - 5 * mm - (index + 0.5) * row_h
        value = float(row.main_rank_gain)
        length = abs(value) / max_abs * bar_w * 0.46
        x0 = zero_x if value >= 0 else zero_x - length
        fill = TEAL if value >= 0 else colors.HexColor("#B9C6CD")
        drawing.add(Rect(x0, y - row_h * 0.30, length, row_h * 0.60, fillColor=fill, strokeColor=INK, strokeWidth=0.3))
        drawing.add(String(label_w - 2 * mm, y - 2, row.team_name, fontName="AORegular", fontSize=6.2, textAnchor="end", fillColor=INK))
        anchor = "start" if value >= 0 else "end"
        label_x = x0 + length + 2 if value >= 0 else x0 - 2
        drawing.add(String(label_x, y - 2, signed(value, 0), fontName="AOBold", fontSize=6.2, textAnchor=anchor, fillColor=INK))
    drawing.add(String(zero_x, height - 1.5 * mm, "0", fontName="AORegular", fontSize=6, textAnchor="middle", fillColor=MUTED))
    return drawing


def shadow_boxplot(ratings: pd.DataFrame) -> Drawing:
    series = [
        ("Surprise", ratings["surprise_final_delta_vs_main"], BLUE),
        ("Progression", ratings["progression_final_delta_vs_main"], GOLD),
        ("Birleşik", ratings["combined_final_delta_vs_main"], TEAL),
    ]
    width, height = 250 * mm, 72 * mm
    drawing = Drawing(width, height)
    left, right, bottom, top = 34 * mm, 10 * mm, 12 * mm, 8 * mm
    values = pd.concat([item[1] for item in series]).astype(float)
    low = math.floor(values.min() / 10) * 10
    high = math.ceil(values.max() / 10) * 10
    scale = lambda value: left + (value - low) / (high - low) * (width - left - right)
    for tick in range(int(low), int(high) + 1, 10):
        x = scale(tick)
        drawing.add(Line(x, bottom, x, height - top, strokeColor=MID, strokeWidth=0.5))
        drawing.add(String(x, bottom - 4 * mm, str(tick), fontName="AORegular", fontSize=6.3, textAnchor="middle", fillColor=MUTED))
    drawing.add(Line(scale(0), bottom, scale(0), height - top, strokeColor=INK, strokeWidth=1))
    for index, (name, data, color) in enumerate(series):
        y = height - top - (index + 0.7) * 16 * mm
        q = data.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).to_dict()
        drawing.add(Line(scale(q[0.0]), y, scale(q[1.0]), y, strokeColor=INK, strokeWidth=0.8))
        drawing.add(Rect(scale(q[0.25]), y - 3 * mm, scale(q[0.75]) - scale(q[0.25]), 6 * mm, fillColor=color, strokeColor=INK, strokeWidth=0.6, fillOpacity=0.55))
        drawing.add(Line(scale(q[0.5]), y - 3 * mm, scale(q[0.5]), y + 3 * mm, strokeColor=INK, strokeWidth=1.2))
        drawing.add(String(left - 4 * mm, y - 2, name, fontName="AOBold", fontSize=7, textAnchor="end", fillColor=INK))
        drawing.add(String(width - right, y - 2, f"medyan {q[0.5]:+.1f} | aralık {q[0.0]:+.1f} / {q[1.0]:+.1f}", fontName="AORegular", fontSize=6.2, textAnchor="end", fillColor=MUTED))
    drawing.add(String(width / 2, 2 * mm, "Shadow final rating - ana final rating (Elo)", fontName="AOBold", fontSize=7.2, textAnchor="middle", fillColor=INK))
    return drawing


def bonus_leader_chart(team_bonus: pd.DataFrame) -> Drawing:
    values = team_bonus.sort_values("bonus", ascending=True)
    width, height = 250 * mm, 83 * mm
    drawing = Drawing(width, height)
    left, right, bottom, top = 60 * mm, 12 * mm, 10 * mm, 5 * mm
    plot_w = width - left - right
    row_h = (height - bottom - top) / max(len(values), 1)
    maximum = max(float(values["bonus"].max()), 1.0)
    for index, row in enumerate(values.itertuples()):
        y = bottom + (index + 0.5) * row_h
        length = float(row.bonus) / maximum * plot_w
        drawing.add(Rect(left, y - row_h * 0.28, length, row_h * 0.56, fillColor=GOLD, strokeColor=INK, strokeWidth=0.3))
        drawing.add(String(left - 2 * mm, y - 2, row.winner_team_name, fontName="AORegular", fontSize=6.3, textAnchor="end", fillColor=INK))
        drawing.add(String(left + length + 2, y - 2, f"+{row.bonus:.0f}", fontName="AOBold", fontSize=6.3, fillColor=INK))
    drawing.add(String(left + plot_w / 2, 2 * mm, "Toplam brüt progression bonusu", fontName="AOBold", fontSize=7.2, textAnchor="middle", fillColor=INK))
    return drawing


def loss_delta_chart(models: pd.DataFrame) -> Drawing:
    shadows = models.loc[~models["model_arm"].eq("MAIN_FINAL_CANDIDATE")].copy()
    labels = {"DOMESTIC_SURPRISE_SHADOW": "Surprise", "PROGRESSION_12_8_4_SHADOW": "Progression", "SURPRISE_PLUS_PROGRESSION_SHADOW": "Birleşik"}
    width, height = 250 * mm, 72 * mm
    drawing = Drawing(width, height)
    left, right, bottom, top = 45 * mm, 12 * mm, 12 * mm, 7 * mm
    plot_w = width - left - right
    all_values = pd.concat([shadows["brier_1x2_delta_vs_main"], shadows["log_loss_1x2_delta_vs_main"]]) * 1000
    low = math.floor(all_values.min() * 10) / 10 - 0.1
    high = max(0.1, math.ceil(all_values.max() * 10) / 10 + 0.1)
    scale = lambda value: left + (value - low) / (high - low) * plot_w
    for tick in [low, (low + high) / 2, 0.0, high]:
        x = scale(tick)
        drawing.add(Line(x, bottom, x, height - top, strokeColor=MID if tick else INK, strokeWidth=0.6 if tick else 1.0))
        drawing.add(String(x, bottom - 4 * mm, f"{tick:.1f}", fontName="AORegular", fontSize=6.2, textAnchor="middle", fillColor=MUTED))
    for index, row in enumerate(shadows.itertuples()):
        y = height - top - (index + 0.8) * 16 * mm
        drawing.add(String(left - 3 * mm, y - 2, labels[row.model_arm], fontName="AOBold", fontSize=7, textAnchor="end", fillColor=INK))
        for offset, metric, color in ((2.2 * mm, row.brier_1x2_delta_vs_main * 1000, BLUE), (-2.2 * mm, row.log_loss_1x2_delta_vs_main * 1000, TEAL)):
            x0, x1 = scale(0.0), scale(metric)
            drawing.add(Rect(min(x0, x1), y + offset - 1.5 * mm, abs(x1 - x0), 3 * mm, fillColor=color, strokeColor=INK, strokeWidth=0.25))
    drawing.add(Rect(width - 60 * mm, height - 5 * mm, 4 * mm, 2.5 * mm, fillColor=BLUE, strokeColor=None))
    drawing.add(String(width - 54 * mm, height - 5.3 * mm, "Brier Δ x1000", fontName="AORegular", fontSize=6.5, fillColor=INK))
    drawing.add(Rect(width - 30 * mm, height - 5 * mm, 4 * mm, 2.5 * mm, fillColor=TEAL, strokeColor=None))
    drawing.add(String(width - 24 * mm, height - 5.3 * mm, "Log Δ x1000", fontName="AORegular", fontSize=6.5, fillColor=INK))
    drawing.add(String(width / 2, 2 * mm, "Negatif değer ana modele göre daha düşük loss demektir", fontName="AOBold", fontSize=7.1, textAnchor="middle", fillColor=INK))
    return drawing


def metric_strip(items: list[tuple[str, str, str]], styles: dict[str, ParagraphStyle]) -> Table:
    cells = []
    for label, value, note in items:
        cells.append(
            Paragraph(
                f"<font size='7' color='#61727E'>{label}</font><br/><font size='17'><b>{value}</b></font><br/><font size='6.5' color='#61727E'>{note}</font>",
                styles["body"],
            )
        )
    table = Table([cells], colWidths=[49 * mm] * len(cells), rowHeights=[25 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE),
                ("BOX", (0, 0), (-1, -1), 0.6, MID),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, MID),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def styled_table(
    rows: list[list[object]],
    styles: dict[str, ParagraphStyle],
    widths: list[float],
    *,
    long: bool = False,
    small: bool = False,
):
    body_style = styles["table_small"] if small else styles["table"]
    formatted = []
    for row_index, row in enumerate(rows):
        style = styles["table_head"] if row_index == 0 else body_style
        formatted.append([Paragraph(html.escape(str(value)), style) for value in row])
    table_class = LongTable if long else Table
    table = table_class(formatted, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), INK),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PALE]),
                ("GRID", (0, 0), (-1, -1), 0.35, MID),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 2.8 if small else 3.4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.8 if small else 3.4),
            ]
        )
    )
    return table


def key_value_table(items, styles, widths=(55 * mm, 190 * mm)) -> Table:
    rows = [
        [
            Paragraph(f"<b>{html.escape(str(label))}</b>", styles["table"]),
            Paragraph(html.escape(str(value)), styles["table"]),
        ]
        for label, value in items
    ]
    table = Table(rows, colWidths=list(widths))
    table.setStyle(
        TableStyle(
            [
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [WHITE, PALE]),
                ("GRID", (0, 0), (-1, -1), 0.35, MID),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def two_column_key_values(items, styles) -> Table:
    midpoint = math.ceil(len(items) / 2)
    left = key_value_table(items[:midpoint], styles, widths=(41 * mm, 79 * mm))
    right = key_value_table(items[midpoint:], styles, widths=(41 * mm, 79 * mm))
    table = Table([[left, right]], colWidths=[124 * mm, 124 * mm])
    table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))
    return table


def bullet_table(items, styles) -> Table:
    rows = [
        [
            Paragraph("•", styles["body"]),
            Paragraph(
                f"<b>{html.escape(str(label))}:</b> {html.escape(str(text))}",
                styles["body"],
            ),
        ]
        for label, text in items
    ]
    table = Table(rows, colWidths=[7 * mm, 240 * mm])
    table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 1), ("RIGHTPADDING", (0, 0), (-1, -1), 2), ("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
    return table


def two_mover_tables(risers, fallers, styles) -> Table:
    def rows(frame):
        result = [["Takım", "Başlangıç", "Final", "Elo Δ", "Sıra Δ"]]
        for row in frame.itertuples():
            result.append([row.team_name, int(row.main_start_rank), int(row.main_end_rank), signed(row.main_live_change, 1), signed(row.main_rank_gain, 0)])
        return styled_table(result, styles, [41 * mm, 20 * mm, 18 * mm, 22 * mm, 18 * mm])
    table = Table([[rows(risers), rows(fallers)]], colWidths=[124 * mm, 124 * mm])
    table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))
    return table


def decision_table(styles) -> Table:
    rows = [
        ["Bileşen", "Durum", "Bu replay'deki yorum"],
        ["Ana AO First + Live Elo", "AKTİF ADAY", "Başlangıç ve maç motoru değişmedi"],
        ["Gol farkı alpha=0.15", "AKTİF ADAY", "Tüm 961 maçta mevcut sözleşme"],
        ["xG ratio=0.30", "AKTİF ADAY", "606 maç; 355 maçta fallback"],
        ["Domestic Surprise", "SHADOW", "Küçük loss sinyali, ranking karışık"],
        ["Progression 12/8/4", "SHADOW", "Küçük UCL ağırlıklı sinyal"],
        ["Surprise + Progression", "DIAGNOSTIC", "Etkileşim görülür; production kolu değildir"],
    ]
    return styled_table(rows, styles, [72 * mm, 44 * mm, 132 * mm])


def signed(value: float, digits: int) -> str:
    return f"{float(value):+.{digits}f}"


if __name__ == "__main__":
    main()
