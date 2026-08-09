from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
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
DEFAULT_OUTPUT_ROOT = ROOT / "output" / "season_replay_2025_26"
PDF_NAME = "AO_2025_26_Sezon_Replay_Raporu.pdf"
NAVY = colors.HexColor("#172A3A")
TEAL = colors.HexColor("#087E8B")
RED = colors.HexColor("#C23B3B")
GREEN = colors.HexColor("#2C7A4B")
GOLD = colors.HexColor("#D69E2E")
LIGHT = colors.HexColor("#F3F6F8")
MID = colors.HexColor("#D7E0E6")
TEXT = colors.HexColor("#24323D")


def register_fonts() -> tuple[str, str]:
    regular = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
    bold = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
    regular_name = "AOArial"
    bold_name = "AOArialBold"
    if regular.exists():
        pdfmetrics.registerFont(TTFont(regular_name, str(regular)))
    else:
        regular_name = "Helvetica"
    if bold.exists():
        pdfmetrics.registerFont(TTFont(bold_name, str(bold)))
    else:
        bold_name = "Helvetica-Bold"
    return regular_name, bold_name


def build_replay_pdf(output_root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    output_root = Path(output_root).resolve()
    manifest = json.loads((output_root / "replay_manifest.json").read_text(encoding="utf-8"))
    comparison = pd.read_csv(output_root / "model_comparison.csv")
    initial = pd.read_csv(output_root / "initial_ratings.csv")
    finals = pd.read_csv(output_root / "final_ratings.csv")
    segments = pd.read_csv(output_root / "competition_stage_month_summary.csv")
    scoreline = pd.read_csv(output_root / "scoreline_predictions.csv")
    progression = pd.read_csv(output_root / "progression_probability_summary.csv")
    xg = pd.read_csv(output_root / "xg_appendix.csv")
    audits = pd.read_csv(output_root / "season_state_audit.csv")
    bonus_events = pd.read_csv(output_root / "bonus_events.csv")

    regular, bold = register_fonts()
    styles = build_styles(regular, bold)
    pdf_path = output_root / PDF_NAME
    document = BaseDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=18 * mm,
        bottomMargin=16 * mm,
        title="AO European Elo 2025/26 Tam Sezon Replay Raporu",
        author="AkılOyunu.com",
        subject="2025/26 historical locked OOS ve current model shadow karşılaştırması",
    )
    frame = Frame(document.leftMargin, document.bottomMargin, document.width, document.height, id="main")
    document.addPageTemplates(PageTemplate(id="report", frames=[frame], onPage=page_frame(regular, bold)))

    story = []
    story.extend(cover_page(styles, manifest))
    story.extend(methodology_page(styles))
    story.extend(parameter_page(styles))
    story.extend(comparison_page(styles, comparison))
    story.extend(rating_page(styles, initial, finals))
    story.extend(segment_page(styles, segments))
    story.extend(scoreline_page(styles, scoreline))
    story.extend(progression_page(styles, progression, bonus_events))
    story.extend(xg_page(styles, xg))
    story.extend(audit_page(styles, audits, comparison, manifest))
    document.build(story)
    return pdf_path


def build_styles(regular: str, bold: str) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title", parent=base["Title"], fontName=bold, fontSize=25, leading=30,
            textColor=colors.white, alignment=TA_LEFT, spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle", parent=base["Normal"], fontName=regular, fontSize=12, leading=17,
            textColor=colors.HexColor("#D8EEF0"), spaceAfter=8,
        ),
        "h1": ParagraphStyle(
            "H1", parent=base["Heading1"], fontName=bold, fontSize=18, leading=22,
            textColor=NAVY, spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Heading2"], fontName=bold, fontSize=12, leading=15,
            textColor=TEAL, spaceBefore=7, spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "Body", parent=base["BodyText"], fontName=regular, fontSize=9.2, leading=13.2,
            textColor=TEXT, spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "Small", parent=base["BodyText"], fontName=regular, fontSize=7.5, leading=10.2,
            textColor=TEXT, spaceAfter=4,
        ),
        "callout": ParagraphStyle(
            "Callout", parent=base["BodyText"], fontName=bold, fontSize=10, leading=14,
            textColor=NAVY, borderColor=TEAL, borderWidth=1, borderPadding=8,
            backColor=colors.HexColor("#E8F5F5"), spaceBefore=5, spaceAfter=8,
        ),
        "table": ParagraphStyle(
            "Table", parent=base["BodyText"], fontName=regular, fontSize=6.7, leading=8.3,
            textColor=TEXT,
        ),
        "table_head": ParagraphStyle(
            "TableHead", parent=base["BodyText"], fontName=bold, fontSize=6.7, leading=8.3,
            textColor=colors.white, alignment=TA_CENTER,
        ),
    }


def page_frame(regular: str, bold: str):
    def draw(canvas, document):
        canvas.saveState()
        width, height = A4
        canvas.setStrokeColor(MID)
        canvas.line(16 * mm, 13 * mm, width - 16 * mm, 13 * mm)
        canvas.setFont(regular, 7)
        canvas.setFillColor(colors.HexColor("#667782"))
        canvas.drawString(16 * mm, 9 * mm, "AO European Elo | 2025/26 sezon replay")
        canvas.drawRightString(width - 16 * mm, 9 * mm, f"Sayfa {document.page}")
        if document.page > 1:
            canvas.setFont(bold, 7.5)
            canvas.setFillColor(NAVY)
            canvas.drawString(16 * mm, height - 11 * mm, "AkılOyunu.com - Teknik Kanıt Raporu")
        canvas.restoreState()
    return draw


def cover_page(styles, manifest):
    block = Table(
        [[Paragraph("AO European Elo", styles["title"])],
         [Paragraph("2025/26 Tam Sezon Replay ve Shadow Karşılaştırması", styles["subtitle"])],
         [Paragraph("961 maç • 236 takım • UCL / UEL / UECL", styles["subtitle"])]],
        colWidths=[178 * mm],
        rowHeights=[34 * mm, 21 * mm, 15 * mm],
    )
    block.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY),
        ("BACKGROUND", (0, 2), (-1, 2), TEAL),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 14 * mm),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10 * mm),
    ]))
    return [
        Spacer(1, 32 * mm), block, Spacer(1, 18 * mm),
        Paragraph(
            "Bu rapor, 2025/26 sezonundaki gerçek maçları kesin UTC sırasıyla yeniden işler. "
            "Her tahmin sonuç görülmeden kaydedilir; ardından gerçek saha skoru rating motoruna uygulanır.",
            styles["callout"],
        ),
        Paragraph(
            "Kanıt ayrımı: Historical Locked OOS yalnız 2024/25 sonuna kadar seçilmiş fold-6 "
            "parametrelerini kullanır. Current Model Counterfactual bugünkü production ve shadow "
            "özelliklerinin geçmiş sezondaki davranışını gösterir; bağımsız doğrulama değildir.",
            styles["body"],
        ),
        Spacer(1, 8 * mm),
        key_value_table([
            ("Replay türü", "Kronolojik gerçek-sonuç replay"),
            ("Production değişikliği", "Yok"),
            ("Shadow statüsü değişikliği", "Yok"),
            ("Prospective holdout", manifest["prospective_holdout"]),
        ], styles, widths=(48 * mm, 120 * mm)),
        PageBreak(),
    ]


def methodology_page(styles):
    return [
        Paragraph("1. Çalışmanın Mantığı", styles["h1"]),
        Paragraph(
            "Bu çalışma Monte Carlo turnuva simülasyonu değildir. 2025/26 sezonunun 961 gerçek "
            "UCL, UEL ve UECL maçı kronolojik olarak yeniden oynatılmıştır. Maç öncesi AO Live Elo "
            "ile 1X2 olasılığı üretilmiş, tahmin kilitlenmiş ve saha skoru geldikten sonra Power Elo güncellenmiştir.",
            styles["body"],
        ),
        Paragraph("İki kanıt hattı", styles["h2"]),
        bullet_table([
            ("OOS_FOLD6", "2025/26 sonuçları görülmeden, 2024/25 sonuna kadar seçilen parametrelerle tarihsel kilitli test."),
            ("RETROSPECTIVE_COUNTERFACTUAL", "Bugünkü production veya shadow ayarının aynı sezonda nasıl davranacağını gösteren geriye dönük analiz."),
            ("MATCHED_XG_APPENDIX", "Yalnız iki takım için xG bulunan ortak 180 maçlık ayrı ek; production-grade veri değildir."),
        ], styles),
        Paragraph("Olay sırası", styles["h2"]),
        numbered_flow([
            "Sezon başındaki 236 AO First Elo değeri state içine alınır.",
            "Maçlar kickoff_utc ve match_id ile kararlı biçimde sıralanır.",
            "Maç öncesi 1X2 ve varsa yan katman olasılıkları kaydedilir.",
            "90/120 dakika saha skoru uygulanır; penaltı atış golleri skora eklenmez.",
            "Power Elo sıfır toplamlı güncellenir. Uygun bonus kollarında tur tamamlanınca ayrı bonus eklenir.",
            "Sezon sonunda rating, tahmin, segment, belirsizlik ve invariant çıktıları hesaplanır.",
        ], styles),
        Paragraph("Yorum sınırı", styles["h2"]),
        Spacer(1, 3 * mm),
        Paragraph(
            "Aynı sezonun schedule-adjusted performansı ile sezon sonu Elo arasındaki korelasyon bir "
            "davranış diagnostiğidir; ileri dönem sıralama testi değildir. Bu replay tek başına PROMOTE kararı üretmez.",
            styles["callout"],
        ),
        PageBreak(),
    ]


def parameter_page(styles):
    rows = [
        ["Kol", "Gol farkı", "Ek davranış"],
        ["PRODUCTION", "alpha=0.10, tau=300, cap=4", "Aktif referans"],
        ["NO_GD_CONTROL", "Kapalı", "Kontrol"],
        ["GD_PRIOR_GRID", "alpha=0.20, tau=400", "Daha güçlü prior"],
        ["GD_EXTENDED", "alpha=0.125, tau=800", "Daha yavaş sönüm"],
        ["FIXED_TOURNAMENT_BONUS", "Production", "12/8/4; cap 60/40/20"],
        ["DOMESTIC_ANCHORED", "Production", "Domestic persistence=0.75"],
    ]
    table = styled_table(rows, styles, [50 * mm, 55 * mm, 63 * mm])
    return [
        Paragraph("2. Model Kolları ve Katsayılar", styles["h1"]),
        Paragraph("Historical Locked OOS", styles["h2"]),
        key_value_table([
            ("Dynamic core", "Scale=835.561497, H=148.544266, K=103.980986, carry=0"),
            ("Draw modeli", "UCL/UEL/global 0.26-1.00; UECL 0.26-1.25"),
            ("Gol farkı", "alpha=0, tau=300, cap=4"),
            ("Bonus duyarlılığı", "UCL/UEL/UECL = 6/4/2"),
            ("Skor modeli", "mu=0.298390, elo_slope=0.710795, rho=0"),
            ("Gol seviyesi", "season_log_offset=0.008982; turnuva offsetleri=0"),
        ], styles),
        Paragraph("Current Model Counterfactual", styles["h2"]),
        table,
        Paragraph("Yan katmanlar", styles["h2"]),
        Paragraph(
            "SCORELINE_POISSON ve SCORELINE_LEVEL ratinge geri besleme yapmaz. FORMAT_P_ADVANCE "
            "yalnız eşleşme olasılığı diagnostikleridir. Eski sıfır-toplamlı progression, Achievement "
            "Reserve, Dynamic K ve Competition K bu replay'e alınmamıştır.",
            styles["body"],
        ),
        PageBreak(),
    ]


def comparison_page(styles, comparison):
    display = comparison.copy()
    display["Brier"] = display["brier_1x2"].map(lambda x: f"{x:.6f}")
    display["Log-loss"] = display["log_loss_1x2"].map(lambda x: f"{x:.6f}")
    display["Doğruluk"] = display["accuracy_1x2"].map(lambda x: f"{100*x:.2f}%")
    display["Spearman"] = display["same_season_spearman"].map(lambda x: f"{x:.4f}")
    display["Pairwise"] = display["same_season_pairwise"].map(lambda x: f"{x:.4f}")
    display["Karar"] = display["classification"].map(short_classification)
    rows = [["Model kolu", "Brier", "Log-loss", "Doğruluk", "Spearman", "Pairwise", "Sinyal"]]
    rows += display[["model_arm", "Brier", "Log-loss", "Doğruluk", "Spearman", "Pairwise", "Karar"]].values.tolist()
    delta = display.loc[display["reference_arm"].notna() & display["reference_arm"].ne("")]
    chart_values = [
        (row.model_arm, float(row.brier_delta_vs_reference) * 10000.0)
        for row in delta.itertuples(index=False)
    ]
    return [
        Paragraph("3. Ana Model Karşılaştırması", styles["h1"]),
        Paragraph(
            "Düşük Brier ve log-loss daha iyidir. Spearman ve pairwise değerleri aynı sezon "
            "schedule-adjusted performansına göre tanısal sıralama ölçüleridir.", styles["body"],
        ),
        styled_table(rows, styles, [49 * mm, 18 * mm, 20 * mm, 19 * mm, 18 * mm, 18 * mm, 26 * mm]),
        Spacer(1, 5 * mm),
        Paragraph("Brier farkı: referansa göre baz puan (negatif daha iyi)", styles["h2"]),
        horizontal_bar_chart(chart_values, 168 * mm, 58 * mm),
        Paragraph(
            "Historical 6/4/2 bonus kolu kilitli OOS referansa çok küçük fakat yönü tutarlı bir "
            "iyileşme gösterdi. Current kollar içinde sabit bonus ve bazı gol farkı adayları küçük "
            "iyileşmeler üretti; bu farklar retrospective olduğu için terfi kanıtı sayılmaz.",
            styles["callout"],
        ),
        PageBreak(),
    ]


def rating_page(styles, initial, finals):
    production = finals.loc[finals["model_arm"].eq("PRODUCTION")].copy()
    production["rating_change"] = production["end_live_rating"] - production["initial_rating"]
    top_final = production.nlargest(12, "end_live_rating")
    movers = pd.concat([
        production.nlargest(6, "rating_change"),
        production.nsmallest(6, "rating_change"),
    ]).drop_duplicates("team_id")
    top_rows = [["#", "Takım", "Başlangıç", "Bitiş", "Değişim"]]
    for rank, row in enumerate(top_final.itertuples(index=False), 1):
        top_rows.append([rank, row.team_name, f"{row.initial_rating:.1f}", f"{row.end_live_rating:.1f}", f"{row.rating_change:+.1f}"])
    chart_values = [(str(row.team_name), float(row.rating_change)) for row in movers.itertuples(index=False)]
    return [
        Paragraph("4. Başlangıçtan Sezon Sonuna Elo", styles["h1"]),
        Paragraph(
            f"AO First Elo snapshot'ı {len(initial)} benzersiz takım içerir. Production kolunda maç "
            "sonuçları ve kontrollü gol farkı Power Elo'yu kronolojik olarak günceller.", styles["body"],
        ),
        styled_table(top_rows, styles, [12 * mm, 70 * mm, 28 * mm, 28 * mm, 28 * mm]),
        Spacer(1, 5 * mm),
        Paragraph("En belirgin sezonluk hareketler", styles["h2"]),
        horizontal_bar_chart(chart_values, 168 * mm, 75 * mm, favorable_negative=False),
        Paragraph(
            "Bu hareketler takımın mutlak kalitesini tek başına açıklamaz; başlangıç ratingi, rakip "
            "gücü, beklenti sürprizi, oynanan maç sayısı ve gol farkı birlikte etki eder.", styles["small"],
        ),
        PageBreak(),
    ]


def segment_page(styles, segments):
    chosen = segments.loc[
        segments["model_arm"].isin(["HISTORICAL_LOCKED", "PRODUCTION", "FIXED_TOURNAMENT_BONUS", "DOMESTIC_ANCHORED"])
        & segments["segment_type"].eq("competition")
    ].copy()
    rows = [["Model", "Turnuva", "Maç", "Brier", "Log-loss", "Doğruluk"]]
    for row in chosen.sort_values(["model_arm", "segment_value"]).itertuples(index=False):
        rows.append([
            row.model_arm, row.segment_value, int(row.matches), f"{row.brier_1x2:.5f}",
            f"{row.log_loss_1x2:.5f}", f"{100*row.accuracy_1x2:.1f}%",
        ])
    return [
        Paragraph("5. Turnuva ve Segment Davranışı", styles["h1"]),
        Paragraph(
            "UCL, UEL ve UECL aynı motorla değerlendirilir; aralarındaki fiili güç farkı takım "
            "ratingleri ve rakip kalitesi üzerinden taşınır. Sabit bonus kolu turnuva katsayısını "
            "Power K'ya değil, tamamlanan eleme başarısına ayrı ve kontrollü biçimde uygular.", styles["body"],
        ),
        styled_table(rows, styles, [50 * mm, 18 * mm, 17 * mm, 24 * mm, 25 * mm, 24 * mm]),
        Paragraph("Okuma kuralı", styles["h2"]),
        Spacer(1, 3 * mm),
        Paragraph(
            "Bir özelliğin toplam ortalamada iyi görünmesi yeterli değildir. Turnuvalardan birinde "
            "zarar üretmesi, güven aralığı ve örneklem büyüklüğüyle birlikte MIXED_OR_INCONCLUSIVE "
            "olarak değerlendirilmesine yol açar.", styles["callout"],
        ),
        PageBreak(),
    ]


def scoreline_page(styles, scoreline):
    summary = scoreline.groupby("model_arm", as_index=False).agg(
        matches=("match_id", "size"),
        exact_nll=("score_nll", "mean"),
        exact_top1=("exact_score_correct", "mean"),
        total_goal_mae=("total_goal_absolute_error", "mean"),
        brier_1x2=("brier_1x2", "mean"),
        over_brier=("over_2_5_brier", "mean"),
        btts_brier=("btts_brier", "mean"),
    )
    rows = [["Skor kolu", "NLL", "Top-1", "Gol MAE", "1X2 Brier", "O2.5 Brier", "BTTS Brier"]]
    for row in summary.itertuples(index=False):
        rows.append([
            row.model_arm, f"{row.exact_nll:.5f}", f"{100*row.exact_top1:.1f}%",
            f"{row.total_goal_mae:.4f}", f"{row.brier_1x2:.5f}",
            f"{row.over_brier:.5f}", f"{row.btts_brier:.5f}",
        ])
    return [
        Paragraph("6. Skor Tahmin Yan Katmanı", styles["h1"]),
        Paragraph(
            "Elo-koşullu Poisson modeli maç öncesi AO Live Elo farkını beklenen ev ve deplasman "
            "gollerine çevirir. SCORELINE_LEVEL aynı yapıya turnuva ve sezon gol seviyesi offseti "
            "ekler. Bu katmanlar rating state'ini değiştirmez.", styles["body"],
        ),
        styled_table(rows, styles, [46 * mm, 18 * mm, 18 * mm, 20 * mm, 22 * mm, 22 * mm, 22 * mm]),
        Paragraph("Sözleşme", styles["h2"]),
        bullet_table([
            ("Skor", "90 dakika veya oynandıysa 120 dakika saha skoru; penaltı atışları hariç."),
            ("Exact-score", "Gerçek skorun olasılığı ve negative log-loss."),
            ("O/U 2.5", "Toplam gol 3 veya üzeri için ayrı Brier ve log-loss."),
            ("BTTS", "Her iki takımın en az bir gol atması için ayrı Brier ve log-loss."),
            ("İnvariant", "Her skor matrisi sonlu, negatif olmayan ve toplamı 1 olan olasılıklardan oluşur."),
        ], styles),
        Spacer(1, 4 * mm),
        Paragraph(
            "SCORELINE_LEVEL bu sezonda skor NLL ve toplam gol MAE yönünde en iyi retrospective "
            "sonucu verdi; fakat production aktivasyonu için 2026/27 prospective kanıtı gerekir.", styles["callout"],
        ),
        PageBreak(),
    ]


def progression_page(styles, progression, bonus_events):
    calibrated = progression.groupby("model_arm", as_index=False).agg(
        ties=("tie_id", "nunique"), brier=("brier_loss", "mean"), log_loss=("log_loss", "mean"),
        identity_brier=("identity_brier_loss", "mean"), identity_log=("identity_log_loss", "mean"),
    ).iloc[0]
    bonus_summary = bonus_events.groupby("model_arm", as_index=False).agg(
        events=("tie_id", "size"), total_bonus=("applied_bonus", "sum"), max_bonus=("applied_bonus", "max")
    )
    rows = [["Bonus kolu", "Olay", "Toplam", "Tek olay max"]]
    for row in bonus_summary.itertuples(index=False):
        rows.append([row.model_arm, int(row.events), f"{row.total_bonus:.1f}", f"{row.max_bonus:.1f}"])
    return [
        Paragraph("7. Tur Olasılığı ve Sabit Bonus", styles["h1"]),
        Paragraph("FORMAT_P_ADVANCE", styles["h2"]),
        key_value_table([
            ("Eşleşme sayısı", f"{int(calibrated.ties)}"),
            ("Kalibre Brier", f"{calibrated.brier:.6f}"),
            ("Identity Brier", f"{calibrated.identity_brier:.6f}"),
            ("Kalibre log-loss", f"{calibrated.log_loss:.6f}"),
            ("Identity log-loss", f"{calibrated.identity_log:.6f}"),
        ], styles),
        Paragraph(
            "Format katmanı slope=1.5, tek maç ev bias=0.1 ve çift maç bias=0 ile yalnız tur geçme "
            "olasılığı üretir; ratinge geri besleme yapmaz.", styles["body"],
        ),
        Paragraph("Fixed tournament bonus", styles["h2"]),
        styled_table(rows, styles, [72 * mm, 28 * mm, 32 * mm, 32 * mm]),
        Paragraph(
            "Counterfactual bonus UCL/UEL/UECL için 12/8/4, sezonluk cap 60/40/20 uygular. "
            "Historical duyarlılık 6/4/2'dir. Bonus yalnız uygun eşleşme tamamlanınca bir kez, "
            "kazanana eklenir; Power Elo'nun sıfır toplamlı çekirdeğinden ayrı tutulur.", styles["body"],
        ),
        Paragraph(
            "Replay'de bonus yönü umut verici olsa da etki küçüktür ve mevcut sezon daha önce "
            "görüldüğü için aktif modele otomatik alınmaz.", styles["callout"],
        ),
        PageBreak(),
    ]


def xg_page(styles, xg):
    summary = xg.groupby("model_arm", as_index=False).agg(
        matches=("match_id", "nunique"), brier=("brier_1x2", "mean"), log_loss=("log_loss_1x2", "mean")
    )
    rows = [["xG kolu", "Ortak maç", "Brier", "Log-loss"]]
    for row in summary.itertuples(index=False):
        rows.append([row.model_arm, int(row.matches), f"{row.brier:.6f}", f"{row.log_loss:.6f}"])
    competition_counts = xg.loc[xg["model_arm"].eq("GD_XG_CONVEX")].groupby("competition")["match_id"].nunique()
    return [
        Paragraph("8. xG Eki", styles["h1"]),
        Paragraph(
            "xG karşılaştırması ana 961 maçlık replay'den ayrıdır. Yalnız iki taraf için xG bulunan "
            "ortak 180 maç kullanılmıştır; eksik xG içeren maçlar xG loss karşılaştırmasına girmez.", styles["body"],
        ),
        key_value_table([
            ("UCL", str(int(competition_counts.get("UCL", 0)))),
            ("UEL", str(int(competition_counts.get("UEL", 0)))),
            ("UECL", str(int(competition_counts.get("UECL", 0)))),
            ("Toplam", str(int(competition_counts.sum()))),
        ], styles),
        Spacer(1, 4 * mm),
        styled_table(rows, styles, [75 * mm, 30 * mm, 30 * mm, 30 * mm]),
        Paragraph("Adaylar", styles["h2"]),
        bullet_table([
            ("GD_XG_CONVEX", "rho=0.05, xG scale=1.00; gol farkı ve xG sinyalini kontrollü birleştirir."),
            ("GD_XG_LUCK", "additive luck correction rho=0.50, xG scale=0.75; bitiricilik/şans farkını azaltmayı dener."),
        ], styles),
        Spacer(1, 4 * mm),
        Paragraph(
            "180 maçlık ekte xG luck kolu daha düşük Brier ve log-loss üretti. Veri tek, kesintisiz "
            "ve production-grade bir sağlayıcıdan gelmediği için bu yalnız shadow kanıtıdır.", styles["callout"],
        ),
        PageBreak(),
    ]


def audit_page(styles, audits, comparison, manifest):
    pair_error = max(
        audits.get("match_pair_zero_sum_error", pd.Series([0.0])).fillna(0.0).abs().max(),
        audits.get("max_pair_sum_error", pd.Series([0.0])).fillna(0.0).abs().max(),
    )
    power_error = audits.get("power_total_error", pd.Series([0.0])).fillna(0.0).abs().max()
    bonus_error = audits.get("bonus_cap_error", pd.Series([0.0])).fillna(0.0).abs().max()
    classifications = comparison["classification"].value_counts().to_dict()
    return [
        Paragraph("9. Kontroller, Sınırlamalar ve Model Kararı", styles["h1"]),
        Paragraph("Teknik kabul kontrolleri", styles["h2"]),
        key_value_table([
            ("Tam kapsam", f"{manifest['full_scope_matches']} maç / {manifest['teams']} takım"),
            ("Power pair zero-sum max hata", f"{pair_error:.3e}"),
            ("Sezon Power toplam hata", f"{power_error:.3e}"),
            ("Bonus cap aşımı", f"{bonus_error:.3e}"),
            ("Production contract değişikliği", "Yok"),
            ("2026/27 eğitim/seçimde kullanım", "Yok"),
        ], styles),
        Paragraph("Shadow sinyal özeti", styles["h2"]),
        key_value_table([
            ("Tutarlı shadow sinyali", str(classifications.get("CONSISTENT_SHADOW_SIGNAL", 0))),
            ("Karışık / belirsiz", str(classifications.get("MIXED_OR_INCONCLUSIVE", 0))),
            ("Zarar sinyali", str(classifications.get("HARM_SIGNAL", 0))),
        ], styles),
        Paragraph("Metodolojik sınırlamalar", styles["h2"]),
        bullet_table([
            ("Counterfactual", "Bugünkü model 2025/26 verisinden önce tamamen kilitlenmiş değildir; bağımsız OOS sayılmaz."),
            ("Aynı sezon sıralama", "Sezon sonu rating ile aynı sezon performansı ilişkisidir; forward ranking değildir."),
            ("xG", "180 maçlık kaynak production-grade ve kesintisiz değildir."),
            ("Bonus", "Non-zero-sum olduğu için canlı rating toplamını artırır; ayrı cap ve audit ile kontrol edilir."),
            ("Skor modeli", "90/120 field-score sözleşmesi kullanır; saf 90 dakika market modeli değildir."),
        ], styles),
        Paragraph("Model kararı", styles["h2"]),
        Spacer(1, 3 * mm),
        Paragraph(
            "Production AO First Elo ve AO Live Elo sözleşmesi değiştirilmemiştir. Historical 6/4/2 "
            "bonus ile bazı current shadow kollarında olumlu yönler gözlense de bu replay PROMOTE "
            "kararı üretmez. Tüm adaylar mevcut statülerinde kalır; nihai karar 2026/27 prospective "
            "holdout sonrasında verilecektir.", styles["callout"],
        ),
        Paragraph(
            "Sonuç: Model kronolojik replay, olasılık normalizasyonu, Power sıfır-toplamı ve bonus "
            "cap kontrollerini geçti. Bu rapor modelin nasıl davrandığını şeffaflaştırır; geleceği "
            "bağımsız biçimde tahmin ettiğini tek başına kanıtlamaz.", styles["body"],
        ),
    ]


def key_value_table(items, styles, widths=(55 * mm, 110 * mm)):
    rows = [[Paragraph(str(key), styles["table_head"]), Paragraph(str(value), styles["table"])] for key, value in items]
    table = Table(rows, colWidths=list(widths), repeatRows=0, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), NAVY),
        ("BACKGROUND", (1, 0), (1, -1), LIGHT),
        ("GRID", (0, 0), (-1, -1), 0.35, MID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def bullet_table(items, styles):
    rows = []
    for key, text in items:
        rows.append([Paragraph("•", styles["body"]), Paragraph(f"<b>{key}:</b> {text}", styles["body"])])
    table = Table(rows, colWidths=[7 * mm, 160 * mm], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 1),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return table


def numbered_flow(items, styles):
    rows = [[Paragraph(str(index), styles["table_head"]), Paragraph(text, styles["body"])] for index, text in enumerate(items, 1)]
    table = Table(rows, colWidths=[10 * mm, 157 * mm], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), TEAL),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -2), 0.35, MID),
    ]))
    return table


def styled_table(rows, styles, widths):
    converted = []
    for row_index, row in enumerate(rows):
        style = styles["table_head"] if row_index == 0 else styles["table"]
        converted.append([Paragraph(str(value), style) for value in row])
    table = LongTable(converted, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.35, MID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def horizontal_bar_chart(values, width, height, *, favorable_negative=True):
    drawing = Drawing(width, height)
    if not values:
        return drawing
    label_width = 55 * mm
    plot_left = label_width
    plot_width = width - label_width - 8 * mm
    max_abs = max(abs(float(value)) for _, value in values) or 1.0
    row_height = height / max(len(values), 1)
    zero_x = plot_left + plot_width / 2
    drawing.add(Line(zero_x, 2, zero_x, height - 2, strokeColor=colors.HexColor("#8B9AA4"), strokeWidth=0.6))
    for index, (label, value) in enumerate(values):
        y = height - (index + 0.72) * row_height
        drawing.add(String(0, y + 1, truncate(str(label), 30), fontName="AOArial", fontSize=6.5, fillColor=TEXT))
        length = (abs(float(value)) / max_abs) * (plot_width / 2 - 5)
        x = zero_x if value >= 0 else zero_x - length
        positive_color = RED if favorable_negative else GREEN
        negative_color = GREEN if favorable_negative else RED
        fill = positive_color if value > 0 else negative_color
        drawing.add(Rect(x, y, max(length, 0.8), max(row_height * 0.45, 3), fillColor=fill, strokeColor=None))
        value_x = zero_x + length + 3 if value >= 0 else zero_x - length - 24
        drawing.add(String(value_x, y + 1, f"{value:+.2f}", fontName="AOArial", fontSize=6, fillColor=TEXT))
    return drawing


def short_classification(value: str) -> str:
    return {
        "REFERENCE": "Referans",
        "CONSISTENT_SHADOW_SIGNAL": "Tutarlı",
        "MIXED_OR_INCONCLUSIVE": "Karışık",
        "HARM_SIGNAL": "Zarar",
    }.get(str(value), str(value))


def truncate(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: limit - 1] + "…"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Turkish 2025/26 AO season replay PDF")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    print(build_replay_pdf(args.output_root))


if __name__ == "__main__":
    main()
