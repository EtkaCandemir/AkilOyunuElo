from __future__ import annotations

import shutil
from html import escape
from pathlib import Path

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
PDF_DIR = ROOT / "output" / "pdf"
DETAILED_PDF = PDF_DIR / "AkilOyunu_Elo_Model_Aciklayici.pdf"
SHORT_PDF = PDF_DIR / "AkilOyunu_Elo_Model_Kisa.pdf"
LEGACY_DETAILED_PDF = ROOT / DETAILED_PDF.name
LEGACY_SHORT_PDF = ROOT / SHORT_PDF.name
FONT_REGULAR = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
DOCUMENT_DATE = "23 Temmuz 2026"
VERSION_LABEL = "AO European Elo v2 - Operasyonel Sözleşme 1.4"


def main() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    register_fonts()
    build_document(
        DETAILED_PDF,
        detailed_story(),
        "AO European Elo v2 - Tam Teknik Model Açıklaması",
    )
    build_document(
        SHORT_PDF,
        short_story(),
        "AO European Elo v2 - Kısa Model Açıklaması",
    )
    shutil.copyfile(DETAILED_PDF, LEGACY_DETAILED_PDF)
    shutil.copyfile(SHORT_PDF, LEGACY_SHORT_PDF)
    print(f"PDF written: {DETAILED_PDF}")
    print(f"PDF written: {SHORT_PDF}")


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("AORegular", FONT_REGULAR))
    pdfmetrics.registerFont(TTFont("AOBold", FONT_BOLD))
    pdfmetrics.registerFontFamily("AO", normal="AORegular", bold="AOBold")


def build_document(path: Path, contents: list, title: str) -> None:
    doc = BaseDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=1.45 * cm,
        rightMargin=1.45 * cm,
        topMargin=1.45 * cm,
        bottomMargin=1.35 * cm,
        title=title,
        author="Akıl Oyunu",
        subject="AO European Elo v2 başlangıç ve canlı rating modeli",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates([PageTemplate(id="normal", frames=[frame], onPage=footer)])
    doc.build(contents)


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D5DDE7"))
    canvas.setLineWidth(0.4)
    canvas.line(doc.leftMargin, 0.98 * cm, A4[0] - doc.rightMargin, 0.98 * cm)
    canvas.setFont("AORegular", 7.1)
    canvas.setFillColor(colors.HexColor("#586575"))
    canvas.drawString(doc.leftMargin, 0.61 * cm, VERSION_LABEL)
    canvas.drawCentredString(A4[0] / 2, 0.61 * cm, DOCUMENT_DATE)
    canvas.drawRightString(A4[0] - doc.rightMargin, 0.61 * cm, f"Sayfa {doc.page}")
    canvas.restoreState()


def detailed_story() -> list:
    s = styles()
    out: list = [
        Spacer(1, 1.15 * cm),
        paragraph("AO European Elo v2", s["Title"]),
        paragraph("Tam Teknik Model Açıklaması", s["Subtitle"]),
        paragraph(
            "500-2000 referans ölçeği, başlangıç rating'i, Dynamic Power ve araştırma katmanları",
            s["CoverLine"],
        ),
        Spacer(1, 0.55 * cm),
        data_table(
            [
                ["Belge tarihi", DOCUMENT_DATE],
                ["Model sürümü", "ao-european-elo-v2.0-dev-freeze"],
                ["Geliştirme dönemi", "2018/19 - 2025/26"],
                ["Holdout", "2026/27 lig aşaması ve sonrası; qualifying kapsam dışı"],
                ["Durum", "Kontrollü gol farkı aktif; prospective izleme sürüyor"],
            ],
            s,
            [5.0 * cm, 11.4 * cm],
            header=False,
        ),
        Spacer(1, 0.45 * cm),
        callout(
            "Ana karar",
            "V2 başlangıç ölçeği ve Dynamic Power çekirdeği hazırdır. Standart H/D/A olasılık "
            "çıktısı ve kontrollü 0.10/300 gol farkı aktiftir. Season carry, progression bonus, "
            "turnuva K ve European Achievement Reserve kapalıdır. V1.1 sonuçları regresyon "
            "karşılaştırmasıdır.",
            s,
        ),
        PageBreak(),
    ]

    out += section(
        "1. Modelin Amacı ve Üç State",
        [
            "Model Avrupa kupalarındaki kulüpleri ortak bir güç uzayında sıralar. Sezon öncesi "
            "kanıt ile oynanan maç kanıtını birbirine karıştırmadan üç state tutar.",
        ],
        s,
    )
    out.append(
        data_table(
            [
                ["State", "Anlam", "Aktif davranış"],
                ["AO First Elo", "Sezon öncesi statik başlangıç gücü", "Ülke, domestic başarı, kulüp geçmişi ve exposure"],
                ["Power Elo", "Maçlarla değişen güç", "Sıfır toplamlı K güncellemesi"],
                ["Achievement Reserve", "Tur/kupa başarısı için ayrı rezerv", "Kodda mevcut, aktif Base=0"],
                ["AO Live Elo", "Maç öncesi kullanılan canlı rating", "Power Elo + Achievement Reserve"],
            ],
            s,
            [3.6 * cm, 6.0 * cm, 6.8 * cm],
        )
    )
    out += formulas(
        ["AO Live Elo = Power Elo + Achievement Reserve"], s
    )
    out += section(
        "2. Neden 500-2000?",
        [
            "V1.1'in 500-904 civarındaki puanları kullanıcı açısından birbirine yakın görünüyordu. "
            "V2, takım sırasını bozmadan farkları daha okunur hale getiren affine dönüşüm kullanır.",
        ],
        s,
    )
    out += formulas(
        [
            "M = 1500 / (903.92 - 500) = 3.713606654783126",
            "AO Elo v2 = 500 + M x (AO Elo v1.1 - 500)",
        ],
        s,
    )
    out.append(
        data_table(
            [
                ["Component", "V1.1", "V2"],
                ["Base Rating", "500", "500"],
                ["Domestic League", "140", "519.904932"],
                ["Domestic Achievement", "160", "594.177065"],
                ["European Prior Max Boost", "420", "1559.714795"],
            ],
            s,
            [7.0 * cm, 4.5 * cm, 4.9 * cm],
        )
    )
    out.append(
        callout(
            "Kritik yorum",
            "Hesap sonunda clipping uygulanmaz. Ancak aktif beta 0/0/0 davranışında AO First "
            "Elo'nun erişilebilir yapısal maksimumu 2000'dir. Dynamic/Live Elo kırpılmaz ve "
            "2000'i aşabilir. Scale, saha avantajı ve K da aynı M ile büyütüldüğü için affine "
            "dönüşüm tek başına ayırma gücünü artırmaz.",
            s,
        )
    )
    out += section(
        "3. Zaman Sözleşmesi",
        [
            "Statik season hedef sezondur. t, hedef sezon başlamadan önce tamamlanan son sezon; "
            "t_minus_4 pencerenin en eski sezonudur. Hedef sezondan veri geri sızdırılmaz.",
        ],
        s,
    )
    out += bullet_list(
        [
            "Dinamik olaylar tam UTC timestamp ile işlenir.",
            "Aynı timestamp'te match_id artan sırası deterministik tie-break'tir.",
            "Kronoloji gerilemesi ve duplicate match_id reddedilir.",
            "Harici benchmark snapshot'ı maçtan kesin olarak eski olmalıdır.",
        ],
        s,
    )
    out.append(PageBreak())

    out += section(
        "4. Statik Input Dosyaları",
        ["AO First Elo dört CSV'den hesaplanır."],
        s,
    )
    out.append(
        data_table(
            [
                ["CSV", "Anahtar", "Modeldeki rol"],
                ["teams.csv", "team_id", "Takım, ülke ve lig metadata'sı"],
                ["country_coefficients.csv", "season + country_code", "Beş sezon ülke/lig puanı"],
                ["domestic_context.csv", "season + team_id", "Lig sırası, şampiyonluk, kupa ve giriş metadata'sı"],
                ["club_european_points.csv", "season + team_id + country_code", "Kulüp puanı, played, matches ve cap"],
            ],
            s,
            [4.3 * cm, 5.0 * cm, 7.1 * cm],
        )
    )
    out += bullet_list(
        [
            "Her hedef takım için club history satırı zorunludur.",
            "Avrupa geçmişi yoksa beş sezon puan, played ve matches açıkça 0 yazılır.",
            "Official toplam/rank/coefficient ve country_part yalnızca denetim alanıdır.",
            "Competition ve entry_round statik formüle girmeyen metadata'dır.",
        ],
        s,
    )
    out += section("5. Beş Sezon Ağırlıkları", [], s)
    out.append(
        data_table(
            [
                ["Dönem", "Ağırlık"],
                ["t_minus_4", "0.07"],
                ["t_minus_3", "0.13"],
                ["t_minus_2", "0.20"],
                ["t_minus_1", "0.27"],
                ["t", "0.33"],
                ["Toplam", "1.00"],
            ],
            s,
            [8.2 * cm, 8.2 * cm],
        )
    )
    out += section(
        "6. Country Strength ve Upper Tail",
        [
            "Ülke katsayısı takımın kendi geçmişi değil, geldiği ligin Avrupa gücü sinyalidir.",
        ],
        s,
    )
    out += formulas(
        [
            "Weighted Country Score = sum(w_i x country_points_i)",
            "u_country = log(1 + Weighted Country Score) / log(1 + 25)",
            "Tail(u,beta) = u, u <= 1; 1 + beta x (u - 1), u > 1",
            "League Strength = Tail(u_country, country_tail_beta) ^ 0.80",
        ],
        s,
    )
    out.append(
        callout(
            "Aktif karar",
            "country_tail_beta=0. Benchmark üzerindeki sinyal mevcut hard-cap davranışını korur. "
            "Uncapped norm ve saturation bayrağı output'ta kaybolmaz.",
            s,
        )
    )
    out.append(PageBreak())

    out += section("7. Domestic Achievement", [], s)
    out += formulas(
        [
            "League Percentile = (team_count - position) / (team_count - 1)",
            "Percentile Score = 0.15 + 0.70 x League Percentile",
            "Champion=true ise League Finish Score = 1.00",
            "Pozisyon bilinmiyor ve champion=false ise League Finish Score = 0.10",
            "Cup Base Score = 0.62, cup_winner=true ise",
            "Cup Double Bonus = 0.08 x League Finish Score, yalnızca lig+kupa dublesinde",
            "Domestic Achievement = min(1.10, max(League Finish, Cup Base) + Double Bonus)",
        ],
        s,
    )
    out.append(
        callout(
            "Şampiyonluk step'i bilinçlidir",
            "Şampiyonluk bayrağı 1.00 override üretir. 1. ve 2. sıra arasındaki basamak bir bug "
            "değildir; şampiyonluğun orantısız değerini temsil eden domain kararıdır. Gelecekte "
            "yalnızca ayrı ranking-first backtest ile değiştirilebilir.",
            s,
        )
    )
    out += section("8. Domestic Prior", [], s)
    out += formulas(
        [
            "Achievement Scale = 0.40 + 0.60 x League Strength",
            "Domestic Prior = 500 + 519.9049316696 x League Strength",
            "+ 594.1770647653 x Domestic Achievement x Achievement Scale",
        ],
        s,
    )
    out += section("9. European Prior", [], s)
    out += formulas(
        [
            "Weighted European History = sum(w_i x club_points_i)",
            "u_europe = log(1 + Weighted European History) / log(1 + 20)",
            "European History Norm = Tail(u_europe, european_tail_beta)",
            "European Prior = 500 + 1559.7147950089 x European History Norm",
        ],
        s,
    )
    out += section("10. European Exposure", [], s)
    out += formulas(
        [
            "Season Exposure = sum(w_i x played_i)",
            "Match Exposure = sum(w_i x min(matches_i / match_cap_i, 1))",
            "European Exposure e = 0.60 x Season Exposure + 0.40 x Match Exposure",
            "Effective e = e, e <= 0.85",
            "Effective e = 0.85 + beta_exp x (e - 0.85), e > 0.85",
        ],
        s,
    )
    out.append(
        callout(
            "Aktif karar",
            "exposure_tail_beta=0. Tam ham exposure 1.0 olsa da effective exposure 0.85'tir. "
            "Domestic Prior katkısının yüzde 15'i korunur.",
            s,
        )
    )
    out.append(PageBreak())

    out += section("11. Final AO First Elo", [], s)
    out += formulas(
        [
            "AO First Elo = Domestic Prior + Effective e x (European Prior - Domestic Prior)"
        ],
        s,
    )
    out += bullet_list(
        [
            "Sıfır exposure takımında AO First Elo = Domestic Prior.",
            "Final rating Domestic Prior ile European Prior arasında kalır.",
            "Final rating 500/2000'e kırpılmaz; aktif statik yapısal üst sınır 2000'dir.",
            "Ham exposure rating_source_type kategorisini belirler.",
        ],
        s,
    )
    out.append(
        data_table(
            [
                ["Ham exposure", "Kategori"],
                ["0", "Pure Domestic Projection"],
                ["0 < e < 0.75", "Mixed Domestic-European Estimate"],
                ["e >= 0.75", "European Evidence-Based Rating"],
            ],
            s,
            [5.0 * cm, 11.4 * cm],
        )
    )
    out += section("12. Tail ve Ranking-First Backtest", [], s)
    out += bullet_list(
        [
            "Country beta: 0/0.25/0.50/0.75/1.00.",
            "European beta: 0/0.25/0.50/0.75/1.00.",
            "Exposure beta: 0/1-3/2-3/1.00.",
            "Toplam 100 kombinasyon ve 6 outer fold.",
            "Sıralama hedefi aday rating'lerinden bağımsız, leave-team-out rakip/saha düzeltmesidir.",
            "Sıralama guardrail'i standart 1X2 Brier/log-loss'tan önce gelir.",
            "Dinamik optional katmanlar aynı sezonla değil, sezon sonu rating ile takip eden sezon performansı arasında ölçülür.",
        ],
        s,
    )
    out += formulas(
        [
            "h_minus_team = shrink(mean(home_score - 0.5), prior_matches=20)",
            "opponent_strength_minus_pair = shrink(mean(venue-neutral opponent score), 0.5, prior=4)",
            "adjusted_match = target_score - venue_sign x h_minus_team + opponent_strength_minus_pair - 0.5",
            "adjusted_target = 0.5 + n/(n+2) x (mean(adjusted_match) - 0.5)",
        ],
        s,
    )
    out.append(
        data_table(
            [
                ["Sonuç", "Değer"],
                ["Karar", "NO_PROMOTION"],
                ["Aktif beta", "0 / 0 / 0"],
                ["İki ranking metriğini iyileştiren fold", "0/6"],
                ["Adjusted target takım-sezon", "2,464"],
                ["Band oranı", "100.000%"],
                ["Medyan / p90", "1075.83 / 1747.65"],
                ["Min / max", "651.49 / 1996.09"],
                ["Tam 500 veya 2000", "0"],
            ],
            s,
            [8.0 * cm, 8.4 * cm],
        )
    )
    out += section("13. Saturation Hipotezi", [], s)
    out += formulas(
        [
            "UCL saturation-count vs expected-score MSE Spearman = +0.0671",
            "UCL saturation-count vs expected-score MSE Pearson = -0.0050",
        ],
        s,
    )
    out += section(
        "",
        [
            "Pozitif ilişki beklenen tavan kümelenmesi hipotezi bu geliştirme örneğinde "
            "desteklenmemiştir. Bu, gelecekte daha geniş veride cap riskinin tekrar test "
            "edilmesini engellemez.",
        ],
        s,
    )
    out.append(PageBreak())

    out += section("14. Dynamic Power ve Standart 1X2", [], s)
    out += formulas(
        [
            "E_home = 1 / (1 + 10 ^ -((Home Live - Away Live + H) / Scale))",
            "Scale = 835.5614973262",
            "Home advantage H = 148.5442661913; nötr sahada 0",
            "K = 103.9809863339",
            "Delta = K x (S_home - E_home)",
            "Home New = Home Old + Delta; Away New = Away Old - Delta",
            "P_draw = 0.24 x (4 x E_home x (1-E_home)) ^ 1.00",
            "P_home = E_home - 0.5 x P_draw",
            "P_away = 1 - E_home - 0.5 x P_draw",
        ],
        s,
    )
    out += bullet_list(
        [
            "E_home ev sahibi galibiyet olasılığı değil, 1/0.5/0 ölçeğinde normalize beklenen puandır.",
            "Her maçta Power toplamı korunur.",
            "Normal maç K'sına UCL/UEL/UECL çarpanı uygulanmaz.",
            "Turnuva zorluğu öncelikle rakiplerin rating farkından gelir.",
            "Maçlar exact-UTC sırada işlenir.",
            "P_home + P_draw + P_away = 1 ve P_home + 0.5 x P_draw = E_home.",
        ],
        s,
    )
    out.append(
        data_table(
            [
                ["Dynamic core standart 1X2 kanıtı", "Sonuç"],
                ["Outer fold wins", "6/6"],
                ["Unseen 1X2 Brier farkı", "-0.008929"],
                ["Dependency %95 zarfı", "[-0.012823, -0.004990]"],
                ["1X2 output kararı", "PROMOTE; draw 0.24 / shape 1.00"],
                ["Full-data start-end Spearman", "0.937454"],
                ["Maksimum rating hareketi", "475.018; guardrail 742.721"],
            ],
            s,
            [8.1 * cm, 8.3 * cm],
        )
    )
    out.append(
        data_table(
            [
                ["1X2 output vs fold climatology", "AO", "Base", "Fark"],
                ["ALL Brier", "0.575337", "0.628739", "-0.053402"],
                ["UCL Brier", "0.557583", "0.627914", "-0.070331"],
                ["UEL Brier", "0.581384", "0.628673", "-0.047289"],
                ["UECL Brier", "0.583028", "0.629335", "-0.046307"],
            ],
            s,
            [7.0 * cm, 3.1 * cm, 3.1 * cm, 3.2 * cm],
        )
    )
    out += section("15. Season Carry", [], s)
    out += formulas(
        [
            "Active New Power Start = Current AO First Elo; carry = 0"
        ],
        s,
    )
    out.append(
        data_table(
            [
                ["Carry kanıtı", "Sonuç"],
                ["Adaylar", "0/0.25/0.50/0.75/0.85/0.90/1.00"],
                ["Full-data aday", "0.85"],
                ["Karar", "DISABLE; aktif carry 0"],
                ["Nested outer fold wins", "5/6"],
                ["Unseen 1X2 Brier farkı", "-0.005895"],
                ["Dependency %95 zarfı", "[-0.009464, -0.002653]"],
            ],
            s,
            [8.1 * cm, 8.3 * cm],
        )
    )
    out.append(PageBreak())

    out += section("16. Kontrollü Gol Farkı: Aktif", [], s)
    out += formulas(
        [
            "D = Home Live - Away Live + H",
            "M_GD = 1 + alpha x ln(min(GD, 4)) x exp(-abs(D) / tau)",
            "alpha = 0.10; tau = 300; goal difference cap = 4",
            "Delta = K x (S - E) x M_GD",
        ],
        s,
    )
    out.append(
        data_table(
            [
                ["Alan", "Sonuç"],
                ["Aktif production config", "alpha 0.10, tau 300, GD cap 4"],
                ["Unseen Brier fold", "6/6"],
                ["Standart 1X2 Brier farkı", "-0.000174"],
                ["Standart 1X2 log-loss farkı", "-0.000250"],
                ["Pooled Spearman / pairwise", "+0.000626 / +0.000094"],
                ["Maksimum gözlenen çarpan", "1.138347"],
                ["Karar", "Aktif; prospective izleme devam eder"],
            ],
            s,
            [8.0 * cm, 8.4 * cm],
        )
    )
    out.append(
        callout(
            "Doğru yorum",
            "1-0 ile 4-0 aynı bilgi değildir. Aktif formül yakın güçteki baskın galibiyeti sınırlı "
            "biçimde büyütür; favori farkı arttıkça bonusu söndürür ve GD'yi 4'te tavanlar. "
            "Beraberlik, tek fark ve penaltı saha beraberliğinde M_GD=1 kalır.",
            s,
        )
    )
    out += section("16.1 Turnuva K: Test Edildi, Kapalı", [], s)
    out.append(
        data_table(
            [
                ["Alan", "Final robustness sonucu"],
                ["Full-data aday", "UCL 1.50 / UEL 0.75 / UECL 0.675"],
                ["Unseen Brier fold", "1/6"],
                ["Forward ranking gerilemesiz", "1/5"],
                ["Standart 1X2 Brier farkı", "+0.000783"],
                ["Dependency %95 zarfı", "[-0.000655, +0.002253]"],
                ["Aktif config", "Tüm turnuvalar 1.00"],
            ],
            s,
            [8.0 * cm, 8.4 * cm],
        )
    )
    out += section(
        "",
        [
            "Turnuva K katsayısı takım gücü değil maçtan öğrenme hızıdır. Global K, hiyerarşiyle "
            "birlikte yeniden ölçeklenmiş; training seçimi unseen sezonlara taşınmamıştır.",
        ],
        s,
    )
    out += section("17. Achievement Reserve: Test Edildi, Kapalı", [], s)
    out += formulas(
        [
            "Advance Reserve = Base x Competition x Stage x (1 - P_advance)",
            "Trophy Reserve = Base x Competition x (1 - P_advance)",
        ],
        s,
    )
    out += bullet_list(
        [
            "P_advance ilk maç öncesi dondurulan nötr tek-maç Elo proxy'sidir; gerçek iki maçlı tur olasılığı değildir.",
            "Reserve tur kesinleştikten sonra eklenir ve sonraki maçta görünür.",
            "UCL=1.00; UEL=0.50/0.65/0.80; UECL=0.25/0.45/0.60 adayları test edildi.",
            "UCL > UEL > UECL zorunlu hiyerarşidir.",
            "Altı stage profili, dört decay ve altı Base ile 961 aday test edildi.",
            "Reserve cap 297.088532 idi.",
        ],
        s,
    )
    out.append(
        callout(
            "Aktif karar",
            "Final full-data aday Base=222.816, UEL=0.80, UECL=0.60, LATE_BALANCED ve "
            "decay=0.50 olsa da unseen sonuç 2/6 Brier ve 1/5 forward-ranking fold'dur. "
            "Brier farkı +0.001982 ve güvenilir zarar zarfı [+0.000686, +0.003321] olduğu "
            "için aktif Base ve decay sıfırdır.",
            s,
        )
    )
    out.append(PageBreak())

    out += section("18. Saha Skoru, Uzatma ve Penaltı", [], s)
    out += bullet_list(
        [
            "Skor 90 dakika veya uzatma oynandıysa 120 dakika saha skorudur.",
            "Penaltı atışı golleri home_goals/away_goals alanına eklenmez.",
            "Penaltıyla sonuçlanan saha beraberliği normal Power Elo'da S=0.5 kalır.",
            "İki maçlı eşleşmede penaltı metadata'sı ikinci ayağın saha skorunu değiştirmez.",
            "Tek maçlı final aynı field-score sözleşmesini kullanır.",
        ],
        s,
    )
    out += section("19. Canlı CSV ve Python API", [], s)
    out.append(
        data_table(
            [
                ["Public API", "Görev"],
                ["expected_score", "Canlı rating farkından maç beklentisi"],
                ["expected_1x2_probabilities", "Beklenen puanı gerçek H/D/A olasılıklarına çevirir"],
                ["initialize_season", "AO First Elo ve önceki state ile sezon başlangıcı"],
                ["update_match", "Tek maçı pure/deterministik kernel ile işler"],
                ["apply_progression", "Opsiyonel reserve event'i; aktif Base=0"],
                ["run_season", "Sıralı maç batch'i"],
                ["lock_prediction", "Sonuçsuz fixture'dan kickoff öncesi tahmin kilidi"],
                ["settle_locked_match", "Daha sonra gelen sonucu aynı kilitle doğrular"],
            ],
            s,
            [5.4 * cm, 11.0 * cm],
        )
    )
    out.append(
        data_table(
            [
                ["Çıktı", "İçerik"],
                ["ratings_state.csv", "İnsan-okunur takım rating snapshot'ı"],
                ["state_checkpoint.json", "Processed maç, açık tie, kronoloji ve ratings checksum"],
                ["match_updates.csv", "Pre/post rating, normalize beklenti, H/D/A, skor ve delta audit"],
                ["replay_predictions.csv", "Retrospective batch audit; holdout kanıtı değildir"],
                ["pre_match_log.csv", "Kickoff öncesi append-only, hash-zincirli tahmin ledger'ı"],
            ],
            s,
            [5.2 * cm, 11.2 * cm],
        )
    )
    out += bullet_list(
        [
            "Duplicate maç ve kronoloji gerilemesi reddedilir.",
            "Eksik takım, geçersiz skor ve config/state uyuşmazlığı reddedilir.",
            "Aynı tie_id farklı takım, turnuva veya stage için kullanılamaz.",
            "Batch replay ile prospective live lock/settle birbirinden ayrı komutlardır.",
            "Eşzamanlı maçlar önce kilitlenir, sonra match_id artan sırada settle edilir.",
            "Aynı input, state ve config deterministik çıktı üretir.",
            "Hash zinciri değişikliği gösterir; harici güvenilir zaman damgası değildir.",
        ],
        s,
    )
    out += section("20. Statik Validation", [], s)
    out += bullet_list(
        [
            "Takım, ülke ve takım-sezon anahtarları duplicate olamaz.",
            "Ülke/kulüp puanları eksik, negatif, NaN veya sonsuz olamaz.",
            "Boolean yalnızca true/false veya 0/1 olabilir.",
            "Pozisyon verildiyse team_count > 1 ve position tam sayı/aralık içi olmalıdır.",
            "Champion=true ve pozisyon verilmişse position=1 olmalıdır.",
            "Match cap pozitif ve sonlu olmalıdır.",
            "Final rating iki prior arasında; exposure [0,1] içinde olmalıdır.",
        ],
        s,
    )
    out.append(PageBreak())

    out += section("21. Gerçek 10 Takım Pilotu", [], s)
    out.append(
        data_table(
            [
                ["Sıra", "Takım", "AO First Elo v2"],
                ["1", "Arsenal", "1992.870"],
                ["2", "Sporting CP", "1926.711"],
                ["3", "Benfica", "1881.335"],
                ["4", "Shakhtar Donetsk", "1764.684"],
                ["5", "Galatasaray", "1756.282"],
                ["6", "AZ Alkmaar", "1741.804"],
                ["7", "Slavia Praha", "1621.856"],
                ["8", "Pafos", "1444.036"],
                ["9", "Como", "1421.436"],
                ["10", "Omonia Nicosia", "1398.415"],
            ],
            s,
            [2.5 * cm, 8.0 * cm, 5.9 * cm],
        )
    )
    out += section(
        "",
        [
            "V2 pilot sırası v1.1 ile birebir aynıdır. Como sıfır exposure kontrolüdür ve "
            "doğrudan v2 Domestic Prior'ını korur; yapay biçimde zirveye çıkmaz.",
        ],
        s,
    )
    out += section("22. Harici ClubElo ve UCL Riski", [], s)
    out.append(
        data_table(
            [
                ["UCL final paired ölçüm", "Standart 1X2 Brier"],
                ["AO Dynamic", "0.601615"],
                ["AO Static", "0.606032"],
                ["ClubElo", "0.572126"],
                ["AO - ClubElo", "+0.029489"],
                ["Dependency envelope", "[-0.002932, +0.064568]"],
            ],
            s,
            [8.2 * cm, 8.2 * cm],
        )
    )
    out.append(
        callout(
            "Açık risk",
            "AO Dynamic, AO Static'e karşı UCL'de iyileşir. ClubElo nokta tahmini daha iyidir; "
            "ancak dependency envelope sıfırı kestiği için final fark güvenilir değildir. Bu "
            "eşitlik anlamına gelmez: arşiv 171 unseen UCL maçı ve ağırlıkla yerleşik kulüpleri "
            "kapsar. UCL prospective holdout'ta ayrı izlenecektir.",
            s,
        )
    )
    out += section("23. 2026/27 Prospective Lig Aşaması Holdout'u", [], s)
    out += bullet_list(
        [
            "Model version, config fingerprint ve schema sözleşmesi maçlardan önce dondurulur.",
            "Sonuçlara bakarak Scale, H, K, carry veya statik parametre değiştirilemez.",
            "Gol farkı 0.10/300/cap 4 olarak kilitlidir; ara sonuçla değiştirilemez.",
            "Kapalı progression/reserve/turnuva K katmanları ara sonuçla açılamaz.",
            "Qualifying ve play-off maçları kapsam dışıdır; başlangıç en erken 8 Eylül 2026'dır.",
            "Sadece generated_at_utc kickoff'tan önce olan live kilitler dahil edilir.",
            "Replay predictions holdout kanıtı değildir; pre-match ledger geçmişe dönük yazılamaz.",
            "İlk ara inceleme UEFA lig aşamaları bittikten sonra yapılır.",
            "Nihai inceleme 2026/27 Avrupa sezonu tamamlandığında yapılır.",
            "Ana olasılık metrikleri standart üç sınıflı Brier ve log-loss'tur.",
        ],
        s,
    )
    out.append(
        callout(
            "Modelin bugünkü doğru tanımı",
            "Başlangıç Elo'su ve maç sonrası Power güncellemesi kodlanmış, test edilmiş ve "
            "geliştirme verisinde dondurulmuş araştırma modelidir. Production genellemesi "
            "2026/27 prospective lig-aşaması holdout'u tamamlanmadan kesinleşmiş sayılmaz.",
            s,
        )
    )
    out.append(PageBreak())
    out += section("24. Bilinen Metodolojik Borçlar", [], s)
    out += bullet_list(
        [
            "Altı outer fold sınırlı örneklemdir; fold'lar bağımsız tekrar gibi yorumlanamaz.",
            "Tie/match, team-season ve calendar-month bootstrap sonuçları duyarlılık analizidir; formal multi-way CI değildir.",
            "Draw modeli global 0.24/1.00 çiftidir; stage ayrımı ancak yeni nested kanıtla eklenebilir.",
            "Reserve yeniden test edilmeden önce P_advance gerçek tek/iki maçlı tur olasılığına çevrilmelidir.",
            "Şampiyonluk step'i bilinçli domain kararıdır ve yalnızca yeni ranking-first backtest ile değişebilir.",
        ],
        s,
    )
    return out


def short_story() -> list:
    s = styles()
    out: list = [
        Spacer(1, 1.2 * cm),
        paragraph("AO European Elo v2", s["Title"]),
        paragraph("Kısa Model Açıklaması", s["Subtitle"]),
        paragraph("Toplantı anlatımı için özet", s["CoverLine"]),
        Spacer(1, 0.45 * cm),
        callout(
            "Tek cümlede model",
            "Takımın ülke gücü, yerel başarısı ve son beş yıllık Avrupa kanıtından başlangıç "
            "rating'i üretir; sonra her maçı rakip gücü ve sonuç sürprizine göre sıfır toplamlı "
            "Power Elo güncellemesine çevirir.",
            s,
        ),
        Spacer(1, 0.35 * cm),
        data_table(
            [
                ["Sürüm", "ao-european-elo-v2.0-dev-freeze"],
                ["Geliştirme", "2018/19 - 2025/26"],
                ["Holdout", "2026/27 lig aşaması+ prospective"],
            ],
            s,
            [4.5 * cm, 11.9 * cm],
            header=False,
        ),
        PageBreak(),
    ]
    out += section("1. Başlangıç Elo'su", [], s)
    out += formulas(
        [
            "Domestic Prior = ülke/lig gücü + yerel başarı",
            "European Prior = kulübün son 5 sezon Avrupa puanı",
            "AO First Elo = Domestic Prior + Effective Exposure x (European Prior - Domestic Prior)",
        ],
        s,
    )
    out += bullet_list(
        [
            "Sezon ağırlıkları: 0.07/0.13/0.20/0.27/0.33.",
            "Country benchmark 25, European history benchmark 20.",
            "Tam Avrupa kanıtında effective exposure 0.85; Domestic Prior'ın yüzde 15'i korunur.",
            "Sıfır exposure takımında final rating Domestic Prior'a eşittir.",
            "Şampiyonluk 1.00 step'i bilinçli domain kararıdır.",
            "Kupa duble bonusu yalnızca lig ve kupa birlikte kazanıldıysa uygulanır.",
        ],
        s,
    )
    out += section("2. 500-2000 Referans Ölçeği", [], s)
    out += formulas(
        [
            "M = 3.7136066548",
            "AO Elo v2 = 500 + M x (AO Elo v1.1 - 500)",
        ],
        s,
    )
    out.append(
        callout(
            "Ne değişti?",
            "Puanlar kullanıcı için daha ayrışmış görünür. Final clipping yoktur; aktif AO First "
            "Elo'nun yapısal maksimumu 2000'dir. Dynamic/Live Elo bu değeri aşabilir. Takım "
            "sırası korunur ve yalnızca ölçek değişimi tahmin doğruluğunu artırmaz.",
            s,
        )
    )
    out.append(PageBreak())
    out += section("3. Maç Sonrası Power Elo", [], s)
    out += formulas(
        [
            "E = 1 / (1 + 10 ^ -((Home Live - Away Live + H) / Scale))",
            "Scale=835.561497, H=148.544266, K=103.980986",
            "M_GD = 1 + 0.10 x ln(min(GD,4)) x exp(-abs(D)/300)",
            "Delta = K x (S - E) x M_GD",
            "Home New = Home Old + Delta; Away New = Away Old - Delta",
        ],
        s,
    )
    out += bullet_list(
        [
            "Power güncellemesi her maçta sıfır toplamlıdır.",
            "Nötr sahada H=0'dır.",
            "Penaltı atışları hariç 90/120 dakika saha skoru kullanılır.",
            "Beraberlik ve tek farklı sonuçta M_GD=1; GD 4'te tavanlanır.",
            "Season carry kapalıdır; yeni sezon Current AO First Elo ile başlar.",
            "UCL/UEL/UECL turnuva etiketi normal maç K'sına çarpan değildir.",
            "E, galibiyet olasılığı değil 1/0.5/0 ölçeğinde beklenen normalize puandır.",
            "Model ayrıca toplamı 1 olan H/D/A olasılıklarını üretir.",
        ],
        s,
    )
    out += section("4. Hangi Katmanlar Aktif?", [], s)
    out.append(
        data_table(
            [
                ["Katman", "Karar"],
                ["V2 statik ölçek", "Aktif"],
                ["Dynamic Scale/H/K", "Aktif, 6/6 fold"],
                ["Standart 1X2 çıktı", "Aktif; draw 0.24 / shape 1.00"],
                ["Season carry", "Kapalı; 5/6 fold, aktif 0"],
                ["Tail uzatmaları", "NO_PROMOTION, beta 0"],
                ["Kontrollü gol farkı", "Aktif; alpha 0.10 / tau 300 / GD cap 4"],
                ["Progression bonus", "Kapalı; base 0"],
                ["Turnuva K", "Kapalı; 1/6 Brier, forward ranking 1/5"],
                ["Achievement Reserve", "Kapalı; 2/6 Brier, güvenilir zarar"],
            ],
            s,
            [7.5 * cm, 8.9 * cm],
        )
    )
    out.append(PageBreak())
    out += section("5. Pilot Sonucu", [], s)
    out.append(
        data_table(
            [
                ["Sıra", "Takım", "V2"],
                ["1", "Arsenal", "1992.870"],
                ["2", "Sporting CP", "1926.711"],
                ["3", "Benfica", "1881.335"],
                ["4", "Shakhtar Donetsk", "1764.684"],
                ["5", "Galatasaray", "1756.282"],
                ["6", "AZ Alkmaar", "1741.804"],
                ["7", "Slavia Praha", "1621.856"],
                ["8", "Pafos", "1444.036"],
                ["9", "Como", "1421.436"],
                ["10", "Omonia Nicosia", "1398.415"],
            ],
            s,
            [2.4 * cm, 9.0 * cm, 5.0 * cm],
        )
    )
    out += section("6. Kanıt ve Açık Risk", [], s)
    out += bullet_list(
        [
            "Dynamic core unseen standart 1X2 Brier farkı -0.008929; 6/6 ve CI iyileşme yönünde.",
            "Kontrollü gol farkı 0.10/300 adayı Brier'ı 6/6 unseen foldda iyileştirdi.",
            "Carry toplamda iyi olsa da 5/6 fold nedeniyle katı sözleşmeyle kapatıldı.",
            "UCL 1X2 Brier nokta tahmini AO 0.601615, ClubElo 0.572126; dependency zarfı sıfırı keser.",
            "AO Dynamic aynı paired UCL örneğinde AO Static'ten 0.004417 daha iyidir.",
            "Saturation korelasyonu cap hipotezini mevcut veride desteklemedi.",
            "ClubElo karşılaştırmasındaki MSE legacy diagnostiktir; ana output gerçek H/D/A'dır.",
            "2026/27 prospective holdout tamamlanmadan production genellemesi kesin değildir.",
        ],
        s,
    )
    out.append(
        callout(
            "Toplantıda doğru kapanış",
            "Modelin başlangıç Elo'su, maç güncellemesi, veri validation'ı ve batch motoru hazır. "
            "Kontrollü gol farkı 0.10/300/cap 4 olarak aktiftir; progression, turnuva K ve reserve "
            "kapalıdır. Şimdi amaç parametreleri oynamadan 2026/27 lig aşaması için kickoff öncesi "
            "hash-zincirli tahminleri toplayıp gerçek holdout performansını ölçmek.",
            s,
        )
    )
    return out


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "Title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="AOBold",
            fontSize=24,
            leading=29,
            textColor=colors.HexColor("#15243B"),
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "Subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontName="AORegular",
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#35516F"),
            alignment=TA_CENTER,
            spaceAfter=7,
        ),
        "CoverLine": ParagraphStyle(
            "CoverLine",
            parent=base["Normal"],
            fontName="AORegular",
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#657284"),
            alignment=TA_CENTER,
        ),
        "H1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName="AOBold",
            fontSize=13.2,
            leading=16.2,
            textColor=colors.HexColor("#173B63"),
            spaceBefore=9,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "Body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="AORegular",
            fontSize=9.0,
            leading=12.8,
            textColor=colors.HexColor("#202B38"),
            alignment=TA_LEFT,
            spaceAfter=6,
        ),
        "Bullet": ParagraphStyle(
            "Bullet",
            parent=base["BodyText"],
            fontName="AORegular",
            fontSize=8.8,
            leading=12.3,
            leftIndent=0.35 * cm,
            firstLineIndent=-0.22 * cm,
            textColor=colors.HexColor("#202B38"),
            spaceAfter=3,
        ),
        "Formula": ParagraphStyle(
            "Formula",
            parent=base["Code"],
            fontName="AORegular",
            fontSize=8.3,
            leading=11.4,
            leftIndent=0.25 * cm,
            rightIndent=0.25 * cm,
            borderColor=colors.HexColor("#CCD8E5"),
            borderWidth=0.45,
            borderPadding=5,
            backColor=colors.HexColor("#F5F8FB"),
            textColor=colors.HexColor("#1B365D"),
            spaceBefore=2,
            spaceAfter=5,
        ),
        "TableHeader": ParagraphStyle(
            "TableHeader",
            parent=base["BodyText"],
            fontName="AOBold",
            fontSize=7.8,
            leading=9.8,
            textColor=colors.white,
        ),
        "TableCell": ParagraphStyle(
            "TableCell",
            parent=base["BodyText"],
            fontName="AORegular",
            fontSize=7.7,
            leading=9.8,
            textColor=colors.HexColor("#202B38"),
        ),
        "CalloutTitle": ParagraphStyle(
            "CalloutTitle",
            parent=base["BodyText"],
            fontName="AOBold",
            fontSize=8.8,
            leading=11,
            textColor=colors.HexColor("#173B63"),
            spaceAfter=3,
        ),
        "CalloutBody": ParagraphStyle(
            "CalloutBody",
            parent=base["BodyText"],
            fontName="AORegular",
            fontSize=8.4,
            leading=11.8,
            textColor=colors.HexColor("#26384A"),
        ),
    }


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(text), style)


def section(title: str, bodies: list[str], s: dict[str, ParagraphStyle]) -> list:
    items: list = []
    if title:
        items.append(paragraph(title, s["H1"]))
    items.extend(paragraph(body, s["Body"]) for body in bodies)
    return items


def formulas(lines: list[str], s: dict[str, ParagraphStyle]) -> list:
    return [paragraph(line, s["Formula"]) for line in lines]


def bullet_list(items: list[str], s: dict[str, ParagraphStyle]) -> list:
    return [paragraph(f"- {item}", s["Bullet"]) for item in items]


def callout(
    title: str,
    body: str,
    s: dict[str, ParagraphStyle],
) -> KeepTogether:
    content = [
        paragraph(title, s["CalloutTitle"]),
        paragraph(body, s["CalloutBody"]),
    ]
    box = Table([[content]], colWidths=[16.4 * cm], hAlign="LEFT")
    box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EEF4F9")),
                ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#AFC4D8")),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return KeepTogether([box, Spacer(1, 0.18 * cm)])


def data_table(
    rows: list[list[str]],
    s: dict[str, ParagraphStyle],
    widths: list[float],
    *,
    header: bool = True,
) -> Table:
    converted = []
    for row_index, row in enumerate(rows):
        style = s["TableHeader"] if header and row_index == 0 else s["TableCell"]
        converted.append([paragraph(str(value), style) for value in row])
    table = Table(
        converted,
        colWidths=widths,
        repeatRows=1 if header else 0,
        hAlign="LEFT",
    )
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        commands.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#284C70")))
        if len(rows) > 1:
            commands.append(("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FC")]))
    else:
        commands.append(("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#F7F9FC")]))
    table.setStyle(TableStyle(commands))
    table.spaceBefore = 3
    table.spaceAfter = 8
    return table


if __name__ == "__main__":
    main()
