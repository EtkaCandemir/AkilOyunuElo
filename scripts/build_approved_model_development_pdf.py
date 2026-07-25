from __future__ import annotations

from html import escape
from pathlib import Path

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
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
PDF_PATH = PDF_DIR / "AkilOyunu_Elo_Onaylanan_Gelistirme_Plani.pdf"
FONT_REGULAR = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
DOCUMENT_DATE = "25 Temmuz 2026"
VERSION_LABEL = "AO European Elo v2 - Onaylanan Model Geliştirme Planı"

NAVY = colors.HexColor("#12283F")
BLUE = colors.HexColor("#246B9E")
CYAN = colors.HexColor("#42A5B3")
GREEN = colors.HexColor("#2B7A66")
AMBER = colors.HexColor("#C8872E")
RED = colors.HexColor("#B04A4A")
INK = colors.HexColor("#1E2A35")
MUTED = colors.HexColor("#5E6B78")
LINE = colors.HexColor("#D7E0E8")
PANEL = colors.HexColor("#F4F7FA")
PALE_BLUE = colors.HexColor("#EAF3F8")
PALE_GREEN = colors.HexColor("#EAF5F1")
PALE_AMBER = colors.HexColor("#FFF4E2")
WHITE = colors.white


def main() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    register_fonts()
    build_document()
    print(f"PDF written: {PDF_PATH}")


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("PlanRegular", FONT_REGULAR))
    pdfmetrics.registerFont(TTFont("PlanBold", FONT_BOLD))
    pdfmetrics.registerFontFamily(
        "Plan",
        normal="PlanRegular",
        bold="PlanBold",
    )


def build_document() -> None:
    doc = BaseDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=1.35 * cm,
        rightMargin=1.35 * cm,
        topMargin=1.35 * cm,
        bottomMargin=1.30 * cm,
        title="AO European Elo - Onaylanan Model Geliştirme Planı",
        author="Akıl Oyunu",
        subject="xG, dinamik K ve European Power Reserve geliştirme sözleşmesi",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates([PageTemplate(id="normal", frames=[frame], onPage=footer)])
    doc.build(story())


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.4)
    canvas.line(doc.leftMargin, 0.96 * cm, A4[0] - doc.rightMargin, 0.96 * cm)
    canvas.setFont("PlanRegular", 7.0)
    canvas.setFillColor(MUTED)
    canvas.drawString(doc.leftMargin, 0.59 * cm, VERSION_LABEL)
    canvas.drawCentredString(A4[0] / 2, 0.59 * cm, DOCUMENT_DATE)
    canvas.drawRightString(A4[0] - doc.rightMargin, 0.59 * cm, f"Sayfa {doc.page}")
    canvas.restoreState()


def story() -> list:
    s = styles()
    out: list = [
        Spacer(1, 0.95 * cm),
        paragraph("AO European Elo v2", s["Title"]),
        paragraph("Onaylanan Model Geliştirme Planı", s["Subtitle"]),
        paragraph(
            "xG performans sinyali, takıma göre dinamik K ve UCL/UEL/UECL "
            "European Power Reserve mimarisi",
            s["CoverLine"],
        ),
        Spacer(1, 0.45 * cm),
        data_table(
            [
                ["Belge tarihi", DOCUMENT_DATE],
                ["Belge türü", "Sunum ve geliştirme kararı"],
                ["Korunan çekirdek", "AO First Elo + Dynamic Power Elo + kontrollü gol farkı"],
                ["Yeni aday katmanlar", "xG harmanlama, dinamik K, European Power Reserve"],
                ["Aktivasyon ilkesi", "Nested walk-forward backtest ve guardrail onayı"],
                ["Bugünkü production", "Yeni üç katman henüz aktif değildir"],
            ],
            s,
            [5.1 * cm, 11.2 * cm],
            header=False,
        ),
        Spacer(1, 0.35 * cm),
        callout(
            "Ana karar",
            "Model zenginleştirilecek; ancak karmaşıklık tek başına hedef değildir. Her yeni katman "
            "mevcut production çekirdeğine karşı bağımsız ve birleşik olarak out-of-sample test "
            "edilecek. Geçemeyen katman kapalı veya shadow mode'da kalacaktır.",
            s,
            PALE_BLUE,
            BLUE,
        ),
        PageBreak(),
    ]

    out += section(
        "1. Yönetici Özeti",
        [
            "Onaylanan geliştirme yönü üç yeni bilgi katmanını kapsar. İlk katman maç skorunun "
            "yanına xG ile ölçülen oyun üretimini ekler. İkinci katman Elo'nun öğrenme hızını, "
            "takım hakkındaki kanıt miktarına göre değiştirir. Üçüncü katman Avrupa'daki tur ve "
            "kupa başarısını Power Elo'dan ayrı, denetlenebilir ek puan rezervi olarak tutar.",
        ],
        s,
    )
    out.append(architecture_diagram())
    out.append(Spacer(1, 0.15 * cm))
    out.append(
        data_table(
            [
                ["Katman", "Sorduğu soru", "Elo içindeki görevi"],
                ["AO First Elo", "Takım sezona hangi güçle giriyor?", "Yapısal başlangıç rating'i"],
                ["Dynamic Power", "Maçlar kalıcı gücü nasıl değiştiriyor?", "Sıfır toplamlı canlı güç"],
                ["xG performansı", "Skor oyun kalitesini doğru yansıttı mı?", "Maç güncellemesini düzeltir"],
                ["Dinamik K", "Bu takım hakkında ne kadar eminiz?", "Öğrenme hızını ayarlar"],
                ["European Power Reserve", "Avrupa başarısının ek değeri nedir?", "Geçici ve ayrı ek puan"],
                ["AO Live Elo", "Bir sonraki maçta hangi rating kullanılır?", "Power + Reserve"],
            ],
            s,
            [4.1 * cm, 6.1 * cm, 6.1 * cm],
        )
    )
    out += bullets(
        [
            "AO First Elo mimarisi ve 500-2000 referans ölçeği değişmeyecektir.",
            "Mevcut kontrollü gol farkı katmanı korunacak ve xG ile birlikte yeniden test edilecektir.",
            "Kırmızı kart düzeltmesi bu planın kapsamına alınmamıştır.",
            "Kadro, sakatlık, seyahat ve beklenen ilk 11 bu paketin dışında tutulmuştur.",
            "Yeni parametrelerin hiçbiri backtest tamamlanmadan production değeri sayılmayacaktır.",
        ],
        s,
    )
    out.append(PageBreak())

    out += section(
        "2. Katman 1 - Gol ve xG Farkının Modele Yedirilmesi",
        [
            "Skor maçın sonucunu, xG ise üretilen şutların gol olma kalitesini özetler. Aynı skor "
            "farklı performanslarla oluşabilir. Bu nedenle xG, sonucu silen bir alternatif değil; "
            "skor kanıtını kontrollü biçimde düzelten ikinci bir performans sinyali olacaktır.",
        ],
        s,
    )
    out.append(
        data_table(
            [
                ["Örnek", "Skor", "xG", "Model yorumu"],
                ["Takım A", "4-0", "1.3-0.7", "Baskın skor; üretim farkı daha sınırlı"],
                ["Takım B", "1-0", "3.2-0.4", "Dar skor; üretim farkı çok güçlü"],
                ["Amaç", "-", "-", "İki maçı yalnızca skora bakarak aynı biçimde okumamak"],
            ],
            s,
            [3.0 * cm, 2.4 * cm, 3.0 * cm, 7.9 * cm],
        )
    )
    out += subheading("Önerilen aday matematik", s)
    out += formulas(
        [
            "xGD = xG_home - xG_away",
            "S_xG = 1 / (1 + exp(-xGD / c_xG))",
            "Delta_result = (S_home - E_home) x M_GD",
            "Delta_xG = S_xG - E_home",
            "Delta = K_effective x [(1-rho) x Delta_result + rho x Delta_xG]",
        ],
        s,
    )
    out.append(
        callout(
            "Neden bu formül?",
            "rho=0 olduğunda sistem bugünkü kontrollü gol farkı modeline tam olarak döner. rho "
            "arttıkça xG performansı küçük bir ağırlık kazanır. Böylece şanslı büyük skor "
            "yumuşatılabilir, skorun göstermediği güçlü üretim ise kontrollü biçimde tanınabilir.",
            s,
            PALE_GREEN,
            GREEN,
        )
    )
    out += subheading("Test edilecek parametre ailesi", s)
    out.append(
        data_table(
            [
                ["Parametre", "Aday grid", "Görevi"],
                ["rho", "0 / 0.05 / 0.10 / 0.15 / 0.20 / 0.25", "xG sinyalinin toplam güncellemedeki ağırlığı"],
                ["c_xG", "0.75 / 1.00 / 1.25 / 1.50", "xG farkının olasılık sinyaline dönüşüm eğimi"],
                ["xG türü", "xG / mümkünse npxG", "Penaltı kaynaklı çifte sayımı kontrol eder"],
                ["Eksik xG", "İmpute edilmez; matched sample", "Eksik maçın sıfır xG sanılmasını önler"],
            ],
            s,
            [3.0 * cm, 5.8 * cm, 7.5 * cm],
        )
    )
    out += bullets(
        [
            "xG yalnızca maç tamamlandıktan sonra Power güncellemesinde kullanılacaktır.",
            "Penaltı atışları skora ve xG sinyaline dahil edilmeyecektir.",
            "Uzatma oynandıysa 120 dakikalık saha skoru ile aynı zaman kapsamındaki xG kullanılacaktır.",
            "Tüm tarihsel sezonlarda aynı xG sağlayıcısı ve aynı metrik tanımı korunacaktır.",
            "xG bulunmayan maçlar için xG=0 yazılmayacak; klasik model kolunda işlenecektir.",
        ],
        s,
    )
    out.append(PageBreak())

    out += section(
        "3. xG Veri Sözleşmesi ve Kalite Kontrolü",
        [
            "xG katmanının doğruluğu formülden önce veri tutarlılığına bağlıdır. Maç kimliği, "
            "zaman kapsamı ve sağlayıcı tanımı açık olmayan xG kayıtları modele alınmayacaktır.",
        ],
        s,
    )
    out.append(
        data_table(
            [
                ["Alan", "Zorunluluk", "Kontrol"],
                ["match_id", "Zorunlu", "Mevcut maç tablosuyla bire bir eşleşme"],
                ["xg_home / xg_away", "Zorunlu", "Sonlu, negatif olmayan sayılar"],
                ["xg_type", "Zorunlu", "xG veya npxG tanımı"],
                ["duration_scope", "Zorunlu", "90 veya oynandıysa 120 dakika"],
                ["provider", "Zorunlu", "Sezonlar boyunca tek sağlayıcı"],
                ["snapshot_time", "Zorunlu", "Maç bitiminden sonra oluşmuş performans kaydı"],
                ["penalty_shootout_excluded", "Zorunlu", "Boolean ve true olmalı"],
            ],
            s,
            [4.8 * cm, 3.0 * cm, 8.5 * cm],
        )
    )
    out.append(
        callout(
            "Veri riski",
            "Ücretsiz veri setleri araştırma için yararlı olabilir; ancak tam UCL, UEL ve UECL "
            "sezon kapsamı garanti edilmez. Production kararı öncesinde turnuva-sezon bazında xG "
            "coverage matrisi hazırlanmalı ve backtest yalnızca karşılaştırılabilir matched "
            "maçlarda raporlanmalıdır.",
            s,
            PALE_AMBER,
            AMBER,
        )
    )
    out += subheading("Zorunlu ablation karşılaştırması", s)
    out.append(
        data_table(
            [
                ["Kol", "Gol farkı", "xG", "Amaç"],
                ["BASE", "Kapalı", "Kapalı", "Klasik Elo referansı"],
                ["GD", "Aktif", "Kapalı", "Bugünkü production davranışı"],
                ["xG", "Kapalı", "Aktif", "xG'nin bağımsız katkısı"],
                ["GD + xG", "Aktif", "Aktif", "Birleşik performans modeli"],
            ],
            s,
            [3.2 * cm, 3.0 * cm, 3.0 * cm, 7.1 * cm],
        )
    )
    out += bullets(
        [
            "Birleşik model, yalnızca gol farkı modelini geçmeden aktive edilmeyecektir.",
            "İyileşme toplam örnekte değil, UCL/UEL/UECL segmentlerinde de kontrol edilecektir.",
            "xG ağırlığı büyüdükçe maksimum tek maç hareketi ve rating oynaklığı izlenecektir.",
        ],
        s,
    )
    out.append(PageBreak())

    out += section(
        "4. Katman 2 - Takıma Göre Dinamik K",
        [
            "K katsayısı Elo'nun öğrenme hızıdır. Sabit K bütün takımların rating'ine aynı "
            "güvenle yaklaşır. Dinamik K, az kanıtı veya uzun kesintisi bulunan takımın yeni "
            "sonuçlardan daha hızlı; yerleşik takımın ise daha kontrollü öğrenmesini amaçlar.",
        ],
        s,
    )
    out.append(dynamic_k_diagram())
    out += formulas(
        [
            "U_team = uncertainty(exposure, prior_matches, inactivity)",
            "K_team = K_base x (1 + lambda x U_team)",
            "K_match = aggregate(K_home, K_away)",
            "Delta = K_match x [(1-rho) x Delta_result + rho x Delta_xG]",
            "Home_new = Home_old + Delta; Away_new = Away_old - Delta",
        ],
        s,
    )
    out.append(
        data_table(
            [
                ["Belirsizlik sinyali", "Yüksek belirsizlik", "Düşük belirsizlik"],
                ["European exposure", "Düşük veya sıfıra yakın", "Yüksek"],
                ["Geçmiş Avrupa maçı", "Az", "Çok"],
                ["Son Avrupa maçından süre", "Uzun", "Kısa"],
                ["Sonuç", "Daha yüksek K_team", "K_base'e yakın K_team"],
            ],
            s,
            [5.1 * cm, 5.6 * cm, 5.6 * cm],
        )
    )
    out.append(
        callout(
            "Sıfır toplam koruması",
            "Takımların K_team değerleri ayrı hesaplanır; fakat maç için tek K_match üretilir. "
            "Aynı Delta ev sahibine eklenip deplasmandan çıkarılır. Böylece dinamik öğrenme "
            "sağlanırken toplam Power Elo kendiliğinden büyümez.",
            s,
            PALE_GREEN,
            GREEN,
        )
    )
    out += subheading("Test edilecek adaylar", s)
    out.append(
        data_table(
            [
                ["Parametre", "Adaylar", "Not"],
                ["lambda", "0 / 0.15 / 0.30 / 0.50", "0 mevcut sabit K kontrolüdür"],
                ["K üst çarpanı", "1.25 / 1.50 / 1.75", "Tek maç hareketini sınırlar"],
                ["K aggregation", "ortalama / geometrik ortalama", "İki takımın K_team değerlerini birleştirir"],
                ["inactivity eşiği", "180 / 270 / 365 gün", "Yalnızca geçmiş bilgi kullanılır"],
                ["match evidence ölçeği", "6 / 10 / 15 maç", "Belirsizliğin sönme hızıdır"],
            ],
            s,
            [4.0 * cm, 5.5 * cm, 6.8 * cm],
        )
    )
    out += bullets(
        [
            "K_base, bugünkü production değeri olan 103.980986 ile başlar; grid bunu referans alır.",
            "Belirsizlik değişkenleri yalnızca maç başlangıcından önceki kayıtlardan hesaplanır.",
            "K_match ve gerçek Delta output'ta ayrı kolonlar olarak raporlanır.",
            "Maksimum hareket, sıralama kararlılığı ve toplam Elo korunumu temel guardrail olacaktır.",
        ],
        s,
    )
    out.append(PageBreak())

    out += section(
        "5. Katman 3 - UCL/UEL/UECL European Power Reserve",
        [
            "Turnuva etiketi normal maç K'sına otomatik prestij çarpanı olarak uygulanmayacaktır. "
            "UCL, UEL ve UECL arasındaki başarı değeri; Dynamic Power'dan ayrı tutulan European "
            "Power Reserve üzerinden temsil edilecektir.",
        ],
        s,
    )
    out += formulas(
        [
            "Reserve Add = Base x Competition Multiplier x Stage Multiplier x (1-P_advance)",
            "European Power Reserve_new = min(Reserve Cap, Reserve_old + Reserve Add)",
            "AO Live Elo = Dynamic Power Elo + European Power Reserve",
        ],
        s,
    )
    out.append(reserve_diagram())
    out.append(
        data_table(
            [
                ["Turnuva", "Çarpan adayları", "Zorunlu ilişki"],
                ["UCL", "1.00", "En yüksek referans"],
                ["UEL", "0.50 / 0.65 / 0.80", "UCL'den düşük"],
                ["UECL", "0.25 / 0.45 / 0.60", "UEL'den düşük"],
            ],
            s,
            [4.2 * cm, 6.0 * cm, 6.1 * cm],
        )
    )
    out.append(
        data_table(
            [
                ["Reserve parametresi", "Aday grid"],
                ["Base", "0 / 74.27 / 111.41 / 148.54 / 185.68 / 222.82"],
                ["Stage profile", "flat / gentle / knockout / late / semifinal-heavy"],
                ["Season decay", "0.25 / 0.50 / 0.75 / 1.00"],
                ["Reserve cap", "297.09 referans; duyarlılık testi zorunlu"],
                ["Trophy reserve", "0 veya seçilen Base ile aynı ölçek"],
            ],
            s,
            [5.2 * cm, 11.1 * cm],
        )
    )
    out.append(
        callout(
            "Sürpriz düzeltmesi",
            "(1-P_advance) terimi favorinin beklenen tur geçişinden sınırlı, sürpriz tur geçişinden "
            "daha yüksek reserve almasını sağlar. Böylece yalnızca turnuva adı veya tur sayısı "
            "üzerinden otomatik puan şişmesi azaltılır.",
            s,
            PALE_BLUE,
            BLUE,
        )
    )
    out += bullets(
        [
            "Reserve yalnızca eşleşme veya final kesinleştikten sonra bir kez eklenir.",
            "Çift ayaklı eşleşmelerde iki maç Power Elo'yu normal biçimde günceller; reserve tur sonunda eklenir.",
            "Penaltıyla tur geçişinde saha maçı Elo sonucu beraberlik kalabilir; tur reserve'i ayrıca uygulanabilir.",
            "Lig aşaması sırası, ilk 8 ve ön eleme için bu sürümde otomatik bonus tanımlanmayacaktır.",
            "Reserve yeni sezonda seçilen decay katsayısıyla azalır; Power Elo'nun içine kalıcı olarak gömülmez.",
        ],
        s,
    )
    out.append(PageBreak())

    out += section(
        "6. Reserve Neden Ayrı Tutuluyor?",
        [
            "Maç sonucu zaten Power Elo'yu değiştirir. Aynı galibiyeti bir kez daha doğrudan Power "
            "bonusuna çevirmek çifte sayım riski taşır. Ayrı reserve, başarı puanını görünür, "
            "sınırlı ve zaman içinde azaltılabilir hale getirir.",
        ],
        s,
    )
    out.append(
        data_table(
            [
                ["Adım", "Power Elo", "Reserve", "AO Live Elo", "Yorum"],
                ["Başlangıç", "1600", "0", "1600", "Maç gücü"],
                ["UCL tur geçişi", "1600", "+24", "1624", "Başarı ek puanı"],
                ["Sonraki maç -30", "1570", "+24", "1594", "Maç Delta'sı Power'a gider"],
                ["Yeni sezon %50 decay", "1570", "+12", "1582", "Başarı etkisi zamanla azalır"],
            ],
            s,
            [3.5 * cm, 3.0 * cm, 3.0 * cm, 3.2 * cm, 3.6 * cm],
        )
    )
    out += subheading("İki kullanım kolu karşılaştırılacak", s)
    out.append(
        data_table(
            [
                ["Kol", "AO Live'a etkisi", "Tahmine etkisi", "Amaç"],
                ["Display-only", "Ayrı gösterilir", "Yok", "Prestij/sıralama açıklaması"],
                ["Predictive reserve", "Power + Reserve", "Sonraki maçtan itibaren var", "Başarının geçici güç sinyali olup olmadığını test etmek"],
            ],
            s,
            [3.4 * cm, 3.8 * cm, 3.6 * cm, 5.5 * cm],
        )
    )
    out.append(
        callout(
            "Production şartı",
            "Onaylanan hedef predictive reserve mimarisidir; ancak geçmiş deneylerde eski reserve "
            "ve progression tasarımları terfi kapılarını geçmemiştir. Yeni sürpriz duyarlı tasarım "
            "mevcut Power + GD tabanına karşı yeniden ve bağımsız olarak kanıtlanmadan production'a "
            "alınmayacaktır.",
            s,
            PALE_AMBER,
            AMBER,
        )
    )
    out += bullets(
        [
            "Power toplamı sıfır toplamlı kalır; reserve toplamı ayrı raporlanır.",
            "Her reserve olayında competition, stage, P_advance, eklenen puan ve kalan cap loglanır.",
            "Reserve'in sıralamayı yapay biçimde kilitleyip kilitlemediği sezonlar arası test edilir.",
            "UCL > UEL > UECL hiyerarşisi yalnızca reserve değerinde zorunlu kısıttır.",
        ],
        s,
    )
    out.append(PageBreak())

    out += section(
        "7. Backtest Tasarımı",
        [
            "Parametre seçimi exact-date rolling / nested walk-forward ile yapılacaktır. Her sezonun "
            "başlangıç Elo'su yalnızca o tarihten önce tamamlanmış verilerle oluşturulacak; xG, tur "
            "sonucu veya sonraki sezon bilgisi geriye sızdırılmayacaktır.",
        ],
        s,
    )
    out.append(backtest_diagram())
    out += subheading("Aşamalı ablation sırası", s)
    out.append(
        data_table(
            [
                ["Aşama", "Model", "Karşılaştırma"],
                ["A", "Mevcut production: Power + GD", "Dondurulmuş referans"],
                ["B", "A + xG", "xG'nin bağımsız katkısı"],
                ["C", "Seçilen B + dinamik K", "Belirsizliğe göre öğrenme katkısı"],
                ["D", "Seçilen C + Power Reserve", "Turnuva başarısının incremental katkısı"],
                ["E", "Tam birleşik model", "Etkileşim ve stabilite kontrolü"],
            ],
            s,
            [2.0 * cm, 7.0 * cm, 7.3 * cm],
        )
    )
    out += subheading("Raporlanacak metrikler", s)
    out.append(
        data_table(
            [
                ["Boyut", "Metrikler"],
                ["Tahmin", "Multiclass Brier, log loss, 1X2 accuracy, calibration"],
                ["Sıralama", "Spearman, pairwise ranking, sezonlar arası rank stabilitesi"],
                ["Segment", "UCL / UEL / UECL, favori / dengeli / underdog"],
                ["Hareket", "Maksimum Delta, volatilite, sezon sonu min/medyan/max"],
                ["Invariant", "Power sıfır toplamı, reserve cap/decay, chronology, replay"],
                ["Veri", "xG coverage, matched sample, eksik ve sağlayıcı tutarlılığı"],
            ],
            s,
            [4.0 * cm, 12.3 * cm],
        )
    )
    out.append(
        callout(
            "Terfi ilkesi",
            "Brier veya log-loss iyileşmesi tek başına yeterli değildir. Sıralama performansı, "
            "turnuva segmentleri, kalibrasyon ve rating stabilitesi birlikte geçmelidir. Pratik "
            "olarak anlamsız farkta daha sade model tercih edilir.",
            s,
            PALE_GREEN,
            GREEN,
        )
    )
    out += bullets(
        [
            "Her aday unseen foldlarda baseline'a karşı raporlanacaktır.",
            "En az 4/6 fold yönsel iyileşme hedeflenir; güvenilir segment zararı kabul edilmez.",
            "UCL, UEL ve UECL pooled sonuçlarının hiçbiri güvenilir biçimde kötüleşmemelidir.",
            "2026/27 prospective kayıtları geliştirme parametresi seçmek için kullanılmayacaktır.",
        ],
        s,
    )
    out.append(PageBreak())

    out += section(
        "8. Veri İhtiyacı ve Uygulama Sırası",
        [
            "Dinamik K mevcut tarihsel maç ve exposure verilerinden üretilebilir. xG için tutarlı "
            "bir dış sağlayıcı gerekir. Reserve için competition, stage, tie ve tur kesinleşme "
            "metadata'sı zorunludur.",
        ],
        s,
    )
    out.append(
        data_table(
            [
                ["Paket", "Yeni veri", "Hazırlık"],
                ["xG", "match_id, xG/npxG, kapsam, sağlayıcı", "Turnuva-sezon coverage denetimi"],
                ["Dinamik K", "Geçmiş maç sayısı, exposure, inactivity", "Mevcut veriden türetilebilir"],
                ["Power Reserve", "competition, stage, tie_id, P_advance, event_time", "Tur olay tablosu hazırlanmalı"],
            ],
            s,
            [3.8 * cm, 6.2 * cm, 6.3 * cm],
        )
    )
    out += subheading("Önerilen çalışma takvimi", s)
    out.append(
        data_table(
            [
                ["Sıra", "Çalışma", "Teslim"],
                ["1", "xG sağlayıcı ve coverage incelemesi", "Veri kalite raporu"],
                ["2", "xG matched walk-forward backtest", "rho ve c_xG kararı"],
                ["3", "Dinamik K shadow motoru", "lambda, cap ve aggregation kararı"],
                ["4", "UCL/UEL/UECL reserve event dataset", "Tur ve kupa olay sözleşmesi"],
                ["5", "Reserve nested backtest", "Base, multiplier, stage, decay kararı"],
                ["6", "Tam model robustness", "Production / shadow / reject kararı"],
                ["7", "Dokümantasyon ve contract freeze", "Tekrarlanabilir production paketi"],
            ],
            s,
            [1.4 * cm, 8.1 * cm, 6.8 * cm],
        )
    )
    out += section(
        "9. Nihai Karar Çerçevesi",
        [
            "Bu belge üç geliştirme yönünü onaylar; katsayıları onaylamaz. Nihai model, her "
            "katmanın veriyle kanıtlanan sürümünden oluşacaktır.",
        ],
        s,
    )
    out.append(
        data_table(
            [
                ["Katman", "Bugünkü durum", "Hedef durum"],
                ["Kontrollü gol farkı", "Production aktif: 0.10 / 300 / cap 4", "xG ile birlikte yeniden doğrulama"],
                ["xG performansı", "Henüz yok", "Backtest sonrası active veya shadow"],
                ["Dinamik K", "Sabit K=103.980986", "Takım belirsizliğine göre kontrollü K"],
                ["European Power Reserve", "Kapalı, Base=0", "UCL > UEL > UECL ek puan sistemi"],
                ["Kırmızı kart", "Yok", "Bu planın dışında"],
                ["Geçici kadro/seyahat", "Yok", "Daha sonraki veri paketi"],
            ],
            s,
            [4.3 * cm, 5.8 * cm, 6.2 * cm],
        )
    )
    out.append(
        callout(
            "Sunum cümlesi",
            "AO European Elo'nun yeni geliştirme yönü; skoru xG performansıyla açıklayan, "
            "belirsiz takımlardan daha hızlı öğrenen ve UCL/UEL/UECL başarısını kalıcı Power'dan "
            "ayrı bir European Power Reserve olarak yöneten modüler bir canlı rating sistemidir.",
            s,
            PALE_BLUE,
            BLUE,
        )
    )
    out += bullets(
        [
            "Başlangıç Elo modeli korunur.",
            "Her yeni feature geçmişe doğru veri sızıntısı olmadan test edilir.",
            "Her bileşen bağımsız açılıp kapatılabilir ve etkisi loglanır.",
            "Production'a yalnızca sıralama, tahmin ve stabilite kapılarını birlikte geçen katman alınır.",
        ],
        s,
    )
    return out


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "Title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="PlanBold",
            fontSize=27,
            leading=31,
            textColor=NAVY,
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "Subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Heading2"],
            fontName="PlanBold",
            fontSize=17,
            leading=21,
            textColor=BLUE,
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
        "CoverLine": ParagraphStyle(
            "CoverLine",
            parent=base["BodyText"],
            fontName="PlanRegular",
            fontSize=10.5,
            leading=15,
            textColor=MUTED,
            alignment=TA_CENTER,
            leftIndent=1.0 * cm,
            rightIndent=1.0 * cm,
            spaceAfter=12,
        ),
        "H1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName="PlanBold",
            fontSize=15.2,
            leading=19,
            textColor=NAVY,
            spaceBefore=5,
            spaceAfter=7,
        ),
        "H2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName="PlanBold",
            fontSize=11.2,
            leading=14,
            textColor=BLUE,
            spaceBefore=6,
            spaceAfter=5,
        ),
        "Body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="PlanRegular",
            fontSize=8.8,
            leading=12.3,
            textColor=INK,
            spaceAfter=6,
        ),
        "Bullet": ParagraphStyle(
            "Bullet",
            parent=base["BodyText"],
            fontName="PlanRegular",
            fontSize=8.5,
            leading=11.7,
            textColor=INK,
            leftIndent=0.40 * cm,
            firstLineIndent=-0.22 * cm,
            spaceAfter=3.2,
        ),
        "CalloutTitle": ParagraphStyle(
            "CalloutTitle",
            parent=base["BodyText"],
            fontName="PlanBold",
            fontSize=9.2,
            leading=12,
            textColor=NAVY,
            spaceAfter=3,
        ),
        "CalloutBody": ParagraphStyle(
            "CalloutBody",
            parent=base["BodyText"],
            fontName="PlanRegular",
            fontSize=8.4,
            leading=11.5,
            textColor=INK,
        ),
        "TableHead": ParagraphStyle(
            "TableHead",
            parent=base["BodyText"],
            fontName="PlanBold",
            fontSize=7.8,
            leading=9.8,
            textColor=WHITE,
            alignment=TA_LEFT,
        ),
        "TableCell": ParagraphStyle(
            "TableCell",
            parent=base["BodyText"],
            fontName="PlanRegular",
            fontSize=7.55,
            leading=9.5,
            textColor=INK,
            alignment=TA_LEFT,
        ),
        "Formula": ParagraphStyle(
            "Formula",
            parent=base["Code"],
            fontName="PlanRegular",
            fontSize=8.25,
            leading=11.5,
            textColor=NAVY,
            leftIndent=0.22 * cm,
            rightIndent=0.22 * cm,
            spaceAfter=2,
        ),
    }


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(text), style)


def rich_paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def section(title: str, bodies: list[str], s: dict[str, ParagraphStyle]) -> list:
    out = [paragraph(title, s["H1"])]
    out.extend(paragraph(body, s["Body"]) for body in bodies)
    return out


def subheading(title: str, s: dict[str, ParagraphStyle]) -> list:
    return [paragraph(title, s["H2"])]


def bullets(items: list[str], s: dict[str, ParagraphStyle]) -> list:
    return [
        rich_paragraph(f"<font color='#246B9E'>•</font> {escape(item)}", s["Bullet"])
        for item in items
    ]


def formulas(items: list[str], s: dict[str, ParagraphStyle]) -> list:
    rows = [[paragraph(item, s["Formula"])] for item in items]
    table = Table(rows, colWidths=[16.0 * cm], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PANEL),
                ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                ("LINEBELOW", (0, 0), (-1, -2), 0.3, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return [table, Spacer(1, 0.18 * cm)]


def callout(
    title: str,
    body: str,
    s: dict[str, ParagraphStyle],
    background: colors.Color,
    accent: colors.Color,
) -> Table:
    content = [
        rich_paragraph(f"<b>{escape(title)}</b>", s["CalloutTitle"]),
        paragraph(body, s["CalloutBody"]),
    ]
    table = Table([[content]], colWidths=[16.0 * cm], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 0.5, accent),
                ("LINEBEFORE", (0, 0), (0, -1), 3.0, accent),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return table


def data_table(
    rows: list[list[str]],
    s: dict[str, ParagraphStyle],
    widths: list[float],
    *,
    header: bool = True,
) -> Table:
    converted: list[list[Paragraph]] = []
    for row_index, row in enumerate(rows):
        style = s["TableHead"] if header and row_index == 0 else s["TableCell"]
        converted.append([paragraph(str(value), style) for value in row])
    table = Table(converted, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PANEL]),
            ]
        )
    else:
        commands.append(("ROWBACKGROUNDS", (0, 0), (-1, -1), [WHITE, PANEL]))
    table.setStyle(TableStyle(commands))
    return table


def architecture_diagram() -> Drawing:
    drawing = Drawing(462, 145)
    boxes = [
        (6, 88, 94, 42, "AO First Elo", "Sezon başlangıcı", BLUE),
        (124, 88, 94, 42, "Dynamic Power", "Maç kanıtı", GREEN),
        (242, 88, 94, 42, "Power Reserve", "Tur ve kupa", AMBER),
        (360, 88, 94, 42, "AO Live Elo", "Power + Reserve", NAVY),
    ]
    for x, y, width, height, title, subtitle, color in boxes:
        drawing.add(Rect(x, y, width, height, rx=3, ry=3, fillColor=color, strokeColor=color))
        drawing.add(
            String(
                x + width / 2,
                y + 25,
                title,
                fontName="PlanBold",
                fontSize=8.2,
                textAnchor="middle",
                fillColor=WHITE,
            )
        )
        drawing.add(
            String(
                x + width / 2,
                y + 11,
                subtitle,
                fontName="PlanRegular",
                fontSize=6.8,
                textAnchor="middle",
                fillColor=WHITE,
            )
        )
    for start, end in ((100, 124), (218, 242), (336, 360)):
        drawing.add(Line(start, 109, end - 5, 109, strokeColor=MUTED, strokeWidth=1.2))
        drawing.add(Polygon([end - 5, 105, end, 109, end - 5, 113], fillColor=MUTED, strokeColor=MUTED))
    drawing.add(Rect(109, 14, 244, 44, rx=3, ry=3, fillColor=PALE_BLUE, strokeColor=CYAN))
    drawing.add(
        String(
            231,
            40,
            "Maç güncellemesini yöneten yeni feature'lar",
            fontName="PlanBold",
            fontSize=8.3,
            textAnchor="middle",
            fillColor=NAVY,
        )
    )
    drawing.add(
        String(
            231,
            25,
            "xG performans sinyali + takıma göre dinamik K",
            fontName="PlanRegular",
            fontSize=7.4,
            textAnchor="middle",
            fillColor=INK,
        )
    )
    drawing.add(Line(231, 58, 171, 88, strokeColor=CYAN, strokeWidth=1.2))
    return drawing


def dynamic_k_diagram() -> Drawing:
    drawing = Drawing(462, 118)
    items = [
        (8, "Düşük exposure", PALE_AMBER, AMBER),
        (121, "Az geçmiş maç", PALE_AMBER, AMBER),
        (234, "Uzun inactivity", PALE_AMBER, AMBER),
    ]
    for x, label, fill, stroke in items:
        drawing.add(Rect(x, 72, 100, 31, rx=3, ry=3, fillColor=fill, strokeColor=stroke))
        drawing.add(
            String(
                x + 50,
                84,
                label,
                fontName="PlanBold",
                fontSize=7.3,
                textAnchor="middle",
                fillColor=INK,
            )
        )
        drawing.add(Line(x + 50, 72, 231, 49, strokeColor=MUTED, strokeWidth=0.8))
    drawing.add(Rect(171, 18, 120, 32, rx=3, ry=3, fillColor=GREEN, strokeColor=GREEN))
    drawing.add(
        String(
            231,
            36,
            "Takım belirsizliği U",
            fontName="PlanBold",
            fontSize=8.0,
            textAnchor="middle",
            fillColor=WHITE,
        )
    )
    drawing.add(
        String(
            231,
            25,
            "yüksek U -> kontrollü yüksek K",
            fontName="PlanRegular",
            fontSize=6.8,
            textAnchor="middle",
            fillColor=WHITE,
        )
    )
    drawing.add(Line(291, 34, 356, 34, strokeColor=MUTED, strokeWidth=1.0))
    drawing.add(Polygon([351, 30, 356, 34, 351, 38], fillColor=MUTED, strokeColor=MUTED))
    drawing.add(Rect(356, 18, 98, 32, rx=3, ry=3, fillColor=NAVY, strokeColor=NAVY))
    drawing.add(
        String(
            405,
            36,
            "Tek K_match",
            fontName="PlanBold",
            fontSize=8.0,
            textAnchor="middle",
            fillColor=WHITE,
        )
    )
    drawing.add(
        String(
            405,
            25,
            "+Delta / -Delta",
            fontName="PlanRegular",
            fontSize=6.8,
            textAnchor="middle",
            fillColor=WHITE,
        )
    )
    return drawing


def reserve_diagram() -> Drawing:
    drawing = Drawing(462, 122)
    competitions = [
        (12, 73, "UCL", "1.00", BLUE),
        (126, 73, "UEL", "0.50-0.80", CYAN),
        (240, 73, "UECL", "0.25-0.60", GREEN),
    ]
    for x, y, title, value, color in competitions:
        drawing.add(Rect(x, y, 100, 34, rx=3, ry=3, fillColor=color, strokeColor=color))
        drawing.add(
            String(
                x + 50,
                y + 20,
                title,
                fontName="PlanBold",
                fontSize=8.4,
                textAnchor="middle",
                fillColor=WHITE,
            )
        )
        drawing.add(
            String(
                x + 50,
                y + 8,
                value,
                fontName="PlanRegular",
                fontSize=6.9,
                textAnchor="middle",
                fillColor=WHITE,
            )
        )
        drawing.add(Line(x + 50, y, 231, 44, strokeColor=MUTED, strokeWidth=0.8))
    drawing.add(Rect(157, 13, 148, 34, rx=3, ry=3, fillColor=AMBER, strokeColor=AMBER))
    drawing.add(
        String(
            231,
            32,
            "European Power Reserve",
            fontName="PlanBold",
            fontSize=8.5,
            textAnchor="middle",
            fillColor=WHITE,
        )
    )
    drawing.add(
        String(
            231,
            20,
            "stage x sürpriz x competition",
            fontName="PlanRegular",
            fontSize=6.8,
            textAnchor="middle",
            fillColor=WHITE,
        )
    )
    drawing.add(Line(305, 30, 358, 30, strokeColor=MUTED, strokeWidth=1.0))
    drawing.add(Polygon([353, 26, 358, 30, 353, 34], fillColor=MUTED, strokeColor=MUTED))
    drawing.add(Rect(358, 13, 96, 34, rx=3, ry=3, fillColor=NAVY, strokeColor=NAVY))
    drawing.add(
        String(
            406,
            32,
            "AO Live Elo",
            fontName="PlanBold",
            fontSize=8.3,
            textAnchor="middle",
            fillColor=WHITE,
        )
    )
    drawing.add(
        String(
            406,
            20,
            "Power + Reserve",
            fontName="PlanRegular",
            fontSize=6.8,
            textAnchor="middle",
            fillColor=WHITE,
        )
    )
    return drawing


def backtest_diagram() -> Drawing:
    drawing = Drawing(462, 112)
    folds = [
        (8, "Train", "Test 1", BLUE),
        (98, "Train + 1", "Test 2", CYAN),
        (188, "Train + 2", "Test 3", GREEN),
        (278, "Train + 3", "Test 4", AMBER),
        (368, "Train + 4", "Test 5/6", NAVY),
    ]
    for x, train, test, color in folds:
        drawing.add(Rect(x, 53, 82, 40, rx=2, ry=2, fillColor=WHITE, strokeColor=color))
        drawing.add(Rect(x, 73, 82, 20, fillColor=color, strokeColor=color))
        drawing.add(
            String(
                x + 41,
                80,
                train,
                fontName="PlanBold",
                fontSize=6.8,
                textAnchor="middle",
                fillColor=WHITE,
            )
        )
        drawing.add(
            String(
                x + 41,
                60,
                test,
                fontName="PlanRegular",
                fontSize=6.8,
                textAnchor="middle",
                fillColor=INK,
            )
        )
    drawing.add(
        String(
            231,
            29,
            "Her test sezonu yalnızca geçmiş sezonlardan seçilen parametrelerle değerlendirilir",
            fontName="PlanBold",
            fontSize=7.6,
            textAnchor="middle",
            fillColor=NAVY,
        )
    )
    drawing.add(
        String(
            231,
            14,
            "Exact-date chronology + no leakage + tournament segment guardrail",
            fontName="PlanRegular",
            fontSize=7.0,
            textAnchor="middle",
            fillColor=MUTED,
        )
    )
    return drawing


if __name__ == "__main__":
    main()
