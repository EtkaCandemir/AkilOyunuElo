from __future__ import annotations

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
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
DETAILED_PDF = ROOT / "AkilOyunu_Elo_Model_Aciklayici.pdf"
SHORT_PDF = ROOT / "AkilOyunu_Elo_Model_Kisa.pdf"
FONT_REGULAR = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
DOCUMENT_DATE = "15 Temmuz 2026"
VERSION_LABEL = "AO European Elo - Araştırma v1"


def main() -> None:
    pdfmetrics.registerFont(TTFont("AORegular", FONT_REGULAR))
    pdfmetrics.registerFont(TTFont("AOBold", FONT_BOLD))
    pdfmetrics.registerFontFamily("AO", normal="AORegular", bold="AOBold")
    build_document(
        DETAILED_PDF,
        detailed_story(),
        "AO European Elo - Tam Teknik Model Açıklaması",
    )
    build_document(
        SHORT_PDF,
        short_story(),
        "AO European Elo - Kısa Model Açıklaması",
    )
    print(f"PDF written: {DETAILED_PDF}")
    print(f"PDF written: {SHORT_PDF}")


def build_document(path: Path, contents: list, title: str) -> None:
    doc = BaseDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=1.55 * cm,
        rightMargin=1.55 * cm,
        topMargin=1.55 * cm,
        bottomMargin=1.45 * cm,
        title=title,
        author="Akıl Oyunu",
        subject="AO European Elo başlangıç ve dinamik maç rating modeli",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates([PageTemplate(id="normal", frames=[frame], onPage=footer)])
    doc.build(contents)


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D7DEE8"))
    canvas.setLineWidth(0.4)
    canvas.line(doc.leftMargin, 1.03 * cm, A4[0] - doc.rightMargin, 1.03 * cm)
    canvas.setFont("AORegular", 7.2)
    canvas.setFillColor(colors.HexColor("#5F6B7A"))
    canvas.drawString(doc.leftMargin, 0.67 * cm, VERSION_LABEL)
    canvas.drawCentredString(A4[0] / 2, 0.67 * cm, DOCUMENT_DATE)
    canvas.drawRightString(A4[0] - doc.rightMargin, 0.67 * cm, f"Sayfa {doc.page}")
    canvas.restoreState()


def detailed_story() -> list:
    s = styles()
    out: list = [
        Spacer(1, 1.5 * cm),
        Paragraph("AO European Elo", s["Title"]),
        Paragraph("Tam Teknik Model Açıklaması", s["Subtitle"]),
        Paragraph("Başlangıç rating'i + maç sonrası dinamik Elo", s["CoverLine"]),
        Spacer(1, 0.65 * cm),
        status_box(
            [
                ["Belge sürümü", "Araştırma v1 - 15 Temmuz 2026"],
                ["Statik başlangıç modeli", "AO First Elo v1.1 - donduruldu"],
                ["Dinamik çekirdek", "Scale 225, saha avantajı 40, K 28"],
                ["Sezonlar arası carry", "0.85 - geçici kabul edilen araştırma katmanı"],
                ["Production durumu", "Kullanılabilir araştırma modeli; untouched holdout bekleniyor"],
            ],
            s,
        ),
        Spacer(1, 0.65 * cm),
        note(
            "Bu belge neyi anlatıyor?",
            "Modelin veri sözleşmesini, bütün aktif formüllerini, parametrelerini, "
            "validation kurallarını, maç sonrası güncellemesini, test sonuçlarını, "
            "reddedilen aday katmanları ve henüz kanıtlanmamış sınırlarını tek yerde "
            "toplar. Aktif model ile yalnızca araştırılmış adaylar özellikle ayrılır.",
            s,
        ),
        PageBreak(),
        Paragraph("İçindekiler ve Okuma Haritası", s["H1"]),
        table(
            [
                ["Bölüm", "İçerik"],
                ["1-3", "Model amacı, iki katmanlı mimari ve veri zamanı"],
                ["4-11", "AO First Elo başlangıç rating'inin tüm formülleri"],
                ["12-16", "Dinamik maç olasılığı, K güncellemesi ve season carry"],
                ["17-18", "Aktif olmayan bonuslar, turnuva ve final davranışı"],
                ["19-23", "Backtest, harici ClubElo benchmarkı ve UCL teşhisi"],
                ["24-26", "Validation, output sözleşmesi ve production takvimi"],
            ],
            s,
            [3.0 * cm, 13.5 * cm],
        ),
        note(
            "En önemli ayrım",
            "AO First Elo sezon başındaki başlangıç gücüdür. Dynamic Power Elo ise "
            "oynanan her maçtan sonra değişen canlı güçtür. Avrupa kupası, tur ve kupa "
            "prestiji hakkında konuşulan bütün adaylar canlı rating'e otomatik olarak "
            "eklenmemiştir; yalnızca backtest eşiğini geçen katmanlar aktiftir.",
            s,
        ),
        Paragraph("1. Amaç ve Kapsam", s["H1"]),
        body(
            "AO European Elo, UCL, UEL ve UECL'de yer alan kulüpleri ortak bir Avrupa "
            "ölçeğinde sıralamak ve maçlardan sonra bu gücü güncellemek için tasarlanmıştır. "
            "Model iki soruyu ayrı cevaplar: Takım sezona hangi güçle başlar ve oynadığı "
            "maçtan sonra gücü ne kadar değişir?",
            s,
        ),
        table(
            [
                ["Katman", "Ürettiği değer", "Temel görevi"],
                ["AO First Elo", "ao_first_elo", "Sezon başlamadan önce açıklanabilir başlangıç rating'i"],
                ["Dynamic Power Elo", "ao_live_rating", "Her maçtan sonra sıfır toplamlı canlı rating"],
                ["Achievement Reserve", "Ayrı rezerv", "Test edildi; trophy katkısı reddedildi ve aktif değil"],
            ],
            s,
            [4.2 * cm, 4.1 * cm, 8.2 * cm],
        ),
        Paragraph("2. Aynı Avrupa Ölçeği Ne Demektir?", s["H1"]),
        body(
            "UCL, UEL ve UECL için üç ayrı Elo tablosu yoktur. Her kulüp aynı rating "
            "uzayındadır. Bir UCL maçının genellikle daha zor görünmesi, rakiplerin daha "
            "yüksek rating'li olmasından gelir. Turnuva adı tek başına normal maç K'sını "
            "artırmaz veya azaltmaz. Böylece rakip gücü bir kez Elo farkında hesaba girer; "
            "turnuva etiketi üzerinden ikinci kez sayılmaz.",
            s,
        ),
        note(
            "UCL=1.00, UEL=0.65, UECL=0.45 konusu",
            "Bu sıra domain mantığı ve araştırma referansı olarak tutuldu. Normal maç "
            "öğrenme hızına doğrudan uygulandığında yeterince tutarlı sonuç vermediği için "
            "aktif K çarpanı değildir. Liglerin sportif olarak eşit olduğu anlamına gelmez.",
            s,
        ),
        Paragraph("3. Zaman Sözleşmesi ve Veri Sızıntısı", s["H1"]),
        *bullets(
            [
                "season, rating'in üretildiği hedef sezondur.",
                "t, hedef sezondan önce tamamlanan son sezondur; t_minus_4 en eski sezondur.",
                "AO First Elo yalnızca hedef sezondan önce bilinen verilerle hesaplanır.",
                "Dinamik maçlar resmi UEFA UTC başlama zamanına göre kronolojik işlenir.",
                "Harici Elo karşılaştırmasında yalnızca maç tarihinden kesin olarak eski snapshot kullanılır.",
                "Aynı gün veya gelecek tarihli rating hiçbir satıra bağlanmaz.",
            ],
            s,
        ),
        PageBreak(),
        Paragraph("4. Statik Modelin Girdi Dosyaları", s["H1"]),
        table(
            [
                ["Dosya", "Zorunlu içerik", "Anahtar"],
                ["teams.csv", "Takım adı, ülke kodu, yerel lig", "team_id"],
                ["country_coefficients.csv", "Beş sezon UEFA ülke puanı", "season + country_code"],
                ["domestic_context.csv", "Lig sırası, takım sayısı, lig/kupa şampiyonluğu, giriş metadata'sı", "season + team_id"],
                ["club_european_points.csv", "Beş sezon kulüp puanı, played, matches ve match_cap", "season + team_id + country_code"],
            ],
            s,
            [4.2 * cm, 8.3 * cm, 4.0 * cm],
        ),
        body(
            "Avrupa geçmişi olmayan takım için kulüp satırı atılmaz. Beş sezonun "
            "club_points, played ve matches değerleri açıkça sıfır yazılır; match_cap "
            "değerleri yine pozitif olmalıdır. competition, entry_round ve "
            "european_entry_type açıklayıcı metadata'dır.",
            s,
        ),
        Paragraph("5. Dondurulan Statik Parametreler", s["H1"]),
        table(
            [
                ["Parametre", "Aktif değer", "Görevi"],
                ["Base Rating", "500", "Bütün prior'ların başlangıç tabanı"],
                ["Country Strength Benchmark", "25", "Ülke puanı log normalizasyon referansı"],
                ["Gamma", "0.80", "Normalize lig gücünün eğrisi"],
                ["Domestic League Component", "140", "Lig gücünün maksimum katkı ölçeği"],
                ["Domestic Achievement Component", "160", "Yerel başarının katkı ölçeği"],
                ["Achievement Alpha", "0.40", "Zayıf ligde başarı katkısının korunmuş tabanı"],
                ["European History Benchmark", "20", "Kulüp Avrupa geçmişi log referansı"],
                ["European Prior Max Boost", "420", "Avrupa prior'unun maksimum artışı"],
                ["Max European Exposure", "0.85", "Final karışımında Avrupa etkisi tavanı"],
                ["Evidence Threshold", "0.75", "Rating source type sınıf eşiği"],
            ],
            s,
            [5.6 * cm, 3.0 * cm, 7.9 * cm],
        ),
        Paragraph("6. Beş Sezon Ağırlıkları", s["H1"]),
        table(
            [
                ["Sezon", "Anlam", "Ağırlık"],
                ["t_minus_4", "Hedef sezondan beş sezon önce tamamlanan dönem", "0.07"],
                ["t_minus_3", "Dört sezon önce", "0.13"],
                ["t_minus_2", "Üç sezon önce", "0.20"],
                ["t_minus_1", "İki sezon önce", "0.27"],
                ["t", "Hedef sezondan önce tamamlanan son sezon", "0.33"],
                ["Toplam", "Ağırlık sözleşmesi", "1.00"],
            ],
            s,
            [4.0 * cm, 9.0 * cm, 3.5 * cm],
        ),
        body(
            "Aynı ağırlıklar ülke puanında, kulüp Avrupa geçmişinde, played exposure'da "
            "ve maç exposure'ında kullanılır. Böylece yakın geçmiş bütün ana Avrupa "
            "sinyallerinde tutarlı biçimde daha önemlidir.",
            s,
        ),
        Paragraph("7. Ülke Gücü", s["H1"]),
        formula("Weighted Country Score = Σ (season_weight_i x country_points_i)", s),
        formula(
            "League Strength Norm = min[1, log(1 + Weighted Country Score) / log(1 + 25)]",
            s,
        ),
        formula("League Strength = League Strength Norm ^ 0.80", s),
        body(
            "Buradaki normalizasyon doğrusal değildir. Log dönüşümü ülke puanındaki "
            "çok büyük farkları yumuşatırken güçlü ve zayıf ligleri ayırır. Daha önce "
            "araştırılan Benchmark=20, Gamma=2.0, League Component=360 adayı aggregate "
            "sonucu iyileştirse de Avrupa exposure'ı sıfır olan takımlarda gerçekçilik ve "
            "ranking guardrail'lerini bozduğu için reddedildi.",
            s,
        ),
        PageBreak(),
        Paragraph("8. Yerel Başarı Skoru", s["H1"]),
        formula(
            "League Percentile = (league_team_count - domestic_position) / "
            "(league_team_count - 1)",
            s,
        ),
        formula("Percentile Score = 0.15 + 0.70 x (League Percentile ^ 1.00)", s),
        formula(
            "League Finish Score = max(Percentile Score, Champion Base if champion)",
            s,
        ),
        formula("Cup Double Bonus = 0.08 x League Finish Score", s),
        formula(
            "Domestic Achievement = min[1.10, max(League Finish Score, Cup Base) + Double Bonus]",
            s,
        ),
        table(
            [
                ["Kural", "Değer / davranış"],
                ["Percentile floor", "0.15"],
                ["Percentile scale", "0.70"],
                ["Percentile delta", "1.00"],
                ["Champion base", "1.00"],
                ["Bilinmeyen lig bitirişi", "0.10; fakat champion=true ise 1.00"],
                ["Cup base", "0.62"],
                ["Double bonus", "Yalnızca lig ve kupa birlikte kazanıldıysa 0.08 x League Finish"],
                ["Achievement cap", "1.10"],
            ],
            s,
            [6.2 * cm, 10.3 * cm],
        ),
        note(
            "Kupa dublesi ile Avrupa kupası bonusunu karıştırmayın",
            "Cup Double Bonus yerel lig + yerel kupa başarısının AO First Elo katkısıdır. "
            "UCL/UEL/UECL şampiyonluğu için önerilen trophy reserve ayrı bir dinamik "
            "katmandı ve backtestte kabul edilmedi.",
            s,
        ),
        Paragraph("9. Domestic Prior", s["H1"]),
        formula(
            "Achievement Scale = 0.40 + (1 - 0.40) x League Strength",
            s,
        ),
        formula(
            "Domestic Prior = 500 + 140 x League Strength + "
            "160 x Domestic Achievement x Achievement Scale",
            s,
        ),
        body(
            "Achievement Scale, yerel başarının güçlü liglerde daha fazla puan üretmesini "
            "sağlar; zayıf ligde bile katkının yüzde 40'ı korunur. Avrupa geçmişi ve "
            "exposure'ı sıfır olan takımın final AO First Elo'su doğrudan Domestic Prior'dur.",
            s,
        ),
        Paragraph("10. Kulübün Avrupa Geçmişi ve European Prior", s["H1"]),
        formula("Weighted European History = Σ (season_weight_i x club_points_i)", s),
        formula(
            "European History Norm = min[1, log(1 + Weighted European History) / log(1 + 20)]",
            s,
        ),
        formula("European Prior = 500 + 420 x European History Norm", s),
        body(
            "Kulüp puanı, ülke puanından ayrıdır: ülke puanı ligin genel gücünü, kulüp "
            "puanı takımın kendi Avrupa kanıtını temsil eder. official_club_coefficient, "
            "official_five_year_total, official_country_rank ve country_part alanları "
            "opsiyonel denetim alanlarıdır; aktif formüle girmez.",
            s,
        ),
        PageBreak(),
        Paragraph("11. European Exposure ve Final AO First Elo", s["H1"]),
        formula("Weighted Season Exposure = Σ (weight_i x played_i)", s),
        formula(
            "Weighted Match Exposure = Σ [weight_i x min(matches_i / match_cap_i, 1)]",
            s,
        ),
        formula(
            "European Exposure = 0.60 x Season Exposure + 0.40 x Match Exposure",
            s,
        ),
        formula("Effective European Exposure = min(European Exposure, 0.85)", s),
        formula(
            "AO First Elo = Domestic Prior + Effective Exposure x (European Prior - Domestic Prior)",
            s,
        ),
        body(
            "played sezon varlığını, matches kanıt miktarını, match_cap ise o sezon için "
            "yeterli maç örneklemi eşiğini anlatır. match_cap bir performans puanı değildir "
            "ve her sezon kolonunda pozitif yazılır. Effective tavan, tam Avrupa geçmişinde "
            "bile Domestic Prior'un yüzde 15'ini korur.",
            s,
        ),
        table(
            [
                ["Ham exposure", "Kaynak türü", "Final davranışı"],
                ["0", "Pure Domestic Projection", "AO First Elo = Domestic Prior"],
                ["0 ile 0.75 arası", "Mixed Domestic-European Estimate", "İki prior karıştırılır"],
                ["0.75 ve üzeri", "European Evidence-Based Rating", "Effective değer en fazla 0.85"],
            ],
            s,
            [3.6 * cm, 6.1 * cm, 6.8 * cm],
        ),
        note(
            "Exposure smoke testleri",
            "Beş sezonun her birinde 2 maç; cap değerleri 6,6,6,8,8 olan sentetik "
            "takımda weighted_match_exposure yaklaşık 0.283 ve total European Exposure "
            "yaklaşık 0.713'tür. Yeterli maç oynanan beş sezonda ham exposure 1.00, "
            "effective exposure 0.85 olur.",
            s,
        ),
        PageBreak(),
        Paragraph("12. Statik Output ve Sıralama", s["H1"]),
        table(
            [
                ["Output", "Anlam"],
                ["weighted_country_score", "Ağırlıklı beş sezon ülke puanı"],
                ["league_strength_norm / league_strength", "Log normalize ve gamma dönüşümlü lig gücü"],
                ["domestic_achievement_score", "Lig ve yerel kupa başarısı"],
                ["domestic_prior", "Ülke gücü + yerel başarı başlangıç prior'u"],
                ["weighted_european_history", "Ağırlıklı kulüp Avrupa puanı"],
                ["european_prior", "Kulübün Avrupa geçmişi prior'u"],
                ["european_exposure", "Ham kanıt miktarı"],
                ["effective_european_exposure", "Final rating'de kullanılan tavanlı exposure"],
                ["ao_first_elo", "Kalıcı sezon başlangıç rating'i"],
                ["ao_first_elo_rank", "Azalan rating; eşitlikte team_id artan deterministik sıra"],
                ["rating_source_type", "Ham exposure'a göre açıklama kategorisi"],
            ],
            s,
            [6.1 * cm, 10.4 * cm],
        ),
        Paragraph("13. Gerçek Pilot Referansı", s["H1"]),
        table(
            [
                ["Sıra", "Takım", "AO First Elo"],
                ["1", "Arsenal", "902.000"],
                ["2", "Sporting CP", "884.185"],
                ["3", "Benfica", "871.966"],
                ["4", "Shakhtar Donetsk", "840.554"],
                ["5", "Galatasaray", "838.292"],
                ["6", "AZ Alkmaar", "834.393"],
                ["7", "Slavia Praha", "802.093"],
                ["8", "Pafos", "754.210"],
                ["9", "Como", "748.124"],
                ["10", "Omonia Nicosia", "741.925"],
            ],
            s,
            [2.4 * cm, 8.6 * cm, 5.5 * cm],
        ),
        body(
            "Bu tablo smoke test referansıdır; gerçek Avrupa sıralaması iddiası değildir. "
            "Como'nun daha eski bir revizyonda exposure sıfır olduğu için 978.316 ile "
            "zirveye çıkması ranking hatası olarak yakalandı; ranking-first guardrail "
            "çalışmaları sonrasında aktif v1.1 pilotunda 748.124'e düzeltildi.",
            s,
        ),
        Paragraph("14. Sezonlar Arası Power Carry", s["H1"]),
        formula(
            "Power Start_new = 0.15 x AO First Elo_current + 0.85 x Power End_previous",
            s,
        ),
        formula("Achievement Reserve Start = 0", s),
        body(
            "team_id her sezon yerel ve değişebilir. Carry için kalıcı kulüp anahtarı "
            "country_code + normalize(team_name) olarak oluşturulur. 1.887 takım-sezon "
            "satırı 506 kalıcı kulübe bağlandı; ülke çakışması bulunmadı. Önceki sezonda "
            "yer almayan kulüp yalnızca güncel AO First Elo ile başlar.",
            s,
        ),
        note(
            "0.85 neden aktif araştırma adayı?",
            "Carry-only model 6/6 görülmemiş fold kazandı. Eşleşmiş Brier farkı "
            "-0.002612 ve yüzde 95 güven aralığı [-0.004017, -0.001286] oldu. Bu nedenle "
            "0.85 geçici kabul edildi; production config'e nihai olarak yazılması untouched "
            "holdout sonrasına bırakıldı.",
            s,
        ),
        Paragraph("15. Maç Öncesi Beklenti", s["H1"]),
        formula(
            "Expected Home Score = 1 / [1 + 10 ^ (-(Home Rating - Away Rating + H) / 225)]",
            s,
        ),
        table(
            [
                ["Parametre", "Aktif değer", "Açıklama"],
                ["Elo Scale", "225", "Rating farkının olasılığa ne kadar sert dönüştüğünü belirler"],
                ["Home Advantage H", "40", "Nötr olmayan maçta ev takımına beklenti hesabında eklenir"],
                ["Neutral H", "0", "Nötr sahada ev avantajı kaldırılır"],
                ["Actual score S", "1 / 0.5 / 0", "Ev galibiyeti / beraberlik / deplasman galibiyeti"],
            ],
            s,
            [5.0 * cm, 3.0 * cm, 8.5 * cm],
        ),
        note(
            "Daha önce konuşulan 300 / 50 neydi?",
            "300 bir Elo scale adayı, 50 ise saha avantajı puanı adayıydı. Backtest "
            "sonucunda aktif değerler 225 ve 40 olarak seçildi. Scale bir takıma verilen "
            "bonus değildir; rating farkını olasılığa çeviren paydadır.",
            s,
        ),
        Paragraph("16. Maç Sonrası Sıfır Toplamlı Güncelleme", s["H1"]),
        formula("Delta = 28 x (Actual Home Score - Expected Home Score)", s),
        formula("Home Rating_new = Home Rating_old + Delta", s),
        formula("Away Rating_new = Away Rating_old - Delta", s),
        body(
            "Bir maçta kazanılan puan diğer takımın kaybettiği puana eşittir; toplam güç "
            "yaratılmaz. Favori beklenen sonucu aldığında az, sürpriz olduğunda daha fazla "
            "puan hareket eder. Beraberlik 0.5 sonuç değeriyle aynı matematiğe girer.",
            s,
        ),
        table(
            [
                ["Örnek", "Hesap"],
                ["Ev 850, deplasman 800, H=40", "Rating farkı 90; Expected Home yaklaşık 0.715"],
                ["Ev kazanır", "Delta = 28 x (1 - 0.715) = +7.98"],
                ["Berabere", "Delta = 28 x (0.5 - 0.715) = -6.02"],
                ["Ev kaybeder", "Delta = 28 x (0 - 0.715) = -20.02"],
            ],
            s,
            [6.0 * cm, 10.5 * cm],
        ),
        PageBreak(),
        Paragraph("17. Maç Türlerinin Aktif Davranışı", s["H1"]),
        table(
            [
                ["Durum", "Aktif model davranışı"],
                ["Normal lig/aşama maçı", "Scale 225, H 40/0 ve K 28 ile tek güncelleme"],
                ["İki ayaklı eşleşme", "Her ayak kendi skoruyla normal maç gibi güncellenir"],
                ["Tek maç eleme", "Normal maç güncellemesi uygulanır"],
                ["Penaltı atışları", "Maç rating sonucu gösterilen skorla belirlenir; shootout ayrı galibiyet sayılmaz"],
                ["Tur atlama", "Ek puan yok; progression katmanı aktif değil"],
                ["Final", "Final maçı normal K=28 güncellemesini alır; ek trophy/final bonusu yok"],
                ["UCL / UEL / UECL", "Normal maç K'sı aynıdır; rakip rating farkı zorluğu taşır"],
                ["1-0 ve 4-0", "Aktif modelde sonuç S aynıdır; gol farkı çarpanı aktif değildir"],
            ],
            s,
            [5.3 * cm, 11.2 * cm],
        ),
        note(
            "Final bonusunun sıfır olması ne demek?",
            "Finalin değersiz olduğu anlamına gelmez. Finaldeki maç sonucu normal Elo'yu "
            "değiştirir. Ek final/trophy rezervi ise finalden sonra aynı sezonda başka maç "
            "olmaması ve sonraki sezon testinde ek fayda göstermemesi nedeniyle sıfırdır.",
            s,
        ),
        Paragraph("18. Araştırılan Fakat Aktif Olmayan Katmanlar", s["H1"]),
        table(
            [
                ["Katman", "Test sonucu", "Aktif karar"],
                ["Gol farkı", "weight 0.5, cap 2; 5/6 fold, Brier -0.000342; CI [-0.000811, +0.000141]", "Aday; güvenilir değil, kapalı"],
                ["Esnek turnuva K", "2/6 fold; küçük aggregate iyileşme", "Aday; stabil değil, kapalı"],
                ["Galibiyet prestij bonusu", "Nested seçim 0/6 fold'da pozitif bonus", "Reddedildi"],
                ["Tur atlama prestiji", "Pozitif challenger 0/6; Brier +0.000268", "Reddedildi"],
                ["Aşama duyarlı progression", "Pozitif challenger 3/6; CI sıfırı kesiyor", "Aday; kapalı"],
                ["Katı 1.00/0.65/0.45 maç K", "2/6 fold; Brier +0.000211", "Normal K için reddedildi"],
                ["Trophy reserve", "Carry-only modele karşı 1/6; fark +0.000070", "Reddedildi"],
                ["UCL forecast scale", "200-400; selected 2/5; fark -0.001917 ama CI sıfırı kesiyor", "225 korundu"],
            ],
            s,
            [4.0 * cm, 8.0 * cm, 4.5 * cm],
        ),
        body(
            "Domain referansı 1.00/0.65/0.45 tamamen silinmedi; achievement veya prestij "
            "araştırmalarında hiyerarşi referansı olarak tutulabilir. Ancak aktif modelde "
            "K_eff = 28 x bu katsayılar değildir.",
            s,
        ),
        Paragraph("19. Backtest Tasarımı", s["H1"]),
        *bullets(
            [
                "2018/19-2025/26 arasında sekiz sezon ve 6.340 UEFA maçı kullanıldı.",
                "Resmi UEFA servisinden 6.340/6.340 kesin UTC maç tarihi eşlendi.",
                "Expanding walk-forward yapıda ilk iki sezon eğitim, sonraki sezon görülmemiş test oldu.",
                "Altı outer fold: 2020/21, 2021/22, 2022/23, 2023/24, 2024/25, 2025/26.",
                "Birincil metrik Brier; ikincil metrik log loss ve takım sıralama guardrail'leridir.",
                "Bir katman yalnız ortalama iyi diye alınmaz; fold galibiyeti, güven aralığı ve stabilite aranır.",
            ],
            s,
        ),
        Paragraph("20. Dinamik Çekirdek Kalibrasyonu", s["H1"]),
        table(
            [
                ["Sonuç", "Değer"],
                ["Seçilen çekirdek", "Scale=225, H=40, K=28"],
                ["Dynamic vs tuned static fold", "6/6 dynamic kazandı"],
                ["Genel paired Brier farkı", "-0.003955"],
                ["Yüzde 95 güven aralığı", "[-0.005520, -0.002333]"],
                ["UCL farkı", "-0.004564"],
                ["UEL farkı", "-0.005297"],
                ["UECL farkı", "-0.002624"],
                ["Karar", "PROVISIONAL_ACCEPT_CORE"],
            ],
            s,
            [7.0 * cm, 9.5 * cm],
        ),
        Paragraph("21. Season Carry ve Trophy Sonucu", s["H1"]),
        table(
            [
                ["Test", "Sonuç"],
                ["Carry-only", "6/6 fold; Brier -0.002612; CI [-0.004017, -0.001286]"],
                ["Tam veri carry adayı", "0.85"],
                ["Carry fold aralığı", "0.85-1.00; 0.85 mod değer"],
                ["Trophy vs matched carry", "1/6 fold; Brier +0.000070; CI [-0.000015, +0.000155]"],
                ["Trophy örnek adayı", "UCL 20 / UEL 13 / UECL 9, decay 0.25"],
                ["Karar", "Power carry geçici kabul; trophy reserve reddedildi"],
            ],
            s,
            [7.0 * cm, 9.5 * cm],
        ),
        PageBreak(),
        Paragraph("22. Harici ClubElo Benchmarkı", s["H1"]),
        table(
            [
                ["Sözleşme", "Değer"],
                ["Kesin UEFA tarihi", "6.340 / 6.340 maç"],
                ["ClubElo snapshot kuralı", "Maçtan kesin eski son 1/15 tarihli snapshot"],
                ["Maksimum izin verilen snapshot yaşı", "31 gün; kullanılan çiftlerde gözlenen maksimum 17 gün"],
                ["Geçerli paired maç", "492"],
                ["Walk-forward görülmemiş paired maç", "363"],
                ["AO Brier", "0.163249"],
                ["ClubElo Brier", "0.158745"],
                ["AO - ClubElo", "+0.004504; CI [-0.005398, +0.014634]"],
            ],
            s,
            [7.4 * cm, 9.1 * cm],
        ),
        body(
            "Genel fark ClubElo yönünde olsa da güven aralığı sıfırı kestiği için genel "
            "üstünlük kesin değildir. Snapshot arşivi daha çok güçlü ve yerleşik kulüpleri "
            "kapsar; tüm eleme takımlarını temsil etmez. 2025/26 rating'leri arşiv tazeliği "
            "yetmediği için harici paired teste alınmadı.",
            s,
        ),
        Paragraph("23. UCL Farkı ve Takım Sıralaması Önceliği", s["H1"]),
        table(
            [
                ["UCL teşhisi", "Brier / sonuç"],
                ["Static start-only", "0.175531"],
                ["Dynamic season reset", "0.170063"],
                ["Dynamic carry 0.50", "0.169083"],
                ["Current carry 0.85", "0.171128"],
                ["ClubElo", "0.156801"],
                ["Current vs ClubElo", "+0.014327; ClubElo güvenilir biçimde daha iyi"],
                ["AO live vs ClubElo rank Spearman", "0.888094"],
                ["Current minimum start/end rank correlation", "0.907974"],
                ["Current maksimum rating hareketi", "136.346657; 200 guardrail'inin altında"],
            ],
            s,
            [8.0 * cm, 8.5 * cm],
        ),
        body(
            "UCL farkı tek bir AO katmanına bağlanamadı. Normal dynamic update static "
            "başlangıcı iyileştirdi. Carry 0.85'in reset modele göre +0.001066 farkı güvenilir "
            "değildi. UCL forecast scale adayları 2/5 fold kazandı; 225 değişmedi. Bu nedenle "
            "takım sıralamasını bozacak bir revizyon yapılmadı. Büyük kulüp-sezon sapmaları "
            "audit dosyasında görünür tutulur; tekil örnekler formül değişikliği sayılmaz.",
            s,
        ),
        Paragraph("24. Validation ve İnvariantlar", s["H1"]),
        *bullets(
            [
                "team_id, ülke ve sezon anahtarlarında duplicate kayıt reddedilir.",
                "season + team_id + country_code kulüp geçmişi eşleşmesi zorunludur.",
                "Eksik, negatif, NaN veya sonsuz ülke/kulüp puanı reddedilir.",
                "Boolean alanlar yalnız tanımlı true/false ve 0/1 biçimlerini kabul eder.",
                "Pozisyon biliniyorsa league_team_count > 1 ve position 1..team_count tam sayıdır.",
                "champion=true iken verilen pozisyon 1 değilse validation hatasıdır.",
                "match_cap pozitif ve sonlu olmalıdır; matches negatif olamaz.",
                "Sezon ağırlıkları negatif olamaz ve toplamı 1.00 olmalıdır.",
                "Exposure ve normalize değerler [0,1] aralığında kalmalıdır.",
                "Effective exposure ham exposure'ı ve 0.85 tavanını aşamaz.",
                "AO First Elo, Domestic Prior ve European Prior arasındaki kapalı aralıkta kalır.",
                "Dinamik hareket toplamı her maçta sıfırdır.",
                "Current maksimum hareket 200'ü, minimum sıralama korelasyonu 0.85 tabanını geçmemelidir.",
            ],
            s,
        ),
        Paragraph("25. Modelin Bugünkü Aktif Sözleşmesi", s["H1"]),
        status_box(
            [
                ["AO First Elo", "Statik v1.1"],
                ["Dynamic Scale / H / K", "225 / 40 / 28"],
                ["Season power carry", "0.85 - research v1"],
                ["Goal margin", "Inactive"],
                ["Competition K", "Inactive; UCL/UEL/UECL normal K eşit"],
                ["Win / progression / stage bonus", "Inactive"],
                ["Final / trophy reserve", "Inactive; final maçı normal Elo alır"],
                ["UCL forecast scale", "225"],
                ["Test paketi", "169 test geçti"],
            ],
            s,
        ),
        Paragraph("26. Ne Zaman Tamamlanmış Sayılacak?", s["H1"]),
        body(
            "Matematiksel araştırma v1 tamamlandı ve yeni kanıt gelene kadar parametreler "
            "donduruldu. Bu sürüm offline simülasyon, pilot raporlama ve maç sonrası puan "
            "motorunun production modülüne aktarılması için kullanılabilir.",
            s,
        ),
        *bullets(
            [
                "2026/27 sezonu untouched holdout olarak kilitlenir; bu veriyle parametre ayarlanmaz.",
                "Tahminler maçtan önce değiştirilemez loga yazılır.",
                "İlk anlamlı kontrol 2027 başında lig aşamaları bittikten sonra yapılır.",
                "Nihai production kararı Avrupa sezonu tamamlandığında, Haziran 2027'de verilir.",
                "Brier, log loss, ranking, rating hareketi ve üç turnuva segmenti birlikte geçmelidir.",
            ],
            s,
        ),
        note(
            "Son yorum",
            "Model bugün kullanılabilir ve denetlenebilir bir araştırma ürünüdür. Ancak "
            "2026/27 untouched holdout tamamlanmadan 'gelecekte kesin kanıtlanmış production "
            "modeli' olarak tanıtılmamalıdır. Bu sınır bir eksiklik saklamak değil, doğru "
            "model yönetişimidir.",
            s,
        ),
    ]
    return out


def short_story() -> list:
    s = styles()
    return [
        Spacer(1, 1.1 * cm),
        Paragraph("AO European Elo", s["Title"]),
        Paragraph("Kısa Model Açıklaması", s["Subtitle"]),
        Paragraph("Toplantı ve hızlı okuma sürümü", s["CoverLine"]),
        Spacer(1, 0.55 * cm),
        note(
            "Modelin tek cümlelik tanımı",
            "Takımın lig gücü, yerel başarısı ve beş sezonluk Avrupa geçmişinden bir "
            "başlangıç Elo'su üretir; sonra her Avrupa maçında rakip gücü ve sürpriz "
            "derecesine göre sıfır toplamlı biçimde günceller.",
            s,
        ),
        Paragraph("1. İki Katman", s["H1"]),
        table(
            [
                ["Katman", "Ne yapar?"],
                ["AO First Elo", "Takımın sezona hangi güçle başladığını hesaplar"],
                ["Dynamic Power Elo", "Oynanan her maçtan sonra canlı gücü değiştirir"],
            ],
            s,
            [5.0 * cm, 11.5 * cm],
        ),
        Paragraph("2. Başlangıç Elo'sunun Dört Sinyali", s["H1"]),
        table(
            [
                ["Sinyal", "Anlam"],
                ["Ülke gücü", "Takımın geldiği yerel ligin Avrupa seviyesi"],
                ["Yerel başarı", "Lig sırası, lig şampiyonluğu ve yerel kupa"],
                ["Kulüp Avrupa geçmişi", "Takımın son beş sezondaki kendi Avrupa puanı"],
                ["Exposure", "Kulübün Avrupa sinyaline ne kadar güvenileceği"],
            ],
            s,
            [5.3 * cm, 11.2 * cm],
        ),
        Paragraph("3. Dondurulan Başlangıç Parametreleri", s["H1"]),
        table(
            [
                ["Parametre", "Değer"],
                ["Base Rating", "500"],
                ["Sezon ağırlıkları", "0.07 / 0.13 / 0.20 / 0.27 / 0.33"],
                ["Country Benchmark / Gamma", "25 / 0.80"],
                ["League / Achievement Component", "140 / 160"],
                ["Achievement Alpha", "0.40"],
                ["European Benchmark / Max Boost", "20 / 420"],
                ["Exposure season / match", "0.60 / 0.40"],
                ["Effective exposure cap", "0.85"],
            ],
            s,
            [8.0 * cm, 8.5 * cm],
        ),
        PageBreak(),
        Paragraph("4. Başlangıç Formülleri", s["H1"]),
        formula(
            "League Strength Norm = min[1, log(1 + weighted country) / log(1 + 25)]",
            s,
        ),
        formula("League Strength = League Strength Norm ^ 0.80", s),
        formula(
            "Domestic Prior = 500 + 140 x League Strength + "
            "160 x Achievement x [0.40 + 0.60 x League Strength]",
            s,
        ),
        formula(
            "European Prior = 500 + 420 x min[1, log(1 + weighted club points) / log(1 + 20)]",
            s,
        ),
        formula("European Exposure = 0.60 x Season Exposure + 0.40 x Match Exposure", s),
        formula("Effective Exposure = min(European Exposure, 0.85)", s),
        formula(
            "AO First Elo = Domestic Prior + Effective Exposure x (European Prior - Domestic Prior)",
            s,
        ),
        body(
            "Avrupa geçmişi yoksa exposure 0 ve final Domestic Prior'dur. Ham exposure "
            "1.00 olsa bile effective değer 0.85'tir; yerel sinyalin yüzde 15'i korunur.",
            s,
        ),
        Paragraph("5. Yerel Başarı Özeti", s["H1"]),
        *bullets(
            [
                "Lig şampiyonu league_finish_score 1.00 alır.",
                "Pozisyon bilinmiyor ve şampiyon değilse lig skoru 0.10'dur.",
                "Yerel kupa şampiyonu cup base 0.62 alır.",
                "Double bonus yalnız lig + kupa birlikteyse 0.08 x league finish'tir.",
                "Domestic Achievement en fazla 1.10 olabilir.",
            ],
            s,
        ),
        Paragraph("6. Sezon Başında Carry", s["H1"]),
        formula(
            "New Power Start = 0.15 x Current AO First Elo + 0.85 x Previous Power End",
            s,
        ),
        body(
            "0.85 carry 6/6 görülmemiş fold kazandı; Brier farkı -0.002612 ve güven "
            "aralığı [-0.004017, -0.001286] oldu. Önceki sezonu olmayan takım güncel "
            "AO First Elo ile başlar.",
            s,
        ),
        PageBreak(),
        Paragraph("7. Maç Olasılığı ve Puan Değişimi", s["H1"]),
        formula(
            "E = 1 / [1 + 10 ^ (-(Home Rating - Away Rating + H) / 225)]",
            s,
        ),
        formula("Delta = 28 x (S - E)", s),
        formula("Home_new = Home_old + Delta; Away_new = Away_old - Delta", s),
        table(
            [
                ["Değer", "Aktif kural"],
                ["Scale", "225"],
                ["Saha avantajı", "40; nötr sahada 0"],
                ["K", "28"],
                ["S", "Galibiyet 1, beraberlik 0.5, mağlubiyet 0"],
                ["Toplam puan", "Her maçta sıfır toplamlı"],
            ],
            s,
            [6.0 * cm, 10.5 * cm],
        ),
        note(
            "Örnek",
            "Ev 850, deplasman 800 ve H=40 ise beklenen ev skoru yaklaşık 0.715'tir. "
            "Ev kazanırsa +7.98; berabere kalırsa -6.02; kaybederse -20.02 puan değişir.",
            s,
        ),
        Paragraph("8. Turnuvalar, Eleme ve Final", s["H1"]),
        *bullets(
            [
                "UCL, UEL ve UECL aynı Avrupa rating ölçeğindedir.",
                "Normal maç K'sı üç turnuvada da 28'dir; zorluk rakip rating'inden gelir.",
                "İki ayaklı turun her maçı ayrı normal Elo güncellemesi alır.",
                "Tur geçme, final veya kupa için ek aktif bonus yoktur.",
                "Final maçı yine normal K=28 güncellemesini alır.",
                "Penaltı shootout ayrı galibiyet olarak Elo'ya yazılmaz.",
                "Aktif model 1-0 ile 4-0'ı aynı S sonucu kabul eder; gol farkı katmanı adaydır ama kapalıdır.",
            ],
            s,
        ),
        Paragraph("9. Neler Test Edildi Ama Eklenmedi?", s["H1"]),
        table(
            [
                ["Katman", "Karar"],
                ["Gol farkı weight 0.5 / cap 2", "Yön iyi, güven aralığı belirsiz - kapalı"],
                ["UCL/UEL/UECL maç K çarpanı", "Stabil değil - kapalı"],
                ["1.00/0.65/0.45 katı K", "UEL'i bozdu - reddedildi"],
                ["Galibiyet ve tur prestij bonusu", "Reddedildi"],
                ["Aşama/final progression", "Yetersiz kanıt - kapalı"],
                ["Trophy reserve", "Carry'ye ek fayda yok - reddedildi"],
                ["UCL özel forecast scale", "2/5 fold - 225 korundu"],
            ],
            s,
            [7.0 * cm, 9.5 * cm],
        ),
        PageBreak(),
        Paragraph("10. Test Sonuçlarının Özeti", s["H1"]),
        table(
            [
                ["Test", "Sonuç"],
                ["Tarihsel veri", "8 sezon, 6.340 kesin tarihli UEFA maçı"],
                ["Dynamic core", "6/6 fold; Scale 225, H 40, K 28"],
                ["Carry", "6/6 fold; 0.85 geçici kabul"],
                ["Harici paired ClubElo", "363 unseen maç; genel fark kesin değil"],
                ["UCL AO vs ClubElo", "AO 0.171128, ClubElo 0.156801; residual risk"],
                ["AO-ClubElo rank Spearman", "0.888094"],
                ["Maksimum current rating hareketi", "136.346657; guardrail 200"],
                ["Minimum current rank korelasyonu", "0.907974; guardrail 0.85"],
                ["Unit test", "169 test geçti"],
            ],
            s,
            [7.3 * cm, 9.2 * cm],
        ),
        Paragraph("11. Bugünkü Model", s["H1"]),
        status_box(
            [
                ["AO First Elo", "v1.1"],
                ["Dynamic model", "225 / 40 / 28"],
                ["Carry", "0.85"],
                ["Aktif bonus", "Yok"],
                ["Turnuva K farkı", "Yok"],
                ["Durum", "Matematiksel araştırma v1 tamamlandı"],
            ],
            s,
        ),
        Paragraph("12. Kalan Tek Büyük Doğrulama", s["H1"]),
        body(
            "2026/27 sezonu untouched holdout olarak kilitlenecek. İlk ara kontrol 2027 "
            "başında, nihai production kararı Haziran 2027'de verilecek. O zamana kadar "
            "model kullanılabilir bir araştırma ve pilot ürünüdür; parametreler yeni kanıt "
            "olmadan değiştirilmemelidir.",
            s,
        ),
    ]


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "Title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="AOBold",
            fontSize=25,
            leading=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#14213D"),
            spaceAfter=7,
        ),
        "Subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontName="AOBold",
            fontSize=14.5,
            leading=19,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#1B365D"),
            spaceAfter=5,
        ),
        "CoverLine": ParagraphStyle(
            "CoverLine",
            parent=base["Normal"],
            fontName="AORegular",
            fontSize=10,
            leading=14,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#5F6B7A"),
            spaceAfter=12,
        ),
        "H1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName="AOBold",
            fontSize=13.2,
            leading=17,
            textColor=colors.HexColor("#163A63"),
            spaceBefore=10,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "H2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName="AOBold",
            fontSize=10.8,
            leading=14,
            textColor=colors.HexColor("#28527A"),
            spaceBefore=7,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "Body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="AORegular",
            fontSize=9.15,
            leading=13.2,
            textColor=colors.HexColor("#202B38"),
            spaceAfter=7,
            alignment=TA_LEFT,
        ),
        "Small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="AORegular",
            fontSize=7.65,
            leading=10.0,
            textColor=colors.HexColor("#202B38"),
        ),
        "Formula": ParagraphStyle(
            "Formula",
            parent=base["BodyText"],
            fontName="AORegular",
            fontSize=8.9,
            leading=13.0,
            textColor=colors.HexColor("#102A43"),
            leftIndent=9,
            rightIndent=9,
            borderColor=colors.HexColor("#AFC1D5"),
            borderWidth=0.55,
            borderPadding=7,
            backColor=colors.HexColor("#F4F7FB"),
            spaceBefore=3,
            spaceAfter=7,
        ),
        "NoteTitle": ParagraphStyle(
            "NoteTitle",
            parent=base["BodyText"],
            fontName="AOBold",
            fontSize=8.8,
            leading=11,
            textColor=colors.HexColor("#163A63"),
            spaceAfter=3,
        ),
    }


def body(text: str, s: dict[str, ParagraphStyle]) -> Paragraph:
    return Paragraph(text, s["Body"])


def formula(text: str, s: dict[str, ParagraphStyle]) -> Paragraph:
    return Paragraph(text, s["Formula"])


def note(title: str, text: str, s: dict[str, ParagraphStyle]) -> Table:
    item = [Paragraph(title, s["NoteTitle"]), Paragraph(text, s["Body"])]
    box = Table([[item]], colWidths=[16.5 * cm], hAlign="LEFT")
    box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EDF4FA")),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#8FAAC3")),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return box


def status_box(rows: list[list[str]], s: dict[str, ParagraphStyle]) -> Table:
    wrapped = [[Paragraph(str(cell), s["Small"]) for cell in row] for row in rows]
    result = Table(wrapped, colWidths=[6.0 * cm, 10.5 * cm], hAlign="LEFT")
    result.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#173B63")),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
                ("FONTNAME", (0, 0), (0, -1), "AOBold"),
                ("BACKGROUND", (1, 0), (1, -1), colors.HexColor("#F2F6FA")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C7D3DF")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return result


def table(rows: list[list[str]], s: dict[str, ParagraphStyle], widths: list[float]) -> Table:
    wrapped = [[Paragraph(str(cell), s["Small"]) for cell in row] for row in rows]
    result = Table(wrapped, colWidths=widths, hAlign="LEFT", repeatRows=1)
    result.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE7F2")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#14213D")),
                ("FONTNAME", (0, 0), (-1, 0), "AOBold"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C7D3DF")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return result


def bullets(items: list[str], s: dict[str, ParagraphStyle]) -> list[Paragraph]:
    return [
        Paragraph(
            f"<font color='#28527A'>•</font> {item}",
            ParagraphStyle(
                f"Bullet{index}",
                parent=s["Body"],
                leftIndent=11,
                firstLineIndent=-8,
                spaceAfter=4,
            ),
        )
        for index, item in enumerate(items)
    ]


if __name__ == "__main__":
    main()
