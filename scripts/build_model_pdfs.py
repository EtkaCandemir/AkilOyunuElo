from __future__ import annotations

from reportlab.lib.units import cm
from reportlab.platypus import Spacer

from pdf_common import (
    PdfSpec,
    body,
    build_pdf,
    bullets,
    callout,
    cover,
    formula,
    h1,
    h2,
    page_break,
    styles,
    table,
)


DOCUMENT_DATE = "28 Ağustos 2026"
MODEL_VERSION = "ao-european-elo-v2.0-dev-freeze"
PREDICTION_VERSION = "ao-ml-poisson-ensemble-v1-production"

DETAILED_SPEC = PdfSpec(
    filename="AkilOyunu_Elo_Model_Aciklayici.pdf",
    title="AO European Elo",
    subtitle="Tam Teknik Model Açıklaması",
    version="AO European Elo v2 | Production revizyonu 2026-08-28",
    document_date=DOCUMENT_DATE,
    subject="AO First Elo, AO Live Elo ve production 1X2 tahmin modelinin tam açıklaması",
)

SHORT_SPEC = PdfSpec(
    filename="AkilOyunu_Elo_Model_Kisa.pdf",
    title="AO European Elo",
    subtitle="Kısa Model Açıklaması",
    version="AO European Elo v2 | Kısa açıklama",
    document_date=DOCUMENT_DATE,
    subject="AO European Elo production modelinin kısa ve sunuma uygun açıklaması",
)


def main() -> None:
    detailed_output, detailed_docs = build_pdf(DETAILED_SPEC, detailed_story())
    short_output, short_docs = build_pdf(SHORT_SPEC, short_story())
    print(f"PDF written: {detailed_output}")
    print(f"PDF synced:  {detailed_docs}")
    print(f"PDF written: {short_output}")
    print(f"PDF synced:  {short_docs}")


def detailed_story() -> list[object]:
    s = styles()
    out: list[object] = cover(
        DETAILED_SPEC,
        [
            ["Rating sürümü", MODEL_VERSION],
            ["Tahmin sürümü", PREDICTION_VERSION],
            ["Geliştirme dönemi", "2018/19 - 2025/26"],
            ["Production kararı", "PROMOTE_WITH_MONITORING"],
            ["Prospective izleme", "2026/27 lig aşaması ve sonrası"],
            ["Teknik otorite", "contracts/ao_european_elo_v2_production.json"],
        ],
        s,
        summary_title="Belgenin amacı",
        summary=(
            "Bu belge güncel production modelini baştan sona açıklar. AO First Elo ve AO Live Elo "
            "takım gücünü, ayrı ML + Domestic Poisson katmanı ise kullanıcıya sunulan 1X2 "
            "olasılığını üretir. Tahmin katmanı rating state'ine geri beslenmez."
        ),
    )

    out += [
        h1("1. Modelin Ürettiği Değerler", s),
        body(
            "Model güç muhasebesi ile maç olasılığı tahminini ayırır. Bu ayrım hem açıklanabilirlik "
            "hem de hata durumunda güvenli fallback için temel tasarım kararıdır.",
            s,
        ),
        table(
            [
                ["Çıktı", "Anlam", "Neyi değiştirir?"],
                ["AO First Elo", "Sezon başındaki yapısal takım gücü", "Power Elo'nun başlangıç noktası"],
                ["Power Elo", "Maçlarla sıfır toplamlı değişen güç", "AO Live Elo"],
                ["Progression Bonus", "Son 16 ve sonrası tur başarısı", "AO Live Elo; sezonluk"],
                ["AO Live Elo", "Kullanıcıya gösterilen tek canlı rating", "Base AO olasılığı ve sonraki maç"],
                ["Production 1X2", "ML + Domestic Poisson ile sunulan H/D/A", "Yalnız olasılık; ratinge feedback yok"],
            ],
            [3.1 * cm, 6.5 * cm, 6.85 * cm],
            s,
        ),
        formula(
            [
                "AO Live Elo = Power Elo + Progression Bonus",
                "Achievement Reserve = 0 (kapalı)",
            ],
            s,
        ),
        callout(
            "En önemli ayrım",
            "Gol farkı ve xG maç sonrası Power Elo'yu değiştirir. Structural ML ve Domestic "
            "Poisson ise yalnız pre-match 1X2 tahminini değiştirir; rating sıralamasını değiştirmez.",
            s,
            tone="green",
        ),
        h1("2. Zaman ve Veri Sızıntısı Sözleşmesi", s),
        body(
            "Statik season ratingin üretildiği hedef sezondur. t, hedef sezondan önce tamamlanan "
            "en son sezondur. Bir maçın tahmini sonuç görülmeden ve kickoff'tan önce kilitlenir.",
            s,
        ),
        *bullets(
            [
                "Beş yıllık pencerede ağırlıklar eskiden yeniye 0.07, 0.13, 0.20, 0.27, 0.33'tür.",
                "Hedef sezonun sonucu AO First Elo geçmişine yazılamaz.",
                "Maçlar exact UTC, ardından match_id artan sırasıyla işlenir.",
                "Aynı kickoff batch'indeki bütün tahminler sonuçlar state'e girmeden üretilir.",
                "2026/27 qualifying geliştirme sürecine temas ettiği için prospective kapsam lig aşaması ve sonrasıdır.",
            ],
            s,
        ),
        page_break(),
        h1("3. AO First Elo Girdileri", s),
        table(
            [
                ["Dosya", "Anahtar", "Rol"],
                ["teams.csv", "team_id", "Kalıcı takım, ülke ve lig kimliği"],
                ["country_coefficients.csv", "season + country_code", "Beş sezon UEFA ülke gücü"],
                ["domestic_context.csv", "season + team_id", "Lig, kupa ve Domestic Surprise geçmişi"],
                ["club_european_points.csv", "season + team_id + country_code", "Kulüp Avrupa puanı ve exposure"],
            ],
            [4.2 * cm, 5.0 * cm, 7.25 * cm],
            s,
        ),
        *bullets(
            [
                "Her hedef takım için club history satırı zorunludur.",
                "Avrupa geçmişi yoksa beş sezon puan, played ve matches açık sıfır yazılır.",
                "Duplicate anahtar, NaN/sonsuz/negatif puan ve geçersiz boolean reddedilir.",
                "Provider kimlikleri AO'nun kalıcı team_id değerinin yerine kullanılamaz.",
            ],
            s,
        ),
        h1("4. 500-2000 Referans Ölçeği", s),
        formula(
            [
                "M = 1500 / (903.92 - 500) = 3.713606654783126",
                "AO Elo v2 = 500 + M * (AO Elo v1.1 - 500)",
            ],
            s,
        ),
        table(
            [
                ["Bileşen", "Aktif v2 değeri"],
                ["Base rating", "500"],
                ["Domestic League", "519.9049316696"],
                ["Domestic Achievement", "594.1770647653"],
                ["European Prior max boost", "1559.7147950089"],
            ],
            [9.0 * cm, 7.45 * cm],
            s,
        ),
        body(
            "500-2000 hard cap değildir. Final rating veya canlı rating clipping ile kesilmez. "
            "Scale, H ve K aynı affine oranla büyütüldüğü için görünür puan aralığı tek başına "
            "olasılık geometrisini değiştirmez.",
            s,
        ),
        h1("5. Ülke ve Lig Gücü", s),
        body(
            "UEFA ülke katsayısı kulübün kendi geçmişi değil, geldiği ligin Avrupa gücü sinyalidir.",
            s,
        ),
        formula(
            [
                "Weighted Country = sum(w_i * country_points_i)",
                "u_country = ln(1 + Weighted Country) / ln(1 + 25)",
                "Country Norm = min(u_country, 1)   [country_tail_beta=0]",
                "League Strength = Country Norm ^ 0.80",
            ],
            s,
        ),
        page_break(),
        h1("6. Domestic Achievement ve Domestic Prior", s),
        formula(
            [
                "Position Percentile = (N - position) / (N - 1)",
                "Percentile Score = 0.15 + 0.70 * Position Percentile",
                "League Finish = max(Percentile Score, Champion Base)",
                "Cup Contribution = 0.129032 * min(League Finish, Cup Base)",
                "Achievement = min(1.10, max(League Finish, Cup Base) + Cup Contribution)",
                "Achievement Scale = 0.40 + 0.60 * League Strength",
            ],
            s,
        ),
        table(
            [
                ["Yerel durum", "Model davranışı"],
                ["Lig şampiyonu", "Champion Base = 1.00"],
                ["Lig sırası bilinmiyor", "League Finish = 0.15 (percentile tabanı)"],
                ["Kupa şampiyonu", "Cup Base = 0.62"],
                ["Kupa katkısı", "+0.129032 * min(League Finish, Cup Base)"],
                ["Achievement safety cap", "1.10 (ulaşılabilir maksimum 1.08)"],
            ],
            [7.5 * cm, 8.95 * cm],
            s,
        ),
        formula(
            [
                "Domestic Prior = 500",
                "  + 519.9049316696 * League Strength",
                "  + 594.1770647653 * Domestic Achievement * Achievement Scale",
            ],
            s,
        ),
        callout(
            "Şampiyonluk basamağı",
            "Şampiyonun 1.00 tabanı ile ikinci sıra arasındaki sıçrama bilinçli domain kararıdır. "
            "Kupa artık taban değil katkıdır: ağırlık 0.08 * 1.00 / 0.62 = 0.129032 olarak "
            "türetilmiştir ve şampiyon+kupa toplamını 1.0800'de birebir korur. Kupa "
            "kazanmayanda terim sıfırdır. Bilinmeyen lig sırası percentile tabanının "
            "(0.15) altına inemez.",
            s,
            tone="amber",
        ),
        h1("7. Variance-Controlled Domestic Surprise", s),
        body(
            "Takımın güncel lig performansı kendi son beş sezonluk normal seviyesiyle "
            "karşılaştırılır. Varyans sinyalin yönünü değiştirmez; oynak geçmişte güveni azaltır.",
            s,
        ),
        formula(
            [
                "Historical Mean = sum(w_i * Historical Position Percentile_i)",
                "Volatility = sqrt(sum(w_i * (P_i - Historical Mean)^2))",
                "Consistency = 1 - 0.50 * min(1, 2 * Volatility)",
                "Effective Surprise = (P_current - Historical Mean) * Consistency",
                "Domestic Adjustment = clip(594.1770647653 * Achievement Scale",
                "                           * 0.40 * Effective Surprise, -30, +30)",
            ],
            s,
        ),
        *bullets(
            [
                "Beş tam lig sezonu yoksa adjustment sıfırdır.",
                "Pozitif ve negatif sürpriz aynı katsayıyla işlenir.",
                "Final AO First etkisi exposure yükseldikçe (1 - e_eff) oranında azalır.",
            ],
            s,
        ),
        h1("8. Kulüp Avrupa Geçmişi ve Exposure", s),
        formula(
            [
                "Weighted European History = sum(w_i * club_points_i)",
                "pw = sum(w_i * played_i)",
                "History Rate = History * (1 + 0.20) / (pw + 0.20),  pw > 0",
                "European Norm = min(ln(1 + History Rate) / ln(1 + 20), 1)",
                "European Prior = 500 + 1559.7147950089 * European Norm",
                "Season Exposure = sum(w_i * played_i)",
                "Match Exposure = sum(w_i * min(1, matches_i / match_cap_i))",
                "e = 0.60 * Season Exposure + 0.40 * Match Exposure",
                "e_eff = min(e, 0.65)",
            ],
            s,
        ),
        body(
            "European points performansı, exposure ise bu performansa ne kadar güvenileceğini "
            "ölçer. 0.65 tavanı tam Avrupa kanıtında bile Domestic Prior'ın en az %35'ini korur. "
            "Katılım normalizasyonu, Avrupa'ya girilmeyen sezonun 'girdim ve puan alamadım' "
            "gibi sayılmasını kaldırır: ağırlıklı geçmiş, kulübün gerçekten girebildiği "
            "ağırlığa bölünür. Beş sezonun tamamında oynayan kulüp tanım gereği hiç hareket "
            "etmez ve katman ratingi asla düşürmez. Exposure ağırlığına dokunulmaz.",
            s,
        ),
        callout(
            "Exposure kararı",
            "Aktif çekirdek 4.884 unseen maçta Brier 0.566413, log-loss 0.956259 ve "
            "accuracy %55.917 üretir; referans ablation koluna karşı iki loss metriği de "
            "6/6 foldda iyileşir.",
            s,
            tone="green",
        ),
        formula(
            [
                "Adjusted Domestic Prior = Domestic Prior + Domestic Adjustment",
                "AO First Elo = Adjusted Domestic Prior",
                "             + e_eff * (European Prior - Adjusted Domestic Prior)",
            ],
            s,
        ),
        page_break(),
        h1("9. Dynamic Power Elo", s),
        formula(
            [
                "D = Home AO Live - Away AO Live + H_effective",
                "E_home = 1 / (1 + 10 ^ (-D / 835.5614973262))",
                "H = 148.5442661913; neutral sahada 0",
                "K = 103.9809863339; power carry = 0",
                "Base Residual = S_home - E_home",
            ],
            s,
        ),
        body(
            "H kalıcı puan değildir; yalnız maç beklentisi hesaplanırken kullanılan geçici saha "
            "avantajıdır. S_home saha galibiyetinde 1, beraberlikte 0.5, mağlubiyette 0'dır.",
            s,
        ),
        h2("9.1 Kontrollü gol farkı", s),
        formula(
            [
                "M_GD = 1, draw / shootout / GD <= 1 ise",
                "M_GD = 1 + 0.15 * ln(min(GD,4)) * exp(-abs(D)/300), aksi halde",
                "GD Residual = Base Residual * M_GD",
            ],
            s,
        ),
        body(
            "Gol farkı 4'te tavanlanır. Rating farkı büyüdükçe bonus söner; ağır favorinin büyük "
            "galibiyeti denk takımlar arasındaki aynı skor kadar rating şişirmez.",
            s,
        ),
        h2("9.2 Bounded xG performansı", s),
        formula(
            [
                "Q_xG = tanh((xG_home - xG_away) / 1.25)",
                "xG Residual = 0.30 * abs(Base Residual) * Q_xG",
                "Final Residual = GD Residual + xG Residual",
                "Power Delta = K * Final Residual",
            ],
            s,
        ),
        *bullets(
            [
                "xG yalnız iki taraf için aynı zaman kapsamıyla doğrulanmışsa kullanılır.",
                "Beraberlik ve shootout xG adjustment üretmez.",
                "Eksik xG sıfır değildir; model gol farkı katmanına fallback yapar.",
                "Kazananın temel sonuç kazanımının en az %70'i analitik olarak korunur.",
            ],
            s,
        ),
        formula(
            [
                "Home Power New = Home Power Old + Power Delta",
                "Away Power New = Away Power Old - Power Delta",
                "Power toplamı her maçta korunur (tolerans 1e-9).",
            ],
            s,
        ),
        h2("9.3 Penaltı ve uzatma", s),
        body(
            "Skor 90 dakika veya uzatma oynandıysa 120 dakika field score'dur. Shootout golleri "
            "eklenmez. Penaltıya giden ikinci ayak sahada 2-0 bittiyse Power Elo için galibiyettir; "
            "decided_on_penalties yalnız GD ve xG ek sinyallerini kapatır.",
            s,
        ),
        page_break(),
        h1("10. European Progression Bonus", s),
        table(
            [
                ["Tamamlanan başarı", "UCL", "UEL", "UECL"],
                ["Son 16", "+12", "+8", "+4"],
                ["Çeyrek final", "+12", "+8", "+4"],
                ["Yarı final", "+12", "+8", "+4"],
                ["Final / şampiyonluk", "+12", "+8", "+4"],
                ["Sezonluk cap", "48", "32", "16"],
            ],
            [6.0 * cm, 3.5 * cm, 3.5 * cm, 3.45 * cm],
            s,
        ),
        *bullets(
            [
                "Knockout play-off, ön elemeler, lig aşaması ve ilk 8 bonus üretmez.",
                "Bonus aynı tie_id için bir kez, eşleşme tamamlanınca kazanana verilir.",
                "Kaybedenden puan düşülmez; bonus yeni sezona taşınmaz.",
                "Penaltıyla tur geçen takım progression bonusunu alır.",
            ],
            s,
        ),
        callout(
            "Conservation notu",
            "Progression winner-only olduğu için görünen AO Live toplamı artabilir. Sıfır-toplam "
            "zorunluluğu yalnız Power Elo güncellemesine aittir.",
            s,
            tone="amber",
        ),
        h1("11. Current AO 1X2 Olasılıkları", s),
        formula(
            [
                "P_draw_raw = d * (4 * E_home * (1 - E_home)) ^ 1.00",
                "P_draw = min(P_draw_raw, 2 * min(E_home, 1-E_home))",
                "P_home = E_home - 0.5 * P_draw",
                "P_away = 1 - E_home - 0.5 * P_draw",
                "d = 0.24 normal/two-leg; d = 0.12 single-match knockout",
            ],
            s,
        ),
        body(
            "E_home galibiyet olasılığı değil, beklenen maç puanıdır. Dönüşüm olasılık toplamını "
            "ve P_home + 0.5*P_draw = E_home kimliğini korur. Single-match bilgisi sonuçtan değil "
            "fikstür metadata'sından gelir.",
            s,
        ),
        h1("12. Production ML + Domestic Poisson", s),
        body(
            "Web sitesine sunulan 1X2, Current AO olasılığını iki bağımsız prediction-only "
            "challenger ile kontrollü biçimde birleştirir.",
            s,
        ),
        formula(
            [
                "P_current_ml = LogBlend(P_AO, P_structural_logistic, 0.90)",
                "P_ao_poisson = LogBlend(P_AO, P_domestic_poisson_rho0, 0.50)",
                "P_served = LogBlend(P_current_ml, P_ao_poisson, 0.50)",
                "LogBlend(A,B,w)_c proportional to A_c^(1-w) * B_c^w",
            ],
            s,
        ),
        h2("12.1 Structural Logistic", s),
        body(
            "Pre-match AO log-odds, rating ve exposure farkı; turnuva/tur/format; geçmiş Avrupa "
            "formu; gol eğilimleri; dinlenme ve maç yoğunluğu; leg ve bilinen aggregate durumu "
            "kullanılır. Hedef maçın sonucu, skoru, xG'si veya post-match ratingi kullanılmaz.",
            s,
        ),
        h2("12.2 Domestic Dynamic Poisson", s),
        table(
            [
                ["Parametre", "Aktif değer"],
                ["Eğitim kapsamı", "45.419 yerel maç, 19 lig, 508 kaynak takım"],
                ["AO ile eşleşen kulüp", "171"],
                ["Team learning rate / carry / shrinkage", "0.02 / 0.90 / 10"],
                ["Team venue context", "Kapalı"],
                ["mu / Elo slope", "0.2804007979 / 0.6663036965"],
                ["Attack / defence coefficient", "0.0920974542 / 0.0606564817"],
                ["Venue / L2 / Dixon-Coles rho", "0 / 10 / 0"],
            ],
            [8.1 * cm, 8.35 * cm],
            s,
        ),
        body(
            "Domestic Poisson lig içi hücum ve savunma profillerini güvenilirlikle küçülterek "
            "Avrupa maçına taşır. İki takımda da yerel geçmiş yoksa Poisson bileşeni Current AO'ya "
            "döner.",
            s,
        ),
        h2("12.3 Artifact ve fallback", s),
        *bullets(
            [
                "ML artifact, feature schema ve Domestic Poisson state SHA-256 ile doğrulanır.",
                "28 Ağustos revision'u parametreleri değiştirmez; giriş ve state hatalarını düzeltir.",
                "Domestic checkpoint 2.0 olay ID'lerini ve lig başına son kickoff'u saklar; tekrar ve eski batch reddedilir.",
                "Provider sezonu kaynak sezonundan veya takvim liglerinde UTC yılından gelir; AO sezonundan geri türetilmez.",
                "Kanonik lig/takım + UTC fikstürü merge/replay girişinde benzersizdir. Skor çelişkisi resmi kaynak kararı gerektirir.",
                "Domestic cutoff üretim zamanını aşamaz ve fixture kickoff'undan önce olmalıdır; aksi halde AO fallback kullanılır.",
                "Timezone'suz tarih, kesirli gol ve çelişen metadata reddedilir. Canonical true/false ile 0/1 aynı hesaplanır.",
                "Aynı takım aynı kickoff'ta iki kez oynayamaz. Tek maçta shootout yoksa tur atlayan, eşit olmayan field score'un galibidir.",
                "Startup artifact hatası strict modda servisi durdurur.",
                "Satır bazlı feature/state hatası final tahmini Current AO 1X2'ye döndürür.",
                "Fallback nedeni ve bütün model fingerprint'leri pre-match loga yazılır.",
                "rating_feedback_applied her production satırında false olmak zorundadır.",
            ],
            s,
        ),
        h1("13. Aktif Parametre Özeti", s),
        table(
            [
                ["Katman", "Aktif sözleşme"],
                ["Season weights", "0.07 / 0.13 / 0.20 / 0.27 / 0.33"],
                ["Country", "benchmark 25, gamma 0.80, tail beta 0"],
                ["European history", "benchmark 20, tail beta 0"],
                ["Exposure", "%60 season + %40 match; effective cap 0.65"],
                ["Domestic Surprise", "theta 0.40, variance 0.50, cap +/-30, 5 sezon"],
                ["Dynamic core", "Scale 835.5615, H 148.5443, K 103.9810, carry 0"],
                ["Draw", "0.24; single-match 0.12; shape 1.00"],
                ["Goal difference", "alpha 0.15, tau 300, cap 4"],
                ["xG", "ratio 0.30, scale 1.25, minimum winner ratio 0.70"],
                ["Progression", "12/8/4; caps 48/32/16; KPO kapalı"],
                ["Production prediction", "%50 Current ML + %50 AO Poisson rho=0"],
            ],
            [6.0 * cm, 10.45 * cm],
            s,
        ),
        h1("14. Tarihsel Performans", s),
        body(
            "Aşağıdaki sonuçlar 2018/19-2025/26 geliştirme penceresindeki 4.884 unseen Avrupa "
            "maçına aittir. Bunlar walk-forward kanıttır; 2026/27 prospective izlemenin yerine geçmez.",
            s,
        ),
        table(
            [
                ["Model", "Brier", "Log-loss", "Accuracy", "Spearman", "Pairwise"],
                ["Reference core", "0.568053", "0.959174", "0.554464", "0.681487", "0.758195"],
                ["Current AO rating core", "0.566413", "0.956259", "0.559173", "0.683258", "0.759421"],
                ["Tarihsel nested ensemble", "0.562065", "0.949965", "0.561220", "-", "-"],
            ],
            [4.4 * cm, 2.4 * cm, 2.7 * cm, 2.4 * cm, 2.3 * cm, 2.25 * cm],
            s,
        ),
        callout(
            "Prediction kazancı",
            "Tarihsel nested ensemble, Current AO'ya göre Brier'ı 0.004348 ve log-loss'u "
            "0.006294 azaltmış, accuracy'yi 0.002048 artırmıştır. Fold başına kaynak/ağırlık "
            "seçen bu ölçüm, sabit production %50/%50 karışımının birebir replay'i değildir.",
            s,
            tone="green",
        ),
        h1("15. Aktif Olmayan Katmanlar", s),
        table(
            [
                ["Katman", "Durum / gerekçe"],
                ["Dynamic K", "Reddedildi; sabit K daha güvenilir"],
                ["Competition K", "Kapalı; normal maç K hiyerarşisi genellenmedi"],
                ["Power carry", "Kapalı; active carry 0"],
                ["Achievement Reserve", "Kapalı; historical backtestte zarar"],
                ["Team venue context", "Shadow; dependency belirsizliği sürüyor"],
                ["Draw shape 0.84", "Shadow; active shape 1.00"],
                ["Q1-Q5 / Q1-Q3 rakip profili", "Diagnostic; sezonlar arası devamlılık zayıf"],
                ["Domestic Surprise MOB", "Diagnostic; tekrar eden kararlı split bulunmadı"],
            ],
            [6.2 * cm, 10.25 * cm],
            s,
        ),
        h1("16. Güvenlik ve Operasyon", s),
        *bullets(
            [
                "Power Delta her maçta 1e-9 toleransında sıfır toplamlıdır.",
                "Duplicate maç, takım/sezon ve tie kayıtları reddedilir.",
                "Exposure [0,1] ve effective exposure <= raw exposure invariantları korunur.",
                "Progression cap, sezon reseti ve single-application kontrol edilir.",
                "Tahmin olasılıkları finite, negatif olmayan ve toplamı 1 olan değerlerdir.",
                "2026/27 monitoring AO fallback ile aynı locked fixture evreninde yürütülür.",
            ],
            s,
        ),
        callout(
            "Production kararı",
            "Rating çekirdeği KEEP durumundadır. ML + Domestic Poisson tahmini "
            "PROMOTE_WITH_MONITORING olarak aktiftir. Yeni parametre seçimi yapılmaz; 2026/27 "
            "kilitli tahminleriyle performans ve fallback oranı izlenir.",
            s,
            tone="blue",
        ),
    ]
    return out


def short_story() -> list[object]:
    s = styles()
    out: list[object] = cover(
        SHORT_SPEC,
        [
            ["Rating modeli", MODEL_VERSION],
            ["Tahmin modeli", PREDICTION_VERSION],
            ["Durum", "PROMOTE_WITH_MONITORING"],
            ["İzleme", "2026/27 lig aşaması ve sonrası"],
        ],
        s,
        summary_title="Tek cümlede model",
        summary=(
            "AO European Elo; ülke gücü, yerel başarı ve kulüp Avrupa geçmişinden sezon başı "
            "rating üretir; maç sonucu, gol farkı ve xG ile canlı gücü günceller; ML ve Domestic "
            "Poisson ile 1X2 olasılıklarını iyileştirir."
        ),
    )
    out += [
        h1("1. Dört Aşamalı Akış", s),
        table(
            [
                ["Aşama", "Ne yapar?", "Ana veri"],
                ["AO First Elo", "Sezon başı gücü belirler", "Ülke, lig/kupa, Avrupa geçmişi"],
                ["Power Elo", "Maç sonucuyla sıfır toplamlı güncellenir", "Rakip, saha, skor, xG"],
                ["Progression", "Son 16 ve sonrası sezonluk bonus", "Turnuva, stage, tie winner"],
                ["Production 1X2", "ML + Poisson ile H/D/A üretir", "Pre-match features, yerel form"],
            ],
            [3.6 * cm, 7.1 * cm, 5.75 * cm],
            s,
        ),
        h1("2. Başlangıç Elo'su", s),
        formula(
            [
                "Domestic Prior = 500 + 519.9049*LeagueStrength",
                "               + 594.1771*Achievement*AchievementScale",
                "European Prior = 500 + 1559.7148*EuropeanHistoryNorm",
                "AO First = Adjusted Domestic Prior",
                "         + e_eff*(European Prior-Adjusted Domestic Prior)",
            ],
            s,
        ),
        *bullets(
            [
                "Beş sezon ağırlığı: 0.07 / 0.13 / 0.20 / 0.27 / 0.33.",
                "Country benchmark 25; European benchmark 20.",
                "Exposure = %60 oynanan sezon + %40 maç kanıtı; effective cap 0.65.",
                "Domestic Surprise: katsayı 0.40, variance penalty 0.50, cap +/-30; beş tam sezon gerekir.",
                "500-2000 referans bandıdır; clipping yoktur.",
            ],
            s,
        ),
        page_break(),
        h1("3. Maç Sonrası Canlı Elo", s),
        formula(
            [
                "E_home = 1/(1+10^(-(HomeLive-AwayLive+H)/Scale))",
                "Scale=835.5615 | H=148.5443 | K=103.9810",
                "M_GD=1+0.15*ln(min(GD,4))*exp(-abs(D)/300)",
                "Q_xG=tanh((xG_home-xG_away)/1.25)",
                "Delta=K*[BaseResidual*M_GD + 0.30*abs(BaseResidual)*Q_xG]",
            ],
            s,
        ),
        *bullets(
            [
                "H geçici saha etkisidir; ratinge kalıcı eklenmez, nötr sahada sıfırdır.",
                "Tek farklı sonuçta gol farkı bonusu yoktur; GD 4'te tavanlanır.",
                "xG yalnız iki taraflı ve scope uyumluysa kullanılır; eksikte GD fallback'i vardır.",
                "28 Ağustos giriş/state düzeltmeleri parametreleri değiştirmez. Domestic checkpoint 2.0 tekrar ve eski batch'i reddeder; tahmin gelecekteki state'i kullanmaz.",
                "Beraberlik ve penaltı shootout xG/GD ek sinyali üretmez.",
                "Power Elo her maçta sıfır toplamlıdır.",
            ],
            s,
        ),
        h1("4. Tur Bonusu", s),
        table(
            [
                ["Turnuva", "Aşama başına", "Sezon cap"],
                ["UCL", "+12", "48"],
                ["UEL", "+8", "32"],
                ["UECL", "+4", "16"],
            ],
            [6.0 * cm, 5.2 * cm, 5.25 * cm],
            s,
        ),
        body(
            "Yalnız Son 16, çeyrek final, yarı final ve final tamamlandığında kazanana verilir. "
            "Knockout play-off ve ön elemeler bonus üretmez. AO Live = Power + Progression.",
            s,
        ),
        page_break(),
        h1("5. Production 1X2", s),
        formula(
            [
                "Current ML = LogBlend(Current AO, Structural Logistic, 0.90)",
                "AO Poisson = LogBlend(Current AO, Domestic Poisson rho=0, 0.50)",
                "Served 1X2 = LogBlend(Current ML, AO Poisson, 0.50)",
            ],
            s,
        ),
        body(
            "Structural Logistic AO ratingi, Avrupa formu, turnuva formatı, dinlenme ve yoğunluk "
            "feature'larını kullanır. Domestic Poisson 45.419 yerel maçtan hücum-savunma profili "
            "öğrenir. Bu katmanlar AO Live Elo'yu değiştirmez.",
            s,
        ),
        table(
            [
                ["Tahmin", "Brier", "Log-loss", "Accuracy"],
                ["Current AO", "0.566413", "0.956259", "0.559173"],
                ["Tarihsel nested ensemble", "0.562065", "0.949965", "0.561220"],
                ["Fark", "-0.004348", "-0.006294", "+0.002048"],
            ],
            [6.1 * cm, 3.45 * cm, 3.45 * cm, 3.45 * cm],
            s,
        ),
        callout(
            "Fallback",
            "Yukarıdaki ölçüm fold başına seçilen nested ensemble'dır, sabit %50/%50 production replay'i değildir. "
            "Artifact, feature veya Domestic Poisson state problemi oluşursa prediction Current AO "
            "1X2'ye döner ve neden loglanır. Rating feedback her durumda kapalıdır.",
            s,
            tone="blue",
        ),
        h1("6. Aktif / Kapalı", s),
        table(
            [
                ["Aktif", "Kapalı veya shadow"],
                ["AO First v2, Domestic Surprise", "Dynamic K, Competition K, Power carry"],
                ["Sabit K, GD alpha 0.15, bounded xG", "Achievement Reserve, stage-weighted bonus"],
                ["Progression 12/8/4", "Team venue, draw shape 0.84"],
                ["ML + Domestic Poisson 1X2", "Rakip profil ve MOB diagnostikleri"],
            ],
            [8.2 * cm, 8.25 * cm],
            s,
        ),
        callout(
            "Güncel karar",
            "Rating çekirdeği korunur. Production 1X2 2026/27 lig aşamasında Current AO fallback'e "
            "karşı kilitli pre-match kayıtlarla izlenir.",
            s,
            tone="green",
        ),
        Spacer(1, 0.2 * cm),
    ]
    return out


if __name__ == "__main__":
    main()
