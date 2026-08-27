from __future__ import annotations

from reportlab.lib.units import cm

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


DOCUMENT_DATE = "15 Ağustos 2026"
RATING_VERSION = "ao-european-elo-v2.0-dev-freeze"
PREDICTION_VERSION = "ao-ml-poisson-ensemble-v1-production"

SPEC = PdfSpec(
    filename="AkilOyunu_ML_Tahmin_Katmani_Aciklayici.pdf",
    title="AO ML Tahmin Katmanı",
    subtitle="Structural Logistic + Domestic Poisson Teknik Açıklaması",
    version="AO ML Tahmin Katmanı | Production revizyonu 2026-08-13",
    document_date=DOCUMENT_DATE,
    subject="AO European Elo production ML ve Domestic Poisson 1X2 tahmin katmanı",
)


def main() -> None:
    output_path, docs_path = build_pdf(SPEC, story())
    print(f"PDF written: {output_path}")
    print(f"PDF synced:  {docs_path}")


def story() -> list[object]:
    s = styles()
    out: list[object] = cover(
        SPEC,
        [
            ["Rating sürümü", RATING_VERSION],
            ["Tahmin sürümü", PREDICTION_VERSION],
            ["Aktivasyon", "13 Ağustos 2026"],
            ["Karar", "PROMOTE_WITH_MONITORING"],
            ["Geliştirme dönemi", "2018/19 - 2025/26"],
            ["Prospective izleme", "2026/27 lig aşaması ve sonrası"],
            ["Teknik otorite", "contracts/ao_european_elo_v2_production.json"],
        ],
        s,
        summary_title="Belgenin amacı",
        summary=(
            "Bu belge kullanıcıya sunulan Home/Draw/Away olasılığının nasıl üretildiğini açıklar. "
            "ML ve Domestic Poisson, AO Live Elo'yu değiştirmeyen prediction-only katmanlardır. "
            "Belge mimariyi, feature'ları, matematiği, performansı, fallback'i ve sınırları kapsar."
        ),
    )

    out += [
        h1("1. ML Katmanı Sisteme Ne Sağlar?", s),
        body(
            "AO Live Elo iki takım arasındaki temel güç farkını tek bir denetlenebilir rating "
            "ekseninde taşır. Fakat aynı Elo farkına sahip maçlar turnuva formatı, güncel Avrupa "
            "formu, dinlenme, fikstür yoğunluğu, aggregate durum ve yerel hücum-savunma profili "
            "bakımından farklı olabilir. Prediction katmanı bu bağlamı yalnız maç öncesi 1X2 "
            "olasılığına ekler.",
            s,
        ),
        table(
            [
                ["Katman", "Ana soru", "Ürettiği değer", "Ratinge etkisi"],
                ["Current AO", "Takımların temel güç farkı nedir?", "Base H/D/A", "AO Live çekirdeği"],
                ["Structural Logistic", "Bu maçın bağlamı base olasılığı nasıl düzeltir?", "Ham ML H/D/A", "Yok"],
                ["Domestic Poisson", "Hücum-savunma profili nasıl bir skor dağılımı üretir?", "Skor matrisi ve H/D/A", "Yok"],
                ["Production ensemble", "Üç kanıt nasıl kontrollü birleşir?", "Served H/D/A", "Yok"],
            ],
            [3.15 * cm, 5.15 * cm, 4.35 * cm, 3.8 * cm],
            s,
        ),
        callout(
            "En önemli ayrım",
            "Gol farkı ve uygun xG maç sonucundan sonra Power Elo'yu günceller. Structural "
            "Logistic ve Domestic Poisson ise maçtan önce olasılık üretir ve settlement "
            "sonucunu hiçbir zaman rating state'ine geri yazmaz.",
            s,
            tone="green",
        ),
        h2("1.1 Uçtan uca tahmin akışı", s),
        formula(
            [
                "AO First Elo -> AO Live Elo -> Current AO 1X2",
                "Current AO + Structural Logistic -> Current ML",
                "Current AO + Domestic Poisson -> AO Poisson",
                "Current ML + AO Poisson -> Served 1X2",
                "Kickoff öncesi lock -> maç sonucu -> yalnız rating settlement",
            ],
            s,
        ),
        *bullets(
            [
                "Tahmin sonucu görülmeden önce üretilir ve generated_at_utc < kickoff_utc şartı aranır.",
                "Aynı kickoff UTC batch'indeki maçlar sonuçlar işlenmeden ortak snapshot kullanır.",
                "Kullanıcıya yine tek AO Live Elo ve tek Home/Draw/Away olasılık seti gösterilir.",
            ],
            s,
        ),
        page_break(),
        h1("2. Structural Logistic Modeli", s),
        body(
            "Structural Logistic üç sınıflı, düzenlileştirilmiş bir lojistik regresyondur. "
            "Hedef sınıflar ev sahibi galibiyeti, saha beraberliği ve deplasman galibiyetidir. "
            "Model AO olasılığını yok saymaz; AO'nun log-odds değerlerini ana feature olarak "
            "okur ve bunların üzerine maç bağlamı ekler.",
            s,
        ),
        table(
            [
                ["Özellik", "Aktif değer / davranış"],
                ["Model ailesi", "Multinomial Logistic Regression"],
                ["Düzenlileştirme", "C=0.03, l1_ratio=0.25"],
                ["Feature sayısı", "33 sayısal + 4 kategorik"],
                ["Eksik sayısal", "Yalnız training medyanıyla doldurma + missing indicator"],
                ["Sayısal ölçek", "Training fold içinde StandardScaler"],
                ["Kategorik", "One-hot; görülmemiş kategori ignore/UNKNOWN güvenliği"],
                ["Training penceresi", "2018/19-2025/26 frozen artifact"],
                ["Model fingerprint", "e7c0cc2f44dd692a"],
            ],
            [5.2 * cm, 11.25 * cm],
            s,
        ),
        h2("2.1 Modelin öğrendiği şey", s),
        body(
            "Multinomial model her sınıf için doğrusal bir skor üretir ve softmax ile üç "
            "olasılığa dönüştürür. Güçlü düzenlileştirme, küçük örneklemde tesadüfen görülen "
            "ilişkilerin katsayılarını küçültür. Katsayılar korelasyon gösterir; tek başına "
            "nedensellik veya futbol kuralı olarak yorumlanmaz.",
            s,
        ),
        formula(
            [
                "score_c = intercept_c + sum_j beta_(c,j) * feature_j",
                "P_structural(c) = exp(score_c) / sum_k exp(score_k)",
                "c in {Home, Draw, Away}",
            ],
            s,
        ),
        h2("2.2 Artifact'taki güçlü sinyaller", s),
        table(
            [
                ["Feature / kategori", "Göreli katsayı büyüklüğü", "Ne taşır?"],
                ["AO home/draw log-odds", "0.366", "AO base tahmininin ana anchor'ı"],
                ["Single-match tie", "0.210", "Tek maç eleme formatındaki draw yapısı"],
                ["Initial rating difference", "0.186", "Sezon başı yapısal güç farkı"],
                ["Home European matches pre", "0.148", "Avrupa kanıt hacmi"],
                ["Live rating difference", "0.132", "Maç öncesi güncel güç farkı"],
                ["Away European matches pre", "0.130", "Rakibin Avrupa kanıt hacmi"],
                ["Expected home score", "0.123", "AO beklenen puanı"],
                ["Away matches in 14d", "0.118", "Kısa dönem fikstür yoğunluğu"],
            ],
            [6.0 * cm, 4.3 * cm, 6.15 * cm],
            s,
        ),
        callout(
            "Katsayı uyarısı",
            "Bu büyüklükler standardize edilmiş ve one-hot dönüştürülmüş artifact uzayındadır. "
            "Bir feature'ın sıralamada üstte olması, tek başına maç sonucuna sebep olduğu anlamına gelmez.",
            s,
            tone="amber",
        ),
        page_break(),
        h1("3. Structural Logistic Feature Sözlüğü", s),
        body(
            "Aşağıdaki bütün alanlar kickoff öncesinde üretilebilir. Hedef maçın skoru, xG'si, "
            "tur sonucu veya post-match ratingi schema içinde bulunmaz.",
            s,
        ),
        h2("3.1 AO gücü ve sezon içi state", s),
        table(
            [
                ["Feature", "Açıklama"],
                ["ao_log_home_draw / ao_log_away_draw", "Current AO H/D ve A/D olasılık oranlarının log'u"],
                ["expected_home_score", "AO çekirdeğinin ev sahibi beklenen puanı"],
                ["live_rating_difference", "Maç öncesi AO Live Elo farkı"],
                ["initial_rating_difference", "İki takımın sezon başı AO First Elo farkı"],
                ["exposure_difference", "Effective European Exposure farkı"],
                ["home_live_change / away_live_change", "Sezon başından maça kadar canlı Elo hareketi"],
                ["home/away_euro_matches_pre", "Maçtan önce tamamlanan Avrupa maçı sayısı"],
            ],
            [6.4 * cm, 10.05 * cm],
            s,
        ),
        h2("3.2 Avrupa formu ve gol eğilimi", s),
        table(
            [
                ["Feature", "Açıklama"],
                ["home/away_euro_residual_h3", "Son 3 maçta S-E performans residual'ı"],
                ["home/away_euro_residual_h8", "Son 8 maçta daha uzun form residual'ı"],
                ["home/away_euro_goals_for_5", "Son 5 Avrupa maçında gol üretimi"],
                ["home/away_euro_goals_against_5", "Son 5 Avrupa maçında yenilen gol"],
                ["home_euro_home_residual", "Ev sahibinin geçmiş Avrupa iç saha residual'ı"],
                ["away_euro_away_residual", "Deplasmanın geçmiş Avrupa dış saha residual'ı"],
            ],
            [6.4 * cm, 10.05 * cm],
            s,
        ),
        h2("3.3 Dinlenme ve maç yoğunluğu", s),
        table(
            [
                ["Feature", "Açıklama"],
                ["home/away_days_since_any_match", "Bilinen son yerel veya Avrupa maçından beri geçen gün"],
                ["home/away_matches_14d", "Son 14 günde tamamlanan yerel + Avrupa maçı"],
                ["home/away_matches_30d", "Son 30 günde tamamlanan yerel + Avrupa maçı"],
            ],
            [6.4 * cm, 10.05 * cm],
            s,
        ),
        page_break(),
        h1("4. Eşleşme ve Format Feature'ları", s),
        h2("4.1 Sayısal ve binary alanlar", s),
        table(
            [
                ["Feature", "Açıklama"],
                ["is_single_match_tie", "Fikstür metadata'sına göre tek maçta biten eleme"],
                ["is_neutral", "Nötr saha bayrağı"],
                ["tie_matches_played_pre", "Eşleşmede bu maçtan önce tamamlanan ayak sayısı"],
                ["aggregate_home_lead_pre", "Ev sahibinin maç öncesi bilinen aggregate avantajı"],
                ["leg_number", "Birinci/ikinci maç veya tanımsız"],
                ["round_sequence", "Turun kronolojik sıra göstergesi"],
                ["month", "Takvim ayı; sezon içi yapısal zaman sinyali"],
            ],
            [6.4 * cm, 10.05 * cm],
            s,
        ),
        h2("4.2 Kategorik alanlar", s),
        table(
            [
                ["Feature", "Örnek değer", "Modeldeki amaç"],
                ["competition", "UCL / UEL / UECL", "Turnuva yapısal farkı"],
                ["stage", "QUALIFYING / LEAGUE / KNOCKOUT", "Aşama bağlamı"],
                ["round", "Play-off / R16 / QF / SF / Final", "Turun ayrıntılı kimliği"],
                ["format_type", "SINGLE_MATCH / TWO_LEG / LEAGUE_OR_GROUP", "Maç formatı"],
            ],
            [4.0 * cm, 5.4 * cm, 7.05 * cm],
            s,
        ),
        callout(
            "Leakage koruması",
            "Aggregate fark yalnız önceki tamamlanmış ayaklardan gelir. Aynı kickoff batch'indeki "
            "başka bir maçın sonucu veya hedef maçın sonucu feature üretimine giremez.",
            s,
            tone="green",
        ),
        h2("4.3 Structural modelde kullanılmayan bilgiler", s),
        *bullets(
            [
                "Hedef maçın golü, xG'si, şutu, possession değeri veya kartları.",
                "Maç sonrası AO Live Elo ve Power Delta.",
                "İleride belli olacak advanced_team_id veya eşleşme sonucu.",
                "Domestic Poisson attack/defence feature'ları; bunlar ayrı Poisson bileşeninde kullanılır.",
            ],
            s,
        ),
        page_break(),
        h1("5. Domestic Dynamic Poisson", s),
        body(
            "Poisson bileşeni yerel lig maçlarından takımın ligine göre göreli hücum ve savunma "
            "state'i öğrenir. Amaç liglerin ham gol seviyelerini doğrudan karşılaştırmak değil, "
            "takımın kendi ligi içindeki profilini AO güç sinyaliyle birlikte Avrupa maçına taşımaktır.",
            s,
        ),
        table(
            [
                ["Kapsam / parametre", "Aktif değer"],
                ["Yerel maç", "45.423"],
                ["Lig / source takım", "19 / 508"],
                ["AO'ya güvenle eşleşen kulüp", "171"],
                ["Team learning rate", "0.02"],
                ["Season carry", "0.90"],
                ["Shrinkage matches", "10"],
                ["Lambda sınırı", "0.20 - 4.50"],
                ["Team venue context", "Kapalı"],
            ],
            [7.0 * cm, 9.45 * cm],
            s,
        ),
        h2("5.1 Causal state güncellemesi", s),
        formula(
            [
                "lambda_home_dom = exp(league goal + league home + home attack - away defence)",
                "lambda_away_dom = exp(league goal + away attack - home defence)",
                "goal residual = observed goals - expected goals",
                "team parameter_new = team parameter_old + 0.02 * residual gradient",
                "new season parameter = 0.90 * previous season parameter",
            ],
            s,
        ),
        body(
            "Takım hücum ve savunma değerleri her lig içinde sıfır merkezlenir. Böylece Premier "
            "League ile başka bir ligin ham state'i keyfî biçimde aynı ölçekmiş gibi okunmaz. "
            "Ligden Avrupa'ya ana seviye geçişini AO Elo taşır; Poisson yalnız lig içi profili ekler.",
            s,
        ),
        h2("5.2 Güvenilirlik küçültmesi", s),
        formula("Reliability = Effective Matches / (Effective Matches + 10)", s),
        table(
            [
                ["Effective maç", "Reliability", "Yorum"],
                ["5", "%33", "Profilin üçte biri kullanılır"],
                ["10", "%50", "Orta kanıt"],
                ["20", "%67", "Daha kararlı profil"],
                ["40", "%80", "Yüksek fakat tam olmayan güven"],
            ],
            [4.0 * cm, 4.0 * cm, 8.45 * cm],
            s,
        ),
        page_break(),
        h1("6. Avrupa Maçına Poisson Transferi", s),
        body(
            "Avrupa transferi AO expected score'un logit'ini ana güç sinyali olarak kullanır. "
            "Yerel attack ve defence değerleri reliability ile küçültülmüş lig-relative z-score'lardır.",
            s,
        ),
        formula(
            [
                "z_AO = log(E_home / (1 - E_home))",
                "log(lambda_home) = mu + 0.5*beta*z_AO + kA*A_home - kD*D_away",
                "log(lambda_away) = mu - 0.5*beta*z_AO + kA*A_away - kD*D_home",
                "mu=0.280401; beta=0.666304; kA=0.092097; kD=0.060656",
                "venue coefficient=0; L2=10; Dixon-Coles rho=0",
            ],
            s,
        ),
        body(
            "Lambda değerlerinden 0-0, 1-0, 1-1 ve diğer olası skorların matrisi üretilir. "
            "Ev sahibi galibiyeti skorları Home, eşit skorlar Draw ve deplasman galibiyeti "
            "skorları Away olasılığında toplanır. Rho=0 aktif sürümün bağımsız Poisson kontrolü "
            "kullandığını gösterir.",
            s,
        ),
        h2("6.1 Coverage davranışı", s),
        table(
            [
                ["Coverage", "Davranış"],
                ["BOTH", "İki takımın reliability-adjusted attack/defence profili kullanılır"],
                ["ONE", "Yalnız mevcut takım profili kullanılır; eksik taraf nötrdür"],
                ["NONE", "Ham Poisson H/D/A birebir Current AO olasılığına döner"],
            ],
            [4.0 * cm, 12.45 * cm],
            s,
        ),
        callout(
            "Neden team venue kapalı?",
            "Yerel veride ayrı takım bazlı saha parametresi test edildi fakat production "
            "sözleşmesine güvenilir ek değerle girmedi. Global ev sahibi etkisi Current AO "
            "üzerinden modele zaten ulaşır; Poisson venue coefficient aktif olarak sıfırdır.",
            s,
            tone="amber",
        ),
        h1("7. Log-Probability Ensemble", s),
        formula(
            [
                "LogBlend(P,Q,w)_c = P_c^(1-w) * Q_c^w / normalization",
                "P_current_ml = LogBlend(P_AO, P_structural, 0.90)",
                "P_ao_poisson = LogBlend(P_AO, P_poisson, 0.50)",
                "P_served = LogBlend(P_current_ml, P_ao_poisson, 0.50)",
            ],
            s,
        ),
        body(
            "İç içe log blend açıldığında coverage bulunan normal bir satırda efektif geometrik "
            "üsler AO %30, Structural Logistic %45 ve ham Poisson %25 olur. Bunlar aritmetik "
            "oy ağırlığı değil, log-olasılık katkılarıdır.",
            s,
        ),
        page_break(),
        h1("8. Sayısal Tahmin Örneği", s),
        body(
            "Aşağıdaki örnek yalnız log-blend davranışını göstermek içindir. Yüzdeler gerçek bir "
            "maça ait değildir.",
            s,
        ),
        table(
            [
                ["Tahmin", "Home", "Draw", "Away"],
                ["Current AO", "%50.0", "%25.0", "%25.0"],
                ["Structural Logistic", "%45.0", "%30.0", "%25.0"],
                ["Domestic Poisson", "%55.0", "%23.0", "%22.0"],
                ["Current ML ara kolu", "%45.5", "%29.5", "%25.0"],
                ["AO Poisson ara kolu", "%52.5", "%24.0", "%23.5"],
                ["Served 1X2", "%49.0", "%26.7", "%24.3"],
            ],
            [6.1 * cm, 3.45 * cm, 3.45 * cm, 3.45 * cm],
            s,
        ),
        body(
            "Structural model beraberlik yönünde, Poisson ise ev sahibi yönünde kanıt vermiştir. "
            "Final tahmin iki challenger'ı ve AO anchor'ını birlikte taşır. Böylece tek bir "
            "modelin aşırı güvenli olasılığı doğrudan kullanıcıya sunulmaz.",
            s,
        ),
        h2("8.1 Prediction ile rating settlement ayrımı", s),
        table(
            [
                ["Olay", "Served 1X2", "AO Live Elo"],
                ["Pre-match Structural ML", "Değiştirir", "Değiştirmez"],
                ["Pre-match Domestic Poisson", "Değiştirir", "Değiştirmez"],
                ["Maç sonucu", "Artık locked kayıt", "Power Delta üretir"],
                ["Gol farkı ve uygun xG", "Hedef maç tahminine giremez", "Settlement delta'sını kontrollü değiştirir"],
                ["Progression olayı", "Yalnız sonraki maç tahminine rating üzerinden ulaşır", "Sezonluk Live bonus ekler"],
            ],
            [4.7 * cm, 5.9 * cm, 5.85 * cm],
            s,
        ),
        callout(
            "Kısa yorum",
            "ML sistemi 'takıma bonus Elo vermek' için değil, mevcut güç bilgisinden daha iyi "
            "olasılık üretmek için kullanılır. Bu tasarım rating açıklanabilirliğini korur.",
            s,
            tone="blue",
        ),
        page_break(),
        h1("9. Walk-Forward Performansı", s),
        body(
            "Model geliştirme penceresinde 6.340 Avrupa maçı kullanılmış, 2020/21-2025/26 "
            "arasındaki altı unseen fold içinde 4.884 maç değerlendirilmiştir. Her test sezonunun "
            "model ve ağırlık seçimi yalnız daha eski sezonlardan yapılmıştır.",
            s,
        ),
        table(
            [
                ["Model", "Brier", "Log-loss", "Accuracy", "AO'ya Brier farkı"],
                ["Current AO", "0.566413", "0.956259", "%55.917", "0"],
                ["Current ML", "0.562524", "0.950903", "%56.061", "-0.003889"],
                ["AO Poisson rho=0", "0.564087", "0.952790", "%56.102", "-0.002325"],
                ["Final ML + Poisson", "0.561935", "0.949792", "%56.143", "-0.004478"],
            ],
            [4.6 * cm, 2.8 * cm, 3.1 * cm, 2.85 * cm, 3.1 * cm],
            s,
        ),
        h2("9.1 Final ensemble fold sonuçları", s),
        table(
            [
                ["Test sezonu", "Maç", "Brier", "Log-loss", "Accuracy", "Current ML'ye loss yönü"],
                ["2020/21", "540", "0.531981", "0.903835", "%59.444", "İyi / iyi"],
                ["2021/22", "816", "0.572171", "0.966765", "%55.147", "İyi / iyi"],
                ["2022/23", "804", "0.581384", "0.977980", "%52.488", "Kötü / kötü"],
                ["2023/24", "806", "0.556752", "0.943075", "%56.452", "İyi / iyi"],
                ["2024/25", "957", "0.568649", "0.961743", "%55.904", "İyi / iyi"],
                ["2025/26", "961", "0.582762", "0.979379", "%54.318", "Kötü / kötü"],
            ],
            [3.2 * cm, 2.0 * cm, 2.65 * cm, 2.8 * cm, 2.7 * cm, 3.1 * cm],
            s,
        ),
        h2("9.2 Kalibrasyon", s),
        table(
            [
                ["Model", "ECE", "Calibration slope", "Mean max probability"],
                ["Current AO", "0.009766", "0.927855", "0.558511"],
                ["Current ML", "0.015590", "0.966811", "0.550279"],
                ["Final ensemble", "0.013066", "0.975041", "0.551092"],
            ],
            [4.7 * cm, 3.4 * cm, 4.1 * cm, 4.25 * cm],
            s,
        ),
        body(
            "Calibration slope'un 1'e yaklaşması olasılık yayılımının daha dengeli olduğunu "
            "gösterir. Buna karşılık ECE Current AO'dan düşük değildir; dolayısıyla bütün "
            "kalibrasyon ölçülerinin aynı anda iyileştiği iddia edilmez.",
            s,
        ),
        page_break(),
        h1("10. Karar, Sınırlamalar ve Monitoring", s),
        callout(
            "Production durumu",
            "Tarihsel otomatik gate sonucu KEEP_SHADOW'dur. Pooled loss, 4/6 fold sonucu ve "
            "kalibrasyon sinyali üzerine açık ürün kararıyla PROMOTE_WITH_MONITORING olarak "
            "aktive edilmiştir. Bu ayrım bilimsel raporda saklanır.",
            s,
            tone="amber",
        ),
        h2("10.1 Neden monitoring zorunlu?", s),
        *bullets(
            [
                "Final ensemble Current ML'ye karşı Brier ve log-loss'ta 4/6 fold kazanmıştır; iki fold gerilemiştir.",
                "Dependency uncertainty gate tarihsel değerlendirmede geçmemiştir.",
                "UECL ve yerel geçmişi olmayan coverage segmentinde Current ML'ye karşı küçük ters yönler vardır.",
                "2018/19-2025/26 tekrar kullanılan geliştirme penceresidir; bağımsız prospective kanıt değildir.",
                "2026/27 lig aşaması ve sonrası her tahmin AO fallback ile yan yana kilitli olarak izlenir.",
            ],
            s,
        ),
        h2("10.2 Runtime güvenlik sözleşmesi", s),
        table(
            [
                ["Kontrol", "Davranış"],
                ["Artifact checksum", "ML model, feature schema ve state SHA-256 doğrulanır"],
                ["Startup hatası", "Strict modda servis durur; degraded mod yalnız açık fallback ile çalışır"],
                ["Satır feature/state hatası", "Served 1X2 birebir Current AO olur"],
                ["Olasılık invariantı", "Finite, negatif olmayan ve toplamı 1"],
                ["Prediction ledger", "Ara olasılıklar, hash'ler, coverage ve fallback nedeni loglanır"],
                ["Rating feedback", "Her satırda false olmak zorundadır"],
            ],
            [5.0 * cm, 11.45 * cm],
            s,
        ),
        h2("10.3 Modelin yapmadıkları", s),
        *bullets(
            [
                "ML bir takımın AO First veya AO Live Elo'suna doğrudan puan eklemez.",
                "Hedef maçın xG'sini pre-match feature olarak kullanmaz.",
                "Kesin skor, O/U 2.5 veya BTTS production ürünü olarak ilan edilmemiştir.",
                "Takım bazlı dinamik ev sahibi avantajı aktif değildir.",
                "Tahmin başarıları otomatik olarak rating formülüne taşınmaz.",
            ],
            s,
        ),
        page_break(),
        h1("11. Toplantı İçin Hazır Anlatım", s),
        body(
            "AO Elo takımın temel gücünü hesaplıyor. Structural Logistic maçın formatını, Avrupa "
            "formunu, dinlenme ve yoğunluk koşullarını kullanarak AO olasılığını kalibre ediyor. "
            "Domestic Poisson ise 45.423 yerel lig maçından öğrenilen lig-relative hücum ve "
            "savunma profilleriyle gol dağılımı oluşturuyor. Bu üç bilgi log-probability "
            "uzayında birleşiyor. ML ve Poisson ratingi değiştirmiyor; yalnız kullanıcıya "
            "gösterilen Home/Draw/Away yüzdelerini iyileştiriyor. Unseen backtestte Brier "
            "0.5721'den 0.5681'e, log-loss 0.9644'ten 0.9592'ye indi. Katman 2026/27'de "
            "Current AO fallback ile birlikte izleniyor.",
            s,
        ),
        h2("11.1 Sık sorulan sorular", s),
        table(
            [
                ["Soru", "Kısa cevap"],
                ["ML Elo puanı veriyor mu?", "Hayır. Yalnız 1X2 olasılığını değiştirir."],
                ["xG ML feature'ı mı?", "Hedef maç xG'si hayır; xG maç sonrası rating settlement katmanındadır."],
                ["Poisson neden gerekli?", "Elo güç farkını, Poisson gol üretim yapısını temsil eder."],
                ["Neden yalnız ML kullanılmıyor?", "AO anchor açıklanabilirlik ve fallback güvenliği sağlar."],
                ["%50/%50 ne demek?", "Current ML ve AO Poisson kolları log-probability uzayında eşit karışır."],
                ["Model kesin kazanan söyler mi?", "Hayır. Üç sonucun olasılığını üretir; en yüksek sınıf accuracy için kullanılır."],
                ["55.38 accuracy yeterli mi?", "Tek başına karar değildir; Brier, log-loss ve kalibrasyon birlikte değerlendirilir."],
                ["Artifact bozulursa ne olur?", "Tahmin Current AO 1X2'ye düşer ve neden loglanır."],
            ],
            [6.0 * cm, 10.45 * cm],
            s,
        ),
        callout(
            "Tek cümlelik sonuç",
            "ML katmanı Elo'nun yerine geçmez; Elo'nun güç sinyalini maç bağlamı ve yerel "
            "hücum-savunma kanıtıyla daha iyi kalibre edilmiş 1X2 olasılığına dönüştürür.",
            s,
            tone="green",
        ),
        h1("12. Otorite ve Yeniden Üretim", s),
        table(
            [
                ["Kaynak", "Rol"],
                ["contracts/ao_european_elo_v2_production.json", "Aktif production parametre otoritesi"],
                ["src/ao_elo/production_prediction.py", "Inference, blend, audit ve fallback"],
                ["src/ao_elo/ml_features.py", "Structural feature schema ve causal feature üretimi"],
                ["src/ao_elo/domestic_poisson.py", "Yerel state, reliability ve Avrupa transferi"],
                ["artifacts/production_prediction/manifest.json", "Frozen artifact/state checksum sözleşmesi"],
                ["output/final_prediction_ensemble_backtest_2018_2026", "Walk-forward kanıt çıktıları"],
            ],
            [7.0 * cm, 9.45 * cm],
            s,
        ),
    ]
    return out


if __name__ == "__main__":
    main()
