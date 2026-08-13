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


SPEC = PdfSpec(
    filename="AkilOyunu_Elo_Onaylanan_Gelistirme_Plani.pdf",
    title="AO European Elo",
    subtitle="Güncel Geliştirme ve Production İzleme Planı",
    version="AO European Elo v2 | Geliştirme planı revizyonu 2026-08-13",
    document_date="13 Ağustos 2026",
    subject="Aktif AO European Elo modelinin production, izleme ve araştırma yol haritası",
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
            ["Rating çekirdeği", "KEEP - AO European Elo v2"],
            ["Tahmin katmanı", "PROMOTE_WITH_MONITORING"],
            ["Sunulan tahmin", "%50 Current ML + %50 AO Domestic Poisson rho=0"],
            ["Rating feedback", "Kapalı"],
            ["Fallback", "CURRENT_AO_1X2"],
            ["İzleme sezonu", "2026/27 lig aşaması ve sonrası"],
        ],
        s,
        summary_title="Planın yeni konumu",
        summary=(
            "Bu belge artık eklenmesi düşünülen eski özellikleri değil, tamamlanan modeli canlıya "
            "alma ve ölçme planını tanımlar. xG production'da aktiftir; Dynamic K ve Achievement "
            "Reserve reddedilmiş veya kapalıdır. Ana görev yeni özellik eklemek değil, dondurulmuş "
            "modeli güvenilir veri, fallback ve prospective monitoring ile işletmektir."
        ),
    )

    out += [
        h1("1. Bugünkü Production Modeli", s),
        table(
            [
                ["Katman", "Durum", "Aktif davranış"],
                ["AO First Elo v2", "ACTIVE", "Country + domestic + Europe + exposure + surprise"],
                ["Dynamic Power", "ACTIVE", "Scale/H/K sabit, carry 0, sıfır toplam"],
                ["Goal difference", "ACTIVE", "alpha 0.15, tau 300, cap 4"],
                ["Bounded xG", "ACTIVE", "ratio 0.30, scale 1.25, eksikte GD fallback"],
                ["Progression", "ACTIVE", "R16 sonrası 12/8/4; caps 48/32/16"],
                ["Prediction ensemble", "ACTIVE + MONITORING", "%50 Current ML + %50 AO Poisson rho=0"],
            ],
            [4.0 * cm, 3.6 * cm, 8.85 * cm],
            s,
        ),
        formula(
            [
                "AO Live Elo = Power Elo + Progression Bonus",
                "P_served = LogBlend(P_current_ml, P_ao_poisson_rho0, 0.50)",
                "rating_feedback_applied = false",
            ],
            s,
        ),
        callout(
            "Ürün kararı",
            "Production tahmini yalnız olasılık katmanıdır. ML veya Poisson hatası takım ratingini "
            "bozamaz; satır Current AO 1X2'ye düşer.",
            s,
            tone="green",
        ),
        h1("2. Tamamlanan Geliştirmeler", s),
        table(
            [
                ["Çalışma", "Sonuç", "Production etkisi"],
                ["500-2000 ölçeği", "Tamamlandı", "AO First ve Dynamic parametreleri aynı affine ölçekte"],
                ["Domestic Surprise", "Aktif", "5 sezon, variance kontrollü, +/-30 cap"],
                ["Tek maç beraberlik", "Aktif", "draw_at_even 0.12; diğer maçlarda 0.24"],
                ["Kontrollü gol farkı", "Aktif", "Denk maçta farklı galibiyeti kontrollü tanır"],
                ["xG performansı", "Aktif", "Sonucu silmeden kazanımı azaltır/güçlendirir"],
                ["KPO progression düzeltmesi", "Tamamlandı", "KPO eligible setten çıkarıldı"],
                ["ML + Poisson ensemble", "Aktif izleme", "1X2 loss metriklerini düşürdü"],
                ["Runtime artifact/fallback", "Tamamlandı", "Checksum, strict load ve satır fallback'i"],
            ],
            [4.5 * cm, 3.1 * cm, 8.85 * cm],
            s,
        ),
        page_break(),
        h1("3. Neden Model Şimdi Dondurulmalı?", s),
        body(
            "Model geliştirme döneminde çok sayıda feature ailesi test edilmiştir. Yeni karmaşıklık "
            "yalnızca açıklamayı zorlaştırmakla kalmaz; aynı tarihsel veri üzerinde tekrar tekrar "
            "seçim yapmak araştırmacı serbestliğini ve overfitting riskini artırır.",
            s,
        ),
        *bullets(
            [
                "Rating çekirdeği reference modele karşı pooled Brier, log-loss ve ranking yönünde daha iyidir.",
                "Prediction ensemble Current AO'ya karşı Brier -0.003999 ve log-loss -0.005129 üretmiştir.",
                "Production davranışını yeniden seçmek için 2026/27 sonucunu beklemeden parametre değiştirmek holdout'u bozar.",
                "Canlı kullanımda en yüksek değer artık yeni feature değil; veri zamanlaması, kimlik doğruluğu ve fallback görünürlüğüdür.",
            ],
            s,
        ),
        callout(
            "Freeze kuralı",
            "2026/27 prospective pencere sırasında Scale, H, K, draw, goal alpha, xG ratio, "
            "progression veya ensemble ağırlıkları değiştirilemez. Acil bug düzeltmesi gerekiyorsa "
            "yeni revision ve açık migration kaydı oluşturulur.",
            s,
            tone="amber",
        ),
        h1("4. Production'a Alma İş Paketleri", s),
        table(
            [
                ["Paket", "Teslim", "Kabul ölçütü"],
                ["A. Sezon başlangıcı", "Dört static CSV + AO First snapshot", "Duplicate yok, tüm takımlar ve açık zero-history satırı"],
                ["B. Live state", "ratings_state + checkpoint", "Config fingerprint uyumu ve exact UTC chronology"],
                ["C. Prediction lock", "Kickoff öncesi append-only ledger", "generated_at < kickoff, normalize olasılık"],
                ["D. Settlement", "Field score, xG, tie sonucu", "Power zero-sum, progression tek uygulama"],
                ["E. Ensemble runtime", "ML artifact + Poisson state", "SHA-256 doğrulama ve AO fallback"],
                ["F. Monitoring", "AO vs served dashboard", "Loss, calibration, coverage ve fallback segmentleri"],
            ],
            [3.3 * cm, 6.2 * cm, 6.95 * cm],
            s,
        ),
        h2("4.1 Sezon başı veri kabulü", s),
        *bullets(
            [
                "team_id kalıcıdır; provider ID değişse bile AO kimliği değişmez.",
                "Ülke ve kulüp puanları hedef sezon başlamadan önceki tamamlanmış sezonlardan gelir.",
                "Domestic Surprise geçmişinde beş tam sezon yoksa adjustment otomatik sıfırdır.",
                "Initial rating dağılımı, exposure bandı ve saturation audit edilmeden yayın yapılmaz.",
            ],
            s,
        ),
        h2("4.2 Maç günü işlem sırası", s),
        formula(
            [
                "Fixture doğrula -> AO state'i oku -> Current AO 1X2 kilitle",
                "-> ML + Poisson production tahminini kilitle -> kickoff",
                "-> 90/120 field score ve uygun xG ile settle",
                "-> tie tamamlandıysa progression uygula -> checkpoint yaz",
            ],
            s,
        ),
        page_break(),
        h1("5. 2026/27 Monitoring Planı", s),
        body(
            "Prospective karşılaştırma aynı locked fixture üzerinde Current AO ve production "
            "ensemble olasılıklarını birlikte saklar. Sonradan yeniden üretilen replay satırları "
            "prospective kanıt sayılmaz.",
            s,
        ),
        table(
            [
                ["Metrik", "Raporlama", "Alarm / yorum"],
                ["Brier ve log-loss", "Pooled, sezon, UCL/UEL/UECL", "Served vs AO farkı"],
                ["Calibration", "Reliability bandı, slope/intercept, ECE", "Aşırı güven veya çekingenlik"],
                ["Accuracy", "İkincil metrik", "Loss'a rağmen yön kaybı var mı?"],
                ["Fallback rate", "Toplam ve hata nedeni", "Artifact/feature pipeline sorunu"],
                ["Poisson coverage", "BOTH / ONE / NONE", "Yerel state erişimi"],
                ["Rating stability", "Max delta, std, rank movement", "Inflation veya aşırı hareket"],
                ["Data quality", "Kimlik, xG scope, chronology", "Sessiz veri bozulması"],
            ],
            [4.3 * cm, 6.0 * cm, 6.15 * cm],
            s,
        ),
        h2("5.1 Değerlendirme takvimi", s),
        *bullets(
            [
                "Her maç: prediction ve fallback audit kaydı.",
                "Aylık: veri kalitesi, coverage, calibration ve loss diagnostikleri.",
                "Lig aşaması sonu: ilk anlamlı ara rapor; parametre değişikliği yapılmaz.",
                "Sezon sonu: tam prospective karar ve gerekirse v2.1 araştırma planı.",
            ],
            s,
        ),
        h1("6. Terfi, Koruma ve Geri Alma Kuralları", s),
        table(
            [
                ["Karar", "Koşul"],
                ["KEEP", "Rating güvenlik invariantları geçiyor ve prospective güvenilir zarar yok"],
                ["KEEP ENSEMBLE", "Served Brier/log-loss AO'dan iyi, calibration ve segmentler kabul edilebilir"],
                ["RECALIBRATE", "Loss iyi fakat sistematik calibration sapması var; ratinge dokunmadan fit araştırılır"],
                ["FALLBACK", "Artifact, schema, state veya feature hatası; Current AO sunulur"],
                ["ROLLBACK", "Normalize olasılık, leakage, identity veya rating-state güvenliği ihlali"],
            ],
            [4.2 * cm, 12.25 * cm],
            s,
        ),
        callout(
            "Rollback ilkesi",
            "Tahmin ensemble'ı ratingden ayrıldığı için rollback, AO Live state'ini geri almadan "
            "Current AO 1X2'ye dönmekle yapılabilir. Bu, operasyonel riskin ana güvencesidir.",
            s,
            tone="green",
        ),
        page_break(),
        h1("7. Kapalı ve Shadow Araştırmalar", s),
        table(
            [
                ["Hipotez", "Durum", "Yeniden açma koşulu"],
                ["Dynamic K", "REJECTED", "Yeni, daha geniş ve güvenilir uncertainty verisi"],
                ["Competition K", "DISABLED", "Turnuva etkisinin opponent strength'ten bağımsız kanıtı"],
                ["Achievement Reserve", "DISABLED", "Gerçek calibrated P_advance ve incremental gain"],
                ["Team venue context", "KEEP_SHADOW", "Daha uzun history + closed-door/attendance metadata"],
                ["Draw shape 0.84", "KEEP_SHADOW", "Yeni formatta daha büyük prospective örneklem"],
                ["Opponent profiles", "DIAGNOSTIC", "Yerel lig destekli, sezonlar arası persistence"],
                ["Domestic Surprise MOB", "DIAGNOSTIC", "Tekrarlanan stabil split ve segment no-harm"],
                ["Stage-weighted bonus", "REJECTED", "Fixed progression'a karşı net ranking/loss üstünlüğü"],
            ],
            [4.0 * cm, 3.5 * cm, 8.95 * cm],
            s,
        ),
        body(
            "Bu modüllerin repository'de kalması production'da aktif oldukları anlamına gelmez. "
            "Aktiflik yalnız production contract, active constructor ve regression testleri aynı "
            "davranışı gösterdiğinde kabul edilir.",
            s,
        ),
        h1("8. 2026/27 Sonrası Araştırma Önceliği", s),
        table(
            [
                ["Öncelik", "Araştırma", "Neden"],
                ["1", "Ensemble prospective değerlendirme", "Mevcut deployment kararını doğrular"],
                ["2", "Calibration-only düzeltme", "Ratingi bozmadan olasılığı iyileştirebilir"],
                ["3", "Takım/lig hiyerarşik home effect", "Global H için açıklanabilir alternatif"],
                ["4", "Tek sağlayıcılı çok sezon xG", "Aktif xG katkısını daha temiz ölçer"],
                ["5", "Progression keep/remove ablation", "Loss katkısı küçük; ranking değeri izlenmeli"],
            ],
            [2.2 * cm, 6.5 * cm, 7.75 * cm],
            s,
        ),
        h1("9. Sorumluluk ve Belge Otoritesi", s),
        *bullets(
            [
                "Production parametre otoritesi: contracts/ao_european_elo_v2_production.json.",
                "Runtime artifact otoritesi: artifacts/production_prediction/manifest.json.",
                "Veri sözleşmesi: docs/ai/DATA_CONTRACTS.md.",
                "Canlı ingest ve settlement: docs/ai/LIVE_DATA_INGESTION.md.",
                "Araştırma statüsü: docs/ai/RESEARCH_STATUS.md.",
                "Evaluation özetleri: reports/current_model ve reports/production_prediction.",
            ],
            s,
        ),
        callout(
            "Nihai yön",
            "Model yeni özellik eklemek için sürekli açılmayacaktır. Önce dondurulmuş sistem canlı "
            "veriyle ölçülecek; sonraki revizyon yalnız prospective kanıt, açık ablation ve sürüm "
            "değişikliğiyle yapılacaktır.",
            s,
            tone="blue",
        ),
    ]
    return out


if __name__ == "__main__":
    main()
