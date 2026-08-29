# Production hata düzeltmeleri — 28 Ağustos 2026

Revision: `2026-08-28-causal-state-and-input-validation-fixes`.
Model: `ao-european-elo-v2.0-dev-freeze`.
Önceki revision: `2026-08-27-participation-cup-and-unknown-position`.
Başlangıç commit'i: `49dd94b`. Donmuş parametreler ve aktif katmanlar değişmedi.

## Kapatılan bulgular

| Bulgu | Önceki somut hata | Uygulanan düzeltme ve regresyon sonucu |
| --- | --- | --- |
| F01 / P1 | Eylül 1 fikstürü Eylül 2 domestic sonucuyla ACTIVE_ENSEMBLE olabiliyordu | İlgili ligin cutoff'u üretim zamanını aşarsa veya fixture kickoff'una eşit/ileriyse Current AO fallback. Tahmin state'i değiştirmiyor. |
| F02 / P1 | Aynı event iki kez, sonra daha eski kickoff uygulanabiliyordu | Kalıcı processed_event_ids ve lig başına last_kickoff_utc. Duplicate/eski/bölünmüş batch ret; restore sonrasında da geçerli. Gecersiz son satırda önceki satır state'i değişmiyor. |
| F03 / P2 | Aynı kickoff'taki A-B sonucu A-C pre-rating'ine giriyordu | Takımın aynı kickoff'ta ikinci settlement'ı reddediliyor. Bağımsız takımların eşzamanlı kilitleri korunuyor. |
| F04 / P2 | Aynı metadata yeniden verilince R16/leg=2, UNKNOWN/leg=0 oluyordu | Ortak alanlar eşitse korunuyor, eksik taraf tamamlanıyor, çelişkide ValueError. |
| F05 / P2 | String false, ML'de 1/SINGLE_MATCH oluyordu | Üç format/venue flag'i ortak boolean dönüşümünden geçiyor. false ve 0 eşdeğer; geçersiz flag ret. |
| F06 / P2 | Tek maçlık A-B 3-0 finalinde B'ye +12 verilebiliyordu | Shootout olmayan tek maçta advancer field-score galibiyle uyuşmalı. İki ayaklı aggregate istisnası korunuyor. |
| F07 / P2 | Timezone'suz 19:00'a sessizce UTC atanıyordu | CSV fixture/state, prediction, ML feature ve domestic girişinde açık timezone zorunlu. +03:00 offset UTC'ye normalize ediliyor. |
| F08 / P2 | 0.999999 gol toleransla kabul edilip 0'a kesiliyordu | Finite, negatif olmayan tam integer zorunlu; 0.999999 ve 1.000001 ret, 1.0 geçerli. |
| F09 / P2 | Eksik contract, degraded modda bile FileNotFoundError üretiyordu | Dosya hash okuması korunan yükleme bloğunda. İzinli degraded mod Current AO ve UNAVAILABLE hash döndürüyor; strict mod hata veriyor. |
| F10 / P2 | Validator'ın kabul ettiği played=true metni float dönüşümünde çöküyordu | Exposure hesabında canonical boolean dönüşümü. 32 participation düzeninde eski sayısal toplama sırası birebir korunuyor. |

Regresyonlar mevcut test dosyalarına eklendi: `test_domestic_poisson.py`,
`test_production_prediction.py`, `test_dynamic_engine.py`, `test_dynamic_csv.py`,
`test_ml_prediction_layer.py`, `test_validation.py`, `test_exposure.py`.
İlk ayağı görülmemiş tie davranışı bu düzeltmeye dahil edilmedi.

## Ölçülen doğrulamalar

- Tam suite: **1175 passed** (önceki 1135 + 40 regresyon), 1 joblib fiziksel CPU sayısı uyarısı, 153.51 saniye.
- Odaklı suite: **185 passed**; son kaynak kontrolünde domestic/prediction/exposure **70 passed**.
- Yeniden üretilen mevcut güvenlik denetimi: **23/23 passed**.
- `ruff check src/`: **27 baseline bulgusu**, değişen production modüllerinde bulgu yok. Tam kaynak lint'i geçti denmiyor.
- Bağımsız formül kontrolleri: AO First 48 senaryo, achievement 368,
  participation 160, 1X2 2002, Match Power 400 maç. En büyük maç sıfır-toplam
  hatası `4.547473508864641e-13 < 1e-9`; 1X2 toplamı ve beklenen skor özdeşliği
  hatası `1.1102230246251565e-16`. Served blend bağımsız hesap farkı
  `5.551115123125783e-17`. Prediction state mutasyonu görülmedi.
- Sonuçlar sentetik regresyonlar ve tarihsel replay içindir; prospective veri
  veya yeni performans üstünlüğü kanıtı değildir.

## Checkpoint geçişi

Eski schema `1.0` tahmini cutoff eklenerek migrate edilmez; kaynak sonuçlar
aynı donmuş config ile yeniden replay edilir. Schema `2.0` tarihsel checkpoint
45.423 event / 19 lig, güncel checkpoint 75.496 event / 34 lig içerir.
Her iki checkpoint'te tüm sayısal lig/takım alanları önceki artifact ile
**birebir aynı**. Eklenenler cutoff ve işlenmiş event ID'leridir.
Production coverage **311** kulüp; Torreense gate dışında kalır.

Yeniden eğitilen ML modelinin katsayıları ve intercept'leri eski artifact ile
birebir aynı; 6.340 feature satırında olasılık farkı sıfır. Son timestamp
doğrulamasıyla yeniden üretilen feature CSV'si de eğitim girdisiyle byte
olarak aynı. Model çıktısı değişmese de kaynak artifact/provenance kimliği
değiştiği için ML paketinin SHA-256 değeri yenilendi.

Bu şema değişikliği eski checkpoint kullanan deployment için uyumsuzdur:
kod ve yenilenmiş artifact/manifest/contract birlikte dağıtılmalıdır.

## Tarihsel ölçüm sınırı

Geliştirme penceresi 2018/19–2025/26; altı test fold'u 2020/21–2025/26,
4.884 maç. `model_summary.csv` önceki ölçümle birebir aynı kaldı.
AO Brier `0.5664125273`, log-loss `0.9562592472`, accuracy `0.5591728`.
Referans Brier `0.5680525`, log-loss `0.9591739`.
Referansa göre Brier/log-loss farkları `-0.0016400 / -0.0029146`; fold kazanımı
6/6. Bağımlılık envelope %95 CI: Brier `[-0.0029053, -0.0005381]`,
log-loss `[-0.0052703, -0.0009461]`. UCL/UEL/UECL detayları güncel
`reports/current_model/competition_summary.csv` içindedir.

Tarihsel nested ensemble Brier `0.5619348711`, log-loss `0.9497915175`.
Bu, sabit production %50/%50 karışımının birebir replay ölçümü değildir.
Güncellenen PDF'lerde de bu ayrım ve güncel fold/kalibrasyon sayıları korunur.

## Propagasyon durumu

RUNBOOK §8.3'teki dokuz komut sırasıyla tamamlandı:

1. Domestic Surprise variance backtest.
2. Gamma sensitivity: `--variance-penalty 0.5`.
3. Current model evaluation: `--bootstrap-samples 4000`.
4. ML: `--blend-weight 0.9 --bootstrap-samples 4000`.
5. Domestic Poisson: `--bootstrap-samples 4000`.
6. Ensemble: `--prospective-poisson-weight 0.5 --bootstrap-samples 4000`.
7. Current external benchmark.
8. 2026/27 preproduction replay: 342 maç, 237 takım.
9. Play-off ilk ayak ranking replay: 342 Q1-Q3 + 43 play-off = 385 maç, 237 takım.

Domestic Poisson adımı mevcut 54-candidate domestic prequential surface'i
kullandı; Avrupa transfer/feature/replay çıktıları yeniden üretildi. Aktif
checkpoint'ler ayrıca kaynak maçlardan baştan üretildi. Coverage audit bayrağı
korundu. Production ve final-candidate contract yeni manifest hash'ine bağlandı.
Son hash güncellemesinden sonra current evaluation/snapshot tekrar üretildi;
curated kopyaları contract ile doğrulandı. Son paket kontrolü **70 passed**.

Güncel manifest SHA-256:
`8771f7a53ddd9a1c05e01135a00c25520382f135a9826dd75001d5c2df532220`.
Komut logları `output/production_bugfix_2026_08_28/01_...09_...` altındadır.

## Değişmeyen operasyon sınırı

Bu değişiklik ingestion servisi, prediction lock queue veya append-only
prospective ledger eklemez. Cutoff, sonucun provider'dan edinilme zamanını
kanıtlamaz. Aynı state'e yazma işleri dış servis tarafından sıralanmalıdır.
Fallback rate ile rows_with_imputed_model_input ayrı monitoring işleri olarak
kalır; normal ACTIVE_ENSEMBLE satırında imputation mümkün olmaya devam eder.

Ham kanıt: `output/production_bugfix_2026_08_28/` altındaki test logları,
`invariants.json`, `checkpoint_comparison.json`, `ruff.json` ve `pdf_review/`.
Önceki audit'in bug üreten script'i ve `evidence.json` dosyası değiştirilmedi.
