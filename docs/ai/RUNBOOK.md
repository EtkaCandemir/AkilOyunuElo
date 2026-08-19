# Gelistirme ve Isletim Runbook'u

Bu belge repository'yi guvenli bicimde calistirma, test etme ve model degisikligi
yapma adimlarini tanimlar.

## 1. Ortam

Repository root:

```bash
cd "/Users/buycell/Desktop/Akil Oyunu Elo"
```

Python import yolu `pyproject.toml` tarafindan testlerde `src` olarak ayarlanir.
Runtime bagimliliklari `requirements.txt` icindedir.

Yeni ortam gerekiyorsa:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Secret/API key repository'ye yazilmaz. Environment veya sistem keychain
kullanilir.

## 2. Ilk Kontrol

Her calisma basinda:

```bash
git status --short
python3 --version
python3 -m pytest -q
```

Dirty worktree normal olabilir. Ilgisiz kullanici degisiklikleri geri alinmaz.

## 3. Active Contract'i Kontrol Etme

```bash
python3 -m json.tool contracts/ao_european_elo_v2_production.json
python3 -m json.tool reports/current_model/active_model_snapshot.json
```

Kontrol edilecek ortak alanlar:

```text
model_version
production_revision
domestic_surprise
dynamic_core
qualification_transition
one_x_two_probability
goal_margin
xg_performance
progression_bonus
achievement_reserve
competition_k
prediction_layer
```

Snapshot bayat olabilir; celiskide production contract ve kaynak kod ustundur.

## 4. Test Komutlari

Tam suite:

```bash
python3 -m pytest -q
```

Static model:

```bash
python3 -m pytest -q tests/test_pipeline.py tests/test_features.py tests/test_validators.py
```

Dynamic model:

```bash
python3 -m pytest -q tests/test_dynamic_engine.py tests/test_dynamic_live.py tests/test_dynamic_csv.py
```

GD/xG/progression:

```bash
python3 -m pytest -q tests/test_controlled_live.py tests/test_xg_live.py tests/test_tournament_bonus.py
```

Contract ve evaluation:

```bash
python3 -m pytest -q tests/test_model_contract.py tests/test_final_candidate_contract.py tests/test_evaluation.py
```

Dosya adlari degisebilecegi icin hedefli komuttan once `rg --files tests`
ile mevcut testler kontrol edilmelidir.

## 5. Pilotlar

V2 static ve dynamic pilotlar:

```bash
python3 scripts/run_v2_pilots.py
```

20 takimlik aciklayici pilot:

```bash
python3 scripts/run_pilot_20_teams.py
```

Not: `run_pilot_10_teams.py` frozen v1.1 regression pilotudur; aktif v2 sonucu
olarak raporlanmamalidir.

## 6. Current Production Evaluation

Tam current evaluation:

```bash
python3 scripts/run_current_model_evaluation.py --bootstrap-samples 4000
```

Ana output:

```text
output/current_model_evaluation_2018_2026/
```

Shareable curated raporlar:

```text
reports/current_model/current_model_evaluation_report.md
reports/current_model/model_summary.csv
reports/current_model/fold_summary.csv
reports/current_model/competition_summary.csv
reports/current_model/dependency_uncertainty.csv
reports/current_model/safety_and_data_quality_audit.csv
```

Evaluation scripti production contract'i mutate etmemelidir.

## 7. Initial Elo External Validation

Cache ve frozen Opta snapshot hazirsa:

```bash
python3 scripts/run_initial_elo_external_comparison_2025_26.py --bootstrap-samples 4000
```

Output:

```text
output/initial_elo_external_comparison_2025_26/
```

Snapshot tarihi ilk UEFA macindan once olmalidir. Sonradan guncellenmis ranking
pre-season validation diye kullanilmaz.

Bu kosu yalniz AO/Opta *uyumunu* olcer. Aktif contract'in dis referanslara karsi
gercek performansi icin iki eksenli benchmark ayri calistirilir:

```bash
python3 scripts/run_current_external_benchmark.py
```

Output ve curated kopya:

```text
output/current_external_benchmark/
reports/external_benchmark/
```

Onkosul: `run_current_model_evaluation.py` ve final prediction ensemble backtesti
daha once kosulmus olmalidir; benchmark bu iki kosunun dondurulmus ciktilarini
puanlar, model parametresi fit etmez. Dynamic core, 1X2 katmani, prediction
ensemble veya AO First Elo formulu degisirse yeniden kosulur ve
`reports/external_benchmark/` guncellenir.

Statik rating tarafinda kupa katkisi challenger'i ayri kosulur:

```bash
python3 scripts/run_cup_achievement_backtest.py
```

Output ve curated kopya:

```text
output/cup_achievement_backtest_2018_2026/
reports/cup_achievement/
```

Onkosul: `output/domestic_surprise_variance_backtest_2018_2026/` altindaki
dondurulmus Domestic Surprise ciktilari. Kosu production parametresi
degistirmez. Domestic achievement formulu, `cup_base_score` veya
`cup_double_bonus_multiplier` degisirse yeniden kosulur.

xG kaynagi ve bounded xG katmaninin yeniden dogrulanmasi:

```bash
python3 scripts/build_2025_26_xg_dataset.py \
  --season 2020/21 --season 2021/22 --season 2022/23 \
  --season 2023/24 --season 2024/25 --season 2025/26 \
  --output-root data/xg_2020_2026

python3 scripts/run_xg_multiseason_backtest.py
```

Output ve curated kopya:

```text
data/xg_2020_2026/
output/xg_multiseason_backtest_2020_2026/
reports/xg_multiseason/
```

Cekim FotMob'un herkese acik uc noktasini kullanir ve `_source_cache/` altinda
onbelleklenir; ayni onbellekle yeniden kosuldugunda deterministiktir. Kamuya
acik erisim yeniden dagitim lisansi vermez, bu alanlar yayimlanmadan once
kaynak kosullari kontrol edilmelidir.

`--season` bayragi tekrarlanabilir; varsayilan `2025/26` oldugu icin mevcut
dondurulmus veri setinin uretimi degismez.

Paylasilan `XG_DATA` sabiti alti sezonluk haritayi gosterir. Tek sezonluk
dondurulmus dosya `LEGACY_SINGLE_SEASON_XG_DATA` altinda provenance icin
erisilebilir kalir.

Canli sezon icin xG ayri toplanir ve preproduction sonuclarina baglanir:

```text
data/xg_2026_27/
data/season_2026_27_preproduction/matches_completed.csv   (xg_* kolonlari)
```

Bu baglanti kurulmadan 2026/27 replay'i xG'siz kosar ve motor mevcut sinyali
gormez.

## 8. Prediction Challenger'lari

Domestic Poisson:

```bash
python3 scripts/run_domestic_poisson_backtest.py
```

ML 1X2:

```bash
python3 scripts/run_ml_1x2_backtest.py
```

Final ML+Poisson ensemble:

```bash
python3 scripts/run_final_prediction_ensemble_backtest.py --bootstrap-samples 4000
```

Bu scriptlerin calismasi production aktivasyonu yapmaz. Her calismadan sonra
`selected_candidate.json` icindeki `production_activated` okunur.

### 8.1 2026/27 ML Feature Koprusu

Servis edilen 1X2'nin `%50 Current ML` yarisi, sezonlar arasi uzanan rolling
Avrupa form penceresi ister. 2026/27 preproduction paketi yalniz kendi sezonunu
tasidigi icin tek basina beslenirse pencereler kirpik kalir ve servis her satirda
sessizce Current AO'ya duser. Koprü bunu kurar:

```bash
python3 scripts/build_2026_27_prediction_features.py --predict
```

Tek fikstur icin:

```bash
python3 scripts/build_2026_27_prediction_features.py \
  --match-id UEFA-2049213 --predict
```

Output:

```text
output/season_2026_27_prediction_features/
```

Onkosul: `run_current_model_evaluation.py` ve
`run_2026_27_preproduction_replay.py` daha once kosulmus olmalidir.

Kritik davranis: script her exact-UTC kickoff grubunu **ayri** kurar. Yaklasan
bir fikstur, baska bir yaklasan fiksturun rolling penceresine placeholder
sonucuyla giremez. Bu ozellik `tests/test_2026_27_prediction_features.py`
icinde iki ayakli tie uzerinden sabitlenmistir.

Izlemede iki ayri sayac gerekir:

```text
fallback rate                      -> ML/Poisson hic calismadi
rows_with_imputed_model_input      -> ML calisti ama girdi uydurulmus
```

Ikincisi `fallback_rate` ile gorunmez: frozen pipeline eksik sayisal girdiyi
impute eder, satir yine `ACTIVE_ENSEMBLE` doner.

### 8.2 Aktif Production Prediction

Frozen artifact'lari tekrar uretme:

```bash
python3 scripts/build_production_prediction_artifacts.py
```

Ayni inputlarla uretilen uc artifact SHA-256 degeri contract ile ayni
kalmalidir. Fark varsa contract kendiliginden guncellenmez; once neden audit
edilir.

Sonucsuz pre-match feature CSV'sinden tahmin:

```bash
python3 scripts/run_production_prediction.py \
  --features path/to/pre_match_features.csv \
  --generated-at-utc 2026-09-01T16:00:00Z \
  --output path/to/pre_match_prediction_log.csv \
  --strict-artifacts
```

Website servisinde `ProductionPredictionService` ayni cekirdegi kullanir.
Normal isletimde artifact'lar startup'ta strict yuklenir; servis devam ederken
satir bazli feature/state hatasi `CURRENT_AO_1X2` fallback'i uretir. Monitoring
su oranlari alarm olarak izlemelidir:

```text
fallback rate
Domestic Poisson BOTH / ONE / NONE coverage
Brier ve log-loss: served vs Current AO
UCL / UEL / UECL segmentleri
calibration ve probability normalization
```

## 9. Production Degisikligi Kontrol Listesi

### Static degisiklik

- [ ] Hedef hipotez ve baseline yazildi.
- [ ] Same-season sonucu history'ye sizmiyor.
- [ ] AO First output kolonlari ve invariantlari guncel.
- [ ] Zero-exposure ve missing-history senaryolari test edildi.
- [ ] Ranking ve loss metrikleri birlikte raporlandi.
- [ ] External alignment gerilemedi.

### Dynamic degisiklik

- [ ] Prediction sonuc gorulmeden kilitleniyor.
- [ ] Scale/H/K semantigi acik.
- [ ] Match Power Delta sifir-toplamli.
- [ ] Penalty/extra-time/two-leg/single-match senaryolari test edildi.
- [ ] Duplicate, chronology ve state/config mismatch reddediliyor.
- [ ] Max movement ve rating inflation audit edildi.

### Prediction-only degisiklik

- [ ] Rating state baseline ile birebir ayni.
- [ ] Missing coverage fallback'i mevcut AO'ya donuyor.
- [ ] Olasiliklar finite, non-negative ve normalize.
- [ ] Calibration, Brier ve log-loss birlikte kontrol edildi.
- [ ] UCL/UEL/UECL segment zarari yok.

### Aktivasyon

- [ ] Kullanici acik production onayi verdi.
- [ ] Production contract guncellendi.
- [ ] Active constructor ayni degerleri kullaniyor.
- [ ] Contract regression testleri eklendi.
- [ ] Current evaluation yeniden uretildi.
- [ ] `CODEX.md` ve ilgili `docs/ai/` belgeleri guncellendi.
- [ ] Contract/feature schema/model artifact hash'leri kaydedildi.

## 10. Model Degisikliginde Ablation

Yeni modeli yalniz eski report ile karsilastirmak yeterli degildir. Ayni replay
icinde en az su kollar bulunmalidir:

```text
CURRENT_PRODUCTION
CURRENT + NEW_FEATURE
CURRENT with relevant existing layer disabled
NEW_FEATURE-only incremental diagnostic, uygulanabiliyorsa
```

Boylece improvement'in yeni feature'dan mi, baska parametre farkindan mi
geldigi anlasilir.

## 11. Leakage Kontrolu

Her fold icin:

1. Model fit sezonlari test sezonundan eski mi?
2. Static seed yalniz hedef sezondan once tamamlanan veriyi kullaniyor mu?
3. Current-season rolling feature sadece daha eski kickoff'lari goruyor mu?
4. Ayni UTC batch sonucu pre-match snapshot'a siziyor mu?
5. Tie advance veya final sonucu pre-match feature'a girmis mi?
6. xG yalniz settlement sonrasi rating update'te mi kullaniliyor?
7. External snapshot gercekten tahmin tarihinden once mi?

## 12. Determinizm Kontrolu

Ayni komutu ayni input/cache ile iki kez calistir. CSV sirasi ve sayisal cikti
kararli olmalidir. Random bootstrap/ML kullaniliyorsa seed manifestte tutulur.

Kontrol icin:

```bash
shasum -a 256 path/to/output.csv
```

Byte-stable olmayan ama sayisal olarak ayni cikti varsa timestamp veya kolon
sirasi gibi non-model farklari ayikla.

## 13. Git Hazirligi

```bash
git status --short
git diff --stat
git diff --check
```

Commit oncesi generated `output/` hacmini ve secret/API key izlerini kontrol et.
Sadece ilgili dosyalari stage et; kullanicinin ilgisiz dirty degisikliklerini
commit'e alma.

## 14. Belge Bakimi

Bu dokumantasyon seti su durumlarda zorunlu guncellenir:

- active parametre veya model version degisirse,
- input/output semasi degisirse,
- yeni public API eklenirse,
- research adayi production'a alinirsa,
- holdout karari veya evaluation penceresi degisirse,
- bilinen bir metodolojik sinir cozulur veya yenisi bulunursa.

Belgede eski sayisal sonucu korumak gerekiyorsa tarih ve kaynak artefact yolu
acikca yazilmalidir.
