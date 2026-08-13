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

### 8.1 Aktif Production Prediction

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
