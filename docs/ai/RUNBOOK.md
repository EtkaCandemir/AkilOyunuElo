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

AO First seed asimetrisi ve bagimsiz domestic-form boost arastirmasi:

```bash
python3 scripts/run_ao_first_seed_boost_backtest.py --bootstrap-samples 2000
```

Output ve curated kopya:

```text
output/ao_first_seed_boost_backtest_2018_2026/
reports/ao_first_seed_boost/
```

Bu kosuda `BOOST_BLIND` yalniz diagnostik kontroldur ve secilebilir production
adayi degildir. `BOOST_DOMESTIC_FORM` form snapshot'ini her sezonun ilk UEFA
kickoff'undan hemen once dondurur. `2026/27` kullanilmaz; production contract
baslangicta ve bitiste SHA-256 ile kontrol edilir.

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

Skor katmani icin xG form challenger'i:

```bash
python3 scripts/run_xg_goal_expectation_backtest.py
```

Output ve curated kopya:

```text
output/xg_goal_expectation_backtest_2020_2026/
reports/xg_goal_expectation/
```

Uc kol kosar: `ELO_ONLY` taban, `GOALS` kontrol, `XG` aday. `GOALS` kolu
kaldirilmamalidir; onsuz xG kazanci ile form terimi kazanci ayirt edilemez.

Ayni form terimini reponun en iyi skor kolunun uzerine koyan takip kosusu:

```bash
python3 scripts/run_xg_domestic_goal_expectation_backtest.py
```

Egitim penceresi duyarliligi:

```bash
python3 scripts/run_xg_domestic_goal_expectation_backtest.py \
  --training-window xg \
  --output-root output/xg_domestic_goal_expectation_backtest_2020_2026/sensitivity_xg_training_window
```

Output ve curated kopya:

```text
output/xg_domestic_goal_expectation_backtest_2020_2026/
reports/xg_domestic_goal_expectation/
```

Kollar `DOMESTIC_AD` taban, `DOMESTIC_AD_GOALS_FORM` kontrol ve
`DOMESTIC_AD_XG_FORM` adaydir. Taban, yayinlanmis
`DOMESTIC_ATTACK_DEFENCE_POISSON` kolunun aynisidir; kosu bunu her calismada
`baseline_matches_published_arm` kapisiyla dogrular. Onkosul olarak
`output/domestic_poisson_backtest_2018_2026/domestic_prequential_results.csv`
ve `output/ml_1x2_backtest_2018_2026/pre_match_feature_store.csv` bulunmalidir.
Kosu production contract'i yalniz hash'lemek icin okur ve cikista hash'in
degismedigini dogrular.

European prior katilim normalizasyonu:

```bash
python3 scripts/run_european_participation_backtest.py
```

Yalniz seed ekseni (mac replay'ini atlar, dakikalar kazandirir):

```bash
python3 scripts/run_european_participation_backtest.py --skip-loss-axis
```

Output ve curated kopya:

```text
output/european_participation_backtest_2018_2026/
reports/european_participation/
```

Uc kol: `BASELINE`, `PARTICIPATION_BLIND_LIFT` kontrol, `PARTICIPATION_NORMALIZED`
aday. Kor kontrol kaldirilmamalidir; onsuz kazancin katilim yapisindan mi yoksa
dusuk exposure'lu kulupleri topluca itmekten mi geldigi ayirt edilemez.

`baseline_reproduces_production` kapisi her kosuda dondurulmus seed
artifact'inin guncel contract'la uyumunu dogrular. Artifact bayatsa kosu
`162` Elo hatayla durur - bu kapi `2026-08-21` cap yayilimindaki staleness'i
bu sekilde yakalamistir.

Exposure cap x katilim etkilesimi:

```bash
python3 scripts/run_participation_exposure_interaction.py --loss-caps 0.70
```

Output ve curated kopya:

```text
output/participation_exposure_interaction_2018_2026/
reports/participation_exposure_interaction/
```

`14` cap degerini hem production prior'i hem katilim-normalize prior uzerinde
tarar ve her cap'i aktif `0.65`e karsi sezon-blogu bootstrap ile karsilastirir.
`--skip-loss-axis` mac replay'ini atlar. Katilim katmani aktive edilirse bu
kosu yeniden uretilmelidir.

Domestic Surprise guclendirme kolu:

```bash
python3 scripts/run_domestic_surprise_amplification_backtest.py \
  --baseline-source rebuilt_current_contract \
  --output-root output/domestic_surprise_amplification_backtest_2018_2026_rebuilt_065
```

Output ve curated kopya:

```text
output/domestic_surprise_amplification_backtest_2018_2026_rebuilt_065/
reports/domestic_surprise_amplification/
```

`--baseline-source` mutlaka `rebuilt_current_contract` olmalidir. `stored`
kolu, exposure cap `0.85`'ten `0.65`'e indigi icin artik production olmayan bir
tabana karsi olcer ve kazanci `53` kat abartir.

Kosu dort kapi tasir: dependency-robust loss envelope, sezon blogu ranking
veto'su (**ranking skorlarinda yuksek iyidir, zarar `upper < 0`**), fold secim
modal payi (`>= 0.50`), ve iki parcali kontrol-artifact uzlastirmasi. Ranking
zarari ile secim kararsizligi her iki terfi kademesini de veto eder.

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

28 Agustos 2026'dan itibaren domestic engine checkpoint semasi `2.0`'dir.
`1.0` state yuklenmez; asagidaki komutlarla sonuc kaynagindan replay edilir.
Checkpoint'te provider event ID listesi ve lig bazinda son kickoff saklanir.
Duplicate, eski sonuc veya ayni lig/kickoff'u bolen yeni batch reddedilir.
Bu durumlarda state'i elle duzenlemek yerine onceki checkpoint'ten tam replay
yapilir. Tahmin uretim zamani ilgili lig cutoff'unu geride birakamaz;
fikstur cutoff'a esit veya eskiyse satir AO fallback'e duser.

Frozen artifact'lari tekrar uretme:

```bash
python3 scripts/build_production_prediction_artifacts.py
```

UCL/UEL domestic coverage checkpoint'ini ayni sabit production parametreleriyle
yeniden kurma:

```bash
python3 scripts/build_production_prediction_artifacts.py \
  --domestic-matches data/domestic_league_expansion_ucl_uel/domestic_matches_candidate.csv \
  --domestic-bridge data/domestic_league_expansion_ucl_uel/domestic_team_bridge_with_live_candidate.csv \
  --live-domestic-matches data/domestic_league_expansion_ucl_uel/live_2026_27_matches.csv \
  --coverage-audit data/domestic_league_expansion_ucl_uel/target_coverage_audit.csv
```

`--coverage-audit` artik CLI tarafindan da **zorunlu kilinir**:
`--live-domestic-matches` verilip bu bayrak verilmezse build hicbir dosya
yazmadan durur.  Onceden yalnizca bu belgede zorunluydu, argparse optional
kabul ediyordu.

`--coverage-audit` **zorunludur**. Bayragi atlamak kapiyi tamamen kapatir:
`excluded_by_coverage_gate` `0` olur ve iki sezon/40 mac esigini gecemeyen
kulup (`União Torreense`) sessizce production Poisson'a dahil edilir. Dogru
kosuda `production_eligible_ao_clubs = 311` ve `excluded_by_coverage_gate = 1`
gorulmelidir.

Lig-sezon seviyesindeki coverage kapisi ayri bir kapidir ve iki lig-sezonunu
bilerek disarida birakir: **GEO 2014** (`120/184 = 0.652`) ve **LIT 2020**
(`60/127 = 0.472`). Bu satirlar `league_season_quality.csv` icinde
`REJECTED / UNAVAILABLE` olarak gorunur -- etiket primary kaynagin hatasini
tasir, gercek sebep secondary'nin coverage reddidir.  Durum **beklenen
davranistir**;
kabul edilmis kapsam boslugu olarak `docs/ai/DATA_CONTRACTS.md` §11.2'de
belgelidir. Bu iki sezonu geri almak icin %95 esigini dusurmeyin veya kapiyi
kapatmayin; dogru mudahale beklenen mac sayisini resmi fikstur kaydindan
pinlemektir. Bu listeye **yeni** bir lig-sezonu eklenirse once nedeni audit edilir.

Kapinin kararinin kaynak-bagimli oldugunu unutmayin: GEO 2020 secondary olcusune
gore (`94/184 = 0.511`) GEO 2014'ten daha kotudur ama primary fallback'i
(`92/72 = 1.278`) sayesinde production'a girer.  `ACCEPTED` bir satir sezonun
tam oldugunu kanitlamaz.  Ayrinti icin `DATA_CONTRACTS.md` §11.2.

**Her artifact build'i temiz, bos bir `--output-root` dizinine kosulmalidir.**
Builder girdileri dogrulamadan once ML artifact'ini yazar; girdi hatasi
downstream'de yakalanirsa dizinde yeni bir `structural_logistic_v1.joblib`
kalir, manifest ve state kalmaz.  Mevcut bir release dizini yeniden
kullanilirsa bu yeni ML dosyasi eski state/manifest ile yan yana durur.
Strict runtime checksum bunu reddeder, ama dizin yine de bozulur.  Ayni kural
`build_2026_27_prediction_features.py` icin de gecerlidir: prediction kilidi
dogrulanmadan once `prediction_features.csv` ve `validation_gates.csv`
yazilir.  Atomik yayinlama (temp dizin + rename) henuz uygulanmadi.

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

## 8.3 Aktivasyon Sonrasi Seed Zinciri Yayilimi

AO First formulunu degistiren her aktivasyon donmus artifact zincirini yukari
kosmak zorundadir. Sira ve **pin bayraklari** onemlidir; pin atlanirsa kosu
ilgili parametreyi sessizce yeniden secer ve yayilim gizli bir model
degisikligine doner.

```bash
python3 scripts/run_domestic_surprise_variance_backtest.py
python3 scripts/run_domestic_surprise_gamma_sensitivity.py --variance-penalty 0.5
python3 scripts/run_current_model_evaluation.py --bootstrap-samples 4000
python3 scripts/run_ml_1x2_backtest.py --blend-weight 0.9 --bootstrap-samples 4000
python3 scripts/run_domestic_poisson_backtest.py --bootstrap-samples 4000
python3 scripts/run_final_prediction_ensemble_backtest.py \
  --bootstrap-samples 4000 --prospective-poisson-weight 0.5
python3 scripts/run_current_external_benchmark.py
python3 scripts/run_2026_27_preproduction_replay.py
python3 scripts/run_2026_27_playoff_first_leg_ranking.py
```

Uc pin ve neden gerekli oldugu:

| Bayrak | Contract degeri | Pin yoksa yuzeyin sectigi |
| --- | --- | --- |
| `--variance-penalty` | `0.5` | `0.85` |
| `--blend-weight` | `0.9` | `1.0` |
| `--prospective-poisson-weight` | `0.5` | `0.4` |

Ucu de `HOLDOUT_PROTOCOL_2026_27.md` `§5` tarafindan donduruldugu icin
yeniden secilemez. Pin uygulandiginda yuzeyin kendi tercihi cikti dosyasina
`surface_would_have_selected` olarak yazilir.

Ayrica `output/v2_dynamic_calibration_2018_2026/selected_dynamic_model.json`
ve `output/v2_ranking_calibration_2018_2026/selected_model.json` icindeki
`static_config`, `AOEuropeanEloConfig.active()` ile alan alan ayni olmalidir
(`domestic_surprise_*` haric; o katman artifact uzerinden uygulanir). Bu iki
manifest butun arastirma zincirinin de-facto static config'idir ve eksik alan
sessizce dataclass default'una duser. `run_current_model_evaluation.py` her
kosuda bunu dogrular ve sapma varsa durur.

Zincir sonunda `contracts/ao_european_elo_v2_production.json` icindeki
`prediction_layer.artifact_manifest.sha256` yeni manifest degeriyle
guncellenmelidir.
Final-candidate contract'taki ayni artifact referansi da eslenmelidir.
Son hash guncellemesinden sonra current evaluation/snapshot yeniden uretilip
`reports/current_model/` altindaki curated kopyalar yenilenir; snapshot eski
contract hash'inde birakilmaz.

## 8.4 Domestic veri/state onarimi (AO First degismiyorsa)

Once kaynak hata duzeltilir; sonra kanitli skor uzlastirmasi ve tekillik guard'i
ile veri yeniden uretilir. Ham cache degistirilmez; onceki artifactlar
karsilastirma icin saklanir. 28 Agustos 2026 onarim sirasi:

```bash
python3 scripts/build_domestic_league_dataset.py --offline
python3 scripts/build_ucl_uel_domestic_expansion.py --offline
python3 scripts/build_ucl_uel_live_domestic_state.py --offline
python3 scripts/run_ml_1x2_backtest.py --blend-weight 0.9 --bootstrap-samples 4000
python3 scripts/run_domestic_poisson_backtest.py --rebuild-domestic-surface --bootstrap-samples 4000
python3 scripts/run_final_prediction_ensemble_backtest.py --prospective-poisson-weight 0.5 --bootstrap-samples 4000
```

Bu sirada eski feature store ve Poisson secim yuzeyi kullanilmaz. Genisletilmis
coverage kaniti `evaluate_ucl_uel_domestic_expansion.py` ile `--reuse-surface`
olmadan yenilenir. Ardindan §8.2'deki iki artifact build'i (tarihsel ve coverage
audit'li live), iki contract'taki manifest hash'i, current evaluation/snapshot,
external benchmark ve 2026/27 replay/tahmin raporlari yenilenir. Model config'i,
AO First ve sayisal blend/transfer parametreleri degistirilmez. 2026/27 ciktilari
yalniz replay'dir; gecmiste kickoff'tan once kayit uretildigini kanitlamaz.

`fixture_reconciliation_audit.csv`, source manifest, kanonik fikstur tekilligi
ve checkpoint `processed_event_ids` birlikte dogrulanir. Skor celiskisine yeni
bir provider gozlemi eklenirse kayitli karar otomatik genisletilmez; build durur.

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
