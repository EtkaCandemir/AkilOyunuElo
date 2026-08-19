# Guncel Production Modeli Dis Benchmark

Iki bagimsiz dis eksen. Hicbir production parametresi degistirilmez;
dondurulmus tahmin ve rating artefact'lari puanlanir.

## Eksen 1: ClubElo'ya karsi mac tahmini

- Eslesmis mac: `363`.
- Puanlanan sezonlar: `2020/21, 2021/22, 2022/23, 2023/24, 2024/25`.
- ClubElo snapshot yas siniri: `31` gun.
- ClubElo ev sahibi avantaji yalniz onceki sezonlardan walk-forward fit edilir.
- Iki tarafa da ayni score-preserving beraberlik modeli uygulanir; bu, karsilastirmayi
  beraberlik modeli farkindan aritip rating kalitesine odaklar.
- `CLUBELO_TUNED_SCALE_AND_H`, ClubElo'ya olcek ve H'yi birlikte fit eden cömert koldur.

```text
                  model_arm  matches  brier_1x2  log_loss_1x2  accuracy_1x2  brier_skill_vs_climatology  log_loss_skill_vs_climatology
   CLIMATOLOGY_WALK_FORWARD      363   0.642983      1.062634      0.457300                    0.000000                       0.000000
CLUBELO_PUBLISHED_SCALE_400      363   0.574983      0.966819      0.537190                   -0.105757                      -0.090167
  CLUBELO_TUNED_SCALE_AND_H      363   0.574890      0.966486      0.531680                   -0.105902                      -0.090481
         AO_RATING_CORE_1X2      363   0.587835      0.987568      0.534435                   -0.085770                      -0.070641
   AO_SERVED_ENSEMBLE_50_50      363   0.585941      0.982793      0.545455                   -0.088715                      -0.075135
```

### Bagimliliga dayanikli guven zarfi

Pozitif fark ClubElo'nun onde oldugunu gosterir.

```text
                  ao_arm                external_arm       metric  matches  mean_ao_minus_external  ci_95_lower  ci_95_upper  ao_reliably_better  external_reliably_better
      AO_RATING_CORE_1X2 CLUBELO_PUBLISHED_SCALE_400    brier_1x2      363                0.012851    -0.014290     0.041184               False                     False
      AO_RATING_CORE_1X2 CLUBELO_PUBLISHED_SCALE_400 log_loss_1x2      363                0.020750    -0.017149     0.060903               False                     False
      AO_RATING_CORE_1X2   CLUBELO_TUNED_SCALE_AND_H    brier_1x2      363                0.012945    -0.013544     0.040409               False                     False
      AO_RATING_CORE_1X2   CLUBELO_TUNED_SCALE_AND_H log_loss_1x2      363                0.021083    -0.015714     0.059768               False                     False
AO_SERVED_ENSEMBLE_50_50 CLUBELO_PUBLISHED_SCALE_400    brier_1x2      363                0.010958    -0.016152     0.039625               False                     False
AO_SERVED_ENSEMBLE_50_50 CLUBELO_PUBLISHED_SCALE_400 log_loss_1x2      363                0.015974    -0.022677     0.056868               False                     False
AO_SERVED_ENSEMBLE_50_50   CLUBELO_TUNED_SCALE_AND_H    brier_1x2      363                0.011051    -0.015150     0.039197               False                     False
AO_SERVED_ENSEMBLE_50_50   CLUBELO_TUNED_SCALE_AND_H log_loss_1x2      363                0.016307    -0.021352     0.055872               False                     False
```

### Turnuva kirilimi

```text
competition                   model_arm  matches  brier_1x2  log_loss_1x2  accuracy_1x2
        UCL          AO_RATING_CORE_1X2      171   0.598254      0.999254      0.532164
        UCL    AO_SERVED_ENSEMBLE_50_50      171   0.597002      0.996697      0.543860
        UCL    CLIMATOLOGY_WALK_FORWARD      171   0.643057      1.062010      0.432749
        UCL CLUBELO_PUBLISHED_SCALE_400      171   0.566977      0.957281      0.578947
        UCL   CLUBELO_TUNED_SCALE_AND_H      171   0.567515      0.957320      0.573099
       UECL          AO_RATING_CORE_1X2       90   0.567277      0.957099      0.555556
       UECL    AO_SERVED_ENSEMBLE_50_50       90   0.559193      0.940262      0.555556
       UECL    CLIMATOLOGY_WALK_FORWARD       90   0.644603      1.064520      0.433333
       UECL CLUBELO_PUBLISHED_SCALE_400       90   0.567717      0.952306      0.500000
       UECL   CLUBELO_TUNED_SCALE_AND_H       90   0.567646      0.952964      0.488889
        UEL          AO_RATING_CORE_1X2      102   0.588506      0.994863      0.519608
        UEL    AO_SERVED_ENSEMBLE_50_50      102   0.590999      0.997010      0.539216
        UEL    CLIMATOLOGY_WALK_FORWARD      102   0.641429      1.062015      0.519608
        UEL CLUBELO_PUBLISHED_SCALE_400      102   0.594816      0.995614      0.500000
        UEL   CLUBELO_TUNED_SCALE_AND_H      102   0.593644      0.993783      0.500000
```

### ClubElo walk-forward kalibrasyonu

```text
 season  train_matches  published_home_advantage  tuned_scale  tuned_home_advantage
2020/21              0                  0.000000   400.000000              0.000000
2021/22             53                -16.918725   344.660869            -15.260164
2022/23            124                  2.081322   401.953967              2.106861
2023/24            204                 52.181902   444.374061             56.655032
2024/25            278                 52.120931   436.085037             55.780686
```

### Kapsam

```text
 season  source_matches  clubelo_paired  scored_matches
2018/19             735              71               0
2019/20             721              58               0
2020/21             540              53              53
2021/22             816              71              71
2022/23             804              80              80
2023/24             806              74              74
2024/25             957              85              85
2025/26             961              89               0
```

## Eksen 2: Sezon basi rating

- Sezon: `2025/26`; takim: `236`.
- Opta snapshot `2025-07-03`, ilk 2025/26 macindan kesin olarak once alinmistir.
- Hedef, hicbir ratingden etkilenmeyen leave-team-out schedule-adjusted
  sezon performansidir.
- `reference_type` kolonu kanit degerini belirler: `EXTERNAL` bagimsiz hakemdir,
  `OWN_INPUT` modelin kendi girdisidir ve yalnizca value-added tabanini olcer.

```text
                          rating reference_type  teams  spearman_vs_realized  pearson_vs_realized  pairwise_accuracy_vs_realized
                    AO_FIRST_ELO          MODEL    236              0.416620             0.386085                       0.644052
   OPTA_PRE_SEASON_POWER_RANKING       EXTERNAL    236              0.486618             0.489748                       0.671870
UEFA_CLUB_COEFFICIENT_PRE_SEASON      OWN_INPUT    236              0.268334             0.284027                       0.596310
```

Eslesmis Spearman farklari:

```text
                                         comparison reference_type  teams  spearman_difference  ci_95_lower  ci_95_upper  ao_reliably_better  benchmark_reliably_better  rank_agreement_spearman
   AO_FIRST_ELO_minus_OPTA_PRE_SEASON_POWER_RANKING       EXTERNAL    236            -0.070047    -0.118927    -0.018841               False                       True                 0.912757
AO_FIRST_ELO_minus_UEFA_CLUB_COEFFICIENT_PRE_SEASON      OWN_INPUT    236             0.147080     0.050283     0.250136                True                      False                 0.756011
```

UEFA kulup katsayisi bagimsiz bir benchmark degildir: `club_points_t_*` girdileri
o katsayinin bes sezonluk bilesenleridir. Bu satir yalniz su soruyu cevaplar:
statik pipeline, tukettigi ham girdinin uzerine deger katiyor mu? Yayinlanmis 2026
katsayisi kullanilmaz, cunku bes sezonluk penceresi tahmin edilen sezonu icerir.

## Yorum siniri

ClubElo arsivi agirlikla yerlesik kulupleri kapsar; eslesmis ornek tum eleme
turu takimlarini temsil etmez. Eksen 1 bu nedenle yararli bir dis diagnostiktir,
evrensel ustunluk iddiasi degildir. AO parametreleri ayni sezonlarda kalibre
edildigi icin bu kosu da saf prospective holdout sayilmaz; 2026/27 kilitli
ledger'i ayri kalir.

Eksen 2 tek sezonluk bir olcumdur ve Opta snapshot'i ticari, kapali bir modeldir.
Sonuc AO First Elo'nun yanlis oldugunu degil, ayni siralamayi daha gurultulu
urettigini gosterir.
