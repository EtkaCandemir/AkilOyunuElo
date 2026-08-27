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
         AO_RATING_CORE_1X2      363   0.577244      0.972194      0.553719                   -0.102241                      -0.085109
   AO_SERVED_ENSEMBLE_50_50      363   0.578708      0.972945      0.550964                   -0.099964                      -0.084402
```

### Bagimliliga dayanikli guven zarfi

Pozitif fark ClubElo'nun onde oldugunu gosterir.

```text
                  ao_arm                external_arm       metric  matches  mean_ao_minus_external  ci_95_lower  ci_95_upper  ao_reliably_better  external_reliably_better
      AO_RATING_CORE_1X2 CLUBELO_PUBLISHED_SCALE_400    brier_1x2      363                0.002261    -0.024497     0.031243               False                     False
      AO_RATING_CORE_1X2 CLUBELO_PUBLISHED_SCALE_400 log_loss_1x2      363                0.005375    -0.032010     0.046298               False                     False
      AO_RATING_CORE_1X2   CLUBELO_TUNED_SCALE_AND_H    brier_1x2      363                0.002355    -0.023640     0.031312               False                     False
      AO_RATING_CORE_1X2   CLUBELO_TUNED_SCALE_AND_H log_loss_1x2      363                0.005708    -0.030970     0.046474               False                     False
AO_SERVED_ENSEMBLE_50_50 CLUBELO_PUBLISHED_SCALE_400    brier_1x2      363                0.003725    -0.023803     0.033440               False                     False
AO_SERVED_ENSEMBLE_50_50 CLUBELO_PUBLISHED_SCALE_400 log_loss_1x2      363                0.006126    -0.032767     0.047342               False                     False
AO_SERVED_ENSEMBLE_50_50   CLUBELO_TUNED_SCALE_AND_H    brier_1x2      363                0.003818    -0.023392     0.032982               False                     False
AO_SERVED_ENSEMBLE_50_50   CLUBELO_TUNED_SCALE_AND_H log_loss_1x2      363                0.006460    -0.032464     0.047016               False                     False
```

### Turnuva kirilimi

```text
competition                   model_arm  matches  brier_1x2  log_loss_1x2  accuracy_1x2
        UCL          AO_RATING_CORE_1X2      171   0.591809      0.989656      0.538012
        UCL    AO_SERVED_ENSEMBLE_50_50      171   0.591962      0.990148      0.549708
        UCL    CLIMATOLOGY_WALK_FORWARD      171   0.643057      1.062010      0.432749
        UCL CLUBELO_PUBLISHED_SCALE_400      171   0.566977      0.957281      0.578947
        UCL   CLUBELO_TUNED_SCALE_AND_H      171   0.567515      0.957320      0.573099
       UECL          AO_RATING_CORE_1X2       90   0.549725      0.933618      0.566667
       UECL    AO_SERVED_ENSEMBLE_50_50       90   0.544277      0.919129      0.555556
       UECL    CLIMATOLOGY_WALK_FORWARD       90   0.644603      1.064520      0.433333
       UECL CLUBELO_PUBLISHED_SCALE_400       90   0.567717      0.952306      0.500000
       UECL   CLUBELO_TUNED_SCALE_AND_H       90   0.567646      0.952964      0.488889
        UEL          AO_RATING_CORE_1X2      102   0.577107      0.976956      0.568627
        UEL    AO_SERVED_ENSEMBLE_50_50      102   0.586869      0.991589      0.549020
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
- `MODEL_ABLATION`, Domestic Surprise kapali seed'dir. Katmanin gerekcesi mac
  tahmini degil sezon basi seed kalitesi oldugu icin dogru eksen budur.

```text
                           rating reference_type  teams  spearman_vs_realized  pearson_vs_realized  pairwise_accuracy_vs_realized
                     AO_FIRST_ELO          MODEL    236              0.423272             0.402056                       0.647167
AO_FIRST_ELO_NO_DOMESTIC_SURPRISE MODEL_ABLATION    236              0.421493             0.400152                       0.646262
    OPTA_PRE_SEASON_POWER_RANKING       EXTERNAL    236              0.486618             0.489748                       0.671870
 UEFA_CLUB_COEFFICIENT_PRE_SEASON      OWN_INPUT    236              0.268334             0.284027                       0.596310
```

### Domestic Surprise'in seed uzerindeki etkisi

- Layer'in oynattigi takim: `181` / `236`.
- Ortalama mutlak seed hareketi: `8.902` Elo.
- Maksimum mutlak seed hareketi: `30.000` Elo.
- Tek sezon ve tek dis referans; bu bir diagnostiktir, terfi/kapatma kapisi degildir.

Eslesmis Spearman farklari:

```text
                                                              comparison reference_type  teams  spearman_difference  ci_95_lower  ci_95_upper  ao_reliably_better  benchmark_reliably_better  rank_agreement_spearman
                    AO_FIRST_ELO_minus_AO_FIRST_ELO_NO_DOMESTIC_SURPRISE MODEL_ABLATION    236             0.001779    -0.004942     0.008528               False                      False                 0.998542
                        AO_FIRST_ELO_minus_OPTA_PRE_SEASON_POWER_RANKING       EXTERNAL    236            -0.063319    -0.106158    -0.020041               False                       True                 0.934826
                     AO_FIRST_ELO_minus_UEFA_CLUB_COEFFICIENT_PRE_SEASON      OWN_INPUT    236             0.153808     0.056536     0.257488                True                      False                 0.754245
   AO_FIRST_ELO_NO_DOMESTIC_SURPRISE_minus_OPTA_PRE_SEASON_POWER_RANKING       EXTERNAL    236            -0.065098    -0.107604    -0.022454               False                       True                 0.934156
AO_FIRST_ELO_NO_DOMESTIC_SURPRISE_minus_UEFA_CLUB_COEFFICIENT_PRE_SEASON      OWN_INPUT    236             0.152029     0.055868     0.254579                True                      False                 0.760978
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
