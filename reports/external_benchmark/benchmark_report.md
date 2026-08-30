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
         AO_RATING_CORE_1X2      363   0.575267      0.969396      0.550964                   -0.105315                      -0.087742
   AO_SERVED_ENSEMBLE_50_50      363   0.577783      0.971384      0.550964                   -0.101402                      -0.085871
```

### Bagimliliga dayanikli guven zarfi

Pozitif fark ClubElo'nun onde oldugunu gosterir.

```text
                  ao_arm                external_arm       metric  matches  mean_ao_minus_external  ci_95_lower  ci_95_upper  ao_reliably_better  external_reliably_better
      AO_RATING_CORE_1X2 CLUBELO_PUBLISHED_SCALE_400    brier_1x2      363                0.000284    -0.025291     0.027927               False                     False
      AO_RATING_CORE_1X2 CLUBELO_PUBLISHED_SCALE_400 log_loss_1x2      363                0.002577    -0.033564     0.041728               False                     False
      AO_RATING_CORE_1X2   CLUBELO_TUNED_SCALE_AND_H    brier_1x2      363                0.000378    -0.024885     0.027829               False                     False
      AO_RATING_CORE_1X2   CLUBELO_TUNED_SCALE_AND_H log_loss_1x2      363                0.002910    -0.032286     0.041422               False                     False
AO_SERVED_ENSEMBLE_50_50 CLUBELO_PUBLISHED_SCALE_400    brier_1x2      363                0.002800    -0.024110     0.031134               False                     False
AO_SERVED_ENSEMBLE_50_50 CLUBELO_PUBLISHED_SCALE_400 log_loss_1x2      363                0.004565    -0.032832     0.043496               False                     False
AO_SERVED_ENSEMBLE_50_50   CLUBELO_TUNED_SCALE_AND_H    brier_1x2      363                0.002894    -0.023516     0.030917               False                     False
AO_SERVED_ENSEMBLE_50_50   CLUBELO_TUNED_SCALE_AND_H log_loss_1x2      363                0.004899    -0.031941     0.043112               False                     False
```

### Turnuva kirilimi

```text
competition                   model_arm  matches  brier_1x2  log_loss_1x2  accuracy_1x2
        UCL          AO_RATING_CORE_1X2      171   0.587840      0.984005      0.538012
        UCL    AO_SERVED_ENSEMBLE_50_50      171   0.590133      0.985844      0.549708
        UCL    CLIMATOLOGY_WALK_FORWARD      171   0.643057      1.062010      0.432749
        UCL CLUBELO_PUBLISHED_SCALE_400      171   0.566977      0.957281      0.578947
        UCL   CLUBELO_TUNED_SCALE_AND_H      171   0.567515      0.957320      0.573099
       UECL          AO_RATING_CORE_1X2       90   0.549196      0.932852      0.566667
       UECL    AO_SERVED_ENSEMBLE_50_50       90   0.544851      0.920984      0.544444
       UECL    CLIMATOLOGY_WALK_FORWARD       90   0.644603      1.064520      0.433333
       UECL CLUBELO_PUBLISHED_SCALE_400       90   0.567717      0.952306      0.500000
       UECL   CLUBELO_TUNED_SCALE_AND_H       90   0.567646      0.952964      0.488889
        UEL          AO_RATING_CORE_1X2      102   0.577194      0.977149      0.558824
        UEL    AO_SERVED_ENSEMBLE_50_50      102   0.586138      0.991614      0.558824
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
                     AO_FIRST_ELO          MODEL    236              0.423578             0.406560                       0.647421
AO_FIRST_ELO_NO_DOMESTIC_SURPRISE MODEL_ABLATION    236              0.421611             0.404673                       0.646334
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
                    AO_FIRST_ELO_minus_AO_FIRST_ELO_NO_DOMESTIC_SURPRISE MODEL_ABLATION    236             0.001970    -0.004736     0.008730               False                      False                 0.998568
                        AO_FIRST_ELO_minus_OPTA_PRE_SEASON_POWER_RANKING       EXTERNAL    236            -0.063009    -0.105594    -0.019895               False                       True                 0.934919
                     AO_FIRST_ELO_minus_UEFA_CLUB_COEFFICIENT_PRE_SEASON      OWN_INPUT    236             0.154118     0.057291     0.258393                True                      False                 0.754347
   AO_FIRST_ELO_NO_DOMESTIC_SURPRISE_minus_OPTA_PRE_SEASON_POWER_RANKING       EXTERNAL    236            -0.064979    -0.107300    -0.022592               False                       True                 0.934174
AO_FIRST_ELO_NO_DOMESTIC_SURPRISE_minus_UEFA_CLUB_COEFFICIENT_PRE_SEASON      OWN_INPUT    236             0.152148     0.056074     0.254445                True                      False                 0.760996
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
