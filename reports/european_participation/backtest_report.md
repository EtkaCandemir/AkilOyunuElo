# European Prior Katilim Normalizasyonu

Karar: **PROMOTE_CANDIDATE**. Production contract degistirilmedi.

## Soru

Production European Prior, bes sezonluk UEFA puanlarinin sabit agirlikli
toplamidir ve **girilmeyen sezon sifir puan katkisi yapar** - girilip hic puan
alinamayan sezondan ayirt edilemez. Boylece "Avrupa'da ne kadar iyiydin" ile
"katilabildin mi" karisir; ikincisi domestic prior'in zaten sahiplendigi bir
olgudur ve iki kez faturalandirilir.

Bu kosu ilk faturayi kaldirir, ikincisine dokunmaz:

```text
rate = weighted_european_history * (1 + k) / (weighted_season_exposure + k)
```

Tam katilimda `rate` yayinlanmis history'nin aynisidir, yani tam kanita sahip
kulup tanim geregi hic hareket etmez.

## Dogrulama kapilari

```text
                          gate  passed                                                         observed                                            requirement
baseline_reproduces_production    True                                                              0.0        The baseline arm must reproduce the served seed
  full_participation_unchanged    True                                                              0.0          Complete five-season evidence must never move
  zero_participation_unchanged    True                                                              0.0           A club that never entered must keep its seed
           control_arm_present    True                                                                3                    The blind control cannot be dropped
       candidate_beats_control    True                                                              1.0     The candidate must beat the blind control on folds
   ranking_not_reliably_harmed    True                                                          0.00345 Higher is better, so harm is an upper bound below zero
           selection_stability    True                                                         0.666667                          Modal fold share at least 0.5
         seed_movement_bounded    True                                                        97.425467                    p95 seed movement at most 100.0 Elo
     contract_sha256_unchanged    True 974dfd1fcf618b225d724ae8e6184e81fed9c83f1e054f80239d186535e9aae9    No production file may change during a research run
```

## Kol ozeti (sezon basi seed Spearman)

```text
                     arm  teams  seasons  spearman  pairwise_accuracy  mean_abs_elo_delta  p95_abs_elo_delta  spearman_vs_baseline  pairwise_accuracy_vs_baseline
                BASELINE   1413        6  0.459768           0.659656        6.661904e-14       2.273737e-13              0.000000                       0.000000
PARTICIPATION_BLIND_LIFT   1413        6  0.457303           0.658590        2.264914e+01       4.503329e+01             -0.002465                      -0.001066
PARTICIPATION_NORMALIZED   1413        6  0.469202           0.663092        2.655590e+01       9.742547e+01              0.009434                       0.003436
```

## Fold bazinda

```text
 fold test_season                      arm  teams  seed_spearman  seed_pairwise_accuracy  delta_seed_spearman  delta_seed_pairwise_accuracy
    1     2020/21                 BASELINE    237       0.464493                0.663913             0.000000                      0.000000
    1     2020/21 PARTICIPATION_BLIND_LIFT    237       0.460700                0.662441            -0.003794                     -0.001472
    1     2020/21 PARTICIPATION_NORMALIZED    237       0.467564                0.664775             0.003071                      0.000862
    2     2021/22                 BASELINE    237       0.466885                0.660975             0.000000                      0.000000
    2     2021/22 PARTICIPATION_BLIND_LIFT    237       0.466432                0.660579            -0.000453                     -0.000395
    2     2021/22 PARTICIPATION_NORMALIZED    237       0.470184                0.661910             0.003299                      0.000935
    3     2022/23                 BASELINE    233       0.456158                0.657518             0.000000                      0.000000
    3     2022/23 PARTICIPATION_BLIND_LIFT    233       0.453330                0.656328            -0.002828                     -0.001190
    3     2022/23 PARTICIPATION_NORMALIZED    233       0.466310                0.662836             0.010152                      0.005318
    4     2023/24                 BASELINE    234       0.468141                0.661696             0.000000                      0.000000
    4     2023/24 PARTICIPATION_BLIND_LIFT    234       0.468176                0.661807             0.000035                      0.000111
    4     2023/24 PARTICIPATION_NORMALIZED    234       0.490178                0.669558             0.022038                      0.007862
    5     2024/25                 BASELINE    236       0.484950                0.668768             0.000000                      0.000000
    5     2024/25 PARTICIPATION_BLIND_LIFT    236       0.480346                0.666812            -0.004604                     -0.001956
    5     2024/25 PARTICIPATION_NORMALIZED    236       0.500012                0.674057             0.015063                      0.005289
    6     2025/26                 BASELINE    236       0.417957                0.645030             0.000000                      0.000000
    6     2025/26 PARTICIPATION_BLIND_LIFT    236       0.414826                0.643545            -0.003131                     -0.001485
    6     2025/26 PARTICIPATION_NORMALIZED    236       0.421107                0.645465             0.003150                      0.000435
```

## Nested secimler

```text
 fold test_season  selected_shrinkage  blind_lift_coefficient
    1     2020/21                0.75               24.928985
    2     2021/22                0.20               40.505163
    3     2022/23                0.20               41.828664
    4     2023/24                0.20               43.326590
    5     2024/25                0.20               45.033293
    6     2025/26                0.00               61.645410
```

## Belirsizlik: tabana karsi

```text
                metric                 method  folds  mean_difference  ci_95_lower  ci_95_upper  reliable_improvement  reliable_harm
         seed_spearman season_block_bootstrap      6         0.009462     0.004316     0.015733                  True          False
seed_pairwise_accuracy season_block_bootstrap      6         0.003450     0.001398     0.005842                  True          False
```

## Belirsizlik: kor kontrole karsi

Iki kol da ayni kutleyi ayni ortalama buyuklukte hareket ettirir. Aralarindaki
guvenilir bir fark yalniz katilim yapisini okumaktan gelebilir.

```text
                metric                 method  folds  mean_difference  ci_95_lower  ci_95_upper  reliable_improvement  reliable_harm
         seed_spearman season_block_bootstrap      6         0.011925     0.006652     0.017587                  True          False
seed_pairwise_accuracy season_block_bootstrap      6         0.004515     0.002498     0.006557                  True          False
```

## Seed hareketi

```text
                     arm  team_seasons  moved_team_seasons  mean_abs_delta  p95_abs_delta  max_abs_delta  full_participation_max_abs_delta  zero_participation_max_abs_delta
                BASELINE          1413                   0    6.661904e-14   2.273737e-13   4.547474e-13                      4.547474e-13                           0.00000
PARTICIPATION_BLIND_LIFT          1413                1413    2.264914e+01   4.503329e+01   6.164541e+01                      2.157589e+01                          61.64541
PARTICIPATION_NORMALIZED          1413                 745    2.655590e+01   9.742547e+01   1.659690e+02                      4.547474e-13                           0.00000
```

## Mac loss no-harm ekseni

```text
                     arm  matches  brier_1x2  log_loss_1x2  brier_1x2_vs_baseline  log_loss_1x2_vs_baseline
                BASELINE     6340   0.570934      0.962432               0.000000                  0.000000
PARTICIPATION_BLIND_LIFT     6340   0.570851      0.962324              -0.000083                 -0.000108
PARTICIPATION_NORMALIZED     6340   0.568884      0.959528              -0.002051                 -0.002904
```

## Opta ekseni (diagnostik, tek sezon)

Opta yalniz `2025/26` icin pre-season snapshot yayinlar; `236` kulup, karar
kapisi degil. Dokumanlarda alintilanan acik bu eksendedir.

```text
 season                      arm  teams  seed_spearman  seed_pairwise_accuracy  opta_published_spearman  gap_to_opta
2025/26                 BASELINE    236       0.417957                0.645030                 0.486618    -0.068661
2025/26 PARTICIPATION_BLIND_LIFT    236       0.414826                0.643545                 0.486618    -0.071792
2025/26 PARTICIPATION_NORMALIZED    236       0.421107                0.645465                 0.486618    -0.065511
```

## Karar girdisi

- Aday tabana karsi guvenilir mi: `True`
- Aday kontrole karsi guvenilir mi: `True`
- Tum kapilar gecti mi: `True`

Karar urun tarafina aittir; bu belge yalniz kanit uretir.
