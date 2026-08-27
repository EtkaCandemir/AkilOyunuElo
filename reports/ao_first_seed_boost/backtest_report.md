# AO First Elo Seed Asimetrisi Boost Backtesti

Karar: **`KEEP_DIAGNOSTIC`**. Production aktivasyonu: `False`.

## Soru

Ince fakat olumsuz Avrupa gecmisi, sifir Avrupa gecmisinden daha agir
cezalandirilan ve son lig bitirisi zayif gorunen kulupte, bagimsiz bir
'su anda iyi' sinyali seed ranking acigini kapatiyor mu?

## Kollar

- `BASELINE`: aktif AO First Elo.
- `BOOST_BLIND`: yalniz zayif Avrupa + zayif son bitiris; kontrol.
- `BOOST_DOMESTIC_FORM`: ayni kural + causal domestic Poisson attack/defence yuzdeligi.
- `BOOST_LONG_HISTORY`: veri yoklugu nedeniyle `UNAVAILABLE_NOT_RUN`; sahte kol uretilmedi.

Her aday additive, exposure modifier ve floor tasarimlarini; finish/deficit
esiklerini ve boost buyuklugunu yalniz onceki sezonlarda nested olarak secer.
Exposure `0` olan takim hicbir kosulda etkilenmez. Maksimum boost `150` Elo'dur.

## Pooled sonuc

```text
                arm  matches  brier_1x2  log_loss_1x2  accuracy_1x2  seed_spearman  seed_pairwise_accuracy  spearman_fold_wins  pairwise_fold_wins  delta_vs_baseline_brier_1x2  delta_vs_baseline_log_loss_1x2  delta_vs_baseline_accuracy_1x2  delta_vs_baseline_seed_spearman  delta_vs_baseline_seed_pairwise_accuracy
           BASELINE     4884   0.571537      0.963594      0.549959       0.461424                0.660291                   0                   0                     0.000000                        0.000000                        0.000000                         0.000000                                  0.000000
        BOOST_BLIND     4884   0.571089      0.962919      0.550164       0.461933                0.660205                   4                   3                    -0.000448                       -0.000675                        0.000205                         0.000509                                 -0.000086
BOOST_DOMESTIC_FORM     4884   0.571472      0.963472      0.550164       0.461717                0.660412                   1                   1                    -0.000065                       -0.000122                        0.000205                         0.000293                                  0.000121
```

## Fold bazli seed ranking

```text
 fold test_season                 arm  teams  seasons  spearman  pairwise_accuracy  delta_spearman_vs_baseline  delta_pairwise_vs_baseline  affected_teams
    1     2020/21            BASELINE    237        1  0.466226           0.665314                    0.000000                    0.000000               0
    1     2020/21         BOOST_BLIND    237        1  0.464538           0.664272                   -0.001688                   -0.001042              43
    1     2020/21 BOOST_DOMESTIC_FORM    237        1  0.466226           0.665314                    0.000000                    0.000000               0
    2     2021/22            BASELINE    237        1  0.476132           0.663923                    0.000000                    0.000000               0
    2     2021/22         BOOST_BLIND    237        1  0.476440           0.663743                    0.000308                   -0.000180              31
    2     2021/22 BOOST_DOMESTIC_FORM    237        1  0.476132           0.663923                    0.000000                    0.000000               0
    3     2022/23            BASELINE    233        1  0.457114           0.658411                    0.000000                    0.000000               0
    3     2022/23         BOOST_BLIND    233        1  0.457429           0.658968                    0.000315                    0.000558              40
    3     2022/23 BOOST_DOMESTIC_FORM    233        1  0.457114           0.658411                    0.000000                    0.000000               0
    4     2023/24            BASELINE    234        1  0.468568           0.661955                    0.000000                    0.000000               0
    4     2023/24         BOOST_BLIND    234        1  0.473645           0.663763                    0.005077                    0.001809              26
    4     2023/24 BOOST_DOMESTIC_FORM    234        1  0.468568           0.661955                    0.000000                    0.000000               0
    5     2024/25            BASELINE    236        1  0.483807           0.668043                    0.000000                    0.000000               0
    5     2024/25         BOOST_BLIND    236        1  0.486145           0.668695                    0.002337                    0.000652              34
    5     2024/25 BOOST_DOMESTIC_FORM    236        1  0.485563           0.668768                    0.001755                    0.000725               1
    6     2025/26            BASELINE    236        1  0.416620           0.644052                    0.000000                    0.000000               0
    6     2025/26         BOOST_BLIND    236        1  0.413372           0.641771                   -0.003248                   -0.002282              40
    6     2025/26 BOOST_DOMESTIC_FORM    236        1  0.416620           0.644052                    0.000000                    0.000000               0
```

## Nested secimler

```text
 fold test_season                 arm                                           selected_candidate_key  train_delta_spearman_vs_baseline
    1     2020/21         BOOST_BLIND                 BLIND_FLOOR_finish0.8_deficit50_m50_form0_cap150                          0.011827
    1     2020/21 BOOST_DOMESTIC_FORM                                                         BASELINE                          0.000000
    2     2021/22         BOOST_BLIND                BLIND_FLOOR_finish0.8_deficit150_m50_form0_cap150                          0.007526
    2     2021/22 BOOST_DOMESTIC_FORM                                                         BASELINE                          0.000000
    3     2022/23         BOOST_BLIND                BLIND_FLOOR_finish0.8_deficit150_m50_form0_cap150                          0.005722
    3     2022/23 BOOST_DOMESTIC_FORM                                                         BASELINE                          0.000000
    4     2023/24         BOOST_BLIND                BLIND_FLOOR_finish0.8_deficit150_m50_form0_cap150                          0.004655
    4     2023/24 BOOST_DOMESTIC_FORM                                                         BASELINE                          0.000000
    5     2024/25         BOOST_BLIND                 BLIND_FLOOR_finish0.8_deficit50_m50_form0_cap150                          0.005120
    5     2024/25 BOOST_DOMESTIC_FORM DOMESTIC_FORM_ADDITIVE_finish0.8_deficit100_m150_form0.65_cap150                          0.000110
    6     2025/26         BOOST_BLIND                 BLIND_FLOOR_finish0.8_deficit50_m50_form0_cap150                          0.004722
    6     2025/26 BOOST_DOMESTIC_FORM DOMESTIC_FORM_ADDITIVE_finish0.8_deficit100_m150_form0.65_cap150                          0.000345
```

## Opta dis benchmark ekseni (2025/26)

```text
                arm  teams  spearman_vs_realized  pairwise_accuracy_vs_realized  opta_spearman_vs_realized  spearman_difference_vs_opta  ci_95_lower  ci_95_upper  opta_reliably_better
           BASELINE    236              0.416620                       0.644052                   0.486618                    -0.069545    -0.119346    -0.021897                  True
        BOOST_BLIND    236              0.413372                       0.641771                   0.486618                    -0.072722    -0.120527    -0.027208                  True
BOOST_DOMESTIC_FORM    236              0.416620                       0.644052                   0.486618                    -0.069545    -0.119346    -0.021897                  True
```

## Ranking belirsizligi

```text
                arm                   metric                method  mean_difference  ci_95_lower  ci_95_upper  reliable_improvement  reliable_harm
        BOOST_BLIND seed_spearman_difference conservative_envelope         0.000509    -0.005165     0.006036                 False          False
BOOST_DOMESTIC_FORM seed_spearman_difference conservative_envelope         0.000293     0.000000     0.001110                 False          False
```

## Match-loss no-harm belirsizligi

```text
                arm competition       metric                method  matches clusters  mean_difference  ci_95_lower  ci_95_upper  reliable_improvement  reliable_harm
        BOOST_BLIND         ALL    brier_1x2 conservative_envelope     4884     <NA>        -0.000448    -0.000928     0.000045                 False          False
        BOOST_BLIND         ALL log_loss_1x2 conservative_envelope     4884     <NA>        -0.000675    -0.001372     0.000045                 False          False
        BOOST_BLIND         UCL    brier_1x2 conservative_envelope     1384     <NA>        -0.000199    -0.000511     0.000011                 False          False
        BOOST_BLIND         UCL log_loss_1x2 conservative_envelope     1384     <NA>        -0.000310    -0.000811     0.000021                 False          False
        BOOST_BLIND         UEL    brier_1x2 conservative_envelope     1427     <NA>        -0.000420    -0.001484     0.000634                 False          False
        BOOST_BLIND         UEL log_loss_1x2 conservative_envelope     1427     <NA>        -0.000732    -0.002274     0.000787                 False          False
        BOOST_BLIND        UECL    brier_1x2 conservative_envelope     2073     <NA>        -0.000634    -0.001590     0.000339                 False          False
        BOOST_BLIND        UECL log_loss_1x2 conservative_envelope     2073     <NA>        -0.000880    -0.002198     0.000499                 False          False
BOOST_DOMESTIC_FORM         ALL    brier_1x2 conservative_envelope     4884     <NA>        -0.000065    -0.000236     0.000039                 False          False
BOOST_DOMESTIC_FORM         ALL log_loss_1x2 conservative_envelope     4884     <NA>        -0.000122    -0.000364     0.000040                 False          False
BOOST_DOMESTIC_FORM         UCL    brier_1x2 conservative_envelope     1384     <NA>         0.000000     0.000000     0.000000                 False          False
BOOST_DOMESTIC_FORM         UCL log_loss_1x2 conservative_envelope     1384     <NA>         0.000000     0.000000     0.000000                 False          False
BOOST_DOMESTIC_FORM         UEL    brier_1x2 conservative_envelope     1427     <NA>        -0.000233    -0.000836     0.000028                 False          False
BOOST_DOMESTIC_FORM         UEL log_loss_1x2 conservative_envelope     1427     <NA>        -0.000406    -0.001299     0.000044                 False          False
BOOST_DOMESTIC_FORM        UECL    brier_1x2 conservative_envelope     2073     <NA>         0.000006    -0.000166     0.000182                 False          False
BOOST_DOMESTIC_FORM        UECL log_loss_1x2 conservative_envelope     2073     <NA>        -0.000008    -0.000239     0.000209                 False          False
```

## Turnuva segmentleri

```text
                arm competition  matches  teams  brier_1x2  log_loss_1x2  seed_spearman  seed_pairwise_accuracy  delta_vs_baseline_brier_1x2  delta_vs_baseline_log_loss_1x2  delta_vs_baseline_seed_spearman  delta_vs_baseline_seed_pairwise_accuracy
           BASELINE         UCL     1384    478   0.552191      0.934552       0.463966                0.662148                     0.000000                        0.000000                         0.000000                                  0.000000
           BASELINE         UEL     1427    538   0.575728      0.969740       0.444736                0.652215                     0.000000                        0.000000                         0.000000                                  0.000000
           BASELINE        UECL     2073    864   0.581568      0.978753       0.436591                0.651473                     0.000000                        0.000000                         0.000000                                  0.000000
        BOOST_BLIND         UCL     1384    478   0.551992      0.934242       0.464225                0.662254                    -0.000199                       -0.000310                         0.000258                                  0.000106
        BOOST_BLIND         UEL     1427    538   0.575308      0.969008       0.447660                0.653110                    -0.000420                       -0.000732                         0.002924                                  0.000895
        BOOST_BLIND        UECL     2073    864   0.580934      0.977873       0.436436                0.651218                    -0.000634                       -0.000880                        -0.000155                                 -0.000255
BOOST_DOMESTIC_FORM         UCL     1384    478   0.552191      0.934552       0.463966                0.662148                     0.000000                        0.000000                         0.000000                                  0.000000
BOOST_DOMESTIC_FORM         UEL     1427    538   0.575495      0.969334       0.446885                0.652874                    -0.000233                       -0.000406                         0.002150                                  0.000659
BOOST_DOMESTIC_FORM        UECL     2073    864   0.581574      0.978745       0.436975                0.651592                     0.000006                       -0.000008                         0.000385                                  0.000119
```

## Exposure ve etkilenen mac segmentleri

```text
                arm exposure_band  teams  affected_teams  mean_boost  maximum_boost  seed_spearman
           BASELINE             0    175               0    0.000000       0.000000       0.458330
           BASELINE      (0,0.25]    142               0    0.000000       0.000000       0.377293
           BASELINE   (0.25,0.50]    231               0    0.000000       0.000000       0.522438
           BASELINE   (0.50,0.75]    267               0    0.000000       0.000000       0.385029
           BASELINE   (0.75,1.00]    598               0    0.000000       0.000000       0.458142
        BOOST_BLIND             0    175               0    0.000000       0.000000       0.458330
        BOOST_BLIND      (0,0.25]    142              75   10.275743      89.134021       0.383401
        BOOST_BLIND   (0.25,0.50]    231              86   17.727234     127.476013       0.498178
        BOOST_BLIND   (0.50,0.75]    267              40    7.049854     150.000000       0.382999
        BOOST_BLIND   (0.75,1.00]    598              13    1.034425     111.597525       0.459606
BOOST_DOMESTIC_FORM             0    175               0    0.000000       0.000000       0.458330
BOOST_DOMESTIC_FORM      (0,0.25]    142               0    0.000000       0.000000       0.377293
BOOST_DOMESTIC_FORM   (0.25,0.50]    231               0    0.000000       0.000000       0.522438
BOOST_DOMESTIC_FORM   (0.50,0.75]    267               1    0.561798     150.000000       0.386131
BOOST_DOMESTIC_FORM   (0.75,1.00]    598               0    0.000000       0.000000       0.458142
```

## Mudahale kutlesi ve Domestic Surprise etkilesimi

```text
                arm  team_seasons  affected_team_seasons  affected_share  mean_boost_all  median_boost_all  mean_boost_affected  median_boost_affected  p90_boost  p95_boost  maximum_boost  cap_hits  negative_surprise_affected  minus_30_surprise_affected
           BASELINE          1413                      0        0.000000        0.000000               0.0             0.000000               0.000000   0.000000    0.00000            0.0         0                           0                           0
        BOOST_BLIND          1413                    186        0.131635        5.700668               0.0            43.306686              38.572496  19.273636   49.25433          150.0         1                          43                           5
BOOST_DOMESTIC_FORM          1413                      1        0.000708        0.106157               0.0           150.000000             150.000000   0.000000    0.00000          150.0         1                           1                           0
```

```text
                arm    segment  matches  brier_1x2  log_loss_1x2  accuracy_1x2
           BASELINE UNAFFECTED     4884   0.571537      0.963594      0.549959
        BOOST_BLIND   AFFECTED      778   0.593682      0.994590      0.520566
        BOOST_BLIND UNAFFECTED     4106   0.566808      0.956918      0.555772
BOOST_DOMESTIC_FORM   AFFECTED       16   0.481219      0.851587      0.750000
BOOST_DOMESTIC_FORM UNAFFECTED     4868   0.571768      0.963840      0.549507
```

## Forward ranking no-harm kontrolu

Bu tablo ileri sezon siralamasini bir terfi hedefi olarak degil, olasi
yan etki kontrolu olarak gosterir.

```text
competition  groups  team_weight  ranking_score  pairwise_accuracy                 arm  delta_vs_baseline_ranking_score  delta_vs_baseline_pairwise_accuracy
        ALL      15         1158       0.470420           0.658813            BASELINE                         0.000000                             0.000000
        UCL       5          353       0.479238           0.671020            BASELINE                         0.000000                             0.000000
       UECL       5          554       0.413378           0.644467            BASELINE                         0.000000                             0.000000
        UEL       5          251       0.583920           0.704640            BASELINE                         0.000000                             0.000000
        ALL      15         1158       0.470412           0.658854         BOOST_BLIND                        -0.000008                             0.000041
        UCL       5          353       0.478808           0.671102         BOOST_BLIND                        -0.000429                             0.000082
       UECL       5          554       0.414388           0.644665         BOOST_BLIND                         0.001010                             0.000198
        UEL       5          251       0.582260           0.703841         BOOST_BLIND                        -0.001660                            -0.000799
        ALL      15         1158       0.470324           0.658792 BOOST_DOMESTIC_FORM                        -0.000096                            -0.000021
        UCL       5          353       0.479008           0.670938 BOOST_DOMESTIC_FORM                        -0.000230                            -0.000082
       UECL       5          554       0.413301           0.644434 BOOST_DOMESTIC_FORM                        -0.000077                            -0.000033
        UEL       5          251       0.583970           0.704799 BOOST_DOMESTIC_FORM                         0.000050                             0.000160
```

## Long-history fizibilitesi

```text
               arm              status                                                     required_signal available_local_source                                                                                                                                                                                                                  reason                                                                                                                     minimum_future_contract
BOOST_LONG_HISTORY UNAVAILABLE_NOT_RUN pre-season club reputation older than the active five-season window                   NONE The repository has no dated 10-20 year UEFA/Opta club-rating snapshots or audited long-run title table. Reusing 2013-2026 domestic goals would duplicate BOOST_DOMESTIC_FORM rather than create independent reputation. dated pre-season snapshot, stable club_id crosswalk, source license, coverage audit and at least two historical outer-fold training seasons
```

## Safety

```text
                             check  passed                                                         observed                                                      requirement
       unique_seed_team_season_arm    True                                                                0                                                  zero duplicates
       zero_exposure_never_boosted    True                                                                0                                                             zero
     boost_non_negative_and_capped    True                                                            150.0                                                      [0,150] Elo
         ineligible_rows_unchanged    True                                                              0.0                                                            0 Elo
production_contract_hash_unchanged    True 6db3ea2852792adbb1585fbd7871cf05ee109999a9bc6621fa7e8c2a7c2b4589 6db3ea2852792adbb1585fbd7871cf05ee109999a9bc6621fa7e8c2a7c2b4589
           power_zero_sum_BASELINE    True                                                              0.0                                                           <=1e-9
        power_zero_sum_BOOST_BLIND    True                                                              0.0                                                           <=1e-9
power_zero_sum_BOOST_DOMESTIC_FORM    True                                                              0.0                                                           <=1e-9
```

Bu calisma development-window kanitidir; 2026/27 prospective sonuc yerine gecmez.
Production contract yalniz hash'lendi ve degistirilmedi.
