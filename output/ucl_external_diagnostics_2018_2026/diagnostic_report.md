# UCL External Elo Gap Diagnostics

## Question

Does the UCL gap against ClubElo come from AO First Elo, normal match updates,
or cross-season power carry? All variants use identical exact-date test matches.

UCL unseen paired matches: 171
Decision: `KEEP_CORE_DIAGNOSE_INITIAL_RATING_AND_UCL_COVERAGE`

## Model summary

```text
                    model  matches    brier  log_loss
         clubelo_external      171 0.156801  0.610759
        dynamic_carry_050      171 0.169083  0.636089
     dynamic_season_reset      171 0.170063  0.639198
dynamic_carry_085_current      171 0.171128  0.641829
        dynamic_carry_100      171 0.173468  0.649890
        static_start_only      171 0.175531  0.651893
```

## Isolated layer differences

Negative Brier difference favors the challenger.

```text
        comparison                challenger             baseline  matches  mean_brier_difference  ci_95_lower  ci_95_upper  folds_won  folds  reliable_improvement  reliable_harm
    dynamic_update      dynamic_season_reset    static_start_only      171              -0.005469    -0.015364     0.004455          3      5                 False          False
carry_050_vs_reset         dynamic_carry_050 dynamic_season_reset      171              -0.000979    -0.005559     0.003755          4      5                 False          False
carry_085_vs_reset dynamic_carry_085_current dynamic_season_reset      171               0.001066    -0.008024     0.010643          4      5                 False          False
carry_100_vs_reset         dynamic_carry_100 dynamic_season_reset      171               0.003405    -0.008470     0.015349          3      5                 False          False
current_vs_clubelo dynamic_carry_085_current     clubelo_external      171               0.014327     0.001008     0.027294          1      5                 False           True
```

## Season detail

```text
test_season                     model  matches    brier  log_loss
    2020/21         static_start_only       32 0.125488  0.579655
    2020/21      dynamic_season_reset       32 0.132718  0.597779
    2020/21         dynamic_carry_050       32 0.129074  0.586536
    2020/21 dynamic_carry_085_current       32 0.126872  0.579307
    2020/21         dynamic_carry_100       32 0.126050  0.576636
    2020/21          clubelo_external       32 0.111803  0.552175
    2021/22         static_start_only       35 0.175739  0.673163
    2021/22      dynamic_season_reset       35 0.183709  0.688238
    2021/22         dynamic_carry_050       35 0.180079  0.679515
    2021/22 dynamic_carry_085_current       35 0.179531  0.677577
    2021/22         dynamic_carry_100       35 0.180167  0.678794
    2021/22          clubelo_external       35 0.161383  0.639106
    2022/23         static_start_only       28 0.169921  0.574522
    2022/23      dynamic_season_reset       28 0.145513  0.522035
    2022/23         dynamic_carry_050       28 0.144049  0.516464
    2022/23 dynamic_carry_085_current       28 0.144718  0.516400
    2022/23         dynamic_carry_100       28 0.145763  0.518368
    2022/23          clubelo_external       28 0.150642  0.536059
    2023/24         static_start_only       36 0.153273  0.604177
    2023/24      dynamic_season_reset       36 0.150091  0.599465
    2023/24         dynamic_carry_050       36 0.145665  0.591361
    2023/24 dynamic_carry_085_current       36 0.143607  0.592954
    2023/24         dynamic_carry_100       36 0.144256  0.601955
    2023/24          clubelo_external       36 0.129210  0.557751
    2024/25         static_start_only       40 0.239342  0.788176
    2024/25      dynamic_season_reset       40 0.223156  0.747195
    2024/25         dynamic_carry_050       40 0.230070  0.761727
    2024/25 dynamic_carry_085_current       40 0.242436  0.792353
    2024/25         dynamic_carry_100       40 0.251224  0.818409
    2024/25          clubelo_external       40 0.217934  0.732820
```

## Phase detail

```text
     phase                     model  matches    brier  log_loss
  KNOCKOUT         static_start_only       51 0.184396  0.681713
  KNOCKOUT      dynamic_season_reset       51 0.181798  0.675203
  KNOCKOUT         dynamic_carry_050       51 0.186328  0.683742
  KNOCKOUT dynamic_carry_085_current       51 0.194452  0.704374
  KNOCKOUT         dynamic_carry_100       51 0.200324  0.722575
  KNOCKOUT          clubelo_external       51 0.165216  0.640500
MAIN_STAGE         static_start_only      106 0.181969  0.661671
MAIN_STAGE      dynamic_season_reset      106 0.175304  0.646205
MAIN_STAGE         dynamic_carry_050      106 0.172758  0.639896
MAIN_STAGE dynamic_carry_085_current      106 0.173039  0.641297
MAIN_STAGE         dynamic_carry_100      106 0.174350  0.646401
MAIN_STAGE          clubelo_external      106 0.159898  0.611554
QUALIFYING         static_start_only       14 0.094496  0.469230
QUALIFYING      dynamic_season_reset       14 0.087629  0.454979
QUALIFYING         dynamic_carry_050       14 0.078446  0.433672
QUALIFYING dynamic_carry_085_current       14 0.071694  0.418006
QUALIFYING         dynamic_carry_100       14 0.068959  0.411525
QUALIFYING          clubelo_external       14 0.102700  0.496400
```

## Rating alignment

```text
                    model  team_match_observations  rating_spearman_vs_clubelo  match_difference_spearman_vs_clubelo
        static_start_only                      342                    0.812533                              0.757539
     dynamic_season_reset                      342                    0.875205                              0.840576
        dynamic_carry_050                      342                    0.887870                              0.860772
dynamic_carry_085_current                      342                    0.888094                              0.860835
        dynamic_carry_100                      342                    0.887303                              0.856726
```

## Largest club rank gaps

Positive rank gap means AO ranks the club higher than ClubElo within the
covered UCL season sample.

```text
 season    team_name  observations  mean_ao_live_rating  mean_clubelo_rating  ao_rank_percentile  clubelo_rank_percentile  ao_minus_clubelo_rank_percentile  absolute_rank_percentile_gap
2023/24        Lazio             6           769.837439          1771.228333            0.250000                 0.562500                         -0.312500                      0.312500
2022/23     Juventus             2           880.626016          1759.515000            0.733333                 0.466667                          0.266667                      0.266667
2024/25      Benfica             4           861.180127          1786.732500            0.687500                 0.437500                          0.250000                      0.250000
2024/25      Bologna             3           728.371637          1757.746667            0.125000                 0.375000                         -0.250000                      0.250000
2022/23         Ajax             4           873.967623          1867.545000            0.666667                 0.866667                         -0.200000                      0.200000
2022/23      Sevilla             2           808.049271          1773.590000            0.400000                 0.600000                         -0.200000                      0.200000
2024/25  Aston Villa             8           800.321208          1799.952500            0.312500                 0.500000                         -0.187500                      0.187500
2024/25  Club Brugge             8           833.551882          1717.018750            0.437500                 0.250000                          0.187500                      0.187500
2023/24      Arsenal             6           854.548621          1941.471667            0.687500                 0.875000                         -0.187500                      0.187500
2024/25      Arsenal             5           906.407881          1981.632000            0.812500                 1.000000                         -0.187500                      0.187500
2023/24      Sevilla             4           831.365667          1736.505000            0.625000                 0.437500                          0.187500                      0.187500
2023/24      Benfica             2           882.555274          1786.945000            0.875000                 0.687500                          0.187500                      0.187500
2024/25  FC Salzburg             2           755.062217          1566.975000            0.250000                 0.062500                          0.187500                      0.187500
2021/22     Juventus             4           934.181349          1832.525000            0.833333                 0.666667                          0.166667                      0.166667
2021/22     Atalanta             2           821.060307          1831.175000            0.444444                 0.611111                         -0.166667                      0.166667
2021/22      Benfica             8           847.400858          1764.006250            0.555556                 0.388889                          0.166667                      0.166667
2021/22   Villarreal             8           899.730210          1815.553750            0.722222                 0.555556                          0.166667                      0.166667
2020/21      Chelsea             6           899.049138          1864.328333            0.846154                 0.692308                          0.153846                      0.153846
2020/21  Real Madrid             6           888.002247          1924.448333            0.692308                 0.846154                         -0.153846                      0.153846
2020/21 FK Krasnodar             4           768.863369          1562.552500            0.230769                 0.076923                          0.153846                      0.153846
```

## Favorite disagreement

```text
 favorite_relation  matches  ao_brier  clubelo_brier  ao_minus_clubelo_brier
     same_favorite      145  0.174155       0.161400                0.012755
different_favorite       26  0.154249       0.131153                0.023096
```

## Guardrails

```text
                    model  power_carry  k_factor  max_abs_rating_change  minimum_start_end_rank_correlation  guardrail_pass
        static_start_only         0.00       0.0               0.000000                            1.000000            True
     dynamic_season_reset         0.00      28.0             127.912969                            0.937454            True
        dynamic_carry_050         0.50      28.0             127.912969                            0.928640            True
dynamic_carry_085_current         0.85      28.0             136.346657                            0.907974            True
        dynamic_carry_100         1.00      28.0             152.223516                            0.888467            True
```

## Interpretation contract

- ClubElo coverage is selective and mostly contains established clubs.
- This run diagnoses AO layers; it does not tune them on ClubElo ratings.
- A layer is revised only when paired match outcomes show reliable harm.
- Final production proof still requires a future untouched season.
