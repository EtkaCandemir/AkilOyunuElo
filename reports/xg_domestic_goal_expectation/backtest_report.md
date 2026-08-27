# xG Form Terimi Domestic Attack/Defence Uzerinde

Onceki kosu xG form terimini **Elo-only** bir tabana ekledi ve xG'nin gol
kontrolunu gectigini gosterdi. Ama o taban reponun mevcut en iyi skor
kolundan basitti. Bu kosu ayni terimi
`DOMESTIC_ATTACK_DEFENCE_POISSON` kolunun **uzerine** koyar, yerine degil.

`DOMESTIC_AD_GOALS_FORM` kontroldur: onsuz xG kolundaki bir kazanc,
xG'nin mi yoksa iki ek parametrenin mi katkisi oldugu ayirt edilemezdi.

Egitim penceresi: `full`. Fold sayisi: `4`. Kol basina mac: `3528`.

## Dogrulama kapilari

```text
                             gate  passed    observed                                                                                                                        requirement
                three_arms_scored    True    3.000000                                                                                  Domestic baseline, goals control and xG candidate
            arms_share_one_sample    True 3528.000000                                                                                    Every arm scores the identical held-out matches
            arms_share_every_fold    True    4.000000                                                                                                      No arm may skip or add a fold
         baseline_carries_no_form    True    0.000000                                                                                      The baseline stays the published domestic arm
 selection_never_sees_test_season    True    4.000000                                                                         Training and inner validation close before the test season
arms_share_domestic_configuration    True    2.000000                                                                              One domestic state per fold, shared by all three arms
              rates_within_bounds    True    4.500000                                                                                     Rates stay inside the production lambda window
    xg_form_is_sparser_than_goals    True    0.587868                                                                              Documented coverage gap must be visible in the sample
   baseline_matches_published_arm    True    0.000061 Baseline within 0.01 nats of the published domestic arm on the shared matches, so the candidate is not beating a weakened stand-in
```

## Taban uzlastirmasi

Bu kosunun tabani yayinlanmis domestic kolunun zayiflatilmis bir
kopyasi olmamalidir; asagidaki tablo ikisini ayni maclarda karsilastirir.

```text
                      scope  matches  exact_score_nll  difference_vs_this_baseline
       repo_arm_full_sample     4884         3.016594                          n/a
     repo_arm_shared_sample     3528         3.021437                     0.000061
this_baseline_shared_sample     3528         3.021498                     0.000000
```

## Kol ozeti

```text
                   arm  matches  exact_score_nll  total_goals_error  over_2_5_brier  btts_brier  exact_score_nll_vs_baseline  total_goals_error_vs_baseline  over_2_5_brier_vs_baseline  btts_brier_vs_baseline
           DOMESTIC_AD     3528         3.021498           1.383476        0.246052    0.249292                     0.000000                       0.000000                    0.000000                0.000000
DOMESTIC_AD_GOALS_FORM     3528         3.021068           1.383251        0.246274    0.249003                    -0.000430                      -0.000225                    0.000223               -0.000289
   DOMESTIC_AD_XG_FORM     3528         3.019349           1.381071        0.245838    0.248906                    -0.002149                      -0.002405                   -0.000213               -0.000386
```

## Segment kirilimi (domestic tabana karsi)

```text
                   arm          segment  matches  exact_score_nll_vs_baseline  total_goals_error_vs_baseline  over_2_5_brier_vs_baseline  btts_brier_vs_baseline
DOMESTIC_AD_GOALS_FORM              ALL     3528                    -0.000430                      -0.000225                    0.000223               -0.000289
DOMESTIC_AD_GOALS_FORM       PHASE:MAIN     1876                    -0.001721                      -0.001590                    0.000071               -0.000219
DOMESTIC_AD_GOALS_FORM PHASE:QUALIFYING     1652                     0.001036                       0.001325                    0.000396               -0.000369
DOMESTIC_AD_GOALS_FORM       XG_PRESENT     2074                    -0.001308                      -0.001316                    0.000058               -0.000219
DOMESTIC_AD_GOALS_FORM        XG_ABSENT     1454                     0.000822                       0.001331                    0.000459               -0.000389
   DOMESTIC_AD_XG_FORM              ALL     3528                    -0.002149                      -0.002405                   -0.000213               -0.000386
   DOMESTIC_AD_XG_FORM       PHASE:MAIN     1876                    -0.006106                      -0.005032                   -0.000706               -0.000487
   DOMESTIC_AD_XG_FORM PHASE:QUALIFYING     1652                     0.002345                       0.000578                    0.000346               -0.000271
   DOMESTIC_AD_XG_FORM       XG_PRESENT     2074                    -0.005104                      -0.004287                   -0.000587               -0.000492
   DOMESTIC_AD_XG_FORM        XG_ABSENT     1454                     0.002066                       0.000279                    0.000320               -0.000234
```

## Fold bazinda

```text
                   arm test_season  matches  exact_score_nll  total_goals_error  over_2_5_brier  btts_brier  exact_score_nll_vs_baseline  total_goals_error_vs_baseline  over_2_5_brier_vs_baseline  btts_brier_vs_baseline
           DOMESTIC_AD     2022/23      804         2.937213           1.339594        0.247291    0.252099                     0.000000                       0.000000                    0.000000                0.000000
DOMESTIC_AD_GOALS_FORM     2022/23      804         2.935429           1.338828        0.247094    0.250986                    -0.001783                      -0.000766                   -0.000197               -0.001113
   DOMESTIC_AD_XG_FORM     2022/23      804         2.933693           1.337857        0.247117    0.251780                    -0.003520                      -0.001737                   -0.000174               -0.000319
           DOMESTIC_AD     2023/24      806         3.005150           1.373667        0.243885    0.247081                     0.000000                       0.000000                    0.000000                0.000000
DOMESTIC_AD_GOALS_FORM     2023/24      806         3.008027           1.374146        0.244409    0.247836                     0.002877                       0.000479                    0.000524                0.000755
   DOMESTIC_AD_XG_FORM     2023/24      806         3.008137           1.374692        0.244791    0.247289                     0.002987                       0.001024                    0.000906                0.000208
           DOMESTIC_AD     2024/25      957         3.072858           1.413658        0.246277    0.247792                     0.000000                       0.000000                    0.000000                0.000000
DOMESTIC_AD_GOALS_FORM     2024/25      957         3.071991           1.412888        0.246684    0.247658                    -0.000867                      -0.000770                    0.000407               -0.000134
   DOMESTIC_AD_XG_FORM     2024/25      957         3.069192           1.407966        0.245222    0.246962                    -0.003667                      -0.005692                   -0.001055               -0.000830
           DOMESTIC_AD     2025/26      961         3.054579           1.398360        0.246608    0.250292                     0.000000                       0.000000                    0.000000                0.000000
DOMESTIC_AD_GOALS_FORM     2025/26      961         3.052944           1.398539        0.246746    0.249661                    -0.001635                       0.000179                    0.000139               -0.000632
   DOMESTIC_AD_XG_FORM     2025/26      961         3.050782           1.395793        0.246261    0.249794                    -0.003797                      -0.002567                   -0.000346               -0.000498
```

## Nested secimler

```text
 fold test_season inner_validation_season                    arm          domestic_candidate_key  selected_l2_strength  attack_coefficient  defence_coefficient  venue_coefficient  form_attack  form_defence
    1     2022/23                 2021/22            DOMESTIC_AD lr0.02_carry0.9_shrink10_venue1                  10.0            0.061972             0.052729           0.074274     0.000000      0.000000
    1     2022/23                 2021/22 DOMESTIC_AD_GOALS_FORM lr0.02_carry0.9_shrink10_venue1                  10.0            0.051734             0.057861           0.073225     0.109121      0.000000
    1     2022/23                 2021/22    DOMESTIC_AD_XG_FORM lr0.02_carry0.9_shrink10_venue1                  10.0            0.055508             0.053691           0.071811     0.132499      0.000000
    2     2023/24                 2022/23            DOMESTIC_AD lr0.02_carry0.9_shrink10_venue1                   0.1            0.083653             0.065224           0.068820     0.000000      0.000000
    2     2023/24                 2022/23 DOMESTIC_AD_GOALS_FORM lr0.02_carry0.9_shrink10_venue1                   0.1            0.073283             0.070503           0.067728     0.103852      0.000000
    2     2023/24                 2022/23    DOMESTIC_AD_XG_FORM lr0.02_carry0.9_shrink10_venue1                   0.1            0.074184             0.067032           0.067441     0.165906      0.012941
    3     2024/25                 2023/24            DOMESTIC_AD lr0.02_carry0.9_shrink10_venue0                  10.0            0.103406             0.080548           0.000000     0.000000      0.000000
    3     2024/25                 2023/24 DOMESTIC_AD_GOALS_FORM lr0.02_carry0.9_shrink10_venue0                  10.0            0.094828             0.084173           0.000000     0.083574      0.000000
    3     2024/25                 2023/24    DOMESTIC_AD_XG_FORM lr0.02_carry0.9_shrink10_venue0                  10.0            0.096486             0.081494           0.000000     0.117855      0.000000
    4     2025/26                 2024/25            DOMESTIC_AD lr0.02_carry0.9_shrink10_venue1                   0.1            0.093762             0.054068           0.026437     0.000000      0.000000
    4     2025/26                 2024/25 DOMESTIC_AD_GOALS_FORM lr0.02_carry0.9_shrink10_venue1                   0.1            0.085117             0.056998           0.027619     0.078913      0.000000
    4     2025/26                 2024/25    DOMESTIC_AD_XG_FORM lr0.02_carry0.9_shrink10_venue1                   0.1            0.084816             0.055145           0.027397     0.138506      0.000000
```

## Fold kazanimlari

```text
                   arm              reference            metric  folds  wins record
DOMESTIC_AD_GOALS_FORM            DOMESTIC_AD   exact_score_nll      4     3    3/4
DOMESTIC_AD_GOALS_FORM            DOMESTIC_AD total_goals_error      4     2    2/4
DOMESTIC_AD_GOALS_FORM            DOMESTIC_AD    over_2_5_brier      4     1    1/4
DOMESTIC_AD_GOALS_FORM            DOMESTIC_AD        btts_brier      4     3    3/4
   DOMESTIC_AD_XG_FORM            DOMESTIC_AD   exact_score_nll      4     3    3/4
   DOMESTIC_AD_XG_FORM            DOMESTIC_AD total_goals_error      4     3    3/4
   DOMESTIC_AD_XG_FORM            DOMESTIC_AD    over_2_5_brier      4     3    3/4
   DOMESTIC_AD_XG_FORM            DOMESTIC_AD        btts_brier      4     3    3/4
   DOMESTIC_AD_XG_FORM DOMESTIC_AD_GOALS_FORM   exact_score_nll      4     3    3/4
   DOMESTIC_AD_XG_FORM DOMESTIC_AD_GOALS_FORM total_goals_error      4     3    3/4
   DOMESTIC_AD_XG_FORM DOMESTIC_AD_GOALS_FORM    over_2_5_brier      4     2    2/4
   DOMESTIC_AD_XG_FORM DOMESTIC_AD_GOALS_FORM        btts_brier      4     2    2/4
```

## Dependency-robust belirsizlik: tabana karsi (conservative envelope)

```text
                   arm          segment          metric  mean_difference  ci_95_lower  ci_95_upper  reliable_improvement  reliable_harm
DOMESTIC_AD_GOALS_FORM              ALL exact_score_nll        -0.000430    -0.002566     0.001641                 False          False
DOMESTIC_AD_GOALS_FORM              ALL  over_2_5_brier         0.000223    -0.000579     0.001004                 False          False
DOMESTIC_AD_GOALS_FORM       PHASE:MAIN exact_score_nll        -0.001721    -0.004722     0.001238                 False          False
DOMESTIC_AD_GOALS_FORM       PHASE:MAIN  over_2_5_brier         0.000071    -0.001044     0.001192                 False          False
DOMESTIC_AD_GOALS_FORM PHASE:QUALIFYING exact_score_nll         0.001036    -0.001994     0.004321                 False          False
DOMESTIC_AD_GOALS_FORM PHASE:QUALIFYING  over_2_5_brier         0.000396    -0.000628     0.001462                 False          False
DOMESTIC_AD_GOALS_FORM       XG_PRESENT exact_score_nll        -0.001308    -0.004222     0.001490                 False          False
DOMESTIC_AD_GOALS_FORM       XG_PRESENT  over_2_5_brier         0.000058    -0.000966     0.001081                 False          False
DOMESTIC_AD_GOALS_FORM        XG_ABSENT exact_score_nll         0.000822    -0.002301     0.004410                 False          False
DOMESTIC_AD_GOALS_FORM        XG_ABSENT  over_2_5_brier         0.000459    -0.000630     0.001730                 False          False
   DOMESTIC_AD_XG_FORM              ALL exact_score_nll        -0.002149    -0.005292     0.000502                 False          False
   DOMESTIC_AD_XG_FORM              ALL  over_2_5_brier        -0.000213    -0.001023     0.000491                 False          False
   DOMESTIC_AD_XG_FORM       PHASE:MAIN exact_score_nll        -0.006106    -0.009441    -0.002732                  True          False
   DOMESTIC_AD_XG_FORM       PHASE:MAIN  over_2_5_brier        -0.000706    -0.001897     0.000405                 False          False
   DOMESTIC_AD_XG_FORM PHASE:QUALIFYING exact_score_nll         0.002345    -0.000732     0.005494                 False          False
   DOMESTIC_AD_XG_FORM PHASE:QUALIFYING  over_2_5_brier         0.000346    -0.000473     0.001296                 False          False
   DOMESTIC_AD_XG_FORM       XG_PRESENT exact_score_nll        -0.005104    -0.008347    -0.001876                  True          False
   DOMESTIC_AD_XG_FORM       XG_PRESENT  over_2_5_brier        -0.000587    -0.001654     0.000457                 False          False
   DOMESTIC_AD_XG_FORM        XG_ABSENT exact_score_nll         0.002066    -0.001321     0.005801                 False          False
   DOMESTIC_AD_XG_FORM        XG_ABSENT  over_2_5_brier         0.000320    -0.000559     0.001448                 False          False
```

## Dependency-robust belirsizlik: xG kolu gol kontrolune karsi

Iki kol da ayni iki ek parametreyi tasir. Aralarindaki guvenilir bir
fark, yalniz form teriminin okundugu kaynaktan gelebilir.

```text
         segment          metric  mean_difference  ci_95_lower  ci_95_upper  reliable_improvement  reliable_harm
             ALL exact_score_nll        -0.001719    -0.004352     0.000659                 False          False
             ALL  over_2_5_brier        -0.000436    -0.001111     0.000278                 False          False
      PHASE:MAIN exact_score_nll        -0.004385    -0.007279    -0.001362                  True          False
      PHASE:MAIN  over_2_5_brier        -0.000776    -0.001696     0.000111                 False          False
PHASE:QUALIFYING exact_score_nll         0.001309    -0.002695     0.004509                 False          False
PHASE:QUALIFYING  over_2_5_brier        -0.000050    -0.001192     0.001121                 False          False
      XG_PRESENT exact_score_nll        -0.003796    -0.006653    -0.001064                  True          False
      XG_PRESENT  over_2_5_brier        -0.000645    -0.001463     0.000202                 False          False
       XG_ABSENT exact_score_nll         0.001244    -0.003015     0.004747                 False          False
       XG_ABSENT  over_2_5_brier        -0.000139    -0.001440     0.001220                 False          False
```

## Karar girdisi

- xG kolu tabani exact-score'da geciyor mu: `True`.
- xG kolu gol kontrolunu geciyor mu: `True`.
- Exact-score fold kaydi: `{'DOMESTIC_AD_GOALS_FORM_vs_DOMESTIC_AD': '3/4', 'DOMESTIC_AD_XG_FORM_vs_DOMESTIC_AD': '3/4', 'DOMESTIC_AD_XG_FORM_vs_DOMESTIC_AD_GOALS_FORM': '3/4'}`.
- Guvenilir iyilesme sayisi: `4`.
- Guvenilir zarar sayisi: `0`.
- Production parametresi degisti mi: `False`.

Karar urun tarafina aittir; bu belge yalniz kanit uretir.
