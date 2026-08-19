# Genellestirilmis Yerel Kupa Katkisi Backtest

Aktif model lig ve kupa basarisini `max` ile birlestirir; bu kupayi bir
*taban* yapar, katki yapmaz. Lig siralamasi kupa tabaninin uzerinde olan
her kupa sahibi kupasindan sifir kredi alir ve duble bonusu yalniz
sampiyon-ve-kupa ciftinde calisir. Bu kosu su tek parametreli
genellestirmeyi olcer:

```text
Achievement = min(cap, max(L, C) + weight * min(L, C))
```

- Sampiyon-esdeger agirlik: `0.129032`
  (mevcut duble bonus buyuklugunu korur, ayni mantigi herkese genisletir).
- Aday agirliklar: `[0.0, 0.05, 0.08, 0.1, 0.13, 0.15, 0.2, 0.25, 0.3]`.
- Gelistirme penceresi: `2018/19-2025/26`.
- Degerlendirme sezonlari: `2020/21, 2021/22, 2022/23, 2023/24, 2024/25, 2025/26`.
- Bu kosu production parametresi degistirmez.

## Yapisal dogrulama kapilari

```text
                            gate  passed  observed                                           requirement
                   seed_coverage    True      1887         Every active team-season has a candidate seed
       inert_outside_cup_winners    True       0.0 Teams without a domestic cup keep the production seed
zero_weight_removes_double_bonus    True -3.961573         weight=0 cannot raise a champion-and-cup seed
              monotone_in_weight    True      True       Cup-winner seeds never fall as the weight rises
```

## Etkilenen kutle

Katman kupa sahibi olmayan takimda tanim geregi inerttir. Gercekten
hareket eden takim-sezon sayisi:

```text
 weight  team_seasons  cup_winners  moved_team_seasons  moved_share  mean_abs_seed_delta  max_abs_seed_delta
   0.00          1887          363                  78     0.041335             8.004415           26.185322
   0.05          1887          363                 363     0.192369             5.557097           17.458782
   0.08          1887          363                 363     0.192369             7.859382           27.934052
   0.10          1887          363                 363     0.192369             9.394238           34.917565
   0.13          1887          363                 363     0.192369            11.722322           45.392834
   0.15          1887          363                 363     0.192369            13.790364           52.376347
   0.20          1887          363                 363     0.192369            18.444484           69.835130
   0.25          1887          363                 363     0.192369            22.948108           87.293912
   0.30          1887          363                 363     0.192369            27.451732          104.752694
```

## Eksen 1: Sezon basi rating

Hedef, hicbir ratingden etkilenmeyen leave-team-out schedule-adjusted
sezon performansidir.

```text
               arm  seasons  team_seasons  weighted_spearman  weighted_pairwise_accuracy
CURRENT_PRODUCTION        6          1880           0.387346                    0.633300
      CUP_WEIGHT_0        6          1880           0.388136                    0.633624
   CUP_WEIGHT_0_05        6          1880           0.387861                    0.633520
   CUP_WEIGHT_0_08        6          1880           0.387877                    0.633500
    CUP_WEIGHT_0_1        6          1880           0.387698                    0.633459
   CUP_WEIGHT_0_13        6          1880           0.387599                    0.633407
   CUP_WEIGHT_0_15        6          1880           0.387495                    0.633387
    CUP_WEIGHT_0_2        6          1880           0.387194                    0.633368
   CUP_WEIGHT_0_25        6          1880           0.386823                    0.633216
    CUP_WEIGHT_0_3        6          1880           0.386856                    0.633274
```

Eslesmis Spearman farklari (sezon icinde takim yeniden orneklemesi):

```text
                              comparison  team_seasons  spearman_difference  ci_95_lower  ci_95_upper  reliable_improvement  reliable_harm
   CUP_WEIGHT_0_minus_CURRENT_PRODUCTION          1880             0.000790    -0.000179     0.001806                 False          False
CUP_WEIGHT_0_05_minus_CURRENT_PRODUCTION          1880             0.000515    -0.000527     0.001637                 False          False
CUP_WEIGHT_0_08_minus_CURRENT_PRODUCTION          1880             0.000531    -0.000776     0.001887                 False          False
 CUP_WEIGHT_0_1_minus_CURRENT_PRODUCTION          1880             0.000352    -0.001117     0.001887                 False          False
CUP_WEIGHT_0_13_minus_CURRENT_PRODUCTION          1880             0.000253    -0.001378     0.001962                 False          False
CUP_WEIGHT_0_15_minus_CURRENT_PRODUCTION          1880             0.000149    -0.001713     0.002056                 False          False
 CUP_WEIGHT_0_2_minus_CURRENT_PRODUCTION          1880            -0.000152    -0.002617     0.002300                 False          False
CUP_WEIGHT_0_25_minus_CURRENT_PRODUCTION          1880            -0.000523    -0.003532     0.002546                 False          False
 CUP_WEIGHT_0_3_minus_CURRENT_PRODUCTION          1880            -0.000490    -0.003937     0.003003                 False          False
```

## Eksen 2: Dynamic replay tahmin kalitesi

Production kernel'i ile tam sezon replay; standart uc sinifli kayiplar.

```text
               arm  matches  brier_1x2  log_loss_1x2  accuracy_1x2
CURRENT_PRODUCTION     4884   0.572093      0.964371      0.550164
      CUP_WEIGHT_0     4884   0.572035      0.964299      0.549959
   CUP_WEIGHT_0_05     4884   0.571994      0.964233      0.549754
   CUP_WEIGHT_0_08     4884   0.571974      0.964201      0.549959
    CUP_WEIGHT_0_1     4884   0.571964      0.964184      0.550164
   CUP_WEIGHT_0_13     4884   0.571952      0.964163      0.550164
   CUP_WEIGHT_0_15     4884   0.571946      0.964152      0.550164
    CUP_WEIGHT_0_2     4884   0.571922      0.964113      0.549959
   CUP_WEIGHT_0_25     4884   0.571902      0.964082      0.550164
    CUP_WEIGHT_0_3     4884   0.571892      0.964066      0.550778
```

Fold kazanimlari:

```text
            arm  brier_fold_wins  log_loss_fold_wins  folds
   CUP_WEIGHT_0                5                   5      6
CUP_WEIGHT_0_05                6                   6      6
CUP_WEIGHT_0_08                6                   6      6
 CUP_WEIGHT_0_1                5                   5      6
CUP_WEIGHT_0_13                5                   5      6
CUP_WEIGHT_0_15                5                   4      6
 CUP_WEIGHT_0_2                4                   4      6
CUP_WEIGHT_0_25                4                   4      6
 CUP_WEIGHT_0_3                4                   4      6
```

Dependency-robust belirsizlik:

```text
            arm       metric                method  matches clusters  mean_difference  ci_95_lower  ci_95_upper  reliable_improvement  reliable_harm
   CUP_WEIGHT_0    brier_1x2          tie_or_match     4884     1612        -0.000057    -0.000124     0.000021                 False          False
   CUP_WEIGHT_0    brier_1x2           team_season     4884     1413        -0.000057    -0.000110    -0.000006                  True          False
   CUP_WEIGHT_0    brier_1x2        calendar_month     4884       65        -0.000057    -0.000118     0.000005                 False          False
   CUP_WEIGHT_0    brier_1x2 conservative_envelope     4884     <NA>        -0.000057    -0.000124     0.000021                 False          False
   CUP_WEIGHT_0 log_loss_1x2          tie_or_match     4884     1612        -0.000072    -0.000159     0.000038                 False          False
   CUP_WEIGHT_0 log_loss_1x2           team_season     4884     1413        -0.000072    -0.000142    -0.000004                  True          False
   CUP_WEIGHT_0 log_loss_1x2        calendar_month     4884       65        -0.000072    -0.000156     0.000014                 False          False
   CUP_WEIGHT_0 log_loss_1x2 conservative_envelope     4884     <NA>        -0.000072    -0.000159     0.000038                 False          False
CUP_WEIGHT_0_05    brier_1x2          tie_or_match     4884     1612        -0.000099    -0.000191     0.000017                 False          False
CUP_WEIGHT_0_05    brier_1x2           team_season     4884     1413        -0.000099    -0.000165    -0.000035                  True          False
CUP_WEIGHT_0_05    brier_1x2        calendar_month     4884       65        -0.000099    -0.000177    -0.000023                  True          False
CUP_WEIGHT_0_05    brier_1x2 conservative_envelope     4884     <NA>        -0.000099    -0.000191     0.000017                 False          False
CUP_WEIGHT_0_05 log_loss_1x2          tie_or_match     4884     1612        -0.000138    -0.000261     0.000026                 False          False
CUP_WEIGHT_0_05 log_loss_1x2           team_season     4884     1413        -0.000138    -0.000230    -0.000051                  True          False
CUP_WEIGHT_0_05 log_loss_1x2        calendar_month     4884       65        -0.000138    -0.000251    -0.000029                  True          False
CUP_WEIGHT_0_05 log_loss_1x2 conservative_envelope     4884     <NA>        -0.000138    -0.000261     0.000026                 False          False
CUP_WEIGHT_0_08    brier_1x2          tie_or_match     4884     1612        -0.000118    -0.000246     0.000043                 False          False
CUP_WEIGHT_0_08    brier_1x2           team_season     4884     1413        -0.000118    -0.000210    -0.000031                  True          False
CUP_WEIGHT_0_08    brier_1x2        calendar_month     4884       65        -0.000118    -0.000225    -0.000018                  True          False
CUP_WEIGHT_0_08    brier_1x2 conservative_envelope     4884     <NA>        -0.000118    -0.000246     0.000043                 False          False
CUP_WEIGHT_0_08 log_loss_1x2          tie_or_match     4884     1612        -0.000169    -0.000341     0.000057                 False          False
CUP_WEIGHT_0_08 log_loss_1x2           team_season     4884     1413        -0.000169    -0.000299    -0.000044                  True          False
CUP_WEIGHT_0_08 log_loss_1x2        calendar_month     4884       65        -0.000169    -0.000324    -0.000025                  True          False
CUP_WEIGHT_0_08 log_loss_1x2 conservative_envelope     4884     <NA>        -0.000169    -0.000341     0.000057                 False          False
 CUP_WEIGHT_0_1    brier_1x2          tie_or_match     4884     1612        -0.000129    -0.000284     0.000066                 False          False
 CUP_WEIGHT_0_1    brier_1x2           team_season     4884     1413        -0.000129    -0.000241    -0.000020                  True          False
 CUP_WEIGHT_0_1    brier_1x2        calendar_month     4884       65        -0.000129    -0.000258    -0.000008                  True          False
 CUP_WEIGHT_0_1    brier_1x2 conservative_envelope     4884     <NA>        -0.000129    -0.000284     0.000066                 False          False
 CUP_WEIGHT_0_1 log_loss_1x2          tie_or_match     4884     1612        -0.000187    -0.000392     0.000084                 False          False
 CUP_WEIGHT_0_1 log_loss_1x2           team_season     4884     1413        -0.000187    -0.000348    -0.000033                  True          False
 CUP_WEIGHT_0_1 log_loss_1x2        calendar_month     4884       65        -0.000187    -0.000375    -0.000010                  True          False
 CUP_WEIGHT_0_1 log_loss_1x2 conservative_envelope     4884     <NA>        -0.000187    -0.000392     0.000084                 False          False
CUP_WEIGHT_0_13    brier_1x2          tie_or_match     4884     1612        -0.000141    -0.000336     0.000101                 False          False
CUP_WEIGHT_0_13    brier_1x2           team_season     4884     1413        -0.000141    -0.000282    -0.000002                  True          False
CUP_WEIGHT_0_13    brier_1x2        calendar_month     4884       65        -0.000141    -0.000307     0.000011                 False          False
CUP_WEIGHT_0_13    brier_1x2 conservative_envelope     4884     <NA>        -0.000141    -0.000336     0.000101                 False          False
CUP_WEIGHT_0_13 log_loss_1x2          tie_or_match     4884     1612        -0.000208    -0.000467     0.000131                 False          False
CUP_WEIGHT_0_13 log_loss_1x2           team_season     4884     1413        -0.000208    -0.000413    -0.000008                  True          False
CUP_WEIGHT_0_13 log_loss_1x2        calendar_month     4884       65        -0.000208    -0.000448     0.000015                 False          False
CUP_WEIGHT_0_13 log_loss_1x2 conservative_envelope     4884     <NA>        -0.000208    -0.000467     0.000131                 False          False
CUP_WEIGHT_0_15    brier_1x2          tie_or_match     4884     1612        -0.000146    -0.000368     0.000130                 False          False
CUP_WEIGHT_0_15    brier_1x2           team_season     4884     1413        -0.000146    -0.000309     0.000015                 False          False
CUP_WEIGHT_0_15    brier_1x2        calendar_month     4884       65        -0.000146    -0.000340     0.000029                 False          False
CUP_WEIGHT_0_15    brier_1x2 conservative_envelope     4884     <NA>        -0.000146    -0.000368     0.000130                 False          False
CUP_WEIGHT_0_15 log_loss_1x2          tie_or_match     4884     1612        -0.000219    -0.000517     0.000166                 False          False
CUP_WEIGHT_0_15 log_loss_1x2           team_season     4884     1413        -0.000219    -0.000451     0.000010                 False          False
CUP_WEIGHT_0_15 log_loss_1x2        calendar_month     4884       65        -0.000219    -0.000497     0.000036                 False          False
CUP_WEIGHT_0_15 log_loss_1x2 conservative_envelope     4884     <NA>        -0.000219    -0.000517     0.000166                 False          False
 CUP_WEIGHT_0_2    brier_1x2          tie_or_match     4884     1612        -0.000171    -0.000467     0.000197                 False          False
 CUP_WEIGHT_0_2    brier_1x2           team_season     4884     1413        -0.000171    -0.000387     0.000046                 False          False
 CUP_WEIGHT_0_2    brier_1x2        calendar_month     4884       65        -0.000171    -0.000429     0.000064                 False          False
 CUP_WEIGHT_0_2    brier_1x2 conservative_envelope     4884     <NA>        -0.000171    -0.000467     0.000197                 False          False
 CUP_WEIGHT_0_2 log_loss_1x2          tie_or_match     4884     1612        -0.000258    -0.000654     0.000256                 False          False
 CUP_WEIGHT_0_2 log_loss_1x2           team_season     4884     1413        -0.000258    -0.000566     0.000047                 False          False
 CUP_WEIGHT_0_2 log_loss_1x2        calendar_month     4884       65        -0.000258    -0.000632     0.000081                 False          False
 CUP_WEIGHT_0_2 log_loss_1x2 conservative_envelope     4884     <NA>        -0.000258    -0.000654     0.000256                 False          False
CUP_WEIGHT_0_25    brier_1x2          tie_or_match     4884     1612        -0.000191    -0.000565     0.000275                 False          False
CUP_WEIGHT_0_25    brier_1x2           team_season     4884     1413        -0.000191    -0.000459     0.000079                 False          False
CUP_WEIGHT_0_25    brier_1x2        calendar_month     4884       65        -0.000191    -0.000517     0.000106                 False          False
CUP_WEIGHT_0_25    brier_1x2 conservative_envelope     4884     <NA>        -0.000191    -0.000565     0.000275                 False          False
CUP_WEIGHT_0_25 log_loss_1x2          tie_or_match     4884     1612        -0.000289    -0.000786     0.000359                 False          False
CUP_WEIGHT_0_25 log_loss_1x2           team_season     4884     1413        -0.000289    -0.000673     0.000089                 False          False
CUP_WEIGHT_0_25 log_loss_1x2        calendar_month     4884       65        -0.000289    -0.000760     0.000141                 False          False
CUP_WEIGHT_0_25 log_loss_1x2 conservative_envelope     4884     <NA>        -0.000289    -0.000786     0.000359                 False          False
 CUP_WEIGHT_0_3    brier_1x2          tie_or_match     4884     1612        -0.000200    -0.000653     0.000364                 False          False
 CUP_WEIGHT_0_3    brier_1x2           team_season     4884     1413        -0.000200    -0.000521     0.000123                 False          False
 CUP_WEIGHT_0_3    brier_1x2        calendar_month     4884       65        -0.000200    -0.000592     0.000160                 False          False
 CUP_WEIGHT_0_3    brier_1x2 conservative_envelope     4884     <NA>        -0.000200    -0.000653     0.000364                 False          False
 CUP_WEIGHT_0_3 log_loss_1x2          tie_or_match     4884     1612        -0.000305    -0.000904     0.000475                 False          False
 CUP_WEIGHT_0_3 log_loss_1x2           team_season     4884     1413        -0.000305    -0.000764     0.000146                 False          False
 CUP_WEIGHT_0_3 log_loss_1x2        calendar_month     4884       65        -0.000305    -0.000870     0.000214                 False          False
 CUP_WEIGHT_0_3 log_loss_1x2 conservative_envelope     4884     <NA>        -0.000305    -0.000904     0.000475                 False          False
```

## Nested walk-forward secim

Agirlik yalniz training sezonlarindan secilir; test sezonu gorulmez.

```text
 fold                                           train_seasons test_season  selected_weight  train_brier_1x2  test_matches  test_brier_1x2  baseline_brier_1x2  test_brier_delta  test_log_loss_delta
    1                                         2018/19|2019/20     2020/21             0.00         0.581094           540        0.529469            0.529633         -0.000164            -0.000182
    2                                 2018/19|2019/20|2020/21     2021/22             0.00         0.567128           816        0.576172            0.576271         -0.000099            -0.000156
    3                         2018/19|2019/20|2020/21|2021/22     2022/23             0.00         0.569752           804        0.592285            0.592254          0.000031             0.000022
    4                 2018/19|2019/20|2020/21|2021/22|2022/23     2023/24             0.00         0.574762           806        0.564184            0.564239         -0.000055            -0.000024
    5         2018/19|2019/20|2020/21|2021/22|2022/23|2023/24     2024/25             0.00         0.572834           957        0.569131            0.569223         -0.000092            -0.000125
    6 2018/19|2019/20|2020/21|2021/22|2022/23|2023/24|2024/25     2025/26             0.25         0.572139           961        0.584735            0.584980         -0.000246            -0.000432
```

## Karar girdisi

- Conservative envelope guvenilir iyilesme: `False`.
- Conservative envelope guvenilir zarar: `False`.
- Tek bir kumeleme gorusunde iyilesme: `True` (kapi degildir).
- Tum fold'lari kazanan kollar: `['CUP_WEIGHT_0_05', 'CUP_WEIGHT_0_08']`.
- Guvenilir rating iyilesmesi: `False`.
- Guvenilir rating zarari: `False`.
- En fazla hareket eden kutle orani: `0.1924`.

Karar urun tarafina aittir. Bu belge yalniz kanit uretir.
