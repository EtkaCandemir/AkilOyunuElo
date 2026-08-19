# xG Bilgili Gol Beklentisi Backtest

Aktif skor katmani her iki orani yalniz Elo farkindan turetir ve
`Diagnostic` statusundedir; climatology'yi ust 2.5'te `0.0018` Brier
gecer. Bu kosu her iki tarafa birer form terimi ekler ve ayni yapiyi iki
farkli kaynakla besler.

`GOALS` kolu kontroldur: onsuz `XG` kolundaki bir kazanc, xG'nin mi yoksa
form teriminin mi katkisi oldugu ayirt edilemezdi.

## Dogrulama kapilari

```text
                         gate  passed    observed                                           requirement
            three_arms_scored    True    3.000000     Elo-only baseline, goals control and xG candidate
        arms_share_one_sample    True 3528.000000       Every arm scores the identical held-out matches
     elo_only_carries_no_form    True    0.000000          The baseline must stay a two-parameter model
          rates_within_bounds    True    4.500000        Rates stay inside the production lambda window
xg_form_is_sparser_than_goals    True    0.587868 Documented coverage gap must be visible in the sample
```

## Kol ozeti

```text
     arm  matches  exact_score_nll  total_goals_error  over_2_5_brier  btts_brier  exact_score_nll_vs_elo  total_goals_error_vs_elo  over_2_5_brier_vs_elo  btts_brier_vs_elo
      XG     3528         3.027008           1.384003        0.246658    0.249700               -0.002834                 -0.004258              -0.000499          -0.000641
   GOALS     3528         3.029642           1.387948        0.247579    0.249816               -0.000201                 -0.000313               0.000423          -0.000525
ELO_ONLY     3528         3.029843           1.388261        0.247156    0.250341                0.000000                  0.000000               0.000000           0.000000
```

## Segment kirilimi (Elo-only'ye karsi)

```text
  arm          segment  matches  exact_score_nll_vs_elo  total_goals_error_vs_elo  over_2_5_brier_vs_elo  btts_brier_vs_elo
GOALS              ALL     3528               -0.000201                 -0.000313               0.000423          -0.000525
GOALS       PHASE:MAIN     1876               -0.001482                 -0.002138               0.000330          -0.000236
GOALS PHASE:QUALIFYING     1652                0.001255                  0.001759               0.000528          -0.000853
GOALS       XG_PRESENT     2074               -0.001142                 -0.002003               0.000251          -0.000264
GOALS        XG_ABSENT     1454                0.001143                  0.002097               0.000668          -0.000897
   XG              ALL     3528               -0.002834                 -0.004258              -0.000499          -0.000641
   XG       PHASE:MAIN     1876               -0.008186                 -0.008927              -0.001304          -0.000692
   XG PHASE:QUALIFYING     1652                0.003243                  0.001046               0.000415          -0.000583
   XG       XG_PRESENT     2074               -0.006970                 -0.007912              -0.001123          -0.000720
   XG        XG_ABSENT     1454                0.003065                  0.000955               0.000391          -0.000528
```

## Fold bazinda

```text
     arm test_season  matches  exact_score_nll  total_goals_error  over_2_5_brier  btts_brier
ELO_ONLY     2022/23      804         2.960489           1.350915        0.248766    0.254565
   GOALS     2022/23      804         2.956156           1.350275        0.248626    0.252755
      XG     2022/23      804         2.953614           1.347826        0.248112    0.253860
ELO_ONLY     2023/24      806         3.012724           1.379887        0.244775    0.246555
   GOALS     2023/24      806         3.020274           1.381358        0.246403    0.247934
      XG     2023/24      806         3.017213           1.380382        0.245943    0.246738
ELO_ONLY     2024/25      957         3.076728           1.418794        0.248069    0.249344
   GOALS     2024/25      957         3.074255           1.416523        0.248097    0.248590
      XG     2024/25      957         3.071625           1.408952        0.246159    0.247892
ELO_ONLY     2025/26      961         3.055533           1.396122        0.246899    0.250975
   GOALS     2025/26      961         3.054550           1.396536        0.247176    0.250157
      XG     2025/26      961         3.052196           1.392463        0.246537    0.250505
```

## Dependency-robust belirsizlik (conservative envelope)

```text
  arm    segment          metric  mean_difference  ci_95_lower  ci_95_upper  reliable_improvement  reliable_harm
GOALS        ALL exact_score_nll        -0.000201    -0.004052     0.003589                 False          False
GOALS        ALL  over_2_5_brier         0.000423    -0.000875     0.001656                 False          False
GOALS XG_PRESENT exact_score_nll        -0.001142    -0.005266     0.002893                 False          False
GOALS XG_PRESENT  over_2_5_brier         0.000251    -0.001250     0.001795                 False          False
   XG        ALL exact_score_nll        -0.002834    -0.007159     0.001017                 False          False
   XG        ALL  over_2_5_brier        -0.000499    -0.001670     0.000566                 False          False
   XG XG_PRESENT exact_score_nll        -0.006970    -0.011187    -0.002668                  True          False
   XG XG_PRESENT  over_2_5_brier        -0.001123    -0.002555     0.000275                 False          False
```

## Karar girdisi

- xG exact-score'da gol formunu geciyor mu: `True`.
- Guvenilir iyilesme sayisi: `1`.

Karar urun tarafina aittir; bu belge yalniz kanit uretir.
