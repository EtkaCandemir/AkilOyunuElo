# AO vs Historical External ClubElo Benchmark

## Contract

- Exact UTC match dates: official UEFA match service.
- External ratings: latest ClubElo-derived snapshot strictly before the match.
- Snapshot age limit: 31 days.
- ClubElo probability scale: 400, matching ClubElo's published Elo equation.
- ClubElo home advantage: selected only on earlier seasons in each fold.
- AO candidate: scale=225, H=40, K=28, season power carry=0.85.

## Result

Full source matches: 6340
Eligible paired matches: 492
Walk-forward unseen test matches: 363
Direction: ClubElo daha iyi
95% paired bootstrap conclusive: False

```text
           model  matches    brier  log_loss
      AO_current    363.0 0.163249  0.625267
ClubElo_external    363.0 0.158745  0.612074
```

## Competition detail

```text
competition  matches  ao_brier  clubelo_brier  ao_minus_clubelo_brier  ao_log_loss  clubelo_log_loss
        UCL      171  0.171128       0.156801                0.014327     0.641829          0.610759
       UECL       90  0.150355       0.159926               -0.009572     0.592504          0.604728
        UEL      102  0.161416       0.160963                0.000454     0.626411          0.620761
```

## Uncertainty

```text
competition  matches  mean_ao_minus_clubelo_brier  ci_95_lower  ci_95_upper  ao_reliably_better  clubelo_reliably_better
        ALL      363                     0.004504    -0.005398     0.014634               False                    False
        UCL      171                     0.014327     0.001144     0.027631               False                     True
       UECL       90                    -0.009572    -0.034141     0.013603               False                    False
        UEL      102                     0.000454    -0.016974     0.018122               False                    False
```

## Selected external H

```text
 fold                                   train_seasons test_season  train_matches  test_matches  selected_clubelo_home_advantage  train_clubelo_brier  train_clubelo_log_loss
    1                                 2018/19,2019/20     2020/21            129            53                             50.0             0.151537                0.621651
    2                         2018/19,2019/20,2020/21     2021/22            182            71                             30.0             0.143339                0.601565
    3                 2018/19,2019/20,2020/21,2021/22     2022/23            253            80                             30.0             0.145819                0.606151
    4         2018/19,2019/20,2020/21,2021/22,2022/23     2023/24            333            74                             60.0             0.151982                0.607731
    5 2018/19,2019/20,2020/21,2021/22,2022/23,2023/24     2024/25            407            85                             50.0             0.149702                0.603469
```

## Coverage

```text
 season competition  matches  exact_dates  eligible_external_pairs  external_pair_coverage
2018/19         UCL      216          216                       24                0.111111
2018/19         UEL      519          519                       47                0.090559
2019/20         UCL      210          210                       28                0.133333
2019/20         UEL      511          511                       30                0.058708
2020/21         UCL      178          178                       32                0.179775
2020/21         UEL      362          362                       21                0.058011
2021/22         UCL      218          218                       35                0.160550
2021/22        UECL      423          423                       22                0.052009
2021/22         UEL      175          175                       14                0.080000
2022/23         UCL      214          214                       28                0.130841
2022/23        UECL      415          415                       20                0.048193
2022/23         UEL      175          175                       32                0.182857
2023/24         UCL      214          214                       36                0.168224
2023/24        UECL      417          417                       24                0.057554
2023/24         UEL      175          175                       14                0.080000
2024/25         UCL      279          279                       40                0.143369
2024/25        UECL      409          409                       24                0.058680
2024/25         UEL      269          269                       21                0.078067
2025/26         UCL      281          281                        0                0.000000
2025/26        UECL      409          409                        0                0.000000
2025/26         UEL      271          271                        0                0.000000
```

## Interpretation limit

The ClubElo snapshot archive covers mostly stronger, established clubs. This paired
sample is therefore not representative of all qualifying-round teams. It is a useful
external diagnostic, not a claim that either model is universally superior.
The AO parameters were previously calibrated on overlapping seasons, so this run is
not a pristine final holdout. A later untouched-season test remains necessary.
