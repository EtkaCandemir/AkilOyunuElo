# Bounded xG Katmani: Alti Sezonluk Yeniden Dogrulama

Aktif contract `xg_performance` katmanini tek sezonluk kanitla acti:
`961` mac, `606` uygun, `manual_product_decision`. FotMob xG 2020/21'e kadar uzandigi icin uygun
orneklem `606` -> `2827` seviyesine cikti.

Bu kosu production parametresi degistirmez.

## Dogrulama kapilari

```text
                               gate  passed    observed                                                 requirement
            wide_superset_of_legacy    True 2827.000000    Six-season map must contain every 2025/26 eligible match
            shared_values_identical    True  606.000000         Re-fetched 2025/26 xG must match the frozen dataset
               eligible_sample_grew    True 2827.000000 Evidence base must be materially wider than the shipped one
coverage_concentrated_in_main_stage    True    0.911567       Documented coverage shape must hold in the loaded map
```

## Kol ozeti

```text
            arm  matches  brier_1x2  log_loss_1x2  accuracy_1x2
  XG_SIX_SEASON     6340   0.573732      0.966444      0.546372
XG_2025_26_ONLY     6340   0.574160      0.967043      0.546530
          NO_XG     6340   0.574346      0.967315      0.545268
```

## Segment kirilimi (NO_XG'ye karsi Brier farki)

Negatif deger xG katmaninin faydali oldugunu gosterir. Kapsam ana asamada
yogunlastigi icin `PHASE:MAIN` ve `XG_PRESENT` satirlari karar icin
havuzlanmis satirdan daha bilgilendiricidir.

```text
            arm          segment  matches  brier_delta_vs_no_xg
  XG_SIX_SEASON              ALL     6340             -0.000614
  XG_SIX_SEASON       PHASE:MAIN     3257             -0.001169
  XG_SIX_SEASON PHASE:QUALIFYING     3083             -0.000028
  XG_SIX_SEASON       XG_PRESENT     2827             -0.001385
  XG_SIX_SEASON        XG_ABSENT     3513              0.000006
XG_2025_26_ONLY              ALL     6340             -0.000186
XG_2025_26_ONLY       PHASE:MAIN     3257             -0.000360
XG_2025_26_ONLY PHASE:QUALIFYING     3083             -0.000003
XG_2025_26_ONLY       XG_PRESENT     2827             -0.000433
XG_2025_26_ONLY        XG_ABSENT     3513              0.000013
```

## Sezon bazinda

```text
 season  matches  brier_delta  log_loss_delta
2018/19      735     0.000000        0.000000
2019/20      721     0.000000        0.000000
2020/21      540    -0.001446       -0.001855
2021/22      816    -0.000465       -0.000807
2022/23      804    -0.002362       -0.003399
2023/24      806     0.000255        0.000334
2024/25      957     0.000147        0.000345
2025/26      961    -0.001228       -0.001795
```

## Dependency-robust belirsizlik (conservative envelope)

```text
            arm    segment       metric  mean_difference  ci_95_lower  ci_95_upper  reliable_improvement  reliable_harm
  XG_SIX_SEASON        ALL    brier_1x2        -0.000614    -0.001149    -0.000108                  True          False
  XG_SIX_SEASON        ALL log_loss_1x2        -0.000870    -0.001638    -0.000135                  True          False
  XG_SIX_SEASON PHASE:MAIN    brier_1x2        -0.001169    -0.002103    -0.000210                  True          False
  XG_SIX_SEASON PHASE:MAIN log_loss_1x2        -0.001659    -0.003007    -0.000253                  True          False
  XG_SIX_SEASON XG_PRESENT    brier_1x2        -0.001385    -0.002465    -0.000278                  True          False
  XG_SIX_SEASON XG_PRESENT log_loss_1x2        -0.001949    -0.003493    -0.000334                  True          False
XG_2025_26_ONLY        ALL    brier_1x2        -0.000186    -0.000618     0.000111                 False          False
XG_2025_26_ONLY        ALL log_loss_1x2        -0.000272    -0.000893     0.000155                 False          False
XG_2025_26_ONLY PHASE:MAIN    brier_1x2        -0.000360    -0.001087     0.000245                 False          False
XG_2025_26_ONLY PHASE:MAIN log_loss_1x2        -0.000539    -0.001586     0.000331                 False          False
XG_2025_26_ONLY XG_PRESENT    brier_1x2        -0.000433    -0.001264     0.000275                 False          False
XG_2025_26_ONLY XG_PRESENT log_loss_1x2        -0.000640    -0.001828     0.000372                 False          False
```

## Karar girdisi

- Guvenilir iyilesme: `True`.
- Guvenilir zarar: `False`.

Karar urun tarafina aittir; bu belge yalniz kanit uretir.
