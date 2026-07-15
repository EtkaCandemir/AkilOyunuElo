# UCL Forecast Probability Scale Calibration

This layer changes only the mapping from pre-match AO rating difference to
forecast probability. Elo updates, K, rating values and team ranking remain frozen.

Eligible UCL matches: 223
Decision: `KEEP_FORECAST_SCALE_225_NO_RELIABLE_UCL_GAIN`

## Nested selections

```text
 fold                                   train_seasons test_season  train_matches  test_matches  selected_scale  train_brier  train_log_loss
    1                                 2018/19,2019/20     2020/21             52            32           400.0     0.160635        0.676072
    2                         2018/19,2019/20,2020/21     2021/22             84            35           375.0     0.149711        0.643649
    3                 2018/19,2019/20,2020/21,2021/22     2022/23            119            28           400.0     0.155818        0.648667
    4         2018/19,2019/20,2020/21,2021/22,2022/23     2023/24            147            36           325.0     0.156054        0.629120
    5 2018/19,2019/20,2020/21,2021/22,2022/23,2023/24     2024/25            183            40           300.0     0.153910        0.621172
```

## Unseen aggregate

```text
             model  matches    brier  log_loss
baseline_scale_225    171.0 0.171128  0.641829
  clubelo_external    171.0 0.156801  0.610759
selected_ucl_scale    171.0 0.169211  0.636877
```

## Paired uncertainty

```text
          comparison  matches  mean_brier_difference  ci_95_lower  ci_95_upper  folds_won  folds  reliable_improvement  reliable_harm
     selected_vs_225      171              -0.001917    -0.009518     0.005268          2      5                 False          False
 selected_vs_clubelo      171               0.012410     0.000107     0.024757          0      5                 False           True
scale_225_vs_clubelo      171               0.014327     0.001017     0.027696          1      5                 False           True
```

## Full-data sensitivity

```text
 forecast_scale  matches    brier  log_loss  distance_from_225
          375.0      223 0.165555  0.641953              150.0
          400.0      223 0.165601  0.642314              175.0
          350.0      223 0.165677  0.641946              125.0
          325.0      223 0.166027  0.642456              100.0
          300.0      223 0.166692  0.643726               75.0
          275.0      223 0.167785  0.646118               50.0
          250.0      223 0.169466  0.650185               25.0
          225.0      223 0.171953  0.656799                0.0
          200.0      223 0.175546  0.667373               25.0
```

## Calibration bands

```text
             model probability_band  matches  mean_prediction  mean_actual_score  calibration_gap
selected_ucl_scale       (0.2, 0.4]       28         0.319206           0.321429        -0.002223
selected_ucl_scale       (0.4, 0.6]       74         0.517144           0.486486         0.030658
selected_ucl_scale       (0.6, 0.8]       56         0.682725           0.616071         0.066654
selected_ucl_scale       (0.8, 1.0]       13         0.828717           0.846154        -0.017437
baseline_scale_225    (-0.001, 0.2]       10         0.158130           0.300000        -0.141870
baseline_scale_225       (0.2, 0.4]       29         0.317310           0.310345         0.006965
baseline_scale_225       (0.4, 0.6]       43         0.519812           0.488372         0.031439
baseline_scale_225       (0.6, 0.8]       55         0.685136           0.545455         0.139681
baseline_scale_225       (0.8, 1.0]       34         0.871450           0.808824         0.062626
  clubelo_external    (-0.001, 0.2]        6         0.157080           0.083333         0.073746
  clubelo_external       (0.2, 0.4]       41         0.309272           0.365854        -0.056582
  clubelo_external       (0.4, 0.6]       48         0.515522           0.364583         0.150939
  clubelo_external       (0.6, 0.8]       53         0.698851           0.735849        -0.036998
  clubelo_external       (0.8, 1.0]       23         0.853357           0.804348         0.049009
```

## Guardrail

This diagnostic never changes AO Elo points or rank order. Acceptance requires
at least four fold wins, a paired 95% improvement and stable scale selection.
