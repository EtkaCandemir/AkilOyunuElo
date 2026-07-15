# AO European Elo Rating Spread Calibration

This analysis calibrates only the conversion from pre-season rating differences
to expected match score. It does not change team order or apply match-by-match Elo updates.

Baseline probability conversion: `elo_scale=400`, `home_advantage=70`.
A lower selected scale means the existing numerical rating gaps should have a
stronger match-probability effect; it does not by itself require rewriting ratings.

## Rating distribution

- Team-seasons: 1887
- Minimum / maximum: 540.795 / 902.867
- Range: 362.072
- Standard deviation: 90.480
- Interquartile range: 120.280

## Nested walk-forward selections

| fold | train_seasons | test_season | selected_elo_scale | selected_home_advantage | equivalent_spread_multiplier | spread_only_elo_scale | spread_only_multiplier | baseline | spread_only | spread_only_delta | joint | joint_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2018/19\|2019/20 | 2020/21 | 300.000000 | 50.000000 | 1.333333 | 350.000000 | 1.142857 | 0.623267 | 0.621581 | -0.001687 | 0.612870 | -0.010397 |
| 2 | 2018/19\|2019/20\|2020/21 | 2021/22 | 300.000000 | 50.000000 | 1.333333 | 350.000000 | 1.142857 | 0.629390 | 0.627394 | -0.001996 | 0.624875 | -0.004515 |
| 3 | 2018/19\|2019/20\|2020/21\|2021/22 | 2022/23 | 300.000000 | 50.000000 | 1.333333 | 350.000000 | 1.142857 | 0.645415 | 0.645693 | 0.000278 | 0.645761 | 0.000346 |
| 4 | 2018/19\|2019/20\|2020/21\|2021/22\|2022/23 | 2023/24 | 300.000000 | 50.000000 | 1.333333 | 350.000000 | 1.142857 | 0.623160 | 0.620566 | -0.002594 | 0.617596 | -0.005564 |
| 5 | 2018/19\|2019/20\|2020/21\|2021/22\|2022/23\|2023/24 | 2024/25 | 300.000000 | 50.000000 | 1.333333 | 350.000000 | 1.142857 | 0.633915 | 0.632145 | -0.001770 | 0.630821 | -0.003094 |
| 6 | 2018/19\|2019/20\|2020/21\|2021/22\|2022/23\|2023/24\|2024/25 | 2025/26 | 300.000000 | 50.000000 | 1.333333 | 350.000000 | 1.142857 | 0.647315 | 0.647434 | 0.000119 | 0.647023 | -0.000292 |

## Competition results

| competition | model | matches | log_loss | brier | log_loss_delta |
| --- | --- | --- | --- | --- | --- |
| ALL | baseline | 4884 | 0.634737 | 0.170999 | 0.000000 |
| ALL | joint | 4884 | 0.631308 | 0.169457 | -0.003429 |
| ALL | spread_only | 4884 | 0.633511 | 0.170560 | -0.001226 |
| UCL | baseline | 1384 | 0.623718 | 0.168459 | 0.000000 |
| UCL | joint | 1384 | 0.616026 | 0.165104 | -0.007692 |
| UCL | spread_only | 1384 | 0.621101 | 0.167476 | -0.002618 |
| UECL | baseline | 2073 | 0.638992 | 0.170760 | 0.000000 |
| UECL | joint | 2073 | 0.635858 | 0.169313 | -0.003135 |
| UECL | spread_only | 2073 | 0.637696 | 0.170301 | -0.001296 |
| UEL | baseline | 1427 | 0.639241 | 0.173809 | 0.000000 |
| UEL | joint | 1427 | 0.639520 | 0.173887 | 0.000279 |
| UEL | spread_only | 1427 | 0.639467 | 0.173929 | 0.000226 |

## Paired match-level uncertainty

| competition | comparison | matches | mean_log_loss_delta | ci_95_lower | ci_95_upper | directionally_reliable_improvement | directionally_reliable_harm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ALL | joint_vs_baseline | 4884 | -0.003429 | -0.005202 | -0.001665 | True | False |
| ALL | spread_only_vs_baseline | 4884 | -0.001226 | -0.002138 | -0.000289 | True | False |
| UCL | joint_vs_baseline | 1384 | -0.007692 | -0.010916 | -0.004397 | True | False |
| UCL | spread_only_vs_baseline | 1384 | -0.002618 | -0.004297 | -0.000902 | True | False |
| UEL | joint_vs_baseline | 1427 | 0.000279 | -0.003333 | 0.003980 | False | False |
| UEL | spread_only_vs_baseline | 1427 | 0.000226 | -0.001615 | 0.002192 | False | False |
| UECL | joint_vs_baseline | 2073 | -0.003135 | -0.005659 | -0.000605 | True | False |
| UECL | spread_only_vs_baseline | 2073 | -0.001296 | -0.002638 | 0.000052 | False | False |

## Observed score by raw rating difference

| difference_band | matches | mean_absolute_rating_difference | higher_rated_team_score | draw_rate |
| --- | --- | --- | --- | --- |
| 0-25 | 1650 | 11.642417 | 0.551818 | 0.223636 |
| 25-50 | 1255 | 36.859644 | 0.609562 | 0.221514 |
| 50-75 | 974 | 62.213692 | 0.632444 | 0.221766 |
| 75-100 | 750 | 87.268451 | 0.654000 | 0.204000 |
| 100-150 | 1099 | 122.387117 | 0.707006 | 0.205641 |
| 150-200 | 471 | 169.738110 | 0.778132 | 0.159236 |
| 200-300 | 141 | 224.362053 | 0.765957 | 0.156028 |

## Decision

- Decision: `KEEP_CURRENT_RATING_SPREAD`
- Unseen folds improved: `4/6`
- Competitions not worse: `2/3`
- Selected scale range: `0`
- Selected home-advantage range: `0`
- Parameter stability gate: `True`

No visual preference for wider ratings can override failed unseen-season or
competition stability gates.
Match-level bootstrap intervals are diagnostic and do not model correlation
between two-legged ties or repeated appearances by the same club.
