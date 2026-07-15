# AO Dynamic Elo Goal-Margin Calibration

Decision: **KEEP_GOAL_MARGIN_AS_CANDIDATE**

## Scope

Seasons: 2018/19 through 2025/26; outer folds: 6.
Each fold uses the Scale/H/K selected only from its earlier seasons.
Competition, stage, progression, caps and season carry remain inactive.

The tested update multiplier is:

```text
G = min(goal_cap, 1 + goal_weight * ln(goal_difference))
G = 1 for draws and one-goal results
```

## Walk-Forward Selections

| Fold | Unseen season | Weight | Cap | Training Brier difference |
| ---: | --- | ---: | ---: | ---: |
| 1 | 2020/21 | 0.5 | 2 | -0.000305 |
| 2 | 2021/22 | 0.5 | 2 | -0.000294 |
| 3 | 2022/23 | 0.5 | 2 | -0.000381 |
| 4 | 2023/24 | 0.5 | 2 | -0.000537 |
| 5 | 2024/25 | 0.5 | 2 | -0.000458 |
| 6 | 2025/26 | 0.5 | 2 | -0.000393 |

Goal margin beat the same fold's core in **5/6** unseen folds.
Overall paired Brier difference: -0.000342 
(95% CI -0.000811 to 0.000141).

## Competition Guardrail

Negative differences favor the goal-margin layer.

| Competition | Matches | Brier difference | Log-loss difference |
| --- | ---: | ---: | ---: |
| UCL | 1384 | -0.000921 | -0.002238 |
| UECL | 2073 | -0.000039 | -0.000111 |
| UEL | 1427 | -0.000221 | -0.000371 |

## Stability

Most common pair: `weight=0.5`, `cap=2` (6/6 folds).

## Full-Data Research Candidate

Core: `Scale=225`, `H=40`, `K=28`.
Goal margin: `weight=0.5`, `cap=2`.
Brier=0.163321; log loss=0.621076.

| Goal difference | Multiplier |
| ---: | ---: |
| 0 | 1.0000 |
| 1 | 1.0000 |
| 2 | 1.3466 |
| 3 | 1.5493 |
| 4 | 1.6931 |
| 5 | 1.8047 |
| 6 | 1.8959 |
| 7 | 1.9730 |
| 8 | 2.0000 |

## Unseen-Season Sensitivity

This diagnostic compares every fixed margin candidate on the outer test seasons.
It is not used to replace the nested fold selections.

| Weight | Cap | Fold wins | Brier difference | Minimum rank correlation |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | 2 | 5/6 | -0.000342 | 0.923 |
| 0.5 | 1.75 | 5/6 | -0.000329 | 0.924 |
| 0.25 | 1.75 | 6/6 | -0.000265 | 0.931 |
| 0.25 | 2 | 6/6 | -0.000265 | 0.931 |
| 0.25 | 1.5 | 6/6 | -0.000265 | 0.931 |

## Goal-Difference Distribution

| Goal difference | Matches | Share |
| ---: | ---: | ---: |
| 0 | 1339 | 21.1% |
| 1 | 2384 | 37.6% |
| 2 | 1348 | 21.3% |
| 3 | 720 | 11.4% |
| 4 | 330 | 5.2% |
| 5 | 153 | 2.4% |
| 6 | 46 | 0.7% |
| 7 | 16 | 0.3% |
| 8+ | 4 | 0.1% |

Displayed scores can include extra time in a small unresolved subset. This run
does not claim a separate 90/120-minute goal-margin policy; that policy must be
tested after exact extra-time enrichment.
