# AO Dynamic Elo Competition-Multiplier Calibration

Decision: **KEEP_COMPETITION_MULTIPLIERS_AS_CANDIDATE**

## Scope

Seasons: 2018/19 through 2025/26; outer folds: 6.
UCL is the fixed 1.0 reference. The enforced hierarchy is UCL >= UEL >= UECL.
These values alter effective K; they are not additions to team strength ratings.
Goal margin, stage, progression, update caps and season carry are inactive.
Each fold inherits the Scale/H/K selected only from its earlier seasons.

## Walk-Forward Selections

| Fold | Unseen season | UEL multiplier | UECL multiplier | Training Brier difference |
| ---: | --- | ---: | ---: | ---: |
| 1 | 2020/21 | 1 | 1 | 0.000000 |
| 2 | 2021/22 | 1 | 1 | 0.000000 |
| 3 | 2022/23 | 1 | 1 | 0.000000 |
| 4 | 2023/24 | 1 | 0.875 | -0.000018 |
| 5 | 2024/25 | 1 | 0.875 | -0.000013 |
| 6 | 2025/26 | 1 | 0.75 | -0.000096 |

Competition multipliers beat the core in **2/6** unseen folds.
Overall paired Brier difference: -0.000133 
(95% CI -0.000260 to -0.000005).

## Competition Guardrail

Negative differences favor the competition layer.

| Competition | Matches | Brier difference | Log-loss difference |
| --- | ---: | ---: | ---: |
| UCL | 1384 | 0.000000 | 0.000000 |
| UECL | 2073 | -0.000314 | -0.000692 |
| UEL | 1427 | 0.000000 | 0.000000 |

## Parameter Stability

| Parameter | Mode | Fold frequency | Range |
| --- | ---: | ---: | ---: |
| selected_uel_multiplier | 1 | 6/6 | 1-1 |
| selected_uecl_multiplier | 1 | 3/6 | 0.75-1 |

## Full-Data Research Candidate

Core: `Scale=225`, `H=40`, `K=28`.
Multipliers: `UCL=1`, `UEL=1`, `UECL=0.625`.
Brier=0.163491; log loss=0.621564.

| Competition | Multiplier | Effective K |
| --- | ---: | ---: |
| UCL | 1 | 28 |
| UEL | 1 | 28 |
| UECL | 0.625 | 17.5 |

## Independent Ablation

`uel_and_below` applies the UEL value to both lower competitions; `uecl_only`
fixes UEL at 1.0. Negative
differences favor the ablated layer against the same core baseline.

| Model | Competition | Matches | Brier difference | Log-loss difference |
| --- | --- | ---: | ---: | ---: |
| combined | UCL | 1384 | 0.000000 | 0.000000 |
| combined | UECL | 2073 | -0.000314 | -0.000692 |
| combined | UEL | 1427 | 0.000000 | 0.000000 |
| uecl_only | UCL | 1384 | 0.000000 | 0.000000 |
| uecl_only | UECL | 2073 | -0.000314 | -0.000692 |
| uecl_only | UEL | 1427 | 0.000000 | 0.000000 |
| uel_and_below | UCL | 1384 | 0.000000 | 0.000000 |
| uel_and_below | UECL | 2073 | 0.000000 | 0.000000 |
| uel_and_below | UEL | 1427 | 0.000000 | 0.000000 |
