# AO Dynamic Elo Core Calibration

Decision: **PROVISIONAL_ACCEPT_CORE**

## Scope

Seasons: 2018/19 through 2025/26; outer folds: 6.
Only Elo scale, home advantage and base K are active. Competition, stage,
goal-margin, progression and season-carry multipliers are fixed at neutral values.
Every season starts from the frozen AO First Elo v1.1 rating.

Brier score is the primary selection metric. The static comparator selects its
own Scale/H on the same training seasons with K=0, so the comparison isolates
the value of chronological match updates.

## Walk-Forward Selections

| Fold | Unseen season | Scale | H | K |
| ---: | --- | ---: | ---: | ---: |
| 1 | 2020/21 | 275 | 50 | 28 |
| 2 | 2021/22 | 250 | 40 | 28 |
| 3 | 2022/23 | 250 | 40 | 28 |
| 4 | 2023/24 | 250 | 40 | 28 |
| 5 | 2024/25 | 225 | 40 | 28 |
| 6 | 2025/26 | 225 | 40 | 28 |

Dynamic beat the tuned static comparator in **6/6** unseen folds.
Overall paired Brier difference: -0.003955 
(95% CI -0.005520 to -0.002333).

## Competition Guardrail

Negative differences favor the dynamic model.

| Competition | Matches | Brier difference | Log-loss difference |
| --- | ---: | ---: | ---: |
| UCL | 1384 | -0.004564 | -0.010607 |
| UECL | 2073 | -0.002624 | -0.006267 |
| UEL | 1427 | -0.005297 | -0.010617 |

## Parameter Stability

| Parameter | Mode | Fold frequency | Range |
| --- | ---: | ---: | ---: |
| selected_scale | 250 | 3/6 | 225-275 |
| selected_home_advantage | 40 | 5/6 | 40-50 |
| selected_k | 28 | 6/6 | 28-28 |

## Full-Data Research Candidate

`Scale=225`, `H=40`, `K=28`; Brier=0.163653; 
log loss=0.621837.

This is a research candidate, not a frozen production parameter set. The next
tests must add one layer at a time: goal margin, competition, stage, progression,
caps and season carry. Exact extra-time policy also remains outside this run.
