# AO Dynamic Elo Competition Prestige Calibration

Decision: **REJECT_WIN_PRESTIGE_BONUS_KEEP_CORE**

## Scope

Seasons: 2018/19 through 2025/26; outer folds: 6.
The base zero-sum Elo update uses the same fold-selected Scale/H/K in every
competition. A separate zero-sum prestige delta is added only on decisive results.
Draws receive no prestige bonus. Goal margin, stage, progression, caps and carry
remain inactive.

```text
Base delta = K * (S - E)
Prestige delta = B_win * C_competition * (S - E)  # decisive only
Total delta = Base delta + Prestige delta
```

The domain constraint is strict: `UCL=1.00 > UEL > UECL`, with at least
0.10 between UEL and UECL.

## Walk-Forward Selections

| Fold | Unseen season | B_win | UEL | UECL | Training Brier difference |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 2020/21 | 0 | 0.65 | 0.45 | 0.000000 |
| 2 | 2021/22 | 0 | 0.65 | 0.45 | 0.000000 |
| 3 | 2022/23 | 0 | 0.65 | 0.45 | 0.000000 |
| 4 | 2023/24 | 0 | 0.65 | 0.45 | 0.000000 |
| 5 | 2024/25 | 0 | 0.65 | 0.45 | 0.000000 |
| 6 | 2025/26 | 0 | 0.65 | 0.45 | 0.000000 |

Prestige bonus beat the core in **0/6** unseen folds.
Overall paired Brier difference: 0.000000 
(95% CI 0.000000 to 0.000000).

## Competition Guardrail

Negative differences favor the prestige layer.

| Competition | Matches | Brier difference | Log-loss difference |
| --- | ---: | ---: | ---: |
| UCL | 1384 | 0.000000 | 0.000000 |
| UECL | 2073 | 0.000000 | 0.000000 |
| UEL | 1427 | 0.000000 | 0.000000 |

## Parameter Stability

| Parameter | Mode | Fold frequency | Range |
| --- | ---: | ---: | ---: |
| selected_win_bonus_base | 0 | 6/6 | 0-0 |
| selected_uel_prestige | 0.65 | 6/6 | 0.65-0.65 |
| selected_uecl_prestige | 0.45 | 6/6 | 0.45-0.45 |

## Full-Data Research Candidate

Core: `Scale=225`, `H=40`, `K=28`.
Prestige: `B_win=0`, `UCL=1`, `UEL=0.65`, `UECL=0.45`.
Brier=0.163653; log loss=0.621837.

| Competition | Prestige | Maximum extra K | Decisive-result total K |
| --- | ---: | ---: | ---: |
| UCL | 1 | 0 | 28 |
| UEL | 0.65 | 0 | 28 |
| UECL | 0.45 | 0 | 28 |
