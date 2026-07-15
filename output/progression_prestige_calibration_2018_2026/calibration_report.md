# AO Dynamic Elo Knockout Progression Prestige Calibration

Decision: **REJECT_PROGRESSION_PRESTIGE_KEEP_CORE**

## Scope

Seasons: 2018/19 through 2025/26; outer folds: 6.
The frozen dynamic core is updated after every match. A separate zero-sum
progression update is applied once, after a knockout tie is decided. The tie
expectation is frozen before its first leg with home advantage set to zero.
Single-match ties, two-legged ties, penalty decisions and finals are included.
Stage multipliers remain fixed at 1.0.

```text
K_progression = K_core * progression_ratio
Delta = K_progression * competition_prestige * (Advanced - ExpectedToAdvance)
```

Competition hierarchy is a hard domain constraint: `UCL=1.00 > UEL > UECL`,
with at least 0.10 between UEL and UECL. It is not inferred from unconstrained K.

## Walk-Forward Selections

| Fold | Unseen season | Ratio | UEL | UECL | Training Brier difference |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 2020/21 | 0 | 0.65 | 0.45 | 0.000000 |
| 2 | 2021/22 | 0 | 0.65 | 0.45 | 0.000000 |
| 3 | 2022/23 | 0 | 0.65 | 0.45 | 0.000000 |
| 4 | 2023/24 | 0 | 0.65 | 0.45 | 0.000000 |
| 5 | 2024/25 | 0 | 0.65 | 0.45 | 0.000000 |
| 6 | 2025/26 | 0 | 0.65 | 0.45 | 0.000000 |

Progression beat the core in **0/6** unseen folds.
Overall paired Brier difference: 0.000000 
(95% CI 0.000000 to 0.000000).

## Forced Positive Challenger

To avoid a zero-selected model hiding positive-bonus behavior, the best
strictly positive candidate on each training window is also tested unseen.

| Fold | Unseen season | Ratio | UEL | UECL | Training Brier difference |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 2020/21 | 0.25 | 0.55 | 0.45 | 0.000248 |
| 2 | 2021/22 | 0.25 | 0.55 | 0.45 | 0.000266 |
| 3 | 2022/23 | 0.25 | 0.55 | 0.35 | 0.000261 |
| 4 | 2023/24 | 0.25 | 0.55 | 0.35 | 0.000255 |
| 5 | 2024/25 | 0.25 | 0.55 | 0.35 | 0.000221 |
| 6 | 2025/26 | 0.25 | 0.55 | 0.35 | 0.000257 |

The forced positive challenger beat core in **0/6** folds.
Its overall paired Brier difference: 0.000268 
(95% CI 0.000157 to 0.000379).

| Competition | Matches | Challenger Brier difference | Log-loss difference |
| --- | ---: | ---: | ---: |
| UCL | 1384 | 0.000564 | 0.001308 |
| UECL | 2073 | 0.000079 | 0.000165 |
| UEL | 1427 | 0.000257 | 0.000533 |

## Competition Guardrail

Negative differences favor the progression layer.

| Competition | Matches | Brier difference | Log-loss difference |
| --- | ---: | ---: | ---: |
| UCL | 1384 | 0.000000 | 0.000000 |
| UECL | 2073 | 0.000000 | 0.000000 |
| UEL | 1427 | 0.000000 | 0.000000 |

## Parameter Stability

| Parameter | Mode | Fold frequency | Range |
| --- | ---: | ---: | ---: |
| selected_progression_ratio | 0 | 6/6 | 0-0 |
| selected_uel_prestige | 0.65 | 6/6 | 0.65-0.65 |
| selected_uecl_prestige | 0.45 | 6/6 | 0.45-0.45 |

## Full-Data Research Candidate

`ratio=0`, `UCL=1`, 
`UEL=0.65`, `UECL=0.45`; 
Brier=0.163653; 
log loss=0.621837; 
progression events=2106.

| Competition | Prestige | Effective progression K |
| --- | ---: | ---: |
| UCL | 1 | 0 |
| UEL | 0.65 | 0 |
| UECL | 0.45 | 0 |

## Decision Rule

Promotion requires at least 5/6 unseen-fold wins, a paired overall 95% CI below
zero, no reliably harmed competition, stable ratio selection and ranking
guardrails. If ratio zero wins, prestige coefficients are unidentified and no
competition multiplier is promoted.
