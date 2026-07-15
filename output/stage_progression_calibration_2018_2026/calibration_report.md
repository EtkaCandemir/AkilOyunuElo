# AO Dynamic Elo Stage-Progression Calibration

Decision: **KEEP_STAGE_PROGRESSION_AS_CANDIDATE**

## Scope

Seasons: 2018/19 through 2025/26; outer folds: 6.
This run tests whether the previously rejected flat progression bonus becomes
useful when early qualifying is reduced and later knockout rounds receive
monotonically larger multipliers. The base match update remains Scale/H/K only.
Competition prestige is fixed as a domain reference at UCL=1.00, UEL=0.65 and
UECL=0.45; these values are not calibrated or promoted by this run.
Final progression is fixed at zero because a post-final update has no later
same-season prediction and season carry is deliberately inactive in this run.

```text
Delta = K_core * progression_ratio * competition_reference
        * stage_multiplier * (Advanced - ExpectedToAdvance)
```

## Tie Coverage

| Competition | Normalized stage | Ties |
| --- | --- | ---: |
| UCL | FINAL | 8 |
| UCL | KNOCKOUT_PLAYOFF | 16 |
| UCL | QUALIFYING | 372 |
| UCL | QUARTERFINAL | 32 |
| UCL | ROUND_OF_16 | 64 |
| UCL | SEMIFINAL | 16 |
| UECL | FINAL | 5 |
| UECL | KNOCKOUT_PLAYOFF | 40 |
| UECL | QUALIFYING | 672 |
| UECL | QUARTERFINAL | 20 |
| UECL | ROUND_OF_16 | 40 |
| UECL | SEMIFINAL | 10 |
| UEL | FINAL | 8 |
| UEL | KNOCKOUT_PLAYOFF | 88 |
| UEL | QUALIFYING | 604 |
| UEL | QUARTERFINAL | 32 |
| UEL | ROUND_OF_16 | 63 |
| UEL | SEMIFINAL | 16 |

## Nested Selections

| Fold | Unseen season | Selected ratio | Selected profile | Positive ratio | Positive profile | Positive train diff |
| ---: | --- | ---: | --- | ---: | --- | ---: |
| 1 | 2020/21 | 0 | FLAT | 0.125 | LATE_STRICT | 0.000001 |
| 2 | 2021/22 | 0 | FLAT | 0.125 | LATE_STRICT | 0.000003 |
| 3 | 2022/23 | 0 | FLAT | 0.125 | LATE_STRICT | 0.000003 |
| 4 | 2023/24 | 0 | FLAT | 0.125 | SEMIFINAL_HEAVY | 0.000001 |
| 5 | 2024/25 | 0 | FLAT | 0.125 | SEMIFINAL_HEAVY | 0.000001 |
| 6 | 2025/26 | 0 | FLAT | 0.125 | LATE_STRICT | 0.000001 |

Nested stage progression beat core in **0/6** folds.
The forced positive challenger beat core in **3/6** folds.
Positive challenger Brier difference: 0.000001 
(95% CI -0.000004 to 0.000005).

## Positive Challenger by Competition

| Competition | Matches | Brier difference | Log-loss difference |
| --- | ---: | ---: | ---: |
| UCL | 1384 | 0.000002 | 0.000005 |
| UECL | 2073 | 0.000000 | -0.000000 |
| UEL | 1427 | 0.000001 | 0.000002 |

## Positive Challenger by Prediction Stage

These rows show where predictions changed because of earlier updates; the FINAL
row does not estimate a post-final bonus effect.

| Stage | Matches | Brier difference | Log-loss difference |
| --- | ---: | ---: | ---: |
| FINAL | 17 | -0.000477 | -0.000978 |
| KNOCKOUT_PLAYOFF | 224 | 0.000000 | 0.000000 |
| LEAGUE | 1896 | 0.000000 | 0.000000 |
| QUALIFYING | 2273 | 0.000000 | 0.000000 |
| QUARTERFINAL | 136 | 0.000006 | -0.000006 |
| ROUND_OF_16 | 270 | 0.000020 | 0.000044 |
| SEMIFINAL | 68 | 0.000089 | 0.000223 |

## Parameter Stability

| Parameter | Mode | Fold frequency | Unique values |
| --- | --- | ---: | ---: |
| selected_progression_ratio | 0.0 | 6/6 | 1 |
| selected_stage_profile | FLAT | 6/6 | 1 |

## Full-Data Candidate

`ratio=0.125`, 
`profile=LATE_STRICT`; 
Brier=0.163653; 
log loss=0.621837.

| Stage | Multiplier | UCL K_progression | UEL K_progression | UECL K_progression |
| --- | ---: | ---: | ---: | ---: |
| QUALIFYING | 0 | 0 | 0 | 0 |
| KNOCKOUT_PLAYOFF | 0 | 0 | 0 | 0 |
| ROUND_OF_16 | 0.5 | 1.75 | 1.1375 | 0.7875 |
| QUARTERFINAL | 0.9 | 3.15 | 2.0475 | 1.4175 |
| SEMIFINAL | 1.3 | 4.55 | 2.9575 | 2.0475 |
| FINAL | 0 | 0 | 0 | 0 |

Because stage profiles were designed after prior 2018-2026 diagnostics, any
positive result remains a research candidate until tested on a future untouched
season. A zero ratio means the competition references are inactive, not that
UCL, UEL or UECL has zero sporting value.
