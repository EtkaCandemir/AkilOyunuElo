# AO Strict Competition Match Multiplier Calibration

Decision: **KEEP_STRICT_COMPETITION_MATCH_MULTIPLIERS_AS_CANDIDATE**

## Scope

Seasons: 2018/19 through 2025/26; outer folds: 6.
This run directly tests competition multipliers inside the normal zero-sum
match Elo update. It does not add a second win or progression bonus.

```text
Delta_match = K_core * C_competition * (S - E)
```

The neutral 1/1/1 core is only the comparator. Every strict candidate satisfies
UCL=1.00 > UEL > UECL with a minimum 0.10 UEL-UECL gap. The requested
1.00/0.65/0.45 reference is reported independently in every unseen fold.

## Fold Selections

| Fold | Unseen season | Selected UEL | Selected UECL | Strict UEL | Strict UECL | Strict train diff |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 2020/21 | 1 | 1 | 0.95 | 0.85 | 0.000015 |
| 2 | 2021/22 | 1 | 1 | 0.95 | 0.85 | 0.000028 |
| 3 | 2022/23 | 1 | 1 | 0.95 | 0.85 | 0.000033 |
| 4 | 2023/24 | 1 | 1 | 0.95 | 0.85 | 0.000007 |
| 5 | 2024/25 | 1 | 1 | 0.95 | 0.85 | 0.000011 |
| 6 | 2025/26 | 0.95 | 0.7 | 0.95 | 0.7 | -0.000072 |

## Unseen Comparison

Negative Brier differences favor the multiplier model.

| Model | Fold wins | Mean difference | CI lower | CI upper |
| --- | ---: | ---: | ---: | ---: |
| Nested selected | 1/6 | -0.000083 | -0.000220 | 0.000040 |
| Forced strict | 3/6 | -0.000138 | -0.000308 | 0.000034 |
| Reference 1/.65/.45 | 2/6 | 0.000211 | -0.000356 | 0.000797 |

## Competition Segments

| Model | Competition | Matches | Brier difference | Log-loss difference |
| --- | --- | ---: | ---: | ---: |
| Forced strict | UCL | 1384 | 0.000000 | 0.000000 |
| Forced strict | UECL | 2073 | -0.000369 | -0.000728 |
| Forced strict | UEL | 1427 | 0.000063 | 0.000108 |
| Reference | UCL | 1384 | 0.000000 | 0.000000 |
| Reference | UECL | 2073 | -0.000118 | 0.000108 |
| Reference | UEL | 1427 | 0.000894 | 0.001771 |

## Parameter Stability

| Parameter | Mode | Fold frequency | Range |
| --- | ---: | ---: | ---: |
| selected_uel_multiplier | 1 | 5/6 | 0.95-1 |
| selected_uecl_multiplier | 1 | 5/6 | 0.7-1 |

## Full-Data Results

All-candidate selection: `UCL=1`, `UEL=0.95`, 
`UECL=0.65`; Brier=0.163522.
Best forced strict: `UCL=1`, `UEL=0.95`, 
`UECL=0.65`; 
Brier=0.163522.

| Competition | Strict multiplier | Strict effective K | Reference multiplier | Reference effective K |
| --- | ---: | ---: | ---: | ---: |
| UCL | 1 | 28 | 1 | 28 |
| UEL | 0.95 | 26.6 | 0.65 | 18.2 |
| UECL | 0.65 | 18.2 | 0.45 | 12.6 |

Promotion requires at least 5/6 nested unseen-fold wins, an overall paired
95% interval below zero, no reliably harmed competition, stable parameters and
ranking guardrails. Domain preference alone cannot override these checks.
