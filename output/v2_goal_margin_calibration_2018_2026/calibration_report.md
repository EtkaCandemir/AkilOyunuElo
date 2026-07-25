# AO European Elo v2 Goal-Margin Calibration

Exact-date seasons: `2018/19` through `2025/26`.
The comparator is the selected dynamic core plus its nested season carry.
The score is the 90/120-minute field score; penalty shootout goals are excluded.

## Decision

- Result: `DISABLE_GOAL_MARGIN`
- Full-data candidate: weight `0.25`, cap `1.75`
- Active config: weight `0`, cap `1`
- Unseen fold wins: `4/6`
- Overall Brier difference: `-0.000168`
- Clustered 95% CI: `[-0.000602, +0.000289]`
- Ranking guardrail: `True`
- Zero-sum update guardrail: `True`

| Competition | Matches | Brier difference | Log-loss difference |
| --- | ---: | ---: | ---: |
| UCL | 1384 | -0.000483 | -0.001389 |
| UECL | 2073 | +0.000031 | -0.000068 |
| UEL | 1427 | -0.000151 | -0.000228 |

A disabled result means every production goal multiplier remains exactly 1.0.
The candidate is retained only as research evidence, not as an active layer.
