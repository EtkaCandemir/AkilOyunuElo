# AO European Elo Ranking-First Calibration

Ranking is the primary objective. Log loss is used only after ranking metrics.
Candidates must pass historical zero-exposure and real-pilot pairwise guardrails.
A candidate is promoted only if it improves every unseen fold, does not worsen
any competition, and the same configuration is selected in every fold.

## Fold selections

| Fold | Train | Test | Selected | Train Spearman | Pairwise | Top quartile |
| ---: | --- | --- | --- | ---: | ---: | ---: |
| 1 | 2021/22, 2022/23 | 2023/24 | country_b15_g2.0_c180 | 0.3912 | 0.6481 | 0.4538 |
| 2 | 2021/22, 2022/23, 2023/24 | 2024/25 | country_b15_g1.4_c180 | 0.4214 | 0.6613 | 0.4655 |
| 3 | 2021/22, 2022/23, 2023/24, 2024/25 | 2025/26 | country_b15_g1.6_c180 | 0.4247 | 0.6620 | 0.4736 |

## Unseen competition ranking

| Competition | Model | Spearman | Delta | Pairwise | Top quartile | Rank MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| UCL | baseline | 0.4443 | +0.0000 | 0.6719 | 0.4040 | 0.2357 |
| UCL | selected | 0.4338 | -0.0106 | 0.6665 | 0.4040 | 0.2397 |
| UECL | baseline | 0.3465 | +0.0000 | 0.6341 | 0.4401 | 0.2595 |
| UECL | selected | 0.3505 | +0.0040 | 0.6354 | 0.4571 | 0.2591 |
| UEL | baseline | 0.5126 | +0.0000 | 0.6972 | 0.5365 | 0.2252 |
| UEL | selected | 0.5354 | +0.0228 | 0.7064 | 0.5930 | 0.2196 |

## Promotion decision

- Decision: `KEEP_V1_1`
- Unseen folds improved: `2/3`
- Competitions not worse: `2/3`
- Same candidate selected in every fold: `False`
- Pilot ranking vetos pass: `True`

No aggregate prediction improvement can override a failed ranking veto.
