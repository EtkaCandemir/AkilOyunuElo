# AO European Elo Ranking-First Calibration

Ranking is the primary objective. Log loss is used only after ranking metrics.
Candidates must pass historical zero-exposure and real-pilot pairwise guardrails.
A candidate is promoted only if it improves every unseen fold, does not worsen
any competition, and the same configuration is selected in every fold.

## Fold selections

| Fold | Train | Test | Selected | Train Spearman | Pairwise | Top quartile |
| ---: | --- | --- | --- | ---: | ---: | ---: |
| 1 | 2018/19, 2019/20 | 2020/21 | country_b25_g2.0_c100 | 0.4519 | 0.6764 | 0.2960 |
| 2 | 2018/19, 2019/20, 2020/21 | 2021/22 | country_b35_g1.4_c100 | 0.4588 | 0.6829 | 0.3099 |
| 3 | 2018/19, 2019/20, 2020/21, 2021/22 | 2022/23 | country_b20_g1.6_c180 | 0.4370 | 0.6715 | 0.3495 |
| 4 | 2018/19, 2019/20, 2020/21, 2021/22, 2022/23 | 2023/24 | country_b15_g2.0_c180 | 0.4193 | 0.6635 | 0.3773 |
| 5 | 2018/19, 2019/20, 2020/21, 2021/22, 2022/23, 2023/24 | 2024/25 | country_b15_g1.4_c180 | 0.4309 | 0.6679 | 0.3959 |
| 6 | 2018/19, 2019/20, 2020/21, 2021/22, 2022/23, 2023/24, 2024/25 | 2025/26 | country_b15_g2.0_c180 | 0.4319 | 0.6677 | 0.4131 |

## Unseen competition ranking

| Competition | Model | Spearman | Delta | Pairwise | Top quartile | Rank MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| UCL | baseline | 0.4367 | +0.0000 | 0.6691 | 0.4520 | 0.2391 |
| UCL | selected | 0.4309 | -0.0058 | 0.6658 | 0.4520 | 0.2398 |
| UECL | baseline | 0.3484 | +0.0000 | 0.6351 | 0.3812 | 0.2594 |
| UECL | selected | 0.3480 | -0.0004 | 0.6347 | 0.3779 | 0.2580 |
| UEL | baseline | 0.4521 | +0.0000 | 0.6756 | 0.4880 | 0.2385 |
| UEL | selected | 0.4686 | +0.0165 | 0.6818 | 0.5162 | 0.2327 |

## Promotion decision

- Decision: `KEEP_V1_1`
- Unseen folds improved: `4/6`
- Competitions not worse: `1/3`
- Same candidate selected in every fold: `False`
- Pilot ranking vetos pass: `True`

No aggregate prediction improvement can override a failed ranking veto.
