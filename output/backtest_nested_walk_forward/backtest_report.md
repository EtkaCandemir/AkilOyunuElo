# AO European Elo Nested Walk-Forward Backtest

Each outer fold selects parameters using past seasons only and evaluates the
selected configuration on the immediately following unseen season.
Home advantage remains fixed at `70` Elo.
Lower log loss and Brier are better.

## Fold selections

| Fold | Train | Test | Country candidate | Domestic candidate |
| ---: | --- | --- | --- | --- |
| 1 | 2021/22, 2022/23 | 2023/24 | country_b20_g2.0_c400 | domestic_c240_a0.0 |
| 2 | 2021/22, 2022/23, 2023/24 | 2024/25 | country_b17.5_g2.2_c360 | domestic_c240_a0.0 |
| 3 | 2021/22, 2022/23, 2023/24, 2024/25 | 2025/26 | country_b20_g2.0_c360 | domestic_c200_a0.0 |

## Pooled unseen-season results

| Model | Matches | Log loss | Delta vs baseline | Brier | Folds improved |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 2724 | 0.635459 | +0.000000 | 0.172770 | 0/3 |
| combined_selected | 2724 | 0.624499 | -0.010960 | 0.167648 | 3/3 |
| country_candidate_fixed | 2724 | 0.623617 | -0.011843 | 0.167312 | 3/3 |
| country_selected | 2724 | 0.624315 | -0.011145 | 0.167614 | 3/3 |

## Competition split

| Competition | Model | Matches | Log loss | Delta | Folds improved |
| --- | --- | ---: | ---: | ---: | ---: |
| UCL | baseline | 774 | 0.628144 | +0.000000 | 0/3 |
| UCL | combined_selected | 774 | 0.627589 | -0.000555 | 2/3 |
| UCL | country_candidate_fixed | 774 | 0.624251 | -0.003893 | 2/3 |
| UCL | country_selected | 774 | 0.626004 | -0.002140 | 2/3 |
| UECL | baseline | 1235 | 0.639789 | +0.000000 | 0/3 |
| UECL | combined_selected | 1235 | 0.627942 | -0.011847 | 3/3 |
| UECL | country_candidate_fixed | 1235 | 0.627630 | -0.012158 | 3/3 |
| UECL | country_selected | 1235 | 0.627936 | -0.011853 | 3/3 |
| UEL | baseline | 715 | 0.635901 | +0.000000 | 0/3 |
| UEL | combined_selected | 715 | 0.615208 | -0.020693 | 3/3 |
| UEL | country_candidate_fixed | 715 | 0.615998 | -0.019903 | 3/3 |
| UEL | country_selected | 715 | 0.616231 | -0.019670 | 3/3 |

## Parameter stability

| Parameter | Value | Folds selected |
| --- | ---: | ---: |
| country_strength_benchmark | 20 | 2/3 |
| country_strength_benchmark | 17.5 | 1/3 |
| gamma | 2 | 2/3 |
| gamma | 2.2 | 1/3 |
| domestic_league_component | 360 | 2/3 |
| domestic_league_component | 400 | 1/3 |
| domestic_achievement_component | 240 | 2/3 |
| domestic_achievement_component | 200 | 1/3 |
| achievement_alpha | 0 | 3/3 |

## Paired uncertainty

Match-level paired bootstrap intervals condition on the evaluated configurations;
they do not include parameter-selection uncertainty.

| Competition | Comparison | Mean delta | 95% interval | Reliable direction |
| --- | --- | ---: | ---: | --- |
| ALL | country_candidate_fixed_vs_baseline | -0.011843 | [-0.017177, -0.006346] | YES |
| ALL | country_selected_vs_baseline | -0.011145 | [-0.016948, -0.005322] | YES |
| ALL | combined_selected_vs_baseline | -0.010960 | [-0.018037, -0.003866] | YES |
| ALL | combined_selected_vs_country_selected | +0.000185 | [-0.001316, +0.001627] | NO |
| UCL | country_candidate_fixed_vs_baseline | -0.003893 | [-0.012824, +0.004759] | NO |
| UCL | country_selected_vs_baseline | -0.002140 | [-0.012106, +0.007874] | NO |
| UCL | combined_selected_vs_baseline | -0.000555 | [-0.013035, +0.012265] | NO |
| UCL | combined_selected_vs_country_selected | +0.001585 | [-0.001322, +0.004649] | NO |
| UEL | country_candidate_fixed_vs_baseline | -0.019903 | [-0.032546, -0.007357] | YES |
| UEL | country_selected_vs_baseline | -0.019670 | [-0.032961, -0.006449] | YES |
| UEL | combined_selected_vs_baseline | -0.020693 | [-0.036103, -0.004873] | YES |
| UEL | combined_selected_vs_country_selected | -0.001023 | [-0.003539, +0.001636] | NO |
| UECL | country_candidate_fixed_vs_baseline | -0.012158 | [-0.019175, -0.004512] | YES |
| UECL | country_selected_vs_baseline | -0.011853 | [-0.019614, -0.003563] | YES |
| UECL | combined_selected_vs_baseline | -0.011847 | [-0.021606, -0.001647] | YES |
| UECL | combined_selected_vs_country_selected | +0.000006 | [-0.002151, +0.002357] | NO |

## Decision rule

- Combined model improves all unseen folds: `PASS`.
- Combined model improves pooled UCL, UEL, and UECL: `PASS`.
- Fixed country candidate improves `3/3` unseen folds with pooled delta `-0.011843`.
- Stable country center: benchmark `20`, gamma `2`, league component `360`.
- Domestic achievement incremental delta: `+0.000185` with 95% interval `[-0.001316, +0.001627]`.
- Reject production promotion despite aggregate improvement: the zero-exposure
  segment can receive implausibly dominant ratings from the enlarged country prior.
- Keep the v1.1 country and domestic defaults until exposure-segment guardrails pass.
