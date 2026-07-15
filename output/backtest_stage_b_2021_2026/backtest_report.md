# AO European Elo Stage B Backtest

Stage B uses domestic positions and league sizes for every required CH/N participant.
Cup-only clubs outside the top division retain the model's explicit unknown-position path.
A fixed 70 Elo home advantage was selected on development data.
The 2025/26 season remained locked until all stage selections were complete.

## Stage selections

| Stage | Winner | Dev log loss | 2024/25 latest-dev log loss |
| --- | --- | ---: | ---: |
| season_weights | weights_current | 0.632969 | 0.633907 |
| european_prior | europe_b28_boost480 | 0.631614 | 0.633851 |
| exposure_cap | exposure_cap_0.75 | 0.631267 | 0.633081 |
| exposure_blend | exposure_season_0.70 | 0.631232 | 0.633181 |
| country_strength | country_b15_g2.2_c340 | 0.619476 | 0.622909 |
| domestic_scale | domestic_component_200_alpha_0.0 | 0.618441 | 0.621982 |
| percentile_shape | percentile_f0.05_s0.70_d1.3 | 0.618051 | 0.620857 |
| cup_values | cup_base_0.40_double_0.12 | 0.617871 | 0.620681 |

## Locked holdout

| Model | Log loss | Brier | Decisive accuracy | Spearman |
| --- | ---: | ---: | ---: | ---: |
| v1.1 baseline | 0.647321 | 0.179010 | 0.628 | 0.366 |
| UEFA official coefficient (official_multiplier_2.75) | 0.663258 | 0.186254 | 0.636 | 0.270 |
| Stage B grid winner | 0.628581 | 0.170476 | 0.672 | 0.405 |

## Independent decisions

| Parameter group | Dev delta | Holdout delta | Seasons improved | Decision |
| --- | ---: | ---: | ---: | --- |
| season_weights | +0.000000 | +0.000000 | 0/5 | KEEP_CURRENT |
| european_prior | -0.001355 | -0.000936 | 5/5 | CANDIDATE |
| exposure_cap | -0.000201 | -0.000996 | 3/5 | WEAK_EVIDENCE |
| exposure_blend | -0.000090 | -0.000003 | 4/5 | WEAK_EVIDENCE |
| country_strength | -0.010519 | -0.014275 | 5/5 | CANDIDATE |
| domestic_scale | -0.003539 | -0.004437 | 5/5 | CANDIDATE |
| percentile_shape | +0.001633 | +0.002540 | 0/5 | KEEP_CURRENT |
| cup_values | -0.000207 | +0.000521 | 3/5 | KEEP_CURRENT |

## Focused ablations

Each row starts from the unchanged v1.1 baseline.
Negative deltas improve log loss.

| Candidate | Dev delta | Holdout delta | Seasons improved |
| --- | ---: | ---: | ---: |
| country_b15_g1_8_c260 | -0.009689 | -0.013181 | 5/5 |
| achievement_c360_alpha0 | -0.006344 | -0.010694 | 5/5 |
| achievement_c280_alpha0 | -0.005842 | -0.008584 | 5/5 |
| league_component260_only | -0.003767 | -0.008442 | 5/5 |
| league_component300_only | -0.003270 | -0.008918 | 5/5 |
| achievement_alpha0_only | -0.001716 | -0.001598 | 5/5 |
| eu_b28_boost480 | -0.001355 | -0.000936 | 5/5 |
| achievement_component280_only | -0.000934 | -0.004503 | 5/5 |
| country_b15_only | -0.000796 | -0.000619 | 5/5 |
| eu_b28_only | -0.000390 | -0.002763 | 2/5 |
| country_gamma1_8_only | +0.000116 | -0.000527 | 3/5 |
| eu_boost480_only | +0.000671 | +0.003793 | 2/5 |

## Grid winner config

- season_weights: `{'t_minus_4': 0.07, 't_minus_3': 0.13, 't_minus_2': 0.2, 't_minus_1': 0.27, 't': 0.33}`
- country benchmark/gamma/component: `15.0/2.2/340.0`
- European benchmark/boost: `28.0/480.0`
- exposure cap and blend: `0.75` and `0.7/0.30000000000000004`
- domestic component/alpha: `200.0/0.0`
- percentile floor/scale/delta: `0.05/0.7/1.3`
- cup base/double: `0.4/0.12`

## Decision

- Keep the current season weights, exposure cap/blend, percentile shape, and cup values.
- Retain `European benchmark 28 / boost 480` as a small but consistent candidate.
- Retain country-strength and domestic-achievement scaling as strong Stage C candidates.
- Do not promote the full grid winner: several country configurations are nearly tied,
  and `achievement_alpha=0` is a boundary result whose exact value is not yet stable.
- Run nested walk-forward folds, competition splits, and paired uncertainty checks before
  changing production defaults.

The UEFA row is a scaled official club-coefficient baseline, not UEFA's own match
prediction model. The AO models are more predictive on this sample, but the result does
not establish universal superiority over UEFA methodology.

A grid winner is not automatically promoted. Production changes require consistent
independent improvement, a non-boundary optimum, and a meaningful effect size.
