# AO European Elo Priority Backtest

The 2025/26 season is a locked holdout and is not used for candidate selection.
A fixed `70` Elo home advantage was selected on development seasons using the v1.1 baseline and then held constant.
Lower log loss and Brier are better; higher accuracy and Spearman are better.

## Stage selections

| Stage | Selected candidate | Development log loss | 2024/25 log loss |
| --- | --- | ---: | ---: |
| season_weights | weights_decay_0.85 | 0.642325 | 0.643436 |
| european_prior | europe_b28_boost420 | 0.640831 | 0.642911 |
| exposure_cap | exposure_cap_0.80 | 0.640797 | 0.642517 |
| exposure_blend | exposure_season_0.70 | 0.640659 | 0.642424 |
| country_strength | country_b15_g1.2_c180 | 0.634351 | 0.635894 |

## Locked holdout

| Model | Log loss | Brier | Decisive accuracy | Season Spearman |
| --- | ---: | ---: | ---: | ---: |
| v1.1 baseline | 0.659005 | 0.184121 | 0.635 | 0.314 |
| UEFA official coefficient (official_multiplier_2.75) | 0.663258 | 0.186254 | 0.636 | 0.270 |
| Selected candidate | 0.647974 | 0.179170 | 0.640 | 0.346 |

Holdout log-loss delta (selected - baseline): -0.011031.

## Selected config fields

- season_weights: `{'t_minus_4': 0.14075442253796464, 't_minus_3': 0.1655934382799584, 't_minus_2': 0.19481580974112753, 't_minus_1': 0.22919507028367947, 't': 0.26964125915726994}`
- european_history_benchmark: `28.0`
- european_prior_max_boost: `420.0`
- max_european_exposure: `0.8`
- exposure season/match: `0.7/0.30000000000000004`
- country_strength_benchmark: `15.0`
- gamma: `1.2`
- domestic_league_component: `180.0`

## Interpretation boundary

Stage A lacks complete domestic position and league-size data for non-champions. Do not change domestic achievement parameters from this result. A parameter should be promoted into the production config only when development, validation, and locked holdout metrics improve consistently rather than from the development winner alone.

## Independent parameter decisions

These checks change one parameter group at a time from v1.1, avoiding compensation between unrelated parameters.

| Candidate | Dev log-loss delta | Holdout delta | Seasons improved | Decision |
| --- | ---: | ---: | ---: | --- |
| season_weights_decay_0.85 | -0.000032 | +0.002014 | 2/5 | KEEP_CURRENT |
| european_history_benchmark_28 | -0.001745 | -0.004090 | 5/5 | STAGE_A_CANDIDATE |
| max_european_exposure_0.80 | -0.000416 | -0.000946 | 3/5 | KEEP_CURRENT |
| exposure_blend_0.70_0.30 | -0.000081 | -0.000032 | 5/5 | KEEP_CURRENT |
| country_b15_g1.2_c180 | -0.006999 | -0.008998 | 5/5 | BLOCKED |

Only `european_history_benchmark_28` is retained as a Stage A candidate. It is not promoted to the main config until the domestic-complete Stage B rerun.
