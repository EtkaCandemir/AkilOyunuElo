# Data Layout

The repository stores compact, model-ready CSV inputs and audit manifests here.
Large provider responses are cached in `_source_cache/` directories and are not
versioned.

## Fixtures

- `pilot_10_teams/`, `pilot_20_teams/`, `real_pilot_10_teams/`: controlled
  smoke-test inputs.
- `dynamic_pilot/`: dynamic Elo replay fixture.

## Historical model inputs

- `backtest_2018_2026/`, `backtest_stage_b_2018_2026/`: walk-forward static
  rating inputs.
- `dynamic_backtest_2018_2026/`: dynamic match replay inputs.
- `external_elo_benchmark_2018_2026/`: exact-date event and external benchmark
  tables.
- `club_identity/`: persistent club identity registry and audits.

## Provider-derived analysis data

- `xg_2025_26/`, `xg_backtest_2018_2026/`: xG analysis tables.
- `thesportsdb_2025_26/`: match-detail extracts.
- `domestic_league_matches_2013_2026/`: domestic schedule/result evidence.
- `opta_league_strength/`: dated external league-strength snapshot.

Generated model results do not belong here; they are written to `output/`.
