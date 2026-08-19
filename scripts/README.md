# Script Guide

Scripts remain in one directory because several backtest runners import shared
helpers from one another. Moving them into subdirectories would change Python
import paths without improving model behavior.

Naming conventions:

- `build_*`: create datasets, audits, or documents.
- `run_*_backtest.py`: historical model evaluation.
- `run_*_pilot.py`: controlled smoke and scenario tests.
- `run_*_replay.py`: chronological season or match replay.
- `verify_*` and `validate_*`: contract and data-quality checks.

The current full evaluation entry point is:

```bash
python3 scripts/run_current_model_evaluation.py
```

The current external validation entry point is:

```bash
python3 scripts/run_current_external_benchmark.py
```

It scores the active contract against ClubElo and the pre-season Opta snapshot.
The two older external runners remain for their original questions:
`run_external_elo_benchmark.py` holds the historical v1-scale comparison, and
`run_initial_elo_external_comparison_2025_26.py` measures AO/Opta rank agreement
without using match results.

The static rating challenger is:

```bash
python3 scripts/run_cup_achievement_backtest.py
```

It scores a weighted domestic cup contribution against the active `max` rule on
both the season-start rating axis and a full dynamic replay. The layer is
`KEEP_SHADOW`; the run writes evidence and changes no production parameter.

Qualification-round K research is reproduced with:

```bash
python3 scripts/run_qualification_stage_k_backtest.py
```

The all-`1.00` production K profile is a reference only. Seven strictly
increasing Q1/Q2/Q3/qualifying-play-off profiles are eligible for nested
selection; the script never changes the production contract.

The incremental stage-K plus qualifier-carry comparison is reproduced with:

```bash
python3 scripts/run_qualification_stage_k_carry_backtest.py
```

This historical research runner compares stage-K and main-entry carry
candidates. The active production contract now embeds `%50` retention in each
qualifier match's effective K and performs no MAIN-entry reset; the runner
remains read-only and does not represent the active runtime behavior.

The active effective multipliers are Q1 `0.20`, Q2 `0.275`, Q3 `0.35`,
qualifying play-off `0.425`, and MAIN `1.00`. Preliminary Round uses Q1. A
club dropping from UCL to UEL or UECL keeps the same continuous Power state.

The frozen 2026/27 preproduction input snapshot is rebuilt from its local
source cache with:

```bash
python3 scripts/build_2026_27_preproduction_inputs.py --offline
```

The builder validates all 237 participant identities, emits the four AO First
Elo input contracts, separates completed matches from upcoming play-off
fixtures, and records unresolved domestic history explicitly. Use `--refresh`
only when intentionally taking a newer UEFA/Kassiesa source snapshot.

Completed Q1-Q3 matches are replayed with the active production contract via:

```bash
python3 scripts/run_2026_27_preproduction_replay.py
```

This is a retrospective preproduction reconstruction, not a prospective
prediction ledger. At the current cutoff it applies continuous retained
Q1/Q2/Q3 effective K multipliers, uses goal-margin fallback because validated
two-sided xG is unavailable, and never changes rating at MAIN entry without a
match.
