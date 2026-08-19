# Current Model External Benchmark

Canonical external evidence package for the active production contract. Every
other evaluation in this repository compares the model against its own ablation
arms; this package compares it against two independent outside references.

- `benchmark_report.md`: both axes, decision-relevant numbers and interpretation limits.
- `benchmark_manifest.json`: machine-readable headline result and reliability flags.
- `prediction_model_summary.csv`: pooled 1X2 losses and skill over walk-forward climatology.
- `prediction_uncertainty.csv`: dependency-robust conservative envelopes, AO minus ClubElo.
- `prediction_competition_summary.csv`: UCL, UEL and UECL breakdown.
- `prediction_clubelo_fits.csv`: training-only ClubElo calibration per outer season.
- `prediction_coverage.csv`: source, paired and scored match counts per season.
- `rating_model_summary.csv`: AO First Elo, pre-season Opta and the UEFA coefficient against realized performance.
- `rating_comparisons.csv`: paired Spearman differences with bootstrap intervals.

## Two axes, three reference types

| Axis | Reference | `reference_type` | Question |
| --- | --- | --- | --- |
| Prediction | ClubElo, paired exact-date sample | external | Are the served 1X2 probabilities competitive? |
| Rating | Opta Power Rankings, `2025-07-03` | `EXTERNAL` | Is the season-start seed competitive? |
| Rating | UEFA club coefficient, pre-season | `OWN_INPUT` | Does the static pipeline beat its own raw input? |

The UEFA arm is **not** an independent benchmark and must never be reported as
one. `club_points_t_minus_4 … club_points_t` in `club_european_points.csv` are
the five seasonal components of that same coefficient, so the model is scored
against something it consumes. It answers the value-added floor question
instead: a static pipeline that only matches its own input has no reason to
exist. A unit test pins this labelling so a later change cannot quietly promote
it to `EXTERNAL`.

The published 2026 UEFA coefficient in `data/external_rankings_2025_26/` is
deliberately unused here: its five-season window already contains 2025/26, the
season being predicted.

Both sides receive the identical score-preserving draw model, so axis 1 measures
rating quality rather than draw-model quality. ClubElo's home advantage is fitted
on earlier seasons only. A second, deliberately generous ClubElo arm gets both
its scale and home advantage fitted.

Axis 2 scores both ratings against the leave-team-out, venue- and
schedule-adjusted realized season performance, which is independent of both.

## Relationship to the other external scripts

- `run_external_elo_benchmark.py` scores a historical v1-scale candidate with
  legacy expected-score Brier. Superseded for current-model questions.
- `run_initial_elo_external_comparison_2025_26.py` measures AO/Opta *agreement*
  and deliberately excludes match results. This package adds the missing
  question: which of the two is actually more predictive.

## Regenerate

```bash
python3 scripts/run_current_external_benchmark.py
```

Outputs land in `output/current_external_benchmark/`; copy the files listed above
into this directory when the result should become curated evidence. Rerun after
any change to the dynamic core, the 1X2 layer, the prediction ensemble or the
AO First Elo formula.
