# Draw Shape Backtest

Canonical evidence package for the sub-unit draw-shape correction.

- `backtest_report.md`: walk-forward result and decision.
- `selected_candidate.json`: selected full-history parameters and gates.
- `fold_selections.csv`: training-only selection in each outer fold.
- `fold_results.csv`: baseline, fixed-draw shape, joint and prespecified arms.
- `arm_summary.csv`: pooled comparison of all four arms.
- `competition_summary.csv`: pooled UCL, UEL and UECL comparison.
- `calibration_deciles.csv`: expected-score decile draw calibration.
- `dependency_uncertainty.csv`: dependency-aware bootstrap intervals.
- `fixed_draw_shape_surface.csv`: full-history surface at draw-at-even 0.24.

Regenerate after the current-model evaluation with:

```bash
python3 scripts/run_draw_shape_backtest.py
```
