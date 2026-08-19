# Current Model Evaluation

Canonical evaluation snapshot for the active AO European Elo production
contract.

Regenerated on 19 August 2026 against the active contract, so the tables now
reflect continuous qualifier retention (effective Q1/Q2/Q3/QPO K multipliers
`0.20/0.275/0.35/0.425`, MAIN `1.00`, no MAIN-entry carry or reset) and the
six-season xG map. For active behavior the production contract and
`active_model_snapshot.json` still take precedence over replay labels here.

Two changes move every metric in this package relative to earlier copies. The
shared `XG_DATA` constant now points at `data/xg_2020_2026/`, which replays six
seasons with xG instead of one and makes the bounded xG layer the most valuable
ablation (`+0.000797` Brier when disabled, up from third place). A permanent
`ABLATION_NO_CUP_DOUBLE_BONUS` arm was also added; its cost is negative
(`-0.000055`), meaning the champion-and-cup double bonus currently earns
nothing measurable.

The `CURRENT_PRODUCTION` rows in this package are the AO rating-core 1X2 replay.
The served `%50 Current ML + %50 AO Domestic Poisson` prediction evidence is
versioned separately under `reports/production_prediction/`.

- `current_model_evaluation_report.md`: primary human-readable report.
- `active_model_snapshot.json`: resolved active parameters and layer order.
- `evaluation_manifest.json`: evaluation scope and source fingerprints.
- `model_summary.csv`: pooled production, reference, and ablation metrics.
- `fold_summary.csv`: season/fold metrics.
- `competition_summary.csv`: UCL, UEL, and UECL metrics.
- `dependency_uncertainty.csv`: dependency-aware bootstrap intervals.
- `initial_elo_impact_summary.csv`: initial Elo movement distribution.
- `exposure_impact_summary.csv`: adjustment by European Exposure band.
- `forward_ranking.csv`: following-season ranking diagnostics.
- `season_end_elo_impact_summary.csv`: final Live Elo movement summary.
- `safety_and_data_quality_audit.csv`: validation and invariant checks.

Format-duyarlı tek maç beraberlik katmanının parametre seçimi, COVID
duyarlılığı ve ayrı belirsizlik sonuçları `../single_match_draw_backtest/`
altındadır.

Regenerate the full local package with:

```bash
python3 scripts/run_current_model_evaluation.py
```
