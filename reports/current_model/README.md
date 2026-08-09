# Current Model Evaluation

Canonical evaluation snapshot for the active AO European Elo production
contract.

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

Regenerate the full local package with:

```bash
python3 scripts/run_current_model_evaluation.py
```
