# Current Model Evaluation

Canonical evaluation snapshot for the active AO European Elo production
contract.

Regenerated on 27 August 2026 against the active contract, so the tables
reflect continuous qualifier retention (effective Q1/Q2/Q3/QPO K multipliers
`0.20/0.275/0.35/0.425`, MAIN `1.00`, no MAIN-entry carry or reset), the
six-season xG map, and the three static changes activated in this revision:
the European prior participation normalization, the generalized domestic cup
contribution, and the unknown-position floor. For active behavior the
production contract and `active_model_snapshot.json` still take precedence
over replay labels here.

The AO rating core scores Brier `0.566413`, log-loss
`0.956259` and accuracy `0.559173` on `4884` unseen
matches. Two ablations show reliable harm on Brier when disabled - the
single-match draw correction and bounded xG - but the single-match sample is
`89%` COVID-season matches and its interval barely excludes zero, so that
layer is defended structurally rather than by sample power. Domestic Surprise
no longer shows reliable harm: the participation layer picks up part of the
same work, so its cost fell to `+0.000132` with an interval crossing zero. The
cup ablation flipped sign once the contribution was generalized, from
`-0.000080` to `+0.000022`.

Earlier copies of this package are stale in three known ways: those produced
before the exposure-cap propagation carry a `0.85`-seeded reference arm, those
produced before this revision carry an un-normalized European Prior, and those
before it also treat the cup as a floor.

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
