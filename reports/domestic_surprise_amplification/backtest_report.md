# Domestic Surprise Amplification Backtest

Decision: **KEEP_CURRENT**. Production contract was not changed.

## Research contract

All team-season inputs were rebuilt with `AOEuropeanEloConfig.active()` and effective European exposure cap `0.65`. The active control is theta `0.40`, variance penalty `0.50`, domestic cap `30`, linear exposure, and final effect cap `±75`.

## Control-artifact audit

The stored production Domestic Surprise artifact was written under the previous European exposure cap. Lowering the cap raises the domestic weight `1 - exposure` for every club-season sitting at it, so the rebuilt control cannot equal the stored artifact by construction. The audit therefore separates the two questions.

- Below the cap, where the domestic weight is untouched, the rebuild reproduces the stored artifact to `5.36e-13` Elo.
- Every non-zero difference sits at the cap: `True`. The largest is `6.000000` Elo, the arithmetic consequence of the weight moving from `1 - 0.85` to `1 - 0.65` on a club at the `±30` domestic cap.

This run uses `rebuilt_current_contract` as its seed baseline. `baseline_seed_reconciliation.csv` records the full migration delta without overwriting production.

## AO Core pooled result

- Brier difference: `-0.000011`
- Log-loss difference: `-0.000032`
- Fold wins: Brier `4/6`, log-loss `4/6`.

## Ranking

Ranking scores are higher-is-better, so the candidate harms ranking when the difference is negative. This is the opposite of the loss convention and the veto is evaluated accordingly.

- Reliably harmed metrics: `['same_season_pairwise_accuracy', 'same_season_spearman']`

## Selection stability

A surface this wide can always produce a per-fold winner. If the winners disagree there is no parameter set to ship, whatever the pooled loss says.

- Folds: `6`, distinct selected candidates: `5`, modal fold share: `0.333` (gate `>= 0.50`).
- theta spanned `0.4`-`1.75`, domestic cap `30`-`150`, variance penalty `0`-`0.75`, across `3` exposure families.

## Served 1X2 boundary

The served ML/Poisson model is intentionally not re-trained or re-selected in this study. Its historical feature/artifact replay must be candidate-seed-aware before this change can be promoted as a served-prediction change; this script therefore treats AO Core results as the valid evidence line rather than inventing a proxy.

## Outputs

See `candidate_surface.csv`, `fold_selections.csv`, `fold_results.csv`, `effect_distribution.csv`, `competition_summary.csv`, `ranking_summary.csv`, `ranking_uncertainty.csv`, `dependency_uncertainty.csv`, and `safety_audit.csv`.
