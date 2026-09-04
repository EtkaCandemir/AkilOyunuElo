from __future__ import annotations

"""Single-axis sweep of the competition quality weights on the European Prior.

    European Prior = 500 + 1559.7148 * quality * history_norm
    quality: UCL 1.0, UEL uel_quality, UECL uecl_quality

Production runs 1.0/1.0/1.0, which says a point earned in the Conference League
is worth a point earned in the Champions League. UEFA's own schedule disagrees
mildly through its stage bonuses, but per match the two are close, and the
club's own history is summed without any competition mark at all.

The axis was swept once, in `run_ranking_first_calibration.py`, jointly with the
benchmark and the league component - and 1-1-1 won there. That is exactly what
happened to gamma in the same study, and gamma turned out to be a real gain once
it was measured alone with everything else pinned. This run gives quality the
same treatment.

Known limitation, carried from the research surface: `quality` is applied by the
club's entry competition *this* season, so a club with four Champions League
seasons that drops into the Conference League has its whole history discounted.
That is a proxy, not a decomposition, and it caps how much this axis can be
trusted even if it wins.

Research only: the active contract is hashed before and after, and nothing in
`contracts/` or `artifacts/` is written.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ao_elo.european_prior_recalibration import (  # noqa: E402
    apply_european_prior_recalibration,
    ranking_uncertainty_summary,
)
from ao_elo.evaluation import schedule_adjusted_team_performance  # noqa: E402
from scripts.run_ao_first_seed_boost_backtest import aggregate_target  # noqa: E402
from scripts.run_current_model_evaluation import (  # noqa: E402
    EvaluationArm,
    evaluate_arm,
)
from scripts.run_dynamic_core_calibration import expanding_folds  # noqa: E402
from scripts.run_european_prior_recalibration_backtest import (  # noqa: E402
    BASELINE_CONFIG,
    aggregate_candidate_surface,
    build_unseen,
    competition_summary,
    load_seed_evidence,
    model_summary,
    nested_selection,
    season_surface,
    select_candidate,
    sha256,
    uncertainty_summary,
)
from scripts.run_opponent_quintile_backtest import load_production_baseline  # noqa: E402
from scripts.run_stage_weighted_progression_backtest import (  # noqa: E402
    EVENTS_PATH,
    PRODUCTION_CONTRACT,
    XG_DATA,
    load_xg_map,
)
from scripts.run_v2_evaluation_upgrade import read_events  # noqa: E402


OUTPUT_ROOT = ROOT / "output" / "competition_quality_backtest_2018_2026"
# `validate` enforces UEL >= UECL, so the grid keeps that ordering. It reaches a
# hard discount (0.80/0.60) so the selection has room to leave the production
# corner rather than being trapped next to it.
QUALITY_PROFILES = (
    (1.00, 0.95),
    (1.00, 0.90),
    (0.95, 0.90),
    (0.95, 0.85),
    (0.90, 0.80),
    (0.85, 0.75),
    (0.80, 0.60),
)


def candidate_grid():
    return (
        BASELINE_CONFIG,
        *(
            replace(BASELINE_CONFIG, uel_quality=uel, uecl_quality=uecl)
            for uel, uecl in QUALITY_PROFILES
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep the UEL and UECL quality weights on their own"
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract_hash = sha256(PRODUCTION_CONTRACT)

    print("Loading frozen production replay and static seed evidence", flush=True)
    _, datasets, core, parameters, production_seed_map = load_production_baseline()
    seasons = tuple(data.season for data in datasets)
    folds = expanding_folds(seasons)
    target_by_competition = schedule_adjusted_team_performance(read_events(EVENTS_PATH))
    target = aggregate_target(target_by_competition)
    xg_map = load_xg_map(XG_DATA, datasets)
    seeds = load_seed_evidence(production_seed_map)

    configs = candidate_grid()
    keys = [c.key for c in configs]
    if len(set(keys)) != len(keys):
        raise ValueError("Candidate keys are not unique")

    predictions: dict[str, pd.DataFrame] = {}
    candidate_seeds: dict[str, pd.DataFrame] = {}
    surface_rows: list[dict[str, object]] = []

    for index, config in enumerate(configs, start=1):
        candidate = apply_european_prior_recalibration(seeds, config)
        if config.key == BASELINE_CONFIG.key:
            error = (
                candidate["candidate_ao_first_elo"]
                - candidate["adjusted_ao_first_elo"]
            ).abs().max()
            if float(error) > 1e-8:
                raise ValueError(f"Baseline does not reproduce production: {error}")
        candidate_seeds[config.key] = candidate
        rating_map = {
            (str(row.season), int(row.team_id)): float(row.candidate_ao_first_elo)
            for row in candidate.itertuples(index=False)
        }
        evaluation = evaluate_arm(
            datasets,
            EvaluationArm(config.key, True, True, True, True, True),
            core=core,
            parameters=parameters,
            current_domestic=rating_map,
            baseline_domestic=production_seed_map,
            xg_map=xg_map,
            target=target_by_competition,
        )
        predictions[config.key] = evaluation.predictions
        surface_rows.extend(
            season_surface(config, candidate, evaluation.predictions, target, seasons)
        )
        print(f"Replayed {index}/{len(configs)} candidates", flush=True)

    surface = pd.DataFrame(surface_rows)
    selections = nested_selection(surface, configs, folds)
    unseen, fold_results = build_unseen(
        selections, predictions, candidate_seeds, target, folds
    )
    comparison = model_summary(unseen, fold_results)
    competition = competition_summary(unseen)
    uncertainty = uncertainty_summary(unseen, int(args.bootstrap_samples))
    ranking_uncertainty = ranking_uncertainty_summary(
        fold_results, int(args.bootstrap_samples)
    )
    candidate_summary = aggregate_candidate_surface(surface)
    full_selected = select_candidate(candidate_summary)

    selected = comparison.loc[comparison["model"].eq("NESTED_RECALIBRATION")].iloc[0]
    brier_wins = int((fold_results["delta_brier_1x2"] < 0.0).sum())
    log_wins = int((fold_results["delta_log_loss_1x2"] < 0.0).sum())
    chosen = selections["selected_candidate_key"].tolist()
    payload = {
        "decision": (
            "PROMOTE_CANDIDATE"
            if brier_wins >= 4
            and log_wins >= 4
            and selected["delta_vs_baseline_brier_1x2"] < 0.0
            and selected["delta_vs_baseline_log_loss_1x2"] < 0.0
            and not bool(ranking_uncertainty["reliable_harm"].any())
            and not bool(
                (competition["delta_brier_1x2"] > 0.0).any()
                or (competition["delta_log_loss_1x2"] > 0.0).any()
            )
            else "KEEP_PRODUCTION"
        ),
        "production_changed": False,
        "candidate_count": len(configs),
        "fold_count": len(folds),
        "unseen_matches": int(comparison["matches"].max()),
        "full_history_candidate_key": full_selected,
        "nested_selected_keys": chosen,
        "same_candidate_every_fold": len(set(chosen)) == 1,
        "brier_fold_wins": brier_wins,
        "log_loss_fold_wins": log_wins,
        "ranking_reliable_harm": bool(ranking_uncertainty["reliable_harm"].any()),
        "pooled_brier_delta": float(selected["delta_vs_baseline_brier_1x2"]),
        "pooled_seed_spearman_delta": float(
            selected["delta_vs_baseline_seed_spearman"]
        ),
        "quality_proxy_note": (
            "Quality is applied by the club's entry competition for the season, "
            "so a club that dropped down has its whole five-season history "
            "discounted. A win here is a reason to build the per-competition "
            "decomposition, not to activate this proxy."
        ),
        "production_contract_sha256": contract_hash,
    }

    unseen.to_csv(output / "unseen_predictions.csv", index=False)
    surface.to_csv(output / "candidate_surface.csv", index=False)
    candidate_summary.to_csv(output / "candidate_summary.csv", index=False)
    selections.to_csv(output / "fold_selections.csv", index=False)
    fold_results.to_csv(output / "fold_results.csv", index=False)
    comparison.to_csv(output / "model_comparison.csv", index=False)
    competition.to_csv(output / "competition_summary.csv", index=False)
    uncertainty.to_csv(output / "dependency_uncertainty.csv", index=False)
    ranking_uncertainty.to_csv(output / "ranking_uncertainty.csv", index=False)
    (output / "decision.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    if sha256(PRODUCTION_CONTRACT) != contract_hash:
        raise ValueError("Production contract changed during the sweep")
    print(f"Decision: {payload['decision']}")
    print(f"Full-history candidate: {full_selected}")


if __name__ == "__main__":
    main()
