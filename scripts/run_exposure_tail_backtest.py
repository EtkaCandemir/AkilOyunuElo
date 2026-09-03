from __future__ import annotations

"""Walk-forward test of the exposure upper tail above the 0.65 cap.

Production computes the blend weight as ``min(raw_exposure, 0.65)``, which
collapses 25 distinct raw exposures onto one weight: a club that played every
European season and a club that played two thirds of them are told apart in
their priors but not in how much those priors are trusted.

`exposure_tail_beta` already exists in the production config at `0.0`, which is
exactly that hard cap. This command measures raising it. The tail is the same
shape that was promoted for the European history norm on 2026-08-30:

    weight = raw                              if raw <= 0.65
    weight = 0.65 + beta * (raw - 0.65)       if raw >  0.65

The distinguishing property, and the reason the linear and soft-power families
in `run_linear_european_exposure_backtest.py` could not have it, is that every
club at or below the cap is left bit-for-bit alone. Only the 74 clubs the cap
actually truncates can move, and they can only move up.

Research only: the active contract is hashed before and after, and no contract
or artifact is written.
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
    current_snapshot_impact,
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


OUTPUT_ROOT = ROOT / "output" / "exposure_tail_backtest_2018_2026"
# Reaches 1.0, which removes the cap above the knee entirely, so the selection
# has an open edge on the permissive side and cannot be pinned there by the
# grid the way the European tail grid first was at 0.75.
TAIL_BETAS = (0.10, 0.20, 0.35, 0.50, 0.75, 1.00)


def candidate_grid():
    return (
        BASELINE_CONFIG,
        *(
            replace(
                BASELINE_CONFIG,
                exposure_family="CAP_TAIL",
                exposure_tail_beta=beta,
            )
            for beta in TAIL_BETAS
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure the European exposure upper tail above the cap"
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples must be at least 100")
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract_hash = sha256(PRODUCTION_CONTRACT)

    print("Loading frozen production replay and static seed evidence", flush=True)
    _, datasets, core, parameters, production_seed_map = load_production_baseline()
    seasons = tuple(data.season for data in datasets)
    folds = expanding_folds(seasons)
    events = read_events(EVENTS_PATH)
    target_by_competition = schedule_adjusted_team_performance(events)
    target = aggregate_target(target_by_competition)
    xg_map = load_xg_map(XG_DATA, datasets)
    seeds = load_seed_evidence(production_seed_map)

    configs = candidate_grid()
    keys = [config.key for config in configs]
    if len(set(keys)) != len(keys):
        raise ValueError("Candidate keys are not unique")
    if BASELINE_CONFIG.key not in set(keys):
        raise ValueError("Grid must contain the live production baseline")

    predictions: dict[str, pd.DataFrame] = {}
    candidate_seeds: dict[str, pd.DataFrame] = {}
    surface_rows: list[dict[str, object]] = []
    exposure_rows: list[dict[str, object]] = []
    for index, config in enumerate(configs, start=1):
        candidate = apply_european_prior_recalibration(seeds, config)
        # The tail must never touch a club at or below the cap, and must never
        # lower one above it. A silent violation here would invalidate every
        # number downstream, so it is checked per candidate rather than trusted.
        below = seeds["european_exposure"] <= BASELINE_CONFIG.exposure_cap
        base_frame = apply_european_prior_recalibration(seeds, BASELINE_CONFIG)
        untouched = (
            candidate.loc[below, "candidate_ao_first_elo"]
            - base_frame.loc[below, "candidate_ao_first_elo"]
        ).abs().max()
        if float(untouched) > 1e-9:
            raise ValueError(
                f"{config.key}: the tail moved a club at or below the cap by "
                f"{untouched}"
            )
        # The invariant is on the weight, not on the rating. Raising the weight
        # lowers a club whose European Prior sits below its Domestic Prior, and
        # that is correct: more trust in European evidence, and that club's
        # European evidence is the worse of the two. The history-norm tail could
        # only ever raise a prior; this one cannot make that promise.
        raised = (
            candidate["candidate_effective_exposure"]
            - base_frame["candidate_effective_exposure"]
        ).min()
        if float(raised) < -1e-12:
            raise ValueError(f"{config.key}: the tail lowered a weight by {raised}")

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
        moved = (
            candidate["candidate_ao_first_elo"]
            - base_frame["candidate_ao_first_elo"]
        )
        exposure_rows.append(
            {
                "candidate_key": config.key,
                "exposure_family": config.exposure_family,
                "exposure_tail_beta": config.exposure_tail_beta,
                "clubs_moved": int((moved.abs() > 1e-9).sum()),
                "clubs_total": int(len(moved)),
                "mean_move_elo": float(moved[moved.abs() > 1e-9].mean())
                if bool((moved.abs() > 1e-9).any())
                else 0.0,
                "max_move_elo": float(moved.max()),
            }
        )
        print(f"Replayed {index}/{len(configs)} candidates", flush=True)

    surface = pd.DataFrame(surface_rows)
    selections = nested_selection(surface, configs, folds)
    unseen, fold_results = build_unseen(
        selections, predictions, candidate_seeds, target, folds
    )
    summary = model_summary(unseen, fold_results)
    competition = competition_summary(unseen)
    uncertainty = uncertainty_summary(unseen, int(args.bootstrap_samples))
    ranking_uncertainty = ranking_uncertainty_summary(
        fold_results, int(args.bootstrap_samples)
    )
    candidate_summary = aggregate_candidate_surface(surface)
    full_selected = select_candidate(candidate_summary)
    current_impact = current_snapshot_impact(
        {config.key: config for config in configs}[full_selected]
    )

    selected = summary.loc[summary["model"].eq("NESTED_RECALIBRATION")].iloc[0]
    brier_wins = int((fold_results["delta_brier_1x2"] < 0.0).sum())
    log_wins = int((fold_results["delta_log_loss_1x2"] < 0.0).sum())
    rank_harm = bool(ranking_uncertainty["reliable_harm"].any())
    segment_harm = bool(
        (competition["delta_brier_1x2"] > 0.0).any()
        or (competition["delta_log_loss_1x2"] > 0.0).any()
    )
    decision = (
        "PROMOTE_CANDIDATE"
        if brier_wins >= 4
        and log_wins >= 4
        and selected["delta_vs_baseline_brier_1x2"] < 0.0
        and selected["delta_vs_baseline_log_loss_1x2"] < 0.0
        and not rank_harm
        and not segment_harm
        else "KEEP_PRODUCTION"
    )
    payload = {
        "decision": decision,
        "production_changed": False,
        "candidate_count": len(configs),
        "fold_count": len(folds),
        "unseen_matches": int(summary["matches"].max()),
        "full_history_candidate_key": full_selected,
        "nested_selected_keys": selections["selected_candidate_key"].tolist(),
        "brier_fold_wins": brier_wins,
        "log_loss_fold_wins": log_wins,
        "ranking_reliable_harm": rank_harm,
        "competition_loss_harm": segment_harm,
        "pooled_brier_delta": float(selected["delta_vs_baseline_brier_1x2"]),
        "pooled_log_loss_delta": float(selected["delta_vs_baseline_log_loss_1x2"]),
        "pooled_seed_spearman_delta": float(
            selected["delta_vs_baseline_seed_spearman"]
        ),
        "pooled_seed_pairwise_delta": float(
            selected["delta_vs_baseline_seed_pairwise_accuracy"]
        ),
        "production_contract_sha256": contract_hash,
    }

    surface.to_csv(output / "candidate_surface.csv", index=False)
    candidate_summary.to_csv(output / "candidate_summary.csv", index=False)
    pd.DataFrame(exposure_rows).to_csv(output / "tail_reach.csv", index=False)
    selections.to_csv(output / "fold_selections.csv", index=False)
    fold_results.to_csv(output / "fold_results.csv", index=False)
    summary.to_csv(output / "model_comparison.csv", index=False)
    competition.to_csv(output / "competition_summary.csv", index=False)
    uncertainty.to_csv(output / "dependency_uncertainty.csv", index=False)
    ranking_uncertainty.to_csv(output / "ranking_uncertainty.csv", index=False)
    current_impact.to_csv(output / "current_2026_27_impact.csv", index=False)
    (output / "decision.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    if sha256(PRODUCTION_CONTRACT) != contract_hash:
        raise ValueError("Production contract changed during research backtest")
    print(f"Decision: {decision}")
    print(f"Full-history candidate: {full_selected}")


if __name__ == "__main__":
    main()
