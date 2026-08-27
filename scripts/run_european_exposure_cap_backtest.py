from __future__ import annotations

"""Focused lower-bound search for AO First Elo European exposure.

This research command keeps the production contract untouched and reuses the
frozen six-fold replay used by the broader European Prior recalibration.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ao_elo.european_prior_recalibration import (  # noqa: E402
    EuropeanPriorRecalibrationConfig,
    apply_european_prior_recalibration,
    exposure_refinement_grid,
    ranking_uncertainty_summary,
)
from ao_elo.evaluation import (  # noqa: E402
    dependency_robust_loss_difference_ci,
    schedule_adjusted_team_performance,
)
from scripts.run_ao_first_seed_boost_backtest import aggregate_target  # noqa: E402
from scripts.run_current_model_evaluation import EvaluationArm, evaluate_arm  # noqa: E402
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
)
from scripts.run_opponent_quintile_backtest import load_production_baseline  # noqa: E402
from scripts.run_stage_weighted_progression_backtest import (  # noqa: E402
    EVENTS_PATH,
    PRODUCTION_CONTRACT,
    XG_DATA,
    load_xg_map,
)
from scripts.run_v2_evaluation_upgrade import read_events  # noqa: E402


OUTPUT_ROOT = ROOT / "output" / "european_exposure_cap_backtest_2018_2026"
def fixed_unseen_curve(
    surface: pd.DataFrame,
    folds: tuple[tuple[tuple[str, ...], str], ...],
) -> pd.DataFrame:
    test_seasons = {test for _, test in folds}
    unseen = surface.loc[surface["season"].isin(test_seasons)].copy()
    summary = aggregate_candidate_surface(unseen)
    fold_wins = []
    baseline = unseen.loc[unseen["candidate_key"].eq(BASELINE_CONFIG.key)][
        ["season", "brier_1x2", "log_loss_1x2"]
    ].rename(
        columns={
            "brier_1x2": "baseline_brier_1x2",
            "log_loss_1x2": "baseline_log_loss_1x2",
        }
    )
    for key, frame in unseen.groupby("candidate_key", sort=True):
        compared = frame.merge(baseline, on="season", validate="one_to_one")
        fold_wins.append(
            {
                "candidate_key": key,
                "brier_fold_wins": int(
                    (compared["brier_1x2"] < compared["baseline_brier_1x2"]).sum()
                ),
                "log_loss_fold_wins": int(
                    (
                        compared["log_loss_1x2"]
                        < compared["baseline_log_loss_1x2"]
                    ).sum()
                ),
            }
        )
    result = summary.merge(pd.DataFrame(fold_wins), on="candidate_key")
    return result.sort_values(["history_benchmark", "exposure_cap"]).reset_index(
        drop=True
    )


def dependency_uncertainty(unseen: pd.DataFrame, samples: int) -> pd.DataFrame:
    rows = []
    for index, metric in enumerate(("brier", "log_loss")):
        row = dependency_robust_loss_difference_ci(
            unseen,
            difference_column=f"{metric}_difference",
            bootstrap_samples=samples,
            seed=20260823 + index,
        )
        row.insert(0, "metric", metric)
        rows.append(row)
    return pd.concat(rows, ignore_index=True)


def current_impact(
    config: EuropeanPriorRecalibrationConfig,
) -> pd.DataFrame:
    current = pd.read_csv(
        ROOT / "output" / "season_2026_27_preproduction" / "ao_first_elo_2026_27.csv"
    ).rename(columns={"ao_first_elo": "adjusted_ao_first_elo"})
    result = apply_european_prior_recalibration(current, config)
    result["current_rank"] = result["adjusted_ao_first_elo"].rank(
        ascending=False, method="min"
    ).astype(int)
    result["candidate_rank"] = result["candidate_ao_first_elo"].rank(
        ascending=False, method="min"
    ).astype(int)
    result["rank_change"] = result["current_rank"] - result["candidate_rank"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples must be at least 100")
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract_hash = hashlib.sha256(PRODUCTION_CONTRACT.read_bytes()).hexdigest()

    _, datasets, core, parameters, production_seed_map = load_production_baseline()
    seasons = tuple(data.season for data in datasets)
    folds = expanding_folds(seasons)
    events = read_events(EVENTS_PATH)
    target_by_competition = schedule_adjusted_team_performance(events)
    target = aggregate_target(target_by_competition)
    xg_map = load_xg_map(XG_DATA, datasets)
    seeds = load_seed_evidence(production_seed_map)
    configs = exposure_refinement_grid()

    predictions: dict[str, pd.DataFrame] = {}
    candidate_seeds: dict[str, pd.DataFrame] = {}
    surface_rows: list[dict[str, object]] = []
    for index, config in enumerate(configs, start=1):
        candidate = apply_european_prior_recalibration(seeds, config)
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
        print(f"Replayed {index}/{len(configs)}: {config.key}", flush=True)

    surface = pd.DataFrame(surface_rows)
    curve = fixed_unseen_curve(surface, folds)
    full_summary = aggregate_candidate_surface(surface)
    full_selected = select_candidate(full_summary)
    selections = nested_selection(surface, configs, folds)
    unseen, fold_results = build_unseen(
        selections, predictions, candidate_seeds, target, folds
    )
    model = model_summary(unseen, fold_results)
    competition = competition_summary(unseen)
    loss_uncertainty = dependency_uncertainty(unseen, int(args.bootstrap_samples))
    rank_uncertainty = ranking_uncertainty_summary(
        fold_results, int(args.bootstrap_samples)
    )
    config_by_key = {config.key: config for config in configs}
    impact = current_impact(config_by_key[full_selected])
    ferenc = impact.loc[impact["team_name"].str.contains("Ferenc", case=False)].iloc[0]

    production_curve = curve.loc[curve["history_benchmark"].eq(20.0)]
    production_loss_optimum = production_curve.sort_values(
        ["brier_1x2", "log_loss_1x2", "exposure_cap"], kind="stable"
    ).iloc[0]
    ranking_first_fixed = production_curve.loc[
        production_curve["delta_vs_baseline_seed_spearman"].ge(-0.002)
        & production_curve["delta_vs_baseline_seed_pairwise_accuracy"].ge(-0.002)
        & production_curve["brier_fold_wins"].ge(4)
        & production_curve["log_loss_fold_wins"].ge(4)
    ].sort_values(["brier_1x2", "log_loss_1x2", "exposure_cap"], kind="stable").iloc[0]

    nested = model.loc[model["model"].eq("NESTED_RECALIBRATION")].iloc[0]
    decision = {
        "decision": "PROMOTE_CANDIDATE"
        if (fold_results["delta_brier_1x2"] < 0).sum() >= 4
        and (fold_results["delta_log_loss_1x2"] < 0).sum() >= 4
        and float(nested["delta_vs_baseline_brier_1x2"]) < 0
        and float(nested["delta_vs_baseline_log_loss_1x2"]) < 0
        and not bool(rank_uncertainty["reliable_harm"].any())
        and not bool(
            (competition["delta_brier_1x2"] > 0).any()
            or (competition["delta_log_loss_1x2"] > 0).any()
        )
        else "KEEP_RESEARCH",
        "production_changed": False,
        "candidate_count": len(configs),
        "full_history_candidate_key": full_selected,
        "production_benchmark_loss_optimum_key": str(
            production_loss_optimum["candidate_key"]
        ),
        "ranking_first_fixed_candidate_key": str(
            ranking_first_fixed["candidate_key"]
        ),
        "nested_selected_keys": selections["selected_candidate_key"].tolist(),
        "brier_fold_wins": int((fold_results["delta_brier_1x2"] < 0).sum()),
        "log_loss_fold_wins": int(
            (fold_results["delta_log_loss_1x2"] < 0).sum()
        ),
        "pooled_brier_delta": float(nested["delta_vs_baseline_brier_1x2"]),
        "pooled_log_loss_delta": float(
            nested["delta_vs_baseline_log_loss_1x2"]
        ),
        "pooled_seed_spearman_delta": float(
            nested["delta_vs_baseline_seed_spearman"]
        ),
        "pooled_seed_pairwise_delta": float(
            nested["delta_vs_baseline_seed_pairwise_accuracy"]
        ),
        "ferencvaros_current_elo": float(ferenc["adjusted_ao_first_elo"]),
        "ferencvaros_candidate_elo": float(ferenc["candidate_ao_first_elo"]),
        "ferencvaros_current_rank": int(ferenc["current_rank"]),
        "ferencvaros_candidate_rank": int(ferenc["candidate_rank"]),
        "production_contract_sha256": contract_hash,
    }

    surface.to_csv(output / "candidate_surface.csv", index=False)
    curve.to_csv(output / "fixed_unseen_exposure_curve.csv", index=False)
    full_summary.to_csv(output / "full_history_candidate_summary.csv", index=False)
    selections.to_csv(output / "fold_selections.csv", index=False)
    fold_results.to_csv(output / "fold_results.csv", index=False)
    model.to_csv(output / "model_comparison.csv", index=False)
    competition.to_csv(output / "competition_summary.csv", index=False)
    loss_uncertainty.to_csv(output / "dependency_uncertainty.csv", index=False)
    rank_uncertainty.to_csv(output / "ranking_uncertainty.csv", index=False)
    impact.to_csv(output / "current_2026_27_impact.csv", index=False)
    (output / "selected_candidate.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "backtest_report.md").write_text(
        "\n".join(
            [
                "# European Exposure Cap Backtest",
                "",
                f"- Decision: `{decision['decision']}`",
                f"- Full-history candidate: `{full_selected}`",
                f"- Production benchmark loss optimum: `{decision['production_benchmark_loss_optimum_key']}`",
                f"- Ranking-first fixed candidate: `{decision['ranking_first_fixed_candidate_key']}`",
                f"- Nested Brier delta: {decision['pooled_brier_delta']:+.6f}",
                f"- Nested log-loss delta: {decision['pooled_log_loss_delta']:+.6f}",
                f"- Nested Spearman delta: {decision['pooled_seed_spearman_delta']:+.6f}",
                f"- Nested pairwise delta: {decision['pooled_seed_pairwise_delta']:+.6f}",
                f"- Ferencvaros: {decision['ferencvaros_current_elo']:.3f} -> {decision['ferencvaros_candidate_elo']:.3f}",
                "- Production changed: no",
                "",
                "The fixed unseen curve is in `fixed_unseen_exposure_curve.csv`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    if hashlib.sha256(PRODUCTION_CONTRACT.read_bytes()).hexdigest() != contract_hash:
        raise ValueError("Production contract changed during exposure backtest")
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
