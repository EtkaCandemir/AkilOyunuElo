from __future__ import annotations

"""Walk-forward research of European Prior scale, quality and exposure.

The active production contract is hashed before and after the run. Candidate
ratings affect only season-start seeds inside the historical replay.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ao_elo.config import AOEuropeanEloConfig  # noqa: E402
from ao_elo.european_prior_recalibration import (  # noqa: E402
    EuropeanPriorRecalibrationConfig,
    apply_european_prior_recalibration,
    candidate_grid,
    tail_and_domestic_grid,
    ranking_uncertainty_summary,
)
from ao_elo.evaluation import (  # noqa: E402
    dependency_robust_loss_difference_ci,
    schedule_adjusted_team_performance,
)
from ao_elo.pipeline import compute_ao_first_elo_from_csv  # noqa: E402
from scripts.run_ao_first_seed_boost_backtest import aggregate_target  # noqa: E402
from scripts.run_current_model_evaluation import (  # noqa: E402
    EvaluationArm,
    evaluate_arm,
    markdown_table,
)
from scripts.run_dynamic_core_calibration import expanding_folds  # noqa: E402
from scripts.run_external_ranking_comparison_2025_26 import pairwise_accuracy  # noqa: E402
from scripts.run_opponent_quintile_backtest import load_production_baseline  # noqa: E402
from scripts.run_stage_weighted_progression_backtest import (  # noqa: E402
    DOMESTIC_ADJUSTMENTS,
    DYNAMIC_MANIFEST,
    EVENTS_PATH,
    PRODUCTION_CONTRACT,
    STATIC_DATA_ROOT,
    XG_DATA,
    load_xg_map,
)
from scripts.run_v2_evaluation_upgrade import read_events  # noqa: E402


OUTPUT_ROOT = ROOT / "output" / "european_prior_recalibration_backtest_2018_2026"
CURRENT_RATINGS = (
    ROOT / "output" / "season_2026_27_preproduction" / "ao_first_elo_2026_27.csv"
)
# The baseline must be the active production configuration, otherwise the
# candidates are measured against a model that is no longer live. The exposure
# cap moved to 0.65 after the original study, so the dataclass defaults no
# longer describe production.
BASELINE_CONFIG = EuropeanPriorRecalibrationConfig(exposure_cap=0.65)
RANKING_TOLERANCE = 0.002
DEFAULT_BOOTSTRAP_SAMPLES = 1000


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test European Prior scale, competition quality and exposure"
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--grid",
        choices=("default", "tail-and-domestic"),
        default="default",
        help=(
            "default: the original 81-candidate study. tail-and-domestic: the "
            "two top-end-compression axes with every other axis pinned to the "
            "active production value."
        ),
    )
    parser.add_argument(
        "--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES
    )
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
    configs = (
        tail_and_domestic_grid()
        if args.grid == "tail-and-domestic"
        else candidate_grid()
    )
    if BASELINE_CONFIG.key not in {config.key for config in configs}:
        raise ValueError(
            f"Grid {args.grid!r} does not contain the production baseline "
            f"{BASELINE_CONFIG.key!r}. The original grid was built around the "
            "pre-0.65 exposure cap; its saved output stands as the evidence of "
            "that study and it cannot be re-run against current production."
        )
    config_by_key = {config.key: config for config in configs}

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
        if index == 1 or index % 10 == 0 or index == len(configs):
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
    ablation = axis_ablation(candidate_summary, full_selected, args.grid)
    current_impact = current_snapshot_impact(config_by_key[full_selected])
    impact = historical_impact(candidate_seeds[full_selected])
    decision = decide(
        selections,
        fold_results,
        summary,
        competition,
        uncertainty,
        ranking_uncertainty,
        full_selected,
        current_impact,
        contract_hash,
    )

    surface.to_csv(output / "candidate_surface.csv", index=False)
    candidate_summary.to_csv(output / "candidate_summary.csv", index=False)
    ablation.to_csv(output / "axis_ablation_summary.csv", index=False)
    selections.to_csv(output / "fold_selections.csv", index=False)
    fold_results.to_csv(output / "fold_results.csv", index=False)
    unseen.to_csv(output / "unseen_predictions.csv", index=False)
    summary.to_csv(output / "model_comparison.csv", index=False)
    competition.to_csv(output / "competition_summary.csv", index=False)
    uncertainty.to_csv(output / "dependency_uncertainty.csv", index=False)
    ranking_uncertainty.to_csv(output / "ranking_uncertainty.csv", index=False)
    impact.to_csv(output / "historical_seed_impact.csv", index=False)
    current_impact.to_csv(output / "current_2026_27_impact.csv", index=False)
    (output / "selected_candidate.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "backtest_report.md").write_text(
        build_report(
            decision,
            summary,
            fold_results,
            competition,
            uncertainty,
            ranking_uncertainty,
            ablation,
            current_impact,
        ),
        encoding="utf-8",
    )

    if sha256(PRODUCTION_CONTRACT) != contract_hash:
        raise ValueError("Production contract changed during research backtest")
    print(f"Decision: {decision['decision']}")
    print(f"Full-history candidate: {full_selected}")
    print(f"Report: {output / 'backtest_report.md'}")


def load_seed_evidence(
    production_seed_map: dict[tuple[str, int], float],
) -> pd.DataFrame:
    dynamic = json.loads(DYNAMIC_MANIFEST.read_text(encoding="utf-8"))
    static_config = AOEuropeanEloConfig(**dynamic["static_config"])
    adjustments = pd.read_csv(DOMESTIC_ADJUSTMENTS, low_memory=False)
    adjustment_columns = [
        "season",
        "team_id",
        "club_id",
        "team_name",
        "country_code",
        "competition",
        "adjusted_domestic_prior",
        "adjusted_ao_first_elo",
    ]
    rows = []
    for folder in sorted(STATIC_DATA_ROOT.glob("20??-??")):
        rating = compute_ao_first_elo_from_csv(
            folder / "teams.csv",
            folder / "country_coefficients.csv",
            folder / "domestic_context.csv",
            folder / "club_european_points.csv",
            static_config,
        )
        rows.append(
            rating[
                [
                    "season",
                    "team_id",
                    "weighted_european_history",
                    "weighted_season_exposure",
                    "european_exposure",
                    "effective_european_exposure",
                    "european_prior",
                    "domestic_prior",
                ]
            ]
        )
    static = pd.concat(rows, ignore_index=True)
    result = adjustments[adjustment_columns].merge(
        static, on=["season", "team_id"], validate="one_to_one"
    )
    expected = {(str(season), int(team)) for season, team in production_seed_map}
    observed = {
        (str(row.season), int(row.team_id)) for row in result.itertuples(index=False)
    }
    if len(result) != 1887 or observed != expected:
        raise ValueError("European prior seed evidence does not cover production seeds")
    baseline = apply_european_prior_recalibration(result, BASELINE_CONFIG)
    error = (
        baseline["candidate_ao_first_elo"] - baseline["adjusted_ao_first_elo"]
    ).abs()
    if float(error.max()) > 1e-8:
        raise ValueError(f"Research baseline does not reproduce production: {error.max()}")
    return result


def season_surface(config, seeds, predictions, target, seasons):
    rows = []
    for season in seasons:
        matches = predictions.loc[predictions["season"].eq(season)]
        rating = seeds.loc[seeds["season"].eq(season)].merge(
            target.loc[target["season"].eq(season)],
            on=["season", "team_id"],
            validate="one_to_one",
        )
        spearman = float(
            spearmanr(
                rating["candidate_ao_first_elo"], rating["schedule_adjusted_score"]
            ).statistic
        )
        pairwise = pairwise_accuracy(
            rating["candidate_ao_first_elo"].to_numpy(float),
            rating["schedule_adjusted_score"].to_numpy(float),
        )
        rows.append(
            {
                "candidate_key": config.key,
                "season": season,
                "matches": len(matches),
                "teams": len(rating),
                "brier_1x2": float(matches["brier_1x2"].mean()),
                "log_loss_1x2": float(matches["log_loss_1x2"].mean()),
                "accuracy_1x2": float(
                    (matches["actual_class"] == matches["predicted_class"]).mean()
                ),
                "seed_spearman": spearman,
                "seed_pairwise_accuracy": pairwise,
                "history_benchmark": config.history_benchmark,
                "prior_boost_scale": config.prior_boost_scale,
                "exposure_cap": config.exposure_cap,
                "european_tail_beta": config.european_tail_beta,
                "domestic_boost_scale": config.domestic_boost_scale,
                "uel_quality": config.uel_quality,
                "uecl_quality": config.uecl_quality,
            }
        )
    return rows


def aggregate_candidate_surface(surface: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, frame in surface.groupby("candidate_key", sort=True):
        rows.append(aggregate_metrics(key, frame))
    result = pd.DataFrame(rows)
    baseline = result.loc[result["candidate_key"].eq(BASELINE_CONFIG.key)].iloc[0]
    return add_baseline_deltas(result, baseline)


def aggregate_metrics(key: str, frame: pd.DataFrame) -> dict[str, object]:
    first = frame.iloc[0]
    return {
        "candidate_key": key,
        "history_benchmark": float(first["history_benchmark"]),
        "prior_boost_scale": float(first["prior_boost_scale"]),
        "exposure_cap": float(first["exposure_cap"]),
        "european_tail_beta": float(first["european_tail_beta"]),
        "domestic_boost_scale": float(first["domestic_boost_scale"]),
        "uel_quality": float(first["uel_quality"]),
        "uecl_quality": float(first["uecl_quality"]),
        "seasons": int(frame["season"].nunique()),
        "matches": int(frame["matches"].sum()),
        "teams": int(frame["teams"].sum()),
        "brier_1x2": float(np.average(frame["brier_1x2"], weights=frame["matches"])),
        "log_loss_1x2": float(
            np.average(frame["log_loss_1x2"], weights=frame["matches"])
        ),
        "accuracy_1x2": float(
            np.average(frame["accuracy_1x2"], weights=frame["matches"])
        ),
        "seed_spearman": float(
            np.average(frame["seed_spearman"], weights=frame["teams"])
        ),
        "seed_pairwise_accuracy": float(
            np.average(frame["seed_pairwise_accuracy"], weights=frame["teams"])
        ),
    }


def axis_ablation(
    summary: pd.DataFrame, full_selected: str, grid: str = "default"
) -> pd.DataFrame:
    """Best candidate when only one axis is allowed to move.

    The axes differ per grid: the original study varied benchmark, prior scale,
    exposure and competition quality, while the tail-and-domestic grid pins all
    of those and varies only the two top-end axes.
    """

    def best(label: str, mask: pd.Series) -> dict[str, object]:
        frame = summary.loc[mask]
        if frame.empty:
            raise ValueError(f"No candidates available for ablation {label}")
        row = frame.sort_values(["brier_1x2", "log_loss_1x2"]).iloc[0].to_dict()
        row["ablation"] = label
        return row

    baseline = summary["candidate_key"].eq(BASELINE_CONFIG.key)
    if grid == "tail-and-domestic":
        rows = [
            best("CURRENT_BASELINE", baseline),
            best(
                "TAIL_ONLY",
                summary["domestic_boost_scale"].eq(1.0)
                & summary["european_tail_beta"].gt(0.0),
            ),
            best(
                "DOMESTIC_SCALE_ONLY",
                summary["european_tail_beta"].eq(0.0)
                & summary["domestic_boost_scale"].gt(1.0),
            ),
            best("SELECTED_COMBINED", summary["candidate_key"].eq(full_selected)),
        ]
        columns = [
            "ablation",
            "candidate_key",
            "brier_1x2",
            "delta_vs_baseline_brier_1x2",
            "log_loss_1x2",
            "delta_vs_baseline_log_loss_1x2",
            "seed_spearman",
            "delta_vs_baseline_seed_spearman",
            "seed_pairwise_accuracy",
            "delta_vs_baseline_seed_pairwise_accuracy",
        ]
        return pd.DataFrame(rows)[columns]

    no_quality = summary["uel_quality"].eq(1.0) & summary["uecl_quality"].eq(1.0)
    rows = [
        best("CURRENT_BASELINE", baseline),
        best(
            "EXPOSURE_ONLY",
            summary["history_benchmark"].eq(20.0)
            & summary["prior_boost_scale"].eq(1.0)
            & no_quality,
        ),
        best(
            "BENCHMARK_ONLY",
            summary["prior_boost_scale"].eq(1.0)
            & summary["exposure_cap"].eq(0.85)
            & no_quality,
        ),
        best(
            "PRIOR_SCALE_ONLY",
            summary["history_benchmark"].eq(20.0)
            & summary["exposure_cap"].eq(0.85)
            & no_quality,
        ),
        best(
            "COMPETITION_QUALITY_ONLY",
            summary["history_benchmark"].eq(20.0)
            & summary["prior_boost_scale"].eq(1.0)
            & summary["exposure_cap"].eq(0.85),
        ),
        best("SELECTED_COMBINED", summary["candidate_key"].eq(full_selected)),
    ]
    columns = [
        "ablation",
        "candidate_key",
        "brier_1x2",
        "delta_vs_baseline_brier_1x2",
        "log_loss_1x2",
        "delta_vs_baseline_log_loss_1x2",
        "seed_spearman",
        "delta_vs_baseline_seed_spearman",
        "seed_pairwise_accuracy",
        "delta_vs_baseline_seed_pairwise_accuracy",
    ]
    return pd.DataFrame(rows)[columns]


def add_baseline_deltas(frame: pd.DataFrame, baseline: pd.Series) -> pd.DataFrame:
    result = frame.copy()
    for metric in (
        "brier_1x2",
        "log_loss_1x2",
        "accuracy_1x2",
        "seed_spearman",
        "seed_pairwise_accuracy",
    ):
        result[f"delta_vs_baseline_{metric}"] = result[metric] - float(baseline[metric])
    return result


def select_candidate(summary: pd.DataFrame) -> str:
    baseline = summary.loc[summary["candidate_key"].eq(BASELINE_CONFIG.key)].iloc[0]
    eligible = summary.loc[
        summary["seed_spearman"].ge(float(baseline["seed_spearman"]) - RANKING_TOLERANCE)
        & summary["seed_pairwise_accuracy"].ge(
            float(baseline["seed_pairwise_accuracy"]) - RANKING_TOLERANCE
        )
    ]
    if eligible.empty:
        return BASELINE_CONFIG.key
    return str(
        eligible.sort_values(
            ["brier_1x2", "log_loss_1x2", "candidate_key"], kind="stable"
        ).iloc[0]["candidate_key"]
    )


def nested_selection(surface, configs, folds) -> pd.DataFrame:
    rows = []
    keys = {config.key for config in configs}
    for fold, (train_seasons, test_season) in enumerate(folds, start=1):
        train = surface.loc[surface["season"].isin(train_seasons)]
        summary = pd.DataFrame(
            [
                aggregate_metrics(key, train.loc[train["candidate_key"].eq(key)])
                for key in sorted(keys)
            ]
        )
        selected = select_candidate(summary)
        row = summary.loc[summary["candidate_key"].eq(selected)].iloc[0]
        rows.append(
            {
                "fold": fold,
                "train_seasons": "|".join(train_seasons),
                "test_season": test_season,
                "selected_candidate_key": selected,
                "train_brier_1x2": row.brier_1x2,
                "train_log_loss_1x2": row.log_loss_1x2,
                "train_seed_spearman": row.seed_spearman,
                "train_seed_pairwise_accuracy": row.seed_pairwise_accuracy,
            }
        )
    return pd.DataFrame(rows)


def build_unseen(selections, predictions, candidate_seeds, target, folds):
    unseen_rows = []
    fold_rows = []
    baseline_predictions = predictions[BASELINE_CONFIG.key]
    baseline_seeds = candidate_seeds[BASELINE_CONFIG.key]
    for fold, (_, test_season) in enumerate(folds, start=1):
        selected_key = str(
            selections.loc[selections["fold"].eq(fold), "selected_candidate_key"].iloc[0]
        )
        selected = predictions[selected_key].loc[
            predictions[selected_key]["season"].eq(test_season)
        ].copy()
        baseline = baseline_predictions.loc[
            baseline_predictions["season"].eq(test_season)
        ][["match_id", "brier_1x2", "log_loss_1x2"]].rename(
            columns={
                "brier_1x2": "baseline_brier_1x2",
                "log_loss_1x2": "baseline_log_loss_1x2",
            }
        )
        selected = selected.merge(baseline, on="match_id", validate="one_to_one")
        selected["fold"] = fold
        selected["selected_candidate_key"] = selected_key
        selected["brier_difference"] = (
            selected["brier_1x2"] - selected["baseline_brier_1x2"]
        )
        selected["log_loss_difference"] = (
            selected["log_loss_1x2"] - selected["baseline_log_loss_1x2"]
        )
        unseen_rows.append(selected)

        selected_rank = rank_metrics(
            candidate_seeds[selected_key], target, test_season
        )
        baseline_rank = rank_metrics(baseline_seeds, target, test_season)
        fold_rows.append(
            {
                "fold": fold,
                "test_season": test_season,
                "selected_candidate_key": selected_key,
                "matches": len(selected),
                "brier_1x2": selected["brier_1x2"].mean(),
                "baseline_brier_1x2": selected["baseline_brier_1x2"].mean(),
                "delta_brier_1x2": selected["brier_difference"].mean(),
                "log_loss_1x2": selected["log_loss_1x2"].mean(),
                "baseline_log_loss_1x2": selected["baseline_log_loss_1x2"].mean(),
                "delta_log_loss_1x2": selected["log_loss_difference"].mean(),
                "accuracy_1x2": (
                    selected["actual_class"] == selected["predicted_class"]
                ).mean(),
                "seed_spearman": selected_rank[0],
                "baseline_seed_spearman": baseline_rank[0],
                "delta_seed_spearman": selected_rank[0] - baseline_rank[0],
                "seed_pairwise_accuracy": selected_rank[1],
                "baseline_seed_pairwise_accuracy": baseline_rank[1],
                "delta_seed_pairwise_accuracy": selected_rank[1] - baseline_rank[1],
            }
        )
    return pd.concat(unseen_rows, ignore_index=True), pd.DataFrame(fold_rows)


def rank_metrics(seeds, target, season):
    frame = seeds.loc[seeds["season"].eq(season)].merge(
        target.loc[target["season"].eq(season)],
        on=["season", "team_id"],
        validate="one_to_one",
    )
    rating = frame["candidate_ao_first_elo"].to_numpy(float)
    outcome = frame["schedule_adjusted_score"].to_numpy(float)
    return float(spearmanr(rating, outcome).statistic), pairwise_accuracy(rating, outcome)


def model_summary(unseen, fold_results):
    matches = len(unseen)
    result = pd.DataFrame(
        [
            {
                "model": "CURRENT_PRODUCTION",
                "matches": matches,
                "brier_1x2": unseen["baseline_brier_1x2"].mean(),
                "log_loss_1x2": unseen["baseline_log_loss_1x2"].mean(),
                "seed_spearman": np.average(
                    fold_results["baseline_seed_spearman"], weights=fold_results["matches"]
                ),
                "seed_pairwise_accuracy": np.average(
                    fold_results["baseline_seed_pairwise_accuracy"],
                    weights=fold_results["matches"],
                ),
            },
            {
                "model": "NESTED_RECALIBRATION",
                "matches": matches,
                "brier_1x2": unseen["brier_1x2"].mean(),
                "log_loss_1x2": unseen["log_loss_1x2"].mean(),
                "seed_spearman": np.average(
                    fold_results["seed_spearman"], weights=fold_results["matches"]
                ),
                "seed_pairwise_accuracy": np.average(
                    fold_results["seed_pairwise_accuracy"], weights=fold_results["matches"]
                ),
            },
        ]
    )
    baseline = result.iloc[0]
    for metric in (
        "brier_1x2",
        "log_loss_1x2",
        "seed_spearman",
        "seed_pairwise_accuracy",
    ):
        result[f"delta_vs_baseline_{metric}"] = result[metric] - float(baseline[metric])
    return result


def competition_summary(unseen):
    rows = []
    for competition, frame in unseen.groupby("competition", sort=True):
        rows.append(
            {
                "competition": competition,
                "matches": len(frame),
                "brier_1x2": frame["brier_1x2"].mean(),
                "baseline_brier_1x2": frame["baseline_brier_1x2"].mean(),
                "delta_brier_1x2": frame["brier_difference"].mean(),
                "log_loss_1x2": frame["log_loss_1x2"].mean(),
                "baseline_log_loss_1x2": frame["baseline_log_loss_1x2"].mean(),
                "delta_log_loss_1x2": frame["log_loss_difference"].mean(),
            }
        )
    return pd.DataFrame(rows)


def uncertainty_summary(unseen, samples):
    rows = []
    for metric in ("brier", "log_loss"):
        column = f"{metric}_difference"
        ci = dependency_robust_loss_difference_ci(
            unseen,
            difference_column=column,
            bootstrap_samples=samples,
            seed=20260820 + len(rows),
        )
        ci.insert(0, "metric", metric)
        rows.append(ci)
    return pd.concat(rows, ignore_index=True)


def historical_impact(frame):
    result = frame.copy()
    result["exposure_band"] = pd.cut(
        result["european_exposure"],
        [-1e-12, 0.0, 0.5, 0.75, 1.0],
        labels=["ZERO", "LOW", "MEDIUM", "HIGH"],
        include_lowest=True,
    )
    return (
        result.groupby(["competition", "exposure_band"], observed=False)
        .agg(
            team_seasons=("team_id", "size"),
            mean_delta=("candidate_elo_delta", "mean"),
            median_delta=("candidate_elo_delta", "median"),
            p90_abs_delta=("candidate_elo_delta", lambda x: x.abs().quantile(0.90)),
            maximum_abs_delta=("candidate_elo_delta", lambda x: x.abs().max()),
        )
        .reset_index()
    )


def current_snapshot_impact(config):
    current = pd.read_csv(CURRENT_RATINGS)
    current = current.rename(columns={"ao_first_elo": "adjusted_ao_first_elo"})
    result = apply_european_prior_recalibration(current, config)
    result["current_rank"] = result["adjusted_ao_first_elo"].rank(
        ascending=False, method="min"
    ).astype(int)
    result["candidate_rank"] = result["candidate_ao_first_elo"].rank(
        ascending=False, method="min"
    ).astype(int)
    result["rank_change"] = result["current_rank"] - result["candidate_rank"]
    return result[
        [
            "season",
            "team_id",
            "team_name",
            "country_code",
            "competition",
            "weighted_european_history",
            "european_exposure",
            "adjusted_domestic_prior",
            "adjusted_ao_first_elo",
            "candidate_european_prior",
            "candidate_effective_exposure",
            "candidate_ao_first_elo",
            "candidate_elo_delta",
            "current_rank",
            "candidate_rank",
            "rank_change",
        ]
    ].sort_values("candidate_rank")


def decide(
    selections,
    folds,
    summary,
    competition,
    uncertainty,
    ranking_uncertainty,
    full_selected,
    current,
    contract_hash,
):
    selected = summary.loc[summary["model"].eq("NESTED_RECALIBRATION")].iloc[0]
    brier_wins = int((folds["delta_brier_1x2"] < 0.0).sum())
    log_wins = int((folds["delta_log_loss_1x2"] < 0.0).sum())
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
        else "KEEP_RESEARCH"
    )
    ferenc = current.loc[current["team_name"].str.contains("Ferenc", case=False)].iloc[0]
    return {
        "decision": decision,
        "production_changed": False,
        "candidate_count": 81,
        "fold_count": len(folds),
        "matches": int(summary["matches"].max()),
        "full_history_candidate_key": full_selected,
        "nested_selected_keys": selections["selected_candidate_key"].tolist(),
        "brier_fold_wins": brier_wins,
        "log_loss_fold_wins": log_wins,
        "ranking_reliable_harm_proxy": rank_harm,
        "competition_loss_harm": segment_harm,
        "pooled_brier_delta": float(selected["delta_vs_baseline_brier_1x2"]),
        "pooled_log_loss_delta": float(selected["delta_vs_baseline_log_loss_1x2"]),
        "pooled_seed_spearman_delta": float(selected["delta_vs_baseline_seed_spearman"]),
        "pooled_seed_pairwise_delta": float(
            selected["delta_vs_baseline_seed_pairwise_accuracy"]
        ),
        "ferencvaros_current_elo": float(ferenc["adjusted_ao_first_elo"]),
        "ferencvaros_candidate_elo": float(ferenc["candidate_ao_first_elo"]),
        "ferencvaros_candidate_rank": int(ferenc["candidate_rank"]),
        "production_contract_sha256": contract_hash,
        "quality_proxy_note": (
            "Competition quality uses the known season-entry competition, not a "
            "complete historical competition-by-points decomposition."
        ),
    }


def build_report(
    decision,
    summary,
    folds,
    competition,
    uncertainty,
    ranking_uncertainty,
    ablation,
    current,
):
    ferenc = current.loc[current["team_name"].str.contains("Ferenc", case=False)].iloc[0]
    return f"""# European Prior Recalibration Backtest

## Kapsam

- 2018/19-2025/26 static seed evreni: 1.887 takim-sezon
- Unseen outer fold: 6
- Unseen mac: {decision['matches']:,}
- Aday: 81
- Production degisti: hayir

Test edilen uc eksen: European History benchmark/prior scale, giris turnuvasi
kalite proxy'si ve effective exposure cap. Turnuva kalite alani gecmis puanlarin
tek tek hangi kupada kazanildigini degil, sezon basinda bilinen entry
competition'i kullanir; bu nedenle sonuc production kaniti degil challenger
kanitidir.

## Karar

`{decision['decision']}`

- Full-history aday: `{decision['full_history_candidate_key']}`
- Brier fold kazanimi: {decision['brier_fold_wins']}/6
- Log-loss fold kazanimi: {decision['log_loss_fold_wins']}/6
- Pooled Brier delta: {decision['pooled_brier_delta']:+.6f}
- Pooled log-loss delta: {decision['pooled_log_loss_delta']:+.6f}
- Seed Spearman delta: {decision['pooled_seed_spearman_delta']:+.6f}
- Seed pairwise delta: {decision['pooled_seed_pairwise_delta']:+.6f}

## Ferencvaros Diagnostigi

- Current AO First Elo: {ferenc['adjusted_ao_first_elo']:.3f}
- Candidate AO First Elo: {ferenc['candidate_ao_first_elo']:.3f}
- Candidate sira: {int(ferenc['candidate_rank'])}
- Elo farki: {ferenc['candidate_elo_delta']:+.3f}

## Pooled model

{markdown_table(summary)}

## Fold sonuclari

{markdown_table(folds)}

## Turnuva segmentleri

{markdown_table(competition)}

## Eksen ablation

{markdown_table(ablation)}

## Belirsizlik

{markdown_table(uncertainty)}

## Ranking belirsizligi

{markdown_table(ranking_uncertainty)}
"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
