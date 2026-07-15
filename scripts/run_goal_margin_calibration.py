from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_dynamic_core_calibration import (  # noqa: E402
    MAX_RATING_MOVE_GUARDRAIL,
    RANK_CORRELATION_FLOOR,
    DynamicCoreConfig,
    SeasonData,
    expanding_folds,
    expected_home_score,
    load_calibration_data,
)


STATIC_DATA_ROOT = ROOT / "data" / "backtest_stage_b_2018_2026"
EVENTS_PATH = ROOT / "data" / "dynamic_backtest_2018_2026" / "matches.csv"
CORE_OUTPUT_ROOT = ROOT / "output" / "dynamic_core_calibration_2018_2026"
OUTPUT_ROOT = ROOT / "output" / "goal_margin_calibration_2018_2026"
GOAL_WEIGHTS = (0.0, 0.25, 0.50, 0.75, 1.0)
GOAL_CAPS = (1.25, 1.50, 1.75, 2.0)


@dataclass(frozen=True, order=True)
class GoalMarginConfig:
    goal_weight: float
    goal_cap: float

    def validate(self) -> None:
        if not math.isfinite(self.goal_weight) or self.goal_weight < 0:
            raise ValueError("goal_weight must be non-negative and finite")
        if not math.isfinite(self.goal_cap) or self.goal_cap < 1:
            raise ValueError("goal_cap must be at least one and finite")


@dataclass(frozen=True)
class MarginSeasonData:
    core: SeasonData
    goal_differences: np.ndarray


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate the controlled AO dynamic Elo goal-margin layer"
    )
    parser.add_argument("--static-data-root", type=Path, default=STATIC_DATA_ROOT)
    parser.add_argument("--events-path", type=Path, default=EVENTS_PATH)
    parser.add_argument("--core-output-root", type=Path, default=CORE_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    static_root = args.static_data_root.resolve()
    events_path = args.events_path.resolve()
    core_output_root = args.core_output_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    datasets = load_margin_data(static_root, events_path)
    seasons = tuple(data.core.season for data in datasets)
    folds = expanding_folds(seasons)
    core_selections = pd.read_csv(core_output_root / "fold_selections.csv")
    full_core = read_full_core_config(core_output_root / "full_candidate_metrics.csv")
    validate_core_fold_contract(core_selections, folds)

    candidates = candidate_grid()
    selection_rows: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    baseline_margin = GoalMarginConfig(0.0, 1.0)

    for fold_number, (train_seasons, test_season) in enumerate(folds, start=1):
        core_row = core_selections.loc[core_selections["fold"].eq(fold_number)].iloc[0]
        core_config = DynamicCoreConfig(
            float(core_row["selected_scale"]),
            float(core_row["selected_home_advantage"]),
            float(core_row["selected_k"]),
        )
        train_data = tuple(data for data in datasets if data.core.season in train_seasons)
        test_data = next(data for data in datasets if data.core.season == test_season)
        selected_margin, train_metrics = select_margin_candidate(
            train_data,
            core_config,
            candidates,
        )
        baseline_train, _ = evaluate_margin_seasons(
            train_data,
            core_config,
            baseline_margin,
        )
        selection_rows.append(
            {
                "fold": fold_number,
                "train_seasons": ",".join(train_seasons),
                "test_season": test_season,
                "core_scale": core_config.elo_scale,
                "core_home_advantage": core_config.home_advantage,
                "core_k": core_config.k_factor,
                "selected_goal_weight": selected_margin.goal_weight,
                "selected_goal_cap": selected_margin.goal_cap,
                "train_brier": train_metrics["brier"],
                "baseline_train_brier": baseline_train["brier"],
                "train_brier_difference": train_metrics["brier"] - baseline_train["brier"],
            }
        )
        predictions_by_model: dict[str, pd.DataFrame] = {}
        for model_name, margin_config in (
            ("goal_margin", selected_margin),
            ("core_baseline", baseline_margin),
        ):
            metrics, predictions = evaluate_margin_seasons(
                (test_data,),
                core_config,
                margin_config,
                return_predictions=True,
            )
            result_rows.append(
                {
                    "fold": fold_number,
                    "test_season": test_season,
                    "model": model_name,
                    "core_scale": core_config.elo_scale,
                    "core_home_advantage": core_config.home_advantage,
                    "core_k": core_config.k_factor,
                    "goal_weight": margin_config.goal_weight,
                    "goal_cap": margin_config.goal_cap,
                    **metrics,
                }
            )
            assert predictions is not None
            predictions_by_model[model_name] = predictions.rename(
                columns={
                    "expected_home_score": f"{model_name}_expected_home_score",
                    "brier_loss": f"{model_name}_brier_loss",
                    "log_loss": f"{model_name}_log_loss",
                }
            )
        joined = predictions_by_model["goal_margin"].merge(
            predictions_by_model["core_baseline"][
                [
                    "match_id",
                    "core_baseline_expected_home_score",
                    "core_baseline_brier_loss",
                    "core_baseline_log_loss",
                ]
            ],
            on="match_id",
            validate="one_to_one",
        )
        joined.insert(0, "fold", fold_number)
        prediction_frames.append(joined)

    selections = pd.DataFrame(selection_rows)
    fold_results = pd.DataFrame(result_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    competition_summary = summarize_competitions(predictions)
    uncertainty = paired_uncertainty(predictions)
    stability = parameter_stability(selections)

    final_margin, final_metrics = select_margin_candidate(datasets, full_core, candidates)
    full_candidate_metrics = candidate_metrics(datasets, full_core, candidates)
    final_season_metrics = evaluate_by_season(datasets, full_core, final_margin)
    multiplier_table = build_multiplier_table(final_margin)
    outer_sensitivity = build_outer_candidate_sensitivity(
        datasets,
        folds,
        core_selections,
        candidates,
    )
    goal_distribution = build_goal_difference_distribution(datasets)
    decision = calibration_decision(
        selections,
        fold_results,
        competition_summary,
        uncertainty,
        stability,
        final_margin,
    )

    selections.to_csv(output_root / "fold_selections.csv", index=False)
    fold_results.to_csv(output_root / "fold_results.csv", index=False)
    predictions.to_csv(output_root / "unseen_match_predictions.csv", index=False)
    competition_summary.to_csv(output_root / "competition_summary.csv", index=False)
    uncertainty.to_csv(output_root / "paired_uncertainty.csv", index=False)
    stability.to_csv(output_root / "parameter_stability.csv", index=False)
    full_candidate_metrics.to_csv(output_root / "full_candidate_metrics.csv", index=False)
    final_season_metrics.to_csv(output_root / "final_candidate_season_metrics.csv", index=False)
    multiplier_table.to_csv(output_root / "goal_multiplier_table.csv", index=False)
    outer_sensitivity.to_csv(output_root / "outer_candidate_sensitivity.csv", index=False)
    goal_distribution.to_csv(output_root / "goal_difference_distribution.csv", index=False)
    write_report(
        output_root / "calibration_report.md",
        seasons,
        selections,
        fold_results,
        competition_summary,
        uncertainty,
        stability,
        full_core,
        final_margin,
        final_metrics,
        multiplier_table,
        outer_sensitivity,
        goal_distribution,
        decision,
    )

    print("AO dynamic Elo goal-margin calibration")
    print(f"Seasons: {len(seasons)}")
    print(f"Matches: {sum(len(data.core.match_ids) for data in datasets)}")
    print(f"Outer folds: {len(folds)}")
    print(
        "Full-data candidate: "
        f"weight={final_margin.goal_weight:g}, cap={final_margin.goal_cap:g}"
    )
    print(f"Decision: {decision}")
    print(f"Report: {output_root / 'calibration_report.md'}")


def load_margin_data(
    static_root: Path,
    events_path: Path,
) -> tuple[MarginSeasonData, ...]:
    core_data = load_calibration_data(static_root, events_path)
    events = pd.read_csv(events_path).sort_values(["season", "event_order"])
    if "goal_difference" not in events.columns:
        raise ValueError("Dynamic event data must contain goal_difference")
    if (
        events["goal_difference"].isna().any()
        or events["goal_difference"].lt(0).any()
        or (events["goal_difference"] % 1).ne(0).any()
    ):
        raise ValueError("goal_difference must contain non-negative integers")

    result: list[MarginSeasonData] = []
    for core in core_data:
        season_events = events.loc[events["season"].eq(core.season)]
        if not np.array_equal(season_events["match_id"].to_numpy(str), core.match_ids):
            raise ValueError(f"{core.season}: goal-margin rows do not align with core events")
        result.append(
            MarginSeasonData(
                core=core,
                goal_differences=season_events["goal_difference"].to_numpy(int),
            )
        )
    return tuple(result)


def read_full_core_config(path: Path) -> DynamicCoreConfig:
    row = pd.read_csv(path).iloc[0]
    return DynamicCoreConfig(
        float(row["elo_scale"]),
        float(row["home_advantage"]),
        float(row["k_factor"]),
    )


def validate_core_fold_contract(
    selections: pd.DataFrame,
    folds: tuple[tuple[tuple[str, ...], str], ...],
) -> None:
    if len(selections) != len(folds) or not selections["fold"].is_unique:
        raise ValueError("Core fold selections do not match the margin calibration folds")
    for fold_number, (_, test_season) in enumerate(folds, start=1):
        row = selections.loc[selections["fold"].eq(fold_number)]
        if len(row) != 1 or row.iloc[0]["test_season"] != test_season:
            raise ValueError(f"Core fold {fold_number} has the wrong unseen season")


def candidate_grid() -> tuple[GoalMarginConfig, ...]:
    candidates = {GoalMarginConfig(0.0, 1.0)}
    candidates.update(
        GoalMarginConfig(weight, cap)
        for weight in GOAL_WEIGHTS
        if weight > 0
        for cap in GOAL_CAPS
    )
    return tuple(sorted(candidates))


def goal_margin_multiplier(
    goal_difference: int | float,
    config: GoalMarginConfig,
) -> float:
    config.validate()
    if not math.isfinite(float(goal_difference)) or goal_difference < 0:
        raise ValueError("goal_difference must be non-negative and finite")
    if goal_difference <= 1 or config.goal_weight == 0:
        return 1.0
    return min(
        config.goal_cap,
        1.0 + config.goal_weight * math.log(float(goal_difference)),
    )


def run_margin_season(
    data: MarginSeasonData,
    core_config: DynamicCoreConfig,
    margin_config: GoalMarginConfig,
    *,
    return_predictions: bool = False,
) -> tuple[dict[str, float | int], pd.DataFrame | None]:
    core_config.validate()
    margin_config.validate()
    core = data.core
    ratings = core.initial_ratings.copy()
    brier_sum = 0.0
    log_loss_sum = 0.0
    max_abs_match_delta = 0.0
    prediction_rows: list[dict[str, object]] = []
    for index, (home_id, away_id, actual, neutral, goal_difference) in enumerate(
        zip(
            core.home_team_ids,
            core.away_team_ids,
            core.actual_home_scores,
            core.neutral_flags,
            data.goal_differences,
        )
    ):
        probability = float(
            np.clip(
                expected_home_score(
                    ratings[home_id],
                    ratings[away_id],
                    core_config,
                    neutral=bool(neutral),
                ),
                1e-12,
                1.0 - 1e-12,
            )
        )
        brier_loss = (probability - actual) ** 2
        log_loss = -(actual * math.log(probability) + (1.0 - actual) * math.log(1.0 - probability))
        multiplier = goal_margin_multiplier(int(goal_difference), margin_config)
        delta = core_config.k_factor * multiplier * (actual - probability)
        brier_sum += brier_loss
        log_loss_sum += log_loss
        max_abs_match_delta = max(max_abs_match_delta, abs(delta))
        ratings[home_id] += delta
        ratings[away_id] -= delta
        if return_predictions:
            prediction_rows.append(
                {
                    "match_id": core.match_ids[index],
                    "season": core.season,
                    "competition": core.competitions[index],
                    "actual_home_score": actual,
                    "goal_difference": int(goal_difference),
                    "goal_multiplier": multiplier,
                    "expected_home_score": probability,
                    "brier_loss": brier_loss,
                    "log_loss": log_loss,
                }
            )

    active = core.active_team_ids
    start_ratings = core.initial_ratings[active]
    end_ratings = ratings[active]
    changes = end_ratings - start_ratings
    if np.allclose(start_ratings, end_ratings):
        rank_correlation = 1.0
    elif np.ptp(start_ratings) == 0 or np.ptp(end_ratings) == 0:
        rank_correlation = 0.0
    else:
        rank_correlation = pd.Series(end_ratings).corr(
            pd.Series(start_ratings),
            method="spearman",
        )
    metrics: dict[str, float | int] = {
        "matches": len(core.match_ids),
        "brier": brier_sum / len(core.match_ids),
        "log_loss": log_loss_sum / len(core.match_ids),
        "mean_rating_change": float(np.mean(changes)),
        "rating_change_std": float(np.std(changes)),
        "max_abs_rating_change": float(np.max(np.abs(changes))),
        "max_abs_match_delta": max_abs_match_delta,
        "start_end_rank_correlation": float(rank_correlation),
    }
    predictions = pd.DataFrame(prediction_rows) if return_predictions else None
    return metrics, predictions


def evaluate_margin_seasons(
    datasets: tuple[MarginSeasonData, ...],
    core_config: DynamicCoreConfig,
    margin_config: GoalMarginConfig,
    *,
    return_predictions: bool = False,
) -> tuple[dict[str, float | int], pd.DataFrame | None]:
    metric_rows: list[dict[str, float | int]] = []
    frames: list[pd.DataFrame] = []
    for data in datasets:
        metrics, predictions = run_margin_season(
            data,
            core_config,
            margin_config,
            return_predictions=return_predictions,
        )
        metric_rows.append(metrics)
        if predictions is not None:
            frames.append(predictions)
    matches = sum(int(row["matches"]) for row in metric_rows)
    aggregate: dict[str, float | int] = {
        "matches": matches,
        "brier": sum(float(row["brier"]) * int(row["matches"]) for row in metric_rows) / matches,
        "log_loss": sum(float(row["log_loss"]) * int(row["matches"]) for row in metric_rows) / matches,
        "mean_rating_change": float(np.mean([row["mean_rating_change"] for row in metric_rows])),
        "rating_change_std": float(np.mean([row["rating_change_std"] for row in metric_rows])),
        "max_abs_rating_change": float(max(row["max_abs_rating_change"] for row in metric_rows)),
        "max_abs_match_delta": float(max(row["max_abs_match_delta"] for row in metric_rows)),
        "start_end_rank_correlation": float(min(row["start_end_rank_correlation"] for row in metric_rows)),
    }
    predictions = pd.concat(frames, ignore_index=True) if frames else None
    return aggregate, predictions


def candidate_metrics(
    datasets: tuple[MarginSeasonData, ...],
    core_config: DynamicCoreConfig,
    candidates: tuple[GoalMarginConfig, ...],
) -> pd.DataFrame:
    rows = []
    for config in candidates:
        metrics, _ = evaluate_margin_seasons(datasets, core_config, config)
        rows.append(
            {
                "goal_weight": config.goal_weight,
                "goal_cap": config.goal_cap,
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["brier", "log_loss", "goal_weight", "goal_cap"]
    ).reset_index(drop=True)


def select_margin_candidate(
    datasets: tuple[MarginSeasonData, ...],
    core_config: DynamicCoreConfig,
    candidates: tuple[GoalMarginConfig, ...],
) -> tuple[GoalMarginConfig, dict[str, float | int]]:
    rows = candidate_metrics(datasets, core_config, candidates)
    selected = rows.iloc[0]
    config = GoalMarginConfig(
        float(selected["goal_weight"]),
        float(selected["goal_cap"]),
    )
    return config, {
        column: selected[column]
        for column in rows.columns
        if column not in {"goal_weight", "goal_cap"}
    }


def evaluate_by_season(
    datasets: tuple[MarginSeasonData, ...],
    core_config: DynamicCoreConfig,
    margin_config: GoalMarginConfig,
) -> pd.DataFrame:
    rows = []
    for data in datasets:
        metrics, _ = run_margin_season(data, core_config, margin_config)
        rows.append({"season": data.core.season, **metrics})
    return pd.DataFrame(rows)


def summarize_competitions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for competition, data in predictions.groupby("competition"):
        margin_brier = data["goal_margin_brier_loss"].mean()
        core_brier = data["core_baseline_brier_loss"].mean()
        margin_log = data["goal_margin_log_loss"].mean()
        core_log = data["core_baseline_log_loss"].mean()
        rows.append(
            {
                "competition": competition,
                "matches": len(data),
                "goal_margin_brier": margin_brier,
                "core_brier": core_brier,
                "brier_difference": margin_brier - core_brier,
                "goal_margin_log_loss": margin_log,
                "core_log_loss": core_log,
                "log_loss_difference": margin_log - core_log,
            }
        )
    return pd.DataFrame(rows)


def paired_uncertainty(
    predictions: pd.DataFrame,
    *,
    bootstrap_samples: int = 4000,
    seed: int = 20260715,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    groups = [("ALL", predictions), *predictions.groupby("competition")]
    for competition, data in groups:
        differences = (
            data["goal_margin_brier_loss"] - data["core_baseline_brier_loss"]
        ).to_numpy(float)
        sampled = rng.choice(differences, size=(bootstrap_samples, len(differences)), replace=True)
        means = sampled.mean(axis=1)
        lower, upper = np.quantile(means, (0.025, 0.975))
        rows.append(
            {
                "competition": competition,
                "matches": len(differences),
                "mean_brier_difference": differences.mean(),
                "ci_95_lower": lower,
                "ci_95_upper": upper,
                "reliable_improvement": bool(upper < 0),
                "reliable_harm": bool(lower > 0),
            }
        )
    return pd.DataFrame(rows)


def parameter_stability(selections: pd.DataFrame) -> pd.DataFrame:
    pair_counts = selections.value_counts(
        ["selected_goal_weight", "selected_goal_cap"]
    ).reset_index(name="count")
    pair_counts["folds"] = len(selections)
    pair_counts["share"] = pair_counts["count"] / len(selections)
    return pair_counts


def build_multiplier_table(config: GoalMarginConfig) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "goal_difference": range(0, 9),
            "goal_multiplier": [
                goal_margin_multiplier(goal_difference, config)
                for goal_difference in range(0, 9)
            ],
        }
    )


def build_outer_candidate_sensitivity(
    datasets: tuple[MarginSeasonData, ...],
    folds: tuple[tuple[tuple[str, ...], str], ...],
    core_selections: pd.DataFrame,
    candidates: tuple[GoalMarginConfig, ...],
) -> pd.DataFrame:
    baseline = GoalMarginConfig(0.0, 1.0)
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        candidate_brier_sum = 0.0
        baseline_brier_sum = 0.0
        candidate_log_sum = 0.0
        baseline_log_sum = 0.0
        matches = 0
        fold_wins = 0
        max_rating_move = 0.0
        max_match_delta = 0.0
        min_rank_correlation = 1.0
        competition_differences: dict[str, list[float]] = {}
        for fold_number, (_, test_season) in enumerate(folds, start=1):
            core_row = core_selections.loc[core_selections["fold"].eq(fold_number)].iloc[0]
            core_config = DynamicCoreConfig(
                float(core_row["selected_scale"]),
                float(core_row["selected_home_advantage"]),
                float(core_row["selected_k"]),
            )
            test_data = next(data for data in datasets if data.core.season == test_season)
            candidate_metrics_row, candidate_predictions = evaluate_margin_seasons(
                (test_data,), core_config, candidate, return_predictions=True
            )
            baseline_metrics_row, baseline_predictions = evaluate_margin_seasons(
                (test_data,), core_config, baseline, return_predictions=True
            )
            fold_matches = int(candidate_metrics_row["matches"])
            matches += fold_matches
            candidate_brier_sum += float(candidate_metrics_row["brier"]) * fold_matches
            baseline_brier_sum += float(baseline_metrics_row["brier"]) * fold_matches
            candidate_log_sum += float(candidate_metrics_row["log_loss"]) * fold_matches
            baseline_log_sum += float(baseline_metrics_row["log_loss"]) * fold_matches
            fold_wins += int(candidate_metrics_row["brier"] < baseline_metrics_row["brier"])
            max_rating_move = max(
                max_rating_move,
                float(candidate_metrics_row["max_abs_rating_change"]),
            )
            max_match_delta = max(
                max_match_delta,
                float(candidate_metrics_row["max_abs_match_delta"]),
            )
            min_rank_correlation = min(
                min_rank_correlation,
                float(candidate_metrics_row["start_end_rank_correlation"]),
            )
            assert candidate_predictions is not None and baseline_predictions is not None
            comparison = candidate_predictions[["match_id", "competition", "brier_loss"]].merge(
                baseline_predictions[["match_id", "brier_loss"]],
                on="match_id",
                suffixes=("_candidate", "_baseline"),
                validate="one_to_one",
            )
            comparison["difference"] = (
                comparison["brier_loss_candidate"] - comparison["brier_loss_baseline"]
            )
            for competition, values in comparison.groupby("competition")["difference"]:
                competition_differences.setdefault(str(competition), []).extend(values.tolist())
        candidate_brier = candidate_brier_sum / matches
        baseline_brier = baseline_brier_sum / matches
        row: dict[str, object] = {
            "goal_weight": candidate.goal_weight,
            "goal_cap": candidate.goal_cap,
            "matches": matches,
            "fold_wins": fold_wins,
            "candidate_brier": candidate_brier,
            "baseline_brier": baseline_brier,
            "brier_difference": candidate_brier - baseline_brier,
            "candidate_log_loss": candidate_log_sum / matches,
            "baseline_log_loss": baseline_log_sum / matches,
            "log_loss_difference": (candidate_log_sum - baseline_log_sum) / matches,
            "max_abs_rating_change": max_rating_move,
            "max_abs_match_delta": max_match_delta,
            "min_start_end_rank_correlation": min_rank_correlation,
        }
        for competition in ("UCL", "UEL", "UECL"):
            values = competition_differences.get(competition, [])
            row[f"{competition.lower()}_brier_difference"] = (
                float(np.mean(values)) if values else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["brier_difference", "log_loss_difference", "goal_weight", "goal_cap"]
    ).reset_index(drop=True)


def build_goal_difference_distribution(
    datasets: tuple[MarginSeasonData, ...],
) -> pd.DataFrame:
    values = np.concatenate([data.goal_differences for data in datasets])
    clipped = np.minimum(values, 8)
    counts = pd.Series(clipped).value_counts().sort_index()
    rows = []
    for goal_difference, count in counts.items():
        rows.append(
            {
                "goal_difference": "8+" if goal_difference == 8 else str(int(goal_difference)),
                "matches": int(count),
                "share": float(count / len(values)),
            }
        )
    return pd.DataFrame(rows)


def calibration_decision(
    selections: pd.DataFrame,
    fold_results: pd.DataFrame,
    competition_summary: pd.DataFrame,
    uncertainty: pd.DataFrame,
    stability: pd.DataFrame,
    final_margin: GoalMarginConfig,
) -> str:
    pivot = fold_results.pivot(index="fold", columns="model", values="brier")
    fold_wins = int((pivot["goal_margin"] < pivot["core_baseline"]).sum())
    overall = uncertainty.loc[uncertainty["competition"].eq("ALL")].iloc[0]
    no_competition_harm = not bool((competition_summary["brier_difference"] > 0).any())
    exact_pair_stability = float(stability.iloc[0]["share"])
    margin_rows = fold_results.loc[fold_results["model"].eq("goal_margin")]
    ranking_safe = bool(
        margin_rows["start_end_rank_correlation"].ge(RANK_CORRELATION_FLOOR).all()
        and margin_rows["max_abs_rating_change"].le(MAX_RATING_MOVE_GUARDRAIL).all()
    )
    nonzero_fold_share = float(selections["selected_goal_weight"].gt(0).mean())
    if final_margin.goal_weight == 0 or nonzero_fold_share < 0.5:
        return "REJECT_GOAL_MARGIN_KEEP_CORE"
    if (
        fold_wins >= 5
        and bool(overall["reliable_improvement"])
        and no_competition_harm
        and exact_pair_stability >= 0.5
        and ranking_safe
    ):
        return "PROVISIONAL_ACCEPT_GOAL_MARGIN"
    return "KEEP_GOAL_MARGIN_AS_CANDIDATE"


def write_report(
    path: Path,
    seasons: tuple[str, ...],
    selections: pd.DataFrame,
    fold_results: pd.DataFrame,
    competition_summary: pd.DataFrame,
    uncertainty: pd.DataFrame,
    stability: pd.DataFrame,
    core_config: DynamicCoreConfig,
    final_margin: GoalMarginConfig,
    final_metrics: dict[str, float | int],
    multiplier_table: pd.DataFrame,
    outer_sensitivity: pd.DataFrame,
    goal_distribution: pd.DataFrame,
    decision: str,
) -> None:
    pivot = fold_results.pivot(index="fold", columns="model", values="brier")
    fold_wins = int((pivot["goal_margin"] < pivot["core_baseline"]).sum())
    overall = uncertainty.loc[uncertainty["competition"].eq("ALL")].iloc[0]
    selection_rows = [
        f"| {row.fold} | {row.test_season} | {row.selected_goal_weight:g} | "
        f"{row.selected_goal_cap:g} | {row.train_brier_difference:.6f} |"
        for row in selections.itertuples(index=False)
    ]
    competition_rows = [
        f"| {row.competition} | {row.matches} | {row.brier_difference:.6f} | "
        f"{row.log_loss_difference:.6f} |"
        for row in competition_summary.itertuples(index=False)
    ]
    multiplier_rows = [
        f"| {int(row.goal_difference)} | {row.goal_multiplier:.4f} |"
        for row in multiplier_table.itertuples(index=False)
    ]
    sensitivity_rows = [
        f"| {row.goal_weight:g} | {row.goal_cap:g} | {int(row.fold_wins)}/6 | "
        f"{row.brier_difference:.6f} | {row.min_start_end_rank_correlation:.3f} |"
        for row in outer_sensitivity.head(5).itertuples(index=False)
    ]
    distribution_rows = [
        f"| {row.goal_difference} | {row.matches} | {row.share:.1%} |"
        for row in goal_distribution.itertuples(index=False)
    ]
    mode = stability.iloc[0]
    text = "\n".join(
        [
            "# AO Dynamic Elo Goal-Margin Calibration",
            "",
            f"Decision: **{decision}**",
            "",
            "## Scope",
            "",
            f"Seasons: {seasons[0]} through {seasons[-1]}; outer folds: {len(selections)}.",
            "Each fold uses the Scale/H/K selected only from its earlier seasons.",
            "Competition, stage, progression, caps and season carry remain inactive.",
            "",
            "The tested update multiplier is:",
            "",
            "```text",
            "G = min(goal_cap, 1 + goal_weight * ln(goal_difference))",
            "G = 1 for draws and one-goal results",
            "```",
            "",
            "## Walk-Forward Selections",
            "",
            "| Fold | Unseen season | Weight | Cap | Training Brier difference |",
            "| ---: | --- | ---: | ---: | ---: |",
            *selection_rows,
            "",
            f"Goal margin beat the same fold's core in **{fold_wins}/{len(selections)}** unseen folds.",
            f"Overall paired Brier difference: {overall.mean_brier_difference:.6f} ",
            f"(95% CI {overall.ci_95_lower:.6f} to {overall.ci_95_upper:.6f}).",
            "",
            "## Competition Guardrail",
            "",
            "Negative differences favor the goal-margin layer.",
            "",
            "| Competition | Matches | Brier difference | Log-loss difference |",
            "| --- | ---: | ---: | ---: |",
            *competition_rows,
            "",
            "## Stability",
            "",
            f"Most common pair: `weight={mode.selected_goal_weight:g}`, "
            f"`cap={mode.selected_goal_cap:g}` ({int(mode['count'])}/{int(mode.folds)} folds).",
            "",
            "## Full-Data Research Candidate",
            "",
            f"Core: `Scale={core_config.elo_scale:g}`, `H={core_config.home_advantage:g}`, "
            f"`K={core_config.k_factor:g}`.",
            f"Goal margin: `weight={final_margin.goal_weight:g}`, `cap={final_margin.goal_cap:g}`.",
            f"Brier={float(final_metrics['brier']):.6f}; log loss={float(final_metrics['log_loss']):.6f}.",
            "",
            "| Goal difference | Multiplier |",
            "| ---: | ---: |",
            *multiplier_rows,
            "",
            "## Unseen-Season Sensitivity",
            "",
            "This diagnostic compares every fixed margin candidate on the outer test seasons.",
            "It is not used to replace the nested fold selections.",
            "",
            "| Weight | Cap | Fold wins | Brier difference | Minimum rank correlation |",
            "| ---: | ---: | ---: | ---: | ---: |",
            *sensitivity_rows,
            "",
            "## Goal-Difference Distribution",
            "",
            "| Goal difference | Matches | Share |",
            "| ---: | ---: | ---: |",
            *distribution_rows,
            "",
            "Displayed scores can include extra time in a small unresolved subset. This run",
            "does not claim a separate 90/120-minute goal-margin policy; that policy must be",
            "tested after exact extra-time enrichment.",
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
