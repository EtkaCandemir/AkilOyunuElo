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
from scripts.run_goal_margin_calibration import (  # noqa: E402
    read_full_core_config,
    validate_core_fold_contract,
)


STATIC_DATA_ROOT = ROOT / "data" / "backtest_stage_b_2018_2026"
EVENTS_PATH = ROOT / "data" / "dynamic_backtest_2018_2026" / "matches.csv"
CORE_OUTPUT_ROOT = ROOT / "output" / "dynamic_core_calibration_2018_2026"
OUTPUT_ROOT = ROOT / "output" / "competition_multiplier_calibration_2018_2026"
MULTIPLIER_CANDIDATES = (0.50, 0.625, 0.75, 0.875, 1.0)
COMPETITIONS = ("UCL", "UEL", "UECL")


@dataclass(frozen=True, order=True)
class CompetitionMultiplierConfig:
    uel_multiplier: float
    uecl_multiplier: float
    ucl_multiplier: float = 1.0

    def validate(self) -> None:
        values = {
            "ucl_multiplier": self.ucl_multiplier,
            "uel_multiplier": self.uel_multiplier,
            "uecl_multiplier": self.uecl_multiplier,
        }
        for name, value in values.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.ucl_multiplier != 1.0:
            raise ValueError("UCL is the fixed 1.0 reference competition")
        if not self.ucl_multiplier >= self.uel_multiplier >= self.uecl_multiplier:
            raise ValueError("Competition multipliers must satisfy UCL >= UEL >= UECL")

    def for_competition(self, competition: str) -> float:
        self.validate()
        if competition == "UCL":
            return self.ucl_multiplier
        if competition == "UEL":
            return self.uel_multiplier
        if competition == "UECL":
            return self.uecl_multiplier
        raise ValueError(f"Unknown competition: {competition}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate UCL, UEL and UECL dynamic Elo K multipliers"
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

    datasets = load_calibration_data(static_root, events_path)
    seasons = tuple(data.season for data in datasets)
    folds = expanding_folds(seasons)
    core_selections = pd.read_csv(core_output_root / "fold_selections.csv")
    full_core = read_full_core_config(core_output_root / "full_candidate_metrics.csv")
    validate_core_fold_contract(core_selections, folds)
    candidates = candidate_grid()
    baseline = CompetitionMultiplierConfig(1.0, 1.0)

    selection_rows: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for fold_number, (train_seasons, test_season) in enumerate(folds, start=1):
        core_row = core_selections.loc[core_selections["fold"].eq(fold_number)].iloc[0]
        core_config = DynamicCoreConfig(
            float(core_row["selected_scale"]),
            float(core_row["selected_home_advantage"]),
            float(core_row["selected_k"]),
        )
        train_data = tuple(data for data in datasets if data.season in train_seasons)
        test_data = next(data for data in datasets if data.season == test_season)
        selected, train_metrics = select_competition_candidate(
            train_data,
            core_config,
            candidates,
        )
        baseline_train, _ = evaluate_competition_seasons(
            train_data,
            core_config,
            baseline,
        )
        selection_rows.append(
            {
                "fold": fold_number,
                "train_seasons": ",".join(train_seasons),
                "test_season": test_season,
                "core_scale": core_config.elo_scale,
                "core_home_advantage": core_config.home_advantage,
                "core_k": core_config.k_factor,
                "selected_ucl_multiplier": selected.ucl_multiplier,
                "selected_uel_multiplier": selected.uel_multiplier,
                "selected_uecl_multiplier": selected.uecl_multiplier,
                "train_brier": train_metrics["brier"],
                "baseline_train_brier": baseline_train["brier"],
                "train_brier_difference": train_metrics["brier"] - baseline_train["brier"],
            }
        )
        model_predictions: dict[str, pd.DataFrame] = {}
        for model_name, multiplier_config in (
            ("competition_layer", selected),
            ("core_baseline", baseline),
        ):
            metrics, predictions = evaluate_competition_seasons(
                (test_data,),
                core_config,
                multiplier_config,
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
                    "ucl_multiplier": multiplier_config.ucl_multiplier,
                    "uel_multiplier": multiplier_config.uel_multiplier,
                    "uecl_multiplier": multiplier_config.uecl_multiplier,
                    **metrics,
                }
            )
            assert predictions is not None
            model_predictions[model_name] = predictions.rename(
                columns={
                    "expected_home_score": f"{model_name}_expected_home_score",
                    "brier_loss": f"{model_name}_brier_loss",
                    "log_loss": f"{model_name}_log_loss",
                }
            )
        joined = model_predictions["competition_layer"].merge(
            model_predictions["core_baseline"][
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
    ablation_results, ablation_summary = build_ablation_analysis(
        datasets,
        folds,
        selections,
    )

    final_config, final_metrics = select_competition_candidate(
        datasets,
        full_core,
        candidates,
    )
    full_candidate_metrics = candidate_metrics(datasets, full_core, candidates)
    final_season_metrics = evaluate_by_season(datasets, full_core, final_config)
    effective_k_table = build_effective_k_table(full_core, final_config)
    decision = calibration_decision(
        selections,
        fold_results,
        competition_summary,
        uncertainty,
        stability,
        final_config,
    )

    selections.to_csv(output_root / "fold_selections.csv", index=False)
    fold_results.to_csv(output_root / "fold_results.csv", index=False)
    predictions.to_csv(output_root / "unseen_match_predictions.csv", index=False)
    competition_summary.to_csv(output_root / "competition_summary.csv", index=False)
    uncertainty.to_csv(output_root / "paired_uncertainty.csv", index=False)
    stability.to_csv(output_root / "parameter_stability.csv", index=False)
    full_candidate_metrics.to_csv(output_root / "full_candidate_metrics.csv", index=False)
    final_season_metrics.to_csv(output_root / "final_candidate_season_metrics.csv", index=False)
    effective_k_table.to_csv(output_root / "effective_k_table.csv", index=False)
    ablation_results.to_csv(output_root / "ablation_fold_results.csv", index=False)
    ablation_summary.to_csv(output_root / "ablation_summary.csv", index=False)
    write_report(
        output_root / "calibration_report.md",
        seasons,
        selections,
        fold_results,
        competition_summary,
        uncertainty,
        stability,
        full_core,
        final_config,
        final_metrics,
        effective_k_table,
        ablation_summary,
        decision,
    )

    print("AO dynamic Elo competition-multiplier calibration")
    print(f"Seasons: {len(seasons)}")
    print(f"Matches: {sum(len(data.match_ids) for data in datasets)}")
    print(f"Outer folds: {len(folds)}")
    print(
        "Full-data candidate: "
        f"UCL=1, UEL={final_config.uel_multiplier:g}, "
        f"UECL={final_config.uecl_multiplier:g}"
    )
    print(f"Decision: {decision}")
    print(f"Report: {output_root / 'calibration_report.md'}")


def candidate_grid() -> tuple[CompetitionMultiplierConfig, ...]:
    return tuple(
        CompetitionMultiplierConfig(uel_multiplier, uecl_multiplier)
        for uel_multiplier in MULTIPLIER_CANDIDATES
        for uecl_multiplier in MULTIPLIER_CANDIDATES
        if uecl_multiplier <= uel_multiplier
    )


def run_competition_season(
    data: SeasonData,
    core_config: DynamicCoreConfig,
    multiplier_config: CompetitionMultiplierConfig,
    *,
    return_predictions: bool = False,
) -> tuple[dict[str, float | int], pd.DataFrame | None]:
    core_config.validate()
    multiplier_config.validate()
    ratings = data.initial_ratings.copy()
    brier_sum = 0.0
    log_loss_sum = 0.0
    max_abs_match_delta = 0.0
    prediction_rows: list[dict[str, object]] = []
    for index, (home_id, away_id, actual, neutral, competition) in enumerate(
        zip(
            data.home_team_ids,
            data.away_team_ids,
            data.actual_home_scores,
            data.neutral_flags,
            data.competitions,
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
        competition_multiplier = multiplier_config.for_competition(str(competition))
        delta = core_config.k_factor * competition_multiplier * (actual - probability)
        brier_sum += brier_loss
        log_loss_sum += log_loss
        max_abs_match_delta = max(max_abs_match_delta, abs(delta))
        ratings[home_id] += delta
        ratings[away_id] -= delta
        if return_predictions:
            prediction_rows.append(
                {
                    "match_id": data.match_ids[index],
                    "season": data.season,
                    "competition": competition,
                    "actual_home_score": actual,
                    "competition_multiplier": competition_multiplier,
                    "expected_home_score": probability,
                    "brier_loss": brier_loss,
                    "log_loss": log_loss,
                }
            )

    active = data.active_team_ids
    start_ratings = data.initial_ratings[active]
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
        "matches": len(data.match_ids),
        "brier": brier_sum / len(data.match_ids),
        "log_loss": log_loss_sum / len(data.match_ids),
        "mean_rating_change": float(np.mean(changes)),
        "rating_change_std": float(np.std(changes)),
        "max_abs_rating_change": float(np.max(np.abs(changes))),
        "max_abs_match_delta": max_abs_match_delta,
        "start_end_rank_correlation": float(rank_correlation),
    }
    predictions = pd.DataFrame(prediction_rows) if return_predictions else None
    return metrics, predictions


def evaluate_competition_seasons(
    datasets: tuple[SeasonData, ...],
    core_config: DynamicCoreConfig,
    multiplier_config: CompetitionMultiplierConfig,
    *,
    return_predictions: bool = False,
) -> tuple[dict[str, float | int], pd.DataFrame | None]:
    metric_rows: list[dict[str, float | int]] = []
    frames: list[pd.DataFrame] = []
    for data in datasets:
        metrics, predictions = run_competition_season(
            data,
            core_config,
            multiplier_config,
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
    datasets: tuple[SeasonData, ...],
    core_config: DynamicCoreConfig,
    candidates: tuple[CompetitionMultiplierConfig, ...],
) -> pd.DataFrame:
    rows = []
    for config in candidates:
        metrics, _ = evaluate_competition_seasons(datasets, core_config, config)
        rows.append(
            {
                "ucl_multiplier": config.ucl_multiplier,
                "uel_multiplier": config.uel_multiplier,
                "uecl_multiplier": config.uecl_multiplier,
                "distance_from_neutral": (
                    abs(config.uel_multiplier - 1.0)
                    + abs(config.uecl_multiplier - 1.0)
                ),
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["brier", "log_loss", "distance_from_neutral", "uel_multiplier", "uecl_multiplier"]
    ).reset_index(drop=True)


def select_competition_candidate(
    datasets: tuple[SeasonData, ...],
    core_config: DynamicCoreConfig,
    candidates: tuple[CompetitionMultiplierConfig, ...],
) -> tuple[CompetitionMultiplierConfig, dict[str, float | int]]:
    rows = candidate_metrics(datasets, core_config, candidates)
    selected = rows.iloc[0]
    config = CompetitionMultiplierConfig(
        float(selected["uel_multiplier"]),
        float(selected["uecl_multiplier"]),
    )
    return config, {
        column: selected[column]
        for column in rows.columns
        if column not in {
            "ucl_multiplier", "uel_multiplier", "uecl_multiplier", "distance_from_neutral"
        }
    }


def evaluate_by_season(
    datasets: tuple[SeasonData, ...],
    core_config: DynamicCoreConfig,
    multiplier_config: CompetitionMultiplierConfig,
) -> pd.DataFrame:
    rows = []
    for data in datasets:
        metrics, _ = run_competition_season(data, core_config, multiplier_config)
        rows.append({"season": data.season, **metrics})
    return pd.DataFrame(rows)


def summarize_competitions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for competition, data in predictions.groupby("competition"):
        layer_brier = data["competition_layer_brier_loss"].mean()
        core_brier = data["core_baseline_brier_loss"].mean()
        layer_log = data["competition_layer_log_loss"].mean()
        core_log = data["core_baseline_log_loss"].mean()
        rows.append(
            {
                "competition": competition,
                "matches": len(data),
                "competition_layer_brier": layer_brier,
                "core_brier": core_brier,
                "brier_difference": layer_brier - core_brier,
                "competition_layer_log_loss": layer_log,
                "core_log_loss": core_log,
                "log_loss_difference": layer_log - core_log,
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
            data["competition_layer_brier_loss"] - data["core_baseline_brier_loss"]
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
    rows = []
    for column in ("selected_uel_multiplier", "selected_uecl_multiplier"):
        counts = selections[column].value_counts().sort_values(ascending=False)
        rows.append(
            {
                "parameter": column,
                "mode": counts.index[0],
                "mode_count": int(counts.iloc[0]),
                "folds": len(selections),
                "mode_share": float(counts.iloc[0] / len(selections)),
                "min": float(selections[column].min()),
                "max": float(selections[column].max()),
            }
        )
    return pd.DataFrame(rows)


def build_effective_k_table(
    core_config: DynamicCoreConfig,
    multiplier_config: CompetitionMultiplierConfig,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "competition": COMPETITIONS,
            "multiplier": [
                multiplier_config.for_competition(competition)
                for competition in COMPETITIONS
            ],
            "effective_k": [
                core_config.k_factor * multiplier_config.for_competition(competition)
                for competition in COMPETITIONS
            ],
        }
    )


def build_ablation_analysis(
    datasets: tuple[SeasonData, ...],
    folds: tuple[tuple[tuple[str, ...], str], ...],
    selections: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for fold_number, (_, test_season) in enumerate(folds, start=1):
        selected = selections.loc[selections["fold"].eq(fold_number)].iloc[0]
        core_config = DynamicCoreConfig(
            float(selected["core_scale"]),
            float(selected["core_home_advantage"]),
            float(selected["core_k"]),
        )
        uel = float(selected["selected_uel_multiplier"])
        uecl = float(selected["selected_uecl_multiplier"])
        test_data = next(data for data in datasets if data.season == test_season)
        configs = (
            ("combined", CompetitionMultiplierConfig(uel, uecl)),
            ("uel_and_below", CompetitionMultiplierConfig(uel, uel)),
            ("uecl_only", CompetitionMultiplierConfig(1.0, uecl)),
            ("core_baseline", CompetitionMultiplierConfig(1.0, 1.0)),
        )
        for model_name, config in configs:
            metrics, predictions = evaluate_competition_seasons(
                (test_data,),
                core_config,
                config,
                return_predictions=True,
            )
            rows.append(
                {
                    "fold": fold_number,
                    "test_season": test_season,
                    "model": model_name,
                    "uel_multiplier": config.uel_multiplier,
                    "uecl_multiplier": config.uecl_multiplier,
                    **metrics,
                }
            )
            assert predictions is not None
            predictions = predictions.copy()
            predictions.insert(0, "model", model_name)
            predictions.insert(0, "fold", fold_number)
            prediction_rows.append(predictions)
    all_predictions = pd.concat(prediction_rows, ignore_index=True)
    baseline = all_predictions.loc[all_predictions["model"].eq("core_baseline")][
        ["fold", "match_id", "brier_loss", "log_loss"]
    ].rename(
        columns={"brier_loss": "baseline_brier_loss", "log_loss": "baseline_log_loss"}
    )
    comparisons = all_predictions.merge(
        baseline,
        on=["fold", "match_id"],
        validate="many_to_one",
    )
    comparisons["brier_difference"] = (
        comparisons["brier_loss"] - comparisons["baseline_brier_loss"]
    )
    comparisons["log_loss_difference"] = (
        comparisons["log_loss"] - comparisons["baseline_log_loss"]
    )
    summary = (
        comparisons.groupby(["model", "competition"])
        .agg(
            matches=("match_id", "size"),
            brier_difference=("brier_difference", "mean"),
            log_loss_difference=("log_loss_difference", "mean"),
        )
        .reset_index()
    )
    return pd.DataFrame(rows), summary


def calibration_decision(
    selections: pd.DataFrame,
    fold_results: pd.DataFrame,
    competition_summary: pd.DataFrame,
    uncertainty: pd.DataFrame,
    stability: pd.DataFrame,
    final_config: CompetitionMultiplierConfig,
) -> str:
    pivot = fold_results.pivot(index="fold", columns="model", values="brier")
    fold_wins = int((pivot["competition_layer"] < pivot["core_baseline"]).sum())
    overall = uncertainty.loc[uncertainty["competition"].eq("ALL")].iloc[0]
    no_competition_harm = not bool((competition_summary["brier_difference"] > 0).any())
    stable = bool(stability["mode_share"].ge(0.5).all())
    layer_rows = fold_results.loc[fold_results["model"].eq("competition_layer")]
    ranking_safe = bool(
        layer_rows["start_end_rank_correlation"].ge(RANK_CORRELATION_FLOOR).all()
        and layer_rows["max_abs_rating_change"].le(MAX_RATING_MOVE_GUARDRAIL).all()
    )
    neutral = final_config == CompetitionMultiplierConfig(1.0, 1.0)
    boundary_hit = (
        final_config.uel_multiplier == min(MULTIPLIER_CANDIDATES)
        or final_config.uecl_multiplier == min(MULTIPLIER_CANDIDATES)
    )
    if neutral:
        return "REJECT_COMPETITION_MULTIPLIERS_KEEP_CORE"
    if (
        fold_wins >= 5
        and bool(overall["reliable_improvement"])
        and no_competition_harm
        and stable
        and ranking_safe
    ):
        if boundary_hit:
            return "COMPETITION_SIGNAL_CONFIRMED_PARAMETER_BOUNDARY"
        return "PROVISIONAL_ACCEPT_COMPETITION_MULTIPLIERS"
    return "KEEP_COMPETITION_MULTIPLIERS_AS_CANDIDATE"


def write_report(
    path: Path,
    seasons: tuple[str, ...],
    selections: pd.DataFrame,
    fold_results: pd.DataFrame,
    competition_summary: pd.DataFrame,
    uncertainty: pd.DataFrame,
    stability: pd.DataFrame,
    core_config: DynamicCoreConfig,
    final_config: CompetitionMultiplierConfig,
    final_metrics: dict[str, float | int],
    effective_k_table: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    decision: str,
) -> None:
    pivot = fold_results.pivot(index="fold", columns="model", values="brier")
    fold_wins = int((pivot["competition_layer"] < pivot["core_baseline"]).sum())
    overall = uncertainty.loc[uncertainty["competition"].eq("ALL")].iloc[0]
    selection_rows = [
        f"| {row.fold} | {row.test_season} | {row.selected_uel_multiplier:g} | "
        f"{row.selected_uecl_multiplier:g} | {row.train_brier_difference:.6f} |"
        for row in selections.itertuples(index=False)
    ]
    competition_rows = [
        f"| {row.competition} | {row.matches} | {row.brier_difference:.6f} | "
        f"{row.log_loss_difference:.6f} |"
        for row in competition_summary.itertuples(index=False)
    ]
    stability_rows = [
        f"| {row.parameter} | {row.mode:g} | {row.mode_count}/{row.folds} | "
        f"{row.min:g}-{row.max:g} |"
        for row in stability.itertuples(index=False)
    ]
    effective_k_rows = [
        f"| {row.competition} | {row.multiplier:g} | {row.effective_k:g} |"
        for row in effective_k_table.itertuples(index=False)
    ]
    ablation_rows = [
        f"| {row.model} | {row.competition} | {row.matches} | "
        f"{row.brier_difference:.6f} | {row.log_loss_difference:.6f} |"
        for row in ablation_summary.itertuples(index=False)
        if row.model != "core_baseline"
    ]
    text = "\n".join(
        [
            "# AO Dynamic Elo Competition-Multiplier Calibration",
            "",
            f"Decision: **{decision}**",
            "",
            "## Scope",
            "",
            f"Seasons: {seasons[0]} through {seasons[-1]}; outer folds: {len(selections)}.",
            "UCL is the fixed 1.0 reference. The enforced hierarchy is UCL >= UEL >= UECL.",
            "These values alter effective K; they are not additions to team strength ratings.",
            "Goal margin, stage, progression, update caps and season carry are inactive.",
            "Each fold inherits the Scale/H/K selected only from its earlier seasons.",
            "",
            "## Walk-Forward Selections",
            "",
            "| Fold | Unseen season | UEL multiplier | UECL multiplier | Training Brier difference |",
            "| ---: | --- | ---: | ---: | ---: |",
            *selection_rows,
            "",
            f"Competition multipliers beat the core in **{fold_wins}/{len(selections)}** unseen folds.",
            f"Overall paired Brier difference: {overall.mean_brier_difference:.6f} ",
            f"(95% CI {overall.ci_95_lower:.6f} to {overall.ci_95_upper:.6f}).",
            "",
            "## Competition Guardrail",
            "",
            "Negative differences favor the competition layer.",
            "",
            "| Competition | Matches | Brier difference | Log-loss difference |",
            "| --- | ---: | ---: | ---: |",
            *competition_rows,
            "",
            "## Parameter Stability",
            "",
            "| Parameter | Mode | Fold frequency | Range |",
            "| --- | ---: | ---: | ---: |",
            *stability_rows,
            "",
            "## Full-Data Research Candidate",
            "",
            f"Core: `Scale={core_config.elo_scale:g}`, `H={core_config.home_advantage:g}`, "
            f"`K={core_config.k_factor:g}`.",
            f"Multipliers: `UCL=1`, `UEL={final_config.uel_multiplier:g}`, "
            f"`UECL={final_config.uecl_multiplier:g}`.",
            f"Brier={float(final_metrics['brier']):.6f}; log loss={float(final_metrics['log_loss']):.6f}.",
            "",
            "| Competition | Multiplier | Effective K |",
            "| --- | ---: | ---: |",
            *effective_k_rows,
            "",
            "## Independent Ablation",
            "",
            "`uel_and_below` applies the UEL value to both lower competitions; `uecl_only`",
            "fixes UEL at 1.0. Negative",
            "differences favor the ablated layer against the same core baseline.",
            "",
            "| Model | Competition | Matches | Brier difference | Log-loss difference |",
            "| --- | --- | ---: | ---: | ---: |",
            *ablation_rows,
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
