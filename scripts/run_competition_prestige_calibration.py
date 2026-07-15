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
OUTPUT_ROOT = ROOT / "output" / "competition_prestige_calibration_2018_2026"
WIN_BONUS_CANDIDATES = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0)
UEL_PRESTIGE_CANDIDATES = (0.55, 0.60, 0.65, 0.70, 0.75)
UECL_PRESTIGE_CANDIDATES = (0.35, 0.40, 0.45, 0.50, 0.55)
REFERENCE_UEL_PRESTIGE = 0.65
REFERENCE_UECL_PRESTIGE = 0.45
MIN_PRESTIGE_GAP = 0.10
COMPETITIONS = ("UCL", "UEL", "UECL")


@dataclass(frozen=True, order=True)
class CompetitionPrestigeConfig:
    win_bonus_base: float
    uel_prestige: float
    uecl_prestige: float
    ucl_prestige: float = 1.0

    def validate(self) -> None:
        values = {
            "win_bonus_base": self.win_bonus_base,
            "ucl_prestige": self.ucl_prestige,
            "uel_prestige": self.uel_prestige,
            "uecl_prestige": self.uecl_prestige,
        }
        for name, value in values.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be non-negative and finite")
        if self.ucl_prestige != 1.0:
            raise ValueError("UCL prestige is the fixed 1.0 reference")
        if not self.ucl_prestige > self.uel_prestige > self.uecl_prestige > 0:
            raise ValueError("Prestige hierarchy must satisfy UCL > UEL > UECL > 0")
        if self.uel_prestige - self.uecl_prestige < MIN_PRESTIGE_GAP - 1e-12:
            raise ValueError("UEL and UECL prestige values must differ by at least 0.10")

    def for_competition(self, competition: str) -> float:
        self.validate()
        if competition == "UCL":
            return self.ucl_prestige
        if competition == "UEL":
            return self.uel_prestige
        if competition == "UECL":
            return self.uecl_prestige
        raise ValueError(f"Unknown competition: {competition}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate strict UCL/UEL/UECL win-prestige bonuses"
    )
    parser.add_argument("--static-data-root", type=Path, default=STATIC_DATA_ROOT)
    parser.add_argument("--events-path", type=Path, default=EVENTS_PATH)
    parser.add_argument("--core-output-root", type=Path, default=CORE_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    datasets = load_calibration_data(
        args.static_data_root.resolve(),
        args.events_path.resolve(),
    )
    seasons = tuple(data.season for data in datasets)
    folds = expanding_folds(seasons)
    core_output_root = args.core_output_root.resolve()
    core_selections = pd.read_csv(core_output_root / "fold_selections.csv")
    full_core = read_full_core_config(core_output_root / "full_candidate_metrics.csv")
    validate_core_fold_contract(core_selections, folds)
    candidates = candidate_grid()
    baseline = baseline_config()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

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
        selected, train_metrics = select_candidate(train_data, core_config, candidates)
        baseline_train, _ = evaluate_seasons(train_data, core_config, baseline)
        selection_rows.append(
            {
                "fold": fold_number,
                "train_seasons": ",".join(train_seasons),
                "test_season": test_season,
                "core_scale": core_config.elo_scale,
                "core_home_advantage": core_config.home_advantage,
                "core_k": core_config.k_factor,
                "selected_win_bonus_base": selected.win_bonus_base,
                "selected_ucl_prestige": selected.ucl_prestige,
                "selected_uel_prestige": selected.uel_prestige,
                "selected_uecl_prestige": selected.uecl_prestige,
                "train_brier": train_metrics["brier"],
                "baseline_train_brier": baseline_train["brier"],
                "train_brier_difference": train_metrics["brier"] - baseline_train["brier"],
            }
        )
        model_predictions: dict[str, pd.DataFrame] = {}
        for model_name, prestige_config in (
            ("prestige_layer", selected),
            ("core_baseline", baseline),
        ):
            metrics, predictions = evaluate_seasons(
                (test_data,),
                core_config,
                prestige_config,
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
                    "win_bonus_base": prestige_config.win_bonus_base,
                    "ucl_prestige": prestige_config.ucl_prestige,
                    "uel_prestige": prestige_config.uel_prestige,
                    "uecl_prestige": prestige_config.uecl_prestige,
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
        joined = model_predictions["prestige_layer"].merge(
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

    final_config, final_metrics = select_candidate(datasets, full_core, candidates)
    full_candidate_metrics = candidate_metrics(datasets, full_core, candidates)
    final_season_metrics = evaluate_by_season(datasets, full_core, final_config)
    effective_bonus_table = build_effective_bonus_table(full_core, final_config)
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
    effective_bonus_table.to_csv(output_root / "effective_bonus_table.csv", index=False)
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
        effective_bonus_table,
        decision,
    )

    print("AO dynamic Elo competition-prestige calibration")
    print(f"Seasons: {len(seasons)}")
    print(f"Matches: {sum(len(data.match_ids) for data in datasets)}")
    print(f"Candidates: {len(candidates)}")
    print(
        "Full-data candidate: "
        f"B={final_config.win_bonus_base:g}, UCL=1, "
        f"UEL={final_config.uel_prestige:g}, UECL={final_config.uecl_prestige:g}"
    )
    print(f"Decision: {decision}")
    print(f"Report: {output_root / 'calibration_report.md'}")


def baseline_config() -> CompetitionPrestigeConfig:
    return CompetitionPrestigeConfig(
        0.0,
        REFERENCE_UEL_PRESTIGE,
        REFERENCE_UECL_PRESTIGE,
    )


def candidate_grid() -> tuple[CompetitionPrestigeConfig, ...]:
    candidates = {baseline_config()}
    candidates.update(
        CompetitionPrestigeConfig(win_bonus, uel, uecl)
        for win_bonus in WIN_BONUS_CANDIDATES
        if win_bonus > 0
        for uel in UEL_PRESTIGE_CANDIDATES
        for uecl in UECL_PRESTIGE_CANDIDATES
        if uel - uecl >= MIN_PRESTIGE_GAP - 1e-12
    )
    return tuple(sorted(candidates))


def prestige_delta(
    actual_home_score: float,
    expected_home_score_value: float,
    competition: str,
    config: CompetitionPrestigeConfig,
) -> float:
    config.validate()
    if actual_home_score not in (0.0, 0.5, 1.0):
        raise ValueError("actual_home_score must be 0, 0.5 or 1")
    if not 0 <= expected_home_score_value <= 1:
        raise ValueError("expected_home_score must be in [0,1]")
    if actual_home_score == 0.5 or config.win_bonus_base == 0:
        return 0.0
    return (
        config.win_bonus_base
        * config.for_competition(competition)
        * (actual_home_score - expected_home_score_value)
    )


def run_season(
    data: SeasonData,
    core_config: DynamicCoreConfig,
    prestige_config: CompetitionPrestigeConfig,
    *,
    return_predictions: bool = False,
) -> tuple[dict[str, float | int], pd.DataFrame | None]:
    core_config.validate()
    prestige_config.validate()
    ratings = data.initial_ratings.copy()
    brier_sum = 0.0
    log_loss_sum = 0.0
    max_abs_match_delta = 0.0
    max_abs_prestige_delta = 0.0
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
        base_delta = core_config.k_factor * (actual - probability)
        bonus_delta = prestige_delta(
            float(actual),
            probability,
            str(competition),
            prestige_config,
        )
        total_delta = base_delta + bonus_delta
        brier_sum += brier_loss
        log_loss_sum += log_loss
        max_abs_match_delta = max(max_abs_match_delta, abs(total_delta))
        max_abs_prestige_delta = max(max_abs_prestige_delta, abs(bonus_delta))
        ratings[home_id] += total_delta
        ratings[away_id] -= total_delta
        if return_predictions:
            prediction_rows.append(
                {
                    "match_id": data.match_ids[index],
                    "season": data.season,
                    "competition": competition,
                    "actual_home_score": actual,
                    "competition_prestige": prestige_config.for_competition(str(competition)),
                    "base_delta": base_delta,
                    "prestige_delta": bonus_delta,
                    "total_delta": total_delta,
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
        "max_abs_prestige_delta": max_abs_prestige_delta,
        "start_end_rank_correlation": float(rank_correlation),
    }
    predictions = pd.DataFrame(prediction_rows) if return_predictions else None
    return metrics, predictions


def evaluate_seasons(
    datasets: tuple[SeasonData, ...],
    core_config: DynamicCoreConfig,
    prestige_config: CompetitionPrestigeConfig,
    *,
    return_predictions: bool = False,
) -> tuple[dict[str, float | int], pd.DataFrame | None]:
    metric_rows: list[dict[str, float | int]] = []
    frames: list[pd.DataFrame] = []
    for data in datasets:
        metrics, predictions = run_season(
            data,
            core_config,
            prestige_config,
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
        "max_abs_prestige_delta": float(max(row["max_abs_prestige_delta"] for row in metric_rows)),
        "start_end_rank_correlation": float(min(row["start_end_rank_correlation"] for row in metric_rows)),
    }
    predictions = pd.concat(frames, ignore_index=True) if frames else None
    return aggregate, predictions


def candidate_metrics(
    datasets: tuple[SeasonData, ...],
    core_config: DynamicCoreConfig,
    candidates: tuple[CompetitionPrestigeConfig, ...],
) -> pd.DataFrame:
    rows = []
    for config in candidates:
        metrics, _ = evaluate_seasons(datasets, core_config, config)
        rows.append(
            {
                "win_bonus_base": config.win_bonus_base,
                "ucl_prestige": config.ucl_prestige,
                "uel_prestige": config.uel_prestige,
                "uecl_prestige": config.uecl_prestige,
                "distance_from_reference": (
                    abs(config.uel_prestige - REFERENCE_UEL_PRESTIGE)
                    + abs(config.uecl_prestige - REFERENCE_UECL_PRESTIGE)
                ),
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["brier", "log_loss", "win_bonus_base", "distance_from_reference"]
    ).reset_index(drop=True)


def select_candidate(
    datasets: tuple[SeasonData, ...],
    core_config: DynamicCoreConfig,
    candidates: tuple[CompetitionPrestigeConfig, ...],
) -> tuple[CompetitionPrestigeConfig, dict[str, float | int]]:
    rows = candidate_metrics(datasets, core_config, candidates)
    selected = rows.iloc[0]
    config = CompetitionPrestigeConfig(
        float(selected["win_bonus_base"]),
        float(selected["uel_prestige"]),
        float(selected["uecl_prestige"]),
    )
    return config, {
        column: selected[column]
        for column in rows.columns
        if column not in {
            "win_bonus_base", "ucl_prestige", "uel_prestige",
            "uecl_prestige", "distance_from_reference",
        }
    }


def evaluate_by_season(
    datasets: tuple[SeasonData, ...],
    core_config: DynamicCoreConfig,
    prestige_config: CompetitionPrestigeConfig,
) -> pd.DataFrame:
    rows = []
    for data in datasets:
        metrics, _ = run_season(data, core_config, prestige_config)
        rows.append({"season": data.season, **metrics})
    return pd.DataFrame(rows)


def summarize_competitions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for competition, data in predictions.groupby("competition"):
        prestige_brier = data["prestige_layer_brier_loss"].mean()
        core_brier = data["core_baseline_brier_loss"].mean()
        prestige_log = data["prestige_layer_log_loss"].mean()
        core_log = data["core_baseline_log_loss"].mean()
        rows.append(
            {
                "competition": competition,
                "matches": len(data),
                "prestige_brier": prestige_brier,
                "core_brier": core_brier,
                "brier_difference": prestige_brier - core_brier,
                "prestige_log_loss": prestige_log,
                "core_log_loss": core_log,
                "log_loss_difference": prestige_log - core_log,
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
            data["prestige_layer_brier_loss"] - data["core_baseline_brier_loss"]
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
    for column in (
        "selected_win_bonus_base",
        "selected_uel_prestige",
        "selected_uecl_prestige",
    ):
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


def build_effective_bonus_table(
    core_config: DynamicCoreConfig,
    prestige_config: CompetitionPrestigeConfig,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "competition": COMPETITIONS,
            "prestige_multiplier": [
                prestige_config.for_competition(competition)
                for competition in COMPETITIONS
            ],
            "max_extra_k_on_decisive_result": [
                prestige_config.win_bonus_base * prestige_config.for_competition(competition)
                for competition in COMPETITIONS
            ],
            "total_effective_k_on_decisive_result": [
                core_config.k_factor
                + prestige_config.win_bonus_base * prestige_config.for_competition(competition)
                for competition in COMPETITIONS
            ],
        }
    )


def calibration_decision(
    selections: pd.DataFrame,
    fold_results: pd.DataFrame,
    competition_summary: pd.DataFrame,
    uncertainty: pd.DataFrame,
    stability: pd.DataFrame,
    final_config: CompetitionPrestigeConfig,
) -> str:
    pivot = fold_results.pivot(index="fold", columns="model", values="brier")
    fold_wins = int((pivot["prestige_layer"] < pivot["core_baseline"]).sum())
    overall = uncertainty.loc[uncertainty["competition"].eq("ALL")].iloc[0]
    no_competition_harm = not bool((competition_summary["brier_difference"] > 0).any())
    stable = bool(stability["mode_share"].ge(0.5).all())
    layer_rows = fold_results.loc[fold_results["model"].eq("prestige_layer")]
    ranking_safe = bool(
        layer_rows["start_end_rank_correlation"].ge(RANK_CORRELATION_FLOOR).all()
        and layer_rows["max_abs_rating_change"].le(MAX_RATING_MOVE_GUARDRAIL).all()
    )
    if final_config.win_bonus_base == 0:
        return "REJECT_WIN_PRESTIGE_BONUS_KEEP_CORE"
    if (
        fold_wins >= 5
        and bool(overall["reliable_improvement"])
        and no_competition_harm
        and stable
        and ranking_safe
    ):
        if final_config.win_bonus_base == max(WIN_BONUS_CANDIDATES):
            return "WIN_PRESTIGE_SIGNAL_CONFIRMED_BONUS_BOUNDARY"
        return "PROVISIONAL_ACCEPT_WIN_PRESTIGE_BONUS"
    return "KEEP_WIN_PRESTIGE_BONUS_AS_CANDIDATE"


def write_report(
    path: Path,
    seasons: tuple[str, ...],
    selections: pd.DataFrame,
    fold_results: pd.DataFrame,
    competition_summary: pd.DataFrame,
    uncertainty: pd.DataFrame,
    stability: pd.DataFrame,
    core_config: DynamicCoreConfig,
    final_config: CompetitionPrestigeConfig,
    final_metrics: dict[str, float | int],
    effective_bonus_table: pd.DataFrame,
    decision: str,
) -> None:
    pivot = fold_results.pivot(index="fold", columns="model", values="brier")
    fold_wins = int((pivot["prestige_layer"] < pivot["core_baseline"]).sum())
    overall = uncertainty.loc[uncertainty["competition"].eq("ALL")].iloc[0]
    selection_rows = [
        f"| {row.fold} | {row.test_season} | {row.selected_win_bonus_base:g} | "
        f"{row.selected_uel_prestige:g} | {row.selected_uecl_prestige:g} | "
        f"{row.train_brier_difference:.6f} |"
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
    bonus_rows = [
        f"| {row.competition} | {row.prestige_multiplier:g} | "
        f"{row.max_extra_k_on_decisive_result:g} | "
        f"{row.total_effective_k_on_decisive_result:g} |"
        for row in effective_bonus_table.itertuples(index=False)
    ]
    text = "\n".join(
        [
            "# AO Dynamic Elo Competition Prestige Calibration",
            "",
            f"Decision: **{decision}**",
            "",
            "## Scope",
            "",
            f"Seasons: {seasons[0]} through {seasons[-1]}; outer folds: {len(selections)}.",
            "The base zero-sum Elo update uses the same fold-selected Scale/H/K in every",
            "competition. A separate zero-sum prestige delta is added only on decisive results.",
            "Draws receive no prestige bonus. Goal margin, stage, progression, caps and carry",
            "remain inactive.",
            "",
            "```text",
            "Base delta = K * (S - E)",
            "Prestige delta = B_win * C_competition * (S - E)  # decisive only",
            "Total delta = Base delta + Prestige delta",
            "```",
            "",
            "The domain constraint is strict: `UCL=1.00 > UEL > UECL`, with at least",
            "0.10 between UEL and UECL.",
            "",
            "## Walk-Forward Selections",
            "",
            "| Fold | Unseen season | B_win | UEL | UECL | Training Brier difference |",
            "| ---: | --- | ---: | ---: | ---: | ---: |",
            *selection_rows,
            "",
            f"Prestige bonus beat the core in **{fold_wins}/{len(selections)}** unseen folds.",
            f"Overall paired Brier difference: {overall.mean_brier_difference:.6f} ",
            f"(95% CI {overall.ci_95_lower:.6f} to {overall.ci_95_upper:.6f}).",
            "",
            "## Competition Guardrail",
            "",
            "Negative differences favor the prestige layer.",
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
            f"Prestige: `B_win={final_config.win_bonus_base:g}`, `UCL=1`, "
            f"`UEL={final_config.uel_prestige:g}`, `UECL={final_config.uecl_prestige:g}`.",
            f"Brier={float(final_metrics['brier']):.6f}; log loss={float(final_metrics['log_loss']):.6f}.",
            "",
            "| Competition | Prestige | Maximum extra K | Decisive-result total K |",
            "| --- | ---: | ---: | ---: |",
            *bonus_rows,
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
