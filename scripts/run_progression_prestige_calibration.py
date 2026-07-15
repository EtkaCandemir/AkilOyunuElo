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

from scripts.run_competition_prestige_calibration import (  # noqa: E402
    MIN_PRESTIGE_GAP,
    REFERENCE_UECL_PRESTIGE,
    REFERENCE_UEL_PRESTIGE,
    UECL_PRESTIGE_CANDIDATES,
    UEL_PRESTIGE_CANDIDATES,
)
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
OUTPUT_ROOT = ROOT / "output" / "progression_prestige_calibration_2018_2026"
PROGRESSION_RATIOS = (0.0, 0.25, 0.50, 0.75, 1.0)
COMPETITIONS = ("UCL", "UEL", "UECL")


@dataclass(frozen=True, order=True)
class ProgressionPrestigeConfig:
    progression_ratio: float
    uel_prestige: float
    uecl_prestige: float
    ucl_prestige: float = 1.0

    def validate(self) -> None:
        values = {
            "progression_ratio": self.progression_ratio,
            "ucl_prestige": self.ucl_prestige,
            "uel_prestige": self.uel_prestige,
            "uecl_prestige": self.uecl_prestige,
        }
        for name, value in values.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be non-negative and finite")
        if self.progression_ratio > 1:
            raise ValueError("progression_ratio must be in [0,1]")
        if self.ucl_prestige != 1.0:
            raise ValueError("UCL prestige is the fixed 1.0 reference")
        if not self.ucl_prestige > self.uel_prestige > self.uecl_prestige > 0:
            raise ValueError("Prestige hierarchy must satisfy UCL > UEL > UECL > 0")
        if self.uel_prestige - self.uecl_prestige < MIN_PRESTIGE_GAP - 1e-12:
            raise ValueError("UEL and UECL prestige values must differ by at least 0.10")

    def for_competition(self, competition: str) -> float:
        self.validate()
        values = {
            "UCL": self.ucl_prestige,
            "UEL": self.uel_prestige,
            "UECL": self.uecl_prestige,
        }
        try:
            return values[competition]
        except KeyError as error:
            raise ValueError(f"Unknown competition: {competition}") from error


@dataclass(frozen=True)
class ProgressionSeasonData:
    core: SeasonData
    tie_ids: np.ndarray
    leg_numbers: np.ndarray
    tie_decider_flags: np.ndarray
    knockout_flags: np.ndarray
    advanced_team_ids: np.ndarray

    @property
    def season(self) -> str:
        return self.core.season


@dataclass(frozen=True)
class TieExpectation:
    team_a: int
    team_b: int
    expected_a_to_advance: float


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate strict UCL/UEL/UECL knockout-progression prestige"
    )
    parser.add_argument("--static-data-root", type=Path, default=STATIC_DATA_ROOT)
    parser.add_argument("--events-path", type=Path, default=EVENTS_PATH)
    parser.add_argument("--core-output-root", type=Path, default=CORE_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    datasets = load_progression_data(
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
    positive_candidates = tuple(
        candidate for candidate in candidates if candidate.progression_ratio > 0
    )
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
        positive_challenger, positive_train_metrics = select_candidate(
            train_data,
            core_config,
            positive_candidates,
        )
        baseline_train, _ = evaluate_seasons(train_data, core_config, baseline)
        selection_rows.append(
            {
                "fold": fold_number,
                "train_seasons": ",".join(train_seasons),
                "test_season": test_season,
                "core_scale": core_config.elo_scale,
                "core_home_advantage": core_config.home_advantage,
                "core_k": core_config.k_factor,
                "selected_progression_ratio": selected.progression_ratio,
                "selected_ucl_prestige": selected.ucl_prestige,
                "selected_uel_prestige": selected.uel_prestige,
                "selected_uecl_prestige": selected.uecl_prestige,
                "train_brier": train_metrics["brier"],
                "baseline_train_brier": baseline_train["brier"],
                "train_brier_difference": train_metrics["brier"] - baseline_train["brier"],
                "challenger_progression_ratio": positive_challenger.progression_ratio,
                "challenger_ucl_prestige": positive_challenger.ucl_prestige,
                "challenger_uel_prestige": positive_challenger.uel_prestige,
                "challenger_uecl_prestige": positive_challenger.uecl_prestige,
                "challenger_train_brier_difference": (
                    positive_train_metrics["brier"] - baseline_train["brier"]
                ),
            }
        )
        model_predictions: dict[str, pd.DataFrame] = {}
        for model_name, progression_config in (
            ("progression_layer", selected),
            ("positive_challenger", positive_challenger),
            ("core_baseline", baseline),
        ):
            metrics, predictions = evaluate_seasons(
                (test_data,),
                core_config,
                progression_config,
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
                    "progression_ratio": progression_config.progression_ratio,
                    "ucl_prestige": progression_config.ucl_prestige,
                    "uel_prestige": progression_config.uel_prestige,
                    "uecl_prestige": progression_config.uecl_prestige,
                    **metrics,
                }
            )
            assert predictions is not None
            model_predictions[model_name] = predictions.rename(
                columns={
                    "expected_home_score": f"{model_name}_expected_home_score",
                    "brier_loss": f"{model_name}_brier_loss",
                    "log_loss": f"{model_name}_log_loss",
                    "progression_delta": f"{model_name}_progression_delta",
                }
            )
        joined = model_predictions["progression_layer"].merge(
            model_predictions["positive_challenger"]
            [[
                "match_id",
                "positive_challenger_expected_home_score",
                "positive_challenger_brier_loss",
                "positive_challenger_log_loss",
                "positive_challenger_progression_delta",
            ]],
            on="match_id",
            validate="one_to_one",
        ).merge(
            model_predictions["core_baseline"]
            [[
                "match_id",
                "core_baseline_expected_home_score",
                "core_baseline_brier_loss",
                "core_baseline_log_loss",
                "core_baseline_progression_delta",
            ]],
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
    challenger_competition_summary = summarize_competitions(
        predictions,
        model="positive_challenger",
    )
    challenger_uncertainty = paired_uncertainty(
        predictions,
        model="positive_challenger",
    )
    stability = parameter_stability(selections)

    final_config, final_metrics = select_candidate(datasets, full_core, candidates)
    full_candidate_metrics = candidate_metrics(datasets, full_core, candidates)
    final_season_metrics = evaluate_by_season(datasets, full_core, final_config)
    progression_table = build_progression_table(full_core, final_config)
    decision = calibration_decision(
        selections,
        fold_results,
        uncertainty,
        stability,
        final_config,
    )

    selections.to_csv(output_root / "fold_selections.csv", index=False)
    fold_results.to_csv(output_root / "fold_results.csv", index=False)
    predictions.to_csv(output_root / "unseen_match_predictions.csv", index=False)
    competition_summary.to_csv(output_root / "competition_summary.csv", index=False)
    uncertainty.to_csv(output_root / "paired_uncertainty.csv", index=False)
    challenger_competition_summary.to_csv(
        output_root / "positive_challenger_competition_summary.csv",
        index=False,
    )
    challenger_uncertainty.to_csv(
        output_root / "positive_challenger_paired_uncertainty.csv",
        index=False,
    )
    stability.to_csv(output_root / "parameter_stability.csv", index=False)
    full_candidate_metrics.to_csv(output_root / "full_candidate_metrics.csv", index=False)
    final_season_metrics.to_csv(output_root / "final_candidate_season_metrics.csv", index=False)
    progression_table.to_csv(output_root / "progression_k_table.csv", index=False)
    write_report(
        output_root / "calibration_report.md",
        seasons,
        selections,
        fold_results,
        competition_summary,
        uncertainty,
        challenger_competition_summary,
        challenger_uncertainty,
        stability,
        full_core,
        final_config,
        final_metrics,
        progression_table,
        decision,
    )

    print("AO dynamic Elo knockout-progression prestige calibration")
    print(f"Seasons: {len(seasons)}")
    print(f"Matches: {sum(len(data.core.match_ids) for data in datasets)}")
    print(f"Knockout ties: {sum(int(data.tie_decider_flags.sum()) for data in datasets)}")
    print(f"Candidates: {len(candidates)}")
    print(
        "Full-data candidate: "
        f"ratio={final_config.progression_ratio:g}, UCL=1, "
        f"UEL={final_config.uel_prestige:g}, UECL={final_config.uecl_prestige:g}"
    )
    print(f"Decision: {decision}")
    print(f"Report: {output_root / 'calibration_report.md'}")


def baseline_config() -> ProgressionPrestigeConfig:
    return ProgressionPrestigeConfig(
        0.0,
        REFERENCE_UEL_PRESTIGE,
        REFERENCE_UECL_PRESTIGE,
    )


def candidate_grid() -> tuple[ProgressionPrestigeConfig, ...]:
    candidates = {baseline_config()}
    candidates.update(
        ProgressionPrestigeConfig(ratio, uel, uecl)
        for ratio in PROGRESSION_RATIOS
        if ratio > 0
        for uel in UEL_PRESTIGE_CANDIDATES
        for uecl in UECL_PRESTIGE_CANDIDATES
        if uel - uecl >= MIN_PRESTIGE_GAP - 1e-12
    )
    return tuple(sorted(candidates))


def expected_to_advance(
    team_a_rating: float,
    team_b_rating: float,
    core_config: DynamicCoreConfig,
) -> float:
    return expected_home_score(
        team_a_rating,
        team_b_rating,
        core_config,
        neutral=True,
    )


def progression_delta(
    advanced_a: float,
    expected_a: float,
    competition: str,
    core_config: DynamicCoreConfig,
    progression_config: ProgressionPrestigeConfig,
) -> float:
    core_config.validate()
    progression_config.validate()
    if advanced_a not in (0.0, 1.0):
        raise ValueError("advanced_a must be 0 or 1")
    if not 0 <= expected_a <= 1:
        raise ValueError("expected_a must be in [0,1]")
    return (
        core_config.k_factor
        * progression_config.progression_ratio
        * progression_config.for_competition(competition)
        * (advanced_a - expected_a)
    )


def load_progression_data(
    static_root: Path,
    events_path: Path,
) -> tuple[ProgressionSeasonData, ...]:
    core_datasets = load_calibration_data(static_root, events_path)
    events = pd.read_csv(events_path).sort_values(["season", "event_order"])
    required = {
        "match_id", "season", "tie_id", "leg_number", "is_tie_decider",
        "is_knockout", "advanced_team_id", "home_team_id", "away_team_id",
    }
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"Progression event data missing columns: {missing}")
    validate_tie_contract(events)
    event_index = events.set_index("match_id")
    if not event_index.index.is_unique:
        raise ValueError("Progression event match_id must be unique")
    result: list[ProgressionSeasonData] = []
    for core in core_datasets:
        aligned = event_index.loc[core.match_ids]
        if not aligned["season"].eq(core.season).all():
            raise ValueError(f"{core.season}: match alignment changed season")
        result.append(
            ProgressionSeasonData(
                core=core,
                tie_ids=aligned["tie_id"].where(aligned["tie_id"].notna(), None).to_numpy(object),
                leg_numbers=aligned["leg_number"].fillna(0).to_numpy(int),
                tie_decider_flags=aligned["is_tie_decider"].to_numpy(bool),
                knockout_flags=aligned["is_knockout"].to_numpy(bool),
                advanced_team_ids=aligned["advanced_team_id"].fillna(-1).to_numpy(int),
            )
        )
    return tuple(result)


def validate_tie_contract(events: pd.DataFrame) -> None:
    knockout = events.loc[events["is_knockout"].astype(bool)].copy()
    if knockout["tie_id"].isna().any():
        raise ValueError("Every knockout match must have a tie_id")
    non_knockout = events.loc[~events["is_knockout"].astype(bool)]
    if non_knockout["is_tie_decider"].astype(bool).any():
        raise ValueError("Non-knockout matches cannot decide a tie")
    for (season, tie_id), tie in knockout.groupby(["season", "tie_id"], sort=False):
        deciders = tie["is_tie_decider"].astype(bool)
        if int(deciders.sum()) != 1 or not bool(deciders.iloc[-1]):
            raise ValueError(f"{season}/{tie_id}: tie must end with exactly one decider")
        teams = set(tie["home_team_id"].astype(int)) | set(tie["away_team_id"].astype(int))
        if len(teams) != 2:
            raise ValueError(f"{season}/{tie_id}: tie must contain exactly two teams")
        advanced = tie.loc[deciders, "advanced_team_id"].iloc[0]
        if pd.isna(advanced) or int(advanced) not in teams:
            raise ValueError(f"{season}/{tie_id}: advanced_team_id must identify a tie participant")


def run_season(
    data: ProgressionSeasonData,
    core_config: DynamicCoreConfig,
    progression_config: ProgressionPrestigeConfig,
    *,
    return_predictions: bool = False,
) -> tuple[dict[str, float | int], pd.DataFrame | None]:
    core_config.validate()
    progression_config.validate()
    ratings = data.core.initial_ratings.copy()
    open_ties: dict[str, TieExpectation] = {}
    brier_sum = 0.0
    log_loss_sum = 0.0
    max_abs_match_delta = 0.0
    max_abs_progression_delta = 0.0
    progression_events = 0
    prediction_rows: list[dict[str, object]] = []

    for index, (home_id, away_id, actual, neutral, competition) in enumerate(
        zip(
            data.core.home_team_ids,
            data.core.away_team_ids,
            data.core.actual_home_scores,
            data.core.neutral_flags,
            data.core.competitions,
        )
    ):
        tie_id_value = data.tie_ids[index]
        tie_id = None if tie_id_value is None else str(tie_id_value)
        if data.knockout_flags[index] and tie_id not in open_ties:
            if tie_id is None:
                raise ValueError(f"{data.season}/{data.core.match_ids[index]}: missing tie_id")
            open_ties[tie_id] = TieExpectation(
                int(home_id),
                int(away_id),
                expected_to_advance(ratings[home_id], ratings[away_id], core_config),
            )

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
        log_loss = -(
            actual * math.log(probability)
            + (1.0 - actual) * math.log(1.0 - probability)
        )
        match_delta = core_config.k_factor * (actual - probability)
        ratings[home_id] += match_delta
        ratings[away_id] -= match_delta
        brier_sum += brier_loss
        log_loss_sum += log_loss
        max_abs_match_delta = max(max_abs_match_delta, abs(match_delta))

        tie_probability = np.nan
        tie_actual = np.nan
        tie_delta = 0.0
        if data.tie_decider_flags[index]:
            if tie_id is None or tie_id not in open_ties:
                raise ValueError(f"{data.season}/{data.core.match_ids[index]}: unknown deciding tie")
            expectation = open_ties.pop(tie_id)
            advanced_id = int(data.advanced_team_ids[index])
            if advanced_id not in (expectation.team_a, expectation.team_b):
                raise ValueError(f"{data.season}/{tie_id}: invalid advanced team")
            tie_probability = expectation.expected_a_to_advance
            tie_actual = 1.0 if advanced_id == expectation.team_a else 0.0
            tie_delta = progression_delta(
                tie_actual,
                tie_probability,
                str(competition),
                core_config,
                progression_config,
            )
            ratings[expectation.team_a] += tie_delta
            ratings[expectation.team_b] -= tie_delta
            progression_events += 1
            max_abs_progression_delta = max(max_abs_progression_delta, abs(tie_delta))

        if return_predictions:
            prediction_rows.append(
                {
                    "match_id": data.core.match_ids[index],
                    "season": data.season,
                    "competition": competition,
                    "tie_id": tie_id,
                    "leg_number": data.leg_numbers[index],
                    "is_tie_decider": bool(data.tie_decider_flags[index]),
                    "actual_home_score": actual,
                    "expected_home_score": probability,
                    "brier_loss": brier_loss,
                    "log_loss": log_loss,
                    "match_delta": match_delta,
                    "expected_team_a_to_advance": tie_probability,
                    "actual_team_a_advanced": tie_actual,
                    "progression_delta": tie_delta,
                }
            )

    if open_ties:
        raise ValueError(f"{data.season}: undecided ties remain: {sorted(open_ties)[:3]}")
    active = data.core.active_team_ids
    start_ratings = data.core.initial_ratings[active]
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
        if pd.isna(rank_correlation):
            rank_correlation = 0.0
    metrics: dict[str, float | int] = {
        "matches": len(data.core.match_ids),
        "progression_events": progression_events,
        "brier": brier_sum / len(data.core.match_ids),
        "log_loss": log_loss_sum / len(data.core.match_ids),
        "mean_rating_change": float(np.mean(changes)),
        "rating_change_std": float(np.std(changes)),
        "max_abs_rating_change": float(np.max(np.abs(changes))),
        "max_abs_match_delta": max_abs_match_delta,
        "max_abs_progression_delta": max_abs_progression_delta,
        "start_end_rank_correlation": float(rank_correlation),
    }
    predictions = pd.DataFrame(prediction_rows) if return_predictions else None
    return metrics, predictions


def evaluate_seasons(
    datasets: tuple[ProgressionSeasonData, ...],
    core_config: DynamicCoreConfig,
    progression_config: ProgressionPrestigeConfig,
    *,
    return_predictions: bool = False,
) -> tuple[dict[str, float | int], pd.DataFrame | None]:
    rows: list[dict[str, float | int]] = []
    frames: list[pd.DataFrame] = []
    for data in datasets:
        metrics, predictions = run_season(
            data,
            core_config,
            progression_config,
            return_predictions=return_predictions,
        )
        rows.append(metrics)
        if predictions is not None:
            frames.append(predictions)
    matches = sum(int(row["matches"]) for row in rows)
    aggregate: dict[str, float | int] = {
        "matches": matches,
        "progression_events": sum(int(row["progression_events"]) for row in rows),
        "brier": sum(float(row["brier"]) * int(row["matches"]) for row in rows) / matches,
        "log_loss": sum(float(row["log_loss"]) * int(row["matches"]) for row in rows) / matches,
        "mean_rating_change": float(np.mean([row["mean_rating_change"] for row in rows])),
        "rating_change_std": float(np.mean([row["rating_change_std"] for row in rows])),
        "max_abs_rating_change": float(max(row["max_abs_rating_change"] for row in rows)),
        "max_abs_match_delta": float(max(row["max_abs_match_delta"] for row in rows)),
        "max_abs_progression_delta": float(max(row["max_abs_progression_delta"] for row in rows)),
        "start_end_rank_correlation": float(min(row["start_end_rank_correlation"] for row in rows)),
    }
    predictions = pd.concat(frames, ignore_index=True) if frames else None
    return aggregate, predictions


def candidate_metrics(
    datasets: tuple[ProgressionSeasonData, ...],
    core_config: DynamicCoreConfig,
    candidates: tuple[ProgressionPrestigeConfig, ...],
) -> pd.DataFrame:
    rows = []
    for config in candidates:
        metrics, _ = evaluate_seasons(datasets, core_config, config)
        rows.append(
            {
                "progression_ratio": config.progression_ratio,
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
        ["brier", "log_loss", "progression_ratio", "distance_from_reference"]
    ).reset_index(drop=True)


def select_candidate(
    datasets: tuple[ProgressionSeasonData, ...],
    core_config: DynamicCoreConfig,
    candidates: tuple[ProgressionPrestigeConfig, ...],
) -> tuple[ProgressionPrestigeConfig, dict[str, float | int]]:
    rows = candidate_metrics(datasets, core_config, candidates)
    selected = rows.iloc[0]
    config = ProgressionPrestigeConfig(
        float(selected["progression_ratio"]),
        float(selected["uel_prestige"]),
        float(selected["uecl_prestige"]),
    )
    excluded = {
        "progression_ratio", "ucl_prestige", "uel_prestige",
        "uecl_prestige", "distance_from_reference",
    }
    return config, {column: selected[column] for column in rows.columns if column not in excluded}


def evaluate_by_season(
    datasets: tuple[ProgressionSeasonData, ...],
    core_config: DynamicCoreConfig,
    progression_config: ProgressionPrestigeConfig,
) -> pd.DataFrame:
    rows = []
    for data in datasets:
        metrics, _ = run_season(data, core_config, progression_config)
        rows.append({"season": data.season, **metrics})
    return pd.DataFrame(rows)


def summarize_competitions(
    predictions: pd.DataFrame,
    *,
    model: str = "progression_layer",
) -> pd.DataFrame:
    rows = []
    for competition, data in predictions.groupby("competition"):
        layer_brier = data[f"{model}_brier_loss"].mean()
        core_brier = data["core_baseline_brier_loss"].mean()
        layer_log = data[f"{model}_log_loss"].mean()
        core_log = data["core_baseline_log_loss"].mean()
        rows.append(
            {
                "competition": competition,
                "matches": len(data),
                "progression_brier": layer_brier,
                "core_brier": core_brier,
                "brier_difference": layer_brier - core_brier,
                "progression_log_loss": layer_log,
                "core_log_loss": core_log,
                "log_loss_difference": layer_log - core_log,
            }
        )
    return pd.DataFrame(rows)


def paired_uncertainty(
    predictions: pd.DataFrame,
    *,
    model: str = "progression_layer",
    bootstrap_samples: int = 4000,
    seed: int = 20260715,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    groups = [("ALL", predictions), *predictions.groupby("competition")]
    for competition, data in groups:
        differences = (
            data[f"{model}_brier_loss"] - data["core_baseline_brier_loss"]
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
        "selected_progression_ratio",
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


def build_progression_table(
    core_config: DynamicCoreConfig,
    progression_config: ProgressionPrestigeConfig,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "competition": COMPETITIONS,
            "prestige_multiplier": [
                progression_config.for_competition(competition)
                for competition in COMPETITIONS
            ],
            "progression_k": [
                core_config.k_factor
                * progression_config.progression_ratio
                * progression_config.for_competition(competition)
                for competition in COMPETITIONS
            ],
        }
    )


def calibration_decision(
    selections: pd.DataFrame,
    fold_results: pd.DataFrame,
    uncertainty: pd.DataFrame,
    stability: pd.DataFrame,
    final_config: ProgressionPrestigeConfig,
) -> str:
    pivot = fold_results.pivot(index="fold", columns="model", values="brier")
    fold_wins = int((pivot["progression_layer"] < pivot["core_baseline"]).sum())
    overall = uncertainty.loc[uncertainty["competition"].eq("ALL")].iloc[0]
    no_reliable_competition_harm = not bool(
        uncertainty.loc[uncertainty["competition"].ne("ALL"), "reliable_harm"].any()
    )
    ratio_stability = float(
        stability.loc[
            stability["parameter"].eq("selected_progression_ratio"), "mode_share"
        ].iloc[0]
    )
    layer_rows = fold_results.loc[fold_results["model"].eq("progression_layer")]
    ranking_safe = bool(
        layer_rows["start_end_rank_correlation"].ge(RANK_CORRELATION_FLOOR).all()
        and layer_rows["max_abs_rating_change"].le(MAX_RATING_MOVE_GUARDRAIL).all()
    )
    if final_config.progression_ratio == 0:
        return "REJECT_PROGRESSION_PRESTIGE_KEEP_CORE"
    if (
        fold_wins >= 5
        and bool(overall["reliable_improvement"])
        and no_reliable_competition_harm
        and ratio_stability >= 0.5
        and ranking_safe
    ):
        if final_config.progression_ratio == max(PROGRESSION_RATIOS):
            return "PROGRESSION_SIGNAL_CONFIRMED_RATIO_BOUNDARY"
        return "PROVISIONAL_ACCEPT_PROGRESSION_PRESTIGE"
    return "KEEP_PROGRESSION_PRESTIGE_AS_CANDIDATE"


def write_report(
    path: Path,
    seasons: tuple[str, ...],
    selections: pd.DataFrame,
    fold_results: pd.DataFrame,
    competition_summary: pd.DataFrame,
    uncertainty: pd.DataFrame,
    challenger_competition_summary: pd.DataFrame,
    challenger_uncertainty: pd.DataFrame,
    stability: pd.DataFrame,
    core_config: DynamicCoreConfig,
    final_config: ProgressionPrestigeConfig,
    final_metrics: dict[str, float | int],
    progression_table: pd.DataFrame,
    decision: str,
) -> None:
    pivot = fold_results.pivot(index="fold", columns="model", values="brier")
    fold_wins = int((pivot["progression_layer"] < pivot["core_baseline"]).sum())
    challenger_wins = int((pivot["positive_challenger"] < pivot["core_baseline"]).sum())
    overall = uncertainty.loc[uncertainty["competition"].eq("ALL")].iloc[0]
    challenger_overall = challenger_uncertainty.loc[
        challenger_uncertainty["competition"].eq("ALL")
    ].iloc[0]
    selection_rows = [
        f"| {row.fold} | {row.test_season} | {row.selected_progression_ratio:g} | "
        f"{row.selected_uel_prestige:g} | {row.selected_uecl_prestige:g} | "
        f"{row.train_brier_difference:.6f} |"
        for row in selections.itertuples(index=False)
    ]
    competition_rows = [
        f"| {row.competition} | {row.matches} | {row.brier_difference:.6f} | "
        f"{row.log_loss_difference:.6f} |"
        for row in competition_summary.itertuples(index=False)
    ]
    challenger_rows = [
        f"| {row.competition} | {row.matches} | {row.brier_difference:.6f} | "
        f"{row.log_loss_difference:.6f} |"
        for row in challenger_competition_summary.itertuples(index=False)
    ]
    challenger_selection_rows = [
        f"| {row.fold} | {row.test_season} | {row.challenger_progression_ratio:g} | "
        f"{row.challenger_uel_prestige:g} | {row.challenger_uecl_prestige:g} | "
        f"{row.challenger_train_brier_difference:.6f} |"
        for row in selections.itertuples(index=False)
    ]
    stability_rows = [
        f"| {row.parameter} | {row.mode:g} | {row.mode_count}/{row.folds} | "
        f"{row.min:g}-{row.max:g} |"
        for row in stability.itertuples(index=False)
    ]
    progression_rows = [
        f"| {row.competition} | {row.prestige_multiplier:g} | {row.progression_k:g} |"
        for row in progression_table.itertuples(index=False)
    ]
    text = "\n".join(
        [
            "# AO Dynamic Elo Knockout Progression Prestige Calibration",
            "",
            f"Decision: **{decision}**",
            "",
            "## Scope",
            "",
            f"Seasons: {seasons[0]} through {seasons[-1]}; outer folds: {len(selections)}.",
            "The frozen dynamic core is updated after every match. A separate zero-sum",
            "progression update is applied once, after a knockout tie is decided. The tie",
            "expectation is frozen before its first leg with home advantage set to zero.",
            "Single-match ties, two-legged ties, penalty decisions and finals are included.",
            "Stage multipliers remain fixed at 1.0.",
            "",
            "```text",
            "K_progression = K_core * progression_ratio",
            "Delta = K_progression * competition_prestige * (Advanced - ExpectedToAdvance)",
            "```",
            "",
            "Competition hierarchy is a hard domain constraint: `UCL=1.00 > UEL > UECL`,",
            "with at least 0.10 between UEL and UECL. It is not inferred from unconstrained K.",
            "",
            "## Walk-Forward Selections",
            "",
            "| Fold | Unseen season | Ratio | UEL | UECL | Training Brier difference |",
            "| ---: | --- | ---: | ---: | ---: | ---: |",
            *selection_rows,
            "",
            f"Progression beat the core in **{fold_wins}/{len(selections)}** unseen folds.",
            f"Overall paired Brier difference: {overall.mean_brier_difference:.6f} ",
            f"(95% CI {overall.ci_95_lower:.6f} to {overall.ci_95_upper:.6f}).",
            "",
            "## Forced Positive Challenger",
            "",
            "To avoid a zero-selected model hiding positive-bonus behavior, the best",
            "strictly positive candidate on each training window is also tested unseen.",
            "",
            "| Fold | Unseen season | Ratio | UEL | UECL | Training Brier difference |",
            "| ---: | --- | ---: | ---: | ---: | ---: |",
            *challenger_selection_rows,
            "",
            f"The forced positive challenger beat core in **{challenger_wins}/{len(selections)}** folds.",
            f"Its overall paired Brier difference: {challenger_overall.mean_brier_difference:.6f} ",
            f"(95% CI {challenger_overall.ci_95_lower:.6f} to "
            f"{challenger_overall.ci_95_upper:.6f}).",
            "",
            "| Competition | Matches | Challenger Brier difference | Log-loss difference |",
            "| --- | ---: | ---: | ---: |",
            *challenger_rows,
            "",
            "## Competition Guardrail",
            "",
            "Negative differences favor the progression layer.",
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
            f"`ratio={final_config.progression_ratio:g}`, `UCL=1`, ",
            f"`UEL={final_config.uel_prestige:g}`, `UECL={final_config.uecl_prestige:g}`; ",
            f"Brier={float(final_metrics['brier']):.6f}; ",
            f"log loss={float(final_metrics['log_loss']):.6f}; ",
            f"progression events={int(final_metrics['progression_events'])}.",
            "",
            "| Competition | Prestige | Effective progression K |",
            "| --- | ---: | ---: |",
            *progression_rows,
            "",
            "## Decision Rule",
            "",
            "Promotion requires at least 5/6 unseen-fold wins, a paired overall 95% CI below",
            "zero, no reliably harmed competition, stable ratio selection and ranking",
            "guardrails. If ratio zero wins, prestige coefficients are unidentified and no",
            "competition multiplier is promoted.",
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
