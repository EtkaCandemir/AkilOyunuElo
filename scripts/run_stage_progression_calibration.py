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
    REFERENCE_UECL_PRESTIGE,
    REFERENCE_UEL_PRESTIGE,
)
from scripts.run_dynamic_core_calibration import (  # noqa: E402
    MAX_RATING_MOVE_GUARDRAIL,
    RANK_CORRELATION_FLOOR,
    DynamicCoreConfig,
    expanding_folds,
    expected_home_score,
)
from scripts.run_goal_margin_calibration import (  # noqa: E402
    read_full_core_config,
    validate_core_fold_contract,
)
from scripts.run_progression_prestige_calibration import (  # noqa: E402
    ProgressionSeasonData,
    TieExpectation,
    expected_to_advance,
    load_progression_data,
    paired_uncertainty,
    summarize_competitions,
)


STATIC_DATA_ROOT = ROOT / "data" / "backtest_stage_b_2018_2026"
EVENTS_PATH = ROOT / "data" / "dynamic_backtest_2018_2026" / "matches.csv"
CORE_OUTPUT_ROOT = ROOT / "output" / "dynamic_core_calibration_2018_2026"
OUTPUT_ROOT = ROOT / "output" / "stage_progression_calibration_2018_2026"
PROGRESSION_RATIOS = (0.0, 0.125, 0.25, 0.50, 0.75, 1.0)
STAGES = (
    "QUALIFYING",
    "KNOCKOUT_PLAYOFF",
    "ROUND_OF_16",
    "QUARTERFINAL",
    "SEMIFINAL",
    "FINAL",
)
CALIBRATABLE_STAGES = STAGES[:-1]
COMPETITION_PRESTIGE = {
    "UCL": 1.0,
    "UEL": REFERENCE_UEL_PRESTIGE,
    "UECL": REFERENCE_UECL_PRESTIGE,
}


@dataclass(frozen=True)
class StageProfile:
    name: str
    qualifying: float
    knockout_playoff: float
    round_of_16: float
    quarterfinal: float
    semifinal: float

    def validate(self) -> None:
        values = self.values()
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("Stage multipliers must be non-negative and finite")
        if any(left > right for left, right in zip(values, values[1:])):
            raise ValueError("Stage multipliers must be non-decreasing")
        if max(values) > 2:
            raise ValueError("Stage multipliers cannot exceed 2.0")

    def values(self) -> tuple[float, ...]:
        return (
            self.qualifying,
            self.knockout_playoff,
            self.round_of_16,
            self.quarterfinal,
            self.semifinal,
        )

    def for_stage(self, stage: str) -> float:
        self.validate()
        if stage in {"LEAGUE", "FINAL"}:
            return 0.0
        try:
            return dict(zip(CALIBRATABLE_STAGES, self.values(), strict=True))[stage]
        except KeyError as error:
            raise ValueError(f"Unknown stage: {stage}") from error


STAGE_PROFILES = (
    StageProfile("FLAT", 1.0, 1.0, 1.0, 1.0, 1.0),
    StageProfile("GENTLE_ASCENDING", 0.50, 0.75, 1.00, 1.15, 1.30),
    StageProfile("KNOCKOUT_ASCENDING", 0.25, 0.50, 0.75, 1.00, 1.25),
    StageProfile("LATE_BALANCED", 0.00, 0.40, 0.70, 1.00, 1.25),
    StageProfile("LATE_STRICT", 0.00, 0.00, 0.50, 0.90, 1.30),
    StageProfile("SEMIFINAL_HEAVY", 0.00, 0.25, 0.50, 1.00, 1.50),
)
PROFILE_BY_NAME = {profile.name: profile for profile in STAGE_PROFILES}


@dataclass(frozen=True, order=True)
class StageProgressionConfig:
    progression_ratio: float
    profile_name: str

    def validate(self) -> None:
        if not math.isfinite(self.progression_ratio) or not 0 <= self.progression_ratio <= 1:
            raise ValueError("progression_ratio must be finite and in [0,1]")
        if self.profile_name not in PROFILE_BY_NAME:
            raise ValueError(f"Unknown stage profile: {self.profile_name}")
        self.profile.validate()

    @property
    def profile(self) -> StageProfile:
        try:
            return PROFILE_BY_NAME[self.profile_name]
        except KeyError as error:
            raise ValueError(f"Unknown stage profile: {self.profile_name}") from error


@dataclass(frozen=True)
class StageSeasonData:
    progression: ProgressionSeasonData
    rounds: np.ndarray
    stages: np.ndarray

    @property
    def season(self) -> str:
        return self.progression.season


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate knockout-stage progression profiles"
    )
    parser.add_argument("--static-data-root", type=Path, default=STATIC_DATA_ROOT)
    parser.add_argument("--events-path", type=Path, default=EVENTS_PATH)
    parser.add_argument("--core-output-root", type=Path, default=CORE_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    datasets = load_stage_data(
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
        challenger, challenger_train = select_candidate(
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
                "selected_stage_profile": selected.profile_name,
                "train_brier_difference": train_metrics["brier"] - baseline_train["brier"],
                "challenger_progression_ratio": challenger.progression_ratio,
                "challenger_stage_profile": challenger.profile_name,
                "challenger_train_brier_difference": (
                    challenger_train["brier"] - baseline_train["brier"]
                ),
            }
        )
        model_predictions: dict[str, pd.DataFrame] = {}
        for model_name, stage_config in (
            ("progression_layer", selected),
            ("positive_challenger", challenger),
            ("core_baseline", baseline),
        ):
            metrics, predictions = evaluate_seasons(
                (test_data,),
                core_config,
                stage_config,
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
                    "progression_ratio": stage_config.progression_ratio,
                    "stage_profile": stage_config.profile_name,
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
    challenger_stage_summary = summarize_stages(predictions, model="positive_challenger")
    stability = parameter_stability(selections)

    final_config, final_metrics = select_candidate(datasets, full_core, candidates)
    full_candidate_metrics = candidate_metrics(datasets, full_core, candidates)
    final_season_metrics = evaluate_by_season(datasets, full_core, final_config)
    stage_table = build_stage_table(full_core, final_config)
    stage_counts = build_stage_counts(datasets)
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
        output_root / "positive_challenger_competition_summary.csv", index=False
    )
    challenger_uncertainty.to_csv(
        output_root / "positive_challenger_paired_uncertainty.csv", index=False
    )
    challenger_stage_summary.to_csv(
        output_root / "positive_challenger_stage_summary.csv", index=False
    )
    stability.to_csv(output_root / "parameter_stability.csv", index=False)
    full_candidate_metrics.to_csv(output_root / "full_candidate_metrics.csv", index=False)
    final_season_metrics.to_csv(output_root / "final_candidate_season_metrics.csv", index=False)
    stage_table.to_csv(output_root / "stage_multiplier_table.csv", index=False)
    stage_counts.to_csv(output_root / "stage_counts.csv", index=False)
    write_report(
        output_root / "calibration_report.md",
        seasons,
        selections,
        fold_results,
        challenger_competition_summary,
        challenger_uncertainty,
        challenger_stage_summary,
        stability,
        final_config,
        final_metrics,
        stage_table,
        stage_counts,
        decision,
    )

    print("AO dynamic Elo stage-progression calibration")
    print(f"Seasons: {len(seasons)}")
    print(f"Matches: {sum(len(data.progression.core.match_ids) for data in datasets)}")
    print(f"Knockout ties: {sum(int(data.progression.tie_decider_flags.sum()) for data in datasets)}")
    print(f"Candidates: {len(candidates)}")
    print(
        "Full-data candidate: "
        f"ratio={final_config.progression_ratio:g}, profile={final_config.profile_name}"
    )
    print(f"Decision: {decision}")
    print(f"Report: {output_root / 'calibration_report.md'}")


def baseline_config() -> StageProgressionConfig:
    return StageProgressionConfig(0.0, "FLAT")


def candidate_grid() -> tuple[StageProgressionConfig, ...]:
    candidates = {baseline_config()}
    candidates.update(
        StageProgressionConfig(ratio, profile.name)
        for ratio in PROGRESSION_RATIOS
        if ratio > 0
        for profile in STAGE_PROFILES
    )
    return tuple(sorted(candidates))


def normalize_stage(round_name: str, competition: str, is_knockout: bool) -> str:
    if competition not in COMPETITION_PRESTIGE:
        raise ValueError(f"Unknown competition: {competition}")
    if not is_knockout:
        if round_name in {"Group Stage", "League Stage"}:
            return "LEAGUE"
        raise ValueError(f"Unknown non-knockout round: {competition}/{round_name}")
    if round_name in {
        "Preliminary Round",
        "1st Qualifying Round",
        "2nd Qualifying Round",
        "3rd Qualifying Round",
        "Qualifying Play-off Round",
    }:
        return "QUALIFYING"
    if round_name == "Knockout round play-offs":
        return "KNOCKOUT_PLAYOFF"
    if round_name == "Round 2":
        return "ROUND_OF_16" if competition == "UCL" else "KNOCKOUT_PLAYOFF"
    if round_name in {"Round of 16", "Round 3"}:
        return "ROUND_OF_16"
    mapping = {
        "Quarter Finals": "QUARTERFINAL",
        "Semi Finals": "SEMIFINAL",
        "Final": "FINAL",
    }
    try:
        return mapping[round_name]
    except KeyError as error:
        raise ValueError(f"Unknown knockout round: {competition}/{round_name}") from error


def load_stage_data(static_root: Path, events_path: Path) -> tuple[StageSeasonData, ...]:
    progression_data = load_progression_data(static_root, events_path)
    events = pd.read_csv(events_path).sort_values(["season", "event_order"])
    event_index = events.set_index("match_id")
    if not event_index.index.is_unique:
        raise ValueError("Stage event match_id must be unique")
    result: list[StageSeasonData] = []
    for progression in progression_data:
        aligned = event_index.loc[progression.core.match_ids]
        rounds = aligned["round"].to_numpy(str)
        stages = np.array(
            [
                normalize_stage(str(round_name), str(competition), bool(knockout))
                for round_name, competition, knockout in zip(
                    rounds,
                    progression.core.competitions,
                    progression.knockout_flags,
                )
            ]
        )
        result.append(StageSeasonData(progression, rounds, stages))
    return tuple(result)


def stage_progression_delta(
    advanced_a: float,
    expected_a: float,
    competition: str,
    stage: str,
    core_config: DynamicCoreConfig,
    stage_config: StageProgressionConfig,
) -> float:
    core_config.validate()
    stage_config.validate()
    if advanced_a not in (0.0, 1.0):
        raise ValueError("advanced_a must be 0 or 1")
    if not 0 <= expected_a <= 1:
        raise ValueError("expected_a must be in [0,1]")
    try:
        competition_multiplier = COMPETITION_PRESTIGE[competition]
    except KeyError as error:
        raise ValueError(f"Unknown competition: {competition}") from error
    return (
        core_config.k_factor
        * stage_config.progression_ratio
        * competition_multiplier
        * stage_config.profile.for_stage(stage)
        * (advanced_a - expected_a)
    )


def run_season(
    data: StageSeasonData,
    core_config: DynamicCoreConfig,
    stage_config: StageProgressionConfig,
    *,
    return_predictions: bool = False,
) -> tuple[dict[str, float | int], pd.DataFrame | None]:
    core_config.validate()
    stage_config.validate()
    progression = data.progression
    core = progression.core
    ratings = core.initial_ratings.copy()
    open_ties: dict[str, TieExpectation] = {}
    brier_sum = 0.0
    log_loss_sum = 0.0
    max_abs_progression_delta = 0.0
    progression_events = 0
    nonzero_progression_events = 0
    prediction_rows: list[dict[str, object]] = []
    for index, (home_id, away_id, actual, neutral, competition) in enumerate(
        zip(
            core.home_team_ids,
            core.away_team_ids,
            core.actual_home_scores,
            core.neutral_flags,
            core.competitions,
        )
    ):
        tie_id_value = progression.tie_ids[index]
        tie_id = None if tie_id_value is None else str(tie_id_value)
        if progression.knockout_flags[index] and tie_id not in open_ties:
            if tie_id is None:
                raise ValueError(f"{data.season}/{core.match_ids[index]}: missing tie_id")
            open_ties[tie_id] = TieExpectation(
                int(home_id),
                int(away_id),
                expected_to_advance(ratings[home_id], ratings[away_id], core_config),
            )
        probability = float(
            np.clip(
                expected_home_score(
                    ratings[home_id], ratings[away_id], core_config, neutral=bool(neutral)
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

        tie_probability = np.nan
        tie_actual = np.nan
        progression_delta_value = 0.0
        stage_multiplier = 0.0
        if progression.tie_decider_flags[index]:
            if tie_id is None or tie_id not in open_ties:
                raise ValueError(f"{data.season}/{core.match_ids[index]}: unknown deciding tie")
            expectation = open_ties.pop(tie_id)
            advanced_id = int(progression.advanced_team_ids[index])
            if advanced_id not in (expectation.team_a, expectation.team_b):
                raise ValueError(f"{data.season}/{tie_id}: invalid advanced team")
            tie_probability = expectation.expected_a_to_advance
            tie_actual = 1.0 if advanced_id == expectation.team_a else 0.0
            stage_multiplier = stage_config.profile.for_stage(str(data.stages[index]))
            progression_delta_value = stage_progression_delta(
                tie_actual,
                tie_probability,
                str(competition),
                str(data.stages[index]),
                core_config,
                stage_config,
            )
            ratings[expectation.team_a] += progression_delta_value
            ratings[expectation.team_b] -= progression_delta_value
            progression_events += 1
            if abs(progression_delta_value) > 1e-12:
                nonzero_progression_events += 1
            max_abs_progression_delta = max(
                max_abs_progression_delta,
                abs(progression_delta_value),
            )
        if return_predictions:
            prediction_rows.append(
                {
                    "match_id": core.match_ids[index],
                    "season": data.season,
                    "competition": competition,
                    "round": data.rounds[index],
                    "stage": data.stages[index],
                    "tie_id": tie_id,
                    "is_tie_decider": bool(progression.tie_decider_flags[index]),
                    "actual_home_score": actual,
                    "expected_home_score": probability,
                    "brier_loss": brier_loss,
                    "log_loss": log_loss,
                    "match_delta": match_delta,
                    "expected_team_a_to_advance": tie_probability,
                    "actual_team_a_advanced": tie_actual,
                    "stage_multiplier": stage_multiplier,
                    "progression_delta": progression_delta_value,
                }
            )
    if open_ties:
        raise ValueError(f"{data.season}: undecided ties remain: {sorted(open_ties)[:3]}")
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
            pd.Series(start_ratings), method="spearman"
        )
        if pd.isna(rank_correlation):
            rank_correlation = 0.0
    metrics: dict[str, float | int] = {
        "matches": len(core.match_ids),
        "progression_events": progression_events,
        "nonzero_progression_events": nonzero_progression_events,
        "brier": brier_sum / len(core.match_ids),
        "log_loss": log_loss_sum / len(core.match_ids),
        "mean_rating_change": float(np.mean(changes)),
        "rating_change_std": float(np.std(changes)),
        "max_abs_rating_change": float(np.max(np.abs(changes))),
        "max_abs_progression_delta": max_abs_progression_delta,
        "start_end_rank_correlation": float(rank_correlation),
    }
    predictions = pd.DataFrame(prediction_rows) if return_predictions else None
    return metrics, predictions


def evaluate_seasons(
    datasets: tuple[StageSeasonData, ...],
    core_config: DynamicCoreConfig,
    stage_config: StageProgressionConfig,
    *,
    return_predictions: bool = False,
) -> tuple[dict[str, float | int], pd.DataFrame | None]:
    rows: list[dict[str, float | int]] = []
    frames: list[pd.DataFrame] = []
    for data in datasets:
        metrics, predictions = run_season(
            data, core_config, stage_config, return_predictions=return_predictions
        )
        rows.append(metrics)
        if predictions is not None:
            frames.append(predictions)
    matches = sum(int(row["matches"]) for row in rows)
    aggregate: dict[str, float | int] = {
        "matches": matches,
        "progression_events": sum(int(row["progression_events"]) for row in rows),
        "nonzero_progression_events": sum(
            int(row["nonzero_progression_events"]) for row in rows
        ),
        "brier": sum(float(row["brier"]) * int(row["matches"]) for row in rows) / matches,
        "log_loss": sum(float(row["log_loss"]) * int(row["matches"]) for row in rows) / matches,
        "mean_rating_change": float(np.mean([row["mean_rating_change"] for row in rows])),
        "rating_change_std": float(np.mean([row["rating_change_std"] for row in rows])),
        "max_abs_rating_change": float(max(row["max_abs_rating_change"] for row in rows)),
        "max_abs_progression_delta": float(max(row["max_abs_progression_delta"] for row in rows)),
        "start_end_rank_correlation": float(min(row["start_end_rank_correlation"] for row in rows)),
    }
    predictions = pd.concat(frames, ignore_index=True) if frames else None
    return aggregate, predictions


def candidate_metrics(
    datasets: tuple[StageSeasonData, ...],
    core_config: DynamicCoreConfig,
    candidates: tuple[StageProgressionConfig, ...],
) -> pd.DataFrame:
    rows = []
    for config in candidates:
        metrics, _ = evaluate_seasons(datasets, core_config, config)
        rows.append(
            {
                "progression_ratio": config.progression_ratio,
                "stage_profile": config.profile_name,
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["brier", "log_loss", "progression_ratio", "stage_profile"]
    ).reset_index(drop=True)


def select_candidate(
    datasets: tuple[StageSeasonData, ...],
    core_config: DynamicCoreConfig,
    candidates: tuple[StageProgressionConfig, ...],
) -> tuple[StageProgressionConfig, dict[str, float | int]]:
    rows = candidate_metrics(datasets, core_config, candidates)
    selected = rows.iloc[0]
    config = StageProgressionConfig(
        float(selected["progression_ratio"]),
        str(selected["stage_profile"]),
    )
    return config, {
        column: selected[column]
        for column in rows.columns
        if column not in {"progression_ratio", "stage_profile"}
    }


def evaluate_by_season(
    datasets: tuple[StageSeasonData, ...],
    core_config: DynamicCoreConfig,
    stage_config: StageProgressionConfig,
) -> pd.DataFrame:
    rows = []
    for data in datasets:
        metrics, _ = run_season(data, core_config, stage_config)
        rows.append({"season": data.season, **metrics})
    return pd.DataFrame(rows)


def summarize_stages(predictions: pd.DataFrame, *, model: str) -> pd.DataFrame:
    rows = []
    for stage, data in predictions.groupby("stage"):
        model_brier = data[f"{model}_brier_loss"].mean()
        core_brier = data["core_baseline_brier_loss"].mean()
        model_log = data[f"{model}_log_loss"].mean()
        core_log = data["core_baseline_log_loss"].mean()
        rows.append(
            {
                "stage": stage,
                "matches": len(data),
                "brier_difference": model_brier - core_brier,
                "log_loss_difference": model_log - core_log,
            }
        )
    return pd.DataFrame(rows)


def parameter_stability(selections: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in ("selected_progression_ratio", "selected_stage_profile"):
        counts = selections[column].value_counts().sort_values(ascending=False)
        rows.append(
            {
                "parameter": column,
                "mode": counts.index[0],
                "mode_count": int(counts.iloc[0]),
                "folds": len(selections),
                "mode_share": float(counts.iloc[0] / len(selections)),
                "unique_values": int(selections[column].nunique()),
            }
        )
    return pd.DataFrame(rows)


def build_stage_table(
    core_config: DynamicCoreConfig,
    stage_config: StageProgressionConfig,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stage": STAGES,
            "stage_multiplier": [stage_config.profile.for_stage(stage) for stage in STAGES],
            "ucl_progression_k": [
                core_config.k_factor
                * stage_config.progression_ratio
                * stage_config.profile.for_stage(stage)
                for stage in STAGES
            ],
            "uel_progression_k": [
                core_config.k_factor
                * stage_config.progression_ratio
                * REFERENCE_UEL_PRESTIGE
                * stage_config.profile.for_stage(stage)
                for stage in STAGES
            ],
            "uecl_progression_k": [
                core_config.k_factor
                * stage_config.progression_ratio
                * REFERENCE_UECL_PRESTIGE
                * stage_config.profile.for_stage(stage)
                for stage in STAGES
            ],
        }
    )


def build_stage_counts(datasets: tuple[StageSeasonData, ...]) -> pd.DataFrame:
    frames = []
    for data in datasets:
        deciders = data.progression.tie_decider_flags
        frames.append(
            pd.DataFrame(
                {
                    "season": data.season,
                    "competition": data.progression.core.competitions[deciders],
                    "stage": data.stages[deciders],
                }
            )
        )
    counts = pd.concat(frames, ignore_index=True)
    return (
        counts.groupby(["competition", "stage"])
        .size()
        .rename("ties")
        .reset_index()
    )


def calibration_decision(
    selections: pd.DataFrame,
    fold_results: pd.DataFrame,
    uncertainty: pd.DataFrame,
    stability: pd.DataFrame,
    final_config: StageProgressionConfig,
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
    profile_stability = float(
        stability.loc[
            stability["parameter"].eq("selected_stage_profile"), "mode_share"
        ].iloc[0]
    )
    layer_rows = fold_results.loc[fold_results["model"].eq("progression_layer")]
    ranking_safe = bool(
        layer_rows["start_end_rank_correlation"].ge(RANK_CORRELATION_FLOOR).all()
        and layer_rows["max_abs_rating_change"].le(MAX_RATING_MOVE_GUARDRAIL).all()
    )
    if final_config.progression_ratio == 0:
        return "REJECT_STAGE_PROGRESSION_KEEP_CORE"
    if (
        fold_wins >= 5
        and bool(overall["reliable_improvement"])
        and no_reliable_competition_harm
        and ratio_stability >= 0.5
        and profile_stability >= 0.5
        and ranking_safe
    ):
        if final_config.progression_ratio in {
            min(value for value in PROGRESSION_RATIOS if value > 0),
            max(PROGRESSION_RATIOS),
        }:
            return "STAGE_PROGRESSION_SIGNAL_CONFIRMED_RATIO_BOUNDARY"
        return "PROVISIONAL_ACCEPT_STAGE_PROGRESSION"
    return "KEEP_STAGE_PROGRESSION_AS_CANDIDATE"


def write_report(
    path: Path,
    seasons: tuple[str, ...],
    selections: pd.DataFrame,
    fold_results: pd.DataFrame,
    challenger_competition_summary: pd.DataFrame,
    challenger_uncertainty: pd.DataFrame,
    challenger_stage_summary: pd.DataFrame,
    stability: pd.DataFrame,
    final_config: StageProgressionConfig,
    final_metrics: dict[str, float | int],
    stage_table: pd.DataFrame,
    stage_counts: pd.DataFrame,
    decision: str,
) -> None:
    pivot = fold_results.pivot(index="fold", columns="model", values="brier")
    fold_wins = int((pivot["progression_layer"] < pivot["core_baseline"]).sum())
    challenger_wins = int((pivot["positive_challenger"] < pivot["core_baseline"]).sum())
    challenger_overall = challenger_uncertainty.loc[
        challenger_uncertainty["competition"].eq("ALL")
    ].iloc[0]
    selection_rows = [
        f"| {row.fold} | {row.test_season} | {row.selected_progression_ratio:g} | "
        f"{row.selected_stage_profile} | {row.challenger_progression_ratio:g} | "
        f"{row.challenger_stage_profile} | {row.challenger_train_brier_difference:.6f} |"
        for row in selections.itertuples(index=False)
    ]
    competition_rows = [
        f"| {row.competition} | {row.matches} | {row.brier_difference:.6f} | "
        f"{row.log_loss_difference:.6f} |"
        for row in challenger_competition_summary.itertuples(index=False)
    ]
    stage_rows = [
        f"| {row.stage} | {row.matches} | {row.brier_difference:.6f} | "
        f"{row.log_loss_difference:.6f} |"
        for row in challenger_stage_summary.itertuples(index=False)
    ]
    multiplier_rows = [
        f"| {row.stage} | {row.stage_multiplier:g} | {row.ucl_progression_k:g} | "
        f"{row.uel_progression_k:g} | {row.uecl_progression_k:g} |"
        for row in stage_table.itertuples(index=False)
    ]
    count_rows = [
        f"| {row.competition} | {row.stage} | {row.ties} |"
        for row in stage_counts.itertuples(index=False)
    ]
    stability_rows = [
        f"| {row.parameter} | {row.mode} | {row.mode_count}/{row.folds} | "
        f"{row.unique_values} |"
        for row in stability.itertuples(index=False)
    ]
    text = "\n".join(
        [
            "# AO Dynamic Elo Stage-Progression Calibration",
            "",
            f"Decision: **{decision}**",
            "",
            "## Scope",
            "",
            f"Seasons: {seasons[0]} through {seasons[-1]}; outer folds: {len(selections)}.",
            "This run tests whether the previously rejected flat progression bonus becomes",
            "useful when early qualifying is reduced and later knockout rounds receive",
            "monotonically larger multipliers. The base match update remains Scale/H/K only.",
            "Competition prestige is fixed as a domain reference at UCL=1.00, UEL=0.65 and",
            "UECL=0.45; these values are not calibrated or promoted by this run.",
            "Final progression is fixed at zero because a post-final update has no later",
            "same-season prediction and season carry is deliberately inactive in this run.",
            "",
            "```text",
            "Delta = K_core * progression_ratio * competition_reference",
            "        * stage_multiplier * (Advanced - ExpectedToAdvance)",
            "```",
            "",
            "## Tie Coverage",
            "",
            "| Competition | Normalized stage | Ties |",
            "| --- | --- | ---: |",
            *count_rows,
            "",
            "## Nested Selections",
            "",
            "| Fold | Unseen season | Selected ratio | Selected profile | Positive ratio | Positive profile | Positive train diff |",
            "| ---: | --- | ---: | --- | ---: | --- | ---: |",
            *selection_rows,
            "",
            f"Nested stage progression beat core in **{fold_wins}/{len(selections)}** folds.",
            f"The forced positive challenger beat core in **{challenger_wins}/{len(selections)}** folds.",
            f"Positive challenger Brier difference: {challenger_overall.mean_brier_difference:.6f} ",
            f"(95% CI {challenger_overall.ci_95_lower:.6f} to "
            f"{challenger_overall.ci_95_upper:.6f}).",
            "",
            "## Positive Challenger by Competition",
            "",
            "| Competition | Matches | Brier difference | Log-loss difference |",
            "| --- | ---: | ---: | ---: |",
            *competition_rows,
            "",
            "## Positive Challenger by Prediction Stage",
            "",
            "These rows show where predictions changed because of earlier updates; the FINAL",
            "row does not estimate a post-final bonus effect.",
            "",
            "| Stage | Matches | Brier difference | Log-loss difference |",
            "| --- | ---: | ---: | ---: |",
            *stage_rows,
            "",
            "## Parameter Stability",
            "",
            "| Parameter | Mode | Fold frequency | Unique values |",
            "| --- | --- | ---: | ---: |",
            *stability_rows,
            "",
            "## Full-Data Candidate",
            "",
            f"`ratio={final_config.progression_ratio:g}`, ",
            f"`profile={final_config.profile_name}`; ",
            f"Brier={float(final_metrics['brier']):.6f}; ",
            f"log loss={float(final_metrics['log_loss']):.6f}.",
            "",
            "| Stage | Multiplier | UCL K_progression | UEL K_progression | UECL K_progression |",
            "| --- | ---: | ---: | ---: | ---: |",
            *multiplier_rows,
            "",
            "Because stage profiles were designed after prior 2018-2026 diagnostics, any",
            "positive result remains a research candidate until tested on a future untouched",
            "season. A zero ratio means the competition references are inactive, not that",
            "UCL, UEL or UECL has zero sporting value.",
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
