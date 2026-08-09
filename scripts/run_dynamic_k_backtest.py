from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ao_elo.config import AOEuropeanEloConfig  # noqa: E402
from ao_elo.controlled_live import update_match_elo  # noqa: E402
from ao_elo.dynamic_k import (  # noqa: E402
    DynamicKConfig,
    baseline_dynamic_k_config,
    calculate_dynamic_match_k,
)
from ao_elo.evaluation import (  # noqa: E402
    dependency_robust_loss_difference_ci,
    schedule_adjusted_team_performance,
)
from ao_elo.pipeline import compute_ao_first_elo_from_csv  # noqa: E402
from ao_elo.robustness import one_x_two_probabilities_scalar  # noqa: E402
from scripts.run_controlled_goal_progression_backtest import (  # noqa: E402
    ControlledSeasonData,
    aggregate_rankings,
    calibration_analysis,
    evaluate_predictions,
    prepare_controlled_data,
    segment_summary,
)
from scripts.run_dynamic_core_calibration import (  # noqa: E402
    DynamicCoreConfig,
    expanding_folds,
)
from scripts.run_final_robustness import (  # noqa: E402
    load_team_season_identity,
    rank_value,
    safe_rank_correlation,
    summarize_ranking,
)
from scripts.run_v2_achievement_reserve_calibration import (  # noqa: E402
    load_reserve_data,
)
from scripts.run_v2_dynamic_calibration import (  # noqa: E402
    MAX_RATING_MOVE_GUARDRAIL,
    RANK_CORRELATION_FLOOR,
)
from scripts.run_v2_evaluation_upgrade import read_events  # noqa: E402


STATIC_DATA_ROOT = ROOT / "data" / "backtest_stage_b_2018_2026"
EVENTS_PATH = (
    ROOT / "data" / "external_elo_benchmark_2018_2026" / "exact_date_events.csv"
)
STATIC_MANIFEST_PATH = (
    ROOT / "output" / "v2_dynamic_calibration_2018_2026"
    / "selected_dynamic_model.json"
)
PRODUCTION_MODEL_PATH = ROOT / "contracts" / "ao_european_elo_v2_production.json"
OUTPUT_ROOT = ROOT / "output" / "dynamic_k_backtest_2018_2026"

LAMBDA_FACTORS = (0.0, 0.15, 0.30, 0.50)
MAX_K_MULTIPLIERS = (1.25, 1.50, 1.75)
AGGREGATIONS = ("ARITHMETIC", "GEOMETRIC")
INACTIVITY_DAYS = (180.0, 270.0, 365.0)
MATCH_EVIDENCE_SCALES = (6.0, 10.0, 15.0)
BASE_MODEL = "BASE"
DYNAMIC_MODEL = "DYNAMIC_K"
RANK_TOLERANCE = 1e-9
PRACTICAL_BRIER_THRESHOLD = 0.0005
SHADOW_BRIER_THRESHOLD = 0.0001


@dataclass(frozen=True)
class DynamicKSeasonData:
    controlled: ControlledSeasonData
    home_exposure: np.ndarray
    away_exposure: np.ndarray
    home_prior_matches: np.ndarray
    away_prior_matches: np.ndarray
    home_days_since: np.ndarray
    away_days_since: np.ndarray

    @property
    def season(self) -> str:
        return self.controlled.season


@dataclass
class DynamicKBacktestEvaluation:
    metrics: dict[str, float | int]
    predictions: pd.DataFrame
    end_ratings: pd.DataFrame
    ranking: pd.DataFrame
    season_metrics: pd.DataFrame


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Backtest team-uncertainty-driven Dynamic K against the frozen "
            "AO production goal-difference baseline"
        )
    )
    parser.add_argument("--static-data-root", type=Path, default=STATIC_DATA_ROOT)
    parser.add_argument("--events-path", type=Path, default=EVENTS_PATH)
    parser.add_argument(
        "--static-manifest",
        type=Path,
        default=STATIC_MANIFEST_PATH,
    )
    parser.add_argument(
        "--production-model",
        type=Path,
        default=PRODUCTION_MODEL_PATH,
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples must be at least 100")

    static_config = read_static_config(args.static_manifest.resolve())
    production = read_production_contract(args.production_model.resolve())
    core = production_core(production)
    events = read_events(args.events_path.resolve())
    reserve_data, _ = load_reserve_data(
        args.static_data_root.resolve(),
        args.events_path.resolve(),
        static_config,
    )
    controlled = prepare_controlled_data(reserve_data, events)
    datasets, data_quality = prepare_dynamic_k_data(
        controlled,
        args.static_data_root.resolve(),
        static_config,
    )
    seasons = tuple(data.season for data in datasets)
    folds = expanding_folds(seasons)
    if len(folds) != 6:
        raise ValueError(f"Expected six outer folds, found {len(folds)}")
    candidates = candidate_grid()
    target = schedule_adjusted_team_performance(events)

    print(
        f"Dynamic K nested walk-forward: {len(candidates)} candidates, "
        f"{len(folds)} folds",
        flush=True,
    )
    nested = run_walk_forward_backtest(
        datasets,
        target,
        folds,
        core,
        production,
        candidates,
    )

    print("Dynamic K fixed-candidate OOS sensitivity", flush=True)
    fixed_metrics = run_fixed_candidate_grid(
        datasets,
        target,
        folds,
        core,
        production,
        candidates,
    )
    full_metrics = evaluate_candidate_set(
        datasets,
        target,
        core,
        production,
        candidates,
    )
    full_candidate_row = select_candidate(full_metrics)
    full_candidate = candidate_from_row(full_candidate_row)

    predictions = add_evidence_bands(nested["predictions"])
    fold_results = nested["fold_results"]
    comparison = build_model_comparison(predictions, fold_results)
    uncertainty = build_uncertainty(
        predictions,
        bootstrap_samples=args.bootstrap_samples,
    )
    competition_summary = segment_summary(predictions, "competition")
    match_band_summary = segment_summary(predictions, "match_band")
    evidence_band_summary = segment_summary(predictions, "evidence_band")
    calibration = calibration_analysis(predictions)
    conservation = build_conservation_summary(
        nested["season_metrics"],
        predictions,
    )
    k_distribution = build_k_distribution(predictions)
    competition_k_distribution = build_segment_k_distribution(
        predictions,
        "competition",
    )
    decision, guardrails = dynamic_k_decision(
        comparison,
        uncertainty,
        fold_results,
        conservation,
    )
    active_candidate = (
        full_candidate
        if decision == "ACTIVATE_DYNAMIC_K"
        else baseline_dynamic_k_config()
    )

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    nested["fold_selections"].to_csv(
        output_root / "fold_selections.csv",
        index=False,
    )
    fold_results.to_csv(output_root / "fold_results.csv", index=False)
    predictions.to_csv(output_root / "unseen_predictions.csv", index=False)
    nested["ranking"].to_csv(output_root / "forward_ranking.csv", index=False)
    nested["rating_distribution"].to_csv(
        output_root / "season_rating_distribution.csv",
        index=False,
    )
    fixed_metrics.to_csv(
        output_root / "fixed_candidate_oos_metrics.csv",
        index=False,
    )
    full_metrics.to_csv(
        output_root / "full_training_candidate_metrics.csv",
        index=False,
    )
    comparison.to_csv(output_root / "model_comparison.csv", index=False)
    uncertainty.to_csv(
        output_root / "dependency_uncertainty.csv",
        index=False,
    )
    competition_summary.to_csv(
        output_root / "competition_summary.csv",
        index=False,
    )
    match_band_summary.to_csv(
        output_root / "match_band_summary.csv",
        index=False,
    )
    evidence_band_summary.to_csv(
        output_root / "evidence_band_summary.csv",
        index=False,
    )
    calibration.to_csv(
        output_root / "calibration_analysis.csv",
        index=False,
    )
    conservation.to_csv(
        output_root / "elo_conservation_audit.csv",
        index=False,
    )
    k_distribution.to_csv(
        output_root / "dynamic_k_distribution.csv",
        index=False,
    )
    competition_k_distribution.to_csv(
        output_root / "competition_k_distribution.csv",
        index=False,
    )
    (output_root / "data_quality_summary.json").write_text(
        json.dumps(data_quality, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_manifest(
        output_root / "selected_dynamic_k_model.json",
        production,
        full_candidate,
        active_candidate,
        decision,
        guardrails,
        comparison,
        uncertainty,
    )
    write_report(
        output_root / "backtest_report.md",
        seasons,
        candidates,
        data_quality,
        nested["fold_selections"],
        comparison,
        uncertainty,
        competition_summary,
        match_band_summary,
        evidence_band_summary,
        k_distribution,
        competition_k_distribution,
        conservation,
        full_candidate,
        active_candidate,
        decision,
        guardrails,
    )
    print(f"Decision: {decision}")
    print(f"Full-development candidate: {full_candidate.key}")
    print(f"Active candidate: {active_candidate.key}")
    print(f"Report: {output_root / 'backtest_report.md'}")


def read_static_config(path: Path) -> AOEuropeanEloConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = AOEuropeanEloConfig(**payload["static_config"])
    config.validate()
    return config


def read_production_contract(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "dynamic_core",
        "active_power_carry",
        "one_x_two_probability",
        "goal_margin",
        "progression_bonus",
        "achievement_reserve",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Production contract missing keys: {missing}")
    core = payload["dynamic_core"]
    goal = payload["goal_margin"]
    if not math.isclose(
        float(core["k_factor"]),
        103.98098633392752,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("Dynamic K backtest requires frozen K=103.98098633392752")
    if not bool(goal["active"]):
        raise ValueError("Dynamic K backtest requires active goal difference")
    expected_goal = {
        "alpha": 0.10,
        "tau": 300.0,
        "goal_difference_cap": 4,
    }
    for key, expected in expected_goal.items():
        if not math.isclose(
            float(goal[key]),
            float(expected),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"Dynamic K baseline requires {key}={expected}")
    if float(payload["active_power_carry"]) != 0.0:
        raise ValueError("Dynamic K backtest requires carry=0")
    if bool(payload["progression_bonus"]["active"]):
        raise ValueError("Dynamic K backtest requires progression disabled")
    if bool(payload["achievement_reserve"]["active"]):
        raise ValueError("Dynamic K backtest requires reserve disabled")
    return payload


def production_core(production: dict[str, object]) -> DynamicCoreConfig:
    core = DynamicCoreConfig(**production["dynamic_core"])
    core.validate()
    return core


def candidate_grid() -> tuple[DynamicKConfig, ...]:
    candidates = {baseline_dynamic_k_config()}
    candidates.update(
        DynamicKConfig(
            lambda_factor,
            max_multiplier,
            aggregation,
            inactivity,
            match_scale,
        )
        for lambda_factor in LAMBDA_FACTORS
        if lambda_factor > 0.0
        for max_multiplier in MAX_K_MULTIPLIERS
        for aggregation in AGGREGATIONS
        for inactivity in INACTIVITY_DAYS
        for match_scale in MATCH_EVIDENCE_SCALES
    )
    result = tuple(sorted(candidates))
    for candidate in result:
        candidate.validate()
    if len(result) != 163:
        raise ValueError(f"Expected 163 unique Dynamic K candidates, found {len(result)}")
    return result


def prepare_dynamic_k_data(
    controlled: tuple[ControlledSeasonData, ...],
    static_root: Path,
    static_config: AOEuropeanEloConfig,
) -> tuple[tuple[DynamicKSeasonData, ...], dict[str, object]]:
    last_match: dict[int, pd.Timestamp] = {}
    result: list[DynamicKSeasonData] = []
    missing_previous_dates = 0
    total_events = 0
    exposure_values: list[float] = []
    prior_match_values: list[float] = []
    inactivity_values: list[float] = []

    for data in controlled:
        season_core = data.reserve.goal.carry.core
        folder = static_root / data.season.replace("/", "-")
        ratings = compute_ao_first_elo_from_csv(
            folder / "teams.csv",
            folder / "country_coefficients.csv",
            folder / "domestic_context.csv",
            folder / "club_european_points.csv",
            static_config,
        )
        exposure_by_team = ratings.set_index("team_id")[
            "european_exposure"
        ].astype(float)
        club = pd.read_csv(folder / "club_european_points.csv")
        match_columns = sorted(
            column for column in club.columns if column.startswith("matches_")
        )
        if len(match_columns) != 5:
            raise ValueError(
                f"{data.season}: expected five historical match columns"
            )
        club["historical_matches"] = club[match_columns].sum(axis=1)
        history_by_team = club.set_index("team_id")["historical_matches"].astype(float)
        active_ids = set(int(value) for value in season_core.active_team_ids)
        if not active_ids.issubset(set(exposure_by_team.index.astype(int))):
            raise ValueError(f"{data.season}: missing exposure for an active team")
        if not active_ids.issubset(set(history_by_team.index.astype(int))):
            raise ValueError(f"{data.season}: missing historical matches for an active team")

        event_count = len(season_core.match_ids)
        home_exposure = np.zeros(event_count, dtype=float)
        away_exposure = np.zeros(event_count, dtype=float)
        home_prior_matches = np.zeros(event_count, dtype=float)
        away_prior_matches = np.zeros(event_count, dtype=float)
        home_days_since = np.full(event_count, np.nan, dtype=float)
        away_days_since = np.full(event_count, np.nan, dtype=float)
        season_matches: dict[int, int] = {}

        for index, (home_value, away_value) in enumerate(
            zip(season_core.home_team_ids, season_core.away_team_ids)
        ):
            home_id = int(home_value)
            away_id = int(away_value)
            kickoff = _utc_timestamp(data.kickoff_utc[index])
            home_exposure[index] = float(exposure_by_team.loc[home_id])
            away_exposure[index] = float(exposure_by_team.loc[away_id])
            home_prior_matches[index] = float(
                history_by_team.loc[home_id] + season_matches.get(home_id, 0)
            )
            away_prior_matches[index] = float(
                history_by_team.loc[away_id] + season_matches.get(away_id, 0)
            )
            for team_id, storage in (
                (home_id, home_days_since),
                (away_id, away_days_since),
            ):
                previous = last_match.get(team_id)
                if previous is None:
                    missing_previous_dates += 1
                else:
                    days = (kickoff - previous).total_seconds() / 86400.0
                    if days < 0.0:
                        raise ValueError(
                            f"{data.season}: team chronology regression for {team_id}"
                        )
                    storage[index] = days
                    inactivity_values.append(days)
                last_match[team_id] = kickoff
            season_matches[home_id] = season_matches.get(home_id, 0) + 1
            season_matches[away_id] = season_matches.get(away_id, 0) + 1
            exposure_values.extend(
                (home_exposure[index], away_exposure[index])
            )
            prior_match_values.extend(
                (home_prior_matches[index], away_prior_matches[index])
            )
            total_events += 1

        result.append(
            DynamicKSeasonData(
                controlled=data,
                home_exposure=home_exposure,
                away_exposure=away_exposure,
                home_prior_matches=home_prior_matches,
                away_prior_matches=away_prior_matches,
                home_days_since=home_days_since,
                away_days_since=away_days_since,
            )
        )

    quality = {
        "seasons": [data.season for data in result],
        "matches": total_events,
        "team_match_sides": total_events * 2,
        "missing_previous_exact_date_sides": missing_previous_dates,
        "missing_previous_exact_date_rate": (
            missing_previous_dates / (total_events * 2)
        ),
        "exposure_min": min(exposure_values),
        "exposure_median": float(np.median(exposure_values)),
        "exposure_max": max(exposure_values),
        "prior_matches_min": min(prior_match_values),
        "prior_matches_median": float(np.median(prior_match_values)),
        "prior_matches_max": max(prior_match_values),
        "known_inactivity_days_median": float(np.median(inactivity_values)),
        "known_inactivity_days_p90": float(np.quantile(inactivity_values, 0.90)),
        "known_inactivity_days_max": max(inactivity_values),
        "uncertainty_weights": {
            "exposure": 1.0 / 3.0,
            "match_evidence": 1.0 / 3.0,
            "inactivity": 1.0 / 3.0,
        },
        "unknown_previous_date_rule": "inactivity_uncertainty=1",
        "ao_first_elo_changed": False,
        "future_information_used": False,
    }
    return tuple(result), quality


def _utc_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def evaluate_sequence(
    datasets: tuple[DynamicKSeasonData, ...],
    target: pd.DataFrame,
    core: DynamicCoreConfig,
    production: dict[str, object],
    candidate: DynamicKConfig,
    *,
    evaluation_seasons: set[str] | None = None,
    ranking_target_seasons: set[str] | None = None,
    return_predictions: bool = False,
) -> DynamicKBacktestEvaluation:
    core.validate()
    candidate.validate()
    evaluation = evaluation_seasons or {data.season for data in datasets}
    goal = production["goal_margin"]
    draw = production["one_x_two_probability"]
    prediction_rows: list[dict[str, object]] = []
    end_rows: list[dict[str, object]] = []
    season_rows: list[dict[str, object]] = []
    total_brier = 0.0
    total_log = 0.0
    total_correct = 0
    total_matches = 0

    for data in datasets:
        controlled = data.controlled
        reserve = controlled.reserve
        season_core = reserve.goal.carry.core
        power = season_core.initial_ratings.copy()
        active_ids = season_core.active_team_ids
        start = power[active_ids].copy()
        initial_total = float(np.sum(start))
        match_deltas: list[float] = []
        match_ks: list[float] = []
        team_ks: list[float] = []
        uncertainty_values: list[float] = []
        max_pair_error = 0.0

        for index, (home_value, away_value, neutral, competition) in enumerate(
            zip(
                season_core.home_team_ids,
                season_core.away_team_ids,
                season_core.neutral_flags,
                season_core.competitions,
            )
        ):
            home_id = int(home_value)
            away_id = int(away_value)
            match_id = str(season_core.match_ids[index])
            dynamic_k = calculate_dynamic_match_k(
                base_k=core.k_factor,
                home_exposure=float(data.home_exposure[index]),
                away_exposure=float(data.away_exposure[index]),
                home_prior_matches=float(data.home_prior_matches[index]),
                away_prior_matches=float(data.away_prior_matches[index]),
                home_days_since_last_match=_optional_days(
                    data.home_days_since[index]
                ),
                away_days_since_last_match=_optional_days(
                    data.away_days_since[index]
                ),
                config=candidate,
            )
            decided_on_penalties = bool(
                reserve.goal.penalty_flags[index]
            )
            home_goals = int(controlled.home_goals[index])
            away_goals = int(controlled.away_goals[index])
            elo_home_goals, elo_away_goals = elo_field_score(
                home_goals,
                away_goals,
                decided_on_penalties,
            )
            update = update_match_elo(
                float(power[home_id]),
                float(power[away_id]),
                elo_home_goals,
                elo_away_goals,
                k_factor=dynamic_k.match_k,
                elo_scale=core.elo_scale,
                home_advantage=core.home_advantage,
                is_neutral=bool(neutral),
                decided_on_penalties=decided_on_penalties,
                alpha=float(goal["alpha"]),
                tau=float(goal["tau"]),
            )
            expected_actual = float(season_core.actual_home_scores[index])
            if not math.isclose(
                update.actual_home_score,
                expected_actual,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"{data.season}/{match_id}: actual score contract changed"
                )
            probabilities = one_x_two_probabilities_scalar(
                update.expected_home_score,
                float(draw["draw_at_even"]),
                float(draw["draw_shape"]),
            )
            observed = (
                0
                if update.actual_home_score == 1.0
                else 1
                if update.actual_home_score == 0.5
                else 2
            )
            target_vector = tuple(
                1.0 if position == observed else 0.0 for position in range(3)
            )
            brier = sum(
                (probability - actual) ** 2
                for probability, actual in zip(probabilities, target_vector)
            )
            log_loss = -math.log(max(probabilities[observed], 1e-15))
            predicted = int(np.argmax(probabilities))

            power[home_id] = update.home_rating_post
            power[away_id] = update.away_rating_post
            match_deltas.append(abs(update.power_delta))
            match_ks.append(dynamic_k.match_k)
            team_ks.extend((dynamic_k.home_k, dynamic_k.away_k))
            uncertainty_values.extend(
                (
                    dynamic_k.home_uncertainty.combined,
                    dynamic_k.away_uncertainty.combined,
                )
            )
            max_pair_error = max(max_pair_error, update.zero_sum_error)

            if data.season in evaluation:
                total_matches += 1
                total_brier += brier
                total_log += log_loss
                total_correct += int(predicted == observed)
                if return_predictions:
                    tie_value = reserve.tie_ids[index]
                    prediction_rows.append(
                        {
                            "candidate_key": candidate.key,
                            "match_id": match_id,
                            "season": data.season,
                            "kickoff_utc": pd.Timestamp(
                                controlled.kickoff_utc[index]
                            ),
                            "competition": str(competition),
                            "round": str(reserve.goal.carry.rounds[index]),
                            "tie_id": (
                                None if tie_value is None else str(tie_value)
                            ),
                            "home_team_id": home_id,
                            "away_team_id": away_id,
                            "home_goals": home_goals,
                            "away_goals": away_goals,
                            "is_neutral": bool(neutral),
                            "decided_on_penalties": decided_on_penalties,
                            "effective_rating_difference": (
                                update.effective_rating_difference
                            ),
                            "expected_home_score": update.expected_home_score,
                            "home_probability": probabilities[0],
                            "draw_probability": probabilities[1],
                            "away_probability": probabilities[2],
                            "actual_class": observed,
                            "predicted_class": predicted,
                            "brier_1x2": brier,
                            "log_loss_1x2": log_loss,
                            "goal_difference": update.goal_difference,
                            "goal_multiplier": (
                                update.goal_difference_multiplier
                            ),
                            "power_delta": update.power_delta,
                            "home_exposure": float(data.home_exposure[index]),
                            "away_exposure": float(data.away_exposure[index]),
                            "home_prior_matches": float(
                                data.home_prior_matches[index]
                            ),
                            "away_prior_matches": float(
                                data.away_prior_matches[index]
                            ),
                            "home_days_since": _optional_days(
                                data.home_days_since[index]
                            ),
                            "away_days_since": _optional_days(
                                data.away_days_since[index]
                            ),
                            "home_uncertainty": (
                                dynamic_k.home_uncertainty.combined
                            ),
                            "away_uncertainty": (
                                dynamic_k.away_uncertainty.combined
                            ),
                            "home_k": dynamic_k.home_k,
                            "away_k": dynamic_k.away_k,
                            "match_k": dynamic_k.match_k,
                            "lambda_factor": candidate.lambda_factor,
                            "max_k_multiplier": candidate.max_k_multiplier,
                            "aggregation": candidate.aggregation,
                            "inactivity_days": candidate.inactivity_days,
                            "match_evidence_scale": (
                                candidate.match_evidence_scale
                            ),
                        }
                    )

        if data.season in evaluation:
            end = power[active_ids]
            season_total_error = abs(float(np.sum(end)) - initial_total)
            season_rows.append(
                {
                    "candidate_key": candidate.key,
                    "season": data.season,
                    "matches": len(season_core.match_ids),
                    "teams": len(active_ids),
                    "rating_min": float(np.min(end)),
                    "rating_max": float(np.max(end)),
                    "rating_mean": float(np.mean(end)),
                    "rating_std": float(np.std(end)),
                    "match_k_mean": float(np.mean(match_ks)),
                    "match_k_p95": float(np.quantile(match_ks, 0.95)),
                    "match_k_max": float(np.max(match_ks)),
                    "team_k_min": float(np.min(team_ks)),
                    "team_k_max": float(np.max(team_ks)),
                    "uncertainty_mean": float(np.mean(uncertainty_values)),
                    "uncertainty_p95": float(
                        np.quantile(uncertainty_values, 0.95)
                    ),
                    "match_delta_mean_abs": float(np.mean(match_deltas)),
                    "match_delta_p95_abs": float(
                        np.quantile(match_deltas, 0.95)
                    ),
                    "match_delta_max_abs": float(np.max(match_deltas)),
                    "max_abs_rating_change": float(
                        np.max(np.abs(end - start))
                    ),
                    "start_end_rank_correlation": safe_rank_correlation(
                        start,
                        end,
                    ),
                    "match_pair_zero_sum_error": max_pair_error,
                    "season_total_elo_error": season_total_error,
                }
            )
            end_rows.extend(
                {
                    "candidate_key": candidate.key,
                    "season": data.season,
                    "team_id": int(team_id),
                    "initial_rating": float(
                        season_core.initial_ratings[team_id]
                    ),
                    "end_live_rating": float(power[team_id]),
                }
                for team_id in active_ids
            )

    if total_matches == 0:
        raise ValueError("No Dynamic K evaluation matches were processed")
    end_ratings = pd.DataFrame(end_rows)
    allowed_targets = ranking_target_seasons or set(target["season"])
    ranking = summarize_ranking(
        end_ratings,
        target,
        allowed_target_seasons=allowed_targets,
        identity=load_team_season_identity(),
    )
    season_metrics = pd.DataFrame(season_rows)
    metrics: dict[str, float | int] = {
        "matches": total_matches,
        "brier_1x2": total_brier / total_matches,
        "log_loss_1x2": total_log / total_matches,
        "accuracy_1x2": total_correct / total_matches,
        "ranking_score": rank_value(ranking, "ranking_score"),
        "pairwise_accuracy": rank_value(ranking, "pairwise_accuracy"),
        "minimum_start_end_rank_correlation": float(
            season_metrics["start_end_rank_correlation"].min()
        ),
        "maximum_abs_rating_change": float(
            season_metrics["max_abs_rating_change"].max()
        ),
        "maximum_abs_match_delta": float(
            season_metrics["match_delta_max_abs"].max()
        ),
        "maximum_match_k": float(season_metrics["match_k_max"].max()),
        "mean_match_k": float(
            np.average(
                season_metrics["match_k_mean"],
                weights=season_metrics["matches"],
            )
        ),
        "mean_team_uncertainty": float(
            np.average(
                season_metrics["uncertainty_mean"],
                weights=season_metrics["matches"],
            )
        ),
        "maximum_total_elo_error": float(
            season_metrics[
                ["match_pair_zero_sum_error", "season_total_elo_error"]
            ].to_numpy(float).max()
        ),
    }
    predictions = pd.DataFrame(prediction_rows)
    if return_predictions:
        metrics.update(evaluate_predictions(predictions))
    return DynamicKBacktestEvaluation(
        metrics=metrics,
        predictions=predictions,
        end_ratings=end_ratings,
        ranking=ranking,
        season_metrics=season_metrics,
    )


def _optional_days(value: float) -> float | None:
    return None if pd.isna(value) else float(value)


def elo_field_score(
    home_goals: int,
    away_goals: int,
    decided_on_penalties: bool,
) -> tuple[int, int]:
    if home_goals < 0 or away_goals < 0:
        raise ValueError("Field-score goals must be non-negative")
    del decided_on_penalties
    return home_goals, away_goals


def candidate_metric_row(
    candidate: DynamicKConfig,
    evaluation: DynamicKBacktestEvaluation,
) -> dict[str, object]:
    return {
        "candidate_key": candidate.key,
        "lambda_factor": candidate.lambda_factor,
        "max_k_multiplier": candidate.max_k_multiplier,
        "aggregation": candidate.aggregation,
        "inactivity_days": candidate.inactivity_days,
        "match_evidence_scale": candidate.match_evidence_scale,
        "complexity": candidate.complexity,
        **evaluation.metrics,
    }


def evaluate_candidate_set(
    datasets: tuple[DynamicKSeasonData, ...],
    target: pd.DataFrame,
    core: DynamicCoreConfig,
    production: dict[str, object],
    candidates: tuple[DynamicKConfig, ...],
) -> pd.DataFrame:
    evaluation_set = {data.season for data in datasets}
    rows = []
    for candidate in candidates:
        evaluation = evaluate_sequence(
            datasets,
            target,
            core,
            production,
            candidate,
            evaluation_seasons=evaluation_set,
            ranking_target_seasons=evaluation_set,
        )
        rows.append(candidate_metric_row(candidate, evaluation))
    return pd.DataFrame(rows)


def candidate_is_safe(row: pd.Series, baseline: pd.Series) -> bool:
    ranking_safe = True
    if pd.notna(row["ranking_score"]) and pd.notna(baseline["ranking_score"]):
        ranking_safe = bool(
            row["ranking_score"] >= baseline["ranking_score"] - RANK_TOLERANCE
            and row["pairwise_accuracy"]
            >= baseline["pairwise_accuracy"] - RANK_TOLERANCE
        )
    return bool(
        ranking_safe
        and row["minimum_start_end_rank_correlation"]
        >= RANK_CORRELATION_FLOOR
        and row["maximum_abs_rating_change"] <= MAX_RATING_MOVE_GUARDRAIL
        and row["maximum_total_elo_error"] <= 1e-9
    )


def select_candidate(metrics: pd.DataFrame) -> pd.Series:
    baseline_key = baseline_dynamic_k_config().key
    baseline_rows = metrics.loc[metrics["candidate_key"].eq(baseline_key)]
    if len(baseline_rows) != 1:
        raise ValueError("Dynamic K metrics must contain exactly one baseline")
    baseline = baseline_rows.iloc[0]
    eligible = metrics.loc[
        metrics.apply(candidate_is_safe, axis=1, baseline=baseline)
    ].copy()
    if eligible.empty:
        return baseline
    aggregation_order = {"ARITHMETIC": 0, "GEOMETRIC": 1}
    eligible["aggregation_order"] = eligible["aggregation"].map(
        aggregation_order
    )
    return eligible.sort_values(
        [
            "ranking_score",
            "pairwise_accuracy",
            "brier_1x2",
            "log_loss_1x2",
            "complexity",
            "lambda_factor",
            "max_k_multiplier",
            "aggregation_order",
            "inactivity_days",
            "match_evidence_scale",
        ],
        ascending=[False, False, True, True, True, True, True, True, True, True],
        na_position="last",
    ).iloc[0]


def candidate_from_row(row: pd.Series) -> DynamicKConfig:
    candidate = DynamicKConfig(
        lambda_factor=float(row["lambda_factor"]),
        max_k_multiplier=float(row["max_k_multiplier"]),
        aggregation=str(row["aggregation"]),
        inactivity_days=float(row["inactivity_days"]),
        match_evidence_scale=float(row["match_evidence_scale"]),
    )
    candidate.validate()
    return candidate


def run_walk_forward_backtest(
    datasets: tuple[DynamicKSeasonData, ...],
    target: pd.DataFrame,
    folds: tuple[tuple[tuple[str, ...], str], ...],
    core: DynamicCoreConfig,
    production: dict[str, object],
    candidates: tuple[DynamicKConfig, ...],
) -> dict[str, pd.DataFrame]:
    by_season = {data.season: data for data in datasets}
    baseline = baseline_dynamic_k_config()
    selection_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    ranking_frames: list[pd.DataFrame] = []
    season_frames: list[pd.DataFrame] = []

    for fold, (train_seasons, test_season) in enumerate(folds, start=1):
        training_data = tuple(by_season[season] for season in train_seasons)
        train_metrics = evaluate_candidate_set(
            training_data,
            target,
            core,
            production,
            candidates,
        )
        selected_row = select_candidate(train_metrics)
        selected = candidate_from_row(selected_row)
        selection_rows.append(
            {
                "fold": fold,
                "train_seasons": "|".join(train_seasons),
                "test_season": test_season,
                "selected_candidate": selected.key,
                "selected_lambda": selected.lambda_factor,
                "selected_max_k_multiplier": selected.max_k_multiplier,
                "selected_aggregation": selected.aggregation,
                "selected_inactivity_days": selected.inactivity_days,
                "selected_match_evidence_scale": selected.match_evidence_scale,
                "train_brier_1x2": float(selected_row["brier_1x2"]),
                "train_log_loss_1x2": float(selected_row["log_loss_1x2"]),
                "train_ranking_score": float(selected_row["ranking_score"]),
                "train_pairwise_accuracy": float(
                    selected_row["pairwise_accuracy"]
                ),
            }
        )
        print(
            f"Fold {fold}/6 test={test_season} selected={selected.key}",
            flush=True,
        )

        for model, candidate in (
            (BASE_MODEL, baseline),
            (DYNAMIC_MODEL, selected),
        ):
            evaluation = evaluate_sequence(
                (by_season[test_season],),
                target,
                core,
                production,
                candidate,
                evaluation_seasons={test_season},
                ranking_target_seasons=set(target["season"]),
                return_predictions=True,
            )
            predictions = evaluation.predictions.copy()
            predictions.insert(0, "fold", fold)
            predictions.insert(1, "model", model)
            prediction_frames.append(predictions)
            seasons = evaluation.season_metrics.copy()
            seasons.insert(0, "fold", fold)
            seasons.insert(1, "model", model)
            season_frames.append(seasons)
            ranking = evaluation.ranking.copy()
            if not ranking.empty:
                ranking.insert(0, "fold", fold)
                ranking.insert(1, "test_season", test_season)
                ranking.insert(2, "model", model)
                ranking_frames.append(ranking)
            fold_rows.append(
                {
                    "fold": fold,
                    "test_season": test_season,
                    "model": model,
                    "candidate_key": candidate.key,
                    "lambda_factor": candidate.lambda_factor,
                    "max_k_multiplier": candidate.max_k_multiplier,
                    "aggregation": candidate.aggregation,
                    "inactivity_days": candidate.inactivity_days,
                    "match_evidence_scale": candidate.match_evidence_scale,
                    **evaluation.metrics,
                }
            )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    base_expected = predictions.loc[
        predictions["model"].eq(BASE_MODEL),
        ["match_id", "expected_home_score"],
    ].rename(columns={"expected_home_score": "base_expected_home_score"})
    predictions = predictions.merge(
        base_expected,
        on="match_id",
        validate="many_to_one",
    )
    predictions["match_band"] = np.select(
        [
            predictions["base_expected_home_score"].ge(0.60),
            predictions["base_expected_home_score"].le(0.40),
        ],
        ["HOME_FAVORITE", "HOME_UNDERDOG"],
        default="BALANCED",
    )
    season_metrics = pd.concat(season_frames, ignore_index=True)
    rating_columns = [
        "fold",
        "model",
        "candidate_key",
        "season",
        "teams",
        "rating_min",
        "rating_max",
        "rating_mean",
        "rating_std",
        "match_k_mean",
        "match_k_p95",
        "match_k_max",
        "uncertainty_mean",
        "uncertainty_p95",
        "match_delta_mean_abs",
        "match_delta_p95_abs",
        "match_delta_max_abs",
        "max_abs_rating_change",
        "start_end_rank_correlation",
    ]
    return {
        "fold_selections": pd.DataFrame(selection_rows),
        "fold_results": pd.DataFrame(fold_rows),
        "predictions": predictions,
        "ranking": (
            pd.concat(ranking_frames, ignore_index=True)
            if ranking_frames
            else pd.DataFrame()
        ),
        "season_metrics": season_metrics,
        "rating_distribution": season_metrics[rating_columns].copy(),
    }


def run_fixed_candidate_grid(
    datasets: tuple[DynamicKSeasonData, ...],
    target: pd.DataFrame,
    folds: tuple[tuple[tuple[str, ...], str], ...],
    core: DynamicCoreConfig,
    production: dict[str, object],
    candidates: tuple[DynamicKConfig, ...],
) -> pd.DataFrame:
    test_seasons = {test_season for _, test_season in folds}
    test_data = tuple(
        data for data in datasets if data.season in test_seasons
    )
    rows = []
    for index, candidate in enumerate(candidates, start=1):
        evaluation = evaluate_sequence(
            test_data,
            target,
            core,
            production,
            candidate,
            evaluation_seasons=test_seasons,
            ranking_target_seasons=set(target["season"]),
        )
        rows.append(candidate_metric_row(candidate, evaluation))
        if index % 25 == 0:
            print(
                f"Fixed OOS Dynamic K candidates: {index}/{len(candidates)}",
                flush=True,
            )
    return pd.DataFrame(rows)


def build_model_comparison(
    predictions: pd.DataFrame,
    fold_results: pd.DataFrame,
) -> pd.DataFrame:
    base_predictions = predictions.loc[predictions["model"].eq(BASE_MODEL)]
    base_metrics = evaluate_predictions(base_predictions)
    base_folds = fold_results.loc[
        fold_results["model"].eq(BASE_MODEL)
    ].set_index("fold")
    rows = []
    for model in (BASE_MODEL, DYNAMIC_MODEL):
        frame = predictions.loc[predictions["model"].eq(model)]
        metrics = evaluate_predictions(frame)
        folds = fold_results.loc[fold_results["model"].eq(model)].set_index("fold")
        common = folds.index.intersection(base_folds.index)
        rank_mask = (
            folds.loc[common, "ranking_score"].notna()
            & base_folds.loc[common, "ranking_score"].notna()
        )
        rank_index = common[rank_mask]
        rank_safe = (
            (
                folds.loc[rank_index, "ranking_score"]
                >= base_folds.loc[rank_index, "ranking_score"] - RANK_TOLERANCE
            )
            & (
                folds.loc[rank_index, "pairwise_accuracy"]
                >= base_folds.loc[rank_index, "pairwise_accuracy"]
                - RANK_TOLERANCE
            )
        )
        rank_both = (
            (
                folds.loc[rank_index, "ranking_score"]
                > base_folds.loc[rank_index, "ranking_score"] + RANK_TOLERANCE
            )
            & (
                folds.loc[rank_index, "pairwise_accuracy"]
                > base_folds.loc[rank_index, "pairwise_accuracy"]
                + RANK_TOLERANCE
            )
        )
        brier_wins = int(
            (
                folds.loc[common, "brier_1x2"]
                < base_folds.loc[common, "brier_1x2"] - 1e-15
            ).sum()
        )
        log_wins = int(
            (
                folds.loc[common, "log_loss_1x2"]
                < base_folds.loc[common, "log_loss_1x2"] - 1e-15
            ).sum()
        )
        rows.append(
            {
                "model": model,
                **metrics,
                "brier_delta_vs_base": (
                    float(metrics["brier_1x2"])
                    - float(base_metrics["brier_1x2"])
                ),
                "log_loss_delta_vs_base": (
                    float(metrics["log_loss_1x2"])
                    - float(base_metrics["log_loss_1x2"])
                ),
                "accuracy_delta_vs_base": (
                    float(metrics["accuracy_1x2"])
                    - float(base_metrics["accuracy_1x2"])
                ),
                "ece_delta_vs_base": (
                    float(metrics["multiclass_ece"])
                    - float(base_metrics["multiclass_ece"])
                ),
                "brier_fold_wins": brier_wins,
                "log_loss_fold_wins": log_wins,
                "ranking_evaluable_folds": len(rank_index),
                "ranking_no_regression_folds": int(rank_safe.sum()),
                "ranking_both_improved_folds": int(rank_both.sum()),
                "maximum_match_k": float(folds["maximum_match_k"].max()),
                "mean_match_k": float(
                    np.average(
                        folds["mean_match_k"],
                        weights=folds["matches"],
                    )
                ),
                "maximum_abs_match_delta": float(
                    folds["maximum_abs_match_delta"].max()
                ),
                "maximum_abs_rating_change": float(
                    folds["maximum_abs_rating_change"].max()
                ),
                "minimum_start_end_rank_correlation": float(
                    folds["minimum_start_end_rank_correlation"].min()
                ),
                "maximum_total_elo_error": float(
                    folds["maximum_total_elo_error"].max()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_uncertainty(
    predictions: pd.DataFrame,
    *,
    bootstrap_samples: int,
) -> pd.DataFrame:
    rows = []
    segments = [("ALL", predictions)]
    segments.extend(
        (str(competition), frame)
        for competition, frame in predictions.groupby("competition", sort=True)
    )
    for segment, frame in segments:
        for metric in ("brier_1x2", "log_loss_1x2"):
            base = frame.loc[
                frame["model"].eq(BASE_MODEL),
                [
                    "match_id",
                    "season",
                    "kickoff_utc",
                    "tie_id",
                    "home_team_id",
                    "away_team_id",
                    metric,
                ],
            ].rename(columns={metric: "base_loss"})
            challenger = frame.loc[
                frame["model"].eq(DYNAMIC_MODEL),
                ["match_id", metric],
            ].rename(columns={metric: "challenger_loss"})
            paired = base.merge(
                challenger,
                on="match_id",
                validate="one_to_one",
            )
            paired["loss_difference"] = (
                paired["challenger_loss"] - paired["base_loss"]
            )
            uncertainty = dependency_robust_loss_difference_ci(
                paired,
                bootstrap_samples=bootstrap_samples,
                seed=20260725 + len(rows) * 1009,
            )
            uncertainty.insert(0, "segment", segment)
            uncertainty.insert(1, "metric", metric)
            rows.append(uncertainty)
    return pd.concat(rows, ignore_index=True)


def build_conservation_summary(
    season_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for model, frame in season_metrics.groupby("model", sort=False):
        model_predictions = predictions.loc[predictions["model"].eq(model)]
        rows.append(
            {
                "model": model,
                "seasons": frame["season"].nunique(),
                "matches": len(model_predictions),
                "maximum_match_pair_error": float(
                    frame["match_pair_zero_sum_error"].max()
                ),
                "maximum_season_total_error": float(
                    frame["season_total_elo_error"].max()
                ),
                "maximum_match_k": float(model_predictions["match_k"].max()),
                "maximum_abs_power_delta": float(
                    model_predictions["power_delta"].abs().max()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_k_distribution(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, frame in predictions.groupby("model", sort=False):
        rows.append(
            {
                "model": model,
                "matches": len(frame),
                "match_k_min": float(frame["match_k"].min()),
                "match_k_mean": float(frame["match_k"].mean()),
                "match_k_median": float(frame["match_k"].median()),
                "match_k_p90": float(frame["match_k"].quantile(0.90)),
                "match_k_p95": float(frame["match_k"].quantile(0.95)),
                "match_k_max": float(frame["match_k"].max()),
                "uncertainty_mean": float(
                    pd.concat(
                        [
                            frame["home_uncertainty"],
                            frame["away_uncertainty"],
                        ],
                        ignore_index=True,
                    ).mean()
                ),
                "max_abs_power_delta": float(frame["power_delta"].abs().max()),
            }
        )
    return pd.DataFrame(rows)


def add_evidence_bands(predictions: pd.DataFrame) -> pd.DataFrame:
    result = predictions.copy()
    result["mean_exposure"] = (
        result["home_exposure"] + result["away_exposure"]
    ) / 2.0
    result["evidence_band"] = pd.cut(
        result["mean_exposure"],
        bins=(-1.0, 0.40, 0.75, 1.01),
        labels=("LOW", "MEDIUM", "HIGH"),
    ).astype(str)
    if result["evidence_band"].isna().any():
        raise ValueError("Every Dynamic K prediction requires an evidence band")
    return result


def build_segment_k_distribution(
    predictions: pd.DataFrame,
    segment: str,
) -> pd.DataFrame:
    if segment not in predictions:
        raise ValueError(f"Unknown K distribution segment: {segment}")
    rows = []
    for (model, value), frame in predictions.groupby(
        ["model", segment],
        sort=True,
    ):
        rows.append(
            {
                "model": model,
                segment: value,
                "matches": len(frame),
                "match_k_mean": float(frame["match_k"].mean()),
                "match_k_p95": float(frame["match_k"].quantile(0.95)),
                "match_k_max": float(frame["match_k"].max()),
                "team_uncertainty_mean": float(
                    pd.concat(
                        [
                            frame["home_uncertainty"],
                            frame["away_uncertainty"],
                        ],
                        ignore_index=True,
                    ).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def dynamic_k_decision(
    comparison: pd.DataFrame,
    uncertainty: pd.DataFrame,
    fold_results: pd.DataFrame,
    conservation: pd.DataFrame,
) -> tuple[str, dict[str, object]]:
    summary = comparison.loc[
        comparison["model"].eq(DYNAMIC_MODEL)
    ].iloc[0]
    envelope = uncertainty.loc[
        uncertainty["segment"].eq("ALL")
        & uncertainty["method"].eq("conservative_envelope")
    ].set_index("metric")
    competition_envelope = uncertainty.loc[
        uncertainty["segment"].isin(("UCL", "UEL", "UECL"))
        & uncertainty["method"].eq("conservative_envelope")
    ]
    point_improvement = bool(
        summary["brier_delta_vs_base"] < 0.0
        and summary["log_loss_delta_vs_base"] < 0.0
    )
    reliable_improvement = bool(
        envelope.loc["brier_1x2", "ci_95_upper"] < 0.0
        and envelope.loc["log_loss_1x2", "ci_95_upper"] < 0.0
    )
    no_competition_harm = bool(
        not competition_envelope["reliable_harm"].astype(bool).any()
    )
    practical = bool(
        summary["brier_delta_vs_base"] <= -PRACTICAL_BRIER_THRESHOLD
    )
    ranking_safe = bool(
        summary["ranking_evaluable_folds"] >= 5
        and summary["ranking_no_regression_folds"]
        == summary["ranking_evaluable_folds"]
    )
    ranking_improves = bool(
        summary["ranking_both_improved_folds"] >= 4
    )
    movement_safe = bool(
        summary["minimum_start_end_rank_correlation"]
        >= RANK_CORRELATION_FLOOR
        and summary["maximum_abs_rating_change"]
        <= MAX_RATING_MOVE_GUARDRAIL
    )
    zero_sum = bool(
        conservation["maximum_match_pair_error"].max() <= 1e-9
        and conservation["maximum_season_total_error"].max() <= 1e-9
    )
    if (
        point_improvement
        and reliable_improvement
        and no_competition_harm
        and practical
        and ranking_safe
        and ranking_improves
        and movement_safe
        and zero_sum
    ):
        decision = "ACTIVATE_DYNAMIC_K"
    elif (
        point_improvement
        and no_competition_harm
        and movement_safe
        and zero_sum
        and (
            reliable_improvement
            or summary["brier_delta_vs_base"] <= -SHADOW_BRIER_THRESHOLD
        )
    ):
        decision = "SHADOW_DYNAMIC_K"
    else:
        decision = "KEEP_FIXED_K"
    guardrails = {
        "point_improvement_both_losses": point_improvement,
        "reliable_improvement_both_losses": reliable_improvement,
        "no_competition_reliable_harm": no_competition_harm,
        "practical_brier_improvement": practical,
        "brier_fold_wins": int(summary["brier_fold_wins"]),
        "log_loss_fold_wins": int(summary["log_loss_fold_wins"]),
        "ranking_evaluable_folds": int(summary["ranking_evaluable_folds"]),
        "ranking_no_regression_folds": int(
            summary["ranking_no_regression_folds"]
        ),
        "ranking_both_improved_folds": int(
            summary["ranking_both_improved_folds"]
        ),
        "ranking_safe": ranking_safe,
        "ranking_improves": ranking_improves,
        "movement_safe": movement_safe,
        "zero_sum": zero_sum,
    }
    return decision, guardrails


def write_manifest(
    path: Path,
    production: dict[str, object],
    full_candidate: DynamicKConfig,
    active_candidate: DynamicKConfig,
    decision: str,
    guardrails: dict[str, object],
    comparison: pd.DataFrame,
    uncertainty: pd.DataFrame,
) -> None:
    summary = comparison.loc[
        comparison["model"].eq(DYNAMIC_MODEL)
    ].iloc[0]
    envelope = uncertainty.loc[
        uncertainty["segment"].eq("ALL")
        & uncertainty["method"].eq("conservative_envelope")
    ]
    payload = {
        "model_version": production["model_version"],
        "ao_first_elo_changed": False,
        "baseline": {
            "dynamic_core": production["dynamic_core"],
            "goal_margin": production["goal_margin"],
            "one_x_two_probability": production["one_x_two_probability"],
        },
        "uncertainty_formula": {
            "exposure": "1 - european_exposure",
            "match_evidence": "exp(-prior_matches / match_evidence_scale)",
            "inactivity": "min(days_since_last_match / inactivity_days, 1)",
            "combined": "equal-weight mean of three components",
        },
        "full_development_candidate": _config_payload(full_candidate),
        "active_dynamic_k": _config_payload(active_candidate),
        "decision": decision,
        "guardrails": guardrails,
        "nested_oos_metrics": {
            "brier_delta_vs_base": float(summary["brier_delta_vs_base"]),
            "log_loss_delta_vs_base": float(
                summary["log_loss_delta_vs_base"]
            ),
            "accuracy_delta_vs_base": float(
                summary["accuracy_delta_vs_base"]
            ),
            "ece_delta_vs_base": float(summary["ece_delta_vs_base"]),
        },
        "dependency_uncertainty": envelope[
            [
                "metric",
                "mean_difference",
                "ci_95_lower",
                "ci_95_upper",
                "reliable_improvement",
                "reliable_harm",
            ]
        ].to_dict(orient="records"),
        "development_data_through": "2025/26",
        "prospective_holdout": "2026/27 league phase and later",
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _config_payload(config: DynamicKConfig) -> dict[str, object]:
    return {
        "active": config.lambda_factor > 0.0,
        "config_id": config.config_id,
        "lambda_factor": config.lambda_factor,
        "max_k_multiplier": config.max_k_multiplier,
        "aggregation": config.aggregation,
        "inactivity_days": config.inactivity_days,
        "match_evidence_scale": config.match_evidence_scale,
    }


def write_report(
    path: Path,
    seasons: tuple[str, ...],
    candidates: tuple[DynamicKConfig, ...],
    data_quality: dict[str, object],
    selections: pd.DataFrame,
    comparison: pd.DataFrame,
    uncertainty: pd.DataFrame,
    competition_summary: pd.DataFrame,
    match_band_summary: pd.DataFrame,
    evidence_band_summary: pd.DataFrame,
    k_distribution: pd.DataFrame,
    competition_k_distribution: pd.DataFrame,
    conservation: pd.DataFrame,
    full_candidate: DynamicKConfig,
    active_candidate: DynamicKConfig,
    decision: str,
    guardrails: dict[str, object],
) -> None:
    dynamic = comparison.loc[
        comparison["model"].eq(DYNAMIC_MODEL)
    ].iloc[0]
    envelope = uncertainty.loc[
        uncertainty["method"].eq("conservative_envelope")
    ]
    lines = [
        "# Dinamik K Backtest Raporu",
        "",
        "## Kapsam",
        "",
        f"- Sezonlar: `{seasons[0]}-{seasons[-1]}`.",
        f"- Aday sayısı: `{len(candidates)}` tekil config.",
        "- Outer fold: `6` expanding nested walk-forward.",
        f"- Maç: `{data_quality['matches']}` exact-date tarihsel olay; "
        f"unseen karşılaştırma `{int(dynamic['matches'])}` maç.",
        "- AO First Elo, Scale, H, kontrollü gol farkı ve 1X2 mapping değiştirilmedi.",
        "- Baseline: `K=103.9809863339`, `alpha=0.10`, `tau=300`, `GD cap=4`.",
        "",
        "## Belirsizlik Formülü",
        "",
        "```text",
        "U_exposure   = 1 - European Exposure",
        "U_matches    = exp(-Prior Matches / Match Evidence Scale)",
        "U_inactivity = min(Days Since Last Match / Inactivity Days, 1)",
        "U_team       = (U_exposure + U_matches + U_inactivity) / 3",
        "K_team       = min(K_base x Max Multiplier, K_base x (1 + lambda x U_team))",
        "K_match      = arithmetic veya geometric aggregate(K_home, K_away)",
        "```",
        "",
        "## Veri Kalitesi",
        "",
        f"- Exposure aralığı: `{data_quality['exposure_min']:.3f}` - "
        f"`{data_quality['exposure_max']:.3f}`; medyan "
        f"`{data_quality['exposure_median']:.3f}`.",
        f"- Önceki maç sayısı medyanı: "
        f"`{data_quality['prior_matches_median']:.1f}`.",
        f"- Önceki exact date bilinmeyen takım-maç tarafı: "
        f"`{data_quality['missing_previous_exact_date_sides']}` "
        f"(`{100 * data_quality['missing_previous_exact_date_rate']:.2f}%`).",
        "- Bilinmeyen önceki tarih için inactivity uncertainty `1.0` kullanıldı.",
        "- Tüm uncertainty alanları maç başlangıcından önceki veriden üretildi.",
        "",
        "## Nested Fold Seçimleri",
        "",
        _markdown_table(
            selections[
                [
                    "fold",
                    "test_season",
                    "selected_lambda",
                    "selected_max_k_multiplier",
                    "selected_aggregation",
                    "selected_inactivity_days",
                    "selected_match_evidence_scale",
                ]
            ]
        ),
        "",
        "## Pooled Unseen Sonuç",
        "",
        _markdown_table(
            comparison[
                [
                    "model",
                    "matches",
                    "brier_1x2",
                    "brier_delta_vs_base",
                    "log_loss_1x2",
                    "log_loss_delta_vs_base",
                    "accuracy_1x2",
                    "multiclass_ece",
                    "brier_fold_wins",
                    "log_loss_fold_wins",
                ]
            ]
        ),
        "",
        "## Ranking-First Sonuç",
        "",
        f"- Evaluable forward fold: `{guardrails['ranking_evaluable_folds']}`.",
        f"- İki ranking metriğinde gerilemeyen fold: "
        f"`{guardrails['ranking_no_regression_folds']}`.",
        f"- İki ranking metriğinde birlikte iyileşen fold: "
        f"`{guardrails['ranking_both_improved_folds']}`.",
        f"- Minimum sezon başlangıç/son rank korelasyonu: "
        f"`{dynamic['minimum_start_end_rank_correlation']:.6f}`.",
        "",
        "## UCL / UEL / UECL",
        "",
        _markdown_table(
            competition_summary[
                [
                    "model",
                    "competition",
                    "matches",
                    "brier_delta_vs_base",
                    "log_loss_delta_vs_base",
                ]
            ]
        ),
        "",
        "## Favori Bantları",
        "",
        _markdown_table(
            match_band_summary[
                [
                    "model",
                    "match_band",
                    "matches",
                    "brier_delta_vs_base",
                    "log_loss_delta_vs_base",
                ]
            ]
        ),
        "",
        "## Exposure Kanıt Bantları",
        "",
        _markdown_table(
            evidence_band_summary[
                [
                    "model",
                    "evidence_band",
                    "matches",
                    "brier_delta_vs_base",
                    "log_loss_delta_vs_base",
                ]
            ]
        ),
        "",
        "## Turnuva Bazında K Dağılımı",
        "",
        _markdown_table(competition_k_distribution),
        "",
        "## Dependency-Sensitive Belirsizlik",
        "",
        _markdown_table(
            envelope[
                [
                    "segment",
                    "metric",
                    "mean_difference",
                    "ci_95_lower",
                    "ci_95_upper",
                    "reliable_improvement",
                    "reliable_harm",
                ]
            ]
        ),
        "",
        "## K ve Hareket Dağılımı",
        "",
        _markdown_table(k_distribution),
        "",
        "## Elo Korunumu",
        "",
        _markdown_table(conservation),
        "",
        "## Model Kararı",
        "",
        f"- Karar: **`{decision}`**.",
        f"- Full-development adayı: `{full_candidate.key}`.",
        f"- Aktif config: `{active_candidate.key}`.",
        f"- Brier farkı: `{dynamic['brier_delta_vs_base']:+.8f}`.",
        f"- Log-loss farkı: `{dynamic['log_loss_delta_vs_base']:+.8f}`.",
        f"- Maksimum K: `{dynamic['maximum_match_k']:.6f}`.",
        f"- Maksimum mutlak maç Delta: "
        f"`{dynamic['maximum_abs_match_delta']:.6f}`.",
        f"- Maksimum toplam Elo hatası: "
        f"`{dynamic['maximum_total_elo_error']:.3e}`.",
        "",
        "Karar nested unseen sonuç, ranking guardrail, turnuva segmentleri, "
        "dependency-sensitive belirsizlik ve sıfır-toplam kontrolü birlikte "
        "değerlendirilerek verilmiştir. Fixed-candidate OOS tablosu duyarlılık "
        "analizidir; tek başına production terfi kanıtı değildir.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.8f}"
        )
    headers = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
