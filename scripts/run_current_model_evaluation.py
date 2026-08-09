from __future__ import annotations

"""Audit the active AO European Elo contract without mutating production files."""

import argparse
import hashlib
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
from ao_elo.evaluation import (  # noqa: E402
    dependency_robust_loss_difference_ci,
    schedule_adjusted_team_performance,
)
from ao_elo.tournament_bonus import (  # noqa: E402
    ELIGIBLE_PROGRESSION_STAGES,
    FixedTournamentBonusConfig,
    apply_tournament_progress_bonus,
)
from ao_elo.xg_live import (  # noqa: E402
    XGBlendConfig,
    XGPerformanceBonusConfig,
    update_match_elo_with_xg,
)
from scripts.run_controlled_goal_progression_backtest import (  # noqa: E402
    ControlledSeasonData,
    prepare_controlled_data,
)
from scripts.run_dynamic_core_calibration import expanding_folds  # noqa: E402
from scripts.run_final_robustness import (  # noqa: E402
    load_team_season_identity,
    summarize_ranking,
)
from scripts.run_opponent_quintile_backtest import load_production_baseline  # noqa: E402
from scripts.run_stage_weighted_progression_backtest import (  # noqa: E402
    DOMESTIC_ADJUSTMENTS,
    DYNAMIC_MANIFEST,
    EVENTS_PATH,
    PRODUCTION_CONTRACT,
    STATIC_DATA_ROOT,
    XG_DATA,
    aggregate_ranking,
    load_domestic_adjustments,
    load_xg_map,
    probability_vector,
    same_season_ranking,
    validate_production_contract,
)
from scripts.run_v2_achievement_reserve_calibration import load_reserve_data  # noqa: E402
from scripts.run_v2_evaluation_upgrade import read_events  # noqa: E402


FINAL_CANDIDATE_CONTRACT = ROOT / "contracts" / "ao_european_elo_v2_final_candidate.json"
SURPRISE_FEATURES = ROOT / "output" / "domestic_surprise_variance_backtest_2018_2026" / "domestic_surprise_features.csv"
OUTPUT_ROOT = ROOT / "output" / "current_model_evaluation_2018_2026"

CURRENT = "CURRENT_PRODUCTION"
REFERENCE = "REFERENCE_CORE_NO_ACTIVE_EXTRAS"
NO_SURPRISE = "ABLATION_NO_DOMESTIC_SURPRISE"
NO_GOAL_MARGIN = "ABLATION_NO_GOAL_MARGIN"
NO_XG = "ABLATION_NO_XG"
NO_PROGRESSION = "ABLATION_NO_PROGRESSION"
ARMS_IN_ORDER = (CURRENT, REFERENCE, NO_SURPRISE, NO_GOAL_MARGIN, NO_XG, NO_PROGRESSION)
COMPETITION_INDEX = {"UCL": 0, "UEL": 1, "UECL": 2}


@dataclass(frozen=True)
class EvaluationArm:
    name: str
    domestic_surprise: bool
    goal_margin: bool
    xg: bool
    progression: bool


@dataclass
class ArmEvaluation:
    predictions: pd.DataFrame
    end_ratings: pd.DataFrame
    season_metrics: pd.DataFrame
    same_season_ranking: pd.DataFrame
    bonus_events: pd.DataFrame


def evaluation_arms() -> tuple[EvaluationArm, ...]:
    return (
        EvaluationArm(CURRENT, True, True, True, True),
        EvaluationArm(REFERENCE, False, False, False, False),
        EvaluationArm(NO_SURPRISE, False, True, True, True),
        EvaluationArm(NO_GOAL_MARGIN, True, False, True, True),
        EvaluationArm(NO_XG, True, True, False, True),
        EvaluationArm(NO_PROGRESSION, True, True, True, False),
    )


def evaluate_arm(
    datasets: tuple[ControlledSeasonData, ...],
    arm: EvaluationArm,
    *,
    core,
    parameters: dict[str, float | int],
    current_domestic: dict[tuple[str, int], float],
    baseline_domestic: dict[tuple[str, int], float],
    xg_map: dict[str, tuple[float, float]],
    target: pd.DataFrame,
) -> ArmEvaluation:
    predictions: list[dict[str, object]] = []
    end_rows: list[dict[str, object]] = []
    season_rows: list[dict[str, object]] = []
    bonus_rows: list[dict[str, object]] = []
    fixed_bonus = FixedTournamentBonusConfig(12.0)
    xg_blend = XGBlendConfig(0.0, 1.0)
    xg_bonus = XGPerformanceBonusConfig(
        float(parameters["xg_ratio"]),
        float(parameters["xg_scale"]),
        float(parameters["xg_floor"]),
    )
    rating_map = current_domestic if arm.domestic_surprise else baseline_domestic

    for data in datasets:
        reserve = data.reserve
        goal = reserve.goal
        season_core = goal.carry.core
        active_ids = season_core.active_team_ids
        power = season_core.initial_ratings.copy()
        for team_id in active_ids:
            power[int(team_id)] = rating_map[(data.season, int(team_id))]
        initial = power.copy()
        bonus = np.zeros((len(power), len(COMPETITION_INDEX)), dtype=float)
        processed_ties: set[str] = set()
        max_match_zero_sum = 0.0
        max_match_delta = 0.0
        max_cap_error = 0.0

        for index, (home_raw, away_raw, neutral_raw, competition_raw) in enumerate(
            zip(
                season_core.home_team_ids,
                season_core.away_team_ids,
                season_core.neutral_flags,
                season_core.competitions,
                strict=True,
            )
        ):
            home_id, away_id = int(home_raw), int(away_raw)
            competition = str(competition_raw)
            match_id = str(season_core.match_ids[index])
            stage = str(reserve.stages[index])
            tie_value = reserve.tie_ids[index]
            tie_id = None if tie_value is None else str(tie_value)
            home_live_pre = float(power[home_id] + bonus[home_id].sum())
            away_live_pre = float(power[away_id] + bonus[away_id].sum())
            penalty = bool(goal.penalty_flags[index])
            xg_values = xg_map.get(match_id) if arm.xg else None
            update = update_match_elo_with_xg(
                home_live_pre,
                away_live_pre,
                int(data.home_goals[index]),
                int(data.away_goals[index]),
                k_factor=core.k_factor,
                elo_scale=core.elo_scale,
                home_advantage=core.home_advantage,
                is_neutral=bool(neutral_raw),
                decided_on_penalties=penalty,
                goal_difference_enabled=arm.goal_margin,
                goal_alpha=float(parameters["goal_alpha"]) if arm.goal_margin else 0.0,
                goal_tau=float(parameters["goal_tau"]),
                goal_difference_cap=int(parameters["goal_cap"]),
                xg_config=xg_blend,
                xg_home=None if xg_values is None else xg_values[0],
                xg_away=None if xg_values is None else xg_values[1],
                xg_performance_bonus_config=xg_bonus if arm.xg else None,
            )
            probabilities = probability_vector(
                update.expected_home_score,
                float(parameters["draw_at_even"]),
                float(parameters["draw_shape"]),
            )
            observed = (
                0 if update.actual_home_score == 1.0
                else 1 if update.actual_home_score == 0.5
                else 2
            )
            brier = float(np.square(probabilities - np.eye(3)[observed]).sum())
            log_loss = -math.log(max(float(probabilities[observed]), 1e-15))
            power[home_id] += update.power_delta
            power[away_id] -= update.power_delta
            max_match_delta = max(max_match_delta, abs(update.power_delta))
            max_match_zero_sum = max(max_match_zero_sum, abs(update.zero_sum_error))

            progression_added = 0.0
            progression_winner = -1
            if (
                arm.progression
                and stage in ELIGIBLE_PROGRESSION_STAGES
                and bool(reserve.tie_decider_flags[index])
            ):
                if tie_id is None:
                    raise ValueError(f"{data.season}/{match_id}: eligible tie lacks tie_id")
                progression_winner = int(reserve.advanced_team_ids[index])
                if progression_winner not in (home_id, away_id):
                    raise ValueError(f"{data.season}/{tie_id}: invalid advancing team")
                component = COMPETITION_INDEX[competition]
                update_bonus = apply_tournament_progress_bonus(
                    float(bonus[progression_winner, component]),
                    competition,
                    stage,
                    tie_id,
                    processed_ties,
                    fixed_bonus,
                )
                bonus[progression_winner, component] = update_bonus.bonus_post
                progression_added = update_bonus.applied_bonus
                max_cap_error = max(
                    max_cap_error,
                    max(0.0, update_bonus.bonus_post - update_bonus.competition_cap),
                )
                bonus_rows.append(
                    {
                        "model": arm.name,
                        "season": data.season,
                        "match_id": match_id,
                        "tie_id": tie_id,
                        "competition": competition,
                        "stage": stage,
                        "winner_team_id": progression_winner,
                        "decided_on_penalties": penalty,
                        "bonus_pre": update_bonus.bonus_pre,
                        "bonus_added": update_bonus.applied_bonus,
                        "bonus_post": update_bonus.bonus_post,
                        "competition_cap": update_bonus.competition_cap,
                    }
                )

            predictions.append(
                {
                    "model": arm.name,
                    "match_id": match_id,
                    "season": data.season,
                    "kickoff_utc": pd.Timestamp(data.kickoff_utc[index]),
                    "competition": competition,
                    "stage": stage,
                    "tie_id": tie_id,
                    "home_team_id": home_id,
                    "away_team_id": away_id,
                    "home_goals": int(data.home_goals[index]),
                    "away_goals": int(data.away_goals[index]),
                    "decided_on_penalties": penalty,
                    "is_neutral": bool(neutral_raw),
                    "xg_applied": xg_values is not None and not penalty,
                    "home_live_pre": home_live_pre,
                    "away_live_pre": away_live_pre,
                    "expected_home_score": update.expected_home_score,
                    "home_probability": probabilities[0],
                    "draw_probability": probabilities[1],
                    "away_probability": probabilities[2],
                    "actual_class": observed,
                    "predicted_class": int(np.argmax(probabilities)),
                    "brier_1x2": brier,
                    "log_loss_1x2": log_loss,
                    "power_delta": update.power_delta,
                    "goal_multiplier": update.goal_difference_multiplier,
                    "xg_performance_adjustment": update.xg_performance_adjustment,
                    "progression_bonus_added": progression_added,
                    "progression_winner_team_id": progression_winner,
                    "zero_sum_error": update.zero_sum_error,
                }
            )

        end_power = power[active_ids]
        end_bonus = bonus[active_ids].sum(axis=1)
        power_total_error = abs(float(end_power.sum()) - float(initial[active_ids].sum()))
        season_rows.append(
            {
                "model": arm.name,
                "season": data.season,
                "matches": len(season_core.match_ids),
                "teams": len(active_ids),
                "start_rating_min": float(initial[active_ids].min()),
                "start_rating_max": float(initial[active_ids].max()),
                "end_power_min": float(end_power.min()),
                "end_power_max": float(end_power.max()),
                "end_live_min": float((end_power + end_bonus).min()),
                "end_live_max": float((end_power + end_bonus).max()),
                "maximum_abs_match_delta": max_match_delta,
                "maximum_match_zero_sum_error": max_match_zero_sum,
                "season_power_conservation_error": power_total_error,
                "maximum_bonus_cap_error": max_cap_error,
                "total_progression_bonus": float(end_bonus.sum()),
                "maximum_team_progression_bonus": float(end_bonus.max()),
            }
        )
        for team_id in active_ids:
            end_rows.append(
                {
                    "model": arm.name,
                    "season": data.season,
                    "team_id": int(team_id),
                    "initial_rating": float(initial[int(team_id)]),
                    "end_power_rating": float(power[int(team_id)]),
                    "end_progression_bonus": float(bonus[int(team_id)].sum()),
                    "end_live_rating": float(power[int(team_id)] + bonus[int(team_id)].sum()),
                }
            )

    prediction_frame = pd.DataFrame(predictions)
    end_frame = pd.DataFrame(end_rows)
    ranking = same_season_ranking(
        end_frame,
        target,
        {data.season for data in datasets},
    )
    ranking["model"] = arm.name
    return ArmEvaluation(
        predictions=prediction_frame,
        end_ratings=end_frame,
        season_metrics=pd.DataFrame(season_rows),
        same_season_ranking=ranking,
        bonus_events=pd.DataFrame(bonus_rows),
    )


def prediction_summary(frame: pd.DataFrame) -> dict[str, float | int]:
    return {
        "matches": int(len(frame)),
        "brier_1x2": float(frame["brier_1x2"].mean()),
        "log_loss_1x2": float(frame["log_loss_1x2"].mean()),
        "accuracy_1x2": float((frame["actual_class"] == frame["predicted_class"]).mean()),
    }


def build_model_summary(
    evaluations: dict[str, ArmEvaluation],
    evaluation_seasons: set[str],
) -> pd.DataFrame:
    rows = []
    for arm_name in ARMS_IN_ORDER:
        evaluation = evaluations[arm_name]
        predictions = evaluation.predictions.loc[
            evaluation.predictions["season"].isin(evaluation_seasons)
        ]
        ranking = aggregate_ranking(
            evaluation.same_season_ranking.loc[
                evaluation.same_season_ranking["season"].isin(evaluation_seasons)
            ]
        )
        all_ranking = ranking.loc[ranking["competition"].eq("ALL")].iloc[0]
        rows.append(
            {
                "model": arm_name,
                **prediction_summary(predictions),
                "same_season_spearman": float(all_ranking["ranking_score"]),
                "same_season_pairwise_accuracy": float(all_ranking["pairwise_accuracy"]),
                "xg_applied_matches": int(predictions["xg_applied"].sum()),
                "total_progression_bonus": float(
                    evaluation.season_metrics.loc[
                        evaluation.season_metrics["season"].isin(evaluation_seasons),
                        "total_progression_bonus",
                    ].sum()
                ),
                "maximum_abs_match_delta": float(
                    evaluation.season_metrics.loc[
                        evaluation.season_metrics["season"].isin(evaluation_seasons),
                        "maximum_abs_match_delta",
                    ].max()
                ),
            }
        )
    result = pd.DataFrame(rows)
    reference = result.loc[result["model"].eq(REFERENCE)].iloc[0]
    for metric in (
        "brier_1x2",
        "log_loss_1x2",
        "accuracy_1x2",
        "same_season_spearman",
        "same_season_pairwise_accuracy",
    ):
        result[f"delta_vs_reference_{metric}"] = result[metric] - reference[metric]
    return result


def build_fold_summary(
    evaluations: dict[str, ArmEvaluation],
    folds,
) -> pd.DataFrame:
    rows = []
    for fold, (_, test_season) in enumerate(folds, start=1):
        for arm_name in ARMS_IN_ORDER:
            evaluation = evaluations[arm_name]
            predictions = evaluation.predictions.loc[
                evaluation.predictions["season"].eq(test_season)
            ]
            ranking = aggregate_ranking(
                evaluation.same_season_ranking.loc[
                    evaluation.same_season_ranking["season"].eq(test_season)
                ]
            )
            all_rank = ranking.loc[ranking["competition"].eq("ALL")].iloc[0]
            rows.append(
                {
                    "fold": fold,
                    "test_season": test_season,
                    "model": arm_name,
                    **prediction_summary(predictions),
                    "same_season_spearman": float(all_rank["ranking_score"]),
                    "same_season_pairwise_accuracy": float(all_rank["pairwise_accuracy"]),
                }
            )
    result = pd.DataFrame(rows)
    reference = result.loc[result["model"].eq(REFERENCE)].set_index("fold")
    for metric in (
        "brier_1x2",
        "log_loss_1x2",
        "accuracy_1x2",
        "same_season_spearman",
        "same_season_pairwise_accuracy",
    ):
        result[f"delta_vs_reference_{metric}"] = result.apply(
            lambda row: row[metric] - reference.loc[row["fold"], metric], axis=1
        )
    return result


def build_competition_summary(
    evaluations: dict[str, ArmEvaluation],
    evaluation_seasons: set[str],
) -> pd.DataFrame:
    rows = []
    for arm_name in ARMS_IN_ORDER:
        evaluation = evaluations[arm_name]
        predictions = evaluation.predictions.loc[
            evaluation.predictions["season"].isin(evaluation_seasons)
        ]
        ranking = aggregate_ranking(
            evaluation.same_season_ranking.loc[
                evaluation.same_season_ranking["season"].isin(evaluation_seasons)
            ]
        ).set_index("competition")
        for competition, frame in predictions.groupby("competition", sort=True):
            rows.append(
                {
                    "model": arm_name,
                    "competition": competition,
                    **prediction_summary(frame),
                    "same_season_spearman": float(ranking.loc[competition, "ranking_score"]),
                    "same_season_pairwise_accuracy": float(ranking.loc[competition, "pairwise_accuracy"]),
                }
            )
    result = pd.DataFrame(rows)
    reference = result.loc[result["model"].eq(REFERENCE)].set_index("competition")
    for metric in (
        "brier_1x2",
        "log_loss_1x2",
        "accuracy_1x2",
        "same_season_spearman",
        "same_season_pairwise_accuracy",
    ):
        result[f"delta_vs_reference_{metric}"] = result.apply(
            lambda row: row[metric] - reference.loc[row["competition"], metric], axis=1
        )
    return result


def build_forward_ranking(
    evaluations: dict[str, ArmEvaluation],
    target: pd.DataFrame,
    identity: pd.DataFrame,
    seasons: tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    allowed_targets = set(seasons[3:])
    for arm_name in ARMS_IN_ORDER:
        ranking = summarize_ranking(
            evaluations[arm_name].end_ratings,
            target,
            allowed_target_seasons=allowed_targets,
            identity=identity,
        )
        ranking["model"] = arm_name
        rows.append(ranking)
    return pd.concat(rows, ignore_index=True)


def build_dependency_uncertainty(
    evaluations: dict[str, ArmEvaluation],
    evaluation_seasons: set[str],
    *,
    bootstrap_samples: int,
) -> pd.DataFrame:
    rows = []
    comparisons = ((CURRENT, REFERENCE),) + tuple(
        (arm_name, CURRENT)
        for arm_name in (NO_SURPRISE, NO_GOAL_MARGIN, NO_XG, NO_PROGRESSION)
    )
    for candidate_name, baseline_name in comparisons:
        candidate = evaluations[candidate_name].predictions.loc[
            evaluations[candidate_name].predictions["season"].isin(evaluation_seasons)
        ]
        baseline = evaluations[baseline_name].predictions.loc[
            evaluations[baseline_name].predictions["season"].isin(evaluation_seasons)
        ]
        for competition in ("ALL", "UCL", "UEL", "UECL"):
            left = candidate if competition == "ALL" else candidate.loc[
                candidate["competition"].eq(competition)
            ]
            right = baseline if competition == "ALL" else baseline.loc[
                baseline["competition"].eq(competition)
            ]
            paired = left.merge(
                right[["match_id", "brier_1x2", "log_loss_1x2"]],
                on="match_id",
                suffixes=("_candidate", "_baseline"),
                validate="one_to_one",
            )
            for metric in ("brier_1x2", "log_loss_1x2"):
                sample = paired[
                    [
                        "season",
                        "match_id",
                        "home_team_id",
                        "away_team_id",
                        "kickoff_utc",
                        "tie_id",
                    ]
                ].copy()
                sample["loss_difference"] = (
                    paired[f"{metric}_candidate"] - paired[f"{metric}_baseline"]
                )
                uncertainty = dependency_robust_loss_difference_ci(
                    sample,
                    bootstrap_samples=bootstrap_samples,
                )
                uncertainty.insert(0, "candidate_model", candidate_name)
                uncertainty.insert(1, "baseline_model", baseline_name)
                uncertainty.insert(2, "competition", competition)
                uncertainty.insert(3, "metric", metric)
                rows.append(uncertainty)
    return pd.concat(rows, ignore_index=True)


def build_initial_impact(
    adjustments_path: Path,
    features_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    adjustments = pd.read_csv(adjustments_path)
    features = pd.read_csv(features_path)
    if adjustments.duplicated(["season", "team_id"]).any():
        raise ValueError("Domestic Surprise adjustments contain duplicate team-season keys")
    if features.duplicated(["season", "team_id"]).any():
        raise ValueError("Domestic Surprise features contain duplicate team-season keys")
    exposure = features[
        ["season", "team_id", "effective_european_exposure", "history_seasons_available"]
    ]
    data = adjustments.merge(
        exposure,
        on=["season", "team_id"],
        how="left",
        validate="one_to_one",
    )
    if data["effective_european_exposure"].isna().any():
        raise ValueError("Domestic Surprise exposure join is incomplete")
    data["baseline_rank"] = data.groupby("season")["baseline_ao_first_elo"].rank(
        method="min", ascending=False
    )
    data["current_rank"] = data.groupby("season")["adjusted_ao_first_elo"].rank(
        method="min", ascending=False
    )
    data["rank_change"] = data["baseline_rank"] - data["current_rank"]
    data["initial_elo_delta"] = (
        data["adjusted_ao_first_elo"] - data["baseline_ao_first_elo"]
    )
    data["changed"] = data["initial_elo_delta"].abs().gt(1e-12)
    data["exposure_band"] = pd.cut(
        data["effective_european_exposure"],
        bins=[-1e-12, 0.0, 0.25, 0.50, 0.75, 1.0],
        labels=["0", "(0,0.25]", "(0.25,0.50]", "(0.50,0.75]", "(0.75,1.00]"],
        include_lowest=True,
    )
    def summarize_scope(scope: str, frame: pd.DataFrame) -> dict[str, object]:
        changed = frame.loc[frame["changed"]]
        return {
            "scope": scope,
            "team_seasons": len(frame),
            "unique_clubs": int(frame["club_id"].nunique()),
            "changed_team_seasons": int(frame["changed"].sum()),
            "changed_share": float(frame["changed"].mean()),
            "mean_abs_initial_delta": float(frame["initial_elo_delta"].abs().mean()),
            "median_abs_initial_delta": float(frame["initial_elo_delta"].abs().median()),
            "p90_abs_initial_delta": float(frame["initial_elo_delta"].abs().quantile(0.90)),
            "p95_abs_initial_delta": float(frame["initial_elo_delta"].abs().quantile(0.95)),
            "maximum_positive_initial_delta": float(frame["initial_elo_delta"].max()),
            "maximum_negative_initial_delta": float(frame["initial_elo_delta"].min()),
            "mean_abs_rank_change": float(frame["rank_change"].abs().mean()),
            "maximum_rank_gain": int(frame["rank_change"].max()),
            "maximum_rank_loss": int(frame["rank_change"].min()),
            "domestic_adjustment_positive_cap_hits": int(
                np.isclose(frame["domestic_prior_adjustment"], 30.0, atol=1e-9).sum()
            ),
            "domestic_adjustment_negative_cap_hits": int(
                np.isclose(frame["domestic_prior_adjustment"], -30.0, atol=1e-9).sum()
            ),
            "ao_first_exact_500": int(
                np.isclose(frame["adjusted_ao_first_elo"], 500.0, atol=1e-9).sum()
            ),
            "ao_first_exact_2000": int(
                np.isclose(frame["adjusted_ao_first_elo"], 2000.0, atol=1e-9).sum()
            ),
            "minimum_current_ao_first_elo": float(frame["adjusted_ao_first_elo"].min()),
            "maximum_current_ao_first_elo": float(frame["adjusted_ao_first_elo"].max()),
            "changed_mean_abs_delta": (
                float(changed["initial_elo_delta"].abs().mean()) if not changed.empty else 0.0
            ),
        }

    latest_season = str(max(data["season"], key=lambda value: int(str(value).split("/")[0])))
    summary = pd.DataFrame(
        [
            summarize_scope("ALL_TEAM_SEASONS", data),
            summarize_scope(latest_season, data.loc[data["season"].eq(latest_season)]),
        ]
    )
    by_exposure = (
        data.groupby("exposure_band", observed=False)
        .agg(
            team_seasons=("team_id", "size"),
            changed_team_seasons=("changed", "sum"),
            mean_effective_exposure=("effective_european_exposure", "mean"),
            mean_initial_delta=("initial_elo_delta", "mean"),
            mean_abs_initial_delta=("initial_elo_delta", lambda values: values.abs().mean()),
            median_abs_initial_delta=("initial_elo_delta", lambda values: values.abs().median()),
            maximum_abs_initial_delta=("initial_elo_delta", lambda values: values.abs().max()),
        )
        .reset_index()
    )
    return data, summary, by_exposure


def build_end_rating_impact(
    evaluations: dict[str, ArmEvaluation],
    team_names: pd.DataFrame,
    evaluation_seasons: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    current = evaluations[CURRENT].end_ratings.loc[
        evaluations[CURRENT].end_ratings["season"].isin(evaluation_seasons)
    ].copy()
    reference = evaluations[REFERENCE].end_ratings.loc[
        evaluations[REFERENCE].end_ratings["season"].isin(evaluation_seasons)
    ][["season", "team_id", "end_live_rating"]].rename(
        columns={"end_live_rating": "reference_end_live_rating"}
    )
    result = current.merge(
        reference,
        on=["season", "team_id"],
        validate="one_to_one",
    ).merge(
        team_names[["season", "team_id", "team_name"]].drop_duplicates(),
        on=["season", "team_id"],
        how="left",
        validate="one_to_one",
    )
    result["end_live_delta_vs_reference"] = (
        result["end_live_rating"] - result["reference_end_live_rating"]
    )
    result["current_end_rank"] = result.groupby("season")["end_live_rating"].rank(
        method="min", ascending=False
    )
    result["reference_end_rank"] = result.groupby("season")["reference_end_live_rating"].rank(
        method="min", ascending=False
    )
    result["rank_change_vs_reference"] = (
        result["reference_end_rank"] - result["current_end_rank"]
    )
    result["season_live_change"] = result["end_live_rating"] - result["initial_rating"]
    summary = pd.DataFrame(
        [
            {
                "team_seasons": len(result),
                "changed_team_seasons": int(result["end_live_delta_vs_reference"].abs().gt(1e-12).sum()),
                "mean_abs_end_delta": float(result["end_live_delta_vs_reference"].abs().mean()),
                "median_abs_end_delta": float(result["end_live_delta_vs_reference"].abs().median()),
                "p90_abs_end_delta": float(result["end_live_delta_vs_reference"].abs().quantile(0.90)),
                "p95_abs_end_delta": float(result["end_live_delta_vs_reference"].abs().quantile(0.95)),
                "maximum_positive_end_delta": float(result["end_live_delta_vs_reference"].max()),
                "maximum_negative_end_delta": float(result["end_live_delta_vs_reference"].min()),
                "mean_abs_rank_change": float(result["rank_change_vs_reference"].abs().mean()),
                "maximum_rank_gain": int(result["rank_change_vs_reference"].max()),
                "maximum_rank_loss": int(result["rank_change_vs_reference"].min()),
            }
        ]
    )
    return result, summary


def contract_audit(production: dict, candidate: dict) -> tuple[pd.DataFrame, dict[str, object]]:
    blocks = (
        "domestic_surprise",
        "dynamic_core",
        "active_power_carry",
        "one_x_two_probability",
        "goal_margin",
        "xg_performance",
        "progression_bonus",
        "achievement_reserve",
        "competition_k",
    )
    documentation_keys = {"formula", "consistency_formula", "exposure_formula"}

    def semantic_value(value):
        if isinstance(value, dict):
            return {
                key: semantic_value(item)
                for key, item in value.items()
                if key not in documentation_keys
            }
        if isinstance(value, list):
            return [semantic_value(item) for item in value]
        return value

    rows = []
    for block in blocks:
        production_value = production.get(block)
        candidate_value = candidate.get(block)
        exact_equal = production_value == candidate_value
        semantic_equal = semantic_value(production_value) == semantic_value(candidate_value)
        rows.append(
            {
                "contract_block": block,
                "production_exactly_equals_final_candidate": exact_equal,
                "production_parameters_equal_final_candidate": semantic_equal,
                "production_value": json.dumps(production_value, sort_keys=True),
                "final_candidate_value": json.dumps(candidate_value, sort_keys=True),
            }
        )
    return pd.DataFrame(rows), {
        "all_active_blocks_exactly_equal": all(
            row["production_exactly_equals_final_candidate"] for row in rows
        ),
        "all_active_parameters_equal": all(
            row["production_parameters_equal_final_candidate"] for row in rows
        ),
        "production_revision": production.get("production_revision"),
        "candidate_revision": candidate.get("candidate_revision"),
        "candidate_status": candidate.get("candidate_status"),
        "candidate_production_activation": candidate.get("production_activation"),
    }


def validate_current_replay_against_pipeline(
    current: pd.DataFrame,
    production_baseline: pd.DataFrame,
) -> dict[str, object]:
    paired = current.merge(
        production_baseline[
            ["match_id", "expected_home_score", "brier_1x2", "log_loss_1x2", "power_delta"]
        ],
        on="match_id",
        suffixes=("_audit", "_pipeline"),
        validate="one_to_one",
    )
    checks = {}
    for column in ("expected_home_score", "brier_1x2", "log_loss_1x2", "power_delta"):
        difference = (paired[f"{column}_audit"] - paired[f"{column}_pipeline"]).abs()
        checks[f"maximum_abs_{column}_difference"] = float(difference.max())
    checks["match_count_equal"] = len(paired) == len(current) == len(production_baseline)
    checks["pipeline_exact_match"] = checks["match_count_equal"] and all(
        value <= 1e-12
        for key, value in checks.items()
        if key.startswith("maximum_abs_")
    )
    return checks


def data_quality_audit(
    events: pd.DataFrame,
    identity: pd.DataFrame,
    adjustments: pd.DataFrame,
    features: pd.DataFrame,
    evaluations: dict[str, ArmEvaluation],
    contract_status: dict[str, object],
    replay_validation: dict[str, object],
) -> pd.DataFrame:
    current = evaluations[CURRENT]
    current_predictions = current.predictions
    current_seasons = current.season_metrics
    chronology_rows = events.sort_values(["season", "event_order"], kind="stable")
    chronology_violations = sum(
        not pd.to_datetime(frame["kickoff_utc"], utc=True).is_monotonic_increasing
        for _, frame in chronology_rows.groupby("season", sort=False)
    )
    event_order_violations = sum(
        frame["event_order"].duplicated().any()
        or not frame["event_order"].is_monotonic_increasing
        for _, frame in chronology_rows.groupby("season", sort=False)
    )
    simultaneous_team_violations = 0
    for (_, kickoff), frame in events.groupby(["season", "kickoff_utc"], sort=False):
        teams = pd.concat([frame["home_team_id"], frame["away_team_id"]], ignore_index=True)
        simultaneous_team_violations += int(teams.duplicated().sum())
    checks = [
        ("contract_active_parameters_equal", bool(contract_status["all_active_parameters_equal"]), int(not contract_status["all_active_parameters_equal"]), "Production and final-candidate calculation parameters must match"),
        ("production_replay_exact_match", bool(replay_validation["pipeline_exact_match"]), int(not replay_validation["pipeline_exact_match"]), "Independent audit replay must match production pipeline"),
        ("event_match_id_unique", not events["match_id"].duplicated().any(), int(events["match_id"].duplicated().sum()), "One event row per match_id"),
        ("event_season_match_unique", not events.duplicated(["season", "match_id"]).any(), int(events.duplicated(["season", "match_id"]).sum()), "No duplicate season-match records"),
        ("identity_team_season_unique", not identity.duplicated(["season", "local_team_id"]).any(), int(identity.duplicated(["season", "local_team_id"]).sum()), "One identity row per team-season"),
        ("adjustment_team_season_unique", not adjustments.duplicated(["season", "team_id"]).any(), int(adjustments.duplicated(["season", "team_id"]).sum()), "One Domestic Surprise adjustment per team-season"),
        ("feature_team_season_unique", not features.duplicated(["season", "team_id"]).any(), int(features.duplicated(["season", "team_id"]).sum()), "One surprise feature row per team-season"),
        ("prediction_match_model_unique", not current_predictions.duplicated(["model", "match_id"]).any(), int(current_predictions.duplicated(["model", "match_id"]).sum()), "One current-model prediction per match"),
        ("probabilities_normalized", bool(np.allclose(current_predictions[["home_probability", "draw_probability", "away_probability"]].sum(axis=1), 1.0, atol=1e-12)), int((~np.isclose(current_predictions[["home_probability", "draw_probability", "away_probability"]].sum(axis=1), 1.0, atol=1e-12)).sum()), "1X2 probabilities sum to one"),
        ("power_zero_sum", float(current_seasons["season_power_conservation_error"].max()) <= 1e-9, float(current_seasons["season_power_conservation_error"].max()), "Power Elo is conserved within season"),
        ("match_zero_sum", float(current_predictions["zero_sum_error"].abs().max()) <= 1e-9, float(current_predictions["zero_sum_error"].abs().max()), "Every match Power update is zero-sum"),
        ("progression_cap", float(current_seasons["maximum_bonus_cap_error"].max()) <= 1e-9, float(current_seasons["maximum_bonus_cap_error"].max()), "Progression bonuses remain within competition caps"),
        ("exposure_range", features["effective_european_exposure"].between(0.0, 1.0).all(), int((~features["effective_european_exposure"].between(0.0, 1.0)).sum()), "Effective exposure stays in [0,1]"),
        ("insufficient_history_zero_adjustment", bool(adjustments.loc[adjustments["history_seasons"].lt(5), "ao_first_elo_adjustment"].abs().le(1e-12).all()), int(adjustments.loc[adjustments["history_seasons"].lt(5), "ao_first_elo_adjustment"].abs().gt(1e-12).sum()), "Fewer than five seasons must produce zero surprise adjustment"),
        ("surprise_sign_preserved", bool((adjustments["domestic_prior_adjustment"] * adjustments["raw_surprise"]).ge(-1e-12).all()), int((adjustments["domestic_prior_adjustment"] * adjustments["raw_surprise"]).lt(-1e-12).sum()), "Surprise adjustment cannot reverse raw signal sign"),
        ("surprise_cap", bool(adjustments["domestic_prior_adjustment"].abs().le(30.0 + 1e-9).all()), int(adjustments["domestic_prior_adjustment"].abs().gt(30.0 + 1e-9).sum()), "Domestic Surprise must respect +/-30 cap"),
        (
            "chronology_sorted",
            bool(chronology_violations == 0 and event_order_violations == 0),
            int(chronology_violations + event_order_violations),
            "event_order must be unique and UTC-monotonic within every season",
        ),
        (
            "simultaneous_team_collision",
            simultaneous_team_violations == 0,
            simultaneous_team_violations,
            "A team cannot appear twice at the same kickoff UTC",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "check": name,
                "passed": passed,
                "observed": observed,
                "requirement": requirement,
                "severity_if_failed": "HIGH" if name in {"production_replay_exact_match", "event_match_id_unique", "power_zero_sum", "match_zero_sum"} else "MEDIUM",
            }
            for name, passed, observed, requirement in checks
        ]
    )


def markdown_table(frame: pd.DataFrame, digits: int = 6) -> str:
    if frame.empty:
        return "_Veri yok._"
    headers = [str(column) for column in frame.columns]
    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for record in frame.itertuples(index=False, name=None):
        values = []
        for value in record:
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.{digits}f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def build_report(
    *,
    production: dict,
    candidate: dict,
    seasons: tuple[str, ...],
    evaluation_seasons: set[str],
    model_summary: pd.DataFrame,
    fold_summary: pd.DataFrame,
    competition_summary: pd.DataFrame,
    forward_ranking: pd.DataFrame,
    initial_summary: pd.DataFrame,
    exposure_summary: pd.DataFrame,
    initial_impact: pd.DataFrame,
    end_impact: pd.DataFrame,
    end_summary: pd.DataFrame,
    uncertainty: pd.DataFrame,
    quality: pd.DataFrame,
    contract_status: dict[str, object],
    replay_validation: dict[str, object],
) -> str:
    current = model_summary.loc[model_summary["model"].eq(CURRENT)].iloc[0]
    reference = model_summary.loc[model_summary["model"].eq(REFERENCE)].iloc[0]
    current_folds = fold_summary.loc[fold_summary["model"].eq(CURRENT)]
    ref_folds = fold_summary.loc[fold_summary["model"].eq(REFERENCE)].set_index("fold")
    brier_wins = int(sum(row.brier_1x2 < ref_folds.loc[row.fold, "brier_1x2"] for row in current_folds.itertuples()))
    log_wins = int(sum(row.log_loss_1x2 < ref_folds.loc[row.fold, "log_loss_1x2"] for row in current_folds.itertuples()))
    rank_wins = int(sum(row.same_season_spearman > ref_folds.loc[row.fold, "same_season_spearman"] for row in current_folds.itertuples()))
    pair_wins = int(sum(row.same_season_pairwise_accuracy > ref_folds.loc[row.fold, "same_season_pairwise_accuracy"] for row in current_folds.itertuples()))
    current_comp = competition_summary.loc[competition_summary["model"].eq(CURRENT)]
    current_forward = forward_ranking.loc[forward_ranking["model"].eq(CURRENT)]
    reference_forward = forward_ranking.loc[forward_ranking["model"].eq(REFERENCE)]
    current_all_forward = current_forward.loc[current_forward["competition"].eq("ALL")].iloc[0]
    reference_all_forward = reference_forward.loc[reference_forward["competition"].eq("ALL")].iloc[0]
    top_up = initial_impact.nlargest(10, "initial_elo_delta")
    top_down = initial_impact.nsmallest(10, "initial_elo_delta")
    live_up = end_impact.nlargest(10, "season_live_change")
    live_down = end_impact.nsmallest(10, "season_live_change")
    ablation = model_summary.loc[model_summary["model"].str.startswith("ABLATION")].copy()
    ablation["brier_cost_when_disabled"] = ablation["brier_1x2"] - current["brier_1x2"]
    ablation["log_loss_cost_when_disabled"] = ablation["log_loss_1x2"] - current["log_loss_1x2"]
    pooled_uncertainty = uncertainty.loc[
        uncertainty["candidate_model"].eq(CURRENT)
        & uncertainty["baseline_model"].eq(REFERENCE)
        & uncertainty["competition"].eq("ALL")
        & uncertainty["method"].eq("conservative_envelope")
    ][["metric", "mean_difference", "ci_95_lower", "ci_95_upper", "reliable_improvement"]]
    decision = "KEEP"
    return "\n".join(
        [
            "# AO European Elo Güncel Production Model Değerlendirmesi",
            "",
            "## Teknik özet",
            "",
            f"- Production contract hesap parametreleri final-candidate ile eşittir: `{contract_status['all_active_parameters_equal']}`. Birebir JSON eşitliği `{contract_status['all_active_blocks_exactly_equal']}`; fark production'a sonradan eklenen açıklayıcı formül alanlarıdır.",
            f"- Anlamlı karşılaştırma için aynı Scale/H/K üzerinde bütün aktif ek katmanları kapalı `{REFERENCE}` kolu üretildi.",
            f"- `{len(evaluation_seasons)}` unseen/development fold sezonunda `{int(current['matches'])}` maç değerlendirildi. Güncel model Brier `{current['brier_1x2']:.6f}`, log-loss `{current['log_loss_1x2']:.6f}`, accuracy `{current['accuracy_1x2']:.4f}` üretti.",
            f"- Referansa karşı farklar: Brier `{current['brier_1x2']-reference['brier_1x2']:+.6f}`, log-loss `{current['log_loss_1x2']-reference['log_loss_1x2']:+.6f}`, accuracy `{current['accuracy_1x2']-reference['accuracy_1x2']:+.4f}`.",
            f"- Fold kazanımları Brier `{brier_wins}/6`, log-loss `{log_wins}/6`, aynı-sezon Spearman `{rank_wins}/6`, pairwise `{pair_wins}/6`.",
            f"- Production kararı: **{decision}**. Model çalışır ve guardrail'ler geçer; ancak 2026/27 lig aşaması sonrası prospective holdout henüz tamamlanmadığı için yeni bir PROMOTE iddiası yapılmamalıdır.",
            "",
            "## Güncel sözleşme ve aktif mimari",
            "",
            f"- Model: `{production['model_version']}`; production revision `{production['production_revision']}`; final candidate `{candidate['candidate_version']}`.",
            f"- Statik: country benchmark `25`, European history benchmark `20`, sezon ağırlıkları `0.07/0.13/0.20/0.27/0.33`, country/european/exposure tail beta `0/0/0`.",
            "- Domestic Prior = 500 + lig gücü bileşeni + lig/kupa başarısının lig gücüyle ölçeklenmiş bileşeni. Şampiyonluk, kupa ve duble kuralları başlangıç ratinginde kullanılır.",
            f"- Domestic Surprise aktiftir: theta `{production['domestic_surprise']['coefficient']}`, variance penalty `{production['domestic_surprise']['variance_penalty']}`, cap `+/-{production['domestic_surprise']['max_abs_adjustment']}`, tam geçmiş `{production['domestic_surprise']['minimum_history_seasons']}` sezon.",
            f"- Dynamic: Scale `{production['dynamic_core']['elo_scale']:.6f}`, H `{production['dynamic_core']['home_advantage']:.6f}`, K `{production['dynamic_core']['k_factor']:.6f}`, carry `{production['active_power_carry']}`.",
            f"- Gol farkı: alpha `{production['goal_margin']['alpha']}`, tau `{production['goal_margin']['tau']}`, GD cap `{production['goal_margin']['goal_difference_cap']}`.",
            f"- xG: ratio `{production['xg_performance']['max_xg_ratio']}`, scale `{production['xg_performance']['xg_scale']}`, winner floor `{production['xg_performance']['minimum_winner_gain_ratio']}`; iki taraf xG yoksa GD-only fallback.",
            f"- Progression: UCL/UEL/UECL `12/8/4`, sezon cap `60/40/20`, winner-only, tek tie uygulaması, sezon resetli.",
            "- Achievement Reserve, Competition K, Dynamic K, season carry ve takım bazlı home context aktif değildir. Ev sahibi avantajı global H olarak uygulanır.",
            "",
            "## Baseline ve güncel model ana metrikleri",
            "",
            markdown_table(model_summary, 7),
            "",
            "Referans tarihsel bir production sürümü değildir; güncel modelin aktif ek katmanları kapatılmış kontrollü ablation çekirdeğidir. Production contract daha yeni otoritedir; final-candidate ile hesap parametreleri aynı olduğundan dürüst performans karşılaştırması bu kontrollü koldur.",
            "",
            "## Fold bazlı performans",
            "",
            markdown_table(fold_summary.loc[fold_summary["model"].isin([CURRENT, REFERENCE])], 7),
            "",
            "### Bağımlılığa dayanıklı pooled güven aralıkları",
            "",
            markdown_table(pooled_uncertainty, 7),
            "",
            "## UCL, UEL ve UECL",
            "",
            markdown_table(current_comp, 7),
            "",
            "## Sıralama ve ileri sezon kontrolü",
            "",
            f"- Aynı-sezon pooled Spearman `{current['same_season_spearman']:.6f}`, pairwise `{current['same_season_pairwise_accuracy']:.6f}`.",
            f"- Beş forward geçişte pooled Spearman current `{current_all_forward['ranking_score']:.6f}`, reference `{reference_all_forward['ranking_score']:.6f}`; pairwise current `{current_all_forward['pairwise_accuracy']:.6f}`, reference `{reference_all_forward['pairwise_accuracy']:.6f}`.",
            "- Aynı-sezon ranking diagnostiktir; forward ranking sezon sonu ratingi yalnız takip eden sezon performansına bağlar.",
            "",
            "## Başlangıç Elo etkisi",
            "",
            markdown_table(initial_summary, 6),
            "",
            markdown_table(exposure_summary, 6),
            "",
            "### En çok yükselen takım-sezonları",
            "",
            markdown_table(top_up[["season", "team_name", "country_code", "effective_european_exposure", "raw_surprise", "consistency_multiplier", "domestic_prior_adjustment", "initial_elo_delta", "rank_change"]], 5),
            "",
            "### En çok düşen takım-sezonları",
            "",
            markdown_table(top_down[["season", "team_name", "country_code", "effective_european_exposure", "raw_surprise", "consistency_multiplier", "domestic_prior_adjustment", "initial_elo_delta", "rank_change"]], 5),
            "",
            "## Sezon sonu rating etkisi",
            "",
            markdown_table(end_summary, 6),
            "",
            "### Güncel modelde en büyük sezon içi yükselişler",
            "",
            markdown_table(live_up[["season", "team_name", "initial_rating", "end_live_rating", "season_live_change", "rank_change_vs_reference"]], 4),
            "",
            "### Güncel modelde en büyük sezon içi düşüşler",
            "",
            markdown_table(live_down[["season", "team_name", "initial_rating", "end_live_rating", "season_live_change", "rank_change_vs_reference"]], 4),
            "",
            "## Ablation: katkı hangi katmandan geliyor?",
            "",
            markdown_table(ablation[["model", "brier_cost_when_disabled", "log_loss_cost_when_disabled", "accuracy_1x2", "same_season_spearman", "same_season_pairwise_accuracy"]], 7),
            "",
            "Pozitif `cost_when_disabled`, katman kapatıldığında loss'un yükseldiğini ve katmanın fayda sağladığını gösterir. Negatif değer katmanın mevcut kombinasyonda loss'a zarar verdiğini gösterir; bu sonuç nedensel veya bağımsız prospective kanıt değildir.",
            "",
            "## Güvenlik ve veri kalitesi",
            "",
            markdown_table(quality, 10),
            "",
            f"Bağımsız audit replay ile production pipeline eşleşmesi: `{replay_validation['pipeline_exact_match']}`. En büyük expected-score farkı `{replay_validation['maximum_abs_expected_home_score_difference']:.3e}`.",
            "",
            "## Güçlü ve zayıf noktalar",
            "",
            "- Güçlü: exact-date kronoloji, açık field-score/penaltı sözleşmesi, Power Elo zero-sum, bounded xG winner guard, progression cap ve eksik xG fallback denetlenebilir durumda.",
            "- Güçlü: Domestic Surprise etkisi exposure ile otomatik azalır; beş tam sezon yoksa düzeltme sıfırdır.",
            "- Zayıf: aktif katmanların bir kısmı manuel product kararıyla aktive edilmiştir; 2018/19-2025/26 geliştirme penceresi bağımsız prospective holdout değildir.",
            "- Zayıf: xG yalnız 2025/26'da geniş kapsamalıdır; önceki sezonlarda current kol çoğunlukla GD fallback kullanır.",
            "- Zayıf: progression winner-only ve non-zero-sum olduğu için AO Live toplamını artırır; Power Elo conservation geçse de görünen Live Elo toplamı korunmaz.",
            "- Zayıf: global H sabittir; takım bazlı saha etkisi güncel production'da yoktur.",
            "",
            "## Production kararı ve sonraki adım",
            "",
            f"**{decision}**: güncel production contract korunmalı. Bu değerlendirme contract değiştirmez. 2026/27 lig aşaması ve sonrası kilitli pre-match ledger ile prospective sonuç geldiğinde aynı paket tekrar çalıştırılmalıdır.",
            "",
            "## Açık sorular",
            "",
            "- xG katmanının çok sezonlu, tek sağlayıcılı per-shot veriyle katkısı aynı yönde kalacak mı?",
            "- Progression bonusunun küçük loss katkısı prospective sıralama katkısına dönüşecek mi?",
            "- Hiyerarşik lig-sezon + takım HomeContext, global H'yi güvenilir biçimde iyileştirebilir mi?",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the active AO European Elo model and fixed ablations")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    args = parser.parse_args()

    production_path = PRODUCTION_CONTRACT.resolve()
    candidate_path = FINAL_CANDIDATE_CONTRACT.resolve()
    production = json.loads(production_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    contract_comparison, contract_status = contract_audit(production, candidate)
    core, parameters = validate_production_contract(production)
    dynamic = json.loads(DYNAMIC_MANIFEST.read_text(encoding="utf-8"))
    static_config = AOEuropeanEloConfig(**dynamic["static_config"])
    static_config.validate()
    events = read_events(EVENTS_PATH)
    reserve, tie_audit = load_reserve_data(STATIC_DATA_ROOT, EVENTS_PATH, static_config)
    datasets = prepare_controlled_data(reserve, events)
    seasons = tuple(data.season for data in datasets)
    folds = expanding_folds(seasons)
    evaluation_seasons = {test for _, test in folds}
    if len(folds) != 6:
        raise ValueError("Expected six expanding folds")
    target = schedule_adjusted_team_performance(events)
    identity = load_team_season_identity()
    current_domestic = load_domestic_adjustments(DOMESTIC_ADJUSTMENTS, datasets)
    baseline_domestic = {
        (data.season, int(team_id)): float(data.reserve.goal.carry.core.initial_ratings[int(team_id)])
        for data in datasets
        for team_id in data.reserve.goal.carry.core.active_team_ids
    }
    xg_map = load_xg_map(XG_DATA, datasets)

    evaluations: dict[str, ArmEvaluation] = {}
    for arm in evaluation_arms():
        print(f"Evaluating {arm.name}", flush=True)
        evaluations[arm.name] = evaluate_arm(
            datasets,
            arm,
            core=core,
            parameters=parameters,
            current_domestic=current_domestic,
            baseline_domestic=baseline_domestic,
            xg_map=xg_map,
            target=target,
        )

    production_baseline, *_ = load_production_baseline()
    replay_validation = validate_current_replay_against_pipeline(
        evaluations[CURRENT].predictions,
        production_baseline,
    )
    model_summary = build_model_summary(evaluations, evaluation_seasons)
    fold_summary = build_fold_summary(evaluations, folds)
    competition_summary = build_competition_summary(evaluations, evaluation_seasons)
    forward_ranking = build_forward_ranking(evaluations, target, identity, seasons)
    uncertainty = build_dependency_uncertainty(
        evaluations,
        evaluation_seasons,
        bootstrap_samples=args.bootstrap_samples,
    )
    adjustments = pd.read_csv(DOMESTIC_ADJUSTMENTS)
    features = pd.read_csv(SURPRISE_FEATURES)
    initial_impact, initial_summary, exposure_summary = build_initial_impact(
        DOMESTIC_ADJUSTMENTS,
        SURPRISE_FEATURES,
    )
    end_impact, end_summary = build_end_rating_impact(
        evaluations,
        adjustments,
        evaluation_seasons,
    )
    quality = data_quality_audit(
        events,
        identity,
        adjustments,
        features,
        evaluations,
        contract_status,
        replay_validation,
    )
    if not quality["passed"].all():
        failures = quality.loc[~quality["passed"], "check"].tolist()
        raise ValueError(f"Current model audit failed: {failures}")

    all_predictions = pd.concat(
        [evaluation.predictions for evaluation in evaluations.values()],
        ignore_index=True,
    )
    all_end_ratings = pd.concat(
        [evaluation.end_ratings for evaluation in evaluations.values()],
        ignore_index=True,
    )
    all_season_metrics = pd.concat(
        [evaluation.season_metrics for evaluation in evaluations.values()],
        ignore_index=True,
    )
    all_rankings = pd.concat(
        [evaluation.same_season_ranking for evaluation in evaluations.values()],
        ignore_index=True,
    )
    all_bonuses = pd.concat(
        [evaluation.bonus_events for evaluation in evaluations.values() if not evaluation.bonus_events.empty],
        ignore_index=True,
    )

    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract_comparison.to_csv(output / "contract_comparison.csv", index=False)
    model_summary.to_csv(output / "model_summary.csv", index=False)
    fold_summary.to_csv(output / "fold_summary.csv", index=False)
    competition_summary.to_csv(output / "competition_summary.csv", index=False)
    forward_ranking.to_csv(output / "forward_ranking.csv", index=False)
    initial_impact.to_csv(output / "initial_elo_impact.csv", index=False)
    initial_summary.to_csv(output / "initial_elo_impact_summary.csv", index=False)
    exposure_summary.to_csv(output / "exposure_impact_summary.csv", index=False)
    end_impact.to_csv(output / "season_end_elo_impact.csv", index=False)
    end_summary.to_csv(output / "season_end_elo_impact_summary.csv", index=False)
    quality.to_csv(output / "safety_and_data_quality_audit.csv", index=False)
    uncertainty.to_csv(output / "dependency_uncertainty.csv", index=False)
    all_predictions.to_csv(output / "model_predictions.csv", index=False)
    all_end_ratings.to_csv(output / "model_end_ratings.csv", index=False)
    all_season_metrics.to_csv(output / "season_invariants.csv", index=False)
    all_rankings.to_csv(output / "same_season_ranking.csv", index=False)
    all_bonuses.to_csv(output / "progression_bonus_events.csv", index=False)
    tie_audit.to_csv(output / "tie_chronology_audit.csv", index=False)
    manifest = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "evaluation_only": True,
        "production_files_changed": False,
        "model_version": production["model_version"],
        "production_revision": production["production_revision"],
        "final_candidate_version": candidate["candidate_version"],
        "contract_status": contract_status,
        "contract_sha256": {
            "production": hashlib.sha256(production_path.read_bytes()).hexdigest(),
            "final_candidate": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
        },
        "seasons": list(seasons),
        "evaluation_fold_seasons": sorted(evaluation_seasons),
        "all_matches": int(len(events)),
        "evaluation_matches": int(model_summary.loc[model_summary["model"].eq(CURRENT), "matches"].iloc[0]),
        "xg_rows_available": len(xg_map),
        "bootstrap_samples": args.bootstrap_samples,
        "replay_validation": replay_validation,
        "decision": "KEEP",
        "caveat": "Development-window replay; not an untouched prospective holdout.",
    }
    active_model_snapshot = {
        "authority": {
            "status_file": str((ROOT / "MODEL_STATUS.md").resolve()),
            "production_contract": str(production_path),
            "static_dynamic_manifest": str(DYNAMIC_MANIFEST.resolve()),
            "precedence": "production_contract_over_final_candidate",
        },
        "model_version": production["model_version"],
        "production_revision": production["production_revision"],
        "static_initial_elo": dynamic["static_config"],
        "active_runtime_contract": {
            key: production[key]
            for key in (
                "domestic_surprise",
                "dynamic_core",
                "active_power_carry",
                "one_x_two_probability",
                "goal_margin",
                "xg_performance",
                "progression_bonus",
                "achievement_reserve",
                "competition_k",
            )
        },
        "calculation_order": [
            "domestic_prior",
            "european_prior",
            "effective_european_exposure_blend",
            "domestic_surprise_exposure_adjustment",
            "match_expected_score_with_global_home_advantage",
            "zero_sum_power_update_with_goal_margin_and_optional_xg",
            "winner_only_season_local_progression_bonus",
        ],
        "evaluation_only": True,
    }
    (output / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "active_model_snapshot.json").write_text(
        json.dumps(active_model_snapshot, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report = build_report(
        production=production,
        candidate=candidate,
        seasons=seasons,
        evaluation_seasons=evaluation_seasons,
        model_summary=model_summary,
        fold_summary=fold_summary,
        competition_summary=competition_summary,
        forward_ranking=forward_ranking,
        initial_summary=initial_summary,
        exposure_summary=exposure_summary,
        initial_impact=initial_impact,
        end_impact=end_impact,
        end_summary=end_summary,
        uncertainty=uncertainty,
        quality=quality,
        contract_status=contract_status,
        replay_validation=replay_validation,
    )
    (output / "current_model_evaluation_report.md").write_text(report, encoding="utf-8")
    print(f"Decision: KEEP")
    print(f"Report: {output / 'current_model_evaluation_report.md'}")


if __name__ == "__main__":
    main()
