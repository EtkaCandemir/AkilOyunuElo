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
from ao_elo.evaluation import (  # noqa: E402
    dependency_robust_loss_difference_ci,
    schedule_adjusted_team_performance,
)
from ao_elo.tournament_bonus import (  # noqa: E402
    COMPETITION_BONUS_RATIOS,
    ELIGIBLE_PROGRESSION_STAGES,
    FixedTournamentBonusConfig,
    apply_tournament_progress_bonus,
)
from scripts.run_controlled_goal_progression_backtest import (  # noqa: E402
    ControlledSeasonData,
    aggregate_rankings,
    calculate_match_probabilities,
    evaluate_predictions,
    prepare_controlled_data,
)
from scripts.run_dynamic_core_calibration import (  # noqa: E402
    DynamicCoreConfig,
    expanding_folds,
)
from scripts.run_final_robustness import (  # noqa: E402
    core_for_fold,
    draw_map_for_fold,
    load_team_season_identity,
    rank_value,
    read_final_production_contract,
    safe_rank_correlation,
    summarize_ranking,
    validate_fold_inputs,
)
from scripts.run_v2_achievement_reserve_calibration import (  # noqa: E402
    load_reserve_data,
)
from scripts.run_v2_dynamic_calibration import (  # noqa: E402
    MAX_RATING_MOVE_GUARDRAIL,
    RANK_CORRELATION_FLOOR,
)
from scripts.run_v2_evaluation_upgrade import (  # noqa: E402
    DrawModelConfig,
    read_events,
)


STATIC_DATA_ROOT = ROOT / "data" / "backtest_stage_b_2018_2026"
EVENTS_PATH = (
    ROOT / "data" / "external_elo_benchmark_2018_2026" / "exact_date_events.csv"
)
DYNAMIC_ROOT = ROOT / "output" / "v2_dynamic_calibration_2018_2026"
EVALUATION_ROOT = ROOT / "output" / "v2_evaluation_upgrade_2018_2026"
PRODUCTION_CONTRACT = ROOT / "contracts" / "ao_european_elo_v2_production.json"
OUTPUT_ROOT = ROOT / "output" / "fixed_tournament_bonus_backtest_2018_2026"

BASE_BONUSES = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0)
MODEL_BASE = "BASE"
MODEL_NESTED = "NESTED_FIXED_BONUS"
MODEL_FIXED_12 = "PRESPECIFIED_12_8_4"
COMPETITION_INDEX = {"UCL": 0, "UEL": 1, "UECL": 2}
RANK_TOLERANCE = 1e-9


@dataclass
class BonusEvaluation:
    metrics: dict[str, float | int]
    predictions: pd.DataFrame
    end_ratings: pd.DataFrame
    ranking: pd.DataFrame
    season_metrics: pd.DataFrame
    bonus_events: pd.DataFrame


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Nested walk-forward test of a fixed, winner-only, season-reset "
            "European tournament progression bonus"
        )
    )
    parser.add_argument("--static-data-root", type=Path, default=STATIC_DATA_ROOT)
    parser.add_argument("--events", type=Path, default=EVENTS_PATH)
    parser.add_argument("--dynamic-root", type=Path, default=DYNAMIC_ROOT)
    parser.add_argument("--evaluation-root", type=Path, default=EVALUATION_ROOT)
    parser.add_argument(
        "--production-contract", type=Path, default=PRODUCTION_CONTRACT
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples must be at least 100")

    contract = json.loads(
        args.production_contract.resolve().read_text(encoding="utf-8")
    )
    goal = validate_production_contract(contract)
    dynamic_root = args.dynamic_root.resolve()
    evaluation_root = args.evaluation_root.resolve()
    dynamic_manifest = json.loads(
        (dynamic_root / "selected_dynamic_model.json").read_text(encoding="utf-8")
    )
    static_config = AOEuropeanEloConfig(**dynamic_manifest["static_config"])
    static_config.validate()
    events = read_events(args.events.resolve())
    reserve_data, tie_audit = load_reserve_data(
        args.static_data_root.resolve(), args.events.resolve(), static_config
    )
    datasets = prepare_controlled_data(reserve_data, events)
    seasons = tuple(data.season for data in datasets)
    folds = expanding_folds(seasons)
    if len(folds) != 6:
        raise ValueError(f"Expected six outer folds, found {len(folds)}")
    core_selections = pd.read_csv(dynamic_root / "core_fold_selections.csv")
    draw_selections = pd.read_csv(
        evaluation_root / "draw_production_fold_selections.csv"
    )
    validate_fold_inputs(core_selections, draw_selections, folds)
    target = schedule_adjusted_team_performance(events)

    print("Fixed tournament bonus: nested walk-forward", flush=True)
    nested = run_walk_forward_backtest(
        datasets,
        target,
        folds,
        core_selections,
        draw_selections,
        alpha=goal["alpha"],
        tau=goal["tau"],
        goal_cap=goal["goal_cap"],
    )
    print("Fixed tournament bonus: fixed-grid OOS sensitivity", flush=True)
    surface = run_fixed_oos_grid(
        datasets,
        target,
        folds,
        core_selections,
        draw_selections,
        alpha=goal["alpha"],
        tau=goal["tau"],
        goal_cap=goal["goal_cap"],
    )

    predictions = nested["predictions"]
    fold_results = nested["fold_results"]
    summary = build_model_summary(predictions, fold_results)
    competition_summary = build_segment_summary(predictions, "competition")
    stage_summary = build_bonus_stage_summary(nested["bonus_events"])
    activation = build_activation_summary(predictions)
    uncertainty = build_uncertainty(
        predictions, bootstrap_samples=args.bootstrap_samples
    )
    decision = decide_model(
        summary,
        fold_results,
        competition_summary,
        uncertainty,
        nested["season_metrics"],
        nested["fold_selections"],
    )

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    nested["fold_selections"].to_csv(
        output_root / "fold_selections.csv", index=False
    )
    fold_results.to_csv(output_root / "fold_results.csv", index=False)
    predictions.to_csv(output_root / "unseen_predictions.csv", index=False)
    nested["ranking"].to_csv(output_root / "forward_ranking.csv", index=False)
    nested["season_metrics"].to_csv(
        output_root / "season_rating_distribution.csv", index=False
    )
    nested["bonus_events"].to_csv(
        output_root / "bonus_events.csv", index=False
    )
    surface.to_csv(output_root / "fixed_candidate_oos_metrics.csv", index=False)
    summary.to_csv(output_root / "model_comparison.csv", index=False)
    competition_summary.to_csv(
        output_root / "competition_summary.csv", index=False
    )
    stage_summary.to_csv(output_root / "stage_summary.csv", index=False)
    activation.to_csv(output_root / "bonus_activation_summary.csv", index=False)
    uncertainty.to_csv(output_root / "dependency_uncertainty.csv", index=False)
    tie_audit.to_csv(output_root / "tie_chronology_audit.csv", index=False)
    (output_root / "decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(
        output_root / "backtest_report.md",
        contract,
        goal,
        summary,
        competition_summary,
        surface,
        activation,
        uncertainty,
        nested["fold_selections"],
        decision,
    )
    print(f"Decision: {decision['decision']}")
    print(f"Report: {output_root / 'backtest_report.md'}")


def validate_production_contract(contract: dict[str, object]) -> dict[str, float | int]:
    if float(contract["active_power_carry"]) != 0.0:
        raise ValueError("Fixed bonus test requires production carry=0")
    goal = contract["goal_margin"]
    if not isinstance(goal, dict) or not bool(goal.get("active")):
        raise ValueError("Controlled goal-difference layer must be active")
    if bool(contract["progression_bonus"]["active"]):
        raise ValueError("Production progression bonus must remain disabled")
    if bool(contract["achievement_reserve"]["active"]):
        raise ValueError("Production achievement reserve must remain disabled")
    values: dict[str, float | int] = {
        "alpha": float(goal["alpha"]),
        "tau": float(goal["tau"]),
        "goal_cap": int(goal["goal_difference_cap"]),
    }
    if values != {"alpha": 0.1, "tau": 300.0, "goal_cap": 4}:
        raise ValueError(f"Unexpected controlled goal configuration: {values}")
    return values


def candidate_grid() -> tuple[FixedTournamentBonusConfig, ...]:
    result = tuple(FixedTournamentBonusConfig(value) for value in BASE_BONUSES)
    for candidate in result:
        candidate.validate()
    return result


def evaluate_sequence(
    datasets: tuple[ControlledSeasonData, ...],
    core: DynamicCoreConfig,
    draw_mapping: dict[str, DrawModelConfig],
    target: pd.DataFrame,
    config: FixedTournamentBonusConfig,
    *,
    alpha: float,
    tau: float,
    goal_cap: int,
    evaluation_seasons: set[str] | None = None,
    ranking_target_seasons: set[str] | None = None,
    return_details: bool = False,
) -> BonusEvaluation:
    core.validate()
    config.validate()
    evaluation = evaluation_seasons or {data.season for data in datasets}
    prediction_rows: list[dict[str, object]] = []
    end_rows: list[dict[str, object]] = []
    season_rows: list[dict[str, object]] = []
    bonus_rows: list[dict[str, object]] = []
    brier_total = 0.0
    log_total = 0.0
    correct_total = 0
    matches_total = 0

    for data in datasets:
        reserve = data.reserve
        goal = reserve.goal
        season_core = goal.carry.core
        power = season_core.initial_ratings.copy()
        bonus = np.zeros((len(power), len(COMPETITION_INDEX)), dtype=float)
        active_ids = season_core.active_team_ids
        processed_ties: set[str] = set()
        initial_power_total = float(np.sum(power[active_ids]))
        match_deltas: list[float] = []
        max_pair_error = 0.0
        max_cap_error = 0.0

        for index, (home_id, away_id, neutral, competition) in enumerate(
            zip(
                season_core.home_team_ids,
                season_core.away_team_ids,
                season_core.neutral_flags,
                season_core.competitions,
            )
        ):
            home_id = int(home_id)
            away_id = int(away_id)
            competition = str(competition)
            match_id = str(season_core.match_ids[index])
            home_bonus_pre = float(bonus[home_id].sum())
            away_bonus_pre = float(bonus[away_id].sum())
            home_live_pre = float(power[home_id]) + home_bonus_pre
            away_live_pre = float(power[away_id]) + away_bonus_pre
            penalty_shootout = bool(goal.penalty_flags[index])
            field_draw = int(data.home_goals[index]) == int(data.away_goals[index])
            # In a two-leg tie, the deciding leg can have a non-draw field
            # score even when the aggregate tie later goes to penalties. The
            # field result remains the match result, while the shootout still
            # suppresses goal-margin amplification.
            penalty_match_draw = penalty_shootout and field_draw
            update = update_match_elo(
                home_live_pre,
                away_live_pre,
                int(data.home_goals[index]),
                int(data.away_goals[index]),
                k_factor=core.k_factor,
                elo_scale=core.elo_scale,
                home_advantage=core.home_advantage,
                is_neutral=bool(neutral),
                decided_on_penalties=penalty_match_draw,
                alpha=0.0 if penalty_shootout else alpha,
                tau=tau,
            )
            probabilities = calculate_match_probabilities(
                update.expected_home_score, competition, draw_mapping
            )
            observed = (
                0
                if update.actual_home_score == 1.0
                else 1
                if update.actual_home_score == 0.5
                else 2
            )
            target_vector = np.eye(3, dtype=float)[observed]
            brier = float(
                np.square(np.asarray(probabilities) - target_vector).sum()
            )
            log_loss = -math.log(max(float(probabilities[observed]), 1e-15))
            predicted = int(np.argmax(probabilities))

            power[home_id] += update.power_delta
            power[away_id] -= update.power_delta
            match_deltas.append(abs(update.power_delta))
            max_pair_error = max(max_pair_error, update.zero_sum_error)

            applied_bonus = 0.0
            winner_id = -1
            stage = str(reserve.stages[index])
            tie_value = reserve.tie_ids[index]
            tie_id = None if tie_value is None else str(tie_value)
            if (
                config.base_bonus > 0.0
                and bool(reserve.tie_decider_flags[index])
                and stage in ELIGIBLE_PROGRESSION_STAGES
            ):
                if tie_id is None:
                    raise ValueError(f"{data.season}/{match_id}: missing deciding tie_id")
                winner_id = int(reserve.advanced_team_ids[index])
                if winner_id not in (home_id, away_id):
                    raise ValueError(
                        f"{data.season}/{tie_id}: advanced team is not a participant"
                    )
                component = COMPETITION_INDEX[competition]
                bonus_update = apply_tournament_progress_bonus(
                    float(bonus[winner_id, component]),
                    competition,
                    stage,
                    tie_id,
                    processed_ties,
                    config,
                )
                bonus[winner_id, component] = bonus_update.bonus_post
                applied_bonus = bonus_update.applied_bonus
                max_cap_error = max(
                    max_cap_error,
                    max(0.0, bonus_update.bonus_post - bonus_update.competition_cap),
                )
                if return_details and data.season in evaluation:
                    bonus_rows.append(
                        {
                            "season": data.season,
                            "match_id": match_id,
                            "tie_id": tie_id,
                            "competition": competition,
                            "stage": stage,
                            "winner_team_id": winner_id,
                            "applied_bonus": applied_bonus,
                            "competition_bonus_post": bonus_update.bonus_post,
                            "competition_cap": bonus_update.competition_cap,
                            "base_bonus": config.base_bonus,
                        }
                    )

            if data.season in evaluation:
                matches_total += 1
                brier_total += brier
                log_total += log_loss
                correct_total += int(predicted == observed)
                if return_details:
                    prediction_rows.append(
                        {
                            "match_id": match_id,
                            "season": data.season,
                            "kickoff_utc": pd.Timestamp(data.kickoff_utc[index]),
                            "competition": competition,
                            "round": str(goal.carry.rounds[index]),
                            "stage": stage,
                            "tie_id": tie_id,
                            "home_team_id": home_id,
                            "away_team_id": away_id,
                            "home_goals": int(data.home_goals[index]),
                            "away_goals": int(data.away_goals[index]),
                            "decided_on_penalties": penalty_shootout,
                            "penalty_match_draw": penalty_match_draw,
                            "is_neutral": bool(neutral),
                            "home_power_pre": float(power[home_id] - update.power_delta),
                            "away_power_pre": float(power[away_id] + update.power_delta),
                            "home_bonus_pre": home_bonus_pre,
                            "away_bonus_pre": away_bonus_pre,
                            "bonus_difference_pre": home_bonus_pre - away_bonus_pre,
                            "home_live_pre": home_live_pre,
                            "away_live_pre": away_live_pre,
                            "expected_home_score": update.expected_home_score,
                            "home_probability": probabilities[0],
                            "draw_probability": probabilities[1],
                            "away_probability": probabilities[2],
                            "actual_class": observed,
                            "predicted_class": predicted,
                            "brier_1x2": brier,
                            "log_loss_1x2": log_loss,
                            "goal_difference": update.goal_difference,
                            "goal_multiplier": update.goal_difference_multiplier,
                            "power_delta": update.power_delta,
                            "bonus_applied_after_match": applied_bonus,
                            "bonus_winner_id": winner_id,
                            "base_bonus": config.base_bonus,
                        }
                    )

        if data.season in evaluation:
            end_power = power[active_ids]
            end_bonus = bonus[active_ids].sum(axis=1)
            end_live = end_power + end_bonus
            start = season_core.initial_ratings[active_ids]
            power_total_error = abs(float(end_power.sum()) - initial_power_total)
            live_accounting_error = abs(
                (float(end_live.sum()) - initial_power_total) - float(end_bonus.sum())
            )
            season_rows.append(
                {
                    "season": data.season,
                    "base_bonus": config.base_bonus,
                    "matches": len(season_core.match_ids),
                    "teams": len(active_ids),
                    "rating_min": float(end_live.min()),
                    "rating_max": float(end_live.max()),
                    "rating_mean": float(end_live.mean()),
                    "rating_std": float(end_live.std()),
                    "total_bonus_added": float(end_bonus.sum()),
                    "teams_with_bonus": int(np.count_nonzero(end_bonus > 0.0)),
                    "maximum_team_bonus": float(end_bonus.max()),
                    "match_delta_max_abs": float(max(match_deltas)),
                    "max_abs_rating_change": float(np.max(np.abs(end_live - start))),
                    "start_end_rank_correlation": safe_rank_correlation(start, end_live),
                    "match_pair_zero_sum_error": max_pair_error,
                    "power_total_error": power_total_error,
                    "bonus_cap_error": max_cap_error,
                    "live_total_accounting_error": live_accounting_error,
                    "season_start_bonus": 0.0,
                }
            )
            end_rows.extend(
                {
                    "season": data.season,
                    "team_id": int(team_id),
                    "initial_rating": float(season_core.initial_ratings[team_id]),
                    "end_power_rating": float(power[team_id]),
                    "end_tournament_bonus": float(bonus[team_id].sum()),
                    "end_live_rating": float(power[team_id] + bonus[team_id].sum()),
                }
                for team_id in active_ids
            )

    if matches_total == 0:
        raise ValueError("No evaluation matches were processed")
    end_ratings = pd.DataFrame(end_rows)
    ranking = summarize_ranking(
        end_ratings,
        target,
        allowed_target_seasons=ranking_target_seasons or set(target["season"]),
        identity=load_team_season_identity(),
    )
    seasons = pd.DataFrame(season_rows)
    metrics: dict[str, float | int] = {
        "matches": matches_total,
        "brier_1x2": brier_total / matches_total,
        "log_loss_1x2": log_total / matches_total,
        "accuracy_1x2": correct_total / matches_total,
        "ranking_score": rank_value(ranking, "ranking_score"),
        "pairwise_accuracy": rank_value(ranking, "pairwise_accuracy"),
        "minimum_start_end_rank_correlation": float(
            seasons["start_end_rank_correlation"].min()
        ),
        "maximum_abs_rating_change": float(seasons["max_abs_rating_change"].max()),
        "maximum_abs_match_delta": float(seasons["match_delta_max_abs"].max()),
        "maximum_power_zero_sum_error": float(
            seasons[["match_pair_zero_sum_error", "power_total_error"]]
            .to_numpy(float)
            .max()
        ),
        "maximum_bonus_cap_error": float(seasons["bonus_cap_error"].max()),
        "maximum_live_accounting_error": float(
            seasons["live_total_accounting_error"].max()
        ),
        "maximum_season_start_bonus": float(seasons["season_start_bonus"].max()),
        "total_bonus_added": float(seasons["total_bonus_added"].sum()),
        "maximum_team_bonus": float(seasons["maximum_team_bonus"].max()),
    }
    return BonusEvaluation(
        metrics=metrics,
        predictions=pd.DataFrame(prediction_rows),
        end_ratings=end_ratings,
        ranking=ranking,
        season_metrics=seasons,
        bonus_events=pd.DataFrame(bonus_rows),
    )


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
        and row["minimum_start_end_rank_correlation"] >= RANK_CORRELATION_FLOOR
        and row["maximum_abs_rating_change"] <= MAX_RATING_MOVE_GUARDRAIL
        and row["maximum_power_zero_sum_error"] <= 1e-9
        and row["maximum_bonus_cap_error"] <= 1e-9
        and row["maximum_live_accounting_error"] <= 1e-9
        and row["maximum_season_start_bonus"] <= 1e-12
    )


def select_candidate(metrics: pd.DataFrame) -> pd.Series:
    baseline = metrics.loc[metrics["base_bonus"].eq(0.0)].iloc[0]
    eligible = metrics.loc[
        metrics.apply(candidate_is_safe, axis=1, baseline=baseline)
    ].copy()
    if eligible.empty:
        return baseline
    return eligible.sort_values(
        [
            "ranking_score",
            "pairwise_accuracy",
            "brier_1x2",
            "log_loss_1x2",
            "base_bonus",
        ],
        ascending=[False, False, True, True, True],
        na_position="last",
    ).iloc[0]


def evaluate_candidate_set(
    datasets: tuple[ControlledSeasonData, ...],
    target: pd.DataFrame,
    core: DynamicCoreConfig,
    draw_mapping: dict[str, DrawModelConfig],
    *,
    alpha: float,
    tau: float,
    goal_cap: int,
) -> pd.DataFrame:
    seasons = {data.season for data in datasets}
    rows = []
    for config in candidate_grid():
        result = evaluate_sequence(
            datasets,
            core,
            draw_mapping,
            target,
            config,
            alpha=alpha,
            tau=tau,
            goal_cap=goal_cap,
            evaluation_seasons=seasons,
            ranking_target_seasons=seasons,
        )
        rows.append({"base_bonus": config.base_bonus, **result.metrics})
    return pd.DataFrame(rows)


def run_walk_forward_backtest(
    datasets: tuple[ControlledSeasonData, ...],
    target: pd.DataFrame,
    folds: tuple[tuple[tuple[str, ...], str], ...],
    core_selections: pd.DataFrame,
    draw_selections: pd.DataFrame,
    *,
    alpha: float,
    tau: float,
    goal_cap: int,
) -> dict[str, pd.DataFrame]:
    by_season = {data.season: data for data in datasets}
    selection_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    ranking_frames: list[pd.DataFrame] = []
    season_frames: list[pd.DataFrame] = []
    bonus_frames: list[pd.DataFrame] = []

    for fold, (train_seasons, test_season) in enumerate(folds, start=1):
        core = core_for_fold(core_selections, fold)
        draw_mapping = draw_map_for_fold(draw_selections, fold)
        train_metrics = evaluate_candidate_set(
            tuple(by_season[season] for season in train_seasons),
            target,
            core,
            draw_mapping,
            alpha=alpha,
            tau=tau,
            goal_cap=goal_cap,
        )
        selected = select_candidate(train_metrics)
        selected_bonus = float(selected["base_bonus"])
        selection_rows.append(
            {
                "fold": fold,
                "train_seasons": "|".join(train_seasons),
                "test_season": test_season,
                "selected_base_bonus": selected_bonus,
                "selected_ucl_bonus": selected_bonus,
                "selected_uel_bonus": selected_bonus * 2.0 / 3.0,
                "selected_uecl_bonus": selected_bonus / 3.0,
                "train_brier_1x2": float(selected["brier_1x2"]),
                "train_log_loss_1x2": float(selected["log_loss_1x2"]),
                "train_ranking_score": float(selected["ranking_score"]),
                "train_pairwise_accuracy": float(selected["pairwise_accuracy"]),
            }
        )
        models = (
            (MODEL_BASE, FixedTournamentBonusConfig(0.0)),
            (MODEL_NESTED, FixedTournamentBonusConfig(selected_bonus)),
            (MODEL_FIXED_12, FixedTournamentBonusConfig(12.0)),
        )
        for model, config in models:
            result = evaluate_sequence(
                (by_season[test_season],),
                core,
                draw_mapping,
                target,
                config,
                alpha=alpha,
                tau=tau,
                goal_cap=goal_cap,
                evaluation_seasons={test_season},
                ranking_target_seasons=set(target["season"]),
                return_details=True,
            )
            predictions = result.predictions.copy()
            predictions.insert(0, "fold", fold)
            predictions.insert(1, "model", model)
            prediction_frames.append(predictions)
            seasons = result.season_metrics.copy()
            seasons.insert(0, "fold", fold)
            seasons.insert(1, "model", model)
            season_frames.append(seasons)
            events = result.bonus_events.copy()
            if not events.empty:
                events.insert(0, "fold", fold)
                events.insert(1, "model", model)
                bonus_frames.append(events)
            ranking = result.ranking.copy()
            if not ranking.empty:
                ranking.insert(0, "fold", fold)
                ranking.insert(1, "model", model)
                ranking_frames.append(ranking)
            fold_rows.append(
                {
                    "fold": fold,
                    "test_season": test_season,
                    "model": model,
                    "base_bonus": config.base_bonus,
                    **result.metrics,
                }
            )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    base = predictions.loc[
        predictions["model"].eq(MODEL_BASE),
        ["match_id", "expected_home_score"],
    ].rename(columns={"expected_home_score": "base_expected_home_score"})
    predictions = predictions.merge(
        base, on="match_id", how="left", validate="many_to_one"
    )
    predictions["match_band"] = np.select(
        [
            predictions["base_expected_home_score"].ge(0.60),
            predictions["base_expected_home_score"].le(0.40),
        ],
        ["HOME_FAVORITE", "HOME_UNDERDOG"],
        default="BALANCED",
    )
    return {
        "fold_selections": pd.DataFrame(selection_rows),
        "fold_results": pd.DataFrame(fold_rows),
        "predictions": predictions,
        "ranking": (
            pd.concat(ranking_frames, ignore_index=True)
            if ranking_frames
            else pd.DataFrame()
        ),
        "season_metrics": pd.concat(season_frames, ignore_index=True),
        "bonus_events": (
            pd.concat(bonus_frames, ignore_index=True)
            if bonus_frames
            else pd.DataFrame()
        ),
    }


def run_fixed_oos_grid(
    datasets: tuple[ControlledSeasonData, ...],
    target: pd.DataFrame,
    folds: tuple[tuple[tuple[str, ...], str], ...],
    core_selections: pd.DataFrame,
    draw_selections: pd.DataFrame,
    *,
    alpha: float,
    tau: float,
    goal_cap: int,
) -> pd.DataFrame:
    by_season = {data.season: data for data in datasets}
    rows = []
    for config in candidate_grid():
        predictions = []
        rankings = []
        seasons = []
        for fold, (_, test_season) in enumerate(folds, start=1):
            result = evaluate_sequence(
                (by_season[test_season],),
                core_for_fold(core_selections, fold),
                draw_map_for_fold(draw_selections, fold),
                target,
                config,
                alpha=alpha,
                tau=tau,
                goal_cap=goal_cap,
                evaluation_seasons={test_season},
                ranking_target_seasons=set(target["season"]),
                return_details=True,
            )
            predictions.append(result.predictions)
            seasons.append(result.season_metrics)
            if not result.ranking.empty:
                rankings.append(result.ranking)
        prediction_frame = pd.concat(predictions, ignore_index=True)
        ranking = aggregate_rankings(pd.concat(rankings, ignore_index=True))
        season_frame = pd.concat(seasons, ignore_index=True)
        rows.append(
            {
                "base_bonus": config.base_bonus,
                "ucl_bonus": config.increment("UCL"),
                "uel_bonus": config.increment("UEL"),
                "uecl_bonus": config.increment("UECL"),
                **evaluate_predictions(prediction_frame),
                "ranking_score": rank_value(ranking, "ranking_score"),
                "pairwise_accuracy": rank_value(ranking, "pairwise_accuracy"),
                "minimum_start_end_rank_correlation": float(
                    season_frame["start_end_rank_correlation"].min()
                ),
                "maximum_abs_rating_change": float(
                    season_frame["max_abs_rating_change"].max()
                ),
                "maximum_team_bonus": float(
                    season_frame["maximum_team_bonus"].max()
                ),
                "total_bonus_added": float(season_frame["total_bonus_added"].sum()),
            }
        )
    frame = pd.DataFrame(rows)
    baseline = frame.loc[frame["base_bonus"].eq(0.0)].iloc[0]
    frame["brier_delta_vs_base"] = frame["brier_1x2"] - baseline["brier_1x2"]
    frame["log_loss_delta_vs_base"] = (
        frame["log_loss_1x2"] - baseline["log_loss_1x2"]
    )
    frame["ranking_delta_vs_base"] = (
        frame["ranking_score"] - baseline["ranking_score"]
    )
    frame["pairwise_delta_vs_base"] = (
        frame["pairwise_accuracy"] - baseline["pairwise_accuracy"]
    )
    return frame


def build_model_summary(
    predictions: pd.DataFrame, fold_results: pd.DataFrame
) -> pd.DataFrame:
    base_metrics = evaluate_predictions(
        predictions.loc[predictions["model"].eq(MODEL_BASE)]
    )
    base_folds = fold_results.loc[fold_results["model"].eq(MODEL_BASE)].set_index(
        "fold"
    )
    rows = []
    for model, frame in predictions.groupby("model", sort=False):
        metrics = evaluate_predictions(frame)
        folds = fold_results.loc[fold_results["model"].eq(model)].set_index("fold")
        common = folds.index.intersection(base_folds.index)
        brier_wins = int(
            (folds.loc[common, "brier_1x2"] < base_folds.loc[common, "brier_1x2"]).sum()
        )
        log_wins = int(
            (
                folds.loc[common, "log_loss_1x2"]
                < base_folds.loc[common, "log_loss_1x2"]
            ).sum()
        )
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
                >= base_folds.loc[rank_index, "pairwise_accuracy"] - RANK_TOLERANCE
            )
        )
        rows.append(
            {
                "model": model,
                **metrics,
                "brier_delta_vs_base": metrics["brier_1x2"] - base_metrics["brier_1x2"],
                "log_loss_delta_vs_base": (
                    metrics["log_loss_1x2"] - base_metrics["log_loss_1x2"]
                ),
                "brier_fold_wins": brier_wins,
                "log_loss_fold_wins": log_wins,
                "ranking_evaluable_folds": len(rank_index),
                "ranking_no_regression_folds": int(rank_safe.sum()),
                "maximum_abs_rating_change": float(
                    folds["maximum_abs_rating_change"].max()
                ),
                "minimum_start_end_rank_correlation": float(
                    folds["minimum_start_end_rank_correlation"].min()
                ),
                "maximum_power_zero_sum_error": float(
                    folds["maximum_power_zero_sum_error"].max()
                ),
                "maximum_bonus_cap_error": float(
                    folds["maximum_bonus_cap_error"].max()
                ),
            }
        )
    return pd.DataFrame(rows)


def paired_loss_frame(
    predictions: pd.DataFrame, model: str, metric: str
) -> pd.DataFrame:
    base = predictions.loc[
        predictions["model"].eq(MODEL_BASE),
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
    challenger = predictions.loc[
        predictions["model"].eq(model), ["match_id", metric]
    ].rename(columns={metric: "challenger_loss"})
    paired = base.merge(challenger, on="match_id", validate="one_to_one")
    paired["loss_difference"] = paired["challenger_loss"] - paired["base_loss"]
    return paired


def build_uncertainty(
    predictions: pd.DataFrame, *, bootstrap_samples: int
) -> pd.DataFrame:
    frames = []
    for model in (MODEL_NESTED, MODEL_FIXED_12):
        for metric in ("brier_1x2", "log_loss_1x2"):
            result = dependency_robust_loss_difference_ci(
                paired_loss_frame(predictions, model, metric),
                bootstrap_samples=bootstrap_samples,
                seed=20260803 + (0 if model == MODEL_NESTED else 10007),
            )
            result.insert(0, "model", model)
            result.insert(1, "metric", metric)
            frames.append(result)
    return pd.concat(frames, ignore_index=True)


def build_segment_summary(predictions: pd.DataFrame, segment: str) -> pd.DataFrame:
    rows = []
    for (model, value), frame in predictions.groupby(["model", segment], sort=True):
        rows.append({"model": model, segment: value, **evaluate_predictions(frame)})
    result = pd.DataFrame(rows)
    base = result.loc[result["model"].eq(MODEL_BASE)].set_index(segment)
    result["brier_delta_vs_base"] = result.apply(
        lambda row: row["brier_1x2"] - base.loc[row[segment], "brier_1x2"], axis=1
    )
    result["log_loss_delta_vs_base"] = result.apply(
        lambda row: row["log_loss_1x2"]
        - base.loc[row[segment], "log_loss_1x2"],
        axis=1,
    )
    return result


def build_bonus_stage_summary(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(
            columns=["model", "competition", "stage", "events", "bonus_added"]
        )
    return (
        events.groupby(["model", "competition", "stage"], as_index=False)
        .agg(events=("tie_id", "nunique"), bonus_added=("applied_bonus", "sum"))
        .sort_values(["model", "competition", "stage"])
    )


def build_activation_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, competition), frame in predictions.groupby(
        ["model", "competition"], sort=True
    ):
        active = frame["bonus_difference_pre"].abs().gt(1e-12)
        changed = (
            frame["expected_home_score"] - frame["base_expected_home_score"]
        ).abs().gt(1e-12)
        rows.append(
            {
                "model": model,
                "competition": competition,
                "matches": len(frame),
                "matches_with_direct_bonus_difference": int(active.sum()),
                "direct_activation_rate": float(active.mean()),
                "matches_with_changed_expectation": int(changed.sum()),
                "prediction_change_rate": float(changed.mean()),
                "maximum_abs_bonus_difference": float(
                    frame["bonus_difference_pre"].abs().max()
                ),
            }
        )
    return pd.DataFrame(rows)


def decide_model(
    summary: pd.DataFrame,
    fold_results: pd.DataFrame,
    competition_summary: pd.DataFrame,
    uncertainty: pd.DataFrame,
    season_metrics: pd.DataFrame,
    selections: pd.DataFrame,
) -> dict[str, object]:
    candidate = summary.loc[summary["model"].eq(MODEL_NESTED)].iloc[0]
    prespecified = summary.loc[summary["model"].eq(MODEL_FIXED_12)].iloc[0]
    envelopes = uncertainty.loc[
        uncertainty["model"].eq(MODEL_NESTED)
        & uncertainty["method"].eq("conservative_envelope")
    ].set_index("metric")
    segments = competition_summary.loc[
        competition_summary["model"].eq(MODEL_NESTED)
    ]
    prespecified_segments = competition_summary.loc[
        competition_summary["model"].eq(MODEL_FIXED_12)
    ]
    candidate_seasons = season_metrics.loc[
        season_metrics["model"].eq(MODEL_NESTED)
    ]
    prespecified_seasons = season_metrics.loc[
        season_metrics["model"].eq(MODEL_FIXED_12)
    ]
    nonzero_folds = int(selections["selected_base_bonus"].gt(0.0).sum())
    gates = {
        "nonzero_selected_at_least_4_of_6": nonzero_folds >= 4,
        "brier_improves_at_least_4_of_6": int(candidate["brier_fold_wins"]) >= 4,
        "log_loss_improves_at_least_4_of_6": int(candidate["log_loss_fold_wins"]) >= 4,
        "pooled_brier_not_worse": float(candidate["brier_delta_vs_base"]) <= 0.0,
        "pooled_log_loss_not_worse": float(candidate["log_loss_delta_vs_base"]) <= 0.0,
        "no_dependency_reliable_harm": bool(
            (envelopes["ci_95_lower"] <= 0.0).all()
        ),
        "ranking_no_regression_all_evaluable_folds": bool(
            int(candidate["ranking_evaluable_folds"]) >= 5
            and int(candidate["ranking_no_regression_folds"])
            == int(candidate["ranking_evaluable_folds"])
        ),
        "no_competition_point_regression": bool(
            (segments["brier_delta_vs_base"] <= 0.0).all()
            and (segments["log_loss_delta_vs_base"] <= 0.0).all()
        ),
        "rank_correlation_guardrail": bool(
            candidate_seasons["start_end_rank_correlation"].min()
            >= RANK_CORRELATION_FLOOR
        ),
        "maximum_movement_guardrail": bool(
            candidate_seasons["max_abs_rating_change"].max()
            <= MAX_RATING_MOVE_GUARDRAIL
        ),
        "power_zero_sum_preserved": bool(
            candidate_seasons[
                ["match_pair_zero_sum_error", "power_total_error"]
            ].to_numpy(float).max()
            <= 1e-9
        ),
        "bonus_cap_preserved": bool(candidate_seasons["bonus_cap_error"].max() <= 1e-9),
        "season_reset_preserved": bool(
            candidate_seasons["season_start_bonus"].max() <= 1e-12
        ),
        "live_total_accounting_preserved": bool(
            candidate_seasons["live_total_accounting_error"].max() <= 1e-9
        ),
    }
    if all(gates.values()):
        nested_decision = "PROMOTE_CANDIDATE"
    elif (
        float(candidate["brier_delta_vs_base"]) < 0.0
        and float(candidate["log_loss_delta_vs_base"]) < 0.0
        and gates["no_dependency_reliable_harm"]
        and gates["ranking_no_regression_all_evaluable_folds"]
    ):
        nested_decision = "KEEP_SHADOW"
    else:
        nested_decision = "KEEP_DISABLED"

    prespecified_envelopes = uncertainty.loc[
        uncertainty["model"].eq(MODEL_FIXED_12)
        & uncertainty["method"].eq("conservative_envelope")
    ].set_index("metric")
    prespecified_gates = {
        "brier_improves_at_least_4_of_6": int(prespecified["brier_fold_wins"]) >= 4,
        "log_loss_improves_at_least_4_of_6": int(prespecified["log_loss_fold_wins"]) >= 4,
        "pooled_brier_improves": float(prespecified["brier_delta_vs_base"]) < 0.0,
        "pooled_log_loss_improves": (
            float(prespecified["log_loss_delta_vs_base"]) < 0.0
        ),
        "dependency_reliable_improvement": bool(
            (prespecified_envelopes["ci_95_upper"] < 0.0).all()
        ),
        "no_dependency_reliable_harm": bool(
            (prespecified_envelopes["ci_95_lower"] <= 0.0).all()
        ),
        "ranking_no_regression_at_least_4_of_5": bool(
            int(prespecified["ranking_evaluable_folds"]) >= 5
            and int(prespecified["ranking_no_regression_folds"]) >= 4
        ),
        "no_competition_point_regression": bool(
            (prespecified_segments["brier_delta_vs_base"] <= 0.0).all()
            and (prespecified_segments["log_loss_delta_vs_base"] <= 0.0).all()
        ),
        "rank_correlation_guardrail": bool(
            prespecified_seasons["start_end_rank_correlation"].min()
            >= RANK_CORRELATION_FLOOR
        ),
        "maximum_movement_guardrail": bool(
            prespecified_seasons["max_abs_rating_change"].max()
            <= MAX_RATING_MOVE_GUARDRAIL
        ),
        "power_zero_sum_preserved": bool(
            prespecified_seasons[
                ["match_pair_zero_sum_error", "power_total_error"]
            ].to_numpy(float).max()
            <= 1e-9
        ),
        "bonus_cap_preserved": bool(
            prespecified_seasons["bonus_cap_error"].max() <= 1e-9
        ),
        "season_reset_preserved": bool(
            prespecified_seasons["season_start_bonus"].max() <= 1e-12
        ),
    }
    prespecified_promotion_gates = (
        "brier_improves_at_least_4_of_6",
        "log_loss_improves_at_least_4_of_6",
        "pooled_brier_improves",
        "pooled_log_loss_improves",
        "dependency_reliable_improvement",
        "ranking_no_regression_at_least_4_of_5",
        "no_competition_point_regression",
        "rank_correlation_guardrail",
        "maximum_movement_guardrail",
        "power_zero_sum_preserved",
        "bonus_cap_preserved",
        "season_reset_preserved",
    )
    if all(prespecified_gates[name] for name in prespecified_promotion_gates):
        prespecified_decision = "PROMOTE_CANDIDATE"
    elif all(
        prespecified_gates[name]
        for name in (
            "brier_improves_at_least_4_of_6",
            "log_loss_improves_at_least_4_of_6",
            "pooled_brier_improves",
            "pooled_log_loss_improves",
            "no_dependency_reliable_harm",
            "ranking_no_regression_at_least_4_of_5",
            "rank_correlation_guardrail",
            "maximum_movement_guardrail",
            "power_zero_sum_preserved",
            "bonus_cap_preserved",
            "season_reset_preserved",
        )
    ):
        prespecified_decision = "KEEP_SHADOW"
    else:
        prespecified_decision = "KEEP_DISABLED"
    decision = (
        "PROMOTE_CANDIDATE"
        if "PROMOTE_CANDIDATE" in (nested_decision, prespecified_decision)
        else "KEEP_SHADOW"
        if "KEEP_SHADOW" in (nested_decision, prespecified_decision)
        else "KEEP_DISABLED"
    )
    return {
        "decision": decision,
        "nested_selection_decision": nested_decision,
        "prespecified_12_8_4_decision": prespecified_decision,
        "candidate": MODEL_FIXED_12 if prespecified_decision == "KEEP_SHADOW" else MODEL_NESTED,
        "selected_nonzero_folds": nonzero_folds,
        "selected_base_bonuses": selections["selected_base_bonus"].tolist(),
        "prespecified_candidate": {
            "base_bonus": 12.0,
            "UCL": 12.0,
            "UEL": 8.0,
            "UECL": 4.0,
            "caps": {"UCL": 60.0, "UEL": 40.0, "UECL": 20.0},
        },
        "gates": gates,
        "prespecified_gates": prespecified_gates,
        "production_contract_changed": False,
    }


def write_report(
    path: Path,
    contract: dict[str, object],
    goal: dict[str, float | int],
    summary: pd.DataFrame,
    competition_summary: pd.DataFrame,
    surface: pd.DataFrame,
    activation: pd.DataFrame,
    uncertainty: pd.DataFrame,
    selections: pd.DataFrame,
    decision: dict[str, object],
) -> None:
    fixed = surface.loc[surface["base_bonus"].eq(12.0)].iloc[0]
    nested = summary.loc[summary["model"].eq(MODEL_NESTED)].iloc[0]
    path.write_text(
        f"""# Fixed Tournament Progress Bonus Backtest

## Model Decision

**{decision['decision']}**

Production contract was not changed.

- Nested parameter-selection decision: **{decision['nested_selection_decision']}**
- Prespecified 12/8/4 decision: **{decision['prespecified_12_8_4_decision']}**

## Tested Contract

- Existing AO First Elo and AO Power Elo are unchanged.
- Existing controlled goal difference remains active: alpha={goal['alpha']}, tau={goal['tau']}, cap={goal['goal_cap']}.
- Winner-only fixed progression bonus is added after a tie is fully decided.
- UCL/UEL/UECL ratio is 1 : 2/3 : 1/3.
- Eligible stages: knockout play-off, round of 16, quarterfinal, semifinal, final.
- The loser receives no deduction.
- Bonus is capped per competition and reset to zero at the next season start.
- League phase, top-eight placement and qualifying rounds receive no bonus.

The prespecified value is UCL +12, UEL +8 and UECL +4 per eligible stage,
with seasonal competition caps of 60/40/20.

## Nested Walk-Forward Result

- Development window: 2018/19-2025/26
- Outer folds: 6
- Untouched holdout: 2026/27
- Selected non-zero folds: {decision['selected_nonzero_folds']}/6
- Pooled Brier delta: {nested['brier_delta_vs_base']:.9f}
- Pooled log-loss delta: {nested['log_loss_delta_vs_base']:.9f}
- Brier fold wins: {int(nested['brier_fold_wins'])}/6
- Log-loss fold wins: {int(nested['log_loss_fold_wins'])}/6

## Prespecified 12/8/4 Sensitivity

- Brier delta: {fixed['brier_delta_vs_base']:.9f}
- Log-loss delta: {fixed['log_loss_delta_vs_base']:.9f}
- Ranking delta: {fixed['ranking_delta_vs_base']:.9f}
- Pairwise ranking delta: {fixed['pairwise_delta_vs_base']:.9f}
- Maximum team bonus observed: {fixed['maximum_team_bonus']:.3f}

## Fold Selections

```csv
{selections.to_csv(index=False).strip()}
```

## Model Comparison

```csv
{summary.to_csv(index=False).strip()}
```

## Competition Segments

```csv
{competition_summary.to_csv(index=False).strip()}
```

## Bonus Activation

```csv
{activation.to_csv(index=False).strip()}
```

## Fixed Candidate Surface

```csv
{surface.to_csv(index=False).strip()}
```

## Dependency-Robust Uncertainty

```csv
{uncertainty.to_csv(index=False).strip()}
```

## Guardrails

```json
{json.dumps(decision['gates'], indent=2)}
```

Prespecified 12/8/4 gates:

```json
{json.dumps(decision['prespecified_gates'], indent=2)}
```

## Interpretation

The progression bonus is intentionally non-zero-sum, while every match Power
Elo update remains zero-sum. It affects a prediction only when the two teams
arrive with different accumulated tournament bonuses, or after an earlier
bonus difference has changed their subsequent Power Elo path. A positive
business interpretation alone is not sufficient for activation; ranking,
probability quality, competition segments and uncertainty must agree.

Frozen model version: {contract['model_version']}.
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
