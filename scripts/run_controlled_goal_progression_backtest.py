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
from ao_elo.controlled_live import (  # noqa: E402
    apply_progression_bonus,
    calculate_goal_difference_multiplier,
    update_match_elo,
)
from ao_elo.evaluation import (  # noqa: E402
    dependency_robust_loss_difference_ci,
    schedule_adjusted_team_performance,
)
from ao_elo.robustness import one_x_two_probabilities_scalar  # noqa: E402
from scripts.run_dynamic_core_calibration import (  # noqa: E402
    DynamicCoreConfig,
    expanding_folds,
)
from scripts.run_final_robustness import (  # noqa: E402
    core_for_fold,
    draw_map_for_fold,
    production_core,
    production_draw_map,
    rank_value,
    read_final_production_contract,
    safe_rank_correlation,
    summarize_ranking,
    validate_fold_inputs,
)
from scripts.run_v2_achievement_reserve_calibration import (  # noqa: E402
    ReserveSeasonData,
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
PRODUCTION_MODEL_PATH = (
    ROOT / "output" / "final_robustness_2018_2026" / "selected_production_model.json"
)
OUTPUT_ROOT = ROOT / "output" / "controlled_goal_progression_backtest_2018_2026"

ALPHAS = (0.00, 0.05, 0.10, 0.15, 0.20)
TAUS = (150.0, 250.0, 300.0, 400.0, 500.0)
BASE_BONUSES = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0)
MODEL_KINDS = (
    "BASE",
    "PROGRESSION_ONLY",
    "GOAL_DIFFERENCE_ONLY",
    "FULL_MODEL",
)
PROGRESSION_STAGES = {
    "KNOCKOUT_PLAYOFF",
    "ROUND_OF_16",
    "QUARTERFINAL",
    "SEMIFINAL",
    "FINAL",
}
PRACTICAL_BRIER_THRESHOLD = 0.0005
SHADOW_BRIER_THRESHOLD = 0.0001
RANK_TOLERANCE = 1e-9


@dataclass(frozen=True, order=True)
class LayerCandidate:
    model_kind: str
    alpha: float
    tau: float
    base_bonus: float

    def validate(self) -> None:
        if self.model_kind not in MODEL_KINDS:
            raise ValueError(f"Unknown model_kind: {self.model_kind}")
        values = (self.alpha, self.tau, self.base_bonus)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Layer candidate parameters must be finite")
        if self.alpha < 0.0 or self.tau <= 0.0 or self.base_bonus < 0.0:
            raise ValueError("Layer candidate parameters are outside their valid range")
        if self.model_kind == "BASE" and (
            self.alpha != 0.0 or self.base_bonus != 0.0
        ):
            raise ValueError("BASE must disable both optional layers")
        if self.model_kind == "PROGRESSION_ONLY" and self.alpha != 0.0:
            raise ValueError("PROGRESSION_ONLY must set alpha=0")
        if self.model_kind == "GOAL_DIFFERENCE_ONLY" and self.base_bonus != 0.0:
            raise ValueError("GOAL_DIFFERENCE_ONLY must set base_bonus=0")

    @property
    def key(self) -> str:
        return (
            f"{self.model_kind.lower()}_a{self.alpha:g}"
            f"_t{self.tau:g}_b{self.base_bonus:g}"
        )

    @property
    def complexity(self) -> int:
        return int(self.alpha > 0.0) + int(self.base_bonus > 0.0)


@dataclass(frozen=True)
class ControlledSeasonData:
    reserve: ReserveSeasonData
    home_goals: np.ndarray
    away_goals: np.ndarray
    kickoff_utc: np.ndarray

    @property
    def season(self) -> str:
        return self.reserve.season


@dataclass
class BacktestEvaluation:
    metrics: dict[str, float | int]
    predictions: pd.DataFrame
    end_ratings: pd.DataFrame
    ranking: pd.DataFrame
    season_metrics: pd.DataFrame


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test controlled goal-difference and zero-sum progression layers "
            "without changing AO First Elo"
        )
    )
    parser.add_argument("--static-data-root", type=Path, default=STATIC_DATA_ROOT)
    parser.add_argument("--events-path", type=Path, default=EVENTS_PATH)
    parser.add_argument("--dynamic-root", type=Path, default=DYNAMIC_ROOT)
    parser.add_argument("--evaluation-root", type=Path, default=EVALUATION_ROOT)
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

    dynamic_root = args.dynamic_root.resolve()
    evaluation_root = args.evaluation_root.resolve()
    dynamic_manifest = json.loads(
        (dynamic_root / "selected_dynamic_model.json").read_text(encoding="utf-8")
    )
    static_config = AOEuropeanEloConfig(**dynamic_manifest["static_config"])
    static_config.validate()
    production = read_final_production_contract(args.production_model.resolve())
    if float(production["active_power_carry"]) != 0.0:
        raise ValueError("Controlled layer test requires the frozen carry=0 baseline")

    events = read_events(args.events_path.resolve())
    reserve_data, tie_audit = load_reserve_data(
        args.static_data_root.resolve(),
        args.events_path.resolve(),
        static_config,
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

    print("Controlled layer backtest: nested walk-forward", flush=True)
    nested = run_walk_forward_backtest(
        datasets,
        target,
        folds,
        core_selections,
        draw_selections,
    )

    print("Controlled layer backtest: fixed candidate OOS sensitivity", flush=True)
    fixed_metrics = run_fixed_candidate_grid(
        datasets,
        target,
        folds,
        core_selections,
        draw_selections,
    )

    predictions = nested["predictions"]
    fold_results = nested["fold_results"]
    model_comparison = build_model_comparison(predictions, fold_results)
    uncertainty = build_uncertainty(
        predictions,
        bootstrap_samples=args.bootstrap_samples,
    )
    competition_summary = segment_summary(predictions, "competition")
    match_band_summary = segment_summary(predictions, "match_band")
    calibration = calibration_analysis(predictions)
    rating_distribution = nested["rating_distribution"]
    conservation = build_conservation_summary(
        nested["season_metrics"],
        predictions,
    )
    decisions = decide_models(
        model_comparison,
        uncertainty,
        fold_results,
    )
    shadow_candidates = select_shadow_candidates(fixed_metrics)
    multiplier_examples = build_multiplier_examples(
        shadow_candidates,
        production_core(production),
    )

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    nested["fold_selections"].to_csv(
        output_root / "fold_selections.csv", index=False
    )
    fold_results.to_csv(output_root / "fold_results.csv", index=False)
    predictions.to_csv(output_root / "unseen_predictions.csv", index=False)
    nested["ranking"].to_csv(output_root / "forward_ranking.csv", index=False)
    fixed_metrics.to_csv(output_root / "fixed_candidate_oos_metrics.csv", index=False)
    model_comparison.to_csv(output_root / "model_comparison.csv", index=False)
    uncertainty.to_csv(output_root / "dependency_uncertainty.csv", index=False)
    competition_summary.to_csv(
        output_root / "competition_summary.csv", index=False
    )
    match_band_summary.to_csv(
        output_root / "match_band_summary.csv", index=False
    )
    calibration.to_csv(output_root / "calibration_analysis.csv", index=False)
    rating_distribution.to_csv(
        output_root / "season_rating_distribution.csv", index=False
    )
    conservation.to_csv(
        output_root / "elo_conservation_audit.csv", index=False
    )
    multiplier_examples.to_csv(
        output_root / "goal_multiplier_examples.csv", index=False
    )
    tie_audit.to_csv(output_root / "tie_chronology_audit.csv", index=False)

    data_quality = build_data_quality_summary(events, datasets, tie_audit)
    (output_root / "data_quality_summary.json").write_text(
        json.dumps(data_quality, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    selected_model = build_selected_model(
        production,
        decisions,
        shadow_candidates,
        model_comparison,
        uncertainty,
    )
    (output_root / "selected_model.json").write_text(
        json.dumps(selected_model, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_report(
        output_root / "backtest_report.md",
        seasons,
        model_comparison,
        uncertainty,
        competition_summary,
        match_band_summary,
        rating_distribution,
        conservation,
        decisions,
        shadow_candidates,
        nested["fold_selections"],
        data_quality,
    )

    print("AO controlled goal/progression backtest complete")
    for row in decisions.itertuples(index=False):
        print(f"{row.model}: {row.decision}")
    print(f"Report: {output_root / 'backtest_report.md'}")


def baseline_candidate(model_kind: str = "BASE") -> LayerCandidate:
    candidate = LayerCandidate(model_kind, 0.0, 300.0, 0.0)
    candidate.validate()
    return candidate


def selection_candidates(model_kind: str) -> tuple[LayerCandidate, ...]:
    if model_kind == "PROGRESSION_ONLY":
        values = (
            LayerCandidate(model_kind, 0.0, 300.0, bonus)
            for bonus in BASE_BONUSES
        )
    elif model_kind == "GOAL_DIFFERENCE_ONLY":
        values = (
            LayerCandidate(model_kind, alpha, tau, 0.0)
            for alpha in ALPHAS
            for tau in TAUS
            if alpha > 0.0 or tau == 300.0
        )
    elif model_kind == "FULL_MODEL":
        values = (
            LayerCandidate(model_kind, alpha, tau, bonus)
            for alpha in ALPHAS
            for tau in TAUS
            if alpha > 0.0 or tau == 300.0
            for bonus in BASE_BONUSES
        )
    elif model_kind == "BASE":
        values = (baseline_candidate(),)
    else:
        raise ValueError(f"Unknown model_kind: {model_kind}")
    candidates = tuple(values)
    for candidate in candidates:
        candidate.validate()
    return candidates


def reporting_candidates() -> tuple[LayerCandidate, ...]:
    candidates = [baseline_candidate()]
    candidates.extend(
        LayerCandidate("PROGRESSION_ONLY", 0.0, 300.0, bonus)
        for bonus in BASE_BONUSES
    )
    candidates.extend(
        LayerCandidate("GOAL_DIFFERENCE_ONLY", alpha, tau, 0.0)
        for alpha in ALPHAS
        for tau in TAUS
    )
    candidates.extend(
        LayerCandidate("FULL_MODEL", alpha, tau, bonus)
        for alpha in ALPHAS
        for tau in TAUS
        for bonus in BASE_BONUSES
    )
    result = tuple(candidates)
    for candidate in result:
        candidate.validate()
    return result


def prepare_controlled_data(
    datasets: tuple[ReserveSeasonData, ...],
    events: pd.DataFrame,
) -> tuple[ControlledSeasonData, ...]:
    event_index = events.set_index("match_id")
    result = []
    for data in datasets:
        match_ids = data.goal.carry.core.match_ids
        aligned = event_index.loc[match_ids]
        kickoff = pd.to_datetime(aligned["kickoff_utc"], utc=True, errors="raise")
        if not kickoff.is_monotonic_increasing:
            raise ValueError(f"{data.season}: matches are not in UTC chronology")
        result.append(
            ControlledSeasonData(
                reserve=data,
                home_goals=aligned["home_goals"].to_numpy(int),
                away_goals=aligned["away_goals"].to_numpy(int),
                kickoff_utc=kickoff.to_numpy(),
            )
        )
    return tuple(result)


def calculate_match_probabilities(
    expected_home_score: float,
    competition: str,
    draw_mapping: dict[str, DrawModelConfig],
) -> tuple[float, float, float]:
    draw = draw_mapping.get(competition, draw_mapping["ALL"])
    return one_x_two_probabilities_scalar(
        expected_home_score,
        draw.draw_at_even,
        draw.draw_shape,
    )


def evaluate_sequence(
    datasets: tuple[ControlledSeasonData, ...],
    core: DynamicCoreConfig,
    draw_mapping: dict[str, DrawModelConfig],
    target: pd.DataFrame,
    candidate: LayerCandidate,
    *,
    evaluation_seasons: set[str] | None = None,
    ranking_target_seasons: set[str] | None = None,
    return_predictions: bool = False,
) -> BacktestEvaluation:
    core.validate()
    candidate.validate()
    evaluation = evaluation_seasons or {data.season for data in datasets}
    prediction_rows: list[dict[str, object]] = []
    end_rows: list[dict[str, object]] = []
    season_rows: list[dict[str, object]] = []
    total_brier = 0.0
    total_log = 0.0
    total_correct = 0
    total_matches = 0

    for data in datasets:
        reserve = data.reserve
        goal = reserve.goal
        carry = goal.carry
        season_core = carry.core
        power = season_core.initial_ratings.copy()
        active_ids = season_core.active_team_ids
        initial_total = float(np.sum(power[active_ids]))
        processed_progressions: set[str] = set()
        match_deltas: list[float] = []
        progression_deltas: list[float] = []
        max_pair_error = 0.0
        max_progression_error = 0.0

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
            update = update_match_elo(
                float(power[home_id]),
                float(power[away_id]),
                int(data.home_goals[index]),
                int(data.away_goals[index]),
                k_factor=core.k_factor,
                elo_scale=core.elo_scale,
                home_advantage=core.home_advantage,
                is_neutral=bool(neutral),
                decided_on_penalties=bool(goal.penalty_flags[index]),
                alpha=candidate.alpha,
                tau=candidate.tau,
            )
            expected_actual = float(season_core.actual_home_scores[index])
            penalty_override = bool(goal.penalty_flags[index])
            if not penalty_override and not math.isclose(
                update.actual_home_score,
                expected_actual,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"{data.season}/{match_id}: score contract changed actual_home_score"
                )
            probabilities = calculate_match_probabilities(
                update.expected_home_score,
                competition,
                draw_mapping,
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
            max_pair_error = max(max_pair_error, update.zero_sum_error)

            progression_bonus = 0.0
            progression_winner = -1
            progression_loser = -1
            stage = str(reserve.stages[index])
            tie_value = reserve.tie_ids[index]
            tie_id = None if tie_value is None else str(tie_value)
            if (
                candidate.base_bonus > 0.0
                and bool(reserve.tie_decider_flags[index])
                and stage in PROGRESSION_STAGES
            ):
                if tie_id is None:
                    raise ValueError(f"{data.season}/{match_id}: missing deciding tie_id")
                if tie_id in processed_progressions:
                    raise ValueError(f"{data.season}/{tie_id}: duplicate progression bonus")
                winner_id = int(reserve.advanced_team_ids[index])
                if winner_id not in (home_id, away_id):
                    raise ValueError(
                        f"{data.season}/{tie_id}: advanced team is not a participant"
                    )
                loser_id = away_id if winner_id == home_id else home_id
                bonus_update = apply_progression_bonus(
                    float(power[winner_id]),
                    float(power[loser_id]),
                    competition,
                    candidate.base_bonus,
                )
                power[winner_id] = bonus_update.winner_rating_post
                power[loser_id] = bonus_update.loser_rating_post
                progression_bonus = bonus_update.applied_bonus
                progression_winner = winner_id
                progression_loser = loser_id
                progression_deltas.append(abs(progression_bonus))
                max_progression_error = max(
                    max_progression_error,
                    bonus_update.zero_sum_error,
                )
                processed_progressions.add(tie_id)

            if data.season in evaluation:
                total_matches += 1
                total_brier += brier
                total_log += log_loss
                total_correct += int(predicted == observed)
                if return_predictions:
                    prediction_rows.append(
                        {
                            "model": candidate.model_kind,
                            "candidate_key": candidate.key,
                            "match_id": match_id,
                            "season": data.season,
                            "kickoff_utc": pd.Timestamp(data.kickoff_utc[index]),
                            "competition": competition,
                            "round": str(carry.rounds[index]),
                            "stage": stage,
                            "tie_id": tie_id,
                            "home_team_id": home_id,
                            "away_team_id": away_id,
                            "home_goals": int(data.home_goals[index]),
                            "away_goals": int(data.away_goals[index]),
                            "decided_on_penalties": bool(goal.penalty_flags[index]),
                            "is_neutral": bool(neutral),
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
                            "progression_bonus_after_match": progression_bonus,
                            "progression_winner_id": progression_winner,
                            "progression_loser_id": progression_loser,
                            "alpha": candidate.alpha,
                            "tau": candidate.tau,
                            "base_bonus": candidate.base_bonus,
                        }
                    )

        if data.season in evaluation:
            end = power[active_ids]
            start = season_core.initial_ratings[active_ids]
            total_error = abs(float(np.sum(end)) - initial_total)
            season_rows.append(
                {
                    "model": candidate.model_kind,
                    "candidate_key": candidate.key,
                    "season": data.season,
                    "matches": len(season_core.match_ids),
                    "teams": len(active_ids),
                    "rating_min": float(np.min(end)),
                    "rating_max": float(np.max(end)),
                    "rating_mean": float(np.mean(end)),
                    "rating_std": float(np.std(end)),
                    "match_delta_mean_abs": float(np.mean(match_deltas)),
                    "match_delta_p95_abs": float(np.quantile(match_deltas, 0.95)),
                    "match_delta_max_abs": float(np.max(match_deltas)),
                    "progression_bonus_max_abs": (
                        max(progression_deltas) if progression_deltas else 0.0
                    ),
                    "max_abs_rating_change": float(np.max(np.abs(end - start))),
                    "start_end_rank_correlation": safe_rank_correlation(start, end),
                    "match_pair_zero_sum_error": max_pair_error,
                    "progression_pair_zero_sum_error": max_progression_error,
                    "season_total_elo_error": total_error,
                }
            )
            end_rows.extend(
                {
                    "model": candidate.model_kind,
                    "candidate_key": candidate.key,
                    "season": data.season,
                    "team_id": int(team_id),
                    "initial_rating": float(season_core.initial_ratings[team_id]),
                    "end_live_rating": float(power[team_id]),
                }
                for team_id in active_ids
            )

    if total_matches == 0:
        raise ValueError("No evaluation matches were processed")
    end_ratings = pd.DataFrame(end_rows)
    allowed_targets = ranking_target_seasons or set(target["season"])
    ranking = summarize_ranking(
        end_ratings,
        target,
        allowed_target_seasons=allowed_targets,
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
        "maximum_total_elo_error": float(
            season_metrics[
                [
                    "match_pair_zero_sum_error",
                    "progression_pair_zero_sum_error",
                    "season_total_elo_error",
                ]
            ].to_numpy(float).max()
        ),
    }
    predictions = pd.DataFrame(prediction_rows)
    if return_predictions:
        metrics.update(evaluate_predictions(predictions))
    return BacktestEvaluation(
        metrics=metrics,
        predictions=predictions,
        end_ratings=end_ratings,
        ranking=ranking,
        season_metrics=season_metrics,
    )


def evaluate_predictions(predictions: pd.DataFrame) -> dict[str, float | int]:
    required = {
        "brier_1x2",
        "log_loss_1x2",
        "actual_class",
        "predicted_class",
        "home_probability",
        "draw_probability",
        "away_probability",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction evaluation missing columns: {missing}")
    if predictions.empty:
        raise ValueError("Prediction evaluation requires at least one row")
    return {
        "matches": len(predictions),
        "brier_1x2": float(predictions["brier_1x2"].mean()),
        "log_loss_1x2": float(predictions["log_loss_1x2"].mean()),
        "accuracy_1x2": float(
            predictions["predicted_class"].eq(predictions["actual_class"]).mean()
        ),
        "multiclass_ece": multiclass_ece(predictions),
    }


def multiclass_ece(predictions: pd.DataFrame, bins: int = 10) -> float:
    if bins < 2:
        raise ValueError("bins must be at least two")
    eces = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for class_index, column in enumerate(
        ("home_probability", "draw_probability", "away_probability")
    ):
        probabilities = predictions[column].to_numpy(float)
        observed = predictions["actual_class"].eq(class_index).to_numpy(float)
        indices = np.minimum(np.digitize(probabilities, edges[1:-1]), bins - 1)
        ece = 0.0
        for bin_index in range(bins):
            mask = indices == bin_index
            if not mask.any():
                continue
            ece += (
                mask.mean()
                * abs(float(probabilities[mask].mean() - observed[mask].mean()))
            )
        eces.append(ece)
    return float(np.mean(eces))


def candidate_metric_row(
    candidate: LayerCandidate,
    evaluation: BacktestEvaluation,
) -> dict[str, object]:
    return {
        "model": candidate.model_kind,
        "candidate_key": candidate.key,
        "alpha": candidate.alpha,
        "tau": candidate.tau,
        "base_bonus": candidate.base_bonus,
        "complexity": candidate.complexity,
        **evaluation.metrics,
    }


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


def select_candidate(metrics: pd.DataFrame, baseline_key: str) -> pd.Series:
    baseline_rows = metrics.loc[metrics["candidate_key"].eq(baseline_key)]
    if len(baseline_rows) != 1:
        raise ValueError(f"Expected one baseline row for {baseline_key}")
    baseline = baseline_rows.iloc[0]
    eligible = metrics.loc[
        metrics.apply(candidate_is_safe, axis=1, baseline=baseline)
    ].copy()
    if eligible.empty:
        return baseline
    eligible["tau_preference"] = (eligible["tau"] - 300.0).abs()
    return eligible.sort_values(
        [
            "ranking_score",
            "pairwise_accuracy",
            "brier_1x2",
            "log_loss_1x2",
            "complexity",
            "alpha",
            "base_bonus",
            "tau_preference",
        ],
        ascending=[False, False, True, True, True, True, True, True],
        na_position="last",
    ).iloc[0]


def evaluate_candidate_set(
    datasets: tuple[ControlledSeasonData, ...],
    target: pd.DataFrame,
    core: DynamicCoreConfig,
    draw_mapping: dict[str, DrawModelConfig],
    candidates: tuple[LayerCandidate, ...],
) -> pd.DataFrame:
    rows = []
    baseline_kind = candidates[0].model_kind
    baseline = baseline_candidate(baseline_kind)
    evaluation_set = {data.season for data in datasets}
    for candidate in candidates:
        evaluation = evaluate_sequence(
            datasets,
            core,
            draw_mapping,
            target,
            candidate,
            evaluation_seasons=evaluation_set,
            ranking_target_seasons=evaluation_set,
        )
        rows.append(candidate_metric_row(candidate, evaluation))
    frame = pd.DataFrame(rows)
    if baseline.key not in set(frame["candidate_key"]):
        raise ValueError(f"Candidate set lacks baseline {baseline.key}")
    return frame


def candidate_from_row(row: pd.Series) -> LayerCandidate:
    candidate = LayerCandidate(
        str(row["model"]),
        float(row["alpha"]),
        float(row["tau"]),
        float(row["base_bonus"]),
    )
    candidate.validate()
    return candidate


def run_walk_forward_backtest(
    datasets: tuple[ControlledSeasonData, ...],
    target: pd.DataFrame,
    folds: tuple[tuple[tuple[str, ...], str], ...],
    core_selections: pd.DataFrame,
    draw_selections: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    by_season = {data.season: data for data in datasets}
    selection_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    ranking_frames: list[pd.DataFrame] = []
    season_frames: list[pd.DataFrame] = []

    for fold, (train_seasons, test_season) in enumerate(folds, start=1):
        core = core_for_fold(core_selections, fold)
        draw_mapping = draw_map_for_fold(draw_selections, fold)
        training_data = tuple(by_season[season] for season in train_seasons)
        test_data = (by_season[test_season],)

        selected: dict[str, LayerCandidate] = {"BASE": baseline_candidate()}
        for model_kind in MODEL_KINDS[1:]:
            candidates = selection_candidates(model_kind)
            train_metrics = evaluate_candidate_set(
                training_data,
                target,
                core,
                draw_mapping,
                candidates,
            )
            baseline_key = baseline_candidate(model_kind).key
            row = select_candidate(train_metrics, baseline_key)
            selected[model_kind] = candidate_from_row(row)
            selection_rows.append(
                {
                    "fold": fold,
                    "train_seasons": "|".join(train_seasons),
                    "test_season": test_season,
                    "model": model_kind,
                    "selected_candidate": selected[model_kind].key,
                    "selected_alpha": selected[model_kind].alpha,
                    "selected_tau": selected[model_kind].tau,
                    "selected_base_bonus": selected[model_kind].base_bonus,
                    "train_brier_1x2": float(row["brier_1x2"]),
                    "train_log_loss_1x2": float(row["log_loss_1x2"]),
                    "train_ranking_score": float(row["ranking_score"]),
                    "train_pairwise_accuracy": float(row["pairwise_accuracy"]),
                }
            )

        for model_kind in MODEL_KINDS:
            candidate = selected[model_kind]
            evaluation = evaluate_sequence(
                test_data,
                core,
                draw_mapping,
                target,
                candidate,
                evaluation_seasons={test_season},
                ranking_target_seasons=set(target["season"]),
                return_predictions=True,
            )
            predictions = evaluation.predictions.copy()
            predictions.insert(0, "fold", fold)
            predictions["model"] = model_kind
            predictions["candidate_key"] = candidate.key
            prediction_frames.append(predictions)
            season_metrics = evaluation.season_metrics.copy()
            season_metrics.insert(0, "fold", fold)
            season_metrics["model"] = model_kind
            season_frames.append(season_metrics)
            ranking = evaluation.ranking.copy()
            if not ranking.empty:
                ranking.insert(0, "fold", fold)
                ranking.insert(1, "test_season", test_season)
                ranking.insert(2, "model", model_kind)
                ranking_frames.append(ranking)
            fold_rows.append(
                {
                    "fold": fold,
                    "test_season": test_season,
                    "model": model_kind,
                    "candidate_key": candidate.key,
                    "alpha": candidate.alpha,
                    "tau": candidate.tau,
                    "base_bonus": candidate.base_bonus,
                    **evaluation.metrics,
                }
            )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    base_expected = predictions.loc[
        predictions["model"].eq("BASE"),
        ["match_id", "expected_home_score"],
    ].rename(columns={"expected_home_score": "base_expected_home_score"})
    predictions = predictions.merge(
        base_expected,
        on="match_id",
        how="left",
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
    rating_distribution = pd.concat(season_frames, ignore_index=True)
    return {
        "fold_selections": pd.DataFrame(selection_rows),
        "fold_results": pd.DataFrame(fold_rows),
        "predictions": predictions,
        "ranking": (
            pd.concat(ranking_frames, ignore_index=True)
            if ranking_frames
            else pd.DataFrame()
        ),
        "season_metrics": rating_distribution,
        "rating_distribution": rating_distribution[
            [
                "fold",
                "model",
                "candidate_key",
                "season",
                "teams",
                "rating_min",
                "rating_max",
                "rating_mean",
                "rating_std",
                "match_delta_mean_abs",
                "match_delta_p95_abs",
                "match_delta_max_abs",
                "progression_bonus_max_abs",
                "max_abs_rating_change",
                "start_end_rank_correlation",
            ]
        ].copy(),
    }


def run_fixed_candidate_grid(
    datasets: tuple[ControlledSeasonData, ...],
    target: pd.DataFrame,
    folds: tuple[tuple[tuple[str, ...], str], ...],
    core_selections: pd.DataFrame,
    draw_selections: pd.DataFrame,
) -> pd.DataFrame:
    by_season = {data.season: data for data in datasets}
    rows = []
    candidates = reporting_candidates()
    for index, candidate in enumerate(candidates, start=1):
        prediction_frames = []
        ranking_frames = []
        season_frames = []
        for fold, (_, test_season) in enumerate(folds, start=1):
            evaluation = evaluate_sequence(
                (by_season[test_season],),
                core_for_fold(core_selections, fold),
                draw_map_for_fold(draw_selections, fold),
                target,
                candidate,
                evaluation_seasons={test_season},
                ranking_target_seasons=set(target["season"]),
                return_predictions=True,
            )
            prediction_frames.append(evaluation.predictions)
            season_frames.append(evaluation.season_metrics)
            if not evaluation.ranking.empty:
                ranking_frames.append(evaluation.ranking)
        predictions = pd.concat(prediction_frames, ignore_index=True)
        seasons = pd.concat(season_frames, ignore_index=True)
        rankings = (
            pd.concat(ranking_frames, ignore_index=True)
            if ranking_frames
            else pd.DataFrame()
        )
        ranking_all = aggregate_rankings(rankings)
        metrics = evaluate_predictions(predictions)
        rows.append(
            {
                "model": candidate.model_kind,
                "candidate_key": candidate.key,
                "alpha": candidate.alpha,
                "tau": candidate.tau,
                "base_bonus": candidate.base_bonus,
                "complexity": candidate.complexity,
                **metrics,
                "ranking_score": rank_value(ranking_all, "ranking_score"),
                "pairwise_accuracy": rank_value(
                    ranking_all,
                    "pairwise_accuracy",
                ),
                "minimum_start_end_rank_correlation": float(
                    seasons["start_end_rank_correlation"].min()
                ),
                "maximum_abs_rating_change": float(
                    seasons["max_abs_rating_change"].max()
                ),
                "maximum_abs_match_delta": float(
                    seasons["match_delta_max_abs"].max()
                ),
                "maximum_total_elo_error": float(
                    seasons[
                        [
                            "match_pair_zero_sum_error",
                            "progression_pair_zero_sum_error",
                            "season_total_elo_error",
                        ]
                    ].to_numpy(float).max()
                ),
            }
        )
        if index % 25 == 0:
            print(
                f"Fixed OOS candidates: {index}/{len(candidates)}",
                flush=True,
            )
    return pd.DataFrame(rows)


def aggregate_rankings(rankings: pd.DataFrame) -> pd.DataFrame:
    if rankings.empty:
        return pd.DataFrame(
            columns=[
                "competition",
                "groups",
                "team_weight",
                "ranking_score",
                "pairwise_accuracy",
            ]
        )
    rows = []
    for competition, frame in rankings.groupby("competition", sort=True):
        team_weights = frame["team_weight"].to_numpy(float)
        pair_weights = np.maximum(team_weights * (team_weights - 1.0) / 2.0, 1.0)
        rows.append(
            {
                "competition": competition,
                "groups": int(frame["groups"].sum()),
                "team_weight": int(frame["team_weight"].sum()),
                "ranking_score": float(
                    np.average(frame["ranking_score"], weights=team_weights)
                ),
                "pairwise_accuracy": float(
                    np.average(frame["pairwise_accuracy"], weights=pair_weights)
                ),
            }
        )
    return pd.DataFrame(rows)


def build_model_comparison(
    predictions: pd.DataFrame,
    fold_results: pd.DataFrame,
) -> pd.DataFrame:
    base = predictions.loc[predictions["model"].eq("BASE")]
    base_metrics = evaluate_predictions(base)
    base_fold = fold_results.loc[fold_results["model"].eq("BASE")].set_index("fold")
    rows = []
    for model, frame in predictions.groupby("model", sort=False):
        metrics = evaluate_predictions(frame)
        folds = fold_results.loc[fold_results["model"].eq(model)].set_index("fold")
        common = folds.index.intersection(base_fold.index)
        rank_mask = (
            folds.loc[common, "ranking_score"].notna()
            & base_fold.loc[common, "ranking_score"].notna()
        )
        rank_index = common[rank_mask]
        no_regression = (
            (
                folds.loc[rank_index, "ranking_score"]
                >= base_fold.loc[rank_index, "ranking_score"] - RANK_TOLERANCE
            )
            & (
                folds.loc[rank_index, "pairwise_accuracy"]
                >= base_fold.loc[rank_index, "pairwise_accuracy"] - RANK_TOLERANCE
            )
        )
        both_improved = (
            (
                folds.loc[rank_index, "ranking_score"]
                > base_fold.loc[rank_index, "ranking_score"] + RANK_TOLERANCE
            )
            & (
                folds.loc[rank_index, "pairwise_accuracy"]
                > base_fold.loc[rank_index, "pairwise_accuracy"] + RANK_TOLERANCE
            )
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
                "ranking_evaluable_folds": len(rank_index),
                "ranking_no_regression_folds": int(no_regression.sum()),
                "ranking_both_improved_folds": int(both_improved.sum()),
                "maximum_abs_match_delta": float(
                    folds["maximum_abs_match_delta"].max()
                ),
                "maximum_abs_rating_change": float(
                    folds["maximum_abs_rating_change"].max()
                ),
                "maximum_total_elo_error": float(
                    folds["maximum_total_elo_error"].max()
                ),
            }
        )
    order = {model: index for index, model in enumerate(MODEL_KINDS)}
    result = pd.DataFrame(rows)
    result["_order"] = result["model"].map(order)
    return result.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def paired_loss_frame(
    predictions: pd.DataFrame,
    model: str,
    metric: str,
) -> pd.DataFrame:
    base = predictions.loc[
        predictions["model"].eq("BASE"),
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
        predictions["model"].eq(model),
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
    return paired


def build_uncertainty(
    predictions: pd.DataFrame,
    *,
    bootstrap_samples: int,
) -> pd.DataFrame:
    frames = []
    for model in MODEL_KINDS[1:]:
        for metric in ("brier_1x2", "log_loss_1x2"):
            paired = paired_loss_frame(predictions, model, metric)
            uncertainty = dependency_robust_loss_difference_ci(
                paired,
                bootstrap_samples=bootstrap_samples,
                seed=20260723 + MODEL_KINDS.index(model) * 10007,
            )
            uncertainty.insert(0, "model", model)
            uncertainty.insert(1, "metric", metric)
            frames.append(uncertainty)
    return pd.concat(frames, ignore_index=True)


def segment_summary(predictions: pd.DataFrame, segment: str) -> pd.DataFrame:
    rows = []
    for (model, value), frame in predictions.groupby(["model", segment], sort=True):
        metrics = evaluate_predictions(frame)
        rows.append({"model": model, segment: value, **metrics})
    result = pd.DataFrame(rows)
    base = result.loc[result["model"].eq("BASE")].set_index(segment)
    deltas = []
    for row in result.itertuples(index=False):
        base_row = base.loc[getattr(row, segment)]
        values = row._asdict()
        values["brier_delta_vs_base"] = (
            float(values["brier_1x2"]) - float(base_row["brier_1x2"])
        )
        values["log_loss_delta_vs_base"] = (
            float(values["log_loss_1x2"]) - float(base_row["log_loss_1x2"])
        )
        deltas.append(values)
    return pd.DataFrame(deltas)


def calibration_analysis(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    class_columns = (
        ("HOME", 0, "home_probability"),
        ("DRAW", 1, "draw_probability"),
        ("AWAY", 2, "away_probability"),
    )
    edges = np.linspace(0.0, 1.0, 11)
    for model, frame in predictions.groupby("model", sort=False):
        for label, class_index, column in class_columns:
            probabilities = frame[column].to_numpy(float)
            observed = frame["actual_class"].eq(class_index).to_numpy(float)
            indices = np.minimum(np.digitize(probabilities, edges[1:-1]), 9)
            for bin_index in range(10):
                mask = indices == bin_index
                if not mask.any():
                    continue
                rows.append(
                    {
                        "model": model,
                        "outcome": label,
                        "probability_bin": (
                            f"{edges[bin_index]:.1f}-{edges[bin_index + 1]:.1f}"
                        ),
                        "matches": int(mask.sum()),
                        "mean_predicted_probability": float(
                            probabilities[mask].mean()
                        ),
                        "observed_rate": float(observed[mask].mean()),
                        "calibration_gap": float(
                            probabilities[mask].mean() - observed[mask].mean()
                        ),
                    }
                )
    return pd.DataFrame(rows)


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
                "maximum_progression_pair_error": float(
                    frame["progression_pair_zero_sum_error"].max()
                ),
                "maximum_season_total_error": float(
                    frame["season_total_elo_error"].max()
                ),
                "progression_events": int(
                    model_predictions["progression_bonus_after_match"].gt(0).sum()
                ),
                "maximum_goal_multiplier": float(
                    model_predictions["goal_multiplier"].max()
                ),
                "maximum_abs_power_delta": float(
                    model_predictions["power_delta"].abs().max()
                ),
            }
        )
    return pd.DataFrame(rows)


def decide_models(
    comparison: pd.DataFrame,
    uncertainty: pd.DataFrame,
    fold_results: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for model in MODEL_KINDS[1:]:
        summary = comparison.loc[comparison["model"].eq(model)].iloc[0]
        envelopes = uncertainty.loc[
            uncertainty["model"].eq(model)
            & uncertainty["method"].eq("conservative_envelope")
        ].set_index("metric")
        brier_ci = envelopes.loc["brier_1x2"]
        log_ci = envelopes.loc["log_loss_1x2"]
        point_improvement = bool(
            summary["brier_delta_vs_base"] < 0.0
            and summary["log_loss_delta_vs_base"] < 0.0
        )
        reliable_improvement = bool(
            brier_ci["ci_95_upper"] < 0.0
            and log_ci["ci_95_upper"] < 0.0
        )
        reliable_harm = bool(
            brier_ci["ci_95_lower"] > 0.0
            or log_ci["ci_95_lower"] > 0.0
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
        if (
            point_improvement
            and reliable_improvement
            and practical
            and ranking_safe
            and ranking_improves
        ):
            decision = "ACTIVE"
        elif (
            point_improvement
            and not reliable_harm
            and (
                reliable_improvement
                or summary["brier_delta_vs_base"] <= -SHADOW_BRIER_THRESHOLD
            )
        ):
            decision = "SHADOW"
        else:
            decision = "REJECT"
        rows.append(
            {
                "model": model,
                "decision": decision,
                "point_improvement": point_improvement,
                "reliable_improvement": reliable_improvement,
                "reliable_harm": reliable_harm,
                "practical_brier_improvement": practical,
                "ranking_safe_all_evaluable_folds": ranking_safe,
                "ranking_both_improved_at_least_4": ranking_improves,
            }
        )
    return pd.DataFrame(rows)


def select_shadow_candidates(fixed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_kind in MODEL_KINDS[1:]:
        candidates = fixed_metrics.loc[fixed_metrics["model"].eq(model_kind)].copy()
        baseline_key = baseline_candidate(model_kind).key
        row = select_candidate(candidates, baseline_key)
        rows.append(row.to_dict())
    return pd.DataFrame(rows)


def build_multiplier_examples(
    shadow_candidates: pd.DataFrame,
    core: DynamicCoreConfig,
) -> pd.DataFrame:
    goal_row = shadow_candidates.loc[
        shadow_candidates["model"].eq("GOAL_DIFFERENCE_ONLY")
    ].iloc[0]
    alpha = float(goal_row["alpha"])
    tau = float(goal_row["tau"])
    rows = []
    for difference in (0, 1, 2, 3, 4, 5, 6):
        for effective_difference in (
            0.0,
            core.home_advantage,
            300.0,
            600.0,
        ):
            rows.append(
                {
                    "alpha": alpha,
                    "tau": tau,
                    "goal_difference": difference,
                    "effective_rating_difference": effective_difference,
                    "goal_multiplier": calculate_goal_difference_multiplier(
                        difference,
                        effective_difference,
                        alpha,
                        tau,
                        is_draw=difference == 0,
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_data_quality_summary(
    events: pd.DataFrame,
    datasets: tuple[ControlledSeasonData, ...],
    tie_audit: pd.DataFrame,
) -> dict[str, object]:
    eligible_deciders = 0
    for data in datasets:
        eligible_deciders += int(
            np.sum(
                data.reserve.tie_decider_flags
                & np.isin(data.reserve.stages, tuple(PROGRESSION_STAGES))
            )
        )
    return {
        "seasons": [data.season for data in datasets],
        "matches": int(len(events)),
        "unique_match_ids": int(events["match_id"].nunique()),
        "duplicate_match_ids": int(events["match_id"].duplicated().sum()),
        "competitions": sorted(events["competition"].unique().tolist()),
        "penalty_decisions": int(events["decided_on_penalties"].astype(bool).sum()),
        "penalty_decisions_with_non_draw_field_score": int(
            (
                events["decided_on_penalties"].astype(bool)
                & events["home_goals"].ne(events["away_goals"])
            ).sum()
        ),
        "penalty_score_override": "S_home=0.5 and M_GD=1 for every shootout",
        "eligible_progression_deciders": eligible_deciders,
        "tie_decider_chronology_corrections": int(len(tie_audit)),
        "kickoff_utc_min": events["kickoff_utc"].min().isoformat(),
        "kickoff_utc_max": events["kickoff_utc"].max().isoformat(),
        "ao_first_elo_rebuilt_per_season": True,
        "power_carry": 0.0,
        "future_season_data_in_ao_first_elo": False,
    }


def build_selected_model(
    production: dict[str, object],
    decisions: pd.DataFrame,
    shadow_candidates: pd.DataFrame,
    comparison: pd.DataFrame,
    uncertainty: pd.DataFrame,
) -> dict[str, object]:
    active_alpha = 0.0
    active_tau = 300.0
    active_bonus = 0.0
    goal_decision = decisions.loc[
        decisions["model"].eq("GOAL_DIFFERENCE_ONLY"), "decision"
    ].iloc[0]
    progression_decision = decisions.loc[
        decisions["model"].eq("PROGRESSION_ONLY"), "decision"
    ].iloc[0]
    full_decision = decisions.loc[
        decisions["model"].eq("FULL_MODEL"), "decision"
    ].iloc[0]
    if full_decision == "ACTIVE":
        row = shadow_candidates.loc[
            shadow_candidates["model"].eq("FULL_MODEL")
        ].iloc[0]
        active_alpha = float(row["alpha"])
        active_tau = float(row["tau"])
        active_bonus = float(row["base_bonus"])
    else:
        if goal_decision == "ACTIVE":
            row = shadow_candidates.loc[
                shadow_candidates["model"].eq("GOAL_DIFFERENCE_ONLY")
            ].iloc[0]
            active_alpha = float(row["alpha"])
            active_tau = float(row["tau"])
        if progression_decision == "ACTIVE":
            row = shadow_candidates.loc[
                shadow_candidates["model"].eq("PROGRESSION_ONLY")
            ].iloc[0]
            active_bonus = float(row["base_bonus"])
    return {
        "model_version": production["model_version"],
        "ao_first_elo_changed": False,
        "dynamic_core": production["dynamic_core"],
        "active_power_carry": production["active_power_carry"],
        "one_x_two_probability": production["one_x_two_probability"],
        "active_controlled_layers": {
            "goal_difference": {
                "active": active_alpha > 0.0,
                "alpha": active_alpha,
                "tau": active_tau,
                "goal_difference_cap": 4,
            },
            "progression_bonus": {
                "active": active_bonus > 0.0,
                "ucl_base_bonus": active_bonus,
                "uel_ratio": 2.0 / 3.0,
                "uecl_ratio": 1.0 / 3.0,
                "eligible_stages": sorted(PROGRESSION_STAGES),
                "zero_sum": True,
            },
        },
        "decisions": {
            row.model: row.decision
            for row in decisions.itertuples(index=False)
        },
        "shadow_candidates": shadow_candidates[
            ["model", "alpha", "tau", "base_bonus"]
        ].to_dict(orient="records"),
        "model_comparison": comparison.to_dict(orient="records"),
        "dependency_uncertainty": uncertainty.loc[
            uncertainty["method"].eq("conservative_envelope"),
            [
                "model",
                "metric",
                "mean_difference",
                "ci_95_lower",
                "ci_95_upper",
                "reliable_improvement",
                "reliable_harm",
            ],
        ].to_dict(orient="records"),
        "development_data_through": "2025/26",
        "untouched_holdout": "2026/27",
    }


def write_report(
    path: Path,
    seasons: tuple[str, ...],
    comparison: pd.DataFrame,
    uncertainty: pd.DataFrame,
    competition: pd.DataFrame,
    match_bands: pd.DataFrame,
    rating_distribution: pd.DataFrame,
    conservation: pd.DataFrame,
    decisions: pd.DataFrame,
    shadow_candidates: pd.DataFrame,
    fold_selections: pd.DataFrame,
    data_quality: dict[str, object],
) -> None:
    def value(model: str, column: str) -> float:
        return float(comparison.loc[comparison["model"].eq(model), column].iloc[0])

    def envelope(model: str, metric: str) -> tuple[float, float]:
        row = uncertainty.loc[
            uncertainty["model"].eq(model)
            & uncertainty["metric"].eq(metric)
            & uncertainty["method"].eq("conservative_envelope")
        ].iloc[0]
        return float(row["ci_95_lower"]), float(row["ci_95_upper"])

    def chosen(model: str) -> pd.Series:
        return shadow_candidates.loc[shadow_candidates["model"].eq(model)].iloc[0]

    lines = [
        "# Kontrollü Gol Farkı ve Tur Bonusu Backtesti",
        "",
        "## Kapsam",
        "",
        f"- Sezonlar: `{seasons[0]}-{seasons[-1]}`.",
        f"- Tarihsel maç: `{data_quality['matches']}`.",
        "- AO First Elo mimarisi ve sezonluk başlangıç ratingleri değiştirilmedi.",
        "- Scale, H, K, carry=0 ve 1X2 draw mapping mevcut nested fold sözleşmesinden alındı.",
        "- Her dış fold için yeni alpha/tau/base_bonus seçimi yalnızca önceki sezonlarda yapıldı.",
        "- Maçlar kesin UTC sırasıyla işlendi; 2026/27 parametre seçiminde kullanılmadı.",
        f"- Penaltıyla biten `{data_quality['penalty_decisions']}` eşleşmede "
        "`S_home=0.5` ve `M_GD=1` uygulandı; bunların "
        f"`{data_quality['penalty_decisions_with_non_draw_field_score']}` tanesinde "
        "tek maçın 90/120 skoru eşit değildi ve kullanıcı sözleşmesi bilinçli override edildi.",
        "",
        "## Formüller",
        "",
        "```text",
        "D = R_home - R_away + H",
        "M_GD = 1 + alpha * ln(min(GD, 4)) * exp(-abs(D) / tau)",
        "Delta = K * (S_home - E_home) * M_GD",
        "```",
        "",
        "- Beraberlikte ve penaltı kararında `GD=0`, `M_GD=1`, `S_home=0.5`.",
        "- Progression bonusu eşleşme sonunda bir kez, kazanana `+B` ve elenene `-B` uygulanır.",
        "- UCL/UEL/UECL oranı `1 : 2/3 : 1/3`; qualifying ve lig aşaması bonus dışıdır.",
        "",
        "## Out-of-sample Model Karşılaştırması",
        "",
        "| Model | Brier | Δ Brier | Log loss | Δ Log | Accuracy | ECE | Rank güvenli |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in MODEL_KINDS:
        row = comparison.loc[comparison["model"].eq(model)].iloc[0]
        lines.append(
            f"| {model} | {row['brier_1x2']:.6f} | "
            f"{row['brier_delta_vs_base']:+.6f} | {row['log_loss_1x2']:.6f} | "
            f"{row['log_loss_delta_vs_base']:+.6f} | {row['accuracy_1x2']:.4f} | "
            f"{row['multiclass_ece']:.4f} | "
            f"{int(row['ranking_no_regression_folds'])}/"
            f"{int(row['ranking_evaluable_folds'])} |"
        )

    lines.extend(["", "## İstatistiksel Belirsizlik", ""])
    for model in MODEL_KINDS[1:]:
        brier_ci = envelope(model, "brier_1x2")
        log_ci = envelope(model, "log_loss_1x2")
        lines.extend(
            [
                f"- **{model}:** Brier zarfı "
                f"`[{brier_ci[0]:+.6f}, {brier_ci[1]:+.6f}]`; "
                f"log-loss zarfı `[{log_ci[0]:+.6f}, {log_ci[1]:+.6f}]`.",
            ]
        )

    lines.extend(["", "## Seçilen Geliştirme Adayları", ""])
    for model in MODEL_KINDS[1:]:
        row = chosen(model)
        selections = fold_selections.loc[fold_selections["model"].eq(model)]
        lines.append(
            f"- **{model}:** fixed OOS ranking-first aday "
            f"`alpha={row['alpha']:g}, tau={row['tau']:g}, "
            f"base_bonus={row['base_bonus']:g}`. "
            f"Altı nested fold seçimi: "
            f"`{' | '.join(selections['selected_candidate'].astype(str))}`."
        )

    lines.extend(["", "## Turnuva Segmentleri", ""])
    lines.append("| Model | Turnuva | Maç | Δ Brier | Δ Log loss |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for row in competition.itertuples(index=False):
        if row.model == "BASE":
            continue
        lines.append(
            f"| {row.model} | {row.competition} | {row.matches} | "
            f"{row.brier_delta_vs_base:+.6f} | "
            f"{row.log_loss_delta_vs_base:+.6f} |"
        )

    lines.extend(["", "## Güç Bantları", ""])
    lines.append("| Model | Bant | Maç | Δ Brier | Δ Log loss |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for row in match_bands.itertuples(index=False):
        if row.model == "BASE":
            continue
        lines.append(
            f"| {row.model} | {row.match_band} | {row.matches} | "
            f"{row.brier_delta_vs_base:+.6f} | "
            f"{row.log_loss_delta_vs_base:+.6f} |"
        )

    lines.extend(["", "## Rating Güvenliği", ""])
    lines.append(
        "| Model | Min rating | Max rating | Max maç delta | Max sezon hareketi | Elo korunumu |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for model in MODEL_KINDS:
        distribution = rating_distribution.loc[rating_distribution["model"].eq(model)]
        audit = conservation.loc[conservation["model"].eq(model)].iloc[0]
        lines.append(
            f"| {model} | {distribution['rating_min'].min():.3f} | "
            f"{distribution['rating_max'].max():.3f} | "
            f"{audit['maximum_abs_power_delta']:.3f} | "
            f"{distribution['max_abs_rating_change'].max():.3f} | "
            f"{audit['maximum_season_total_error']:.3e} |"
        )

    lines.extend(["", "## Altı Sorunun Cevabı", ""])
    goal = chosen("GOAL_DIFFERENCE_ONLY")
    progression = chosen("PROGRESSION_ONLY")
    full = chosen("FULL_MODEL")
    lines.extend(
        [
            "1. **Gol farkı Brier ve log-loss'u iyileştiriyor mu?** "
            + evidence_sentence(
                "GOAL_DIFFERENCE_ONLY",
                comparison,
                uncertainty,
            ),
            f"2. **En iyi alpha/tau hangisi?** Geliştirme verisindeki ranking-first "
            f"sensitivity adayı `alpha={goal['alpha']:g}, tau={goal['tau']:g}`. "
            "Nested seçim kararlılığı ve terfi kararı aşağıdaki model kararında dikkate alındı.",
            "3. **Tur bonusu tek başına faydalı mı?** "
            + evidence_sentence(
                "PROGRESSION_ONLY",
                comparison,
                uncertainty,
            )
            + f" En iyi sensitivity base_bonus değeri `{progression['base_bonus']:g}`.",
            "4. **Full model anlamlı iyileşiyor mu?** "
            + evidence_sentence("FULL_MODEL", comparison, uncertainty)
            + f" Sensitivity adayı `alpha={full['alpha']:g}, tau={full['tau']:g}, "
            f"base_bonus={full['base_bonus']:g}`.",
            "5. **Fark küçükse sade model mi?** Evet. Promotion için hem dependency "
            "zarfının iyileşme yönünde olması hem pratik eşik hem de forward-ranking "
            "guardrail'lerinin geçilmesi zorunludur.",
            "6. **Nihai parametre?** Aktif parametreler aşağıdaki veri temelli karar "
            "tablosuna göre belirlenmiştir; geçmeyen katmanlar üretim ratingine eklenmez.",
        ]
    )

    lines.extend(["", "## Model Kararı", ""])
    lines.append("| Bileşen | Karar |")
    lines.append("| --- | --- |")
    for row in decisions.itertuples(index=False):
        lines.append(f"| {row.model} | **{row.decision}** |")
    active = decisions.loc[decisions["decision"].eq("ACTIVE"), "model"].tolist()
    shadow = decisions.loc[decisions["decision"].eq("SHADOW"), "model"].tolist()
    rejected = decisions.loc[decisions["decision"].eq("REJECT"), "model"].tolist()
    lines.extend(
        [
            "",
            f"- Aktif modele alınacak: `{', '.join(active) if active else 'yalnızca BASE'}`.",
            f"- Shadow mode: `{', '.join(shadow) if shadow else 'yok'}`.",
            f"- Reddedilen: `{', '.join(rejected) if rejected else 'yok'}`.",
            "- Aktif kontrollü katman parametreleri: `alpha=0`, `tau=300` "
            "(etkisiz kontrol), `base_bonus=0`.",
            "- AO First Elo ve dondurulmuş klasik canlı çekirdek bu deneyden bağımsızdır.",
            "",
            "## Veri ve Metodoloji Sınırları",
            "",
            "- Fold'lar bağımsız tekrar değildir; dependency bootstrap sonuçları "
            "tie/match, team-season ve calendar-month duyarlılık zarfıdır.",
            "- Fixed-candidate OOS tablosu parametre duyarlılığıdır; dürüst model kararı "
            "yalnızca training penceresinde seçilen nested OOS karşılaştırmasına dayanır.",
            "- 2026/27 prospective holdout parametre seçimi için kullanılmayacaktır.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evidence_sentence(
    model: str,
    comparison: pd.DataFrame,
    uncertainty: pd.DataFrame,
) -> str:
    row = comparison.loc[comparison["model"].eq(model)].iloc[0]
    brier = uncertainty.loc[
        uncertainty["model"].eq(model)
        & uncertainty["metric"].eq("brier_1x2")
        & uncertainty["method"].eq("conservative_envelope")
    ].iloc[0]
    log_loss = uncertainty.loc[
        uncertainty["model"].eq(model)
        & uncertainty["metric"].eq("log_loss_1x2")
        & uncertainty["method"].eq("conservative_envelope")
    ].iloc[0]
    return (
        f"Nested OOS ΔBrier `{row['brier_delta_vs_base']:+.6f}`, "
        f"Δlog-loss `{row['log_loss_delta_vs_base']:+.6f}`; "
        f"Brier zarfı `[{brier['ci_95_lower']:+.6f}, "
        f"{brier['ci_95_upper']:+.6f}]`, log-loss zarfı "
        f"`[{log_loss['ci_95_lower']:+.6f}, {log_loss['ci_95_upper']:+.6f}]`."
    )


if __name__ == "__main__":
    main()
