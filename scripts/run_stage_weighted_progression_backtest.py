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
from ao_elo.evaluation import (  # noqa: E402
    dependency_robust_loss_difference_ci,
    schedule_adjusted_team_performance,
)
from ao_elo.robustness import one_x_two_probabilities_scalar  # noqa: E402
from ao_elo.tournament_bonus import (  # noqa: E402
    ELIGIBLE_PROGRESSION_STAGES,
    STAGE_WEIGHTED_PROGRESSION_STAGES,
    FiveStageWeightedTournamentBonusConfig,
    FixedTournamentBonusConfig,
    StageWeightedTournamentBonusConfig,
    apply_stage_weighted_tournament_bonus,
    apply_five_stage_weighted_tournament_bonus,
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
from scripts.run_dynamic_core_calibration import (  # noqa: E402
    DynamicCoreConfig,
    expanding_folds,
)
from scripts.run_final_robustness import (  # noqa: E402
    load_team_season_identity,
    pairwise_ranking_accuracy,
    safe_rank_correlation,
    summarize_ranking,
)
from scripts.run_v2_achievement_reserve_calibration import (  # noqa: E402
    load_reserve_data,
)
from scripts.run_v2_evaluation_upgrade import read_events  # noqa: E402


STATIC_DATA_ROOT = ROOT / "data" / "backtest_stage_b_2018_2026"
EVENTS_PATH = (
    ROOT / "data" / "external_elo_benchmark_2018_2026" / "exact_date_events.csv"
)
DYNAMIC_MANIFEST = (
    ROOT / "output" / "v2_dynamic_calibration_2018_2026" / "selected_dynamic_model.json"
)
PRODUCTION_CONTRACT = ROOT / "contracts" / "ao_european_elo_v2_production.json"
DOMESTIC_ADJUSTMENTS = (
    ROOT
    / "output"
    / "domestic_surprise_variance_backtest_2018_2026"
    / "gamma_sensitivity_cap_30"
    / "selected_candidate_team_adjustments.csv"
)
XG_DATA = ROOT / "data" / "xg_2025_26" / "uefa_2025_26_matches_with_xg.csv"
OUTPUT_ROOT = ROOT / "output" / "stage_weighted_progression_backtest_2018_2026"

UCL_TOTAL_CAPS = (30.0, 45.0, 60.0, 75.0, 90.0)
PROFILE_WEIGHTS = {
    "GENTLE": (0.15, 0.20, 0.30, 0.35),
    "LINEAR": (0.10, 0.20, 0.30, 0.40),
    "FINAL_HEAVY": (0.05, 0.15, 0.30, 0.50),
    "EQUAL_FOUR": (0.25, 0.25, 0.25, 0.25),
}
COMPETITION_INDEX = {"UCL": 0, "UEL": 1, "UECL": 2}
MODEL_CURRENT = "CURRENT_FIXED_12_8_4_X5"
MODEL_NONE = "NO_PROGRESSION"
MODEL_NESTED = "NESTED_STAGE_WEIGHTED"
RANK_TOLERANCE = 1e-9
MAX_RATING_MOVE = 742.72


@dataclass(frozen=True)
class BonusCandidate:
    key: str
    family: str
    profile: str
    ucl_total_cap: float
    promotable: bool
    config: (
        StageWeightedTournamentBonusConfig
        | FiveStageWeightedTournamentBonusConfig
        | None
    ) = None


@dataclass
class CandidateEvaluation:
    predictions: pd.DataFrame
    end_ratings: pd.DataFrame
    same_season_ranking: pd.DataFrame
    season_metrics: pd.DataFrame
    bonus_events: pd.DataFrame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nested ranking-first test of stage-weighted progression bonuses"
    )
    parser.add_argument("--static-data-root", type=Path, default=STATIC_DATA_ROOT)
    parser.add_argument("--events", type=Path, default=EVENTS_PATH)
    parser.add_argument("--dynamic-manifest", type=Path, default=DYNAMIC_MANIFEST)
    parser.add_argument("--production-contract", type=Path, default=PRODUCTION_CONTRACT)
    parser.add_argument("--domestic-adjustments", type=Path, default=DOMESTIC_ADJUSTMENTS)
    parser.add_argument("--xg-data", type=Path, default=XG_DATA)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples must be at least 100")

    contract = json.loads(
        args.production_contract.resolve().read_text(encoding="utf-8")
    )
    core, model_parameters = validate_production_contract(contract)
    dynamic_manifest = json.loads(
        args.dynamic_manifest.resolve().read_text(encoding="utf-8")
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
    target = schedule_adjusted_team_performance(events)
    identity = load_team_season_identity()
    domestic = load_domestic_adjustments(args.domestic_adjustments.resolve(), datasets)
    xg = load_xg_map(args.xg_data.resolve(), datasets)
    candidates = candidate_grid()

    print("Stage-weighted progression: nested walk-forward", flush=True)
    nested = run_nested_backtest(
        datasets,
        folds,
        target,
        identity,
        core,
        model_parameters,
        domestic,
        xg,
        candidates,
    )
    print("Stage-weighted progression: fixed candidate surface", flush=True)
    surface, surface_details = run_candidate_surface(
        datasets,
        folds,
        target,
        identity,
        core,
        model_parameters,
        domestic,
        xg,
        candidates,
    )

    predictions = nested["predictions"]
    final_ranking = build_final_ranking_summary(
        nested["same_season_ranking"], MODEL_CURRENT
    )
    forward_ranking = build_forward_ranking_summary(
        nested["end_ratings"], target, identity, seasons
    )
    competition_stage = build_competition_stage_summary(
        predictions, nested["bonus_events"]
    )
    uncertainty = build_dependency_uncertainty(
        predictions, bootstrap_samples=args.bootstrap_samples
    )
    full_selection = select_training_candidate(
        datasets,
        target,
        core,
        model_parameters,
        domestic,
        xg,
        candidates,
    )
    decision = decide_model(
        nested["fold_results"],
        final_ranking,
        forward_ranking,
        competition_stage,
        uncertainty,
        nested["season_metrics"],
        full_selection,
    )

    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    surface.to_csv(output / "candidate_surface.csv", index=False)
    nested["fold_selections"].to_csv(output / "fold_selections.csv", index=False)
    nested["fold_results"].to_csv(output / "fold_results.csv", index=False)
    competition_stage.to_csv(output / "competition_stage_summary.csv", index=False)
    final_ranking.to_csv(output / "final_ranking_summary.csv", index=False)
    forward_ranking.to_csv(output / "forward_ranking_summary.csv", index=False)
    uncertainty.to_csv(output / "dependency_uncertainty.csv", index=False)
    nested["bonus_events"].to_csv(output / "bonus_event_audit.csv", index=False)
    (output / "selected_candidate.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(
        output / "backtest_report.md",
        contract,
        seasons,
        surface,
        nested["fold_selections"],
        nested["fold_results"],
        final_ranking,
        forward_ranking,
        competition_stage,
        uncertainty,
        decision,
        xg,
        tie_audit,
    )
    write_surface_diagnostics(output, surface_details)
    print(f"Decision: {decision['decision']}")
    print(f"Selected full-history candidate: {decision['full_history_candidate_key']}")
    print(f"Report: {output / 'backtest_report.md'}")


def validate_production_contract(
    contract: dict[str, object],
) -> tuple[DynamicCoreConfig, dict[str, float | int]]:
    dynamic = contract.get("dynamic_core")
    goal = contract.get("goal_margin")
    xg = contract.get("xg_performance")
    surprise = contract.get("domestic_surprise")
    progression = contract.get("progression_bonus")
    if not all(isinstance(value, dict) for value in (dynamic, goal, xg, surprise, progression)):
        raise ValueError("Production contract is missing required model sections")
    assert isinstance(dynamic, dict) and isinstance(goal, dict)
    assert isinstance(xg, dict) and isinstance(surprise, dict)
    assert isinstance(progression, dict)
    core = DynamicCoreConfig(
        float(dynamic["elo_scale"]),
        float(dynamic["home_advantage"]),
        float(dynamic["k_factor"]),
    )
    core.validate()
    expected = {
        "elo_scale": 835.5614973262034,
        "home_advantage": 148.54426619132505,
        "k_factor": 103.98098633392752,
        "goal_alpha": 0.15,
        "goal_tau": 300.0,
        "goal_cap": 4,
        "xg_ratio": 0.30,
        "xg_scale": 1.25,
        "xg_floor": 0.70,
        "draw_at_even": 0.24,
        "draw_shape": 1.0,
        "single_match_draw_at_even": 0.12,
    }
    actual: dict[str, float | int] = {
        "elo_scale": core.elo_scale,
        "home_advantage": core.home_advantage,
        "k_factor": core.k_factor,
        "goal_alpha": float(goal["alpha"]),
        "goal_tau": float(goal["tau"]),
        "goal_cap": int(goal["goal_difference_cap"]),
        "xg_ratio": float(xg["max_xg_ratio"]),
        "xg_scale": float(xg["xg_scale"]),
        "xg_floor": float(xg["minimum_winner_gain_ratio"]),
        "draw_at_even": float(contract["one_x_two_probability"]["draw_at_even"]),
        "draw_shape": float(contract["one_x_two_probability"]["draw_shape"]),
        "single_match_draw_at_even": float(
            contract["one_x_two_probability"]["single_match_draw_at_even"]
        ),
    }
    for key, expected_value in expected.items():
        if not math.isclose(float(actual[key]), float(expected_value), abs_tol=1e-9):
            raise ValueError(f"Unexpected production {key}: {actual[key]}")
    if float(contract["active_power_carry"]) != 0.0:
        raise ValueError("Backtest requires production carry=0")
    if not bool(contract["one_x_two_probability"]["single_match_draw_enabled"]):
        raise ValueError("Production single-match draw correction must be active")
    if not bool(goal.get("active")) or not bool(xg.get("active")):
        raise ValueError("Production goal-margin and xG layers must be active")
    if not bool(surprise.get("active")):
        raise ValueError("Production Domestic Surprise must be active")
    if not bool(progression.get("active")) or float(progression["base_bonus"]) != 12.0:
        raise ValueError("Production fixed 12/8/4 progression baseline must be active")
    if bool(contract["achievement_reserve"]["active"]):
        raise ValueError("Achievement Reserve must remain disabled")
    return core, actual


def profile_pairs(values: tuple[float, ...]) -> tuple[tuple[str, float], ...]:
    return tuple(zip(STAGE_WEIGHTED_PROGRESSION_STAGES, values, strict=True))


def candidate_grid() -> tuple[BonusCandidate, ...]:
    candidates = [
        BonusCandidate(MODEL_CURRENT, "FIXED_FIVE", "FIXED_EQUAL_FIVE", 60.0, True),
        BonusCandidate(MODEL_NONE, "NONE", "NONE", 0.0, False),
    ]
    for profile, weights in PROFILE_WEIGHTS.items():
        for cap in UCL_TOTAL_CAPS:
            config = StageWeightedTournamentBonusConfig(
                ucl_total_cap=cap,
                stage_weights=profile_pairs(weights),
            )
            config.validate()
            promotable = profile != "EQUAL_FOUR"
            candidates.append(
                BonusCandidate(
                    f"{profile}_CAP_{int(cap)}",
                    "STAGE_WEIGHTED",
                    profile,
                    cap,
                    promotable,
                    config,
                )
            )
    keys = [candidate.key for candidate in candidates]
    if len(keys) != len(set(keys)) or len(candidates) != 22:
        raise ValueError("Unexpected stage-weighted candidate grid")
    return tuple(candidates)


def load_domestic_adjustments(
    path: Path,
    datasets: tuple[ControlledSeasonData, ...],
) -> dict[tuple[str, int], float]:
    frame = pd.read_csv(path)
    required = {"season", "team_id", "adjusted_ao_first_elo"}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"Domestic adjustments missing columns: {missing}")
    if frame.duplicated(["season", "team_id"]).any():
        raise ValueError("Domestic adjustments contain duplicate team-season keys")
    mapping = {
        (str(row.season), int(row.team_id)): float(row.adjusted_ao_first_elo)
        for row in frame.itertuples()
    }
    expected = {
        (data.season, int(team_id))
        for data in datasets
        for team_id in data.reserve.goal.carry.core.active_team_ids
    }
    missing_keys = expected - set(mapping)
    if missing_keys:
        raise ValueError(f"Domestic adjustments missing {len(missing_keys)} active teams")
    return mapping


def load_xg_map(
    path: Path,
    datasets: tuple[ControlledSeasonData, ...],
) -> dict[str, tuple[float, float]]:
    frame = pd.read_csv(path)
    required = {"match_id", "season", "xg_home", "xg_away", "xg_analysis_eligible"}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"xG data missing columns: {missing}")
    if frame["match_id"].duplicated().any():
        raise ValueError("xG data contains duplicate match_id")
    eligible = frame.loc[frame["xg_analysis_eligible"].astype(bool)].copy()
    if eligible[["xg_home", "xg_away"]].isna().any().any():
        raise ValueError("Eligible xG rows require both team values")
    known = {
        str(match_id)
        for data in datasets
        for match_id in data.reserve.goal.carry.core.match_ids
    }
    unknown = set(eligible["match_id"].astype(str)) - known
    if unknown:
        raise ValueError(f"xG data has {len(unknown)} unknown eligible matches")
    return {
        str(row.match_id): (float(row.xg_home), float(row.xg_away))
        for row in eligible.itertuples()
    }


def probability_vector(expected: float, draw_at_even: float, draw_shape: float) -> np.ndarray:
    return np.asarray(
        one_x_two_probabilities_scalar(expected, draw_at_even, draw_shape),
        dtype=float,
    )


def evaluate_candidate(
    datasets: tuple[ControlledSeasonData, ...],
    target: pd.DataFrame,
    core: DynamicCoreConfig,
    parameters: dict[str, float | int],
    domestic: dict[tuple[str, int], float],
    xg: dict[str, tuple[float, float]],
    candidate: BonusCandidate,
    *,
    evaluation_seasons: set[str],
    return_details: bool = False,
) -> CandidateEvaluation:
    prediction_rows: list[dict[str, object]] = []
    end_rows: list[dict[str, object]] = []
    season_rows: list[dict[str, object]] = []
    bonus_rows: list[dict[str, object]] = []
    xg_blend = XGBlendConfig(0.0, 1.0)
    xg_bonus = XGPerformanceBonusConfig(
        float(parameters["xg_ratio"]),
        float(parameters["xg_scale"]),
        float(parameters["xg_floor"]),
    )
    fixed = FixedTournamentBonusConfig(12.0)

    for data in datasets:
        reserve = data.reserve
        goal = reserve.goal
        season_core = goal.carry.core
        power = season_core.initial_ratings.copy()
        active_ids = season_core.active_team_ids
        for team_id in active_ids:
            power[int(team_id)] = domestic[(data.season, int(team_id))]
        initial = power.copy()
        bonus = np.zeros((len(power), len(COMPETITION_INDEX)), dtype=float)
        processed_ties: set[str] = set()
        initial_power_total = float(power[active_ids].sum())
        max_zero_sum_error = 0.0
        max_cap_error = 0.0
        max_delta = 0.0
        tie_counts = pd.Series(reserve.tie_ids, dtype="object").dropna().value_counts()

        for index, (home_raw, away_raw, neutral_raw, competition_raw) in enumerate(
            zip(
                season_core.home_team_ids,
                season_core.away_team_ids,
                season_core.neutral_flags,
                season_core.competitions,
            )
        ):
            home_id, away_id = int(home_raw), int(away_raw)
            competition = str(competition_raw)
            match_id = str(season_core.match_ids[index])
            stage = str(reserve.stages[index])
            tie_value = reserve.tie_ids[index]
            tie_id = None if tie_value is None else str(tie_value)
            home_bonus_pre = float(bonus[home_id].sum())
            away_bonus_pre = float(bonus[away_id].sum())
            home_live_pre = float(power[home_id] + home_bonus_pre)
            away_live_pre = float(power[away_id] + away_bonus_pre)
            xg_values = xg.get(match_id)
            penalty = bool(goal.penalty_flags[index])
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
                goal_difference_enabled=True,
                goal_alpha=float(parameters["goal_alpha"]),
                goal_tau=float(parameters["goal_tau"]),
                goal_difference_cap=int(parameters["goal_cap"]),
                xg_config=xg_blend,
                xg_home=None if xg_values is None else xg_values[0],
                xg_away=None if xg_values is None else xg_values[1],
                xg_performance_bonus_config=xg_bonus,
            )
            is_single_match_tie = bool(
                tie_id is not None
                and int(tie_counts.get(tie_id, 0)) == 1
                and bool(reserve.tie_decider_flags[index])
            )
            effective_draw_at_even = (
                float(parameters["single_match_draw_at_even"])
                if is_single_match_tie
                else float(parameters["draw_at_even"])
            )
            probabilities = probability_vector(
                update.expected_home_score,
                effective_draw_at_even,
                float(parameters["draw_shape"]),
            )
            observed = 0 if update.actual_home_score == 1.0 else 1 if update.actual_home_score == 0.5 else 2
            target_vector = np.eye(3, dtype=float)[observed]
            brier = float(np.square(probabilities - target_vector).sum())
            log_loss = -math.log(max(float(probabilities[observed]), 1e-15))
            power[home_id] += update.power_delta
            power[away_id] -= update.power_delta
            max_delta = max(max_delta, abs(update.power_delta))
            max_zero_sum_error = max(max_zero_sum_error, update.zero_sum_error)

            applied = 0.0
            winner_id = -1
            audit: dict[str, object] = {}
            eligible = False
            if candidate.family == "FIXED_FIVE":
                eligible = stage in ELIGIBLE_PROGRESSION_STAGES
            elif candidate.family == "STAGE_WEIGHTED":
                eligible = stage in STAGE_WEIGHTED_PROGRESSION_STAGES
            elif candidate.family == "FIVE_STAGE_WEIGHTED":
                eligible = stage in ELIGIBLE_PROGRESSION_STAGES
            if eligible and bool(reserve.tie_decider_flags[index]):
                if tie_id is None:
                    raise ValueError(f"{data.season}/{match_id}: missing tie_id")
                winner_id = int(reserve.advanced_team_ids[index])
                if winner_id not in (home_id, away_id):
                    raise ValueError(f"{data.season}/{tie_id}: invalid advancing team")
                component = COMPETITION_INDEX[competition]
                pre = float(bonus[winner_id, component])
                if candidate.family == "FIXED_FIVE":
                    fixed_update = apply_tournament_progress_bonus(
                        pre, competition, stage, tie_id, processed_ties, fixed
                    )
                    bonus[winner_id, component] = fixed_update.bonus_post
                    applied = fixed_update.applied_bonus
                    audit = {
                        "stage_weight": 0.20,
                        "competition_ratio": fixed.increment(competition) / fixed.base_bonus,
                        "stage_bonus_requested": fixed.increment(competition),
                        "stage_bonus_applied": applied,
                        "competition_bonus_pre": fixed_update.bonus_pre,
                        "competition_bonus_post": fixed_update.bonus_post,
                        "competition_cap": fixed_update.competition_cap,
                    }
                elif candidate.family == "STAGE_WEIGHTED":
                    assert candidate.config is not None
                    assert isinstance(candidate.config, StageWeightedTournamentBonusConfig)
                    weighted = apply_stage_weighted_tournament_bonus(
                        pre,
                        competition,
                        stage,
                        tie_id,
                        processed_ties,
                        candidate.config,
                    )
                    bonus[winner_id, component] = weighted.competition_bonus_post
                    applied = weighted.stage_bonus_applied
                    audit = {
                        "stage_weight": weighted.stage_weight,
                        "competition_ratio": weighted.competition_ratio,
                        "stage_bonus_requested": weighted.stage_bonus_requested,
                        "stage_bonus_applied": weighted.stage_bonus_applied,
                        "competition_bonus_pre": weighted.competition_bonus_pre,
                        "competition_bonus_post": weighted.competition_bonus_post,
                        "competition_cap": weighted.competition_cap,
                    }
                else:
                    assert candidate.family == "FIVE_STAGE_WEIGHTED"
                    assert isinstance(candidate.config, FiveStageWeightedTournamentBonusConfig)
                    weighted = apply_five_stage_weighted_tournament_bonus(
                        pre,
                        competition,
                        stage,
                        tie_id,
                        processed_ties,
                        candidate.config,
                    )
                    bonus[winner_id, component] = weighted.competition_bonus_post
                    applied = weighted.stage_bonus_applied
                    audit = {
                        "stage_weight": weighted.stage_weight,
                        "competition_ratio": weighted.competition_ratio,
                        "stage_bonus_requested": weighted.stage_bonus_requested,
                        "stage_bonus_applied": weighted.stage_bonus_applied,
                        "competition_bonus_pre": weighted.competition_bonus_pre,
                        "competition_bonus_post": weighted.competition_bonus_post,
                        "competition_cap": weighted.competition_cap,
                    }
                max_cap_error = max(
                    max_cap_error,
                    max(0.0, float(audit["competition_bonus_post"]) - float(audit["competition_cap"])),
                )
                if return_details and data.season in evaluation_seasons:
                    bonus_rows.append(
                        {
                            "model": candidate.key,
                            "season": data.season,
                            "match_id": match_id,
                            "tie_id": tie_id,
                            "competition": competition,
                            "stage": stage,
                            "winner_team_id": winner_id,
                            "decided_on_penalties": penalty,
                            "profile": candidate.profile,
                            "ucl_total_cap": candidate.ucl_total_cap,
                            **audit,
                        }
                    )

            if return_details and data.season in evaluation_seasons:
                prediction_rows.append(
                    {
                        "model": candidate.key,
                        "match_id": match_id,
                        "season": data.season,
                        "kickoff_utc": pd.Timestamp(data.kickoff_utc[index]),
                        "competition": competition,
                        "stage": stage,
                        "tie_id": tie_id,
                        "is_single_match_tie": is_single_match_tie,
                        "effective_draw_at_even": effective_draw_at_even,
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
                        "bonus_applied_after_match": applied,
                        "bonus_winner_id": winner_id,
                    }
                )

        end_power = power[active_ids]
        end_bonus = bonus[active_ids].sum(axis=1)
        end_live = end_power + end_bonus
        power_total_error = abs(float(end_power.sum()) - initial_power_total)
        if data.season in evaluation_seasons:
            season_rows.append(
                {
                    "model": candidate.key,
                    "season": data.season,
                    "matches": len(season_core.match_ids),
                    "teams": len(active_ids),
                    "rating_min": float(end_live.min()),
                    "rating_max": float(end_live.max()),
                    "rating_std": float(end_live.std()),
                    "total_bonus_added": float(end_bonus.sum()),
                    "maximum_team_bonus": float(end_bonus.max()),
                    "maximum_abs_match_delta": max_delta,
                    "maximum_abs_rating_change": float(np.max(np.abs(end_live - initial[active_ids]))),
                    "start_end_rank_correlation": safe_rank_correlation(initial[active_ids], end_live),
                    "maximum_power_zero_sum_error": max(max_zero_sum_error, power_total_error),
                    "maximum_bonus_cap_error": max_cap_error,
                    "season_start_bonus": 0.0,
                }
            )
        end_rows.extend(
            {
                "model": candidate.key,
                "season": data.season,
                "team_id": int(team_id),
                "initial_rating": float(initial[int(team_id)]),
                "end_power_rating": float(power[int(team_id)]),
                "end_tournament_bonus": float(bonus[int(team_id)].sum()),
                "end_live_rating": float(power[int(team_id)] + bonus[int(team_id)].sum()),
            }
            for team_id in active_ids
        )

    end_ratings = pd.DataFrame(end_rows)
    same_ranking = same_season_ranking(end_ratings, target, evaluation_seasons)
    return CandidateEvaluation(
        predictions=pd.DataFrame(prediction_rows),
        end_ratings=end_ratings,
        same_season_ranking=same_ranking,
        season_metrics=pd.DataFrame(season_rows),
        bonus_events=pd.DataFrame(bonus_rows),
    )


def same_season_ranking(
    end_ratings: pd.DataFrame,
    target: pd.DataFrame,
    seasons: set[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (season, competition), actual in target.loc[target["season"].isin(seasons)].groupby(
        ["season", "competition"], sort=True
    ):
        predicted = end_ratings.loc[end_ratings["season"].eq(season)]
        table = actual[["team_id", "schedule_adjusted_score"]].merge(
            predicted[["team_id", "end_live_rating"]], on="team_id", validate="one_to_one"
        )
        if len(table) < 3:
            continue
        rows.append(
            {
                "season": season,
                "competition": competition,
                "teams": len(table),
                "pair_weight": len(table) * (len(table) - 1) / 2,
                "ranking_score": float(table["end_live_rating"].corr(table["schedule_adjusted_score"], method="spearman")),
                "pairwise_accuracy": pairwise_ranking_accuracy(
                    table["end_live_rating"].to_numpy(float),
                    table["schedule_adjusted_score"].to_numpy(float),
                ),
            }
        )
    return pd.DataFrame(rows)


def aggregate_ranking(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame(columns=["competition", "groups", "teams", "ranking_score", "pairwise_accuracy"])
    rows = []
    for competition, frame in [("ALL", detail), *detail.groupby("competition", sort=True)]:
        rows.append(
            {
                "competition": competition,
                "groups": len(frame),
                "teams": int(frame["teams"].sum()),
                "ranking_score": float(np.average(frame["ranking_score"], weights=frame["teams"])),
                "pairwise_accuracy": float(np.average(frame["pairwise_accuracy"], weights=frame["pair_weight"])),
            }
        )
    return pd.DataFrame(rows)


def evaluation_metrics(evaluation: CandidateEvaluation) -> dict[str, float | int]:
    predictions = evaluation.predictions
    ranking = aggregate_ranking(evaluation.same_season_ranking)
    all_rank = ranking.loc[ranking["competition"].eq("ALL")]
    if predictions.empty or len(all_rank) != 1:
        raise ValueError("Evaluation lacks predictions or ranking summary")
    seasons = evaluation.season_metrics
    return {
        "matches": len(predictions),
        "brier_1x2": float(predictions["brier_1x2"].mean()),
        "log_loss_1x2": float(predictions["log_loss_1x2"].mean()),
        "accuracy_1x2": float((predictions["actual_class"] == predictions["predicted_class"]).mean()),
        "ranking_score": float(all_rank.iloc[0]["ranking_score"]),
        "pairwise_accuracy": float(all_rank.iloc[0]["pairwise_accuracy"]),
        "minimum_start_end_rank_correlation": float(seasons["start_end_rank_correlation"].min()),
        "maximum_abs_rating_change": float(seasons["maximum_abs_rating_change"].max()),
        "maximum_abs_match_delta": float(seasons["maximum_abs_match_delta"].max()),
        "maximum_power_zero_sum_error": float(seasons["maximum_power_zero_sum_error"].max()),
        "maximum_bonus_cap_error": float(seasons["maximum_bonus_cap_error"].max()),
        "maximum_season_start_bonus": float(seasons["season_start_bonus"].max()),
        "total_bonus_added": float(seasons["total_bonus_added"].sum()),
        "maximum_team_bonus": float(seasons["maximum_team_bonus"].max()),
    }


def select_training_candidate(
    datasets: tuple[ControlledSeasonData, ...],
    target: pd.DataFrame,
    core: DynamicCoreConfig,
    parameters: dict[str, float | int],
    domestic: dict[tuple[str, int], float],
    xg: dict[str, tuple[float, float]],
    candidates: tuple[BonusCandidate, ...],
) -> dict[str, object]:
    seasons = {data.season for data in datasets}
    rows = []
    for candidate in candidates:
        if not candidate.promotable:
            continue
        evaluation = evaluate_candidate(
            datasets,
            target,
            core,
            parameters,
            domestic,
            xg,
            candidate,
            evaluation_seasons=seasons,
            return_details=True,
        )
        rows.append({"candidate_key": candidate.key, **evaluation_metrics(evaluation)})
    metrics = pd.DataFrame(rows)
    baseline = metrics.loc[metrics["candidate_key"].eq(MODEL_CURRENT)].iloc[0]
    safe = metrics.loc[
        (metrics["ranking_score"] >= baseline["ranking_score"] - RANK_TOLERANCE)
        & (metrics["pairwise_accuracy"] >= baseline["pairwise_accuracy"] - RANK_TOLERANCE)
        & (metrics["minimum_start_end_rank_correlation"] >= 0.85)
        & (metrics["maximum_abs_rating_change"] <= MAX_RATING_MOVE)
        & (metrics["maximum_power_zero_sum_error"] <= 1e-9)
        & (metrics["maximum_bonus_cap_error"] <= 1e-9)
        & (metrics["maximum_season_start_bonus"] <= 1e-12)
    ].copy()
    selected = safe.sort_values(
        ["pairwise_accuracy", "ranking_score", "brier_1x2", "log_loss_1x2", "candidate_key"],
        ascending=[False, False, True, True, True],
        kind="stable",
    ).iloc[0]
    return selected.to_dict()


def run_nested_backtest(
    datasets: tuple[ControlledSeasonData, ...],
    folds: tuple[tuple[tuple[str, ...], str], ...],
    target: pd.DataFrame,
    identity: pd.DataFrame,
    core: DynamicCoreConfig,
    parameters: dict[str, float | int],
    domestic: dict[tuple[str, int], float],
    xg: dict[str, tuple[float, float]],
    candidates: tuple[BonusCandidate, ...],
) -> dict[str, pd.DataFrame]:
    by_season = {data.season: data for data in datasets}
    by_key = {candidate.key: candidate for candidate in candidates}
    selection_rows: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    end_frames: list[pd.DataFrame] = []
    ranking_frames: list[pd.DataFrame] = []
    season_frames: list[pd.DataFrame] = []
    bonus_frames: list[pd.DataFrame] = []

    for fold, (train_seasons, test_season) in enumerate(folds, start=1):
        training = tuple(by_season[season] for season in train_seasons)
        selection = select_training_candidate(
            training, target, core, parameters, domestic, xg, candidates
        )
        selected = by_key[str(selection["candidate_key"])]
        selection_rows.append(
            {
                "fold": fold,
                "train_seasons": "|".join(train_seasons),
                "test_season": test_season,
                "selected_candidate_key": selected.key,
                "selected_profile": selected.profile,
                "selected_ucl_total_cap": selected.ucl_total_cap,
                "train_ranking_score": selection["ranking_score"],
                "train_pairwise_accuracy": selection["pairwise_accuracy"],
                "train_brier_1x2": selection["brier_1x2"],
                "train_log_loss_1x2": selection["log_loss_1x2"],
            }
        )
        model_candidates = {
            MODEL_CURRENT: by_key[MODEL_CURRENT],
            MODEL_NESTED: selected,
            MODEL_NONE: by_key[MODEL_NONE],
        }
        for model, candidate in model_candidates.items():
            evaluation = evaluate_candidate(
                (by_season[test_season],),
                target,
                core,
                parameters,
                domestic,
                xg,
                candidate,
                evaluation_seasons={test_season},
                return_details=True,
            )
            metrics = evaluation_metrics(evaluation)
            result_rows.append(
                {
                    "fold": fold,
                    "test_season": test_season,
                    "model": model,
                    "candidate_key": candidate.key,
                    "profile": candidate.profile,
                    "ucl_total_cap": candidate.ucl_total_cap,
                    **metrics,
                }
            )
            predictions = evaluation.predictions.copy()
            predictions["fold"] = fold
            predictions["model"] = model
            predictions["candidate_key"] = candidate.key
            prediction_frames.append(predictions)
            end = evaluation.end_ratings.copy()
            end["fold"] = fold
            end["model"] = model
            end["candidate_key"] = candidate.key
            end_frames.append(end)
            ranking = evaluation.same_season_ranking.copy()
            ranking["fold"] = fold
            ranking["model"] = model
            ranking_frames.append(ranking)
            season_metrics = evaluation.season_metrics.copy()
            season_metrics["fold"] = fold
            season_metrics["model"] = model
            season_frames.append(season_metrics)
            if not evaluation.bonus_events.empty:
                bonuses = evaluation.bonus_events.copy()
                bonuses["fold"] = fold
                bonuses["model"] = model
                bonuses["candidate_key"] = candidate.key
                bonus_frames.append(bonuses)
        print(f"  fold {fold}/6 -> {test_season}: {selected.key}", flush=True)

    return {
        "fold_selections": pd.DataFrame(selection_rows),
        "fold_results": pd.DataFrame(result_rows),
        "predictions": pd.concat(prediction_frames, ignore_index=True),
        "end_ratings": pd.concat(end_frames, ignore_index=True),
        "same_season_ranking": pd.concat(ranking_frames, ignore_index=True),
        "season_metrics": pd.concat(season_frames, ignore_index=True),
        "bonus_events": pd.concat(bonus_frames, ignore_index=True) if bonus_frames else pd.DataFrame(),
    }


def run_candidate_surface(
    datasets: tuple[ControlledSeasonData, ...],
    folds: tuple[tuple[tuple[str, ...], str], ...],
    target: pd.DataFrame,
    identity: pd.DataFrame,
    core: DynamicCoreConfig,
    parameters: dict[str, float | int],
    domestic: dict[tuple[str, int], float],
    xg: dict[str, tuple[float, float]],
    candidates: tuple[BonusCandidate, ...],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    test_seasons = {test for _, test in folds}
    rows = []
    ranking_frames = []
    end_frames = []
    prediction_frames = []
    for index, candidate in enumerate(candidates, start=1):
        evaluation = evaluate_candidate(
            datasets,
            target,
            core,
            parameters,
            domestic,
            xg,
            candidate,
            evaluation_seasons=test_seasons,
            return_details=True,
        )
        metrics = evaluation_metrics(evaluation)
        forward = summarize_ranking(
            evaluation.end_ratings,
            target,
            allowed_target_seasons=set(sorted(test_seasons)[1:]),
            identity=identity,
        )
        forward_all = forward.loc[forward["competition"].eq("ALL")]
        rows.append(
            {
                "candidate_key": candidate.key,
                "family": candidate.family,
                "profile": candidate.profile,
                "ucl_total_cap": candidate.ucl_total_cap,
                "promotable": candidate.promotable,
                **metrics,
                "forward_ranking_score": float(forward_all.iloc[0]["ranking_score"]) if len(forward_all) else np.nan,
                "forward_pairwise_accuracy": float(forward_all.iloc[0]["pairwise_accuracy"]) if len(forward_all) else np.nan,
            }
        )
        rank = aggregate_ranking(evaluation.same_season_ranking)
        rank["candidate_key"] = candidate.key
        ranking_frames.append(rank)
        end = evaluation.end_ratings.copy()
        end["candidate_key"] = candidate.key
        end_frames.append(end)
        predictions = evaluation.predictions.copy()
        predictions["candidate_key"] = candidate.key
        prediction_frames.append(predictions)
        if index % 5 == 0 or index == len(candidates):
            print(f"  candidate {index}/{len(candidates)}", flush=True)
    surface = pd.DataFrame(rows)
    baseline = surface.loc[surface["candidate_key"].eq(MODEL_CURRENT)].iloc[0]
    for metric in ("brier_1x2", "log_loss_1x2", "ranking_score", "pairwise_accuracy", "forward_ranking_score", "forward_pairwise_accuracy"):
        surface[f"delta_vs_current_{metric}"] = surface[metric] - float(baseline[metric])
    return surface, {
        "candidate_ranking_by_competition": pd.concat(ranking_frames, ignore_index=True),
        "candidate_end_ratings": pd.concat(end_frames, ignore_index=True),
        "candidate_predictions": pd.concat(prediction_frames, ignore_index=True),
    }


def build_final_ranking_summary(detail: pd.DataFrame, baseline_model: str) -> pd.DataFrame:
    frames = []
    for model, group in detail.groupby("model", sort=False):
        summary = aggregate_ranking(group)
        summary["model"] = model
        frames.append(summary)
    result = pd.concat(frames, ignore_index=True)
    baseline = result.loc[result["model"].eq(baseline_model)].set_index("competition")
    result["ranking_delta_vs_current"] = result.apply(
        lambda row: row["ranking_score"] - baseline.loc[row["competition"], "ranking_score"], axis=1
    )
    result["pairwise_delta_vs_current"] = result.apply(
        lambda row: row["pairwise_accuracy"] - baseline.loc[row["competition"], "pairwise_accuracy"], axis=1
    )
    return result


def build_forward_ranking_summary(
    end_ratings: pd.DataFrame,
    target: pd.DataFrame,
    identity: pd.DataFrame,
    seasons: tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    next_season = {seasons[index]: seasons[index + 1] for index in range(len(seasons) - 1)}
    for model, model_end in end_ratings.groupby("model", sort=False):
        for source in seasons[2:-1]:
            target_season = next_season[source]
            source_end = model_end.loc[model_end["season"].eq(source)]
            ranking = summarize_ranking(
                source_end,
                target,
                allowed_target_seasons={target_season},
                identity=identity,
            )
            for row in ranking.itertuples(index=False):
                rows.append(
                    {
                        "model": model,
                        "source_season": source,
                        "target_season": target_season,
                        "competition": row.competition,
                        "groups": row.groups,
                        "team_weight": row.team_weight,
                        "ranking_score": row.ranking_score,
                        "pairwise_accuracy": row.pairwise_accuracy,
                    }
                )
    result = pd.DataFrame(rows)
    baseline = result.loc[result["model"].eq(MODEL_CURRENT)].set_index(
        ["source_season", "target_season", "competition"]
    )
    result["ranking_delta_vs_current"] = result.apply(
        lambda row: row["ranking_score"] - baseline.loc[(row["source_season"], row["target_season"], row["competition"]), "ranking_score"], axis=1
    )
    result["pairwise_delta_vs_current"] = result.apply(
        lambda row: row["pairwise_accuracy"] - baseline.loc[(row["source_season"], row["target_season"], row["competition"]), "pairwise_accuracy"], axis=1
    )
    return result


def build_competition_stage_summary(predictions: pd.DataFrame, bonuses: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, candidate_key, competition, stage), frame in predictions.groupby(
        ["model", "candidate_key", "competition", "stage"], sort=True
    ):
        rows.append(
            {
                "model": model,
                "candidate_key": candidate_key,
                "competition": competition,
                "stage": stage,
                "matches": len(frame),
                "brier_1x2": float(frame["brier_1x2"].mean()),
                "log_loss_1x2": float(frame["log_loss_1x2"].mean()),
                "accuracy_1x2": float((frame["actual_class"] == frame["predicted_class"]).mean()),
                "bonus_events": 0,
                "bonus_added": 0.0,
            }
        )
    result = pd.DataFrame(rows)
    if not bonuses.empty:
        summary = bonuses.groupby(
            ["model", "candidate_key", "competition", "stage"], sort=True
        ).agg(
            bonus_events=("stage_bonus_applied", "size"),
            bonus_added=("stage_bonus_applied", "sum"),
        ).reset_index()
        result = result.drop(columns=["bonus_events", "bonus_added"]).merge(
            summary,
            on=["model", "candidate_key", "competition", "stage"],
            how="left",
        )
        result[["bonus_events", "bonus_added"]] = result[["bonus_events", "bonus_added"]].fillna(0)
    return result


def build_dependency_uncertainty(predictions: pd.DataFrame, *, bootstrap_samples: int) -> pd.DataFrame:
    baseline = predictions.loc[predictions["model"].eq(MODEL_CURRENT)]
    candidate = predictions.loc[predictions["model"].eq(MODEL_NESTED)]
    rows = []
    for competition in ("ALL", "UCL", "UEL", "UECL"):
        left = candidate if competition == "ALL" else candidate.loc[candidate["competition"].eq(competition)]
        right = baseline if competition == "ALL" else baseline.loc[baseline["competition"].eq(competition)]
        paired = left.merge(
            right[["match_id", "brier_1x2", "log_loss_1x2"]],
            on="match_id",
            suffixes=("_candidate", "_baseline"),
            validate="one_to_one",
        )
        for loss in ("brier_1x2", "log_loss_1x2"):
            sample = paired[["season", "match_id", "home_team_id", "away_team_id", "kickoff_utc", "tie_id"]].copy()
            sample["loss_difference"] = paired[f"{loss}_candidate"] - paired[f"{loss}_baseline"]
            ci = dependency_robust_loss_difference_ci(
                sample,
                bootstrap_samples=bootstrap_samples,
            )
            ci["competition"] = competition
            ci["metric"] = loss
            rows.append(ci)
    return pd.concat(rows, ignore_index=True)


def decide_model(
    folds: pd.DataFrame,
    final_ranking: pd.DataFrame,
    forward: pd.DataFrame,
    competition_stage: pd.DataFrame,
    uncertainty: pd.DataFrame,
    season_metrics: pd.DataFrame,
    full_selection: dict[str, object],
) -> dict[str, object]:
    current = folds.loc[folds["model"].eq(MODEL_CURRENT)].set_index("fold")
    nested = folds.loc[folds["model"].eq(MODEL_NESTED)].set_index("fold")
    rank_delta = nested["ranking_score"] - current["ranking_score"]
    pair_delta = nested["pairwise_accuracy"] - current["pairwise_accuracy"]
    rank_no_regression = bool((rank_delta >= -RANK_TOLERANCE).all() and (pair_delta >= -RANK_TOLERANCE).all())
    rank_wins = int(((rank_delta > RANK_TOLERANCE) & (pair_delta > RANK_TOLERANCE)).sum())
    pooled = final_ranking.loc[
        final_ranking["model"].eq(MODEL_NESTED) & final_ranking["competition"].eq("ALL")
    ].iloc[0]
    pooled_rank_positive = bool(
        pooled["ranking_delta_vs_current"] > 0.0 and pooled["pairwise_delta_vs_current"] > 0.0
    )
    competition_rank_safe = bool(
        (
            final_ranking.loc[final_ranking["model"].eq(MODEL_NESTED), "ranking_delta_vs_current"]
            >= -RANK_TOLERANCE
        ).all()
        and (
            final_ranking.loc[final_ranking["model"].eq(MODEL_NESTED), "pairwise_delta_vs_current"]
            >= -RANK_TOLERANCE
        ).all()
    )
    pooled_brier = float(
        np.average(nested["brier_1x2"], weights=nested["matches"])
        - np.average(current["brier_1x2"], weights=current["matches"])
    )
    pooled_log = float(
        np.average(nested["log_loss_1x2"], weights=nested["matches"])
        - np.average(current["log_loss_1x2"], weights=current["matches"])
    )
    losses_not_worse = pooled_brier <= 1e-12 and pooled_log <= 1e-12
    envelopes = uncertainty.loc[uncertainty["method"].eq("conservative_envelope")]
    competition_loss_no_reliable_harm = not bool(envelopes["reliable_harm"].any())
    forward_all = forward.loc[
        forward["model"].eq(MODEL_NESTED) & forward["competition"].eq("ALL")
    ]
    forward_no_regression = bool(
        (forward_all["ranking_delta_vs_current"] >= -RANK_TOLERANCE).all()
        and (forward_all["pairwise_delta_vs_current"] >= -RANK_TOLERANCE).all()
    )
    forward_wins = int(
        (
            (forward_all["ranking_delta_vs_current"] > RANK_TOLERANCE)
            & (forward_all["pairwise_delta_vs_current"] > RANK_TOLERANCE)
        ).sum()
    )
    nested_seasons = season_metrics.loc[season_metrics["model"].eq(MODEL_NESTED)]
    invariants = {
        "power_zero_sum": bool(nested_seasons["maximum_power_zero_sum_error"].max() <= 1e-9),
        "bonus_cap": bool(nested_seasons["maximum_bonus_cap_error"].max() <= 1e-9),
        "season_reset": bool(nested_seasons["season_start_bonus"].max() <= 1e-12),
        "rating_move": bool(nested_seasons["maximum_abs_rating_change"].max() <= MAX_RATING_MOVE),
    }
    gates = {
        "six_fold_ranking_no_regression": rank_no_regression,
        "ranking_wins_at_least_4_of_6": rank_wins >= 4,
        "pooled_ranking_positive": pooled_rank_positive,
        "competition_ranking_no_point_harm": competition_rank_safe,
        "pooled_brier_log_not_worse": losses_not_worse,
        "competition_loss_no_reliable_harm": competition_loss_no_reliable_harm,
        "five_forward_transitions_no_regression": forward_no_regression and len(forward_all) == 5,
        "forward_wins_at_least_3_of_5": forward_wins >= 3,
        "all_invariants": all(invariants.values()),
        "full_history_selects_stage_weighted": str(full_selection["candidate_key"]) != MODEL_CURRENT,
    }
    promoted = all(gates.values())
    return {
        "decision": "PROMOTE_CANDIDATE" if promoted else "KEEP_CURRENT_PRODUCTION",
        "production_changed": False,
        "full_history_candidate_key": full_selection["candidate_key"],
        "full_history_candidate_metrics": full_selection,
        "nested_rank_wins": f"{rank_wins}/6",
        "forward_rank_wins": f"{forward_wins}/5",
        "pooled_brier_delta_vs_current": pooled_brier,
        "pooled_log_loss_delta_vs_current": pooled_log,
        "gates": gates,
        "invariants": invariants,
        "interpretation": "Backtest recommendation only; production contract remains unchanged.",
    }


def write_surface_diagnostics(output: Path, details: dict[str, pd.DataFrame]) -> None:
    diagnostics = output / "diagnostics"
    diagnostics.mkdir(exist_ok=True)
    for name, frame in details.items():
        frame.to_csv(diagnostics / f"{name}.csv", index=False)


def write_report(
    path: Path,
    contract: dict[str, object],
    seasons: tuple[str, ...],
    surface: pd.DataFrame,
    selections: pd.DataFrame,
    folds: pd.DataFrame,
    final_ranking: pd.DataFrame,
    forward: pd.DataFrame,
    competition_stage: pd.DataFrame,
    uncertainty: pd.DataFrame,
    decision: dict[str, object],
    xg: dict[str, tuple[float, float]],
    tie_audit: pd.DataFrame,
) -> None:
    current = folds.loc[folds["model"].eq(MODEL_CURRENT)]
    nested = folds.loc[folds["model"].eq(MODEL_NESTED)]
    top = surface.sort_values(
        ["pairwise_accuracy", "ranking_score", "brier_1x2"],
        ascending=[False, False, True],
    ).head(10)
    nested_rank = final_ranking.loc[
        final_ranking["model"].eq(MODEL_NESTED)
    ][["competition", "ranking_delta_vs_current", "pairwise_delta_vs_current"]]
    forward_all = forward.loc[
        forward["model"].eq(MODEL_NESTED) & forward["competition"].eq("ALL")
    ][["source_season", "target_season", "ranking_delta_vs_current", "pairwise_delta_vs_current"]]
    envelope = uncertainty.loc[uncertainty["method"].eq("conservative_envelope")][
        ["competition", "metric", "mean_difference", "ci_95_lower", "ci_95_upper", "reliable_harm"]
    ]
    bonus_counts = competition_stage.loc[
        competition_stage["model"].eq(MODEL_NESTED) & competition_stage["bonus_events"].gt(0)
    ][["candidate_key", "competition", "stage", "bonus_events", "bonus_added"]]
    current_brier = float(np.average(current["brier_1x2"], weights=current["matches"]))
    nested_brier = float(np.average(nested["brier_1x2"], weights=nested["matches"]))
    current_log = float(np.average(current["log_loss_1x2"], weights=current["matches"]))
    nested_log = float(np.average(nested["log_loss_1x2"], weights=nested["matches"]))
    lines = [
        "# Aşama Ağırlıklı European Progression Bonus Backtesti",
        "",
        "## Kapsam ve Sözleşme",
        "",
        f"- Sezonlar: `{seasons[0]}`–`{seasons[-1]}`; altı expanding outer fold.",
        f"- Doğrulanmış xG kullanılan maç sayısı: `{len(xg)}`; diğer maçlarda gol farkı fallback.",
        "- Production karşılaştırması: winner-only `12/8/4 × 5 aşama`, sezon resetli.",
        "- Adaylar: KPO'suz dört aşama; Gentle, Linear, Final Heavy ve diagnostik Equal Four; UCL cap `30/45/60/75/90`.",
        "- UCL/UEL/UECL oranı her adayda `3:2:1`; kaybedenden puan düşülmez.",
        "- Penaltı atışı Power/GD/xG sinyali üretmez; penaltıyla tur geçen takım progression bonusunu tam alır.",
        "- Bu çalışma production sözleşmesini otomatik değiştirmez.",
        "",
        "## Production Çekirdeği",
        "",
        f"- Scale `{contract['dynamic_core']['elo_scale']:.6f}`, H `{contract['dynamic_core']['home_advantage']:.6f}`, K `{contract['dynamic_core']['k_factor']:.6f}`.",
        f"- Gol farkı alpha `{contract['goal_margin']['alpha']}`, tau `{contract['goal_margin']['tau']}`, cap `{contract['goal_margin']['goal_difference_cap']}`.",
        f"- xG ratio `{contract['xg_performance']['max_xg_ratio']}`, scale `{contract['xg_performance']['xg_scale']}`, floor `{contract['xg_performance']['minimum_winner_gain_ratio']}`.",
        "- Domestic Surprise başlangıç ratinglerinde aktiftir.",
        "",
        "## Fold Seçimleri",
        "",
        markdown_table(selections),
        "",
        "## Pooled Sonuç",
        "",
        f"- Current Brier `{current_brier:.9f}`, nested `{nested_brier:.9f}`, fark `{decision['pooled_brier_delta_vs_current']:+.9f}`.",
        f"- Current log-loss `{current_log:.9f}`, nested `{nested_log:.9f}`, fark `{decision['pooled_log_loss_delta_vs_current']:+.9f}`.",
        f"- Aynı-sezon ranking ortak iyileşme: `{decision['nested_rank_wins']}`.",
        f"- Forward-ranking ortak iyileşme: `{decision['forward_rank_wins']}`.",
        "",
        "### Turnuva Bazlı Aynı-Sezon Ranking Farkı",
        "",
        markdown_table(nested_rank, float_digits=9),
        "",
        "### Beş Forward Geçiş",
        "",
        markdown_table(forward_all, float_digits=9),
        "",
        "### Loss Belirsizliği",
        "",
        markdown_table(envelope, float_digits=9),
        "",
        "### Nested Bonus Olayları",
        "",
        markdown_table(bonus_counts, float_digits=3) if not bonus_counts.empty else "Nested seçim production ile aynı kaldığı için ayrı bonus olayı yok.",
        "",
        "Not: Nested kol production adayını seçtiği foldlarda KPO olayları görünür. Dört aşamalı aday profillerin hiçbiri KPO bonusu üretmez.",
        "",
        "## Aday Yüzeyi: İlk 10",
        "",
        markdown_table(
            top[["candidate_key", "ranking_score", "pairwise_accuracy", "brier_1x2", "log_loss_1x2", "forward_ranking_score", "forward_pairwise_accuracy"]],
            float_digits=9,
        ),
        "",
        "## Karar",
        "",
        f"**{decision['decision']}**",
        "",
        f"Tam geçmiş seçimi: `{decision['full_history_candidate_key']}`.",
        "",
        "Karar kapıları:",
        "",
        *[f"- `{key}`: `{value}`" for key, value in decision["gates"].items()],
        "",
        "Production manifesti değiştirilmedi. `PROMOTE_CANDIDATE` çıksa bile aktivasyon ayrı kullanıcı onayı gerektirir.",
        "",
        "## Metodolojik Notlar",
        "",
        "- Profil/cap seçimi yalnız fold eğitim sezonlarında yapıldı; test sezonu seçime girmedi.",
        "- Aynı-sezon ranking, sezon sonu rating ile aynı sezon schedule-adjusted performansı karşılaştırır.",
        "- Forward-ranking, kalıcı `club_id` ile sezon sonu ratingi izleyen sezon performansına bağlar; 2025/26 için 2026/27 kullanılmadı.",
        "- Final bonusu aynı sezonda sonraki maç olmadığı için 1X2 loss'a doğrudan etki etmez; esas sinyali sezon sonu ve forward rankingdir.",
        f"- Tie kronoloji audit satırı: `{len(tie_audit)}`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame, *, float_digits: int = 6) -> str:
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
                values.append(f"{float(value):.{float_digits}f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    main()
