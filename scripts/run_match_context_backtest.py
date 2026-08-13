from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ao_elo.config import AOEuropeanEloConfig  # noqa: E402
from ao_elo.controlled_live import calculate_goal_difference_multiplier  # noqa: E402
from ao_elo.evaluation import (  # noqa: E402
    dependency_robust_loss_difference_ci,
    schedule_adjusted_team_performance,
)
from ao_elo.match_context import (  # noqa: E402
    AggregateStateConfig,
    DomesticRegressionConfig,
    DrawContextConfig,
    HomeAdvantageProfile,
    apply_aggregate_state,
    domestic_anchored_start_rating,
    effective_home_advantage,
)
from ao_elo.pipeline import compute_ao_first_elo_from_csv  # noqa: E402
from ao_elo.robustness import one_x_two_probabilities_scalar  # noqa: E402
from scripts.run_dynamic_core_calibration import (  # noqa: E402
    DynamicCoreConfig,
    expanding_folds,
)
from scripts.run_final_robustness import (  # noqa: E402
    ControlledGoalConfig,
    load_team_season_identity,
    rank_value,
    safe_rank_correlation,
    summarize_ranking,
)
from scripts.run_v2_achievement_reserve_calibration import (  # noqa: E402
    ReserveSeasonData,
    load_reserve_data,
)


STATIC_ROOT = ROOT / "data" / "backtest_stage_b_2018_2026"
EVENTS_PATH = (
    ROOT / "data" / "external_elo_benchmark_2018_2026" / "exact_date_events.csv"
)
DYNAMIC_ROOT = ROOT / "output" / "v2_dynamic_calibration_2018_2026"
OUTPUT_ROOT = ROOT / "output" / "match_context_backtest_2018_2026"
CONTROLLED_GOAL = ControlledGoalConfig(0.10, 300.0, 4)
RANK_TOLERANCE = 1e-9


@dataclass(frozen=True)
class ContextModelConfig:
    aggregate: AggregateStateConfig = AggregateStateConfig()
    home: HomeAdvantageProfile = HomeAdvantageProfile()
    draw: DrawContextConfig = DrawContextConfig()
    domestic: DomesticRegressionConfig = DomesticRegressionConfig()

    def validate(self) -> None:
        self.aggregate.validate()
        self.home.validate()
        self.draw.validate()
        self.domestic.validate()


@dataclass(frozen=True)
class ContextSeasonData:
    reserve: ReserveSeasonData
    aggregate_home_goal_differences: np.ndarray
    second_leg_flags: np.ndarray
    home_goals: np.ndarray
    away_goals: np.ndarray
    domestic_priors: np.ndarray

    @property
    def season(self) -> str:
        return self.reserve.season


@dataclass
class ContextEvaluation:
    metrics: dict[str, float | int]
    predictions: pd.DataFrame
    end_ratings: pd.DataFrame
    ranking: pd.DataFrame


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Nested walk-forward tests for aggregate state, dynamic home edge, "
            "contextual draws, and Domestic-Prior season regression"
        )
    )
    parser.add_argument("--static-root", type=Path, default=STATIC_ROOT)
    parser.add_argument("--events-path", type=Path, default=EVENTS_PATH)
    parser.add_argument("--dynamic-root", type=Path, default=DYNAMIC_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()

    dynamic_manifest = json.loads(
        (args.dynamic_root.resolve() / "selected_dynamic_model.json").read_text(
            encoding="utf-8"
        )
    )
    static_config = AOEuropeanEloConfig(**dynamic_manifest["static_config"])
    static_config.validate()
    events = read_context_events(args.events_path.resolve())
    datasets, audit = load_context_data(
        args.static_root.resolve(),
        args.events_path.resolve(),
        static_config,
    )
    seasons = tuple(data.season for data in datasets)
    folds = expanding_folds(seasons)
    if len(folds) != 6:
        raise ValueError(f"Expected six outer folds, found {len(folds)}")
    core_selections = pd.read_csv(
        args.dynamic_root.resolve() / "core_fold_selections.csv"
    )
    validate_core_selections(core_selections, folds)
    target = schedule_adjusted_team_performance(events)
    baseline = ContextModelConfig()
    baseline.validate()

    layer_candidates = {
        "aggregate_state": aggregate_candidates(baseline),
        "dynamic_home": home_candidates(baseline),
        "contextual_draw": draw_candidates(baseline),
        "domestic_regression": domestic_candidates(baseline),
    }
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_root / "data_quality_audit.csv", index=False)

    layer_results: dict[str, dict[str, object]] = {}
    for layer, candidates in layer_candidates.items():
        print(f"Match context backtest: {layer} ({len(candidates)} candidates)", flush=True)
        result = run_layer_backtest(
            layer,
            candidates,
            baseline,
            datasets,
            events,
            target,
            folds,
            core_selections,
            bootstrap_samples=args.bootstrap_samples,
        )
        layer_results[layer] = result
        write_layer_outputs(output_root / layer, result)

    joint = run_joint_backtest(
        baseline,
        layer_results,
        datasets,
        events,
        target,
        folds,
        core_selections,
        bootstrap_samples=args.bootstrap_samples,
    )
    write_layer_outputs(output_root / "joint_promoted", joint)
    manifest = build_manifest(layer_results, joint, audit, datasets)
    (output_root / "decision_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=json_default),
        encoding="utf-8",
    )
    write_report(output_root / "backtest_report.md", manifest, layer_results, joint)

    print("AO Elo match-context backtest complete")
    for layer, result in layer_results.items():
        print(f"{layer}: {result['decision']}")
    print(f"joint_promoted: {joint['decision']}")
    print(f"Report: {output_root / 'backtest_report.md'}")


def read_context_events(path: Path) -> pd.DataFrame:
    events = pd.read_csv(path).sort_values(["season", "event_order"]).reset_index(drop=True)
    required = {
        "match_id",
        "season",
        "event_order",
        "competition",
        "tie_id",
        "is_knockout",
        "is_neutral",
        "home_team_id",
        "away_team_id",
        "home_goals",
        "away_goals",
        "actual_home_score",
        "kickoff_utc",
    }
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"Context event data missing columns: {missing}")
    if events["match_id"].isna().any() or events["match_id"].duplicated().any():
        raise ValueError("Context match_id must be non-null and unique")
    events["kickoff_utc"] = pd.to_datetime(events["kickoff_utc"], utc=True, errors="raise")
    for season, frame in events.groupby("season", sort=True):
        if not frame["kickoff_utc"].is_monotonic_increasing:
            raise ValueError(f"{season}: events are not in exact UTC order")
    return derive_aggregate_state(events)


def derive_aggregate_state(events: pd.DataFrame) -> pd.DataFrame:
    result = events.copy()
    result["aggregate_home_goal_difference_before"] = 0
    result["is_second_leg_derived"] = False
    knockout = result.loc[result["is_knockout"].astype(bool)]
    if knockout["tie_id"].isna().any():
        raise ValueError("Every knockout match must have a tie_id")
    for (season, tie_id), tie in knockout.groupby(["season", "tie_id"], sort=False):
        tie = tie.sort_values("event_order")
        if len(tie) not in (1, 2):
            raise ValueError(f"{season}/{tie_id}: expected one or two matches")
        teams = set(tie["home_team_id"].astype(int)) | set(tie["away_team_id"].astype(int))
        if len(teams) != 2:
            raise ValueError(f"{season}/{tie_id}: expected exactly two teams")
        totals = {team_id: 0 for team_id in teams}
        for position, row in enumerate(tie.itertuples()):
            home_id = int(row.home_team_id)
            away_id = int(row.away_team_id)
            result.loc[row.Index, "aggregate_home_goal_difference_before"] = (
                totals[home_id] - totals[away_id]
            )
            result.loc[row.Index, "is_second_leg_derived"] = position == 1
            totals[home_id] += int(row.home_goals)
            totals[away_id] += int(row.away_goals)
    result["aggregate_home_goal_difference_before"] = result[
        "aggregate_home_goal_difference_before"
    ].astype(int)
    result["is_second_leg_derived"] = result["is_second_leg_derived"].astype(bool)
    return result


def load_context_data(
    static_root: Path,
    events_path: Path,
    static_config: AOEuropeanEloConfig,
) -> tuple[tuple[ContextSeasonData, ...], pd.DataFrame]:
    reserve_data, tie_audit = load_reserve_data(static_root, events_path, static_config)
    events = read_context_events(events_path)
    event_index = events.set_index("match_id")
    datasets: list[ContextSeasonData] = []
    audit_rows: list[dict[str, object]] = []
    for reserve in reserve_data:
        core = reserve.goal.carry.core
        aligned = event_index.loc[core.match_ids]
        folder = static_root / reserve.season.replace("/", "-")
        ratings = compute_ao_first_elo_from_csv(
            folder / "teams.csv",
            folder / "country_coefficients.csv",
            folder / "domestic_context.csv",
            folder / "club_european_points.csv",
            static_config,
        )
        domestic = np.full_like(core.initial_ratings, np.nan, dtype=float)
        ids = ratings["team_id"].astype(int).to_numpy()
        domestic[ids] = ratings["domestic_prior"].to_numpy(float)
        if np.isnan(domestic[core.active_team_ids]).any():
            raise ValueError(f"{reserve.season}: active team lacks Domestic Prior")
        datasets.append(
            ContextSeasonData(
                reserve=reserve,
                aggregate_home_goal_differences=aligned[
                    "aggregate_home_goal_difference_before"
                ].to_numpy(int),
                second_leg_flags=aligned["is_second_leg_derived"].to_numpy(bool),
                home_goals=aligned["home_goals"].to_numpy(int),
                away_goals=aligned["away_goals"].to_numpy(int),
                domestic_priors=domestic,
            )
        )
        audit_rows.append(
            {
                "season": reserve.season,
                "matches": len(aligned),
                "knockout_matches": int(aligned["is_knockout"].astype(bool).sum()),
                "second_leg_matches": int(aligned["is_second_leg_derived"].sum()),
                "neutral_matches": int(aligned["is_neutral"].astype(bool).sum()),
                "aggregate_state_nonzero": int(
                    aligned["aggregate_home_goal_difference_before"].ne(0).sum()
                ),
                "domestic_prior_missing": int(np.isnan(domestic[core.active_team_ids]).sum()),
                "exact_utc": bool(aligned["kickoff_utc"].notna().all()),
            }
        )
    audit = pd.DataFrame(audit_rows)
    audit.loc[len(audit)] = {
        "season": "ALL",
        "matches": int(audit["matches"].sum()),
        "knockout_matches": int(audit["knockout_matches"].sum()),
        "second_leg_matches": int(audit["second_leg_matches"].sum()),
        "neutral_matches": int(audit["neutral_matches"].sum()),
        "aggregate_state_nonzero": int(audit["aggregate_state_nonzero"].sum()),
        "domestic_prior_missing": int(audit["domestic_prior_missing"].sum()),
        "exact_utc": bool(audit["exact_utc"].all()),
    }
    if len(tie_audit):
        audit["tie_chronology_corrections"] = 0
        audit.loc[audit["season"].eq("ALL"), "tie_chronology_corrections"] = len(tie_audit)
    return tuple(datasets), audit


def aggregate_candidates(baseline: ContextModelConfig) -> tuple[ContextModelConfig, ...]:
    return tuple(
        replace(baseline, aggregate=AggregateStateConfig(points, 3))
        for points in (0.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0)
    )


def home_candidates(baseline: ContextModelConfig) -> tuple[ContextModelConfig, ...]:
    candidates = {
        replace(
            baseline,
            home=HomeAdvantageProfile(1.0, uel, uecl, knockout, second),
        )
        for uel in (0.75, 1.0, 1.25)
        for uecl in (0.75, 1.0, 1.25)
        for knockout in (0.75, 1.0, 1.25)
        for second in (0.50, 0.75, 1.0, 1.25)
    }
    return tuple(sorted(candidates, key=config_key))


def draw_candidates(baseline: ContextModelConfig) -> tuple[ContextModelConfig, ...]:
    candidates = {baseline}
    bases = (0.20, 0.22, 0.24, 0.26, 0.28, 0.30)
    shapes = (1.0, 1.25, 1.50)
    offsets = (-0.02, 0.0, 0.02)
    for base in bases:
        for shape in shapes:
            for uel in offsets:
                for uecl in offsets:
                    candidates.add(
                        replace(
                            baseline,
                            draw=DrawContextConfig(base, shape, uel, uecl, 0.0, 0.0),
                        )
                    )
            for knockout in offsets:
                for second in offsets:
                    candidates.add(
                        replace(
                            baseline,
                            draw=DrawContextConfig(
                                base, shape, 0.0, 0.0, knockout, second
                            ),
                        )
                    )
    result = tuple(sorted(candidates, key=config_key))
    for candidate in result:
        candidate.validate()
    return result


def domestic_candidates(baseline: ContextModelConfig) -> tuple[ContextModelConfig, ...]:
    candidates = [baseline]
    candidates.extend(
        replace(
            baseline,
            domestic=DomesticRegressionConfig("DOMESTIC_ANCHORED", persistence),
        )
        for persistence in (0.0, 0.25, 0.50, 0.75, 1.0)
    )
    return tuple(candidates)


def core_for_fold(core_selections: pd.DataFrame, fold: int) -> DynamicCoreConfig:
    row = core_selections.loc[core_selections["fold"].eq(fold)].iloc[0]
    config = DynamicCoreConfig(
        float(row["selected_scale"]),
        float(row["selected_home_advantage"]),
        float(row["selected_k"]),
    )
    config.validate()
    return config


def validate_core_selections(
    selections: pd.DataFrame,
    folds: tuple[tuple[tuple[str, ...], str], ...],
) -> None:
    if set(selections["fold"]) != set(range(1, len(folds) + 1)):
        raise ValueError("Core selections do not cover all outer folds")
    for fold, (_, test_season) in enumerate(folds, start=1):
        row = selections.loc[selections["fold"].eq(fold)]
        if len(row) != 1 or row.iloc[0]["test_season"] != test_season:
            raise ValueError(f"Core selection mismatch in fold {fold}")


def evaluate_context_sequence(
    datasets: tuple[ContextSeasonData, ...],
    core: DynamicCoreConfig,
    config: ContextModelConfig,
    target: pd.DataFrame,
    *,
    evaluation_seasons: set[str] | None = None,
    ranking_target_seasons: set[str] | None = None,
    return_predictions: bool = False,
) -> ContextEvaluation:
    core.validate()
    config.validate()
    CONTROLLED_GOAL.validate()
    evaluation = evaluation_seasons or {data.season for data in datasets}
    previous_power: dict[str, float] = {}
    metric_rows: list[dict[str, float | int]] = []
    prediction_rows: list[dict[str, object]] = []
    end_rows: list[dict[str, object]] = []

    for data in datasets:
        reserve = data.reserve
        season_core = reserve.goal.carry.core
        club_keys = reserve.goal.carry.club_keys
        power = season_core.initial_ratings.copy()
        start = power.copy()
        prior_power_by_team = np.full_like(power, np.nan, dtype=float)
        for team_id in season_core.active_team_ids:
            key = str(club_keys[team_id])
            prior_power = previous_power.get(key)
            if prior_power is not None:
                prior_power_by_team[team_id] = prior_power
            power[team_id] = domestic_anchored_start_rating(
                float(season_core.initial_ratings[team_id]),
                float(data.domestic_priors[team_id]),
                prior_power,
                config.domestic,
            )
            start[team_id] = power[team_id]

        brier_sum = 0.0
        log_sum = 0.0
        max_delta = 0.0
        max_pair_error = 0.0
        for index, (home_id, away_id, actual, neutral, competition) in enumerate(
            zip(
                season_core.home_team_ids,
                season_core.away_team_ids,
                season_core.actual_home_scores,
                season_core.neutral_flags,
                season_core.competitions,
            )
        ):
            competition = str(competition)
            knockout = bool(reserve.knockout_flags[index])
            second_leg = bool(data.second_leg_flags[index])
            home_edge = effective_home_advantage(
                core.home_advantage,
                competition,
                is_neutral=bool(neutral),
                is_knockout=knockout,
                is_second_leg=second_leg,
                profile=config.home,
            )
            raw_difference = float(power[home_id] - power[away_id] + home_edge)
            effective_difference = apply_aggregate_state(
                raw_difference,
                int(data.aggregate_home_goal_differences[index]),
                is_second_leg=second_leg,
                config=config.aggregate,
            )
            expected = float(
                np.clip(
                    1.0 / (1.0 + 10.0 ** (-effective_difference / core.elo_scale)),
                    1e-12,
                    1.0 - 1e-12,
                )
            )
            goal_multiplier = calculate_goal_difference_multiplier(
                int(reserve.goal.goal_differences[index]),
                effective_difference,
                CONTROLLED_GOAL.alpha,
                CONTROLLED_GOAL.tau,
                decided_on_penalties=bool(reserve.goal.penalty_flags[index]),
                is_draw=float(actual) == 0.5,
                goal_cap=CONTROLLED_GOAL.goal_cap,
            )
            delta = core.k_factor * goal_multiplier * (float(actual) - expected)
            pair_before = float(power[home_id] + power[away_id])
            power[home_id] += delta
            power[away_id] -= delta
            max_pair_error = max(
                max_pair_error,
                abs(float(power[home_id] + power[away_id]) - pair_before),
            )
            max_delta = max(max_delta, abs(float(delta)))

            draw_at_even = config.draw.for_match(
                competition,
                is_knockout=knockout,
                is_second_leg=second_leg,
            )
            probabilities = one_x_two_probabilities_scalar(
                expected,
                draw_at_even,
                config.draw.draw_shape,
            )
            observed = 0 if actual == 1.0 else 1 if actual == 0.5 else 2
            target_vector = tuple(
                1.0 if position == observed else 0.0 for position in range(3)
            )
            brier = sum(
                (probability - observed_value) ** 2
                for probability, observed_value in zip(probabilities, target_vector)
            )
            log_loss = -math.log(max(probabilities[observed], 1e-15))
            if data.season in evaluation:
                brier_sum += brier
                log_sum += log_loss
                if return_predictions:
                    prediction_rows.append(
                        {
                            "match_id": str(season_core.match_ids[index]),
                            "season": data.season,
                            "competition": competition,
                            "home_team_id": int(home_id),
                            "away_team_id": int(away_id),
                            "actual_home_score": float(actual),
                            "expected_home_score": expected,
                            "home_probability": probabilities[0],
                            "draw_probability": probabilities[1],
                            "away_probability": probabilities[2],
                            "brier_1x2": brier,
                            "log_loss_1x2": log_loss,
                            "is_knockout": knockout,
                            "is_second_leg": second_leg,
                            "aggregate_home_goal_difference": int(
                                data.aggregate_home_goal_differences[index]
                            ),
                            "effective_home_advantage": home_edge,
                            "effective_rating_difference": effective_difference,
                            "goal_multiplier": goal_multiplier,
                            "power_delta": float(delta),
                        }
                    )

        active = season_core.active_team_ids
        previous_power = {
            str(club_keys[team_id]): float(power[team_id]) for team_id in active
        }
        if data.season in evaluation:
            matches = len(season_core.match_ids)
            metric_rows.append(
                {
                    "matches": matches,
                    "brier_1x2": brier_sum / matches,
                    "log_loss_1x2": log_sum / matches,
                    "max_abs_match_delta": max_delta,
                    "max_pair_sum_error": max_pair_error,
                    "max_abs_rating_change": float(
                        np.max(np.abs(power[active] - start[active]))
                    ),
                    "start_end_rank_correlation": safe_rank_correlation(
                        start[active], power[active]
                    ),
                }
            )
            end_rows.extend(
                {
                    "season": data.season,
                    "team_id": int(team_id),
                    "initial_rating": float(start[team_id]),
                    "ao_first_elo": float(season_core.initial_ratings[team_id]),
                    "domestic_prior": float(data.domestic_priors[team_id]),
                    "previous_power_elo": (
                        float(prior_power_by_team[team_id])
                        if np.isfinite(prior_power_by_team[team_id])
                        else np.nan
                    ),
                    "end_live_rating": float(power[team_id]),
                }
                for team_id in active
            )

    if not metric_rows:
        raise ValueError("No context evaluation seasons were processed")
    matches = sum(int(row["matches"]) for row in metric_rows)
    metrics: dict[str, float | int] = {
        "matches": matches,
        "brier_1x2": sum(
            float(row["brier_1x2"]) * int(row["matches"]) for row in metric_rows
        )
        / matches,
        "log_loss_1x2": sum(
            float(row["log_loss_1x2"]) * int(row["matches"]) for row in metric_rows
        )
        / matches,
        "max_abs_match_delta": max(row["max_abs_match_delta"] for row in metric_rows),
        "max_pair_sum_error": max(row["max_pair_sum_error"] for row in metric_rows),
        "max_abs_rating_change": max(
            row["max_abs_rating_change"] for row in metric_rows
        ),
        "start_end_rank_correlation": min(
            row["start_end_rank_correlation"] for row in metric_rows
        ),
    }
    end_ratings = pd.DataFrame(end_rows)
    ranking = summarize_ranking(
        end_ratings,
        target,
        allowed_target_seasons=ranking_target_seasons
        or {data.season for data in datasets},
        identity=load_team_season_identity(),
    )
    return ContextEvaluation(
        metrics,
        pd.DataFrame(prediction_rows),
        end_ratings,
        ranking,
    )


def run_layer_backtest(
    layer: str,
    candidates: tuple[ContextModelConfig, ...],
    baseline: ContextModelConfig,
    datasets: tuple[ContextSeasonData, ...],
    events: pd.DataFrame,
    target: pd.DataFrame,
    folds: tuple[tuple[tuple[str, ...], str], ...],
    core_selections: pd.DataFrame,
    *,
    bootstrap_samples: int,
) -> dict[str, object]:
    train_frames: list[pd.DataFrame] = []
    selection_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    selected_configs: dict[int, ContextModelConfig] = {}

    for fold, (train_seasons, test_season) in enumerate(folds, start=1):
        core = core_for_fold(core_selections, fold)
        train_data = tuple(data for data in datasets if data.season in train_seasons)
        train_table = context_metric_table(train_data, core, candidates, baseline, target)
        selected_row = select_context_candidate(train_table, config_key(baseline))
        selected = next(
            candidate
            for candidate in candidates
            if config_key(candidate) == selected_row["candidate_key"]
        )
        selected_configs[fold] = selected
        train_table.insert(0, "fold", fold)
        train_table.insert(1, "test_season", test_season)
        train_frames.append(train_table)
        selection_rows.append(
            {
                "fold": fold,
                "train_seasons": "|".join(train_seasons),
                "test_season": test_season,
                "selected_candidate": config_key(selected),
                "selected_is_baseline": selected == baseline,
                "train_brier_1x2": float(selected_row["brier_1x2"]),
                "train_log_loss_1x2": float(selected_row["log_loss_1x2"]),
                "train_ranking_score": float(selected_row["ranking_score"]),
                "train_pairwise_accuracy": float(selected_row["pairwise_accuracy"]),
                "selected_config": json.dumps(asdict(selected), sort_keys=True),
            }
        )
        sequence = tuple(
            data for data in datasets if seasons_index(datasets, data.season) <= seasons_index(datasets, test_season)
        )
        evaluations = {}
        for model, candidate in (("candidate", selected), ("baseline", baseline)):
            evaluated = evaluate_context_sequence(
                sequence,
                core,
                candidate,
                target,
                evaluation_seasons={test_season},
                ranking_target_seasons=set(target["season"]),
                return_predictions=True,
            )
            evaluations[model] = evaluated
            fold_rows.append(evaluated_row(evaluated, fold, test_season, model, candidate))
        prediction_frames.append(
            paired_predictions(evaluations["candidate"], evaluations["baseline"], fold, events)
        )
        print(f"  {layer} outer fold {fold}/{len(folds)}", flush=True)

    full_core = full_core_config(core_selections)
    full_table = context_metric_table(datasets, full_core, candidates, baseline, target)
    full_selected_row = select_context_candidate(full_table, config_key(baseline))
    full_candidate = next(
        candidate
        for candidate in candidates
        if config_key(candidate) == full_selected_row["candidate_key"]
    )
    selections = pd.DataFrame(selection_rows)
    fold_results = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    uncertainty = comparison_uncertainty(predictions, bootstrap_samples)
    ranking_uncertainty = ranking_difference_uncertainty(
        fold_results, bootstrap_samples
    )
    competition = competition_summary(predictions)
    guardrails = layer_guardrails(
        fold_results,
        predictions,
        competition,
        uncertainty,
        ranking_uncertainty,
    )
    promotion_gates = (
        "brier_fold_gate",
        "log_loss_fold_gate",
        "ranking_gate",
        "overall_loss_gate",
        "competition_gate",
        "zero_sum_gate",
    )
    if all(guardrails[gate] for gate in promotion_gates):
        decision = "PROMOTE"
    elif all(
        guardrails[gate]
        for gate in (
            "brier_fold_gate",
            "log_loss_fold_gate",
            "overall_loss_gate",
            "competition_gate",
            "zero_sum_gate",
        )
    ):
        decision = "SHADOW_ONLY_RANKING_GAP"
    else:
        decision = "NO_PROMOTION"
    return {
        "layer": layer,
        "decision": decision,
        "candidates": candidates,
        "selected_configs": selected_configs,
        "full_candidate": full_candidate,
        "train_metrics": pd.concat(train_frames, ignore_index=True),
        "fold_selections": selections,
        "fold_results": fold_results,
        "predictions": predictions,
        "competition_summary": competition,
        "uncertainty": uncertainty,
        "ranking_uncertainty": ranking_uncertainty,
        "guardrails": guardrails,
        "full_metrics": full_table,
    }


def context_metric_table(
    datasets: tuple[ContextSeasonData, ...],
    core: DynamicCoreConfig,
    candidates: tuple[ContextModelConfig, ...],
    baseline: ContextModelConfig,
    target: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        evaluated = evaluate_context_sequence(datasets, core, candidate, target)
        rows.append(
            {
                "candidate_key": config_key(candidate),
                "is_baseline": candidate == baseline,
                "distance": config_distance(candidate, baseline),
                **evaluated.metrics,
                "ranking_score": rank_value(evaluated.ranking, "ranking_score"),
                "pairwise_accuracy": rank_value(evaluated.ranking, "pairwise_accuracy"),
                "config": json.dumps(asdict(candidate), sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def select_context_candidate(table: pd.DataFrame, baseline_key: str) -> pd.Series:
    baseline = table.loc[table["candidate_key"].eq(baseline_key)]
    if len(baseline) != 1:
        raise ValueError("Candidate table must contain one baseline")
    baseline_row = baseline.iloc[0]
    eligible = table.loc[
        table["ranking_score"].ge(baseline_row["ranking_score"] - RANK_TOLERANCE)
        & table["pairwise_accuracy"].ge(
            baseline_row["pairwise_accuracy"] - RANK_TOLERANCE
        )
        & table["max_pair_sum_error"].le(1e-9)
    ]
    if eligible.empty:
        return baseline_row
    return eligible.sort_values(
        ["brier_1x2", "log_loss_1x2", "distance", "candidate_key"],
        kind="stable",
    ).iloc[0]


def run_joint_backtest(
    baseline: ContextModelConfig,
    layer_results: dict[str, dict[str, object]],
    datasets: tuple[ContextSeasonData, ...],
    events: pd.DataFrame,
    target: pd.DataFrame,
    folds: tuple[tuple[tuple[str, ...], str], ...],
    core_selections: pd.DataFrame,
    *,
    bootstrap_samples: int,
) -> dict[str, object]:
    promoted = [
        layer for layer, result in layer_results.items() if result["decision"] == "PROMOTE"
    ]
    fold_rows = []
    selection_rows = []
    prediction_frames = []
    selected_configs: dict[int, ContextModelConfig] = {}
    for fold, (_, test_season) in enumerate(folds, start=1):
        joint = baseline
        for layer in promoted:
            selected = layer_results[layer]["selected_configs"][fold]
            joint = apply_layer(joint, selected, layer)
        selected_configs[fold] = joint
        sequence = tuple(
            data for data in datasets if seasons_index(datasets, data.season) <= seasons_index(datasets, test_season)
        )
        core = core_for_fold(core_selections, fold)
        evaluations = {}
        for model, candidate in (("candidate", joint), ("baseline", baseline)):
            evaluated = evaluate_context_sequence(
                sequence,
                core,
                candidate,
                target,
                evaluation_seasons={test_season},
                ranking_target_seasons=set(target["season"]),
                return_predictions=True,
            )
            evaluations[model] = evaluated
            fold_rows.append(evaluated_row(evaluated, fold, test_season, model, candidate))
        prediction_frames.append(
            paired_predictions(evaluations["candidate"], evaluations["baseline"], fold, events)
        )
        selection_rows.append(
            {
                "fold": fold,
                "test_season": test_season,
                "promoted_layers": "|".join(promoted),
                "selected_candidate": config_key(joint),
                "selected_config": json.dumps(asdict(joint), sort_keys=True),
            }
        )
    fold_results = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    uncertainty = comparison_uncertainty(predictions, bootstrap_samples)
    ranking_uncertainty = ranking_difference_uncertainty(
        fold_results, bootstrap_samples
    )
    competition = competition_summary(predictions)
    guardrails = layer_guardrails(
        fold_results,
        predictions,
        competition,
        uncertainty,
        ranking_uncertainty,
    )
    decision = (
        "PROMOTE_JOINT_CONTEXT"
        if promoted and all(guardrails[gate] for gate in (
            "brier_fold_gate",
            "log_loss_fold_gate",
            "ranking_gate",
            "overall_loss_gate",
            "competition_gate",
            "zero_sum_gate",
        ))
        else "KEEP_CURRENT_PRODUCTION"
    )
    full_candidate = baseline
    for layer in promoted:
        full_candidate = apply_layer(full_candidate, layer_results[layer]["full_candidate"], layer)
    return {
        "layer": "joint_promoted",
        "decision": decision,
        "promoted_layers": promoted,
        "selected_configs": selected_configs,
        "full_candidate": full_candidate,
        "train_metrics": pd.DataFrame(),
        "fold_selections": pd.DataFrame(selection_rows),
        "fold_results": fold_results,
        "predictions": predictions,
        "competition_summary": competition,
        "uncertainty": uncertainty,
        "ranking_uncertainty": ranking_uncertainty,
        "guardrails": guardrails,
        "full_metrics": pd.DataFrame(),
    }


def apply_layer(
    target: ContextModelConfig,
    source: ContextModelConfig,
    layer: str,
) -> ContextModelConfig:
    try:
        field = {
            "aggregate_state": "aggregate",
            "dynamic_home": "home",
            "contextual_draw": "draw",
            "domestic_regression": "domestic",
        }[layer]
    except KeyError as error:
        raise ValueError(f"Unknown layer: {layer}") from error
    return replace(target, **{field: getattr(source, field)})


def evaluated_row(
    evaluation: ContextEvaluation,
    fold: int,
    test_season: str,
    model: str,
    config: ContextModelConfig,
) -> dict[str, object]:
    return {
        "fold": fold,
        "test_season": test_season,
        "model": model,
        "candidate_key": config_key(config),
        **evaluation.metrics,
        "ranking_score": rank_value(evaluation.ranking, "ranking_score"),
        "pairwise_accuracy": rank_value(evaluation.ranking, "pairwise_accuracy"),
    }


def paired_predictions(
    candidate: ContextEvaluation,
    baseline: ContextEvaluation,
    fold: int,
    events: pd.DataFrame,
) -> pd.DataFrame:
    candidate_columns = candidate.predictions.rename(
        columns={
            column: f"candidate_{column}"
            for column in (
                "expected_home_score",
                "home_probability",
                "draw_probability",
                "away_probability",
                "brier_1x2",
                "log_loss_1x2",
                "effective_home_advantage",
                "effective_rating_difference",
                "power_delta",
            )
        }
    )
    baseline_columns = baseline.predictions[[
        "match_id",
        "expected_home_score",
        "home_probability",
        "draw_probability",
        "away_probability",
        "brier_1x2",
        "log_loss_1x2",
        "effective_home_advantage",
        "effective_rating_difference",
        "power_delta",
    ]].rename(columns={column: f"baseline_{column}" for column in (
        "expected_home_score",
        "home_probability",
        "draw_probability",
        "away_probability",
        "brier_1x2",
        "log_loss_1x2",
        "effective_home_advantage",
        "effective_rating_difference",
        "power_delta",
    )})
    paired = candidate_columns.merge(baseline_columns, on="match_id", validate="one_to_one")
    metadata = events[["match_id", "tie_id", "kickoff_utc"]]
    paired = paired.merge(metadata, on="match_id", validate="one_to_one")
    paired.insert(0, "fold", fold)
    paired["brier_difference"] = (
        paired["candidate_brier_1x2"] - paired["baseline_brier_1x2"]
    )
    paired["log_loss_difference"] = (
        paired["candidate_log_loss_1x2"] - paired["baseline_log_loss_1x2"]
    )
    return paired


def comparison_uncertainty(predictions: pd.DataFrame, bootstrap_samples: int) -> pd.DataFrame:
    frames = []
    for metric, column in (
        ("brier_1x2", "brier_difference"),
        ("log_loss_1x2", "log_loss_difference"),
    ):
        frame = dependency_robust_loss_difference_ci(
            predictions,
            difference_column=column,
            bootstrap_samples=bootstrap_samples,
        )
        frame.insert(0, "metric", metric)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def ranking_difference_uncertainty(
    fold_results: pd.DataFrame,
    bootstrap_samples: int,
    *,
    seed: int = 20260810,
) -> pd.DataFrame:
    """Quantify ranking deltas with target-season folds as dependency clusters."""
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    pivot = fold_results.pivot(index="fold", columns="model")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for metric in ("ranking_score", "pairwise_accuracy"):
        differences = (
            pivot[metric]["candidate"] - pivot[metric]["baseline"]
        ).dropna().to_numpy(float)
        if len(differences) < 2:
            raise ValueError(
                f"At least two evaluable ranking folds are required for {metric}"
            )
        mean_difference = float(differences.mean())
        samples = rng.choice(
            differences,
            size=(bootstrap_samples, len(differences)),
            replace=True,
        ).mean(axis=1)
        bootstrap_lower, bootstrap_upper = np.quantile(samples, [0.025, 0.975])
        standard_error = float(differences.std(ddof=1) / math.sqrt(len(differences)))
        critical = float(student_t.ppf(0.975, df=len(differences) - 1))
        t_lower = mean_difference - critical * standard_error
        t_upper = mean_difference + critical * standard_error
        methods = (
            ("target_season_cluster_bootstrap", bootstrap_lower, bootstrap_upper),
            ("small_sample_t_interval", t_lower, t_upper),
        )
        for method, lower, upper in methods:
            rows.append(
                {
                    "metric": metric,
                    "method": method,
                    "evaluable_folds": len(differences),
                    "mean_difference": mean_difference,
                    "ci_95_lower": float(lower),
                    "ci_95_upper": float(upper),
                    "reliable_improvement": bool(lower > 0.0),
                    "reliable_harm": bool(upper < 0.0),
                }
            )
        envelope_lower = min(bootstrap_lower, t_lower)
        envelope_upper = max(bootstrap_upper, t_upper)
        rows.append(
            {
                "metric": metric,
                "method": "conservative_envelope",
                "evaluable_folds": len(differences),
                "mean_difference": mean_difference,
                "ci_95_lower": float(envelope_lower),
                "ci_95_upper": float(envelope_upper),
                "reliable_improvement": bool(envelope_lower > 0.0),
                "reliable_harm": bool(envelope_upper < 0.0),
            }
        )
    return pd.DataFrame(rows)


def competition_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for competition, frame in [("ALL", predictions), *predictions.groupby("competition", sort=True)]:
        rows.append(
            {
                "competition": competition,
                "matches": len(frame),
                "candidate_brier_1x2": float(frame["candidate_brier_1x2"].mean()),
                "baseline_brier_1x2": float(frame["baseline_brier_1x2"].mean()),
                "brier_difference": float(frame["brier_difference"].mean()),
                "candidate_log_loss_1x2": float(
                    frame["candidate_log_loss_1x2"].mean()
                ),
                "baseline_log_loss_1x2": float(
                    frame["baseline_log_loss_1x2"].mean()
                ),
                "log_loss_difference": float(frame["log_loss_difference"].mean()),
            }
        )
    return pd.DataFrame(rows)


def layer_guardrails(
    fold_results: pd.DataFrame,
    predictions: pd.DataFrame,
    competition: pd.DataFrame,
    uncertainty: pd.DataFrame,
    ranking_uncertainty: pd.DataFrame | None = None,
) -> dict[str, object]:
    pivot = fold_results.pivot(index="fold", columns="model")
    brier_delta = pivot["brier_1x2"]["candidate"] - pivot["brier_1x2"]["baseline"]
    log_delta = pivot["log_loss_1x2"]["candidate"] - pivot["log_loss_1x2"]["baseline"]
    rank_delta = pivot["ranking_score"]["candidate"] - pivot["ranking_score"]["baseline"]
    pair_delta = pivot["pairwise_accuracy"]["candidate"] - pivot["pairwise_accuracy"]["baseline"]
    rank_evaluable = rank_delta.notna() & pair_delta.notna()
    rank_safe = (rank_delta[rank_evaluable] >= -RANK_TOLERANCE) & (
        pair_delta[rank_evaluable] >= -RANK_TOLERANCE
    )
    if ranking_uncertainty is None:
        ranking_uncertainty = ranking_difference_uncertainty(
            fold_results, bootstrap_samples=2000
        )
    ranking_envelope = ranking_uncertainty.loc[
        ranking_uncertainty["method"].eq("conservative_envelope")
    ]
    if set(ranking_envelope["metric"]) != {
        "ranking_score",
        "pairwise_accuracy",
    }:
        raise ValueError("Ranking uncertainty must contain both ranking metrics")
    reliable_ranking_harm = bool(ranking_envelope["reliable_harm"].any())
    spearman_interval = ranking_envelope.loc[
        ranking_envelope["metric"].eq("ranking_score")
    ].iloc[0]
    pairwise_interval = ranking_envelope.loc[
        ranking_envelope["metric"].eq("pairwise_accuracy")
    ].iloc[0]
    brier_uncertainty = uncertainty.loc[uncertainty["metric"].eq("brier_1x2")]
    envelope_upper = float(brier_uncertainty["ci_95_upper"].max())
    overall = competition.loc[competition["competition"].eq("ALL")].iloc[0]
    segments = competition.loc[competition["competition"].ne("ALL")]
    return {
        "brier_fold_wins": int((brier_delta < -1e-12).sum()),
        "log_loss_fold_wins": int((log_delta < -1e-12).sum()),
        "ranking_evaluable_folds": int(rank_evaluable.sum()),
        "ranking_no_regression_folds": int(rank_safe.sum()),
        "mean_ranking_score_difference": float(spearman_interval["mean_difference"]),
        "ranking_score_conservative_lower_95": float(spearman_interval["ci_95_lower"]),
        "ranking_score_conservative_upper_95": float(spearman_interval["ci_95_upper"]),
        "mean_pairwise_accuracy_difference": float(pairwise_interval["mean_difference"]),
        "pairwise_accuracy_conservative_lower_95": float(pairwise_interval["ci_95_lower"]),
        "pairwise_accuracy_conservative_upper_95": float(pairwise_interval["ci_95_upper"]),
        "ranking_reliable_harm": reliable_ranking_harm,
        "overall_brier_difference": float(overall["brier_difference"]),
        "overall_log_loss_difference": float(overall["log_loss_difference"]),
        "conservative_brier_upper_95": envelope_upper,
        "max_competition_brier_difference": float(segments["brier_difference"].max()),
        "max_pair_sum_error": float(fold_results["max_pair_sum_error"].max()),
        "brier_fold_gate": int((brier_delta < -1e-12).sum()) >= 4,
        "log_loss_fold_gate": int((log_delta < -1e-12).sum()) >= 4,
        "ranking_gate": bool(
            int(rank_evaluable.sum()) >= 5 and not reliable_ranking_harm
        ),
        "overall_loss_gate": bool(
            overall["brier_difference"] < 0.0
            and overall["log_loss_difference"] < 0.0
            and envelope_upper <= 0.0
        ),
        "competition_gate": bool(segments["brier_difference"].max() <= 0.0),
        "zero_sum_gate": bool(fold_results["max_pair_sum_error"].max() <= 1e-9),
    }


def full_core_config(core_selections: pd.DataFrame) -> DynamicCoreConfig:
    manifest = json.loads(
        (DYNAMIC_ROOT / "selected_dynamic_model.json").read_text(encoding="utf-8")
    )
    config = DynamicCoreConfig(**manifest["dynamic_core"])
    config.validate()
    return config


def config_key(config: ContextModelConfig) -> str:
    return json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))


def config_distance(candidate: ContextModelConfig, baseline: ContextModelConfig) -> float:
    candidate_values = flatten_numeric(asdict(candidate))
    baseline_values = flatten_numeric(asdict(baseline))
    return float(
        sum(abs(candidate_values[key] - baseline_values.get(key, 0.0)) for key in candidate_values)
    )


def flatten_numeric(payload: dict[str, object], prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in payload.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten_numeric(value, full_key))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            result[full_key] = float(value)
    return result


def seasons_index(datasets: tuple[ContextSeasonData, ...], season: str) -> int:
    seasons = tuple(data.season for data in datasets)
    return seasons.index(season)


def write_layer_outputs(path: Path, result: dict[str, object]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for key, filename in (
        ("train_metrics", "train_candidate_metrics.csv"),
        ("fold_selections", "fold_selections.csv"),
        ("fold_results", "fold_results.csv"),
        ("predictions", "unseen_predictions.csv"),
        ("competition_summary", "competition_summary.csv"),
        ("uncertainty", "dependency_uncertainty.csv"),
        ("ranking_uncertainty", "ranking_uncertainty.csv"),
        ("full_metrics", "full_candidate_metrics.csv"),
    ):
        frame = result[key]
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            frame.to_csv(path / filename, index=False)
    decision = {
        "layer": result["layer"],
        "decision": result["decision"],
        "promoted_layers": result.get("promoted_layers", []),
        "full_candidate": asdict(result["full_candidate"]),
        "guardrails": result["guardrails"],
    }
    (path / "decision.json").write_text(
        json.dumps(decision, indent=2, default=json_default), encoding="utf-8"
    )


def build_manifest(
    layers: dict[str, dict[str, object]],
    joint: dict[str, object],
    audit: pd.DataFrame,
    datasets: tuple[ContextSeasonData, ...],
) -> dict[str, object]:
    return {
        "model": "AO European Elo match-context robustness",
        "data_window": "2018/19-2025/26",
        "seasons": len(datasets),
        "matches": int(audit.loc[audit["season"].eq("ALL"), "matches"].iloc[0]),
        "baseline": {
            "fixed_k": 103.98098633392752,
            "controlled_goal": asdict(CONTROLLED_GOAL),
            "draw_at_even": 0.24,
            "draw_shape": 1.0,
        },
        "layers": {
            name: {
                "decision": result["decision"],
                "full_candidate": asdict(result["full_candidate"]),
                "guardrails": result["guardrails"],
            }
            for name, result in layers.items()
        },
        "joint": {
            "decision": joint["decision"],
            "promoted_layers": joint.get("promoted_layers", []),
            "full_candidate": asdict(joint["full_candidate"]),
            "guardrails": joint["guardrails"],
        },
    }


def write_report(
    path: Path,
    manifest: dict[str, object],
    layers: dict[str, dict[str, object]],
    joint: dict[str, object],
) -> None:
    lines = [
        "# AO European Elo Match Context Backtest",
        "",
        "## Scope",
        "",
        "- 2018/19-2025/26 exact-UTC European matches",
        f"- Matches: `{manifest['matches']}`",
        "- Six nested outer folds; ranking-first candidate eligibility",
        "- Baseline: fixed K, controlled GD 0.10/300/cap 4, draw 0.24/1.00",
        "- No 2026/27 data used",
        "",
        "## Decisions",
        "",
        "| Layer | Decision | Brier delta | Log-loss delta | Fold wins | Ranking safe | Reliable ranking harm |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for layer, result in layers.items():
        guard = result["guardrails"]
        lines.append(
            f"| {layer} | {result['decision']} | "
            f"{guard['overall_brier_difference']:+.6f} | "
            f"{guard['overall_log_loss_difference']:+.6f} | "
            f"{guard['brier_fold_wins']}/6 | "
            f"{guard['ranking_no_regression_folds']}/{guard['ranking_evaluable_folds']} | "
            f"{guard['ranking_reliable_harm']} |"
        )
    lines.extend(
        [
            "",
            "## Forward-Ranking Uncertainty",
            "",
            "Ranking differences use `candidate - baseline`; a negative interval is harm. "
            "The strict fold count is retained as a diagnostic, while veto now requires "
            "the conservative 95% interval to remain entirely below zero.",
            "",
            "| Layer | Spearman mean | Spearman 95% | Pairwise mean | Pairwise 95% | Reliable harm |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for layer, result in layers.items():
        guard = result["guardrails"]
        lines.append(
            f"| {layer} | {guard['mean_ranking_score_difference']:+.6f} | "
            f"[{guard['ranking_score_conservative_lower_95']:+.6f}, "
            f"{guard['ranking_score_conservative_upper_95']:+.6f}] | "
            f"{guard['mean_pairwise_accuracy_difference']:+.6f} | "
            f"[{guard['pairwise_accuracy_conservative_lower_95']:+.6f}, "
            f"{guard['pairwise_accuracy_conservative_upper_95']:+.6f}] | "
            f"{guard['ranking_reliable_harm']} |"
        )
    guard = joint["guardrails"]
    lines.extend(
        [
            "",
            "## Joint Promoted Model",
            "",
            f"- Promoted layers: `{','.join(joint.get('promoted_layers', [])) or 'none'}`",
            f"- Decision: `{joint['decision']}`",
            f"- Brier delta: `{guard['overall_brier_difference']:+.6f}`",
            f"- Log-loss delta: `{guard['overall_log_loss_difference']:+.6f}`",
            f"- Conservative Brier upper 95%: `{guard['conservative_brier_upper_95']:+.6f}`",
            "",
            "A layer is promoted only when it wins at least four outer folds in both loss metrics, "
            "has no dependency-robust reliable forward-ranking harm, has no "
            "competition-level Brier regression, preserves zero-sum updates, and its conservative "
            "Brier interval does not cross zero.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


if __name__ == "__main__":
    main()
