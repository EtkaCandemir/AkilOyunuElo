from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ao_elo.config import AO_MODEL_V2_VERSION, AOEuropeanEloConfig  # noqa: E402
from ao_elo.evaluation import (  # noqa: E402
    dependency_robust_loss_difference_ci,
    schedule_adjusted_team_performance,
)
from ao_elo.robustness import (  # noqa: E402
    CompetitionKCandidate,
    GoalMarginCandidate,
    baseline_competition_k,
    baseline_goal_margin,
    competition_k_candidates,
    goal_margin_candidates,
    goal_margin_multiplier,
    one_x_two_probabilities_scalar,
)
from scripts.run_dynamic_core_calibration import (  # noqa: E402
    DynamicCoreConfig,
    expanding_folds,
    expected_home_score,
)
from scripts.run_external_elo_benchmark import (  # noqa: E402
    CLUBELO_HOME_ADVANTAGES,
    CLUBELO_SCALE,
    clubelo_expected_home_score,
    load_benchmark_data,
)
from scripts.run_ranking_first_calibration import (  # noqa: E402
    pairwise_ranking_accuracy,
)
from scripts.run_v2_achievement_reserve_calibration import (  # noqa: E402
    AchievementReserveConfig,
    ReserveSeasonData,
    baseline_config as baseline_reserve,
    candidate_grid as reserve_candidates,
    load_reserve_data,
)
from scripts.run_v2_evaluation_upgrade import (  # noqa: E402
    DrawModelConfig,
    draw_model_candidates,
    read_events,
)
from scripts.run_v2_dynamic_calibration import (  # noqa: E402
    MAX_RATING_MOVE_GUARDRAIL,
    RANK_CORRELATION_FLOOR,
)


STATIC_DATA_ROOT = ROOT / "data" / "backtest_stage_b_2018_2026"
EVENTS_PATH = (
    ROOT / "data" / "external_elo_benchmark_2018_2026" / "exact_date_events.csv"
)
BENCHMARK_PATH = (
    ROOT
    / "data"
    / "external_elo_benchmark_2018_2026"
    / "matches_with_dates_and_external_elo.csv"
)
DYNAMIC_ROOT = ROOT / "output" / "v2_dynamic_calibration_2018_2026"
EVALUATION_ROOT = ROOT / "output" / "v2_evaluation_upgrade_2018_2026"
OUTPUT_ROOT = ROOT / "output" / "final_robustness_2018_2026"
COMPETITIONS = ("UCL", "UEL", "UECL")
RANK_TOLERANCE = 1e-9


@dataclass
class SequenceEvaluation:
    metrics: dict[str, float | int]
    predictions: pd.DataFrame
    end_ratings: pd.DataFrame
    ranking: pd.DataFrame


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Revalidate ClubElo, goal margin, competition K, and achievement "
            "layers on the frozen carry=0 standard-1X2 AO Elo baseline"
        )
    )
    parser.add_argument("--static-data-root", type=Path, default=STATIC_DATA_ROOT)
    parser.add_argument("--events-path", type=Path, default=EVENTS_PATH)
    parser.add_argument("--benchmark-path", type=Path, default=BENCHMARK_PATH)
    parser.add_argument("--dynamic-root", type=Path, default=DYNAMIC_ROOT)
    parser.add_argument("--evaluation-root", type=Path, default=EVALUATION_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    args = parser.parse_args()

    dynamic_root = args.dynamic_root.resolve()
    evaluation_root = args.evaluation_root.resolve()
    production = read_final_production_contract(
        evaluation_root / "selected_production_model.json"
    )
    dynamic_manifest = json.loads(
        (dynamic_root / "selected_dynamic_model.json").read_text(encoding="utf-8")
    )
    static_config = AOEuropeanEloConfig(**dynamic_manifest["static_config"])
    static_config.validate()
    events = read_events(args.events_path.resolve())
    datasets, tie_audit = load_reserve_data(
        args.static_data_root.resolve(),
        args.events_path.resolve(),
        static_config,
    )
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

    print("Final robustness: goal-margin candidates", flush=True)
    margin = run_margin_calibration(
        datasets,
        events,
        target,
        folds,
        core_selections,
        draw_selections,
        production,
        bootstrap_samples=args.bootstrap_samples,
    )
    active_margin = (
        margin["full_candidate"]
        if margin["decision"] == "PROMOTE_GOAL_MARGIN"
        else baseline_goal_margin()
    )

    print("Final robustness: competition-K candidates", flush=True)
    competition_k = run_competition_k_calibration(
        datasets,
        events,
        target,
        folds,
        core_selections,
        draw_selections,
        production,
        active_margin,
        bootstrap_samples=args.bootstrap_samples,
    )
    active_k = (
        competition_k["full_candidate"]
        if str(competition_k["decision"]).startswith("PROMOTE_")
        else baseline_competition_k()
    )

    print("Final robustness: achievement-reserve candidates", flush=True)
    reserve = run_reserve_calibration(
        datasets,
        events,
        target,
        folds,
        core_selections,
        draw_selections,
        production,
        active_margin,
        active_k,
        bootstrap_samples=args.bootstrap_samples,
    )
    active_reserve = (
        reserve["full_candidate"]
        if reserve["decision"] == "PROMOTE_ACHIEVEMENT_RESERVE"
        else baseline_reserve()
    )

    print("Final robustness: exact-date ClubElo benchmark", flush=True)
    external = run_final_external_benchmark(
        datasets,
        events,
        folds,
        core_selections,
        draw_selections,
        production,
        active_margin,
        active_k,
        active_reserve,
        load_benchmark_data(args.benchmark_path.resolve()),
        bootstrap_samples=args.bootstrap_samples,
    )
    final_oof = run_fixed_oof(
        datasets,
        events,
        target,
        folds,
        core_selections,
        draw_selections,
        active_margin,
        active_k,
        active_reserve,
    )
    residuals = competition_residual_diagnostics(
        final_oof["predictions"],
        bootstrap_samples=args.bootstrap_samples,
    )

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    write_layer_outputs(output_root / "goal_margin", margin)
    write_layer_outputs(output_root / "competition_k", competition_k)
    write_layer_outputs(output_root / "achievement_reserve", reserve)
    write_external_outputs(output_root / "external_clubelo", external)
    residuals["summary"].to_csv(
        output_root / "competition_residual_summary.csv", index=False
    )
    residuals["uncertainty"].to_csv(
        output_root / "competition_residual_uncertainty.csv", index=False
    )
    tie_audit.to_csv(output_root / "tie_chronology_audit.csv", index=False)
    final_oof["predictions"].to_csv(
        output_root / "final_model_oof_predictions.csv", index=False
    )
    final_oof["ranking"].to_csv(
        output_root / "final_model_ranking_summary.csv", index=False
    )
    manifest = build_manifest(
        production,
        margin,
        competition_k,
        reserve,
        external,
        active_margin,
        active_k,
        active_reserve,
    )
    (output_root / "robustness_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=json_default), encoding="utf-8"
    )
    (output_root / "selected_production_model.json").write_text(
        json.dumps(manifest["recommended_production_model"], indent=2),
        encoding="utf-8",
    )
    write_report(
        output_root / "robustness_report.md",
        manifest,
        margin,
        competition_k,
        reserve,
        external,
        residuals,
    )

    print("AO European Elo final robustness package")
    print(f"Goal margin: {margin['decision']}")
    print(f"Competition K: {competition_k['decision']}")
    print(f"Achievement reserve: {reserve['decision']}")
    print(f"External status: {external['status']}")
    print(f"Overall status: {manifest['overall_status']}")
    print(f"Report: {output_root / 'robustness_report.md'}")


def read_final_production_contract(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"dynamic_core", "active_power_carry", "one_x_two_probability"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Production manifest missing keys: {missing}")
    if float(payload["active_power_carry"]) != 0.0:
        raise ValueError("Final robustness baseline requires active_power_carry=0")
    one_x_two = payload["one_x_two_probability"]
    if not one_x_two.get("active") or not one_x_two.get("standard_brier"):
        raise ValueError("Final robustness baseline requires active standard 1X2")
    return payload


def validate_fold_inputs(
    core: pd.DataFrame,
    draw: pd.DataFrame,
    folds: tuple[tuple[tuple[str, ...], str], ...],
) -> None:
    expected = set(range(1, len(folds) + 1))
    if set(core["fold"]) != expected or set(draw["fold"]) != expected:
        raise ValueError("Core/draw selections do not cover the six outer folds")
    if not draw["power_carry"].eq(0.0).all():
        raise ValueError("Draw selections must be calibrated on carry=0")
    for fold, (train, test) in enumerate(folds, start=1):
        core_row = core.loc[core["fold"].eq(fold)].iloc[0]
        if core_row["test_season"] != test:
            raise ValueError(f"Core fold {fold} has a season mismatch")
        draw_rows = draw.loc[draw["fold"].eq(fold)]
        if not draw_rows["test_season"].eq(test).all():
            raise ValueError(f"Draw fold {fold} has a season mismatch")
        if "|".join(train) != core_row["train_seasons"]:
            raise ValueError(f"Core fold {fold} has a training-window mismatch")


def core_for_fold(core: pd.DataFrame, fold: int) -> DynamicCoreConfig:
    row = core.loc[core["fold"].eq(fold)].iloc[0]
    return DynamicCoreConfig(
        float(row["selected_scale"]),
        float(row["selected_home_advantage"]),
        float(row["selected_k"]),
    )


def production_core(production: dict[str, object]) -> DynamicCoreConfig:
    return DynamicCoreConfig(**production["dynamic_core"])


def draw_map_for_fold(draw: pd.DataFrame, fold: int) -> dict[str, DrawModelConfig]:
    rows = draw.loc[draw["fold"].eq(fold)]
    mapping = {
        str(row.competition): DrawModelConfig(
            float(row.draw_at_even), float(row.draw_shape)
        )
        for row in rows.itertuples(index=False)
    }
    if "ALL" not in mapping:
        raise ValueError(f"Fold {fold} lacks a pooled draw mapping")
    return mapping


def production_draw_map(production: dict[str, object]) -> dict[str, DrawModelConfig]:
    config = production["one_x_two_probability"]
    draw = DrawModelConfig(
        float(config["draw_at_even"]),
        float(config["draw_shape"]),
    )
    return {"ALL": draw}


def _draw_for(
    mapping: dict[str, DrawModelConfig], competition: str
) -> DrawModelConfig:
    return mapping.get(competition, mapping["ALL"])


def evaluate_sequence(
    datasets: tuple[ReserveSeasonData, ...],
    core: DynamicCoreConfig,
    margin: GoalMarginCandidate,
    competition_k: CompetitionKCandidate,
    reserve: AchievementReserveConfig,
    draw_mapping: dict[str, DrawModelConfig],
    target: pd.DataFrame,
    *,
    evaluation_seasons: set[str] | None = None,
    ranking_target_seasons: set[str] | None = None,
    return_predictions: bool = False,
) -> SequenceEvaluation:
    core.validate()
    margin.validate()
    competition_k.validate()
    reserve.validate()
    evaluation = evaluation_seasons or {data.season for data in datasets}
    previous_reserve: dict[str, float] = {}
    metric_rows: list[dict[str, float | int]] = []
    prediction_rows: list[dict[str, object]] = []
    end_rows: list[dict[str, object]] = []

    for data in datasets:
        goal = data.goal
        carry = goal.carry
        season = data.season
        season_core = carry.core
        power = season_core.initial_ratings.copy()
        reserve_state = np.zeros_like(power)
        if reserve.reserve_base > 0.0 and reserve.reserve_decay > 0.0:
            for team_id in season_core.active_team_ids:
                key = str(carry.club_keys[team_id])
                reserve_state[team_id] = min(
                    reserve.reserve_cap,
                    reserve.reserve_decay * previous_reserve.get(key, 0.0),
                )

        open_ties: dict[str, tuple[int, int, float]] = {}
        brier_sum = 0.0
        log_sum = 0.0
        max_delta = 0.0
        max_pair_error = 0.0
        reserve_added = 0.0
        trophy_added = 0.0

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
            tie_value = data.tie_ids[index]
            tie_id = None if tie_value is None else str(tie_value)
            reserve_active = reserve.reserve_base > 0.0
            if reserve_active and data.knockout_flags[index] and tie_id not in open_ties:
                if tie_id is None:
                    raise ValueError(f"{season}/{season_core.match_ids[index]}: missing tie_id")
                neutral_tie_expectation = expected_home_score(
                    power[home_id] + reserve_state[home_id],
                    power[away_id] + reserve_state[away_id],
                    core,
                    neutral=True,
                )
                open_ties[tie_id] = (int(home_id), int(away_id), neutral_tie_expectation)

            home_before = float(power[home_id] + reserve_state[home_id])
            away_before = float(power[away_id] + reserve_state[away_id])
            expected = float(
                np.clip(
                    expected_home_score(
                        home_before,
                        away_before,
                        core,
                        neutral=bool(neutral),
                    ),
                    1e-12,
                    1.0 - 1e-12,
                )
            )
            winner_expected = expected if actual == 1.0 else 1.0 - expected if actual == 0.0 else 0.5
            margin_multiplier = goal_margin_multiplier(
                int(goal.goal_differences[index]),
                float(winner_expected),
                margin,
            )
            k_multiplier = competition_k.for_competition(competition)
            delta = core.k_factor * k_multiplier * margin_multiplier * (actual - expected)
            pair_before = float(power[home_id] + power[away_id])
            power[home_id] += delta
            power[away_id] -= delta
            max_pair_error = max(
                max_pair_error,
                abs(float(power[home_id] + power[away_id]) - pair_before),
            )
            max_delta = max(max_delta, abs(float(delta)))

            added = 0.0
            added_trophy = 0.0
            if reserve_active and data.tie_decider_flags[index]:
                if tie_id is None or tie_id not in open_ties:
                    raise ValueError(
                        f"{season}/{season_core.match_ids[index]}: unknown deciding tie"
                    )
                team_a, team_b, expected_a = open_ties.pop(tie_id)
                winner_id = int(data.advanced_team_ids[index])
                if winner_id not in (team_a, team_b):
                    raise ValueError(f"{season}/{tie_id}: invalid advanced team")
                winner_probability = expected_a if winner_id == team_a else 1.0 - expected_a
                stage = str(data.stages[index])
                competition_multiplier = reserve.competition_multiplier(competition)
                added = (
                    reserve.reserve_base
                    * competition_multiplier
                    * reserve.profile.for_stage(stage)
                    * (1.0 - winner_probability)
                )
                if stage == "FINAL":
                    added_trophy = (
                        reserve.reserve_base
                        * competition_multiplier
                        * (1.0 - winner_probability)
                    )
                available = max(0.0, reserve.reserve_cap - reserve_state[winner_id])
                total = min(available, added + added_trophy)
                if added + added_trophy > 0.0:
                    share = added / (added + added_trophy)
                    added = total * share
                    added_trophy = total - added
                reserve_state[winner_id] += total
                reserve_added += added
                trophy_added += added_trophy

            draw = _draw_for(draw_mapping, competition)
            probabilities = one_x_two_probabilities_scalar(
                expected,
                draw.draw_at_even,
                draw.draw_shape,
            )
            observed = 0 if actual == 1.0 else 1 if actual == 0.5 else 2
            target_vector = tuple(1.0 if position == observed else 0.0 for position in range(3))
            brier = sum(
                (probability - observed_value) ** 2
                for probability, observed_value in zip(probabilities, target_vector)
            )
            log_loss = -math.log(max(probabilities[observed], 1e-15))
            if season in evaluation:
                brier_sum += brier
                log_sum += log_loss
                if return_predictions:
                    prediction_rows.append(
                        {
                            "match_id": str(season_core.match_ids[index]),
                            "season": season,
                            "competition": competition,
                            "stage": str(data.stages[index]),
                            "home_team_id": int(home_id),
                            "away_team_id": int(away_id),
                            "actual_home_score": float(actual),
                            "expected_home_score": expected,
                            "home_probability": probabilities[0],
                            "draw_probability": probabilities[1],
                            "away_probability": probabilities[2],
                            "brier_1x2": brier,
                            "log_loss_1x2": log_loss,
                            "goal_difference": int(goal.goal_differences[index]),
                            "goal_multiplier": margin_multiplier,
                            "competition_k_multiplier": k_multiplier,
                            "power_delta": float(delta),
                            "home_live_rating": home_before,
                            "away_live_rating": away_before,
                            "reserve_added_after_match": added + added_trophy,
                        }
                    )

        if open_ties:
            raise ValueError(f"{season}: undecided knockout ties remain open")
        active_ids = season_core.active_team_ids
        end_live = power[active_ids] + reserve_state[active_ids]
        start = season_core.initial_ratings[active_ids]
        if season in evaluation:
            matches = len(season_core.match_ids)
            metric_rows.append(
                {
                    "matches": matches,
                    "brier_1x2": brier_sum / matches,
                    "log_loss_1x2": log_sum / matches,
                    "max_abs_match_delta": max_delta,
                    "max_pair_sum_error": max_pair_error,
                    "max_abs_rating_change": float(np.max(np.abs(end_live - start))),
                    "start_end_rank_correlation": safe_rank_correlation(start, end_live),
                    "max_reserve": float(np.max(reserve_state[active_ids])),
                    "reserve_added": reserve_added,
                    "trophy_added": trophy_added,
                }
            )
            end_rows.extend(
                {
                    "season": season,
                    "team_id": int(team_id),
                    "initial_rating": float(season_core.initial_ratings[team_id]),
                    "end_power_rating": float(power[team_id]),
                    "end_reserve": float(reserve_state[team_id]),
                    "end_live_rating": float(power[team_id] + reserve_state[team_id]),
                }
                for team_id in active_ids
            )
        if reserve_active:
            previous_reserve = {
                str(carry.club_keys[team_id]): float(reserve_state[team_id])
                for team_id in active_ids
            }

    if not metric_rows:
        raise ValueError("No evaluation seasons were processed")
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
        "max_abs_rating_change": max(row["max_abs_rating_change"] for row in metric_rows),
        "start_end_rank_correlation": min(
            row["start_end_rank_correlation"] for row in metric_rows
        ),
        "max_reserve": max(row["max_reserve"] for row in metric_rows),
        "reserve_added": sum(row["reserve_added"] for row in metric_rows),
        "trophy_added": sum(row["trophy_added"] for row in metric_rows),
    }
    end_ratings = pd.DataFrame(end_rows)
    allowed_ranking_targets = ranking_target_seasons or {
        data.season for data in datasets
    }
    ranking = summarize_ranking(
        end_ratings,
        target,
        allowed_target_seasons=allowed_ranking_targets,
    )
    return SequenceEvaluation(
        metrics,
        pd.DataFrame(prediction_rows),
        end_ratings,
        ranking,
    )


def safe_rank_correlation(start: np.ndarray, end: np.ndarray) -> float:
    if np.allclose(start, end):
        return 1.0
    if np.ptp(start) == 0 or np.ptp(end) == 0:
        return 0.0
    value = pd.Series(start).corr(pd.Series(end), method="spearman")
    return 0.0 if pd.isna(value) else float(value)


def summarize_ranking(
    end_ratings: pd.DataFrame,
    target: pd.DataFrame,
    *,
    allowed_target_seasons: set[str],
) -> pd.DataFrame:
    """Compare each season-end rating only with the following season."""
    target_seasons = tuple(sorted(target["season"].unique()))
    previous_season = {
        target_seasons[index]: target_seasons[index - 1]
        for index in range(1, len(target_seasons))
    }
    rows: list[dict[str, object]] = []
    eligible_target = target.loc[target["season"].isin(allowed_target_seasons)]
    for (season, competition), actual in eligible_target.groupby(
        ["season", "competition"], sort=True
    ):
        source_season = previous_season.get(str(season))
        if source_season is None:
            continue
        predicted = end_ratings.loc[end_ratings["season"].eq(source_season)]
        table = actual[["team_id", "schedule_adjusted_score"]].merge(
            predicted[["team_id", "end_live_rating"]],
            on="team_id",
            validate="one_to_one",
        )
        if len(table) < 3:
            continue
        spearman = table["end_live_rating"].corr(
            table["schedule_adjusted_score"], method="spearman"
        )
        pairwise = pairwise_ranking_accuracy(
            table["end_live_rating"].to_numpy(float),
            table["schedule_adjusted_score"].to_numpy(float),
        )
        rows.append(
            {
                "source_season": source_season,
                "target_season": season,
                "competition": competition,
                "teams": len(table),
                "pair_weight": len(table) * (len(table) - 1) / 2,
                "ranking_score": float(spearman),
                "pairwise_accuracy": float(pairwise),
            }
        )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return pd.DataFrame(
            columns=[
                "competition",
                "groups",
                "team_weight",
                "ranking_score",
                "pairwise_accuracy",
            ]
        )
    summaries = []
    for competition, frame in [("ALL", detail), *detail.groupby("competition", sort=True)]:
        summaries.append(
            {
                "competition": competition,
                "groups": len(frame),
                "team_weight": int(frame["teams"].sum()),
                "ranking_score": float(
                    np.average(frame["ranking_score"], weights=frame["teams"])
                ),
                "pairwise_accuracy": float(
                    np.average(
                        frame["pairwise_accuracy"], weights=frame["pair_weight"]
                    )
                ),
            }
        )
    return pd.DataFrame(summaries)


def rank_value(ranking: pd.DataFrame, column: str, competition: str = "ALL") -> float:
    if ranking.empty:
        return float("nan")
    row = ranking.loc[ranking["competition"].eq(competition)]
    if len(row) != 1:
        raise ValueError(f"Ranking summary lacks one {competition} row")
    return float(row.iloc[0][column])


def evaluated_row(
    evaluation: SequenceEvaluation,
    **identifiers: object,
) -> dict[str, object]:
    return {
        **identifiers,
        **evaluation.metrics,
        "ranking_score": rank_value(evaluation.ranking, "ranking_score"),
        "pairwise_accuracy": rank_value(evaluation.ranking, "pairwise_accuracy"),
    }


def candidate_is_eligible(row: pd.Series, baseline: pd.Series) -> bool:
    return bool(
        row["start_end_rank_correlation"] >= RANK_CORRELATION_FLOOR
        and row["max_abs_rating_change"] <= MAX_RATING_MOVE_GUARDRAIL
        and row["max_pair_sum_error"] <= 1e-9
        and row["ranking_score"] >= baseline["ranking_score"] - RANK_TOLERANCE
        and row["pairwise_accuracy"]
        >= baseline["pairwise_accuracy"] - RANK_TOLERANCE
    )


def select_ranking_first(
    metrics: pd.DataFrame,
    baseline_key: str,
) -> pd.Series:
    baseline = metrics.loc[metrics["candidate_key"].eq(baseline_key)]
    if len(baseline) != 1:
        raise ValueError(f"Candidate metrics lack one baseline row: {baseline_key}")
    baseline_row = baseline.iloc[0]
    result = metrics.copy()
    result["eligible"] = result.apply(
        lambda row: candidate_is_eligible(row, baseline_row), axis=1
    )
    result["complexity"] = result["candidate_key"].ne(baseline_key).astype(int)
    result = result.sort_values(
        [
            "eligible",
            "pairwise_accuracy",
            "ranking_score",
            "brier_1x2",
            "log_loss_1x2",
            "complexity",
            "candidate_key",
        ],
        ascending=[False, False, False, True, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    eligible = result.loc[result["eligible"]]
    if eligible.empty:
        raise ValueError("No candidate passes ranking-first guardrails")
    return eligible.iloc[0]


def paired_predictions(
    candidate: SequenceEvaluation,
    baseline: SequenceEvaluation,
    fold: int,
    events: pd.DataFrame,
) -> pd.DataFrame:
    if candidate.predictions.empty or baseline.predictions.empty:
        raise ValueError("Paired comparison requires prediction rows")
    candidate_columns = [
        "match_id",
        "season",
        "competition",
        "stage",
        "home_team_id",
        "away_team_id",
        "actual_home_score",
        "expected_home_score",
        "home_probability",
        "draw_probability",
        "away_probability",
        "brier_1x2",
        "log_loss_1x2",
        "goal_multiplier",
        "competition_k_multiplier",
        "power_delta",
        "reserve_added_after_match",
    ]
    candidate_frame = candidate.predictions[candidate_columns].rename(
        columns={
            column: f"candidate_{column}"
            for column in candidate_columns[7:]
        }
    )
    baseline_frame = baseline.predictions[
        [
            "match_id",
            "expected_home_score",
            "home_probability",
            "draw_probability",
            "away_probability",
            "brier_1x2",
            "log_loss_1x2",
        ]
    ].rename(
        columns={
            column: f"baseline_{column}"
            for column in (
                "expected_home_score",
                "home_probability",
                "draw_probability",
                "away_probability",
                "brier_1x2",
                "log_loss_1x2",
            )
        }
    )
    joined = candidate_frame.merge(
        baseline_frame,
        on="match_id",
        validate="one_to_one",
    )
    metadata = events[
        ["match_id", "tie_id", "kickoff_utc", "round"]
    ]
    joined = joined.merge(metadata, on="match_id", validate="one_to_one")
    joined.insert(0, "fold", fold)
    return joined


def comparison_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [("ALL", predictions), *predictions.groupby("competition", sort=True)]
    for competition, frame in groups:
        rows.append(
            {
                "competition": competition,
                "matches": len(frame),
                "candidate_brier_1x2": frame["candidate_brier_1x2"].mean(),
                "baseline_brier_1x2": frame["baseline_brier_1x2"].mean(),
                "brier_difference": (
                    frame["candidate_brier_1x2"]
                    - frame["baseline_brier_1x2"]
                ).mean(),
                "candidate_log_loss_1x2": frame[
                    "candidate_log_loss_1x2"
                ].mean(),
                "baseline_log_loss_1x2": frame[
                    "baseline_log_loss_1x2"
                ].mean(),
                "log_loss_difference": (
                    frame["candidate_log_loss_1x2"]
                    - frame["baseline_log_loss_1x2"]
                ).mean(),
            }
        )
    return pd.DataFrame(rows)


def comparison_uncertainty(
    predictions: pd.DataFrame,
    *,
    bootstrap_samples: int,
) -> pd.DataFrame:
    rows = []
    groups = [("ALL", predictions), *predictions.groupby("competition", sort=True)]
    for competition, frame in groups:
        for loss in ("brier_1x2", "log_loss_1x2"):
            values = frame.copy()
            values["loss_difference"] = (
                values[f"candidate_{loss}"] - values[f"baseline_{loss}"]
            )
            uncertainty = dependency_robust_loss_difference_ci(
                values,
                bootstrap_samples=bootstrap_samples,
            )
            uncertainty.insert(0, "competition", competition)
            uncertainty.insert(1, "loss", loss)
            rows.append(uncertainty)
    return pd.concat(rows, ignore_index=True)


def promotion_guardrails(
    selections: pd.DataFrame,
    fold_results: pd.DataFrame,
    uncertainty: pd.DataFrame,
    *,
    full_candidate_active: bool,
) -> dict[str, object]:
    candidate = fold_results.loc[fold_results["model"].eq("candidate")].set_index(
        "fold"
    )
    baseline = fold_results.loc[fold_results["model"].eq("baseline")].set_index(
        "fold"
    )
    if set(candidate.index) != set(baseline.index):
        raise ValueError("Candidate and baseline fold results do not align")
    brier_wins = int(
        (candidate["brier_1x2"] < baseline["brier_1x2"] - RANK_TOLERANCE).sum()
    )
    ranking_index = candidate.index[
        candidate[["ranking_score", "pairwise_accuracy"]].notna().all(axis=1)
        & baseline[["ranking_score", "pairwise_accuracy"]].notna().all(axis=1)
    ]
    ranked_candidate = candidate.loc[ranking_index]
    ranked_baseline = baseline.loc[ranking_index]
    rank_no_regression = (
        ranked_candidate["ranking_score"]
        >= ranked_baseline["ranking_score"] - RANK_TOLERANCE
    ) & (
        ranked_candidate["pairwise_accuracy"]
        >= ranked_baseline["pairwise_accuracy"] - RANK_TOLERANCE
    )
    rank_improved = (
        ranked_candidate["ranking_score"]
        > ranked_baseline["ranking_score"] + RANK_TOLERANCE
    ) & (
        ranked_candidate["pairwise_accuracy"]
        > ranked_baseline["pairwise_accuracy"] + RANK_TOLERANCE
    )
    envelope = uncertainty.loc[
        uncertainty["method"].eq("conservative_envelope")
        & uncertainty["loss"].eq("brier_1x2")
    ]
    overall = envelope.loc[envelope["competition"].eq("ALL")].iloc[0]
    segments = envelope.loc[envelope["competition"].ne("ALL")]
    nonbaseline_share = float(selections["selected_is_baseline"].eq(False).mean())
    return {
        "full_candidate_active": full_candidate_active,
        "nonbaseline_fold_share": nonbaseline_share,
        "brier_fold_wins": brier_wins,
        "ranking_evaluable_folds": len(ranking_index),
        "ranking_no_regression_folds": int(rank_no_regression.sum()),
        "ranking_both_improved_folds": int(rank_improved.sum()),
        "overall_reliable_brier_improvement": bool(
            overall["reliable_improvement"]
        ),
        "overall_reliable_brier_harm": bool(overall["reliable_harm"]),
        "no_competition_reliable_harm": not bool(segments["reliable_harm"].any()),
        "all_updates_zero_sum": bool(
            candidate["max_pair_sum_error"].le(1e-9).all()
        ),
        "ranking_floor_safe": bool(
            candidate["start_end_rank_correlation"]
            .ge(RANK_CORRELATION_FLOOR)
            .all()
        ),
        "movement_safe": bool(
            candidate["max_abs_rating_change"]
            .le(MAX_RATING_MOVE_GUARDRAIL)
            .all()
        ),
    }


def strict_layer_pass(guardrails: dict[str, object]) -> bool:
    ranking_folds = int(guardrails["ranking_evaluable_folds"])
    return bool(
        guardrails["full_candidate_active"]
        and guardrails["nonbaseline_fold_share"] >= 0.5
        and guardrails["brier_fold_wins"] >= 5
        and ranking_folds >= 5
        and guardrails["ranking_no_regression_folds"] == ranking_folds
        and guardrails["ranking_both_improved_folds"] >= 4
        and guardrails["overall_reliable_brier_improvement"]
        and guardrails["no_competition_reliable_harm"]
        and guardrails["all_updates_zero_sum"]
        and guardrails["ranking_floor_safe"]
        and guardrails["movement_safe"]
    )


def margin_metric_table(
    datasets: tuple[ReserveSeasonData, ...],
    core: DynamicCoreConfig,
    draw_mapping: dict[str, DrawModelConfig],
    target: pd.DataFrame,
    candidates: Iterable[GoalMarginCandidate],
) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        evaluation = evaluate_sequence(
            datasets,
            core,
            candidate,
            baseline_competition_k(),
            baseline_reserve(),
            draw_mapping,
            target,
        )
        rows.append(
            evaluated_row(
                evaluation,
                candidate_key=candidate.key,
                family=candidate.family,
                weight=candidate.weight,
                cap=candidate.cap,
                favorite_damping=candidate.favorite_damping,
            )
        )
    return pd.DataFrame(rows)


def margin_from_row(row: pd.Series) -> GoalMarginCandidate:
    return GoalMarginCandidate(
        str(row["family"]),
        float(row["weight"]),
        float(row["cap"]),
        float(row["favorite_damping"]),
    )


def run_margin_calibration(
    datasets: tuple[ReserveSeasonData, ...],
    events: pd.DataFrame,
    target: pd.DataFrame,
    folds: tuple[tuple[tuple[str, ...], str], ...],
    core_selections: pd.DataFrame,
    draw_selections: pd.DataFrame,
    production: dict[str, object],
    *,
    bootstrap_samples: int,
) -> dict[str, object]:
    candidates = goal_margin_candidates()
    baseline = baseline_goal_margin()
    selection_rows = []
    fold_rows = []
    train_frames = []
    prediction_frames = []
    for fold, (train_seasons, test_season) in enumerate(folds, start=1):
        core = core_for_fold(core_selections, fold)
        draw = draw_map_for_fold(draw_selections, fold)
        train = tuple(data for data in datasets if data.season in train_seasons)
        test = tuple(data for data in datasets if data.season == test_season)
        metrics = margin_metric_table(train, core, draw, target, candidates)
        selected_row = select_ranking_first(metrics, baseline.key)
        selected = margin_from_row(selected_row)
        metrics.insert(0, "fold", fold)
        metrics.insert(1, "test_season", test_season)
        train_frames.append(metrics)
        selection_rows.append(
            {
                "fold": fold,
                "train_seasons": "|".join(train_seasons),
                "test_season": test_season,
                "selected_candidate": selected.key,
                "selected_is_baseline": not selected.active,
                "selected_family": selected.family,
                "selected_weight": selected.weight,
                "selected_cap": selected.cap,
                "selected_favorite_damping": selected.favorite_damping,
                "train_ranking_score": selected_row["ranking_score"],
                "train_pairwise_accuracy": selected_row["pairwise_accuracy"],
                "train_brier_1x2": selected_row["brier_1x2"],
            }
        )
        evaluations = {}
        for model, config in (("candidate", selected), ("baseline", baseline)):
            evaluated = evaluate_sequence(
                test,
                core,
                config,
                baseline_competition_k(),
                baseline_reserve(),
                draw,
                target,
                ranking_target_seasons=set(target["season"]),
                return_predictions=True,
            )
            evaluations[model] = evaluated
            fold_rows.append(
                evaluated_row(
                    evaluated,
                    fold=fold,
                    test_season=test_season,
                    model=model,
                    candidate_key=config.key,
                )
            )
        prediction_frames.append(
            paired_predictions(
                evaluations["candidate"], evaluations["baseline"], fold, events
            )
        )
        print(f"  goal-margin outer fold {fold}/{len(folds)}", flush=True)

    selections = pd.DataFrame(selection_rows)
    fold_results = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    uncertainty = comparison_uncertainty(
        predictions, bootstrap_samples=bootstrap_samples
    )
    summary = comparison_summary(predictions)
    full_metrics = margin_metric_table(
        datasets,
        production_core(production),
        production_draw_map(production),
        target,
        candidates,
    )
    full_row = select_ranking_first(full_metrics, baseline.key)
    full_candidate = margin_from_row(full_row)
    guardrails = promotion_guardrails(
        selections,
        fold_results,
        uncertainty,
        full_candidate_active=full_candidate.active,
    )
    decision = (
        "PROMOTE_GOAL_MARGIN" if strict_layer_pass(guardrails) else "DISABLE_GOAL_MARGIN"
    )
    return {
        "decision": decision,
        "full_candidate": full_candidate,
        "active_candidate": (
            full_candidate if decision == "PROMOTE_GOAL_MARGIN" else baseline
        ),
        "guardrails": guardrails,
        "selections": selections,
        "fold_results": fold_results,
        "train_metrics": pd.concat(train_frames, ignore_index=True),
        "full_metrics": full_metrics,
        "predictions": predictions,
        "summary": summary,
        "uncertainty": uncertainty,
    }


def competition_k_metric_table(
    datasets: tuple[ReserveSeasonData, ...],
    core: DynamicCoreConfig,
    draw_mapping: dict[str, DrawModelConfig],
    target: pd.DataFrame,
    margin: GoalMarginCandidate,
    candidates: Iterable[CompetitionKCandidate],
) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        evaluation = evaluate_sequence(
            datasets,
            core,
            margin,
            candidate,
            baseline_reserve(),
            draw_mapping,
            target,
        )
        rows.append(
            evaluated_row(
                evaluation,
                candidate_key=candidate.key,
                profile=candidate.profile,
                ucl_multiplier=candidate.ucl_multiplier,
                uel_multiplier=candidate.uel_multiplier,
                uecl_multiplier=candidate.uecl_multiplier,
            )
        )
    return pd.DataFrame(rows)


def competition_k_from_row(row: pd.Series) -> CompetitionKCandidate:
    return CompetitionKCandidate(
        str(row["profile"]),
        float(row["ucl_multiplier"]),
        float(row["uel_multiplier"]),
        float(row["uecl_multiplier"]),
    )


def run_competition_k_calibration(
    datasets: tuple[ReserveSeasonData, ...],
    events: pd.DataFrame,
    target: pd.DataFrame,
    folds: tuple[tuple[tuple[str, ...], str], ...],
    core_selections: pd.DataFrame,
    draw_selections: pd.DataFrame,
    production: dict[str, object],
    margin: GoalMarginCandidate,
    *,
    bootstrap_samples: int,
) -> dict[str, object]:
    candidates = competition_k_candidates()
    baseline = baseline_competition_k()
    selection_rows = []
    fold_rows = []
    train_frames = []
    prediction_frames = []
    for fold, (train_seasons, test_season) in enumerate(folds, start=1):
        core = core_for_fold(core_selections, fold)
        draw = draw_map_for_fold(draw_selections, fold)
        train = tuple(data for data in datasets if data.season in train_seasons)
        test = tuple(data for data in datasets if data.season == test_season)
        metrics = competition_k_metric_table(
            train,
            core,
            draw,
            target,
            margin,
            candidates,
        )
        selected_row = select_ranking_first(metrics, baseline.key)
        selected = competition_k_from_row(selected_row)
        metrics.insert(0, "fold", fold)
        metrics.insert(1, "test_season", test_season)
        train_frames.append(metrics)
        selection_rows.append(
            {
                "fold": fold,
                "train_seasons": "|".join(train_seasons),
                "test_season": test_season,
                "selected_candidate": selected.key,
                "selected_is_baseline": selected.is_equal_baseline,
                "selected_profile": selected.profile,
                "selected_ucl_multiplier": selected.ucl_multiplier,
                "selected_uel_multiplier": selected.uel_multiplier,
                "selected_uecl_multiplier": selected.uecl_multiplier,
                "train_ranking_score": selected_row["ranking_score"],
                "train_pairwise_accuracy": selected_row["pairwise_accuracy"],
                "train_brier_1x2": selected_row["brier_1x2"],
            }
        )
        evaluations = {}
        for model, config in (("candidate", selected), ("baseline", baseline)):
            evaluated = evaluate_sequence(
                test,
                core,
                margin,
                config,
                baseline_reserve(),
                draw,
                target,
                ranking_target_seasons=set(target["season"]),
                return_predictions=True,
            )
            evaluations[model] = evaluated
            fold_rows.append(
                evaluated_row(
                    evaluated,
                    fold=fold,
                    test_season=test_season,
                    model=model,
                    candidate_key=config.key,
                )
            )
        prediction_frames.append(
            paired_predictions(
                evaluations["candidate"], evaluations["baseline"], fold, events
            )
        )
        print(f"  competition-K outer fold {fold}/{len(folds)}", flush=True)

    selections = pd.DataFrame(selection_rows)
    fold_results = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    uncertainty = comparison_uncertainty(
        predictions, bootstrap_samples=bootstrap_samples
    )
    summary = comparison_summary(predictions)
    full_metrics = competition_k_metric_table(
        datasets,
        production_core(production),
        production_draw_map(production),
        target,
        margin,
        candidates,
    )
    full_row = select_ranking_first(full_metrics, baseline.key)
    full_candidate = competition_k_from_row(full_row)
    guardrails = promotion_guardrails(
        selections,
        fold_results,
        uncertainty,
        full_candidate_active=not full_candidate.is_equal_baseline,
    )
    guardrails["hierarchy_fold_share"] = float(
        selections["selected_profile"].eq("HIERARCHY").mean()
    )
    passed = strict_layer_pass(guardrails)
    if not passed:
        decision = "DISABLE_COMPETITION_K"
    elif full_candidate.profile == "HIERARCHY":
        decision = "PROMOTE_COMPETITION_K_HIERARCHY"
    else:
        decision = "PROMOTE_GLOBAL_K_RECALIBRATION"
    return {
        "decision": decision,
        "full_candidate": full_candidate,
        "active_candidate": full_candidate if passed else baseline,
        "guardrails": guardrails,
        "selections": selections,
        "fold_results": fold_results,
        "train_metrics": pd.concat(train_frames, ignore_index=True),
        "full_metrics": full_metrics,
        "predictions": predictions,
        "summary": summary,
        "uncertainty": uncertainty,
    }


def reserve_key(config: AchievementReserveConfig) -> str:
    if config.reserve_base == 0.0:
        return "reserve_none"
    return (
        f"reserve_b{config.reserve_base:g}_uel{config.uel_multiplier:g}"
        f"_uecl{config.uecl_multiplier:g}_{config.stage_profile.lower()}"
        f"_d{config.reserve_decay:g}"
    )


def reserve_metric_table(
    datasets: tuple[ReserveSeasonData, ...],
    core: DynamicCoreConfig,
    draw_mapping: dict[str, DrawModelConfig],
    target: pd.DataFrame,
    margin: GoalMarginCandidate,
    competition_k: CompetitionKCandidate,
    candidates: Iterable[AchievementReserveConfig],
    *,
    progress_label: str | None = None,
) -> pd.DataFrame:
    candidate_list = tuple(candidates)
    rows = []
    for index, candidate in enumerate(candidate_list, start=1):
        evaluation = evaluate_sequence(
            datasets,
            core,
            margin,
            competition_k,
            candidate,
            draw_mapping,
            target,
        )
        rows.append(
            evaluated_row(
                evaluation,
                candidate_key=reserve_key(candidate),
                reserve_base=candidate.reserve_base,
                uel_multiplier=candidate.uel_multiplier,
                uecl_multiplier=candidate.uecl_multiplier,
                stage_profile=candidate.stage_profile,
                reserve_decay=candidate.reserve_decay,
                reserve_cap=candidate.reserve_cap,
            )
        )
        if progress_label and index % 200 == 0:
            print(
                f"    reserve {progress_label}: {index}/{len(candidate_list)}",
                flush=True,
            )
    return pd.DataFrame(rows)


def reserve_from_row(row: pd.Series) -> AchievementReserveConfig:
    return AchievementReserveConfig(
        float(row["reserve_base"]),
        float(row["uel_multiplier"]),
        float(row["uecl_multiplier"]),
        str(row["stage_profile"]),
        float(row["reserve_decay"]),
        float(row["reserve_cap"]),
    )


def run_reserve_calibration(
    datasets: tuple[ReserveSeasonData, ...],
    events: pd.DataFrame,
    target: pd.DataFrame,
    folds: tuple[tuple[tuple[str, ...], str], ...],
    core_selections: pd.DataFrame,
    draw_selections: pd.DataFrame,
    production: dict[str, object],
    margin: GoalMarginCandidate,
    competition_k: CompetitionKCandidate,
    *,
    bootstrap_samples: int,
) -> dict[str, object]:
    candidates = reserve_candidates()
    baseline = baseline_reserve()
    baseline_key = reserve_key(baseline)
    selection_rows = []
    fold_rows = []
    train_frames = []
    prediction_frames = []
    for fold, (train_seasons, test_season) in enumerate(folds, start=1):
        core = core_for_fold(core_selections, fold)
        draw = draw_map_for_fold(draw_selections, fold)
        train = tuple(data for data in datasets if data.season in train_seasons)
        sequence = tuple(
            data for data in datasets if data.season in (*train_seasons, test_season)
        )
        metrics = reserve_metric_table(
            train,
            core,
            draw,
            target,
            margin,
            competition_k,
            candidates,
            progress_label=f"fold {fold}",
        )
        selected_row = select_ranking_first(metrics, baseline_key)
        selected = reserve_from_row(selected_row)
        metrics.insert(0, "fold", fold)
        metrics.insert(1, "test_season", test_season)
        train_frames.append(metrics)
        selection_rows.append(
            {
                "fold": fold,
                "train_seasons": "|".join(train_seasons),
                "test_season": test_season,
                "selected_candidate": reserve_key(selected),
                "selected_is_baseline": selected.reserve_base == 0.0,
                "selected_reserve_base": selected.reserve_base,
                "selected_uel_multiplier": selected.uel_multiplier,
                "selected_uecl_multiplier": selected.uecl_multiplier,
                "selected_stage_profile": selected.stage_profile,
                "selected_reserve_decay": selected.reserve_decay,
                "train_ranking_score": selected_row["ranking_score"],
                "train_pairwise_accuracy": selected_row["pairwise_accuracy"],
                "train_brier_1x2": selected_row["brier_1x2"],
            }
        )
        evaluations = {}
        for model, config in (("candidate", selected), ("baseline", baseline)):
            evaluated = evaluate_sequence(
                sequence,
                core,
                margin,
                competition_k,
                config,
                draw,
                target,
                evaluation_seasons={test_season},
                ranking_target_seasons=set(target["season"]),
                return_predictions=True,
            )
            evaluations[model] = evaluated
            fold_rows.append(
                evaluated_row(
                    evaluated,
                    fold=fold,
                    test_season=test_season,
                    model=model,
                    candidate_key=reserve_key(config),
                )
            )
        prediction_frames.append(
            paired_predictions(
                evaluations["candidate"], evaluations["baseline"], fold, events
            )
        )
        print(f"  achievement-reserve outer fold {fold}/{len(folds)}", flush=True)

    selections = pd.DataFrame(selection_rows)
    fold_results = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    uncertainty = comparison_uncertainty(
        predictions, bootstrap_samples=bootstrap_samples
    )
    summary = comparison_summary(predictions)
    full_metrics = reserve_metric_table(
        datasets,
        production_core(production),
        production_draw_map(production),
        target,
        margin,
        competition_k,
        candidates,
        progress_label="full-data",
    )
    full_row = select_ranking_first(full_metrics, baseline_key)
    full_candidate = reserve_from_row(full_row)
    guardrails = promotion_guardrails(
        selections,
        fold_results,
        uncertainty,
        full_candidate_active=full_candidate.reserve_base > 0.0,
    )
    candidate_rows = fold_results.loc[fold_results["model"].eq("candidate")]
    guardrails["reserve_cap_safe"] = bool(
        candidate_rows["max_reserve"].le(full_candidate.reserve_cap + 1e-9).all()
    )
    guardrails["strict_competition_hierarchy"] = bool(
        full_candidate.uel_multiplier > full_candidate.uecl_multiplier
    )
    passed = bool(
        strict_layer_pass(guardrails)
        and guardrails["reserve_cap_safe"]
        and guardrails["strict_competition_hierarchy"]
    )
    decision = (
        "PROMOTE_ACHIEVEMENT_RESERVE"
        if passed
        else "DISABLE_ACHIEVEMENT_RESERVE"
    )
    return {
        "decision": decision,
        "full_candidate": full_candidate,
        "active_candidate": full_candidate if passed else baseline,
        "guardrails": guardrails,
        "selections": selections,
        "fold_results": fold_results,
        "train_metrics": pd.concat(train_frames, ignore_index=True),
        "full_metrics": full_metrics,
        "predictions": predictions,
        "summary": summary,
        "uncertainty": uncertainty,
    }


def run_fixed_oof(
    datasets: tuple[ReserveSeasonData, ...],
    events: pd.DataFrame,
    target: pd.DataFrame,
    folds: tuple[tuple[tuple[str, ...], str], ...],
    core_selections: pd.DataFrame,
    draw_selections: pd.DataFrame,
    margin: GoalMarginCandidate,
    competition_k: CompetitionKCandidate,
    reserve: AchievementReserveConfig,
) -> dict[str, pd.DataFrame]:
    prediction_frames = []
    ranking_frames = []
    metric_rows = []
    for fold, (train_seasons, test_season) in enumerate(folds, start=1):
        sequence = tuple(
            data for data in datasets if data.season in (*train_seasons, test_season)
        )
        evaluation = evaluate_sequence(
            sequence,
            core_for_fold(core_selections, fold),
            margin,
            competition_k,
            reserve,
            draw_map_for_fold(draw_selections, fold),
            target,
            evaluation_seasons={test_season},
            ranking_target_seasons=set(target["season"]),
            return_predictions=True,
        )
        predictions = evaluation.predictions.copy()
        predictions.insert(0, "fold", fold)
        prediction_frames.append(predictions)
        ranking = evaluation.ranking.copy()
        ranking.insert(0, "fold", fold)
        ranking.insert(1, "test_season", test_season)
        ranking_frames.append(ranking)
        metric_rows.append(
            evaluated_row(
                evaluation,
                fold=fold,
                test_season=test_season,
            )
        )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions = predictions.merge(
        events[["match_id", "tie_id", "kickoff_utc", "round"]],
        on="match_id",
        validate="one_to_one",
    )
    return {
        "predictions": predictions,
        "ranking": pd.concat(ranking_frames, ignore_index=True),
        "fold_metrics": pd.DataFrame(metric_rows),
    }


def score_probabilities(
    expected: np.ndarray,
    actual: np.ndarray,
    draw: DrawModelConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    draw_probability = draw.draw_at_even * (
        4.0 * expected * (1.0 - expected)
    ) ** draw.draw_shape
    home_probability = expected - 0.5 * draw_probability
    away_probability = 1.0 - expected - 0.5 * draw_probability
    outcomes = np.where(actual == 1.0, 0, np.where(actual == 0.5, 1, 2))
    matrix = np.column_stack(
        (home_probability, draw_probability, away_probability)
    )
    target = np.eye(3)[outcomes]
    brier = np.square(matrix - target).sum(axis=1)
    log_loss = -np.log(np.clip(matrix[np.arange(len(matrix)), outcomes], 1e-15, 1.0))
    return home_probability, draw_probability, away_probability, brier, log_loss


def select_clubelo_1x2(train: pd.DataFrame) -> dict[str, float]:
    if train.empty:
        raise ValueError("ClubElo calibration requires non-empty prior seasons")
    actual = train["actual_home_score"].to_numpy(float)
    neutral = train["is_neutral"].to_numpy(bool)
    home = train["clubelo_home_elo"].to_numpy(float)
    away = train["clubelo_away_elo"].to_numpy(float)
    rows = []
    for home_advantage in CLUBELO_HOME_ADVANTAGES:
        advantage = np.where(neutral, 0.0, home_advantage)
        expected = 1.0 / (
            1.0 + 10.0 ** (-((home - away + advantage) / CLUBELO_SCALE))
        )
        for draw in draw_model_candidates():
            _, _, _, brier, log_loss = score_probabilities(expected, actual, draw)
            rows.append(
                {
                    "home_advantage": home_advantage,
                    "draw_at_even": draw.draw_at_even,
                    "draw_shape": draw.draw_shape,
                    "brier_1x2": float(brier.mean()),
                    "log_loss_1x2": float(log_loss.mean()),
                    "distance_from_reference": (
                        abs(home_advantage - 60.0) / 100.0
                        + abs(draw.draw_at_even - 0.28)
                        + 0.02 * abs(draw.draw_shape - 1.0)
                    ),
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "brier_1x2",
                "log_loss_1x2",
                "distance_from_reference",
                "home_advantage",
                "draw_at_even",
                "draw_shape",
            ],
            kind="stable",
        )
        .iloc[0]
        .to_dict()
    )


def add_clubelo_1x2(
    data: pd.DataFrame,
    selection: dict[str, float],
) -> pd.DataFrame:
    result = data.copy()
    expected = np.array(
        [
            clubelo_expected_home_score(
                float(home),
                float(away),
                float(selection["home_advantage"]),
                neutral=bool(neutral),
            )
            for home, away, neutral in zip(
                result["clubelo_home_elo"],
                result["clubelo_away_elo"],
                result["is_neutral"],
            )
        ]
    )
    probabilities = score_probabilities(
        expected,
        result["actual_home_score"].to_numpy(float),
        DrawModelConfig(
            float(selection["draw_at_even"]),
            float(selection["draw_shape"]),
        ),
    )
    result["clubelo_expected_home_score"] = expected
    result["clubelo_home_probability"] = probabilities[0]
    result["clubelo_draw_probability"] = probabilities[1]
    result["clubelo_away_probability"] = probabilities[2]
    result["clubelo_brier_1x2"] = probabilities[3]
    result["clubelo_log_loss_1x2"] = probabilities[4]
    result["clubelo_expected_score_mse"] = np.square(
        expected - result["actual_home_score"].to_numpy(float)
    )
    return result


def run_final_external_benchmark(
    datasets: tuple[ReserveSeasonData, ...],
    events: pd.DataFrame,
    folds: tuple[tuple[tuple[str, ...], str], ...],
    core_selections: pd.DataFrame,
    draw_selections: pd.DataFrame,
    production: dict[str, object],
    margin: GoalMarginCandidate,
    competition_k: CompetitionKCandidate,
    reserve: AchievementReserveConfig,
    benchmark: pd.DataFrame,
    *,
    bootstrap_samples: int,
) -> dict[str, object]:
    target = schedule_adjusted_team_performance(events)
    final_oof = run_fixed_oof(
        datasets,
        events,
        target,
        folds,
        core_selections,
        draw_selections,
        margin,
        competition_k,
        reserve,
    )["predictions"]
    static_core_selections = core_selections.copy()
    static_core_selections["selected_scale"] = static_core_selections["static_scale"]
    static_core_selections["selected_home_advantage"] = static_core_selections[
        "static_home_advantage"
    ]
    static_core_selections["selected_k"] = 0.0
    static_oof = run_fixed_oof(
        datasets,
        events,
        target,
        folds,
        static_core_selections,
        draw_selections,
        baseline_goal_margin(),
        baseline_competition_k(),
        baseline_reserve(),
    )["predictions"]
    eligible = benchmark.loc[
        benchmark["external_elo_pair_available"].astype(bool)
    ].copy()
    paired = final_oof.merge(
        eligible[
            [
                "match_id",
                "clubelo_home_elo",
                "clubelo_away_elo",
                "is_neutral",
            ]
        ],
        on="match_id",
        how="inner",
        validate="one_to_one",
    )
    paired = paired.merge(
        static_oof[
            [
                "match_id",
                "expected_home_score",
                "brier_1x2",
                "log_loss_1x2",
            ]
        ].rename(
            columns={
                "expected_home_score": "static_expected_home_score",
                "brier_1x2": "static_brier_1x2",
                "log_loss_1x2": "static_log_loss_1x2",
            }
        ),
        on="match_id",
        validate="one_to_one",
    )
    selection_rows = []
    prediction_frames = []
    for fold, (train_seasons, test_season) in enumerate(folds, start=1):
        train = eligible.loc[eligible["season"].isin(train_seasons)]
        test = paired.loc[paired["fold"].eq(fold)]
        if train.empty or test.empty:
            continue
        selection = select_clubelo_1x2(train)
        predicted = add_clubelo_1x2(test, selection)
        predicted["ao_expected_score_mse"] = np.square(
            predicted["expected_home_score"] - predicted["actual_home_score"]
        )
        prediction_frames.append(predicted)
        selection_rows.append(
            {
                "fold": fold,
                "train_seasons": "|".join(train_seasons),
                "test_season": test_season,
                "train_external_pairs": len(train),
                "test_external_pairs": len(test),
                "selected_clubelo_home_advantage": selection["home_advantage"],
                "selected_clubelo_draw_at_even": selection["draw_at_even"],
                "selected_clubelo_draw_shape": selection["draw_shape"],
                "train_clubelo_brier_1x2": selection["brier_1x2"],
                "train_clubelo_log_loss_1x2": selection["log_loss_1x2"],
            }
        )
    if not prediction_frames:
        raise ValueError("No exact-date external benchmark fold could be evaluated")
    predictions = pd.concat(prediction_frames, ignore_index=True)
    rows = []
    for competition, frame in [
        ("ALL", predictions),
        *predictions.groupby("competition", sort=True),
    ]:
        rows.append(
            {
                "competition": competition,
                "matches": len(frame),
                "ao_brier_1x2": frame["brier_1x2"].mean(),
                "clubelo_brier_1x2": frame["clubelo_brier_1x2"].mean(),
                "ao_minus_clubelo_brier_1x2": (
                    frame["brier_1x2"] - frame["clubelo_brier_1x2"]
                ).mean(),
                "ao_log_loss_1x2": frame["log_loss_1x2"].mean(),
                "clubelo_log_loss_1x2": frame["clubelo_log_loss_1x2"].mean(),
                "ao_minus_clubelo_log_loss_1x2": (
                    frame["log_loss_1x2"] - frame["clubelo_log_loss_1x2"]
                ).mean(),
                "static_brier_1x2": frame["static_brier_1x2"].mean(),
                "dynamic_minus_static_brier_1x2": (
                    frame["brier_1x2"] - frame["static_brier_1x2"]
                ).mean(),
                "ao_expected_score_mse": frame["ao_expected_score_mse"].mean(),
                "clubelo_expected_score_mse": frame[
                    "clubelo_expected_score_mse"
                ].mean(),
            }
        )
    summary = pd.DataFrame(rows)
    uncertainty_rows = []
    for competition, frame in [
        ("ALL", predictions),
        *predictions.groupby("competition", sort=True),
    ]:
        for loss, ao_column, external_column in (
            ("brier_1x2", "brier_1x2", "clubelo_brier_1x2"),
            ("log_loss_1x2", "log_loss_1x2", "clubelo_log_loss_1x2"),
            (
                "expected_score_mse",
                "ao_expected_score_mse",
                "clubelo_expected_score_mse",
            ),
        ):
            values = frame.copy()
            values["loss_difference"] = values[ao_column] - values[external_column]
            result = dependency_robust_loss_difference_ci(
                values,
                bootstrap_samples=bootstrap_samples,
            )
            result.insert(0, "competition", competition)
            result.insert(1, "loss", loss)
            uncertainty_rows.append(result)
    uncertainty = pd.concat(uncertainty_rows, ignore_index=True)
    ucl_brier = envelope_row(uncertainty, "UCL", "brier_1x2")
    ucl_expected = envelope_row(uncertainty, "UCL", "expected_score_mse")
    if bool(ucl_brier["reliable_harm"]):
        status = "UCL_CLUBELO_RELIABLY_BETTER_1X2"
    elif bool(ucl_expected["reliable_harm"]):
        status = "UCL_CLUBELO_RELIABLY_BETTER_EXPECTED_SCORE"
    elif bool(ucl_brier["reliable_improvement"]):
        status = "UCL_AO_RELIABLY_BETTER"
    else:
        ucl_summary = summary.loc[summary["competition"].eq("UCL")].iloc[0]
        status = (
            "UCL_CLUBELO_POINT_ESTIMATE_BETTER_INCONCLUSIVE"
            if ucl_summary["ao_minus_clubelo_brier_1x2"] > 0.0
            else "UCL_AO_POINT_ESTIMATE_BETTER_INCONCLUSIVE"
        )
    diagnostics = external_diagnostic_groups(predictions)
    return {
        "status": status,
        "selections": pd.DataFrame(selection_rows),
        "predictions": predictions,
        "summary": summary,
        "uncertainty": uncertainty,
        "diagnostics": diagnostics,
        "eligible_pairs": len(eligible),
        "unseen_pairs": len(predictions),
        "production_contract": production,
    }


def external_diagnostic_groups(predictions: pd.DataFrame) -> pd.DataFrame:
    ucl = predictions.loc[predictions["competition"].eq("UCL")].copy()
    ucl["ao_expected_bin"] = pd.cut(
        ucl["expected_home_score"],
        bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        include_lowest=True,
    ).astype(str)
    rows = []
    for level, grouped in (
        ("season", ucl.groupby("season", sort=True)),
        ("stage", ucl.groupby("stage", sort=True)),
        ("ao_expected_bin", ucl.groupby("ao_expected_bin", sort=True)),
    ):
        for key, frame in grouped:
            rows.append(
                {
                    "level": level,
                    "segment": str(key),
                    "matches": len(frame),
                    "ao_brier_1x2": frame["brier_1x2"].mean(),
                    "static_brier_1x2": frame["static_brier_1x2"].mean(),
                    "clubelo_brier_1x2": frame["clubelo_brier_1x2"].mean(),
                    "ao_minus_static_brier_1x2": (
                        frame["brier_1x2"] - frame["static_brier_1x2"]
                    ).mean(),
                    "ao_minus_clubelo_brier_1x2": (
                        frame["brier_1x2"] - frame["clubelo_brier_1x2"]
                    ).mean(),
                    "ao_expected_score_mse": frame["ao_expected_score_mse"].mean(),
                    "clubelo_expected_score_mse": frame[
                        "clubelo_expected_score_mse"
                    ].mean(),
                }
            )
    return pd.DataFrame(rows)


def competition_residual_diagnostics(
    predictions: pd.DataFrame,
    *,
    bootstrap_samples: int,
) -> dict[str, pd.DataFrame]:
    values = predictions.copy()
    values["score_residual"] = (
        values["actual_home_score"] - values["expected_home_score"]
    )
    rows = []
    groups = [
        ("competition", values.groupby("competition", sort=True)),
        ("competition_stage", values.groupby(["competition", "stage"], sort=True)),
    ]
    for level, grouped in groups:
        for key, frame in grouped:
            key_tuple = key if isinstance(key, tuple) else (key,)
            rows.append(
                {
                    "level": level,
                    "competition": key_tuple[0],
                    "stage": key_tuple[1] if len(key_tuple) > 1 else "ALL",
                    "matches": len(frame),
                    "mean_expected_home_score": frame[
                        "expected_home_score"
                    ].mean(),
                    "mean_actual_home_score": frame["actual_home_score"].mean(),
                    "mean_score_residual": frame["score_residual"].mean(),
                    "mean_absolute_score_residual": frame[
                        "score_residual"
                    ].abs().mean(),
                    "brier_1x2": frame["brier_1x2"].mean(),
                    "log_loss_1x2": frame["log_loss_1x2"].mean(),
                }
            )
    uncertainty_rows = []
    for competition, frame in values.groupby("competition", sort=True):
        sample = frame.copy()
        sample["loss_difference"] = sample["score_residual"]
        result = dependency_robust_loss_difference_ci(
            sample,
            bootstrap_samples=bootstrap_samples,
        )
        result.insert(0, "competition", competition)
        result = result.rename(
            columns={
                "mean_difference": "mean_score_residual",
                "reliable_improvement": "reliably_negative_residual",
                "reliable_harm": "reliably_positive_residual",
            }
        )
        uncertainty_rows.append(result)
    return {
        "summary": pd.DataFrame(rows),
        "uncertainty": pd.concat(uncertainty_rows, ignore_index=True),
    }


def write_layer_outputs(root: Path, result: dict[str, object]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "fold_selections.csv": "selections",
        "fold_results.csv": "fold_results",
        "train_candidate_metrics.csv": "train_metrics",
        "full_candidate_metrics.csv": "full_metrics",
        "unseen_predictions.csv": "predictions",
        "competition_summary.csv": "summary",
        "dependency_uncertainty.csv": "uncertainty",
    }
    for filename, key in outputs.items():
        frame = result[key]
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"Layer output {key} is not a DataFrame")
        frame.to_csv(root / filename, index=False)
    payload = {
        "decision": result["decision"],
        "full_candidate": asdict(result["full_candidate"]),
        "active_candidate": asdict(result["active_candidate"]),
        "guardrails": result["guardrails"],
    }
    (root / "decision.json").write_text(
        json.dumps(payload, indent=2, default=json_default), encoding="utf-8"
    )


def write_external_outputs(root: Path, result: dict[str, object]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for filename, key in (
        ("fold_selections.csv", "selections"),
        ("paired_predictions.csv", "predictions"),
        ("competition_summary.csv", "summary"),
        ("dependency_uncertainty.csv", "uncertainty"),
        ("ucl_diagnostics.csv", "diagnostics"),
    ):
        frame = result[key]
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"External output {key} is not a DataFrame")
        frame.to_csv(root / filename, index=False)


def envelope_row(
    uncertainty: pd.DataFrame,
    competition: str,
    loss: str,
) -> pd.Series:
    row = uncertainty.loc[
        uncertainty["competition"].eq(competition)
        & uncertainty["loss"].eq(loss)
        & uncertainty["method"].eq("conservative_envelope")
    ]
    if len(row) != 1:
        raise ValueError(f"Missing one envelope for {competition}/{loss}")
    return row.iloc[0]


def build_manifest(
    production: dict[str, object],
    margin: dict[str, object],
    competition_k: dict[str, object],
    reserve: dict[str, object],
    external: dict[str, object],
    active_margin: GoalMarginCandidate,
    active_k: CompetitionKCandidate,
    active_reserve: AchievementReserveConfig,
) -> dict[str, object]:
    recommended = json.loads(json.dumps(production))
    recommended["goal_margin"] = {
        "active": active_margin.active,
        "family": active_margin.family,
        "goal_weight": active_margin.weight,
        "goal_cap": active_margin.cap,
        "favorite_damping": active_margin.favorite_damping,
    }
    recommended["competition_k"] = {
        "active": not active_k.is_equal_baseline,
        "profile": active_k.profile,
        "ucl_multiplier": active_k.ucl_multiplier,
        "uel_multiplier": active_k.uel_multiplier,
        "uecl_multiplier": active_k.uecl_multiplier,
        "interpretation": "learning-rate multiplier, not intrinsic competition strength",
    }
    recommended["achievement_reserve"] = {
        "active": active_reserve.reserve_base > 0.0,
        "reserve_base": active_reserve.reserve_base,
        "ucl_multiplier": 1.0,
        "uel_multiplier": active_reserve.uel_multiplier,
        "uecl_multiplier": active_reserve.uecl_multiplier,
        "stage_profile": active_reserve.stage_profile,
        "reserve_decay": active_reserve.reserve_decay,
        "reserve_cap": active_reserve.reserve_cap,
        "trophy_uses_same_base": True,
    }
    recommended["decisions"].update(
        {
            "goal_margin": margin["decision"],
            "competition_k": competition_k["decision"],
            "achievement_reserve": reserve["decision"],
            "external_clubelo": external["status"],
        }
    )
    external_blocker = str(external["status"]).startswith(
        "UCL_CLUBELO_RELIABLY_BETTER"
    )
    optional_promoted = any(
        (
            active_margin.active,
            not active_k.is_equal_baseline,
            active_reserve.reserve_base > 0.0,
        )
    )
    overall_status = (
        "NEEDS_REVISION_BEFORE_HOLDOUT"
        if external_blocker
        else "READY_FOR_HOLDOUT_WITH_CAVEATS"
    )
    return {
        "robustness_contract_version": "ao-final-robustness-v1.0",
        "model_version": AO_MODEL_V2_VERSION,
        "development_window": "2018/19-2025/26",
        "untouched_holdout": "2026/27",
        "baseline_contract": {
            "power_carry": 0.0,
            "standard_1x2": True,
            "nested_outer_folds": 6,
            "ranking_first": True,
            "dependency_views": [
                "tie_or_match",
                "team_season",
                "calendar_month",
            ],
        },
        "decisions": {
            "goal_margin": margin["decision"],
            "competition_k": competition_k["decision"],
            "achievement_reserve": reserve["decision"],
            "external_clubelo": external["status"],
        },
        "optional_layer_promoted": optional_promoted,
        "overall_status": overall_status,
        "recommended_production_model": recommended,
    }


def _layer_report_lines(name: str, result: dict[str, object]) -> list[str]:
    overall = envelope_row(result["uncertainty"], "ALL", "brier_1x2")
    guardrails = result["guardrails"]
    return [
        f"### {name}",
        "",
        f"- Karar: `{result['decision']}`",
        f"- Tam veri adayi: `{asdict(result['full_candidate'])}`",
        f"- Brier farki: `{overall.mean_difference:+.6f}`",
        f"- Conservative envelope: `[{overall.ci_95_lower:+.6f}, {overall.ci_95_upper:+.6f}]`",
        f"- Brier fold galibiyeti: `{guardrails['brier_fold_wins']}/6`",
        f"- Forward siralama gerilemesiz fold: `{guardrails['ranking_no_regression_folds']}/{guardrails['ranking_evaluable_folds']}`",
        f"- Iki forward siralama metrigi de iyilesen fold: `{guardrails['ranking_both_improved_folds']}/{guardrails['ranking_evaluable_folds']}`",
        "",
    ]


def write_report(
    path: Path,
    manifest: dict[str, object],
    margin: dict[str, object],
    competition_k: dict[str, object],
    reserve: dict[str, object],
    external: dict[str, object],
    residuals: dict[str, pd.DataFrame],
) -> None:
    lines = [
        "# AO European Elo Final Robustness",
        "",
        "## Karar Ozeti",
        "",
        f"- Genel durum: `{manifest['overall_status']}`",
        "- Nihai comparator: carry=0, exact UTC chronology, nested six-fold core ve training-only 1X2 draw mapping.",
        "- Siralama hedefi, sezon sonu ratingini yalnizca bir sonraki sezonun schedule-adjusted performansiyla karsilastirir.",
        "- Forward siralama, optional katman seciminde Brier ve log-loss'tan once gelir.",
        "- 2026/27 hicbir parametre seciminde kullanilmamistir.",
        "",
        "## Optional Katmanlar",
        "",
    ]
    lines.extend(_layer_report_lines("Gol Farki", margin))
    lines.extend(_layer_report_lines("Turnuva K", competition_k))
    lines.extend(_layer_report_lines("European Achievement Reserve", reserve))
    lines.extend(
        [
            "## Final ClubElo Karsilastirmasi",
            "",
            f"- Durum: `{external['status']}`",
            f"- External rating bulunan eslesme: `{external['eligible_pairs']}`",
            f"- Walk-forward unseen eslesme: `{external['unseen_pairs']}`",
            "",
            "| Segment | Mac | AO Dynamic | AO Static | ClubElo | AO-ClubElo | Dynamic-Static |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in external["summary"].itertuples(index=False):
        lines.append(
            f"| {row.competition} | {row.matches} | {row.ao_brier_1x2:.6f} | "
            f"{row.static_brier_1x2:.6f} | {row.clubelo_brier_1x2:.6f} | "
            f"{row.ao_minus_clubelo_brier_1x2:+.6f} | "
            f"{row.dynamic_minus_static_brier_1x2:+.6f} |"
        )
    ucl_brier = envelope_row(external["uncertainty"], "UCL", "brier_1x2")
    ucl_expected = envelope_row(
        external["uncertainty"], "UCL", "expected_score_mse"
    )
    lines.extend(
        [
            "",
            f"- UCL 1X2 Brier envelope: `[{ucl_brier.ci_95_lower:+.6f}, {ucl_brier.ci_95_upper:+.6f}]`",
            f"- UCL expected-score MSE envelope: `[{ucl_expected.ci_95_lower:+.6f}, {ucl_expected.ci_95_upper:+.6f}]`",
            "",
            "Bu paired sample ClubElo arsivinin kapsadigi daha yerlesik takimlara agirlik verir; tum on eleme evrenini temsil etmez.",
            "Bu nedenle external sonuc guclu bir diagnostiktir, untouched holdout yerine gecmez.",
            "",
            "## Turnuva Residual Diagnostigi",
            "",
            "Pozitif residual ev sahibinin model beklentisinden daha fazla puan aldigini, negatif residual daha az aldigini gosterir. Bu tek basina turnuva gucu degildir; yalnizca etiket sonrasinda kalan sistematik kalibrasyon isaretidir.",
            "",
            "| Turnuva | Mac | Beklenen | Gerceklesen | Residual | Brier |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    competition_rows = residuals["summary"].loc[
        residuals["summary"]["level"].eq("competition")
    ]
    for row in competition_rows.itertuples(index=False):
        lines.append(
            f"| {row.competition} | {row.matches} | {row.mean_expected_home_score:.4f} | "
            f"{row.mean_actual_home_score:.4f} | {row.mean_score_residual:+.4f} | "
            f"{row.brier_1x2:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Yorumlama Siniri",
            "",
            "Bir katmanin kapali kalmasi fikrin mantiksiz oldugunu degil, bu veri ve onceden tanimli terfi kapilarinda ek tahmin/siralama kaniti uretmedigini ifade eder.",
            "UCL/UEL/UECL K katsayilari turnuva prestiji degil o turnuvadaki maclardan ogrenme hizidir. Prestij gerekirse Power Elo disinda ayri Achievement Index olarak sunulmalidir.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def json_default(value: object) -> object:
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


if __name__ == "__main__":
    main()
