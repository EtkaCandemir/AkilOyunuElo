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

from ao_elo.config import AO_MODEL_V2_VERSION  # noqa: E402
from scripts.run_achievement_carry_calibration import (  # noqa: E402
    CarrySeasonData,
    load_carry_data,
)
from scripts.run_dynamic_core_calibration import (  # noqa: E402
    DynamicCoreConfig,
    expanding_folds,
    expected_home_score,
)
from scripts.run_goal_margin_calibration import (  # noqa: E402
    GoalMarginConfig,
    build_goal_difference_distribution,
    build_multiplier_table,
    candidate_grid,
    goal_margin_multiplier,
)
from scripts.run_v2_dynamic_calibration import (  # noqa: E402
    MAX_RATING_MOVE_GUARDRAIL,
    RANK_CORRELATION_FLOOR,
    add_event_clusters,
    clustered_uncertainty,
    prefix_losses,
    read_event_metadata,
    read_static_config,
    summarize_loss_differences,
)


STATIC_DATA_ROOT = ROOT / "data" / "backtest_stage_b_2018_2026"
EXACT_EVENTS_PATH = (
    ROOT / "data" / "external_elo_benchmark_2018_2026" / "exact_date_events.csv"
)
STATIC_MANIFEST_PATH = (
    ROOT / "output" / "v2_ranking_calibration_2018_2026" / "selected_model.json"
)
DYNAMIC_OUTPUT_ROOT = ROOT / "output" / "v2_dynamic_calibration_2018_2026"
OUTPUT_ROOT = ROOT / "output" / "v2_goal_margin_calibration_2018_2026"
MODEL_NAME = "goal_margin"
BASELINE_NAME = "matched_baseline"


@dataclass(frozen=True)
class GoalCarrySeasonData:
    carry: CarrySeasonData
    goal_differences: np.ndarray
    penalty_flags: np.ndarray

    @property
    def season(self) -> str:
        return self.carry.season


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate goal-margin updates on the selected AO Elo v2 core+carry"
    )
    parser.add_argument("--static-data-root", type=Path, default=STATIC_DATA_ROOT)
    parser.add_argument("--events-path", type=Path, default=EXACT_EVENTS_PATH)
    parser.add_argument("--static-manifest", type=Path, default=STATIC_MANIFEST_PATH)
    parser.add_argument("--dynamic-output-root", type=Path, default=DYNAMIC_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    static_config = read_static_config(args.static_manifest.resolve())
    dynamic_root = args.dynamic_output_root.resolve()
    dynamic_manifest = read_dynamic_manifest(
        dynamic_root / "selected_dynamic_model.json"
    )
    events_path = args.events_path.resolve()
    event_metadata = read_event_metadata(events_path)
    datasets = load_goal_data(
        args.static_data_root.resolve(),
        events_path,
        static_config,
    )
    seasons = tuple(data.season for data in datasets)
    folds = expanding_folds(seasons)
    core_selections = pd.read_csv(dynamic_root / "core_fold_selections.csv")
    carry_selections = pd.read_csv(dynamic_root / "carry_fold_selections.csv")
    validate_fold_contract(core_selections, carry_selections, folds)

    selections, fold_results, predictions = run_walk_forward(
        datasets,
        folds,
        core_selections,
        carry_selections,
        event_metadata,
    )
    competition_summary = summarize_loss_differences(
        predictions, MODEL_NAME, BASELINE_NAME
    )
    uncertainty = clustered_uncertainty(predictions, MODEL_NAME, BASELINE_NAME)
    full_core = DynamicCoreConfig(**dynamic_manifest["dynamic_core"])
    active_carry = float(dynamic_manifest["active_power_carry"])
    final_candidate, full_candidate_metrics = select_candidate(
        datasets,
        full_core,
        active_carry,
    )
    decision, guardrails = promotion_decision(
        selections,
        fold_results,
        uncertainty,
        final_candidate,
    )
    active_config = (
        final_candidate
        if decision == "PROMOTE_GOAL_MARGIN"
        else GoalMarginConfig(0.0, 1.0)
    )
    multiplier_table = build_multiplier_table(active_config)
    distribution = build_goal_difference_distribution(
        tuple(
            type("MarginView", (), {"goal_differences": data.goal_differences})()
            for data in datasets
        )
    )

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    selections.to_csv(output_root / "fold_selections.csv", index=False)
    fold_results.to_csv(output_root / "fold_results.csv", index=False)
    predictions.to_csv(output_root / "unseen_predictions.csv", index=False)
    competition_summary.to_csv(
        output_root / "competition_summary.csv", index=False
    )
    uncertainty.to_csv(output_root / "clustered_uncertainty.csv", index=False)
    full_candidate_metrics.to_csv(
        output_root / "full_candidate_metrics.csv", index=False
    )
    multiplier_table.to_csv(output_root / "goal_multiplier_table.csv", index=False)
    distribution.to_csv(output_root / "goal_difference_distribution.csv", index=False)
    write_manifest(
        output_root / "selected_goal_model.json",
        dynamic_manifest,
        final_candidate,
        active_config,
        decision,
        guardrails,
    )
    write_report(
        output_root / "calibration_report.md",
        seasons,
        selections,
        competition_summary,
        uncertainty,
        final_candidate,
        active_config,
        decision,
        guardrails,
    )

    print("AO European Elo v2 goal-margin calibration")
    print(f"Candidate: weight={final_candidate.goal_weight:g}, cap={final_candidate.goal_cap:g}")
    print(f"Decision: {decision}")
    print(
        f"Active: weight={active_config.goal_weight:g}, cap={active_config.goal_cap:g}"
    )
    print(f"Report: {output_root / 'calibration_report.md'}")


def read_dynamic_manifest(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("dynamic_core_decision") != "PROMOTE_DYNAMIC_CORE":
        raise ValueError("Goal-margin calibration requires a promoted dynamic core")
    required = {"dynamic_core", "active_power_carry", "carry_decision"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Dynamic manifest missing keys: {missing}")
    return payload


def load_goal_data(
    static_root: Path,
    events_path: Path,
    static_config,
) -> tuple[GoalCarrySeasonData, ...]:
    carry_data, _ = load_carry_data(
        static_root,
        events_path,
        static_config,
        require_exact_utc=True,
    )
    events = pd.read_csv(events_path).sort_values(["season", "event_order"])
    required = {
        "match_id",
        "goal_difference",
        "decided_on_penalties",
        "actual_home_score",
        "home_goals",
        "away_goals",
        "result_basis",
    }
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"Goal-margin event data missing columns: {missing}")
    validate_score_contract(events)
    event_index = events.set_index("match_id")
    result = []
    for carry in carry_data:
        aligned = event_index.loc[carry.core.match_ids]
        result.append(
            GoalCarrySeasonData(
                carry=carry,
                goal_differences=aligned["goal_difference"].to_numpy(int),
                penalty_flags=aligned["decided_on_penalties"].to_numpy(bool),
            )
        )
    return tuple(result)


def validate_score_contract(events: pd.DataFrame) -> None:
    differences = pd.to_numeric(events["goal_difference"], errors="coerce")
    if differences.isna().any() or differences.lt(0).any() or (differences % 1).ne(0).any():
        raise ValueError("goal_difference must contain non-negative integers")
    if not events["actual_home_score"].isin((0.0, 0.5, 1.0)).all():
        raise ValueError("actual_home_score must contain 0, 0.5 or 1")
    home_goals = pd.to_numeric(events["home_goals"], errors="coerce")
    away_goals = pd.to_numeric(events["away_goals"], errors="coerce")
    if home_goals.isna().any() or away_goals.isna().any():
        raise ValueError("Field-score goals must be numeric")
    if not differences.eq((home_goals - away_goals).abs()).all():
        raise ValueError("goal_difference must match the 90/120-minute field score")
    expected_score = pd.Series(0.5, index=events.index)
    expected_score.loc[home_goals > away_goals] = 1.0
    expected_score.loc[home_goals < away_goals] = 0.0
    if not events["actual_home_score"].eq(expected_score).all():
        raise ValueError("actual_home_score must follow the 90/120-minute field score")
    if not events["result_basis"].eq("displayed_score_excluding_shootout").all():
        raise ValueError("Goal margin must exclude penalty shootout goals")


def validate_fold_contract(
    core: pd.DataFrame,
    carry: pd.DataFrame,
    folds: tuple[tuple[tuple[str, ...], str], ...],
) -> None:
    for label, frame in (("core", core), ("carry", carry)):
        if len(frame) != len(folds) or not frame["fold"].is_unique:
            raise ValueError(f"{label} fold selections do not match goal folds")
        for fold, (_, season) in enumerate(folds, start=1):
            row = frame.loc[frame["fold"].eq(fold)]
            if len(row) != 1 or row.iloc[0]["test_season"] != season:
                raise ValueError(f"{label} fold {fold} has wrong unseen season")


def evaluate_sequence(
    datasets: tuple[GoalCarrySeasonData, ...],
    core_config: DynamicCoreConfig,
    power_carry: float,
    margin_config: GoalMarginConfig,
    *,
    evaluation_seasons: set[str] | None = None,
    return_predictions: bool = False,
) -> tuple[dict[str, float | int], pd.DataFrame | None]:
    core_config.validate()
    margin_config.validate()
    if not math.isfinite(power_carry) or not 0.0 <= power_carry <= 1.0:
        raise ValueError("power_carry must be in [0,1]")
    evaluation = evaluation_seasons or {data.season for data in datasets}
    previous_power: dict[str, float] = {}
    metric_rows: list[dict[str, float | int]] = []
    prediction_rows: list[dict[str, object]] = []
    for data in datasets:
        carry = data.carry
        core = carry.core
        power = core.initial_ratings.copy()
        for team_id in core.active_team_ids:
            key = str(carry.club_keys[team_id])
            if key in previous_power:
                power[team_id] = (
                    (1.0 - power_carry) * core.initial_ratings[team_id]
                    + power_carry * previous_power[key]
                )
        brier_sum = 0.0
        log_loss_sum = 0.0
        max_match_delta = 0.0
        max_pair_sum_error = 0.0
        for index, (home_id, away_id, actual, neutral, difference) in enumerate(
            zip(
                core.home_team_ids,
                core.away_team_ids,
                core.actual_home_scores,
                core.neutral_flags,
                data.goal_differences,
            )
        ):
            probability = float(
                np.clip(
                    expected_home_score(
                        power[home_id],
                        power[away_id],
                        core_config,
                        neutral=bool(neutral),
                    ),
                    1e-12,
                    1.0 - 1e-12,
                )
            )
            multiplier = goal_margin_multiplier(int(difference), margin_config)
            delta = core_config.k_factor * multiplier * (actual - probability)
            home_before = power[home_id]
            away_before = power[away_id]
            power[home_id] += delta
            power[away_id] -= delta
            pair_sum_error = abs(
                (power[home_id] + power[away_id]) - (home_before + away_before)
            )
            max_pair_sum_error = max(max_pair_sum_error, pair_sum_error)
            brier_loss = (probability - actual) ** 2
            log_loss = -(
                actual * math.log(probability)
                + (1.0 - actual) * math.log(1.0 - probability)
            )
            if data.season in evaluation:
                brier_sum += brier_loss
                log_loss_sum += log_loss
                max_match_delta = max(max_match_delta, abs(delta))
                if return_predictions:
                    prediction_rows.append(
                        {
                            "match_id": core.match_ids[index],
                            "season": data.season,
                            "competition": core.competitions[index],
                            "actual_home_score": actual,
                            "decided_on_penalties": bool(data.penalty_flags[index]),
                            "goal_difference": int(difference),
                            "goal_multiplier": multiplier,
                            "home_power": home_before,
                            "away_power": away_before,
                            "expected_home_score": probability,
                            "power_delta": delta,
                            "brier_loss": brier_loss,
                            "log_loss": log_loss,
                        }
                    )
        active = core.active_team_ids
        start = core.initial_ratings[active]
        end = power[active]
        rank_correlation = safe_rank_correlation(start, end)
        if data.season in evaluation:
            changes = end - start
            metric_rows.append(
                {
                    "matches": len(core.match_ids),
                    "brier": brier_sum / len(core.match_ids),
                    "log_loss": log_loss_sum / len(core.match_ids),
                    "max_abs_match_delta": max_match_delta,
                    "max_pair_sum_error": max_pair_sum_error,
                    "max_abs_rating_change": float(np.max(np.abs(changes))),
                    "start_end_rank_correlation": rank_correlation,
                }
            )
        previous_power = {
            str(carry.club_keys[team_id]): float(power[team_id])
            for team_id in active
        }
    if not metric_rows:
        raise ValueError("No goal-margin evaluation seasons were processed")
    matches = sum(int(row["matches"]) for row in metric_rows)
    metrics: dict[str, float | int] = {
        "matches": matches,
        "brier": sum(float(row["brier"]) * int(row["matches"]) for row in metric_rows)
        / matches,
        "log_loss": sum(
            float(row["log_loss"]) * int(row["matches"]) for row in metric_rows
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
    predictions = pd.DataFrame(prediction_rows) if return_predictions else None
    return metrics, predictions


def safe_rank_correlation(start: np.ndarray, end: np.ndarray) -> float:
    if np.allclose(start, end):
        return 1.0
    if np.ptp(start) == 0 or np.ptp(end) == 0:
        return 0.0
    value = pd.Series(start).corr(pd.Series(end), method="spearman")
    return 0.0 if pd.isna(value) else float(value)


def candidate_metrics(
    datasets: tuple[GoalCarrySeasonData, ...],
    core: DynamicCoreConfig,
    power_carry: float,
) -> pd.DataFrame:
    rows = []
    for config in candidate_grid():
        metrics, _ = evaluate_sequence(datasets, core, power_carry, config)
        rows.append(
            {
                "goal_weight": config.goal_weight,
                "goal_cap": config.goal_cap,
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["brier", "log_loss", "goal_weight", "goal_cap"]
    ).reset_index(drop=True)


def select_candidate(
    datasets: tuple[GoalCarrySeasonData, ...],
    core: DynamicCoreConfig,
    power_carry: float,
) -> tuple[GoalMarginConfig, pd.DataFrame]:
    metrics = candidate_metrics(datasets, core, power_carry)
    eligible = metrics.loc[
        metrics["start_end_rank_correlation"].ge(RANK_CORRELATION_FLOOR)
        & metrics["max_abs_rating_change"].le(MAX_RATING_MOVE_GUARDRAIL)
        & metrics["max_pair_sum_error"].le(1e-9)
    ]
    if eligible.empty:
        raise ValueError("No goal-margin candidate passes mandatory guardrails")
    row = eligible.iloc[0]
    return GoalMarginConfig(float(row["goal_weight"]), float(row["goal_cap"])), metrics


def run_walk_forward(
    datasets: tuple[GoalCarrySeasonData, ...],
    folds: tuple[tuple[tuple[str, ...], str], ...],
    core_selections: pd.DataFrame,
    carry_selections: pd.DataFrame,
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline = GoalMarginConfig(0.0, 1.0)
    selection_rows: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for fold, (train_seasons, test_season) in enumerate(folds, start=1):
        core_row = core_selections.loc[core_selections["fold"].eq(fold)].iloc[0]
        carry_row = carry_selections.loc[carry_selections["fold"].eq(fold)].iloc[0]
        core = DynamicCoreConfig(
            float(core_row["selected_scale"]),
            float(core_row["selected_home_advantage"]),
            float(core_row["selected_k"]),
        )
        power_carry = float(carry_row["selected_power_carry"])
        train = tuple(data for data in datasets if data.season in train_seasons)
        sequence = tuple(
            data for data in datasets if data.season in (*train_seasons, test_season)
        )
        selected, train_metrics = select_candidate(train, core, power_carry)
        baseline_metrics, _ = evaluate_sequence(train, core, power_carry, baseline)
        selected_train = train_metrics.loc[
            train_metrics["goal_weight"].eq(selected.goal_weight)
            & train_metrics["goal_cap"].eq(selected.goal_cap)
        ].iloc[0]
        selection_rows.append(
            {
                "fold": fold,
                "train_seasons": "|".join(train_seasons),
                "test_season": test_season,
                "core_scale": core.elo_scale,
                "core_home_advantage": core.home_advantage,
                "core_k": core.k_factor,
                "power_carry": power_carry,
                "selected_goal_weight": selected.goal_weight,
                "selected_goal_cap": selected.goal_cap,
                "train_brier_difference": (
                    selected_train["brier"] - baseline_metrics["brier"]
                ),
            }
        )
        models: dict[str, pd.DataFrame] = {}
        for model, config in ((MODEL_NAME, selected), (BASELINE_NAME, baseline)):
            metrics, predictions = evaluate_sequence(
                sequence,
                core,
                power_carry,
                config,
                evaluation_seasons={test_season},
                return_predictions=True,
            )
            result_rows.append(
                {
                    "fold": fold,
                    "test_season": test_season,
                    "model": model,
                    "goal_weight": config.goal_weight,
                    "goal_cap": config.goal_cap,
                    **metrics,
                }
            )
            assert predictions is not None
            models[model] = prefix_losses(predictions, model)
        joined = models[MODEL_NAME].merge(
            models[BASELINE_NAME][
                [
                    "match_id",
                    f"{BASELINE_NAME}_expected_home_score",
                    f"{BASELINE_NAME}_brier_loss",
                    f"{BASELINE_NAME}_log_loss",
                ]
            ],
            on="match_id",
            validate="one_to_one",
        )
        joined.insert(0, "fold", fold)
        prediction_frames.append(add_event_clusters(joined, events))
    return (
        pd.DataFrame(selection_rows),
        pd.DataFrame(result_rows),
        pd.concat(prediction_frames, ignore_index=True),
    )


def promotion_decision(
    selections: pd.DataFrame,
    fold_results: pd.DataFrame,
    uncertainty: pd.DataFrame,
    final_candidate: GoalMarginConfig,
) -> tuple[str, dict[str, object]]:
    pivot = fold_results.pivot(index="fold", columns="model", values="brier")
    fold_wins = int((pivot[MODEL_NAME] < pivot[BASELINE_NAME]).sum())
    model_rows = fold_results.loc[fold_results["model"].eq(MODEL_NAME)]
    overall = uncertainty.loc[uncertainty["competition"].eq("ALL")].iloc[0]
    segments = uncertainty.loc[uncertainty["competition"].ne("ALL")]
    nonzero_fold_share = float(selections["selected_goal_weight"].gt(0).mean())
    ranking_safe = bool(
        model_rows["start_end_rank_correlation"].ge(RANK_CORRELATION_FLOOR).all()
    )
    movement_safe = bool(
        model_rows["max_abs_rating_change"].le(MAX_RATING_MOVE_GUARDRAIL).all()
    )
    zero_sum = bool(model_rows["max_pair_sum_error"].le(1e-9).all())
    no_segment_harm = not bool(segments["reliable_harm"].any())
    passed = (
        final_candidate.goal_weight > 0
        and nonzero_fold_share >= 0.5
        and fold_wins >= 5
        and bool(overall["reliable_improvement"])
        and ranking_safe
        and movement_safe
        and zero_sum
        and no_segment_harm
    )
    guardrails = {
        "fold_wins": fold_wins,
        "nonzero_fold_share": nonzero_fold_share,
        "overall_reliable_improvement": bool(overall["reliable_improvement"]),
        "ranking_safe": ranking_safe,
        "movement_safe": movement_safe,
        "zero_sum": zero_sum,
        "no_competition_reliable_harm": no_segment_harm,
    }
    return ("PROMOTE_GOAL_MARGIN" if passed else "DISABLE_GOAL_MARGIN"), guardrails


def write_manifest(
    path: Path,
    dynamic_manifest: dict[str, object],
    candidate: GoalMarginConfig,
    active: GoalMarginConfig,
    decision: str,
    guardrails: dict[str, object],
) -> None:
    payload = {
        "model_version": AO_MODEL_V2_VERSION,
        "dynamic_core": dynamic_manifest["dynamic_core"],
        "active_power_carry": dynamic_manifest["active_power_carry"],
        "goal_margin_candidate": {
            "goal_weight": candidate.goal_weight,
            "goal_cap": candidate.goal_cap,
        },
        "goal_margin": {
            "active": active.goal_weight > 0,
            "goal_weight": active.goal_weight,
            "goal_cap": active.goal_cap,
        },
        "decision": decision,
        "guardrails": guardrails,
        "achievement_reserve": {"active": False, "base": 0.0},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_report(
    path: Path,
    seasons: tuple[str, ...],
    selections: pd.DataFrame,
    competitions: pd.DataFrame,
    uncertainty: pd.DataFrame,
    candidate: GoalMarginConfig,
    active: GoalMarginConfig,
    decision: str,
    guardrails: dict[str, object],
) -> None:
    overall = uncertainty.loc[uncertainty["competition"].eq("ALL")].iloc[0]
    lines = [
        "# AO European Elo v2 Goal-Margin Calibration",
        "",
        f"Exact-date seasons: `{seasons[0]}` through `{seasons[-1]}`.",
        "The comparator is the selected dynamic core plus its nested season carry.",
        "The score is the 90/120-minute field score; penalty shootout goals are excluded.",
        "",
        "## Decision",
        "",
        f"- Result: `{decision}`",
        f"- Full-data candidate: weight `{candidate.goal_weight:g}`, cap `{candidate.goal_cap:g}`",
        f"- Active config: weight `{active.goal_weight:g}`, cap `{active.goal_cap:g}`",
        f"- Unseen fold wins: `{guardrails['fold_wins']}/6`",
        f"- Overall Brier difference: `{overall.mean_brier_difference:+.6f}`",
        f"- Clustered 95% CI: `[{overall.ci_95_lower:+.6f}, "
        f"{overall.ci_95_upper:+.6f}]`",
        f"- Ranking guardrail: `{guardrails['ranking_safe']}`",
        f"- Zero-sum update guardrail: `{guardrails['zero_sum']}`",
        "",
        "| Competition | Matches | Brier difference | Log-loss difference |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in competitions.itertuples(index=False):
        lines.append(
            f"| {row.competition} | {row.matches} | {row.brier_difference:+.6f} | "
            f"{row.log_loss_difference:+.6f} |"
        )
    lines.extend(
        [
            "",
            "A disabled result means every production goal multiplier remains exactly 1.0.",
            "The candidate is retained only as research evidence, not as an active layer.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
