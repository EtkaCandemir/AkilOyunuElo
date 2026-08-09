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

from ao_elo.evaluation import (  # noqa: E402
    dependency_robust_loss_difference_ci,
    schedule_adjusted_team_performance,
)
from ao_elo.robustness import one_x_two_probabilities_scalar  # noqa: E402
from ao_elo.xg_live import (  # noqa: E402
    XGBlendConfig,
    XGPerformanceBonusConfig,
    update_match_elo_with_xg,
)
from scripts.run_controlled_goal_progression_backtest import (  # noqa: E402
    evaluate_predictions,
)
from scripts.run_fotmob_xg_backtest_2025_26 import (  # noqa: E402
    DATA_PATH,
    MANIFEST_PATH,
    PRODUCTION_MODEL_PATH,
    SEASON,
    STATIC_DATA_ROOT,
    STATIC_MANIFEST_PATH,
    markdown_table,
    read_fotmob_xg_dataset,
    validate_manifest,
)
from scripts.run_xg_goal_ablation_backtest import (  # noqa: E402
    elo_field_score,
    load_initial_ratings,
    read_production_contract,
    read_static_config,
    same_season_ranking,
)


OUTPUT_ROOT = ROOT / "output" / "xg_performance_bonus_walk_forward_2025_26"
BASELINE_KEY = "GD_PRODUCTION"
BETA_GRID = (0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00)
XG_SCALE_GRID = (0.75, 1.00, 1.50, 2.00, 3.00)
WINNER_FLOOR_GRID = (0.05, 0.10, 0.25, 0.50)


@dataclass(frozen=True, order=True)
class WalkForwardFold:
    key: str
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass(frozen=True, order=True)
class XGPerformanceCandidate:
    key: str
    beta: float
    xg_scale: float
    minimum_winner_gain_ratio: float

    @property
    def is_baseline(self) -> bool:
        return self.beta == 0.0

    @property
    def config(self) -> XGPerformanceBonusConfig | None:
        if self.is_baseline:
            return None
        return XGPerformanceBonusConfig(
            self.beta,
            self.xg_scale,
            self.minimum_winner_gain_ratio,
        )

    def validate(self) -> None:
        if self.is_baseline:
            if self.key != BASELINE_KEY:
                raise ValueError("The beta=0 candidate must be production baseline")
            return
        assert self.config is not None
        self.config.validate()


@dataclass
class CandidateReplay:
    predictions: pd.DataFrame
    snapshots: dict[pd.Timestamp, pd.DataFrame]
    final_ratings: pd.DataFrame


def outer_folds() -> tuple[WalkForwardFold, ...]:
    values = (
        ("F1_NOV_DEC", "2025-11-01", "2026-01-01"),
        ("F2_JAN", "2026-01-01", "2026-02-01"),
        ("F3_FEB", "2026-02-01", "2026-03-01"),
        ("F4_MAR", "2026-03-01", "2026-04-01"),
        ("F5_APR_MAY", "2026-04-01", "2026-06-01"),
    )
    return tuple(
        WalkForwardFold(
            key,
            pd.Timestamp(start, tz="UTC"),
            pd.Timestamp(end, tz="UTC"),
        )
        for key, start, end in values
    )


def candidate_grid() -> tuple[XGPerformanceCandidate, ...]:
    candidates = [XGPerformanceCandidate(BASELINE_KEY, 0.0, 1.0, 1.0)]
    candidates.extend(
        XGPerformanceCandidate(
            f"XG_BONUS_beta{beta:g}_scale{xg_scale:g}_floor{floor:g}",
            beta,
            xg_scale,
            floor,
        )
        for beta in BETA_GRID
        for xg_scale in XG_SCALE_GRID
        for floor in WINNER_FLOOR_GRID
    )
    result = tuple(candidates)
    for candidate in result:
        candidate.validate()
    if len(result) != 201:
        raise ValueError(f"Expected 201 candidates, found {len(result)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Tune a direction-preserving, two-sided xG performance bonus with "
            "five expanding walk-forward folds"
        )
    )
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--static-data-root", type=Path, default=STATIC_DATA_ROOT)
    parser.add_argument("--static-manifest", type=Path, default=STATIC_MANIFEST_PATH)
    parser.add_argument("--production-model", type=Path, default=PRODUCTION_MODEL_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples must be at least 100")

    events = read_fotmob_xg_dataset(args.data.resolve(), strict_contract=True)
    manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    validate_manifest(manifest, events)
    production = read_production_contract(args.production_model.resolve())
    static_config = read_static_config(args.static_manifest.resolve())
    initial_ratings = load_initial_ratings(
        args.static_data_root.resolve(), events, static_config
    )[SEASON]
    folds = outer_folds()
    candidates = candidate_grid()
    validate_fold_contract(events, folds)

    print(
        f"xG performance bonus: {len(candidates)} candidates, "
        f"{len(folds)} expanding folds, 387 unseen matches",
        flush=True,
    )
    cutoffs = tuple(
        sorted({fold.test_start for fold in folds} | {fold.test_end for fold in folds})
    )
    evaluations = {
        candidate.key: replay_candidate(
            events,
            initial_ratings,
            production,
            candidate,
            snapshot_times=cutoffs,
        )
        for candidate in candidates
    }

    selection_rows: list[dict[str, object]] = []
    fold_result_rows: list[dict[str, object]] = []
    outer_prediction_frames: list[pd.DataFrame] = []
    for fold in folds:
        train_events = events.loc[events["kickoff_utc"].lt(fold.test_start)]
        test_events = events.loc[
            events["kickoff_utc"].ge(fold.test_start)
            & events["kickoff_utc"].lt(fold.test_end)
        ]
        train_target = schedule_adjusted_team_performance(train_events)
        test_target = schedule_adjusted_team_performance(test_events)
        training_metrics = build_candidate_metrics(
            evaluations,
            candidates,
            train_target,
            prediction_start=None,
            prediction_end=fold.test_start,
            rating_time=fold.test_start,
        )
        selected, selection = select_candidate(training_metrics, candidates)
        selection_rows.append(
            {
                "fold": fold.key,
                "train_matches": len(train_events),
                "train_xg_matches": int(train_events["xg_analysis_eligible"].sum()),
                "test_matches": len(test_events),
                "selected_candidate": selected.key,
                "selected_beta": selected.beta,
                "selected_xg_scale": selected.xg_scale,
                "selected_minimum_winner_gain_ratio": (
                    selected.minimum_winner_gain_ratio
                ),
                **selection,
            }
        )
        baseline_predictions = prediction_slice(
            evaluations[BASELINE_KEY].predictions,
            fold.test_start,
            fold.test_end,
        ).assign(model="GD_PRODUCTION", fold=fold.key)
        selected_predictions = prediction_slice(
            evaluations[selected.key].predictions,
            fold.test_start,
            fold.test_end,
        ).assign(
            model="NESTED_SELECTED_XG",
            fold=fold.key,
            selected_candidate=selected.key,
        )
        outer_prediction_frames.extend([baseline_predictions, selected_predictions])
        for model, candidate, frame in (
            ("GD_PRODUCTION", candidates[0], baseline_predictions),
            ("NESTED_SELECTED_XG", selected, selected_predictions),
        ):
            rating_snapshot = evaluations[candidate.key].snapshots[fold.test_start]
            ranking = same_season_ranking(rating_snapshot, test_target, {SEASON})
            pooled = ranking.loc[ranking["competition"].eq("ALL")].iloc[0]
            fold_result_rows.append(
                {
                    "fold": fold.key,
                    "model": model,
                    "candidate_key": candidate.key,
                    **evaluate_predictions(frame),
                    "forward_ranking_score": float(pooled["ranking_score"]),
                    "forward_pairwise_accuracy": float(pooled["pairwise_accuracy"]),
                    "max_abs_match_delta": float(frame["power_delta"].abs().max()),
                    "minimum_winner_update": float(
                        frame.loc[frame["actual_class"].ne(1), "winner_elo_gain"].min()
                    ),
                    "maximum_zero_sum_error": float(frame["zero_sum_error"].max()),
                }
            )

    fold_selections = pd.DataFrame(selection_rows)
    fold_results = add_fold_deltas(pd.DataFrame(fold_result_rows))
    outer_predictions = pd.concat(outer_prediction_frames, ignore_index=True)
    pooled_comparison = build_pooled_comparison(outer_predictions, fold_results)
    competition_summary = build_competition_summary(outer_predictions)
    uncertainty = build_uncertainty(
        outer_predictions,
        bootstrap_samples=args.bootstrap_samples,
    )

    full_target = schedule_adjusted_team_performance(events)
    full_metrics = build_candidate_metrics(
        evaluations,
        candidates,
        full_target,
        prediction_start=None,
        prediction_end=pd.Timestamp("2026-06-01", tz="UTC"),
        rating_time=pd.Timestamp("2026-06-01", tz="UTC"),
    )
    deployment_candidate, deployment_selection = select_candidate(
        full_metrics,
        candidates,
    )
    decision, guardrails = make_decision(
        fold_selections,
        fold_results,
        pooled_comparison,
        competition_summary,
        uncertainty,
        deployment_candidate,
    )
    examples = build_examples(
        evaluations[deployment_candidate.key].predictions
        if not deployment_candidate.is_baseline
        else outer_predictions.loc[
            outer_predictions["model"].eq("NESTED_SELECTED_XG")
        ]
    )

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([candidate.__dict__ for candidate in candidates]).to_csv(
        output_root / "candidate_grid.csv", index=False
    )
    fold_selections.to_csv(output_root / "fold_selections.csv", index=False)
    fold_results.to_csv(output_root / "fold_results.csv", index=False)
    outer_predictions.to_csv(output_root / "outer_predictions.csv", index=False)
    pooled_comparison.to_csv(output_root / "pooled_comparison.csv", index=False)
    competition_summary.to_csv(
        output_root / "competition_summary.csv", index=False
    )
    uncertainty.to_csv(output_root / "dependency_uncertainty.csv", index=False)
    full_metrics.to_csv(output_root / "full_data_candidate_metrics.csv", index=False)
    examples.to_csv(output_root / "xg_bonus_examples.csv", index=False)

    payload = {
        "analysis": "XG_PERFORMANCE_BONUS_EXPANDING_WALK_FORWARD_2025_26",
        "formula": {
            "base": "Delta_base=K*(S-E)",
            "goal_margin": "Delta_GD=Delta_base*(M_GD-1)",
            "xg_signal": "Q_xG=tanh((xG_home-xG_away)/xG_scale)",
            "xg_adjustment": "Delta_xG=beta*abs(Delta_base)*Q_xG",
            "raw": "Delta_raw=Delta_base+Delta_GD+Delta_xG",
            "floor": (
                "Winner update is clamped to the result direction at "
                "minimum_winner_gain_ratio*abs(Delta_base)"
            ),
        },
        "outer_folds": [
            {
                "key": fold.key,
                "test_start": fold.test_start.isoformat(),
                "test_end": fold.test_end.isoformat(),
            }
            for fold in folds
        ],
        "deployment_candidate_from_full_2025_26": deployment_candidate.__dict__,
        "deployment_selection": deployment_selection,
        "decision": decision,
        "guardrails": guardrails,
        "production_changed": False,
        "evidence_class": "RETROSPECTIVE_EXPANDING_WALK_FORWARD",
        "prospective_confirmation_required": True,
    }
    (output_root / "selected_xg_performance_bonus_model.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_root / "backtest_report.md").write_text(
        build_report(
            fold_selections,
            fold_results,
            pooled_comparison,
            competition_summary,
            uncertainty,
            deployment_candidate,
            deployment_selection,
            decision,
            guardrails,
        ),
        encoding="utf-8",
    )
    print(f"Full-data deployment candidate: {deployment_candidate.key}")
    print(f"Decision: {decision}")
    print(f"Report: {output_root / 'backtest_report.md'}")


def validate_fold_contract(
    events: pd.DataFrame,
    folds: tuple[WalkForwardFold, ...],
) -> None:
    test_ids: list[str] = []
    expected_counts = (180, 72, 48, 48, 39)
    for fold, expected in zip(folds, expected_counts):
        test = events.loc[
            events["kickoff_utc"].ge(fold.test_start)
            & events["kickoff_utc"].lt(fold.test_end)
        ]
        if len(test) != expected or not test["xg_analysis_eligible"].all():
            raise ValueError(
                f"{fold.key}: expected {expected} fully xG-covered matches"
            )
        test_ids.extend(test["match_id"].astype(str))
    if len(test_ids) != 387 or len(set(test_ids)) != 387:
        raise ValueError("Outer folds must contain 387 disjoint matches")


def replay_candidate(
    events: pd.DataFrame,
    initial_ratings: dict[int, float],
    production: dict[str, object],
    candidate: XGPerformanceCandidate,
    *,
    snapshot_times: tuple[pd.Timestamp, ...],
) -> CandidateReplay:
    candidate.validate()
    core = production["dynamic_core"]
    goal = production["goal_margin"]
    draw = production["one_x_two_probability"]
    power = {int(team_id): float(value) for team_id, value in initial_ratings.items()}
    active_ids = sorted(
        set(events["home_team_id"].astype(int))
        | set(events["away_team_id"].astype(int))
    )
    missing = sorted(set(active_ids) - set(power))
    if missing:
        raise ValueError(f"Initial ratings missing teams: {missing[:5]}")
    start = dict(power)
    names = dict(
        zip(events["home_team_id"].astype(int), events["home_team_name"].astype(str))
    )
    names.update(
        zip(events["away_team_id"].astype(int), events["away_team_name"].astype(str))
    )
    snapshots: dict[pd.Timestamp, pd.DataFrame] = {}
    pending = list(snapshot_times)
    rows: list[dict[str, object]] = []

    def rating_frame() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "candidate_key": candidate.key,
                    "season": SEASON,
                    "team_id": team_id,
                    "team_name": names.get(team_id, str(team_id)),
                    "start_rating": start[team_id],
                    "end_live_rating": power[team_id],
                    "rating_change": power[team_id] - start[team_id],
                }
                for team_id in active_ids
            ]
        )

    for row in events.itertuples(index=False):
        while pending and row.kickoff_utc >= pending[0]:
            snapshots[pending.pop(0)] = rating_frame()
        home_id = int(row.home_team_id)
        away_id = int(row.away_team_id)
        penalties = bool(row.decided_on_penalties)
        home_goals, away_goals = elo_field_score(
            int(row.home_goals), int(row.away_goals), penalties
        )
        eligible = bool(row.xg_analysis_eligible)
        xg_home = float(row.xg_home) if eligible else None
        xg_away = float(row.xg_away) if eligible else None
        update = update_match_elo_with_xg(
            power[home_id],
            power[away_id],
            home_goals,
            away_goals,
            k_factor=float(core["k_factor"]),
            elo_scale=float(core["elo_scale"]),
            home_advantage=float(core["home_advantage"]),
            is_neutral=bool(row.is_neutral),
            decided_on_penalties=penalties,
            goal_difference_enabled=True,
            goal_alpha=float(goal["alpha"]),
            goal_tau=float(goal["tau"]),
            goal_difference_cap=int(goal["goal_difference_cap"]),
            xg_config=XGBlendConfig(0.0, 1.0),
            xg_home=xg_home,
            xg_away=xg_away,
            xg_performance_bonus_config=candidate.config,
        )
        probabilities = one_x_two_probabilities_scalar(
            update.expected_home_score,
            float(draw["draw_at_even"]),
            float(draw["draw_shape"]),
        )
        # Penalty shoot-outs remain draws for the match Elo and 1X2 target,
        # even when the source's displayed field score differs after extra time.
        actual_score = float(row.actual_home_score)
        observed = 0 if actual_score == 1.0 else 1 if actual_score == 0.5 else 2
        target = tuple(1.0 if index == observed else 0.0 for index in range(3))
        brier = sum(
            (probability - actual) ** 2
            for probability, actual in zip(probabilities, target)
        )
        log_loss = -math.log(max(probabilities[observed], 1e-15))
        power[home_id] = update.home_rating_post
        power[away_id] = update.away_rating_post
        winner_gain = (
            abs(update.power_delta) if actual_score in {0.0, 1.0} else 0.0
        )
        if actual_score == 1.0 and update.power_delta <= 0.0:
            raise ValueError(f"{row.match_id}: home winner did not gain Elo")
        if actual_score == 0.0 and update.power_delta >= 0.0:
            raise ValueError(f"{row.match_id}: away winner did not gain Elo")
        rows.append(
            {
                "candidate_key": candidate.key,
                "beta": candidate.beta,
                "xg_scale": candidate.xg_scale,
                "minimum_winner_gain_ratio": candidate.minimum_winner_gain_ratio,
                "match_id": str(row.match_id),
                "season": str(row.season),
                "competition": str(row.competition),
                "round": str(row.round),
                "kickoff_utc": row.kickoff_utc,
                "tie_id": None if pd.isna(row.tie_id) else str(row.tie_id),
                "home_team_id": home_id,
                "away_team_id": away_id,
                "home_team_name": str(row.home_team_name),
                "away_team_name": str(row.away_team_name),
                "home_goals": int(row.home_goals),
                "away_goals": int(row.away_goals),
                "decided_on_penalties": penalties,
                "xg_analysis_eligible": eligible,
                "xg_home": xg_home,
                "xg_away": xg_away,
                "xg_difference": None if not eligible else xg_home - xg_away,
                "home_rating_pre": update.home_rating_pre,
                "away_rating_pre": update.away_rating_pre,
                "expected_home_score": update.expected_home_score,
                "home_probability": probabilities[0],
                "draw_probability": probabilities[1],
                "away_probability": probabilities[2],
                "actual_class": observed,
                "predicted_class": int(np.argmax(probabilities)),
                "brier_1x2": brier,
                "log_loss_1x2": log_loss,
                "goal_difference": update.goal_difference,
                "goal_multiplier": update.goal_difference_multiplier,
                "base_result_residual": update.base_result_residual,
                "goal_bonus_residual": update.goal_bonus_residual,
                "xg_performance_signal": update.xg_performance_signal,
                "xg_performance_adjustment": update.xg_performance_adjustment,
                "direction_floor_residual": update.direction_floor_residual,
                "blended_residual": update.blended_residual,
                "power_delta": update.power_delta,
                "winner_elo_gain": winner_gain,
                "home_rating_post": update.home_rating_post,
                "away_rating_post": update.away_rating_post,
                "zero_sum_error": update.zero_sum_error,
            }
        )
    while pending:
        snapshots[pending.pop(0)] = rating_frame()
    return CandidateReplay(pd.DataFrame(rows), snapshots, rating_frame())


def prediction_slice(
    predictions: pd.DataFrame,
    start: pd.Timestamp | None,
    end: pd.Timestamp,
) -> pd.DataFrame:
    mask = predictions["kickoff_utc"].lt(end)
    if start is not None:
        mask &= predictions["kickoff_utc"].ge(start)
    result = predictions.loc[mask].copy()
    if result.empty:
        raise ValueError("Prediction slice cannot be empty")
    return result


def build_candidate_metrics(
    evaluations: dict[str, CandidateReplay],
    candidates: tuple[XGPerformanceCandidate, ...],
    ranking_target: pd.DataFrame,
    *,
    prediction_start: pd.Timestamp | None,
    prediction_end: pd.Timestamp,
    rating_time: pd.Timestamp,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        predictions = prediction_slice(
            evaluations[candidate.key].predictions,
            prediction_start,
            prediction_end,
        )
        ranking = same_season_ranking(
            evaluations[candidate.key].snapshots[rating_time],
            ranking_target,
            {SEASON},
        )
        pooled = ranking.loc[ranking["competition"].eq("ALL")].iloc[0]
        rows.append(
            {
                "candidate_key": candidate.key,
                "beta": candidate.beta,
                "xg_scale": candidate.xg_scale,
                "minimum_winner_gain_ratio": candidate.minimum_winner_gain_ratio,
                **evaluate_predictions(predictions),
                "ranking_score": float(pooled["ranking_score"]),
                "pairwise_accuracy": float(pooled["pairwise_accuracy"]),
                "max_abs_match_delta": float(predictions["power_delta"].abs().max()),
                "minimum_winner_update": float(
                    predictions.loc[
                        predictions["actual_class"].ne(1), "winner_elo_gain"
                    ].min()
                ),
                "maximum_zero_sum_error": float(predictions["zero_sum_error"].max()),
            }
        )
    result = pd.DataFrame(rows)
    baseline = result.loc[result["candidate_key"].eq(BASELINE_KEY)].iloc[0]
    for metric in (
        "brier_1x2",
        "log_loss_1x2",
        "ranking_score",
        "pairwise_accuracy",
    ):
        result[f"{metric}_delta_vs_production"] = (
            result[metric] - float(baseline[metric])
        )
    return result


def select_candidate(
    metrics: pd.DataFrame,
    candidates: tuple[XGPerformanceCandidate, ...],
) -> tuple[XGPerformanceCandidate, dict[str, object]]:
    baseline = metrics.loc[metrics["candidate_key"].eq(BASELINE_KEY)].iloc[0]
    values = metrics.copy()
    values["ranking_safe"] = (
        values["ranking_score"].ge(float(baseline["ranking_score"]))
        & values["pairwise_accuracy"].ge(float(baseline["pairwise_accuracy"]))
    )
    values["loss_objective"] = (
        values["brier_1x2_delta_vs_production"] / float(baseline["brier_1x2"])
        + values["log_loss_1x2_delta_vs_production"]
        / float(baseline["log_loss_1x2"])
    )
    pool = values.loc[values["ranking_safe"]].copy()
    selected_row = pool.sort_values(
        [
            "loss_objective",
            "ranking_score",
            "pairwise_accuracy",
            "beta",
            "xg_scale",
            "minimum_winner_gain_ratio",
        ],
        ascending=[True, False, False, True, False, False],
    ).iloc[0]
    selected = next(
        candidate
        for candidate in candidates
        if candidate.key == selected_row["candidate_key"]
    )
    return selected, {
        "ranking_safe_candidates": int(values["ranking_safe"].sum()),
        "selected_is_production_baseline": selected.is_baseline,
        "selected_training_brier_delta": float(
            selected_row["brier_1x2_delta_vs_production"]
        ),
        "selected_training_log_loss_delta": float(
            selected_row["log_loss_1x2_delta_vs_production"]
        ),
        "selected_training_ranking_delta": float(
            selected_row["ranking_score_delta_vs_production"]
        ),
        "selected_training_pairwise_delta": float(
            selected_row["pairwise_accuracy_delta_vs_production"]
        ),
        "future_results_used_for_selection": False,
    }


def add_fold_deltas(results: pd.DataFrame) -> pd.DataFrame:
    output = results.copy()
    for fold in output["fold"].unique():
        mask = output["fold"].eq(fold)
        baseline = output.loc[mask & output["model"].eq("GD_PRODUCTION")].iloc[0]
        for metric in (
            "brier_1x2",
            "log_loss_1x2",
            "forward_ranking_score",
            "forward_pairwise_accuracy",
        ):
            output.loc[mask, f"{metric}_delta_vs_production"] = (
                output.loc[mask, metric] - float(baseline[metric])
            )
    return output


def build_pooled_comparison(
    predictions: pd.DataFrame,
    fold_results: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, frame in predictions.groupby("model", sort=True):
        folds = fold_results.loc[fold_results["model"].eq(model)]
        rows.append(
            {
                "model": model,
                **evaluate_predictions(frame),
                "fold_brier_wins": int(
                    (folds["brier_1x2_delta_vs_production"] < 0.0).sum()
                ),
                "fold_log_loss_wins": int(
                    (folds["log_loss_1x2_delta_vs_production"] < 0.0).sum()
                ),
                "pooled_forward_ranking_score": float(
                    np.average(folds["forward_ranking_score"], weights=folds["matches"])
                ),
                "pooled_forward_pairwise_accuracy": float(
                    np.average(
                        folds["forward_pairwise_accuracy"], weights=folds["matches"]
                    )
                ),
                "max_abs_match_delta": float(frame["power_delta"].abs().max()),
                "minimum_winner_update": float(
                    frame.loc[frame["actual_class"].ne(1), "winner_elo_gain"].min()
                ),
                "maximum_zero_sum_error": float(frame["zero_sum_error"].max()),
            }
        )
    result = pd.DataFrame(rows)
    baseline = result.loc[result["model"].eq("GD_PRODUCTION")].iloc[0]
    for metric in (
        "brier_1x2",
        "log_loss_1x2",
        "pooled_forward_ranking_score",
        "pooled_forward_pairwise_accuracy",
    ):
        result[f"{metric}_delta_vs_production"] = (
            result[metric] - float(baseline[metric])
        )
    return result


def build_competition_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, competition), frame in predictions.groupby(
        ["model", "competition"], sort=True
    ):
        rows.append(
            {"model": model, "competition": competition, **evaluate_predictions(frame)}
        )
    result = pd.DataFrame(rows)
    baseline = result.loc[result["model"].eq("GD_PRODUCTION")].set_index(
        "competition"
    )
    result["brier_delta_vs_production"] = result.apply(
        lambda row: row["brier_1x2"]
        - baseline.loc[row["competition"], "brier_1x2"],
        axis=1,
    )
    result["log_loss_delta_vs_production"] = result.apply(
        lambda row: row["log_loss_1x2"]
        - baseline.loc[row["competition"], "log_loss_1x2"],
        axis=1,
    )
    return result


def build_uncertainty(
    predictions: pd.DataFrame,
    *,
    bootstrap_samples: int,
) -> pd.DataFrame:
    baseline = predictions.loc[
        predictions["model"].eq("GD_PRODUCTION")
    ].set_index("match_id")
    candidate = predictions.loc[
        predictions["model"].eq("NESTED_SELECTED_XG")
    ].set_index("match_id")
    if not candidate.index.equals(baseline.index):
        raise ValueError("Outer uncertainty requires paired predictions")
    rows: list[pd.DataFrame] = []
    for index, metric in enumerate(("brier_1x2", "log_loss_1x2")):
        paired = candidate[
            ["season", "home_team_id", "away_team_id", "kickoff_utc", "tie_id"]
        ].reset_index()
        paired["loss_difference"] = (
            candidate[metric].to_numpy(float) - baseline[metric].to_numpy(float)
        )
        uncertainty = dependency_robust_loss_difference_ci(
            paired,
            bootstrap_samples=bootstrap_samples,
            seed=20260806 + index * 10007,
        )
        uncertainty.insert(0, "metric", metric)
        rows.append(uncertainty)
    return pd.concat(rows, ignore_index=True)


def make_decision(
    selections: pd.DataFrame,
    fold_results: pd.DataFrame,
    pooled: pd.DataFrame,
    competition: pd.DataFrame,
    uncertainty: pd.DataFrame,
    deployment_candidate: XGPerformanceCandidate,
) -> tuple[str, dict[str, object]]:
    candidate = pooled.loc[pooled["model"].eq("NESTED_SELECTED_XG")].iloc[0]
    segments = competition.loc[competition["model"].eq("NESTED_SELECTED_XG")]
    ci = uncertainty.loc[uncertainty["method"].eq("conservative_envelope")]
    xg_folds = int((~selections["selected_is_production_baseline"]).sum())
    pooled_loss_improvement = bool(
        candidate["brier_1x2_delta_vs_production"] < 0.0
        and candidate["log_loss_1x2_delta_vs_production"] < 0.0
    )
    ranking_guardrail = bool(
        candidate["pooled_forward_ranking_score_delta_vs_production"] >= 0.0
        and candidate["pooled_forward_pairwise_accuracy_delta_vs_production"] >= 0.0
    )
    no_competition_regression = bool(
        len(segments) == 3
        and (segments["brier_delta_vs_production"] <= 0.0).all()
        and (segments["log_loss_delta_vs_production"] <= 0.0).all()
    )
    no_reliable_harm = bool(len(ci) == 2 and not (ci["ci_95_lower"] > 0.0).any())
    reliable_improvement = bool(len(ci) == 2 and (ci["ci_95_upper"] < 0.0).all())
    guardrails = {
        "xg_selected_outer_folds": xg_folds,
        "pooled_brier_delta": float(candidate["brier_1x2_delta_vs_production"]),
        "pooled_log_loss_delta": float(
            candidate["log_loss_1x2_delta_vs_production"]
        ),
        "brier_fold_wins": int(candidate["fold_brier_wins"]),
        "log_loss_fold_wins": int(candidate["fold_log_loss_wins"]),
        "pooled_forward_ranking_delta": float(
            candidate["pooled_forward_ranking_score_delta_vs_production"]
        ),
        "pooled_forward_pairwise_delta": float(
            candidate["pooled_forward_pairwise_accuracy_delta_vs_production"]
        ),
        "pooled_loss_improvement": pooled_loss_improvement,
        "ranking_guardrail": ranking_guardrail,
        "no_competition_regression": no_competition_regression,
        "no_cluster_reliable_harm": no_reliable_harm,
        "cluster_reliable_improvement": reliable_improvement,
        "winner_always_positive": bool(candidate["minimum_winner_update"] > 0.0),
        "zero_sum_preserved": bool(candidate["maximum_zero_sum_error"] <= 1e-9),
        "full_data_candidate_is_xg": not deployment_candidate.is_baseline,
    }
    passes = all(
        (
            xg_folds >= 3,
            pooled_loss_improvement,
            int(candidate["fold_brier_wins"]) >= 3,
            int(candidate["fold_log_loss_wins"]) >= 3,
            ranking_guardrail,
            no_competition_regression,
            no_reliable_harm,
            guardrails["winner_always_positive"],
            guardrails["zero_sum_preserved"],
            not deployment_candidate.is_baseline,
        )
    )
    return (
        "PROMISING_SHADOW_REQUIRES_PROSPECTIVE_CONFIRMATION"
        if passes
        else "NO_PROMOTION_KEEP_PRODUCTION"
    ), guardrails


def build_examples(predictions: pd.DataFrame) -> pd.DataFrame:
    values = predictions.loc[
        predictions["xg_performance_adjustment"].abs().gt(1e-15)
    ].copy()
    values["xg_adjustment_elo"] = (
        values["xg_performance_adjustment"] * 103.98098633392752
    )
    columns = [
        "match_id",
        "competition",
        "round",
        "kickoff_utc",
        "home_team_name",
        "away_team_name",
        "home_goals",
        "away_goals",
        "xg_home",
        "xg_away",
        "xg_difference",
        "xg_performance_signal",
        "goal_multiplier",
        "xg_adjustment_elo",
        "power_delta",
        "winner_elo_gain",
    ]
    return values.loc[:, columns].sort_values(
        "xg_adjustment_elo", key=lambda series: series.abs(), ascending=False
    ).reset_index(drop=True)


def build_report(
    selections: pd.DataFrame,
    fold_results: pd.DataFrame,
    pooled: pd.DataFrame,
    competition: pd.DataFrame,
    uncertainty: pd.DataFrame,
    deployment_candidate: XGPerformanceCandidate,
    deployment_selection: dict[str, object],
    decision: str,
    guardrails: dict[str, object],
) -> str:
    selection_view = selections[
        [
            "fold",
            "train_matches",
            "train_xg_matches",
            "test_matches",
            "selected_candidate",
            "selected_beta",
            "selected_xg_scale",
            "selected_minimum_winner_gain_ratio",
            "ranking_safe_candidates",
        ]
    ]
    fold_view = fold_results[
        [
            "fold",
            "model",
            "candidate_key",
            "matches",
            "brier_1x2",
            "brier_1x2_delta_vs_production",
            "log_loss_1x2",
            "log_loss_1x2_delta_vs_production",
            "forward_ranking_score_delta_vs_production",
            "forward_pairwise_accuracy_delta_vs_production",
        ]
    ]
    pooled_view = pooled[
        [
            "model",
            "matches",
            "brier_1x2",
            "brier_1x2_delta_vs_production",
            "log_loss_1x2",
            "log_loss_1x2_delta_vs_production",
            "fold_brier_wins",
            "fold_log_loss_wins",
            "pooled_forward_ranking_score_delta_vs_production",
            "pooled_forward_pairwise_accuracy_delta_vs_production",
            "minimum_winner_update",
        ]
    ]
    competition_view = competition[
        [
            "model",
            "competition",
            "matches",
            "brier_delta_vs_production",
            "log_loss_delta_vs_production",
        ]
    ]
    ci_view = uncertainty.loc[
        uncertainty["method"].eq("conservative_envelope"),
        ["metric", "mean_difference", "ci_95_lower", "ci_95_upper"],
    ]
    return "\n".join(
        [
            "# Çift Yönlü xG Performans Bonusu Walk-Forward Backtesti",
            "",
            "## Nihai Formül Adayı",
            "",
            "```text",
            "Delta_base = K * (S-E)",
            "Delta_GD = Delta_base * (M_GD-1)",
            "Q_xG = tanh((xG_home-xG_away)/xG_scale)",
            "Delta_xG = beta * abs(Delta_base) * Q_xG",
            "Delta_raw = Delta_base + Delta_GD + Delta_xG",
            "Delta_final = direction-preserving positive-winner clamp(Delta_raw)",
            "```",
            "",
            "Kazanan her zaman pozitif Elo alır. xG farkı sonucu destekliyorsa kazanç",
            "artar, ters yöndeyse azalır. Beraberlik/penaltı ve eksik xG'de bu katman",
            "sıfırdır. Gol farkı bonusu ayrı kalır.",
            "",
            "## Fold Seçimleri",
            "",
            markdown_table(selection_view),
            "",
            "## Unseen Fold Sonuçları",
            "",
            markdown_table(fold_view),
            "",
            "## Pooled Unseen Sonuç",
            "",
            markdown_table(pooled_view),
            "",
            "## Turnuva Segmentleri",
            "",
            markdown_table(competition_view),
            "",
            "## Cluster Güven Aralığı",
            "",
            markdown_table(ci_view),
            "",
            "## 2026/27 Shadow Adayı",
            "",
            f"`{deployment_candidate.key}`",
            "",
            markdown_table(pd.DataFrame([deployment_selection])),
            "",
            "## Karar",
            "",
            f"`{decision}`",
            "",
            markdown_table(pd.DataFrame([guardrails])),
        ]
    )


if __name__ == "__main__":
    main()
