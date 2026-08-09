from __future__ import annotations

import argparse
import copy
import json
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

from ao_elo.evaluation import schedule_adjusted_team_performance  # noqa: E402
from ao_elo.xg_live import XGPerformanceBonusConfig  # noqa: E402
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
    load_initial_ratings,
    read_production_contract,
    read_static_config,
    same_season_ranking,
)
from scripts.run_xg_performance_bonus_walk_forward_2025_26 import (  # noqa: E402
    CandidateReplay,
    build_uncertainty,
    outer_folds,
    prediction_slice,
    replay_candidate,
    validate_fold_contract,
)


OUTPUT_ROOT = ROOT / "output" / "goal_alpha_with_bounded_xg_backtest_2025_26"
BASELINE_KEY = "CURRENT_XG_GOAL_ALPHA_0.10"
BASELINE_MODEL = "CURRENT_BOUNDED_XG_ALPHA_0.10"
SELECTED_MODEL = "NESTED_SELECTED_GOAL_ALPHA"
GOAL_ALPHA_GRID = (0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25)
XG_RATIO = 0.30
XG_SCALE = 1.25
GOAL_TAU = 300.0
GOAL_CAP = 4


@dataclass(frozen=True, order=True)
class GoalAlphaCandidate:
    key: str
    goal_alpha: float

    @property
    def beta(self) -> float:
        return XG_RATIO

    @property
    def xg_scale(self) -> float:
        return XG_SCALE

    @property
    def minimum_winner_gain_ratio(self) -> float:
        return 1.0 - XG_RATIO

    @property
    def config(self) -> XGPerformanceBonusConfig:
        return XGPerformanceBonusConfig(
            XG_RATIO,
            XG_SCALE,
            1.0 - XG_RATIO,
        )

    @property
    def is_baseline(self) -> bool:
        return self.key == BASELINE_KEY

    def validate(self) -> None:
        self.config.validate()
        if self.goal_alpha not in GOAL_ALPHA_GRID:
            raise ValueError(f"Unsupported goal_alpha: {self.goal_alpha}")
        if self.is_baseline and self.goal_alpha != 0.10:
            raise ValueError("Baseline goal_alpha must be 0.10")


def goal_alpha_grid() -> tuple[GoalAlphaCandidate, ...]:
    candidates = tuple(
        GoalAlphaCandidate(
            BASELINE_KEY
            if alpha == 0.10
            else f"GOAL_ALPHA_{alpha:.3f}".rstrip("0"),
            alpha,
        )
        for alpha in GOAL_ALPHA_GRID
    )
    for candidate in candidates:
        candidate.validate()
    if len(candidates) != 7:
        raise ValueError(f"Expected 7 candidates, found {len(candidates)}")
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune goal_alpha with bounded xG and every other parameter fixed"
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
    candidates = goal_alpha_grid()
    validate_fold_contract(events, folds)
    cutoffs = tuple(
        sorted({fold.test_start for fold in folds} | {fold.test_end for fold in folds})
    )

    evaluations: dict[str, CandidateReplay] = {}
    print(
        f"Goal alpha isolation: {len(candidates)} candidates, "
        f"{len(folds)} folds, 387 unseen matches",
        flush=True,
    )
    for candidate in candidates:
        candidate_production = production_with_goal_alpha(
            production, candidate.goal_alpha
        )
        evaluations[candidate.key] = replay_candidate(
            events,
            initial_ratings,
            candidate_production,
            candidate,
            snapshot_times=cutoffs,
        )

    selection_rows: list[dict[str, object]] = []
    fold_result_rows: list[dict[str, object]] = []
    outer_frames: list[pd.DataFrame] = []
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
        selected, audit = select_brier_first_candidate(training_metrics, candidates)
        selection_rows.append(
            {
                "fold": fold.key,
                "train_matches": len(train_events),
                "train_xg_matches": int(train_events["xg_analysis_eligible"].sum()),
                "test_matches": len(test_events),
                "selected_candidate": selected.key,
                "selected_goal_alpha": selected.goal_alpha,
                **audit,
            }
        )
        baseline_predictions = prediction_slice(
            evaluations[BASELINE_KEY].predictions,
            fold.test_start,
            fold.test_end,
        ).assign(model=BASELINE_MODEL, fold=fold.key)
        selected_predictions = prediction_slice(
            evaluations[selected.key].predictions,
            fold.test_start,
            fold.test_end,
        ).assign(
            model=SELECTED_MODEL,
            fold=fold.key,
            selected_candidate=selected.key,
            selected_goal_alpha=selected.goal_alpha,
        )
        outer_frames.extend([baseline_predictions, selected_predictions])
        for model, candidate, frame in (
            (BASELINE_MODEL, candidates[0], baseline_predictions),
            (SELECTED_MODEL, selected, selected_predictions),
        ):
            ranking = same_season_ranking(
                evaluations[candidate.key].snapshots[fold.test_start],
                test_target,
                {SEASON},
            )
            pooled = ranking.loc[ranking["competition"].eq("ALL")].iloc[0]
            fold_result_rows.append(
                {
                    "fold": fold.key,
                    "model": model,
                    "candidate_key": candidate.key,
                    "goal_alpha": candidate.goal_alpha,
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
    outer_predictions = pd.concat(outer_frames, ignore_index=True)
    pooled_comparison = build_pooled_comparison(outer_predictions, fold_results)
    competition_summary = build_competition_summary(outer_predictions)
    uncertainty = build_paired_uncertainty(
        outer_predictions, bootstrap_samples=args.bootstrap_samples
    )
    fixed_metrics = build_fixed_outer_metrics(
        evaluations, candidates, events, folds
    )
    fixed_competition_metrics = build_all_fixed_competition_metrics(
        evaluations, candidates, folds
    )
    fixed_best_key = str(fixed_metrics.iloc[0]["candidate_key"])
    fixed_best_candidate = next(
        candidate for candidate in candidates if candidate.key == fixed_best_key
    )
    fixed_best = build_fixed_candidate_diagnostics(
        evaluations,
        fixed_best_candidate,
        folds,
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
    future_candidate, future_selection = select_brier_first_candidate(
        full_metrics, candidates
    )
    fixed_future = build_fixed_candidate_diagnostics(
        evaluations,
        future_candidate,
        folds,
        bootstrap_samples=args.bootstrap_samples,
    )
    decision, guardrails = make_decision(
        fold_selections,
        fold_results,
        pooled_comparison,
        competition_summary,
        uncertainty,
        future_candidate,
    )

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "candidate_key": candidate.key,
                "goal_alpha": candidate.goal_alpha,
                "xg_ratio": XG_RATIO,
                "xg_scale": XG_SCALE,
                "goal_tau": GOAL_TAU,
                "goal_cap": GOAL_CAP,
                "max_match_delta": None,
            }
            for candidate in candidates
        ]
    ).to_csv(output_root / "candidate_grid.csv", index=False)
    fold_selections.to_csv(output_root / "fold_selections.csv", index=False)
    fold_results.to_csv(output_root / "fold_results.csv", index=False)
    outer_predictions.to_csv(output_root / "outer_predictions.csv", index=False)
    pooled_comparison.to_csv(output_root / "pooled_comparison.csv", index=False)
    competition_summary.to_csv(output_root / "competition_summary.csv", index=False)
    uncertainty.to_csv(output_root / "dependency_uncertainty.csv", index=False)
    fixed_metrics.to_csv(output_root / "fixed_candidate_outer_metrics.csv", index=False)
    fixed_competition_metrics.to_csv(
        output_root / "fixed_candidate_competition_summary.csv", index=False
    )
    fixed_best["pooled"].to_csv(
        output_root / "fixed_oos_best_pooled_comparison.csv", index=False
    )
    fixed_best["competition"].to_csv(
        output_root / "fixed_oos_best_competition_summary.csv", index=False
    )
    fixed_best["uncertainty"].to_csv(
        output_root / "fixed_oos_best_dependency_uncertainty.csv", index=False
    )
    full_metrics.to_csv(output_root / "full_data_candidate_metrics.csv", index=False)
    fixed_future["pooled"].to_csv(
        output_root / "fixed_future_pooled_comparison.csv", index=False
    )
    fixed_future["competition"].to_csv(
        output_root / "fixed_future_competition_summary.csv", index=False
    )
    fixed_future["uncertainty"].to_csv(
        output_root / "fixed_future_dependency_uncertainty.csv", index=False
    )
    payload = {
        "analysis": "GOAL_ALPHA_WITH_BOUNDED_XG_WALK_FORWARD_2025_26",
        "selection_objective": "BRIER_FIRST_WITH_RANKING_GUARDRAIL",
        "isolated_parameter": "goal_alpha",
        "candidate_values": list(GOAL_ALPHA_GRID),
        "fixed_parameters": {
            "xg_ratio": XG_RATIO,
            "xg_scale": XG_SCALE,
            "goal_tau": GOAL_TAU,
            "goal_cap": GOAL_CAP,
            "max_match_delta": None,
        },
        "future_shadow_candidate": {
            "key": future_candidate.key,
            "goal_alpha": future_candidate.goal_alpha,
        },
        "retrospective_fixed_oos_best": {
            "key": fixed_best_candidate.key,
            "goal_alpha": fixed_best_candidate.goal_alpha,
            "selection_uses_unseen_results": True,
        },
        "future_selection": future_selection,
        "decision": decision,
        "guardrails": guardrails,
        "production_changed": False,
    }
    (output_root / "selected_goal_alpha_model.json").write_text(
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
            fixed_metrics,
            fixed_competition_metrics,
            fixed_best,
            fixed_best_candidate,
            fixed_future,
            future_candidate,
            future_selection,
            decision,
            guardrails,
        ),
        encoding="utf-8",
    )
    print(f"Future goal_alpha candidate: {future_candidate.goal_alpha:g}")
    print(f"Decision: {decision}")
    print(f"Report: {output_root / 'backtest_report.md'}")


def production_with_goal_alpha(
    production: dict[str, object], goal_alpha: float
) -> dict[str, object]:
    result = copy.deepcopy(production)
    goal_margin = result["goal_margin"]
    if not isinstance(goal_margin, dict):
        raise ValueError("production goal_margin must be an object")
    goal_margin["alpha"] = float(goal_alpha)
    goal_margin["tau"] = GOAL_TAU
    goal_margin["goal_difference_cap"] = GOAL_CAP
    return result


def build_candidate_metrics(
    evaluations: dict[str, CandidateReplay],
    candidates: tuple[GoalAlphaCandidate, ...],
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
                "goal_alpha": candidate.goal_alpha,
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
    for metric in ("brier_1x2", "log_loss_1x2", "ranking_score", "pairwise_accuracy"):
        result[f"{metric}_delta_vs_baseline"] = result[metric] - float(baseline[metric])
    return result


def select_brier_first_candidate(
    metrics: pd.DataFrame,
    candidates: tuple[GoalAlphaCandidate, ...],
) -> tuple[GoalAlphaCandidate, dict[str, object]]:
    baseline = metrics.loc[metrics["candidate_key"].eq(BASELINE_KEY)].iloc[0]
    values = metrics.copy()
    values["ranking_safe"] = (
        values["ranking_score"].ge(float(baseline["ranking_score"]))
        & values["pairwise_accuracy"].ge(float(baseline["pairwise_accuracy"]))
    )
    selected_row = values.loc[values["ranking_safe"]].sort_values(
        ["brier_1x2", "log_loss_1x2", "goal_alpha"],
        ascending=[True, True, True],
    ).iloc[0]
    selected = next(
        candidate
        for candidate in candidates
        if candidate.key == selected_row["candidate_key"]
    )
    return selected, {
        "selection_pool": "RANKING_SAFE",
        "ranking_safe_candidates": int(values["ranking_safe"].sum()),
        "selected_is_baseline": selected.is_baseline,
        "selected_training_brier_delta": float(
            selected_row["brier_1x2_delta_vs_baseline"]
        ),
        "selected_training_log_loss_delta": float(
            selected_row["log_loss_1x2_delta_vs_baseline"]
        ),
        "selected_training_ranking_delta": float(
            selected_row["ranking_score_delta_vs_baseline"]
        ),
        "selected_training_pairwise_delta": float(
            selected_row["pairwise_accuracy_delta_vs_baseline"]
        ),
        "future_results_used_for_selection": False,
    }


def add_fold_deltas(results: pd.DataFrame) -> pd.DataFrame:
    output = results.copy()
    for fold in output["fold"].unique():
        mask = output["fold"].eq(fold)
        baseline = output.loc[mask & output["model"].eq(BASELINE_MODEL)].iloc[0]
        for metric in (
            "brier_1x2",
            "log_loss_1x2",
            "forward_ranking_score",
            "forward_pairwise_accuracy",
        ):
            output.loc[mask, f"{metric}_delta_vs_baseline"] = (
                output.loc[mask, metric] - float(baseline[metric])
            )
    return output


def build_pooled_comparison(
    predictions: pd.DataFrame, fold_results: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, frame in predictions.groupby("model", sort=True):
        folds = fold_results.loc[fold_results["model"].eq(model)]
        rows.append(
            {
                "model": model,
                **evaluate_predictions(frame),
                "fold_brier_wins": int(
                    (folds["brier_1x2_delta_vs_baseline"] < 0.0).sum()
                ),
                "fold_log_loss_wins": int(
                    (folds["log_loss_1x2_delta_vs_baseline"] < 0.0).sum()
                ),
                "weighted_forward_ranking_score": float(
                    np.average(folds["forward_ranking_score"], weights=folds["matches"])
                ),
                "weighted_forward_pairwise_accuracy": float(
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
    baseline = result.loc[result["model"].eq(BASELINE_MODEL)].iloc[0]
    for metric in (
        "brier_1x2",
        "log_loss_1x2",
        "weighted_forward_ranking_score",
        "weighted_forward_pairwise_accuracy",
    ):
        result[f"{metric}_delta_vs_baseline"] = result[metric] - float(baseline[metric])
    return result


def build_competition_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"model": model, "competition": competition, **evaluate_predictions(frame)}
        for (model, competition), frame in predictions.groupby(
            ["model", "competition"], sort=True
        )
    ]
    result = pd.DataFrame(rows)
    baseline = result.loc[result["model"].eq(BASELINE_MODEL)].set_index("competition")
    for metric in ("brier_1x2", "log_loss_1x2", "accuracy_1x2"):
        result[f"{metric}_delta_vs_baseline"] = result.apply(
            lambda row: row[metric] - baseline.loc[row["competition"], metric], axis=1
        )
    return result


def build_paired_uncertainty(
    predictions: pd.DataFrame, *, bootstrap_samples: int
) -> pd.DataFrame:
    values = predictions.copy()
    values["model"] = values["model"].replace(
        {BASELINE_MODEL: "GD_PRODUCTION", SELECTED_MODEL: "NESTED_SELECTED_XG"}
    )
    return build_uncertainty(values, bootstrap_samples=bootstrap_samples)


def build_fixed_outer_metrics(
    evaluations: dict[str, CandidateReplay],
    candidates: tuple[GoalAlphaCandidate, ...],
    events: pd.DataFrame,
    folds: tuple,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    baseline_fold: dict[str, dict[str, float]] = {}
    candidate_fold: dict[str, list[dict[str, float]]] = {}
    for candidate in candidates:
        frames: list[pd.DataFrame] = []
        ranking_scores: list[float] = []
        pairwise_scores: list[float] = []
        weights: list[int] = []
        fold_metrics: list[dict[str, float]] = []
        for fold in folds:
            frame = prediction_slice(
                evaluations[candidate.key].predictions,
                fold.test_start,
                fold.test_end,
            )
            frames.append(frame)
            metrics = evaluate_predictions(frame)
            fold_metrics.append(
                {
                    "brier_1x2": float(metrics["brier_1x2"]),
                    "log_loss_1x2": float(metrics["log_loss_1x2"]),
                }
            )
            if candidate.is_baseline:
                baseline_fold[fold.key] = fold_metrics[-1]
            target_events = events.loc[
                events["kickoff_utc"].ge(fold.test_start)
                & events["kickoff_utc"].lt(fold.test_end)
            ]
            ranking = same_season_ranking(
                evaluations[candidate.key].snapshots[fold.test_start],
                schedule_adjusted_team_performance(target_events),
                {SEASON},
            )
            pooled = ranking.loc[ranking["competition"].eq("ALL")].iloc[0]
            ranking_scores.append(float(pooled["ranking_score"]))
            pairwise_scores.append(float(pooled["pairwise_accuracy"]))
            weights.append(len(frame))
        candidate_fold[candidate.key] = fold_metrics
        pooled_predictions = pd.concat(frames, ignore_index=True)
        rows.append(
            {
                "candidate_key": candidate.key,
                "goal_alpha": candidate.goal_alpha,
                **evaluate_predictions(pooled_predictions),
                "weighted_forward_ranking_score": float(
                    np.average(ranking_scores, weights=weights)
                ),
                "weighted_forward_pairwise_accuracy": float(
                    np.average(pairwise_scores, weights=weights)
                ),
                "max_abs_match_delta": float(
                    pooled_predictions["power_delta"].abs().max()
                ),
                "minimum_winner_update": float(
                    pooled_predictions.loc[
                        pooled_predictions["actual_class"].ne(1), "winner_elo_gain"
                    ].min()
                ),
                "maximum_zero_sum_error": float(
                    pooled_predictions["zero_sum_error"].max()
                ),
            }
        )
    result = pd.DataFrame(rows)
    baseline = result.loc[result["candidate_key"].eq(BASELINE_KEY)].iloc[0]
    for metric in (
        "brier_1x2",
        "log_loss_1x2",
        "weighted_forward_ranking_score",
        "weighted_forward_pairwise_accuracy",
    ):
        result[f"{metric}_delta_vs_baseline"] = result[metric] - float(baseline[metric])
    fold_keys = [fold.key for fold in folds]
    result["brier_fold_wins"] = [
        sum(
            values[index]["brier_1x2"] < baseline_fold[key]["brier_1x2"]
            for index, key in enumerate(fold_keys)
        )
        for candidate in candidates
        for values in [candidate_fold[candidate.key]]
    ]
    result["log_loss_fold_wins"] = [
        sum(
            values[index]["log_loss_1x2"] < baseline_fold[key]["log_loss_1x2"]
            for index, key in enumerate(fold_keys)
        )
        for candidate in candidates
        for values in [candidate_fold[candidate.key]]
    ]
    return result.sort_values(
        ["brier_1x2", "log_loss_1x2", "goal_alpha"]
    ).reset_index(drop=True)


def build_fixed_candidate_diagnostics(
    evaluations: dict[str, CandidateReplay],
    candidate: GoalAlphaCandidate,
    folds: tuple,
    *,
    bootstrap_samples: int,
) -> dict[str, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for fold in folds:
        frames.append(
            prediction_slice(
                evaluations[BASELINE_KEY].predictions,
                fold.test_start,
                fold.test_end,
            ).assign(model=BASELINE_MODEL, fold=fold.key)
        )
        frames.append(
            prediction_slice(
                evaluations[candidate.key].predictions,
                fold.test_start,
                fold.test_end,
            ).assign(model="FIXED_FUTURE_GOAL_ALPHA", fold=fold.key)
        )
    predictions = pd.concat(frames, ignore_index=True)
    pooled_rows = [
        {"model": model, **evaluate_predictions(frame)}
        for model, frame in predictions.groupby("model", sort=True)
    ]
    pooled = pd.DataFrame(pooled_rows)
    baseline = pooled.loc[pooled["model"].eq(BASELINE_MODEL)].iloc[0]
    for metric in ("brier_1x2", "log_loss_1x2", "accuracy_1x2", "multiclass_ece"):
        pooled[f"{metric}_delta_vs_baseline"] = pooled[metric] - float(baseline[metric])
    competition = build_competition_summary(
        predictions.assign(
            model=predictions["model"].replace(
                {"FIXED_FUTURE_GOAL_ALPHA": SELECTED_MODEL}
            )
        )
    )
    uncertainty_values = predictions.copy()
    uncertainty_values["model"] = uncertainty_values["model"].replace(
        {BASELINE_MODEL: "GD_PRODUCTION", "FIXED_FUTURE_GOAL_ALPHA": "NESTED_SELECTED_XG"}
    )
    uncertainty = build_uncertainty(
        uncertainty_values, bootstrap_samples=bootstrap_samples
    )
    return {"pooled": pooled, "competition": competition, "uncertainty": uncertainty}


def build_all_fixed_competition_metrics(
    evaluations: dict[str, CandidateReplay],
    candidates: tuple[GoalAlphaCandidate, ...],
    folds: tuple,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        predictions = pd.concat(
            [
                prediction_slice(
                    evaluations[candidate.key].predictions,
                    fold.test_start,
                    fold.test_end,
                )
                for fold in folds
            ],
            ignore_index=True,
        )
        for competition, frame in predictions.groupby("competition", sort=True):
            rows.append(
                {
                    "candidate_key": candidate.key,
                    "goal_alpha": candidate.goal_alpha,
                    "competition": competition,
                    **evaluate_predictions(frame),
                }
            )
    result = pd.DataFrame(rows)
    baseline = result.loc[result["candidate_key"].eq(BASELINE_KEY)].set_index(
        "competition"
    )
    for metric in ("brier_1x2", "log_loss_1x2", "accuracy_1x2", "multiclass_ece"):
        result[f"{metric}_delta_vs_baseline"] = result.apply(
            lambda row: row[metric] - baseline.loc[row["competition"], metric], axis=1
        )
    return result.sort_values(["goal_alpha", "competition"]).reset_index(drop=True)


def make_decision(
    selections: pd.DataFrame,
    fold_results: pd.DataFrame,
    pooled: pd.DataFrame,
    competition: pd.DataFrame,
    uncertainty: pd.DataFrame,
    future_candidate: GoalAlphaCandidate,
) -> tuple[str, dict[str, object]]:
    candidate = pooled.loc[pooled["model"].eq(SELECTED_MODEL)].iloc[0]
    folds = fold_results.loc[fold_results["model"].eq(SELECTED_MODEL)]
    segments = competition.loc[competition["model"].eq(SELECTED_MODEL)]
    conservative = uncertainty.loc[uncertainty["method"].eq("conservative_envelope")]
    guardrails = {
        "non_baseline_selected_folds": int((~selections["selected_is_baseline"]).sum()),
        "pooled_brier_delta": float(candidate["brier_1x2_delta_vs_baseline"]),
        "pooled_log_loss_delta": float(candidate["log_loss_1x2_delta_vs_baseline"]),
        "brier_fold_wins": int((folds["brier_1x2_delta_vs_baseline"] < 0.0).sum()),
        "log_loss_fold_wins": int(
            (folds["log_loss_1x2_delta_vs_baseline"] < 0.0).sum()
        ),
        "ranking_delta": float(candidate["weighted_forward_ranking_score_delta_vs_baseline"]),
        "pairwise_delta": float(
            candidate["weighted_forward_pairwise_accuracy_delta_vs_baseline"]
        ),
        "no_competition_brier_regression": bool(
            segments["brier_1x2_delta_vs_baseline"].le(0.0).all()
        ),
        "no_competition_log_loss_regression": bool(
            segments["log_loss_1x2_delta_vs_baseline"].le(0.0).all()
        ),
        "no_cluster_reliable_harm": bool((~conservative["reliable_harm"]).all()),
        "winner_positive": float(candidate["minimum_winner_update"]) > 0.0,
        "zero_sum_preserved": float(candidate["maximum_zero_sum_error"]) <= 1e-9,
        "future_candidate_is_non_baseline": not future_candidate.is_baseline,
        "max_match_delta_cap_used": False,
    }
    passes = (
        guardrails["non_baseline_selected_folds"] >= 3
        and guardrails["pooled_brier_delta"] < 0.0
        and guardrails["pooled_log_loss_delta"] <= 0.0
        and guardrails["brier_fold_wins"] >= 3
        and guardrails["ranking_delta"] >= 0.0
        and guardrails["pairwise_delta"] >= 0.0
        and guardrails["no_competition_brier_regression"]
        and guardrails["no_cluster_reliable_harm"]
        and guardrails["winner_positive"]
        and guardrails["zero_sum_preserved"]
        and guardrails["future_candidate_is_non_baseline"]
    )
    return (
        "PROMISING_GOAL_ALPHA_SHADOW"
        if passes
        else "KEEP_GOAL_ALPHA_0.10"
    ), guardrails


def build_report(
    selections: pd.DataFrame,
    fold_results: pd.DataFrame,
    pooled: pd.DataFrame,
    competition: pd.DataFrame,
    uncertainty: pd.DataFrame,
    fixed_metrics: pd.DataFrame,
    fixed_competition_metrics: pd.DataFrame,
    fixed_best: dict[str, pd.DataFrame],
    fixed_best_candidate: GoalAlphaCandidate,
    fixed_future: dict[str, pd.DataFrame],
    future_candidate: GoalAlphaCandidate,
    future_selection: dict[str, object],
    decision: str,
    guardrails: dict[str, object],
) -> str:
    selection_view = selections[
        [
            "fold",
            "train_matches",
            "train_xg_matches",
            "test_matches",
            "selected_goal_alpha",
            "ranking_safe_candidates",
        ]
    ]
    fold_view = fold_results[
        [
            "fold",
            "model",
            "goal_alpha",
            "matches",
            "brier_1x2",
            "brier_1x2_delta_vs_baseline",
            "log_loss_1x2_delta_vs_baseline",
            "forward_ranking_score_delta_vs_baseline",
            "max_abs_match_delta",
        ]
    ]
    pooled_view = pooled[
        [
            "model",
            "matches",
            "brier_1x2",
            "brier_1x2_delta_vs_baseline",
            "log_loss_1x2_delta_vs_baseline",
            "fold_brier_wins",
            "fold_log_loss_wins",
            "weighted_forward_ranking_score_delta_vs_baseline",
            "weighted_forward_pairwise_accuracy_delta_vs_baseline",
            "max_abs_match_delta",
        ]
    ]
    competition_view = competition[
        [
            "model",
            "competition",
            "matches",
            "brier_1x2_delta_vs_baseline",
            "log_loss_1x2_delta_vs_baseline",
        ]
    ]
    uncertainty_view = uncertainty.loc[
        uncertainty["method"].eq("conservative_envelope"),
        ["metric", "mean_difference", "ci_95_lower", "ci_95_upper"],
    ]
    fixed_view = fixed_metrics[
        [
            "goal_alpha",
            "brier_1x2",
            "brier_1x2_delta_vs_baseline",
            "log_loss_1x2_delta_vs_baseline",
            "brier_fold_wins",
            "weighted_forward_ranking_score_delta_vs_baseline",
            "max_abs_match_delta",
        ]
    ]
    fixed_competition_view = fixed_competition_metrics[
        [
            "goal_alpha",
            "competition",
            "brier_1x2_delta_vs_baseline",
            "log_loss_1x2_delta_vs_baseline",
            "accuracy_1x2_delta_vs_baseline",
            "multiclass_ece_delta_vs_baseline",
        ]
    ]
    future_pooled_view = fixed_future["pooled"]
    future_competition_view = fixed_future["competition"][
        [
            "model",
            "competition",
            "matches",
            "brier_1x2_delta_vs_baseline",
            "log_loss_1x2_delta_vs_baseline",
        ]
    ]
    future_uncertainty_view = fixed_future["uncertainty"].loc[
        fixed_future["uncertainty"]["method"].eq("conservative_envelope"),
        ["metric", "mean_difference", "ci_95_lower", "ci_95_upper"],
    ]
    fixed_best_competition_view = fixed_best["competition"][
        [
            "model",
            "competition",
            "matches",
            "brier_1x2_delta_vs_baseline",
            "log_loss_1x2_delta_vs_baseline",
        ]
    ]
    fixed_best_uncertainty_view = fixed_best["uncertainty"].loc[
        fixed_best["uncertainty"]["method"].eq("conservative_envelope"),
        ["metric", "mean_difference", "ci_95_lower", "ci_95_upper"],
    ]
    return f"""# Goal Alpha Izole Walk-Forward Backtesti

```text
Test edilen goal_alpha: 0.10 / 0.125 / 0.15 / 0.175 / 0.20 / 0.225 / 0.25
Sabit xG ratio: 0.30
Sabit xG scale: 1.25
Sabit goal_tau: 300
Sabit goal_cap: 4
Mac hareket tavani: KAPALI
```

## Fold Secimleri

{markdown_table(selection_view)}

## Unseen Fold Sonuclari

{markdown_table(fold_view)}

## Pooled Unseen Sonuc

{markdown_table(pooled_view)}

## Turnuva Segmentleri

{markdown_table(competition_view)}

## Cluster Guven Araligi

{markdown_table(uncertainty_view)}

## Sabit Adaylarin 387 Mac Diagnostigi

{markdown_table(fixed_view)}

### Sabit Adaylarin Turnuva Diagnostigi

{markdown_table(fixed_competition_view)}

### Retrospective En Iyi Sabit Aday

`goal_alpha={fixed_best_candidate.goal_alpha:g}`. Bu secim unseen sonuclara
baktigi icin production secim kaniti degil, yalniz katsayi davranisi
diagnostigidir.

{markdown_table(fixed_best_competition_view)}

{markdown_table(fixed_best_uncertainty_view)}

## 2026/27 Icin Tam Veri Secimi

`goal_alpha={future_candidate.goal_alpha:g}`

{markdown_table(pd.DataFrame([future_selection]))}

{markdown_table(future_pooled_view)}

{markdown_table(future_competition_view)}

{markdown_table(future_uncertainty_view)}

## Karar

`{decision}`

{markdown_table(pd.DataFrame([guardrails]))}
"""


if __name__ == "__main__":
    main()
