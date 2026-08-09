from __future__ import annotations

import argparse
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

from ao_elo.evaluation import (  # noqa: E402
    schedule_adjusted_team_performance,
)
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
    BASELINE_KEY,
    CandidateReplay,
    add_fold_deltas,
    build_competition_summary,
    build_pooled_comparison,
    build_uncertainty,
    outer_folds,
    prediction_slice,
    replay_candidate,
    validate_fold_contract,
)


OUTPUT_ROOT = ROOT / "output" / "bounded_xg_adjustment_walk_forward_2025_26"
RATIO_GRID = (0.15, 0.20, 0.25, 0.30)
XG_SCALE_GRID = (0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00)


@dataclass(frozen=True, order=True)
class BoundedXGCandidate:
    key: str
    max_xg_ratio: float
    xg_scale: float

    @property
    def beta(self) -> float:
        return self.max_xg_ratio

    @property
    def minimum_winner_gain_ratio(self) -> float:
        return 1.0 if self.is_baseline else 1.0 - self.max_xg_ratio

    @property
    def is_baseline(self) -> bool:
        return self.max_xg_ratio == 0.0

    @property
    def config(self) -> XGPerformanceBonusConfig | None:
        if self.is_baseline:
            return None
        return XGPerformanceBonusConfig(
            beta=self.max_xg_ratio,
            xg_scale=self.xg_scale,
            minimum_winner_gain_ratio=self.minimum_winner_gain_ratio,
        )

    def validate(self) -> None:
        if self.is_baseline:
            if self.key != BASELINE_KEY:
                raise ValueError("The zero-ratio candidate must be production baseline")
            return
        if self.config is None:
            raise ValueError("Bounded xG candidate requires a config")
        self.config.validate()
        if not 0.0 < self.max_xg_ratio < 1.0:
            raise ValueError("max_xg_ratio must be in (0, 1)")


def bounded_candidate_grid() -> tuple[BoundedXGCandidate, ...]:
    candidates = [BoundedXGCandidate(BASELINE_KEY, 0.0, 1.0)]
    candidates.extend(
        BoundedXGCandidate(
            key=f"BOUNDED_XG_ratio{ratio:g}_scale{xg_scale:g}",
            max_xg_ratio=ratio,
            xg_scale=xg_scale,
        )
        for ratio in RATIO_GRID
        for xg_scale in XG_SCALE_GRID
    )
    result = tuple(candidates)
    for candidate in result:
        candidate.validate()
    if len(result) != 29:
        raise ValueError(f"Expected 29 candidates, found {len(result)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Brier-first walk-forward test for a bounded xG Elo adjustment"
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
    candidates = bounded_candidate_grid()
    validate_fold_contract(events, folds)
    cutoffs = tuple(
        sorted({fold.test_start for fold in folds} | {fold.test_end for fold in folds})
    )

    print(
        f"Bounded xG adjustment: {len(candidates)} candidates, "
        f"{len(folds)} folds, 387 unseen matches",
        flush=True,
    )
    evaluations: dict[str, CandidateReplay] = {
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
    result_rows: list[dict[str, object]] = []
    outer_frames: list[pd.DataFrame] = []
    for fold in folds:
        train_events = events.loc[events["kickoff_utc"].lt(fold.test_start)]
        test_events = events.loc[
            events["kickoff_utc"].ge(fold.test_start)
            & events["kickoff_utc"].lt(fold.test_end)
        ]
        train_target = schedule_adjusted_team_performance(train_events)
        test_target = schedule_adjusted_team_performance(test_events)
        training_metrics = build_bounded_candidate_metrics(
            evaluations,
            candidates,
            train_target,
            prediction_start=None,
            prediction_end=fold.test_start,
            rating_time=fold.test_start,
        )
        selected, selection = select_brier_first_candidate(
            training_metrics, candidates
        )
        selection_rows.append(
            {
                "fold": fold.key,
                "train_matches": len(train_events),
                "train_xg_matches": int(train_events["xg_analysis_eligible"].sum()),
                "test_matches": len(test_events),
                "selected_candidate": selected.key,
                "selected_max_xg_ratio": selected.max_xg_ratio,
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
            model="NESTED_SELECTED_BOUNDED_XG",
            fold=fold.key,
            selected_candidate=selected.key,
        )
        outer_frames.extend([baseline_predictions, selected_predictions])
        for model, candidate, frame in (
            ("GD_PRODUCTION", candidates[0], baseline_predictions),
            ("NESTED_SELECTED_BOUNDED_XG", selected, selected_predictions),
        ):
            ranking = same_season_ranking(
                evaluations[candidate.key].snapshots[fold.test_start],
                test_target,
                {SEASON},
            )
            pooled_ranking = ranking.loc[ranking["competition"].eq("ALL")].iloc[0]
            result_rows.append(
                {
                    "fold": fold.key,
                    "model": model,
                    "candidate_key": candidate.key,
                    **evaluate_predictions(frame),
                    "forward_ranking_score": float(pooled_ranking["ranking_score"]),
                    "forward_pairwise_accuracy": float(
                        pooled_ranking["pairwise_accuracy"]
                    ),
                    "max_abs_match_delta": float(frame["power_delta"].abs().max()),
                    "minimum_winner_update": float(
                        frame.loc[frame["actual_class"].ne(1), "winner_elo_gain"].min()
                    ),
                    "maximum_zero_sum_error": float(frame["zero_sum_error"].max()),
                }
            )

    fold_selections = pd.DataFrame(selection_rows)
    fold_results = add_fold_deltas(pd.DataFrame(result_rows))
    outer_predictions = pd.concat(outer_frames, ignore_index=True)
    pooled_comparison = build_pooled_comparison(outer_predictions, fold_results)
    competition_summary = build_competition_summary(outer_predictions)
    uncertainty_input = outer_predictions.copy()
    uncertainty_input["model"] = uncertainty_input["model"].replace(
        {"NESTED_SELECTED_BOUNDED_XG": "NESTED_SELECTED_XG"}
    )
    uncertainty = build_uncertainty(
        uncertainty_input, bootstrap_samples=args.bootstrap_samples
    )
    fixed_candidate_metrics = build_fixed_outer_metrics(
        evaluations, candidates, events, folds
    )

    full_target = schedule_adjusted_team_performance(events)
    full_metrics = build_bounded_candidate_metrics(
        evaluations,
        candidates,
        full_target,
        prediction_start=None,
        prediction_end=pd.Timestamp("2026-06-01", tz="UTC"),
        rating_time=pd.Timestamp("2026-06-01", tz="UTC"),
    )
    shadow_candidate, shadow_selection = select_brier_first_candidate(
        full_metrics, candidates
    )
    (
        fixed_shadow_pooled,
        fixed_shadow_competition,
        fixed_shadow_uncertainty,
    ) = build_fixed_shadow_diagnostics(
        evaluations,
        shadow_candidate,
        folds,
        bootstrap_samples=args.bootstrap_samples,
    )
    decision, guardrails = make_bounded_decision(
        fold_selections,
        fold_results,
        pooled_comparison,
        competition_summary,
        uncertainty,
        shadow_candidate,
    )

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "candidate_key": candidate.key,
                "max_xg_ratio": candidate.max_xg_ratio,
                "xg_scale": candidate.xg_scale,
                "minimum_winner_gain_ratio": candidate.minimum_winner_gain_ratio,
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
    fixed_candidate_metrics.to_csv(
        output_root / "fixed_candidate_outer_metrics.csv", index=False
    )
    fixed_shadow_pooled.to_csv(
        output_root / "fixed_shadow_pooled_comparison.csv", index=False
    )
    fixed_shadow_competition.to_csv(
        output_root / "fixed_shadow_competition_summary.csv", index=False
    )
    fixed_shadow_uncertainty.to_csv(
        output_root / "fixed_shadow_dependency_uncertainty.csv", index=False
    )
    full_metrics.to_csv(output_root / "full_data_candidate_metrics.csv", index=False)

    payload = {
        "analysis": "BOUNDED_XG_ADJUSTMENT_WALK_FORWARD_2025_26",
        "formula": {
            "classic": "Delta_base=K*(S-E)",
            "goal_margin": "Delta_GD_bonus=Delta_base*(M_GD-1)",
            "xg_signal": "Q_xG=tanh((xG_home-xG_away)/xg_scale)",
            "xg_adjustment": (
                "Delta_xG=max_xg_ratio*abs(Delta_base)*Q_xG"
            ),
            "final": "Delta=Delta_base+Delta_GD_bonus+Delta_xG",
            "winner_bound": (
                "winner gain is at least (1-max_xg_ratio)*classic gain"
            ),
        },
        "selection_objective": "BRIER_FIRST_WITH_RANKING_GUARDRAIL",
        "future_shadow_candidate": {
            "key": shadow_candidate.key,
            "max_xg_ratio": shadow_candidate.max_xg_ratio,
            "xg_scale": shadow_candidate.xg_scale,
            "minimum_winner_gain_ratio": shadow_candidate.minimum_winner_gain_ratio,
        },
        "shadow_selection": shadow_selection,
        "decision": decision,
        "guardrails": guardrails,
        "production_changed": False,
        "prospective_confirmation_required": True,
    }
    (output_root / "selected_bounded_xg_model.json").write_text(
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
            fixed_candidate_metrics,
            fixed_shadow_pooled,
            fixed_shadow_competition,
            fixed_shadow_uncertainty,
            shadow_candidate,
            shadow_selection,
            decision,
            guardrails,
        ),
        encoding="utf-8",
    )
    print(f"Future shadow candidate: {shadow_candidate.key}")
    print(f"Decision: {decision}")
    print(f"Report: {output_root / 'backtest_report.md'}")


def build_bounded_candidate_metrics(
    evaluations: dict[str, CandidateReplay],
    candidates: tuple[BoundedXGCandidate, ...],
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
                "max_xg_ratio": candidate.max_xg_ratio,
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
    for metric in ("brier_1x2", "log_loss_1x2", "ranking_score", "pairwise_accuracy"):
        result[f"{metric}_delta_vs_production"] = (
            result[metric] - float(baseline[metric])
        )
    return result


def select_brier_first_candidate(
    metrics: pd.DataFrame,
    candidates: tuple[BoundedXGCandidate, ...],
) -> tuple[BoundedXGCandidate, dict[str, object]]:
    baseline = metrics.loc[metrics["candidate_key"].eq(BASELINE_KEY)].iloc[0]
    values = metrics.copy()
    values["ranking_safe"] = (
        values["ranking_score"].ge(float(baseline["ranking_score"]))
        & values["pairwise_accuracy"].ge(float(baseline["pairwise_accuracy"]))
    )
    pool = values.loc[values["ranking_safe"]].copy()
    selected_row = pool.sort_values(
        ["brier_1x2", "log_loss_1x2", "max_xg_ratio", "xg_scale"],
        ascending=[True, True, True, False],
    ).iloc[0]
    selected = next(
        candidate
        for candidate in candidates
        if candidate.key == selected_row["candidate_key"]
    )
    return selected, {
        "selection_pool": "RANKING_SAFE",
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


def build_fixed_outer_metrics(
    evaluations: dict[str, CandidateReplay],
    candidates: tuple[BoundedXGCandidate, ...],
    events: pd.DataFrame,
    folds: tuple,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    baseline_fold_brier: dict[str, float] = {}
    baseline_fold_log: dict[str, float] = {}
    candidate_fold_metrics: dict[str, list[dict[str, float]]] = {}
    for candidate in candidates:
        frames: list[pd.DataFrame] = []
        ranking_values: list[float] = []
        pairwise_values: list[float] = []
        weights: list[int] = []
        fold_values: list[dict[str, float]] = []
        for fold in folds:
            frame = prediction_slice(
                evaluations[candidate.key].predictions,
                fold.test_start,
                fold.test_end,
            )
            frames.append(frame)
            metrics = evaluate_predictions(frame)
            fold_values.append(
                {
                    "brier_1x2": float(metrics["brier_1x2"]),
                    "log_loss_1x2": float(metrics["log_loss_1x2"]),
                }
            )
            if candidate.is_baseline:
                baseline_fold_brier[fold.key] = float(metrics["brier_1x2"])
                baseline_fold_log[fold.key] = float(metrics["log_loss_1x2"])
            target_events = events.loc[
                events["kickoff_utc"].ge(fold.test_start)
                & events["kickoff_utc"].lt(fold.test_end)
            ]
            target = schedule_adjusted_team_performance(target_events)
            ranking = same_season_ranking(
                evaluations[candidate.key].snapshots[fold.test_start],
                target,
                {SEASON},
            )
            pooled = ranking.loc[ranking["competition"].eq("ALL")].iloc[0]
            ranking_values.append(float(pooled["ranking_score"]))
            pairwise_values.append(float(pooled["pairwise_accuracy"]))
            weights.append(len(frame))
        candidate_fold_metrics[candidate.key] = fold_values
        pooled_predictions = pd.concat(frames, ignore_index=True)
        rows.append(
            {
                "candidate_key": candidate.key,
                "max_xg_ratio": candidate.max_xg_ratio,
                "xg_scale": candidate.xg_scale,
                **evaluate_predictions(pooled_predictions),
                "weighted_forward_ranking_score": float(
                    np.average(ranking_values, weights=weights)
                ),
                "weighted_forward_pairwise_accuracy": float(
                    np.average(pairwise_values, weights=weights)
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
        result[f"{metric}_delta_vs_production"] = (
            result[metric] - float(baseline[metric])
        )
    fold_keys = [fold.key for fold in folds]
    result["brier_fold_wins"] = [
        sum(
            values[index]["brier_1x2"] < baseline_fold_brier[key]
            for index, key in enumerate(fold_keys)
        )
        for candidate, values in (
            (candidate, candidate_fold_metrics[candidate.key])
            for candidate in candidates
        )
    ]
    result["log_loss_fold_wins"] = [
        sum(
            values[index]["log_loss_1x2"] < baseline_fold_log[key]
            for index, key in enumerate(fold_keys)
        )
        for candidate, values in (
            (candidate, candidate_fold_metrics[candidate.key])
            for candidate in candidates
        )
    ]
    return result.sort_values(
        ["brier_1x2", "log_loss_1x2", "max_xg_ratio", "xg_scale"]
    ).reset_index(drop=True)


def build_fixed_shadow_diagnostics(
    evaluations: dict[str, CandidateReplay],
    candidate: BoundedXGCandidate,
    folds: tuple,
    *,
    bootstrap_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for fold in folds:
        frames.append(
            prediction_slice(
                evaluations[BASELINE_KEY].predictions,
                fold.test_start,
                fold.test_end,
            ).assign(model="GD_PRODUCTION", fold=fold.key)
        )
        frames.append(
            prediction_slice(
                evaluations[candidate.key].predictions,
                fold.test_start,
                fold.test_end,
            ).assign(model="FIXED_SHADOW_BOUNDED_XG", fold=fold.key)
        )
    predictions = pd.concat(frames, ignore_index=True)
    pooled_rows: list[dict[str, object]] = []
    for model, frame in predictions.groupby("model", sort=True):
        pooled_rows.append(
            {
                "model": model,
                **evaluate_predictions(frame),
                "max_abs_match_delta": float(frame["power_delta"].abs().max()),
                "minimum_winner_update": float(
                    frame.loc[frame["actual_class"].ne(1), "winner_elo_gain"].min()
                ),
                "maximum_zero_sum_error": float(frame["zero_sum_error"].max()),
            }
        )
    pooled = pd.DataFrame(pooled_rows)
    baseline = pooled.loc[pooled["model"].eq("GD_PRODUCTION")].iloc[0]
    for metric in ("brier_1x2", "log_loss_1x2", "accuracy_1x2", "multiclass_ece"):
        pooled[f"{metric}_delta_vs_production"] = (
            pooled[metric] - float(baseline[metric])
        )
    competition = build_competition_summary(predictions)
    uncertainty_input = predictions.copy()
    uncertainty_input["model"] = uncertainty_input["model"].replace(
        {"FIXED_SHADOW_BOUNDED_XG": "NESTED_SELECTED_XG"}
    )
    uncertainty = build_uncertainty(
        uncertainty_input,
        bootstrap_samples=bootstrap_samples,
    )
    return pooled, competition, uncertainty


def make_bounded_decision(
    selections: pd.DataFrame,
    fold_results: pd.DataFrame,
    pooled: pd.DataFrame,
    competition: pd.DataFrame,
    uncertainty: pd.DataFrame,
    shadow_candidate: BoundedXGCandidate,
) -> tuple[str, dict[str, object]]:
    candidate = pooled.loc[
        pooled["model"].eq("NESTED_SELECTED_BOUNDED_XG")
    ].iloc[0]
    segments = competition.loc[
        competition["model"].eq("NESTED_SELECTED_BOUNDED_XG")
    ]
    conservative = uncertainty.loc[
        uncertainty["method"].eq("conservative_envelope")
    ]
    xg_fold_results = fold_results.loc[
        fold_results["model"].eq("NESTED_SELECTED_BOUNDED_XG")
    ]
    guardrails = {
        "xg_selected_outer_folds": int(
            (~selections["selected_is_production_baseline"]).sum()
        ),
        "pooled_brier_delta": float(candidate["brier_1x2_delta_vs_production"]),
        "pooled_log_loss_delta": float(
            candidate["log_loss_1x2_delta_vs_production"]
        ),
        "brier_fold_wins": int(
            (xg_fold_results["brier_1x2_delta_vs_production"] < 0.0).sum()
        ),
        "log_loss_fold_wins": int(
            (xg_fold_results["log_loss_1x2_delta_vs_production"] < 0.0).sum()
        ),
        "ranking_delta": float(
            candidate["pooled_forward_ranking_score_delta_vs_production"]
        ),
        "pairwise_delta": float(
            candidate["pooled_forward_pairwise_accuracy_delta_vs_production"]
        ),
        "no_competition_brier_regression": bool(
            segments["brier_delta_vs_production"].le(0.0).all()
        ),
        "no_competition_log_loss_regression": bool(
            segments["log_loss_delta_vs_production"].le(0.0).all()
        ),
        "no_cluster_reliable_harm": bool(
            (~conservative["reliable_harm"]).all()
        ),
        "cluster_reliable_brier_improvement": bool(
            conservative.loc[
                conservative["metric"].eq("brier_1x2"), "reliable_improvement"
            ].all()
        ),
        "winner_minimum_ratio_respected": bool(
            float(candidate["minimum_winner_update"]) > 0.0
        ),
        "zero_sum_preserved": bool(
            float(candidate["maximum_zero_sum_error"]) <= 1e-9
        ),
        "full_data_candidate_is_xg": not shadow_candidate.is_baseline,
    }
    passes = (
        guardrails["xg_selected_outer_folds"] >= 3
        and guardrails["pooled_brier_delta"] < 0.0
        and guardrails["pooled_log_loss_delta"] <= 0.0
        and guardrails["brier_fold_wins"] >= 3
        and guardrails["ranking_delta"] >= 0.0
        and guardrails["pairwise_delta"] >= 0.0
        and guardrails["no_competition_brier_regression"]
        and guardrails["no_cluster_reliable_harm"]
        and guardrails["winner_minimum_ratio_respected"]
        and guardrails["zero_sum_preserved"]
        and guardrails["full_data_candidate_is_xg"]
    )
    return (
        "PROMISING_SHADOW_REQUIRES_PROSPECTIVE_CONFIRMATION"
        if passes
        else "NO_PROMOTION_KEEP_PRODUCTION"
    ), guardrails


def build_report(
    selections: pd.DataFrame,
    fold_results: pd.DataFrame,
    pooled: pd.DataFrame,
    competition: pd.DataFrame,
    uncertainty: pd.DataFrame,
    fixed_metrics: pd.DataFrame,
    fixed_shadow_pooled: pd.DataFrame,
    fixed_shadow_competition: pd.DataFrame,
    fixed_shadow_uncertainty: pd.DataFrame,
    shadow_candidate: BoundedXGCandidate,
    shadow_selection: dict[str, object],
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
            "selected_max_xg_ratio",
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
    uncertainty_view = uncertainty.loc[
        uncertainty["method"].eq("conservative_envelope"),
        ["metric", "mean_difference", "ci_95_lower", "ci_95_upper"],
    ]
    fixed_view = fixed_metrics.head(10)[
        [
            "candidate_key",
            "max_xg_ratio",
            "xg_scale",
            "brier_1x2",
            "brier_1x2_delta_vs_production",
            "log_loss_1x2_delta_vs_production",
            "brier_fold_wins",
            "weighted_forward_ranking_score_delta_vs_production",
            "minimum_winner_update",
        ]
    ]
    guardrail_view = pd.DataFrame([guardrails])
    shadow_view = pd.DataFrame([shadow_selection])
    fixed_shadow_pooled_view = fixed_shadow_pooled[
        [
            "model",
            "matches",
            "brier_1x2",
            "brier_1x2_delta_vs_production",
            "log_loss_1x2",
            "log_loss_1x2_delta_vs_production",
            "accuracy_1x2",
            "minimum_winner_update",
        ]
    ]
    fixed_shadow_competition_view = fixed_shadow_competition[
        [
            "model",
            "competition",
            "matches",
            "brier_delta_vs_production",
            "log_loss_delta_vs_production",
        ]
    ]
    fixed_shadow_uncertainty_view = fixed_shadow_uncertainty.loc[
        fixed_shadow_uncertainty["method"].eq("conservative_envelope"),
        ["metric", "mean_difference", "ci_95_lower", "ci_95_upper"],
    ]
    return f"""# Kontrollu xG Duzeltmesi Brier-First Walk-Forward Backtesti

## Formul

```text
Delta_base = K * (S-E)
Delta_GD_bonus = Delta_base * (M_GD-1)
Q_xG = tanh((xG_home-xG_away)/xG_scale)
Delta_xG = max_xg_ratio * abs(Delta_base) * Q_xG
Delta_final = Delta_base + Delta_GD_bonus + Delta_xG
```

`max_xg_ratio` en fazla `%15-%30` araligindadir. Bu nedenle kazananin mac
sonucu Elo'su xG tarafindan neredeyse sifirlanamaz. Gol farki bonusu ayri kalir.

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

## Sabit Aday Retrospective Diagnostigi

Bu tablo katsayi davranisini anlamak icindir; unseen sonuclara bakarak yapilan
post-hoc secim production kaniti sayilmaz.

{markdown_table(fixed_view)}

## 2026/27 Shadow Adayi

`{shadow_candidate.key}`

{markdown_table(shadow_view)}

### Sabit Shadow Adayinin Unseen Diagnostigi

{markdown_table(fixed_shadow_pooled_view)}

{markdown_table(fixed_shadow_competition_view)}

{markdown_table(fixed_shadow_uncertainty_view)}

## Karar

`{decision}`

{markdown_table(guardrail_view)}
"""


if __name__ == "__main__":
    main()
