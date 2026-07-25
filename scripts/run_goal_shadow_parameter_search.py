from __future__ import annotations

import argparse
import json
import math
import sys
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
from scripts.run_controlled_goal_progression_backtest import (  # noqa: E402
    DYNAMIC_ROOT,
    EVALUATION_ROOT,
    EVENTS_PATH,
    PRODUCTION_MODEL_PATH,
    STATIC_DATA_ROOT,
    BacktestEvaluation,
    ControlledSeasonData,
    LayerCandidate,
    candidate_metric_row,
    evaluate_candidate_set,
    evaluate_predictions,
    evaluate_sequence,
    prepare_controlled_data,
    select_candidate,
)
from scripts.run_dynamic_core_calibration import expanding_folds  # noqa: E402
from scripts.run_final_robustness import (  # noqa: E402
    core_for_fold,
    draw_map_for_fold,
    read_final_production_contract,
    validate_fold_inputs,
)
from scripts.run_v2_achievement_reserve_calibration import (  # noqa: E402
    load_reserve_data,
)
from scripts.run_v2_evaluation_upgrade import read_events  # noqa: E402


OUTPUT_ROOT = ROOT / "output" / "goal_shadow_parameter_search_2018_2026"
EXTENDED_ALPHAS = (
    0.000,
    0.025,
    0.050,
    0.075,
    0.100,
    0.125,
    0.150,
    0.175,
    0.200,
    0.225,
    0.250,
    0.275,
    0.300,
)
EXTENDED_TAUS = (100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 650.0, 800.0)
PRE_REGISTERED_ARMS = (
    ("BASE", 0.0, 300.0),
    ("PRE_SPECIFIED", 0.10, 300.0),
    ("PRIOR_GRID_BEST", 0.20, 400.0),
)
RANK_TOLERANCE = 1e-9
PRACTICAL_SPEARMAN_TOLERANCE = 0.005
PRACTICAL_PAIRWISE_TOLERANCE = 0.0025


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search a wider alpha/tau surface for prospective GD shadow arms"
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
    events = read_events(args.events_path.resolve())
    reserve_data, _ = load_reserve_data(
        args.static_data_root.resolve(),
        args.events_path.resolve(),
        static_config,
    )
    datasets = prepare_controlled_data(reserve_data, events)
    seasons = tuple(data.season for data in datasets)
    folds = expanding_folds(seasons)
    core_selections = pd.read_csv(dynamic_root / "core_fold_selections.csv")
    draw_selections = pd.read_csv(
        evaluation_root / "draw_production_fold_selections.csv"
    )
    validate_fold_inputs(core_selections, draw_selections, folds)
    target = schedule_adjusted_team_performance(events)
    candidates = extended_candidates()

    print(f"Extended GD fixed surface: {len(candidates)} candidates", flush=True)
    fixed_fold_metrics, fixed_summary = run_fixed_surface(
        datasets,
        target,
        folds,
        core_selections,
        draw_selections,
        candidates,
    )
    selected_extended = choose_extended_shadow_candidate(fixed_summary)

    print("Extended GD nested walk-forward", flush=True)
    nested_selections, nested_results, nested_predictions = run_nested_search(
        datasets,
        target,
        folds,
        core_selections,
        draw_selections,
        candidates,
    )
    nested_uncertainty = compare_to_base(
        nested_predictions,
        "NESTED_EXTENDED",
        args.bootstrap_samples,
    )

    arm_candidates = build_shadow_arms(selected_extended)
    print(
        "Fixed shadow arms: "
        + ", ".join(f"{name}={candidate.alpha:g}/{candidate.tau:g}" for name, candidate in arm_candidates),
        flush=True,
    )
    arm_metrics, arm_predictions, arm_uncertainty = evaluate_fixed_arms(
        datasets,
        target,
        folds,
        core_selections,
        draw_selections,
        arm_candidates,
        args.bootstrap_samples,
    )
    plateau = build_plateau(fixed_summary)
    recommendation = build_recommendation(
        selected_extended,
        fixed_summary,
        nested_results,
        nested_selections,
        nested_uncertainty,
        arm_metrics,
        arm_uncertainty,
    )

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    fixed_fold_metrics.to_csv(
        output_root / "extended_candidate_fold_metrics.csv",
        index=False,
    )
    fixed_summary.to_csv(
        output_root / "extended_candidate_summary.csv",
        index=False,
    )
    plateau.to_csv(output_root / "parameter_plateau.csv", index=False)
    nested_selections.to_csv(
        output_root / "nested_fold_selections.csv",
        index=False,
    )
    nested_results.to_csv(
        output_root / "nested_fold_results.csv",
        index=False,
    )
    nested_predictions.to_csv(
        output_root / "nested_unseen_predictions.csv",
        index=False,
    )
    nested_uncertainty.to_csv(
        output_root / "nested_dependency_uncertainty.csv",
        index=False,
    )
    arm_metrics.to_csv(output_root / "shadow_arm_metrics.csv", index=False)
    arm_predictions.to_csv(
        output_root / "shadow_arm_unseen_predictions.csv",
        index=False,
    )
    arm_uncertainty.to_csv(
        output_root / "shadow_arm_dependency_uncertainty.csv",
        index=False,
    )
    (output_root / "recommended_shadow_config.json").write_text(
        json.dumps(recommendation, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_report(
        output_root / "parameter_search_report.md",
        seasons,
        selected_extended,
        fixed_summary,
        plateau,
        nested_selections,
        nested_results,
        nested_uncertainty,
        arm_metrics,
        arm_uncertainty,
        recommendation,
    )
    print(f"Report: {output_root / 'parameter_search_report.md'}")


def extended_candidates() -> tuple[LayerCandidate, ...]:
    candidates = tuple(
        LayerCandidate("GOAL_DIFFERENCE_ONLY", alpha, tau, 0.0)
        for alpha in EXTENDED_ALPHAS
        for tau in EXTENDED_TAUS
        if alpha > 0.0 or tau == 300.0
    )
    for candidate in candidates:
        candidate.validate()
    return candidates


def run_fixed_surface(
    datasets: tuple[ControlledSeasonData, ...],
    target: pd.DataFrame,
    folds,
    core_selections: pd.DataFrame,
    draw_selections: pd.DataFrame,
    candidates: tuple[LayerCandidate, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_season = {data.season: data for data in datasets}
    fold_rows = []
    summary_rows = []
    for candidate_index, candidate in enumerate(candidates, start=1):
        candidate_rows = []
        for fold, (_, test_season) in enumerate(folds, start=1):
            evaluation = evaluate_sequence(
                (by_season[test_season],),
                core_for_fold(core_selections, fold),
                draw_map_for_fold(draw_selections, fold),
                target,
                candidate,
                evaluation_seasons={test_season},
                ranking_target_seasons=set(target["season"]),
            )
            row = {
                "fold": fold,
                "test_season": test_season,
                **candidate_metric_row(candidate, evaluation),
            }
            fold_rows.append(row)
            candidate_rows.append(row)
        frame = pd.DataFrame(candidate_rows)
        summary_rows.append(summarize_fixed_candidate(candidate, frame))
        if candidate_index % 25 == 0:
            print(
                f"Extended candidates: {candidate_index}/{len(candidates)}",
                flush=True,
            )
    fold_metrics = pd.DataFrame(fold_rows)
    summary = pd.DataFrame(summary_rows)
    base = summary.loc[summary["alpha"].eq(0.0)].iloc[0]
    summary["brier_delta_vs_base"] = summary["brier_1x2"] - base["brier_1x2"]
    summary["log_loss_delta_vs_base"] = (
        summary["log_loss_1x2"] - base["log_loss_1x2"]
    )
    summary["ranking_delta_vs_base"] = (
        summary["ranking_score"] - base["ranking_score"]
    )
    summary["pairwise_delta_vs_base"] = (
        summary["pairwise_accuracy"] - base["pairwise_accuracy"]
    )

    base_folds = fold_metrics.loc[fold_metrics["alpha"].eq(0.0)].set_index("fold")
    guardrail_rows = []
    for candidate, frame in fold_metrics.groupby("candidate_key", sort=False):
        indexed = frame.set_index("fold")
        rank_index = indexed.index[
            indexed["ranking_score"].notna()
            & base_folds["ranking_score"].notna()
        ]
        no_regression = (
            indexed.loc[rank_index, "ranking_score"]
            >= base_folds.loc[rank_index, "ranking_score"] - RANK_TOLERANCE
        ) & (
            indexed.loc[rank_index, "pairwise_accuracy"]
            >= base_folds.loc[rank_index, "pairwise_accuracy"] - RANK_TOLERANCE
        )
        practical_no_regression = (
            indexed.loc[rank_index, "ranking_score"]
            >= base_folds.loc[rank_index, "ranking_score"]
            - PRACTICAL_SPEARMAN_TOLERANCE
        ) & (
            indexed.loc[rank_index, "pairwise_accuracy"]
            >= base_folds.loc[rank_index, "pairwise_accuracy"]
            - PRACTICAL_PAIRWISE_TOLERANCE
        )
        both_improved = (
            indexed.loc[rank_index, "ranking_score"]
            > base_folds.loc[rank_index, "ranking_score"] + RANK_TOLERANCE
        ) & (
            indexed.loc[rank_index, "pairwise_accuracy"]
            > base_folds.loc[rank_index, "pairwise_accuracy"] + RANK_TOLERANCE
        )
        guardrail_rows.append(
            {
                "candidate_key": candidate,
                "ranking_evaluable_folds": len(rank_index),
                "ranking_no_regression_folds": int(no_regression.sum()),
                "ranking_practical_no_regression_folds": int(
                    practical_no_regression.sum()
                ),
                "ranking_both_improved_folds": int(both_improved.sum()),
                "brier_win_folds": int(
                    (
                        indexed["brier_1x2"]
                        < base_folds["brier_1x2"] - 1e-12
                    ).sum()
                ),
                "log_loss_win_folds": int(
                    (
                        indexed["log_loss_1x2"]
                        < base_folds["log_loss_1x2"] - 1e-12
                    ).sum()
                ),
            }
        )
    summary = summary.merge(
        pd.DataFrame(guardrail_rows),
        on="candidate_key",
        validate="one_to_one",
    )
    return fold_metrics, summary


def summarize_fixed_candidate(
    candidate: LayerCandidate,
    frame: pd.DataFrame,
) -> dict[str, object]:
    weights = frame["matches"].to_numpy(float)
    ranking_frame = frame.loc[frame["ranking_score"].notna()]
    return {
        "candidate_key": candidate.key,
        "alpha": candidate.alpha,
        "tau": candidate.tau,
        "matches": int(frame["matches"].sum()),
        "brier_1x2": float(np.average(frame["brier_1x2"], weights=weights)),
        "log_loss_1x2": float(
            np.average(frame["log_loss_1x2"], weights=weights)
        ),
        "accuracy_1x2": float(
            np.average(frame["accuracy_1x2"], weights=weights)
        ),
        "ranking_score": float(
            np.average(
                ranking_frame["ranking_score"],
                weights=ranking_frame["matches"],
            )
        ),
        "pairwise_accuracy": float(
            np.average(
                ranking_frame["pairwise_accuracy"],
                weights=ranking_frame["matches"],
            )
        ),
        "maximum_abs_match_delta": float(frame["maximum_abs_match_delta"].max()),
        "maximum_abs_rating_change": float(
            frame["maximum_abs_rating_change"].max()
        ),
        "maximum_total_elo_error": float(
            frame["maximum_total_elo_error"].max()
        ),
    }


def choose_extended_shadow_candidate(summary: pd.DataFrame) -> LayerCandidate:
    candidates = summary.loc[
        summary["alpha"].gt(0.0)
        & summary["brier_delta_vs_base"].lt(0.0)
        & summary["log_loss_delta_vs_base"].lt(0.0)
        & summary["ranking_delta_vs_base"].ge(-RANK_TOLERANCE)
        & summary["pairwise_delta_vs_base"].ge(-RANK_TOLERANCE)
    ].copy()
    if candidates.empty:
        return LayerCandidate("GOAL_DIFFERENCE_ONLY", 0.0, 300.0, 0.0)
    candidates["distance_from_prior"] = (
        (candidates["alpha"] - 0.10).abs() / 0.10
        + (candidates["tau"] - 300.0).abs() / 300.0
    )
    row = candidates.sort_values(
        [
            "ranking_no_regression_folds",
            "ranking_both_improved_folds",
            "brier_1x2",
            "log_loss_1x2",
            "distance_from_prior",
            "alpha",
        ],
        ascending=[False, False, True, True, True, True],
    ).iloc[0]
    return LayerCandidate(
        "GOAL_DIFFERENCE_ONLY",
        float(row["alpha"]),
        float(row["tau"]),
        0.0,
    )


def run_nested_search(
    datasets: tuple[ControlledSeasonData, ...],
    target: pd.DataFrame,
    folds,
    core_selections: pd.DataFrame,
    draw_selections: pd.DataFrame,
    candidates: tuple[LayerCandidate, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_season = {data.season: data for data in datasets}
    baseline = LayerCandidate("GOAL_DIFFERENCE_ONLY", 0.0, 300.0, 0.0)
    selection_rows = []
    result_rows = []
    prediction_frames = []
    for fold, (train_seasons, test_season) in enumerate(folds, start=1):
        core = core_for_fold(core_selections, fold)
        draw = draw_map_for_fold(draw_selections, fold)
        training = tuple(by_season[season] for season in train_seasons)
        metrics = evaluate_candidate_set(
            training,
            target,
            core,
            draw,
            candidates,
        )
        selected_row = select_candidate(metrics, baseline.key)
        selected = LayerCandidate(
            "GOAL_DIFFERENCE_ONLY",
            float(selected_row["alpha"]),
            float(selected_row["tau"]),
            0.0,
        )
        selection_rows.append(
            {
                "fold": fold,
                "train_seasons": "|".join(train_seasons),
                "test_season": test_season,
                "selected_alpha": selected.alpha,
                "selected_tau": selected.tau,
                "selected_candidate": selected.key,
                "train_brier_1x2": float(selected_row["brier_1x2"]),
                "train_log_loss_1x2": float(selected_row["log_loss_1x2"]),
                "train_ranking_score": float(selected_row["ranking_score"]),
                "train_pairwise_accuracy": float(
                    selected_row["pairwise_accuracy"]
                ),
            }
        )
        for label, candidate in (("BASE", baseline), ("NESTED_EXTENDED", selected)):
            evaluation = evaluate_sequence(
                (by_season[test_season],),
                core,
                draw,
                target,
                candidate,
                evaluation_seasons={test_season},
                ranking_target_seasons=set(target["season"]),
                return_predictions=True,
            )
            predictions = evaluation.predictions.copy()
            predictions.insert(0, "fold", fold)
            predictions["model"] = label
            predictions["candidate_key"] = candidate.key
            prediction_frames.append(predictions)
            result_rows.append(
                {
                    "fold": fold,
                    "test_season": test_season,
                    "model": label,
                    "alpha": candidate.alpha,
                    "tau": candidate.tau,
                    **evaluation.metrics,
                }
            )
    return (
        pd.DataFrame(selection_rows),
        pd.DataFrame(result_rows),
        pd.concat(prediction_frames, ignore_index=True),
    )


def compare_to_base(
    predictions: pd.DataFrame,
    challenger_label: str,
    bootstrap_samples: int,
) -> pd.DataFrame:
    frames = []
    for metric in ("brier_1x2", "log_loss_1x2"):
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
            predictions["model"].eq(challenger_label),
            ["match_id", metric],
        ].rename(columns={metric: "challenger_loss"})
        paired = base.merge(challenger, on="match_id", validate="one_to_one")
        paired["loss_difference"] = (
            paired["challenger_loss"] - paired["base_loss"]
        )
        result = dependency_robust_loss_difference_ci(
            paired,
            bootstrap_samples=bootstrap_samples,
            seed=20260723,
        )
        result.insert(0, "model", challenger_label)
        result.insert(1, "metric", metric)
        frames.append(result)
    return pd.concat(frames, ignore_index=True)


def build_shadow_arms(
    selected_extended: LayerCandidate,
) -> tuple[tuple[str, LayerCandidate], ...]:
    arms = [
        (
            name,
            LayerCandidate("GOAL_DIFFERENCE_ONLY", alpha, tau, 0.0),
        )
        for name, alpha, tau in PRE_REGISTERED_ARMS
    ]
    if all(
        selected_extended.alpha != candidate.alpha
        or selected_extended.tau != candidate.tau
        for _, candidate in arms
    ):
        arms.append(("EXTENDED_BEST", selected_extended))
    return tuple(arms)


def evaluate_fixed_arms(
    datasets: tuple[ControlledSeasonData, ...],
    target: pd.DataFrame,
    folds,
    core_selections: pd.DataFrame,
    draw_selections: pd.DataFrame,
    arms: tuple[tuple[str, LayerCandidate], ...],
    bootstrap_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_season = {data.season: data for data in datasets}
    prediction_frames = []
    metric_rows = []
    for label, candidate in arms:
        frames = []
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
            frame = evaluation.predictions.copy()
            frame.insert(0, "fold", fold)
            frame["model"] = label
            frames.append(frame)
        predictions = pd.concat(frames, ignore_index=True)
        prediction_frames.append(predictions)
        metric_rows.append(
            {
                "model": label,
                "alpha": candidate.alpha,
                "tau": candidate.tau,
                **evaluate_predictions(predictions),
                "maximum_goal_multiplier": float(
                    predictions["goal_multiplier"].max()
                ),
                "p99_goal_multiplier": float(
                    predictions["goal_multiplier"].quantile(0.99)
                ),
                "maximum_abs_power_delta": float(
                    predictions["power_delta"].abs().max()
                ),
            }
        )
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    base_metrics = next(row for row in metric_rows if row["model"] == "BASE")
    for row in metric_rows:
        row["brier_delta_vs_base"] = (
            float(row["brier_1x2"]) - float(base_metrics["brier_1x2"])
        )
        row["log_loss_delta_vs_base"] = (
            float(row["log_loss_1x2"]) - float(base_metrics["log_loss_1x2"])
        )
    uncertainty_frames = []
    for label, _ in arms:
        if label == "BASE":
            continue
        uncertainty_frames.append(
            compare_to_base(all_predictions, label, bootstrap_samples)
        )
    return (
        pd.DataFrame(metric_rows),
        all_predictions,
        pd.concat(uncertainty_frames, ignore_index=True),
    )


def build_plateau(summary: pd.DataFrame) -> pd.DataFrame:
    active = summary.loc[summary["alpha"].gt(0.0)].copy()
    best_brier = float(active["brier_1x2"].min())
    active["within_0_00005_of_best"] = active["brier_1x2"].le(
        best_brier + 0.00005
    )
    return active.sort_values(
        ["within_0_00005_of_best", "brier_1x2", "ranking_score"],
        ascending=[False, True, False],
    )


def build_recommendation(
    selected_extended: LayerCandidate,
    fixed_summary: pd.DataFrame,
    nested_results: pd.DataFrame,
    nested_selections: pd.DataFrame,
    nested_uncertainty: pd.DataFrame,
    arm_metrics: pd.DataFrame,
    arm_uncertainty: pd.DataFrame,
) -> dict[str, object]:
    nested_base = nested_results.loc[nested_results["model"].eq("BASE")]
    nested_goal = nested_results.loc[
        nested_results["model"].eq("NESTED_EXTENDED")
    ]
    weights = nested_base["matches"].to_numpy(float)
    nested_brier_delta = float(
        np.average(nested_goal["brier_1x2"], weights=weights)
        - np.average(nested_base["brier_1x2"], weights=weights)
    )
    nested_rank = nested_base.merge(
        nested_goal,
        on=["fold", "test_season"],
        suffixes=("_base", "_goal"),
        validate="one_to_one",
    ).dropna(
        subset=[
            "ranking_score_base",
            "ranking_score_goal",
            "pairwise_accuracy_base",
            "pairwise_accuracy_goal",
        ]
    )
    ranking_safe = (
        nested_rank["ranking_score_goal"].to_numpy(float)
        >= nested_rank["ranking_score_base"].to_numpy(float)
        - RANK_TOLERANCE
    ) & (
        nested_rank["pairwise_accuracy_goal"].to_numpy(float)
        >= nested_rank["pairwise_accuracy_base"].to_numpy(float)
        - RANK_TOLERANCE
    )
    practical_ranking_safe = (
        nested_rank["ranking_score_goal"].to_numpy(float)
        >= nested_rank["ranking_score_base"].to_numpy(float)
        - PRACTICAL_SPEARMAN_TOLERANCE
    ) & (
        nested_rank["pairwise_accuracy_goal"].to_numpy(float)
        >= nested_rank["pairwise_accuracy_base"].to_numpy(float)
        - PRACTICAL_PAIRWISE_TOLERANCE
    )
    pre_specified = fixed_summary.loc[
        fixed_summary["alpha"].eq(0.10)
        & fixed_summary["tau"].eq(300.0)
    ].iloc[0]
    arms = [
        {
            "name": str(row.model),
            "alpha": float(row.alpha),
            "tau": float(row.tau),
            "active_rating_effect": False,
        }
        for row in arm_metrics.itertuples(index=False)
    ]
    return {
        "status": "SHADOW_EVIDENCE_COLLECTION",
        "active_model_goal_difference": {
            "active": False,
            "alpha": 0.0,
            "tau": 300.0,
        },
        "prospective_shadow_arms": arms,
        "extended_fixed_candidate": {
            "alpha": selected_extended.alpha,
            "tau": selected_extended.tau,
        },
        "nested_evidence": {
            "brier_delta_vs_base": nested_brier_delta,
            "ranking_evaluable_folds": int(len(nested_rank)),
            "ranking_no_regression_folds": int(ranking_safe.sum()),
            "ranking_practical_no_regression_folds": int(
                practical_ranking_safe.sum()
            ),
            "selected_alpha_tau_by_fold": nested_selections[
                ["fold", "selected_alpha", "selected_tau"]
            ].to_dict(orient="records"),
            "dependency_envelope": nested_uncertainty.loc[
                nested_uncertainty["method"].eq("conservative_envelope"),
                [
                    "metric",
                    "mean_difference",
                    "ci_95_lower",
                    "ci_95_upper",
                ],
            ].to_dict(orient="records"),
        },
        "pre_specified_historical_evidence": {
            "alpha": 0.10,
            "tau": 300.0,
            "brier_delta_vs_base": float(
                pre_specified["brier_delta_vs_base"]
            ),
            "log_loss_delta_vs_base": float(
                pre_specified["log_loss_delta_vs_base"]
            ),
            "ranking_delta_vs_base": float(
                pre_specified["ranking_delta_vs_base"]
            ),
            "pairwise_delta_vs_base": float(
                pre_specified["pairwise_delta_vs_base"]
            ),
            "brier_win_folds": int(pre_specified["brier_win_folds"]),
            "ranking_exact_no_regression_folds": int(
                pre_specified["ranking_no_regression_folds"]
            ),
            "ranking_practical_no_regression_folds": int(
                pre_specified["ranking_practical_no_regression_folds"]
            ),
            "historical_practical_tolerance_is_post_hoc": True,
        },
        "fixed_arm_metrics": arm_metrics.to_dict(orient="records"),
        "fixed_arm_dependency_envelopes": arm_uncertainty.loc[
            arm_uncertainty["method"].eq("conservative_envelope"),
            [
                "model",
                "metric",
                "mean_difference",
                "ci_95_lower",
                "ci_95_upper",
            ],
        ].to_dict(orient="records"),
        "activation_gate": {
            "prospective_min_matches": 300,
            "prospective_min_ucl_matches": 75,
            "brier_and_log_loss_point_improvement_required": True,
            "dependency_ci_must_not_show_harm": True,
            "forward_pooled_ranking_delta_must_be_nonnegative": True,
            "forward_max_spearman_regression": (
                PRACTICAL_SPEARMAN_TOLERANCE
            ),
            "forward_max_pairwise_regression": (
                PRACTICAL_PAIRWISE_TOLERANCE
            ),
            "maximum_goal_multiplier": 1.30,
            "full_activation_requires_next_season_forward_validation": True,
            "manual_activation_only": True,
        },
        "fixed_surface_rows": int(len(fixed_summary)),
        "development_data_through": "2025/26",
        "untouched_holdout": "2026/27 league phase and later",
    }


def write_report(
    path: Path,
    seasons: tuple[str, ...],
    selected_extended: LayerCandidate,
    fixed_summary: pd.DataFrame,
    plateau: pd.DataFrame,
    nested_selections: pd.DataFrame,
    nested_results: pd.DataFrame,
    nested_uncertainty: pd.DataFrame,
    arm_metrics: pd.DataFrame,
    arm_uncertainty: pd.DataFrame,
    recommendation: dict[str, object],
) -> None:
    best_probability = fixed_summary.sort_values(
        ["brier_1x2", "log_loss_1x2"]
    ).iloc[0]
    nested_base = nested_results.loc[nested_results["model"].eq("BASE")]
    nested_goal = nested_results.loc[
        nested_results["model"].eq("NESTED_EXTENDED")
    ]
    weights = nested_base["matches"].to_numpy(float)
    nested_brier_delta = (
        np.average(nested_goal["brier_1x2"], weights=weights)
        - np.average(nested_base["brier_1x2"], weights=weights)
    )
    nested_log_delta = (
        np.average(nested_goal["log_loss_1x2"], weights=weights)
        - np.average(nested_base["log_loss_1x2"], weights=weights)
    )
    envelope = nested_uncertainty.loc[
        nested_uncertainty["method"].eq("conservative_envelope")
    ].set_index("metric")
    nested_rank = nested_base.merge(
        nested_goal,
        on=["fold", "test_season"],
        suffixes=("_base", "_goal"),
        validate="one_to_one",
    ).dropna(
        subset=[
            "ranking_score_base",
            "ranking_score_goal",
            "pairwise_accuracy_base",
            "pairwise_accuracy_goal",
        ]
    )
    nested_exact_safe = (
        nested_rank["ranking_score_goal"]
        >= nested_rank["ranking_score_base"] - RANK_TOLERANCE
    ) & (
        nested_rank["pairwise_accuracy_goal"]
        >= nested_rank["pairwise_accuracy_base"] - RANK_TOLERANCE
    )
    nested_practical_safe = (
        nested_rank["ranking_score_goal"]
        >= nested_rank["ranking_score_base"]
        - PRACTICAL_SPEARMAN_TOLERANCE
    ) & (
        nested_rank["pairwise_accuracy_goal"]
        >= nested_rank["pairwise_accuracy_base"]
        - PRACTICAL_PAIRWISE_TOLERANCE
    )
    pre_specified = fixed_summary.loc[
        fixed_summary["alpha"].eq(0.10)
        & fixed_summary["tau"].eq(300.0)
    ].iloc[0]
    pre_specified_arm = arm_metrics.loc[
        arm_metrics["model"].eq("PRE_SPECIFIED")
    ].iloc[0]
    pre_specified_envelope = arm_uncertainty.loc[
        arm_uncertainty["model"].eq("PRE_SPECIFIED")
        & arm_uncertainty["method"].eq("conservative_envelope")
    ].set_index("metric")
    lines = [
        "# Gol Farkı Shadow Parametre Araması",
        "",
        "## Kapsam",
        "",
        f"- Sezonlar: `{seasons[0]}-{seasons[-1]}`.",
        f"- Formül değişmedi; `{len(EXTENDED_ALPHAS)} x {len(EXTENDED_TAUS)}` "
        "alpha/tau yüzeyi, alpha=0 tekrarları tekilleştirilerek test edildi.",
        "- Scale, H, K, AO First Elo ve carry değiştirilmedi.",
        "- Fixed yüzey post-hoc duyarlılık, nested seçim ise dürüst unseen kanıttır.",
        "",
        "## Geniş Yüzey Sonucu",
        "",
        f"- En düşük fixed OOS Brier: `alpha={best_probability['alpha']:g}`, "
        f"`tau={best_probability['tau']:g}`, ΔBrier "
        f"`{best_probability['brier_delta_vs_base']:+.6f}`.",
        f"- Ranking-first shadow adayı: `alpha={selected_extended.alpha:g}`, "
        f"`tau={selected_extended.tau:g}`.",
        f"- En iyi Brier'in `0.00005` çevresindeki plateau aday sayısı: "
        f"`{int(plateau['within_0_00005_of_best'].sum())}`.",
        "",
        "## Nested Walk-forward",
        "",
        f"- ΔBrier: `{nested_brier_delta:+.6f}`.",
        f"- Δlog-loss: `{nested_log_delta:+.6f}`.",
        f"- Brier zarfı: "
        f"`[{envelope.loc['brier_1x2','ci_95_lower']:+.6f}, "
        f"{envelope.loc['brier_1x2','ci_95_upper']:+.6f}]`.",
        f"- Log-loss zarfı: "
        f"`[{envelope.loc['log_loss_1x2','ci_95_lower']:+.6f}, "
        f"{envelope.loc['log_loss_1x2','ci_95_upper']:+.6f}]`.",
        f"- Nested sıralama: tam sıfır toleransla "
        f"`{int(nested_exact_safe.sum())}/{len(nested_rank)}`, ileriye dönük "
        f"pratik eşiklerle `{int(nested_practical_safe.sum())}/{len(nested_rank)}` "
        "fold güvenli.",
        "",
        "| Fold | Test | Seçilen alpha | Seçilen tau |",
        "| ---: | --- | ---: | ---: |",
    ]
    for row in nested_selections.itertuples(index=False):
        lines.append(
            f"| {row.fold} | {row.test_season} | "
            f"{row.selected_alpha:g} | {row.selected_tau:g} |"
        )
    lines.extend(
        [
            "",
            "## Prospective Shadow Kolları",
            "",
            "| Kol | alpha | tau | ΔBrier | Δlog-loss |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in arm_metrics.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.alpha:g} | {row.tau:g} | "
            f"{row.brier_delta_vs_base:+.6f} | "
            f"{row.log_loss_delta_vs_base:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Önceden Belirlenen Aday",
            "",
            f"- `alpha=0.10, tau=300` Brier'da "
            f"`{int(pre_specified['brier_win_folds'])}/6` fold kazandı; toplam "
            f"ΔBrier `{pre_specified['brier_delta_vs_base']:+.6f}`, Δlog-loss "
            f"`{pre_specified['log_loss_delta_vs_base']:+.6f}`.",
            f"- Dependency zarfı Brier için "
            f"`[{pre_specified_envelope.loc['brier_1x2','ci_95_lower']:+.6f}, "
            f"{pre_specified_envelope.loc['brier_1x2','ci_95_upper']:+.6f}]`, "
            f"log-loss için "
            f"`[{pre_specified_envelope.loc['log_loss_1x2','ci_95_lower']:+.6f}, "
            f"{pre_specified_envelope.loc['log_loss_1x2','ci_95_upper']:+.6f}]`.",
            f"- Pooled sıralama değişimi Spearman "
            f"`{pre_specified['ranking_delta_vs_base']:+.6f}`, pairwise "
            f"`{pre_specified['pairwise_delta_vs_base']:+.6f}`.",
            f"- Tam sıfır toleransla "
            f"`{int(pre_specified['ranking_no_regression_folds'])}/"
            f"{int(pre_specified['ranking_evaluable_folds'])}`, pratik eşiklerle "
            f"`{int(pre_specified['ranking_practical_no_regression_folds'])}/"
            f"{int(pre_specified['ranking_evaluable_folds'])}` fold güvenli.",
            f"- Maksimum gol çarpanı "
            f"`{pre_specified_arm['maximum_goal_multiplier']:.6f}`, yüzde 99 "
            f"`{pre_specified_arm['p99_goal_multiplier']:.6f}`; maksimum mutlak "
            f"Power delta `{pre_specified_arm['maximum_abs_power_delta']:.3f}`.",
            "- Pratik eşikler bu tarihsel sonuç görüldükten sonra tanımlandığı için "
            "geçmiş veride geriye dönük terfi kanıtı sayılamaz; 2026/27 shadow "
            "ölçümünde önceden kayıtlı kapı olarak kullanılacaktır.",
            "",
            "## Karar",
            "",
            "- Gol farkı ana kod tabanının kalıcı bir bileşenidir, fakat aktif rating "
            "etkisi prospective kapılar geçilene kadar sıfırdır.",
            "- PRE_SPECIFIED, PRIOR_GRID_BEST ve gerekirse EXTENDED_BEST aynı maçlarda "
            "paralel shadow state olarak izlenecektir.",
            "- En az 300 prospective maç ve 75 UCL maçı oluşmadan aktivasyon kararı "
            "verilmeyecektir.",
            f"- Forward sıralama kapısı: pooled fark negatif olmayacak; tek fold "
            f"Spearman gerilemesi `{PRACTICAL_SPEARMAN_TOLERANCE:.4f}`, pairwise "
            f"gerilemesi `{PRACTICAL_PAIRWISE_TOLERANCE:.4f}` değerini aşmayacak.",
            "- Bu ara kapı adayı aday gösterebilir; tam aktivasyon bir sonraki sezonun "
            "forward doğrulamasını ve manuel model kararı gerektirir.",
            "- Brier/log-loss yönü, dependency zararı ve forward sıralama birlikte "
            "değerlendirilecektir; yalnızca modelin daha karmaşık görünmesi terfi "
            "gerekçesi değildir.",
            "",
            f"Makinece okunabilir karar: `{recommendation['status']}`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
