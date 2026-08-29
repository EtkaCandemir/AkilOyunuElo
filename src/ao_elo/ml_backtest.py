from __future__ import annotations

import itertools
import json
import math
import warnings
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from ao_elo.evaluation import dependency_robust_loss_difference_ci
from ao_elo.holdout_window import (
    untouched_holdout_label,
    validate_development_window,
)
from ao_elo.ml_features import FEATURE_SCHEMAS, FeatureSchema, validate_feature_store
from ao_elo.ml_prediction import (
    TrainedML1X2,
    blend_probabilities,
    calibration_metrics,
    fit_ml_1x2,
    multiclass_losses,
    predict_ml_1x2,
    raw_feature_importance,
)


CURRENT_AO = "CURRENT_AO"
AO_CALIBRATION = "AO_CALIBRATION"
STRUCTURAL_LOGISTIC = "STRUCTURAL_LOGISTIC"
DOMESTIC_LOGISTIC = "DOMESTIC_LOGISTIC"
HIST_GRADIENT_BOOSTING = "HIST_GRADIENT_BOOSTING"
AO_ML_BLEND = "AO_ML_BLEND"
CHALLENGER_ARMS = (
    AO_CALIBRATION,
    STRUCTURAL_LOGISTIC,
    DOMESTIC_LOGISTIC,
    HIST_GRADIENT_BOOSTING,
)
ALL_ARMS = (CURRENT_AO,) + CHALLENGER_ARMS + (AO_ML_BLEND,)
BLEND_WEIGHTS = tuple(round(value, 1) for value in np.linspace(0.0, 1.0, 11))


@dataclass(frozen=True)
class ArmSpec:
    name: str
    family: str
    schema: FeatureSchema


@dataclass
class MLBacktestResult:
    candidate_surface: pd.DataFrame
    fold_selections: pd.DataFrame
    fold_results: pd.DataFrame
    unseen_predictions: pd.DataFrame
    model_comparison: pd.DataFrame
    feature_ablation: pd.DataFrame
    feature_importance: pd.DataFrame
    calibration_summary: pd.DataFrame
    competition_segment_summary: pd.DataFrame
    dependency_uncertainty: pd.DataFrame
    selected_model: TrainedML1X2
    selected_blend_weight: float
    decision: dict[str, object]


def run_ml_walk_forward_backtest(
    feature_store: pd.DataFrame,
    *,
    bootstrap_samples: int = 1000,
    pinned_blend_weight: float | None = None,
) -> MLBacktestResult:
    validate_feature_store(feature_store)
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    data = feature_store.sort_values(["kickoff_utc", "match_id"], kind="stable").reset_index(drop=True)
    seasons = validate_development_window(data, label="ml walk-forward backtest")
    outer_tests = seasons[2:]
    if len(outer_tests) != 6:
        raise ValueError("Expected six outer test seasons")

    surface_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []

    for fold, test_season in enumerate(outer_tests, start=1):
        train_seasons = seasons[: seasons.index(test_season)]
        inner_validation = train_seasons[-1]
        inner_train_seasons = train_seasons[:-1]
        inner_train = data[data["season"].isin(inner_train_seasons)]
        inner_valid = data[data["season"].eq(inner_validation)]
        outer_train = data[data["season"].isin(train_seasons)]
        outer_test = data[data["season"].eq(test_season)]
        if inner_train.empty or inner_valid.empty or outer_test.empty:
            raise ValueError(f"Fold {fold} has an empty temporal split")

        selected_by_arm: dict[str, tuple[dict[str, float | int], float]] = {}
        for spec in arm_specs():
            candidates = parameter_grid(spec.family)
            best_key: tuple[float, float, str] | None = None
            best_parameters: dict[str, float | int] | None = None
            best_objective = float("inf")
            baseline_probabilities = _ao_probabilities(inner_valid)
            baseline_metrics = _summary(baseline_probabilities, inner_valid["actual_class"].to_numpy(int))
            for parameters in candidates:
                model = _fit_quietly(inner_train, spec, parameters)
                probabilities = predict_ml_1x2(model, inner_valid)
                metrics = _summary(probabilities, inner_valid["actual_class"].to_numpy(int))
                objective = 0.5 * (
                    metrics["brier_1x2"] / baseline_metrics["brier_1x2"]
                    + metrics["log_loss_1x2"] / baseline_metrics["log_loss_1x2"]
                )
                parameter_json = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
                surface_rows.append(
                    {
                        "fold": fold,
                        "test_season": test_season,
                        "inner_validation_season": inner_validation,
                        "arm": spec.name,
                        "parameters": parameter_json,
                        "objective": objective,
                        **metrics,
                        "delta_brier_vs_ao": metrics["brier_1x2"] - baseline_metrics["brier_1x2"],
                        "delta_log_loss_vs_ao": metrics["log_loss_1x2"] - baseline_metrics["log_loss_1x2"],
                    }
                )
                key = (objective, metrics["log_loss_1x2"], parameter_json)
                if best_key is None or key < best_key:
                    best_key = key
                    best_parameters = parameters
                    best_objective = objective
            assert best_parameters is not None
            selected_by_arm[spec.name] = (best_parameters, best_objective)

        best_ml_arm = min(
            CHALLENGER_ARMS,
            key=lambda name: (selected_by_arm[name][1], name),
        )
        best_spec = spec_by_name(best_ml_arm)
        best_parameters = selected_by_arm[best_ml_arm][0]
        inner_model = _fit_quietly(inner_train, best_spec, best_parameters)
        inner_ml = predict_ml_1x2(inner_model, inner_valid)
        inner_ao = _ao_probabilities(inner_valid)
        blend_scores = []
        for weight in BLEND_WEIGHTS:
            probabilities = blend_probabilities(inner_ao, inner_ml, weight)
            metrics = _summary(probabilities, inner_valid["actual_class"].to_numpy(int))
            objective = 0.5 * (
                metrics["brier_1x2"] / _summary(inner_ao, inner_valid["actual_class"].to_numpy(int))["brier_1x2"]
                + metrics["log_loss_1x2"] / _summary(inner_ao, inner_valid["actual_class"].to_numpy(int))["log_loss_1x2"]
            )
            blend_scores.append((objective, metrics["log_loss_1x2"], weight, metrics))
        _, _, blend_weight, blend_inner_metrics = min(blend_scores)

        selected_models: dict[str, TrainedML1X2] = {}
        for spec in arm_specs():
            parameters = selected_by_arm[spec.name][0]
            model = _fit_quietly(outer_train, spec, parameters)
            selected_models[spec.name] = model
            probabilities = predict_ml_1x2(model, outer_test)
            prediction_frames.append(
                _prediction_frame(outer_test, probabilities, spec.name, fold)
            )

        ao_test = _ao_probabilities(outer_test)
        prediction_frames.append(_prediction_frame(outer_test, ao_test, CURRENT_AO, fold))
        blend_test = blend_probabilities(
            ao_test,
            predict_ml_1x2(selected_models[best_ml_arm], outer_test),
            blend_weight,
        )
        prediction_frames.append(_prediction_frame(outer_test, blend_test, AO_ML_BLEND, fold))
        importance = raw_feature_importance(selected_models[best_ml_arm], outer_test)
        importance.insert(0, "fold", fold)
        importance.insert(1, "test_season", test_season)
        importance.insert(2, "source_arm", best_ml_arm)
        importance_frames.append(importance)

        for arm in CHALLENGER_ARMS:
            parameters, objective = selected_by_arm[arm]
            selection_rows.append(
                {
                    "fold": fold,
                    "test_season": test_season,
                    "train_seasons": "|".join(train_seasons),
                    "inner_validation_season": inner_validation,
                    "arm": arm,
                    "selected_parameters": json.dumps(parameters, sort_keys=True),
                    "inner_objective": objective,
                    "selected_as_blend_source": arm == best_ml_arm,
                    "blend_weight": blend_weight if arm == best_ml_arm else np.nan,
                    "blend_inner_brier": blend_inner_metrics["brier_1x2"] if arm == best_ml_arm else np.nan,
                    "blend_inner_log_loss": blend_inner_metrics["log_loss_1x2"] if arm == best_ml_arm else np.nan,
                }
            )

    unseen = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["fold", "match_id", "model"], kind="stable"
    )
    fold_results = _fold_results(unseen)
    comparison = _model_comparison(unseen)
    ablation = _feature_ablation(comparison)
    calibration = _calibration_summary(unseen)
    segments = _segment_summary(unseen)
    uncertainty = _dependency_uncertainty(unseen, bootstrap_samples)

    full_train = data[data["season"].isin(seasons[:-1])]
    full_valid = data[data["season"].eq(seasons[-1])]
    full_selected: dict[str, tuple[dict[str, float | int], float]] = {}
    for spec in arm_specs():
        baseline_metrics = _summary(_ao_probabilities(full_valid), full_valid["actual_class"].to_numpy(int))
        scores = []
        for parameters in parameter_grid(spec.family):
            model = _fit_quietly(full_train, spec, parameters)
            probabilities = predict_ml_1x2(model, full_valid)
            metrics = _summary(probabilities, full_valid["actual_class"].to_numpy(int))
            objective = 0.5 * (
                metrics["brier_1x2"] / baseline_metrics["brier_1x2"]
                + metrics["log_loss_1x2"] / baseline_metrics["log_loss_1x2"]
            )
            scores.append((objective, metrics["log_loss_1x2"], json.dumps(parameters, sort_keys=True), parameters))
        objective, _, _, parameters = min(scores)
        full_selected[spec.name] = (parameters, objective)
    full_arm = min(CHALLENGER_ARMS, key=lambda name: (full_selected[name][1], name))
    full_spec = spec_by_name(full_arm)
    full_parameters = full_selected[full_arm][0]
    validation_model = _fit_quietly(full_train, full_spec, full_parameters)
    validation_ml = predict_ml_1x2(validation_model, full_valid)
    validation_ao = _ao_probabilities(full_valid)
    full_blend_weight = min(
        BLEND_WEIGHTS,
        key=lambda weight: (
            _combined_objective(
                blend_probabilities(validation_ao, validation_ml, weight),
                full_valid["actual_class"].to_numpy(int),
                validation_ao,
            ),
            weight,
        ),
    )
    # Rebuilding this backtest after an unrelated activation would otherwise
    # silently re-open the served AO/ML blend weight, which the 2026/27 holdout
    # protocol freezes. Pinning keeps the rebuild a propagation; the surface's
    # own preference stays in `full_history_blend_weight`.
    if pinned_blend_weight is None:
        selected_blend_weight = full_blend_weight
    else:
        selected_blend_weight = float(pinned_blend_weight)
        if not 0.0 <= selected_blend_weight <= 1.0:
            raise ValueError("pinned_blend_weight must be in [0,1]")
    final_model = _fit_quietly(data, full_spec, full_parameters)
    full_importance = raw_feature_importance(final_model, data)
    full_importance.insert(0, "fold", 0)
    full_importance.insert(1, "test_season", "FULL_HISTORY")
    full_importance.insert(2, "source_arm", full_arm)
    importance_frames.append(full_importance)
    feature_importance = pd.concat(importance_frames, ignore_index=True)
    decision = _decision(
        comparison,
        fold_results,
        calibration,
        segments,
        uncertainty,
        pd.DataFrame(selection_rows),
        full_arm,
        full_parameters,
        full_blend_weight,
        final_model,
        untouched_holdout_label(data, label="ml decision"),
    )
    return MLBacktestResult(
        candidate_surface=pd.DataFrame(surface_rows),
        fold_selections=pd.DataFrame(selection_rows),
        fold_results=fold_results,
        unseen_predictions=unseen,
        model_comparison=comparison,
        feature_ablation=ablation,
        feature_importance=feature_importance,
        calibration_summary=calibration,
        competition_segment_summary=segments,
        dependency_uncertainty=uncertainty,
        selected_model=final_model,
        selected_blend_weight=float(selected_blend_weight),
        decision=decision,
    )


def arm_specs() -> tuple[ArmSpec, ...]:
    return (
        ArmSpec(AO_CALIBRATION, "LOGISTIC", FEATURE_SCHEMAS[AO_CALIBRATION]),
        ArmSpec(STRUCTURAL_LOGISTIC, "LOGISTIC", FEATURE_SCHEMAS[STRUCTURAL_LOGISTIC]),
        ArmSpec(DOMESTIC_LOGISTIC, "LOGISTIC", FEATURE_SCHEMAS[DOMESTIC_LOGISTIC]),
        ArmSpec(
            HIST_GRADIENT_BOOSTING,
            "HIST_GRADIENT_BOOSTING",
            FEATURE_SCHEMAS[HIST_GRADIENT_BOOSTING],
        ),
    )


def spec_by_name(name: str) -> ArmSpec:
    for spec in arm_specs():
        if spec.name == name:
            return spec
    raise ValueError(f"Unknown arm: {name}")


def parameter_grid(family: str) -> tuple[dict[str, float | int], ...]:
    if family == "LOGISTIC":
        return tuple(
            {"C": c_value, "l1_ratio": ratio}
            for c_value, ratio in itertools.product(
                (0.03, 0.10, 0.30, 1.0, 3.0),
                (0.0, 0.25, 0.50, 0.75),
            )
        )
    if family == "HIST_GRADIENT_BOOSTING":
        return tuple(
            {
                "learning_rate": learning_rate,
                "max_leaf_nodes": leaves,
                "min_samples_leaf": minimum,
                "l2_regularization": regularization,
            }
            for learning_rate, leaves, minimum, regularization in itertools.product(
                (0.03, 0.05),
                (7, 15),
                (50, 100),
                (1.0, 10.0),
            )
        )
    raise ValueError(f"Unknown family: {family}")


def _fit_quietly(
    frame: pd.DataFrame,
    spec: ArmSpec,
    parameters: Mapping[str, float | int],
) -> TrainedML1X2:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return fit_ml_1x2(
            frame,
            arm_name=spec.name,
            family=spec.family,
            schema=spec.schema,
            parameters=parameters,
        )


def _ao_probabilities(frame: pd.DataFrame) -> np.ndarray:
    return frame[
        ["ao_home_probability", "ao_draw_probability", "ao_away_probability"]
    ].to_numpy(float)


def _summary(probabilities: np.ndarray, outcomes: np.ndarray) -> dict[str, float]:
    losses = multiclass_losses(probabilities, outcomes)
    return {
        "brier_1x2": float(losses["brier_1x2"].mean()),
        "log_loss_1x2": float(losses["log_loss_1x2"].mean()),
        "accuracy_1x2": float(losses["correct"].mean()),
    }


def _combined_objective(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    ao_probabilities: np.ndarray,
) -> float:
    metrics = _summary(probabilities, outcomes)
    baseline = _summary(ao_probabilities, outcomes)
    return 0.5 * (
        metrics["brier_1x2"] / baseline["brier_1x2"]
        + metrics["log_loss_1x2"] / baseline["log_loss_1x2"]
    )


def _prediction_frame(
    source: pd.DataFrame,
    probabilities: np.ndarray,
    model: str,
    fold: int,
) -> pd.DataFrame:
    losses = multiclass_losses(probabilities, source["actual_class"].to_numpy(int))
    columns = [
        "match_id",
        "season",
        "kickoff_utc",
        "competition",
        "stage",
        "tie_id",
        "home_team_id",
        "away_team_id",
        "home_club_id",
        "away_club_id",
        "actual_class",
        "is_single_match_tie",
        "both_domestic_covered",
        "home_domestic_covered",
        "away_domestic_covered",
        "expected_home_score",
        "baseline_power_delta",
    ]
    result = source[columns].reset_index(drop=True).copy()
    result.insert(0, "fold", fold)
    result.insert(1, "model", model)
    result["home_probability"] = probabilities[:, 0]
    result["draw_probability"] = probabilities[:, 1]
    result["away_probability"] = probabilities[:, 2]
    return pd.concat([result, losses], axis=1)


def _fold_results(unseen: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fold, season, model), group in unseen.groupby(["fold", "season", "model"], sort=True):
        rows.append(
            {
                "fold": fold,
                "test_season": season,
                "model": model,
                "matches": len(group),
                "brier_1x2": float(group["brier_1x2"].mean()),
                "log_loss_1x2": float(group["log_loss_1x2"].mean()),
                "accuracy_1x2": float(group["correct"].mean()),
            }
        )
    result = pd.DataFrame(rows)
    baseline = result[result["model"].eq(CURRENT_AO)][
        ["fold", "brier_1x2", "log_loss_1x2", "accuracy_1x2"]
    ].rename(
        columns={
            "brier_1x2": "baseline_brier",
            "log_loss_1x2": "baseline_log_loss",
            "accuracy_1x2": "baseline_accuracy",
        }
    )
    result = result.merge(baseline, on="fold", validate="many_to_one")
    result["delta_brier_vs_ao"] = result["brier_1x2"] - result["baseline_brier"]
    result["delta_log_loss_vs_ao"] = result["log_loss_1x2"] - result["baseline_log_loss"]
    result["delta_accuracy_vs_ao"] = result["accuracy_1x2"] - result["baseline_accuracy"]
    return result


def _model_comparison(unseen: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in ALL_ARMS:
        group = unseen[unseen["model"].eq(model)]
        rows.append({"model": model, "matches": len(group), **_summary(
            group[["home_probability", "draw_probability", "away_probability"]].to_numpy(float),
            group["actual_class"].to_numpy(int),
        )})
    result = pd.DataFrame(rows)
    baseline = result[result["model"].eq(CURRENT_AO)].iloc[0]
    for metric in ("brier_1x2", "log_loss_1x2", "accuracy_1x2"):
        result[f"delta_vs_ao_{metric}"] = result[metric] - float(baseline[metric])
    return result


def _feature_ablation(comparison: pd.DataFrame) -> pd.DataFrame:
    order = {
        CURRENT_AO: "AO only",
        AO_CALIBRATION: "+ format calibration",
        STRUCTURAL_LOGISTIC: "+ European form and context",
        DOMESTIC_LOGISTIC: "+ domestic form",
        HIST_GRADIENT_BOOSTING: "+ nonlinear interactions",
        AO_ML_BLEND: "+ AO anchored blend",
    }
    result = comparison.copy()
    result.insert(1, "feature_layer", result["model"].map(order))
    return result


def _calibration_summary(unseen: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in unseen.groupby("model", sort=True):
        rows.append(
            {
                "model": model,
                "scope": "POOLED",
                "matches": len(group),
                **calibration_metrics(
                    group[["home_probability", "draw_probability", "away_probability"]].to_numpy(float),
                    group["actual_class"].to_numpy(int),
                ),
            }
        )
    return pd.DataFrame(rows)


def _segment_summary(unseen: pd.DataFrame) -> pd.DataFrame:
    rows = []
    segment_specs = (
        ("competition", "competition"),
        ("domestic_coverage", "both_domestic_covered"),
        ("format", "is_single_match_tie"),
    )
    for segment_type, column in segment_specs:
        for (value, model), group in unseen.groupby([column, "model"], sort=True, dropna=False):
            rows.append(
                {
                    "segment_type": segment_type,
                    "segment_value": str(value),
                    "model": model,
                    "matches": len(group),
                    **_summary(
                        group[["home_probability", "draw_probability", "away_probability"]].to_numpy(float),
                        group["actual_class"].to_numpy(int),
                    ),
                }
            )
    result = pd.DataFrame(rows)
    baseline = result[result["model"].eq(CURRENT_AO)][
        ["segment_type", "segment_value", "brier_1x2", "log_loss_1x2"]
    ].rename(columns={"brier_1x2": "baseline_brier", "log_loss_1x2": "baseline_log_loss"})
    result = result.merge(baseline, on=["segment_type", "segment_value"], validate="many_to_one")
    result["delta_brier_vs_ao"] = result["brier_1x2"] - result["baseline_brier"]
    result["delta_log_loss_vs_ao"] = result["log_loss_1x2"] - result["baseline_log_loss"]
    return result


def _dependency_uncertainty(unseen: pd.DataFrame, bootstrap_samples: int) -> pd.DataFrame:
    baseline = unseen[unseen["model"].eq(CURRENT_AO)].copy()
    candidate = unseen[unseen["model"].eq(AO_ML_BLEND)].copy()
    merged = candidate.merge(
        baseline[["match_id", "brier_1x2", "log_loss_1x2"]],
        on="match_id",
        suffixes=("", "_baseline"),
        validate="one_to_one",
    )
    rows = []
    scopes: list[tuple[str, pd.DataFrame]] = [("ALL", merged)]
    scopes.extend((f"competition:{value}", group) for value, group in merged.groupby("competition", sort=True))
    scopes.append(("domestic_coverage:0", merged[merged["both_domestic_covered"].eq(0)]))
    for scope, group in scopes:
        for metric in ("brier_1x2", "log_loss_1x2"):
            audit = group.copy()
            audit["loss_difference"] = audit[metric] - audit[f"{metric}_baseline"]
            ci = dependency_robust_loss_difference_ci(
                audit,
                bootstrap_samples=bootstrap_samples,
                seed=20260812 + len(rows),
            )
            ci.insert(0, "scope", scope)
            ci.insert(1, "metric", metric)
            rows.append(ci)
    return pd.concat(rows, ignore_index=True)


def _decision(
    comparison: pd.DataFrame,
    fold_results: pd.DataFrame,
    calibration: pd.DataFrame,
    segments: pd.DataFrame,
    uncertainty: pd.DataFrame,
    selections: pd.DataFrame,
    full_arm: str,
    full_parameters: Mapping[str, float | int],
    full_blend_weight: float,
    final_model: TrainedML1X2,
    holdout: str,
) -> dict[str, object]:
    baseline = comparison[comparison["model"].eq(CURRENT_AO)].iloc[0]
    candidate = comparison[comparison["model"].eq(AO_ML_BLEND)].iloc[0]
    folds = fold_results[fold_results["model"].eq(AO_ML_BLEND)]
    brier_wins = int((folds["delta_brier_vs_ao"] < 0.0).sum())
    log_wins = int((folds["delta_log_loss_vs_ao"] < 0.0).sum())
    envelope = uncertainty[
        uncertainty["scope"].eq("ALL") & uncertainty["method"].eq("conservative_envelope")
    ]
    brier_ci = envelope[envelope["metric"].eq("brier_1x2")].iloc[0]
    log_ci = envelope[envelope["metric"].eq("log_loss_1x2")].iloc[0]
    segment_envelopes = uncertainty[
        uncertainty["method"].eq("conservative_envelope") & ~uncertainty["scope"].eq("ALL")
    ]
    no_reliable_segment_harm = not bool(segment_envelopes["reliable_harm"].any())
    base_cal = calibration[calibration["model"].eq(CURRENT_AO)].iloc[0]
    candidate_cal = calibration[calibration["model"].eq(AO_ML_BLEND)].iloc[0]
    calibration_safe = float(candidate_cal["ece"]) <= float(base_cal["ece"]) + 1e-12
    nonzero_blend_folds = int(
        selections[selections["selected_as_blend_source"]]["blend_weight"].fillna(0.0).gt(0.0).sum()
    )
    pooled_safe = bool(
        candidate["brier_1x2"] < baseline["brier_1x2"]
        and candidate["log_loss_1x2"] < baseline["log_loss_1x2"]
    )
    uncertainty_safe = bool(
        (bool(brier_ci["reliable_improvement"]) or bool(log_ci["reliable_improvement"]))
        and not bool(brier_ci["reliable_harm"])
        and not bool(log_ci["reliable_harm"])
    )
    gates = {
        "brier_fold_wins": brier_wins,
        "log_loss_fold_wins": log_wins,
        "fold_gate": brier_wins >= 4 and log_wins >= 4,
        "pooled_loss_gate": pooled_safe,
        "uncertainty_gate": uncertainty_safe,
        "competition_and_missing_segment_gate": no_reliable_segment_harm,
        "calibration_gate": calibration_safe,
        "nonzero_blend_folds": nonzero_blend_folds,
        "blend_usage_gate": nonzero_blend_folds >= 4,
        "probability_gate": True,
        "rating_state_identity_gate": True,
    }
    passed = all(
        bool(gates[key])
        for key in (
            "fold_gate",
            "pooled_loss_gate",
            "uncertainty_gate",
            "competition_and_missing_segment_gate",
            "calibration_gate",
            "blend_usage_gate",
            "probability_gate",
            "rating_state_identity_gate",
        )
    )
    return {
        "decision": "PROMOTE_CANDIDATE" if passed else "KEEP_SHADOW",
        "production_activated": False,
        "candidate_arm": AO_ML_BLEND,
        "full_history_ml_source": full_arm,
        "full_history_parameters": dict(full_parameters),
        "full_history_blend_weight": float(full_blend_weight),
        "model_fingerprint": final_model.fingerprint,
        "pooled_brier": float(candidate["brier_1x2"]),
        "pooled_log_loss": float(candidate["log_loss_1x2"]),
        "delta_brier_vs_ao": float(candidate["brier_1x2"] - baseline["brier_1x2"]),
        "delta_log_loss_vs_ao": float(candidate["log_loss_1x2"] - baseline["log_loss_1x2"]),
        "gates": gates,
        "holdout": f"{holdout}_UNTOUCHED",
    }
