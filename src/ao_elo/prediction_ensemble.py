from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from ao_elo.domestic_poisson import (
    DomesticPoissonConfig,
    build_domestic_poisson_feature_store,
    fit_european_poisson_transfer,
    predict_european_poisson_transfer,
)
from ao_elo.evaluation import dependency_robust_loss_difference_ci
from ao_elo.holdout_window import (
    untouched_holdout_label,
    validate_development_window,
)
from ao_elo.ml_backtest import spec_by_name
from ao_elo.ml_features import validate_feature_store
from ao_elo.ml_prediction import (
    blend_probabilities,
    calibration_metrics,
    fit_ml_1x2,
    multiclass_losses,
    predict_ml_1x2,
)


CURRENT_AO = "CURRENT_AO"
CURRENT_ML_BLEND = "CURRENT_ML_BLEND"
AO_POISSON_BLEND = "AO_POISSON_BLEND"
AO_POISSON_RHO0_CONTROL = "AO_POISSON_RHO0_CONTROL"
ML_POISSON_ENSEMBLE = "ML_POISSON_ENSEMBLE"
SOURCE_DOMESTIC_POISSON_RHO0 = "DOMESTIC_ATTACK_DEFENCE_POISSON"
POISSON_SOURCES = (AO_POISSON_RHO0_CONTROL, AO_POISSON_BLEND)
ALL_MODELS = (
    CURRENT_AO,
    CURRENT_ML_BLEND,
    AO_POISSON_BLEND,
    AO_POISSON_RHO0_CONTROL,
    ML_POISSON_ENSEMBLE,
)
BLEND_WEIGHTS = tuple(round(value, 1) for value in np.linspace(0.0, 1.0, 11))
PROBABILITY_COLUMNS = (
    "home_probability",
    "draw_probability",
    "away_probability",
)


@dataclass
class PredictionEnsembleBacktestResult:
    inner_weight_surface: pd.DataFrame
    fold_selections: pd.DataFrame
    fold_results: pd.DataFrame
    unseen_predictions: pd.DataFrame
    model_comparison: pd.DataFrame
    calibration_summary: pd.DataFrame
    competition_coverage_summary: pd.DataFrame
    dependency_uncertainty: pd.DataFrame
    prospective_weight_surface: pd.DataFrame
    selected_candidate: dict[str, object]


def run_prediction_ensemble_walk_forward_backtest(
    feature_store: pd.DataFrame,
    domestic_matches: pd.DataFrame,
    bridge: pd.DataFrame,
    domestic_predictions: pd.DataFrame,
    ml_selections: pd.DataFrame,
    domestic_selections: pd.DataFrame,
    transfer_selections: pd.DataFrame,
    *,
    bootstrap_samples: int = 1000,
    pinned_poisson_weight: float | None = None,
) -> PredictionEnsembleBacktestResult:
    """Blend the existing ML and domestic Poisson OOS candidates without Elo feedback."""

    validate_feature_store(feature_store)
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    base = feature_store.sort_values(
        ["kickoff_utc", "match_id"], kind="stable"
    ).reset_index(drop=True)
    seasons = validate_development_window(base, label="prediction ensemble backtest")

    source = _validate_and_align_source_predictions(
        domestic_predictions, seasons[2:]
    )
    ml_rows = _selected_ml_rows(ml_selections, seasons[2:])
    domestic_rows = _selection_rows(
        domestic_selections, seasons[2:], "domestic selections"
    )
    transfer_rows = _selection_rows(
        transfer_selections, seasons[2:], "transfer selections"
    )
    rho0_outer = _rho0_outer_control(source, transfer_rows)

    feature_cache: dict[str, pd.DataFrame] = {}
    surface_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    ensemble_frames: list[pd.DataFrame] = []

    for fold, test_season in enumerate(seasons[2:], start=1):
        train_seasons = seasons[: seasons.index(test_season)]
        inner_validation_season = train_seasons[-1]
        inner_train_seasons = train_seasons[:-1]
        if not inner_train_seasons:
            raise ValueError(f"Fold {fold} has no inner training season")

        inner_valid = base[base["season"].eq(inner_validation_season)].copy()
        inner_ml = _inner_ml_probabilities(
            base,
            inner_train_seasons,
            inner_valid,
            ml_rows.loc[fold],
        )
        inner_poisson = _inner_poisson_probabilities(
            base,
            domestic_matches,
            bridge,
            feature_cache,
            inner_train_seasons,
            inner_valid,
            domestic_rows.loc[fold],
            transfer_rows.loc[fold],
        )
        outcomes = inner_valid["actual_class"].to_numpy(int)
        selected, rows = select_prediction_ensemble(
            inner_ml,
            inner_poisson,
            outcomes,
        )
        for row in rows:
            surface_rows.append(
                {
                    "fold": fold,
                    "test_season": test_season,
                    "inner_validation_season": inner_validation_season,
                    **row,
                }
            )
        selection_rows.append(
            {
                "fold": fold,
                "test_season": test_season,
                "train_seasons": "|".join(train_seasons),
                "inner_train_seasons": "|".join(inner_train_seasons),
                "inner_validation_season": inner_validation_season,
                **selected,
            }
        )

        ml_test = source[CURRENT_ML_BLEND]
        ml_test = ml_test[ml_test["fold"].eq(fold)].reset_index(drop=True)
        poisson_test = (
            rho0_outer
            if selected["poisson_source"] == AO_POISSON_RHO0_CONTROL
            else source[AO_POISSON_BLEND]
        )
        poisson_test = poisson_test[poisson_test["fold"].eq(fold)].reset_index(
            drop=True
        )
        _assert_same_match_order(ml_test, poisson_test)
        probabilities = blend_probabilities(
            _probabilities(ml_test),
            _probabilities(poisson_test),
            float(selected["poisson_weight"]),
        )
        ensemble_frames.append(
            _prediction_frame(ml_test, probabilities, ML_POISSON_ENSEMBLE)
        )

    ensemble = pd.concat(ensemble_frames, ignore_index=True)
    prediction_frames = [source[model] for model in (CURRENT_AO, CURRENT_ML_BLEND, AO_POISSON_BLEND)]
    prediction_frames.extend((rho0_outer, ensemble))
    unseen = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["fold", "match_id", "model"], kind="stable"
    )
    _validate_unseen_predictions(unseen, source[CURRENT_AO])

    fold_results = _fold_results(unseen)
    comparison = _model_comparison(unseen)
    calibration = _calibration_summary(unseen)
    segments = _segment_summary(unseen)
    uncertainty = _dependency_uncertainty(unseen, bootstrap_samples)

    full_ml = _probabilities(source[CURRENT_ML_BLEND])
    full_outcomes = source[CURRENT_ML_BLEND]["actual_class"].to_numpy(int)
    full_sources = {
        AO_POISSON_RHO0_CONTROL: _probabilities(rho0_outer),
        AO_POISSON_BLEND: _probabilities(source[AO_POISSON_BLEND]),
    }
    prospective_selection, prospective_rows = select_prediction_ensemble(
        full_ml,
        full_sources,
        full_outcomes,
    )
    if pinned_poisson_weight is not None:
        # Rebuilding this backtest after an unrelated activation would
        # otherwise silently re-open the served blend weight, which the
        # 2026/27 holdout protocol forbids. Pinning keeps the rebuild a
        # propagation; the surface's own preference is still reported.
        surface_choice = float(prospective_selection["poisson_weight"])
        pinned = float(pinned_poisson_weight)
        if not 0.0 <= pinned <= 1.0:
            raise ValueError("pinned_poisson_weight must be in [0,1]")
        prospective_selection = dict(prospective_selection)
        prospective_selection["poisson_weight"] = pinned
        prospective_selection["ml_weight"] = 1.0 - pinned
        prospective_selection["weight_pinned"] = True
        prospective_selection["surface_would_have_selected"] = surface_choice
    prospective_surface = pd.DataFrame(prospective_rows)
    decision = _decision(
        comparison,
        fold_results,
        calibration,
        uncertainty,
        prospective_selection,
        pd.DataFrame(selection_rows),
        untouched_holdout_label(base, label="prediction ensemble decision"),
    )
    return PredictionEnsembleBacktestResult(
        inner_weight_surface=pd.DataFrame(surface_rows),
        fold_selections=pd.DataFrame(selection_rows),
        fold_results=fold_results,
        unseen_predictions=unseen,
        model_comparison=comparison,
        calibration_summary=calibration,
        competition_coverage_summary=segments,
        dependency_uncertainty=uncertainty,
        prospective_weight_surface=prospective_surface,
        selected_candidate=decision,
    )


def select_prediction_ensemble(
    current_ml_probabilities: np.ndarray,
    poisson_sources: Mapping[str, np.ndarray],
    outcomes: np.ndarray,
    *,
    weights: Sequence[float] = BLEND_WEIGHTS,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Select a Poisson source and log-probability blend weight on past data."""

    current_ml = np.asarray(current_ml_probabilities, dtype=float)
    targets = np.asarray(outcomes, dtype=int)
    _validate_probability_matrix(current_ml, "current ML probabilities")
    if targets.shape != (len(current_ml),):
        raise ValueError("outcomes must align with probability rows")
    if set(poisson_sources) != set(POISSON_SOURCES):
        raise ValueError(f"Poisson sources must be exactly {POISSON_SOURCES}")
    if not weights:
        raise ValueError("At least one ensemble weight is required")

    baseline = _summary(current_ml, targets)
    rows: list[dict[str, object]] = []
    for source_name in POISSON_SOURCES:
        source_probability = np.asarray(poisson_sources[source_name], dtype=float)
        _validate_probability_matrix(source_probability, source_name)
        if source_probability.shape != current_ml.shape:
            raise ValueError(f"{source_name} does not align with current ML")
        for weight in weights:
            if not math.isfinite(float(weight)) or not 0.0 <= float(weight) <= 1.0:
                raise ValueError("Ensemble weights must be finite and in [0,1]")
            probabilities = blend_probabilities(
                current_ml, source_probability, float(weight)
            )
            metrics = _summary(probabilities, targets)
            objective = 0.5 * (
                metrics["brier_1x2"] / baseline["brier_1x2"]
                + metrics["log_loss_1x2"] / baseline["log_loss_1x2"]
            )
            rows.append(
                {
                    "poisson_source": source_name,
                    "poisson_weight": float(weight),
                    "ml_weight": 1.0 - float(weight),
                    "objective": float(objective),
                    **metrics,
                    "delta_brier_vs_current_ml": (
                        metrics["brier_1x2"] - baseline["brier_1x2"]
                    ),
                    "delta_log_loss_vs_current_ml": (
                        metrics["log_loss_1x2"] - baseline["log_loss_1x2"]
                    ),
                }
            )
    selected = min(
        rows,
        key=lambda row: (
            round(float(row["objective"]), 12),
            round(float(row["log_loss_1x2"]), 12),
            round(float(row["brier_1x2"]), 12),
            row["poisson_weight"],
            0 if row["poisson_source"] == AO_POISSON_RHO0_CONTROL else 1,
        ),
    )
    return dict(selected), rows


def _inner_ml_probabilities(
    base: pd.DataFrame,
    inner_train_seasons: Sequence[str],
    inner_valid: pd.DataFrame,
    selection: pd.Series,
) -> np.ndarray:
    spec = spec_by_name(str(selection["arm"]))
    parameters = json.loads(str(selection["selected_parameters"]))
    train = base[base["season"].isin(inner_train_seasons)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = fit_ml_1x2(
            train,
            arm_name=spec.name,
            family=spec.family,
            schema=spec.schema,
            parameters=parameters,
        )
    raw_ml = predict_ml_1x2(model, inner_valid)
    return blend_probabilities(
        _ao_probabilities(inner_valid),
        raw_ml,
        float(selection["blend_weight"]),
    )


def _inner_poisson_probabilities(
    base: pd.DataFrame,
    domestic_matches: pd.DataFrame,
    bridge: pd.DataFrame,
    cache: dict[str, pd.DataFrame],
    inner_train_seasons: Sequence[str],
    inner_valid: pd.DataFrame,
    domestic_selection: pd.Series,
    transfer_selection: pd.Series,
) -> dict[str, np.ndarray]:
    domestic_config = DomesticPoissonConfig(
        team_learning_rate=float(domestic_selection["team_learning_rate"]),
        season_carry=float(domestic_selection["season_carry"]),
        shrinkage_matches=float(domestic_selection["shrinkage_matches"]),
        venue_context=_as_bool(domestic_selection["venue_context"]),
    )
    if domestic_config.key not in cache:
        features = build_domestic_poisson_feature_store(
            domestic_matches, base, bridge, domestic_config
        )
        cache[domestic_config.key] = base.merge(
            features, on="match_id", validate="one_to_one"
        )
    enriched = cache[domestic_config.key]
    train = enriched[enriched["season"].isin(inner_train_seasons)]
    validation = enriched[enriched["season"].eq(str(inner_valid["season"].iloc[0]))]
    _assert_same_match_order(
        inner_valid.reset_index(drop=True), validation.reset_index(drop=True)
    )
    fitted = fit_european_poisson_transfer(
        train,
        l2_strength=float(transfer_selection["selected_l2_strength"]),
        use_reliability=True,
        use_venue=domestic_config.venue_context,
    )
    ao = _ao_probabilities(validation)
    poisson_weight = float(transfer_selection["poisson_blend_weight"])
    result: dict[str, np.ndarray] = {}
    for source_name, rho in (
        (AO_POISSON_RHO0_CONTROL, 0.0),
        (AO_POISSON_BLEND, float(transfer_selection["selected_rho"])),
    ):
        predicted = predict_european_poisson_transfer(
            validation,
            replace(fitted, rho=rho),
            fallback_to_ao_without_history=True,
        )
        result[source_name] = blend_probabilities(
            ao,
            _probabilities(predicted),
            poisson_weight,
        )
    return result


def _rho0_outer_control(
    source: Mapping[str, pd.DataFrame], transfer_rows: pd.DataFrame
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for fold in sorted(source[CURRENT_AO]["fold"].unique()):
        ao = source[CURRENT_AO]
        ao = ao[ao["fold"].eq(fold)].reset_index(drop=True)
        raw = source[SOURCE_DOMESTIC_POISSON_RHO0]
        raw = raw[raw["fold"].eq(fold)].reset_index(drop=True)
        _assert_same_match_order(ao, raw)
        probability = blend_probabilities(
            _probabilities(ao),
            _probabilities(raw),
            float(transfer_rows.loc[fold, "poisson_blend_weight"]),
        )
        frames.append(
            _prediction_frame(ao, probability, AO_POISSON_RHO0_CONTROL)
        )
    return pd.concat(frames, ignore_index=True)


def _prediction_frame(
    source: pd.DataFrame, probabilities: np.ndarray, model: str
) -> pd.DataFrame:
    result = source[
        [
            "fold",
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
            "expected_home_score",
            "baseline_power_delta",
            "domestic_poisson_coverage",
        ]
    ].reset_index(drop=True).copy()
    result.insert(1, "model", model)
    result[list(PROBABILITY_COLUMNS)] = probabilities
    losses = multiclass_losses(
        probabilities, result["actual_class"].to_numpy(int)
    )
    return pd.concat([result, losses], axis=1)


def _fold_results(unseen: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fold, season, model), group in unseen.groupby(
        ["fold", "season", "model"], sort=True
    ):
        rows.append(
            {
                "fold": fold,
                "test_season": season,
                "model": model,
                "matches": len(group),
                **_summary(
                    _probabilities(group), group["actual_class"].to_numpy(int)
                ),
            }
        )
    result = pd.DataFrame(rows)
    for baseline_model, suffix in (
        (CURRENT_AO, "ao"),
        (CURRENT_ML_BLEND, "current_ml"),
    ):
        baseline = result[result["model"].eq(baseline_model)][
            ["fold", "brier_1x2", "log_loss_1x2", "accuracy_1x2"]
        ].rename(
            columns={
                "brier_1x2": f"{suffix}_brier",
                "log_loss_1x2": f"{suffix}_log_loss",
                "accuracy_1x2": f"{suffix}_accuracy",
            }
        )
        result = result.merge(baseline, on="fold", validate="many_to_one")
        result[f"delta_brier_vs_{suffix}"] = (
            result["brier_1x2"] - result[f"{suffix}_brier"]
        )
        result[f"delta_log_loss_vs_{suffix}"] = (
            result["log_loss_1x2"] - result[f"{suffix}_log_loss"]
        )
        result[f"delta_accuracy_vs_{suffix}"] = (
            result["accuracy_1x2"] - result[f"{suffix}_accuracy"]
        )
    return result


def _model_comparison(unseen: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in ALL_MODELS:
        group = unseen[unseen["model"].eq(model)]
        rows.append(
            {
                "model": model,
                "matches": len(group),
                **_summary(
                    _probabilities(group), group["actual_class"].to_numpy(int)
                ),
            }
        )
    result = pd.DataFrame(rows)
    for baseline_model, suffix in (
        (CURRENT_AO, "ao"),
        (CURRENT_ML_BLEND, "current_ml"),
    ):
        baseline = result[result["model"].eq(baseline_model)].iloc[0]
        for metric in ("brier_1x2", "log_loss_1x2", "accuracy_1x2"):
            result[f"delta_{metric}_vs_{suffix}"] = (
                result[metric] - float(baseline[metric])
            )
    return result


def _calibration_summary(unseen: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in ALL_MODELS:
        group = unseen[unseen["model"].eq(model)]
        rows.append(
            {
                "model": model,
                "matches": len(group),
                **calibration_metrics(
                    _probabilities(group), group["actual_class"].to_numpy(int)
                ),
            }
        )
    return pd.DataFrame(rows)


def _segment_summary(unseen: pd.DataFrame) -> pd.DataFrame:
    values = unseen.copy()
    values["favorite_band"] = pd.cut(
        values["expected_home_score"],
        bins=(-np.inf, 0.35, 0.65, np.inf),
        labels=("AWAY_FAVORITE", "BALANCED", "HOME_FAVORITE"),
    ).astype(str)
    rows = []
    for segment_type, column in (
        ("competition", "competition"),
        ("coverage", "domestic_poisson_coverage"),
        ("favorite_band", "favorite_band"),
    ):
        for (segment, model), group in values.groupby(
            [column, "model"], sort=True
        ):
            rows.append(
                {
                    "segment_type": segment_type,
                    "segment_value": str(segment),
                    "model": model,
                    "matches": len(group),
                    **_summary(
                        _probabilities(group),
                        group["actual_class"].to_numpy(int),
                    ),
                }
            )
    result = pd.DataFrame(rows)
    for baseline_model, suffix in (
        (CURRENT_AO, "ao"),
        (CURRENT_ML_BLEND, "current_ml"),
    ):
        baseline = result[result["model"].eq(baseline_model)][
            ["segment_type", "segment_value", "brier_1x2", "log_loss_1x2"]
        ].rename(
            columns={
                "brier_1x2": f"{suffix}_brier",
                "log_loss_1x2": f"{suffix}_log_loss",
            }
        )
        result = result.merge(
            baseline,
            on=["segment_type", "segment_value"],
            validate="many_to_one",
        )
        result[f"delta_brier_vs_{suffix}"] = (
            result["brier_1x2"] - result[f"{suffix}_brier"]
        )
        result[f"delta_log_loss_vs_{suffix}"] = (
            result["log_loss_1x2"] - result[f"{suffix}_log_loss"]
        )
    return result


def _dependency_uncertainty(
    unseen: pd.DataFrame, bootstrap_samples: int
) -> pd.DataFrame:
    candidate = unseen[unseen["model"].eq(ML_POISSON_ENSEMBLE)]
    rows = []
    for baseline_model in (CURRENT_ML_BLEND, CURRENT_AO):
        baseline = unseen[unseen["model"].eq(baseline_model)]
        merged = candidate.merge(
            baseline[["match_id", "brier_1x2", "log_loss_1x2"]],
            on="match_id",
            suffixes=("", "_baseline"),
            validate="one_to_one",
        )
        scopes: list[tuple[str, pd.DataFrame]] = [("ALL", merged)]
        scopes.extend(
            (f"competition:{value}", group)
            for value, group in merged.groupby("competition", sort=True)
        )
        scopes.extend(
            (f"coverage:{value}", group)
            for value, group in merged.groupby(
                "domestic_poisson_coverage", sort=True
            )
        )
        for scope, group in scopes:
            for metric in ("brier_1x2", "log_loss_1x2"):
                audit = group.copy()
                audit["loss_difference"] = (
                    audit[metric] - audit[f"{metric}_baseline"]
                )
                result = dependency_robust_loss_difference_ci(
                    audit,
                    bootstrap_samples=bootstrap_samples,
                    seed=20260813 + len(rows) * 1009,
                )
                result.insert(0, "baseline_model", baseline_model)
                result.insert(1, "candidate_model", ML_POISSON_ENSEMBLE)
                result.insert(2, "scope", scope)
                result.insert(3, "metric", metric)
                rows.append(result)
    return pd.concat(rows, ignore_index=True)


def _decision(
    comparison: pd.DataFrame,
    fold_results: pd.DataFrame,
    calibration: pd.DataFrame,
    uncertainty: pd.DataFrame,
    prospective_selection: Mapping[str, object],
    fold_selections: pd.DataFrame,
    holdout: str,
) -> dict[str, object]:
    candidate = comparison[comparison["model"].eq(ML_POISSON_ENSEMBLE)].iloc[0]
    current_ml = comparison[comparison["model"].eq(CURRENT_ML_BLEND)].iloc[0]
    ao = comparison[comparison["model"].eq(CURRENT_AO)].iloc[0]
    folds = fold_results[fold_results["model"].eq(ML_POISSON_ENSEMBLE)]
    brier_wins = int(folds["delta_brier_vs_current_ml"].lt(-1e-12).sum())
    log_wins = int(folds["delta_log_loss_vs_current_ml"].lt(-1e-12).sum())

    envelopes = uncertainty[
        uncertainty["method"].eq("conservative_envelope")
        & uncertainty["baseline_model"].eq(CURRENT_ML_BLEND)
    ]
    pooled = envelopes[envelopes["scope"].eq("ALL")]
    segment = envelopes[~envelopes["scope"].eq("ALL")]
    reliable_pooled_improvement = bool(pooled["reliable_improvement"].any())
    reliable_pooled_harm = bool(pooled["reliable_harm"].any())
    reliable_segment_harm = bool(segment["reliable_harm"].any())

    candidate_calibration = calibration[
        calibration["model"].eq(ML_POISSON_ENSEMBLE)
    ].iloc[0]
    ml_calibration = calibration[
        calibration["model"].eq(CURRENT_ML_BLEND)
    ].iloc[0]
    pooled_loss_gate = bool(
        candidate["brier_1x2"] < current_ml["brier_1x2"]
        and candidate["log_loss_1x2"] < current_ml["log_loss_1x2"]
    )
    gates = {
        "brier_fold_wins_vs_current_ml": brier_wins,
        "log_loss_fold_wins_vs_current_ml": log_wins,
        "fold_gate": brier_wins >= 4 and log_wins >= 4,
        "pooled_loss_gate_vs_current_ml": pooled_loss_gate,
        "uncertainty_gate": (
            reliable_pooled_improvement and not reliable_pooled_harm
        ),
        "competition_and_coverage_no_harm_gate": not reliable_segment_harm,
        "calibration_gate": bool(
            candidate_calibration["ece"] <= ml_calibration["ece"] + 1e-12
        ),
        "probability_gate": True,
        "rating_state_identity_gate": True,
        "beats_ao_brier": bool(candidate["brier_1x2"] < ao["brier_1x2"]),
        "beats_ao_log_loss": bool(
            candidate["log_loss_1x2"] < ao["log_loss_1x2"]
        ),
    }
    passed = all(
        bool(gates[key])
        for key in (
            "fold_gate",
            "pooled_loss_gate_vs_current_ml",
            "uncertainty_gate",
            "competition_and_coverage_no_harm_gate",
            "calibration_gate",
            "probability_gate",
            "rating_state_identity_gate",
        )
    )
    if reliable_pooled_harm or reliable_segment_harm:
        decision = "REJECT"
    elif passed:
        decision = "BEST_SHADOW_CANDIDATE"
    elif pooled_loss_gate:
        decision = "KEEP_SHADOW"
    else:
        decision = "KEEP_SEPARATE_MODELS"
    return {
        "decision": decision,
        "production_activated": False,
        "candidate_arm": ML_POISSON_ENSEMBLE,
        "pooled_brier": float(candidate["brier_1x2"]),
        "pooled_log_loss": float(candidate["log_loss_1x2"]),
        "pooled_accuracy": float(candidate["accuracy_1x2"]),
        "delta_brier_vs_current_ml": float(
            candidate["brier_1x2"] - current_ml["brier_1x2"]
        ),
        "delta_log_loss_vs_current_ml": float(
            candidate["log_loss_1x2"] - current_ml["log_loss_1x2"]
        ),
        "delta_accuracy_vs_current_ml": float(
            candidate["accuracy_1x2"] - current_ml["accuracy_1x2"]
        ),
        "delta_brier_vs_ao": float(
            candidate["brier_1x2"] - ao["brier_1x2"]
        ),
        "delta_log_loss_vs_ao": float(
            candidate["log_loss_1x2"] - ao["log_loss_1x2"]
        ),
        "gates": gates,
        "candidate_calibration": _calibration_dict(candidate_calibration),
        "current_ml_calibration": _calibration_dict(ml_calibration),
        "fold_poisson_sources": fold_selections[
            ["fold", "test_season", "poisson_source", "poisson_weight"]
        ].to_dict(orient="records"),
        "prospective_2026_27_selection": {
            "poisson_source": str(prospective_selection["poisson_source"]),
            "poisson_weight": float(prospective_selection["poisson_weight"]),
            "ml_weight": float(prospective_selection["ml_weight"]),
            "selection_data": "2020/21-2025/26 OOS development predictions",
            **(
                {
                    "weight_pinned": True,
                    "surface_would_have_selected": float(
                        prospective_selection["surface_would_have_selected"]
                    ),
                }
                if prospective_selection.get("weight_pinned")
                else {}
            ),
        },
        "untouched_holdout": holdout,
        "rating_feedback": False,
    }


def _validate_and_align_source_predictions(
    frame: pd.DataFrame, expected_seasons: Sequence[str]
) -> dict[str, pd.DataFrame]:
    required = {
        "fold",
        "model",
        "match_id",
        "season",
        "kickoff_utc",
        "competition",
        "stage",
        "actual_class",
        "baseline_power_delta",
        "domestic_poisson_coverage",
        *PROBABILITY_COLUMNS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Domestic prediction source missing columns: {missing}")
    required_models = (
        CURRENT_AO,
        CURRENT_ML_BLEND,
        AO_POISSON_BLEND,
        SOURCE_DOMESTIC_POISSON_RHO0,
    )
    result: dict[str, pd.DataFrame] = {}
    reference: pd.DataFrame | None = None
    for model in required_models:
        values = frame[frame["model"].eq(model)].copy()
        values = values.sort_values(["fold", "match_id"], kind="stable").reset_index(
            drop=True
        )
        if values.empty or values.duplicated(["fold", "match_id"]).any():
            raise ValueError(f"{model} must contain unique OOS match predictions")
        if set(values["season"].astype(str)) != set(expected_seasons):
            raise ValueError(f"{model} season coverage is invalid")
        _validate_probability_matrix(_probabilities(values), model)
        if reference is not None:
            _assert_same_match_order(reference, values)
            if not np.allclose(
                reference["baseline_power_delta"].to_numpy(float),
                values["baseline_power_delta"].to_numpy(float),
                atol=1e-12,
                rtol=0.0,
            ):
                raise ValueError("Prediction models changed the AO rating state")
        else:
            reference = values
        result[model] = values
    return result


def _selected_ml_rows(
    selections: pd.DataFrame, expected_seasons: Sequence[str]
) -> pd.DataFrame:
    required = {
        "fold",
        "test_season",
        "inner_validation_season",
        "arm",
        "selected_parameters",
        "selected_as_blend_source",
        "blend_weight",
    }
    missing = sorted(required - set(selections.columns))
    if missing:
        raise ValueError(f"ML selections missing columns: {missing}")
    selected = selections[
        selections["selected_as_blend_source"].map(_as_bool)
    ].copy()
    selected = selected.sort_values("fold", kind="stable").set_index("fold")
    _validate_fold_index(selected, expected_seasons, "ML selections")
    if selected["blend_weight"].isna().any():
        raise ValueError("Selected ML rows require blend weights")
    return selected


def _selection_rows(
    selections: pd.DataFrame,
    expected_seasons: Sequence[str],
    label: str,
) -> pd.DataFrame:
    if "fold" not in selections or "test_season" not in selections:
        raise ValueError(f"{label} missing fold/test_season columns")
    result = selections.sort_values("fold", kind="stable").set_index("fold")
    _validate_fold_index(result, expected_seasons, label)
    return result


def _validate_fold_index(
    frame: pd.DataFrame, expected_seasons: Sequence[str], label: str
) -> None:
    expected_folds = list(range(1, len(expected_seasons) + 1))
    if frame.index.tolist() != expected_folds:
        raise ValueError(f"{label} must contain folds {expected_folds}")
    if frame["test_season"].astype(str).tolist() != list(expected_seasons):
        raise ValueError(f"{label} test seasons are not chronological")


def _validate_unseen_predictions(
    unseen: pd.DataFrame, expected: pd.DataFrame
) -> None:
    counts = unseen.groupby("model")["match_id"].nunique()
    if set(counts.index) != set(ALL_MODELS) or counts.ne(len(expected)).any():
        raise ValueError("Every ensemble arm must predict every OOS match")
    if unseen.duplicated(["model", "match_id"]).any():
        raise ValueError("model+match_id must be unique")
    for _, group in unseen.groupby("model", sort=True):
        _validate_probability_matrix(_probabilities(group), str(group["model"].iloc[0]))


def _assert_same_match_order(left: pd.DataFrame, right: pd.DataFrame) -> None:
    if len(left) != len(right) or not np.array_equal(
        left["match_id"].astype(str).to_numpy(),
        right["match_id"].astype(str).to_numpy(),
    ):
        raise ValueError("Prediction sources do not share the same match order")
    if "actual_class" in left and "actual_class" in right and not np.array_equal(
        left["actual_class"].to_numpy(int), right["actual_class"].to_numpy(int)
    ):
        raise ValueError("Prediction sources disagree on match outcomes")


def _summary(probabilities: np.ndarray, outcomes: np.ndarray) -> dict[str, float]:
    losses = multiclass_losses(probabilities, outcomes)
    return {
        "brier_1x2": float(losses["brier_1x2"].mean()),
        "log_loss_1x2": float(losses["log_loss_1x2"].mean()),
        "accuracy_1x2": float(losses["correct"].mean()),
    }


def _probabilities(frame: pd.DataFrame) -> np.ndarray:
    return frame[list(PROBABILITY_COLUMNS)].to_numpy(float)


def _ao_probabilities(frame: pd.DataFrame) -> np.ndarray:
    values = frame[
        ["ao_home_probability", "ao_draw_probability", "ao_away_probability"]
    ].to_numpy(float)
    _validate_probability_matrix(values, "AO probabilities")
    return values


def _validate_probability_matrix(probabilities: np.ndarray, label: str) -> None:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"{label} must have shape (n,3)")
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError(f"{label} must be finite and non-negative")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError(f"{label} must sum to one")


def _calibration_dict(row: pd.Series) -> dict[str, float]:
    return {
        "ece": float(row["ece"]),
        "calibration_slope": float(row["calibration_slope"]),
        "calibration_intercept": float(row["calibration_intercept"]),
        "mean_max_probability": float(row["mean_max_probability"]),
    }


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")
