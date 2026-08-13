from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass, replace
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from ao_elo.domestic_poisson import (
    DomesticPoissonConfig,
    EuropeanPoissonTransferConfig,
    build_domestic_poisson_feature_store,
    domestic_candidate_grid,
    fit_european_poisson_transfer,
    predict_european_poisson_transfer,
    select_dixon_coles_rho,
)
from ao_elo.evaluation import dependency_robust_loss_difference_ci
from ao_elo.ml_features import FEATURE_SCHEMAS, FeatureSchema, validate_feature_store
from ao_elo.ml_prediction import (
    blend_probabilities,
    calibration_metrics,
    fit_ml_1x2,
    multiclass_losses,
    predict_ml_1x2,
)


CURRENT_AO = "CURRENT_AO"
EXISTING_ELO_POISSON = "EXISTING_ELO_POISSON"
DOMESTIC_AD_RAW = "DOMESTIC_AD_RAW"
DOMESTIC_AD_RELIABLE = "DOMESTIC_AD_RELIABLE"
DOMESTIC_ATTACK_DEFENCE_POISSON = "DOMESTIC_ATTACK_DEFENCE_POISSON"
DOMESTIC_ATTACK_DEFENCE_DIXON_COLES = "DOMESTIC_ATTACK_DEFENCE_DIXON_COLES"
AO_POISSON_BLEND = "AO_POISSON_BLEND"
CURRENT_ML_BLEND = "CURRENT_ML_BLEND"
SOURCE_CURRENT_ML_BLEND = "AO_ML_BLEND"
POISSON_FEATURE_LOGISTIC = "POISSON_FEATURE_LOGISTIC"
AO_ML_POISSON_BLEND = "AO_ML_POISSON_BLEND"
ALL_MODELS = (
    CURRENT_AO,
    EXISTING_ELO_POISSON,
    DOMESTIC_AD_RAW,
    DOMESTIC_AD_RELIABLE,
    DOMESTIC_ATTACK_DEFENCE_POISSON,
    DOMESTIC_ATTACK_DEFENCE_DIXON_COLES,
    AO_POISSON_BLEND,
    CURRENT_ML_BLEND,
    POISSON_FEATURE_LOGISTIC,
    AO_ML_POISSON_BLEND,
)
SCORE_MODELS = (
    EXISTING_ELO_POISSON,
    DOMESTIC_ATTACK_DEFENCE_POISSON,
    DOMESTIC_ATTACK_DEFENCE_DIXON_COLES,
)
L2_GRID = (0.1, 1.0, 10.0)
BLEND_WEIGHTS = tuple(round(value, 1) for value in np.linspace(0.0, 1.0, 11))
LOGISTIC_GRID = tuple(
    {"C": c_value, "l1_ratio": ratio}
    for c_value in (0.03, 0.10, 0.30, 1.0, 3.0)
    for ratio in (0.0, 0.25, 0.50, 0.75)
)
POISSON_NUMERIC_FEATURES = (
    "poisson_lambda_home",
    "poisson_lambda_away",
    "poisson_expected_total_goals",
    "poisson_log_home_draw",
    "poisson_log_away_draw",
    "home_domestic_poisson_attack",
    "away_domestic_poisson_attack",
    "home_domestic_poisson_defence",
    "away_domestic_poisson_defence",
    "home_domestic_poisson_reliability",
    "away_domestic_poisson_reliability",
    "domestic_poisson_venue_edge",
    "home_domestic_poisson_effective_matches",
    "away_domestic_poisson_effective_matches",
)
POISSON_LOGISTIC_SCHEMA = FeatureSchema(
    numeric=tuple(dict.fromkeys(FEATURE_SCHEMAS["DOMESTIC_LOGISTIC"].numeric + POISSON_NUMERIC_FEATURES)),
    categorical=FEATURE_SCHEMAS["DOMESTIC_LOGISTIC"].categorical,
)


@dataclass
class DomesticPoissonBacktestResult:
    domestic_prequential_results: pd.DataFrame
    fold_domestic_selections: pd.DataFrame
    domestic_poisson_feature_store: pd.DataFrame
    fold_transfer_parameters: pd.DataFrame
    fold_results: pd.DataFrame
    unseen_predictions: pd.DataFrame
    model_comparison: pd.DataFrame
    feature_ablation: pd.DataFrame
    competition_coverage_summary: pd.DataFrame
    scoreline_diagnostics: pd.DataFrame
    dependency_uncertainty: pd.DataFrame
    selected_candidate: dict[str, object]


def run_domestic_poisson_walk_forward_backtest(
    feature_store: pd.DataFrame,
    domestic_matches: pd.DataFrame,
    bridge: pd.DataFrame,
    domestic_candidate_surface: pd.DataFrame,
    current_ml_predictions: pd.DataFrame,
    *,
    bootstrap_samples: int = 1000,
) -> DomesticPoissonBacktestResult:
    validate_feature_store(feature_store)
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    base = feature_store.sort_values(["kickoff_utc", "match_id"], kind="stable").reset_index(drop=True)
    seasons = tuple(dict.fromkeys(base["season"].astype(str)))
    if len(seasons) != 8 or seasons[-1] == "2026/27":
        raise ValueError("Expected 2018/19-2025/26 development seasons only")
    current_ml = _validate_current_ml_predictions(current_ml_predictions, seasons[2:])
    candidates = {candidate.key: candidate for candidate in domestic_candidate_grid()}
    _validate_domestic_surface(domestic_candidate_surface, candidates)

    feature_cache: dict[str, pd.DataFrame] = {}
    domestic_selection_rows: list[dict[str, object]] = []
    transfer_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    oos_feature_frames: list[pd.DataFrame] = []

    for fold, test_season in enumerate(seasons[2:], start=1):
        train_seasons = seasons[: seasons.index(test_season)]
        inner_validation_season = train_seasons[-1]
        inner_train_seasons = train_seasons[:-1]
        domestic_config = select_domestic_config(
            domestic_candidate_surface,
            candidates,
            inner_validation_season,
        )
        domestic_selection_rows.append(
            {
                "fold": fold,
                "test_season": test_season,
                "validation_season": inner_validation_season,
                **asdict(domestic_config),
                "candidate_key": domestic_config.key,
            }
        )
        enriched = _features_for_config(
            feature_cache, base, domestic_matches, bridge, domestic_config
        )
        inner_train = enriched[enriched["season"].isin(inner_train_seasons)].copy()
        inner_valid = enriched[enriched["season"].eq(inner_validation_season)].copy()
        outer_train = enriched[enriched["season"].isin(train_seasons)].copy()
        outer_test = enriched[enriched["season"].eq(test_season)].copy()
        if min(len(inner_train), len(inner_valid), len(outer_test)) == 0:
            raise ValueError(f"Fold {fold} contains an empty temporal split")

        selected = _select_transfer(inner_train, inner_valid, domestic_config)
        fitted = _fit_outer_models(outer_train, selected, domestic_config)
        predictions = _predict_fold_models(
            outer_test,
            inner_train,
            inner_valid,
            outer_train,
            fitted,
            selected,
            current_ml[current_ml["season"].eq(test_season)],
            fold,
        )
        prediction_frames.extend(predictions["frames"])
        transfer_rows.append(
            {
                "fold": fold,
                "test_season": test_season,
                "inner_validation_season": inner_validation_season,
                "domestic_candidate_key": domestic_config.key,
                **selected["audit"],
                "outer_transfer_config": json.dumps(
                    asdict(fitted["domestic"]), sort_keys=True
                ),
                "outer_existing_config": json.dumps(
                    asdict(fitted["existing"]), sort_keys=True
                ),
                "logistic_parameters": json.dumps(
                    predictions["logistic_parameters"], sort_keys=True
                ),
            }
        )
        feature_columns = [
            "match_id",
            "season",
            "kickoff_utc",
            "domestic_poisson_config",
            "domestic_poisson_coverage",
            *[column for column in enriched.columns if "domestic_poisson_" in column],
        ]
        fold_features = outer_test[list(dict.fromkeys(feature_columns))].copy()
        fold_features.insert(0, "fold", fold)
        oos_feature_frames.append(fold_features)

    unseen = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["fold", "match_id", "model"], kind="stable"
    )
    oos_features = pd.concat(oos_feature_frames, ignore_index=True).sort_values(
        ["fold", "kickoff_utc", "match_id"], kind="stable"
    )
    _validate_oos_predictions(unseen, base[base["season"].isin(seasons[2:])])
    fold_results = _fold_results(unseen)
    comparison = _model_comparison(unseen)
    ablation = _feature_ablation(comparison)
    segments = _segment_summary(unseen)
    scoreline = _scoreline_diagnostics(unseen)
    uncertainty = _dependency_uncertainty(unseen, bootstrap_samples)

    final_domestic = select_domestic_config(
        domestic_candidate_surface, candidates, seasons[-1]
    )
    final_enriched = _features_for_config(
        feature_cache, base, domestic_matches, bridge, final_domestic
    )
    final_inner_train = final_enriched[final_enriched["season"].isin(seasons[:-1])]
    final_inner_valid = final_enriched[final_enriched["season"].eq(seasons[-1])]
    final_selection = _select_transfer(
        final_inner_train, final_inner_valid, final_domestic
    )
    final_transfer = fit_european_poisson_transfer(
        final_enriched,
        l2_strength=float(final_selection["l2_strength"]),
        use_reliability=True,
        use_venue=final_domestic.venue_context,
    )
    final_transfer = replace(final_transfer, rho=float(final_selection["rho"]))
    decision = _decision(
        comparison,
        fold_results,
        segments,
        unseen,
        uncertainty,
        scoreline,
        final_domestic,
        final_transfer,
        final_selection,
    )
    return DomesticPoissonBacktestResult(
        domestic_prequential_results=domestic_candidate_surface.copy(),
        fold_domestic_selections=pd.DataFrame(domestic_selection_rows),
        domestic_poisson_feature_store=oos_features,
        fold_transfer_parameters=pd.DataFrame(transfer_rows),
        fold_results=fold_results,
        unseen_predictions=unseen,
        model_comparison=comparison,
        feature_ablation=ablation,
        competition_coverage_summary=segments,
        scoreline_diagnostics=scoreline,
        dependency_uncertainty=uncertainty,
        selected_candidate=decision,
    )


def select_domestic_config(
    surface: pd.DataFrame,
    candidates: Mapping[str, DomesticPoissonConfig],
    validation_season: str,
) -> DomesticPoissonConfig:
    rows = surface[surface["season"].astype(str).eq(str(validation_season))].copy()
    if rows.empty:
        prior = surface[
            surface["season"].map(_season_start) < _season_start(validation_season)
        ].copy()
        if prior.empty:
            raise ValueError(
                f"No historical domestic candidate evidence before {validation_season}"
            )
        rows = (
            prior.groupby("candidate_key", as_index=False)
            .agg(
                goal_nll=("goal_nll", "mean"),
                home_goal_bias=("home_goal_bias", "mean"),
                away_goal_bias=("away_goal_bias", "mean"),
            )
        )
    rows["absolute_goal_bias"] = (
        rows["home_goal_bias"].abs() + rows["away_goal_bias"].abs()
    )
    selected = rows.sort_values(
        ["goal_nll", "absolute_goal_bias", "candidate_key"], kind="stable"
    ).iloc[0]
    return candidates[str(selected["candidate_key"])]


def _features_for_config(
    cache: dict[str, pd.DataFrame],
    base: pd.DataFrame,
    domestic: pd.DataFrame,
    bridge: pd.DataFrame,
    config: DomesticPoissonConfig,
) -> pd.DataFrame:
    if config.key not in cache:
        features = build_domestic_poisson_feature_store(domestic, base, bridge, config)
        cache[config.key] = base.merge(features, on="match_id", validate="one_to_one")
    return cache[config.key]


def _select_transfer(
    inner_train: pd.DataFrame,
    inner_valid: pd.DataFrame,
    domestic_config: DomesticPoissonConfig,
) -> dict[str, object]:
    baseline = _probabilities(inner_valid, "ao")
    outcomes = inner_valid["actual_class"].to_numpy(int)
    surface = []
    fitted_configs: dict[float, EuropeanPoissonTransferConfig] = {}
    for l2_strength in L2_GRID:
        fitted = fit_european_poisson_transfer(
            inner_train,
            l2_strength=l2_strength,
            use_reliability=True,
            use_venue=domestic_config.venue_context,
        )
        fitted_configs[l2_strength] = fitted
        predicted = predict_european_poisson_transfer(inner_valid, fitted)
        probabilities = _probabilities(predicted, "prediction")
        metrics = _summary(probabilities, outcomes)
        surface.append(
            {
                "l2_strength": l2_strength,
                **metrics,
                "objective": _relative_objective(probabilities, baseline, outcomes),
            }
        )
    best_l2 = float(
        pd.DataFrame(surface)
        .sort_values(["objective", "log_loss_1x2", "l2_strength"], kind="stable")
        .iloc[0]["l2_strength"]
    )
    base_config = fitted_configs[best_l2]
    rho, rho_surface = select_dixon_coles_rho(inner_valid, base_config)
    dc_config = replace(base_config, rho=rho)
    poisson_probability = _probabilities(
        predict_european_poisson_transfer(inner_valid, dc_config), "prediction"
    )
    poisson_weight = _select_blend_weight(baseline, poisson_probability, outcomes)
    return {
        "l2_strength": best_l2,
        "rho": rho,
        "poisson_blend_weight": poisson_weight,
        "audit": {
            "selected_l2_strength": best_l2,
            "selected_rho": rho,
            "poisson_blend_weight": poisson_weight,
            "l2_surface": json.dumps(surface, sort_keys=True),
            "rho_surface": rho_surface.to_json(orient="records"),
        },
    }


def _fit_outer_models(
    outer_train: pd.DataFrame,
    selected: Mapping[str, object],
    domestic_config: DomesticPoissonConfig,
) -> dict[str, EuropeanPoissonTransferConfig]:
    existing = fit_european_poisson_transfer(
        _zero_domestic_features(outer_train),
        l2_strength=0.0,
        use_reliability=True,
        use_venue=False,
    )
    existing = replace(
        existing,
        attack_coefficient=0.0,
        defence_coefficient=0.0,
        venue_coefficient=0.0,
        rho=0.0,
    )
    raw = fit_european_poisson_transfer(
        outer_train,
        l2_strength=float(selected["l2_strength"]),
        use_reliability=False,
        use_venue=False,
    )
    reliable = fit_european_poisson_transfer(
        outer_train,
        l2_strength=float(selected["l2_strength"]),
        use_reliability=True,
        use_venue=False,
    )
    domestic = fit_european_poisson_transfer(
        outer_train,
        l2_strength=float(selected["l2_strength"]),
        use_reliability=True,
        use_venue=domestic_config.venue_context,
    )
    return {
        "existing": existing,
        "raw": raw,
        "reliable": reliable,
        "domestic": domestic,
        "dc": replace(domestic, rho=float(selected["rho"])),
    }


def _predict_fold_models(
    outer_test: pd.DataFrame,
    inner_train: pd.DataFrame,
    inner_valid: pd.DataFrame,
    outer_train: pd.DataFrame,
    fitted: Mapping[str, EuropeanPoissonTransferConfig],
    selected: Mapping[str, object],
    current_ml: pd.DataFrame,
    fold: int,
) -> dict[str, object]:
    ao = _probabilities(outer_test, "ao")
    frames = [_prediction_frame(outer_test, ao, CURRENT_AO, fold)]
    predicted: dict[str, pd.DataFrame] = {}
    for key, model in (
        (EXISTING_ELO_POISSON, fitted["existing"]),
        (DOMESTIC_AD_RAW, fitted["raw"]),
        (DOMESTIC_AD_RELIABLE, fitted["reliable"]),
        (DOMESTIC_ATTACK_DEFENCE_POISSON, fitted["domestic"]),
        (DOMESTIC_ATTACK_DEFENCE_DIXON_COLES, fitted["dc"]),
    ):
        prediction = predict_european_poisson_transfer(
            outer_test,
            model,
            fallback_to_ao_without_history=key != EXISTING_ELO_POISSON,
        )
        predicted[key] = prediction
        frames.append(
            _prediction_frame(
                outer_test,
                _probabilities(prediction, "prediction"),
                key,
                fold,
                score_prediction=prediction,
            )
        )
    dc_probability = _probabilities(
        predicted[DOMESTIC_ATTACK_DEFENCE_DIXON_COLES], "prediction"
    )
    frames.append(
        _prediction_frame(
            outer_test,
            blend_probabilities(ao, dc_probability, float(selected["poisson_blend_weight"])),
            AO_POISSON_BLEND,
            fold,
        )
    )
    current_ml_aligned = outer_test[["match_id"]].merge(
        current_ml[
            ["match_id", "home_probability", "draw_probability", "away_probability"]
        ],
        on="match_id",
        how="left",
        validate="one_to_one",
    )
    if current_ml_aligned.isna().any().any():
        raise ValueError(f"Current ML predictions do not cover fold {fold}")
    frames.append(
        _prediction_frame(
            outer_test,
            _probabilities(current_ml_aligned, "prediction"),
            CURRENT_ML_BLEND,
            fold,
        )
    )

    inner_model = fit_european_poisson_transfer(
        inner_train,
        l2_strength=float(selected["l2_strength"]),
        use_reliability=True,
        use_venue=fitted["domestic"].use_venue,
    )
    inner_model = replace(inner_model, rho=float(selected["rho"]))
    inner_train_augmented = _cross_fitted_poisson_features(
        inner_train,
        l2_strength=float(selected["l2_strength"]),
        rho=float(selected["rho"]),
        use_venue=fitted["domestic"].use_venue,
    )
    inner_valid_augmented = _augment_poisson_features(
        inner_valid,
        predict_european_poisson_transfer(inner_valid, inner_model),
    )
    if inner_train_augmented.empty:
        logistic_parameters = {"C": 0.03, "l1_ratio": 0.0}
        logistic_blend_weight = 0.0
    else:
        logistic_parameters = _select_logistic_parameters(
            inner_train_augmented, inner_valid_augmented
        )
        inner_logistic = _fit_logistic(inner_train_augmented, logistic_parameters)
        inner_logistic_probability = predict_ml_1x2(
            inner_logistic, inner_valid_augmented
        )
        logistic_blend_weight = _select_blend_weight(
            _probabilities(inner_valid, "ao"),
            inner_logistic_probability,
            inner_valid["actual_class"].to_numpy(int),
        )
    outer_train_augmented = _cross_fitted_poisson_features(
        outer_train,
        l2_strength=float(selected["l2_strength"]),
        rho=float(selected["rho"]),
        use_venue=fitted["domestic"].use_venue,
    )
    outer_test_augmented = _augment_poisson_features(
        outer_test,
        predicted[DOMESTIC_ATTACK_DEFENCE_DIXON_COLES],
    )
    if outer_train_augmented.empty:
        raise ValueError(f"Fold {fold} has no causal Poisson features for logistic fitting")
    logistic = _fit_logistic(outer_train_augmented, logistic_parameters)
    logistic_probability = predict_ml_1x2(logistic, outer_test_augmented)
    frames.append(
        _prediction_frame(
            outer_test, logistic_probability, POISSON_FEATURE_LOGISTIC, fold
        )
    )
    frames.append(
        _prediction_frame(
            outer_test,
            blend_probabilities(ao, logistic_probability, logistic_blend_weight),
            AO_ML_POISSON_BLEND,
            fold,
        )
    )
    logistic_parameters = {
        **logistic_parameters,
        "blend_weight": logistic_blend_weight,
    }
    return {"frames": frames, "logistic_parameters": logistic_parameters}


def _cross_fitted_poisson_features(
    frame: pd.DataFrame,
    *,
    l2_strength: float,
    rho: float,
    use_venue: bool,
) -> pd.DataFrame:
    seasons = tuple(dict.fromkeys(frame["season"].astype(str)))
    rows: list[pd.DataFrame] = []
    for index in range(1, len(seasons)):
        train = frame[frame["season"].isin(seasons[:index])]
        validation = frame[frame["season"].eq(seasons[index])]
        if train.empty or validation.empty:
            continue
        fitted = fit_european_poisson_transfer(
            train,
            l2_strength=l2_strength,
            use_reliability=True,
            use_venue=use_venue,
        )
        fitted = replace(fitted, rho=rho)
        prediction = predict_european_poisson_transfer(validation, fitted)
        rows.append(_augment_poisson_features(validation, prediction))
    if not rows:
        return pd.DataFrame(columns=frame.columns)
    return pd.concat(rows, ignore_index=True).sort_values(
        ["kickoff_utc", "match_id"], kind="stable"
    )


def _select_logistic_parameters(
    train: pd.DataFrame, validation: pd.DataFrame
) -> dict[str, float]:
    ao = _probabilities(validation, "ao")
    outcomes = validation["actual_class"].to_numpy(int)
    rows = []
    for parameters in LOGISTIC_GRID:
        model = _fit_logistic(train, parameters)
        probabilities = predict_ml_1x2(model, validation)
        metrics = _summary(probabilities, outcomes)
        rows.append(
            {
                **parameters,
                **metrics,
                "objective": _relative_objective(probabilities, ao, outcomes),
            }
        )
    selected = pd.DataFrame(rows).sort_values(
        ["objective", "log_loss_1x2", "C", "l1_ratio"], kind="stable"
    ).iloc[0]
    return {"C": float(selected["C"]), "l1_ratio": float(selected["l1_ratio"])}


def _fit_logistic(frame: pd.DataFrame, parameters: Mapping[str, float]):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return fit_ml_1x2(
            frame,
            arm_name=POISSON_FEATURE_LOGISTIC,
            family="LOGISTIC",
            schema=POISSON_LOGISTIC_SCHEMA,
            parameters=parameters,
        )


def _augment_poisson_features(
    frame: pd.DataFrame, prediction: pd.DataFrame
) -> pd.DataFrame:
    result = frame.merge(
        prediction[
            [
                "match_id",
                "lambda_home",
                "lambda_away",
                "home_probability",
                "draw_probability",
                "away_probability",
            ]
        ],
        on="match_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_poisson"),
    )
    epsilon = 1e-12
    result["poisson_lambda_home"] = result["lambda_home"]
    result["poisson_lambda_away"] = result["lambda_away"]
    result["poisson_expected_total_goals"] = result["lambda_home"] + result["lambda_away"]
    result["poisson_log_home_draw"] = np.log(
        np.maximum(result["home_probability"], epsilon)
        / np.maximum(result["draw_probability"], epsilon)
    )
    result["poisson_log_away_draw"] = np.log(
        np.maximum(result["away_probability"], epsilon)
        / np.maximum(result["draw_probability"], epsilon)
    )
    return result


def _prediction_frame(
    source: pd.DataFrame,
    probabilities: np.ndarray,
    model: str,
    fold: int,
    *,
    score_prediction: pd.DataFrame | None = None,
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
        "home_goals",
        "away_goals",
        "actual_class",
        "is_single_match_tie",
        "expected_home_score",
        "baseline_power_delta",
        "domestic_poisson_coverage",
    ]
    result = source[columns].reset_index(drop=True).copy()
    result.insert(0, "fold", fold)
    result.insert(1, "model", model)
    result["home_probability"] = probabilities[:, 0]
    result["draw_probability"] = probabilities[:, 1]
    result["away_probability"] = probabilities[:, 2]
    result = pd.concat([result, losses], axis=1)
    for column in (
        "lambda_home",
        "lambda_away",
        "exact_score_probability",
        "expected_total_goals",
        "over_2_5_probability",
        "btts_probability",
        "covered_probability_mass",
        "ao_fallback",
    ):
        result[column] = (
            score_prediction[column].to_numpy()
            if score_prediction is not None
            else np.nan
        )
    return result


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
    for model in ALL_MODELS:
        group = unseen[unseen["model"].eq(model)]
        rows.append(
            {
                "model": model,
                "matches": len(group),
                **_summary(
                    _probabilities(group, "prediction"),
                    group["actual_class"].to_numpy(int),
                ),
            }
        )
    result = pd.DataFrame(rows)
    baseline = result[result["model"].eq(CURRENT_AO)].iloc[0]
    for metric in ("brier_1x2", "log_loss_1x2", "accuracy_1x2"):
        result[f"delta_vs_ao_{metric}"] = result[metric] - float(baseline[metric])
    return result


def _feature_ablation(comparison: pd.DataFrame) -> pd.DataFrame:
    layer = {
        CURRENT_AO: "AO only",
        EXISTING_ELO_POISSON: "+ Elo-conditioned Poisson",
        DOMESTIC_AD_RAW: "+ raw domestic attack/defence",
        DOMESTIC_AD_RELIABLE: "+ reliability shrinkage",
        DOMESTIC_ATTACK_DEFENCE_POISSON: "+ team venue context if selected",
        DOMESTIC_ATTACK_DEFENCE_DIXON_COLES: "+ Dixon-Coles",
        AO_POISSON_BLEND: "+ AO/Poisson blend",
        CURRENT_ML_BLEND: "current ML shadow reference",
        POISSON_FEATURE_LOGISTIC: "+ multinomial logistic transfer",
        AO_ML_POISSON_BLEND: "+ AO anchored ML/Poisson blend",
    }
    result = comparison.copy()
    result.insert(1, "feature_layer", result["model"].map(layer))
    return result


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
        for (segment, model), group in values.groupby([column, "model"], sort=True):
            rows.append(
                {
                    "segment_type": segment_type,
                    "segment_value": str(segment),
                    "model": model,
                    "matches": len(group),
                    **_summary(
                        _probabilities(group, "prediction"),
                        group["actual_class"].to_numpy(int),
                    ),
                }
            )
    result = pd.DataFrame(rows)
    baseline = result[result["model"].eq(CURRENT_AO)][
        ["segment_type", "segment_value", "brier_1x2", "log_loss_1x2"]
    ].rename(columns={"brier_1x2": "baseline_brier", "log_loss_1x2": "baseline_log_loss"})
    result = result.merge(
        baseline, on=["segment_type", "segment_value"], validate="many_to_one"
    )
    result["delta_brier_vs_ao"] = result["brier_1x2"] - result["baseline_brier"]
    result["delta_log_loss_vs_ao"] = result["log_loss_1x2"] - result["baseline_log_loss"]
    return result


def _scoreline_diagnostics(unseen: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in SCORE_MODELS:
        group = unseen[unseen["model"].eq(model)].copy()
        actual_total = group["home_goals"] + group["away_goals"]
        over_target = actual_total.ge(3).astype(float)
        btts_target = (
            group["home_goals"].gt(0) & group["away_goals"].gt(0)
        ).astype(float)
        rows.append(
            {
                "model": model,
                "matches": len(group),
                "exact_score_nll": float(
                    -np.log(np.clip(group["exact_score_probability"], 1e-15, 1.0)).mean()
                ),
                "total_goals_mae": float(
                    np.abs(group["expected_total_goals"] - actual_total).mean()
                ),
                "total_goals_bias": float(
                    (group["expected_total_goals"] - actual_total).mean()
                ),
                "over_2_5_brier": float(
                    np.square(group["over_2_5_probability"] - over_target).mean()
                ),
                "over_2_5_log_loss": _binary_log_loss(
                    group["over_2_5_probability"], over_target
                ),
                "btts_brier": float(
                    np.square(group["btts_probability"] - btts_target).mean()
                ),
                "btts_log_loss": _binary_log_loss(
                    group["btts_probability"], btts_target
                ),
            }
        )
    return pd.DataFrame(rows)


def _dependency_uncertainty(
    unseen: pd.DataFrame, bootstrap_samples: int
) -> pd.DataFrame:
    baseline = unseen[unseen["model"].eq(CURRENT_AO)]
    rows = []
    for candidate_model in (AO_POISSON_BLEND, AO_ML_POISSON_BLEND):
        candidate = unseen[unseen["model"].eq(candidate_model)]
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
            for value, group in merged.groupby("domestic_poisson_coverage", sort=True)
        )
        for scope, group in scopes:
            for metric in ("brier_1x2", "log_loss_1x2"):
                audit = group.copy()
                audit["loss_difference"] = audit[metric] - audit[f"{metric}_baseline"]
                result = dependency_robust_loss_difference_ci(
                    audit,
                    bootstrap_samples=bootstrap_samples,
                    seed=20260813 + len(rows) * 1009,
                )
                result.insert(0, "candidate_model", candidate_model)
                result.insert(1, "scope", scope)
                result.insert(2, "metric", metric)
                rows.append(result)
    return pd.concat(rows, ignore_index=True)


def _decision(
    comparison: pd.DataFrame,
    fold_results: pd.DataFrame,
    segments: pd.DataFrame,
    unseen: pd.DataFrame,
    uncertainty: pd.DataFrame,
    scoreline: pd.DataFrame,
    final_domestic: DomesticPoissonConfig,
    final_transfer: EuropeanPoissonTransferConfig,
    final_selection: Mapping[str, object],
) -> dict[str, object]:
    baseline = comparison[comparison["model"].eq(CURRENT_AO)].iloc[0]
    candidate = comparison[comparison["model"].eq(AO_POISSON_BLEND)].iloc[0]
    current_ml = comparison[comparison["model"].eq(CURRENT_ML_BLEND)].iloc[0]
    folds = fold_results[fold_results["model"].eq(AO_POISSON_BLEND)]
    brier_wins = int(folds["delta_brier_vs_ao"].lt(-1e-12).sum())
    log_wins = int(folds["delta_log_loss_vs_ao"].lt(-1e-12).sum())
    candidate_calibration = calibration_metrics(
        _probabilities(unseen[unseen["model"].eq(AO_POISSON_BLEND)], "prediction"),
        unseen[unseen["model"].eq(AO_POISSON_BLEND)]["actual_class"].to_numpy(int),
    )
    baseline_calibration = calibration_metrics(
        _probabilities(unseen[unseen["model"].eq(CURRENT_AO)], "prediction"),
        unseen[unseen["model"].eq(CURRENT_AO)]["actual_class"].to_numpy(int),
    )
    envelopes = uncertainty[
        uncertainty["method"].eq("conservative_envelope")
        & uncertainty["candidate_model"].eq(AO_POISSON_BLEND)
    ]
    all_envelope = envelopes[envelopes["scope"].eq("ALL")]
    reliable_improvement = bool(all_envelope["reliable_improvement"].any())
    reliable_harm = bool(envelopes["reliable_harm"].any())
    score_lookup = scoreline.set_index("model")
    score_improved = bool(
        score_lookup.loc[DOMESTIC_ATTACK_DEFENCE_DIXON_COLES, "exact_score_nll"]
        < score_lookup.loc[EXISTING_ELO_POISSON, "exact_score_nll"]
    )
    gates = {
        "brier_fold_wins": brier_wins,
        "log_loss_fold_wins": log_wins,
        "fold_gate": brier_wins >= 4 and log_wins >= 4,
        "pooled_loss_gate": bool(
            candidate["brier_1x2"] < baseline["brier_1x2"]
            and candidate["log_loss_1x2"] < baseline["log_loss_1x2"]
        ),
        "uncertainty_gate": reliable_improvement and not reliable_harm,
        "segment_no_harm_gate": not reliable_harm,
        "calibration_gate": bool(
            candidate_calibration["ece"] <= baseline_calibration["ece"] + 1e-12
        ),
        "probability_gate": True,
        "rating_state_identity_gate": True,
        "scoreline_improvement": score_improved,
        "beats_current_ml_brier": bool(
            candidate["brier_1x2"] < current_ml["brier_1x2"]
        ),
        "beats_current_ml_log_loss": bool(
            candidate["log_loss_1x2"] < current_ml["log_loss_1x2"]
        ),
    }
    promotion = all(
        bool(gates[key])
        for key in (
            "fold_gate",
            "pooled_loss_gate",
            "uncertainty_gate",
            "segment_no_harm_gate",
            "calibration_gate",
            "probability_gate",
            "rating_state_identity_gate",
        )
    )
    if reliable_harm:
        decision = "REJECT"
    elif promotion and gates["beats_current_ml_brier"] and gates["beats_current_ml_log_loss"]:
        decision = "BEST_SHADOW_CANDIDATE"
    elif promotion:
        decision = "PROMOTE_CANDIDATE"
    elif score_improved:
        decision = "KEEP_DIAGNOSTIC"
    else:
        decision = "KEEP_SHADOW"
    return {
        "decision": decision,
        "production_activated": False,
        "candidate_arm": AO_POISSON_BLEND,
        "pooled_brier": float(candidate["brier_1x2"]),
        "pooled_log_loss": float(candidate["log_loss_1x2"]),
        "delta_brier_vs_ao": float(candidate["brier_1x2"] - baseline["brier_1x2"]),
        "delta_log_loss_vs_ao": float(
            candidate["log_loss_1x2"] - baseline["log_loss_1x2"]
        ),
        "delta_brier_vs_current_ml": float(
            candidate["brier_1x2"] - current_ml["brier_1x2"]
        ),
        "delta_log_loss_vs_current_ml": float(
            candidate["log_loss_1x2"] - current_ml["log_loss_1x2"]
        ),
        "gates": gates,
        "baseline_calibration": baseline_calibration,
        "candidate_calibration": candidate_calibration,
        "selected_domestic_config": asdict(final_domestic),
        "selected_transfer_config": asdict(final_transfer),
        "selected_poisson_blend_weight": float(final_selection["poisson_blend_weight"]),
        "untouched_holdout": "2026/27",
        "rating_feedback": False,
    }


def _select_blend_weight(
    ao: np.ndarray, challenger: np.ndarray, outcomes: np.ndarray
) -> float:
    return float(
        min(
            BLEND_WEIGHTS,
            key=lambda weight: (
                _relative_objective(
                    blend_probabilities(ao, challenger, weight), ao, outcomes
                ),
                weight,
            ),
        )
    )


def _relative_objective(
    probabilities: np.ndarray, ao: np.ndarray, outcomes: np.ndarray
) -> float:
    metrics = _summary(probabilities, outcomes)
    baseline = _summary(ao, outcomes)
    return 0.5 * (
        metrics["brier_1x2"] / baseline["brier_1x2"]
        + metrics["log_loss_1x2"] / baseline["log_loss_1x2"]
    )


def _summary(probabilities: np.ndarray, outcomes: np.ndarray) -> dict[str, float]:
    losses = multiclass_losses(probabilities, outcomes)
    return {
        "brier_1x2": float(losses["brier_1x2"].mean()),
        "log_loss_1x2": float(losses["log_loss_1x2"].mean()),
        "accuracy_1x2": float(losses["correct"].mean()),
    }


def _probabilities(frame: pd.DataFrame, source: str) -> np.ndarray:
    prefix = "ao_" if source == "ao" else ""
    values = frame[
        [
            f"{prefix}home_probability",
            f"{prefix}draw_probability",
            f"{prefix}away_probability",
        ]
    ].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0.0).any() or not np.allclose(
        values.sum(axis=1), 1.0, atol=1e-10
    ):
        raise ValueError("Probability matrix must be finite, non-negative, and normalized")
    return values


def _zero_domestic_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if "domestic_poisson_" in column and column != "domestic_poisson_coverage":
            result[column] = 0.0
    result["domestic_poisson_coverage"] = "BOTH"
    return result


def _validate_current_ml_predictions(
    frame: pd.DataFrame, expected_seasons: Sequence[str]
) -> pd.DataFrame:
    required = {
        "match_id",
        "season",
        "model",
        "home_probability",
        "draw_probability",
        "away_probability",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"current ML predictions missing columns: {missing}")
    result = frame[frame["model"].eq(SOURCE_CURRENT_ML_BLEND)].copy()
    if result["match_id"].duplicated().any() or set(result["season"].astype(str)) != set(expected_seasons):
        raise ValueError("Current ML OOS prediction identity/season contract failed")
    _probabilities(result, "prediction")
    return result


def _validate_domestic_surface(
    surface: pd.DataFrame, candidates: Mapping[str, DomesticPoissonConfig]
) -> None:
    required = {
        "candidate_key",
        "season",
        "matches",
        "goal_nll",
        "home_goal_bias",
        "away_goal_bias",
    }
    missing = sorted(required - set(surface.columns))
    if missing:
        raise ValueError(f"domestic candidate surface missing columns: {missing}")
    if set(surface["candidate_key"].unique()) != set(candidates):
        raise ValueError("Domestic candidate surface must cover all 54 candidates")
    if surface.duplicated(["candidate_key", "season"]).any():
        raise ValueError("Domestic candidate surface candidate+season must be unique")


def _validate_oos_predictions(unseen: pd.DataFrame, expected: pd.DataFrame) -> None:
    expected_matches = int(len(expected))
    counts = unseen.groupby("model")["match_id"].nunique()
    if set(counts.index) != set(ALL_MODELS) or counts.ne(expected_matches).any():
        raise ValueError("Every model must predict every unseen match exactly once")
    if unseen.duplicated(["model", "match_id"]).any():
        raise ValueError("OOS model+match_id must be unique")
    if set(unseen["match_id"].unique()) != set(expected["match_id"]):
        raise ValueError("OOS match identity does not match expected development window")
    for _, group in unseen.groupby("model"):
        _probabilities(group, "prediction")


def _binary_log_loss(probability: pd.Series, target: pd.Series) -> float:
    values = np.clip(probability.to_numpy(float), 1e-15, 1.0 - 1e-15)
    observed = target.to_numpy(float)
    return float(-(observed * np.log(values) + (1.0 - observed) * np.log(1.0 - values)).mean())


def _season_start(season: str) -> int:
    return int(str(season).replace("-", "/").split("/", 1)[0])
