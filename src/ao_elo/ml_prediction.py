from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ao_elo.ml_features import FeatureSchema


MODEL_FAMILIES = {"LOGISTIC", "HIST_GRADIENT_BOOSTING"}
PROBABILITY_EPSILON = 1e-15


@dataclass
class TrainedML1X2:
    arm_name: str
    family: str
    schema: FeatureSchema
    parameters: dict[str, float | int]
    estimator: Pipeline

    @property
    def fingerprint(self) -> str:
        payload = {
            "arm_name": self.arm_name,
            "family": self.family,
            "numeric": self.schema.numeric,
            "categorical": self.schema.categorical,
            "parameters": self.parameters,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def fit_ml_1x2(
    frame: pd.DataFrame,
    *,
    arm_name: str,
    family: str,
    schema: FeatureSchema,
    parameters: Mapping[str, float | int],
) -> TrainedML1X2:
    if family not in MODEL_FAMILIES:
        raise ValueError(f"Unknown ML family: {family}")
    _validate_training_frame(frame, schema)
    preprocessor = _preprocessor(schema, dense=family == "HIST_GRADIENT_BOOSTING")
    if family == "LOGISTIC":
        estimator = LogisticRegression(
            solver="saga",
            C=float(parameters["C"]),
            l1_ratio=float(parameters["l1_ratio"]),
            max_iter=500,
            tol=2e-3,
            random_state=20260812,
        )
    else:
        estimator = HistGradientBoostingClassifier(
            learning_rate=float(parameters["learning_rate"]),
            max_leaf_nodes=int(parameters["max_leaf_nodes"]),
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            l2_regularization=float(parameters["l2_regularization"]),
            max_iter=250,
            early_stopping=False,
            random_state=20260812,
        )
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    pipeline.fit(frame[list(schema.columns)], frame["actual_class"].to_numpy(int))
    trained = TrainedML1X2(
        arm_name=arm_name,
        family=family,
        schema=schema,
        parameters={key: _json_number(value) for key, value in parameters.items()},
        estimator=pipeline,
    )
    probabilities = predict_ml_1x2(trained, frame)
    if probabilities.shape != (len(frame), 3):
        raise ValueError("Fitted model did not produce three 1X2 probabilities")
    return trained


def predict_ml_1x2(model: TrainedML1X2, frame: pd.DataFrame) -> np.ndarray:
    missing = sorted(set(model.schema.columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction frame missing features: {missing}")
    probabilities = np.asarray(
        model.estimator.predict_proba(frame[list(model.schema.columns)]), dtype=float
    )
    classes = np.asarray(model.estimator.named_steps["model"].classes_, dtype=int)
    if set(classes.tolist()) != {0, 1, 2}:
        raise ValueError(f"Model classes must be 0/1/2, got {classes.tolist()}")
    ordered = probabilities[:, np.argsort(classes)]
    _validate_probabilities(ordered)
    return ordered


def blend_probabilities(
    ao_probabilities: np.ndarray,
    ml_probabilities: np.ndarray,
    weight: float,
) -> np.ndarray:
    ao = np.asarray(ao_probabilities, dtype=float)
    ml = np.asarray(ml_probabilities, dtype=float)
    _validate_probabilities(ao)
    _validate_probabilities(ml)
    if ao.shape != ml.shape:
        raise ValueError("AO and ML probability matrices must have the same shape")
    if not math.isfinite(float(weight)) or not 0.0 <= float(weight) <= 1.0:
        raise ValueError("Blend weight must be in [0,1]")
    logits = (1.0 - float(weight)) * np.log(np.maximum(ao, PROBABILITY_EPSILON))
    logits += float(weight) * np.log(np.maximum(ml, PROBABILITY_EPSILON))
    logits -= logits.max(axis=1, keepdims=True)
    result = np.exp(logits)
    result /= result.sum(axis=1, keepdims=True)
    _validate_probabilities(result)
    return result


def multiclass_losses(probabilities: np.ndarray, outcomes: np.ndarray) -> pd.DataFrame:
    values = np.asarray(probabilities, dtype=float)
    targets = np.asarray(outcomes, dtype=int)
    _validate_probabilities(values)
    if targets.shape != (len(values),) or ((targets < 0) | (targets > 2)).any():
        raise ValueError("Outcomes must be a one-dimensional 0/1/2 array")
    one_hot = np.eye(3)[targets]
    observed = values[np.arange(len(values)), targets]
    return pd.DataFrame(
        {
            "brier_1x2": np.square(values - one_hot).sum(axis=1),
            "log_loss_1x2": -np.log(np.maximum(observed, PROBABILITY_EPSILON)),
            "predicted_class": values.argmax(axis=1),
            "correct": values.argmax(axis=1) == targets,
        }
    )


def calibration_metrics(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    *,
    bins: int = 10,
) -> dict[str, float]:
    values = np.asarray(probabilities, dtype=float)
    targets = np.asarray(outcomes, dtype=int)
    _validate_probabilities(values)
    confidence = values.max(axis=1)
    correct = (values.argmax(axis=1) == targets).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        mask = (confidence >= edges[index]) & (
            confidence <= edges[index + 1] if index == bins - 1 else confidence < edges[index + 1]
        )
        if mask.any():
            ece += float(mask.mean()) * abs(float(confidence[mask].mean() - correct[mask].mean()))

    slopes: list[float] = []
    intercepts: list[float] = []
    weights: list[int] = []
    for klass in range(3):
        y = (targets == klass).astype(int)
        if y.min() == y.max():
            continue
        p = np.clip(values[:, klass], 1e-8, 1.0 - 1e-8)
        logit = np.log(p / (1.0 - p)).reshape(-1, 1)
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        model.fit(logit, y)
        slopes.append(float(model.coef_[0, 0]))
        intercepts.append(float(model.intercept_[0]))
        weights.append(int(y.sum()))
    return {
        "ece": float(ece),
        "calibration_slope": float(np.average(slopes, weights=weights)),
        "calibration_intercept": float(np.average(intercepts, weights=weights)),
        "mean_max_probability": float(confidence.mean()),
    }


def raw_feature_importance(
    model: TrainedML1X2,
    frame: pd.DataFrame,
    *,
    max_rows: int = 1000,
    repeats: int = 3,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])
    sample = frame.sort_values(["kickoff_utc", "match_id"], kind="stable").tail(max_rows)
    result = permutation_importance(
        model.estimator,
        sample[list(model.schema.columns)],
        sample["actual_class"].to_numpy(int),
        scoring="neg_log_loss",
        n_repeats=repeats,
        random_state=20260812,
    )
    return (
        pd.DataFrame(
            {
                "feature": list(model.schema.columns),
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values(["importance_mean", "feature"], ascending=[False, True], kind="stable")
        .reset_index(drop=True)
    )


def mean_log_loss(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    _validate_probabilities(probabilities)
    return float(log_loss(outcomes, probabilities, labels=[0, 1, 2]))


def _preprocessor(schema: FeatureSchema, *, dense: bool) -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=not dense),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, list(schema.numeric)),
            ("categorical", categorical, list(schema.categorical)),
        ],
        sparse_threshold=0.0 if dense else 0.3,
    )


def _validate_training_frame(frame: pd.DataFrame, schema: FeatureSchema) -> None:
    missing = sorted((set(schema.columns) | {"actual_class"}) - set(frame.columns))
    if missing:
        raise ValueError(f"Training frame missing columns: {missing}")
    if frame.empty:
        raise ValueError("Training frame cannot be empty")
    targets = pd.to_numeric(frame["actual_class"], errors="coerce")
    if targets.isna().any() or set(targets.astype(int).unique()) != {0, 1, 2}:
        raise ValueError("Training target must contain all three 1X2 classes")


def _validate_probabilities(probabilities: np.ndarray) -> None:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("Probability matrix must have shape (n,3)")
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("Probabilities must be finite and non-negative")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("Probabilities must sum to one")


def _json_number(value: Any) -> float | int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Model parameters must be finite")
    return result
