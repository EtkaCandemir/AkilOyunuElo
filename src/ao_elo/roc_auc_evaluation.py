from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


CLASS_LABELS: Mapping[int, str] = {0: "HOME", 1: "DRAW", 2: "AWAY"}
PROBABILITY_COLUMNS = (
    "home_probability",
    "draw_probability",
    "away_probability",
)
REQUIRED_COLUMNS = {
    "fold",
    "model",
    "match_id",
    "season",
    "kickoff_utc",
    "competition",
    "actual_class",
    *PROBABILITY_COLUMNS,
}


@dataclass(frozen=True)
class RocAucEvaluationResult:
    model_summary: pd.DataFrame
    class_summary: pd.DataFrame
    fold_summary: pd.DataFrame
    competition_summary: pd.DataFrame
    paired_uncertainty: pd.DataFrame
    roc_curves: pd.DataFrame
    data_audit: dict[str, object]


def evaluate_multiclass_roc_auc(
    predictions: pd.DataFrame,
    *,
    models: Sequence[str],
    candidate_model: str,
    baselines: Sequence[str],
    bootstrap_samples: int = 4000,
    random_seed: int = 20260817,
) -> RocAucEvaluationResult:
    """Evaluate 1X2 discrimination without changing model probabilities.

    Multiclass ROC-AUC is reported with one-vs-rest class curves plus macro,
    weighted, and micro averages. Paired uncertainty resamples calendar-month
    clusters so every compared model receives the same sampled matches.
    """

    data = validate_roc_auc_predictions(predictions, models=models)
    if candidate_model not in models:
        raise ValueError("candidate_model must be included in models")
    if not baselines or any(model not in models for model in baselines):
        raise ValueError("Every baseline must be included in models")
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")

    model_summary = _model_summary(data, models)
    class_summary = _class_summary(data, models)
    fold_summary = _segment_summary(data, models, "fold", include_season=True)
    competition_summary = _segment_summary(data, models, "competition")
    paired_uncertainty = _paired_cluster_uncertainty(
        data,
        candidate_model=candidate_model,
        baselines=baselines,
        bootstrap_samples=bootstrap_samples,
        random_seed=random_seed,
    )
    curves = _roc_curves(data, models)
    unique = data[data["model"].eq(models[0])]
    class_counts = unique["actual_class"].value_counts().sort_index()
    audit = {
        "matches": int(len(unique)),
        "models": list(models),
        "seasons": sorted(unique["season"].astype(str).unique().tolist()),
        "competitions": sorted(unique["competition"].astype(str).unique().tolist()),
        "class_counts": {
            CLASS_LABELS[index]: int(class_counts.get(index, 0))
            for index in CLASS_LABELS
        },
        "bootstrap_unit": "SEASON_CALENDAR_MONTH",
        "bootstrap_samples": int(bootstrap_samples),
        "random_seed": int(random_seed),
        "probabilities_normalized": True,
        "model_match_alignment": True,
        "contains_2026_27": bool(unique["season"].astype(str).eq("2026/27").any()),
    }
    return RocAucEvaluationResult(
        model_summary=model_summary,
        class_summary=class_summary,
        fold_summary=fold_summary,
        competition_summary=competition_summary,
        paired_uncertainty=paired_uncertainty,
        roc_curves=curves,
        data_audit=audit,
    )


def validate_roc_auc_predictions(
    predictions: pd.DataFrame,
    *,
    models: Sequence[str],
) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS - set(predictions.columns))
    if missing:
        raise ValueError(f"ROC-AUC predictions missing columns: {missing}")
    if not models or len(set(models)) != len(models):
        raise ValueError("models must be a non-empty unique sequence")

    data = predictions[predictions["model"].isin(models)].copy()
    observed_models = set(data["model"].astype(str).unique())
    missing_models = sorted(set(models) - observed_models)
    if missing_models:
        raise ValueError(f"ROC-AUC predictions missing models: {missing_models}")
    if data.duplicated(["model", "match_id"]).any():
        raise ValueError("Each model must contain one row per match_id")

    data["kickoff_utc"] = pd.to_datetime(data["kickoff_utc"], utc=True, errors="coerce")
    if data["kickoff_utc"].isna().any():
        raise ValueError("kickoff_utc contains invalid timestamps")
    probabilities = data[list(PROBABILITY_COLUMNS)].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(float)
    if not np.isfinite(probabilities).all() or (probabilities < 0.0).any():
        raise ValueError("ROC-AUC probabilities must be finite and non-negative")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("ROC-AUC probabilities must sum to one")
    if not data["actual_class"].isin(CLASS_LABELS).all():
        raise ValueError("actual_class must contain only 0, 1, or 2")

    reference = (
        data[data["model"].eq(models[0])]
        .sort_values("match_id", kind="stable")
        .reset_index(drop=True)
    )
    if set(reference["actual_class"].astype(int)) != set(CLASS_LABELS):
        raise ValueError("Reference model must contain all three outcome classes")
    reference_keys = reference[
        ["match_id", "actual_class", "season", "kickoff_utc", "competition"]
    ]
    for model in models[1:]:
        candidate = (
            data[data["model"].eq(model)]
            .sort_values("match_id", kind="stable")
            .reset_index(drop=True)
        )
        if len(candidate) != len(reference):
            raise ValueError(f"Model {model} does not have the reference row count")
        candidate_keys = candidate[
            ["match_id", "actual_class", "season", "kickoff_utc", "competition"]
        ]
        if not reference_keys.equals(candidate_keys):
            raise ValueError(f"Model {model} does not align with reference matches")

    return data.sort_values(["model", "kickoff_utc", "match_id"], kind="stable").reset_index(
        drop=True
    )


def _model_summary(data: pd.DataFrame, models: Sequence[str]) -> pd.DataFrame:
    rows = []
    for model in models:
        frame = _model_frame(data, model)
        metrics = _auc_metrics(frame)
        rows.append({"model": model, "matches": len(frame), **metrics})
    return pd.DataFrame(rows)


def _class_summary(data: pd.DataFrame, models: Sequence[str]) -> pd.DataFrame:
    rows = []
    for model in models:
        frame = _model_frame(data, model)
        outcomes = frame["actual_class"].to_numpy(int)
        probabilities = frame[list(PROBABILITY_COLUMNS)].to_numpy(float)
        for index, label in CLASS_LABELS.items():
            binary = (outcomes == index).astype(int)
            rows.append(
                {
                    "model": model,
                    "class_index": index,
                    "class_label": label,
                    "positives": int(binary.sum()),
                    "prevalence": float(binary.mean()),
                    "roc_auc_ovr": float(roc_auc_score(binary, probabilities[:, index])),
                }
            )
    return pd.DataFrame(rows)


def _segment_summary(
    data: pd.DataFrame,
    models: Sequence[str],
    segment: str,
    *,
    include_season: bool = False,
) -> pd.DataFrame:
    rows = []
    for segment_value in sorted(data[segment].unique()):
        for model in models:
            frame = _model_frame(
                data[data[segment].eq(segment_value)], model
            )
            if set(frame["actual_class"].astype(int)) != set(CLASS_LABELS):
                continue
            row = {
                segment: segment_value,
                "model": model,
                "matches": len(frame),
                **_auc_metrics(frame),
            }
            if include_season:
                seasons = frame["season"].astype(str).unique()
                row["test_season"] = seasons[0] if len(seasons) == 1 else "MULTIPLE"
            rows.append(row)
    columns = [segment]
    if include_season:
        columns.append("test_season")
    columns += [
        "model",
        "matches",
        "home_roc_auc_ovr",
        "draw_roc_auc_ovr",
        "away_roc_auc_ovr",
        "macro_roc_auc_ovr",
        "weighted_roc_auc_ovr",
        "micro_roc_auc_ovr",
        "macro_roc_auc_ovo",
    ]
    return pd.DataFrame(rows)[columns]


def _auc_metrics(frame: pd.DataFrame) -> dict[str, float]:
    outcomes = frame["actual_class"].to_numpy(int)
    probabilities = frame[list(PROBABILITY_COLUMNS)].to_numpy(float)
    one_hot = np.eye(3, dtype=int)[outcomes]
    class_auc = [
        float(roc_auc_score(one_hot[:, index], probabilities[:, index]))
        for index in CLASS_LABELS
    ]
    return {
        "home_roc_auc_ovr": class_auc[0],
        "draw_roc_auc_ovr": class_auc[1],
        "away_roc_auc_ovr": class_auc[2],
        "macro_roc_auc_ovr": float(np.mean(class_auc)),
        "weighted_roc_auc_ovr": float(
            roc_auc_score(outcomes, probabilities, multi_class="ovr", average="weighted")
        ),
        "micro_roc_auc_ovr": float(roc_auc_score(one_hot.ravel(), probabilities.ravel())),
        "macro_roc_auc_ovo": float(
            roc_auc_score(outcomes, probabilities, multi_class="ovo", average="macro")
        ),
    }


def _paired_cluster_uncertainty(
    data: pd.DataFrame,
    *,
    candidate_model: str,
    baselines: Sequence[str],
    bootstrap_samples: int,
    random_seed: int,
) -> pd.DataFrame:
    aligned = {
        model: _model_frame(data, model)
        for model in (candidate_model, *baselines)
    }
    reference = aligned[candidate_model]
    cluster_keys = (
        reference["season"].astype(str)
        + "|"
        + reference["kickoff_utc"].dt.strftime("%Y-%m")
    ).to_numpy(str)
    clusters = np.array(sorted(set(cluster_keys)), dtype=object)
    cluster_indices = {
        cluster: np.flatnonzero(cluster_keys == cluster) for cluster in clusters
    }
    rng = np.random.default_rng(random_seed)
    metric_names = (
        "home_roc_auc_ovr",
        "draw_roc_auc_ovr",
        "away_roc_auc_ovr",
        "macro_roc_auc_ovr",
        "weighted_roc_auc_ovr",
        "micro_roc_auc_ovr",
    )
    rows = []
    candidate_full = _auc_metrics(reference)
    for baseline in baselines:
        baseline_full = _auc_metrics(aligned[baseline])
        distributions = {metric: [] for metric in metric_names}
        valid_samples = 0
        for _ in range(bootstrap_samples):
            sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
            indices = np.concatenate([cluster_indices[cluster] for cluster in sampled_clusters])
            outcomes = reference.iloc[indices]["actual_class"]
            if set(outcomes.astype(int)) != set(CLASS_LABELS):
                continue
            candidate_metrics = _auc_metrics(reference.iloc[indices])
            baseline_metrics = _auc_metrics(aligned[baseline].iloc[indices])
            for metric in metric_names:
                distributions[metric].append(
                    candidate_metrics[metric] - baseline_metrics[metric]
                )
            valid_samples += 1
        if valid_samples < int(0.95 * bootstrap_samples):
            raise ValueError("Too many invalid ROC-AUC bootstrap samples")
        for metric in metric_names:
            values = np.asarray(distributions[metric], dtype=float)
            difference = candidate_full[metric] - baseline_full[metric]
            lower, upper = np.quantile(values, [0.025, 0.975])
            rows.append(
                {
                    "candidate_model": candidate_model,
                    "baseline_model": baseline,
                    "metric": metric,
                    "candidate_value": candidate_full[metric],
                    "baseline_value": baseline_full[metric],
                    "difference": difference,
                    "ci_95_lower": float(lower),
                    "ci_95_upper": float(upper),
                    "reliable_improvement": bool(lower > 0.0),
                    "reliable_harm": bool(upper < 0.0),
                    "valid_bootstrap_samples": valid_samples,
                    "cluster_count": len(clusters),
                    "cluster_unit": "SEASON_CALENDAR_MONTH",
                }
            )
    return pd.DataFrame(rows)


def _roc_curves(data: pd.DataFrame, models: Sequence[str]) -> pd.DataFrame:
    rows = []
    for model in models:
        frame = _model_frame(data, model)
        outcomes = frame["actual_class"].to_numpy(int)
        probabilities = frame[list(PROBABILITY_COLUMNS)].to_numpy(float)
        for index, label in CLASS_LABELS.items():
            fpr, tpr, thresholds = roc_curve(
                (outcomes == index).astype(int), probabilities[:, index]
            )
            for point, (false_positive, true_positive, threshold) in enumerate(
                zip(fpr, tpr, thresholds, strict=True)
            ):
                rows.append(
                    {
                        "model": model,
                        "class_label": label,
                        "point": point,
                        "false_positive_rate": float(false_positive),
                        "true_positive_rate": float(true_positive),
                        "threshold": float(threshold),
                    }
                )
    return pd.DataFrame(rows)


def _model_frame(data: pd.DataFrame, model: str) -> pd.DataFrame:
    return (
        data[data["model"].eq(model)]
        .sort_values("match_id", kind="stable")
        .reset_index(drop=True)
    )
