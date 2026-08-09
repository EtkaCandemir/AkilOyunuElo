from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


EVIDENCE_CLASSES = {
    "OOS_FOLD6",
    "RETROSPECTIVE_COUNTERFACTUAL",
    "MATCHED_XG_APPENDIX",
}
SHADOW_CLASSIFICATIONS = {
    "CONSISTENT_SHADOW_SIGNAL",
    "MIXED_OR_INCONCLUSIVE",
    "HARM_SIGNAL",
    "REFERENCE",
}


@dataclass(frozen=True, order=True)
class ReplayArmSpec:
    model_arm: str
    evidence_class: str
    alpha: float
    tau: float
    tournament_bonus_base: float = 0.0
    domestic_persistence: float = 0.0
    reference_arm: str = ""

    def validate(self) -> None:
        if not self.model_arm.strip():
            raise ValueError("model_arm must be non-empty")
        if self.evidence_class not in EVIDENCE_CLASSES:
            raise ValueError(f"Unknown evidence_class: {self.evidence_class}")
        for name in ("alpha", "tau", "tournament_bonus_base", "domestic_persistence"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.alpha < 0.0 or self.tau <= 0.0 or self.tournament_bonus_base < 0.0:
            raise ValueError("Replay arm parameters are outside their valid range")
        if not 0.0 <= self.domestic_persistence <= 1.0:
            raise ValueError("domestic_persistence must be in [0,1]")

    @property
    def config_fingerprint(self) -> str:
        self.validate()
        return stable_config_fingerprint(asdict(self))


def stable_config_fingerprint(payload: object) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def prediction_metrics(predictions: pd.DataFrame) -> dict[str, float | int]:
    required = {
        "home_probability",
        "draw_probability",
        "away_probability",
        "actual_class",
    }
    missing = sorted(required - set(predictions.columns))
    if missing or predictions.empty:
        raise ValueError(f"Prediction frame is invalid; missing={missing}")
    probabilities = predictions[
        ["home_probability", "draw_probability", "away_probability"]
    ].to_numpy(float)
    outcomes = predictions["actual_class"].to_numpy(int)
    if not np.isfinite(probabilities).all() or (probabilities < 0.0).any():
        raise ValueError("Probabilities must be finite and non-negative")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("1X2 probabilities must sum to one")
    if ((outcomes < 0) | (outcomes > 2)).any():
        raise ValueError("actual_class must be 0, 1, or 2")
    targets = np.eye(3)[outcomes]
    observed = probabilities[np.arange(len(probabilities)), outcomes]
    brier = np.square(probabilities - targets).sum(axis=1)
    log_loss = -np.log(np.maximum(observed, 1e-15))
    return {
        "matches": int(len(predictions)),
        "brier_1x2": float(brier.mean()),
        "log_loss_1x2": float(log_loss.mean()),
        "accuracy_1x2": float((probabilities.argmax(axis=1) == outcomes).mean()),
        "mean_max_probability": float(probabilities.max(axis=1).mean()),
    }


def same_season_ranking(
    final_ratings: pd.DataFrame,
    target: pd.DataFrame,
    *,
    season: str,
) -> pd.DataFrame:
    rows = []
    actual_season = target.loc[target["season"].astype(str).eq(season)]
    for competition, actual in actual_season.groupby("competition", sort=True):
        table = actual[["team_id", "schedule_adjusted_score"]].merge(
            final_ratings[["team_id", "end_live_rating"]],
            on="team_id",
            validate="one_to_one",
        )
        if len(table) < 3:
            continue
        rows.append(
            {
                "competition": competition,
                "teams": len(table),
                "team_weight": len(table),
                "pair_weight": len(table) * (len(table) - 1) / 2,
                "ranking_score": float(
                    table["end_live_rating"].corr(
                        table["schedule_adjusted_score"], method="spearman"
                    )
                ),
                "pairwise_accuracy": pairwise_accuracy(
                    table["end_live_rating"].to_numpy(float),
                    table["schedule_adjusted_score"].to_numpy(float),
                ),
            }
        )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail
    all_row = {
        "competition": "ALL",
        "teams": int(actual_season["team_id"].nunique()),
        "team_weight": int(detail["team_weight"].sum()),
        "pair_weight": float(detail["pair_weight"].sum()),
        "ranking_score": float(
            np.average(detail["ranking_score"], weights=detail["team_weight"])
        ),
        "pairwise_accuracy": float(
            np.average(detail["pairwise_accuracy"], weights=detail["pair_weight"])
        ),
    }
    return pd.concat([pd.DataFrame([all_row]), detail], ignore_index=True)


def pairwise_accuracy(predicted: np.ndarray, actual: np.ndarray) -> float:
    if len(predicted) != len(actual) or len(predicted) < 2:
        raise ValueError("Pairwise accuracy requires aligned arrays of length >=2")
    correct = 0.0
    comparisons = 0
    for left in range(len(predicted)):
        for right in range(left + 1, len(predicted)):
            actual_direction = np.sign(actual[left] - actual[right])
            predicted_direction = np.sign(predicted[left] - predicted[right])
            if actual_direction == 0:
                continue
            comparisons += 1
            correct += 1.0 if predicted_direction == actual_direction else 0.5 if predicted_direction == 0 else 0.0
    return float(correct / comparisons) if comparisons else float("nan")


def classify_shadow_signal(
    brier_delta: float,
    log_loss_delta: float,
    competition_no_harm: bool,
    reliable_harm: bool,
) -> str:
    values = (brier_delta, log_loss_delta)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Loss deltas must be finite")
    if reliable_harm or (brier_delta > 0.0 and log_loss_delta > 0.0):
        return "HARM_SIGNAL"
    if brier_delta < 0.0 and log_loss_delta < 0.0 and competition_no_harm:
        return "CONSISTENT_SHADOW_SIGNAL"
    return "MIXED_OR_INCONCLUSIVE"
