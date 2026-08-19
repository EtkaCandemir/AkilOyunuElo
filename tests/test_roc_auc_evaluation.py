from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ao_elo.roc_auc_evaluation import evaluate_multiclass_roc_auc


def _predictions() -> pd.DataFrame:
    outcomes = [0, 1, 2, 0, 1, 2, 0, 1, 2] * 4
    rows = []
    for model, strength in (("BASE", 0.55), ("CANDIDATE", 0.75)):
        for index, outcome in enumerate(outcomes):
            if model == "BASE":
                probability = [0.45, 0.30, 0.25]
            else:
                remainder = (1.0 - strength) / 2.0
                probability = [remainder, remainder, remainder]
                probability[outcome] = strength
            rows.append(
                {
                    "fold": 1 + index // 18,
                    "model": model,
                    "match_id": f"m{index:03d}",
                    "season": "2020/21" if index < 18 else "2021/22",
                    "kickoff_utc": pd.Timestamp("2020-08-01", tz="UTC")
                    + pd.Timedelta(days=index),
                    "competition": "UCL",
                    "actual_class": outcome,
                    "home_probability": probability[0],
                    "draw_probability": probability[1],
                    "away_probability": probability[2],
                }
            )
    return pd.DataFrame(rows)


def test_multiclass_roc_auc_detects_better_candidate() -> None:
    result = evaluate_multiclass_roc_auc(
        _predictions(),
        models=("BASE", "CANDIDATE"),
        candidate_model="CANDIDATE",
        baselines=("BASE",),
        bootstrap_samples=100,
        random_seed=7,
    )

    summary = result.model_summary.set_index("model")
    assert summary.loc["CANDIDATE", "macro_roc_auc_ovr"] == pytest.approx(1.0)
    assert summary.loc["BASE", "macro_roc_auc_ovr"] == pytest.approx(0.5)
    assert result.data_audit["matches"] == 36
    assert result.data_audit["contains_2026_27"] is False


def test_multiclass_roc_auc_rejects_misaligned_probabilities() -> None:
    frame = _predictions()
    frame.loc[0, "home_probability"] = 0.9

    with pytest.raises(ValueError, match="sum to one"):
        evaluate_multiclass_roc_auc(
            frame,
            models=("BASE", "CANDIDATE"),
            candidate_model="CANDIDATE",
            baselines=("BASE",),
            bootstrap_samples=100,
        )


def test_multiclass_roc_auc_is_deterministic() -> None:
    kwargs = {
        "models": ("BASE", "CANDIDATE"),
        "candidate_model": "CANDIDATE",
        "baselines": ("BASE",),
        "bootstrap_samples": 100,
        "random_seed": 11,
    }
    first = evaluate_multiclass_roc_auc(_predictions(), **kwargs)
    second = evaluate_multiclass_roc_auc(_predictions(), **kwargs)
    pd.testing.assert_frame_equal(first.paired_uncertainty, second.paired_uncertainty)
