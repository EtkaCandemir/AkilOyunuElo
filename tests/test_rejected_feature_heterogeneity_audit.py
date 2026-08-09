from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_rejected_feature_heterogeneity_audit import (  # noqa: E402
    build_leave_one_out,
    classify_feature,
    grouped_summary,
    metric_summary,
)


def prediction_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fold": 1,
                "match_id": "m1",
                "season": "2024/25",
                "competition": "UCL",
                "home_team_id": 1,
                "away_team_id": 2,
                "home_team_name": "A",
                "away_team_name": "B",
                "brier_difference": 0.04,
                "log_loss_difference": 0.04,
            },
            {
                "fold": 2,
                "match_id": "m2",
                "season": "2025/26",
                "competition": "UEL",
                "home_team_id": 3,
                "away_team_id": 4,
                "home_team_name": "C",
                "away_team_name": "D",
                "brier_difference": -0.01,
                "log_loss_difference": -0.01,
            },
        ]
    )


def test_leave_one_team_out_detects_sign_flipping_outlier():
    data = prediction_rows()
    audit = build_leave_one_out(data)
    team = audit.loc[
        audit["cluster_type"].eq("team") & audit["cluster_id"].eq("1")
    ].iloc[0]
    assert bool(team["both_metrics_sign_flip"])


def test_classifier_distinguishes_outlier_sensitive_harm():
    data = prediction_rows()
    classification = classify_feature(
        metric_summary(data),
        grouped_summary(data, "fold"),
        grouped_summary(data, "competition"),
        build_leave_one_out(data),
    )
    assert classification["classification"] == "OUTLIER_SENSITIVE_GLOBAL_HARM"
    assert classification["outlier_driven"] is True
