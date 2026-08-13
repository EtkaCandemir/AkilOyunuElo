from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_domestic_regression_diagnostics import (  # noqa: E402
    diagnostic_decision,
    markdown_table,
)


def test_diagnostic_identifies_inaugural_uecl_outlier_sensitivity():
    competition = pd.DataFrame(
        [
            {
                "competition": "ALL",
                "spearman_difference": -0.001,
                "pairwise_difference": -0.001,
            }
        ]
    )
    leave_one_out = pd.DataFrame(
        [
            {
                "removed_competition": "UECL",
                "pooled_delta_nonnegative": True,
            }
        ]
    )
    sensitivity = pd.DataFrame(
        [
            {
                "persistence": 0.75,
                "ranking_difference": -0.001,
                "pairwise_difference": -0.001,
                "brier_difference": -0.002,
                "log_loss_difference": -0.003,
            }
        ]
    )
    match_metrics = pd.DataFrame(
        [
            {
                "competition": "ALL",
                "brier_difference": -0.002,
                "log_loss_difference": -0.003,
            }
        ]
    )
    decision = diagnostic_decision(
        competition,
        leave_one_out,
        sensitivity,
        match_metrics,
    )
    assert decision["diagnosis"] == "INAUGURAL_UECL_OUTLIER_SENSITIVE_GAP"
    assert decision["status"] == "NO_AUTOMATIC_RANKING_VETO"
    assert "no longer vetoes" in decision["reason"]


def test_markdown_table_does_not_require_optional_tabulate_dependency():
    rendered = markdown_table(
        pd.DataFrame([{"team": "A", "delta": -0.001234}]),
        float_digits=3,
    )
    assert "| team | delta |" in rendered
    assert "-0.001" in rendered
