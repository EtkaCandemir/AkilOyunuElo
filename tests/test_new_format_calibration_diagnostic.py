from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_new_format_calibration_diagnostic import evaluate_format_parameters


def _matches() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": ["m1", "m2"],
            "season": ["2025/26", "2025/26"],
            "competition": ["UCL", "UCL"],
            "home_live_pre": [1600.0, 1400.0],
            "away_live_pre": [1400.0, 1600.0],
            "is_neutral": [False, True],
            "home_goals": [1, 0],
            "away_goals": [0, 1],
            "effective_draw_at_even": [0.24, 0.12],
        }
    )


def test_format_calibration_changes_predictions_not_rating_state() -> None:
    source = _matches()
    baseline = evaluate_format_parameters(source)
    candidate = evaluate_format_parameters(source, 1.13, 1.14, 0.75)

    assert np.allclose(
        baseline[["home_live_pre", "away_live_pre"]],
        candidate[["home_live_pre", "away_live_pre"]],
    )
    assert candidate["rating_state_changed"].eq(False).all()
    assert np.allclose(
        candidate[["home_probability", "draw_probability", "away_probability"]].sum(axis=1),
        1.0,
    )
    assert candidate.loc[0, "forecast_elo_scale"] > baseline.loc[0, "forecast_elo_scale"]
    assert candidate.loc[1, "effective_home_advantage"] == 0.0
