from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_forecast_scale_calibration import evaluate_multiplier, metrics


def _matches() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": ["m1", "m2"],
            "season": ["2025/26", "2025/26"],
            "competition": ["UCL", "UCL"],
            "home_live_pre": [1700.0, 1300.0],
            "away_live_pre": [1300.0, 1700.0],
            "is_neutral": [False, True],
            "home_goals": [1, 0],
            "away_goals": [0, 1],
            "effective_draw_at_even": [0.24, 0.12],
        }
    )


def test_forecast_scale_changes_probabilities_not_pre_match_rating_state() -> None:
    source = _matches()
    baseline = evaluate_multiplier(source, 1.0)
    softened = evaluate_multiplier(source, 1.10)

    assert np.allclose(
        baseline[["home_live_pre", "away_live_pre"]],
        softened[["home_live_pre", "away_live_pre"]],
    )
    assert baseline["rating_state_changed"].eq(False).all()
    assert softened["rating_state_changed"].eq(False).all()
    assert softened.loc[0, "home_probability"] < baseline.loc[0, "home_probability"]
    assert np.allclose(
        softened[["home_probability", "draw_probability", "away_probability"]].sum(axis=1),
        1.0,
    )
    assert metrics(softened)["matches"] == 2
