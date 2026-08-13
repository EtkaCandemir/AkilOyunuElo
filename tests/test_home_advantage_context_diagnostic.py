from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_home_advantage_context_diagnostic import (
    evaluate_home_advantage_multiplier,
    metrics,
)


def _matches() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": ["m1", "m2"],
            "season": ["2025/26", "2025/26"],
            "competition": ["UCL", "UCL"],
            "home_live_pre": [1500.0, 1500.0],
            "away_live_pre": [1500.0, 1500.0],
            "is_neutral": [False, True],
            "home_goals": [1, 0],
            "away_goals": [0, 1],
            "effective_draw_at_even": [0.24, 0.12],
        }
    )


def test_home_advantage_context_is_prediction_only_and_respects_neutral_venues() -> None:
    source = _matches()
    baseline = evaluate_home_advantage_multiplier(source, 1.0)
    closed_doors_proxy = evaluate_home_advantage_multiplier(source, 0.0)

    assert np.allclose(
        baseline[["home_live_pre", "away_live_pre"]],
        closed_doors_proxy[["home_live_pre", "away_live_pre"]],
    )
    assert baseline["rating_state_changed"].eq(False).all()
    assert closed_doors_proxy["rating_state_changed"].eq(False).all()
    assert closed_doors_proxy.loc[0, "home_probability"] < baseline.loc[0, "home_probability"]
    assert closed_doors_proxy.loc[1, "home_probability"] == baseline.loc[1, "home_probability"]
    assert np.allclose(
        closed_doors_proxy[["home_probability", "draw_probability", "away_probability"]].sum(axis=1),
        1.0,
    )
    assert metrics(closed_doors_proxy)["matches"] == 2
