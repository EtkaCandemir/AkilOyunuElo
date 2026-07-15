from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_ucl_probability_scale_calibration import (  # noqa: E402
    BASELINE_SCALE,
    calibration_decision,
    forecast_probability,
    score_scale,
)


def test_forecast_scale_does_not_change_rating_difference() -> None:
    data = pd.DataFrame(
        {
            "home_power": [900.0],
            "away_power": [800.0],
            "is_neutral": [False],
        }
    )

    baseline = forecast_probability(data, BASELINE_SCALE)[0]
    softer = forecast_probability(data, 400.0)[0]

    assert baseline > softer > 0.5


def test_score_scale_uses_fractional_draw_outcome() -> None:
    data = pd.DataFrame(
        {
            "home_power": [800.0],
            "away_power": [800.0],
            "is_neutral": [True],
            "actual_home_score": [0.5],
        }
    )

    brier, log_loss = score_scale(data, 300.0)

    assert brier == 0.0
    assert log_loss == pytest.approx(-0.5 * __import__("math").log(0.5) * 2)


def test_scale_acceptance_requires_reliable_stable_four_fold_gain() -> None:
    selections = pd.DataFrame({"selected_scale": [350.0, 350.0, 375.0, 350.0, 375.0]})
    uncertainty = pd.DataFrame(
        [
            {
                "comparison": "selected_vs_225",
                "reliable_improvement": True,
                "folds_won": 4,
            }
        ]
    )

    assert calibration_decision(selections, uncertainty) == (
        "PROVISIONAL_ACCEPT_UCL_FORECAST_SCALE_LAYER"
    )


def test_scale_is_kept_when_confidence_interval_is_inconclusive() -> None:
    selections = pd.DataFrame({"selected_scale": [400.0, 375.0, 400.0, 325.0, 300.0]})
    uncertainty = pd.DataFrame(
        [
            {
                "comparison": "selected_vs_225",
                "reliable_improvement": False,
                "folds_won": 2,
            }
        ]
    )

    assert calibration_decision(selections, uncertainty) == (
        "KEEP_FORECAST_SCALE_225_NO_RELIABLE_UCL_GAIN"
    )
