from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_external_elo_benchmark import (  # noqa: E402
    add_clubelo_predictions,
    clubelo_expected_home_score,
    select_clubelo_home_advantage,
)


def test_clubelo_probability_uses_published_400_scale() -> None:
    assert clubelo_expected_home_score(1800, 1800, 0, neutral=False) == 0.5
    assert clubelo_expected_home_score(1800, 1600, 0, neutral=True) == pytest.approx(
        1.0 / (1.0 + 10 ** -0.5)
    )


def test_neutral_match_removes_home_advantage() -> None:
    neutral = clubelo_expected_home_score(1700, 1700, 80, neutral=True)
    home = clubelo_expected_home_score(1700, 1700, 80, neutral=False)

    assert neutral == 0.5
    assert home > neutral


def test_home_advantage_selection_uses_training_losses() -> None:
    train = pd.DataFrame(
        {
            "clubelo_home_elo": [1700.0] * 12,
            "clubelo_away_elo": [1700.0] * 12,
            "is_neutral": [False] * 12,
            "actual_home_score": [1.0] * 9 + [0.0] * 3,
        }
    )

    selected, metrics = select_clubelo_home_advantage(train)

    assert selected > 0
    assert metrics.matches == 12


def test_prediction_losses_are_row_aligned() -> None:
    data = pd.DataFrame(
        {
            "clubelo_home_elo": [1700.0, 1700.0],
            "clubelo_away_elo": [1700.0, 1700.0],
            "is_neutral": [True, True],
            "actual_home_score": [1.0, 0.0],
        }
    )

    result = add_clubelo_predictions(data, 100)

    assert result["clubelo_expected_home_score"].tolist() == [0.5, 0.5]
    assert result["clubelo_brier_loss"].tolist() == [0.25, 0.25]
