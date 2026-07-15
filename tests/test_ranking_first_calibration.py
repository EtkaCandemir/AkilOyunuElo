from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_ranking_first_calibration import (  # noqa: E402
    pairwise_ranking_accuracy,
    ranking_metrics,
)


def test_pairwise_accuracy_rewards_correct_order() -> None:
    actual = np.array([0.8, 0.5, 0.2])

    assert pairwise_ranking_accuracy(np.array([900, 700, 500]), actual) == 1.0
    assert pairwise_ranking_accuracy(np.array([500, 700, 900]), actual) == 0.0


def test_ranking_metrics_flag_zero_exposure_overranking() -> None:
    table = pd.DataFrame(
        {
            "team_id": [1, 2, 3, 4],
            "ao_first_elo": [1000, 900, 800, 700],
            "actual_score_rate": [0.1, 0.9, 0.6, 0.3],
            "european_exposure": [0.0, 1.0, 1.0, 1.0],
            "predicted_percentile": [1.0, 0.75, 0.5, 0.25],
            "actual_percentile": [0.25, 1.0, 0.75, 0.5],
            "percentile_error": [0.75, -0.25, -0.25, -0.25],
        }
    )

    metrics = ranking_metrics(table)

    assert metrics["zero_exposure_extreme_overrank_count"] == 1
    assert metrics["zero_exposure_mean_overrank"] == pytest.approx(0.75)
