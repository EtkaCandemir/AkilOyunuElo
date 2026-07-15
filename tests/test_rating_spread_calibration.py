from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_rating_spread_calibration import (
    ProbabilityConfig,
    expanding_folds,
    expected_home_score,
    observed_difference_bands,
    select_candidate,
    select_spread_only_candidate,
    summarize_paired_uncertainty,
)


def test_expected_score_is_symmetric_without_home_advantage() -> None:
    config = ProbabilityConfig(elo_scale=400, home_advantage=0)
    probabilities = expected_home_score(np.array([100.0, -100.0]), config)

    assert probabilities[0] == pytest.approx(1.0 - probabilities[1])
    assert probabilities[0] > 0.5


def test_lower_scale_is_selected_for_decisive_rating_signal() -> None:
    data = pd.DataFrame(
        {
            "rating_difference": [100.0, -100.0] * 100,
            "actual_home_score": [1.0, 0.0] * 100,
        }
    )

    selected, _ = select_candidate(data)

    assert selected.elo_scale == 100
    assert selected.home_advantage == 0
    assert selected.spread_multiplier == 4.0

    spread_only, _ = select_spread_only_candidate(data)
    assert spread_only.elo_scale == 100
    assert spread_only.home_advantage == 70


def test_expanding_folds_never_train_on_test_or_future_seasons() -> None:
    seasons = ("2018/19", "2019/20", "2020/21", "2021/22")

    folds = expanding_folds(seasons)

    assert folds == (
        (("2018/19", "2019/20"), "2020/21"),
        (("2018/19", "2019/20", "2020/21"), "2021/22"),
    )


def test_observed_difference_bands_orient_score_to_higher_rated_team() -> None:
    matches = pd.DataFrame(
        {
            "rating_difference": [40.0, -40.0],
            "actual_home_score": [1.0, 0.0],
        }
    )

    bands = observed_difference_bands(matches)

    assert bands.iloc[0]["difference_band"] == "25-50"
    assert bands.iloc[0]["higher_rated_team_score"] == 1.0


def test_paired_uncertainty_detects_consistent_improvement() -> None:
    paired = pd.DataFrame(
        {
            "competition": ["UCL"] * 20,
            "baseline_loss": [0.7] * 20,
            "joint_loss": [0.6] * 20,
            "spread_only_loss": [0.6] * 20,
        }
    )

    uncertainty = summarize_paired_uncertainty(paired, bootstrap_samples=200)
    overall = uncertainty.loc[
        uncertainty["competition"].eq("ALL")
        & uncertainty["comparison"].eq("spread_only_vs_baseline")
    ].iloc[0]

    assert bool(overall["directionally_reliable_improvement"])
    assert not bool(overall["directionally_reliable_harm"])
