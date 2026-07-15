from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_dynamic_core_calibration import DynamicCoreConfig, SeasonData  # noqa: E402
from scripts.run_goal_margin_calibration import (  # noqa: E402
    GoalMarginConfig,
    MarginSeasonData,
    candidate_grid,
    goal_margin_multiplier,
    run_margin_season,
)


@pytest.mark.parametrize("goal_difference", [0, 1])
def test_draws_and_one_goal_results_keep_unit_multiplier(goal_difference: int) -> None:
    config = GoalMarginConfig(goal_weight=1.0, goal_cap=2.0)

    assert goal_margin_multiplier(goal_difference, config) == 1.0


def test_goal_multiplier_is_capped() -> None:
    config = GoalMarginConfig(goal_weight=1.0, goal_cap=1.5)

    assert goal_margin_multiplier(2, config) == pytest.approx(1.5)
    assert goal_margin_multiplier(8, config) == pytest.approx(1.5)


def test_candidate_grid_contains_one_unmodified_core_baseline() -> None:
    candidates = candidate_grid()

    assert candidates.count(GoalMarginConfig(0.0, 1.0)) == 1
    assert len(candidates) == 17


def test_goal_margin_updates_remain_zero_sum() -> None:
    core = SeasonData(
        season="2024/25",
        initial_ratings=np.array([np.nan, 800.0, 790.0, 780.0]),
        active_team_ids=np.array([1, 2, 3]),
        home_team_ids=np.array([1, 2]),
        away_team_ids=np.array([2, 3]),
        actual_home_scores=np.array([1.0, 0.0]),
        neutral_flags=np.array([False, False]),
        competitions=np.array(["UCL", "UCL"]),
        match_ids=np.array(["m1", "m2"]),
    )
    data = MarginSeasonData(core=core, goal_differences=np.array([3, 2]))
    metrics, predictions = run_margin_season(
        data,
        DynamicCoreConfig(225, 40, 28),
        GoalMarginConfig(0.5, 1.5),
        return_predictions=True,
    )

    assert metrics["mean_rating_change"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["max_abs_match_delta"] <= 28 * 1.5
    assert predictions is not None
    assert predictions["goal_multiplier"].between(1.0, 1.5).all()
