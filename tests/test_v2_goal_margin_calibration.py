from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_achievement_carry_calibration import CarrySeasonData  # noqa: E402
from scripts.run_dynamic_core_calibration import DynamicCoreConfig, SeasonData  # noqa: E402
from scripts.run_goal_margin_calibration import GoalMarginConfig  # noqa: E402
from scripts.run_v2_goal_margin_calibration import (  # noqa: E402
    BASELINE_NAME,
    MODEL_NAME,
    GoalCarrySeasonData,
    evaluate_sequence,
    promotion_decision,
    validate_score_contract,
)


def synthetic_goal_season() -> GoalCarrySeasonData:
    initial = np.array([np.nan, 1200.0, 1100.0])
    core = SeasonData(
        season="2024/25",
        initial_ratings=initial,
        active_team_ids=np.array([1, 2]),
        home_team_ids=np.array([1]),
        away_team_ids=np.array([2]),
        actual_home_scores=np.array([1.0]),
        neutral_flags=np.array([False]),
        competitions=np.array(["UCL"]),
        match_ids=np.array(["m1"]),
    )
    keys = np.array([None, "ENG::home", "ESP::away"], dtype=object)
    carry = CarrySeasonData(
        core=core,
        club_keys=keys,
        rounds=np.array(["League Stage"]),
        tie_decider_flags=np.array([False]),
        advanced_team_ids=np.array([-1]),
    )
    return GoalCarrySeasonData(
        carry=carry,
        goal_differences=np.array([3]),
        penalty_flags=np.array([False]),
    )


def test_goal_update_remains_zero_sum_with_carry_engine() -> None:
    metrics, predictions = evaluate_sequence(
        (synthetic_goal_season(),),
        DynamicCoreConfig(835.0, 148.0, 104.0),
        0.85,
        GoalMarginConfig(0.5, 2.0),
        return_predictions=True,
    )

    assert metrics["max_pair_sum_error"] == pytest.approx(0.0, abs=1e-12)
    assert predictions is not None
    assert predictions.iloc[0]["goal_multiplier"] > 1.0


def test_penalty_shootout_must_remain_a_field_score_draw() -> None:
    valid = pd.DataFrame(
        [
            {
                "goal_difference": 0,
                "decided_on_penalties": True,
                "actual_home_score": 0.5,
                "home_goals": 1,
                "away_goals": 1,
                "result_basis": "displayed_score_excluding_shootout",
            }
        ]
    )
    validate_score_contract(valid)
    invalid = valid.copy()
    invalid.loc[0, "goal_difference"] = 1

    with pytest.raises(ValueError, match="must match the 90/120-minute field score"):
        validate_score_contract(invalid)


def test_non_conclusive_clustered_signal_disables_goal_margin() -> None:
    selections = pd.DataFrame(
        [{"selected_goal_weight": 0.5} for _ in range(6)]
    )
    rows = []
    for fold in range(1, 7):
        rows.extend(
            [
                {
                    "fold": fold,
                    "model": BASELINE_NAME,
                    "brier": 0.17,
                    "start_end_rank_correlation": 0.95,
                    "max_abs_rating_change": 100.0,
                    "max_pair_sum_error": 0.0,
                },
                {
                    "fold": fold,
                    "model": MODEL_NAME,
                    "brier": 0.16,
                    "start_end_rank_correlation": 0.90,
                    "max_abs_rating_change": 150.0,
                    "max_pair_sum_error": 0.0,
                },
            ]
        )
    uncertainty = pd.DataFrame(
        [
            {
                "competition": "ALL",
                "reliable_improvement": False,
                "reliable_harm": False,
            },
            {
                "competition": "UCL",
                "reliable_improvement": False,
                "reliable_harm": False,
            },
        ]
    )

    decision, guardrails = promotion_decision(
        selections,
        pd.DataFrame(rows),
        uncertainty,
        GoalMarginConfig(0.5, 2.0),
    )

    assert decision == "DISABLE_GOAL_MARGIN"
    assert not guardrails["overall_reliable_improvement"]
