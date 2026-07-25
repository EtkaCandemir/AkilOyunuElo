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
from scripts.run_v2_achievement_reserve_calibration import (  # noqa: E402
    BASELINE_NAME,
    MODEL_NAME,
    AchievementReserveConfig,
    ReserveSeasonData,
    candidate_grid,
    evaluate_sequence,
    normalize_exact_tie_deciders,
    promotion_decision,
)
from scripts.run_v2_goal_margin_calibration import GoalCarrySeasonData  # noqa: E402


def synthetic_reserve_season() -> ReserveSeasonData:
    initial = np.array([np.nan, 1200.0, 1200.0])
    core = SeasonData(
        season="2024/25",
        initial_ratings=initial,
        active_team_ids=np.array([1, 2]),
        home_team_ids=np.array([1, 1]),
        away_team_ids=np.array([2, 2]),
        actual_home_scores=np.array([0.5, 0.5]),
        neutral_flags=np.array([True, True]),
        competitions=np.array(["UCL", "UCL"]),
        match_ids=np.array(["semi", "next"]),
    )
    keys = np.array([None, "ENG::winner", "ESP::opponent"], dtype=object)
    carry = CarrySeasonData(
        core=core,
        club_keys=keys,
        rounds=np.array(["Semi Finals", "League Stage"]),
        tie_decider_flags=np.array([True, False]),
        advanced_team_ids=np.array([1, -1]),
    )
    goal = GoalCarrySeasonData(
        carry=carry,
        goal_differences=np.array([0, 0]),
        penalty_flags=np.array([False, False]),
    )
    return ReserveSeasonData(
        goal=goal,
        tie_ids=np.array(["tie-1", None], dtype=object),
        knockout_flags=np.array([True, False]),
        tie_decider_flags=np.array([True, False]),
        advanced_team_ids=np.array([1, -1]),
        stages=np.array(["SEMIFINAL", "LEAGUE"]),
    )


def test_reserve_grid_matches_full_contract() -> None:
    candidates = candidate_grid()

    assert len(candidates) == 961
    assert sum(candidate.reserve_base == 0 for candidate in candidates) == 1
    assert all(candidate.uel_multiplier > candidate.uecl_multiplier for candidate in candidates)


def test_invalid_competition_hierarchy_is_rejected() -> None:
    config = AchievementReserveConfig(100.0, 0.50, 0.60, "FLAT", 0.5)

    with pytest.raises(ValueError, match="UCL > UEL > UECL"):
        config.validate()


def test_postponed_first_leg_moves_decider_to_chronological_last_match() -> None:
    events = pd.DataFrame(
        [
            {
                "season": "2023/24",
                "event_order": 1,
                "match_id": "source-leg-2",
                "tie_id": "tie",
                "home_team_id": 1,
                "away_team_id": 2,
                "is_knockout": True,
                "is_tie_decider": True,
                "advanced_team_id": 2,
            },
            {
                "season": "2023/24",
                "event_order": 2,
                "match_id": "postponed-source-leg-1",
                "tie_id": "tie",
                "home_team_id": 2,
                "away_team_id": 1,
                "is_knockout": True,
                "is_tie_decider": False,
                "advanced_team_id": np.nan,
            },
        ]
    )

    normalized, audit = normalize_exact_tie_deciders(events)

    assert normalized["is_tie_decider"].tolist() == [False, True]
    assert normalized["advanced_team_id"].tolist()[1] == 2
    assert audit.iloc[0]["new_decider_match_id"] == "postponed-source-leg-1"


def test_reserve_is_added_after_tie_and_visible_on_next_match() -> None:
    config = AchievementReserveConfig(
        reserve_base=100.0,
        uel_multiplier=0.65,
        uecl_multiplier=0.45,
        stage_profile="SEMIFINAL_HEAVY",
        reserve_decay=0.5,
    )
    metrics, predictions = evaluate_sequence(
        (synthetic_reserve_season(),),
        DynamicCoreConfig(835.0, 0.0, 100.0),
        0.85,
        GoalMarginConfig(0.0, 1.0),
        config,
        return_predictions=True,
    )

    assert predictions is not None
    assert predictions.iloc[0]["expected_home_score"] == pytest.approx(0.5)
    assert predictions.iloc[0]["advance_reserve_added_after_match"] == pytest.approx(75.0)
    assert predictions.iloc[1]["home_reserve"] == pytest.approx(75.0)
    assert predictions.iloc[1]["expected_home_score"] > 0.5
    assert metrics["max_reserve"] == pytest.approx(75.0)
    assert metrics["max_pair_sum_error"] == pytest.approx(0.0, abs=1e-12)


def test_non_conclusive_reserve_signal_stays_disabled() -> None:
    selections = pd.DataFrame(
        [{"selected_reserve_base": 100.0} for _ in range(6)]
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
                    "max_reserve": 0.0,
                    "max_pair_sum_error": 0.0,
                },
                {
                    "fold": fold,
                    "model": MODEL_NAME,
                    "brier": 0.16,
                    "start_end_rank_correlation": 0.90,
                    "max_abs_rating_change": 150.0,
                    "max_reserve": 100.0,
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
    candidate = AchievementReserveConfig(100.0, 0.65, 0.45, "FLAT", 0.5)

    decision, guardrails = promotion_decision(
        selections,
        pd.DataFrame(rows),
        uncertainty,
        candidate,
    )

    assert decision == "DISABLE_ACHIEVEMENT_RESERVE"
    assert not guardrails["overall_reliable_improvement"]
