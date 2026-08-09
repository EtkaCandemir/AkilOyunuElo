from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_fotmob_xg_backtest_2025_26 import (
    BASELINE_ARM,
    ConfirmatoryXGArm,
    confirmatory_arms,
    read_fotmob_xg_dataset,
    replay_arm,
)


DATA_PATH = ROOT / "data" / "xg_2025_26" / "uefa_2025_26_matches_with_xg.csv"


PRODUCTION = {
    "dynamic_core": {
        "elo_scale": 835.5614973262034,
        "home_advantage": 148.54426619132505,
        "k_factor": 103.98098633392752,
    },
    "one_x_two_probability": {
        "draw_at_even": 0.24,
        "draw_shape": 1.0,
    },
    "goal_margin": {
        "alpha": 0.10,
        "tau": 300.0,
        "goal_difference_cap": 4,
    },
}


def event_frame(*, xg_eligible: bool = True, penalties: bool = False) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "synthetic-1",
                "season": "2025/26",
                "competition": "UCL",
                "round": "League Stage",
                "kickoff_utc": pd.Timestamp("2025-09-01T19:00:00Z"),
                "event_order": 1,
                "tie_id": None,
                "home_team_id": 1,
                "away_team_id": 2,
                "home_team_name": "AO Home",
                "away_team_name": "AO Away",
                "home_goals": 2,
                "away_goals": 0,
                "actual_home_score": 1.0,
                "is_neutral": False,
                "decided_on_penalties": penalties,
                "xg_analysis_eligible": xg_eligible,
                "xg_home": 0.2 if xg_eligible else None,
                "xg_away": 2.4 if xg_eligible else None,
            }
        ]
    )


def test_first_match_prediction_is_recorded_before_current_xg_update() -> None:
    events = event_frame()
    baseline = replay_arm(
        events,
        {1: 1500.0, 2: 1500.0},
        PRODUCTION,
        confirmatory_arms()[0],
    )
    candidate = replay_arm(
        events,
        {1: 1500.0, 2: 1500.0},
        PRODUCTION,
        confirmatory_arms()[2],
    )
    probability_columns = [
        "home_probability",
        "draw_probability",
        "away_probability",
    ]
    assert candidate.predictions.loc[0, probability_columns].to_list() == pytest.approx(
        baseline.predictions.loc[0, probability_columns].to_list()
    )
    assert candidate.predictions.loc[0, "power_delta"] != pytest.approx(
        baseline.predictions.loc[0, "power_delta"]
    )


def test_missing_xg_falls_back_exactly_to_production_update() -> None:
    events = event_frame(xg_eligible=False)
    baseline = replay_arm(
        events,
        {1: 1500.0, 2: 1500.0},
        PRODUCTION,
        confirmatory_arms()[0],
    )
    candidate = replay_arm(
        events,
        {1: 1500.0, 2: 1500.0},
        PRODUCTION,
        confirmatory_arms()[1],
    )
    assert candidate.predictions.loc[0, "power_delta"] == pytest.approx(
        baseline.predictions.loc[0, "power_delta"]
    )
    assert candidate.end_ratings["end_live_rating"].to_list() == pytest.approx(
        baseline.end_ratings["end_live_rating"].to_list()
    )


def test_penalty_decider_keeps_field_result_without_extra_multipliers() -> None:
    result = replay_arm(
        event_frame(penalties=True),
        {1: 1500.0, 2: 1500.0},
        PRODUCTION,
        confirmatory_arms()[1],
    )
    row = result.predictions.iloc[0]
    assert row["actual_class"] == 0
    assert row["goal_difference"] == 2
    assert row["goal_multiplier"] == pytest.approx(1.0)


def test_confirmatory_arm_contract_is_fixed() -> None:
    arms = confirmatory_arms()
    assert [arm.key for arm in arms] == [
        BASELINE_ARM,
        "GD_XG_CONVEX_PREREG",
        "GD_XG_ADDITIVE_PREREG",
        "GD_XG_BOUNDED_PREREG",
    ]
    assert [(arm.rho, arm.xg_scale) for arm in arms] == [
        (0.0, 1.0),
        (0.05, 1.0),
        (0.50, 0.75),
        (0.05, 1.0),
    ]
    with pytest.raises(ValueError, match="rho=0"):
        ConfirmatoryXGArm(
            BASELINE_ARM,
            0.1,
            1.0,
            "CONVEX_BLEND",
            "INVALID",
        ).validate()


def test_checked_in_dataset_matches_confirmatory_contract() -> None:
    events = read_fotmob_xg_dataset(DATA_PATH, strict_contract=True)
    assert len(events) == 961
    assert int(events["xg_analysis_eligible"].sum()) == 606
    assert events["competition"].value_counts().to_dict() == {
        "UECL": 409,
        "UCL": 281,
        "UEL": 271,
    }
    assert events["kickoff_utc"].is_monotonic_increasing
