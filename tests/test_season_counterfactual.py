from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ao_elo.season_counterfactual import (
    apply_score_overrides,
    build_all_wins_one_goal_overrides,
)


def _events() -> pd.DataFrame:
    rows = []
    for index in range(8):
        team_home = index % 2 == 0
        rows.append(
            {
                "match_id": f"m{index}",
                "season": "2025/26",
                "competition": "UCL",
                "round": "League Stage",
                "kickoff_utc": f"2025-09-{index + 1:02d}T20:00:00Z",
                "home_team_id": 170 if team_home else index + 1,
                "away_team_id": index + 1 if team_home else 170,
                "home_team_name": "Bodø/Glimt" if team_home else f"Team {index}",
                "away_team_name": f"Team {index}" if team_home else "Bodø/Glimt",
                "home_goals": index % 3,
                "away_goals": (index + 1) % 3,
            }
        )
    return pd.DataFrame(rows)


def test_all_wins_overrides_are_eight_one_goal_wins() -> None:
    overrides = build_all_wins_one_goal_overrides(
        _events(),
        season="2025/26",
        competition="UCL",
        round_name="League Stage",
        team_id=170,
        expected_matches=8,
    )
    assert len(overrides) == 8
    assert (
        overrides["counterfactual_home_goals"]
        - overrides["counterfactual_away_goals"]
    ).abs().eq(1).all()
    for row in overrides.itertuples(index=False):
        if row.home_team_id == 170:
            assert (row.counterfactual_home_goals, row.counterfactual_away_goals) == (1, 0)
        else:
            assert (row.counterfactual_home_goals, row.counterfactual_away_goals) == (0, 1)
    assert overrides["counterfactual_xg_rule"].eq(
        "DISABLED_UNKNOWN_COUNTERFACTUAL_XG"
    ).all()


def test_apply_score_overrides_does_not_mutate_inputs() -> None:
    events = _events()
    overrides = build_all_wins_one_goal_overrides(
        events,
        season="2025/26",
        competition="UCL",
        round_name="League Stage",
        team_id=170,
        expected_matches=8,
    )
    home = events["home_goals"].to_numpy()
    away = events["away_goals"].to_numpy()
    original_home = home.copy()
    original_away = away.copy()
    changed_home, changed_away = apply_score_overrides(
        events["match_id"], home, away, overrides
    )
    assert np.array_equal(home, original_home)
    assert np.array_equal(away, original_away)
    assert np.array_equal(changed_home, np.array([1, 0, 1, 0, 1, 0, 1, 0]))
    assert np.array_equal(changed_away, np.array([0, 1, 0, 1, 0, 1, 0, 1]))


def test_counterfactual_rejects_wrong_match_count() -> None:
    with pytest.raises(ValueError, match="Expected 8 counterfactual matches"):
        build_all_wins_one_goal_overrides(
            _events().iloc[:-1],
            season="2025/26",
            competition="UCL",
            round_name="League Stage",
            team_id=170,
            expected_matches=8,
        )
