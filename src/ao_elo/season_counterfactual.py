from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def build_all_wins_one_goal_overrides(
    events: pd.DataFrame,
    *,
    season: str,
    competition: str,
    round_name: str,
    team_id: int,
    expected_matches: int,
) -> pd.DataFrame:
    """Create an explicit fixed-schedule, one-goal-win counterfactual."""
    required = {
        "match_id",
        "season",
        "competition",
        "round",
        "kickoff_utc",
        "home_team_id",
        "away_team_id",
        "home_team_name",
        "away_team_name",
        "home_goals",
        "away_goals",
    }
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"Counterfactual events are missing columns: {missing}")
    if not isinstance(team_id, int) or team_id < 0:
        raise ValueError("team_id must be a non-negative integer")
    if not isinstance(expected_matches, int) or expected_matches <= 0:
        raise ValueError("expected_matches must be a positive integer")

    rows = events.loc[
        events["season"].astype(str).eq(season)
        & events["competition"].astype(str).str.upper().eq(competition.upper())
        & events["round"].astype(str).str.casefold().eq(round_name.casefold())
        & (
            events["home_team_id"].astype(int).eq(team_id)
            | events["away_team_id"].astype(int).eq(team_id)
        )
    ].copy()
    rows = rows.sort_values(["kickoff_utc", "match_id"], kind="stable")
    if len(rows) != expected_matches:
        raise ValueError(
            f"Expected {expected_matches} counterfactual matches, found {len(rows)}"
        )
    if rows["match_id"].duplicated().any():
        raise ValueError("Counterfactual match_id values must be unique")

    is_home = rows["home_team_id"].astype(int).eq(team_id)
    output = rows[
        [
            "match_id",
            "kickoff_utc",
            "competition",
            "round",
            "home_team_id",
            "away_team_id",
            "home_team_name",
            "away_team_name",
            "home_goals",
            "away_goals",
        ]
    ].rename(
        columns={
            "home_goals": "actual_home_goals",
            "away_goals": "actual_away_goals",
        }
    )
    output["counterfactual_home_goals"] = np.where(is_home, 1, 0)
    output["counterfactual_away_goals"] = np.where(is_home, 0, 1)
    output["counterfactual_winner_team_id"] = team_id
    output["counterfactual_score_rule"] = "ONE_GOAL_WIN_1_0_OR_0_1"
    output["counterfactual_xg_rule"] = "DISABLED_UNKNOWN_COUNTERFACTUAL_XG"
    output["fixture_rule"] = "FIXED_ACTUAL_SCHEDULE_AFTER_LEAGUE_STAGE"
    output["result_changed"] = ~(
        (
            is_home
            & (output["actual_home_goals"] > output["actual_away_goals"])
        )
        | (
            ~is_home
            & (output["actual_away_goals"] > output["actual_home_goals"])
        )
    )
    return output.reset_index(drop=True)


def apply_score_overrides(
    match_ids: Sequence[object],
    home_goals: Sequence[int],
    away_goals: Sequence[int],
    overrides: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply validated score overrides without mutating the caller's arrays."""
    required = {
        "match_id",
        "counterfactual_home_goals",
        "counterfactual_away_goals",
    }
    missing = sorted(required - set(overrides.columns))
    if missing:
        raise ValueError(f"Score overrides are missing columns: {missing}")
    ids = np.asarray(match_ids, dtype=str)
    home = np.asarray(home_goals, dtype=int).copy()
    away = np.asarray(away_goals, dtype=int).copy()
    if len(ids) != len(home) or len(ids) != len(away):
        raise ValueError("match_ids and score arrays must have equal length")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("match_ids must be unique")
    if overrides["match_id"].astype(str).duplicated().any():
        raise ValueError("Override match_id values must be unique")

    positions = {match_id: index for index, match_id in enumerate(ids)}
    for row in overrides.itertuples(index=False):
        match_id = str(row.match_id)
        if match_id not in positions:
            raise ValueError(f"Override match_id is not in replay data: {match_id}")
        home_value = int(row.counterfactual_home_goals)
        away_value = int(row.counterfactual_away_goals)
        if min(home_value, away_value) < 0 or home_value == away_value:
            raise ValueError(f"Counterfactual score must be a non-negative win: {match_id}")
        index = positions[match_id]
        home[index] = home_value
        away[index] = away_value
    return home, away
