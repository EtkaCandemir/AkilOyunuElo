from __future__ import annotations

import pandas as pd
import pytest

from ao_elo.relative_opponent_profile import (
    EVEN_OPPONENT,
    STRONG_OPPONENT,
    WEAK_OPPONENT,
    DomesticRatingConfig,
    RelativeOpponentProfileConfig,
    classify_relative_opponent,
    estimate_relative_profile,
    replay_domestic_perspectives,
)


def test_relative_band_uses_neutral_expected_strength() -> None:
    assert classify_relative_opponent(0.34, lower=0.35, upper=0.65) == STRONG_OPPONENT
    assert classify_relative_opponent(0.50, lower=0.35, upper=0.65) == EVEN_OPPONENT
    assert classify_relative_opponent(0.66, lower=0.35, upper=0.65) == WEAK_OPPONENT


def test_same_timestamp_domestic_matches_share_pre_match_snapshot() -> None:
    matches = pd.DataFrame(
        [
            {
                "match_id": "M1", "sportsdb_league_id": "L", "provider_season": "2020-2021", "ao_season": "2020/21",
                "kickoff_utc": "2020-08-01T14:00:00Z", "home_source_team_id": "A", "away_source_team_id": "B",
                "home_ao_club_id": "AO-A", "away_ao_club_id": "AO-B", "actual_home_score": 1.0, "country_code": "ENG",
            },
            {
                "match_id": "M2", "sportsdb_league_id": "L", "provider_season": "2020-2021", "ao_season": "2020/21",
                "kickoff_utc": "2020-08-01T14:00:00Z", "home_source_team_id": "C", "away_source_team_id": "D",
                "home_ao_club_id": "AO-C", "away_ao_club_id": "AO-D", "actual_home_score": 0.0, "country_code": "ENG",
            },
        ]
    )
    result = replay_domestic_perspectives(
        matches,
        config=DomesticRatingConfig(835.561497, 148.544266, 103.980986),
        lower=0.35,
        upper=0.65,
    )
    home = result.loc[result["club_id"].isin(("AO-A", "AO-C"))]
    assert home["expected_score"].nunique() == 1
    assert home.iloc[0]["expected_score"] > 0.5
    assert set(home["venue"]) == {"HOME"}
    assert set(result.loc[result["club_id"].isin(("AO-B", "AO-D")), "venue"]) == {"AWAY"}


def test_profile_shrinks_and_centers_relative_effects() -> None:
    history = pd.DataFrame(
        [
            {"opponent_band": STRONG_OPPONENT, "expected_score": 0.30, "actual_score": 1.0, "weight": 1.0},
            {"opponent_band": STRONG_OPPONENT, "expected_score": 0.30, "actual_score": 0.5, "weight": 1.0},
            {"opponent_band": WEAK_OPPONENT, "expected_score": 0.70, "actual_score": 0.0, "weight": 1.0},
            {"opponent_band": WEAK_OPPONENT, "expected_score": 0.70, "actual_score": 0.5, "weight": 1.0},
        ]
    )
    config = RelativeOpponentProfileConfig(0.35, 0.65, 3, 1.0, 1.0, 2.0, 50.0)
    profile = estimate_relative_profile("AO-A", history, config=config, elo_scale=835.561497)
    assert profile.effect_for(STRONG_OPPONENT) > 0.0
    assert profile.effect_for(WEAK_OPPONENT) < 0.0
    assert profile.effect_for(EVEN_OPPONENT) == pytest.approx(0.0)


def test_empty_profile_history_produces_zero_effects() -> None:
    config = RelativeOpponentProfileConfig(0.35, 0.65, 3, 1.0, 1.0, 20.0, 25.0)
    profile = estimate_relative_profile(
        "AO-NEW",
        pd.DataFrame(),
        config=config,
        elo_scale=835.561497,
    )
    assert profile.effects == pytest.approx((0.0, 0.0, 0.0))
