import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ao_elo.team_venue_context import (
    TeamVenueContextConfig,
    contextual_home_expectation,
    estimate_team_venue_effect,
)
from scripts.run_team_venue_context_backtest import (
    BASELINE_KEY,
    MODEL_BASELINE,
    MODEL_NESTED,
    build_season_profiles,
    candidate_grid,
    prediction_only_state_invariant,
)


CONFIG = TeamVenueContextConfig(3, 0.75, 10.0, 100.0, 100.0)


def test_no_history_returns_global_prior_deviation_zero() -> None:
    estimate = estimate_team_venue_effect(
        [], [], [], elo_scale=835.561497, shrinkage_matches=10.0, max_team_effect=100.0
    )
    assert estimate.observations == 0
    assert estimate.reliability == 0.0
    assert estimate.shrunk_effect == 0.0


def test_repeated_overperformance_produces_positive_effect() -> None:
    positive = estimate_team_venue_effect(
        [0.40] * 8,
        [1.0] * 6 + [0.5, 0.0],
        [1.0] * 8,
        elo_scale=835.561497,
        shrinkage_matches=10.0,
        max_team_effect=100.0,
    )
    negative = estimate_team_venue_effect(
        [0.70] * 8,
        [0.0] * 5 + [0.5, 1.0, 1.0],
        [1.0] * 8,
        elo_scale=835.561497,
        shrinkage_matches=10.0,
        max_team_effect=100.0,
    )
    assert positive.raw_effect > 0.0
    assert positive.shrunk_effect > 0.0
    assert negative.raw_effect < 0.0
    assert negative.shrunk_effect < 0.0


def test_shrinkage_reduces_small_sample_effect() -> None:
    small = estimate_team_venue_effect(
        [0.50],
        [1.0],
        [1.0],
        elo_scale=835.561497,
        shrinkage_matches=15.0,
        max_team_effect=150.0,
    )
    large = estimate_team_venue_effect(
        [0.50] * 15,
        [1.0] * 15,
        [1.0] * 15,
        elo_scale=835.561497,
        shrinkage_matches=15.0,
        max_team_effect=150.0,
    )
    assert small.reliability < large.reliability
    assert abs(small.shrunk_effect) < abs(large.shrunk_effect)


def test_home_effect_raises_and_away_strength_lowers_home_expectation() -> None:
    stronger_home = contextual_home_expectation(
        0.60,
        60.0,
        0.0,
        global_home_advantage=148.544266,
        elo_scale=835.561497,
        is_neutral=False,
        config=CONFIG,
    )
    stronger_away = contextual_home_expectation(
        0.60,
        0.0,
        60.0,
        global_home_advantage=148.544266,
        elo_scale=835.561497,
        is_neutral=False,
        config=CONFIG,
    )
    assert stronger_home.expected_home_score > 0.60
    assert stronger_away.expected_home_score < 0.60
    assert stronger_home.effective_home_advantage == pytest.approx(208.544266)
    assert stronger_away.effective_home_advantage == pytest.approx(88.544266)


def test_neutral_match_ignores_both_team_effects() -> None:
    result = contextual_home_expectation(
        0.50,
        100.0,
        -100.0,
        global_home_advantage=148.544266,
        elo_scale=835.561497,
        is_neutral=True,
        config=CONFIG,
    )
    assert result.expected_home_score == pytest.approx(0.50)
    assert result.applied_context_offset == 0.0
    assert result.effective_home_advantage == 0.0


def test_context_and_home_advantage_caps_are_enforced() -> None:
    result = contextual_home_expectation(
        0.60,
        500.0,
        -500.0,
        global_home_advantage=250.0,
        elo_scale=835.561497,
        is_neutral=False,
        config=CONFIG,
    )
    assert result.raw_context_offset == pytest.approx(1000.0)
    assert result.effective_home_advantage == pytest.approx(300.0)
    assert result.applied_context_offset == pytest.approx(50.0)


@pytest.mark.parametrize(
    "config",
    [
        TeamVenueContextConfig(0, 0.75, 10.0, 100.0, 100.0),
        TeamVenueContextConfig(3, 0.0, 10.0, 100.0, 100.0),
        TeamVenueContextConfig(3, 1.1, 10.0, 100.0, 100.0),
        TeamVenueContextConfig(3, 0.75, -1.0, 100.0, 100.0),
        TeamVenueContextConfig(3, 0.75, 10.0, -1.0, 100.0),
        TeamVenueContextConfig(3, 0.75, 10.0, 100.0, 100.0, 301.0, 300.0),
    ],
)
def test_config_validation(config: TeamVenueContextConfig) -> None:
    with pytest.raises(ValueError):
        config.validate()


def test_candidate_grid_includes_global_control_and_96_venue_candidates() -> None:
    candidates = candidate_grid()
    assert len(candidates) == 97
    assert candidates[0].key == BASELINE_KEY
    assert len({candidate.key for candidate in candidates}) == 97
    assert {candidate.config.max_context_offset for candidate in candidates[1:]} == {
        25.0,
        35.0,
        50.0,
        75.0,
        100.0,
        150.0,
    }


def test_season_profiles_use_only_each_clubs_own_prior_history() -> None:
    history = pd.DataFrame(
        [
            {
                "season": "2022/23",
                "home_club_id": "A",
                "away_club_id": "B",
                "expected_home_score": 0.50,
                "actual_home_score": 1.0,
            },
            {
                "season": "2023/24",
                "home_club_id": "B",
                "away_club_id": "A",
                "expected_home_score": 0.60,
                "actual_home_score": 0.0,
            },
        ]
    )
    season_index = {"2022/23": 0, "2023/24": 1, "2024/25": 2}
    profiles = build_season_profiles(
        history,
        ["A", "B", "NEW"],
        "2024/25",
        2,
        season_index,
        835.561497,
        CONFIG,
    ).set_index("club_id")

    assert profiles.loc["A", "history_max_season"] == "2023/24"
    assert profiles.loc["B", "history_max_season"] == "2023/24"
    assert pd.isna(profiles.loc["NEW", "history_max_season"])
    assert profiles.loc["NEW", "home_observations"] == 0
    assert profiles.loc["NEW", "away_observations"] == 0
    assert profiles.loc["NEW", "home_effect_centered"] == 0.0
    assert profiles.loc["NEW", "away_effect_centered"] == 0.0


def test_target_season_data_cannot_enter_profile_history() -> None:
    history = pd.DataFrame(
        [
            {
                "season": "2024/25",
                "home_club_id": "A",
                "away_club_id": "B",
                "expected_home_score": 0.50,
                "actual_home_score": 1.0,
            }
        ]
    )
    with pytest.raises(ValueError, match="current or future season"):
        build_season_profiles(
            history,
            ["A", "B"],
            "2024/25",
            1,
            {"2023/24": 0, "2024/25": 1},
            835.561497,
            CONFIG,
        )


def test_prediction_only_state_invariant_detects_rating_update_changes() -> None:
    shared = {
        "match_id": "M1",
        "home_live_pre": 1500.0,
        "away_live_pre": 1400.0,
        "expected_home_score": 0.60,
        "power_delta": 20.0,
        "goal_multiplier": 1.10,
        "xg_performance_adjustment": 2.0,
        "bonus_applied_after_match": 0.0,
        "bonus_winner_id": pd.NA,
    }
    predictions = pd.DataFrame(
        [{"model": MODEL_BASELINE, **shared}, {"model": MODEL_NESTED, **shared}]
    )
    assert prediction_only_state_invariant(predictions)

    predictions.loc[predictions["model"].eq(MODEL_NESTED), "power_delta"] = 20.1
    assert not prediction_only_state_invariant(predictions)
