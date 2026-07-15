from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_achievement_carry_calibration import (  # noqa: E402
    BASELINE,
    REFERENCE,
    AchievementCarryConfig,
    CarrySeasonData,
    competition_phase,
    candidate_grid,
    club_key,
    evaluate_sequence,
)
from scripts.run_dynamic_core_calibration import DynamicCoreConfig, SeasonData  # noqa: E402


CORE = DynamicCoreConfig(225, 0, 28)


def make_season(
    season: str,
    *,
    home_id: int,
    away_id: int,
    home_key: str,
    away_key: str,
    round_name: str,
    actual: float = 0.5,
    winner_id: int = -1,
) -> CarrySeasonData:
    size = max(home_id, away_id) + 1
    initial = np.full(size, np.nan)
    initial[[home_id, away_id]] = 800.0
    keys = np.full(size, None, dtype=object)
    keys[home_id] = home_key
    keys[away_id] = away_key
    is_final = round_name == "Final"
    core = SeasonData(
        season=season,
        initial_ratings=initial,
        active_team_ids=np.array(sorted((home_id, away_id))),
        home_team_ids=np.array([home_id]),
        away_team_ids=np.array([away_id]),
        actual_home_scores=np.array([actual]),
        neutral_flags=np.array([True]),
        competitions=np.array(["UCL"]),
        match_ids=np.array([f"{season}-m1"]),
    )
    return CarrySeasonData(
        core=core,
        club_keys=keys,
        rounds=np.array([round_name]),
        tie_decider_flags=np.array([is_final]),
        advanced_team_ids=np.array([winner_id]),
    )


def two_season_sequence() -> tuple[CarrySeasonData, ...]:
    champion = "ENG::champion"
    opponent = "ESP::opponent"
    return (
        make_season(
            "2023/24",
            home_id=1,
            away_id=2,
            home_key=champion,
            away_key=opponent,
            round_name="Final",
            winner_id=1,
        ),
        make_season(
            "2024/25",
            home_id=2,
            away_id=1,
            home_key=champion,
            away_key=opponent,
            round_name="Group Stage",
        ),
    )


def test_global_club_key_survives_alias_and_local_id_changes() -> None:
    assert club_key("Dinamo Kiev", "UKR") == club_key("Dynamo Kyiv", "UKR")
    assert club_key("Liverpool", "ENG") == "ENG::liverpool"


def test_candidate_grid_contains_reference_and_expected_size() -> None:
    candidates = candidate_grid()

    assert len(candidates) == 147
    assert BASELINE in candidates
    assert REFERENCE in candidates


def test_trophy_bonus_has_strict_competition_order() -> None:
    config = AchievementCarryConfig(0.5, 0.5, 40)

    assert [config.trophy_bonus(c) for c in ("UCL", "UEL", "UECL")] == [40, 26, 18]


def test_trophy_reserve_is_applied_after_final_and_visible_next_season() -> None:
    metrics, predictions, _ = evaluate_sequence(
        two_season_sequence(),
        CORE,
        AchievementCarryConfig(0, 1, 40),
        evaluation_seasons={"2024/25"},
        return_predictions=True,
    )

    assert predictions is not None
    assert predictions.iloc[0]["home_reserve"] == 40
    assert predictions.iloc[0]["expected_home_score"] > 0.5
    assert metrics["max_reserve"] == 40


def test_baseline_resets_each_season_and_has_no_reserve_effect() -> None:
    _, predictions, _ = evaluate_sequence(
        two_season_sequence(),
        CORE,
        BASELINE,
        evaluation_seasons={"2024/25"},
        return_predictions=True,
    )

    assert predictions is not None
    assert predictions.iloc[0]["home_reserve"] == 0
    assert predictions.iloc[0]["expected_home_score"] == pytest.approx(0.5)


def test_power_carry_uses_global_key_not_local_team_id() -> None:
    first, second = two_season_sequence()
    first = make_season(
        "2023/24",
        home_id=1,
        away_id=2,
        home_key="ENG::champion",
        away_key="ESP::opponent",
        round_name="Final",
        actual=1.0,
        winner_id=1,
    )
    _, predictions, _ = evaluate_sequence(
        (first, second),
        CORE,
        AchievementCarryConfig(1, 0, 0),
        evaluation_seasons={"2024/25"},
        return_predictions=True,
    )

    assert predictions is not None
    assert predictions.iloc[0]["home_power"] > predictions.iloc[0]["away_power"]


def test_reserve_cap_limits_repeated_trophy_accumulation() -> None:
    sequence = two_season_sequence()
    second_final = make_season(
        "2024/25",
        home_id=2,
        away_id=1,
        home_key="ENG::champion",
        away_key="ESP::opponent",
        round_name="Final",
        winner_id=2,
    )
    metrics, _, ratings = evaluate_sequence(
        (sequence[0], second_final),
        CORE,
        AchievementCarryConfig(0, 1, 60),
        evaluation_seasons={"2024/25"},
        return_team_ratings=True,
    )

    champion = ratings.loc[ratings["club_key"].eq("ENG::champion")].iloc[0]
    assert champion["achievement_reserve_end"] == 80
    assert metrics["max_reserve"] == 80


def test_competition_phase_separates_field_strength_grains() -> None:
    assert competition_phase("2nd Qualifying Round") == "QUALIFYING"
    assert competition_phase("League Stage") == "MAIN_STAGE"
    assert competition_phase("Quarter Finals") == "KNOCKOUT"
