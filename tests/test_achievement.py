from __future__ import annotations

import pytest

from ao_elo.config import AOEuropeanEloConfig
from ao_elo.features import compute_domestic_achievement
from ao_elo.validators import validate_domestic_context


@pytest.fixture
def config() -> AOEuropeanEloConfig:
    return AOEuropeanEloConfig(
        country_strength_benchmark=25,
        european_history_benchmark=20,
    )


def test_unknown_domestic_finish_uses_unknown_score(config: AOEuropeanEloConfig) -> None:
    result = compute_domestic_achievement(None, None, False, False, config)

    assert result.domestic_position_percentile is None
    assert result.league_finish_score == pytest.approx(0.10)
    assert result.domestic_achievement_score == pytest.approx(0.10)


def test_unknown_finish_plus_cup_gets_no_double_bonus(
    config: AOEuropeanEloConfig,
) -> None:
    result = compute_domestic_achievement(None, None, False, True, config)

    assert result.league_finish_score == pytest.approx(0.10)
    assert result.cup_base_score == pytest.approx(0.62)
    assert result.cup_double_bonus == pytest.approx(0.0)
    assert result.domestic_achievement_score == pytest.approx(0.62)


def test_champion_position_mismatch_is_rejected() -> None:
    import pandas as pd

    with pytest.raises(ValueError, match="requires domestic_position=1"):
        validate_domestic_context(
            pd.DataFrame(
                [
                    {
                        "season": "2025/26",
                        "team_id": 1,
                        "domestic_position": 2,
                        "league_team_count": 20,
                        "is_league_champion": True,
                        "is_cup_winner": False,
                        "european_entry_type": "League Champion",
                    }
                ]
            )
        )


def test_champion_without_position_keeps_champion_base(
    config: AOEuropeanEloConfig,
) -> None:
    result = compute_domestic_achievement(None, None, True, False, config)

    assert result.domestic_position_percentile is None
    assert result.league_finish_score == pytest.approx(1.0)
    assert result.domestic_achievement_score == pytest.approx(1.0)


@pytest.mark.parametrize(
    (
        "position",
        "team_count",
        "champion",
        "cup_winner",
        "expected_finish",
        "expected_bonus",
        "expected_achievement",
    ),
    [
        (1, 20, True, False, 1.0, 0.0, 1.0),
        (1, 20, True, True, 1.0, 0.08, 1.08),
        (2, 20, False, False, 0.8131578947, 0.0, 0.8131578947),
        (2, 20, False, True, 0.8131578947, 0.0, 0.8131578947),
        (2, 6, False, False, 0.71, 0.0, 0.71),
        (6, 6, False, False, 0.15, 0.0, 0.15),
        (15, 20, False, True, 0.3342105263, 0.0, 0.62),
        (None, None, False, False, 0.10, 0.0, 0.10),
        (None, None, False, True, 0.10, 0.0, 0.62),
        (None, None, True, False, 1.0, 0.0, 1.0),
        (None, None, True, True, 1.0, 0.08, 1.08),
    ],
)
def test_domestic_achievement_reference_table(
    config: AOEuropeanEloConfig,
    position: int | None,
    team_count: int | None,
    champion: bool,
    cup_winner: bool,
    expected_finish: float,
    expected_bonus: float,
    expected_achievement: float,
) -> None:
    result = compute_domestic_achievement(
        position,
        team_count,
        champion,
        cup_winner,
        config,
    )

    assert result.league_finish_score == pytest.approx(expected_finish)
    assert result.cup_double_bonus == pytest.approx(expected_bonus)
    assert result.domestic_achievement_score == pytest.approx(expected_achievement)
