from __future__ import annotations

import pandas as pd
import pytest

from ao_elo.config import AOEuropeanEloConfig, SEASON_KEYS
from ao_elo.pipeline import compute_ao_first_elo
from ao_elo.scoring import compute_ao_first_elo as final_formula


def test_final_rating_shrinks_between_domestic_and_european_prior() -> None:
    assert final_formula(700, 820, 0.00) == pytest.approx(700)
    assert final_formula(700, 820, 0.25) == pytest.approx(730)
    assert final_formula(700, 820, 1.00) == pytest.approx(820)
    assert final_formula(700, 600, 0.75) == pytest.approx(625)


def test_invalid_benchmarks_raise() -> None:
    config = AOEuropeanEloConfig(
        country_strength_benchmark=None,
        european_history_benchmark=20,
    )

    with pytest.raises(ValueError, match="Country_Strength_Benchmark"):
        config.validate()


def test_pipeline_no_european_history_equals_domestic_prior() -> None:
    output = compute_ao_first_elo(
        teams=pd.DataFrame(
            [
                {
                    "team_id": 1,
                    "team_name": "No Europe FC",
                    "country": "Exampleland",
                    "country_code": "EX",
                    "domestic_league": "Example League",
                }
            ]
        ),
        country_coefficients=pd.DataFrame(
            [
                {
                    "season": "2025/26",
                    "country": "Exampleland",
                    "country_code": "EX",
                    "points_t_minus_4": 3,
                    "points_t_minus_3": 4,
                    "points_t_minus_2": 5,
                    "points_t_minus_1": 6,
                    "points_t": 7,
                    "official_five_year_total": 25,
                    "official_country_rank": 12,
                }
            ]
        ),
        domestic_context=pd.DataFrame(
            [
                {
                    "season": "2025/26",
                    "team_id": 1,
                    "domestic_position": 1,
                    "league_team_count": 20,
                    "is_league_champion": True,
                    "is_cup_winner": False,
                    "european_entry_type": "League Champion",
                }
            ]
        ),
        club_european_points=zero_club_history(),
        config=AOEuropeanEloConfig(
            country_strength_benchmark=25,
            european_history_benchmark=20,
        ),
    )

    row = output.iloc[0]
    assert row["european_exposure"] == pytest.approx(0.0)
    assert row["ao_first_elo"] == pytest.approx(row["domestic_prior"])
    assert row["rating_source_type"] == "Pure Domestic Projection"
    assert row["season"] == "2025/26"
    assert row["domestic_league"] == "Example League"
    assert row["domestic_position"] == 1
    assert row["league_team_count"] == 20


def zero_club_history() -> pd.DataFrame:
    row: dict[str, object] = {
        "season": "2025/26",
        "team_id": 1,
        "team_name_source": "No Europe FC",
        "country_code": "EX",
    }
    for key in SEASON_KEYS:
        row[f"club_points_{key}"] = 0
        row[f"played_{key}"] = 0
        row[f"matches_{key}"] = 0
        row[f"match_cap_{key}"] = 6
    return pd.DataFrame([row])
