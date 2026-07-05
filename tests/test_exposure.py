from __future__ import annotations

import pandas as pd
import pytest

from ao_elo.config import AOEuropeanEloConfig, SEASON_KEYS
from ao_elo.features import (
    compute_weighted_match_exposure,
    compute_weighted_season_exposure,
)
from ao_elo.scoring import rating_source_type


@pytest.fixture
def config() -> AOEuropeanEloConfig:
    return AOEuropeanEloConfig(
        country_strength_benchmark=25,
        european_history_benchmark=20,
    )


def club_row(played: list[int], matches: list[int], caps: list[int]) -> pd.Series:
    data = {}
    for key, played_value, match_value, cap_value in zip(
        SEASON_KEYS,
        played,
        matches,
        caps,
        strict=True,
    ):
        data[f"played_{key}"] = played_value
        data[f"matches_{key}"] = match_value
        data[f"match_cap_{key}"] = cap_value
    return pd.Series(data)


def test_no_european_history_has_zero_exposure(config: AOEuropeanEloConfig) -> None:
    row = club_row([0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [6, 6, 6, 8, 8])

    season = compute_weighted_season_exposure(row, config)
    match = compute_weighted_match_exposure(row, config)
    exposure = 0.60 * season + 0.40 * match

    assert exposure == pytest.approx(0.0)
    assert rating_source_type(exposure, 0.75) == "Pure Domestic Projection"


def test_five_seasons_with_few_matches_has_expected_exposure(
    config: AOEuropeanEloConfig,
) -> None:
    row = club_row([1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [6, 6, 6, 8, 8])

    season = compute_weighted_season_exposure(row, config)
    match = compute_weighted_match_exposure(row, config)
    exposure = 0.60 * season + 0.40 * match

    assert season == pytest.approx(1.0)
    assert match == pytest.approx(0.2833333333)
    assert exposure == pytest.approx(0.7133333333)
    assert rating_source_type(exposure, 0.75) == "Mixed Domestic-European Estimate"


def test_five_full_seasons_has_full_exposure(config: AOEuropeanEloConfig) -> None:
    row = club_row([1, 1, 1, 1, 1], [6, 6, 6, 8, 8], [6, 6, 6, 8, 8])

    season = compute_weighted_season_exposure(row, config)
    match = compute_weighted_match_exposure(row, config)
    exposure = 0.60 * season + 0.40 * match

    assert exposure == pytest.approx(1.0)
    assert rating_source_type(exposure, 0.75) == "European Evidence-Based Rating"

