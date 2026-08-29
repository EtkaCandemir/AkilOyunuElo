from __future__ import annotations

from itertools import product

import pandas as pd
import pytest

from ao_elo.config import AOEuropeanEloConfig, SEASON_KEYS
from ao_elo.features import (
    compute_weighted_match_exposure,
    compute_weighted_season_exposure,
    weighted_sum,
)
from ao_elo.scoring import rating_source_type


def test_all_participation_patterns_preserve_numeric_accumulation_exactly() -> None:
    config = AOEuropeanEloConfig.active()
    for flags in product((0, 1), repeat=len(SEASON_KEYS)):
        numeric = club_row(list(flags), list(flags), [10] * len(flags))
        strings = numeric.copy().astype(object)
        for key, value in zip(SEASON_KEYS, flags, strict=True):
            strings[f"played_{key}"] = "true" if value else "false"
        expected = weighted_sum(numeric, "played", config)
        assert compute_weighted_season_exposure(numeric, config) == expected
        assert compute_weighted_season_exposure(strings, config) == expected


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


@pytest.mark.parametrize(
    (
        "played",
        "matches",
        "caps",
        "expected_season",
        "expected_match",
        "expected_final",
        "source",
    ),
    [
        (
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [6, 6, 6, 8, 8],
            0.0,
            0.0,
            0.0,
            "Pure Domestic Projection",
        ),
        (
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 8],
            [6, 6, 6, 8, 8],
            0.33,
            0.33,
            0.33,
            "Mixed Domestic-European Estimate",
        ),
        (
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 6],
            [6, 6, 6, 6, 6],
            0.33,
            0.33,
            0.33,
            "Mixed Domestic-European Estimate",
        ),
        (
            [0, 0, 0, 1, 1],
            [0, 0, 0, 8, 8],
            [6, 6, 6, 8, 8],
            0.60,
            0.60,
            0.60,
            "Mixed Domestic-European Estimate",
        ),
        (
            [1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2],
            [6, 6, 6, 8, 8],
            1.0,
            0.2833333333,
            0.7133333333,
            "Mixed Domestic-European Estimate",
        ),
        (
            [1, 1, 1, 1, 1],
            [6, 6, 6, 8, 8],
            [6, 6, 6, 8, 8],
            1.0,
            1.0,
            1.0,
            "European Evidence-Based Rating",
        ),
        (
            [1, 1, 1, 0, 0],
            [6, 6, 6, 0, 0],
            [6, 6, 6, 8, 8],
            0.40,
            0.40,
            0.40,
            "Mixed Domestic-European Estimate",
        ),
    ],
)
def test_exposure_reference_table(
    config: AOEuropeanEloConfig,
    played: list[int],
    matches: list[int],
    caps: list[int],
    expected_season: float,
    expected_match: float,
    expected_final: float,
    source: str,
) -> None:
    row = club_row(played, matches, caps)
    season = compute_weighted_season_exposure(row, config)
    match = compute_weighted_match_exposure(row, config)
    exposure = 0.60 * season + 0.40 * match

    assert season == pytest.approx(expected_season)
    assert match == pytest.approx(expected_match)
    assert exposure == pytest.approx(expected_final)
    assert rating_source_type(exposure, 0.75) == source
