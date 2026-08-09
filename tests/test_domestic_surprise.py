from __future__ import annotations

import math

import pytest

from ao_elo.domestic_surprise import (
    DomesticSurpriseConfig,
    calculate_domestic_surprise_adjustment,
    league_finish_score,
    weighted_historical_finish_score,
)


def adjustment(
    current: float,
    history: list[float | None],
    config: DomesticSurpriseConfig,
    exposure: float = 0.0,
):
    return calculate_domestic_surprise_adjustment(
        current_finish_score=current,
        historical_finish_scores=history,
        domestic_prior=1200.0,
        european_prior=1600.0,
        effective_european_exposure=exposure,
        domestic_achievement_component=600.0,
        achievement_scale=1.0,
        config=config,
    )


def test_champion_finish_score_preserves_intentional_step() -> None:
    assert league_finish_score(1, 20) == 1.0
    assert league_finish_score(2, 20) == pytest.approx(0.8131578947368421)


def test_positive_surprise_rewards_usual_sixth_more_than_usual_second() -> None:
    config = DomesticSurpriseConfig(0.20, 0.20, 3, 2, 100.0)
    usual_sixth = league_finish_score(6, 20)
    usual_second = league_finish_score(2, 20)
    outsider = adjustment(1.0, [usual_sixth] * 3, config)
    established = adjustment(1.0, [usual_second] * 3, config)
    assert outsider.domestic_prior_adjustment > established.domestic_prior_adjustment > 0


def test_negative_surprise_reduces_domestic_prior_but_not_european_prior() -> None:
    config = DomesticSurpriseConfig(0.20, 0.30, 3, 2, 100.0)
    result = adjustment(league_finish_score(8, 20), [league_finish_score(2, 20)] * 3, config)
    assert result.domestic_prior_adjustment < 0
    assert result.adjusted_domestic_prior < 1200.0
    assert result.adjusted_ao_first_elo == result.adjusted_domestic_prior


def test_european_exposure_dampens_domestic_surprise_effect() -> None:
    config = DomesticSurpriseConfig(0.30, 0.30, 3, 2, 100.0)
    no_exposure = adjustment(1.0, [0.5, 0.5, 0.5], config, exposure=0.0)
    high_exposure = adjustment(1.0, [0.5, 0.5, 0.5], config, exposure=0.85)
    no_exposure_shift = no_exposure.adjusted_ao_first_elo - 1200.0
    baseline_high_exposure = 1200.0 + 0.85 * (1600.0 - 1200.0)
    high_exposure_shift = high_exposure.adjusted_ao_first_elo - baseline_high_exposure
    assert high_exposure_shift == pytest.approx(0.15 * no_exposure_shift)


def test_insufficient_history_produces_zero_adjustment() -> None:
    config = DomesticSurpriseConfig(0.40, 0.40, 4, 2, 75.0)
    result = adjustment(1.0, [None, None, None, 0.4], config)
    assert result.history_seasons == 1
    assert result.domestic_prior_adjustment == 0
    assert result.adjusted_ao_first_elo == 1200.0


def test_adjustment_is_capped_in_both_directions() -> None:
    config = DomesticSurpriseConfig(1.0, 1.0, 2, 2, 25.0)
    positive = adjustment(1.0, [0.15, 0.15], config)
    negative = adjustment(0.15, [1.0, 1.0], config)
    assert positive.domestic_prior_adjustment == 25.0
    assert negative.domestic_prior_adjustment == -25.0


def test_historical_score_renormalizes_available_recency_weights() -> None:
    score, count, reliability = weighted_historical_finish_score(
        [None, 0.4, 0.8], 3
    )
    expected = (0.4 * 0.27 + 0.8 * 0.33) / (0.27 + 0.33)
    assert score == pytest.approx(expected)
    assert count == 2
    assert reliability == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    "config",
    [
        DomesticSurpriseConfig(-0.1, 0.0),
        DomesticSurpriseConfig(0.0, -0.1),
        DomesticSurpriseConfig(0.1, 0.1, 5),
        DomesticSurpriseConfig(0.1, 0.1, 2, 3),
        DomesticSurpriseConfig(0.1, 0.1, 2, 2, math.inf),
    ],
)
def test_invalid_config_is_rejected(config: DomesticSurpriseConfig) -> None:
    with pytest.raises(ValueError):
        config.validate()
