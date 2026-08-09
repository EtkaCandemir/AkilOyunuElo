from __future__ import annotations

import pytest

from ao_elo.domestic_surprise_variance import (
    VarianceDomesticSurpriseConfig,
    calculate_variance_domestic_surprise_adjustment,
    weighted_five_season_distribution,
)


def calculate(
    current: float,
    history: list[float | None],
    *,
    coefficient: float = 0.20,
    variance_penalty: float = 0.50,
    exposure: float = 0.0,
):
    return calculate_variance_domestic_surprise_adjustment(
        current_finish_score=current,
        historical_finish_scores=history,
        domestic_prior=1200.0,
        european_prior=1600.0,
        effective_european_exposure=exposure,
        domestic_achievement_component=600.0,
        achievement_scale=1.0,
        config=VarianceDomesticSurpriseConfig(
            coefficient=coefficient,
            variance_penalty=variance_penalty,
            max_abs_adjustment=100.0,
        ),
    )


def test_constant_history_has_zero_variance() -> None:
    mean, variance, volatility, count = weighted_five_season_distribution([0.5] * 5)
    assert mean == pytest.approx(0.5)
    assert variance == pytest.approx(0.0)
    assert volatility == pytest.approx(0.0)
    assert count == 5


def test_newest_season_has_033_weight() -> None:
    mean, _, _, _ = weighted_five_season_distribution([0, 0, 0, 0, 1])
    assert mean == pytest.approx(0.33)


def test_variance_never_reverses_surprise_direction() -> None:
    positive = calculate(0.9, [0.1, 0.9, 0.1, 0.9, 0.1], variance_penalty=1.0)
    negative = calculate(0.1, [0.9, 0.1, 0.9, 0.1, 0.9], variance_penalty=1.0)
    assert positive.domestic_prior_adjustment >= 0
    assert negative.domestic_prior_adjustment <= 0


def test_volatile_history_shrinks_equal_raw_surprise() -> None:
    stable = calculate(0.7, [0.5] * 5)
    volatile_history = [0.2, 0.8, 0.2, 0.8, 0.2]
    mean, _, _, _ = weighted_five_season_distribution(volatile_history)
    volatile = calculate(float(mean) + 0.2, volatile_history)
    assert volatile.raw_surprise == pytest.approx(stable.raw_surprise)
    assert abs(volatile.domestic_prior_adjustment) < abs(
        stable.domestic_prior_adjustment
    )


def test_positive_and_negative_effects_are_symmetric() -> None:
    positive = calculate(0.75, [0.5] * 5)
    negative = calculate(0.25, [0.5] * 5)
    assert positive.domestic_prior_adjustment == pytest.approx(
        -negative.domestic_prior_adjustment
    )


def test_missing_history_disables_adjustment() -> None:
    result = calculate(0.8, [None, 0.5, 0.5, 0.5, 0.5])
    assert result.history_seasons == 4
    assert result.domestic_prior_adjustment == 0.0


def test_european_exposure_preserves_existing_linear_blend() -> None:
    no_exposure = calculate(0.75, [0.5] * 5, exposure=0.0)
    high_exposure = calculate(0.75, [0.5] * 5, exposure=0.85)
    baseline_high = 1200.0 + 0.85 * 400.0
    assert high_exposure.adjusted_ao_first_elo - baseline_high == pytest.approx(
        0.15 * no_exposure.domestic_prior_adjustment
    )


@pytest.mark.parametrize(
    "config",
    [
        VarianceDomesticSurpriseConfig(coefficient=-0.1),
        VarianceDomesticSurpriseConfig(variance_penalty=-0.1),
        VarianceDomesticSurpriseConfig(variance_penalty=1.1),
        VarianceDomesticSurpriseConfig(max_abs_adjustment=-1),
        VarianceDomesticSurpriseConfig(minimum_history_seasons=4),
    ],
)
def test_invalid_config_is_rejected(config: VarianceDomesticSurpriseConfig) -> None:
    with pytest.raises(ValueError):
        config.validate()
