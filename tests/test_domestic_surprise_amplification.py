from __future__ import annotations

import pytest

from ao_elo.domestic_surprise_amplification import (
    EXPOSURE_FAMILIES,
    DomesticSurpriseAmplificationConfig,
    calculate_domestic_surprise_amplification,
    exposure_weight,
    production_control_config,
)


def calculate(*, current: float = 1.0, history=(0.5,) * 5, **kwargs):
    return calculate_domestic_surprise_amplification(
        current_finish_score=current,
        historical_finish_scores=history,
        effective_european_exposure=kwargs.pop("exposure", 0.25),
        domestic_achievement_component=594.1770647653,
        achievement_scale=0.75,
        config=kwargs.pop("config", production_control_config()),
    )


def test_production_control_has_documented_active_values() -> None:
    config = production_control_config()
    assert config.theta == 0.40
    assert config.variance_penalty == 0.50
    assert config.domestic_cap == 30.0
    assert config.exposure_family == "LINEAR"
    assert config.final_effect_cap == 75.0


def test_exposure_families_are_bounded_and_monotone() -> None:
    for family in EXPOSURE_FAMILIES:
        values = [exposure_weight(value / 100.0, family) for value in range(101)]
        assert values[0] == pytest.approx(1.0)
        assert all(0.0 <= value <= 1.0 for value in values)
        assert all(left >= right for left, right in zip(values, values[1:]))


def test_positive_and_negative_surprises_are_symmetric() -> None:
    config = DomesticSurpriseAmplificationConfig(theta=1.75, domestic_cap=150.0)
    positive = calculate(current=0.8, history=(0.2,) * 5, exposure=0.2, config=config)
    negative = calculate(current=0.2, history=(0.8,) * 5, exposure=0.2, config=config)
    assert positive.ao_first_elo_adjustment == pytest.approx(
        -negative.ao_first_elo_adjustment
    )


def test_final_effect_cap_and_sign_guard_apply() -> None:
    config = DomesticSurpriseAmplificationConfig(theta=2.0, domestic_cap=150.0)
    adjustment = calculate(current=1.0, history=(0.0,) * 5, exposure=0.0, config=config)
    assert adjustment.ao_first_elo_adjustment == 75.0
    assert adjustment.final_effect_capped
    assert adjustment.ao_first_elo_adjustment * adjustment.raw_surprise > 0.0


def test_incomplete_history_returns_zero_effect() -> None:
    adjustment = calculate(history=(0.4, None, 0.4, 0.4, 0.4))
    assert adjustment.history_seasons == 4
    assert adjustment.domestic_adjustment == 0.0
    assert adjustment.ao_first_elo_adjustment == 0.0


def test_invalid_exposure_is_rejected() -> None:
    with pytest.raises(ValueError, match="effective_european_exposure"):
        exposure_weight(1.01, "LINEAR")

