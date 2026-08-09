from __future__ import annotations

import pytest

from ao_elo.domestic_surprise_symmetric import (
    FIVE_SEASON_WEIGHTS,
    SymmetricDomesticSurpriseConfig,
    calculate_symmetric_domestic_surprise_adjustment,
    direct_league_position_score,
    normalized_finish_score,
    weighted_five_season_finish_score,
)


def calculate(current: float, history: list[float | None], exposure: float = 0.0):
    return calculate_symmetric_domestic_surprise_adjustment(
        current_finish_score=current,
        historical_finish_scores=history,
        domestic_prior=1200.0,
        european_prior=1600.0,
        effective_european_exposure=exposure,
        domestic_achievement_component=600.0,
        achievement_scale=1.0,
        config=SymmetricDomesticSurpriseConfig(
            coefficient=0.20,
            normalization="direct_percentile",
            minimum_history_seasons=5,
            max_abs_adjustment=100.0,
        ),
    )


def test_five_season_weights_match_contract() -> None:
    assert FIVE_SEASON_WEIGHTS == (0.07, 0.13, 0.20, 0.27, 0.33)
    assert sum(FIVE_SEASON_WEIGHTS) == pytest.approx(1.0)


def test_direct_position_score_has_no_champion_step() -> None:
    assert direct_league_position_score(1, 20) == 1.0
    assert direct_league_position_score(2, 20) == pytest.approx(18 / 19)
    assert direct_league_position_score(20, 20) == 0.0
    assert normalized_finish_score(2, 20, "legacy_finish_curve") == pytest.approx(
        0.8131578947368421
    )


def test_weighted_history_uses_newest_to_oldest_contract() -> None:
    score, count, coverage = weighted_five_season_finish_score([0, 0, 0, 0, 1])
    assert score == pytest.approx(0.33)
    assert count == 5
    assert coverage == pytest.approx(1.0)


def test_equal_surprises_have_equal_opposite_adjustments() -> None:
    positive = calculate(0.75, [0.5] * 5)
    negative = calculate(0.25, [0.5] * 5)
    assert positive.domestic_prior_adjustment == pytest.approx(
        -negative.domestic_prior_adjustment
    )


def test_strict_five_season_requirement_disables_partial_history() -> None:
    result = calculate(0.75, [None, 0.5, 0.5, 0.5, 0.5])
    assert result.history_seasons == 4
    assert result.domestic_prior_adjustment == 0.0


def test_european_exposure_attenuates_only_final_elo_effect() -> None:
    no_exposure = calculate(0.75, [0.5] * 5, exposure=0.0)
    high_exposure = calculate(0.75, [0.5] * 5, exposure=0.85)
    baseline_high = 1200.0 + 0.85 * 400.0
    assert high_exposure.domestic_prior_adjustment == pytest.approx(
        no_exposure.domestic_prior_adjustment
    )
    assert high_exposure.adjusted_ao_first_elo - baseline_high == pytest.approx(
        0.15 * no_exposure.domestic_prior_adjustment
    )


def test_exposure_power_controls_domestic_signal_retention() -> None:
    config = SymmetricDomesticSurpriseConfig(
        coefficient=0.20,
        normalization="direct_percentile",
        minimum_history_seasons=5,
        max_abs_adjustment=100.0,
        exposure_power=0.5,
    )
    result = calculate_symmetric_domestic_surprise_adjustment(
        current_finish_score=0.75,
        historical_finish_scores=[0.5] * 5,
        domestic_prior=1200.0,
        european_prior=1600.0,
        effective_european_exposure=0.75,
        domestic_achievement_component=600.0,
        achievement_scale=1.0,
        config=config,
    )
    baseline = 1200.0 + 0.75 * 400.0
    assert result.adjusted_ao_first_elo - baseline == pytest.approx(
        0.5 * result.domestic_prior_adjustment
    )


@pytest.mark.parametrize(
    "position, teams",
    [(0, 20), (21, 20), (1, 1), (True, 20), (1.5, 20)],
)
def test_invalid_direct_position_is_rejected(position: object, teams: object) -> None:
    with pytest.raises(ValueError):
        direct_league_position_score(position, teams)  # type: ignore[arg-type]
