from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest

from ao_elo import compute_ao_first_elo_from_csv
from ao_elo.config import AOEuropeanEloConfig
from ao_elo.cup_achievement import (
    CupContributionConfig,
    achievement_delta_to_ao_first_elo,
    candidate_weights,
    champion_equivalent_weight,
    generalized_domestic_achievement,
)
from ao_elo.features import compute_domestic_achievement


ROOT = Path(__file__).resolve().parents[1]
STATIC_SEASON = ROOT / "data" / "backtest_stage_b_2018_2026" / "2023-24"


@pytest.fixture(name="config")
def fixture_config() -> AOEuropeanEloConfig:
    return AOEuropeanEloConfig.v2()


def score(
    config: AOEuropeanEloConfig,
    weight: float,
    *,
    position: object = 5,
    team_count: object = 20,
    champion: bool = False,
    cup: bool = False,
) -> float:
    return generalized_domestic_achievement(
        position,
        team_count,
        champion,
        cup,
        config,
        CupContributionConfig(weight),
    ).domestic_achievement_score


# ---------------------------------------------------------------------------
# the layer must be inert wherever there is no cup
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("weight", candidate_weights())
@pytest.mark.parametrize("position", [1, 4, 11, 20])
def test_non_cup_winners_match_production_exactly(
    config: AOEuropeanEloConfig, weight: float, position: int
) -> None:
    """Without a cup the weighted minimum is zero, so nothing may move."""
    production = compute_domestic_achievement(position, 20, False, False, config)
    candidate = score(config, weight, position=position)

    assert candidate == pytest.approx(
        production.domestic_achievement_score, abs=1e-12
    )


@pytest.mark.parametrize("weight", candidate_weights())
def test_champion_without_cup_matches_production(
    config: AOEuropeanEloConfig, weight: float
) -> None:
    production = compute_domestic_achievement(1, 20, True, False, config)
    candidate = score(config, weight, position=1, champion=True)

    assert candidate == pytest.approx(
        production.domestic_achievement_score, abs=1e-12
    )


# ---------------------------------------------------------------------------
# the combination rule itself
# ---------------------------------------------------------------------------


def test_zero_weight_is_the_pure_maximum(config: AOEuropeanEloConfig) -> None:
    """weight=0 drops the champion-and-cup double bonus the active model adds."""
    production = compute_domestic_achievement(1, 20, True, True, config)
    candidate = score(config, 0.0, position=1, champion=True, cup=True)

    assert candidate == pytest.approx(config.champion_base_score, abs=1e-12)
    assert candidate < production.domestic_achievement_score


def test_champion_equivalent_weight_reproduces_the_double_bonus(
    config: AOEuropeanEloConfig,
) -> None:
    production = compute_domestic_achievement(1, 20, True, True, config)
    weight = champion_equivalent_weight(config)
    candidate = score(config, weight, position=1, champion=True, cup=True)

    assert candidate == pytest.approx(
        production.domestic_achievement_score, abs=1e-12
    )


def test_mid_table_cup_winner_gains_only_under_the_generalization(
    config: AOEuropeanEloConfig,
) -> None:
    """A top-half cup winner is exactly the case the active max rule ignores."""
    production = compute_domestic_achievement(4, 20, False, True, config)
    league_finish = production.league_finish_score

    assert league_finish > config.cup_base_score
    assert production.domestic_achievement_score == pytest.approx(league_finish)

    candidate = score(config, 0.15, position=4, cup=True)
    expected = league_finish + 0.15 * config.cup_base_score

    assert candidate == pytest.approx(expected, abs=1e-12)
    assert candidate > production.domestic_achievement_score


def test_weak_finisher_cup_winner_keeps_the_cup_floor(
    config: AOEuropeanEloConfig,
) -> None:
    """Below the cup base the cup still sets the level and the league adds."""
    candidate = generalized_domestic_achievement(
        18, 20, False, True, config, CupContributionConfig(0.15)
    )
    league_finish = candidate.league_finish_score

    assert league_finish < config.cup_base_score
    assert candidate.domestic_achievement_score == pytest.approx(
        config.cup_base_score + 0.15 * league_finish, abs=1e-12
    )


@pytest.mark.parametrize("position", [2, 7, 13, 19])
def test_score_is_monotone_in_weight_for_cup_winners(
    config: AOEuropeanEloConfig, position: int
) -> None:
    scores = [score(config, w, position=position, cup=True) for w in candidate_weights()]

    assert scores == sorted(scores)


@pytest.mark.parametrize("weight", candidate_weights())
def test_cap_is_never_exceeded(config: AOEuropeanEloConfig, weight: float) -> None:
    candidate = score(config, weight, position=1, champion=True, cup=True)

    assert candidate <= config.achievement_cap + 1e-12


def test_unknown_position_still_resolves(config: AOEuropeanEloConfig) -> None:
    candidate = generalized_domestic_achievement(
        None, None, False, True, config, CupContributionConfig(0.15)
    )

    assert math.isfinite(candidate.domestic_achievement_score)
    assert candidate.domestic_position_percentile is None


# ---------------------------------------------------------------------------
# configuration validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("weight", [-0.01, 1.5, float("nan"), float("inf")])
def test_invalid_weights_are_rejected(weight: float) -> None:
    with pytest.raises(ValueError):
        CupContributionConfig(weight).validate()


def test_boolean_weight_is_rejected() -> None:
    with pytest.raises(ValueError):
        CupContributionConfig(True).validate()


def test_candidate_grid_is_sorted_and_starts_at_zero() -> None:
    weights = candidate_weights()

    assert weights[0] == 0.0
    assert list(weights) == sorted(weights)
    assert len(set(weights)) == len(weights)


# ---------------------------------------------------------------------------
# the linear seed transfer used by the backtest
# ---------------------------------------------------------------------------


def test_zero_achievement_delta_moves_no_rating(config: AOEuropeanEloConfig) -> None:
    assert achievement_delta_to_ao_first_elo(0.0, 0.7, 0.5, config) == 0.0


def test_full_exposure_mutes_the_domestic_change(
    config: AOEuropeanEloConfig,
) -> None:
    """A purely European-evidenced team cannot move on a domestic change."""
    assert achievement_delta_to_ao_first_elo(0.5, 0.7, 1.0, config) == pytest.approx(0.0)


@pytest.mark.parametrize("exposure", [-0.01, 1.01])
def test_invalid_exposure_is_rejected(
    config: AOEuropeanEloConfig, exposure: float
) -> None:
    with pytest.raises(ValueError):
        achievement_delta_to_ao_first_elo(0.1, 0.7, exposure, config)


@pytest.mark.skipif(
    not (STATIC_SEASON / "teams.csv").exists(),
    reason="static backtest inputs are not available",
)
def test_linear_transfer_matches_the_production_pipeline(
    config: AOEuropeanEloConfig,
) -> None:
    """Validate the seed seam end to end through a real config change.

    Moving `cup_base_score` is a genuine pipeline perturbation, so predicting
    the resulting AO First Elo shift from the achievement shift alone proves
    the backtest may re-price seeds without re-implementing the blend.
    """
    altered = replace(config, cup_base_score=0.80)
    baseline = compute_ao_first_elo_from_csv(
        STATIC_SEASON / "teams.csv",
        STATIC_SEASON / "country_coefficients.csv",
        STATIC_SEASON / "domestic_context.csv",
        STATIC_SEASON / "club_european_points.csv",
        config,
    )
    candidate = compute_ao_first_elo_from_csv(
        STATIC_SEASON / "teams.csv",
        STATIC_SEASON / "country_coefficients.csv",
        STATIC_SEASON / "domestic_context.csv",
        STATIC_SEASON / "club_european_points.csv",
        altered,
    )
    merged = baseline[
        [
            "team_id",
            "league_strength",
            "effective_european_exposure",
            "domestic_achievement_score",
            "ao_first_elo_before_domestic_surprise",
        ]
    ].merge(
        candidate[["team_id", "domestic_achievement_score", "ao_first_elo_before_domestic_surprise"]],
        on="team_id",
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    moved = 0
    for row in merged.itertuples(index=False):
        predicted = achievement_delta_to_ao_first_elo(
            row.domestic_achievement_score_candidate
            - row.domestic_achievement_score_baseline,
            row.league_strength,
            row.effective_european_exposure,
            config,
        )
        observed = (
            row.ao_first_elo_before_domestic_surprise_candidate
            - row.ao_first_elo_before_domestic_surprise_baseline
        )
        assert predicted == pytest.approx(observed, abs=1e-9)
        if abs(observed) > 1e-9:
            moved += 1

    assert moved > 0


# ---------------------------------------------------------------------------
# the permanent evaluation ablation arm
# ---------------------------------------------------------------------------


def test_evaluation_registers_the_cup_bonus_ablation() -> None:
    """The arm must exist and be the only one with the bonus disabled."""
    from scripts.run_current_model_evaluation import NO_CUP_BONUS, evaluation_arms

    arms = evaluation_arms()
    disabled = [arm for arm in arms if not arm.cup_double_bonus]

    assert [arm.name for arm in disabled] == [NO_CUP_BONUS]
    assert NO_CUP_BONUS.startswith("ABLATION")


def test_cup_bonus_ablation_keeps_every_other_layer_active() -> None:
    """An ablation isolates one layer; the rest must stay on."""
    from scripts.run_current_model_evaluation import NO_CUP_BONUS, evaluation_arms

    arm = next(item for item in evaluation_arms() if item.name == NO_CUP_BONUS)

    assert arm.domestic_surprise
    assert arm.goal_margin
    assert arm.xg
    assert arm.progression
    assert arm.format_draw


def test_production_arm_keeps_the_double_bonus() -> None:
    """Adding the ablation must not disable the bonus in the shipped arm."""
    from scripts.run_current_model_evaluation import CURRENT, evaluation_arms

    production = next(item for item in evaluation_arms() if item.name == CURRENT)

    assert production.cup_double_bonus is True


def test_cup_bonus_ablation_requires_its_own_seed_map() -> None:
    """Forgetting the seed map must fail loudly, not silently reuse production."""
    from scripts.run_current_model_evaluation import NO_CUP_BONUS, evaluate_arm, evaluation_arms

    arm = next(item for item in evaluation_arms() if item.name == NO_CUP_BONUS)

    with pytest.raises(ValueError, match="without the double bonus"):
        evaluate_arm(
            (),
            arm,
            core=None,
            parameters={},
            current_domestic={},
            baseline_domestic={},
            xg_map={},
            target=None,
            cup_bonus_free_domestic=None,
        )
