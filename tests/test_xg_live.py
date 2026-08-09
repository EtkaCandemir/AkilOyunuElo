from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ao_elo.controlled_live import update_match_elo
from ao_elo.xg_live import (
    UnsupportedMarginConfig,
    XGBlendConfig,
    XGPerformanceBonusConfig,
    calculate_unsupported_margin_penalty,
    calculate_xg_performance_adjustment,
    calculate_xg_performance_score,
    update_match_elo_with_xg,
)
from scripts.run_xg_goal_ablation_backtest import (
    candidate_grid,
    elo_field_score,
)


BASE_ARGS = {
    "home_rating": 1500.0,
    "away_rating": 1500.0,
    "home_goals": 3,
    "away_goals": 0,
    "k_factor": 103.98098633392752,
    "elo_scale": 835.5614973262034,
    "home_advantage": 148.54426619132505,
    "is_neutral": False,
    "decided_on_penalties": False,
    "goal_difference_enabled": True,
    "goal_alpha": 0.10,
    "goal_tau": 300.0,
    "goal_difference_cap": 4,
    "xg_config": XGBlendConfig(0.10, 1.0),
    "xg_home": 2.5,
    "xg_away": 0.5,
}


def test_xg_score_is_balanced_at_equal_xg() -> None:
    assert calculate_xg_performance_score(1.2, 1.2, 1.0) == pytest.approx(0.5)


def test_xg_score_is_symmetric() -> None:
    home = calculate_xg_performance_score(2.4, 0.7, 1.25)
    away = calculate_xg_performance_score(0.7, 2.4, 1.25)
    assert home + away == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("home_xg", "away_xg", "scale"),
    [
        (-0.1, 1.0, 1.0),
        (1.0, -0.1, 1.0),
        (math.nan, 1.0, 1.0),
        (1.0, math.inf, 1.0),
        (1.0, 1.0, 0.0),
    ],
)
def test_xg_score_rejects_invalid_values(
    home_xg: float,
    away_xg: float,
    scale: float,
) -> None:
    with pytest.raises(ValueError):
        calculate_xg_performance_score(home_xg, away_xg, scale)


@pytest.mark.parametrize(
    "config",
    [
        XGBlendConfig(-0.1, 1.0),
        XGBlendConfig(1.1, 1.0),
        XGBlendConfig(0.1, 0.0),
        XGBlendConfig(0.1, math.inf),
    ],
)
def test_xg_config_validation(config: XGBlendConfig) -> None:
    with pytest.raises(ValueError):
        config.validate()


def test_rho_zero_matches_controlled_goal_difference_update() -> None:
    expected = update_match_elo(
        1500.0,
        1500.0,
        3,
        0,
        k_factor=103.98098633392752,
        elo_scale=835.5614973262034,
        home_advantage=148.54426619132505,
        is_neutral=False,
        decided_on_penalties=False,
        alpha=0.10,
        tau=300.0,
    )
    actual = update_match_elo_with_xg(
        **{
            **BASE_ARGS,
            "xg_config": XGBlendConfig(0.0, 1.0),
        }
    )
    assert actual.power_delta == pytest.approx(expected.power_delta)
    assert actual.goal_difference_multiplier == pytest.approx(
        expected.goal_difference_multiplier
    )


def test_missing_xg_falls_back_to_result_arm() -> None:
    with_xg_disabled = update_match_elo_with_xg(
        **{
            **BASE_ARGS,
            "xg_config": XGBlendConfig(0.0, 1.0),
            "xg_home": None,
            "xg_away": None,
        }
    )
    missing_xg = update_match_elo_with_xg(
        **{
            **BASE_ARGS,
            "xg_config": XGBlendConfig(0.25, 1.0),
            "xg_home": None,
            "xg_away": None,
        }
    )
    assert missing_xg.power_delta == pytest.approx(with_xg_disabled.power_delta)
    assert missing_xg.xg_available is False


def test_xg_inputs_must_be_provided_as_a_pair() -> None:
    with pytest.raises(ValueError, match="provided together"):
        update_match_elo_with_xg(
            **{
                **BASE_ARGS,
                "xg_away": None,
            }
        )


def test_blended_residual_matches_contract() -> None:
    update = update_match_elo_with_xg(**BASE_ARGS)
    assert update.xg_residual is not None
    expected = (
        0.90 * update.result_residual + 0.10 * update.xg_residual
    )
    assert update.blended_residual == pytest.approx(expected)
    assert update.power_delta == pytest.approx(
        BASE_ARGS["k_factor"] * expected
    )


def test_additive_luck_correction_preserves_goal_arm() -> None:
    update = update_match_elo_with_xg(
        **{
            **BASE_ARGS,
            "xg_config": XGBlendConfig(
                0.15,
                1.0,
                "ADDITIVE_LUCK_CORRECTION",
            ),
        }
    )
    assert update.xg_home_score is not None
    expected = update.result_residual + 0.15 * (
        update.xg_home_score - update.actual_home_score
    )
    assert update.blended_residual == pytest.approx(expected)
    assert update.power_delta == pytest.approx(
        BASE_ARGS["k_factor"] * expected
    )


def test_additive_luck_correction_reduces_lucky_win_reward() -> None:
    baseline = update_match_elo_with_xg(
        **{
            **BASE_ARGS,
            "home_goals": 1,
            "away_goals": 0,
            "is_neutral": True,
            "home_advantage": 0.0,
            "xg_config": XGBlendConfig(
                0.0,
                1.0,
                "ADDITIVE_LUCK_CORRECTION",
            ),
            "xg_home": 0.1,
            "xg_away": 2.5,
        }
    )
    corrected = update_match_elo_with_xg(
        **{
            **BASE_ARGS,
            "home_goals": 1,
            "away_goals": 0,
            "is_neutral": True,
            "home_advantage": 0.0,
            "xg_config": XGBlendConfig(
                0.15,
                1.0,
                "ADDITIVE_LUCK_CORRECTION",
            ),
            "xg_home": 0.1,
            "xg_away": 2.5,
        }
    )
    assert 0.0 < corrected.power_delta < baseline.power_delta


def test_additive_luck_correction_missing_xg_falls_back_to_gd() -> None:
    baseline = update_match_elo_with_xg(
        **{
            **BASE_ARGS,
            "xg_config": XGBlendConfig(0.0, 1.0),
            "xg_home": None,
            "xg_away": None,
        }
    )
    corrected = update_match_elo_with_xg(
        **{
            **BASE_ARGS,
            "xg_config": XGBlendConfig(
                0.25,
                1.0,
                "ADDITIVE_LUCK_CORRECTION",
            ),
            "xg_home": None,
            "xg_away": None,
        }
    )
    assert corrected.power_delta == pytest.approx(baseline.power_delta)


def test_direction_preserving_correction_does_not_reverse_winner() -> None:
    corrected = update_match_elo_with_xg(
        **{
            **BASE_ARGS,
            "home_rating": 1900.0,
            "away_rating": 1200.0,
            "home_goals": 1,
            "away_goals": 0,
            "xg_config": XGBlendConfig(
                0.50,
                0.75,
                "DIRECTION_PRESERVING_LUCK_CORRECTION",
            ),
            "xg_home": 0.1,
            "xg_away": 2.5,
        }
    )
    assert corrected.result_residual > 0.0
    assert corrected.power_delta == pytest.approx(0.0)


def test_direction_preserving_correction_does_not_reverse_loser() -> None:
    corrected = update_match_elo_with_xg(
        **{
            **BASE_ARGS,
            "home_rating": 1200.0,
            "away_rating": 1900.0,
            "home_goals": 0,
            "away_goals": 1,
            "xg_config": XGBlendConfig(
                0.50,
                0.75,
                "DIRECTION_PRESERVING_LUCK_CORRECTION",
            ),
            "xg_home": 2.5,
            "xg_away": 0.1,
        }
    )
    assert corrected.result_residual < 0.0
    assert corrected.power_delta == pytest.approx(0.0)


def test_direction_preserving_correction_keeps_unreversed_update() -> None:
    additive = update_match_elo_with_xg(
        **{
            **BASE_ARGS,
            "home_goals": 1,
            "away_goals": 0,
            "xg_config": XGBlendConfig(
                0.15,
                1.0,
                "ADDITIVE_LUCK_CORRECTION",
            ),
            "xg_home": 0.5,
            "xg_away": 1.5,
        }
    )
    bounded = update_match_elo_with_xg(
        **{
            **BASE_ARGS,
            "home_goals": 1,
            "away_goals": 0,
            "xg_config": XGBlendConfig(
                0.15,
                1.0,
                "DIRECTION_PRESERVING_LUCK_CORRECTION",
            ),
            "xg_home": 0.5,
            "xg_away": 1.5,
        }
    )
    assert additive.power_delta > 0.0
    assert bounded.power_delta == pytest.approx(additive.power_delta)


def test_xg_update_preserves_total_elo() -> None:
    update = update_match_elo_with_xg(**BASE_ARGS)
    assert update.home_rating_post + update.away_rating_post == pytest.approx(
        BASE_ARGS["home_rating"] + BASE_ARGS["away_rating"]
    )
    assert update.zero_sum_error <= 1e-9


def test_unsupported_margin_does_not_penalize_supported_win() -> None:
    baseline = update_match_elo_with_xg(
        **{
            **BASE_ARGS,
            "home_goals": 1,
            "away_goals": 0,
            "xg_config": XGBlendConfig(0.0, 1.0),
            "xg_home": 2.0,
            "xg_away": 0.5,
        }
    )
    corrected = update_match_elo_with_xg(
        **{
            **BASE_ARGS,
            "home_goals": 1,
            "away_goals": 0,
            "xg_config": XGBlendConfig(0.0, 1.0),
            "xg_home": 2.0,
            "xg_away": 0.5,
            "unsupported_margin_config": UnsupportedMarginConfig(
                0.75,
                0.10,
                0.75,
            ),
        }
    )
    assert corrected.unsupported_margin == pytest.approx(0.0)
    assert corrected.xg_penalty_multiplier == pytest.approx(1.0)
    assert corrected.power_delta == pytest.approx(baseline.power_delta)


def test_unsupported_margin_reduces_lucky_win_without_reversing_it() -> None:
    corrected = update_match_elo_with_xg(
        **{
            **BASE_ARGS,
            "home_goals": 1,
            "away_goals": 0,
            "is_neutral": True,
            "home_advantage": 0.0,
            "xg_config": XGBlendConfig(0.0, 1.0),
            "xg_home": 0.2,
            "xg_away": 2.0,
            "unsupported_margin_config": UnsupportedMarginConfig(
                0.75,
                0.10,
                0.75,
            ),
        }
    )
    assert corrected.winner_xg_advantage == pytest.approx(-1.8)
    assert corrected.unsupported_margin == pytest.approx(2.05)
    assert corrected.xg_penalty_multiplier == pytest.approx(
        1.0 - 0.10 * math.log1p(2.05)
    )
    assert 0.0 < corrected.blended_residual < corrected.result_residual


def test_goal_bonus_guard_preserves_base_result_and_only_reduces_bonus() -> None:
    corrected = update_match_elo_with_xg(
        **{
            **BASE_ARGS,
            "home_goals": 2,
            "away_goals": 0,
            "is_neutral": True,
            "home_advantage": 0.0,
            "xg_config": XGBlendConfig(0.0, 1.0),
            "xg_home": 0.2,
            "xg_away": 2.0,
            "unsupported_margin_config": UnsupportedMarginConfig(
                0.75,
                0.20,
                0.0,
                "GOAL_BONUS_ONLY",
            ),
        }
    )
    assert corrected.goal_bonus_residual > 0.0
    assert corrected.xg_penalty_multiplier < 1.0
    assert corrected.blended_residual == pytest.approx(
        corrected.base_result_residual
        + corrected.goal_bonus_residual * corrected.xg_penalty_multiplier
    )
    assert corrected.blended_residual >= corrected.base_result_residual
    assert corrected.blended_residual < corrected.result_residual


def test_goal_bonus_guard_does_not_change_single_goal_win() -> None:
    baseline = update_match_elo_with_xg(
        **{
            **BASE_ARGS,
            "home_goals": 1,
            "away_goals": 0,
            "xg_config": XGBlendConfig(0.0, 1.0),
            "xg_home": 0.2,
            "xg_away": 2.0,
        }
    )
    guarded = update_match_elo_with_xg(
        **{
            **BASE_ARGS,
            "home_goals": 1,
            "away_goals": 0,
            "xg_config": XGBlendConfig(0.0, 1.0),
            "xg_home": 0.2,
            "xg_away": 2.0,
            "unsupported_margin_config": UnsupportedMarginConfig(
                0.75,
                0.20,
                0.0,
                "GOAL_BONUS_ONLY",
            ),
        }
    )
    assert guarded.goal_bonus_residual == pytest.approx(0.0)
    assert guarded.power_delta == pytest.approx(baseline.power_delta)


def test_xg_performance_bonus_rewards_supported_winner() -> None:
    baseline = update_match_elo_with_xg(
        **{
            **BASE_ARGS,
            "home_goals": 3,
            "away_goals": 2,
            "is_neutral": True,
            "home_advantage": 0.0,
            "xg_config": XGBlendConfig(0.0, 1.0),
            "xg_home": 5.5,
            "xg_away": 1.2,
        }
    )
    rewarded = update_match_elo_with_xg(
        **{
            **BASE_ARGS,
            "home_goals": 3,
            "away_goals": 2,
            "is_neutral": True,
            "home_advantage": 0.0,
            "xg_config": XGBlendConfig(0.0, 1.0),
            "xg_home": 5.5,
            "xg_away": 1.2,
            "xg_performance_bonus_config": XGPerformanceBonusConfig(
                0.20,
                2.0,
                0.25,
            ),
        }
    )
    assert rewarded.xg_performance_signal is not None
    assert rewarded.xg_performance_signal > 0.0
    assert rewarded.xg_performance_adjustment > 0.0
    assert rewarded.power_delta > baseline.power_delta > 0.0


def test_xg_performance_bonus_reduces_lucky_winner_but_keeps_gain_positive() -> None:
    baseline = update_match_elo_with_xg(
        **{
            **BASE_ARGS,
            "home_goals": 3,
            "away_goals": 2,
            "is_neutral": True,
            "home_advantage": 0.0,
            "xg_config": XGBlendConfig(0.0, 1.0),
            "xg_home": 1.0,
            "xg_away": 4.5,
        }
    )
    corrected = update_match_elo_with_xg(
        **{
            **BASE_ARGS,
            "home_goals": 3,
            "away_goals": 2,
            "is_neutral": True,
            "home_advantage": 0.0,
            "xg_config": XGBlendConfig(0.0, 1.0),
            "xg_home": 1.0,
            "xg_away": 4.5,
            "xg_performance_bonus_config": XGPerformanceBonusConfig(
                0.20,
                2.0,
                0.25,
            ),
        }
    )
    assert corrected.xg_performance_signal is not None
    assert corrected.xg_performance_signal < 0.0
    assert corrected.xg_performance_adjustment < 0.0
    assert 0.0 < corrected.power_delta < baseline.power_delta


def test_xg_performance_bonus_is_symmetric_for_away_winner() -> None:
    home_signal, home_adjustment = calculate_xg_performance_adjustment(
        5.5,
        1.2,
        0.5,
        config=XGPerformanceBonusConfig(0.20, 2.0, 0.25),
    )
    away_signal, away_adjustment = calculate_xg_performance_adjustment(
        1.2,
        5.5,
        -0.5,
        config=XGPerformanceBonusConfig(0.20, 2.0, 0.25),
    )
    assert away_signal == pytest.approx(-home_signal)
    assert away_adjustment == pytest.approx(-home_adjustment)


def test_xg_performance_bonus_floor_keeps_extreme_lucky_winner_positive() -> None:
    corrected = update_match_elo_with_xg(
        **{
            **BASE_ARGS,
            "home_goals": 1,
            "away_goals": 0,
            "is_neutral": True,
            "home_advantage": 0.0,
            "xg_config": XGBlendConfig(0.0, 1.0),
            "xg_home": 0.0,
            "xg_away": 10.0,
            "xg_performance_bonus_config": XGPerformanceBonusConfig(
                1.0,
                0.5,
                0.25,
            ),
        }
    )
    assert corrected.direction_floor_residual == pytest.approx(
        0.25 * corrected.base_result_residual
    )
    assert corrected.blended_residual == pytest.approx(
        corrected.direction_floor_residual
    )
    assert corrected.power_delta > 0.0


def test_xg_performance_bonus_does_not_change_draw_or_missing_xg() -> None:
    config = XGPerformanceBonusConfig(0.30, 1.0, 0.25)
    draw = update_match_elo_with_xg(
        **{
            **BASE_ARGS,
            "home_goals": 1,
            "away_goals": 1,
            "xg_config": XGBlendConfig(0.0, 1.0),
            "xg_performance_bonus_config": config,
        }
    )
    missing = update_match_elo_with_xg(
        **{
            **BASE_ARGS,
            "xg_config": XGBlendConfig(0.0, 1.0),
            "xg_home": None,
            "xg_away": None,
            "xg_performance_bonus_config": config,
        }
    )
    baseline_missing = update_match_elo_with_xg(
        **{
            **BASE_ARGS,
            "xg_config": XGBlendConfig(0.0, 1.0),
            "xg_home": None,
            "xg_away": None,
        }
    )
    assert draw.xg_performance_adjustment == pytest.approx(0.0)
    assert missing.power_delta == pytest.approx(baseline_missing.power_delta)


def test_xg_performance_bonus_rejects_other_xg_modes() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        update_match_elo_with_xg(
            **{
                **BASE_ARGS,
                "xg_config": XGBlendConfig(0.05, 1.0),
                "xg_performance_bonus_config": XGPerformanceBonusConfig(
                    0.20,
                    2.0,
                    0.25,
                ),
            }
        )


@pytest.mark.parametrize(
    "config",
    [
        XGPerformanceBonusConfig(-0.1, 1.0, 0.25),
        XGPerformanceBonusConfig(float("inf"), 1.0, 0.25),
        XGPerformanceBonusConfig(0.2, 0.0, 0.25),
        XGPerformanceBonusConfig(0.2, 1.0, 0.0),
        XGPerformanceBonusConfig(0.2, 1.0, 1.1),
    ],
)
def test_xg_performance_bonus_config_validation(
    config: XGPerformanceBonusConfig,
) -> None:
    with pytest.raises(ValueError):
        config.validate()


def test_unsupported_margin_is_symmetric_for_away_winner() -> None:
    home_lucky = calculate_unsupported_margin_penalty(
        1,
        0,
        0.2,
        2.0,
        goal_difference_cap=4,
        config=UnsupportedMarginConfig(0.75, 0.10, 0.75),
    )
    away_lucky = calculate_unsupported_margin_penalty(
        0,
        1,
        2.0,
        0.2,
        goal_difference_cap=4,
        config=UnsupportedMarginConfig(0.75, 0.10, 0.75),
    )
    assert away_lucky == home_lucky


def test_unsupported_margin_respects_goal_cap_and_minimum_multiplier() -> None:
    penalty = calculate_unsupported_margin_penalty(
        7,
        0,
        0.1,
        3.5,
        goal_difference_cap=4,
        config=UnsupportedMarginConfig(0.0, 1.0, 0.80),
    )
    assert penalty.actual_goal_margin == 4
    assert penalty.unsupported_margin == pytest.approx(7.4)
    assert penalty.penalty_multiplier == pytest.approx(0.80)


@pytest.mark.parametrize("penalties", [False, True])
def test_unsupported_margin_does_not_change_draws(penalties: bool) -> None:
    penalty = calculate_unsupported_margin_penalty(
        1,
        1,
        2.4,
        0.2,
        goal_difference_cap=4,
        config=UnsupportedMarginConfig(0.75, 0.10, 0.75),
        decided_on_penalties=penalties,
    )
    assert penalty.unsupported_margin == pytest.approx(0.0)
    assert penalty.penalty_multiplier == pytest.approx(1.0)


def test_unsupported_margin_missing_xg_falls_back_to_goal_model() -> None:
    baseline = update_match_elo_with_xg(
        **{
            **BASE_ARGS,
            "xg_config": XGBlendConfig(0.0, 1.0),
            "xg_home": None,
            "xg_away": None,
        }
    )
    corrected = update_match_elo_with_xg(
        **{
            **BASE_ARGS,
            "xg_config": XGBlendConfig(0.0, 1.0),
            "xg_home": None,
            "xg_away": None,
            "unsupported_margin_config": UnsupportedMarginConfig(
                0.75,
                0.10,
                0.75,
            ),
        }
    )
    assert corrected.power_delta == pytest.approx(baseline.power_delta)
    assert corrected.xg_penalty_multiplier == pytest.approx(1.0)


def test_unsupported_margin_cannot_be_mixed_with_legacy_xg_blend() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        update_match_elo_with_xg(
            **{
                **BASE_ARGS,
                "xg_config": XGBlendConfig(0.05, 1.0),
                "unsupported_margin_config": UnsupportedMarginConfig(
                    0.75,
                    0.10,
                    0.75,
                ),
            }
        )


@pytest.mark.parametrize(
    "config",
    [
        UnsupportedMarginConfig(-0.1, 0.1, 0.75),
        UnsupportedMarginConfig(0.75, -0.1, 0.75),
        UnsupportedMarginConfig(0.75, 0.1, -0.1),
        UnsupportedMarginConfig(0.75, 0.1, 1.1),
        UnsupportedMarginConfig(0.75, 0.1, 0.75, "UNKNOWN"),
    ],
)
def test_unsupported_margin_config_rejects_invalid_values(
    config: UnsupportedMarginConfig,
) -> None:
    with pytest.raises(ValueError):
        config.validate()


def test_goal_difference_can_be_disabled_independently() -> None:
    update = update_match_elo_with_xg(
        **{
            **BASE_ARGS,
            "goal_difference_enabled": False,
            "goal_alpha": 0.0,
        }
    )
    assert update.goal_difference_multiplier == 1.0
    assert update.xg_available is True


def test_penalty_field_score_preserves_the_90_or_120_minute_result() -> None:
    assert elo_field_score(2, 1, True) == (2, 1)
    assert elo_field_score(2, 1, False) == (2, 1)


def test_penalty_decider_uses_field_result_but_disables_gd_and_xg_bonus() -> None:
    update = update_match_elo_with_xg(
        **{
            **BASE_ARGS,
            "home_goals": 2,
            "away_goals": 0,
            "decided_on_penalties": True,
            "xg_home": 3.0,
            "xg_away": 0.5,
            "xg_config": XGBlendConfig(0.0, 1.0),
            "xg_performance_bonus_config": XGPerformanceBonusConfig(
                0.30,
                1.25,
                0.70,
            ),
        }
    )
    assert update.actual_home_score == pytest.approx(1.0)
    assert update.goal_difference == 2
    assert update.goal_difference_multiplier == pytest.approx(1.0)
    assert update.xg_performance_adjustment == pytest.approx(0.0)
    assert update.power_delta == pytest.approx(
        BASE_ARGS["k_factor"] * (1.0 - update.expected_home_score)
    )


def test_ablation_grid_has_four_arms_and_42_candidates() -> None:
    candidates = candidate_grid()
    assert len(candidates) == 42
    assert {candidate.model for candidate in candidates} == {
        "BASE",
        "GD",
        "XG",
        "GD_XG",
    }
    assert sum(candidate.model == "XG" for candidate in candidates) == 20
    assert sum(candidate.model == "GD_XG" for candidate in candidates) == 20
