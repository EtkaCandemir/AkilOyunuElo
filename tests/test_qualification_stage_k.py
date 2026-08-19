from __future__ import annotations

import math

import pytest

from ao_elo.qualification_stage_k import (
    CURRENT_K_REFERENCE,
    QualificationStageKConfig,
    all_configs,
    candidate_configs,
    effective_match_k,
    qualification_round_key,
    reference_config,
    stage_k_multiplier,
)
from ao_elo.xg_live import XGBlendConfig, XGPerformanceBonusConfig, update_match_elo_with_xg


BASE_K = 103.98098633392752


@pytest.mark.parametrize(
    ("round_name", "expected"),
    [
        ("Preliminary Round", "Q1"),
        ("1st Qualifying Round", "Q1"),
        ("2nd Qualifying Round", "Q2"),
        ("3rd Qualifying Round", "Q3"),
        ("Qualifying Play-off Round", "QUALIFYING_PLAYOFF"),
        ("League Phase", "MAIN"),
        ("Knockout round play-offs", "MAIN"),
        ("Round of 16", "MAIN"),
        ("Final", "MAIN"),
    ],
)
def test_round_mapping_keeps_qualifying_and_knockout_playoffs_distinct(
    round_name: str, expected: str
) -> None:
    assert qualification_round_key(round_name) == expected


def test_candidate_grid_excludes_all_one_reference_from_selection() -> None:
    candidates = candidate_configs()
    configs = all_configs()
    assert len(candidates) == 7
    assert len(configs) == 8
    assert configs[0].profile == CURRENT_K_REFERENCE
    assert configs[0].selectable is False
    assert all(config.selectable for config in candidates)
    assert all(config.qualifying_playoff_multiplier < 1.0 for config in candidates)


def test_every_candidate_is_strictly_increasing_and_main_is_full_k() -> None:
    for config in candidate_configs():
        values = (
            config.q1_multiplier,
            config.q2_multiplier,
            config.q3_multiplier,
            config.qualifying_playoff_multiplier,
            config.main_multiplier,
        )
        assert all(left < right for left, right in zip(values, values[1:]))
        assert stage_k_multiplier("League Phase", config) == pytest.approx(1.0)


def test_preliminary_uses_q1_and_effective_k_scales_base() -> None:
    config = candidate_configs()[3]
    assert stage_k_multiplier("Preliminary Round", config) == pytest.approx(0.45)
    assert effective_match_k(BASE_K, "1st Qualifying Round", config) == pytest.approx(
        BASE_K * 0.45
    )
    assert effective_match_k(BASE_K, "Knockout round play-offs", config) == pytest.approx(
        BASE_K
    )


@pytest.mark.parametrize(
    "config",
    [
        QualificationStageKConfig("bad", 0.4, 0.4, 0.8, 0.9),
        QualificationStageKConfig("bad", 0.4, 0.6, 0.5, 0.9),
        QualificationStageKConfig("bad", 0.4, 0.6, 0.8, 1.0),
        QualificationStageKConfig("bad", 0.4, 0.6, 0.8, 0.9, 0.9),
        QualificationStageKConfig("bad", math.nan, 0.6, 0.8, 0.9),
    ],
)
def test_invalid_candidate_contracts_are_rejected(config: QualificationStageKConfig) -> None:
    with pytest.raises(ValueError):
        config.validate()


def test_full_xg_goal_update_scales_with_stage_k_and_stays_zero_sum() -> None:
    config = candidate_configs()[3]
    kwargs = dict(
        home_rating=1500.0,
        away_rating=1450.0,
        home_goals=3,
        away_goals=0,
        elo_scale=835.5614973262034,
        home_advantage=148.54426619132505,
        is_neutral=False,
        decided_on_penalties=False,
        goal_difference_enabled=True,
        goal_alpha=0.15,
        goal_tau=300.0,
        goal_difference_cap=4,
        xg_config=XGBlendConfig(0.0, 1.0),
        xg_home=2.4,
        xg_away=0.5,
        xg_performance_bonus_config=XGPerformanceBonusConfig(0.30, 1.25, 0.70),
    )
    full = update_match_elo_with_xg(k_factor=BASE_K, **kwargs)
    q1 = update_match_elo_with_xg(
        k_factor=effective_match_k(BASE_K, "1st Qualifying Round", config),
        **kwargs,
    )
    assert q1.power_delta == pytest.approx(full.power_delta * 0.45)
    assert q1.goal_difference_multiplier == pytest.approx(full.goal_difference_multiplier)
    assert q1.xg_performance_signal == pytest.approx(full.xg_performance_signal)
    assert q1.zero_sum_error <= 1e-9


def test_reference_reproduces_full_k_for_every_round() -> None:
    config = reference_config()
    for round_name in (
        "1st Qualifying Round",
        "2nd Qualifying Round",
        "3rd Qualifying Round",
        "Qualifying Play-off Round",
        "League Phase",
    ):
        assert effective_match_k(BASE_K, round_name, config) == pytest.approx(BASE_K)
