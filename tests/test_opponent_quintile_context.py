from __future__ import annotations

import math

import pandas as pd
import pytest

from ao_elo.opponent_quintile_context import (
    CATEGORICAL_1X2_OUTCOME,
    DYNAMIC_QUINTILES,
    FIXED_THRESHOLDS,
    SEASON_LOCKED,
    OpponentQuintileContextConfig,
    assign_fixed_threshold_membership,
    assign_quintiles,
    contextual_matchup_expectation,
    estimate_categorical_1x2_elo_offset,
    estimate_quintile_profile,
)


def test_assign_quintiles_has_balanced_deterministic_bands() -> None:
    snapshot = assign_quintiles(
        list(range(1, 12)),
        [f"AO-{index:02d}" for index in range(1, 12)],
        [1000.0] * 11,
    )
    assignments = snapshot.assignments
    counts = assignments["quintile"].value_counts().sort_index()
    assert counts.max() - counts.min() <= 1
    assert assignments.loc[0, "club_id"] == "AO-01"
    assert assignments.loc[0, "quintile"] == 1
    assert assignments.loc[len(assignments) - 1, "club_id"] == "AO-11"
    assert assignments.loc[len(assignments) - 1, "quintile"] == 5
    assert snapshot.boundary_ties == (True, True, True, True)


def test_fixed_threshold_membership_uses_current_rating() -> None:
    membership = assign_fixed_threshold_membership(
        [1, 2, 3],
        ["A", "B", "C"],
        [110.0, 201.0, 500.0],
        [100.0, 200.0, 300.0, 400.0],
    ).set_index("club_id")
    assert membership.loc["A", "quintile"] == 2
    assert membership.loc["B", "quintile"] == 3
    assert membership.loc["C", "quintile"] == 5


def test_assign_quintiles_supports_three_dynamic_strength_bands() -> None:
    snapshot = assign_quintiles(
        list(range(1, 11)),
        [f"AO-{index:02d}" for index in range(1, 11)],
        list(range(1000, 1100, 10)),
        band_count=3,
    )

    counts = snapshot.assignments["quintile"].value_counts().sort_index()
    assert counts.to_dict() == {1: 4, 2: 3, 3: 3}
    assert len(snapshot.thresholds) == 2
    assert len(snapshot.boundary_ties) == 2


def test_quintile_profile_is_centered_and_shrunk() -> None:
    history = pd.DataFrame(
        {
            "opponent_quintile": [1, 1, 1, 5, 5, 5],
            "expected_score": [0.5] * 6,
            "actual_score": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            "weight": [1.0] * 6,
        }
    )
    profile = estimate_quintile_profile(
        "AO-1",
        history,
        elo_scale=800.0,
        shrinkage_matches=3.0,
        effect_cap=50.0,
    )
    assert profile.effect_for(1) > 0.0
    assert profile.effect_for(5) < 0.0
    weighted = sum(
        effect * weight
        for effect, weight in zip(profile.effects, profile.effective_matches, strict=True)
    ) / sum(profile.effective_matches)
    assert weighted == pytest.approx(0.0, abs=1e-9)
    assert all(0.0 <= value <= 1.0 for value in profile.reliabilities)
    assert all(abs(value) <= 50.0 for value in profile.effects)


def test_single_observed_band_has_no_specific_effect() -> None:
    history = pd.DataFrame(
        {
            "opponent_quintile": [3, 3],
            "expected_score": [0.5, 0.5],
            "actual_score": [1.0, 0.0],
            "weight": [1.0, 1.0],
        }
    )
    profile = estimate_quintile_profile(
        "AO-1",
        history,
        elo_scale=800.0,
        shrinkage_matches=6.0,
        effect_cap=75.0,
    )
    assert profile.effects == (0.0, 0.0, 0.0, 0.0, 0.0)


def test_three_band_profile_remains_centered() -> None:
    history = pd.DataFrame(
        {
            "opponent_quintile": [1, 1, 2, 2, 3, 3],
            "expected_score": [0.5] * 6,
            "actual_score": [1.0, 1.0, 0.5, 0.5, 0.0, 0.0],
            "weight": [1.0] * 6,
        }
    )
    profile = estimate_quintile_profile(
        "AO-1",
        history,
        elo_scale=800.0,
        shrinkage_matches=3.0,
        effect_cap=50.0,
        band_count=3,
    )

    assert len(profile.effects) == 3
    assert profile.effect_for(1) > 0.0
    assert profile.effect_for(3) < 0.0
    assert sum(
        effect * weight
        for effect, weight in zip(profile.effects, profile.effective_matches, strict=True)
    ) == pytest.approx(0.0, abs=1e-9)


def test_categorical_draw_moves_underdog_and_favorite_toward_even() -> None:
    underdog = estimate_categorical_1x2_elo_offset(
        [0.20] * 8,
        [0.5] * 8,
        [1.0] * 8,
        elo_scale=835.561497,
        draw_at_even=0.24,
        draw_shape=1.0,
    )
    favorite = estimate_categorical_1x2_elo_offset(
        [0.80] * 8,
        [0.5] * 8,
        [1.0] * 8,
        elo_scale=835.561497,
        draw_at_even=0.24,
        draw_shape=1.0,
    )
    even = estimate_categorical_1x2_elo_offset(
        [0.50] * 8,
        [0.5] * 8,
        [1.0] * 8,
        elo_scale=835.561497,
        draw_at_even=0.24,
        draw_shape=1.0,
    )

    assert underdog.offset > 0.0
    assert favorite.offset == pytest.approx(-underdog.offset)
    assert even.offset == pytest.approx(0.0)


def test_categorical_three_band_profile_is_centered() -> None:
    history = pd.DataFrame(
        {
            "opponent_quintile": [1, 1, 2, 2, 3, 3],
            "expected_score": [0.65, 0.60, 0.50, 0.50, 0.30, 0.25],
            "actual_score": [1.0, 0.5, 0.5, 0.5, 0.5, 0.0],
            "weight": [1.0] * 6,
        }
    )
    profile = estimate_quintile_profile(
        "AO-1",
        history,
        elo_scale=835.561497,
        shrinkage_matches=3.0,
        effect_cap=50.0,
        band_count=3,
        outcome_model=CATEGORICAL_1X2_OUTCOME,
        draw_at_even=0.24,
        draw_shape=1.0,
    )

    assert len(profile.effects) == 3
    assert all(math.isfinite(value) for value in profile.effects)
    assert sum(
        effect * weight
        for effect, weight in zip(profile.effects, profile.effective_matches, strict=True)
    ) == pytest.approx(0.0, abs=1e-9)


def test_contextual_matchup_expectation_is_bounded_and_directional() -> None:
    context = contextual_matchup_expectation(
        0.5,
        100.0,
        -100.0,
        elo_scale=800.0,
        context_cap=50.0,
    )
    assert context.applied_offset == 50.0
    assert context.expected_home_score > 0.5
    neutral = contextual_matchup_expectation(
        0.5,
        0.0,
        0.0,
        elo_scale=800.0,
        context_cap=0.0,
    )
    assert neutral.expected_home_score == pytest.approx(0.5)


def test_config_rejects_unknown_mode() -> None:
    config = OpponentQuintileContextConfig(
        "UNKNOWN",
        SEASON_LOCKED,
        3,
        0.75,
        6.0,
        25.0,
    )
    with pytest.raises(ValueError, match="Unknown quintile band mode"):
        config.validate()
    valid = OpponentQuintileContextConfig(
        DYNAMIC_QUINTILES,
        SEASON_LOCKED,
        3,
        0.75,
        6.0,
        25.0,
    )
    valid.validate()
    assert "dynamic_locked" in valid.key
