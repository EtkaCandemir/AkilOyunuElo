from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_domestic_surprise_effect_size_sensitivity import (
    add_baseline_deltas,
    candidate_grid,
    candidate_key,
    nearest_effect_models,
)


def production_contract() -> dict[str, object]:
    return {
        "domestic_surprise": {
            "coefficient": 0.40,
            "variance_penalty": 0.50,
            "max_abs_adjustment": 30.0,
            "minimum_history_seasons": 5,
        }
    }


def test_grid_contains_production_and_high_effect_candidates() -> None:
    grid = candidate_grid(production_contract())
    keys = {candidate_key(config) for config in grid}
    assert "theta_0p4_cap_30" in keys
    assert "theta_0_cap_30" in keys
    assert "theta_4_cap_150" in keys
    assert len(keys) == len(grid)


def test_baseline_deltas_are_zero_for_current() -> None:
    frame = pd.DataFrame(
        [
            {
                "candidate_key": "current",
                "brier_1x2": 0.5,
                "log_loss_1x2": 0.7,
                "accuracy_1x2": 0.5,
                "same_season_spearman": 0.4,
                "same_season_pairwise_accuracy": 0.6,
                "forward_season_spearman": 0.3,
                "forward_season_pairwise_accuracy": 0.55,
            },
            {
                "candidate_key": "candidate",
                "brier_1x2": 0.49,
                "log_loss_1x2": 0.69,
                "accuracy_1x2": 0.51,
                "same_season_spearman": 0.41,
                "same_season_pairwise_accuracy": 0.61,
                "forward_season_spearman": 0.31,
                "forward_season_pairwise_accuracy": 0.56,
            },
        ]
    )
    result = add_baseline_deltas(frame, "current")
    current = result.loc[result["candidate_key"].eq("current")].iloc[0]
    assert current.filter(like="delta_vs_current_").eq(0.0).all()
    candidate = result.loc[result["candidate_key"].eq("candidate")].iloc[0]
    assert candidate["delta_vs_current_brier_1x2"] == pytest.approx(-0.01)


def test_nearest_effect_model_uses_realized_not_theta_distance() -> None:
    frame = pd.DataFrame(
        [
            {"candidate_key": "current", "theta": 0.4, "cap": 30.0, "changed_mean_abs_initial_delta": 7.0},
            {"candidate_key": "near", "theta": 2.0, "cap": 50.0, "changed_mean_abs_initial_delta": 29.8},
            {"candidate_key": "far", "theta": 1.0, "cap": 30.0, "changed_mean_abs_initial_delta": 20.0},
        ]
    )
    decision = nearest_effect_models(frame, "current")
    selected = decision.loc[decision["effect_target"].eq("~30_ELO")].iloc[0]
    assert selected["candidate_key"] == "near"
