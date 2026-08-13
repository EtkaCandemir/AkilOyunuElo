from __future__ import annotations

import json
from pathlib import Path

import pytest

from ao_elo.final_candidate import (
    load_final_candidate_runtime,
    update_final_candidate_match,
)


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "contracts" / "ao_european_elo_v2_final_candidate.json"
PRODUCTION = ROOT / "contracts" / "ao_european_elo_v2_production.json"


def test_final_candidate_contract_loads_selected_parameters() -> None:
    runtime = load_final_candidate_runtime(CANDIDATE)
    assert runtime.candidate_version.endswith("2026-08-13")
    assert len(runtime.contract_sha256) == 64
    assert runtime.dynamic_config.goal_alpha == pytest.approx(0.15)
    assert runtime.dynamic_config.goal_tau == pytest.approx(300.0)
    assert runtime.dynamic_config.goal_difference_cap == 4
    assert runtime.dynamic_config.progression_bonus_enabled is True
    assert runtime.dynamic_config.progression_base_bonus == pytest.approx(12.0)
    assert runtime.dynamic_config.progression_stages_per_competition == 4
    assert runtime.xg_config.beta == pytest.approx(0.30)
    assert runtime.xg_config.xg_scale == pytest.approx(1.25)
    assert runtime.xg_config.minimum_winner_gain_ratio == pytest.approx(0.70)


def test_final_candidate_supported_win_combines_goal_and_xg() -> None:
    runtime = load_final_candidate_runtime(CANDIDATE)
    update = update_final_candidate_match(
        runtime,
        1500.0,
        1500.0,
        2,
        0,
        is_neutral=True,
        decided_on_penalties=False,
        xg_home=3.5,
        xg_away=0.5,
    )
    assert update.power_delta == pytest.approx(72.738575, abs=1e-5)
    assert update.xg_performance_adjustment > 0.0
    assert update.goal_difference_multiplier > 1.0
    assert update.zero_sum_error <= 1e-9


def test_final_candidate_missing_xg_uses_goal_margin_only() -> None:
    runtime = load_final_candidate_runtime(CANDIDATE)
    update = update_final_candidate_match(
        runtime,
        1500.0,
        1500.0,
        3,
        0,
        is_neutral=True,
        decided_on_penalties=False,
    )
    assert update.power_delta == pytest.approx(60.558102, abs=1e-5)
    assert update.xg_performance_adjustment == 0.0
    assert update.xg_available is False


def test_final_candidate_draw_does_not_apply_xg() -> None:
    runtime = load_final_candidate_runtime(CANDIDATE)
    update = update_final_candidate_match(
        runtime,
        1500.0,
        1500.0,
        2,
        2,
        is_neutral=True,
        decided_on_penalties=False,
        xg_home=4.0,
        xg_away=0.5,
    )
    assert update.power_delta == pytest.approx(0.0)
    assert update.xg_performance_adjustment == 0.0


def test_active_production_contract_keeps_dynamic_core_and_adds_surprise() -> None:
    production = json.loads(PRODUCTION.read_text(encoding="utf-8"))
    assert production["goal_margin"]["alpha"] == pytest.approx(0.15)
    assert production["progression_bonus"]["active"] is True
    assert production["progression_bonus"]["increments"] == {
        "UCL": 12.0,
        "UEL": 8.0,
        "UECL": 4.0,
    }
    assert production["progression_bonus"]["season_caps"] == {
        "UCL": 48.0,
        "UEL": 32.0,
        "UECL": 16.0,
    }
    assert production["xg_performance"]["active"] is True
    assert production["xg_performance"]["max_xg_ratio"] == pytest.approx(0.30)
    assert production["xg_performance"]["xg_scale"] == pytest.approx(1.25)
    assert production["xg_performance"]["minimum_winner_gain_ratio"] == pytest.approx(0.70)
    assert production["domestic_surprise"] == {
        "active": True,
        "family": "FIVE_SEASON_VARIANCE_CONTROLLED_DIRECT_PERCENTILE",
        "coefficient": 0.4,
        "variance_penalty": 0.5,
        "max_abs_adjustment": 30.0,
        "minimum_history_seasons": 5,
        "season_weights_oldest_to_newest": [0.07, 0.13, 0.2, 0.27, 0.33],
        "volatility_normalization": "min(1, 2 * weighted_volatility)",
        "consistency_formula": (
            "1 - variance_penalty * normalized_volatility"
        ),
        "exposure_formula": (
            "AO adjustment = (1 - effective European exposure) * domestic adjustment"
        ),
        "insufficient_history_behavior": "NO_ADJUSTMENT",
    }
    prediction = production["prediction_layer"]
    assert prediction["active"] is True
    assert prediction["decision"] == "PROMOTE_WITH_MONITORING"
    assert prediction["top_level_blend"] == {
        "space": "LOG_PROBABILITY",
        "current_ml_weight": 0.5,
        "ao_domestic_poisson_weight": 0.5,
    }
    assert prediction["ao_domestic_poisson_component"]["transfer_config"][
        "rho"
    ] == pytest.approx(0.0)
    assert prediction["rating_feedback"] is False
    assert prediction["fallback"] == "CURRENT_AO_1X2"
