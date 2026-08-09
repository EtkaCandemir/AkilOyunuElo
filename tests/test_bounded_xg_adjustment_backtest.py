from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ao_elo.xg_live import XGBlendConfig, update_match_elo_with_xg
from scripts.run_bounded_xg_adjustment_walk_forward_2025_26 import (
    BASELINE_KEY,
    bounded_candidate_grid,
    select_brier_first_candidate,
)


BASE_ARGS = {
    "home_rating": 1500.0,
    "away_rating": 1500.0,
    "home_goals": 1,
    "away_goals": 0,
    "k_factor": 103.98098633392752,
    "elo_scale": 835.5614973262034,
    "home_advantage": 148.54426619132505,
    "is_neutral": True,
    "decided_on_penalties": False,
    "goal_difference_enabled": True,
    "goal_alpha": 0.10,
    "goal_tau": 300.0,
    "goal_difference_cap": 4,
    "xg_config": XGBlendConfig(0.0, 1.0),
}


def test_bounded_grid_has_28_candidates_and_production_control() -> None:
    candidates = bounded_candidate_grid()
    assert len(candidates) == 29
    assert candidates[0].key == BASELINE_KEY
    bounded = [candidate for candidate in candidates if not candidate.is_baseline]
    assert {candidate.max_xg_ratio for candidate in bounded} == {
        0.15,
        0.20,
        0.25,
        0.30,
    }
    assert {candidate.xg_scale for candidate in bounded} == {
        0.75,
        1.0,
        1.25,
        1.5,
        2.0,
        2.5,
        3.0,
    }
    assert all(
        candidate.minimum_winner_gain_ratio == 1.0 - candidate.max_xg_ratio
        for candidate in bounded
    )


def test_max_ratio_bounds_extreme_lucky_and_supported_winner() -> None:
    candidate = next(
        value
        for value in bounded_candidate_grid()
        if value.max_xg_ratio == 0.30 and value.xg_scale == 2.0
    )
    classic = update_match_elo_with_xg(
        **BASE_ARGS,
        xg_home=None,
        xg_away=None,
    )
    lucky = update_match_elo_with_xg(
        **BASE_ARGS,
        xg_home=0.0,
        xg_away=100.0,
        xg_performance_bonus_config=candidate.config,
    )
    supported = update_match_elo_with_xg(
        **BASE_ARGS,
        xg_home=100.0,
        xg_away=0.0,
        xg_performance_bonus_config=candidate.config,
    )
    assert lucky.power_delta >= 0.70 * classic.base_result_residual * BASE_ARGS["k_factor"]
    assert supported.power_delta <= 1.30 * classic.base_result_residual * BASE_ARGS["k_factor"]
    assert lucky.power_delta > 0.0


def test_brier_first_selection_keeps_ranking_guardrail() -> None:
    candidates = bounded_candidate_grid()
    safe = candidates[1]
    unsafe = candidates[-1]
    metrics = pd.DataFrame(
        [
            _row(BASELINE_KEY, 0.0, 1.0, 0.60, 1.00, 0.70, 0.75),
            _row(safe.key, safe.max_xg_ratio, safe.xg_scale, 0.59, 1.01, 0.71, 0.76),
            _row(unsafe.key, unsafe.max_xg_ratio, unsafe.xg_scale, 0.50, 0.80, 0.69, 0.74),
        ]
    )
    selected, audit = select_brier_first_candidate(metrics, candidates)
    assert selected.key == safe.key
    assert audit["selection_pool"] == "RANKING_SAFE"
    assert audit["future_results_used_for_selection"] is False


def _row(
    key: str,
    ratio: float,
    scale: float,
    brier: float,
    log_loss: float,
    ranking: float,
    pairwise: float,
) -> dict[str, float | str]:
    return {
        "candidate_key": key,
        "max_xg_ratio": ratio,
        "xg_scale": scale,
        "brier_1x2": brier,
        "log_loss_1x2": log_loss,
        "ranking_score": ranking,
        "pairwise_accuracy": pairwise,
        "brier_1x2_delta_vs_production": brier - 0.60,
        "log_loss_1x2_delta_vs_production": log_loss - 1.00,
        "ranking_score_delta_vs_production": ranking - 0.70,
        "pairwise_accuracy_delta_vs_production": pairwise - 0.75,
    }
