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

from scripts.run_goal_alpha_with_bounded_xg_backtest_2025_26 import (
    BASELINE_KEY,
    GOAL_ALPHA_GRID,
    goal_alpha_grid,
    production_with_goal_alpha,
    select_brier_first_candidate,
)


def test_goal_alpha_grid_matches_pre_registered_values() -> None:
    candidates = goal_alpha_grid()
    assert len(candidates) == 7
    assert tuple(candidate.goal_alpha for candidate in candidates) == GOAL_ALPHA_GRID
    assert candidates[0].key == BASELINE_KEY
    assert all(candidate.beta == 0.30 for candidate in candidates)
    assert all(candidate.xg_scale == 1.25 for candidate in candidates)
    assert all(candidate.minimum_winner_gain_ratio == 0.70 for candidate in candidates)


def test_goal_alpha_override_does_not_add_a_match_delta_cap() -> None:
    production = {
        "goal_margin": {
            "alpha": 0.10,
            "tau": 300.0,
            "goal_difference_cap": 4,
        },
        "untouched": {"value": 1},
    }
    updated = production_with_goal_alpha(production, 0.20)
    assert updated["goal_margin"] == {
        "alpha": 0.20,
        "tau": 300.0,
        "goal_difference_cap": 4,
    }
    assert "max_match_delta" not in updated
    assert production["goal_margin"]["alpha"] == 0.10


def test_brier_first_selection_respects_ranking_guardrail() -> None:
    candidates = goal_alpha_grid()
    safe = candidates[2]
    unsafe = candidates[-1]
    metrics = pd.DataFrame(
        [
            _row(BASELINE_KEY, 0.10, 0.60, 1.00, 0.70, 0.75),
            _row(safe.key, safe.goal_alpha, 0.59, 1.01, 0.71, 0.76),
            _row(unsafe.key, unsafe.goal_alpha, 0.50, 0.80, 0.69, 0.74),
        ]
    )
    selected, audit = select_brier_first_candidate(metrics, candidates)
    assert selected.key == safe.key
    assert audit["selection_pool"] == "RANKING_SAFE"
    assert audit["future_results_used_for_selection"] is False


def _row(
    key: str,
    alpha: float,
    brier: float,
    log_loss: float,
    ranking: float,
    pairwise: float,
) -> dict[str, float | str]:
    return {
        "candidate_key": key,
        "goal_alpha": alpha,
        "brier_1x2": brier,
        "log_loss_1x2": log_loss,
        "ranking_score": ranking,
        "pairwise_accuracy": pairwise,
        "brier_1x2_delta_vs_baseline": brier - 0.60,
        "log_loss_1x2_delta_vs_baseline": log_loss - 1.00,
        "ranking_score_delta_vs_baseline": ranking - 0.70,
        "pairwise_accuracy_delta_vs_baseline": pairwise - 0.75,
    }
