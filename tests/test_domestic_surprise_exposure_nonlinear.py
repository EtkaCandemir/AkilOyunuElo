from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_domestic_surprise_exposure_nonlinear import (
    CURRENT_KEY,
    ExposureScalingCandidate,
    candidate_grid,
    classify_result,
    mark_pareto,
)


def production_contract() -> dict[str, object]:
    return {
        "domestic_surprise": {
            "coefficient": 0.40,
            "max_abs_adjustment": 30.0,
        }
    }


def test_candidate_grid_has_required_families_and_controls() -> None:
    candidates = candidate_grid(production_contract())
    keys = {candidate.key for candidate in candidates}
    families = {candidate.family for candidate in candidates}
    assert CURRENT_KEY in keys
    assert "no_surprise" in keys
    assert "global_strong_theta_1p75_cap_150" in keys
    assert {"CONTROL", "PIECEWISE", "LINEAR", "POWER", "LOGISTIC"} == families
    assert len(candidates) >= 35


@pytest.mark.parametrize("family", ["LINEAR", "POWER", "LOGISTIC"])
def test_smooth_candidates_are_monotone(family: str) -> None:
    candidates = [candidate for candidate in candidate_grid(production_contract()) if candidate.family == family]
    candidate = candidates[0]
    exposures = [0.0, 0.25, 0.50, 0.75, 1.0]
    theta = [candidate.theta(value) for value in exposures]
    caps = [candidate.cap(value) for value in exposures]
    assert theta == sorted(theta, reverse=True)
    assert caps == sorted(caps, reverse=True)


def test_piecewise_boundaries_use_expected_bands() -> None:
    candidate = ExposureScalingCandidate(
        "piecewise",
        "PIECEWISE",
        (5.0, 4.0, 3.0, 2.0, 1.0),
        (50.0, 40.0, 30.0, 20.0, 10.0),
        10,
    )
    assert candidate.theta(0.0) == 5.0
    assert candidate.theta(0.25) == 4.0
    assert candidate.theta(0.250001) == 3.0
    assert candidate.theta(1.0) == 1.0


def test_pareto_marks_strictly_dominated_candidate() -> None:
    frame = pd.DataFrame(
        [
            {
                "candidate_key": "better",
                "brier_1x2": 0.50,
                "log_loss_1x2": 0.70,
                "complexity_parameters": 2,
                "same_season_spearman": 0.60,
                "same_season_pairwise_accuracy": 0.70,
                "forward_season_spearman": 0.50,
                "forward_season_pairwise_accuracy": 0.65,
            },
            {
                "candidate_key": "worse",
                "brier_1x2": 0.51,
                "log_loss_1x2": 0.71,
                "complexity_parameters": 3,
                "same_season_spearman": 0.59,
                "same_season_pairwise_accuracy": 0.69,
                "forward_season_spearman": 0.49,
                "forward_season_pairwise_accuracy": 0.64,
            },
        ]
    )
    result = mark_pareto(frame).set_index("candidate_key")
    assert bool(result.loc["better", "is_pareto_frontier"])
    assert bool(result.loc["worse", "pareto_dominated"])


def test_tiny_uncertain_loss_gain_is_not_called_balanced_win() -> None:
    current = pd.Series(
        {
            "brier_1x2": 0.57210,
            "log_loss_1x2": 0.96440,
            "same_season_spearman": 0.6720,
            "same_season_pairwise_accuracy": 0.7535,
            "forward_season_spearman": 0.4684,
            "forward_season_pairwise_accuracy": 0.6581,
        }
    )
    strong = current.copy()
    strong["brier_1x2"] -= 0.00070
    strong["log_loss_1x2"] -= 0.00100
    candidate = current.copy()
    candidate["candidate_key"] = "adaptive"
    candidate["brier_1x2"] -= 0.00002
    candidate["log_loss_1x2"] -= 0.00003
    candidate["same_season_spearman"] += 0.0001
    candidate["same_season_pairwise_accuracy"] += 0.0001
    candidate["forward_season_spearman"] += 0.0001
    candidate["forward_season_pairwise_accuracy"] += 0.0001
    candidate["spearman_non_regressed_folds"] = 4
    candidate["pairwise_non_regressed_folds"] = 4
    candidate["forward_spearman_non_regressed_folds"] = 3
    candidate["forward_pairwise_non_regressed_folds"] = 4
    uncertainty = pd.DataFrame(
        [
            {
                "candidate_key": "adaptive",
                "competition": "ALL",
                "method": "conservative_envelope",
                "reliable_improvement": False,
            },
            {
                "candidate_key": "adaptive",
                "competition": "ALL",
                "method": "conservative_envelope",
                "reliable_improvement": False,
            },
        ]
    )
    assert classify_result(candidate, current, strong, uncertainty) == "INCONCLUSIVE"
