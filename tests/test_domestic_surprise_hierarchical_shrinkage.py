from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_domestic_surprise_hierarchical_shrinkage import (  # noqa: E402
    HierarchicalCandidate,
    hierarchical_candidate_grid,
    percentile_features,
)


def test_grid_covers_required_families_with_bounded_complexity() -> None:
    candidates = hierarchical_candidate_grid()
    assert {candidate.family for candidate in candidates} == {
        "MULTIPLICATIVE", "ADDITIVE", "GATED", "SMOOTH"
    }
    assert len(candidates) == 120
    assert min(candidate.parameter_count for candidate in candidates) >= 4
    assert max(candidate.parameter_count for candidate in candidates) <= 6
    assert {candidate.floor for candidate in candidates} == {0.20, 0.35, 0.50}
    assert {candidate.final_cap for candidate in candidates} == {40.0, 50.0, 60.0, 75.0, 100.0}


@pytest.mark.parametrize("family,parameters", [
    ("MULTIPLICATIVE", (1.0, 1.0, 0.25)),
    ("ADDITIVE", (0.5, 0.35, 0.15)),
    ("GATED", (0.35, 0.20)),
    ("SMOOTH", (0.55, 0.35, 4.0, 0.35)),
])
def test_reliability_is_finite_and_bounded(family: str, parameters: tuple[float, ...]) -> None:
    candidate = HierarchicalCandidate("test", family, 0.35, 50.0, parameters, 2 + len(parameters))
    candidate.validate()
    grid = np.linspace(0.0, 1.0, 11)
    reliability = candidate.reliability(grid, grid[::-1], grid)
    assert np.isfinite(reliability).all()
    assert (reliability >= 0.35).all()
    assert (reliability <= 1.0).all()


def test_gated_candidate_shrinks_riskiest_profile_most() -> None:
    candidate = HierarchicalCandidate("gated", "GATED", 0.35, 40.0, (0.35, 0.20), 4)
    risky = candidate.reliability(np.array([0.0]), np.array([0.0]), np.array([0.0]))[0]
    strong = candidate.reliability(np.array([1.0]), np.array([1.0]), np.array([0.0]))[0]
    strong_low_exposure = candidate.reliability(np.array([1.0]), np.array([0.8]), np.array([0.0]))[0]
    assert risky == pytest.approx(0.35)
    assert strong == pytest.approx(1.0)
    assert strong_low_exposure > risky


def test_percentiles_are_computed_within_each_season() -> None:
    frame = pd.DataFrame({
        "season": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
        "league_strength": [1, 2, 3, 4, 5, 10, 20, 30, 40, 50],
        "baseline_ao_first_elo": [500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400],
        "effective_european_exposure": [0.0] * 10,
    })
    result = percentile_features(frame)
    assert result.groupby("season")["league_reliability"].min().eq(0.2).all()
    assert result.groupby("season")["league_reliability"].max().eq(1.0).all()
    assert result.groupby("season")["rating_quintile"].nunique().eq(5).all()
