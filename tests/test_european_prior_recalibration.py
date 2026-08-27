from __future__ import annotations

import math

import pandas as pd
import pytest

from ao_elo.european_prior_recalibration import (
    EuropeanPriorRecalibrationConfig,
    apply_european_prior_recalibration,
    candidate_grid,
    ranking_uncertainty_summary,
)


def seed_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": ["2026/27"] * 3,
            "team_id": [1, 2, 3],
            "competition": ["UCL", "UEL", "UECL"],
            "weighted_european_history": [12.0, 12.0, 12.0],
            "european_exposure": [1.0, 1.0, 0.0],
            "adjusted_domestic_prior": [1200.0, 1200.0, 1100.0],
            "adjusted_ao_first_elo": [1722.0, 1722.0, 1100.0],
        }
    )


def test_baseline_formula_matches_the_active_contract() -> None:
    frame = seed_frame().iloc[[0]].copy()
    result = apply_european_prior_recalibration(
        frame, EuropeanPriorRecalibrationConfig()
    ).iloc[0]
    expected_norm = math.log1p(12.0) / math.log1p(20.0)
    assert result.candidate_history_norm == pytest.approx(expected_norm)
    expected_prior = 500.0 + 1559.714795008913 * float(result.candidate_history_norm)
    expected = 1200.0 + 0.85 * (expected_prior - 1200.0)
    assert result.candidate_ao_first_elo == pytest.approx(expected)


def test_lower_quality_and_exposure_reduce_only_supported_evidence() -> None:
    config = EuropeanPriorRecalibrationConfig(
        history_benchmark=28.0,
        prior_boost_scale=0.925,
        exposure_cap=0.80,
        uel_quality=0.90,
        uecl_quality=0.80,
    )
    result = apply_european_prior_recalibration(seed_frame(), config)

    assert result.loc[1, "candidate_ao_first_elo"] < result.loc[0, "candidate_ao_first_elo"]
    assert result.loc[2, "candidate_ao_first_elo"] == pytest.approx(1100.0)
    assert result["candidate_effective_exposure"].max() <= 0.80


def test_candidate_grid_preserves_quality_hierarchy_and_unique_keys() -> None:
    candidates = candidate_grid()
    assert len(candidates) == 81
    assert len({candidate.key for candidate in candidates}) == len(candidates)
    assert all(candidate.uel_quality >= candidate.uecl_quality for candidate in candidates)


def test_ranking_veto_requires_reliable_harm() -> None:
    mixed = pd.DataFrame(
        {
            "delta_seed_spearman": [-0.004, -0.002, 0.0, 0.003, 0.001, 0.0],
            "delta_seed_pairwise_accuracy": [-0.002, -0.001, 0.0, 0.001, 0.001, 0.0],
        }
    )
    harmful = pd.DataFrame(
        {
            "delta_seed_spearman": [-0.01] * 6,
            "delta_seed_pairwise_accuracy": [-0.005] * 6,
        }
    )

    mixed_result = ranking_uncertainty_summary(mixed, 500)
    harmful_result = ranking_uncertainty_summary(harmful, 500)

    assert not mixed_result["reliable_harm"].any()
    assert harmful_result["reliable_harm"].all()
