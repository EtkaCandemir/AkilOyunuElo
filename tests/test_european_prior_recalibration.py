from __future__ import annotations

import math

import pandas as pd
import pytest

from ao_elo.european_prior_recalibration import (
    EuropeanPriorRecalibrationConfig,
    apply_european_prior_recalibration,
    candidate_grid,
    ranking_uncertainty_summary,
    tail_and_domestic_grid,
)
from ao_elo.scoring import participation_normalized_history


def seed_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": ["2026/27"] * 3,
            "team_id": [1, 2, 3],
            "competition": ["UCL", "UEL", "UECL"],
            "weighted_european_history": [12.0, 12.0, 12.0],
            # Full participation keeps the normalized rate equal to the raw
            # history, so these fixtures isolate the axis under test.
            "weighted_season_exposure": [1.0, 1.0, 1.0],
            "european_exposure": [1.0, 1.0, 0.0],
            "domestic_prior": [1200.0, 1200.0, 1100.0],
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


def test_full_participation_leaves_the_rate_equal_to_raw_history() -> None:
    # (1+k)/(1+k) = 1: bu ozdeslik olmadan arastirma yuzeyi production'i
    # yeniden uretemez.
    frame = seed_frame().iloc[[0]].copy()
    result = apply_european_prior_recalibration(
        frame, EuropeanPriorRecalibrationConfig()
    ).iloc[0]
    assert result.candidate_history_rate == pytest.approx(12.0)


def test_partial_participation_raises_the_rate() -> None:
    frame = seed_frame().iloc[[0]].copy()
    frame["weighted_season_exposure"] = 0.5
    result = apply_european_prior_recalibration(
        frame, EuropeanPriorRecalibrationConfig()
    ).iloc[0]
    expected = participation_normalized_history(12.0, 0.5, 0.2)
    assert result.candidate_history_rate == pytest.approx(expected)
    assert result.candidate_history_rate > 12.0


def test_tail_beta_zero_reproduces_the_production_clip() -> None:
    frame = seed_frame().iloc[[0]].copy()
    frame["weighted_european_history"] = 40.0  # benchmark 20 -> norm > 1
    result = apply_european_prior_recalibration(
        frame, EuropeanPriorRecalibrationConfig()
    ).iloc[0]
    assert result.candidate_uncapped_history_norm > 1.0
    assert result.candidate_history_norm == pytest.approx(1.0)
    assert not bool(result.candidate_tail_active)


def test_tail_beta_separates_clubs_past_the_benchmark() -> None:
    frame = pd.concat([seed_frame().iloc[[0]]] * 2, ignore_index=True)
    frame["team_id"] = [1, 2]
    frame["weighted_european_history"] = [30.0, 40.0]
    flat = apply_european_prior_recalibration(
        frame, EuropeanPriorRecalibrationConfig()
    )
    tailed = apply_european_prior_recalibration(
        frame, EuropeanPriorRecalibrationConfig(european_tail_beta=0.5)
    )
    # beta = 0 iki kulubu esitler; beta > 0 ayristirir ve hicbirini dusurmez.
    assert flat.loc[0, "candidate_european_prior"] == pytest.approx(
        flat.loc[1, "candidate_european_prior"]
    )
    assert tailed.loc[1, "candidate_european_prior"] > tailed.loc[0, "candidate_european_prior"]
    assert (tailed["candidate_european_prior"] >= flat["candidate_european_prior"]).all()
    assert tailed["candidate_tail_active"].all()


def test_tail_beta_leaves_clubs_below_the_benchmark_untouched() -> None:
    frame = seed_frame().copy()  # gecmis 12 < benchmark 20
    flat = apply_european_prior_recalibration(frame, EuropeanPriorRecalibrationConfig())
    tailed = apply_european_prior_recalibration(
        frame, EuropeanPriorRecalibrationConfig(european_tail_beta=0.75)
    )
    pd.testing.assert_series_equal(
        flat["candidate_ao_first_elo"], tailed["candidate_ao_first_elo"]
    )


def test_domestic_scale_lifts_the_prior_without_scaling_the_surprise() -> None:
    frame = seed_frame().iloc[[0]].copy()
    frame["domestic_prior"] = 1200.0
    frame["adjusted_domestic_prior"] = 1230.0  # +30 surpriz, donmus cap
    scaled = apply_european_prior_recalibration(
        frame, EuropeanPriorRecalibrationConfig(domestic_boost_scale=1.2)
    ).iloc[0]
    # Yalniz base ustu bilesen olceklenir; surpriz birebir tasinir.
    assert scaled.candidate_adjusted_domestic_prior == pytest.approx(
        500.0 + 1.2 * (1200.0 - 500.0) + 30.0
    )


def test_tail_and_domestic_grid_pins_every_other_axis() -> None:
    grid = tail_and_domestic_grid()
    assert len(grid) == 28
    assert len({config.key for config in grid}) == len(grid)
    assert {config.exposure_cap for config in grid} == {0.65}
    assert {config.history_benchmark for config in grid} == {20.0}
    assert {config.prior_boost_scale for config in grid} == {1.0}
    assert {config.uel_quality for config in grid} == {1.0}
