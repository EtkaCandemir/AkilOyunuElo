from __future__ import annotations

import math

import pandas as pd
import pytest

from ao_elo.european_prior_recalibration import (
    EuropeanPriorRecalibrationConfig,
    apply_european_prior_recalibration,
    candidate_grid,
    participation_grid,
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


def test_linear_exposure_scale_preserves_raw_evidence_ordering() -> None:
    frame = pd.concat([seed_frame().iloc[[0]]] * 5, ignore_index=True)
    frame["team_id"] = range(1, 6)
    frame["european_exposure"] = [0.0, 0.4, 0.67, 0.8, 1.0]
    result = apply_european_prior_recalibration(
        frame,
        EuropeanPriorRecalibrationConfig(
            exposure_family="LINEAR_SCALE",
            exposure_scale=0.65,
        ),
    )

    assert result["candidate_effective_exposure"].tolist() == pytest.approx(
        [0.0, 0.26, 0.4355, 0.52, 0.65]
    )
    assert result["candidate_effective_exposure"].is_monotonic_increasing
    assert result.loc[0, "candidate_ao_first_elo"] == pytest.approx(1200.0)


def test_linear_exposure_key_is_distinct_and_validation_is_strict() -> None:
    baseline = EuropeanPriorRecalibrationConfig()
    linear = EuropeanPriorRecalibrationConfig(
        exposure_family="LINEAR_SCALE", exposure_scale=0.65
    )
    assert linear.key != baseline.key
    assert linear.key.endswith("_xlinear0.65")

    with pytest.raises(ValueError, match="exposure_family"):
        EuropeanPriorRecalibrationConfig(exposure_family="UNKNOWN").validate()
    with pytest.raises(ValueError, match="exposure_scale"):
        EuropeanPriorRecalibrationConfig(exposure_scale=1.01).validate()


def test_cap_tail_changes_only_exposure_above_the_active_cap() -> None:
    frame = pd.concat([seed_frame().iloc[[0]]] * 6, ignore_index=True)
    frame["team_id"] = range(1, 7)
    frame["european_exposure"] = [0.0, 0.4, 0.65, 0.67, 0.8, 1.0]
    result = apply_european_prior_recalibration(
        frame,
        EuropeanPriorRecalibrationConfig(
            exposure_family="CAP_TAIL",
            exposure_cap=0.65,
            exposure_tail_beta=0.2,
        ),
    )

    assert result["candidate_effective_exposure"].tolist() == pytest.approx(
        [0.0, 0.4, 0.65, 0.654, 0.68, 0.72]
    )
    assert result["candidate_effective_exposure"].is_monotonic_increasing


def test_cap_tail_endpoints_match_hard_cap_and_no_cap() -> None:
    frame = pd.concat([seed_frame().iloc[[0]]] * 5, ignore_index=True)
    frame["team_id"] = range(1, 6)
    frame["european_exposure"] = [0.0, 0.4, 0.65, 0.8, 1.0]
    hard_cap = apply_european_prior_recalibration(
        frame, EuropeanPriorRecalibrationConfig()
    )
    beta_zero = apply_european_prior_recalibration(
        frame,
        EuropeanPriorRecalibrationConfig(
            exposure_family="CAP_TAIL", exposure_tail_beta=0.0
        ),
    )
    beta_one = apply_european_prior_recalibration(
        frame,
        EuropeanPriorRecalibrationConfig(
            exposure_family="CAP_TAIL", exposure_tail_beta=1.0
        ),
    )

    assert beta_zero["candidate_effective_exposure"].tolist() == pytest.approx(
        hard_cap["candidate_effective_exposure"].tolist()
    )
    assert beta_one["candidate_effective_exposure"].tolist() == pytest.approx(
        frame["european_exposure"].tolist()
    )
    assert beta_zero.iloc[0]["candidate_key"].endswith("_xtail0")
    assert beta_one.iloc[0]["candidate_key"].endswith("_xtail1")


def test_cap_tail_rejects_beta_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match="exposure_tail_beta"):
        EuropeanPriorRecalibrationConfig(exposure_tail_beta=-0.01).validate()
    with pytest.raises(ValueError, match="exposure_tail_beta"):
        EuropeanPriorRecalibrationConfig(exposure_tail_beta=1.01).validate()


@pytest.mark.parametrize("knee", [0.4, 0.5, 0.55, 0.6, 0.625, 0.64])
@pytest.mark.parametrize("power", [1.0, 1.5, 2.0, 3.0])
def test_capped_bridge_is_monotonic_bounded_and_keeps_endpoints(
    knee: float,
    power: float,
) -> None:
    frame = pd.concat([seed_frame().iloc[[0]]] * 101, ignore_index=True)
    frame["team_id"] = range(1, 102)
    frame["european_exposure"] = [index / 100.0 for index in range(101)]
    result = apply_european_prior_recalibration(
        frame,
        EuropeanPriorRecalibrationConfig(
            exposure_family="CAPPED_BRIDGE",
            exposure_cap=0.65,
            exposure_knee=knee,
            exposure_power=power,
        ),
    )
    effective = result["candidate_effective_exposure"]
    below_knee = frame["european_exposure"] <= knee

    assert effective.loc[below_knee].tolist() == pytest.approx(
        frame.loc[below_knee, "european_exposure"].tolist()
    )
    assert effective.iloc[-1] == pytest.approx(0.65)
    assert effective.is_monotonic_increasing
    assert (effective <= frame["european_exposure"] + 1e-12).all()
    assert effective.max() <= 0.65 + 1e-12


def test_capped_bridge_separates_partial_from_full_exposure() -> None:
    frame = pd.concat([seed_frame().iloc[[0]]] * 4, ignore_index=True)
    frame["team_id"] = range(1, 5)
    frame["european_exposure"] = [0.5, 0.6, 0.67, 1.0]
    result = apply_european_prior_recalibration(
        frame,
        EuropeanPriorRecalibrationConfig(
            exposure_family="CAPPED_BRIDGE",
            exposure_cap=0.65,
            exposure_knee=0.6,
            exposure_power=1.0,
        ),
    )

    assert result["candidate_effective_exposure"].tolist() == pytest.approx(
        [0.5, 0.6, 0.60875, 0.65]
    )
    assert result.iloc[0]["candidate_key"].endswith("_xbridge0.6g1")


def test_capped_bridge_validation_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError, match="exposure_knee"):
        EuropeanPriorRecalibrationConfig(exposure_knee=-0.01).validate()
    with pytest.raises(ValueError, match="below exposure_cap"):
        EuropeanPriorRecalibrationConfig(
            exposure_family="CAPPED_BRIDGE",
            exposure_cap=0.65,
            exposure_knee=0.65,
        ).validate()
    with pytest.raises(ValueError, match="at least 1"):
        EuropeanPriorRecalibrationConfig(
            exposure_family="CAPPED_BRIDGE",
            exposure_knee=0.6,
            exposure_power=0.9,
        ).validate()


@pytest.mark.parametrize("power", [0.5, 1.0, 1.5, 1.8, 0.65 / 0.35])
def test_soft_power_exposure_is_monotonic_and_keeps_endpoints(power: float) -> None:
    frame = pd.concat([seed_frame().iloc[[0]]] * 101, ignore_index=True)
    frame["team_id"] = range(1, 102)
    frame["european_exposure"] = [index / 100.0 for index in range(101)]
    result = apply_european_prior_recalibration(
        frame,
        EuropeanPriorRecalibrationConfig(
            exposure_family="SOFT_POWER",
            exposure_scale=0.65,
            exposure_power=power,
        ),
    )
    effective = result["candidate_effective_exposure"]

    assert effective.iloc[0] == pytest.approx(0.0)
    assert effective.iloc[-1] == pytest.approx(0.65)
    assert effective.is_monotonic_increasing
    assert (effective <= frame["european_exposure"] + 1e-12).all()


def test_soft_power_rejects_a_non_monotonic_parameter() -> None:
    with pytest.raises(ValueError, match="monotonicity"):
        EuropeanPriorRecalibrationConfig(
            exposure_family="SOFT_POWER",
            exposure_scale=0.65,
            exposure_power=2.0,
        ).validate()


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


def test_participation_grid_pins_every_other_axis_to_production() -> None:
    from ao_elo.config import AOEuropeanEloConfig

    active = AOEuropeanEloConfig.active()
    grid = participation_grid()
    assert len(grid) == 7
    assert len({config.key for config in grid}) == len(grid)
    # Every other axis sits on the live production value, so a difference in
    # this grid can only be caused by the shrinkage.
    assert {config.exposure_cap for config in grid} == {active.max_european_exposure}
    assert {config.european_tail_beta for config in grid} == {active.european_tail_beta}
    assert {config.history_benchmark for config in grid} == {
        active.european_history_benchmark
    }
    assert {config.domestic_boost_scale for config in grid} == {1.0}
    assert {config.prior_boost_scale for config in grid} == {1.0}
    assert {config.uel_quality for config in grid} == {1.0}
    assert {config.uecl_quality for config in grid} == {1.0}
    # The active value must be a candidate, otherwise nothing measures "keep
    # what we serve today".
    assert active.european_participation_shrinkage in {
        config.participation_shrinkage for config in grid
    }


def test_the_shrinkage_never_moves_a_club_that_played_every_season() -> None:
    """k is neutral at full participation, so it cannot touch Bayern or Barca.

    This is what makes the axis safe to move at all: it only reaches clubs
    whose record is thin enough to be inflated by the normalization.
    """

    frame = seed_frame()
    assert (frame["weighted_season_exposure"] == 1.0).all()
    base = apply_european_prior_recalibration(
        frame, EuropeanPriorRecalibrationConfig(exposure_cap=0.65, european_tail_beta=1.0)
    )
    for shrinkage in (0.35, 0.5, 1.0, 2.5):
        moved = apply_european_prior_recalibration(
            frame,
            EuropeanPriorRecalibrationConfig(
                exposure_cap=0.65,
                european_tail_beta=1.0,
                participation_shrinkage=shrinkage,
            ),
        )
        pd.testing.assert_series_equal(
            moved["candidate_european_prior"],
            base["candidate_european_prior"],
        )


def test_raising_the_shrinkage_shrinks_a_partial_record_towards_the_raw_sum() -> None:
    for played in (0.2, 0.5, 0.8):
        previous = None
        for shrinkage in (0.2, 0.35, 0.5, 1.0, 2.5):
            rate = participation_normalized_history(12.0, played, shrinkage)
            assert rate >= 12.0
            if previous is not None:
                assert rate < previous
            previous = rate
        # The limit is the raw sum: no normalization at all.
        assert participation_normalized_history(12.0, played, 1e9) == pytest.approx(
            12.0, rel=1e-6
        )
