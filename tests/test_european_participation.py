from __future__ import annotations

"""Guard the participation normalization of the European Prior.

The layer's whole safety argument is that it cannot touch a club whose
five-season European evidence is complete: at full participation the rate is
the published history exactly, so the correction can only ever act on a real
gap. These tests pin that, the two boundaries around it, and the control arm
that makes a gain attributable to the participation structure rather than to
the mere fact of lifting low-exposure clubs.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ao_elo.config import AOEuropeanEloConfig
from ao_elo.european_participation import (
    ARMS,
    BASELINE,
    BLIND_LIFT,
    NORMALIZED,
    SHRINKAGE_GRID,
    ParticipationNormalizationConfig,
    apply_participation_normalization,
    blind_control_config,
    calibrate_blind_lift,
    candidate_grid,
    participation_rate,
    production_control_config,
)


def seed_frame(rows: list[dict[str, object]] | None = None) -> pd.DataFrame:
    """A small seed frame spanning full, partial and zero participation."""
    if rows is None:
        rows = [
            # full participation: must never move
            {"team_id": "full", "hist": 8.0, "pw": 1.00, "exposure": 0.90},
            # partial participation: the population the layer exists for
            {"team_id": "half", "hist": 1.63, "pw": 0.47, "exposure": 0.46},
            {"team_id": "thin", "hist": 0.50, "pw": 0.13, "exposure": 0.12},
            # never entered: exposure is zero, so the prior is ignored
            {"team_id": "none", "hist": 0.00, "pw": 0.00, "exposure": 0.00},
        ]
    frame = pd.DataFrame(rows)
    return pd.DataFrame(
        {
            "season": "2024/25",
            "team_id": frame["team_id"],
            "weighted_european_history": frame["hist"],
            "weighted_season_exposure": frame["pw"],
            "european_exposure": frame["exposure"],
            "adjusted_domestic_prior": 1200.0,
            "adjusted_ao_first_elo": np.nan,
        }
    ).assign(
        adjusted_ao_first_elo=lambda d: apply_participation_normalization(
            d.assign(adjusted_ao_first_elo=0.0), production_control_config()
        )["candidate_ao_first_elo"]
    )


# ---------------------------------------------------------------------------
# the control arm is the production formula
# ---------------------------------------------------------------------------


def test_production_control_matches_the_active_contract() -> None:
    active = AOEuropeanEloConfig.active()
    control = production_control_config()

    assert control.arm == BASELINE
    assert control.exposure_cap == pytest.approx(active.max_european_exposure)
    assert control.history_benchmark == pytest.approx(active.european_history_benchmark)
    assert control.european_prior_max_boost == pytest.approx(
        active.european_prior_max_boost
    )
    assert control.base_rating == pytest.approx(active.base_rating)


def test_the_baseline_arm_reproduces_the_seed_it_was_handed() -> None:
    """Identity is what makes every other arm's delta meaningful."""
    frame = seed_frame()
    result = apply_participation_normalization(frame, production_control_config())

    assert result["candidate_elo_delta"].abs().max() == pytest.approx(0.0, abs=1e-9)


def test_the_baseline_arm_cannot_carry_a_candidate_parameter() -> None:
    with pytest.raises(ValueError, match="baseline arm cannot carry"):
        ParticipationNormalizationConfig(arm=BASELINE, shrinkage=0.2).validate()
    with pytest.raises(ValueError, match="baseline arm cannot carry"):
        ParticipationNormalizationConfig(
            arm=BASELINE, blind_lift_coefficient=5.0
        ).validate()


# ---------------------------------------------------------------------------
# the load-bearing invariant: complete evidence never moves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shrinkage", SHRINKAGE_GRID)
def test_full_participation_keeps_its_published_history(shrinkage: float) -> None:
    rate = participation_rate(np.array([8.0]), np.array([1.0]), shrinkage)

    assert rate[0] == pytest.approx(8.0, rel=0, abs=1e-12)


@pytest.mark.parametrize("shrinkage", SHRINKAGE_GRID)
def test_full_participation_seeds_are_untouched(shrinkage: float) -> None:
    frame = seed_frame()
    result = apply_participation_normalization(
        frame, ParticipationNormalizationConfig(arm=NORMALIZED, shrinkage=shrinkage)
    ).set_index("team_id")

    assert result.loc["full", "candidate_elo_delta"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("shrinkage", SHRINKAGE_GRID)
def test_a_club_that_never_entered_gets_no_rate(shrinkage: float) -> None:
    rate = participation_rate(np.array([0.0]), np.array([0.0]), shrinkage)

    assert rate[0] == 0.0
    assert np.isfinite(rate).all()


def test_zero_exposure_seeds_land_on_the_adjusted_domestic_prior() -> None:
    frame = seed_frame()
    result = apply_participation_normalization(
        frame, ParticipationNormalizationConfig(arm=NORMALIZED, shrinkage=0.2)
    ).set_index("team_id")

    assert result.loc["none", "candidate_ao_first_elo"] == pytest.approx(1200.0)
    assert result.loc["none", "candidate_elo_delta"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# shape of the correction
# ---------------------------------------------------------------------------


def test_the_correction_grows_as_participation_falls() -> None:
    history = np.full(4, 2.0)
    played = np.array([1.00, 0.75, 0.50, 0.25])
    rate = participation_rate(history, played, 0.2)

    assert np.all(np.diff(rate) > 0.0)
    assert rate[0] == pytest.approx(2.0)


def test_shrinkage_damps_the_correction() -> None:
    """`k` exists to stop a tiny surviving denominator from exploding."""
    rates = [
        participation_rate(np.array([2.0]), np.array([0.13]), k)[0]
        for k in (0.0, 0.10, 0.35, 0.75)
    ]

    assert rates == sorted(rates, reverse=True)
    assert rates[0] > rates[-1]


def test_normalization_never_lowers_a_rate() -> None:
    frame = seed_frame()
    result = apply_participation_normalization(
        frame, ParticipationNormalizationConfig(arm=NORMALIZED, shrinkage=0.35)
    )

    assert (
        result["candidate_history_rate"]
        >= result["weighted_european_history"] - 1e-12
    ).all()


def test_a_partial_participant_is_raised() -> None:
    frame = seed_frame()
    result = apply_participation_normalization(
        frame, ParticipationNormalizationConfig(arm=NORMALIZED, shrinkage=0.2)
    ).set_index("team_id")

    assert result.loc["half", "candidate_elo_delta"] > 1.0
    assert result.loc["thin", "candidate_elo_delta"] > 1.0


# ---------------------------------------------------------------------------
# the blind control
# ---------------------------------------------------------------------------


def test_the_blind_control_matches_the_candidate_movement() -> None:
    """A smaller control would make a null result unreadable."""
    frame = seed_frame()
    candidate = ParticipationNormalizationConfig(arm=NORMALIZED, shrinkage=0.2)
    coefficient = calibrate_blind_lift(frame, candidate)

    moved_candidate = apply_participation_normalization(frame, candidate)
    moved_control = apply_participation_normalization(
        frame, blind_control_config(coefficient)
    )

    assert moved_control["candidate_elo_delta"].abs().mean() == pytest.approx(
        moved_candidate["candidate_elo_delta"].abs().mean()
    )


def test_the_blind_control_reads_no_participation() -> None:
    """Two clubs with identical exposure but different participation must
    receive the identical blind lift; that is what makes it blind."""
    frame = seed_frame(
        [
            {"team_id": "played_well", "hist": 6.0, "pw": 1.00, "exposure": 0.50},
            {"team_id": "barely_played", "hist": 6.0, "pw": 0.20, "exposure": 0.50},
        ]
    )
    result = apply_participation_normalization(
        frame, blind_control_config(40.0)
    ).set_index("team_id")

    lift = result["candidate_ao_first_elo"] - result["adjusted_ao_first_elo"]

    assert lift.loc["played_well"] == pytest.approx(lift.loc["barely_played"])


def test_the_blind_control_is_calibrated_against_a_candidate() -> None:
    with pytest.raises(ValueError, match="calibrated against a normalized arm"):
        calibrate_blind_lift(seed_frame(), production_control_config())


def test_arms_cannot_borrow_each_others_parameters() -> None:
    with pytest.raises(ValueError, match="not lifted"):
        ParticipationNormalizationConfig(
            arm=NORMALIZED, shrinkage=0.2, blind_lift_coefficient=5.0
        ).validate()
    with pytest.raises(ValueError, match="no participation parameter"):
        ParticipationNormalizationConfig(
            arm=BLIND_LIFT, shrinkage=0.2, blind_lift_coefficient=5.0
        ).validate()


# ---------------------------------------------------------------------------
# plumbing
# ---------------------------------------------------------------------------


def test_the_grid_opens_with_the_production_control() -> None:
    grid = candidate_grid()

    assert grid[0].key == BASELINE
    assert [config.shrinkage for config in grid[1:]] == list(SHRINKAGE_GRID)
    assert len({config.key for config in grid}) == len(grid)
    assert set(ARMS) == {BASELINE, BLIND_LIFT, NORMALIZED}


def test_unknown_arms_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown arm"):
        ParticipationNormalizationConfig(arm="PARTICIPATION_GUESS").validate()


@pytest.mark.parametrize(
    "column", ["weighted_season_exposure", "weighted_european_history"]
)
def test_missing_input_columns_are_rejected(column: str) -> None:
    frame = seed_frame().drop(columns=[column])

    with pytest.raises(ValueError, match="missing columns"):
        apply_participation_normalization(frame, production_control_config())


def test_duplicate_team_seasons_are_rejected() -> None:
    frame = pd.concat([seed_frame(), seed_frame()], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate team-season"):
        apply_participation_normalization(frame, production_control_config())


def test_non_finite_inputs_are_rejected() -> None:
    frame = seed_frame()
    frame.loc[0, "weighted_european_history"] = np.nan

    with pytest.raises(ValueError, match="must be finite"):
        apply_participation_normalization(frame, production_control_config())


def test_participation_outside_the_unit_interval_is_rejected() -> None:
    frame = seed_frame()
    frame.loc[0, "weighted_season_exposure"] = 1.4

    with pytest.raises(ValueError, match="within"):
        apply_participation_normalization(frame, production_control_config())


def test_negative_shrinkage_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        participation_rate(np.array([1.0]), np.array([0.5]), -0.1)
