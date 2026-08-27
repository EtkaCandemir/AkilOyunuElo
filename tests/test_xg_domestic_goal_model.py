from __future__ import annotations

"""Guard the xG form term that sits on top of the domestic attack/defence arm.

The claim this run supports is one step stronger than the previous one and one
step easier to overstate: the xG form term still pays *after* the model already
carries domestic attack and defence, in the segment where xG exists. Three
properties make that claim checkable, and each has a test below.

1. The baseline really is the published domestic arm. If the `none` arm drifted
   away from `fit_european_poisson_transfer`, the candidate would be beating a
   weakened stand-in rather than the repository's best score model.
2. The form term is additive. Zero coefficients must leave the transfer's rates
   untouched, which is what "on top of, not instead of" means numerically.
3. A coefficient is never applied where it was never fitted - the run drops a
   fold whose training half carries no xG, and the fit refuses that frame.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ao_elo.domestic_poisson import (
    fit_european_poisson_transfer,
    predict_european_poisson_transfer,
)
from ao_elo.xg_domestic_goal_model import (
    ARM_BY_SOURCE,
    COEFFICIENT_MAX,
    DOMESTIC_AD,
    DOMESTIC_AD_GOALS_FORM,
    DOMESTIC_AD_XG_FORM,
    DomesticFormExpectationConfig,
    attach_form_features,
    candidate_form_sources,
    fit_domestic_form_expectation,
    independent_poisson_nll,
    predict_domestic_form_expectation,
)
from ao_elo.xg_goal_model import LAMBDA_MAX, LAMBDA_MIN, build_form_features, form_prior


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def match(
    index: int,
    home: str,
    away: str,
    home_goals: int,
    away_goals: int,
    *,
    xg: tuple[float, float] | None = None,
    expected_home_score: float = 0.55,
    attack: tuple[float, float] = (0.30, -0.20),
    defence: tuple[float, float] = (-0.10, 0.25),
) -> dict[str, object]:
    eligible = xg is not None
    outcome = 0 if home_goals > away_goals else 1 if home_goals == away_goals else 2
    home_probability = float(expected_home_score) * 0.9
    away_probability = (1.0 - float(expected_home_score)) * 0.9
    row: dict[str, object] = {
        "match_id": f"m{index:03d}",
        "season": "2024/25",
        "kickoff_utc": pd.Timestamp("2024-08-01T19:00:00Z") + pd.Timedelta(days=index),
        "home_club_id": home,
        "away_club_id": away,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "actual_class": outcome,
        "expected_home_score": expected_home_score,
        "ao_home_probability": home_probability,
        "ao_draw_probability": 1.0 - home_probability - away_probability,
        "ao_away_probability": away_probability,
        "domestic_poisson_coverage": "BOTH",
        "domestic_poisson_venue_edge": 0.05,
        "domestic_poisson_venue_edge_raw": 0.04,
        "home_domestic_poisson_attack": attack[0],
        "away_domestic_poisson_attack": attack[1],
        "home_domestic_poisson_defence": defence[0],
        "away_domestic_poisson_defence": defence[1],
        "home_domestic_poisson_attack_raw_z": attack[0] * 1.1,
        "away_domestic_poisson_attack_raw_z": attack[1] * 1.1,
        "home_domestic_poisson_defence_raw_z": defence[0] * 1.1,
        "away_domestic_poisson_defence_raw_z": defence[1] * 1.1,
        "xg_home": float(xg[0]) if eligible else 0.0,
        "xg_away": float(xg[1]) if eligible else 0.0,
        "xg_analysis_eligible": eligible,
    }
    return row


def sample_frame(*, with_xg: bool = True) -> pd.DataFrame:
    """A small league where every club plays often enough to build history."""

    clubs = ["A", "B", "C", "D"]
    rows: list[dict[str, object]] = []
    index = 0
    for round_number in range(9):
        for offset in range(2):
            home = clubs[(round_number + offset) % 4]
            away = clubs[(round_number + offset + 1) % 4]
            home_goals = (round_number + offset) % 4
            away_goals = (round_number * 2 + offset) % 3
            rows.append(
                match(
                    index,
                    home,
                    away,
                    home_goals,
                    away_goals,
                    xg=(home_goals + 0.3, away_goals + 0.2) if with_xg else None,
                    expected_home_score=0.45 + 0.02 * ((index % 5) - 2),
                    attack=(0.10 * (index % 3), -0.10 * (index % 2)),
                    defence=(-0.05 * (index % 4), 0.05 * (index % 3)),
                )
            )
            index += 1
    return pd.DataFrame(rows)


@pytest.fixture(name="frame")
def fixture_frame() -> pd.DataFrame:
    return sample_frame()


@pytest.fixture(name="featured")
def fixture_featured(frame: pd.DataFrame) -> pd.DataFrame:
    return attach_form_features(frame, "xg")[0]


# ---------------------------------------------------------------------------
# 1. the baseline stays the published domestic arm
# ---------------------------------------------------------------------------


def test_baseline_arm_reproduces_the_published_transfer(frame: pd.DataFrame) -> None:
    """`none` must fit the identical model `fit_european_poisson_transfer` does.

    This is the load-bearing test of the whole comparison. The candidate is
    only interesting if what it beats is the repository's best score arm.
    """
    featured, _ = attach_form_features(frame, "none")

    ours = fit_domestic_form_expectation(
        featured, "none", l2_strength=1.0, use_reliability=True, use_venue=True
    )
    theirs = fit_european_poisson_transfer(
        frame, l2_strength=1.0, use_reliability=True, use_venue=True
    )

    for name in (
        "mu",
        "elo_slope",
        "attack_coefficient",
        "defence_coefficient",
        "venue_coefficient",
    ):
        assert getattr(ours, name) == pytest.approx(getattr(theirs, name), abs=1e-8)


def test_baseline_arm_reproduces_the_published_rates(frame: pd.DataFrame) -> None:
    featured, _ = attach_form_features(frame, "none")
    ours = fit_domestic_form_expectation(
        featured, "none", l2_strength=1.0, use_reliability=True, use_venue=False
    )
    theirs = fit_european_poisson_transfer(
        frame, l2_strength=1.0, use_reliability=True, use_venue=False
    )

    mine = predict_domestic_form_expectation(featured, ours)
    published = predict_european_poisson_transfer(frame, theirs)

    assert mine["lambda_home"].to_numpy() == pytest.approx(
        published["lambda_home"].to_numpy(), abs=1e-8
    )
    assert mine["lambda_away"].to_numpy() == pytest.approx(
        published["lambda_away"].to_numpy(), abs=1e-8
    )


def test_baseline_arm_cannot_carry_form_coefficients() -> None:
    with pytest.raises(ValueError, match="baseline arm cannot carry form"):
        DomesticFormExpectationConfig(
            form_source="none",
            mu=0.3,
            elo_slope=0.7,
            attack_coefficient=0.1,
            defence_coefficient=0.1,
            venue_coefficient=0.0,
            form_attack=0.2,
            form_defence=0.0,
            l2_strength=1.0,
        ).validate()


def test_baseline_arm_refuses_a_frame_that_already_carries_form(
    featured: pd.DataFrame,
) -> None:
    """Handing the baseline a populated form column would silently bias it."""
    with pytest.raises(ValueError, match="must be handed zeroed form"):
        fit_domestic_form_expectation(featured, "none", l2_strength=1.0)


# ---------------------------------------------------------------------------
# 2. the form term is additive, never a replacement
# ---------------------------------------------------------------------------


def test_zero_form_coefficients_leave_the_transfer_rates_untouched(
    frame: pd.DataFrame, featured: pd.DataFrame
) -> None:
    """Populated form columns must be inert while their coefficients are zero."""
    shared = dict(
        mu=0.30,
        elo_slope=0.70,
        attack_coefficient=0.09,
        defence_coefficient=0.06,
        venue_coefficient=0.02,
        l2_strength=1.0,
        form_attack=0.0,
        form_defence=0.0,
    )
    zeroed, _ = attach_form_features(frame, "none")
    without = predict_domestic_form_expectation(
        zeroed, DomesticFormExpectationConfig(form_source="none", **shared)
    )
    with_source = predict_domestic_form_expectation(
        featured, DomesticFormExpectationConfig(form_source="xg", **shared)
    )

    assert featured["home_form_for"].abs().sum() > 0.0
    assert with_source["lambda_home"].to_numpy() == pytest.approx(
        without["lambda_home"].to_numpy()
    )
    assert with_source["lambda_away"].to_numpy() == pytest.approx(
        without["lambda_away"].to_numpy()
    )


def test_form_moves_the_rate_in_the_signed_direction(featured: pd.DataFrame) -> None:
    """Scoring form lifts a side's rate; the opponent's conceding form lowers it."""
    config = DomesticFormExpectationConfig(
        form_source="xg",
        mu=0.30,
        elo_slope=0.70,
        attack_coefficient=0.09,
        defence_coefficient=0.06,
        venue_coefficient=0.0,
        form_attack=0.40,
        form_defence=0.30,
        l2_strength=1.0,
    )
    rates = predict_domestic_form_expectation(featured, config)
    baseline = predict_domestic_form_expectation(
        featured,
        DomesticFormExpectationConfig(
            form_source="xg",
            mu=0.30,
            elo_slope=0.70,
            attack_coefficient=0.09,
            defence_coefficient=0.06,
            venue_coefficient=0.0,
            form_attack=0.0,
            form_defence=0.0,
            l2_strength=1.0,
        ),
    )
    expected_sign = np.sign(
        0.40 * featured["home_form_for"].to_numpy(float)
        - 0.30 * featured["away_form_against"].to_numpy(float)
    )
    observed_sign = np.sign(
        rates["lambda_home"].to_numpy() - baseline["lambda_home"].to_numpy()
    )
    moved = expected_sign != 0.0

    assert moved.any()
    assert np.array_equal(observed_sign[moved], expected_sign[moved])


def test_rates_stay_inside_the_production_window(featured: pd.DataFrame) -> None:
    config = DomesticFormExpectationConfig(
        form_source="xg",
        mu=1.30,
        elo_slope=3.00,
        attack_coefficient=COEFFICIENT_MAX,
        defence_coefficient=COEFFICIENT_MAX,
        venue_coefficient=COEFFICIENT_MAX,
        form_attack=COEFFICIENT_MAX,
        form_defence=COEFFICIENT_MAX,
        l2_strength=0.0,
    )
    rates = predict_domestic_form_expectation(featured, config)

    assert rates["lambda_home"].between(LAMBDA_MIN, LAMBDA_MAX).all()
    assert rates["lambda_away"].between(LAMBDA_MIN, LAMBDA_MAX).all()


# ---------------------------------------------------------------------------
# 3. a coefficient is never applied where it was never fitted
# ---------------------------------------------------------------------------


def test_fit_refuses_a_source_with_no_signal_in_training() -> None:
    """A fold whose training half carries no xG must fail loudly, not quietly.

    Left to itself the optimizer would keep its starting value on an all-zero
    column and that unfitted coefficient would then be applied to a test season
    that does carry xG.
    """
    frame = sample_frame(with_xg=False)
    featured = build_form_features(frame, "goals")
    featured[
        ["home_form_for", "home_form_against", "away_form_for", "away_form_against"]
    ] = 0.0

    with pytest.raises(ValueError, match="carries no signal"):
        fit_domestic_form_expectation(featured, "goals", l2_strength=0.0)


def test_xg_history_ignores_matches_without_xg() -> None:
    """An ineligible row must not advance the xG history it has no value for."""
    frame = sample_frame(with_xg=True)
    frame.loc[frame.index[:8], "xg_analysis_eligible"] = False
    frame.loc[frame.index[:8], ["xg_home", "xg_away"]] = 0.0
    featured, _ = attach_form_features(frame, "xg")

    blind = featured.iloc[:8][["home_form_for", "away_form_for"]]

    assert blind.to_numpy().sum() == 0.0
    assert featured.iloc[8:]["home_form_for"].abs().sum() > 0.0


# ---------------------------------------------------------------------------
# the training-only centering level
# ---------------------------------------------------------------------------


def test_form_prior_defaults_to_what_the_builder_would_derive(
    frame: pd.DataFrame,
) -> None:
    default = build_form_features(frame, "goals")
    explicit = build_form_features(frame, "goals", prior=form_prior(frame, "goals"))

    for column in ("home_form_for", "away_form_against"):
        assert explicit[column].to_numpy() == pytest.approx(default[column].to_numpy())


def test_training_prior_is_taken_from_training_rows_only(frame: pd.DataFrame) -> None:
    training = frame.iloc[:10]
    _, prior = attach_form_features(frame, "goals", training_frame=training)

    assert prior == pytest.approx(form_prior(training, "goals"))
    assert prior != pytest.approx(form_prior(frame, "goals"))


def test_a_higher_prior_lowers_every_scoring_form_value(frame: pd.DataFrame) -> None:
    low = build_form_features(frame, "goals", prior=0.5)
    high = build_form_features(frame, "goals", prior=2.5)
    moved = low["home_form_for"].ne(0.0) | high["home_form_for"].ne(0.0)

    assert moved.any()
    assert (high.loc[moved, "home_form_for"] < low.loc[moved, "home_form_for"]).all()


# ---------------------------------------------------------------------------
# plumbing
# ---------------------------------------------------------------------------


def test_three_arms_are_offered_and_named() -> None:
    assert candidate_form_sources() == ("none", "goals", "xg")
    assert ARM_BY_SOURCE == {
        "none": DOMESTIC_AD,
        "goals": DOMESTIC_AD_GOALS_FORM,
        "xg": DOMESTIC_AD_XG_FORM,
    }


def test_fitted_config_records_the_prior_it_was_centered_on(
    frame: pd.DataFrame,
) -> None:
    training = frame.iloc[:12]
    featured, prior = attach_form_features(frame, "xg", training_frame=training)
    config = fit_domestic_form_expectation(
        featured, "xg", l2_strength=1.0, form_prior_value=prior
    )

    assert config.form_prior == pytest.approx(prior)
    assert config.arm == DOMESTIC_AD_XG_FORM
    assert config.fingerprint != fit_domestic_form_expectation(
        featured, "xg", l2_strength=10.0, form_prior_value=prior
    ).fingerprint


def test_independent_poisson_nll_matches_the_closed_form() -> None:
    value = independent_poisson_nll([2.0], [1.0], [1.5], [1.2])
    expected = (
        1.5 - 2.0 * np.log(1.5) + np.log(2.0) + 1.2 - 1.0 * np.log(1.2) + 0.0
    )

    assert value == pytest.approx(expected)


@pytest.mark.parametrize(
    "field,value",
    [
        ("elo_slope", 4.0),
        ("attack_coefficient", -0.1),
        ("form_attack", COEFFICIENT_MAX + 0.1),
        ("form_defence", -0.01),
        ("rho", 0.4),
    ],
)
def test_out_of_range_parameters_are_rejected(field: str, value: float) -> None:
    parameters = dict(
        form_source="xg",
        mu=0.3,
        elo_slope=0.7,
        attack_coefficient=0.1,
        defence_coefficient=0.1,
        venue_coefficient=0.0,
        form_attack=0.2,
        form_defence=0.1,
        l2_strength=1.0,
    )
    parameters[field] = value

    with pytest.raises(ValueError):
        DomesticFormExpectationConfig(**parameters).validate()


def test_unknown_form_source_is_rejected(featured: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="unknown form source"):
        fit_domestic_form_expectation(featured, "shots", l2_strength=1.0)
