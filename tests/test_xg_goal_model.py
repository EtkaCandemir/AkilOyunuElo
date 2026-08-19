from __future__ import annotations

"""Guard the xG-informed goal-expectation model.

The claim this model supports is narrow and easy to overstate: recent xG
predicts goals better than recent goals do, *where xG exists*. These tests pin
the properties that make that claim checkable - causal history, a neutral
fallback when a club has no past, and a goals control built from the identical
structure so the comparison isolates the source rather than the extra
parameters.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ao_elo.xg_goal_model import (
    LAMBDA_MAX,
    LAMBDA_MIN,
    GoalExpectationConfig,
    build_form_features,
    candidate_sources,
    elo_z,
    fit_goal_expectation,
    predict_goal_expectations,
)


def frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    data = pd.DataFrame(rows)
    data["kickoff_utc"] = pd.to_datetime(data["kickoff_utc"], utc=True)
    return data


def match(mid, day, home, away, hg, ag, *, xgh=None, xga=None, elo_h=1500.0, elo_a=1500.0):
    eligible = xgh is not None and xga is not None
    return {
        "match_id": mid, "season": "2024/25",
        "kickoff_utc": f"2024-09-{day:02d}T19:00:00Z",
        "home_club_id": home, "away_club_id": away,
        "home_goals": hg, "away_goals": ag,
        "xg_home": xgh, "xg_away": xga, "xg_analysis_eligible": eligible,
        "home_live_pre": elo_h, "away_live_pre": elo_a, "is_neutral": False,
    }


# ---------------------------------------------------------------------------
# causality: a match may never see itself or anything later
# ---------------------------------------------------------------------------


def test_first_appearance_carries_no_form() -> None:
    data = frame([match("m1", 1, "A", "B", 3, 0, xgh=2.5, xga=0.4)])

    built = build_form_features(data, "xg")

    assert built.loc[0, "home_form_for"] == 0.0
    assert built.loc[0, "away_form_against"] == 0.0


def test_history_only_flows_forward() -> None:
    """The second match sees the first; the first never sees the second."""
    data = frame([
        match("m1", 1, "A", "B", 3, 0, xgh=2.8, xga=0.3),
        match("m2", 8, "A", "C", 0, 0, xgh=0.2, xga=0.1),
    ])

    built = build_form_features(data, "xg").set_index("match_id")

    assert built.loc["m1", "home_form_for"] == 0.0
    assert built.loc["m2", "home_form_for"] > 0.0, "A scored heavily in m1"


def test_reordering_input_does_not_change_features() -> None:
    """Features must follow kickoff order, not row order."""
    rows = [
        match("m1", 1, "A", "B", 2, 1, xgh=1.9, xga=1.1),
        match("m2", 8, "A", "C", 1, 1, xgh=1.2, xga=1.3),
        match("m3", 15, "B", "A", 0, 2, xgh=0.6, xga=2.1),
    ]
    forward = build_form_features(frame(rows), "xg").set_index("match_id")
    shuffled = build_form_features(frame(rows[::-1]), "xg").set_index("match_id")

    for column in ("home_form_for", "away_form_against"):
        assert np.allclose(
            forward[column].to_numpy(float), shuffled.loc[forward.index, column].to_numpy(float)
        )


def test_matches_without_xg_do_not_update_xg_history() -> None:
    """A blind match must leave the xG history exactly as it was."""
    with_blind = frame([
        match("m1", 1, "A", "B", 3, 0, xgh=2.6, xga=0.3),
        match("m2", 8, "A", "C", 4, 0),
        match("m3", 15, "A", "D", 1, 1, xgh=1.0, xga=1.0),
    ])
    without_blind = frame([
        match("m1", 1, "A", "B", 3, 0, xgh=2.6, xga=0.3),
        match("m3", 15, "A", "D", 1, 1, xgh=1.0, xga=1.0),
    ])

    a = build_form_features(with_blind, "xg").set_index("match_id")
    b = build_form_features(without_blind, "xg").set_index("match_id")

    assert a.loc["m3", "home_form_for"] == pytest.approx(b.loc["m3", "home_form_for"])


def test_goals_history_uses_every_match() -> None:
    """The control has no coverage gap, which is why it is the control."""
    data = frame([
        match("m1", 1, "A", "B", 3, 0),
        match("m2", 8, "A", "C", 2, 0),
    ])

    built = build_form_features(data, "goals").set_index("match_id")

    assert built.loc["m2", "home_form_for"] > 0.0


# ---------------------------------------------------------------------------
# the neutral fallback
# ---------------------------------------------------------------------------


def test_elo_only_source_produces_no_form_at_all() -> None:
    data = frame([
        match("m1", 1, "A", "B", 3, 0, xgh=2.6, xga=0.3),
        match("m2", 8, "A", "C", 1, 1, xgh=1.0, xga=1.0),
    ])

    built = build_form_features(data, "none")

    for column in ("home_form_for", "home_form_against", "away_form_for", "away_form_against"):
        assert built[column].eq(0.0).all()


def test_shrinkage_pulls_a_short_history_toward_neutral() -> None:
    """One extreme result must not move the rate as far as five would."""
    single = frame([
        match("m1", 1, "A", "B", 5, 0, xgh=4.0, xga=0.2),
        match("m2", 8, "A", "C", 1, 1, xgh=1.0, xga=1.0),
    ])

    weak = build_form_features(single, "xg", shrinkage=50.0).set_index("match_id")
    strong = build_form_features(single, "xg", shrinkage=0.0).set_index("match_id")

    assert abs(weak.loc["m2", "home_form_for"]) < abs(strong.loc["m2", "home_form_for"])


def test_unknown_source_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown form source"):
        build_form_features(frame([match("m1", 1, "A", "B", 1, 1)]), "shots")


# ---------------------------------------------------------------------------
# the fitted model
# ---------------------------------------------------------------------------


@pytest.fixture(name="sample")
def fixture_sample() -> pd.DataFrame:
    generator = np.random.default_rng(7)
    clubs = [f"C{index}" for index in range(12)]
    rows = []
    for index in range(240):
        home, away = generator.choice(clubs, 2, replace=False)
        hg, ag = generator.poisson(1.6), generator.poisson(1.2)
        rows.append(match(
            f"m{index}", 1 + index % 28, home, away, int(hg), int(ag),
            xgh=float(hg) + generator.normal(0, 0.3),
            xga=float(ag) + generator.normal(0, 0.3),
            elo_h=1500 + generator.normal(0, 80), elo_a=1500 + generator.normal(0, 80),
        ))
    data = pd.DataFrame(rows)
    data["kickoff_utc"] = pd.to_datetime(data["kickoff_utc"], utc=True)
    data[["xg_home", "xg_away"]] = data[["xg_home", "xg_away"]].clip(lower=0.0)
    return data


@pytest.mark.parametrize("source", candidate_sources())
def test_fit_returns_a_valid_config(sample: pd.DataFrame, source: str) -> None:
    config = fit_goal_expectation(sample, source, elo_scale=835.56, home_advantage=148.54)

    config.validate()
    assert config.source == source


def test_elo_only_fit_keeps_form_coefficients_at_zero(sample: pd.DataFrame) -> None:
    config = fit_goal_expectation(sample, "none", elo_scale=835.56, home_advantage=148.54)

    assert config.attack == 0.0
    assert config.defence == 0.0


def test_predicted_rates_stay_inside_the_production_window(sample: pd.DataFrame) -> None:
    config = fit_goal_expectation(sample, "xg", elo_scale=835.56, home_advantage=148.54)
    featured = build_form_features(sample, "xg")

    rates = predict_goal_expectations(
        featured, config, elo_scale=835.56, home_advantage=148.54
    )

    for column in ("lambda_home", "lambda_away"):
        assert rates[column].between(LAMBDA_MIN, LAMBDA_MAX).all()


def test_prediction_requires_prebuilt_form_columns(sample: pd.DataFrame) -> None:
    config = fit_goal_expectation(sample, "xg", elo_scale=835.56, home_advantage=148.54)

    with pytest.raises(ValueError, match="form features missing"):
        predict_goal_expectations(sample, config, elo_scale=835.56, home_advantage=148.54)


def test_home_advantage_lifts_the_home_rate(sample: pd.DataFrame) -> None:
    config = fit_goal_expectation(sample, "none", elo_scale=835.56, home_advantage=148.54)
    featured = build_form_features(sample, "none")

    with_edge = predict_goal_expectations(
        featured, config, elo_scale=835.56, home_advantage=148.54
    )
    neutral = predict_goal_expectations(
        featured.assign(is_neutral=True), config, elo_scale=835.56, home_advantage=148.54
    )

    assert (with_edge["lambda_home"] >= neutral["lambda_home"] - 1e-12).all()


def test_elo_z_is_zero_for_an_even_neutral_match() -> None:
    even = frame([match("m1", 1, "A", "B", 1, 1)]).assign(is_neutral=True)

    assert elo_z(even, elo_scale=835.56, home_advantage=148.54)[0] == pytest.approx(0.0)


def test_config_rejects_form_on_the_elo_only_arm() -> None:
    with pytest.raises(ValueError, match="Elo-only arm"):
        GoalExpectationConfig(source="none", mu=0.3, elo_slope=0.5, attack=0.2).validate()
