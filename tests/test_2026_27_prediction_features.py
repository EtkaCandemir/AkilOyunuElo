from __future__ import annotations

"""Guard the bridge that lets the Structural ML half run on 2026/27.

Without it the rolling European windows are truncated to a single season and
the service silently serves Current AO for every fixture, so the properties
tested here are the ones that make the served prediction real rather than a
quiet fallback.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_2026_27_prediction_features import (
    BASELINE_COLUMNS,
    SYNTHETIC_TEAM_ID_BASE,
    _integer_team_ids,
    _non_finite_model_features,
    build_features,
    load_completed_matches,
    load_domestic_matches,
    load_metadata,
    load_upcoming_fixtures,
    validation_gates,
    DEFAULT_DOMESTIC,
    DEFAULT_FIXTURES,
    DEFAULT_HISTORY,
    DEFAULT_IDENTITY,
)


SYNTHETIC_FIRST_LEG = "TEST-TWO-LEG-1"
SYNTHETIC_SECOND_LEG = "TEST-TWO-LEG-2"
REAL_INTER_FIXTURE = "UEFA-2049553"
INPUTS_PRESENT = (
    DEFAULT_HISTORY.exists()
    and DEFAULT_DOMESTIC.exists()
    and DEFAULT_FIXTURES.exists()
    and (ROOT / "output/season_2026_27_preproduction/playoff_pre_match_team_ratings.csv").exists()
)
requires_inputs = pytest.mark.skipif(
    not INPUTS_PRESENT, reason="bridge inputs are not built in this checkout"
)


def _two_leg_fixture_path(tmp_path: Path) -> Path:
    """Build a stable two-leg input without depending on mutable live fixtures."""

    fixtures = pd.read_csv(DEFAULT_FIXTURES)
    source = fixtures.loc[fixtures["match_id"].eq(REAL_INTER_FIXTURE)]
    assert len(source) == 1, "the synthetic tie needs two current league participants"

    first = source.iloc[0].copy()
    first["match_id"] = SYNTHETIC_FIRST_LEG
    first["kickoff_utc"] = "2026-09-08T19:00:00Z"
    first["round"] = "Qualifying Play-off Round"
    first["tie_id"] = "TEST-TWO-LEG-TIE"
    first["leg_number"] = 1
    first["is_knockout"] = True
    first["is_tie_decider"] = False
    first["is_single_match_tie"] = False
    first["stage"] = "QUALIFYING"

    second = first.copy()
    second["match_id"] = SYNTHETIC_SECOND_LEG
    second["kickoff_utc"] = "2026-09-15T19:00:00Z"
    second["leg_number"] = 2
    second["is_tie_decider"] = True
    for column in ("team_id", "team_name"):
        second[f"home_{column}"] = first[f"away_{column}"]
        second[f"away_{column}"] = first[f"home_{column}"]

    path = tmp_path / "two_leg_fixtures.csv"
    pd.DataFrame([first, second]).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# the synthetic team id is a carrier, never a feature
# ---------------------------------------------------------------------------


def test_numeric_team_ids_are_left_alone() -> None:
    frame = pd.DataFrame({"home_team_id": ["12", "7"], "away_team_id": ["3", "99"]})

    result = _integer_team_ids(frame)

    assert result["home_team_id"].tolist() == [12, 7]
    assert result["away_team_id"].tolist() == [3, 99]


def test_string_team_ids_get_stable_synthetic_integers() -> None:
    frame = pd.DataFrame(
        {
            "home_team_id": ["AO-UEFA-1", "AO-UEFA-2", "AO-UEFA-1"],
            "away_team_id": ["AO-UEFA-2", "AO-UEFA-1", "AO-UEFA-2"],
        }
    )

    result = _integer_team_ids(frame)

    assert result["home_team_id"].iloc[0] == result["home_team_id"].iloc[2]
    assert result["home_team_id"].iloc[0] == result["away_team_id"].iloc[1]
    assert result["home_team_id"].min() >= SYNTHETIC_TEAM_ID_BASE


def test_synthetic_ids_never_collide_with_real_ones() -> None:
    frame = pd.DataFrame(
        {"home_team_id": ["5", "AO-UEFA-X"], "away_team_id": ["6", "AO-UEFA-Y"]}
    )

    result = _integer_team_ids(frame)
    real = {5, 6}
    synthetic = set(result[["home_team_id", "away_team_id"]].to_numpy().ravel()) - real

    assert all(value >= SYNTHETIC_TEAM_ID_BASE for value in synthetic)


# ---------------------------------------------------------------------------
# the leakage property the whole design turns on
# ---------------------------------------------------------------------------


@requires_inputs
def test_two_legs_of_one_tie_build_in_separate_batches(tmp_path: Path) -> None:
    """The second leg must not see the first leg's placeholder result."""
    fixture_path = _two_leg_fixture_path(tmp_path)
    completed = load_completed_matches(DEFAULT_HISTORY, DEFAULT_IDENTITY)
    fixtures = load_upcoming_fixtures(
        fixture_path, [SYNTHETIC_FIRST_LEG, SYNTHETIC_SECOND_LEG]
    )
    domestic = load_domestic_matches(DEFAULT_DOMESTIC)

    features, batches = build_features(
        completed, load_metadata(fixture_path), fixtures, domestic
    )

    assert batches == 2, "each leg has its own kickoff, so each needs its own build"
    first = features.loc[features["match_id"].eq(SYNTHETIC_FIRST_LEG)].iloc[0]
    second = features.loc[features["match_id"].eq(SYNTHETIC_SECOND_LEG)].iloc[0]

    # The clubs swap venues in the second leg. If the first leg had entered
    # history, the second leg would count one more match.
    assert first["home_euro_matches_pre"] == second["away_euro_matches_pre"]
    assert first["away_euro_matches_pre"] == second["home_euro_matches_pre"]


@requires_inputs
def test_rolling_windows_reach_past_the_current_season(tmp_path: Path) -> None:
    fixture_path = _two_leg_fixture_path(tmp_path)
    completed = load_completed_matches(DEFAULT_HISTORY, DEFAULT_IDENTITY)
    fixtures = load_upcoming_fixtures(fixture_path, [SYNTHETIC_FIRST_LEG])
    domestic = load_domestic_matches(DEFAULT_DOMESTIC)

    features, _ = build_features(
        completed, load_metadata(fixture_path), fixtures, domestic
    )
    row = features.iloc[0]

    # 2026/27 alone would give at most a handful of qualifiers per club.
    assert row["home_euro_matches_pre"] > 20
    assert row["away_euro_matches_pre"] > 20


@requires_inputs
def test_completed_history_spans_every_season() -> None:
    completed = load_completed_matches(DEFAULT_HISTORY, DEFAULT_IDENTITY)

    seasons = set(completed["season"])

    assert {"2018/19", "2025/26", "2026/27"} <= seasons
    assert completed["match_id"].is_unique
    assert set(BASELINE_COLUMNS) <= set(completed.columns)


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------


@requires_inputs
def test_validation_gates_pass_for_a_single_tie(tmp_path: Path) -> None:
    fixture_path = _two_leg_fixture_path(tmp_path)
    completed = load_completed_matches(DEFAULT_HISTORY, DEFAULT_IDENTITY)
    fixtures = load_upcoming_fixtures(
        fixture_path, [SYNTHETIC_FIRST_LEG, SYNTHETIC_SECOND_LEG]
    )
    domestic = load_domestic_matches(DEFAULT_DOMESTIC)
    features, batches = build_features(
        completed, load_metadata(fixture_path), fixtures, domestic
    )

    gates = validation_gates(features, fixtures, completed, batches)

    assert gates["passed"].all()
    assert "one_batch_per_kickoff" in set(gates["gate"])


@requires_inputs
def test_imputed_rows_are_reported_not_failed() -> None:
    """A debutant club is a legitimate state, not a broken build."""
    completed = load_completed_matches(DEFAULT_HISTORY, DEFAULT_IDENTITY)
    fixtures = load_upcoming_fixtures(DEFAULT_FIXTURES, None)
    domestic = load_domestic_matches(DEFAULT_DOMESTIC)
    features, batches = build_features(completed, load_metadata(), fixtures, domestic)

    gates = validation_gates(features, fixtures, completed, batches)
    reported = gates.loc[gates["gate"].eq("rows_with_imputed_model_input")].iloc[0]

    assert bool(reported["passed"])
    assert reported["observed"] == len(_non_finite_model_features(features))


def test_unknown_fixture_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown fixture match_id"):
        load_upcoming_fixtures(DEFAULT_FIXTURES, ["NOT-A-FIXTURE"])
