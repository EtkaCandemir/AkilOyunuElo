from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ao_elo.domestic_poisson import (
    DomesticPoissonConfig,
    DynamicDomesticPoisson,
    EuropeanPoissonTransferConfig,
    build_domestic_poisson_feature_store,
    domestic_candidate_grid,
    predict_european_poisson_transfer,
)


def test_candidate_grid_contains_54_unique_configs() -> None:
    candidates = domestic_candidate_grid()
    assert len(candidates) == 54
    assert len({candidate.key for candidate in candidates}) == 54


def test_same_kickoff_batch_uses_one_pre_match_snapshot() -> None:
    engine = DynamicDomesticPoisson(_config(venue=True))
    result = engine.update_batch(
        _matches(
            ("m1", "2020-01-01T12:00:00Z", "A", "B", 4, 0),
            ("m2", "2020-01-01T12:00:00Z", "C", "D", 0, 4),
        )
    )
    assert result["lambda_home"].nunique() == 1
    assert result["lambda_away"].nunique() == 1
    engine.validate_state()


def test_attack_and_defence_signals_recover_known_direction() -> None:
    engine = DynamicDomesticPoisson(_config(venue=False))
    for index in range(30):
        if index % 2 == 0:
            match = (f"m{index}", f"2020-02-{index % 27 + 1:02d}T12:00:00Z", "A", "B", 3, 0)
        else:
            match = (f"m{index}", f"2020-03-{index % 27 + 1:02d}T12:00:00Z", "B", "A", 0, 2)
        engine.update_batch(_matches(match))
    a = engine.snapshot("L", "2020/21", "A")
    b = engine.snapshot("L", "2020/21", "B")
    assert a.attack > b.attack
    assert a.defence > b.defence
    assert 0.0 < a.reliability < 1.0
    engine.validate_state()


def test_venue_disabled_ignores_team_venue_parameters() -> None:
    engine = DynamicDomesticPoisson(_config(venue=False))
    engine.ensure_season("L", "2020/21")
    league = engine.leagues["L"]
    league.teams["A"] = league.teams.setdefault("A", _team_state())
    league.teams["B"] = league.teams.setdefault("B", _team_state())
    league.teams["A"].home_strength = 0.7
    league.teams["B"].away_strength = -0.7
    before = engine.predict_match("L", "2020/21", "A", "B")
    league.teams["A"].home_strength = -0.7
    league.teams["B"].away_strength = 0.7
    assert engine.predict_match("L", "2020/21", "A", "B") == before


def test_season_carry_shrinks_state_and_effective_matches() -> None:
    engine = DynamicDomesticPoisson(_config(venue=True, carry=0.5))
    engine.update_batch(
        _matches(("m1", "2020-01-01T12:00:00Z", "A", "B", 3, 0))
    )
    before = engine.leagues["L"].teams["A"].effective_matches
    engine.ensure_season("L", "2021/22")
    assert engine.leagues["L"].teams["A"].effective_matches == before * 0.5


def test_unmapped_domestic_team_trains_state_but_is_not_exported() -> None:
    domestic = _matches(
        ("m1", "2020-01-01T12:00:00Z", "A", "B", 3, 0),
    )
    european = pd.DataFrame(
        [
            {
                "match_id": "e1",
                "season": "2020/21",
                "kickoff_utc": "2020-01-02T12:00:00Z",
                "home_club_id": "AO-A",
                "away_club_id": "AO-MISSING",
            }
        ]
    )
    bridge = pd.DataFrame(
        [
            {
                "source_team_id": "A",
                "ao_club_id": "AO-A",
                "identity_ambiguous": False,
            },
            {
                "source_team_id": "B",
                "ao_club_id": np.nan,
                "identity_ambiguous": False,
            },
        ]
    )
    result = build_domestic_poisson_feature_store(
        domestic, european, bridge, _config(venue=False)
    )
    assert result.loc[0, "domestic_poisson_coverage"] == "ONE"
    assert bool(result.loc[0, "home_domestic_poisson_covered"])
    assert not bool(result.loc[0, "away_domestic_poisson_covered"])


def test_domestic_result_at_same_kickoff_is_not_visible() -> None:
    domestic = _matches(
        ("m1", "2020-01-01T12:00:00Z", "A", "B", 3, 0),
    )
    european = pd.DataFrame(
        [
            {
                "match_id": "e1",
                "season": "2020/21",
                "kickoff_utc": "2020-01-01T12:00:00Z",
                "home_club_id": "AO-A",
                "away_club_id": "AO-B",
            }
        ]
    )
    bridge = pd.DataFrame(
        [
            {"source_team_id": "A", "ao_club_id": "AO-A", "identity_ambiguous": False},
            {"source_team_id": "B", "ao_club_id": "AO-B", "identity_ambiguous": False},
        ]
    )
    result = build_domestic_poisson_feature_store(
        domestic, european, bridge, _config(venue=False)
    )
    assert result.loc[0, "domestic_poisson_coverage"] == "NONE"


def test_dynamic_state_is_deterministic() -> None:
    matches = _matches(
        ("m1", "2020-01-01T12:00:00Z", "A", "B", 3, 0),
        ("m2", "2020-01-01T12:00:00Z", "C", "D", 1, 2),
    )
    first = DynamicDomesticPoisson(_config(venue=True))
    second = DynamicDomesticPoisson(_config(venue=True))
    pd.testing.assert_frame_equal(first.update_batch(matches), second.update_batch(matches))
    assert first.snapshot("L", "2020/21", "A") == second.snapshot("L", "2020/21", "A")


def test_no_history_prediction_falls_back_exactly_to_ao() -> None:
    frame = _transfer_frame("NONE")
    config = EuropeanPoissonTransferConfig(
        mu=0.2,
        elo_slope=0.7,
        attack_coefficient=0.5,
        defence_coefficient=0.5,
        venue_coefficient=0.0,
        l2_strength=1.0,
    )
    predicted = predict_european_poisson_transfer(frame, config)
    np.testing.assert_allclose(
        predicted.loc[0, ["home_probability", "draw_probability", "away_probability"]].to_numpy(float),
        [0.50, 0.25, 0.25],
        atol=1e-12,
    )
    assert bool(predicted.loc[0, "ao_fallback"])


def test_transfer_attack_and_defence_change_expected_goals_monotonically() -> None:
    weak = _transfer_frame("BOTH")
    strong = weak.copy()
    strong["home_domestic_poisson_attack"] = 1.0
    strong["away_domestic_poisson_defence"] = -1.0
    config = EuropeanPoissonTransferConfig(
        mu=0.2,
        elo_slope=0.7,
        attack_coefficient=0.4,
        defence_coefficient=0.4,
        venue_coefficient=0.0,
        l2_strength=1.0,
    )
    weak_prediction = predict_european_poisson_transfer(weak, config)
    strong_prediction = predict_european_poisson_transfer(strong, config)
    assert strong_prediction.loc[0, "lambda_home"] > weak_prediction.loc[0, "lambda_home"]


def _config(*, venue: bool, carry: float = 0.75) -> DomesticPoissonConfig:
    return DomesticPoissonConfig(0.05, carry, 20.0, venue)


def _team_state():
    from ao_elo.domestic_poisson import DomesticTeamState

    return DomesticTeamState()


def _matches(*rows: tuple[str, str, str, str, int, int]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_event_id": match_id,
                "sportsdb_league_id": "L",
                "ao_season": "2020/21",
                "kickoff_utc": kickoff,
                "home_source_team_id": home,
                "away_source_team_id": away,
                "home_goals": home_goals,
                "away_goals": away_goals,
            }
            for match_id, kickoff, home, away, home_goals, away_goals in rows
        ]
    )


def _transfer_frame(coverage: str) -> pd.DataFrame:
    row: dict[str, object] = {
        "match_id": "e1",
        "home_goals": 1,
        "away_goals": 0,
        "actual_class": 0,
        "expected_home_score": 0.6,
        "ao_home_probability": 0.50,
        "ao_draw_probability": 0.25,
        "ao_away_probability": 0.25,
        "domestic_poisson_coverage": coverage,
        "domestic_poisson_venue_edge": 0.0,
        "domestic_poisson_venue_edge_raw": 0.0,
    }
    for side in ("home", "away"):
        for feature in ("attack", "defence"):
            row[f"{side}_domestic_poisson_{feature}"] = 0.0
            row[f"{side}_domestic_poisson_{feature}_raw_z"] = 0.0
    return pd.DataFrame([row])
