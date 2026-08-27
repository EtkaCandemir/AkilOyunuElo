from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.run_ao_first_seed_boost_backtest import (
    BASELINE_KEY,
    BOOST_BLIND,
    BOOST_DOMESTIC_FORM,
    OUTPUT_ROOT,
    EVALUATION_SEASONS,
    _weighted_season_spearman_difference,
    build_domestic_form_snapshots,
    nested_selection,
)


def test_domestic_snapshot_closes_strictly_before_first_european_kickoff() -> None:
    domestic = pd.DataFrame(
        [
            {
                "source_event_id": "before",
                "sportsdb_league_id": "L1",
                "ao_season": "2019/20",
                "kickoff_utc": "2020-07-01T12:00:00Z",
                "home_source_team_id": "A",
                "away_source_team_id": "B",
                "home_goals": 4,
                "away_goals": 0,
            },
            {
                "source_event_id": "after",
                "sportsdb_league_id": "L1",
                "ao_season": "2020/21",
                "kickoff_utc": "2020-09-01T12:00:00Z",
                "home_source_team_id": "A",
                "away_source_team_id": "B",
                "home_goals": 0,
                "away_goals": 5,
            },
        ]
    )
    bridge = pd.DataFrame(
        [
            {"source_team_id": "A", "ao_club_id": "AO-A", "identity_ambiguous": False},
            {"source_team_id": "B", "ao_club_id": "AO-B", "identity_ambiguous": False},
        ]
    )
    identity = pd.DataFrame(
        [
            {"season": "2020/21", "local_team_id": 1, "club_id": "AO-A"},
            {"season": "2020/21", "local_team_id": 2, "club_id": "AO-B"},
        ]
    )
    european = pd.DataFrame(
        [{"season": "2020/21", "kickoff_utc": "2020-08-01T12:00:00Z"}]
    )

    result = build_domestic_form_snapshots(domestic, bridge, identity, european)

    assert set(result["team_id"]) == {1, 2}
    assert result["domestic_form_effective_matches"].eq(1.0).all()
    assert result.loc[result["team_id"].eq(1), "domestic_form_attack"].iloc[0] > 0


def selection_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    seeds = []
    targets = []
    for season_index, season in enumerate(("2018/19", "2019/20", "2020/21")):
        for team_id in range(1, 9):
            rating = 900.0 + 20.0 * team_id
            seeds.append(
                {
                    "season": season,
                    "team_id": team_id,
                    "club_id": f"AO-{team_id}",
                    "team_name": f"Team {team_id}",
                    "country_code": "TST",
                    "competition": "UEL",
                    "current_direct_percentile": 0.60,
                    "domestic_prior": 1200.0,
                    "european_prior": 800.0,
                    "effective_european_exposure": 0.50,
                    "domestic_prior_adjustment": 0.0,
                    "adjusted_domestic_prior": 1200.0,
                    "adjusted_ao_first_elo": rating,
                    "domestic_form_covered": True,
                    "domestic_form_score": float(team_id),
                    "domestic_form_percentile": team_id / 8,
                }
            )
            targets.append(
                {
                    "season": season,
                    "team_id": team_id,
                    "matches": 4,
                    "schedule_adjusted_score": rating / 2000 + season_index * 0.001,
                }
            )
    return pd.DataFrame(seeds), pd.DataFrame(targets)


def test_nested_selection_never_reads_the_test_season_target() -> None:
    seeds, target = selection_fixture()
    folds = ((("2018/19", "2019/20"), "2020/21"),)

    _, before, _ = nested_selection(seeds, target, folds)
    changed = target.copy()
    mask = changed["season"].eq("2020/21")
    changed.loc[mask, "schedule_adjusted_score"] = changed.loc[
        mask, "schedule_adjusted_score"
    ].iloc[::-1].to_numpy()
    _, after, _ = nested_selection(seeds, changed, folds)

    assert before["selected_candidate_key"].tolist() == after[
        "selected_candidate_key"
    ].tolist()


def test_nested_output_always_keeps_the_blind_control() -> None:
    seeds, target = selection_fixture()
    folds = ((("2018/19", "2019/20"), "2020/21"),)

    _, selections, audit = nested_selection(seeds, target, folds)

    assert set(selections["arm"]) == {BOOST_BLIND, BOOST_DOMESTIC_FORM}
    assert set(audit["arm"]) == {BASELINE_KEY, BOOST_BLIND, BOOST_DOMESTIC_FORM}


@pytest.mark.skipif(
    not (OUTPUT_ROOT / "selected_candidate.json").is_file(),
    reason="seed boost backtest has not been run",
)
def test_backtest_output_keeps_production_contract_unchanged() -> None:
    manifest = json.loads((OUTPUT_ROOT / "selected_candidate.json").read_text())

    assert manifest["production_activated"] is False
    assert len(manifest["production_contract_sha256"]) == 64
    assert set(manifest["production_contract_sha256"]) <= set("0123456789abcdef")
    assert manifest["selected_arm"] != BOOST_BLIND
    assert "2026/27" not in EVALUATION_SEASONS


def test_ranking_uncertainty_uses_the_same_season_weighted_definition() -> None:
    frame = pd.DataFrame(
        [
            {
                "season": season,
                "adjusted_ao_first_elo": baseline,
                "candidate_ao_first_elo": candidate,
                "schedule_adjusted_score": actual,
            }
            for season, baseline, candidate, actual in (
                ("2020/21", 1000.0, 1000.0, 0.1),
                ("2020/21", 1100.0, 1200.0, 0.3),
                ("2020/21", 1200.0, 1100.0, 0.2),
                ("2021/22", 900.0, 900.0, 0.1),
                ("2021/22", 1000.0, 1000.0, 0.2),
                ("2021/22", 1100.0, 1100.0, 0.3),
                ("2021/22", 1200.0, 1200.0, 0.4),
            )
        ]
    )

    result = _weighted_season_spearman_difference(frame)

    season_one = 0.5  # candidate rho=1, baseline rho=0.5
    season_two = 0.0
    assert result == pytest.approx((3 * season_one + 4 * season_two) / 7)
