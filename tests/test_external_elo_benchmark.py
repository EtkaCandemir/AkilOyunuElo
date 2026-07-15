from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_external_elo_benchmark import (  # noqa: E402
    attach_exact_dates,
    canonical_round,
    lookup_snapshots,
    normalize_country_code,
    resolve_teams_from_fixture_graph,
    season_end_year,
)


def test_season_and_country_contracts() -> None:
    assert season_end_year("2018/19") == 2019
    assert season_end_year("2025/26") == 2026
    assert normalize_country_code("ROM") == "ROU"
    assert normalize_country_code("BLS") == "BLR"
    assert normalize_country_code("ENG") == "ENG"


def test_round_aliases_disambiguate_same_fixture() -> None:
    assert canonical_round("League Stage") == "league"
    assert canonical_round("League phase") == "league"
    assert canonical_round("Knockout round play-offs") == "knockout_playoff"
    assert canonical_round("Round of 16") == "round_of_16"


def test_snapshot_lookup_is_strictly_pre_match() -> None:
    snapshots = pd.DataFrame(
        {
            "snapshot_date": pd.to_datetime(["2024-01-01", "2024-01-15"]),
            "country_code": ["ENG", "ENG"],
            "normalized_name": ["arsenal", "arsenal"],
            "elo": [1800.0, 1810.0],
        }
    )
    looked_up = lookup_snapshots(
        pd.Series(pd.to_datetime(["2024-01-15", "2024-01-16"])),
        pd.Series(["ENG", "ENG"]),
        pd.Series(["arsenal", "arsenal"]),
        snapshots,
    )

    assert looked_up["elo"].tolist() == [1800.0, 1810.0]
    assert looked_up["snapshot_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-01",
        "2024-01-15",
    ]


def test_fixture_graph_resolves_club_from_verified_opponent() -> None:
    events = pd.DataFrame(
        [
            {
                "season": "2024/25",
                "competition": "UCL",
                "home_team_id": 1,
                "away_team_id": 2,
                "home_team_name": "Galatasaray",
                "away_team_name": "Liverpool",
                "home_goals": 2,
                "away_goals": 0,
            }
        ]
    )
    official = pd.DataFrame(
        [
            {
                "season": "2024/25",
                "competition": "UCL",
                "uefa_home_team_id": "101",
                "uefa_away_team_id": "202",
                "uefa_home_country_code": "ENG",
                "uefa_away_country_code": "TUR",
                "uefa_home_goals_total": 2,
                "uefa_away_goals_total": 0,
                "uefa_home_goals_regular": 2,
                "uefa_away_goals_regular": 0,
                "uefa_home_team_name": "Arsenal",
                "uefa_away_team_name": "Galatasaray",
                "uefa_home_aliases": ("arsenal",),
                "uefa_away_aliases": ("galatasaray",),
            }
        ]
    )
    team_map = pd.DataFrame(
        [
            {
                "season": "2024/25",
                "local_team_id": 1,
                "uefa_country_code": "ENG",
                "uefa_team_id": "101",
                "uefa_team_name": "Arsenal",
                "resolution_method": "exact_name_country",
                "similarity": 1.0,
                "runner_up_similarity": None,
            },
            {
                "season": "2024/25",
                "local_team_id": 2,
                "uefa_country_code": "TUR",
                "uefa_team_id": None,
                "uefa_team_name": None,
                "resolution_method": "unresolved_name",
                "similarity": 0.5,
                "runner_up_similarity": 0.4,
            },
        ]
    )

    resolved = resolve_teams_from_fixture_graph(events, official, team_map)
    galatasaray = resolved.loc[resolved["local_team_id"].eq(2)].iloc[0]

    assert galatasaray["uefa_team_id"] == "202"
    assert galatasaray["resolution_method"] == "fixture_graph_unique"


def test_exact_date_join_uses_round_before_score_for_repeated_fixture() -> None:
    events = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "season": "2025/26",
                "competition": "UCL",
                "round": "League Stage",
                "home_team_id": 1,
                "away_team_id": 2,
                "home_team_name": "Galatasaray",
                "away_team_name": "Liverpool",
                "home_goals": 1,
                "away_goals": 0,
            },
            {
                "match_id": "m2",
                "season": "2025/26",
                "competition": "UCL",
                "round": "Round of 16",
                "home_team_id": 1,
                "away_team_id": 2,
                "home_goals": 1,
                "away_goals": 0,
            },
        ]
    )
    official = pd.DataFrame(
        [
            _official_match("u1", "League phase", "2025-09-30T19:00:00Z"),
            _official_match("u2", "Round of 16", "2026-03-10T20:00:00Z"),
        ]
    )
    team_map = pd.DataFrame(
        [
            {"season": "2025/26", "local_team_id": 1, "uefa_team_id": "101"},
            {"season": "2025/26", "local_team_id": 2, "uefa_team_id": "202"},
        ]
    )

    dated, audit = attach_exact_dates(events, official, team_map)

    assert dated["uefa_match_id"].tolist() == ["u1", "u2"]
    assert audit["match_status"].tolist() == ["MATCHED", "MATCHED"]


def _official_match(match_id: str, round_name: str, kickoff: str) -> dict[str, object]:
    return {
        "season": "2025/26",
        "competition": "UCL",
        "uefa_match_id": match_id,
        "kickoff_date": kickoff[:10],
        "kickoff_utc": kickoff,
        "uefa_round": round_name,
        "uefa_home_team_id": "101",
        "uefa_away_team_id": "202",
        "uefa_home_goals_total": 1,
        "uefa_away_goals_total": 0,
        "uefa_home_goals_regular": 1,
        "uefa_away_goals_regular": 0,
    }
