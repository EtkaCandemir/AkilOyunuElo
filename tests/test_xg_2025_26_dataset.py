from __future__ import annotations

import math

import pandas as pd
import pytest

from ao_elo.xg_dataset import (
    MASTER_COLUMNS,
    attach_secondary_xg,
    build_coverage_summary,
    flatten_fotmob_date_payload,
    parse_fotmob_xg,
    resolve_fotmob_identity,
    validate_master_dataset,
)


def event_row(*, penalties: bool = False) -> pd.Series:
    return pd.Series(
        {
            "match_id": "202526-0001",
            "season": "2025/26",
            "competition": "UCL",
            "round": "League Stage",
            "round_sequence": 4,
            "matchday": 1,
            "tie_id": "",
            "leg_number": math.nan,
            "is_tie_decider": False,
            "is_knockout": False,
            "is_neutral": False,
            "kickoff_date": "2025-09-16",
            "kickoff_utc": pd.Timestamp("2025-09-16T19:00:00Z"),
            "event_order": 1,
            "home_team_id": 1,
            "away_team_id": 2,
            "home_team_name": "AO Istanbul",
            "away_team_name": "Data United",
            "uefa_match_id": "9001",
            "uefa_home_team_id": "101",
            "uefa_away_team_id": "102",
            "home_goals": 2 if not penalties else 1,
            "away_goals": 1,
            "actual_home_score": 1.0 if not penalties else 0.5,
            "goal_difference": 1 if not penalties else 0,
            "result_basis": "displayed_score_excluding_shootout",
            "decided_on_penalties": penalties,
            "home_penalty_goals": 4 if penalties else math.nan,
            "away_penalty_goals": 3 if penalties else math.nan,
            "advanced_team_id": 1 if penalties else math.nan,
            "score_verified": True,
            "chronology_verified": True,
        }
    )


def date_payload() -> dict[str, object]:
    return {
        "leagues": [
            {
                "name": "Champions League",
                "matches": [
                    {
                        "id": 5001,
                        "home": {"id": 11, "name": "AO Istanbul", "score": 2},
                        "away": {"id": 12, "name": "Data United", "score": 1},
                        "status": {"utcTime": "2025-09-16T19:00:00.000Z"},
                    }
                ],
            }
        ]
    }


def detail_payload(*, penalties: bool = False, include_shootout: bool = False) -> dict[str, object]:
    home_score = 1 if penalties else 2
    shots = [
        {"teamId": 11, "expectedGoals": 0.8, "period": "FirstHalf"},
        {"teamId": 11, "expectedGoals": 0.4, "period": "SecondHalf"},
        {"teamId": 12, "expectedGoals": 0.7, "period": "SecondHalf"},
    ]
    if include_shootout:
        shots.append(
            {"teamId": 11, "expectedGoals": 0.7884, "period": "PenaltyShootout"}
        )
    return {
        "general": {
            "matchId": "5001",
            "matchTimeUTCDate": "2025-09-16T19:00:00.000Z",
            "homeTeam": {"id": 11, "name": "AO Istanbul"},
            "awayTeam": {"id": 12, "name": "Data United"},
        },
        "header": {
            "teams": [
                {"id": 11, "name": "AO Istanbul", "score": home_score},
                {"id": 12, "name": "Data United", "score": 1},
            ],
            "status": {
                "utcTime": "2025-09-16T19:00:00.000Z",
                "halfs": {"firstExtraHalfStarted": "", "secondExtraHalfStarted": ""},
            },
        },
        "content": {
            "stats": {
                "Periods": {
                    "All": {
                        "stats": [
                            {
                                "key": "top_stats",
                                "stats": [
                                    {
                                        "key": "expected_goals",
                                        "stats": ["1.20", "0.70"],
                                    }
                                ],
                            }
                        ]
                    }
                }
            },
            "shotmap": {"shots": shots},
        },
    }


def candidate() -> object:
    rows = pd.DataFrame(flatten_fotmob_date_payload(date_payload()))
    value, audit = resolve_fotmob_identity(event_row(), rows)
    assert audit["identity_accepted"] is True
    assert value is not None
    return value


def test_date_payload_and_identity_are_resolved_without_source_ids() -> None:
    rows = pd.DataFrame(flatten_fotmob_date_payload(date_payload()))
    result, audit = resolve_fotmob_identity(event_row(), rows)
    assert result is not None
    assert result.source_match_id == "5001"
    assert result.pair_similarity == pytest.approx(1.0)
    assert audit["identity_resolution"] == "utc_time_competition_score_high_confidence_name"


def test_regular_fotmob_xg_is_accepted_and_recomputed() -> None:
    values, audit = parse_fotmob_xg(
        event_row(),
        candidate(),
        detail_payload(),
        fetch_status="OK",
        snapshot_utc="2026-08-04T00:00:00+00:00",
    )
    assert audit["xg_disposition"] == "ACCEPTED"
    assert values["xg_covered"] is True
    assert values["xg_home"] == pytest.approx(1.2)
    assert values["xg_away"] == pytest.approx(0.7)
    assert values["xg_total"] == pytest.approx(1.9)
    assert values["xg_difference"] == pytest.approx(0.5)
    assert values["home_goals_minus_xg"] == pytest.approx(0.8)
    assert values["away_goals_minus_xg"] == pytest.approx(0.3)


def test_shootout_requires_auditable_exclusion() -> None:
    values, _ = parse_fotmob_xg(
        event_row(penalties=True),
        candidate(),
        detail_payload(penalties=True, include_shootout=False),
        fetch_status="OK",
        snapshot_utc="2026-08-04T00:00:00+00:00",
    )
    assert values["xg_covered"] is False
    assert values["xg_missing_reason"] == "PENALTY_SCOPE_UNVERIFIED"

    accepted, _ = parse_fotmob_xg(
        event_row(penalties=True),
        candidate(),
        detail_payload(penalties=True, include_shootout=True),
        fetch_status="OK",
        snapshot_utc="2026-08-04T00:00:00+00:00",
    )
    assert accepted["xg_covered"] is True
    assert accepted["xg_shootout_excluded"] is True


def test_secondary_xg_never_fills_primary_columns() -> None:
    master = pd.DataFrame(
        {
            "match_id": ["m1", "m2"],
            "xg_home": [math.nan, 1.5],
            "xg_away": [math.nan, 0.8],
        }
    )
    secondary = pd.DataFrame(
        {
            "match_id": ["m1"],
            "season": ["2025/26"],
            "xg_home": [2.2],
            "xg_away": [1.1],
            "xg_type": ["coarse_zone_derived_xg"],
            "provider": ["secondary"],
            "xg_covered": [True],
        }
    )
    result = attach_secondary_xg(master, secondary)
    assert math.isnan(result.loc[0, "xg_home"])
    assert result.loc[0, "secondary_xg_home"] == pytest.approx(2.2)
    assert bool(result.loc[0, "secondary_xg_covered"]) is True


def test_committed_master_dataset_contract() -> None:
    path = "data/xg_2025_26/uefa_2025_26_matches_with_xg.csv"
    master = pd.read_csv(path, keep_default_na=False, na_values=[""])
    master["xg_covered"] = master["xg_covered"].astype(bool)
    master["xg_analysis_eligible"] = master["xg_analysis_eligible"].astype(bool)
    master["secondary_xg_covered"] = master["secondary_xg_covered"].astype(bool)
    master["score_verified"] = master["score_verified"].astype(bool)
    master["chronology_verified"] = master["chronology_verified"].astype(bool)
    assert tuple(master.columns) == MASTER_COLUMNS
    validate_master_dataset(master)
    coverage = build_coverage_summary(master)
    total = coverage.loc[
        coverage["competition"].eq("ALL") & coverage["round"].eq("ALL")
    ].iloc[0]
    assert int(total["matches"]) == 961

