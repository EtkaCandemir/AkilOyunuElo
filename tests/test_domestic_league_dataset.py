from __future__ import annotations

import pandas as pd
import pytest

from ao_elo.domestic_league_dataset import (
    DomesticLeagueSpec,
    DomesticFixtureConflictError,
    ao_season_for_provider_event,
    assess_league_season,
    attach_domestic_club_ids,
    build_domestic_team_bridge,
    normalize_schedule,
    reconcile_domestic_fixture_observations,
    table_expected_matches,
    validate_domestic_dataset,
)


def test_identical_source_observations_are_audited_not_silently_dropped() -> None:
    payload = schedule_payload()
    payload["schedule"].append({**payload["schedule"][0], "idEvent": "12", "strTimestamp": "2020-08-01T17:00:00+03:00"})
    with pytest.raises(ValueError, match="duplicate domestic fixture"):
        normalize_schedule(payload, SPEC, "2020-2021")
    audit = []
    result = normalize_schedule(payload, SPEC, "2020-2021", reconciliation_audit=audit)
    assert len(result) == 2
    assert [(row["source_event_id"], row["action"]) for row in audit] == [("10", "KEEP"), ("12", "REMOVE_OBSERVATION")]
    payload["schedule"][-1]["intAwayScore"] = "3"
    with pytest.raises(DomesticFixtureConflictError, match="Unresolved"):
        normalize_schedule(payload, SPEC, "2020-2021", reconciliation_audit=[])


@pytest.mark.parametrize("league,date,home,away,keep,reject,score,bad_score", [
    ("4336", "2021-02-15T15:15:00Z", "135709", "133753", "1068057", "1091502", (2, 4), (2, 3)),
    ("4649", "2020-10-17T10:00:00Z", "138041", "138038", "1039304", "1039365", (2, 2), (0, 0)),
    ("4339", "2021-01-31T13:00:00Z", "135676", "133806", "1049070", "1084686", (3, 1), (3, 2)),
])
def test_official_score_resolution_checks_exact_evidence(league, date, home, away, keep, reject, score, bad_score) -> None:
    frame = pd.DataFrame([{"source_event_id": event, "sportsdb_league_id": league, "kickoff_utc": date, "home_source_team_id": home, "away_source_team_id": away, "home_goals": goals[0], "away_goals": goals[1]} for event, goals in [(reject, bad_score), (keep, score)]])
    audit = []
    result = reconcile_domestic_fixture_observations(frame, audit)
    assert result.source_event_id.tolist() == [keep]
    assert tuple(result[["home_goals", "away_goals"]].iloc[0]) == score
    assert all(row["source_url"].startswith("https://") for row in audit)
    frame.loc[0, "away_goals"] = 99
    with pytest.raises(DomesticFixtureConflictError, match="Unresolved"):
        reconcile_domestic_fixture_observations(frame, [])


SPEC = DomesticLeagueSpec("ENG", "4328", "English Premier League")


def schedule_payload() -> dict:
    return {
        "schedule": [
            {
                "idEvent": "10",
                "idLeague": "4328",
                "idHomeTeam": "100",
                "idAwayTeam": "200",
                "strHomeTeam": "Alpha FC",
                "strAwayTeam": "Beta FC",
                "intHomeScore": "2",
                "intAwayScore": "1",
                "strTimestamp": "2020-08-01T14:00:00+00:00",
                "strStatus": "Match Finished",
                "intRound": "1",
            },
            {
                "idEvent": "11",
                "idLeague": "4328",
                "idHomeTeam": "300",
                "idAwayTeam": "400",
                "strHomeTeam": "Gamma FC",
                "strAwayTeam": "Delta FC",
                "intHomeScore": "0",
                "intAwayScore": "0",
                "strTimestamp": "2020-08-02T14:00:00+00:00",
                "strStatus": "Match Finished",
                "intRound": "1",
            },
        ]
    }


def registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"club_id": "AO-UEFA-1", "canonical_name": "Alpha", "country_code": "ENG", "observed_names": "Alpha FC"},
            {"club_id": "AO-UEFA-2", "canonical_name": "Beta", "country_code": "ENG", "observed_names": "Beta FC"},
        ]
    )


def test_schedule_normalization_and_quality_gate() -> None:
    matches = normalize_schedule(schedule_payload(), SPEC, "2020-2021")
    assert list(matches["match_id"]) == ["TSD-4328-10", "TSD-4328-11"]
    assert list(matches["ao_season"]) == ["2020/21", "2020/21"]
    assessment = assess_league_season(matches, spec=SPEC, provider_season="2020-2021", expected_matches=2)
    assert assessment["quality_status"] == "ACCEPTED"
    assert assessment["coverage_rate"] == pytest.approx(1.0)


def test_final_table_expected_matches_requires_even_played_total() -> None:
    assert table_expected_matches({"table": [{"intPlayed": "2"}, {"intPlayed": "2"}]}) == 2
    assert table_expected_matches({"table": [{"intPlayed": "1"}, {"intPlayed": "2"}]}) is None


def test_bridge_maps_only_existing_ao_clubs_and_dataset_stays_valid() -> None:
    matches = normalize_schedule(schedule_payload(), SPEC, "2020-2021")
    bridge = build_domestic_team_bridge(matches, registry())
    attached = attach_domestic_club_ids(matches, bridge)
    assert attached.loc[0, "home_ao_club_id"] == "AO-UEFA-1"
    assert attached.loc[0, "away_ao_club_id"] == "AO-UEFA-2"
    assert pd.isna(attached.loc[1, "home_ao_club_id"])
    quality = pd.DataFrame([
        {"country_code": "ENG", "provider_season": "2020-2021", "quality_status": "ACCEPTED"}
    ])
    validate_domestic_dataset(attached, quality, bridge)

    ukraine = pd.DataFrame(
        [
            {
                "country_code": "UKR",
                "home_source_team_id": "133944",
                "away_source_team_id": "134422",
                "home_team_name": "Dynamo Kiev",
                "away_team_name": "Zorya",
            }
        ]
    )
    ukraine_registry = pd.DataFrame(
        [
            {"club_id": "AO-UEFA-52723", "canonical_name": "Dynamo Kyiv", "country_code": "UKR"},
            {"club_id": "AO-UEFA-65130", "canonical_name": "Zorya Luhansk", "country_code": "UKR"},
        ]
    )
    verified = build_domestic_team_bridge(ukraine, ukraine_registry)
    assert verified["ao_club_id"].tolist() == ["AO-UEFA-52723", "AO-UEFA-65130"]
    assert verified["identity_method"].eq("VERIFIED_PROVIDER_ALIAS").all()


def test_duplicate_source_event_is_rejected() -> None:
    payload = schedule_payload()
    payload["schedule"][1]["idEvent"] = "10"
    with pytest.raises(ValueError, match="duplicate source event"):
        normalize_schedule(payload, SPEC, "2020-2021")


def test_provider_season_preserves_covid_delayed_competition_year() -> None:
    kickoff = pd.Timestamp("2020-07-01T18:30:00Z")
    assert ao_season_for_provider_event(SPEC, "2019-2020", kickoff) == "2019/20"
