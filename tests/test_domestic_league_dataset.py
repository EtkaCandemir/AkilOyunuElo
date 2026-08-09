from __future__ import annotations

import pandas as pd
import pytest

from ao_elo.domestic_league_dataset import (
    DomesticLeagueSpec,
    ao_season_for_provider_event,
    assess_league_season,
    attach_domestic_club_ids,
    build_domestic_team_bridge,
    normalize_schedule,
    table_expected_matches,
    validate_domestic_dataset,
)


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


def test_duplicate_source_event_is_rejected() -> None:
    payload = schedule_payload()
    payload["schedule"][1]["idEvent"] = "10"
    with pytest.raises(ValueError, match="duplicate source event"):
        normalize_schedule(payload, SPEC, "2020-2021")


def test_provider_season_preserves_covid_delayed_competition_year() -> None:
    kickoff = pd.Timestamp("2020-07-01T18:30:00Z")
    assert ao_season_for_provider_event(SPEC, "2019-2020", kickoff) == "2019/20"
