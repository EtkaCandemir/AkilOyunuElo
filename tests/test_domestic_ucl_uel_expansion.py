from __future__ import annotations

import pandas as pd

from ao_elo.domestic_ucl_uel_expansion import (
    apply_verified_target_aliases,
    build_target_team_audit,
    canonicalize_candidate_domestic_state,
    merge_domestic_candidate,
    target_domestic_coverage_audit,
)


def test_target_universe_uses_direct_entries_and_ucl_uel_playoffs() -> None:
    teams = pd.DataFrame(
        [
            {"team_id": "A", "team_name": "Alpha", "country_code": "HUN", "domestic_league": "NB I"},
            {"team_id": "B", "team_name": "Beta", "country_code": "TUR", "domestic_league": "Super Lig"},
            {"team_id": "C", "team_name": "Gamma", "country_code": "LIT", "domestic_league": "A Lyga"},
        ]
    )
    context = pd.DataFrame(
        [
            {"team_id": "A", "competition": "UCL", "entry_round": "CL-LS"},
            {"team_id": "B", "competition": "UECL", "entry_round": "CO-LS"},
            {"team_id": "C", "competition": "UEL", "entry_round": "EL-Q3"},
        ]
    )
    fixtures = pd.DataFrame(
        [
            {"competition": "UEL", "home_team_id": "B", "away_team_id": "C"},
            {"competition": "UECL", "home_team_id": "A", "away_team_id": "C"},
        ]
    )
    result = build_target_team_audit(teams, context, fixtures)
    assert result["ao_club_id"].tolist() == ["A", "C", "B"]
    assert result.set_index("ao_club_id").loc["A", "selection_reason"] == "DIRECT_CL-LS"
    assert result.set_index("ao_club_id").loc["C", "selection_reason"] == "UPCOMING_UEL_PLAYOFF"
    assert result.set_index("ao_club_id").loc["B", "is_expansion_target_country"] == False


def test_target_coverage_requires_unique_identity_matches_and_seasons() -> None:
    targets = pd.DataFrame(
        [{"ao_club_id": "A", "team_name": "Alpha", "country_code": "HUN"}]
    )
    bridge = pd.DataFrame(
        [{"country_code": "HUN", "source_team_id": "1", "ao_club_id": "A", "identity_ambiguous": False}]
    )
    matches = pd.DataFrame(
        [
            {
                "country_code": "HUN", "ao_season": "2024/25", "home_source_team_id": "1", "away_source_team_id": "2",
            },
            {
                "country_code": "HUN", "ao_season": "2025/26", "home_source_team_id": "2", "away_source_team_id": "1",
            },
        ]
    )
    result = target_domestic_coverage_audit(
        targets, bridge, matches, minimum_matches=2, minimum_seasons=2
    )
    assert result.loc[0, "candidate_state_eligible"]
    assert result.loc[0, "coverage_status"] == "COVERED"


def test_target_coverage_allows_multiple_source_ids_for_one_verified_club() -> None:
    targets = pd.DataFrame(
        [{"ao_club_id": "A", "team_name": "Alpha", "country_code": "HUN"}]
    )
    bridge = pd.DataFrame(
        [
            {"country_code": "HUN", "source_team_id": "TSD:1", "ao_club_id": "A", "identity_ambiguous": False},
            {"country_code": "HUN", "source_team_id": "HF:1", "ao_club_id": "A", "identity_ambiguous": False},
        ]
    )
    matches = pd.DataFrame(
        [
            {"country_code": "HUN", "ao_season": "2024/25", "home_source_team_id": "TSD:1", "away_source_team_id": "2"},
            {"country_code": "HUN", "ao_season": "2025/26", "home_source_team_id": "HF:1", "away_source_team_id": "3"},
        ]
    )
    result = target_domestic_coverage_audit(
        targets, bridge, matches, minimum_matches=2, minimum_seasons=2
    )
    assert result.loc[0, "identity_ok"]
    assert result.loc[0, "candidate_state_eligible"]


def test_candidate_merge_rejects_duplicate_provider_event_within_league() -> None:
    base = pd.DataFrame(
        [{"match_id": "a", "source_event_id": "1", "sportsdb_league_id": "10", "country_code": "TUR", "ao_season": "2025/26", "kickoff_utc": "2025-08-01T12:00:00Z", "home_source_team_id": "1", "away_source_team_id": "2", "home_goals": 1, "away_goals": 0}]
    )
    expansion = base.assign(match_id="b")
    try:
        merge_domestic_candidate(base, expansion)
    except ValueError as error:
        assert "duplicate provider event" in str(error)
    else:
        raise AssertionError("Expected duplicate provider event validation")


def test_canonical_state_keys_join_selected_sources_for_one_club() -> None:
    matches = pd.DataFrame(
        [
            {"match_id": "p", "country_code": "HUN", "sportsdb_league_id": "4690", "kickoff_utc": "2024-01-01T12:00:00Z", "home_source_team_id": "TSD:1", "away_source_team_id": "TSD:2"},
            {"match_id": "s", "country_code": "HUN", "sportsdb_league_id": "HF:47", "kickoff_utc": "2025-01-01T12:00:00Z", "home_source_team_id": "HF:1", "away_source_team_id": "HF:2"},
        ]
    )
    bridge = pd.DataFrame(
        [
            {"country_code": "HUN", "source_team_id": "TSD:1", "ao_club_id": "A"},
            {"country_code": "HUN", "source_team_id": "HF:1", "ao_club_id": "A"},
        ]
    )
    result, canonical_bridge = canonicalize_candidate_domestic_state(
        matches, bridge, source_switch_countries=("HUN",)
    )
    assert result["sportsdb_league_id"].tolist() == ["AO-DOMESTIC:HUN", "AO-DOMESTIC:HUN"]
    assert result["home_source_team_id"].tolist() == ["AO:A", "AO:A"]
    assert canonical_bridge[canonical_bridge["ao_club_id"].eq("A")].shape[0] == 1


def test_verified_aliases_reject_paris_fc_false_positive() -> None:
    targets = pd.DataFrame(
        [{"ao_club_id": "AO-UEFA-52747", "country_code": "FRA", "team_name": "Paris Saint-Germain"}]
    )
    bridge = pd.DataFrame(
        [
            {
                "country_code": "FRA", "source_team_id": "1", "source_team_name": "Paris FC",
                "ao_club_id": "AO-UEFA-52747", "identity_method": "EXACT_NAME",
                "identity_ambiguous": False,
            },
            {
                "country_code": "FRA", "source_team_id": "2", "source_team_name": "Paris SG",
                "ao_club_id": pd.NA, "identity_method": "NO_AO_COUNTERPART",
                "identity_ambiguous": False,
            },
        ]
    )
    result, audit = apply_verified_target_aliases(bridge, targets)
    assert result.loc[result["source_team_id"].eq("1"), "ao_club_id"].isna().all()
    assert result.loc[result["source_team_id"].eq("2"), "ao_club_id"].item() == "AO-UEFA-52747"
    assert audit.loc[0, "override_status"] == "APPLIED"
