from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ao_elo.domestic_league_dataset import DomesticLeagueSpec

from ao_elo.domestic_ucl_uel_expansion import (
    apply_verified_target_aliases,
    assess_secondary_league_seasons,
    build_target_team_audit,
    canonicalize_candidate_domestic_state,
    merge_domestic_candidate,
    select_source_safe_seasons,
    normalize_secondary_fixtures,
    SECONDARY_LEAGUE_IDS,
    target_domestic_coverage_audit,
)


@pytest.mark.parametrize("code", ["LIT", "KAZ", "GEO"])
def test_calendar_provider_season_and_filter_do_not_use_ao_year(code) -> None:
    spec = DomesticLeagueSpec(code, "1", "Calendar", True)
    source = pd.DataFrame([
        {"id": index, "date_utc": date, "league_id": SECONDARY_LEAGUE_IDS[code],
         "home_team_id": 1, "away_team_id": 2, "goals_home": 1, "goals_away": 0, "is_played": True}
        for index, date in enumerate(["2025-06-30T23:00:00Z", "2025-07-01T00:00:00Z", "2026-01-01T01:00:00+02:00", "2026-01-01T00:00:00Z"])
    ])
    result = normalize_secondary_fixtures(source, pd.DataFrame({"id": [1, 2], "name": ["A", "B"]}), spec, start_year=2025, end_year=2025)
    assert result.provider_season.tolist() == ["2025", "2025", "2025"]
    assert result.ao_season.tolist() == ["2024/25", "2025/26", "2025/26"]


def test_secondary_explicit_source_season_preserves_delayed_winter_fixture() -> None:
    source = pd.DataFrame([{"id": 1, "date_utc": "2020-07-15T12:00:00Z", "season": "2019-2020", "league_id": SECONDARY_LEAGUE_IDS["SCO"], "home_team_id": 1, "away_team_id": 2, "goals_home": 1, "goals_away": 0, "is_played": True}])
    result = normalize_secondary_fixtures(source, pd.DataFrame({"id": [1, 2], "name": ["A", "B"]}), DomesticLeagueSpec("SCO", "4330", "Scottish"), start_year=2019, end_year=2019)
    assert result.provider_season.tolist() == ["2019-2020"]
    assert result.ao_season.tolist() == ["2019/20"]


@pytest.mark.parametrize("counts,expected,method", [([10, 100, 101], 100, "MEDIAN_NO_RECURRING_MODE"), ([100, 100, 200], 100, "RECURRING_MODE")])
def test_secondary_format_requires_a_recurring_mode(counts, expected, method) -> None:
    frame = pd.DataFrame([
        {"provider_season": str(2020 + season), "source_event_id": f"{season}:{index}", "home_source_team_id": str(index % 4), "away_source_team_id": str((index + 1) % 4), "kickoff_utc": "2020-01-01T12:00:00Z", "home_goals": 1, "away_goals": 0}
        for season, count in enumerate(counts) for index in range(count)
    ])
    result = assess_secondary_league_seasons(frame, DomesticLeagueSpec("LIT", "4651", "Lithuanian", True))
    assert result.table_expected_matches.eq(expected).all()
    assert result.format_expectation_method.eq(method).all()
    assert result.quality_status.tolist() == ["ACCEPTED" if count >= .95 * expected else "REJECTED" for count in counts]


def test_merge_checks_canonical_fixture_after_provider_alias_resolution() -> None:
    rows = pd.DataFrame([
        {"match_id": "a", "source_event_id": "1", "sportsdb_league_id": "4690", "country_code": "HUN", "ao_season": "2025/26", "kickoff_utc": "2025-08-01T12:00:00Z", "home_source_team_id": "TSD:1", "away_source_team_id": "TSD:2", "home_goals": 1, "away_goals": 0},
        {"match_id": "b", "source_event_id": "2", "sportsdb_league_id": "HF:47", "country_code": "HUN", "ao_season": "2024/25", "kickoff_utc": "2025-08-01T15:00:00+03:00", "home_source_team_id": "HF:1", "away_source_team_id": "HF:2", "home_goals": 1, "away_goals": 0},
    ])
    bridge = pd.DataFrame([{"country_code": "HUN", "source_team_id": f"{provider}:{team}", "ao_club_id": str(team)} for provider in ("TSD", "HF") for team in (1, 2)])
    rows, _ = canonicalize_candidate_domestic_state(rows, bridge, source_switch_countries=("HUN",))
    with pytest.raises(ValueError, match="duplicate domestic fixture"):
        merge_domestic_candidate(rows.iloc[:1], rows.iloc[1:])


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


def test_numeric_provider_league_without_switched_country_remains_mergeable() -> None:
    matches = pd.DataFrame([{"match_id": "a", "country_code": "ENG", "sportsdb_league_id": 4328, "kickoff_utc": "2025-01-01T12:00:00Z", "home_source_team_id": 1, "away_source_team_id": 2}])
    bridge = pd.DataFrame([{"country_code": "ENG", "source_team_id": "1", "ao_club_id": "A"}])
    result, _ = canonicalize_candidate_domestic_state(matches, bridge, source_switch_countries=("HUN",))
    assert result.sportsdb_league_id.tolist() == ["4328"]


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


# --- Kabul edilmis kapsam bosluklari (DATA_CONTRACTS.md 11.2) ---------------
#
# Lig-sezon coverage kapisi iki sezonu bilerek disarida birakir.  Asagidaki
# testler listenin sessizce buyumesini engeller: yeni bir lig-sezonu duserse ya
# da bu ikisi geri gelirse test kirilir ve neden audit edilir.  Testler yalnizca
# commit edilmis dosyalari okur; secondary parquet cache gitignore'dadir.

_ROOT = Path(__file__).resolve().parents[1]
_EXPANSION = _ROOT / "data" / "domestic_league_expansion_ucl_uel"

# (country_code, provider_season) -> coverage kapisinda kalan lig-sezonlari.
ACCEPTED_COVERAGE_GAPS = {("GEO", "2014"), ("LIT", "2020")}

# Bir ulkenin kendi ilk ve son sezonu arasinda kalan bos yillar.  GEO 2014 bu
# kumede degildir cunku GEO'nun veri araligi artik 2015'te basliyor; o bosluk
# asagida ayrica pinlenir.
KNOWN_INTERIOR_SEASON_HOLES = {
    ("ARM", 2017),  # kaynak-sezonu duzeltmesinden devralindi, incelenmedi
    ("FRA", 2016),  # kaynak-sezonu duzeltmesinden devralindi, incelenmedi
    ("LIT", 2020),  # coverage kapisi, kabul edilmis bosluk
}


def _production_domestic_input() -> pd.DataFrame:
    return merge_domestic_candidate(
        pd.read_csv(_EXPANSION / "domestic_matches_candidate.csv", low_memory=False),
        pd.read_csv(_EXPANSION / "live_2026_27_matches.csv", low_memory=False),
    )


def test_documented_coverage_gaps_carry_no_matches_in_the_quality_audit() -> None:
    quality = pd.read_csv(_EXPANSION / "league_season_quality.csv")
    for country, season in sorted(ACCEPTED_COVERAGE_GAPS):
        row = quality[
            quality["country_code"].astype(str).eq(country)
            & quality["provider_season"].astype(str).eq(season)
        ]
        assert len(row) == 1, f"{country} {season} icin tek kalite satiri bekleniyor"
        assert row["quality_status"].item() != "ACCEPTED"
        assert int(row["schedule_matches"].item()) == 0


def test_documented_coverage_gaps_are_absent_from_the_production_input() -> None:
    matches = _production_domestic_input()
    year = pd.to_datetime(matches["kickoff_utc"], utc=True).dt.year
    for country, season in sorted(ACCEPTED_COVERAGE_GAPS):
        present = matches["country_code"].astype(str).eq(country) & year.eq(int(season))
        assert not present.any(), f"{country} {season} beklenmedik sekilde geri geldi"


def test_production_domestic_input_has_no_undocumented_season_hole() -> None:
    matches = _production_domestic_input()
    year = pd.to_datetime(matches["kickoff_utc"], utc=True).dt.year
    holes: set[tuple[str, int]] = set()
    for country, group in year.groupby(matches["country_code"].astype(str)):
        played = set(group.tolist())
        holes.update(
            (country, candidate)
            for candidate in range(min(played) + 1, max(played))
            if candidate not in played
        )
    assert holes == KNOWN_INTERIOR_SEASON_HOLES


# --- Kaynak secimi audit izi (Tur 2: yayimlanan gerekce yaniltiyordu) -------


def _quality_row(country: str, season: str, status: str, reason: str, matches: int, expected, coverage) -> dict[str, object]:
    return {
        "country_code": country, "provider_season": season, "quality_status": status,
        "quality_reason": reason, "schedule_matches": matches,
        "table_expected_matches": expected, "coverage_rate": coverage,
    }


def _match_row(country: str, season: str, provider: str) -> dict[str, object]:
    return {
        "match_id": f"{provider}-{country}-{season}", "country_code": country,
        "provider_season": season, "kickoff_utc": "2020-08-01T12:00:00+00:00",
        "source_provider": provider,
    }


def test_rejected_fallback_publishes_the_secondary_reason_not_the_primary_error() -> None:
    # LIT 2020 / GEO 2014 kaliba: primary fetch hatasi, secondary coverage reddi.
    primary_q = pd.DataFrame([_quality_row("LIT", "2020", "REJECTED", "FETCH_OR_PARSE_ERROR:ValueError", 0, None, None)])
    secondary_q = pd.DataFrame([_quality_row("LIT", "2020", "REJECTED", "SECONDARY_INFERRED_FORMAT_BELOW_95_PCT", 60, 127.0, 0.472441)])
    _, quality = select_source_safe_seasons(
        pd.DataFrame(columns=["match_id", "country_code", "provider_season", "kickoff_utc", "source_provider"]),
        primary_q,
        pd.DataFrame(columns=["match_id", "country_code", "provider_season", "kickoff_utc", "source_provider"]),
        secondary_q,
    )
    row = quality.iloc[0]
    assert row["source_selection"] == "UNAVAILABLE"
    # Yayimlanan gerekce hala primary'nin; ama secondary'ninki artik kaybolmuyor.
    assert row["secondary_quality_reason"] == "SECONDARY_INFERRED_FORMAT_BELOW_95_PCT"
    assert int(row["secondary_schedule_matches"]) == 60
    assert float(row["secondary_coverage_rate"]) == pytest.approx(0.472441)


def test_accepted_primary_records_the_secondary_verdict_it_overrode() -> None:
    # GEO 2020 kaliba: secondary reddediyor, primary daha gevsek beklentiyle kabul ediyor.
    primary_q = pd.DataFrame([_quality_row("GEO", "2020", "ACCEPTED", "TABLE_COVERAGE_OK", 92, 72.0, 1.277778)])
    secondary_q = pd.DataFrame([_quality_row("GEO", "2020", "REJECTED", "SECONDARY_INFERRED_FORMAT_BELOW_95_PCT", 94, 184.0, 0.510870)])
    selected, quality = select_source_safe_seasons(
        pd.DataFrame([_match_row("GEO", "2020", "PRIMARY")]),
        primary_q,
        pd.DataFrame([_match_row("GEO", "2020", "SECONDARY")]),
        secondary_q,
    )
    row = quality.iloc[0]
    assert row["source_selection"] == "PRIMARY_ACCEPTED"
    assert selected["source_provider"].tolist() == ["PRIMARY"]
    # Kapinin kaynak-bagimliligi artik artifact'ta gorunur.
    assert row["secondary_quality_status"] == "REJECTED"
    assert float(row["table_expected_matches"]) == 72.0
    assert float(row["secondary_table_expected_matches"]) == 184.0


def test_missing_counterpart_is_recorded_as_absent_not_blank() -> None:
    primary_q = pd.DataFrame([_quality_row("ENG", "2020", "ACCEPTED", "TABLE_COVERAGE_OK", 380, 380.0, 1.0)])
    _, quality = select_source_safe_seasons(
        pd.DataFrame([_match_row("ENG", "2020", "PRIMARY")]),
        primary_q,
        pd.DataFrame(columns=["country_code", "provider_season", "quality_status", "quality_reason", "schedule_matches", "table_expected_matches", "coverage_rate"]),
        pd.DataFrame(columns=["country_code", "provider_season", "quality_status", "quality_reason", "schedule_matches", "table_expected_matches", "coverage_rate"]),
    )
    assert quality.iloc[0]["secondary_quality_status"] == "ABSENT"


# --- Tur 2 #8: naive kickoff downstream guard'i asmamali -------------------


def _minimal_candidate_row(kickoff: str, match_id: str = "T-1") -> dict[str, object]:
    return {
        "match_id": match_id, "source_event_id": match_id,
        "sportsdb_league_id": "4328", "league_name": "L", "country_code": "ZZZ",
        "provider_season": "2026", "ao_season": "2026/27", "kickoff_utc": kickoff,
        "home_source_team_id": "1", "away_source_team_id": "2",
        "home_team_name": "H", "away_team_name": "A",
        "home_goals": 1, "away_goals": 0, "status": "FT", "source_provider": "TEST",
    }


def test_merge_refuses_a_naive_kickoff_instead_of_assuming_utc() -> None:
    # replay_domestic_poisson_state naive degeri dogru reddediyor; merge onu
    # UTC'ye cevirirse o guard etkisiz kalir ve checkpoint'e yanlis gun girer.
    empty = pd.DataFrame(columns=list(_minimal_candidate_row("x")))
    with pytest.raises(ValueError, match="must be timezone-aware"):
        merge_domestic_candidate(
            pd.DataFrame([_minimal_candidate_row("2026-04-02 12:00:00")]), empty
        )


def test_merge_still_accepts_an_explicit_offset() -> None:
    empty = pd.DataFrame(columns=list(_minimal_candidate_row("x")))
    merged = merge_domestic_candidate(
        pd.DataFrame([_minimal_candidate_row("2026-04-02T12:00:00+00:00")]), empty
    )
    assert str(merged["kickoff_utc"].iloc[0]) == "2026-04-02 12:00:00+00:00"


def test_merge_normalizes_a_non_utc_offset_rather_than_dropping_it() -> None:
    empty = pd.DataFrame(columns=list(_minimal_candidate_row("x")))
    merged = merge_domestic_candidate(
        pd.DataFrame([_minimal_candidate_row("2026-04-02T12:00:00-04:00")]), empty
    )
    assert str(merged["kickoff_utc"].iloc[0]) == "2026-04-02 16:00:00+00:00"
