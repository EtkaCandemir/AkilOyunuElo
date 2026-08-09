from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ao_elo.thesportsdb_dataset import (
    build_stats_wide,
    build_xg_source_comparison,
    fallback_identity,
    numeric_or_text,
    payload_rows,
    provider_name,
    stat_slug,
)
from scripts.build_thesportsdb_2025_26_dataset import load_api_key, retry_delay


def test_payload_rows_reads_list_and_ignores_message_payload() -> None:
    assert payload_rows({"lookup": [{"idEvent": "1"}]}) == [{"idEvent": "1"}]
    assert payload_rows({"Message": "No data"}) == []


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Olympique Lyon", "lyon"),
        ("Lyon", "lyon"),
        ("PAOK Thessaloniki", "paok"),
        ("SK Brann Bergen", "brann"),
        ("Red Bull Salzburg", "salzburg"),
        ("FC Salzburg", "salzburg"),
    ],
)
def test_provider_name_aliases(source: str, expected: str) -> None:
    assert provider_name(source) == expected


def test_fallback_identity_can_match_name_and_time_when_score_disagrees() -> None:
    event = pd.Series(
        {
            "competition": "UEL",
            "kickoff_utc": pd.Timestamp("2026-01-29T20:00:00Z"),
            "home_team_name": "Olympique Lyon",
            "away_team_name": "PAOK Thessaloniki",
            "home_goals": 4,
            "away_goals": 2,
        }
    )
    schedules = pd.DataFrame(
        [
            {
                "source_match_id": "A",
                "competition": "UEL",
                "kickoff_parsed": pd.Timestamp("2026-01-29T20:00:00Z"),
                "home_team_name": "Lyon",
                "away_team_name": "PAOK",
                "home_goals": 3,
                "away_goals": 2,
            },
            {
                "source_match_id": "B",
                "competition": "UEL",
                "kickoff_parsed": pd.Timestamp("2026-01-29T20:00:00Z"),
                "home_team_name": "Basel",
                "away_team_name": "Roma",
                "home_goals": 4,
                "away_goals": 2,
            },
        ]
    )
    resolved = fallback_identity(event, schedules)
    assert resolved is not None
    assert resolved["source_match_id"] == "A"
    assert resolved["pair_similarity"] == pytest.approx(1.0)


def test_stats_wide_preserves_all_stats_and_marks_xg_coverage() -> None:
    events = pd.DataFrame(
        [
            {
                "match_id": "M1",
                "competition": "UCL",
                "sportsdb_idEvent": "10",
            },
            {
                "match_id": "M2",
                "competition": "UECL",
                "sportsdb_idEvent": "20",
            },
        ]
    )
    stats = pd.DataFrame(
        [
            {
                "match_id": "M1",
                "strStat": "Expected Goals",
                "intHome": "2.54",
                "intAway": "0.44",
            },
            {
                "match_id": "M1",
                "strStat": "Shots on Goal",
                "intHome": "3",
                "intAway": "1",
            },
        ]
    )
    wide = build_stats_wide(events, stats).set_index("match_id")
    assert wide.loc["M1", "home_expected_goals"] == pytest.approx(2.54)
    assert wide.loc["M1", "away_shots_on_goal"] == 1
    assert bool(wide.loc["M1", "xg_covered"])
    assert bool(wide.loc["M1", "xg_analysis_eligible"])
    assert wide.loc["M1", "xg_quality_status"] == "ELIGIBLE"
    assert not bool(wide.loc["M2", "xg_covered"])


def test_stats_wide_rejects_placeholder_zero_xg() -> None:
    events = pd.DataFrame(
        [{"match_id": "M1", "competition": "UCL", "sportsdb_idEvent": "10"}]
    )
    stats = pd.DataFrame(
        [
            {
                "match_id": "M1",
                "strStat": "Expected Goals",
                "intHome": "0",
                "intAway": "0",
            },
            {
                "match_id": "M1",
                "strStat": "Total Shots",
                "intHome": "16",
                "intAway": "8",
            },
        ]
    )
    row = build_stats_wide(events, stats).iloc[0]
    assert bool(row["xg_raw_present"])
    assert bool(row["xg_placeholder_suspected"])
    assert not bool(row["xg_analysis_eligible"])
    assert row["xg_quality_status"] == "ZERO_INCONSISTENT_WITH_SHOTS"


def test_xg_source_comparison_uses_only_common_eligible_matches() -> None:
    source = pd.DataFrame(
        [
            {
                "match_id": "M1",
                "competition": "UCL",
                "home_expected_goals": 1.5,
                "away_expected_goals": 0.5,
                "xg_raw_present": True,
                "xg_analysis_eligible": True,
                "xg_quality_status": "ELIGIBLE",
            },
            {
                "match_id": "M2",
                "competition": "UCL",
                "home_expected_goals": 0.0,
                "away_expected_goals": 0.0,
                "xg_raw_present": True,
                "xg_analysis_eligible": False,
                "xg_quality_status": "BOTH_ZERO_PLACEHOLDER",
            },
        ]
    )
    fotmob = pd.DataFrame(
        [
            {
                "match_id": "M1",
                "xg_home": 1.4,
                "xg_away": 0.6,
                "xg_covered": True,
                "xg_analysis_eligible": True,
            },
            {
                "match_id": "M2",
                "xg_home": 0.8,
                "xg_away": 0.7,
                "xg_covered": True,
                "xg_analysis_eligible": True,
            },
        ]
    )
    comparison, summary = build_xg_source_comparison(source, fotmob)
    assert int(comparison["common_analysis_eligible"].sum()) == 1
    overall = summary.loc[summary["competition"].eq("ALL")].iloc[0]
    assert overall["common_analysis_eligible"] == 1
    assert overall["home_mae"] == pytest.approx(0.1)


def test_stat_and_numeric_normalization() -> None:
    assert stat_slug("Passes %") == "passes"
    assert numeric_or_text("12") == 12
    assert numeric_or_text("2.5") == pytest.approx(2.5)
    assert pd.isna(numeric_or_text(""))


def test_load_api_key_prefers_environment(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("THESPORTSDB_API_KEY", "environment-key")
    path = tmp_path / ".env.local"
    path.write_text("THESPORTSDB_API_KEY=file-key\n", encoding="utf-8")
    assert load_api_key(path) == "environment-key"


def test_load_api_key_reads_local_ignored_file(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("THESPORTSDB_API_KEY", raising=False)
    path = tmp_path / ".env.local"
    path.write_text('THESPORTSDB_API_KEY="file-key"\n', encoding="utf-8")
    assert load_api_key(path) == "file-key"


def test_retry_delay_uses_numeric_header_or_bounded_backoff() -> None:
    assert retry_delay("12", 1) == 12.0
    assert retry_delay(None, 2) == 20.0
