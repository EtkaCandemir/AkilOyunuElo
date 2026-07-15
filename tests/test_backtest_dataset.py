from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_backtest_dataset import (
    history_team_key,
    parse_matches,
    parse_qualification,
    read_cached_text,
)
from scripts.build_backtest_stage_b import parse_league_urls, standings_team_key


def test_extended_builder_cli_uses_explicit_year_range() -> None:
    source = Path(ROOT / "scripts" / "build_backtest_dataset.py").read_text()

    assert "--start-end-year" in source
    assert "--last-end-year" in source
    assert "--output-root" in source


def test_cached_text_supports_legacy_windows_encoding(tmp_path: Path) -> None:
    path = tmp_path / "legacy.html"
    path.write_bytes("UEFA - 2018/19".replace("-", "–").encode("windows-1252"))

    assert read_cached_text(path) == "UEFA – 2018/19"


@pytest.mark.parametrize(
    ("old_name", "new_name"),
    [
        ("Mladost Podgorica", "OFK Titograd"),
        ("FK Trakai", "FK Riteriai"),
    ],
)
def test_historical_club_renames_share_a_key(old_name: str, new_name: str) -> None:
    assert history_team_key(old_name) == history_team_key(new_name)


def test_legacy_qualification_page_yields_league_url() -> None:
    source = """
    <pre><div class="yellow">  1 Spain 104.998\n---</div>
    (league: <a href="https://en.wikipedia.org/wiki/2017-18_La_Liga">wiki</a>)
    (cup: <a href="https://en.wikipedia.org/wiki/2017-18_Copa_del_Rey">wiki</a>)
    <div class="yellow">  2 Germany 79.498\n---</div></pre>
    """

    assert parse_league_urls(source)["spain"].endswith("2017-18_La_Liga")


def test_legacy_qualification_routes_preserve_league_positions() -> None:
    source = """
    <pre><div class="yellow">  1 Example 25.000
    ----------------------------------------
    CL1=Alpha FC 10.000 (ch/GS)
    CL2=Beta FC 5.000 (ch/Q2)
     CL=Gamma FC 2.000 (ch/Q1)
    EL1=Delta FC 1.000 (eu/Q1)
    </div></pre>
    """

    rows = parse_qualification(source)

    assert rows["route"].tolist() == ["CL1", "CL2", "CL", "EL1"]
    assert rows["is_league_champion"].tolist() == [True, False, True, False]


def test_legacy_league_url_survives_malformed_anchor_siblings() -> None:
    source = """
    <a href=http://broken.example>*<a>)
    <div class="yellow"
    > 27 Bulgaria 15.875
    ----------------------------------------
    CL=Champion 10.000 (ch/Q1)
    </div>
    (league: <a
     href="https://en.wikipedia.org/wiki/Bulgarian_League">wiki</a>)
    """

    assert parse_league_urls(source)["bulgaria"].endswith("Bulgarian_League")


@pytest.mark.parametrize(
    ("participant_name", "standings_name"),
    [
        ("FC København", "Copenhagen"),
        ("Kalju Nomme", "Nõmme Kalju"),
        ("Stade Rennais", "Rennes"),
        ("Levadia Tallinn", "FCI Levadia"),
        ("TSC Backa Topola", "TSC"),
        ("FK Riteriai", "Trakai"),
        ("FK Trakai", "Trakai"),
        ("B36 Torshavn", "B36"),
        ("Rabotnicki Skopje", "Rabotnicki"),
    ],
)
def test_historical_standings_names_share_a_key(
    participant_name: str,
    standings_name: str,
) -> None:
    assert standings_team_key(participant_name) == standings_team_key(standings_name)


def test_two_leg_score_is_reversed_with_home_team() -> None:
    html = """
    <div class="cupheader">CHAMPIONS LEAGUE</div>
    <div class="roundheader">1st Qualifying Round</div>
    <table><tr>
      <td>Slovan Bratislava</td><td>Svk</td>
      <td>FC Struga</td><td>Mac</td>
      <td>4-2</td><td>2-1</td>
    </tr></table>
    """

    matches = parse_matches(html)

    assert matches.iloc[0]["home_team_name"] == "Slovan Bratislava"
    assert matches.iloc[0]["home_goals"] == 4
    assert matches.iloc[0]["away_goals"] == 2
    assert matches.iloc[1]["home_team_name"] == "FC Struga"
    assert matches.iloc[1]["home_goals"] == 1
    assert matches.iloc[1]["away_goals"] == 2
