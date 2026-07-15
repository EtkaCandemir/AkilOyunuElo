from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_dynamic_backtest_dataset import (  # noqa: E402
    annotate_group_stage,
    annotate_knockout_round,
    parse_penalty_records,
)


def match_row(
    home_id: int,
    away_id: int,
    *,
    home_goals: int = 1,
    away_goals: int = 0,
) -> dict[str, object]:
    return {
        "season": "2024/25",
        "competition": "UCL",
        "round": "Group Stage",
        "home_team_id": home_id,
        "away_team_id": away_id,
        "home_team_name": f"Team {home_id}",
        "away_team_name": f"Team {away_id}",
        "home_goals": home_goals,
        "away_goals": away_goals,
    }


def test_group_stage_reconstruction_gives_each_team_one_match_per_matchday() -> None:
    pair_order = ((1, 2), (3, 4), (1, 3), (2, 4), (1, 4), (2, 3))
    rows = []
    for home_id, away_id in pair_order:
        rows.append(match_row(home_id, away_id))
        rows.append(match_row(away_id, home_id))

    events = annotate_group_stage(pd.DataFrame(rows))

    assert sorted(events["matchday"].unique()) == [1, 2, 3, 4, 5, 6]
    for _, matchday in events.groupby("matchday"):
        team_ids = list(matchday["home_team_id"]) + list(matchday["away_team_id"])
        assert sorted(team_ids) == [1, 2, 3, 4]


def test_penalty_shootout_is_separate_from_match_score_and_sets_advancing_team() -> None:
    source = """
    <div class="cupheader">CHAMPIONS LEAGUE</div>
    <div class="roundheader">Final</div>
    <table>
      <tr><td>Alpha</td><td>Eng</td><td>Beta</td><td>Esp</td><td>1-1</td><td></td></tr>
      <tr><td colspan="5" class="pstext">Penalty shootout: Alpha - Beta</td>
          <td class="psresult">4 - 3</td></tr>
    </table>
    """
    penalties = parse_penalty_records(source)
    row = match_row(1, 2, home_goals=1, away_goals=1)
    row["round"] = "Final"
    row["home_team_name"] = "Alpha"
    row["away_team_name"] = "Beta"
    events, matched = annotate_knockout_round(
        pd.DataFrame([row]),
        {penalties[0].lookup_key: penalties[0]},
    )

    assert len(matched) == 1
    assert bool(events.iloc[0]["decided_on_penalties"])
    assert events.iloc[0]["advanced_team_id"] == 1
    assert events.iloc[0]["home_goals"] == events.iloc[0]["away_goals"] == 1
    assert bool(events.iloc[0]["is_neutral"])


def test_two_leg_tie_has_one_decider_and_one_advancing_team() -> None:
    first = match_row(1, 2, home_goals=2, away_goals=0)
    first["round"] = "Quarter Finals"
    second = match_row(2, 1, home_goals=1, away_goals=0)
    second["round"] = "Quarter Finals"

    events, _ = annotate_knockout_round(pd.DataFrame([first, second]), {})

    assert events["leg_number"].tolist() == [1, 2]
    assert events["is_tie_decider"].tolist() == [False, True]
    assert pd.isna(events.iloc[0]["advanced_team_id"])
    assert events.iloc[1]["advanced_team_id"] == 1
    assert events["tie_id"].nunique() == 1
