from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_pilot_20_teams import DATA_ROOT, run_pilot


def test_twenty_team_pilot_exercises_production_goal_difference(
    tmp_path: Path,
) -> None:
    results = run_pilot(DATA_ROOT, tmp_path)
    teams = results["teams"]
    matches = results["matches"]
    competitions = results["competitions"]

    assert len(teams) == 20
    assert len(matches) == 33
    assert set(teams["competition_track"]) == {"UCL", "UEL", "UECL"}
    assert set(matches["competition"]) == {"UCL", "UEL", "UECL"}
    assert teams["start_elo"].min() == 940.0
    assert teams["start_elo"].max() == 1940.0
    assert teams["total_elo_change"].sum() == pytest.approx(0.0, abs=1e-9)
    assert teams["goal_difference_net_effect_vs_classic"].sum() == pytest.approx(
        0.0,
        abs=1e-9,
    )

    assert matches.loc[
        matches["goal_difference"].le(1),
        "goal_multiplier",
    ].eq(1.0).all()
    assert matches["goal_multiplier"].max() == pytest.approx(1.197721, abs=1e-6)
    assert matches["goal_multiplier"].gt(1.0).any()
    assert matches.loc[
        matches["decided_on_penalties"],
        "goal_multiplier",
    ].eq(1.0).all()
    assert matches["progression_reserve_added"].eq(0.0).all()
    assert matches["trophy_reserve_added"].eq(0.0).all()
    assert matches["total_power_zero_sum_error"].max() <= 1e-9
    assert competitions["elo_conservation_error"].max() <= 1e-9

    expected_outputs = {
        "team_start_end_summary.csv",
        "match_updates_detailed.csv",
        "scenario_summary.csv",
        "competition_summary.csv",
        "model_parameters.csv",
        "pilot_report.md",
    }
    assert expected_outputs.issubset(
        {path.name for path in tmp_path.iterdir() if path.is_file()}
    )
