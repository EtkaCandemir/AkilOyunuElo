from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_domestic_surprise_tradeoff_diagnostic import (  # noqa: E402
    CURRENT,
    GLOBAL,
    add_team_loss_contributions,
    discover_configs,
    exposure_band,
    magnitude_band,
)


def test_discover_configs_reads_current_and_best_loss() -> None:
    production = {
        "domestic_surprise": {
            "coefficient": 0.4,
            "variance_penalty": 0.5,
            "max_abs_adjustment": 30.0,
            "minimum_history_seasons": 5,
        }
    }
    surface = pd.DataFrame(
        [
            {"theta": 0.4, "gamma": 0.5, "cap": 30.0, "brier_1x2": 0.50, "log_loss_1x2": 0.70},
            {"theta": 1.75, "gamma": 0.5, "cap": 150.0, "brier_1x2": 0.49, "log_loss_1x2": 0.69},
        ]
    )
    configs = discover_configs(production, surface)
    assert configs[CURRENT].coefficient == 0.4
    assert configs[GLOBAL].coefficient == 1.75
    assert configs[GLOBAL].max_abs_adjustment == 150.0


def test_exposure_and_magnitude_boundaries() -> None:
    exposure = exposure_band(pd.Series([0.0, 0.25, 0.2501, 1.0])).astype(str).tolist()
    assert exposure == ["0", "(0,0.25]", "(0.25,0.50]", "(0.75,1.00]"]
    magnitude = magnitude_band(pd.Series([0.0, 10.0, 10.1, 100.1])).astype(str).tolist()
    assert magnitude == ["0-10", "0-10", "10-20", "100+"]


def test_quadrant_assignment_uses_loss_and_ranking_signs() -> None:
    teams = pd.DataFrame(
        {
            "season": ["A", "A", "A", "A"],
            "team_id": [1, 2, 3, 4],
            "ranking_error_difference": [-1.0, 1.0, -1.0, 1.0],
        }
    )
    matches = pd.DataFrame(
        {
            "season": ["A", "A", "A", "A"],
            "match_id": ["m1", "m2", "m3", "m4"],
            "competition": ["UCL"] * 4,
            "home_team_id": [1, 2, 3, 4],
            "away_team_id": [9, 9, 9, 9],
            "brier_difference": [-1.0, -1.0, 1.0, 1.0],
            "log_loss_difference": [-1.0, -1.0, 1.0, 1.0],
        }
    )
    result = add_team_loss_contributions(teams, matches).set_index("team_id")
    assert result.loc[1, "loss_ranking_quadrant"] == "A_WIN_WIN"
    assert result.loc[2, "loss_ranking_quadrant"] == "B_LOSS_WIN_RANK_HARM"
    assert result.loc[3, "loss_ranking_quadrant"] == "C_LOSS_HARM_RANK_WIN"
    assert result.loc[4, "loss_ranking_quadrant"] == "D_LOSE_LOSE"
