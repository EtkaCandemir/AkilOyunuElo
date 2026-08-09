from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_opponent_quintile_backtest import (
    BASELINE_KEY,
    build_team_perspectives,
    candidate_grid,
)


def test_candidate_grid_contains_baseline_and_full_regularization_surface() -> None:
    candidates = candidate_grid()

    assert len(candidates) == 193
    assert candidates[0].key == BASELINE_KEY
    assert len({candidate.key for candidate in candidates}) == len(candidates)


def test_team_perspectives_keep_penalty_field_draw_as_half_score() -> None:
    baseline = pd.DataFrame(
        {
            "match_id": ["M-1"],
            "season": ["2025/26"],
            "kickoff_utc": ["2025-08-01T18:00:00Z"],
            "snapshot_id": ["S-1"],
            "home_club_id": ["AO-HOME"],
            "away_club_id": ["AO-AWAY"],
            "expected_home_score": [0.62],
            "actual_home_score": [0.5],
            "home_quintile_dynamic": [4],
            "away_quintile_dynamic": [2],
            "home_quintile_fixed": [4],
            "away_quintile_fixed": [2],
        }
    )

    perspectives = build_team_perspectives(baseline).set_index("club_id")

    assert perspectives.loc["AO-HOME", "actual_score"] == 0.5
    assert perspectives.loc["AO-AWAY", "actual_score"] == 0.5
    assert perspectives.loc["AO-HOME", "expected_score"] == 0.62
    assert perspectives.loc["AO-AWAY", "expected_score"] == 0.38
    assert perspectives.loc["AO-HOME", "opponent_quintile_dynamic"] == 2
    assert perspectives.loc["AO-AWAY", "opponent_quintile_dynamic"] == 4
