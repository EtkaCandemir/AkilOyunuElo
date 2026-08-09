from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_progression_probability_calibration import (  # noqa: E402
    candidate_grid,
    run_walk_forward,
)


def test_progression_probability_grid_contains_identity() -> None:
    candidates = candidate_grid()

    assert len(candidates) == 150
    assert any(
        candidate.logit_slope == 1.0
        and candidate.single_home_bias == 0.0
        and candidate.two_leg_first_home_bias == 0.0
        for candidate in candidates
    )


def test_walk_forward_uses_only_earlier_seasons() -> None:
    ties = pd.DataFrame(
        [
            {
                "season": season,
                "tie_id": f"tie-{season}",
                "match_id": f"match-{season}",
                "competition": "UCL",
                "tie_match_count": 2,
                "first_match_neutral": False,
                "team_a_id": 1,
                "team_b_id": 2,
                "first_kickoff_utc": f"{year}-01-01T00:00:00Z",
                "raw_probability": 0.6,
                "actual_advanced": float(index % 2),
                "fold": index + 1,
            }
            for index, (season, year) in enumerate(
                (("2020/21", 2020), ("2021/22", 2021), ("2022/23", 2022))
            )
        ]
    )

    selections, _ = run_walk_forward(
        ties,
        ("2020/21", "2021/22", "2022/23"),
        candidate_grid(),
    )

    assert selections.iloc[0]["training_ties"] == 0
    assert selections.iloc[1]["train_seasons"] == "2020/21"
    assert selections.iloc[2]["train_seasons"] == "2020/21|2021/22"
