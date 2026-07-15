from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ao_elo.backtest import evaluate_match_predictions
from scripts.run_nested_walk_forward import (
    expanding_folds,
    match_log_losses,
    select_candidate,
)


def test_expanding_folds_never_train_on_future_season() -> None:
    seasons = ("2021/22", "2022/23", "2023/24", "2024/25")

    folds = expanding_folds(seasons)

    assert folds == [
        (("2021/22", "2022/23"), "2023/24"),
        (("2021/22", "2022/23", "2023/24"), "2024/25"),
    ]


def test_selection_uses_only_requested_training_seasons() -> None:
    rows = []
    for candidate, losses in {
        "stable": (0.60, 0.60, 0.99),
        "future_winner": (0.70, 0.70, 0.01),
    }.items():
        for season, loss in zip(("2021/22", "2022/23", "2023/24"), losses):
            rows.append(
                {
                    "candidate": candidate,
                    "season": season,
                    "competition": "ALL",
                    "log_loss": loss,
                    "brier": loss / 2,
                }
            )

    selected = select_candidate(pd.DataFrame(rows), ("2021/22", "2022/23"))

    assert selected == "stable"


def test_selection_rejects_incomplete_training_coverage() -> None:
    metrics = pd.DataFrame(
        [
            {
                "candidate": "only_one_season",
                "season": "2021/22",
                "competition": "ALL",
                "log_loss": 0.6,
                "brier": 0.2,
            }
        ]
    )

    with pytest.raises(ValueError, match="cover every training season"):
        select_candidate(metrics, ("2021/22", "2022/23"))


def test_paired_match_losses_reconcile_to_backtest_log_loss() -> None:
    ratings = pd.DataFrame(
        [
            {"season": "2024/25", "team_id": 1, "ao_first_elo": 720.0},
            {"season": "2024/25", "team_id": 2, "ao_first_elo": 640.0},
        ]
    )
    matches = pd.DataFrame(
        [
            {
                "season": "2024/25",
                "home_team_id": 1,
                "away_team_id": 2,
                "home_goals": 2,
                "away_goals": 0,
            },
            {
                "season": "2024/25",
                "home_team_id": 2,
                "away_team_id": 1,
                "home_goals": 1,
                "away_goals": 1,
            },
        ]
    )

    expected = evaluate_match_predictions(
        ratings,
        matches,
        home_advantage=70.0,
    )["log_loss"]

    assert match_log_losses(ratings, matches).mean() == pytest.approx(expected)
