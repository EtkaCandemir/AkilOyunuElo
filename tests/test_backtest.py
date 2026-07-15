from __future__ import annotations

import pandas as pd
import pytest

from ao_elo.backtest import evaluate_match_predictions


def test_perfect_rating_order_beats_reversed_order() -> None:
    matches = pd.DataFrame(
        [
            {"season": "2024/25", "home_team_id": 1, "away_team_id": 2, "home_goals": 2, "away_goals": 0},
            {"season": "2024/25", "home_team_id": 2, "away_team_id": 1, "home_goals": 0, "away_goals": 1},
        ]
    )
    correct = pd.DataFrame(
        [
            {"season": "2024/25", "team_id": 1, "ao_first_elo": 800},
            {"season": "2024/25", "team_id": 2, "ao_first_elo": 600},
        ]
    )
    reversed_ratings = correct.assign(ao_first_elo=[600, 800])

    correct_metrics = evaluate_match_predictions(correct, matches)
    reversed_metrics = evaluate_match_predictions(reversed_ratings, matches)

    assert correct_metrics["log_loss"] < reversed_metrics["log_loss"]
    assert correct_metrics["brier"] < reversed_metrics["brier"]
    assert correct_metrics["decisive_accuracy"] == pytest.approx(1.0)
    assert reversed_metrics["decisive_accuracy"] == pytest.approx(0.0)


def test_draw_is_scored_as_half_result() -> None:
    ratings = pd.DataFrame(
        [
            {"season": "2024/25", "team_id": 1, "ao_first_elo": 700},
            {"season": "2024/25", "team_id": 2, "ao_first_elo": 700},
        ]
    )
    matches = pd.DataFrame(
        [{"season": "2024/25", "home_team_id": 1, "away_team_id": 2, "home_goals": 1, "away_goals": 1}]
    )

    metrics = evaluate_match_predictions(ratings, matches)

    assert metrics["brier"] == pytest.approx(0.0)
    assert metrics["log_loss"] == pytest.approx(0.69314718056)


def test_missing_team_rating_is_rejected() -> None:
    ratings = pd.DataFrame(
        [{"season": "2024/25", "team_id": 1, "ao_first_elo": 700}]
    )
    matches = pd.DataFrame(
        [{"season": "2024/25", "home_team_id": 1, "away_team_id": 2, "home_goals": 1, "away_goals": 0}]
    )

    with pytest.raises(ValueError, match="without ratings"):
        evaluate_match_predictions(ratings, matches)
