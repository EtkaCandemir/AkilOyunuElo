from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_dynamic_core_calibration import (  # noqa: E402
    DynamicCoreConfig,
    SeasonData,
    expanding_folds,
    expected_home_score,
    load_calibration_data,
    run_season,
)


def synthetic_season() -> SeasonData:
    initial = np.array([np.nan, 800.0, 800.0])
    return SeasonData(
        season="2024/25",
        initial_ratings=initial,
        active_team_ids=np.array([1, 2]),
        home_team_ids=np.array([1, 2]),
        away_team_ids=np.array([2, 1]),
        actual_home_scores=np.array([1.0, 0.5]),
        neutral_flags=np.array([False, True]),
        competitions=np.array(["UCL", "UCL"]),
        match_ids=np.array(["m1", "m2"]),
    )


def test_expected_score_uses_no_home_advantage_at_neutral_venue() -> None:
    config = DynamicCoreConfig(elo_scale=300, home_advantage=50, k_factor=20)

    assert expected_home_score(800, 800, config, neutral=True) == pytest.approx(0.5)
    assert expected_home_score(800, 800, config, neutral=False) > 0.5


def test_dynamic_updates_are_zero_sum() -> None:
    config = DynamicCoreConfig(elo_scale=300, home_advantage=50, k_factor=20)
    metrics, predictions = run_season(synthetic_season(), config, return_predictions=True)

    assert metrics["mean_rating_change"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["max_abs_rating_change"] > 0
    assert predictions is not None
    assert len(predictions) == 2


def test_zero_k_preserves_all_ratings() -> None:
    config = DynamicCoreConfig(elo_scale=300, home_advantage=50, k_factor=0)
    metrics, _ = run_season(synthetic_season(), config)

    assert metrics["rating_change_std"] == pytest.approx(0.0)
    assert metrics["max_abs_rating_change"] == pytest.approx(0.0)
    assert metrics["start_end_rank_correlation"] == pytest.approx(1.0)


def test_walk_forward_never_uses_test_or_future_season() -> None:
    seasons = ("2018/19", "2019/20", "2020/21", "2021/22")

    assert expanding_folds(seasons) == (
        (("2018/19", "2019/20"), "2020/21"),
        (("2018/19", "2019/20", "2020/21"), "2021/22"),
    )


def test_exact_date_loader_rejects_chronology_regression(tmp_path: Path) -> None:
    events = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "season": "2024/25",
                "event_order": 1,
                "competition": "UCL",
                "home_team_id": 1,
                "away_team_id": 2,
                "actual_home_score": 1.0,
                "is_neutral": False,
                "kickoff_utc": "2025-01-02T20:00:00Z",
            },
            {
                "match_id": "m2",
                "season": "2024/25",
                "event_order": 2,
                "competition": "UCL",
                "home_team_id": 2,
                "away_team_id": 1,
                "actual_home_score": 0.0,
                "is_neutral": False,
                "kickoff_utc": "2025-01-01T20:00:00Z",
            },
        ]
    )
    path = tmp_path / "events.csv"
    events.to_csv(path, index=False)

    with pytest.raises(ValueError, match="exact kickoff_utc chronology"):
        load_calibration_data(tmp_path, path, require_exact_utc=True)
