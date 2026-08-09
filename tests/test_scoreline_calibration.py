from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ao_elo.scoreline import fit_elo_poisson
from ao_elo.scoreline_calibration import (
    GoalLevelCalibrationConfig,
    fit_goal_level_calibration,
    goal_level_candidates,
    predict_with_goal_levels,
    run_goal_level_walk_forward_backtest,
)


def matches_for_seasons(seasons: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    scores = ((2, 0), (1, 0), (1, 1), (0, 1), (2, 1), (0, 0))
    match_id = 0
    for season_index, season in enumerate(seasons):
        for repeat in range(4):
            for score_index, (home_goals, away_goals) in enumerate(scores):
                match_id += 1
                competition = ("UCL", "UEL", "UECL")[score_index % 3]
                home_rating = 1300.0 + 40.0 * score_index
                away_rating = 1250.0 - 20.0 * score_index
                expected = 1.0 / (
                    1.0 + 10.0 ** (-((home_rating - away_rating + 40.0) / 400.0))
                )
                draw = 0.24 * 4.0 * expected * (1.0 - expected)
                rows.append(
                    {
                        "match_id": f"level-{match_id}",
                        "season": season,
                        "competition": competition,
                        "kickoff_utc": pd.Timestamp("2018-07-01", tz="UTC")
                        + pd.Timedelta(days=365 * season_index + repeat * 10 + score_index),
                        "tie_id": f"tie-{match_id}",
                        "stage": "LEAGUE",
                        "is_knockout": False,
                        "decided_on_penalties": False,
                        "home_team_id": 1 + score_index,
                        "away_team_id": 20 + score_index,
                        "home_goals": home_goals + (1 if competition == "UCL" else 0),
                        "away_goals": away_goals,
                        "is_neutral": False,
                        "home_live_pre": home_rating,
                        "away_live_pre": away_rating,
                        "ao_home_probability": expected - 0.5 * draw,
                        "ao_draw_probability": draw,
                        "ao_away_probability": 1.0 - expected - 0.5 * draw,
                    }
                )
    return pd.DataFrame(rows)


def test_candidate_grid_has_65_unique_options() -> None:
    candidates = goal_level_candidates()

    assert len(candidates) == 65
    assert len({candidate.key for candidate in candidates}) == 65
    assert sum(candidate.is_baseline for candidate in candidates) == 1


def test_competition_calibration_raises_high_scoring_ucl_level() -> None:
    matches = matches_for_seasons(("2018/19", "2019/20"))
    scoreline = fit_elo_poisson(
        matches, elo_scale=400.0, home_advantage=40.0, rho_grid=(0.0,)
    )
    fitted = fit_goal_level_calibration(
        matches,
        scoreline,
        GoalLevelCalibrationConfig(1.0, 0.0, 1),
        elo_scale=400.0,
        home_advantage=40.0,
    )

    assert fitted.competition_log_offsets["UCL"] > fitted.competition_log_offsets["UEL"]


def test_prior_season_calibration_uses_only_requested_completed_window() -> None:
    matches = matches_for_seasons(("2018/19", "2019/20", "2020/21"))
    scoreline = fit_elo_poisson(
        matches, elo_scale=400.0, home_advantage=40.0, rho_grid=(0.0,)
    )
    fitted = fit_goal_level_calibration(
        matches,
        scoreline,
        GoalLevelCalibrationConfig(0.0, 1.0, 2),
        elo_scale=400.0,
        home_advantage=40.0,
    )

    assert fitted.season_source == ("2019/20", "2020/21")


def test_goal_level_multiplier_preserves_home_away_rate_ratio() -> None:
    matches = matches_for_seasons(("2018/19", "2019/20"))
    scoreline = fit_elo_poisson(
        matches, elo_scale=400.0, home_advantage=40.0, rho_grid=(0.0,)
    )
    baseline = fit_goal_level_calibration(
        matches,
        scoreline,
        GoalLevelCalibrationConfig(0.0, 0.0, 1),
        elo_scale=400.0,
        home_advantage=40.0,
    )
    calibrated = fit_goal_level_calibration(
        matches,
        scoreline,
        GoalLevelCalibrationConfig(1.0, 0.5, 1),
        elo_scale=400.0,
        home_advantage=40.0,
    )
    base_predictions = predict_with_goal_levels(
        matches, scoreline, baseline, elo_scale=400.0, home_advantage=40.0
    )
    calibrated_predictions = predict_with_goal_levels(
        matches, scoreline, calibrated, elo_scale=400.0, home_advantage=40.0
    )

    np.testing.assert_allclose(
        base_predictions["lambda_home"] / base_predictions["lambda_away"],
        calibrated_predictions["lambda_home"]
        / calibrated_predictions["lambda_away"],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        calibrated_predictions[
            ["home_probability", "draw_probability", "away_probability"]
        ].sum(axis=1),
        1.0,
        atol=1e-12,
    )


def test_walk_forward_is_leakage_safe_and_does_not_mutate_ratings() -> None:
    matches = matches_for_seasons(
        ("2018/19", "2019/20", "2020/21", "2021/22")
    )
    folds = (
        (("2018/19", "2019/20"), "2020/21"),
        (("2018/19", "2019/20", "2020/21"), "2021/22"),
    )
    ratings = matches[["home_live_pre", "away_live_pre"]].copy(deep=True)
    result = run_goal_level_walk_forward_backtest(
        matches,
        folds,
        elo_scale=400.0,
        home_advantage=40.0,
        bootstrap_samples=100,
    )

    assert result.fold_selections["train_seasons"].tolist() == [
        "2018/19|2019/20",
        "2018/19|2019/20|2020/21",
    ]
    assert len(result.unseen_predictions) == 48
    assert set(result.unseen_predictions["season"]) == {"2020/21", "2021/22"}
    pd.testing.assert_frame_equal(
        matches[["home_live_pre", "away_live_pre"]], ratings
    )
    assert result.guardrails["elo_identity_preserved"] is True


def test_future_training_season_is_rejected() -> None:
    matches = matches_for_seasons(("2019/20", "2020/21", "2021/22"))

    with pytest.raises(ValueError, match="must precede"):
        run_goal_level_walk_forward_backtest(
            matches,
            ((("2021/22",), "2020/21"),),
            elo_scale=400.0,
            home_advantage=40.0,
            bootstrap_samples=100,
        )


def test_disabled_season_calibration_requires_canonical_lookback() -> None:
    with pytest.raises(ValueError, match="lookback=1"):
        GoalLevelCalibrationConfig(0.5, 0.0, 2).validate()
