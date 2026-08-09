from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ao_elo.scoreline import (
    ScorelineModelConfig,
    dixon_coles_tau,
    elo_goal_expectations,
    exact_score_probability,
    fit_elo_poisson,
    predict_scoreline,
    run_scoreline_walk_forward_backtest,
    scoreline_matrix,
    scoreline_to_1x2,
)


def config(rho: float = 0.0) -> ScorelineModelConfig:
    return ScorelineModelConfig(mu=math.log(1.4), elo_slope=0.8, rho=rho)


def synthetic_matches(seasons: tuple[str, ...] = ("2018/19", "2019/20")) -> pd.DataFrame:
    rows = []
    match_number = 0
    scores = ((1, 0), (1, 1), (0, 1), (2, 0), (2, 1), (0, 0))
    for season_index, season in enumerate(seasons):
        for repeat in range(4):
            for score_index, (home_goals, away_goals) in enumerate(scores):
                match_number += 1
                home_rating = 1200.0 + 35.0 * (score_index - 2)
                away_rating = 1200.0 - 20.0 * (score_index - 2)
                expected = 1.0 / (1.0 + 10.0 ** (-((home_rating - away_rating + 40.0) / 400.0)))
                draw = 0.24 * 4.0 * expected * (1.0 - expected)
                rows.append(
                    {
                        "match_id": f"m{match_number}",
                        "season": season,
                        "competition": ("UCL", "UEL", "UECL")[score_index % 3],
                        "kickoff_utc": pd.Timestamp("2018-07-01", tz="UTC")
                        + pd.Timedelta(days=season_index * 365 + repeat * 8 + score_index),
                        "tie_id": f"t{match_number}",
                        "stage": "LEAGUE",
                        "is_knockout": False,
                        "home_team_id": 1 + score_index,
                        "away_team_id": 20 + score_index,
                        "home_goals": home_goals,
                        "away_goals": away_goals,
                        "is_neutral": False,
                        "decided_on_penalties": False,
                        "home_live_pre": home_rating,
                        "away_live_pre": away_rating,
                        "ao_home_probability": expected - 0.5 * draw,
                        "ao_draw_probability": draw,
                        "ao_away_probability": 1.0 - expected - 0.5 * draw,
                    }
                )
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        ((0, 0), 1.0 - 1.4 * 1.2 * 0.1),
        ((0, 1), 1.0 + 1.4 * 0.1),
        ((1, 0), 1.0 + 1.2 * 0.1),
        ((1, 1), 0.9),
        ((2, 1), 1.0),
    ],
)
def test_dixon_coles_low_score_corrections(
    score: tuple[int, int], expected: float
) -> None:
    assert dixon_coles_tau(*score, 1.4, 1.2, 0.1) == pytest.approx(expected)


def test_rho_zero_is_independent_poisson() -> None:
    matrix, covered = scoreline_matrix(1.4, 1.1, config(0.0))
    independent_00 = math.exp(-1.4) * math.exp(-1.1)

    assert matrix.sum() == pytest.approx(1.0, abs=1e-12)
    assert covered == pytest.approx(1.0, abs=1e-10)
    assert matrix[0, 0] == pytest.approx(independent_00, rel=1e-9)


def test_equal_neutral_ratings_are_symmetric() -> None:
    prediction = predict_scoreline(
        1200.0,
        1200.0,
        is_neutral=True,
        elo_scale=400.0,
        home_advantage=50.0,
        config=config(),
    )

    assert prediction.lambda_home == pytest.approx(prediction.lambda_away)
    assert prediction.home_probability == pytest.approx(prediction.away_probability)


def test_stronger_home_team_increases_goal_and_win_expectation() -> None:
    even = predict_scoreline(
        1200.0, 1200.0, is_neutral=True, elo_scale=400.0, home_advantage=50.0, config=config()
    )
    stronger = predict_scoreline(
        1500.0, 1200.0, is_neutral=True, elo_scale=400.0, home_advantage=50.0, config=config()
    )

    assert stronger.lambda_home > even.lambda_home
    assert stronger.lambda_away < even.lambda_away
    assert stronger.home_probability > even.home_probability


def test_scoreline_probabilities_are_valid() -> None:
    matrix, covered = scoreline_matrix(1.8, 0.9, config(-0.05))
    home, draw, away = scoreline_to_1x2(matrix)

    assert matrix.min() >= 0.0
    assert covered == pytest.approx(1.0, abs=1e-10)
    assert home + draw + away == pytest.approx(1.0)


def test_invalid_dixon_coles_correction_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        dixon_coles_tau(0, 0, 4.0, 4.0, 0.15)


def test_fit_is_deterministic_and_uses_finite_parameters() -> None:
    matches = synthetic_matches()
    first = fit_elo_poisson(matches, elo_scale=400.0, home_advantage=40.0)
    second = fit_elo_poisson(matches, elo_scale=400.0, home_advantage=40.0)

    assert first == second
    assert math.isfinite(first.mu)
    assert 0.05 <= first.elo_slope <= 3.0


def test_poisson_fit_recovers_known_elo_parameters() -> None:
    rng = np.random.default_rng(20260802)
    count = 3000
    mu = math.log(1.35)
    beta = 0.75
    differences = rng.normal(0.0, 180.0, count)
    z = math.log(10.0) * differences / 400.0
    home_goals = rng.poisson(np.exp(mu + 0.5 * beta * z))
    away_goals = rng.poisson(np.exp(mu - 0.5 * beta * z))
    matches = pd.DataFrame(
        {
            "match_id": [f"recovery-{index}" for index in range(count)],
            "season": "2024/25",
            "competition": "UCL",
            "kickoff_utc": pd.date_range("2024-07-01", periods=count, freq="h", tz="UTC"),
            "tie_id": [f"rt-{index}" for index in range(count)],
            "stage": "LEAGUE",
            "is_knockout": False,
            "home_team_id": np.arange(count),
            "away_team_id": np.arange(count) + count,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "is_neutral": True,
            "decided_on_penalties": False,
            "home_live_pre": 1200.0 + differences,
            "away_live_pre": 1200.0,
        }
    )
    fitted = fit_elo_poisson(
        matches,
        elo_scale=400.0,
        home_advantage=0.0,
        rho_grid=(0.0,),
    )

    assert fitted.mu == pytest.approx(mu, abs=0.04)
    assert fitted.elo_slope == pytest.approx(beta, abs=0.08)


def test_exact_score_probability_accepts_120_minute_field_score() -> None:
    probability = exact_score_probability(4, 3, 1.6, 1.2, 0.0)

    assert 0.0 < probability < 1.0


def test_penalty_decided_two_leg_match_keeps_its_field_score() -> None:
    matches = synthetic_matches()
    matches.loc[0, "decided_on_penalties"] = True
    fitted = fit_elo_poisson(matches, elo_scale=400.0, home_advantage=40.0)

    assert math.isfinite(fitted.mu)


def test_walk_forward_excludes_test_season_from_fit_and_preserves_elo() -> None:
    matches = synthetic_matches(("2018/19", "2019/20", "2020/21", "2021/22"))
    folds = (
        (("2018/19", "2019/20"), "2020/21"),
        (("2018/19", "2019/20", "2020/21"), "2021/22"),
    )
    rating_snapshot = matches[["home_live_pre", "away_live_pre"]].copy(deep=True)
    result = run_scoreline_walk_forward_backtest(
        matches,
        folds,
        elo_scale=400.0,
        home_advantage=40.0,
        bootstrap_samples=100,
    )

    assert result.fold_parameters["train_seasons"].tolist() == [
        "2018/19|2019/20",
        "2018/19|2019/20|2020/21",
    ]
    assert len(result.unseen_predictions) == 48
    assert {
        "lambda_home",
        "lambda_away",
        "rho",
        "home_probability",
        "draw_probability",
        "away_probability",
        "expected_total_goals",
        "most_likely_score",
        "over_2_5_probability",
        "btts_probability",
        "exact_score_probability",
    }.issubset(result.unseen_predictions.columns)
    pd.testing.assert_frame_equal(
        matches[["home_live_pre", "away_live_pre"]], rating_snapshot
    )
    assert result.guardrails["elo_identity_preserved"] is True


def test_untouched_holdout_is_rejected() -> None:
    matches = synthetic_matches(("2024/25", "2025/26", "2026/27"))

    with pytest.raises(ValueError, match="untouched holdout"):
        run_scoreline_walk_forward_backtest(
            matches,
            (("2024/25", "2025/26"), "2026/27"),
            elo_scale=400.0,
            home_advantage=40.0,
            bootstrap_samples=100,
        )


def test_elo_goal_expectations_use_no_home_edge_on_neutral_ground() -> None:
    neutral = elo_goal_expectations(
        1200.0,
        1200.0,
        is_neutral=True,
        elo_scale=400.0,
        home_advantage=50.0,
        config=config(),
    )
    home = elo_goal_expectations(
        1200.0,
        1200.0,
        is_neutral=False,
        elo_scale=400.0,
        home_advantage=50.0,
        config=config(),
    )

    assert neutral[0] == pytest.approx(neutral[1])
    assert home[0] > home[1]
