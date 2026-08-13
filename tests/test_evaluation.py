from __future__ import annotations

from math import log

import numpy as np
import pandas as pd
import pytest

from ao_elo.evaluation import (
    dependency_robust_loss_difference_ci,
    evaluate_1x2_predictions,
    schedule_adjusted_team_performance,
    score_preserving_1x2_probabilities,
    standard_1x2_losses,
)


def test_score_preserving_1x2_probabilities_are_valid_and_preserve_elo_score() -> None:
    expected = pd.Series([0.0, 0.25, 0.5, 0.75, 1.0], index=list("abcde"))

    probabilities = score_preserving_1x2_probabilities(
        expected,
        draw_at_even=0.28,
        draw_shape=1.5,
    )

    assert probabilities.index.tolist() == expected.index.tolist()
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert probabilities.to_numpy().min() >= 0.0
    assert probabilities.to_numpy().max() <= 1.0
    reconstructed = (
        probabilities["home_probability"]
        + 0.5 * probabilities["draw_probability"]
    )
    assert np.allclose(reconstructed, expected)
    assert probabilities.loc["c", "draw_probability"] == pytest.approx(0.28)
    assert probabilities.loc["a"].tolist() == pytest.approx([0.0, 0.0, 1.0])
    assert probabilities.loc["e"].tolist() == pytest.approx([1.0, 0.0, 0.0])


@pytest.mark.parametrize(
    ("draw_at_even", "draw_shape"),
    [(-0.1, 1.0), (0.51, 1.0), (0.25, 0.0), (0.25, -0.5), (0.25, float("inf"))],
)
def test_score_preserving_1x2_rejects_invalid_parameters(
    draw_at_even: float,
    draw_shape: float,
) -> None:
    with pytest.raises(ValueError):
        score_preserving_1x2_probabilities(
            [0.5],
            draw_at_even=draw_at_even,
            draw_shape=draw_shape,
        )


def test_shape_below_one_raises_tail_draws_without_invalid_probabilities() -> None:
    expected = pd.Series([0.0, 0.01, 0.27, 0.5, 0.86, 0.99, 1.0])

    candidate = score_preserving_1x2_probabilities(
        expected,
        draw_at_even=0.24,
        draw_shape=0.84,
    )
    baseline = score_preserving_1x2_probabilities(
        expected,
        draw_at_even=0.24,
        draw_shape=1.0,
    )

    assert candidate.to_numpy().min() >= 0.0
    assert np.allclose(candidate.sum(axis=1), 1.0)
    assert np.allclose(
        candidate["home_probability"] + 0.5 * candidate["draw_probability"],
        expected,
    )
    assert candidate.loc[3, "draw_probability"] == pytest.approx(0.24)
    assert candidate.loc[2, "draw_probability"] > baseline.loc[2, "draw_probability"]
    assert candidate.loc[4, "draw_probability"] > baseline.loc[4, "draw_probability"]
    assert candidate.loc[1, "draw_probability"] <= 0.02 + 1e-12
    assert candidate.loc[5, "draw_probability"] <= 0.02 + 1e-12


def test_standard_1x2_losses_use_three_class_definitions() -> None:
    probabilities = pd.DataFrame(
        {
            "home_probability": [0.7, 0.1, 0.1],
            "draw_probability": [0.2, 0.8, 0.2],
            "away_probability": [0.1, 0.1, 0.7],
        }
    )

    losses = standard_1x2_losses(
        probabilities,
        home_goals=[2, 1, 0],
        away_goals=[0, 1, 3],
    )

    assert losses["outcome_1x2"].tolist() == ["H", "D", "A"]
    assert losses["brier_1x2"].tolist() == pytest.approx([0.14, 0.06, 0.14])
    assert losses["log_loss_1x2"].tolist() == pytest.approx(
        [-log(0.7), -log(0.8), -log(0.7)]
    )


def test_evaluate_1x2_predictions_combines_probabilities_and_losses() -> None:
    result = evaluate_1x2_predictions(
        [0.6, 0.4],
        [2, 1],
        [0, 1],
        draw_at_even=0.25,
        draw_shape=1.0,
    )

    assert result.columns.tolist() == [
        "home_probability",
        "draw_probability",
        "away_probability",
        "outcome_1x2",
        "brier_1x2",
        "log_loss_1x2",
    ]
    assert result["outcome_1x2"].tolist() == ["H", "D"]


def _ranking_matches() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("A-Strong", "Strong", "A", 0, 1),
            ("B-Weak", "B", "Weak", 1, 0),
            ("Strong-C", "Strong", "C", 2, 0),
            ("D-Strong", "D", "Strong", 0, 2),
            ("Weak-C", "Weak", "C", 0, 2),
            ("D-Weak", "D", "Weak", 2, 0),
        ],
        columns=[
            "match_id",
            "home_team_id",
            "away_team_id",
            "home_goals",
            "away_goals",
        ],
    ).assign(season="2025/26", competition="UCL", is_neutral=False)


def test_schedule_adjustment_rewards_away_result_against_stronger_opponent() -> None:
    performance = schedule_adjusted_team_performance(
        _ranking_matches(),
        home_edge_prior_matches=0.0,
        opponent_prior_matches=1.0,
        team_prior_matches=0.0,
    ).set_index("team_id")

    assert performance.loc["A", "raw_score_rate"] == 1.0
    assert performance.loc["B", "raw_score_rate"] == 1.0
    assert (
        performance.loc["A", "mean_opponent_strength"]
        > performance.loc["B", "mean_opponent_strength"]
    )
    assert (
        performance.loc["A", "schedule_adjusted_score"]
        > performance.loc["B", "schedule_adjusted_score"]
    )


def test_schedule_adjustment_excludes_target_match_from_context_estimates() -> None:
    original = _ranking_matches()
    changed = original.copy()
    changed.loc[changed["match_id"].eq("A-Strong"), ["home_goals", "away_goals"]] = [3, 0]

    original_a = schedule_adjusted_team_performance(original).set_index("team_id").loc["A"]
    changed_a = schedule_adjusted_team_performance(changed).set_index("team_id").loc["A"]

    assert changed_a["leave_team_out_home_edge"] == pytest.approx(
        original_a["leave_team_out_home_edge"]
    )
    assert changed_a["mean_opponent_strength"] == pytest.approx(
        original_a["mean_opponent_strength"]
    )
    assert changed_a["raw_score_rate"] != original_a["raw_score_rate"]


def test_schedule_adjustment_treats_neutral_match_without_venue_correction() -> None:
    matches = pd.DataFrame(
        {
            "season": ["2025/26"],
            "competition": ["UCL"],
            "home_team_id": ["A"],
            "away_team_id": ["B"],
            "home_goals": [1],
            "away_goals": [0],
            "is_neutral": [True],
        }
    )

    performance = schedule_adjusted_team_performance(
        matches,
        opponent_prior_matches=0.0,
        team_prior_matches=0.0,
    ).set_index("team_id")

    assert performance.loc["A", "leave_team_out_home_edge"] == 0.0
    assert performance.loc["A", "venue_neutral_score_rate"] == 1.0
    assert performance.loc["B", "venue_neutral_score_rate"] == 0.0


def _uncertainty_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": [f"m{index}" for index in range(8)],
            "season": ["2024/25"] * 4 + ["2025/26"] * 4,
            "competition": ["UCL", "UCL", "UEL", "UEL"] * 2,
            "tie_id": ["t1", "t1", "t2", "t2", "t3", "t3", "t4", "t4"],
            "home_team_id": ["A", "B", "C", "D", "A", "B", "C", "D"],
            "away_team_id": ["B", "A", "D", "C", "C", "D", "A", "B"],
            "kickoff_utc": pd.date_range("2024-09-01", periods=8, freq="35D", tz="UTC"),
            "loss_difference": [-0.04, -0.03, -0.02, -0.01] * 2,
        }
    )


def test_dependency_robust_bootstrap_is_deterministic_and_conservative() -> None:
    first = dependency_robust_loss_difference_ci(
        _uncertainty_predictions(),
        bootstrap_samples=500,
        seed=17,
    )
    second = dependency_robust_loss_difference_ci(
        _uncertainty_predictions(),
        bootstrap_samples=500,
        seed=17,
    )

    pd.testing.assert_frame_equal(first, second)
    methods = first.loc[first["method"].ne("conservative_envelope")]
    envelope = first.loc[first["method"].eq("conservative_envelope")].iloc[0]
    assert envelope["ci_95_lower"] == pytest.approx(methods["ci_95_lower"].min())
    assert envelope["ci_95_upper"] == pytest.approx(methods["ci_95_upper"].max())
    assert bool(envelope["reliable_improvement"])
    assert not bool(envelope["reliable_harm"])


def test_dependency_bootstrap_rejects_duplicate_match_ids() -> None:
    predictions = _uncertainty_predictions()
    predictions.loc[1, "match_id"] = predictions.loc[0, "match_id"]

    with pytest.raises(ValueError, match="match_id"):
        dependency_robust_loss_difference_ci(
            predictions,
            bootstrap_samples=100,
        )
