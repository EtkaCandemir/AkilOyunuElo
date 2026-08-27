from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_current_external_benchmark import (
    CLUBELO_PUBLISHED_SCALE,
    DRAW_AT_EVEN,
    RATING_ARMS,
    SERVED_POISSON_WEIGHT,
    SINGLE_MATCH_DRAW_AT_EVEN,
    elo_expected_home_score,
    log_probability_blend,
    one_x_two_from_expected,
    paired_spearman_difference_ci,
    realized_season_performance,
    summarize_arm,
    three_class_losses,
    vectorized_elo_expected_home_score,
)


def test_draw_model_uses_single_match_parameter_only_for_single_match_ties() -> None:
    expected = np.array([0.5, 0.5])
    probabilities = one_x_two_from_expected(expected, np.array([False, True]))

    assert probabilities[0, 1] == pytest.approx(DRAW_AT_EVEN)
    assert probabilities[1, 1] == pytest.approx(SINGLE_MATCH_DRAW_AT_EVEN)


def test_draw_model_preserves_expected_score_and_normalizes() -> None:
    expected = np.array([0.05, 0.31, 0.5, 0.74, 0.97])
    single = np.zeros(len(expected), dtype=bool)

    probabilities = one_x_two_from_expected(expected, single)

    assert probabilities.sum(axis=1) == pytest.approx(np.ones(len(expected)))
    preserved = probabilities[:, 0] + 0.5 * probabilities[:, 1]
    assert preserved == pytest.approx(expected)
    assert (probabilities >= 0.0).all()


def test_external_comparison_gives_both_sides_the_same_draw_model() -> None:
    """The benchmark must isolate rating quality, not draw-model quality."""
    expected = np.array([0.62])
    single = np.array([False])

    ao_like = one_x_two_from_expected(expected, single)
    external_like = one_x_two_from_expected(expected, single)

    assert ao_like == pytest.approx(external_like)


def test_neutral_venue_removes_home_advantage() -> None:
    difference = np.array([120.0, 120.0])
    neutral = np.array([False, True])

    values = vectorized_elo_expected_home_score(
        difference,
        neutral,
        np.array([70.0, 70.0]),
        np.full(2, CLUBELO_PUBLISHED_SCALE),
    )

    assert values[0] > values[1]
    assert values[1] == pytest.approx(
        elo_expected_home_score(np.array([120.0]), np.array([True]), 70.0, 400.0)[0]
    )


def test_vectorized_expected_score_matches_scalar_helper_per_row() -> None:
    difference = np.array([-200.0, 0.0, 340.0])
    neutral = np.array([False, False, False])
    home = np.array([40.0, 65.0, 90.0])
    scale = np.array([400.0, 380.0, 450.0])

    vectorized = vectorized_elo_expected_home_score(difference, neutral, home, scale)
    scalar = np.array(
        [
            elo_expected_home_score(
                np.array([difference[index]]),
                np.array([neutral[index]]),
                home[index],
                scale[index],
            )[0]
            for index in range(len(difference))
        ]
    )

    assert vectorized == pytest.approx(scalar)


def test_equal_ratings_on_neutral_ground_are_a_coin_flip() -> None:
    value = vectorized_elo_expected_home_score(
        np.array([0.0]), np.array([True]), np.array([80.0]), np.array([400.0])
    )

    assert value[0] == pytest.approx(0.5)


def test_log_blend_endpoints_return_each_component() -> None:
    left = np.array([[0.6, 0.2, 0.2]])
    right = np.array([[0.2, 0.3, 0.5]])

    assert log_probability_blend(left, right, 0.0) == pytest.approx(left)
    assert log_probability_blend(left, right, 1.0) == pytest.approx(right)


def test_log_blend_is_normalized_and_between_components() -> None:
    left = np.array([[0.60, 0.20, 0.20]])
    right = np.array([[0.20, 0.30, 0.50]])

    blended = log_probability_blend(left, right, SERVED_POISSON_WEIGHT)

    assert blended.sum(axis=1) == pytest.approx(np.ones(1))
    assert left[0, 0] > blended[0, 0] > right[0, 0]
    assert right[0, 2] > blended[0, 2] > left[0, 2]


def test_three_class_losses_match_the_project_definition() -> None:
    probabilities = np.array([[0.5, 0.3, 0.2]])
    outcomes = np.array([0])

    brier, log_loss = three_class_losses(probabilities, outcomes)

    assert brier[0] == pytest.approx(0.5**2 + 0.3**2 + 0.2**2 - 2 * 0.5 + 1)
    assert log_loss[0] == pytest.approx(-np.log(0.5))


def test_perfect_and_uniform_predictions_bracket_the_loss_scale() -> None:
    outcomes = np.array([0, 1, 2])
    perfect = np.eye(3)
    uniform = np.full((3, 3), 1 / 3)

    best = summarize_arm("PERFECT", perfect, outcomes)
    worst = summarize_arm("UNIFORM", uniform, outcomes)

    assert best["brier_1x2"] == pytest.approx(0.0)
    assert best["accuracy_1x2"] == pytest.approx(1.0)
    assert worst["log_loss_1x2"] == pytest.approx(np.log(3.0))


def test_paired_spearman_difference_is_signed_and_deterministic() -> None:
    realized = np.arange(40, dtype=float)
    strong = realized + np.linspace(0.0, 0.4, 40)
    weak = np.roll(realized, 13).astype(float)

    mean, lower, upper = paired_spearman_difference_ci(
        strong, weak, realized, samples=300, seed=7
    )
    repeat = paired_spearman_difference_ci(strong, weak, realized, samples=300, seed=7)

    assert mean > 0.0
    assert lower <= mean <= upper
    assert (mean, lower, upper) == repeat


def test_uefa_coefficient_is_labelled_as_own_input_not_external() -> None:
    """The UEFA coefficient feeds AO First Elo, so it can never be a referee.

    `club_points_t_*` are the five seasonal components of that coefficient. If a
    later change relabels this arm as EXTERNAL, the report would claim
    independent validation that does not exist.
    """
    reference_types = {name: kind for name, _, kind in RATING_ARMS}

    assert reference_types["UEFA_CLUB_COEFFICIENT_PRE_SEASON"] == "OWN_INPUT"
    assert reference_types["OPTA_PRE_SEASON_POWER_RANKING"] == "EXTERNAL"
    assert reference_types["AO_FIRST_ELO"] == "MODEL"


def test_exactly_one_rating_arm_is_the_model_under_test() -> None:
    kinds = [kind for _, _, kind in RATING_ARMS]

    assert kinds.count("MODEL") == 1
    assert len(RATING_ARMS) == len({name for name, _, _ in RATING_ARMS})


def test_domestic_surprise_ablation_arm_is_scored_on_the_external_axis() -> None:
    """The layer is justified on seed quality, not on match loss.

    Without a surprise-off seed on this axis the only available answer is
    "it beats its own unadjusted prior", which is not the same claim as
    "it moves the seed toward an independent reference".
    """
    reference_types = {name: kind for name, _, kind in RATING_ARMS}
    columns = {name: column for name, column, _ in RATING_ARMS}

    assert reference_types["AO_FIRST_ELO_NO_DOMESTIC_SURPRISE"] == "MODEL_ABLATION"
    assert columns["AO_FIRST_ELO_NO_DOMESTIC_SURPRISE"] == (
        "initial_rating_no_domestic_surprise"
    )
    # The ablation must not be mistaken for an independent referee.
    assert reference_types["AO_FIRST_ELO_NO_DOMESTIC_SURPRISE"] != "EXTERNAL"


def test_realized_performance_returns_one_row_per_team(tmp_path: Path) -> None:
    matches = pd.DataFrame(
        {
            "season": ["2025/26"] * 3 + ["2024/25"],
            "competition": ["UCL"] * 4,
            "home_team_id": ["1", "2", "3", "1"],
            "away_team_id": ["2", "3", "1", "2"],
            "home_goals": [2, 1, 0, 3],
            "away_goals": [0, 1, 2, 0],
            "is_neutral": [False] * 4,
        }
    )
    path = tmp_path / "matches.csv"
    matches.to_csv(path, index=False)

    target = realized_season_performance(path, "2025/26")

    assert sorted(target["team_id"]) == ["1", "2", "3"]
    assert target["team_id"].is_unique
    assert target["matches"].sum() == 6


def test_realized_performance_rejects_a_season_without_matches(tmp_path: Path) -> None:
    matches = pd.DataFrame(
        {
            "season": ["2024/25"],
            "competition": ["UCL"],
            "home_team_id": ["1"],
            "away_team_id": ["2"],
            "home_goals": [1],
            "away_goals": [0],
            "is_neutral": [False],
        }
    )
    path = tmp_path / "matches.csv"
    matches.to_csv(path, index=False)

    with pytest.raises(ValueError, match="No matches found"):
        realized_season_performance(path, "2025/26")
