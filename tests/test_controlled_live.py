import math
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ao_elo.controlled_live import (
    apply_progression_bonus,
    calculate_goal_difference_multiplier,
    update_match_elo,
)
from scripts.run_controlled_goal_progression_backtest import (
    evaluate_predictions,
    reporting_candidates,
    selection_candidates,
)


def test_goal_difference_multiplier_has_diminishing_returns_and_cap() -> None:
    values = [
        calculate_goal_difference_multiplier(
            difference,
            0.0,
            0.10,
            300.0,
        )
        for difference in (1, 2, 3, 4, 5, 6)
    ]
    assert values[0] == pytest.approx(1.0)
    assert values[1] == pytest.approx(1.0 + 0.10 * math.log(2.0))
    assert values[2] == pytest.approx(1.0 + 0.10 * math.log(3.0))
    assert values[3] == pytest.approx(1.0 + 0.10 * math.log(4.0))
    assert values[4] == pytest.approx(values[3])
    assert values[5] == pytest.approx(values[3])
    increments = [values[index + 1] - values[index] for index in range(3)]
    assert increments[0] > increments[1] > increments[2] > 0.0


def test_rating_difference_damps_goal_signal() -> None:
    balanced = calculate_goal_difference_multiplier(4, 0.0, 0.10, 300.0)
    moderate = calculate_goal_difference_multiplier(4, 300.0, 0.10, 300.0)
    strong = calculate_goal_difference_multiplier(4, 900.0, 0.10, 300.0)
    assert balanced > moderate > strong > 1.0


@pytest.mark.parametrize(
    ("difference", "penalties", "draw"),
    [(0, False, True), (3, True, False), (1, False, False)],
)
def test_draw_penalty_and_one_goal_never_receive_multiplier(
    difference: int,
    penalties: bool,
    draw: bool,
) -> None:
    assert calculate_goal_difference_multiplier(
        difference,
        0.0,
        0.20,
        150.0,
        decided_on_penalties=penalties,
        is_draw=draw,
    ) == pytest.approx(1.0)


def test_multiplier_never_falls_below_one() -> None:
    for difference in range(8):
        for rating_difference in (-2000.0, -300.0, 0.0, 300.0, 2000.0):
            assert (
                calculate_goal_difference_multiplier(
                    difference,
                    rating_difference,
                    0.20,
                    150.0,
                    is_draw=difference == 0,
                )
                >= 1.0
            )


def test_match_update_uses_home_advantage_inside_damping_difference() -> None:
    update = update_match_elo(
        1500.0,
        1500.0,
        3,
        0,
        k_factor=100.0,
        elo_scale=800.0,
        home_advantage=150.0,
        is_neutral=False,
        decided_on_penalties=False,
        alpha=0.10,
        tau=300.0,
    )
    expected_multiplier = 1.0 + 0.10 * math.log(3.0) * math.exp(-150.0 / 300.0)
    assert update.effective_rating_difference == pytest.approx(150.0)
    assert update.goal_difference_multiplier == pytest.approx(expected_multiplier)
    assert update.zero_sum_error <= 1e-9


def test_penalty_decision_forces_draw_score_and_no_goal_signal() -> None:
    update = update_match_elo(
        1500.0,
        1500.0,
        3,
        3,
        k_factor=100.0,
        elo_scale=800.0,
        home_advantage=0.0,
        is_neutral=True,
        decided_on_penalties=True,
        alpha=0.20,
        tau=150.0,
    )
    assert update.actual_home_score == pytest.approx(0.5)
    assert update.goal_difference == 0
    assert update.goal_difference_multiplier == pytest.approx(1.0)
    assert update.power_delta == pytest.approx(0.0)


def test_penalty_decider_preserves_non_draw_field_result_but_disables_goal_bonus() -> None:
    update = update_match_elo(
        1500.0,
        1500.0,
        2,
        0,
        k_factor=100.0,
        elo_scale=800.0,
        home_advantage=0.0,
        is_neutral=True,
        decided_on_penalties=True,
        alpha=0.20,
        tau=150.0,
    )

    assert update.actual_home_score == pytest.approx(1.0)
    assert update.goal_difference == 2
    assert update.goal_difference_multiplier == pytest.approx(1.0)
    assert update.power_delta == pytest.approx(50.0)


def test_progression_bonus_keeps_exact_competition_ratios_and_zero_sum() -> None:
    ucl = apply_progression_bonus(1600.0, 1500.0, "UCL", 6.0)
    uel = apply_progression_bonus(1600.0, 1500.0, "UEL", 6.0)
    uecl = apply_progression_bonus(1600.0, 1500.0, "UECL", 6.0)
    assert ucl.applied_bonus == pytest.approx(6.0)
    assert uel.applied_bonus == pytest.approx(4.0)
    assert uecl.applied_bonus == pytest.approx(2.0)
    for update in (ucl, uel, uecl):
        assert update.zero_sum_error <= 1e-9
        assert update.winner_rating_post + update.loser_rating_post == pytest.approx(
            update.winner_rating_pre + update.loser_rating_pre
        )


def test_candidate_grids_cover_requested_values() -> None:
    reporting = reporting_candidates()
    assert len(reporting) == 208
    assert len(selection_candidates("PROGRESSION_ONLY")) == 7
    assert len(selection_candidates("GOAL_DIFFERENCE_ONLY")) == 21
    assert len(selection_candidates("FULL_MODEL")) == 147
    assert any(
        candidate.alpha == 0.10 and candidate.tau == 300.0
        for candidate in reporting
    )


def test_prediction_evaluation_returns_standard_multiclass_metrics() -> None:
    predictions = pd.DataFrame(
        {
            "brier_1x2": [0.14, 0.54],
            "log_loss_1x2": [-math.log(0.7), -math.log(0.4)],
            "actual_class": [0, 1],
            "predicted_class": [0, 0],
            "home_probability": [0.7, 0.4],
            "draw_probability": [0.2, 0.3],
            "away_probability": [0.1, 0.3],
        }
    )
    metrics = evaluate_predictions(predictions)
    assert metrics["matches"] == 2
    assert metrics["brier_1x2"] == pytest.approx(0.34)
    assert metrics["accuracy_1x2"] == pytest.approx(0.5)
    assert 0.0 <= metrics["multiclass_ece"] <= 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"goal_difference": -1},
        {"effective_rating_difference": math.inf},
        {"alpha": -0.1},
        {"tau": 0.0},
    ],
)
def test_goal_multiplier_rejects_invalid_parameters(kwargs: dict[str, float]) -> None:
    values = {
        "goal_difference": 2,
        "effective_rating_difference": 0.0,
        "alpha": 0.1,
        "tau": 300.0,
    }
    values.update(kwargs)
    with pytest.raises(ValueError):
        calculate_goal_difference_multiplier(**values)
