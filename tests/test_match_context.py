import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ao_elo.match_context import (
    AggregateStateConfig,
    DomesticRegressionConfig,
    DrawContextConfig,
    HomeAdvantageProfile,
    apply_aggregate_state,
    domestic_anchored_start_rating,
    effective_home_advantage,
)
from scripts.run_match_context_backtest import (
    derive_aggregate_state,
    layer_guardrails,
    ranking_difference_uncertainty,
)


def test_aggregate_state_reduces_match_expectation_for_aggregate_leader():
    config = AggregateStateConfig(50.0, 3)
    assert apply_aggregate_state(100.0, 2, is_second_leg=True, config=config) == 0.0
    assert apply_aggregate_state(100.0, -2, is_second_leg=True, config=config) == 200.0
    assert apply_aggregate_state(100.0, 4, is_second_leg=True, config=config) == -50.0
    assert apply_aggregate_state(100.0, 2, is_second_leg=False, config=config) == 100.0


def test_home_profile_preserves_neutral_zero_and_combines_context():
    profile = HomeAdvantageProfile(1.0, 0.8, 1.2, 0.75, 0.5)
    assert effective_home_advantage(
        100.0,
        "UEL",
        is_neutral=False,
        is_knockout=True,
        is_second_leg=True,
        profile=profile,
    ) == pytest.approx(30.0)
    assert effective_home_advantage(
        100.0,
        "UECL",
        is_neutral=True,
        is_knockout=True,
        is_second_leg=True,
        profile=profile,
    ) == 0.0


def test_draw_context_adds_only_relevant_offsets():
    config = DrawContextConfig(0.24, 1.25, 0.01, -0.01, 0.02, 0.03)
    config.validate()
    assert config.for_match(
        "UCL", is_knockout=False, is_second_leg=False
    ) == pytest.approx(0.24)
    assert config.for_match(
        "UEL", is_knockout=True, is_second_leg=True
    ) == pytest.approx(0.30)


def test_draw_context_rejects_out_of_range_combination():
    with pytest.raises(ValueError, match="must remain"):
        DrawContextConfig(0.48, 1.0, 0.03, 0.0, 0.02, 0.02).validate()


def test_domestic_regression_preserves_new_team_and_baseline():
    baseline = DomesticRegressionConfig()
    active = DomesticRegressionConfig("DOMESTIC_ANCHORED", 0.5)
    assert domestic_anchored_start_rating(1400.0, 1200.0, 1600.0, baseline) == 1400.0
    assert domestic_anchored_start_rating(1400.0, 1200.0, None, active) == 1400.0
    assert domestic_anchored_start_rating(1400.0, 1200.0, 1600.0, active) == 1400.0


@pytest.mark.parametrize(
    "config",
    [
        AggregateStateConfig(-1.0, 3),
        AggregateStateConfig(1.0, 0),
    ],
)
def test_invalid_aggregate_configs_are_rejected(config):
    with pytest.raises(ValueError):
        config.validate()


def test_aggregate_state_is_derived_without_second_leg_lookahead():
    events = pd.DataFrame(
        [
            {
                "season": "2024/25",
                "event_order": 1,
                "is_knockout": True,
                "tie_id": "tie-1",
                "home_team_id": 10,
                "away_team_id": 20,
                "home_goals": 2,
                "away_goals": 0,
            },
            {
                "season": "2024/25",
                "event_order": 2,
                "is_knockout": True,
                "tie_id": "tie-1",
                "home_team_id": 20,
                "away_team_id": 10,
                "home_goals": 1,
                "away_goals": 0,
            },
        ]
    )
    result = derive_aggregate_state(events)
    assert result["aggregate_home_goal_difference_before"].tolist() == [0, -2]
    assert result["is_second_leg_derived"].tolist() == [False, True]


def test_ranking_gate_ignores_one_unreliable_fold_regression():
    fold_rows = []
    for fold in range(1, 7):
        for model in ("candidate", "baseline"):
            fold_rows.append(
                {
                    "fold": fold,
                    "model": model,
                    "brier_1x2": 0.49 if model == "candidate" else 0.50,
                    "log_loss_1x2": 0.69 if model == "candidate" else 0.70,
                    "ranking_score": (
                        float("nan")
                        if fold == 6
                        else 0.499
                        if fold == 1 and model == "candidate"
                        else 0.50
                    ),
                    "pairwise_accuracy": (
                        float("nan")
                        if fold == 6
                        else 0.499
                        if fold == 1 and model == "candidate"
                        else 0.50
                    ),
                    "max_pair_sum_error": 0.0,
                }
            )
    competition = pd.DataFrame(
        [
            {"competition": competition, "brier_difference": -0.01, "log_loss_difference": -0.01}
            for competition in ("ALL", "UCL", "UEL", "UECL")
        ]
    )
    uncertainty = pd.DataFrame(
        [{"metric": "brier_1x2", "ci_95_upper": -0.001}]
    )
    guardrails = layer_guardrails(
        pd.DataFrame(fold_rows),
        pd.DataFrame(),
        competition,
        uncertainty,
    )
    assert guardrails["ranking_evaluable_folds"] == 5
    assert guardrails["ranking_no_regression_folds"] == 4
    assert guardrails["ranking_reliable_harm"] is False
    assert guardrails["ranking_gate"] is True


def test_ranking_gate_rejects_dependency_robust_harm():
    fold_rows = []
    for fold in range(1, 7):
        for model in ("candidate", "baseline"):
            fold_rows.append(
                {
                    "fold": fold,
                    "model": model,
                    "brier_1x2": 0.49 if model == "candidate" else 0.50,
                    "log_loss_1x2": 0.69 if model == "candidate" else 0.70,
                    "ranking_score": (
                        float("nan")
                        if fold == 6
                        else 0.48
                        if model == "candidate"
                        else 0.50
                    ),
                    "pairwise_accuracy": (
                        float("nan")
                        if fold == 6
                        else 0.48
                        if model == "candidate"
                        else 0.50
                    ),
                    "max_pair_sum_error": 0.0,
                }
            )
    fold_results = pd.DataFrame(fold_rows)
    ranking_uncertainty = ranking_difference_uncertainty(
        fold_results, bootstrap_samples=500
    )
    competition = pd.DataFrame(
        [
            {
                "competition": competition,
                "brier_difference": -0.01,
                "log_loss_difference": -0.01,
            }
            for competition in ("ALL", "UCL", "UEL", "UECL")
        ]
    )
    uncertainty = pd.DataFrame(
        [{"metric": "brier_1x2", "ci_95_upper": -0.001}]
    )
    guardrails = layer_guardrails(
        fold_results,
        pd.DataFrame(),
        competition,
        uncertainty,
        ranking_uncertainty,
    )
    assert guardrails["ranking_reliable_harm"] is True
    assert guardrails["ranking_gate"] is False
