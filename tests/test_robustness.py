from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ao_elo.robustness import (  # noqa: E402
    CompetitionKCandidate,
    GoalMarginCandidate,
    baseline_competition_k,
    competition_k_candidates,
    goal_margin_candidates,
    goal_margin_multiplier,
    one_x_two_probabilities_scalar,
    standard_1x2_loss_scalar,
)
from scripts.run_final_robustness import (  # noqa: E402
    select_ranking_first,
    summarize_ranking,
)


def test_log_goal_margin_has_diminishing_incremental_returns() -> None:
    config = GoalMarginCandidate("LOG", 0.25, 2.0)
    values = [goal_margin_multiplier(difference, 0.5, config) for difference in range(1, 6)]
    increments = [right - left for left, right in zip(values, values[1:])]

    assert values[0] == 1.0
    assert values == sorted(values)
    assert increments == sorted(increments, reverse=True)
    assert values[3] == pytest.approx(1.0 + 0.25 * 1.3862943611198906)


def test_one_goal_and_draw_never_receive_goal_margin_bonus() -> None:
    for candidate in goal_margin_candidates():
        assert goal_margin_multiplier(0, 0.5, candidate) == 1.0
        assert goal_margin_multiplier(1, 0.9, candidate) == 1.0


def test_favorite_damping_reduces_only_favorite_blowout_bonus() -> None:
    plain = GoalMarginCandidate("LOG", 0.5, 2.0)
    damped = GoalMarginCandidate("FAVORITE_DAMPED_LOG", 0.5, 2.0, 1.0)

    assert goal_margin_multiplier(4, 0.8, damped) < goal_margin_multiplier(4, 0.8, plain)
    assert goal_margin_multiplier(4, 0.3, damped) == pytest.approx(
        goal_margin_multiplier(4, 0.3, plain)
    )


def test_goal_margin_grid_is_unique_and_contains_baseline() -> None:
    candidates = goal_margin_candidates()

    assert len(candidates) == 101
    assert len(set(candidates)) == len(candidates)
    assert sum(not candidate.active for candidate in candidates) == 1


def test_competition_k_grid_jointly_recalibrates_global_k() -> None:
    candidates = competition_k_candidates()

    assert baseline_competition_k() in candidates
    assert CompetitionKCandidate("HIERARCHY", 1.0, 0.65, 0.45) in candidates
    assert any(
        candidate.profile == "GLOBAL" and candidate.ucl_multiplier == 1.25
        for candidate in candidates
    )
    assert all(
        candidate.profile == "GLOBAL"
        or candidate.ucl_multiplier > candidate.uel_multiplier > candidate.uecl_multiplier
        for candidate in candidates
    )


def test_score_preserving_1x2_scalar_contract() -> None:
    home, draw, away = one_x_two_probabilities_scalar(0.63, 0.24, 1.0)

    assert home + draw + away == pytest.approx(1.0)
    assert home + 0.5 * draw == pytest.approx(0.63)
    brier, log_loss = standard_1x2_loss_scalar((home, draw, away), 2, 1)
    assert 0.0 <= brier <= 2.0
    assert log_loss > 0.0


def test_ranking_first_selection_will_not_trade_ranking_for_brier() -> None:
    metrics = pd.DataFrame(
        [
            {
                "candidate_key": "baseline",
                "start_end_rank_correlation": 0.90,
                "max_abs_rating_change": 100.0,
                "max_pair_sum_error": 0.0,
                "ranking_score": 0.70,
                "pairwise_accuracy": 0.72,
                "brier_1x2": 0.58,
                "log_loss_1x2": 0.98,
            },
            {
                "candidate_key": "better_brier_worse_rank",
                "start_end_rank_correlation": 0.90,
                "max_abs_rating_change": 100.0,
                "max_pair_sum_error": 0.0,
                "ranking_score": 0.69,
                "pairwise_accuracy": 0.71,
                "brier_1x2": 0.55,
                "log_loss_1x2": 0.94,
            },
        ]
    )

    selected = select_ranking_first(metrics, "baseline")

    assert selected["candidate_key"] == "baseline"


def test_dynamic_ranking_uses_only_following_season_target() -> None:
    end_ratings = pd.DataFrame(
        {
            "season": ["2022/23"] * 3,
            "team_id": [1, 2, 3],
            "end_live_rating": [1900.0, 1700.0, 1500.0],
        }
    )
    target = pd.DataFrame(
        {
            "season": ["2022/23"] * 3 + ["2023/24"] * 3,
            "competition": ["UCL"] * 6,
            "team_id": [1, 2, 3, 1, 2, 3],
            # Same-season order is reversed; next-season order matches ratings.
            "schedule_adjusted_score": [0.1, 0.5, 0.9, 0.9, 0.5, 0.1],
        }
    )

    ranking = summarize_ranking(
        end_ratings,
        target,
        allowed_target_seasons={"2022/23", "2023/24"},
    )

    assert ranking.loc[ranking["competition"].eq("ALL"), "ranking_score"].item() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "candidate",
    [
        GoalMarginCandidate("UNKNOWN", 0.25, 1.5),
        GoalMarginCandidate("LOG", 0.0, 1.5),
        GoalMarginCandidate("SQRT", 0.25, 1.0),
        GoalMarginCandidate("LOG", 0.25, 1.5, 1.0),
    ],
)
def test_invalid_goal_margin_configs_are_rejected(
    candidate: GoalMarginCandidate,
) -> None:
    with pytest.raises(ValueError):
        candidate.validate()
