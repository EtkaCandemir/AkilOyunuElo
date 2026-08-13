from __future__ import annotations

import math
from dataclasses import dataclass


COMPETITION_BONUS_RATIOS = {
    "UCL": 1.0,
    "UEL": 2.0 / 3.0,
    "UECL": 1.0 / 3.0,
}


@dataclass(frozen=True)
class MatchEloUpdate:
    home_rating_pre: float
    away_rating_pre: float
    effective_rating_difference: float
    expected_home_score: float
    actual_home_score: float
    goal_difference: int
    goal_difference_multiplier: float
    power_delta: float
    home_rating_post: float
    away_rating_post: float
    zero_sum_error: float


@dataclass(frozen=True)
class ProgressionBonusUpdate:
    competition: str
    base_bonus: float
    applied_bonus: float
    winner_rating_pre: float
    loser_rating_pre: float
    winner_rating_post: float
    loser_rating_post: float
    zero_sum_error: float


def calculate_goal_difference_multiplier(
    goal_difference: int,
    effective_rating_difference: float,
    alpha: float,
    tau: float,
    *,
    decided_on_penalties: bool = False,
    is_draw: bool = False,
    goal_cap: int = 4,
) -> float:
    """Return the controlled, favorite-damped goal-difference multiplier."""
    _require_non_negative_integer("goal_difference", goal_difference)
    _require_finite("effective_rating_difference", effective_rating_difference)
    _require_non_negative_finite("alpha", alpha)
    _require_positive_finite("tau", tau)
    _require_positive_integer("goal_cap", goal_cap)
    _require_boolean("decided_on_penalties", decided_on_penalties)
    _require_boolean("is_draw", is_draw)

    if decided_on_penalties or is_draw or goal_difference <= 1 or alpha == 0.0:
        return 1.0

    capped_difference = min(goal_difference, goal_cap)
    signal = math.log(float(capped_difference))
    damping = math.exp(-abs(float(effective_rating_difference)) / float(tau))
    multiplier = 1.0 + float(alpha) * signal * damping
    if not math.isfinite(multiplier):
        raise ValueError("Goal-difference multiplier must be finite")
    return max(1.0, multiplier)


def update_match_elo(
    home_rating: float,
    away_rating: float,
    home_goals: int,
    away_goals: int,
    *,
    k_factor: float,
    elo_scale: float,
    home_advantage: float,
    is_neutral: bool,
    decided_on_penalties: bool,
    alpha: float,
    tau: float,
) -> MatchEloUpdate:
    """Apply one zero-sum Elo update using the controlled GD multiplier."""
    _require_finite("home_rating", home_rating)
    _require_finite("away_rating", away_rating)
    _require_non_negative_integer("home_goals", home_goals)
    _require_non_negative_integer("away_goals", away_goals)
    _require_non_negative_finite("k_factor", k_factor)
    _require_positive_finite("elo_scale", elo_scale)
    _require_non_negative_finite("home_advantage", home_advantage)
    _require_boolean("is_neutral", is_neutral)
    _require_boolean("decided_on_penalties", decided_on_penalties)
    _require_non_negative_finite("alpha", alpha)
    _require_positive_finite("tau", tau)
    advantage = 0.0 if is_neutral else float(home_advantage)
    difference = float(home_rating) - float(away_rating) + advantage
    exponent = -(difference / float(elo_scale))
    if exponent >= 300.0:
        expected = 0.0
    elif exponent <= -300.0:
        expected = 1.0
    else:
        expected = 1.0 / (1.0 + 10.0**exponent)

    field_draw = home_goals == away_goals
    # Shoot-outs settle the tie, not the 90/120-minute field result. They turn
    # off only the GD signal below; a 2-0 second-leg field win remains a win.
    if field_draw:
        actual = 0.5
        goal_difference = 0
    elif home_goals > away_goals:
        actual = 1.0
        goal_difference = home_goals - away_goals
    else:
        actual = 0.0
        goal_difference = away_goals - home_goals

    multiplier = calculate_goal_difference_multiplier(
        goal_difference,
        difference,
        alpha,
        tau,
        decided_on_penalties=decided_on_penalties,
        is_draw=field_draw,
    )
    delta = float(k_factor) * (actual - expected) * multiplier
    home_post = float(home_rating) + delta
    away_post = float(away_rating) - delta
    zero_sum_error = abs(
        (home_post + away_post) - (float(home_rating) + float(away_rating))
    )
    if zero_sum_error > 1e-9:
        raise ValueError("Match Elo update violated the zero-sum invariant")

    return MatchEloUpdate(
        home_rating_pre=float(home_rating),
        away_rating_pre=float(away_rating),
        effective_rating_difference=difference,
        expected_home_score=expected,
        actual_home_score=actual,
        goal_difference=goal_difference,
        goal_difference_multiplier=multiplier,
        power_delta=delta,
        home_rating_post=home_post,
        away_rating_post=away_post,
        zero_sum_error=zero_sum_error,
    )


def apply_progression_bonus(
    winner_rating: float,
    loser_rating: float,
    competition: str,
    base_bonus: float,
) -> ProgressionBonusUpdate:
    """Apply one competition-scaled, zero-sum tie progression adjustment."""
    _require_finite("winner_rating", winner_rating)
    _require_finite("loser_rating", loser_rating)
    _require_non_negative_finite("base_bonus", base_bonus)
    try:
        ratio = COMPETITION_BONUS_RATIOS[competition]
    except KeyError as error:
        raise ValueError(f"Unknown competition: {competition}") from error

    bonus = float(base_bonus) * ratio
    winner_post = float(winner_rating) + bonus
    loser_post = float(loser_rating) - bonus
    zero_sum_error = abs(
        (winner_post + loser_post)
        - (float(winner_rating) + float(loser_rating))
    )
    if zero_sum_error > 1e-9:
        raise ValueError("Progression bonus violated the zero-sum invariant")
    return ProgressionBonusUpdate(
        competition=competition,
        base_bonus=float(base_bonus),
        applied_bonus=bonus,
        winner_rating_pre=float(winner_rating),
        loser_rating_pre=float(loser_rating),
        winner_rating_post=winner_post,
        loser_rating_post=loser_post,
        zero_sum_error=zero_sum_error,
    )


def _require_boolean(name: str, value: bool) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")


def _require_finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


def _require_non_negative_finite(name: str, value: float) -> None:
    _require_finite(name, value)
    if float(value) < 0.0:
        raise ValueError(f"{name} must be non-negative")


def _require_positive_finite(name: str, value: float) -> None:
    _require_finite(name, value)
    if float(value) <= 0.0:
        raise ValueError(f"{name} must be positive")


def _require_non_negative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
