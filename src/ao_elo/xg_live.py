from __future__ import annotations

import math
from dataclasses import dataclass

from ao_elo.controlled_live import calculate_goal_difference_multiplier


@dataclass(frozen=True, order=True)
class XGBlendConfig:
    rho: float
    xg_scale: float
    mode: str = "CONVEX_BLEND"

    def validate(self) -> None:
        _require_between_zero_and_one("rho", self.rho)
        _require_positive_finite("xg_scale", self.xg_scale)
        if self.mode not in {
            "CONVEX_BLEND",
            "ADDITIVE_LUCK_CORRECTION",
            "DIRECTION_PRESERVING_LUCK_CORRECTION",
        }:
            raise ValueError(
                "mode must be CONVEX_BLEND, ADDITIVE_LUCK_CORRECTION, "
                "or DIRECTION_PRESERVING_LUCK_CORRECTION"
            )

    @property
    def key(self) -> str:
        mode = {
            "CONVEX_BLEND": "blend",
            "ADDITIVE_LUCK_CORRECTION": "luck",
            "DIRECTION_PRESERVING_LUCK_CORRECTION": "bounded_luck",
        }[self.mode]
        return f"{mode}_rho{self.rho:g}_cxg{self.xg_scale:g}"


@dataclass(frozen=True, order=True)
class UnsupportedMarginConfig:
    tolerance: float
    penalty_lambda: float
    minimum_multiplier: float
    application_scope: str = "FULL_UPDATE"

    def validate(self) -> None:
        _require_non_negative_finite("tolerance", self.tolerance)
        _require_non_negative_finite("penalty_lambda", self.penalty_lambda)
        _require_between_zero_and_one(
            "minimum_multiplier", self.minimum_multiplier
        )
        if self.application_scope not in {"FULL_UPDATE", "GOAL_BONUS_ONLY"}:
            raise ValueError(
                "application_scope must be FULL_UPDATE or GOAL_BONUS_ONLY"
            )

    @property
    def key(self) -> str:
        return (
            f"unsupported_tol{self.tolerance:g}_lambda{self.penalty_lambda:g}"
            f"_min{self.minimum_multiplier:g}_{self.application_scope.lower()}"
        )


@dataclass(frozen=True)
class UnsupportedMarginPenalty:
    actual_goal_margin: int
    winner_xg_advantage: float | None
    unsupported_margin: float
    penalty_multiplier: float


@dataclass(frozen=True, order=True)
class XGPerformanceBonusConfig:
    beta: float
    xg_scale: float
    minimum_winner_gain_ratio: float

    def validate(self) -> None:
        _require_non_negative_finite("beta", self.beta)
        _require_positive_finite("xg_scale", self.xg_scale)
        _require_between_zero_and_one(
            "minimum_winner_gain_ratio", self.minimum_winner_gain_ratio
        )
        if self.minimum_winner_gain_ratio <= 0.0:
            raise ValueError("minimum_winner_gain_ratio must be positive")

    @property
    def key(self) -> str:
        return (
            f"xg_performance_beta{self.beta:g}_scale{self.xg_scale:g}"
            f"_floor{self.minimum_winner_gain_ratio:g}"
        )


@dataclass(frozen=True)
class XGMatchEloUpdate:
    home_rating_pre: float
    away_rating_pre: float
    effective_rating_difference: float
    expected_home_score: float
    actual_home_score: float
    goal_difference: int
    goal_difference_multiplier: float
    xg_available: bool
    xg_home: float | None
    xg_away: float | None
    xg_home_score: float | None
    result_residual: float
    base_result_residual: float
    goal_bonus_residual: float
    adjusted_goal_bonus_residual: float
    xg_residual: float | None
    blended_residual: float
    power_delta: float
    home_rating_post: float
    away_rating_post: float
    zero_sum_error: float
    winner_xg_advantage: float | None
    unsupported_margin: float
    xg_penalty_multiplier: float
    xg_performance_signal: float | None
    xg_performance_adjustment: float
    direction_floor_residual: float | None


def calculate_xg_performance_score(
    xg_home: float,
    xg_away: float,
    xg_scale: float,
) -> float:
    """Map post-match xG difference to a bounded home performance score."""
    _require_non_negative_finite("xg_home", xg_home)
    _require_non_negative_finite("xg_away", xg_away)
    _require_positive_finite("xg_scale", xg_scale)
    exponent = -(float(xg_home) - float(xg_away)) / float(xg_scale)
    if exponent >= 700.0:
        return 0.0
    if exponent <= -700.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(exponent))


def calculate_unsupported_margin_penalty(
    home_goals: int,
    away_goals: int,
    xg_home: float,
    xg_away: float,
    *,
    goal_difference_cap: int,
    config: UnsupportedMarginConfig,
    decided_on_penalties: bool = False,
) -> UnsupportedMarginPenalty:
    """Reduce only the score margin that is unsupported by the winner's xG."""
    _require_non_negative_integer("home_goals", home_goals)
    _require_non_negative_integer("away_goals", away_goals)
    _require_non_negative_finite("xg_home", xg_home)
    _require_non_negative_finite("xg_away", xg_away)
    _require_positive_integer("goal_difference_cap", goal_difference_cap)
    _require_boolean("decided_on_penalties", decided_on_penalties)
    config.validate()

    field_draw = home_goals == away_goals
    if decided_on_penalties or field_draw:
        return UnsupportedMarginPenalty(0, None, 0.0, 1.0)

    result_direction = 1.0 if home_goals > away_goals else -1.0
    actual_margin = min(abs(home_goals - away_goals), goal_difference_cap)
    winner_xg_advantage = result_direction * (float(xg_home) - float(xg_away))
    unsupported_margin = max(
        0.0,
        float(actual_margin)
        - winner_xg_advantage
        - float(config.tolerance),
    )
    multiplier = max(
        float(config.minimum_multiplier),
        1.0
        - float(config.penalty_lambda) * math.log1p(unsupported_margin),
    )
    if not 0.0 <= multiplier <= 1.0:
        raise ValueError("Unsupported-margin multiplier must remain in [0, 1]")
    return UnsupportedMarginPenalty(
        actual_goal_margin=actual_margin,
        winner_xg_advantage=winner_xg_advantage,
        unsupported_margin=unsupported_margin,
        penalty_multiplier=multiplier,
    )


def calculate_xg_performance_adjustment(
    xg_home: float,
    xg_away: float,
    base_result_residual: float,
    *,
    config: XGPerformanceBonusConfig,
) -> tuple[float, float]:
    """Return a signed xG signal and residual-scale performance adjustment."""
    _require_non_negative_finite("xg_home", xg_home)
    _require_non_negative_finite("xg_away", xg_away)
    _require_finite("base_result_residual", base_result_residual)
    config.validate()
    signal = math.tanh(
        (float(xg_home) - float(xg_away)) / float(config.xg_scale)
    )
    adjustment = (
        float(config.beta) * abs(float(base_result_residual)) * signal
    )
    return signal, adjustment


def update_match_elo_with_xg(
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
    goal_difference_enabled: bool,
    goal_alpha: float,
    goal_tau: float,
    goal_difference_cap: int,
    xg_config: XGBlendConfig,
    xg_home: float | None = None,
    xg_away: float | None = None,
    unsupported_margin_config: UnsupportedMarginConfig | None = None,
    xg_performance_bonus_config: XGPerformanceBonusConfig | None = None,
) -> XGMatchEloUpdate:
    """Apply one zero-sum Elo update with optional GD and xG evidence."""
    _require_finite("home_rating", home_rating)
    _require_finite("away_rating", away_rating)
    _require_non_negative_integer("home_goals", home_goals)
    _require_non_negative_integer("away_goals", away_goals)
    _require_non_negative_finite("k_factor", k_factor)
    _require_positive_finite("elo_scale", elo_scale)
    _require_non_negative_finite("home_advantage", home_advantage)
    _require_boolean("is_neutral", is_neutral)
    _require_boolean("decided_on_penalties", decided_on_penalties)
    _require_boolean("goal_difference_enabled", goal_difference_enabled)
    _require_non_negative_finite("goal_alpha", goal_alpha)
    _require_positive_finite("goal_tau", goal_tau)
    _require_positive_integer("goal_difference_cap", goal_difference_cap)
    xg_config.validate()
    if unsupported_margin_config is not None:
        unsupported_margin_config.validate()
        if xg_config.rho != 0.0:
            raise ValueError(
                "Unsupported-margin mode cannot be combined with xG blending"
            )
    if xg_performance_bonus_config is not None:
        xg_performance_bonus_config.validate()
        if xg_config.rho != 0.0 or unsupported_margin_config is not None:
            raise ValueError(
                "xG performance bonus cannot be combined with another xG mode"
            )
    if not goal_difference_enabled and goal_alpha != 0.0:
        raise ValueError("Disabled goal difference requires goal_alpha=0")

    xg_available = xg_home is not None or xg_away is not None
    if (xg_home is None) != (xg_away is None):
        raise ValueError("xg_home and xg_away must be provided together")
    if xg_available:
        assert xg_home is not None and xg_away is not None
        _require_non_negative_finite("xg_home", xg_home)
        _require_non_negative_finite("xg_away", xg_away)
    xg_adjustment_eligible = xg_available and not decided_on_penalties

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
        goal_alpha if goal_difference_enabled else 0.0,
        goal_tau,
        decided_on_penalties=decided_on_penalties,
        is_draw=field_draw,
        goal_cap=goal_difference_cap,
    )
    base_result_residual = actual - expected
    result_residual = base_result_residual * multiplier
    goal_bonus_residual = result_residual - base_result_residual
    xg_score: float | None = None
    xg_residual: float | None = None
    if xg_available:
        assert xg_home is not None and xg_away is not None
        xg_score = calculate_xg_performance_score(
            xg_home,
            xg_away,
            xg_config.xg_scale,
        )
        xg_residual = xg_score - expected

    winner_xg_advantage: float | None = None
    unsupported_margin = 0.0
    xg_penalty_multiplier = 1.0
    xg_performance_signal: float | None = None
    xg_performance_adjustment = 0.0
    direction_floor_residual: float | None = None
    if (
        xg_performance_bonus_config is not None
        and xg_adjustment_eligible
        and actual != 0.5
    ):
        assert xg_home is not None and xg_away is not None
        (
            xg_performance_signal,
            xg_performance_adjustment,
        ) = calculate_xg_performance_adjustment(
            xg_home,
            xg_away,
            base_result_residual,
            config=xg_performance_bonus_config,
        )
        raw_blended = result_residual + xg_performance_adjustment
        direction_floor_residual = (
            float(xg_performance_bonus_config.minimum_winner_gain_ratio)
            * base_result_residual
        )
        if base_result_residual > 0.0:
            blended = max(direction_floor_residual, raw_blended)
        else:
            blended = min(direction_floor_residual, raw_blended)
    elif unsupported_margin_config is not None and xg_adjustment_eligible:
        assert xg_home is not None and xg_away is not None
        penalty = calculate_unsupported_margin_penalty(
            home_goals,
            away_goals,
            xg_home,
            xg_away,
            goal_difference_cap=goal_difference_cap,
            config=unsupported_margin_config,
            decided_on_penalties=decided_on_penalties,
        )
        winner_xg_advantage = penalty.winner_xg_advantage
        unsupported_margin = penalty.unsupported_margin
        xg_penalty_multiplier = penalty.penalty_multiplier
        if unsupported_margin_config.application_scope == "GOAL_BONUS_ONLY":
            if math.isclose(goal_bonus_residual, 0.0, abs_tol=1e-15):
                xg_penalty_multiplier = 1.0
            blended = (
                base_result_residual
                + goal_bonus_residual * xg_penalty_multiplier
            )
        else:
            blended = result_residual * xg_penalty_multiplier
    elif xg_config.mode in {
        "ADDITIVE_LUCK_CORRECTION",
        "DIRECTION_PRESERVING_LUCK_CORRECTION",
    }:
        rho = xg_config.rho if xg_adjustment_eligible else 0.0
        blended = result_residual
        if xg_score is not None:
            blended += rho * (xg_score - actual)
        if xg_config.mode == "DIRECTION_PRESERVING_LUCK_CORRECTION":
            if result_residual > 0.0:
                blended = max(0.0, blended)
            elif result_residual < 0.0:
                blended = min(0.0, blended)
            else:
                blended = 0.0
    else:
        rho = xg_config.rho if xg_adjustment_eligible else 0.0
        blended = (1.0 - rho) * result_residual
        if xg_residual is not None:
            blended += rho * xg_residual
    delta = float(k_factor) * blended
    adjusted_goal_bonus_residual = blended - base_result_residual
    home_post = float(home_rating) + delta
    away_post = float(away_rating) - delta
    zero_sum_error = abs(
        (home_post + away_post) - (float(home_rating) + float(away_rating))
    )
    if zero_sum_error > 1e-9:
        raise ValueError("xG Elo update violated the zero-sum invariant")

    return XGMatchEloUpdate(
        home_rating_pre=float(home_rating),
        away_rating_pre=float(away_rating),
        effective_rating_difference=difference,
        expected_home_score=expected,
        actual_home_score=actual,
        goal_difference=goal_difference,
        goal_difference_multiplier=multiplier,
        xg_available=xg_available,
        xg_home=None if xg_home is None else float(xg_home),
        xg_away=None if xg_away is None else float(xg_away),
        xg_home_score=xg_score,
        result_residual=result_residual,
        base_result_residual=base_result_residual,
        goal_bonus_residual=goal_bonus_residual,
        adjusted_goal_bonus_residual=adjusted_goal_bonus_residual,
        xg_residual=xg_residual,
        blended_residual=blended,
        power_delta=delta,
        home_rating_post=home_post,
        away_rating_post=away_post,
        zero_sum_error=zero_sum_error,
        winner_xg_advantage=winner_xg_advantage,
        unsupported_margin=unsupported_margin,
        xg_penalty_multiplier=xg_penalty_multiplier,
        xg_performance_signal=xg_performance_signal,
        xg_performance_adjustment=xg_performance_adjustment,
        direction_floor_residual=direction_floor_residual,
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


def _require_between_zero_and_one(name: str, value: float) -> None:
    _require_finite(name, value)
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


def _require_non_negative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
