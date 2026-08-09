from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, order=True)
class TeamVenueContextConfig:
    lookback_seasons: int
    season_decay: float
    shrinkage_matches: float
    max_team_effect: float
    max_context_offset: float
    minimum_home_advantage: float = 0.0
    maximum_home_advantage: float = 300.0

    def validate(self) -> None:
        if (
            isinstance(self.lookback_seasons, bool)
            or not isinstance(self.lookback_seasons, int)
            or self.lookback_seasons <= 0
        ):
            raise ValueError("lookback_seasons must be a positive integer")
        _require_probability("season_decay", self.season_decay, positive=True)
        _require_non_negative_finite("shrinkage_matches", self.shrinkage_matches)
        _require_non_negative_finite("max_team_effect", self.max_team_effect)
        _require_non_negative_finite("max_context_offset", self.max_context_offset)
        _require_non_negative_finite(
            "minimum_home_advantage", self.minimum_home_advantage
        )
        _require_non_negative_finite(
            "maximum_home_advantage", self.maximum_home_advantage
        )
        if self.maximum_home_advantage < self.minimum_home_advantage:
            raise ValueError(
                "maximum_home_advantage must be >= minimum_home_advantage"
            )

    @property
    def key(self) -> str:
        return (
            f"venue_w{self.lookback_seasons}_d{self.season_decay:g}"
            f"_k{self.shrinkage_matches:g}_team{self.max_team_effect:g}"
            f"_ctx{self.max_context_offset:g}"
        )


@dataclass(frozen=True)
class TeamVenueEffectEstimate:
    observations: int
    effective_matches: float
    raw_effect: float
    reliability: float
    shrunk_effect: float


@dataclass(frozen=True)
class ContextualHomeExpectation:
    base_expected_home_score: float
    raw_context_offset: float
    applied_context_offset: float
    effective_home_advantage: float
    expected_home_score: float


def estimate_team_venue_effect(
    base_expected_scores: Sequence[float],
    actual_scores: Sequence[float],
    weights: Sequence[float],
    *,
    elo_scale: float,
    shrinkage_matches: float,
    max_team_effect: float,
) -> TeamVenueEffectEstimate:
    """Estimate a regularized Elo-point offset from pre-match residuals."""

    _require_positive_finite("elo_scale", elo_scale)
    _require_non_negative_finite("shrinkage_matches", shrinkage_matches)
    _require_non_negative_finite("max_team_effect", max_team_effect)
    expected = tuple(float(value) for value in base_expected_scores)
    actual = tuple(float(value) for value in actual_scores)
    sample_weights = tuple(float(value) for value in weights)
    if not (len(expected) == len(actual) == len(sample_weights)):
        raise ValueError("expected, actual and weights must have equal length")
    if not expected:
        return TeamVenueEffectEstimate(0, 0.0, 0.0, 0.0, 0.0)
    for value in expected:
        if not math.isfinite(value) or not 0.0 < value < 1.0:
            raise ValueError("base_expected_scores must be finite and in (0, 1)")
    for value in actual:
        if not math.isfinite(value) or value not in (0.0, 0.5, 1.0):
            raise ValueError("actual_scores must contain only 0, 0.5 or 1")
    for value in sample_weights:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("weights must be positive and finite")

    effective_matches = float(sum(sample_weights))
    logit_scale = math.log(10.0) / float(elo_scale)
    offset = 0.0
    for _ in range(60):
        gradient = 0.0
        hessian = 0.0
        for base, observed, weight in zip(
            expected, actual, sample_weights, strict=True
        ):
            base_logit = math.log(base / (1.0 - base))
            probability = _logistic(base_logit + logit_scale * offset)
            gradient += weight * (observed - probability) * logit_scale
            hessian -= (
                weight
                * probability
                * (1.0 - probability)
                * logit_scale
                * logit_scale
            )
        if abs(hessian) < 1e-18:
            break
        updated = max(-2000.0, min(2000.0, offset - gradient / hessian))
        if abs(updated - offset) < 1e-10:
            offset = updated
            break
        offset = updated

    reliability = (
        effective_matches / (effective_matches + float(shrinkage_matches))
        if effective_matches > 0.0
        else 0.0
    )
    shrunk = max(
        -float(max_team_effect),
        min(float(max_team_effect), offset * reliability),
    )
    return TeamVenueEffectEstimate(
        observations=len(expected),
        effective_matches=effective_matches,
        raw_effect=float(offset),
        reliability=float(reliability),
        shrunk_effect=float(shrunk),
    )


def contextual_home_expectation(
    base_expected_home_score: float,
    home_team_effect: float,
    away_team_effect: float,
    *,
    global_home_advantage: float,
    elo_scale: float,
    is_neutral: bool,
    config: TeamVenueContextConfig,
) -> ContextualHomeExpectation:
    """Adjust a baseline expectation without changing either stored rating."""

    config.validate()
    _require_probability(
        "base_expected_home_score", base_expected_home_score, positive=True
    )
    if float(base_expected_home_score) >= 1.0:
        raise ValueError("base_expected_home_score must be less than 1")
    _require_finite("home_team_effect", home_team_effect)
    _require_finite("away_team_effect", away_team_effect)
    _require_non_negative_finite("global_home_advantage", global_home_advantage)
    _require_positive_finite("elo_scale", elo_scale)
    if not isinstance(is_neutral, bool):
        raise ValueError("is_neutral must be boolean")
    if is_neutral:
        return ContextualHomeExpectation(
            base_expected_home_score=float(base_expected_home_score),
            raw_context_offset=0.0,
            applied_context_offset=0.0,
            effective_home_advantage=0.0,
            expected_home_score=float(base_expected_home_score),
        )

    raw_offset = float(home_team_effect) - float(away_team_effect)
    bounded_offset = max(
        -float(config.max_context_offset),
        min(float(config.max_context_offset), raw_offset),
    )
    effective_home_advantage = max(
        float(config.minimum_home_advantage),
        min(
            float(config.maximum_home_advantage),
            float(global_home_advantage) + bounded_offset,
        ),
    )
    applied_offset = effective_home_advantage - float(global_home_advantage)
    base = float(base_expected_home_score)
    base_logit = math.log(base / (1.0 - base))
    adjusted = _logistic(
        base_logit + math.log(10.0) * applied_offset / float(elo_scale)
    )
    return ContextualHomeExpectation(
        base_expected_home_score=base,
        raw_context_offset=raw_offset,
        applied_context_offset=applied_offset,
        effective_home_advantage=effective_home_advantage,
        expected_home_score=adjusted,
    )


def _logistic(value: float) -> float:
    if value >= 0.0:
        exponential = math.exp(-value)
        return 1.0 / (1.0 + exponential)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


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


def _require_probability(name: str, value: float, *, positive: bool) -> None:
    _require_finite(name, value)
    lower_ok = float(value) > 0.0 if positive else float(value) >= 0.0
    if not lower_ok or float(value) > 1.0:
        boundary = "(0, 1]" if positive else "[0, 1]"
        raise ValueError(f"{name} must be in {boundary}")
