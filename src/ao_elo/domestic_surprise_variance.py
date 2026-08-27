from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


FIVE_SEASON_WEIGHTS = (0.07, 0.13, 0.20, 0.27, 0.33)
DOMESTIC_HISTORY_KEYS = tuple(f"t_minus_{offset}" for offset in range(5, 0, -1))


@dataclass(frozen=True, order=True)
class VarianceDomesticSurpriseConfig:
    coefficient: float = 0.0
    variance_penalty: float = 0.0
    max_abs_adjustment: float = 50.0
    minimum_history_seasons: int = 5

    def validate(self) -> None:
        for name, value in (
            ("coefficient", self.coefficient),
            ("variance_penalty", self.variance_penalty),
            ("max_abs_adjustment", self.max_abs_adjustment),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be non-negative and finite")
        if self.variance_penalty > 1:
            raise ValueError("variance_penalty must not exceed one")
        if self.minimum_history_seasons != 5:
            raise ValueError("minimum_history_seasons must equal five")

    @property
    def active(self) -> bool:
        return self.coefficient > 0


@dataclass(frozen=True)
class VarianceDomesticSurpriseAdjustment:
    historical_mean: float | None
    historical_variance: float | None
    historical_volatility: float | None
    normalized_volatility: float | None
    consistency_multiplier: float
    history_seasons: int
    raw_surprise: float
    effective_surprise: float
    domestic_prior_adjustment: float
    adjusted_domestic_prior: float
    adjusted_ao_first_elo: float


def weighted_five_season_distribution(
    finish_scores: Iterable[float | None],
) -> tuple[float | None, float | None, float | None, int]:
    values = list(finish_scores)
    if len(values) != 5:
        raise ValueError("Exactly five historical finish values are required")
    observed: list[tuple[float, float]] = []
    for value, weight in zip(values, FIVE_SEASON_WEIGHTS, strict=True):
        if value is None:
            continue
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError("historical finish scores must be finite and in [0,1]")
        observed.append((number, weight))
    if not observed:
        return None, None, None, 0
    weight_sum = sum(weight for _, weight in observed)
    mean = sum(value * weight for value, weight in observed) / weight_sum
    variance = (
        sum(weight * (value - mean) ** 2 for value, weight in observed) / weight_sum
    )
    volatility = math.sqrt(max(0.0, variance))
    return mean, variance, volatility, len(observed)


def calculate_variance_domestic_surprise_adjustment(
    *,
    current_finish_score: float,
    historical_finish_scores: Iterable[float | None],
    domestic_prior: float,
    european_prior: float,
    effective_european_exposure: float,
    domestic_achievement_component: float,
    achievement_scale: float,
    config: VarianceDomesticSurpriseConfig,
) -> VarianceDomesticSurpriseAdjustment:
    config.validate()
    numeric = (
        current_finish_score,
        domestic_prior,
        european_prior,
        effective_european_exposure,
        domestic_achievement_component,
        achievement_scale,
    )
    if any(not math.isfinite(value) for value in numeric):
        raise ValueError("Domestic surprise inputs must be finite")
    if not 0.0 <= current_finish_score <= 1.0:
        raise ValueError("current_finish_score must be in [0,1]")
    if not 0.0 <= effective_european_exposure <= 1.0:
        raise ValueError("effective_european_exposure must be in [0,1]")
    if domestic_achievement_component < 0 or achievement_scale < 0:
        raise ValueError("Domestic component and achievement scale must be non-negative")

    mean, variance, volatility, count = weighted_five_season_distribution(
        historical_finish_scores
    )
    raw_surprise = 0.0 if mean is None else current_finish_score - mean
    normalized_volatility = (
        None if volatility is None else min(1.0, 2.0 * volatility)
    )
    consistency_multiplier = (
        1.0
        if normalized_volatility is None
        else 1.0 - config.variance_penalty * normalized_volatility
    )
    if not 0.0 <= consistency_multiplier <= 1.0:
        raise ValueError("consistency_multiplier must be in [0,1]")
    effective_surprise = raw_surprise * consistency_multiplier
    if mean is None or count < config.minimum_history_seasons or not config.active:
        adjustment = 0.0
    else:
        raw_adjustment = (
            domestic_achievement_component
            * achievement_scale
            * config.coefficient
            * effective_surprise
        )
        adjustment = max(
            -config.max_abs_adjustment,
            min(config.max_abs_adjustment, raw_adjustment),
        )

    adjusted_domestic = domestic_prior + adjustment
    baseline_first = domestic_prior + effective_european_exposure * (
        european_prior - domestic_prior
    )
    adjusted_first = baseline_first + (1.0 - effective_european_exposure) * adjustment
    if not all(math.isfinite(value) for value in (adjusted_domestic, adjusted_first)):
        raise ValueError("Domestic surprise produced a non-finite rating")
    return VarianceDomesticSurpriseAdjustment(
        historical_mean=mean,
        historical_variance=variance,
        historical_volatility=volatility,
        normalized_volatility=normalized_volatility,
        consistency_multiplier=consistency_multiplier,
        history_seasons=count,
        raw_surprise=raw_surprise,
        effective_surprise=effective_surprise,
        domestic_prior_adjustment=adjustment,
        adjusted_domestic_prior=adjusted_domestic,
        adjusted_ao_first_elo=adjusted_first,
    )
