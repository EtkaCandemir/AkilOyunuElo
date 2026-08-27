"""Research-only amplification of the active five-season domestic surprise.

This module deliberately does not participate in ``AOEuropeanEloConfig.active``.
It exposes a small, auditable candidate surface for walk-forward research while
preserving the production formula as an explicit control configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Literal

from ao_elo.domestic_surprise_variance import (
    VarianceDomesticSurpriseConfig,
    weighted_five_season_distribution,
)


ExposureFamily = Literal[
    "LINEAR", "FLOOR_50", "FLOOR_65", "POWER_075", "POWER_050"
]
EXPOSURE_FAMILIES: tuple[ExposureFamily, ...] = (
    "LINEAR",
    "FLOOR_50",
    "FLOOR_65",
    "POWER_075",
    "POWER_050",
)


@dataclass(frozen=True, order=True)
class DomesticSurpriseAmplificationConfig:
    """Candidate parameters; never a production configuration by itself."""

    theta: float = 0.40
    domestic_cap: float = 30.0
    variance_penalty: float = 0.50
    exposure_family: ExposureFamily = "LINEAR"
    final_effect_cap: float = 75.0
    minimum_history_seasons: int = 5

    def validate(self) -> None:
        for name, value in (
            ("theta", self.theta),
            ("domestic_cap", self.domestic_cap),
            ("variance_penalty", self.variance_penalty),
            ("final_effect_cap", self.final_effect_cap),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be non-negative and finite")
        if self.variance_penalty > 1.0:
            raise ValueError("variance_penalty must not exceed one")
        if self.exposure_family not in EXPOSURE_FAMILIES:
            raise ValueError(f"Unsupported exposure_family: {self.exposure_family}")
        if self.minimum_history_seasons != 5:
            raise ValueError("minimum_history_seasons must equal five")


@dataclass(frozen=True)
class DomesticSurpriseAmplificationAdjustment:
    historical_mean: float | None
    historical_variance: float | None
    historical_volatility: float | None
    normalized_volatility: float | None
    consistency_multiplier: float
    history_seasons: int
    raw_surprise: float
    effective_surprise: float
    domestic_adjustment: float
    exposure_weight: float
    ao_first_elo_adjustment: float
    final_effect_capped: bool


def exposure_weight(exposure: float, family: ExposureFamily) -> float:
    """Return a finite, monotone non-increasing domestic-evidence weight."""
    if not math.isfinite(exposure) or not 0.0 <= exposure <= 1.0:
        raise ValueError("effective_european_exposure must be finite and in [0,1]")
    if family == "LINEAR":
        value = 1.0 - exposure
    elif family == "FLOOR_50":
        value = max(1.0 - exposure, 0.50)
    elif family == "FLOOR_65":
        value = max(1.0 - exposure, 0.65)
    elif family == "POWER_075":
        value = (1.0 - exposure) ** 0.75
    elif family == "POWER_050":
        value = math.sqrt(1.0 - exposure)
    else:  # pragma: no cover - kept for defensive runtime validation.
        raise ValueError(f"Unsupported exposure family: {family}")
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("exposure weight must be finite and in [0,1]")
    return value


def calculate_domestic_surprise_amplification(
    *,
    current_finish_score: float,
    historical_finish_scores: Iterable[float | None],
    effective_european_exposure: float,
    domestic_achievement_component: float,
    achievement_scale: float,
    config: DomesticSurpriseAmplificationConfig,
) -> DomesticSurpriseAmplificationAdjustment:
    """Calculate the research candidate's AO First Elo effect.

    The result is an *effect relative to the no-surprise AO First Elo seed*.
    It is intentionally separate from the production pipeline so experiments
    cannot change served ratings by import side effect.
    """
    config.validate()
    numeric = (
        current_finish_score,
        effective_european_exposure,
        domestic_achievement_component,
        achievement_scale,
    )
    if any(not math.isfinite(value) for value in numeric):
        raise ValueError("Domestic surprise amplification inputs must be finite")
    if not 0.0 <= current_finish_score <= 1.0:
        raise ValueError("current_finish_score must be in [0,1]")
    if domestic_achievement_component < 0.0 or achievement_scale < 0.0:
        raise ValueError("domestic achievement inputs must be non-negative")

    mean, variance, volatility, history_seasons = weighted_five_season_distribution(
        historical_finish_scores
    )
    raw_surprise = 0.0 if mean is None else current_finish_score - mean
    normalized_volatility = (
        None if volatility is None else min(1.0, 2.0 * volatility)
    )
    consistency = (
        1.0
        if normalized_volatility is None
        else 1.0 - config.variance_penalty * normalized_volatility
    )
    if not 0.0 <= consistency <= 1.0:
        raise ValueError("consistency multiplier must be in [0,1]")
    effective_surprise = raw_surprise * consistency

    if mean is None or history_seasons < config.minimum_history_seasons:
        domestic_adjustment = 0.0
    else:
        raw_adjustment = (
            domestic_achievement_component
            * achievement_scale
            * config.theta
            * effective_surprise
        )
        domestic_adjustment = max(
            -config.domestic_cap,
            min(config.domestic_cap, raw_adjustment),
        )

    weight = exposure_weight(effective_european_exposure, config.exposure_family)
    raw_effect = weight * domestic_adjustment
    effect = max(-config.final_effect_cap, min(config.final_effect_cap, raw_effect))
    if raw_effect and effect and raw_effect * effect < 0.0:
        raise AssertionError("Domestic surprise effect sign reversal")
    return DomesticSurpriseAmplificationAdjustment(
        historical_mean=mean,
        historical_variance=variance,
        historical_volatility=volatility,
        normalized_volatility=normalized_volatility,
        consistency_multiplier=consistency,
        history_seasons=history_seasons,
        raw_surprise=raw_surprise,
        effective_surprise=effective_surprise,
        domestic_adjustment=domestic_adjustment,
        exposure_weight=weight,
        ao_first_elo_adjustment=effect,
        final_effect_capped=abs(raw_effect) > config.final_effect_cap + 1e-12,
    )


def production_control_config() -> DomesticSurpriseAmplificationConfig:
    """The active production surprise parameters expressed in this API."""
    return DomesticSurpriseAmplificationConfig()

