from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite


SEASON_KEYS: tuple[str, ...] = (
    "t_minus_4",
    "t_minus_3",
    "t_minus_2",
    "t_minus_1",
    "t",
)


DEFAULT_SEASON_WEIGHTS: dict[str, float] = {
    "t_minus_4": 0.07,
    "t_minus_3": 0.13,
    "t_minus_2": 0.20,
    "t_minus_1": 0.27,
    "t": 0.33,
}


@dataclass(frozen=True)
class AOEuropeanEloConfig:
    """Parameters for AO European Elo; defaults retain the validated v1.1 model."""

    country_strength_benchmark: float | None
    european_history_benchmark: float | None

    base_rating: float = 500.0
    season_weights: dict[str, float] = field(
        default_factory=lambda: DEFAULT_SEASON_WEIGHTS.copy()
    )

    gamma: float = 0.80
    domestic_league_component: float = 140.0
    domestic_achievement_component: float = 160.0

    achievement_alpha: float = 0.40
    percentile_floor: float = 0.15
    percentile_scale: float = 0.70
    percentile_delta: float = 1.00
    champion_base_score: float = 1.00
    unknown_league_finish_score: float = 0.10
    cup_base_score: float = 0.62
    cup_double_bonus_multiplier: float = 0.08
    achievement_cap: float = 1.10

    european_prior_max_boost: float = 420.0
    exposure_season_weight: float = 0.60
    exposure_match_weight: float = 0.40
    max_european_exposure: float = 0.85
    rating_source_evidence_threshold: float = 0.75

    @classmethod
    def v1_1(
        cls,
        country_strength_benchmark: float = 25.0,
        european_history_benchmark: float = 20.0,
    ) -> AOEuropeanEloConfig:
        """Return the frozen v1.1 configuration for reproducible comparisons."""
        return cls(
            country_strength_benchmark=country_strength_benchmark,
            european_history_benchmark=european_history_benchmark,
            gamma=0.80,
            domestic_league_component=140.0,
        )

    @classmethod
    def experimental_country_candidate(
        cls,
        country_strength_benchmark: float = 20.0,
        european_history_benchmark: float = 20.0,
    ) -> AOEuropeanEloConfig:
        """Return the rejected 20/2.0/360 candidate for diagnostic backtests."""
        return cls(
            country_strength_benchmark=country_strength_benchmark,
            european_history_benchmark=european_history_benchmark,
            gamma=2.0,
            domestic_league_component=360.0,
        )

    def validate(self) -> None:
        """Fail loudly when model parameters cannot produce bounded ratings."""
        if not _is_positive_finite(self.country_strength_benchmark):
            raise ValueError("Country_Strength_Benchmark must be calibrated and > 0")

        if not _is_positive_finite(self.european_history_benchmark):
            raise ValueError("European_History_Benchmark must be calibrated and > 0")

        missing_weights = set(SEASON_KEYS) - set(self.season_weights)
        if missing_weights:
            missing = ", ".join(sorted(missing_weights))
            raise ValueError(f"Missing season weights: {missing}")

        unexpected_weights = set(self.season_weights) - set(SEASON_KEYS)
        if unexpected_weights:
            unexpected = ", ".join(sorted(unexpected_weights))
            raise ValueError(f"Unexpected season weights: {unexpected}")

        for key in SEASON_KEYS:
            weight = self.season_weights[key]
            if not _is_non_negative_finite(weight):
                raise ValueError(f"Season weight {key} must be finite and >= 0")

        total_weight = sum(self.season_weights[key] for key in SEASON_KEYS)
        if abs(total_weight - 1.0) > 1e-9:
            raise ValueError("Season weights must sum to 1.0")

        _require_finite("base_rating", self.base_rating)
        _require_positive("gamma", self.gamma)
        for name in (
            "domestic_league_component",
            "domestic_achievement_component",
            "percentile_floor",
            "percentile_scale",
            "champion_base_score",
            "unknown_league_finish_score",
            "cup_base_score",
            "cup_double_bonus_multiplier",
            "european_prior_max_boost",
        ):
            _require_non_negative(name, getattr(self, name))

        _require_between_zero_and_one("achievement_alpha", self.achievement_alpha)
        _require_positive("percentile_delta", self.percentile_delta)
        _require_positive("achievement_cap", self.achievement_cap)
        _require_between_zero_and_one(
            "exposure_season_weight",
            self.exposure_season_weight,
        )
        _require_between_zero_and_one(
            "exposure_match_weight",
            self.exposure_match_weight,
        )
        _require_between_zero_and_one(
            "max_european_exposure",
            self.max_european_exposure,
        )
        _require_between_zero_and_one(
            "rating_source_evidence_threshold",
            self.rating_source_evidence_threshold,
        )

        exposure_weight_total = self.exposure_season_weight + self.exposure_match_weight
        if abs(exposure_weight_total - 1.0) > 1e-9:
            raise ValueError("Exposure season/match weights must sum to 1.0")


def _is_positive_finite(value: float | None) -> bool:
    return value is not None and isfinite(float(value)) and float(value) > 0


def _is_non_negative_finite(value: float) -> bool:
    return isfinite(float(value)) and float(value) >= 0


def _require_finite(name: str, value: float) -> None:
    if not isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


def _require_positive(name: str, value: float) -> None:
    if not isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{name} must be finite and > 0")


def _require_non_negative(name: str, value: float) -> None:
    if not _is_non_negative_finite(value):
        raise ValueError(f"{name} must be finite and >= 0")


def _require_between_zero_and_one(name: str, value: float) -> None:
    if not isfinite(float(value)) or not 0 <= float(value) <= 1:
        raise ValueError(f"{name} must be finite and between 0 and 1")
