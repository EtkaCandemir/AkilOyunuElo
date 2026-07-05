from __future__ import annotations

from dataclasses import dataclass, field


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
    """Parameters for the pre-calibration AO European Elo model."""

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
    rating_source_evidence_threshold: float = 0.75

    def validate(self) -> None:
        """Fail loudly when calibration-critical parameters are unavailable."""
        if (
            self.country_strength_benchmark is None
            or self.country_strength_benchmark <= 0
        ):
            raise ValueError("Country_Strength_Benchmark must be calibrated and > 0")

        if (
            self.european_history_benchmark is None
            or self.european_history_benchmark <= 0
        ):
            raise ValueError("European_History_Benchmark must be calibrated and > 0")

        missing_weights = set(SEASON_KEYS) - set(self.season_weights)
        if missing_weights:
            missing = ", ".join(sorted(missing_weights))
            raise ValueError(f"Missing season weights: {missing}")

        total_weight = sum(self.season_weights[key] for key in SEASON_KEYS)
        if abs(total_weight - 1.0) > 1e-9:
            raise ValueError("Season weights must sum to 1.0")

        exposure_weight_total = self.exposure_season_weight + self.exposure_match_weight
        if abs(exposure_weight_total - 1.0) > 1e-9:
            raise ValueError("Exposure season/match weights must sum to 1.0")

