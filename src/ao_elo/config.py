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

V2_REFERENCE_MAX_RATING = 903.92
V2_RATING_MULTIPLIER = 1500.0 / (V2_REFERENCE_MAX_RATING - 500.0)
AO_MODEL_V2_VERSION = "ao-european-elo-v2.0-dev-freeze"


@dataclass(frozen=True)
class AOEuropeanEloConfig:
    """Parameters for AO European Elo; defaults retain the validated v1.1 model."""

    country_strength_benchmark: float | None
    european_history_benchmark: float | None

    model_version: str = "v1.1"
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
    # The percentile curve is defined on [percentile_floor, +scale]. An unknown
    # position must not land below that floor, or absence of evidence would be
    # punished harder than the worst possible evidence.
    unknown_league_finish_score: float = 0.15
    cup_base_score: float = 0.62
    cup_double_bonus_multiplier: float = 0.08
    # `max(L, C)` makes the cup a floor rather than a contribution. When the
    # contribution is enabled the weaker of the two achievements is added with
    # this weight, which is inert for non-cup-winners because min(L, C) = 0.
    cup_contribution_enabled: bool = False
    cup_contribution_weight: float = 0.0
    achievement_cap: float = 1.10

    domestic_surprise_enabled: bool = False
    domestic_surprise_coefficient: float = 0.0
    domestic_surprise_variance_penalty: float = 0.0
    domestic_surprise_max_abs_adjustment: float = 0.0
    domestic_surprise_minimum_history_seasons: int = 5

    european_prior_max_boost: float = 420.0
    # A season the club could not enter contributes zero UEFA points, which is
    # arithmetically identical to a season it entered and scored nothing. The
    # normalization renormalizes the weighted history over the participated
    # weight only; it is neutral by construction at full participation.
    european_participation_enabled: bool = False
    european_participation_shrinkage: float = 0.0
    exposure_season_weight: float = 0.60
    exposure_match_weight: float = 0.40
    # The active production cap. A config built without an explicit value must
    # inherit the shipped contract, not the superseded 0.85 research value.
    max_european_exposure: float = 0.65
    country_tail_beta: float = 0.0
    european_tail_beta: float = 0.0
    exposure_tail_beta: float = 0.0
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
            # Pinned: 0.85 is the historical v1.1 cap, kept so the frozen
            # regression pilots stay reproducible. Production uses 0.65.
            max_european_exposure=0.85,
            # Pinned to the historical v1.1 value for the same reason.
            unknown_league_finish_score=0.10,
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
            # Pinned to the v1.1-era cap this candidate was rejected under.
            max_european_exposure=0.85,
            unknown_league_finish_score=0.10,
        )

    @classmethod
    def v2(
        cls,
        *,
        country_tail_beta: float = 0.0,
        european_tail_beta: float = 0.0,
        exposure_tail_beta: float = 0.0,
        country_strength_benchmark: float = 25.0,
        european_history_benchmark: float = 20.0,
    ) -> AOEuropeanEloConfig:
        """Return the development-frozen 500-2000 reference-scale v2 config."""
        return cls(
            country_strength_benchmark=country_strength_benchmark,
            european_history_benchmark=european_history_benchmark,
            model_version=AO_MODEL_V2_VERSION,
            domestic_league_component=140.0 * V2_RATING_MULTIPLIER,
            domestic_achievement_component=160.0 * V2_RATING_MULTIPLIER,
            european_prior_max_boost=420.0 * V2_RATING_MULTIPLIER,
            domestic_surprise_enabled=True,
            domestic_surprise_coefficient=0.40,
            domestic_surprise_variance_penalty=0.50,
            domestic_surprise_max_abs_adjustment=30.0,
            domestic_surprise_minimum_history_seasons=5,
            european_participation_enabled=True,
            european_participation_shrinkage=0.20,
            cup_contribution_enabled=True,
            # champion_equivalent_weight(): cup_double_bonus_multiplier *
            # champion_base_score / cup_base_score. Written out because the
            # helper needs a config, and pinned by a test so it stays derived.
            cup_contribution_weight=0.12903225806451613,
            max_european_exposure=0.65,
            country_tail_beta=country_tail_beta,
            european_tail_beta=european_tail_beta,
            exposure_tail_beta=exposure_tail_beta,
        )

    @classmethod
    def active(cls) -> AOEuropeanEloConfig:
        """Return the selected model contract; v1.1 remains available explicitly."""
        return cls.v2()

    def validate(self) -> None:
        """Fail loudly when model parameters cannot produce valid ratings."""
        if not isinstance(self.model_version, str) or not self.model_version.strip():
            raise ValueError("model_version must be a non-empty string")
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
        if not isinstance(self.domestic_surprise_enabled, bool):
            raise ValueError("domestic_surprise_enabled must be boolean")
        _require_between_zero_and_one(
            "domestic_surprise_coefficient",
            self.domestic_surprise_coefficient,
        )
        _require_between_zero_and_one(
            "domestic_surprise_variance_penalty",
            self.domestic_surprise_variance_penalty,
        )
        _require_non_negative(
            "domestic_surprise_max_abs_adjustment",
            self.domestic_surprise_max_abs_adjustment,
        )
        if self.domestic_surprise_minimum_history_seasons != 5:
            raise ValueError(
                "domestic_surprise_minimum_history_seasons must equal five"
            )
        if self.domestic_surprise_enabled:
            _require_positive(
                "domestic_surprise_coefficient",
                self.domestic_surprise_coefficient,
            )
            _require_positive(
                "domestic_surprise_max_abs_adjustment",
                self.domestic_surprise_max_abs_adjustment,
            )
        if not isinstance(self.european_participation_enabled, bool):
            raise ValueError("european_participation_enabled must be boolean")
        if not isinstance(self.cup_contribution_enabled, bool):
            raise ValueError("cup_contribution_enabled must be boolean")
        _require_between_zero_and_one(
            "cup_contribution_weight",
            self.cup_contribution_weight,
        )
        _require_non_negative(
            "european_participation_shrinkage",
            self.european_participation_shrinkage,
        )
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
        for name in (
            "country_tail_beta",
            "european_tail_beta",
            "exposure_tail_beta",
        ):
            _require_between_zero_and_one(name, getattr(self, name))
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
