from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Literal

from ao_elo.domestic_surprise import league_finish_score


FIVE_SEASON_WEIGHTS = (0.07, 0.13, 0.20, 0.27, 0.33)
NormalizationMethod = Literal["legacy_finish_curve", "direct_percentile"]


@dataclass(frozen=True, order=True)
class SymmetricDomesticSurpriseConfig:
    coefficient: float = 0.0
    normalization: NormalizationMethod = "legacy_finish_curve"
    minimum_history_seasons: int = 5
    max_abs_adjustment: float = 50.0
    exposure_power: float = 1.0

    def validate(self) -> None:
        if not math.isfinite(self.coefficient) or self.coefficient < 0:
            raise ValueError("coefficient must be non-negative and finite")
        if self.normalization not in ("legacy_finish_curve", "direct_percentile"):
            raise ValueError("Unknown domestic finish normalization")
        if not 1 <= self.minimum_history_seasons <= 5:
            raise ValueError("minimum_history_seasons must be between one and five")
        if not math.isfinite(self.max_abs_adjustment) or self.max_abs_adjustment < 0:
            raise ValueError("max_abs_adjustment must be non-negative and finite")
        if not math.isfinite(self.exposure_power) or self.exposure_power < 0:
            raise ValueError("exposure_power must be non-negative and finite")

    @property
    def active(self) -> bool:
        return self.coefficient > 0


@dataclass(frozen=True)
class SymmetricDomesticSurpriseAdjustment:
    historical_finish_score: float | None
    history_seasons: int
    history_weight_coverage: float
    surprise_score: float
    surprise_component: float
    domestic_prior_adjustment: float
    adjusted_domestic_prior: float
    adjusted_ao_first_elo: float


def direct_league_position_score(position: int, league_team_count: int) -> float:
    if isinstance(position, bool) or isinstance(league_team_count, bool):
        raise ValueError("position and league_team_count must be integers")
    if int(position) != position or int(league_team_count) != league_team_count:
        raise ValueError("position and league_team_count must be integers")
    position = int(position)
    league_team_count = int(league_team_count)
    if league_team_count <= 1:
        raise ValueError("league_team_count must be greater than one")
    if not 1 <= position <= league_team_count:
        raise ValueError("position must be within the league table")
    return (league_team_count - position) / (league_team_count - 1)


def normalized_finish_score(
    position: int,
    league_team_count: int,
    normalization: NormalizationMethod,
) -> float:
    if normalization == "legacy_finish_curve":
        return league_finish_score(position, league_team_count)
    if normalization == "direct_percentile":
        return direct_league_position_score(position, league_team_count)
    raise ValueError("Unknown domestic finish normalization")


def weighted_five_season_finish_score(
    finish_scores: Iterable[float | None],
) -> tuple[float | None, int, float]:
    values = list(finish_scores)
    if len(values) != 5:
        raise ValueError("Exactly five historical finish values are required")
    observed = []
    for value, weight in zip(values, FIVE_SEASON_WEIGHTS):
        if value is None:
            continue
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError("historical finish scores must be finite and in [0,1]")
        observed.append((number, weight))
    if not observed:
        return None, 0, 0.0
    weight_coverage = sum(weight for _, weight in observed)
    historical = sum(value * weight for value, weight in observed) / weight_coverage
    return historical, len(observed), weight_coverage


def calculate_symmetric_domestic_surprise_adjustment(
    *,
    current_finish_score: float,
    historical_finish_scores: Iterable[float | None],
    domestic_prior: float,
    european_prior: float,
    effective_european_exposure: float,
    domestic_achievement_component: float,
    achievement_scale: float,
    config: SymmetricDomesticSurpriseConfig,
) -> SymmetricDomesticSurpriseAdjustment:
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

    historical, count, weight_coverage = weighted_five_season_finish_score(
        historical_finish_scores
    )
    surprise = 0.0 if historical is None else current_finish_score - historical
    if historical is None or count < config.minimum_history_seasons or not config.active:
        component = 0.0
        adjustment = 0.0
    else:
        component = weight_coverage * config.coefficient * surprise
        raw_adjustment = domestic_achievement_component * achievement_scale * component
        adjustment = max(
            -config.max_abs_adjustment,
            min(config.max_abs_adjustment, raw_adjustment),
        )

    adjusted_domestic = domestic_prior + adjustment
    baseline_first = domestic_prior + effective_european_exposure * (
        european_prior - domestic_prior
    )
    exposure_retention = (1.0 - effective_european_exposure) ** config.exposure_power
    adjusted_first = baseline_first + exposure_retention * adjustment
    if not all(math.isfinite(value) for value in (adjusted_domestic, adjusted_first)):
        raise ValueError("Domestic surprise produced a non-finite rating")
    return SymmetricDomesticSurpriseAdjustment(
        historical_finish_score=historical,
        history_seasons=count,
        history_weight_coverage=weight_coverage,
        surprise_score=surprise,
        surprise_component=component,
        domestic_prior_adjustment=adjustment,
        adjusted_domestic_prior=adjusted_domestic,
        adjusted_ao_first_elo=adjusted_first,
    )
