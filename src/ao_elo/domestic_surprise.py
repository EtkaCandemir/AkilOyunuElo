from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


HISTORY_WEIGHTS = (0.13, 0.20, 0.27, 0.33)


@dataclass(frozen=True, order=True)
class DomesticSurpriseConfig:
    positive_weight: float = 0.0
    negative_weight: float = 0.0
    lookback_seasons: int = 3
    minimum_history_seasons: int = 2
    max_abs_adjustment: float = 50.0

    def validate(self) -> None:
        for name, value in (
            ("positive_weight", self.positive_weight),
            ("negative_weight", self.negative_weight),
            ("max_abs_adjustment", self.max_abs_adjustment),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be non-negative and finite")
        if self.lookback_seasons not in (2, 3, 4):
            raise ValueError("lookback_seasons must be 2, 3 or 4")
        if not 1 <= self.minimum_history_seasons <= self.lookback_seasons:
            raise ValueError(
                "minimum_history_seasons must be between one and lookback_seasons"
            )

    @property
    def active(self) -> bool:
        return self.positive_weight > 0 or self.negative_weight > 0


@dataclass(frozen=True)
class DomesticSurpriseAdjustment:
    historical_finish_score: float | None
    history_seasons: int
    history_reliability: float
    surprise_score: float
    surprise_component: float
    domestic_prior_adjustment: float
    adjusted_domestic_prior: float
    adjusted_ao_first_elo: float


def league_finish_score(position: int, league_team_count: int) -> float:
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
    if position == 1:
        return 1.0
    percentile = (league_team_count - position) / (league_team_count - 1)
    return 0.15 + 0.70 * percentile


def weighted_historical_finish_score(
    finish_scores: Iterable[float | None],
    lookback_seasons: int,
) -> tuple[float | None, int, float]:
    if lookback_seasons not in (2, 3, 4):
        raise ValueError("lookback_seasons must be 2, 3 or 4")
    values = list(finish_scores)[-lookback_seasons:]
    if len(values) < lookback_seasons:
        values = [None] * (lookback_seasons - len(values)) + values
    weights = HISTORY_WEIGHTS[-lookback_seasons:]
    observed = [
        (float(value), weight)
        for value, weight in zip(values, weights)
        if value is not None and math.isfinite(float(value))
    ]
    if not observed:
        return None, 0, 0.0
    if any(not 0.0 <= value <= 1.0 for value, _ in observed):
        raise ValueError("historical finish scores must be in [0,1]")
    weight_total = sum(weight for _, weight in observed)
    historical = sum(value * weight for value, weight in observed) / weight_total
    count = len(observed)
    return historical, count, count / lookback_seasons


def calculate_domestic_surprise_adjustment(
    *,
    current_finish_score: float,
    historical_finish_scores: Iterable[float | None],
    domestic_prior: float,
    european_prior: float,
    effective_european_exposure: float,
    domestic_achievement_component: float,
    achievement_scale: float,
    config: DomesticSurpriseConfig,
) -> DomesticSurpriseAdjustment:
    config.validate()
    numeric = {
        "current_finish_score": current_finish_score,
        "domestic_prior": domestic_prior,
        "european_prior": european_prior,
        "effective_european_exposure": effective_european_exposure,
        "domestic_achievement_component": domestic_achievement_component,
        "achievement_scale": achievement_scale,
    }
    if any(not math.isfinite(value) for value in numeric.values()):
        raise ValueError("Domestic surprise inputs must be finite")
    if not 0.0 <= current_finish_score <= 1.0:
        raise ValueError("current_finish_score must be in [0,1]")
    if not 0.0 <= effective_european_exposure <= 1.0:
        raise ValueError("effective_european_exposure must be in [0,1]")
    if domestic_achievement_component < 0 or achievement_scale < 0:
        raise ValueError("Domestic component and achievement scale must be non-negative")

    historical, count, reliability = weighted_historical_finish_score(
        historical_finish_scores,
        config.lookback_seasons,
    )
    if historical is None or count < config.minimum_history_seasons or not config.active:
        surprise = 0.0 if historical is None else current_finish_score - historical
        component = 0.0
        adjustment = 0.0
    else:
        surprise = current_finish_score - historical
        directional = (
            config.positive_weight * max(surprise, 0.0)
            - config.negative_weight * max(-surprise, 0.0)
        )
        component = reliability * directional
        raw_adjustment = (
            domestic_achievement_component * achievement_scale * component
        )
        adjustment = max(
            -config.max_abs_adjustment,
            min(config.max_abs_adjustment, raw_adjustment),
        )

    adjusted_domestic = domestic_prior + adjustment
    adjusted_first = adjusted_domestic + effective_european_exposure * (
        european_prior - adjusted_domestic
    )
    if not all(math.isfinite(value) for value in (adjusted_domestic, adjusted_first)):
        raise ValueError("Domestic surprise produced a non-finite rating")
    return DomesticSurpriseAdjustment(
        historical_finish_score=historical,
        history_seasons=count,
        history_reliability=reliability,
        surprise_score=surprise,
        surprise_component=component,
        domestic_prior_adjustment=adjustment,
        adjusted_domestic_prior=adjusted_domestic,
        adjusted_ao_first_elo=adjusted_first,
    )
