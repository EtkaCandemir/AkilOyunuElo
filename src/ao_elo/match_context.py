from __future__ import annotations

import math
from dataclasses import dataclass


COMPETITIONS = ("UCL", "UEL", "UECL")


@dataclass(frozen=True, order=True)
class AggregateStateConfig:
    rating_points_per_goal: float = 0.0
    goal_cap: int = 3

    def validate(self) -> None:
        if not math.isfinite(self.rating_points_per_goal):
            raise ValueError("aggregate rating_points_per_goal must be finite")
        if self.rating_points_per_goal < 0.0:
            raise ValueError("aggregate rating_points_per_goal cannot be negative")
        if isinstance(self.goal_cap, bool) or not isinstance(self.goal_cap, int):
            raise ValueError("aggregate goal_cap must be an integer")
        if self.goal_cap <= 0:
            raise ValueError("aggregate goal_cap must be positive")

    @property
    def active(self) -> bool:
        return self.rating_points_per_goal > 0.0


@dataclass(frozen=True, order=True)
class HomeAdvantageProfile:
    ucl_multiplier: float = 1.0
    uel_multiplier: float = 1.0
    uecl_multiplier: float = 1.0
    knockout_multiplier: float = 1.0
    second_leg_multiplier: float = 1.0

    def validate(self) -> None:
        values = (
            self.ucl_multiplier,
            self.uel_multiplier,
            self.uecl_multiplier,
            self.knockout_multiplier,
            self.second_leg_multiplier,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("home-advantage multipliers must be positive and finite")

    def multiplier(
        self,
        competition: str,
        *,
        is_knockout: bool,
        is_second_leg: bool,
    ) -> float:
        self.validate()
        try:
            competition_multiplier = {
                "UCL": self.ucl_multiplier,
                "UEL": self.uel_multiplier,
                "UECL": self.uecl_multiplier,
            }[competition]
        except KeyError as error:
            raise ValueError(f"Unknown competition: {competition}") from error
        phase_multiplier = self.knockout_multiplier if is_knockout else 1.0
        leg_multiplier = self.second_leg_multiplier if is_second_leg else 1.0
        return competition_multiplier * phase_multiplier * leg_multiplier


@dataclass(frozen=True, order=True)
class DrawContextConfig:
    draw_at_even: float = 0.24
    draw_shape: float = 1.0
    uel_offset: float = 0.0
    uecl_offset: float = 0.0
    knockout_offset: float = 0.0
    second_leg_offset: float = 0.0

    def validate(self) -> None:
        values = (
            self.draw_at_even,
            self.draw_shape,
            self.uel_offset,
            self.uecl_offset,
            self.knockout_offset,
            self.second_leg_offset,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("draw-context parameters must be finite")
        if not 0.0 <= self.draw_at_even <= 0.5:
            raise ValueError("draw_at_even must be in [0,0.5]")
        if self.draw_shape < 1.0:
            raise ValueError("draw_shape must be at least one")
        for competition in COMPETITIONS:
            for knockout in (False, True):
                for second_leg in (False, True):
                    value = self.for_match(
                        competition,
                        is_knockout=knockout,
                        is_second_leg=second_leg,
                    )
                    if not 0.0 <= value <= 0.5:
                        raise ValueError(
                            "context-adjusted draw_at_even must remain in [0,0.5]"
                        )

    def for_match(
        self,
        competition: str,
        *,
        is_knockout: bool,
        is_second_leg: bool,
    ) -> float:
        if competition not in COMPETITIONS:
            raise ValueError(f"Unknown competition: {competition}")
        competition_offset = (
            self.uel_offset
            if competition == "UEL"
            else self.uecl_offset
            if competition == "UECL"
            else 0.0
        )
        return (
            self.draw_at_even
            + competition_offset
            + (self.knockout_offset if is_knockout else 0.0)
            + (self.second_leg_offset if is_second_leg else 0.0)
        )


@dataclass(frozen=True, order=True)
class DomesticRegressionConfig:
    mode: str = "AO_FIRST_BASELINE"
    persistence: float = 0.0

    def validate(self) -> None:
        if self.mode not in ("AO_FIRST_BASELINE", "DOMESTIC_ANCHORED"):
            raise ValueError(f"Unknown domestic regression mode: {self.mode}")
        if not math.isfinite(self.persistence) or not 0.0 <= self.persistence <= 1.0:
            raise ValueError("domestic persistence must be in [0,1]")
        if self.mode == "AO_FIRST_BASELINE" and self.persistence != 0.0:
            raise ValueError("AO First baseline persistence must be zero")

    @property
    def active(self) -> bool:
        return self.mode == "DOMESTIC_ANCHORED"


def apply_aggregate_state(
    rating_difference: float,
    aggregate_home_goal_difference: int,
    *,
    is_second_leg: bool,
    config: AggregateStateConfig,
) -> float:
    config.validate()
    if not math.isfinite(rating_difference):
        raise ValueError("rating_difference must be finite")
    if isinstance(aggregate_home_goal_difference, bool) or not isinstance(
        aggregate_home_goal_difference, int
    ):
        raise ValueError("aggregate_home_goal_difference must be an integer")
    if not is_second_leg or not config.active:
        return float(rating_difference)
    bounded = max(
        -config.goal_cap,
        min(config.goal_cap, aggregate_home_goal_difference),
    )
    return float(rating_difference - config.rating_points_per_goal * bounded)


def effective_home_advantage(
    global_home_advantage: float,
    competition: str,
    *,
    is_neutral: bool,
    is_knockout: bool,
    is_second_leg: bool,
    profile: HomeAdvantageProfile,
) -> float:
    profile.validate()
    if not math.isfinite(global_home_advantage) or global_home_advantage < 0.0:
        raise ValueError("global_home_advantage must be non-negative and finite")
    if is_neutral:
        return 0.0
    return float(
        global_home_advantage
        * profile.multiplier(
            competition,
            is_knockout=is_knockout,
            is_second_leg=is_second_leg,
        )
    )


def domestic_anchored_start_rating(
    ao_first_elo: float,
    domestic_prior: float,
    previous_power_elo: float | None,
    config: DomesticRegressionConfig,
) -> float:
    config.validate()
    if not math.isfinite(ao_first_elo) or not math.isfinite(domestic_prior):
        raise ValueError("season-start ratings must be finite")
    if previous_power_elo is not None and not math.isfinite(previous_power_elo):
        raise ValueError("previous_power_elo must be finite when provided")
    if not config.active or previous_power_elo is None:
        return float(ao_first_elo)
    return float(
        domestic_prior
        + config.persistence * (previous_power_elo - domestic_prior)
    )
