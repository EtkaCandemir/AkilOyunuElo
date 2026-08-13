from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product

from ao_elo.draw_probability import score_preserving_1x2_scalar


GOAL_MARGIN_FAMILIES = (
    "NONE",
    "LOG",
    "SQRT",
    "FAVORITE_DAMPED_LOG",
)
GOAL_MARGIN_CAPS = (1.25, 1.50, 1.75, 2.00)
LOG_WEIGHTS = (0.125, 0.25, 0.50, 0.75, 1.00)
SQRT_WEIGHTS = (0.25, 0.50, 0.75, 1.00, 1.50)
FAVORITE_DAMPING = (0.50, 1.00, 2.00)

GLOBAL_K_SCALES = (0.75, 1.00, 1.25, 1.50)
UEL_K_RATIOS = (0.50, 0.65, 0.80, 0.90)
UECL_K_RATIOS = (0.25, 0.45, 0.60, 0.75)


@dataclass(frozen=True, order=True)
class GoalMarginCandidate:
    family: str
    weight: float
    cap: float
    favorite_damping: float = 0.0

    def validate(self) -> None:
        if self.family not in GOAL_MARGIN_FAMILIES:
            raise ValueError(f"Unknown goal-margin family: {self.family}")
        values = (self.weight, self.cap, self.favorite_damping)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Goal-margin parameters must be finite")
        if self.family == "NONE":
            if (self.weight, self.cap, self.favorite_damping) != (0.0, 1.0, 0.0):
                raise ValueError("NONE goal-margin config must be exactly 0/1/0")
            return
        if self.weight <= 0.0:
            raise ValueError("Active goal-margin weight must be positive")
        if self.cap <= 1.0:
            raise ValueError("Active goal-margin cap must exceed one")
        if self.family == "FAVORITE_DAMPED_LOG":
            if self.favorite_damping <= 0.0:
                raise ValueError("Favorite-damped log requires positive damping")
        elif self.favorite_damping != 0.0:
            raise ValueError("Only favorite-damped log may set favorite_damping")

    @property
    def active(self) -> bool:
        return self.family != "NONE"

    @property
    def key(self) -> str:
        if not self.active:
            return "margin_none"
        suffix = f"_d{self.favorite_damping:g}" if self.favorite_damping else ""
        return f"margin_{self.family.lower()}_w{self.weight:g}_c{self.cap:g}{suffix}"


def baseline_goal_margin() -> GoalMarginCandidate:
    return GoalMarginCandidate("NONE", 0.0, 1.0, 0.0)


def goal_margin_candidates() -> tuple[GoalMarginCandidate, ...]:
    candidates = {baseline_goal_margin()}
    candidates.update(
        GoalMarginCandidate("LOG", weight, cap)
        for weight, cap in product(LOG_WEIGHTS, GOAL_MARGIN_CAPS)
    )
    candidates.update(
        GoalMarginCandidate("SQRT", weight, cap)
        for weight, cap in product(SQRT_WEIGHTS, GOAL_MARGIN_CAPS)
    )
    candidates.update(
        GoalMarginCandidate("FAVORITE_DAMPED_LOG", weight, cap, damping)
        for weight, cap, damping in product(
            LOG_WEIGHTS,
            GOAL_MARGIN_CAPS,
            FAVORITE_DAMPING,
        )
    )
    result = tuple(sorted(candidates))
    for candidate in result:
        candidate.validate()
    return result


def goal_margin_multiplier(
    goal_difference: int,
    winner_expected_score: float,
    config: GoalMarginCandidate,
) -> float:
    config.validate()
    if isinstance(goal_difference, bool) or not isinstance(goal_difference, int):
        raise ValueError("goal_difference must be an integer")
    if goal_difference < 0:
        raise ValueError("goal_difference must be non-negative")
    if not math.isfinite(winner_expected_score) or not 0.0 <= winner_expected_score <= 1.0:
        raise ValueError("winner_expected_score must be in [0,1]")
    if not config.active or goal_difference <= 1:
        return 1.0

    if config.family in ("LOG", "FAVORITE_DAMPED_LOG"):
        signal = math.log(goal_difference)
    elif config.family == "SQRT":
        signal = math.sqrt(goal_difference) - 1.0
    else:  # pragma: no cover - guarded by config validation
        raise ValueError(f"Unsupported goal-margin family: {config.family}")

    correction = 1.0
    if config.family == "FAVORITE_DAMPED_LOG":
        favorite_strength = max(0.0, 2.0 * winner_expected_score - 1.0)
        correction = 1.0 / (1.0 + config.favorite_damping * favorite_strength)
    return min(config.cap, 1.0 + config.weight * signal * correction)


@dataclass(frozen=True, order=True)
class CompetitionKCandidate:
    profile: str
    ucl_multiplier: float
    uel_multiplier: float
    uecl_multiplier: float

    def validate(self) -> None:
        if self.profile not in ("GLOBAL", "HIERARCHY"):
            raise ValueError(f"Unknown competition-K profile: {self.profile}")
        values = (
            self.ucl_multiplier,
            self.uel_multiplier,
            self.uecl_multiplier,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("Competition-K multipliers must be positive and finite")
        if self.profile == "GLOBAL" and len(set(values)) != 1:
            raise ValueError("GLOBAL profile requires identical multipliers")
        if self.profile == "HIERARCHY" and not (
            self.ucl_multiplier > self.uel_multiplier > self.uecl_multiplier
        ):
            raise ValueError("HIERARCHY profile requires UCL > UEL > UECL")

    @property
    def key(self) -> str:
        return (
            f"k_{self.profile.lower()}_ucl{self.ucl_multiplier:g}"
            f"_uel{self.uel_multiplier:g}_uecl{self.uecl_multiplier:g}"
        )

    @property
    def is_equal_baseline(self) -> bool:
        return (
            self.profile == "GLOBAL"
            and self.ucl_multiplier == 1.0
            and self.uel_multiplier == 1.0
            and self.uecl_multiplier == 1.0
        )

    def for_competition(self, competition: str) -> float:
        self.validate()
        try:
            return {
                "UCL": self.ucl_multiplier,
                "UEL": self.uel_multiplier,
                "UECL": self.uecl_multiplier,
            }[competition]
        except KeyError as error:
            raise ValueError(f"Unknown competition: {competition}") from error


def baseline_competition_k() -> CompetitionKCandidate:
    return CompetitionKCandidate("GLOBAL", 1.0, 1.0, 1.0)


def competition_k_candidates() -> tuple[CompetitionKCandidate, ...]:
    candidates = {
        CompetitionKCandidate("GLOBAL", scale, scale, scale)
        for scale in GLOBAL_K_SCALES
    }
    candidates.update(
        CompetitionKCandidate(
            "HIERARCHY",
            global_scale,
            global_scale * uel_ratio,
            global_scale * uecl_ratio,
        )
        for global_scale, uel_ratio, uecl_ratio in product(
            GLOBAL_K_SCALES,
            UEL_K_RATIOS,
            UECL_K_RATIOS,
        )
        if uel_ratio > uecl_ratio
    )
    result = tuple(sorted(candidates))
    for candidate in result:
        candidate.validate()
    return result


def one_x_two_probabilities_scalar(
    expected_home_score: float,
    draw_at_even: float,
    draw_shape: float,
) -> tuple[float, float, float]:
    return score_preserving_1x2_scalar(
        expected_home_score,
        draw_at_even,
        draw_shape,
    )


def standard_1x2_loss_scalar(
    probabilities: tuple[float, float, float],
    home_goals: int,
    away_goals: int,
) -> tuple[float, float]:
    if min(home_goals, away_goals) < 0:
        raise ValueError("Goals must be non-negative")
    observed = 0 if home_goals > away_goals else 1 if home_goals == away_goals else 2
    target = (1.0 if observed == 0 else 0.0, 1.0 if observed == 1 else 0.0, 1.0 if observed == 2 else 0.0)
    brier = sum((probability - actual) ** 2 for probability, actual in zip(probabilities, target))
    log_loss = -math.log(max(probabilities[observed], 1e-15))
    return brier, log_loss
