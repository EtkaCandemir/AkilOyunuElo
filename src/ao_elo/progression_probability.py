from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class ProgressionProbabilityConfig:
    logit_slope: float = 1.0
    single_home_bias: float = 0.0
    two_leg_first_home_bias: float = 0.0

    def validate(self) -> None:
        values = {
            "logit_slope": self.logit_slope,
            "single_home_bias": self.single_home_bias,
            "two_leg_first_home_bias": self.two_leg_first_home_bias,
        }
        if any(not math.isfinite(float(value)) for value in values.values()):
            raise ValueError("Progression probability parameters must be finite")
        if self.logit_slope <= 0.0:
            raise ValueError("logit_slope must be positive")

    @property
    def key(self) -> str:
        return (
            f"slope{self.logit_slope:g}"
            f"_single{self.single_home_bias:g}"
            f"_twoleg{self.two_leg_first_home_bias:g}"
        )


def identity_progression_probability_config() -> ProgressionProbabilityConfig:
    return ProgressionProbabilityConfig()


def calibrate_progression_probability(
    raw_probability: float,
    tie_match_count: int,
    first_match_neutral: bool,
    config: ProgressionProbabilityConfig,
) -> float:
    config.validate()
    if not math.isfinite(float(raw_probability)):
        raise ValueError("raw_probability must be finite")
    if not 0.0 <= float(raw_probability) <= 1.0:
        raise ValueError("raw_probability must be in [0,1]")
    if isinstance(tie_match_count, bool) or not isinstance(tie_match_count, int):
        raise ValueError("tie_match_count must be an integer")
    if tie_match_count <= 0:
        raise ValueError("tie_match_count must be positive")
    if not isinstance(first_match_neutral, bool):
        raise ValueError("first_match_neutral must be boolean")

    probability = min(max(float(raw_probability), 1e-12), 1.0 - 1e-12)
    raw_logit = math.log(probability / (1.0 - probability))
    offset = 0.0
    if tie_match_count == 1 and not first_match_neutral:
        offset = config.single_home_bias
    elif tie_match_count >= 2:
        offset = config.two_leg_first_home_bias
    adjusted_logit = config.logit_slope * raw_logit + offset
    if adjusted_logit >= 700.0:
        return 1.0
    if adjusted_logit <= -700.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-adjusted_logit))
