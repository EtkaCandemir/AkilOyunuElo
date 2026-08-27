"""Research-only participation normalization for the European Prior.

The production European Prior is built from a fixed-weight sum of five seasons
of UEFA club points:

```text
weighted_european_history = sum(weight_k * club_points_k)
```

A season the club did not enter contributes `club_points = 0`, and
`validators.py` enforces exactly that. A season it did enter and lost every
match also contributes zero. The two are therefore **indistinguishable** in the
history channel, so the prior answers a mixture of two questions - "how good
were you in Europe" and "did you qualify at all" - when only the first belongs
to it. Whether a club qualifies is a domestic achievement, and the Domestic
Prior already owns it.

The consequence is a double charge. Missing a season lowers the history (fewer
points over the same denominator) *and* lowers `european_exposure`, which is
the weight the blend puts on the prior. The second charge is the designed one
and is left untouched here. This module removes only the first.

```text
pw   = weighted_season_exposure          # sum(weight_k * played_k), already published
rate = weighted_european_history * (1 + k) / (pw + k)
```

The `(1 + k)` numerator is what makes the correction safe: at `pw = 1` the rate
is exactly the published history, so a club with five seasons of evidence does
not move at all. Only a real participation gap produces a change, in proportion
to the gap. `k` shrinks the correction for clubs whose surviving denominator is
small, where the per-season rate would otherwise be estimated off almost
nothing.

`BLIND_LIFT` is the control arm. It raises the same population by the same
average amount using exposure alone, so a gain in the candidate arm can be
attributed to the participation structure rather than to the mere fact of
lifting low-exposure clubs.

The production pipeline is deliberately not imported. Callers provide an
audited production seed frame and receive `candidate_*` columns, leaving the
active contract untouched; this module never participates in
`AOEuropeanEloConfig.active`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


BASELINE = "BASELINE"
BLIND_LIFT = "PARTICIPATION_BLIND_LIFT"
NORMALIZED = "PARTICIPATION_NORMALIZED"
ARMS: tuple[str, ...] = (BASELINE, BLIND_LIFT, NORMALIZED)

SHRINKAGE_GRID: tuple[float, ...] = (0.0, 0.10, 0.20, 0.35, 0.50, 0.75)

REQUIRED_COLUMNS = {
    "season",
    "team_id",
    "weighted_european_history",
    "weighted_season_exposure",
    "european_exposure",
    "adjusted_domestic_prior",
    "adjusted_ao_first_elo",
}


@dataclass(frozen=True)
class ParticipationNormalizationConfig:
    """Candidate parameters; never a production configuration by itself."""

    arm: str = BASELINE
    shrinkage: float = 0.0
    blind_lift_coefficient: float = 0.0
    # Production constants, restated so this module needs no production import.
    history_benchmark: float = 20.0
    base_rating: float = 500.0
    european_prior_max_boost: float = 1559.714795008913
    exposure_cap: float = 0.65

    def validate(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"unknown arm: {self.arm}")
        for name in ("shrinkage", "blind_lift_coefficient"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be non-negative and finite")
        for name in ("history_benchmark", "european_prior_max_boost"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.base_rating):
            raise ValueError("base_rating must be finite")
        if not math.isfinite(self.exposure_cap) or not 0.0 < self.exposure_cap <= 1.0:
            raise ValueError("exposure_cap must be within (0,1]")
        if self.arm == BASELINE and (self.shrinkage or self.blind_lift_coefficient):
            raise ValueError("the baseline arm cannot carry a candidate parameter")
        if self.arm == NORMALIZED and self.blind_lift_coefficient:
            raise ValueError("the normalized arm is not lifted; it is renormalized")
        if self.arm == BLIND_LIFT and self.shrinkage:
            raise ValueError("the blind control has no participation parameter")

    @property
    def key(self) -> str:
        self.validate()
        if self.arm == BASELINE:
            return BASELINE
        if self.arm == BLIND_LIFT:
            return f"{BLIND_LIFT}_c{self.blind_lift_coefficient:g}"
        return f"{NORMALIZED}_k{self.shrinkage:g}"


def production_control_config() -> ParticipationNormalizationConfig:
    """The active production European Prior expressed in this API."""

    return ParticipationNormalizationConfig()


def candidate_grid(
    shrinkages: tuple[float, ...] = SHRINKAGE_GRID,
) -> tuple[ParticipationNormalizationConfig, ...]:
    """Baseline plus one normalized candidate per shrinkage value.

    The blind control is absent here on purpose: its coefficient is solved per
    fold against the candidate it is controlling for, so it cannot be
    enumerated ahead of time. `blind_control_config` builds it.
    """

    return (production_control_config(),) + tuple(
        ParticipationNormalizationConfig(arm=NORMALIZED, shrinkage=value)
        for value in shrinkages
    )


def blind_control_config(coefficient: float) -> ParticipationNormalizationConfig:
    return ParticipationNormalizationConfig(
        arm=BLIND_LIFT, blind_lift_coefficient=float(coefficient)
    )


def participation_rate(
    history: pd.Series | np.ndarray,
    played_weight: pd.Series | np.ndarray,
    shrinkage: float,
) -> np.ndarray:
    """Return the participation-normalized history rate.

    Neutral at full participation by construction, and zero where the club
    never entered - those rows carry `european_exposure = 0`, so the prior is
    ignored by the blend anyway and inventing a rate there would be noise.
    """

    history = np.asarray(history, dtype=float)
    played_weight = np.asarray(played_weight, dtype=float)
    shrinkage = float(shrinkage)
    if shrinkage < 0.0 or not math.isfinite(shrinkage):
        raise ValueError("shrinkage must be non-negative and finite")
    rate = np.zeros_like(history)
    entered = played_weight > 0.0
    rate[entered] = (
        history[entered] * (1.0 + shrinkage) / (played_weight[entered] + shrinkage)
    )
    return rate


def apply_participation_normalization(
    seeds: pd.DataFrame,
    config: ParticipationNormalizationConfig,
) -> pd.DataFrame:
    """Return the seed frame with this arm's `candidate_*` columns attached."""

    config.validate()
    missing = sorted(REQUIRED_COLUMNS - set(seeds.columns))
    if missing:
        raise ValueError(f"Participation input missing columns: {missing}")
    if seeds.duplicated(["season", "team_id"]).any():
        raise ValueError("Participation input contains duplicate team-season keys")

    result = seeds.copy()
    numeric_columns = (
        "weighted_european_history",
        "weighted_season_exposure",
        "european_exposure",
        "adjusted_domestic_prior",
        "adjusted_ao_first_elo",
    )
    numeric = result.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("Participation numeric inputs must be finite")
    if (numeric["weighted_european_history"] < 0.0).any():
        raise ValueError("weighted_european_history cannot be negative")
    for column in ("weighted_season_exposure", "european_exposure"):
        if not numeric[column].between(0.0, 1.0).all():
            raise ValueError(f"{column} must be within [0,1]")

    history = numeric["weighted_european_history"]
    played_weight = numeric["weighted_season_exposure"]
    domestic = numeric["adjusted_domestic_prior"]
    effective_exposure = numeric["european_exposure"].clip(upper=config.exposure_cap)

    if config.arm == NORMALIZED:
        rate = pd.Series(
            participation_rate(history, played_weight, config.shrinkage),
            index=result.index,
        )
    else:
        rate = history

    history_norm = (
        np.log1p(rate) / math.log1p(config.history_benchmark)
    ).clip(upper=1.0)
    prior = config.base_rating + config.european_prior_max_boost * history_norm
    candidate = domestic + effective_exposure * (prior - domestic)
    if config.arm == BLIND_LIFT:
        # The control must move the same population in the same direction
        # without reading participation, so it is a function of exposure alone.
        candidate = candidate + config.blind_lift_coefficient * (
            1.0 - effective_exposure
        )

    result["candidate_key"] = config.key
    result["candidate_arm"] = config.arm
    result["candidate_played_weight"] = played_weight
    result["candidate_history_rate"] = rate
    result["candidate_history_norm"] = history_norm
    result["candidate_european_prior"] = prior
    result["candidate_effective_exposure"] = effective_exposure
    result["candidate_ao_first_elo"] = candidate
    result["candidate_elo_delta"] = candidate - numeric["adjusted_ao_first_elo"]

    _assert_invariants(result, numeric, config)
    return result


def calibrate_blind_lift(
    seeds: pd.DataFrame,
    config: ParticipationNormalizationConfig,
) -> float:
    """Solve the control coefficient that matches a candidate's mean movement.

    The control only means something if it disturbs the seed as much as the
    candidate does; otherwise a null result would just say the control was
    smaller. `(1 - effective_exposure)` is the lift shape, so matching mean
    absolute movement is one division.
    """

    if config.arm != NORMALIZED:
        raise ValueError("the blind control is calibrated against a normalized arm")
    candidate = apply_participation_normalization(seeds, config)
    target = float(candidate["candidate_elo_delta"].abs().mean())
    shape = (
        1.0
        - pd.to_numeric(seeds["european_exposure"], errors="coerce").clip(
            upper=config.exposure_cap
        )
    ).abs().mean()
    if not math.isfinite(shape) or shape <= 0.0:
        raise ValueError("blind lift shape has no mass to calibrate against")
    return target / float(shape)


def _assert_invariants(
    result: pd.DataFrame,
    numeric: pd.DataFrame,
    config: ParticipationNormalizationConfig,
) -> None:
    if not np.isfinite(result["candidate_ao_first_elo"]).all():
        raise ValueError("Candidate AO First Elo must be finite")

    zero_exposure = numeric["european_exposure"].eq(0.0)
    if config.arm != BLIND_LIFT and not np.allclose(
        result.loc[zero_exposure, "candidate_ao_first_elo"],
        numeric.loc[zero_exposure, "adjusted_domestic_prior"],
        atol=1e-10,
    ):
        raise ValueError("Zero-exposure candidate must equal adjusted Domestic Prior")

    if config.arm == NORMALIZED:
        # The whole safety argument of this layer: complete evidence is never
        # touched, so the correction can only ever act on a real gap.
        full = numeric["weighted_season_exposure"].ge(1.0 - 1e-12)
        moved = (
            result.loc[full, "candidate_history_rate"]
            - numeric.loc[full, "weighted_european_history"]
        ).abs()
        if len(moved) and float(moved.max()) > 1e-12:
            raise ValueError(
                "Full-participation clubs must keep their published history"
            )
        gap = numeric["weighted_season_exposure"].between(1e-12, 1.0 - 1e-12)
        raised = (
            result.loc[gap, "candidate_history_rate"]
            >= numeric.loc[gap, "weighted_european_history"] - 1e-12
        )
        if len(raised) and not bool(raised.all()):
            raise ValueError("Participation normalization must never lower a rate")
