from __future__ import annotations

"""Research-only rules for the AO First Elo negative-history asymmetry.

The production seed deliberately remains outside this module.  Callers pass a
fully resolved production seed table and receive an audited candidate table.
No club name or identifier can participate in the rule.
"""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


BoostSignal = Literal["BLIND", "DOMESTIC_FORM", "LONG_HISTORY"]
BoostDesign = Literal["ADDITIVE", "EXPOSURE_MODIFIER", "FLOOR"]

BASELINE_KEY = "BASELINE"
BOOST_BLIND = "BOOST_BLIND"
BOOST_DOMESTIC_FORM = "BOOST_DOMESTIC_FORM"
BOOST_LONG_HISTORY = "BOOST_LONG_HISTORY"

REQUIRED_SEED_COLUMNS = {
    "season",
    "team_id",
    "club_id",
    "current_direct_percentile",
    "adjusted_domestic_prior",
    "european_prior",
    "effective_european_exposure",
    "adjusted_ao_first_elo",
}


@dataclass(frozen=True)
class SeedBoostConfig:
    signal: BoostSignal
    design: BoostDesign
    maximum_finish_percentile: float
    minimum_european_deficit: float
    magnitude: float
    minimum_form_percentile: float = 0.0
    maximum_boost: float = 150.0
    model_version: str = "ao-first-seed-boost-v1-research"

    def validate(self) -> None:
        if self.signal not in ("BLIND", "DOMESTIC_FORM", "LONG_HISTORY"):
            raise ValueError(f"unknown seed boost signal: {self.signal}")
        if self.design not in ("ADDITIVE", "EXPOSURE_MODIFIER", "FLOOR"):
            raise ValueError(f"unknown seed boost design: {self.design}")
        for name in ("maximum_finish_percentile", "minimum_form_percentile"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0,1]")
        if not math.isfinite(self.minimum_european_deficit) or self.minimum_european_deficit < 0:
            raise ValueError("minimum_european_deficit must be finite and non-negative")
        if not math.isfinite(self.magnitude) or self.magnitude < 0:
            raise ValueError("magnitude must be finite and non-negative")
        if not math.isfinite(self.maximum_boost) or self.maximum_boost <= 0:
            raise ValueError("maximum_boost must be finite and positive")
        if self.design == "EXPOSURE_MODIFIER" and self.magnitude > 1.0:
            raise ValueError("exposure modifier magnitude must be in [0,1]")
        if not self.model_version.strip():
            raise ValueError("model_version must be non-empty")

    @property
    def key(self) -> str:
        self.validate()
        return (
            f"{self.signal}_{self.design}_finish{self.maximum_finish_percentile:g}_"
            f"deficit{self.minimum_european_deficit:g}_m{self.magnitude:g}_"
            f"form{self.minimum_form_percentile:g}_cap{self.maximum_boost:g}"
        )


def candidate_grid(signal: BoostSignal) -> tuple[SeedBoostConfig, ...]:
    """Return the fixed research surface; BASELINE is handled by the caller."""

    if signal == "LONG_HISTORY":
        return ()
    form_thresholds = (0.0,) if signal == "BLIND" else (0.50, 0.65, 0.80, 0.90)
    candidates: list[SeedBoostConfig] = []
    for finish in (0.50, 0.65, 0.80):
        for deficit in (50.0, 100.0, 150.0):
            for form in form_thresholds:
                for amount in (25.0, 50.0, 75.0, 100.0, 150.0):
                    candidates.append(
                        SeedBoostConfig(signal, "ADDITIVE", finish, deficit, amount, form)
                    )
                for reduction in (0.25, 0.50, 0.75, 1.00):
                    candidates.append(
                        SeedBoostConfig(
                            signal, "EXPOSURE_MODIFIER", finish, deficit, reduction, form
                        )
                    )
                for floor_gap in (0.0, 25.0, 50.0, 75.0, 100.0):
                    candidates.append(
                        SeedBoostConfig(signal, "FLOOR", finish, deficit, floor_gap, form)
                    )
    keys = [candidate.key for candidate in candidates]
    if len(keys) != len(set(keys)):
        raise ValueError("seed boost candidate grid contains duplicate keys")
    return tuple(candidates)


def apply_seed_boost(
    seeds: pd.DataFrame,
    config: SeedBoostConfig,
) -> pd.DataFrame:
    """Apply one auditable rule to every supplied team-season.

    `current_direct_percentile` is 1 for the champion and approaches zero at
    the bottom.  Therefore the upper threshold deliberately selects teams
    whose latest league finish was not elite.  Exposure zero is explicitly
    excluded: those teams already retain the full adjusted domestic prior.
    """

    config.validate()
    missing = sorted(REQUIRED_SEED_COLUMNS - set(seeds.columns))
    if missing:
        raise ValueError(f"seed boost input missing columns: {missing}")
    if seeds.duplicated(["season", "team_id"]).any():
        raise ValueError("seed boost input contains duplicate team-season keys")
    result = seeds.copy()
    numeric = [
        "adjusted_domestic_prior",
        "european_prior",
        "effective_european_exposure",
        "adjusted_ao_first_elo",
    ]
    values = result[numeric].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any() or not np.isfinite(values.to_numpy(float)).all():
        raise ValueError("seed boost numeric inputs must be finite")
    if not values["effective_european_exposure"].between(0.0, 1.0).all():
        raise ValueError("effective_european_exposure must be in [0,1]")
    finish = pd.to_numeric(result["current_direct_percentile"], errors="coerce")
    if not finish.dropna().between(0.0, 1.0).all():
        raise ValueError("known current_direct_percentile values must be in [0,1]")

    result["european_prior_deficit"] = (
        result["adjusted_domestic_prior"] - result["european_prior"]
    )
    result["negative_european_drag"] = (
        result["adjusted_domestic_prior"] - result["adjusted_ao_first_elo"]
    )
    eligible = (
        result["effective_european_exposure"].gt(0.0)
        & result["european_prior_deficit"].ge(config.minimum_european_deficit)
        & result["negative_european_drag"].gt(0.0)
        & finish.notna()
        & finish.le(config.maximum_finish_percentile)
    )

    if config.signal == "DOMESTIC_FORM":
        required = {"domestic_form_covered", "domestic_form_percentile"}
        if missing_form := sorted(required - set(result.columns)):
            raise ValueError(f"domestic-form boost missing columns: {missing_form}")
        eligible &= result["domestic_form_covered"].astype(bool)
        eligible &= result["domestic_form_percentile"].ge(
            config.minimum_form_percentile
        )
    elif config.signal == "LONG_HISTORY":
        required = {"long_history_covered", "long_history_percentile"}
        if missing_history := sorted(required - set(result.columns)):
            raise ValueError(f"long-history boost missing columns: {missing_history}")
        eligible &= result["long_history_covered"].astype(bool)
        eligible &= result["long_history_percentile"].ge(
            config.minimum_form_percentile
        )

    if config.design == "ADDITIVE":
        requested = np.full(len(result), config.magnitude, dtype=float)
    elif config.design == "EXPOSURE_MODIFIER":
        requested = result["negative_european_drag"].to_numpy(float) * config.magnitude
    else:
        floor = result["adjusted_domestic_prior"].to_numpy(float) - config.magnitude
        requested = floor - result["adjusted_ao_first_elo"].to_numpy(float)

    requested = np.maximum(0.0, requested)
    applied = np.where(eligible, np.minimum(requested, config.maximum_boost), 0.0)
    result["seed_boost_eligible"] = eligible
    result["seed_boost_requested"] = requested
    result["seed_boost_applied"] = applied
    result["seed_boost_cap_hit"] = eligible & np.isclose(
        applied, config.maximum_boost, atol=1e-12
    ) & (requested >= config.maximum_boost)
    result["seed_boost_model"] = config.signal
    result["seed_boost_candidate_key"] = config.key
    result["candidate_ao_first_elo"] = result["adjusted_ao_first_elo"] + applied

    if (result.loc[~eligible, "seed_boost_applied"].abs() > 1e-12).any():
        raise ValueError("ineligible team received a seed boost")
    if (result["seed_boost_applied"] < -1e-12).any():
        raise ValueError("seed boost cannot reverse direction")
    if (result["seed_boost_applied"] > config.maximum_boost + 1e-12).any():
        raise ValueError("seed boost cap violated")
    if result.loc[result["effective_european_exposure"].eq(0.0), "seed_boost_applied"].any():
        raise ValueError("zero-exposure team cannot receive an asymmetry boost")
    return result


def season_relative_percentile(
    frame: pd.DataFrame,
    value_column: str,
    covered_column: str,
) -> pd.Series:
    """Deterministic within-season percentile for a sparse external signal."""

    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, index in frame.loc[frame[covered_column].astype(bool)].groupby("season").groups.items():
        values = frame.loc[index, value_column]
        result.loc[index] = values.rank(method="average", pct=True).to_numpy(float)
    return result
