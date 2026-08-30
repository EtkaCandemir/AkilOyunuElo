"""Research-only European evidence recalibration for AO First Elo.

The production pipeline is deliberately not imported here. Callers provide an
audited production seed frame and receive a candidate rating column, leaving
the active contract untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math

import numpy as np
import pandas as pd

# Shared pure helper, not the production pipeline: re-implementing the tail here
# would risk diverging from the curve the active model uses.
from ao_elo.scoring import apply_upper_tail, participation_normalized_history


REQUIRED_COLUMNS = {
    "season",
    "team_id",
    "competition",
    "weighted_european_history",
    "weighted_season_exposure",
    "european_exposure",
    "domestic_prior",
    "adjusted_domestic_prior",
    "adjusted_ao_first_elo",
}
COMPETITIONS = ("UCL", "UEL", "UECL")


@dataclass(frozen=True)
class EuropeanPriorRecalibrationConfig:
    history_benchmark: float = 20.0
    prior_boost_scale: float = 1.0
    exposure_cap: float = 0.85
    european_tail_beta: float = 0.0
    domestic_boost_scale: float = 1.0
    # Pinned to the active production layer: this study measures the tail and
    # the domestic scale, not participation. Without it the research surface
    # normalizes raw history and stops reproducing production.
    participation_shrinkage: float = 0.2
    uel_quality: float = 1.0
    uecl_quality: float = 1.0
    base_rating: float = 500.0
    european_prior_max_boost: float = 1559.714795008913

    def validate(self) -> None:
        finite_positive = {
            "history_benchmark": self.history_benchmark,
            "prior_boost_scale": self.prior_boost_scale,
            "european_prior_max_boost": self.european_prior_max_boost,
        }
        for name, value in finite_positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.base_rating):
            raise ValueError("base_rating must be finite")
        for name, value in {
            "exposure_cap": self.exposure_cap,
            "uel_quality": self.uel_quality,
            "uecl_quality": self.uecl_quality,
        }.items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0,1]")
        if self.uel_quality < self.uecl_quality:
            raise ValueError("competition quality must preserve UCL >= UEL >= UECL")
        if not math.isfinite(self.european_tail_beta) or not (
            0.0 <= self.european_tail_beta <= 1.0
        ):
            raise ValueError("european_tail_beta must be within [0,1]")
        if not math.isfinite(self.domestic_boost_scale) or not (
            1.0 <= self.domestic_boost_scale <= 1.5
        ):
            raise ValueError("domestic_boost_scale must be within [1,1.5]")
        if not math.isfinite(self.participation_shrinkage) or (
            self.participation_shrinkage < 0.0
        ):
            raise ValueError("participation_shrinkage must be non-negative")

    @property
    def key(self) -> str:
        return (
            f"b{self.history_benchmark:g}_s{self.prior_boost_scale:g}_"
            f"e{self.exposure_cap:g}_q1-{self.uel_quality:g}-{self.uecl_quality:g}_"
            f"t{self.european_tail_beta:g}_d{self.domestic_boost_scale:g}"
        )


def candidate_grid() -> tuple[EuropeanPriorRecalibrationConfig, ...]:
    quality_profiles = ((1.0, 1.0), (0.95, 0.90), (0.90, 0.80))
    candidates = {
        config.key: config
        for benchmark, scale, exposure, quality in product(
            (20.0, 24.0, 28.0),
            (0.85, 0.925, 1.0),
            (0.75, 0.80, 0.85),
            quality_profiles,
        )
        for config in (
            EuropeanPriorRecalibrationConfig(
                history_benchmark=benchmark,
                prior_boost_scale=scale,
                exposure_cap=exposure,
                uel_quality=quality[0],
                uecl_quality=quality[1],
            ),
        )
    }
    return tuple(candidates[key] for key in sorted(candidates))


def tail_and_domestic_grid() -> tuple[EuropeanPriorRecalibrationConfig, ...]:
    """Grid for the two axes that address top-end compression.

    `candidate_grid` is left untouched: it is the evidence of the earlier study
    and still carries the pre-0.65 exposure default. Here the exposure cap,
    benchmark, prior scale and competition quality are all pinned to the active
    production values so the two new axes are measured on their own.
    """

    candidates = {
        config.key: config
        for tail, domestic in product(
            # Reaches 1.0 so the selection is not forced to the grid edge.
            # beta = 1 removes the truncation entirely: the log curve simply
            # continues, which is the natural formulation the cap overrides.
            (0.0, 0.25, 0.35, 0.50, 0.75, 0.90, 1.0),
            (1.0, 1.10, 1.20, 1.343),
        )
        for config in (
            EuropeanPriorRecalibrationConfig(
                history_benchmark=20.0,
                prior_boost_scale=1.0,
                exposure_cap=0.65,
                uel_quality=1.0,
                uecl_quality=1.0,
                european_tail_beta=tail,
                domestic_boost_scale=domestic,
            ),
        )
    }
    return tuple(candidates[key] for key in sorted(candidates))


def exposure_refinement_grid() -> tuple[EuropeanPriorRecalibrationConfig, ...]:
    exposures = (
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.675,
        0.70,
        0.725,
        0.75,
        0.775,
        0.80,
        0.825,
        0.85,
    )
    return tuple(
        EuropeanPriorRecalibrationConfig(
            history_benchmark=benchmark,
            prior_boost_scale=1.0,
            exposure_cap=exposure,
            uel_quality=1.0,
            uecl_quality=1.0,
        )
        for benchmark in (20.0, 24.0)
        for exposure in exposures
    )


def apply_european_prior_recalibration(
    seeds: pd.DataFrame,
    config: EuropeanPriorRecalibrationConfig,
) -> pd.DataFrame:
    config.validate()
    missing = sorted(REQUIRED_COLUMNS - set(seeds.columns))
    if missing:
        raise ValueError(f"European prior input missing columns: {missing}")
    if seeds.duplicated(["season", "team_id"]).any():
        raise ValueError("European prior input contains duplicate team-season keys")

    result = seeds.copy()
    numeric_columns = (
        "weighted_european_history",
        "weighted_season_exposure",
        "european_exposure",
        "domestic_prior",
        "adjusted_domestic_prior",
        "adjusted_ao_first_elo",
    )
    numeric = result.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("European prior numeric inputs must be finite")
    if (numeric["weighted_european_history"] < 0.0).any():
        raise ValueError("weighted_european_history cannot be negative")
    if not numeric["european_exposure"].between(0.0, 1.0).all():
        raise ValueError("european_exposure must be within [0,1]")
    competition = result["competition"].astype(str).str.upper()
    if not competition.isin(COMPETITIONS).all():
        invalid = sorted(competition.loc[~competition.isin(COMPETITIONS)].unique())
        raise ValueError(f"Unsupported competition values: {invalid}")

    # Production normalizes participation-adjusted history, not the raw sum.
    # Skipping this made the research baseline miss production by up to 131 Elo.
    rate = pd.Series(
        [
            participation_normalized_history(
                history, played, config.participation_shrinkage
            )
            for history, played in zip(
                numeric["weighted_european_history"],
                numeric["weighted_season_exposure"],
                strict=True,
            )
        ],
        index=numeric.index,
    )
    uncapped_norm = np.log1p(rate) / math.log1p(config.history_benchmark)
    # beta = 0 reproduces the production clip exactly; above one the curve keeps
    # going with slope beta so clubs past the benchmark stay distinguishable.
    history_norm = uncapped_norm.map(
        lambda value: apply_upper_tail(value, config.european_tail_beta)
    )
    quality = competition.map(
        {"UCL": 1.0, "UEL": config.uel_quality, "UECL": config.uecl_quality}
    ).astype(float)
    prior = config.base_rating + (
        config.european_prior_max_boost
        * config.prior_boost_scale
        * quality
        * history_norm
    )
    # Scale only the evidence components above the base rating. The Domestic
    # Surprise adjustment rides along in adjusted_domestic_prior and is added
    # back unscaled: it is capped at +-30 by a frozen parameter, and scaling it
    # here would silently widen that cap.
    surprise = numeric["adjusted_domestic_prior"] - numeric["domestic_prior"]
    scaled_domestic = config.base_rating + config.domestic_boost_scale * (
        numeric["domestic_prior"] - config.base_rating
    )
    scaled_adjusted_domestic = scaled_domestic + surprise
    effective_exposure = numeric["european_exposure"].clip(upper=config.exposure_cap)
    candidate = scaled_adjusted_domestic + effective_exposure * (
        prior - scaled_adjusted_domestic
    )

    result["candidate_key"] = config.key
    result["candidate_history_norm"] = history_norm
    result["candidate_competition_quality"] = quality
    result["candidate_history_rate"] = rate
    result["candidate_uncapped_history_norm"] = uncapped_norm
    result["candidate_tail_active"] = (uncapped_norm > 1.0) & (
        config.european_tail_beta > 0.0
    )
    result["candidate_adjusted_domestic_prior"] = scaled_adjusted_domestic
    result["candidate_european_prior"] = prior
    result["candidate_effective_exposure"] = effective_exposure
    result["candidate_ao_first_elo"] = candidate
    result["candidate_elo_delta"] = candidate - numeric["adjusted_ao_first_elo"]

    zero_exposure = numeric["european_exposure"].eq(0.0)
    if not np.allclose(
        result.loc[zero_exposure, "candidate_ao_first_elo"],
        scaled_adjusted_domestic.loc[zero_exposure],
        atol=1e-10,
    ):
        raise ValueError("Zero-exposure candidate must equal adjusted Domestic Prior")
    if not np.isfinite(result["candidate_ao_first_elo"]).all():
        raise ValueError("Candidate AO First Elo must be finite")
    return result


def ranking_uncertainty_summary(
    folds: pd.DataFrame,
    samples: int,
    *,
    seed: int = 20260822,
) -> pd.DataFrame:
    """Estimate ranking-delta uncertainty by resampling complete season folds."""
    required = {"delta_seed_spearman", "delta_seed_pairwise_accuracy"}
    missing = sorted(required - set(folds.columns))
    if missing:
        raise ValueError(f"Ranking uncertainty input missing columns: {missing}")
    if samples <= 0 or folds.empty:
        raise ValueError("Ranking uncertainty requires folds and positive samples")

    rng = np.random.default_rng(seed)
    rows = []
    for metric, column in (
        ("seed_spearman", "delta_seed_spearman"),
        ("seed_pairwise_accuracy", "delta_seed_pairwise_accuracy"),
    ):
        values = pd.to_numeric(folds[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} must contain finite values")
        draws = np.empty(samples, dtype=float)
        for index in range(samples):
            selected = rng.integers(0, len(values), len(values))
            draws[index] = float(values[selected].mean())
        lower, upper = np.quantile(draws, (0.025, 0.975))
        rows.append(
            {
                "metric": metric,
                "method": "season_block_bootstrap",
                "folds": len(values),
                "mean_difference": float(values.mean()),
                "ci_95_lower": float(lower),
                "ci_95_upper": float(upper),
                "reliable_improvement": bool(lower > 0.0),
                "reliable_harm": bool(upper < 0.0),
            }
        )
    return pd.DataFrame(rows)
