from __future__ import annotations

from math import log

from ao_elo.config import AOEuropeanEloConfig


def normalize_log_score(value: float, benchmark: float) -> float:
    """Benchmark-based log normalization for country and club history scores."""
    value = max(float(value), 0.0)
    return min(1.0, log(1.0 + value) / log(1.0 + benchmark))


def compute_domestic_prior(
    league_strength: float,
    domestic_achievement_score: float,
    config: AOEuropeanEloConfig,
) -> float:
    """Domestic Prior: base + league strength + scaled local achievement."""
    achievement_scale = config.achievement_alpha + (
        (1.0 - config.achievement_alpha) * league_strength
    )
    return (
        config.base_rating
        + config.domestic_league_component * league_strength
        + config.domestic_achievement_component
        * domestic_achievement_score
        * achievement_scale
    )


def compute_european_prior(
    european_history_norm: float,
    config: AOEuropeanEloConfig,
) -> float:
    """European Prior: base plus benchmark-normalized European history boost."""
    return config.base_rating + config.european_prior_max_boost * european_history_norm


def compute_ao_first_elo(
    domestic_prior: float,
    european_prior: float,
    european_exposure: float,
) -> float:
    """Final shrinkage formula between Domestic Prior and European Prior."""
    return domestic_prior + european_exposure * (european_prior - domestic_prior)


def rating_source_type(exposure: float, threshold: float) -> str:
    """Human-readable evidence category for the output rating."""
    if exposure == 0:
        return "Pure Domestic Projection"
    if exposure < threshold:
        return "Mixed Domestic-European Estimate"
    return "European Evidence-Based Rating"

