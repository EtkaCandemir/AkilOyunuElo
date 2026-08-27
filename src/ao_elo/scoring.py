from __future__ import annotations

from math import log

from ao_elo.config import AOEuropeanEloConfig


def normalize_log_score_uncapped(value: float, benchmark: float) -> float:
    """Return the non-negative log ratio before any upper-tail treatment."""
    value = max(float(value), 0.0)
    return log(1.0 + value) / log(1.0 + benchmark)


def apply_upper_tail(value: float, beta: float) -> float:
    """Preserve values through one and continue the upper tail with slope beta."""
    value = float(value)
    return value if value <= 1.0 else 1.0 + float(beta) * (value - 1.0)


def normalize_log_score(value: float, benchmark: float, tail_beta: float = 0.0) -> float:
    """Benchmark log normalization with a continuous configurable upper tail."""
    return apply_upper_tail(
        normalize_log_score_uncapped(value, benchmark),
        tail_beta,
    )


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


def participation_normalized_history(
    weighted_european_history: float,
    weighted_season_exposure: float,
    shrinkage: float,
) -> float:
    """Renormalize club European history over the seasons it could enter.

    The `(1 + shrinkage)` numerator is what makes the layer safe: at full
    participation the rate equals the published history exactly, so a club with
    five complete seasons of evidence cannot move. The correction is
    proportional to the real participation gap and nothing else.

    A club that never entered keeps a zero rate. Those rows carry
    `european_exposure = 0`, so the prior is ignored by the blend anyway and
    inventing a rate there would be noise on an empty denominator.
    """

    history = max(float(weighted_european_history), 0.0)
    played_weight = float(weighted_season_exposure)
    shrinkage = float(shrinkage)
    if shrinkage < 0.0:
        raise ValueError("shrinkage must be non-negative")
    if not 0.0 <= played_weight <= 1.0:
        raise ValueError("weighted_season_exposure must be in [0,1]")
    if played_weight <= 0.0:
        return 0.0
    return history * (1.0 + shrinkage) / (played_weight + shrinkage)


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


def compute_effective_european_exposure(
    european_exposure: float,
    maximum: float,
    tail_beta: float = 0.0,
) -> float:
    """Apply a continuous tail above the domestic-preservation threshold."""
    exposure = float(european_exposure)
    maximum = float(maximum)
    if exposure <= maximum:
        return exposure
    return maximum + float(tail_beta) * (exposure - maximum)


def rating_source_type(exposure: float, threshold: float) -> str:
    """Human-readable evidence category for the output rating."""
    if exposure == 0:
        return "Pure Domestic Projection"
    if exposure < threshold:
        return "Mixed Domestic-European Estimate"
    return "European Evidence-Based Rating"
