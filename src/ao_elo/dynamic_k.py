from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass


AGGREGATION_METHODS = ("ARITHMETIC", "GEOMETRIC")


@dataclass(frozen=True, order=True)
class DynamicKConfig:
    lambda_factor: float
    max_k_multiplier: float
    aggregation: str
    inactivity_days: float
    match_evidence_scale: float

    def validate(self) -> None:
        values = {
            "lambda_factor": self.lambda_factor,
            "max_k_multiplier": self.max_k_multiplier,
            "inactivity_days": self.inactivity_days,
            "match_evidence_scale": self.match_evidence_scale,
        }
        if any(not math.isfinite(value) for value in values.values()):
            raise ValueError("Dynamic K config values must be finite")
        if self.lambda_factor < 0.0:
            raise ValueError("lambda_factor must be non-negative")
        if self.max_k_multiplier < 1.0:
            raise ValueError("max_k_multiplier must be at least one")
        if self.inactivity_days <= 0.0:
            raise ValueError("inactivity_days must be positive")
        if self.match_evidence_scale <= 0.0:
            raise ValueError("match_evidence_scale must be positive")
        if self.aggregation not in AGGREGATION_METHODS:
            raise ValueError(f"Unknown Dynamic K aggregation: {self.aggregation}")

    @property
    def config_id(self) -> str:
        self.validate()
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @property
    def key(self) -> str:
        self.validate()
        return (
            f"l{self.lambda_factor:g}_c{self.max_k_multiplier:g}_"
            f"{self.aggregation.lower()}_i{self.inactivity_days:g}_"
            f"m{self.match_evidence_scale:g}"
        )

    @property
    def complexity(self) -> int:
        return int(self.lambda_factor > 0.0)


@dataclass(frozen=True)
class TeamUncertainty:
    exposure: float
    match_evidence: float
    inactivity: float
    combined: float


@dataclass(frozen=True)
class DynamicKMatch:
    home_uncertainty: TeamUncertainty
    away_uncertainty: TeamUncertainty
    home_k: float
    away_k: float
    match_k: float


def baseline_dynamic_k_config() -> DynamicKConfig:
    config = DynamicKConfig(
        lambda_factor=0.0,
        max_k_multiplier=1.25,
        aggregation="ARITHMETIC",
        inactivity_days=180.0,
        match_evidence_scale=6.0,
    )
    config.validate()
    return config


def calculate_team_uncertainty(
    european_exposure: float,
    prior_matches: float,
    days_since_last_match: float | None,
    config: DynamicKConfig,
) -> TeamUncertainty:
    config.validate()
    _require_finite("european_exposure", european_exposure)
    _require_finite("prior_matches", prior_matches)
    if not 0.0 <= european_exposure <= 1.0:
        raise ValueError("european_exposure must be in [0,1]")
    if prior_matches < 0.0:
        raise ValueError("prior_matches must be non-negative")
    if days_since_last_match is not None:
        _require_finite("days_since_last_match", days_since_last_match)
        if days_since_last_match < 0.0:
            raise ValueError("days_since_last_match must be non-negative")

    exposure_uncertainty = 1.0 - float(european_exposure)
    match_uncertainty = math.exp(
        -float(prior_matches) / config.match_evidence_scale
    )
    inactivity_uncertainty = (
        1.0
        if days_since_last_match is None
        else min(float(days_since_last_match) / config.inactivity_days, 1.0)
    )
    combined = (
        exposure_uncertainty + match_uncertainty + inactivity_uncertainty
    ) / 3.0
    if not 0.0 <= combined <= 1.0:
        raise ValueError("Combined team uncertainty must be in [0,1]")
    return TeamUncertainty(
        exposure=exposure_uncertainty,
        match_evidence=match_uncertainty,
        inactivity=inactivity_uncertainty,
        combined=combined,
    )


def calculate_team_k(
    base_k: float,
    uncertainty: float,
    config: DynamicKConfig,
) -> float:
    config.validate()
    _require_positive_finite("base_k", base_k)
    _require_finite("uncertainty", uncertainty)
    if not 0.0 <= uncertainty <= 1.0:
        raise ValueError("uncertainty must be in [0,1]")
    multiplier = min(
        config.max_k_multiplier,
        1.0 + config.lambda_factor * float(uncertainty),
    )
    return float(base_k) * multiplier


def aggregate_match_k(
    home_k: float,
    away_k: float,
    aggregation: str,
) -> float:
    _require_positive_finite("home_k", home_k)
    _require_positive_finite("away_k", away_k)
    if aggregation == "ARITHMETIC":
        return (float(home_k) + float(away_k)) / 2.0
    if aggregation == "GEOMETRIC":
        return math.sqrt(float(home_k) * float(away_k))
    raise ValueError(f"Unknown Dynamic K aggregation: {aggregation}")


def calculate_dynamic_match_k(
    *,
    base_k: float,
    home_exposure: float,
    away_exposure: float,
    home_prior_matches: float,
    away_prior_matches: float,
    home_days_since_last_match: float | None,
    away_days_since_last_match: float | None,
    config: DynamicKConfig,
) -> DynamicKMatch:
    config.validate()
    home_uncertainty = calculate_team_uncertainty(
        home_exposure,
        home_prior_matches,
        home_days_since_last_match,
        config,
    )
    away_uncertainty = calculate_team_uncertainty(
        away_exposure,
        away_prior_matches,
        away_days_since_last_match,
        config,
    )
    home_k = calculate_team_k(base_k, home_uncertainty.combined, config)
    away_k = calculate_team_k(base_k, away_uncertainty.combined, config)
    match_k = aggregate_match_k(home_k, away_k, config.aggregation)
    maximum = float(base_k) * config.max_k_multiplier
    if not float(base_k) <= match_k <= maximum + 1e-12:
        raise ValueError("Dynamic match K violated its configured bounds")
    return DynamicKMatch(
        home_uncertainty=home_uncertainty,
        away_uncertainty=away_uncertainty,
        home_k=home_k,
        away_k=away_k,
        match_k=match_k,
    )


def _require_finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


def _require_positive_finite(name: str, value: float) -> None:
    _require_finite(name, value)
    if float(value) <= 0.0:
        raise ValueError(f"{name} must be positive")
