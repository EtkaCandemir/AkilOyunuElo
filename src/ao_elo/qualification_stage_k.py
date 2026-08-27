from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass


CURRENT_K_REFERENCE = "CURRENT_K_REFERENCE"

QUALIFICATION_ROUND_KEYS = {
    "Preliminary Round": "Q1",
    "1st Qualifying Round": "Q1",
    "2nd Qualifying Round": "Q2",
    "3rd Qualifying Round": "Q3",
    "Qualifying Play-off Round": "QUALIFYING_PLAYOFF",
}


@dataclass(frozen=True)
class QualificationStageKConfig:
    profile: str
    q1_multiplier: float
    q2_multiplier: float
    q3_multiplier: float
    qualifying_playoff_multiplier: float
    main_multiplier: float = 1.0
    selectable: bool = True

    def validate(self) -> None:
        if not isinstance(self.profile, str) or not self.profile.strip():
            raise ValueError("profile must be a non-empty string")
        values = (
            self.q1_multiplier,
            self.q2_multiplier,
            self.q3_multiplier,
            self.qualifying_playoff_multiplier,
            self.main_multiplier,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Stage K multipliers must be finite")
        if not all(0.0 < value <= 1.0 for value in values):
            raise ValueError("Stage K multipliers must be in (0, 1]")
        if not math.isclose(self.main_multiplier, 1.0, abs_tol=1e-12):
            raise ValueError("Main-stage K multiplier must equal 1.0")
        qualifier_values = values[:4]
        if self.selectable:
            if not all(
                left < right
                # Pairwise sliding window: the two operands differ in length by
                # one by construction, so `strict=True` would raise every call.
                for left, right in zip(qualifier_values, qualifier_values[1:])  # noqa: B905
            ):
                raise ValueError("Selectable qualifier multipliers must strictly increase")
            if self.qualifying_playoff_multiplier >= self.main_multiplier:
                raise ValueError("Qualifying play-off multiplier must be below main stage")
        elif not all(math.isclose(value, 1.0, abs_tol=1e-12) for value in values):
            raise ValueError("The non-selectable reference must use multiplier 1.0")

    @property
    def config_id(self) -> str:
        self.validate()
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def multiplier_for_key(self, round_key: str) -> float:
        self.validate()
        try:
            return {
                "Q1": self.q1_multiplier,
                "Q2": self.q2_multiplier,
                "Q3": self.q3_multiplier,
                "QUALIFYING_PLAYOFF": self.qualifying_playoff_multiplier,
                "MAIN": self.main_multiplier,
            }[round_key]
        except KeyError as error:
            raise ValueError(f"Unknown qualification round key: {round_key}") from error


def qualification_round_key(round_name: str) -> str:
    if not isinstance(round_name, str) or not round_name.strip():
        raise ValueError("round_name must be a non-empty string")
    return QUALIFICATION_ROUND_KEYS.get(round_name.strip(), "MAIN")


def stage_k_multiplier(round_name: str, config: QualificationStageKConfig) -> float:
    return config.multiplier_for_key(qualification_round_key(round_name))


def effective_match_k(
    base_k: float,
    round_name: str,
    config: QualificationStageKConfig,
) -> float:
    if not math.isfinite(base_k) or base_k < 0.0:
        raise ValueError("base_k must be finite and non-negative")
    return float(base_k) * stage_k_multiplier(round_name, config)


def reference_config() -> QualificationStageKConfig:
    config = QualificationStageKConfig(
        profile=CURRENT_K_REFERENCE,
        q1_multiplier=1.0,
        q2_multiplier=1.0,
        q3_multiplier=1.0,
        qualifying_playoff_multiplier=1.0,
        selectable=False,
    )
    config.validate()
    return config


def candidate_configs() -> tuple[QualificationStageKConfig, ...]:
    values = (
        ("VERY_STRICT", 0.20, 0.40, 0.60, 0.80),
        ("STRICT", 0.30, 0.45, 0.65, 0.85),
        ("CONSERVATIVE", 0.35, 0.50, 0.70, 0.90),
        ("BALANCED", 0.45, 0.60, 0.75, 0.90),
        ("MODERATE", 0.50, 0.65, 0.80, 0.95),
        ("MILD", 0.60, 0.75, 0.90, 0.95),
        ("VERY_MILD", 0.70, 0.80, 0.90, 0.95),
    )
    configs = tuple(QualificationStageKConfig(*value) for value in values)
    for config in configs:
        config.validate()
    return configs


def all_configs() -> tuple[QualificationStageKConfig, ...]:
    return (reference_config(), *candidate_configs())
