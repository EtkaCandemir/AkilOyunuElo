from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

from ao_elo.qualification_stage_k import QualificationStageKConfig


@dataclass(frozen=True)
class QualificationTransitionConfig:
    profile: str
    stage_k: QualificationStageKConfig
    qualifier_carry: float
    selectable: bool = True

    def validate(self) -> None:
        if not isinstance(self.profile, str) or not self.profile.strip():
            raise ValueError("profile must be a non-empty string")
        self.stage_k.validate()
        if not math.isfinite(self.qualifier_carry):
            raise ValueError("qualifier_carry must be finite")
        if not 0.0 <= self.qualifier_carry <= 1.0:
            raise ValueError("qualifier_carry must be in [0, 1]")
        if not self.selectable and (
            self.stage_k.selectable
            or not math.isclose(self.qualifier_carry, 1.0, abs_tol=1e-12)
        ):
            raise ValueError("The reference must use full stage K and carry=1.0")

    @property
    def config_id(self) -> str:
        self.validate()
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class QualificationCarryResult:
    anchor_rating: float
    pre_carry_rating: float
    post_carry_rating: float
    raw_qualifier_change: float
    carried_qualifier_change: float
    carry_adjustment: float


def apply_qualifier_carry(
    anchor_rating: float,
    pre_carry_rating: float,
    carry: float,
) -> QualificationCarryResult:
    for name, value in (
        ("anchor_rating", anchor_rating),
        ("pre_carry_rating", pre_carry_rating),
        ("carry", carry),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 0.0 <= carry <= 1.0:
        raise ValueError("carry must be in [0, 1]")
    raw_change = float(pre_carry_rating) - float(anchor_rating)
    carried_change = float(carry) * raw_change
    post = float(anchor_rating) + carried_change
    return QualificationCarryResult(
        anchor_rating=float(anchor_rating),
        pre_carry_rating=float(pre_carry_rating),
        post_carry_rating=post,
        raw_qualifier_change=raw_change,
        carried_qualifier_change=carried_change,
        carry_adjustment=post - float(pre_carry_rating),
    )
