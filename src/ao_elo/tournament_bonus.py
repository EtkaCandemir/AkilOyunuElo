from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass


COMPETITION_BONUS_RATIOS = {
    "UCL": 1.0,
    "UEL": 2.0 / 3.0,
    "UECL": 1.0 / 3.0,
}

ELIGIBLE_PROGRESSION_STAGES = frozenset(
    {
        "KNOCKOUT_PLAYOFF",
        "ROUND_OF_16",
        "QUARTERFINAL",
        "SEMIFINAL",
        "FINAL",
    }
)

STAGE_WEIGHTED_PROGRESSION_STAGES = (
    "ROUND_OF_16",
    "QUARTERFINAL",
    "SEMIFINAL",
    "FINAL",
)
DEFAULT_STAGE_WEIGHTED_COMPETITION_RATIOS = (
    ("UCL", 1.0),
    ("UEL", 2.0 / 3.0),
    ("UECL", 1.0 / 3.0),
)


@dataclass(frozen=True, order=True)
class FixedTournamentBonusConfig:
    """Winner-only, season-local tournament progression bonus."""

    base_bonus: float
    stages_per_competition: int = 5

    def validate(self) -> None:
        if (
            isinstance(self.base_bonus, bool)
            or not isinstance(self.base_bonus, (int, float))
            or not math.isfinite(float(self.base_bonus))
            or float(self.base_bonus) < 0.0
        ):
            raise ValueError("base_bonus must be non-negative and finite")
        if (
            isinstance(self.stages_per_competition, bool)
            or not isinstance(self.stages_per_competition, int)
            or self.stages_per_competition <= 0
        ):
            raise ValueError("stages_per_competition must be a positive integer")

    def increment(self, competition: str) -> float:
        self.validate()
        try:
            ratio = COMPETITION_BONUS_RATIOS[competition]
        except KeyError as error:
            raise ValueError(f"Unknown competition: {competition}") from error
        return float(self.base_bonus) * ratio

    def cap(self, competition: str) -> float:
        return self.increment(competition) * self.stages_per_competition


@dataclass(frozen=True)
class TournamentBonusUpdate:
    competition: str
    stage: str
    tie_id: str
    bonus_pre: float
    applied_bonus: float
    bonus_post: float
    competition_cap: float


@dataclass(frozen=True)
class StageWeightedTournamentBonusConfig:
    """Winner-only, season-local progression bonus with stage weights."""

    ucl_total_cap: float
    competition_ratios: tuple[tuple[str, float], ...] = (
        DEFAULT_STAGE_WEIGHTED_COMPETITION_RATIOS
    )
    stage_weights: tuple[tuple[str, float], ...] = (
        ("ROUND_OF_16", 0.10),
        ("QUARTERFINAL", 0.20),
        ("SEMIFINAL", 0.30),
        ("FINAL", 0.40),
    )
    season_reset: bool = True
    winner_only: bool = True

    def validate(self) -> None:
        _require_non_negative_finite("ucl_total_cap", self.ucl_total_cap)
        if not isinstance(self.season_reset, bool):
            raise ValueError("season_reset must be boolean")
        if not isinstance(self.winner_only, bool):
            raise ValueError("winner_only must be boolean")
        if not self.season_reset:
            raise ValueError("Stage-weighted bonus currently requires season_reset=true")
        if not self.winner_only:
            raise ValueError("Stage-weighted bonus currently requires winner_only=true")

        ratios = _validated_pairs(
            "competition_ratios",
            self.competition_ratios,
            expected_keys=set(COMPETITION_BONUS_RATIOS),
        )
        if not ratios["UCL"] > ratios["UEL"] > ratios["UECL"] > 0.0:
            raise ValueError("competition_ratios must preserve UCL > UEL > UECL > 0")

        weights = _validated_pairs(
            "stage_weights",
            self.stage_weights,
            expected_keys=set(STAGE_WEIGHTED_PROGRESSION_STAGES),
        )
        if any(value < 0.0 for value in weights.values()):
            raise ValueError("stage_weights cannot be negative")
        if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-12):
            raise ValueError("stage_weights must sum to 1")

    def competition_ratio(self, competition: str) -> float:
        self.validate()
        try:
            return _pairs_to_mapping(self.competition_ratios)[competition]
        except KeyError as error:
            raise ValueError(f"Unknown competition: {competition}") from error

    def stage_weight(self, stage: str) -> float:
        self.validate()
        try:
            return _pairs_to_mapping(self.stage_weights)[stage]
        except KeyError as error:
            raise ValueError(
                f"Stage is not eligible for a stage-weighted bonus: {stage}"
            ) from error

    def stage_bonus(self, competition: str, stage: str) -> float:
        return (
            float(self.ucl_total_cap)
            * self.competition_ratio(competition)
            * self.stage_weight(stage)
        )

    def cap(self, competition: str) -> float:
        return float(self.ucl_total_cap) * self.competition_ratio(competition)


@dataclass(frozen=True)
class StageWeightedTournamentBonusUpdate:
    competition: str
    stage: str
    tie_id: str
    stage_weight: float
    competition_ratio: float
    stage_bonus_requested: float
    stage_bonus_applied: float
    competition_bonus_pre: float
    competition_bonus_post: float
    competition_cap: float


@dataclass(frozen=True)
class FiveStageWeightedTournamentBonusConfig:
    """Five-stage weighted variant that retains knockout play-off evidence."""

    ucl_total_cap: float = 60.0
    competition_ratios: tuple[tuple[str, float], ...] = (
        DEFAULT_STAGE_WEIGHTED_COMPETITION_RATIOS
    )
    stage_weights: tuple[tuple[str, float], ...] = (
        ("KNOCKOUT_PLAYOFF", 0.10),
        ("ROUND_OF_16", 0.15),
        ("QUARTERFINAL", 0.20),
        ("SEMIFINAL", 0.25),
        ("FINAL", 0.30),
    )
    season_reset: bool = True
    winner_only: bool = True

    def validate(self) -> None:
        _require_non_negative_finite("ucl_total_cap", self.ucl_total_cap)
        if not isinstance(self.season_reset, bool) or not self.season_reset:
            raise ValueError("Five-stage weighted bonus requires season_reset=true")
        if not isinstance(self.winner_only, bool) or not self.winner_only:
            raise ValueError("Five-stage weighted bonus requires winner_only=true")
        ratios = _validated_pairs(
            "competition_ratios",
            self.competition_ratios,
            expected_keys=set(COMPETITION_BONUS_RATIOS),
        )
        if not ratios["UCL"] > ratios["UEL"] > ratios["UECL"] > 0.0:
            raise ValueError("competition_ratios must preserve UCL > UEL > UECL > 0")
        weights = _validated_pairs(
            "stage_weights",
            self.stage_weights,
            expected_keys=set(ELIGIBLE_PROGRESSION_STAGES),
        )
        if any(value < 0.0 for value in weights.values()):
            raise ValueError("stage_weights cannot be negative")
        if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-12):
            raise ValueError("stage_weights must sum to 1")

    def competition_ratio(self, competition: str) -> float:
        self.validate()
        try:
            return _pairs_to_mapping(self.competition_ratios)[competition]
        except KeyError as error:
            raise ValueError(f"Unknown competition: {competition}") from error

    def stage_weight(self, stage: str) -> float:
        self.validate()
        try:
            return _pairs_to_mapping(self.stage_weights)[stage]
        except KeyError as error:
            raise ValueError(
                f"Stage is not eligible for a five-stage weighted bonus: {stage}"
            ) from error

    def stage_bonus(self, competition: str, stage: str) -> float:
        return (
            float(self.ucl_total_cap)
            * self.competition_ratio(competition)
            * self.stage_weight(stage)
        )

    def cap(self, competition: str) -> float:
        return float(self.ucl_total_cap) * self.competition_ratio(competition)


def apply_tournament_progress_bonus(
    current_competition_bonus: float,
    competition: str,
    stage: str,
    tie_id: str,
    processed_ties: set[str],
    config: FixedTournamentBonusConfig,
) -> TournamentBonusUpdate:
    """Apply one fixed bonus after a tie is decided.

    The caller owns the seasonal state. Passing a fresh state and an empty
    processed-tie set at season start enforces the no-carry contract.
    """

    config.validate()
    _require_non_negative_finite("current_competition_bonus", current_competition_bonus)
    if stage not in ELIGIBLE_PROGRESSION_STAGES:
        raise ValueError(f"Stage is not eligible for a progression bonus: {stage}")
    if not isinstance(tie_id, str) or not tie_id.strip():
        raise ValueError("tie_id must be a non-empty string")
    if tie_id in processed_ties:
        raise ValueError(f"Progression bonus already applied for tie_id: {tie_id}")

    increment = config.increment(competition)
    cap = config.cap(competition)
    pre = float(current_competition_bonus)
    if pre > cap + 1e-9:
        raise ValueError("Current tournament bonus exceeds its competition cap")
    post = min(cap, pre + increment)
    applied = post - pre
    if applied < -1e-12 or post > cap + 1e-9:
        raise ValueError("Tournament bonus update violated its cap invariant")
    processed_ties.add(tie_id)
    return TournamentBonusUpdate(
        competition=competition,
        stage=stage,
        tie_id=tie_id,
        bonus_pre=pre,
        applied_bonus=applied,
        bonus_post=post,
        competition_cap=cap,
    )


def apply_stage_weighted_tournament_bonus(
    current_competition_bonus: float,
    competition: str,
    stage: str,
    tie_id: str,
    processed_ties: set[str],
    config: StageWeightedTournamentBonusConfig,
) -> StageWeightedTournamentBonusUpdate:
    """Apply one stage-weighted bonus after a tie or final is decided."""

    config.validate()
    _require_non_negative_finite(
        "current_competition_bonus", current_competition_bonus
    )
    if stage not in STAGE_WEIGHTED_PROGRESSION_STAGES:
        raise ValueError(
            f"Stage is not eligible for a stage-weighted bonus: {stage}"
        )
    if not isinstance(tie_id, str) or not tie_id.strip():
        raise ValueError("tie_id must be a non-empty string")
    if tie_id in processed_ties:
        raise ValueError(f"Progression bonus already applied for tie_id: {tie_id}")

    stage_weight = config.stage_weight(stage)
    competition_ratio = config.competition_ratio(competition)
    requested = config.stage_bonus(competition, stage)
    cap = config.cap(competition)
    pre = float(current_competition_bonus)
    if pre > cap + 1e-9:
        raise ValueError("Current tournament bonus exceeds its competition cap")
    post = min(cap, pre + requested)
    applied = post - pre
    if applied < -1e-12 or post > cap + 1e-9:
        raise ValueError("Stage-weighted tournament bonus violated its cap invariant")
    processed_ties.add(tie_id)
    return StageWeightedTournamentBonusUpdate(
        competition=competition,
        stage=stage,
        tie_id=tie_id,
        stage_weight=stage_weight,
        competition_ratio=competition_ratio,
        stage_bonus_requested=requested,
        stage_bonus_applied=applied,
        competition_bonus_pre=pre,
        competition_bonus_post=post,
        competition_cap=cap,
    )


def apply_five_stage_weighted_tournament_bonus(
    current_competition_bonus: float,
    competition: str,
    stage: str,
    tie_id: str,
    processed_ties: set[str],
    config: FiveStageWeightedTournamentBonusConfig,
) -> StageWeightedTournamentBonusUpdate:
    """Apply one weighted bonus across the full five-stage production path."""

    config.validate()
    _require_non_negative_finite(
        "current_competition_bonus", current_competition_bonus
    )
    if stage not in ELIGIBLE_PROGRESSION_STAGES:
        raise ValueError(
            f"Stage is not eligible for a five-stage weighted bonus: {stage}"
        )
    if not isinstance(tie_id, str) or not tie_id.strip():
        raise ValueError("tie_id must be a non-empty string")
    if tie_id in processed_ties:
        raise ValueError(f"Progression bonus already applied for tie_id: {tie_id}")
    weight = config.stage_weight(stage)
    ratio = config.competition_ratio(competition)
    requested = config.stage_bonus(competition, stage)
    cap = config.cap(competition)
    pre = float(current_competition_bonus)
    if pre > cap + 1e-9:
        raise ValueError("Current tournament bonus exceeds its competition cap")
    post = min(cap, pre + requested)
    applied = post - pre
    if applied < -1e-12 or post > cap + 1e-9:
        raise ValueError("Five-stage weighted bonus violated its cap invariant")
    processed_ties.add(tie_id)
    return StageWeightedTournamentBonusUpdate(
        competition=competition,
        stage=stage,
        tie_id=tie_id,
        stage_weight=weight,
        competition_ratio=ratio,
        stage_bonus_requested=requested,
        stage_bonus_applied=applied,
        competition_bonus_pre=pre,
        competition_bonus_post=post,
        competition_cap=cap,
    )


def _require_non_negative_finite(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} must be non-negative and finite")


def _validated_pairs(
    name: str,
    pairs: tuple[tuple[str, float], ...],
    *,
    expected_keys: set[str],
) -> dict[str, float]:
    if not isinstance(pairs, tuple):
        raise ValueError(f"{name} must be a tuple of key/value pairs")
    mapping = _pairs_to_mapping(pairs)
    if set(mapping) != expected_keys or len(pairs) != len(expected_keys):
        raise ValueError(f"{name} must contain exactly {sorted(expected_keys)}")
    for key, value in mapping.items():
        _require_non_negative_finite(f"{name}.{key}", value)
    return mapping


def _pairs_to_mapping(pairs: tuple[tuple[str, float], ...]) -> dict[str, float]:
    try:
        mapping: Mapping[str, float] = dict(pairs)
    except (TypeError, ValueError) as error:
        raise ValueError("Expected key/value pairs") from error
    return dict(mapping)
