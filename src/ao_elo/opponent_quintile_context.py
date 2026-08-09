from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


QUINTILES = (1, 2, 3, 4, 5)
FIXED_THRESHOLDS = "FIXED_THRESHOLDS_DYNAMIC_MEMBERSHIP"
DYNAMIC_QUINTILES = "DYNAMIC_QUINTILES"
SEASON_LOCKED = "SEASON_LOCKED"
CAUSAL_ONLINE = "CAUSAL_ONLINE"
EXPECTED_SCORE_OUTCOME = "EXPECTED_SCORE"
CATEGORICAL_1X2_OUTCOME = "CATEGORICAL_1X2"


@dataclass(frozen=True, order=True)
class OpponentQuintileContextConfig:
    band_mode: str
    profile_update_mode: str
    lookback_seasons: int
    season_decay: float
    shrinkage_matches: float
    effect_and_context_cap: float
    band_count: int = 5
    outcome_model: str = EXPECTED_SCORE_OUTCOME

    def validate(self) -> None:
        if self.band_mode not in (FIXED_THRESHOLDS, DYNAMIC_QUINTILES):
            raise ValueError(f"Unknown quintile band mode: {self.band_mode}")
        if self.profile_update_mode not in (SEASON_LOCKED, CAUSAL_ONLINE):
            raise ValueError(
                f"Unknown quintile profile update mode: {self.profile_update_mode}"
            )
        if (
            isinstance(self.lookback_seasons, bool)
            or not isinstance(self.lookback_seasons, int)
            or self.lookback_seasons <= 0
        ):
            raise ValueError("lookback_seasons must be a positive integer")
        _require_probability("season_decay", self.season_decay, allow_zero=False)
        _require_non_negative("shrinkage_matches", self.shrinkage_matches)
        _require_non_negative("effect_and_context_cap", self.effect_and_context_cap)
        if self.band_count not in (3, 5):
            raise ValueError("band_count must be either 3 or 5")
        if self.outcome_model not in (
            EXPECTED_SCORE_OUTCOME,
            CATEGORICAL_1X2_OUTCOME,
        ):
            raise ValueError(f"Unknown outcome_model: {self.outcome_model}")

    @property
    def key(self) -> str:
        mode = "dynamic" if self.band_mode == DYNAMIC_QUINTILES else "fixed"
        update = "online" if self.profile_update_mode == CAUSAL_ONLINE else "locked"
        prefix = "tercile_" if self.band_count == 3 else ""
        outcome = "categorical_" if self.outcome_model == CATEGORICAL_1X2_OUTCOME else ""
        return (
            f"{prefix}{outcome}{mode}_{update}_w{self.lookback_seasons}_d{self.season_decay:g}"
            f"_k{self.shrinkage_matches:g}_cap{self.effect_and_context_cap:g}"
        )


@dataclass(frozen=True)
class QuintileSnapshot:
    assignments: pd.DataFrame
    thresholds: tuple[float, ...]
    boundary_ties: tuple[bool, ...]


@dataclass(frozen=True)
class QuintileProfile:
    club_id: str
    overall_effect: float
    overall_observations: int
    overall_effective_matches: float
    effects: tuple[float, ...]
    observations: tuple[int, ...]
    effective_matches: tuple[float, ...]
    reliabilities: tuple[float, ...]
    raw_specific_effects: tuple[float, ...]

    def effect_for(self, quintile: int) -> float:
        _require_band(quintile, len(self.effects))
        return self.effects[quintile - 1]


@dataclass(frozen=True)
class MatchupExpectation:
    base_expected_home_score: float
    home_effect: float
    away_effect: float
    raw_offset: float
    applied_offset: float
    expected_home_score: float


def assign_quintiles(
    team_ids: Sequence[int],
    club_ids: Sequence[str],
    ratings: Sequence[float],
    *,
    band_count: int = 5,
) -> QuintileSnapshot:
    """Assign deterministic, equal-sized dynamic strength bands."""

    if not (len(team_ids) == len(club_ids) == len(ratings)):
        raise ValueError("team_ids, club_ids and ratings must have equal length")
    _require_band_count(band_count)
    bands = tuple(range(1, band_count + 1))
    if len(team_ids) < band_count:
        raise ValueError("strength-band universe must contain at least one club per band")
    frame = pd.DataFrame(
        {
            "team_id": list(team_ids),
            "club_id": [str(value) for value in club_ids],
            "ao_live_elo": pd.to_numeric(pd.Series(ratings), errors="raise"),
        }
    )
    if frame["team_id"].duplicated().any() or frame["club_id"].duplicated().any():
        raise ValueError("quintile universe team_id and club_id must be unique")
    if not np.isfinite(frame["ao_live_elo"].to_numpy(float)).all():
        raise ValueError("quintile universe ratings must be finite")
    frame = frame.sort_values(["ao_live_elo", "club_id"], kind="stable").reset_index(
        drop=True
    )
    frame["rank_zero_based"] = np.arange(len(frame), dtype=int)
    frame["quintile"] = (
        1
        + np.minimum(band_count - 1, (band_count * frame["rank_zero_based"] // len(frame))).astype(int)
    )
    thresholds = []
    boundary_ties = []
    for quintile in bands[:-1]:
        lower = frame.loc[frame["quintile"].eq(quintile)].iloc[-1]
        upper = frame.loc[frame["quintile"].eq(quintile + 1)].iloc[0]
        thresholds.append(float(lower["ao_live_elo"]))
        boundary_ties.append(
            bool(math.isclose(float(lower["ao_live_elo"]), float(upper["ao_live_elo"])))
        )
    return QuintileSnapshot(
        assignments=frame,
        thresholds=tuple(thresholds),
        boundary_ties=tuple(boundary_ties),
    )


def assign_fixed_threshold_membership(
    team_ids: Sequence[int],
    club_ids: Sequence[str],
    ratings: Sequence[float],
    thresholds: Sequence[float],
) -> pd.DataFrame:
    """Assign current ratings to fixed, season-start Q1-Q5 Elo thresholds."""

    if len(thresholds) not in (2, 4):
        raise ValueError("fixed strength-band membership requires two or four thresholds")
    threshold_values = np.asarray(thresholds, dtype=float)
    if not np.isfinite(threshold_values).all() or np.any(np.diff(threshold_values) < 0.0):
        raise ValueError("fixed quintile thresholds must be finite and non-decreasing")
    if not (len(team_ids) == len(club_ids) == len(ratings)):
        raise ValueError("team_ids, club_ids and ratings must have equal length")
    values = np.asarray(ratings, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("fixed membership ratings must be finite")
    result = pd.DataFrame(
        {
            "team_id": list(team_ids),
            "club_id": [str(value) for value in club_ids],
            "ao_live_elo": values,
        }
    )
    if result["team_id"].duplicated().any() or result["club_id"].duplicated().any():
        raise ValueError("fixed membership team_id and club_id must be unique")
    result["quintile"] = np.searchsorted(threshold_values, values, side="left") + 1
    return result.sort_values(["team_id"], kind="stable").reset_index(drop=True)


def estimate_quintile_profile(
    club_id: str,
    history: pd.DataFrame,
    *,
    elo_scale: float,
    shrinkage_matches: float,
    effect_cap: float,
    band_count: int = 5,
    outcome_model: str = EXPECTED_SCORE_OUTCOME,
    draw_at_even: float = 0.24,
    draw_shape: float = 1.0,
) -> QuintileProfile:
    """Estimate centered, regularized opponent-band effects for one club."""

    _require_positive("elo_scale", elo_scale)
    _require_non_negative("shrinkage_matches", shrinkage_matches)
    _require_non_negative("effect_cap", effect_cap)
    _require_band_count(band_count)
    _require_outcome_model(outcome_model)
    _require_draw_parameters(draw_at_even, draw_shape)
    bands = tuple(range(1, band_count + 1))
    required = {"opponent_quintile", "expected_score", "actual_score", "weight"}
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"quintile history missing columns: {sorted(missing)}")
    if not history.empty:
        _validate_profile_history(history, band_count)

    overall_offset = estimate_outcome_offset(
        history.get("expected_score", pd.Series(dtype=float)),
        history.get("actual_score", pd.Series(dtype=float)),
        history.get("weight", pd.Series(dtype=float)),
        elo_scale=elo_scale,
        outcome_model=outcome_model,
        draw_at_even=draw_at_even,
        draw_shape=draw_shape,
    )
    band_offsets: list[float] = []
    observations: list[int] = []
    effective_matches: list[float] = []
    raw_specific: list[float] = []
    reliabilities: list[float] = []
    preliminary: list[float] = []
    for quintile in bands:
        band = history.loc[history["opponent_quintile"].eq(quintile)]
        estimate = estimate_outcome_offset(
            band.get("expected_score", pd.Series(dtype=float)),
            band.get("actual_score", pd.Series(dtype=float)),
            band.get("weight", pd.Series(dtype=float)),
            elo_scale=elo_scale,
            outcome_model=outcome_model,
            draw_at_even=draw_at_even,
            draw_shape=draw_shape,
        )
        observations.append(int(len(band)))
        effective_matches.append(estimate.effective_matches)
        raw = estimate.offset - overall_offset.offset if len(band) else 0.0
        reliability = (
            estimate.effective_matches
            / (estimate.effective_matches + shrinkage_matches)
            if estimate.effective_matches > 0.0
            else 0.0
        )
        raw_specific.append(float(raw))
        reliabilities.append(float(reliability))
        preliminary.append(float(np.clip(raw * reliability, -effect_cap, effect_cap)))
        band_offsets.append(estimate.offset)

    observed = np.asarray(effective_matches, dtype=float) > 0.0
    if int(observed.sum()) < 2:
        effects = np.zeros(band_count, dtype=float)
    else:
        center = float(
            np.average(
                np.asarray(preliminary, dtype=float)[observed],
                weights=np.asarray(effective_matches, dtype=float)[observed],
            )
        )
        effects = np.asarray(preliminary, dtype=float)
        effects[observed] = np.clip(
            effects[observed] - center,
            -effect_cap,
            effect_cap,
        )
        effects[~observed] = 0.0
    return QuintileProfile(
        club_id=str(club_id),
        overall_effect=float(overall_offset.offset),
        overall_observations=int(len(history)),
        overall_effective_matches=float(overall_offset.effective_matches),
        effects=tuple(float(value) for value in effects),
        observations=tuple(observations),
        effective_matches=tuple(float(value) for value in effective_matches),
        reliabilities=tuple(float(value) for value in reliabilities),
        raw_specific_effects=tuple(float(value) for value in raw_specific),
    )


@dataclass(frozen=True)
class EloOffsetEstimate:
    offset: float
    observations: int
    effective_matches: float


def estimate_outcome_offset(
    expected_scores: Iterable[float],
    actual_scores: Iterable[float],
    weights: Iterable[float],
    *,
    elo_scale: float,
    outcome_model: str,
    draw_at_even: float,
    draw_shape: float,
) -> EloOffsetEstimate:
    _require_outcome_model(outcome_model)
    if outcome_model == EXPECTED_SCORE_OUTCOME:
        return estimate_elo_offset(
            expected_scores,
            actual_scores,
            weights,
            elo_scale=elo_scale,
        )
    return estimate_categorical_1x2_elo_offset(
        expected_scores,
        actual_scores,
        weights,
        elo_scale=elo_scale,
        draw_at_even=draw_at_even,
        draw_shape=draw_shape,
    )


def estimate_categorical_1x2_elo_offset(
    expected_scores: Iterable[float],
    actual_scores: Iterable[float],
    weights: Iterable[float],
    *,
    elo_scale: float,
    draw_at_even: float,
    draw_shape: float,
) -> EloOffsetEstimate:
    """Fit one Elo offset against explicit win/draw/loss likelihood."""

    _require_positive("elo_scale", elo_scale)
    _require_draw_parameters(draw_at_even, draw_shape)
    expected = np.asarray(list(expected_scores), dtype=float)
    actual = np.asarray(list(actual_scores), dtype=float)
    sample_weights = np.asarray(list(weights), dtype=float)
    if not (len(expected) == len(actual) == len(sample_weights)):
        raise ValueError("expected_scores, actual_scores and weights must have equal length")
    if len(expected) == 0:
        return EloOffsetEstimate(0.0, 0, 0.0)
    if (
        ~np.isfinite(expected)
        | ~np.isfinite(actual)
        | ~np.isfinite(sample_weights)
        | (expected <= 0.0)
        | (expected >= 1.0)
        | ~np.isin(actual, (0.0, 0.5, 1.0))
        | (sample_weights <= 0.0)
    ).any():
        raise ValueError("invalid categorical Elo-offset observations")

    outcome = np.where(actual == 1.0, 0, np.where(actual == 0.5, 1, 2))
    base_logit = np.log(expected / (1.0 - expected))
    slope = math.log(10.0) / elo_scale

    def derivatives(offset: float) -> tuple[float, float, float]:
        probability = _logistic_array(base_logit + slope * offset)
        first_expected = probability * (1.0 - probability)
        second_expected = first_expected * (1.0 - 2.0 * probability)
        draw = draw_at_even * (
            4.0 * probability * (1.0 - probability)
        ) ** draw_shape
        draw_log_derivative = draw_shape * (1.0 - 2.0 * probability)
        first_draw = draw * draw_log_derivative
        second_draw = draw * (
            draw_log_derivative**2 - 2.0 * draw_shape * first_expected
        )
        probabilities = np.column_stack(
            (
                probability - 0.5 * draw,
                draw,
                1.0 - probability - 0.5 * draw,
            )
        )
        first = np.column_stack(
            (
                first_expected - 0.5 * first_draw,
                first_draw,
                -first_expected - 0.5 * first_draw,
            )
        )
        second = np.column_stack(
            (
                second_expected - 0.5 * second_draw,
                second_draw,
                -second_expected - 0.5 * second_draw,
            )
        )
        row = np.arange(len(expected))
        chosen = np.clip(probabilities[row, outcome], 1e-15, 1.0)
        chosen_first = first[row, outcome]
        chosen_second = second[row, outcome]
        nll = float(-np.sum(sample_weights * np.log(chosen)))
        gradient = float(
            -slope * np.sum(sample_weights * chosen_first / chosen)
        )
        hessian = float(
            -slope**2
            * np.sum(
                sample_weights
                * (chosen_second / chosen - (chosen_first / chosen) ** 2)
            )
        )
        return nll, gradient, hessian

    offset = 0.0
    for _ in range(60):
        nll, gradient, hessian = derivatives(offset)
        if not math.isfinite(hessian) or hessian <= 1e-14:
            break
        step = gradient / hessian
        candidate = float(np.clip(offset - step, -2000.0, 2000.0))
        candidate_nll = derivatives(candidate)[0]
        for _ in range(24):
            if candidate_nll <= nll + 1e-12:
                break
            candidate = 0.5 * (candidate + offset)
            candidate_nll = derivatives(candidate)[0]
        if candidate_nll > nll + 1e-12:
            break
        if abs(candidate - offset) < 1e-10:
            offset = candidate
            break
        offset = candidate
    return EloOffsetEstimate(offset, int(len(expected)), float(sample_weights.sum()))


def estimate_elo_offset(
    expected_scores: Iterable[float],
    actual_scores: Iterable[float],
    weights: Iterable[float],
    *,
    elo_scale: float,
) -> EloOffsetEstimate:
    _require_positive("elo_scale", elo_scale)
    expected = np.asarray(list(expected_scores), dtype=float)
    actual = np.asarray(list(actual_scores), dtype=float)
    sample_weights = np.asarray(list(weights), dtype=float)
    if not (len(expected) == len(actual) == len(sample_weights)):
        raise ValueError("expected_scores, actual_scores and weights must have equal length")
    if len(expected) == 0:
        return EloOffsetEstimate(0.0, 0, 0.0)
    if (
        ~np.isfinite(expected)
        | ~np.isfinite(actual)
        | ~np.isfinite(sample_weights)
        | (expected <= 0.0)
        | (expected >= 1.0)
        | ~np.isin(actual, (0.0, 0.5, 1.0))
        | (sample_weights <= 0.0)
    ).any():
        raise ValueError("invalid Elo-offset observations")
    slope = math.log(10.0) / elo_scale
    base_logit = np.log(expected / (1.0 - expected))
    offset = 0.0
    for _ in range(40):
        probability = _logistic_array(base_logit + slope * offset)
        gradient = float(np.sum(sample_weights * (actual - probability) * slope))
        hessian = float(
            -np.sum(sample_weights * probability * (1.0 - probability) * slope * slope)
        )
        if abs(hessian) < 1e-18:
            break
        updated = float(np.clip(offset - gradient / hessian, -2000.0, 2000.0))
        if abs(updated - offset) < 1e-10:
            offset = updated
            break
        offset = updated
    return EloOffsetEstimate(offset, int(len(expected)), float(sample_weights.sum()))


def contextual_matchup_expectation(
    base_expected_home_score: float,
    home_effect: float,
    away_effect: float,
    *,
    elo_scale: float,
    context_cap: float,
) -> MatchupExpectation:
    _require_probability("base_expected_home_score", base_expected_home_score, allow_zero=False)
    if base_expected_home_score >= 1.0:
        raise ValueError("base_expected_home_score must be less than one")
    _require_finite("home_effect", home_effect)
    _require_finite("away_effect", away_effect)
    _require_positive("elo_scale", elo_scale)
    _require_non_negative("context_cap", context_cap)
    raw = float(home_effect - away_effect)
    applied = float(np.clip(raw, -context_cap, context_cap))
    base_logit = math.log(base_expected_home_score / (1.0 - base_expected_home_score))
    expected = _logistic_scalar(base_logit + math.log(10.0) * applied / elo_scale)
    return MatchupExpectation(
        base_expected_home_score=float(base_expected_home_score),
        home_effect=float(home_effect),
        away_effect=float(away_effect),
        raw_offset=raw,
        applied_offset=applied,
        expected_home_score=expected,
    )


def profile_to_record(profile: QuintileProfile) -> dict[str, object]:
    record: dict[str, object] = {
        "club_id": profile.club_id,
        "overall_effect": profile.overall_effect,
        "overall_observations": profile.overall_observations,
        "overall_effective_matches": profile.overall_effective_matches,
    }
    for index, quintile in enumerate(range(1, len(profile.effects) + 1)):
        record[f"effect_q{quintile}"] = profile.effects[index]
        record[f"raw_specific_effect_q{quintile}"] = profile.raw_specific_effects[index]
        record[f"observations_q{quintile}"] = profile.observations[index]
        record[f"effective_matches_q{quintile}"] = profile.effective_matches[index]
        record[f"reliability_q{quintile}"] = profile.reliabilities[index]
    return record


def _validate_profile_history(history: pd.DataFrame, band_count: int = 5) -> None:
    quintiles = pd.to_numeric(history["opponent_quintile"], errors="raise")
    if not quintiles.isin(range(1, band_count + 1)).all():
        raise ValueError(f"opponent_quintile must be in 1..{band_count}")
    for column in ("expected_score", "actual_score", "weight"):
        numeric = pd.to_numeric(history[column], errors="raise")
        if not np.isfinite(numeric.to_numpy(float)).all():
            raise ValueError(f"{column} must be finite")
    if not history["expected_score"].between(0.0, 1.0, inclusive="neither").all():
        raise ValueError("expected_score must be strictly between zero and one")
    if not history["actual_score"].isin((0.0, 0.5, 1.0)).all():
        raise ValueError("actual_score must contain only 0, 0.5 or 1")
    if not history["weight"].gt(0.0).all():
        raise ValueError("weight must be positive")


def _logistic_array(values: np.ndarray) -> np.ndarray:
    positive = values >= 0.0
    result = np.empty_like(values, dtype=float)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


def _logistic_scalar(value: float) -> float:
    if value >= 0.0:
        exponential = math.exp(-value)
        return 1.0 / (1.0 + exponential)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _require_band_count(value: int) -> None:
    if value not in (3, 5):
        raise ValueError("band_count must be either 3 or 5")


def _require_band(value: int, band_count: int) -> None:
    if value not in range(1, band_count + 1):
        raise ValueError(f"strength band must be an integer in 1..{band_count}")


def _require_outcome_model(value: str) -> None:
    if value not in (EXPECTED_SCORE_OUTCOME, CATEGORICAL_1X2_OUTCOME):
        raise ValueError(f"Unknown outcome_model: {value}")


def _require_draw_parameters(draw_at_even: float, draw_shape: float) -> None:
    _require_finite("draw_at_even", draw_at_even)
    _require_finite("draw_shape", draw_shape)
    if not 0.0 < float(draw_at_even) <= 0.5:
        raise ValueError("draw_at_even must be in (0, 0.5]")
    if float(draw_shape) < 1.0:
        raise ValueError("draw_shape must be at least one")


def _require_finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


def _require_non_negative(name: str, value: float) -> None:
    _require_finite(name, value)
    if float(value) < 0.0:
        raise ValueError(f"{name} must be non-negative")


def _require_positive(name: str, value: float) -> None:
    _require_finite(name, value)
    if float(value) <= 0.0:
        raise ValueError(f"{name} must be positive")


def _require_probability(name: str, value: float, *, allow_zero: bool) -> None:
    _require_finite(name, value)
    lower = float(value) >= 0.0 if allow_zero else float(value) > 0.0
    if not lower or float(value) > 1.0:
        raise ValueError(f"{name} must be in {'[0, 1]' if allow_zero else '(0, 1]'}")
