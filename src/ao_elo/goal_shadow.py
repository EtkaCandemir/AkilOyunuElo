from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from typing import Iterable, Mapping

from ao_elo.controlled_live import update_match_elo
from ao_elo.dynamic import (
    DynamicEloConfig,
    MatchFixture,
    MatchInput,
    TeamSeed,
    expected_1x2_probabilities,
    expected_score,
)


@dataclass(frozen=True, order=True)
class GoalShadowArm:
    name: str
    alpha: float
    tau: float

    def validate(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Shadow arm name must be a non-empty string")
        if not math.isfinite(self.alpha) or self.alpha < 0.0:
            raise ValueError("Shadow arm alpha must be non-negative and finite")
        if not math.isfinite(self.tau) or self.tau <= 0.0:
            raise ValueError("Shadow arm tau must be positive and finite")


@dataclass(frozen=True)
class GoalShadowState:
    season: str
    arms: tuple[GoalShadowArm, ...]
    team_names: Mapping[str, str]
    ao_first_elo: Mapping[str, float]
    ratings: Mapping[str, Mapping[str, float]]
    processed_match_ids: frozenset[str]
    last_event_utc: datetime | None
    last_match_id: str | None
    active_config_id: str
    shadow_config_id: str


@dataclass(frozen=True)
class GoalShadowPrediction:
    arm_name: str
    alpha: float
    tau: float
    match_id: str
    season: str
    kickoff_utc: datetime
    generated_at_utc: datetime
    competition: str
    round: str
    home_team_id: str
    away_team_id: str
    is_neutral: bool
    home_rating_pre: float
    away_rating_pre: float
    expected_home_score: float
    home_win_probability: float
    draw_probability: float
    away_win_probability: float
    active_config_id: str
    shadow_config_id: str


@dataclass(frozen=True)
class GoalShadowUpdate:
    arm_name: str
    alpha: float
    tau: float
    match_id: str
    season: str
    kickoff_utc: datetime
    competition: str
    round: str
    home_team_id: str
    away_team_id: str
    home_goals: int
    away_goals: int
    decided_on_penalties: bool
    home_rating_pre: float
    away_rating_pre: float
    effective_rating_difference: float
    expected_home_score: float
    home_win_probability: float
    draw_probability: float
    away_win_probability: float
    actual_home_score: float
    goal_difference: int
    goal_multiplier: float
    power_delta: float
    home_rating_post: float
    away_rating_post: float
    zero_sum_error: float
    active_config_id: str
    shadow_config_id: str


def initialize_goal_shadow(
    season: str,
    seeds: Iterable[TeamSeed],
    arms: Iterable[GoalShadowArm],
    active_config: DynamicEloConfig,
) -> GoalShadowState:
    active_config.validate()
    if not isinstance(season, str) or not season.strip():
        raise ValueError("season must be a non-empty string")
    seed_list = list(seeds)
    arm_list = tuple(arms)
    if not seed_list:
        raise ValueError("At least one team seed is required")
    if not arm_list:
        raise ValueError("At least one shadow arm is required")
    for seed in seed_list:
        seed.validate()
    for arm in arm_list:
        arm.validate()
    team_ids = [seed.team_id for seed in seed_list]
    arm_names = [arm.name for arm in arm_list]
    if len(team_ids) != len(set(team_ids)):
        raise ValueError("Shadow team seeds must have unique team_id values")
    if len(arm_names) != len(set(arm_names)):
        raise ValueError("Shadow arms must have unique names")
    if "BASE" not in arm_names:
        raise ValueError("Shadow arms must include a BASE control")
    base = next(arm for arm in arm_list if arm.name == "BASE")
    if base.alpha != 0.0:
        raise ValueError("BASE shadow arm must set alpha=0")

    initial = {seed.team_id: float(seed.ao_first_elo) for seed in seed_list}
    state = GoalShadowState(
        season=season,
        arms=arm_list,
        team_names={seed.team_id: seed.team_name for seed in seed_list},
        ao_first_elo=initial,
        ratings={arm.name: dict(initial) for arm in arm_list},
        processed_match_ids=frozenset(),
        last_event_utc=None,
        last_match_id=None,
        active_config_id=active_config.config_id,
        shadow_config_id=shadow_config_id(arm_list, active_config),
    )
    validate_goal_shadow_state(state, active_config)
    return state


def lock_goal_shadow(
    state: GoalShadowState,
    fixture: MatchFixture,
    active_config: DynamicEloConfig,
    *,
    generated_at_utc: datetime,
) -> tuple[GoalShadowPrediction, ...]:
    validate_goal_shadow_state(state, active_config)
    fixture.validate()
    _validate_fixture(state, fixture)
    if not isinstance(generated_at_utc, datetime) or generated_at_utc.tzinfo is None:
        raise ValueError("generated_at_utc must be timezone-aware")
    if generated_at_utc >= fixture.kickoff_utc:
        raise ValueError("Shadow prediction must be generated before kickoff")
    predictions = []
    for arm in state.arms:
        ratings = state.ratings[arm.name]
        home = float(ratings[fixture.home_team_id])
        away = float(ratings[fixture.away_team_id])
        expected = expected_score(
            home,
            away,
            active_config,
            neutral=fixture.is_neutral,
        )
        home_probability, draw_probability, away_probability = (
            expected_1x2_probabilities(
                home,
                away,
                active_config,
                neutral=fixture.is_neutral,
            )
        )
        predictions.append(
            GoalShadowPrediction(
                arm_name=arm.name,
                alpha=arm.alpha,
                tau=arm.tau,
                match_id=fixture.match_id,
                season=fixture.season,
                kickoff_utc=fixture.kickoff_utc,
                generated_at_utc=generated_at_utc,
                competition=fixture.competition,
                round=fixture.round,
                home_team_id=fixture.home_team_id,
                away_team_id=fixture.away_team_id,
                is_neutral=fixture.is_neutral,
                home_rating_pre=home,
                away_rating_pre=away,
                expected_home_score=expected,
                home_win_probability=home_probability,
                draw_probability=draw_probability,
                away_win_probability=away_probability,
                active_config_id=state.active_config_id,
                shadow_config_id=state.shadow_config_id,
            )
        )
    return tuple(predictions)


def settle_goal_shadow(
    state: GoalShadowState,
    match: MatchInput,
    locked_predictions: Iterable[GoalShadowPrediction],
    active_config: DynamicEloConfig,
) -> tuple[GoalShadowState, tuple[GoalShadowUpdate, ...]]:
    validate_goal_shadow_state(state, active_config)
    match.validate()
    _validate_fixture(state, match.fixture())
    prediction_by_arm = {
        prediction.arm_name: prediction for prediction in locked_predictions
    }
    if set(prediction_by_arm) != {arm.name for arm in state.arms}:
        raise ValueError("Locked shadow predictions do not cover every configured arm")

    updated_ratings = {
        arm_name: dict(ratings)
        for arm_name, ratings in state.ratings.items()
    }
    updates = []
    for arm in state.arms:
        prediction = prediction_by_arm[arm.name]
        _validate_locked_prediction(state, match, prediction, arm)
        result = update_match_elo(
            prediction.home_rating_pre,
            prediction.away_rating_pre,
            match.home_goals,
            match.away_goals,
            k_factor=active_config.k_factor,
            elo_scale=active_config.elo_scale,
            home_advantage=active_config.home_advantage,
            is_neutral=match.is_neutral,
            decided_on_penalties=match.decided_on_penalties,
            alpha=arm.alpha,
            tau=arm.tau,
        )
        if not math.isclose(
            result.expected_home_score,
            prediction.expected_home_score,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("Locked shadow expectation changed before settlement")
        ratings = updated_ratings[arm.name]
        ratings[match.home_team_id] = result.home_rating_post
        ratings[match.away_team_id] = result.away_rating_post
        updates.append(
            GoalShadowUpdate(
                arm_name=arm.name,
                alpha=arm.alpha,
                tau=arm.tau,
                match_id=match.match_id,
                season=match.season,
                kickoff_utc=match.kickoff_utc,
                competition=match.competition,
                round=match.round,
                home_team_id=match.home_team_id,
                away_team_id=match.away_team_id,
                home_goals=match.home_goals,
                away_goals=match.away_goals,
                decided_on_penalties=match.decided_on_penalties,
                home_rating_pre=result.home_rating_pre,
                away_rating_pre=result.away_rating_pre,
                effective_rating_difference=result.effective_rating_difference,
                expected_home_score=result.expected_home_score,
                home_win_probability=prediction.home_win_probability,
                draw_probability=prediction.draw_probability,
                away_win_probability=prediction.away_win_probability,
                actual_home_score=result.actual_home_score,
                goal_difference=result.goal_difference,
                goal_multiplier=result.goal_difference_multiplier,
                power_delta=result.power_delta,
                home_rating_post=result.home_rating_post,
                away_rating_post=result.away_rating_post,
                zero_sum_error=result.zero_sum_error,
                active_config_id=state.active_config_id,
                shadow_config_id=state.shadow_config_id,
            )
        )

    updated_state = replace(
        state,
        ratings=updated_ratings,
        processed_match_ids=state.processed_match_ids | {match.match_id},
        last_event_utc=match.kickoff_utc,
        last_match_id=match.match_id,
    )
    validate_goal_shadow_state(updated_state, active_config)
    return updated_state, tuple(updates)


def validate_goal_shadow_state(
    state: GoalShadowState,
    active_config: DynamicEloConfig,
) -> None:
    active_config.validate()
    if state.active_config_id != active_config.config_id:
        raise ValueError("Shadow state does not match the active Elo config")
    if state.shadow_config_id != shadow_config_id(state.arms, active_config):
        raise ValueError("Shadow state arm fingerprint is invalid")
    if set(state.team_names) != set(state.ao_first_elo):
        raise ValueError("Shadow state team metadata is inconsistent")
    expected_teams = set(state.team_names)
    expected_arms = {arm.name for arm in state.arms}
    if set(state.ratings) != expected_arms:
        raise ValueError("Shadow state ratings do not cover every arm")
    for arm in state.arms:
        arm.validate()
        ratings = state.ratings[arm.name]
        if set(ratings) != expected_teams:
            raise ValueError(f"Shadow arm {arm.name} has incomplete team ratings")
        if any(not math.isfinite(float(value)) for value in ratings.values()):
            raise ValueError(f"Shadow arm {arm.name} contains invalid ratings")
    if (state.last_event_utc is None) != (state.last_match_id is None):
        raise ValueError("Shadow state chronology metadata is incomplete")


def shadow_config_id(
    arms: Iterable[GoalShadowArm],
    active_config: DynamicEloConfig,
) -> str:
    payload = {
        "active_config_id": active_config.config_id,
        "arms": [asdict(arm) for arm in sorted(tuple(arms))],
        "goal_difference_cap": 4,
        "penalty_rule": "S=0.5;M_GD=1",
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def _validate_fixture(
    state: GoalShadowState,
    fixture: MatchFixture,
) -> None:
    if fixture.season != state.season:
        raise ValueError("Shadow fixture season does not match state")
    if fixture.match_id in state.processed_match_ids:
        raise ValueError(f"Duplicate shadow match_id: {fixture.match_id}")
    if fixture.home_team_id not in state.team_names:
        raise ValueError(f"Missing shadow home team: {fixture.home_team_id}")
    if fixture.away_team_id not in state.team_names:
        raise ValueError(f"Missing shadow away team: {fixture.away_team_id}")
    if state.last_event_utc is not None:
        assert state.last_match_id is not None
        if (fixture.kickoff_utc, fixture.match_id) <= (
            state.last_event_utc,
            state.last_match_id,
        ):
            raise ValueError("Shadow fixture chronology regressed")


def _validate_locked_prediction(
    state: GoalShadowState,
    match: MatchInput,
    prediction: GoalShadowPrediction,
    arm: GoalShadowArm,
) -> None:
    expected = {
        "arm_name": arm.name,
        "alpha": arm.alpha,
        "tau": arm.tau,
        "match_id": match.match_id,
        "season": match.season,
        "kickoff_utc": match.kickoff_utc,
        "competition": match.competition,
        "round": match.round,
        "home_team_id": match.home_team_id,
        "away_team_id": match.away_team_id,
        "is_neutral": match.is_neutral,
        "active_config_id": state.active_config_id,
        "shadow_config_id": state.shadow_config_id,
    }
    for name, value in expected.items():
        if getattr(prediction, name) != value:
            raise ValueError(f"Locked shadow prediction {name} does not match")
    ratings = state.ratings[arm.name]
    if not math.isclose(
        prediction.home_rating_pre,
        float(ratings[match.home_team_id]),
        rel_tol=0.0,
        abs_tol=1e-9,
    ) or not math.isclose(
        prediction.away_rating_pre,
        float(ratings[match.away_team_id]),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("Shadow ratings changed after prediction lock")
