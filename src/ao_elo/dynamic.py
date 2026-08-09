from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from typing import Iterable, Mapping

from ao_elo.config import AO_MODEL_V2_VERSION, V2_RATING_MULTIPLIER
from ao_elo.controlled_live import calculate_goal_difference_multiplier
from ao_elo.tournament_bonus import (
    ELIGIBLE_PROGRESSION_STAGES,
    FixedTournamentBonusConfig,
    apply_tournament_progress_bonus,
)
from ao_elo.xg_live import (
    XGBlendConfig,
    XGPerformanceBonusConfig,
    update_match_elo_with_xg,
)


COMPETITIONS = ("UCL", "UEL", "UECL")
EXPECTED_SCORE_SEMANTICS = "normalized match points: win=1, draw=0.5, loss=0"
ONE_X_TWO_PROBABILITY_SEMANTICS = (
    "score-preserving H/D/A probabilities: P(H)+0.5*P(D)=expected_home_score"
)
STAGE_MULTIPLIERS: dict[str, dict[str, float]] = {
    "FLAT": {
        "QUALIFYING": 1.0,
        "KNOCKOUT_PLAYOFF": 1.0,
        "ROUND_OF_16": 1.0,
        "QUARTERFINAL": 1.0,
        "SEMIFINAL": 1.0,
        "FINAL": 0.0,
        "LEAGUE": 0.0,
    },
    "GENTLE_ASCENDING": {
        "QUALIFYING": 0.50,
        "KNOCKOUT_PLAYOFF": 0.75,
        "ROUND_OF_16": 1.00,
        "QUARTERFINAL": 1.15,
        "SEMIFINAL": 1.30,
        "FINAL": 0.0,
        "LEAGUE": 0.0,
    },
    "KNOCKOUT_ASCENDING": {
        "QUALIFYING": 0.25,
        "KNOCKOUT_PLAYOFF": 0.50,
        "ROUND_OF_16": 0.75,
        "QUARTERFINAL": 1.00,
        "SEMIFINAL": 1.25,
        "FINAL": 0.0,
        "LEAGUE": 0.0,
    },
    "LATE_BALANCED": {
        "QUALIFYING": 0.0,
        "KNOCKOUT_PLAYOFF": 0.40,
        "ROUND_OF_16": 0.70,
        "QUARTERFINAL": 1.00,
        "SEMIFINAL": 1.25,
        "FINAL": 0.0,
        "LEAGUE": 0.0,
    },
    "LATE_STRICT": {
        "QUALIFYING": 0.0,
        "KNOCKOUT_PLAYOFF": 0.0,
        "ROUND_OF_16": 0.50,
        "QUARTERFINAL": 0.90,
        "SEMIFINAL": 1.30,
        "FINAL": 0.0,
        "LEAGUE": 0.0,
    },
    "SEMIFINAL_HEAVY": {
        "QUALIFYING": 0.0,
        "KNOCKOUT_PLAYOFF": 0.25,
        "ROUND_OF_16": 0.50,
        "QUARTERFINAL": 1.00,
        "SEMIFINAL": 1.50,
        "FINAL": 0.0,
        "LEAGUE": 0.0,
    },
}


@dataclass(frozen=True)
class DynamicEloConfig:
    model_version: str
    elo_scale: float
    home_advantage: float
    k_factor: float
    power_carry: float
    draw_at_even: float = 0.24
    draw_shape: float = 1.0
    goal_difference_enabled: bool = False
    goal_alpha: float = 0.0
    goal_tau: float = 300.0
    goal_difference_cap: int = 4
    xg_performance_enabled: bool = False
    xg_performance_ratio: float = 0.0
    xg_performance_scale: float = 1.25
    minimum_winner_gain_ratio: float = 0.70
    progression_bonus_enabled: bool = False
    progression_base_bonus: float = 0.0
    progression_stages_per_competition: int = 5
    reserve_base: float = 0.0
    reserve_cap: float = 80.0 * V2_RATING_MULTIPLIER
    reserve_decay: float = 0.0
    ucl_multiplier: float = 1.0
    uel_multiplier: float = 0.65
    uecl_multiplier: float = 0.45
    stage_profile: str = "FLAT"
    trophy_uses_same_base: bool = True

    @classmethod
    def calibrated_v2(cls) -> DynamicEloConfig:
        """Return the active v2 parameters frozen after development and review."""
        return cls(
            model_version=AO_MODEL_V2_VERSION,
            elo_scale=225.0 * V2_RATING_MULTIPLIER,
            home_advantage=40.0 * V2_RATING_MULTIPLIER,
            k_factor=28.0 * V2_RATING_MULTIPLIER,
            power_carry=0.0,
            draw_at_even=0.24,
            draw_shape=1.0,
            goal_difference_enabled=True,
            goal_alpha=0.15,
            goal_tau=300.0,
            goal_difference_cap=4,
            xg_performance_enabled=True,
            xg_performance_ratio=0.30,
            xg_performance_scale=1.25,
            minimum_winner_gain_ratio=0.70,
            progression_bonus_enabled=True,
            progression_base_bonus=12.0,
            progression_stages_per_competition=5,
            reserve_base=0.0,
            reserve_decay=0.0,
        )

    def validate(self) -> None:
        if not isinstance(self.model_version, str) or not self.model_version.strip():
            raise ValueError("model_version must be a non-empty string")
        _require_positive_finite("elo_scale", self.elo_scale)
        _require_non_negative_finite("home_advantage", self.home_advantage)
        _require_non_negative_finite("k_factor", self.k_factor)
        _require_between_zero_and_one("power_carry", self.power_carry)
        _require_between_zero_and_one("draw_at_even", self.draw_at_even)
        if self.draw_at_even > 0.5:
            raise ValueError("draw_at_even must be <= 0.5")
        _require_positive_finite("draw_shape", self.draw_shape)
        if self.draw_shape < 1.0:
            raise ValueError("draw_shape must be >= 1.0")
        if not isinstance(self.goal_difference_enabled, bool):
            raise ValueError("goal_difference_enabled must be boolean")
        _require_non_negative_finite("goal_alpha", self.goal_alpha)
        _require_positive_finite("goal_tau", self.goal_tau)
        _require_positive_integer(
            "goal_difference_cap",
            self.goal_difference_cap,
        )
        if self.goal_difference_enabled and self.goal_alpha <= 0.0:
            raise ValueError(
                "Enabled goal-difference config must have positive goal_alpha"
            )
        if not self.goal_difference_enabled and self.goal_alpha != 0.0:
            raise ValueError(
                "Disabled goal-difference config must have goal_alpha=0"
            )
        if not isinstance(self.xg_performance_enabled, bool):
            raise ValueError("xg_performance_enabled must be boolean")
        _require_between_zero_and_one(
            "xg_performance_ratio",
            self.xg_performance_ratio,
        )
        _require_positive_finite(
            "xg_performance_scale",
            self.xg_performance_scale,
        )
        _require_between_zero_and_one(
            "minimum_winner_gain_ratio",
            self.minimum_winner_gain_ratio,
        )
        if self.minimum_winner_gain_ratio <= 0.0:
            raise ValueError("minimum_winner_gain_ratio must be positive")
        if self.xg_performance_enabled and self.xg_performance_ratio <= 0.0:
            raise ValueError(
                "Enabled xG performance config must have positive xg_performance_ratio"
            )
        if not self.xg_performance_enabled and self.xg_performance_ratio != 0.0:
            raise ValueError(
                "Disabled xG performance config must have xg_performance_ratio=0"
            )
        if not isinstance(self.progression_bonus_enabled, bool):
            raise ValueError("progression_bonus_enabled must be boolean")
        _require_non_negative_finite(
            "progression_base_bonus",
            self.progression_base_bonus,
        )
        _require_positive_integer(
            "progression_stages_per_competition",
            self.progression_stages_per_competition,
        )
        if self.progression_bonus_enabled and self.progression_base_bonus <= 0.0:
            raise ValueError(
                "Enabled progression bonus must have positive progression_base_bonus"
            )
        if not self.progression_bonus_enabled and self.progression_base_bonus != 0.0:
            raise ValueError(
                "Disabled progression bonus must have progression_base_bonus=0"
            )
        _require_non_negative_finite("reserve_base", self.reserve_base)
        _require_positive_finite("reserve_cap", self.reserve_cap)
        _require_between_zero_and_one("reserve_decay", self.reserve_decay)
        if self.reserve_base > self.reserve_cap:
            raise ValueError("reserve_base cannot exceed reserve_cap")
        if self.reserve_base == 0.0 and self.reserve_decay != 0.0:
            raise ValueError("reserve_decay must be zero when reserve is disabled")
        for name in ("ucl_multiplier", "uel_multiplier", "uecl_multiplier"):
            _require_positive_finite(name, getattr(self, name))
        if not self.ucl_multiplier > self.uel_multiplier > self.uecl_multiplier:
            raise ValueError("Competition hierarchy must satisfy UCL > UEL > UECL")
        if self.stage_profile not in STAGE_MULTIPLIERS:
            raise ValueError(f"Unknown stage_profile: {self.stage_profile}")
        if not isinstance(self.trophy_uses_same_base, bool):
            raise ValueError("trophy_uses_same_base must be boolean")

    @property
    def config_id(self) -> str:
        self.validate()
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def competition_multiplier(self, competition: str) -> float:
        self.validate()
        try:
            return {
                "UCL": self.ucl_multiplier,
                "UEL": self.uel_multiplier,
                "UECL": self.uecl_multiplier,
            }[competition]
        except KeyError as error:
            raise ValueError(f"Unknown competition: {competition}") from error

    def stage_multiplier(self, stage: str) -> float:
        self.validate()
        try:
            return STAGE_MULTIPLIERS[self.stage_profile][stage]
        except KeyError as error:
            raise ValueError(f"Unknown stage: {stage}") from error

    @property
    def fixed_progression_config(self) -> FixedTournamentBonusConfig:
        config = FixedTournamentBonusConfig(
            base_bonus=self.progression_base_bonus,
            stages_per_competition=self.progression_stages_per_competition,
        )
        config.validate()
        return config

    @property
    def xg_performance_config(self) -> XGPerformanceBonusConfig:
        config = XGPerformanceBonusConfig(
            beta=self.xg_performance_ratio,
            xg_scale=self.xg_performance_scale,
            minimum_winner_gain_ratio=self.minimum_winner_gain_ratio,
        )
        config.validate()
        return config


@dataclass(frozen=True)
class TeamSeed:
    team_id: str
    team_name: str
    ao_first_elo: float

    def validate(self) -> None:
        _require_identifier("team_id", self.team_id)
        _require_identifier("team_name", self.team_name)
        _require_finite("ao_first_elo", self.ao_first_elo)


@dataclass(frozen=True)
class TeamRating:
    team_id: str
    team_name: str
    ao_first_elo: float
    power_elo: float
    achievement_reserve: float
    progression_bonus_ucl: float = 0.0
    progression_bonus_uel: float = 0.0
    progression_bonus_uecl: float = 0.0
    last_event_utc: datetime | None = None
    last_match_id: str | None = None

    @property
    def ao_live_elo(self) -> float:
        return self.power_elo + self.achievement_reserve + self.progression_bonus_total

    @property
    def progression_bonus_total(self) -> float:
        return (
            self.progression_bonus_ucl
            + self.progression_bonus_uel
            + self.progression_bonus_uecl
        )

    def progression_bonus_for(self, competition: str) -> float:
        try:
            return {
                "UCL": self.progression_bonus_ucl,
                "UEL": self.progression_bonus_uel,
                "UECL": self.progression_bonus_uecl,
            }[competition]
        except KeyError as error:
            raise ValueError(f"Unknown competition: {competition}") from error


@dataclass(frozen=True)
class TieState:
    tie_id: str
    team_a_id: str
    team_b_id: str
    expected_a_to_advance: float
    competition: str
    stage: str


@dataclass(frozen=True)
class SeasonState:
    season: str
    ratings: Mapping[str, TeamRating]
    processed_match_ids: frozenset[str]
    processed_tie_ids: frozenset[str]
    open_ties: Mapping[str, TieState]
    last_event_utc: datetime | None
    last_match_id: str | None
    model_version: str
    config_id: str


@dataclass(frozen=True)
class MatchFixture:
    match_id: str
    season: str
    kickoff_utc: datetime
    competition: str
    round: str
    home_team_id: str
    away_team_id: str
    is_neutral: bool
    tie_id: str | None = None
    is_knockout: bool = False
    is_tie_decider: bool = False
    stage: str | None = None

    def validate(self) -> None:
        _require_identifier("match_id", self.match_id)
        _require_identifier("season", self.season)
        _require_utc_datetime("kickoff_utc", self.kickoff_utc)
        if self.competition not in COMPETITIONS:
            raise ValueError(f"Unknown competition: {self.competition}")
        _require_identifier("round", self.round)
        _require_identifier("home_team_id", self.home_team_id)
        _require_identifier("away_team_id", self.away_team_id)
        if self.home_team_id == self.away_team_id:
            raise ValueError("home_team_id and away_team_id must differ")
        for name in ("is_neutral", "is_knockout", "is_tie_decider"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        if self.is_knockout and not self.tie_id:
            raise ValueError("Knockout matches require tie_id")
        if not self.is_knockout and self.tie_id is not None:
            raise ValueError("Non-knockout matches cannot have tie_id")
        if self.is_tie_decider and not self.is_knockout:
            raise ValueError("A tie decider must be a knockout match")
        if self.stage is not None and self.stage not in STAGE_MULTIPLIERS["FLAT"]:
            raise ValueError(f"Unknown stage: {self.stage}")
        if self.is_knockout and self.stage == "LEAGUE":
            raise ValueError("Knockout matches cannot use the LEAGUE stage")
        if not self.is_knockout and self.stage not in (None, "LEAGUE"):
            raise ValueError("Non-knockout matches must use the LEAGUE stage")


@dataclass(frozen=True)
class MatchInput:
    match_id: str
    season: str
    kickoff_utc: datetime
    competition: str
    round: str
    home_team_id: str
    away_team_id: str
    home_goals: int
    away_goals: int
    is_neutral: bool
    decided_on_penalties: bool
    tie_id: str | None = None
    is_knockout: bool = False
    is_tie_decider: bool = False
    advanced_team_id: str | None = None
    stage: str | None = None
    xg_home: float | None = None
    xg_away: float | None = None
    xg_analysis_eligible: bool = False

    def fixture(self) -> MatchFixture:
        return MatchFixture(
            match_id=self.match_id,
            season=self.season,
            kickoff_utc=self.kickoff_utc,
            competition=self.competition,
            round=self.round,
            home_team_id=self.home_team_id,
            away_team_id=self.away_team_id,
            is_neutral=self.is_neutral,
            tie_id=self.tie_id,
            is_knockout=self.is_knockout,
            is_tie_decider=self.is_tie_decider,
            stage=self.stage,
        )

    def validate(self) -> None:
        self.fixture().validate()
        _require_non_negative_integer("home_goals", self.home_goals)
        _require_non_negative_integer("away_goals", self.away_goals)
        if not isinstance(self.decided_on_penalties, bool):
            raise ValueError("decided_on_penalties must be boolean")
        if self.is_tie_decider and not self.advanced_team_id:
            raise ValueError("A tie decider requires advanced_team_id")
        if self.advanced_team_id is not None and self.advanced_team_id not in (
            self.home_team_id,
            self.away_team_id,
        ):
            raise ValueError("advanced_team_id must identify a match participant")
        if self.advanced_team_id is not None and not self.is_tie_decider:
            raise ValueError("advanced_team_id is only valid on a tie decider")
        if self.decided_on_penalties and not self.is_tie_decider:
            raise ValueError("decided_on_penalties requires a deciding knockout match")
        if not isinstance(self.xg_analysis_eligible, bool):
            raise ValueError("xg_analysis_eligible must be boolean")
        if (self.xg_home is None) != (self.xg_away is None):
            raise ValueError("xg_home and xg_away must be provided together")
        if self.xg_home is not None:
            _require_non_negative_finite("xg_home", self.xg_home)
            assert self.xg_away is not None
            _require_non_negative_finite("xg_away", self.xg_away)
        if self.xg_analysis_eligible and self.xg_home is None:
            raise ValueError("Eligible xG requires both xg_home and xg_away")
        if not self.xg_analysis_eligible and self.xg_home is not None:
            raise ValueError("Ineligible xG rows must leave xg_home and xg_away empty")


@dataclass(frozen=True)
class LockedPrediction:
    match_id: str
    season: str
    kickoff_utc: datetime
    generated_at_utc: datetime
    competition: str
    round: str
    home_team_id: str
    away_team_id: str
    tie_id: str | None
    stage: str
    is_knockout: bool
    is_tie_decider: bool
    is_neutral: bool
    home_power_pre: float
    away_power_pre: float
    home_reserve_pre: float
    away_reserve_pre: float
    home_live_pre: float
    away_live_pre: float
    expected_home_score: float
    expected_score_semantics: str
    home_win_probability: float
    draw_probability: float
    away_win_probability: float
    probability_semantics: str
    state_last_event_utc: datetime | None
    state_last_match_id: str | None
    model_version: str
    config_id: str

    def validate(self) -> None:
        _require_identifier("prediction match_id", self.match_id)
        _require_identifier("prediction season", self.season)
        _require_utc_datetime("prediction kickoff_utc", self.kickoff_utc)
        _require_utc_datetime("prediction generated_at_utc", self.generated_at_utc)
        if self.generated_at_utc >= self.kickoff_utc:
            raise ValueError("Prediction must be locked before kickoff_utc")
        if self.competition not in COMPETITIONS:
            raise ValueError(f"Unknown prediction competition: {self.competition}")
        for name in ("round", "home_team_id", "away_team_id", "stage"):
            _require_identifier(f"prediction {name}", getattr(self, name))
        for name in ("is_knockout", "is_tie_decider", "is_neutral"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"prediction {name} must be boolean")
        if self.is_knockout and not self.tie_id:
            raise ValueError("Knockout prediction requires tie_id")
        if not self.is_knockout and self.tie_id is not None:
            raise ValueError("Non-knockout prediction cannot have tie_id")
        if self.is_tie_decider and not self.is_knockout:
            raise ValueError("A deciding prediction must be a knockout match")
        for name in (
            "home_power_pre",
            "away_power_pre",
            "home_reserve_pre",
            "away_reserve_pre",
            "home_live_pre",
            "away_live_pre",
            "expected_home_score",
        ):
            _require_finite(f"prediction {name}", getattr(self, name))
        _require_between_zero_and_one(
            "prediction expected_home_score",
            self.expected_home_score,
        )
        if self.expected_score_semantics != EXPECTED_SCORE_SEMANTICS:
            raise ValueError("Prediction expected-score semantics do not match the model")
        _validate_1x2_probabilities(
            self.home_win_probability,
            self.draw_probability,
            self.away_win_probability,
            self.expected_home_score,
            "prediction",
        )
        if self.probability_semantics != ONE_X_TWO_PROBABILITY_SEMANTICS:
            raise ValueError("Prediction 1X2 semantics do not match the model")
        if (self.state_last_event_utc is None) != (self.state_last_match_id is None):
            raise ValueError("Prediction state event metadata is incomplete")
        if self.state_last_event_utc is not None:
            _require_utc_datetime(
                "prediction state_last_event_utc",
                self.state_last_event_utc,
            )
            assert self.state_last_match_id is not None
            _require_identifier(
                "prediction state_last_match_id",
                self.state_last_match_id,
            )
        _require_identifier("prediction model_version", self.model_version)
        _require_identifier("prediction config_id", self.config_id)


@dataclass(frozen=True)
class MatchUpdate:
    match_id: str
    season: str
    kickoff_utc: datetime
    competition: str
    round: str
    home_team_id: str
    away_team_id: str
    tie_id: str | None
    stage: str
    is_knockout: bool
    is_tie_decider: bool
    advanced_team_id: str | None
    home_goals: int
    away_goals: int
    is_neutral: bool
    decided_on_penalties: bool
    home_power_pre: float
    away_power_pre: float
    home_reserve_pre: float
    away_reserve_pre: float
    home_progression_bonus_pre: float
    away_progression_bonus_pre: float
    home_live_pre: float
    away_live_pre: float
    effective_rating_difference: float
    expected_home_score: float
    home_win_probability: float
    draw_probability: float
    away_win_probability: float
    actual_home_score: float
    goal_difference_enabled: bool
    goal_alpha: float
    goal_tau: float
    goal_difference_cap: int
    goal_difference: int
    goal_multiplier: float
    xg_performance_enabled: bool
    xg_analysis_eligible: bool
    xg_applied: bool
    xg_fallback_used: bool
    xg_home: float | None
    xg_away: float | None
    xg_performance_ratio: float
    xg_performance_scale: float
    minimum_winner_gain_ratio: float
    xg_performance_signal: float | None
    power_delta_before_xg: float
    xg_power_adjustment: float
    power_delta: float
    home_reserve_delta: float
    away_reserve_delta: float
    home_progression_bonus_delta: float
    away_progression_bonus_delta: float
    home_power_post: float
    away_power_post: float
    home_reserve_post: float
    away_reserve_post: float
    home_progression_bonus_post: float
    away_progression_bonus_post: float
    home_live_post: float
    away_live_post: float
    winner_probability: float | None
    progression_bonus_recipient_id: str | None
    progression_bonus_added: float
    progression_bonus_competition_pre: float
    progression_bonus_competition_post: float
    progression_bonus_competition_cap: float
    progression_reserve_added: float
    trophy_reserve_added: float
    model_version: str
    config_id: str


def expected_score(
    home_rating: float,
    away_rating: float,
    config: DynamicEloConfig,
    *,
    neutral: bool = False,
) -> float:
    """Compute the expected home score using the selected v2 Elo scale."""
    config.validate()
    _require_finite("home_rating", home_rating)
    _require_finite("away_rating", away_rating)
    if not isinstance(neutral, bool):
        raise ValueError("neutral must be boolean")
    advantage = 0.0 if neutral else config.home_advantage
    exponent = -((float(home_rating) - float(away_rating) + advantage) / config.elo_scale)
    if exponent >= 300.0:
        return 0.0
    if exponent <= -300.0:
        return 1.0
    return 1.0 / (1.0 + 10.0**exponent)


def expected_1x2_probabilities(
    home_rating: float,
    away_rating: float,
    config: DynamicEloConfig,
    *,
    neutral: bool = False,
) -> tuple[float, float, float]:
    """Return calibrated home-win, draw, and away-win probabilities."""
    expected = expected_score(home_rating, away_rating, config, neutral=neutral)
    draw = config.draw_at_even * (
        4.0 * expected * (1.0 - expected)
    ) ** config.draw_shape
    home = expected - 0.5 * draw
    away = 1.0 - expected - 0.5 * draw
    _validate_1x2_probabilities(home, draw, away, expected, "expected")
    return float(home), float(draw), float(away)


def lock_prediction(
    state: SeasonState,
    fixture: MatchFixture,
    config: DynamicEloConfig,
    *,
    generated_at_utc: datetime,
) -> LockedPrediction:
    """Create a result-free, pre-kickoff prediction from the current state."""
    config.validate()
    _validate_state_config(state, config)
    fixture.validate()
    _require_utc_datetime("generated_at_utc", generated_at_utc)
    if generated_at_utc >= fixture.kickoff_utc:
        raise ValueError("Prediction must be locked before kickoff_utc")
    if state.last_event_utc is not None and generated_at_utc < state.last_event_utc:
        raise ValueError("Prediction timestamp cannot predate the current state")
    if fixture.season != state.season:
        raise ValueError(
            f"Fixture season {fixture.season} does not match state season {state.season}"
        )
    if fixture.match_id in state.processed_match_ids:
        raise ValueError(f"Duplicate match_id: {fixture.match_id}")
    if fixture.home_team_id not in state.ratings:
        raise ValueError(f"Missing home team_id in state: {fixture.home_team_id}")
    if fixture.away_team_id not in state.ratings:
        raise ValueError(f"Missing away team_id in state: {fixture.away_team_id}")
    _validate_chronology_key(state, fixture.kickoff_utc, fixture.match_id)

    home = state.ratings[fixture.home_team_id]
    away = state.ratings[fixture.away_team_id]
    stage = fixture.stage or normalize_stage(
        fixture.round,
        fixture.competition,
        fixture.is_knockout,
    )
    if fixture.is_knockout and fixture.tie_id in state.open_ties:
        assert fixture.tie_id is not None
        _validate_existing_tie(
            state.open_ties[fixture.tie_id],
            fixture.home_team_id,
            fixture.away_team_id,
            fixture.competition,
            stage,
        )
    probability = expected_score(
        home.ao_live_elo,
        away.ao_live_elo,
        config,
        neutral=fixture.is_neutral,
    )
    home_win_probability, draw_probability, away_win_probability = (
        expected_1x2_probabilities(
            home.ao_live_elo,
            away.ao_live_elo,
            config,
            neutral=fixture.is_neutral,
        )
    )
    prediction = LockedPrediction(
        match_id=fixture.match_id,
        season=fixture.season,
        kickoff_utc=fixture.kickoff_utc,
        generated_at_utc=generated_at_utc,
        competition=fixture.competition,
        round=fixture.round,
        home_team_id=fixture.home_team_id,
        away_team_id=fixture.away_team_id,
        tie_id=fixture.tie_id,
        stage=stage,
        is_knockout=fixture.is_knockout,
        is_tie_decider=fixture.is_tie_decider,
        is_neutral=fixture.is_neutral,
        home_power_pre=home.power_elo,
        away_power_pre=away.power_elo,
        home_reserve_pre=home.achievement_reserve,
        away_reserve_pre=away.achievement_reserve,
        home_live_pre=home.ao_live_elo,
        away_live_pre=away.ao_live_elo,
        expected_home_score=probability,
        expected_score_semantics=EXPECTED_SCORE_SEMANTICS,
        home_win_probability=home_win_probability,
        draw_probability=draw_probability,
        away_win_probability=away_win_probability,
        probability_semantics=ONE_X_TWO_PROBABILITY_SEMANTICS,
        state_last_event_utc=state.last_event_utc,
        state_last_match_id=state.last_match_id,
        model_version=config.model_version,
        config_id=config.config_id,
    )
    prediction.validate()
    return prediction


def settle_locked_match(
    state: SeasonState,
    match: MatchInput,
    prediction: LockedPrediction,
    config: DynamicEloConfig,
) -> tuple[SeasonState, MatchUpdate]:
    """Settle a result only when it matches an earlier locked prediction."""
    prediction.validate()
    match.validate()
    _validate_state_config(state, config)
    stage = match.stage or normalize_stage(
        match.round,
        match.competition,
        match.is_knockout,
    )
    expected_metadata = {
        "match_id": match.match_id,
        "season": match.season,
        "kickoff_utc": match.kickoff_utc,
        "competition": match.competition,
        "round": match.round,
        "home_team_id": match.home_team_id,
        "away_team_id": match.away_team_id,
        "tie_id": match.tie_id,
        "stage": stage,
        "is_knockout": match.is_knockout,
        "is_tie_decider": match.is_tie_decider,
        "is_neutral": match.is_neutral,
    }
    for name, value in expected_metadata.items():
        if getattr(prediction, name) != value:
            raise ValueError(f"Locked prediction {name} does not match the result")
    if prediction.model_version != config.model_version:
        raise ValueError("Locked prediction model_version does not match active config")
    if prediction.config_id != config.config_id:
        raise ValueError("Locked prediction config_id does not match active config")
    state_position_changed = (
        prediction.state_last_event_utc != state.last_event_utc
        or prediction.state_last_match_id != state.last_match_id
    )
    same_kickoff_predecessor = (
        state.last_event_utc == match.kickoff_utc
        and state.last_match_id is not None
        and state.last_match_id < match.match_id
    )
    if state_position_changed and not same_kickoff_predecessor:
        raise ValueError("State changed after the prediction was locked")
    if match.home_team_id not in state.ratings or match.away_team_id not in state.ratings:
        raise ValueError("Locked match references a team missing from current state")
    home = state.ratings[match.home_team_id]
    away = state.ratings[match.away_team_id]
    locked_values = {
        "home_power_pre": home.power_elo,
        "away_power_pre": away.power_elo,
        "home_reserve_pre": home.achievement_reserve,
        "away_reserve_pre": away.achievement_reserve,
        "home_live_pre": home.ao_live_elo,
        "away_live_pre": away.ao_live_elo,
    }
    for name, value in locked_values.items():
        if not math.isclose(
            float(getattr(prediction, name)),
            float(value),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(f"State {name} changed after the prediction was locked")
    current_expectation = expected_score(
        home.ao_live_elo,
        away.ao_live_elo,
        config,
        neutral=match.is_neutral,
    )
    if not math.isclose(
        prediction.expected_home_score,
        current_expectation,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Locked expected_home_score does not match current state")
    current_1x2 = expected_1x2_probabilities(
        home.ao_live_elo,
        away.ao_live_elo,
        config,
        neutral=match.is_neutral,
    )
    locked_1x2 = (
        prediction.home_win_probability,
        prediction.draw_probability,
        prediction.away_win_probability,
    )
    if any(
        not math.isclose(locked, current, rel_tol=0.0, abs_tol=1e-12)
        for locked, current in zip(locked_1x2, current_1x2)
    ):
        raise ValueError("Locked 1X2 probabilities do not match current state")

    updated_state, update = update_match(state, match, config)
    if not math.isclose(
        update.expected_home_score,
        prediction.expected_home_score,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Settlement changed the locked expected_home_score")
    update_1x2 = (
        update.home_win_probability,
        update.draw_probability,
        update.away_win_probability,
    )
    if any(
        not math.isclose(locked, settled, rel_tol=0.0, abs_tol=1e-12)
        for locked, settled in zip(locked_1x2, update_1x2)
    ):
        raise ValueError("Settlement changed the locked 1X2 probabilities")
    return updated_state, update


def initialize_season(
    season: str,
    seeds: Iterable[TeamSeed],
    config: DynamicEloConfig,
    *,
    previous_state: SeasonState | None = None,
) -> SeasonState:
    """Create a season state, applying calibrated power carry when available."""
    config.validate()
    _require_identifier("season", season)
    seed_list = list(seeds)
    if not seed_list:
        raise ValueError("At least one team seed is required")
    identifiers = []
    for seed in seed_list:
        seed.validate()
        identifiers.append(seed.team_id)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Team seeds must have unique team_id values")
    if previous_state is not None:
        _validate_state_config(previous_state, config)
        if _season_start_year(previous_state.season) >= _season_start_year(season):
            raise ValueError("previous_state must come from an earlier season")

    ratings: dict[str, TeamRating] = {}
    for seed in seed_list:
        previous = (
            previous_state.ratings.get(seed.team_id)
            if previous_state is not None
            else None
        )
        power = seed.ao_first_elo
        reserve = 0.0
        if previous is not None:
            power = (
                (1.0 - config.power_carry) * seed.ao_first_elo
                + config.power_carry * previous.power_elo
            )
            reserve = min(
                config.reserve_cap,
                config.reserve_decay * previous.achievement_reserve,
            )
        ratings[seed.team_id] = TeamRating(
            team_id=seed.team_id,
            team_name=seed.team_name,
            ao_first_elo=float(seed.ao_first_elo),
            power_elo=float(power),
            achievement_reserve=float(reserve),
        )
    return SeasonState(
        season=season,
        ratings=ratings,
        processed_match_ids=frozenset(),
        processed_tie_ids=frozenset(),
        open_ties={},
        last_event_utc=None,
        last_match_id=None,
        model_version=config.model_version,
        config_id=config.config_id,
    )


def apply_progression(
    state: SeasonState,
    winner_team_id: str,
    competition: str,
    stage: str,
    winner_probability: float,
    config: DynamicEloConfig,
    *,
    trophy: bool = False,
) -> tuple[SeasonState, float]:
    """Add non-zero-sum reserve after a tie is decided, subject to the cap."""
    _validate_state_config(state, config)
    if winner_team_id not in state.ratings:
        raise ValueError(f"Unknown winner team_id: {winner_team_id}")
    if not 0.0 <= float(winner_probability) <= 1.0:
        raise ValueError("winner_probability must be in [0,1]")
    if not isinstance(trophy, bool):
        raise ValueError("trophy must be boolean")
    competition_multiplier = config.competition_multiplier(competition)
    stage_multiplier = 1.0 if trophy else config.stage_multiplier(stage)
    requested = (
        config.reserve_base
        * competition_multiplier
        * stage_multiplier
        * (1.0 - float(winner_probability))
    )
    winner = state.ratings[winner_team_id]
    addition = min(
        max(0.0, config.reserve_cap - winner.achievement_reserve),
        requested,
    )
    if addition == 0.0:
        return state, 0.0
    ratings = dict(state.ratings)
    ratings[winner_team_id] = replace(
        winner,
        achievement_reserve=winner.achievement_reserve + addition,
    )
    return replace(state, ratings=ratings), addition


def update_match(
    state: SeasonState,
    match: MatchInput,
    config: DynamicEloConfig,
) -> tuple[SeasonState, MatchUpdate]:
    """Process one match without mutating the supplied state."""
    config.validate()
    _validate_state_config(state, config)
    match.validate()
    if match.season != state.season:
        raise ValueError(
            f"Match season {match.season} does not match state season {state.season}"
        )
    if match.match_id in state.processed_match_ids:
        raise ValueError(f"Duplicate match_id: {match.match_id}")
    if match.home_team_id not in state.ratings:
        raise ValueError(f"Missing home team_id in state: {match.home_team_id}")
    if match.away_team_id not in state.ratings:
        raise ValueError(f"Missing away team_id in state: {match.away_team_id}")
    _validate_chronology(state, match)

    ratings = dict(state.ratings)
    open_ties = dict(state.open_ties)
    processed_tie_ids = set(state.processed_tie_ids)
    home = ratings[match.home_team_id]
    away = ratings[match.away_team_id]
    stage = match.stage or normalize_stage(
        match.round,
        match.competition,
        match.is_knockout,
    )
    if match.is_knockout and match.tie_id in processed_tie_ids:
        raise ValueError(f"Knockout tie_id already completed: {match.tie_id}")
    if match.is_knockout and match.tie_id not in open_ties:
        assert match.tie_id is not None
        open_ties[match.tie_id] = TieState(
            tie_id=match.tie_id,
            team_a_id=match.home_team_id,
            team_b_id=match.away_team_id,
            expected_a_to_advance=expected_score(
                home.ao_live_elo,
                away.ao_live_elo,
                config,
                neutral=True,
            ),
            competition=match.competition,
            stage=stage,
        )
    elif match.is_knockout:
        assert match.tie_id is not None
        tie = open_ties[match.tie_id]
        _validate_existing_tie(
            tie,
            match.home_team_id,
            match.away_team_id,
            match.competition,
            stage,
        )

    probability = expected_score(
        home.ao_live_elo,
        away.ao_live_elo,
        config,
        neutral=match.is_neutral,
    )
    home_win_probability, draw_probability, away_win_probability = (
        expected_1x2_probabilities(
            home.ao_live_elo,
            away.ao_live_elo,
            config,
            neutral=match.is_neutral,
        )
    )
    xg_home = match.xg_home if match.xg_analysis_eligible else None
    xg_away = match.xg_away if match.xg_analysis_eligible else None
    elo_update = update_match_elo_with_xg(
        home.ao_live_elo,
        away.ao_live_elo,
        match.home_goals,
        match.away_goals,
        k_factor=config.k_factor,
        elo_scale=config.elo_scale,
        home_advantage=config.home_advantage,
        is_neutral=match.is_neutral,
        decided_on_penalties=match.decided_on_penalties,
        goal_difference_enabled=config.goal_difference_enabled,
        goal_alpha=config.goal_alpha,
        goal_tau=config.goal_tau,
        goal_difference_cap=config.goal_difference_cap,
        xg_config=XGBlendConfig(0.0, 1.0),
        xg_home=xg_home,
        xg_away=xg_away,
        xg_performance_bonus_config=(
            config.xg_performance_config
            if config.xg_performance_enabled
            else None
        ),
    )
    actual = elo_update.actual_home_score
    difference = elo_update.goal_difference
    effective_rating_difference = elo_update.effective_rating_difference
    multiplier = elo_update.goal_difference_multiplier
    delta = elo_update.power_delta
    power_delta_before_xg = config.k_factor * elo_update.result_residual
    xg_power_adjustment = delta - power_delta_before_xg
    xg_applied = (
        config.xg_performance_enabled
        and match.xg_analysis_eligible
        and actual != 0.5
        and not match.decided_on_penalties
    )
    home_post = replace(
        home,
        power_elo=home.power_elo + delta,
        last_event_utc=match.kickoff_utc,
        last_match_id=match.match_id,
    )
    away_post = replace(
        away,
        power_elo=away.power_elo - delta,
        last_event_utc=match.kickoff_utc,
        last_match_id=match.match_id,
    )
    if abs(
        (home_post.power_elo + away_post.power_elo)
        - (home.power_elo + away.power_elo)
    ) > 1e-9:
        raise ValueError("Power Elo update violated the zero-sum invariant")
    ratings[match.home_team_id] = home_post
    ratings[match.away_team_id] = away_post
    working = replace(state, ratings=ratings, open_ties=open_ties)

    winner_probability: float | None = None
    progression_bonus_recipient_id: str | None = None
    progression_bonus_added = 0.0
    progression_bonus_competition_pre = 0.0
    progression_bonus_competition_post = 0.0
    progression_bonus_competition_cap = config.fixed_progression_config.cap(
        match.competition
    )
    progression_added = 0.0
    trophy_added = 0.0
    if match.is_tie_decider:
        assert match.tie_id is not None and match.advanced_team_id is not None
        if match.tie_id not in open_ties:
            raise ValueError(f"Unknown deciding tie_id: {match.tie_id}")
        tie = open_ties.pop(match.tie_id)
        if match.advanced_team_id not in (tie.team_a_id, tie.team_b_id):
            raise ValueError("advanced_team_id does not belong to the recorded tie")
        winner_probability = (
            tie.expected_a_to_advance
            if match.advanced_team_id == tie.team_a_id
            else 1.0 - tie.expected_a_to_advance
        )
        if match.tie_id in processed_tie_ids:
            raise ValueError(f"Progression bonus already processed for tie_id: {match.tie_id}")
        winner = working.ratings[match.advanced_team_id]
        progression_bonus_competition_pre = winner.progression_bonus_for(
            match.competition
        )
        if (
            config.progression_bonus_enabled
            and stage in ELIGIBLE_PROGRESSION_STAGES
        ):
            bonus_update = apply_tournament_progress_bonus(
                progression_bonus_competition_pre,
                match.competition,
                stage,
                match.tie_id,
                processed_tie_ids,
                config.fixed_progression_config,
            )
            progression_bonus_added = bonus_update.applied_bonus
            progression_bonus_competition_post = bonus_update.bonus_post
            progression_bonus_competition_cap = bonus_update.competition_cap
            progression_bonus_recipient_id = match.advanced_team_id
            field_name = {
                "UCL": "progression_bonus_ucl",
                "UEL": "progression_bonus_uel",
                "UECL": "progression_bonus_uecl",
            }[match.competition]
            winner = replace(
                winner,
                **{field_name: progression_bonus_competition_post},
            )
            working_ratings = dict(working.ratings)
            working_ratings[match.advanced_team_id] = winner
            working = replace(working, ratings=working_ratings)
        else:
            processed_tie_ids.add(match.tie_id)
            progression_bonus_competition_post = progression_bonus_competition_pre
        working = replace(
            working,
            open_ties=open_ties,
            processed_tie_ids=frozenset(processed_tie_ids),
        )
        working, progression_added = apply_progression(
            working,
            match.advanced_team_id,
            match.competition,
            stage,
            winner_probability,
            config,
        )
        if stage == "FINAL" and config.trophy_uses_same_base:
            working, trophy_added = apply_progression(
                working,
                match.advanced_team_id,
                match.competition,
                stage,
                winner_probability,
                config,
                trophy=True,
            )

    final_ratings = dict(working.ratings)
    final_home = final_ratings[match.home_team_id]
    final_away = final_ratings[match.away_team_id]
    updated_state = replace(
        working,
        ratings=final_ratings,
        processed_match_ids=state.processed_match_ids | {match.match_id},
        open_ties=open_ties,
        last_event_utc=match.kickoff_utc,
        last_match_id=match.match_id,
    )
    update = MatchUpdate(
        match_id=match.match_id,
        season=match.season,
        kickoff_utc=match.kickoff_utc,
        competition=match.competition,
        round=match.round,
        home_team_id=match.home_team_id,
        away_team_id=match.away_team_id,
        tie_id=match.tie_id,
        stage=stage,
        is_knockout=match.is_knockout,
        is_tie_decider=match.is_tie_decider,
        advanced_team_id=match.advanced_team_id,
        home_goals=match.home_goals,
        away_goals=match.away_goals,
        is_neutral=match.is_neutral,
        decided_on_penalties=match.decided_on_penalties,
        home_power_pre=home.power_elo,
        away_power_pre=away.power_elo,
        home_reserve_pre=home.achievement_reserve,
        away_reserve_pre=away.achievement_reserve,
        home_progression_bonus_pre=home.progression_bonus_total,
        away_progression_bonus_pre=away.progression_bonus_total,
        home_live_pre=home.ao_live_elo,
        away_live_pre=away.ao_live_elo,
        effective_rating_difference=effective_rating_difference,
        expected_home_score=probability,
        home_win_probability=home_win_probability,
        draw_probability=draw_probability,
        away_win_probability=away_win_probability,
        actual_home_score=actual,
        goal_difference_enabled=config.goal_difference_enabled,
        goal_alpha=config.goal_alpha,
        goal_tau=config.goal_tau,
        goal_difference_cap=config.goal_difference_cap,
        goal_difference=difference,
        goal_multiplier=multiplier,
        xg_performance_enabled=config.xg_performance_enabled,
        xg_analysis_eligible=match.xg_analysis_eligible,
        xg_applied=xg_applied,
        xg_fallback_used=(
            config.xg_performance_enabled and not match.xg_analysis_eligible
        ),
        xg_home=match.xg_home,
        xg_away=match.xg_away,
        xg_performance_ratio=config.xg_performance_ratio,
        xg_performance_scale=config.xg_performance_scale,
        minimum_winner_gain_ratio=config.minimum_winner_gain_ratio,
        xg_performance_signal=elo_update.xg_performance_signal,
        power_delta_before_xg=power_delta_before_xg,
        xg_power_adjustment=xg_power_adjustment,
        power_delta=delta,
        home_reserve_delta=(
            final_home.achievement_reserve - home.achievement_reserve
        ),
        away_reserve_delta=(
            final_away.achievement_reserve - away.achievement_reserve
        ),
        home_progression_bonus_delta=(
            final_home.progression_bonus_total - home.progression_bonus_total
        ),
        away_progression_bonus_delta=(
            final_away.progression_bonus_total - away.progression_bonus_total
        ),
        home_power_post=final_home.power_elo,
        away_power_post=final_away.power_elo,
        home_reserve_post=final_home.achievement_reserve,
        away_reserve_post=final_away.achievement_reserve,
        home_progression_bonus_post=final_home.progression_bonus_total,
        away_progression_bonus_post=final_away.progression_bonus_total,
        home_live_post=final_home.ao_live_elo,
        away_live_post=final_away.ao_live_elo,
        winner_probability=winner_probability,
        progression_bonus_recipient_id=progression_bonus_recipient_id,
        progression_bonus_added=progression_bonus_added,
        progression_bonus_competition_pre=progression_bonus_competition_pre,
        progression_bonus_competition_post=progression_bonus_competition_post,
        progression_bonus_competition_cap=progression_bonus_competition_cap,
        progression_reserve_added=progression_added,
        trophy_reserve_added=trophy_added,
        model_version=config.model_version,
        config_id=config.config_id,
    )
    return updated_state, update


def run_season(
    initial_state: SeasonState,
    matches: Iterable[MatchInput],
    config: DynamicEloConfig,
) -> tuple[SeasonState, tuple[MatchUpdate, ...]]:
    """Run an ordered season batch through the same pure single-match kernel."""
    _validate_state_config(initial_state, config)
    state = initial_state
    updates = []
    for match in matches:
        state, update = update_match(state, match, config)
        updates.append(update)
    return state, tuple(updates)


def actual_home_score(home_goals: int, away_goals: int) -> float:
    _require_non_negative_integer("home_goals", home_goals)
    _require_non_negative_integer("away_goals", away_goals)
    if home_goals > away_goals:
        return 1.0
    if home_goals < away_goals:
        return 0.0
    return 0.5


def goal_multiplier(
    goal_difference: int,
    effective_rating_difference: float,
    config: DynamicEloConfig,
    *,
    decided_on_penalties: bool = False,
    is_draw: bool = False,
) -> float:
    """Apply the production controlled and favorite-damped GD formula."""
    config.validate()
    alpha = config.goal_alpha if config.goal_difference_enabled else 0.0
    return calculate_goal_difference_multiplier(
        goal_difference,
        effective_rating_difference,
        alpha,
        config.goal_tau,
        decided_on_penalties=decided_on_penalties,
        is_draw=is_draw,
        goal_cap=config.goal_difference_cap,
    )


def normalize_stage(round_name: str, competition: str, is_knockout: bool) -> str:
    if competition not in COMPETITIONS:
        raise ValueError(f"Unknown competition: {competition}")
    if not is_knockout:
        if round_name in {"Group Stage", "League Stage", "League Phase"}:
            return "LEAGUE"
        raise ValueError(f"Unknown non-knockout round: {competition}/{round_name}")
    if round_name in {
        "Preliminary Round",
        "1st Qualifying Round",
        "2nd Qualifying Round",
        "3rd Qualifying Round",
        "Qualifying Play-off Round",
    }:
        return "QUALIFYING"
    if round_name == "Knockout round play-offs":
        return "KNOCKOUT_PLAYOFF"
    if round_name == "Round 2":
        return "ROUND_OF_16" if competition == "UCL" else "KNOCKOUT_PLAYOFF"
    if round_name in {"Round of 16", "Round 3"}:
        return "ROUND_OF_16"
    try:
        return {
            "Quarter Finals": "QUARTERFINAL",
            "Semi Finals": "SEMIFINAL",
            "Final": "FINAL",
        }[round_name]
    except KeyError as error:
        raise ValueError(f"Unknown knockout round: {competition}/{round_name}") from error


def validate_state(state: SeasonState, config: DynamicEloConfig) -> None:
    """Validate a persisted or in-memory state against the active config."""
    _validate_state_config(state, config)


def _validate_state_config(state: SeasonState, config: DynamicEloConfig) -> None:
    config.validate()
    _require_identifier("state.season", state.season)
    if state.model_version != config.model_version:
        raise ValueError(
            f"State model_version {state.model_version} does not match "
            f"config {config.model_version}"
        )
    if state.config_id != config.config_id:
        raise ValueError(
            f"State config_id {state.config_id} does not match config {config.config_id}"
        )
    if not state.ratings:
        raise ValueError("State must contain at least one team rating")
    for team_id, rating in state.ratings.items():
        _require_identifier("state team_id", team_id)
        if rating.team_id != team_id:
            raise ValueError(f"State rating key mismatch for team_id: {team_id}")
        _require_identifier("state team_name", rating.team_name)
        _require_finite("state ao_first_elo", rating.ao_first_elo)
        _require_finite("state power_elo", rating.power_elo)
        _require_non_negative_finite(
            "state achievement_reserve",
            rating.achievement_reserve,
        )
        if rating.achievement_reserve > config.reserve_cap + 1e-9:
            raise ValueError(f"State reserve exceeds cap for team_id: {team_id}")
        for competition, value in (
            ("UCL", rating.progression_bonus_ucl),
            ("UEL", rating.progression_bonus_uel),
            ("UECL", rating.progression_bonus_uecl),
        ):
            _require_non_negative_finite(
                f"state {competition} progression bonus",
                value,
            )
            cap = config.fixed_progression_config.cap(competition)
            if value > cap + 1e-9:
                raise ValueError(
                    f"State {competition} progression bonus exceeds cap for "
                    f"team_id: {team_id}"
                )
            if not config.progression_bonus_enabled and value != 0.0:
                raise ValueError(
                    f"Disabled progression bonus must be zero for team_id: {team_id}"
                )
        if (rating.last_event_utc is None) != (rating.last_match_id is None):
            raise ValueError(
                f"State last event metadata is incomplete for team_id: {team_id}"
            )
        if rating.last_event_utc is not None:
            _require_utc_datetime("state last_event_utc", rating.last_event_utc)
            assert rating.last_match_id is not None
            _require_identifier("state last_match_id", rating.last_match_id)
    for match_id in state.processed_match_ids:
        _require_identifier("processed match_id", match_id)
    for tie_id in state.processed_tie_ids:
        _require_identifier("processed tie_id", tie_id)
        if tie_id in state.open_ties:
            raise ValueError(f"Processed tie_id cannot remain open: {tie_id}")
    for tie_id, tie in state.open_ties.items():
        if tie.tie_id != tie_id:
            raise ValueError(f"State open tie key mismatch for tie_id: {tie_id}")
        if tie.team_a_id not in state.ratings or tie.team_b_id not in state.ratings:
            raise ValueError(f"State open tie references an unknown team: {tie_id}")
        if tie.team_a_id == tie.team_b_id:
            raise ValueError(f"State open tie must contain two teams: {tie_id}")
        if not 0.0 <= tie.expected_a_to_advance <= 1.0:
            raise ValueError(f"State open tie probability is invalid: {tie_id}")
        config.competition_multiplier(tie.competition)
        config.stage_multiplier(tie.stage)
    if (state.last_event_utc is None) != (state.last_match_id is None):
        raise ValueError("State global last event metadata is incomplete")
    if state.last_event_utc is not None:
        _require_utc_datetime("state last_event_utc", state.last_event_utc)
        assert state.last_match_id is not None
        _require_identifier("state last_match_id", state.last_match_id)


def _validate_existing_tie(
    tie: TieState,
    home_team_id: str,
    away_team_id: str,
    competition: str,
    stage: str,
) -> None:
    if {home_team_id, away_team_id} != {tie.team_a_id, tie.team_b_id}:
        raise ValueError(
            f"tie_id {tie.tie_id} is already assigned to different teams"
        )
    if competition != tie.competition or stage != tie.stage:
        raise ValueError(
            f"tie_id {tie.tie_id} competition/stage does not match its first leg"
        )


def _validate_chronology(state: SeasonState, match: MatchInput) -> None:
    _validate_chronology_key(state, match.kickoff_utc, match.match_id)


def _validate_chronology_key(
    state: SeasonState,
    kickoff_utc: datetime,
    match_id: str,
) -> None:
    if state.last_event_utc is None:
        return
    assert state.last_match_id is not None
    current_key = (kickoff_utc, match_id)
    previous_key = (state.last_event_utc, state.last_match_id)
    if current_key <= previous_key:
        raise ValueError(
            f"Chronology regression: {match_id} at {kickoff_utc.isoformat()} "
            f"does not follow {state.last_match_id} at "
            f"{state.last_event_utc.isoformat()}"
        )


def _season_start_year(season: str) -> int:
    _require_identifier("season", season)
    parts = season.strip().split("/")
    if len(parts) != 2 or len(parts[0]) != 4 or not all(part.isdigit() for part in parts):
        raise ValueError("season must use YYYY/YY or YYYY/YYYY format")
    start = int(parts[0])
    end = int(parts[1])
    expected = start + 1
    if len(parts[1]) == 2:
        if end != expected % 100:
            raise ValueError("season end year must follow its start year")
    elif len(parts[1]) == 4:
        if end != expected:
            raise ValueError("season end year must follow its start year")
    else:
        raise ValueError("season must use YYYY/YY or YYYY/YYYY format")
    return start


def _validate_1x2_probabilities(
    home: float,
    draw: float,
    away: float,
    expected_home_score: float,
    label: str,
) -> None:
    for name, value in (("home", home), ("draw", draw), ("away", away)):
        _require_between_zero_and_one(f"{label} {name} probability", value)
    if not math.isclose(home + draw + away, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label} 1X2 probabilities must sum to 1")
    if not math.isclose(
        home + 0.5 * draw,
        expected_home_score,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{label} 1X2 probabilities do not preserve expected score")


def _require_utc_datetime(name: str, value: datetime) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")
    if value.utcoffset() != timezone.utc.utcoffset(value):
        raise ValueError(f"{name} must be normalized to UTC")


def _require_identifier(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_non_negative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


def _require_positive_finite(name: str, value: float) -> None:
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{name} must be positive and finite")


def _require_non_negative_finite(name: str, value: float) -> None:
    if not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"{name} must be non-negative and finite")


def _require_between_zero_and_one(name: str, value: float) -> None:
    if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be finite and in [0,1]")
