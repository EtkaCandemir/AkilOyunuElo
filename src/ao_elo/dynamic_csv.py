from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from ao_elo.config import AO_MODEL_V2_VERSION
from ao_elo.dynamic import (
    DynamicEloConfig,
    EXPECTED_SCORE_SEMANTICS,
    ONE_X_TWO_PROBABILITY_SEMANTICS,
    LockedPrediction,
    MatchFixture,
    MatchInput,
    MatchUpdate,
    SeasonState,
    TeamRating,
    TeamSeed,
    TieState,
    initialize_season,
    lock_prediction,
    run_season,
    settle_locked_match,
    validate_state,
)


STATE_CHECKPOINT_SCHEMA_VERSION = "2.0.0"
PREDICTION_LEDGER_SCHEMA_VERSION = "1.1.0"
STATE_CHECKPOINT_FILENAME = "state_checkpoint.json"

FIXTURE_INPUT_COLUMNS = [
    "match_id",
    "season",
    "kickoff_utc",
    "competition",
    "round",
    "tie_id",
    "is_knockout",
    "is_tie_decider",
    "stage",
    "home_team_id",
    "away_team_id",
    "is_neutral",
]

MATCH_INPUT_COLUMNS = [
    "match_id",
    "season",
    "kickoff_utc",
    "competition",
    "round",
    "tie_id",
    "is_knockout",
    "is_tie_decider",
    "stage",
    "home_team_id",
    "away_team_id",
    "home_goals",
    "away_goals",
    "xg_home",
    "xg_away",
    "xg_analysis_eligible",
    "is_neutral",
    "decided_on_penalties",
    "advanced_team_id",
]

RATINGS_STATE_COLUMNS = [
    "season",
    "team_id",
    "team_name",
    "ao_first_elo",
    "power_elo",
    "achievement_reserve",
    "progression_bonus_ucl",
    "progression_bonus_uel",
    "progression_bonus_uecl",
    "progression_bonus_total",
    "ao_live_elo",
    "last_event_utc",
    "last_match_id",
    "model_version",
    "config_id",
]

MATCH_UPDATE_COLUMNS = [
    "match_id",
    "season",
    "kickoff_utc",
    "competition",
    "round",
    "home_team_id",
    "away_team_id",
    "tie_id",
    "stage",
    "is_knockout",
    "is_tie_decider",
    "advanced_team_id",
    "home_goals",
    "away_goals",
    "is_neutral",
    "decided_on_penalties",
    "home_power_pre",
    "away_power_pre",
    "home_reserve_pre",
    "away_reserve_pre",
    "home_progression_bonus_pre",
    "away_progression_bonus_pre",
    "home_live_pre",
    "away_live_pre",
    "effective_rating_difference",
    "expected_home_score",
    "expected_score_semantics",
    "home_win_probability",
    "draw_probability",
    "away_win_probability",
    "probability_semantics",
    "actual_home_score",
    "goal_difference_enabled",
    "goal_alpha",
    "goal_tau",
    "goal_difference_cap",
    "goal_difference",
    "goal_multiplier",
    "xg_performance_enabled",
    "xg_analysis_eligible",
    "xg_applied",
    "xg_fallback_used",
    "xg_home",
    "xg_away",
    "xg_performance_ratio",
    "xg_performance_scale",
    "minimum_winner_gain_ratio",
    "xg_performance_signal",
    "power_delta_before_xg",
    "xg_power_adjustment",
    "power_delta",
    "home_reserve_delta",
    "away_reserve_delta",
    "home_progression_bonus_delta",
    "away_progression_bonus_delta",
    "home_power_post",
    "away_power_post",
    "home_reserve_post",
    "away_reserve_post",
    "home_progression_bonus_post",
    "away_progression_bonus_post",
    "home_live_post",
    "away_live_post",
    "winner_probability",
    "progression_bonus_recipient_id",
    "progression_bonus_added",
    "progression_bonus_competition_pre",
    "progression_bonus_competition_post",
    "progression_bonus_competition_cap",
    "progression_reserve_added",
    "trophy_reserve_added",
    "model_version",
    "config_id",
]

REPLAY_PREDICTION_COLUMNS = [
    "match_id",
    "season",
    "kickoff_utc",
    "competition",
    "round",
    "home_team_id",
    "away_team_id",
    "home_power_pre",
    "away_power_pre",
    "home_reserve_pre",
    "away_reserve_pre",
    "home_live_pre",
    "away_live_pre",
    "expected_home_score",
    "expected_score_semantics",
    "home_win_probability",
    "draw_probability",
    "away_win_probability",
    "probability_semantics",
    "record_type",
    "model_version",
    "config_id",
]

PRE_MATCH_LOG_COLUMNS = [
    "ledger_schema_version",
    "match_id",
    "season",
    "kickoff_utc",
    "generated_at_utc",
    "competition",
    "round",
    "tie_id",
    "stage",
    "is_knockout",
    "is_tie_decider",
    "home_team_id",
    "away_team_id",
    "is_neutral",
    "home_power_pre",
    "away_power_pre",
    "home_reserve_pre",
    "away_reserve_pre",
    "home_live_pre",
    "away_live_pre",
    "expected_home_score",
    "expected_score_semantics",
    "home_win_probability",
    "draw_probability",
    "away_win_probability",
    "probability_semantics",
    "state_last_event_utc",
    "state_last_match_id",
    "model_version",
    "config_id",
    "previous_record_hash",
    "record_hash",
]


def load_selected_v2_config(manifest_path: str | Path) -> DynamicEloConfig:
    """Load final rating and 1X2 parameters from the production manifest."""
    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("model_version") != AO_MODEL_V2_VERSION:
        raise ValueError("Model manifest version does not match the frozen v2 contract")
    required = {
        "dynamic_core",
        "active_power_carry",
        "one_x_two_probability",
        "goal_margin",
        "xg_performance",
        "progression_bonus",
        "competition_k",
        "achievement_reserve",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Model manifest missing keys: {missing}")
    core = payload["dynamic_core"]
    goal = payload["goal_margin"]
    xg = payload["xg_performance"]
    progression = payload["progression_bonus"]
    competition_k = payload["competition_k"]
    reserve = payload["achievement_reserve"]
    probability = payload["one_x_two_probability"]
    probability_active = _manifest_boolean(
        probability.get("active"),
        "one_x_two_probability.active",
    )
    goal_active = _manifest_boolean(goal.get("active"), "goal_margin.active")
    xg_active = _manifest_boolean(
        xg.get("active"),
        "xg_performance.active",
    )
    progression_active = _manifest_boolean(
        progression.get("active"),
        "progression_bonus.active",
    )
    competition_k_active = _manifest_boolean(
        competition_k.get("active"),
        "competition_k.active",
    )
    reserve_active = _manifest_boolean(
        reserve.get("active"),
        "achievement_reserve.active",
    )
    trophy_uses_same_base = _manifest_boolean(
        reserve.get("trophy_uses_same_base"),
        "achievement_reserve.trophy_uses_same_base",
    )
    goal_alpha = float(goal.get("alpha", 0.0))
    if not goal_active and goal_alpha != 0.0:
        raise ValueError("Inactive goal-difference config must have alpha=0")
    if goal_active and str(goal.get("family")) != "CONTROLLED_FAVORITE_DAMPED_LOG":
        raise ValueError(
            "Active goal-difference config must use CONTROLLED_FAVORITE_DAMPED_LOG"
        )
    xg_ratio = float(xg.get("max_xg_ratio", 0.0))
    xg_scale = float(xg.get("xg_scale", 1.25))
    minimum_winner_gain_ratio = float(
        xg.get("minimum_winner_gain_ratio", 1.0)
    )
    if xg_active:
        if str(xg.get("family")) != "BOUNDED_TWO_SIDED_PERFORMANCE_ADJUSTMENT":
            raise ValueError(
                "Active xG config must use BOUNDED_TWO_SIDED_PERFORMANCE_ADJUSTMENT"
            )
        if xg.get("missing_xg_behavior") != "FALL_BACK_TO_GOAL_MARGIN_ONLY":
            raise ValueError("Production xG must fall back to goal-margin-only Elo")
        if xg.get("draw_behavior") != "NO_XG_ADJUSTMENT":
            raise ValueError("Production draws cannot receive xG adjustment")
        if xg.get("penalty_shootout_behavior") != "NO_XG_ADJUSTMENT":
            raise ValueError("Production shoot-outs cannot receive xG adjustment")
        if not _manifest_boolean(xg.get("requires_both_teams"), "xg_performance.requires_both_teams"):
            raise ValueError("Production xG requires both teams' xG values")
        if not _manifest_boolean(xg.get("winner_direction_guard"), "xg_performance.winner_direction_guard"):
            raise ValueError("Production xG must preserve the match-result direction")
        if not _manifest_boolean(xg.get("zero_sum"), "xg_performance.zero_sum"):
            raise ValueError("Production xG must remain zero-sum")
        if not math.isclose(
            minimum_winner_gain_ratio,
            1.0 - xg_ratio,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "xG minimum_winner_gain_ratio must equal 1-max_xg_ratio"
            )
    elif xg_ratio != 0.0:
        raise ValueError("Inactive xG performance config must have max_xg_ratio=0")
    progression_base = float(progression.get("base_bonus", 0.0))
    progression_stages = progression.get("stages_per_competition", 5)
    if progression_active:
        if str(progression.get("family")) != "WINNER_ONLY_SEASON_LOCAL_FIXED":
            raise ValueError(
                "Active progression bonus must use WINNER_ONLY_SEASON_LOCAL_FIXED"
            )
        if progression_base <= 0.0:
            raise ValueError("Active progression bonus must have positive base_bonus")
        if not _manifest_boolean(
            progression.get("winner_only"),
            "progression_bonus.winner_only",
        ):
            raise ValueError("Production progression bonus must be winner-only")
        if _manifest_boolean(
            progression.get("loser_deduction"),
            "progression_bonus.loser_deduction",
        ):
            raise ValueError("Production progression bonus cannot deduct from the loser")
        if not _manifest_boolean(
            progression.get("season_reset"),
            "progression_bonus.season_reset",
        ):
            raise ValueError("Production progression bonus must reset each season")
        expected_ratios = {"UCL": 1.0, "UEL": 2.0 / 3.0, "UECL": 1.0 / 3.0}
        ratios = progression.get("competition_ratios")
        if not isinstance(ratios, dict) or set(ratios) != set(expected_ratios):
            raise ValueError("progression_bonus.competition_ratios are invalid")
        for competition, expected_ratio in expected_ratios.items():
            if not math.isclose(
                float(ratios[competition]),
                expected_ratio,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "Progression competition ratios must preserve UCL/UEL/UECL "
                    "as 1, 2/3 and 1/3"
                )
    elif progression_base != 0.0:
        raise ValueError("Inactive progression bonus must have base_bonus=0")
    if isinstance(progression_stages, bool) or not isinstance(progression_stages, int):
        raise ValueError("progression_bonus.stages_per_competition must be an integer")
    if competition_k_active:
        raise ValueError("Production competition K must remain disabled")
    for name in ("ucl_multiplier", "uel_multiplier", "uecl_multiplier"):
        if float(competition_k.get(name, 1.0)) != 1.0:
            raise ValueError(
                "Disabled production competition K multipliers must equal 1"
            )
    goal_difference_cap = goal.get("goal_difference_cap")
    if (
        isinstance(goal_difference_cap, bool)
        or not isinstance(goal_difference_cap, int)
    ):
        raise ValueError("goal_margin.goal_difference_cap must be an integer")
    if not reserve_active and float(reserve.get("reserve_base", 0.0)) != 0.0:
        raise ValueError("Inactive reserve config must have reserve_base=0")
    if not probability_active:
        raise ValueError("Production manifest must activate the calibrated 1X2 output")
    config = DynamicEloConfig(
        model_version=AO_MODEL_V2_VERSION,
        elo_scale=float(core["elo_scale"]),
        home_advantage=float(core["home_advantage"]),
        k_factor=float(core["k_factor"]),
        power_carry=float(payload["active_power_carry"]),
        draw_at_even=float(probability["draw_at_even"]),
        draw_shape=float(probability["draw_shape"]),
        goal_difference_enabled=goal_active,
        goal_alpha=goal_alpha,
        goal_tau=float(goal["tau"]),
        goal_difference_cap=goal_difference_cap,
        xg_performance_enabled=xg_active,
        xg_performance_ratio=xg_ratio,
        xg_performance_scale=xg_scale,
        minimum_winner_gain_ratio=minimum_winner_gain_ratio,
        progression_bonus_enabled=progression_active,
        progression_base_bonus=progression_base,
        progression_stages_per_competition=progression_stages,
        reserve_base=float(reserve["reserve_base"]),
        reserve_cap=float(reserve["reserve_cap"]),
        reserve_decay=float(reserve["reserve_decay"]),
        ucl_multiplier=float(reserve["ucl_multiplier"]),
        uel_multiplier=float(reserve["uel_multiplier"]),
        uecl_multiplier=float(reserve["uecl_multiplier"]),
        stage_profile=str(reserve["stage_profile"]),
        trophy_uses_same_base=trophy_uses_same_base,
    )
    config.validate()
    return config


def read_team_seeds(path: str | Path) -> tuple[str, tuple[TeamSeed, ...]]:
    data = pd.read_csv(path)
    required = {"season", "team_id", "team_name", "ao_first_elo"}
    _require_columns(data, required, "initial ratings")
    if data.empty:
        raise ValueError("Initial ratings CSV cannot be empty")
    seasons = data["season"].dropna().astype(str).str.strip().unique()
    if len(seasons) != 1:
        raise ValueError("Initial ratings CSV must contain exactly one season")
    seeds = []
    for row in data.itertuples(index=False):
        rating = _finite_number(row.ao_first_elo, "ao_first_elo")
        seeds.append(
            TeamSeed(
                team_id=_required_id(row.team_id, "team_id"),
                team_name=_required_text(row.team_name, "team_name"),
                ao_first_elo=rating,
            )
        )
    return str(seasons[0]), tuple(seeds)


def read_fixtures(path: str | Path) -> tuple[MatchFixture, ...]:
    """Read result-free fixtures used to lock genuine pre-match predictions."""
    data = pd.read_csv(path)
    _require_columns(data, set(FIXTURE_INPUT_COLUMNS), "fixtures.csv")
    if data.empty:
        raise ValueError("fixtures.csv cannot be empty")
    if data["match_id"].duplicated().any():
        raise ValueError("fixtures.csv contains duplicate match_id values")
    fixtures = []
    for row in data.to_dict(orient="records"):
        fixture = _fixture_from_row(row)
        fixture.validate()
        fixtures.append(fixture)
    return tuple(fixtures)


def read_matches(path: str | Path) -> tuple[MatchInput, ...]:
    data = pd.read_csv(path)
    required = {
        "match_id",
        "season",
        "kickoff_utc",
        "competition",
        "round",
        "home_team_id",
        "away_team_id",
        "home_goals",
        "away_goals",
        "is_neutral",
        "decided_on_penalties",
    }
    _require_columns(data, required, "matches.csv")
    if data["match_id"].duplicated().any():
        raise ValueError("matches.csv contains duplicate match_id values")
    matches = []
    for row in data.to_dict(orient="records"):
        fixture = _fixture_from_row(row)
        match = MatchInput(
            match_id=fixture.match_id,
            season=fixture.season,
            kickoff_utc=fixture.kickoff_utc,
            competition=fixture.competition,
            round=fixture.round,
            tie_id=fixture.tie_id,
            is_knockout=fixture.is_knockout,
            is_tie_decider=fixture.is_tie_decider,
            stage=fixture.stage,
            home_team_id=fixture.home_team_id,
            away_team_id=fixture.away_team_id,
            home_goals=_non_negative_integer(row["home_goals"], "home_goals"),
            away_goals=_non_negative_integer(row["away_goals"], "away_goals"),
            xg_home=_optional_non_negative_number(row.get("xg_home"), "xg_home"),
            xg_away=_optional_non_negative_number(row.get("xg_away"), "xg_away"),
            xg_analysis_eligible=_optional_boolean(
                row.get("xg_analysis_eligible"),
                "xg_analysis_eligible",
                default=False,
            ),
            is_neutral=fixture.is_neutral,
            decided_on_penalties=_boolean(
                row["decided_on_penalties"], "decided_on_penalties"
            ),
            advanced_team_id=_optional_id(row.get("advanced_team_id")),
        )
        match.validate()
        matches.append(match)
    return tuple(matches)


def _fixture_from_row(row: dict[str, object]) -> MatchFixture:
    kickoff = pd.to_datetime(row["kickoff_utc"], utc=True, errors="raise")
    return MatchFixture(
        match_id=_required_id(row["match_id"], "match_id"),
        season=_required_text(row["season"], "season"),
        kickoff_utc=kickoff.to_pydatetime().astimezone(timezone.utc),
        competition=_required_text(row["competition"], "competition"),
        round=_required_text(row["round"], "round"),
        tie_id=_optional_id(row.get("tie_id")),
        is_knockout=_boolean(row.get("is_knockout", False), "is_knockout"),
        is_tie_decider=_boolean(
            row.get("is_tie_decider", False),
            "is_tie_decider",
        ),
        stage=_optional_text(row.get("stage")),
        home_team_id=_required_id(row["home_team_id"], "home_team_id"),
        away_team_id=_required_id(row["away_team_id"], "away_team_id"),
        is_neutral=_boolean(row["is_neutral"], "is_neutral"),
    )


def state_to_frame(state: SeasonState) -> pd.DataFrame:
    rows = []
    for team_id in sorted(state.ratings):
        rating = state.ratings[team_id]
        rows.append(
            {
                "season": state.season,
                "team_id": rating.team_id,
                "team_name": rating.team_name,
                "ao_first_elo": rating.ao_first_elo,
                "power_elo": rating.power_elo,
                "achievement_reserve": rating.achievement_reserve,
                "progression_bonus_ucl": rating.progression_bonus_ucl,
                "progression_bonus_uel": rating.progression_bonus_uel,
                "progression_bonus_uecl": rating.progression_bonus_uecl,
                "progression_bonus_total": rating.progression_bonus_total,
                "ao_live_elo": rating.ao_live_elo,
                "last_event_utc": _format_datetime(rating.last_event_utc),
                "last_match_id": rating.last_match_id or "",
                "model_version": state.model_version,
                "config_id": state.config_id,
            }
        )
    return pd.DataFrame(rows, columns=RATINGS_STATE_COLUMNS)


def state_from_frame(data: pd.DataFrame, config: DynamicEloConfig) -> SeasonState:
    """Load team rows only; use load_state_checkpoint for same-season resume."""
    _require_columns(data, set(RATINGS_STATE_COLUMNS), "ratings_state.csv")
    if data.empty:
        raise ValueError("ratings_state.csv cannot be empty")
    season_values = data["season"].astype(str).unique()
    version_values = data["model_version"].astype(str).unique()
    config_values = data["config_id"].astype(str).unique()
    if len(season_values) != 1 or len(version_values) != 1 or len(config_values) != 1:
        raise ValueError("ratings_state.csv metadata must be constant across rows")
    if version_values[0] != config.model_version or config_values[0] != config.config_id:
        raise ValueError("ratings_state.csv config does not match active model config")
    ratings: dict[str, TeamRating] = {}
    for row in data.to_dict(orient="records"):
        team_id = _required_id(row["team_id"], "team_id")
        if team_id in ratings:
            raise ValueError(f"ratings_state.csv duplicate team_id: {team_id}")
        event_time = _optional_datetime(row["last_event_utc"])
        rating = TeamRating(
            team_id=team_id,
            team_name=_required_text(row["team_name"], "team_name"),
            ao_first_elo=_finite_number(row["ao_first_elo"], "ao_first_elo"),
            power_elo=_finite_number(row["power_elo"], "power_elo"),
            achievement_reserve=_finite_number(
                row["achievement_reserve"], "achievement_reserve"
            ),
            progression_bonus_ucl=_finite_number(
                row["progression_bonus_ucl"], "progression_bonus_ucl"
            ),
            progression_bonus_uel=_finite_number(
                row["progression_bonus_uel"], "progression_bonus_uel"
            ),
            progression_bonus_uecl=_finite_number(
                row["progression_bonus_uecl"], "progression_bonus_uecl"
            ),
            last_event_utc=event_time,
            last_match_id=_optional_id(row["last_match_id"]),
        )
        if abs(
            _finite_number(row["ao_live_elo"], "ao_live_elo") - rating.ao_live_elo
        ) > 1e-9:
            raise ValueError("ratings_state.csv ao_live_elo is inconsistent")
        if abs(
            _finite_number(
                row["progression_bonus_total"],
                "progression_bonus_total",
            )
            - rating.progression_bonus_total
        ) > 1e-9:
            raise ValueError(
                "ratings_state.csv progression_bonus_total is inconsistent"
            )
        ratings[team_id] = rating
    ordered_events = [
        (rating.last_event_utc, rating.last_match_id)
        for rating in ratings.values()
        if rating.last_event_utc is not None and rating.last_match_id is not None
    ]
    last_event, last_match = max(ordered_events) if ordered_events else (None, None)
    return SeasonState(
        season=str(season_values[0]),
        ratings=ratings,
        processed_match_ids=frozenset(),
        processed_tie_ids=frozenset(),
        open_ties={},
        last_event_utc=last_event,
        last_match_id=last_match,
        model_version=config.model_version,
        config_id=config.config_id,
    )


def save_state_checkpoint(
    state: SeasonState,
    directory: str | Path,
    config: DynamicEloConfig,
) -> None:
    """Persist a resumable state bundle with ratings, open ties and processed IDs."""
    validate_state(state, config)
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    ratings_bytes = state_to_frame(state).to_csv(
        index=False,
        lineterminator="\n",
    ).encode("utf-8")
    metadata = {
        "schema_version": STATE_CHECKPOINT_SCHEMA_VERSION,
        "season": state.season,
        "model_version": state.model_version,
        "config_id": state.config_id,
        "ratings_state_sha256": hashlib.sha256(ratings_bytes).hexdigest(),
        "processed_match_ids": sorted(state.processed_match_ids),
        "processed_tie_ids": sorted(state.processed_tie_ids),
        "open_ties": [
            asdict(state.open_ties[tie_id]) for tie_id in sorted(state.open_ties)
        ],
        "last_event_utc": _format_datetime(state.last_event_utc),
        "last_match_id": state.last_match_id or "",
    }
    _atomic_write_bytes(output / "ratings_state.csv", ratings_bytes)
    _atomic_write_bytes(
        output / STATE_CHECKPOINT_FILENAME,
        (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def load_state_checkpoint(
    directory: str | Path,
    config: DynamicEloConfig,
) -> SeasonState:
    """Load and verify a complete same-season state checkpoint."""
    root = Path(directory)
    ratings_path = root / "ratings_state.csv"
    metadata_path = root / STATE_CHECKPOINT_FILENAME
    if not ratings_path.exists() or not metadata_path.exists():
        raise ValueError(
            "State checkpoint requires ratings_state.csv and state_checkpoint.json"
        )
    ratings_bytes = ratings_path.read_bytes()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != STATE_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Unsupported state checkpoint schema_version")
    checksum = hashlib.sha256(ratings_bytes).hexdigest()
    if metadata.get("ratings_state_sha256") != checksum:
        raise ValueError("ratings_state.csv checksum does not match state checkpoint")
    base = state_from_frame(pd.read_csv(ratings_path), config)
    if metadata.get("season") != base.season:
        raise ValueError("State checkpoint season does not match ratings_state.csv")
    if metadata.get("model_version") != config.model_version:
        raise ValueError("State checkpoint model_version does not match active config")
    if metadata.get("config_id") != config.config_id:
        raise ValueError("State checkpoint config_id does not match active config")

    processed_values = metadata.get("processed_match_ids")
    if not isinstance(processed_values, list):
        raise ValueError("State checkpoint processed_match_ids must be a list")
    processed = [_required_id(value, "processed match_id") for value in processed_values]
    if len(processed) != len(set(processed)):
        raise ValueError("State checkpoint contains duplicate processed match_id values")

    processed_tie_values = metadata.get("processed_tie_ids")
    if not isinstance(processed_tie_values, list):
        raise ValueError("State checkpoint processed_tie_ids must be a list")
    processed_ties = [
        _required_id(value, "processed tie_id") for value in processed_tie_values
    ]
    if len(processed_ties) != len(set(processed_ties)):
        raise ValueError("State checkpoint contains duplicate processed tie_id values")

    tie_values = metadata.get("open_ties")
    if not isinstance(tie_values, list):
        raise ValueError("State checkpoint open_ties must be a list")
    open_ties: dict[str, TieState] = {}
    for value in tie_values:
        if not isinstance(value, dict):
            raise ValueError("State checkpoint open_ties entries must be objects")
        tie = TieState(
            tie_id=_required_id(value.get("tie_id"), "tie_id"),
            team_a_id=_required_id(value.get("team_a_id"), "team_a_id"),
            team_b_id=_required_id(value.get("team_b_id"), "team_b_id"),
            expected_a_to_advance=_finite_number(
                value.get("expected_a_to_advance"),
                "expected_a_to_advance",
            ),
            competition=_required_text(value.get("competition"), "competition"),
            stage=_required_text(value.get("stage"), "stage"),
        )
        if tie.tie_id in open_ties:
            raise ValueError(f"State checkpoint duplicate tie_id: {tie.tie_id}")
        open_ties[tie.tie_id] = tie

    last_event = _optional_datetime(metadata.get("last_event_utc"))
    last_match = _optional_id(metadata.get("last_match_id"))
    state = SeasonState(
        season=base.season,
        ratings=base.ratings,
        processed_match_ids=frozenset(processed),
        processed_tie_ids=frozenset(processed_ties),
        open_ties=open_ties,
        last_event_utc=last_event,
        last_match_id=last_match,
        model_version=config.model_version,
        config_id=config.config_id,
    )
    validate_state(state, config)
    return state


def updates_to_frame(updates: Iterable[MatchUpdate]) -> pd.DataFrame:
    rows = []
    for update in updates:
        row = asdict(update)
        row["kickoff_utc"] = _format_datetime(update.kickoff_utc)
        row["expected_score_semantics"] = EXPECTED_SCORE_SEMANTICS
        row["probability_semantics"] = ONE_X_TWO_PROBABILITY_SEMANTICS
        rows.append(row)
    return pd.DataFrame(rows, columns=MATCH_UPDATE_COLUMNS)


def updates_to_replay_prediction_frame(updates: Iterable[MatchUpdate]) -> pd.DataFrame:
    """Return retrospective pre-update values, never a prospective holdout log."""
    update_frame = updates_to_frame(updates)
    columns = [
        column
        for column in REPLAY_PREDICTION_COLUMNS
        if column not in {"expected_score_semantics", "record_type"}
    ]
    result = update_frame.loc[:, columns].copy()
    result["expected_score_semantics"] = EXPECTED_SCORE_SEMANTICS
    result["record_type"] = "RETROSPECTIVE_REPLAY"
    return result.loc[:, REPLAY_PREDICTION_COLUMNS]


def updates_to_pre_match_frame(updates: Iterable[MatchUpdate]) -> pd.DataFrame:
    """Backward-compatible alias for retrospective replay output."""
    return updates_to_replay_prediction_frame(updates)


def append_prediction_lock(
    path: str | Path,
    prediction: LockedPrediction,
) -> str:
    """Append a tamper-evident prediction without rewriting prior ledger rows."""
    prediction.validate()
    ledger_path = Path(path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    records = _read_prediction_records(ledger_path)
    if any(row["match_id"] == prediction.match_id for row in records):
        raise ValueError(f"Prediction ledger already contains match_id: {prediction.match_id}")
    previous_hash = records[-1]["record_hash"] if records else ""
    row = _prediction_to_record(prediction)
    row["previous_record_hash"] = previous_hash
    row["record_hash"] = _prediction_record_hash(row)
    write_header = not ledger_path.exists() or ledger_path.stat().st_size == 0
    with ledger_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PRE_MATCH_LOG_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return row["record_hash"]


def read_locked_prediction(
    path: str | Path,
    match_id: str,
) -> LockedPrediction:
    """Read one lock after validating the entire ledger hash chain."""
    target = _required_id(match_id, "match_id")
    records = _read_prediction_records(Path(path))
    matches = [row for row in records if row["match_id"] == target]
    if len(matches) != 1:
        raise ValueError(f"Prediction ledger must contain exactly one row for {target}")
    return _prediction_from_record(matches[0])


def lock_fixture_to_ledger(
    state: SeasonState,
    fixture: MatchFixture,
    config: DynamicEloConfig,
    ledger_path: str | Path,
    *,
    generated_at_utc: datetime,
) -> LockedPrediction:
    prediction = lock_prediction(
        state,
        fixture,
        config,
        generated_at_utc=generated_at_utc,
    )
    append_prediction_lock(ledger_path, prediction)
    return prediction


def settle_match_from_ledger(
    state: SeasonState,
    match: MatchInput,
    config: DynamicEloConfig,
    ledger_path: str | Path,
) -> tuple[SeasonState, MatchUpdate]:
    prediction = read_locked_prediction(ledger_path, match.match_id)
    return settle_locked_match(state, match, prediction, config)


def append_match_update(path: str | Path, update: MatchUpdate) -> None:
    """Append one settlement audit row and reject duplicate match IDs."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size:
        existing = pd.read_csv(output, usecols=["match_id"])
        if existing["match_id"].astype(str).eq(update.match_id).any():
            raise ValueError(f"match_updates.csv already contains match_id: {update.match_id}")
    frame = updates_to_frame((update,))
    frame.to_csv(
        output,
        mode="a",
        header=not output.exists() or output.stat().st_size == 0,
        index=False,
        lineterminator="\n",
    )


def run_batch(
    initial_ratings_csv: str | Path,
    matches_csv: str | Path,
    output_dir: str | Path,
    config: DynamicEloConfig,
    *,
    previous_state_csv: str | Path | None = None,
) -> tuple[SeasonState, tuple[MatchUpdate, ...]]:
    """Replay completed results; this is not a prospective prediction logger."""
    config.validate()
    season, seeds = read_team_seeds(initial_ratings_csv)
    previous = None
    if previous_state_csv is not None:
        previous = state_from_frame(pd.read_csv(previous_state_csv), config)
    initial_state = initialize_season(
        season,
        seeds,
        config,
        previous_state=previous,
    )
    matches = read_matches(matches_csv)
    final_state, updates = run_season(initial_state, matches, config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    legacy_log = output / "pre_match_log.csv"
    if legacy_log.exists():
        raise ValueError(
            "Retrospective batch output directory contains pre_match_log.csv; "
            "archive it or use a clean directory so replay output cannot be "
            "mistaken for prospective holdout evidence"
        )
    save_state_checkpoint(final_state, output, config)
    updates_to_frame(updates).to_csv(output / "match_updates.csv", index=False)
    updates_to_replay_prediction_frame(updates).to_csv(
        output / "replay_predictions.csv",
        index=False,
    )
    _atomic_write_bytes(
        output / "batch_manifest.json",
        (
            json.dumps(
                {
                    "mode": "RETROSPECTIVE_REPLAY",
                    "prospective_holdout_evidence": False,
                    "model_version": config.model_version,
                    "config_id": config.config_id,
                    "matches_processed": len(updates),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8"),
    )
    return final_state, updates


def _prediction_to_record(prediction: LockedPrediction) -> dict[str, str]:
    return {
        "ledger_schema_version": PREDICTION_LEDGER_SCHEMA_VERSION,
        "match_id": prediction.match_id,
        "season": prediction.season,
        "kickoff_utc": _format_datetime(prediction.kickoff_utc),
        "generated_at_utc": _format_datetime(prediction.generated_at_utc),
        "competition": prediction.competition,
        "round": prediction.round,
        "tie_id": prediction.tie_id or "",
        "stage": prediction.stage,
        "is_knockout": _format_boolean(prediction.is_knockout),
        "is_tie_decider": _format_boolean(prediction.is_tie_decider),
        "home_team_id": prediction.home_team_id,
        "away_team_id": prediction.away_team_id,
        "is_neutral": _format_boolean(prediction.is_neutral),
        "home_power_pre": _format_number(prediction.home_power_pre),
        "away_power_pre": _format_number(prediction.away_power_pre),
        "home_reserve_pre": _format_number(prediction.home_reserve_pre),
        "away_reserve_pre": _format_number(prediction.away_reserve_pre),
        "home_live_pre": _format_number(prediction.home_live_pre),
        "away_live_pre": _format_number(prediction.away_live_pre),
        "expected_home_score": _format_number(prediction.expected_home_score),
        "expected_score_semantics": prediction.expected_score_semantics,
        "home_win_probability": _format_number(
            prediction.home_win_probability
        ),
        "draw_probability": _format_number(prediction.draw_probability),
        "away_win_probability": _format_number(
            prediction.away_win_probability
        ),
        "probability_semantics": prediction.probability_semantics,
        "state_last_event_utc": _format_datetime(prediction.state_last_event_utc),
        "state_last_match_id": prediction.state_last_match_id or "",
        "model_version": prediction.model_version,
        "config_id": prediction.config_id,
        "previous_record_hash": "",
        "record_hash": "",
    }


def _prediction_from_record(row: dict[str, str]) -> LockedPrediction:
    if row["ledger_schema_version"] != PREDICTION_LEDGER_SCHEMA_VERSION:
        raise ValueError("Unsupported prediction ledger schema version")
    prediction = LockedPrediction(
        match_id=_required_id(row["match_id"], "match_id"),
        season=_required_text(row["season"], "season"),
        kickoff_utc=_required_datetime(row["kickoff_utc"], "kickoff_utc"),
        generated_at_utc=_required_datetime(
            row["generated_at_utc"],
            "generated_at_utc",
        ),
        competition=_required_text(row["competition"], "competition"),
        round=_required_text(row["round"], "round"),
        tie_id=_optional_id(row["tie_id"]),
        stage=_required_text(row["stage"], "stage"),
        is_knockout=_boolean(row["is_knockout"], "is_knockout"),
        is_tie_decider=_boolean(row["is_tie_decider"], "is_tie_decider"),
        home_team_id=_required_id(row["home_team_id"], "home_team_id"),
        away_team_id=_required_id(row["away_team_id"], "away_team_id"),
        is_neutral=_boolean(row["is_neutral"], "is_neutral"),
        home_power_pre=_finite_number(row["home_power_pre"], "home_power_pre"),
        away_power_pre=_finite_number(row["away_power_pre"], "away_power_pre"),
        home_reserve_pre=_finite_number(
            row["home_reserve_pre"],
            "home_reserve_pre",
        ),
        away_reserve_pre=_finite_number(
            row["away_reserve_pre"],
            "away_reserve_pre",
        ),
        home_live_pre=_finite_number(row["home_live_pre"], "home_live_pre"),
        away_live_pre=_finite_number(row["away_live_pre"], "away_live_pre"),
        expected_home_score=_finite_number(
            row["expected_home_score"],
            "expected_home_score",
        ),
        expected_score_semantics=_required_text(
            row["expected_score_semantics"],
            "expected_score_semantics",
        ),
        home_win_probability=_finite_number(
            row["home_win_probability"],
            "home_win_probability",
        ),
        draw_probability=_finite_number(
            row["draw_probability"],
            "draw_probability",
        ),
        away_win_probability=_finite_number(
            row["away_win_probability"],
            "away_win_probability",
        ),
        probability_semantics=_required_text(
            row["probability_semantics"],
            "probability_semantics",
        ),
        state_last_event_utc=_optional_datetime(row["state_last_event_utc"]),
        state_last_match_id=_optional_id(row["state_last_match_id"]),
        model_version=_required_text(row["model_version"], "model_version"),
        config_id=_required_text(row["config_id"], "config_id"),
    )
    prediction.validate()
    return prediction


def _read_prediction_records(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != PRE_MATCH_LOG_COLUMNS:
            raise ValueError("Prediction ledger columns do not match the frozen schema")
        records = [dict(row) for row in reader]
    previous_hash = ""
    match_ids: set[str] = set()
    for index, row in enumerate(records, start=1):
        if row["previous_record_hash"] != previous_hash:
            raise ValueError(f"Prediction ledger hash chain breaks at row {index}")
        expected_hash = _prediction_record_hash(row)
        if row["record_hash"] != expected_hash:
            raise ValueError(f"Prediction ledger record hash is invalid at row {index}")
        if row["match_id"] in match_ids:
            raise ValueError(f"Prediction ledger duplicate match_id: {row['match_id']}")
        _prediction_from_record(row)
        match_ids.add(row["match_id"])
        previous_hash = row["record_hash"]
    return records


def _prediction_record_hash(row: dict[str, str]) -> str:
    payload = {
        column: row[column]
        for column in PRE_MATCH_LOG_COLUMNS
        if column != "record_hash"
    }
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _require_columns(data: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def _required_id(value: object, label: str) -> str:
    if pd.isna(value):
        raise ValueError(f"{label} must be present")
    if isinstance(value, float) and value.is_integer():
        result = str(int(value))
    else:
        result = str(value).strip()
    if not result:
        raise ValueError(f"{label} must be a non-empty identifier")
    return result


def _optional_id(value: object) -> str | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    return _required_id(value, "optional identifier")


def _required_text(value: object, label: str) -> str:
    if value is None or pd.isna(value) or not str(value).strip():
        raise ValueError(f"{label} must be non-empty")
    return str(value).strip()


def _optional_text(value: object) -> str | None:
    if value is None or pd.isna(value) or not str(value).strip():
        return None
    return str(value).strip()


def _boolean(value: object, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool) and value in (0, 1):
        return bool(value)
    if isinstance(value, float) and value in (0.0, 1.0):
        return bool(int(value))
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    raise ValueError(f"{label} must be true/false or 0/1")


def _optional_boolean(value: object, label: str, *, default: bool) -> bool:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return default
    return _boolean(value, label)


def _manifest_boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a JSON boolean")
    return value


def _non_negative_integer(value: object, label: str) -> int:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric) or not math.isfinite(float(numeric)):
        raise ValueError(f"{label} must be a non-negative integer")
    if float(numeric) < 0 or not float(numeric).is_integer():
        raise ValueError(f"{label} must be a non-negative integer")
    return int(numeric)


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite") from error
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite")
    return numeric


def _optional_non_negative_number(value: object, label: str) -> float | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    numeric = _finite_number(value, label)
    if numeric < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return numeric


def _format_number(value: float) -> str:
    if not math.isfinite(float(value)):
        raise ValueError("Ledger numeric values must be finite")
    return format(float(value), ".17g")


def _format_boolean(value: bool) -> str:
    if not isinstance(value, bool):
        raise ValueError("Ledger boolean values must be real booleans")
    return "true" if value else "false"


def _format_datetime(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _required_datetime(value: object, label: str) -> datetime:
    result = _optional_datetime(value)
    if result is None:
        raise ValueError(f"{label} must be a timezone-aware timestamp")
    return result


def _optional_datetime(value: object) -> datetime | None:
    if value is None or pd.isna(value) or not str(value).strip():
        return None
    return pd.to_datetime(value, utc=True, errors="raise").to_pydatetime()


def _atomic_write_bytes(path: Path, contents: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(contents)
    temporary.replace(path)
