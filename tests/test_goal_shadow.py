from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ao_elo.dynamic import (  # noqa: E402
    DynamicEloConfig,
    MatchFixture,
    MatchInput,
    TeamSeed,
)
from ao_elo.goal_shadow import (  # noqa: E402
    GoalShadowArm,
    initialize_goal_shadow,
    lock_goal_shadow,
    settle_goal_shadow,
    validate_goal_shadow_state,
)
from scripts.run_goal_difference_shadow import (  # noqa: E402
    append_predictions,
    append_updates,
    evaluate_updates,
    load_shadow_state,
    read_locked_predictions,
    save_shadow_state,
)


def config() -> DynamicEloConfig:
    return DynamicEloConfig.calibrated_v2()


def arms() -> tuple[GoalShadowArm, ...]:
    return (
        GoalShadowArm("BASE", 0.0, 300.0),
        GoalShadowArm("PRE_SPECIFIED", 0.10, 300.0),
        GoalShadowArm("EXTENDED", 0.125, 800.0),
    )


def seeds() -> tuple[TeamSeed, ...]:
    return (
        TeamSeed("A", "Alpha", 1500.0),
        TeamSeed("B", "Beta", 1500.0),
    )


def league_fixture(match_id: str, kickoff: datetime) -> MatchFixture:
    return MatchFixture(
        match_id=match_id,
        season="2026/27",
        kickoff_utc=kickoff,
        competition="UCL",
        round="League Stage",
        home_team_id="A",
        away_team_id="B",
        is_neutral=True,
        stage="LEAGUE",
    )


def league_match(
    match_id: str,
    kickoff: datetime,
    home_goals: int,
    away_goals: int,
) -> MatchInput:
    return MatchInput(
        match_id=match_id,
        season="2026/27",
        kickoff_utc=kickoff,
        competition="UCL",
        round="League Stage",
        home_team_id="A",
        away_team_id="B",
        home_goals=home_goals,
        away_goals=away_goals,
        is_neutral=True,
        decided_on_penalties=False,
        stage="LEAGUE",
    )


def test_shadow_settlement_keeps_base_and_candidates_independent() -> None:
    active = config()
    state = initialize_goal_shadow("2026/27", seeds(), arms(), active)
    kickoff = datetime(2026, 9, 8, 18, tzinfo=timezone.utc)
    predictions = lock_goal_shadow(
        state,
        league_fixture("m1", kickoff),
        active,
        generated_at_utc=kickoff - timedelta(minutes=5),
    )
    assert len({prediction.expected_home_score for prediction in predictions}) == 1

    final_state, updates = settle_goal_shadow(
        state,
        league_match("m1", kickoff, 3, 0),
        predictions,
        active,
    )
    by_arm = {update.arm_name: update for update in updates}
    assert by_arm["BASE"].goal_multiplier == pytest.approx(1.0)
    assert by_arm["PRE_SPECIFIED"].goal_multiplier > 1.0
    assert by_arm["EXTENDED"].goal_multiplier > 1.0
    assert by_arm["PRE_SPECIFIED"].power_delta > by_arm["BASE"].power_delta
    assert (
        final_state.ratings["PRE_SPECIFIED"]["A"]
        > final_state.ratings["BASE"]["A"]
    )
    assert all(update.zero_sum_error <= 1e-9 for update in updates)


def test_penalty_shadow_match_forces_draw_and_multiplier_one() -> None:
    active = config()
    state = initialize_goal_shadow("2026/27", seeds(), arms(), active)
    kickoff = datetime(2027, 3, 10, 20, tzinfo=timezone.utc)
    fixture = MatchFixture(
        match_id="pen1",
        season="2026/27",
        kickoff_utc=kickoff,
        competition="UCL",
        round="Round of 16",
        home_team_id="A",
        away_team_id="B",
        is_neutral=False,
        tie_id="tie1",
        is_knockout=True,
        is_tie_decider=True,
        stage="ROUND_OF_16",
    )
    predictions = lock_goal_shadow(
        state,
        fixture,
        active,
        generated_at_utc=kickoff - timedelta(minutes=5),
    )
    match = MatchInput(
        match_id="pen1",
        season="2026/27",
        kickoff_utc=kickoff,
        competition="UCL",
        round="Round of 16",
        home_team_id="A",
        away_team_id="B",
        home_goals=1,
        away_goals=1,
        is_neutral=False,
        decided_on_penalties=True,
        tie_id="tie1",
        is_knockout=True,
        is_tie_decider=True,
        advanced_team_id="B",
        stage="ROUND_OF_16",
    )
    _, updates = settle_goal_shadow(state, match, predictions, active)
    assert all(update.actual_home_score == pytest.approx(0.5) for update in updates)
    assert all(update.goal_difference == 0 for update in updates)
    assert all(update.goal_multiplier == pytest.approx(1.0) for update in updates)


def test_shadow_state_rejects_duplicate_and_chronology_regression() -> None:
    active = config()
    state = initialize_goal_shadow("2026/27", seeds(), arms(), active)
    kickoff = datetime(2026, 9, 8, 18, tzinfo=timezone.utc)
    predictions = lock_goal_shadow(
        state,
        league_fixture("m1", kickoff),
        active,
        generated_at_utc=kickoff - timedelta(minutes=5),
    )
    state, _ = settle_goal_shadow(
        state,
        league_match("m1", kickoff, 1, 0),
        predictions,
        active,
    )
    with pytest.raises(ValueError, match="Duplicate"):
        lock_goal_shadow(
            state,
            league_fixture("m1", kickoff),
            active,
            generated_at_utc=kickoff - timedelta(minutes=1),
        )
    with pytest.raises(ValueError, match="chronology"):
        lock_goal_shadow(
            state,
            league_fixture("m0", kickoff - timedelta(days=1)),
            active,
            generated_at_utc=kickoff - timedelta(days=1, minutes=5),
        )


def test_shadow_checkpoint_and_hash_ledger_round_trip(tmp_path: Path) -> None:
    active = config()
    state = initialize_goal_shadow("2026/27", seeds(), arms(), active)
    save_shadow_state(state, tmp_path, active)
    loaded = load_shadow_state(tmp_path, active)
    validate_goal_shadow_state(loaded, active)
    assert loaded.shadow_config_id == state.shadow_config_id

    kickoff = datetime(2026, 9, 8, 18, tzinfo=timezone.utc)
    predictions = lock_goal_shadow(
        loaded,
        league_fixture("m1", kickoff),
        active,
        generated_at_utc=kickoff - timedelta(minutes=5),
    )
    ledger = tmp_path / "ledger.csv"
    append_predictions(ledger, predictions)
    restored = read_locked_predictions(ledger, "m1")
    assert restored == predictions
    with pytest.raises(ValueError, match="Duplicate"):
        append_predictions(ledger, predictions)


def test_shadow_update_evaluation_compares_every_arm_to_base(
    tmp_path: Path,
) -> None:
    active = config()
    state = initialize_goal_shadow("2026/27", seeds(), arms(), active)
    kickoff = datetime(2026, 9, 8, 18, tzinfo=timezone.utc)
    predictions = lock_goal_shadow(
        state,
        league_fixture("m1", kickoff),
        active,
        generated_at_utc=kickoff - timedelta(minutes=5),
    )
    _, updates = settle_goal_shadow(
        state,
        league_match("m1", kickoff, 4, 0),
        predictions,
        active,
    )
    path = tmp_path / "updates.csv"
    append_updates(path, updates)
    data = pd.read_csv(path)
    summary = evaluate_updates(data)
    assert set(summary["arm_name"]) == {arm.name for arm in arms()}
    assert summary.loc[summary["arm_name"].eq("BASE"), "brier_delta_vs_base"].iloc[
        0
    ] == pytest.approx(0.0)
    assert summary["maximum_zero_sum_error"].max() <= 1e-9
    with pytest.raises(ValueError, match="Duplicate"):
        append_updates(path, updates)
