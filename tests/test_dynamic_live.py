from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from ao_elo.dynamic import (
    DynamicEloConfig,
    EXPECTED_SCORE_SEMANTICS,
    MatchFixture,
    MatchInput,
    TeamSeed,
    initialize_season,
    lock_prediction,
    settle_locked_match,
    update_match,
)
from ao_elo.dynamic_csv import (
    FIXTURE_INPUT_COLUMNS,
    MATCH_INPUT_COLUMNS,
    PRE_MATCH_LOG_COLUMNS,
    append_prediction_lock,
    load_state_checkpoint,
    read_locked_prediction,
    save_state_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT / "contracts" / "ao_european_elo_v2_production.json"
)
KICKOFF = datetime(2099, 9, 1, 19, 0, tzinfo=timezone.utc)
GENERATED = KICKOFF - timedelta(hours=2)


def config() -> DynamicEloConfig:
    return DynamicEloConfig.calibrated_v2()


def seeds() -> tuple[TeamSeed, ...]:
    return (
        TeamSeed("A", "Alpha", 1500.0),
        TeamSeed("B", "Beta", 1400.0),
        TeamSeed("C", "Gamma", 1300.0),
    )


def fixture(
    match_id: str = "m1",
    *,
    kickoff: datetime = KICKOFF,
    home: str = "A",
    away: str = "B",
    tie_id: str | None = None,
    knockout: bool = False,
    decider: bool = False,
    round_name: str = "League Stage",
    stage: str | None = "LEAGUE",
    single_match: bool = False,
) -> MatchFixture:
    return MatchFixture(
        match_id=match_id,
        season="2099/00",
        kickoff_utc=kickoff,
        competition="UCL",
        round=round_name,
        home_team_id=home,
        away_team_id=away,
        is_neutral=False,
        tie_id=tie_id,
        is_knockout=knockout,
        is_tie_decider=decider,
        stage=stage,
        is_single_match_tie=single_match,
    )


def result(
    match_id: str = "m1",
    *,
    kickoff: datetime = KICKOFF,
    home: str = "A",
    away: str = "B",
    home_goals: int = 2,
    away_goals: int = 1,
    tie_id: str | None = None,
    knockout: bool = False,
    decider: bool = False,
    advanced: str | None = None,
    round_name: str = "League Stage",
    stage: str | None = "LEAGUE",
    single_match: bool = False,
) -> MatchInput:
    return MatchInput(
        match_id=match_id,
        season="2099/00",
        kickoff_utc=kickoff,
        competition="UCL",
        round=round_name,
        home_team_id=home,
        away_team_id=away,
        home_goals=home_goals,
        away_goals=away_goals,
        is_neutral=False,
        decided_on_penalties=False,
        tie_id=tie_id,
        is_knockout=knockout,
        is_tie_decider=decider,
        advanced_team_id=advanced,
        stage=stage,
        is_single_match_tie=single_match,
    )


def test_locked_single_match_prediction_audits_format_draw_intercept() -> None:
    selected = config()
    state = initialize_season("2099/00", seeds(), selected)
    single = fixture(
        tie_id="final-1",
        knockout=True,
        decider=True,
        round_name="Final",
        stage="FINAL",
        single_match=True,
    )

    prediction = lock_prediction(
        state,
        single,
        selected,
        generated_at_utc=GENERATED,
    )

    assert prediction.is_single_match_tie is True
    assert prediction.effective_draw_at_even == pytest.approx(0.12)


def test_result_free_prediction_is_locked_before_kickoff_and_settled() -> None:
    selected = config()
    state = initialize_season("2099/00", seeds(), selected)
    prediction = lock_prediction(
        state,
        fixture(),
        selected,
        generated_at_utc=GENERATED,
    )

    assert prediction.expected_score_semantics == EXPECTED_SCORE_SEMANTICS
    assert prediction.generated_at_utc < prediction.kickoff_utc
    final_state, update = settle_locked_match(state, result(), prediction, selected)

    assert update.expected_home_score == prediction.expected_home_score
    assert "m1" in final_state.processed_match_ids


def test_prediction_cannot_be_backfilled_or_settled_against_changed_state() -> None:
    selected = config()
    state = initialize_season("2099/00", seeds(), selected)
    with pytest.raises(ValueError, match="before kickoff"):
        lock_prediction(
            state,
            fixture(),
            selected,
            generated_at_utc=KICKOFF,
        )

    prediction = lock_prediction(
        state,
        fixture(),
        selected,
        generated_at_utc=GENERATED,
    )
    changed = replace(
        state,
        ratings={
            **state.ratings,
            "A": replace(state.ratings["A"], power_elo=1510.0),
        },
    )
    with pytest.raises(ValueError, match="changed after"):
        settle_locked_match(changed, result(), prediction, selected)


def test_prediction_timestamp_cannot_predate_current_state() -> None:
    selected = config()
    state = initialize_season("2099/00", seeds(), selected)
    state, _ = update_match(
        state,
        result("prior", kickoff=KICKOFF - timedelta(days=1)),
        selected,
    )

    with pytest.raises(ValueError, match="cannot predate"):
        lock_prediction(
            state,
            fixture("next"),
            selected,
            generated_at_utc=KICKOFF - timedelta(days=2),
        )


def test_same_kickoff_disjoint_predictions_settle_in_match_id_order() -> None:
    selected = config()
    state = initialize_season(
        "2099/00",
        (*seeds(), TeamSeed("D", "Delta", 1200.0)),
        selected,
    )
    first_prediction = lock_prediction(
        state,
        fixture("m1", home="A", away="B"),
        selected,
        generated_at_utc=GENERATED,
    )
    second_prediction = lock_prediction(
        state,
        fixture("m2", home="C", away="D"),
        selected,
        generated_at_utc=GENERATED,
    )

    state, _ = settle_locked_match(
        state,
        result("m1", home="A", away="B"),
        first_prediction,
        selected,
    )
    final_state, _ = settle_locked_match(
        state,
        result("m2", home="C", away="D"),
        second_prediction,
        selected,
    )

    assert final_state.processed_match_ids == frozenset({"m1", "m2"})


def test_prediction_ledger_is_append_only_hash_chained_and_tamper_evident(
    tmp_path: Path,
) -> None:
    selected = config()
    state = initialize_season("2099/00", seeds(), selected)
    ledger = tmp_path / "pre_match_log.csv"
    first = lock_prediction(
        state,
        fixture("m1"),
        selected,
        generated_at_utc=GENERATED,
    )
    second = lock_prediction(
        state,
        fixture("m2", kickoff=KICKOFF + timedelta(days=1), home="B", away="C"),
        selected,
        generated_at_utc=GENERATED,
    )
    first_hash = append_prediction_lock(ledger, first)
    second_hash = append_prediction_lock(ledger, second)

    data = pd.read_csv(ledger, dtype=str, keep_default_na=False)
    assert list(data.columns) == PRE_MATCH_LOG_COLUMNS
    assert data.loc[0, "record_hash"] == first_hash
    assert data.loc[1, "previous_record_hash"] == first_hash
    assert data.loc[1, "record_hash"] == second_hash
    loaded = read_locked_prediction(ledger, "m2")
    assert loaded == second
    assert loaded.expected_home_score == pytest.approx(
        second.expected_home_score,
        abs=1e-15,
    )
    with pytest.raises(ValueError, match="already contains"):
        append_prediction_lock(ledger, first)

    data.loc[0, "expected_home_score"] = "0.123"
    data.to_csv(ledger, index=False)
    with pytest.raises(ValueError, match="record hash"):
        read_locked_prediction(ledger, "m1")


def test_state_checkpoint_preserves_open_ties_and_processed_matches(
    tmp_path: Path,
) -> None:
    selected = config()
    state = initialize_season("2099/00", seeds(), selected)
    first_leg = result(
        "semi-1",
        tie_id="semi",
        knockout=True,
        round_name="Semi Finals",
        stage="SEMIFINAL",
    )
    state, _ = update_match(state, first_leg, selected)
    save_state_checkpoint(state, tmp_path, selected)
    loaded = load_state_checkpoint(tmp_path, selected)

    assert loaded.processed_match_ids == frozenset({"semi-1"})
    assert loaded.open_ties == state.open_ties
    assert loaded.last_event_utc == state.last_event_utc
    assert loaded.last_match_id == state.last_match_id
    assert loaded.ratings["A"].power_elo == pytest.approx(state.ratings["A"].power_elo)

    second_leg = result(
        "semi-2",
        kickoff=KICKOFF + timedelta(days=7),
        home="B",
        away="A",
        tie_id="semi",
        knockout=True,
        decider=True,
        advanced="A",
        round_name="Semi Finals",
        stage="SEMIFINAL",
    )
    completed, _ = update_match(loaded, second_leg, selected)
    assert completed.processed_match_ids == frozenset({"semi-1", "semi-2"})
    assert completed.open_ties == {}

    checkpoint = json.loads((tmp_path / "state_checkpoint.json").read_text())
    checkpoint["processed_match_ids"].append("semi-1")
    (tmp_path / "state_checkpoint.json").write_text(json.dumps(checkpoint))
    with pytest.raises(ValueError, match="duplicate processed"):
        load_state_checkpoint(tmp_path, selected)


def test_state_checkpoint_rejects_tampered_ratings_csv(tmp_path: Path) -> None:
    selected = config()
    state = initialize_season("2099/00", seeds(), selected)
    save_state_checkpoint(state, tmp_path, selected)
    ratings_path = tmp_path / "ratings_state.csv"
    ratings = pd.read_csv(ratings_path)
    ratings.loc[0, "power_elo"] += 1.0
    ratings.to_csv(ratings_path, index=False)

    with pytest.raises(ValueError, match="checksum"):
        load_state_checkpoint(tmp_path, selected)


def test_future_previous_season_state_is_rejected() -> None:
    selected = config()
    future = initialize_season("2099/00", seeds(), selected)
    with pytest.raises(ValueError, match="earlier season"):
        initialize_season("2098/99", seeds(), selected, previous_state=future)


def test_live_cli_initializes_locks_and_settles_one_match(tmp_path: Path) -> None:
    ratings = tmp_path / "initial_ratings.csv"
    fixtures = tmp_path / "fixture.csv"
    results = tmp_path / "result.csv"
    state_dir = tmp_path / "state"
    pd.DataFrame(
        [
            {"season": "2099/00", "team_id": "A", "team_name": "Alpha", "ao_first_elo": 1500},
            {"season": "2099/00", "team_id": "B", "team_name": "Beta", "ao_first_elo": 1400},
        ]
    ).to_csv(ratings, index=False)
    pd.DataFrame(
        [
            {
                "match_id": "m1",
                "season": "2099/00",
                "kickoff_utc": "2099-09-01T19:00:00Z",
                "competition": "UCL",
                "round": "League Stage",
                "tie_id": "",
                "is_knockout": False,
                "is_tie_decider": False,
                "stage": "LEAGUE",
                "home_team_id": "A",
                "away_team_id": "B",
                "is_neutral": False,
            }
        ],
        columns=FIXTURE_INPUT_COLUMNS,
    ).to_csv(fixtures, index=False)
    pd.DataFrame(
        [
            {
                **pd.read_csv(fixtures).iloc[0].to_dict(),
                "home_goals": 2,
                "away_goals": 1,
                "decided_on_penalties": False,
                "advanced_team_id": "",
            }
        ],
        columns=MATCH_INPUT_COLUMNS,
    ).to_csv(results, index=False)

    base = [
        sys.executable,
        str(ROOT / "scripts" / "run_dynamic_live.py"),
        "--model-manifest",
        str(MANIFEST),
    ]
    subprocess.run(
        [*base, "initialize", "--initial-ratings", str(ratings), "--state-dir", str(state_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [*base, "lock", "--state-dir", str(state_dir), "--fixture", str(fixtures)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [*base, "settle", "--state-dir", str(state_dir), "--result", str(results)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert (state_dir / "pre_match_log.csv").exists()
    assert (state_dir / "match_updates.csv").exists()
    selected = config()
    final_state = load_state_checkpoint(state_dir, selected)
    assert final_state.processed_match_ids == frozenset({"m1"})
