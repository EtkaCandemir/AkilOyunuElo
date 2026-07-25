from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ao_elo.dynamic_csv import (  # noqa: E402
    load_selected_v2_config,
    read_fixtures,
    read_matches,
    read_team_seeds,
)
from ao_elo.goal_shadow import (  # noqa: E402
    GoalShadowArm,
    GoalShadowPrediction,
    GoalShadowState,
    GoalShadowUpdate,
    initialize_goal_shadow,
    lock_goal_shadow,
    settle_goal_shadow,
    validate_goal_shadow_state,
)


DEFAULT_MODEL_MANIFEST = (
    ROOT / "contracts" / "ao_european_elo_v2_production.json"
)
DEFAULT_SHADOW_CONFIG = (
    ROOT
    / "output"
    / "goal_shadow_parameter_search_2018_2026"
    / "recommended_shadow_config.json"
)
STATE_FILENAME = "goal_shadow_state.json"
LEDGER_FILENAME = "goal_shadow_pre_match_log.csv"
UPDATES_FILENAME = "goal_shadow_updates.csv"

PREDICTION_COLUMNS = [
    "schema_version",
    "arm_name",
    "alpha",
    "tau",
    "match_id",
    "season",
    "kickoff_utc",
    "generated_at_utc",
    "competition",
    "round",
    "home_team_id",
    "away_team_id",
    "is_neutral",
    "home_rating_pre",
    "away_rating_pre",
    "expected_home_score",
    "home_win_probability",
    "draw_probability",
    "away_win_probability",
    "active_config_id",
    "shadow_config_id",
    "previous_record_hash",
    "record_hash",
]

UPDATE_COLUMNS = [
    "schema_version",
    "arm_name",
    "alpha",
    "tau",
    "match_id",
    "season",
    "kickoff_utc",
    "competition",
    "round",
    "home_team_id",
    "away_team_id",
    "home_goals",
    "away_goals",
    "decided_on_penalties",
    "home_rating_pre",
    "away_rating_pre",
    "effective_rating_difference",
    "expected_home_score",
    "home_win_probability",
    "draw_probability",
    "away_win_probability",
    "actual_home_score",
    "goal_difference",
    "goal_multiplier",
    "power_delta",
    "home_rating_post",
    "away_rating_post",
    "zero_sum_error",
    "active_config_id",
    "shadow_config_id",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Operate independent AO goal-difference shadow states"
    )
    parser.add_argument(
        "--model-manifest",
        type=Path,
        default=DEFAULT_MODEL_MANIFEST,
    )
    parser.add_argument(
        "--shadow-config",
        type=Path,
        default=DEFAULT_SHADOW_CONFIG,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    initialize = commands.add_parser("initialize")
    initialize.add_argument("--initial-ratings", type=Path, required=True)
    initialize.add_argument("--state-dir", type=Path, required=True)

    lock = commands.add_parser("lock")
    lock.add_argument("--state-dir", type=Path, required=True)
    lock.add_argument("--fixture", type=Path, required=True)
    lock.add_argument("--ledger", type=Path)

    settle = commands.add_parser("settle")
    settle.add_argument("--state-dir", type=Path, required=True)
    settle.add_argument("--result", type=Path, required=True)
    settle.add_argument("--ledger", type=Path)
    settle.add_argument("--updates-log", type=Path)

    replay = commands.add_parser("replay")
    replay.add_argument("--initial-ratings", type=Path, required=True)
    replay.add_argument("--matches", type=Path, required=True)
    replay.add_argument("--output-dir", type=Path, required=True)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--updates-log", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args()
    config = load_selected_v2_config(args.model_manifest.resolve())
    arms = load_shadow_arms(args.shadow_config.resolve())
    if args.command == "initialize":
        initialize_command(args, config, arms)
    elif args.command == "lock":
        lock_command(args, config)
    elif args.command == "settle":
        settle_command(args, config)
    elif args.command == "replay":
        replay_command(args, config, arms)
    else:
        evaluate_command(args)


def load_shadow_arms(path: Path) -> tuple[GoalShadowArm, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("prospective_shadow_arms")
    if not isinstance(records, list) or not records:
        raise ValueError("Shadow config lacks prospective_shadow_arms")
    arms = tuple(
        GoalShadowArm(
            name=str(record["name"]),
            alpha=float(record["alpha"]),
            tau=float(record["tau"]),
        )
        for record in records
    )
    for arm in arms:
        arm.validate()
    return arms


def initialize_command(args, config, arms) -> None:
    season, seeds = read_team_seeds(args.initial_ratings.resolve())
    state = initialize_goal_shadow(season, seeds, arms, config)
    state_dir = args.state_dir.resolve()
    save_shadow_state(state, state_dir, config)
    print("Goal-difference shadow state initialized")
    print(f"Season: {state.season}")
    print(f"Arms: {', '.join(arm.name for arm in state.arms)}")
    print(f"State directory: {state_dir}")


def lock_command(args, config) -> None:
    state_dir = args.state_dir.resolve()
    state = load_shadow_state(state_dir, config)
    fixtures = read_fixtures(args.fixture.resolve())
    if len(fixtures) != 1:
        raise ValueError("Shadow lock requires exactly one fixture")
    ledger = args.ledger.resolve() if args.ledger else state_dir / LEDGER_FILENAME
    predictions = lock_goal_shadow(
        state,
        fixtures[0],
        config,
        generated_at_utc=datetime.now(timezone.utc),
    )
    append_predictions(ledger, predictions)
    print("Goal-difference shadow predictions locked")
    print(f"Match: {fixtures[0].match_id}")
    print(f"Arms: {len(predictions)}")
    print(f"Ledger: {ledger}")


def settle_command(args, config) -> None:
    state_dir = args.state_dir.resolve()
    state = load_shadow_state(state_dir, config)
    matches = read_matches(args.result.resolve())
    if len(matches) != 1:
        raise ValueError("Shadow settle requires exactly one result")
    match = matches[0]
    ledger = args.ledger.resolve() if args.ledger else state_dir / LEDGER_FILENAME
    updates_log = (
        args.updates_log.resolve()
        if args.updates_log
        else state_dir / UPDATES_FILENAME
    )
    locked = read_locked_predictions(ledger, match.match_id)
    final_state, updates = settle_goal_shadow(state, match, locked, config)
    append_updates(updates_log, updates)
    save_shadow_state(final_state, state_dir, config)
    print("Goal-difference shadow match settled")
    print(f"Match: {match.match_id}")
    for update in updates:
        print(
            f"{update.arm_name}: M={update.goal_multiplier:.6f}, "
            f"delta={update.power_delta:+.6f}"
        )


def replay_command(args, config, arms) -> None:
    season, seeds = read_team_seeds(args.initial_ratings.resolve())
    matches = read_matches(args.matches.resolve())
    ordered = tuple(sorted(matches, key=lambda match: (match.kickoff_utc, match.match_id)))
    state = initialize_goal_shadow(season, seeds, arms, config)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = output_dir / LEDGER_FILENAME
    updates_log = output_dir / UPDATES_FILENAME
    for match in ordered:
        predictions = lock_goal_shadow(
            state,
            match.fixture(),
            config,
            generated_at_utc=match.kickoff_utc - timedelta(seconds=1),
        )
        append_predictions(ledger, predictions)
        state, updates = settle_goal_shadow(state, match, predictions, config)
        append_updates(updates_log, updates)
    save_shadow_state(state, output_dir, config)
    summary = evaluate_updates(pd.read_csv(updates_log))
    summary.to_csv(output_dir / "goal_shadow_summary.csv", index=False)
    print("Goal-difference shadow replay complete")
    print(f"Matches: {len(ordered)}")
    print(f"Output: {output_dir}")


def evaluate_command(args) -> None:
    updates = pd.read_csv(args.updates_log.resolve())
    summary = evaluate_updates(updates)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "goal_shadow_summary.csv", index=False)
    print(summary.to_string(index=False))


def save_shadow_state(
    state: GoalShadowState,
    root: Path,
    config,
) -> None:
    validate_goal_shadow_state(state, config)
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0.0",
        "season": state.season,
        "arms": [asdict(arm) for arm in state.arms],
        "team_names": dict(state.team_names),
        "ao_first_elo": dict(state.ao_first_elo),
        "ratings": {
            arm: dict(ratings) for arm, ratings in state.ratings.items()
        },
        "processed_match_ids": sorted(state.processed_match_ids),
        "last_event_utc": (
            state.last_event_utc.isoformat()
            if state.last_event_utc is not None
            else None
        ),
        "last_match_id": state.last_match_id,
        "active_config_id": state.active_config_id,
        "shadow_config_id": state.shadow_config_id,
    }
    contents = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
    path = root / STATE_FILENAME
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(contents)
    temporary.replace(path)


def load_shadow_state(root: Path, config) -> GoalShadowState:
    path = root / STATE_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0.0":
        raise ValueError("Unsupported shadow state schema")
    state = GoalShadowState(
        season=str(payload["season"]),
        arms=tuple(GoalShadowArm(**record) for record in payload["arms"]),
        team_names={
            str(team_id): str(name)
            for team_id, name in payload["team_names"].items()
        },
        ao_first_elo={
            str(team_id): float(value)
            for team_id, value in payload["ao_first_elo"].items()
        },
        ratings={
            str(arm): {
                str(team_id): float(value)
                for team_id, value in ratings.items()
            }
            for arm, ratings in payload["ratings"].items()
        },
        processed_match_ids=frozenset(
            str(match_id) for match_id in payload["processed_match_ids"]
        ),
        last_event_utc=(
            datetime.fromisoformat(payload["last_event_utc"])
            if payload["last_event_utc"]
            else None
        ),
        last_match_id=(
            str(payload["last_match_id"])
            if payload["last_match_id"] is not None
            else None
        ),
        active_config_id=str(payload["active_config_id"]),
        shadow_config_id=str(payload["shadow_config_id"]),
    )
    validate_goal_shadow_state(state, config)
    return state


def append_predictions(
    path: Path,
    predictions: tuple[GoalShadowPrediction, ...],
) -> None:
    records = read_prediction_records(path)
    existing = {
        (record["match_id"], record["arm_name"]) for record in records
    }
    previous_hash = records[-1]["record_hash"] if records else ""
    new_records = []
    for prediction in predictions:
        key = (prediction.match_id, prediction.arm_name)
        if key in existing:
            raise ValueError(f"Duplicate shadow prediction: {key}")
        record = {
            "schema_version": "1.0.0",
            "arm_name": prediction.arm_name,
            "alpha": _number(prediction.alpha),
            "tau": _number(prediction.tau),
            "match_id": prediction.match_id,
            "season": prediction.season,
            "kickoff_utc": prediction.kickoff_utc.isoformat(),
            "generated_at_utc": prediction.generated_at_utc.isoformat(),
            "competition": prediction.competition,
            "round": prediction.round,
            "home_team_id": prediction.home_team_id,
            "away_team_id": prediction.away_team_id,
            "is_neutral": str(prediction.is_neutral).lower(),
            "home_rating_pre": _number(prediction.home_rating_pre),
            "away_rating_pre": _number(prediction.away_rating_pre),
            "expected_home_score": _number(prediction.expected_home_score),
            "home_win_probability": _number(prediction.home_win_probability),
            "draw_probability": _number(prediction.draw_probability),
            "away_win_probability": _number(prediction.away_win_probability),
            "active_config_id": prediction.active_config_id,
            "shadow_config_id": prediction.shadow_config_id,
            "previous_record_hash": previous_hash,
            "record_hash": "",
        }
        record["record_hash"] = _record_hash(record, PREDICTION_COLUMNS)
        previous_hash = record["record_hash"]
        new_records.append(record)
    _append_records(path, PREDICTION_COLUMNS, new_records)


def read_locked_predictions(
    path: Path,
    match_id: str,
) -> tuple[GoalShadowPrediction, ...]:
    records = read_prediction_records(path)
    selected = [record for record in records if record["match_id"] == match_id]
    if not selected:
        raise ValueError(f"No shadow prediction found for match_id: {match_id}")
    return tuple(
        GoalShadowPrediction(
            arm_name=record["arm_name"],
            alpha=float(record["alpha"]),
            tau=float(record["tau"]),
            match_id=record["match_id"],
            season=record["season"],
            kickoff_utc=datetime.fromisoformat(record["kickoff_utc"]),
            generated_at_utc=datetime.fromisoformat(record["generated_at_utc"]),
            competition=record["competition"],
            round=record["round"],
            home_team_id=record["home_team_id"],
            away_team_id=record["away_team_id"],
            is_neutral=_boolean(record["is_neutral"]),
            home_rating_pre=float(record["home_rating_pre"]),
            away_rating_pre=float(record["away_rating_pre"]),
            expected_home_score=float(record["expected_home_score"]),
            home_win_probability=float(record["home_win_probability"]),
            draw_probability=float(record["draw_probability"]),
            away_win_probability=float(record["away_win_probability"]),
            active_config_id=record["active_config_id"],
            shadow_config_id=record["shadow_config_id"],
        )
        for record in selected
    )


def read_prediction_records(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != PREDICTION_COLUMNS:
            raise ValueError("Shadow prediction ledger schema mismatch")
        records = list(reader)
    previous_hash = ""
    for record in records:
        if record["previous_record_hash"] != previous_hash:
            raise ValueError("Shadow prediction hash chain is broken")
        expected = _record_hash(record, PREDICTION_COLUMNS)
        if record["record_hash"] != expected:
            raise ValueError("Shadow prediction record hash is invalid")
        previous_hash = record["record_hash"]
    return records


def append_updates(path: Path, updates: tuple[GoalShadowUpdate, ...]) -> None:
    existing: set[tuple[str, str]] = set()
    if path.exists():
        current = pd.read_csv(path, dtype=str)
        if list(current.columns) != UPDATE_COLUMNS:
            raise ValueError("Shadow update log schema mismatch")
        existing = set(zip(current["match_id"], current["arm_name"]))
    records = []
    for update in updates:
        key = (update.match_id, update.arm_name)
        if key in existing:
            raise ValueError(f"Duplicate shadow update: {key}")
        record = {
            "schema_version": "1.0.0",
            **{
                key: _serialize(value)
                for key, value in asdict(update).items()
            },
        }
        records.append(record)
    _append_records(path, UPDATE_COLUMNS, records)


def evaluate_updates(updates: pd.DataFrame) -> pd.DataFrame:
    required = set(UPDATE_COLUMNS) - {"schema_version"}
    missing = sorted(required - set(updates.columns))
    if missing:
        raise ValueError(f"Shadow updates missing columns: {missing}")
    rows = []
    for arm, frame in updates.groupby("arm_name", sort=False):
        probabilities = frame[
            ["home_win_probability", "draw_probability", "away_win_probability"]
        ].to_numpy(float)
        actual_score = frame["actual_home_score"].to_numpy(float)
        observed = np.where(
            actual_score == 1.0,
            0,
            np.where(actual_score == 0.5, 1, 2),
        )
        targets = pd.get_dummies(observed).reindex(columns=[0, 1, 2], fill_value=0)
        brier = ((probabilities - targets.to_numpy(float)) ** 2).sum(axis=1)
        chosen = probabilities[range(len(probabilities)), observed]
        log_loss = -pd.Series(chosen).clip(lower=1e-15).map(math.log)
        rows.append(
            {
                "arm_name": arm,
                "alpha": float(frame["alpha"].iloc[0]),
                "tau": float(frame["tau"].iloc[0]),
                "matches": len(frame),
                "brier_1x2": float(brier.mean()),
                "log_loss_1x2": float(log_loss.mean()),
                "accuracy_1x2": float(
                    (probabilities.argmax(axis=1) == observed).mean()
                ),
                "maximum_goal_multiplier": float(
                    frame["goal_multiplier"].max()
                ),
                "maximum_abs_power_delta": float(
                    frame["power_delta"].abs().max()
                ),
                "maximum_zero_sum_error": float(
                    frame["zero_sum_error"].max()
                ),
            }
        )
    result = pd.DataFrame(rows)
    base = result.loc[result["arm_name"].eq("BASE")]
    if len(base) != 1:
        raise ValueError("Shadow update evaluation requires exactly one BASE arm")
    result["brier_delta_vs_base"] = (
        result["brier_1x2"] - float(base.iloc[0]["brier_1x2"])
    )
    result["log_loss_delta_vs_base"] = (
        result["log_loss_1x2"] - float(base.iloc[0]["log_loss_1x2"])
    )
    return result
def _append_records(
    path: Path,
    columns: list[str],
    records: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerows(records)


def _record_hash(record: dict[str, str], columns: list[str]) -> str:
    payload = {
        column: ("" if column == "record_hash" else record[column])
        for column in columns
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _serialize(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return _number(value)
    return str(value)


def _number(value: float) -> str:
    return format(float(value), ".17g")


def _boolean(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"Invalid boolean: {value}")


if __name__ == "__main__":
    main()
