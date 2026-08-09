from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ao_elo.dynamic import initialize_season  # noqa: E402
from ao_elo.dynamic_csv import (  # noqa: E402
    append_match_update,
    load_selected_v2_config,
    load_state_checkpoint,
    lock_fixture_to_ledger,
    read_fixtures,
    read_matches,
    read_team_seeds,
    save_state_checkpoint,
    settle_match_from_ledger,
)


DEFAULT_MANIFEST = (
    ROOT / "contracts" / "ao_european_elo_v2_production.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Operate the AO European Elo two-phase live workflow"
    )
    parser.add_argument(
        "--model-manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Frozen v2 calibration manifest",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    initialize = commands.add_parser("initialize", help="Create a resumable season state")
    initialize.add_argument("--initial-ratings", type=Path, required=True)
    initialize.add_argument("--state-dir", type=Path, required=True)
    initialize.add_argument(
        "--previous-season-state-dir",
        type=Path,
        help="Optional complete checkpoint used for season carry",
    )

    lock = commands.add_parser("lock", help="Lock one result-free pre-match prediction")
    lock.add_argument("--state-dir", type=Path, required=True)
    lock.add_argument("--fixture", type=Path, required=True)
    lock.add_argument(
        "--ledger",
        type=Path,
        help="Append-only prediction ledger; defaults inside state-dir",
    )

    settle = commands.add_parser("settle", help="Settle one previously locked result")
    settle.add_argument("--state-dir", type=Path, required=True)
    settle.add_argument("--result", type=Path, required=True)
    settle.add_argument(
        "--ledger",
        type=Path,
        help="Prediction ledger; defaults inside state-dir",
    )
    settle.add_argument(
        "--updates-log",
        type=Path,
        help="Append-only settlement audit; defaults inside state-dir",
    )

    args = parser.parse_args()
    config = load_selected_v2_config(args.model_manifest.resolve())
    if args.command == "initialize":
        initialize_command(args, config)
    elif args.command == "lock":
        lock_command(args, config)
    else:
        settle_command(args, config)


def initialize_command(args: argparse.Namespace, config) -> None:
    season, seeds = read_team_seeds(args.initial_ratings.resolve())
    previous = None
    if args.previous_season_state_dir:
        previous = load_state_checkpoint(
            args.previous_season_state_dir.resolve(),
            config,
        )
    state = initialize_season(season, seeds, config, previous_state=previous)
    save_state_checkpoint(state, args.state_dir.resolve(), config)
    print("AO live state initialized")
    print(f"Season: {state.season}")
    print(f"Teams: {len(state.ratings)}")
    print(f"State directory: {args.state_dir.resolve()}")


def lock_command(args: argparse.Namespace, config) -> None:
    state_dir = args.state_dir.resolve()
    state = load_state_checkpoint(state_dir, config)
    fixtures = read_fixtures(args.fixture.resolve())
    if len(fixtures) != 1:
        raise ValueError("Live lock command requires exactly one fixture row")
    ledger = args.ledger.resolve() if args.ledger else state_dir / "pre_match_log.csv"
    generated_at = datetime.now(timezone.utc)
    prediction = lock_fixture_to_ledger(
        state,
        fixtures[0],
        config,
        ledger,
        generated_at_utc=generated_at,
    )
    print("Pre-match prediction locked")
    print(f"Match: {prediction.match_id}")
    print(f"Generated UTC: {prediction.generated_at_utc.isoformat()}")
    print(f"Expected home score: {prediction.expected_home_score:.6f}")
    print(
        "1X2 probabilities: "
        f"H={prediction.home_win_probability:.6f}, "
        f"D={prediction.draw_probability:.6f}, "
        f"A={prediction.away_win_probability:.6f}"
    )
    print(f"Ledger: {ledger}")


def settle_command(args: argparse.Namespace, config) -> None:
    state_dir = args.state_dir.resolve()
    state = load_state_checkpoint(state_dir, config)
    matches = read_matches(args.result.resolve())
    if len(matches) != 1:
        raise ValueError("Live settle command requires exactly one result row")
    ledger = args.ledger.resolve() if args.ledger else state_dir / "pre_match_log.csv"
    updates_log = (
        args.updates_log.resolve() if args.updates_log else state_dir / "match_updates.csv"
    )
    final_state, update = settle_match_from_ledger(
        state,
        matches[0],
        config,
        ledger,
    )
    append_match_update(updates_log, update)
    save_state_checkpoint(final_state, state_dir, config)
    print("Locked match settled")
    print(f"Match: {update.match_id}")
    print(f"Power delta: {update.power_delta:+.6f}")
    if update.progression_bonus_added > 0.0:
        print(
            "Progression bonus: "
            f"{update.progression_bonus_recipient_id} "
            f"+{update.progression_bonus_added:.3f}"
        )
    print(f"State directory: {state_dir}")


if __name__ == "__main__":
    main()
