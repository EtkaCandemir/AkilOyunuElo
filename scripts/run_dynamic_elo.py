from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ao_elo.dynamic_csv import load_selected_v2_config, run_batch  # noqa: E402


DEFAULT_MANIFEST = (
    ROOT / "contracts" / "ao_european_elo_v2_production.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay completed AO European Elo v2 matches in exact UTC order; "
            "this command does not create prospective holdout evidence"
        )
    )
    parser.add_argument(
        "--initial-ratings",
        type=Path,
        required=True,
        help="CSV containing season, team_id, team_name and ao_first_elo",
    )
    parser.add_argument(
        "--matches",
        type=Path,
        required=True,
        help="Exact-UTC matches.csv input",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--previous-state",
        type=Path,
        help="Optional prior-season ratings_state.csv used for carry",
    )
    parser.add_argument(
        "--model-manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Frozen v2 calibration manifest",
    )
    args = parser.parse_args()

    config = load_selected_v2_config(args.model_manifest.resolve())
    final_state, updates = run_batch(
        args.initial_ratings.resolve(),
        args.matches.resolve(),
        args.output_dir.resolve(),
        config,
        previous_state_csv=(
            args.previous_state.resolve() if args.previous_state else None
        ),
    )
    ratings = list(final_state.ratings.values())
    print("AO European Elo v2 retrospective replay complete")
    print(f"Model version: {config.model_version}")
    print(f"Config id: {config.config_id}")
    print(f"Matches processed: {len(updates)}")
    print(
        "Progression bonus events: "
        f"{sum(update.progression_bonus_added > 0.0 for update in updates)}"
    )
    print(
        "Progression bonus added: "
        f"{sum(update.progression_bonus_added for update in updates):.3f}"
    )
    print(f"Teams in state: {len(ratings)}")
    print(
        "AO Live Elo range: "
        f"{min(row.ao_live_elo for row in ratings):.3f} - "
        f"{max(row.ao_live_elo for row in ratings):.3f}"
    )
    print("Prospective holdout evidence: false")
    print("Prediction audit: replay_predictions.csv")
    print(f"Output directory: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
