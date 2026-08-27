from __future__ import annotations

"""Rebuild the current-contract Domestic Surprise seed artifact without mutation.

The historical evaluator previously read a seed artifact derived from an older
0.85 exposure snapshot.  This script writes a separate, reviewable 0.65
artifact.  Switching any production-facing pointer remains an explicit change.
"""

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ao_elo.config import AOEuropeanEloConfig  # noqa: E402
from ao_elo.domestic_surprise_amplification import production_control_config  # noqa: E402
from scripts.run_domestic_surprise_amplification_backtest import (  # noqa: E402
    CLUB_IDENTITY,
    DOMESTIC_ADJUSTMENTS,
    STATIC_DATA_ROOT,
    build_candidate_effects,
)
from scripts.run_domestic_surprise_5y_backtest import build_domestic_history_features  # noqa: E402
from scripts.run_v2_achievement_reserve_calibration import load_reserve_data  # noqa: E402
from scripts.run_stage_weighted_progression_backtest import EVENTS_PATH  # noqa: E402


OUTPUT_ROOT = ROOT / "output" / "domestic_surprise_current_contract_baseline_2018_2026"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a clean current-contract Surprise seed artifact")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    config = AOEuropeanEloConfig.active()
    config.validate()
    if config.max_european_exposure != 0.65:
        raise AssertionError("Expected active effective European exposure cap 0.65")
    datasets, _ = load_reserve_data(STATIC_DATA_ROOT, EVENTS_PATH, config)
    seasons = tuple(data.season for data in datasets)
    features, history, coverage = build_domestic_history_features(
        STATIC_DATA_ROOT, config, seasons, CLUB_IDENTITY
    )
    effects = build_candidate_effects(features, production_control_config(), config)
    effects["adjusted_ao_first_elo"] = (
        effects["ao_first_elo_without_rebuilt_surprise"]
        + effects["ao_first_elo_adjustment"]
    )
    if effects.duplicated(["season", "team_id"]).any():
        raise ValueError("Rebuilt seed artifact contains duplicate season/team keys")

    stored = pd.read_csv(DOMESTIC_ADJUSTMENTS)[
        ["season", "team_id", "adjusted_ao_first_elo"]
    ].rename(columns={"adjusted_ao_first_elo": "stored_adjusted_ao_first_elo"})
    audit = effects.merge(stored, on=["season", "team_id"], validate="one_to_one")
    audit["seed_difference"] = (
        audit["adjusted_ao_first_elo"] - audit["stored_adjusted_ao_first_elo"]
    )

    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    effects.to_csv(output / "rebuilt_current_contract_seed_adjustments.csv", index=False)
    features.to_csv(output / "rebuilt_domestic_surprise_features.csv", index=False)
    history.to_csv(output / "rebuilt_domestic_history_long.csv", index=False)
    coverage.to_csv(output / "history_coverage.csv", index=False)
    audit.to_csv(output / "stored_seed_reconciliation.csv", index=False)
    summary = {
        "production_change": False,
        "model_version": config.model_version,
        "effective_european_exposure_cap": config.max_european_exposure,
        "domestic_surprise": {
            "theta": 0.40,
            "variance_penalty": 0.50,
            "domestic_cap": 30.0,
            "exposure_family": "LINEAR",
        },
        "team_seasons": int(len(effects)),
        "maximum_absolute_seed_difference_vs_stored": float(
            audit["seed_difference"].abs().max()
        ),
        "mean_absolute_seed_difference_vs_stored": float(
            audit["seed_difference"].abs().mean()
        ),
        "artifact": "rebuilt_current_contract_seed_adjustments.csv",
        "stored_artifact": str(DOMESTIC_ADJUSTMENTS.relative_to(ROOT)),
    }
    (output / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Output: {output}")


if __name__ == "__main__":
    main()

