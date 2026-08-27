from __future__ import annotations

"""Rebuild the pre-match feature store from participation-normalized seeds.

The served ML/Poisson blend was selected while every arm read the production
seed. The participation candidate moves 53.7% of club-seasons, and the
Structural Logistic features include the AO log-odds and the AO First/Live
rating differences, so the served layer cannot be assumed safe under the new
seed without re-scoring it. This script produces the candidate-seed half of
that comparison; the ensemble backtest then consumes it through
`--feature-store`.

Nothing here changes a production file. The production contract is hashed on
entry and on exit and the run fails if it moved.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ao_elo.european_participation import (  # noqa: E402
    NORMALIZED,
    ParticipationNormalizationConfig,
    apply_participation_normalization,
    production_control_config,
)
from ao_elo.ml_features import build_pre_match_feature_store  # noqa: E402
from scripts.run_european_participation_backtest import (  # noqa: E402
    load_seed_evidence,
)
from scripts.run_opponent_quintile_backtest import (  # noqa: E402
    load_production_baseline,
)
from scripts.run_team_venue_context_backtest import (  # noqa: E402
    build_production_baseline,
)


DOMESTIC_MATCHES = (
    ROOT / "data" / "domestic_league_matches_2013_2026" / "domestic_matches.csv"
)
MATCH_METADATA = (
    ROOT / "data" / "external_elo_benchmark_2018_2026" / "exact_date_events.csv"
)
INITIAL_CONTEXT = (
    ROOT / "output" / "current_model_evaluation_2018_2026" / "initial_elo_impact.csv"
)
PRODUCTION_CONTRACT = ROOT / "contracts" / "ao_european_elo_v2_production.json"
OUTPUT_ROOT = ROOT / "output" / "participation_served_ensemble_2018_2026"

# Modal fold selection in reports/european_participation/fold_selections.csv.
DEFAULT_SHRINKAGE = 0.20
EXPECTED_MATCHES = 6340


def seed_map(seeds: pd.DataFrame, config: ParticipationNormalizationConfig):
    """Return {(season, team_id): AO First Elo} for one participation arm."""

    applied = apply_participation_normalization(seeds, config)
    return {
        (str(row.season), int(row.team_id)): float(row.candidate_ao_first_elo)
        for row in applied.itertuples(index=False)
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the participation-normalized pre-match feature store so the "
            "served ML/Poisson blend can be re-scored under the candidate seed"
        )
    )
    parser.add_argument("--shrinkage", type=float, default=DEFAULT_SHRINKAGE)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--domestic-matches", type=Path, default=DOMESTIC_MATCHES)
    parser.add_argument("--match-metadata", type=Path, default=MATCH_METADATA)
    parser.add_argument("--initial-context", type=Path, default=INITIAL_CONTEXT)
    args = parser.parse_args()

    contract_sha_before = hashlib.sha256(
        PRODUCTION_CONTRACT.read_bytes()
    ).hexdigest()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)

    print("Loading production replay and seed evidence", flush=True)
    baseline_replay, datasets, core, parameters, production_domestic = (
        load_production_baseline()
    )
    seasons = tuple(data.season for data in datasets)
    seeds, baseline_error = load_seed_evidence()
    print(f"  baseline reproduces served seed within {baseline_error:.3e} Elo")

    control = seed_map(seeds, production_control_config())
    candidate = seed_map(
        seeds,
        ParticipationNormalizationConfig(arm=NORMALIZED, shrinkage=args.shrinkage),
    )

    # The control arm must be the served seed the replay already uses, or the
    # two feature stores would differ for a reason other than the layer.
    drift = max(
        abs(control[key] - value) for key, value in production_domestic.items()
    )
    print(f"  control seed matches the replay seed within {drift:.3e} Elo")
    if drift > 1e-6:
        raise ValueError(
            f"Control seed drifted from the served seed by {drift:.6f} Elo"
        )

    moved = sum(
        1
        for key, value in candidate.items()
        if abs(value - control[key]) > 1e-9
    )
    print(
        f"  candidate k={args.shrinkage:g} moves {moved} / {len(candidate)} "
        "team-seasons"
    )

    print("Replaying matches on the candidate seed", flush=True)
    identity = _load_identity()
    xg = _load_xg(datasets)
    candidate_replay = build_production_baseline(
        datasets,
        core,
        parameters,
        candidate,
        xg,
        identity,
        seasons,
    )
    if len(candidate_replay) != EXPECTED_MATCHES:
        raise ValueError(
            f"Candidate replay produced {len(candidate_replay)} matches, "
            f"expected {EXPECTED_MATCHES}"
        )

    domestic_matches = pd.read_csv(args.domestic_matches.resolve(), low_memory=False)
    metadata = pd.read_csv(args.match_metadata.resolve(), low_memory=False)
    initial = pd.read_csv(args.initial_context.resolve(), low_memory=False)

    for label, replay, seed in (
        ("control", baseline_replay, control),
        ("candidate", candidate_replay, candidate),
    ):
        print(f"Building {label} pre-match feature store", flush=True)
        # `initial_context` carries the static seed features. Left at the
        # production file it would hand the candidate arm control-seed
        # `home_initial_rating` values, so the two stores would disagree on the
        # AO probability while agreeing on the seed that produced it.
        # `effective_european_exposure` is untouched: the layer renormalizes
        # history only and never changes the exposure weight.
        arm_initial = _rewrite_initial_context(initial, seed)
        features = build_pre_match_feature_store(
            replay,
            domestic_matches,
            match_metadata=metadata,
            initial_context=arm_initial,
        )
        path = output / f"pre_match_feature_store_{label}.csv"
        features.to_csv(path, index=False)
        print(f"  wrote {path.name} ({len(features)} rows)")

    contract_sha_after = hashlib.sha256(PRODUCTION_CONTRACT.read_bytes()).hexdigest()
    if contract_sha_before != contract_sha_after:
        raise ValueError("The production contract changed during this run")

    manifest = {
        "shrinkage": float(args.shrinkage),
        "matches": EXPECTED_MATCHES,
        "team_seasons": len(candidate),
        "team_seasons_moved_by_layer": moved,
        "baseline_reproduces_served_seed_elo": baseline_error,
        "control_seed_drift_elo": drift,
        "contract_sha256": contract_sha_after,
        "changes_production_parameters": False,
    }
    (output / "feature_store_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Output: {output}")


def _rewrite_initial_context(
    initial: pd.DataFrame,
    seed: dict[tuple[str, int], float],
) -> pd.DataFrame:
    """Return the seed context with this arm's AO First Elo substituted."""

    result = initial.copy()
    keys = list(zip(result["season"].astype(str), result["team_id"].astype(int)))
    missing = [key for key in keys if key not in seed]
    if missing:
        raise ValueError(
            f"Initial context carries {len(missing)} team-seasons absent from the seed map"
        )
    result["adjusted_ao_first_elo"] = [seed[key] for key in keys]
    return result


def _load_identity():
    from scripts.run_final_robustness import load_team_season_identity

    return load_team_season_identity()


def _load_xg(datasets):
    from scripts.run_stage_weighted_progression_backtest import XG_DATA, load_xg_map

    return load_xg_map(XG_DATA, datasets)


if __name__ == "__main__":
    main()
