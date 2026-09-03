from __future__ import annotations

"""Start the isolated 2026/27 Positive Bridge AO-core shadow arm.

This command never calls the production feature service and never writes the
production contract, rating state, served predictions, or prediction ledger.
It first reproduces the production AO-core probabilities, then creates a fresh
candidate seed, replays every completed qualifier, and locks one result-free
shadow prediction for every league-stage fixture.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ao_elo.dynamic import (  # noqa: E402
    LockedPrediction,
    MatchFixture,
    SeasonState,
    initialize_season,
    lock_prediction,
    run_season,
)
from ao_elo.dynamic_csv import (  # noqa: E402
    load_selected_v2_config,
    load_state_checkpoint,
    read_fixtures,
    read_matches,
    read_team_seeds,
    run_batch,
)
from ao_elo.european_exposure_shadow import (  # noqa: E402
    POSITIVE_BRIDGE_ARM_ID,
    POSITIVE_BRIDGE_ETA,
    POSITIVE_BRIDGE_EXPOSURE_CAP,
    apply_positive_bridge_seed_transform,
)
from ao_elo.prediction_ledger import (  # noqa: E402
    PREDICTION,
    LedgerEntry,
    ledger_head,
    read_entries,
    verify_ledger,
)
from ao_elo.shadow_prediction_ledger import (  # noqa: E402
    AO_CORE_SHADOW,
    REQUIRED_SHADOW_PREDICTION_FIELDS,
    SHADOW_PREDICTION,
    append_shadow_predictions,
    normalize_shadow_prediction_payload,
    read_shadow_entries,
    shadow_ledger_head,
    verify_shadow_ledger,
)


SEASON = "2026/27"
SHADOW_MODEL_VERSION = "ao-positive-bridge-020-core-shadow-v1"
EXPECTED_FIXTURE_COUNTS = {"UCL": 144, "UEL": 144, "UECL": 108}
BASELINE_TOLERANCE = 1e-12
ARTIFACT_PREDICTIONS_FILENAME = "ao_core_shadow_predictions_preledger.csv"

DEFAULT_CONTRACT = ROOT / "contracts" / "ao_european_elo_v2_production.json"
DEFAULT_PRODUCTION_STATE = (
    ROOT / "artifacts" / "production_prediction" / "domestic_poisson_state_2026_27.json"
)
DEFAULT_PRODUCTION_SEEDS = (
    ROOT / "output" / "season_2026_27_preproduction" / "ao_first_elo_2026_27.csv"
)
DEFAULT_COMPLETED_MATCHES = (
    ROOT / "data" / "season_2026_27_preproduction" / "matches_completed.csv"
)
DEFAULT_FIXTURES = (
    ROOT / "data" / "season_2026_27_preproduction" / "fixtures_upcoming.csv"
)
DEFAULT_PRODUCTION_CHECKPOINT = (
    ROOT / "output" / "season_2026_27_preproduction" / "q3_completed_replay"
)
DEFAULT_SERVED_PREDICTIONS = (
    ROOT
    / "output"
    / "season_2026_27_prediction_features"
    / "served_predictions.csv"
)
DEFAULT_PRODUCTION_LEDGER = (
    ROOT / "data" / "prediction_ledger" / "prediction_ledger_2026_27.jsonl"
)
DEFAULT_SPEC = (
    ROOT
    / "reports"
    / "european_exposure_shadow_monitoring"
    / "positive_bridge_020_shadow_spec.json"
)
DEFAULT_ARTIFACT_ROOT = (
    ROOT
    / "artifacts"
    / "shadow_prediction"
    / "positive_bridge_020"
    / "ao_core_2026_27_start"
)
DEFAULT_SHADOW_LEDGER = (
    ROOT
    / "data"
    / "prediction_ledger"
    / "shadow"
    / "positive_bridge_020"
    / "ao_core_shadow_2026_27.jsonl"
)
DEFAULT_SHADOW_ANCHOR = (
    ROOT
    / "data"
    / "prediction_ledger"
    / "shadow"
    / "positive_bridge_020"
    / "ao_core_shadow_anchor_2026_27.json"
)
DEFAULT_RUN_OUTPUT = (
    ROOT
    / "output"
    / "european_exposure_shadow_monitoring_2026_27"
    / "positive_bridge_020_start"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def utc_iso(value: object) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"Timestamp must be timezone-aware: {value!r}")
    return timestamp.tz_convert("UTC").isoformat()


def capture_production_fingerprints(paths: argparse.Namespace) -> dict[str, str]:
    files = {
        "contract": paths.contract,
        "production_state": paths.production_state,
        "production_seeds": paths.production_seeds,
        "completed_matches": paths.completed_matches,
        "fixtures": paths.fixtures,
        "checkpoint_ratings": paths.production_checkpoint / "ratings_state.csv",
        "checkpoint_metadata": paths.production_checkpoint / "state_checkpoint.json",
        "served_predictions": paths.served_predictions,
        "production_ledger": paths.production_ledger,
    }
    missing = [str(path) for path in files.values() if not Path(path).is_file()]
    if missing:
        raise ValueError(f"Required source files are missing: {missing}")
    return {name: sha256_file(Path(path)) for name, path in files.items()}


def implementation_fingerprints() -> dict[str, str]:
    files = {
        "runner": Path(__file__).resolve(),
        "positive_bridge_transform": SRC / "ao_elo" / "european_exposure_shadow.py",
        "shadow_ledger": SRC / "ao_elo" / "shadow_prediction_ledger.py",
        "production_ledger_reader": SRC / "ao_elo" / "prediction_ledger.py",
        "dynamic": SRC / "ao_elo" / "dynamic.py",
        "dynamic_csv": SRC / "ao_elo" / "dynamic_csv.py",
        "config": SRC / "ao_elo" / "config.py",
        "controlled_live": SRC / "ao_elo" / "controlled_live.py",
        "draw_probability": SRC / "ao_elo" / "draw_probability.py",
        "qualification_stage_k": SRC / "ao_elo" / "qualification_stage_k.py",
        "qualification_transition": SRC / "ao_elo" / "qualification_transition.py",
        "tournament_bonus": SRC / "ao_elo" / "tournament_bonus.py",
        "validators": SRC / "ao_elo" / "validators.py",
        "xg_live": SRC / "ao_elo" / "xg_live.py",
    }
    return {name: sha256_file(path) for name, path in files.items()}


def validate_output_isolation(args: argparse.Namespace) -> None:
    """Refuse aliases that could overwrite any production or source input."""

    source_files = {
        args.spec,
        args.contract,
        args.production_state,
        args.production_seeds,
        args.completed_matches,
        args.fixtures,
        args.served_predictions,
        args.production_ledger,
        args.production_checkpoint / "ratings_state.csv",
        args.production_checkpoint / "state_checkpoint.json",
    }
    file_outputs = (args.shadow_ledger, args.shadow_anchor)
    directory_outputs = (args.artifact_root, args.run_output)
    if (
        len(set(file_outputs)) != len(file_outputs)
        or file_outputs[0] in file_outputs[1].parents
        or file_outputs[1] in file_outputs[0].parents
    ):
        raise ValueError("Shadow ledger and anchor paths must be distinct and non-overlapping")
    for left_index, left in enumerate(directory_outputs):
        for right in directory_outputs[left_index + 1 :]:
            if (
                left == right
                or left in right.parents
                or right in left.parents
            ):
                raise ValueError("Shadow artifact and run-output directories cannot overlap")
    for output_file in file_outputs:
        for output_directory in directory_outputs:
            if (
                output_file == output_directory
                or output_file in output_directory.parents
                or output_directory in output_file.parents
            ):
                raise ValueError(
                    "Shadow ledger/anchor paths cannot overlap an output directory"
                )

    aliases = sorted(
        str(path)
        for path in (set(file_outputs) | set(directory_outputs)) & source_files
    )
    if aliases:
        raise ValueError(f"Shadow output aliases a protected source file: {aliases}")

    protected_artifact_root = (ROOT / "artifacts" / "production_prediction").resolve()
    for output_directory in directory_outputs:
        if (
            output_directory == protected_artifact_root
            or protected_artifact_root in output_directory.parents
        ):
            raise ValueError("Shadow output cannot be inside production_prediction")
        if any(
            output_directory == source.parent or output_directory in source.parents
            for source in source_files
        ):
            raise ValueError("Shadow output directory cannot contain a protected source file")
    for output_file in file_outputs:
        if (
            output_file == protected_artifact_root
            or protected_artifact_root in output_file.parents
        ):
            raise ValueError("Shadow output cannot be inside production_prediction")


def validate_executable_spec(
    spec_path: Path,
    *,
    contract_sha256: str,
    production_state_sha256: str,
    production_revision: str,
    source_hashes: Mapping[str, str],
    production_ledger_head: str,
    production_ledger_entries: int,
    implementation_hashes: Mapping[str, str],
) -> tuple[dict[str, object], str]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec_sha = sha256_file(spec_path)
    expected = {
        "arm_id": POSITIVE_BRIDGE_ARM_ID,
        "initial_scope": AO_CORE_SHADOW,
        "source_production_revision": production_revision,
        "source_production_contract_sha256": contract_sha256,
        "source_production_state_sha256": production_state_sha256,
        "spec_immutability": "RUN_STATUS_AND_ARTIFACT_POINTERS_MUST_LIVE_OUTSIDE_THIS_FILE",
    }
    mismatches = {
        key: (spec.get(key), value)
        for key, value in expected.items()
        if spec.get(key) != value
    }
    parameters = spec.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("Positive Bridge spec parameters must be an object")
    if not math.isclose(
        float(parameters.get("eta", math.nan)),
        POSITIVE_BRIDGE_ETA,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        mismatches["parameters.eta"] = (
            parameters.get("eta"),
            POSITIVE_BRIDGE_ETA,
        )
    if not math.isclose(
        float(parameters.get("exposure_cap", math.nan)),
        POSITIVE_BRIDGE_EXPOSURE_CAP,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        mismatches["parameters.exposure_cap"] = (
            parameters.get("exposure_cap"),
            POSITIVE_BRIDGE_EXPOSURE_CAP,
        )
    if mismatches:
        raise ValueError(f"Executable Positive Bridge spec mismatch: {mismatches}")

    expected_inputs = {
        "production_seeds_sha256": source_hashes["production_seeds"],
        "completed_matches_sha256": source_hashes["completed_matches"],
        "fixtures_sha256": source_hashes["fixtures"],
        "checkpoint_ratings_sha256": source_hashes["checkpoint_ratings"],
        "checkpoint_metadata_sha256": source_hashes["checkpoint_metadata"],
        "served_predictions_sha256": source_hashes["served_predictions"],
        "production_ledger_sha256": source_hashes["production_ledger"],
        "production_ledger_head": production_ledger_head,
        "production_ledger_entries": production_ledger_entries,
    }
    if spec.get("initial_run_inputs") != expected_inputs:
        raise ValueError(
            "Executable Positive Bridge spec does not match the canonical run inputs"
        )
    runtime_keys = (
        "positive_bridge_transform",
        "dynamic",
        "dynamic_csv",
        "config",
        "controlled_live",
        "draw_probability",
        "qualification_stage_k",
        "qualification_transition",
        "tournament_bonus",
        "validators",
        "xg_live",
    )
    expected_runtime = {key: implementation_hashes[key] for key in runtime_keys}
    if spec.get("candidate_runtime_source_sha256") != expected_runtime:
        raise ValueError(
            "Executable Positive Bridge spec does not match candidate runtime source"
        )
    return spec, spec_sha


def validate_fixture_universe(fixtures: Sequence[MatchFixture]) -> dict[str, object]:
    if len(fixtures) != sum(EXPECTED_FIXTURE_COUNTS.values()):
        raise ValueError(
            f"Expected 396 fixtures, found {len(fixtures)}; shadow start refused"
        )
    counts = pd.Series([fixture.competition for fixture in fixtures]).value_counts()
    actual_counts = {
        competition: int(counts.get(competition, 0))
        for competition in EXPECTED_FIXTURE_COUNTS
    }
    if actual_counts != EXPECTED_FIXTURE_COUNTS:
        raise ValueError(
            f"Fixture competition counts are incomplete: {actual_counts}; "
            f"expected {EXPECTED_FIXTURE_COUNTS}"
        )
    if any(
        fixture.stage != "LEAGUE"
        or fixture.round != "League Stage"
        or fixture.is_knockout
        or fixture.tie_id is not None
        for fixture in fixtures
    ):
        raise ValueError("Shadow start expects only canonical League Stage fixtures")
    kickoffs = [pd.Timestamp(fixture.kickoff_utc) for fixture in fixtures]
    return {
        "fixtures": len(fixtures),
        "competition_counts": actual_counts,
        "earliest_kickoff_utc": min(kickoffs).isoformat(),
        "latest_kickoff_utc": max(kickoffs).isoformat(),
    }


def compare_states(reproduced: SeasonState, checkpoint: SeasonState) -> dict[str, float]:
    scalar_fields = (
        "season",
        "processed_match_ids",
        "processed_tie_ids",
        "open_ties",
        "qualification_participants",
        "qualification_carry_applied",
        "last_event_utc",
        "last_match_id",
        "model_version",
        "config_id",
    )
    for field in scalar_fields:
        if getattr(reproduced, field) != getattr(checkpoint, field):
            raise ValueError(f"Production replay state mismatch: {field}")
    if set(reproduced.ratings) != set(checkpoint.ratings):
        raise ValueError("Production replay rating team universe differs from checkpoint")

    maxima = {
        "ao_first_elo": 0.0,
        "power_elo": 0.0,
        "achievement_reserve": 0.0,
        "progression_bonus_ucl": 0.0,
        "progression_bonus_uel": 0.0,
        "progression_bonus_uecl": 0.0,
        "ao_live_elo": 0.0,
    }
    for team_id in reproduced.ratings:
        left = reproduced.ratings[team_id]
        right = checkpoint.ratings[team_id]
        if (
            left.team_id != right.team_id
            or left.team_name != right.team_name
            or left.last_event_utc != right.last_event_utc
            or left.last_match_id != right.last_match_id
        ):
            raise ValueError(f"Production replay team metadata mismatch: {team_id}")
        for field in maxima:
            difference = abs(float(getattr(left, field)) - float(getattr(right, field)))
            maxima[field] = max(maxima[field], difference)
    if max(maxima.values()) > BASELINE_TOLERANCE:
        raise ValueError(f"Production replay exceeds tolerance: {maxima}")
    return maxima


def latest_production_predictions(
    production_ledger: Path,
    served: pd.DataFrame,
) -> tuple[dict[str, LedgerEntry], dict[str, object]]:
    report = verify_ledger(production_ledger)
    if not report["valid"]:
        raise ValueError(f"Production ledger is invalid: {report['problems']}")
    latest: dict[str, LedgerEntry] = {}
    for entry in read_entries(production_ledger):
        if entry.kind == PREDICTION:
            latest[str(entry.payload["match_id"])] = entry
    served_ids = served["match_id"].astype(str)
    if served_ids.duplicated().any():
        raise ValueError("served_predictions.csv contains duplicate match_id values")
    if set(latest) != set(served_ids):
        missing = sorted(set(served_ids) - set(latest))
        extra = sorted(set(latest) - set(served_ids))
        raise ValueError(
            "Latest production ledger fixture universe differs from served CSV: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    served_by_id = served.set_index(served_ids, drop=False)
    for match_id, entry in latest.items():
        row = served_by_id.loc[match_id]
        for column in served.columns:
            if not semantically_equal(row[column], entry.payload.get(column)):
                raise ValueError(
                    f"Latest production ledger differs from served CSV for "
                    f"{match_id}.{column}: {entry.payload.get(column)!r} vs "
                    f"{row[column]!r}"
                )
    return latest, report


def semantically_equal(left: object, right: object) -> bool:
    if pd.isna(left) and right is None:
        return True
    if left is None and right is None:
        return True
    if isinstance(left, bool) or isinstance(right, bool):
        return bool(left) is bool(right)
    try:
        left_number = float(left)
        right_number = float(right)
    except (TypeError, ValueError):
        return str(left) == str(right)
    return math.isfinite(left_number) and math.isfinite(right_number) and math.isclose(
        left_number,
        right_number,
        rel_tol=0.0,
        abs_tol=1e-15,
    )


def lock_all(
    state: SeasonState,
    fixtures: Sequence[MatchFixture],
    config: object,
    generated_at: datetime,
) -> dict[str, LockedPrediction]:
    locked: dict[str, LockedPrediction] = {}
    for fixture in fixtures:
        prediction = lock_prediction(
            state,
            fixture,
            config,
            generated_at_utc=generated_at,
        )
        if prediction.match_id in locked:
            raise ValueError(f"Duplicate locked match_id: {prediction.match_id}")
        locked[prediction.match_id] = prediction
    return locked


def compare_baseline_probabilities(
    locked: Mapping[str, LockedPrediction], served: pd.DataFrame
) -> dict[str, float]:
    served_by_id = served.set_index(served["match_id"].astype(str), drop=False)
    if set(locked) != set(served_by_id.index):
        raise ValueError("Reproduced AO probabilities do not cover served fixture universe")
    maxima = {"home": 0.0, "draw": 0.0, "away": 0.0, "sum": 0.0, "score": 0.0}
    for match_id, prediction in locked.items():
        row = served_by_id.loc[match_id]
        values = (
            prediction.home_win_probability,
            prediction.draw_probability,
            prediction.away_win_probability,
        )
        baseline = (
            float(row["ao_home_probability"]),
            float(row["ao_draw_probability"]),
            float(row["ao_away_probability"]),
        )
        for name, actual, expected in zip(
            ("home", "draw", "away"),
            values,
            baseline,
            strict=True,
        ):
            maxima[name] = max(maxima[name], abs(actual - expected))
        maxima["sum"] = max(maxima["sum"], abs(sum(values) - 1.0))
        maxima["score"] = max(
            maxima["score"],
            abs(values[0] + 0.5 * values[1] - prediction.expected_home_score),
        )
    if max(maxima.values()) > BASELINE_TOLERANCE:
        raise ValueError(f"Production AO baseline reproduction failed: {maxima}")
    return maxima


def build_shadow_prediction_frame(
    *,
    candidate_locked: Mapping[str, LockedPrediction],
    baseline_locked: Mapping[str, LockedPrediction],
    latest_entries: Mapping[str, LedgerEntry],
    candidate_seeds: pd.DataFrame,
    source_hashes: Mapping[str, str],
    spec_sha256: str,
    production_ledger_head: str,
) -> pd.DataFrame:
    seed_lookup = candidate_seeds.set_index("team_id")
    rows: list[dict[str, object]] = []
    for match_id in sorted(
        candidate_locked,
        key=lambda value: (
            candidate_locked[value].kickoff_utc,
            value,
        ),
    ):
        shadow = candidate_locked[match_id]
        baseline = baseline_locked[match_id]
        base_entry = latest_entries[match_id]
        base = base_entry.payload
        expected_base_identity = {
            "match_id": match_id,
            "season": shadow.season,
            "kickoff_utc": utc_iso(shadow.kickoff_utc),
            "competition": shadow.competition,
            "round": shadow.round,
            "stage": shadow.stage,
            "format_type": "LEAGUE_OR_GROUP",
            "is_neutral": shadow.is_neutral,
            "home_club_id": shadow.home_team_id,
            "away_club_id": shadow.away_team_id,
        }
        actual_base_identity = {
            key: utc_iso(base[key]) if key == "kickoff_utc" else base.get(key)
            for key in expected_base_identity
        }
        if any(
            not semantically_equal(actual_base_identity[key], expected)
            for key, expected in expected_base_identity.items()
        ):
            raise ValueError(f"Production base prediction identity mismatch: {match_id}")
        home_seed = seed_lookup.loc[shadow.home_team_id]
        away_seed = seed_lookup.loc[shadow.away_team_id]
        rows.append(
            {
                "shadow_arm_id": POSITIVE_BRIDGE_ARM_ID,
                "shadow_scope": AO_CORE_SHADOW,
                "shadow_spec_sha256": spec_sha256,
                "source_production_contract_sha256": source_hashes["contract"],
                "source_fixture_sha256": source_hashes["fixtures"],
                "source_state_sha256": source_hashes["production_state"],
                "source_production_seed_sha256": source_hashes["production_seeds"],
                "source_completed_matches_sha256": source_hashes["completed_matches"],
                "source_served_predictions_sha256": source_hashes["served_predictions"],
                "source_production_ledger_sha256": source_hashes["production_ledger"],
                "source_production_ledger_head": production_ledger_head,
                "base_prediction_entry_hash": base_entry.entry_hash,
                "base_prediction_ledger_revision": int(base["ledger_revision"]),
                "base_prediction_generated_at_utc": utc_iso(base["generated_at_utc"]),
                "prediction_model_version": SHADOW_MODEL_VERSION,
                "dynamic_model_version": shadow.model_version,
                "dynamic_config_id": shadow.config_id,
                "match_id": match_id,
                "season": shadow.season,
                "kickoff_utc": utc_iso(shadow.kickoff_utc),
                "generated_at_utc": utc_iso(shadow.generated_at_utc),
                "competition": shadow.competition,
                "round": shadow.round,
                "stage": shadow.stage,
                "format_type": str(base["format_type"]),
                "is_neutral": shadow.is_neutral,
                "home_club_id": shadow.home_team_id,
                "away_club_id": shadow.away_team_id,
                "home_live_pre": shadow.home_live_pre,
                "away_live_pre": shadow.away_live_pre,
                "expected_home_score": shadow.expected_home_score,
                "home_probability": shadow.home_win_probability,
                "draw_probability": shadow.draw_probability,
                "away_probability": shadow.away_win_probability,
                "rating_feedback_applied": False,
                "state_last_event_utc": utc_iso(shadow.state_last_event_utc),
                "state_last_match_id": shadow.state_last_match_id,
                "baseline_home_live_pre": baseline.home_live_pre,
                "baseline_away_live_pre": baseline.away_live_pre,
                "baseline_expected_home_score": baseline.expected_home_score,
                "baseline_home_probability": baseline.home_win_probability,
                "baseline_draw_probability": baseline.draw_probability,
                "baseline_away_probability": baseline.away_win_probability,
                "home_probability_delta": (
                    shadow.home_win_probability - baseline.home_win_probability
                ),
                "draw_probability_delta": (
                    shadow.draw_probability - baseline.draw_probability
                ),
                "away_probability_delta": (
                    shadow.away_win_probability - baseline.away_win_probability
                ),
                "home_production_ao_first_elo": float(
                    home_seed["production_ao_first_elo"]
                ),
                "home_shadow_ao_first_elo": float(home_seed["ao_first_elo"]),
                "home_positive_bridge_applied": bool(
                    home_seed["positive_bridge_applied"]
                ),
                "away_production_ao_first_elo": float(
                    away_seed["production_ao_first_elo"]
                ),
                "away_shadow_ao_first_elo": float(away_seed["ao_first_elo"]),
                "away_positive_bridge_applied": bool(
                    away_seed["positive_bridge_applied"]
                ),
            }
        )
    frame = pd.DataFrame(rows)
    if frame["match_id"].duplicated().any() or len(frame) != 396:
        raise ValueError("Shadow prediction frame must contain 396 unique fixtures")
    probability_error = (
        frame["home_probability"]
        + frame["draw_probability"]
        + frame["away_probability"]
        - 1.0
    ).abs().max()
    score_error = (
        frame["home_probability"]
        + 0.5 * frame["draw_probability"]
        - frame["expected_home_score"]
    ).abs().max()
    if max(float(probability_error), float(score_error)) > 1e-12:
        raise ValueError("Shadow probability invariants failed before artifact write")
    return frame


def artifact_file_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    }


def fsync_tree(root: Path) -> None:
    """Make every staged artifact durable before publishing its directory."""

    files = [path for path in root.rglob("*") if path.is_file()]
    for path in files:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    directories = [root, *(path for path in root.rglob("*") if path.is_dir())]
    for path in sorted(directories, key=lambda value: len(value.parts), reverse=True):
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def verify_artifact_directory(root: Path, expected_manifest_sha256: str) -> None:
    manifest_path = root / "artifact_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Candidate artifact manifest is missing after publish")
    actual_manifest_sha = sha256_file(manifest_path)
    if actual_manifest_sha != expected_manifest_sha256:
        raise ValueError(
            "Candidate artifact manifest SHA changed after publish: "
            f"{actual_manifest_sha} != {expected_manifest_sha256}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_files = manifest.get("artifact_files")
    if not isinstance(expected_files, dict):
        raise ValueError("Candidate artifact manifest lacks artifact_files")
    actual_files = artifact_file_hashes(root)
    if actual_files != expected_files:
        raise ValueError("Candidate artifact files do not match their manifest")


def current_runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": importlib.metadata.version("numpy"),
    }


def read_published_artifact_predictions(
    artifact_root: Path,
    manifest_sha256: str,
) -> pd.DataFrame:
    """Read the hashed CSV representation that must become the ledger payload."""

    predictions_path = artifact_root / ARTIFACT_PREDICTIONS_FILENAME
    if not predictions_path.is_file():
        raise ValueError("Candidate pre-ledger prediction artifact is missing")
    frame = pd.read_csv(predictions_path)
    if frame.columns.duplicated().any():
        raise ValueError("Candidate pre-ledger prediction artifact has duplicate columns")
    if "candidate_artifact_manifest_sha256" in frame.columns:
        raise ValueError(
            "Pre-ledger artifact cannot contain its circular manifest SHA field"
        )
    frame["candidate_artifact_manifest_sha256"] = manifest_sha256
    missing = sorted(set(REQUIRED_SHADOW_PREDICTION_FIELDS).difference(frame.columns))
    if missing:
        raise ValueError(
            f"Candidate pre-ledger prediction artifact is missing fields: {missing}"
        )
    return frame


def load_and_validate_resume_artifact(
    args: argparse.Namespace,
    *,
    source_hashes: Mapping[str, str],
    implementation_hashes: Mapping[str, str],
    spec_sha256: str,
    production_revision: str,
    production_ledger_head: str,
    dynamic_model_version: str,
    dynamic_config_id: str,
    fixtures: Sequence[MatchFixture],
    latest_entries: Mapping[str, LedgerEntry],
) -> tuple[dict[str, object], str, pd.DataFrame]:
    """Load a published pre-ledger artifact and bind it to canonical inputs."""

    if not args.artifact_root.is_dir():
        raise ValueError(
            f"Candidate artifact root is missing or not a directory: "
            f"{args.artifact_root}"
        )
    manifest_path = args.artifact_root / "artifact_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Candidate artifact manifest is missing for resume")
    manifest_sha = sha256_file(manifest_path)
    verify_artifact_directory(args.artifact_root, manifest_sha)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Candidate artifact manifest must be a JSON object")

    expected_identity = {
        "schema_version": 1,
        "artifact_type": "AO_CORE_SHADOW_START",
        "arm_id": POSITIVE_BRIDGE_ARM_ID,
        "scope": AO_CORE_SHADOW,
        "season": SEASON,
        "spec_sha256": spec_sha256,
        "production_revision": production_revision,
        "source_hashes": dict(source_hashes),
        "implementation_hashes": dict(implementation_hashes),
        "runtime_versions": current_runtime_versions(),
        "source_production_ledger_head": production_ledger_head,
        "dynamic_model_version": dynamic_model_version,
        "dynamic_config_id": dynamic_config_id,
    }
    mismatches = {
        key: (manifest.get(key), expected)
        for key, expected in expected_identity.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "Candidate artifact does not match canonical source/spec/runtime "
            f"identity: {mismatches}"
        )

    fixture_summary = validate_fixture_universe(fixtures)
    if manifest.get("fixture_universe") != fixture_summary:
        raise ValueError(
            "Candidate artifact fixture summary does not match canonical fixtures"
        )
    candidate_summary = manifest.get("candidate")
    if not isinstance(candidate_summary, dict):
        raise ValueError("Candidate artifact manifest lacks candidate summary")
    if (
        candidate_summary.get("prediction_rows") != 396
        or candidate_summary.get("completed_matches_replayed") != 428
    ):
        raise ValueError("Candidate artifact manifest has incomplete replay counts")
    if not isinstance(manifest.get("baseline_reproduction"), dict):
        raise ValueError("Candidate artifact manifest lacks baseline reproduction")

    generated_at = pd.Timestamp(manifest.get("generated_at_utc"))
    if generated_at.tzinfo is None:
        raise ValueError("Candidate artifact generated_at_utc must be timezone-aware")
    generated_at = generated_at.tz_convert("UTC")
    earliest_kickoff = pd.Timestamp(fixture_summary["earliest_kickoff_utc"])
    if generated_at >= earliest_kickoff:
        raise ValueError("Candidate artifact was not generated before earliest kickoff")

    frame = read_published_artifact_predictions(args.artifact_root, manifest_sha)
    match_ids = frame["match_id"].astype(str)
    if len(frame) != 396 or match_ids.duplicated().any():
        raise ValueError("Candidate pre-ledger artifact must have 396 unique matches")

    fixtures_by_id = {fixture.match_id: fixture for fixture in fixtures}
    if len(fixtures_by_id) != len(fixtures):
        raise ValueError("Canonical fixture input contains duplicate match_id values")
    if set(match_ids) != set(fixtures_by_id) or set(match_ids) != set(latest_entries):
        raise ValueError(
            "Candidate artifact match universe does not match canonical fixtures "
            "and production base predictions"
        )

    common_expected = {
        "shadow_arm_id": POSITIVE_BRIDGE_ARM_ID,
        "shadow_scope": AO_CORE_SHADOW,
        "shadow_spec_sha256": spec_sha256,
        "candidate_artifact_manifest_sha256": manifest_sha,
        "source_production_contract_sha256": source_hashes["contract"],
        "source_fixture_sha256": source_hashes["fixtures"],
        "source_state_sha256": source_hashes["production_state"],
        "source_production_seed_sha256": source_hashes["production_seeds"],
        "source_completed_matches_sha256": source_hashes["completed_matches"],
        "source_served_predictions_sha256": source_hashes["served_predictions"],
        "source_production_ledger_sha256": source_hashes["production_ledger"],
        "source_production_ledger_head": production_ledger_head,
        "prediction_model_version": SHADOW_MODEL_VERSION,
        "dynamic_model_version": dynamic_model_version,
        "dynamic_config_id": dynamic_config_id,
        "season": SEASON,
        "generated_at_utc": generated_at.isoformat(),
        "rating_feedback_applied": False,
    }
    for position, row in frame.iterrows():
        match_id = str(row["match_id"])
        fixture = fixtures_by_id[match_id]
        base_entry = latest_entries[match_id]
        base = base_entry.payload
        expected = {
            **common_expected,
            "match_id": match_id,
            "kickoff_utc": utc_iso(fixture.kickoff_utc),
            "competition": fixture.competition,
            "round": fixture.round,
            "stage": fixture.stage,
            "is_neutral": fixture.is_neutral,
            "home_club_id": fixture.home_team_id,
            "away_club_id": fixture.away_team_id,
            "format_type": base.get("format_type"),
            "base_prediction_entry_hash": base_entry.entry_hash,
            "base_prediction_ledger_revision": base.get("ledger_revision"),
            "base_prediction_generated_at_utc": utc_iso(base["generated_at_utc"]),
        }
        for field, expected_value in expected.items():
            if field not in row or not semantically_equal(row[field], expected_value):
                raise ValueError(
                    "Candidate artifact failed canonical/base-entry membership "
                    f"check at row {position} ({match_id}).{field}: "
                    f"{row.get(field)!r} != {expected_value!r}"
                )

        base_identity = {
            "match_id": match_id,
            "season": row["season"],
            "kickoff_utc": row["kickoff_utc"],
            "competition": row["competition"],
            "round": row["round"],
            "stage": row["stage"],
            "format_type": row["format_type"],
            "is_neutral": row["is_neutral"],
            "home_club_id": row["home_club_id"],
            "away_club_id": row["away_club_id"],
        }
        for field, expected_value in base_identity.items():
            actual = base.get(field)
            if field == "kickoff_utc":
                actual = utc_iso(actual)
            if not semantically_equal(actual, expected_value):
                raise ValueError(
                    f"Production base-entry identity mismatch during resume: "
                    f"{match_id}.{field}"
                )
    return manifest, manifest_sha, frame


def verify_shadow_ledger_matches_artifact(
    ledger_path: Path,
    artifact_rows: pd.DataFrame,
) -> tuple[dict[str, object], str]:
    """Verify a complete existing ledger without attempting a duplicate append."""

    report = verify_shadow_ledger(
        ledger_path,
        expected_arm_id=POSITIVE_BRIDGE_ARM_ID,
        expected_scope=AO_CORE_SHADOW,
    )
    if not report["valid"]:
        raise ValueError(f"Existing shadow ledger is invalid: {report['problems']}")
    entries = read_shadow_entries(ledger_path)
    if len(entries) != 396 or len(entries) != len(artifact_rows):
        raise ValueError(
            "Resume requires an empty ledger or the complete 396-entry atomic batch"
        )
    expected_payloads = [
        normalize_shadow_prediction_payload(row)
        for row in artifact_rows.to_dict("records")
    ]
    for position, (entry, expected) in enumerate(
        zip(entries, expected_payloads, strict=True)
    ):
        if canonical_json_bytes(entry.payload) != canonical_json_bytes(expected):
            differing = next(
                (
                    key
                    for key in sorted(set(entry.payload) | set(expected))
                    if canonical_json_bytes(entry.payload.get(key))
                    != canonical_json_bytes(expected.get(key))
                ),
                "<unknown>",
            )
            raise ValueError(
                "Existing shadow ledger payload differs from the published "
                f"artifact at entry {position}.{differing}"
            )
    recorded_values = {entry.recorded_at_utc for entry in entries}
    if len(recorded_values) != 1:
        raise ValueError("Shadow start ledger must be one atomic recorded-at batch")
    normalized_report = dict(report)
    normalized_report["appended"] = len(entries)
    normalized_report["kind"] = SHADOW_PREDICTION
    return normalized_report, entries[0].recorded_at_utc


def recover_or_verify_shadow_ledger(
    ledger_path: Path,
    artifact_rows: pd.DataFrame,
    *,
    production_ledger_path: Path,
) -> tuple[dict[str, object], str]:
    """Append the missing atomic batch, or verify the already complete batch."""

    ledger_has_entries = ledger_path.is_file() and ledger_path.stat().st_size > 0
    if not ledger_has_entries:
        append_shadow_predictions(
            ledger_path,
            artifact_rows,
            recorded_at=datetime.now(timezone.utc),
            expected_arm_id=POSITIVE_BRIDGE_ARM_ID,
            production_ledger_path=production_ledger_path,
            expected_scope=AO_CORE_SHADOW,
        )
    return verify_shadow_ledger_matches_artifact(ledger_path, artifact_rows)


def atomic_write_or_verify(path: Path, content: bytes, *, label: str) -> bool:
    """Publish a missing recovery output, or require an exact existing copy."""

    if path.exists():
        if not path.is_file():
            raise ValueError(f"{label} exists but is not a file: {path}")
        if path.read_bytes() != content:
            raise ValueError(f"Existing {label} differs from recovered content: {path}")
        return False
    atomic_write(path, content)
    return True


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def build_report(summary: Mapping[str, object]) -> str:
    counts = summary["fixture_universe"]["competition_counts"]
    baseline = summary["baseline_reproduction"]
    return f"""# POSITIVE_BRIDGE_020 — 2026/27 AO-core Shadow Start

Bu koşu production modeli değiştirmedi. Frozen Positive Bridge seed'i ayrı bir
state üzerinde 428 tamamlanmış qualifier maçıyla replay edildi ve sonuçlar
görülmeden 396 lig aşaması fikstürü için AO-core tahmini ayrı hash zincirine
yazıldı.

## Başlangıç

- Generated at UTC: `{summary['generated_at_utc']}`
- Recorded at UTC: `{summary['recorded_at_utc']}`
- Fikstür: **{summary['fixture_universe']['fixtures']}**
- Kupa dağılımı: UCL **{counts['UCL']}**, UEL **{counts['UEL']}**, UECL **{counts['UECL']}**
- İlk kickoff: `{summary['fixture_universe']['earliest_kickoff_utc']}`
- Son kickoff: `{summary['fixture_universe']['latest_kickoff_utc']}`
- Değişen seed: **{summary['candidate']['affected_seed_rows']} / {summary['candidate']['seed_rows']}**
- Fikstür evreninde değişen takım: **{summary['candidate']['affected_fixture_teams']}**
- AO olasılığı değişen maç: **{summary['candidate']['changed_prediction_rows']} / 396**

## Baseline kapısı

- Replay rating maksimum farkı: `{baseline['maximum_state_difference']:.3e}`
- AO home maksimum farkı: `{baseline['probability_maxima']['home']:.3e}`
- AO draw maksimum farkı: `{baseline['probability_maxima']['draw']:.3e}`
- AO away maksimum farkı: `{baseline['probability_maxima']['away']:.3e}`
- Olasılık toplamı maksimum hatası: `{baseline['probability_maxima']['sum']:.3e}`
- Expected-score maksimum hatası: `{baseline['probability_maxima']['score']:.3e}`

## Ayrı kanıt zinciri

- Scope: `AO_CORE_SHADOW`
- Ledger satırı: **{summary['shadow_ledger']['entries']}**
- Ledger head: `{summary['shadow_ledger']['head_hash']}`
- Spec SHA-256: `{summary['spec_sha256']}`
- Candidate artifact manifest SHA-256: `{summary['candidate_artifact_manifest_sha256']}`
- Production ledger head (eşleştirme anı): `{summary['production_ledger']['head_hash']}`

## Ölçüm sınırı

Bu tahminler yalnız AO rating çekirdeğini karşılaştırır. Candidate-specific
Structural Logistic, Domestic Poisson ve served 0.50/0.50 ensemble bu kolda
çalıştırılmadı. Henüz maç sonucu olmadığı için prospective Brier, log-loss veya
accuracy ölçümü yoktur. Yerel anchor dosyası zincir kimliğini korur fakat
commit/push veya bağımsız immutable bir depoya yayınlanmadığı sürece dış zaman
damgası kanıtı değildir.
"""


def resume_shadow_start(
    args: argparse.Namespace,
    *,
    source_hashes: Mapping[str, str],
    implementation_hashes: Mapping[str, str],
    spec_sha256: str,
    production_revision: str,
    production_ledger_head: str,
    production_ledger_report: Mapping[str, object],
    dynamic_model_version: str,
    dynamic_config_id: str,
    fixtures: Sequence[MatchFixture],
    latest_entries: Mapping[str, LedgerEntry],
) -> None:
    """Finish ledger/anchor/report publication from an immutable artifact."""

    manifest, manifest_sha, shadow_frame = load_and_validate_resume_artifact(
        args,
        source_hashes=source_hashes,
        implementation_hashes=implementation_hashes,
        spec_sha256=spec_sha256,
        production_revision=production_revision,
        production_ledger_head=production_ledger_head,
        dynamic_model_version=dynamic_model_version,
        dynamic_config_id=dynamic_config_id,
        fixtures=fixtures,
        latest_entries=latest_entries,
    )
    if args.shadow_ledger.exists() and not args.shadow_ledger.is_file():
        raise ValueError(
            f"Shadow ledger exists but is not a file: {args.shadow_ledger}"
        )
    if args.shadow_anchor.exists() and not args.shadow_anchor.is_file():
        raise ValueError(
            f"Shadow anchor exists but is not a file: {args.shadow_anchor}"
        )
    if args.run_output.exists() and not args.run_output.is_dir():
        raise ValueError(f"Run output exists but is not a directory: {args.run_output}")

    current_hashes = capture_production_fingerprints(args)
    if current_hashes != dict(source_hashes):
        raise RuntimeError("Canonical production sources changed before shadow resume")
    if ledger_head(args.production_ledger).entry_hash != production_ledger_head:
        raise RuntimeError("Production ledger head changed before shadow resume")
    if implementation_fingerprints() != dict(implementation_hashes):
        raise RuntimeError("Shadow implementation changed during resume validation")

    ledger_has_entries = (
        args.shadow_ledger.is_file() and args.shadow_ledger.stat().st_size > 0
    )
    if not ledger_has_entries:
        downstream = (
            args.shadow_anchor,
            args.run_output / "run_summary.json",
            args.run_output / "REPORT.md",
        )
        existing_downstream = [str(path) for path in downstream if path.exists()]
        if existing_downstream:
            raise ValueError(
                "Recovery outputs exist without the atomic shadow ledger batch: "
                f"{existing_downstream}"
            )
    ledger_report, recorded_at_utc = recover_or_verify_shadow_ledger(
        args.shadow_ledger,
        shadow_frame,
        production_ledger_path=args.production_ledger,
    )
    shadow_head = shadow_ledger_head(args.shadow_ledger)

    final_hashes = capture_production_fingerprints(args)
    if final_hashes != dict(source_hashes):
        raise RuntimeError("Canonical production sources changed during shadow resume")
    if ledger_head(args.production_ledger).entry_hash != production_ledger_head:
        raise RuntimeError("Production ledger head changed during shadow resume")

    fixture_summary = manifest["fixture_universe"]
    anchor = {
        "schema_version": 1,
        "anchor_status": "LOCAL_ONLY_NOT_EXTERNALLY_TIMESTAMPED",
        "arm_id": POSITIVE_BRIDGE_ARM_ID,
        "scope": AO_CORE_SHADOW,
        "season": SEASON,
        "created_at_utc": recorded_at_utc,
        "ledger_path": display_path(args.shadow_ledger),
        "ledger_sha256": sha256_file(args.shadow_ledger),
        "ledger_entries": shadow_head.entries,
        "ledger_head_hash": shadow_head.entry_hash,
        "shadow_spec_sha256": spec_sha256,
        "candidate_artifact_manifest_sha256": manifest_sha,
        "source_production_ledger_head": production_ledger_head,
        "earliest_kickoff_utc": fixture_summary["earliest_kickoff_utc"],
        "latest_kickoff_utc": fixture_summary["latest_kickoff_utc"],
    }
    atomic_write_or_verify(
        args.shadow_anchor,
        canonical_json_bytes(anchor),
        label="shadow anchor",
    )

    summary = {
        "arm_id": POSITIVE_BRIDGE_ARM_ID,
        "scope": AO_CORE_SHADOW,
        "production_changed": False,
        "generated_at_utc": utc_iso(manifest["generated_at_utc"]),
        "recorded_at_utc": recorded_at_utc,
        "spec_sha256": spec_sha256,
        "candidate_artifact_manifest_sha256": manifest_sha,
        "fixture_universe": fixture_summary,
        "candidate": manifest["candidate"],
        "baseline_reproduction": manifest["baseline_reproduction"],
        "production_fingerprints": dict(source_hashes),
        "production_ledger": dict(production_ledger_report),
        "shadow_ledger": ledger_report,
        "shadow_anchor": anchor,
        "measurement_limit": (
            "AO_CORE_ONLY; candidate-specific ML, Domestic Poisson and served "
            "ensemble were not run; no outcomes exist yet"
        ),
    }
    args.run_output.mkdir(parents=True, exist_ok=True)
    atomic_write_or_verify(
        args.run_output / "run_summary.json",
        canonical_json_bytes(summary),
        label="run summary",
    )
    atomic_write_or_verify(
        args.run_output / "REPORT.md",
        build_report(summary).encode("utf-8"),
        label="run report",
    )

    postwrite_hashes = capture_production_fingerprints(args)
    if postwrite_hashes != dict(source_hashes):
        raise RuntimeError("Canonical production sources changed after shadow resume")
    if ledger_head(args.production_ledger).entry_hash != production_ledger_head:
        raise RuntimeError("Production ledger head changed after shadow resume")
    verify_shadow_ledger_matches_artifact(args.shadow_ledger, shadow_frame)
    verify_artifact_directory(args.artifact_root, manifest_sha)

    print("POSITIVE_BRIDGE_020 AO-core shadow resume completed")
    print(f"Predictions: {shadow_head.predictions}")
    print(f"Changed AO rows: {manifest['candidate']['changed_prediction_rows']}")
    print(f"Shadow ledger head: {shadow_head.entry_hash}")
    print(f"Production state SHA-256: {source_hashes['production_state']}")
    print(f"Report: {args.run_output / 'REPORT.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate, start, or recover the isolated Positive Bridge AO-core shadow"
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--start",
        action="store_true",
        help="publish artifacts and append the shadow ledger",
    )
    mode.add_argument(
        "--resume",
        action="store_true",
        help="verify a published artifact and finish missing ledger/report steps",
    )
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--production-state", type=Path, default=DEFAULT_PRODUCTION_STATE)
    parser.add_argument("--production-seeds", type=Path, default=DEFAULT_PRODUCTION_SEEDS)
    parser.add_argument("--completed-matches", type=Path, default=DEFAULT_COMPLETED_MATCHES)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--production-checkpoint", type=Path, default=DEFAULT_PRODUCTION_CHECKPOINT)
    parser.add_argument("--served-predictions", type=Path, default=DEFAULT_SERVED_PREDICTIONS)
    parser.add_argument("--production-ledger", type=Path, default=DEFAULT_PRODUCTION_LEDGER)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--shadow-ledger", type=Path, default=DEFAULT_SHADOW_LEDGER)
    parser.add_argument("--shadow-anchor", type=Path, default=DEFAULT_SHADOW_ANCHOR)
    parser.add_argument("--run-output", type=Path, default=DEFAULT_RUN_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for field in vars(args):
        value = getattr(args, field)
        if isinstance(value, Path):
            setattr(args, field, value.expanduser().resolve())

    validate_output_isolation(args)
    if args.start:
        if args.artifact_root.exists():
            raise ValueError(f"Candidate artifact root already exists: {args.artifact_root}")
        if args.shadow_ledger.exists() and args.shadow_ledger.stat().st_size:
            raise ValueError(f"Shadow ledger already started: {args.shadow_ledger}")
        if args.shadow_anchor.exists():
            raise ValueError(f"Shadow anchor already exists: {args.shadow_anchor}")
        if args.run_output.exists() and not args.run_output.is_dir():
            raise ValueError(f"Run output exists but is not a directory: {args.run_output}")
        if args.shadow_ledger.exists() and not args.shadow_ledger.is_file():
            raise ValueError(f"Shadow ledger exists but is not a file: {args.shadow_ledger}")

    source_hashes = capture_production_fingerprints(args)
    implementation_hashes = implementation_fingerprints()
    initial_production_report = verify_ledger(args.production_ledger)
    if not initial_production_report["valid"]:
        raise ValueError(
            f"Production ledger is invalid: {initial_production_report['problems']}"
        )
    initial_production_head = ledger_head(args.production_ledger)
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    production_revision = str(contract["production_revision"])
    _, spec_sha = validate_executable_spec(
        args.spec,
        contract_sha256=source_hashes["contract"],
        production_state_sha256=source_hashes["production_state"],
        production_revision=production_revision,
        source_hashes=source_hashes,
        production_ledger_head=initial_production_head.entry_hash,
        production_ledger_entries=initial_production_head.entries,
        implementation_hashes=implementation_hashes,
    )
    if args.resume:
        config = load_selected_v2_config(args.contract)
        fixtures = read_fixtures(args.fixtures)
        validate_fixture_universe(fixtures)
        served = pd.read_csv(args.served_predictions)
        latest_entries, production_ledger_report = latest_production_predictions(
            args.production_ledger,
            served,
        )
        resume_shadow_start(
            args,
            source_hashes=source_hashes,
            implementation_hashes=implementation_hashes,
            spec_sha256=spec_sha,
            production_revision=production_revision,
            production_ledger_head=initial_production_head.entry_hash,
            production_ledger_report=production_ledger_report,
            dynamic_model_version=config.model_version,
            dynamic_config_id=config.config_id,
            fixtures=fixtures,
            latest_entries=latest_entries,
        )
        return
    config = load_selected_v2_config(args.contract)
    season, production_seeds_tuple = read_team_seeds(args.production_seeds)
    if season != SEASON:
        raise ValueError(f"Production seed season must be {SEASON}, got {season}")
    completed_matches = read_matches(args.completed_matches)
    fixtures = read_fixtures(args.fixtures)
    fixture_summary = validate_fixture_universe(fixtures)
    served = pd.read_csv(args.served_predictions)
    latest_entries, production_ledger_report = latest_production_predictions(
        args.production_ledger,
        served,
    )

    baseline_initial = initialize_season(SEASON, production_seeds_tuple, config)
    baseline_state, baseline_updates = run_season(
        baseline_initial,
        completed_matches,
        config,
    )
    if len(baseline_updates) != 428:
        raise ValueError(f"Expected 428 completed matches, found {len(baseline_updates)}")
    checkpoint = load_state_checkpoint(args.production_checkpoint, config)
    state_maxima = compare_states(baseline_state, checkpoint)

    production_seed_frame = pd.read_csv(args.production_seeds)
    candidate_seed_frame = apply_positive_bridge_seed_transform(production_seed_frame)
    if "ao_first_elo_rank" in candidate_seed_frame:
        candidate_seed_frame["production_ao_first_elo_rank"] = candidate_seed_frame[
            "ao_first_elo_rank"
        ]
        candidate_seed_frame["ao_first_elo_rank"] = (
            candidate_seed_frame["ao_first_elo"]
            .rank(method="min", ascending=False)
            .astype(int)
        )

    staging_parent: Path | None = None
    if args.start:
        args.artifact_root.parent.mkdir(parents=True, exist_ok=True)
        staging_parent = args.artifact_root.parent
    with tempfile.TemporaryDirectory(
        dir=staging_parent,
        prefix=f".{args.artifact_root.name}.",
    ) as staging_name:
        staging = Path(staging_name)
        candidate_seed_path = staging / "positive_bridge_020_ao_first_elo_2026_27.csv"
        candidate_seed_frame.to_csv(candidate_seed_path, index=False, lineterminator="\n")
        candidate_state, candidate_updates = run_batch(
            candidate_seed_path,
            args.completed_matches,
            staging / "qualifier_replay",
            config,
        )
        if len(candidate_updates) != 428:
            raise ValueError("Candidate replay did not process all 428 completed matches")

        generated_at = datetime.now(timezone.utc)
        earliest_kickoff = min(fixture.kickoff_utc for fixture in fixtures)
        if generated_at >= earliest_kickoff:
            raise ValueError(
                f"Shadow generation time {generated_at.isoformat()} is not before "
                f"earliest kickoff {earliest_kickoff.isoformat()}"
            )
        baseline_locked = lock_all(baseline_state, fixtures, config, generated_at)
        probability_maxima = compare_baseline_probabilities(baseline_locked, served)
        candidate_locked = lock_all(candidate_state, fixtures, config, generated_at)
        production_head = ledger_head(args.production_ledger)
        shadow_frame = build_shadow_prediction_frame(
            candidate_locked=candidate_locked,
            baseline_locked=baseline_locked,
            latest_entries=latest_entries,
            candidate_seeds=candidate_seed_frame,
            source_hashes=source_hashes,
            spec_sha256=spec_sha,
            production_ledger_head=production_head.entry_hash,
        )
        artifact_predictions = shadow_frame.copy()
        artifact_predictions.to_csv(
            staging / ARTIFACT_PREDICTIONS_FILENAME,
            index=False,
            lineterminator="\n",
        )

        fixture_teams = set(
            pd.concat(
                [
                    pd.Series([fixture.home_team_id for fixture in fixtures]),
                    pd.Series([fixture.away_team_id for fixture in fixtures]),
                ],
                ignore_index=True,
            )
        )
        affected = candidate_seed_frame["positive_bridge_applied"].astype(bool)
        affected_fixture_teams = int(
            candidate_seed_frame.loc[
                affected & candidate_seed_frame["team_id"].isin(fixture_teams),
                "team_id",
            ].nunique()
        )
        changed_rows = int(
            shadow_frame[
                [
                    "home_probability_delta",
                    "draw_probability_delta",
                    "away_probability_delta",
                ]
            ]
            .abs()
            .max(axis=1)
            .gt(1e-15)
            .sum()
        )

        manifest: dict[str, object] = {
            "schema_version": 1,
            "artifact_type": "AO_CORE_SHADOW_START",
            "arm_id": POSITIVE_BRIDGE_ARM_ID,
            "scope": AO_CORE_SHADOW,
            "season": SEASON,
            "generated_at_utc": generated_at.isoformat(),
            "spec_sha256": spec_sha,
            "production_revision": production_revision,
            "source_hashes": dict(source_hashes),
            "implementation_hashes": implementation_hashes,
            "runtime_versions": current_runtime_versions(),
            "source_production_ledger_head": production_head.entry_hash,
            "dynamic_model_version": config.model_version,
            "dynamic_config_id": config.config_id,
            "candidate": {
                "seed_rows": len(candidate_seed_frame),
                "affected_seed_rows": int(affected.sum()),
                "affected_fixture_teams": affected_fixture_teams,
                "completed_matches_replayed": len(candidate_updates),
                "prediction_rows": len(shadow_frame),
                "changed_prediction_rows": changed_rows,
                "minimum_elo_delta": float(
                    candidate_seed_frame["positive_bridge_elo_delta"].min()
                ),
                "maximum_elo_delta": float(
                    candidate_seed_frame["positive_bridge_elo_delta"].max()
                ),
            },
            "fixture_universe": fixture_summary,
            "baseline_reproduction": {
                "state_maxima": state_maxima,
                "maximum_state_difference": max(state_maxima.values()),
                "probability_maxima": probability_maxima,
            },
        }
        manifest["artifact_files"] = artifact_file_hashes(staging)
        manifest_bytes = canonical_json_bytes(manifest)
        (staging / "artifact_manifest.json").write_bytes(manifest_bytes)
        manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        shadow_frame["candidate_artifact_manifest_sha256"] = manifest_sha

        if not args.start:
            print("Validation-only run passed; no artifact or ledger was published.")
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return

        prepublish_hashes = capture_production_fingerprints(args)
        if prepublish_hashes != source_hashes:
            changed = {
                key: (source_hashes[key], prepublish_hashes[key])
                for key in source_hashes
                if source_hashes[key] != prepublish_hashes[key]
            }
            raise RuntimeError(
                f"Production sources changed before shadow publish: {changed}"
            )
        prepublish_head = ledger_head(args.production_ledger)
        if prepublish_head != production_head:
            raise RuntimeError("Production ledger head changed before shadow publish")
        if implementation_fingerprints() != implementation_hashes:
            raise RuntimeError("Shadow implementation changed during generation")

        fsync_tree(staging)
        os.replace(staging, args.artifact_root)
        artifact_directory = os.open(args.artifact_root.parent, os.O_RDONLY)
        try:
            os.fsync(artifact_directory)
        finally:
            os.close(artifact_directory)
        verify_artifact_directory(args.artifact_root, manifest_sha)
        shadow_frame = read_published_artifact_predictions(
            args.artifact_root,
            manifest_sha,
        )

    recorded_at = datetime.now(timezone.utc)
    ledger_report = append_shadow_predictions(
        args.shadow_ledger,
        shadow_frame,
        recorded_at=recorded_at,
        expected_arm_id=POSITIVE_BRIDGE_ARM_ID,
        production_ledger_path=args.production_ledger,
        expected_scope=AO_CORE_SHADOW,
    )
    verified_shadow = verify_shadow_ledger(
        args.shadow_ledger,
        expected_arm_id=POSITIVE_BRIDGE_ARM_ID,
        expected_scope=AO_CORE_SHADOW,
    )
    if not verified_shadow["valid"] or verified_shadow["predictions"] != 396:
        raise ValueError(f"Published shadow ledger failed verification: {verified_shadow}")
    shadow_head = shadow_ledger_head(args.shadow_ledger)

    final_hashes = capture_production_fingerprints(args)
    if final_hashes != source_hashes:
        changed = {
            key: (source_hashes[key], final_hashes[key])
            for key in source_hashes
            if source_hashes[key] != final_hashes[key]
        }
        raise RuntimeError(f"Production sources changed during shadow start: {changed}")
    final_production_head = ledger_head(args.production_ledger)
    if final_production_head != production_head:
        raise RuntimeError("Production ledger head changed during shadow start")

    anchor = {
        "schema_version": 1,
        "anchor_status": "LOCAL_ONLY_NOT_EXTERNALLY_TIMESTAMPED",
        "arm_id": POSITIVE_BRIDGE_ARM_ID,
        "scope": AO_CORE_SHADOW,
        "season": SEASON,
        "created_at_utc": recorded_at.isoformat(),
        "ledger_path": display_path(args.shadow_ledger),
        "ledger_sha256": sha256_file(args.shadow_ledger),
        "ledger_entries": shadow_head.entries,
        "ledger_head_hash": shadow_head.entry_hash,
        "shadow_spec_sha256": spec_sha,
        "candidate_artifact_manifest_sha256": manifest_sha,
        "source_production_ledger_head": production_head.entry_hash,
        "earliest_kickoff_utc": fixture_summary["earliest_kickoff_utc"],
        "latest_kickoff_utc": fixture_summary["latest_kickoff_utc"],
    }
    atomic_write(args.shadow_anchor, canonical_json_bytes(anchor))

    summary = {
        "arm_id": POSITIVE_BRIDGE_ARM_ID,
        "scope": AO_CORE_SHADOW,
        "production_changed": False,
        "generated_at_utc": generated_at.isoformat(),
        "recorded_at_utc": recorded_at.isoformat(),
        "spec_sha256": spec_sha,
        "candidate_artifact_manifest_sha256": manifest_sha,
        "fixture_universe": fixture_summary,
        "candidate": manifest["candidate"],
        "baseline_reproduction": manifest["baseline_reproduction"],
        "production_fingerprints": source_hashes,
        "production_ledger": production_ledger_report,
        "shadow_ledger": ledger_report,
        "shadow_anchor": anchor,
        "measurement_limit": (
            "AO_CORE_ONLY; candidate-specific ML, Domestic Poisson and served "
            "ensemble were not run; no outcomes exist yet"
        ),
    }
    args.run_output.mkdir(parents=True, exist_ok=True)
    atomic_write(args.run_output / "run_summary.json", canonical_json_bytes(summary))
    atomic_write(
        args.run_output / "REPORT.md",
        build_report(summary).encode("utf-8"),
    )

    postwrite_hashes = capture_production_fingerprints(args)
    if postwrite_hashes != source_hashes:
        changed = {
            key: (source_hashes[key], postwrite_hashes[key])
            for key in source_hashes
            if source_hashes[key] != postwrite_hashes[key]
        }
        raise RuntimeError(f"Production sources changed after shadow publish: {changed}")
    if ledger_head(args.production_ledger) != production_head:
        raise RuntimeError("Production ledger head changed after shadow publish")
    postwrite_shadow = verify_shadow_ledger(
        args.shadow_ledger,
        expected_arm_id=POSITIVE_BRIDGE_ARM_ID,
        expected_scope=AO_CORE_SHADOW,
    )
    if not postwrite_shadow["valid"] or postwrite_shadow["predictions"] != 396:
        raise RuntimeError("Shadow ledger changed after anchor/report publication")
    verify_artifact_directory(args.artifact_root, manifest_sha)

    print("POSITIVE_BRIDGE_020 AO-core shadow started")
    print(f"Predictions: {shadow_head.predictions}")
    print(f"Changed AO rows: {manifest['candidate']['changed_prediction_rows']}")
    print(f"Shadow ledger head: {shadow_head.entry_hash}")
    print(f"Production state SHA-256: {source_hashes['production_state']}")
    print(f"Report: {args.run_output / 'REPORT.md'}")


if __name__ == "__main__":
    main()
