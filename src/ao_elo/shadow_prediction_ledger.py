"""Isolated append-only ledger for prospective AO-core shadow predictions.

This module deliberately does not reuse the production prediction schema.  An
AO-core challenger has no ML or Poisson component, so representing it as a
production row with twelve null component probabilities makes an accidental
production append much harder to detect.  The smaller schema below records the
candidate rating state, its 1X2 probabilities, and the exact production entry
used for the paired comparison.

One file is one immutable ``shadow_arm_id`` + ``shadow_scope`` + spec hash.
There are no revisions in this first prospective protocol: a repeated
``match_id`` is an error.  A later operational need for revisions must receive
an explicit protocol rather than silently inheriting production semantics.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import fcntl
import pandas as pd

from ao_elo.validators import require_utc_timestamp


GENESIS_HASH = "0" * 64
SHADOW_PREDICTION = "SHADOW_PREDICTION"
AO_CORE_SHADOW = "AO_CORE_SHADOW"

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

REQUIRED_SHADOW_PREDICTION_FIELDS: tuple[str, ...] = (
    "shadow_arm_id",
    "shadow_scope",
    "shadow_spec_sha256",
    "candidate_artifact_manifest_sha256",
    "source_production_contract_sha256",
    "source_fixture_sha256",
    "source_state_sha256",
    "base_prediction_entry_hash",
    "prediction_model_version",
    "match_id",
    "season",
    "kickoff_utc",
    "generated_at_utc",
    "competition",
    "round",
    "stage",
    "format_type",
    "is_neutral",
    "home_club_id",
    "away_club_id",
    "home_live_pre",
    "away_live_pre",
    "expected_home_score",
    "home_probability",
    "draw_probability",
    "away_probability",
    "rating_feedback_applied",
)

HASH_FIELDS: tuple[str, ...] = (
    "shadow_spec_sha256",
    "candidate_artifact_manifest_sha256",
    "source_production_contract_sha256",
    "source_fixture_sha256",
    "source_state_sha256",
    "base_prediction_entry_hash",
)

FORBIDDEN_SHADOW_PREDICTION_FIELDS: frozenset[str] = frozenset(
    {
        "home_goals",
        "away_goals",
        "actual_class",
        "outcome",
        "result",
        "xg_home",
        "xg_away",
        "home_score",
        "away_score",
        "settled_outcome",
    }
)


@dataclass(frozen=True)
class ShadowLedgerEntry:
    sequence: int
    kind: str
    recorded_at_utc: str
    payload: dict[str, object]
    previous_hash: str
    entry_hash: str

    def as_dict(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "kind": self.kind,
            "recorded_at_utc": self.recorded_at_utc,
            "payload": self.payload,
            "previous_hash": self.previous_hash,
            "entry_hash": self.entry_hash,
        }


@dataclass(frozen=True)
class ShadowLedgerHead:
    sequence: int
    entry_hash: str
    entries: int
    predictions: int
    shadow_arm_id: str | None
    shadow_scope: str | None
    shadow_spec_sha256: str | None


def require_isolated_shadow_path(
    ledger_path: Path,
    *,
    production_ledger_path: Path,
) -> None:
    """Refuse the production ledger, including a symlink or relative alias."""

    candidate = Path(ledger_path).expanduser().resolve(strict=False)
    production = Path(production_ledger_path).expanduser().resolve(strict=False)
    if candidate == production:
        raise ValueError("Shadow predictions cannot use the production ledger path")


def read_shadow_entries(ledger_path: Path) -> list[ShadowLedgerEntry]:
    path = Path(ledger_path)
    if not path.exists():
        return []
    entries: list[ShadowLedgerEntry] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"Shadow ledger line {number} is blank")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Shadow ledger line {number} is not valid JSON"
            ) from exc
        required_top_level = {
            "sequence",
            "kind",
            "recorded_at_utc",
            "payload",
            "previous_hash",
            "entry_hash",
        }
        if set(record) != required_top_level:
            raise ValueError(
                f"Shadow ledger line {number} has an invalid entry schema"
            )
        try:
            entries.append(
                ShadowLedgerEntry(
                    sequence=int(record["sequence"]),
                    kind=str(record["kind"]),
                    recorded_at_utc=str(record["recorded_at_utc"]),
                    payload=dict(record["payload"]),
                    previous_hash=str(record["previous_hash"]),
                    entry_hash=str(record["entry_hash"]),
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Shadow ledger line {number} has invalid typed fields"
            ) from exc
    return entries


def verify_shadow_ledger(
    ledger_path: Path,
    *,
    expected_arm_id: str | None = None,
    expected_scope: str | None = None,
) -> dict[str, object]:
    """Verify both the hash chain and every prediction's semantics."""

    entries = read_shadow_entries(ledger_path)
    return _verification_report(
        Path(ledger_path),
        entries,
        expected_arm_id=expected_arm_id,
        expected_scope=expected_scope,
    )


def shadow_ledger_head(ledger_path: Path) -> ShadowLedgerHead:
    """Return a head only after fail-closed semantic verification."""

    entries = read_shadow_entries(ledger_path)
    report = _verification_report(Path(ledger_path), entries)
    if not report["valid"]:
        raise ValueError(f"Shadow ledger is invalid: {report['problems']}")
    first = entries[0].payload if entries else {}
    return ShadowLedgerHead(
        sequence=entries[-1].sequence if entries else -1,
        entry_hash=entries[-1].entry_hash if entries else GENESIS_HASH,
        entries=len(entries),
        predictions=len(entries),
        shadow_arm_id=str(first["shadow_arm_id"]) if first else None,
        shadow_scope=str(first["shadow_scope"]) if first else None,
        shadow_spec_sha256=str(first["shadow_spec_sha256"]) if first else None,
    )


def append_shadow_predictions(
    ledger_path: Path,
    rows: pd.DataFrame,
    *,
    recorded_at: object,
    expected_arm_id: str,
    production_ledger_path: Path,
    expected_scope: str = AO_CORE_SHADOW,
) -> dict[str, object]:
    """Append one prospective batch atomically to an isolated shadow ledger."""

    path = Path(ledger_path)
    require_isolated_shadow_path(
        path,
        production_ledger_path=Path(production_ledger_path),
    )
    if not str(expected_arm_id).strip():
        raise ValueError("expected_arm_id cannot be blank")
    if expected_scope != AO_CORE_SHADOW:
        raise ValueError(f"Unsupported shadow scope: {expected_scope!r}")
    if rows.empty:
        raise ValueError("Shadow prediction batch cannot be empty")

    recorded_ts = require_utc_timestamp(recorded_at, "recorded_at")
    recorded = recorded_ts.isoformat()
    with _exclusive_shadow_lock(path):
        existing = read_shadow_entries(path)
        existing_report = _verification_report(
            path,
            existing,
            expected_arm_id=expected_arm_id,
            expected_scope=expected_scope,
        )
        if not existing_report["valid"]:
            raise ValueError(
                f"Shadow ledger is invalid before append: {existing_report['problems']}"
            )
        if existing:
            previous_recorded = require_utc_timestamp(
                existing[-1].recorded_at_utc,
                "last shadow ledger recorded_at_utc",
            )
            if recorded_ts < previous_recorded:
                raise ValueError(
                    "Shadow ledger recorded_at_utc cannot move backwards "
                    f"({recorded} < {previous_recorded.isoformat()})"
                )

        known_ids = {str(entry.payload["match_id"]) for entry in existing}
        payloads: list[dict[str, object]] = []
        batch_ids: set[str] = set()
        for position, row in enumerate(rows.to_dict("records")):
            payload = normalize_shadow_prediction_payload(row)
            label = f"row {position} ({payload.get('match_id')})"
            _validate_shadow_prediction(
                payload,
                label,
                recorded_at=recorded_ts,
                expected_arm_id=expected_arm_id,
                expected_scope=expected_scope,
            )
            match_id = str(payload["match_id"])
            if match_id in known_ids or match_id in batch_ids:
                raise ValueError(
                    f"{label}: match_id already exists in this immutable shadow "
                    "ledger; revisions are not allowed"
                )
            batch_ids.add(match_id)
            payloads.append(payload)

        earliest_kickoff = min(
            require_utc_timestamp(payload["kickoff_utc"], "kickoff_utc")
            for payload in payloads
        )
        if _utc_now() >= earliest_kickoff:
            raise ValueError(
                "Shadow ledger publish began at or after the earliest kickoff"
            )

        if existing:
            identity = existing[0].payload
            for field in ("shadow_arm_id", "shadow_scope", "shadow_spec_sha256"):
                if any(payload[field] != identity[field] for payload in payloads):
                    raise ValueError(
                        f"Shadow ledger identity cannot change: {field} differs"
                    )

        return _atomic_append(
            path,
            payloads,
            recorded_at=recorded,
            existing=existing,
            expected_arm_id=expected_arm_id,
            expected_scope=expected_scope,
            publish_deadline=earliest_kickoff,
        )


def normalize_shadow_prediction_payload(
    row: Mapping[str, object],
) -> dict[str, object]:
    """Return the canonical payload representation stored in the ledger.

    Recovery code uses this same normalization before comparing a published
    pre-ledger artifact with an existing immutable ledger.
    """

    payload = {str(key): _normalize(value) for key, value in row.items()}
    for field in ("kickoff_utc", "generated_at_utc"):
        if field in payload:
            payload[field] = require_utc_timestamp(payload[field], field).isoformat()
    for field in (
        "home_live_pre",
        "away_live_pre",
        "expected_home_score",
        "home_probability",
        "draw_probability",
        "away_probability",
    ):
        if field in payload:
            payload[field] = _finite_float(payload[field], field)
    if "is_neutral" in payload:
        payload["is_neutral"] = _canonical_boolean(payload["is_neutral"], "is_neutral")
    if "rating_feedback_applied" in payload:
        payload["rating_feedback_applied"] = _canonical_boolean(
            payload["rating_feedback_applied"], "rating_feedback_applied"
        )
    return payload


def _validate_shadow_prediction(
    payload: Mapping[str, object],
    label: str,
    *,
    recorded_at: pd.Timestamp,
    expected_arm_id: str,
    expected_scope: str,
) -> None:
    missing = sorted(set(REQUIRED_SHADOW_PREDICTION_FIELDS).difference(payload))
    if missing:
        raise ValueError(f"{label}: shadow prediction is missing fields {missing}")
    forbidden = sorted(FORBIDDEN_SHADOW_PREDICTION_FIELDS.intersection(payload))
    if forbidden:
        raise ValueError(
            f"{label}: a pre-match shadow record cannot carry outcome fields "
            f"{forbidden}"
        )

    for field in (
        "shadow_arm_id",
        "shadow_scope",
        "prediction_model_version",
        "match_id",
        "season",
        "competition",
        "round",
        "stage",
        "format_type",
        "home_club_id",
        "away_club_id",
    ):
        if not isinstance(payload[field], str) or not str(payload[field]).strip():
            raise ValueError(f"{label}: {field} must be a non-empty string")
    if payload["shadow_arm_id"] != expected_arm_id:
        raise ValueError(
            f"{label}: shadow_arm_id must be {expected_arm_id!r}, got "
            f"{payload['shadow_arm_id']!r}"
        )
    if payload["shadow_scope"] != expected_scope:
        raise ValueError(
            f"{label}: shadow_scope must be {expected_scope!r}, got "
            f"{payload['shadow_scope']!r}"
        )
    for field in HASH_FIELDS:
        _require_sha256(payload[field], f"{label}: {field}")
    if payload["base_prediction_entry_hash"] == GENESIS_HASH:
        raise ValueError(f"{label}: base_prediction_entry_hash cannot be genesis")
    if payload["rating_feedback_applied"] is not False:
        raise ValueError(f"{label}: rating_feedback_applied must be false")

    kickoff = require_utc_timestamp(payload["kickoff_utc"], f"{label}: kickoff_utc")
    generated = require_utc_timestamp(
        payload["generated_at_utc"], f"{label}: generated_at_utc"
    )
    if generated > recorded_at:
        raise ValueError(
            f"{label}: generated_at_utc must be at or before recorded_at_utc"
        )
    if recorded_at >= kickoff:
        raise ValueError(f"{label}: shadow prediction must be recorded before kickoff")

    expected = _finite_float(payload["expected_home_score"], "expected_home_score")
    probabilities = tuple(
        _finite_float(payload[field], field)
        for field in ("home_probability", "draw_probability", "away_probability")
    )
    if not 0.0 <= expected <= 1.0:
        raise ValueError(f"{label}: expected_home_score must lie in [0,1]")
    if any(not 0.0 <= value <= 1.0 for value in probabilities):
        raise ValueError(f"{label}: shadow probabilities must lie in [0,1]")
    if abs(sum(probabilities) - 1.0) > 1e-9:
        raise ValueError(f"{label}: shadow probabilities must sum to one")
    if abs(probabilities[0] + 0.5 * probabilities[1] - expected) > 1e-9:
        raise ValueError(
            f"{label}: home_probability + 0.5*draw_probability must equal "
            "expected_home_score"
        )


def _verification_report(
    ledger_path: Path,
    entries: Sequence[ShadowLedgerEntry],
    *,
    expected_arm_id: str | None = None,
    expected_scope: str | None = None,
) -> dict[str, object]:
    problems: list[str] = []
    previous = GENESIS_HASH
    previous_recorded: pd.Timestamp | None = None
    match_ids: set[str] = set()
    identity: tuple[str, str, str] | None = None
    for index, entry in enumerate(entries):
        if entry.sequence != index:
            problems.append(f"entry {index}: sequence is {entry.sequence}")
        if entry.kind != SHADOW_PREDICTION:
            problems.append(f"entry {index}: kind is {entry.kind!r}")
        if entry.previous_hash != previous:
            problems.append(
                f"entry {index}: previous_hash does not match entry {index - 1}"
            )
        expected_hash = _entry_hash(
            entry.sequence,
            entry.kind,
            entry.recorded_at_utc,
            entry.payload,
            entry.previous_hash,
        )
        if entry.entry_hash != expected_hash:
            problems.append(f"entry {index}: content does not match its hash")

        arm = str(entry.payload.get("shadow_arm_id", ""))
        scope = str(entry.payload.get("shadow_scope", ""))
        spec = str(entry.payload.get("shadow_spec_sha256", ""))
        current_identity = (arm, scope, spec)
        if identity is None:
            identity = current_identity
        elif current_identity != identity:
            problems.append(f"entry {index}: shadow ledger identity changed")
        match_id = str(entry.payload.get("match_id", ""))
        if match_id in match_ids:
            problems.append(f"entry {index}: duplicate match_id {match_id!r}")
        match_ids.add(match_id)

        try:
            recorded_at = require_utc_timestamp(
                entry.recorded_at_utc, f"entry {index}: recorded_at_utc"
            )
            if previous_recorded is not None and recorded_at < previous_recorded:
                problems.append(
                    f"entry {index}: recorded_at_utc moved backwards from "
                    f"{previous_recorded.isoformat()} to {recorded_at.isoformat()}"
                )
            previous_recorded = recorded_at
            normalized = normalize_shadow_prediction_payload(entry.payload)
            _validate_shadow_prediction(
                normalized,
                f"entry {index}",
                recorded_at=recorded_at,
                expected_arm_id=expected_arm_id or arm,
                expected_scope=expected_scope or scope,
            )
        except (TypeError, ValueError) as exc:
            problems.append(f"entry {index}: {exc}")
        previous = entry.entry_hash

    if entries and expected_arm_id is not None and identity is not None:
        if identity[0] != expected_arm_id:
            problems.append(
                f"ledger shadow_arm_id is {identity[0]!r}, expected {expected_arm_id!r}"
            )
    if entries and expected_scope is not None and identity is not None:
        if identity[1] != expected_scope:
            problems.append(
                f"ledger shadow_scope is {identity[1]!r}, expected {expected_scope!r}"
            )

    return {
        "ledger": str(ledger_path),
        "entries": len(entries),
        "predictions": len(entries),
        "head_hash": previous,
        "shadow_arm_id": identity[0] if identity else None,
        "shadow_scope": identity[1] if identity else None,
        "shadow_spec_sha256": identity[2] if identity else None,
        "valid": not problems,
        "problems": problems,
    }


@contextmanager
def _exclusive_shadow_lock(ledger_path: Path) -> Iterator[None]:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = ledger_path.with_name(f"{ledger_path.name}.lock")
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _atomic_append(
    ledger_path: Path,
    payloads: Sequence[Mapping[str, object]],
    *,
    recorded_at: str,
    existing: Sequence[ShadowLedgerEntry],
    expected_arm_id: str,
    expected_scope: str,
    publish_deadline: pd.Timestamp,
) -> dict[str, object]:
    sequence = len(existing)
    previous = existing[-1].entry_hash if existing else GENESIS_HASH
    lines: list[str] = []
    for payload in payloads:
        entry_hash = _entry_hash(
            sequence,
            SHADOW_PREDICTION,
            recorded_at,
            payload,
            previous,
        )
        entry = ShadowLedgerEntry(
            sequence=sequence,
            kind=SHADOW_PREDICTION,
            recorded_at_utc=recorded_at,
            payload=dict(payload),
            previous_hash=previous,
            entry_hash=entry_hash,
        )
        lines.append(_canonical(entry.as_dict()))
        previous = entry_hash
        sequence += 1

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    original = ledger_path.read_bytes() if ledger_path.exists() else b""
    descriptor, temporary_name = tempfile.mkstemp(
        dir=ledger_path.parent,
        prefix=f".{ledger_path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(original)
            if original and not original.endswith(b"\n"):
                handle.write(b"\n")
            for line in lines:
                handle.write(line.encode("utf-8") + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        if ledger_path.exists():
            os.chmod(temporary_path, ledger_path.stat().st_mode & 0o777)
        else:
            os.chmod(temporary_path, 0o644)

        report = verify_shadow_ledger(
            temporary_path,
            expected_arm_id=expected_arm_id,
            expected_scope=expected_scope,
        )
        if not report["valid"]:
            raise ValueError(
                f"Shadow ledger candidate is invalid before publish: "
                f"{report['problems']}"
            )
        if _utc_now() >= publish_deadline:
            raise ValueError(
                "Shadow ledger publish reached or passed the earliest kickoff"
            )
        os.replace(temporary_path, ledger_path)
        _fsync_directory(ledger_path.parent)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    report["ledger"] = str(ledger_path)
    report["appended"] = len(lines)
    report["kind"] = SHADOW_PREDICTION
    return report


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _utc_now() -> pd.Timestamp:
    """Wall-clock guard used immediately before the atomic ledger commit."""

    return pd.Timestamp.now(tz="UTC")


def _entry_hash(
    sequence: int,
    kind: str,
    recorded_at_utc: str,
    payload: Mapping[str, object],
    previous_hash: str,
) -> str:
    body = _canonical(
        {
            "sequence": sequence,
            "kind": kind,
            "recorded_at_utc": recorded_at_utc,
            "payload": dict(payload),
            "previous_hash": previous_hash,
        }
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _normalize(value: object) -> object:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be a finite number")
    return numeric


def _canonical_boolean(value: object, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    raise ValueError(f"{label} must be a canonical boolean")


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    return value
