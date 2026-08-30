"""Append-only, hash-chained ledger for prospective 1X2 predictions.

`docs/HOLDOUT_PROTOCOL_2026_27.md` section 4 requires that a prospective
prediction be written before kickoff and never rewritten afterwards.  A CSV that
gets regenerated on every run cannot support that claim: it proves what the model
says today, not what it said before the match.

A prediction may be revised while the match is unsettled and the revision is
still before kickoff -- that is how a matchday lock at `kickoff - 15 min`
replaces a provisional entry written weeks earlier.  Revisions are appended, not
overwritten: every version stays readable, numbered by `ledger_revision`, and the
last one before kickoff is the locked prediction.  Once a settlement is in the
chain, no further prediction for that match is accepted.

Each entry here stores the hash of the entry before it, so the file is a chain.
Editing or removing any historical entry changes its hash and every hash after
it, which `verify_ledger` detects.

What the chain does NOT do is prove a timestamp.  Anyone able to rewrite the
whole file can rebuild a consistent chain with invented times.  The chain is
only evidence when an earlier head hash exists somewhere the writer cannot
retroactively change -- committing and pushing the ledger, or publishing the
head hash externally.  `ledger_head` exists for exactly that anchoring step.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ao_elo.validators import require_utc_timestamp

GENESIS_HASH = "0" * 64

PREDICTION = "PREDICTION"
SETTLEMENT = "SETTLEMENT"

#: Fields every prospective row must carry, from HOLDOUT_PROTOCOL section 4.
REQUIRED_PREDICTION_FIELDS: tuple[str, ...] = (
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
    "ao_home_probability",
    "ao_draw_probability",
    "ao_away_probability",
    "current_ml_home_probability",
    "current_ml_draw_probability",
    "current_ml_away_probability",
    "ao_poisson_home_probability",
    "ao_poisson_draw_probability",
    "ao_poisson_away_probability",
    "home_probability",
    "draw_probability",
    "away_probability",
    "domestic_poisson_coverage",
    "prediction_status",
    "rating_feedback_applied",
    "prediction_model_version",
    "config_id",
    "contract_sha256",
    "artifact_manifest_sha256",
    "ml_artifact_sha256",
    "domestic_state_sha256",
)

#: A prospective row may never carry anything only knowable after kickoff.
FORBIDDEN_PREDICTION_FIELDS: frozenset[str] = frozenset(
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

REQUIRED_SETTLEMENT_FIELDS: tuple[str, ...] = (
    "match_id",
    "kickoff_utc",
    "home_club_id",
    "away_club_id",
    "home_goals",
    "away_goals",
    "outcome",
)

#: The fixture identity a settlement must reproduce, so a result cannot be
#: attached to a different match than the one that was predicted.
SETTLEMENT_IDENTITY_FIELDS: tuple[str, ...] = (
    "kickoff_utc",
    "home_club_id",
    "away_club_id",
)


@dataclass(frozen=True)
class LedgerEntry:
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
class LedgerHead:
    sequence: int
    entry_hash: str
    entries: int
    predictions: int
    settlements: int


def _canonical(value: object) -> str:
    """Stable serialization, so the same content always hashes the same way."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


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


def _normalize(value: object) -> object:
    """Make a row JSON-safe without losing what it said."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value
    if pd.isna(value):
        return None
    return str(value)


def read_entries(ledger_path: Path) -> list[LedgerEntry]:
    if not ledger_path.exists():
        return []
    entries: list[LedgerEntry] = []
    for number, line in enumerate(
        ledger_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Ledger line {number} is not valid JSON") from exc
        entries.append(
            LedgerEntry(
                sequence=int(record["sequence"]),
                kind=str(record["kind"]),
                recorded_at_utc=str(record["recorded_at_utc"]),
                payload=dict(record["payload"]),
                previous_hash=str(record["previous_hash"]),
                entry_hash=str(record["entry_hash"]),
            )
        )
    return entries


def verify_ledger(ledger_path: Path) -> dict[str, object]:
    """Recompute the whole chain and report where, if anywhere, it breaks."""

    entries = read_entries(ledger_path)
    problems: list[str] = []
    previous = GENESIS_HASH
    for index, entry in enumerate(entries):
        if entry.sequence != index:
            problems.append(f"entry {index}: sequence is {entry.sequence}")
        if entry.previous_hash != previous:
            problems.append(
                f"entry {index}: previous_hash does not match entry {index - 1}"
            )
        expected = _entry_hash(
            entry.sequence,
            entry.kind,
            entry.recorded_at_utc,
            entry.payload,
            entry.previous_hash,
        )
        if entry.entry_hash != expected:
            problems.append(f"entry {index}: content does not match its hash")
        previous = entry.entry_hash

    predictions = [e for e in entries if e.kind == PREDICTION]
    settlements = [e for e in entries if e.kind == SETTLEMENT]
    predicted_ids = [str(e.payload["match_id"]) for e in predictions]
    settled_at: dict[str, int] = {}
    for entry in entries:
        key = str(entry.payload["match_id"])
        if entry.kind == SETTLEMENT:
            settled_at.setdefault(key, entry.sequence)
        elif entry.kind == PREDICTION and key in settled_at:
            problems.append(
                f"entry {entry.sequence}: predicts {key} after it was settled at "
                f"entry {settled_at[key]}"
            )
    settled_ids = [str(e.payload["match_id"]) for e in settlements]
    if len(set(settled_ids)) != len(settled_ids):
        problems.append("a match_id was settled more than once")
    unknown = sorted(set(settled_ids) - set(predicted_ids))
    if unknown:
        problems.append(f"settlements without a prediction: {unknown[:5]}")

    return {
        "ledger": str(ledger_path),
        "entries": len(entries),
        "predictions": len(predictions),
        "settlements": len(settlements),
        "head_hash": previous,
        "valid": not problems,
        "problems": problems,
    }


def ledger_head(ledger_path: Path) -> LedgerHead:
    """Head hash to anchor externally; anchoring is what makes the chain proof."""

    entries = read_entries(ledger_path)
    return LedgerHead(
        sequence=entries[-1].sequence if entries else -1,
        entry_hash=entries[-1].entry_hash if entries else GENESIS_HASH,
        entries=len(entries),
        predictions=sum(1 for e in entries if e.kind == PREDICTION),
        settlements=sum(1 for e in entries if e.kind == SETTLEMENT),
    )


def _validate_prediction(payload: Mapping[str, object], label: str) -> None:
    missing = sorted(set(REQUIRED_PREDICTION_FIELDS).difference(payload))
    if missing:
        raise ValueError(f"{label}: prediction is missing fields {missing}")
    present = FORBIDDEN_PREDICTION_FIELDS.intersection(payload)
    if present:
        raise ValueError(
            f"{label}: a pre-match record cannot carry outcome fields "
            f"{sorted(present)}"
        )
    kickoff = require_utc_timestamp(payload["kickoff_utc"], f"{label}: kickoff_utc")
    generated = require_utc_timestamp(
        payload["generated_at_utc"], f"{label}: generated_at_utc"
    )
    if generated >= kickoff:
        raise ValueError(
            f"{label}: generated_at_utc {generated.isoformat()} is not before "
            f"kickoff {kickoff.isoformat()}; the row is not prospective"
        )
    if payload["rating_feedback_applied"] not in (False, "False", "false", 0):
        raise ValueError(f"{label}: rating_feedback_applied must be false")
    triple = (
        float(payload["home_probability"]),
        float(payload["draw_probability"]),
        float(payload["away_probability"]),
    )
    if any(not 0.0 <= value <= 1.0 for value in triple):
        raise ValueError(f"{label}: served probabilities must lie in [0,1]")
    if abs(sum(triple) - 1.0) > 1e-9:
        raise ValueError(f"{label}: served probabilities must sum to one")


def append_predictions(
    ledger_path: Path,
    rows: pd.DataFrame,
    *,
    recorded_at: object,
) -> dict[str, object]:
    """Append prospective rows. Re-predicting a settled match is refused."""

    recorded_ts = require_utc_timestamp(recorded_at, "recorded_at")
    recorded = recorded_ts.isoformat()
    entries = read_entries(ledger_path)
    revisions: dict[str, int] = {}
    for entry in entries:
        if entry.kind == PREDICTION:
            key = str(entry.payload["match_id"])
            revisions[key] = revisions.get(key, -1) + 1
    settled = {
        str(entry.payload["match_id"]) for entry in entries if entry.kind == SETTLEMENT
    }
    payloads: list[dict[str, object]] = []
    for position, row in enumerate(rows.to_dict("records")):
        payload = {key: _normalize(value) for key, value in row.items()}
        match_id = str(payload.get("match_id"))
        label = f"row {position} ({match_id})"
        _validate_prediction(payload, label)
        if match_id in settled:
            raise ValueError(
                f"{label}: the match is already settled; a prediction can never be "
                "added or revised after the result is in the ledger"
            )
        kickoff = require_utc_timestamp(payload["kickoff_utc"], label)
        if recorded_ts >= kickoff:
            raise ValueError(
                f"{label}: recorded_at {recorded} is not before kickoff "
                f"{kickoff.isoformat()}; the entry would not be prospective"
            )
        revision = revisions.get(match_id, -1) + 1
        revisions[match_id] = revision
        payload["ledger_revision"] = revision
        payloads.append(payload)
    return _append(ledger_path, PREDICTION, payloads, recorded, entries)


def append_settlements(
    ledger_path: Path,
    rows: pd.DataFrame,
    *,
    recorded_at: object,
) -> dict[str, object]:
    """Append results. The prediction row is never touched."""

    recorded = require_utc_timestamp(recorded_at, "recorded_at").isoformat()
    entries = read_entries(ledger_path)
    # Sonuncu pre-kickoff tahmin kilitli olandir; settlement ona baglanir.
    predicted: dict[str, Mapping[str, object]] = {}
    for entry in entries:
        if entry.kind == PREDICTION:
            predicted[str(entry.payload["match_id"])] = entry.payload
    settled = {
        str(entry.payload["match_id"]) for entry in entries if entry.kind == SETTLEMENT
    }
    payloads: list[dict[str, object]] = []
    for position, row in enumerate(rows.to_dict("records")):
        payload = {key: _normalize(value) for key, value in row.items()}
        match_id = str(payload.get("match_id"))
        label = f"row {position} ({match_id})"
        missing = sorted(set(REQUIRED_SETTLEMENT_FIELDS).difference(payload))
        if missing:
            raise ValueError(f"{label}: settlement is missing fields {missing}")
        if match_id not in predicted:
            raise ValueError(f"{label}: settles a match that was never predicted")
        if match_id in settled:
            raise ValueError(f"{label}: already settled")
        original = predicted[match_id]
        for field in SETTLEMENT_IDENTITY_FIELDS:
            if str(payload[field]) != str(original[field]):
                raise ValueError(
                    f"{label}: settlement {field} does not match the predicted "
                    f"fixture ({payload[field]!r} vs {original[field]!r})"
                )
        settled.add(match_id)
        payloads.append(payload)
    return _append(ledger_path, SETTLEMENT, payloads, recorded, entries)


def _append(
    ledger_path: Path,
    kind: str,
    payloads: Sequence[Mapping[str, object]],
    recorded_at: str,
    entries: Iterable[LedgerEntry],
) -> dict[str, object]:
    existing = list(entries)
    sequence = len(existing)
    previous = existing[-1].entry_hash if existing else GENESIS_HASH
    lines: list[str] = []
    for payload in payloads:
        entry_hash = _entry_hash(sequence, kind, recorded_at, payload, previous)
        entry = LedgerEntry(
            sequence=sequence,
            kind=kind,
            recorded_at_utc=recorded_at,
            payload=dict(payload),
            previous_hash=previous,
            entry_hash=entry_hash,
        )
        lines.append(_canonical(entry.as_dict()))
        previous = entry_hash
        sequence += 1

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    # Open in append mode so an existing entry is never rewritten, even by a bug
    # in this function.
    with ledger_path.open("a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line + "\n")

    report = verify_ledger(ledger_path)
    if not report["valid"]:
        raise ValueError(f"Ledger became invalid after append: {report['problems']}")
    report["appended"] = len(lines)
    report["kind"] = kind
    return report
