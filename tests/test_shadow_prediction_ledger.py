"""Fail-closed contract tests for the isolated AO-core shadow ledger."""

from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path

import pandas as pd
import pytest

from ao_elo.shadow_prediction_ledger import (
    AO_CORE_SHADOW,
    GENESIS_HASH,
    append_shadow_predictions,
    read_shadow_entries,
    require_isolated_shadow_path,
    shadow_ledger_head,
    verify_shadow_ledger,
)


ARM = "POSITIVE_BRIDGE_020"
RECORDED = "2026-09-01T10:00:01+00:00"
PRODUCTION = Path("/tmp/production_prediction_ledger.jsonl")


def _prediction(match_id: str = "UEFA-1", **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "shadow_arm_id": ARM,
        "shadow_scope": AO_CORE_SHADOW,
        "shadow_spec_sha256": "a" * 64,
        "candidate_artifact_manifest_sha256": "b" * 64,
        "source_production_contract_sha256": "c" * 64,
        "source_fixture_sha256": "d" * 64,
        "source_state_sha256": "e" * 64,
        "base_prediction_entry_hash": "f" * 64,
        "prediction_model_version": "positive-bridge-020-ao-core-shadow-v1",
        "match_id": match_id,
        "season": "2026/27",
        "kickoff_utc": "2099-09-08T16:45:00+00:00",
        "generated_at_utc": "2026-09-01T10:00:00+00:00",
        "competition": "UCL",
        "round": "League Stage",
        "stage": "LEAGUE",
        "format_type": "LEAGUE_OR_GROUP",
        "is_neutral": False,
        "home_club_id": "AO-UEFA-1",
        "away_club_id": "AO-UEFA-2",
        "home_live_pre": 1500.0,
        "away_live_pre": 1500.0,
        "expected_home_score": 0.525,
        "home_probability": 0.4,
        "draw_probability": 0.25,
        "away_probability": 0.35,
        "rating_feedback_applied": False,
    }
    row.update(overrides)
    return row


def _append(path: Path, rows: list[dict[str, object]], **kwargs: object):
    return append_shadow_predictions(
        path,
        pd.DataFrame(rows),
        recorded_at=kwargs.pop("recorded_at", RECORDED),
        expected_arm_id=kwargs.pop("expected_arm_id", ARM),
        production_ledger_path=kwargs.pop("production_ledger_path", PRODUCTION),
        **kwargs,
    )


def _concurrent_worker(
    path: str,
    match_id: str,
    barrier: multiprocessing.synchronize.Barrier,
    queue: multiprocessing.Queue,
) -> None:
    try:
        barrier.wait(timeout=10)
        _append(Path(path), [_prediction(match_id)])
        queue.put(None)
    except Exception as exc:  # pragma: no cover - parent checks result
        queue.put(f"{type(exc).__name__}: {exc}")


def test_valid_batch_is_hash_chained_and_has_a_fail_closed_head(tmp_path: Path) -> None:
    path = tmp_path / "shadow.jsonl"
    report = _append(path, [_prediction("UEFA-1"), _prediction("UEFA-2")])
    entries = read_shadow_entries(path)
    head = shadow_ledger_head(path)

    assert report["valid"]
    assert [entry.sequence for entry in entries] == [0, 1]
    assert entries[0].previous_hash == GENESIS_HASH
    assert entries[1].previous_hash == entries[0].entry_hash
    assert head.entry_hash == entries[-1].entry_hash
    assert head.shadow_arm_id == ARM
    assert head.shadow_scope == AO_CORE_SHADOW


def test_schema_is_ao_core_and_needs_no_null_ml_or_poisson_columns(tmp_path: Path) -> None:
    path = tmp_path / "shadow.jsonl"
    _append(path, [_prediction()])
    payload = read_shadow_entries(path)[0].payload
    assert not any(key.startswith(("ml_", "poisson_", "ao_poisson_")) for key in payload)
    assert "ao_home_probability" not in payload
    assert payload["home_probability"] == 0.4


def test_production_path_and_symlink_alias_are_rejected(tmp_path: Path) -> None:
    production = tmp_path / "production.jsonl"
    production.write_text("", encoding="utf-8")
    alias = tmp_path / "alias.jsonl"
    alias.symlink_to(production)

    for candidate in (production, alias, tmp_path / "." / "production.jsonl"):
        with pytest.raises(ValueError, match="production ledger path"):
            require_isolated_shadow_path(
                candidate,
                production_ledger_path=production,
            )


def test_same_match_cannot_be_revised_and_refusal_is_atomic(tmp_path: Path) -> None:
    path = tmp_path / "shadow.jsonl"
    _append(path, [_prediction()])
    before = path.read_bytes()
    with pytest.raises(ValueError, match="revisions are not allowed"):
        _append(path, [_prediction()])
    assert path.read_bytes() == before


def test_recorded_time_cannot_move_backwards_between_appends(tmp_path: Path) -> None:
    path = tmp_path / "shadow.jsonl"
    _append(
        path,
        [_prediction("UEFA-1")],
        recorded_at="2026-09-02T10:00:00+00:00",
    )
    before = path.read_bytes()

    with pytest.raises(ValueError, match="cannot move backwards"):
        _append(
            path,
            [_prediction("UEFA-2")],
            recorded_at="2026-09-01T10:00:01+00:00",
        )

    assert path.read_bytes() == before


def test_duplicate_match_inside_one_batch_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "shadow.jsonl"
    with pytest.raises(ValueError, match="revisions are not allowed"):
        _append(path, [_prediction(), _prediction()])
    assert not path.exists()


def test_empty_batch_is_rejected_without_creating_a_lock_or_parent(tmp_path: Path) -> None:
    parent = tmp_path / "new"
    path = parent / "shadow.jsonl"

    with pytest.raises(ValueError, match="cannot be empty"):
        _append(path, [])

    assert not parent.exists()


def test_arm_scope_and_spec_cannot_change_in_one_ledger(tmp_path: Path) -> None:
    path = tmp_path / "shadow.jsonl"
    _append(path, [_prediction("UEFA-1")])
    before = path.read_bytes()

    with pytest.raises(ValueError, match="shadow_arm_id"):
        _append(
            path,
            [_prediction("UEFA-2", shadow_arm_id="OTHER_ARM")],
            expected_arm_id="OTHER_ARM",
        )
    with pytest.raises(ValueError, match="shadow_scope"):
        _append(path, [_prediction("UEFA-2", shadow_scope="FULL_SERVED_SHADOW")])
    with pytest.raises(ValueError, match="identity cannot change"):
        _append(path, [_prediction("UEFA-2", shadow_spec_sha256="1" * 64)])
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"generated_at_utc": "2026-09-01T10:00:02+00:00"}, "at or before"),
        ({"kickoff_utc": RECORDED}, "before kickoff"),
        ({"kickoff_utc": "2026-09-08 16:45:00"}, "timezone-aware"),
    ],
)
def test_prediction_must_be_prospective(
    tmp_path: Path, overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _append(tmp_path / "shadow.jsonl", [_prediction(**overrides)])


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"home_probability": 0.9}, "sum to one"),
        ({"expected_home_score": 0.6}, "must equal expected_home_score"),
        ({"home_probability": float("nan")}, "finite number"),
        ({"rating_feedback_applied": True}, "must be false"),
    ],
)
def test_probability_and_feedback_invariants_are_enforced(
    tmp_path: Path, overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _append(tmp_path / "shadow.jsonl", [_prediction(**overrides)])


@pytest.mark.parametrize("field", ["base_prediction_entry_hash", "shadow_spec_sha256"])
def test_linkage_hashes_are_mandatory_lowercase_sha256(tmp_path: Path, field: str) -> None:
    for value in (None, "", "A" * 64, "f" * 63):
        with pytest.raises(ValueError, match="SHA-256"):
            _append(tmp_path / f"{len(str(value))}.jsonl", [_prediction(**{field: value})])


def test_base_prediction_hash_cannot_be_genesis(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot be genesis"):
        _append(
            tmp_path / "shadow.jsonl",
            [_prediction(base_prediction_entry_hash=GENESIS_HASH)],
        )


def test_result_fields_are_rejected_before_any_write(tmp_path: Path) -> None:
    path = tmp_path / "shadow.jsonl"
    with pytest.raises(ValueError, match="cannot carry outcome fields"):
        _append(path, [_prediction(home_goals=1)])
    assert not path.exists()


def test_semantic_tampering_is_detected_even_if_chain_hash_is_left_unchanged(
    tmp_path: Path,
) -> None:
    path = tmp_path / "shadow.jsonl"
    _append(path, [_prediction()])
    record = json.loads(path.read_text(encoding="utf-8"))
    record["payload"]["home_probability"] = 2.0
    path.write_text(json.dumps(record, separators=(",", ":")) + "\n", encoding="utf-8")

    report = verify_shadow_ledger(path)
    assert not report["valid"]
    with pytest.raises(ValueError, match="invalid"):
        shadow_ledger_head(path)


def test_publish_failure_leaves_existing_ledger_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "shadow.jsonl"
    _append(path, [_prediction("UEFA-1")])
    before = path.read_bytes()

    def fail_publish(source: object, destination: object) -> None:
        if Path(destination) == path:
            raise OSError("simulated publish failure")
        os.replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_publish)
    with pytest.raises(OSError, match="simulated publish failure"):
        _append(path, [_prediction("UEFA-2")])
    assert path.read_bytes() == before
    assert verify_shadow_ledger(path)["valid"]
    assert not list(tmp_path.glob("*.tmp"))


def test_actual_publish_time_cannot_cross_kickoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ao_elo.shadow_prediction_ledger as ledger_module

    path = tmp_path / "shadow.jsonl"
    monkeypatch.setattr(
        ledger_module,
        "_utc_now",
        lambda: pd.Timestamp("2100-01-01T00:00:00+00:00"),
    )

    with pytest.raises(ValueError, match="at or after the earliest kickoff"):
        _append(path, [_prediction()])
    assert not path.exists()


def test_two_processes_append_without_lost_updates(tmp_path: Path) -> None:
    path = tmp_path / "shadow.jsonl"
    context = multiprocessing.get_context("fork")
    barrier = context.Barrier(2)
    queue = context.Queue()
    processes = [
        context.Process(
            target=_concurrent_worker,
            args=(str(path), match_id, barrier, queue),
        )
        for match_id in ("UEFA-1", "UEFA-2")
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=15)
        assert not process.is_alive()

    assert [queue.get(timeout=2) for _ in processes] == [None, None]
    report = verify_shadow_ledger(path, expected_arm_id=ARM, expected_scope=AO_CORE_SHADOW)
    assert report["valid"], report["problems"]
    assert report["predictions"] == 2
