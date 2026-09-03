from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from ao_elo.prediction_ledger import LedgerEntry
from ao_elo.shadow_prediction_ledger import (
    AO_CORE_SHADOW,
    append_shadow_predictions,
)
from scripts.run_positive_bridge_shadow_2026_27 import (
    build_shadow_prediction_frame,
    recover_or_verify_shadow_ledger,
    validate_output_isolation,
)


def runner_args(tmp_path: Path) -> SimpleNamespace:
    source = tmp_path / "sources"
    return SimpleNamespace(
        spec=source / "spec.json",
        contract=source / "contract.json",
        production_state=source / "state.json",
        production_seeds=source / "seeds.csv",
        completed_matches=source / "matches.csv",
        fixtures=source / "fixtures.csv",
        served_predictions=source / "served.csv",
        production_ledger=source / "production.jsonl",
        production_checkpoint=source / "checkpoint",
        shadow_ledger=tmp_path / "shadow" / "ledger.jsonl",
        shadow_anchor=tmp_path / "shadow" / "anchor.json",
        artifact_root=tmp_path / "artifacts" / "candidate",
        run_output=tmp_path / "output" / "run",
    )


def recovery_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(396):
        rows.append(
            {
                "shadow_arm_id": "POSITIVE_BRIDGE_020",
                "shadow_scope": AO_CORE_SHADOW,
                "shadow_spec_sha256": "a" * 64,
                "candidate_artifact_manifest_sha256": "b" * 64,
                "source_production_contract_sha256": "c" * 64,
                "source_fixture_sha256": "d" * 64,
                "source_state_sha256": "e" * 64,
                "base_prediction_entry_hash": hashlib.sha256(
                    f"base-{index}".encode()
                ).hexdigest(),
                "prediction_model_version": "ao-positive-bridge-020-core-shadow-v1",
                "match_id": f"MATCH-{index:03d}",
                "season": "2026/27",
                "kickoff_utc": "2099-09-08T16:45:00+00:00",
                "generated_at_utc": "2026-09-01T10:00:00+00:00",
                "competition": "UCL",
                "round": "League Stage",
                "stage": "LEAGUE",
                "format_type": "LEAGUE_OR_GROUP",
                "is_neutral": False,
                "home_club_id": f"HOME-{index:03d}",
                "away_club_id": f"AWAY-{index:03d}",
                "home_live_pre": 1500.0,
                "away_live_pre": 1500.0,
                "expected_home_score": 0.525,
                "home_probability": 0.4,
                "draw_probability": 0.25,
                "away_probability": 0.35,
                "rating_feedback_applied": False,
            }
        )
    return pd.DataFrame(rows)


def test_output_isolation_rejects_aliases_and_overlapping_directories(
    tmp_path: Path,
) -> None:
    args = runner_args(tmp_path)
    validate_output_isolation(args)

    args.shadow_anchor = args.shadow_ledger
    with pytest.raises(ValueError, match="must be distinct"):
        validate_output_isolation(args)

    args = runner_args(tmp_path)
    args.shadow_anchor = args.shadow_ledger / "anchor.json"
    with pytest.raises(ValueError, match="non-overlapping"):
        validate_output_isolation(args)

    args = runner_args(tmp_path)
    args.shadow_ledger = args.artifact_root / "ledger.jsonl"
    with pytest.raises(ValueError, match="cannot overlap"):
        validate_output_isolation(args)

    args = runner_args(tmp_path)
    args.run_output = args.artifact_root / "run"
    with pytest.raises(ValueError, match="cannot overlap"):
        validate_output_isolation(args)


def test_base_prediction_link_requires_full_fixture_identity() -> None:
    kickoff = datetime(2026, 9, 8, 16, 45, tzinfo=timezone.utc)
    locked = SimpleNamespace(
        kickoff_utc=kickoff,
        season="2026/27",
        competition="UCL",
        round="League Stage",
        stage="LEAGUE",
        is_neutral=False,
        home_team_id="AO-1",
        away_team_id="AO-2",
    )
    payload = {
        "match_id": "MATCH-1",
        "season": "2026/27",
        "kickoff_utc": kickoff.isoformat(),
        "competition": "UEL",
        "round": "League Stage",
        "stage": "LEAGUE",
        "format_type": "LEAGUE_OR_GROUP",
        "is_neutral": False,
        "home_club_id": "AO-1",
        "away_club_id": "AO-2",
    }
    entry = LedgerEntry(
        sequence=0,
        kind="PREDICTION",
        recorded_at_utc="2026-09-01T00:00:00+00:00",
        payload=payload,
        previous_hash="0" * 64,
        entry_hash="a" * 64,
    )

    with pytest.raises(ValueError, match="identity mismatch"):
        build_shadow_prediction_frame(
            candidate_locked={"MATCH-1": locked},
            baseline_locked={"MATCH-1": locked},
            latest_entries={"MATCH-1": entry},
            candidate_seeds=pd.DataFrame(
                {
                    "team_id": ["AO-1", "AO-2"],
                    "production_ao_first_elo": [1500.0, 1500.0],
                    "ao_first_elo": [1500.0, 1500.0],
                    "positive_bridge_applied": [False, False],
                }
            ),
            source_hashes={},
            spec_sha256="b" * 64,
            production_ledger_head="c" * 64,
        )


def test_resume_verifies_complete_ledger_without_duplicate_append(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = tmp_path / "shadow.jsonl"
    rows = recovery_rows()
    append_shadow_predictions(
        ledger,
        rows,
        recorded_at="2026-09-01T10:00:01+00:00",
        expected_arm_id="POSITIVE_BRIDGE_020",
        production_ledger_path=tmp_path / "production.jsonl",
        expected_scope=AO_CORE_SHADOW,
    )
    before = ledger.read_bytes()

    def reject_duplicate_append(*args: object, **kwargs: object) -> None:
        raise AssertionError("resume attempted a duplicate append")

    monkeypatch.setitem(
        recover_or_verify_shadow_ledger.__globals__,
        "append_shadow_predictions",
        reject_duplicate_append,
    )
    report, recorded_at = recover_or_verify_shadow_ledger(
        ledger,
        rows,
        production_ledger_path=tmp_path / "production.jsonl",
    )

    assert report["valid"]
    assert report["entries"] == 396
    assert recorded_at == "2026-09-01T10:00:01+00:00"
    assert ledger.read_bytes() == before


def test_resume_rejects_ledger_payload_that_differs_from_artifact(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "shadow.jsonl"
    rows = recovery_rows()
    append_shadow_predictions(
        ledger,
        rows,
        recorded_at="2026-09-01T10:00:01+00:00",
        expected_arm_id="POSITIVE_BRIDGE_020",
        production_ledger_path=tmp_path / "production.jsonl",
        expected_scope=AO_CORE_SHADOW,
    )
    before = ledger.read_bytes()
    changed = rows.copy()
    changed.loc[0, "home_live_pre"] = 1500.5

    with pytest.raises(ValueError, match="differs from the published artifact"):
        recover_or_verify_shadow_ledger(
            ledger,
            changed,
            production_ledger_path=tmp_path / "production.jsonl",
        )

    assert ledger.read_bytes() == before


def test_resume_rejects_partial_nonempty_ledger(tmp_path: Path) -> None:
    ledger = tmp_path / "shadow.jsonl"
    rows = recovery_rows()
    append_shadow_predictions(
        ledger,
        rows.iloc[:1],
        recorded_at="2026-09-01T10:00:01+00:00",
        expected_arm_id="POSITIVE_BRIDGE_020",
        production_ledger_path=tmp_path / "production.jsonl",
        expected_scope=AO_CORE_SHADOW,
    )
    before = ledger.read_bytes()

    with pytest.raises(ValueError, match="complete 396-entry atomic batch"):
        recover_or_verify_shadow_ledger(
            ledger,
            rows,
            production_ledger_path=tmp_path / "production.jsonl",
        )

    assert ledger.read_bytes() == before
