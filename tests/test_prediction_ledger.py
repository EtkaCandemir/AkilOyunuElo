"""Ledger'in kirilmasi gereken yerlerde kirildigini dogrular."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ao_elo.prediction_ledger import (
    GENESIS_HASH,
    append_predictions,
    append_settlements,
    ledger_head,
    read_entries,
    verify_ledger,
)


def _prediction(match_id: str = "UEFA-1", **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "match_id": match_id,
        "season": "2026/27",
        "kickoff_utc": "2026-09-08T16:45:00+00:00",
        "generated_at_utc": "2026-09-08T16:30:00+00:00",
        "competition": "UCL",
        "round": "League Stage",
        "stage": "LEAGUE",
        "format_type": "LEAGUE_OR_GROUP",
        "is_neutral": False,
        "home_club_id": "AO-UEFA-1",
        "away_club_id": "AO-UEFA-2",
        "domestic_poisson_coverage": "BOTH",
        "prediction_status": "ACTIVE_ENSEMBLE",
        "rating_feedback_applied": False,
        "prediction_model_version": "ao-ml-poisson-ensemble-v1-production",
        "config_id": "9c2ed2fad929252b",
        "contract_sha256": "a" * 64,
        "artifact_manifest_sha256": "b" * 64,
        "ml_artifact_sha256": "c" * 64,
        "domestic_state_sha256": "d" * 64,
    }
    for prefix in ("ao", "current_ml", "ao_poisson"):
        row[f"{prefix}_home_probability"] = 0.4
        row[f"{prefix}_draw_probability"] = 0.25
        row[f"{prefix}_away_probability"] = 0.35
    row["home_probability"] = 0.4
    row["draw_probability"] = 0.25
    row["away_probability"] = 0.35
    row.update(overrides)
    return row


def _settlement(match_id: str = "UEFA-1", **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "match_id": match_id,
        "kickoff_utc": "2026-09-08T16:45:00+00:00",
        "home_club_id": "AO-UEFA-1",
        "away_club_id": "AO-UEFA-2",
        "home_goals": 2,
        "away_goals": 1,
        "outcome": "HOME",
    }
    row.update(overrides)
    return row


RECORDED = "2026-09-08T16:31:00+00:00"
SETTLED = "2026-09-08T18:45:00+00:00"


# --- zincir ---------------------------------------------------------------


def test_empty_ledger_head_is_the_genesis_hash(tmp_path: Path) -> None:
    head = ledger_head(tmp_path / "l.jsonl")
    assert head.entry_hash == GENESIS_HASH
    assert head.entries == 0


def test_appending_links_each_entry_to_the_previous_one(tmp_path: Path) -> None:
    path = tmp_path / "l.jsonl"
    append_predictions(
        path,
        pd.DataFrame([_prediction("UEFA-1"), _prediction("UEFA-2")]),
        recorded_at=RECORDED,
    )
    entries = read_entries(path)
    assert [e.sequence for e in entries] == [0, 1]
    assert entries[0].previous_hash == GENESIS_HASH
    assert entries[1].previous_hash == entries[0].entry_hash
    assert verify_ledger(path)["valid"]


def test_editing_a_historical_entry_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "l.jsonl"
    append_predictions(
        path,
        pd.DataFrame([_prediction("UEFA-1"), _prediction("UEFA-2")]),
        recorded_at=RECORDED,
    )
    assert verify_ledger(path)["valid"]

    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["payload"]["home_probability"] = 0.99  # tahmini sonradan degistir
    lines[0] = json.dumps(record, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = verify_ledger(path)
    assert not report["valid"]
    assert any("does not match its hash" in p for p in report["problems"])


def test_deleting_an_entry_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "l.jsonl"
    append_predictions(
        path,
        pd.DataFrame([_prediction("UEFA-1"), _prediction("UEFA-2"), _prediction("UEFA-3")]),
        recorded_at=RECORDED,
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join([lines[0], lines[2]]) + "\n", encoding="utf-8")
    assert not verify_ledger(path)["valid"]


# --- prospective kural ----------------------------------------------------


def test_a_row_generated_after_kickoff_is_refused(tmp_path: Path) -> None:
    row = _prediction(generated_at_utc="2026-09-08T16:46:00+00:00")
    with pytest.raises(ValueError, match="not before kickoff"):
        append_predictions(tmp_path / "l.jsonl", pd.DataFrame([row]), recorded_at=RECORDED)


def test_a_row_generated_exactly_at_kickoff_is_refused(tmp_path: Path) -> None:
    row = _prediction(generated_at_utc="2026-09-08T16:45:00+00:00")
    with pytest.raises(ValueError, match="not before kickoff"):
        append_predictions(tmp_path / "l.jsonl", pd.DataFrame([row]), recorded_at=RECORDED)


def test_a_naive_timestamp_is_refused(tmp_path: Path) -> None:
    row = _prediction(kickoff_utc="2026-09-08 16:45:00")
    with pytest.raises(ValueError, match="timezone-aware"):
        append_predictions(tmp_path / "l.jsonl", pd.DataFrame([row]), recorded_at=RECORDED)


def test_an_outcome_field_makes_the_row_not_prospective(tmp_path: Path) -> None:
    row = _prediction(home_goals=2)
    with pytest.raises(ValueError, match="cannot carry outcome fields"):
        append_predictions(tmp_path / "l.jsonl", pd.DataFrame([row]), recorded_at=RECORDED)


def test_rating_feedback_must_be_false(tmp_path: Path) -> None:
    row = _prediction(rating_feedback_applied=True)
    with pytest.raises(ValueError, match="rating_feedback_applied"):
        append_predictions(tmp_path / "l.jsonl", pd.DataFrame([row]), recorded_at=RECORDED)


def test_probabilities_must_sum_to_one(tmp_path: Path) -> None:
    row = _prediction(home_probability=0.9)
    with pytest.raises(ValueError, match="sum to one"):
        append_predictions(tmp_path / "l.jsonl", pd.DataFrame([row]), recorded_at=RECORDED)


# --- degistirilemezlik ----------------------------------------------------


def test_a_prediction_can_be_revised_before_kickoff_and_both_versions_survive(
    tmp_path: Path,
) -> None:
    # Haftalar once yazilan gecici kayit, macgunu kilidiyle degistirilebilir;
    # eski surum silinmez, ledger_revision ile numaralanir.
    path = tmp_path / "l.jsonl"
    append_predictions(
        path, pd.DataFrame([_prediction()]), recorded_at="2026-08-30T12:00:00+00:00"
    )
    append_predictions(
        path,
        pd.DataFrame(
            [_prediction(home_probability=0.5, draw_probability=0.2, away_probability=0.3)]
        ),
        recorded_at="2026-09-08T16:30:00+00:00",
    )
    entries = [e for e in read_entries(path) if e.kind == "PREDICTION"]
    assert [e.payload["ledger_revision"] for e in entries] == [0, 1]
    assert entries[0].payload["home_probability"] == 0.4
    assert entries[1].payload["home_probability"] == 0.5
    assert verify_ledger(path)["valid"]


def test_a_revision_recorded_after_kickoff_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "l.jsonl"
    append_predictions(path, pd.DataFrame([_prediction()]), recorded_at=RECORDED)
    with pytest.raises(ValueError, match="not before kickoff"):
        append_predictions(
            path, pd.DataFrame([_prediction()]), recorded_at="2026-09-08T17:00:00+00:00"
        )


def test_no_prediction_is_accepted_after_settlement(tmp_path: Path) -> None:
    path = tmp_path / "l.jsonl"
    append_predictions(path, pd.DataFrame([_prediction()]), recorded_at=RECORDED)
    append_settlements(path, pd.DataFrame([_settlement()]), recorded_at=SETTLED)
    with pytest.raises(ValueError, match="already settled"):
        append_predictions(path, pd.DataFrame([_prediction()]), recorded_at=RECORDED)


def test_a_refused_batch_leaves_the_ledger_untouched(tmp_path: Path) -> None:
    path = tmp_path / "l.jsonl"
    append_predictions(path, pd.DataFrame([_prediction("UEFA-1")]), recorded_at=RECORDED)
    before = path.read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        append_predictions(
            path,
            pd.DataFrame(
                [_prediction("UEFA-2"), _prediction("UEFA-3", home_goals=1)]
            ),
            recorded_at=RECORDED,
        )
    assert path.read_text(encoding="utf-8") == before


# --- settlement -----------------------------------------------------------


def test_settlement_is_appended_not_written_over_the_prediction(tmp_path: Path) -> None:
    path = tmp_path / "l.jsonl"
    append_predictions(path, pd.DataFrame([_prediction()]), recorded_at=RECORDED)
    append_settlements(path, pd.DataFrame([_settlement()]), recorded_at=SETTLED)
    entries = read_entries(path)
    assert [e.kind for e in entries] == ["PREDICTION", "SETTLEMENT"]
    assert entries[0].payload["home_probability"] == 0.4
    assert verify_ledger(path)["valid"]


def test_settling_an_unpredicted_match_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "l.jsonl"
    append_predictions(path, pd.DataFrame([_prediction("UEFA-1")]), recorded_at=RECORDED)
    with pytest.raises(ValueError, match="never predicted"):
        append_settlements(path, pd.DataFrame([_settlement("UEFA-9")]), recorded_at=SETTLED)


def test_settling_twice_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "l.jsonl"
    append_predictions(path, pd.DataFrame([_prediction()]), recorded_at=RECORDED)
    append_settlements(path, pd.DataFrame([_settlement()]), recorded_at=SETTLED)
    with pytest.raises(ValueError, match="already settled"):
        append_settlements(path, pd.DataFrame([_settlement()]), recorded_at=SETTLED)


def test_settlement_must_match_the_predicted_fixture(tmp_path: Path) -> None:
    path = tmp_path / "l.jsonl"
    append_predictions(path, pd.DataFrame([_prediction()]), recorded_at=RECORDED)
    with pytest.raises(ValueError, match="does not match the predicted fixture"):
        append_settlements(
            path,
            pd.DataFrame([_settlement(away_club_id="AO-UEFA-99")]),
            recorded_at=SETTLED,
        )


# --- yayimlanan ledger ----------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
LIVE_LEDGER = ROOT / "data" / "prediction_ledger" / "prediction_ledger_2026_27.jsonl"
LIVE_ANCHOR = ROOT / "data" / "prediction_ledger" / "ledger_anchor_2026_27.json"


def test_the_committed_ledger_chain_is_intact() -> None:
    if not LIVE_LEDGER.exists():  # pragma: no cover - ledger henuz kurulmadiysa
        pytest.skip("ledger not created yet")
    report = verify_ledger(LIVE_LEDGER)
    assert report["valid"], report["problems"]


def test_the_anchor_still_describes_the_committed_ledger() -> None:
    # Anchor eslesmiyorsa ya ledger disaridan degistirilmis ya da anchor
    # tazelenmemistir; ikisi de elle incelenmelidir.
    if not LIVE_ANCHOR.exists():  # pragma: no cover
        pytest.skip("anchor not created yet")
    anchor = json.loads(LIVE_ANCHOR.read_text(encoding="utf-8"))
    head = ledger_head(LIVE_LEDGER)
    assert head.entry_hash == anchor["head_hash"]
    assert head.entries == anchor["entries"]


def test_every_committed_prediction_is_prospective() -> None:
    if not LIVE_LEDGER.exists():  # pragma: no cover
        pytest.skip("ledger not created yet")
    for entry in read_entries(LIVE_LEDGER):
        if entry.kind != "PREDICTION":
            continue
        generated = pd.Timestamp(entry.payload["generated_at_utc"])
        kickoff = pd.Timestamp(entry.payload["kickoff_utc"])
        assert generated < kickoff, entry.payload["match_id"]
