"""Tur 3 F1/F2/F3/F7/F8: builder ve contract sinir kontrolleri."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from ao_elo.production_prediction import load_production_prediction_runtime

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts" / "ao_european_elo_v2_production.json"
LIVE_BUILDER = ROOT / "scripts" / "build_ucl_uel_live_domestic_state.py"


def _load_script(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# --- F1 -------------------------------------------------------------------


def test_live_build_refuses_to_start_without_the_coverage_audit(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_production_prediction_artifacts.py"),
            "--live-domestic-matches",
            str(ROOT / "data/domestic_league_expansion_ucl_uel/live_2026_27_matches.csv"),
            "--output-root",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode != 0
    assert "--coverage-audit is required" in result.stderr
    # Hicbir kismi artifact yazilmamis olmali.
    assert not (tmp_path / "out").exists() or not list((tmp_path / "out").iterdir())


# --- F2 -------------------------------------------------------------------


def test_metadata_follows_the_fixtures_override(tmp_path: Path) -> None:
    builder = _load_script(ROOT / "scripts" / "build_2026_27_prediction_features.py")
    default = pd.read_csv(builder.DEFAULT_FIXTURES)
    override = default.head(1).copy()
    override["match_id"] = "OVERRIDE-TEST-0001"
    path = tmp_path / "fixtures.csv"
    override.to_csv(path, index=False)

    metadata = builder.load_metadata(path)
    row = metadata[metadata["match_id"].eq("OVERRIDE-TEST-0001")]
    assert len(row) == 1, "override fixture must reach the metadata frame"
    assert row["round"].item() == override["round"].item()
    assert int(row["leg_number"].item()) == int(override["leg_number"].item())

    # Varsayilan dosya okunsaydi bu ID hic bulunmazdi.
    assert builder.load_metadata()["match_id"].eq("OVERRIDE-TEST-0001").sum() == 0


# --- F3 -------------------------------------------------------------------


def test_live_state_builder_refuses_a_failed_league(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script(LIVE_BUILDER)
    source = LIVE_BUILDER.read_text(encoding="utf-8")
    # Guard, kismi state yazilmadan once ve audit yazildiktan sonra olmali.
    guard = source.index("Refusing to write a domestic state")
    assert guard < source.index('if live.empty:')
    assert source.index("live_fixture_reconciliation_audit.csv") < guard
    assert hasattr(module, "_merge_bridges")


# --- F7 -------------------------------------------------------------------


@pytest.mark.parametrize("block", ["domestic_config", "transfer_config"])
def test_contract_block_missing_a_field_is_refused(block: str, tmp_path: Path) -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    def locate(node: object) -> dict | None:
        if isinstance(node, dict):
            if block in node:
                return node
            for value in node.values():
                found = locate(value)
                if found is not None:
                    return found
        return None

    holder = locate(contract)
    assert holder is not None
    removed = sorted(holder[block])[0]
    holder[block].pop(removed)
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match=f"{block} is missing contract fields"):
        load_production_prediction_runtime(path, repository_root=ROOT)


def test_contract_block_with_an_unknown_field_is_refused(tmp_path: Path) -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    def locate(node: object) -> dict | None:
        if isinstance(node, dict):
            if "domestic_config" in node:
                return node
            for value in node.values():
                found = locate(value)
                if found is not None:
                    return found
        return None

    holder = locate(contract)
    assert holder is not None
    holder["domestic_config"]["not_a_real_field"] = 1
    path = tmp_path / "extra.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown contract fields"):
        load_production_prediction_runtime(path, repository_root=ROOT)


# --- F8 -------------------------------------------------------------------


def test_bridge_merge_keeps_a_newly_resolved_identity_over_an_old_null() -> None:
    module = _load_script(LIVE_BUILDER)
    existing = pd.DataFrame([{"country_code": "ZZZ", "source_team_id": "1", "ao_club_id": None}])
    extra = pd.DataFrame([{"country_code": "ZZZ", "source_team_id": "1", "ao_club_id": "AO-X"}])
    assert module._merge_bridges(existing, extra)["ao_club_id"].tolist() == ["AO-X"]


def test_bridge_merge_still_rejects_two_different_identities() -> None:
    module = _load_script(LIVE_BUILDER)
    left = pd.DataFrame([{"country_code": "ZZZ", "source_team_id": "1", "ao_club_id": "AO-X"}])
    right = pd.DataFrame([{"country_code": "ZZZ", "source_team_id": "1", "ao_club_id": "AO-Y"}])
    with pytest.raises(ValueError, match="identity conflicts"):
        module._merge_bridges(left, right)
