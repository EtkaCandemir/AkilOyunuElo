"""Model explainer PDF cells must be derived from live config/contract inputs."""

from __future__ import annotations

import copy
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

from reportlab.platypus import Table

from ao_elo.config import AOEuropeanEloConfig


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_model_explainer_pdf.py"


def _load_script():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("build_model_explainer_pdf", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _table_rows(flowables: list[object], first_cell: str) -> list[list[object]]:
    for flowable in flowables:
        if (
            isinstance(flowable, Table)
            and _cell_text(flowable._cellvalues[0][0]) == first_cell
        ):
            return flowable._cellvalues
    raise AssertionError(f"table beginning with {first_cell!r} was not produced")


def _cell_text(value: object) -> str:
    getter = getattr(value, "getPlainText", None)
    return str(getter()) if getter is not None else str(value)


def test_layer_cells_follow_in_memory_config_and_contract_values() -> None:
    module = _load_script()
    contract = copy.deepcopy(module._contract())
    config = replace(
        AOEuropeanEloConfig.active(), domestic_surprise_max_abs_adjustment=17.0
    )
    prediction = contract["prediction_layer"]
    prediction["top_level_blend"]["current_ml_weight"] = 0.4
    prediction["top_level_blend"]["ao_domestic_poisson_weight"] = 0.6
    prediction["current_ml_component"]["ml_weight"] = 0.8
    prediction["current_ml_component"]["ao_weight"] = 0.2
    prediction["ao_domestic_poisson_component"]["poisson_weight"] = 0.3
    prediction["ao_domestic_poisson_component"]["ao_weight"] = 0.7
    effective = module._effective_prediction_weights(contract)

    rows = _table_rows(
        module._section_layers(module.styles(), contract, config, effective), "Katman"
    )
    by_layer = {_cell_text(row[0]): row for row in rows[1:]}
    assert "±17" in _cell_text(by_layer["Domestic Surprise"][2])
    assert _cell_text(by_layer["Structural ML"][3]) == "0,32 efektif ağırlık"
    assert _cell_text(by_layer["Domestic Poisson"][3]) == "0,18 efektif ağırlık"


def test_pdf_coverage_table_distinguishes_one_from_none() -> None:
    module = _load_script()
    contract = module._contract()
    config = module.selected_production_prediction_config()
    effective = module._effective_prediction_weights(contract)
    flowables = module._section_prediction(
        module.styles(), contract, config, *effective
    )
    rows = _table_rows(flowables, "Durum")
    one = next(
        _cell_text(row[1])
        for row in rows
        if _cell_text(row[0]) == "Yalnız birinde var"
    )
    none = next(
        _cell_text(row[1])
        for row in rows
        if _cell_text(row[0]) == "Hiçbirinde yok"
    )
    assert "mevcut tarafın profili" in one
    assert "AO tabanına düşer" not in one
    assert "AO tabanına düşer" in none
