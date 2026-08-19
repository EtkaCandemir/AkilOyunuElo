from __future__ import annotations

"""Guard the six-season xG dataset and the layer revalidation it feeds.

The bounded xG layer shipped on one season of evidence. Widening that base is
only meaningful if the wider data keeps the same acceptance contract, so these
tests pin the properties that make the new sample comparable to the old one
rather than merely larger.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_xg_multiseason_backtest import (
    QUALIFYING_ROUNDS,
    match_context,
    validation_gates,
)


WIDE = ROOT / "data" / "xg_2020_2026" / "uefa_2020_2026_matches_with_xg.csv"
LEGACY = ROOT / "data" / "xg_2025_26" / "uefa_2025_26_matches_with_xg.csv"
COVERAGE_SEASONS = {"2020/21", "2021/22", "2022/23", "2023/24", "2024/25", "2025/26"}

requires_dataset = pytest.mark.skipif(
    not WIDE.exists(), reason="six-season xG dataset is not built in this checkout"
)


@pytest.fixture(name="wide", scope="module")
def fixture_wide() -> pd.DataFrame:
    return pd.read_csv(WIDE)


# ---------------------------------------------------------------------------
# acceptance contract: the wider sample must not relax what the shipped one did
# ---------------------------------------------------------------------------


@requires_dataset
def test_eligible_rows_always_carry_both_sides(wide: pd.DataFrame) -> None:
    eligible = wide.loc[wide["xg_analysis_eligible"].astype(bool)]

    assert not eligible[["xg_home", "xg_away"]].isna().any().any()
    assert (eligible[["xg_home", "xg_away"]] >= 0).all().all()


@requires_dataset
def test_ineligible_rows_are_never_imputed(wide: pd.DataFrame) -> None:
    """Missing xG must stay missing rather than become a zero."""
    ineligible = wide.loc[~wide["xg_analysis_eligible"].astype(bool)]

    assert ineligible["xg_home"].isna().all()
    assert ineligible["xg_away"].isna().all()


@requires_dataset
def test_every_ineligible_row_records_a_reason(wide: pd.DataFrame) -> None:
    ineligible = wide.loc[~wide["xg_analysis_eligible"].astype(bool)]

    assert ineligible["xg_missing_reason"].notna().all()


@requires_dataset
def test_seasons_are_limited_to_the_covered_window(wide: pd.DataFrame) -> None:
    """FotMob has no xG before 2020/21, so earlier seasons must not appear."""
    assert set(wide["season"]) == COVERAGE_SEASONS


@requires_dataset
def test_match_ids_are_unique(wide: pd.DataFrame) -> None:
    assert wide["match_id"].is_unique


# ---------------------------------------------------------------------------
# determinism against the frozen single-season dataset
# ---------------------------------------------------------------------------


@requires_dataset
@pytest.mark.skipif(not LEGACY.exists(), reason="frozen 2025/26 dataset is missing")
def test_2025_26_slice_reproduces_the_frozen_dataset(wide: pd.DataFrame) -> None:
    """A re-fetch that changed values would invalidate the comparison."""
    legacy = pd.read_csv(LEGACY)
    current = wide.loc[wide["season"].eq("2025/26")]
    merged = legacy[["match_id", "xg_home", "xg_away"]].merge(
        current[["match_id", "xg_home", "xg_away"]],
        on="match_id",
        suffixes=("_frozen", "_current"),
        validate="one_to_one",
    )

    assert len(merged) == len(legacy)
    for side in ("home", "away"):
        frozen = merged[f"xg_{side}_frozen"]
        fresh = merged[f"xg_{side}_current"]
        assert frozen.isna().equals(fresh.isna())
        both = frozen.notna()
        assert np.allclose(frozen[both], fresh[both], atol=1e-9)


@requires_dataset
def test_wide_sample_is_materially_larger_than_the_shipped_evidence(
    wide: pd.DataFrame,
) -> None:
    """The contract activated the layer on 606 eligible matches."""
    eligible = int(wide["xg_analysis_eligible"].astype(bool).sum())

    assert eligible > 4 * 606


# ---------------------------------------------------------------------------
# coverage shape: the layer can only act where xG exists
# ---------------------------------------------------------------------------


@requires_dataset
def test_coverage_is_concentrated_in_the_main_stage(wide: pd.DataFrame) -> None:
    eligible = wide.loc[wide["xg_analysis_eligible"].astype(bool)]
    main_share = eligible["competition_phase"].eq("MAIN").mean()

    assert main_share > 0.80


@requires_dataset
def test_qualifying_coverage_is_documented_as_thin(wide: pd.DataFrame) -> None:
    """Reported so a future run notices if the source backfills qualifiers."""
    qualifying = wide.loc[wide["competition_phase"].eq("QUALIFYING")]

    assert qualifying["xg_analysis_eligible"].astype(bool).mean() < 0.30


def test_phase_classification_matches_the_round_names() -> None:
    events = pd.DataFrame(
        {
            "match_id": ["a", "b", "c", "d"],
            "season": ["2024/25"] * 4,
            "competition": ["UCL"] * 4,
            "round": [
                "1st Qualifying Round",
                "Qualifying Play-off Round",
                "League Stage",
                "Final",
            ],
        }
    )

    context = match_context(events)

    assert context["phase"].tolist() == [
        "QUALIFYING",
        "QUALIFYING",
        "MAIN",
        "MAIN",
    ]


def test_every_qualifying_round_name_is_classified() -> None:
    events = pd.DataFrame(
        {
            "match_id": list(QUALIFYING_ROUNDS),
            "season": ["2024/25"] * len(QUALIFYING_ROUNDS),
            "competition": ["UEL"] * len(QUALIFYING_ROUNDS),
            "round": sorted(QUALIFYING_ROUNDS),
        }
    )

    assert match_context(events)["phase"].eq("QUALIFYING").all()


# ---------------------------------------------------------------------------
# the backtest gates
# ---------------------------------------------------------------------------


@requires_dataset
def test_gates_reject_a_map_that_is_not_a_superset() -> None:
    context = match_context(
        pd.DataFrame(
            {
                "match_id": ["x"],
                "season": ["2025/26"],
                "competition": ["UCL"],
                "round": ["League Stage"],
            }
        )
    )
    wide_map = {"a": (1.0, 1.0)}
    legacy_map = {"a": (1.0, 1.0), "b": (2.0, 2.0)}

    gates = validation_gates(wide_map, legacy_map, WIDE, context)
    superset = gates.loc[gates["gate"].eq("wide_superset_of_legacy")].iloc[0]

    assert not bool(superset["passed"])


@requires_dataset
def test_gates_reject_changed_shared_values() -> None:
    context = match_context(
        pd.DataFrame(
            {
                "match_id": ["x"],
                "season": ["2025/26"],
                "competition": ["UCL"],
                "round": ["League Stage"],
            }
        )
    )
    wide_map = {"a": (1.0, 1.0), "b": (2.0, 2.0)}
    legacy_map = {"a": (9.9, 1.0)}

    gates = validation_gates(wide_map, legacy_map, WIDE, context)
    shared = gates.loc[gates["gate"].eq("shared_values_identical")].iloc[0]

    assert not bool(shared["passed"])


# ---------------------------------------------------------------------------
# contract regression: the revalidation may strengthen the evidence record but
# must never move the layer it describes
# ---------------------------------------------------------------------------


import json

CONTRACT = ROOT / "contracts" / "ao_european_elo_v2_production.json"
SHIPPED_LAYER = {
    "active": True,
    "family": "BOUNDED_TWO_SIDED_PERFORMANCE_ADJUSTMENT",
    "max_xg_ratio": 0.3,
    "xg_scale": 1.25,
    "minimum_winner_gain_ratio": 0.7,
    "requires_both_teams": True,
    "missing_xg_behavior": "FALL_BACK_TO_GOAL_MARGIN_ONLY",
    "draw_behavior": "NO_XG_ADJUSTMENT",
    "penalty_shootout_behavior": "NO_XG_ADJUSTMENT",
    "zero_sum": True,
}


@pytest.fixture(name="contract", scope="module")
def fixture_contract() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_layer_parameters_are_untouched_by_the_revalidation(contract: dict) -> None:
    """Evidence widened; behaviour must be byte-identical to what shipped."""
    layer = contract["xg_performance"]

    for key, value in SHIPPED_LAYER.items():
        assert layer[key] == value, f"xg_performance.{key} changed"


def test_evidence_records_the_wider_sample(contract: dict) -> None:
    evidence = contract["xg_performance_evidence"]

    assert evidence["status"] == "REVALIDATED_ON_SIX_SEASONS"
    assert evidence["evaluation_window"] == "2020/21-2025/26"
    assert evidence["xg_eligible_matches"] > 4 * 606
    assert evidence["layer_parameters_changed"] is False


@requires_dataset
def test_evidence_counts_match_the_dataset(
    contract: dict, wide: pd.DataFrame
) -> None:
    """A stale evidence block is worse than none: pin it to the real file."""
    evidence = contract["xg_performance_evidence"]
    eligible = int(wide["xg_analysis_eligible"].astype(bool).sum())

    assert evidence["scored_matches"] == len(wide)
    assert evidence["xg_eligible_matches"] == eligible
    assert evidence["missing_xg_fallback_matches"] == len(wide) - eligible


def test_evidence_claims_are_backed_by_a_reliable_envelope(contract: dict) -> None:
    evidence = contract["xg_performance_evidence"]
    envelope = evidence["conservative_envelope"]

    assert evidence["dependency_reliable_improvement"] is True
    assert evidence["dependency_reliable_harm"] is False
    for key in ("all_brier", "main_stage_brier", "xg_present_brier", "all_log_loss"):
        block = envelope[key]
        assert block["reliable_improvement"] is True
        assert block["ci_95"][1] < 0.0, f"{key} interval must exclude zero"


def test_original_activation_record_is_preserved(contract: dict) -> None:
    """Provenance: the layer was first activated on a manual product decision."""
    superseded = contract["xg_performance_evidence"]["superseded_evidence"]

    assert superseded["full_season_matches"] == 961
    assert superseded["xg_eligible_matches"] == 606
    assert superseded["manual_product_decision"] is True
