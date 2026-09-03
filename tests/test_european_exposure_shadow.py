from __future__ import annotations

import math

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ao_elo.european_exposure_shadow import (
    POSITIVE_BRIDGE_ARM_ID,
    apply_positive_bridge_seed_transform,
)


def seed_frame() -> pd.DataFrame:
    domestic = [1200.0, 1200.0, 1200.0, 1200.0, 1400.0, 1400.0]
    european = [1600.0, 1600.0, 1600.0, 1600.0, 1400.0, 1300.0]
    exposure = [0.0, 0.20, 0.649, 0.65, 0.30, 0.30]
    ao_first = [
        d + w * (e - d)
        for d, e, w in zip(domestic, european, exposure, strict=True)
    ]
    return pd.DataFrame(
        {
            "season": ["2026/27"] * len(domestic),
            "team_id": [f"TEAM-{index}" for index in range(len(domestic))],
            "team_name": [f"Team {index}" for index in range(len(domestic))],
            "adjusted_domestic_prior": domestic,
            "european_prior": european,
            "effective_european_exposure": exposure,
            "ao_first_elo": ao_first,
            "unrelated_audit_column": range(len(domestic)),
        }
    )


def test_positive_bridge_changes_only_the_frozen_eligible_region() -> None:
    source = seed_frame()
    result = apply_positive_bridge_seed_transform(source)

    assert result["positive_bridge_applied"].tolist() == [
        False,
        True,
        True,
        False,
        False,
        False,
    ]
    assert result["shadow_effective_european_exposure"].tolist() == pytest.approx(
        [0.0, 0.29, 0.6492, 0.65, 0.30, 0.30]
    )
    assert result.loc[1, "ao_first_elo"] == pytest.approx(1316.0)
    assert result.loc[2, "ao_first_elo"] == pytest.approx(1459.68)

    inactive = ~result["positive_bridge_applied"]
    assert result.loc[inactive, "ao_first_elo"].equals(
        source.loc[inactive, "ao_first_elo"]
    )
    assert (result.loc[result["positive_bridge_applied"], "positive_bridge_elo_delta"] > 0).all()
    assert result["shadow_effective_european_exposure"].max() <= 0.65


def test_transform_preserves_the_source_and_adds_complete_audit_columns() -> None:
    source = seed_frame()
    before = source.copy(deep=True)

    result = apply_positive_bridge_seed_transform(source)

    assert_frame_equal(source, before, check_exact=True)
    assert result is not source
    assert result["production_ao_first_elo"].tolist() == pytest.approx(
        before["ao_first_elo"].tolist()
    )
    assert result["positive_bridge_arm_id"].eq(POSITIVE_BRIDGE_ARM_ID).all()
    assert result["positive_bridge_eta"].eq(0.20).all()
    assert result["positive_bridge_exposure_cap"].eq(0.65).all()
    assert result["unrelated_audit_column"].equals(before["unrelated_audit_column"])


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("adjusted_domestic_prior", math.nan),
        ("european_prior", math.inf),
        ("effective_european_exposure", -math.inf),
        ("ao_first_elo", math.nan),
    ],
)
def test_transform_rejects_non_finite_numeric_inputs(column: str, value: float) -> None:
    source = seed_frame()
    source.loc[1, column] = value

    with pytest.raises(ValueError, match=rf"{column} must be finite.*TEAM-1"):
        apply_positive_bridge_seed_transform(source)


@pytest.mark.parametrize("exposure", [-0.001, 0.650001])
def test_transform_rejects_exposure_outside_the_active_range(exposure: float) -> None:
    source = seed_frame()
    source.loc[1, "effective_european_exposure"] = exposure
    source.loc[1, "ao_first_elo"] = 1200.0 + exposure * 400.0

    with pytest.raises(ValueError, match=r"within \[0, 0.65\].*TEAM-1"):
        apply_positive_bridge_seed_transform(source)


def test_transform_rejects_duplicate_or_empty_team_ids() -> None:
    duplicate = seed_frame()
    duplicate.loc[1, "team_id"] = " TEAM-0 "
    with pytest.raises(ValueError, match="duplicate team_id.*TEAM-0"):
        apply_positive_bridge_seed_transform(duplicate)

    empty = seed_frame()
    empty.loc[1, "team_id"] = " "
    with pytest.raises(ValueError, match="team_id values must be non-empty"):
        apply_positive_bridge_seed_transform(empty)


def test_transform_rejects_a_missing_or_duplicate_required_column() -> None:
    missing = seed_frame().drop(columns="european_prior")
    with pytest.raises(ValueError, match="missing columns.*european_prior"):
        apply_positive_bridge_seed_transform(missing)

    duplicate = pd.concat(
        [seed_frame(), seed_frame()[["european_prior"]]], axis=1
    )
    with pytest.raises(ValueError, match="duplicate columns.*european_prior"):
        apply_positive_bridge_seed_transform(duplicate)


def test_transform_rejects_a_production_ao_formula_mismatch() -> None:
    source = seed_frame()
    source.loc[1, "ao_first_elo"] += 1e-6

    with pytest.raises(ValueError, match=r"active AO blend.*TEAM-1"):
        apply_positive_bridge_seed_transform(source)


def test_consistency_tolerance_does_not_recompute_an_ineligible_row() -> None:
    source = seed_frame()
    source.loc[0, "ao_first_elo"] += 5e-10

    result = apply_positive_bridge_seed_transform(source)

    assert result.loc[0, "ao_first_elo"] == source.loc[0, "ao_first_elo"]
    assert not result.loc[0, "positive_bridge_applied"]
