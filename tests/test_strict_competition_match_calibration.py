from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_competition_multiplier_calibration import (  # noqa: E402
    CompetitionMultiplierConfig,
    run_competition_season,
)
from scripts.run_dynamic_core_calibration import DynamicCoreConfig, SeasonData  # noqa: E402
from scripts.run_strict_competition_match_calibration import (  # noqa: E402
    BASELINE,
    REFERENCE,
    build_effective_k_table,
    is_strict,
    strict_candidate_grid,
)


CORE = DynamicCoreConfig(225, 40, 28)


def make_matches() -> SeasonData:
    return SeasonData(
        season="2024/25",
        initial_ratings=np.array([np.nan, 800.0, 800.0, 800.0, 800.0]),
        active_team_ids=np.array([1, 2, 3, 4]),
        home_team_ids=np.array([1, 1, 1]),
        away_team_ids=np.array([2, 3, 4]),
        actual_home_scores=np.array([1.0, 1.0, 1.0]),
        neutral_flags=np.array([True, True, True]),
        competitions=np.array(["UCL", "UEL", "UECL"]),
        match_ids=np.array(["ucl", "uel", "uecl"]),
    )


def test_grid_contains_requested_reference_and_only_strict_candidates() -> None:
    candidates = strict_candidate_grid()

    assert REFERENCE in candidates
    assert all(is_strict(candidate) for candidate in candidates)
    assert BASELINE not in candidates


def test_equal_or_inverted_hierarchy_is_not_strict() -> None:
    assert not is_strict(CompetitionMultiplierConfig(1.0, 0.45))
    assert not is_strict(CompetitionMultiplierConfig(0.65, 0.65))
    assert not is_strict(CompetitionMultiplierConfig(0.60, 0.55))


def test_reference_produces_expected_effective_k_values() -> None:
    table = build_effective_k_table(CORE, REFERENCE, REFERENCE)

    assert table["reference_effective_k"].tolist() == pytest.approx([28.0, 18.2, 12.6])


def test_same_surprise_has_strict_ucl_uel_uecl_match_delta_order() -> None:
    _, predictions = run_competition_season(
        make_matches(),
        CORE,
        REFERENCE,
        return_predictions=True,
    )

    assert predictions is not None
    deltas = (
        CORE.k_factor
        * predictions["competition_multiplier"]
        * (predictions["actual_home_score"] - predictions["expected_home_score"])
    )
    assert deltas.iloc[0] > deltas.iloc[1] > deltas.iloc[2] > 0


def test_neutral_baseline_keeps_same_k_in_every_competition() -> None:
    table = build_effective_k_table(CORE, BASELINE, REFERENCE)

    assert table["strict_effective_k"].tolist() == [28.0, 28.0, 28.0]
