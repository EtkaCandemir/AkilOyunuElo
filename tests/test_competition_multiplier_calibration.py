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
    candidate_grid,
    run_competition_season,
)
from scripts.run_dynamic_core_calibration import DynamicCoreConfig, SeasonData  # noqa: E402


def test_ucl_is_fixed_reference_and_other_multipliers_are_independent() -> None:
    config = CompetitionMultiplierConfig(uel_multiplier=0.875, uecl_multiplier=0.75)

    assert config.for_competition("UCL") == 1.0
    assert config.for_competition("UEL") == 0.875
    assert config.for_competition("UECL") == 0.75


def test_competition_candidate_grid_enforces_15_ordered_pairs() -> None:
    candidates = candidate_grid()

    assert len(candidates) == 15
    assert len(set(candidates)) == 15
    assert CompetitionMultiplierConfig(1.0, 1.0) in candidates
    assert all(
        candidate.ucl_multiplier >= candidate.uel_multiplier >= candidate.uecl_multiplier
        for candidate in candidates
    )


def test_competition_updates_remain_zero_sum() -> None:
    data = SeasonData(
        season="2024/25",
        initial_ratings=np.array([np.nan, 800.0, 790.0, 780.0]),
        active_team_ids=np.array([1, 2, 3]),
        home_team_ids=np.array([1, 2]),
        away_team_ids=np.array([2, 3]),
        actual_home_scores=np.array([1.0, 0.0]),
        neutral_flags=np.array([False, False]),
        competitions=np.array(["UEL", "UECL"]),
        match_ids=np.array(["m1", "m2"]),
    )
    metrics, predictions = run_competition_season(
        data,
        DynamicCoreConfig(225, 40, 28),
        CompetitionMultiplierConfig(0.875, 0.75),
        return_predictions=True,
    )

    assert metrics["mean_rating_change"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["max_abs_match_delta"] <= 28 * 0.875
    assert predictions is not None
    assert predictions["competition_multiplier"].tolist() == [0.875, 0.75]


def test_unknown_competition_is_rejected() -> None:
    config = CompetitionMultiplierConfig(1.0, 1.0)

    with pytest.raises(ValueError, match="Unknown competition"):
        config.for_competition("OTHER")


def test_inverted_competition_hierarchy_is_rejected() -> None:
    with pytest.raises(ValueError, match="UCL >= UEL >= UECL"):
        CompetitionMultiplierConfig(uel_multiplier=1.125, uecl_multiplier=0.75).validate()
