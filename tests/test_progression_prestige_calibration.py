from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_dynamic_core_calibration import DynamicCoreConfig, SeasonData  # noqa: E402
from scripts.run_progression_prestige_calibration import (  # noqa: E402
    ProgressionPrestigeConfig,
    ProgressionSeasonData,
    candidate_grid,
    expected_to_advance,
    progression_delta,
    run_season,
    validate_tie_contract,
)


CORE = DynamicCoreConfig(225, 40, 28)


def make_data(
    *,
    actual_scores: tuple[float, ...] = (1.0, 0.5),
    tie_ids: tuple[object, ...] = ("tie-1", "tie-1"),
    deciders: tuple[bool, ...] = (False, True),
    advanced_ids: tuple[int, ...] = (-1, 1),
    home_ids: tuple[int, ...] = (1, 2),
    away_ids: tuple[int, ...] = (2, 1),
) -> ProgressionSeasonData:
    size = len(actual_scores)
    core = SeasonData(
        season="2024/25",
        initial_ratings=np.array([np.nan, 800.0, 800.0]),
        active_team_ids=np.array([1, 2]),
        home_team_ids=np.array(home_ids),
        away_team_ids=np.array(away_ids),
        actual_home_scores=np.array(actual_scores),
        neutral_flags=np.full(size, False),
        competitions=np.full(size, "UCL"),
        match_ids=np.array([f"m{index + 1}" for index in range(size)]),
    )
    return ProgressionSeasonData(
        core=core,
        tie_ids=np.array(tie_ids, dtype=object),
        leg_numbers=np.arange(1, size + 1),
        tie_decider_flags=np.array(deciders),
        knockout_flags=np.full(size, True),
        advanced_team_ids=np.array(advanced_ids),
    )


def test_equal_ratings_have_symmetric_advance_expectation_without_home_advantage() -> None:
    assert expected_to_advance(800, 800, CORE) == pytest.approx(0.5)


def test_grid_contains_reference_and_enforces_strict_hierarchy() -> None:
    candidates = candidate_grid()

    assert len(candidates) == 89
    assert ProgressionPrestigeConfig(0.5, 0.65, 0.45) in candidates
    assert all(
        candidate.ucl_prestige > candidate.uel_prestige > candidate.uecl_prestige
        for candidate in candidates
    )


def test_inverted_or_equal_hierarchy_is_rejected() -> None:
    with pytest.raises(ValueError, match="UCL > UEL > UECL"):
        ProgressionPrestigeConfig(0.5, 1.0, 0.45).validate()
    with pytest.raises(ValueError, match="at least 0.10"):
        ProgressionPrestigeConfig(0.5, 0.65, 0.60).validate()


def test_two_leg_tie_applies_one_zero_sum_progression_update_after_decider() -> None:
    metrics, predictions = run_season(
        make_data(),
        CORE,
        ProgressionPrestigeConfig(0.5, 0.65, 0.45),
        return_predictions=True,
    )

    assert predictions is not None
    assert predictions.iloc[0]["progression_delta"] == 0
    assert predictions.iloc[1]["progression_delta"] == pytest.approx(7.0)
    assert predictions.iloc[1]["expected_team_a_to_advance"] == pytest.approx(0.5)
    assert metrics["progression_events"] == 1
    assert metrics["mean_rating_change"] == pytest.approx(0.0, abs=1e-12)


def test_single_match_draw_can_award_progression_to_penalty_winner() -> None:
    data = make_data(
        actual_scores=(0.5,),
        tie_ids=("final",),
        deciders=(True,),
        advanced_ids=(2,),
        home_ids=(1,),
        away_ids=(2,),
    )
    _, predictions = run_season(
        data,
        CORE,
        ProgressionPrestigeConfig(0.5, 0.65, 0.45),
        return_predictions=True,
    )

    assert predictions is not None
    assert predictions.iloc[0]["match_delta"] < 0
    assert predictions.iloc[0]["progression_delta"] == pytest.approx(-7.0)


def test_zero_ratio_matches_core_predictions_exactly() -> None:
    _, baseline = run_season(
        make_data(),
        CORE,
        ProgressionPrestigeConfig(0.0, 0.65, 0.45),
        return_predictions=True,
    )
    _, alternate_coefficients = run_season(
        make_data(),
        CORE,
        ProgressionPrestigeConfig(0.0, 0.75, 0.55),
        return_predictions=True,
    )

    assert baseline is not None and alternate_coefficients is not None
    np.testing.assert_array_equal(
        baseline["expected_home_score"].to_numpy(),
        alternate_coefficients["expected_home_score"].to_numpy(),
    )
    assert baseline["progression_delta"].eq(0).all()


def test_progression_delta_respects_competition_order() -> None:
    config = ProgressionPrestigeConfig(0.5, 0.65, 0.45)
    deltas = [
        progression_delta(1.0, 0.4, competition, CORE, config)
        for competition in ("UCL", "UEL", "UECL")
    ]

    assert deltas[0] > deltas[1] > deltas[2] > 0


def test_tie_contract_rejects_missing_or_invalid_winner() -> None:
    events = pd.DataFrame(
        {
            "season": ["2024/25"],
            "tie_id": ["t1"],
            "is_knockout": [True],
            "is_tie_decider": [True],
            "home_team_id": [1],
            "away_team_id": [2],
            "advanced_team_id": [3],
        }
    )

    with pytest.raises(ValueError, match="advanced_team_id"):
        validate_tie_contract(events)
