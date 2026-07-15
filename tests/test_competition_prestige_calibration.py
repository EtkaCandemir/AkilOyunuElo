from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_competition_prestige_calibration import (  # noqa: E402
    CompetitionPrestigeConfig,
    candidate_grid,
    prestige_delta,
    run_season,
)
from scripts.run_dynamic_core_calibration import DynamicCoreConfig, SeasonData  # noqa: E402


def test_reference_candidate_is_in_strict_hierarchy_grid() -> None:
    candidates = candidate_grid()

    assert CompetitionPrestigeConfig(6.0, 0.65, 0.45) in candidates
    assert all(
        candidate.ucl_prestige > candidate.uel_prestige > candidate.uecl_prestige
        for candidate in candidates
    )


def test_inverted_or_equal_prestige_hierarchy_is_rejected() -> None:
    with pytest.raises(ValueError, match="UCL > UEL > UECL"):
        CompetitionPrestigeConfig(6.0, 1.0, 0.45).validate()
    with pytest.raises(ValueError, match="at least 0.10"):
        CompetitionPrestigeConfig(6.0, 0.65, 0.60).validate()


def test_draw_receives_no_prestige_bonus() -> None:
    config = CompetitionPrestigeConfig(8.0, 0.65, 0.45)

    assert prestige_delta(0.5, 0.4, "UCL", config) == 0.0


def test_identical_win_has_strict_ucl_uel_uecl_bonus_order() -> None:
    config = CompetitionPrestigeConfig(8.0, 0.65, 0.45)

    deltas = [prestige_delta(1.0, 0.4, competition, config) for competition in ("UCL", "UEL", "UECL")]

    assert deltas[0] > deltas[1] > deltas[2] > 0


def test_prestige_update_remains_zero_sum() -> None:
    data = SeasonData(
        season="2024/25",
        initial_ratings=np.array([np.nan, 800.0, 790.0, 780.0]),
        active_team_ids=np.array([1, 2, 3]),
        home_team_ids=np.array([1, 2]),
        away_team_ids=np.array([2, 3]),
        actual_home_scores=np.array([1.0, 0.0]),
        neutral_flags=np.array([False, False]),
        competitions=np.array(["UCL", "UECL"]),
        match_ids=np.array(["m1", "m2"]),
    )
    metrics, predictions = run_season(
        data,
        DynamicCoreConfig(225, 40, 28),
        CompetitionPrestigeConfig(8.0, 0.65, 0.45),
        return_predictions=True,
    )

    assert metrics["mean_rating_change"] == pytest.approx(0.0, abs=1e-12)
    assert predictions is not None
    assert predictions.iloc[0]["prestige_delta"] > abs(predictions.iloc[1]["prestige_delta"])
