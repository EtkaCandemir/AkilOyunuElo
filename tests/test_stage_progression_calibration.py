from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_dynamic_core_calibration import DynamicCoreConfig, SeasonData  # noqa: E402
from scripts.run_progression_prestige_calibration import ProgressionSeasonData  # noqa: E402
from scripts.run_stage_progression_calibration import (  # noqa: E402
    STAGE_PROFILES,
    StageProfile,
    StageProgressionConfig,
    StageSeasonData,
    candidate_grid,
    normalize_stage,
    run_season,
    stage_progression_delta,
)


CORE = DynamicCoreConfig(225, 40, 28)


def make_single_tie(stage: str, round_name: str = "Final") -> StageSeasonData:
    core = SeasonData(
        season="2024/25",
        initial_ratings=np.array([np.nan, 800.0, 800.0]),
        active_team_ids=np.array([1, 2]),
        home_team_ids=np.array([1]),
        away_team_ids=np.array([2]),
        actual_home_scores=np.array([0.5]),
        neutral_flags=np.array([True]),
        competitions=np.array(["UCL"]),
        match_ids=np.array(["m1"]),
    )
    progression = ProgressionSeasonData(
        core=core,
        tie_ids=np.array(["t1"], dtype=object),
        leg_numbers=np.array([1]),
        tie_decider_flags=np.array([True]),
        knockout_flags=np.array([True]),
        advanced_team_ids=np.array([1]),
    )
    return StageSeasonData(
        progression=progression,
        rounds=np.array([round_name]),
        stages=np.array([stage]),
    )


@pytest.mark.parametrize(
    ("competition", "round_name", "is_knockout", "expected"),
    [
        ("UCL", "1st Qualifying Round", True, "QUALIFYING"),
        ("UCL", "Group Stage", False, "LEAGUE"),
        ("UCL", "Knockout round play-offs", True, "KNOCKOUT_PLAYOFF"),
        ("UCL", "Round 2", True, "ROUND_OF_16"),
        ("UEL", "Round 2", True, "KNOCKOUT_PLAYOFF"),
        ("UEL", "Round 3", True, "ROUND_OF_16"),
        ("UECL", "Quarter Finals", True, "QUARTERFINAL"),
        ("UCL", "Semi Finals", True, "SEMIFINAL"),
        ("UEL", "Final", True, "FINAL"),
    ],
)
def test_round_names_are_normalized_with_format_context(
    competition: str,
    round_name: str,
    is_knockout: bool,
    expected: str,
) -> None:
    assert normalize_stage(round_name, competition, is_knockout) == expected


def test_unknown_round_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown knockout round"):
        normalize_stage("Mystery Round", "UCL", True)


def test_all_profiles_are_monotonic_and_bounded() -> None:
    for profile in STAGE_PROFILES:
        profile.validate()
        assert tuple(sorted(profile.values())) == profile.values()
        assert max(profile.values()) <= 2


def test_decreasing_profile_is_rejected() -> None:
    profile = StageProfile("INVALID", 0, 1, 0.5, 1, 1.5)

    with pytest.raises(ValueError, match="non-decreasing"):
        profile.validate()


def test_candidate_grid_contains_one_baseline_and_thirty_positive_candidates() -> None:
    candidates = candidate_grid()

    assert len(candidates) == 31
    assert sum(candidate.progression_ratio == 0 for candidate in candidates) == 1
    assert StageProgressionConfig(0.125, "LATE_STRICT") in candidates


def test_late_balanced_profile_disables_qualifying_progression() -> None:
    config = StageProgressionConfig(0.5, "LATE_BALANCED")

    assert stage_progression_delta(1, 0.5, "UCL", "QUALIFYING", CORE, config) == 0


def test_identical_surprise_has_monotonic_stage_delta() -> None:
    config = StageProgressionConfig(0.5, "KNOCKOUT_ASCENDING")
    deltas = [
        stage_progression_delta(1, 0.4, "UCL", stage, CORE, config)
        for stage in (
            "QUALIFYING",
            "KNOCKOUT_PLAYOFF",
            "ROUND_OF_16",
            "QUARTERFINAL",
            "SEMIFINAL",
        )
    ]

    assert deltas == sorted(deltas)
    assert deltas[0] > 0


def test_stage_update_is_applied_once_and_remains_zero_sum() -> None:
    metrics, predictions = run_season(
        make_single_tie("SEMIFINAL", "Semi Finals"),
        CORE,
        StageProgressionConfig(0.5, "SEMIFINAL_HEAVY"),
        return_predictions=True,
    )

    assert predictions is not None
    assert predictions.iloc[0]["progression_delta"] == pytest.approx(10.5)
    assert metrics["progression_events"] == 1
    assert metrics["nonzero_progression_events"] == 1
    assert metrics["mean_rating_change"] == pytest.approx(0.0, abs=1e-12)


def test_final_progression_is_deferred_until_season_carry_can_test_it() -> None:
    _, predictions = run_season(
        make_single_tie("FINAL"),
        CORE,
        StageProgressionConfig(0.5, "SEMIFINAL_HEAVY"),
        return_predictions=True,
    )

    assert predictions is not None
    assert predictions.iloc[0]["progression_delta"] == 0


def test_zero_ratio_disables_every_stage_profile() -> None:
    _, predictions = run_season(
        make_single_tie("FINAL"),
        CORE,
        StageProgressionConfig(0.0, "SEMIFINAL_HEAVY"),
        return_predictions=True,
    )

    assert predictions is not None
    assert predictions.iloc[0]["progression_delta"] == 0
