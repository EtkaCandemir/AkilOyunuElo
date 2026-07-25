from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ao_elo.config import V2_RATING_MULTIPLIER  # noqa: E402
from scripts.run_dynamic_core_calibration import (  # noqa: E402
    DynamicCoreConfig,
    expected_home_score,
)
from scripts.run_v2_dynamic_calibration import (  # noqa: E402
    CARRY_BASELINE_NAME,
    CARRY_MODEL_NAME,
    CORE_BASELINE_NAME,
    CORE_MODEL_NAME,
    MAX_RATING_MOVE_GUARDRAIL,
    carry_promotion_decision,
    core_promotion_decision,
    scaled_core_grid,
)


def test_scaled_grid_preserves_full_v1_candidate_contract() -> None:
    grid = scaled_core_grid()

    assert len(grid) == 13 * 9 * 9
    assert DynamicCoreConfig(
        225.0 * V2_RATING_MULTIPLIER,
        40.0 * V2_RATING_MULTIPLIER,
        28.0 * V2_RATING_MULTIPLIER,
    ) in grid
    assert MAX_RATING_MOVE_GUARDRAIL == pytest.approx(
        200.0 * V2_RATING_MULTIPLIER
    )


def test_affine_scale_preserves_expected_score_and_update_ratio() -> None:
    v1 = DynamicCoreConfig(225.0, 40.0, 28.0)
    v2 = DynamicCoreConfig(
        225.0 * V2_RATING_MULTIPLIER,
        40.0 * V2_RATING_MULTIPLIER,
        28.0 * V2_RATING_MULTIPLIER,
    )
    v1_probability = expected_home_score(850.0, 700.0, v1, neutral=False)
    v2_probability = expected_home_score(
        500.0 + (850.0 - 500.0) * V2_RATING_MULTIPLIER,
        500.0 + (700.0 - 500.0) * V2_RATING_MULTIPLIER,
        v2,
        neutral=False,
    )

    assert v2_probability == pytest.approx(v1_probability)
    assert v2.k_factor * (1.0 - v2_probability) == pytest.approx(
        V2_RATING_MULTIPLIER * v1.k_factor * (1.0 - v1_probability)
    )


def test_core_requires_six_of_six_unseen_wins() -> None:
    rows = []
    for fold in range(1, 7):
        rows.extend(
            [
                {
                    "fold": fold,
                    "model": CORE_BASELINE_NAME,
                    "brier": 0.17,
                    "mean_rating_change": 0.0,
                    "start_end_rank_correlation": 1.0,
                    "max_abs_rating_change": 0.0,
                },
                {
                    "fold": fold,
                    "model": CORE_MODEL_NAME,
                    "brier": 0.16 if fold < 6 else 0.18,
                    "mean_rating_change": 0.0,
                    "start_end_rank_correlation": 0.95,
                    "max_abs_rating_change": 100.0,
                },
            ]
        )
    uncertainty = pd.DataFrame(
        [
            {
                "competition": competition,
                "reliable_improvement": competition == "ALL",
                "reliable_harm": False,
            }
            for competition in ("ALL", "UCL", "UEL", "UECL")
        ]
    )

    decision, guardrails = core_promotion_decision(pd.DataFrame(rows), uncertainty)

    assert decision == "REJECT_DYNAMIC_CORE"
    assert guardrails["fold_wins"] == 5


def test_carry_is_disabled_when_ucl_has_reliable_harm() -> None:
    rows = []
    for fold in range(1, 7):
        rows.extend(
            [
                {
                    "fold": fold,
                    "model": CARRY_BASELINE_NAME,
                    "brier": 0.17,
                    "start_end_rank_correlation": 0.95,
                    "max_abs_rating_change": 100.0,
                },
                {
                    "fold": fold,
                    "model": CARRY_MODEL_NAME,
                    "brier": 0.16,
                    "start_end_rank_correlation": 0.90,
                    "max_abs_rating_change": 150.0,
                },
            ]
        )
    uncertainty = pd.DataFrame(
        [
            {"competition": "ALL", "reliable_harm": False},
            {"competition": "UCL", "reliable_harm": True},
        ]
    )

    decision, guardrails = carry_promotion_decision(pd.DataFrame(rows), uncertainty)

    assert decision == "DISABLE_CARRY"
    assert guardrails["fold_wins"] == 6
    assert guardrails["ucl_reliable_harm"]
