from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_v2_evaluation_upgrade import (  # noqa: E402
    COMPETITIONS,
    DRAW_SHAPE_CANDIDATES,
    apply_fold_draw_models,
    draw_model_candidates,
    probability_output_decision,
    select_draw_models,
    validate_fold_contract,
)


def _prediction_frame() -> pd.DataFrame:
    rows = []
    for competition in COMPETITIONS:
        for index, (expected, home_goals, away_goals) in enumerate(
            [(0.65, 2, 0), (0.50, 1, 1), (0.35, 0, 1)] * 8
        ):
            rows.append(
                {
                    "fold": 1,
                    "competition": competition,
                    "match_id": f"{competition}-{index}",
                    "expected_home_score": expected,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                }
            )
    return pd.DataFrame(rows)


def test_draw_candidate_grid_is_explicit_and_complete() -> None:
    candidates = draw_model_candidates()

    assert len(candidates) == 143
    assert len(set(candidates)) == len(candidates)
    assert min(candidate.draw_at_even for candidate in candidates) == pytest.approx(0.18)
    assert max(candidate.draw_at_even for candidate in candidates) == pytest.approx(0.38)


def test_draw_selection_returns_one_training_winner_per_competition() -> None:
    selected, metrics = select_draw_models(_prediction_frame())

    assert selected["competition"].tolist() == ["ALL", *sorted(COMPETITIONS)]
    assert len(metrics) == len(draw_model_candidates()) * (len(COMPETITIONS) + 1)
    assert selected["draw_at_even"].between(0.18, 0.38).all()
    assert selected["draw_shape"].isin(DRAW_SHAPE_CANDIDATES).all()


def test_fold_draw_application_covers_every_row_and_preserves_expected_score() -> None:
    predictions = _prediction_frame()
    selected, _ = select_draw_models(predictions)
    selected.insert(0, "fold", 1)

    evaluated = apply_fold_draw_models(
        predictions,
        selected,
        expected_column="expected_home_score",
        prefix="ao",
    )

    reconstructed = (
        evaluated["ao_home_probability"]
        + 0.5 * evaluated["ao_draw_probability"]
    )
    assert np.allclose(reconstructed, evaluated["expected_home_score"])
    assert evaluated[
        [
            "ao_home_probability",
            "ao_draw_probability",
            "ao_away_probability",
            "ao_brier_1x2",
            "ao_log_loss_1x2",
        ]
    ].notna().all().all()


def test_fold_draw_application_rejects_missing_mapping() -> None:
    predictions = _prediction_frame()
    selected, _ = select_draw_models(predictions)
    selected = selected.loc[~selected["competition"].isin(("ALL", "UEL"))].copy()
    selected.insert(0, "fold", 1)

    with pytest.raises(ValueError, match="missing"):
        apply_fold_draw_models(
            predictions,
            selected,
            expected_column="expected_home_score",
            prefix="ao",
        )


def test_probability_decision_requires_both_losses_in_every_segment() -> None:
    summary = pd.DataFrame(
        {
            "competition": ["ALL", "UCL", "UEL", "UECL"],
            "brier_skill_difference": [-0.02, -0.01, -0.01, -0.01],
            "log_loss_skill_difference": [-0.03, -0.01, -0.01, -0.01],
        }
    )

    assert probability_output_decision(summary) == "PROMOTE_1X2_OUTPUT"
    summary.loc[summary["competition"].eq("UEL"), "log_loss_skill_difference"] = 0.001
    assert probability_output_decision(summary) == "HOLD_1X2_OUTPUT"


def test_validate_fold_contract_rejects_season_mismatch() -> None:
    folds = (
        (("2020/21", "2021/22"), "2022/23"),
        (("2020/21", "2021/22", "2022/23"), "2023/24"),
    )
    rows = [
        {
            "fold": 1,
            "train_seasons": "2020/21|2021/22",
            "test_season": "2022/23",
        },
        {
            "fold": 2,
            "train_seasons": "2020/21|2021/22|2022/23",
            "test_season": "2023/24",
        },
    ]
    core = pd.DataFrame(rows)
    carry = pd.DataFrame(rows)
    carry.loc[1, "test_season"] = "2099/00"

    with pytest.raises(ValueError, match="season contract"):
        validate_fold_contract(core, carry, folds)
