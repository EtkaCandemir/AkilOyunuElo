from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_fotmob_xg_backtest_2025_26 import read_fotmob_xg_dataset
from scripts.run_unsupported_margin_backtest_2025_26 import (
    BASELINE_KEY,
    LEGACY_KEY,
    VALIDATION_START,
    candidate_grid,
    select_development_candidate,
)
from scripts.run_xg_goal_bonus_guard_backtest_2025_26 import (
    goal_bonus_guard_grid,
)


DATA_PATH = ROOT / "data" / "xg_2025_26" / "uefa_2025_26_matches_with_xg.csv"


def test_unsupported_margin_grid_has_controls_and_36_candidates() -> None:
    candidates = candidate_grid()
    assert len(candidates) == 38
    assert candidates[0].key == BASELINE_KEY
    assert candidates[1].key == LEGACY_KEY
    unsupported = [
        candidate for candidate in candidates if candidate.kind == "UNSUPPORTED_MARGIN"
    ]
    assert len(unsupported) == 36
    assert {candidate.tolerance for candidate in unsupported} == {0.50, 0.75, 1.00}
    assert {candidate.penalty_lambda for candidate in unsupported} == {
        0.05,
        0.10,
        0.15,
        0.20,
    }
    assert {candidate.minimum_multiplier for candidate in unsupported} == {
        0.70,
        0.80,
        0.90,
    }


def test_temporal_validation_is_fully_xg_covered() -> None:
    events = read_fotmob_xg_dataset(DATA_PATH, strict_contract=True)
    development = events.loc[events["kickoff_utc"].lt(VALIDATION_START)]
    validation = events.loc[events["kickoff_utc"].ge(VALIDATION_START)]
    assert len(development) == 754
    assert int(development["xg_analysis_eligible"].sum()) == 399
    assert len(validation) == 207
    assert validation["xg_analysis_eligible"].all()
    assert validation["competition"].value_counts().to_dict() == {
        "UCL": 81,
        "UEL": 81,
        "UECL": 45,
    }


def test_goal_bonus_guard_grid_has_48_scoped_candidates() -> None:
    candidates = goal_bonus_guard_grid()
    assert len(candidates) == 50
    guards = [candidate for candidate in candidates if candidate.kind == "GOAL_BONUS_GUARD"]
    assert len(guards) == 48
    assert {candidate.config.application_scope for candidate in guards} == {
        "GOAL_BONUS_ONLY"
    }
    assert {candidate.minimum_multiplier for candidate in guards} == {
        0.0,
        0.25,
        0.50,
        0.75,
    }


def test_selection_uses_ranking_safe_development_pool_only() -> None:
    rows = [
        {
            "candidate_key": BASELINE_KEY,
            "candidate_kind": "PRODUCTION",
            "tolerance": 0.0,
            "penalty_lambda": 0.0,
            "minimum_multiplier": 1.0,
            "brier_1x2": 0.60,
            "log_loss_1x2": 1.00,
            "ranking_score": 0.70,
            "pairwise_accuracy": 0.75,
        },
        {
            "candidate_key": "UNSUPPORTED_tol0.5_lambda0.2_min0.7",
            "candidate_kind": "UNSUPPORTED_MARGIN",
            "tolerance": 0.5,
            "penalty_lambda": 0.2,
            "minimum_multiplier": 0.7,
            "brier_1x2": 0.58,
            "log_loss_1x2": 0.98,
            "ranking_score": 0.69,
            "pairwise_accuracy": 0.74,
        },
        {
            "candidate_key": "UNSUPPORTED_tol1_lambda0.05_min0.9",
            "candidate_kind": "UNSUPPORTED_MARGIN",
            "tolerance": 1.0,
            "penalty_lambda": 0.05,
            "minimum_multiplier": 0.9,
            "brier_1x2": 0.59,
            "log_loss_1x2": 0.99,
            "ranking_score": 0.71,
            "pairwise_accuracy": 0.76,
        },
    ]
    selected, audit = select_development_candidate(pd.DataFrame(rows))
    assert selected.key == "UNSUPPORTED_tol1_lambda0.05_min0.9"
    assert audit["selection_pool"] == "RANKING_SAFE"
    assert audit["validation_metrics_used_for_selection"] is False
