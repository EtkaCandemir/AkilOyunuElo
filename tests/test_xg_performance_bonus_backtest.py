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
from scripts.run_xg_performance_bonus_walk_forward_2025_26 import (
    BASELINE_KEY,
    candidate_grid,
    outer_folds,
    select_candidate,
    validate_fold_contract,
)


DATA_PATH = ROOT / "data" / "xg_2025_26" / "uefa_2025_26_matches_with_xg.csv"


def test_performance_bonus_grid_has_200_xg_candidates_and_baseline() -> None:
    candidates = candidate_grid()
    assert len(candidates) == 201
    assert candidates[0].key == BASELINE_KEY
    xg_candidates = [candidate for candidate in candidates if not candidate.is_baseline]
    assert len(xg_candidates) == 200
    assert {candidate.beta for candidate in xg_candidates} == {
        0.05,
        0.10,
        0.20,
        0.30,
        0.50,
        0.75,
        1.00,
        1.25,
        1.50,
        2.00,
    }
    assert {candidate.xg_scale for candidate in xg_candidates} == {
        0.75,
        1.00,
        1.50,
        2.00,
        3.00,
    }
    assert {candidate.minimum_winner_gain_ratio for candidate in xg_candidates} == {
        0.05,
        0.10,
        0.25,
        0.50,
    }


def test_outer_folds_are_disjoint_and_fully_xg_covered() -> None:
    events = read_fotmob_xg_dataset(DATA_PATH, strict_contract=True)
    folds = outer_folds()
    validate_fold_contract(events, folds)
    counts = []
    match_ids: set[str] = set()
    for fold in folds:
        frame = events.loc[
            events["kickoff_utc"].ge(fold.test_start)
            & events["kickoff_utc"].lt(fold.test_end)
        ]
        counts.append(len(frame))
        assert frame["xg_analysis_eligible"].all()
        current = set(frame["match_id"].astype(str))
        assert match_ids.isdisjoint(current)
        match_ids.update(current)
    assert counts == [180, 72, 48, 48, 39]
    assert len(match_ids) == 387


def test_selection_rejects_loss_winner_that_breaks_ranking_guardrail() -> None:
    candidates = candidate_grid()
    safe_candidate = candidates[1]
    unsafe_candidate = candidates[-1]
    metrics = pd.DataFrame(
        [
            _metric_row(BASELINE_KEY, 0.0, 1.0, 1.0, 0.60, 1.00, 0.70, 0.75),
            _metric_row(
                safe_candidate.key,
                safe_candidate.beta,
                safe_candidate.xg_scale,
                safe_candidate.minimum_winner_gain_ratio,
                0.59,
                0.99,
                0.71,
                0.76,
            ),
            _metric_row(
                unsafe_candidate.key,
                unsafe_candidate.beta,
                unsafe_candidate.xg_scale,
                unsafe_candidate.minimum_winner_gain_ratio,
                0.50,
                0.80,
                0.69,
                0.74,
            ),
        ]
    )
    selected, audit = select_candidate(metrics, candidates)
    assert selected.key == safe_candidate.key
    assert audit["ranking_safe_candidates"] == 2
    assert audit["future_results_used_for_selection"] is False


def _metric_row(
    candidate_key: str,
    beta: float,
    xg_scale: float,
    floor: float,
    brier: float,
    log_loss: float,
    ranking: float,
    pairwise: float,
) -> dict[str, float | str]:
    return {
        "candidate_key": candidate_key,
        "beta": beta,
        "xg_scale": xg_scale,
        "minimum_winner_gain_ratio": floor,
        "brier_1x2": brier,
        "log_loss_1x2": log_loss,
        "ranking_score": ranking,
        "pairwise_accuracy": pairwise,
        "brier_1x2_delta_vs_production": brier - 0.60,
        "log_loss_1x2_delta_vs_production": log_loss - 1.00,
        "ranking_score_delta_vs_production": ranking - 0.70,
        "pairwise_accuracy_delta_vs_production": pairwise - 0.75,
    }
