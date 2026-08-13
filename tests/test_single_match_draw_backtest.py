from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_single_match_draw_backtest import (
    EVENTS,
    PREDICTIONS,
    PRESPECIFIED_SINGLE_DRAW,
    evaluate_format_model,
    load_data,
)


def test_historical_format_classification_matches_frozen_schedule() -> None:
    data = load_data(PREDICTIONS, EVENTS)

    assert len(data) == 6340
    assert int(data["is_single_match_tie"].sum()) == 248
    assert int(data["is_two_legged_tie"].sum()) == 3716


def test_format_draw_only_changes_single_match_probabilities() -> None:
    data = load_data(PREDICTIONS, EVENTS)
    baseline = evaluate_format_model(data, 0.24)
    candidate = evaluate_format_model(data, PRESPECIFIED_SINGLE_DRAW)
    single = data["is_single_match_tie"]

    assert candidate.loc[single, "draw_probability"].mean() < baseline.loc[
        single, "draw_probability"
    ].mean()
    assert candidate.loc[~single, "draw_probability"].to_numpy() == pytest.approx(
        baseline.loc[~single, "draw_probability"].to_numpy()
    )
