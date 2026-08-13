from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_draw_shape_backtest import (
    BASELINE_DRAW,
    DRAW_CANDIDATES,
    SHAPE_CANDIDATES,
    DrawCandidate,
    candidate_grid,
)


def test_draw_shape_grid_contains_subunit_tail_candidates() -> None:
    candidates = candidate_grid()

    assert len(candidates) == len(DRAW_CANDIDATES) * len(SHAPE_CANDIDATES)
    assert len(set(candidates)) == len(candidates)
    assert DrawCandidate(BASELINE_DRAW, 1.0) in candidates
    assert DrawCandidate(BASELINE_DRAW, 0.84) in candidates
    assert min(candidate.draw_shape for candidate in candidates) == pytest.approx(0.50)
