from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_domestic_surprise_weak_league_partial_pooling import (  # noqa: E402
    PoolingCandidate,
    build_pooled_adjustments,
    candidate_grid,
    context_segment,
    learned_multipliers,
)


def test_grid_covers_global_and_targeted_scopes() -> None:
    candidates = candidate_grid()
    assert len(candidates) == 144
    assert {candidate.scope for candidate in candidates} == {"GLOBAL_BASE", "TARGETED_ONLY"}
    assert {candidate.parameter_count for candidate in candidates} == {6}
    assert len({candidate.key for candidate in candidates}) == len(candidates)


def test_context_segments_are_deterministic() -> None:
    frame = pd.DataFrame({
        "competition": ["UECL", "UEL", "UECL", "UCL", "UEL"],
        "league_quintile": ["Q1", "Q1", "Q3", "Q5", "Q3"],
    })
    assert context_segment(frame).tolist() == [
        "Q1_UECL", "Q1_OTHER", "MID_UECL", "Q5_OTHER", "MID_OTHER"
    ]


def test_learned_multiplier_uses_only_previous_seasons() -> None:
    evidence = pd.DataFrame({
        "season": ["2019/20", "2019/20", "2022/23"],
        "segment": ["Q1_UECL", "MID_OTHER", "Q1_UECL"],
        "rank_success": [0.0, 1.0, 1.0],
        "loss_success": [0.0, 1.0, 1.0],
    })
    candidate = PoolingCandidate("test", "TARGETED_ONLY", 20.0, 0.25, 1.5, 60.0, 0.5)
    multipliers, rows = learned_multipliers(evidence, "2021/22", candidate)
    assert multipliers["Q1_UECL"] < 1.0
    q1 = next(row for row in rows if row["segment"] == "Q1_UECL")
    assert q1["history_team_seasons"] == 1


def test_targeted_scope_leaves_mid_other_at_current() -> None:
    base = pd.DataFrame({
        "season": ["2021/22", "2021/22"],
        "team_id": [1, 2],
        "competition": ["UEL", "UECL"],
        "league_quintile": ["Q3", "Q1"],
        "ao_first_elo_adjustment": [20.0, 20.0],
        "baseline_ao_first_elo": [1000.0, 1000.0],
    })
    current = pd.DataFrame({
        "season": ["2021/22", "2021/22"],
        "team_id": [1, 2],
        "ao_first_elo_adjustment": [5.0, 5.0],
    })
    evidence = pd.DataFrame({
        "season": ["2020/21", "2020/21"],
        "segment": ["MID_OTHER", "Q1_UECL"],
        "rank_success": [1.0, 0.0],
        "loss_success": [1.0, 0.0],
    })
    candidate = PoolingCandidate("test", "TARGETED_ONLY", 20.0, 0.25, 1.5, 60.0, 0.5)
    result, _ = build_pooled_adjustments(pd.DataFrame(), current, base, evidence, candidate)
    result = result.set_index("team_id")
    assert result.loc[1, "ao_first_elo_adjustment"] == pytest.approx(5.0)
    assert 5.0 <= result.loc[2, "ao_first_elo_adjustment"] < 20.0
