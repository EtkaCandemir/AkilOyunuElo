from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_goal_alpha_015_xg_full_season_replay_2025_26 import (  # noqa: E402
    CANDIDATE_ALPHA,
    PRODUCTION_ALPHA,
    candidate_contract,
)


def test_candidate_contract_changes_only_goal_alpha() -> None:
    production = {
        "dynamic_core": {"k_factor": 100.0},
        "goal_margin": {
            "alpha": PRODUCTION_ALPHA,
            "tau": 300.0,
            "goal_difference_cap": 4,
        },
    }
    candidate = candidate_contract(production)
    assert candidate["goal_margin"]["alpha"] == CANDIDATE_ALPHA
    assert production["goal_margin"]["alpha"] == PRODUCTION_ALPHA
    assert candidate["goal_margin"]["tau"] == 300.0
    assert candidate["dynamic_core"] == production["dynamic_core"]


def test_candidate_contract_rejects_wrong_comparator() -> None:
    production = {"goal_margin": {"alpha": 0.20}}
    try:
        candidate_contract(production)
    except ValueError as exc:
        assert "goal_alpha=0.10" in str(exc)
    else:
        raise AssertionError("Wrong production alpha was accepted")
