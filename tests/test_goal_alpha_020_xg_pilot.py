from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_goal_alpha_020_xg_pilot import (  # noqa: E402
    run_comparison_pilot,
    validate_comparison_pilot,
)
from scripts.run_xg_performance_bonus_pilot import (  # noqa: E402
    DEFAULT_INPUT,
    read_scenarios,
)


def test_alpha_020_xg_pilot_preserves_core_invariants() -> None:
    results = run_comparison_pilot(read_scenarios(DEFAULT_INPUT))
    validate_comparison_pilot(results)
    decisive = results.loc[results["is_decisive"]]
    assert decisive["alpha_020_xg_winner_gain"].gt(0.0).all()
    assert results["alpha_020_zero_sum_error"].max() <= 1e-9


def test_alpha_020_increases_supported_multi_goal_win() -> None:
    results = run_comparison_pilot(read_scenarios(DEFAULT_INPUT)).set_index(
        "scenario_id"
    )
    supported = results.loc["BAL_4_0_SUPPORTED"]
    assert (
        supported["alpha_020_xg_winner_gain"]
        > supported["alpha_010_xg_winner_gain"]
    )


def test_one_goal_win_has_no_goal_alpha_uplift() -> None:
    results = run_comparison_pilot(read_scenarios(DEFAULT_INPUT)).set_index(
        "scenario_id"
    )
    one_goal = results.loc["BAL_1_0_SUPPORTED"]
    assert abs(one_goal["combined_gain_difference"]) <= 1e-12
