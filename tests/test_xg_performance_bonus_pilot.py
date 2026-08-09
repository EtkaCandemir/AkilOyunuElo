from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_xg_performance_bonus_pilot import (
    DEFAULT_INPUT,
    read_scenarios,
    run_pilot,
    validate_results,
)


def test_xg_performance_bonus_pilot_contract_and_invariants() -> None:
    scenarios = read_scenarios(DEFAULT_INPUT)
    results = run_pilot(scenarios)
    validate_results(results)
    assert len(results) == 21
    assert results.loc[results["is_decisive"], "candidate_winner_gain"].gt(0).all()
    assert results["zero_sum_error"].max() <= 1e-9


def test_supported_and_lucky_wins_move_in_expected_directions() -> None:
    results = run_pilot(read_scenarios(DEFAULT_INPUT)).set_index("scenario_id")
    equal = results.loc["BAL_1_0_EQUAL_XG"]
    supported = results.loc["BAL_1_0_SUPPORTED"]
    lucky = results.loc["BAL_1_0_LUCKY"]
    extreme = results.loc["BAL_1_0_EXTREME_LUCKY"]
    assert supported["candidate_winner_gain"] > equal["candidate_winner_gain"]
    assert lucky["candidate_winner_gain"] < equal["candidate_winner_gain"]
    assert extreme["winner_floor_activated"]
    assert extreme["candidate_winner_gain"] > 0.0


def test_goal_margin_and_xg_effects_remain_separate() -> None:
    results = run_pilot(read_scenarios(DEFAULT_INPUT)).set_index("scenario_id")
    one_goal = results.loc["BAL_1_0_EQUAL_XG"]
    two_goal = results.loc["BAL_2_0_EQUAL_XG"]
    assert one_goal["xg_effect_on_winner_vs_gd"] == 0.0
    assert two_goal["xg_effect_on_winner_vs_gd"] == 0.0
    assert two_goal["goal_difference_bonus_on_winner"] > 0.0
