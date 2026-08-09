from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_bounded_xg_adjustment_pilot import (
    run_bounded_pilot,
    validate_bounded_pilot,
)
from scripts.run_xg_performance_bonus_pilot import DEFAULT_INPUT, read_scenarios


def test_bounded_pilot_keeps_winner_in_controlled_range() -> None:
    results = run_bounded_pilot(read_scenarios(DEFAULT_INPUT))
    validate_bounded_pilot(results)
    covered = results.loc[results["is_decisive"] & results["xg_signal_home"].notna()]
    assert covered["candidate_gain_ratio_vs_classic"].min() >= 0.70 - 1e-12
    assert results["zero_sum_error"].max() <= 1e-9


def test_extreme_lucky_result_is_reduced_but_not_erased() -> None:
    results = run_bounded_pilot(read_scenarios(DEFAULT_INPUT)).set_index("scenario_id")
    extreme = results.loc["BAL_1_0_EXTREME_LUCKY"]
    assert extreme["candidate_winner_gain"] < extreme["classic_winner_gain"]
    assert extreme["candidate_winner_gain"] >= 0.70 * extreme["classic_winner_gain"]
