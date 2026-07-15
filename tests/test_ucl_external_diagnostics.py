from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_ucl_external_diagnostics import (  # noqa: E402
    diagnostic_decision,
    rank_discrepancies,
)


def test_rank_discrepancies_preserve_direction() -> None:
    ucl = pd.DataFrame(
        {
            "test_season": ["2024/25", "2024/25"],
            "home_team_name": ["AO High", "AO Low"],
            "away_team_name": ["AO Low", "AO High"],
            "dynamic_carry_085_current_home_power": [900.0, 700.0],
            "dynamic_carry_085_current_away_power": [700.0, 900.0],
            "clubelo_home_elo": [1600.0, 1900.0],
            "clubelo_away_elo": [1900.0, 1600.0],
        }
    )

    result = rank_discrepancies(ucl)
    high = result.loc[result["team_name"].eq("AO High")].iloc[0]

    assert high["ao_minus_clubelo_rank_percentile"] > 0
    assert high["absolute_rank_percentile_gap"] == 0.5


def test_reliable_carry_harm_blocks_current_ucl_carry() -> None:
    summary = pd.DataFrame(
        [
            {"model": "dynamic_carry_085_current", "brier": 0.17},
            {"model": "clubelo_external", "brier": 0.16},
        ]
    )
    uncertainty = pd.DataFrame(
        [
            {
                "comparison": "dynamic_update",
                "reliable_harm": False,
            },
            {
                "comparison": "carry_085_vs_reset",
                "reliable_harm": True,
            },
        ]
    )
    guardrails = pd.DataFrame([{"guardrail_pass": True}])

    assert diagnostic_decision(summary, uncertainty, guardrails) == (
        "REVISE_OR_REMOVE_POWER_CARRY_FOR_UCL"
    )


def test_guardrail_failure_has_priority() -> None:
    assert diagnostic_decision(
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame([{"guardrail_pass": False}]),
    ) == "BLOCK_MODEL_FREEZE_GUARDRAIL_FAILURE"
