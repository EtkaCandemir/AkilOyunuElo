from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scripts.run_qualification_stage_k_carry_backtest import REFERENCE, transition_grid


def test_joint_grid_has_reference_and_eleven_selectable_candidates() -> None:
    configs = transition_grid()
    assert len(configs) == 12
    assert configs[0].profile == REFERENCE
    assert configs[0].selectable is False
    assert sum(config.selectable for config in configs) == 11


def test_joint_grid_covers_stage_only_carry_only_and_joint_models() -> None:
    configs = {config.profile: config for config in transition_grid()}
    assert configs["FULL_CARRY_050"].stage_k.selectable is False
    assert configs["MILD_CARRY_100"].qualifier_carry == 1.0
    assert configs["MILD_CARRY_050"].stage_k.q1_multiplier == 0.70
    assert configs["BALANCED_CARRY_050"].stage_k.q1_multiplier == 0.60
