from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from scripts.run_2025_26_final_shadow_replay import (
    COMBINED_ARM,
    MAIN_ARM,
    PROGRESSION_ARM,
    SURPRISE_ARM,
    coerce_bool,
    normalize_surprise_input,
    replay_arms,
    surprise_manifest_parameters,
    stable_rank,
)


def test_replay_arms_keep_shadow_features_out_of_main() -> None:
    arms = {arm.key: arm for arm in replay_arms()}

    assert set(arms) == {MAIN_ARM, SURPRISE_ARM, PROGRESSION_ARM, COMBINED_ARM}
    assert not arms[MAIN_ARM].surprise_enabled
    assert not arms[MAIN_ARM].progression_enabled
    assert arms[SURPRISE_ARM].surprise_enabled
    assert not arms[SURPRISE_ARM].progression_enabled
    assert not arms[PROGRESSION_ARM].surprise_enabled
    assert arms[PROGRESSION_ARM].progression_enabled
    assert arms[COMBINED_ARM].surprise_enabled
    assert arms[COMBINED_ARM].progression_enabled
    assert arms[SURPRISE_ARM].status == "SHADOW_ONLY"
    assert arms[PROGRESSION_ARM].status == "SHADOW_ONLY"


def test_stable_rank_uses_club_id_as_deterministic_tie_breaker() -> None:
    frame = pd.DataFrame(
        {
            "club_id": ["AO-UEFA-3", "AO-UEFA-1", "AO-UEFA-2"],
            "rating": [1000.0, 1000.0, 900.0],
        }
    )

    ranks = stable_rank(frame, "rating", tie_column="club_id")

    assert ranks.tolist() == [2, 1, 3]


def test_coerce_bool_accepts_only_controlled_values() -> None:
    values = pd.Series([True, False, 1, 0, "true", "false", "1", "0"])

    assert coerce_bool(values, "flag").tolist() == [
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
    ]

    with pytest.raises(ValueError, match="invalid boolean"):
        coerce_bool(pd.Series(["yes"]), "flag")


def test_variance_surprise_input_maps_to_replay_contract() -> None:
    frame = pd.DataFrame(
        {
            "team_id": [1],
            "club_id": ["AO-UEFA-1"],
            "historical_mean": [0.5],
            "history_seasons": [5],
            "consistency_multiplier": [0.8],
            "raw_surprise": [0.2],
            "surprise_direction": ["POSITIVE"],
            "domestic_prior_adjustment": [30.0],
            "ao_first_elo_adjustment": [12.0],
            "adjusted_ao_first_elo": [1512.0],
        }
    )

    result = normalize_surprise_input(frame)

    assert result.loc[0, "historical_finish_score"] == pytest.approx(0.5)
    assert result.loc[0, "history_reliability"] == pytest.approx(0.8)
    assert result.loc[0, "surprise_score"] == pytest.approx(0.2)
    assert result.loc[0, "surprise_component"] == pytest.approx(30.0)


def test_gamma_decision_is_serialized_into_replay_manifest() -> None:
    result = surprise_manifest_parameters(
        {
            "selected_gamma": 0.5,
            "fixed_parameters": {
                "coefficient": 0.4,
                "max_abs_adjustment": 30.0,
                "minimum_history_seasons": 5,
                "volatility_normalization": "min(1,2*volatility)",
            },
        }
    )

    assert result == {
        "coefficient": 0.4,
        "variance_penalty": 0.5,
        "max_abs_adjustment": 30.0,
        "minimum_history_seasons": 5,
        "volatility_normalization": "min(1,2*volatility)",
    }
