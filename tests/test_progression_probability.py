from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ao_elo.progression_probability import (  # noqa: E402
    ProgressionProbabilityConfig,
    calibrate_progression_probability,
)


def test_identity_config_preserves_probability() -> None:
    for probability in (0.1, 0.25, 0.5, 0.8):
        assert calibrate_progression_probability(
            probability,
            2,
            False,
            ProgressionProbabilityConfig(),
        ) == pytest.approx(probability)


def test_single_home_bias_only_applies_to_non_neutral_single_match() -> None:
    config = ProgressionProbabilityConfig(1.0, 0.3, 0.0)
    home = calibrate_progression_probability(0.5, 1, False, config)
    neutral = calibrate_progression_probability(0.5, 1, True, config)

    assert home > 0.5
    assert neutral == pytest.approx(0.5)


def test_two_leg_bias_is_oriented_to_first_leg_home_team() -> None:
    config = ProgressionProbabilityConfig(1.0, 0.0, -0.2)

    assert calibrate_progression_probability(0.5, 2, False, config) < 0.5


@pytest.mark.parametrize(
    ("probability", "matches", "neutral", "config"),
    [
        (-0.1, 2, False, ProgressionProbabilityConfig()),
        (1.1, 2, False, ProgressionProbabilityConfig()),
        (math.nan, 2, False, ProgressionProbabilityConfig()),
        (0.5, 0, False, ProgressionProbabilityConfig()),
        (0.5, 2, False, ProgressionProbabilityConfig(0.0, 0.0, 0.0)),
    ],
)
def test_invalid_progression_probability_inputs_are_rejected(
    probability: float,
    matches: int,
    neutral: bool,
    config: ProgressionProbabilityConfig,
) -> None:
    with pytest.raises(ValueError):
        calibrate_progression_probability(
            probability,
            matches,
            neutral,
            config,
        )
