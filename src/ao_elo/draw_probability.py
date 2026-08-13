from __future__ import annotations

import math


def validate_draw_parameters(draw_at_even: float, draw_shape: float) -> None:
    """Validate the score-preserving draw-model parameters."""

    for name, value in (
        ("draw_at_even", draw_at_even),
        ("draw_shape", draw_shape),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"{name} must be finite numeric")
    if not 0.0 <= float(draw_at_even) <= 0.5:
        raise ValueError("draw_at_even must be in [0,0.5]")
    if float(draw_shape) <= 0.0:
        raise ValueError("draw_shape must be positive")


def score_preserving_draw_probability(
    expected_home_score: float,
    draw_at_even: float,
    draw_shape: float,
) -> float:
    """Return a valid draw probability while preserving Elo expected points.

    Shapes below one raise draw probability in imbalanced matches. The raw
    curve is bounded by the largest draw probability compatible with
    non-negative home and away win probabilities.
    """

    validate_draw_parameters(draw_at_even, draw_shape)
    if (
        isinstance(expected_home_score, bool)
        or not isinstance(expected_home_score, (int, float))
        or not math.isfinite(float(expected_home_score))
        or not 0.0 <= float(expected_home_score) <= 1.0
    ):
        raise ValueError("expected_home_score must be finite and in [0,1]")
    expected = float(expected_home_score)
    raw_draw = float(draw_at_even) * (
        4.0 * expected * (1.0 - expected)
    ) ** float(draw_shape)
    score_preserving_limit = 2.0 * min(expected, 1.0 - expected)
    return min(raw_draw, score_preserving_limit)


def score_preserving_1x2_scalar(
    expected_home_score: float,
    draw_at_even: float,
    draw_shape: float,
) -> tuple[float, float, float]:
    """Return home/draw/away probabilities preserving the expected score."""

    draw = score_preserving_draw_probability(
        expected_home_score,
        draw_at_even,
        draw_shape,
    )
    expected = float(expected_home_score)
    home = expected - 0.5 * draw
    away = 1.0 - expected - 0.5 * draw
    probabilities = tuple(max(0.0, value) for value in (home, draw, away))
    total = sum(probabilities)
    if abs(total - 1.0) > 1e-12:
        raise ValueError("Score-preserving 1X2 probabilities must sum to one")
    return probabilities
