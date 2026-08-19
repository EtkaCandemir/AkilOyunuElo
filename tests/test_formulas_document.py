from __future__ import annotations

"""Pin the mathematical reference document to the active configuration.

`docs/ai/FORMULAS.md` sits third in the authority order defined by
`AGENTS.md`, which makes silent drift between it and the code a real hazard:
a reader cannot tell a stale constant from a current one. These tests fail the
moment a documented value stops matching the active config, and they also
catch the subtler defect of the document quoting one constant at two different
precisions.
"""

import re
from pathlib import Path

import pytest

from ao_elo.config import AOEuropeanEloConfig
from ao_elo.dynamic import DynamicEloConfig


ROOT = Path(__file__).resolve().parents[1]
FORMULAS = ROOT / "docs" / "ai" / "FORMULAS.md"


@pytest.fixture(name="document", scope="module")
def fixture_document() -> str:
    return FORMULAS.read_text(encoding="utf-8")


@pytest.fixture(name="dynamic", scope="module")
def fixture_dynamic() -> DynamicEloConfig:
    return DynamicEloConfig.calibrated_v2()


@pytest.fixture(name="static", scope="module")
def fixture_static() -> AOEuropeanEloConfig:
    return AOEuropeanEloConfig.active()


def documented_numbers(document: str, value: float) -> list[str]:
    """Return every literal in the document that starts with this value.

    Matching on the repr prefix finds truncations such as `103.9809863339`
    alongside the exact `103.98098633392752`.
    """
    exact = repr(float(value))
    stem = exact.rstrip("0").rstrip(".")
    pattern = re.compile(rf"(?<![\d.]){re.escape(stem[: stem.index('.') + 5])}\d*")
    return pattern.findall(document)


@pytest.mark.parametrize(
    "name",
    ["elo_scale", "home_advantage", "k_factor"],
)
def test_dynamic_core_constants_are_quoted_at_full_precision(
    document: str, dynamic: DynamicEloConfig, name: str
) -> None:
    value = float(getattr(dynamic, name))
    exact = repr(value)
    found = documented_numbers(document, value)

    assert found, f"{name} is not documented in FORMULAS.md"
    truncated = sorted({item for item in found if item != exact})
    assert not truncated, (
        f"{name} appears at mixed precision: {truncated} alongside {exact}"
    )


def test_v2_multiplier_matches_the_config(
    document: str, static: AOEuropeanEloConfig
) -> None:
    from ao_elo.config import V2_RATING_MULTIPLIER

    assert repr(V2_RATING_MULTIPLIER) in document


@pytest.mark.parametrize(
    "attribute",
    [
        "domestic_league_component",
        "domestic_achievement_component",
        "european_prior_max_boost",
    ],
)
def test_static_components_are_quoted_at_full_precision(
    document: str, static: AOEuropeanEloConfig, attribute: str
) -> None:
    value = float(getattr(static, attribute))
    found = documented_numbers(document, value)

    assert found, f"{attribute} is not documented in FORMULAS.md"
    assert set(found) == {repr(value)}


def test_qualifier_multipliers_match_the_active_config(
    document: str, dynamic: DynamicEloConfig
) -> None:
    """The effective column is what the engine actually multiplies K by."""
    expected = {
        "Q1": dynamic.qualification_q1_multiplier,
        "Q2": dynamic.qualification_q2_multiplier,
        "Q3": dynamic.qualification_q3_multiplier,
        "Qualifying Play-off": dynamic.qualification_playoff_multiplier,
    }
    for label, multiplier in expected.items():
        row = re.search(rf"^{re.escape(label)}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)$",
                        document, re.MULTILINE)

        assert row, f"{label} row is missing from the qualifier table"
        base, retention, effective = (float(group) for group in row.groups())
        assert effective == pytest.approx(multiplier)
        assert base * retention == pytest.approx(multiplier)


def test_draw_and_goal_parameters_match_the_active_config(
    document: str, dynamic: DynamicEloConfig
) -> None:
    for value in (
        dynamic.draw_at_even,
        dynamic.single_match_draw_at_even,
        dynamic.goal_alpha,
        dynamic.xg_performance_ratio,
        dynamic.xg_performance_scale,
        dynamic.minimum_winner_gain_ratio,
    ):
        assert f"{value:g}" in document or f"{value:.2f}" in document
