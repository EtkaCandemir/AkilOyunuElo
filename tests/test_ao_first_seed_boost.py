from __future__ import annotations

import pandas as pd
import pytest

from ao_elo.ao_first_seed_boost import (
    SeedBoostConfig,
    apply_seed_boost,
    candidate_grid,
    season_relative_percentile,
)


def seed(**overrides) -> pd.DataFrame:
    row = {
        "season": "2025/26",
        "team_id": 1,
        "club_id": "AO-1",
        "current_direct_percentile": 0.60,
        "adjusted_domestic_prior": 1200.0,
        "european_prior": 900.0,
        "effective_european_exposure": 0.50,
        "adjusted_ao_first_elo": 1050.0,
        "domestic_form_covered": True,
        "domestic_form_percentile": 0.90,
    }
    row.update(overrides)
    return pd.DataFrame([row])


@pytest.mark.parametrize(
    ("design", "magnitude", "expected"),
    (("ADDITIVE", 50.0, 50.0), ("EXPOSURE_MODIFIER", 0.5, 75.0), ("FLOOR", 25.0, 125.0)),
)
def test_three_designs_target_the_same_negative_history_asymmetry(
    design: str, magnitude: float, expected: float
) -> None:
    config = SeedBoostConfig("BLIND", design, 0.65, 100.0, magnitude)

    result = apply_seed_boost(seed(), config).iloc[0]

    assert result.seed_boost_eligible
    assert result.seed_boost_applied == pytest.approx(expected)
    assert result.candidate_ao_first_elo == pytest.approx(1050.0 + expected)


def test_zero_exposure_is_never_treated_as_a_problem() -> None:
    config = SeedBoostConfig("BLIND", "ADDITIVE", 0.65, 100.0, 100.0)

    result = apply_seed_boost(
        seed(effective_european_exposure=0.0, adjusted_ao_first_elo=1200.0), config
    ).iloc[0]

    assert not result.seed_boost_eligible
    assert result.seed_boost_applied == 0.0


def test_unknown_finish_is_explicitly_ineligible() -> None:
    config = SeedBoostConfig("BLIND", "ADDITIVE", 0.65, 100.0, 100.0)

    result = apply_seed_boost(seed(current_direct_percentile=float("nan")), config).iloc[0]

    assert not result.seed_boost_eligible
    assert result.seed_boost_applied == 0.0


@pytest.mark.parametrize(
    "overrides",
    (
        {"current_direct_percentile": 0.80},
        {"european_prior": 1175.0},
        {"adjusted_ao_first_elo": 1225.0},
    ),
)
def test_blind_rule_requires_every_structural_condition(overrides) -> None:
    config = SeedBoostConfig("BLIND", "ADDITIVE", 0.65, 100.0, 100.0)

    result = apply_seed_boost(seed(**overrides), config).iloc[0]

    assert not result.seed_boost_eligible
    assert result.seed_boost_applied == 0.0


def test_domestic_form_is_an_incremental_gate_not_a_replacement_rule() -> None:
    config = SeedBoostConfig("DOMESTIC_FORM", "ADDITIVE", 0.65, 100.0, 100.0, 0.80)

    uncovered = apply_seed_boost(seed(domestic_form_covered=False), config).iloc[0]
    weak = apply_seed_boost(seed(domestic_form_percentile=0.79), config).iloc[0]
    strong = apply_seed_boost(seed(domestic_form_percentile=0.80), config).iloc[0]

    assert uncovered.seed_boost_applied == 0.0
    assert weak.seed_boost_applied == 0.0
    assert strong.seed_boost_applied == 100.0


def test_boost_is_direction_guarded_and_capped() -> None:
    config = SeedBoostConfig("BLIND", "FLOOR", 0.65, 50.0, 0.0, maximum_boost=150.0)

    result = apply_seed_boost(seed(adjusted_ao_first_elo=700.0), config).iloc[0]

    assert result.seed_boost_applied == 150.0
    assert result.seed_boost_cap_hit


def test_candidate_grids_are_complete_and_unique() -> None:
    blind = candidate_grid("BLIND")
    domestic = candidate_grid("DOMESTIC_FORM")

    assert len(blind) == 126
    assert len(domestic) == 504
    assert len({candidate.key for candidate in blind}) == len(blind)
    assert len({candidate.key for candidate in domestic}) == len(domestic)
    assert candidate_grid("LONG_HISTORY") == ()


def test_season_relative_percentile_uses_only_covered_rows() -> None:
    frame = pd.DataFrame(
        {
            "season": ["2025/26"] * 4,
            "score": [1.0, 2.0, 3.0, 99.0],
            "covered": [True, True, True, False],
        }
    )

    result = season_relative_percentile(frame, "score", "covered")

    assert result.iloc[:3].tolist() == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert pd.isna(result.iloc[3])


def test_long_history_cannot_silently_run_without_data() -> None:
    config = SeedBoostConfig("LONG_HISTORY", "ADDITIVE", 0.65, 100.0, 50.0, 0.80)

    with pytest.raises(ValueError, match="long-history boost missing columns"):
        apply_seed_boost(seed(), config)
