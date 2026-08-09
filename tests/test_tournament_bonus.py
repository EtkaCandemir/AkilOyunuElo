import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ao_elo.tournament_bonus import (
    ELIGIBLE_PROGRESSION_STAGES,
    FixedTournamentBonusConfig,
    apply_tournament_progress_bonus,
)
from scripts.run_fixed_tournament_bonus_backtest import (
    candidate_grid,
    validate_production_contract,
)


def test_fixed_bonus_uses_requested_12_8_4_hierarchy_and_caps() -> None:
    config = FixedTournamentBonusConfig(12.0)
    assert config.increment("UCL") == pytest.approx(12.0)
    assert config.increment("UEL") == pytest.approx(8.0)
    assert config.increment("UECL") == pytest.approx(4.0)
    assert config.cap("UCL") == pytest.approx(60.0)
    assert config.cap("UEL") == pytest.approx(40.0)
    assert config.cap("UECL") == pytest.approx(20.0)


def test_bonus_is_winner_only_and_does_not_require_loser_state() -> None:
    processed: set[str] = set()
    update = apply_tournament_progress_bonus(
        0.0,
        "UEL",
        "ROUND_OF_16",
        "2024-uel-r16-1",
        processed,
        FixedTournamentBonusConfig(12.0),
    )
    assert update.applied_bonus == pytest.approx(8.0)
    assert update.bonus_post == pytest.approx(8.0)
    assert processed == {"2024-uel-r16-1"}


def test_bonus_is_applied_once_per_decided_tie() -> None:
    processed: set[str] = set()
    config = FixedTournamentBonusConfig(12.0)
    apply_tournament_progress_bonus(
        0.0,
        "UCL",
        "FINAL",
        "final-1",
        processed,
        config,
    )
    with pytest.raises(ValueError, match="already applied"):
        apply_tournament_progress_bonus(
            12.0,
            "UCL",
            "FINAL",
            "final-1",
            processed,
            config,
        )


def test_cap_prevents_more_than_five_stage_increments() -> None:
    config = FixedTournamentBonusConfig(12.0)
    processed: set[str] = set()
    current = 0.0
    stages = tuple(sorted(ELIGIBLE_PROGRESSION_STAGES))
    for index, stage in enumerate(stages):
        update = apply_tournament_progress_bonus(
            current,
            "UCL",
            stage,
            f"tie-{index}",
            processed,
            config,
        )
        current = update.bonus_post
    extra = apply_tournament_progress_bonus(
        current,
        "UCL",
        "FINAL",
        "tie-extra",
        processed,
        config,
    )
    assert current == pytest.approx(60.0)
    assert extra.applied_bonus == pytest.approx(0.0)
    assert extra.bonus_post == pytest.approx(60.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"base_bonus": -1.0},
        {"base_bonus": float("nan")},
        {"base_bonus": float("inf")},
        {"base_bonus": True},
        {"base_bonus": 12.0, "stages_per_competition": 0},
    ],
)
def test_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        FixedTournamentBonusConfig(**kwargs).validate()


def test_non_knockout_stage_is_rejected() -> None:
    with pytest.raises(ValueError, match="not eligible"):
        apply_tournament_progress_bonus(
            0.0,
            "UCL",
            "LEAGUE",
            "league-stage",
            set(),
            FixedTournamentBonusConfig(12.0),
        )


def test_backtest_grid_contains_baseline_prespecified_and_neighbors() -> None:
    values = tuple(config.base_bonus for config in candidate_grid())
    assert values == (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0)


def test_production_contract_is_read_only_baseline_for_bonus_test() -> None:
    contract = {
        "active_power_carry": 0.0,
        "goal_margin": {
            "active": True,
            "alpha": 0.1,
            "tau": 300.0,
            "goal_difference_cap": 4,
        },
        "progression_bonus": {"active": False},
        "achievement_reserve": {"active": False},
    }
    assert validate_production_contract(contract) == {
        "alpha": 0.1,
        "tau": 300.0,
        "goal_cap": 4,
    }
    contract["progression_bonus"]["active"] = True
    with pytest.raises(ValueError, match="must remain disabled"):
        validate_production_contract(contract)
