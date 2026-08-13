import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ao_elo.tournament_bonus import (
    ELIGIBLE_PROGRESSION_STAGES,
    FIVE_STAGE_WEIGHTED_PROGRESSION_STAGES,
    STAGE_WEIGHTED_PROGRESSION_STAGES,
    FiveStageWeightedTournamentBonusConfig,
    FixedTournamentBonusConfig,
    StageWeightedTournamentBonusConfig,
    apply_five_stage_weighted_tournament_bonus,
    apply_stage_weighted_tournament_bonus,
)
from scripts.run_stage_weighted_progression_backtest import (
    MODEL_CURRENT,
    candidate_grid,
)
from scripts.run_five_stage_weighted_progression_backtest import (
    candidate_grid as five_stage_candidate_grid,
)


LINEAR_WEIGHTS = (
    ("ROUND_OF_16", 0.10),
    ("QUARTERFINAL", 0.20),
    ("SEMIFINAL", 0.30),
    ("FINAL", 0.40),
)


def linear_config(cap: float = 60.0) -> StageWeightedTournamentBonusConfig:
    return StageWeightedTournamentBonusConfig(
        ucl_total_cap=cap,
        stage_weights=LINEAR_WEIGHTS,
    )


@pytest.mark.parametrize(
    ("stage", "ucl", "uel", "uecl"),
    [
        ("ROUND_OF_16", 6.0, 4.0, 2.0),
        ("QUARTERFINAL", 12.0, 8.0, 4.0),
        ("SEMIFINAL", 18.0, 12.0, 6.0),
        ("FINAL", 24.0, 16.0, 8.0),
    ],
)
def test_linear_60_uses_expected_stage_and_competition_hierarchy(
    stage: str,
    ucl: float,
    uel: float,
    uecl: float,
) -> None:
    config = linear_config()
    assert config.stage_bonus("UCL", stage) == pytest.approx(ucl)
    assert config.stage_bonus("UEL", stage) == pytest.approx(uel)
    assert config.stage_bonus("UECL", stage) == pytest.approx(uecl)


def test_four_stage_bonuses_sum_to_each_competition_cap() -> None:
    config = linear_config()
    for competition, expected in (("UCL", 60.0), ("UEL", 40.0), ("UECL", 20.0)):
        total = sum(
            config.stage_bonus(competition, stage)
            for stage in STAGE_WEIGHTED_PROGRESSION_STAGES
        )
        assert total == pytest.approx(expected)
        assert config.cap(competition) == pytest.approx(expected)


@pytest.mark.parametrize(
    "stage",
    ["KNOCKOUT_PLAYOFF", "LEAGUE", "TOP_8", "QUALIFYING"],
)
def test_non_eligible_stages_do_not_receive_bonus(stage: str) -> None:
    with pytest.raises(ValueError, match="not eligible"):
        apply_stage_weighted_tournament_bonus(
            0.0,
            "UCL",
            stage,
            f"tie-{stage}",
            set(),
            linear_config(),
        )


def test_bonus_is_winner_only_and_needs_no_loser_state() -> None:
    update = apply_stage_weighted_tournament_bonus(
        0.0,
        "UEL",
        "SEMIFINAL",
        "uel-sf-1",
        set(),
        linear_config(),
    )
    assert update.stage_weight == pytest.approx(0.30)
    assert update.competition_ratio == pytest.approx(2.0 / 3.0)
    assert update.stage_bonus_requested == pytest.approx(12.0)
    assert update.stage_bonus_applied == pytest.approx(12.0)
    assert update.competition_bonus_post == pytest.approx(12.0)


def test_bonus_is_applied_once_per_tie() -> None:
    processed: set[str] = set()
    config = linear_config()
    apply_stage_weighted_tournament_bonus(
        0.0, "UCL", "FINAL", "ucl-final", processed, config
    )
    with pytest.raises(ValueError, match="already applied"):
        apply_stage_weighted_tournament_bonus(
            24.0, "UCL", "FINAL", "ucl-final", processed, config
        )


def test_season_reset_is_explicit_and_fresh_state_starts_at_zero() -> None:
    config = linear_config()
    first = apply_stage_weighted_tournament_bonus(
        0.0, "UECL", "FINAL", "season-a-final", set(), config
    )
    second = apply_stage_weighted_tournament_bonus(
        0.0, "UECL", "FINAL", "season-b-final", set(), config
    )
    assert config.season_reset is True
    assert first.competition_bonus_pre == second.competition_bonus_pre == 0.0
    assert first.competition_bonus_post == second.competition_bonus_post


@pytest.mark.parametrize(
    "config",
    [
        StageWeightedTournamentBonusConfig(-1.0),
        StageWeightedTournamentBonusConfig(float("nan")),
        StageWeightedTournamentBonusConfig(60.0, season_reset=False),
        StageWeightedTournamentBonusConfig(60.0, winner_only=False),
        StageWeightedTournamentBonusConfig(
            60.0,
            stage_weights=(("ROUND_OF_16", 0.2),),
        ),
        StageWeightedTournamentBonusConfig(
            60.0,
            stage_weights=(
                ("ROUND_OF_16", 0.1),
                ("QUARTERFINAL", 0.2),
                ("SEMIFINAL", 0.3),
                ("FINAL", 0.5),
            ),
        ),
    ],
)
def test_config_rejects_invalid_contracts(
    config: StageWeightedTournamentBonusConfig,
) -> None:
    with pytest.raises(ValueError):
        config.validate()


def test_backtest_grid_has_controls_diagnostic_and_fifteen_promotable_profiles() -> None:
    candidates = candidate_grid()
    assert len(candidates) == 22
    assert candidates[0].key == MODEL_CURRENT
    assert sum(candidate.family == "STAGE_WEIGHTED" for candidate in candidates) == 20
    assert sum(candidate.promotable for candidate in candidates) == 16
    assert all(
        not candidate.promotable
        for candidate in candidates
        if candidate.profile == "EQUAL_FOUR"
    )


def test_every_promotable_stage_profile_is_strictly_increasing() -> None:
    for candidate in candidate_grid():
        if candidate.family != "STAGE_WEIGHTED" or not candidate.promotable:
            continue
        assert candidate.config is not None
        weights = [
            candidate.config.stage_weight(stage)
            for stage in STAGE_WEIGHTED_PROGRESSION_STAGES
        ]
        assert weights == sorted(weights)
        assert all(left < right for left, right in zip(weights, weights[1:]))


def test_penalty_decider_does_not_reduce_progression_bonus() -> None:
    # Progression is intentionally independent of how the tie was decided.
    update = apply_stage_weighted_tournament_bonus(
        0.0,
        "UCL",
        "FINAL",
        "penalty-final",
        set(),
        linear_config(),
    )
    assert update.stage_bonus_applied == pytest.approx(24.0)


@pytest.mark.parametrize(
    ("stage", "ucl", "uel", "uecl"),
    [
        ("KNOCKOUT_PLAYOFF", 6.0, 4.0, 2.0),
        ("ROUND_OF_16", 9.0, 6.0, 3.0),
        ("QUARTERFINAL", 12.0, 8.0, 4.0),
        ("SEMIFINAL", 15.0, 10.0, 5.0),
        ("FINAL", 18.0, 12.0, 6.0),
    ],
)
def test_five_stage_gentle_profile_uses_expected_distribution(
    stage: str,
    ucl: float,
    uel: float,
    uecl: float,
) -> None:
    config = FiveStageWeightedTournamentBonusConfig()
    assert config.stage_bonus("UCL", stage) == pytest.approx(ucl)
    assert config.stage_bonus("UEL", stage) == pytest.approx(uel)
    assert config.stage_bonus("UECL", stage) == pytest.approx(uecl)


def test_equal_five_profile_is_numerically_identical_to_current_production() -> None:
    weighted = FiveStageWeightedTournamentBonusConfig(
        stage_weights=tuple(
            (stage, 0.20) for stage in FIVE_STAGE_WEIGHTED_PROGRESSION_STAGES
        )
    )
    fixed = FixedTournamentBonusConfig(12.0, stages_per_competition=5)
    for competition in ("UCL", "UEL", "UECL"):
        for stage in FIVE_STAGE_WEIGHTED_PROGRESSION_STAGES:
            assert weighted.stage_bonus(competition, stage) == pytest.approx(
                fixed.increment(competition)
            )
        assert weighted.cap(competition) == pytest.approx(fixed.cap(competition))


def test_five_stage_bonus_accepts_penalty_decider_without_discount() -> None:
    update = apply_five_stage_weighted_tournament_bonus(
        0.0,
        "UCL",
        "FINAL",
        "five-stage-penalty-final",
        set(),
        FiveStageWeightedTournamentBonusConfig(),
    )
    assert update.stage_bonus_requested == pytest.approx(18.0)
    assert update.stage_bonus_applied == pytest.approx(18.0)


def test_five_stage_backtest_isolates_weighting_at_fixed_total_cap() -> None:
    candidates = five_stage_candidate_grid()
    assert [candidate.profile for candidate in candidates] == [
        "EQUAL_FIVE",
        "NONE",
        "GENTLE",
        "LINEAR",
        "LATE_HEAVY",
    ]
    assert all(
        candidate.ucl_total_cap == pytest.approx(60.0)
        for candidate in candidates
        if candidate.family != "NONE"
    )
    for candidate in candidates:
        if candidate.family != "FIVE_STAGE_WEIGHTED":
            continue
        assert isinstance(candidate.config, FiveStageWeightedTournamentBonusConfig)
        weights = [
            candidate.config.stage_weight(stage)
            for stage in FIVE_STAGE_WEIGHTED_PROGRESSION_STAGES
        ]
        assert sum(weights) == pytest.approx(1.0)
