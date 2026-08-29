from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from ao_elo.config import V2_RATING_MULTIPLIER
from ao_elo.dynamic import (
    AchievementReserveConfig,
    DynamicEloConfig,
    MatchInput,
    TeamSeed,
    expected_score,
    expected_1x2_probabilities,
    initialize_season,
    lock_prediction,
    run_season,
    update_match,
)


KICKOFF = datetime(2026, 9, 1, 19, 0, tzinfo=timezone.utc)


def test_same_team_cannot_play_twice_at_one_kickoff() -> None:
    config = DynamicEloConfig.calibrated_v2()
    initial = initialize_season("2026/27", seeds(), config)
    first = match("a", home_goals=3, away_goals=0)
    second = match("b", away="C", home_goals=0, away_goals=0)
    state, _ = update_match(initial, first, config)
    with pytest.raises(ValueError, match="same kickoff"):
        update_match(state, second, config)
    with pytest.raises(ValueError):
        lock_prediction(state, second.fixture(), config, generated_at_utc=KICKOFF - timedelta(minutes=1))
    with pytest.raises(ValueError, match="same kickoff"):
        run_season(initial, [first, second], config)


@pytest.mark.parametrize("single", [True, False])
def test_advancer_must_match_field_winner_only_for_single_match_tie(single) -> None:
    config = DynamicEloConfig.calibrated_v2()
    initial = initialize_season("2026/27", seeds(), config)
    final = replace(match(home_goals=3, away_goals=0, knockout=True, decider=True,
                          round_name="Final", stage="FINAL", tie_id="final", advanced="B"),
                    is_single_match_tie=single)
    if single:
        with pytest.raises(ValueError, match="field-score winner"):
            update_match(initial, final, config)
    else:
        _, update = update_match(initial, final, config)
        assert update.progression_bonus_recipient_id == "B"


def seeds() -> tuple[TeamSeed, ...]:
    return (
        TeamSeed("A", "Alpha", 1500.0),
        TeamSeed("B", "Beta", 1400.0),
        TeamSeed("C", "Gamma", 1300.0),
    )


def match(
    match_id: str = "m1",
    *,
    kickoff: datetime = KICKOFF,
    home: str = "A",
    away: str = "B",
    home_goals: int = 2,
    away_goals: int = 1,
    neutral: bool = False,
    penalties: bool = False,
    competition: str = "UCL",
    round_name: str = "League Stage",
    tie_id: str | None = None,
    knockout: bool = False,
    decider: bool = False,
    advanced: str | None = None,
    stage: str | None = None,
) -> MatchInput:
    return MatchInput(
        match_id=match_id,
        season="2026/27",
        kickoff_utc=kickoff,
        competition=competition,
        round=round_name,
        home_team_id=home,
        away_team_id=away,
        home_goals=home_goals,
        away_goals=away_goals,
        is_neutral=neutral,
        decided_on_penalties=penalties,
        tie_id=tie_id,
        is_knockout=knockout,
        is_tie_decider=decider,
        advanced_team_id=advanced,
        stage=stage,
    )


def test_calibrated_v2_parameters_match_selected_model() -> None:
    config = DynamicEloConfig.calibrated_v2()

    assert config.elo_scale == pytest.approx(225.0 * V2_RATING_MULTIPLIER)
    assert config.home_advantage == pytest.approx(40.0 * V2_RATING_MULTIPLIER)
    assert config.k_factor == pytest.approx(28.0 * V2_RATING_MULTIPLIER)
    assert config.power_carry == 0.0
    assert config.draw_at_even == pytest.approx(0.24)
    assert config.draw_shape == pytest.approx(1.0)
    assert config.single_match_draw_enabled is True
    assert config.single_match_draw_at_even == pytest.approx(0.12)
    assert config.goal_difference_enabled is True
    assert config.goal_alpha == pytest.approx(0.15)
    assert config.goal_tau == pytest.approx(300.0)
    assert config.goal_difference_cap == 4
    assert config.xg_performance_enabled is True
    assert config.xg_performance_ratio == pytest.approx(0.30)
    assert config.xg_performance_scale == pytest.approx(1.25)
    assert config.minimum_winner_gain_ratio == pytest.approx(0.70)
    assert config.progression_bonus_enabled is True
    assert config.progression_base_bonus == pytest.approx(12.0)
    assert config.progression_stages_per_competition == 4
    assert config.qualification_stage_k_enabled is True
    assert config.qualification_q1_multiplier == pytest.approx(0.20)
    assert config.qualification_q2_multiplier == pytest.approx(0.275)
    assert config.qualification_q3_multiplier == pytest.approx(0.35)
    assert config.qualification_playoff_multiplier == pytest.approx(0.425)
    assert config.qualifier_to_main_carry == pytest.approx(1.0)
    assert config.fixed_progression_config.increment("UCL") == pytest.approx(12.0)
    assert config.fixed_progression_config.increment("UEL") == pytest.approx(8.0)
    assert config.fixed_progression_config.increment("UECL") == pytest.approx(4.0)
    assert config.achievement_reserve is None
    config.validate()


def test_expected_score_uses_selected_scale_and_neutral_contract() -> None:
    config = DynamicEloConfig.calibrated_v2()
    expected = 1.0 / (1.0 + 10.0 ** (-(100.0 / config.elo_scale)))

    assert expected_score(1500.0, 1400.0, config, neutral=True) == pytest.approx(
        expected
    )
    assert expected_score(1400.0, 1400.0, config, neutral=True) == 0.5
    assert expected_score(1400.0, 1400.0, config, neutral=False) > 0.5


def test_expected_score_is_stable_for_extreme_uncapped_ratings() -> None:
    config = DynamicEloConfig.calibrated_v2()

    assert expected_score(1e300, -1e300, config, neutral=True) == 1.0
    assert expected_score(-1e300, 1e300, config, neutral=True) == 0.0


def test_1x2_probabilities_preserve_expected_score() -> None:
    config = DynamicEloConfig.calibrated_v2()
    expected = expected_score(1500.0, 1400.0, config, neutral=False)

    home, draw, away = expected_1x2_probabilities(
        1500.0,
        1400.0,
        config,
        neutral=False,
    )

    assert home + draw + away == pytest.approx(1.0)
    assert home + 0.5 * draw == pytest.approx(expected)
    assert all(0.0 <= value <= 1.0 for value in (home, draw, away))


def test_single_match_draw_intercept_changes_only_1x2_decomposition() -> None:
    config = DynamicEloConfig.calibrated_v2()
    expected = expected_score(1400.0, 1400.0, config, neutral=True)
    regular = expected_1x2_probabilities(
        1400.0,
        1400.0,
        config,
        neutral=True,
    )
    single = expected_1x2_probabilities(
        1400.0,
        1400.0,
        config,
        neutral=True,
        is_single_match_tie=True,
    )

    assert regular == pytest.approx((0.38, 0.24, 0.38))
    assert single == pytest.approx((0.44, 0.12, 0.44))
    assert regular[0] + 0.5 * regular[1] == pytest.approx(expected)
    assert single[0] + 0.5 * single[1] == pytest.approx(expected)


def test_single_match_format_requires_a_deciding_knockout_tie() -> None:
    invalid = match()
    invalid = replace(invalid, is_single_match_tie=True)

    with pytest.raises(ValueError, match="single-match tie"):
        invalid.validate()


def test_match_update_is_pure_deterministic_and_power_zero_sum() -> None:
    config = DynamicEloConfig.calibrated_v2()
    initial = initialize_season("2026/27", seeds(), config)
    original_power = sum(row.power_elo for row in initial.ratings.values())

    first_state, first_update = update_match(initial, match(), config)
    second_state, second_update = update_match(initial, match(), config)

    assert first_state == second_state
    assert first_update == second_update
    assert initial.processed_match_ids == frozenset()
    assert initial.ratings["A"].power_elo == 1500.0
    assert sum(row.power_elo for row in first_state.ratings.values()) == pytest.approx(
        original_power
    )
    assert first_update.power_delta > 0.0
    assert first_state.ratings["A"].power_elo == pytest.approx(
        1500.0 + first_update.power_delta
    )
    assert first_state.ratings["B"].power_elo == pytest.approx(
        1400.0 - first_update.power_delta
    )


def test_disabled_season_carry_uses_new_first_elo() -> None:
    config = DynamicEloConfig.calibrated_v2()
    previous = initialize_season("2025/26", seeds(), config)
    previous_rating = replace(previous.ratings["A"], power_elo=1700.0)
    previous = replace(
        previous,
        ratings={**previous.ratings, "A": previous_rating},
    )
    current_seeds = (
        TeamSeed("A", "Alpha", 1500.0),
        TeamSeed("B", "Beta", 1400.0),
    )

    current = initialize_season(
        "2026/27",
        current_seeds,
        config,
        previous_state=previous,
    )

    assert current.ratings["A"].power_elo == pytest.approx(1500.0)
    assert current.ratings["B"].power_elo == pytest.approx(1400.0)


def test_cross_competition_qualifier_route_uses_continuous_retention_without_reset() -> None:
    config = DynamicEloConfig.calibrated_v2()
    state = initialize_season("2026/27", seeds(), config)

    state, ucl_q3 = update_match(
        state,
        match(
            "ucl-q3",
            competition="UCL",
            round_name="3rd Qualifying Round",
            home="A",
            away="B",
            home_goals=0,
            away_goals=1,
            tie_id="ucl-q3-tie",
            knockout=True,
            decider=True,
            advanced="B",
            stage="QUALIFYING",
        ),
        config,
    )
    assert ucl_q3.stage_k_multiplier == pytest.approx(0.35)
    assert ucl_q3.effective_k == pytest.approx(config.k_factor * 0.35)

    state, uel_playoff = update_match(
        state,
        match(
            "uel-po",
            kickoff=KICKOFF + timedelta(days=7),
            competition="UEL",
            round_name="Qualifying Play-off Round",
            home="A",
            away="C",
            home_goals=0,
            away_goals=1,
            tie_id="uel-po-tie",
            knockout=True,
            decider=True,
            advanced="C",
            stage="QUALIFYING",
        ),
        config,
    )
    assert uel_playoff.stage_k_multiplier == pytest.approx(0.425)
    pre_entry_power = state.ratings["A"].power_elo

    state, first_main = update_match(
        state,
        match(
            "uecl-main-1",
            kickoff=KICKOFF + timedelta(days=14),
            competition="UECL",
            round_name="League Phase",
            home="A",
            away="B",
        ),
        config,
    )
    assert first_main.stage_k_multiplier == pytest.approx(1.0)
    assert first_main.effective_k == pytest.approx(config.k_factor)
    assert first_main.home_qualifier_carry_applied is False
    assert first_main.home_power_pre == pytest.approx(pre_entry_power)
    assert first_main.home_qualifier_carry_adjustment == pytest.approx(0.0)
    assert "A" not in state.qualification_carry_applied

    _, second_main = update_match(
        state,
        match(
            "uecl-main-2",
            kickoff=KICKOFF + timedelta(days=21),
            competition="UECL",
            round_name="League Phase",
            home="A",
            away="C",
        ),
        config,
    )
    assert second_main.home_qualifier_carry_applied is False
    assert second_main.home_qualifier_carry_adjustment == pytest.approx(0.0)


def test_penalty_shootout_does_not_replace_field_draw() -> None:
    config = DynamicEloConfig.calibrated_v2()
    initial = initialize_season("2026/27", seeds(), config)
    penalty_match = match(
        home_goals=1,
        away_goals=1,
        penalties=True,
        neutral=True,
        tie_id="final",
        knockout=True,
        decider=True,
        advanced="A",
        round_name="Final",
    )

    _, update = update_match(initial, penalty_match, config)

    assert update.actual_home_score == 0.5
    assert update.goal_difference == 0
    assert update.goal_multiplier == 1.0
    assert update.progression_bonus_added == pytest.approx(12.0)


@pytest.mark.parametrize(
    ("competition", "expected_bonus", "expected_cap"),
    [("UCL", 12.0, 48.0), ("UEL", 8.0, 32.0), ("UECL", 4.0, 16.0)],
)
def test_fixed_progression_bonus_uses_competition_ratio_after_decider(
    competition: str,
    expected_bonus: float,
    expected_cap: float,
) -> None:
    config = DynamicEloConfig.calibrated_v2()
    initial = initialize_season("2026/27", seeds(), config)

    state, update = update_match(
        initial,
        match(
            competition=competition,
            round_name="Quarter Finals",
            tie_id=f"{competition}-qf",
            knockout=True,
            decider=True,
            advanced="A",
        ),
        config,
    )

    assert update.progression_bonus_recipient_id == "A"
    assert update.progression_bonus_added == pytest.approx(expected_bonus)
    assert update.progression_bonus_competition_pre == 0.0
    assert update.progression_bonus_competition_post == pytest.approx(expected_bonus)
    assert update.progression_bonus_competition_cap == pytest.approx(expected_cap)
    assert state.ratings["A"].progression_bonus_total == pytest.approx(expected_bonus)
    assert state.ratings["B"].progression_bonus_total == 0.0
    assert state.ratings["A"].ao_live_elo == pytest.approx(
        state.ratings["A"].power_elo + expected_bonus
    )


def test_knockout_playoff_winner_receives_no_progression_bonus() -> None:
    config = DynamicEloConfig.calibrated_v2()
    initial = initialize_season("2026/27", seeds(), config)

    state, update = update_match(
        initial,
        match(
            round_name="Knockout round play-offs",
            tie_id="ucl-kpo-1",
            knockout=True,
            decider=True,
            advanced="A",
        ),
        config,
    )

    assert update.stage == "KNOCKOUT_PLAYOFF"
    assert update.progression_bonus_recipient_id is None
    assert update.progression_bonus_added == 0.0
    assert state.ratings["A"].progression_bonus_total == 0.0
    assert state.ratings["B"].progression_bonus_total == 0.0


def test_fixed_progression_bonus_caps_and_resets_at_new_season() -> None:
    config = DynamicEloConfig.calibrated_v2()
    state = initialize_season("2026/27", seeds(), config)
    rounds = (
        "Knockout round play-offs",
        "Round of 16",
        "Quarter Finals",
        "Semi Finals",
        "Final",
        "Final",
    )
    last_update = None
    for index, round_name in enumerate(rounds):
        state, last_update = update_match(
            state,
            match(
                f"progress-{index}",
                kickoff=KICKOFF + timedelta(days=index),
                round_name=round_name,
                tie_id=f"tie-{index}",
                knockout=True,
                decider=True,
                advanced="A",
            ),
            config,
        )

    assert last_update is not None
    assert state.ratings["A"].progression_bonus_ucl == pytest.approx(48.0)
    assert last_update.progression_bonus_added == 0.0
    assert len(state.processed_tie_ids) == 6

    next_season = initialize_season(
        "2027/28",
        seeds(),
        config,
        previous_state=state,
    )
    assert next_season.ratings["A"].progression_bonus_total == 0.0


def test_completed_tie_id_cannot_receive_progression_twice() -> None:
    config = DynamicEloConfig.calibrated_v2()
    initial = initialize_season("2026/27", seeds(), config)
    state, _ = update_match(
        initial,
        match(
            "qf-1",
            round_name="Quarter Finals",
            tie_id="completed-qf",
            knockout=True,
            decider=True,
            advanced="A",
        ),
        config,
    )

    with pytest.raises(ValueError, match="already completed"):
        update_match(
            state,
            match(
                "qf-2",
                kickoff=KICKOFF + timedelta(days=1),
                round_name="Quarter Finals",
                tie_id="completed-qf",
                knockout=True,
                decider=True,
                advanced="A",
            ),
            config,
        )


@pytest.mark.parametrize(
    ("home_goals", "away_goals"),
    [(1, 0), (2, 2)],
)
def test_active_goal_layer_keeps_draw_and_one_goal_result_at_one(
    home_goals: int,
    away_goals: int,
) -> None:
    config = DynamicEloConfig.calibrated_v2()
    initial = initialize_season("2026/27", seeds(), config)

    _, update = update_match(
        initial,
        match(home_goals=home_goals, away_goals=away_goals),
        config,
    )

    assert update.goal_multiplier == 1.0


def test_production_goal_layer_uses_damped_log_margin_and_goal_cap() -> None:
    config = DynamicEloConfig.calibrated_v2()
    initial = initialize_season("2026/27", seeds(), config)
    effective_difference = 1500.0 - 1400.0 + config.home_advantage
    expected_multiplier = 1.0 + 0.15 * math.log(4.0) * math.exp(
        -abs(effective_difference) / 300.0
    )

    _, four_goal = update_match(
        initial,
        match(home_goals=4, away_goals=0),
        config,
    )
    _, five_goal = update_match(
        initial,
        match(home_goals=5, away_goals=0),
        config,
    )

    assert four_goal.effective_rating_difference == pytest.approx(
        effective_difference
    )
    assert four_goal.goal_multiplier == pytest.approx(expected_multiplier)
    assert five_goal.goal_multiplier == pytest.approx(expected_multiplier)
    assert four_goal.goal_difference_enabled is True
    assert four_goal.goal_alpha == pytest.approx(0.15)
    assert four_goal.goal_tau == pytest.approx(300.0)
    assert four_goal.goal_difference_cap == 4


def test_explicitly_disabled_goal_layer_keeps_large_result_at_one() -> None:
    config = replace(
        DynamicEloConfig.calibrated_v2(),
        goal_difference_enabled=False,
        goal_alpha=0.0,
    )
    initial = initialize_season("2026/27", seeds(), config)

    _, update = update_match(
        initial,
        match(home_goals=5, away_goals=0),
        config,
    )

    assert update.goal_multiplier == 1.0


def test_two_leg_progression_reserve_is_visible_only_after_decider() -> None:
    config = replace(
        DynamicEloConfig.calibrated_v2(),
        achievement_reserve=AchievementReserveConfig(
            reserve_base=100.0,
            reserve_decay=0.5,
        ),
    )
    initial = initialize_season("2026/27", seeds(), config)
    first_leg = match(
        "semi-1",
        tie_id="semi",
        knockout=True,
        round_name="Semi Finals",
        home_goals=1,
        away_goals=0,
    )
    after_first, first_update = update_match(initial, first_leg, config)
    second_leg = match(
        "semi-2",
        kickoff=KICKOFF + timedelta(days=7),
        home="B",
        away="A",
        tie_id="semi",
        knockout=True,
        decider=True,
        advanced="A",
        round_name="Semi Finals",
        home_goals=1,
        away_goals=1,
    )

    after_second, second_update = update_match(after_first, second_leg, config)

    assert first_update.progression_reserve_added == 0.0
    assert after_first.ratings["A"].achievement_reserve == 0.0
    assert second_update.progression_reserve_added > 0.0
    assert after_second.ratings["A"].achievement_reserve == pytest.approx(
        second_update.progression_reserve_added
    )


def test_single_match_final_uses_trophy_reserve_after_field_update() -> None:
    config = replace(
        DynamicEloConfig.calibrated_v2(),
        achievement_reserve=AchievementReserveConfig(
            reserve_base=100.0,
            reserve_decay=0.5,
        ),
    )
    initial = initialize_season("2026/27", seeds(), config)
    final = match(
        tie_id="final",
        knockout=True,
        decider=True,
        advanced="A",
        round_name="Final",
        neutral=True,
    )

    state, update = update_match(initial, final, config)

    assert update.progression_reserve_added == 0.0
    assert update.trophy_reserve_added > 0.0
    assert state.ratings["A"].achievement_reserve == pytest.approx(
        update.trophy_reserve_added
    )


def test_run_season_matches_repeated_single_match_kernel() -> None:
    config = DynamicEloConfig.calibrated_v2()
    initial = initialize_season("2026/27", seeds(), config)
    matches = (
        match("m1"),
        match(
            "m2",
            kickoff=KICKOFF + timedelta(days=1),
            home="B",
            away="C",
            home_goals=0,
            away_goals=2,
        ),
    )
    manual_state, first = update_match(initial, matches[0], config)
    manual_state, second = update_match(manual_state, matches[1], config)

    batch_state, updates = run_season(initial, matches, config)

    assert batch_state == manual_state
    assert updates == (first, second)


def test_duplicate_match_and_chronology_regression_are_rejected() -> None:
    config = DynamicEloConfig.calibrated_v2()
    initial = initialize_season("2026/27", seeds(), config)
    state, _ = update_match(initial, match("m1"), config)

    with pytest.raises(ValueError, match="Duplicate match_id"):
        update_match(state, match("m1"), config)
    with pytest.raises(ValueError, match="Chronology regression"):
        update_match(
            state,
            match("m0", kickoff=KICKOFF - timedelta(seconds=1)),
            config,
        )


def test_missing_team_invalid_score_and_config_mismatch_are_rejected() -> None:
    config = DynamicEloConfig.calibrated_v2()
    initial = initialize_season("2026/27", seeds(), config)

    with pytest.raises(ValueError, match="Missing away team_id"):
        update_match(initial, match(away="UNKNOWN"), config)
    with pytest.raises(ValueError, match="home_goals"):
        update_match(initial, match(home_goals=-1), config)
    with pytest.raises(ValueError, match="config_id"):
        update_match(initial, match(), replace(config, k_factor=config.k_factor + 1.0))
    _, penalty_update = update_match(
        initial,
        match(
            home_goals=2,
            away_goals=1,
            penalties=True,
            tie_id="final",
            knockout=True,
            decider=True,
            advanced="A",
            round_name="Final",
        ),
        config,
    )
    assert penalty_update.actual_home_score == 1.0
    assert penalty_update.goal_difference == 1
    assert penalty_update.goal_multiplier == 1.0
    assert penalty_update.xg_applied is False


def test_dynamic_xg_applied_uses_kernel_eligibility_signal() -> None:
    config = DynamicEloConfig.calibrated_v2()
    initial = initialize_season("2026/27", seeds(), config)
    eligible = replace(
        match(home_goals=2, away_goals=0),
        xg_home=3.0,
        xg_away=0.5,
        xg_analysis_eligible=True,
    )
    _, applied = update_match(initial, eligible, config)
    penalty = replace(
        eligible,
        match_id="penalty",
        round="Final",
        tie_id="penalty-tie",
        is_knockout=True,
        is_tie_decider=True,
        is_single_match_tie=True,
        advanced_team_id="A",
        decided_on_penalties=True,
    )
    _, suppressed = update_match(initial, penalty, config)

    assert applied.xg_performance_signal is not None
    assert applied.xg_applied is True
    assert suppressed.xg_performance_signal is None
    assert suppressed.xg_applied is False


@pytest.mark.parametrize(
    "config",
    [
        replace(
            DynamicEloConfig.calibrated_v2(),
            goal_difference_enabled=False,
        ),
        replace(DynamicEloConfig.calibrated_v2(), goal_alpha=0.0),
        replace(DynamicEloConfig.calibrated_v2(), goal_tau=0.0),
        replace(DynamicEloConfig.calibrated_v2(), goal_difference_cap=4.5),
    ],
)
def test_invalid_goal_difference_config_is_rejected(
    config: DynamicEloConfig,
) -> None:
    with pytest.raises(ValueError):
        config.validate()


def test_active_progression_rejects_five_stage_contract() -> None:
    config = replace(
        DynamicEloConfig.calibrated_v2(),
        progression_stages_per_competition=5,
    )
    with pytest.raises(ValueError, match="four eligible post-R16 stages"):
        config.validate()


def test_dynamic_config_accepts_subunit_draw_shape_safely() -> None:
    config = replace(DynamicEloConfig.calibrated_v2(), draw_shape=0.84)
    config.validate()

    home, draw, away = expected_1x2_probabilities(
        2500.0,
        500.0,
        config,
        neutral=True,
    )
    expected = expected_score(2500.0, 500.0, config, neutral=True)
    assert min(home, draw, away) >= 0.0
    assert home + draw + away == pytest.approx(1.0)
    assert home + 0.5 * draw == pytest.approx(expected)


def test_knockout_tie_identity_cannot_be_reused_for_other_teams() -> None:
    config = DynamicEloConfig.calibrated_v2()
    initial = initialize_season("2026/27", seeds(), config)
    state, _ = update_match(
        initial,
        match("tie-1", tie_id="shared", knockout=True, round_name="Semi Finals"),
        config,
    )

    with pytest.raises(ValueError, match="different teams"):
        update_match(
            state,
            match(
                "tie-2",
                kickoff=KICKOFF + timedelta(days=1),
                home="A",
                away="C",
                tie_id="shared",
                knockout=True,
                round_name="Semi Finals",
            ),
            config,
        )


def test_non_knockout_match_rejects_tie_metadata() -> None:
    config = DynamicEloConfig.calibrated_v2()
    initial = initialize_season("2026/27", seeds(), config)

    with pytest.raises(ValueError, match="Non-knockout"):
        update_match(initial, match(tie_id="invalid"), config)


def test_invalid_state_rating_is_rejected_before_match_processing() -> None:
    config = DynamicEloConfig.calibrated_v2()
    initial = initialize_season("2026/27", seeds(), config)
    invalid_rating = replace(initial.ratings["A"], achievement_reserve=-1.0)
    invalid_state = replace(
        initial,
        ratings={**initial.ratings, "A": invalid_rating},
    )

    with pytest.raises(ValueError, match="achievement_reserve"):
        update_match(invalid_state, match(), config)


def test_non_utc_kickoff_is_rejected() -> None:
    config = DynamicEloConfig.calibrated_v2()
    initial = initialize_season("2026/27", seeds(), config)
    non_utc = datetime(2026, 9, 1, 22, 0, tzinfo=timezone(timedelta(hours=3)))

    with pytest.raises(ValueError, match="normalized to UTC"):
        update_match(initial, match(kickoff=non_utc), config)
