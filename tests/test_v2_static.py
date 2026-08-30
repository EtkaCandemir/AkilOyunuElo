from __future__ import annotations

import struct
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ao_elo.config import (
    AOEuropeanEloConfig,
    V2_RATING_MULTIPLIER,
    V2_REFERENCE_MAX_RATING,
)
from ao_elo.features import (
    CupContributionConfig,
    champion_equivalent_weight,
    compute_domestic_achievement,
    generalized_domestic_achievement,
)
from ao_elo.pipeline import compute_ao_first_elo, compute_ao_first_elo_from_csv
from ao_elo.scoring import (
    apply_upper_tail,
    compute_effective_european_exposure,
    normalize_log_score,
    normalize_log_score_uncapped,
    participation_normalized_history,
)
from scripts.run_pilot_10_teams import (
    EXPECTED_AO_FIRST_ELO as EXPECTED_SYNTHETIC_V1_RATINGS,
)
from scripts.run_real_pilot_10_teams import (
    EXPECTED_AO_FIRST_ELO as EXPECTED_REAL_V1_RATINGS,
)


ROOT = Path(__file__).resolve().parents[1]
PILOT_ROOT = ROOT / "data" / "pilot_10_teams"
REAL_PILOT_ROOT = ROOT / "data" / "real_pilot_10_teams"


def test_v2_reference_multiplier_and_components() -> None:
    assert V2_RATING_MULTIPLIER == pytest.approx(
        1500.0 / (V2_REFERENCE_MAX_RATING - 500.0)
    )
    config = AOEuropeanEloConfig.v2()
    assert config.base_rating == 500.0
    assert config.domestic_league_component == pytest.approx(519.904930177)
    assert config.domestic_achievement_component == pytest.approx(594.177064768)
    assert config.european_prior_max_boost == pytest.approx(1559.714790999)
    assert config.domestic_surprise_enabled is True
    assert config.domestic_surprise_coefficient == pytest.approx(0.40)
    assert config.domestic_surprise_variance_penalty == pytest.approx(0.50)
    assert config.domestic_surprise_max_abs_adjustment == pytest.approx(30.0)


def test_v2_reference_scale_with_legacy_exposure_is_exact_affine_transform() -> None:
    v1 = run_pilot(AOEuropeanEloConfig.v1_1()).sort_values("team_id")
    # The claim is about the affine scale transform alone, so every layer v1.1
    # does not have must be off and every value it pins must be restored.
    # Domestic Surprise is inert on this fixture (no five-season history) but
    # participation normalization and the cup contribution are not.
    reference_config = replace(
        AOEuropeanEloConfig.v2(),
        max_european_exposure=0.85,
        european_participation_enabled=False,
        european_participation_shrinkage=0.0,
        cup_contribution_enabled=False,
        cup_contribution_weight=0.0,
        unknown_league_finish_score=0.10,
    )
    v2 = run_pilot(reference_config).sort_values("team_id")
    expected = 500.0 + V2_RATING_MULTIPLIER * (v1["ao_first_elo"] - 500.0)

    np.testing.assert_allclose(v2["ao_first_elo"], expected, atol=1e-10, rtol=0)
    assert v2["ao_first_elo_rank"].tolist() == v1["ao_first_elo_rank"].tolist()
    assert v1["model_version"].eq("v1.1").all()
    assert v2["model_version"].eq("ao-european-elo-v2.0-dev-freeze").all()


@pytest.mark.parametrize(
    ("data_root", "expected"),
    [
        (PILOT_ROOT, EXPECTED_SYNTHETIC_V1_RATINGS),
        (REAL_PILOT_ROOT, EXPECTED_REAL_V1_RATINGS),
    ],
)
def test_v1_1_pilot_rating_bytes_are_frozen(
    data_root: Path,
    expected: dict[str, float],
) -> None:
    output = compute_ao_first_elo_from_csv(
        teams_csv=data_root / "teams.csv",
        country_coefficients_csv=data_root / "country_coefficients.csv",
        domestic_context_csv=data_root / "domestic_context.csv",
        club_european_points_csv=data_root / "club_european_points.csv",
        config=AOEuropeanEloConfig.v1_1(),
    ).set_index("team_name")

    for team_name, expected_rating in expected.items():
        assert struct.pack("!d", float(output.loc[team_name, "ao_first_elo"])) == (
            struct.pack("!d", expected_rating)
        )


@pytest.mark.parametrize("beta", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_upper_tail_is_continuous_at_benchmark(beta: float) -> None:
    assert apply_upper_tail(1.0, beta) == pytest.approx(1.0)
    assert apply_upper_tail(1.0 + 1e-9, beta) == pytest.approx(
        1.0 + beta * 1e-9
    )


def test_tail_beta_zero_reproduces_hard_cap_and_beta_one_keeps_signal() -> None:
    uncapped = normalize_log_score_uncapped(100.0, 25.0)

    assert uncapped > 1.0
    assert normalize_log_score(100.0, 25.0, tail_beta=0.0) == 1.0
    assert normalize_log_score(100.0, 25.0, tail_beta=1.0) == pytest.approx(
        uncapped
    )
    assert apply_upper_tail(uncapped, 0.5) == pytest.approx(
        1.0 + 0.5 * (uncapped - 1.0)
    )


@pytest.mark.parametrize(
    ("beta", "expected"),
    [(0.0, 0.85), (1 / 3, 0.90), (2 / 3, 0.95), (1.0, 1.0)],
)
def test_exposure_tail_maps_full_evidence_to_candidate_grid(
    beta: float,
    expected: float,
) -> None:
    assert compute_effective_european_exposure(1.0, 0.85, beta) == pytest.approx(
        expected
    )
    assert compute_effective_european_exposure(0.70, 0.85, beta) == 0.70


def test_v2_zero_exposure_still_equals_scaled_domestic_prior() -> None:
    output = run_pilot(AOEuropeanEloConfig.v2())
    row = output.loc[output["team_name"].eq("Metro Albion")].iloc[0]

    assert row["european_exposure"] == 0.0
    assert row["effective_european_exposure"] == 0.0
    assert row["ao_first_elo"] == pytest.approx(row["domestic_prior"])


def test_active_v2_normalizes_european_history_over_participation() -> None:
    config = AOEuropeanEloConfig.active()

    assert config.european_participation_enabled is True
    assert config.european_participation_shrinkage == pytest.approx(0.20)


def test_participation_normalization_is_neutral_at_full_participation() -> None:
    """A club with five complete seasons must not move, for any shrinkage.

    This is the property that makes the layer safe to ship: the correction is
    proportional to the participation gap, so complete evidence is untouched.
    """
    for shrinkage in (0.0, 0.20, 0.75, 5.0):
        assert participation_normalized_history(7.5, 1.0, shrinkage) == pytest.approx(
            7.5
        )


def test_participation_normalization_lifts_only_the_participation_gap() -> None:
    # Trabzonspor 2022/23: played the weights worth 0.20 + 0.33 of five seasons.
    rate = participation_normalized_history(1.425, 0.53, 0.20)

    assert rate == pytest.approx(1.425 * 1.20 / 0.73)
    assert rate > 1.425
    # Shrinkage damps the correction rather than removing it.
    assert participation_normalized_history(1.425, 0.53, 0.75) < rate
    # A club that never entered keeps a zero rate; its exposure is zero, so the
    # prior never reaches the blend and a rate there would divide by nothing.
    assert participation_normalized_history(0.0, 0.0, 0.20) == 0.0


def test_participation_normalization_moves_only_partial_participants() -> None:
    frames = read_pilot_frames()
    active = compute_ao_first_elo(config=AOEuropeanEloConfig.active(), **frames)
    disabled = compute_ao_first_elo(
        config=replace(
            AOEuropeanEloConfig.active(),
            european_participation_enabled=False,
            european_participation_shrinkage=0.0,
        ),
        **frames,
    )
    merged = active[
        ["team_id", "ao_first_elo", "weighted_season_exposure"]
    ].merge(
        disabled[["team_id", "ao_first_elo"]], on="team_id", suffixes=("", "_off")
    )
    delta = (merged["ao_first_elo"] - merged["ao_first_elo_off"]).abs()

    complete = merged["weighted_season_exposure"] >= 1.0 - 1e-12
    absent = merged["weighted_season_exposure"] <= 1e-12
    assert delta[complete].max() < 1e-9
    assert delta[absent].max() == 0.0
    # The layer never lowers a rating: renormalizing can only raise the rate.
    assert (merged["ao_first_elo"] >= merged["ao_first_elo_off"] - 1e-9).all()


def test_active_v2_generalizes_the_domestic_cup_contribution() -> None:
    config = AOEuropeanEloConfig.active()

    assert config.cup_contribution_enabled is True
    # The shipped weight must stay the derived one, not a hand-picked number.
    assert config.cup_contribution_weight == pytest.approx(
        champion_equivalent_weight(config)
    )


def test_cup_contribution_leaves_non_cup_winners_untouched() -> None:
    """The layer may only speak where the cup carries information."""
    config = AOEuropeanEloConfig.active()
    cup_config = CupContributionConfig(config.cup_contribution_weight)

    for position, count, champion in ((1, 20, True), (3, 20, False), (18, 20, False)):
        base = compute_domestic_achievement(position, count, champion, False, config)
        generalized = generalized_domestic_achievement(
            position, count, champion, False, config, cup_config
        )
        assert generalized.domestic_achievement_score == pytest.approx(
            base.domestic_achievement_score
        )


def test_cup_contribution_preserves_the_champion_and_cup_total() -> None:
    """The group the old rule already rewarded must not move at all.

    That is the whole point of the champion-equivalent weight: the change is
    "extend the same logic to everyone else", not "re-price the double".
    """
    config = AOEuropeanEloConfig.active()
    cup_config = CupContributionConfig(config.cup_contribution_weight)

    before = compute_domestic_achievement(1, 20, True, True, config)
    after = generalized_domestic_achievement(1, 20, True, True, config, cup_config)

    assert before.domestic_achievement_score == pytest.approx(1.08)
    assert after.domestic_achievement_score == pytest.approx(
        before.domestic_achievement_score
    )


def test_cup_contribution_never_lowers_achievement() -> None:
    config = AOEuropeanEloConfig.active()
    cup_config = CupContributionConfig(config.cup_contribution_weight)

    for position in range(1, 21):
        base = compute_domestic_achievement(position, 20, position == 1, True, config)
        generalized = generalized_domestic_achievement(
            position, 20, position == 1, True, config, cup_config
        )
        assert (
            generalized.domestic_achievement_score
            >= base.domestic_achievement_score - 1e-12
        )


def test_mid_table_cup_winner_now_earns_credit_for_the_trophy() -> None:
    """Under the old rule a cup winner above the cup base got nothing."""
    config = AOEuropeanEloConfig.active()
    cup_config = CupContributionConfig(config.cup_contribution_weight)

    base = compute_domestic_achievement(3, 20, False, True, config)
    generalized = generalized_domestic_achievement(3, 20, False, True, config, cup_config)

    assert base.league_finish_score > config.cup_base_score
    assert base.domestic_achievement_score == pytest.approx(base.league_finish_score)
    assert generalized.domestic_achievement_score > base.domestic_achievement_score


def test_unknown_position_is_never_worse_than_finishing_last() -> None:
    """Absence of evidence must not be punished harder than the worst evidence.

    The percentile curve floors at `percentile_floor`, so an unknown position
    landing below it would rank a team beneath an actual last-place finish.
    Legacy configs are exempt: they are frozen for reproducibility.
    """
    config = AOEuropeanEloConfig.active()
    last_place = compute_domestic_achievement(20, 20, False, False, config)
    unknown = compute_domestic_achievement(None, None, False, False, config)

    assert config.unknown_league_finish_score >= config.percentile_floor
    assert unknown.league_finish_score >= last_place.league_finish_score
    assert last_place.league_finish_score == pytest.approx(config.percentile_floor)
    # The frozen v1.1 baseline keeps the historical value on purpose.
    assert AOEuropeanEloConfig.v1_1().unknown_league_finish_score == pytest.approx(0.10)


def test_active_v2_caps_european_exposure_at_selected_value() -> None:
    config = AOEuropeanEloConfig.active()

    assert config.max_european_exposure == pytest.approx(0.65)
    assert AOEuropeanEloConfig.v1_1().max_european_exposure == pytest.approx(0.85)


def test_active_v2_applies_variance_controlled_domestic_surprise() -> None:
    frames = read_pilot_frames()
    domestic = frames["domestic_context"]
    for offset in range(5, 0, -1):
        domestic[f"history_position_t_minus_{offset}"] = 6
        domestic[f"history_team_count_t_minus_{offset}"] = 16

    output = compute_ao_first_elo(
        **frames,
        config=AOEuropeanEloConfig.active(),
    ).set_index("team_name")
    row = output.loc["Midoria Champions"]

    assert row["domestic_surprise_status"] == "APPLIED"
    assert row["domestic_surprise_history_seasons"] == 5
    assert row["domestic_surprise_historical_mean"] == pytest.approx(2 / 3)
    assert row["domestic_surprise_historical_volatility"] == pytest.approx(0.0)
    assert row["domestic_surprise_consistency_multiplier"] == pytest.approx(1.0)
    assert row["domestic_surprise_raw_score"] == pytest.approx(1 / 3)
    assert row["domestic_surprise_domestic_adjustment"] == pytest.approx(30.0)
    assert row["adjusted_domestic_prior"] == pytest.approx(
        row["domestic_prior"] + 30.0
    )
    assert row["domestic_surprise_ao_first_elo_adjustment"] == pytest.approx(30.0)
    assert row["ao_first_elo"] == pytest.approx(row["adjusted_domestic_prior"])


def test_domestic_surprise_is_damped_by_european_exposure() -> None:
    frames = read_pilot_frames()
    domestic = frames["domestic_context"]
    for offset in range(5, 0, -1):
        domestic[f"history_position_t_minus_{offset}"] = 1
        domestic[f"history_team_count_t_minus_{offset}"] = 20

    output = compute_ao_first_elo(
        **frames,
        config=AOEuropeanEloConfig.active(),
    ).set_index("team_name")
    row = output.loc["Few Match Wanderers"]

    assert row["domestic_surprise_domestic_adjustment"] < 0.0
    assert abs(row["domestic_surprise_ao_first_elo_adjustment"]) < abs(
        row["domestic_surprise_domestic_adjustment"]
    )
    assert row["domestic_surprise_ao_first_elo_adjustment"] == pytest.approx(
        (1.0 - row["effective_european_exposure"])
        * row["domestic_surprise_domestic_adjustment"]
    )


def test_missing_domestic_history_preserves_previous_v2_rating() -> None:
    output = run_pilot(AOEuropeanEloConfig.active())

    assert output["domestic_surprise_status"].eq("INSUFFICIENT_HISTORY").all()
    assert output["domestic_surprise_domestic_adjustment"].eq(0.0).all()
    np.testing.assert_allclose(
        output["ao_first_elo"],
        output["ao_first_elo_before_domestic_surprise"],
        atol=0.0,
        rtol=0.0,
    )


def test_tail_output_exposes_uncapped_values_and_activation_flags() -> None:
    frames = read_pilot_frames()
    country_columns = [
        "points_t_minus_4",
        "points_t_minus_3",
        "points_t_minus_2",
        "points_t_minus_1",
        "points_t",
    ]
    frames["country_coefficients"].loc[:, country_columns] = 100.0
    club = frames["club_european_points"]
    club.loc[:, [column for column in club if column.startswith("club_points_")]] = 100.0
    club.loc[:, [column for column in club if column.startswith("played_")]] = 1
    club.loc[:, [column for column in club if column.startswith("matches_")]] = 8
    club.loc[:, [column for column in club if column.startswith("match_cap_")]] = 8
    output = compute_ao_first_elo(
        **frames,
        config=AOEuropeanEloConfig.v2(
            country_tail_beta=0.5,
            european_tail_beta=0.5,
            exposure_tail_beta=1.0,
        ),
    )

    country_tail = output.loc[output["country_tail_active"]]
    european_tail = output.loc[output["european_tail_active"]]
    exposure_tail = output.loc[output["exposure_tail_active"]]
    assert not country_tail.empty
    assert not european_tail.empty
    assert not exposure_tail.empty
    assert (country_tail["country_strength_uncapped_norm"] > 1.0).all()
    assert (european_tail["european_history_uncapped_norm"] > 1.0).all()
    assert (exposure_tail["effective_european_exposure"] <= 1.0).all()


def test_achievement_saturation_tracks_safety_cap_not_champion_step() -> None:
    config = AOEuropeanEloConfig.v2()
    output = run_pilot(config)
    expected_saturated = (
        output["domestic_achievement_uncapped_score"]
        >= config.achievement_cap - 1e-12
    )
    expected_active = (
        output["domestic_achievement_uncapped_score"]
        > config.achievement_cap + 1e-12
    )

    assert output["achievement_saturated"].equals(expected_saturated)
    assert output["achievement_cap_active"].equals(expected_active)


def test_reference_band_is_not_a_hard_cap() -> None:
    frames = read_pilot_frames()
    frames["country_coefficients"].loc[:, [
        "points_t_minus_4",
        "points_t_minus_3",
        "points_t_minus_2",
        "points_t_minus_1",
        "points_t",
    ]] = 250.0
    club_columns = [
        column
        for column in frames["club_european_points"].columns
        if column.startswith("club_points_")
    ]
    frames["club_european_points"].loc[:, club_columns] = 250.0
    played_columns = [
        column
        for column in frames["club_european_points"].columns
        if column.startswith("played_")
    ]
    match_columns = [
        column
        for column in frames["club_european_points"].columns
        if column.startswith("matches_")
    ]
    cap_columns = [
        column
        for column in frames["club_european_points"].columns
        if column.startswith("match_cap_")
    ]
    frames["club_european_points"].loc[:, played_columns] = 1
    frames["club_european_points"].loc[:, match_columns] = 8
    frames["club_european_points"].loc[:, cap_columns] = 8

    output = compute_ao_first_elo(
        config=AOEuropeanEloConfig.v2(
            country_tail_beta=1.0,
            european_tail_beta=1.0,
            exposure_tail_beta=1.0,
        ),
        **frames,
    )

    assert output["ao_first_elo"].max() > 2000.0
    assert not output["ao_first_elo"].eq(2000.0).any()


@pytest.mark.parametrize(
    "field",
    ["country_tail_beta", "european_tail_beta", "exposure_tail_beta"],
)
@pytest.mark.parametrize("value", [-0.01, 1.01, float("inf")])
def test_invalid_v2_tail_beta_is_rejected(field: str, value: float) -> None:
    config = AOEuropeanEloConfig.v2()
    invalid = AOEuropeanEloConfig(
        **{
            **config.__dict__,
            field: value,
        }
    )

    with pytest.raises(ValueError, match=field):
        invalid.validate()


def run_pilot(config: AOEuropeanEloConfig) -> pd.DataFrame:
    return compute_ao_first_elo_from_csv(config=config, **pilot_csv_paths())


def pilot_csv_paths() -> dict[str, Path]:
    return {
        "teams_csv": PILOT_ROOT / "teams.csv",
        "country_coefficients_csv": PILOT_ROOT / "country_coefficients.csv",
        "domestic_context_csv": PILOT_ROOT / "domestic_context.csv",
        "club_european_points_csv": PILOT_ROOT / "club_european_points.csv",
    }


def read_pilot_frames() -> dict[str, pd.DataFrame]:
    return {
        "teams": pd.read_csv(PILOT_ROOT / "teams.csv"),
        "country_coefficients": pd.read_csv(
            PILOT_ROOT / "country_coefficients.csv"
        ),
        "domestic_context": pd.read_csv(PILOT_ROOT / "domestic_context.csv"),
        "club_european_points": pd.read_csv(
            PILOT_ROOT / "club_european_points.csv"
        ),
    }


def test_active_european_tail_is_not_truncated() -> None:
    # beta = 1 kesme yapmaz. beta = 0 iken benchmark'i asan butun kulupler tek
    # bir European Prior'a iniyordu; 2026/27'de bu 14 kulup demekti.
    config = AOEuropeanEloConfig.active()
    assert config.european_tail_beta == pytest.approx(1.0)
    assert apply_upper_tail(1.25, config.european_tail_beta) == pytest.approx(1.25)


def test_country_and_exposure_tails_stay_closed() -> None:
    # Karar yalniz Avrupa gecmisi normuna uygulandi.
    config = AOEuropeanEloConfig.active()
    assert config.country_tail_beta == pytest.approx(0.0)
    assert config.exposure_tail_beta == pytest.approx(0.0)


def test_legacy_configs_keep_the_hard_cap() -> None:
    # Donmus regresyon pilotlari eski davranisi korumali.
    assert AOEuropeanEloConfig.v1_1().european_tail_beta == pytest.approx(0.0)
    assert AOEuropeanEloConfig.experimental_country_candidate().european_tail_beta == (
        pytest.approx(0.0)
    )


def test_the_tail_never_lowers_a_european_prior() -> None:
    config = AOEuropeanEloConfig.active()
    for uncapped in (0.25, 0.9, 1.0, 1.0001, 1.16, 2.0):
        flat = apply_upper_tail(uncapped, 0.0)
        active = apply_upper_tail(uncapped, config.european_tail_beta)
        assert active >= flat
    # Benchmark altinda kalan kulupler birebir ayni kalir.
    assert apply_upper_tail(0.9, config.european_tail_beta) == pytest.approx(
        apply_upper_tail(0.9, 0.0)
    )
