from __future__ import annotations

import pandas as pd
import pytest

from ao_elo.config import AOEuropeanEloConfig, DEFAULT_SEASON_WEIGHTS, SEASON_KEYS
from ao_elo.pipeline import OUTPUT_COLUMNS, compute_ao_first_elo


def test_explicit_zero_history_row_and_output_contract() -> None:
    output = run_model(valid_inputs())
    row = output.iloc[0]

    assert list(output.columns) == OUTPUT_COLUMNS
    assert row["effective_european_exposure"] == pytest.approx(0.0)
    assert row["ao_first_elo"] == pytest.approx(row["domestic_prior"])
    assert row["ao_first_elo_rank"] == 1
    assert min(row["domestic_prior"], row["european_prior"]) <= row["ao_first_elo"]
    assert row["ao_first_elo"] <= max(row["domestic_prior"], row["european_prior"])


@pytest.mark.parametrize(
    "history_mode",
    ["missing", "wrong_country_code", "wrong_season"],
)
def test_missing_or_mismatched_club_history_fails_with_target_key(
    history_mode: str,
) -> None:
    inputs = valid_inputs()
    if history_mode == "missing":
        inputs["club_european_points"] = inputs["club_european_points"].iloc[0:0]
    elif history_mode == "wrong_season":
        inputs["club_european_points"].loc[0, "season"] = "2024/25"
    else:
        inputs["club_european_points"].loc[0, "country_code"] = "BAD"

    with pytest.raises(
        ValueError,
        match=r"missing explicit history row.*season=2025/26, team_id=1, country_code=EX",
    ):
        run_model(inputs)


@pytest.mark.parametrize(
    ("frame_name", "error_label"),
    [
        ("teams", "teams.csv duplicate key"),
        ("country_coefficients", "country_coefficients.csv duplicate key"),
        ("domestic_context", "domestic_context.csv duplicate key"),
        ("club_european_points", "club_european_points.csv duplicate key"),
    ],
)
def test_duplicate_keys_are_rejected(frame_name: str, error_label: str) -> None:
    inputs = valid_inputs()
    inputs[frame_name] = pd.concat(
        [inputs[frame_name], inputs[frame_name]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match=error_label):
        run_model(inputs)


@pytest.mark.parametrize("invalid_value", [None, -1, float("inf"), "invalid"])
def test_invalid_country_points_are_rejected(invalid_value: object) -> None:
    inputs = valid_inputs()
    inputs["country_coefficients"]["points_t"] = inputs[
        "country_coefficients"
    ]["points_t"].astype(object)
    inputs["country_coefficients"].loc[0, "points_t"] = invalid_value

    with pytest.raises(ValueError, match=r"points_t must"):
        run_model(inputs)


@pytest.mark.parametrize("invalid_value", [None, -0.5, float("inf"), "invalid"])
def test_invalid_club_points_are_rejected(invalid_value: object) -> None:
    inputs = valid_inputs()
    inputs["club_european_points"]["club_points_t"] = inputs[
        "club_european_points"
    ]["club_points_t"].astype(object)
    inputs["club_european_points"].loc[0, "club_points_t"] = invalid_value

    with pytest.raises(ValueError, match=r"club_points_t must"):
        run_model(inputs)


def test_missing_country_coefficient_row_is_rejected() -> None:
    inputs = valid_inputs()
    inputs["country_coefficients"] = inputs["country_coefficients"].iloc[0:0]

    with pytest.raises(
        ValueError,
        match=r"Missing country coefficients for country_code\(s\): EX",
    ):
        run_model(inputs)


@pytest.mark.parametrize(
    ("frame_name", "column"),
    [
        ("domestic_context", "is_league_champion"),
        ("domestic_context", "is_cup_winner"),
        ("club_european_points", "played_t"),
    ],
)
def test_non_canonical_boolean_values_are_rejected(
    frame_name: str,
    column: str,
) -> None:
    inputs = valid_inputs()
    inputs[frame_name][column] = inputs[frame_name][column].astype(object)
    inputs[frame_name].loc[0, column] = "yes"

    with pytest.raises(ValueError, match=r"must be true/false or 0/1"):
        run_model(inputs)


def test_one_team_league_is_rejected() -> None:
    inputs = valid_inputs()
    inputs["domestic_context"].loc[0, "league_team_count"] = 1

    with pytest.raises(ValueError, match=r"league_team_count must be >= 2"):
        run_model(inputs)


def test_domestic_context_allows_only_one_target_season() -> None:
    inputs = valid_inputs()
    second_season = inputs["domestic_context"].copy()
    second_season.loc[0, "season"] = "2024/25"
    second_season.loc[0, "team_id"] = 2
    inputs["domestic_context"] = pd.concat(
        [inputs["domestic_context"], second_season],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match=r"exactly one target season"):
        run_model(inputs)


def test_match_cap_must_be_positive_even_without_european_history() -> None:
    inputs = valid_inputs()
    inputs["club_european_points"].loc[0, "match_cap_t"] = 0

    with pytest.raises(ValueError, match=r"match_cap_t must be > 0"):
        run_model(inputs)


@pytest.mark.parametrize(
    "config",
    [
        AOEuropeanEloConfig(float("inf"), 20),
        AOEuropeanEloConfig(25, 20, gamma=0),
        AOEuropeanEloConfig(25, 20, rating_source_evidence_threshold=1.1),
        AOEuropeanEloConfig(25, 20, max_european_exposure=1.1),
        AOEuropeanEloConfig(25, 20, domestic_league_component=-1),
        AOEuropeanEloConfig(
            25,
            20,
            exposure_season_weight=1.1,
            exposure_match_weight=-0.1,
        ),
        AOEuropeanEloConfig(
            25,
            20,
            season_weights={
                **DEFAULT_SEASON_WEIGHTS,
                "t_minus_4": -0.07,
                "t": 0.47,
            },
        ),
    ],
)
def test_invalid_config_ranges_are_rejected(config: AOEuropeanEloConfig) -> None:
    with pytest.raises(ValueError):
        config.validate()


def test_frozen_config_and_experimental_candidate_are_explicit() -> None:
    v1_1 = AOEuropeanEloConfig.v1_1()
    candidate = AOEuropeanEloConfig.experimental_country_candidate()

    assert (
        v1_1.country_strength_benchmark,
        v1_1.gamma,
        v1_1.domestic_league_component,
    ) == (25, 0.8, 140)
    assert (
        candidate.country_strength_benchmark,
        candidate.gamma,
        candidate.domestic_league_component,
    ) == (20, 2.0, 360)
    assert candidate.domestic_achievement_component == 160
    assert candidate.achievement_alpha == 0.40


def test_optional_audit_columns_are_accepted_when_present() -> None:
    inputs = valid_inputs()
    inputs["country_coefficients"]["official_five_year_total"] = 25.0
    inputs["country_coefficients"]["official_country_rank"] = 12
    inputs["club_european_points"]["official_club_coefficient"] = 0.0
    inputs["club_european_points"]["country_part"] = 0.0

    assert len(run_model(inputs)) == 1


def run_model(inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return compute_ao_first_elo(
        **inputs,
        config=AOEuropeanEloConfig(
            country_strength_benchmark=25,
            european_history_benchmark=20,
        ),
    )


def valid_inputs() -> dict[str, pd.DataFrame]:
    teams = pd.DataFrame(
        [
            {
                "team_id": 1,
                "team_name": "No Europe FC",
                "country": "Exampleland",
                "country_code": "EX",
                "domestic_league": "Example League",
            }
        ]
    )
    country_coefficients = pd.DataFrame(
        [
            {
                "season": "2025/26",
                "country": "Exampleland",
                "country_code": "EX",
                "points_t_minus_4": 3,
                "points_t_minus_3": 4,
                "points_t_minus_2": 5,
                "points_t_minus_1": 6,
                "points_t": 7,
            }
        ]
    )
    domestic_context = pd.DataFrame(
        [
            {
                "season": "2025/26",
                "team_id": 1,
                "domestic_position": 1,
                "league_team_count": 20,
                "is_league_champion": True,
                "is_cup_winner": False,
                "european_entry_type": "League Champion",
                "competition": "UCL",
                "entry_round": "League Phase",
            }
        ]
    )
    club_row: dict[str, object] = {
        "season": "2025/26",
        "team_id": 1,
        "team_name_source": "No Europe FC",
        "country_code": "EX",
    }
    for key in SEASON_KEYS:
        club_row[f"club_points_{key}"] = 0
        club_row[f"played_{key}"] = 0
        club_row[f"matches_{key}"] = 0
        club_row[f"match_cap_{key}"] = 6

    return {
        "teams": teams,
        "country_coefficients": country_coefficients,
        "domestic_context": domestic_context,
        "club_european_points": pd.DataFrame([club_row]),
    }
