from __future__ import annotations

from math import isfinite
from pathlib import Path

import pandas as pd

from ao_elo.config import AOEuropeanEloConfig
from ao_elo.features import (
    compute_domestic_achievement,
    compute_league_strength,
    compute_weighted_country_score,
    compute_weighted_european_history,
    compute_weighted_match_exposure,
    compute_weighted_season_exposure,
)
from ao_elo.scoring import (
    compute_ao_first_elo as compute_final_rating,
    compute_domestic_prior,
    compute_effective_european_exposure,
    compute_european_prior,
    normalize_log_score,
    rating_source_type,
)
from ao_elo.validators import (
    validate_club_european_points,
    validate_country_coefficients,
    validate_domestic_context,
    validate_input_columns,
    validate_required_club_history_rows,
    validate_teams,
)


OUTPUT_COLUMNS = [
    "season",
    "team_id",
    "team_name",
    "country",
    "country_code",
    "domestic_league",
    "competition",
    "entry_round",
    "domestic_position",
    "league_team_count",
    "weighted_country_score",
    "league_strength_norm",
    "league_strength",
    "domestic_position_percentile",
    "league_finish_score",
    "cup_base_score",
    "cup_double_bonus",
    "domestic_achievement_score",
    "domestic_prior",
    "weighted_european_history",
    "european_history_norm",
    "european_prior",
    "weighted_season_exposure",
    "weighted_match_exposure",
    "european_exposure",
    "effective_european_exposure",
    "ao_first_elo",
    "ao_first_elo_rank",
    "rating_source_type",
    "validation_warnings",
]


def compute_ao_first_elo(
    teams: pd.DataFrame,
    country_coefficients: pd.DataFrame,
    domestic_context: pd.DataFrame,
    club_european_points: pd.DataFrame,
    config: AOEuropeanEloConfig,
) -> pd.DataFrame:
    """Build the AO European Elo starting-rating output dataframe."""
    config.validate()
    validate_input_columns(
        teams,
        country_coefficients,
        domestic_context,
        club_european_points,
    )
    validate_teams(teams)
    validate_country_coefficients(country_coefficients)
    validate_club_european_points(club_european_points)
    domestic_warnings = validate_domestic_context(domestic_context)

    domestic = domestic_context.copy()
    domestic["validation_warnings"] = ["; ".join(items) for items in domestic_warnings]

    country = country_coefficients.copy()
    country["weighted_country_score"] = country.apply(
        compute_weighted_country_score,
        axis=1,
        config=config,
    )
    strength = country["weighted_country_score"].apply(
        lambda score: compute_league_strength(score, config)
    )
    country["league_strength_norm"] = strength.apply(lambda values: values[0])
    country["league_strength"] = strength.apply(lambda values: values[1])

    data = teams.merge(domestic, on="team_id", how="left", validate="one_to_one")
    if data["season"].isna().any():
        missing = data.loc[data["season"].isna(), "team_id"]
        raise ValueError(
            "Missing domestic context for team_id(s): "
            + ", ".join(sorted(missing.astype(str).unique()))
        )

    validate_required_club_history_rows(data, club_european_points)

    data = (
        data.merge(
            country[
                [
                    "season",
                    "country_code",
                    "weighted_country_score",
                    "league_strength_norm",
                    "league_strength",
                ]
            ],
            on=["season", "country_code"],
            how="left",
            validate="many_to_one",
        ).merge(
            club_european_points,
            on=["season", "team_id", "country_code"],
            how="left",
            validate="one_to_one",
        )
    )

    if data["weighted_country_score"].isna().any():
        missing = data.loc[data["weighted_country_score"].isna(), "country_code"]
        raise ValueError(
            "Missing country coefficients for country_code(s): "
            + ", ".join(sorted(missing.astype(str).unique()))
        )

    domestic_features = data.apply(
        lambda row: compute_domestic_achievement(
            row["domestic_position"],
            row["league_team_count"],
            row["is_league_champion"],
            row["is_cup_winner"],
            config,
        ),
        axis=1,
    )
    data["domestic_position_percentile"] = domestic_features.apply(
        lambda item: item.domestic_position_percentile
    )
    data["league_finish_score"] = domestic_features.apply(
        lambda item: item.league_finish_score
    )
    data["cup_base_score"] = domestic_features.apply(lambda item: item.cup_base_score)
    data["cup_double_bonus"] = domestic_features.apply(
        lambda item: item.cup_double_bonus
    )
    data["domestic_achievement_score"] = domestic_features.apply(
        lambda item: item.domestic_achievement_score
    )

    data["domestic_prior"] = data.apply(
        lambda row: compute_domestic_prior(
            row["league_strength"],
            row["domestic_achievement_score"],
            config,
        ),
        axis=1,
    )

    data["weighted_european_history"] = data.apply(
        compute_weighted_european_history,
        axis=1,
        config=config,
    )
    data["european_history_norm"] = data["weighted_european_history"].apply(
        lambda value: normalize_log_score(
            value,
            float(config.european_history_benchmark),
        )
    )
    data["european_prior"] = data["european_history_norm"].apply(
        lambda norm: compute_european_prior(norm, config)
    )

    data["weighted_season_exposure"] = data.apply(
        compute_weighted_season_exposure,
        axis=1,
        config=config,
    )
    data["weighted_match_exposure"] = data.apply(
        compute_weighted_match_exposure,
        axis=1,
        config=config,
    )
    data["european_exposure"] = (
        config.exposure_season_weight * data["weighted_season_exposure"]
        + config.exposure_match_weight * data["weighted_match_exposure"]
    )
    data["effective_european_exposure"] = data["european_exposure"].apply(
        lambda exposure: compute_effective_european_exposure(
            exposure,
            config.max_european_exposure,
        )
    )

    data["ao_first_elo"] = data.apply(
        lambda row: compute_final_rating(
            row["domestic_prior"],
            row["european_prior"],
            row["effective_european_exposure"],
        ),
        axis=1,
    )
    _validate_output_invariants(data, config)

    data["rating_source_type"] = data["european_exposure"].apply(
        lambda exposure: rating_source_type(
            exposure,
            config.rating_source_evidence_threshold,
        )
    )

    data["competition"] = data.get("competition", pd.NA)
    data["entry_round"] = data.get("entry_round", data["european_entry_type"])
    data["ao_first_elo_rank"] = data["ao_first_elo"].rank(
        method="min",
        ascending=False,
    ).astype(int)
    data = data.sort_values(
        ["ao_first_elo", "team_id"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return data[OUTPUT_COLUMNS].copy()


def compute_ao_first_elo_from_csv(
    teams_csv: str | Path,
    country_coefficients_csv: str | Path,
    domestic_context_csv: str | Path,
    club_european_points_csv: str | Path,
    config: AOEuropeanEloConfig,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Read CSV inputs, compute ratings, and optionally write the output CSV."""
    output = compute_ao_first_elo(
        teams=pd.read_csv(teams_csv),
        country_coefficients=pd.read_csv(country_coefficients_csv),
        domestic_context=pd.read_csv(domestic_context_csv),
        club_european_points=pd.read_csv(club_european_points_csv),
        config=config,
    )
    if output_csv is not None:
        output.to_csv(output_csv, index=False)
    return output


def _validate_output_invariants(
    data: pd.DataFrame,
    config: AOEuropeanEloConfig,
) -> None:
    """Guard the bounded model contract after feature computation."""
    numeric_columns = [
        "weighted_country_score",
        "league_strength_norm",
        "league_strength",
        "league_finish_score",
        "cup_base_score",
        "cup_double_bonus",
        "domestic_achievement_score",
        "domestic_prior",
        "weighted_european_history",
        "european_history_norm",
        "european_prior",
        "weighted_season_exposure",
        "weighted_match_exposure",
        "european_exposure",
        "effective_european_exposure",
        "ao_first_elo",
    ]
    for column in numeric_columns:
        invalid = ~data[column].map(lambda value: isfinite(float(value)))
        _raise_invariant_error(data, invalid, f"{column} must be finite")

    for column in (
        "league_strength_norm",
        "league_strength",
        "european_history_norm",
        "weighted_season_exposure",
        "weighted_match_exposure",
        "european_exposure",
        "effective_european_exposure",
    ):
        outside = (data[column] < -1e-12) | (data[column] > 1.0 + 1e-12)
        _raise_invariant_error(data, outside, f"{column} must be between 0 and 1")

    effective_exposure_invalid = (
        data["effective_european_exposure"] > data["european_exposure"] + 1e-12
    ) | (
        data["effective_european_exposure"]
        > config.max_european_exposure + 1e-12
    )
    _raise_invariant_error(
        data,
        effective_exposure_invalid,
        "effective_european_exposure must not exceed raw exposure or config cap",
    )

    achievement_outside = (data["domestic_achievement_score"] < -1e-12) | (
        data["domestic_achievement_score"] > config.achievement_cap + 1e-12
    )
    _raise_invariant_error(
        data,
        achievement_outside,
        "domestic_achievement_score must respect achievement_cap",
    )

    lower_prior = data[["domestic_prior", "european_prior"]].min(axis=1)
    upper_prior = data[["domestic_prior", "european_prior"]].max(axis=1)
    outside_priors = (data["ao_first_elo"] < lower_prior - 1e-9) | (
        data["ao_first_elo"] > upper_prior + 1e-9
    )
    _raise_invariant_error(
        data,
        outside_priors,
        "ao_first_elo must stay between domestic_prior and european_prior",
    )


def _raise_invariant_error(
    data: pd.DataFrame,
    invalid: pd.Series,
    message: str,
) -> None:
    if not invalid.any():
        return
    rows = data.loc[invalid, ["season", "team_id", "country_code"]]
    keys = "; ".join(
        f"season={row['season']}, team_id={row['team_id']}, "
        f"country_code={row['country_code']}"
        for _, row in rows.iterrows()
    )
    raise ValueError(f"Model invariant failed: {message} ({keys})")
