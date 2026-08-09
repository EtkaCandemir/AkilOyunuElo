from __future__ import annotations

from math import isfinite
from pathlib import Path

import pandas as pd

from ao_elo.config import AOEuropeanEloConfig
from ao_elo.domestic_surprise_variance import (
    DOMESTIC_HISTORY_KEYS,
    VarianceDomesticSurpriseConfig,
    calculate_variance_domestic_surprise_adjustment,
)
from ao_elo.features import (
    as_bool,
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
    apply_upper_tail,
    normalize_log_score_uncapped,
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
    "model_version",
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
    "country_strength_uncapped_norm",
    "league_strength_norm",
    "league_strength",
    "country_strength_saturated",
    "country_tail_excess",
    "country_tail_active",
    "domestic_position_percentile",
    "league_finish_score",
    "cup_base_score",
    "cup_double_bonus",
    "domestic_achievement_uncapped_score",
    "domestic_achievement_score",
    "achievement_saturated",
    "achievement_cap_active",
    "domestic_prior",
    "domestic_surprise_enabled",
    "domestic_surprise_status",
    "domestic_surprise_current_finish_score",
    "domestic_surprise_history_seasons",
    "domestic_surprise_historical_mean",
    "domestic_surprise_historical_variance",
    "domestic_surprise_historical_volatility",
    "domestic_surprise_normalized_volatility",
    "domestic_surprise_consistency_multiplier",
    "domestic_surprise_raw_score",
    "domestic_surprise_effective_score",
    "domestic_surprise_domestic_adjustment",
    "adjusted_domestic_prior",
    "weighted_european_history",
    "european_history_uncapped_norm",
    "european_history_norm",
    "european_prior",
    "european_history_saturated",
    "european_tail_excess",
    "european_tail_active",
    "weighted_season_exposure",
    "weighted_match_exposure",
    "european_exposure",
    "effective_european_exposure",
    "exposure_saturated",
    "exposure_tail_excess",
    "exposure_tail_active",
    "saturation_count",
    "ao_first_elo_before_domestic_surprise",
    "domestic_surprise_ao_first_elo_adjustment",
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
    country["country_strength_uncapped_norm"] = strength.apply(lambda values: values[0])
    country["league_strength_norm"] = strength.apply(lambda values: values[1])
    country["league_strength"] = strength.apply(lambda values: values[2])
    country["country_tail_excess"] = (
        country["country_strength_uncapped_norm"] - 1.0
    ).clip(lower=0.0)
    country["country_strength_saturated"] = (
        country["country_strength_uncapped_norm"] >= 1.0 - 1e-12
    )
    country["country_tail_active"] = (
        (country["country_tail_excess"] > 0.0) & (config.country_tail_beta > 0.0)
    )

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
                    "country_strength_uncapped_norm",
                    "league_strength_norm",
                    "league_strength",
                    "country_strength_saturated",
                    "country_tail_excess",
                    "country_tail_active",
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
    data["domestic_achievement_uncapped_score"] = domestic_features.apply(
        lambda item: item.domestic_achievement_uncapped_score
    )
    data["domestic_achievement_score"] = domestic_features.apply(
        lambda item: item.domestic_achievement_score
    )
    data["achievement_saturated"] = (
        data["domestic_achievement_uncapped_score"]
        >= config.achievement_cap - 1e-12
    )
    data["achievement_cap_active"] = (
        data["domestic_achievement_uncapped_score"]
        > config.achievement_cap + 1e-12
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
    data["european_history_uncapped_norm"] = data["weighted_european_history"].apply(
        lambda value: normalize_log_score_uncapped(
            value,
            float(config.european_history_benchmark),
        )
    )
    data["european_history_norm"] = data["european_history_uncapped_norm"].apply(
        lambda value: apply_upper_tail(value, config.european_tail_beta)
    )
    data["european_tail_excess"] = (
        data["european_history_uncapped_norm"] - 1.0
    ).clip(lower=0.0)
    data["european_history_saturated"] = (
        data["european_history_uncapped_norm"] >= 1.0 - 1e-12
    )
    data["european_tail_active"] = (
        (data["european_tail_excess"] > 0.0) & (config.european_tail_beta > 0.0)
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
            config.exposure_tail_beta,
        )
    )
    data["exposure_tail_excess"] = (
        data["european_exposure"] - config.max_european_exposure
    ).clip(lower=0.0)
    data["exposure_saturated"] = (
        data["european_exposure"] >= config.max_european_exposure - 1e-12
    )
    data["exposure_tail_active"] = (
        (data["exposure_tail_excess"] > 0.0) & (config.exposure_tail_beta > 0.0)
    )
    data["saturation_count"] = data[
        [
            "country_strength_saturated",
            "achievement_saturated",
            "european_history_saturated",
            "exposure_saturated",
        ]
    ].sum(axis=1).astype(int)

    data["ao_first_elo_before_domestic_surprise"] = data.apply(
        lambda row: compute_final_rating(
            row["domestic_prior"],
            row["european_prior"],
            row["effective_european_exposure"],
        ),
        axis=1,
    )
    surprise_config = VarianceDomesticSurpriseConfig(
        coefficient=(
            config.domestic_surprise_coefficient
            if config.domestic_surprise_enabled
            else 0.0
        ),
        variance_penalty=config.domestic_surprise_variance_penalty,
        max_abs_adjustment=config.domestic_surprise_max_abs_adjustment,
        minimum_history_seasons=config.domestic_surprise_minimum_history_seasons,
    )
    surprise_results = data.apply(
        lambda row: _compute_domestic_surprise(row, config, surprise_config),
        axis=1,
    )
    data["domestic_surprise_enabled"] = config.domestic_surprise_enabled
    data["domestic_surprise_status"] = surprise_results.apply(lambda item: item[0])
    data["domestic_surprise_current_finish_score"] = surprise_results.apply(
        lambda item: item[1]
    )
    data["domestic_surprise_history_seasons"] = surprise_results.apply(
        lambda item: item[2].history_seasons
    )
    data["domestic_surprise_historical_mean"] = surprise_results.apply(
        lambda item: item[2].historical_mean
    )
    data["domestic_surprise_historical_variance"] = surprise_results.apply(
        lambda item: item[2].historical_variance
    )
    data["domestic_surprise_historical_volatility"] = surprise_results.apply(
        lambda item: item[2].historical_volatility
    )
    data["domestic_surprise_normalized_volatility"] = surprise_results.apply(
        lambda item: item[2].normalized_volatility
    )
    data["domestic_surprise_consistency_multiplier"] = surprise_results.apply(
        lambda item: item[2].consistency_multiplier
    )
    data["domestic_surprise_raw_score"] = surprise_results.apply(
        lambda item: item[2].raw_surprise
    )
    data["domestic_surprise_effective_score"] = surprise_results.apply(
        lambda item: item[2].effective_surprise
    )
    data["domestic_surprise_domestic_adjustment"] = surprise_results.apply(
        lambda item: item[2].domestic_prior_adjustment
    )
    data["adjusted_domestic_prior"] = surprise_results.apply(
        lambda item: item[2].adjusted_domestic_prior
    )
    data["ao_first_elo"] = surprise_results.apply(
        lambda item: item[2].adjusted_ao_first_elo
    )
    data["domestic_surprise_ao_first_elo_adjustment"] = (
        data["ao_first_elo"] - data["ao_first_elo_before_domestic_surprise"]
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
    data["model_version"] = config.model_version
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


def _compute_domestic_surprise(
    row: pd.Series,
    config: AOEuropeanEloConfig,
    surprise_config: VarianceDomesticSurpriseConfig,
):
    current_finish_score: float | None
    if pd.notna(row["domestic_position_percentile"]):
        current_finish_score = float(row["domestic_position_percentile"])
    elif as_bool(row["is_league_champion"]):
        current_finish_score = 1.0
    else:
        current_finish_score = None

    history = [_historical_finish_score(row, key) for key in DOMESTIC_HISTORY_KEYS]
    if current_finish_score is None:
        calculation_current = 0.0
        calculation_history: list[float | None] = [None] * len(history)
    else:
        calculation_current = current_finish_score
        calculation_history = history
    achievement_scale = config.achievement_alpha + (
        (1.0 - config.achievement_alpha) * float(row["league_strength"])
    )
    adjustment = calculate_variance_domestic_surprise_adjustment(
        current_finish_score=calculation_current,
        historical_finish_scores=calculation_history,
        domestic_prior=float(row["domestic_prior"]),
        european_prior=float(row["european_prior"]),
        effective_european_exposure=float(row["effective_european_exposure"]),
        domestic_achievement_component=config.domestic_achievement_component,
        achievement_scale=achievement_scale,
        config=surprise_config,
    )
    if not config.domestic_surprise_enabled:
        status = "DISABLED"
    elif current_finish_score is None or adjustment.history_seasons < 5:
        status = "INSUFFICIENT_HISTORY"
    elif abs(adjustment.domestic_prior_adjustment) <= 1e-12:
        status = "NO_CHANGE"
    else:
        status = "APPLIED"
    return status, current_finish_score, adjustment


def _historical_finish_score(row: pd.Series, key: str) -> float | None:
    position = row.get(f"history_position_{key}", pd.NA)
    team_count = row.get(f"history_team_count_{key}", pd.NA)
    if pd.isna(position) or pd.isna(team_count):
        return None
    return (float(team_count) - float(position)) / (float(team_count) - 1.0)


def _validate_output_invariants(
    data: pd.DataFrame,
    config: AOEuropeanEloConfig,
) -> None:
    """Guard the model contract after feature computation."""
    numeric_columns = [
        "weighted_country_score",
        "country_strength_uncapped_norm",
        "league_strength_norm",
        "league_strength",
        "league_finish_score",
        "cup_base_score",
        "cup_double_bonus",
        "domestic_achievement_uncapped_score",
        "domestic_achievement_score",
        "domestic_prior",
        "domestic_surprise_consistency_multiplier",
        "domestic_surprise_raw_score",
        "domestic_surprise_effective_score",
        "domestic_surprise_domestic_adjustment",
        "adjusted_domestic_prior",
        "weighted_european_history",
        "european_history_uncapped_norm",
        "european_history_norm",
        "european_prior",
        "weighted_season_exposure",
        "weighted_match_exposure",
        "european_exposure",
        "effective_european_exposure",
        "saturation_count",
        "ao_first_elo_before_domestic_surprise",
        "domestic_surprise_ao_first_elo_adjustment",
        "ao_first_elo",
    ]
    for column in numeric_columns:
        invalid = ~data[column].map(lambda value: isfinite(float(value)))
        _raise_invariant_error(data, invalid, f"{column} must be finite")

    for column in (
        "weighted_season_exposure",
        "weighted_match_exposure",
        "european_exposure",
        "effective_european_exposure",
    ):
        outside = (data[column] < -1e-12) | (data[column] > 1.0 + 1e-12)
        _raise_invariant_error(data, outside, f"{column} must be between 0 and 1")

    for column in (
        "country_strength_uncapped_norm",
        "league_strength_norm",
        "league_strength",
        "european_history_uncapped_norm",
        "european_history_norm",
    ):
        negative = data[column] < -1e-12
        _raise_invariant_error(data, negative, f"{column} must be non-negative")

    effective_exposure_invalid = (
        data["effective_european_exposure"] > data["european_exposure"] + 1e-12
    )
    _raise_invariant_error(
        data,
        effective_exposure_invalid,
        "effective_european_exposure must not exceed raw exposure",
    )

    achievement_outside = (data["domestic_achievement_score"] < -1e-12) | (
        data["domestic_achievement_score"] > config.achievement_cap + 1e-12
    )
    _raise_invariant_error(
        data,
        achievement_outside,
        "domestic_achievement_score must respect achievement_cap",
    )

    invalid_saturation_count = (
        (data["saturation_count"] < 0)
        | (data["saturation_count"] > 4)
        | (data["saturation_count"] % 1 != 0)
    )
    _raise_invariant_error(
        data,
        invalid_saturation_count,
        "saturation_count must be an integer between 0 and 4",
    )

    adjustment_outside = (
        data["domestic_surprise_domestic_adjustment"].abs()
        > config.domestic_surprise_max_abs_adjustment + 1e-9
    )
    _raise_invariant_error(
        data,
        adjustment_outside,
        "domestic surprise adjustment must respect its absolute cap",
    )
    expected_first_adjustment = (
        (1.0 - data["effective_european_exposure"])
        * data["domestic_surprise_domestic_adjustment"]
    )
    adjustment_identity_invalid = (
        data["domestic_surprise_ao_first_elo_adjustment"]
        - expected_first_adjustment
    ).abs() > 1e-9
    _raise_invariant_error(
        data,
        adjustment_identity_invalid,
        "AO First Elo surprise adjustment must preserve exposure identity",
    )

    lower_prior = data[["adjusted_domestic_prior", "european_prior"]].min(axis=1)
    upper_prior = data[["adjusted_domestic_prior", "european_prior"]].max(axis=1)
    outside_priors = (data["ao_first_elo"] < lower_prior - 1e-9) | (
        data["ao_first_elo"] > upper_prior + 1e-9
    )
    _raise_invariant_error(
        data,
        outside_priors,
        "ao_first_elo must stay between adjusted_domestic_prior and european_prior",
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
