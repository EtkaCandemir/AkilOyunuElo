from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from ao_elo.config import SEASON_KEYS
from ao_elo.features import as_bool, is_missing


TEAM_COLUMNS = {
    "team_id",
    "team_name",
    "country",
    "country_code",
    "domestic_league",
}

COUNTRY_COLUMNS = {
    "season",
    "country",
    "country_code",
    "points_t_minus_4",
    "points_t_minus_3",
    "points_t_minus_2",
    "points_t_minus_1",
    "points_t",
    "official_five_year_total",
    "official_country_rank",
}

DOMESTIC_COLUMNS = {
    "season",
    "team_id",
    "domestic_position",
    "league_team_count",
    "is_league_champion",
    "is_cup_winner",
    "european_entry_type",
}

CLUB_BASE_COLUMNS = {
    "season",
    "team_id",
    "team_name_source",
    "country_code",
    "official_club_coefficient",
    "country_part",
}

CLUB_SEASON_COLUMNS = {
    f"{prefix}_{key}"
    for prefix in ("club_points", "played", "matches", "match_cap")
    for key in SEASON_KEYS
}

CLUB_COLUMNS = CLUB_BASE_COLUMNS | CLUB_SEASON_COLUMNS


def require_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {', '.join(missing)}")


def validate_input_columns(
    teams: pd.DataFrame,
    country_coefficients: pd.DataFrame,
    domestic_context: pd.DataFrame,
    club_european_points: pd.DataFrame,
) -> None:
    require_columns(teams, TEAM_COLUMNS, "teams.csv")
    require_columns(country_coefficients, COUNTRY_COLUMNS, "country_coefficients.csv")
    require_columns(domestic_context, DOMESTIC_COLUMNS, "domestic_context.csv")
    require_columns(club_european_points, CLUB_COLUMNS, "club_european_points.csv")


def validate_domestic_context(domestic_context: pd.DataFrame) -> list[list[str]]:
    """Validate domestic rows and return non-blocking warning lists."""
    warnings_by_row: list[list[str]] = []
    for index, row in domestic_context.iterrows():
        row_warnings: list[str] = []
        position = row["domestic_position"]
        team_count = row["league_team_count"]

        position_known = not is_missing(position)
        team_count_known = not is_missing(team_count)

        if position_known:
            if float(position) < 1:
                raise ValueError(f"domestic row {index}: domestic_position must be >= 1")
            if not float(position).is_integer():
                raise ValueError(
                    f"domestic row {index}: domestic_position must be an integer"
                )

        if team_count_known:
            if float(team_count) < 1:
                raise ValueError(f"domestic row {index}: league_team_count must be >= 1")
            if not float(team_count).is_integer():
                raise ValueError(
                    f"domestic row {index}: league_team_count must be an integer"
                )

        if position_known and team_count_known and float(position) > float(team_count):
            raise ValueError(
                f"domestic row {index}: domestic_position must be <= league_team_count"
            )

        if (
            as_bool(row["is_league_champion"])
            and position_known
            and float(position) != 1.0
        ):
            row_warnings.append(
                "is_league_champion is true but domestic_position is not 1; "
                "use final standings if possible"
            )

        warnings_by_row.append(row_warnings)
    return warnings_by_row


def validate_club_european_points(club_european_points: pd.DataFrame) -> None:
    """Fail on inconsistent played/matches/points season data."""
    for index, row in club_european_points.iterrows():
        for key in SEASON_KEYS:
            played = row[f"played_{key}"]
            matches = row[f"matches_{key}"]
            club_points = row[f"club_points_{key}"]
            match_cap = row[f"match_cap_{key}"]

            if is_missing(played) or float(played) not in {0.0, 1.0}:
                raise ValueError(f"club row {index}: played_{key} must be 0 or 1")

            if is_missing(matches) or float(matches) < 0:
                raise ValueError(
                    f"club row {index}: matches_{key} must be non-negative"
                )
            if not float(matches).is_integer():
                raise ValueError(f"club row {index}: matches_{key} must be an integer")

            if is_missing(club_points) or float(club_points) < 0:
                raise ValueError(
                    f"club row {index}: club_points_{key} must be non-negative"
                )

            if float(played) == 0.0:
                if float(matches) != 0.0:
                    raise ValueError(
                        f"club row {index}: matches_{key} must be 0 when played_{key}=0"
                    )
                if float(club_points) != 0.0:
                    raise ValueError(
                        f"club row {index}: club_points_{key} must be 0 when "
                        f"played_{key}=0"
                    )

            if float(played) == 1.0 and float(matches) < 1.0:
                raise ValueError(
                    f"club row {index}: matches_{key} must be >= 1 when played_{key}=1"
                )

            if float(matches) > 0.0 and (
                is_missing(match_cap) or float(match_cap) <= 0.0
            ):
                raise ValueError(
                    f"club row {index}: match_cap_{key} must be > 0 when "
                    f"matches_{key}>0"
                )

