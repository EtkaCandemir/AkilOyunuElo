from __future__ import annotations

from collections.abc import Iterable
from math import isfinite
from typing import Any

import pandas as pd

from ao_elo.config import SEASON_KEYS
from ao_elo.features import is_missing


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
}

COUNTRY_AUDIT_COLUMNS = {
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

DOMESTIC_HISTORY_KEYS = tuple(f"t_minus_{offset}" for offset in range(5, 0, -1))
DOMESTIC_HISTORY_POSITION_COLUMNS = {
    f"history_position_{key}" for key in DOMESTIC_HISTORY_KEYS
}
DOMESTIC_HISTORY_TEAM_COUNT_COLUMNS = {
    f"history_team_count_{key}" for key in DOMESTIC_HISTORY_KEYS
}

CLUB_BASE_COLUMNS = {
    "season",
    "team_id",
    "team_name_source",
    "country_code",
}

CLUB_AUDIT_COLUMNS = {
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


def validate_teams(teams: pd.DataFrame) -> None:
    validate_unique_key(teams, ["team_id"], "teams.csv")
    _require_non_empty_values(teams, TEAM_COLUMNS, "teams.csv")


def validate_country_coefficients(country_coefficients: pd.DataFrame) -> None:
    validate_unique_key(
        country_coefficients,
        ["season", "country_code"],
        "country_coefficients.csv",
    )
    _require_non_empty_values(
        country_coefficients,
        ["season", "country", "country_code"],
        "country_coefficients.csv",
    )

    for index, row in country_coefficients.iterrows():
        key = _row_key(row, ["season", "country_code"])
        for season_key in SEASON_KEYS:
            _require_number(
                row[f"points_{season_key}"],
                label=f"country_coefficients.csv {key} points_{season_key}",
                minimum=0,
            )

        if "official_five_year_total" in country_coefficients.columns and not is_missing(
            row["official_five_year_total"]
        ):
            _require_number(
                row["official_five_year_total"],
                label=f"country_coefficients.csv {key} official_five_year_total",
                minimum=0,
            )

        if "official_country_rank" in country_coefficients.columns and not is_missing(
            row["official_country_rank"]
        ):
            rank = _require_number(
                row["official_country_rank"],
                label=f"country_coefficients.csv {key} official_country_rank",
                minimum=1,
            )
            if not rank.is_integer():
                raise ValueError(
                    f"country_coefficients.csv {key} official_country_rank "
                    "must be an integer"
                )


def validate_domestic_context(domestic_context: pd.DataFrame) -> list[list[str]]:
    """Validate domestic rows and return non-blocking warning lists."""
    validate_unique_key(
        domestic_context,
        ["season", "team_id"],
        "domestic_context.csv",
    )
    _require_non_empty_values(
        domestic_context,
        ["season", "team_id", "european_entry_type"],
        "domestic_context.csv",
    )

    target_seasons = domestic_context["season"].dropna().astype(str).str.strip().unique()
    if len(target_seasons) > 1:
        seasons = ", ".join(sorted(target_seasons))
        raise ValueError(
            "domestic_context.csv must contain exactly one target season; "
            f"found: {seasons}"
        )

    for key in DOMESTIC_HISTORY_KEYS:
        position_column = f"history_position_{key}"
        count_column = f"history_team_count_{key}"
        if (position_column in domestic_context.columns) != (
            count_column in domestic_context.columns
        ):
            raise ValueError(
                "domestic_context.csv historical finish columns must be supplied in "
                f"pairs: {position_column}, {count_column}"
            )

    warnings_by_row: list[list[str]] = []
    for _, row in domestic_context.iterrows():
        row_warnings: list[str] = []
        key = _row_key(row, ["season", "team_id"])
        position = row["domestic_position"]
        team_count = row["league_team_count"]

        champion = _require_boolean(
            row["is_league_champion"],
            f"domestic_context.csv {key} is_league_champion",
        )
        _require_boolean(
            row["is_cup_winner"],
            f"domestic_context.csv {key} is_cup_winner",
        )

        position_known = not is_missing(position)
        team_count_known = not is_missing(team_count)

        position_value: float | None = None
        team_count_value: float | None = None
        if position_known:
            position_value = _require_number(
                position,
                label=f"domestic_context.csv {key} domestic_position",
                minimum=1,
            )
            if not position_value.is_integer():
                raise ValueError(
                    f"domestic_context.csv {key} domestic_position must be an integer"
                )
            if not team_count_known:
                raise ValueError(
                    f"domestic_context.csv {key} league_team_count is required when "
                    "domestic_position is provided"
                )

        if team_count_known:
            team_count_value = _require_number(
                team_count,
                label=f"domestic_context.csv {key} league_team_count",
                minimum=2,
            )
            if not team_count_value.is_integer():
                raise ValueError(
                    f"domestic_context.csv {key} league_team_count must be an integer"
                )

        if (
            position_value is not None
            and team_count_value is not None
            and position_value > team_count_value
        ):
            raise ValueError(
                f"domestic_context.csv {key} domestic_position must be <= "
                "league_team_count"
            )

        if champion and position_value is not None and position_value != 1.0:
            raise ValueError(
                f"domestic_context.csv {key} is_league_champion=true requires "
                "domestic_position=1 when domestic_position is provided"
            )

        for history_key in DOMESTIC_HISTORY_KEYS:
            history_position_column = f"history_position_{history_key}"
            history_count_column = f"history_team_count_{history_key}"
            if history_position_column not in domestic_context.columns:
                continue
            history_position = row[history_position_column]
            history_count = row[history_count_column]
            history_position_missing = is_missing(history_position)
            history_count_missing = is_missing(history_count)
            if history_position_missing and history_count_missing:
                continue
            if history_position_missing != history_count_missing:
                raise ValueError(
                    f"domestic_context.csv {key} {history_key} historical position "
                    "and team count must both be provided or both be empty"
                )
            history_position_value = _require_number(
                history_position,
                label=(
                    f"domestic_context.csv {key} {history_position_column}"
                ),
                minimum=1,
            )
            history_count_value = _require_number(
                history_count,
                label=f"domestic_context.csv {key} {history_count_column}",
                minimum=2,
            )
            if not history_position_value.is_integer():
                raise ValueError(
                    f"domestic_context.csv {key} {history_position_column} "
                    "must be an integer"
                )
            if not history_count_value.is_integer():
                raise ValueError(
                    f"domestic_context.csv {key} {history_count_column} must be an integer"
                )
            if history_position_value > history_count_value:
                raise ValueError(
                    f"domestic_context.csv {key} {history_position_column} must be <= "
                    f"{history_count_column}"
                )

        warnings_by_row.append(row_warnings)
    return warnings_by_row


def validate_club_european_points(club_european_points: pd.DataFrame) -> None:
    """Fail on inconsistent played/matches/points season data."""
    validate_unique_key(
        club_european_points,
        ["season", "team_id", "country_code"],
        "club_european_points.csv",
    )
    _require_non_empty_values(
        club_european_points,
        CLUB_BASE_COLUMNS,
        "club_european_points.csv",
    )

    for _, row in club_european_points.iterrows():
        key = _row_key(row, ["season", "team_id", "country_code"])
        for season_key in SEASON_KEYS:
            played = _require_boolean(
                row[f"played_{season_key}"],
                f"club_european_points.csv {key} played_{season_key}",
            )
            matches = _require_number(
                row[f"matches_{season_key}"],
                label=f"club_european_points.csv {key} matches_{season_key}",
                minimum=0,
            )
            if not matches.is_integer():
                raise ValueError(
                    f"club_european_points.csv {key} matches_{season_key} "
                    "must be an integer"
                )

            club_points = _require_number(
                row[f"club_points_{season_key}"],
                label=f"club_european_points.csv {key} club_points_{season_key}",
                minimum=0,
            )
            _require_number(
                row[f"match_cap_{season_key}"],
                label=f"club_european_points.csv {key} match_cap_{season_key}",
                minimum=0,
                strictly_greater=True,
            )

            if not played and matches != 0.0:
                raise ValueError(
                    f"club_european_points.csv {key} matches_{season_key} must be 0 "
                    f"when played_{season_key}=0"
                )
            if not played and club_points != 0.0:
                raise ValueError(
                    f"club_european_points.csv {key} club_points_{season_key} must be "
                    f"0 when played_{season_key}=0"
                )
            if played and matches < 1.0:
                raise ValueError(
                    f"club_european_points.csv {key} matches_{season_key} must be >= 1 "
                    f"when played_{season_key}=1"
                )

        for column in CLUB_AUDIT_COLUMNS & set(club_european_points.columns):
            if not is_missing(row[column]):
                _require_number(
                    row[column],
                    label=f"club_european_points.csv {key} {column}",
                    minimum=0,
                )


def validate_required_club_history_rows(
    teams_with_season: pd.DataFrame,
    club_european_points: pd.DataFrame,
) -> None:
    """Require an explicit European-history row for every target team."""
    key_columns = ["season", "team_id", "country_code"]
    expected = teams_with_season[key_columns].drop_duplicates()
    actual = club_european_points[key_columns].drop_duplicates()
    missing = expected.merge(actual, on=key_columns, how="left", indicator=True)
    missing = missing.loc[missing["_merge"] == "left_only", key_columns]
    if missing.empty:
        return

    keys = "; ".join(_row_key(row, key_columns) for _, row in missing.iterrows())
    raise ValueError(
        "club_european_points.csv missing explicit history row(s): "
        f"{keys}. Teams with no European history must provide an all-zero row."
    )


def validate_unique_key(df: pd.DataFrame, columns: list[str], label: str) -> None:
    null_rows = df.loc[df[columns].isna().any(axis=1), columns]
    if not null_rows.empty:
        indexes = ", ".join(null_rows.index.astype(str))
        raise ValueError(
            f"{label} key columns {', '.join(columns)} must not be missing "
            f"(row index(es): {indexes})"
        )

    duplicates = df.loc[df.duplicated(columns, keep=False), columns].drop_duplicates()
    if not duplicates.empty:
        keys = "; ".join(_row_key(row, columns) for _, row in duplicates.iterrows())
        raise ValueError(f"{label} duplicate key(s): {keys}")


def _require_non_empty_values(
    df: pd.DataFrame,
    columns: Iterable[str],
    label: str,
) -> None:
    for column in columns:
        for index, value in df[column].items():
            if is_missing(value) or (isinstance(value, str) and not value.strip()):
                raise ValueError(f"{label} row {index} {column} must not be empty")


def _require_number(
    value: Any,
    *,
    label: str,
    minimum: float | None = None,
    strictly_greater: bool = False,
) -> float:
    if is_missing(value):
        raise ValueError(f"{label} must not be missing")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not isfinite(number):
        raise ValueError(f"{label} must be finite")
    if minimum is not None:
        invalid = number <= minimum if strictly_greater else number < minimum
        if invalid:
            operator = ">" if strictly_greater else ">="
            raise ValueError(f"{label} must be {operator} {minimum:g}")
    return number


def _require_boolean(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if is_missing(value):
        raise ValueError(f"{label} must not be missing")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
        raise ValueError(f"{label} must be true/false or 0/1")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be true/false or 0/1") from exc
    if isfinite(number) and number in {0.0, 1.0}:
        return bool(number)
    raise ValueError(f"{label} must be true/false or 0/1")


def _row_key(row: pd.Series, columns: list[str]) -> str:
    return ", ".join(f"{column}={row[column]}" for column in columns)
