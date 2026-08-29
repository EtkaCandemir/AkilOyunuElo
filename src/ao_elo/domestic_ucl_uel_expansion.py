"""Audits for the UCL/UEL domestic-Poisson coverage expansion.

The helpers intentionally do not alter a production artifact.  They turn the
current UEFA participant snapshot and an independently built domestic dataset
into inspectable coverage decisions for the candidate review.
"""

from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from ao_elo.domestic_league_dataset import (
    DomesticLeagueSpec,
    SCHEDULE_COLUMNS,
    ao_season_for_provider_event,
)
from ao_elo.validators import (
    require_utc_column,
    require_utc_timestamp,
    validate_domestic_fixture_uniqueness,
)
from ao_elo.xg_dataset import normalize_name


DIRECT_ENTRY_ROUNDS = frozenset({"CL-LS", "EL-LS"})
PRIORITY_COMPETITIONS = frozenset({"UCL", "UEL"})
TARGET_COUNTRY_CODES = frozenset(
    {"ALB", "ARM", "AZB", "BUL", "CRO", "GEO", "HUN", "KAZ", "LIT", "ROM", "SCO", "SLO", "SRB", "SVK", "UKR"}
)


# The historical fixture archive stores the API-Football competition identifier.
# These IDs were resolved against its frozen leagues.parquet catalogue on
# 2026-08-21.  Albania is intentionally absent: the archive has no Albanian
# top-flight competition, while the required target (Egnatia) has primary data.
SECONDARY_LEAGUE_IDS: Mapping[str, int] = {
    "ARM": 144,
    "AZB": 59,
    "BUL": 66,
    "CRO": 43,
    "GEO": 71,
    "HUN": 47,
    "KAZ": 132,
    "LIT": 134,
    "ROM": 32,
    "SCO": 17,
    "SLO": 55,
    "SRB": 63,
    "SVK": 54,
    "UKR": 61,
}


# The existing domestic source predates a number of UEFA's current canonical
# names.  These aliases are deliberately local to the candidate bridge: each
# one is checked against the frozen 80-team universe and an exact country/name
# match before it can be applied.  They do not mutate club_registry.csv.
VERIFIED_TARGET_ALIASES: Mapping[str, tuple[str, str]] = {
    "AO-UEFA-52883": ("BEL", "St.Truiden"),
    "AO-UEFA-52498": ("CZE", "Slavia Prague"),
    "AO-UEFA-2601124": ("ENG", "Bournemouth"),
    "AO-UEFA-53360": ("ENG", "Sunderland"),
    "AO-UEFA-50124": ("ESP", "Ath Madrid"),
    "AO-UEFA-53043": ("ESP", "Celta Vigo"),
    "AO-UEFA-52265": ("ESP", "Betis"),
    "AO-UEFA-50123": ("ESP", "Sociedad"),
    "AO-UEFA-52747": ("FRA", "Paris SG"),
    "AO-UEFA-50037": ("GER", "Bayern Munich"),
    "AO-UEFA-52758": ("GER", "Dortmund"),
    "AO-UEFA-2603790": ("GER", "RasenBallsport Leipzig"),
    "AO-UEFA-50129": ("GRE", "AEK"),
    "AO-UEFA-52953": ("GRE", "OFI"),
    "AO-UEFA-79946": ("ITA", "Como"),
    "AO-UEFA-52330": ("NED", "Nijmegen"),
    "AO-UEFA-59333": ("NOR", "Bodoe/Glimt"),
    "AO-UEFA-52314": ("NOR", "Lillestroem"),
    "AO-UEFA-2603107": ("POR", "Torreense"),
    "AO-UEFA-74233": ("SWE", "Mjallby"),
}

# Explicitly rejected false positive produced by the legacy short-name matcher.
# Paris FC is not Paris Saint-Germain and has no current AO target identity.
REJECTED_SOURCE_NAME_MAPPINGS: tuple[tuple[str, str, str], ...] = (
    ("FRA", "Paris FC", "AO-UEFA-52747"),
)


def apply_verified_target_aliases(
    bridge: pd.DataFrame,
    targets: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply auditable, frozen source-name aliases for priority teams only."""

    _require(
        bridge,
        {"country_code", "source_team_id", "source_team_name", "ao_club_id", "identity_method", "identity_ambiguous"},
        "bridge",
    )
    _require(targets, {"ao_club_id", "country_code", "team_name"}, "targets")
    target_index = targets.set_index("ao_club_id")[["country_code", "team_name"]]
    result = bridge.copy()
    audit_rows: list[dict[str, object]] = []
    for country, source_name, incorrect_club_id in REJECTED_SOURCE_NAME_MAPPINGS:
        rejected = result.loc[
            result["country_code"].astype(str).eq(country)
            & result["source_team_name"].fillna("").map(normalize_name).eq(normalize_name(source_name))
            & result["ao_club_id"].fillna("").astype(str).eq(incorrect_club_id)
        ]
        if not rejected.empty:
            result.loc[rejected.index, "ao_club_id"] = pd.NA
            result.loc[rejected.index, "identity_method"] = "REJECTED_FALSE_POSITIVE"
            result.loc[rejected.index, "identity_ambiguous"] = True
    for club_id, (country, alias) in VERIFIED_TARGET_ALIASES.items():
        if club_id not in target_index.index:
            continue
        target = target_index.loc[club_id]
        if str(target.country_code) != country:
            raise ValueError(f"Verified alias country conflicts with target: {club_id}")
        matching = result.loc[
            result["country_code"].astype(str).eq(country)
            & result["source_team_name"].fillna("").map(normalize_name).eq(normalize_name(alias))
        ]
        if matching.empty:
            audit_rows.append(
                {
                    "ao_club_id": club_id,
                    "country_code": country,
                    "ao_team_name": str(target.team_name),
                    "source_alias": alias,
                    "source_rows": 0,
                    "override_status": "SOURCE_ALIAS_ABSENT",
                }
            )
            continue
        conflicts = matching["ao_club_id"].fillna("").astype(str)
        conflicts = conflicts[(conflicts.ne("") & conflicts.ne(club_id))]
        if not conflicts.empty:
            raise ValueError(f"Verified alias conflicts with existing club mapping: {club_id}")
        result.loc[matching.index, "ao_club_id"] = club_id
        result.loc[matching.index, "identity_method"] = "VERIFIED_TARGET_ALIAS"
        result.loc[matching.index, "identity_score"] = 1.0
        result.loc[matching.index, "runner_up_score"] = 0.0
        result.loc[matching.index, "identity_ambiguous"] = False
        audit_rows.append(
            {
                "ao_club_id": club_id,
                "country_code": country,
                "ao_team_name": str(target.team_name),
                "source_alias": alias,
                "source_rows": int(len(matching)),
                "override_status": "APPLIED",
            }
        )
    audit = pd.DataFrame(audit_rows).sort_values(["country_code", "ao_team_name"], kind="stable")
    if (audit["override_status"].eq("SOURCE_ALIAS_ABSENT")).any():
        absent = audit.loc[audit["override_status"].eq("SOURCE_ALIAS_ABSENT"), "ao_club_id"].tolist()
        raise ValueError(f"Verified target aliases missing from domestic source: {absent}")
    if result.duplicated(["country_code", "source_team_id"]).any():
        raise ValueError("Alias application produced duplicate provider identities")
    return result.sort_values(["country_code", "source_team_id"], kind="stable").reset_index(drop=True), audit.reset_index(drop=True)


def normalize_secondary_fixtures(
    fixtures: pd.DataFrame,
    teams: pd.DataFrame,
    spec: DomesticLeagueSpec,
    *,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Normalize one archived API-Football league into the domestic contract.

    The archive is a secondary source, so identifiers are source-qualified.  It
    contains completed fixtures only; quality acceptance happens separately at
    league-season level and never fills individual primary rows.
    """

    league_id = SECONDARY_LEAGUE_IDS.get(spec.country_code)
    if league_id is None:
        return pd.DataFrame(columns=SCHEDULE_COLUMNS)
    required = {
        "id", "date_utc", "league_id", "home_team_id", "away_team_id",
        "goals_home", "goals_away", "is_played",
    }
    missing = sorted(required.difference(fixtures.columns))
    if missing:
        raise ValueError(f"secondary fixtures missing columns: {missing}")
    _require(teams, {"id", "name"}, "secondary teams")
    names = teams.set_index("id")["name"].astype(str).to_dict()
    source = fixtures.loc[
        fixtures["league_id"].eq(league_id) & fixtures["is_played"].astype(bool)
    ].copy()
    rows: list[dict[str, object]] = []
    for row in source.itertuples(index=False):
        kickoff = pd.to_datetime(row.date_utc, utc=True, errors="coerce")
        home_goals = _strict_integer(row.goals_home)
        away_goals = _strict_integer(row.goals_away)
        if pd.isna(kickoff) or home_goals is None or away_goals is None:
            continue
        provider_season = _secondary_provider_season(row, spec, kickoff)
        season_start = int(provider_season[:4])
        if not start_year <= season_start <= end_year:
            continue
        ao_season = ao_season_for_provider_event(spec, provider_season, kickoff)
        home_id = f"HF:{row.home_team_id}"
        away_id = f"HF:{row.away_team_id}"
        if home_id == away_id or home_goals < 0 or away_goals < 0:
            continue
        event_id = f"HF:{row.id}"
        rows.append(
            {
                "match_id": f"HF-{league_id}-{row.id}",
                "source_event_id": event_id,
                "sportsdb_league_id": f"HF:{league_id}",
                "league_name": spec.league_name,
                "country_code": spec.country_code,
                "provider_season": provider_season,
                "ao_season": ao_season,
                "kickoff_utc": kickoff.isoformat(),
                "home_source_team_id": home_id,
                "away_source_team_id": away_id,
                "home_team_name": str(names.get(row.home_team_id, "")).strip(),
                "away_team_name": str(names.get(row.away_team_id, "")).strip(),
                "home_goals": home_goals,
                "away_goals": away_goals,
                "actual_home_score": 1.0 if home_goals > away_goals else 0.5 if home_goals == away_goals else 0.0,
                "status": "ARCHIVE_PLAYED",
                "round": pd.NA,
            }
        )
    result = pd.DataFrame(rows, columns=SCHEDULE_COLUMNS)
    if result.empty:
        return result
    if result["match_id"].duplicated().any() or result["source_event_id"].duplicated().any():
        raise ValueError(f"{spec.country_code}: duplicate secondary fixture ID")
    return result.sort_values(["kickoff_utc", "match_id"], kind="stable").reset_index(drop=True)


def assess_secondary_league_seasons(
    matches: pd.DataFrame,
    spec: DomesticLeagueSpec,
) -> pd.DataFrame:
    """Apply a source-wide 95% completeness gate without mixing providers.

    The public archive has no final-table endpoint.  For each archived league,
    a recurring modal fixture count is an inferred format expectation. When no
    count recurs, use the ceiling of the median, not the smallest singleton.
    This is a heuristic, not independently verified competition-format data.
    Each accepted season must reach 95% of that
    expectation and satisfy the same event, timestamp, score, and team checks.
    """

    columns = [
        "country_code", "sportsdb_league_id", "league_name", "provider_season",
        "schedule_matches", "table_expected_matches", "coverage_rate", "unique_events",
        "unique_teams", "timestamps_valid", "scores_valid", "quality_status",
        "quality_reason", "source_provider", "source_selection", "format_expectation_method",
    ]
    if matches.empty:
        return pd.DataFrame(columns=columns)
    season_counts = matches.groupby("provider_season", sort=True).size()
    # A count must recur to be a format expectation; isolated counts retain the
    # conservative median to avoid silently accepting partial archived seasons.
    counts = season_counts.astype(int).tolist()
    frequencies = pd.Series(counts).value_counts()
    if int(frequencies.max()) >= 2:
        expected = int(pd.Series(counts).mode().iloc[0])
        expectation_method = "RECURRING_MODE"
    else:
        expected = int(math.ceil(float(pd.Series(counts).median())))
        expectation_method = "MEDIAN_NO_RECURRING_MODE"
    rows: list[dict[str, object]] = []
    for provider_season, group in matches.groupby("provider_season", sort=True):
        count = int(len(group))
        unique_events = int(group["source_event_id"].nunique())
        unique_teams = int(pd.concat([group["home_source_team_id"], group["away_source_team_id"]]).nunique())
        timestamps_valid = bool(pd.to_datetime(group["kickoff_utc"], utc=True, errors="coerce").notna().all())
        scores_valid = bool(group[["home_goals", "away_goals"]].notna().all().all() and group[["home_goals", "away_goals"]].ge(0).all().all())
        coverage = count / expected
        accepted = bool(
            coverage >= 0.95 and count == unique_events and unique_teams >= 4
            and timestamps_valid and scores_valid
        )
        rows.append(
            {
                "country_code": spec.country_code,
                "sportsdb_league_id": f"HF:{SECONDARY_LEAGUE_IDS[spec.country_code]}",
                "league_name": spec.league_name,
                "provider_season": provider_season,
                "schedule_matches": count,
                "table_expected_matches": expected,
                "coverage_rate": coverage,
                "unique_events": unique_events,
                "unique_teams": unique_teams,
                "timestamps_valid": timestamps_valid,
                "scores_valid": scores_valid,
                "quality_status": "ACCEPTED" if accepted else "REJECTED",
                "quality_reason": "SECONDARY_INFERRED_FORMAT_ACCEPTED" if accepted else "SECONDARY_INFERRED_FORMAT_BELOW_95_PCT",
                "source_provider": "HF_API_FOOTBALL_ARCHIVE",
                "source_selection": "SECONDARY_ACCEPTED" if accepted else "UNAVAILABLE",
                "format_expectation_method": expectation_method,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def select_source_safe_seasons(
    primary_matches: pd.DataFrame,
    primary_quality: pd.DataFrame,
    secondary_matches: pd.DataFrame,
    secondary_quality: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose one accepted source per country and provider season.

    Primary is authoritative when it passes.  The secondary archive is used
    only for a whole rejected primary season; no rows are coalesced.
    """

    selected_frames: list[pd.DataFrame] = []
    selected_quality: list[pd.DataFrame] = []
    primary = primary_quality.copy()
    secondary = secondary_quality.copy()

    def _counterpart(frame: pd.DataFrame, country: str, season: str, prefix: str) -> dict[str, object]:
        """Carry the losing source's own verdict into the published audit row.

        Only one row survives per country/season, so without this the reason a
        fallback was refused is lost: a season the secondary rejected on coverage
        is published carrying the primary's fetch error instead.  It also makes
        the gate's source dependence visible -- the same league can hold two
        different `table_expected_matches`, so `ACCEPTED` alone never proves a
        season is complete.
        """

        match = frame.loc[
            frame["country_code"].eq(country) & frame["provider_season"].eq(season)
        ]
        if match.empty:
            return {
                f"{prefix}_quality_status": "ABSENT",
                f"{prefix}_quality_reason": pd.NA,
                f"{prefix}_schedule_matches": pd.NA,
                f"{prefix}_table_expected_matches": pd.NA,
                f"{prefix}_coverage_rate": pd.NA,
            }
        row = match.iloc[0]
        return {
            f"{prefix}_quality_status": row.get("quality_status", pd.NA),
            f"{prefix}_quality_reason": row.get("quality_reason", pd.NA),
            f"{prefix}_schedule_matches": row.get("schedule_matches", pd.NA),
            f"{prefix}_table_expected_matches": row.get("table_expected_matches", pd.NA),
            f"{prefix}_coverage_rate": row.get("coverage_rate", pd.NA),
        }

    keys = sorted(set(zip(primary["country_code"], primary["provider_season"])))
    for country, season in keys:
        primary_slice = primary.loc[
            primary["country_code"].eq(country) & primary["provider_season"].eq(season)
        ]
        primary_row = primary_slice.iloc[0]
        secondary_audit = _counterpart(secondary, country, season, "secondary")
        primary_audit = _counterpart(primary, country, season, "primary")
        if str(primary_row["quality_status"]) == "ACCEPTED":
            selected_frames.append(primary_matches.loc[
                primary_matches["country_code"].eq(country) & primary_matches["provider_season"].eq(season)
            ])
            selected_quality.append(
                primary_slice.assign(source_selection="PRIMARY_ACCEPTED", **secondary_audit)
            )
            continue
        alternate = secondary.loc[
            secondary["country_code"].eq(country) & secondary["provider_season"].eq(season)
            & secondary["quality_status"].eq("ACCEPTED")
        ]
        if not alternate.empty:
            selected_frames.append(secondary_matches.loc[
                secondary_matches["country_code"].eq(country) & secondary_matches["provider_season"].eq(season)
            ])
            selected_quality.append(
                alternate.assign(source_selection="SECONDARY_ACCEPTED", **primary_audit)
            )
        else:
            selected_quality.append(
                primary_slice.assign(source_selection="UNAVAILABLE", **secondary_audit)
            )
    selected = pd.concat(selected_frames, ignore_index=True, sort=False) if selected_frames else pd.DataFrame(columns=SCHEDULE_COLUMNS)
    quality = pd.concat(selected_quality, ignore_index=True, sort=False)
    if not selected.empty:
        mixed = selected.groupby(["country_code", "provider_season"])["source_provider"].nunique()
        if mixed.gt(1).any():
            raise ValueError("A candidate league-season cannot mix providers")
        selected = selected.sort_values(["kickoff_utc", "match_id"], kind="stable").reset_index(drop=True)
    return selected, quality.sort_values(["country_code", "provider_season"], kind="stable").reset_index(drop=True)


def select_causal_domestic_results(
    matches: pd.DataFrame,
    cutoff_utc: object,
) -> pd.DataFrame:
    """Return deterministic completed domestic events strictly before a cutoff."""

    _require(
        matches,
        {"match_id", "kickoff_utc", "home_goals", "away_goals"},
        "live domestic matches",
    )
    # A naive value here would be assumed UTC and could shift a match across the
    # causal cutoff, letting a result played after kickoff reach the prediction.
    cutoff = require_utc_timestamp(cutoff_utc, "live domestic cutoff_utc")
    result = matches.copy()
    result["kickoff_utc"] = require_utc_column(
        result["kickoff_utc"], "live domestic matches kickoff_utc"
    )
    if result["match_id"].astype(str).duplicated().any():
        raise ValueError("live domestic matches contain duplicate match_id values")
    if not result[["home_goals", "away_goals"]].notna().all().all():
        raise ValueError("live domestic matches must contain completed scores")
    if not result[["home_goals", "away_goals"]].ge(0).all().all():
        raise ValueError("live domestic matches cannot contain negative scores")
    return result.loc[result["kickoff_utc"].lt(cutoff)].sort_values(
        ["kickoff_utc", "match_id"], kind="stable"
    ).reset_index(drop=True)


def canonicalize_candidate_domestic_state(
    matches: pd.DataFrame,
    bridge: pd.DataFrame,
    *,
    source_switch_countries: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use stable internal league/team keys where a provider can change by season.

    A Poisson state is causal only when a club's selected primary and secondary
    records advance the same internal state.  Source IDs remain in match_id and
    source_event_id for provenance; engine keys are replaced only for mapped AO
    clubs in the explicitly selected expansion countries.
    """

    _require(
        matches,
        {"country_code", "sportsdb_league_id", "home_source_team_id", "away_source_team_id"},
        "candidate matches",
    )
    _require(
        bridge,
        {"country_code", "source_team_id", "ao_club_id"},
        "candidate bridge",
    )
    countries = {str(value) for value in source_switch_countries}
    mapping = {
        (str(row.country_code), str(row.source_team_id)): str(row.ao_club_id)
        for row in bridge.itertuples(index=False)
        if str(row.country_code) in countries
        and pd.notna(row.ao_club_id)
        and str(row.ao_club_id).strip()
    }
    result = matches.copy()
    for side in ("home", "away"):
        column = f"{side}_source_team_id"
        result[column] = [
            f"AO:{mapping[(str(country), str(team_id))]}"
            if (str(country), str(team_id)) in mapping
            else str(team_id)
            for country, team_id in zip(result["country_code"], result[column], strict=True)
        ]
    result["sportsdb_league_id"] = result["sportsdb_league_id"].astype(str)
    mask = result["country_code"].astype(str).isin(countries)
    result.loc[mask, "sportsdb_league_id"] = (
        "AO-DOMESTIC:" + result.loc[mask, "country_code"].astype(str)
    )

    bridge_result = bridge.copy()
    bridge_mask = bridge_result["country_code"].astype(str).isin(countries) & bridge_result["ao_club_id"].notna()
    bridge_result.loc[bridge_mask, "source_team_id"] = "AO:" + bridge_result.loc[bridge_mask, "ao_club_id"].astype(str)
    if "source_club_key" in bridge_result.columns:
        bridge_result.loc[bridge_mask, "source_club_key"] = "AO_CANONICAL:" + bridge_result.loc[bridge_mask, "ao_club_id"].astype(str)
    key = ["country_code", "source_team_id"]
    duplicates = bridge_result.duplicated(key, keep=False)
    if duplicates.any():
        for _, group in bridge_result.loc[duplicates].groupby(key, sort=False):
            clubs = group["ao_club_id"].fillna("").astype(str)
            if clubs[clubs.ne("")].nunique() > 1:
                raise ValueError("Canonical provider state key maps to multiple AO clubs")
        bridge_result = bridge_result.drop_duplicates(key, keep="first")
    return (
        result.sort_values(["kickoff_utc", "match_id"], kind="stable").reset_index(drop=True),
        bridge_result.sort_values(key, kind="stable").reset_index(drop=True),
    )


def _secondary_provider_season(row: object, spec: DomesticLeagueSpec, kickoff: pd.Timestamp) -> str:
    """Source season wins; a missing calendar season uses the UTC calendar year.

    The archive currently lacks season metadata. For winter leagues only, the
    fallback is a July boundary; an explicit source season preserves delayed
    fixtures in their actual competition season.
    """
    for field in ("provider_season", "season"):
        value = getattr(row, field, None)
        if value is None or pd.isna(value):
            continue
        raw = str(value).strip()
        if not raw:
            continue
        first = raw.replace("/", "-").split("-", 1)[0]
        start = _strict_integer(first)
        if start is None or not 1900 <= start <= 2200:
            raise ValueError(f"Invalid secondary provider season: {value!r}")
        return spec.provider_season(start)
    start = kickoff.year if spec.calendar_season or kickoff.month >= 7 else kickoff.year - 1
    return spec.provider_season(start)


def _strict_integer(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or not numeric.is_integer():
        return None
    return int(numeric)


def build_target_team_audit(
    teams: pd.DataFrame,
    domestic_context: pd.DataFrame,
    upcoming_fixtures: pd.DataFrame,
) -> pd.DataFrame:
    """Freeze the current UCL/UEL direct-entry plus play-off team universe."""

    _require(teams, {"team_id", "team_name", "country_code", "domestic_league"}, "teams")
    _require(domestic_context, {"team_id", "competition", "entry_round"}, "domestic_context")
    _require(upcoming_fixtures, {"competition", "home_team_id", "away_team_id"}, "upcoming_fixtures")

    selected: dict[str, set[str]] = defaultdict(set)
    for row in domestic_context.itertuples(index=False):
        if str(row.entry_round) in DIRECT_ENTRY_ROUNDS:
            selected[str(row.team_id)].add(f"DIRECT_{row.entry_round}")
    for row in upcoming_fixtures.itertuples(index=False):
        if str(row.competition) not in PRIORITY_COMPETITIONS:
            continue
        selected[str(row.home_team_id)].add(f"UPCOMING_{row.competition}_PLAYOFF")
        selected[str(row.away_team_id)].add(f"UPCOMING_{row.competition}_PLAYOFF")

    indexed = teams.copy()
    indexed["team_id"] = indexed["team_id"].astype(str)
    if indexed["team_id"].duplicated().any():
        raise ValueError("teams.team_id must be unique")
    unknown = sorted(set(selected).difference(indexed["team_id"]))
    if unknown:
        raise ValueError(f"UCL/UEL target contains unknown team_id values: {unknown[:5]}")
    result = indexed.loc[indexed["team_id"].isin(selected)].copy()
    result["selection_reason"] = result["team_id"].map(
        lambda value: "|".join(sorted(selected[str(value)]))
    )
    result["is_expansion_target_country"] = result["country_code"].isin(TARGET_COUNTRY_CODES)
    result = result.rename(columns={"team_id": "ao_club_id"})
    result = result.sort_values(["country_code", "team_name", "ao_club_id"], kind="stable")
    return result.reset_index(drop=True)


def target_domestic_coverage_audit(
    targets: pd.DataFrame,
    bridge: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    minimum_matches: int = 40,
    minimum_seasons: int = 2,
) -> pd.DataFrame:
    """Return explicit target-club coverage and activation eligibility."""

    if minimum_matches < 1 or minimum_seasons < 1:
        raise ValueError("minimum coverage thresholds must be positive")
    _require(targets, {"ao_club_id", "country_code"}, "targets")
    _require(
        bridge,
        {"country_code", "source_team_id", "ao_club_id", "identity_ambiguous"},
        "bridge",
    )
    _require(
        matches,
        {"country_code", "ao_season", "home_source_team_id", "away_source_team_id"},
        "matches",
    )

    bridge_clean = bridge.copy()
    bridge_clean["ao_club_id"] = bridge_clean["ao_club_id"].fillna("").astype(str)
    mapped = bridge_clean.loc[
        bridge_clean["ao_club_id"].ne("") & ~bridge_clean["identity_ambiguous"].astype(bool)
    ].copy()
    provider_to_club = {
        (str(row.country_code), str(row.source_team_id)): str(row.ao_club_id)
        for row in mapped.itertuples(index=False)
    }
    ambiguous_clubs = set(
        bridge_clean.loc[
            bridge_clean["identity_ambiguous"].astype(bool) & bridge_clean["ao_club_id"].ne(""),
            "ao_club_id",
        ].astype(str)
    )

    appearances: dict[str, int] = defaultdict(int)
    seasons: dict[str, set[str]] = defaultdict(set)
    for row in matches.itertuples(index=False):
        country = str(row.country_code)
        season = str(row.ao_season)
        for source_id in (row.home_source_team_id, row.away_source_team_id):
            club_id = provider_to_club.get((country, str(source_id)))
            if club_id is not None:
                appearances[club_id] += 1
                seasons[club_id].add(season)

    rows: list[dict[str, object]] = []
    for target in targets.itertuples(index=False):
        club_id = str(target.ao_club_id)
        matches_count = appearances[club_id]
        seasons_count = len(seasons[club_id])
        mapped_rows = mapped.loc[mapped["ao_club_id"].eq(club_id)]
        exact_target_country = mapped_rows["country_code"].astype(str).eq(str(target.country_code)).any()
        # A club can legitimately have one primary and one secondary provider ID.
        # Identity is decisive when every accepted provider row maps to this same
        # club/country and none of its provider identities is ambiguous.
        identity_ok = bool(
            len(mapped_rows) >= 1
            and exact_target_country
            and club_id not in ambiguous_clubs
        )
        eligible = bool(identity_ok and matches_count >= minimum_matches and seasons_count >= minimum_seasons)
        rows.append(
            {
                "ao_club_id": club_id,
                "country_code": str(target.country_code),
                "team_name": getattr(target, "team_name", ""),
                "selection_reason": getattr(target, "selection_reason", ""),
                "provider_identity_rows": int(len(mapped_rows)),
                "identity_ok": identity_ok,
                "accepted_domestic_matches": int(matches_count),
                "accepted_domestic_seasons": int(seasons_count),
                "minimum_matches": int(minimum_matches),
                "minimum_seasons": int(minimum_seasons),
                "candidate_state_eligible": eligible,
                "coverage_status": "COVERED" if eligible else "FALLBACK_REQUIRED",
            }
        )
    result = pd.DataFrame(rows).sort_values(["country_code", "team_name", "ao_club_id"], kind="stable")
    if result["ao_club_id"].duplicated().any():
        raise ValueError("target coverage audit must contain one row per AO club")
    return result.reset_index(drop=True)


def merge_domestic_candidate(
    existing: pd.DataFrame,
    expansion: pd.DataFrame,
) -> pd.DataFrame:
    """Merge accepted source data without silently replacing existing rows."""

    required = {
        "match_id", "source_event_id", "sportsdb_league_id", "country_code",
        "ao_season", "kickoff_utc", "home_source_team_id", "away_source_team_id",
        "home_goals", "away_goals",
    }
    _require(existing, required, "existing")
    _require(expansion, required, "expansion")
    overlap = set(existing["match_id"].astype(str)).intersection(expansion["match_id"].astype(str))
    if overlap:
        raise ValueError(f"candidate merge has duplicate match_id values: {sorted(overlap)[:5]}")
    result = pd.concat([existing, expansion], ignore_index=True, sort=False)
    if result["source_event_id"].astype(str).duplicated().any():
        # Provider event IDs are only provider-local.  A source-qualified match ID is
        # canonical, while this check remains meaningful only within a league.
        duplicate = result.duplicated(["sportsdb_league_id", "source_event_id"])
        if duplicate.any():
            raise ValueError("candidate merge has duplicate provider event IDs within a league")
    # `replay_domestic_poisson_state` refuses naive timestamps, but coercing them
    # to UTC here defeated that guard: the merge is what feeds the checkpoint.
    result["kickoff_utc"] = require_utc_column(
        result["kickoff_utc"], "candidate merge kickoff_utc"
    )
    validate_domestic_fixture_uniqueness(result, "candidate merge")
    return result.sort_values(["kickoff_utc", "match_id"], kind="stable").reset_index(drop=True)


def _require(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")
