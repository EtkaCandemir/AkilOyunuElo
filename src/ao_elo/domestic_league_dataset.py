from __future__ import annotations

import math
import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from ao_elo.xg_dataset import normalize_name
from ao_elo.validators import DOMESTIC_FIXTURE_COLUMNS, domestic_fixture_keys, validate_domestic_fixture_uniqueness


FIXTURE_RECONCILIATIONS = Path(__file__).resolve().parents[2] / "data/domestic_fixture_reconciliations.json"
FIXTURE_AUDIT_COLUMNS = [*DOMESTIC_FIXTURE_COLUMNS, "source_event_id", "home_goals", "away_goals", "action", "retained_source_event_id", "reason", "source_url"]
VERIFIED_DOMESTIC_PROVIDER_ALIASES = {
    ("UKR", "dynamo kiev"): "AO-UEFA-52723",
    ("UKR", "vorskla"): "AO-UEFA-62174",
    ("UKR", "zorya"): "AO-UEFA-65130",
}


class DomesticFixtureConflictError(ValueError):
    """Unresolved observations must abort a build, not hide an entire season."""


def reconcile_domestic_fixture_observations(
    matches: pd.DataFrame, audit: list[dict[str, object]],
) -> pd.DataFrame:
    """Audit identical observations and apply only reviewed score resolutions.

    This is a source-normalization operation, never a replay/merge dedup policy.
    A new conflict, or changed evidence for a reviewed conflict, fails closed.
    """
    if matches.empty:
        return matches.copy()
    keys = domestic_fixture_keys(matches)
    duplicates = keys.duplicated(keep=False)
    if not duplicates.any():
        return matches.copy()
    resolutions = json.loads(FIXTURE_RECONCILIATIONS.read_text(encoding="utf-8"))["resolutions"]
    remove: list[int] = []
    for key, group in keys.loc[duplicates].groupby(list(DOMESTIC_FIXTURE_COLUMNS), sort=True):
        observations = matches.loc[group.index].sort_values("source_event_id", kind="stable")
        scores = {
            str(row.source_event_id): [int(row.home_goals), int(row.away_goals)]
            for row in observations.itertuples(index=False)
        }
        keep = str(observations.iloc[0]["source_event_id"])
        reason = "IDENTICAL_FIXTURE_AND_SCORE"
        source_url = ""
        if len({tuple(score) for score in scores.values()}) > 1:
            rule = next((rule for rule in resolutions if (
                str(rule["sportsdb_league_id"]), pd.Timestamp(rule["kickoff_utc"]),
                str(rule["home_source_team_id"]), str(rule["away_source_team_id"]),
            ) == key), None)
            if rule is None or rule["observed_scores"] != scores:
                raise DomesticFixtureConflictError(f"Unresolved domestic fixture score conflict: {key}, {scores}")
            keep = rule["keep_source_event_id"]
            if scores[keep] != rule["verified_score"]:
                raise DomesticFixtureConflictError("Reviewed fixture score and retained observation disagree")
            reason = "OFFICIAL_SOURCE_RECONCILIATION"
            source_url = rule["source_url"]
        for index, row in observations.iterrows():
            retained = str(row["source_event_id"]) == keep
            audit.append({
                **{column: row[column] for column in DOMESTIC_FIXTURE_COLUMNS},
                "source_event_id": str(row["source_event_id"]),
                "home_goals": row["home_goals"], "away_goals": row["away_goals"],
                "action": "KEEP" if retained else "REMOVE_OBSERVATION",
                "retained_source_event_id": keep, "reason": reason, "source_url": source_url,
            })
            if not retained:
                remove.append(index)
    result = matches.drop(index=remove).reset_index(drop=True)
    validate_domestic_fixture_uniqueness(result, "reconciled source schedule")
    return result


CALENDAR_SEASON_COUNTRIES = frozenset({"NOR", "SWE"})


@dataclass(frozen=True, order=True)
class DomesticLeagueSpec:
    country_code: str
    sportsdb_league_id: str
    league_name: str
    calendar_season: bool = False

    def provider_season(self, start_year: int) -> str:
        if self.calendar_season:
            return str(start_year)
        return f"{start_year}-{start_year + 1}"


PILOT_LEAGUES: tuple[DomesticLeagueSpec, ...] = (
    DomesticLeagueSpec("ENG", "4328", "English Premier League"),
    DomesticLeagueSpec("ITA", "4332", "Italian Serie A"),
    DomesticLeagueSpec("ESP", "4335", "Spanish La Liga"),
    DomesticLeagueSpec("GER", "4331", "German Bundesliga"),
    DomesticLeagueSpec("FRA", "4334", "French Ligue 1"),
    DomesticLeagueSpec("NED", "4337", "Dutch Eredivisie"),
    DomesticLeagueSpec("POR", "4344", "Portuguese Primeira Liga"),
    DomesticLeagueSpec("BEL", "4338", "Belgian Pro League"),
    DomesticLeagueSpec("CZE", "4631", "Czech First League"),
    DomesticLeagueSpec("TUR", "4339", "Turkish Super Lig"),
    DomesticLeagueSpec("NOR", "4358", "Norwegian Eliteserien", True),
    DomesticLeagueSpec("GRE", "4336", "Greek Super League 1"),
    DomesticLeagueSpec("AUT", "4621", "Austrian Bundesliga"),
    DomesticLeagueSpec("SCO", "4330", "Scottish Premier League"),
    DomesticLeagueSpec("POL", "4422", "Polish Ekstraklasa"),
    DomesticLeagueSpec("DEN", "4340", "Danish Superliga"),
    DomesticLeagueSpec("SUI", "4675", "Swiss Super League"),
    DomesticLeagueSpec("ISR", "4644", "Israeli Premier League"),
    DomesticLeagueSpec("CYP", "4630", "Cypriot First Division"),
    DomesticLeagueSpec("SWE", "4347", "Swedish Allsvenskan", True),
)


# These are the additional first-tier leagues required to give the frozen
# 2026/27 UCL/UEL participant snapshot full domestic-Poisson coverage.  The
# provider IDs were resolved from TheSportsDB's Soccer league catalogue on
# 2026-08-21.  Keep this set separate from PILOT_LEAGUES: it is a candidate
# data expansion, not an implicit change to the canonical production dataset.
UCL_UEL_EXPANSION_LEAGUES: tuple[DomesticLeagueSpec, ...] = (
    DomesticLeagueSpec("ALB", "4617", "Albanian Superliga"),
    DomesticLeagueSpec("ARM", "4619", "Armenian Premier League"),
    DomesticLeagueSpec("AZB", "4693", "Azerbaijani Premier League"),
    DomesticLeagueSpec("BUL", "4626", "Bulgarian First League"),
    DomesticLeagueSpec("CRO", "4629", "Croatian First Football League"),
    DomesticLeagueSpec("GEO", "4638", "Georgian Erovnuli Liga", True),
    DomesticLeagueSpec("HUN", "4690", "Hungarian NB I"),
    DomesticLeagueSpec("KAZ", "4649", "Kazakhstan Premier League", True),
    DomesticLeagueSpec("LIT", "4651", "Lithuanian TOPLYGA", True),
    DomesticLeagueSpec("ROM", "4691", "Romanian Liga I"),
    DomesticLeagueSpec("SCO", "4330", "Scottish Premier League"),
    DomesticLeagueSpec("SLO", "4692", "Slovenian 1. SNL"),
    DomesticLeagueSpec("SRB", "4671", "Serbian Super Liga"),
    DomesticLeagueSpec("SVK", "4672", "Slovak First Football League"),
    DomesticLeagueSpec("UKR", "4354", "Ukrainian Premier League"),
)

SCHEDULE_COLUMNS = (
    "match_id",
    "source_event_id",
    "sportsdb_league_id",
    "league_name",
    "country_code",
    "provider_season",
    "ao_season",
    "kickoff_utc",
    "home_source_team_id",
    "away_source_team_id",
    "home_team_name",
    "away_team_name",
    "home_goals",
    "away_goals",
    "actual_home_score",
    "status",
    "round",
)


def payload_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    for value in payload.values():
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    return []


def normalize_schedule(
    payload: Mapping[str, Any],
    spec: DomesticLeagueSpec,
    provider_season: str,
    *,
    reconciliation_audit: list[dict[str, object]] | None = None,
) -> pd.DataFrame:
    """Normalize one provider season without accepting incomplete match results."""

    rows: list[dict[str, object]] = []
    for event in payload_rows(payload):
        event_id = _text(event.get("idEvent"))
        home_id = _text(event.get("idHomeTeam"))
        away_id = _text(event.get("idAwayTeam"))
        league_id = _text(event.get("idLeague"))
        if not event_id or not home_id or not away_id:
            continue
        if league_id != spec.sportsdb_league_id:
            raise ValueError(
                f"{spec.country_code} {provider_season}: unexpected league ID {league_id}"
            )
        kickoff = pd.to_datetime(event.get("strTimestamp"), utc=True, errors="coerce")
        home_goals = _integer(event.get("intHomeScore"))
        away_goals = _integer(event.get("intAwayScore"))
        if pd.isna(kickoff) or home_goals is None or away_goals is None:
            continue
        if home_goals < 0 or away_goals < 0:
            raise ValueError(f"{event_id}: domestic goals cannot be negative")
        rows.append(
            {
                "match_id": f"TSD-{spec.sportsdb_league_id}-{event_id}",
                "source_event_id": event_id,
                "sportsdb_league_id": spec.sportsdb_league_id,
                "league_name": str(event.get("strLeague") or spec.league_name),
                "country_code": spec.country_code,
                "provider_season": provider_season,
                "ao_season": ao_season_for_provider_event(spec, provider_season, kickoff),
                "kickoff_utc": kickoff.isoformat(),
                "home_source_team_id": home_id,
                "away_source_team_id": away_id,
                "home_team_name": str(event.get("strHomeTeam") or "").strip(),
                "away_team_name": str(event.get("strAwayTeam") or "").strip(),
                "home_goals": int(home_goals),
                "away_goals": int(away_goals),
                "actual_home_score": (
                    1.0 if home_goals > away_goals else 0.5 if home_goals == away_goals else 0.0
                ),
                "status": str(event.get("strStatus") or ""),
                "round": _integer(event.get("intRound")),
            }
        )
    result = pd.DataFrame(rows, columns=SCHEDULE_COLUMNS)
    if result.empty:
        return result
    if result["source_event_id"].duplicated().any():
        raise ValueError(f"{spec.country_code} {provider_season}: duplicate source event ID")
    if result["match_id"].duplicated().any():
        raise ValueError(f"{spec.country_code} {provider_season}: duplicate match_id")
    if result["home_source_team_id"].eq(result["away_source_team_id"]).any():
        raise ValueError(f"{spec.country_code} {provider_season}: same home and away team")
    if reconciliation_audit is not None:
        result = reconcile_domestic_fixture_observations(result, reconciliation_audit)
    else:
        validate_domestic_fixture_uniqueness(result, "source schedule")
    return result.sort_values(["kickoff_utc", "match_id"], kind="stable").reset_index(drop=True)


def table_expected_matches(payload: Mapping[str, Any]) -> int | None:
    rows = payload_rows(payload)
    played: list[int] = []
    for row in rows:
        value = _integer(row.get("intPlayed"))
        if value is not None and value >= 0:
            played.append(value)
    if not played:
        return None
    total = sum(played)
    if total <= 0 or total % 2:
        return None
    return total // 2


def assess_league_season(
    matches: pd.DataFrame,
    *,
    spec: DomesticLeagueSpec,
    provider_season: str,
    expected_matches: int | None,
) -> dict[str, object]:
    """Return an auditable all-or-nothing quality decision for one league season."""

    match_count = int(len(matches))
    unique_events = int(matches["source_event_id"].nunique()) if match_count else 0
    unique_teams = int(
        pd.concat(
            [matches.get("home_source_team_id", pd.Series(dtype=str)), matches.get("away_source_team_id", pd.Series(dtype=str))],
            ignore_index=True,
        ).nunique()
    )
    timestamps_valid = bool(
        match_count > 0
        and pd.to_datetime(matches["kickoff_utc"], utc=True, errors="coerce").notna().all()
    )
    scores_valid = bool(
        match_count > 0
        and matches[["home_goals", "away_goals"]].notna().all().all()
        and matches[["home_goals", "away_goals"]].ge(0).all().all()
    )
    coverage = (
        float(match_count / expected_matches)
        if expected_matches is not None and expected_matches > 0
        else math.nan
    )
    accepted = bool(
        expected_matches is not None
        and coverage >= 0.95
        and match_count == unique_events
        and unique_teams >= 4
        and timestamps_valid
        and scores_valid
    )
    if expected_matches is None:
        reason = "FINAL_TABLE_UNAVAILABLE"
    elif coverage < 0.95:
        reason = "TABLE_COVERAGE_BELOW_95_PCT"
    elif match_count != unique_events:
        reason = "DUPLICATE_SOURCE_EVENT"
    elif unique_teams < 4:
        reason = "IMPLAUSIBLE_TEAM_UNIVERSE"
    elif not timestamps_valid:
        reason = "INVALID_KICKOFF_UTC"
    elif not scores_valid:
        reason = "INVALID_SCORE"
    else:
        reason = "ACCEPTED"
    return {
        "country_code": spec.country_code,
        "sportsdb_league_id": spec.sportsdb_league_id,
        "league_name": spec.league_name,
        "provider_season": provider_season,
        "schedule_matches": match_count,
        "table_expected_matches": expected_matches,
        "coverage_rate": coverage,
        "unique_events": unique_events,
        "unique_teams": unique_teams,
        "timestamps_valid": timestamps_valid,
        "scores_valid": scores_valid,
        "quality_status": "ACCEPTED" if accepted else "REJECTED",
        "quality_reason": reason,
    }


def build_domestic_team_bridge(
    matches: pd.DataFrame,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    """Map provider teams to existing AO clubs only when identity is decisive."""

    _require_columns(matches, {"country_code", "home_source_team_id", "away_source_team_id", "home_team_name", "away_team_name"}, "matches")
    _require_columns(registry, {"club_id", "canonical_name", "country_code"}, "registry")
    teams = pd.concat(
        [
            matches[["country_code", "home_source_team_id", "home_team_name"]].rename(
                columns={"home_source_team_id": "source_team_id", "home_team_name": "source_team_name"}
            ),
            matches[["country_code", "away_source_team_id", "away_team_name"]].rename(
                columns={"away_source_team_id": "source_team_id", "away_team_name": "source_team_name"}
            ),
        ],
        ignore_index=True,
    ).drop_duplicates(["country_code", "source_team_id"], keep="first")
    registry_values = _registry_name_index(registry)
    rows: list[dict[str, object]] = []
    for team in teams.itertuples(index=False):
        candidates = registry_values.get(str(team.country_code), [])
        source_name = normalize_name(str(team.source_team_name))
        verified_alias = VERIFIED_DOMESTIC_PROVIDER_ALIASES.get(
            (str(team.country_code), source_name)
        )
        if verified_alias is not None:
            alias_candidates = [
                value for value in candidates if value["club_id"] == verified_alias
            ]
            if len(alias_candidates) != 1:
                raise ValueError(
                    "Verified domestic provider alias is absent or ambiguous in registry: "
                    f"{team.country_code}/{team.source_team_name}/{verified_alias}"
                )
            rows.append(
                _bridge_record(
                    team,
                    verified_alias,
                    "VERIFIED_PROVIDER_ALIAS",
                    1.0,
                    1.0,
                )
            )
            continue
        exact = [value for value in candidates if source_name in value["names"]]
        if len(exact) == 1:
            selected = exact[0]
            rows.append(_bridge_record(team, selected["club_id"], "EXACT_NAME", 1.0, 1.0))
            continue
        if len(exact) > 1:
            rows.append(_bridge_record(team, pd.NA, "AMBIGUOUS_EXACT", 1.0, 1.0))
            continue
        scored = sorted(
            (
                (
                    max(SequenceMatcher(None, source_name, name).ratio() for name in value["names"]),
                    value["club_id"],
                )
                for value in candidates
            ),
            reverse=True,
        )
        if not scored or scored[0][0] < 0.92:
            rows.append(_bridge_record(team, pd.NA, "NO_AO_COUNTERPART", scored[0][0] if scored else 0.0, scored[1][0] if len(scored) > 1 else 0.0))
            continue
        best, club_id = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else 0.0
        if best - runner_up < 0.03:
            rows.append(_bridge_record(team, pd.NA, "AMBIGUOUS_FUZZY", best, runner_up))
            continue
        rows.append(_bridge_record(team, club_id, "HIGH_CONFIDENCE_FUZZY", best, runner_up))
    bridge = pd.DataFrame(rows).sort_values(["country_code", "source_team_id"], kind="stable")
    if bridge.duplicated(["country_code", "source_team_id"]).any():
        raise ValueError("Domestic bridge must be unique by country and provider team")
    return bridge.reset_index(drop=True)


def attach_domestic_club_ids(matches: pd.DataFrame, bridge: pd.DataFrame) -> pd.DataFrame:
    _require_columns(bridge, {"country_code", "source_team_id", "ao_club_id"}, "bridge")
    result = matches.copy()
    home = bridge.rename(
        columns={"source_team_id": "home_source_team_id", "ao_club_id": "home_ao_club_id"}
    )[["country_code", "home_source_team_id", "home_ao_club_id"]]
    away = bridge.rename(
        columns={"source_team_id": "away_source_team_id", "ao_club_id": "away_ao_club_id"}
    )[["country_code", "away_source_team_id", "away_ao_club_id"]]
    result = result.merge(home, on=["country_code", "home_source_team_id"], how="left", validate="many_to_one")
    result = result.merge(away, on=["country_code", "away_source_team_id"], how="left", validate="many_to_one")
    return result


def validate_domestic_dataset(matches: pd.DataFrame, quality: pd.DataFrame, bridge: pd.DataFrame) -> None:
    _require_columns(matches, set(SCHEDULE_COLUMNS) | {"home_ao_club_id", "away_ao_club_id"}, "matches")
    _require_columns(quality, {"country_code", "provider_season", "quality_status"}, "quality")
    _require_columns(bridge, {"country_code", "source_team_id", "identity_method"}, "bridge")
    if matches.empty:
        raise ValueError("Domestic match dataset cannot be empty")
    if matches["match_id"].duplicated().any() or matches["source_event_id"].duplicated().any():
        raise ValueError("Domestic matches require unique match and source IDs")
    if matches["home_source_team_id"].eq(matches["away_source_team_id"]).any():
        raise ValueError("Domestic matches cannot have identical teams")
    if pd.to_datetime(matches["kickoff_utc"], utc=True, errors="coerce").isna().any():
        raise ValueError("Domestic matches require valid UTC timestamps")
    if not matches[["home_goals", "away_goals"]].ge(0).all().all():
        raise ValueError("Domestic match goals cannot be negative")
    if not quality["quality_status"].eq("ACCEPTED").all():
        raise ValueError("Normalized domestic match output may only contain accepted league seasons")
    if bridge.duplicated(["country_code", "source_team_id"]).any():
        raise ValueError("Domestic bridge has duplicate provider team mappings")


def ao_season_from_kickoff(kickoff: pd.Timestamp) -> str:
    if kickoff.tzinfo is None:
        raise ValueError("kickoff must be timezone aware")
    year = int(kickoff.year)
    start = year if kickoff.month >= 7 else year - 1
    return f"{start}/{str(start + 1)[-2:]}"


def ao_season_for_provider_event(
    spec: DomesticLeagueSpec,
    provider_season: str,
    kickoff: pd.Timestamp,
) -> str:
    """Prefer the provider competition season over a simple July-to-June guess.

    This keeps COVID-delayed domestic matches in their actual 2019/20 competition
    season even when they were played in July 2020.
    """

    if spec.calendar_season:
        return ao_season_from_kickoff(kickoff)
    try:
        start_year = int(str(provider_season).split("-", 1)[0])
    except (TypeError, ValueError):
        return ao_season_from_kickoff(kickoff)
    return f"{start_year}/{str(start_year + 1)[-2:]}"


def _registry_name_index(registry: pd.DataFrame) -> dict[str, list[dict[str, object]]]:
    values: dict[str, list[dict[str, object]]] = {}
    for row in registry.itertuples(index=False):
        names = {normalize_name(str(row.canonical_name))}
        observed = getattr(row, "observed_names", "")
        names.update(normalize_name(value) for value in str(observed).split(" | ") if value)
        names.discard("")
        values.setdefault(str(row.country_code), []).append({"club_id": str(row.club_id), "names": names})
    return values


def _bridge_record(team: object, club_id: object, method: str, score: float, runner_up: float) -> dict[str, object]:
    return {
        "country_code": str(getattr(team, "country_code")),
        "source_team_id": str(getattr(team, "source_team_id")),
        "source_club_key": f"THESPORTSDB:{getattr(team, 'source_team_id')}",
        "source_team_name": str(getattr(team, "source_team_name")),
        "ao_club_id": club_id,
        "identity_method": method,
        "identity_score": float(score),
        "runner_up_score": float(runner_up),
        "identity_ambiguous": method.startswith("AMBIGUOUS"),
    }


def _text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _integer(value: object) -> int | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or not numeric.is_integer():
        return None
    return int(numeric)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")
