from __future__ import annotations

import math
import re
from difflib import SequenceMatcher
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ao_elo.xg_dataset import (
    EXPECTED_COMPETITION_ROWS,
    EXPECTED_MATCHES,
    normalize_name,
    resolve_fotmob_identity,
)


SPORTSDB_LEAGUES = {"UCL": "4480", "UEL": "4481", "UECL": "5071"}
SPORTSDB_SEASON = "2025-2026"
DETAIL_ENDPOINTS = {
    "event": "event",
    "stats": "event_stats",
    "timeline": "event_timeline",
    "lineup": "event_lineup",
}


def payload_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    for value in payload.values():
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    return []


def flatten_schedules(
    payloads: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for competition, expected_league_id in SPORTSDB_LEAGUES.items():
        if competition not in payloads:
            raise ValueError(f"Missing TheSportsDB schedule for {competition}")
        schedule = payload_rows(payloads[competition])
        expected_rows = EXPECTED_COMPETITION_ROWS[competition]
        if len(schedule) != expected_rows:
            raise ValueError(
                f"{competition} schedule expected {expected_rows} rows, found {len(schedule)}"
            )
        for event in schedule:
            event_id = id_text(event.get("idEvent"))
            home_id = id_text(event.get("idHomeTeam"))
            away_id = id_text(event.get("idAwayTeam"))
            league_id = id_text(event.get("idLeague"))
            if not event_id or not home_id or not away_id:
                raise ValueError(f"{competition} schedule contains missing event/team ID")
            if league_id != expected_league_id:
                raise ValueError(
                    f"{competition} schedule has unexpected league ID {league_id}"
                )
            rows.append(
                {
                    "source_match_id": event_id,
                    "competition": competition,
                    "league_name": str(event.get("strLeague") or ""),
                    "kickoff_utc": str(event.get("strTimestamp") or ""),
                    "home_team_id": home_id,
                    "away_team_id": away_id,
                    "home_team_name": str(event.get("strHomeTeam") or ""),
                    "away_team_name": str(event.get("strAwayTeam") or ""),
                    "home_goals": integer(event.get("intHomeScore")),
                    "away_goals": integer(event.get("intAwayScore")),
                    "status": str(event.get("strStatus") or ""),
                    "round": integer(event.get("intRound")),
                    "schedule_record": event,
                }
            )
    result = pd.DataFrame(rows)
    if len(result) != EXPECTED_MATCHES:
        raise ValueError(
            f"TheSportsDB schedules expected {EXPECTED_MATCHES} rows, found {len(result)}"
        )
    if result["source_match_id"].duplicated().any():
        raise ValueError("TheSportsDB schedules contain duplicate event IDs")
    result["kickoff_parsed"] = pd.to_datetime(
        result["kickoff_utc"], utc=True, errors="coerce"
    )
    if result["kickoff_parsed"].isna().any():
        raise ValueError("TheSportsDB schedules contain invalid UTC timestamps")
    return result.sort_values(
        ["kickoff_parsed", "source_match_id"], kind="stable"
    ).reset_index(drop=True)


def resolve_match_identities(
    events: pd.DataFrame,
    schedules: pd.DataFrame,
) -> pd.DataFrame:
    if len(events) != EXPECTED_MATCHES or events["match_id"].duplicated().any():
        raise ValueError("AO 2025/26 events must contain 961 unique matches")
    rows: list[dict[str, Any]] = []
    for _, event in events.iterrows():
        candidate, primary_audit = resolve_fotmob_identity(
            event,
            schedules,
            min_side_similarity=0.60,
            min_pair_similarity=0.72,
            min_runner_up_gap=0.02,
        )
        method = "DATE_SCORE_TEAM"
        if candidate is None:
            fallback = fallback_identity(event, schedules)
            if fallback is None:
                rows.append(
                    {
                        "match_id": event["match_id"],
                        "source_event_id": pd.NA,
                        "identity_status": "UNRESOLVED",
                        "identity_method": "NONE",
                        "home_name_similarity": primary_audit.get(
                            "best_home_similarity"
                        ),
                        "away_name_similarity": primary_audit.get(
                            "best_away_similarity"
                        ),
                        "pair_name_similarity": primary_audit.get(
                            "best_pair_similarity"
                        ),
                    }
                )
                continue
            source = fallback
            method = "DATE_TEAM_FALLBACK"
            source_id = str(source["source_match_id"])
            home_similarity = float(source["home_similarity"])
            away_similarity = float(source["away_similarity"])
            pair_similarity = float(source["pair_similarity"])
        else:
            source_id = candidate.source_match_id
            home_similarity = candidate.home_similarity
            away_similarity = candidate.away_similarity
            pair_similarity = candidate.pair_similarity
            source = schedules.loc[
                schedules["source_match_id"].eq(source_id)
            ].iloc[0]
        score_agrees = bool(
            int(event["home_goals"]) == int(source["home_goals"])
            and int(event["away_goals"]) == int(source["away_goals"])
        )
        kickoff_delta_seconds = abs(
            (pd.Timestamp(event["kickoff_utc"]) - source["kickoff_parsed"])
            .total_seconds()
        )
        rows.append(
            {
                "match_id": event["match_id"],
                "source_event_id": source_id,
                "identity_status": "MATCHED",
                "identity_method": method,
                "home_name_similarity": home_similarity,
                "away_name_similarity": away_similarity,
                "pair_name_similarity": pair_similarity,
                "score_agrees": score_agrees,
                "kickoff_delta_seconds": kickoff_delta_seconds,
                "sportsdb_home_goals": int(source["home_goals"]),
                "sportsdb_away_goals": int(source["away_goals"]),
                "sportsdb_status": source["status"],
                "sportsdb_round": int(source["round"]),
                "sportsdb_home_team_id": source["home_team_id"],
                "sportsdb_away_team_id": source["away_team_id"],
                "sportsdb_home_team_name": source["home_team_name"],
                "sportsdb_away_team_name": source["away_team_name"],
            }
        )
    result = pd.DataFrame(rows)
    if result["source_event_id"].isna().any():
        sample = result.loc[result["source_event_id"].isna(), "match_id"].head(10)
        raise ValueError(
            f"Unresolved TheSportsDB match identities: {sample.tolist()}"
        )
    if result["source_event_id"].duplicated().any():
        raise ValueError("One TheSportsDB event mapped to multiple AO matches")
    return result


def fallback_identity(
    event: pd.Series,
    schedules: pd.DataFrame,
) -> dict[str, Any] | None:
    kickoff = pd.Timestamp(event["kickoff_utc"])
    candidates = schedules.loc[
        schedules["competition"].eq(str(event["competition"]))
        & (schedules["kickoff_parsed"] - kickoff).abs().le(pd.Timedelta(hours=36))
    ]
    scored: list[dict[str, Any]] = []
    home_key = provider_name(event["home_team_name"])
    away_key = provider_name(event["away_team_name"])
    for _, row in candidates.iterrows():
        home_similarity = SequenceMatcher(
            None, home_key, provider_name(row["home_team_name"])
        ).ratio()
        away_similarity = SequenceMatcher(
            None, away_key, provider_name(row["away_team_name"])
        ).ratio()
        scored.append(
            {
                **row.to_dict(),
                "home_similarity": home_similarity,
                "away_similarity": away_similarity,
                "pair_similarity": (home_similarity + away_similarity) / 2.0,
            }
        )
    scored.sort(key=lambda row: row["pair_similarity"], reverse=True)
    if not scored:
        return None
    best = scored[0]
    runner_up = scored[1]["pair_similarity"] if len(scored) > 1 else 0.0
    if (
        best["pair_similarity"] < 0.82
        or min(best["home_similarity"], best["away_similarity"]) < 0.70
        or best["pair_similarity"] - runner_up < 0.04
    ):
        return None
    return best


def provider_name(value: object) -> str:
    base = normalize_name(value)
    aliases = {
        "olympique lyon": "lyon",
        "paok thessaloniki": "paok",
        "brann bergen": "brann",
        "lille osc": "lille",
        "red star belgrade": "crvena zvezda",
        "red bull salzburg": "salzburg",
        "hapoel beer sheva": "hapoel be er sheva",
    }
    return aliases.get(base, base)


def build_event_table(
    events: pd.DataFrame,
    identities: pd.DataFrame,
    schedules: pd.DataFrame,
    details: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    bridge = events.merge(identities, on="match_id", validate="one_to_one")
    schedule_map = schedules.set_index("source_match_id")
    rows: list[dict[str, Any]] = []
    for row in bridge.itertuples(index=False):
        source_id = str(row.source_event_id)
        schedule_record = schedule_map.loc[source_id, "schedule_record"]
        detail_records = payload_rows(details[source_id])
        detail = detail_records[0] if detail_records else {}
        raw = {**schedule_record, **detail}
        output = {
            "match_id": row.match_id,
            "club_id_home": f"AO-UEFA-{int(row.uefa_home_team_id)}",
            "club_id_away": f"AO-UEFA-{int(row.uefa_away_team_id)}",
            "competition": row.competition,
            "ao_kickoff_utc": row.kickoff_utc,
            "ao_home_team_name": row.home_team_name,
            "ao_away_team_name": row.away_team_name,
            "ao_home_goals": row.home_goals,
            "ao_away_goals": row.away_goals,
            "ao_decided_on_penalties": row.decided_on_penalties,
            "ao_home_penalty_goals": row.home_penalty_goals,
            "ao_away_penalty_goals": row.away_penalty_goals,
            "identity_method": row.identity_method,
            "identity_score_agrees": row.score_agrees,
        }
        output.update({f"sportsdb_{key}": value for key, value in raw.items()})
        rows.append(output)
    result = pd.DataFrame(rows).sort_values(
        ["ao_kickoff_utc", "match_id"], kind="stable"
    )
    if len(result) != EXPECTED_MATCHES or result["match_id"].duplicated().any():
        raise ValueError("TheSportsDB event table must contain 961 unique matches")
    return result.reset_index(drop=True)


def flatten_child_records(
    identities: pd.DataFrame,
    payloads: Mapping[str, Mapping[str, Any]],
    *,
    endpoint: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    identity_map = identities.set_index("source_event_id")
    for source_id, payload in payloads.items():
        if source_id not in identity_map.index:
            raise ValueError(f"Unknown source event ID in {endpoint}: {source_id}")
        identity = identity_map.loc[source_id]
        for index, record in enumerate(payload_rows(payload), start=1):
            rows.append(
                {
                    "match_id": identity["match_id"],
                    "competition": identity.get("competition", pd.NA),
                    "source_event_id": source_id,
                    "record_order": index,
                    **record,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["match_id", "competition", "source_event_id", "record_order"]
        )
    return pd.DataFrame(rows).sort_values(
        ["match_id", "record_order"], kind="stable"
    ).reset_index(drop=True)


def build_stats_wide(
    events: pd.DataFrame,
    stats: pd.DataFrame,
) -> pd.DataFrame:
    index = events[
        ["match_id", "competition", "sportsdb_idEvent"]
    ].rename(columns={"sportsdb_idEvent": "source_event_id"})
    rows: list[dict[str, Any]] = []
    for match_id, group in stats.groupby("match_id", sort=False):
        row: dict[str, Any] = {"match_id": match_id}
        for stat in group.itertuples(index=False):
            slug = stat_slug(stat.strStat)
            row[f"home_{slug}"] = numeric_or_text(stat.intHome)
            row[f"away_{slug}"] = numeric_or_text(stat.intAway)
        rows.append(row)
    wide = index.merge(pd.DataFrame(rows), on="match_id", how="left", validate="one_to_one")
    wide["stats_covered"] = wide.filter(regex=r"^home_").notna().any(axis=1)
    home_xg = pd.to_numeric(
        wide.get("home_expected_goals", pd.Series(index=wide.index, dtype=float)),
        errors="coerce",
    )
    away_xg = pd.to_numeric(
        wide.get("away_expected_goals", pd.Series(index=wide.index, dtype=float)),
        errors="coerce",
    )
    home_shots = pd.to_numeric(
        wide.get("home_total_shots", pd.Series(index=wide.index, dtype=float)),
        errors="coerce",
    )
    away_shots = pd.to_numeric(
        wide.get("away_total_shots", pd.Series(index=wide.index, dtype=float)),
        errors="coerce",
    )
    raw_present = home_xg.notna() & away_xg.notna()
    zero_with_shots = raw_present & (
        (home_xg.eq(0.0) & home_shots.gt(0.0))
        | (away_xg.eq(0.0) & away_shots.gt(0.0))
    )
    both_zero = raw_present & home_xg.eq(0.0) & away_xg.eq(0.0)
    placeholder = zero_with_shots | both_zero

    wide["xg_raw_present"] = raw_present
    # Backward-compatible source coverage flag. Model eligibility is stricter.
    wide["xg_covered"] = raw_present
    wide["xg_zero_inconsistent_with_shots"] = zero_with_shots
    wide["xg_both_zero"] = both_zero
    wide["xg_placeholder_suspected"] = placeholder
    wide["xg_analysis_eligible"] = raw_present & ~placeholder
    wide["xg_quality_status"] = np.select(
        [~raw_present, zero_with_shots, both_zero],
        [
            "NO_RAW_XG",
            "ZERO_INCONSISTENT_WITH_SHOTS",
            "BOTH_ZERO_PLACEHOLDER",
        ],
        default="ELIGIBLE",
    )
    return wide


def build_coverage_summary(
    events: pd.DataFrame,
    stats_wide: pd.DataFrame,
    timeline: pd.DataFrame,
    lineup: pd.DataFrame,
) -> pd.DataFrame:
    frame = events[
        ["match_id", "competition", "sportsdb_intRound", "sportsdb_idEvent"]
    ].rename(
        columns={
            "sportsdb_intRound": "round",
            "sportsdb_idEvent": "source_event_id",
        }
    )
    frame = frame.merge(
        stats_wide[
            [
                "match_id",
                "stats_covered",
                "xg_covered",
                "xg_placeholder_suspected",
                "xg_analysis_eligible",
            ]
        ],
        on="match_id",
        validate="one_to_one",
    )
    timeline_counts = timeline.groupby("match_id").size()
    lineup_counts = lineup.groupby("match_id").size()
    frame["timeline_records"] = frame["match_id"].map(timeline_counts).fillna(0).astype(int)
    frame["lineup_records"] = frame["match_id"].map(lineup_counts).fillna(0).astype(int)
    rows = []
    groups = [("ALL", "ALL", frame)]
    groups.extend(
        (competition, "ALL", group)
        for competition, group in frame.groupby("competition", sort=True)
    )
    groups.extend(
        (competition, str(round_value), group)
        for (competition, round_value), group in frame.groupby(
            ["competition", "round"], sort=True
        )
    )
    for competition, round_value, group in groups:
        matches = len(group)
        rows.append(
            {
                "competition": competition,
                "round": round_value,
                "matches": matches,
                "stats_covered": int(group["stats_covered"].sum()),
                "stats_coverage_rate": float(group["stats_covered"].mean()),
                "xg_covered": int(group["xg_covered"].sum()),
                "xg_coverage_rate": float(group["xg_covered"].mean()),
                "xg_placeholder_suspected": int(
                    group["xg_placeholder_suspected"].sum()
                ),
                "xg_analysis_eligible": int(group["xg_analysis_eligible"].sum()),
                "xg_analysis_eligible_rate": float(
                    group["xg_analysis_eligible"].mean()
                ),
                "timeline_covered": int(group["timeline_records"].gt(0).sum()),
                "timeline_coverage_rate": float(group["timeline_records"].gt(0).mean()),
                "lineup_covered": int(group["lineup_records"].gt(0).sum()),
                "lineup_coverage_rate": float(group["lineup_records"].gt(0).mean()),
            }
        )
    return pd.DataFrame(rows)


def build_field_catalog(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for table_name, frame in tables.items():
        for column in frame.columns:
            values = frame[column]
            non_null = int(values.notna().sum())
            rows.append(
                {
                    "table": table_name,
                    "column": column,
                    "dtype": str(values.dtype),
                    "rows": len(frame),
                    "non_null": non_null,
                    "null_rate": 0.0 if len(frame) == 0 else 1.0 - non_null / len(frame),
                    "distinct_non_null": int(values.dropna().astype(str).nunique()),
                }
            )
    return pd.DataFrame(rows)


def validate_normalized_dataset(
    events: pd.DataFrame,
    identities: pd.DataFrame,
    stats: pd.DataFrame,
    timeline: pd.DataFrame,
    lineup: pd.DataFrame,
    stats_wide: pd.DataFrame,
) -> None:
    if len(events) != EXPECTED_MATCHES or events["match_id"].duplicated().any():
        raise ValueError("events.csv grain must be one row per 961 AO matches")
    if len(identities) != EXPECTED_MATCHES:
        raise ValueError("identity audit must contain 961 rows")
    valid_match_ids = set(events["match_id"])
    for name, frame in (("stats", stats), ("timeline", timeline), ("lineup", lineup)):
        if not set(frame["match_id"]).issubset(valid_match_ids):
            raise ValueError(f"{name} contains orphan match IDs")
    if len(stats_wide) != EXPECTED_MATCHES or stats_wide["match_id"].duplicated().any():
        raise ValueError("event_stats_wide must contain 961 unique matches")
    for column in ("home_expected_goals", "away_expected_goals"):
        if column in stats_wide:
            numeric = pd.to_numeric(stats_wide[column], errors="coerce")
            if numeric.dropna().lt(0.0).any() or not np.isfinite(numeric.dropna()).all():
                raise ValueError(f"{column} must be finite and non-negative")
    required_xg_flags = {
        "xg_raw_present",
        "xg_covered",
        "xg_placeholder_suspected",
        "xg_analysis_eligible",
        "xg_quality_status",
    }
    missing_flags = required_xg_flags.difference(stats_wide.columns)
    if missing_flags:
        raise ValueError(f"event_stats_wide missing xG quality fields: {missing_flags}")
    if not stats_wide["xg_covered"].equals(stats_wide["xg_raw_present"]):
        raise ValueError("xg_covered must preserve raw provider-field coverage")
    if (
        stats_wide["xg_analysis_eligible"]
        & (
            ~stats_wide["xg_raw_present"]
            | stats_wide["xg_placeholder_suspected"]
        )
    ).any():
        raise ValueError("analysis-eligible xG cannot be missing or placeholder-like")


def build_xg_source_comparison(
    stats_wide: pd.DataFrame,
    fotmob: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "match_id",
        "competition",
        "home_expected_goals",
        "away_expected_goals",
        "xg_raw_present",
        "xg_analysis_eligible",
        "xg_quality_status",
    }
    missing = required.difference(stats_wide.columns)
    if missing:
        raise ValueError(f"TheSportsDB xG comparison fields missing: {missing}")
    fotmob_required = {
        "match_id",
        "xg_home",
        "xg_away",
        "xg_covered",
        "xg_analysis_eligible",
    }
    fotmob_missing = fotmob_required.difference(fotmob.columns)
    if fotmob_missing:
        raise ValueError(f"FotMob xG comparison fields missing: {fotmob_missing}")

    source = stats_wide[
        [
            "match_id",
            "competition",
            "home_expected_goals",
            "away_expected_goals",
            "xg_raw_present",
            "xg_analysis_eligible",
            "xg_quality_status",
        ]
    ].rename(
        columns={
            "home_expected_goals": "thesportsdb_xg_home",
            "away_expected_goals": "thesportsdb_xg_away",
            "xg_raw_present": "thesportsdb_xg_raw_present",
            "xg_analysis_eligible": "thesportsdb_xg_analysis_eligible",
            "xg_quality_status": "thesportsdb_xg_quality_status",
        }
    )
    reference_columns = [
        "match_id",
        "xg_home",
        "xg_away",
        "xg_covered",
        "xg_analysis_eligible",
    ]
    for optional in ("home_team_name", "away_team_name"):
        if optional in fotmob.columns:
            reference_columns.append(optional)
    reference = fotmob[reference_columns].rename(
        columns={
            "xg_home": "fotmob_xg_home",
            "xg_away": "fotmob_xg_away",
            "xg_covered": "fotmob_xg_covered",
            "xg_analysis_eligible": "fotmob_xg_analysis_eligible",
        }
    )
    comparison = source.merge(reference, on="match_id", how="left", validate="one_to_one")
    comparison["common_analysis_eligible"] = (
        comparison["thesportsdb_xg_analysis_eligible"].astype(bool)
        & comparison["fotmob_xg_analysis_eligible"].fillna(False).astype(bool)
    )
    for side in ("home", "away"):
        comparison[f"xg_{side}_difference"] = (
            pd.to_numeric(comparison[f"thesportsdb_xg_{side}"], errors="coerce")
            - pd.to_numeric(comparison[f"fotmob_xg_{side}"], errors="coerce")
        ).where(comparison["common_analysis_eligible"])
    comparison["thesportsdb_xg_total"] = (
        pd.to_numeric(comparison["thesportsdb_xg_home"], errors="coerce")
        + pd.to_numeric(comparison["thesportsdb_xg_away"], errors="coerce")
    )
    comparison["fotmob_xg_total"] = (
        pd.to_numeric(comparison["fotmob_xg_home"], errors="coerce")
        + pd.to_numeric(comparison["fotmob_xg_away"], errors="coerce")
    )
    comparison["xg_total_difference"] = (
        comparison["thesportsdb_xg_total"] - comparison["fotmob_xg_total"]
    ).where(comparison["common_analysis_eligible"])

    groups = [("ALL", comparison)]
    groups.extend(
        (competition, group)
        for competition, group in comparison.groupby("competition", sort=True)
    )
    summaries = []
    for competition, group in groups:
        common = group.loc[group["common_analysis_eligible"]].copy()
        summaries.append(
            {
                "competition": competition,
                "matches": len(group),
                "thesportsdb_raw_xg": int(group["thesportsdb_xg_raw_present"].sum()),
                "thesportsdb_analysis_eligible": int(
                    group["thesportsdb_xg_analysis_eligible"].sum()
                ),
                "fotmob_analysis_eligible": int(
                    group["fotmob_xg_analysis_eligible"].fillna(False).sum()
                ),
                "common_analysis_eligible": len(common),
                "home_pearson": correlation(
                    common["thesportsdb_xg_home"], common["fotmob_xg_home"], "pearson"
                ),
                "home_spearman": correlation(
                    common["thesportsdb_xg_home"], common["fotmob_xg_home"], "spearman"
                ),
                "away_pearson": correlation(
                    common["thesportsdb_xg_away"], common["fotmob_xg_away"], "pearson"
                ),
                "away_spearman": correlation(
                    common["thesportsdb_xg_away"], common["fotmob_xg_away"], "spearman"
                ),
                "home_mae": mean_absolute(common["xg_home_difference"]),
                "away_mae": mean_absolute(common["xg_away_difference"]),
                "total_mae": mean_absolute(common["xg_total_difference"]),
                "home_bias": finite_mean(common["xg_home_difference"]),
                "away_bias": finite_mean(common["xg_away_difference"]),
                "total_bias": finite_mean(common["xg_total_difference"]),
            }
        )
    return comparison, pd.DataFrame(summaries)


def correlation(left: pd.Series, right: pd.Series, method: str) -> float:
    frame = pd.DataFrame({"left": left, "right": right}).apply(
        pd.to_numeric, errors="coerce"
    ).dropna()
    if len(frame) < 2 or frame["left"].nunique() < 2 or frame["right"].nunique() < 2:
        return math.nan
    return float(frame["left"].corr(frame["right"], method=method))


def finite_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return math.nan if numeric.empty else float(numeric.mean())


def mean_absolute(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return math.nan if numeric.empty else float(numeric.abs().mean())


def stat_slug(value: object) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().casefold()).strip("_")
    if not text:
        raise ValueError("Statistic name cannot be empty")
    return text


def numeric_or_text(value: object) -> object:
    if value is None or str(value).strip() == "":
        return pd.NA
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    return int(number) if number.is_integer() else number


def integer(value: object) -> int:
    if value is None or isinstance(value, bool):
        raise ValueError("Expected integer value")
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"Expected integer value, received {value!r}")
    return int(number)


def id_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text
