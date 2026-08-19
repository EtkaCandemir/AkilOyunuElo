from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Sequence, Any, Iterable

import pandas as pd


COMPETITIONS = ("UCL", "UEL", "UECL")
EXPECTED_COMPETITION_ROWS = {"UCL": 281, "UEL": 271, "UECL": 409}
EXPECTED_MATCHES = sum(EXPECTED_COMPETITION_ROWS.values())
EXPECTED_TEAMS = 236
MISSING_REASONS = {
    "NO_PUBLIC_XG",
    "SOURCE_PAGE_UNAVAILABLE",
    "ACCESS_BLOCKED",
    "IDENTITY_AMBIGUOUS",
    "SCORE_MISMATCH",
    "TIME_SCOPE_UNKNOWN",
    "EXTRA_TIME_SCOPE_MISMATCH",
    "PENALTY_SCOPE_UNVERIFIED",
}
PRIMARY_XG_COLUMNS = (
    "xg_home",
    "xg_away",
    "xg_total",
    "xg_difference",
    "xg_home_share",
    "home_goals_minus_xg",
    "away_goals_minus_xg",
    "xg_result",
)
MASTER_COLUMNS = (
    "match_id",
    "season",
    "competition",
    "round",
    "round_sequence",
    "matchday",
    "tie_id",
    "leg_number",
    "is_tie_decider",
    "is_knockout",
    "is_neutral",
    "kickoff_date",
    "kickoff_utc",
    "event_order",
    "home_team_id",
    "away_team_id",
    "home_team_name",
    "away_team_name",
    "uefa_match_id",
    "uefa_home_team_id",
    "uefa_away_team_id",
    "home_goals",
    "away_goals",
    "actual_home_score",
    "goal_difference",
    "result_basis",
    "decided_on_penalties",
    "home_penalty_goals",
    "away_penalty_goals",
    "advanced_team_id",
    "xg_home",
    "xg_away",
    "xg_total",
    "xg_difference",
    "xg_home_share",
    "home_goals_minus_xg",
    "away_goals_minus_xg",
    "xg_result",
    "xg_covered",
    "xg_analysis_eligible",
    "xg_home_raw",
    "xg_away_raw",
    "xg_provider",
    "xg_type",
    "xg_duration_scope",
    "xg_in_match_penalties_included",
    "xg_extra_time_included",
    "xg_shootout_excluded",
    "xg_scope_status",
    "xg_missing_reason",
    "xg_source_match_id",
    "xg_source_url",
    "xg_snapshot_utc",
    "identity_match_method",
    "identity_confidence",
    "score_verified",
    "chronology_verified",
    "secondary_xg_home",
    "secondary_xg_away",
    "secondary_xg_type",
    "secondary_xg_provider",
    "secondary_xg_covered",
)


@dataclass(frozen=True)
class FotMobCandidate:
    source_match_id: str
    league_name: str
    kickoff_utc: str
    home_team_id: str
    away_team_id: str
    home_team_name: str
    away_team_name: str
    home_goals: int
    away_goals: int
    home_similarity: float
    away_similarity: float
    pair_similarity: float


def normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.casefold().replace("&", " and ")
    text = re.sub(r"\b(fc|cf|fk|sc|ac|afc|sk|nk|ks|sv|ss|club)\b", " ", text)
    normalized = re.sub(r"[^a-z0-9]+", " ", text).strip()
    aliases = {
        "aik stockholm": "aik",
        "eto gyor": "gyor",
        "gyori eto": "gyor",
        "fci levadia": "levadia",
        "kups kuopio": "kups",
        "racing genk": "genk",
        "red star belgrade": "crvena zvezda",
        "rfs riga": "rfs",
        "shkendija tetovo": "shkendija",
        "the new saints": "tns",
    }
    return aliases.get(normalized, normalized)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_dump(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_season_events(
    path: Path,
    seasons: Sequence[str] = ("2025/26",),
) -> pd.DataFrame:
    """Return the exact-date event rows for the requested seasons.

    The default keeps the original single-season behaviour so existing callers
    and the frozen 2025/26 dataset are unaffected.
    """

    events = pd.read_csv(path)
    missing = sorted(set(MASTER_COLUMNS[:30]) - set(events.columns))
    if missing:
        raise ValueError(f"events missing columns: {missing}")
    wanted = tuple(str(season) for season in seasons)
    if not wanted:
        raise ValueError("seasons cannot be empty")
    events = events.loc[events["season"].astype(str).isin(wanted)].copy()
    if events.empty:
        raise ValueError(f"events contains no rows for seasons: {list(wanted)}")
    if events["match_id"].isna().any() or events["match_id"].duplicated().any():
        raise ValueError("events.match_id must be non-null and unique")
    events["kickoff_utc"] = pd.to_datetime(
        events["kickoff_utc"], utc=True, errors="coerce"
    )
    if events["kickoff_utc"].isna().any():
        raise ValueError("events.kickoff_utc contains invalid values")
    return events.sort_values(["kickoff_utc", "match_id"], kind="stable").reset_index(
        drop=True
    )


def fotmob_competition(league_name: object) -> str | None:
    value = normalize_name(league_name)
    if "women" in value or "youth" in value:
        return None
    if "conference league" in value:
        return "UECL"
    if "europa league" in value:
        return "UEL"
    if "champions league" in value:
        return "UCL"
    return None


def flatten_fotmob_date_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for league in payload.get("leagues") or []:
        competition = fotmob_competition(league.get("name"))
        if competition is None:
            continue
        for match in league.get("matches") or []:
            home = match.get("home") or {}
            away = match.get("away") or {}
            status = match.get("status") or {}
            if home.get("score") is None or away.get("score") is None:
                continue
            rows.append(
                {
                    "source_match_id": str(match.get("id") or ""),
                    "competition": competition,
                    "league_name": str(league.get("name") or ""),
                    "kickoff_utc": str(status.get("utcTime") or ""),
                    "home_team_id": str(home.get("id") or ""),
                    "away_team_id": str(away.get("id") or ""),
                    "home_team_name": str(home.get("longName") or home.get("name") or ""),
                    "away_team_name": str(away.get("longName") or away.get("name") or ""),
                    "home_goals": int(home["score"]),
                    "away_goals": int(away["score"]),
                }
            )
    return rows


def resolve_fotmob_identity(
    event: pd.Series,
    source_rows: pd.DataFrame,
    *,
    min_side_similarity: float = 0.70,
    min_pair_similarity: float = 0.80,
    min_runner_up_gap: float = 0.03,
) -> tuple[FotMobCandidate | None, dict[str, object]]:
    kickoff = pd.Timestamp(event["kickoff_utc"])
    candidates = source_rows.loc[
        source_rows["competition"].eq(str(event["competition"]))
        & source_rows["home_goals"].eq(int(event["home_goals"]))
        & source_rows["away_goals"].eq(int(event["away_goals"]))
    ].copy()
    candidates["source_kickoff"] = pd.to_datetime(
        candidates["kickoff_utc"], utc=True, errors="coerce"
    )
    candidates = candidates.loc[
        (candidates["source_kickoff"] - kickoff).abs().le(pd.Timedelta(hours=36))
    ]
    scored: list[FotMobCandidate] = []
    home_key = normalize_name(event["home_team_name"])
    away_key = normalize_name(event["away_team_name"])
    for row in candidates.itertuples(index=False):
        home_similarity = SequenceMatcher(
            None, home_key, normalize_name(row.home_team_name)
        ).ratio()
        away_similarity = SequenceMatcher(
            None, away_key, normalize_name(row.away_team_name)
        ).ratio()
        pair_similarity = (home_similarity + away_similarity) / 2.0
        scored.append(
            FotMobCandidate(
                source_match_id=str(row.source_match_id),
                league_name=str(row.league_name),
                kickoff_utc=str(row.kickoff_utc),
                home_team_id=str(row.home_team_id),
                away_team_id=str(row.away_team_id),
                home_team_name=str(row.home_team_name),
                away_team_name=str(row.away_team_name),
                home_goals=int(row.home_goals),
                away_goals=int(row.away_goals),
                home_similarity=float(home_similarity),
                away_similarity=float(away_similarity),
                pair_similarity=float(pair_similarity),
            )
        )
    scored.sort(key=lambda item: (-item.pair_similarity, item.source_match_id))
    best = scored[0] if scored else None
    runner_up = scored[1].pair_similarity if len(scored) > 1 else 0.0
    high_confidence_name = bool(
        best
        and best.home_similarity >= min_side_similarity
        and best.away_similarity >= min_side_similarity
        and best.pair_similarity >= min_pair_similarity
        and best.pair_similarity - runner_up >= min_runner_up_gap
    )
    unique_exact_fixture = bool(
        best
        and len(scored) == 1
        and best.home_similarity >= 0.25
        and best.away_similarity >= 0.25
        and best.pair_similarity >= 0.45
    )
    separated_alias_match = bool(
        best
        and len(scored) > 1
        and best.home_similarity >= 0.35
        and best.away_similarity >= 0.35
        and best.pair_similarity >= 0.55
        and best.pair_similarity - runner_up >= 0.12
    )
    accepted = high_confidence_name or unique_exact_fixture or separated_alias_match
    audit = {
        "fotmob_candidate_count": len(scored),
        "fotmob_source_match_id": best.source_match_id if best else "",
        "fotmob_home_team_id": best.home_team_id if best else "",
        "fotmob_away_team_id": best.away_team_id if best else "",
        "fotmob_home_team_name": best.home_team_name if best else "",
        "fotmob_away_team_name": best.away_team_name if best else "",
        "home_similarity": best.home_similarity if best else math.nan,
        "away_similarity": best.away_similarity if best else math.nan,
        "pair_similarity": best.pair_similarity if best else math.nan,
        "runner_up_similarity": runner_up if scored else math.nan,
        "identity_accepted": accepted,
        "identity_resolution": (
            "utc_time_competition_score_high_confidence_name"
            if high_confidence_name
            else "unique_utc_time_competition_score_alias"
            if unique_exact_fixture
            else "separated_utc_time_competition_score_alias"
            if separated_alias_match
            else "no_time_competition_score_candidate"
            if not scored
            else "identity_below_threshold_or_ambiguous"
        ),
    }
    return (best if accepted else None), audit


def _find_stat(period: dict[str, Any], key: str) -> tuple[float, float] | None:
    for group in period.get("stats") or []:
        for stat in group.get("stats") or []:
            if stat.get("key") != key:
                continue
            values = stat.get("stats") or []
            if len(values) != 2 or values[0] is None or values[1] is None:
                continue
            try:
                result = (float(values[0]), float(values[1]))
            except (TypeError, ValueError):
                continue
            if all(math.isfinite(item) and item >= 0.0 for item in result):
                return result
    return None


def _has_extra_time(payload: dict[str, Any]) -> bool:
    halfs = ((payload.get("header") or {}).get("status") or {}).get("halfs") or {}
    if halfs.get("firstExtraHalfStarted") or halfs.get("secondExtraHalfStarted"):
        return True
    periods = (((payload.get("content") or {}).get("stats") or {}).get("Periods") or {})
    return any(name in periods for name in ("FirstExtraHalf", "SecondExtraHalf"))


def _shotmap_scope_check(
    payload: dict[str, Any], xg: tuple[float, float]
) -> tuple[bool, bool]:
    general = payload.get("general") or {}
    home_id = str((general.get("homeTeam") or {}).get("id") or "")
    away_id = str((general.get("awayTeam") or {}).get("id") or "")
    shots = (((payload.get("content") or {}).get("shotmap") or {}).get("shots") or [])
    if not shots:
        return False, False
    sums = {home_id: 0.0, away_id: 0.0}
    shootout_present = False
    for shot in shots:
        if str(shot.get("period") or "") == "PenaltyShootout":
            shootout_present = True
            continue
        team_id = str(shot.get("teamId") or "")
        value = shot.get("expectedGoals")
        if team_id in sums and value is not None:
            sums[team_id] += float(value)
    # FotMob's displayed two-decimal aggregate can lag individual shot-model
    # revisions slightly; a 0.10 audit tolerance remains far below shootout xG.
    reconciled = abs(sums[home_id] - xg[0]) <= 0.10 and abs(sums[away_id] - xg[1]) <= 0.10
    return shootout_present, reconciled


def parse_fotmob_xg(
    event: pd.Series,
    candidate: FotMobCandidate,
    payload: dict[str, Any] | None,
    *,
    fetch_status: str,
    snapshot_utc: str,
) -> tuple[dict[str, object], dict[str, object]]:
    source_url = f"https://www.fotmob.com/match/{candidate.source_match_id}"
    base: dict[str, object] = {
        **{column: math.nan for column in PRIMARY_XG_COLUMNS},
        "xg_covered": False,
        "xg_analysis_eligible": False,
        "xg_home_raw": math.nan,
        "xg_away_raw": math.nan,
        "xg_provider": "FotMob",
        "xg_type": "shot_based_expected_goals",
        "xg_duration_scope": "",
        "xg_in_match_penalties_included": pd.NA,
        "xg_extra_time_included": pd.NA,
        "xg_shootout_excluded": pd.NA,
        "xg_scope_status": "NOT_EVALUATED",
        "xg_missing_reason": "SOURCE_PAGE_UNAVAILABLE",
        "xg_source_match_id": candidate.source_match_id,
        "xg_source_url": source_url,
        "xg_snapshot_utc": snapshot_utc,
    }
    audit: dict[str, object] = {
        "fotmob_fetch_status": fetch_status,
        "detail_identity_verified": False,
        "detail_score_verified": False,
        "detail_chronology_verified": False,
        "xg_disposition": "SOURCE_PAGE_UNAVAILABLE",
    }
    if fetch_status == "ACCESS_BLOCKED":
        base["xg_missing_reason"] = "ACCESS_BLOCKED"
        audit["xg_disposition"] = "ACCESS_BLOCKED"
        return base, audit
    if payload is None:
        return base, audit

    general = payload.get("general") or {}
    header = payload.get("header") or {}
    teams = header.get("teams") or []
    if len(teams) != 2:
        base["xg_missing_reason"] = "IDENTITY_AMBIGUOUS"
        audit["xg_disposition"] = "IDENTITY_AMBIGUOUS"
        return base, audit
    detail_match_id = str(general.get("matchId") or "")
    detail_home = str((general.get("homeTeam") or {}).get("name") or teams[0].get("name") or "")
    detail_away = str((general.get("awayTeam") or {}).get("name") or teams[1].get("name") or "")
    identity_ok = bool(
        detail_match_id == candidate.source_match_id
        and str((general.get("homeTeam") or {}).get("id") or teams[0].get("id") or "")
        == candidate.home_team_id
        and str((general.get("awayTeam") or {}).get("id") or teams[1].get("id") or "")
        == candidate.away_team_id
    )
    audit["detail_identity_verified"] = identity_ok
    if not identity_ok:
        base["xg_missing_reason"] = "IDENTITY_AMBIGUOUS"
        audit["xg_disposition"] = "IDENTITY_AMBIGUOUS"
        return base, audit

    score_ok = bool(
        teams[0].get("score") is not None
        and teams[1].get("score") is not None
        and int(teams[0]["score"]) == int(event["home_goals"])
        and int(teams[1]["score"]) == int(event["away_goals"])
    )
    audit["detail_score_verified"] = score_ok
    if not score_ok:
        base["xg_missing_reason"] = "SCORE_MISMATCH"
        audit["xg_disposition"] = "SCORE_MISMATCH"
        return base, audit

    detail_time = pd.to_datetime(
        general.get("matchTimeUTCDate") or (header.get("status") or {}).get("utcTime"),
        utc=True,
        errors="coerce",
    )
    chronology_ok = bool(
        pd.notna(detail_time)
        and abs(detail_time - pd.Timestamp(event["kickoff_utc"])) <= pd.Timedelta(hours=36)
    )
    audit["detail_chronology_verified"] = chronology_ok
    if not chronology_ok:
        base["xg_missing_reason"] = "IDENTITY_AMBIGUOUS"
        audit["xg_disposition"] = "CHRONOLOGY_MISMATCH"
        return base, audit

    periods = (((payload.get("content") or {}).get("stats") or {}).get("Periods") or {})
    xg = _find_stat(periods.get("All") or {}, "expected_goals")
    if xg is None:
        base["xg_scope_status"] = "NO_PUBLIC_XG"
        base["xg_missing_reason"] = "NO_PUBLIC_XG"
        audit["xg_disposition"] = "NO_PUBLIC_XG"
        return base, audit
    base["xg_home_raw"], base["xg_away_raw"] = xg

    extra_time = _has_extra_time(payload)
    base["xg_duration_scope"] = (
        "120_minutes_including_extra_time" if extra_time else "90_minutes"
    )
    base["xg_extra_time_included"] = extra_time
    base["xg_in_match_penalties_included"] = True
    decided_on_penalties = _coerce_bool(event["decided_on_penalties"])
    shootout_present, shotmap_reconciled = _shotmap_scope_check(payload, xg)
    if decided_on_penalties and not (shootout_present and shotmap_reconciled):
        base["xg_shootout_excluded"] = pd.NA
        base["xg_scope_status"] = "PENALTY_SCOPE_UNVERIFIED"
        base["xg_missing_reason"] = "PENALTY_SCOPE_UNVERIFIED"
        audit["xg_disposition"] = "PENALTY_SCOPE_UNVERIFIED"
        return base, audit
    base["xg_shootout_excluded"] = True

    if extra_time:
        extra_periods = [periods.get("FirstExtraHalf"), periods.get("SecondExtraHalf")]
        if not any(_find_stat(period or {}, "expected_goals") for period in extra_periods):
            base["xg_scope_status"] = "EXTRA_TIME_SCOPE_MISMATCH"
            base["xg_missing_reason"] = "EXTRA_TIME_SCOPE_MISMATCH"
            audit["xg_disposition"] = "EXTRA_TIME_SCOPE_MISMATCH"
            return base, audit

    home_xg, away_xg = xg
    total = home_xg + away_xg
    base.update(
        {
            "xg_home": home_xg,
            "xg_away": away_xg,
            "xg_total": total,
            "xg_difference": home_xg - away_xg,
            "xg_home_share": home_xg / total if total > 0.0 else 0.5,
            "home_goals_minus_xg": int(event["home_goals"]) - home_xg,
            "away_goals_minus_xg": int(event["away_goals"]) - away_xg,
            "xg_result": 1.0 if home_xg > away_xg else 0.0 if home_xg < away_xg else 0.5,
            "xg_covered": True,
            "xg_analysis_eligible": True,
            "xg_scope_status": "ELIGIBLE_120" if extra_time else "ELIGIBLE_90",
            "xg_missing_reason": "",
        }
    )
    audit["xg_disposition"] = "ACCEPTED"
    return base, audit


def empty_primary_xg(reason: str) -> dict[str, object]:
    if reason not in MISSING_REASONS:
        raise ValueError(f"unknown xG missing reason: {reason}")
    return {
        **{column: math.nan for column in PRIMARY_XG_COLUMNS},
        "xg_covered": False,
        "xg_analysis_eligible": False,
        "xg_home_raw": math.nan,
        "xg_away_raw": math.nan,
        "xg_provider": "FotMob",
        "xg_type": "shot_based_expected_goals",
        "xg_duration_scope": "",
        "xg_in_match_penalties_included": pd.NA,
        "xg_extra_time_included": pd.NA,
        "xg_shootout_excluded": pd.NA,
        "xg_scope_status": reason,
        "xg_missing_reason": reason,
        "xg_source_match_id": "",
        "xg_source_url": "",
        "xg_snapshot_utc": "",
    }


def attach_secondary_xg(master: pd.DataFrame, secondary: pd.DataFrame) -> pd.DataFrame:
    values = secondary.loc[secondary["season"].astype(str).eq("2025/26")].copy()
    if values["match_id"].duplicated().any():
        raise ValueError("secondary xG contains duplicate match_id values")
    values = values.rename(
        columns={
            "xg_home": "secondary_xg_home",
            "xg_away": "secondary_xg_away",
            "xg_type": "secondary_xg_type",
            "provider": "secondary_xg_provider",
            "xg_covered": "secondary_xg_covered",
        }
    )
    columns = [
        "match_id",
        "secondary_xg_home",
        "secondary_xg_away",
        "secondary_xg_type",
        "secondary_xg_provider",
        "secondary_xg_covered",
    ]
    result = master.merge(values[columns], on="match_id", how="left", validate="one_to_one")
    result["secondary_xg_covered"] = result["secondary_xg_covered"].fillna(False).astype(bool)
    result["secondary_xg_type"] = result["secondary_xg_type"].fillna("")
    result["secondary_xg_provider"] = result["secondary_xg_provider"].fillna("")
    return result


def build_coverage_summary(master: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_keys = list(master.groupby(["competition", "round"], dropna=False, sort=True))
    group_keys += [
        ((competition, "ALL"), frame)
        for competition, frame in master.groupby("competition", sort=True)
    ]
    group_keys.append((("ALL", "ALL"), master))
    for (competition, round_name), frame in group_keys:
        covered = int(frame["xg_covered"].sum())
        rows.append(
            {
                "competition": competition,
                "round": round_name,
                "matches": len(frame),
                "xg_covered": covered,
                "xg_analysis_eligible": int(frame["xg_analysis_eligible"].sum()),
                "coverage_rate": covered / len(frame),
                "missing_xg": int((~frame["xg_covered"]).sum()),
                "identity_rejected": int(
                    frame["xg_missing_reason"].isin({"IDENTITY_AMBIGUOUS", "SCORE_MISMATCH"}).sum()
                ),
                "time_scope_rejected": int(
                    frame["xg_missing_reason"].isin(
                        {
                            "TIME_SCOPE_UNKNOWN",
                            "EXTRA_TIME_SCOPE_MISMATCH",
                            "PENALTY_SCOPE_UNVERIFIED",
                        }
                    ).sum()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["competition", "round"], kind="stable").reset_index(drop=True)


def build_source_comparison(master: pd.DataFrame) -> pd.DataFrame:
    common = master.loc[master["xg_covered"] & master["secondary_xg_covered"]].copy()
    columns = [
        "match_id",
        "competition",
        "kickoff_utc",
        "home_team_name",
        "away_team_name",
        "xg_home",
        "xg_away",
        "secondary_xg_home",
        "secondary_xg_away",
        "home_xg_delta_primary_minus_secondary",
        "away_xg_delta_primary_minus_secondary",
        "pair_absolute_difference",
    ]
    if common.empty:
        return pd.DataFrame(columns=columns)
    common["home_xg_delta_primary_minus_secondary"] = common["xg_home"] - common["secondary_xg_home"]
    common["away_xg_delta_primary_minus_secondary"] = common["xg_away"] - common["secondary_xg_away"]
    common["pair_absolute_difference"] = (
        common["home_xg_delta_primary_minus_secondary"].abs()
        + common["away_xg_delta_primary_minus_secondary"].abs()
    ) / 2.0
    return common[columns].reset_index(drop=True)


def validate_master_dataset(master: pd.DataFrame, *, strict_counts: bool = True) -> None:
    missing = sorted(set(MASTER_COLUMNS) - set(master.columns))
    if missing:
        raise ValueError(f"master dataset missing columns: {missing}")
    if master["match_id"].isna().any() or master["match_id"].duplicated().any():
        raise ValueError("master.match_id must be non-null and unique")
    if master.duplicated().any():
        raise ValueError("master dataset contains duplicate rows")
    if master["uefa_match_id"].isna().any() or master["uefa_match_id"].duplicated().any():
        raise ValueError("master.uefa_match_id must be non-null and unique")
    # Unresolved rows carry an empty string rather than a null, so blanks have
    # to be excluded explicitly before the uniqueness check.
    source_ids = master["xg_source_match_id"].astype("string").str.strip()
    resolved = source_ids.loc[source_ids.notna() & source_ids.ne("")]
    if resolved.duplicated().any():
        raise ValueError("FotMob source match IDs must be unique")
    if strict_counts and len(resolved) != len(master):
        # The frozen 2025/26 dataset resolved every match. Wider windows carry a
        # small unresolved tail, and those rows already record why in
        # `xg_missing_reason`, so completeness is a frozen-dataset guarantee
        # rather than a property of the collector.
        raise ValueError("all FotMob source match IDs must be populated and unique")
    if strict_counts:
        if len(master) != EXPECTED_MATCHES:
            raise ValueError(f"expected {EXPECTED_MATCHES} matches, found {len(master)}")
        counts = master["competition"].value_counts().to_dict()
        if counts != EXPECTED_COMPETITION_ROWS:
            raise ValueError(f"unexpected competition counts: {counts}")
        teams = pd.unique(pd.concat([master["home_team_id"], master["away_team_id"]]))
        if len(teams) != EXPECTED_TEAMS:
            raise ValueError(f"expected {EXPECTED_TEAMS} teams, found {len(teams)}")
    if strict_counts and not master["season"].astype(str).eq("2025/26").all():
        raise ValueError("master dataset contains another season")
    kickoff = pd.to_datetime(master["kickoff_utc"], utc=True, errors="coerce")
    if kickoff.isna().any():
        raise ValueError("master.kickoff_utc contains invalid values")
    expected_order = master.assign(_kickoff=kickoff).sort_values(
        ["_kickoff", "match_id"], kind="stable"
    )["match_id"].tolist()
    if master["match_id"].tolist() != expected_order:
        raise ValueError("master dataset is not sorted by kickoff_utc + match_id")
    if not master["event_order"].tolist() == list(range(1, len(master) + 1)):
        raise ValueError("master.event_order must be consecutive chronological order")
    if not master["score_verified"].eq(True).all():
        raise ValueError("all UEFA scores must be verified")
    if not master["chronology_verified"].eq(True).all():
        raise ValueError("all UEFA kickoff timestamps must be verified")
    if not master["kickoff_date"].astype(str).eq(kickoff.dt.date.astype(str)).all():
        raise ValueError("kickoff_date must equal the UTC calendar date")

    # `team_id` is a season-local identifier, so the same value legitimately
    # names different clubs in different seasons. The bijection therefore has
    # to be checked inside a season, not across the whole window.
    team_identity = pd.concat(
        [
            master[["season", "home_team_id", "uefa_home_team_id"]].rename(
                columns={"home_team_id": "ao_team_id", "uefa_home_team_id": "uefa_team_id"}
            ),
            master[["season", "away_team_id", "uefa_away_team_id"]].rename(
                columns={"away_team_id": "ao_team_id", "uefa_away_team_id": "uefa_team_id"}
            ),
        ],
        ignore_index=True,
    ).drop_duplicates()
    if (
        team_identity.groupby(["season", "ao_team_id"])["uefa_team_id"]
        .nunique()
        .gt(1)
        .any()
    ):
        raise ValueError("an AO team maps to multiple UEFA team IDs")
    if (
        team_identity.groupby(["season", "uefa_team_id"])["ao_team_id"]
        .nunique()
        .gt(1)
        .any()
    ):
        raise ValueError("a UEFA team maps to multiple AO team IDs")

    expected_score = master.apply(
        lambda row: 1.0
        if int(row["home_goals"]) > int(row["away_goals"])
        else 0.0
        if int(row["home_goals"]) < int(row["away_goals"])
        else 0.5,
        axis=1,
    )
    _assert_close(master["actual_home_score"], expected_score, "actual_home_score")
    _assert_close(
        master["goal_difference"],
        (master["home_goals"] - master["away_goals"]).abs(),
        "goal_difference",
    )

    covered = master["xg_covered"]
    if master.loc[covered, ["xg_home", "xg_away"]].isna().any().any():
        raise ValueError("covered xG rows require both values")
    if master.loc[~covered, ["xg_home", "xg_away"]].notna().any().any():
        raise ValueError("uncovered xG rows must leave both main values empty")
    for column in ("xg_home", "xg_away"):
        values = pd.to_numeric(master.loc[covered, column], errors="coerce")
        if values.isna().any() or not values.map(math.isfinite).all() or values.lt(0.0).any():
            raise ValueError(f"covered {column} must be finite and non-negative")
    reasons = set(master.loc[~covered, "xg_missing_reason"].astype(str))
    unknown = sorted(reasons - MISSING_REASONS)
    if unknown:
        raise ValueError(f"unknown xG missing reasons: {unknown}")
    if master.loc[covered, "xg_missing_reason"].fillna("").astype(str).ne("").any():
        raise ValueError("covered xG rows cannot have a missing reason")

    expected_total = master.loc[covered, "xg_home"] + master.loc[covered, "xg_away"]
    expected_difference = master.loc[covered, "xg_home"] - master.loc[covered, "xg_away"]
    _assert_close(master.loc[covered, "xg_total"], expected_total, "xg_total")
    _assert_close(master.loc[covered, "xg_difference"], expected_difference, "xg_difference")
    _assert_close(
        master.loc[covered, "home_goals_minus_xg"],
        master.loc[covered, "home_goals"] - master.loc[covered, "xg_home"],
        "home_goals_minus_xg",
    )
    _assert_close(
        master.loc[covered, "away_goals_minus_xg"],
        master.loc[covered, "away_goals"] - master.loc[covered, "xg_away"],
        "away_goals_minus_xg",
    )
    if (
        master.loc[covered, "xg_provider"].ne("FotMob").any()
        or master.loc[covered, "xg_type"].ne("shot_based_expected_goals").any()
    ):
        raise ValueError("primary xG columns may contain only FotMob shot-based xG")
    if master.loc[master["secondary_xg_covered"], "secondary_xg_type"].ne(
        "coarse_zone_derived_xg"
    ).any():
        raise ValueError("secondary xG must remain coarse_zone_derived_xg")


def _assert_close(actual: Iterable[float], expected: Iterable[float], label: str) -> None:
    for actual_value, expected_value in zip(actual, expected):
        if not math.isclose(float(actual_value), float(expected_value), abs_tol=1e-9):
            raise ValueError(f"{label} failed recomputation")


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().casefold() in {"true", "1"}
