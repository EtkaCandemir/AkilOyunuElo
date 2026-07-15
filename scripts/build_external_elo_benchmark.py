from __future__ import annotations

import argparse
import hashlib
import json
import math
import ssl
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import certifi
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_backtest_dataset import normalize_name  # noqa: E402


EVENTS_PATH = ROOT / "data" / "dynamic_backtest_2018_2026" / "matches.csv"
STATIC_DATA_ROOT = ROOT / "data" / "backtest_stage_b_2018_2026"
OUTPUT_ROOT = ROOT / "data" / "external_elo_benchmark_2018_2026"
CACHE_ROOT = OUTPUT_ROOT / "_source_cache"
UEFA_MATCHES_URL = "https://match.uefa.com/v5/matches"
CLUBELO_ARCHIVE_URL = (
    "https://raw.githubusercontent.com/xgabora/"
    "Club-Football-Match-Data-2000-2025/main/data/EloRatings.csv"
)
COMPETITION_IDS = {"UCL": 1, "UEL": 14, "UECL": 2019}
PAGE_SIZE = 500
FUZZY_MIN_SCORE = 0.86
FUZZY_MIN_GAP = 0.08
MAX_CLUBELO_SNAPSHOT_AGE_DAYS = 31
COUNTRY_CODE_ALIASES = {
    "AZB": "AZE",
    "BLS": "BLR",
    "BOS": "BIH",
    "FAR": "FRO",
    "LAT": "LVA",
    "LIT": "LTU",
    "MAC": "MKD",
    "MOL": "MDA",
    "MON": "MNE",
    "ROM": "ROU",
    "SLO": "SVN",
    "SMA": "SMR",
}


@dataclass(frozen=True)
class TeamResolution:
    uefa_team_id: str | None
    uefa_team_name: str | None
    method: str
    similarity: float | None
    runner_up_similarity: float | None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build exact-date UEFA matches with a pre-match external ClubElo benchmark"
    )
    parser.add_argument("--events-path", type=Path, default=EVENTS_PATH)
    parser.add_argument("--static-data-root", type=Path, default=STATIC_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--allow-fuzzy-team-matches",
        action="store_true",
        help="Accept high-confidence same-country UEFA name matches and audit them",
    )
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    cache_root = output_root / "_source_cache"
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    events = load_events(args.events_path.resolve())
    official_matches = load_uefa_matches(
        events,
        cache_root / "uefa",
        refresh=args.refresh,
    )
    team_map = resolve_uefa_teams(
        events,
        args.static_data_root.resolve(),
        official_matches,
        allow_fuzzy=args.allow_fuzzy_team_matches,
    )
    dated, match_audit = attach_exact_dates(events, official_matches, team_map)

    clubelo_path = cache_root / "clubelo" / "EloRatings.csv"
    download_file(CLUBELO_ARCHIVE_URL, clubelo_path, refresh=args.refresh)
    snapshots = load_clubelo_snapshots(clubelo_path)
    benchmark, provider_audit = attach_external_elo(dated, snapshots)
    exact_events = make_exact_events(benchmark, events.columns.tolist())
    build_audit = build_audit_table(benchmark, team_map, match_audit, provider_audit)
    manifest = source_manifest(cache_root, clubelo_path, benchmark)

    benchmark.to_csv(output_root / "matches_with_dates_and_external_elo.csv", index=False)
    exact_events.to_csv(output_root / "exact_date_events.csv", index=False)
    team_map.to_csv(output_root / "uefa_team_identity_audit.csv", index=False)
    match_audit.to_csv(output_root / "uefa_match_identity_audit.csv", index=False)
    provider_audit.to_csv(output_root / "clubelo_identity_audit.csv", index=False)
    build_audit.to_csv(output_root / "build_audit.csv", index=False)
    (output_root / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    write_readme(output_root / "README.md", benchmark, build_audit, manifest)

    exact = int(benchmark["exact_date_available"].sum())
    paired = int(benchmark["external_elo_pair_available"].sum())
    print("AO historical external Elo benchmark dataset")
    print(f"Matches: {len(benchmark)}")
    print(f"Exact UEFA dates: {exact}/{len(benchmark)} ({exact / len(benchmark):.1%})")
    print(f"Paired ClubElo ratings: {paired}/{len(benchmark)} ({paired / len(benchmark):.1%})")
    print(f"Output: {output_root / 'matches_with_dates_and_external_elo.csv'}")


def load_events(path: Path) -> pd.DataFrame:
    events = pd.read_csv(path)
    required = {
        "match_id",
        "season",
        "competition",
        "home_team_id",
        "away_team_id",
        "home_team_name",
        "away_team_name",
        "home_goals",
        "away_goals",
        "actual_home_score",
        "is_neutral",
    }
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"Event data missing columns: {missing}")
    if events["match_id"].duplicated().any():
        raise ValueError("Event match_id must be unique")
    unknown = sorted(set(events["competition"]) - set(COMPETITION_IDS))
    if unknown:
        raise ValueError(f"Unsupported competitions: {unknown}")
    return events.sort_values(["season", "event_order"], kind="stable").reset_index(drop=True)


def download_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "AO-European-Elo/1.0"})
    context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(request, timeout=90, context=context) as response:
        return response.read()


def download_file(url: str, path: Path, *, refresh: bool) -> None:
    if path.exists() and not refresh:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = download_bytes(url)
    if not payload:
        raise ValueError(f"Downloaded source is empty: {url}")
    path.write_bytes(payload)


def season_end_year(season: str) -> int:
    start_text, end_text = str(season).split("/", maxsplit=1)
    start = int(start_text)
    century = start // 100 * 100
    end = century + int(end_text)
    if end < start:
        end += 100
    return end


def load_uefa_matches(events: pd.DataFrame, cache_dir: Path, *, refresh: bool) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    combinations = events[["season", "competition"]].drop_duplicates().itertuples(index=False)
    for season, competition in combinations:
        end_year = season_end_year(str(season))
        offset = 0
        while True:
            cache_path = cache_dir / f"{competition}_{end_year}_offset_{offset:04d}.json"
            query = urlencode(
                {
                    "competitionId": COMPETITION_IDS[str(competition)],
                    "seasonYear": end_year,
                    "limit": PAGE_SIZE,
                    "offset": offset,
                    "order": "ASC",
                }
            )
            download_file(f"{UEFA_MATCHES_URL}?{query}", cache_path, refresh=refresh)
            page = json.loads(cache_path.read_text(encoding="utf-8"))
            if not isinstance(page, list):
                raise ValueError(f"Unexpected UEFA payload in {cache_path}")
            for item in page:
                parsed = parse_uefa_match(item)
                parsed["season"] = season
                parsed["competition"] = competition
                records.append(parsed)
            if len(page) < PAGE_SIZE:
                break
            offset += PAGE_SIZE
    result = pd.DataFrame(records)
    if result.empty:
        raise ValueError("No UEFA matches were downloaded")
    if result["uefa_match_id"].duplicated().any():
        duplicated = result.loc[result["uefa_match_id"].duplicated(False), "uefa_match_id"]
        raise ValueError(f"Duplicate UEFA match IDs: {duplicated.head(3).tolist()}")
    return result


def english_translation(team: dict[str, object], key: str) -> str:
    translations = team.get("translations") or {}
    values = translations.get(key) or {}
    return str(values.get("EN") or "")


def team_aliases(team: dict[str, object]) -> tuple[str, ...]:
    values = {
        str(team.get("internationalName") or ""),
        str(team.get("teamCode") or ""),
        english_translation(team, "displayName"),
        english_translation(team, "displayOfficialName"),
        english_translation(team, "shortName"),
    }
    return tuple(sorted({normalize_name(value) for value in values if normalize_name(value)}))


def parse_uefa_match(item: dict[str, object]) -> dict[str, object]:
    home = item.get("homeTeam") or {}
    away = item.get("awayTeam") or {}
    kickoff = item.get("kickOffTime") or {}
    score = item.get("score") or {}
    total = score.get("total") or {}
    regular = score.get("regular") or {}
    round_data = item.get("round") or {}
    round_translations = round_data.get("translations") or {}
    round_names = round_translations.get("name") or {}
    return {
        "uefa_match_id": str(item.get("id") or ""),
        "kickoff_date": str(kickoff.get("date") or ""),
        "kickoff_utc": str(kickoff.get("dateTime") or ""),
        "uefa_status": str(item.get("status") or ""),
        "uefa_round": str(round_names.get("EN") or (round_data.get("metaData") or {}).get("name") or ""),
        "uefa_home_team_id": str(home.get("id") or ""),
        "uefa_away_team_id": str(away.get("id") or ""),
        "uefa_home_team_name": str(home.get("internationalName") or english_translation(home, "displayName")),
        "uefa_away_team_name": str(away.get("internationalName") or english_translation(away, "displayName")),
        "uefa_home_country_code": str(home.get("countryCode") or "").upper(),
        "uefa_away_country_code": str(away.get("countryCode") or "").upper(),
        "uefa_home_aliases": team_aliases(home),
        "uefa_away_aliases": team_aliases(away),
        "uefa_home_goals_total": numeric_or_none(total.get("home")),
        "uefa_away_goals_total": numeric_or_none(total.get("away")),
        "uefa_home_goals_regular": numeric_or_none(regular.get("home")),
        "uefa_away_goals_regular": numeric_or_none(regular.get("away")),
    }


def numeric_or_none(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def uefa_team_catalog(official: pd.DataFrame, season: str) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    season_data = official.loc[official["season"].eq(season)]
    for row in season_data.itertuples(index=False):
        for side in ("home", "away"):
            team_id = str(getattr(row, f"uefa_{side}_team_id"))
            name = str(getattr(row, f"uefa_{side}_team_name"))
            country = str(getattr(row, f"uefa_{side}_country_code"))
            aliases = set(getattr(row, f"uefa_{side}_aliases"))
            existing = rows.setdefault(
                team_id,
                {"name": name, "country_code": country, "aliases": set()},
            )
            if existing["country_code"] != country:
                raise ValueError(f"UEFA team {team_id} has multiple country codes")
            existing["aliases"].update(aliases)
    return rows


def normalize_country_code(value: object) -> str:
    code = str(value).strip().upper()
    return COUNTRY_CODE_ALIASES.get(code, code)


def resolve_uefa_teams(
    events: pd.DataFrame,
    static_root: Path,
    official: pd.DataFrame,
    *,
    allow_fuzzy: bool,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for season in events["season"].drop_duplicates():
        teams = pd.read_csv(static_root / str(season).replace("/", "-") / "teams.csv")
        catalog = uefa_team_catalog(official, str(season))
        for team in teams.itertuples(index=False):
            normalized = normalize_name(str(team.team_name))
            uefa_country_code = normalize_country_code(team.country_code)
            candidates = [
                (team_id, details)
                for team_id, details in catalog.items()
                if details["country_code"] == uefa_country_code
            ]
            resolution = resolve_one_team(normalized, candidates, allow_fuzzy=False)
            rows.append(
                {
                    "season": season,
                    "local_team_id": int(team.team_id),
                    "ao_team_name": team.team_name,
                    "country_code": str(team.country_code).upper(),
                    "uefa_country_code": uefa_country_code,
                    "ao_normalized_name": normalized,
                    "uefa_team_id": resolution.uefa_team_id,
                    "uefa_team_name": resolution.uefa_team_name,
                    "resolution_method": resolution.method,
                    "similarity": resolution.similarity,
                    "runner_up_similarity": resolution.runner_up_similarity,
                }
            )
    result = pd.DataFrame(rows)
    result = resolve_teams_from_fixture_graph(events, official, result)
    if allow_fuzzy:
        result = resolve_remaining_teams_by_name(official, result)
    if result.duplicated(["season", "local_team_id"]).any():
        raise ValueError("UEFA team audit contains duplicate season/local_team_id keys")
    mapped = result.dropna(subset=["uefa_team_id"])
    duplicates = mapped.duplicated(["season", "uefa_team_id"], keep=False)
    if duplicates.any():
        sample = mapped.loc[duplicates, ["season", "uefa_team_id", "ao_team_name"]]
        raise ValueError(f"Multiple AO teams map to one UEFA team:\n{sample.head(6)}")
    return result


def resolve_teams_from_fixture_graph(
    events: pd.DataFrame,
    official: pd.DataFrame,
    team_map: pd.DataFrame,
) -> pd.DataFrame:
    result = team_map.copy()
    for season in events["season"].drop_duplicates():
        season_mask = result["season"].eq(season)
        season_events = events.loc[events["season"].eq(season)]
        season_official = official.loc[official["season"].eq(season)]
        while True:
            local = result.loc[season_mask].set_index("local_team_id")
            resolved = local["uefa_team_id"].dropna().astype(str).to_dict()
            countries = local["uefa_country_code"].astype(str).to_dict()
            used_ids = set(resolved.values())
            proposals: dict[int, set[str]] = {}
            evidence_counts: dict[int, int] = {}
            for event in season_events.itertuples(index=False):
                home_id = int(event.home_team_id)
                away_id = int(event.away_team_id)
                candidates = season_official.loc[
                    season_official["competition"].eq(event.competition)
                    & season_official["uefa_home_country_code"].eq(countries[home_id])
                    & season_official["uefa_away_country_code"].eq(countries[away_id])
                ]
                candidates = candidates.loc[
                    candidates.apply(
                        lambda row: score_agrees(
                            int(event.home_goals),
                            int(event.away_goals),
                            row,
                        ),
                        axis=1,
                    )
                ]
                if home_id in resolved:
                    candidates = candidates.loc[
                        candidates["uefa_home_team_id"].astype(str).eq(resolved[home_id])
                    ]
                if away_id in resolved:
                    candidates = candidates.loc[
                        candidates["uefa_away_team_id"].astype(str).eq(resolved[away_id])
                    ]
                if candidates.empty:
                    continue
                for local_id, column in (
                    (home_id, "uefa_home_team_id"),
                    (away_id, "uefa_away_team_id"),
                ):
                    if local_id in resolved:
                        continue
                    candidates_for_event = set(candidates[column].astype(str)) - used_ids
                    if not candidates_for_event:
                        continue
                    if local_id not in proposals:
                        proposals[local_id] = candidates_for_event
                    else:
                        proposals[local_id] &= candidates_for_event
                    evidence_counts[local_id] = evidence_counts.get(local_id, 0) + 1
            accepted = {
                local_id: next(iter(candidate_ids))
                for local_id, candidate_ids in proposals.items()
                if len(candidate_ids) == 1
            }
            duplicate_proposals = {
                team_id
                for team_id in accepted.values()
                if list(accepted.values()).count(team_id) > 1
            }
            accepted = {
                local_id: team_id
                for local_id, team_id in accepted.items()
                if team_id not in duplicate_proposals
            }
            if not accepted:
                break
            catalog = uefa_team_catalog(official, str(season))
            for local_id, uefa_id in accepted.items():
                row_mask = season_mask & result["local_team_id"].eq(local_id)
                result.loc[row_mask, "uefa_team_id"] = uefa_id
                result.loc[row_mask, "uefa_team_name"] = catalog[uefa_id]["name"]
                result.loc[row_mask, "resolution_method"] = "fixture_graph_unique"
                result.loc[row_mask, "similarity"] = float("nan")
                result.loc[row_mask, "runner_up_similarity"] = float("nan")
    return result


def resolve_remaining_teams_by_name(
    official: pd.DataFrame,
    team_map: pd.DataFrame,
) -> pd.DataFrame:
    result = team_map.copy()
    for season in result["season"].drop_duplicates():
        catalog = uefa_team_catalog(official, str(season))
        season_mask = result["season"].eq(season)
        used = set(result.loc[season_mask, "uefa_team_id"].dropna().astype(str))
        unresolved_indices = result.index[season_mask & result["uefa_team_id"].isna()]
        for index in unresolved_indices:
            row = result.loc[index]
            candidates = [
                (team_id, details)
                for team_id, details in catalog.items()
                if details["country_code"] == row["uefa_country_code"] and team_id not in used
            ]
            resolution = resolve_one_team(
                str(row["ao_normalized_name"]),
                candidates,
                allow_fuzzy=True,
            )
            if resolution.uefa_team_id is None:
                continue
            result.loc[index, "uefa_team_id"] = resolution.uefa_team_id
            result.loc[index, "uefa_team_name"] = resolution.uefa_team_name
            result.loc[index, "resolution_method"] = resolution.method
            result.loc[index, "similarity"] = resolution.similarity
            result.loc[index, "runner_up_similarity"] = resolution.runner_up_similarity
            used.add(resolution.uefa_team_id)
    return result


def resolve_one_team(
    normalized_name: str,
    candidates: list[tuple[str, dict[str, object]]],
    *,
    allow_fuzzy: bool,
) -> TeamResolution:
    exact = [
        (team_id, details)
        for team_id, details in candidates
        if normalized_name in details["aliases"]
    ]
    if len(exact) == 1:
        team_id, details = exact[0]
        return TeamResolution(team_id, str(details["name"]), "exact_name_country", 1.0, None)
    if len(exact) > 1:
        return TeamResolution(None, None, "ambiguous_exact", 1.0, 1.0)

    scored: list[tuple[float, str, dict[str, object]]] = []
    for team_id, details in candidates:
        similarity = max(
            (SequenceMatcher(None, normalized_name, alias).ratio() for alias in details["aliases"]),
            default=0.0,
        )
        scored.append((similarity, team_id, details))
    scored.sort(reverse=True, key=lambda item: item[0])
    if not scored:
        return TeamResolution(None, None, "no_same_country_candidate", None, None)
    best_score, best_id, best = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else 0.0
    accepted = allow_fuzzy and best_score >= FUZZY_MIN_SCORE and best_score - runner_up >= FUZZY_MIN_GAP
    if accepted:
        return TeamResolution(best_id, str(best["name"]), "high_confidence_fuzzy_country", best_score, runner_up)
    return TeamResolution(None, str(best["name"]), "unresolved_name", best_score, runner_up)


def attach_exact_dates(
    events: pd.DataFrame,
    official: pd.DataFrame,
    team_map: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = team_map.set_index(["season", "local_team_id"])["uefa_team_id"]
    dated = events.copy()
    dated["uefa_home_team_id"] = [
        mapping.get((season, int(team_id)))
        for season, team_id in zip(dated["season"], dated["home_team_id"])
    ]
    dated["uefa_away_team_id"] = [
        mapping.get((season, int(team_id)))
        for season, team_id in zip(dated["season"], dated["away_team_id"])
    ]
    lookup: dict[tuple[str, str, str, str], list[int]] = {}
    for index, row in official.iterrows():
        key = (
            str(row["season"]),
            str(row["competition"]),
            str(row["uefa_home_team_id"]),
            str(row["uefa_away_team_id"]),
        )
        lookup.setdefault(key, []).append(index)

    attached_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    used_official_ids: set[str] = set()
    for row in dated.itertuples(index=False):
        key = (
            str(row.season),
            str(row.competition),
            str(row.uefa_home_team_id),
            str(row.uefa_away_team_id),
        )
        candidate_indices = lookup.get(key, []) if None not in key else []
        candidates = official.loc[candidate_indices]
        if len(candidates) > 1:
            round_mask = candidates["uefa_round"].map(canonical_round).eq(
                canonical_round(str(row.round))
            )
            if round_mask.sum() == 1:
                candidates = candidates.loc[round_mask]
        if len(candidates) > 1:
            score_mask = candidates.apply(
                lambda candidate: score_agrees(
                    int(row.home_goals),
                    int(row.away_goals),
                    candidate,
                ),
                axis=1,
            )
            if score_mask.sum() == 1:
                candidates = candidates.loc[score_mask]
        if len(candidates) == 1:
            official_row = candidates.iloc[0]
            official_id = str(official_row["uefa_match_id"])
            if official_id in used_official_ids:
                raise ValueError(f"UEFA match {official_id} matched more than once")
            used_official_ids.add(official_id)
            score_match = score_agrees(int(row.home_goals), int(row.away_goals), official_row)
            metadata = official_row.to_dict()
            status = "MATCHED" if score_match else "MATCHED_SCORE_DIFFERENCE"
        else:
            metadata = {column: None for column in official.columns if column not in {"season", "competition"}}
            status = "UNMATCHED" if len(candidates) == 0 else "AMBIGUOUS"
            score_match = False
        attached_rows.append(metadata)
        audit_rows.append(
            {
                "match_id": row.match_id,
                "season": row.season,
                "competition": row.competition,
                "home_team_name": row.home_team_name,
                "away_team_name": row.away_team_name,
                "candidate_count": len(candidates),
                "match_status": status,
                "score_agrees": score_match,
            }
        )
    metadata = pd.DataFrame(attached_rows).drop(columns=["season", "competition"], errors="ignore")
    duplicated_columns = [column for column in metadata if column in dated.columns]
    metadata = metadata.drop(columns=duplicated_columns, errors="ignore")
    result = pd.concat([dated.reset_index(drop=True), metadata.reset_index(drop=True)], axis=1)
    result["exact_date_available"] = result["kickoff_utc"].notna()
    return result, pd.DataFrame(audit_rows)


def canonical_round(value: str) -> str:
    key = normalize_name(value)
    if "preliminary" in key:
        return "preliminary"
    if "qualifying" in key:
        if "1st" in key or "first" in key:
            return "qualifying_1"
        if "2nd" in key or "second" in key:
            return "qualifying_2"
        if "3rd" in key or "third" in key:
            return "qualifying_3"
        if "playoff" in key or "playoffs" in key:
            return "qualifying_playoff"
    if "knockout" in key and "playoff" in key:
        return "knockout_playoff"
    if "group" in key:
        return "group"
    if "league" in key:
        return "league"
    if "roundof16" in key:
        return "round_of_16"
    if "quarter" in key:
        return "quarter_final"
    if "semi" in key:
        return "semi_final"
    if key == "final":
        return "final"
    if key in {"round2", "secondround"}:
        return "round_2"
    if key in {"round3", "thirdround"}:
        return "round_3"
    return key


def score_agrees(home_goals: int, away_goals: int, official: pd.Series) -> bool:
    pairs = {
        (official.get("uefa_home_goals_total"), official.get("uefa_away_goals_total")),
        (official.get("uefa_home_goals_regular"), official.get("uefa_away_goals_regular")),
    }
    return (home_goals, away_goals) in pairs


def load_clubelo_snapshots(path: Path) -> pd.DataFrame:
    snapshots = pd.read_csv(path)
    snapshots.columns = [str(column).strip().lower() for column in snapshots.columns]
    required = {"date", "club", "country", "elo"}
    missing = sorted(required - set(snapshots.columns))
    if missing:
        raise ValueError(f"ClubElo archive missing columns: {missing}")
    snapshots["snapshot_date"] = pd.to_datetime(snapshots["date"], errors="raise")
    snapshots["country_code"] = snapshots["country"].map(normalize_country_code)
    snapshots["normalized_name"] = snapshots["club"].map(normalize_name)
    snapshots["elo"] = pd.to_numeric(snapshots["elo"], errors="raise")
    if not snapshots["elo"].map(math.isfinite).all():
        raise ValueError("ClubElo archive contains non-finite ratings")
    key = ["snapshot_date", "country_code", "normalized_name"]
    if snapshots.duplicated(key).any():
        raise ValueError("ClubElo archive contains duplicate snapshot club keys")
    return snapshots.sort_values("snapshot_date").reset_index(drop=True)


def attach_external_elo(
    dated: pd.DataFrame,
    snapshots: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = dated.copy()
    result["kickoff_date_parsed"] = pd.to_datetime(result["kickoff_date"], errors="coerce")
    identity_audit = resolve_clubelo_identities(result, snapshots)
    provider_keys = identity_audit.set_index(
        ["country_code", "ao_normalized_name"]
    )[["clubelo_normalized_name", "clubelo_club"]]
    for side in ("home", "away"):
        ao_names = result[f"{side}_team_name"].map(normalize_name)
        countries = result[f"uefa_{side}_country_code"].fillna("").astype(str).str.upper()
        result[f"clubelo_{side}_normalized_name"] = ao_names
        resolved_keys = [
            provider_keys.loc[(country, name), "clubelo_normalized_name"]
            if (country, name) in provider_keys.index
            else None
            for country, name in zip(countries, ao_names)
        ]
        result[f"clubelo_{side}_normalized_name"] = resolved_keys
        result[f"clubelo_{side}_club"] = [
            provider_keys.loc[(country, name), "clubelo_club"]
            if (country, name) in provider_keys.index
            else None
            for country, name in zip(countries, ao_names)
        ]
        looked_up = lookup_snapshots(
            result["kickoff_date_parsed"],
            countries,
            pd.Series(resolved_keys, index=result.index),
            snapshots,
        )
        result[f"clubelo_{side}_elo"] = looked_up["elo"]
        result[f"clubelo_{side}_snapshot_date"] = looked_up["snapshot_date"]
        result[f"clubelo_{side}_snapshot_age_days"] = (
            result["kickoff_date_parsed"] - looked_up["snapshot_date"]
        ).dt.days
    result["clubelo_snapshot_rule"] = "latest_snapshot_strictly_before_match_date"
    result["external_elo_pair_available"] = (
        result["exact_date_available"]
        & result["clubelo_home_elo"].notna()
        & result["clubelo_away_elo"].notna()
        & result["clubelo_home_snapshot_age_days"].le(MAX_CLUBELO_SNAPSHOT_AGE_DAYS)
        & result["clubelo_away_snapshot_age_days"].le(MAX_CLUBELO_SNAPSHOT_AGE_DAYS)
    )
    result = result.drop(columns=["kickoff_date_parsed"])
    return result, identity_audit


def resolve_clubelo_identities(dated: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
    catalog = snapshots[["country_code", "normalized_name", "club"]].drop_duplicates()
    teams: dict[tuple[str, str], str] = {}
    for side in ("home", "away"):
        for country, name in zip(
            dated[f"uefa_{side}_country_code"].fillna("").astype(str).str.upper(),
            dated[f"{side}_team_name"].astype(str),
        ):
            if country:
                teams.setdefault((country, normalize_name(name)), name)

    rows: list[dict[str, object]] = []
    for (country, normalized), ao_name in sorted(teams.items()):
        candidates = catalog.loc[catalog["country_code"].eq(country)]
        exact = candidates.loc[candidates["normalized_name"].eq(normalized)]
        if len(exact) == 1:
            selected = exact.iloc[0]
            method = "exact_name_country"
            similarity = 1.0
            runner_up = None
        else:
            scored = [
                (
                    SequenceMatcher(None, normalized, str(row.normalized_name)).ratio(),
                    row,
                )
                for row in candidates.itertuples(index=False)
            ]
            scored.sort(key=lambda item: item[0], reverse=True)
            best_score = scored[0][0] if scored else 0.0
            runner_up = scored[1][0] if len(scored) > 1 else 0.0
            if scored and best_score >= FUZZY_MIN_SCORE and best_score - runner_up >= FUZZY_MIN_GAP:
                selected = scored[0][1]
                method = "high_confidence_fuzzy_country"
                similarity = best_score
            else:
                selected = None
                method = "unresolved"
                similarity = best_score if scored else None
        rows.append(
            {
                "country_code": country,
                "ao_team_name": ao_name,
                "ao_normalized_name": normalized,
                "clubelo_club": selected.club if selected is not None else None,
                "clubelo_normalized_name": (
                    selected.normalized_name if selected is not None else None
                ),
                "resolution_method": method,
                "resolution_status": "MATCHED" if selected is not None else "UNMATCHED",
                "similarity": similarity,
                "runner_up_similarity": runner_up,
            }
        )
    result = pd.DataFrame(rows)
    if result.duplicated(["country_code", "ao_normalized_name"]).any():
        raise ValueError("ClubElo identity audit contains duplicate AO club keys")
    return result


def lookup_snapshots(
    match_dates: pd.Series,
    countries: pd.Series,
    names: pd.Series,
    snapshots: pd.DataFrame,
) -> pd.DataFrame:
    queries = pd.DataFrame(
        {
            "row_id": range(len(match_dates)),
            "match_date": match_dates,
            "country_code": countries.to_numpy(),
            "normalized_name": names.to_numpy(),
        }
    )
    valid = queries.dropna(subset=["match_date"])
    valid = valid.loc[valid["country_code"].ne("") & valid["normalized_name"].ne("")]
    values = snapshots[["snapshot_date", "country_code", "normalized_name", "elo"]]
    joined_parts: list[pd.DataFrame] = []
    for key, group in valid.groupby(["country_code", "normalized_name"], sort=False):
        available = values.loc[
            values["country_code"].eq(key[0]) & values["normalized_name"].eq(key[1])
        ]
        if available.empty:
            continue
        joined = pd.merge_asof(
            group.sort_values("match_date"),
            available.sort_values("snapshot_date"),
            left_on="match_date",
            right_on="snapshot_date",
            direction="backward",
            allow_exact_matches=False,
        )
        joined_parts.append(joined[["row_id", "elo", "snapshot_date"]])
    result = pd.DataFrame(
        {
            "elo": pd.Series(float("nan"), index=range(len(match_dates)), dtype=float),
            "snapshot_date": pd.Series(pd.NaT, index=range(len(match_dates)), dtype="datetime64[ns]"),
        }
    )
    if joined_parts:
        joined = pd.concat(joined_parts, ignore_index=True).dropna(subset=["elo"])
        row_ids = joined["row_id"].astype(int).to_numpy()
        result.loc[row_ids, "elo"] = joined["elo"].to_numpy(float)
        result.loc[row_ids, "snapshot_date"] = joined["snapshot_date"].to_numpy()
    return result


def make_exact_events(
    benchmark: pd.DataFrame,
    original_columns: list[str],
) -> pd.DataFrame:
    exact = benchmark.loc[benchmark["exact_date_available"]].copy()
    exact["kickoff_utc"] = pd.to_datetime(exact["kickoff_utc"], utc=True)
    exact = exact.sort_values(
        ["season", "kickoff_utc", "uefa_match_id", "match_id"],
        kind="stable",
    )
    exact["original_event_order"] = exact["event_order"]
    exact["event_order"] = exact.groupby("season").cumcount() + 1
    extra = [
        "original_event_order",
        "kickoff_date",
        "kickoff_utc",
        "uefa_match_id",
        "uefa_home_team_id",
        "uefa_away_team_id",
    ]
    return exact[original_columns + extra].reset_index(drop=True)


def build_audit_table(
    benchmark: pd.DataFrame,
    team_map: pd.DataFrame,
    match_audit: pd.DataFrame,
    provider_audit: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for season, data in benchmark.groupby("season", sort=True):
        rows.extend(
            [
                audit_row(season, "all_matches_have_exact_uefa_date", int(data["exact_date_available"].sum()), len(data)),
                audit_row(season, "uefa_scores_agree", int(match_audit.loc[match_audit["season"].eq(season), "score_agrees"].sum()), len(data), required=False),
                audit_row(season, "external_elo_pair_coverage", int(data["external_elo_pair_available"].sum()), len(data), required=False),
            ]
        )
    mapped_teams = int(team_map["uefa_team_id"].notna().sum())
    rows.append(audit_row("ALL", "uefa_team_identity_coverage", mapped_teams, len(team_map)))
    matched_provider = int(provider_audit["resolution_status"].eq("MATCHED").sum())
    rows.append(audit_row("ALL", "clubelo_identity_coverage", matched_provider, len(provider_audit), required=False))
    paired = benchmark.loc[benchmark["external_elo_pair_available"]]
    kickoff_dates = pd.to_datetime(paired["kickoff_date"], errors="coerce")
    home_snapshots = pd.to_datetime(paired["clubelo_home_snapshot_date"], errors="coerce")
    away_snapshots = pd.to_datetime(paired["clubelo_away_snapshot_date"], errors="coerce")
    leakage_pass = bool(
        paired.empty
        or (
            kickoff_dates.notna().all()
            and home_snapshots.notna().all()
            and away_snapshots.notna().all()
            and home_snapshots.lt(kickoff_dates).all()
            and away_snapshots.lt(kickoff_dates).all()
        )
    )
    rows.append(
        {
            "season": "ALL",
            "check": "strictly_pre_match_snapshot_rule",
            "passed": leakage_pass,
            "observed": len(paired),
            "expected": len(paired),
            "required": True,
        }
    )
    return pd.DataFrame(rows)


def audit_row(
    season: str,
    check: str,
    observed: int,
    expected: int,
    *,
    required: bool = True,
) -> dict[str, object]:
    return {
        "season": season,
        "check": check,
        "passed": observed == expected if required else True,
        "observed": observed,
        "expected": expected,
        "required": required,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest(cache_root: Path, clubelo_path: Path, benchmark: pd.DataFrame) -> dict[str, object]:
    uefa_files = sorted((cache_root / "uefa").glob("*.json"))
    return {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "uefa": {
            "endpoint": UEFA_MATCHES_URL,
            "competition_ids": COMPETITION_IDS,
            "cache_files": len(uefa_files),
            "exact_matches": int(benchmark["exact_date_available"].sum()),
        },
        "clubelo": {
            "archive_url": CLUBELO_ARCHIVE_URL,
            "local_sha256": sha256(clubelo_path),
            "snapshot_rule": "latest snapshot strictly before match date",
            "maximum_snapshot_age_days": MAX_CLUBELO_SNAPSHOT_AGE_DAYS,
            "archive_note": "ClubElo-derived snapshots published on the 1st and 15th",
        },
    }


def write_readme(
    path: Path,
    benchmark: pd.DataFrame,
    audit: pd.DataFrame,
    manifest: dict[str, object],
) -> None:
    coverage = (
        benchmark.groupby(["season", "competition"])
        .agg(
            matches=("match_id", "size"),
            exact_dates=("exact_date_available", "sum"),
            external_pairs=("external_elo_pair_available", "sum"),
        )
        .reset_index()
    )
    failures = audit.loc[audit["required"] & ~audit["passed"]]
    lines = [
        "# Exact-Date External Elo Benchmark Data",
        "",
        "This dataset attaches official UEFA kickoff timestamps and a historical external",
        "ClubElo benchmark to the AO dynamic match history.",
        "",
        "## Leakage contract",
        "",
        "- UEFA kickoff timestamps come from the official UEFA match service.",
        "- ClubElo values use the latest archived snapshot strictly before the match date.",
        "- Missing provider coverage stays missing; values are never imputed from the future.",
        f"- Snapshots older than {MAX_CLUBELO_SNAPSHOT_AGE_DAYS} days are excluded from paired coverage.",
        "- ClubElo archive snapshots are available on the 1st and 15th, so ratings can be stale",
        "  by up to roughly two weeks. This limitation must accompany benchmark results.",
        "",
        "## Files",
        "",
        "- `matches_with_dates_and_external_elo.csv`: full row-level benchmark contract.",
        "- `exact_date_events.csv`: dynamic events reordered by exact UTC kickoff.",
        "- `uefa_team_identity_audit.csv`: AO-to-UEFA club identity decisions.",
        "- `uefa_match_identity_audit.csv`: one-to-one fixture reconciliation.",
        "- `clubelo_identity_audit.csv`: provider name coverage.",
        "- `build_audit.csv`: required checks and descriptive coverage checks.",
        "- `source_manifest.json`: source endpoints, retrieval time and checksum.",
        "",
        "## Coverage",
        "",
        "```text",
        coverage.to_string(index=False),
        "```",
        "",
        f"Required audit failures: {len(failures)}",
        "",
        "ClubElo archive URL:",
        str(manifest["clubelo"]["archive_url"]),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
