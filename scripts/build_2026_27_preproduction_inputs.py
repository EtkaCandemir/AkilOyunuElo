from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ao_elo.club_identity import permanent_club_id  # noqa: E402
from ao_elo.config import AOEuropeanEloConfig  # noqa: E402
from ao_elo.dynamic_csv import read_fixtures, read_matches  # noqa: E402
from ao_elo.pipeline import compute_ao_first_elo_from_csv  # noqa: E402
from scripts.build_backtest_dataset import (  # noqa: E402
    SEASON_KEYS,
    fetch,
    history_team_key,
    match_cap,
    normalize_name,
    parse_ccoef,
    parse_crank,
    parse_qualification,
    parse_trank,
    read_cached_text,
)
from scripts.build_backtest_stage_b import (  # noqa: E402
    normalize_country,
    resolve_position,
)
from scripts.build_external_elo_benchmark import (  # noqa: E402
    load_uefa_matches,
    uefa_team_catalog,
)
from scripts.run_domestic_surprise_backtest import (  # noqa: E402
    load_standings_cache,
)


SEASON = "2026/27"
# Cache years are labelled by the season's closing year, so 2026 is the
# 2025/26 campaign that decided 2026/27 European entry and 2025 is 2024/25.
CURRENT_SEASON_VINTAGE = 2026
CACHE_FALLBACK_VINTAGE = 2025
END_YEAR = 2027
DEFAULT_DATA_ROOT = ROOT / "data" / "season_2026_27_preproduction"
DEFAULT_OUTPUT_ROOT = ROOT / "output" / "season_2026_27_preproduction"
HISTORICAL_STATIC_ROOT = ROOT / "data" / "backtest_stage_b_2018_2026"
HISTORICAL_SOURCE_CACHE = ROOT / "data" / "backtest_2018_2026" / "_source_cache"
CLUB_REGISTRY_PATH = ROOT / "data" / "club_identity" / "club_registry.csv"
KASSIESA_BASE = "https://kassiesa.net/uefa"

ROUND_NAME_MAP = {
    "Preliminary round": "Preliminary Round",
    "First qualifying round": "1st Qualifying Round",
    "Second qualifying round": "2nd Qualifying Round",
    "Third qualifying round": "3rd Qualifying Round",
    "Play-offs": "Qualifying Play-off Round",
}

# These IDs are official UEFA club-page or current fixture identities. They are
# only used when the source display name differs from both the current fixture
# aliases and the historical AO registry.
UEFA_ID_OVERRIDES = {
    "uniaotorreense": ("2603107", "https://www.uefa.com/uefaeuropaleague/clubs/2603107--torreense/"),
    "como1907": ("79946", "https://www.uefa.com/uefachampionsleague/clubs/79946--como/"),
    "afcbournemouth": ("2601124", "https://www.uefa.com/uefaeuropaleague/clubs/2601124--bournemouth/"),
    "sunderlandafc": ("53360", "https://www.uefa.com/uefaeuropaleague/clubs/53360--sunderland/"),
    "hapoeltelaviv": ("59869", "UEFA_2026_27_MATCH_FEED"),
    "ifvestri": ("2610384", "UEFA_2026_27_MATCH_FEED"),
    "fkyelimay": ("79966", "UEFA_2026_27_MATCH_FEED"),
    "zeleznicarpancevo": ("77924", "UEFA_2026_27_MATCH_FEED"),
    "gaisgoteborg": ("91862", "UEFA_2026_27_MATCH_FEED"),
    "sinttruiden": ("52883", "UEFA_2026_27_MATCH_FEED"),
}

QUALIFICATION_OVERRIDES = {
    "realbetis": {
        "route": "EPS",
        "entry_round": "CL-LS",
        "is_league_champion": False,
        "is_cup_winner": False,
        "source": "UEFA_EUROPEAN_PERFORMANCE_SPOT",
    },
    "liverpool": {
        "route": "EPS",
        "entry_round": "CL-LS",
        "is_league_champion": False,
        "is_cup_winner": False,
        "source": "UEFA_EUROPEAN_PERFORMANCE_SPOT",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build audited 2026/27 AO First Elo and current UEFA replay inputs"
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    cache_root = data_root / "_source_cache"
    data_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    source_paths = prepare_sources(cache_root, refresh=args.refresh, offline=args.offline)
    official = load_official_matches(cache_root, refresh=args.refresh, offline=args.offline)
    sources = parse_sources(source_paths)

    identity, teams = build_team_identity(sources["clubs_2027"], official)
    qualification = build_qualification_metadata(
        teams,
        sources["qualification_2026"],
        sources["clubs_2027"],
    )
    country = build_country_coefficients(teams, sources["country_rank_2026"])
    domestic, domestic_audit = build_domestic_context(teams, qualification, identity)
    club, club_audit = build_club_history(
        teams,
        identity,
        sources["club_rank_2026"],
        sources["historical_clubs"],
    )

    teams.to_csv(data_root / "teams.csv", index=False, lineterminator="\n")
    country.to_csv(
        data_root / "country_coefficients.csv", index=False, lineterminator="\n"
    )
    domestic.to_csv(
        data_root / "domestic_context.csv", index=False, lineterminator="\n"
    )
    club.to_csv(
        data_root / "club_european_points.csv", index=False, lineterminator="\n"
    )
    identity.to_csv(data_root / "team_identity_audit.csv", index=False, lineterminator="\n")
    domestic_audit.to_csv(
        data_root / "domestic_history_audit.csv", index=False, lineterminator="\n"
    )
    club_audit.to_csv(
        data_root / "club_history_audit.csv", index=False, lineterminator="\n"
    )

    finished, upcoming = build_dynamic_inputs(cache_root, teams)
    finished.to_csv(
        data_root / "matches_completed.csv", index=False, lineterminator="\n"
    )
    upcoming.to_csv(
        data_root / "fixtures_upcoming.csv", index=False, lineterminator="\n"
    )

    ratings_path = output_root / "ao_first_elo_2026_27.csv"
    ratings = compute_ao_first_elo_from_csv(
        data_root / "teams.csv",
        data_root / "country_coefficients.csv",
        data_root / "domestic_context.csv",
        data_root / "club_european_points.csv",
        AOEuropeanEloConfig.active(),
        ratings_path,
    )
    read_matches(data_root / "matches_completed.csv")
    read_fixtures(data_root / "fixtures_upcoming.csv")

    quality = build_quality_audit(
        teams,
        country,
        domestic,
        club,
        identity,
        domestic_audit,
        club_audit,
        finished,
        upcoming,
        ratings,
    )
    quality.to_csv(data_root / "data_quality_audit.csv", index=False, lineterminator="\n")
    failed = quality.loc[quality["status"].eq("FAIL")]

    manifest = build_manifest(
        data_root,
        output_root,
        source_paths,
        cache_root,
        teams,
        finished,
        upcoming,
        ratings,
        quality,
    )
    (data_root / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (data_root / "README.md").write_text(
        build_readme(teams, finished, upcoming, domestic_audit, ratings, quality),
        encoding="utf-8",
    )

    print("AO 2026/27 preproduction inputs")
    print(f"Teams: {len(teams)}")
    print(f"Completed matches: {len(finished)}")
    print(f"Upcoming fixtures: {len(upcoming)}")
    print(
        "Domestic Surprise complete histories: "
        f"{int(domestic_audit['history_complete'].sum())}/{len(domestic_audit)}"
    )
    print(f"AO First Elo: {ratings['ao_first_elo'].min():.3f} .. {ratings['ao_first_elo'].max():.3f}")
    print(f"Quality checks: {len(quality) - len(failed)}/{len(quality)} passed")
    print(f"Data: {data_root}")
    print(f"Ratings: {ratings_path}")
    if not failed.empty:
        raise ValueError(f"2026/27 preproduction audit failed: {failed.to_dict('records')}")


def prepare_sources(cache_root: Path, *, refresh: bool, offline: bool) -> dict[str, Path]:
    paths = {
        "ccoef2027": cache_root / "kassiesa" / "ccoef2027.html",
        "trank2026": cache_root / "kassiesa" / "trank2026.html",
        "crank2026": cache_root / "kassiesa" / "crank2026.html",
        "qual2026": cache_root / "kassiesa" / "qual2026.html",
    }
    urls = {
        "ccoef2027": f"{KASSIESA_BASE}/data/method5/ccoef2027.html",
        "trank2026": f"{KASSIESA_BASE}/data/method5/trank2026.html",
        "crank2026": f"{KASSIESA_BASE}/data/method5/crank2026.html",
        "qual2026": f"{KASSIESA_BASE}/qual2026.html",
    }
    for key, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not refresh:
            continue
        if offline:
            raise FileNotFoundError(f"Offline source cache is missing: {path}")
        fetch(urls[key], path, refresh=True)

    for year in range(2022, 2027):
        source = HISTORICAL_SOURCE_CACHE / f"ccoef{year}.html"
        target = cache_root / "kassiesa" / f"ccoef{year}.html"
        if not source.is_file():
            raise FileNotFoundError(f"Historical source is missing: {source}")
        if not target.exists():
            shutil.copyfile(source, target)
        paths[f"ccoef{year}"] = target
    return paths


def load_official_matches(
    cache_root: Path,
    *,
    refresh: bool,
    offline: bool,
) -> pd.DataFrame:
    uefa_cache = cache_root / "uefa"
    cached = sorted(uefa_cache.glob("*_2027_offset_*.json"))
    if offline and not cached:
        raise FileNotFoundError(f"Offline UEFA cache is missing: {uefa_cache}")
    combinations = pd.DataFrame(
        {"season": [SEASON] * 3, "competition": ["UCL", "UEL", "UECL"]}
    )
    return load_uefa_matches(
        combinations,
        uefa_cache,
        refresh=refresh and not offline,
    )


def parse_sources(paths: dict[str, Path]) -> dict[str, Any]:
    _, clubs_2027 = parse_ccoef(read_cached_text(paths["ccoef2027"]))
    club_rank_2026 = parse_trank(read_cached_text(paths["trank2026"]))
    country_rank_2026 = parse_crank(read_cached_text(paths["crank2026"]))
    qualification_2026 = parse_qualification(read_cached_text(paths["qual2026"]))
    historical_clubs = {
        year: parse_ccoef(read_cached_text(paths[f"ccoef{year}"]))[1]
        for year in range(2022, 2027)
    }
    if len(clubs_2027) != 237:
        raise ValueError(f"Expected 237 2026/27 participants, found {len(clubs_2027)}")
    return {
        "clubs_2027": clubs_2027,
        "club_rank_2026": club_rank_2026,
        "country_rank_2026": country_rank_2026,
        "qualification_2026": qualification_2026,
        "historical_clubs": historical_clubs,
    }


def build_team_identity(
    clubs: pd.DataFrame,
    official: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    catalog = uefa_team_catalog(official, SEASON)
    registry = pd.read_csv(CLUB_REGISTRY_PATH, dtype={"uefa_team_id": "string"})
    fixture_lookup: dict[str, set[str]] = {}
    for uefa_id, details in catalog.items():
        aliases = set(details["aliases"]) | {normalize_name(details["name"])}
        for alias in aliases:
            fixture_lookup.setdefault(alias, set()).add(str(uefa_id))
    registry_lookup: dict[str, set[str]] = {}
    for row in registry.itertuples(index=False):
        names = str(row.observed_names).split(" | ") + [str(row.canonical_name)]
        for name in names:
            registry_lookup.setdefault(normalize_name(name), set()).add(
                str(row.uefa_team_id)
            )

    country_codes = historical_country_code_map()
    rows: list[dict[str, object]] = []
    team_rows: list[dict[str, object]] = []
    for source in clubs.sort_values(["country", "team_name"]).itertuples(index=False):
        key = normalize_name(source.team_name)
        ids = fixture_lookup.get(key, set())
        method = "CURRENT_UEFA_FIXTURE_EXACT"
        source_reference = "UEFA_2026_27_MATCH_FEED"
        if not ids:
            ids = registry_lookup.get(key, set())
            method = "HISTORICAL_UEFA_REGISTRY_EXACT"
            source_reference = str(CLUB_REGISTRY_PATH)
        if not ids and key in UEFA_ID_OVERRIDES:
            uefa_id, source_reference = UEFA_ID_OVERRIDES[key]
            ids = {uefa_id}
            method = "VERIFIED_UEFA_ID_OVERRIDE"
        if len(ids) != 1:
            raise ValueError(f"Cannot resolve unique UEFA identity for {source.team_name}: {ids}")
        uefa_id = next(iter(ids))
        country_key = normalize_country(str(source.country))
        if country_key not in country_codes:
            raise ValueError(f"Missing AO country code for {source.country}")
        country_code = country_codes[country_key]
        club_id = permanent_club_id(uefa_id)
        official_name = catalog.get(uefa_id, {}).get("name", source.team_name)
        rows.append(
            {
                "season": SEASON,
                "team_id": club_id,
                "club_id": club_id,
                "uefa_team_id": uefa_id,
                "team_name": source.team_name,
                "uefa_team_name": official_name,
                "country": source.country,
                "country_code": country_code,
                "identity_method": method,
                "identity_source": source_reference,
                "identity_verified": True,
            }
        )
        team_rows.append(
            {
                "team_id": club_id,
                "team_name": source.team_name,
                "country": source.country,
                "country_code": country_code,
                "domestic_league": f"{source.country} top division",
            }
        )
    identity = pd.DataFrame(rows).sort_values(["country", "team_name"], kind="stable")
    teams = pd.DataFrame(team_rows).sort_values(["country", "team_name"], kind="stable")
    if identity["uefa_team_id"].duplicated().any() or teams["team_id"].duplicated().any():
        raise ValueError("2026/27 team identity is not one-to-one")
    return identity.reset_index(drop=True), teams.reset_index(drop=True)


def historical_country_code_map() -> dict[str, str]:
    frame = pd.read_csv(
        HISTORICAL_STATIC_ROOT / "2025-26" / "country_coefficients.csv"
    )
    return {
        normalize_country(country): str(code).upper()
        for country, code in zip(frame["country"], frame["country_code"])
    }


def build_qualification_metadata(
    teams: pd.DataFrame,
    qualification: pd.DataFrame,
    clubs: pd.DataFrame,
) -> pd.DataFrame:
    q = qualification.copy()
    q["team_key"] = q["team_name"].map(normalize_name)
    if q["team_key"].duplicated().any():
        raise ValueError("Qualification source contains duplicate team names")
    lookup = q.set_index("team_key")
    current_competition = {
        normalize_name(name): competition
        for name, competition in zip(clubs["team_name"], clubs["competition"])
    }
    rows = []
    for team in teams.itertuples(index=False):
        key = normalize_name(team.team_name)
        if key in lookup.index:
            row = lookup.loc[key]
            meta = {
                "route": str(row.route),
                "entry_round": str(row.entry_round),
                "is_league_champion": bool(row.is_league_champion),
                "is_cup_winner": bool(row.is_cup_winner),
                "source": "KASSIESA_QUALIFICATION_2026",
            }
        elif key in QUALIFICATION_OVERRIDES:
            meta = QUALIFICATION_OVERRIDES[key]
        else:
            raise ValueError(f"Missing entry metadata for {team.team_name}")
        rows.append(
            {
                "team_id": team.team_id,
                "team_name": team.team_name,
                **meta,
                "entry_competition": entry_competition(str(meta["entry_round"])),
                "source_current_competition": current_competition[key],
            }
        )
    return pd.DataFrame(rows)


def entry_competition(entry_round: str) -> str:
    prefix = entry_round.split("-", maxsplit=1)[0].upper()
    try:
        return {"CL": "UCL", "EL": "UEL", "CO": "UECL"}[prefix]
    except KeyError as error:
        raise ValueError(f"Unknown European entry round: {entry_round}") from error


def build_country_coefficients(
    teams: pd.DataFrame,
    country_rank: pd.DataFrame,
) -> pd.DataFrame:
    rank = country_rank.copy()
    rank["country_key"] = rank["country"].map(normalize_country)
    lookup = rank.drop_duplicates("country_key").set_index("country_key")
    rows = []
    for country, group in teams.groupby("country", sort=True):
        key = normalize_country(country)
        if key not in lookup.index:
            raise ValueError(f"Country ranking 2026 is missing {country}")
        source = lookup.loc[key]
        code_values = group["country_code"].unique()
        if len(code_values) != 1:
            raise ValueError(f"Country code is not unique for {country}")
        rows.append(
            {
                "season": SEASON,
                "country": country,
                "country_code": code_values[0],
                **{
                    f"points_{season_key}": float(source[f"p{i}"])
                    for i, season_key in enumerate(SEASON_KEYS)
                },
                "official_five_year_total": float(source["total"]),
                "official_country_rank": int(source["rank"]),
            }
        )
    return pd.DataFrame(rows).sort_values("country_code", kind="stable")


def aliases_for_team(
    team: object,
    identity: pd.DataFrame,
    registry: pd.DataFrame,
) -> tuple[str, ...]:
    names = [str(team.team_name)]
    identity_row = identity.loc[identity["team_id"].eq(team.team_id)].iloc[0]
    names.append(str(identity_row["uefa_team_name"]))
    historical = registry.loc[
        registry["uefa_team_id"].astype(str).eq(str(identity_row["uefa_team_id"]))
    ]
    if not historical.empty:
        names.extend(str(historical.iloc[0]["observed_names"]).split(" | "))
        names.append(str(historical.iloc[0]["canonical_name"]))
    return tuple(dict.fromkeys(name for name in names if name and name != "nan"))


def preferred_position(
    aliases: tuple[str, ...],
    lookup: dict[str, int],
) -> tuple[int | None, str]:
    for index, name in enumerate(aliases):
        position, method = resolve_position(name, lookup)
        if position is not None:
            prefix = "CURRENT_NAME" if index == 0 else "IDENTITY_ALIAS"
            return int(position), f"{prefix}_{method.upper()}"
    return None, "UNRESOLVED"


def build_domestic_context(
    teams: pd.DataFrame,
    qualification: pd.DataFrame,
    identity: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # The history window must end one season before whatever vintage the
    # current position comes from. Route-resolved teams carry the 2025/26
    # finish, but a cup-route team falls back to the cached 2024/25 table, and
    # comparing that season against a window containing itself would damp the
    # Domestic Surprise by exactly its own t_minus_1 weight. Loading one extra
    # year lets each team get a clean, self-exclusive five-season window.
    cache = load_standings_cache(
        HISTORICAL_STATIC_ROOT,
        tuple(range(2020, 2026)),
    )
    registry = pd.read_csv(CLUB_REGISTRY_PATH, dtype={"uefa_team_id": "string"})
    q = qualification.set_index("team_id")
    rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    route_position = re.compile(r"^N(\d+)$")
    for team in teams.itertuples(index=False):
        aliases = aliases_for_team(team, identity, registry)
        country_key = normalize_country(str(team.country))
        def season_row(year: int) -> tuple[int | None, int | None, str, str]:
            cached = cache.get((year, country_key))
            if cached is None:
                return (None, None, "NO_LEAGUE_TABLE", "")
            lookup, team_count, source_path = cached
            position, method = preferred_position(aliases, lookup)
            if position is not None and not 1 <= position <= team_count:
                return (None, None, "UNRESOLVED_OUT_OF_RANGE", str(source_path))
            return (
                position,
                team_count if position is not None else None,
                method,
                str(source_path),
            )

        meta = q.loc[team.team_id]
        latest_cached = cache.get((2025, country_key))
        latest_team_count = latest_cached[1] if latest_cached is not None else None
        latest_position, _, latest_method, latest_source = season_row(2025)
        route = str(meta.route)
        route_match = route_position.match(route)
        if bool(meta.is_league_champion) or route == "CH":
            current_position = 1 if latest_team_count is not None else None
            current_source = "QUALIFICATION_CHAMPION"
            current_vintage = CURRENT_SEASON_VINTAGE
        elif route_match and latest_team_count is not None:
            current_position = int(route_match.group(1))
            current_source = "QUALIFICATION_ROUTE"
            current_vintage = CURRENT_SEASON_VINTAGE
        else:
            # No route position is available, so the newest cached table is the
            # best evidence. It is one season older, and the history window
            # below shifts with it so the season never scores against itself.
            current_position = latest_position
            current_source = latest_method
            current_vintage = CACHE_FALLBACK_VINTAGE
        current_count = latest_team_count if current_position is not None else None
        history = [
            season_row(year)
            for year in range(current_vintage - 5, current_vintage)
        ]

        row: dict[str, object] = {
            "season": SEASON,
            "team_id": team.team_id,
            "domestic_position": current_position,
            "league_team_count": current_count,
            "is_league_champion": bool(meta.is_league_champion),
            "is_cup_winner": bool(meta.is_cup_winner),
            "european_entry_type": route,
            "competition": str(meta.entry_competition),
            "entry_round": str(meta.entry_round),
        }
        for offset, (position, count, _, _) in zip(range(5, 0, -1), history):
            row[f"history_position_t_minus_{offset}"] = position
            row[f"history_team_count_t_minus_{offset}"] = count
        rows.append(row)
        audit_rows.append(
            {
                "season": SEASON,
                "team_id": team.team_id,
                "team_name": team.team_name,
                "country_code": team.country_code,
                "route": route,
                "current_position": current_position,
                "current_team_count": current_count,
                "current_position_source": current_source,
                "current_position_vintage": current_vintage,
                "history_window": f"{current_vintage - 5}-{current_vintage - 1}",
                "current_source_cache": latest_source,
                "history_seasons_available": sum(item[0] is not None for item in history),
                "history_complete": all(item[0] is not None for item in history),
                **{
                    f"history_t_minus_{offset}_method": item[2]
                    for offset, item in zip(range(5, 0, -1), history)
                },
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(audit_rows)


def find_history_row(
    frame: pd.DataFrame,
    aliases: tuple[str, ...],
) -> pd.Series | None:
    keys = {history_team_key(name) for name in aliases}
    keyed = frame.assign(_history_key=frame["team_name"].map(history_team_key))
    candidates = keyed.loc[keyed["_history_key"].isin(keys)]
    if candidates.empty:
        return None
    unique_names = candidates["team_name"].drop_duplicates()
    if len(unique_names) > 1:
        raise ValueError(f"Ambiguous club history aliases {aliases}: {unique_names.tolist()}")
    return candidates.iloc[0]


def build_club_history(
    teams: pd.DataFrame,
    identity: pd.DataFrame,
    club_rank: pd.DataFrame,
    historical_clubs: dict[int, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    registry = pd.read_csv(CLUB_REGISTRY_PATH, dtype={"uefa_team_id": "string"})
    rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for team in teams.itertuples(index=False):
        aliases = aliases_for_team(team, identity, registry)
        source = find_history_row(club_rank, aliases)
        points = [float(source[f"p{i}"]) if source is not None else 0.0 for i in range(5)]
        row: dict[str, object] = {
            "season": SEASON,
            "team_id": team.team_id,
            "team_name_source": team.team_name,
            "country_code": team.country_code,
        }
        match_counts = []
        match_sources = []
        for index, (season_key, year) in enumerate(zip(SEASON_KEYS, range(2022, 2027))):
            historical = find_history_row(historical_clubs[year], aliases)
            matches = int(historical["matches"]) if historical is not None else 0
            competition = str(historical["competition"]) if historical is not None else ""
            if points[index] > 0.0 and matches == 0:
                raise ValueError(
                    f"{team.team_name} {year}: positive club points without match evidence"
                )
            row[f"club_points_{season_key}"] = points[index]
            row[f"played_{season_key}"] = int(matches > 0)
            row[f"matches_{season_key}"] = matches
            row[f"match_cap_{season_key}"] = match_cap(year, competition)
            match_counts.append(matches)
            match_sources.append(historical["team_name"] if historical is not None else "")
        row["official_club_coefficient"] = float(source["total"]) if source is not None else 0.0
        row["country_part"] = float(source["country_part"]) if source is not None else 0.0
        rows.append(row)
        audit_rows.append(
            {
                "season": SEASON,
                "team_id": team.team_id,
                "team_name": team.team_name,
                "history_row_found": source is not None,
                "history_source_name": source["team_name"] if source is not None else "",
                "five_year_points": sum(points),
                "five_year_matches": sum(match_counts),
                "explicit_all_zero_history": source is None and sum(match_counts) == 0,
                "match_source_names": " | ".join(str(value) for value in match_sources if value),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(audit_rows)


def build_dynamic_inputs(
    cache_root: Path,
    teams: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    known = set(teams["team_id"].astype(str))
    competition_by_file = {"UCL": "UCL", "UEL": "UEL", "UECL": "UECL"}
    finished_rows: list[dict[str, object]] = []
    upcoming_rows: list[dict[str, object]] = []
    for path in sorted((cache_root / "uefa").glob("*_2027_offset_*.json")):
        competition = competition_by_file[path.name.split("_", maxsplit=1)[0]]
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload:
            status = str(item.get("status") or "")
            if status not in {"FINISHED", "UPCOMING"}:
                continue
            home = item.get("homeTeam") or {}
            away = item.get("awayTeam") or {}
            home_id = permanent_club_id(str(home.get("id") or ""))
            away_id = permanent_club_id(str(away.get("id") or ""))
            if home_id not in known or away_id not in known:
                raise ValueError(f"UEFA match references unknown team: {item.get('id')}")
            round_data = item.get("round") or {}
            round_names = (round_data.get("translations") or {}).get("name") or {}
            source_round = str(round_names.get("EN") or (round_data.get("metaData") or {}).get("name") or "")
            round_name = ROUND_NAME_MAP.get(source_round, source_round)
            related_ids = [str(match.get("id")) for match in (item.get("relatedMatches") or []) if match.get("id")]
            tie_members = [str(item.get("id")), *related_ids]
            tie_id = f"UEFA-TIE-{min(tie_members, key=int)}"
            leg = item.get("leg") or {}
            leg_number = int(leg.get("number") or 1)
            mode_detail = str(round_data.get("modeDetail") or "")
            single_match = "ONE_LEG" in mode_detail or str(item.get("type") or "") == "SINGLE"
            is_decider = single_match or leg_number == 2
            base = {
                "match_id": f"UEFA-{item.get('id')}",
                "uefa_match_id": str(item.get("id")),
                "season": SEASON,
                "kickoff_utc": str((item.get("kickOffTime") or {}).get("dateTime") or ""),
                "competition": competition,
                "round": round_name,
                "source_round": source_round,
                "tie_id": tie_id,
                "leg_number": leg_number,
                "is_knockout": True,
                "is_tie_decider": is_decider,
                "is_single_match_tie": single_match,
                "stage": "QUALIFYING",
                "home_team_id": home_id,
                "away_team_id": away_id,
                "home_team_name": str(home.get("internationalName") or ""),
                "away_team_name": str(away.get("internationalName") or ""),
                "is_neutral": False,
            }
            if status == "UPCOMING":
                upcoming_rows.append(base)
                continue
            score = item.get("score") or {}
            total = score.get("total") or {}
            penalty = score.get("penalty") or {}
            home_goals = int(total.get("home"))
            away_goals = int(total.get("away"))
            decided_on_penalties = bool(penalty)
            advanced = ""
            winner = item.get("winner") or {}
            aggregate_winner = winner.get("aggregate") or {}
            winner_team = aggregate_winner.get("team") or {}
            if is_decider and winner_team.get("id"):
                advanced = permanent_club_id(str(winner_team["id"]))
            finished_rows.append(
                {
                    **base,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "actual_home_score": 1.0 if home_goals > away_goals else 0.0 if home_goals < away_goals else 0.5,
                    "goal_difference": abs(home_goals - away_goals),
                    "result_basis": "UEFA_FIELD_SCORE_90_OR_120_EXCLUDES_SHOOTOUT",
                    "decided_on_penalties": decided_on_penalties,
                    "home_penalty_goals": int(penalty["home"]) if penalty else "",
                    "away_penalty_goals": int(penalty["away"]) if penalty else "",
                    "advanced_team_id": advanced,
                    "xg_home": "",
                    "xg_away": "",
                    "xg_analysis_eligible": False,
                    "xg_fallback": "GOAL_MARGIN_ONLY",
                }
            )
    finished = stable_match_order(pd.DataFrame(finished_rows))
    upcoming = stable_match_order(pd.DataFrame(upcoming_rows))
    return finished, upcoming


def stable_match_order(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["_kickoff"] = pd.to_datetime(result["kickoff_utc"], utc=True, errors="raise")
    result = result.sort_values(["_kickoff", "match_id"], kind="stable").drop(columns="_kickoff")
    result.insert(0, "event_order", range(1, len(result) + 1))
    return result.reset_index(drop=True)


def build_quality_audit(
    teams: pd.DataFrame,
    country: pd.DataFrame,
    domestic: pd.DataFrame,
    club: pd.DataFrame,
    identity: pd.DataFrame,
    domestic_audit: pd.DataFrame,
    club_audit: pd.DataFrame,
    finished: pd.DataFrame,
    upcoming: pd.DataFrame,
    ratings: pd.DataFrame,
) -> pd.DataFrame:
    checks = [
        check("participant_count", len(teams) == 237, len(teams), 237),
        check("team_id_unique", teams["team_id"].is_unique, teams["team_id"].nunique(), 237),
        check("uefa_identity_complete", identity["identity_verified"].all(), int(identity["identity_verified"].sum()), 237),
        check("uefa_identity_unique", identity["uefa_team_id"].is_unique, identity["uefa_team_id"].nunique(), 237),
        check("country_rows_complete", len(country) == teams["country_code"].nunique(), len(country), teams["country_code"].nunique()),
        check("domestic_rows_complete", len(domestic) == len(teams), len(domestic), len(teams)),
        check("club_history_rows_complete", len(club) == len(teams), len(club), len(teams)),
        check("explicit_zero_history_rows", int(club_audit["explicit_all_zero_history"].sum()) > 0, int(club_audit["explicit_all_zero_history"].sum()), ">0"),
        check("domestic_history_coverage_audited", domestic_audit["history_seasons_available"].between(0, 5).all(), len(domestic_audit), len(teams)),
        check("completed_match_count", len(finished) == 342, len(finished), 342),
        check("upcoming_fixture_count", len(upcoming) == 86, len(upcoming), 86),
        check("completed_match_id_unique", finished["match_id"].is_unique, finished["match_id"].nunique(), len(finished)),
        check("upcoming_match_id_unique", upcoming["match_id"].is_unique, upcoming["match_id"].nunique(), len(upcoming)),
        check("match_team_identity_coverage", set(finished["home_team_id"]) | set(finished["away_team_id"]) <= set(teams["team_id"]), len(set(finished["home_team_id"]) | set(finished["away_team_id"])), "subset"),
        check("rating_rows_complete", len(ratings) == len(teams), len(ratings), len(teams)),
        check("rating_finite", ratings["ao_first_elo"].notna().all(), int(ratings["ao_first_elo"].notna().sum()), len(ratings)),
        check("validation_warnings_zero", ratings["validation_warnings"].fillna("").eq("").all(), int(ratings["validation_warnings"].fillna("").ne("").sum()), 0),
    ]
    return pd.DataFrame(checks)


def check(name: str, passed: bool, actual: object, expected: object) -> dict[str, object]:
    return {
        "check": name,
        "status": "PASS" if passed else "FAIL",
        "actual": actual,
        "expected": expected,
    }


def build_manifest(
    data_root: Path,
    output_root: Path,
    source_paths: dict[str, Path],
    cache_root: Path,
    teams: pd.DataFrame,
    finished: pd.DataFrame,
    upcoming: pd.DataFrame,
    ratings: pd.DataFrame,
    quality: pd.DataFrame,
) -> dict[str, object]:
    source_files = sorted(set(source_paths.values()) | set((cache_root / "uefa").glob("*.json")))
    latest_finished = pd.to_datetime(finished["kickoff_utc"], utc=True).max()
    return {
        "dataset_version": "ao-2026-27-preproduction-inputs-v1",
        "season": SEASON,
        "as_of_last_finished_match_utc": latest_finished.isoformat(),
        "production_contract_changed": False,
        "teams": len(teams),
        "completed_matches": len(finished),
        "upcoming_fixtures": len(upcoming),
        "rating_min": float(ratings["ao_first_elo"].min()),
        "rating_max": float(ratings["ao_first_elo"].max()),
        "all_quality_checks_passed": bool(quality["status"].eq("PASS").all()),
        "data_root": str(data_root),
        "output_root": str(output_root),
        "sources": [
            {
                "path": str(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in source_files
        ],
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_readme(
    teams: pd.DataFrame,
    finished: pd.DataFrame,
    upcoming: pd.DataFrame,
    domestic_audit: pd.DataFrame,
    ratings: pd.DataFrame,
    quality: pd.DataFrame,
) -> str:
    complete_history = int(domestic_audit["history_complete"].sum())
    applied = int(ratings["domestic_surprise_status"].eq("APPLIED").sum())
    return f"""# 2026/27 AO Preproduction Input Snapshot

Bu klasor production contract'ini degistirmeyen, 2026/27 sezonuna ait tarihli
bir preproduction veri snapshot'idir.

- Katilimci: {len(teams)}
- Tamamlanmis UEFA maci: {len(finished)}
- Yaklasan play-off fiksturu: {len(upcoming)}
- Bes sezon domestic history tam: {complete_history}/{len(teams)}
- Domestic Surprise uygulanan takim: {applied}
- AO First Elo araligi: {ratings['ao_first_elo'].min():.3f} - {ratings['ao_first_elo'].max():.3f}
- Kalite kontrolleri: {int(quality['status'].eq('PASS').sum())}/{len(quality)} PASS

Ana inputlar:

- `teams.csv`
- `country_coefficients.csv`
- `domestic_context.csv`
- `club_european_points.csv`

Replay girdileri:

- `matches_completed.csv`
- `fixtures_upcoming.csv`

Eksik domestic history tahmin edilmez. Bes tam sezonu olmayan takimda aktif
pipeline Domestic Surprise'i `INSUFFICIENT_HISTORY` olarak sifirlar. Iki tarafli
ve zaman kapsami dogrulanmis xG henuz bulunmadigi icin tamamlanmis 2026/27
maclari `GOAL_MARGIN_ONLY` fallback'iyle isaretlenmistir.
"""


if __name__ == "__main__":
    main()
