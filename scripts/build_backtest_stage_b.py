from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import hashlib
import io
import re
import shutil
import ssl
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import certifi
import pandas as pd
from lxml import html as lxml_html

try:
    from build_backtest_dataset import normalize_name, read_cached_text
except ModuleNotFoundError:
    from scripts.build_backtest_dataset import normalize_name, read_cached_text


ROOT = Path(__file__).resolve().parents[1]
STAGE_A_ROOT = ROOT / "data" / "backtest_2021_2026"
OUTPUT_ROOT = ROOT / "data" / "backtest_stage_b_2021_2026"
CACHE_DIR = OUTPUT_ROOT / "_source_cache"
ROUTE_POSITION_RE = re.compile(r"^(?:N|CL)(\d+)$")
COUNTRY_ALIASES = {
    "czechia": "czechrepublic",
    "macedonia": "northmacedonia",
    "turkiye": "turkey",
}
NO_DOMESTIC_LEAGUE = {"liechtenstein"}
CLUB_STANDINGS_ALIASES = {
    "agfaarhus": "agf",
    "aikstockholm": "aik",
    "azalkmaar": "az",
    "b36torshavn": "b36",
    "cardiffmu": "cardiffmetropolitanuniversity",
    "dinamotirana": "dinamocity",
    "fckbenhavn": "copenhagen",
    "fcilevadia": "levadiatallinn",
    "fcsanktgallen": "stgallen",
    "fcufa": "ufa",
    "fcbmagpies": "brunosmagpies",
    "fhhafnarfjardar": "fh",
    "fktrakai": "riteriai",
    "fkriteriai": "riteriai",
    "hbtorshavn": "hb",
    "havnarboltfelag": "hb",
    "kaakureyri": "ka",
    "kaljunomme": "nommekalju",
    "krreykjavik": "kr",
    "levadia": "levadiatallinn",
    "nsirunavik": "nsi",
    "ofiheraklion": "ofi",
    "olympiakospiraeus": "olympiacos",
    "staderennais": "rennes",
    "rabotnickiskopje": "rabotnicki",
    "transnarva": "narvatrans",
    "trakai": "riteriai",
    "tscbackatopola": "tsc",
    "urartufc": "banants",
    "vardarskopje": "vardar",
    "zeljeznicarsarajevo": "zeljeznicar",
}


def main() -> None:
    global STAGE_A_ROOT, OUTPUT_ROOT, CACHE_DIR
    parser = argparse.ArgumentParser(description="Enrich AO Elo backtest with domestic standings")
    parser.add_argument("--refresh", action="store_true", help="Redownload league pages")
    parser.add_argument("--stage-a-root", type=Path, default=STAGE_A_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    STAGE_A_ROOT = args.stage_a_root.resolve()
    OUTPUT_ROOT = args.output_root.resolve()
    CACHE_DIR = OUTPUT_ROOT / "_source_cache"

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    audits: list[dict[str, object]] = []
    for source_folder in sorted(STAGE_A_ROOT.glob("20??-??")):
        audits.extend(build_season(source_folder, refresh=args.refresh))

    audit = pd.DataFrame(audits)
    audit.to_csv(OUTPUT_ROOT / "build_audit.csv", index=False)
    write_readme(audit)
    failed = audit.loc[audit["status"] == "FAIL"]
    print(f"Stage B data written: {OUTPUT_ROOT}")
    print(audit.groupby(["check", "status"]).size().to_string())
    if not failed.empty:
        raise ValueError(f"Stage B audit failed with {len(failed)} issue(s)")


def build_season(source_folder: Path, refresh: bool) -> list[dict[str, object]]:
    season = source_folder.name.replace("-", "/")
    start_year = int(source_folder.name[:4])
    qual_path = STAGE_A_ROOT / "_source_cache" / f"qual{start_year}.html"
    league_urls = parse_league_urls(read_cached_text(qual_path))

    target = OUTPUT_ROOT / source_folder.name
    target.mkdir(parents=True, exist_ok=True)
    for filename in ("teams.csv", "country_coefficients.csv", "club_european_points.csv", "matches.csv"):
        shutil.copyfile(source_folder / filename, target / filename)

    teams = pd.read_csv(source_folder / "teams.csv")
    domestic = pd.read_csv(source_folder / "domestic_context.csv")
    data = teams.merge(domestic, on="team_id", validate="one_to_one")
    data["country_key"] = data["country"].map(normalize_country)

    audits: list[dict[str, object]] = []
    enriched_rows: list[dict[str, object]] = []
    for country, group in data.groupby("country", sort=True):
        country_key = normalize_country(country)
        if country_key in NO_DOMESTIC_LEAGUE:
            for row in group.itertuples(index=False):
                enriched_rows.append(enriched_row(row, None, None, "no_domestic_league"))
            audits.append(audit_row(season, country, "league_table_read", True, "not_applicable"))
            audits.append(
                audit_row(
                    season,
                    country,
                    "participant_position_coverage",
                    True,
                    f"not_applicable={len(group)}",
                )
            )
            continue
        url = league_urls.get(country_key)
        if not url:
            audits.append(audit_row(season, country, "league_url_found", False, "missing"))
            continue
        cache_path = CACHE_DIR / f"{start_year}_{country_key}_{hashlib.sha1(url.encode()).hexdigest()[:8]}.html"
        try:
            fetch(url, cache_path, refresh)
            standings = read_standings(cache_path)
        except Exception as exc:
            optional = bool(group["is_cup_winner"].all())
            audits.append(
                audit_row(
                    season,
                    country,
                    "league_table_read",
                    optional,
                    f"{type(exc).__name__}; optional_cup_only={optional}",
                )
            )
            if optional:
                for row in group.itertuples(index=False):
                    enriched_rows.append(enriched_row(row, None, None, "unresolved_cup_only"))
            continue

        league_team_count = max((len(table) for table in standings), default=0)
        plausible = 4 <= league_team_count <= 30
        audits.append(
            audit_row(season, country, "league_team_count_plausible", plausible, league_team_count)
        )
        if not plausible:
            continue

        position_lookup = build_position_lookup(standings)
        country_positions = 0
        for row in group.itertuples(index=False):
            route = str(row.european_entry_type)
            route_match = ROUTE_POSITION_RE.match(route)
            position: int | None = None
            source = ""
            if bool(row.is_league_champion) or route == "CH":
                position = 1
                source = "qualification_champion"
            elif route_match:
                position = int(route_match.group(1))
                source = "qualification_route"
            else:
                position, source = resolve_position(row.team_name, position_lookup)

            if position is not None and not 1 <= position <= league_team_count:
                position = None
                source = ""
            if position is not None:
                country_positions += 1
            enriched_rows.append(enriched_row(row, position, league_team_count, source))
        required = group.loc[~group["is_cup_winner"].astype(bool)]
        resolved_required = sum(
            1
            for row in enriched_rows[-len(group):]
            if row["team_id"] in set(required["team_id"]) and row["domestic_position"] is not None
        )
        audits.append(
            audit_row(
                season,
                country,
                "participant_position_coverage",
                resolved_required == len(required),
                f"all={country_positions}/{len(group)}; required={resolved_required}/{len(required)}",
            )
        )

    enriched = pd.DataFrame(enriched_rows).sort_values("team_id")
    if len(enriched) != len(domestic):
        missing = sorted(set(domestic["team_id"]) - set(enriched["team_id"]))
        audits.append(audit_row(season, "ALL", "all_teams_enriched", False, missing[:10]))
    else:
        audits.append(audit_row(season, "ALL", "all_teams_enriched", True, len(enriched)))

    model_columns = list(domestic.columns)
    enriched[model_columns].to_csv(target / "domestic_context.csv", index=False)
    enriched.to_csv(target / "domestic_context_audit.csv", index=False)
    return audits


def parse_league_urls(html: str) -> dict[str, str]:
    document = lxml_html.fromstring(html)
    result: dict[str, str] = {}
    associations = document.xpath(
        '//div[contains(concat(" ", normalize-space(@class), " "), " assoc ")]'
    )
    for association in associations:
        pre_text = "".join(association.xpath(".//pre//text()"))
        country = parse_country(pre_text)
        links = association.xpath('.//a[contains(@href,"wikipedia.org/wiki/")]/@href')
        if country and links:
            result[normalize_country(country)] = links[0]
    legacy_associations = document.xpath(
        '//div[contains(concat(" ", normalize-space(@class), " "), " yellow ")]'
    )
    for association in legacy_associations:
        country = parse_country("".join(association.itertext()))
        league_url = ""
        for sibling in association.itersiblings():
            if sibling.tag == "div":
                break
            href = sibling.get("href", "") if sibling.tag == "a" else ""
            if "wikipedia.org/wiki/" in href:
                league_url = href
                break
        if country and league_url:
            result[normalize_country(country)] = league_url

    # Malformed anchors occur between association blocks on some old pages.
    # Recover the local yellow-block/league-link pair without DOM siblings.
    legacy_pattern = re.compile(
        r'<div\s+class=["\']yellow["\'][^>]*>\s*(?P<body>.*?)</div>\s*'
        r'\(league:\s*<a\s+href=["\'](?P<href>[^"\']+)',
        flags=re.IGNORECASE | re.DOTALL,
    )
    for match in legacy_pattern.finditer(html):
        body_text = " ".join(lxml_html.fromstring(match.group("body")).itertext())
        country = parse_country(body_text)
        if country:
            result.setdefault(normalize_country(country), match.group("href"))
    return result


def parse_country(pre_text: str) -> str:
    lines = [line for line in pre_text.splitlines() if line.strip()]
    if not lines:
        return ""
    match = re.match(r"\s*\d+\s+(.+?)\s+\d+\.\d+\s*$", lines[0])
    return match.group(1).strip() if match else ""


def fetch(url: str, path: Path, refresh: bool) -> None:
    if path.exists() and not refresh:
        return
    safe_url = quote(url, safe=":/?=&%")
    request = Request(safe_url, headers={"User-Agent": "AO-European-Elo-Backtest/1.0"})
    context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(request, timeout=45, context=context) as response:
        path.write_bytes(response.read())


def read_standings(path: Path) -> list[pd.DataFrame]:
    candidates: list[pd.DataFrame] = []
    source = path.read_text(encoding="utf-8")
    source = re.sub(
        r"(colspan|rowspan)\s*=\s*(['\"])(.*?)\2",
        sanitize_span,
        source,
        flags=re.DOTALL,
    )
    for table in pd.read_html(io.StringIO(source)):
        frame = table.copy()
        frame.columns = [flatten_column(column) for column in frame.columns]
        position_column = next(
            (c for c in frame.columns if c.startswith("Pos") or c.startswith("Rank")),
            None,
        )
        team_column = next(
            (c for c in frame.columns if c.startswith("Team") or c.startswith("Club")),
            None,
        )
        if not position_column or not team_column:
            continue
        positions = pd.to_numeric(
            frame[position_column].astype(str).str.extract(r"(\d+)", expand=False),
            errors="coerce",
        )
        cleaned = pd.DataFrame(
            {
                "position": positions,
                "team_name": frame[team_column].map(clean_team_name),
            }
        ).dropna(subset=["position", "team_name"])
        cleaned = cleaned.loc[cleaned["team_name"].ne("")].copy()
        cleaned["position"] = cleaned["position"].astype(int)
        cleaned["team_key"] = cleaned["team_name"].map(standings_team_key)
        cleaned = cleaned.drop_duplicates("team_key")
        if len(cleaned) >= 4:
            candidates.append(cleaned)
    if not candidates:
        raise ValueError(f"No standings table found in {path}")
    return candidates


def flatten_column(column: object) -> str:
    if isinstance(column, tuple):
        values = [str(value) for value in column if not str(value).startswith("Unnamed")]
        return " ".join(dict.fromkeys(values))
    return str(column)


def clean_team_name(value: object) -> str:
    text = str(value)
    text = re.sub(r"\[[^]]*]", "", text)
    text = re.sub(r"\s+\([^)]{1,5}\)\s*$", "", text)
    return " ".join(text.split()).strip()


def build_position_lookup(tables: list[pd.DataFrame]) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for table in sorted(tables, key=len, reverse=True):
        for row in table.itertuples(index=False):
            lookup.setdefault(row.team_key, int(row.position))
    return lookup


def resolve_position(team_name: str, lookup: dict[str, int]) -> tuple[int | None, str]:
    key = standings_team_key(team_name)
    if key in lookup:
        return lookup[key], "league_table_exact"

    substring_matches = [
        candidate
        for candidate in lookup
        if min(len(key), len(candidate)) >= 4 and (key in candidate or candidate in key)
    ]
    if len(substring_matches) == 1:
        return lookup[substring_matches[0]], "league_table_substring"

    scores = sorted(
        ((SequenceMatcher(None, key, candidate).ratio(), candidate) for candidate in lookup),
        reverse=True,
    )
    if scores and scores[0][0] >= 0.78 and (len(scores) == 1 or scores[0][0] - scores[1][0] >= 0.05):
        return lookup[scores[0][1]], "league_table_fuzzy"
    return None, "unresolved"


def standings_team_key(team_name: str) -> str:
    key = normalize_name(team_name)
    return CLUB_STANDINGS_ALIASES.get(key, key)


def enriched_row(
    row: object,
    position: int | None,
    league_team_count: int | None,
    source: str,
) -> dict[str, object]:
    return {
        "season": row.season,
        "team_id": row.team_id,
        "domestic_position": position,
        "league_team_count": league_team_count if position is not None else pd.NA,
        "is_league_champion": bool(row.is_league_champion),
        "is_cup_winner": bool(row.is_cup_winner),
        "european_entry_type": row.european_entry_type,
        "competition": row.competition,
        "entry_round": row.entry_round,
        "domestic_position_source": source,
    }


def sanitize_span(match: re.Match[str]) -> str:
    number = re.search(r"\d+", match.group(3))
    return f'{match.group(1)}="{number.group(0) if number else "1"}"'


def normalize_country(value: str) -> str:
    key = normalize_name(value)
    return COUNTRY_ALIASES.get(key, key)


def audit_row(
    season: str,
    country: str,
    check: str,
    passed: bool,
    detail: object,
) -> dict[str, object]:
    return {
        "season": season,
        "country": country,
        "check": check,
        "status": "PASS" if passed else "FAIL",
        "detail": detail,
    }


def write_readme(audit: pd.DataFrame) -> None:
    coverage = audit.loc[audit["check"] == "participant_position_coverage", "detail"]
    text = "\n".join(
        [
            "# AO European Elo Backtest Stage B",
            "",
            "Stage B enriches the Stage A UEFA snapshots with domestic position and league size.",
            "Qualification routes provide CH/N positions; linked historical league tables provide",
            "league team counts and fallback positions for cup/titleholder entries.",
            "",
            "Every country-season must pass URL, table, plausible team-count, and participant",
            "position coverage checks before this dataset can be used for domestic calibration.",
            "",
            f"Coverage audit rows: {len(coverage)}.",
        ]
    )
    (OUTPUT_ROOT / "README.md").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
