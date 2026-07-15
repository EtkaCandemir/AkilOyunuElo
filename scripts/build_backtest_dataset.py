from __future__ import annotations

import argparse
import re
import ssl
import unicodedata
from collections import Counter
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
import certifi


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "data" / "backtest_2021_2026"
CACHE_DIR = OUTPUT_ROOT / "_source_cache"
BASE_URL = "https://kassiesa.net/uefa"
TARGET_END_YEARS = range(2022, 2027)
CLUB_HISTORY_ALIASES = {
    "mladostpodgorica": "ofktitograd",
    "fktrakai": "fkriteriai",
}
SEASON_KEYS = ("t_minus_4", "t_minus_3", "t_minus_2", "t_minus_1", "t")
SCORE_RE = re.compile(r"^(\d+)\s*-\s*(\d+)$")
TEAM_ALIASES = {
    "banantsyerevan": "urartufc",
    "lasklinz": "lask",
    "interbaku": "keslafk",
    "videotonfehervar": "fcfehervar",
    "steauabucuresti": "fcsb",
    "viitorulconstanta": "farulconstanta",
    "dinamokiev": "dynamokyiv",
}
QUAL_LINE_RE = re.compile(
    r"^\s*(?P<route>CL-TH|EL-TH|CO-TH|CH|CW|CL\d*|EL\d+|N\d+)="
    r"(?P<team>.+?)\s+\d+(?:\.\d+)?\s+\((?P<entry>[^)]+)\)\s*$"
)


@dataclass(frozen=True)
class HtmlRow:
    row_class: str
    cells: tuple[str, ...]
    cup: str
    round_name: str


class KassiesaParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[HtmlRow] = []
        self.pre_blocks: list[str] = []
        self._table_depth = 0
        self._row_class = ""
        self._cells: list[str] | None = None
        self._cell_parts: list[str] | None = None
        self._capture_div = ""
        self._div_parts: list[str] = []
        self._cup = ""
        self._round = ""
        self._pre_parts: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = dict(attrs)
        classes = attr.get("class", "") or ""
        if tag == "table":
            self._table_depth += 1
        elif tag == "tr" and self._table_depth:
            self._row_class = classes
            self._cells = []
        elif tag in {"td", "th"} and self._cells is not None:
            self._cell_parts = []
        elif tag == "div" and classes in {"cupheader", "roundheader"}:
            self._capture_div = classes
            self._div_parts = []
        elif tag == "pre":
            self._pre_parts = []
        elif tag == "br" and self._cell_parts is not None:
            self._cell_parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag == "table":
            self._table_depth = max(0, self._table_depth - 1)
        elif tag in {"td", "th"} and self._cell_parts is not None:
            assert self._cells is not None
            self._cells.append(clean_text("".join(self._cell_parts)))
            self._cell_parts = None
        elif tag == "tr" and self._cells is not None:
            self.rows.append(
                HtmlRow(self._row_class, tuple(self._cells), self._cup, self._round)
            )
            self._cells = None
            self._row_class = ""
        elif tag == "div" and self._capture_div:
            value = clean_text("".join(self._div_parts))
            if self._capture_div == "cupheader":
                self._cup = value
                self._round = ""
            else:
                self._round = value
            self._capture_div = ""
            self._div_parts = []
        elif tag == "pre" and self._pre_parts is not None:
            self.pre_blocks.append("".join(self._pre_parts))
            self._pre_parts = None

    def handle_data(self, data: str) -> None:
        if self._cell_parts is not None:
            self._cell_parts.append(data)
        if self._capture_div:
            self._div_parts.append(data)
        if self._pre_parts is not None:
            self._pre_parts.append(data)


def main() -> None:
    global OUTPUT_ROOT, CACHE_DIR, TARGET_END_YEARS
    parser = argparse.ArgumentParser(description="Build AO Elo historical backtest data")
    parser.add_argument("--refresh", action="store_true", help="Redownload source HTML")
    parser.add_argument("--start-end-year", type=int, default=min(TARGET_END_YEARS))
    parser.add_argument("--last-end-year", type=int, default=max(TARGET_END_YEARS))
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    if args.start_end_year > args.last_end_year:
        raise ValueError("start-end-year must be <= last-end-year")
    TARGET_END_YEARS = range(args.start_end_year, args.last_end_year + 1)
    OUTPUT_ROOT = args.output_root.resolve()
    CACHE_DIR = OUTPUT_ROOT / "_source_cache"

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    parsed = load_sources(refresh=args.refresh)
    audits: list[dict[str, object]] = []
    for end_year in TARGET_END_YEARS:
        audits.extend(build_target_season(end_year, parsed))

    audit = pd.DataFrame(audits)
    audit.to_csv(OUTPUT_ROOT / "build_audit.csv", index=False)
    write_readme(audit)
    failed = audit.loc[audit["status"] == "FAIL"]
    print(f"Built {len(TARGET_END_YEARS)} target seasons in {OUTPUT_ROOT}")
    print(audit.groupby(["check", "status"]).size().to_string())
    if not failed.empty:
        raise ValueError(f"Backtest build audit failed with {len(failed)} issue(s)")


def load_sources(refresh: bool) -> dict[str, dict[int, object]]:
    needed_ccoef = range(min(TARGET_END_YEARS) - 5, max(TARGET_END_YEARS) + 1)
    result: dict[str, dict[int, object]] = {
        "ccoef": {},
        "trank": {},
        "crank": {},
        "match": {},
        "qual": {},
    }
    for year in needed_ccoef:
        method = "method4" if year <= 2017 else "method5"
        html = fetch(
            f"{BASE_URL}/data/{method}/ccoef{year}.html",
            CACHE_DIR / f"ccoef{year}.html",
            refresh,
        )
        result["ccoef"][year] = parse_ccoef(html)

    for year in TARGET_END_YEARS:
        prior = year - 1
        for kind in ("trank", "crank"):
            html = fetch(
                f"{BASE_URL}/data/method5/{kind}{prior}.html",
                CACHE_DIR / f"{kind}{prior}.html",
                refresh,
            )
            result[kind][prior] = (
                parse_trank(html) if kind == "trank" else parse_crank(html)
            )
        match_html = fetch(
            f"{BASE_URL}/data/method5/match{year}.html",
            CACHE_DIR / f"match{year}.html",
            refresh,
        )
        result["match"][year] = parse_matches(match_html)
        qual_html = fetch(
            f"{BASE_URL}/history/qual{prior}.html",
            CACHE_DIR / f"qual{prior}.html",
            refresh,
        )
        result["qual"][prior] = parse_qualification(qual_html)
    return result


def build_target_season(
    end_year: int,
    parsed: dict[str, dict[int, object]],
) -> list[dict[str, object]]:
    season = season_label(end_year)
    folder = OUTPUT_ROOT / season.replace("/", "-")
    folder.mkdir(parents=True, exist_ok=True)

    target_clubs = parsed["ccoef"][end_year][1].copy()
    target_clubs = target_clubs.loc[
        target_clubs["competition"].isin({"UCL", "UEL", "UECL"})
        & target_clubs["matches"].gt(0)
    ].copy()
    matches = parsed["match"][end_year].copy()
    history = parsed["trank"][end_year - 1].copy()
    countries = parsed["crank"][end_year - 1].copy()
    qualification = parsed["qual"][end_year - 1].copy()

    target_clubs["team_key"] = target_clubs["team_name"].map(normalize_name)
    target_clubs["history_key"] = target_clubs["team_name"].map(history_team_key)
    matches["home_key"] = matches["home_team_name"].map(normalize_name)
    matches["away_key"] = matches["away_team_name"].map(normalize_name)
    history["team_key"] = history["team_name"].map(normalize_name)
    history["history_key"] = history["team_name"].map(history_team_key)
    qualification["team_key"] = qualification["team_name"].map(normalize_name)

    match_codes: dict[str, str] = {}
    for side in ("home", "away"):
        for key, code in zip(matches[f"{side}_key"], matches[f"{side}_country_code"]):
            match_codes.setdefault(key, str(code).upper())
    target_clubs["country_code"] = target_clubs["team_key"].map(match_codes)
    if target_clubs["country_code"].isna().any():
        missing = target_clubs.loc[target_clubs["country_code"].isna(), "team_name"].tolist()
        raise ValueError(f"{season}: missing match country code for {missing}")

    target_clubs = target_clubs.sort_values(["country", "team_name"]).reset_index(drop=True)
    target_clubs["team_id"] = range(1, len(target_clubs) + 1)
    id_by_key = dict(zip(target_clubs["team_key"], target_clubs["team_id"]))

    teams = target_clubs[["team_id", "team_name", "country", "country_code"]].copy()
    teams["domestic_league"] = "Historical domestic league"
    teams.to_csv(folder / "teams.csv", index=False)

    country_code_by_country = (
        target_clubs.groupby("country")["country_code"]
        .agg(lambda values: Counter(values).most_common(1)[0][0])
        .to_dict()
    )
    countries["country_key"] = countries["country"].map(normalize_name)
    country_lookup = countries.set_index("country_key")
    country_rows = []
    for country, code in sorted(country_code_by_country.items()):
        key = normalize_name(country)
        if key not in country_lookup.index:
            raise ValueError(f"{season}: missing prior country history for {country}")
        source = country_lookup.loc[key]
        country_rows.append(
            {
                "season": season,
                "country": country,
                "country_code": code,
                **{f"points_{key_name}": source[f"p{i}"] for i, key_name in enumerate(SEASON_KEYS)},
                "official_five_year_total": source["total"],
                "official_country_rank": source["rank"],
            }
        )
    pd.DataFrame(country_rows).to_csv(folder / "country_coefficients.csv", index=False)

    qual_lookup = qualification.drop_duplicates("team_key").set_index("team_key")
    domestic_rows = []
    for row in target_clubs.itertuples(index=False):
        route = ""
        entry = row.competition
        champion = False
        cup_winner = False
        if row.team_key in qual_lookup.index:
            q = qual_lookup.loc[row.team_key]
            route = str(q["route"])
            entry = str(q["entry_round"])
            champion = bool(q["is_league_champion"])
            cup_winner = bool(q["is_cup_winner"])
        domestic_rows.append(
            {
                "season": season,
                "team_id": row.team_id,
                "domestic_position": pd.NA,
                "league_team_count": pd.NA,
                "is_league_champion": champion,
                "is_cup_winner": cup_winner,
                "european_entry_type": route or "Historical participant",
                "competition": row.competition,
                "entry_round": entry,
            }
        )
    pd.DataFrame(domestic_rows).to_csv(folder / "domestic_context.csv", index=False)

    history_lookup = history.drop_duplicates("history_key").set_index("history_key")
    historical_ccoef = {
        year: frame[1].assign(
            history_key=lambda x: x["team_name"].map(history_team_key)
        )
        for year, frame in parsed["ccoef"].items()
        if end_year - 5 <= year < end_year
    }
    club_rows = []
    history_match_mismatches = 0
    for row in target_clubs.itertuples(index=False):
        source = (
            history_lookup.loc[row.history_key]
            if row.history_key in history_lookup.index
            else None
        )
        points = [float(source[f"p{i}"]) if source is not None else 0.0 for i in range(5)]
        match_counts: list[int] = []
        caps: list[int] = []
        for i, historical_year in enumerate(range(end_year - 5, end_year)):
            frame = historical_ccoef[historical_year]
            candidate = frame.loc[frame["history_key"] == row.history_key]
            matches_played = int(candidate.iloc[0]["matches"]) if len(candidate) == 1 else 0
            historical_competition = str(candidate.iloc[0]["competition"]) if len(candidate) == 1 else ""
            match_counts.append(matches_played)
            caps.append(match_cap(historical_year, historical_competition))
            if points[i] > 0 and matches_played == 0:
                history_match_mismatches += 1
        club_row: dict[str, object] = {
            "season": season,
            "team_id": row.team_id,
            "team_name_source": row.team_name,
            "country_code": row.country_code,
        }
        for i, key_name in enumerate(SEASON_KEYS):
            club_row[f"club_points_{key_name}"] = points[i]
            club_row[f"played_{key_name}"] = int(match_counts[i] > 0)
            club_row[f"matches_{key_name}"] = match_counts[i]
            club_row[f"match_cap_{key_name}"] = caps[i]
        club_row["official_club_coefficient"] = float(source["total"]) if source is not None else 0.0
        club_row["country_part"] = float(source["country_part"]) if source is not None else 0.0
        club_rows.append(club_row)
    pd.DataFrame(club_rows).to_csv(folder / "club_european_points.csv", index=False)

    matches["home_team_id"] = matches["home_key"].map(id_by_key)
    matches["away_team_id"] = matches["away_key"].map(id_by_key)
    matched = matches.dropna(subset=["home_team_id", "away_team_id"]).copy()
    matched["home_team_id"] = matched["home_team_id"].astype(int)
    matched["away_team_id"] = matched["away_team_id"].astype(int)
    matched.insert(0, "season", season)
    matched[
        [
            "season", "competition", "round", "home_team_id", "away_team_id",
            "home_team_name", "away_team_name", "home_goals", "away_goals",
        ]
    ].to_csv(folder / "matches.csv", index=False)

    target_source_matches = int(target_clubs["matches"].sum())
    parsed_team_appearances = len(matched) * 2
    audits = [
        audit_row(season, "all_target_clubs_have_match_codes", target_clubs["country_code"].notna().all(), len(target_clubs)),
        audit_row(season, "match_team_join_coverage", len(matched) == len(matches), f"{len(matched)}/{len(matches)}"),
        audit_row(season, "source_match_appearance_reconciliation", target_source_matches == parsed_team_appearances, f"source={target_source_matches}; parsed={parsed_team_appearances}"),
        audit_row(season, "history_points_have_match_evidence", history_match_mismatches == 0, history_match_mismatches),
    ]
    return audits


def fetch(url: str, cache_path: Path, refresh: bool) -> str:
    if refresh or not cache_path.exists():
        request = Request(url, headers={"User-Agent": "AO-European-Elo-Backtest/1.0"})
        context = ssl.create_default_context(cafile=certifi.where())
        with urlopen(request, timeout=30, context=context) as response:
            cache_path.write_bytes(response.read())
    return read_cached_text(cache_path)


def read_cached_text(path: Path) -> str:
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("windows-1252")


def history_team_key(value: str) -> str:
    key = normalize_name(value)
    return CLUB_HISTORY_ALIASES.get(key, key)


def parse_html(html: str) -> KassiesaParser:
    parser = KassiesaParser()
    parser.feed(html)
    return parser


def parse_ccoef(html: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    parser = parse_html(html)
    country = ""
    countries: list[dict[str, object]] = []
    clubs: list[dict[str, object]] = []
    for row in parser.rows:
        if "countryline" in row.row_class and len(row.cells) >= 12:
            country = re.sub(r"\s+\d+\s+teams?$", "", row.cells[1]).strip()
            countries.append({"country": country, "coefficient": to_float(row.cells[11])})
        elif "clubline" in row.row_class and len(row.cells) >= 11:
            values = [to_int(value) for value in row.cells[3:9]]
            clubs.append(
                {
                    "team_name": row.cells[1],
                    "country": country,
                    "competition": normalize_competition(row.cells[2]),
                    "qual_wins": values[0], "qual_draws": values[1], "qual_losses": values[2],
                    "wins": values[3], "draws": values[4], "losses": values[5],
                    "bonus": to_float(row.cells[9]), "points": to_float(row.cells[10]),
                    "matches": sum(values),
                }
            )
    return pd.DataFrame(countries), pd.DataFrame(clubs)


def parse_trank(html: str) -> pd.DataFrame:
    rows = []
    for row in parse_html(html).rows:
        if "clubline" not in row.row_class or len(row.cells) < 11:
            continue
        rows.append(
            {
                "rank": to_int(row.cells[0]), "team_name": row.cells[2],
                "country_code": row.cells[3].upper(),
                **{f"p{i}": to_float(row.cells[4 + i]) for i in range(5)},
                "total": to_float(row.cells[9]), "country_part": to_float(row.cells[10]),
            }
        )
    return pd.DataFrame(rows)


def parse_crank(html: str) -> pd.DataFrame:
    rows = []
    for row in parse_html(html).rows:
        if "countryline" not in row.row_class or len(row.cells) < 10:
            continue
        rows.append(
            {
                "rank": to_int(row.cells[0]), "country": row.cells[2],
                **{f"p{i}": to_float(row.cells[3 + i]) for i in range(5)},
                "total": to_float(row.cells[8]),
            }
        )
    return pd.DataFrame(rows)


def parse_matches(html: str) -> pd.DataFrame:
    matches = []
    for row in parse_html(html).rows:
        if len(row.cells) != 6:
            continue
        first = parse_score(row.cells[4])
        second = parse_score(row.cells[5])
        if first:
            matches.append(match_row(row, first, reverse=False))
        if second:
            matches.append(match_row(row, second, reverse=True))
    return pd.DataFrame(matches)


def match_row(row: HtmlRow, score: tuple[int, int], reverse: bool) -> dict[str, object]:
    left_name, left_code, right_name, right_code = row.cells[:4]
    if reverse:
        left_name, right_name = right_name, left_name
        left_code, right_code = right_code, left_code
        score = (score[1], score[0])
    return {
        "competition": normalize_competition(row.cup), "round": row.round_name,
        "home_team_name": left_name, "home_country_code": left_code.upper(),
        "away_team_name": right_name, "away_country_code": right_code.upper(),
        "home_goals": score[0], "away_goals": score[1],
    }


def parse_qualification(html: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cup_winners = {
        normalize_name(value)
        for value in re.findall(r"cup winner:\s*([^,<\n]+)", html, flags=re.IGNORECASE)
    }
    for block in parse_html(html).pre_blocks:
        lines = [line for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        country_match = re.match(r"\s*\d+\s+(.+?)\s+\d+\.\d+\s*$", lines[0])
        country = country_match.group(1).strip() if country_match else ""
        for line in lines[2:]:
            match = QUAL_LINE_RE.match(line)
            if not match:
                continue
            route = match.group("route")
            rows.append(
                {
                    "country": country, "route": route,
                    "team_name": match.group("team").strip(),
                    "entry_round": match.group("entry"),
                    "is_league_champion": route in {"CH", "CL", "CL1"},
                    "is_cup_winner": route == "CW" or normalize_name(match.group("team")) in cup_winners,
                }
            )
    return pd.DataFrame(rows)


def normalize_competition(value: str) -> str:
    upper = value.upper()
    if "CHAMPIONS" in upper or upper == "CL":
        return "UCL"
    if "EUROPA LEAGUE" in upper or upper == "EL":
        return "UEL"
    if "CONFERENCE" in upper or upper in {"CO", "ECL"}:
        return "UECL"
    return value.strip()


def normalize_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    text = text.lower().replace("&", " and ")
    key = re.sub(r"[^a-z0-9]+", "", text)
    return TEAM_ALIASES.get(key, key)


def clean_text(value: str) -> str:
    return " ".join(value.split())


def to_float(value: str) -> float:
    return float(value) if value.strip() else 0.0


def to_int(value: str) -> int:
    return int(value) if value.strip() else 0


def parse_score(value: str) -> tuple[int, int] | None:
    match = SCORE_RE.match(value.strip())
    return (int(match.group(1)), int(match.group(2))) if match else None


def match_cap(end_year: int, competition: str) -> int:
    if end_year >= 2025 and competition in {"UCL", "UEL"}:
        return 8
    return 6


def season_label(end_year: int) -> str:
    return f"{end_year - 1}/{str(end_year)[-2:]}"


def audit_row(season: str, check: str, passed: bool, detail: object) -> dict[str, object]:
    return {"season": season, "check": check, "status": "PASS" if passed else "FAIL", "detail": detail}


def write_readme(audit: pd.DataFrame) -> None:
    season_rows = []
    for end_year in TARGET_END_YEARS:
        folder = OUTPUT_ROOT / season_label(end_year).replace("/", "-")
        teams = pd.read_csv(folder / "teams.csv")
        matches = pd.read_csv(folder / "matches.csv")
        season_rows.append(f"| {season_label(end_year)} | {len(teams)} | {len(matches)} |")
    text = "\n".join(
        [
            "# AO European Elo Historical Backtest Data",
            "",
            "This dataset is generated from pre-season and completed-season Kassiesa UEFA tables.",
            "Every target season uses only information available before that season starts.",
            "",
            "| Target season | Teams | Matches |",
            "| --- | ---: | ---: |",
            *season_rows,
            "",
            "## Stage A limitation",
            "",
            "Domestic qualification routes identify league champions and cup entrants, but this",
            "first layer intentionally leaves domestic position and league team count empty for",
            "non-champions. It is suitable for calibrating European-history and exposure parameters,",
            "not the domestic percentile/component parameters.",
            "",
            "## Sources",
            "",
            "- https://kassiesa.net/uefa/data/index.html",
            "- https://kassiesa.net/uefa/history/qual2024.html (year changes by season)",
            "",
            f"Build audit: {int((audit['status'] == 'PASS').sum())}/{len(audit)} checks passed.",
        ]
    )
    (OUTPUT_ROOT / "README.md").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
