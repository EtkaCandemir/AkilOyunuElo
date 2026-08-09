from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_backtest_dataset import fetch as fetch_text  # noqa: E402
from scripts.build_backtest_stage_b import (  # noqa: E402
    fetch as fetch_binary,
    normalize_country,
    parse_league_urls,
    read_standings,
)


DEFAULT_CACHE_ROOT = ROOT / "data" / "backtest_stage_b_2018_2026" / "_source_cache"
BASE_URL = "https://kassiesa.net/uefa/history"
NO_DOMESTIC_LEAGUE = {"liechtenstein"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build older full-league-table cache for five-season domestic history"
    )
    parser.add_argument("--start-year", type=int, default=2013)
    parser.add_argument("--end-year", type=int, default=2017)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--request-delay", type=float, default=0.15)
    args = parser.parse_args()
    if args.start_year > args.end_year:
        raise ValueError("start-year must not exceed end-year")
    if args.request_delay < 0:
        raise ValueError("request-delay must be non-negative")

    cache_root = args.cache_root.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for year in range(args.start_year, args.end_year + 1):
        qualification_path = cache_root / f"qual{year}.html"
        qualification_url = f"{BASE_URL}/qual{year}.html"
        qualification_html = fetch_text(
            qualification_url, qualification_path, args.refresh
        )
        league_urls = parse_league_urls(qualification_html)
        if not league_urls:
            raise ValueError(f"No league URLs found in {qualification_path}")
        for country_key, url in sorted(league_urls.items()):
            country_key = normalize_country(country_key)
            digest = hashlib.sha1(url.encode()).hexdigest()[:8]
            target = cache_root / f"{year}_{country_key}_{digest}.html"
            if country_key in NO_DOMESTIC_LEAGUE:
                rows.append(
                    {
                        "source_year": year,
                        "country_key": country_key,
                        "status": "NOT_APPLICABLE",
                        "league_team_count": 0,
                        "source_url": url,
                        "cache_path": str(target),
                        "detail": "country has no domestic league",
                    }
                )
                continue
            status = "CACHED" if target.exists() and not args.refresh else "FETCHED"
            detail = ""
            team_count = 0
            try:
                fetch_binary(url, target, args.refresh)
                tables = read_standings(target)
                team_count = max(len(table) for table in tables)
                if not 4 <= team_count <= 30:
                    raise ValueError(f"implausible team count: {team_count}")
            except Exception as exc:
                status = "FAILED"
                detail = f"{type(exc).__name__}: {exc}"
            rows.append(
                {
                    "source_year": year,
                    "country_key": country_key,
                    "status": status,
                    "league_team_count": team_count,
                    "source_url": url,
                    "cache_path": str(target),
                    "detail": detail,
                }
            )
            if status == "FETCHED" and args.request_delay:
                time.sleep(args.request_delay)

    audit = pd.DataFrame(rows).sort_values(["source_year", "country_key"])
    audit_path = cache_root.parent / "domestic_history_cache_audit.csv"
    audit.to_csv(audit_path, index=False)
    summary = audit.groupby(["source_year", "status"]).size().unstack(fill_value=0)
    print(summary.to_string())
    print(f"Audit: {audit_path}")
    failed = audit.loc[audit["status"].eq("FAILED")]
    if not failed.empty:
        raise ValueError(f"Historical standings cache has {len(failed)} failed pages")


if __name__ == "__main__":
    main()
