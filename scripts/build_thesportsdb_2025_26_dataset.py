from __future__ import annotations

import argparse
import hashlib
import json
import os
import ssl
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import certifi
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ao_elo.thesportsdb_dataset import (  # noqa: E402
    DETAIL_ENDPOINTS,
    SPORTSDB_LEAGUES,
    SPORTSDB_SEASON,
    build_coverage_summary,
    build_event_table,
    build_field_catalog,
    build_stats_wide,
    build_xg_source_comparison,
    flatten_child_records,
    flatten_schedules,
    payload_rows,
    resolve_match_identities,
    validate_normalized_dataset,
)
from ao_elo.xg_dataset import (  # noqa: E402
    read_season_events,
    sha256_file,
    stable_json_dump,
)


EVENTS_PATH = (
    ROOT / "data" / "external_elo_benchmark_2018_2026" / "exact_date_events.csv"
)
OUTPUT_ROOT = ROOT / "data" / "thesportsdb_2025_26"
FOTMOB_XG_PATH = ROOT / "data" / "xg_2025_26" / "uefa_2025_26_matches_with_xg.csv"
ENV_PATH = ROOT / ".env.local"
API_BASE = "https://www.thesportsdb.com/api/v2/json"
USER_AGENT = "AO-European-Elo-TheSportsDB-dataset/1.0"


class RequestLimiter:
    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = max(0.0, float(interval_seconds))
        self._lock = threading.Lock()
        self._last_request = 0.0

    def wait(self) -> None:
        with self._lock:
            remaining = self.interval_seconds - (time.monotonic() - self._last_request)
            if remaining > 0.0:
                time.sleep(remaining)
            self._last_request = time.monotonic()


class SportsDbClient:
    def __init__(self, api_key: str, limiter: RequestLimiter) -> None:
        if not api_key.strip():
            raise ValueError("TheSportsDB API key cannot be empty")
        self._api_key = api_key.strip()
        self._limiter = limiter
        self._ssl_context = ssl.create_default_context(cafile=certifi.where())

    def fetch(
        self,
        path: str,
        cache_path: Path,
        *,
        refresh: bool,
        offline: bool,
    ) -> dict[str, Any]:
        if cache_path.is_file() and not refresh:
            return read_json(cache_path)
        if offline:
            raise ValueError(f"Offline cache is missing: {cache_path}")
        url = f"{API_BASE}{path}"
        for attempt in range(1, 6):
            self._limiter.wait()
            request = Request(
                url,
                headers={
                    "X-API-KEY": self._api_key,
                    "User-Agent": USER_AGENT,
                    "Accept": "application/json",
                },
            )
            try:
                with urlopen(
                    request, timeout=45, context=self._ssl_context
                ) as response:
                    payload = json.load(response)
                if not isinstance(payload, dict):
                    raise ValueError(f"Unexpected non-object payload from {path}")
                atomic_json_dump(payload, cache_path)
                return payload
            except HTTPError as error:
                if error.code == 429 and attempt < 5:
                    delay = retry_delay(error.headers.get("Retry-After"), attempt)
                    time.sleep(delay)
                    continue
                if 500 <= error.code < 600 and attempt < 5:
                    time.sleep(min(30.0, 2.0**attempt))
                    continue
                raise RuntimeError(
                    f"TheSportsDB request failed for {path}: HTTP {error.code}"
                ) from error
            except (URLError, TimeoutError) as error:
                if attempt == 5:
                    raise RuntimeError(
                        f"TheSportsDB request failed for {path}: {error}"
                    ) from error
                time.sleep(min(30.0, 2.0**attempt))
        raise RuntimeError(f"TheSportsDB request exhausted retries: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the complete TheSportsDB 2025/26 UEFA match dataset"
    )
    parser.add_argument("--events", type=Path, default=EVENTS_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--fotmob-xg", type=Path, default=FOTMOB_XG_PATH)
    parser.add_argument("--env-file", type=Path, default=ENV_PATH)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--request-delay", type=float, default=0.70)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Development-only request limit; normalized outputs require all events",
    )
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 8:
        raise ValueError("workers must be in 1..8")
    if args.request_delay < 0.60:
        raise ValueError("request-delay must be at least 0.60 seconds")
    if args.max_events is not None and args.max_events < 1:
        raise ValueError("max-events must be positive")

    output = args.output_root.resolve()
    cache = output / "_source_cache"
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    api_key = load_api_key(args.env_file.resolve())
    client = SportsDbClient(api_key, RequestLimiter(args.request_delay))

    schedules_payload = {}
    for competition, league_id in SPORTSDB_LEAGUES.items():
        schedules_payload[competition] = client.fetch(
            f"/schedule/league/{league_id}/{SPORTSDB_SEASON}",
            cache / "schedule" / f"{competition.lower()}.json",
            refresh=args.refresh,
            offline=args.offline,
        )
    schedules = flatten_schedules(schedules_payload)
    events = read_season_events(args.events.resolve())
    identities = resolve_match_identities(events, schedules)
    identity_context = identities.merge(
        events[["match_id", "competition", "home_team_name", "away_team_name"]],
        on="match_id",
        validate="one_to_one",
    )
    print(
        "TheSportsDB schedules: "
        f"{len(schedules)} matches; identity {len(identities)}/{len(events)}",
        flush=True,
    )

    source_ids = identities["source_event_id"].astype(str).tolist()
    if args.max_events is not None:
        source_ids = source_ids[: args.max_events]
    fetch_endpoint_payloads(
        client,
        source_ids,
        cache,
        refresh=args.refresh,
        offline=args.offline,
        workers=args.workers,
    )
    audit_event_results_endpoint(
        client,
        identity_context,
        cache,
        refresh=args.refresh,
        offline=args.offline,
    )
    if args.max_events is not None:
        print(
            f"Development fetch complete for {len(source_ids)} events; normalized outputs skipped."
        )
        return

    endpoint_payloads = {
        endpoint: load_endpoint_payloads(cache, endpoint, source_ids)
        for endpoint in DETAIL_ENDPOINTS
    }
    event_table = build_event_table(
        events,
        identities,
        schedules,
        endpoint_payloads["event"],
    )
    stats = flatten_child_records(
        identity_context, endpoint_payloads["stats"], endpoint="stats"
    )
    timeline = flatten_child_records(
        identity_context, endpoint_payloads["timeline"], endpoint="timeline"
    )
    lineup = flatten_child_records(
        identity_context, endpoint_payloads["lineup"], endpoint="lineup"
    )
    stats_wide = build_stats_wide(event_table, stats)
    coverage = build_coverage_summary(event_table, stats_wide, timeline, lineup)
    fotmob = pd.read_csv(args.fotmob_xg.resolve())
    xg_comparison, xg_comparison_summary = build_xg_source_comparison(
        stats_wide, fotmob
    )
    teams = build_team_identity(identities, events)
    tables = {
        "events": event_table,
        "event_stats_long": stats,
        "event_stats_wide": stats_wide,
        "event_timeline": timeline,
        "event_lineups": lineup,
        "team_identity": teams,
        "match_identity_audit": identity_context,
        "coverage_summary": coverage,
        "xg_source_comparison": xg_comparison,
        "xg_source_comparison_summary": xg_comparison_summary,
    }
    catalog = build_field_catalog(tables)
    validate_normalized_dataset(
        event_table, identities, stats, timeline, lineup, stats_wide
    )
    write_tables(output, tables, catalog)
    manifest = build_manifest(
        args.events.resolve(), output, cache, tables, catalog, args.request_delay
    )
    stable_json_dump(manifest, output / "source_manifest.json")
    (output / "README.md").write_text(
        build_readme(coverage, identity_context, tables, xg_comparison_summary),
        encoding="utf-8",
    )
    (output / "data_quality_report.md").write_text(
        build_quality_report(
            coverage, identity_context, tables, xg_comparison_summary
        ),
        encoding="utf-8",
    )

    all_row = coverage.loc[
        coverage["competition"].eq("ALL") & coverage["round"].eq("ALL")
    ].iloc[0]
    print("TheSportsDB 2025/26 UEFA dataset complete", flush=True)
    print(f"Events: {len(event_table)}", flush=True)
    print(f"Stats rows: {len(stats)}", flush=True)
    print(f"Timeline rows: {len(timeline)}", flush=True)
    print(f"Lineup rows: {len(lineup)}", flush=True)
    print(
        f"xG raw fields: {int(all_row['xg_covered'])}/{int(all_row['matches'])}; "
        f"analysis eligible: {int(all_row['xg_analysis_eligible'])}/"
        f"{int(all_row['matches'])} ({all_row['xg_analysis_eligible_rate']:.1%})",
        flush=True,
    )
    print(f"Output: {output}", flush=True)


def fetch_endpoint_payloads(
    client: SportsDbClient,
    source_ids: list[str],
    cache: Path,
    *,
    refresh: bool,
    offline: bool,
    workers: int,
) -> None:
    tasks = [
        (endpoint, path_name, source_id)
        for endpoint, path_name in DETAIL_ENDPOINTS.items()
        for source_id in source_ids
        if refresh or not (cache / endpoint / f"{source_id}.json").is_file()
    ]
    total = len(tasks)
    cached = len(DETAIL_ENDPOINTS) * len(source_ids) - total
    print(
        f"Detail cache: {cached} existing; {total} API requests pending",
        flush=True,
    )
    if total == 0:
        return

    def fetch_one(task: tuple[str, str, str]) -> tuple[str, str]:
        endpoint, path_name, source_id = task
        client.fetch(
            f"/lookup/{path_name}/{source_id}",
            cache / endpoint / f"{source_id}.json",
            refresh=refresh,
            offline=offline,
        )
        return endpoint, source_id

    completed = 0
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch_one, task) for task in tasks]
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 100 == 0 or completed == total:
                elapsed = time.monotonic() - started
                rate = completed / elapsed * 60.0 if elapsed else 0.0
                remaining = (total - completed) / max(rate, 1e-9)
                print(
                    f"  requests {completed}/{total}; {rate:.1f}/min; "
                    f"ETA {remaining:.1f} min",
                    flush=True,
                )


def audit_event_results_endpoint(
    client: SportsDbClient,
    identities: pd.DataFrame,
    cache: Path,
    *,
    refresh: bool,
    offline: bool,
) -> None:
    samples = (
        identities.sort_values("match_id", kind="stable")
        .groupby("competition", sort=True)
        .head(1)
    )
    rows = []
    for row in samples.itertuples(index=False):
        payload = client.fetch(
            f"/lookup/event_results/{row.source_event_id}",
            cache / "event_results_schema_audit" / f"{row.source_event_id}.json",
            refresh=refresh,
            offline=offline,
        )
        rows.append(
            {
                "competition": row.competition,
                "source_event_id": row.source_event_id,
                "records": len(payload_rows(payload)),
                "top_level_fields": sorted(payload),
            }
        )
    if any(row["records"] for row in rows):
        raise ValueError(
            "event_results returned soccer records; add it to the full collector"
        )
    stable_json_dump(
        {
            "status": "NOT_APPLICABLE_FOR_SOCCER",
            "reason": "One UCL, UEL and UECL sample returned no result records.",
            "samples": rows,
        },
        cache.parent / "event_results_endpoint_audit.json",
    )


def load_endpoint_payloads(
    cache: Path,
    endpoint: str,
    source_ids: list[str],
) -> dict[str, dict[str, Any]]:
    payloads = {}
    for source_id in source_ids:
        path = cache / endpoint / f"{source_id}.json"
        if not path.is_file():
            raise ValueError(f"Missing {endpoint} cache for event {source_id}")
        payloads[source_id] = read_json(path)
    return payloads


def build_team_identity(
    identities: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    bridge = identities.merge(
        events[
            [
                "match_id",
                "uefa_home_team_id",
                "uefa_away_team_id",
                "home_team_name",
                "away_team_name",
            ]
        ],
        on="match_id",
        suffixes=("_source", "_ao"),
        validate="one_to_one",
    )
    rows = []
    for row in bridge.itertuples(index=False):
        rows.extend(
            [
                {
                    "sportsdb_team_id": str(row.sportsdb_home_team_id),
                    "sportsdb_team_name": row.sportsdb_home_team_name,
                    "club_id": f"AO-UEFA-{int(row.uefa_home_team_id)}",
                    "uefa_team_id": int(row.uefa_home_team_id),
                    "ao_team_name": row.home_team_name,
                },
                {
                    "sportsdb_team_id": str(row.sportsdb_away_team_id),
                    "sportsdb_team_name": row.sportsdb_away_team_name,
                    "club_id": f"AO-UEFA-{int(row.uefa_away_team_id)}",
                    "uefa_team_id": int(row.uefa_away_team_id),
                    "ao_team_name": row.away_team_name,
                },
            ]
        )
    frame = pd.DataFrame(rows).drop_duplicates().sort_values(
        ["club_id", "sportsdb_team_id"], kind="stable"
    )
    if frame.groupby("sportsdb_team_id")["club_id"].nunique().gt(1).any():
        raise ValueError("One TheSportsDB team ID maps to multiple AO club IDs")
    if frame.groupby("club_id")["sportsdb_team_id"].nunique().gt(1).any():
        raise ValueError("One AO club ID maps to multiple TheSportsDB team IDs")
    return frame.reset_index(drop=True)


def write_tables(
    output: Path,
    tables: dict[str, pd.DataFrame],
    catalog: pd.DataFrame,
) -> None:
    for name, frame in tables.items():
        frame.to_csv(
            output / f"{name}.csv",
            index=False,
            lineterminator="\n",
            float_format="%.10g",
        )
    catalog.to_csv(
        output / "field_catalog.csv",
        index=False,
        lineterminator="\n",
        float_format="%.10g",
    )


def build_manifest(
    events_path: Path,
    output: Path,
    cache: Path,
    tables: dict[str, pd.DataFrame],
    catalog: pd.DataFrame,
    request_delay: float,
) -> dict[str, Any]:
    cache_files = sorted(cache.rglob("*.json"))
    return {
        "dataset_version": "ao-thesportsdb-uefa-2025-26-v1",
        "provider": "TheSportsDB Premium API v2",
        "season": "2025/26",
        "competitions": SPORTSDB_LEAGUES,
        "request_delay_seconds": request_delay,
        "rate_limit_contract": (
            "100 requests/minute premium; configured client maximum "
            f"{60.0 / request_delay:.1f}/minute"
        ),
        "events_source": str(events_path),
        "events_source_sha256": sha256_file(events_path),
        "table_rows": {name: len(frame) for name, frame in tables.items()},
        "field_catalog_rows": len(catalog),
        "cache_files": len(cache_files),
        "cache_sha256": aggregate_hash(cache_files),
        "normalized_files": sorted(
            path.name for path in output.glob("*.csv")
        ),
        "event_results_endpoint": "NOT_APPLICABLE_FOR_SOCCER",
        "xg_scope": (
            "RAW_PROVIDER_FIELD_PRESERVED; ZERO_PLACEHOLDERS_EXCLUDED_FROM_"
            "ANALYSIS; DURATION/PENALTY_SEMANTICS_UNVERIFIED"
        ),
    }


def build_readme(
    coverage: pd.DataFrame,
    identities: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    xg_comparison_summary: pd.DataFrame,
) -> str:
    all_row = coverage.loc[
        coverage["competition"].eq("ALL") & coverage["round"].eq("ALL")
    ].iloc[0]
    xg_all = xg_comparison_summary.loc[
        xg_comparison_summary["competition"].eq("ALL")
    ].iloc[0]
    return f"""# TheSportsDB 2025/26 UEFA Match Dataset

This directory contains the complete premium API extraction for 2025/26 UCL,
UEL and UECL matches. The grain of `events.csv` and `event_stats_wide.csv` is
one row per AO match. Stats, timeline and lineup tables are long-form child
tables joined by `match_id` and `source_event_id`.

## Coverage

- Matches: {len(tables['events'])}
- Match identities: {len(identities)}
- Stats: {int(all_row['stats_covered'])}/{int(all_row['matches'])}
- Raw xG fields: {int(all_row['xg_covered'])}/{int(all_row['matches'])}
- Analysis-eligible TheSportsDB xG: {int(all_row['xg_analysis_eligible'])}/{int(all_row['matches'])}
- Suspected xG placeholders: {int(all_row['xg_placeholder_suspected'])}
- Common eligible TheSportsDB/FotMob xG: {int(xg_all['common_analysis_eligible'])}
- Timeline: {int(all_row['timeline_covered'])}/{int(all_row['matches'])}
- Lineup: {int(all_row['lineup_covered'])}/{int(all_row['matches'])}

`expected_goals` is preserved exactly as returned by TheSportsDB. Its provider
model, penalty inclusion and 90/120-minute scope are not documented, so it must
not replace the audited production xG source without a separate comparison.
Provider rows where xG is zero despite recorded shots, and rows where both xG
values are zero, remain in the raw columns but are excluded by
`xg_analysis_eligible=false`. `xg_source_comparison.csv` contains the match-level
FotMob comparison and `xg_source_comparison_summary.csv` contains aggregate
agreement metrics.

Raw API payloads are stored under ignored `_source_cache/`. API credentials are
read from `.env.local` and are never written to any output.
"""


def build_quality_report(
    coverage: pd.DataFrame,
    identities: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    xg_comparison_summary: pd.DataFrame,
) -> str:
    all_row = coverage.loc[
        coverage["competition"].eq("ALL") & coverage["round"].eq("ALL")
    ].iloc[0]
    score_mismatches = int((~identities["score_agrees"].astype(bool)).sum())
    fallback = int(identities["identity_method"].eq("DATE_TEAM_FALLBACK").sum())
    xg_all = xg_comparison_summary.loc[
        xg_comparison_summary["competition"].eq("ALL")
    ].iloc[0]
    return f"""# TheSportsDB 2025/26 Veri Kalitesi Raporu

## Veri ve Grain

- `events.csv`: {len(tables['events'])} maç, bir satır bir AO maçı.
- `event_stats_long.csv`: {len(tables['event_stats_long'])} istatistik kaydı.
- `event_timeline.csv`: {len(tables['event_timeline'])} olay kaydı.
- `event_lineups.csv`: {len(tables['event_lineups'])} kadro kaydı.

## Kimlik ve Tutarlılık

- Eşleşen maç: {len(identities)}/{len(identities)}.
- İkinci aşama ad/tarih fallback: {fallback}.
- AO ile TheSportsDB saha skoru uyuşmazlığı: {score_mismatches}.
- Provider event ID ve AO `match_id` ilişkisi bire birdir.

## Kapsam

- Stats: {int(all_row['stats_covered'])}/{int(all_row['matches'])} ({all_row['stats_coverage_rate']:.1%}).
- Ham xG alani: {int(all_row['xg_covered'])}/{int(all_row['matches'])} ({all_row['xg_coverage_rate']:.1%}).
- Supheli xG yer tutucu: {int(all_row['xg_placeholder_suspected'])}.
- Analize uygun TheSportsDB xG: {int(all_row['xg_analysis_eligible'])}/{int(all_row['matches'])} ({all_row['xg_analysis_eligible_rate']:.1%}).
- FotMob ile ortak analize uygun mac: {int(xg_all['common_analysis_eligible'])}.
- Ortak orneklem toplam xG MAE: {xg_all['total_mae']:.4f}; toplam xG bias (TheSportsDB - FotMob): {xg_all['total_bias']:+.4f}.
- Timeline: {int(all_row['timeline_covered'])}/{int(all_row['matches'])} ({all_row['timeline_coverage_rate']:.1%}).
- Lineup: {int(all_row['lineup_covered'])}/{int(all_row['matches'])} ({all_row['lineup_coverage_rate']:.1%}).

## Risk ve Kullanım Kararı

- **High:** Skor uyuşmazlıkları analiz öncesi maç bazında incelenmelidir; provider skoru AO sonucunun üzerine yazılmamıştır.
- **High:** xG süre ve penaltı kapsamı belgelenmemiştir; TheSportsDB xG production girdisi olarak otomatik kabul edilmemelidir.
- **High:** Sifir xG ile pozitif sut sayisi veya iki tarafli sifir xG bulunan provider satirlari ham olarak korunmus, analizden dislanmistir.
- **Medium:** FotMob karsilastirmasi kaynaklar arasi sistematik farki olcer; iki kaynaktan hangisinin gercek saha kalitesine daha yakin oldugunu tek basina kanitlamaz.
- **Medium:** Stats/timeline/lineup eksikliği ağırlıklı olarak ön eleme segmentlerinde beklenebilir; `coverage_summary.csv` aşama bazında kanıttır.
- **Low:** Event detail alanlarındaki medya, açıklama ve hava durumu değerleri seyrek olabilir; ham biçimde korunmuştur.
"""


def load_api_key(path: Path) -> str:
    existing = os.environ.get("THESPORTSDB_API_KEY", "").strip()
    if existing:
        return existing
    if not path.is_file():
        raise ValueError(
            "THESPORTSDB_API_KEY is missing from environment and .env.local"
        )
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == "THESPORTSDB_API_KEY":
            return value.strip().strip('"').strip("'")
    raise ValueError("THESPORTSDB_API_KEY is missing from .env.local")


def retry_delay(value: str | None, attempt: int) -> float:
    if value:
        try:
            return max(1.0, float(value))
        except ValueError:
            try:
                timestamp = parsedate_to_datetime(value)
                return max(
                    1.0,
                    (timestamp - datetime.now(timezone.utc)).total_seconds(),
                )
            except (TypeError, ValueError):
                pass
    return min(60.0, 10.0 * attempt)


def atomic_json_dump(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Cached API payload must be an object: {path}")
    return value


def aggregate_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(path.parents[2])).encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


if __name__ == "__main__":
    main()
