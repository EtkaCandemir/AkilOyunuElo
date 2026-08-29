from __future__ import annotations

import argparse
import hashlib
import json
import math
import ssl
import sys
import threading
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import certifi
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ao_elo.domestic_league_dataset import (  # noqa: E402
    PILOT_LEAGUES,
    DomesticFixtureConflictError,
    FIXTURE_RECONCILIATIONS,
    FIXTURE_AUDIT_COLUMNS,
    assess_league_season,
    attach_domestic_club_ids,
    build_domestic_team_bridge,
    normalize_schedule,
    payload_rows,
    table_expected_matches,
    validate_domestic_dataset,
)


OUTPUT_ROOT = ROOT / "data" / "domestic_league_matches_2013_2026"
REGISTRY_PATH = ROOT / "data" / "club_identity" / "club_registry.csv"
ENV_PATH = ROOT / ".env.local"
API_V2_BASE = "https://www.thesportsdb.com/api/v2/json"
API_V1_BASE = "https://www.thesportsdb.com/api/v1/json"
USER_AGENT = "AO-European-Elo-domestic-dataset/1.0"


class RequestLimiter:
    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = float(interval_seconds)
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
        if not api_key:
            raise ValueError("THESPORTSDB_API_KEY is missing")
        self.api_key = api_key
        self.limiter = limiter
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())

    def fetch_v2(
        self,
        path: str,
        cache_path: Path,
        *,
        refresh: bool,
        offline: bool,
    ) -> dict[str, Any]:
        return self._fetch(
            f"{API_V2_BASE}{path}",
            cache_path,
            refresh=refresh,
            offline=offline,
            headers={"X-API-KEY": self.api_key},
        )

    def fetch_table(
        self,
        league_id: str,
        provider_season: str,
        cache_path: Path,
        *,
        refresh: bool,
        offline: bool,
    ) -> dict[str, Any]:
        query = urlencode({"l": league_id, "s": provider_season})
        # The v1 table endpoint keeps its key in the path; cache filenames and manifests
        # never contain the resulting URL, so the key is not persisted to the repository.
        url = f"{API_V1_BASE}/{self.api_key}/lookuptable.php?{query}"
        return self._fetch(
            url,
            cache_path,
            refresh=refresh,
            offline=offline,
            headers={},
        )

    def _fetch(
        self,
        url: str,
        cache_path: Path,
        *,
        refresh: bool,
        offline: bool,
        headers: dict[str, str],
    ) -> dict[str, Any]:
        if cache_path.is_file() and not refresh:
            return read_json(cache_path)
        if offline:
            raise ValueError(f"Offline cache is missing: {cache_path}")
        for attempt in range(1, 6):
            self.limiter.wait()
            request = Request(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "application/json",
                    **headers,
                },
            )
            try:
                with urlopen(request, timeout=45, context=self.ssl_context) as response:
                    payload = json.load(response)
                if not isinstance(payload, dict):
                    raise ValueError("TheSportsDB returned a non-object JSON payload")
                atomic_json_dump(payload, cache_path)
                return payload
            except (HTTPError, json.JSONDecodeError, ValueError):
                # A missing season or non-JSON provider response is not transient.  It is
                # recorded by the caller as a rejected league-season rather than retried.
                raise
            except (TimeoutError, URLError):
                if attempt == 5:
                    raise
                time.sleep(min(20.0, 2.0**attempt))
        raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build 2013/14-2025/26 domestic schedule-only pilot dataset"
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--env-file", type=Path, default=ENV_PATH)
    parser.add_argument("--start-year", type=int, default=2013)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--request-delay", type=float, default=0.62)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--countries",
        default="",
        help="Optional comma-separated pilot country subset, e.g. ENG,TUR",
    )
    args = parser.parse_args()
    if args.start_year > args.end_year:
        raise ValueError("start-year must not exceed end-year")
    if args.request_delay < 0.60:
        raise ValueError("request-delay must be at least 0.60 seconds")
    selected_codes = {item.strip().upper() for item in args.countries.split(",") if item.strip()}
    specs = tuple(spec for spec in PILOT_LEAGUES if not selected_codes or spec.country_code in selected_codes)
    if not specs or selected_codes.difference({spec.country_code for spec in specs}):
        raise ValueError("--countries must name one or more pilot country codes")

    output = args.output_root.resolve()
    cache = output / "_source_cache"
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    client = SportsDbClient(load_api_key(args.env_file.resolve()), RequestLimiter(args.request_delay))

    quality_rows: list[dict[str, object]] = []
    accepted_frames: list[pd.DataFrame] = []
    raw_frames: list[pd.DataFrame] = []
    fixture_audit: list[dict[str, object]] = []
    total = len(specs) * (args.end_year - args.start_year + 1)
    completed = 0
    for spec in specs:
        for year in range(args.start_year, args.end_year + 1):
            provider_season = spec.provider_season(year)
            cache_key = provider_season.replace("/", "-")
            try:
                schedule = client.fetch_v2(
                    f"/schedule/league/{spec.sportsdb_league_id}/{provider_season}",
                    cache / "schedule" / spec.country_code / f"{cache_key}.json",
                    refresh=args.refresh,
                    offline=args.offline,
                )
                table = client.fetch_table(
                    spec.sportsdb_league_id,
                    provider_season,
                    cache / "table" / spec.country_code / f"{cache_key}.json",
                    refresh=args.refresh,
                    offline=args.offline,
                )
                matches = normalize_schedule(schedule, spec, provider_season, reconciliation_audit=fixture_audit)
                expected = table_expected_matches(table)
                assessment = assess_league_season(
                    matches,
                    spec=spec,
                    provider_season=provider_season,
                    expected_matches=expected,
                )
                raw_frames.append(matches.assign(quality_status=assessment["quality_status"]))
                if assessment["quality_status"] == "ACCEPTED":
                    accepted_frames.append(matches)
            except DomesticFixtureConflictError:
                raise
            except Exception as error:
                assessment = {
                    "country_code": spec.country_code,
                    "sportsdb_league_id": spec.sportsdb_league_id,
                    "league_name": spec.league_name,
                    "provider_season": provider_season,
                    "schedule_matches": 0,
                    "table_expected_matches": pd.NA,
                    "coverage_rate": math.nan,
                    "unique_events": 0,
                    "unique_teams": 0,
                    "timestamps_valid": False,
                    "scores_valid": False,
                    "quality_status": "REJECTED",
                    "quality_reason": f"FETCH_OR_PARSE_ERROR:{type(error).__name__}",
                    "error_detail": str(error),
                }
            quality_rows.append(assessment)
            completed += 1
            if completed % 10 == 0 or completed == total:
                print(f"Domestic league seasons: {completed}/{total}", flush=True)

    quality = pd.DataFrame(quality_rows).sort_values(
        ["country_code", "provider_season"], kind="stable"
    ).reset_index(drop=True)
    raw = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()
    accepted = pd.concat(accepted_frames, ignore_index=True) if accepted_frames else pd.DataFrame()
    if accepted.empty:
        raise ValueError("No domestic league seasons passed the quality gate")
    accepted = accepted.sort_values(["kickoff_utc", "match_id"], kind="stable").reset_index(drop=True)
    registry = pd.read_csv(args.registry.resolve())
    bridge = build_domestic_team_bridge(accepted, registry)
    accepted = attach_domestic_club_ids(accepted, bridge)
    accepted_quality = quality.loc[quality["quality_status"].eq("ACCEPTED")].copy()
    validate_domestic_dataset(accepted, accepted_quality, bridge)

    ambiguous = bridge["identity_ambiguous"].mean() if len(bridge) else 0.0
    identity_audit = pd.DataFrame(
        [
            {
                "bridge_rows": int(len(bridge)),
                "mapped_ao_clubs": int(bridge["ao_club_id"].notna().sum()),
                "ambiguous_rows": int(bridge["identity_ambiguous"].sum()),
                "ambiguous_rate": float(ambiguous),
                "identity_gate": "PASS" if ambiguous <= 0.01 else "FAIL",
            }
        ]
    )
    if ambiguous > 0.01:
        raise ValueError(
            "Domestic AO identity ambiguity exceeds the 1% acceptance limit: "
            f"{ambiguous:.2%}"
        )
    raw.to_csv(output / "raw_schedule_matches.csv", index=False)
    accepted.to_csv(output / "domestic_matches.csv", index=False)
    quality.to_csv(output / "league_season_quality.csv", index=False)
    bridge.to_csv(output / "domestic_team_bridge.csv", index=False)
    identity_audit.to_csv(output / "identity_audit.csv", index=False)
    pd.DataFrame(fixture_audit, columns=FIXTURE_AUDIT_COLUMNS).to_csv(output / "fixture_reconciliation_audit.csv", index=False)
    manifest = build_manifest(args, specs, output, quality, accepted)
    stable_json_dump(manifest, output / "source_manifest.json")
    (output / "README.md").write_text(build_readme(quality, accepted, identity_audit), encoding="utf-8")
    print(f"Accepted domestic matches: {len(accepted)}")
    print(f"Accepted league-seasons: {int(quality['quality_status'].eq('ACCEPTED').sum())}/{len(quality)}")
    print(f"Identity ambiguity: {ambiguous:.2%}")
    print(f"Output: {output}")


def load_api_key(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"TheSportsDB env file is missing: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("THESPORTSDB_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise ValueError("THESPORTSDB_API_KEY is missing from env file")


def build_manifest(args: argparse.Namespace, specs, output: Path, quality: pd.DataFrame, matches: pd.DataFrame) -> dict[str, object]:
    return {
        "dataset": "AO domestic league schedule pilot",
        "provider": "TheSportsDB Premium API",
        "provider_endpoints": {
            "schedule": "/api/v2/json/schedule/league/{league_id}/{season}",
            "final_table": "/api/v1/json/<redacted>/lookuptable.php?l={league_id}&s={season}",
        },
        "period": f"{args.start_year}/{str(args.start_year + 1)[-2:]}-{args.end_year}/{str(args.end_year + 1)[-2:]}",
        "pilot_countries": [spec.country_code for spec in specs],
        "request_delay_seconds": args.request_delay,
        "quality_gate": "valid UTC and score, unique IDs, final-table coverage >=95%",
        "fixture_reconciliations_sha256": sha256_file(FIXTURE_RECONCILIATIONS),
        "accepted_league_seasons": int(quality["quality_status"].eq("ACCEPTED").sum()),
        "accepted_matches": int(len(matches)),
        "files": {
            path.name: sha256_file(path)
            for path in sorted(output.glob("*.csv"))
            if path.is_file()
        },
        "raw_response_sha256": {
            str(path.relative_to(output)): sha256_file(path)
            for path in sorted((output / "_source_cache").rglob("*.json"))
            if path.is_file()
        },
    }


def build_readme(quality: pd.DataFrame, matches: pd.DataFrame, identity: pd.DataFrame) -> str:
    accepted = int(quality["quality_status"].eq("ACCEPTED").sum())
    return f"""# Domestic League Schedule Pilot\n\n- Grain: one completed domestic match per row.\n- Accepted league-seasons: {accepted}/{len(quality)}.\n- Accepted matches: {len(matches)}.\n- AO identity ambiguity rate: {float(identity.iloc[0]['ambiguous_rate']):.2%}.\n- `domestic_matches.csv` contains only league-seasons passing the all-or-nothing quality gate.\n- xG, lineups and match statistics are intentionally out of scope.\n"""


def atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Cached payload is not an object: {path}")
    return payload


def stable_json_dump(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
