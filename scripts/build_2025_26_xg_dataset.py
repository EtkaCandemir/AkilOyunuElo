from __future__ import annotations

import argparse
import json
import math
import ssl
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ao_elo.xg_dataset import (  # noqa: E402
    MASTER_COLUMNS,
    attach_secondary_xg,
    build_coverage_summary,
    build_source_comparison,
    empty_primary_xg,
    flatten_fotmob_date_payload,
    parse_fotmob_xg,
    read_season_events,
    resolve_fotmob_identity,
    sha256_file,
    stable_json_dump,
    validate_master_dataset,
)
from scripts.build_external_elo_benchmark import load_uefa_matches  # noqa: E402


EVENTS_PATH = (
    ROOT / "data" / "external_elo_benchmark_2018_2026" / "exact_date_events.csv"
)
SECONDARY_XG_PATH = ROOT / "data" / "xg_backtest_2018_2026" / "xg_matches.csv"
OUTPUT_ROOT = ROOT / "data" / "xg_2025_26"
FOTMOB_DATE_URL = "https://www.fotmob.com/api/data/matches"
FOTMOB_MATCH_URL = "https://www.fotmob.com/api/data/matchDetails"
FOTMOB_FAQ_URL = "https://www.fotmob.com/pt-BR/faq"
USER_AGENT = "AO-European-Elo-xG-dataset/1.0 (+public research cache)"


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the audited 2025/26 UEFA results and free FotMob xG dataset"
    )
    parser.add_argument("--events", type=Path, default=EVENTS_PATH)
    parser.add_argument(
        "--season",
        action="append",
        default=None,
        help="season to build; repeat for several (default 2025/26)",
    )
    parser.add_argument("--secondary-xg", type=Path, default=SECONDARY_XG_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh-uefa", action="store_true")
    parser.add_argument("--refresh-fotmob", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--request-delay", type=float, default=0.35)
    parser.add_argument(
        "--max-match-pages",
        type=int,
        default=None,
        help="Development-only limit; omitted for the complete dataset",
    )
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 8:
        raise ValueError("workers must be in 1..8")
    if args.request_delay < 0.0:
        raise ValueError("request-delay cannot be negative")

    output_root = args.output_root.resolve()
    cache_root = output_root / "_source_cache"
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    events_path = args.events.resolve()
    secondary_path = args.secondary_xg.resolve()
    seasons = tuple(args.season) if args.season else ("2025/26",)
    events = read_season_events(events_path, seasons)
    refreshed, uefa_audit = refresh_uefa_results(
        events,
        cache_root / "uefa",
        refresh=args.refresh_uefa,
        offline=args.offline,
    )

    limiter = RequestLimiter(args.request_delay)
    date_rows, date_status = load_fotmob_date_rows(
        refreshed,
        cache_root / "fotmob_dates",
        refresh=args.refresh_fotmob,
        offline=args.offline,
        limiter=limiter,
    )
    identities, unresolved_reasons, identity_audit = resolve_all_identities(
        refreshed, date_rows, date_status
    )
    details = load_fotmob_details(
        identities,
        cache_root / "fotmob_matches",
        refresh=args.refresh_fotmob,
        offline=args.offline,
        workers=args.workers,
        limiter=limiter,
        max_pages=args.max_match_pages,
    )
    master, xg_audit = build_master(refreshed, identities, unresolved_reasons, details)

    if not secondary_path.is_file():
        raise ValueError(f"secondary xG file not found: {secondary_path}")
    master = attach_secondary_xg(master, pd.read_csv(secondary_path))
    master = master.loc[:, MASTER_COLUMNS]
    # The frozen-dataset guarantees (exact match counts, single season, complete
    # identity resolution) only hold for the original 2025/26 build.
    validate_master_dataset(master, strict_counts=seasons == ("2025/26",))

    audit = uefa_audit.merge(identity_audit, on="match_id", how="left", validate="one_to_one")
    audit = audit.merge(xg_audit, on="match_id", how="left", validate="one_to_one")
    coverage = build_coverage_summary(master)
    comparison = build_source_comparison(master)

    master_path = output_root / "uefa_2025_26_matches_with_xg.csv"
    audit_path = output_root / "xg_identity_audit.csv"
    coverage_path = output_root / "xg_coverage_summary.csv"
    comparison_path = output_root / "xg_source_comparison.csv"
    master.to_csv(master_path, index=False, lineterminator="\n", float_format="%.10g")
    audit.to_csv(audit_path, index=False, lineterminator="\n", float_format="%.10g")
    coverage.to_csv(coverage_path, index=False, lineterminator="\n", float_format="%.10g")
    comparison.to_csv(comparison_path, index=False, lineterminator="\n", float_format="%.10g")

    manifest = build_manifest(
        events_path,
        secondary_path,
        cache_root,
        master,
        coverage,
        details,
        uefa_audit,
    )
    stable_json_dump(manifest, output_root / "source_manifest.json")
    (output_root / "README.md").write_text(
        build_readme(master, coverage, comparison, args.offline),
        encoding="utf-8",
    )

    print("AO 2025/26 UEFA + free xG dataset")
    print(f"Matches: {len(master)}")
    print(f"Competitions: {master['competition'].value_counts().sort_index().to_dict()}")
    print(f"Primary FotMob xG covered: {int(master['xg_covered'].sum())}")
    print(f"Primary xG analysis eligible: {int(master['xg_analysis_eligible'].sum())}")
    print(f"Secondary coarse xG covered: {int(master['secondary_xg_covered'].sum())}")
    print(f"Output: {master_path}")


def refresh_uefa_results(
    events: pd.DataFrame,
    cache_dir: Path,
    *,
    refresh: bool,
    offline: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_files = list(cache_dir.glob("*.json")) if cache_dir.exists() else []
    if offline and not cache_files:
        # The canonical exact-date dataset was already built from the same UEFA source.
        result = events.copy()
        result["score_verified"] = True
        result["chronology_verified"] = True
        audit = pd.DataFrame(
            {
                "match_id": result["match_id"],
                "uefa_verification_status": "CANONICAL_UEFA_SNAPSHOT",
                "uefa_score_changed": False,
                "uefa_kickoff_changed": False,
                "uefa_team_identity_changed": False,
            }
        )
        result["event_order"] = range(1, len(result) + 1)
        return result, audit

    official = load_uefa_matches(events, cache_dir, refresh=refresh and not offline)
    official = official.copy()
    official["uefa_match_id"] = official["uefa_match_id"].map(id_text)
    if official["uefa_match_id"].duplicated().any():
        raise ValueError("UEFA refresh contains duplicate match IDs")
    by_id = official.set_index("uefa_match_id")
    rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for source in events.itertuples(index=False):
        event = source._asdict()
        uefa_id = id_text(event["uefa_match_id"])
        if uefa_id not in by_id.index:
            raise ValueError(f"UEFA refresh missing match {event['match_id']} / {uefa_id}")
        current = by_id.loc[uefa_id]
        home_id = id_text(event["uefa_home_team_id"])
        away_id = id_text(event["uefa_away_team_id"])
        team_changed = bool(
            home_id != id_text(current["uefa_home_team_id"])
            or away_id != id_text(current["uefa_away_team_id"])
        )
        if team_changed:
            raise ValueError(
                f"UEFA team identity changed for {event['match_id']} / {uefa_id}"
            )
        official_home = current["uefa_home_goals_total"]
        official_away = current["uefa_away_goals_total"]
        if pd.isna(official_home) or pd.isna(official_away):
            raise ValueError(f"UEFA result is incomplete for {event['match_id']} / {uefa_id}")
        official_home = int(official_home)
        official_away = int(official_away)
        score_changed = bool(
            official_home != int(event["home_goals"])
            or official_away != int(event["away_goals"])
        )
        official_kickoff = pd.to_datetime(current["kickoff_utc"], utc=True, errors="coerce")
        if pd.isna(official_kickoff):
            raise ValueError(f"UEFA kickoff is invalid for {event['match_id']} / {uefa_id}")
        kickoff_changed = bool(
            abs(official_kickoff - pd.Timestamp(event["kickoff_utc"]))
            > pd.Timedelta(seconds=1)
        )
        event["home_goals"] = official_home
        event["away_goals"] = official_away
        event["actual_home_score"] = (
            1.0 if official_home > official_away else 0.0 if official_home < official_away else 0.5
        )
        event["goal_difference"] = abs(official_home - official_away)
        event["kickoff_utc"] = official_kickoff
        event["kickoff_date"] = official_kickoff.date().isoformat()
        event["uefa_match_id"] = uefa_id
        event["uefa_home_team_id"] = home_id
        event["uefa_away_team_id"] = away_id
        event["score_verified"] = True
        event["chronology_verified"] = True
        rows.append(event)
        audit_rows.append(
            {
                "match_id": event["match_id"],
                "uefa_verification_status": "LIVE_REFRESH_VERIFIED",
                "uefa_score_changed": score_changed,
                "uefa_kickoff_changed": kickoff_changed,
                "uefa_team_identity_changed": team_changed,
            }
        )
    result = pd.DataFrame(rows).sort_values(["kickoff_utc", "match_id"], kind="stable")
    result = result.reset_index(drop=True)
    result["event_order"] = range(1, len(result) + 1)
    return result, pd.DataFrame(audit_rows)


def load_fotmob_date_rows(
    events: pd.DataFrame,
    cache_dir: Path,
    *,
    refresh: bool,
    offline: bool,
    limiter: RequestLimiter,
) -> tuple[pd.DataFrame, dict[str, str]]:
    rows: list[dict[str, object]] = []
    status: dict[str, str] = {}
    event_dates = {value.date() for value in events["kickoff_utc"]}
    requested_dates = sorted(
        {
            event_date + timedelta(days=offset)
            for event_date in event_dates
            for offset in (-1, 0, 1)
        }
    )
    for event_date in requested_dates:
        date = event_date.strftime("%Y%m%d")
        url = f"{FOTMOB_DATE_URL}?{urlencode({'date': date, 'timezone': 'UTC', 'ccode3': 'USA'})}"
        payload, metadata = fetch_json_cached(
            url,
            cache_dir / f"{date}.json",
            refresh=refresh,
            offline=offline,
            limiter=limiter,
        )
        status[date] = str(metadata["status"])
        if payload is not None:
            rows.extend(flatten_fotmob_date_payload(payload))
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(
            columns=[
                "source_match_id",
                "competition",
                "league_name",
                "kickoff_utc",
                "home_team_id",
                "away_team_id",
                "home_team_name",
                "away_team_name",
                "home_goals",
                "away_goals",
            ]
        )
    return frame, status


def resolve_all_identities(
    events: pd.DataFrame,
    date_rows: pd.DataFrame,
    date_status: dict[str, str],
) -> tuple[dict[str, object], dict[str, str], pd.DataFrame]:
    identities: dict[str, object] = {}
    unresolved_reasons: dict[str, str] = {}
    audit_rows: list[dict[str, object]] = []
    for _, event in events.iterrows():
        date_key = pd.Timestamp(event["kickoff_utc"]).strftime("%Y%m%d")
        source_kickoff = pd.to_datetime(date_rows["kickoff_utc"], utc=True, errors="coerce")
        candidates = date_rows.loc[
            (source_kickoff - pd.Timestamp(event["kickoff_utc"]))
            .abs()
            .le(pd.Timedelta(hours=36))
        ]
        candidate, audit = resolve_fotmob_identity(event, candidates)
        match_id = str(event["match_id"])
        identities[match_id] = candidate
        if candidate is None:
            unresolved_reasons[match_id] = (
                "SOURCE_PAGE_UNAVAILABLE"
                if date_status.get(date_key) != "OK"
                else "IDENTITY_AMBIGUOUS"
            )
        audit_rows.append(
            {
                "match_id": str(event["match_id"]),
                "fotmob_date_feed_status": date_status.get(date_key, "SOURCE_PAGE_UNAVAILABLE"),
                **audit,
            }
        )
    return identities, unresolved_reasons, pd.DataFrame(audit_rows)


def load_fotmob_details(
    identities: dict[str, object],
    cache_dir: Path,
    *,
    refresh: bool,
    offline: bool,
    workers: int,
    limiter: RequestLimiter,
    max_pages: int | None,
) -> dict[str, tuple[dict[str, Any] | None, dict[str, Any]]]:
    accepted = [(match_id, value) for match_id, value in identities.items() if value is not None]
    accepted.sort(key=lambda item: item[0])
    if max_pages is not None:
        accepted = accepted[: max(0, max_pages)]
    results: dict[str, tuple[dict[str, Any] | None, dict[str, Any]]] = {}

    def fetch(item: tuple[str, object]) -> tuple[str, dict[str, Any] | None, dict[str, Any]]:
        match_id, candidate = item
        source_match_id = str(getattr(candidate, "source_match_id"))
        url = f"{FOTMOB_MATCH_URL}?{urlencode({'matchId': source_match_id})}"
        payload, metadata = fetch_json_cached(
            url,
            cache_dir / f"{source_match_id}.json",
            refresh=refresh,
            offline=offline,
            limiter=limiter,
        )
        return match_id, payload, metadata

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch, item): item[0] for item in accepted}
        for index, future in enumerate(as_completed(futures), start=1):
            match_id, payload, metadata = future.result()
            results[match_id] = (payload, metadata)
            if index % 100 == 0 or index == len(futures):
                print(f"FotMob detail cache: {index}/{len(futures)}")
    return results


def fetch_json_cached(
    url: str,
    cache_path: Path,
    *,
    refresh: bool,
    offline: bool,
    limiter: RequestLimiter,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    metadata_path = cache_path.with_suffix(".meta.json")
    if cache_path.is_file() and metadata_path.is_file() and not refresh:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        try:
            return json.loads(cache_path.read_text(encoding="utf-8")), metadata
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid cached JSON: {cache_path}") from exc
    if offline:
        return None, {
            "url": url,
            "status": "SOURCE_PAGE_UNAVAILABLE",
            "snapshot_utc": "",
            "sha256": "",
        }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    limiter.wait()
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    context = ssl.create_default_context(cafile=certifi.where())
    try:
        with urlopen(request, timeout=90, context=context) as response:
            payload = response.read()
            response_date = response.headers.get("Date")
    except HTTPError as exc:
        status = "ACCESS_BLOCKED" if exc.code in {401, 403, 429} else "SOURCE_PAGE_UNAVAILABLE"
        metadata = {
            "url": url,
            "status": status,
            "http_status": exc.code,
            "snapshot_utc": datetime.now(timezone.utc).isoformat(),
            "sha256": "",
        }
        stable_json_dump(metadata, metadata_path)
        return None, metadata
    except (TimeoutError, URLError, OSError) as exc:
        metadata = {
            "url": url,
            "status": "SOURCE_PAGE_UNAVAILABLE",
            "error": type(exc).__name__,
            "snapshot_utc": datetime.now(timezone.utc).isoformat(),
            "sha256": "",
        }
        stable_json_dump(metadata, metadata_path)
        return None, metadata
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"source did not return JSON: {url}") from exc
    cache_path.write_bytes(payload)
    snapshot = _response_snapshot(response_date)
    metadata = {
        "url": url,
        "status": "OK",
        "http_status": 200,
        "snapshot_utc": snapshot,
        "sha256": sha256_file(cache_path),
    }
    stable_json_dump(metadata, metadata_path)
    return parsed, metadata


def build_master(
    events: pd.DataFrame,
    identities: dict[str, object],
    unresolved_reasons: dict[str, str],
    details: dict[str, tuple[dict[str, Any] | None, dict[str, Any]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for _, event in events.iterrows():
        match_id = str(event["match_id"])
        candidate = identities.get(match_id)
        if candidate is None:
            missing_reason = unresolved_reasons.get(match_id, "IDENTITY_AMBIGUOUS")
            xg = empty_primary_xg(missing_reason)
            xg_audit = {
                "fotmob_fetch_status": "NOT_REQUESTED",
                "detail_identity_verified": False,
                "detail_score_verified": False,
                "detail_chronology_verified": False,
                "xg_disposition": missing_reason,
            }
            identity_method = "UEFA_MATCH_ID_VERIFIED_FOTMOB_UNRESOLVED"
            identity_confidence = math.nan
        else:
            payload, metadata = details.get(
                match_id,
                (
                    None,
                    {
                        "status": "SOURCE_PAGE_UNAVAILABLE",
                        "snapshot_utc": "",
                    },
                ),
            )
            xg, xg_audit = parse_fotmob_xg(
                event,
                candidate,
                payload,
                fetch_status=str(metadata.get("status") or "SOURCE_PAGE_UNAVAILABLE"),
                snapshot_utc=str(metadata.get("snapshot_utc") or ""),
            )
            identity_method = "UEFA_ID_PLUS_FOTMOB_UTC_SCORE_NAME"
            identity_confidence = float(getattr(candidate, "pair_similarity"))
        base = event.to_dict()
        base.update(xg)
        base["identity_match_method"] = identity_method
        base["identity_confidence"] = identity_confidence
        rows.append(base)
        audit_rows.append({"match_id": match_id, **xg_audit})
    result = pd.DataFrame(rows).sort_values(["kickoff_utc", "match_id"], kind="stable")
    result = result.reset_index(drop=True)
    result["event_order"] = range(1, len(result) + 1)
    return result, pd.DataFrame(audit_rows)


def build_manifest(
    events_path: Path,
    secondary_path: Path,
    cache_root: Path,
    master: pd.DataFrame,
    coverage: pd.DataFrame,
    details: dict[str, tuple[dict[str, Any] | None, dict[str, Any]]],
    uefa_audit: pd.DataFrame,
) -> dict[str, object]:
    source_files = []
    for path in sorted(cache_root.rglob("*.json")):
        if path.name.endswith(".meta.json"):
            continue
        source_files.append(
            {
                "path": path.relative_to(cache_root).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    status_counts: dict[str, int] = {}
    for _, metadata in details.values():
        key = str(metadata.get("status") or "UNKNOWN")
        status_counts[key] = status_counts.get(key, 0) + 1
    all_summary = coverage.loc[
        coverage["competition"].eq("ALL") & coverage["round"].eq("ALL")
    ].iloc[0]
    snapshots = pd.to_datetime(master["xg_snapshot_utc"], utc=True, errors="coerce")
    return {
        "dataset": "AO 2025/26 UEFA match results with public free xG coverage",
        "model_inputs_changed": False,
        "season": "2025/26",
        "sources": {
            "official_results": "https://match.uefa.com/v5/matches",
            "primary_xg_page_data": FOTMOB_MATCH_URL,
            "primary_xg_documentation": FOTMOB_FAQ_URL,
            "secondary_xg": "API-Football via eatpizzanot/soccer-dataset",
        },
        "input_sha256": {
            "exact_date_events.csv": sha256_file(events_path),
            "secondary_xg_matches.csv": sha256_file(secondary_path),
        },
        "cache_files": source_files,
        "fotmob_detail_status_counts": status_counts,
        "source_as_of_utc": (
            snapshots.max().isoformat() if snapshots.notna().any() else ""
        ),
        "uefa_refresh": {
            "verification_status_counts": {
                str(key): int(value)
                for key, value in uefa_audit["uefa_verification_status"].value_counts().items()
            },
            "score_changes_detected": int(uefa_audit["uefa_score_changed"].sum()),
            "kickoff_changes_detected": int(uefa_audit["uefa_kickoff_changed"].sum()),
            "team_identity_changes_detected": int(
                uefa_audit["uefa_team_identity_changed"].sum()
            ),
        },
        "rows": {
            "matches": int(len(master)),
            "teams": int(
                len(pd.unique(pd.concat([master["home_team_id"], master["away_team_id"]])))
            ),
            "competition_counts": {
                str(key): int(value)
                for key, value in master["competition"].value_counts().sort_index().items()
            },
            "primary_xg_covered": int(all_summary["xg_covered"]),
            "primary_xg_analysis_eligible": int(all_summary["xg_analysis_eligible"]),
            "secondary_xg_covered": int(master["secondary_xg_covered"].sum()),
        },
        "contract": {
            "field_score": "90 minutes or 120 minutes when extra time was played; shootout excluded",
            "primary_xg": "FotMob shot-based match xG; in-match penalties included; shootout excluded",
            "missing_xg": "never imputed",
            "secondary_isolation": "coarse xG is never copied into primary xG columns",
            "production_eligibility": False,
        },
    }


def build_readme(
    master: pd.DataFrame,
    coverage: pd.DataFrame,
    comparison: pd.DataFrame,
    offline: bool,
) -> str:
    competition = coverage.loc[coverage["round"].eq("ALL") & coverage["competition"].ne("ALL")]
    lines = [
        "| Turnuva | Maç | FotMob xG | Analize uygun | Kapsama |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in competition.itertuples(index=False):
        lines.append(
            f"| {row.competition} | {int(row.matches)} | {int(row.xg_covered)} | "
            f"{int(row.xg_analysis_eligible)} | {float(row.coverage_rate):.1%} |"
        )
    return f"""# AO 2025/26 UEFA Maç ve xG Veri Seti

Bu klasör 2025/26 sezonundaki UCL, UEL ve UECL maçlarını AO takım
kimlikleriyle birleştirir. Ana CSV her maç için tek satır içerir. AO First Elo ve
AO Live Elo bu çalışma sırasında değiştirilmemiştir.

## Kaynak Sözleşmesi

- Maç kimliği, tarih ve saha skoru UEFA `match.uefa.com/v5/matches` kaynağıyla
  doğrulanır.
- Birincil xG, FotMob'un herkese açık maç sayfasını besleyen kimlik doğrulamasız
  sayfa-verisi yanıtında açıkça gösterilen `Expected goals (xG)` toplamıdır.
- FotMob xG şut kalitesi temellidir. Maç içi penaltılar dahildir; penaltı atışları
  toplamdan çıkarılır. Uzatma oynandıysa yalnız 120 dakikayı kapsadığı
  doğrulanabilen değer analize uygun sayılır.
- Eksik veya kapsamı belirsiz xG sıfırla, ortalamayla ya da tahminle doldurulmaz.
- API-Football tabanlı coarse xG yalnız `secondary_xg_*` kolonlarında tutulur ve
  birincil FotMob kolonlarına taşınmaz.
- FotMob açıklaması: {FOTMOB_FAQ_URL}

Kamuya açık erişim, otomatik olarak yeniden dağıtım lisansı vermez. Bu CSV veya
FotMob kaynaklı alanlar herkese açık bir repoda yayımlanmadan önce güncel kaynak
kullanım ve yeniden dağıtım koşulları ayrıca kontrol edilmelidir.

## Kapsam

{chr(10).join(lines)}

- Toplam maç: {len(master)}
- Benzersiz AO takımı: {len(pd.unique(pd.concat([master['home_team_id'], master['away_team_id']])))}
- İki kaynakta ortak xG maçı: {len(comparison)}
- Çalıştırma modu: {'offline/cache' if offline else 'public-source refresh/cache'}

## Dosyalar

- `uefa_2025_26_matches_with_xg.csv`: 961 maçlık ana analiz tablosu.
- `xg_identity_audit.csv`: UEFA yenileme, FotMob kimlik ve xG kabul denetimi.
- `xg_coverage_summary.csv`: turnuva ve tur bazlı kapsam.
- `xg_source_comparison.csv`: iki xG kaynağının ortak maçları.
- `source_manifest.json`: kaynak URL'leri, SHA-256 değerleri ve sözleşme.

Ham yanıtlar `_source_cache/` altında tutulur ve Git'e eklenmez. Aynı cache ile
yeniden çalıştırıldığında ana CSV deterministik olarak üretilir.
"""


def id_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def _response_snapshot(value: str | None) -> str:
    if value:
        try:
            return parsedate_to_datetime(value).astimezone(timezone.utc).isoformat()
        except (TypeError, ValueError):
            pass
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
