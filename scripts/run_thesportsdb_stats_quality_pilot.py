from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ao_elo.thesportsdb_dataset import payload_rows, stat_slug  # noqa: E402
from scripts.build_thesportsdb_2025_26_dataset import (  # noqa: E402
    RequestLimiter,
    SportsDbClient,
    load_api_key,
)


DOMESTIC_MATCHES = ROOT / "data" / "domestic_league_matches_2013_2026" / "domestic_matches.csv"
OUTPUT_ROOT = ROOT / "data" / "thesportsdb_stats_quality_pilot_2013_2026"
ENV_PATH = ROOT / ".env.local"
CORE_STATS = {"total_shots", "shots_on_goal"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a stratified historical TheSportsDB event-stats quality pilot"
    )
    parser.add_argument("--domestic-matches", type=Path, default=DOMESTIC_MATCHES)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--env-file", type=Path, default=ENV_PATH)
    parser.add_argument("--sample-size", type=int, default=600)
    parser.add_argument("--request-delay", type=float, default=0.62)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    if args.sample_size < 60:
        raise ValueError("sample-size must be at least 60")
    if args.request_delay < 0.60:
        raise ValueError("request-delay must be at least 0.60 seconds")
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be in 1..8")

    output = args.output_root.resolve()
    cache = output / "_source_cache" / "event_stats"
    output.mkdir(parents=True, exist_ok=True)
    matches = pd.read_csv(args.domestic_matches.resolve(), low_memory=False)
    sample = stratified_sample(matches, args.sample_size)
    client = SportsDbClient(
        load_api_key(args.env_file.resolve()), RequestLimiter(args.request_delay)
    )
    payloads: dict[str, dict] = {}

    def fetch(source_id: str) -> tuple[str, dict]:
        payload = client.fetch(
            f"/lookup/event_stats/{source_id}",
            cache / f"{source_id}.json",
            refresh=args.refresh,
            offline=args.offline,
        )
        return source_id, payload

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(fetch, str(source_id)): str(source_id)
            for source_id in sample["source_event_id"]
        }
        completed = 0
        for future in as_completed(futures):
            source_id, payload = future.result()
            payloads[source_id] = payload
            completed += 1
            if completed % 100 == 0 or completed == len(futures):
                print(f"event_stats {completed}/{len(futures)}", flush=True)

    audit = build_audit(sample, payloads)
    coverage = build_coverage(audit)
    fields = build_field_catalog(payloads)
    decision = decide(audit, coverage, fields)
    audit.to_csv(output / "sample_event_stats_audit.csv", index=False)
    coverage.to_csv(output / "coverage_by_country_era.csv", index=False)
    fields.to_csv(output / "field_catalog.csv", index=False)
    (output / "decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "quality_report.md").write_text(
        build_report(decision, coverage), encoding="utf-8"
    )
    print(f"Decision: {decision['decision']}", flush=True)
    print(f"Output: {output}", flush=True)


def stratified_sample(matches: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    required = {"source_event_id", "country_code", "ao_season", "sportsdb_league_id"}
    missing = sorted(required - set(matches.columns))
    if missing:
        raise ValueError(f"domestic matches missing columns: {missing}")
    data = matches.copy()
    if data["source_event_id"].isna().any() or data["source_event_id"].duplicated().any():
        raise ValueError("source_event_id must be non-null and unique")
    start_year = pd.to_numeric(data["ao_season"].astype(str).str[:4], errors="coerce")
    if start_year.isna().any():
        raise ValueError("ao_season must start with a four-digit year")
    data["era"] = pd.cut(
        start_year,
        bins=[2012, 2017, 2021, 2026],
        labels=["2013_2017", "2018_2021", "2022_2026"],
    ).astype(str)
    groups = list(data.groupby(["country_code", "era"], sort=True))
    per_group = max(1, sample_size // len(groups))
    selected = []
    for _, group in groups:
        selected.append(group.sample(min(per_group, len(group)), random_state=20260812))
    result = pd.concat(selected, ignore_index=True)
    if len(result) < sample_size:
        remaining = data[~data["source_event_id"].isin(result["source_event_id"])]
        result = pd.concat(
            [result, remaining.sample(sample_size - len(result), random_state=20260813)],
            ignore_index=True,
        )
    elif len(result) > sample_size:
        result = result.sample(sample_size, random_state=20260814)
    return result.sort_values(["country_code", "era", "source_event_id"], kind="stable").reset_index(drop=True)


def build_audit(sample: pd.DataFrame, payloads: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for match in sample.itertuples(index=False):
        source_id = str(match.source_event_id)
        records = payload_rows(payloads[source_id])
        slugs = {stat_slug(record.get("strStat")) for record in records if record.get("strStat")}
        conflicting_ids = {
            str(record.get("idEvent"))
            for record in records
            if record.get("idEvent") not in (None, "") and str(record.get("idEvent")) != source_id
        }
        values = {
            stat_slug(record.get("strStat")): (record.get("intHome"), record.get("intAway"))
            for record in records
            if record.get("strStat")
        }
        xg_values = values.get("expected_goals")
        shots = values.get("total_shots")
        xg_placeholder = False
        if xg_values is not None:
            try:
                home_xg, away_xg = float(xg_values[0]), float(xg_values[1])
                home_shots, away_shots = (
                    (float(shots[0]), float(shots[1])) if shots is not None else (0.0, 0.0)
                )
                xg_placeholder = (
                    (home_xg == 0.0 and home_shots > 0.0)
                    or (away_xg == 0.0 and away_shots > 0.0)
                    or (home_xg == 0.0 and away_xg == 0.0)
                )
            except (TypeError, ValueError):
                xg_placeholder = True
        rows.append(
            {
                "source_event_id": source_id,
                "country_code": match.country_code,
                "ao_season": match.ao_season,
                "era": match.era,
                "sportsdb_league_id": match.sportsdb_league_id,
                "identity_match": not conflicting_ids,
                "stats_covered": bool(records),
                "core_shots_covered": CORE_STATS.issubset(slugs),
                "xg_present": xg_values is not None,
                "xg_placeholder_suspected": xg_placeholder,
                "stat_field_count": len(slugs),
            }
        )
    return pd.DataFrame(rows)


def build_coverage(audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in audit.groupby(["country_code", "era"], sort=True):
        rows.append(
            {
                "country_code": keys[0],
                "era": keys[1],
                "sample_matches": len(group),
                "identity_rate": float(group["identity_match"].mean()),
                "stats_coverage_rate": float(group["stats_covered"].mean()),
                "core_shots_coverage_rate": float(group["core_shots_covered"].mean()),
                "xg_coverage_rate": float(group["xg_present"].mean()),
                "xg_placeholder_rate_among_present": float(
                    group.loc[group["xg_present"], "xg_placeholder_suspected"].mean()
                )
                if group["xg_present"].any()
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_field_catalog(payloads: dict[str, dict]) -> pd.DataFrame:
    counts: dict[str, int] = {}
    for payload in payloads.values():
        for record in payload_rows(payload):
            if record.get("strStat"):
                slug = stat_slug(record["strStat"])
                counts[slug] = counts.get(slug, 0) + 1
    return pd.DataFrame(
        [{"stat_field": field, "matches_or_records": count} for field, count in sorted(counts.items())]
    )


def decide(audit: pd.DataFrame, coverage: pd.DataFrame, fields: pd.DataFrame) -> dict[str, object]:
    identity_rate = float(audit["identity_match"].mean())
    core_rate = float(audit["core_shots_covered"].mean())
    xg_present = audit[audit["xg_present"]]
    placeholder_rate = (
        float(xg_present["xg_placeholder_suspected"].mean()) if not xg_present.empty else 1.0
    )
    covered_eras = coverage.groupby("era")["core_shots_coverage_rate"].mean()
    semantics_consistent = bool(
        CORE_STATS.issubset(set(fields["stat_field"]))
        and len(covered_eras) == 3
        and (covered_eras > 0.0).all()
    )
    gates = {
        "identity_rate_gte_0_95": identity_rate >= 0.95,
        "core_stats_coverage_gte_0_60": core_rate >= 0.60,
        "xg_placeholder_rate_lte_0_15": placeholder_rate <= 0.15,
        "field_semantics_consistent": semantics_consistent,
    }
    passed = all(gates.values())
    return {
        "decision": "PROCEED_TO_TARGETED_COLLECTION" if passed else "STOP_RICH_STATS_COLLECTION",
        "sample_matches": len(audit),
        "identity_rate": identity_rate,
        "core_stats_coverage_rate": core_rate,
        "xg_coverage_rate": float(audit["xg_present"].mean()),
        "xg_placeholder_rate_among_present": placeholder_rate,
        "gates": gates,
        "automatic_production_activation": False,
    }


def build_report(decision: dict[str, object], coverage: pd.DataFrame) -> str:
    return f"""# TheSportsDB Tarihsel İstatistik Kalite Pilotu

## Karar

**{decision['decision']}**

- Örnek maç: {decision['sample_matches']}
- Kimlik doğruluğu: {decision['identity_rate']:.1%}
- Core şut kapsamı: {decision['core_stats_coverage_rate']:.1%}
- xG kapsamı: {decision['xg_coverage_rate']:.1%}
- Mevcut xG içindeki şüpheli placeholder: {decision['xg_placeholder_rate_among_present']:.1%}

Bu pilot yalnız çok sezonlu zengin feature toplamaya başlanıp başlanmayacağını
belirler. Sonuçlar AO Elo veya ML production katmanını otomatik değiştirmez.

## Dönem ve ülke kapsamı

`coverage_by_country_era.csv` her ülke ve dönem için kimlik, şut ve xG kapsamını
ayrı gösterir. Kapsam eşiğini geçmeyen strata sonraki toplamada dışarıda bırakılır.
"""


if __name__ == "__main__":
    main()
