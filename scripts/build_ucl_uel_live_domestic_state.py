from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ao_elo.domestic_league_dataset import (  # noqa: E402
    PILOT_LEAGUES,
    DomesticFixtureConflictError,
    FIXTURE_AUDIT_COLUMNS,
    UCL_UEL_EXPANSION_LEAGUES,
    attach_domestic_club_ids,
    build_domestic_team_bridge,
    normalize_schedule,
)
from ao_elo.domestic_poisson import (  # noqa: E402
    DomesticPoissonConfig,
    build_domestic_club_mapping,
    replay_domestic_poisson_state,
)
from ao_elo.domestic_ucl_uel_expansion import (  # noqa: E402
    apply_verified_target_aliases,
    canonicalize_candidate_domestic_state,
    merge_domestic_candidate,
    select_causal_domestic_results,
)
from scripts.build_domestic_league_dataset import (  # noqa: E402
    RequestLimiter,
    SportsDbClient,
    load_api_key,
    sha256_file,
)


DATA_ROOT = ROOT / "data" / "domestic_league_expansion_ucl_uel"
REPORT_ROOT = ROOT / "reports" / "domestic_poisson_ucl_uel_expansion"
REGISTRY = ROOT / "data" / "club_identity" / "club_registry.csv"
ENV_PATH = ROOT / ".env.local"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a causal 2026/27 domestic-Poisson candidate state for UCL/UEL"
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--env-file", type=Path, default=ENV_PATH)
    parser.add_argument("--cutoff-utc", default="2026-08-21T23:59:59Z")
    parser.add_argument("--request-delay", type=float, default=0.62)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.request_delay < 0.60:
        raise ValueError("request-delay must be at least 0.60 seconds")

    data_root = args.data_root.resolve()
    report_root = args.report_root.resolve()
    report_root.mkdir(parents=True, exist_ok=True)
    candidate = pd.read_csv(data_root / "domestic_matches_candidate.csv", low_memory=False)
    bridge = pd.read_csv(data_root / "domestic_team_bridge_candidate.csv", low_memory=False)
    targets = pd.read_csv(data_root / "target_team_audit.csv", low_memory=False)
    registry = pd.read_csv(args.registry.resolve(), low_memory=False)
    cache = data_root / "_live_2026_27_cache"
    cache.mkdir(parents=True, exist_ok=True)
    api_key = load_api_key(args.env_file.resolve())
    client = SportsDbClient(api_key, RequestLimiter(args.request_delay))

    specs = {spec.country_code: spec for spec in (*PILOT_LEAGUES, *UCL_UEL_EXPANSION_LEAGUES)}
    live_frames: list[pd.DataFrame] = []
    fixture_audit: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for code, spec in sorted(specs.items()):
        provider_season = spec.provider_season(2026)
        try:
            payload = client.fetch_v2(
                f"/schedule/league/{spec.sportsdb_league_id}/{provider_season}",
                cache / code / f"{provider_season.replace('/', '-')}.json",
                refresh=args.refresh,
                offline=args.offline,
            )
            normalized = normalize_schedule(payload, spec, provider_season, reconciliation_audit=fixture_audit)
            normalized = select_causal_domestic_results(normalized, args.cutoff_utc)
            if not normalized.empty:
                live_frames.append(
                    normalized.assign(
                        source_provider="THESPORTSDB_PREMIUM_LIVE",
                        source_url=(
                            "https://www.thesportsdb.com/api/v2/json/"
                            f"schedule/league/{spec.sportsdb_league_id}/{provider_season}"
                        ),
                    )
                )
            audit_rows.append(
                {
                    "country_code": code,
                    "sportsdb_league_id": spec.sportsdb_league_id,
                    "provider_season": provider_season,
                    "completed_matches_before_cutoff": int(len(normalized)),
                    "status": "ACCEPTED" if not normalized.empty else "NO_COMPLETED_MATCHES",
                }
            )
        except DomesticFixtureConflictError:
            raise
        except Exception as exc:  # noqa: BLE001 - audited provider issue
            audit_rows.append(
                {
                    "country_code": code,
                    "sportsdb_league_id": spec.sportsdb_league_id,
                    "provider_season": provider_season,
                    "completed_matches_before_cutoff": 0,
                    "status": f"FETCH_OR_PARSE_ERROR:{type(exc).__name__}",
                    "error_detail": str(exc),
                }
            )

    pd.DataFrame(fixture_audit, columns=FIXTURE_AUDIT_COLUMNS).to_csv(data_root / "live_fixture_reconciliation_audit.csv", index=False)
    # A single failed league used to be recorded in the audit and skipped, so the
    # build still exited 0 and published a checkpoint that was quietly short of
    # that league's completed results.  Only a total failure stopped it.
    failed = [
        f"{row['country_code']} {row['provider_season']}: {row['status']}"
        for row in audit_rows
        if str(row["status"]).startswith("FETCH_OR_PARSE_ERROR")
    ]
    if failed:
        raise ValueError(
            "Refusing to write a domestic state with unresolved league failures: "
            + "; ".join(failed)
        )

    live = pd.concat(live_frames, ignore_index=True, sort=False) if live_frames else pd.DataFrame()
    if live.empty:
        raise ValueError("No 2026/27 completed domestic results were retrieved")
    live = live.sort_values(["kickoff_utc", "match_id"], kind="stable").reset_index(drop=True)
    live_bridge = build_domestic_team_bridge(live, registry)
    live_bridge = _restrict_live_bridge_to_production_universe(
        live_bridge,
        bridge,
        targets,
    )
    combined_bridge = _merge_bridges(bridge, live_bridge)
    combined_bridge, alias_audit = apply_verified_target_aliases(combined_bridge, targets)
    live, combined_bridge = canonicalize_candidate_domestic_state(
        live,
        combined_bridge,
        source_switch_countries=(spec.country_code for spec in UCL_UEL_EXPANSION_LEAGUES),
    )
    live = attach_domestic_club_ids(live, combined_bridge)
    all_matches = merge_domestic_candidate(candidate, live)
    config = DomesticPoissonConfig(0.02, 0.90, 10.0, False)
    engine = replay_domestic_poisson_state(all_matches, config)
    identity_map = build_domestic_club_mapping(all_matches, combined_bridge)

    _write_csv(live, data_root / "live_2026_27_matches.csv")
    _write_csv(combined_bridge, data_root / "domestic_team_bridge_with_live_candidate.csv")
    _write_csv(pd.DataFrame(audit_rows), data_root / "live_2026_27_quality.csv")
    _write_csv(alias_audit, data_root / "live_target_identity_alias_audit.csv")
    # The historical manifest also inventories live CSVs. Refresh it only after
    # all live outputs exist, otherwise a rebuild leaves stale bridge/audit hashes.
    manifest_path = data_root / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = {path.name: sha256_file(path) for path in sorted(data_root.glob("*.csv"))}
    manifest["live_cutoff_utc"] = pd.Timestamp(args.cutoff_utc).isoformat()
    manifest["live_raw_response_sha256"] = {
        str(path.relative_to(data_root)): sha256_file(path)
        for path in sorted(cache.rglob("*.json"))
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    state_path = report_root / "candidate_domestic_poisson_state_2026_27.json"
    state_path.write_text(
        json.dumps(
            {
                "artifact_version": "ao-domestic-poisson-state-v1-candidate",
                "status": "CANDIDATE_ONLY_NOT_PRODUCTION",
                "state_cutoff_utc": str(pd.to_datetime(args.cutoff_utc, utc=True).isoformat()),
                "config": config.__dict__,
                "historical_candidate_matches": int(len(candidate)),
                "causal_live_matches": int(len(live)),
                "total_matches": int(len(all_matches)),
                "identity_map": {
                    club_id: {"league_id": league_id, "source_team_id": team_id}
                    for club_id, (league_id, team_id) in sorted(identity_map.items())
                },
                "engine_state": engine.to_payload(),
                "input_sha256": {
                    "historical_candidate": sha256_file(data_root / "domestic_matches_candidate.csv"),
                    "live_results": sha256_file(data_root / "live_2026_27_matches.csv"),
                    "bridge": sha256_file(data_root / "domestic_team_bridge_with_live_candidate.csv"),
                },
            },
            indent=2,
            ensure_ascii=True,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Live completed domestic results: {len(live)}")
    print(f"Candidate state identities: {len(identity_map)}")
    print(f"Live data: {data_root / 'live_2026_27_matches.csv'}")
    print(f"Candidate state: {state_path}")


def _merge_bridges(existing: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    """Combine the candidate bridge with identities rebuilt from live schedules.

    `keep="first"` alone discards a newly resolved AO id whenever the candidate
    row for the same provider key has a null one: the conflict check only rejects
    two *different* non-null values, so old-null beat new-mapped and the identity
    was silently lost.  Conflicts still raise; agreement now coalesces.
    """

    result = pd.concat([existing, extra], ignore_index=True, sort=False)
    key = ["country_code", "source_team_id"]
    resolved: dict[tuple[object, ...], object] = {}
    for group_key, group in result.loc[result.duplicated(key, keep=False)].groupby(key, sort=False):
        clubs = group["ao_club_id"].fillna("").astype(str)
        present = clubs[clubs.ne("")]
        if present.nunique() > 1:
            raise ValueError("Live domestic provider identity conflicts with candidate bridge")
        if not present.empty:
            resolved[tuple(group_key) if isinstance(group_key, tuple) else (group_key,)] = present.iloc[0]
    merged = result.drop_duplicates(key, keep="first").copy()
    if resolved:
        merged["ao_club_id"] = [
            resolved.get((country, team_id), club)
            if (pd.isna(club) or not str(club).strip())
            else club
            for country, team_id, club in zip(
                merged["country_code"], merged["source_team_id"], merged["ao_club_id"], strict=True
            )
        ]
    return merged.sort_values(key, kind="stable").reset_index(drop=True)


def _restrict_live_bridge_to_production_universe(
    live_bridge: pd.DataFrame,
    existing_bridge: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    """Do not activate unaudited AO identities discovered in live schedules.

    Live results must still update league and opponent state, but the coverage
    audit only evaluates the explicit UCL/UEL target universe.  A registry name
    match outside that universe would otherwise become production-eligible
    without the two-season/40-match gate ever seeing it.  Existing mappings are
    retained and target mappings may still repair an old null bridge row.
    """

    required_bridge = {"ao_club_id", "identity_method"}
    missing = required_bridge.difference(live_bridge.columns)
    if missing:
        raise ValueError(f"Live bridge missing columns: {sorted(missing)}")
    if "ao_club_id" not in existing_bridge or "ao_club_id" not in targets:
        raise ValueError("Production identity universe requires ao_club_id columns")
    allowed = set(existing_bridge["ao_club_id"].dropna().astype(str)) | set(
        targets["ao_club_id"].dropna().astype(str)
    )
    result = live_bridge.copy()
    mapped = result["ao_club_id"].notna()
    outside = mapped & ~result["ao_club_id"].astype(str).isin(allowed)
    result.loc[outside, "ao_club_id"] = pd.NA
    result.loc[outside, "identity_method"] = "OUTSIDE_PRODUCTION_TARGET_UNIVERSE"
    return result


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


if __name__ == "__main__":
    main()
