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
            normalized = normalize_schedule(payload, spec, provider_season)
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

    live = pd.concat(live_frames, ignore_index=True, sort=False) if live_frames else pd.DataFrame()
    if live.empty:
        raise ValueError("No 2026/27 completed domestic results were retrieved")
    live = live.sort_values(["kickoff_utc", "match_id"], kind="stable").reset_index(drop=True)
    live_bridge = build_domestic_team_bridge(live, registry)
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
    result = pd.concat([existing, extra], ignore_index=True, sort=False)
    key = ["country_code", "source_team_id"]
    for _, group in result.loc[result.duplicated(key, keep=False)].groupby(key, sort=False):
        clubs = group["ao_club_id"].fillna("").astype(str)
        if clubs[clubs.ne("")].nunique() > 1:
            raise ValueError("Live domestic provider identity conflicts with candidate bridge")
    return result.drop_duplicates(key, keep="first").sort_values(key, kind="stable").reset_index(drop=True)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


if __name__ == "__main__":
    main()
