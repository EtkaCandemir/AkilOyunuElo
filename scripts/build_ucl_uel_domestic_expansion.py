from __future__ import annotations

import argparse
import hashlib
import json
import math
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
    UCL_UEL_EXPANSION_LEAGUES,
    DomesticFixtureConflictError,
    FIXTURE_RECONCILIATIONS,
    FIXTURE_AUDIT_COLUMNS,
    assess_league_season,
    attach_domestic_club_ids,
    build_domestic_team_bridge,
    normalize_schedule,
    validate_domestic_dataset,
)
from ao_elo.domestic_ucl_uel_expansion import (  # noqa: E402
    apply_verified_target_aliases,
    assess_secondary_league_seasons,
    build_target_team_audit,
    canonicalize_candidate_domestic_state,
    merge_domestic_candidate,
    normalize_secondary_fixtures,
    select_source_safe_seasons,
    target_domestic_coverage_audit,
)
from ao_elo.validators import require_utc_column  # noqa: E402
from scripts.build_domestic_league_dataset import (  # noqa: E402
    RequestLimiter,
    SportsDbClient,
    load_api_key,
    sha256_file,
    stable_json_dump,
)


BASE_ROOT = ROOT / "data" / "domestic_league_matches_2013_2026"
OUTPUT_ROOT = ROOT / "data" / "domestic_league_expansion_ucl_uel"
REGISTRY_PATH = ROOT / "data" / "club_identity" / "club_registry.csv"
PREPRODUCTION = ROOT / "data" / "season_2026_27_preproduction"
ENV_PATH = ROOT / ".env.local"
FROZEN_ACCEPTED_COVERAGE_GAPS = frozenset({("GEO", "2014"), ("LIT", "2020")})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a candidate UCL/UEL domestic-Poisson coverage expansion"
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--base-root", type=Path, default=BASE_ROOT)
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--preproduction-root", type=Path, default=PREPRODUCTION)
    parser.add_argument("--env-file", type=Path, default=ENV_PATH)
    parser.add_argument("--start-year", type=int, default=2013)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--request-delay", type=float, default=0.62)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--secondary-cache",
        type=Path,
        default=OUTPUT_ROOT / "_secondary_cache",
        help="Hugging Face/API-Football fixture archive cache directory",
    )
    parser.add_argument("--minimum-matches", type=int, default=40)
    parser.add_argument("--minimum-seasons", type=int, default=2)
    args = parser.parse_args()
    if args.start_year > args.end_year:
        raise ValueError("start-year must not exceed end-year")
    if args.request_delay < 0.60:
        raise ValueError("request-delay must be at least 0.60 seconds")

    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "_source_cache"
    cache.mkdir(parents=True, exist_ok=True)

    preproduction = args.preproduction_root.resolve()
    teams = pd.read_csv(preproduction / "teams.csv", low_memory=False)
    context = pd.read_csv(preproduction / "domestic_context.csv", low_memory=False)
    fixtures = pd.read_csv(preproduction / "fixtures_upcoming.csv", low_memory=False)
    targets = build_target_team_audit(teams, context, fixtures)
    if len(targets) != 80:
        raise ValueError(f"Expected frozen UCL/UEL target universe of 80 teams, found {len(targets)}")

    api_key = load_api_key(args.env_file.resolve())
    client = SportsDbClient(api_key, RequestLimiter(args.request_delay))
    quality_rows: list[dict[str, object]] = []
    frames: list[pd.DataFrame] = []
    fixture_audit: list[dict[str, object]] = []
    total = len(UCL_UEL_EXPANSION_LEAGUES) * (args.end_year - args.start_year + 1)
    completed = 0
    for spec in UCL_UEL_EXPANSION_LEAGUES:
        for year in range(args.start_year, args.end_year + 1):
            provider_season = spec.provider_season(year)
            key = provider_season.replace("/", "-")
            try:
                schedule = client.fetch_v2(
                    f"/schedule/league/{spec.sportsdb_league_id}/{provider_season}",
                    cache / "schedule" / spec.country_code / f"{key}.json",
                    refresh=args.refresh,
                    offline=args.offline,
                )
                table = client.fetch_table(
                    spec.sportsdb_league_id,
                    provider_season,
                    cache / "table" / spec.country_code / f"{key}.json",
                    refresh=args.refresh,
                    offline=args.offline,
                )
                matches = normalize_schedule(schedule, spec, provider_season, reconciliation_audit=fixture_audit)
                expected = _table_expected_matches(table)
                assessment = assess_league_season(
                    matches,
                    spec=spec,
                    provider_season=provider_season,
                    expected_matches=expected,
                )
                if (
                    assessment["quality_status"] == "ACCEPTED"
                    and (spec.country_code, str(provider_season))
                    not in FROZEN_ACCEPTED_COVERAGE_GAPS
                ):
                    frames.append(
                        matches.assign(
                            source_provider="THESPORTSDB_PREMIUM",
                            source_url=(
                                "https://www.thesportsdb.com/api/v2/json/"
                                f"schedule/league/{spec.sportsdb_league_id}/{provider_season}"
                            ),
                        )
                    )
                assessment["source_provider"] = "THESPORTSDB_PREMIUM"
                assessment["source_selection"] = (
                    "PRIMARY_ACCEPTED" if assessment["quality_status"] == "ACCEPTED" else "SECONDARY_REQUIRED"
                )
            except DomesticFixtureConflictError:
                raise
            except Exception as exc:  # noqa: BLE001 - source failures are audited data
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
                    "quality_reason": f"FETCH_OR_PARSE_ERROR:{type(exc).__name__}",
                    "error_detail": str(exc),
                    "source_provider": "THESPORTSDB_PREMIUM",
                    "source_selection": "SECONDARY_REQUIRED",
                }
            if (spec.country_code, str(provider_season)) in FROZEN_ACCEPTED_COVERAGE_GAPS:
                assessment["quality_status"] = "REJECTED"
                assessment["quality_reason"] = "FROZEN_ACCEPTED_COVERAGE_GAP"
                assessment["source_selection"] = "UNAVAILABLE"
            quality_rows.append(assessment)
            completed += 1
            if completed % 10 == 0 or completed == total:
                print(f"Expansion league seasons: {completed}/{total}", flush=True)

    quality = pd.DataFrame(quality_rows).sort_values(
        ["country_code", "provider_season"], kind="stable"
    ).reset_index(drop=True)
    primary_expansion = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if primary_expansion.empty:
        raise ValueError("No primary expansion league-seasons passed the quality gate")
    primary_expansion = primary_expansion.sort_values(["kickoff_utc", "match_id"], kind="stable").reset_index(drop=True)

    secondary_root = args.secondary_cache.resolve()
    secondary_fixtures_path = secondary_root / "fixtures.parquet"
    secondary_teams_path = secondary_root / "teams.parquet"
    if not secondary_fixtures_path.is_file() or not secondary_teams_path.is_file():
        raise FileNotFoundError(
            "Secondary archive cache is required; expected fixtures.parquet and teams.parquet in "
            f"{secondary_root}"
        )
    secondary_fixtures = pd.read_parquet(secondary_fixtures_path)
    secondary_teams = pd.read_parquet(secondary_teams_path)
    secondary_frames: list[pd.DataFrame] = []
    secondary_quality_frames: list[pd.DataFrame] = []
    for spec in UCL_UEL_EXPANSION_LEAGUES:
        normalized = normalize_secondary_fixtures(
            secondary_fixtures,
            secondary_teams,
            spec,
            start_year=args.start_year,
            end_year=args.end_year,
        )
        if normalized.empty:
            continue
        normalized = normalized.assign(
            source_provider="HF_API_FOOTBALL_ARCHIVE",
            source_url="https://huggingface.co/datasets/eatpizzanot/soccer-dataset",
        )
        secondary_frames.append(normalized)
        secondary_quality_frames.append(assess_secondary_league_seasons(normalized, spec))
    secondary_expansion = (
        pd.concat(secondary_frames, ignore_index=True, sort=False)
        if secondary_frames
        else pd.DataFrame(columns=primary_expansion.columns)
    )
    secondary_quality = (
        pd.concat(secondary_quality_frames, ignore_index=True, sort=False)
        if secondary_quality_frames
        else pd.DataFrame(columns=quality.columns)
    )
    expansion, selected_quality = select_source_safe_seasons(
        primary_expansion,
        quality,
        secondary_expansion,
        secondary_quality,
    )
    if expansion.empty:
        raise ValueError("No expansion league-season passed either source quality gate")

    registry = pd.read_csv(args.registry.resolve(), low_memory=False)
    expansion_bridge = build_domestic_team_bridge(expansion, registry)
    expansion = attach_domestic_club_ids(expansion, expansion_bridge)
    accepted_quality = selected_quality.loc[selected_quality["quality_status"].eq("ACCEPTED")].copy()
    validate_domestic_dataset(expansion, accepted_quality, expansion_bridge)

    base_root = args.base_root.resolve()
    base_matches = pd.read_csv(base_root / "domestic_matches.csv", low_memory=False)
    base_bridge = pd.read_csv(base_root / "domestic_team_bridge.csv", low_memory=False)
    base_matches = base_matches.assign(source_provider="THESPORTSDB_PREMIUM_EXISTING")
    candidate_bridge = _merge_bridges(base_bridge, expansion_bridge)
    candidate_bridge, alias_audit = apply_verified_target_aliases(candidate_bridge, targets)
    canonical_base, _ = canonicalize_candidate_domestic_state(
        base_matches,
        candidate_bridge,
        source_switch_countries=(spec.country_code for spec in UCL_UEL_EXPANSION_LEAGUES),
    )
    canonical_expansion, candidate_bridge = canonicalize_candidate_domestic_state(
        expansion,
        candidate_bridge,
        source_switch_countries=(spec.country_code for spec in UCL_UEL_EXPANSION_LEAGUES),
    )
    canonical_expansion, base_overlap_audit = _prefer_base_for_identical_overlap(
        canonical_base,
        canonical_expansion,
    )
    candidate = merge_domestic_candidate(canonical_base, canonical_expansion)
    candidate = candidate.drop(
        columns=["home_ao_club_id", "away_ao_club_id"], errors="ignore"
    )
    candidate = attach_domestic_club_ids(candidate, candidate_bridge)
    candidate_quality = pd.concat(
        [
            pd.read_csv(base_root / "league_season_quality.csv", low_memory=False).assign(source_selection="EXISTING_ACCEPTED"),
            selected_quality,
        ],
        ignore_index=True,
        sort=False,
    )

    before = target_domestic_coverage_audit(
        targets,
        base_bridge,
        base_matches,
        minimum_matches=args.minimum_matches,
        minimum_seasons=args.minimum_seasons,
    ).rename(columns={
        "candidate_state_eligible": "baseline_state_eligible",
        "coverage_status": "baseline_coverage_status",
        "accepted_domestic_matches": "baseline_domestic_matches",
        "accepted_domestic_seasons": "baseline_domestic_seasons",
    })
    after = target_domestic_coverage_audit(
        targets,
        candidate_bridge,
        candidate,
        minimum_matches=args.minimum_matches,
        minimum_seasons=args.minimum_seasons,
    ).rename(columns={
        "candidate_state_eligible": "candidate_state_eligible",
        "coverage_status": "candidate_coverage_status",
        "accepted_domestic_matches": "candidate_domestic_matches",
        "accepted_domestic_seasons": "candidate_domestic_seasons",
    })
    audit = before[[
        "ao_club_id", "baseline_state_eligible", "baseline_coverage_status",
        "baseline_domestic_matches", "baseline_domestic_seasons",
    ]].merge(after, on="ao_club_id", validate="one_to_one")
    audit = audit.merge(
        targets[["ao_club_id", "is_expansion_target_country"]],
        on="ao_club_id",
        validate="one_to_one",
    )
    audit["coverage_transition"] = (
        audit["baseline_coverage_status"] + "->" + audit["candidate_coverage_status"]
    )

    _write_csv(targets, output / "target_team_audit.csv")
    _write_csv(_league_registry_frame(), output / "league_registry.csv")
    _write_csv(primary_expansion, output / "expansion_matches_primary.csv")
    _write_csv(pd.DataFrame(fixture_audit, columns=FIXTURE_AUDIT_COLUMNS), output / "fixture_reconciliation_audit.csv")
    _write_csv(base_overlap_audit, output / "base_expansion_overlap_audit.csv")
    _write_csv(secondary_expansion, output / "expansion_matches_secondary.csv")
    _write_csv(expansion, output / "expansion_matches_selected.csv")
    _write_csv(expansion_bridge, output / "expansion_team_bridge.csv")
    _write_csv(candidate, output / "domestic_matches_candidate.csv")
    _write_csv(candidate_bridge, output / "domestic_team_bridge_candidate.csv")
    _write_csv(alias_audit, output / "target_identity_alias_audit.csv")
    _write_csv(candidate_quality, output / "league_season_quality.csv")
    _write_csv(audit, output / "target_coverage_audit.csv")
    _write_manifest(output, args, targets, expansion, candidate, selected_quality, audit)

    required = audit.loc[audit["is_expansion_target_country"].astype(bool)]
    missing = required.loc[~required["candidate_state_eligible"].astype(bool)]
    print(f"Target teams: {len(targets)}; candidate covered: {int(audit['candidate_state_eligible'].sum())}")
    print(
        f"Expansion matches: {len(expansion)} "
        f"(primary={len(primary_expansion)}, secondary cache={len(secondary_expansion)}); "
        f"candidate matches: {len(candidate)}"
    )
    print(f"Target expansion coverage: {len(required) - len(missing)}/{len(required)}")
    if not missing.empty:
        print("Secondary provider required for:", ", ".join(missing["team_name"].astype(str).tolist()))
    print(f"Output: {output}")


def _table_expected_matches(payload: dict[str, object]) -> int | None:
    rows = next((value for value in payload.values() if isinstance(value, list)), [])
    played: list[int] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            value = int(float(str(row.get("intPlayed", ""))))
        except (TypeError, ValueError):
            continue
        if value >= 0:
            played.append(value)
    total = sum(played)
    return total // 2 if total > 0 and total % 2 == 0 else None


def _merge_bridges(existing: pd.DataFrame, expansion: pd.DataFrame) -> pd.DataFrame:
    result = pd.concat([existing, expansion], ignore_index=True, sort=False)
    key = ["country_code", "source_team_id"]
    duplicated = result.duplicated(key, keep=False)
    if duplicated.any():
        values = result.loc[duplicated].sort_values(key, kind="stable")
        for _, group in values.groupby(key, sort=False):
            if group["ao_club_id"].fillna("").astype(str).nunique() > 1:
                raise ValueError("Provider identity bridge has conflicting AO club mappings")
        result = result.drop_duplicates(key, keep="first")
    return result.sort_values(key, kind="stable").reset_index(drop=True)


BASE_EXPANSION_OVERLAP_AUDIT_COLUMNS = (
    "match_id",
    "action",
    "reason",
    "source_event_id",
    "sportsdb_league_id",
    "country_code",
    "provider_season",
    "ao_season",
    "kickoff_utc",
    "home_source_team_id",
    "away_source_team_id",
    "home_goals",
    "away_goals",
)


def _prefer_base_for_identical_overlap(
    base: pd.DataFrame,
    expansion: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep one explicitly audited copy of the same provider observation.

    A league can be present in the canonical base and in the coverage expansion.
    The general merge guard must continue to reject duplicate match IDs.  Before
    reaching it, this source-selection step permits only a byte-equivalent state
    observation: provenance identity, canonical fixture and field score must all
    agree.  Any disagreement remains a hard error and no row is dropped.
    """

    required = {
        "match_id",
        "source_event_id",
        "sportsdb_league_id",
        "country_code",
        "provider_season",
        "ao_season",
        "kickoff_utc",
        "home_source_team_id",
        "away_source_team_id",
        "home_goals",
        "away_goals",
    }
    for label, frame in (("base", base), ("expansion", expansion)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{label} overlap input missing columns: {missing}")
        if frame["match_id"].astype(str).duplicated().any():
            raise ValueError(f"{label} overlap input has duplicate match_id values")

    overlap = sorted(
        set(base["match_id"].astype(str)).intersection(
            expansion["match_id"].astype(str)
        )
    )
    if not overlap:
        return (
            expansion.copy(),
            pd.DataFrame(columns=BASE_EXPANSION_OVERLAP_AUDIT_COLUMNS),
        )

    left = base.assign(match_id=base["match_id"].astype(str)).set_index("match_id").loc[overlap]
    right = (
        expansion.assign(match_id=expansion["match_id"].astype(str))
        .set_index("match_id")
        .loc[overlap]
    )
    compare_columns = sorted(required.difference({"match_id", "kickoff_utc"}))
    conflicts: dict[str, list[str]] = {}
    for column in compare_columns:
        equal = left[column].fillna("<NA>").astype(str).eq(
            right[column].fillna("<NA>").astype(str)
        )
        if not equal.all():
            conflicts[column] = equal.index[~equal].tolist()[:5]
    left_kickoff = require_utc_column(
        left["kickoff_utc"], "base overlap kickoff_utc"
    )
    right_kickoff = require_utc_column(
        right["kickoff_utc"], "expansion overlap kickoff_utc"
    )
    kickoff_equal = left_kickoff.eq(right_kickoff)
    if not kickoff_equal.all():
        conflicts["kickoff_utc"] = kickoff_equal.index[~kickoff_equal].tolist()[:5]
    if conflicts:
        raise ValueError(
            "base/expansion overlapping observations conflict: "
            + "; ".join(
                f"{column}={match_ids}" for column, match_ids in conflicts.items()
            )
        )

    audit = left.reset_index()[list(BASE_EXPANSION_OVERLAP_AUDIT_COLUMNS[3:])]
    audit.insert(0, "reason", "IDENTICAL_PROVIDER_OBSERVATION")
    audit.insert(0, "action", "KEEP_BASE_REMOVE_EXPANSION")
    audit.insert(0, "match_id", overlap)
    audit = audit[list(BASE_EXPANSION_OVERLAP_AUDIT_COLUMNS)]
    filtered = expansion.loc[~expansion["match_id"].astype(str).isin(overlap)].copy()
    return filtered.reset_index(drop=True), audit.reset_index(drop=True)


def _league_registry_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "country_code": spec.country_code,
                "sportsdb_league_id": spec.sportsdb_league_id,
                "league_name": spec.league_name,
                "calendar_season": spec.calendar_season,
                "catalogue_verified_at": "2026-08-21",
                "provider": "THESPORTSDB_PREMIUM",
            }
            for spec in UCL_UEL_EXPANSION_LEAGUES
        ]
    )


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def _write_manifest(
    output: Path,
    args: argparse.Namespace,
    targets: pd.DataFrame,
    expansion: pd.DataFrame,
    candidate: pd.DataFrame,
    quality: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    payload = {
        "dataset": "AO UCL/UEL domestic Poisson candidate expansion",
        "target_snapshot": "2026-08-21",
        "target_teams": int(len(targets)),
        "expansion_leagues": [spec.country_code for spec in UCL_UEL_EXPANSION_LEAGUES],
        "period": {"start_year": args.start_year, "end_year": args.end_year},
        "provider": "TheSportsDB Premium API",
        "secondary_provider": "Hugging Face eatpizzanot/soccer-dataset (API-Football historical fixture archive)",
        "request_delay_seconds": args.request_delay,
        "quality_gate": "valid UTC/scores, unique events, final-table coverage >=95%",
        "fixture_reconciliations_sha256": sha256_file(FIXTURE_RECONCILIATIONS),
        "accepted_primary_league_seasons": int(quality["quality_status"].eq("ACCEPTED").sum()),
        "expansion_matches": int(len(expansion)),
        "candidate_matches": int(len(candidate)),
        "candidate_target_covered": int(audit["candidate_state_eligible"].sum()),
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
        "secondary_archive_sha256": {
            path.name: sha256_file(path)
            for path in sorted((output / "_secondary_cache").glob("*.parquet"))
            if path.is_file()
        },
    }
    (output / "source_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
