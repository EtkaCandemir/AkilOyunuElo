from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ao_elo.club_identity import (  # noqa: E402
    attach_match_club_ids,
    permanent_club_id,
    validate_club_registry,
    validate_team_season_identity,
)


STATIC_ROOT = ROOT / "data" / "backtest_stage_b_2018_2026"
EVENTS_PATH = ROOT / "data" / "external_elo_benchmark_2018_2026" / "exact_date_events.csv"
UEFA_AUDIT_PATH = ROOT / "data" / "external_elo_benchmark_2018_2026" / "uefa_team_identity_audit.csv"
OUTPUT_ROOT = ROOT / "data" / "club_identity"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the permanent AO club identity registry")
    parser.add_argument("--static-root", type=Path, default=STATIC_ROOT)
    parser.add_argument("--events-path", type=Path, default=EVENTS_PATH)
    parser.add_argument("--uefa-audit-path", type=Path, default=UEFA_AUDIT_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    team_season = build_team_season_identity(
        args.static_root.resolve(), args.uefa_audit_path.resolve()
    )
    registry = build_registry(team_season)
    events = pd.read_csv(
        args.events_path.resolve(),
        dtype={"uefa_home_team_id": "string", "uefa_away_team_id": "string"},
    )
    matches = build_match_identity(events, team_season)
    audit = identity_audit(team_season, registry, matches, events)
    failed = audit.loc[audit["status"].ne("PASS")]
    if not failed.empty:
        raise ValueError(
            "Permanent club identity audit failed: "
            f"{failed.to_dict(orient='records')}"
        )

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    registry.to_csv(output_root / "club_registry.csv", index=False)
    team_season.to_csv(output_root / "team_season_identity.csv", index=False)
    matches.to_csv(output_root / "match_identity.csv", index=False)
    audit.to_csv(output_root / "identity_audit.csv", index=False)
    write_manifest(output_root / "identity_manifest.json", registry, team_season, matches, audit)
    write_readme(output_root / "README.md", registry, team_season, matches, audit)
    print("AO permanent club identity registry built")
    print(f"Clubs: {len(registry)}")
    print(f"Team-seasons: {len(team_season)}")
    print(f"Matches: {len(matches)}")
    print(f"Output: {output_root}")


def build_team_season_identity(static_root: Path, audit_path: Path) -> pd.DataFrame:
    audit = pd.read_csv(audit_path, dtype={"uefa_team_id": "string"})
    if audit["uefa_team_id"].isna().any():
        raise ValueError("Every team-season must have a verified UEFA team ID")
    rows = []
    for folder in sorted(static_root.glob("20??-??")):
        season = folder.name.replace("-", "/")
        teams = pd.read_csv(folder / "teams.csv")
        rows.append(teams.assign(season=season))
    source = pd.concat(rows, ignore_index=True)
    result = source.merge(
        audit,
        left_on=["season", "team_id"],
        right_on=["season", "local_team_id"],
        validate="one_to_one",
        suffixes=("", "_audit"),
    )
    if len(result) != len(source) or len(result) != len(audit):
        raise ValueError("Static teams and UEFA identity audit do not have full coverage")
    if not result["team_name"].eq(result["ao_team_name"]).all():
        raise ValueError("Static team name and UEFA audit team name disagree")
    if not result["country_code"].eq(result["country_code_audit"]).all():
        raise ValueError("Static country and UEFA audit country disagree")
    result["club_id"] = result["uefa_team_id"].map(permanent_club_id)
    result["identity_status"] = "VERIFIED"
    result["event_identity_verified"] = True
    result = result[
        [
            "season",
            "team_id",
            "team_name",
            "country",
            "country_code",
            "club_id",
            "uefa_team_id",
            "uefa_team_name",
            "resolution_method",
            "similarity",
            "runner_up_similarity",
            "identity_status",
            "event_identity_verified",
        ]
    ].rename(columns={"team_id": "local_team_id"})
    validate_team_season_identity(result)
    return result.sort_values(["season", "local_team_id"], kind="stable").reset_index(drop=True)


def build_registry(team_season: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for club_id, group in team_season.groupby("club_id", sort=True):
        canonical_names = group["uefa_team_name"].dropna().astype(str).unique()
        countries = group["country_code"].dropna().astype(str).unique()
        uefa_ids = group["uefa_team_id"].dropna().astype(str).unique()
        if len(canonical_names) != 1 or len(countries) != 1 or len(uefa_ids) != 1:
            raise ValueError(f"{club_id}: canonical identity is not unique")
        names = sorted(set(group["team_name"].astype(str)))
        methods = sorted(set(group["resolution_method"].astype(str)))
        seasons = sorted(set(group["season"].astype(str)))
        rows.append(
            {
                "club_id": club_id,
                "uefa_team_id": uefa_ids[0],
                "canonical_name": canonical_names[0],
                "country_code": countries[0],
                "first_season": seasons[0],
                "last_season": seasons[-1],
                "seasons_observed": len(seasons),
                "observed_names": " | ".join(names),
                "name_variant_count": len(names),
                "resolution_methods": " | ".join(methods),
                "identity_status": "VERIFIED",
            }
        )
    registry = pd.DataFrame(rows).sort_values("club_id", kind="stable").reset_index(drop=True)
    validate_club_registry(registry)
    return registry


def build_match_identity(events: pd.DataFrame, team_season: pd.DataFrame) -> pd.DataFrame:
    attached = attach_match_club_ids(events, team_season)
    expected_home = attached["uefa_home_team_id"].map(permanent_club_id)
    expected_away = attached["uefa_away_team_id"].map(permanent_club_id)
    attached["home_identity_verified"] = attached["home_club_id"].eq(expected_home)
    attached["away_identity_verified"] = attached["away_club_id"].eq(expected_away)
    if not attached[["home_identity_verified", "away_identity_verified"]].all().all():
        raise ValueError("Match UEFA IDs and permanent club IDs disagree")
    return attached[
        [
            "match_id",
            "season",
            "event_order",
            "competition",
            "home_team_id",
            "home_team_name",
            "home_club_id",
            "uefa_home_team_id",
            "home_identity_verified",
            "away_team_id",
            "away_team_name",
            "away_club_id",
            "uefa_away_team_id",
            "away_identity_verified",
        ]
    ].sort_values(["season", "event_order"], kind="stable").reset_index(drop=True)


def identity_audit(
    team_season: pd.DataFrame,
    registry: pd.DataFrame,
    matches: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    checks = [
        check("team_season_complete", len(team_season) == 1887, len(team_season)),
        check("registry_unique_clubs", len(registry) == registry["club_id"].nunique(), len(registry)),
        check(
            "season_local_key_unique",
            not team_season.duplicated(["season", "local_team_id"]).any(),
            len(team_season),
        ),
        check(
            "season_club_key_unique",
            not team_season.duplicated(["season", "club_id"]).any(),
            len(team_season),
        ),
        check(
            "club_country_stable",
            not team_season.groupby("club_id")["country_code"].nunique().gt(1).any(),
            team_season["club_id"].nunique(),
        ),
        check("match_count_complete", len(matches) == len(events) == 6340, len(matches)),
        check("match_id_unique", matches["match_id"].is_unique, matches["match_id"].nunique()),
        check(
            "match_provider_identity_agrees",
            bool(matches[["home_identity_verified", "away_identity_verified"]].all().all()),
            len(matches),
        ),
        check(
            "no_self_matches",
            not matches["home_club_id"].eq(matches["away_club_id"]).any(),
            len(matches),
        ),
        check(
            "name_changes_preserve_identity",
            int((team_season.groupby("club_id")["team_name"].nunique() > 1).sum()) == 7,
            int((team_season.groupby("club_id")["team_name"].nunique() > 1).sum()),
        ),
    ]
    return pd.DataFrame(checks)


def check(name: str, passed: bool, detail: object) -> dict[str, object]:
    return {"check": name, "status": "PASS" if passed else "FAIL", "detail": detail}


def write_manifest(
    path: Path,
    registry: pd.DataFrame,
    team_season: pd.DataFrame,
    matches: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    variants = registry.loc[registry["name_variant_count"].gt(1), ["club_id", "observed_names"]]
    payload = {
        "identity_version": "ao-club-identity-v1",
        "club_id_contract": "AO-UEFA-<positive UEFA team ID>",
        "provider": "UEFA",
        "clubs": len(registry),
        "team_seasons": len(team_season),
        "matches": len(matches),
        "name_variant_clubs": variants.to_dict(orient="records"),
        "all_checks_passed": bool(audit["status"].eq("PASS").all()),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_readme(
    path: Path,
    registry: pd.DataFrame,
    team_season: pd.DataFrame,
    matches: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    text = f"""# AO Permanent Club Identity Registry

`club_id` is the stable cross-season identifier. Local `team_id` values remain valid only
inside their own season and must never be used for cross-season joins.

Contract: `AO-UEFA-<positive UEFA team ID>`.

- Verified clubs: {len(registry)}
- Verified team-seasons: {len(team_season)}
- Verified matches: {len(matches)}
- Audit checks passed: {int(audit['status'].eq('PASS').sum())}/{len(audit)}

`team_season_identity.csv` is the mandatory bridge from `season + local_team_id` to
`club_id`. `club_registry.csv` stores one canonical row per club. `match_identity.csv`
proves both event sides agree with UEFA's provider IDs.
"""
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
