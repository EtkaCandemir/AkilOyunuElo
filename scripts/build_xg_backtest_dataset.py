from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_backtest_dataset import normalize_name  # noqa: E402


EVENTS_PATH = (
    ROOT / "data" / "external_elo_benchmark_2018_2026"
    / "exact_date_events.csv"
)
OUTPUT_ROOT = ROOT / "data" / "xg_backtest_2018_2026"
LEAGUE_TO_COMPETITION = {89: "UCL", 90: "UEL", 91: "UECL"}
SOURCE_URLS = {
    "fixtures": (
        "https://huggingface.co/datasets/eatpizzanot/soccer-dataset/"
        "resolve/main/fixtures.parquet"
    ),
    "match_stats": (
        "https://huggingface.co/datasets/eatpizzanot/soccer-dataset/"
        "resolve/main/match_stats.parquet"
    ),
    "leagues": (
        "https://huggingface.co/datasets/eatpizzanot/soccer-dataset/"
        "resolve/main/leagues.parquet"
    ),
    "teams": (
        "https://huggingface.co/datasets/eatpizzanot/soccer-dataset/"
        "resolve/main/teams.parquet"
    ),
}
MIN_SIDE_SIMILARITY = 0.72
MIN_PAIR_SIMILARITY = 0.82
MIN_RUNNER_UP_GAP = 0.03


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the audited AO xG matched-sample dataset"
    )
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--match-stats", type=Path, required=True)
    parser.add_argument("--leagues", type=Path, required=True)
    parser.add_argument("--teams", type=Path, required=True)
    parser.add_argument("--events", type=Path, default=EVENTS_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    paths = {
        "fixtures": args.fixtures.resolve(),
        "match_stats": args.match_stats.resolve(),
        "leagues": args.leagues.resolve(),
        "teams": args.teams.resolve(),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise ValueError(f"{name} source file not found: {path}")

    events = read_events(args.events.resolve())
    source, source_quality = read_source(paths)
    matched, audit = match_xg_to_ao(events, source)
    coverage = build_coverage_matrix(events, matched)
    source_coverage = build_source_coverage(source)
    validate_output(events, matched, audit)

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    matched.to_csv(output_root / "xg_matches.csv", index=False)
    audit.to_csv(output_root / "identity_audit.csv", index=False)
    coverage.to_csv(output_root / "coverage_matrix.csv", index=False)
    source_coverage.to_csv(
        output_root / "source_coverage_matrix.csv",
        index=False,
    )
    manifest = build_manifest(paths, source_quality, events, matched)
    (output_root / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_root / "README.md").write_text(
        build_readme(manifest, coverage),
        encoding="utf-8",
    )

    eligible = int(matched["eligible_for_ablation"].sum())
    print(f"AO matches: {len(events)}")
    print(f"Source xG rows: {len(source)}")
    print(f"Matched xG rows: {len(matched)}")
    print(f"Eligible ablation rows: {eligible}")
    print(f"Output: {output_root}")


def read_events(path: Path) -> pd.DataFrame:
    events = pd.read_csv(path)
    required = {
        "match_id",
        "season",
        "competition",
        "kickoff_utc",
        "home_team_id",
        "away_team_id",
        "home_team_name",
        "away_team_name",
        "home_goals",
        "away_goals",
        "decided_on_penalties",
    }
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"events missing columns: {missing}")
    if events["match_id"].isna().any() or events["match_id"].duplicated().any():
        raise ValueError("events.match_id must be non-null and unique")
    events = events.copy()
    events["kickoff_utc"] = pd.to_datetime(
        events["kickoff_utc"],
        utc=True,
        errors="coerce",
    )
    if events["kickoff_utc"].isna().any():
        raise ValueError("events.kickoff_utc contains invalid timestamps")
    if not set(events["competition"]).issubset({"UCL", "UEL", "UECL"}):
        raise ValueError("events contains an unknown competition")
    return events.sort_values(["kickoff_utc", "match_id"]).reset_index(drop=True)


def read_source(
    paths: dict[str, Path],
) -> tuple[pd.DataFrame, dict[str, object]]:
    fixtures = pd.read_parquet(paths["fixtures"])
    stats = pd.read_parquet(paths["match_stats"])
    leagues = pd.read_parquet(paths["leagues"])
    teams = pd.read_parquet(paths["teams"])
    _require_columns(
        fixtures,
        {
            "id",
            "date_utc",
            "league_id",
            "home_team_id",
            "away_team_id",
            "goals_home",
            "goals_away",
            "status_norm",
            "is_played",
        },
        "fixtures",
    )
    _require_columns(
        stats,
        {
            "fixture_id",
            "home_xg",
            "away_xg",
            "xg_covered",
            "xg_nulled",
            "known_at",
        },
        "match_stats",
    )
    _require_columns(leagues, {"id", "name"}, "leagues")
    _require_columns(teams, {"id", "name"}, "teams")

    expected_leagues = {
        "UEFA Champions League": 89,
        "UEFA Europa League": 90,
        "UEFA Conference League": 91,
    }
    league_map = leagues.set_index("name")["id"].to_dict()
    for name, expected_id in expected_leagues.items():
        if int(league_map.get(name, -1)) != expected_id:
            raise ValueError(f"Unexpected source league mapping for {name}")

    source = fixtures.loc[
        fixtures["league_id"].isin(LEAGUE_TO_COMPETITION)
        & fixtures["is_played"].eq(True)
    ].merge(
        stats[
            [
                "fixture_id",
                "home_xg",
                "away_xg",
                "xg_covered",
                "xg_nulled",
                "known_at",
            ]
        ],
        left_on="id",
        right_on="fixture_id",
        how="left",
        validate="one_to_one",
    )
    names = teams.set_index("id")["name"]
    source["source_home_team_name"] = source["home_team_id"].map(names)
    source["source_away_team_name"] = source["away_team_id"].map(names)
    if source[
        ["source_home_team_name", "source_away_team_name"]
    ].isna().any().any():
        raise ValueError("Source team identity is missing")
    source["competition"] = source["league_id"].map(LEAGUE_TO_COMPETITION)
    source["source_kickoff_utc"] = pd.to_datetime(
        source["date_utc"],
        utc=True,
        errors="coerce",
    )
    source["known_at"] = pd.to_datetime(
        source["known_at"],
        utc=True,
        errors="coerce",
    )
    source["source_home_key"] = source["source_home_team_name"].map(
        normalize_name
    )
    source["source_away_key"] = source["source_away_team_name"].map(
        normalize_name
    )
    source["has_xg"] = source["home_xg"].notna() & source["away_xg"].notna()
    source = source.loc[source["has_xg"]].copy()
    for column in ("home_xg", "away_xg"):
        numeric = pd.to_numeric(source[column], errors="coerce")
        if numeric.isna().any() or not numeric.map(math.isfinite).all():
            raise ValueError(f"Source {column} must be finite numeric")
        if numeric.lt(0.0).any():
            raise ValueError(f"Source {column} must be non-negative")
        source[column] = numeric.astype(float)
    if source["id"].duplicated().any():
        raise ValueError("Source fixture IDs must be unique")
    quality = {
        "all_played_european_matches": int(
            fixtures.loc[
                fixtures["league_id"].isin(LEAGUE_TO_COMPETITION)
                & fixtures["is_played"].eq(True)
            ].shape[0]
        ),
        "rows_with_both_xg": int(len(source)),
        "provider_metric": "coarse zone-derived xG estimate",
        "provider_warning": (
            "Not an Opta/StatsBomb-style per-shot model; source documentation "
            "describes API-Football xG as a deterministic shots-by-zone estimate."
        ),
    }
    return source.reset_index(drop=True), quality


def match_xg_to_ao(
    events: pd.DataFrame,
    source: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ao = events.copy()
    ao["ao_date"] = ao["kickoff_utc"].dt.date
    ao["ao_home_key"] = ao["home_team_name"].map(normalize_name)
    ao["ao_away_key"] = ao["away_team_name"].map(normalize_name)
    source = source.copy()
    source["source_date"] = source["source_kickoff_utc"].dt.date

    matched_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    used_source_ids: set[int] = set()

    eligible_source = source.groupby(
        ["competition", "source_date"],
        sort=False,
    )
    for row in ao.itertuples(index=False):
        key = (str(row.competition), row.ao_date)
        try:
            candidates = eligible_source.get_group(key)
        except KeyError:
            candidates = source.iloc[0:0]
        scores: list[tuple[float, float, float, pd.Series]] = []
        for _, candidate in candidates.iterrows():
            source_id = int(candidate["id"])
            if source_id in used_source_ids:
                continue
            if int(candidate["goals_home"]) != int(row.home_goals):
                continue
            if int(candidate["goals_away"]) != int(row.away_goals):
                continue
            home_similarity = SequenceMatcher(
                None,
                str(row.ao_home_key),
                str(candidate["source_home_key"]),
            ).ratio()
            away_similarity = SequenceMatcher(
                None,
                str(row.ao_away_key),
                str(candidate["source_away_key"]),
            ).ratio()
            pair_similarity = (home_similarity + away_similarity) / 2.0
            scores.append(
                (
                    pair_similarity,
                    home_similarity,
                    away_similarity,
                    candidate,
                )
            )
        scores.sort(key=lambda value: value[0], reverse=True)
        best = scores[0] if scores else None
        runner_up = scores[1][0] if len(scores) > 1 else 0.0
        accepted = bool(
            best is not None
            and best[1] >= MIN_SIDE_SIMILARITY
            and best[2] >= MIN_SIDE_SIMILARITY
            and best[0] >= MIN_PAIR_SIMILARITY
            and best[0] - runner_up >= MIN_RUNNER_UP_GAP
        )
        audit = {
            "match_id": str(row.match_id),
            "season": str(row.season),
            "competition": str(row.competition),
            "kickoff_utc": row.kickoff_utc,
            "home_team_name": str(row.home_team_name),
            "away_team_name": str(row.away_team_name),
            "home_goals": int(row.home_goals),
            "away_goals": int(row.away_goals),
            "candidate_count": len(scores),
            "matched": accepted,
            "source_fixture_id": (
                int(best[3]["id"]) if accepted and best is not None else None
            ),
            "home_similarity": (
                float(best[1]) if best is not None else None
            ),
            "away_similarity": (
                float(best[2]) if best is not None else None
            ),
            "pair_similarity": (
                float(best[0]) if best is not None else None
            ),
            "runner_up_similarity": (
                float(runner_up) if scores else None
            ),
            "resolution": (
                "date_competition_score_high_confidence_name"
                if accepted
                else "no_source_xg_on_date"
                if candidates.empty
                else "no_score_match"
                if not scores
                else "identity_below_threshold_or_ambiguous"
            ),
        }
        audit_rows.append(audit)
        if not accepted or best is None:
            continue

        candidate = best[3]
        source_id = int(candidate["id"])
        used_source_ids.add(source_id)
        known_after_match = bool(
            pd.notna(candidate["known_at"])
            and candidate["known_at"] > row.kickoff_utc
        )
        status = str(candidate["status_norm"]).upper()
        eligible = bool(
            status == "FT"
            and bool(candidate["xg_covered"])
            and not bool(candidate["xg_nulled"])
            and known_after_match
        )
        matched_rows.append(
            {
                "match_id": str(row.match_id),
                "season": str(row.season),
                "competition": str(row.competition),
                "kickoff_utc": row.kickoff_utc,
                "home_team_id": int(row.home_team_id),
                "away_team_id": int(row.away_team_id),
                "home_team_name": str(row.home_team_name),
                "away_team_name": str(row.away_team_name),
                "home_goals": int(row.home_goals),
                "away_goals": int(row.away_goals),
                "decided_on_penalties": bool(row.decided_on_penalties),
                "source_fixture_id": source_id,
                "source_home_team_name": str(
                    candidate["source_home_team_name"]
                ),
                "source_away_team_name": str(
                    candidate["source_away_team_name"]
                ),
                "source_status": status,
                "xg_home": float(candidate["home_xg"]),
                "xg_away": float(candidate["away_xg"]),
                "xg_type": "coarse_zone_derived_xg",
                "duration_scope": "90_minutes_ft_only",
                "provider": "API-Football via eatpizzanot/soccer-dataset",
                "snapshot_time": candidate["known_at"],
                "penalty_shootout_excluded": status == "FT",
                "xg_covered": bool(candidate["xg_covered"]),
                "xg_nulled": bool(candidate["xg_nulled"]),
                "known_after_kickoff": known_after_match,
                "pair_similarity": float(best[0]),
                "eligible_for_ablation": eligible,
            }
        )

    matched = pd.DataFrame(matched_rows)
    audit = pd.DataFrame(audit_rows)
    if not matched.empty:
        matched = matched.sort_values(["kickoff_utc", "match_id"]).reset_index(
            drop=True
        )
    return matched, audit


def build_coverage_matrix(
    events: pd.DataFrame,
    matched: pd.DataFrame,
) -> pd.DataFrame:
    total = (
        events.groupby(["season", "competition"], sort=True)
        .size()
        .rename("ao_matches")
        .reset_index()
    )
    if matched.empty:
        total["matched_xg"] = 0
        total["eligible_xg"] = 0
    else:
        summary = (
            matched.groupby(["season", "competition"], sort=True)
            .agg(
                matched_xg=("match_id", "size"),
                eligible_xg=("eligible_for_ablation", "sum"),
            )
            .reset_index()
        )
        total = total.merge(
            summary,
            on=["season", "competition"],
            how="left",
        )
        total[["matched_xg", "eligible_xg"]] = total[
            ["matched_xg", "eligible_xg"]
        ].fillna(0).astype(int)
    total["eligible_coverage_rate"] = (
        total["eligible_xg"] / total["ao_matches"]
    )
    return total


def build_source_coverage(source: pd.DataFrame) -> pd.DataFrame:
    values = source.copy()
    values["calendar_year"] = values["source_kickoff_utc"].dt.year
    return (
        values.groupby(["calendar_year", "competition"], sort=True)
        .agg(
            source_xg_matches=("id", "size"),
            ft_matches=("status_norm", lambda s: int(s.eq("FT").sum())),
            covered_matches=("xg_covered", "sum"),
        )
        .reset_index()
    )


def validate_output(
    events: pd.DataFrame,
    matched: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    if len(audit) != len(events):
        raise ValueError("Identity audit must contain exactly one row per AO match")
    if audit["match_id"].duplicated().any():
        raise ValueError("Identity audit contains duplicate AO match IDs")
    if matched.empty:
        raise ValueError("No source xG matches were resolved")
    if matched["match_id"].duplicated().any():
        raise ValueError("Matched xG contains duplicate AO match IDs")
    if matched["source_fixture_id"].duplicated().any():
        raise ValueError("Matched xG contains duplicate source fixture IDs")
    if not set(matched["match_id"]).issubset(set(events["match_id"])):
        raise ValueError("Matched xG contains an unknown AO match ID")
    eligible = matched.loc[matched["eligible_for_ablation"]]
    if eligible.empty:
        raise ValueError("No xG rows passed the ablation eligibility contract")
    if not eligible["penalty_shootout_excluded"].eq(True).all():
        raise ValueError("Eligible xG must exclude penalty shootouts")
    if not eligible["source_status"].eq("FT").all():
        raise ValueError("Eligible xG must use 90-minute FT matches")
    if not eligible["known_after_kickoff"].eq(True).all():
        raise ValueError("Eligible xG must be a post-match observation")


def build_manifest(
    paths: dict[str, Path],
    source_quality: dict[str, object],
    events: pd.DataFrame,
    matched: pd.DataFrame,
) -> dict[str, object]:
    eligible = matched.loc[matched["eligible_for_ablation"]]
    return {
        "dataset": "AO xG matched-sample ablation dataset",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "license": "CC-BY-4.0",
        "attribution": (
            "eatpizzanot/soccer-dataset; underlying sources API-Football "
            "and football-data.co.uk"
        ),
        "source_urls": SOURCE_URLS,
        "source_sha256": {
            name: sha256_file(path) for name, path in paths.items()
        },
        "source_quality": source_quality,
        "contract": {
            "match_identity": (
                "competition + UTC date + exact score + high-confidence "
                "home/away normalized-name match"
            ),
            "xg_type": "coarse_zone_derived_xg",
            "duration_scope": "90-minute FT matches only",
            "missing_xg": "never imputed; excluded from matched evaluation",
            "penalty_shootouts": "excluded from eligible sample",
            "extra_time": "excluded because duration scope is not provider-auditable",
            "future_information": (
                "xG is post-match evidence used only after the prediction and result"
            ),
        },
        "rows": {
            "ao_matches": int(len(events)),
            "source_xg_matches": int(source_quality["rows_with_both_xg"]),
            "matched_xg_matches": int(len(matched)),
            "eligible_xg_matches": int(len(eligible)),
        },
        "production_eligibility": False,
        "production_blocker": (
            "Provider xG is a coarse zone-derived estimate with sparse and "
            "non-contiguous tournament-season coverage."
        ),
    }


def build_readme(
    manifest: dict[str, object],
    coverage: pd.DataFrame,
) -> str:
    rows = manifest["rows"]
    eligible_seasons = coverage.loc[
        coverage["eligible_xg"].gt(0),
        ["season", "competition", "eligible_xg", "ao_matches"],
    ]
    table_lines = [
        "| season | competition | eligible_xg | ao_matches |",
        "|---|---:|---:|---:|",
    ]
    table_lines.extend(
        "| {season} | {competition} | {eligible_xg} | {ao_matches} |".format(
            season=row.season,
            competition=row.competition,
            eligible_xg=int(row.eligible_xg),
            ao_matches=int(row.ao_matches),
        )
        for row in eligible_seasons.itertuples(index=False)
    )
    table = "\n".join(table_lines)
    return f"""# AO xG Backtest Dataset

Bu klasör, AO European Elo'nun gol farkı ve xG ablation çalışması için
denetlenmiş matched-sample verisini içerir.

## Kaynak ve Lisans

- Kaynak: `eatpizzanot/soccer-dataset`
- Kaynak URL: https://huggingface.co/datasets/eatpizzanot/soccer-dataset
- Veri sözlüğü: https://huggingface.co/datasets/eatpizzanot/soccer-dataset/blob/main/data_dictionary.md
- Alt kaynaklar: API-Football ve football-data.co.uk
- Lisans: CC-BY-4.0
- Metrik: `coarse_zone_derived_xg`

Kaynağın kendi veri sözlüğüne göre bu değer Opta/StatsBomb tipi per-shot xG
değildir; şut bölgelerinden türetilmiş kaba bir sağlayıcı tahminidir. Bu nedenle
sonuçlar yalnızca exploratory/shadow kanıt sayılır ve production aktivasyonu için
tek başına yeterli değildir.

## Sözleşme

- Eksik xG sıfırla doldurulmaz.
- Yalnızca 90 dakikada `FT` biten, iki taraf xG'si sonlu ve negatif olmayan maçlar
  ablation örneklemine alınır.
- Uzatma ve penaltı maçları, sağlayıcının süre kapsamı denetlenemediği için dışarıda
  bırakılır.
- Eşleşme anahtarı turnuva, UTC tarih, skor ve yüksek güvenli takım adı eşleşmesidir.
- xG yalnızca maç bittikten sonra rating güncellemesinde kullanılır.

## Satır Sayıları

- AO maçları: {rows["ao_matches"]}
- Kaynakta iki taraf xG bulunan maç: {rows["source_xg_matches"]}
- AO ile eşleşen xG maçı: {rows["matched_xg_matches"]}
- Ablation için uygun maç: {rows["eligible_xg_matches"]}

## Uygun Kapsam

{table}

`source_manifest.json` kaynak hash'lerini ve production engelini,
`identity_audit.csv` her AO maçının eşleşme kararını,
`coverage_matrix.csv` sezon-turnuva kapsamını içerir.
"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    label: str,
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


if __name__ == "__main__":
    main()
