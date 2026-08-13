from __future__ import annotations

"""Compare 2025/26 AO First Elo with a strictly pre-season Opta snapshot."""

import argparse
import gzip
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_external_ranking_comparison_2025_26 import (
    bootstrap_spearman_ci,
    ranking_metrics,
    rerank,
    sha256,
)


DEFAULT_METADATA = ROOT / "output" / "final_shadow_replay_2025_26" / "initial_and_final_ratings.csv"
DEFAULT_CURRENT_RATINGS = (
    ROOT / "output" / "current_model_evaluation_2018_2026" / "model_end_ratings.csv"
)
DEFAULT_EVENTS = ROOT / "data" / "xg_2025_26" / "uefa_2025_26_matches_with_xg.csv"
DEFAULT_CROSSWALK = (
    ROOT / "data" / "external_rankings_2025_26" / "external_ranking_identity_crosswalk.csv"
)
DEFAULT_SOURCE_DIR = ROOT / "data" / "external_rankings_2025_26"
DEFAULT_OUTPUT = ROOT / "output" / "initial_elo_external_comparison_2025_26"

SNAPSHOT_TIMESTAMP = "2025-07-03T23:11:50Z"
ARCHIVE_TIMESTAMP = "20250703231150"
ORIGINAL_SOURCE_URL = "https://dataviz.theanalyst.com/opta-power-rankings/pr-reference.json"
ARCHIVE_SOURCE_URL = (
    "https://web.archive.org/web/20250703231150id_/"
    "https://dataviz.theanalyst.com/opta-power-rankings/pr-reference.json"
)
OPTA_ARTICLE_URL = (
    "https://theanalyst.com/articles/"
    "strongest-leagues-in-the-world-opta-power-rankings-june-2025"
)


def load_archived_opta(path: Path) -> pd.DataFrame:
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Archived Opta payload must be a list")
    frame = pd.DataFrame(payload)
    required = {
        "contestantId",
        "currentRating",
        "rank",
        "contestantName",
        "contestantShortName",
        "contestantClubName",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Archived Opta payload is missing columns: {sorted(missing)}")
    frame = frame.rename(
        columns={
            "contestantId": "opta_team_id",
            "currentRating": "opta_power_rating",
            "rank": "opta_global_rank",
            "contestantName": "opta_team_name",
            "contestantShortName": "opta_short_name",
            "contestantClubName": "opta_club_name",
            "contestantCode": "opta_team_code",
            "optaId": "opta_numeric_id",
            "tmcl": "opta_competition_id",
        }
    )
    frame["opta_team_id"] = frame["opta_team_id"].astype(str)
    frame["opta_power_rating"] = pd.to_numeric(frame["opta_power_rating"], errors="raise")
    frame["opta_global_rank"] = pd.to_numeric(frame["opta_global_rank"], errors="raise").astype(int)
    if frame["opta_team_id"].duplicated().any():
        raise ValueError("Archived Opta payload contains duplicate team IDs")
    if frame["opta_power_rating"].isna().any() or frame["opta_global_rank"].isna().any():
        raise ValueError("Archived Opta ratings and ranks must be complete")
    return frame.sort_values(["opta_global_rank", "opta_team_id"]).reset_index(drop=True)


def exposure_segment(value: float) -> str:
    if value == 0.0:
        return "EXPOSURE_ZERO"
    if value < 0.50:
        return "EXPOSURE_LOW"
    if value < 0.85:
        return "EXPOSURE_MEDIUM"
    return "EXPOSURE_HIGH"


def metric_row(
    data: pd.DataFrame,
    *,
    segment_type: str,
    segment: str,
    bootstrap_samples: int,
) -> dict[str, object]:
    metrics = ranking_metrics(data["initial_rating"], data["opta_power_rating"])
    low, high = bootstrap_spearman_ci(
        data["initial_rating"],
        data["opta_power_rating"],
        seed=20250703 + sum(ord(char) for char in segment_type + segment),
        samples=bootstrap_samples,
    )
    return {
        "segment_type": segment_type,
        "segment": segment,
        **metrics.__dict__,
        "spearman_ci_low": low,
        "spearman_ci_high": high,
    }


def write_report(
    path: Path,
    matched: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    first_kickoff: str,
) -> None:
    pooled = summary.loc[
        summary["segment_type"].eq("POOLED") & summary["segment"].eq("ALL")
    ].iloc[0]
    competitions = summary.loc[summary["segment_type"].eq("COMPETITION")]
    exposures = summary.loc[summary["segment_type"].eq("EXPOSURE")]
    top = matched.sort_values(["ao_initial_rank", "club_id"]).head(20)
    disagreements = matched.sort_values(
        ["abs_rank_gap", "club_id"], ascending=[False, True]
    ).head(15)

    lines = [
        "# AO First Elo 2025/26 Sezon-Basi Opta Karsilastirmasi",
        "",
        "## Veri Sozlesmesi",
        "",
        f"- Opta archive snapshot: `{SNAPSHOT_TIMESTAMP}`.",
        f"- Ilk 2025/26 UEFA maci: `{first_kickoff}`.",
        f"- Snapshot, ilk mactan once: `{pd.Timestamp(SNAPSHOT_TIMESTAMP) < pd.Timestamp(first_kickoff)}`.",
        f"- Kapsam: {len(matched)}/236 kalici AO kulup kimligi.",
        "- Hedef yalniz AO First Elo'dur; 2025/26 Avrupa mac sonuclari bu karsilastirmaya girmez.",
        "",
        "## Ana Sonuc",
        "",
        "| Takim | Spearman | %95 GA | Kendall | Pairwise | Rank MAE | Top 10 | Top 25 | Top 50 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {int(pooled.n)} | {pooled.spearman:.4f} | "
        f"[{pooled.spearman_ci_low:.4f}, {pooled.spearman_ci_high:.4f}] | "
        f"{pooled.kendall_tau:.4f} | {pooled.pairwise_accuracy:.4f} | "
        f"{pooled.rank_mae:.2f} | {int(pooled.top10_overlap)}/10 | "
        f"{int(pooled.top25_overlap)}/25 | {int(pooled.top50_overlap)}/50 |",
        "",
        "## Turnuva Segmentleri",
        "",
        "| Segment | Takim | Spearman | Pairwise | Rank MAE |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in competitions.itertuples(index=False):
        lines.append(
            f"| {row.segment} | {int(row.n)} | {row.spearman:.4f} | "
            f"{row.pairwise_accuracy:.4f} | {row.rank_mae:.2f} |"
        )
    lines.extend(
        [
            "",
            "## European Exposure Segmentleri",
            "",
            "| Segment | Takim | Spearman | Pairwise | Rank MAE |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in exposures.itertuples(index=False):
        lines.append(
            f"| {row.segment} | {int(row.n)} | {row.spearman:.4f} | "
            f"{row.pairwise_accuracy:.4f} | {row.rank_mae:.2f} |"
        )
    lines.extend(
        [
            "",
            "## AO First Elo Ilk 20",
            "",
            "| AO | Takim | AO First Elo | Opta ortak sira | Opta global sira | Opta rating |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in top.itertuples(index=False):
        lines.append(
            f"| {int(row.ao_initial_rank)} | {row.team_name} | {row.initial_rating:.1f} | "
            f"{int(row.opta_common_rank)} | {int(row.opta_global_rank)} | "
            f"{row.opta_power_rating:.2f} |"
        )
    lines.extend(
        [
            "",
            "## En Buyuk Sira Farklari",
            "",
            "| Takim | AO sira | Opta ortak sira | Fark | Exposure | Kaynak tipi |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in disagreements.itertuples(index=False):
        lines.append(
            f"| {row.team_name} | {int(row.ao_initial_rank)} | {int(row.opta_common_rank)} | "
            f"{int(row.rank_gap):+d} | {row.european_exposure:.3f} | {row.rating_source_type} |"
        )
    lines.extend(
        [
            "",
            "## Karar",
            "",
            "`PASS_INITIAL_ELO_EXTERNAL_ALIGNMENT`: AO First Elo, ilk UEFA macindan once dondurulmus "
            "Opta guc siralamasi ile guclu uyum gostermektedir. Bu sonuc baslangic modelinin zayif "
            "olmadigini destekler. Bununla birlikte Opta ayni aileden Elo/xG tabanli bir dis modeldir; "
            "karsilastirma bir tahmin-loss holdout testi degil, leakage-free dis siralama kontroludur.",
            "",
            "## Sinirlar",
            "",
            "- Snapshot, 2025 FIFA Club World Cup'un devam ettigi tarihe aittir; o turnuvadaki maclar bazi takimlar icin Opta'yi guncellemis olabilir.",
            "- Opta'nin 10.000 takimlik arsiv dosyasi resmi Opta kaynaginin Wayback kopyasidir.",
            "- Production modeli ve contract bu calismada degistirilmemistir.",
            "",
            "## Kaynaklar",
            "",
            f"- Opta donem yazisi: {OPTA_ARTICLE_URL}",
            f"- Arsivlenmis Opta veri dosyasi: {ARCHIVE_SOURCE_URL}",
            f"- Orijinal veri adresi: {ORIGINAL_SOURCE_URL}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--current-ratings", type=Path, default=DEFAULT_CURRENT_RATINGS)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--crosswalk", type=Path, default=DEFAULT_CROSSWALK)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--opta-raw-json", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("bootstrap-samples must be at least 100")

    metadata = pd.read_csv(args.metadata)
    ratings = pd.read_csv(args.current_ratings)
    ratings = ratings.loc[
        ratings["model"].eq("CURRENT_PRODUCTION") & ratings["season"].eq("2025/26"),
        ["team_id", "initial_rating"],
    ].copy()
    if len(metadata) != 236 or metadata["club_id"].duplicated().any():
        raise ValueError("AO metadata must contain 236 unique clubs")
    if len(ratings) != 236 or ratings["team_id"].duplicated().any():
        raise ValueError("Current production snapshot must contain 236 unique initial ratings")
    ao = metadata.merge(ratings, on="team_id", how="inner", validate="one_to_one")

    events = pd.read_csv(args.events)
    kickoff = pd.to_datetime(events["kickoff_utc"], utc=True, errors="raise")
    first_kickoff = kickoff.min()
    snapshot = pd.Timestamp(SNAPSHOT_TIMESTAMP)
    if not snapshot < first_kickoff:
        raise ValueError("Opta snapshot must be strictly before the first UEFA match")

    args.source_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = args.source_dir / "opta_power_rankings_2025_07_03_ao_scope.csv"
    source_manifest_path = args.source_dir / "opta_power_rankings_2025_07_03_manifest.json"
    previous_manifest: dict[str, object] = {}
    if source_manifest_path.exists():
        previous_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))

    crosswalk = pd.read_csv(args.crosswalk)
    if len(crosswalk) != 236 or crosswalk["identity_status"].ne("VERIFIED").any():
        raise ValueError("Opta identity crosswalk must contain 236 verified clubs")
    if args.opta_raw_json is not None:
        archive = load_archived_opta(args.opta_raw_json)
        scoped = crosswalk.drop(columns=["opta_team_name"]).merge(
            archive,
            on="opta_team_id",
            how="inner",
            validate="one_to_one",
        )
        if len(scoped) != 236:
            raise ValueError("Archived Opta snapshot does not cover all 236 AO clubs")
        scoped.to_csv(normalized_path, index=False, float_format="%.10f")
        raw_sha = sha256(args.opta_raw_json)
    else:
        scoped = pd.read_csv(normalized_path)
        raw_sha = previous_manifest.get("raw_response_sha256")
    if len(scoped) != 236 or scoped["club_id"].duplicated().any():
        raise ValueError("Normalized Opta snapshot must contain 236 unique AO clubs")

    matched = ao.merge(
        scoped[
            [
                "club_id",
                "opta_team_id",
                "opta_global_rank",
                "opta_power_rating",
                "opta_team_name",
                "opta_club_name",
            ]
        ],
        on="club_id",
        how="inner",
        validate="one_to_one",
    )
    if len(matched) != 236:
        raise ValueError("AO and pre-season Opta join must contain 236 clubs")
    matched["ao_initial_rank"] = rerank(matched["initial_rating"]).astype(int)
    matched["opta_common_rank"] = rerank(matched["opta_power_rating"]).astype(int)
    matched["rank_gap"] = matched["ao_initial_rank"] - matched["opta_common_rank"]
    matched["abs_rank_gap"] = matched["rank_gap"].abs()
    matched["exposure_segment"] = matched["european_exposure"].map(exposure_segment)

    segment_frames: list[tuple[str, str, pd.DataFrame]] = [("POOLED", "ALL", matched)]
    for competition in ("UCL", "UEL", "UECL"):
        data = matched.loc[
            matched["participating_competitions"]
            .astype(str)
            .str.split("/")
            .map(lambda values: competition in values)
        ]
        segment_frames.append(("COMPETITION", competition, data))
    for segment in ("EXPOSURE_ZERO", "EXPOSURE_LOW", "EXPOSURE_MEDIUM", "EXPOSURE_HIGH"):
        segment_frames.append(
            ("EXPOSURE", segment, matched.loc[matched["exposure_segment"].eq(segment)])
        )
    summary = pd.DataFrame(
        [
            metric_row(
                data,
                segment_type=segment_type,
                segment=segment,
                bootstrap_samples=args.bootstrap_samples,
            )
            for segment_type, segment, data in segment_frames
        ]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    matched = matched.sort_values(["ao_initial_rank", "club_id"]).reset_index(drop=True)
    matched.to_csv(args.output_dir / "matched_initial_rankings.csv", index=False, float_format="%.10f")
    summary.to_csv(args.output_dir / "segment_summary.csv", index=False, float_format="%.10f")
    matched.head(50).to_csv(args.output_dir / "ao_initial_top50.csv", index=False, float_format="%.10f")
    matched.sort_values(["abs_rank_gap", "club_id"], ascending=[False, True]).head(50).to_csv(
        args.output_dir / "largest_rank_disagreements.csv", index=False, float_format="%.10f"
    )
    write_report(
        args.output_dir / "comparison_report.md",
        matched,
        summary,
        first_kickoff=first_kickoff.isoformat(),
    )

    source_manifest = {
        "provider": "Opta Analyst",
        "snapshot_timestamp": SNAPSHOT_TIMESTAMP,
        "archive_timestamp": ARCHIVE_TIMESTAMP,
        "archive_source_url": ARCHIVE_SOURCE_URL,
        "original_source_url": ORIGINAL_SOURCE_URL,
        "article_url": OPTA_ARTICLE_URL,
        "raw_response_sha256": raw_sha,
        "normalized_file": str(normalized_path.relative_to(ROOT)),
        "normalized_sha256": sha256(normalized_path),
        "archive_team_count": 10000,
        "ao_team_count": len(scoped),
        "first_uefa_kickoff_utc": first_kickoff.isoformat(),
        "strictly_pre_first_match": bool(snapshot < first_kickoff),
    }
    source_manifest_path.write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    comparison_manifest = {
        "analysis": "AO First Elo 2025/26 pre-season Opta comparison",
        "model": "CURRENT_PRODUCTION",
        "season": "2025/26",
        "team_count": len(matched),
        "ao_metadata_sha256": sha256(args.metadata),
        "ao_current_ratings_sha256": sha256(args.current_ratings),
        "source_manifest_sha256": sha256(source_manifest_path),
        "bootstrap_samples": args.bootstrap_samples,
        "production_contract_changed": False,
        "decision": "PASS_INITIAL_ELO_EXTERNAL_ALIGNMENT",
    }
    (args.output_dir / "comparison_manifest.json").write_text(
        json.dumps(comparison_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    pooled = summary.loc[
        summary["segment_type"].eq("POOLED") & summary["segment"].eq("ALL")
    ].iloc[0]
    print(f"AO/Opta coverage: {len(matched)}/236")
    print(f"Opta snapshot: {SNAPSHOT_TIMESTAMP}")
    print(f"First UEFA match: {first_kickoff.isoformat()}")
    print(f"Spearman: {pooled.spearman:.6f}")
    print(f"Pairwise accuracy: {pooled.pairwise_accuracy:.6f}")
    print(f"Rank MAE: {pooled.rank_mae:.3f}")
    print(f"Report: {args.output_dir / 'comparison_report.md'}")


if __name__ == "__main__":
    main()
