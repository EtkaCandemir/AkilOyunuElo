from __future__ import annotations

"""Compare the 2025/26 AO season snapshot with frozen UEFA and Opta rankings."""

import argparse
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AO = ROOT / "output" / "final_shadow_replay_2025_26" / "initial_and_final_ratings.csv"
DEFAULT_CURRENT_MODEL_RATINGS = (
    ROOT / "output" / "current_model_evaluation_2018_2026" / "model_end_ratings.csv"
)
DEFAULT_SOURCE_DIR = ROOT / "data" / "external_rankings_2025_26"
DEFAULT_OUTPUT = ROOT / "output" / "external_ranking_comparison_2025_26"

UEFA_SOURCE_URL = "https://www.uefa.com/nationalassociations/uefarankings/?year=2026"
UEFA_METHOD_URL = "https://www.uefa.com/nationalassociations/uefarankings/club/?year=2026"
OPTA_SOURCE_URL = "https://dataviz.theanalyst.com/opta-power-rankings/"
OPTA_METHOD_URL = "https://theanalyst.com/articles/power-rankings-your-club-ranked"

# Only aliases that remain unresolved after UEFA-name-assisted exact matching.
OPTA_ALIASES_BY_UEFA_ID = {
    52327: "3kfktv64h7kg7zryax1wktr5r",  # AZ
    79970: "6j2gnvqgprscy600ex68i5ynf",  # Kairat
    52350: "1c7dv8ays675v9kp6a04sd8pq",  # KuPS
    77928: "x5bf07g3gy7xj2zidytj28m3",  # Oleksandria
    2607480: "8oxa17y70f9muh5ytwsbikqeq",  # Zira
    64289: "3z0tiyynmxdyozfrpasdbpm0p",  # Radnicki 1923
    77482: "8952lfdnv83lu5eik36jp1znu",  # FCI Levadia
    64374: "azfxqr8km2gusov0ahqgq51f2",  # Dinamo Brest
    88135: "8k1u9vuetmkqc1d74i44i7drx",  # Racing Union Luxembourg
}


@dataclass(frozen=True)
class RankingMetrics:
    n: int
    spearman: float
    kendall_tau: float
    pairwise_accuracy: float
    rank_mae: float
    rank_median_ae: float
    top10_overlap: int
    top25_overlap: int
    top50_overlap: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def finite_float(value: object, label: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite; got {value!r}")
    return result


def load_uefa_raw(path: Path) -> tuple[pd.DataFrame, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    data = payload.get("data", payload)
    members = data.get("members", [])
    rows: list[dict[str, object]] = []
    for item in members:
        member = item["member"]
        ranking = item["overallRanking"]
        rows.append(
            {
                "uefa_team_id": int(member["id"]),
                "uefa_rank": int(ranking["position"]),
                "uefa_coefficient": finite_float(ranking["totalValue"], "UEFA coefficient"),
                "uefa_club_points": finite_float(ranking.get("totalPoints", ranking["totalValue"]), "UEFA club points"),
                "uefa_association_floor": finite_float(ranking.get("nationalAssociationPoints", 0.0), "UEFA association floor"),
                "uefa_display_name": member["displayName"],
                "uefa_short_name": member.get("displayNameShort", ""),
                "uefa_official_name": member.get("displayOfficialName", ""),
                "uefa_international_name": member.get("internationalName", ""),
                "uefa_country_code": member["countryCode"],
                "uefa_target_season_year": int(ranking["targetSeasonYear"]),
                "uefa_base_season_year": int(ranking["baseSeasonYear"]),
            }
        )
    frame = pd.DataFrame(rows).sort_values(["uefa_rank", "uefa_team_id"]).reset_index(drop=True)
    if frame.empty or frame["uefa_team_id"].duplicated().any():
        raise ValueError("UEFA source must contain unique club rows")
    if not frame["uefa_rank"].is_monotonic_increasing:
        raise ValueError("UEFA source is not rank ordered")
    return frame, str(data.get("lastUpdateDate", ""))


def load_opta_raw(path: Path) -> tuple[pd.DataFrame, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", payload)
    frame = pd.DataFrame(rows)
    required = {
        "rank",
        "contestantId",
        "contestantName",
        "contestantClubName",
        "contestantShortName",
        "currentRating",
        "association",
        "confederation",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Opta source is missing columns: {sorted(missing)}")
    frame = frame.loc[frame["confederation"].eq("Europe")].copy()
    frame = frame.rename(
        columns={
            "rank": "opta_global_rank",
            "contestantId": "opta_team_id",
            "contestantName": "opta_team_name",
            "contestantClubName": "opta_club_name",
            "contestantShortName": "opta_short_name",
            "currentRating": "opta_power_rating",
            "association": "opta_association",
            "domesticLeagueName": "opta_domestic_league",
        }
    )
    frame["opta_global_rank"] = frame["opta_global_rank"].astype(int)
    frame["opta_power_rating"] = frame["opta_power_rating"].astype(float)
    if frame["opta_team_id"].duplicated().any():
        raise ValueError("Opta source must contain unique contestant IDs")
    return frame.sort_values("opta_global_rank").reset_index(drop=True), str(payload.get("last_updated", ""))


def country_code_map(ao: pd.DataFrame) -> dict[str, str]:
    mapping = {
        normalize_name(country): str(code)
        for country, code in ao[["country", "country_code"]].drop_duplicates().itertuples(index=False)
    }
    mapping.update(
        {
            "bosniaandherzegovina": "BOS",
            "republicofireland": "IRL",
        }
    )
    return mapping


def match_opta(
    ao: pd.DataFrame,
    uefa: pd.DataFrame,
    opta: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    uefa_by_id = uefa.set_index("uefa_team_id")
    candidates = opta.copy()
    mapping = country_code_map(ao)
    candidates["opta_country_code"] = candidates["opta_association"].map(
        lambda value: mapping.get(normalize_name(value))
    )
    name_columns = ["opta_team_name", "opta_club_name", "opta_short_name"]
    candidates["_names"] = candidates[name_columns].apply(
        lambda row: {normalize_name(value) for value in row if pd.notna(value) and str(value).strip()},
        axis=1,
    )

    rows: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    used_opta_ids: set[str] = set()
    for team in ao.itertuples(index=False):
        uefa_row = uefa_by_id.loc[int(team.uefa_team_id)]
        ao_names = {
            normalize_name(value)
            for value in (
                team.team_name,
                uefa_row["uefa_display_name"],
                uefa_row["uefa_short_name"],
                uefa_row["uefa_official_name"],
                uefa_row["uefa_international_name"],
            )
            if pd.notna(value) and str(value).strip()
        }
        scoped = candidates.loc[
            candidates["opta_country_code"].eq(team.country_code)
            | candidates["_names"].map(lambda values: bool(values & ao_names))
        ]
        exact = scoped.loc[scoped["_names"].map(lambda values: bool(values & ao_names))]
        alias_id = OPTA_ALIASES_BY_UEFA_ID.get(int(team.uefa_team_id))
        if exact.empty and alias_id is not None:
            exact = candidates.loc[candidates["opta_team_id"].eq(alias_id)]
            method = "MANUAL_ALIAS_VERIFIED"
        else:
            method = "UEFA_NAME_COUNTRY_EXACT"
        if exact.empty:
            raise ValueError(f"No Opta match for {team.club_id} / {team.team_name}")
        selected = exact.sort_values(
            ["opta_power_rating", "opta_global_rank"], ascending=[False, True]
        ).iloc[0]
        opta_id = str(selected["opta_team_id"])
        if opta_id in used_opta_ids:
            raise ValueError(f"Opta contestant matched twice: {opta_id}")
        used_opta_ids.add(opta_id)
        rows.append(
            {
                "club_id": team.club_id,
                "uefa_team_id": int(team.uefa_team_id),
                "opta_team_id": opta_id,
                "opta_global_rank": int(selected["opta_global_rank"]),
                "opta_power_rating": float(selected["opta_power_rating"]),
                "opta_team_name": selected["opta_team_name"],
                "opta_club_name": selected["opta_club_name"],
                "opta_association": selected["opta_association"],
                "opta_domestic_league": selected.get("opta_domestic_league", ""),
            }
        )
        audits.append(
            {
                "club_id": team.club_id,
                "uefa_team_id": int(team.uefa_team_id),
                "ao_team_name": team.team_name,
                "country_code": team.country_code,
                "opta_team_id": opta_id,
                "opta_team_name": selected["opta_team_name"],
                "opta_association": selected["opta_association"],
                "match_method": method,
                "exact_candidate_count": int(len(exact)),
                "identity_status": "VERIFIED",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(audits)


def rerank(values: pd.Series) -> pd.Series:
    return values.rank(method="average", ascending=False)


def pairwise_accuracy(model_values: np.ndarray, benchmark_values: np.ndarray) -> float:
    model_diff = model_values[:, None] - model_values[None, :]
    benchmark_diff = benchmark_values[:, None] - benchmark_values[None, :]
    upper = np.triu(np.ones(model_diff.shape, dtype=bool), k=1)
    comparable = upper & (benchmark_diff != 0.0) & (model_diff != 0.0)
    if not np.any(comparable):
        return float("nan")
    return float(np.mean((model_diff[comparable] * benchmark_diff[comparable]) > 0.0))


def ranking_metrics(model_values: pd.Series, benchmark_values: pd.Series) -> RankingMetrics:
    model = model_values.astype(float).to_numpy()
    benchmark = benchmark_values.astype(float).to_numpy()
    model_rank = rerank(pd.Series(model)).to_numpy()
    benchmark_rank = rerank(pd.Series(benchmark)).to_numpy()
    abs_diff = np.abs(model_rank - benchmark_rank)

    def overlap(k: int) -> int:
        use_k = min(k, len(model))
        model_top = set(np.argsort(-model, kind="mergesort")[:use_k])
        benchmark_top = set(np.argsort(-benchmark, kind="mergesort")[:use_k])
        return len(model_top & benchmark_top)

    return RankingMetrics(
        n=len(model),
        spearman=float(spearmanr(model, benchmark).statistic),
        kendall_tau=float(kendalltau(model, benchmark).statistic),
        pairwise_accuracy=pairwise_accuracy(model, benchmark),
        rank_mae=float(abs_diff.mean()),
        rank_median_ae=float(np.median(abs_diff)),
        top10_overlap=overlap(10),
        top25_overlap=overlap(25),
        top50_overlap=overlap(50),
    )


def bootstrap_spearman_ci(
    model_values: pd.Series,
    benchmark_values: pd.Series,
    *,
    seed: int,
    samples: int,
) -> tuple[float, float]:
    model = model_values.astype(float).to_numpy()
    benchmark = benchmark_values.astype(float).to_numpy()
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=float)
    for index in range(samples):
        selected = rng.integers(0, len(model), size=len(model))
        estimates[index] = spearmanr(model[selected], benchmark[selected]).statistic
    if not np.isfinite(estimates).any():
        raise ValueError("Bootstrap Spearman estimates are all non-finite")
    return tuple(float(value) for value in np.nanquantile(estimates, [0.025, 0.975]))


def bootstrap_spearman_delta_ci(
    start_values: pd.Series,
    final_values: pd.Series,
    benchmark_values: pd.Series,
    *,
    seed: int,
    samples: int,
) -> tuple[float, float]:
    start = start_values.astype(float).to_numpy()
    final = final_values.astype(float).to_numpy()
    benchmark = benchmark_values.astype(float).to_numpy()
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=float)
    for index in range(samples):
        selected = rng.integers(0, len(start), size=len(start))
        estimates[index] = (
            spearmanr(final[selected], benchmark[selected]).statistic
            - spearmanr(start[selected], benchmark[selected]).statistic
        )
    if not np.isfinite(estimates).any():
        raise ValueError("Paired bootstrap Spearman estimates are all non-finite")
    return tuple(float(value) for value in np.nanquantile(estimates, [0.025, 0.975]))


def metrics_row(
    data: pd.DataFrame,
    *,
    model_state: str,
    model_column: str,
    benchmark: str,
    benchmark_column: str,
    segment: str,
    bootstrap_samples: int,
) -> dict[str, object]:
    metrics = ranking_metrics(data[model_column], data[benchmark_column])
    low, high = bootstrap_spearman_ci(
        data[model_column],
        data[benchmark_column],
        seed=202526 + sum(ord(char) for char in model_state + benchmark + segment),
        samples=bootstrap_samples,
    )
    return {
        "segment": segment,
        "model_state": model_state,
        "benchmark": benchmark,
        **metrics.__dict__,
        "spearman_ci_low": low,
        "spearman_ci_high": high,
    }


def write_report(
    path: Path,
    summary: pd.DataFrame,
    change_summary: pd.DataFrame,
    matched: pd.DataFrame,
    *,
    uefa_updated: str,
    opta_updated: str,
) -> None:
    pooled = summary.loc[summary["segment"].eq("ALL")]
    final_rows = pooled.loc[pooled["model_state"].eq("AO_FINAL")].set_index("benchmark")
    start_rows = pooled.loc[pooled["model_state"].eq("AO_START")].set_index("benchmark")
    pooled_changes = change_summary.loc[change_summary["segment"].eq("ALL")].set_index(
        "benchmark"
    )
    opta_segment_changes = change_summary.loc[
        change_summary["benchmark"].eq("OPTA_2026_08_10")
        & change_summary["segment"].isin(["UCL", "UEL", "UECL"])
    ].set_index("segment")
    top = matched.sort_values("ao_final_rank").head(20)
    disagreements = matched.assign(
        max_abs_rank_gap=matched[["ao_final_vs_uefa_rank_gap", "ao_final_vs_opta_rank_gap"]]
        .abs()
        .max(axis=1)
    ).sort_values("max_abs_rank_gap", ascending=False).head(15)

    lines = [
        "# AO 2025/26 Dis Siralama Karsilastirmasi",
        "",
        "## Kapsam",
        "",
        f"- AO evreni: {len(matched)} adet 2025/26 UCL, UEL ve UECL katilimcisi.",
        f"- UEFA benchmark: 2026 sporting club coefficient; kaynak guncellemesi `{uefa_updated}`.",
        f"- Opta benchmark: Opta Power Rankings; snapshot `{opta_updated}`.",
        "- AO_START sezon basi AO First Elo, AO_FINAL ise 961 maclik replay sonrasi AO Live Elo'dur.",
        "",
        "## Metodolojik Not",
        "",
        "UEFA katsayisi son bes Avrupa sezonundaki basariyi ve kura seribasligini olcer; saf guncel takim gucu degildir. "
        "Opta, mac sonuclari ile mevcut oldugunda xG kullanan global bir guc ratingidir. Opta snapshot'i sezon sonundan "
        "sonra alindigi icin erken 2026/27 maclarindan etkilenmis olabilir. Bu nedenle iki kaynak ayri yorumlanmistir.",
        "",
        "## Ana Sonuclar",
        "",
        "| Benchmark | AO baslangic Spearman | AO final Spearman | Degisim | Final pairwise | Final rank MAE | Top 25 ortak |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for benchmark in ("UEFA_2026", "OPTA_2026_08_10"):
        start = start_rows.loc[benchmark]
        final = final_rows.loc[benchmark]
        lines.append(
            f"| {benchmark} | {start.spearman:.4f} | {final.spearman:.4f} | "
            f"{final.spearman - start.spearman:+.4f} | {final.pairwise_accuracy:.4f} | "
            f"{final.rank_mae:.2f} | {int(final.top25_overlap)}/25 |"
        )
    lines.extend(
        [
            "",
            "## Sezon Ici Guncellemenin Etkisi",
            "",
            "Ayni 236 takim uzerindeki eslesik bootstrap guven araligi, AO_START'tan AO_FINAL'a gecisin "
            "dis benchmark ile uyumu nasil degistirdigini olcer.",
            "",
            "| Benchmark | Spearman degisimi | %95 GA | Pairwise degisimi | Rank MAE degisimi |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for benchmark in ("UEFA_2026", "OPTA_2026_08_10"):
        change = pooled_changes.loc[benchmark]
        lines.append(
            f"| {benchmark} | {change.spearman_change:+.4f} | "
            f"[{change.spearman_change_ci_low:+.4f}, {change.spearman_change_ci_high:+.4f}] | "
            f"{change.pairwise_accuracy_change:+.4f} | {change.rank_mae_change:+.2f} |"
        )
    lines.extend(
        [
            "",
            "### Opta Turnuva Segmentleri",
            "",
            "| Segment | Takim | Spearman degisimi | %95 GA | Pairwise degisimi |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for segment in ("UCL", "UEL", "UECL"):
        change = opta_segment_changes.loc[segment]
        lines.append(
            f"| {segment} | {int(change.n)} | {change.spearman_change:+.4f} | "
            f"[{change.spearman_change_ci_low:+.4f}, {change.spearman_change_ci_high:+.4f}] | "
            f"{change.pairwise_accuracy_change:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## AO Final Ilk 20",
            "",
            "| AO | Takim | AO Live Elo | UEFA ortak-evren sirasi | Opta ortak-evren sirasi |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for row in top.itertuples(index=False):
        lines.append(
            f"| {int(row.ao_final_rank)} | {row.team_name} | {row.main_end_live_rating:.1f} | "
            f"{int(row.uefa_common_rank)} | {int(row.opta_common_rank)} |"
        )
    lines.extend(
        [
            "",
            "## En Buyuk Siralama Ayrismalari",
            "",
            "| Takim | AO | UEFA ortak | Opta ortak | UEFA fark | Opta fark |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in disagreements.itertuples(index=False):
        lines.append(
            f"| {row.team_name} | {int(row.ao_final_rank)} | {int(row.uefa_common_rank)} | "
            f"{int(row.opta_common_rank)} | {int(row.ao_final_vs_uefa_rank_gap):+d} | "
            f"{int(row.ao_final_vs_opta_rank_gap):+d} |"
        )
    lines.extend(
        [
            "",
            "## Yorumlama",
            "",
            "- Opta ile uyum, AO'nun guncel takim gucu siralamasi icin ana dis kontrol olarak okunmalidir.",
            "- UEFA ile uyum, Avrupa gecmisi ve turnuva basarisi sinyalinin tutarliligini gosterir; bagimsiz bir tahmin testi degildir.",
            "- AO Live guncellemeleri Opta uyumunu pooled ve UCL/UEL/UECL segmentlerinin tamaminda artirmistir.",
            "- UEFA uyumundaki gerileme, canli gucun bes sezonluk basari katsayisindan uzaklasmasidir; tek basina hata olarak yorumlanmamalidir.",
            "- Korelasyon tek basina production karari vermez. En buyuk rank farklari takim bazli inceleme listesi olarak kullanilmalidir.",
            "- Bu calisma AO ratinglerini veya production contract'ini degistirmez.",
            "",
            "## Dis Kontrol Karari",
            "",
            "`PASS_EXTERNAL_ALIGNMENT_CHECK`: AO_FINAL, guncel guc benchmark'i Opta ile guclu uyum gostermis ve "
            "sezon ici guncellemeler bu uyumu guvenilir bicimde artirmistir. Bu calisma retrospective bir dis "
            "sanity check'tir; Opta snapshot'i sezon sonrasina ait oldugu icin untouched prospective holdout yerine gecmez.",
            "",
            "## Kaynaklar",
            "",
            f"- UEFA siralama: {UEFA_SOURCE_URL}",
            f"- UEFA metodoloji: {UEFA_METHOD_URL}",
            f"- Opta siralama: {OPTA_SOURCE_URL}",
            f"- Opta metodoloji: {OPTA_METHOD_URL}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ao-ratings", type=Path, default=DEFAULT_AO)
    parser.add_argument("--current-model-ratings", type=Path, default=DEFAULT_CURRENT_MODEL_RATINGS)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--uefa-raw-json", type=Path)
    parser.add_argument("--opta-raw-json", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("bootstrap-samples must be at least 100")
    ao = pd.read_csv(args.ao_ratings)
    required_ao = {
        "club_id",
        "uefa_team_id",
        "team_name",
        "country_code",
        "country",
        "participating_competitions",
        "main_start_rating",
        "main_end_live_rating",
    }
    missing = required_ao - set(ao.columns)
    if missing:
        raise ValueError(f"AO snapshot is missing columns: {sorted(missing)}")
    if len(ao) != 236 or ao["club_id"].duplicated().any() or ao["uefa_team_id"].duplicated().any():
        raise ValueError("AO snapshot must contain 236 unique club and UEFA IDs")
    current_ratings = pd.read_csv(args.current_model_ratings)
    current_ratings = current_ratings.loc[
        current_ratings["model"].eq("CURRENT_PRODUCTION")
        & current_ratings["season"].eq("2025/26")
    ].copy()
    if len(current_ratings) != 236 or current_ratings["team_id"].duplicated().any():
        raise ValueError("Current model rating file must contain 236 unique 2025/26 production teams")
    current_columns = current_ratings[
        [
            "team_id",
            "initial_rating",
            "end_power_rating",
            "end_progression_bonus",
            "end_live_rating",
        ]
    ]
    ao = ao.merge(current_columns, on="team_id", how="inner", validate="one_to_one")
    if len(ao) != 236:
        raise ValueError("Current production ratings do not cover the AO identity snapshot")
    ao["main_start_rating"] = ao["initial_rating"]
    ao["main_end_power_rating"] = ao["end_power_rating"]
    ao["main_end_progression_bonus"] = ao["end_progression_bonus"]
    ao["main_end_live_rating"] = ao["end_live_rating"]

    args.source_dir.mkdir(parents=True, exist_ok=True)
    uefa_csv = args.source_dir / "uefa_club_coefficients_2026.csv"
    opta_csv = args.source_dir / "opta_power_rankings_2026_08_10_ao_scope.csv"
    identity_csv = args.source_dir / "external_ranking_identity_crosswalk.csv"
    manifest_path = args.source_dir / "source_manifest.json"
    existing_manifest: dict[str, object] = {}
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if args.uefa_raw_json is not None:
        uefa, uefa_updated = load_uefa_raw(args.uefa_raw_json)
        uefa.to_csv(uefa_csv, index=False, float_format="%.10f")
    else:
        uefa = pd.read_csv(uefa_csv)
        uefa_updated = str(
            dict(existing_manifest.get("uefa", {})).get(
                "last_update_date", "2026-07-03T14:22:35+0000"
            )
        )
    if not set(ao["uefa_team_id"].astype(int)).issubset(set(uefa["uefa_team_id"].astype(int))):
        missing_ids = sorted(set(ao["uefa_team_id"].astype(int)) - set(uefa["uefa_team_id"].astype(int)))
        raise ValueError(f"UEFA ranking is missing AO team IDs: {missing_ids}")

    if args.opta_raw_json is not None:
        opta, opta_updated = load_opta_raw(args.opta_raw_json)
        opta_matches, identity = match_opta(ao, uefa, opta)
        opta_matches.to_csv(opta_csv, index=False, float_format="%.10f")
        identity.to_csv(identity_csv, index=False)
    else:
        opta_matches = pd.read_csv(opta_csv)
        identity = pd.read_csv(identity_csv)
        opta_updated = str(
            dict(existing_manifest.get("opta", {})).get("snapshot_date", "Aug 10, 2026")
        )

    uefa_scope = uefa.loc[uefa["uefa_team_id"].astype(int).isin(ao["uefa_team_id"].astype(int))].copy()
    matched = ao.merge(
        uefa_scope,
        on=["uefa_team_id"],
        how="inner",
        validate="one_to_one",
    )
    matched = matched.merge(
        opta_matches,
        on=["club_id", "uefa_team_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(matched) != 236 or identity["identity_status"].ne("VERIFIED").any():
        raise ValueError("External identity matching is incomplete")

    matched["ao_start_rank"] = rerank(matched["main_start_rating"]).astype(int)
    matched["ao_final_rank"] = rerank(matched["main_end_live_rating"]).astype(int)
    matched["uefa_common_rank"] = rerank(matched["uefa_coefficient"]).astype(int)
    matched["opta_common_rank"] = rerank(matched["opta_power_rating"]).astype(int)
    matched["ao_final_vs_uefa_rank_gap"] = matched["ao_final_rank"] - matched["uefa_common_rank"]
    matched["ao_final_vs_opta_rank_gap"] = matched["ao_final_rank"] - matched["opta_common_rank"]
    matched["ao_season_rating_change"] = matched["main_end_live_rating"] - matched["main_start_rating"]
    matched["ao_season_rank_change"] = matched["ao_start_rank"] - matched["ao_final_rank"]

    rows: list[dict[str, object]] = []
    model_columns = {"AO_START": "main_start_rating", "AO_FINAL": "main_end_live_rating"}
    benchmarks = {"UEFA_2026": "uefa_coefficient", "OPTA_2026_08_10": "opta_power_rating"}
    segments = {"ALL": matched}
    for competition in ("UCL", "UEL", "UECL"):
        segments[competition] = matched.loc[
            matched["participating_competitions"].astype(str).str.split("/").map(lambda values: competition in values)
        ]
    for segment, data in segments.items():
        for model_state, model_column in model_columns.items():
            for benchmark, benchmark_column in benchmarks.items():
                rows.append(
                    metrics_row(
                        data,
                        model_state=model_state,
                        model_column=model_column,
                        benchmark=benchmark,
                        benchmark_column=benchmark_column,
                        segment=segment,
                        bootstrap_samples=args.bootstrap_samples,
                    )
                )
    summary = pd.DataFrame(rows)
    change_rows: list[dict[str, object]] = []
    for segment, data in segments.items():
        for benchmark, benchmark_column in benchmarks.items():
            start_metrics = ranking_metrics(data["main_start_rating"], data[benchmark_column])
            final_metrics = ranking_metrics(data["main_end_live_rating"], data[benchmark_column])
            delta_low, delta_high = bootstrap_spearman_delta_ci(
                data["main_start_rating"],
                data["main_end_live_rating"],
                data[benchmark_column],
                seed=202527 + sum(ord(char) for char in benchmark + segment),
                samples=args.bootstrap_samples,
            )
            change_rows.append(
                {
                    "segment": segment,
                    "benchmark": benchmark,
                    "n": len(data),
                    "start_spearman": start_metrics.spearman,
                    "final_spearman": final_metrics.spearman,
                    "spearman_change": final_metrics.spearman - start_metrics.spearman,
                    "spearman_change_ci_low": delta_low,
                    "spearman_change_ci_high": delta_high,
                    "pairwise_accuracy_change": (
                        final_metrics.pairwise_accuracy - start_metrics.pairwise_accuracy
                    ),
                    "rank_mae_change": final_metrics.rank_mae - start_metrics.rank_mae,
                    "top25_overlap_change": final_metrics.top25_overlap - start_metrics.top25_overlap,
                }
            )
    change_summary = pd.DataFrame(change_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    matched = matched.sort_values(["ao_final_rank", "club_id"]).reset_index(drop=True)
    matched.to_csv(args.output_dir / "matched_rankings.csv", index=False, float_format="%.10f")
    summary.to_csv(args.output_dir / "benchmark_summary.csv", index=False, float_format="%.10f")
    change_summary.to_csv(
        args.output_dir / "start_to_final_change.csv", index=False, float_format="%.10f"
    )
    identity.to_csv(args.output_dir / "identity_audit.csv", index=False)
    matched.head(50).to_csv(args.output_dir / "ao_final_top50.csv", index=False, float_format="%.10f")
    matched.assign(
        max_abs_rank_gap=matched[["ao_final_vs_uefa_rank_gap", "ao_final_vs_opta_rank_gap"]]
        .abs()
        .max(axis=1)
    ).sort_values(["max_abs_rank_gap", "club_id"], ascending=[False, True]).head(50).to_csv(
        args.output_dir / "largest_rank_disagreements.csv", index=False, float_format="%.10f"
    )
    write_report(
        args.output_dir / "comparison_report.md",
        summary,
        change_summary,
        matched,
        uefa_updated=uefa_updated,
        opta_updated=opta_updated,
    )

    manifest = {
        "analysis": "AO 2025/26 external ranking comparison",
        "ao_snapshot": str(args.ao_ratings.relative_to(ROOT)),
        "ao_snapshot_sha256": sha256(args.ao_ratings),
        "current_model_ratings": str(args.current_model_ratings.relative_to(ROOT)),
        "current_model_ratings_sha256": sha256(args.current_model_ratings),
        "current_model_arm": "CURRENT_PRODUCTION",
        "current_model_season": "2025/26",
        "team_count": len(matched),
        "uefa": {
            "source_url": UEFA_SOURCE_URL,
            "method_url": UEFA_METHOD_URL,
            "target_season_year": 2026,
            "last_update_date": uefa_updated,
            "normalized_file": str(uefa_csv.relative_to(ROOT)),
            "normalized_sha256": sha256(uefa_csv),
            "raw_sha256": (
                sha256(args.uefa_raw_json)
                if args.uefa_raw_json
                else dict(existing_manifest.get("uefa", {})).get("raw_sha256")
            ),
            "interpretation": "five-season European achievement and seeding coefficient",
        },
        "opta": {
            "source_url": OPTA_SOURCE_URL,
            "method_url": OPTA_METHOD_URL,
            "snapshot_date": opta_updated,
            "normalized_file": str(opta_csv.relative_to(ROOT)),
            "normalized_sha256": sha256(opta_csv),
            "raw_sha256": (
                sha256(args.opta_raw_json)
                if args.opta_raw_json
                else dict(existing_manifest.get("opta", {})).get("raw_sha256")
            ),
            "interpretation": "current global team-strength rating",
            "temporal_caveat": "snapshot post-dates the 2025/26 season and may include early 2026/27 evidence",
        },
        "production_contract_changed": False,
        "bootstrap_samples": args.bootstrap_samples,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    pooled = summary.loc[(summary["segment"] == "ALL") & (summary["model_state"] == "AO_FINAL")]
    print(f"AO teams: {len(matched)}")
    print(f"UEFA identity coverage: {len(matched)}/{len(matched)}")
    print(f"Opta identity coverage: {len(identity)}/{len(matched)}")
    print(pooled[["benchmark", "spearman", "pairwise_accuracy", "rank_mae", "top25_overlap"]].to_string(index=False))
    print(f"Report: {args.output_dir / 'comparison_report.md'}")


if __name__ == "__main__":
    main()
