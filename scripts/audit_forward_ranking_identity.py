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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ao_elo.config import AOEuropeanEloConfig  # noqa: E402
from ao_elo.domestic_surprise import DomesticSurpriseConfig  # noqa: E402
from ao_elo.evaluation import schedule_adjusted_team_performance  # noqa: E402
from scripts.run_domestic_surprise_backtest import (  # noqa: E402
    build_adjustments,
    evaluate_live,
    live_config_from_contract,
)
from scripts.run_dynamic_core_calibration import DynamicCoreConfig  # noqa: E402
from scripts.run_final_robustness import summarize_ranking  # noqa: E402
from scripts.run_match_context_backtest import (  # noqa: E402
    load_context_data,
    read_context_events,
)


STATIC_ROOT = ROOT / "data" / "backtest_stage_b_2018_2026"
EVENTS_PATH = ROOT / "data" / "external_elo_benchmark_2018_2026" / "exact_date_events.csv"
DYNAMIC_MANIFEST = ROOT / "output" / "v2_dynamic_calibration_2018_2026" / "selected_dynamic_model.json"
FINAL_CONTRACT = ROOT / "contracts" / "ao_european_elo_v2_final_candidate.json"
IDENTITY_PATH = ROOT / "data" / "club_identity" / "team_season_identity.csv"
FEATURES_PATH = ROOT / "output" / "domestic_surprise_backtest_2018_2026" / "domestic_surprise_features.csv"
OUTPUT_ROOT = ROOT / "output" / "club_identity_audit_2018_2026"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare legacy local-team ranking with permanent club identity"
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    dynamic_manifest = json.loads(DYNAMIC_MANIFEST.read_text(encoding="utf-8"))
    static_config = AOEuropeanEloConfig(**dynamic_manifest["static_config"])
    final_contract = json.loads(FINAL_CONTRACT.read_text(encoding="utf-8"))
    events = read_context_events(EVENTS_PATH)
    datasets, _ = load_context_data(STATIC_ROOT, EVENTS_PATH, static_config)
    features = pd.read_csv(FEATURES_PATH)
    baseline_adjustments = build_adjustments(
        features, DomesticSurpriseConfig(), static_config
    )
    evaluation = evaluate_live(
        tuple(data.reserve for data in datasets),
        baseline_adjustments,
        set(events["season"]),
        DynamicCoreConfig(**final_contract["dynamic_core"]),
        live_config_from_contract(final_contract),
    )
    target = schedule_adjusted_team_performance(events)
    allowed = set(target["season"])
    identity = pd.read_csv(IDENTITY_PATH, dtype={"uefa_team_id": "string"})
    legacy = summarize_ranking(
        evaluation.end_ratings,
        target,
        allowed_target_seasons=allowed,
    ).assign(method="LEGACY_LOCAL_TEAM_ID")
    corrected = summarize_ranking(
        evaluation.end_ratings,
        target,
        allowed_target_seasons=allowed,
        identity=identity,
    ).assign(method="PERMANENT_CLUB_ID")
    comparison = legacy.merge(
        corrected,
        on="competition",
        suffixes=("_legacy", "_permanent"),
        validate="one_to_one",
    )
    comparison["ranking_score_difference"] = (
        comparison["ranking_score_permanent"] - comparison["ranking_score_legacy"]
    )
    comparison["pairwise_accuracy_difference"] = (
        comparison["pairwise_accuracy_permanent"]
        - comparison["pairwise_accuracy_legacy"]
    )
    affected = affected_backtests()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_root / "ranking_identity_comparison.csv", index=False)
    affected.to_csv(output_root / "affected_backtest_inventory.csv", index=False)
    evaluation.end_ratings.to_csv(output_root / "baseline_end_ratings.csv", index=False)
    write_report(output_root / "identity_audit_report.md", comparison, affected)
    print("Forward-ranking identity audit complete")
    print(comparison.to_string(index=False))
    print(f"Report: {output_root / 'identity_audit_report.md'}")


def affected_backtests() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "script": "scripts/run_final_robustness.py",
                "output": "output/final_robustness_2018_2026",
                "identity_code_fixed": True,
                "historical_output_status": "STALE_REQUIRES_RERUN",
            },
            {
                "script": "scripts/run_match_context_backtest.py",
                "output": "output/match_context_backtest_2018_2026",
                "identity_code_fixed": True,
                "historical_output_status": "STALE_REQUIRES_RERUN",
            },
            {
                "script": "scripts/run_dynamic_k_backtest.py",
                "output": "output/dynamic_k_backtest_2018_2026",
                "identity_code_fixed": True,
                "historical_output_status": "STALE_REQUIRES_RERUN",
            },
            {
                "script": "scripts/run_controlled_goal_progression_backtest.py",
                "output": "output/controlled_goal_progression_backtest_2018_2026",
                "identity_code_fixed": True,
                "historical_output_status": "STALE_REQUIRES_RERUN",
            },
            {
                "script": "scripts/run_fixed_tournament_bonus_backtest.py",
                "output": "output/fixed_tournament_bonus_backtest_2018_2026",
                "identity_code_fixed": True,
                "historical_output_status": "STALE_REQUIRES_RERUN",
            },
        ]
    )


def write_report(path: Path, comparison: pd.DataFrame, affected: pd.DataFrame) -> None:
    pooled = comparison.loc[comparison["competition"].eq("ALL")].iloc[0]
    lines = [
        "# Kalici Kulup Kimligi ve Forward Ranking Denetimi",
        "",
        "Sonuc: sezonlar arasi birlestirmede yerel `team_id` kullanimi gecersizdir.",
        "Yerel kimlik yalniz kendi sezonu icinde anlamlidir. Tum sezonlar arasi ranking",
        "hesaplari `club_id` ile yapilmalidir.",
        "",
        "## Ana Etki",
        "",
        f"- Legacy pooled Spearman: `{pooled['ranking_score_legacy']:.6f}`",
        f"- Kalici-ID pooled Spearman: `{pooled['ranking_score_permanent']:.6f}`",
        f"- Legacy pooled pairwise: `{pooled['pairwise_accuracy_legacy']:.6f}`",
        f"- Kalici-ID pooled pairwise: `{pooled['pairwise_accuracy_permanent']:.6f}`",
        f"- Dogru ortak kulup gozlemi: `{int(pooled['team_weight_permanent'])}`",
        "",
        "Kalici-ID sonucu daha yuksek olmakla birlikte asil bulgu metrik artisi degildir:",
        "legacy yontem farkli kulüpleri ayni yerel sira numarasi nedeniyle yanlis eslestirmistir.",
        "",
        "## Turnuva Karsilastirmasi",
        "",
        comparison.to_csv(index=False),
        "",
        "## Yeniden Calistirilacak Backtestler",
        "",
        affected.to_csv(index=False),
        "",
        "Kod yollari kalici kimlige gecirildi. Eski output dosyalari yeniden calistirilana",
        "kadar model karari icin kullanilmamalidir. Ayni sezon icindeki Brier, log-loss,",
        "mac guncellemesi ve rating degerleri bu kimlik hatasindan etkilenmez.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
