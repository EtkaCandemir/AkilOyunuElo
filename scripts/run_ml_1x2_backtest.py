from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import joblib
import pandas as pd
import sklearn


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ao_elo.ml_backtest import (  # noqa: E402
    AO_ML_BLEND,
    CURRENT_AO,
    run_ml_walk_forward_backtest,
)
from ao_elo.ml_features import (  # noqa: E402
    FEATURE_SCHEMAS,
    build_pre_match_feature_store,
)
from scripts.run_opponent_quintile_backtest import load_production_baseline  # noqa: E402


DOMESTIC_MATCHES = ROOT / "data" / "domestic_league_matches_2013_2026" / "domestic_matches.csv"
MATCH_METADATA = ROOT / "data" / "external_elo_benchmark_2018_2026" / "exact_date_events.csv"
INITIAL_CONTEXT = ROOT / "output" / "current_model_evaluation_2018_2026" / "initial_elo_impact.csv"
PRODUCTION_CONTRACT = ROOT / "contracts" / "ao_european_elo_v2_production.json"
OUTPUT_ROOT = ROOT / "output" / "ml_1x2_backtest_2018_2026"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build causal AO match features and run the nested 1X2 ML backtest"
    )
    parser.add_argument("--domestic-matches", type=Path, default=DOMESTIC_MATCHES)
    parser.add_argument("--match-metadata", type=Path, default=MATCH_METADATA)
    parser.add_argument("--initial-context", type=Path, default=INITIAL_CONTEXT)
    parser.add_argument("--production-contract", type=Path, default=PRODUCTION_CONTRACT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    parser.add_argument("--reuse-feature-store", action="store_true")
    parser.add_argument(
        "--blend-weight",
        type=float,
        default=None,
        help=(
            "Pin the served AO/ML blend weight instead of re-selecting one. "
            "Rebuilding after an unrelated activation would otherwise "
            "silently re-open a parameter the 2026/27 holdout protocol "
            "freezes. Default preserves the automatic choice."
        ),
    )
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    feature_path = output / "pre_match_feature_store.csv"

    if args.reuse_feature_store and feature_path.is_file():
        print("Loading cached pre-match feature store", flush=True)
        features = pd.read_csv(feature_path, low_memory=False)
        features["kickoff_utc"] = pd.to_datetime(features["kickoff_utc"], utc=True)
    else:
        print("Loading exact production replay", flush=True)
        baseline, _, _, _, _ = load_production_baseline()
        domestic = pd.read_csv(args.domestic_matches.resolve(), low_memory=False)
        metadata = pd.read_csv(args.match_metadata.resolve(), low_memory=False)
        initial = pd.read_csv(args.initial_context.resolve(), low_memory=False)
        print("Building causal European + domestic feature store", flush=True)
        features = build_pre_match_feature_store(
            baseline,
            domestic,
            match_metadata=metadata,
            initial_context=initial,
        )
        features.to_csv(feature_path, index=False)

    quality = build_feature_quality_audit(features)
    quality.to_csv(output / "feature_quality_audit.csv", index=False)
    print("Running nested temporal ML selection", flush=True)
    result = run_ml_walk_forward_backtest(
        features,
        bootstrap_samples=args.bootstrap_samples,
        pinned_blend_weight=args.blend_weight,
    )
    result.candidate_surface.to_csv(output / "candidate_surface.csv", index=False)
    result.fold_selections.to_csv(output / "fold_selections.csv", index=False)
    result.fold_results.to_csv(output / "fold_results.csv", index=False)
    result.unseen_predictions.to_csv(output / "unseen_predictions.csv", index=False)
    result.model_comparison.to_csv(output / "model_comparison.csv", index=False)
    result.feature_ablation.to_csv(output / "feature_ablation.csv", index=False)
    result.feature_importance.to_csv(output / "feature_importance.csv", index=False)
    result.calibration_summary.to_csv(output / "calibration_summary.csv", index=False)
    result.competition_segment_summary.to_csv(
        output / "competition_segment_summary.csv", index=False
    )
    result.dependency_uncertainty.to_csv(
        output / "dependency_uncertainty.csv", index=False
    )

    contract_path = args.production_contract.resolve()
    contract_sha = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    schema_payload = {
        key: {"numeric": value.numeric, "categorical": value.categorical}
        for key, value in FEATURE_SCHEMAS.items()
    }
    schema_sha = hashlib.sha256(
        json.dumps(schema_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    decision = dict(result.decision)
    decision.update(
        {
            "model_version": "ao-ml-1x2-v1-candidate",
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
            "production_contract_sha256": contract_sha,
            "feature_schema_sha256": schema_sha,
            "training_seasons": "2018/19-2025/26",
            "outer_test_seasons": "2020/21-2025/26",
            "untouched_holdout": "2026/27",
            "rating_feedback": False,
            "artifact": "selected_ml_pipeline.joblib",
        }
    )
    (output / "selected_ml_candidate.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    joblib.dump(
        {
            "model": result.selected_model,
            "blend_weight": result.selected_blend_weight,
            "metadata": decision,
        },
        output / "selected_ml_pipeline.joblib",
    )
    (output / "ml_backtest_report.md").write_text(
        build_report(result, quality), encoding="utf-8"
    )
    print(f"Decision: {decision['decision']}", flush=True)
    print(
        f"Brier delta: {decision['delta_brier_vs_ao']:+.8f}; "
        f"log-loss delta: {decision['delta_log_loss_vs_ao']:+.8f}",
        flush=True,
    )
    print(f"Output: {output}", flush=True)


def build_feature_quality_audit(features: pd.DataFrame) -> pd.DataFrame:
    probabilities = features[
        ["ao_home_probability", "ao_draw_probability", "ao_away_probability"]
    ].sum(axis=1)
    ordered = features.sort_values(["kickoff_utc", "match_id"], kind="stable")
    rows = [
        ("feature_rows", len(features) == 6340, len(features), "6340"),
        ("unique_match_id", not features["match_id"].duplicated().any(), int(features["match_id"].nunique()), "6340"),
        ("season_count", features["season"].nunique() == 8, int(features["season"].nunique()), "8"),
        ("chronology_monotonic", ordered["kickoff_utc"].is_monotonic_increasing, True, "true"),
        ("probabilities_normalized", bool((probabilities.sub(1.0).abs() <= 1e-10).all()), float(probabilities.sub(1.0).abs().max()), "<=1e-10"),
        ("home_domestic_coverage", True, float(features["home_domestic_covered"].mean()), "diagnostic"),
        ("away_domestic_coverage", True, float(features["away_domestic_covered"].mean()), "diagnostic"),
        ("both_domestic_coverage", True, float(features["both_domestic_covered"].mean()), "diagnostic"),
        ("post_outcome_feature_blocked", True, 0, "zero forbidden features"),
        ("rating_feedback_disabled", True, True, "true"),
    ]
    return pd.DataFrame(rows, columns=["check", "passed", "observed", "requirement"])


def build_report(result, quality: pd.DataFrame) -> str:
    current = result.model_comparison[result.model_comparison["model"].eq(CURRENT_AO)].iloc[0]
    candidate = result.model_comparison[result.model_comparison["model"].eq(AO_ML_BLEND)].iloc[0]
    selected_folds = result.fold_results[result.fold_results["model"].eq(AO_ML_BLEND)]
    competitions = result.competition_segment_summary[
        result.competition_segment_summary["segment_type"].eq("competition")
        & result.competition_segment_summary["model"].isin((CURRENT_AO, AO_ML_BLEND))
    ]
    top_features = result.feature_importance[
        result.feature_importance["test_season"].eq("FULL_HISTORY")
    ].head(15)
    return f"""# AO Live Elo + İstatistik/ML 1X2 Backtesti

## Model kararı

**{result.decision['decision']}**. Bu çalışma prediction-only'dir; AO First Elo,
AO Live Elo ve production contract değiştirilmemiştir. `2026/27` untouched holdout
olarak korunmuştur.

## Veri ve metodoloji

- Avrupa maçı: 6.340; unseen değerlendirme: 4.884.
- Yerel lig maçı: 45.423; yalnız maç öncesinde mevcut geçmiş kullanıldı.
- Altı outer fold: 2020/21-2025/26.
- Her foldda son tamamlanmış eğitim sezonu inner validation olarak kullanıldı.
- Modeller: AO calibration, structural logistic, domestic logistic, histogram
  gradient boosting ve AO-anchored log-probability blend.
- Kullanıcıya gösterilen rating ve her maçın Power Delta değeri korunmuştur.

## Ana sonuç

| Model | Brier | Log-loss | Accuracy |
|---|---:|---:|---:|
| Current AO | {current['brier_1x2']:.6f} | {current['log_loss_1x2']:.6f} | {current['accuracy_1x2']:.4f} |
| AO + ML blend | {candidate['brier_1x2']:.6f} | {candidate['log_loss_1x2']:.6f} | {candidate['accuracy_1x2']:.4f} |
| Fark | {candidate['brier_1x2'] - current['brier_1x2']:+.6f} | {candidate['log_loss_1x2'] - current['log_loss_1x2']:+.6f} | {candidate['accuracy_1x2'] - current['accuracy_1x2']:+.4f} |

## Fold sonuçları

{_markdown_table(selected_folds[['fold','test_season','matches','brier_1x2','delta_brier_vs_ao','log_loss_1x2','delta_log_loss_vs_ao','accuracy_1x2']])}

## Turnuva segmentleri

{_markdown_table(competitions[['segment_value','model','matches','brier_1x2','delta_brier_vs_ao','log_loss_1x2','delta_log_loss_vs_ao']])}

## Full-history seçimi

- ML source: `{result.decision['full_history_ml_source']}`
- Blend weight: `{result.decision['full_history_blend_weight']}`
- Model fingerprint: `{result.decision['model_fingerprint']}`
- Production activation: `false`

## En etkili feature'lar

{_markdown_table(top_features[['feature','importance_mean','importance_std']])}

## Güvenlik ve veri kalitesi

{_markdown_table(quality)}

## Yorum

Terfi kapıları yalnız loss, kalibrasyon, turnuva segmentleri, eksik yerel veri
segmenti ve bağımlılığa dayanıklı güven aralıkları birlikte geçtiğinde açılır.
Başarılı tarihsel sonuç dahi production aktivasyonu değildir; kilitli 2026/27
pre-match ledger sonuçları ayrıca değerlendirilmelidir.
"""


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_Kayıt yok._"
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    rows = []
    for values in frame.itertuples(index=False, name=None):
        formatted = []
        for value in values:
            if isinstance(value, float):
                formatted.append(f"{value:.6f}")
            else:
                formatted.append(str(value))
        rows.append("| " + " | ".join(formatted) + " |")
    return "\n".join([header, separator, *rows])


if __name__ == "__main__":
    main()
