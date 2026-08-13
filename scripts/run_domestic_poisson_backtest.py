from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import pandas as pd
import scipy
import sklearn


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ao_elo.domestic_poisson import evaluate_domestic_candidates  # noqa: E402
from ao_elo.domestic_poisson_backtest import (  # noqa: E402
    AO_ML_POISSON_BLEND,
    AO_POISSON_BLEND,
    CURRENT_AO,
    CURRENT_ML_BLEND,
    run_domestic_poisson_walk_forward_backtest,
)
from ao_elo.ml_features import build_pre_match_feature_store  # noqa: E402
from scripts.run_opponent_quintile_backtest import load_production_baseline  # noqa: E402


DOMESTIC_MATCHES = ROOT / "data" / "domestic_league_matches_2013_2026" / "domestic_matches.csv"
DOMESTIC_BRIDGE = ROOT / "data" / "domestic_league_matches_2013_2026" / "domestic_team_bridge.csv"
MATCH_METADATA = ROOT / "data" / "external_elo_benchmark_2018_2026" / "exact_date_events.csv"
INITIAL_CONTEXT = ROOT / "output" / "current_model_evaluation_2018_2026" / "initial_elo_impact.csv"
ML_FEATURE_STORE = ROOT / "output" / "ml_1x2_backtest_2018_2026" / "pre_match_feature_store.csv"
CURRENT_ML_PREDICTIONS = ROOT / "output" / "ml_1x2_backtest_2018_2026" / "unseen_predictions.csv"
PRODUCTION_CONTRACT = ROOT / "contracts" / "ao_european_elo_v2_production.json"
OUTPUT_ROOT = ROOT / "output" / "domestic_poisson_backtest_2018_2026"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the leakage-safe domestic attack/defence Poisson backtest"
    )
    parser.add_argument("--domestic-matches", type=Path, default=DOMESTIC_MATCHES)
    parser.add_argument("--domestic-bridge", type=Path, default=DOMESTIC_BRIDGE)
    parser.add_argument("--feature-store", type=Path, default=ML_FEATURE_STORE)
    parser.add_argument("--current-ml-predictions", type=Path, default=CURRENT_ML_PREDICTIONS)
    parser.add_argument("--production-contract", type=Path, default=PRODUCTION_CONTRACT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    parser.add_argument("--rebuild-domestic-surface", action="store_true")
    args = parser.parse_args()

    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    domestic = pd.read_csv(args.domestic_matches.resolve(), low_memory=False)
    bridge = pd.read_csv(args.domestic_bridge.resolve(), low_memory=False)
    features = _load_or_build_feature_store(args.feature_store.resolve(), domestic)
    current_ml = pd.read_csv(args.current_ml_predictions.resolve(), low_memory=False)
    surface_path = output / "domestic_prequential_results.csv"
    if surface_path.is_file() and not args.rebuild_domestic_surface:
        print("Loading cached 54-candidate domestic prequential surface", flush=True)
        surface = pd.read_csv(surface_path)
    else:
        print("Evaluating 54 dynamic domestic Poisson candidates", flush=True)
        surface = evaluate_domestic_candidates(domestic)
        surface.to_csv(surface_path, index=False)

    contract_path = args.production_contract.resolve()
    contract_before = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    print("Running six-fold nested European transfer backtest", flush=True)
    result = run_domestic_poisson_walk_forward_backtest(
        features,
        domestic,
        bridge,
        surface,
        current_ml,
        bootstrap_samples=args.bootstrap_samples,
    )
    contract_after = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    if contract_after != contract_before:
        raise ValueError("Production contract changed during prediction-only backtest")

    result.domestic_prequential_results.to_csv(surface_path, index=False)
    result.fold_domestic_selections.to_csv(
        output / "fold_domestic_selections.csv", index=False
    )
    result.domestic_poisson_feature_store.to_csv(
        output / "domestic_poisson_feature_store.csv", index=False
    )
    result.fold_transfer_parameters.to_csv(
        output / "fold_transfer_parameters.csv", index=False
    )
    result.fold_results.to_csv(output / "fold_results.csv", index=False)
    result.unseen_predictions.to_csv(output / "unseen_predictions.csv", index=False)
    result.model_comparison.to_csv(output / "model_comparison.csv", index=False)
    result.feature_ablation.to_csv(output / "feature_ablation.csv", index=False)
    result.competition_coverage_summary.to_csv(
        output / "competition_coverage_summary.csv", index=False
    )
    result.scoreline_diagnostics.to_csv(
        output / "scoreline_diagnostics.csv", index=False
    )
    result.dependency_uncertainty.to_csv(
        output / "dependency_uncertainty.csv", index=False
    )
    decision = dict(result.selected_candidate)
    decision.update(
        {
            "model_version": "ao-domestic-poisson-1x2-v1-shadow",
            "production_contract_sha256": contract_after,
            "python_version": platform.python_version(),
            "scipy_version": scipy.__version__,
            "scikit_learn_version": sklearn.__version__,
            "development_matches": 6340,
            "unseen_matches": 4884,
            "domestic_matches": int(len(domestic)),
            "domestic_leagues": int(domestic["sportsdb_league_id"].nunique()),
            "domestic_source_teams": int(
                len(
                    set(domestic["home_source_team_id"].astype(str))
                    | set(domestic["away_source_team_id"].astype(str))
                )
            ),
            "mapped_ao_clubs": int(bridge["ao_club_id"].notna().sum()),
        }
    )
    (output / "selected_candidate.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "backtest_report.md").write_text(
        build_report(result, decision), encoding="utf-8"
    )
    print(f"Decision: {decision['decision']}", flush=True)
    print(
        f"AO delta Brier={decision['delta_brier_vs_ao']:+.8f}; "
        f"log-loss={decision['delta_log_loss_vs_ao']:+.8f}",
        flush=True,
    )
    print(
        f"Current ML delta Brier={decision['delta_brier_vs_current_ml']:+.8f}; "
        f"log-loss={decision['delta_log_loss_vs_current_ml']:+.8f}",
        flush=True,
    )
    print(f"Output: {output}", flush=True)


def _load_or_build_feature_store(path: Path, domestic: pd.DataFrame) -> pd.DataFrame:
    if path.is_file():
        features = pd.read_csv(path, low_memory=False)
        features["kickoff_utc"] = pd.to_datetime(features["kickoff_utc"], utc=True)
        return features
    print("Existing ML feature store missing; rebuilding causal base features", flush=True)
    baseline, _, _, _, _ = load_production_baseline()
    metadata = pd.read_csv(MATCH_METADATA, low_memory=False)
    initial = pd.read_csv(INITIAL_CONTEXT, low_memory=False)
    return build_pre_match_feature_store(
        baseline,
        domestic,
        match_metadata=metadata,
        initial_context=initial,
    )


def build_report(result, decision: dict[str, object]) -> str:
    comparison = result.model_comparison[
        result.model_comparison["model"].isin(
            (CURRENT_AO, AO_POISSON_BLEND, CURRENT_ML_BLEND, AO_ML_POISSON_BLEND)
        )
    ]
    folds = result.fold_results[
        result.fold_results["model"].eq(AO_POISSON_BLEND)
    ]
    segments = result.competition_coverage_summary[
        result.competition_coverage_summary["model"].isin(
            (CURRENT_AO, AO_ML_POISSON_BLEND)
        )
        & result.competition_coverage_summary["segment_type"].isin(
            ("competition", "coverage")
        )
    ]
    selected = result.fold_domestic_selections[
        [
            "fold",
            "test_season",
            "team_learning_rate",
            "season_carry",
            "shrinkage_matches",
            "venue_context",
        ]
    ]
    return f"""# Yerel Lig Destekli Dynamic Poisson/Dixon-Coles Backtesti

## Model kararı

**{decision['decision']}**. Katman yalnız maç öncesi 1X2 ve skor tahmini üretir;
AO First Elo, AO Live Elo, Power Delta ve production contract değiştirilmemiştir.
`2026/27` untouched holdout olarak korunmuştur.

## Veri ve yöntem

- Yerel lig: {decision['domestic_matches']:,} maç, {decision['domestic_leagues']} lig,
  {decision['domestic_source_teams']} kaynak takım ve {decision['mapped_ao_clubs']} AO eşleşmesi.
- Avrupa: 6.340 geliştirme maçı; 4.884 unseen maç ve altı outer fold.
- Yerel hyperparameter seçimi geçmiş sezon prequential gol NLL'siyle yapıldı.
- Avrupa transfer, Dixon-Coles, logistic ve blend seçimleri yalnız inner geçmişte yapıldı.
- Ana karar metrikleri Brier ve log-loss; skor NLL diagnostiktir.

## Ana karşılaştırma

{_markdown_table(comparison)}

## Fold sonuçları

{_markdown_table(folds[['fold','test_season','matches','brier_1x2','delta_brier_vs_ao','log_loss_1x2','delta_log_loss_vs_ao','accuracy_1x2']])}

## Yerel model seçimleri

{_markdown_table(selected)}

## Turnuva ve coverage segmentleri

{_markdown_table(segments[['segment_type','segment_value','model','matches','brier_1x2','delta_brier_vs_ao','log_loss_1x2','delta_log_loss_vs_ao']])}

## Skor diagnostikleri

{_markdown_table(result.scoreline_diagnostics)}

## Karar kapıları

```json
{json.dumps(decision['gates'], indent=2, ensure_ascii=False)}
```

## Yorum

Production aktivasyonu bu çalışmanın parçası değildir. Aday yalnız önceden
tanımlanan fold, pooled loss, bağımlılığa dayanıklı belirsizlik, segment ve
kalibrasyon kapılarının tamamını geçerse terfi önerisi alır. Eksik yerel geçmiş
durumunda AO olasılıklarına birebir fallback uygulanmıştır.
"""


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_Kayıt yok._"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for values in frame.itertuples(index=False, name=None):
        formatted = []
        for value in values:
            if isinstance(value, float):
                formatted.append(f"{value:.6f}")
            else:
                formatted.append(str(value))
        lines.append("| " + " | ".join(formatted) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
