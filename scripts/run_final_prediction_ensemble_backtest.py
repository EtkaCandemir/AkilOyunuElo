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

from ao_elo.prediction_ensemble import (  # noqa: E402
    AO_POISSON_BLEND,
    AO_POISSON_RHO0_CONTROL,
    CURRENT_AO,
    CURRENT_ML_BLEND,
    ML_POISSON_ENSEMBLE,
    run_prediction_ensemble_walk_forward_backtest,
)


DOMESTIC_MATCHES = (
    ROOT / "data" / "domestic_league_matches_2013_2026" / "domestic_matches.csv"
)
DOMESTIC_BRIDGE = (
    ROOT
    / "data"
    / "domestic_league_matches_2013_2026"
    / "domestic_team_bridge.csv"
)
FEATURE_STORE = (
    ROOT / "output" / "ml_1x2_backtest_2018_2026" / "pre_match_feature_store.csv"
)
ML_SELECTIONS = (
    ROOT / "output" / "ml_1x2_backtest_2018_2026" / "fold_selections.csv"
)
DOMESTIC_ROOT = ROOT / "output" / "domestic_poisson_backtest_2018_2026"
DOMESTIC_PREDICTIONS = DOMESTIC_ROOT / "unseen_predictions.csv"
DOMESTIC_SELECTIONS = DOMESTIC_ROOT / "fold_domestic_selections.csv"
TRANSFER_SELECTIONS = DOMESTIC_ROOT / "fold_transfer_parameters.csv"
PRODUCTION_CONTRACT = ROOT / "contracts" / "ao_european_elo_v2_production.json"
OUTPUT_ROOT = ROOT / "output" / "final_prediction_ensemble_backtest_2018_2026"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the leakage-safe final ML/Poisson probability ensemble backtest"
    )
    parser.add_argument("--feature-store", type=Path, default=FEATURE_STORE)
    parser.add_argument("--domestic-matches", type=Path, default=DOMESTIC_MATCHES)
    parser.add_argument("--domestic-bridge", type=Path, default=DOMESTIC_BRIDGE)
    parser.add_argument(
        "--domestic-predictions", type=Path, default=DOMESTIC_PREDICTIONS
    )
    parser.add_argument("--ml-selections", type=Path, default=ML_SELECTIONS)
    parser.add_argument(
        "--domestic-selections", type=Path, default=DOMESTIC_SELECTIONS
    )
    parser.add_argument(
        "--transfer-selections", type=Path, default=TRANSFER_SELECTIONS
    )
    parser.add_argument("--production-contract", type=Path, default=PRODUCTION_CONTRACT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    parser.add_argument(
        "--prospective-poisson-weight",
        type=float,
        default=None,
        help=(
            "Pin the served blend weight instead of re-selecting one from the "
            "surface. Rebuilding this backtest after an unrelated activation "
            "would otherwise silently re-open a parameter the 2026/27 holdout "
            "protocol freezes; pinning it to the production contract value "
            "keeps the rebuild a propagation. Default preserves the automatic "
            "choice."
        ),
    )
    args = parser.parse_args()

    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    feature_store = pd.read_csv(args.feature_store.resolve(), low_memory=False)
    feature_store["kickoff_utc"] = pd.to_datetime(
        feature_store["kickoff_utc"], utc=True
    )
    domestic_matches = pd.read_csv(
        args.domestic_matches.resolve(), low_memory=False
    )
    bridge = pd.read_csv(args.domestic_bridge.resolve(), low_memory=False)
    predictions = pd.read_csv(
        args.domestic_predictions.resolve(), low_memory=False
    )
    ml_selections = pd.read_csv(args.ml_selections.resolve(), low_memory=False)
    domestic_selections = pd.read_csv(
        args.domestic_selections.resolve(), low_memory=False
    )
    transfer_selections = pd.read_csv(
        args.transfer_selections.resolve(), low_memory=False
    )

    contract_path = args.production_contract.resolve()
    contract_before = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    print("Running six-fold nested ML/Poisson ensemble backtest", flush=True)
    result = run_prediction_ensemble_walk_forward_backtest(
        feature_store,
        domestic_matches,
        bridge,
        predictions,
        ml_selections,
        domestic_selections,
        transfer_selections,
        bootstrap_samples=args.bootstrap_samples,
        pinned_poisson_weight=args.prospective_poisson_weight,
    )
    contract_after = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    if contract_after != contract_before:
        raise ValueError("Production contract changed during prediction-only backtest")

    result.inner_weight_surface.to_csv(
        output / "inner_weight_surface.csv", index=False
    )
    result.fold_selections.to_csv(output / "fold_selections.csv", index=False)
    result.fold_results.to_csv(output / "fold_results.csv", index=False)
    result.unseen_predictions.to_csv(
        output / "unseen_predictions.csv", index=False
    )
    result.model_comparison.to_csv(output / "model_comparison.csv", index=False)
    result.calibration_summary.to_csv(
        output / "calibration_summary.csv", index=False
    )
    result.competition_coverage_summary.to_csv(
        output / "competition_coverage_summary.csv", index=False
    )
    result.dependency_uncertainty.to_csv(
        output / "dependency_uncertainty.csv", index=False
    )
    result.prospective_weight_surface.to_csv(
        output / "prospective_weight_surface.csv", index=False
    )

    decision = dict(result.selected_candidate)
    decision.update(
        {
            "model_version": "ao-ml-poisson-ensemble-v1-shadow",
            "production_contract_sha256": contract_after,
            "python_version": platform.python_version(),
            "scipy_version": scipy.__version__,
            "scikit_learn_version": sklearn.__version__,
            "development_matches": 6340,
            "unseen_matches": 4884,
            "domestic_matches": int(len(domestic_matches)),
            "selection_method": "nested inner-season log-probability blend",
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
        f"Current ML delta Brier={decision['delta_brier_vs_current_ml']:+.8f}; "
        f"log-loss={decision['delta_log_loss_vs_current_ml']:+.8f}",
        flush=True,
    )
    print(f"Output: {output}", flush=True)


def build_report(result, decision: dict[str, object]) -> str:
    comparison = result.model_comparison[
        result.model_comparison["model"].isin(
            (
                CURRENT_AO,
                CURRENT_ML_BLEND,
                AO_POISSON_BLEND,
                AO_POISSON_RHO0_CONTROL,
                ML_POISSON_ENSEMBLE,
            )
        )
    ]
    folds = result.fold_results[
        result.fold_results["model"].eq(ML_POISSON_ENSEMBLE)
    ][
        [
            "fold",
            "test_season",
            "matches",
            "brier_1x2",
            "delta_brier_vs_current_ml",
            "log_loss_1x2",
            "delta_log_loss_vs_current_ml",
            "accuracy_1x2",
        ]
    ]
    selections = result.fold_selections[
        [
            "fold",
            "test_season",
            "inner_validation_season",
            "poisson_source",
            "ml_weight",
            "poisson_weight",
        ]
    ]
    segments = result.competition_coverage_summary[
        result.competition_coverage_summary["model"].eq(ML_POISSON_ENSEMBLE)
        & result.competition_coverage_summary["segment_type"].isin(
            ("competition", "coverage")
        )
    ][
        [
            "segment_type",
            "segment_value",
            "matches",
            "brier_1x2",
            "delta_brier_vs_current_ml",
            "log_loss_1x2",
            "delta_log_loss_vs_current_ml",
        ]
    ]
    prospective = decision["prospective_2026_27_selection"]
    return f"""# AO ML + Domestic Poisson Final Ensemble Backtesti

## Model kararı

**{decision['decision']}**. Bu katman yalnız maç öncesi 1X2 olasılıklarını
birleştirir. AO First Elo, AO Live Elo, Power Delta, gol farkı/xG güncellemesi
ve production contract değiştirilmemiştir.

## Veri ve yöntem

- Geliştirme penceresi: `2018/19-2025/26`, toplam 6.340 Avrupa maçı.
- Unseen değerlendirme: `2020/21-2025/26`, altı fold ve 4.884 maç.
- Her foldun model kaynağı ve ağırlığı yalnız bir önceki inner-validation
  sezonunda seçildi; test sezonu seçime girmedi.
- Karışım log-olasılık uzayında yapıldı. `w=0` mevcut ML, `w=1` Poisson'dur.
- Poisson kaynağında `rho=0` ve inner-selected Dixon-Coles ayrı adaylardır.
- `2026/27` untouched prospective holdout olarak korunmuştur.

## Ana karşılaştırma

{_markdown_table(comparison)}

## Fold seçimleri

{_markdown_table(selections)}

## Unseen fold sonuçları

{_markdown_table(folds)}

## Turnuva ve coverage segmentleri

{_markdown_table(segments)}

## Kalibrasyon

{_markdown_table(result.calibration_summary)}

## 2026/27 için dondurulabilir seçim

- ML ağırlığı: `{prospective['ml_weight']}`
- Poisson ağırlığı: `{prospective['poisson_weight']}`
- Poisson kaynağı: `{prospective['poisson_source']}`

Bu tam-geliştirme seçimi 2026/27 verisini görmemiştir; yine de production
aktivasyonu ayrı onay ve prospective doğrulama gerektirir.

## Karar kapıları

```json
{json.dumps(decision['gates'], indent=2, ensure_ascii=False)}
```
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
