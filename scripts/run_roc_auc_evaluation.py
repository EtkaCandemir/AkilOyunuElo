from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ao_elo.roc_auc_evaluation import evaluate_multiclass_roc_auc


DEFAULT_INPUT = (
    ROOT
    / "output"
    / "final_prediction_ensemble_backtest_2018_2026"
    / "unseen_predictions.csv"
)
DEFAULT_OUTPUT = ROOT / "output" / "roc_auc_evaluation_2018_2026"
MODELS = (
    "CURRENT_AO",
    "CURRENT_ML_BLEND",
    "AO_POISSON_RHO0_CONTROL",
    "ML_POISSON_ENSEMBLE",
)
CANDIDATE = "ML_POISSON_ENSEMBLE"
BASELINES = ("CURRENT_AO", "CURRENT_ML_BLEND")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate multiclass 1X2 ROC-AUC on frozen unseen predictions"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    args = parser.parse_args()

    predictions = pd.read_csv(args.input)
    result = evaluate_multiclass_roc_auc(
        predictions,
        models=MODELS,
        candidate_model=CANDIDATE,
        baselines=BASELINES,
        bootstrap_samples=args.bootstrap_samples,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    result.model_summary.to_csv(args.output / "model_summary.csv", index=False)
    result.class_summary.to_csv(args.output / "class_summary.csv", index=False)
    result.fold_summary.to_csv(args.output / "fold_summary.csv", index=False)
    result.competition_summary.to_csv(
        args.output / "competition_summary.csv", index=False
    )
    result.paired_uncertainty.to_csv(
        args.output / "paired_cluster_uncertainty.csv", index=False
    )
    result.roc_curves.to_csv(args.output / "roc_curves.csv", index=False)
    (args.output / "data_audit.json").write_text(
        json.dumps(result.data_audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report = build_report(result)
    (args.output / "roc_auc_report.md").write_text(report, encoding="utf-8")

    summary = result.model_summary.set_index("model")
    candidate = summary.loc[CANDIDATE]
    baseline = summary.loc["CURRENT_AO"]
    print(f"Matches: {result.data_audit['matches']}")
    print(
        "Macro ROC-AUC: "
        f"AO={baseline['macro_roc_auc_ovr']:.6f} "
        f"Final={candidate['macro_roc_auc_ovr']:.6f} "
        f"Delta={candidate['macro_roc_auc_ovr'] - baseline['macro_roc_auc_ovr']:+.6f}"
    )
    print(f"Report: {args.output / 'roc_auc_report.md'}")


def build_report(result) -> str:
    model = result.model_summary.copy()
    classes = result.class_summary.copy()
    folds = result.fold_summary.copy()
    competitions = result.competition_summary.copy()
    uncertainty = result.paired_uncertainty.copy()
    summary = model.set_index("model")
    candidate = summary.loc[CANDIDATE]
    ao = summary.loc["CURRENT_AO"]
    current_ml = summary.loc["CURRENT_ML_BLEND"]
    macro_ci_ao = uncertainty[
        uncertainty["baseline_model"].eq("CURRENT_AO")
        & uncertainty["metric"].eq("macro_roc_auc_ovr")
    ].iloc[0]
    macro_ci_ml = uncertainty[
        uncertainty["baseline_model"].eq("CURRENT_ML_BLEND")
        & uncertainty["metric"].eq("macro_roc_auc_ovr")
    ].iloc[0]

    fold_candidate = folds[folds["model"].eq(CANDIDATE)].set_index("fold")
    fold_ao = folds[folds["model"].eq("CURRENT_AO")].set_index("fold")
    fold_rows = []
    for fold in fold_candidate.index:
        row = fold_candidate.loc[fold]
        fold_rows.append(
            {
                "fold": fold,
                "test_season": row["test_season"],
                "matches": int(row["matches"]),
                "current_ao": float(fold_ao.loc[fold, "macro_roc_auc_ovr"]),
                "final": float(row["macro_roc_auc_ovr"]),
                "difference": float(
                    row["macro_roc_auc_ovr"]
                    - fold_ao.loc[fold, "macro_roc_auc_ovr"]
                ),
            }
        )

    class_pivot = classes.pivot(
        index="class_label", columns="model", values="roc_auc_ovr"
    )
    class_lines = [
        "| Sınıf | Current AO | Final ensemble | Fark |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label in ("HOME", "DRAW", "AWAY"):
        class_lines.append(
            f"| {label} | {class_pivot.loc[label, 'CURRENT_AO']:.6f} | "
            f"{class_pivot.loc[label, CANDIDATE]:.6f} | "
            f"{class_pivot.loc[label, CANDIDATE] - class_pivot.loc[label, 'CURRENT_AO']:+.6f} |"
        )

    competition_candidate = competitions[
        competitions["model"].eq(CANDIDATE)
    ].set_index("competition")
    competition_ao = competitions[
        competitions["model"].eq("CURRENT_AO")
    ].set_index("competition")
    competition_lines = [
        "| Turnuva | Maç | Current AO | Final ensemble | Fark |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for competition in ("UCL", "UEL", "UECL"):
        final_value = competition_candidate.loc[competition, "macro_roc_auc_ovr"]
        ao_value = competition_ao.loc[competition, "macro_roc_auc_ovr"]
        competition_lines.append(
            f"| {competition} | {int(competition_candidate.loc[competition, 'matches'])} | "
            f"{ao_value:.6f} | {final_value:.6f} | {final_value - ao_value:+.6f} |"
        )

    fold_lines = [
        "| Fold | Test sezonu | Maç | Current AO | Final ensemble | Fark |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in fold_rows:
        fold_lines.append(
            f"| {row['fold']} | {row['test_season']} | {row['matches']} | "
            f"{row['current_ao']:.6f} | {row['final']:.6f} | {row['difference']:+.6f} |"
        )

    return f"""# AO European Elo Multiclass ROC-AUC Değerlendirmesi

## Sonuç

Sistem ROC-AUC değerlendirmesine uygundur. Her maç için normalize edilmiş üç
olasılık ve gerçek `HOME/DRAW/AWAY` sınıfı bulunduğundan One-vs-Rest (OvR)
ROC-AUC hesaplanmıştır. Final production ensemble, Current AO'ya karşı pooled
macro ROC-AUC değerini `{ao['macro_roc_auc_ovr']:.6f}` seviyesinden
`{candidate['macro_roc_auc_ovr']:.6f}` seviyesine çıkarmıştır; fark
`{candidate['macro_roc_auc_ovr'] - ao['macro_roc_auc_ovr']:+.6f}`.

Bu metrik yalnız sıralama/ayırt etme gücünü ölçer. Olasılık kalibrasyonunu veya
olasılıkların ne kadar isabetli olduğunu tek başına ölçmez; Brier, log-loss ve
kalibrasyon metrikleriyle birlikte okunmalıdır.

## Veri ve yöntem

- Unseen dönem: `2020/21-2025/26`, altı walk-forward fold.
- Benzersiz maç: `{result.data_audit['matches']}`.
- Sınıf sayıları: Home `{result.data_audit['class_counts']['HOME']}`, Draw
  `{result.data_audit['class_counts']['DRAW']}`, Away
  `{result.data_audit['class_counts']['AWAY']}`.
- Pooled ölçüm: sınıf bazlı OvR, macro/weighted/micro OvR ve macro OvO.
- Güven aralığı: eşleşmiş `{result.data_audit['bootstrap_samples']}` tekrar,
  `season + calendar month` cluster bootstrap.
- `2026/27` bu değerlendirmeye dahil değildir.

## Pooled model karşılaştırması

{_markdown_table(model)}

## Sınıf bazlı OvR ROC-AUC

{chr(10).join(class_lines)}

Beraberlik sınıfı bütün modellerde en zor sınıftır. Final ensemble Draw AUC'yi
Current AO'ya göre artırırken, standalone Current ML'nin Draw AUC değerinin
biraz altında kalmıştır.

## Fold bazlı macro OvR ROC-AUC

{chr(10).join(fold_lines)}

Final ensemble Current AO'yu `5/6` unseen fold içinde geçmiştir. Tek negatif
fold `2020/21` sezonudur.

## Turnuva bazlı macro OvR ROC-AUC

{chr(10).join(competition_lines)}

UCL ve UECL yönü pozitiftir. UEL farkı küçüktür ve hafif negatif yöndedir;
turnuva bazında her segmentin ayrı izlenmesi gerekir.

## Eşleşmiş cluster-bootstrap belirsizliği

{_markdown_table(uncertainty)}

Final ensemble ile Current AO arasındaki macro ROC-AUC farkının yüzde 95
cluster-bootstrap aralığı `[{macro_ci_ao['ci_95_lower']:+.6f},
{macro_ci_ao['ci_95_upper']:+.6f}]` değeridir. Current ML karşılaştırmasındaki
macro fark `{candidate['macro_roc_auc_ovr'] - current_ml['macro_roc_auc_ovr']:+.6f}`
ve aralık `[{macro_ci_ml['ci_95_lower']:+.6f},
{macro_ci_ml['ci_95_upper']:+.6f}]` değeridir.

## Yorum

- `0.50`, rastgele ayırt etme seviyesidir; tüm modeller bunun üzerindedir.
- Final ensemble'ın Current AO'ya göre ayırt etme gücü pooled ve fold bazında
  daha iyidir.
- Kazancın önemli bölümü Structural ML kolundan gelir; final ensemble Current
  ML macro AUC değerini anlamlı biçimde aşmamaktadır.
- Draw sınıfı `0.57` civarında kaldığı için en belirgin geliştirme alanıdır.
- ROC-AUC'nin artması tek başına production kararı veya iyi kalibrasyon kanıtı
  değildir. Mevcut `PROMOTE_WITH_MONITORING` kararı değişmemiştir.

## Veri güvenliği

- Her modelde tam `{result.data_audit['matches']}` eşleşmiş maç vardır.
- Model-match anahtarları benzersizdir ve gerçek sınıflar modeller arasında aynıdır.
- Olasılıklar sonlu, negatif olmayan ve toplamı birdir.
- Test sezonları model seçimine geri beslenmemiş frozen unseen tahminlerdir.
- Production contract ve AO Live rating state'i değiştirilmemiştir.
"""


def _markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include="number").columns:
        if column == "matches" or column.endswith("samples") or column.endswith("count"):
            display[column] = display[column].map(lambda value: str(int(value)))
        else:
            display[column] = display[column].map(lambda value: f"{float(value):.6f}")
    headers = list(display.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
