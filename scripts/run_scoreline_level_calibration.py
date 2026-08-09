from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ao_elo.scoreline_calibration import (  # noqa: E402
    GoalLevelBacktestResult,
    goal_level_candidates,
    run_goal_level_walk_forward_backtest,
)
from scripts.run_dynamic_core_calibration import expanding_folds  # noqa: E402
from scripts.run_scoreline_backtest import (  # noqa: E402
    EVENTS_PATH,
    MODEL_CONTRACT_PATH,
    PRODUCTION_CONTRACT_PATH,
    STATIC_DATA_ROOT,
    build_production_snapshots,
    static_config_from_contract,
    validate_production_contract,
)
from scripts.run_v2_evaluation_upgrade import read_events  # noqa: E402


OUTPUT_ROOT = ROOT / "output" / "scoreline_level_calibration_2018_2026"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest competition and prior-season goal-level calibration"
    )
    parser.add_argument("--static-data-root", type=Path, default=STATIC_DATA_ROOT)
    parser.add_argument("--events-path", type=Path, default=EVENTS_PATH)
    parser.add_argument("--model-contract", type=Path, default=MODEL_CONTRACT_PATH)
    parser.add_argument("--production-contract", type=Path, default=PRODUCTION_CONTRACT_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    args = parser.parse_args()

    model_contract = json.loads(args.model_contract.resolve().read_text(encoding="utf-8"))
    production = json.loads(
        args.production_contract.resolve().read_text(encoding="utf-8")
    )
    validate_production_contract(production)
    static_config = static_config_from_contract(model_contract)
    events = read_events(args.events_path.resolve())
    snapshots, identity_preserved, seasons = build_production_snapshots(
        args.static_data_root.resolve(),
        args.events_path.resolve(),
        static_config,
        production,
        events,
    )
    rating_snapshot = snapshots[["home_live_pre", "away_live_pre"]].copy(deep=True)
    folds = expanding_folds(seasons)
    if len(folds) != 6:
        raise ValueError(f"Expected six outer folds, found {len(folds)}")
    core = production["dynamic_core"]
    result = run_goal_level_walk_forward_backtest(
        snapshots,
        folds,
        elo_scale=float(core["elo_scale"]),
        home_advantage=float(core["home_advantage"]),
        bootstrap_samples=args.bootstrap_samples,
        elo_identity_preserved=identity_preserved,
    )
    if not snapshots[["home_live_pre", "away_live_pre"]].equals(rating_snapshot):
        raise ValueError("Goal-level calibration mutated production Elo snapshots")
    output_root = args.output_root.resolve()
    write_outputs(output_root, result, production, len(events))

    print("AO scoreline competition/season goal-level calibration")
    print(f"Candidates: {len(goal_level_candidates())}")
    print(f"Matches: {len(events)}")
    print(f"Unseen matches: {len(result.unseen_predictions)}")
    print(f"Selected full candidate: {result.selected_config.key}")
    print(f"Decision: {result.decision}")
    print(f"Report: {output_root / 'backtest_report.md'}")


def write_outputs(
    output_root: Path,
    result: GoalLevelBacktestResult,
    production: dict[str, object],
    total_matches: int,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    result.fold_selections.to_csv(output_root / "fold_selections.csv", index=False)
    result.fold_results.to_csv(output_root / "fold_results.csv", index=False)
    result.inner_candidate_metrics.to_csv(
        output_root / "inner_candidate_metrics.csv", index=False
    )
    result.unseen_predictions.to_csv(
        output_root / "unseen_predictions.csv", index=False
    )
    result.competition_summary.to_csv(
        output_root / "competition_summary.csv", index=False
    )
    result.season_summary.to_csv(output_root / "season_summary.csv", index=False)
    result.dependency_uncertainty.to_csv(
        output_root / "dependency_uncertainty.csv", index=False
    )
    fitted = result.fitted_full_calibration
    manifest = {
        "status": result.decision,
        "production_activated": False,
        "candidate_count": len(goal_level_candidates()),
        "selected_config": asdict(result.selected_config),
        "full_scoreline_config": asdict(result.full_scoreline_config),
        "full_competition_log_offsets": fitted.competition_log_offsets,
        "full_season_log_offset": fitted.season_log_offset,
        "full_season_source": list(fitted.season_source),
        "guardrails": result.guardrails,
        "development_matches": total_matches,
        "unseen_matches": len(result.unseen_predictions),
        "development_data_through": "2025/26",
        "untouched_holdout": "2026/27",
        "source_production_revision": production.get("production_revision"),
        "rating_feedback_enabled": False,
    }
    (output_root / "selected_level_model.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (output_root / "backtest_report.md").write_text(
        build_report(result, manifest), encoding="utf-8"
    )


def build_report(
    result: GoalLevelBacktestResult,
    manifest: dict[str, object],
) -> str:
    selected = result.fold_results.loc[
        result.fold_results["model"].eq("selected_level")
    ].set_index("fold")
    poisson = result.fold_results.loc[
        result.fold_results["model"].eq("elo_poisson")
    ].set_index("fold")
    ao = result.fold_results.loc[
        result.fold_results["model"].eq("current_ao_1x2")
    ].set_index("fold")
    lines = [
        "# Turnuva ve Geçmiş-Sezon Gol Seviyesi Kalibrasyonu",
        "",
        "## Model kararı",
        "",
        f"**{result.decision}**. Bu araştırma production Elo veya 1X2 sözleşmesini otomatik değiştirmez.",
        "",
        "## Sözleşme",
        "",
        "```text",
        "lambda_home_cal = lambda_home x exp(c_competition + c_prior_season)",
        "lambda_away_cal = lambda_away x exp(c_competition + c_prior_season)",
        "```",
        "",
        "Turnuva düzeltmesi yalnız geçmiş UCL/UEL/UECL maçlarından; sezon düzeltmesi yalnız test sezonundan önce tamamlanmış son 1/2/3 sezondan hesaplanmıştır. Her outer foldun aday gücü ayrı inner walk-forward ile seçilmiştir.",
        "",
        "## Tam veri adayı",
        "",
        f"- Aday: `{result.selected_config.key}`",
        f"- Competition strength: {result.selected_config.competition_strength:g}",
        f"- Prior-season strength: {result.selected_config.season_strength:g}",
        f"- Prior-season lookback: {result.selected_config.season_lookback}",
        f"- UCL log offset: {result.fitted_full_calibration.competition_log_offsets['UCL']:+.8f}",
        f"- UEL log offset: {result.fitted_full_calibration.competition_log_offsets['UEL']:+.8f}",
        f"- UECL log offset: {result.fitted_full_calibration.competition_log_offsets['UECL']:+.8f}",
        f"- Prior-season log offset: {result.fitted_full_calibration.season_log_offset:+.8f}",
        "",
        "## Fold sonuçları",
        "",
        "| Fold | Sezon | Aday | Score NLL farkı | Brier farkı vs AO | Log-loss farkı vs AO | Gol bias |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    selections = result.fold_selections.set_index("fold")
    for fold in selected.index:
        row = selected.loc[fold]
        lines.append(
            f"| {fold} | {row.test_season} | `{selections.loc[fold, 'candidate']}` | "
            f"{row.score_nll - poisson.loc[fold, 'score_nll']:+.6f} | "
            f"{row.brier_1x2 - ao.loc[fold, 'brier_1x2']:+.6f} | "
            f"{row.log_loss_1x2 - ao.loc[fold, 'log_loss_1x2']:+.6f} | "
            f"{row.total_goals_bias:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Turnuva sonuçları",
            "",
            "| Turnuva | Maç | Score NLL farkı | Brier farkı vs AO | Log-loss farkı vs AO | Gol bias |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for value in ("ALL", "UCL", "UEL", "UECL"):
        row = result.competition_summary.loc[
            result.competition_summary["value"].eq(value)
        ].iloc[0]
        lines.append(
            f"| {value} | {int(row.matches)} | "
            f"{row.score_nll_difference_vs_elo_poisson:+.6f} | "
            f"{row.brier_difference_vs_current_ao:+.6f} | "
            f"{row.log_loss_difference_vs_current_ao:+.6f} | "
            f"{row.total_goals_bias:+.6f} |"
        )
    lines.extend(["", "## Terfi korumaları", ""])
    for key, value in result.guardrails.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Yorum",
            "",
            "Katman yalnız gol beklentisini kalibre eder ve takım ratinglerine geri beslenmez. Terfi için exact-score NLL, mevcut AO 1X2, turnuva güvenliği ve gol bias kapılarının tamamının birlikte geçilmesi gerekir.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
