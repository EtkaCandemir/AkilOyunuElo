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

from ao_elo.config import AOEuropeanEloConfig  # noqa: E402
from ao_elo.evaluation import schedule_adjusted_team_performance  # noqa: E402
from ao_elo.robustness import baseline_competition_k, baseline_goal_margin  # noqa: E402
from ao_elo.scoreline import (  # noqa: E402
    DEFAULT_RHO_GRID,
    ScorelineBacktestResult,
    run_scoreline_walk_forward_backtest,
)
from scripts.run_dynamic_core_calibration import (  # noqa: E402
    DynamicCoreConfig,
    expanding_folds,
)
from scripts.run_final_robustness import (  # noqa: E402
    ControlledGoalConfig,
    evaluate_sequence,
)
from scripts.run_v2_achievement_reserve_calibration import (  # noqa: E402
    baseline_config as baseline_reserve,
    load_reserve_data,
)
from scripts.run_v2_evaluation_upgrade import DrawModelConfig, read_events  # noqa: E402


STATIC_DATA_ROOT = ROOT / "data" / "backtest_stage_b_2018_2026"
EVENTS_PATH = ROOT / "data" / "external_elo_benchmark_2018_2026" / "exact_date_events.csv"
MODEL_CONTRACT_PATH = ROOT / "contracts" / "ao_european_elo_v2.json"
PRODUCTION_CONTRACT_PATH = ROOT / "contracts" / "ao_european_elo_v2_production.json"
OUTPUT_ROOT = ROOT / "output" / "scoreline_backtest_2018_2026"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the leakage-safe AO Live Elo Poisson/Dixon-Coles scoreline backtest"
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
    result = run_scoreline_walk_forward_backtest(
        snapshots,
        folds,
        elo_scale=float(core["elo_scale"]),
        home_advantage=float(core["home_advantage"]),
        rho_grid=DEFAULT_RHO_GRID,
        elo_identity_preserved=identity_preserved,
        bootstrap_samples=args.bootstrap_samples,
    )
    if not snapshots[["home_live_pre", "away_live_pre"]].equals(rating_snapshot):
        raise ValueError("Scoreline backtest mutated production Elo snapshots")
    output_root = args.output_root.resolve()
    write_outputs(output_root, result, production, seasons, len(events))

    print("AO Live Elo + Dixon-Coles scoreline backtest")
    print(f"Matches: {len(events)}")
    print(f"Unseen matches: {len(result.unseen_predictions)}")
    print(f"Folds: {len(result.fold_parameters)}")
    print(
        "Full-data parameters: "
        f"mu={result.selected_config.mu:.6f}, "
        f"beta={result.selected_config.elo_slope:.6f}, "
        f"rho={result.selected_config.rho:.2f}"
    )
    print(f"Decision: {result.decision}")
    print(f"Elo identity preserved: {identity_preserved}")
    print(f"Report: {output_root / 'backtest_report.md'}")


def validate_production_contract(production: dict[str, object]) -> None:
    required = {
        "dynamic_core",
        "active_power_carry",
        "one_x_two_probability",
        "goal_margin",
        "progression_bonus",
        "achievement_reserve",
        "competition_k",
    }
    missing = sorted(required - set(production))
    if missing:
        raise ValueError(f"Production contract missing keys: {missing}")
    if float(production["active_power_carry"]) != 0.0:
        raise ValueError("Scoreline backtest requires production carry=0")
    goal = production["goal_margin"]
    if not goal.get("active") or goal.get("family") != "CONTROLLED_FAVORITE_DAMPED_LOG":
        raise ValueError("Scoreline backtest requires active controlled goal difference")
    if production["progression_bonus"].get("active"):
        raise ValueError("Scoreline baseline requires progression bonus disabled")
    if production["achievement_reserve"].get("active"):
        raise ValueError("Scoreline baseline requires achievement reserve disabled")
    if production["competition_k"].get("active"):
        raise ValueError("Scoreline baseline requires competition K disabled")


def static_config_from_contract(contract: dict[str, object]) -> AOEuropeanEloConfig:
    static = dict(contract["static"])
    static.pop("tail_decision", None)
    static["model_version"] = str(contract["model_version"])
    config = AOEuropeanEloConfig(**static)
    config.validate()
    return config


def build_production_snapshots(
    static_root: Path,
    events_path: Path,
    static_config: AOEuropeanEloConfig,
    production: dict[str, object],
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, bool, tuple[str, ...]]:
    datasets, _ = load_reserve_data(static_root, events_path, static_config)
    seasons = tuple(data.season for data in datasets)
    if "2026/27" in seasons:
        raise ValueError("2026/27 untouched holdout cannot enter scoreline fitting")
    core = DynamicCoreConfig(**production["dynamic_core"])
    one_x_two = production["one_x_two_probability"]
    draw = DrawModelConfig(
        float(one_x_two["draw_at_even"]),
        float(one_x_two["draw_shape"]),
    )
    goal = production["goal_margin"]
    controlled_goal = ControlledGoalConfig(
        float(goal["alpha"]),
        float(goal["tau"]),
        int(goal["goal_difference_cap"]),
    )
    target = schedule_adjusted_team_performance(events)
    common = dict(
        datasets=datasets,
        core=core,
        margin=baseline_goal_margin(),
        competition_k=baseline_competition_k(),
        reserve=baseline_reserve(),
        draw_mapping={"ALL": draw},
        target=target,
        evaluation_seasons=set(seasons),
        ranking_target_seasons=set(seasons),
        controlled_goal_config=controlled_goal,
    )
    evaluated = evaluate_sequence(**common, return_predictions=True)
    reference = evaluate_sequence(**common, return_predictions=False)
    identity_columns = [
        "season",
        "team_id",
        "initial_rating",
        "end_power_rating",
        "end_reserve",
        "end_live_rating",
    ]
    identity_preserved = evaluated.end_ratings[identity_columns].equals(
        reference.end_ratings[identity_columns]
    )
    if not identity_preserved:
        raise ValueError("Production replay is not deterministic at the Elo state level")

    predictions = evaluated.predictions.rename(
        columns={
            "home_live_rating": "home_live_pre",
            "away_live_rating": "away_live_pre",
            "home_probability": "ao_home_probability",
            "draw_probability": "ao_draw_probability",
            "away_probability": "ao_away_probability",
        }
    )
    metadata_columns = [
        "match_id",
        "kickoff_utc",
        "tie_id",
        "round",
        "is_knockout",
        "is_neutral",
        "decided_on_penalties",
        "home_goals",
        "away_goals",
    ]
    snapshots = predictions.merge(
        events[metadata_columns],
        on="match_id",
        validate="one_to_one",
        suffixes=("", "_event"),
    )
    if "is_neutral_event" in snapshots:
        if not snapshots["is_neutral"].eq(snapshots["is_neutral_event"]).all():
            raise ValueError("Snapshot/event neutral-site metadata mismatch")
        snapshots = snapshots.drop(columns=["is_neutral_event"])
    return snapshots, identity_preserved, seasons


def write_outputs(
    output_root: Path,
    result: ScorelineBacktestResult,
    production: dict[str, object],
    seasons: tuple[str, ...],
    total_matches: int,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    result.fold_parameters.to_csv(output_root / "fold_parameters.csv", index=False)
    result.fold_results.to_csv(output_root / "fold_results.csv", index=False)
    result.unseen_predictions.to_csv(output_root / "unseen_predictions.csv", index=False)
    result.competition_summary.to_csv(output_root / "competition_summary.csv", index=False)
    result.calibration_summary.to_csv(output_root / "calibration_summary.csv", index=False)
    result.segment_summary.to_csv(output_root / "segment_summary.csv", index=False)
    result.dependency_uncertainty.to_csv(
        output_root / "dependency_uncertainty.csv", index=False
    )
    manifest = {
        "status": result.decision,
        "production_activated": False,
        "scoreline_model": asdict(result.selected_config),
        "guardrails": result.guardrails,
        "rho_grid": list(DEFAULT_RHO_GRID),
        "development_seasons": list(seasons),
        "development_matches": total_matches,
        "outer_folds": len(result.fold_parameters),
        "unseen_matches": len(result.unseen_predictions),
        "untouched_holdout": "2026/27",
        "source_production_revision": production.get("production_revision"),
        "rating_feedback_enabled": False,
    }
    (output_root / "selected_scoreline_model.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (output_root / "backtest_report.md").write_text(
        build_report(result, manifest), encoding="utf-8"
    )


def build_report(
    result: ScorelineBacktestResult,
    manifest: dict[str, object],
) -> str:
    selected = result.fold_results.loc[
        result.fold_results["model"].eq("elo_dixon_coles")
    ]
    current = result.fold_results.loc[
        result.fold_results["model"].eq("current_ao_1x2")
    ]
    intercept = result.fold_results.loc[
        result.fold_results["model"].eq("intercept_poisson")
    ]
    pooled = result.competition_summary
    lines = [
        "# AO Live Elo + Dixon-Coles Skor Tahmin Backtesti",
        "",
        "## Model kararı",
        "",
        f"**{result.decision}**. Bu paket yalnız öneri üretir; production sözleşmesi ve Elo güncelleme yolu değiştirilmemiştir.",
        "",
        "## Veri ve model",
        "",
        f"- Geliştirme maçları: {manifest['development_matches']}",
        f"- Out-of-sample maçlar: {manifest['unseen_matches']}",
        "- Sezonlar: 2018/19-2025/26; altı expanding outer fold",
        "- Untouched holdout dışarıda tutuldu: 2026/27",
        "- Hedef: penaltı atışları hariç 90/120 dakika saha skoru",
        "- AO First Elo, AO Live Elo ve kontrollü gol farkı güncellemeleri değişmedi",
        "- Gerçek OOS lambda değerlerinde 1e-10 tail garantisini korumak için adaptif skor matrisi üst desteği 20'den 25'e çıkarıldı; olasılık toleransı gevşetilmedi",
        "",
        "## Tam veri önerisi",
        "",
        f"- mu: {result.selected_config.mu:.8f}",
        f"- beta: {result.selected_config.elo_slope:.8f}",
        f"- rho: {result.selected_config.rho:.2f}",
        f"- Dixon-Coles OOS NLL farkı (DC - Elo-Poisson): {result.guardrails['dixon_coles_delta_vs_elo_poisson']:+.8f}",
        "",
        "## Fold karşılaştırması",
        "",
        "| Fold | Sezon | Skor NLL | Intercept NLL | 1X2 Brier | AO Brier | 1X2 log-loss | AO log-loss |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for fold in sorted(selected["fold"].unique()):
        candidate = selected.loc[selected["fold"].eq(fold)].iloc[0]
        base_score = intercept.loc[intercept["fold"].eq(fold)].iloc[0]
        base_ao = current.loc[current["fold"].eq(fold)].iloc[0]
        lines.append(
            f"| {fold} | {candidate.test_season} | {candidate.score_nll:.6f} | "
            f"{base_score.score_nll:.6f} | {candidate.brier_1x2:.6f} | "
            f"{base_ao.brier_1x2:.6f} | {candidate.log_loss_1x2:.6f} | "
            f"{base_ao.log_loss_1x2:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Turnuva sonuçları",
            "",
            "| Turnuva | Maç | Aday Brier | AO Brier | Aday log-loss | AO log-loss | Gol yanlılığı |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    calibration = result.calibration_summary.loc[
        result.calibration_summary["calibration"].eq("competition")
    ].set_index("segment")
    for competition in ("ALL", "UCL", "UEL", "UECL"):
        candidate = pooled.loc[
            pooled["model"].eq("elo_dixon_coles") & pooled["value"].eq(competition)
        ].iloc[0]
        ao = pooled.loc[
            pooled["model"].eq("current_ao_1x2") & pooled["value"].eq(competition)
        ].iloc[0]
        lines.append(
            f"| {competition} | {int(candidate.matches)} | {candidate.brier_1x2:.6f} | "
            f"{ao.brier_1x2:.6f} | {candidate.log_loss_1x2:.6f} | "
            f"{ao.log_loss_1x2:.6f} | {calibration.loc[competition, 'total_goals_bias']:.6f} |"
        )
    lines.extend(["", "## Terfi korumaları", ""])
    for key, value in result.guardrails.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Yorum",
            "",
            "Dixon-Coles yalnız bir çıktı katmanıdır. Maç öncesi AO Live Elo değerini skor, 1X2, O/U 2.5 ve BTTS olasılıklarına çevirir. Golleri, olasılıkları veya öğrenilen parametreleri takım ratinglerine geri beslemez. Bu testte exact-score NLL güçlü biçimde iyileşmiş; ancak 1X2 Brier/log-loss kapıları ve gol kalibrasyonu geçilememiştir. Bu nedenle katman production'a alınmamış, shadow araştırma adayı olarak bırakılmıştır.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
