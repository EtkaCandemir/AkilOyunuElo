from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ao_elo.config import AOEuropeanEloConfig  # noqa: E402
from ao_elo.evaluation import schedule_adjusted_team_performance  # noqa: E402
from ao_elo.tournament_bonus import (  # noqa: E402
    FiveStageWeightedTournamentBonusConfig,
)
from scripts.run_controlled_goal_progression_backtest import (  # noqa: E402
    prepare_controlled_data,
)
from scripts.run_dynamic_core_calibration import expanding_folds  # noqa: E402
from scripts.run_final_robustness import load_team_season_identity  # noqa: E402
from scripts.run_stage_weighted_progression_backtest import (  # noqa: E402
    DOMESTIC_ADJUSTMENTS,
    DYNAMIC_MANIFEST,
    EVENTS_PATH,
    MODEL_CURRENT,
    MODEL_NESTED,
    MODEL_NONE,
    PRODUCTION_CONTRACT,
    STATIC_DATA_ROOT,
    XG_DATA,
    BonusCandidate,
    build_competition_stage_summary,
    build_dependency_uncertainty,
    build_final_ranking_summary,
    build_forward_ranking_summary,
    decide_model,
    load_domestic_adjustments,
    load_xg_map,
    markdown_table,
    run_candidate_surface,
    run_nested_backtest,
    select_training_candidate,
    validate_production_contract,
    write_surface_diagnostics,
)
from scripts.run_v2_achievement_reserve_calibration import (  # noqa: E402
    load_reserve_data,
)
from scripts.run_v2_evaluation_upgrade import read_events  # noqa: E402


OUTPUT_ROOT = ROOT / "output" / "five_stage_weighted_progression_backtest_2018_2026"
FIVE_STAGES = (
    "KNOCKOUT_PLAYOFF",
    "ROUND_OF_16",
    "QUARTERFINAL",
    "SEMIFINAL",
    "FINAL",
)
PROFILES = {
    "GENTLE": (0.10, 0.15, 0.20, 0.25, 0.30),
    "LINEAR": (1.0 / 15.0, 2.0 / 15.0, 3.0 / 15.0, 4.0 / 15.0, 5.0 / 15.0),
    "LATE_HEAVY": (0.05, 0.10, 0.15, 0.25, 0.45),
}


def candidate_grid() -> tuple[BonusCandidate, ...]:
    candidates = [
        BonusCandidate(MODEL_CURRENT, "FIXED_FIVE", "EQUAL_FIVE", 60.0, True),
        BonusCandidate(MODEL_NONE, "NONE", "NONE", 0.0, False),
    ]
    for profile, weights in PROFILES.items():
        config = FiveStageWeightedTournamentBonusConfig(
            ucl_total_cap=60.0,
            stage_weights=tuple(zip(FIVE_STAGES, weights, strict=True)),
        )
        config.validate()
        candidates.append(
            BonusCandidate(
                f"FIVE_STAGE_{profile}_CAP_60",
                "FIVE_STAGE_WEIGHTED",
                profile,
                60.0,
                True,
                config,
            )
        )
    return tuple(candidates)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nested test of five-stage weighted progression profiles"
    )
    parser.add_argument("--static-data-root", type=Path, default=STATIC_DATA_ROOT)
    parser.add_argument("--events", type=Path, default=EVENTS_PATH)
    parser.add_argument("--dynamic-manifest", type=Path, default=DYNAMIC_MANIFEST)
    parser.add_argument("--production-contract", type=Path, default=PRODUCTION_CONTRACT)
    parser.add_argument("--domestic-adjustments", type=Path, default=DOMESTIC_ADJUSTMENTS)
    parser.add_argument("--xg-data", type=Path, default=XG_DATA)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples must be at least 100")

    contract = json.loads(
        args.production_contract.resolve().read_text(encoding="utf-8")
    )
    core, parameters = validate_production_contract(contract)
    dynamic = json.loads(
        args.dynamic_manifest.resolve().read_text(encoding="utf-8")
    )
    static_config = AOEuropeanEloConfig(**dynamic["static_config"])
    static_config.validate()
    events = read_events(args.events.resolve())
    reserve, tie_audit = load_reserve_data(
        args.static_data_root.resolve(), args.events.resolve(), static_config
    )
    datasets = prepare_controlled_data(reserve, events)
    seasons = tuple(data.season for data in datasets)
    folds = expanding_folds(seasons)
    if len(folds) != 6:
        raise ValueError(f"Expected six outer folds, found {len(folds)}")
    target = schedule_adjusted_team_performance(events)
    identity = load_team_season_identity()
    domestic = load_domestic_adjustments(args.domestic_adjustments.resolve(), datasets)
    xg = load_xg_map(args.xg_data.resolve(), datasets)
    candidates = candidate_grid()

    print("Five-stage weighting: nested walk-forward", flush=True)
    nested = run_nested_backtest(
        datasets,
        folds,
        target,
        identity,
        core,
        parameters,
        domestic,
        xg,
        candidates,
    )
    print("Five-stage weighting: fixed profile surface", flush=True)
    surface, diagnostics = run_candidate_surface(
        datasets,
        folds,
        target,
        identity,
        core,
        parameters,
        domestic,
        xg,
        candidates,
    )
    predictions = nested["predictions"]
    final_ranking = build_final_ranking_summary(
        nested["same_season_ranking"], MODEL_CURRENT
    )
    forward = build_forward_ranking_summary(
        nested["end_ratings"], target, identity, seasons
    )
    competition_stage = build_competition_stage_summary(
        predictions, nested["bonus_events"]
    )
    uncertainty = build_dependency_uncertainty(
        predictions, bootstrap_samples=args.bootstrap_samples
    )
    fixed_uncertainty = build_fixed_profile_uncertainty(
        diagnostics["candidate_predictions"],
        bootstrap_samples=args.bootstrap_samples,
    )
    full_selection = select_training_candidate(
        datasets,
        target,
        core,
        parameters,
        domestic,
        xg,
        candidates,
    )
    decision = decide_model(
        nested["fold_results"],
        final_ranking,
        forward,
        competition_stage,
        uncertainty,
        nested["season_metrics"],
        full_selection,
    )

    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    surface.to_csv(output / "candidate_surface.csv", index=False)
    nested["fold_selections"].to_csv(output / "fold_selections.csv", index=False)
    nested["fold_results"].to_csv(output / "fold_results.csv", index=False)
    competition_stage.to_csv(output / "competition_stage_summary.csv", index=False)
    final_ranking.to_csv(output / "final_ranking_summary.csv", index=False)
    forward.to_csv(output / "forward_ranking_summary.csv", index=False)
    uncertainty.to_csv(output / "dependency_uncertainty.csv", index=False)
    fixed_uncertainty.to_csv(output / "fixed_profile_uncertainty.csv", index=False)
    nested["bonus_events"].to_csv(output / "bonus_event_audit.csv", index=False)
    (output / "selected_candidate.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_surface_diagnostics(output, diagnostics)
    write_report(
        output / "backtest_report.md",
        contract,
        seasons,
        surface,
        nested["fold_selections"],
        nested["fold_results"],
        final_ranking,
        forward,
        uncertainty,
        fixed_uncertainty,
        decision,
        len(xg),
        len(tie_audit),
    )
    print(f"Decision: {decision['decision']}")
    print(f"Full-history selection: {decision['full_history_candidate_key']}")
    print(f"Report: {output / 'backtest_report.md'}")


def build_fixed_profile_uncertainty(
    predictions: pd.DataFrame,
    *,
    bootstrap_samples: int,
) -> pd.DataFrame:
    frames = []
    baseline = predictions.loc[predictions["candidate_key"].eq(MODEL_CURRENT)].copy()
    baseline["model"] = MODEL_CURRENT
    candidate_keys = [
        key
        for key in predictions["candidate_key"].drop_duplicates()
        if key.startswith("FIVE_STAGE_")
    ]
    for candidate_key in candidate_keys:
        candidate = predictions.loc[
            predictions["candidate_key"].eq(candidate_key)
        ].copy()
        candidate["model"] = MODEL_NESTED
        comparison = pd.concat([baseline, candidate], ignore_index=True)
        result = build_dependency_uncertainty(
            comparison,
            bootstrap_samples=bootstrap_samples,
        )
        result["candidate_key"] = candidate_key
        frames.append(result)
    return pd.concat(frames, ignore_index=True)


def write_report(
    path: Path,
    contract: dict[str, object],
    seasons: tuple[str, ...],
    surface: pd.DataFrame,
    selections: pd.DataFrame,
    folds: pd.DataFrame,
    final_ranking: pd.DataFrame,
    forward: pd.DataFrame,
    uncertainty: pd.DataFrame,
    fixed_uncertainty: pd.DataFrame,
    decision: dict[str, object],
    xg_matches: int,
    tie_audit_rows: int,
) -> None:
    current = folds.loc[folds["model"].eq(MODEL_CURRENT)]
    nested = folds.loc[folds["model"].eq(MODEL_NESTED)]
    current_brier = float(np.average(current["brier_1x2"], weights=current["matches"]))
    nested_brier = float(np.average(nested["brier_1x2"], weights=nested["matches"]))
    current_log = float(np.average(current["log_loss_1x2"], weights=current["matches"]))
    nested_log = float(np.average(nested["log_loss_1x2"], weights=nested["matches"]))
    rank = final_ranking.loc[final_ranking["model"].eq(MODEL_NESTED)][
        ["competition", "ranking_delta_vs_current", "pairwise_delta_vs_current"]
    ]
    forward_all = forward.loc[
        forward["model"].eq(MODEL_NESTED) & forward["competition"].eq("ALL")
    ][
        ["source_season", "target_season", "ranking_delta_vs_current", "pairwise_delta_vs_current"]
    ]
    envelope = uncertainty.loc[uncertainty["method"].eq("conservative_envelope")][
        ["competition", "metric", "mean_difference", "ci_95_lower", "ci_95_upper", "reliable_harm"]
    ]
    fixed_envelope = fixed_uncertainty.loc[
        fixed_uncertainty["method"].eq("conservative_envelope")
        & fixed_uncertainty["competition"].eq("ALL")
    ][
        ["candidate_key", "metric", "mean_difference", "ci_95_lower", "ci_95_upper", "reliable_improvement", "reliable_harm"]
    ]
    fixed_harm = fixed_uncertainty.loc[
        fixed_uncertainty["method"].eq("conservative_envelope")
        & fixed_uncertainty["reliable_harm"].astype(bool)
    ][
        ["candidate_key", "competition", "metric", "mean_difference", "ci_95_lower", "ci_95_upper"]
    ]
    lines = [
        "# Beş Aşamalı Ağırlıklı European Progression Backtesti",
        "",
        "## Tasarım",
        "",
        f"- Dönem: `{seasons[0]}`–`{seasons[-1]}`; altı expanding outer fold.",
        "- UCL/UEL/UECL toplam cap değerleri sabit `60/40/20` tutuldu.",
        "- Böylece yalnız beş aşama arasındaki dağılım test edildi; toplam bonus miktarı değiştirilmedi.",
        "- Equal Five production kontrolüdür: her aşama `%20`, yani `12/8/4`.",
        "- Gentle: `%10/%15/%20/%25/%30`.",
        "- Linear: `1/15, 2/15, 3/15, 4/15, 5/15`.",
        "- Late Heavy: `%5/%10/%15/%25/%45`.",
        "- Sıra: KPO, Son 16, Çeyrek Final, Yarı Final, Şampiyonluk.",
        f"- Doğrulanmış xG: `{xg_matches}` maç; diğerlerinde production GD fallback.",
        "- Production sözleşmesi bu çalışma tarafından değiştirilmez.",
        "",
        "## Fold Seçimleri",
        "",
        markdown_table(selections),
        "",
        "## Pooled Tahmin Sonuçları",
        "",
        f"- Brier: current `{current_brier:.9f}`, nested `{nested_brier:.9f}`, fark `{decision['pooled_brier_delta_vs_current']:+.9f}`.",
        f"- Log-loss: current `{current_log:.9f}`, nested `{nested_log:.9f}`, fark `{decision['pooled_log_loss_delta_vs_current']:+.9f}`.",
        f"- Aynı-sezon ortak ranking win: `{decision['nested_rank_wins']}`.",
        f"- Forward ortak ranking win: `{decision['forward_rank_wins']}`.",
        "",
        "### Turnuva Bazlı Ranking",
        "",
        markdown_table(rank, float_digits=9),
        "",
        "### Forward Geçişler",
        "",
        markdown_table(forward_all, float_digits=9),
        "",
        "### Dependency Belirsizliği",
        "",
        "Nested seçim altı foldun tamamında production profilinde kaldığı için bu tablodaki farklar sıfırdır.",
        "",
        markdown_table(envelope, float_digits=9),
        "",
        "### Sabit Profillerin Pooled Belirsizliği",
        "",
        markdown_table(fixed_envelope, float_digits=9),
        "",
        "### Sabit Profillerde Güvenilir Zarar Segmentleri",
        "",
        markdown_table(fixed_harm, float_digits=9),
        "",
        "## Sabit Profil Yüzeyi",
        "",
        markdown_table(
            surface[[
                "candidate_key", "brier_1x2", "log_loss_1x2", "ranking_score",
                "pairwise_accuracy", "forward_ranking_score", "forward_pairwise_accuracy",
            ]],
            float_digits=9,
        ),
        "",
        "## Karar",
        "",
        f"**{decision['decision']}**",
        "",
        f"Tam geçmiş seçimi: `{decision['full_history_candidate_key']}`.",
        "",
        *[f"- `{key}`: `{value}`" for key, value in decision["gates"].items()],
        "",
        "Bu sonuç öneridir; production JSON'u değiştirilmemiştir.",
        "",
        "## Audit",
        "",
        f"- Tie kronoloji audit satırı: `{tie_audit_rows}`.",
        f"- Scale/H/K: `{contract['dynamic_core']['elo_scale']:.6f}` / `{contract['dynamic_core']['home_advantage']:.6f}` / `{contract['dynamic_core']['k_factor']:.6f}`.",
        "- Bonus winner-only, sezon-resetli ve tek tie uygulamalıdır.",
        "- Penaltı atışının kendisi GD/xG sinyali üretmez; penaltıyla tur geçen takım aşama bonusunu tam alır.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
