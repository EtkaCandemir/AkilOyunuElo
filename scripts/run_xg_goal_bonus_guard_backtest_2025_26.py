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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ao_elo.evaluation import schedule_adjusted_team_performance  # noqa: E402
from scripts.run_fotmob_xg_backtest_2025_26 import (  # noqa: E402
    DATA_PATH,
    MANIFEST_PATH,
    PRODUCTION_MODEL_PATH,
    SEASON,
    STATIC_DATA_ROOT,
    STATIC_MANIFEST_PATH,
    markdown_table,
    read_fotmob_xg_dataset,
    validate_manifest,
)
from scripts.run_unsupported_margin_backtest_2025_26 import (  # noqa: E402
    BASELINE_KEY,
    LEGACY_KEY,
    VALIDATION_START,
    UnsupportedMarginCandidate,
    build_competition_summary,
    build_metric_table,
    build_penalty_examples,
    build_uncertainty,
    make_decision,
    replay_candidate,
    select_development_candidate,
)
from scripts.run_xg_goal_ablation_backtest import (  # noqa: E402
    load_initial_ratings,
    read_production_contract,
    read_static_config,
)


OUTPUT_ROOT = ROOT / "output" / "xg_goal_bonus_guard_backtest_2025_26"
TOLERANCE_GRID = (0.50, 0.75, 1.00)
LAMBDA_GRID = (0.10, 0.20, 0.30, 0.50)
MINIMUM_BONUS_MULTIPLIER_GRID = (0.00, 0.25, 0.50, 0.75)


def goal_bonus_guard_grid() -> tuple[UnsupportedMarginCandidate, ...]:
    candidates = [
        UnsupportedMarginCandidate(BASELINE_KEY, "PRODUCTION"),
        UnsupportedMarginCandidate(LEGACY_KEY, "LEGACY_XG"),
    ]
    candidates.extend(
        UnsupportedMarginCandidate(
            (
                f"GOAL_BONUS_GUARD_tol{tolerance:g}_lambda{penalty_lambda:g}"
                f"_min{minimum_multiplier:g}"
            ),
            "GOAL_BONUS_GUARD",
            tolerance,
            penalty_lambda,
            minimum_multiplier,
        )
        for tolerance in TOLERANCE_GRID
        for penalty_lambda in LAMBDA_GRID
        for minimum_multiplier in MINIMUM_BONUS_MULTIPLIER_GRID
    )
    result = tuple(candidates)
    for candidate in result:
        candidate.validate()
    if len(result) != 50:
        raise ValueError(f"Expected 50 candidates, found {len(result)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Backtest xG as a guard over only the controlled goal-margin bonus"
        )
    )
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--static-data-root", type=Path, default=STATIC_DATA_ROOT)
    parser.add_argument("--static-manifest", type=Path, default=STATIC_MANIFEST_PATH)
    parser.add_argument("--production-model", type=Path, default=PRODUCTION_MODEL_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples must be at least 100")

    events = read_fotmob_xg_dataset(args.data.resolve(), strict_contract=True)
    manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    validate_manifest(manifest, events)
    production = read_production_contract(args.production_model.resolve())
    static_config = read_static_config(args.static_manifest.resolve())
    initial_ratings = load_initial_ratings(
        args.static_data_root.resolve(), events, static_config
    )[SEASON]
    development_events = events.loc[events["kickoff_utc"].lt(VALIDATION_START)]
    validation_events = events.loc[events["kickoff_utc"].ge(VALIDATION_START)]
    if int(development_events["xg_analysis_eligible"].sum()) != 399:
        raise ValueError("Expected 399 development xG matches")
    if len(validation_events) != 207 or not validation_events[
        "xg_analysis_eligible"
    ].all():
        raise ValueError("Expected 207 fully covered validation matches")

    candidates = goal_bonus_guard_grid()
    print(
        f"Goal-bonus xG guard: {len(candidates)} candidates; "
        "development=754 matches/399 xG, validation=207 xG matches",
        flush=True,
    )
    evaluations = {
        candidate.key: replay_candidate(
            events,
            initial_ratings,
            production,
            candidate,
            validation_start=VALIDATION_START,
        )
        for candidate in candidates
    }
    development_metrics = build_metric_table(
        evaluations,
        candidates,
        schedule_adjusted_team_performance(development_events),
        split="DEVELOPMENT",
    )
    selected, selection = select_development_candidate(
        development_metrics,
        candidate_kind="GOAL_BONUS_GUARD",
    )
    validation_metrics = build_metric_table(
        evaluations,
        candidates,
        schedule_adjusted_team_performance(events),
        split="VALIDATION",
    )
    comparison_keys = (BASELINE_KEY, LEGACY_KEY, selected.key)
    comparison = pd.concat(
        [
            development_metrics.loc[
                development_metrics["candidate_key"].isin(comparison_keys)
            ],
            validation_metrics.loc[
                validation_metrics["candidate_key"].isin(comparison_keys)
            ],
        ],
        ignore_index=True,
    )
    selected_predictions = pd.concat(
        [evaluations[key].predictions for key in comparison_keys],
        ignore_index=True,
    )
    selected_predictions["evaluation_split"] = selected_predictions[
        "kickoff_utc"
    ].map(lambda value: "DEVELOPMENT" if value < VALIDATION_START else "VALIDATION")
    validation_predictions = selected_predictions.loc[
        selected_predictions["evaluation_split"].eq("VALIDATION")
    ]
    competition = build_competition_summary(validation_predictions)
    uncertainty = build_uncertainty(
        selected_predictions,
        selected.key,
        bootstrap_samples=args.bootstrap_samples,
    )
    selected_validation = evaluations[selected.key].predictions.loc[
        lambda frame: frame["kickoff_utc"].ge(VALIDATION_START)
    ]
    penalty_examples = build_penalty_examples(selected_validation)
    decision, guardrails = make_decision(
        comparison,
        competition,
        uncertainty,
        selected,
        selection,
    )

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    development_metrics.to_csv(
        output_root / "development_candidate_metrics.csv", index=False
    )
    validation_metrics.to_csv(
        output_root / "validation_candidate_metrics.csv", index=False
    )
    comparison.to_csv(output_root / "selected_model_comparison.csv", index=False)
    selected_predictions.to_csv(
        output_root / "selected_match_updates.csv", index=False
    )
    competition.to_csv(output_root / "competition_summary.csv", index=False)
    uncertainty.to_csv(output_root / "dependency_uncertainty.csv", index=False)
    penalty_examples.to_csv(output_root / "penalty_examples.csv", index=False)
    pd.concat(
        [evaluations[key].final_ratings for key in comparison_keys],
        ignore_index=True,
    ).to_csv(output_root / "final_ratings.csv", index=False)

    payload = {
        "analysis": "XG_GOAL_BONUS_GUARD_2025_26_TEMPORAL_SPLIT",
        "formula": {
            "base": "Delta_base=K*(S-E)",
            "goal_bonus": "Delta_bonus=K*(S-E)*(M_GD-1)",
            "unsupported_margin": (
                "U=max(0,min(abs(GD),4)-winner_xg_advantage-tolerance)"
            ),
            "xg_guard": (
                "P=max(minimum_bonus_multiplier,1-lambda*ln(1+U))"
            ),
            "final": "Delta_final=Delta_base+P*Delta_bonus",
        },
        "development_end_exclusive": VALIDATION_START.isoformat(),
        "development_matches": len(development_events),
        "development_xg_matches": int(
            development_events["xg_analysis_eligible"].sum()
        ),
        "validation_matches": len(validation_events),
        "validation_xg_matches": int(validation_events["xg_analysis_eligible"].sum()),
        "selected_candidate": selected.__dict__,
        "selection": selection,
        "decision": decision,
        "guardrails": guardrails,
        "production_changed": False,
        "evidence_class": "ONE_SEASON_TEMPORAL_SHADOW_VALIDATION",
    }
    (output_root / "selected_goal_bonus_guard_model.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_root / "backtest_report.md").write_text(
        build_report(
            comparison,
            competition,
            uncertainty,
            penalty_examples,
            selected,
            selection,
            decision,
            guardrails,
        ),
        encoding="utf-8",
    )
    print(f"Selected: {selected.key}")
    print(f"Decision: {decision}")
    print(f"Report: {output_root / 'backtest_report.md'}")


def build_report(
    comparison: pd.DataFrame,
    competition: pd.DataFrame,
    uncertainty: pd.DataFrame,
    examples: pd.DataFrame,
    selected: UnsupportedMarginCandidate,
    selection: dict[str, object],
    decision: str,
    guardrails: dict[str, object],
) -> str:
    comparison_view = comparison[
        [
            "split",
            "candidate_key",
            "matches",
            "brier_1x2",
            "brier_1x2_delta_vs_production",
            "log_loss_1x2",
            "log_loss_1x2_delta_vs_production",
            "accuracy_1x2",
            "ranking_score",
            "ranking_score_delta_vs_production",
            "pairwise_accuracy",
            "pairwise_accuracy_delta_vs_production",
            "penalized_matches",
            "mean_penalty_multiplier",
        ]
    ]
    competition_view = competition[
        [
            "candidate_key",
            "competition",
            "matches",
            "brier_delta_vs_production",
            "log_loss_delta_vs_production",
        ]
    ]
    ci_view = uncertainty.loc[
        uncertainty["method"].eq("conservative_envelope"),
        [
            "candidate_key",
            "metric",
            "mean_difference",
            "ci_95_lower",
            "ci_95_upper",
        ],
    ]
    return "\n".join(
        [
            "# xG Goal Bonus Guard 2025/26 Backtesti",
            "",
            "## Model Sözleşmesi",
            "",
            "```text",
            "Delta_base = K * (S-E)",
            "Delta_bonus = K * (S-E) * (M_GD-1)",
            "U = max(0, min(abs(GD),4) - winner_xg_advantage - tolerance)",
            "P = max(minimum_bonus_multiplier, 1-lambda*ln(1+U))",
            "Delta_final = Delta_base + P * Delta_bonus",
            "```",
            "",
            "xG temel galibiyet/mağlubiyet Elo'sunu değiştirmez. Tek farklı maçta",
            "`Delta_bonus=0` olduğu için xG etkisi yoktur. Eksik xG, beraberlik ve",
            "penaltı kararında bonus mevcut production davranışını korur.",
            "",
            "## Temporal Tasarım",
            "",
            "- Development: 1 Ocak 2026 öncesi 754 maç, 399 xG.",
            "- Validation: sonraki 207 maç, tamamında xG.",
            "- Parametre yalnız development bölümünden seçildi.",
            "",
            "## Seçilen Aday",
            "",
            f"`{selected.key}`",
            "",
            markdown_table(pd.DataFrame([selection])),
            "",
            "## Ana Karşılaştırma",
            "",
            markdown_table(comparison_view),
            "",
            "## Validation Turnuva Segmentleri",
            "",
            markdown_table(competition_view),
            "",
            "## Cluster Güven Aralıkları",
            "",
            markdown_table(ci_view),
            "",
            "## En Güçlü Bonus Kısıntıları",
            "",
            markdown_table(examples.head(15)),
            "",
            "## Model Kararı",
            "",
            f"`{decision}`",
            "",
            markdown_table(pd.DataFrame([guardrails])),
        ]
    )


if __name__ == "__main__":
    main()
