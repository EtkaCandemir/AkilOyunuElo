from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ao_elo.config import AOEuropeanEloConfig  # noqa: E402
from ao_elo.domestic_surprise_variance import (  # noqa: E402
    VarianceDomesticSurpriseConfig,
)
from ao_elo.evaluation import schedule_adjusted_team_performance  # noqa: E402
from scripts.run_domestic_surprise_backtest import (  # noqa: E402
    evaluate_live,
    fold_core,
    initial_ranking,
    json_default,
    live_config_from_contract,
    markdown_table,
    pair_predictions,
    summarize_competitions,
)
from scripts.run_domestic_surprise_variance_backtest import (  # noqa: E402
    build_adjustments,
)
from scripts.run_dynamic_core_calibration import expanding_folds  # noqa: E402
from scripts.run_match_context_backtest import (  # noqa: E402
    comparison_uncertainty,
    read_context_events,
)
from scripts.run_v2_achievement_reserve_calibration import (  # noqa: E402
    load_reserve_data,
)


FEATURE_ROOT = ROOT / "output" / "domestic_surprise_variance_backtest_2018_2026"
STATIC_ROOT = ROOT / "data" / "backtest_stage_b_2018_2026"
EVENTS_PATH = ROOT / "data" / "external_elo_benchmark_2018_2026" / "exact_date_events.csv"
DYNAMIC_ROOT = ROOT / "output" / "v2_dynamic_calibration_2018_2026"
FINAL_CONTRACT = ROOT / "contracts" / "ao_european_elo_v2_final_candidate.json"
OUTPUT_ROOT = FEATURE_ROOT / "cap_sensitivity_wide"
CAPS = (15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0)
COEFFICIENT = 0.40
VARIANCE_PENALTY = 0.70


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wide cap sensitivity for variance Domestic Surprise"
    )
    parser.add_argument("--feature-root", type=Path, default=FEATURE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument(
        "--caps",
        type=float,
        nargs="+",
        default=list(CAPS),
        help="Positive maximum absolute Domestic Prior adjustments to test",
    )
    args = parser.parse_args()
    caps = tuple(sorted(set(args.caps)))
    if not caps or any(not np.isfinite(cap) or cap <= 0.0 for cap in caps):
        raise ValueError("--caps must contain positive finite values")

    feature_root = args.feature_root.resolve()
    features_path = feature_root / "domestic_surprise_features.csv"
    if not features_path.exists():
        raise FileNotFoundError(
            "Run scripts/run_domestic_surprise_variance_backtest.py first"
        )
    features = pd.read_csv(features_path)
    events = read_context_events(EVENTS_PATH)
    target = schedule_adjusted_team_performance(events)
    dynamic_manifest = json.loads(
        (DYNAMIC_ROOT / "selected_dynamic_model.json").read_text(encoding="utf-8")
    )
    static_config = AOEuropeanEloConfig(**dynamic_manifest["static_config"])
    final_contract = json.loads(FINAL_CONTRACT.read_text(encoding="utf-8"))
    live = live_config_from_contract(final_contract)
    datasets, _ = load_reserve_data(STATIC_ROOT, EVENTS_PATH, static_config)
    folds = expanding_folds(tuple(data.season for data in datasets))
    core_selections = pd.read_csv(DYNAMIC_ROOT / "core_fold_selections.csv")

    baseline = VarianceDomesticSurpriseConfig()
    candidates = tuple(
        VarianceDomesticSurpriseConfig(
            coefficient=COEFFICIENT,
            variance_penalty=VARIANCE_PENALTY,
            max_abs_adjustment=cap,
        )
        for cap in caps
    )
    adjustments = {
        config: build_adjustments(features, config, static_config)
        for config in (baseline, *candidates)
    }
    baseline_metrics = {}
    baseline_rankings = {}
    baseline_predictions = {}
    for fold, (_, season) in enumerate(folds, start=1):
        core = fold_core(core_selections, fold)
        evaluated = evaluate_live(
            datasets,
            adjustments[baseline],
            {season},
            core,
            live,
            return_predictions=True,
        )
        baseline_metrics[fold] = evaluated.metrics
        baseline_rankings[fold] = initial_ranking(
            adjustments[baseline], target, {season}
        )
        baseline_predictions[fold] = evaluated.predictions

    fold_frames = []
    prediction_frames = []
    competition_frames = []
    uncertainty_frames = []
    decision_rows = []
    for config in candidates:
        cap = config.max_abs_adjustment
        rows = []
        paired_frames = []
        for fold, (_, season) in enumerate(folds, start=1):
            core = fold_core(core_selections, fold)
            evaluated = evaluate_live(
                datasets,
                adjustments[config],
                {season},
                core,
                live,
                return_predictions=True,
            )
            ranking = initial_ranking(adjustments[config], target, {season})
            baseline_metric = baseline_metrics[fold]
            baseline_ranking = baseline_rankings[fold]
            rows.append(
                {
                    "cap": cap,
                    "fold": fold,
                    "test_season": season,
                    "matches": evaluated.metrics["matches"],
                    "brier_1x2": evaluated.metrics["brier_1x2"],
                    "baseline_brier_1x2": baseline_metric["brier_1x2"],
                    "brier_difference": (
                        evaluated.metrics["brier_1x2"]
                        - baseline_metric["brier_1x2"]
                    ),
                    "log_loss_1x2": evaluated.metrics["log_loss_1x2"],
                    "baseline_log_loss_1x2": baseline_metric["log_loss_1x2"],
                    "log_loss_difference": (
                        evaluated.metrics["log_loss_1x2"]
                        - baseline_metric["log_loss_1x2"]
                    ),
                    "ranking_score": ranking["ranking_score"],
                    "ranking_difference": (
                        ranking["ranking_score"]
                        - baseline_ranking["ranking_score"]
                    ),
                    "pairwise_accuracy": ranking["pairwise_accuracy"],
                    "pairwise_difference": (
                        ranking["pairwise_accuracy"]
                        - baseline_ranking["pairwise_accuracy"]
                    ),
                    "max_abs_match_delta": evaluated.metrics["max_abs_match_delta"],
                    "max_pair_sum_error": evaluated.metrics["max_pair_sum_error"],
                }
            )
            paired = pair_predictions(
                evaluated.predictions,
                baseline_predictions[fold],
                events,
                fold,
            )
            paired.insert(0, "cap", cap)
            paired_frames.append(paired)
        fold_table = pd.DataFrame(rows)
        predictions = pd.concat(paired_frames, ignore_index=True)
        competition = summarize_competitions(predictions)
        competition.insert(0, "cap", cap)
        uncertainty = comparison_uncertainty(predictions, args.bootstrap_samples)
        uncertainty.insert(0, "cap", cap)
        decision_rows.append(decision_row(config, fold_table, competition, uncertainty))
        fold_frames.append(fold_table)
        prediction_frames.append(predictions)
        competition_frames.append(competition)
        uncertainty_frames.append(uncertainty)
        print(
            f"cap={cap:g}: pooled Brier {competition.iloc[0].brier_difference:+.8f}, "
            f"log-loss {competition.iloc[0].log_loss_difference:+.8f}",
            flush=True,
        )

    fold_results = pd.concat(fold_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    competition = pd.concat(competition_frames, ignore_index=True)
    uncertainty = pd.concat(uncertainty_frames, ignore_index=True)
    decisions = pd.DataFrame(decision_rows)
    distribution, team_comparison = adjustment_distribution(adjustments, candidates)
    selected_cap = select_cap(decisions)
    balanced_cap = select_balanced_cap(decisions)

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    fold_results.to_csv(output_root / "fold_results.csv", index=False)
    predictions.to_csv(output_root / "unseen_predictions.csv", index=False)
    competition.to_csv(output_root / "competition_summary.csv", index=False)
    uncertainty.to_csv(output_root / "dependency_uncertainty.csv", index=False)
    decisions.to_csv(output_root / "cap_decision_table.csv", index=False)
    distribution.to_csv(output_root / "adjustment_distribution.csv", index=False)
    team_comparison.to_csv(output_root / "team_adjustment_comparison_2025_26.csv", index=False)
    manifest = {
        "feature": "DOMESTIC_SURPRISE_CAP_SENSITIVITY",
        "production_change": False,
        "fixed_parameters": {
            "coefficient": COEFFICIENT,
            "variance_penalty": VARIANCE_PENALTY,
            "minimum_history_seasons": 5,
        },
        "tested_caps": list(caps),
        "ranking_first_selected_cap": selected_cap,
        "balanced_effect_selected_cap": balanced_cap,
        "decision": "SHADOW_CAP_RECOMMENDATION",
        "decision_table": decisions.to_dict(orient="records"),
    }
    (output_root / "decision.json").write_text(
        json.dumps(manifest, indent=2, default=json_default), encoding="utf-8"
    )
    write_report(
        output_root / "cap_sensitivity_report.md",
        manifest,
        decisions,
        fold_results,
        competition,
        uncertainty,
        distribution,
        team_comparison,
    )
    print(f"Selected cap recommendation: {selected_cap:g}")
    print(f"Balanced effect recommendation: {balanced_cap:g}")
    print(f"Report: {output_root / 'cap_sensitivity_report.md'}")


def decision_row(
    config: VarianceDomesticSurpriseConfig,
    folds: pd.DataFrame,
    competition: pd.DataFrame,
    uncertainty: pd.DataFrame,
) -> dict[str, object]:
    overall = competition.loc[competition["competition"].eq("ALL")].iloc[0]
    segments = competition.loc[competition["competition"].ne("ALL")]
    brier_ci = uncertainty.loc[
        (uncertainty["metric"].eq("brier_1x2"))
        & (uncertainty["method"].eq("conservative_envelope"))
    ].iloc[0]
    return {
        "cap": config.max_abs_adjustment,
        "brier_difference": float(overall.brier_difference),
        "log_loss_difference": float(overall.log_loss_difference),
        "brier_fold_wins": int(folds["brier_difference"].lt(-1e-12).sum()),
        "log_loss_fold_wins": int(folds["log_loss_difference"].lt(-1e-12).sum()),
        "ranking_no_regression_folds": int(
            folds["ranking_difference"].ge(-1e-12).sum()
        ),
        "pairwise_no_regression_folds": int(
            folds["pairwise_difference"].ge(-1e-12).sum()
        ),
        "ranking_improvement_folds": int(
            folds["ranking_difference"].gt(1e-12).sum()
        ),
        "pairwise_improvement_folds": int(
            folds["pairwise_difference"].gt(1e-12).sum()
        ),
        "worst_ranking_difference": float(folds["ranking_difference"].min()),
        "worst_pairwise_difference": float(folds["pairwise_difference"].min()),
        "max_competition_brier_difference": float(segments.brier_difference.max()),
        "brier_ci_95_upper": float(brier_ci.ci_95_upper),
        "zero_sum": bool(folds["max_pair_sum_error"].max() <= 1e-9),
    }


def adjustment_distribution(
    adjustments: dict[VarianceDomesticSurpriseConfig, pd.DataFrame],
    candidates: tuple[VarianceDomesticSurpriseConfig, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    latest_frames = []
    for config in candidates:
        cap = config.max_abs_adjustment
        table = adjustments[config].copy()
        table["cap"] = cap
        table["cap_hit"] = table["domestic_prior_adjustment"].abs().ge(cap - 1e-9)
        table["active"] = table["ao_first_elo_adjustment"].abs().gt(1e-12)
        for season, frame in [("ALL", table), *table.groupby("season", sort=True)]:
            eligible = frame.loc[frame["history_seasons"].eq(5)]
            active = eligible.loc[eligible["active"]]
            rows.append(
                {
                    "cap": cap,
                    "season": season,
                    "eligible_team_seasons": len(eligible),
                    "active_adjustments": len(active),
                    "cap_hits": int(eligible["cap_hit"].sum()),
                    "cap_hit_rate": float(eligible["cap_hit"].mean()),
                    "median_abs_domestic_adjustment": float(
                        eligible["domestic_prior_adjustment"].abs().median()
                    ),
                    "p90_abs_domestic_adjustment": float(
                        eligible["domestic_prior_adjustment"].abs().quantile(0.90)
                    ),
                    "median_abs_ao_adjustment": float(
                        eligible["ao_first_elo_adjustment"].abs().median()
                    ),
                    "p90_abs_ao_adjustment": float(
                        eligible["ao_first_elo_adjustment"].abs().quantile(0.90)
                    ),
                    "max_abs_ao_adjustment": float(
                        eligible["ao_first_elo_adjustment"].abs().max()
                    ),
                }
            )
        latest = table.loc[table["season"].eq("2025/26")].copy()
        latest_frames.append(
            latest[
                [
                    "team_id",
                    "team_name",
                    "club_id",
                    "cap",
                    "historical_mean",
                    "historical_volatility",
                    "raw_surprise",
                    "effective_surprise",
                    "domestic_prior_adjustment",
                    "ao_first_elo_adjustment",
                    "cap_hit",
                ]
            ]
        )
    latest_long = pd.concat(latest_frames, ignore_index=True)
    values = latest_long.pivot(
        index=[
            "team_id",
            "team_name",
            "club_id",
            "historical_mean",
            "historical_volatility",
            "raw_surprise",
            "effective_surprise",
        ],
        columns="cap",
        values=["domestic_prior_adjustment", "ao_first_elo_adjustment", "cap_hit"],
    )
    values.columns = [
        f"{metric}_cap_{int(cap)}" for metric, cap in values.columns.to_flat_index()
    ]
    return pd.DataFrame(rows), values.reset_index()


def select_cap(decisions: pd.DataFrame) -> float:
    safe = decisions.loc[
        decisions["brier_fold_wins"].eq(6)
        & decisions["log_loss_fold_wins"].eq(6)
        & decisions["ranking_no_regression_folds"].eq(6)
    ]
    if safe.empty:
        safe = decisions.sort_values(
            [
                "ranking_no_regression_folds",
                "pairwise_no_regression_folds",
                "brier_difference",
            ],
            ascending=[False, False, True],
        ).head(1)
    else:
        safe = safe.sort_values(
            ["pairwise_no_regression_folds", "brier_difference", "cap"],
            ascending=[False, True, True],
        ).head(1)
    return float(safe.iloc[0].cap)


def select_balanced_cap(decisions: pd.DataFrame) -> float:
    balanced = decisions.loc[
        decisions["ranking_no_regression_folds"].eq(6)
        & decisions["pairwise_no_regression_folds"].ge(5)
        & decisions["brier_ci_95_upper"].lt(0.0)
        & decisions["zero_sum"]
    ]
    if balanced.empty:
        return select_cap(decisions)
    balanced = balanced.sort_values(
        ["brier_difference", "log_loss_difference", "cap"],
        ascending=[True, True, True],
    )
    return float(balanced.iloc[0].cap)


def write_report(
    path: Path,
    manifest: dict[str, object],
    decisions: pd.DataFrame,
    folds: pd.DataFrame,
    competition: pd.DataFrame,
    uncertainty: pd.DataFrame,
    distribution: pd.DataFrame,
    team_comparison: pd.DataFrame,
) -> None:
    selected = float(manifest["ranking_first_selected_cap"])
    balanced = float(manifest["balanced_effect_selected_cap"])
    latest_distribution = distribution.loc[distribution["season"].eq("2025/26")]
    largest_change = team_comparison.reindex(
        team_comparison[f"ao_first_elo_adjustment_cap_{int(balanced)}"]
        .abs()
        .sort_values(ascending=False)
        .index
    ).head(20)
    lines = [
        "# Domestic Surprise Cap Sensitivity",
        "",
        "Fixed parameters: theta=0.40, gamma=0.70, five complete seasons, direct percentile.",
        "Production was not changed.",
        "",
        f"Ranking-first shadow cap recommendation: **{selected:g} Elo**.",
        f"Balanced-effect shadow cap recommendation: **{balanced:g} Elo**.",
        "",
        "## Decision table",
        "",
        markdown_table(decisions),
        "",
        "## Fold results",
        "",
        markdown_table(folds),
        "",
        "## Competition summary",
        "",
        markdown_table(competition),
        "",
        "## Dependency-aware uncertainty",
        "",
        markdown_table(uncertainty),
        "",
        "## 2025/26 adjustment distribution",
        "",
        markdown_table(latest_distribution),
        "",
        "## Largest 2025/26 balanced-cap effects",
        "",
        markdown_table(largest_change),
        "",
        "The cap is a clipping boundary, not a flat team bonus. The distribution and cap-hit",
        "rate show how often each boundary changes the otherwise continuous adjustment.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
