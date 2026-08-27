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
from scripts.run_domestic_surprise_cap_sensitivity import (  # noqa: E402
    DYNAMIC_ROOT,
    EVENTS_PATH,
    FEATURE_ROOT,
    FINAL_CONTRACT,
    STATIC_ROOT,
    decision_row,
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


OUTPUT_ROOT = FEATURE_ROOT / "gamma_sensitivity_cap_30"
GAMMAS = (0.0, 0.25, 0.50, 0.70, 0.85, 1.0)
COEFFICIENT = 0.40
CAP = 30.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Variance penalty sensitivity with theta=0.40 and cap=30"
    )
    parser.add_argument("--feature-root", type=Path, default=FEATURE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    parser.add_argument(
        "--variance-penalty",
        type=float,
        default=None,
        help=(
            "Pin the emitted candidate to this variance penalty instead of "
            "re-selecting one from the surface. Rebuilding the seed after an "
            "unrelated activation would otherwise silently re-open this "
            "parameter; pinning it to the value the production contract "
            "declares keeps the rebuild a propagation rather than a new "
            "selection. Default preserves the automatic choice."
        ),
    )
    args = parser.parse_args()

    features_path = args.feature_root.resolve() / "domestic_surprise_features.csv"
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
            variance_penalty=gamma,
            max_abs_adjustment=CAP,
        )
        for gamma in GAMMAS
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
                    "variance_penalty": config.variance_penalty,
                    "fold": fold,
                    "test_season": season,
                    "matches": evaluated.metrics["matches"],
                    "brier_1x2": evaluated.metrics["brier_1x2"],
                    "brier_difference": evaluated.metrics["brier_1x2"]
                    - baseline_metric["brier_1x2"],
                    "log_loss_1x2": evaluated.metrics["log_loss_1x2"],
                    "log_loss_difference": evaluated.metrics["log_loss_1x2"]
                    - baseline_metric["log_loss_1x2"],
                    "ranking_score": ranking["ranking_score"],
                    "ranking_difference": ranking["ranking_score"]
                    - baseline_ranking["ranking_score"],
                    "pairwise_accuracy": ranking["pairwise_accuracy"],
                    "pairwise_difference": ranking["pairwise_accuracy"]
                    - baseline_ranking["pairwise_accuracy"],
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
            paired.insert(0, "variance_penalty", config.variance_penalty)
            paired_frames.append(paired)

        fold_table = pd.DataFrame(rows)
        predictions = pd.concat(paired_frames, ignore_index=True)
        competition = summarize_competitions(predictions)
        competition.insert(0, "variance_penalty", config.variance_penalty)
        uncertainty = comparison_uncertainty(predictions, args.bootstrap_samples)
        uncertainty.insert(0, "variance_penalty", config.variance_penalty)
        decision = decision_row(config, fold_table, competition, uncertainty)
        decision["variance_penalty"] = config.variance_penalty
        decision_rows.append(decision)
        fold_frames.append(fold_table)
        prediction_frames.append(predictions)
        competition_frames.append(competition)
        uncertainty_frames.append(uncertainty)

    fold_results = pd.concat(fold_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    competition = pd.concat(competition_frames, ignore_index=True)
    uncertainty = pd.concat(uncertainty_frames, ignore_index=True)
    gamma_zero_uncertainty = compare_with_gamma_zero(
        predictions, args.bootstrap_samples
    )
    decisions = add_gamma_zero_comparison(
        pd.DataFrame(decision_rows), competition, gamma_zero_uncertainty
    )
    distribution = adjustment_distribution(adjustments, candidates)
    selected_gamma = select_gamma(decisions)
    if args.variance_penalty is not None:
        requested = float(args.variance_penalty)
        available = sorted({config.variance_penalty for config in candidates})
        if not any(abs(requested - value) <= 1e-12 for value in available):
            raise ValueError(
                f"--variance-penalty {requested} is not on the evaluated surface {available}"
            )
        if abs(requested - selected_gamma) > 1e-12:
            print(
                f"Pinning variance penalty to {requested:g}; the surface would "
                f"have selected {selected_gamma:g}",
                flush=True,
            )
        selected_gamma = requested
    selected_config = next(
        config
        for config in candidates
        if abs(config.variance_penalty - selected_gamma) <= 1e-12
    )
    selected_adjustments = adjustments[selected_config].copy()

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    fold_results.to_csv(output_root / "fold_results.csv", index=False)
    predictions.to_csv(output_root / "unseen_predictions.csv", index=False)
    competition.to_csv(output_root / "competition_summary.csv", index=False)
    uncertainty.to_csv(output_root / "dependency_uncertainty.csv", index=False)
    gamma_zero_uncertainty.to_csv(
        output_root / "gamma_zero_dependency_uncertainty.csv", index=False
    )
    decisions.to_csv(output_root / "gamma_decision_table.csv", index=False)
    distribution.to_csv(output_root / "adjustment_distribution.csv", index=False)
    selected_adjustments.to_csv(
        output_root / "selected_candidate_team_adjustments.csv", index=False
    )
    manifest = {
        "feature": "DOMESTIC_SURPRISE_GAMMA_SENSITIVITY",
        "production_change": False,
        "fixed_parameters": {
            "coefficient": COEFFICIENT,
            "max_abs_adjustment": CAP,
            "minimum_history_seasons": 5,
            "volatility_normalization": "min(1,2*volatility)",
        },
        "tested_variance_penalties": list(GAMMAS),
        "selected_gamma": selected_gamma,
        "decision": "SHADOW_GAMMA_RECOMMENDATION",
    }
    (output_root / "decision.json").write_text(
        json.dumps(manifest, indent=2, default=json_default), encoding="utf-8"
    )
    write_report(
        output_root / "gamma_sensitivity_report.md",
        manifest,
        decisions,
        fold_results,
        competition,
        uncertainty,
        gamma_zero_uncertainty,
        distribution,
    )
    print(f"Selected gamma recommendation: {selected_gamma:g}")
    print(f"Report: {output_root / 'gamma_sensitivity_report.md'}")


def add_gamma_zero_comparison(
    decisions: pd.DataFrame,
    competition: pd.DataFrame,
    gamma_zero_uncertainty: pd.DataFrame,
) -> pd.DataFrame:
    gamma_zero = competition.loc[
        competition["variance_penalty"].eq(0.0)
        & competition["competition"].eq("ALL")
    ].iloc[0]
    decisions["brier_vs_gamma_zero"] = (
        decisions["brier_difference"] - gamma_zero.brier_difference
    )
    decisions["log_loss_vs_gamma_zero"] = (
        decisions["log_loss_difference"] - gamma_zero.log_loss_difference
    )
    brier_envelope = gamma_zero_uncertainty.loc[
        gamma_zero_uncertainty["metric"].eq("brier_1x2")
        & gamma_zero_uncertainty["method"].eq("conservative_envelope"),
        ["variance_penalty", "ci_95_lower", "ci_95_upper"],
    ].rename(
        columns={
            "ci_95_lower": "brier_vs_gamma_zero_ci_95_lower",
            "ci_95_upper": "brier_vs_gamma_zero_ci_95_upper",
        }
    )
    decisions = decisions.merge(brier_envelope, on="variance_penalty", how="left")
    return decisions.sort_values("variance_penalty").reset_index(drop=True)


def compare_with_gamma_zero(
    predictions: pd.DataFrame, bootstrap_samples: int
) -> pd.DataFrame:
    keys = ["fold", "match_id"]
    gamma_zero = predictions.loc[
        predictions["variance_penalty"].eq(0.0),
        keys + ["candidate_brier_1x2", "candidate_log_loss_1x2"],
    ].rename(
        columns={
            "candidate_brier_1x2": "gamma_zero_brier_1x2",
            "candidate_log_loss_1x2": "gamma_zero_log_loss_1x2",
        }
    )
    frames = []
    for gamma, frame in predictions.groupby("variance_penalty", sort=True):
        paired = frame.merge(gamma_zero, on=keys, how="inner", validate="one_to_one")
        paired["brier_difference"] = (
            paired["candidate_brier_1x2"] - paired["gamma_zero_brier_1x2"]
        )
        paired["log_loss_difference"] = (
            paired["candidate_log_loss_1x2"] - paired["gamma_zero_log_loss_1x2"]
        )
        result = comparison_uncertainty(paired, bootstrap_samples)
        result.insert(0, "variance_penalty", gamma)
        frames.append(result)
    return pd.concat(frames, ignore_index=True)


def adjustment_distribution(
    adjustments: dict[VarianceDomesticSurpriseConfig, pd.DataFrame],
    candidates: tuple[VarianceDomesticSurpriseConfig, ...],
) -> pd.DataFrame:
    rows = []
    for config in candidates:
        table = adjustments[config].copy()
        table["cap_hit"] = table["domestic_prior_adjustment"].abs().ge(CAP - 1e-9)
        for season, frame in [("ALL", table), *table.groupby("season", sort=True)]:
            eligible = frame.loc[frame["history_seasons"].eq(5)]
            rows.append(
                {
                    "variance_penalty": config.variance_penalty,
                    "season": season,
                    "eligible_team_seasons": len(eligible),
                    "cap_hits": int(eligible["cap_hit"].sum()),
                    "cap_hit_rate": float(eligible["cap_hit"].mean()),
                    "median_consistency_multiplier": float(
                        eligible["consistency_multiplier"].median()
                    ),
                    "p10_consistency_multiplier": float(
                        eligible["consistency_multiplier"].quantile(0.10)
                    ),
                    "median_abs_domestic_adjustment": float(
                        eligible["domestic_prior_adjustment"].abs().median()
                    ),
                    "p90_abs_ao_adjustment": float(
                        eligible["ao_first_elo_adjustment"].abs().quantile(0.90)
                    ),
                    "max_abs_ao_adjustment": float(
                        eligible["ao_first_elo_adjustment"].abs().max()
                    ),
                }
            )
    return pd.DataFrame(rows)


def select_gamma(decisions: pd.DataFrame) -> float:
    eligible = decisions.loc[
        decisions["ranking_no_regression_folds"].eq(6)
        & decisions["pairwise_no_regression_folds"].ge(5)
        & decisions["brier_ci_95_upper"].lt(0.0)
        & decisions["zero_sum"]
    ]
    if eligible.empty:
        eligible = decisions
    eligible = eligible.sort_values(
        ["brier_difference", "log_loss_difference", "variance_penalty"],
        ascending=[True, True, True],
    )
    return float(eligible.iloc[0].variance_penalty)


def write_report(
    path: Path,
    manifest: dict[str, object],
    decisions: pd.DataFrame,
    folds: pd.DataFrame,
    competition: pd.DataFrame,
    uncertainty: pd.DataFrame,
    gamma_zero_uncertainty: pd.DataFrame,
    distribution: pd.DataFrame,
) -> None:
    latest = distribution.loc[distribution["season"].eq("2025/26")]
    lines = [
        "# Domestic Surprise Gamma Sensitivity",
        "",
        "Fixed: theta=0.40, cap=+/-30 Elo, five complete seasons.",
        "Production was not changed.",
        "",
        f"Selected shadow gamma: **{manifest['selected_gamma']:g}**.",
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
        "## Variance penalty versus gamma=0",
        "",
        markdown_table(gamma_zero_uncertainty),
        "",
        "## 2025/26 adjustment distribution",
        "",
        markdown_table(latest),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
