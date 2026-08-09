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
    calculate_variance_domestic_surprise_adjustment,
)
from ao_elo.evaluation import schedule_adjusted_team_performance  # noqa: E402
from scripts.run_domestic_surprise_5y_backtest import (  # noqa: E402
    build_domestic_history_features,
)
from scripts.run_domestic_surprise_backtest import (  # noqa: E402
    Evaluation,
    LiveBacktestConfig,
    evaluate_live,
    fold_core,
    initial_ranking,
    json_default,
    live_config_from_contract,
    markdown_table,
    pair_predictions,
    summarize_competitions,
)
from scripts.run_dynamic_core_calibration import (  # noqa: E402
    DynamicCoreConfig,
    expanding_folds,
)
from scripts.run_match_context_backtest import (  # noqa: E402
    comparison_uncertainty,
    read_context_events,
)
from scripts.run_v2_achievement_reserve_calibration import (  # noqa: E402
    ReserveSeasonData,
    load_reserve_data,
)


STATIC_ROOT = ROOT / "data" / "backtest_stage_b_2018_2026"
EVENTS_PATH = ROOT / "data" / "external_elo_benchmark_2018_2026" / "exact_date_events.csv"
DYNAMIC_ROOT = ROOT / "output" / "v2_dynamic_calibration_2018_2026"
FINAL_CONTRACT = ROOT / "contracts" / "ao_european_elo_v2_final_candidate.json"
CLUB_IDENTITY_PATH = ROOT / "data" / "club_identity" / "team_season_identity.csv"
OUTPUT_ROOT = ROOT / "output" / "domestic_surprise_variance_backtest_2018_2026"
COEFFICIENTS = (0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40)
VARIANCE_PENALTIES = (0.0, 0.25, 0.50, 0.75, 1.0)
ADJUSTMENT_CAPS = (10.0, 15.0, 20.0, 25.0, 50.0)
RANK_TOLERANCE = 1e-9


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate five-season variance-controlled domestic surprise"
    )
    parser.add_argument("--static-root", type=Path, default=STATIC_ROOT)
    parser.add_argument("--events-path", type=Path, default=EVENTS_PATH)
    parser.add_argument("--dynamic-root", type=Path, default=DYNAMIC_ROOT)
    parser.add_argument("--final-contract", type=Path, default=FINAL_CONTRACT)
    parser.add_argument("--club-identity", type=Path, default=CLUB_IDENTITY_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()

    static_root = args.static_root.resolve()
    events = read_context_events(args.events_path.resolve())
    dynamic_root = args.dynamic_root.resolve()
    dynamic_manifest = json.loads(
        (dynamic_root / "selected_dynamic_model.json").read_text(encoding="utf-8")
    )
    static_config = AOEuropeanEloConfig(**dynamic_manifest["static_config"])
    static_config.validate()
    final_contract = json.loads(
        args.final_contract.resolve().read_text(encoding="utf-8")
    )
    live = live_config_from_contract(final_contract)
    datasets, _ = load_reserve_data(static_root, args.events_path.resolve(), static_config)
    seasons = tuple(data.season for data in datasets)
    folds = expanding_folds(seasons)
    if len(folds) != 6:
        raise ValueError(f"Expected six outer folds, found {len(folds)}")
    core_selections = pd.read_csv(dynamic_root / "core_fold_selections.csv")
    target = schedule_adjusted_team_performance(events)

    print("Variance backtest: building strict five-season features", flush=True)
    features, history_long, coverage = build_domestic_history_features(
        static_root,
        static_config,
        seasons,
        args.club_identity.resolve(),
    )
    candidates = candidate_grid()
    adjustments = {
        candidate: build_adjustments(features, candidate, static_config)
        for candidate in candidates
    }
    full_core = DynamicCoreConfig(**final_contract["dynamic_core"])
    model_families = {
        "variance_grid": candidates,
        "surprise_only": tuple(
            candidate
            for candidate in candidates
            if candidate == baseline_config() or candidate.variance_penalty == 0.0
        ),
    }

    family_outputs = []
    for family, family_candidates in model_families.items():
        print(
            f"Family {family}: {len(family_candidates)} candidates, {len(folds)} folds",
            flush=True,
        )
        selections, results, predictions, unseen_adjustments = run_walk_forward_backtest(
            datasets,
            adjustments,
            events,
            target,
            folds,
            core_selections,
            family_candidates,
            live,
        )
        competition = summarize_competitions(predictions)
        uncertainty = comparison_uncertainty(predictions, args.bootstrap_samples)
        full_candidate, full_metrics = select_candidate_from_adjustments(
            datasets,
            adjustments,
            target,
            set(seasons),
            full_core,
            family_candidates,
            live,
        )
        decision, guards = promotion_decision(
            selections, results, competition, uncertainty
        )
        family_outputs.append(
            {
                "family": family,
                "candidates": family_candidates,
                "selections": selections,
                "results": results,
                "predictions": predictions,
                "unseen_adjustments": unseen_adjustments,
                "competition": competition,
                "uncertainty": uncertainty,
                "full_candidate": full_candidate,
                "full_metrics": full_metrics,
                "decision": decision,
                "guards": guards,
            }
        )

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_root / "domestic_surprise_features.csv", index=False)
    history_long.to_csv(output_root / "domestic_history_long.csv", index=False)
    coverage.to_csv(output_root / "history_coverage.csv", index=False)
    write_family_outputs(output_root, family_outputs)
    variance_output = next(
        output for output in family_outputs if output["family"] == "variance_grid"
    )
    final_candidate = variance_output["full_candidate"]
    final_adjustments = adjustments[final_candidate]
    final_adjustments.to_csv(
        output_root / "final_candidate_team_adjustments.csv", index=False
    )
    comparison = family_comparison(family_outputs)
    comparison.to_csv(output_root / "family_comparison.csv", index=False)
    manifest = {
        "feature": "FIVE_SEASON_VARIANCE_CONTROLLED_DOMESTIC_SURPRISE",
        "decision": variance_output["decision"],
        "production_change": False,
        "matches": int(len(events)),
        "seasons": list(seasons),
        "candidate_count": len(candidates),
        "selected_full_candidate": asdict(final_candidate),
        "history_contract": {
            "strict_completed_seasons": 5,
            "weights_oldest_to_newest": [0.07, 0.13, 0.20, 0.27, 0.33],
            "normalization": "direct_percentile",
            "missing_history_behavior": "ZERO_ADJUSTMENT",
        },
        "formula_contract": {
            "variance": "sum(w_i*(P_i-weighted_mean)^2)",
            "normalized_volatility": "min(1,2*sqrt(variance))",
            "consistency_multiplier": "1-variance_penalty*normalized_volatility",
            "effective_surprise": "raw_surprise*consistency_multiplier",
            "positive_negative_symmetric": True,
            "exposure_retention": "1-effective_european_exposure",
        },
        "guardrails": variance_output["guards"],
        "family_comparison": comparison.to_dict(orient="records"),
    }
    (output_root / "decision.json").write_text(
        json.dumps(manifest, indent=2, default=json_default), encoding="utf-8"
    )
    write_report(
        output_root / "backtest_report.md",
        manifest,
        coverage,
        comparison,
        variance_output,
        final_adjustments,
    )
    print(f"Decision: {variance_output['decision']}")
    print(f"Full-data diagnostic candidate: {asdict(final_candidate)}")
    print(f"Report: {output_root / 'backtest_report.md'}")


def baseline_config() -> VarianceDomesticSurpriseConfig:
    return VarianceDomesticSurpriseConfig()


def candidate_grid() -> tuple[VarianceDomesticSurpriseConfig, ...]:
    candidates = {baseline_config()}
    candidates.update(
        VarianceDomesticSurpriseConfig(
            coefficient=coefficient,
            variance_penalty=variance_penalty,
            max_abs_adjustment=cap,
        )
        for coefficient in COEFFICIENTS
        for variance_penalty in VARIANCE_PENALTIES
        for cap in ADJUSTMENT_CAPS
    )
    result = tuple(sorted(candidates))
    for candidate in result:
        candidate.validate()
    return result


def build_adjustments(
    features: pd.DataFrame,
    config: VarianceDomesticSurpriseConfig,
    static_config: AOEuropeanEloConfig,
) -> pd.DataFrame:
    rows = []
    for row in features.itertuples(index=False):
        history = [
            None
            if pd.isna(getattr(row, f"history_direct_t_minus_{offset}"))
            else float(getattr(row, f"history_direct_t_minus_{offset}"))
            for offset in range(5, 0, -1)
        ]
        current = row.current_direct_percentile
        if not bool(row.current_finish_eligible) or pd.isna(current):
            current = 0.0
            history = [None] * 5
        adjustment = calculate_variance_domestic_surprise_adjustment(
            current_finish_score=float(current),
            historical_finish_scores=history,
            domestic_prior=float(row.domestic_prior),
            european_prior=float(row.european_prior),
            effective_european_exposure=float(row.effective_european_exposure),
            domestic_achievement_component=static_config.domestic_achievement_component,
            achievement_scale=float(row.achievement_scale),
            config=config,
        )
        rows.append(
            {
                "season": row.season,
                "team_id": int(row.team_id),
                "team_name": row.team_name,
                "club_id": row.club_id,
                "country_code": row.country_code,
                "competition": row.competition,
                "coefficient": config.coefficient,
                "variance_penalty": config.variance_penalty,
                "max_abs_adjustment": config.max_abs_adjustment,
                "current_domestic_position": row.current_domestic_position,
                "current_finish_score": current,
                "historical_mean": adjustment.historical_mean,
                "historical_variance": adjustment.historical_variance,
                "historical_volatility": adjustment.historical_volatility,
                "normalized_volatility": adjustment.normalized_volatility,
                "consistency_multiplier": adjustment.consistency_multiplier,
                "history_seasons": adjustment.history_seasons,
                "raw_surprise": adjustment.raw_surprise,
                "effective_surprise": adjustment.effective_surprise,
                "surprise_direction": (
                    "POSITIVE"
                    if adjustment.raw_surprise > 0
                    else "NEGATIVE"
                    if adjustment.raw_surprise < 0
                    else "NEUTRAL_OR_UNAVAILABLE"
                ),
                "domestic_prior_adjustment": adjustment.domestic_prior_adjustment,
                "ao_first_elo_adjustment": (
                    adjustment.adjusted_ao_first_elo - row.baseline_ao_first_elo
                ),
                "baseline_domestic_prior": row.domestic_prior,
                "adjusted_domestic_prior": adjustment.adjusted_domestic_prior,
                "baseline_ao_first_elo": row.baseline_ao_first_elo,
                "adjusted_ao_first_elo": adjustment.adjusted_ao_first_elo,
            }
        )
    return pd.DataFrame(rows)


def run_walk_forward_backtest(
    datasets: tuple[ReserveSeasonData, ...],
    adjustments: dict[VarianceDomesticSurpriseConfig, pd.DataFrame],
    events: pd.DataFrame,
    target: pd.DataFrame,
    folds: tuple[tuple[tuple[str, ...], str], ...],
    core_selections: pd.DataFrame,
    candidates: tuple[VarianceDomesticSurpriseConfig, ...],
    live: LiveBacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline = baseline_config()
    selection_rows = []
    result_rows = []
    prediction_frames = []
    adjustment_frames = []
    for fold, (train_seasons, test_season) in enumerate(folds, start=1):
        core = fold_core(core_selections, fold)
        selected, table = select_candidate_from_adjustments(
            datasets,
            adjustments,
            target,
            set(train_seasons),
            core,
            candidates,
            live,
        )
        selected_row = table.loc[
            table["candidate_key"].eq(config_key(selected))
        ].iloc[0]
        selection_rows.append(
            {
                "fold": fold,
                "train_seasons": "|".join(train_seasons),
                "test_season": test_season,
                "selected_candidate": config_key(selected),
                "selected_coefficient": selected.coefficient,
                "selected_variance_penalty": selected.variance_penalty,
                "selected_cap": selected.max_abs_adjustment,
                "selected_is_baseline": selected == baseline,
                "train_brier_1x2": selected_row.brier_1x2,
                "train_log_loss_1x2": selected_row.log_loss_1x2,
                "train_ranking_score": selected_row.ranking_score,
                "train_pairwise_accuracy": selected_row.pairwise_accuracy,
            }
        )
        predictions: dict[str, pd.DataFrame] = {}
        for model, config in (("candidate", selected), ("baseline", baseline)):
            evaluated: Evaluation = evaluate_live(
                datasets,
                adjustments[config],
                {test_season},
                core,
                live,
                return_predictions=True,
            )
            ranking = initial_ranking(adjustments[config], target, {test_season})
            result_rows.append(
                {
                    "fold": fold,
                    "test_season": test_season,
                    "model": model,
                    "candidate_key": config_key(config),
                    **evaluated.metrics,
                    **ranking,
                }
            )
            predictions[model] = evaluated.predictions
        prediction_frames.append(
            pair_predictions(
                predictions["candidate"], predictions["baseline"], events, fold
            )
        )
        selected_adjustments = adjustments[selected].loc[
            adjustments[selected]["season"].eq(test_season)
        ].copy()
        selected_adjustments.insert(0, "fold", fold)
        adjustment_frames.append(selected_adjustments)
        print(
            f"  fold {fold}/{len(folds)} {test_season}: "
            f"theta={selected.coefficient:g}, gamma={selected.variance_penalty:g}, "
            f"cap={selected.max_abs_adjustment:g}",
            flush=True,
        )
    return (
        pd.DataFrame(selection_rows),
        pd.DataFrame(result_rows),
        pd.concat(prediction_frames, ignore_index=True),
        pd.concat(adjustment_frames, ignore_index=True),
    )


def select_candidate_from_adjustments(
    datasets: tuple[ReserveSeasonData, ...],
    adjustments: dict[VarianceDomesticSurpriseConfig, pd.DataFrame],
    target: pd.DataFrame,
    seasons: set[str],
    core: DynamicCoreConfig,
    candidates: tuple[VarianceDomesticSurpriseConfig, ...],
    live: LiveBacktestConfig,
) -> tuple[VarianceDomesticSurpriseConfig, pd.DataFrame]:
    baseline = baseline_config()
    baseline_rank = initial_ranking(adjustments[baseline], target, seasons)
    rows = []
    for candidate in candidates:
        ranking = initial_ranking(adjustments[candidate], target, seasons)
        rank_safe = bool(
            ranking["ranking_score"] >= baseline_rank["ranking_score"] - RANK_TOLERANCE
            and ranking["pairwise_accuracy"]
            >= baseline_rank["pairwise_accuracy"] - RANK_TOLERANCE
        )
        if rank_safe:
            metrics = evaluate_live(
                datasets, adjustments[candidate], seasons, core, live
            ).metrics
        else:
            metrics = {
                "matches": 0,
                "brier_1x2": float("inf"),
                "log_loss_1x2": float("inf"),
                "max_abs_match_delta": float("inf"),
                "max_pair_sum_error": float("inf"),
                "max_abs_rating_change": float("inf"),
            }
        rows.append(
            {
                "candidate_key": config_key(candidate),
                "coefficient": candidate.coefficient,
                "variance_penalty": candidate.variance_penalty,
                "max_abs_adjustment": candidate.max_abs_adjustment,
                "is_baseline": candidate == baseline,
                "rank_safe": rank_safe,
                **metrics,
                **ranking,
                "distance": (
                    0.0
                    if candidate == baseline
                    else candidate.coefficient
                    + candidate.variance_penalty / 1000.0
                    + candidate.max_abs_adjustment / 10000.0
                ),
                "config": json.dumps(asdict(candidate), sort_keys=True),
            }
        )
    table = pd.DataFrame(rows)
    selected_row = table.loc[table["rank_safe"]].sort_values(
        ["brier_1x2", "log_loss_1x2", "distance", "candidate_key"],
        kind="stable",
    ).iloc[0]
    selected = next(
        candidate
        for candidate in candidates
        if config_key(candidate) == selected_row.candidate_key
    )
    return selected, table


def promotion_decision(
    selections: pd.DataFrame,
    results: pd.DataFrame,
    competition: pd.DataFrame,
    uncertainty: pd.DataFrame,
) -> tuple[str, dict[str, object]]:
    pivot = results.pivot(index="fold", columns="model")
    brier_delta = pivot["brier_1x2"]["candidate"] - pivot["brier_1x2"]["baseline"]
    log_delta = (
        pivot["log_loss_1x2"]["candidate"]
        - pivot["log_loss_1x2"]["baseline"]
    )
    rank_delta = (
        pivot["ranking_score"]["candidate"] - pivot["ranking_score"]["baseline"]
    )
    pair_delta = (
        pivot["pairwise_accuracy"]["candidate"]
        - pivot["pairwise_accuracy"]["baseline"]
    )
    overall = competition.loc[competition["competition"].eq("ALL")].iloc[0]
    segments = competition.loc[competition["competition"].ne("ALL")]
    brier_ci = uncertainty.loc[
        (uncertainty["metric"].eq("brier_1x2"))
        & (uncertainty["method"].eq("conservative_envelope"))
    ].iloc[0]
    guards = {
        "selected_active_folds": int((~selections["selected_is_baseline"]).sum()),
        "selected_variance_folds": int(
            selections["selected_variance_penalty"].gt(0).sum()
        ),
        "brier_fold_wins": int((brier_delta < -1e-12).sum()),
        "log_loss_fold_wins": int((log_delta < -1e-12).sum()),
        "ranking_no_regression_folds": int((rank_delta >= -RANK_TOLERANCE).sum()),
        "pairwise_no_regression_folds": int((pair_delta >= -RANK_TOLERANCE).sum()),
        "ranking_improvement_folds": int((rank_delta > RANK_TOLERANCE).sum()),
        "pairwise_improvement_folds": int((pair_delta > RANK_TOLERANCE).sum()),
        "overall_brier_difference": float(overall.brier_difference),
        "overall_log_loss_difference": float(overall.log_loss_difference),
        "max_competition_brier_difference": float(segments.brier_difference.max()),
        "brier_ci_upper_95": float(brier_ci.ci_95_upper),
        "power_zero_sum": bool(results["max_pair_sum_error"].max() <= 1e-9),
    }
    passed = bool(
        guards["selected_active_folds"] >= 4
        and guards["selected_variance_folds"] >= 4
        and guards["brier_fold_wins"] >= 4
        and guards["log_loss_fold_wins"] >= 4
        and guards["ranking_no_regression_folds"] == 6
        and guards["pairwise_no_regression_folds"] == 6
        and guards["ranking_improvement_folds"] >= 4
        and guards["pairwise_improvement_folds"] >= 4
        and guards["overall_brier_difference"] <= 0
        and guards["overall_log_loss_difference"] <= 0
        and guards["max_competition_brier_difference"] <= 0
        and guards["brier_ci_upper_95"] <= 0
        and guards["power_zero_sum"]
    )
    return ("PROMOTE_CANDIDATE" if passed else "KEEP_BASELINE"), guards


def write_family_outputs(output_root: Path, outputs: list[dict[str, object]]) -> None:
    table_names = (
        "selections",
        "results",
        "predictions",
        "unseen_adjustments",
        "competition",
        "uncertainty",
        "full_metrics",
    )
    filenames = {
        "selections": "family_fold_selections.csv",
        "results": "family_fold_results.csv",
        "predictions": "family_unseen_predictions.csv",
        "unseen_adjustments": "family_unseen_team_adjustments.csv",
        "competition": "family_competition_summary.csv",
        "uncertainty": "family_dependency_uncertainty.csv",
        "full_metrics": "family_full_candidate_metrics.csv",
    }
    for name in table_names:
        frames = []
        for output in outputs:
            frame = output[name].copy()
            frame.insert(0, "model_family", output["family"])
            frames.append(frame)
        pd.concat(frames, ignore_index=True).to_csv(
            output_root / filenames[name], index=False
        )


def family_comparison(outputs: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for output in outputs:
        overall = output["competition"].loc[
            output["competition"]["competition"].eq("ALL")
        ].iloc[0]
        selections = output["selections"]
        rows.append(
            {
                "model_family": output["family"],
                "decision": output["decision"],
                "selected_full_candidate": config_key(output["full_candidate"]),
                "active_folds": int((~selections["selected_is_baseline"]).sum()),
                "variance_active_folds": int(
                    selections["selected_variance_penalty"].gt(0).sum()
                ),
                "brier_difference": float(overall.brier_difference),
                "log_loss_difference": float(overall.log_loss_difference),
                **{
                    key: value
                    for key, value in output["guards"].items()
                    if key
                    in {
                        "ranking_no_regression_folds",
                        "pairwise_no_regression_folds",
                        "ranking_improvement_folds",
                        "pairwise_improvement_folds",
                        "brier_ci_upper_95",
                    }
                },
            }
        )
    return pd.DataFrame(rows)


def config_key(config: VarianceDomesticSurpriseConfig) -> str:
    return json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))


def write_report(
    path: Path,
    manifest: dict[str, object],
    coverage: pd.DataFrame,
    comparison: pd.DataFrame,
    variance_output: dict[str, object],
    adjustments: pd.DataFrame,
) -> None:
    unseen = variance_output["unseen_adjustments"]
    latest = unseen.loc[unseen["season"].eq("2025/26")]
    columns = [
        "team_name",
        "current_finish_score",
        "historical_mean",
        "historical_volatility",
        "consistency_multiplier",
        "raw_surprise",
        "effective_surprise",
        "ao_first_elo_adjustment",
    ]
    positive = latest.nlargest(12, "ao_first_elo_adjustment")[columns]
    negative = latest.nsmallest(12, "ao_first_elo_adjustment")[columns]
    lines = [
        "# Five-Season Variance-Controlled Domestic Surprise Backtest",
        "",
        f"Decision: **{manifest['decision']}**. Production was not changed.",
        "",
        "## Formula",
        "",
        "Every domestic finish is converted to direct league percentile. The same five",
        "completed seasons and 0.33/0.27/0.20/0.13/0.07 recency weights define both the",
        "historical mean and weighted variance. Volatility can only shrink the surprise",
        "signal; it cannot reverse its sign. Positive and negative adjustments are symmetric.",
        "The adjustment is blended with the existing linear European exposure contract.",
        "",
        "## Coverage",
        "",
        markdown_table(coverage),
        "",
        "## Family comparison",
        "",
        markdown_table(comparison),
        "",
        "## Variance-grid fold selections",
        "",
        markdown_table(variance_output["selections"]),
        "",
        "## Variance-grid unseen results",
        "",
        markdown_table(variance_output["results"]),
        "",
        "## Competition summary",
        "",
        markdown_table(variance_output["competition"]),
        "",
        "## Dependency-aware uncertainty",
        "",
        markdown_table(variance_output["uncertainty"]),
        "",
        "## 2025/26 largest positive effects",
        "",
        markdown_table(positive),
        "",
        "## 2025/26 largest negative effects",
        "",
        markdown_table(negative),
        "",
        "## Guardrails",
        "",
        "```json",
        json.dumps(manifest["guardrails"], indent=2),
        "```",
        "",
        "The full-data candidate is diagnostic only. Coefficient approval depends on unseen",
        "fold stability and ranking-first guardrails, not the pooled retrospective optimum.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
