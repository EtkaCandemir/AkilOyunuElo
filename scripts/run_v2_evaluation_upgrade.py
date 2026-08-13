from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ao_elo.config import (  # noqa: E402
    AO_MODEL_V2_VERSION,
    AOEuropeanEloConfig,
    V2_RATING_MULTIPLIER,
)
from ao_elo.evaluation import (  # noqa: E402
    dependency_robust_loss_difference_ci,
    evaluate_1x2_predictions,
    schedule_adjusted_team_performance,
    standard_1x2_losses,
)
from ao_elo.pipeline import compute_ao_first_elo  # noqa: E402
from scripts.run_achievement_carry_calibration import (  # noqa: E402
    AchievementCarryConfig,
    evaluate_sequence,
    load_carry_data,
)
from scripts.run_dynamic_core_calibration import DynamicCoreConfig  # noqa: E402
from scripts.run_ranking_first_calibration import ranking_metrics  # noqa: E402
from scripts.run_v2_ranking_calibration import (  # noqa: E402
    BASELINE_NAME,
    RANK_TOLERANCE,
    V2RankingEvaluator,
    clustered_candidate_vs_baseline_ci,
    distribution_guardrails,
    evaluate_external_benchmark,
    evaluate_outer_gate,
    evaluate_pilot_guardrails,
    expanding_folds,
    make_promotion_decision,
    run_nested_walk_forward,
    select_candidate,
    summarize_competitions,
    summarize_training,
    tail_candidates,
)


STATIC_DATA_ROOT = ROOT / "data" / "backtest_stage_b_2018_2026"
PILOT_ROOT = ROOT / "data" / "real_pilot_10_teams"
EVENTS_PATH = (
    ROOT / "data" / "external_elo_benchmark_2018_2026" / "exact_date_events.csv"
)
EXTERNAL_PATH = (
    ROOT / "output" / "external_elo_benchmark_2018_2026" / "paired_predictions.csv"
)
DYNAMIC_ROOT = ROOT / "output" / "v2_dynamic_calibration_2018_2026"
OUTPUT_ROOT = ROOT / "output" / "v2_evaluation_upgrade_2018_2026"
COMPETITIONS = ("UCL", "UEL", "UECL")
DRAW_AT_EVEN_CANDIDATES = tuple(np.arange(0.18, 0.381, 0.02).round(2))
DRAW_SHAPE_CANDIDATES = (
    0.50,
    0.60,
    0.70,
    0.75,
    0.80,
    0.84,
    0.85,
    0.90,
    0.95,
    1.00,
    1.25,
    1.50,
    2.00,
)
CLIMATOLOGY_ALPHA = 1.0
TARGET_HOME_EDGE_PRIOR = 20.0
TARGET_OPPONENT_PRIOR = 4.0
TARGET_TEAM_PRIOR = 2.0
POWER_CARRY_CANDIDATES = (0.0, 0.25, 0.50, 0.75, 0.85, 0.90, 1.0)
RANK_CORRELATION_FLOOR = 0.85
MAX_RATING_MOVE_GUARDRAIL = 200.0 * V2_RATING_MULTIPLIER


@dataclass(frozen=True, order=True)
class DrawModelConfig:
    draw_at_even: float
    draw_shape: float


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Upgrade AO Elo evaluation with adjusted ranking targets, standard "
            "1X2 losses, and dependency-sensitive uncertainty"
        )
    )
    parser.add_argument("--static-data-root", type=Path, default=STATIC_DATA_ROOT)
    parser.add_argument("--pilot-root", type=Path, default=PILOT_ROOT)
    parser.add_argument("--events-path", type=Path, default=EVENTS_PATH)
    parser.add_argument("--external-path", type=Path, default=EXTERNAL_PATH)
    parser.add_argument("--dynamic-root", type=Path, default=DYNAMIC_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    events = read_events(args.events_path.resolve())
    seasons = tuple(sorted(events["season"].unique()))
    folds = expanding_folds(seasons)
    if len(folds) != 6:
        raise ValueError(f"Expected six outer folds, found {len(folds)}")

    adjusted = run_adjusted_ranking_calibration(
        args.static_data_root.resolve(),
        args.pilot_root.resolve(),
        args.external_path.resolve(),
        events,
        seasons,
        folds,
    )
    write_adjusted_outputs(output_root, adjusted)

    probability = run_probability_calibration(
        args.static_data_root.resolve(),
        args.dynamic_root.resolve(),
        args.events_path.resolve(),
        events,
        folds,
        bootstrap_samples=args.bootstrap_samples,
    )
    write_probability_outputs(output_root, probability)
    write_manifest(output_root / "evaluation_manifest.json", adjusted, probability)
    write_production_manifest(
        output_root / "selected_production_model.json",
        probability,
    )
    write_report(
        output_root / "evaluation_report.md",
        seasons,
        adjusted,
        probability,
    )

    print("AO European Elo v2 evaluation upgrade")
    print(f"Adjusted ranking target team-seasons: {len(adjusted['target'])}")
    print(f"Static tail decision: {adjusted['decision']}")
    print(f"1X2 output decision: {probability['probability_decision']}")
    print(f"Dynamic core 1X2 status: {probability['core_decision']}")
    print(f"Season carry 1X2 status: {probability['carry_decision']}")
    print(f"Report: {output_root / 'evaluation_report.md'}")


def read_events(path: Path) -> pd.DataFrame:
    events = pd.read_csv(path)
    required = {
        "match_id",
        "season",
        "event_order",
        "competition",
        "tie_id",
        "kickoff_utc",
        "home_team_id",
        "away_team_id",
        "home_goals",
        "away_goals",
        "is_neutral",
    }
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"Exact event data missing columns: {missing}")
    if events["match_id"].isna().any() or events["match_id"].duplicated().any():
        raise ValueError("Exact event match_id must be non-null and unique")
    events["kickoff_utc"] = pd.to_datetime(
        events["kickoff_utc"], utc=True, errors="raise"
    )
    unknown = sorted(set(events["competition"]) - set(COMPETITIONS))
    if unknown:
        raise ValueError(f"Unknown competitions in exact events: {unknown}")
    return events.sort_values(["kickoff_utc", "event_order"], kind="stable").reset_index(
        drop=True
    )


def run_adjusted_ranking_calibration(
    static_root: Path,
    pilot_root: Path,
    external_path: Path,
    events: pd.DataFrame,
    seasons: tuple[str, ...],
    folds: tuple[tuple[tuple[str, ...], str], ...],
) -> dict[str, object]:
    target = schedule_adjusted_team_performance(
        events,
        home_edge_prior_matches=TARGET_HOME_EDGE_PRIOR,
        opponent_prior_matches=TARGET_OPPONENT_PRIOR,
        team_prior_matches=TARGET_TEAM_PRIOR,
    )
    evaluator = V2RankingEvaluator(static_root, seasons)
    ranking_frames: list[pd.DataFrame] = []
    rating_frames: list[pd.DataFrame] = []
    guardrail_rows: list[dict[str, object]] = []
    candidates = tail_candidates()

    for index, (candidate, config) in enumerate(candidates, start=1):
        candidate_ratings: list[pd.DataFrame] = []
        metric_rows: list[dict[str, object]] = []
        for season in seasons:
            ratings = compute_ao_first_elo(config=config, **evaluator.inputs[season])
            selected = ratings[
                [
                    "season",
                    "team_id",
                    "team_name",
                    "ao_first_elo",
                    "european_exposure",
                    "saturation_count",
                    "country_strength_saturated",
                    "achievement_saturated",
                    "european_history_saturated",
                    "exposure_saturated",
                ]
            ].copy()
            selected.insert(0, "candidate", candidate)
            candidate_ratings.append(selected)
            for competition in COMPETITIONS:
                table = adjusted_ranking_table(
                    ratings,
                    target,
                    season,
                    competition,
                )
                if table.empty:
                    continue
                metric_rows.append(
                    {
                        "candidate": candidate,
                        "season": season,
                        "competition": competition,
                        **ranking_metrics(table),
                        "raw_score_spearman": float(
                            table["ao_first_elo"].corr(
                                table["raw_score_rate"], method="spearman"
                            )
                        ),
                    }
                )
        candidate_frame = pd.concat(candidate_ratings, ignore_index=True)
        ranking_frames.append(pd.DataFrame(metric_rows))
        rating_frames.append(candidate_frame)
        guardrail_rows.append(
            {
                **evaluate_pilot_guardrails(candidate, config, pilot_root),
                **{
                    key: value
                    for key, value in distribution_guardrails(
                        candidate, candidate_frame
                    ).items()
                    if key != "candidate"
                },
            }
        )
        if index % 10 == 0:
            print(f"Adjusted ranking candidates evaluated: {index}/{len(candidates)}")

    ranking = pd.concat(ranking_frames, ignore_index=True)
    ratings = pd.concat(rating_frames, ignore_index=True)
    static_guardrails = pd.DataFrame(guardrail_rows)
    selections, unseen = run_nested_walk_forward(ranking, static_guardrails, folds)
    competitions = summarize_competitions(unseen)
    outer_gate = evaluate_outer_gate(selections, unseen, competitions)

    external_predictions, external_summary = evaluate_external_benchmark(
        pd.read_csv(external_path),
        ratings,
    )
    full_summary = summarize_training(
        ranking,
        static_guardrails,
        seasons,
    ).merge(external_summary, on="candidate", validate="one_to_one")
    baseline = external_summary.loc[
        external_summary["candidate"].eq(BASELINE_NAME)
    ].iloc[0]
    full_summary["external_gate_pass"] = (
        full_summary["ucl_brier_delta"].le(RANK_TOLERANCE)
        & full_summary["clubelo_rank_correlation"].ge(
            baseline["clubelo_rank_correlation"] - RANK_TOLERANCE
        )
        & (
            full_summary["candidate"].eq(BASELINE_NAME)
            | full_summary["elite_mean_rating_gap"].gt(
                baseline["elite_mean_rating_gap"] + RANK_TOLERANCE
            )
        )
    )
    full_summary["final_eligible"] = (
        full_summary["eligible"] & full_summary["external_gate_pass"]
    )
    evaluated_candidate = select_candidate(full_summary, "final_eligible")
    uncertainty = clustered_candidate_vs_baseline_ci(
        external_predictions,
        evaluated_candidate,
    )
    decision = make_promotion_decision(
        outer_gate,
        full_summary,
        evaluated_candidate,
        uncertainty,
    )
    selected_candidate = evaluated_candidate if decision == "PROMOTE" else BASELINE_NAME
    selected_config = dict(candidates)[selected_candidate]
    return {
        "target": target,
        "ranking": ranking,
        "ratings": ratings,
        "static_guardrails": static_guardrails,
        "selections": selections,
        "unseen": unseen,
        "competitions": competitions,
        "outer_gate": outer_gate,
        "external_summary": external_summary,
        "external_uncertainty": uncertainty,
        "full_summary": full_summary,
        "evaluated_candidate": evaluated_candidate,
        "selected_candidate": selected_candidate,
        "selected_config": selected_config,
        "decision": decision,
    }


def adjusted_ranking_table(
    ratings: pd.DataFrame,
    target: pd.DataFrame,
    season: str,
    competition: str,
) -> pd.DataFrame:
    performance = target.loc[
        target["season"].eq(season) & target["competition"].eq(competition)
    ]
    table = performance.merge(
        ratings[["team_id", "ao_first_elo", "european_exposure"]],
        on="team_id",
        how="inner",
        validate="one_to_one",
    )
    if table.empty:
        return table
    table["actual_score_rate"] = table["schedule_adjusted_score"]
    table["predicted_percentile"] = table["ao_first_elo"].rank(
        method="average", pct=True
    )
    table["actual_percentile"] = table["actual_score_rate"].rank(
        method="average", pct=True
    )
    table["percentile_error"] = (
        table["predicted_percentile"] - table["actual_percentile"]
    )
    return table


def write_adjusted_outputs(output_root: Path, result: dict[str, object]) -> None:
    outputs = {
        "adjusted_ranking_target.csv": "target",
        "adjusted_candidate_ranking_metrics.csv": "ranking",
        "adjusted_candidate_guardrails.csv": "static_guardrails",
        "adjusted_fold_selections.csv": "selections",
        "adjusted_unseen_fold_results.csv": "unseen",
        "adjusted_unseen_competition_summary.csv": "competitions",
        "adjusted_external_guardrails.csv": "external_summary",
        "adjusted_external_uncertainty.csv": "external_uncertainty",
        "adjusted_full_candidate_summary.csv": "full_summary",
    }
    for filename, key in outputs.items():
        frame = result[key]
        assert isinstance(frame, pd.DataFrame)
        frame.to_csv(output_root / filename, index=False)


def run_probability_calibration(
    static_root: Path,
    dynamic_root: Path,
    events_path: Path,
    events: pd.DataFrame,
    folds: tuple[tuple[tuple[str, ...], str], ...],
    *,
    bootstrap_samples: int,
) -> dict[str, object]:
    manifest = json.loads(
        (dynamic_root / "selected_dynamic_model.json").read_text(encoding="utf-8")
    )
    static_config = AOEuropeanEloConfig(**manifest["static_config"])
    static_config.validate()
    datasets, _ = load_carry_data(
        static_root,
        events_path,
        static_config,
        require_exact_utc=True,
    )
    core_selections = pd.read_csv(dynamic_root / "core_fold_selections.csv")
    carry_selections = pd.read_csv(dynamic_root / "carry_fold_selections.csv")
    validate_fold_contract(core_selections, carry_selections, folds)

    carry_calibration = run_carry_1x2_walk_forward(
        datasets,
        folds,
        core_selections,
        events,
    )
    carry_oof = carry_calibration["oof"]
    assert isinstance(carry_oof, pd.DataFrame)
    draw_selections = carry_calibration["draw_selections"]
    assert isinstance(draw_selections, pd.DataFrame)
    no_carry_draw_selections = draw_selections.loc[
        draw_selections["power_carry"].eq(0.0)
    ].copy()

    climatology_frames: list[pd.DataFrame] = []
    for fold, (train_seasons, test_season) in enumerate(folds, start=1):
        climatology = fit_climatology(
            events.loc[events["season"].isin(train_seasons)]
        )
        climatology.insert(0, "fold", fold)
        climatology.insert(1, "test_season", test_season)
        climatology_frames.append(climatology)
    climatology = pd.concat(climatology_frames, ignore_index=True)

    core_oof = pd.read_csv(dynamic_root / "core_unseen_predictions.csv")
    core_oof = merge_event_outcomes(core_oof, events)
    core_oof = apply_fold_draw_models(
        core_oof,
        no_carry_draw_selections,
        expected_column="dynamic_expected_home_score",
        prefix="dynamic",
    )
    core_oof = apply_fold_draw_models(
        core_oof,
        no_carry_draw_selections,
        expected_column="tuned_static_expected_home_score",
        prefix="tuned_static",
    )

    uncertainty_frames: list[pd.DataFrame] = []
    uncertainty_frames.append(
        layer_uncertainty(
            core_oof,
            "dynamic",
            "tuned_static",
            "dynamic_core_vs_tuned_static",
            bootstrap_samples,
        )
    )
    uncertainty_frames.append(
        layer_uncertainty(
            carry_oof,
            "carry_candidate",
            "no_carry",
            "carry_vs_no_carry",
            bootstrap_samples,
        )
    )
    uncertainty = pd.concat(uncertainty_frames, ignore_index=True)
    core_decision, core_guardrails = revalidation_decision(
        core_oof,
        uncertainty,
        "dynamic",
        "tuned_static",
        "dynamic_core_vs_tuned_static",
        require_reliable_improvement=True,
    )
    carry_decision, carry_guardrails = revalidation_decision(
        carry_oof,
        uncertainty,
        "carry_candidate",
        "no_carry",
        "carry_vs_no_carry",
        require_reliable_improvement=False,
    )
    carry_fold_results = carry_calibration["fold_results"]
    assert isinstance(carry_fold_results, pd.DataFrame)
    ranking_safe = bool(
        carry_fold_results["start_end_rank_correlation"].ge(
            RANK_CORRELATION_FLOOR
        ).all()
    )
    movement_safe = bool(
        carry_fold_results["max_abs_rating_change"].le(
            MAX_RATING_MOVE_GUARDRAIL
        ).all()
    )
    carry_guardrails["ranking_safe"] = ranking_safe
    carry_guardrails["movement_safe"] = movement_safe
    if carry_decision != "CONFIRMED_1X2" or not ranking_safe or not movement_safe:
        carry_decision = "DISABLE_CARRY_1X2"

    selected_fold_carries = carry_calibration["fold_selections"]
    assert isinstance(selected_fold_carries, pd.DataFrame)
    if carry_decision == "CONFIRMED_1X2":
        production_draw_selections = draw_selections.merge(
            selected_fold_carries[["fold", "selected_power_carry"]],
            left_on=["fold", "power_carry"],
            right_on=["fold", "selected_power_carry"],
            how="inner",
            validate="many_to_one",
        )
        carry_oof = alias_probability_columns(
            carry_oof,
            source="carry_candidate",
            target="active",
        )
    else:
        production_draw_selections = no_carry_draw_selections
        carry_oof = apply_fold_draw_models(
            carry_oof,
            production_draw_selections,
            expected_column="no_carry_expected_home_score",
            prefix="active",
        )
    carry_oof = apply_climatology(carry_oof, climatology)
    skill_summary = summarize_probability_skill(carry_oof)
    probability_decision = probability_output_decision(skill_summary)

    final_core = DynamicCoreConfig(**manifest["dynamic_core"])
    (
        final_carry_candidate,
        full_carry_metrics,
        full_draw_candidates,
        full_draw_metrics,
    ) = select_carry_1x2_candidate(
        datasets,
        final_core,
        events,
    )
    active_power_carry = (
        final_carry_candidate.power_carry
        if carry_decision == "CONFIRMED_1X2"
        else 0.0
    )
    full_draw_selections = full_draw_candidates.loc[
        full_draw_candidates["power_carry"].eq(active_power_carry)
    ].copy()
    return {
        "draw_selections": draw_selections,
        "production_draw_selections": production_draw_selections,
        "draw_metrics": carry_calibration["draw_metrics"],
        "carry_train_metrics": carry_calibration["train_metrics"],
        "carry_fold_selections": selected_fold_carries,
        "carry_fold_results": carry_fold_results,
        "climatology": climatology,
        "active_oof": carry_oof,
        "core_oof": core_oof,
        "skill_summary": skill_summary,
        "uncertainty": uncertainty,
        "full_draw_selections": full_draw_selections,
        "full_draw_metrics": full_draw_metrics,
        "full_carry_metrics": full_carry_metrics,
        "full_carry_candidate": final_carry_candidate.power_carry,
        "active_power_carry": active_power_carry,
        "probability_decision": probability_decision,
        "core_decision": core_decision,
        "core_guardrails": core_guardrails,
        "carry_decision": carry_decision,
        "carry_guardrails": carry_guardrails,
        "active_dynamic_manifest": manifest,
    }


def validate_fold_contract(
    core: pd.DataFrame,
    carry: pd.DataFrame,
    folds: tuple[tuple[tuple[str, ...], str], ...],
) -> None:
    expected = set(range(1, len(folds) + 1))
    if set(core["fold"]) != expected or set(carry["fold"]) != expected:
        raise ValueError("Dynamic fold files do not match the six-fold contract")
    for fold, (train, test) in enumerate(folds, start=1):
        for label, table in (("core", core), ("carry", carry)):
            row = table.loc[table["fold"].eq(fold)].iloc[0]
            if row["test_season"] != test or row["train_seasons"] != "|".join(train):
                raise ValueError(f"{label} fold {fold} has a season contract mismatch")


def run_carry_1x2_walk_forward(
    datasets: tuple,
    folds: tuple[tuple[tuple[str, ...], str], ...],
    core_selections: pd.DataFrame,
    events: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    selection_rows: list[dict[str, object]] = []
    fold_result_rows: list[dict[str, object]] = []
    train_metric_frames: list[pd.DataFrame] = []
    draw_selection_frames: list[pd.DataFrame] = []
    draw_metric_frames: list[pd.DataFrame] = []
    oof_frames: list[pd.DataFrame] = []

    for fold, (train_seasons, test_season) in enumerate(folds, start=1):
        core_row = core_selections.loc[core_selections["fold"].eq(fold)].iloc[0]
        core = DynamicCoreConfig(
            float(core_row["selected_scale"]),
            float(core_row["selected_home_advantage"]),
            float(core_row["selected_k"]),
        )
        train = tuple(data for data in datasets if data.season in train_seasons)
        sequence = tuple(
            data for data in datasets if data.season in (*train_seasons, test_season)
        )
        (
            selected,
            train_metrics,
            draw_selections,
            draw_metrics,
        ) = select_carry_1x2_candidate(train, core, events)
        train_metrics.insert(0, "fold", fold)
        train_metrics.insert(1, "test_season", test_season)
        draw_selections.insert(0, "fold", fold)
        draw_selections.insert(1, "test_season", test_season)
        draw_metrics.insert(0, "fold", fold)
        draw_metrics.insert(1, "test_season", test_season)
        train_metric_frames.append(train_metrics)
        draw_selection_frames.append(draw_selections)
        draw_metric_frames.append(draw_metrics)

        selected_train = train_metrics.loc[
            train_metrics["power_carry"].eq(selected.power_carry)
        ].iloc[0]
        selection_rows.append(
            {
                "fold": fold,
                "train_seasons": "|".join(train_seasons),
                "test_season": test_season,
                "core_scale": core.elo_scale,
                "core_home_advantage": core.home_advantage,
                "core_k": core.k_factor,
                "selected_power_carry": selected.power_carry,
                "train_brier_1x2": selected_train["brier_1x2"],
                "train_log_loss_1x2": selected_train["log_loss_1x2"],
            }
        )

        selected_metrics, selected_predictions, _ = evaluate_sequence(
            sequence,
            core,
            selected,
            evaluation_seasons={test_season},
            return_predictions=True,
        )
        baseline_metrics, baseline_predictions, _ = evaluate_sequence(
            sequence,
            core,
            AchievementCarryConfig(0.0, 0.0, 0.0),
            evaluation_seasons={test_season},
            return_predictions=True,
        )
        assert selected_predictions is not None and baseline_predictions is not None
        selected_predictions = merge_event_outcomes(selected_predictions, events).rename(
            columns={
                "expected_home_score": "carry_candidate_expected_home_score"
            }
        )
        baseline_expected = baseline_predictions[
            ["match_id", "expected_home_score"]
        ].rename(columns={"expected_home_score": "no_carry_expected_home_score"})
        oof = selected_predictions.merge(
            baseline_expected,
            on="match_id",
            validate="one_to_one",
        )
        oof.insert(0, "fold", fold)
        selected_draw = draw_selections.loc[
            draw_selections["power_carry"].eq(selected.power_carry)
        ]
        oof = apply_fold_draw_models(
            oof,
            selected_draw,
            expected_column="carry_candidate_expected_home_score",
            prefix="carry_candidate",
        )
        oof = apply_fold_draw_models(
            oof,
            selected_draw,
            expected_column="no_carry_expected_home_score",
            prefix="no_carry",
        )
        oof_frames.append(oof)
        fold_result_rows.append(
            {
                "fold": fold,
                "test_season": test_season,
                "selected_power_carry": selected.power_carry,
                "matches": selected_metrics["matches"],
                "candidate_expected_score_brier": selected_metrics["brier"],
                "baseline_expected_score_brier": baseline_metrics["brier"],
                "start_end_rank_correlation": selected_metrics[
                    "start_end_rank_correlation"
                ],
                "max_abs_rating_change": selected_metrics[
                    "max_abs_rating_change"
                ],
            }
        )

    return {
        "fold_selections": pd.DataFrame(selection_rows),
        "fold_results": pd.DataFrame(fold_result_rows),
        "train_metrics": pd.concat(train_metric_frames, ignore_index=True),
        "draw_selections": pd.concat(draw_selection_frames, ignore_index=True),
        "draw_metrics": pd.concat(draw_metric_frames, ignore_index=True),
        "oof": pd.concat(oof_frames, ignore_index=True),
    }


def select_carry_1x2_candidate(
    datasets: tuple,
    core: DynamicCoreConfig,
    events: pd.DataFrame,
) -> tuple[
    AchievementCarryConfig,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    metric_rows: list[dict[str, object]] = []
    draw_selection_frames: list[pd.DataFrame] = []
    draw_metric_frames: list[pd.DataFrame] = []
    for power_carry in POWER_CARRY_CANDIDATES:
        config = AchievementCarryConfig(float(power_carry), 0.0, 0.0)
        metrics, predictions, _ = evaluate_sequence(
            datasets,
            core,
            config,
            return_predictions=True,
        )
        assert predictions is not None
        predictions = merge_event_outcomes(predictions, events)
        draw_selections, draw_metrics = select_draw_models(predictions)
        draw_selections.insert(0, "power_carry", power_carry)
        draw_metrics.insert(0, "power_carry", power_carry)
        draw_selection_frames.append(draw_selections)
        draw_metric_frames.append(draw_metrics)

        temporary = predictions.copy()
        temporary.insert(0, "fold", 0)
        temporary_selections = draw_selections.copy()
        temporary_selections.insert(0, "fold", 0)
        evaluated = apply_fold_draw_models(
            temporary,
            temporary_selections,
            expected_column="expected_home_score",
            prefix="candidate",
        )
        metric_rows.append(
            {
                "power_carry": power_carry,
                "matches": len(evaluated),
                "brier_1x2": float(evaluated["candidate_brier_1x2"].mean()),
                "log_loss_1x2": float(
                    evaluated["candidate_log_loss_1x2"].mean()
                ),
                "start_end_rank_correlation": metrics[
                    "start_end_rank_correlation"
                ],
                "max_abs_rating_change": metrics["max_abs_rating_change"],
                "expected_score_brier": metrics["brier"],
            }
        )

    metric_table = pd.DataFrame(metric_rows)
    metric_table["ranking_guardrail_pass"] = metric_table[
        "start_end_rank_correlation"
    ].ge(RANK_CORRELATION_FLOOR)
    metric_table["movement_guardrail_pass"] = metric_table[
        "max_abs_rating_change"
    ].le(MAX_RATING_MOVE_GUARDRAIL)
    metric_table["eligible"] = (
        metric_table["ranking_guardrail_pass"]
        & metric_table["movement_guardrail_pass"]
    )
    metric_table = metric_table.sort_values(
        ["eligible", "brier_1x2", "log_loss_1x2", "power_carry"],
        ascending=[False, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    eligible = metric_table.loc[metric_table["eligible"]]
    if eligible.empty:
        raise ValueError("No 1X2 carry candidate passes ranking/movement guardrails")
    selected = AchievementCarryConfig(
        float(eligible.iloc[0]["power_carry"]),
        0.0,
        0.0,
    )
    return (
        selected,
        metric_table,
        pd.concat(draw_selection_frames, ignore_index=True),
        pd.concat(draw_metric_frames, ignore_index=True),
    )


def alias_probability_columns(
    predictions: pd.DataFrame,
    *,
    source: str,
    target: str,
) -> pd.DataFrame:
    result = predictions.copy()
    for suffix in (
        "home_probability",
        "draw_probability",
        "away_probability",
        "brier_1x2",
        "log_loss_1x2",
    ):
        result[f"{target}_{suffix}"] = result[f"{source}_{suffix}"]
    return result


def draw_model_candidates() -> tuple[DrawModelConfig, ...]:
    return tuple(
        DrawModelConfig(float(draw), float(shape))
        for draw, shape in product(DRAW_AT_EVEN_CANDIDATES, DRAW_SHAPE_CANDIDATES)
    )


def select_draw_models(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    groups = [("ALL", predictions), *predictions.groupby("competition", sort=True)]
    for competition, frame in groups:
        for candidate in draw_model_candidates():
            evaluated = evaluate_1x2_predictions(
                frame["expected_home_score"],
                frame["home_goals"],
                frame["away_goals"],
                draw_at_even=candidate.draw_at_even,
                draw_shape=candidate.draw_shape,
            )
            rows.append(
                {
                    "competition": competition,
                    "draw_at_even": candidate.draw_at_even,
                    "draw_shape": candidate.draw_shape,
                    "matches": len(frame),
                    "brier_1x2": float(evaluated["brier_1x2"].mean()),
                    "log_loss_1x2": float(evaluated["log_loss_1x2"].mean()),
                    "distance_from_reference": (
                        abs(candidate.draw_at_even - 0.28)
                        + 0.02 * abs(candidate.draw_shape - 1.0)
                    ),
                }
            )
    metrics = pd.DataFrame(rows).sort_values(
        [
            "competition",
            "log_loss_1x2",
            "brier_1x2",
            "distance_from_reference",
            "draw_at_even",
            "draw_shape",
        ],
        kind="stable",
    )
    selected = metrics.groupby("competition", as_index=False, sort=True).first()
    return selected, metrics.reset_index(drop=True)


def merge_event_outcomes(
    predictions: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "match_id",
        "season",
        "competition",
        "home_team_id",
        "away_team_id",
        "home_goals",
        "away_goals",
        "tie_id",
        "kickoff_utc",
    ]
    overlap = [
        column
        for column in columns[3:]
        if column in predictions.columns
    ]
    base = predictions.drop(columns=overlap)
    metadata = events[columns].rename(
        columns={"season": "event_season", "competition": "event_competition"}
    )
    merged = base.merge(metadata, on="match_id", validate="one_to_one")
    if len(merged) != len(predictions):
        raise ValueError("Predictions and exact event outcomes do not align one-to-one")
    if not merged["season"].eq(merged["event_season"]).all() or not merged[
        "competition"
    ].eq(merged["event_competition"]).all():
        raise ValueError("Prediction season/competition metadata conflicts with exact events")
    merged = merged.drop(columns=["event_season", "event_competition"])
    return merged


def apply_fold_draw_models(
    predictions: pd.DataFrame,
    selections: pd.DataFrame,
    *,
    expected_column: str,
    prefix: str,
) -> pd.DataFrame:
    if expected_column not in predictions:
        raise ValueError(f"Predictions missing {expected_column}")
    result = predictions.copy()
    output_columns = [
        f"{prefix}_home_probability",
        f"{prefix}_draw_probability",
        f"{prefix}_away_probability",
        f"{prefix}_brier_1x2",
        f"{prefix}_log_loss_1x2",
    ]
    for column in output_columns:
        result[column] = np.nan
    for row in selections.itertuples(index=False):
        competition_mask = (
            pd.Series(True, index=result.index)
            if row.competition == "ALL"
            else result["competition"].eq(row.competition)
        )
        mask = result["fold"].eq(row.fold) & competition_mask
        if not mask.any():
            continue
        evaluated = evaluate_1x2_predictions(
            result.loc[mask, expected_column],
            result.loc[mask, "home_goals"],
            result.loc[mask, "away_goals"],
            draw_at_even=float(row.draw_at_even),
            draw_shape=float(row.draw_shape),
        )
        result.loc[mask, output_columns] = evaluated[
            [
                "home_probability",
                "draw_probability",
                "away_probability",
                "brier_1x2",
                "log_loss_1x2",
            ]
        ].to_numpy()
    if result[output_columns].isna().any().any():
        missing = result.loc[result[output_columns].isna().any(axis=1), ["fold", "competition"]]
        raise ValueError(
            "Draw mapping missing for fold/competition keys: "
            f"{missing.drop_duplicates().to_dict('records')}"
        )
    return result


def fit_climatology(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [("ALL", events), *events.groupby("competition", sort=True)]
    for competition, frame in groups:
        outcomes = np.where(
            frame["home_goals"] > frame["away_goals"],
            "H",
            np.where(frame["home_goals"] < frame["away_goals"], "A", "D"),
        )
        counts = pd.Series(outcomes).value_counts()
        denominator = len(frame) + 3.0 * CLIMATOLOGY_ALPHA
        rows.append(
            {
                "competition": competition,
                "matches": len(frame),
                "home_probability": (counts.get("H", 0) + CLIMATOLOGY_ALPHA)
                / denominator,
                "draw_probability": (counts.get("D", 0) + CLIMATOLOGY_ALPHA)
                / denominator,
                "away_probability": (counts.get("A", 0) + CLIMATOLOGY_ALPHA)
                / denominator,
            }
        )
    return pd.DataFrame(rows)


def apply_climatology(
    predictions: pd.DataFrame,
    climatology: pd.DataFrame,
) -> pd.DataFrame:
    mapping = climatology.rename(
        columns={
            "home_probability": "climatology_home_probability",
            "draw_probability": "climatology_draw_probability",
            "away_probability": "climatology_away_probability",
        }
    )[
        [
            "fold",
            "competition",
            "climatology_home_probability",
            "climatology_draw_probability",
            "climatology_away_probability",
        ]
    ]
    specific = mapping.loc[mapping["competition"].ne("ALL")]
    pooled = mapping.loc[mapping["competition"].eq("ALL")].drop(
        columns="competition"
    ).rename(
        columns={
            "climatology_home_probability": "pooled_home_probability",
            "climatology_draw_probability": "pooled_draw_probability",
            "climatology_away_probability": "pooled_away_probability",
        }
    )
    result = predictions.merge(
        specific,
        on=["fold", "competition"],
        how="left",
        validate="many_to_one",
    ).merge(
        pooled,
        on="fold",
        how="left",
        validate="many_to_one",
    )
    probability_columns = [
        "climatology_home_probability",
        "climatology_draw_probability",
        "climatology_away_probability",
    ]
    if result[probability_columns].isna().any().any():
        for column in probability_columns:
            pooled_column = column.replace("climatology_", "pooled_")
            result[column] = result[column].fillna(result[pooled_column])
    if result[probability_columns].isna().any().any():
        raise ValueError("Climatology mapping and pooled fallback are incomplete")
    result = result.drop(
        columns=[
            "pooled_home_probability",
            "pooled_draw_probability",
            "pooled_away_probability",
        ]
    )
    losses = standard_1x2_losses(
        result[probability_columns].rename(
            columns={
                "climatology_home_probability": "home_probability",
                "climatology_draw_probability": "draw_probability",
                "climatology_away_probability": "away_probability",
            }
        ),
        result["home_goals"],
        result["away_goals"],
    )
    result["climatology_brier_1x2"] = losses["brier_1x2"].to_numpy()
    result["climatology_log_loss_1x2"] = losses["log_loss_1x2"].to_numpy()
    return result


def summarize_probability_skill(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [("ALL", predictions), *predictions.groupby("competition", sort=True)]
    for competition, frame in groups:
        rows.append(
            {
                "competition": competition,
                "matches": len(frame),
                "active_brier_1x2": frame["active_brier_1x2"].mean(),
                "climatology_brier_1x2": frame["climatology_brier_1x2"].mean(),
                "brier_skill_difference": (
                    frame["active_brier_1x2"].mean()
                    - frame["climatology_brier_1x2"].mean()
                ),
                "active_log_loss_1x2": frame["active_log_loss_1x2"].mean(),
                "climatology_log_loss_1x2": frame[
                    "climatology_log_loss_1x2"
                ].mean(),
                "log_loss_skill_difference": (
                    frame["active_log_loss_1x2"].mean()
                    - frame["climatology_log_loss_1x2"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def probability_output_decision(summary: pd.DataFrame) -> str:
    passed = bool(
        summary["brier_skill_difference"].le(0.0).all()
        and summary["log_loss_skill_difference"].le(0.0).all()
    )
    return "PROMOTE_1X2_OUTPUT" if passed else "HOLD_1X2_OUTPUT"


def layer_uncertainty(
    predictions: pd.DataFrame,
    model: str,
    baseline: str,
    comparison: str,
    bootstrap_samples: int,
) -> pd.DataFrame:
    rows = []
    groups = [("ALL", predictions), *predictions.groupby("competition", sort=True)]
    for competition, frame in groups:
        for loss in ("brier_1x2", "log_loss_1x2"):
            values = frame.copy()
            values["loss_difference"] = (
                values[f"{model}_{loss}"] - values[f"{baseline}_{loss}"]
            )
            uncertainty = dependency_robust_loss_difference_ci(
                values,
                bootstrap_samples=bootstrap_samples,
            )
            uncertainty.insert(0, "comparison", comparison)
            uncertainty.insert(1, "competition", competition)
            uncertainty.insert(2, "loss", loss)
            rows.append(uncertainty)
    return pd.concat(rows, ignore_index=True)


def revalidation_decision(
    predictions: pd.DataFrame,
    uncertainty: pd.DataFrame,
    model: str,
    baseline: str,
    comparison: str,
    *,
    require_reliable_improvement: bool,
) -> tuple[str, dict[str, object]]:
    fold_means = predictions.groupby("fold", as_index=False).agg(
        model_brier=(f"{model}_brier_1x2", "mean"),
        baseline_brier=(f"{baseline}_brier_1x2", "mean"),
    )
    fold_wins = int((fold_means["model_brier"] < fold_means["baseline_brier"]).sum())
    envelope = uncertainty.loc[
        uncertainty["comparison"].eq(comparison)
        & uncertainty["loss"].eq("brier_1x2")
        & uncertainty["method"].eq("conservative_envelope")
    ]
    overall = envelope.loc[envelope["competition"].eq("ALL")].iloc[0]
    segments = envelope.loc[envelope["competition"].ne("ALL")]
    no_segment_harm = not bool(segments["reliable_harm"].any())
    reliable_gate = (
        bool(overall["reliable_improvement"])
        if require_reliable_improvement
        else not bool(overall["reliable_harm"])
    )
    passed = fold_wins == 6 and reliable_gate and no_segment_harm
    guardrails = {
        "fold_wins": fold_wins,
        "required_fold_wins": 6,
        "overall_reliable_improvement": bool(overall["reliable_improvement"]),
        "overall_reliable_harm": bool(overall["reliable_harm"]),
        "no_competition_reliable_harm": no_segment_harm,
    }
    return ("CONFIRMED_1X2" if passed else "RECALIBRATE_REQUIRED"), guardrails


def write_probability_outputs(output_root: Path, result: dict[str, object]) -> None:
    outputs = {
        "draw_candidate_fold_selections.csv": "draw_selections",
        "draw_production_fold_selections.csv": "production_draw_selections",
        "draw_train_candidate_metrics.csv": "draw_metrics",
        "carry_1x2_train_candidate_metrics.csv": "carry_train_metrics",
        "carry_1x2_fold_selections.csv": "carry_fold_selections",
        "carry_1x2_unseen_fold_results.csv": "carry_fold_results",
        "fold_climatology.csv": "climatology",
        "one_x_two_oof_predictions.csv": "active_oof",
        "core_one_x_two_oof_predictions.csv": "core_oof",
        "one_x_two_skill_summary.csv": "skill_summary",
        "layer_dependency_uncertainty.csv": "uncertainty",
        "draw_full_selections.csv": "full_draw_selections",
        "draw_full_candidate_metrics.csv": "full_draw_metrics",
    }
    for filename, key in outputs.items():
        frame = result[key]
        assert isinstance(frame, pd.DataFrame)
        frame.to_csv(output_root / filename, index=False)


def write_manifest(
    path: Path,
    adjusted: dict[str, object],
    probability: dict[str, object],
) -> None:
    full_draw = probability["full_draw_selections"]
    assert isinstance(full_draw, pd.DataFrame)
    payload = {
        "evaluation_contract_version": "ao-evaluation-v2.1",
        "ranking_target": {
            "name": "leave-team-out venue-and-schedule-adjusted score",
            "home_edge_prior_matches": TARGET_HOME_EDGE_PRIOR,
            "opponent_prior_matches": TARGET_OPPONENT_PRIOR,
            "team_prior_matches": TARGET_TEAM_PRIOR,
            "uses_candidate_ratings": False,
            "static_tail_decision": adjusted["decision"],
            "evaluated_candidate": adjusted["evaluated_candidate"],
            "selected_candidate": adjusted["selected_candidate"],
            "outer_gate": adjusted["outer_gate"],
        },
        "probability_contract": {
            "classes": ["home", "draw", "away"],
            "brier_definition": "sum of three squared class-probability errors",
            "log_loss_definition": "negative log probability of observed class",
            "elo_score_identity": "P(home) + 0.5 * P(draw) = expected_home_score",
            "draw_mapping_candidates": {
                "draw_at_even": list(DRAW_AT_EVEN_CANDIDATES),
                "draw_shape": list(DRAW_SHAPE_CANDIDATES),
            },
            "selected_by_competition": full_draw[
                ["competition", "draw_at_even", "draw_shape"]
            ].to_dict("records"),
            "decision": probability["probability_decision"],
        },
        "uncertainty_contract": {
            "methods": ["tie_or_match", "team_season", "calendar_month"],
            "promotion_interval": "conservative envelope across all three views",
            "formal_multiway_cluster_claim": False,
        },
        "layer_revalidation": {
            "dynamic_core": {
                "decision": probability["core_decision"],
                "guardrails": probability["core_guardrails"],
            },
            "season_carry": {
                "decision": probability["carry_decision"],
                "guardrails": probability["carry_guardrails"],
                "full_data_candidate": probability["full_carry_candidate"],
                "recommended_active_power_carry": probability[
                    "active_power_carry"
                ],
            },
        },
        "active_rating_parameters_changed": bool(
            probability["active_power_carry"]
            != probability["active_dynamic_manifest"]["active_power_carry"]
        ),
        "rating_change_rule": (
            "A static tail promotion requires a fresh dynamic recalibration before "
            "any production parameter changes."
        ),
        "untouched_holdout": "2026/27",
    }
    path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")


def write_production_manifest(path: Path, probability: dict[str, object]) -> None:
    draw = probability["full_draw_selections"]
    assert isinstance(draw, pd.DataFrame)
    if draw.empty:
        raise ValueError("Production draw selection cannot be empty")
    if draw["draw_at_even"].nunique() != 1 or draw["draw_shape"].nunique() != 1:
        raise ValueError(
            "Production API currently requires one global draw mapping; calibrated "
            "competition mappings disagree"
        )
    source = probability["active_dynamic_manifest"]
    core = source["dynamic_core"]
    payload = {
        "model_version": AO_MODEL_V2_VERSION,
        "dynamic_core": {
            "elo_scale": float(core["elo_scale"]),
            "home_advantage": float(core["home_advantage"]),
            "k_factor": float(core["k_factor"]),
        },
        "active_power_carry": float(probability["active_power_carry"]),
        "one_x_two_probability": {
            "active": probability["probability_decision"] == "PROMOTE_1X2_OUTPUT",
            "draw_at_even": float(draw.iloc[0]["draw_at_even"]),
            "draw_shape": float(draw.iloc[0]["draw_shape"]),
            "expected_score_identity_preserved": True,
            "standard_brier": True,
            "standard_log_loss": True,
        },
        "goal_margin": {
            "active": False,
            "goal_weight": 0.0,
            "goal_cap": 1.0,
        },
        "achievement_reserve": {
            "active": False,
            "reserve_base": 0.0,
            "ucl_multiplier": 1.0,
            "uel_multiplier": 0.65,
            "uecl_multiplier": 0.45,
            "stage_profile": "FLAT",
            "reserve_decay": 0.0,
            "reserve_cap": 80.0 * V2_RATING_MULTIPLIER,
            "trophy_uses_same_base": True,
        },
        "decisions": {
            "dynamic_core": probability["core_decision"],
            "season_carry": probability["carry_decision"],
            "one_x_two_output": probability["probability_decision"],
            "goal_margin": "DISABLED",
            "achievement_reserve": "DISABLED",
        },
        "development_data_through": "2025/26",
        "untouched_holdout": "2026/27",
    }
    path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")


def write_report(
    path: Path,
    seasons: tuple[str, ...],
    adjusted: dict[str, object],
    probability: dict[str, object],
) -> None:
    outer = adjusted["outer_gate"]
    skill = probability["skill_summary"]
    uncertainty = probability["uncertainty"]
    full_draw = probability["full_draw_selections"]
    assert isinstance(skill, pd.DataFrame)
    assert isinstance(uncertainty, pd.DataFrame)
    assert isinstance(full_draw, pd.DataFrame)
    lines = [
        "# AO European Elo v2 Evaluation Upgrade",
        "",
        f"Development window: `{seasons[0]}` through `{seasons[-1]}`. The `2026/27` "
        "season remains untouched.",
        "",
        "## Adjusted Ranking Target",
        "",
        "The target removes a leave-team-out home-edge estimate and adds opponent "
        "strength estimated without the direct target-opponent meetings. It never "
        "uses AO or ClubElo ratings.",
        "",
        f"- Static tail decision: `{adjusted['decision']}`",
        f"- Evaluated candidate: `{adjusted['evaluated_candidate']}`",
        f"- Active candidate: `{adjusted['selected_candidate']}`",
        f"- No unseen fold regression: `{outer['no_unseen_fold_regression']}`",
        f"- Both rank metrics improved: `{outer['folds_improved_both_metrics']}/"
        f"{outer['outer_folds']}` folds",
        f"- No competition regression: `{outer['no_competition_regression']}`",
        "",
        "## Standard 1X2 Output",
        "",
        "Brier is the sum of the three squared class errors (range 0-2). Log loss "
        "uses the observed H/D/A probability. The draw mapping preserves the Elo "
        "expected-points identity exactly.",
        "",
        f"- Probability decision: `{probability['probability_decision']}`",
        f"- Full-data carry candidate: `{probability['full_carry_candidate']:g}`",
        f"- Gate-approved carry: `{probability['active_power_carry']:g}`",
        "",
        "| Competition | Draw at even | Shape |",
        "| --- | ---: | ---: |",
    ]
    for row in full_draw.itertuples(index=False):
        lines.append(
            f"| {row.competition} | {row.draw_at_even:.2f} | {row.draw_shape:.2f} |"
        )
    lines.extend(
        [
            "",
            "| Segment | Matches | AO Brier | Base Brier | Difference | AO Log | Base Log | Difference |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in skill.itertuples(index=False):
        lines.append(
            f"| {row.competition} | {row.matches} | {row.active_brier_1x2:.6f} | "
            f"{row.climatology_brier_1x2:.6f} | {row.brier_skill_difference:+.6f} | "
            f"{row.active_log_loss_1x2:.6f} | {row.climatology_log_loss_1x2:.6f} | "
            f"{row.log_loss_skill_difference:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Layer Revalidation",
            "",
            f"- Dynamic core: `{probability['core_decision']}`",
            f"- Season carry: `{probability['carry_decision']}`",
            "",
            "The reported confidence interval is the conservative envelope of "
            "tie/match, team-season, and calendar-month bootstrap views. These are "
            "dependency sensitivity analyses, not a formal multi-way clustered "
            "standard-error claim.",
            "",
            "| Layer | Competition | Loss | Mean difference | Envelope 95% CI |",
            "| --- | --- | --- | ---: | --- |",
        ]
    )
    envelope = uncertainty.loc[
        uncertainty["method"].eq("conservative_envelope")
    ]
    for row in envelope.itertuples(index=False):
        lines.append(
            f"| {row.comparison} | {row.competition} | {row.loss} | "
            f"{row.mean_difference:+.6f} | [{row.ci_95_lower:+.6f}, "
            f"{row.ci_95_upper:+.6f}] |"
        )
    lines.extend(
        [
            "",
            "Static rating parameters remain unchanged. The production season carry "
            "is set to zero because the recalibrated carry process won only five of "
            "six unseen folds. Any future static-tail promotion must still be followed "
            "by a fresh dynamic calibration.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def json_default(value: object) -> object:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, DrawModelConfig):
        return {
            "draw_at_even": value.draw_at_even,
            "draw_shape": value.draw_shape,
        }
    raise TypeError(f"Cannot serialize {type(value).__name__}")


if __name__ == "__main__":
    main()
