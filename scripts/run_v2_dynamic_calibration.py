from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
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
from scripts.run_achievement_carry_calibration import (  # noqa: E402
    AchievementCarryConfig,
    evaluate_sequence,
    load_carry_data,
)
from scripts.run_dynamic_core_calibration import (  # noqa: E402
    ELO_SCALES,
    HOME_ADVANTAGES,
    K_FACTORS,
    DynamicCoreConfig,
    candidate_metrics,
    evaluate_seasons,
    expanding_folds,
    load_calibration_data,
)


STATIC_DATA_ROOT = ROOT / "data" / "backtest_stage_b_2018_2026"
EXACT_EVENTS_PATH = (
    ROOT / "data" / "external_elo_benchmark_2018_2026" / "exact_date_events.csv"
)
STATIC_MANIFEST_PATH = (
    ROOT / "output" / "v2_ranking_calibration_2018_2026" / "selected_model.json"
)
OUTPUT_ROOT = ROOT / "output" / "v2_dynamic_calibration_2018_2026"
POWER_CARRY_CANDIDATES = (0.0, 0.25, 0.50, 0.75, 0.85, 0.90, 1.0)
RANK_CORRELATION_FLOOR = 0.85
MAX_RATING_MOVE_GUARDRAIL = 200.0 * V2_RATING_MULTIPLIER
CORE_BASELINE_NAME = "tuned_static"
CORE_MODEL_NAME = "dynamic"
CARRY_BASELINE_NAME = "no_carry"
CARRY_MODEL_NAME = "carry"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate exact-date AO Elo v2 dynamic core and season carry"
    )
    parser.add_argument("--static-data-root", type=Path, default=STATIC_DATA_ROOT)
    parser.add_argument("--events-path", type=Path, default=EXACT_EVENTS_PATH)
    parser.add_argument("--static-manifest", type=Path, default=STATIC_MANIFEST_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    static_config = read_static_config(args.static_manifest.resolve())
    events_path = args.events_path.resolve()
    events = read_event_metadata(events_path)
    datasets = load_calibration_data(
        args.static_data_root.resolve(),
        events_path,
        static_config,
        require_exact_utc=True,
    )
    seasons = tuple(data.season for data in datasets)
    folds = expanding_folds(seasons)
    if len(folds) != 6:
        raise ValueError(f"Expected six dynamic outer folds, found {len(folds)}")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    core_candidates = scaled_core_grid()
    (
        core_selections,
        core_fold_results,
        core_predictions,
    ) = run_core_walk_forward(datasets, folds, core_candidates, events)
    core_competitions = summarize_loss_differences(
        core_predictions,
        CORE_MODEL_NAME,
        CORE_BASELINE_NAME,
    )
    core_uncertainty = clustered_uncertainty(
        core_predictions,
        CORE_MODEL_NAME,
        CORE_BASELINE_NAME,
    )
    final_core, final_core_metrics, full_core_metrics = select_guarded_core(
        datasets,
        core_candidates,
    )
    core_decision, core_guardrails = core_promotion_decision(
        core_fold_results,
        core_uncertainty,
    )
    if core_decision != "PROMOTE_DYNAMIC_CORE":
        raise RuntimeError(
            "V2 dynamic core failed mandatory promotion gates; production integration "
            "must remain on the static model"
        )

    carry_datasets, identity_audit = load_carry_data(
        args.static_data_root.resolve(),
        events_path,
        static_config,
        require_exact_utc=True,
    )
    (
        carry_selections,
        carry_fold_results,
        carry_predictions,
    ) = run_carry_walk_forward(
        carry_datasets,
        folds,
        core_selections,
        events,
    )
    carry_competitions = summarize_loss_differences(
        carry_predictions,
        CARRY_MODEL_NAME,
        CARRY_BASELINE_NAME,
    )
    carry_uncertainty = clustered_uncertainty(
        carry_predictions,
        CARRY_MODEL_NAME,
        CARRY_BASELINE_NAME,
    )
    final_carry, full_carry_metrics = select_carry_candidate(
        carry_datasets,
        final_core,
    )
    carry_decision, carry_guardrails = carry_promotion_decision(
        carry_fold_results,
        carry_uncertainty,
    )
    active_carry = final_carry.power_carry if carry_decision == "PROMOTE_CARRY" else 0.0

    core_selections.to_csv(output_root / "core_fold_selections.csv", index=False)
    core_fold_results.to_csv(output_root / "core_fold_results.csv", index=False)
    core_predictions.to_csv(output_root / "core_unseen_predictions.csv", index=False)
    core_competitions.to_csv(output_root / "core_competition_summary.csv", index=False)
    core_uncertainty.to_csv(output_root / "core_clustered_uncertainty.csv", index=False)
    full_core_metrics.to_csv(output_root / "core_full_candidate_metrics.csv", index=False)
    carry_selections.to_csv(output_root / "carry_fold_selections.csv", index=False)
    carry_fold_results.to_csv(output_root / "carry_fold_results.csv", index=False)
    carry_predictions.to_csv(output_root / "carry_unseen_predictions.csv", index=False)
    carry_competitions.to_csv(
        output_root / "carry_competition_summary.csv", index=False
    )
    carry_uncertainty.to_csv(
        output_root / "carry_clustered_uncertainty.csv", index=False
    )
    full_carry_metrics.to_csv(output_root / "carry_full_candidate_metrics.csv", index=False)
    identity_audit.to_csv(output_root / "club_identity_audit.csv", index=False)
    write_manifest(
        output_root / "selected_dynamic_model.json",
        static_config,
        final_core,
        final_core_metrics,
        core_decision,
        core_guardrails,
        final_carry,
        active_carry,
        carry_decision,
        carry_guardrails,
    )
    write_report(
        output_root / "calibration_report.md",
        seasons,
        core_selections,
        core_competitions,
        core_uncertainty,
        final_core,
        core_decision,
        carry_selections,
        carry_competitions,
        carry_uncertainty,
        final_carry,
        active_carry,
        carry_decision,
    )

    print("AO European Elo v2 exact-date dynamic calibration")
    print(f"Matches: {sum(len(data.match_ids) for data in datasets)}")
    print(f"Outer folds: {len(folds)}")
    print(
        f"Core: Scale={final_core.elo_scale:.6f}, "
        f"H={final_core.home_advantage:.6f}, K={final_core.k_factor:.6f}"
    )
    print(f"Core decision: {core_decision}")
    print(f"Carry candidate: {final_carry.power_carry:g}")
    print(f"Carry decision: {carry_decision}; active={active_carry:g}")
    print(f"Report: {output_root / 'calibration_report.md'}")


def read_static_config(path: Path) -> AOEuropeanEloConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("selected_candidate") is None or "config" not in payload:
        raise ValueError("Static v2 manifest lacks selected candidate/config")
    config = AOEuropeanEloConfig(**payload["config"])
    config.validate()
    if config.model_version != AO_MODEL_V2_VERSION:
        raise ValueError("Dynamic v2 calibration requires a v2 static config")
    return config


def read_event_metadata(path: Path) -> pd.DataFrame:
    events = pd.read_csv(path)
    required = {"match_id", "season", "tie_id", "kickoff_utc"}
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"Exact event metadata missing columns: {missing}")
    if events["match_id"].duplicated().any():
        raise ValueError("Exact event match_id must be unique")
    events["kickoff_utc"] = pd.to_datetime(events["kickoff_utc"], utc=True, errors="raise")
    return events[["match_id", "season", "tie_id", "kickoff_utc"]]


def scaled_core_grid() -> tuple[DynamicCoreConfig, ...]:
    return tuple(
        DynamicCoreConfig(
            float(scale) * V2_RATING_MULTIPLIER,
            float(home_advantage) * V2_RATING_MULTIPLIER,
            float(k_factor) * V2_RATING_MULTIPLIER,
        )
        for scale in ELO_SCALES
        for home_advantage in HOME_ADVANTAGES
        for k_factor in K_FACTORS
    )


def select_guarded_core(
    datasets: tuple,
    candidates: tuple[DynamicCoreConfig, ...],
) -> tuple[DynamicCoreConfig, dict[str, float | int], pd.DataFrame]:
    metrics = candidate_metrics(datasets, candidates)
    eligible = metrics.loc[
        metrics["start_end_rank_correlation"].ge(RANK_CORRELATION_FLOOR)
        & metrics["max_abs_rating_change"].le(MAX_RATING_MOVE_GUARDRAIL)
        & metrics["mean_rating_change"].abs().le(1e-9)
    ]
    if eligible.empty:
        raise ValueError("No dynamic core candidate passes ranking/movement guardrails")
    selected = eligible.iloc[0]
    config = DynamicCoreConfig(
        float(selected["elo_scale"]),
        float(selected["home_advantage"]),
        float(selected["k_factor"]),
    )
    metric_values = {
        key: selected[key]
        for key in (
            "matches",
            "brier",
            "log_loss",
            "mean_rating_change",
            "rating_change_std",
            "max_abs_rating_change",
            "start_end_rank_correlation",
        )
    }
    return config, metric_values, metrics


def run_core_walk_forward(
    datasets: tuple,
    folds: tuple[tuple[tuple[str, ...], str], ...],
    candidates: tuple[DynamicCoreConfig, ...],
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selections: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    static_candidates = tuple(candidate for candidate in candidates if candidate.k_factor == 0)
    for fold, (train_seasons, test_season) in enumerate(folds, start=1):
        train = tuple(data for data in datasets if data.season in train_seasons)
        test = next(data for data in datasets if data.season == test_season)
        dynamic, train_metrics, _ = select_guarded_core(train, candidates)
        static, static_train, _ = select_guarded_core(train, static_candidates)
        selections.append(
            {
                "fold": fold,
                "train_seasons": "|".join(train_seasons),
                "test_season": test_season,
                "selected_scale": dynamic.elo_scale,
                "selected_home_advantage": dynamic.home_advantage,
                "selected_k": dynamic.k_factor,
                "train_brier": train_metrics["brier"],
                "static_scale": static.elo_scale,
                "static_home_advantage": static.home_advantage,
                "static_train_brier": static_train["brier"],
            }
        )
        models: dict[str, pd.DataFrame] = {}
        for model, config in ((CORE_MODEL_NAME, dynamic), (CORE_BASELINE_NAME, static)):
            metrics, predictions = evaluate_seasons(
                (test,), config, return_predictions=True
            )
            result_rows.append(
                {
                    "fold": fold,
                    "test_season": test_season,
                    "model": model,
                    "elo_scale": config.elo_scale,
                    "home_advantage": config.home_advantage,
                    "k_factor": config.k_factor,
                    **metrics,
                }
            )
            assert predictions is not None
            models[model] = prefix_losses(predictions, model)
        joined = models[CORE_MODEL_NAME].merge(
            models[CORE_BASELINE_NAME][
                [
                    "match_id",
                    f"{CORE_BASELINE_NAME}_expected_home_score",
                    f"{CORE_BASELINE_NAME}_brier_loss",
                    f"{CORE_BASELINE_NAME}_log_loss",
                ]
            ],
            on="match_id",
            validate="one_to_one",
        )
        joined.insert(0, "fold", fold)
        prediction_frames.append(add_event_clusters(joined, events))
        print(f"Core outer fold complete: {fold}/{len(folds)}")
    return (
        pd.DataFrame(selections),
        pd.DataFrame(result_rows),
        pd.concat(prediction_frames, ignore_index=True),
    )


def prefix_losses(predictions: pd.DataFrame, model: str) -> pd.DataFrame:
    return predictions.rename(
        columns={
            "expected_home_score": f"{model}_expected_home_score",
            "brier_loss": f"{model}_brier_loss",
            "log_loss": f"{model}_log_loss",
        }
    )


def add_event_clusters(predictions: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    metadata = events[["match_id", "tie_id", "kickoff_utc"]]
    result = predictions.merge(metadata, on="match_id", validate="one_to_one")
    result["cluster_id"] = result["season"].astype(str) + "|" + result[
        "tie_id"
    ].fillna(result["match_id"]).astype(str)
    return result


def summarize_loss_differences(
    predictions: pd.DataFrame,
    model: str,
    baseline: str,
) -> pd.DataFrame:
    rows = []
    for competition, frame in predictions.groupby("competition", sort=True):
        rows.append(
            {
                "competition": competition,
                "matches": len(frame),
                "model_brier": frame[f"{model}_brier_loss"].mean(),
                "baseline_brier": frame[f"{baseline}_brier_loss"].mean(),
                "brier_difference": (
                    frame[f"{model}_brier_loss"].mean()
                    - frame[f"{baseline}_brier_loss"].mean()
                ),
                "log_loss_difference": (
                    frame[f"{model}_log_loss"].mean()
                    - frame[f"{baseline}_log_loss"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def clustered_uncertainty(
    predictions: pd.DataFrame,
    model: str,
    baseline: str,
    *,
    bootstrap_samples: int = 4000,
    seed: int = 20260717,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    groups = [("ALL", predictions), *predictions.groupby("competition", sort=True)]
    for competition, frame in groups:
        values = frame.copy()
        values["difference"] = (
            values[f"{model}_brier_loss"] - values[f"{baseline}_brier_loss"]
        )
        clusters = values.groupby("cluster_id")["difference"].agg(["sum", "count"])
        indices = rng.integers(
            0,
            len(clusters),
            size=(bootstrap_samples, len(clusters)),
        )
        means = (
            clusters["sum"].to_numpy()[indices].sum(axis=1)
            / clusters["count"].to_numpy()[indices].sum(axis=1)
        )
        lower, upper = np.quantile(means, (0.025, 0.975))
        rows.append(
            {
                "competition": competition,
                "matches": len(values),
                "clusters": len(clusters),
                "mean_brier_difference": values["difference"].mean(),
                "ci_95_lower": lower,
                "ci_95_upper": upper,
                "reliable_improvement": bool(upper < 0.0),
                "reliable_harm": bool(lower > 0.0),
            }
        )
    return pd.DataFrame(rows)


def core_promotion_decision(
    fold_results: pd.DataFrame,
    uncertainty: pd.DataFrame,
) -> tuple[str, dict[str, object]]:
    pivot = fold_results.pivot(index="fold", columns="model", values="brier")
    fold_wins = int((pivot[CORE_MODEL_NAME] < pivot[CORE_BASELINE_NAME]).sum())
    dynamic = fold_results.loc[fold_results["model"].eq(CORE_MODEL_NAME)]
    overall = uncertainty.loc[uncertainty["competition"].eq("ALL")].iloc[0]
    segments = uncertainty.loc[uncertainty["competition"].ne("ALL")]
    zero_sum = bool(dynamic["mean_rating_change"].abs().le(1e-9).all())
    ranking_safe = bool(
        dynamic["start_end_rank_correlation"].ge(RANK_CORRELATION_FLOOR).all()
    )
    movement_safe = bool(
        dynamic["max_abs_rating_change"].le(MAX_RATING_MOVE_GUARDRAIL).all()
    )
    no_segment_harm = not bool(segments["reliable_harm"].any())
    passed = (
        fold_wins == 6
        and bool(overall["reliable_improvement"])
        and zero_sum
        and ranking_safe
        and movement_safe
        and no_segment_harm
    )
    guardrails = {
        "fold_wins": fold_wins,
        "required_fold_wins": 6,
        "overall_reliable_improvement": bool(overall["reliable_improvement"]),
        "zero_sum": zero_sum,
        "ranking_safe": ranking_safe,
        "movement_safe": movement_safe,
        "no_competition_reliable_harm": no_segment_harm,
    }
    return ("PROMOTE_DYNAMIC_CORE" if passed else "REJECT_DYNAMIC_CORE"), guardrails


def carry_candidates() -> tuple[AchievementCarryConfig, ...]:
    return tuple(
        AchievementCarryConfig(float(carry), 0.0, 0.0)
        for carry in POWER_CARRY_CANDIDATES
    )


def select_carry_candidate(
    datasets: tuple,
    core: DynamicCoreConfig,
) -> tuple[AchievementCarryConfig, pd.DataFrame]:
    rows = []
    for candidate in carry_candidates():
        metrics, _, _ = evaluate_sequence(datasets, core, candidate)
        rows.append({"power_carry": candidate.power_carry, **metrics})
    table = pd.DataFrame(rows).sort_values(
        ["brier", "log_loss", "power_carry"]
    ).reset_index(drop=True)
    eligible = table.loc[
        table["start_end_rank_correlation"].ge(RANK_CORRELATION_FLOOR)
        & table["max_abs_rating_change"].le(MAX_RATING_MOVE_GUARDRAIL)
    ]
    if eligible.empty:
        raise ValueError("No season-carry candidate passes ranking/movement guardrails")
    return AchievementCarryConfig(float(eligible.iloc[0]["power_carry"]), 0.0, 0.0), table


def run_carry_walk_forward(
    datasets: tuple,
    folds: tuple[tuple[tuple[str, ...], str], ...],
    core_selections: pd.DataFrame,
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selections: list[dict[str, object]] = []
    results: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    baseline = AchievementCarryConfig(0.0, 0.0, 0.0)
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
        selected, train_table = select_carry_candidate(train, core)
        selected_train = train_table.loc[
            train_table["power_carry"].eq(selected.power_carry)
        ].iloc[0]
        selections.append(
            {
                "fold": fold,
                "train_seasons": "|".join(train_seasons),
                "test_season": test_season,
                "core_scale": core.elo_scale,
                "core_home_advantage": core.home_advantage,
                "core_k": core.k_factor,
                "selected_power_carry": selected.power_carry,
                "train_brier": selected_train["brier"],
            }
        )
        models: dict[str, pd.DataFrame] = {}
        for model, config in ((CARRY_MODEL_NAME, selected), (CARRY_BASELINE_NAME, baseline)):
            metrics, predictions, _ = evaluate_sequence(
                sequence,
                core,
                config,
                evaluation_seasons={test_season},
                return_predictions=True,
            )
            results.append(
                {
                    "fold": fold,
                    "test_season": test_season,
                    "model": model,
                    "power_carry": config.power_carry,
                    **metrics,
                }
            )
            assert predictions is not None
            models[model] = prefix_losses(predictions, model)
        joined = models[CARRY_MODEL_NAME].merge(
            models[CARRY_BASELINE_NAME][
                [
                    "match_id",
                    f"{CARRY_BASELINE_NAME}_expected_home_score",
                    f"{CARRY_BASELINE_NAME}_brier_loss",
                    f"{CARRY_BASELINE_NAME}_log_loss",
                ]
            ],
            on="match_id",
            validate="one_to_one",
        )
        joined.insert(0, "fold", fold)
        prediction_frames.append(add_event_clusters(joined, events))
    return (
        pd.DataFrame(selections),
        pd.DataFrame(results),
        pd.concat(prediction_frames, ignore_index=True),
    )


def carry_promotion_decision(
    fold_results: pd.DataFrame,
    uncertainty: pd.DataFrame,
) -> tuple[str, dict[str, object]]:
    pivot = fold_results.pivot(index="fold", columns="model", values="brier")
    fold_wins = int((pivot[CARRY_MODEL_NAME] < pivot[CARRY_BASELINE_NAME]).sum())
    selected = fold_results.loc[fold_results["model"].eq(CARRY_MODEL_NAME)]
    ucl = uncertainty.loc[uncertainty["competition"].eq("UCL")].iloc[0]
    ranking_safe = bool(
        selected["start_end_rank_correlation"].ge(RANK_CORRELATION_FLOOR).all()
    )
    movement_safe = bool(
        selected["max_abs_rating_change"].le(MAX_RATING_MOVE_GUARDRAIL).all()
    )
    passed = (
        fold_wins == 6
        and not bool(ucl["reliable_harm"])
        and ranking_safe
        and movement_safe
    )
    guardrails = {
        "fold_wins": fold_wins,
        "required_fold_wins": 6,
        "ucl_reliable_harm": bool(ucl["reliable_harm"]),
        "ranking_safe": ranking_safe,
        "movement_safe": movement_safe,
    }
    return ("PROMOTE_CARRY" if passed else "DISABLE_CARRY"), guardrails


def write_manifest(
    path: Path,
    static_config: AOEuropeanEloConfig,
    core: DynamicCoreConfig,
    core_metrics: dict[str, float | int],
    core_decision: str,
    core_guardrails: dict[str, object],
    carry_candidate: AchievementCarryConfig,
    active_carry: float,
    carry_decision: str,
    carry_guardrails: dict[str, object],
) -> None:
    payload = {
        "model_version": AO_MODEL_V2_VERSION,
        "static_config": asdict(static_config),
        "dynamic_core": asdict(core),
        "dynamic_core_metrics": core_metrics,
        "dynamic_core_decision": core_decision,
        "dynamic_core_guardrails": core_guardrails,
        "carry_candidate": carry_candidate.power_carry,
        "active_power_carry": active_carry,
        "carry_decision": carry_decision,
        "carry_guardrails": carry_guardrails,
        "goal_margin": {"active": False, "weight": 0.0, "cap": 1.0},
        "achievement_reserve": {"active": False, "base": 0.0},
    }
    path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")


def json_default(value: object) -> object:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_report(
    path: Path,
    seasons: tuple[str, ...],
    core_selections: pd.DataFrame,
    core_competitions: pd.DataFrame,
    core_uncertainty: pd.DataFrame,
    final_core: DynamicCoreConfig,
    core_decision: str,
    carry_selections: pd.DataFrame,
    carry_competitions: pd.DataFrame,
    carry_uncertainty: pd.DataFrame,
    final_carry: AchievementCarryConfig,
    active_carry: float,
    carry_decision: str,
) -> None:
    core_overall = core_uncertainty.loc[
        core_uncertainty["competition"].eq("ALL")
    ].iloc[0]
    carry_overall = carry_uncertainty.loc[
        carry_uncertainty["competition"].eq("ALL")
    ].iloc[0]
    lines = [
        "# AO European Elo v2 Dynamic Core And Carry Calibration",
        "",
        f"Exact-date development seasons: `{seasons[0]}` through `{seasons[-1]}`.",
        "All matches are processed in exact UTC order. Penalty shootouts do not alter "
        "the field score contract.",
        "",
        "## Dynamic Core",
        "",
        f"- Decision: `{core_decision}`",
        f"- Scale: `{final_core.elo_scale:.6f}`",
        f"- Home advantage: `{final_core.home_advantage:.6f}`",
        f"- K: `{final_core.k_factor:.6f}`",
        f"- Unseen overall Brier difference: "
        f"`{core_overall.mean_brier_difference:+.6f}`",
        f"- Clustered 95% CI: `[{core_overall.ci_95_lower:+.6f}, "
        f"{core_overall.ci_95_upper:+.6f}]`",
        "",
        "| Competition | Matches | Brier difference | Log-loss difference |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in core_competitions.itertuples(index=False):
        lines.append(
            f"| {row.competition} | {row.matches} | {row.brier_difference:+.6f} | "
            f"{row.log_loss_difference:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Season Carry",
            "",
            f"- Decision: `{carry_decision}`",
            f"- Full-data carry candidate: `{final_carry.power_carry:g}`",
            f"- Active carry: `{active_carry:g}`",
            f"- Unseen overall Brier difference: "
            f"`{carry_overall.mean_brier_difference:+.6f}`",
            f"- Clustered 95% CI: `[{carry_overall.ci_95_lower:+.6f}, "
            f"{carry_overall.ci_95_upper:+.6f}]`",
            "",
            "| Competition | Matches | Brier difference | Log-loss difference |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in carry_competitions.itertuples(index=False):
        lines.append(
            f"| {row.competition} | {row.matches} | {row.brier_difference:+.6f} | "
            f"{row.log_loss_difference:+.6f} |"
        )
    lines.extend(
        [
            "",
            "The core and carry are promoted independently. Goal margin and European "
            "Achievement Reserve remain disabled until their incremental tests pass.",
            "",
            "The `2026/27` season remains untouched for future holdout evaluation.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
