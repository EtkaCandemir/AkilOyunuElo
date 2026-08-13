from __future__ import annotations

"""Shadow-only response-curve analysis for Domestic Surprise effect size."""

import argparse
import hashlib
import json
import math
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

from ao_elo.config import AOEuropeanEloConfig  # noqa: E402
from ao_elo.domestic_surprise_variance import (  # noqa: E402
    VarianceDomesticSurpriseConfig,
)
from ao_elo.evaluation import (  # noqa: E402
    dependency_robust_loss_difference_ci,
    schedule_adjusted_team_performance,
)
from scripts.run_controlled_goal_progression_backtest import (  # noqa: E402
    prepare_controlled_data,
)
from scripts.run_current_model_evaluation import (  # noqa: E402
    EvaluationArm,
    aggregate_ranking,
    evaluate_arm,
    prediction_summary,
)
from scripts.run_domestic_surprise_variance_backtest import (  # noqa: E402
    build_adjustments,
)
from scripts.run_dynamic_core_calibration import expanding_folds  # noqa: E402
from scripts.run_final_robustness import (  # noqa: E402
    load_team_season_identity,
    summarize_ranking,
)
from scripts.run_stage_weighted_progression_backtest import (  # noqa: E402
    DOMESTIC_ADJUSTMENTS,
    DYNAMIC_MANIFEST,
    EVENTS_PATH,
    PRODUCTION_CONTRACT,
    STATIC_DATA_ROOT,
    XG_DATA,
    load_domestic_adjustments,
    load_xg_map,
    validate_production_contract,
)
from scripts.run_v2_achievement_reserve_calibration import load_reserve_data  # noqa: E402
from scripts.run_v2_evaluation_upgrade import read_events  # noqa: E402


FEATURES_PATH = (
    ROOT
    / "output"
    / "domestic_surprise_variance_backtest_2018_2026"
    / "domestic_surprise_features.csv"
)
OUTPUT_ROOT = ROOT / "output" / "domestic_surprise_effect_size_sensitivity"
COEFFICIENT_GRID = (0.0, 0.40, 0.60, 0.80, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00, 4.00)
CAP_GRID = (30.0, 50.0, 75.0, 100.0, 150.0)
TARGET_EFFECTS = (10.0, 15.0, 20.0, 25.0, 30.0, 40.0)
NO_SURPRISE_KEY = "theta_0_cap_30"


def candidate_key(config: VarianceDomesticSurpriseConfig) -> str:
    theta = f"{config.coefficient:g}".replace(".", "p")
    cap = f"{config.max_abs_adjustment:g}".replace(".", "p")
    return f"theta_{theta}_cap_{cap}"


def candidate_grid(production: dict[str, object]) -> tuple[VarianceDomesticSurpriseConfig, ...]:
    block = production["domestic_surprise"]
    gamma = float(block["variance_penalty"])
    minimum_history = int(block["minimum_history_seasons"])
    production_config = VarianceDomesticSurpriseConfig(
        coefficient=float(block["coefficient"]),
        variance_penalty=gamma,
        max_abs_adjustment=float(block["max_abs_adjustment"]),
        minimum_history_seasons=minimum_history,
    )
    candidates = {production_config}
    for theta in COEFFICIENT_GRID:
        caps = (float(block["max_abs_adjustment"]),) if theta == 0.0 else CAP_GRID
        for cap in caps:
            candidates.add(
                VarianceDomesticSurpriseConfig(
                    coefficient=theta,
                    variance_penalty=gamma,
                    max_abs_adjustment=cap,
                    minimum_history_seasons=minimum_history,
                )
            )
    result = tuple(sorted(candidates))
    for candidate in result:
        candidate.validate()
    if production_config not in result:
        raise AssertionError("Production Domestic Surprise config is missing from grid")
    return result


def effect_distribution(
    adjustments: pd.DataFrame,
    config: VarianceDomesticSurpriseConfig,
    static_config: AOEuropeanEloConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = adjustments.copy()
    frame["candidate_key"] = candidate_key(config)
    frame["theta"] = config.coefficient
    frame["cap"] = config.max_abs_adjustment
    frame["initial_delta"] = frame["ao_first_elo_adjustment"]
    frame["changed"] = frame["initial_delta"].abs().gt(1e-12)
    frame["baseline_rank"] = frame.groupby("season")["baseline_ao_first_elo"].rank(
        method="min", ascending=False
    )
    frame["candidate_rank"] = frame.groupby("season")["adjusted_ao_first_elo"].rank(
        method="min", ascending=False
    )
    frame["rank_change"] = frame["baseline_rank"] - frame["candidate_rank"]
    frame["positive_cap_hit"] = np.isclose(
        frame["domestic_prior_adjustment"], config.max_abs_adjustment, atol=1e-9
    )
    frame["negative_cap_hit"] = np.isclose(
        frame["domestic_prior_adjustment"], -config.max_abs_adjustment, atol=1e-9
    )
    frame["exposure_band"] = pd.cut(
        frame["effective_european_exposure"],
        bins=[-1e-12, 0.0, 0.25, 0.50, 0.75, 1.0],
        labels=["0", "(0,0.25]", "(0.25,0.50]", "(0.50,0.75]", "(0.75,1.00]"],
        include_lowest=True,
    )
    frame["domestic_achievement_contribution"] = (
        frame["baseline_domestic_prior"]
        - static_config.base_rating
        - static_config.domestic_league_component
        * frame["league_strength"]
    )
    changed = frame.loc[frame["changed"]]
    summary = {
        "candidate_key": candidate_key(config),
        "theta": config.coefficient,
        "gamma": config.variance_penalty,
        "cap": config.max_abs_adjustment,
        "team_seasons": len(frame),
        "changed_team_seasons": int(frame["changed"].sum()),
        "changed_share": float(frame["changed"].mean()),
        "mean_abs_initial_delta": float(frame["initial_delta"].abs().mean()),
        "changed_mean_abs_initial_delta": (
            float(changed["initial_delta"].abs().mean()) if len(changed) else 0.0
        ),
        "median_abs_initial_delta": float(frame["initial_delta"].abs().median()),
        "p75_abs_initial_delta": float(frame["initial_delta"].abs().quantile(0.75)),
        "p90_abs_initial_delta": float(frame["initial_delta"].abs().quantile(0.90)),
        "p95_abs_initial_delta": float(frame["initial_delta"].abs().quantile(0.95)),
        "maximum_positive_initial_delta": float(frame["initial_delta"].max()),
        "maximum_negative_initial_delta": float(frame["initial_delta"].min()),
        "positive_cap_hits": int(frame["positive_cap_hit"].sum()),
        "negative_cap_hits": int(frame["negative_cap_hit"].sum()),
        "total_cap_hit_rate": float(
            (frame["positive_cap_hit"] | frame["negative_cap_hit"]).mean()
        ),
        "mean_abs_rank_change": float(frame["rank_change"].abs().mean()),
        "maximum_rank_gain": int(frame["rank_change"].max()),
        "maximum_rank_loss": int(frame["rank_change"].min()),
        "minimum_initial_rating": float(frame["adjusted_ao_first_elo"].min()),
        "maximum_initial_rating": float(frame["adjusted_ao_first_elo"].max()),
    }
    return frame, summary


def model_metrics(
    evaluation,
    evaluation_seasons: set[str],
    target: pd.DataFrame,
    identity: pd.DataFrame,
    seasons: tuple[str, ...],
) -> dict[str, object]:
    predictions = evaluation.predictions.loc[
        evaluation.predictions["season"].isin(evaluation_seasons)
    ]
    ranking = aggregate_ranking(
        evaluation.same_season_ranking.loc[
            evaluation.same_season_ranking["season"].isin(evaluation_seasons)
        ]
    )
    pooled = ranking.loc[ranking["competition"].eq("ALL")].iloc[0]
    forward = summarize_ranking(
        evaluation.end_ratings,
        target,
        allowed_target_seasons=set(seasons[3:]),
        identity=identity,
    )
    forward_all = forward.loc[forward["competition"].eq("ALL")].iloc[0]
    return {
        **prediction_summary(predictions),
        "same_season_spearman": float(pooled["ranking_score"]),
        "same_season_pairwise_accuracy": float(pooled["pairwise_accuracy"]),
        "forward_season_spearman": float(forward_all["ranking_score"]),
        "forward_season_pairwise_accuracy": float(forward_all["pairwise_accuracy"]),
        "maximum_abs_match_delta": float(
            evaluation.season_metrics.loc[
                evaluation.season_metrics["season"].isin(evaluation_seasons),
                "maximum_abs_match_delta",
            ].max()
        ),
        "maximum_match_zero_sum_error": float(
            evaluation.season_metrics["maximum_match_zero_sum_error"].max()
        ),
        "maximum_power_conservation_error": float(
            evaluation.season_metrics["season_power_conservation_error"].max()
        ),
    }


def fold_metrics(evaluation, folds) -> pd.DataFrame:
    rows = []
    seasons = tuple(test for _, test in folds)
    for fold, test_season in enumerate(seasons, start=1):
        predictions = evaluation.predictions.loc[
            evaluation.predictions["season"].eq(test_season)
        ]
        ranking = aggregate_ranking(
            evaluation.same_season_ranking.loc[
                evaluation.same_season_ranking["season"].eq(test_season)
            ]
        )
        pooled = ranking.loc[ranking["competition"].eq("ALL")].iloc[0]
        rows.append(
            {
                "fold": fold,
                "test_season": test_season,
                **prediction_summary(predictions),
                "same_season_spearman": float(pooled["ranking_score"]),
                "same_season_pairwise_accuracy": float(pooled["pairwise_accuracy"]),
            }
        )
    return pd.DataFrame(rows)


def competition_metrics(evaluation, evaluation_seasons: set[str]) -> pd.DataFrame:
    predictions = evaluation.predictions.loc[
        evaluation.predictions["season"].isin(evaluation_seasons)
    ]
    ranking = aggregate_ranking(
        evaluation.same_season_ranking.loc[
            evaluation.same_season_ranking["season"].isin(evaluation_seasons)
        ]
    ).set_index("competition")
    rows = []
    for competition, frame in predictions.groupby("competition", sort=True):
        rows.append(
            {
                "competition": competition,
                **prediction_summary(frame),
                "same_season_spearman": float(ranking.loc[competition, "ranking_score"]),
                "same_season_pairwise_accuracy": float(
                    ranking.loc[competition, "pairwise_accuracy"]
                ),
            }
        )
    return pd.DataFrame(rows)


def add_baseline_deltas(frame: pd.DataFrame, baseline_key: str) -> pd.DataFrame:
    result = frame.copy()
    baseline = result.loc[result["candidate_key"].eq(baseline_key)]
    if len(baseline) != 1:
        raise ValueError("Exactly one production candidate is required")
    baseline = baseline.iloc[0]
    for metric in (
        "brier_1x2",
        "log_loss_1x2",
        "accuracy_1x2",
        "same_season_spearman",
        "same_season_pairwise_accuracy",
        "forward_season_spearman",
        "forward_season_pairwise_accuracy",
    ):
        result[f"delta_vs_current_{metric}"] = result[metric] - baseline[metric]
    return result


def add_fold_deltas(frame: pd.DataFrame, baseline_key: str) -> pd.DataFrame:
    baseline = frame.loc[frame["candidate_key"].eq(baseline_key)].set_index("fold")
    result = frame.copy()
    for metric in (
        "brier_1x2",
        "log_loss_1x2",
        "accuracy_1x2",
        "same_season_spearman",
        "same_season_pairwise_accuracy",
    ):
        result[f"delta_vs_current_{metric}"] = result.apply(
            lambda row: row[metric] - baseline.loc[row["fold"], metric], axis=1
        )
    return result


def summarize_fold_stability(fold_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, frame in fold_results.groupby("candidate_key", sort=False):
        rows.append(
            {
                "candidate_key": key,
                "brier_improved_folds": int(
                    frame["delta_vs_current_brier_1x2"].lt(-1e-12).sum()
                ),
                "log_loss_improved_folds": int(
                    frame["delta_vs_current_log_loss_1x2"].lt(-1e-12).sum()
                ),
                "spearman_non_regressed_folds": int(
                    frame["delta_vs_current_same_season_spearman"].ge(-1e-12).sum()
                ),
                "spearman_improved_folds": int(
                    frame["delta_vs_current_same_season_spearman"].gt(1e-12).sum()
                ),
                "pairwise_non_regressed_folds": int(
                    frame["delta_vs_current_same_season_pairwise_accuracy"].ge(-1e-12).sum()
                ),
                "pairwise_improved_folds": int(
                    frame["delta_vs_current_same_season_pairwise_accuracy"].gt(1e-12).sum()
                ),
                "maximum_fold_brier_harm": float(
                    frame["delta_vs_current_brier_1x2"].max()
                ),
                "maximum_fold_log_loss_harm": float(
                    frame["delta_vs_current_log_loss_1x2"].max()
                ),
            }
        )
    return pd.DataFrame(rows)


def dependency_uncertainty(
    evaluations: dict[str, object],
    selected_keys: list[str],
    current_key: str,
    evaluation_seasons: set[str],
    *,
    bootstrap_samples: int,
) -> pd.DataFrame:
    rows = []
    for key in selected_keys:
        baseline_key = NO_SURPRISE_KEY if key == current_key else current_key
        candidate = evaluations[key].predictions.loc[
            evaluations[key].predictions["season"].isin(evaluation_seasons)
        ]
        baseline = evaluations[baseline_key].predictions.loc[
            evaluations[baseline_key].predictions["season"].isin(evaluation_seasons)
        ]
        for competition in ("ALL", "UCL", "UEL", "UECL"):
            left = candidate if competition == "ALL" else candidate.loc[
                candidate["competition"].eq(competition)
            ]
            right = baseline if competition == "ALL" else baseline.loc[
                baseline["competition"].eq(competition)
            ]
            paired = left.merge(
                right[["match_id", "brier_1x2", "log_loss_1x2"]],
                on="match_id",
                suffixes=("_candidate", "_baseline"),
                validate="one_to_one",
            )
            for metric in ("brier_1x2", "log_loss_1x2"):
                sample = paired[
                    [
                        "season",
                        "match_id",
                        "home_team_id",
                        "away_team_id",
                        "kickoff_utc",
                        "tie_id",
                    ]
                ].copy()
                sample["loss_difference"] = (
                    paired[f"{metric}_candidate"] - paired[f"{metric}_baseline"]
                )
                result = dependency_robust_loss_difference_ci(
                    sample, bootstrap_samples=bootstrap_samples
                )
                result.insert(0, "candidate_key", key)
                result.insert(1, "baseline_key", baseline_key)
                result.insert(2, "competition", competition)
                result.insert(3, "metric", metric)
                rows.append(result)
    return pd.concat(rows, ignore_index=True)


def exposure_analysis(distributions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (key, band), frame in distributions.groupby(
        ["candidate_key", "exposure_band"], observed=False, sort=False
    ):
        changed = frame.loc[frame["changed"]]
        rows.append(
            {
                "candidate_key": key,
                "theta": float(frame["theta"].iloc[0]),
                "cap": float(frame["cap"].iloc[0]),
                "exposure_band": str(band),
                "team_seasons": len(frame),
                "changed_team_seasons": int(frame["changed"].sum()),
                "mean_abs_initial_delta": float(frame["initial_delta"].abs().mean()),
                "changed_mean_abs_initial_delta": (
                    float(changed["initial_delta"].abs().mean()) if len(changed) else 0.0
                ),
                "p90_abs_initial_delta": float(frame["initial_delta"].abs().quantile(0.90)),
                "maximum_abs_initial_delta": float(frame["initial_delta"].abs().max()),
                "positive_cap_hits": int(frame["positive_cap_hit"].sum()),
                "negative_cap_hits": int(frame["negative_cap_hit"].sum()),
            }
        )
    return pd.DataFrame(rows)


def achievement_double_counting(distributions: pd.DataFrame) -> pd.DataFrame:
    def safe_correlation(left: pd.Series, right: pd.Series, method: str) -> float:
        if left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
            return float("nan")
        return float(left.corr(right, method=method))

    rows = []
    for key, frame in distributions.groupby("candidate_key", sort=False):
        eligible = frame.loc[frame["history_seasons"].ge(5)].copy()
        rows.append(
            {
                "candidate_key": key,
                "theta": float(frame["theta"].iloc[0]),
                "cap": float(frame["cap"].iloc[0]),
                "eligible_team_seasons": len(eligible),
                "achievement_vs_raw_surprise_pearson": safe_correlation(
                    eligible["domestic_achievement_contribution"],
                    eligible["raw_surprise"],
                    "pearson",
                ),
                "achievement_vs_raw_surprise_spearman": safe_correlation(
                    eligible["domestic_achievement_contribution"],
                    eligible["raw_surprise"],
                    "spearman",
                ),
                "achievement_vs_adjustment_pearson": safe_correlation(
                    eligible["domestic_achievement_contribution"],
                    eligible["domestic_prior_adjustment"],
                    "pearson",
                ),
                "achievement_vs_adjustment_spearman": safe_correlation(
                    eligible["domestic_achievement_contribution"],
                    eligible["domestic_prior_adjustment"],
                    "spearman",
                ),
            }
        )
    return pd.DataFrame(rows)


def nearest_effect_models(surface: pd.DataFrame, current_key: str) -> pd.DataFrame:
    rows = []
    current = surface.loc[surface["candidate_key"].eq(current_key)].iloc[0]
    rows.append({"effect_target": "CURRENT", **current.to_dict()})
    for target in TARGET_EFFECTS:
        selected = surface.assign(
            distance_to_target=(surface["changed_mean_abs_initial_delta"] - target).abs()
        ).sort_values(
            ["distance_to_target", "cap", "theta", "candidate_key"], kind="stable"
        ).iloc[0]
        payload = selected.drop(labels=["distance_to_target"]).to_dict()
        rows.append({"effect_target": f"~{target:g}_ELO", **payload})
    return pd.DataFrame(rows)


def select_diagnostic_models(surface: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    best_loss = surface.sort_values(
        ["brier_1x2", "log_loss_1x2", "changed_mean_abs_initial_delta", "candidate_key"],
        kind="stable",
    ).iloc[0]
    best_ranking = surface.sort_values(
        ["same_season_spearman", "same_season_pairwise_accuracy", "brier_1x2"],
        ascending=[False, False, True],
        kind="stable",
    ).iloc[0]
    return best_loss, best_ranking


def classify_conclusion(
    surface: pd.DataFrame,
    current_key: str,
    best_loss: pd.Series,
    uncertainty: pd.DataFrame,
) -> str:
    current = surface.loc[surface["candidate_key"].eq(current_key)].iloc[0]
    best_key = str(best_loss["candidate_key"])
    envelope = uncertainty.loc[
        uncertainty["candidate_key"].eq(best_key)
        & uncertainty["baseline_key"].eq(current_key)
        & uncertainty["competition"].eq("ALL")
        & uncertainty["method"].eq("conservative_envelope")
    ]
    reliable_both = bool(
        best_key != current_key
        and len(envelope) == 2
        and envelope["reliable_improvement"].all()
    )
    ranking_safe = bool(
        best_loss["delta_vs_current_same_season_spearman"] >= 0.0
        and best_loss["delta_vs_current_same_season_pairwise_accuracy"] >= 0.0
    )
    high = surface.loc[surface["changed_mean_abs_initial_delta"].ge(30.0)]
    high_harmful = bool(
        len(high)
        and (high["delta_vs_current_brier_1x2"] > 0.0).all()
        and (high["delta_vs_current_log_loss_1x2"] > 0.0).all()
    )
    if reliable_both and ranking_safe:
        ratio = float(
            best_loss["changed_mean_abs_initial_delta"]
            / max(current["changed_mean_abs_initial_delta"], 1e-12)
        )
        return "STRONG_UPSCALING_SUPPORTED" if ratio >= 3.0 else "MODERATE_UPSCALING_SUPPORTED"
    if best_key == current_key:
        return "CURRENT_SCALE_BEST"
    if high_harmful:
        return "HIGH_SCALE_HARMFUL"
    return "NO_CLEAR_OPTIMUM"


def safety_audit(
    distributions: pd.DataFrame,
    evaluations: dict[str, object],
    configs: dict[str, VarianceDomesticSurpriseConfig],
    production_hash_before: str,
    production_hash_after: str,
) -> pd.DataFrame:
    rows = []
    for key, frame in distributions.groupby("candidate_key", sort=False):
        config = configs[key]
        evaluation = evaluations[key]
        probabilities = evaluation.predictions[
            ["home_probability", "draw_probability", "away_probability"]
        ]
        checks = {
            "team_season_unique": not frame.duplicated(["season", "team_id"]).any(),
            "probabilities_normalized": bool(
                np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)
            ),
            "exposure_in_range": bool(
                frame["effective_european_exposure"].between(0.0, 1.0).all()
            ),
            "insufficient_history_zero": bool(
                frame.loc[frame["history_seasons"].lt(5), "initial_delta"]
                .abs()
                .le(1e-12)
                .all()
            ),
            "sign_preserved": bool(
                (frame["domestic_prior_adjustment"] * frame["raw_surprise"])
                .ge(-1e-12)
                .all()
            ),
            "cap_respected": bool(
                frame["domestic_prior_adjustment"]
                .abs()
                .le(config.max_abs_adjustment + 1e-9)
                .all()
            ),
            "match_zero_sum": bool(
                evaluation.predictions["zero_sum_error"].abs().max() <= 1e-9
            ),
            "power_conserved": bool(
                evaluation.season_metrics["season_power_conservation_error"].max()
                <= 1e-9
            ),
            "finite_ratings": bool(
                np.isfinite(frame["adjusted_ao_first_elo"]).all()
                and np.isfinite(evaluation.end_ratings["end_live_rating"]).all()
            ),
        }
        rows.extend(
            {
                "candidate_key": key,
                "check": check,
                "passed": passed,
            }
            for check, passed in checks.items()
        )
    rows.append(
        {
            "candidate_key": "ALL",
            "check": "production_contract_unchanged",
            "passed": production_hash_before == production_hash_after,
        }
    )
    return pd.DataFrame(rows)


def write_response_plot(surface: pd.DataFrame, path: Path) -> None:
    ordered = surface.sort_values("changed_mean_abs_initial_delta")
    metrics = (
        ("delta_vs_current_brier_1x2", "Brier difference", True),
        ("delta_vs_current_log_loss_1x2", "Log-loss difference", True),
        ("delta_vs_current_same_season_spearman", "Spearman difference", False),
        (
            "delta_vs_current_same_season_pairwise_accuracy",
            "Pairwise difference",
            False,
        ),
    )
    width, height = 1200, 820
    panel_width, panel_height = 520, 300
    origins = ((80, 100), (650, 100), (80, 480), (650, 480))
    x_values = ordered["changed_mean_abs_initial_delta"].to_numpy(float)
    x_min, x_max = float(x_values.min()), float(x_values.max())
    cap_min, cap_max = float(ordered["cap"].min()), float(ordered["cap"].max())

    def scale(value: float, low: float, high: float, start: float, span: float) -> float:
        return start + span * (value - low) / max(high - low, 1e-12)

    def color(cap: float) -> str:
        ratio = (cap - cap_min) / max(cap_max - cap_min, 1e-12)
        red = int(35 + 200 * ratio)
        blue = int(220 - 150 * ratio)
        return f"rgb({red},95,{blue})"

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="600" y="38" text-anchor="middle" font-family="Arial" font-size="22" font-weight="bold">Domestic Surprise realized effect response curve</text>',
    ]
    for (column, title, lower_is_better), (origin_x, origin_y) in zip(
        metrics, origins, strict=True
    ):
        y_values = ordered[column].to_numpy(float)
        y_min, y_max = float(y_values.min()), float(y_values.max())
        padding = max((y_max - y_min) * 0.08, 1e-6)
        y_min -= padding
        y_max += padding
        svg.extend(
            [
                f'<text x="{origin_x + panel_width / 2}" y="{origin_y - 28}" text-anchor="middle" font-family="Arial" font-size="16" font-weight="bold">{title} ({"lower" if lower_is_better else "higher"} is better)</text>',
                f'<line x1="{origin_x}" y1="{origin_y}" x2="{origin_x}" y2="{origin_y + panel_height}" stroke="#333"/>',
                f'<line x1="{origin_x}" y1="{origin_y + panel_height}" x2="{origin_x + panel_width}" y2="{origin_y + panel_height}" stroke="#333"/>',
            ]
        )
        if y_min <= 0.0 <= y_max:
            zero_y = scale(0.0, y_min, y_max, origin_y + panel_height, -panel_height)
            svg.append(
                f'<line x1="{origin_x}" y1="{zero_y:.2f}" x2="{origin_x + panel_width}" y2="{zero_y:.2f}" stroke="#777" stroke-dasharray="4 4"/>'
            )
        for row in ordered.itertuples(index=False):
            x = scale(
                float(row.changed_mean_abs_initial_delta),
                x_min,
                x_max,
                origin_x,
                panel_width,
            )
            y = scale(float(getattr(row, column)), y_min, y_max, origin_y + panel_height, -panel_height)
            svg.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color(float(row.cap))}" fill-opacity="0.78"><title>{row.candidate_key}: effect={row.changed_mean_abs_initial_delta:.3f}, {column}={getattr(row, column):.6f}</title></circle>'
            )
        svg.extend(
            [
                f'<text x="{origin_x + panel_width / 2}" y="{origin_y + panel_height + 35}" text-anchor="middle" font-family="Arial" font-size="12">Changed-team mean absolute initial Elo effect</text>',
                f'<text x="{origin_x - 12}" y="{origin_y + 5}" text-anchor="end" font-family="Arial" font-size="11">{y_max:.4f}</text>',
                f'<text x="{origin_x - 12}" y="{origin_y + panel_height}" text-anchor="end" font-family="Arial" font-size="11">{y_min:.4f}</text>',
            ]
        )
    svg.append('</svg>')
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame, digits: int = 6) -> str:
    if frame.empty:
        return "_No rows._"
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.{digits}f}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    *,
    production: dict[str, object],
    current: pd.Series,
    best_loss: pd.Series,
    best_ranking: pd.Series,
    decision_table: pd.DataFrame,
    uncertainty: pd.DataFrame,
    competition: pd.DataFrame,
    exposure: pd.DataFrame,
    double_counting: pd.DataFrame,
    conclusion: str,
    matches: int,
) -> None:
    display_columns = [
        "effect_target",
        "candidate_key",
        "theta",
        "cap",
        "changed_mean_abs_initial_delta",
        "brier_1x2",
        "delta_vs_current_brier_1x2",
        "log_loss_1x2",
        "delta_vs_current_log_loss_1x2",
        "same_season_spearman",
        "delta_vs_current_same_season_spearman",
        "same_season_pairwise_accuracy",
        "delta_vs_current_same_season_pairwise_accuracy",
    ]
    selected_uncertainty = uncertainty.loc[
        uncertainty["method"].eq("conservative_envelope")
        & uncertainty["competition"].eq("ALL")
    ]
    best_competition = competition.loc[
        competition["candidate_key"].eq(best_loss["candidate_key"])
    ]
    current_exposure = exposure.loc[
        exposure["candidate_key"].eq(current["candidate_key"])
    ]
    best_exposure = exposure.loc[
        exposure["candidate_key"].eq(best_loss["candidate_key"])
    ]
    double_rows = double_counting.loc[
        double_counting["candidate_key"].isin(
            [current["candidate_key"], best_loss["candidate_key"], best_ranking["candidate_key"]]
        )
    ]
    lines = [
        "# Domestic Surprise Effect-Size Sensitivity",
        "",
        f"Shadow-only analysis; production contract was not changed. Evaluated matches: `{matches}`.",
        "",
        "## Frozen Contract",
        "",
        f"- Production theta: `{production['domestic_surprise']['coefficient']}`",
        f"- Variance penalty: `{production['domestic_surprise']['variance_penalty']}`",
        f"- Production cap: `+/-{production['domestic_surprise']['max_abs_adjustment']}`",
        "- Five-season history, variance/consistency, exposure attenuation, Scale/H/K, goal margin, xG, progression and format draw were held fixed.",
        "- Candidate labels are retrospective response-curve diagnostics, not parameter promotions.",
        "",
        "## Decision Table",
        "",
        markdown_table(decision_table[display_columns]),
        "",
        "## Current",
        "",
        markdown_table(pd.DataFrame([current])),
        "",
        "## Best Loss Model",
        "",
        markdown_table(pd.DataFrame([best_loss])),
        "",
        "## Best Ranking Model",
        "",
        markdown_table(pd.DataFrame([best_ranking])),
        "",
        "## Dependency-Robust Uncertainty",
        "",
        markdown_table(selected_uncertainty),
        "",
        "## Best-Loss Competition Segments",
        "",
        markdown_table(best_competition),
        "",
        "## Exposure Response: Current",
        "",
        markdown_table(current_exposure),
        "",
        "## Exposure Response: Best Loss",
        "",
        markdown_table(best_exposure),
        "",
        "## Achievement Double-Counting Diagnostic",
        "",
        markdown_table(double_rows),
        "",
        "## Conclusion",
        "",
        f"Decision category: **{conclusion}**.",
        "",
        "A stronger coefficient is supported only if pooled loss, fold stability, dependency-aware uncertainty and ranking move together. The minimum point on this development-window surface is not an untouched-holdout production selection.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Domestic Surprise realized-effect sensitivity on the current production replay"
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    args = parser.parse_args()

    production_path = PRODUCTION_CONTRACT.resolve()
    production_hash_before = hashlib.sha256(production_path.read_bytes()).hexdigest()
    production = json.loads(production_path.read_text(encoding="utf-8"))
    core, parameters = validate_production_contract(production)
    dynamic = json.loads(DYNAMIC_MANIFEST.read_text(encoding="utf-8"))
    static_config = AOEuropeanEloConfig(**dynamic["static_config"])
    static_config.validate()
    events = read_events(EVENTS_PATH)
    reserve, _ = load_reserve_data(STATIC_DATA_ROOT, EVENTS_PATH, static_config)
    datasets = prepare_controlled_data(reserve, events)
    seasons = tuple(data.season for data in datasets)
    folds = expanding_folds(seasons)
    if len(folds) != 6:
        raise ValueError("Expected six expanding folds")
    evaluation_seasons = {test for _, test in folds}
    target = schedule_adjusted_team_performance(events)
    identity = load_team_season_identity()
    xg_map = load_xg_map(XG_DATA, datasets)
    features = pd.read_csv(FEATURES_PATH)
    production_domestic = load_domestic_adjustments(DOMESTIC_ADJUSTMENTS, datasets)

    configs = candidate_grid(production)
    production_config = VarianceDomesticSurpriseConfig(
        coefficient=float(production["domestic_surprise"]["coefficient"]),
        variance_penalty=float(production["domestic_surprise"]["variance_penalty"]),
        max_abs_adjustment=float(production["domestic_surprise"]["max_abs_adjustment"]),
        minimum_history_seasons=int(production["domestic_surprise"]["minimum_history_seasons"]),
    )
    current_key = candidate_key(production_config)
    config_by_key = {candidate_key(config): config for config in configs}

    evaluations = {}
    distribution_frames = []
    effect_rows = []
    metric_rows = []
    fold_frames = []
    competition_frames = []
    for index, config in enumerate(configs, start=1):
        key = candidate_key(config)
        adjustments = build_adjustments(features, config, static_config)
        adjustments = adjustments.merge(
            features[
                [
                    "season",
                    "team_id",
                    "league_strength",
                    "effective_european_exposure",
                    "history_seasons_available",
                ]
            ],
            on=["season", "team_id"],
            how="left",
            validate="one_to_one",
        )
        if adjustments[["league_strength", "effective_european_exposure"]].isna().any().any():
            raise ValueError("Domestic Surprise feature enrichment is incomplete")
        distribution, effect = effect_distribution(adjustments, config, static_config)
        rating_map = {
            (str(row.season), int(row.team_id)): float(row.adjusted_ao_first_elo)
            for row in adjustments.itertuples()
        }
        if key == current_key:
            errors = [
                abs(value - production_domestic[index_key])
                for index_key, value in rating_map.items()
            ]
            if max(errors, default=0.0) > 1e-9:
                raise ValueError("Generated production Domestic Surprise ratings do not match runtime input")
        arm = EvaluationArm(key, True, True, True, True, True)
        evaluation = evaluate_arm(
            datasets,
            arm,
            core=core,
            parameters=parameters,
            current_domestic=rating_map,
            baseline_domestic=rating_map,
            xg_map=xg_map,
            target=target,
        )
        evaluations[key] = evaluation
        metrics = model_metrics(
            evaluation, evaluation_seasons, target, identity, seasons
        )
        metric_rows.append({**effect, **metrics})
        folds_for_candidate = fold_metrics(evaluation, folds)
        folds_for_candidate.insert(0, "candidate_key", key)
        fold_frames.append(folds_for_candidate)
        competition_for_candidate = competition_metrics(evaluation, evaluation_seasons)
        competition_for_candidate.insert(0, "candidate_key", key)
        competition_frames.append(competition_for_candidate)
        distribution_frames.append(distribution)
        effect_rows.append(effect)
        print(f"  candidate {index}/{len(configs)}: {key}", flush=True)

    surface = add_baseline_deltas(pd.DataFrame(metric_rows), current_key)
    folds_all = add_fold_deltas(pd.concat(fold_frames, ignore_index=True), current_key)
    stability = summarize_fold_stability(folds_all)
    surface = surface.merge(stability, on="candidate_key", validate="one_to_one")
    distributions = pd.concat(distribution_frames, ignore_index=True)
    competition = pd.concat(competition_frames, ignore_index=True)
    current_competition = competition.loc[
        competition["candidate_key"].eq(current_key)
    ].set_index("competition")
    for metric in (
        "brier_1x2",
        "log_loss_1x2",
        "accuracy_1x2",
        "same_season_spearman",
        "same_season_pairwise_accuracy",
    ):
        competition[f"delta_vs_current_{metric}"] = competition.apply(
            lambda row: row[metric] - current_competition.loc[row["competition"], metric],
            axis=1,
        )

    best_loss, best_ranking = select_diagnostic_models(surface)
    decision_table = nearest_effect_models(surface, current_key)
    uncertainty_keys = list(
        dict.fromkeys(
            [
                current_key,
                str(best_loss["candidate_key"]),
                str(best_ranking["candidate_key"]),
                *decision_table["candidate_key"].astype(str).tolist(),
            ]
        )
    )
    uncertainty = dependency_uncertainty(
        evaluations,
        uncertainty_keys,
        current_key,
        evaluation_seasons,
        bootstrap_samples=args.bootstrap_samples,
    )
    conclusion = classify_conclusion(
        surface, current_key, best_loss, uncertainty
    )
    exposure = exposure_analysis(distributions)
    double_counting = achievement_double_counting(distributions)
    production_hash_after = hashlib.sha256(production_path.read_bytes()).hexdigest()
    safety = safety_audit(
        distributions,
        evaluations,
        config_by_key,
        production_hash_before,
        production_hash_after,
    )
    if not safety["passed"].all():
        failures = safety.loc[~safety["passed"], ["candidate_key", "check"]]
        raise ValueError(f"Sensitivity safety audit failed: {failures.to_dict(orient='records')}")

    top_positive = (
        distributions.sort_values(
            ["candidate_key", "initial_delta"], ascending=[True, False]
        )
        .groupby("candidate_key", sort=False)
        .head(15)
    )
    top_negative = (
        distributions.sort_values(
            ["candidate_key", "initial_delta"], ascending=[True, True]
        )
        .groupby("candidate_key", sort=False)
        .head(15)
    )
    cap_profiles = distributions.loc[
        distributions["positive_cap_hit"] | distributions["negative_cap_hit"]
    ].copy()
    response = surface.sort_values(
        ["changed_mean_abs_initial_delta", "cap", "theta"], kind="stable"
    ).reset_index(drop=True)

    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    decision_table.to_csv(output / "effect_size_decision_table.csv", index=False)
    surface.to_csv(output / "full_parameter_grid_results.csv", index=False)
    folds_all.to_csv(output / "fold_results.csv", index=False)
    distributions.to_csv(output / "adjustment_distribution.csv", index=False)
    exposure.to_csv(output / "exposure_band_analysis.csv", index=False)
    competition.to_csv(output / "competition_summary.csv", index=False)
    uncertainty.to_csv(output / "dependency_uncertainty.csv", index=False)
    top_positive.to_csv(output / "top_positive_adjustments.csv", index=False)
    top_negative.to_csv(output / "top_negative_adjustments.csv", index=False)
    cap_profiles.to_csv(output / "cap_hit_profiles.csv", index=False)
    double_counting.to_csv(output / "achievement_double_counting.csv", index=False)
    safety.to_csv(output / "safety_audit.csv", index=False)
    response.to_csv(output / "effect_size_response_curve.csv", index=False)
    write_response_plot(response, output / "effect_size_response_curve.svg")
    current = surface.loc[surface["candidate_key"].eq(current_key)].iloc[0]
    write_report(
        output / "effect_size_sensitivity_report.md",
        production=production,
        current=current,
        best_loss=best_loss,
        best_ranking=best_ranking,
        decision_table=decision_table,
        uncertainty=uncertainty,
        competition=competition,
        exposure=exposure,
        double_counting=double_counting,
        conclusion=conclusion,
        matches=int(current["matches"]),
    )
    manifest = {
        "analysis": "DOMESTIC_SURPRISE_REALIZED_EFFECT_SIZE_SENSITIVITY",
        "evaluation_only": True,
        "production_changed": False,
        "production_revision": production["production_revision"],
        "production_contract_sha256": production_hash_after,
        "candidate_count": len(configs),
        "evaluation_seasons": sorted(evaluation_seasons),
        "evaluation_matches": int(current["matches"]),
        "bootstrap_samples": args.bootstrap_samples,
        "current_candidate_key": current_key,
        "best_loss_candidate_key": str(best_loss["candidate_key"]),
        "best_ranking_candidate_key": str(best_ranking["candidate_key"]),
        "conclusion": conclusion,
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Conclusion: {conclusion}")
    print(f"Current: {current_key}")
    print(f"Best loss: {best_loss['candidate_key']}")
    print(f"Best ranking: {best_ranking['candidate_key']}")
    print(f"Report: {output / 'effect_size_sensitivity_report.md'}")


if __name__ == "__main__":
    main()
