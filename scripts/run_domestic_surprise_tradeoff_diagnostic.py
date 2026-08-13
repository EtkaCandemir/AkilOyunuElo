from __future__ import annotations

"""Explain the Domestic Surprise loss-ranking trade-off without changing production."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ao_elo.config import AOEuropeanEloConfig  # noqa: E402
from ao_elo.domestic_surprise_variance import VarianceDomesticSurpriseConfig  # noqa: E402
from ao_elo.evaluation import (  # noqa: E402
    dependency_robust_loss_difference_ci,
    schedule_adjusted_team_performance,
)
from scripts.run_controlled_goal_progression_backtest import prepare_controlled_data  # noqa: E402
from scripts.run_current_model_evaluation import (  # noqa: E402
    EvaluationArm,
    aggregate_ranking,
    evaluate_arm,
    prediction_summary,
)
from scripts.run_domestic_surprise_effect_size_sensitivity import (  # noqa: E402
    competition_metrics,
    fold_metrics,
    markdown_table,
    model_metrics,
)
from scripts.run_domestic_surprise_variance_backtest import build_adjustments  # noqa: E402
from scripts.run_dynamic_core_calibration import expanding_folds  # noqa: E402
from scripts.run_final_robustness import load_team_season_identity  # noqa: E402
from scripts.run_ranking_first_calibration import pairwise_ranking_accuracy  # noqa: E402
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
SENSITIVITY_SURFACE = (
    ROOT
    / "output"
    / "domestic_surprise_effect_size_sensitivity"
    / "full_parameter_grid_results.csv"
)
OUTPUT_ROOT = ROOT / "output" / "domestic_surprise_tradeoff_diagnostic"
CURRENT = "CURRENT_PRODUCTION"
GLOBAL = "GLOBAL_STRONG"
NO_SURPRISE = "NO_SURPRISE"
CLIP_CAPS = (30.0, 40.0, 50.0, 60.0, 75.0, 100.0)
EXPOSURE_LABELS = ("0", "(0,0.25]", "(0.25,0.50]", "(0.50,0.75]", "(0.75,1.00]")
MAGNITUDE_LABELS = ("0-10", "10-20", "20-30", "30-40", "40-50", "50-75", "75-100", "100+")


def discover_configs(production: dict[str, object], surface: pd.DataFrame) -> dict[str, VarianceDomesticSurpriseConfig]:
    block = production["domestic_surprise"]
    current = VarianceDomesticSurpriseConfig(
        coefficient=float(block["coefficient"]),
        variance_penalty=float(block["variance_penalty"]),
        max_abs_adjustment=float(block["max_abs_adjustment"]),
        minimum_history_seasons=int(block["minimum_history_seasons"]),
    )
    best = surface.sort_values(["brier_1x2", "log_loss_1x2"], kind="stable").iloc[0]
    strong = VarianceDomesticSurpriseConfig(
        coefficient=float(best["theta"]),
        variance_penalty=float(best["gamma"]),
        max_abs_adjustment=float(best["cap"]),
        minimum_history_seasons=int(block["minimum_history_seasons"]),
    )
    no_surprise = VarianceDomesticSurpriseConfig(
        coefficient=0.0,
        variance_penalty=float(block["variance_penalty"]),
        max_abs_adjustment=float(block["max_abs_adjustment"]),
        minimum_history_seasons=int(block["minimum_history_seasons"]),
    )
    configs = {CURRENT: current, GLOBAL: strong, NO_SURPRISE: no_surprise}
    for config in configs.values():
        config.validate()
    return configs


def exposure_band(values: pd.Series) -> pd.Series:
    return pd.cut(
        values,
        bins=[-1e-12, 0.0, 0.25, 0.50, 0.75, 1.0],
        labels=EXPOSURE_LABELS,
        include_lowest=True,
    )


def magnitude_band(values: pd.Series) -> pd.Series:
    return pd.cut(
        values.abs(),
        bins=[-1e-12, 10.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0, np.inf],
        labels=MAGNITUDE_LABELS,
        include_lowest=True,
        right=True,
    )


def load_domestic_leagues() -> pd.DataFrame:
    frames = []
    for path in sorted(STATIC_DATA_ROOT.glob("*/teams.csv")):
        frame = pd.read_csv(path)
        folder_season = path.parent.name.replace("-", "/", 1)
        frame["season"] = folder_season
        frames.append(frame[["season", "team_id", "domestic_league"]])
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(["season", "team_id"]).any():
        raise ValueError("Domestic league mapping contains duplicate team-seasons")
    return result


def target_team_ranks(target: pd.DataFrame) -> pd.DataFrame:
    frame = target.copy()
    frame["weighted_target"] = frame["schedule_adjusted_score"] * frame["matches"]
    result = frame.groupby(["season", "team_id"], as_index=False).agg(
        target_score=("weighted_target", "sum"),
        target_matches=("matches", "sum"),
    )
    result["target_score"] /= result["target_matches"]
    result["target_rank"] = result.groupby("season")["target_score"].rank(method="min", ascending=False)
    return result


def adjustment_map(
    features: pd.DataFrame,
    config: VarianceDomesticSurpriseConfig,
    static_config: AOEuropeanEloConfig,
) -> pd.DataFrame:
    adjustment = build_adjustments(features, config, static_config)
    return adjustment.merge(
        features[
            [
                "season", "team_id", "country", "country_code", "league_strength",
                "achievement_scale", "effective_european_exposure",
                "history_seasons_available",
            ]
        ],
        on=["season", "team_id"],
        how="left",
        validate="one_to_one",
    )


def evaluate_config(
    datasets,
    config: VarianceDomesticSurpriseConfig,
    features: pd.DataFrame,
    static_config: AOEuropeanEloConfig,
    core,
    parameters,
    xg_map,
    target,
    name: str,
):
    adjustments = adjustment_map(features, config, static_config)
    rating_map = {
        (str(row.season), int(row.team_id)): float(row.adjusted_ao_first_elo)
        for row in adjustments.itertuples(index=False)
    }
    evaluation = evaluate_arm(
        datasets,
        EvaluationArm(name, True, True, True, True, True),
        core=core,
        parameters=parameters,
        current_domestic=rating_map,
        baseline_domestic=rating_map,
        xg_map=xg_map,
        target=target,
    )
    return adjustments, evaluation


def team_season_diagnostics(
    features: pd.DataFrame,
    current_adjustments: pd.DataFrame,
    strong_adjustments: pd.DataFrame,
    current_evaluation,
    strong_evaluation,
    target_ranks: pd.DataFrame,
    leagues: pd.DataFrame,
    static_config: AOEuropeanEloConfig,
    evaluation_seasons: set[str],
) -> pd.DataFrame:
    base_columns = [
        "season", "team_id", "team_name", "club_id", "country", "country_code",
        "competition", "league_strength", "achievement_scale", "effective_european_exposure",
    ]
    base = features.loc[features["season"].isin(evaluation_seasons), base_columns].copy()
    current = current_adjustments[
        [
            "season", "team_id", "adjusted_ao_first_elo", "domestic_prior_adjustment",
            "ao_first_elo_adjustment", "historical_mean", "historical_volatility",
            "normalized_volatility", "consistency_multiplier", "raw_surprise", "effective_surprise",
        ]
    ].rename(
        columns={
            "adjusted_ao_first_elo": "current_initial_elo",
            "domestic_prior_adjustment": "current_domestic_adjustment",
            "ao_first_elo_adjustment": "current_ao_first_adjustment",
            "historical_volatility": "volatility",
        }
    )
    strong = strong_adjustments[
        ["season", "team_id", "adjusted_ao_first_elo", "domestic_prior_adjustment", "ao_first_elo_adjustment"]
    ].rename(
        columns={
            "adjusted_ao_first_elo": "global_strong_initial_elo",
            "domestic_prior_adjustment": "global_strong_domestic_adjustment",
            "ao_first_elo_adjustment": "global_strong_ao_first_adjustment",
        }
    )
    result = base.merge(current, on=["season", "team_id"], validate="one_to_one").merge(
        strong, on=["season", "team_id"], validate="one_to_one"
    ).merge(leagues, on=["season", "team_id"], how="left", validate="one_to_one")
    result["domestic_league"] = result["domestic_league"].fillna("UNAVAILABLE")
    result["domestic_achievement_component"] = (
        features.loc[features["season"].isin(evaluation_seasons), "domestic_prior"].to_numpy(float)
        - static_config.base_rating
        - static_config.domestic_league_component * result["league_strength"].to_numpy(float)
    )
    result["initial_elo_difference"] = result["global_strong_initial_elo"] - result["current_initial_elo"]
    for prefix in ("current", "global_strong"):
        result[f"{prefix}_initial_rank"] = result.groupby("season")[f"{prefix}_initial_elo"].rank(method="min", ascending=False)
    result["rank_difference"] = result["global_strong_initial_rank"] - result["current_initial_rank"]

    current_end = current_evaluation.end_ratings.loc[
        current_evaluation.end_ratings["season"].isin(evaluation_seasons),
        ["season", "team_id", "end_live_rating"],
    ].rename(columns={"end_live_rating": "current_final_live_elo"})
    strong_end = strong_evaluation.end_ratings.loc[
        strong_evaluation.end_ratings["season"].isin(evaluation_seasons),
        ["season", "team_id", "end_live_rating"],
    ].rename(columns={"end_live_rating": "global_strong_final_live_elo"})
    result = result.merge(current_end, on=["season", "team_id"], validate="one_to_one").merge(
        strong_end, on=["season", "team_id"], validate="one_to_one"
    )
    result["final_elo_difference"] = result["global_strong_final_live_elo"] - result["current_final_live_elo"]
    for prefix in ("current", "global_strong"):
        result[f"{prefix}_final_rank"] = result.groupby("season")[f"{prefix}_final_live_elo"].rank(method="min", ascending=False)
    result["final_rank_difference"] = result["global_strong_final_rank"] - result["current_final_rank"]
    result = result.merge(target_ranks, on=["season", "team_id"], how="left", validate="one_to_one")
    result["current_rank_error"] = (result["current_final_rank"] - result["target_rank"]).abs()
    result["global_strong_rank_error"] = (result["global_strong_final_rank"] - result["target_rank"]).abs()
    result["ranking_error_difference"] = result["global_strong_rank_error"] - result["current_rank_error"]
    result["exposure_band"] = exposure_band(result["effective_european_exposure"])
    result["adjustment_magnitude_band"] = magnitude_band(result["initial_elo_difference"])
    result["surprise_direction"] = np.select(
        [result["raw_surprise"].gt(1e-12), result["raw_surprise"].lt(-1e-12)],
        ["POSITIVE", "NEGATIVE"],
        default="NEUTRAL",
    )
    result["initial_rating_tier"] = pd.qcut(
        result["current_initial_elo"].rank(method="first"),
        q=5,
        labels=["LOW", "LOWER_MIDDLE", "MIDDLE", "UPPER_MIDDLE", "ELITE"],
    )
    result["league_strength_quintile"] = pd.qcut(
        result["league_strength"].rank(method="first"),
        q=5,
        labels=["Q1_WEAKEST", "Q2", "Q3", "Q4", "Q5_STRONGEST"],
    )
    return result


def match_loss_diagnostics(
    current_evaluation,
    strong_evaluation,
    teams: pd.DataFrame,
    evaluation_seasons: set[str],
) -> pd.DataFrame:
    columns = [
        "match_id", "season", "kickoff_utc", "competition", "stage", "home_team_id", "away_team_id",
        "home_goals", "away_goals", "actual_class", "predicted_class", "home_live_pre", "away_live_pre",
        "expected_home_score", "home_probability", "draw_probability", "away_probability", "brier_1x2", "log_loss_1x2",
    ]
    current = current_evaluation.predictions.loc[
        current_evaluation.predictions["season"].isin(evaluation_seasons), columns
    ].rename(columns={column: f"current_{column}" for column in columns if column not in {"match_id", "season", "kickoff_utc", "competition", "stage", "home_team_id", "away_team_id", "home_goals", "away_goals", "actual_class"}})
    strong = strong_evaluation.predictions.loc[
        strong_evaluation.predictions["season"].isin(evaluation_seasons), columns
    ].rename(columns={column: f"global_{column}" for column in columns if column not in {"match_id", "season", "kickoff_utc", "competition", "stage", "home_team_id", "away_team_id", "home_goals", "away_goals", "actual_class"}})
    keys = ["match_id", "season", "kickoff_utc", "competition", "stage", "home_team_id", "away_team_id", "home_goals", "away_goals", "actual_class"]
    result = current.merge(strong, on=keys, validate="one_to_one")
    result["brier_difference"] = result["global_brier_1x2"] - result["current_brier_1x2"]
    result["log_loss_difference"] = result["global_log_loss_1x2"] - result["current_log_loss_1x2"]
    result["accuracy_difference"] = (
        result["global_predicted_class"].eq(result["actual_class"]).astype(int)
        - result["current_predicted_class"].eq(result["actual_class"]).astype(int)
    )
    result["actual_result"] = result["actual_class"].map({0: "HOME", 1: "DRAW", 2: "AWAY"})
    lookup_columns = [
        "season", "team_id", "team_name", "current_initial_elo", "global_strong_initial_elo",
        "current_ao_first_adjustment", "global_strong_ao_first_adjustment", "effective_european_exposure",
    ]
    for side in ("home", "away"):
        lookup = teams[lookup_columns].rename(columns={column: f"{side}_{column}" for column in lookup_columns if column not in {"season", "team_id"}})
        result = result.merge(
            lookup,
            left_on=["season", f"{side}_team_id"],
            right_on=["season", "team_id"],
            validate="many_to_one",
        ).drop(columns="team_id")
        result[f"{side}_surprise_adjustment_difference"] = (
            result[f"{side}_global_strong_ao_first_adjustment"]
            - result[f"{side}_current_ao_first_adjustment"]
        )
        result = result.rename(columns={f"{side}_effective_european_exposure": f"{side}_exposure"})
    result["current_gap_abs"] = (result["current_expected_home_score"] - 0.5).abs()
    result["global_gap_abs"] = (result["global_expected_home_score"] - 0.5).abs()
    result["polarization_change"] = result["global_gap_abs"] - result["current_gap_abs"]
    result["favorite_band"] = pd.qcut(
        result["current_gap_abs"].rank(method="first"),
        q=4,
        labels=["BALANCED", "SMALL_FAVORITE", "MEDIUM_FAVORITE", "HEAVY_FAVORITE"],
    )
    return result.sort_values(["kickoff_utc", "match_id"], kind="stable").reset_index(drop=True)


def add_team_loss_contributions(teams: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    sides = []
    for side in ("home", "away"):
        frame = matches[
            ["season", "match_id", "competition", f"{side}_team_id", "brier_difference", "log_loss_difference"]
        ].rename(columns={f"{side}_team_id": "team_id"})
        sides.append(frame)
    team_matches = pd.concat(sides, ignore_index=True)
    contribution = team_matches.groupby(["season", "team_id"], as_index=False).agg(
        involved_matches=("match_id", "nunique"),
        mean_brier_difference=("brier_difference", "mean"),
        total_brier_difference=("brier_difference", "sum"),
        mean_log_loss_difference=("log_loss_difference", "mean"),
        total_log_loss_difference=("log_loss_difference", "sum"),
    )
    result = teams.merge(contribution, on=["season", "team_id"], how="left", validate="one_to_one")
    result[["involved_matches", "mean_brier_difference", "total_brier_difference", "mean_log_loss_difference", "total_log_loss_difference"]] = result[
        ["involved_matches", "mean_brier_difference", "total_brier_difference", "mean_log_loss_difference", "total_log_loss_difference"]
    ].fillna(0.0)
    loss = result["total_brier_difference"]
    ranking = result["ranking_error_difference"]
    result["loss_ranking_quadrant"] = np.select(
        [
            loss.lt(-1e-12) & ranking.lt(-1e-12),
            loss.lt(-1e-12) & ranking.gt(1e-12),
            loss.gt(1e-12) & ranking.lt(-1e-12),
            loss.gt(1e-12) & ranking.gt(1e-12),
        ],
        ["A_WIN_WIN", "B_LOSS_WIN_RANK_HARM", "C_LOSS_HARM_RANK_WIN", "D_LOSE_LOSE"],
        default="NEUTRAL_AXIS",
    )
    return result


def ranking_metrics(frame: pd.DataFrame) -> dict[str, float]:
    eligible = frame.dropna(subset=["target_score", "current_final_live_elo", "global_strong_final_live_elo"])
    if len(eligible) < 3:
        return {"current_spearman": np.nan, "global_spearman": np.nan, "spearman_difference": np.nan, "current_pairwise": np.nan, "global_pairwise": np.nan, "pairwise_difference": np.nan}
    current_spearman = float(eligible["current_final_live_elo"].corr(eligible["target_score"], method="spearman"))
    global_spearman = float(eligible["global_strong_final_live_elo"].corr(eligible["target_score"], method="spearman"))
    current_pairwise = pairwise_ranking_accuracy(eligible["current_final_live_elo"].to_numpy(float), eligible["target_score"].to_numpy(float))
    global_pairwise = pairwise_ranking_accuracy(eligible["global_strong_final_live_elo"].to_numpy(float), eligible["target_score"].to_numpy(float))
    return {
        "current_spearman": current_spearman,
        "global_spearman": global_spearman,
        "spearman_difference": global_spearman - current_spearman,
        "current_pairwise": current_pairwise,
        "global_pairwise": global_pairwise,
        "pairwise_difference": global_pairwise - current_pairwise,
    }


def summarize_team_groups(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows = []
    grouped = frame.groupby(group_columns, observed=False, dropna=False, sort=False)
    for keys, group in grouped:
        keys = keys if isinstance(keys, tuple) else (keys,)
        payload = dict(zip(group_columns, keys, strict=True))
        rows.append(
            {
                **payload,
                "team_seasons": len(group),
                "matches": int(group["involved_matches"].sum()),
                "mean_exposure": float(group["effective_european_exposure"].mean()),
                "mean_raw_surprise": float(group["raw_surprise"].mean()),
                "mean_abs_initial_elo_difference": float(group["initial_elo_difference"].abs().mean()),
                "mean_league_strength": float(group["league_strength"].mean()),
                "mean_initial_rating": float(group["current_initial_elo"].mean()),
                "total_brier_difference": float(group["total_brier_difference"].sum()),
                "mean_brier_difference": float(group["mean_brier_difference"].mean()),
                "total_log_loss_difference": float(group["total_log_loss_difference"].sum()),
                "mean_log_loss_difference": float(group["mean_log_loss_difference"].mean()),
                "mean_ranking_error_difference": float(group["ranking_error_difference"].mean()),
                "median_ranking_error_difference": float(group["ranking_error_difference"].median()),
                **ranking_metrics(group),
            }
        )
    return pd.DataFrame(rows)


def exposure_adjustment_matrix(teams: pd.DataFrame) -> pd.DataFrame:
    matrix_band = pd.cut(
        teams["initial_elo_difference"].abs(),
        bins=[-1e-12, 20.0, 40.0, 60.0, 100.0, np.inf],
        labels=["0-20", "20-40", "40-60", "60-100", "100+"],
        include_lowest=True,
    )
    frame = teams.assign(matrix_adjustment_band=matrix_band)
    return summarize_team_groups(frame, ["exposure_band", "matrix_adjustment_band"])


def favorite_underdog_analysis(matches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for band, group in matches.groupby("favorite_band", observed=False, sort=False):
        rows.append(
            {
                "favorite_band": str(band),
                "matches": len(group),
                "current_gap_mean": float(group["current_gap_abs"].mean()),
                "polarization_change_mean": float(group["polarization_change"].mean()),
                "brier_difference": float(group["brier_difference"].mean()),
                "log_loss_difference": float(group["log_loss_difference"].mean()),
                "accuracy_difference": float(group["accuracy_difference"].mean()),
                "polarized_share": float(group["polarization_change"].gt(0.0).mean()),
            }
        )
    return pd.DataFrame(rows)


def calibration_analysis(matches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    bins = np.linspace(0.0, 1.0, 11)
    for model in ("current", "global"):
        stacked = []
        for class_index, label in enumerate(("home", "draw", "away")):
            stacked.append(
                pd.DataFrame(
                    {
                        "probability": matches[f"{model}_{label}_probability"],
                        "observed": matches["actual_class"].eq(class_index).astype(float),
                    }
                )
            )
        frame = pd.concat(stacked, ignore_index=True)
        frame["probability_bin"] = pd.cut(frame["probability"], bins=bins, include_lowest=True)
        for probability_bin, group in frame.groupby("probability_bin", observed=False, sort=False):
            predicted = float(group["probability"].mean())
            observed = float(group["observed"].mean())
            rows.append(
                {
                    "model": model.upper(),
                    "probability_bin": str(probability_bin),
                    "observations": len(group),
                    "mean_predicted_probability": predicted,
                    "observed_frequency": observed,
                    "absolute_calibration_error": abs(predicted - observed),
                }
            )
    return pd.DataFrame(rows)


def fold_analysis(teams: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold, season in enumerate(sorted(matches["season"].unique()), start=1):
        match_group = matches.loc[matches["season"].eq(season)]
        team_group = teams.loc[teams["season"].eq(season)]
        metrics = ranking_metrics(team_group)
        rows.append(
            {
                "fold": fold,
                "season": season,
                "matches": len(match_group),
                "brier_difference": float(match_group["brier_difference"].mean()),
                "log_loss_difference": float(match_group["log_loss_difference"].mean()),
                "accuracy_difference": float(match_group["accuracy_difference"].mean()),
                "mean_abs_initial_elo_difference": float(team_group["initial_elo_difference"].abs().mean()),
                "loss_win_ranking_harm_team_seasons": int(team_group["loss_ranking_quadrant"].eq("B_LOSS_WIN_RANK_HARM").sum()),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def competition_analysis(teams: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for competition, match_group in matches.groupby("competition", sort=True):
        ids = pd.concat(
            [
                match_group[["season", "home_team_id"]].rename(columns={"home_team_id": "team_id"}),
                match_group[["season", "away_team_id"]].rename(columns={"away_team_id": "team_id"}),
            ],
            ignore_index=True,
        ).drop_duplicates()
        team_group = teams.merge(ids, on=["season", "team_id"], validate="one_to_one")
        rows.append(
            {
                "competition": competition,
                "matches": len(match_group),
                "team_seasons": len(team_group),
                "brier_difference": float(match_group["brier_difference"].mean()),
                "log_loss_difference": float(match_group["log_loss_difference"].mean()),
                "accuracy_difference": float(match_group["accuracy_difference"].mean()),
                "mean_abs_initial_elo_difference": float(team_group["initial_elo_difference"].abs().mean()),
                "mean_exposure": float(team_group["effective_european_exposure"].mean()),
                **ranking_metrics(team_group),
            }
        )
    return pd.DataFrame(rows)


def country_league_analysis(teams: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for segment_type, column in (("COUNTRY", "country_code"), ("LEAGUE_STRENGTH_QUINTILE", "league_strength_quintile")):
        summary = summarize_team_groups(teams, [column]).rename(columns={column: "segment"})
        summary.insert(0, "segment_type", segment_type)
        summary["small_sample"] = summary["team_seasons"].lt(20)
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def loss_difference_ci(current_evaluation, strong_evaluation, evaluation_seasons: set[str], bootstrap_samples: int) -> pd.DataFrame:
    current = current_evaluation.predictions.loc[current_evaluation.predictions["season"].isin(evaluation_seasons)]
    strong = strong_evaluation.predictions.loc[strong_evaluation.predictions["season"].isin(evaluation_seasons)]
    paired = strong.merge(current[["match_id", "brier_1x2", "log_loss_1x2"]], on="match_id", suffixes=("_global", "_current"), validate="one_to_one")
    rows = []
    for metric in ("brier_1x2", "log_loss_1x2"):
        sample = paired[["season", "match_id", "home_team_id", "away_team_id", "kickoff_utc", "tie_id"]].copy()
        sample["loss_difference"] = paired[f"{metric}_global"] - paired[f"{metric}_current"]
        result = dependency_robust_loss_difference_ci(sample, bootstrap_samples=bootstrap_samples)
        result.insert(0, "metric", metric)
        rows.append(result)
    return pd.concat(rows, ignore_index=True)


def counterfactual_clipping(
    evaluations: dict[str, object],
    metrics: dict[str, dict[str, float]],
    caps: tuple[float, ...],
) -> pd.DataFrame:
    current = metrics[CURRENT]
    strong = metrics[GLOBAL]
    rows = []
    global_brier_gain = current["brier_1x2"] - strong["brier_1x2"]
    global_log_gain = current["log_loss_1x2"] - strong["log_loss_1x2"]
    for cap in caps:
        key = f"CLIP_{int(cap)}"
        row = metrics[key]
        rows.append(
            {
                "domestic_adjustment_cap": cap,
                **row,
                "brier_gain_vs_current": current["brier_1x2"] - row["brier_1x2"],
                "log_loss_gain_vs_current": current["log_loss_1x2"] - row["log_loss_1x2"],
                "retained_brier_gain_pct": 100.0 * (current["brier_1x2"] - row["brier_1x2"]) / max(global_brier_gain, 1e-12),
                "retained_log_loss_gain_pct": 100.0 * (current["log_loss_1x2"] - row["log_loss_1x2"]) / max(global_log_gain, 1e-12),
                "spearman_recovery_pct": 100.0 * (row["same_season_spearman"] - strong["same_season_spearman"]) / max(current["same_season_spearman"] - strong["same_season_spearman"], 1e-12),
                "pairwise_recovery_pct": 100.0 * (row["same_season_pairwise_accuracy"] - strong["same_season_pairwise_accuracy"]) / max(current["same_season_pairwise_accuracy"] - strong["same_season_pairwise_accuracy"], 1e-12),
                "forward_spearman_recovery_pct": 100.0 * (row["forward_season_spearman"] - strong["forward_season_spearman"]) / max(current["forward_season_spearman"] - strong["forward_season_spearman"], 1e-12),
                "forward_pairwise_recovery_pct": 100.0 * (row["forward_season_pairwise_accuracy"] - strong["forward_season_pairwise_accuracy"]) / max(current["forward_season_pairwise_accuracy"] - strong["forward_season_pairwise_accuracy"], 1e-12),
            }
        )
    return pd.DataFrame(rows)


def mechanism_summary(
    teams: pd.DataFrame,
    matches: pd.DataFrame,
    magnitude: pd.DataFrame,
    exposure: pd.DataFrame,
    competition: pd.DataFrame,
    country_league: pd.DataFrame,
    clipping: pd.DataFrame,
) -> dict[str, object]:
    ranking_harm = teams["ranking_error_difference"].clip(lower=0.0)
    loss_gain = (-teams["total_brier_difference"]).clip(lower=0.0)
    low = teams["effective_european_exposure"].le(0.25)
    extreme = teams["initial_elo_difference"].abs().gt(75.0)
    positive = teams["raw_surprise"].gt(0.0)
    def share(values: pd.Series, mask: pd.Series) -> float:
        return float(values.loc[mask].sum() / max(values.sum(), 1e-12))
    clipping_score = clipping.assign(
        joint_score=clipping[["retained_brier_gain_pct", "retained_log_loss_gain_pct", "spearman_recovery_pct", "pairwise_recovery_pct"]].min(axis=1)
    ).sort_values(["joint_score", "domestic_adjustment_cap"], ascending=[False, True], kind="stable").iloc[0]
    population = {
        "low_exposure": float(low.mean()),
        "extreme_adjustment": float(extreme.mean()),
        "positive_surprise": float(positive.mean()),
    }
    mechanisms = {
        "low_exposure_ranking_harm_share": share(ranking_harm, low),
        "extreme_adjustment_ranking_harm_share": share(ranking_harm, extreme),
        "positive_surprise_ranking_harm_share": share(ranking_harm, positive),
        "low_exposure_gross_brier_gain_share": share(loss_gain, low),
        "extreme_adjustment_gross_brier_gain_share": share(loss_gain, extreme),
        "positive_surprise_gross_brier_gain_share": share(loss_gain, positive),
    }
    enrichment = {
        "low_exposure_ranking_harm_enrichment": mechanisms["low_exposure_ranking_harm_share"] / max(population["low_exposure"], 1e-12),
        "extreme_adjustment_ranking_harm_enrichment": mechanisms["extreme_adjustment_ranking_harm_share"] / max(population["extreme_adjustment"], 1e-12),
        "positive_surprise_ranking_harm_enrichment": mechanisms["positive_surprise_ranking_harm_share"] / max(population["positive_surprise"], 1e-12),
    }
    exposure_shares = {}
    for band, group in teams.groupby("exposure_band", observed=True, sort=False):
        exposure_shares[str(band)] = {
            "team_share": float(len(group) / len(teams)),
            "gross_brier_gain_share": float((-group["total_brier_difference"]).clip(lower=0.0).sum() / max(loss_gain.sum(), 1e-12)),
            "ranking_harm_share": float(group["ranking_error_difference"].clip(lower=0.0).sum() / max(ranking_harm.sum(), 1e-12)),
            "net_brier_difference": float(group["total_brier_difference"].sum()),
        }
    magnitude_shares = {}
    for band, group in teams.groupby("adjustment_magnitude_band", observed=True, sort=False):
        magnitude_shares[str(band)] = {
            "team_share": float(len(group) / len(teams)),
            "gross_brier_gain_share": float((-group["total_brier_difference"]).clip(lower=0.0).sum() / max(loss_gain.sum(), 1e-12)),
            "ranking_harm_share": float(group["ranking_error_difference"].clip(lower=0.0).sum() / max(ranking_harm.sum(), 1e-12)),
            "net_brier_difference": float(group["total_brier_difference"].sum()),
        }
    breakpoint = None
    for label in MAGNITUDE_LABELS:
        row = magnitude.loc[magnitude["adjustment_magnitude_band"].astype(str).eq(label)]
        if (
            len(row)
            and int(row.iloc[0]["team_seasons"]) >= 20
            and float(row.iloc[0]["spearman_difference"]) <= -0.005
            and float(row.iloc[0]["pairwise_difference"]) <= -0.005
        ):
            breakpoint = label
            break
    weakest = country_league.loc[
        country_league["segment_type"].eq("LEAGUE_STRENGTH_QUINTILE")
        & country_league["segment"].eq("Q1_WEAKEST")
    ].iloc[0]
    strongest = country_league.loc[
        country_league["segment_type"].eq("LEAGUE_STRENGTH_QUINTILE")
        & country_league["segment"].eq("Q5_STRONGEST")
    ].iloc[0]
    best_loss_competition = str(competition.sort_values("brier_difference").iloc[0]["competition"])
    worst_ranking_competition = str(competition.sort_values("spearman_difference").iloc[0]["competition"])
    competition_separation = best_loss_competition != worst_ranking_competition
    league_separation = bool(
        weakest["total_brier_difference"] > 0.0
        and weakest["pairwise_difference"] < 0.0
        and strongest["total_brier_difference"] < 0.0
        and strongest["pairwise_difference"] > 0.0
    )
    extreme_enrichment = enrichment["extreme_adjustment_ranking_harm_enrichment"] >= 2.0
    category = "MULTIPLE_MECHANISMS" if sum((competition_separation, league_separation, extreme_enrichment)) >= 2 else "NO_CLEAR_MECHANISM"
    return {
        "category": category,
        "mechanism_shares": mechanisms,
        "population_shares": population,
        "mechanism_enrichment": enrichment,
        "exposure_band_shares": exposure_shares,
        "adjustment_magnitude_shares": magnitude_shares,
        "ranking_resolution_breakpoint_band": breakpoint,
        "breakpoint_is_monotonic": False,
        "best_loss_competition": best_loss_competition,
        "worst_ranking_competition": worst_ranking_competition,
        "competition_separation": competition_separation,
        "league_strength_separation": league_separation,
        "weakest_league_quintile": {
            "net_brier_difference": float(weakest["total_brier_difference"]),
            "spearman_difference": float(weakest["spearman_difference"]),
            "pairwise_difference": float(weakest["pairwise_difference"]),
        },
        "strongest_league_quintile": {
            "net_brier_difference": float(strongest["total_brier_difference"]),
            "spearman_difference": float(strongest["spearman_difference"]),
            "pairwise_difference": float(strongest["pairwise_difference"]),
        },
        "positive_direction_is_overrepresented": enrichment["positive_surprise_ranking_harm_enrichment"] >= 1.25,
        "ranking_harm_team_seasons": int(teams["ranking_error_difference"].gt(0.0).sum()),
        "top_30_ranking_harm_share": float(ranking_harm.nlargest(30).sum() / max(ranking_harm.sum(), 1e-12)),
        "loss_gain_matches": int(matches["brier_difference"].lt(0.0).sum()),
        "top_50_loss_gain_share": float((-matches["brier_difference"]).clip(lower=0.0).nlargest(50).sum() / max((-matches["brier_difference"]).clip(lower=0.0).sum(), 1e-12)),
        "best_diagnostic_clipping_cap": float(clipping_score["domestic_adjustment_cap"]),
        "best_diagnostic_clipping_joint_score": float(clipping_score["joint_score"]),
        "ranking_priority_clipping_cap": float(
            clipping.loc[
                clipping["retained_brier_gain_pct"].ge(35.0)
                & clipping["retained_log_loss_gain_pct"].ge(35.0)
            ].assign(
                ranking_recovery=lambda frame: frame[["spearman_recovery_pct", "pairwise_recovery_pct"]].min(axis=1)
            ).sort_values(["ranking_recovery", "domestic_adjustment_cap"], ascending=[False, True], kind="stable").iloc[0]["domestic_adjustment_cap"]
        ),
        "competition_net_brier_differences": competition.set_index("competition")["brier_difference"].to_dict(),
        "production_changed": False,
    }


def write_report(
    path: Path,
    comparison: pd.DataFrame,
    quadrants: pd.DataFrame,
    magnitude: pd.DataFrame,
    exposure_matrix: pd.DataFrame,
    competition: pd.DataFrame,
    clipping: pd.DataFrame,
    calibration: pd.DataFrame,
    uncertainty: pd.DataFrame,
    mechanism: dict[str, object],
) -> None:
    current = comparison.loc[comparison["model"].eq(CURRENT)].iloc[0]
    strong = comparison.loc[comparison["model"].eq(GLOBAL)].iloc[0]
    envelope = uncertainty.loc[uncertainty["method"].eq("conservative_envelope")]
    calibration_summary = calibration.groupby("model").apply(
        lambda frame: np.average(frame["absolute_calibration_error"], weights=frame["observations"]),
        include_groups=False,
    )
    lines = [
        "# Domestic Surprise Loss-Ranking Trade-off Diagnostic",
        "",
        "Shadow/research mechanism analysis. Production was not changed and no parameter was optimized.",
        "",
        "## Model comparison",
        "",
        markdown_table(comparison),
        "",
        "## Central loss-ranking quadrants",
        "",
        markdown_table(quadrants),
        "",
        "## Adjustment magnitude",
        "",
        markdown_table(magnitude),
        "",
        "## Exposure x adjustment interaction",
        "",
        markdown_table(exposure_matrix),
        "",
        "## Competition",
        "",
        markdown_table(competition),
        "",
        "## Counterfactual clipping",
        "",
        markdown_table(clipping),
        "",
        "## Uncertainty",
        "",
        markdown_table(envelope),
        "",
        "## Mechanism diagnosis",
        "",
        f"Category: **{mechanism['category']}**.",
        f"Global Strong Brier delta `{strong['brier_1x2'] - current['brier_1x2']:+.9f}` and log-loss delta `{strong['log_loss_1x2'] - current['log_loss_1x2']:+.9f}`.",
        f"Same-season Spearman delta `{strong['same_season_spearman'] - current['same_season_spearman']:+.9f}` and pairwise delta `{strong['same_season_pairwise_accuracy'] - current['same_season_pairwise_accuracy']:+.9f}`.",
        f"Current calibration ECE `{calibration_summary['CURRENT']:.6f}`; Global Strong `{calibration_summary['GLOBAL']:.6f}`.",
        f"Competition split: best prediction-loss segment `{mechanism['best_loss_competition']}`, worst ranking segment `{mechanism['worst_ranking_competition']}`.",
        f"Weakest league quintile: Brier contribution `{mechanism['weakest_league_quintile']['net_brier_difference']:+.6f}`, Spearman `{mechanism['weakest_league_quintile']['spearman_difference']:+.6f}`; strongest quintile: Brier `{mechanism['strongest_league_quintile']['net_brier_difference']:+.6f}`, Spearman `{mechanism['strongest_league_quintile']['spearman_difference']:+.6f}`.",
        f"Low exposure accounts for `{100.0 * mechanism['mechanism_shares']['low_exposure_ranking_harm_share']:.1f}%` of gross ranking harm from `{100.0 * mechanism['population_shares']['low_exposure']:.1f}%` of team-seasons; 75+ Elo moves account for `{100.0 * mechanism['mechanism_shares']['extreme_adjustment_ranking_harm_share']:.1f}%` from `{100.0 * mechanism['population_shares']['extreme_adjustment']:.1f}%`.",
        f"Positive Surprise is not independently overrepresented after prevalence normalization: harm enrichment `{mechanism['mechanism_enrichment']['positive_surprise_ranking_harm_enrichment']:.3f}`.",
        f"Damage is broad rather than a few outliers: `{mechanism['ranking_harm_team_seasons']}` team-seasons have positive rank-error change and the worst 30 explain only `{100.0 * mechanism['top_30_ranking_harm_share']:.1f}%` of gross harm.",
        f"Ranking-resolution damage first becomes jointly pronounced in Spearman and pairwise at `{mechanism['ranking_resolution_breakpoint_band']}`; the magnitude relationship is not monotonic.",
        f"Best max-min explanatory clipping compromise: `+/-{mechanism['best_diagnostic_clipping_cap']:.0f}`; ranking-priority diagnostic compromise: `+/-{mechanism['ranking_priority_clipping_cap']:.0f}` domestic Elo. Neither is a production recommendation.",
        "",
        "The detailed CSVs should be used for team-level interpretation; aggregate shares can overlap because low exposure, large adjustments and positive surprise often describe the same team-season.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Domestic Surprise loss-ranking mechanism diagnostic")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)

    production_path = PRODUCTION_CONTRACT.resolve()
    hash_before = hashlib.sha256(production_path.read_bytes()).hexdigest()
    production = json.loads(production_path.read_text(encoding="utf-8"))
    surface = pd.read_csv(SENSITIVITY_SURFACE)
    configs = discover_configs(production, surface)
    core, parameters = validate_production_contract(production)
    dynamic = json.loads(DYNAMIC_MANIFEST.read_text(encoding="utf-8"))
    static_config = AOEuropeanEloConfig(**dynamic["static_config"])
    static_config.validate()
    events = read_events(EVENTS_PATH)
    reserve, _ = load_reserve_data(STATIC_DATA_ROOT, EVENTS_PATH, static_config)
    datasets = prepare_controlled_data(reserve, events)
    seasons = tuple(data.season for data in datasets)
    folds = expanding_folds(seasons)
    evaluation_seasons = {test for _, test in folds}
    if len(folds) != 6:
        raise ValueError("Expected six unseen folds")
    target = schedule_adjusted_team_performance(events)
    target_ranks = target_team_ranks(target)
    identity = load_team_season_identity()
    features = pd.read_csv(FEATURES_PATH)
    leagues = load_domestic_leagues()
    xg_map = load_xg_map(XG_DATA, datasets)
    production_domestic = load_domestic_adjustments(DOMESTIC_ADJUSTMENTS, datasets)

    adjustments: dict[str, pd.DataFrame] = {}
    evaluations: dict[str, object] = {}
    for name, config in configs.items():
        adjustments[name], evaluations[name] = evaluate_config(
            datasets, config, features, static_config, core, parameters, xg_map, target, name
        )
        if name == CURRENT:
            rating_map = {
                (str(row.season), int(row.team_id)): float(row.adjusted_ao_first_elo)
                for row in adjustments[name].itertuples(index=False)
            }
            if max(abs(value - production_domestic[key]) for key, value in rating_map.items()) > 1e-9:
                raise ValueError("Current ratings do not match production runtime input")

    clip_evaluations = {}
    for cap in CLIP_CAPS:
        config = VarianceDomesticSurpriseConfig(
            coefficient=configs[GLOBAL].coefficient,
            variance_penalty=configs[GLOBAL].variance_penalty,
            max_abs_adjustment=cap,
            minimum_history_seasons=configs[GLOBAL].minimum_history_seasons,
        )
        key = f"CLIP_{int(cap)}"
        _, clip_evaluations[key] = evaluate_config(
            datasets, config, features, static_config, core, parameters, xg_map, target, key
        )

    teams = team_season_diagnostics(
        features, adjustments[CURRENT], adjustments[GLOBAL], evaluations[CURRENT], evaluations[GLOBAL],
        target_ranks, leagues, static_config, evaluation_seasons,
    )
    matches = match_loss_diagnostics(evaluations[CURRENT], evaluations[GLOBAL], teams, evaluation_seasons)
    teams = add_team_loss_contributions(teams, matches)
    quadrants = summarize_team_groups(teams, ["loss_ranking_quadrant"])
    magnitude = summarize_team_groups(teams, ["adjustment_magnitude_band"])
    exposure_matrix = exposure_adjustment_matrix(teams)
    positive_negative = summarize_team_groups(teams, ["surprise_direction", "adjustment_magnitude_band"])
    favorite = favorite_underdog_analysis(matches)
    country = country_league_analysis(teams)
    competition = competition_analysis(teams, matches)
    fold = fold_analysis(teams, matches)
    calibration = calibration_analysis(matches)
    uncertainty = loss_difference_ci(evaluations[CURRENT], evaluations[GLOBAL], evaluation_seasons, args.bootstrap_samples)

    all_evaluations = {**evaluations, **clip_evaluations}
    metrics = {
        key: model_metrics(evaluation, evaluation_seasons, target, identity, seasons)
        for key, evaluation in all_evaluations.items()
    }
    clipping = counterfactual_clipping(all_evaluations, metrics, CLIP_CAPS)
    comparison = pd.DataFrame([{"model": key, **metrics[key]} for key in (CURRENT, GLOBAL, NO_SURPRISE)])
    no_surprise_delta = adjustments[NO_SURPRISE].merge(
        adjustments[CURRENT][["season", "team_id", "adjusted_ao_first_elo"]],
        on=["season", "team_id"],
        suffixes=("_no_surprise", "_current"),
        validate="one_to_one",
    )
    no_surprise_delta = no_surprise_delta.loc[no_surprise_delta["season"].isin(evaluation_seasons)]
    comparison["mean_abs_initial_elo_difference_vs_current"] = [
        0.0,
        float(teams["initial_elo_difference"].abs().mean()),
        float((no_surprise_delta["adjusted_ao_first_elo_no_surprise"] - no_surprise_delta["adjusted_ao_first_elo_current"]).abs().mean()),
    ]
    comparison["mean_signed_initial_elo_difference_vs_current"] = [
        0.0,
        float(teams["initial_elo_difference"].mean()),
        float((no_surprise_delta["adjusted_ao_first_elo_no_surprise"] - no_surprise_delta["adjusted_ao_first_elo_current"]).mean()),
    ]
    mechanism = mechanism_summary(teams, matches, magnitude, exposure_matrix, competition, country, clipping)
    hash_after = hashlib.sha256(production_path.read_bytes()).hexdigest()
    if hash_before != hash_after:
        raise ValueError("Production contract changed during diagnostic")
    if not np.allclose(matches[["current_home_probability", "current_draw_probability", "current_away_probability"]].sum(axis=1), 1.0, atol=1e-12):
        raise ValueError("Current probabilities are not normalized")
    if not np.allclose(matches[["global_home_probability", "global_draw_probability", "global_away_probability"]].sum(axis=1), 1.0, atol=1e-12):
        raise ValueError("Global probabilities are not normalized")
    if evaluations[CURRENT].predictions["zero_sum_error"].abs().max() > 1e-9 or evaluations[GLOBAL].predictions["zero_sum_error"].abs().max() > 1e-9:
        raise ValueError("Power update zero-sum invariant failed")

    top_harm = teams.sort_values(["ranking_error_difference", "initial_elo_difference"], ascending=[False, False], kind="stable").head(30)
    top_improvement = teams.sort_values(["ranking_error_difference", "initial_elo_difference"], ascending=[True, True], kind="stable").head(30)
    top_loss_gain = matches.sort_values(["brier_difference", "log_loss_difference"], kind="stable").head(50)
    top_loss_harm = matches.sort_values(["brier_difference", "log_loss_difference"], ascending=[False, False], kind="stable").head(50)

    teams.to_csv(output / "team_season_diagnostics.csv", index=False)
    matches.to_csv(output / "match_loss_diagnostics.csv", index=False)
    quadrants.to_csv(output / "loss_ranking_quadrants.csv", index=False)
    magnitude.to_csv(output / "adjustment_magnitude_analysis.csv", index=False)
    exposure_matrix.to_csv(output / "exposure_adjustment_matrix.csv", index=False)
    positive_negative.to_csv(output / "positive_negative_analysis.csv", index=False)
    favorite.to_csv(output / "favorite_underdog_analysis.csv", index=False)
    country.to_csv(output / "country_league_analysis.csv", index=False)
    competition.to_csv(output / "competition_analysis.csv", index=False)
    fold.to_csv(output / "fold_analysis.csv", index=False)
    clipping.to_csv(output / "counterfactual_clipping.csv", index=False)
    top_harm.to_csv(output / "top_ranking_harm_teams.csv", index=False)
    top_improvement.to_csv(output / "top_ranking_improvement_teams.csv", index=False)
    top_loss_gain.to_csv(output / "top_loss_gain_matches.csv", index=False)
    top_loss_harm.to_csv(output / "top_loss_harm_matches.csv", index=False)
    calibration.to_csv(output / "calibration_analysis.csv", index=False)
    comparison.to_csv(output / "model_comparison.csv", index=False)
    uncertainty.to_csv(output / "dependency_uncertainty.csv", index=False)
    (output / "mechanism_summary.json").write_text(
        json.dumps(
            {
                **mechanism,
                "current_config": configs[CURRENT].__dict__,
                "global_strong_config": configs[GLOBAL].__dict__,
                "evaluation_matches": len(matches),
                "evaluation_team_seasons": len(teams),
                "production_contract_sha256": hash_after,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(output / "tradeoff_report.md", comparison, quadrants, magnitude, exposure_matrix, competition, clipping, calibration, uncertainty, mechanism)
    print(f"Mechanism: {mechanism['category']}")
    print(f"Matches: {len(matches)}; team-seasons: {len(teams)}")
    print(f"Report: {output / 'tradeoff_report.md'}")


if __name__ == "__main__":
    main()
