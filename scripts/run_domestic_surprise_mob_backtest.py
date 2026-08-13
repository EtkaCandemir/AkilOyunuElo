from __future__ import annotations

"""Walk-forward MOB research for the Domestic Surprise adjustment."""

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ao_elo.config import AOEuropeanEloConfig  # noqa: E402
from ao_elo.evaluation import (  # noqa: E402
    dependency_robust_loss_difference_ci,
    schedule_adjusted_team_performance,
)
from ao_elo.model_based_partitioning import (  # noqa: E402
    MOBConfig,
    MOBNode,
    MOBTree,
    fit_mob_tree,
)
from scripts.run_controlled_goal_progression_backtest import prepare_controlled_data  # noqa: E402
from scripts.run_current_model_evaluation import EvaluationArm, evaluate_arm  # noqa: E402
from scripts.run_domestic_surprise_effect_size_sensitivity import (  # noqa: E402
    competition_metrics,
    fold_metrics,
    markdown_table,
    model_metrics,
)
from scripts.run_domestic_surprise_exposure_nonlinear import (  # noqa: E402
    CURRENT_KEY,
    GLOBAL_STRONG_KEY,
    NO_SURPRISE_KEY,
    build_exposure_adjustments,
    candidate_grid as exposure_candidate_grid,
    forward_fold_metrics,
)
from scripts.run_domestic_surprise_hierarchical_shrinkage import (  # noqa: E402
    EXPOSURE_ONLY_KEY,
    EXPOSURE_SELECTION,
    FEATURES_PATH,
    HierarchicalCandidate,
    build_hierarchical_adjustments,
    percentile_features,
    target_ranks,
)
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
    aggregate_ranking,
    load_domestic_adjustments,
    load_xg_map,
    validate_production_contract,
)
from scripts.run_v2_achievement_reserve_calibration import load_reserve_data  # noqa: E402
from scripts.run_v2_evaluation_upgrade import read_events  # noqa: E402


OUTPUT_ROOT = ROOT / "output/domestic_surprise_mob_backtest_2018_2026"
PREVIOUS_POOLING_ROOT = ROOT / "output/domestic_surprise_weak_league_partial_pooling"
PRE_UECL = "PRE_UECL"
UECL_ERA = "UECL_ERA"
MOB_ROOT_ONLY = "MOB_ROOT_ONLY"
MOB_TREE = "MOB_TREE"
MOB_NO_COMPETITION = "MOB_ABLATION_NO_COMPETITION"
MOB_NO_EXPOSURE = "MOB_ABLATION_NO_EXPOSURE"
MOB_NO_LEAGUE = "MOB_ABLATION_NO_LEAGUE_STRENGTH"
MOB_NO_RATING = "MOB_ABLATION_NO_RATING_PERCENTILE"
HIERARCHICAL_BEST = "HIERARCHICAL_BEST"
NESTED_PARTIAL_POOLING = "NESTED_PARTIAL_POOLING"
ACTIVE_MOB_SEASONS = {"2020/21", "2023/24", "2024/25", "2025/26"}
EARLY_UECL_FALLBACK_SEASONS = {"2021/22", "2022/23"}
MODEL_ORDER = (
    CURRENT_KEY,
    NO_SURPRISE_KEY,
    GLOBAL_STRONG_KEY,
    EXPOSURE_ONLY_KEY,
    HIERARCHICAL_BEST,
    NESTED_PARTIAL_POOLING,
    MOB_ROOT_ONLY,
    MOB_TREE,
)


def season_start(season: str) -> int:
    try:
        return int(str(season).split("/", 1)[0])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid season label: {season}") from exc


def era_for_season(season: str) -> str:
    return PRE_UECL if season_start(season) <= 2020 else UECL_ERA


def add_rating_percentile(features: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    ratings = current[["season", "team_id", "adjusted_ao_first_elo"]].rename(
        columns={"adjusted_ao_first_elo": "current_initial_rating"}
    )
    result = features.merge(ratings, on=["season", "team_id"], validate="one_to_one")
    result["rating_percentile"] = result.groupby("season")["current_initial_rating"].rank(
        method="average", pct=True
    )
    return result


def build_learning_frame(
    features: pd.DataFrame,
    current: pd.DataFrame,
    strong: pd.DataFrame,
    target: pd.DataFrame,
) -> pd.DataFrame:
    targets = target_ranks(target)[["season", "team_id", "target_score", "target_matches"]]
    current_columns = [
        "season",
        "team_id",
        "team_name",
        "club_id",
        "country_code",
        "competition",
        "league_strength",
        "effective_european_exposure",
        "history_seasons_available",
        "raw_surprise",
        "ao_first_elo_adjustment",
        "adjusted_ao_first_elo",
    ]
    result = current[current_columns].rename(
        columns={
            "ao_first_elo_adjustment": "current_adjustment",
            "adjusted_ao_first_elo": "current_initial_rating",
        }
    ).merge(
        strong[["season", "team_id", "ao_first_elo_adjustment"]].rename(
            columns={"ao_first_elo_adjustment": "strong_adjustment"}
        ),
        on=["season", "team_id"],
        validate="one_to_one",
    ).merge(
        features[["season", "team_id", "rating_percentile"]],
        on=["season", "team_id"],
        validate="one_to_one",
    ).merge(
        targets,
        on=["season", "team_id"],
        validate="one_to_one",
    )
    result["strong_increment"] = result["strong_adjustment"] - result["current_adjustment"]
    result["format_era"] = result["season"].map(era_for_season)
    return result.sort_values(["season", "team_id"], kind="stable").reset_index(drop=True)


def _tree_nodes(node: MOBNode, *, target_season: str, model: str) -> Iterable[dict[str, object]]:
    yield {
        "model": model,
        "target_season": target_season,
        "node_id": node.node_id,
        "depth": node.depth,
        "is_leaf": node.is_leaf,
        "observations": node.observations,
        "training_seasons": ",".join(node.seasons),
        "nonzero_increment_observations": node.nonzero_increment_observations,
        "theta": node.theta,
        "theta_standard_error": node.theta_standard_error,
        "diagnostic_intercept": node.diagnostic_intercept,
        "weighted_rss": node.weighted_rss,
        "split_variable": node.split_variable,
        "split_type": node.split_type,
        "split_threshold": node.split_threshold,
        "left_categories": "|".join(node.left_categories),
        "raw_p_value": node.raw_p_value,
        "adjusted_p_value": node.adjusted_p_value,
        "stop_reason": node.stop_reason,
    }
    if node.left is not None:
        yield from _tree_nodes(node.left, target_season=target_season, model=model)
    if node.right is not None:
        yield from _tree_nodes(node.right, target_season=target_season, model=model)


def _apply_mob_adjustment(
    group: pd.DataFrame,
    tree: MOBTree | None,
    config: MOBConfig,
    fallback_reason: str,
    model: str,
) -> pd.DataFrame:
    result = group.copy()
    multipliers: list[float] = []
    leaves: list[str] = []
    fallbacks: list[bool] = []
    reasons: list[str] = []
    for row in result.to_dict(orient="records"):
        if tree is None:
            theta, leaf, fallback, reason = 0.0, "CURRENT_FALLBACK", True, fallback_reason
        else:
            prediction = tree.predict(row)
            theta, leaf = prediction.theta, prediction.leaf_id
            fallback, reason = prediction.fallback, prediction.fallback_reason
        multipliers.append(float(theta))
        leaves.append(leaf)
        fallbacks.append(bool(fallback))
        reasons.append(reason)

    current = result["current_adjustment"].to_numpy(float)
    increment = result["strong_increment"].to_numpy(float)
    raw = current + np.asarray(multipliers) * increment
    capped = np.clip(raw, -config.final_adjustment_cap, config.final_adjustment_cap)
    sign_reversed = capped * result["raw_surprise"].to_numpy(float) < -1e-12
    final = np.where(sign_reversed, 0.0, capped)
    exposure = result["effective_european_exposure"].to_numpy(float)

    result["mob_theta"] = multipliers
    result["mob_leaf_id"] = leaves
    result["mob_fallback"] = fallbacks
    result["mob_fallback_reason"] = reasons
    result["mob_uncapped_adjustment"] = raw
    result["mob_cap_hit"] = np.abs(raw) > config.final_adjustment_cap + 1e-12
    result["mob_sign_guard_applied"] = sign_reversed
    result["ao_first_elo_adjustment"] = final
    result["adjusted_ao_first_elo"] = result["baseline_ao_first_elo"] + final
    result["domestic_prior_adjustment"] = np.where(
        1.0 - exposure > 1e-12,
        final / np.maximum(1.0 - exposure, 1e-12),
        0.0,
    )
    result["candidate_key"] = model
    result["model_family"] = "MODEL_BASED_RECURSIVE_PARTITIONING"
    result["parameter_count"] = np.nan if tree is None else sum(1 for _ in leaf_nodes(tree.root))
    result["parameter_json"] = json.dumps(asdict(config), separators=(",", ":"))
    result["final_context_cap"] = config.final_adjustment_cap
    return result


def leaf_nodes(node: MOBNode) -> Iterable[MOBNode]:
    if node.is_leaf:
        yield node
        return
    if node.left is not None:
        yield from leaf_nodes(node.left)
    if node.right is not None:
        yield from leaf_nodes(node.right)


def build_walk_forward_mob_adjustments(
    learning: pd.DataFrame,
    current: pd.DataFrame,
    config: MOBConfig,
    model: str,
    *,
    tree_output_root: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config.validate()
    current_lookup = current.rename(columns={"ao_first_elo_adjustment": "current_adjustment"})
    context = learning[[
        "season",
        "team_id",
        "strong_adjustment",
        "strong_increment",
        "rating_percentile",
        "format_era",
    ]]
    base = current_lookup.merge(context, on=["season", "team_id"], validate="one_to_one")
    adjustment_frames: list[pd.DataFrame] = []
    assignment_frames: list[pd.DataFrame] = []
    node_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []

    seasons = tuple(sorted(base["season"].astype(str).unique(), key=season_start))
    for target_season in seasons:
        era = era_for_season(target_season)
        history = learning.loc[
            learning["format_era"].eq(era)
            & learning["season"].map(season_start).lt(season_start(target_season))
        ].copy()
        history_seasons = tuple(sorted(history["season"].unique(), key=season_start))
        target_group = base.loc[base["season"].eq(target_season)].copy()
        tree: MOBTree | None = None
        fallback_reason = ""
        if len(history_seasons) < config.minimum_era_history_seasons:
            fallback_reason = "INSUFFICIENT_ERA_HISTORY"
        else:
            tree = fit_mob_tree(history, config)
            node_rows.extend(_tree_nodes(tree.root, target_season=target_season, model=model))
            for instability in tree.instability_tests:
                row = instability.as_dict()
                row.update({
                    "model": model,
                    "target_season": target_season,
                    "format_era": era,
                    "training_seasons": ",".join(history_seasons),
                })
                test_rows.append(row)
            if tree_output_root is not None:
                tree_output_root.mkdir(parents=True, exist_ok=True)
                (tree_output_root / f"{model.lower()}_{target_season.replace('/', '_')}.json").write_text(
                    tree.to_json(), encoding="utf-8"
                )
        applied = _apply_mob_adjustment(
            target_group, tree, config, fallback_reason, model
        )
        applied["format_era"] = era
        applied["mob_training_seasons"] = ",".join(history_seasons)
        adjustment_frames.append(applied)
        assignment_frames.append(applied[[
            "candidate_key",
            "season",
            "team_id",
            "team_name",
            "club_id",
            "country_code",
            "competition",
            "format_era",
            "mob_training_seasons",
            "league_strength",
            "rating_percentile",
            "effective_european_exposure",
            "current_adjustment",
            "strong_adjustment",
            "strong_increment",
            "mob_theta",
            "mob_leaf_id",
            "mob_fallback",
            "mob_fallback_reason",
            "mob_uncapped_adjustment",
            "ao_first_elo_adjustment",
            "mob_cap_hit",
            "mob_sign_guard_applied",
            "adjusted_ao_first_elo",
        ]])

    adjustments = pd.concat(adjustment_frames, ignore_index=True).sort_values(
        ["season", "team_id"], kind="stable"
    ).reset_index(drop=True)
    assignments = pd.concat(assignment_frames, ignore_index=True).sort_values(
        ["season", "team_id"], kind="stable"
    ).reset_index(drop=True)
    return adjustments, assignments, pd.DataFrame(node_rows), pd.DataFrame(test_rows)


def evaluate_ratings(datasets, key, frame, core, parameters, xg_map, target):
    ratings = {
        (str(row.season), int(row.team_id)): float(row.adjusted_ao_first_elo)
        for row in frame.itertuples(index=False)
    }
    return evaluate_arm(
        datasets,
        EvaluationArm(key, True, True, True, True, True),
        core=core,
        parameters=parameters,
        current_domestic=ratings,
        baseline_domestic=ratings,
        xg_map=xg_map,
        target=target,
    )


def adjustment_effect(frame: pd.DataFrame) -> dict[str, object]:
    absolute = frame["ao_first_elo_adjustment"].abs()
    changed = absolute.gt(1e-12)
    current_increment = (
        frame["ao_first_elo_adjustment"] - frame["current_adjustment_reference"]
    )
    current_absolute = current_increment.abs()
    candidate_rank = frame.groupby("season")["adjusted_ao_first_elo"].rank(
        method="min", ascending=False
    )
    current_rank = frame.groupby("season")["current_initial_rating_reference"].rank(
        method="min", ascending=False
    )
    rank_change = current_rank - candidate_rank
    return {
        "changed_team_seasons": int(changed.sum()),
        "changed_mean_abs_elo": float(absolute.loc[changed].mean()) if changed.any() else 0.0,
        "mean_abs_elo": float(absolute.mean()),
        "median_abs_elo": float(absolute.median()),
        "p90_abs_elo": float(absolute.quantile(0.90)),
        "p95_abs_elo": float(absolute.quantile(0.95)),
        "maximum_positive_elo": float(frame["ao_first_elo_adjustment"].max()),
        "maximum_negative_elo": float(frame["ao_first_elo_adjustment"].min()),
        "max_abs_elo": float(absolute.max()),
        "cap_hit_rate": float(frame.get("mob_cap_hit", pd.Series(False, index=frame.index)).mean()),
        "fallback_rate": float(frame.get("mob_fallback", pd.Series(False, index=frame.index)).mean()),
        "sign_guard_rate": float(frame.get("mob_sign_guard_applied", pd.Series(False, index=frame.index)).mean()),
        "changed_vs_current_team_seasons": int(current_absolute.gt(1e-12).sum()),
        "mean_abs_elo_change_vs_current": float(current_absolute.mean()),
        "median_abs_elo_change_vs_current": float(current_absolute.median()),
        "p90_abs_elo_change_vs_current": float(current_absolute.quantile(0.90)),
        "p95_abs_elo_change_vs_current": float(current_absolute.quantile(0.95)),
        "maximum_positive_elo_change_vs_current": float(current_increment.max()),
        "maximum_negative_elo_change_vs_current": float(current_increment.min()),
        "mean_abs_rank_change_vs_current": float(rank_change.abs().mean()),
        "maximum_rank_gain_vs_current": int(rank_change.max()),
        "maximum_rank_loss_vs_current": int(rank_change.min()),
    }


def comparison_row(key, evaluation, adjustment, evaluation_seasons, target, identity, seasons):
    return {
        "model": key,
        **model_metrics(evaluation, evaluation_seasons, target, identity, seasons),
        **adjustment_effect(adjustment.loc[adjustment["season"].isin(evaluation_seasons)]),
    }


def add_comparison_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    current = result.loc[result["model"].eq(CURRENT_KEY)].iloc[0]
    for metric in (
        "brier_1x2",
        "log_loss_1x2",
        "accuracy_1x2",
        "same_season_spearman",
        "same_season_pairwise_accuracy",
        "forward_season_spearman",
        "forward_season_pairwise_accuracy",
    ):
        result[f"delta_vs_current_{metric}"] = result[metric] - float(current[metric])
    return result


def add_fold_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    baseline = result.loc[result["model"].eq(CURRENT_KEY)].set_index("test_season")
    for metric in (
        "brier_1x2",
        "log_loss_1x2",
        "accuracy_1x2",
        "same_season_spearman",
        "same_season_pairwise_accuracy",
    ):
        result[f"delta_vs_current_{metric}"] = result.apply(
            lambda row: row[metric] - baseline.loc[row["test_season"], metric], axis=1
        )
    return result


def competition_results(evaluations, evaluation_seasons):
    frames = []
    for key, evaluation in evaluations.items():
        frame = competition_metrics(evaluation, evaluation_seasons)
        frame.insert(0, "model", key)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    baseline = result.loc[result["model"].eq(CURRENT_KEY)].set_index("competition")
    for metric in (
        "brier_1x2",
        "log_loss_1x2",
        "accuracy_1x2",
        "same_season_spearman",
        "same_season_pairwise_accuracy",
    ):
        result[f"delta_vs_current_{metric}"] = result.apply(
            lambda row: row[metric] - baseline.loc[row["competition"], metric], axis=1
        )
    return result


def loss_uncertainty(candidate, current, evaluation_seasons, bootstrap_samples):
    paired = candidate.predictions.merge(
        current.predictions[["match_id", "brier_1x2", "log_loss_1x2"]],
        on="match_id",
        suffixes=("_candidate", "_current"),
        validate="one_to_one",
    )
    paired = paired.loc[paired["season"].isin(evaluation_seasons)]
    rows = []
    segments = [("ALL", "ALL", paired)]
    segments.extend(("COMPETITION", value, group) for value, group in paired.groupby("competition", sort=True))
    segments.extend(("FOLD", value, group) for value, group in paired.groupby("season", sort=True))
    for segment_type, segment, group in segments:
        for metric in ("brier_1x2", "log_loss_1x2"):
            sample = group[["season", "match_id", "home_team_id", "away_team_id", "kickoff_utc", "tie_id"]].copy()
            sample["loss_difference"] = group[f"{metric}_candidate"] - group[f"{metric}_current"]
            result = dependency_robust_loss_difference_ci(
                sample,
                bootstrap_samples=bootstrap_samples,
                seed=20260812 + sum(ord(character) for character in f"{segment_type}{segment}{metric}"),
            )
            result.insert(0, "candidate_key", MOB_TREE)
            result.insert(1, "segment_type", segment_type)
            result.insert(2, "segment", segment)
            result.insert(3, "metric", metric)
            rows.append(result)
    return pd.concat(rows, ignore_index=True)


def ranking_uncertainty(candidate, current, folds, bootstrap_samples):
    rng = np.random.default_rng(20260812)
    rows: list[dict[str, object]] = []
    candidate_fold = fold_metrics(candidate, folds).set_index("test_season")
    current_fold = fold_metrics(current, folds).set_index("test_season")
    for metric in ("same_season_spearman", "same_season_pairwise_accuracy"):
        delta = (candidate_fold[metric] - current_fold[metric]).to_numpy(float)
        samples = rng.choice(delta, size=(bootstrap_samples, len(delta)), replace=True).mean(axis=1)
        lower, upper = np.quantile(samples, (0.025, 0.975))
        rows.append({
            "candidate_key": MOB_TREE,
            "segment_type": "ALL",
            "segment": "ALL",
            "metric": metric,
            "method": "season_block_bootstrap",
            "observations": len(delta),
            "mean_difference": float(delta.mean()),
            "ci_95_lower": float(lower),
            "ci_95_upper": float(upper),
            "reliable_improvement": bool(lower > 0.0),
            "reliable_harm": bool(upper < 0.0),
        })

    candidate_rows = candidate.same_season_ranking
    current_rows = current.same_season_ranking
    for competition in ("UCL", "UEL", "UECL"):
        left = candidate_rows.loc[candidate_rows["competition"].eq(competition), ["season", "ranking_score", "pairwise_accuracy"]]
        right = current_rows.loc[current_rows["competition"].eq(competition), ["season", "ranking_score", "pairwise_accuracy"]]
        merged = left.merge(right, on="season", suffixes=("_candidate", "_current"), validate="one_to_one")
        for source, label in (("ranking_score", "same_season_spearman"), ("pairwise_accuracy", "same_season_pairwise_accuracy")):
            delta = (merged[f"{source}_candidate"] - merged[f"{source}_current"]).to_numpy(float)
            samples = rng.choice(delta, size=(bootstrap_samples, len(delta)), replace=True).mean(axis=1)
            lower, upper = np.quantile(samples, (0.025, 0.975))
            rows.append({
                "candidate_key": MOB_TREE,
                "segment_type": "COMPETITION",
                "segment": competition,
                "metric": label,
                "method": "season_block_bootstrap",
                "observations": len(delta),
                "mean_difference": float(delta.mean()),
                "ci_95_lower": float(lower),
                "ci_95_upper": float(upper),
                "reliable_improvement": bool(lower > 0.0),
                "reliable_harm": bool(upper < 0.0),
            })
    return pd.DataFrame(rows)


def segment_diagnostics(assignments, candidate, current, target, evaluation_seasons):
    targets = target_ranks(target)[["season", "team_id", "target_score", "target_rank"]]
    candidate_end = candidate.end_ratings[["season", "team_id", "end_live_rating"]]
    current_end = current.end_ratings[["season", "team_id", "end_live_rating"]].rename(
        columns={"end_live_rating": "current_end_live_rating"}
    )
    frame = assignments.loc[assignments["season"].isin(evaluation_seasons)].merge(
        candidate_end, on=["season", "team_id"], validate="one_to_one"
    ).merge(current_end, on=["season", "team_id"], validate="one_to_one").merge(
        targets, on=["season", "team_id"], validate="one_to_one"
    )
    frame["candidate_rank"] = frame.groupby("season")["end_live_rating"].rank(method="min", ascending=False)
    frame["current_rank"] = frame.groupby("season")["current_end_live_rating"].rank(method="min", ascending=False)
    frame["rank_error_delta"] = (
        (frame["candidate_rank"] - frame["target_rank"]).abs()
        - (frame["current_rank"] - frame["target_rank"]).abs()
    )
    frame["exposure_band"] = pd.cut(
        frame["effective_european_exposure"],
        [-1e-12, 0.0, 0.25, 0.50, 0.75, 1.0],
        labels=["0", "(0,0.25]", "(0.25,0.50]", "(0.50,0.75]", "(0.75,1.00]"],
        include_lowest=True,
    )
    frame["league_band"] = frame.groupby("season")["league_strength"].transform(
        lambda values: pd.qcut(values.rank(method="first"), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    )
    frame["rating_band"] = frame.groupby("season")["rating_percentile"].transform(
        lambda values: pd.qcut(values.rank(method="first"), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    )
    rows = []
    for dimension in ("exposure_band", "league_band", "rating_band", "competition", "format_era", "mob_leaf_id"):
        for value, group in frame.groupby(dimension, observed=False, dropna=False):
            rows.append({
                "dimension": dimension,
                "segment": str(value),
                "team_seasons": len(group),
                "mean_theta": float(group["mob_theta"].mean()),
                "mean_adjustment": float(group["ao_first_elo_adjustment"].mean()),
                "mean_abs_adjustment": float(group["ao_first_elo_adjustment"].abs().mean()),
                "p90_abs_adjustment": float(group["ao_first_elo_adjustment"].abs().quantile(0.90)),
                "mean_rank_error_delta": float(group["rank_error_delta"].mean()),
                "fallback_rate": float(group["mob_fallback"].mean()),
                "cap_hit_rate": float(group["mob_cap_hit"].mean()),
            })
    return pd.DataFrame(rows), frame


def safety_audit(
    adjustment,
    assignments,
    evaluation,
    learning,
    contract_hash_before,
    contract_hash_after,
    production_ratings,
):
    probabilities = evaluation.predictions[["home_probability", "draw_probability", "away_probability"]]
    training_order_ok = True
    for row in assignments.itertuples(index=False):
        for training_season in str(row.mob_training_seasons).split(","):
            if training_season and season_start(training_season) >= season_start(str(row.season)):
                training_order_ok = False
    checks = {
        "unique_learning_team_seasons": not learning.duplicated(["season", "team_id"]).any(),
        "unique_adjustment_team_seasons": not adjustment.duplicated(["season", "team_id"]).any(),
        "unique_match_predictions": not evaluation.predictions.duplicated(["season", "match_id"]).any(),
        "chronology_valid": bool(evaluation.predictions.groupby("season")["kickoff_utc"].apply(lambda values: values.is_monotonic_increasing).all()),
        "probabilities_finite": bool(np.isfinite(probabilities).all().all()),
        "probabilities_normalized": bool(np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)),
        "exposure_invariant": bool(adjustment["effective_european_exposure"].between(0.0, 1.0).all()),
        "theta_bounded": bool(adjustment["mob_theta"].between(0.0, 1.25).all()),
        "final_cap_respected": bool(adjustment["ao_first_elo_adjustment"].abs().le(60.0 + 1e-9).all()),
        "sign_preserved": bool((adjustment["ao_first_elo_adjustment"] * adjustment["raw_surprise"]).ge(-1e-9).all()),
        "insufficient_history_zero": bool(adjustment.loc[adjustment["history_seasons"].lt(5), "ao_first_elo_adjustment"].abs().le(1e-9).all()),
        "early_uecl_era_current_fallback": bool(assignments.loc[assignments["season"].isin(EARLY_UECL_FALLBACK_SEASONS), "mob_fallback"].all()),
        "early_uecl_era_zero_theta": bool(assignments.loc[assignments["season"].isin(EARLY_UECL_FALLBACK_SEASONS), "mob_theta"].abs().le(1e-12).all()),
        "training_precedes_target": training_order_ok,
        "pre_and_post_uecl_not_mixed": bool(assignments.apply(lambda row: all(era_for_season(value) == row["format_era"] for value in str(row["mob_training_seasons"]).split(",") if value), axis=1).all()),
        "match_zero_sum": bool(evaluation.predictions["zero_sum_error"].abs().max() <= 1e-9),
        "season_power_conservation": bool(evaluation.season_metrics["season_power_conservation_error"].max() <= 1e-9),
        "finite_ratings": bool(np.isfinite(adjustment["adjusted_ao_first_elo"]).all() and np.isfinite(evaluation.end_ratings["end_live_rating"]).all()),
        "production_contract_unchanged": contract_hash_before == contract_hash_after,
        "holdout_2026_27_absent": not learning["season"].eq("2026/27").any(),
        "production_reference_complete": len(production_ratings) == len(adjustment),
    }
    return pd.DataFrame([{"check": name, "passed": bool(value)} for name, value in checks.items()])


def classify_result(comparison, folds, competition, uncertainty, tree_nodes):
    current = comparison.loc[comparison["model"].eq(CURRENT_KEY)].iloc[0]
    candidate = comparison.loc[comparison["model"].eq(MOB_TREE)].iloc[0]
    active = folds.loc[folds["model"].eq(MOB_TREE) & folds["test_season"].isin(ACTIVE_MOB_SEASONS)]
    brier_wins = int(active["delta_vs_current_brier_1x2"].lt(-1e-12).sum())
    log_wins = int(active["delta_vs_current_log_loss_1x2"].lt(-1e-12).sum())
    pooled_loss_better = bool(
        candidate["brier_1x2"] < current["brier_1x2"]
        and candidate["log_loss_1x2"] < current["log_loss_1x2"]
    )
    reliable_loss_harm = bool(
        uncertainty.loc[
            uncertainty["method"].eq("conservative_envelope")
            & uncertainty["segment_type"].isin(["FOLD", "COMPETITION"]),
            "reliable_harm",
        ].fillna(False).any()
    )
    reliable_rank_harm = bool(
        uncertainty.loc[uncertainty["method"].eq("season_block_bootstrap"), "reliable_harm"].fillna(False).any()
    )
    split_counts = tree_nodes.loc[tree_nodes["stop_reason"].eq("SPLIT")].groupby("split_variable")["target_season"].nunique()
    repeated_split = bool((split_counts >= 2).any())
    competition_directional_harm = bool(
        (competition.loc[competition["model"].eq(MOB_TREE), ["delta_vs_current_brier_1x2", "delta_vs_current_log_loss_1x2"]] > 0.0).all(axis=1).any()
    )
    promotion = all((
        brier_wins >= 3,
        log_wins >= 3,
        pooled_loss_better,
        not reliable_loss_harm,
        not reliable_rank_harm,
        repeated_split,
    ))
    if promotion:
        verdict = "PROMOTE_SHADOW_CANDIDATE"
    elif reliable_loss_harm or reliable_rank_harm:
        verdict = "REJECT"
    else:
        verdict = "KEEP_DIAGNOSTIC"

    rank_metrics = (
        "same_season_spearman",
        "same_season_pairwise_accuracy",
        "forward_season_spearman",
        "forward_season_pairwise_accuracy",
    )
    rank_reliable_safe = not reliable_rank_harm
    previous = comparison.loc[comparison["model"].isin([EXPOSURE_ONLY_KEY, HIERARCHICAL_BEST, NESTED_PARTIAL_POOLING])]
    best_shadow = bool(
        promotion
        and rank_reliable_safe
        and candidate["brier_1x2"] < previous["brier_1x2"].min()
        and candidate["log_loss_1x2"] < previous["log_loss_1x2"].min()
        and all(math.isfinite(float(candidate[metric])) for metric in rank_metrics)
    )
    return verdict, best_shadow, {
        "active_fold_brier_wins": brier_wins,
        "active_fold_log_loss_wins": log_wins,
        "pooled_loss_better": pooled_loss_better,
        "reliable_loss_harm": reliable_loss_harm,
        "reliable_ranking_harm": reliable_rank_harm,
        "repeated_split": repeated_split,
        "competition_directional_harm_diagnostic": competition_directional_harm,
    }


def write_report(path, comparison, folds, competition, nodes, tests, segments, uncertainty, verdict, best_shadow, gates):
    selected = comparison.loc[comparison["model"].isin(MODEL_ORDER)]
    active_folds = folds.loc[folds["model"].isin([CURRENT_KEY, MOB_TREE])]
    split_nodes = nodes.loc[nodes["stop_reason"].eq("SPLIT")]
    lines = [
        "# Domestic Surprise MOB Walk-Forward Backtesti",
        "",
        "Bu çalışma yalnız shadow/research değerlendirmesidir. Production contract ve canlı Elo motoru değiştirilmemiştir.",
        "",
        "## Yöntem",
        "",
        "MOB, Current başlangıç ratinginin schedule-adjusted performansla WLS kalibrasyonundan kalan residual üzerinde Strong Increment katsayısının kararsızlığını test eder. PRE_UECL ve UECL_ERA ayrı eğitilir; yeni dönemin ilk iki sezonu Current fallback'tir. Leaf interceptleri diagnostiktir ve ratinge uygulanmaz.",
        "",
        "## Ana karşılaştırma",
        "",
        markdown_table(selected),
        "",
        "## Fold sonuçları",
        "",
        markdown_table(active_folds),
        "",
        "## Turnuva sonuçları",
        "",
        markdown_table(competition.loc[competition["model"].isin([CURRENT_KEY, MOB_TREE])]),
        "",
        "## Bulunan splitler",
        "",
        markdown_table(split_nodes) if len(split_nodes) else "Anlamlı tekrarlanan split bulunmadı.",
        "",
        "## Segmentler",
        "",
        markdown_table(segments),
        "",
        "## Belirsizlik",
        "",
        markdown_table(uncertainty.loc[uncertainty["method"].isin(["conservative_envelope", "season_block_bootstrap"])]),
        "",
        "## Model kararı",
        "",
        f"Karar: **{verdict}**.",
        f"Önceki shadow adaylara göre en güçlü güvenli aday: **{'EVET' if best_shadow else 'HAYIR'}**.",
        "",
        "Karar kapıları:",
        "",
        *[f"- `{name}`: `{value}`" for name, value in gates.items()],
        "",
        f"Parametre kararsızlığı test satırı: `{len(tests)}`; ağaç node satırı: `{len(nodes)}`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    parser.add_argument("--permutations", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("bootstrap-samples must be at least 100")
    if args.permutations < 1:
        raise ValueError("permutations must be positive")
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tree_output = output / "fold_trees"

    production_path = PRODUCTION_CONTRACT.resolve()
    contract_hash_before = hashlib.sha256(production_path.read_bytes()).hexdigest()
    production = json.loads(production_path.read_text(encoding="utf-8"))
    core, parameters = validate_production_contract(production)
    dynamic = json.loads(DYNAMIC_MANIFEST.read_text(encoding="utf-8"))
    static_config = AOEuropeanEloConfig(**dynamic["static_config"])
    events = read_events(EVENTS_PATH)
    reserve, _ = load_reserve_data(STATIC_DATA_ROOT, EVENTS_PATH, static_config)
    datasets = prepare_controlled_data(reserve, events)
    seasons = tuple(data.season for data in datasets)
    folds = expanding_folds(seasons)
    evaluation_seasons = {test for _, test in folds}
    target = schedule_adjusted_team_performance(events)
    identity = load_team_season_identity()
    xg_map = load_xg_map(XG_DATA, datasets)
    raw_features = pd.read_csv(FEATURES_PATH)
    production_ratings = load_domestic_adjustments(DOMESTIC_ADJUSTMENTS, datasets)

    candidates = {candidate.key: candidate for candidate in exposure_candidate_grid(production)}
    exposure_payload = json.loads(EXPOSURE_SELECTION.read_text(encoding="utf-8"))
    exposure_source = candidates[exposure_payload["candidate_key"]]
    exposure_candidate = type(exposure_source)(
        EXPOSURE_ONLY_KEY,
        exposure_source.family,
        exposure_source.theta_parameters,
        exposure_source.cap_parameters,
        exposure_source.complexity_parameters,
    )
    control_candidates = {
        CURRENT_KEY: candidates[CURRENT_KEY],
        NO_SURPRISE_KEY: candidates[NO_SURPRISE_KEY],
        GLOBAL_STRONG_KEY: candidates[GLOBAL_STRONG_KEY],
        EXPOSURE_ONLY_KEY: exposure_candidate,
    }
    adjustments: dict[str, pd.DataFrame] = {}
    evaluations = {}
    for key, candidate in control_candidates.items():
        frame = build_exposure_adjustments(raw_features, candidate, static_config, production)
        frame["candidate_key"] = key
        adjustments[key] = frame
    features = add_rating_percentile(percentile_features(raw_features), adjustments[CURRENT_KEY])

    current_errors = [
        abs(float(row.adjusted_ao_first_elo) - production_ratings[(str(row.season), int(row.team_id))])
        for row in adjustments[CURRENT_KEY].itertuples(index=False)
    ]
    if max(current_errors, default=0.0) > 1e-9:
        raise ValueError("Current Domestic Surprise ratings do not match production")

    hierarchical_candidate = HierarchicalCandidate(
        HIERARCHICAL_BEST, "GATED", 0.35, 40.0, (0.35, 0.20), 4
    )
    hierarchical = build_hierarchical_adjustments(
        features, adjustments[GLOBAL_STRONG_KEY], hierarchical_candidate
    )
    hierarchical["candidate_key"] = HIERARCHICAL_BEST
    adjustments[HIERARCHICAL_BEST] = hierarchical
    learning = build_learning_frame(
        features, adjustments[CURRENT_KEY], adjustments[GLOBAL_STRONG_KEY], target
    )
    if len(learning) != 1887:
        raise ValueError(f"Expected 1,887 MOB team-seasons, found {len(learning):,}")

    base_config = MOBConfig(permutations=args.permutations)
    mob_configs = {
        MOB_ROOT_ONLY: replace(base_config, max_depth=0, max_leaves=1),
        MOB_TREE: base_config,
        MOB_NO_COMPETITION: replace(
            base_config,
            partition_variables=tuple(value for value in base_config.partition_variables if value != "competition"),
        ),
        MOB_NO_EXPOSURE: replace(
            base_config,
            partition_variables=tuple(value for value in base_config.partition_variables if value != "effective_european_exposure"),
        ),
        MOB_NO_LEAGUE: replace(
            base_config,
            partition_variables=tuple(value for value in base_config.partition_variables if value != "league_strength"),
        ),
        MOB_NO_RATING: replace(
            base_config,
            partition_variables=tuple(value for value in base_config.partition_variables if value != "rating_percentile"),
        ),
    }
    assignment_frames = []
    node_frames = []
    instability_frames = []
    for index, (key, config) in enumerate(mob_configs.items(), start=1):
        frame, assignments_frame, nodes, instability = build_walk_forward_mob_adjustments(
            learning,
            adjustments[CURRENT_KEY],
            config,
            key,
            tree_output_root=tree_output if key == MOB_TREE else None,
        )
        adjustments[key] = frame
        assignment_frames.append(assignments_frame)
        node_frames.append(nodes)
        instability_frames.append(instability)
        print(f"  MOB arm {index}/{len(mob_configs)}: {key}", flush=True)

    current_reference = adjustments[CURRENT_KEY][[
        "season", "team_id", "ao_first_elo_adjustment", "adjusted_ao_first_elo"
    ]].rename(columns={
        "ao_first_elo_adjustment": "current_adjustment_reference",
        "adjusted_ao_first_elo": "current_initial_rating_reference",
    })
    for key, frame in tuple(adjustments.items()):
        adjustments[key] = frame.merge(
            current_reference, on=["season", "team_id"], validate="one_to_one"
        )

    evaluation_keys = [
        CURRENT_KEY,
        NO_SURPRISE_KEY,
        GLOBAL_STRONG_KEY,
        EXPOSURE_ONLY_KEY,
        HIERARCHICAL_BEST,
        *mob_configs,
    ]
    for index, key in enumerate(evaluation_keys, start=1):
        evaluations[key] = evaluate_ratings(
            datasets, key, adjustments[key], core, parameters, xg_map, target
        )
        print(f"  replay {index}/{len(evaluation_keys)}: {key}", flush=True)

    comparison = pd.DataFrame([
        comparison_row(key, evaluations[key], adjustments[key], evaluation_seasons, target, identity, seasons)
        for key in evaluation_keys
    ])
    previous_path = PREVIOUS_POOLING_ROOT / "model_comparison.csv"
    previous_payload = PREVIOUS_POOLING_ROOT / "selected_shadow_candidate.json"
    if previous_path.exists() and previous_payload.exists():
        previous_hash = json.loads(previous_payload.read_text(encoding="utf-8")).get("production_contract_sha256")
        if previous_hash == contract_hash_before:
            previous = pd.read_csv(previous_path)
            nested = previous.loc[previous["model"].eq("NESTED_PARTIAL_POOLING")].copy()
            if len(nested) == 1:
                nested["model"] = NESTED_PARTIAL_POOLING
                for column in comparison.columns:
                    if column not in nested.columns:
                        nested[column] = np.nan
                comparison = pd.concat([comparison, nested[comparison.columns]], ignore_index=True)
    comparison = add_comparison_deltas(comparison)

    fold_frames = []
    forward_frames = []
    for key, evaluation in evaluations.items():
        fold = fold_metrics(evaluation, folds)
        fold.insert(0, "model", key)
        fold_frames.append(fold)
        forward = forward_fold_metrics(evaluation, target, identity, seasons)
        forward.insert(0, "model", key)
        forward_frames.append(forward)
    folds_all = add_fold_deltas(pd.concat(fold_frames, ignore_index=True))
    forward_all = pd.concat(forward_frames, ignore_index=True)
    competition = competition_results(evaluations, evaluation_seasons)

    assignments_all = pd.concat(assignment_frames, ignore_index=True)
    assignments_full = assignments_all.loc[assignments_all["candidate_key"].eq(MOB_TREE)].copy()
    nodes_all = pd.concat(node_frames, ignore_index=True) if any(len(frame) for frame in node_frames) else pd.DataFrame()
    tests_all = pd.concat(instability_frames, ignore_index=True) if any(len(frame) for frame in instability_frames) else pd.DataFrame()
    segments, team_diagnostics = segment_diagnostics(
        assignments_full,
        evaluations[MOB_TREE],
        evaluations[CURRENT_KEY],
        target,
        evaluation_seasons,
    )
    loss_ci = loss_uncertainty(
        evaluations[MOB_TREE], evaluations[CURRENT_KEY], evaluation_seasons, args.bootstrap_samples
    )
    rank_ci = ranking_uncertainty(
        evaluations[MOB_TREE], evaluations[CURRENT_KEY], folds, args.bootstrap_samples
    )
    uncertainty = pd.concat([loss_ci, rank_ci], ignore_index=True, sort=False)
    verdict, best_shadow, gates = classify_result(
        comparison, folds_all, competition, uncertainty, nodes_all
    )

    contract_hash_after = hashlib.sha256(production_path.read_bytes()).hexdigest()
    audit = safety_audit(
        adjustments[MOB_TREE],
        assignments_full,
        evaluations[MOB_TREE],
        learning,
        contract_hash_before,
        contract_hash_after,
        production_ratings,
    )
    if not audit["passed"].all():
        failed = audit.loc[~audit["passed"], "check"].tolist()
        raise ValueError(f"MOB safety audit failed: {failed}")

    comparison.to_csv(output / "model_comparison.csv", index=False)
    folds_all.to_csv(output / "fold_results.csv", index=False)
    forward_all.to_csv(output / "forward_ranking.csv", index=False)
    competition.to_csv(output / "competition_results.csv", index=False)
    assignments_all.to_csv(output / "mob_tree_assignments.csv", index=False)
    nodes_all.to_csv(output / "mob_tree_nodes.csv", index=False)
    tests_all.to_csv(output / "parameter_instability_tests.csv", index=False)
    comparison.loc[comparison["model"].isin(mob_configs)].to_csv(output / "ablation_results.csv", index=False)
    uncertainty.to_csv(output / "dependency_uncertainty.csv", index=False)
    segments.to_csv(output / "segment_analysis.csv", index=False)
    team_diagnostics.to_csv(output / "team_season_diagnostics.csv", index=False)
    audit.to_csv(output / "safety_audit.csv", index=False)

    candidate = comparison.loc[comparison["model"].eq(MOB_TREE)].iloc[0]
    selected = {
        "status": "SHADOW_RESEARCH_ONLY",
        "production_changed": False,
        "verdict": verdict,
        "best_shadow_candidate": best_shadow,
        "candidate_key": MOB_TREE,
        "parameters": asdict(base_config),
        "active_mob_seasons": sorted(ACTIVE_MOB_SEASONS, key=season_start),
        "fallback_seasons": sorted(EARLY_UECL_FALLBACK_SEASONS, key=season_start),
        "decision_gates": gates,
        "metrics": {
            name: float(candidate[name])
            for name in (
                "brier_1x2",
                "log_loss_1x2",
                "accuracy_1x2",
                "same_season_spearman",
                "same_season_pairwise_accuracy",
                "forward_season_spearman",
                "forward_season_pairwise_accuracy",
            )
        },
        "production_contract_sha256": contract_hash_after,
    }
    (output / "selected_shadow_candidate.json").write_text(
        json.dumps(selected, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest = {
        "analysis": "DOMESTIC_SURPRISE_MODEL_BASED_RECURSIVE_PARTITIONING",
        "team_seasons": len(learning),
        "evaluation_matches": int(candidate["matches"]),
        "evaluation_seasons": sorted(evaluation_seasons, key=season_start),
        "permutations_per_node": args.permutations,
        "bootstrap_samples": args.bootstrap_samples,
        "evaluation_only": True,
        "production_changed": False,
        "production_contract_sha256": contract_hash_after,
        "verdict": verdict,
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_report(
        output / "mob_backtest_report.md",
        comparison,
        folds_all,
        competition,
        nodes_all,
        tests_all,
        segments,
        uncertainty,
        verdict,
        best_shadow,
        gates,
    )
    print(f"Verdict: {verdict}")
    print(f"MOB Brier: {candidate['brier_1x2']:.9f}")
    print(f"MOB log-loss: {candidate['log_loss_1x2']:.9f}")
    print(f"Report: {output / 'mob_backtest_report.md'}")


if __name__ == "__main__":
    main()
