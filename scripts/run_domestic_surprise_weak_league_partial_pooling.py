from __future__ import annotations

"""Causal weak-league/UECL partial-pooling research for Domestic Surprise."""

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ao_elo.config import AOEuropeanEloConfig  # noqa: E402
from ao_elo.evaluation import dependency_robust_loss_difference_ci, schedule_adjusted_team_performance  # noqa: E402
from scripts.run_controlled_goal_progression_backtest import prepare_controlled_data  # noqa: E402
from scripts.run_current_model_evaluation import ArmEvaluation, EvaluationArm, evaluate_arm, prediction_summary  # noqa: E402
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
    same_season_ranking,
    validate_production_contract,
)
from scripts.run_v2_achievement_reserve_calibration import load_reserve_data  # noqa: E402
from scripts.run_v2_evaluation_upgrade import read_events  # noqa: E402


OUTPUT_ROOT = ROOT / "output/domestic_surprise_weak_league_partial_pooling"
BASE_HIERARCHICAL_KEY = "hierarchical_base_gated_f0p35_c40"
NESTED_KEY = "NESTED_PARTIAL_POOLING"
Q_LABELS = ("Q1", "Q2", "Q3", "Q4", "Q5")


@dataclass(frozen=True)
class PoolingCandidate:
    key: str
    scope: str
    prior_strength: float
    risk_floor: float
    strong_ceiling: float
    strong_cap: float
    ranking_weight: float
    parameter_count: int = 6

    def validate(self) -> None:
        if self.scope not in {"GLOBAL_BASE", "TARGETED_ONLY"}:
            raise ValueError("scope must be GLOBAL_BASE or TARGETED_ONLY")
        if self.prior_strength <= 0 or not math.isfinite(self.prior_strength):
            raise ValueError("prior_strength must be positive and finite")
        if not 0 <= self.risk_floor <= 1:
            raise ValueError("risk_floor must be in [0,1]")
        if not 1 <= self.strong_ceiling <= 2:
            raise ValueError("strong_ceiling must be in [1,2]")
        if not 40 <= self.strong_cap <= 100:
            raise ValueError("strong_cap must be in [40,100]")
        if not 0 <= self.ranking_weight <= 1:
            raise ValueError("ranking_weight must be in [0,1]")

    def as_json(self) -> str:
        return json.dumps(self.__dict__, separators=(",", ":"))


def candidate_grid() -> tuple[PoolingCandidate, ...]:
    candidates = []
    for scope in ("GLOBAL_BASE", "TARGETED_ONLY"):
        for prior in (20.0, 50.0, 100.0):
            for floor in (0.25, 0.50):
                for ceiling in (1.25, 1.50):
                    for cap in (40.0, 60.0, 75.0):
                        for rank_weight in (0.50, 0.75):
                            scope_token = "global" if scope == "GLOBAL_BASE" else "targeted"
                            key = (
                                f"pool_{scope_token}_k{int(prior)}_f{str(floor).replace('.', 'p')}_"
                                f"u{str(ceiling).replace('.', 'p')}_c{int(cap)}_rw{str(rank_weight).replace('.', 'p')}"
                            )
                            candidate = PoolingCandidate(key, scope, prior, floor, ceiling, cap, rank_weight)
                            candidate.validate()
                            candidates.append(candidate)
    return tuple(candidates)


def context_segment(frame: pd.DataFrame) -> pd.Series:
    is_uecl = frame["competition"].astype(str).str.contains("UECL", regex=False)
    league = frame["league_quintile"].astype(str)
    return pd.Series(
        np.select(
            [league.eq("Q1") & is_uecl, league.eq("Q1"), league.eq("Q5") & is_uecl, league.eq("Q5"), is_uecl],
            ["Q1_UECL", "Q1_OTHER", "Q5_UECL", "Q5_OTHER", "MID_UECL"],
            default="MID_OTHER",
        ),
        index=frame.index,
        dtype="object",
    )


def target_team_table(target: pd.DataFrame) -> pd.DataFrame:
    return target_ranks(target)[["season", "team_id", "target_score", "target_rank"]]


def build_learning_evidence(current_adjustments, base_adjustments, current_eval, base_eval, target):
    targets = target_team_table(target)
    current_end = current_eval.end_ratings[["season", "team_id", "end_live_rating"]].rename(
        columns={"end_live_rating": "current_end_rating"}
    )
    base_end = base_eval.end_ratings[["season", "team_id", "end_live_rating"]].rename(
        columns={"end_live_rating": "base_end_rating"}
    )
    frame = base_adjustments[[
        "season", "team_id", "team_name", "club_id", "country_code", "competition",
        "league_quintile", "rating_quintile", "effective_european_exposure",
        "league_strength", "baseline_ao_first_elo", "ao_first_elo_adjustment",
    ]].merge(
        current_adjustments[["season", "team_id", "ao_first_elo_adjustment"]].rename(
            columns={"ao_first_elo_adjustment": "current_adjustment"}
        ), on=["season", "team_id"], validate="one_to_one"
    ).merge(current_end, on=["season", "team_id"], validate="one_to_one").merge(
        base_end, on=["season", "team_id"], validate="one_to_one"
    ).merge(targets, on=["season", "team_id"], how="left", validate="one_to_one")
    frame["current_rank"] = frame.groupby("season")["current_end_rating"].rank(method="min", ascending=False)
    frame["base_rank"] = frame.groupby("season")["base_end_rating"].rank(method="min", ascending=False)
    frame["current_rank_error"] = (frame["current_rank"] - frame["target_rank"]).abs()
    frame["base_rank_error"] = (frame["base_rank"] - frame["target_rank"]).abs()
    frame["rank_gain"] = frame["current_rank_error"] - frame["base_rank_error"]

    current_predictions = current_eval.predictions[["season", "match_id", "home_team_id", "away_team_id", "brier_1x2"]]
    base_predictions = base_eval.predictions[["season", "match_id", "brier_1x2"]]
    matches = current_predictions.merge(base_predictions, on=["season", "match_id"], suffixes=("_current", "_base"), validate="one_to_one")
    matches["brier_gain"] = matches["brier_1x2_current"] - matches["brier_1x2_base"]
    sides = pd.concat([
        matches[["season", "home_team_id", "brier_gain"]].rename(columns={"home_team_id": "team_id"}),
        matches[["season", "away_team_id", "brier_gain"]].rename(columns={"away_team_id": "team_id"}),
    ], ignore_index=True)
    loss = sides.groupby(["season", "team_id"], as_index=False).agg(
        involved_matches=("brier_gain", "size"), total_brier_gain=("brier_gain", "sum")
    )
    frame = frame.merge(loss, on=["season", "team_id"], how="left", validate="one_to_one")
    frame[["involved_matches", "total_brier_gain"]] = frame[["involved_matches", "total_brier_gain"]].fillna(0.0)
    frame["rank_success"] = np.select(
        [frame["rank_gain"].gt(1e-12), frame["rank_gain"].lt(-1e-12)], [1.0, 0.0], default=0.5
    )
    frame["loss_success"] = np.select(
        [frame["total_brier_gain"].gt(1e-12), frame["total_brier_gain"].lt(-1e-12)], [1.0, 0.0], default=0.5
    )
    frame["segment"] = context_segment(frame)
    return frame


def learned_multipliers(evidence, target_season, candidate):
    history = evidence.loc[evidence["season"].lt(target_season)].copy()
    if history.empty:
        return {segment: 1.0 for segment in ("Q1_UECL", "Q1_OTHER", "Q5_UECL", "Q5_OTHER", "MID_UECL", "MID_OTHER")}, []
    history["utility"] = (
        candidate.ranking_weight * history["rank_success"]
        + (1.0 - candidate.ranking_weight) * history["loss_success"]
    )
    global_mean = float(history["utility"].mean())
    multipliers = {}
    rows = []
    for segment in ("Q1_UECL", "Q1_OTHER", "Q5_UECL", "Q5_OTHER", "MID_UECL", "MID_OTHER"):
        group = history.loc[history["segment"].eq(segment)]
        posterior = (
            float(group["utility"].sum()) + candidate.prior_strength * global_mean
        ) / (len(group) + candidate.prior_strength)
        raw = posterior / max(global_mean, 1e-12)
        if segment.startswith("Q1") or segment == "MID_UECL":
            multiplier = float(np.clip(raw, candidate.risk_floor, 1.0))
        elif segment.startswith("Q5"):
            multiplier = float(np.clip(raw, 1.0, candidate.strong_ceiling))
        else:
            multiplier = 1.0
        multipliers[segment] = multiplier
        rows.append({
            "candidate_key": candidate.key, "target_season": target_season, "segment": segment,
            "history_team_seasons": len(group), "global_utility_prior": global_mean,
            "segment_utility_mean": float(group["utility"].mean()) if len(group) else np.nan,
            "posterior_utility": posterior, "raw_multiplier": raw, "applied_multiplier": multiplier,
        })
    return multipliers, rows


def build_pooled_adjustments(features, current, base, evidence, candidate):
    keys = ["season", "team_id"]
    current_lookup = current[keys + ["ao_first_elo_adjustment"]].rename(
        columns={"ao_first_elo_adjustment": "current_adjustment"}
    )
    frame = base.merge(current_lookup, on=keys, validate="one_to_one")
    frame["segment"] = context_segment(frame)
    frames = []
    posterior_rows = []
    for season, group in frame.groupby("season", sort=True):
        multipliers, rows = learned_multipliers(evidence, str(season), candidate)
        posterior_rows.extend(rows)
        group = group.copy()
        group["partial_pool_multiplier"] = group["segment"].map(multipliers).astype(float)
        increment = group["ao_first_elo_adjustment"] - group["current_adjustment"]
        final = group["current_adjustment"] + group["partial_pool_multiplier"] * increment
        if candidate.scope == "TARGETED_ONLY":
            targeted = group["segment"].isin(["Q1_UECL", "Q1_OTHER", "MID_UECL", "Q5_UECL", "Q5_OTHER"])
            final = np.where(targeted, final, group["current_adjustment"])
        cap = np.where(group["league_quintile"].astype(str).eq("Q5"), candidate.strong_cap, 40.0)
        group["final_context_cap"] = cap
        group["ao_first_elo_adjustment"] = np.clip(final, -cap, cap)
        group["adjusted_ao_first_elo"] = group["baseline_ao_first_elo"] + group["ao_first_elo_adjustment"]
        group["candidate_key"] = candidate.key
        group["model_family"] = "CAUSAL_PARTIAL_POOLING"
        group["parameter_count"] = candidate.parameter_count
        group["parameter_json"] = candidate.as_json()
        frames.append(group)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(posterior_rows)


def evaluate_ratings(datasets, key, frame, core, parameters, xg_map, target):
    ratings = {(str(row.season), int(row.team_id)): float(row.adjusted_ao_first_elo) for row in frame.itertuples(index=False)}
    return evaluate_arm(
        datasets, EvaluationArm(key, True, True, True, True, True), core=core, parameters=parameters,
        current_domestic=ratings, baseline_domestic=ratings, xg_map=xg_map, target=target,
    )


def train_metrics(evaluation, seasons):
    predictions = evaluation.predictions.loc[evaluation.predictions["season"].isin(seasons)]
    ranking = aggregate_ranking(evaluation.same_season_ranking.loc[evaluation.same_season_ranking["season"].isin(seasons)])
    pooled = ranking.loc[ranking["competition"].eq("ALL")].iloc[0]
    return {
        **prediction_summary(predictions),
        "same_season_spearman": float(pooled["ranking_score"]),
        "same_season_pairwise_accuracy": float(pooled["pairwise_accuracy"]),
    }


def season_ranking_metrics(evaluation, seasons):
    rows = []
    for season in seasons:
        ranking = aggregate_ranking(
            evaluation.same_season_ranking.loc[evaluation.same_season_ranking["season"].eq(season)]
        )
        pooled = ranking.loc[ranking["competition"].eq("ALL")].iloc[0]
        rows.append({
            "season": season,
            "same_season_spearman": float(pooled["ranking_score"]),
            "same_season_pairwise_accuracy": float(pooled["pairwise_accuracy"]),
        })
    return pd.DataFrame(rows)


def ranking_reliable_harm(evaluation, current_evaluation, train_seasons, bootstrap_samples, seed):
    candidate = season_ranking_metrics(evaluation, train_seasons).set_index("season")
    current = season_ranking_metrics(current_evaluation, train_seasons).set_index("season")
    rng = np.random.default_rng(seed)
    result = {}
    for metric in ("same_season_spearman", "same_season_pairwise_accuracy"):
        delta = (candidate[metric] - current[metric]).to_numpy(float)
        samples = rng.choice(delta, size=(bootstrap_samples, len(delta)), replace=True).mean(axis=1)
        lower, upper = np.quantile(samples, [.025, .975])
        result[f"{metric}_train_delta"] = float(delta.mean())
        result[f"{metric}_train_ci_lower"] = float(lower)
        result[f"{metric}_train_ci_upper"] = float(upper)
        result[f"{metric}_reliable_harm"] = bool(upper < 0.0)
    return result


def select_nested(evaluations, candidates, folds, bootstrap_samples):
    rows = []
    for fold, (train_seasons, test_season) in enumerate(folds, start=1):
        current = train_metrics(evaluations[CURRENT_KEY], set(train_seasons))
        candidate_rows = []
        for candidate_index, candidate in enumerate(candidates):
            metrics = train_metrics(evaluations[candidate.key], set(train_seasons))
            ranking_guard = ranking_reliable_harm(
                evaluations[candidate.key], evaluations[CURRENT_KEY], train_seasons,
                bootstrap_samples, seed=20260812 + fold * 1000 + candidate_index,
            )
            candidate_rows.append({
                "candidate_key": candidate.key, **metrics,
                "spearman_delta": metrics["same_season_spearman"] - current["same_season_spearman"],
                "pairwise_delta": metrics["same_season_pairwise_accuracy"] - current["same_season_pairwise_accuracy"],
                **ranking_guard,
            })
        surface = pd.DataFrame(candidate_rows)
        safe = surface.loc[
            ~surface["same_season_spearman_reliable_harm"]
            & ~surface["same_season_pairwise_accuracy_reliable_harm"]
        ]
        if safe.empty:
            selected_key = CURRENT_KEY
            reason = "CURRENT_FALLBACK_ALL_HAVE_RELIABLE_RANK_HARM"
            selected_metrics = current
        else:
            selected = safe.sort_values(["brier_1x2", "log_loss_1x2"], kind="stable").iloc[0]
            selected_key = str(selected["candidate_key"])
            reason = "NO_RELIABLE_RANK_HARM_MINIMUM_TRAIN_LOSS"
            selected_metrics = selected.to_dict()
        rows.append({
            "fold": fold, "train_seasons": ",".join(train_seasons), "test_season": test_season,
            "selected_candidate_key": selected_key, "selection_reason": reason,
            **{f"train_{name}": value for name, value in selected_metrics.items() if name != "candidate_key"},
            "rank_safe_candidates": len(safe),
        })
    return pd.DataFrame(rows)


def composite_evaluation(evaluations, selections, target):
    parts = {name: [] for name in ("predictions", "end_ratings", "season_metrics", "bonus_events")}
    for row in selections.itertuples(index=False):
        evaluation = evaluations[row.selected_candidate_key]
        for name in parts:
            frame = getattr(evaluation, name)
            if frame.empty:
                continue
            parts[name].append(frame.loc[frame["season"].eq(row.test_season)].copy())
    predictions = pd.concat(parts["predictions"], ignore_index=True)
    end_ratings = pd.concat(parts["end_ratings"], ignore_index=True)
    season_metrics = pd.concat(parts["season_metrics"], ignore_index=True)
    bonus_events = pd.concat(parts["bonus_events"], ignore_index=True) if parts["bonus_events"] else pd.DataFrame()
    ranking = same_season_ranking(end_ratings, target, set(selections["test_season"]))
    ranking["model"] = NESTED_KEY
    return ArmEvaluation(predictions, end_ratings, season_metrics, ranking, bonus_events)


def composite_adjustments(adjustments, selections):
    frames = []
    for row in selections.itertuples(index=False):
        frame = adjustments[row.selected_candidate_key]
        frames.append(frame.loc[frame["season"].eq(row.test_season)].copy())
    result = pd.concat(frames, ignore_index=True)
    result["candidate_key"] = NESTED_KEY
    return result


def effect_metrics(frame):
    absolute = frame["ao_first_elo_adjustment"].abs()
    changed = absolute.gt(1e-12)
    return {
        "changed_team_seasons": int(changed.sum()),
        "changed_mean_abs_elo": float(absolute.loc[changed].mean()) if changed.any() else 0.0,
        "p90_abs_elo": float(absolute.quantile(.90)), "p95_abs_elo": float(absolute.quantile(.95)),
        "max_abs_elo": float(absolute.max()),
    }


def comparison_row(key, evaluation, adjustment, evaluation_seasons, target, identity, seasons):
    return {
        "model": key, **model_metrics(evaluation, evaluation_seasons, target, identity, seasons),
        **effect_metrics(adjustment.loc[adjustment["season"].isin(evaluation_seasons)]),
    }


def segment_analysis(key, adjustment, evaluation, current_eval, target, evaluation_seasons):
    targets = target_team_table(target)
    frame = adjustment.loc[adjustment["season"].isin(evaluation_seasons)].copy()
    end = evaluation.end_ratings[["season", "team_id", "end_live_rating"]]
    current_end = current_eval.end_ratings[["season", "team_id", "end_live_rating"]].rename(columns={"end_live_rating": "current_end_rating"})
    frame = frame.merge(end, on=["season", "team_id"], validate="one_to_one").merge(current_end, on=["season", "team_id"], validate="one_to_one").merge(targets, on=["season", "team_id"], how="left", validate="one_to_one")
    frame["segment"] = context_segment(frame)
    frame["rank"] = frame.groupby("season")["end_live_rating"].rank(method="min", ascending=False)
    frame["current_rank"] = frame.groupby("season")["current_end_rating"].rank(method="min", ascending=False)
    frame["rank_error_delta"] = (frame["rank"] - frame["target_rank"]).abs() - (frame["current_rank"] - frame["target_rank"]).abs()
    predictions = evaluation.predictions.merge(
        current_eval.predictions[["match_id", "brier_1x2", "log_loss_1x2"]], on="match_id", suffixes=("_candidate", "_current"), validate="one_to_one"
    )
    predictions["brier_delta"] = predictions["brier_1x2_candidate"] - predictions["brier_1x2_current"]
    predictions["log_loss_delta"] = predictions["log_loss_1x2_candidate"] - predictions["log_loss_1x2_current"]
    sides = pd.concat([
        predictions[["season", "home_team_id", "brier_delta", "log_loss_delta"]].rename(columns={"home_team_id": "team_id"}),
        predictions[["season", "away_team_id", "brier_delta", "log_loss_delta"]].rename(columns={"away_team_id": "team_id"}),
    ])
    losses = sides.groupby(["season", "team_id"], as_index=False).agg(matches=("brier_delta", "size"), brier_delta=("brier_delta", "mean"), log_loss_delta=("log_loss_delta", "mean"))
    frame = frame.merge(losses, on=["season", "team_id"], how="left", validate="one_to_one")
    rows = []
    for segment, group in frame.groupby("segment", sort=True):
        eligible = group.dropna(subset=["target_score"])
        rows.append({
            "model": key, "segment": segment, "team_seasons": len(group), "matches": int(group["matches"].fillna(0).sum()),
            "mean_adjustment": group["ao_first_elo_adjustment"].mean(), "mean_abs_adjustment": group["ao_first_elo_adjustment"].abs().mean(),
            "mean_multiplier": group.get("partial_pool_multiplier", pd.Series(1.0, index=group.index)).mean(),
            "brier_delta_vs_current": np.average(group["brier_delta"].fillna(0), weights=group["matches"].fillna(0)) if group["matches"].fillna(0).sum() else 0.0,
            "log_loss_delta_vs_current": np.average(group["log_loss_delta"].fillna(0), weights=group["matches"].fillna(0)) if group["matches"].fillna(0).sum() else 0.0,
            "spearman_delta_vs_current": eligible["end_live_rating"].corr(eligible["target_score"], method="spearman") - eligible["current_end_rating"].corr(eligible["target_score"], method="spearman"),
            "pairwise_delta_vs_current": pairwise_ranking_accuracy(eligible["end_live_rating"].to_numpy(float), eligible["target_score"].to_numpy(float)) - pairwise_ranking_accuracy(eligible["current_end_rating"].to_numpy(float), eligible["target_score"].to_numpy(float)),
            "mean_rank_error_delta": group["rank_error_delta"].mean(),
        })
    return pd.DataFrame(rows), frame


def uncertainty(candidate_eval, current_eval, evaluation_seasons, bootstrap_samples):
    paired = candidate_eval.predictions.merge(
        current_eval.predictions[["match_id", "brier_1x2", "log_loss_1x2"]], on="match_id", suffixes=("_candidate", "_current"), validate="one_to_one"
    )
    paired = paired.loc[paired["season"].isin(evaluation_seasons)]
    rows = []
    for metric in ("brier_1x2", "log_loss_1x2"):
        sample = paired[["season", "match_id", "home_team_id", "away_team_id", "kickoff_utc", "tie_id"]].copy()
        sample["loss_difference"] = paired[f"{metric}_candidate"] - paired[f"{metric}_current"]
        result = dependency_robust_loss_difference_ci(sample, bootstrap_samples=bootstrap_samples)
        result.insert(0, "candidate_key", NESTED_KEY)
        result.insert(1, "metric", metric)
        rows.append(result)
    return pd.concat(rows, ignore_index=True)


def ranking_uncertainty(fold_results, forward_results, bootstrap_samples):
    rng = np.random.default_rng(20260812)
    rows = []
    current_fold = fold_results.loc[fold_results["model"].eq(CURRENT_KEY)].set_index("test_season")
    nested_fold = fold_results.loc[fold_results["model"].eq(NESTED_KEY)].set_index("test_season")
    for metric in ("same_season_spearman", "same_season_pairwise_accuracy"):
        delta = (nested_fold[metric] - current_fold[metric]).to_numpy(float)
        samples = rng.choice(delta, size=(bootstrap_samples, len(delta)), replace=True).mean(axis=1)
        lower, upper = np.quantile(samples, [.025, .975])
        rows.append({
            "candidate_key": NESTED_KEY, "metric": metric, "method": "unseen_fold_resampling",
            "matches": np.nan, "clusters": len(delta), "mean_difference": delta.mean(),
            "ci_95_lower": lower, "ci_95_upper": upper,
            "reliable_improvement": lower > 0.0, "reliable_harm": upper < 0.0,
        })
    current_forward = forward_results.loc[forward_results["model"].eq(CURRENT_KEY)].set_index("target_season")
    nested_forward = forward_results.loc[forward_results["model"].eq(NESTED_KEY)].set_index("target_season")
    for metric in ("forward_season_spearman", "forward_season_pairwise_accuracy"):
        delta = (nested_forward[metric] - current_forward[metric]).dropna().to_numpy(float)
        samples = rng.choice(delta, size=(bootstrap_samples, len(delta)), replace=True).mean(axis=1)
        lower, upper = np.quantile(samples, [.025, .975])
        rows.append({
            "candidate_key": NESTED_KEY, "metric": metric, "method": "forward_fold_resampling",
            "matches": np.nan, "clusters": len(delta), "mean_difference": delta.mean(),
            "ci_95_lower": lower, "ci_95_upper": upper,
            "reliable_improvement": lower > 0.0, "reliable_harm": upper < 0.0,
        })
    return pd.DataFrame(rows)


def safety_audit(adjustment, evaluation, selections, production_hash_before, production_hash_after):
    probabilities = evaluation.predictions[["home_probability", "draw_probability", "away_probability"]]
    checks = {
        "probabilities_normalized": np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12),
        "unique_matches": not evaluation.predictions.duplicated(["season", "match_id"]).any(),
        "unique_team_seasons": not adjustment.duplicated(["season", "team_id"]).any(),
        "chronology": evaluation.predictions.groupby("season")["kickoff_utc"].apply(lambda values: values.is_monotonic_increasing).all(),
        "exposure_in_range": adjustment["effective_european_exposure"].between(0, 1).all(),
        "sign_preserved": (adjustment["ao_first_elo_adjustment"] * adjustment["raw_surprise"]).ge(-1e-9).all(),
        "cap_respected": adjustment["ao_first_elo_adjustment"].abs().le(adjustment.get("final_context_cap", 150.0) + 1e-9).all(),
        "zero_sum_updates": evaluation.predictions["zero_sum_error"].abs().max() <= 1e-9,
        "power_conservation": evaluation.season_metrics["season_power_conservation_error"].max() <= 1e-9,
        "finite_ratings": np.isfinite(evaluation.end_ratings["end_live_rating"]).all(),
        "no_holdout_selection_leakage": all(str(row.test_season) not in str(row.train_seasons).split(",") for row in selections.itertuples()),
        "production_unchanged": production_hash_before == production_hash_after,
    }
    return pd.DataFrame([{"check": key, "passed": bool(value)} for key, value in checks.items()])


def write_report(path, comparison, selections, segments, competition, uncertainty_frame, verdict):
    nested = comparison.loc[comparison["model"].eq(NESTED_KEY)].iloc[0]
    current = comparison.loc[comparison["model"].eq(CURRENT_KEY)].iloc[0]
    lines = [
        "# Weak-League / UECL Partial-Pooling Domestic Surprise",
        "",
        "Shadow/research only. Production was not changed.",
        "",
        "## Causal method",
        "",
        "For each target season, segment reliability used only completed earlier seasons. Q1/UECL multipliers could only shrink toward Current; Q5 multipliers could only release the selected hierarchical base. Hyperparameters were selected inside each expanding fold, and ranking vetoed an arm only when training-season bootstrap showed reliable harm.",
        "",
        "## Comparison",
        "", markdown_table(comparison), "",
        "## Fold selections",
        "", markdown_table(selections), "",
        "## Segments",
        "", markdown_table(segments), "",
        "## Competitions",
        "", markdown_table(competition), "",
        "## Uncertainty",
        "", markdown_table(uncertainty_frame), "",
        "## Decision",
        "",
        f"Verdict: **{verdict}**.",
        f"Nested candidate deltas vs Current: Brier `{nested['brier_1x2'] - current['brier_1x2']:+.9f}`, log-loss `{nested['log_loss_1x2'] - current['log_loss_1x2']:+.9f}`, same-season Spearman `{nested['same_season_spearman'] - current['same_season_spearman']:+.9f}`, pairwise `{nested['same_season_pairwise_accuracy'] - current['same_season_pairwise_accuracy']:+.9f}`.",
        "",
        "## Next research direction",
        "",
        "Freeze Domestic Surprise research and validate the simpler Exposure-only candidate prospectively on 2026/27. It remains the only tested variant that improves pooled loss and all four ranking metrics without a segment-specific learned gate.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)

    contract_path = PRODUCTION_CONTRACT.resolve()
    hash_before = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    production = json.loads(contract_path.read_text(encoding="utf-8"))
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
    features = percentile_features(pd.read_csv(FEATURES_PATH))
    production_ratings = load_domestic_adjustments(DOMESTIC_ADJUSTMENTS, datasets)

    exposure_controls = {candidate.key: candidate for candidate in exposure_candidate_grid(production)}
    exposure_payload = json.loads(EXPOSURE_SELECTION.read_text(encoding="utf-8"))
    source = exposure_controls[exposure_payload["candidate_key"]]
    controls = {
        CURRENT_KEY: exposure_controls[CURRENT_KEY],
        NO_SURPRISE_KEY: exposure_controls[NO_SURPRISE_KEY],
        GLOBAL_STRONG_KEY: exposure_controls[GLOBAL_STRONG_KEY],
        EXPOSURE_ONLY_KEY: type(source)(EXPOSURE_ONLY_KEY, source.family, source.theta_parameters, source.cap_parameters, source.complexity_parameters),
    }
    adjustments = {}
    evaluations = {}
    for key, candidate in controls.items():
        frame = build_exposure_adjustments(features, candidate, static_config, production)
        frame = frame.merge(features[["season", "team_id", "league_reliability", "rating_reliability", "league_quintile", "rating_quintile", "exposure_tier"]], on=["season", "team_id"], validate="one_to_one")
        frame["partial_pool_multiplier"] = 1.0
        frame["final_context_cap"] = frame["cap_effective"]
        adjustments[key] = frame
        evaluations[key] = evaluate_ratings(datasets, key, frame, core, parameters, xg_map, target)
    errors = [abs(float(row.adjusted_ao_first_elo) - production_ratings[(str(row.season), int(row.team_id))]) for row in adjustments[CURRENT_KEY].itertuples()]
    if max(errors, default=0.0) > 1e-9:
        raise ValueError("Current ratings do not match production")

    base_candidate = HierarchicalCandidate(BASE_HIERARCHICAL_KEY, "GATED", 0.35, 40.0, (0.35, 0.20), 4)
    base = build_hierarchical_adjustments(features, adjustments[GLOBAL_STRONG_KEY], base_candidate)
    base["partial_pool_multiplier"] = 1.0
    base["final_context_cap"] = 40.0
    adjustments[BASE_HIERARCHICAL_KEY] = base
    evaluations[BASE_HIERARCHICAL_KEY] = evaluate_ratings(datasets, BASE_HIERARCHICAL_KEY, base, core, parameters, xg_map, target)
    evidence = build_learning_evidence(adjustments[CURRENT_KEY], base, evaluations[CURRENT_KEY], evaluations[BASE_HIERARCHICAL_KEY], target)

    candidates = candidate_grid()
    posterior_frames = []
    for index, candidate in enumerate(candidates, start=1):
        frame, posteriors = build_pooled_adjustments(features, adjustments[CURRENT_KEY], base, evidence, candidate)
        adjustments[candidate.key] = frame
        evaluations[candidate.key] = evaluate_ratings(datasets, candidate.key, frame, core, parameters, xg_map, target)
        posterior_frames.append(posteriors)
        print(f"  candidate {index}/{len(candidates)}: {candidate.key}", flush=True)

    selections = select_nested(evaluations, candidates, folds, args.bootstrap_samples)
    nested_eval = composite_evaluation(evaluations, selections, target)
    nested_adjustment = composite_adjustments(adjustments, selections)
    evaluations[NESTED_KEY] = nested_eval
    adjustments[NESTED_KEY] = nested_adjustment

    comparison_keys = [CURRENT_KEY, NO_SURPRISE_KEY, GLOBAL_STRONG_KEY, EXPOSURE_ONLY_KEY, BASE_HIERARCHICAL_KEY, NESTED_KEY]
    comparison = pd.DataFrame([
        comparison_row(key, evaluations[key], adjustments[key], evaluation_seasons, target, identity, seasons)
        for key in comparison_keys
    ])
    fold_frames = []
    for key in comparison_keys:
        frame = fold_metrics(evaluations[key], folds)
        frame.insert(0, "model", key)
        fold_frames.append(frame)
    fold_results = pd.concat(fold_frames, ignore_index=True)
    current_fold = fold_results.loc[fold_results["model"].eq(CURRENT_KEY)].set_index("test_season")
    for metric in ("brier_1x2", "log_loss_1x2", "same_season_spearman", "same_season_pairwise_accuracy"):
        fold_results[f"delta_vs_current_{metric}"] = fold_results.apply(lambda row: row[metric] - current_fold.loc[row["test_season"], metric], axis=1)
    forward_frames = []
    for key in comparison_keys:
        frame = forward_fold_metrics(evaluations[key], target, identity, seasons)
        frame.insert(0, "model", key)
        forward_frames.append(frame)
    forward_results = pd.concat(forward_frames, ignore_index=True)

    segment_frames = []
    team_frames = []
    for key in (GLOBAL_STRONG_KEY, BASE_HIERARCHICAL_KEY, NESTED_KEY):
        segment, team = segment_analysis(key, adjustments[key], evaluations[key], evaluations[CURRENT_KEY], target, evaluation_seasons)
        segment_frames.append(segment)
        team_frames.append(team.assign(model=key))
    segments = pd.concat(segment_frames, ignore_index=True)
    team_diagnostic = pd.concat(team_frames, ignore_index=True)
    competition_frames = []
    current_comp = competition_metrics(evaluations[CURRENT_KEY], evaluation_seasons).set_index("competition")
    for key in comparison_keys:
        frame = competition_metrics(evaluations[key], evaluation_seasons)
        frame.insert(0, "model", key)
        for metric in ("brier_1x2", "log_loss_1x2", "same_season_spearman", "same_season_pairwise_accuracy"):
            frame[f"delta_vs_current_{metric}"] = frame.apply(lambda row: row[metric] - current_comp.loc[row["competition"], metric], axis=1)
        competition_frames.append(frame)
    competition = pd.concat(competition_frames, ignore_index=True)
    uncertainty_frame = uncertainty(nested_eval, evaluations[CURRENT_KEY], evaluation_seasons, args.bootstrap_samples)
    uncertainty_frame = pd.concat(
        [uncertainty_frame, ranking_uncertainty(fold_results, forward_results, args.bootstrap_samples)],
        ignore_index=True,
        sort=False,
    )

    nested_row = comparison.loc[comparison["model"].eq(NESTED_KEY)].iloc[0]
    current_row = comparison.loc[comparison["model"].eq(CURRENT_KEY)].iloc[0]
    loss_better = nested_row["brier_1x2"] < current_row["brier_1x2"] and nested_row["log_loss_1x2"] < current_row["log_loss_1x2"]
    ranking_rows = uncertainty_frame.loc[uncertainty_frame["method"].isin(["unseen_fold_resampling", "forward_fold_resampling"])]
    reliable_rank_harm = bool(ranking_rows["reliable_harm"].any())
    verdict = (
        "PARTIAL_POOLING_BALANCED_SHADOW_SIGNAL"
        if loss_better and not reliable_rank_harm
        else "PARTIAL_POOLING_LOSS_WITH_RELIABLE_RANKING_COST"
        if loss_better
        else "NO_ADVANTAGE_OVER_CURRENT"
    )
    hash_after = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    audit = safety_audit(nested_adjustment, nested_eval, selections, hash_before, hash_after)
    if not audit["passed"].all():
        raise ValueError(f"Safety audit failed: {audit.loc[~audit['passed']].to_dict(orient='records')}")

    comparison.to_csv(output / "model_comparison.csv", index=False)
    candidate_results = pd.DataFrame([
        comparison_row(candidate.key, evaluations[candidate.key], adjustments[candidate.key], evaluation_seasons, target, identity, seasons)
        | {"parameter_json": candidate.as_json()}
        for candidate in candidates
    ])
    candidate_results.to_csv(output / "candidate_results.csv", index=False)
    selections.to_csv(output / "fold_selections.csv", index=False)
    fold_results.to_csv(output / "fold_results.csv", index=False)
    forward_results.to_csv(output / "forward_ranking.csv", index=False)
    pd.concat(posterior_frames, ignore_index=True).to_csv(output / "learned_segment_posteriors.csv", index=False)
    evidence.to_csv(output / "historical_learning_evidence.csv", index=False)
    segments.to_csv(output / "segment_analysis.csv", index=False)
    team_diagnostic.to_csv(output / "team_season_diagnostics.csv", index=False)
    competition.to_csv(output / "competition_analysis.csv", index=False)
    uncertainty_frame.to_csv(output / "dependency_uncertainty.csv", index=False)
    audit.to_csv(output / "safety_audit.csv", index=False)
    payload = {
        "status": "SHADOW_RESEARCH_ONLY", "production_changed": False, "verdict": verdict,
        "nested_fold_candidates": selections[["fold", "test_season", "selected_candidate_key"]].to_dict(orient="records"),
        "metrics": {key: float(nested_row[key]) for key in ("brier_1x2", "log_loss_1x2", "accuracy_1x2", "same_season_spearman", "same_season_pairwise_accuracy", "forward_season_spearman", "forward_season_pairwise_accuracy")},
        "production_contract_sha256": hash_after,
    }
    (output / "selected_shadow_candidate.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(output / "partial_pooling_report.md", comparison, selections, segments, competition, uncertainty_frame, verdict)
    print(f"Verdict: {verdict}")
    print(f"Report: {output / 'partial_pooling_report.md'}")


if __name__ == "__main__":
    main()
