from __future__ import annotations

"""Shadow-only hierarchical shrinkage research for Domestic Surprise."""

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
from scripts.run_current_model_evaluation import EvaluationArm, evaluate_arm  # noqa: E402
from scripts.run_domestic_surprise_effect_size_sensitivity import (  # noqa: E402
    add_baseline_deltas,
    add_fold_deltas,
    competition_metrics,
    fold_metrics,
    markdown_table,
    model_metrics,
    summarize_fold_stability,
)
from scripts.run_domestic_surprise_exposure_nonlinear import (  # noqa: E402
    CURRENT_KEY,
    GLOBAL_STRONG_KEY,
    NO_SURPRISE_KEY,
    ExposureScalingCandidate,
    add_forward_stability,
    build_exposure_adjustments,
    candidate_grid as exposure_candidate_grid,
    forward_fold_metrics,
    mark_pareto,
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
    load_domestic_adjustments,
    load_xg_map,
    validate_production_contract,
)
from scripts.run_v2_achievement_reserve_calibration import load_reserve_data  # noqa: E402
from scripts.run_v2_evaluation_upgrade import read_events  # noqa: E402


FEATURES_PATH = ROOT / "output/domestic_surprise_variance_backtest_2018_2026/domestic_surprise_features.csv"
EXPOSURE_SELECTION = ROOT / "output/domestic_surprise_exposure_nonlinear/selected_shadow_candidate.json"
OUTPUT_ROOT = ROOT / "output/domestic_surprise_hierarchical_shrinkage"
EXPOSURE_ONLY_KEY = "exposure_only_best"
MAGNITUDE_BINS = (-1e-12, 20.0, 30.0, 40.0, 50.0, 75.0, np.inf)
MAGNITUDE_LABELS = ("0-20", "20-30", "30-40", "40-50", "50-75", "75+")
Q_LABELS = ("Q1", "Q2", "Q3", "Q4", "Q5")


@dataclass(frozen=True)
class HierarchicalCandidate:
    key: str
    family: str
    floor: float
    final_cap: float
    parameters: tuple[float, ...]
    parameter_count: int

    def validate(self) -> None:
        if self.family not in {"MULTIPLICATIVE", "ADDITIVE", "GATED", "SMOOTH"}:
            raise ValueError(f"Unsupported hierarchical family: {self.family}")
        if not 0.0 <= self.floor <= 1.0:
            raise ValueError("Reliability floor must be in [0,1]")
        if not math.isfinite(self.final_cap) or self.final_cap <= 0.0:
            raise ValueError("Final cap must be finite and positive")
        if any(not math.isfinite(value) for value in self.parameters):
            raise ValueError("Hierarchical parameters must be finite")

    def reliability(self, league: np.ndarray, rating: np.ndarray, exposure: np.ndarray) -> np.ndarray:
        if self.family == "MULTIPLICATIVE":
            league_power, rating_power, exposure_penalty = self.parameters
            core = np.power(league, league_power) * np.power(0.5 + 0.5 * rating, rating_power)
            core *= 1.0 - exposure_penalty * exposure
        elif self.family == "ADDITIVE":
            league_weight, rating_weight, exposure_weight = self.parameters
            denominator = league_weight + rating_weight + exposure_weight
            core = (
                league_weight * league
                + rating_weight * rating
                + exposure_weight * (1.0 - exposure)
            ) / denominator
        elif self.family == "GATED":
            rating_modifier, exposure_penalty = self.parameters
            league_gate = self.floor + (1.0 - self.floor) * league
            core = league_gate * (1.0 - rating_modifier * (1.0 - rating))
            core *= 1.0 - exposure_penalty * exposure
            return np.clip(core, self.floor, 1.0)
        else:
            league_weight, rating_weight, slope, midpoint = self.parameters
            exposure_weight = max(0.0, 1.0 - league_weight - rating_weight)
            score = league_weight * league + rating_weight * rating - exposure_weight * exposure
            core = 1.0 / (1.0 + np.exp(-slope * (score - midpoint)))
        return np.clip(self.floor + (1.0 - self.floor) * core, self.floor, 1.0)

    def parameter_json(self) -> str:
        return json.dumps(
            {"floor": self.floor, "final_cap": self.final_cap, "parameters": self.parameters},
            separators=(",", ":"),
        )


def hierarchical_candidate_grid() -> tuple[HierarchicalCandidate, ...]:
    candidates: list[HierarchicalCandidate] = []
    floors = (0.20, 0.35, 0.50)
    caps = (40.0, 50.0, 60.0, 75.0, 100.0)
    specs = {
        "MULTIPLICATIVE": ((1.0, 1.0, 0.25), (0.75, 0.50, 0.15)),
        "ADDITIVE": ((0.50, 0.35, 0.15), (0.60, 0.25, 0.15)),
        "GATED": ((0.35, 0.20), (0.20, 0.10)),
        "SMOOTH": ((0.55, 0.35, 4.0, 0.35), (0.60, 0.25, 6.0, 0.30)),
    }
    for family, family_specs in specs.items():
        for spec_index, parameters in enumerate(family_specs, start=1):
            for floor in floors:
                for cap in caps:
                    key = f"{family.lower()}_{spec_index}_f{str(floor).replace('.', 'p')}_c{int(cap)}"
                    parameter_count = 2 + len(parameters)
                    candidate = HierarchicalCandidate(key, family, floor, cap, parameters, parameter_count)
                    candidate.validate()
                    candidates.append(candidate)
    return tuple(candidates)


def percentile_features(features: pd.DataFrame) -> pd.DataFrame:
    frame = features.copy()
    frame["league_reliability"] = frame.groupby("season")["league_strength"].rank(method="average", pct=True)
    frame["rating_reliability"] = frame.groupby("season")["baseline_ao_first_elo"].rank(method="average", pct=True)
    frame["league_quintile"] = frame.groupby("season")["league_strength"].transform(
        lambda values: pd.qcut(values.rank(method="first"), 5, labels=Q_LABELS)
    )
    frame["rating_quintile"] = frame.groupby("season")["baseline_ao_first_elo"].transform(
        lambda values: pd.qcut(values.rank(method="first"), 5, labels=Q_LABELS)
    )
    frame["exposure_tier"] = pd.cut(
        frame["effective_european_exposure"],
        [-1e-12, 0.25, 0.75, 1.0],
        labels=["LOW", "MEDIUM", "HIGH"],
        include_lowest=True,
    )
    return frame


def build_hierarchical_adjustments(
    features: pd.DataFrame,
    global_adjustments: pd.DataFrame,
    candidate: HierarchicalCandidate,
) -> pd.DataFrame:
    keys = ["season", "team_id"]
    columns = keys + [
        "league_reliability", "rating_reliability", "league_quintile", "rating_quintile",
        "exposure_tier", "league_strength", "baseline_ao_first_elo", "effective_european_exposure",
        "history_seasons_available", "team_name", "club_id", "country_code", "competition",
    ]
    frame = global_adjustments.merge(features[columns], on=keys, validate="one_to_one", suffixes=("", "_feature"))
    reliability = candidate.reliability(
        frame["league_reliability"].to_numpy(float),
        frame["rating_reliability"].to_numpy(float),
        frame["effective_european_exposure"].to_numpy(float),
    )
    raw = frame["ao_first_elo_adjustment"].to_numpy(float)
    final = np.clip(raw * reliability, -candidate.final_cap, candidate.final_cap)
    frame["hierarchical_reliability"] = reliability
    frame["global_strong_ao_first_adjustment"] = raw
    frame["ao_first_elo_adjustment"] = final
    frame["adjusted_ao_first_elo"] = frame["baseline_ao_first_elo"] + final
    frame["domestic_prior_adjustment"] = np.where(
        1.0 - frame["effective_european_exposure"].to_numpy(float) > 1e-12,
        final / np.maximum(1.0 - frame["effective_european_exposure"].to_numpy(float), 1e-12),
        0.0,
    )
    frame["candidate_key"] = candidate.key
    frame["model_family"] = candidate.family
    frame["parameter_count"] = candidate.parameter_count
    frame["parameter_json"] = candidate.parameter_json()
    frame["final_cap"] = candidate.final_cap
    return frame


def evaluate_ratings(datasets, key, adjustments, core, parameters, xg_map, target):
    rating_map = {
        (str(row.season), int(row.team_id)): float(row.adjusted_ao_first_elo)
        for row in adjustments.itertuples(index=False)
    }
    return evaluate_arm(
        datasets,
        EvaluationArm(key, True, True, True, True, True),
        core=core,
        parameters=parameters,
        current_domestic=rating_map,
        baseline_domestic=rating_map,
        xg_map=xg_map,
        target=target,
    )


def adjustment_effect(frame: pd.DataFrame) -> dict[str, object]:
    absolute = frame["ao_first_elo_adjustment"].abs()
    changed = absolute.gt(1e-12)
    return {
        "candidate_key": frame["candidate_key"].iloc[0],
        "model_family": frame["model_family"].iloc[0],
        "parameter_json": frame["parameter_json"].iloc[0],
        "parameter_count": int(frame["parameter_count"].iloc[0]),
        "complexity_parameters": int(frame["parameter_count"].iloc[0]),
        "team_seasons": len(frame),
        "changed_team_seasons": int(changed.sum()),
        "changed_mean_abs_initial_delta": float(absolute.loc[changed].mean()) if changed.any() else 0.0,
        "mean_abs_initial_delta": float(absolute.mean()),
        "p90_abs_initial_delta": float(absolute.quantile(0.90)),
        "p95_abs_initial_delta": float(absolute.quantile(0.95)),
        "maximum_abs_initial_delta": float(absolute.max()),
        "mean_reliability": float(frame["hierarchical_reliability"].mean()),
        "cap_hit_rate": float(np.isclose(absolute, frame["final_cap"], atol=1e-9).mean()),
    }


def target_ranks(target: pd.DataFrame) -> pd.DataFrame:
    frame = target.assign(weighted=lambda data: data["schedule_adjusted_score"] * data["matches"])
    result = frame.groupby(["season", "team_id"], as_index=False).agg(
        weighted=("weighted", "sum"), target_matches=("matches", "sum")
    )
    result["target_score"] = result["weighted"] / result["target_matches"]
    result["target_rank"] = result.groupby("season")["target_score"].rank(method="min", ascending=False)
    return result


def team_diagnostics(adjustments, evaluations, selected_keys, target, evaluation_seasons):
    targets = target_ranks(target)
    frames = []
    for key in selected_keys:
        frame = adjustments[key].loc[adjustments[key]["season"].isin(evaluation_seasons)].copy()
        end = evaluations[key].end_ratings[["season", "team_id", "end_live_rating"]]
        frame = frame.merge(end, on=["season", "team_id"], validate="one_to_one")
        frame = frame.merge(targets, on=["season", "team_id"], how="left", validate="one_to_one")
        frame["final_rank"] = frame.groupby("season")["end_live_rating"].rank(method="min", ascending=False)
        frame["rank_error"] = (frame["final_rank"] - frame["target_rank"]).abs()
        frames.append(frame)
    return {key: frame for key, frame in zip(selected_keys, frames, strict=True)}


def match_differences(current_eval, candidate_eval, evaluation_seasons):
    columns = [
        "match_id", "season", "competition", "home_team_id", "away_team_id", "brier_1x2", "log_loss_1x2"
    ]
    current = current_eval.predictions.loc[current_eval.predictions["season"].isin(evaluation_seasons), columns]
    candidate = candidate_eval.predictions.loc[candidate_eval.predictions["season"].isin(evaluation_seasons), columns]
    merged = candidate.merge(current, on=["match_id", "season", "competition", "home_team_id", "away_team_id"], suffixes=("_candidate", "_current"), validate="one_to_one")
    merged["brier_delta"] = merged["brier_1x2_candidate"] - merged["brier_1x2_current"]
    merged["log_loss_delta"] = merged["log_loss_1x2_candidate"] - merged["log_loss_1x2_current"]
    return merged


def group_analysis(candidate_key, teams, matches, group_columns):
    current = teams[CURRENT_KEY][["season", "team_id", "rank_error", "end_live_rating"]].rename(
        columns={"rank_error": "current_rank_error", "end_live_rating": "current_end_rating"}
    )
    frame = teams[candidate_key].merge(current, on=["season", "team_id"], validate="one_to_one")
    frame["rank_error_delta"] = frame["rank_error"] - frame["current_rank_error"]
    sides = pd.concat([
        matches[["season", "home_team_id", "brier_delta", "log_loss_delta"]].rename(columns={"home_team_id": "team_id"}),
        matches[["season", "away_team_id", "brier_delta", "log_loss_delta"]].rename(columns={"away_team_id": "team_id"}),
    ], ignore_index=True)
    losses = sides.groupby(["season", "team_id"], as_index=False).agg(
        matches=("brier_delta", "size"), brier_delta=("brier_delta", "mean"), log_loss_delta=("log_loss_delta", "mean")
    )
    frame = frame.merge(losses, on=["season", "team_id"], how="left", validate="one_to_one")
    rows = []
    for labels, group in frame.groupby(group_columns, observed=False, dropna=False):
        if not isinstance(labels, tuple):
            labels = (labels,)
        eligible = group.dropna(subset=["target_score"])
        current_spearman = eligible["current_end_rating"].corr(eligible["target_score"], method="spearman") if len(eligible) >= 3 else np.nan
        candidate_spearman = eligible["end_live_rating"].corr(eligible["target_score"], method="spearman") if len(eligible) >= 3 else np.nan
        current_pairwise = pairwise_ranking_accuracy(eligible["current_end_rating"].to_numpy(float), eligible["target_score"].to_numpy(float)) if len(eligible) >= 3 else np.nan
        candidate_pairwise = pairwise_ranking_accuracy(eligible["end_live_rating"].to_numpy(float), eligible["target_score"].to_numpy(float)) if len(eligible) >= 3 else np.nan
        rows.append({
            "candidate_key": candidate_key,
            **dict(zip(group_columns, map(str, labels), strict=True)),
            "team_seasons": len(group),
            "matches": int(group["matches"].fillna(0).sum()),
            "mean_exposure": float(group["effective_european_exposure"].mean()),
            "mean_league_strength": float(group["league_strength"].mean()),
            "mean_initial_rating": float(group["baseline_ao_first_elo"].mean()),
            "mean_adjustment": float(group["ao_first_elo_adjustment"].mean()),
            "mean_abs_adjustment": float(group["ao_first_elo_adjustment"].abs().mean()),
            "p90_abs_adjustment": float(group["ao_first_elo_adjustment"].abs().quantile(.90)),
            "p95_abs_adjustment": float(group["ao_first_elo_adjustment"].abs().quantile(.95)),
            "brier_delta_vs_current": float(np.average(group["brier_delta"].fillna(0), weights=group["matches"].fillna(0))) if group["matches"].fillna(0).sum() else 0.0,
            "log_loss_delta_vs_current": float(np.average(group["log_loss_delta"].fillna(0), weights=group["matches"].fillna(0))) if group["matches"].fillna(0).sum() else 0.0,
            "spearman_delta_vs_current": float(candidate_spearman - current_spearman),
            "pairwise_delta_vs_current": float(candidate_pairwise - current_pairwise),
            "mean_rank_error_delta": float(group["rank_error_delta"].mean()),
            "small_sample": len(group) < 20,
        })
    return pd.DataFrame(rows)


def competition_analysis(selected_keys, evaluations, teams, evaluation_seasons):
    rows = []
    current = competition_metrics(evaluations[CURRENT_KEY], evaluation_seasons).set_index("competition")
    for key in selected_keys:
        metrics = competition_metrics(evaluations[key], evaluation_seasons)
        for row in metrics.itertuples(index=False):
            team_ids = pd.concat([
                evaluations[key].predictions.loc[evaluations[key].predictions["competition"].eq(row.competition), ["season", "home_team_id"]].rename(columns={"home_team_id": "team_id"}),
                evaluations[key].predictions.loc[evaluations[key].predictions["competition"].eq(row.competition), ["season", "away_team_id"]].rename(columns={"away_team_id": "team_id"}),
            ]).drop_duplicates()
            subset = teams[key].merge(team_ids, on=["season", "team_id"], how="inner")
            base = current.loc[row.competition]
            rows.append({
                "candidate_key": key, "competition": row.competition, "matches": row.matches,
                "brier_1x2": row.brier_1x2, "log_loss_1x2": row.log_loss_1x2,
                "same_season_spearman": row.same_season_spearman,
                "same_season_pairwise_accuracy": row.same_season_pairwise_accuracy,
                "brier_delta_vs_current": row.brier_1x2 - base.brier_1x2,
                "log_loss_delta_vs_current": row.log_loss_1x2 - base.log_loss_1x2,
                "spearman_delta_vs_current": row.same_season_spearman - base.same_season_spearman,
                "pairwise_delta_vs_current": row.same_season_pairwise_accuracy - base.same_season_pairwise_accuracy,
                "mean_adjustment": subset["ao_first_elo_adjustment"].mean(),
                "mean_league_strength": subset["league_strength"].mean(),
                "mean_rating": subset["baseline_ao_first_elo"].mean(),
                "mean_exposure": subset["effective_european_exposure"].mean(),
            })
    return pd.DataFrame(rows)


def ranking_fold_uncertainty(folds, candidate_key, bootstrap_samples):
    rows = []
    rng = np.random.default_rng(20260811)
    for metric in ("same_season_spearman", "same_season_pairwise_accuracy", "forward_season_spearman", "forward_season_pairwise_accuracy"):
        column = f"delta_vs_current_{metric}"
        values = folds.loc[folds["candidate_key"].eq(candidate_key), column].dropna().to_numpy(float)
        if not len(values):
            continue
        samples = rng.choice(values, size=(bootstrap_samples, len(values)), replace=True).mean(axis=1)
        rows.append({
            "candidate_key": candidate_key, "competition": "ALL", "metric": metric,
            "method": "fold_resampling", "mean_difference": values.mean(),
            "ci95_lower": np.quantile(samples, .025), "ci95_upper": np.quantile(samples, .975),
            "reliable_improvement": np.quantile(samples, .975) < 0 if "loss" in metric else np.quantile(samples, .025) > 0,
            "reliable_harm": np.quantile(samples, .025) > 0 if "loss" in metric else np.quantile(samples, .975) < 0,
        })
    return pd.DataFrame(rows)


def choose_models(surface):
    current = surface.loc[surface["candidate_key"].eq(CURRENT_KEY)].iloc[0]
    hierarchical = surface.loc[surface["model_family"].isin(["MULTIPLICATIVE", "ADDITIVE", "GATED", "SMOOTH"])].copy()
    best_loss = hierarchical.sort_values(["brier_1x2", "log_loss_1x2", "parameter_count"], kind="stable").iloc[0]
    ranking_ok = hierarchical.loc[
        hierarchical["same_season_spearman"].ge(current["same_season_spearman"] - 1e-12)
        & hierarchical["same_season_pairwise_accuracy"].ge(current["same_season_pairwise_accuracy"] - 1e-12)
    ]
    forward_ok = ranking_ok.loc[
        ranking_ok["forward_season_spearman"].ge(current["forward_season_spearman"] - 1e-12)
        & ranking_ok["forward_season_pairwise_accuracy"].ge(current["forward_season_pairwise_accuracy"] - 1e-12)
    ]
    balanced_pool = forward_ok.loc[
        forward_ok["brier_1x2"].lt(current["brier_1x2"])
        & forward_ok["log_loss_1x2"].lt(current["log_loss_1x2"])
    ]
    if balanced_pool.empty:
        hierarchical["ranking_shortfall"] = (
            (current["same_season_spearman"] - hierarchical["same_season_spearman"]).clip(lower=0)
            + (current["same_season_pairwise_accuracy"] - hierarchical["same_season_pairwise_accuracy"]).clip(lower=0)
            + (current["forward_season_spearman"] - hierarchical["forward_season_spearman"]).clip(lower=0)
            + (current["forward_season_pairwise_accuracy"] - hierarchical["forward_season_pairwise_accuracy"]).clip(lower=0)
        )
        best_balanced = hierarchical.sort_values(["ranking_shortfall", "brier_1x2", "log_loss_1x2", "parameter_count"], kind="stable").iloc[0]
    else:
        best_balanced = balanced_pool.sort_values(["brier_1x2", "log_loss_1x2", "parameter_count"], kind="stable").iloc[0]
    best_ranking = hierarchical.sort_values(["same_season_spearman", "same_season_pairwise_accuracy", "brier_1x2"], ascending=[False, False, True], kind="stable").iloc[0]
    best_forward = hierarchical.sort_values(["forward_season_spearman", "forward_season_pairwise_accuracy", "brier_1x2"], ascending=[False, False, True], kind="stable").iloc[0]
    pareto = hierarchical.loc[hierarchical["is_pareto_frontier"]]
    simplest = pareto.sort_values(["parameter_count", "brier_1x2", "log_loss_1x2"], kind="stable").iloc[0]
    return {"BEST LOSS": best_loss, "BEST RANKING": best_ranking, "BEST FORWARD": best_forward, "BEST BALANCED": best_balanced, "SIMPLEST PARETO": simplest}


def retained_and_recovered(candidate, current, strong):
    result = {}
    for metric, label in (("brier_1x2", "brier"), ("log_loss_1x2", "log_loss")):
        denominator = current[metric] - strong[metric]
        result[f"retained_global_{label}_gain_pct"] = 100.0 * (current[metric] - candidate[metric]) / denominator
    for metric, label in (
        ("same_season_spearman", "same_season_spearman"),
        ("same_season_pairwise_accuracy", "pairwise"),
        ("forward_season_spearman", "forward_spearman"),
        ("forward_season_pairwise_accuracy", "forward_pairwise"),
    ):
        denominator = current[metric] - strong[metric]
        result[f"{label}_recovery_pct"] = 100.0 * (candidate[metric] - strong[metric]) / denominator if abs(denominator) > 1e-12 else np.nan
    return result


def classify(candidate, current, uncertainty):
    loss_better = candidate["brier_1x2"] < current["brier_1x2"] and candidate["log_loss_1x2"] < current["log_loss_1x2"]
    same_ok = candidate["same_season_spearman"] >= current["same_season_spearman"] and candidate["same_season_pairwise_accuracy"] >= current["same_season_pairwise_accuracy"]
    forward_ok = candidate["forward_season_spearman"] >= current["forward_season_spearman"] and candidate["forward_season_pairwise_accuracy"] >= current["forward_season_pairwise_accuracy"]
    envelope = uncertainty.loc[(uncertainty["candidate_key"].eq(candidate["candidate_key"])) & uncertainty["method"].eq("conservative_envelope")]
    reliable = len(envelope) == 2 and bool(envelope["reliable_improvement"].all())
    if loss_better and same_ok and forward_ok and reliable:
        return "HIERARCHICAL_CLEAR_WIN"
    if loss_better and same_ok and forward_ok:
        return "HIERARCHICAL_BALANCED_IMPROVEMENT"
    if loss_better and not (same_ok and forward_ok):
        return "LOSS_GAIN_WITH_MINOR_RANKING_COST"
    if not loss_better and same_ok and forward_ok:
        return "RANKING_RECOVERY_BUT_LOSS_GAIN_LOST"
    if candidate["parameter_count"] >= 10:
        return "OVERCOMPLEX_FOR_GAIN"
    if candidate["brier_1x2"] >= current["brier_1x2"] and candidate["log_loss_1x2"] >= current["log_loss_1x2"]:
        return "NO_ADVANTAGE_OVER_CURRENT"
    return "INCONCLUSIVE"


def safety_audit(adjustments, evaluations, contract_before, contract_after):
    rows = []
    for key, frame in adjustments.items():
        predictions = evaluations[key].predictions
        probabilities = predictions[["home_probability", "draw_probability", "away_probability"]]
        checks = {
            "probabilities_normalized": np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12),
            "duplicate_matches_absent": not predictions.duplicated(["season", "match_id"]).any(),
            "duplicate_team_seasons_absent": not frame.duplicated(["season", "team_id"]).any(),
            "chronology_valid": bool(predictions.groupby("season")["kickoff_utc"].apply(lambda x: x.is_monotonic_increasing).all()),
            "exposure_invariant": frame["effective_european_exposure"].between(0, 1).all(),
            "insufficient_history_zero": frame.loc[frame["history_seasons_available"].lt(5), "ao_first_elo_adjustment"].abs().le(1e-9).all(),
            "surprise_sign_preserved": (frame["ao_first_elo_adjustment"] * frame["raw_surprise"]).ge(-1e-9).all(),
            "final_cap_respected": frame["ao_first_elo_adjustment"].abs().le(frame["final_cap"] + 1e-9).all(),
            "zero_sum_match_update": predictions["zero_sum_error"].abs().max() <= 1e-9,
            "power_conservation": evaluations[key].season_metrics["season_power_conservation_error"].max() <= 1e-9,
            "no_rating_explosion": evaluations[key].end_ratings["end_live_rating"].abs().max() < 5000,
            "numerically_stable": np.isfinite(evaluations[key].end_ratings["end_live_rating"]).all(),
        }
        rows.extend({"candidate_key": key, "check": name, "passed": bool(value)} for name, value in checks.items())
    rows.append({"candidate_key": "ALL", "check": "production_contract_unchanged", "passed": contract_before == contract_after})
    return pd.DataFrame(rows)


def write_report(path, table, selections, recovery, league, competition, breakpoint, uncertainty, classification, candidate_count):
    best = selections["BEST BALANCED"]
    comparison_keys = [CURRENT_KEY, GLOBAL_STRONG_KEY, str(best["candidate_key"])]
    weak = league.loc[league["candidate_key"].isin(comparison_keys) & league["league_quintile"].eq("Q1")]
    strong = league.loc[league["candidate_key"].isin(comparison_keys) & league["league_quintile"].eq("Q5")]
    comp = competition.loc[competition["candidate_key"].isin(comparison_keys)]
    lines = [
        "# Domestic Surprise Hierarchical Shrinkage",
        "",
        "Shadow/research only. Production contract and runtime defaults were not changed.",
        "",
        "## Method",
        "",
        "Global Strong AO First adjustment was multiplied by a causal season-start reliability derived from existing league strength, pre-surprise initial rating percentile and effective European Exposure, then clipped by a final safety cap.",
        f"`{candidate_count}` hierarchical candidates across multiplicative, additive, gated and smooth families were evaluated on the same 4,884-match, six-fold window.",
        "",
        "## Main comparison",
        "",
        markdown_table(table),
        "",
        "## Decision",
        "",
        f"Classification: **{classification}**.",
        f"Best balanced candidate: `{best['candidate_key']}` with `{int(best['parameter_count'])}` parameters.",
        f"Global Brier gain retained: `{recovery['retained_global_brier_gain_pct']:.1f}%`; log-loss gain retained: `{recovery['retained_global_log_loss_gain_pct']:.1f}%`.",
        f"Same-season Spearman recovery: `{recovery['same_season_spearman_recovery_pct']:.1f}%`; pairwise recovery: `{recovery['pairwise_recovery_pct']:.1f}%`.",
        f"Forward Spearman recovery: `{recovery['forward_spearman_recovery_pct']:.1f}%`; forward pairwise recovery: `{recovery['forward_pairwise_recovery_pct']:.1f}%`.",
        "",
        "## Weakest league quintile",
        "", markdown_table(weak), "",
        "## Strongest league quintile",
        "", markdown_table(strong), "",
        "## Competition",
        "", markdown_table(comp), "",
        "## Adjustment breakpoint",
        "", markdown_table(breakpoint.loc[breakpoint["candidate_key"].eq(best["candidate_key"])]), "",
        "## Dependency-aware uncertainty",
        "", markdown_table(uncertainty.loc[uncertainty["candidate_key"].eq(best["candidate_key"])]), "",
        "## Next research direction",
        "",
        "Test a single weak-league/UECL partial-pooling prior learned only from preceding seasons; keep all other contexts at the selected shadow reliability. This directly tests whether the remaining damage is a segment-specific evidence problem.",
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
    contract_before = hashlib.sha256(contract_path.read_bytes()).hexdigest()
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
    if len(folds) != 6:
        raise ValueError("Expected six unseen folds")
    target = schedule_adjusted_team_performance(events)
    identity = load_team_season_identity()
    xg_map = load_xg_map(XG_DATA, datasets)
    features = percentile_features(pd.read_csv(FEATURES_PATH))
    production_ratings = load_domestic_adjustments(DOMESTIC_ADJUSTMENTS, datasets)

    exposure_controls = {candidate.key: candidate for candidate in exposure_candidate_grid(production)}
    exposure_payload = json.loads(EXPOSURE_SELECTION.read_text(encoding="utf-8"))
    exposure_source = exposure_controls[exposure_payload["candidate_key"]]
    controls = {
        CURRENT_KEY: exposure_controls[CURRENT_KEY],
        NO_SURPRISE_KEY: exposure_controls[NO_SURPRISE_KEY],
        GLOBAL_STRONG_KEY: exposure_controls[GLOBAL_STRONG_KEY],
        EXPOSURE_ONLY_KEY: ExposureScalingCandidate(
            EXPOSURE_ONLY_KEY, exposure_source.family, exposure_source.theta_parameters,
            exposure_source.cap_parameters, exposure_source.complexity_parameters,
        ),
    }
    adjustments = {}
    evaluations = {}
    metric_rows = []
    fold_frames = []
    forward_frames = []

    for key, candidate in controls.items():
        frame = build_exposure_adjustments(features, candidate, static_config, production)
        frame = frame.merge(
            features[["season", "team_id", "league_reliability", "rating_reliability", "league_quintile", "rating_quintile", "exposure_tier"]],
            on=["season", "team_id"],
            validate="one_to_one",
        )
        frame["hierarchical_reliability"] = 1.0
        frame["candidate_key"] = key
        frame["model_family"] = "CONTROL" if key != EXPOSURE_ONLY_KEY else "EXPOSURE_ONLY"
        frame["parameter_count"] = candidate.complexity_parameters
        frame["parameter_json"] = candidate.parameter_json()
        frame["final_cap"] = frame["cap_effective"]
        adjustments[key] = frame
    current_errors = [abs(float(row.adjusted_ao_first_elo) - production_ratings[(str(row.season), int(row.team_id))]) for row in adjustments[CURRENT_KEY].itertuples(index=False)]
    if max(current_errors, default=0.0) > 1e-9:
        raise ValueError("Current control does not match production ratings")

    global_frame = adjustments[GLOBAL_STRONG_KEY]
    hierarchical = hierarchical_candidate_grid()
    for candidate in hierarchical:
        adjustments[candidate.key] = build_hierarchical_adjustments(features, global_frame, candidate)

    for index, (key, frame) in enumerate(adjustments.items(), start=1):
        evaluation = evaluate_ratings(datasets, key, frame, core, parameters, xg_map, target)
        evaluations[key] = evaluation
        metric_rows.append({**adjustment_effect(frame), **model_metrics(evaluation, evaluation_seasons, target, identity, seasons)})
        fold = fold_metrics(evaluation, folds)
        fold.insert(0, "candidate_key", key)
        fold_frames.append(fold)
        forward = forward_fold_metrics(evaluation, target, identity, seasons)
        forward.insert(0, "candidate_key", key)
        forward_frames.append(forward)
        print(f"  candidate {index}/{len(adjustments)}: {key}", flush=True)

    surface = add_baseline_deltas(pd.DataFrame(metric_rows), CURRENT_KEY)
    fold_all = add_fold_deltas(pd.concat(fold_frames, ignore_index=True), CURRENT_KEY)
    surface = surface.merge(summarize_fold_stability(fold_all), on="candidate_key", validate="one_to_one")
    surface, forward_all = add_forward_stability(surface, pd.concat(forward_frames, ignore_index=True), CURRENT_KEY)
    surface = mark_pareto(surface)
    selections = choose_models(surface)
    current = surface.loc[surface["candidate_key"].eq(CURRENT_KEY)].iloc[0]
    global_strong = surface.loc[surface["candidate_key"].eq(GLOBAL_STRONG_KEY)].iloc[0]
    best = selections["BEST BALANCED"]
    selected_keys = list(dict.fromkeys([CURRENT_KEY, NO_SURPRISE_KEY, GLOBAL_STRONG_KEY, EXPOSURE_ONLY_KEY] + [str(row["candidate_key"]) for row in selections.values()]))
    teams = team_diagnostics(adjustments, evaluations, selected_keys, target, evaluation_seasons)
    match_maps = {key: match_differences(evaluations[CURRENT_KEY], evaluations[key], evaluation_seasons) for key in selected_keys}

    league = pd.concat([group_analysis(key, teams, match_maps[key], ["league_quintile"]) for key in selected_keys], ignore_index=True)
    rating = pd.concat([group_analysis(key, teams, match_maps[key], ["rating_quintile"]) for key in selected_keys], ignore_index=True)
    matrix = pd.concat([group_analysis(key, teams, match_maps[key], ["league_quintile", "rating_quintile"]) for key in selected_keys], ignore_index=True)
    cube = pd.concat([group_analysis(key, teams, match_maps[key], ["league_quintile", "rating_quintile", "exposure_tier"]) for key in selected_keys], ignore_index=True)
    cube = cube.loc[cube["team_seasons"].ge(20)].reset_index(drop=True)
    competition = competition_analysis(selected_keys, evaluations, teams, evaluation_seasons)

    for key in selected_keys:
        teams[key]["adjustment_band"] = pd.cut(
            teams[key]["ao_first_elo_adjustment"].abs(),
            MAGNITUDE_BINS,
            labels=MAGNITUDE_LABELS,
            include_lowest=True,
        )
    distribution = pd.concat([teams[key].assign(candidate_key=key) for key in selected_keys], ignore_index=True)
    breakpoint = pd.concat([group_analysis(key, teams, match_maps[key], ["adjustment_band"]) for key in selected_keys], ignore_index=True)
    adjustment_summary = distribution.groupby("candidate_key", as_index=False).agg(
        team_seasons=("team_id", "size"), mean_adjustment=("ao_first_elo_adjustment", "mean"),
        mean_abs_adjustment=("ao_first_elo_adjustment", lambda x: x.abs().mean()),
        p90_abs_adjustment=("ao_first_elo_adjustment", lambda x: x.abs().quantile(.90)),
        p95_abs_adjustment=("ao_first_elo_adjustment", lambda x: x.abs().quantile(.95)),
        max_abs_adjustment=("ao_first_elo_adjustment", lambda x: x.abs().max()),
        mean_reliability=("hierarchical_reliability", "mean"),
    )

    current_team = teams[CURRENT_KEY][["season", "team_id", "rank_error"]].rename(columns={"rank_error": "current_rank_error"})
    failure_rows = []
    top_files = {"top_ranking_harm.csv": [], "top_ranking_improvement.csv": [], "top_loss_gain.csv": [], "top_loss_harm.csv": []}
    for key in selected_keys:
        team = teams[key].merge(current_team, on=["season", "team_id"], validate="one_to_one")
        team["rank_error_delta"] = team["rank_error"] - team["current_rank_error"]
        match = match_maps[key]
        sides = pd.concat([
            match[["season", "home_team_id", "brier_delta", "log_loss_delta"]].rename(columns={"home_team_id": "team_id"}),
            match[["season", "away_team_id", "brier_delta", "log_loss_delta"]].rename(columns={"away_team_id": "team_id"}),
        ])
        loss = sides.groupby(["season", "team_id"], as_index=False).agg(total_brier_delta=("brier_delta", "sum"), total_log_loss_delta=("log_loss_delta", "sum"))
        team = team.merge(loss, on=["season", "team_id"], how="left", validate="one_to_one")
        failure_rows.append(team.assign(candidate_key=key))
        top_files["top_ranking_harm.csv"].append(team.nlargest(25, "rank_error_delta").assign(candidate_key=key))
        top_files["top_ranking_improvement.csv"].append(team.nsmallest(25, "rank_error_delta").assign(candidate_key=key))
        top_files["top_loss_gain.csv"].append(team.nsmallest(25, "total_brier_delta").assign(candidate_key=key))
        top_files["top_loss_harm.csv"].append(team.nlargest(25, "total_brier_delta").assign(candidate_key=key))
    failure = pd.concat(failure_rows, ignore_index=True)

    uncertainty_rows = []
    baseline_predictions = evaluations[CURRENT_KEY].predictions
    for key in selected_keys:
        if key == CURRENT_KEY:
            continue
        paired = evaluations[key].predictions.merge(
            baseline_predictions[["match_id", "brier_1x2", "log_loss_1x2"]], on="match_id", suffixes=("_candidate", "_current"), validate="one_to_one"
        )
        paired = paired.loc[paired["season"].isin(evaluation_seasons)]
        for metric in ("brier_1x2", "log_loss_1x2"):
            sample = paired[["season", "match_id", "home_team_id", "away_team_id", "kickoff_utc", "tie_id"]].copy()
            sample["loss_difference"] = paired[f"{metric}_candidate"] - paired[f"{metric}_current"]
            result = dependency_robust_loss_difference_ci(sample, bootstrap_samples=args.bootstrap_samples)
            result.insert(0, "candidate_key", key)
            result.insert(1, "competition", "ALL")
            result.insert(2, "metric", metric)
            uncertainty_rows.append(result)
    uncertainty = pd.concat(uncertainty_rows, ignore_index=True)
    folds_output = fold_all.merge(forward_all.rename(columns={"source_season": "test_season"}), on=["candidate_key", "fold", "test_season"], how="left", validate="one_to_one")
    ranking_uncertainty = ranking_fold_uncertainty(folds_output, str(best["candidate_key"]), args.bootstrap_samples)
    uncertainty = pd.concat([uncertainty, ranking_uncertainty], ignore_index=True, sort=False)

    recovery = retained_and_recovered(best, current, global_strong)
    classification = classify(best, current, uncertainty)
    contract_after = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    audit = safety_audit({key: adjustments[key] for key in selected_keys}, {key: evaluations[key] for key in selected_keys}, contract_before, contract_after)
    if not audit["passed"].all():
        raise ValueError(f"Safety audit failed: {audit.loc[~audit['passed']].to_dict(orient='records')}")

    named = {
        "CURRENT": current,
        "NO SURPRISE": surface.loc[surface["candidate_key"].eq(NO_SURPRISE_KEY)].iloc[0],
        "GLOBAL STRONG": global_strong,
        "EXPOSURE-ONLY BEST": surface.loc[surface["candidate_key"].eq(EXPOSURE_ONLY_KEY)].iloc[0],
        "BEST HIERARCHICAL": best,
        "SIMPLEST PARETO": selections["SIMPLEST PARETO"],
    }
    table = pd.DataFrame([{
        "model": label, "candidate_key": row["candidate_key"], "Brier": row["brier_1x2"], "Log-Loss": row["log_loss_1x2"],
        "Accuracy": row["accuracy_1x2"], "Same-season Spearman": row["same_season_spearman"], "Pairwise": row["same_season_pairwise_accuracy"],
        "Forward Spearman": row["forward_season_spearman"], "Forward Pairwise": row["forward_season_pairwise_accuracy"],
        "changed mean abs Elo": row["changed_mean_abs_initial_delta"], "P90 Elo": row["p90_abs_initial_delta"], "P95 Elo": row["p95_abs_initial_delta"],
        "max adjustment": row["maximum_abs_initial_delta"], "parameter count": int(row["parameter_count"]),
        "ranking non-regressed folds": f"{int(row['spearman_non_regressed_folds'])}/6; {int(row['pairwise_non_regressed_folds'])}/6",
    } for label, row in named.items()])

    selected_rows = []
    for label, row in selections.items():
        selected_rows.append({"selection": label, **row.to_dict(), **retained_and_recovered(row, current, global_strong)})
    pd.DataFrame(selected_rows).to_csv(output / "hierarchical_model_results.csv", index=False)
    surface.to_csv(output / "parameter_grid_results.csv", index=False)
    surface.loc[surface["is_pareto_frontier"]].to_csv(output / "pareto_frontier.csv", index=False)
    folds_output.to_csv(output / "fold_results.csv", index=False)
    league.to_csv(output / "league_quintile_analysis.csv", index=False)
    rating.to_csv(output / "rating_quintile_analysis.csv", index=False)
    matrix.to_csv(output / "league_rating_matrix.csv", index=False)
    cube.to_csv(output / "league_rating_exposure_profiles.csv", index=False)
    competition.to_csv(output / "competition_analysis.csv", index=False)
    adjustment_summary.to_csv(output / "adjustment_distribution.csv", index=False)
    breakpoint.to_csv(output / "breakpoint_analysis.csv", index=False)
    failure.to_csv(output / "failure_analysis.csv", index=False)
    uncertainty.to_csv(output / "dependency_uncertainty.csv", index=False)
    for filename, frames in top_files.items():
        pd.concat(frames, ignore_index=True).to_csv(output / filename, index=False)
    audit.to_csv(output / "safety_audit.csv", index=False)
    table.to_csv(output / "main_model_comparison.csv", index=False)
    payload = {
        "status": "SHADOW_RESEARCH_ONLY", "production_changed": False, "classification": classification,
        "candidate_key": str(best["candidate_key"]), "family": str(best["model_family"]),
        "parameter_count": int(best["parameter_count"]), "parameters": json.loads(str(best["parameter_json"])),
        "metrics": {name: float(best[name]) for name in ("brier_1x2", "log_loss_1x2", "accuracy_1x2", "same_season_spearman", "same_season_pairwise_accuracy", "forward_season_spearman", "forward_season_pairwise_accuracy")},
        "gain_retention_and_ranking_recovery": recovery, "production_contract_sha256": contract_after,
    }
    (output / "selected_shadow_candidate.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(output / "hierarchical_shrinkage_report.md", table, selections, recovery, league, competition, breakpoint, uncertainty, classification, len(hierarchical))
    print(f"Classification: {classification}")
    print(f"Best balanced: {best['candidate_key']}")
    print(f"Report: {output / 'hierarchical_shrinkage_report.md'}")


if __name__ == "__main__":
    main()
