from __future__ import annotations

"""Shadow-only European Exposure-aware Domestic Surprise experiment."""

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
from ao_elo.domestic_surprise_variance import (  # noqa: E402
    VarianceDomesticSurpriseConfig,
    calculate_variance_domestic_surprise_adjustment,
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
from scripts.run_domestic_surprise_effect_size_sensitivity import (  # noqa: E402
    add_baseline_deltas,
    add_fold_deltas,
    competition_metrics,
    fold_metrics,
    markdown_table,
    model_metrics,
    summarize_fold_stability,
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
OUTPUT_ROOT = ROOT / "output" / "domestic_surprise_exposure_nonlinear"
EXPOSURE_BINS = (-1e-12, 0.0, 0.25, 0.50, 0.75, 1.0)
EXPOSURE_LABELS = ("0", "(0,0.25]", "(0.25,0.50]", "(0.50,0.75]", "(0.75,1.00]")
CURRENT_KEY = "current_production"
NO_SURPRISE_KEY = "no_surprise"
GLOBAL_STRONG_KEY = "global_strong_theta_1p75_cap_150"


def _token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


@dataclass(frozen=True)
class ExposureScalingCandidate:
    key: str
    family: str
    theta_parameters: tuple[float, ...]
    cap_parameters: tuple[float, ...]
    complexity_parameters: int

    def validate(self) -> None:
        if self.family not in {"CONTROL", "PIECEWISE", "LINEAR", "POWER", "LOGISTIC"}:
            raise ValueError(f"Unsupported family: {self.family}")
        values = (*self.theta_parameters, *self.cap_parameters)
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("Exposure scaling parameters must be finite and non-negative")
        grid = np.linspace(0.0, 1.0, 101)
        theta = np.array([self.theta(float(value)) for value in grid])
        caps = np.array([self.cap(float(value)) for value in grid])
        if np.any(np.diff(theta) > 1e-10) or np.any(np.diff(caps) > 1e-10):
            raise ValueError(f"{self.key}: theta/cap must be non-increasing with exposure")
        if np.any(caps <= 0.0):
            raise ValueError(f"{self.key}: caps must be positive")

    def theta(self, exposure: float) -> float:
        return self._value(exposure, self.theta_parameters, is_cap=False)

    def cap(self, exposure: float) -> float:
        return self._value(exposure, self.cap_parameters, is_cap=True)

    def _value(
        self, exposure: float, parameters: tuple[float, ...], *, is_cap: bool
    ) -> float:
        if not math.isfinite(exposure) or not 0.0 <= exposure <= 1.0:
            raise ValueError("European Exposure must be finite and in [0,1]")
        if self.family == "CONTROL":
            return parameters[0]
        if self.family == "PIECEWISE":
            index = 0 if exposure == 0.0 else min(4, int(math.ceil(exposure * 4.0)))
            return parameters[index]
        if self.family == "LINEAR":
            maximum, minimum = parameters
            return minimum + (maximum - minimum) * (1.0 - exposure)
        if self.family == "POWER":
            if is_cap:
                maximum, minimum = parameters
                return minimum + (maximum - minimum) * (1.0 - exposure)
            maximum, minimum, power = parameters
            return minimum + (maximum - minimum) * (1.0 - exposure) ** power
        if self.family == "LOGISTIC":
            if is_cap:
                maximum, minimum = parameters
                return minimum + (maximum - minimum) * (1.0 - exposure)
            maximum, minimum, midpoint, steepness = parameters
            raw = lambda value: 1.0 / (1.0 + math.exp(steepness * (value - midpoint)))
            low, high = raw(1.0), raw(0.0)
            normalized = (raw(exposure) - low) / max(high - low, 1e-12)
            return minimum + (maximum - minimum) * normalized
        raise AssertionError("unreachable")

    def parameter_json(self) -> str:
        return json.dumps(
            {"theta": self.theta_parameters, "cap": self.cap_parameters},
            separators=(",", ":"),
        )


def candidate_grid(production: dict[str, object]) -> tuple[ExposureScalingCandidate, ...]:
    block = production["domestic_surprise"]
    candidates = [
        ExposureScalingCandidate(
            CURRENT_KEY,
            "CONTROL",
            (float(block["coefficient"]),),
            (float(block["max_abs_adjustment"]),),
            2,
        ),
        ExposureScalingCandidate(NO_SURPRISE_KEY, "CONTROL", (0.0,), (30.0,), 1),
        ExposureScalingCandidate(GLOBAL_STRONG_KEY, "CONTROL", (1.75,), (150.0,), 2),
    ]

    theta_profiles = (
        (0.60, 0.60, 0.55, 0.45, 0.35),
        (0.80, 0.75, 0.65, 0.50, 0.35),
        (1.00, 0.90, 0.75, 0.55, 0.35),
        (1.25, 1.10, 0.85, 0.60, 0.35),
        (1.50, 1.30, 1.00, 0.65, 0.35),
        (1.75, 1.50, 1.10, 0.70, 0.35),
        (2.00, 1.75, 1.25, 0.75, 0.35),
    )
    cap_profiles = (
        (30.0, 30.0, 30.0, 25.0, 15.0),
        (50.0, 45.0, 35.0, 25.0, 15.0),
        (75.0, 60.0, 45.0, 30.0, 15.0),
    )
    piecewise_pairs = [
        (theta_profiles[0], cap_profiles[0]),
        (theta_profiles[1], cap_profiles[0]),
        (theta_profiles[2], cap_profiles[0]),
        (theta_profiles[2], cap_profiles[1]),
        (theta_profiles[3], cap_profiles[1]),
        (theta_profiles[4], cap_profiles[1]),
        (theta_profiles[4], cap_profiles[2]),
        (theta_profiles[5], cap_profiles[1]),
        (theta_profiles[5], cap_profiles[2]),
        (theta_profiles[6], cap_profiles[1]),
        (theta_profiles[6], cap_profiles[2]),
    ]
    for index, (theta, cap) in enumerate(piecewise_pairs, start=1):
        candidates.append(
            ExposureScalingCandidate(f"piecewise_{index:02d}", "PIECEWISE", theta, cap, 10)
        )

    linear_specs = (
        (0.60, 0.40, 30.0, 15.0),
        (0.80, 0.40, 30.0, 15.0),
        (1.00, 0.40, 50.0, 15.0),
        (1.25, 0.40, 50.0, 15.0),
        (1.50, 0.40, 60.0, 15.0),
        (1.75, 0.40, 75.0, 15.0),
        (1.25, 0.25, 60.0, 15.0),
        (1.75, 0.25, 90.0, 15.0),
    )
    for maximum, minimum, cap_max, cap_min in linear_specs:
        candidates.append(
            ExposureScalingCandidate(
                f"linear_t{_token(maximum)}_{_token(minimum)}_c{_token(cap_max)}",
                "LINEAR",
                (maximum, minimum),
                (cap_max, cap_min),
                4,
            )
        )

    power_specs = (
        (0.80, 0.35, 0.50, 40.0),
        (0.80, 0.35, 1.50, 40.0),
        (1.00, 0.35, 0.50, 50.0),
        (1.00, 0.35, 1.50, 50.0),
        (1.25, 0.35, 0.75, 60.0),
        (1.25, 0.35, 1.50, 60.0),
        (1.50, 0.35, 0.75, 75.0),
        (1.50, 0.35, 1.50, 75.0),
        (1.75, 0.25, 0.75, 75.0),
        (1.75, 0.25, 1.50, 75.0),
        (2.00, 0.25, 1.50, 90.0),
        (2.00, 0.25, 2.50, 90.0),
    )
    for maximum, minimum, power, cap_max in power_specs:
        candidates.append(
            ExposureScalingCandidate(
                f"power_t{_token(maximum)}_{_token(minimum)}_p{_token(power)}_c{_token(cap_max)}",
                "POWER",
                (maximum, minimum, power),
                (cap_max, 15.0),
                5,
            )
        )

    logistic_specs = (
        (1.00, 0.35, 0.30, 5.0, 50.0),
        (1.00, 0.35, 0.50, 5.0, 50.0),
        (1.25, 0.35, 0.30, 8.0, 60.0),
        (1.25, 0.35, 0.50, 8.0, 60.0),
        (1.50, 0.30, 0.30, 5.0, 75.0),
        (1.50, 0.30, 0.50, 5.0, 75.0),
        (1.75, 0.25, 0.30, 8.0, 90.0),
        (1.75, 0.25, 0.50, 8.0, 90.0),
    )
    for maximum, minimum, midpoint, steepness, cap_max in logistic_specs:
        candidates.append(
            ExposureScalingCandidate(
                f"logistic_t{_token(maximum)}_{_token(minimum)}_m{_token(midpoint)}_s{_token(steepness)}",
                "LOGISTIC",
                (maximum, minimum, midpoint, steepness),
                (cap_max, 15.0),
                6,
            )
        )

    keys = [candidate.key for candidate in candidates]
    if len(keys) != len(set(keys)):
        raise ValueError("Candidate keys must be unique")
    for candidate in candidates:
        candidate.validate()
    return tuple(candidates)


def build_exposure_adjustments(
    features: pd.DataFrame,
    candidate: ExposureScalingCandidate,
    static_config: AOEuropeanEloConfig,
    production: dict[str, object],
) -> pd.DataFrame:
    block = production["domestic_surprise"]
    rows: list[dict[str, object]] = []
    for row in features.itertuples(index=False):
        exposure = float(row.effective_european_exposure)
        theta = candidate.theta(exposure)
        cap = candidate.cap(exposure)
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
        config = VarianceDomesticSurpriseConfig(
            coefficient=theta,
            variance_penalty=float(block["variance_penalty"]),
            max_abs_adjustment=cap,
            minimum_history_seasons=int(block["minimum_history_seasons"]),
        )
        adjustment = calculate_variance_domestic_surprise_adjustment(
            current_finish_score=float(current),
            historical_finish_scores=history,
            domestic_prior=float(row.domestic_prior),
            european_prior=float(row.european_prior),
            effective_european_exposure=exposure,
            domestic_achievement_component=static_config.domestic_achievement_component,
            achievement_scale=float(row.achievement_scale),
            config=config,
        )
        domestic_delta = adjustment.domestic_prior_adjustment
        rows.append(
            {
                "season": row.season,
                "team_id": int(row.team_id),
                "team_name": row.team_name,
                "club_id": row.club_id,
                "country_code": row.country_code,
                "competition": row.competition,
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
                    "POSITIVE" if adjustment.raw_surprise > 0 else
                    "NEGATIVE" if adjustment.raw_surprise < 0 else
                    "NEUTRAL_OR_UNAVAILABLE"
                ),
                "theta_effective": theta,
                "cap_effective": cap,
                "domestic_prior_adjustment": domestic_delta,
                "ao_first_elo_adjustment": adjustment.adjusted_ao_first_elo - row.baseline_ao_first_elo,
                "baseline_domestic_prior": row.domestic_prior,
                "adjusted_domestic_prior": adjustment.adjusted_domestic_prior,
                "baseline_ao_first_elo": row.baseline_ao_first_elo,
                "adjusted_ao_first_elo": adjustment.adjusted_ao_first_elo,
                "league_strength": row.league_strength,
                "effective_european_exposure": exposure,
                "history_seasons_available": row.history_seasons_available,
                "achievement_scale": row.achievement_scale,
            }
        )
    return pd.DataFrame(rows)


def enrich_distribution(
    adjustments: pd.DataFrame,
    candidate: ExposureScalingCandidate,
    static_config: AOEuropeanEloConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = adjustments.copy()
    frame["candidate_key"] = candidate.key
    frame["model_family"] = candidate.family
    frame["complexity_parameters"] = candidate.complexity_parameters
    frame["parameter_json"] = candidate.parameter_json()
    frame["initial_delta"] = frame["ao_first_elo_adjustment"]
    frame["changed"] = frame["initial_delta"].abs().gt(1e-12)
    frame["baseline_rank"] = frame.groupby("season")["baseline_ao_first_elo"].rank(method="min", ascending=False)
    frame["candidate_rank"] = frame.groupby("season")["adjusted_ao_first_elo"].rank(method="min", ascending=False)
    frame["rank_change"] = frame["baseline_rank"] - frame["candidate_rank"]
    frame["positive_cap_hit"] = np.isclose(frame["domestic_prior_adjustment"], frame["cap_effective"], atol=1e-9)
    frame["negative_cap_hit"] = np.isclose(frame["domestic_prior_adjustment"], -frame["cap_effective"], atol=1e-9)
    frame["exposure_band"] = pd.cut(
        frame["effective_european_exposure"],
        bins=EXPOSURE_BINS,
        labels=EXPOSURE_LABELS,
        include_lowest=True,
    )
    frame["domestic_achievement_contribution"] = (
        frame["baseline_domestic_prior"]
        - static_config.base_rating
        - static_config.domestic_league_component * frame["league_strength"]
    )
    changed = frame.loc[frame["changed"]]
    absolute = frame["initial_delta"].abs()
    effect = {
        "candidate_key": candidate.key,
        "model_family": candidate.family,
        "parameter_json": candidate.parameter_json(),
        "complexity_parameters": candidate.complexity_parameters,
        "team_seasons": len(frame),
        "changed_team_seasons": int(frame["changed"].sum()),
        "changed_share": float(frame["changed"].mean()),
        "mean_abs_initial_delta": float(absolute.mean()),
        "changed_mean_abs_initial_delta": float(changed["initial_delta"].abs().mean()) if len(changed) else 0.0,
        "median_abs_initial_delta": float(absolute.median()),
        "p75_abs_initial_delta": float(absolute.quantile(0.75)),
        "p90_abs_initial_delta": float(absolute.quantile(0.90)),
        "p95_abs_initial_delta": float(absolute.quantile(0.95)),
        "maximum_abs_initial_delta": float(absolute.max()),
        "maximum_positive_initial_delta": float(frame["initial_delta"].max()),
        "maximum_negative_initial_delta": float(frame["initial_delta"].min()),
        "positive_cap_hits": int(frame["positive_cap_hit"].sum()),
        "negative_cap_hits": int(frame["negative_cap_hit"].sum()),
        "total_cap_hit_rate": float((frame["positive_cap_hit"] | frame["negative_cap_hit"]).mean()),
        "mean_abs_rank_change": float(frame["rank_change"].abs().mean()),
        "maximum_rank_gain": int(frame["rank_change"].max()),
        "maximum_rank_loss": int(frame["rank_change"].min()),
        "minimum_initial_rating": float(frame["adjusted_ao_first_elo"].min()),
        "maximum_initial_rating": float(frame["adjusted_ao_first_elo"].max()),
    }
    return frame, effect


def forward_fold_metrics(evaluation, target: pd.DataFrame, identity: pd.DataFrame, seasons: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for target_season in seasons[3:]:
        summary = summarize_ranking(
            evaluation.end_ratings,
            target,
            allowed_target_seasons={target_season},
            identity=identity,
        )
        pooled = summary.loc[summary["competition"].eq("ALL")]
        if pooled.empty:
            continue
        source_index = seasons.index(target_season) - 1
        row = pooled.iloc[0]
        rows.append(
            {
                "fold": source_index - 1,
                "source_season": seasons[source_index],
                "target_season": target_season,
                "forward_season_spearman": float(row["ranking_score"]),
                "forward_season_pairwise_accuracy": float(row["pairwise_accuracy"]),
            }
        )
    return pd.DataFrame(rows)


def add_forward_stability(
    surface: pd.DataFrame,
    forward_folds: pd.DataFrame,
    current_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = forward_folds.loc[forward_folds["candidate_key"].eq(current_key)].set_index("target_season")
    frame = forward_folds.copy()
    for metric in ("forward_season_spearman", "forward_season_pairwise_accuracy"):
        frame[f"delta_vs_current_{metric}"] = frame.apply(
            lambda row: row[metric] - baseline.loc[row["target_season"], metric], axis=1
        )
    summary = []
    for key, group in frame.groupby("candidate_key", sort=False):
        summary.append(
            {
                "candidate_key": key,
                "forward_spearman_non_regressed_folds": int(group["delta_vs_current_forward_season_spearman"].ge(-1e-12).sum()),
                "forward_spearman_improved_folds": int(group["delta_vs_current_forward_season_spearman"].gt(1e-12).sum()),
                "forward_pairwise_non_regressed_folds": int(group["delta_vs_current_forward_season_pairwise_accuracy"].ge(-1e-12).sum()),
                "forward_pairwise_improved_folds": int(group["delta_vs_current_forward_season_pairwise_accuracy"].gt(1e-12).sum()),
            }
        )
    return surface.merge(pd.DataFrame(summary), on="candidate_key", validate="one_to_one"), frame


def exposure_band_results(distributions: pd.DataFrame, evaluations: dict[str, object]) -> pd.DataFrame:
    rows = []
    for key, frame in distributions.groupby("candidate_key", sort=False):
        predictions = evaluations[key].predictions.copy()
        lookup = frame[["season", "team_id", "exposure_band"]]
        home = predictions.merge(lookup, left_on=["season", "home_team_id"], right_on=["season", "team_id"], validate="many_to_one")
        away = predictions.merge(lookup, left_on=["season", "away_team_id"], right_on=["season", "team_id"], validate="many_to_one")
        sides = pd.concat([home, away], ignore_index=True)
        for band, group in frame.groupby("exposure_band", observed=False, sort=False):
            changed = group.loc[group["changed"]]
            match_group = sides.loc[sides["exposure_band"].astype(str).eq(str(band))]
            absolute = group["initial_delta"].abs()
            rows.append(
                {
                    "candidate_key": key,
                    "model_family": frame["model_family"].iloc[0],
                    "exposure_band": str(band),
                    "team_seasons": len(group),
                    "changed_team_seasons": int(group["changed"].sum()),
                    "mean_abs_initial_delta": float(absolute.mean()),
                    "changed_mean_abs_initial_delta": float(changed["initial_delta"].abs().mean()) if len(changed) else 0.0,
                    "median_abs_initial_delta": float(absolute.median()),
                    "p75_abs_initial_delta": float(absolute.quantile(0.75)),
                    "p90_abs_initial_delta": float(absolute.quantile(0.90)),
                    "p95_abs_initial_delta": float(absolute.quantile(0.95)),
                    "maximum_abs_initial_delta": float(absolute.max()),
                    "positive_cap_hits": int(group["positive_cap_hit"].sum()),
                    "negative_cap_hits": int(group["negative_cap_hit"].sum()),
                    "mean_abs_rank_change": float(group["rank_change"].abs().mean()),
                    "match_team_sides": len(match_group),
                    "brier_contribution": float(match_group["brier_1x2"].mean()),
                    "log_loss_contribution": float(match_group["log_loss_1x2"].mean()),
                }
            )
    return pd.DataFrame(rows)


def mark_pareto(surface: pd.DataFrame) -> pd.DataFrame:
    result = surface.copy()
    lower = ["brier_1x2", "log_loss_1x2", "complexity_parameters"]
    higher = [
        "same_season_spearman",
        "same_season_pairwise_accuracy",
        "forward_season_spearman",
        "forward_season_pairwise_accuracy",
    ]
    dominated = []
    for index, row in result.iterrows():
        is_dominated = False
        for other_index, other in result.iterrows():
            if index == other_index:
                continue
            no_worse = all(other[column] <= row[column] + 1e-12 for column in lower) and all(
                other[column] >= row[column] - 1e-12 for column in higher
            )
            strict = any(other[column] < row[column] - 1e-12 for column in lower) or any(
                other[column] > row[column] + 1e-12 for column in higher
            )
            if no_worse and strict:
                is_dominated = True
                break
        dominated.append(is_dominated)
    result["pareto_dominated"] = dominated
    result["is_pareto_frontier"] = ~result["pareto_dominated"]
    return result


def select_family_candidate(group: pd.DataFrame, current: pd.Series) -> pd.Series:
    frame = group.copy()
    ranking_columns = (
        "same_season_spearman",
        "same_season_pairwise_accuracy",
        "forward_season_spearman",
        "forward_season_pairwise_accuracy",
    )
    frame["ranking_non_regressions"] = sum(
        frame[column].ge(float(current[column]) - 1e-12).astype(int)
        for column in ranking_columns
    )
    frame["loss_improvements"] = (
        frame["brier_1x2"].lt(float(current["brier_1x2"]) - 1e-12).astype(int)
        + frame["log_loss_1x2"].lt(float(current["log_loss_1x2"]) - 1e-12).astype(int)
    )
    return frame.sort_values(
        ["ranking_non_regressions", "loss_improvements", "brier_1x2", "log_loss_1x2", "complexity_parameters"],
        ascending=[False, False, True, True, True],
        kind="stable",
    ).iloc[0]


def selected_models(surface: pd.DataFrame) -> tuple[dict[str, pd.Series], pd.Series]:
    current = surface.loc[surface["candidate_key"].eq(CURRENT_KEY)].iloc[0]
    selected: dict[str, pd.Series] = {
        "CURRENT": current,
        "NO SURPRISE": surface.loc[surface["candidate_key"].eq(NO_SURPRISE_KEY)].iloc[0],
        "GLOBAL STRONG": surface.loc[surface["candidate_key"].eq(GLOBAL_STRONG_KEY)].iloc[0],
    }
    for family in ("PIECEWISE", "LINEAR", "POWER", "LOGISTIC"):
        selected[f"BEST {family}"] = select_family_candidate(
            surface.loc[surface["model_family"].eq(family)], current
        )
    candidates = surface.loc[~surface["model_family"].isin(["CONTROL"])]
    balanced = select_family_candidate(candidates, current)
    selected["BEST BALANCED"] = balanced
    return selected, current


def dependency_uncertainty(
    evaluations: dict[str, object], selected_keys: list[str], evaluation_seasons: set[str], bootstrap_samples: int
) -> pd.DataFrame:
    rows = []
    baseline = evaluations[CURRENT_KEY].predictions
    baseline = baseline.loc[baseline["season"].isin(evaluation_seasons)]
    for key in selected_keys:
        if key == CURRENT_KEY:
            continue
        candidate = evaluations[key].predictions
        candidate = candidate.loc[candidate["season"].isin(evaluation_seasons)]
        for competition in ("ALL", "UCL", "UEL", "UECL"):
            left = candidate if competition == "ALL" else candidate.loc[candidate["competition"].eq(competition)]
            right = baseline if competition == "ALL" else baseline.loc[baseline["competition"].eq(competition)]
            paired = left.merge(right[["match_id", "brier_1x2", "log_loss_1x2"]], on="match_id", suffixes=("_candidate", "_current"), validate="one_to_one")
            for metric in ("brier_1x2", "log_loss_1x2"):
                sample = paired[["season", "match_id", "home_team_id", "away_team_id", "kickoff_utc", "tie_id"]].copy()
                sample["loss_difference"] = paired[f"{metric}_candidate"] - paired[f"{metric}_current"]
                result = dependency_robust_loss_difference_ci(sample, bootstrap_samples=bootstrap_samples)
                result.insert(0, "candidate_key", key)
                result.insert(1, "baseline_key", CURRENT_KEY)
                result.insert(2, "competition", competition)
                result.insert(3, "metric", metric)
                rows.append(result)
    return pd.concat(rows, ignore_index=True)


def sign_asymmetry(distributions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (key, direction), group in distributions.loc[
        distributions["surprise_direction"].isin(["POSITIVE", "NEGATIVE"])
    ].groupby(["candidate_key", "surprise_direction"], sort=False):
        absolute = group["initial_delta"].abs()
        rows.append(
            {
                "candidate_key": key,
                "direction": direction,
                "team_seasons": len(group),
                "mean_adjustment": float(group["initial_delta"].mean()),
                "mean_abs_adjustment": float(absolute.mean()),
                "p90_abs_adjustment": float(absolute.quantile(0.90)),
                "cap_hits": int((group["positive_cap_hit"] | group["negative_cap_hit"]).sum()),
                "mean_rank_change": float(group["rank_change"].mean()),
                "mean_abs_rank_change": float(group["rank_change"].abs().mean()),
            }
        )
    return pd.DataFrame(rows)


def adaptive_achievement_double_counting(distributions: pd.DataFrame) -> pd.DataFrame:
    def correlation(left: pd.Series, right: pd.Series, method: str) -> float:
        if left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
            return float("nan")
        return float(left.corr(right, method=method))

    rows = []
    for key, frame in distributions.groupby("candidate_key", sort=False):
        eligible = frame.loc[frame["history_seasons"].ge(5)]
        rows.append(
            {
                "candidate_key": key,
                "model_family": frame["model_family"].iloc[0],
                "eligible_team_seasons": len(eligible),
                "achievement_vs_raw_surprise_pearson": correlation(
                    eligible["domestic_achievement_contribution"], eligible["raw_surprise"], "pearson"
                ),
                "achievement_vs_raw_surprise_spearman": correlation(
                    eligible["domestic_achievement_contribution"], eligible["raw_surprise"], "spearman"
                ),
                "achievement_vs_adjustment_pearson": correlation(
                    eligible["domestic_achievement_contribution"], eligible["domestic_prior_adjustment"], "pearson"
                ),
                "achievement_vs_adjustment_spearman": correlation(
                    eligible["domestic_achievement_contribution"], eligible["domestic_prior_adjustment"], "spearman"
                ),
            }
        )
    return pd.DataFrame(rows)


def failure_analysis(
    distributions: pd.DataFrame,
    selected_key: str,
    target: pd.DataFrame,
) -> pd.DataFrame:
    target_team = (
        target.assign(weighted=lambda x: x["schedule_adjusted_score"] * x["matches"])
        .groupby(["season", "team_id"], as_index=False)
        .agg(weighted=("weighted", "sum"), matches=("matches", "sum"))
    )
    target_team["target_score"] = target_team["weighted"] / target_team["matches"]
    target_team["target_rank"] = target_team.groupby("season")["target_score"].rank(method="min", ascending=False)
    columns = [
        "season", "team_id", "team_name", "club_id", "effective_european_exposure",
        "raw_surprise", "consistency_multiplier", "initial_delta", "candidate_rank",
    ]
    current = distributions.loc[distributions["candidate_key"].eq(CURRENT_KEY), columns].rename(
        columns={"initial_delta": "current_adjustment", "candidate_rank": "current_rank"}
    )
    strong = distributions.loc[distributions["candidate_key"].eq(GLOBAL_STRONG_KEY), columns].rename(
        columns={"initial_delta": "global_strong_adjustment", "candidate_rank": "global_strong_rank"}
    ).drop(columns=["team_name", "club_id", "effective_european_exposure", "raw_surprise", "consistency_multiplier"])
    selected = distributions.loc[distributions["candidate_key"].eq(selected_key), columns].rename(
        columns={"initial_delta": "exposure_aware_adjustment", "candidate_rank": "exposure_aware_rank"}
    ).drop(columns=["team_name", "club_id", "effective_european_exposure", "raw_surprise", "consistency_multiplier"])
    result = current.merge(strong, on=["season", "team_id"], validate="one_to_one").merge(
        selected, on=["season", "team_id"], validate="one_to_one"
    ).merge(target_team[["season", "team_id", "target_score", "target_rank"]], on=["season", "team_id"], how="left", validate="one_to_one")
    result["global_rank_error_increase_vs_current"] = (
        (result["global_strong_rank"] - result["target_rank"]).abs()
        - (result["current_rank"] - result["target_rank"]).abs()
    )
    result["exposure_aware_rank_error_change_vs_current"] = (
        (result["exposure_aware_rank"] - result["target_rank"]).abs()
        - (result["current_rank"] - result["target_rank"]).abs()
    )
    return result.sort_values(
        ["global_rank_error_increase_vs_current", "global_strong_adjustment"],
        ascending=[False, False],
        kind="stable",
    ).reset_index(drop=True)


def safety_audit(
    distributions: pd.DataFrame,
    evaluations: dict[str, object],
    candidates: dict[str, ExposureScalingCandidate],
    contract_hash_before: str,
    contract_hash_after: str,
) -> pd.DataFrame:
    rows = []
    for key, frame in distributions.groupby("candidate_key", sort=False):
        evaluation = evaluations[key]
        probabilities = evaluation.predictions[["home_probability", "draw_probability", "away_probability"]]
        chronological_by_season = evaluation.predictions.groupby("season", sort=False)[
            "kickoff_utc"
        ].apply(lambda values: values.is_monotonic_increasing)
        checks = {
            "team_season_unique": not frame.duplicated(["season", "team_id"]).any(),
            "match_id_unique_per_season": not evaluation.predictions.duplicated(["season", "match_id"]).any(),
            "chronological_order_within_season": bool(chronological_by_season.all()),
            "probabilities_normalized": bool(np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)),
            "probabilities_finite_nonnegative": bool(np.isfinite(probabilities).all().all() and probabilities.ge(0.0).all().all()),
            "exposure_in_range": bool(frame["effective_european_exposure"].between(0.0, 1.0).all()),
            "insufficient_history_zero": bool(frame.loc[frame["history_seasons"].lt(5), "initial_delta"].abs().le(1e-12).all()),
            "surprise_sign_preserved": bool((frame["domestic_prior_adjustment"] * frame["raw_surprise"]).ge(-1e-12).all()),
            "adaptive_cap_respected": bool(frame["domestic_prior_adjustment"].abs().le(frame["cap_effective"] + 1e-9).all()),
            "theta_monotone": all(candidates[key].theta(float(x)) >= candidates[key].theta(float(x + 0.01)) - 1e-12 for x in np.arange(0.0, 1.0, 0.01)),
            "match_zero_sum": bool(evaluation.predictions["zero_sum_error"].abs().max() <= 1e-9),
            "power_conserved": bool(evaluation.season_metrics["season_power_conservation_error"].max() <= 1e-9),
            "finite_ratings": bool(np.isfinite(frame["adjusted_ao_first_elo"]).all() and np.isfinite(evaluation.end_ratings["end_live_rating"]).all()),
            "rating_explosion_guard": bool(frame["adjusted_ao_first_elo"].abs().max() < 5000 and evaluation.end_ratings["end_live_rating"].abs().max() < 5000),
        }
        rows.extend({"candidate_key": key, "check": check, "passed": passed} for check, passed in checks.items())
    rows.append({"candidate_key": "ALL", "check": "production_contract_unchanged", "passed": contract_hash_before == contract_hash_after})
    return pd.DataFrame(rows)


def classify_result(
    best: pd.Series,
    current: pd.Series,
    global_strong: pd.Series,
    uncertainty: pd.DataFrame,
) -> str:
    loss_better = best["brier_1x2"] < current["brier_1x2"] and best["log_loss_1x2"] < current["log_loss_1x2"]
    ranking = ["same_season_spearman", "same_season_pairwise_accuracy", "forward_season_spearman", "forward_season_pairwise_accuracy"]
    rank_non_regressed = all(best[column] >= current[column] - 1e-12 for column in ranking)
    rank_better = all(best[column] > current[column] + 1e-12 for column in ranking)
    envelope = uncertainty.loc[
        uncertainty["candidate_key"].eq(best["candidate_key"])
        & uncertainty["competition"].eq("ALL")
        & uncertainty["method"].eq("conservative_envelope")
    ]
    reliable_loss = len(envelope) == 2 and bool(envelope["reliable_improvement"].all())
    stable_same_season = bool(
        best["spearman_non_regressed_folds"] >= 4
        and best["pairwise_non_regressed_folds"] >= 4
    )
    stable_forward = bool(
        best["forward_spearman_non_regressed_folds"] >= 3
        and best["forward_pairwise_non_regressed_folds"] >= 3
    )
    global_brier_gain = max(float(current["brier_1x2"] - global_strong["brier_1x2"]), 0.0)
    global_log_gain = max(float(current["log_loss_1x2"] - global_strong["log_loss_1x2"]), 0.0)
    brier_retention = float(current["brier_1x2"] - best["brier_1x2"]) / max(global_brier_gain, 1e-12)
    log_retention = float(current["log_loss_1x2"] - best["log_loss_1x2"]) / max(global_log_gain, 1e-12)
    meaningful_retention = brier_retention >= 0.25 and log_retention >= 0.25
    if loss_better and rank_better and reliable_loss and stable_same_season and stable_forward:
        return "NONLINEAR_CLEAR_WIN"
    if loss_better and rank_non_regressed and meaningful_retention and stable_same_season and stable_forward:
        return "NONLINEAR_BALANCED_IMPROVEMENT"
    if loss_better and not rank_non_regressed:
        return "LOSS_GAIN_ONLY"
    if rank_better and not loss_better:
        return "RANKING_GAIN_ONLY"
    if best["candidate_key"] == CURRENT_KEY:
        return "NO_ADVANTAGE_OVER_CURRENT"
    return "INCONCLUSIVE"


def report_table(selected: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for label, row in selected.items():
        rows.append(
            {
                "MODEL": label,
                "candidate_key": row["candidate_key"],
                "Brier": row["brier_1x2"],
                "Log-Loss": row["log_loss_1x2"],
                "Accuracy": row["accuracy_1x2"],
                "Same-season Spearman": row["same_season_spearman"],
                "Pairwise": row["same_season_pairwise_accuracy"],
                "Forward Spearman": row["forward_season_spearman"],
                "Forward Pairwise": row["forward_season_pairwise_accuracy"],
                "changed mean abs Elo": row["changed_mean_abs_initial_delta"],
                "P90 Elo": row["p90_abs_initial_delta"],
                "P95 Elo": row["p95_abs_initial_delta"],
                "max Elo adjustment": row["maximum_abs_initial_delta"],
                "ranking non-regressed folds": f"{int(row['spearman_non_regressed_folds'])}/6",
            }
        )
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    production: dict[str, object],
    table: pd.DataFrame,
    best: pd.Series,
    current: pd.Series,
    global_strong: pd.Series,
    exposure: pd.DataFrame,
    uncertainty: pd.DataFrame,
    conclusion: str,
    candidate_count: int,
) -> None:
    exposure_best = exposure.loc[exposure["candidate_key"].eq(best["candidate_key"])]
    envelope = uncertainty.loc[
        uncertainty["candidate_key"].eq(best["candidate_key"])
        & uncertainty["competition"].eq("ALL")
        & uncertainty["method"].eq("conservative_envelope")
    ]
    brier_retention = (current["brier_1x2"] - best["brier_1x2"]) / max(
        current["brier_1x2"] - global_strong["brier_1x2"], 1e-12
    )
    log_retention = (current["log_loss_1x2"] - best["log_loss_1x2"]) / max(
        current["log_loss_1x2"] - global_strong["log_loss_1x2"], 1e-12
    )
    lines = [
        "# European Exposure-aware Non-linear Domestic Surprise",
        "",
        "Shadow/research analysis only. The production contract was not changed.",
        "",
        "## Verified production formula",
        "",
        "`Raw Surprise -> variance/consistency filter -> theta -> Domestic Prior adjustment -> cap -> (1 - effective European Exposure) attenuation -> AO First Elo adjustment`",
        "",
        f"Production remains theta `{production['domestic_surprise']['coefficient']}`, variance penalty `{production['domestic_surprise']['variance_penalty']}`, cap `+/-{production['domestic_surprise']['max_abs_adjustment']}` and five-season minimum history.",
        "",
        "## Experiment",
        "",
        f"Evaluated `{candidate_count}` candidates on the same six unseen folds and `{int(current['matches'])}` matches. Only theta and cap as functions of effective European Exposure changed.",
        "",
        "## Required model comparison",
        "",
        markdown_table(table),
        "",
        "## Best balanced exposure bands",
        "",
        markdown_table(exposure_best),
        "",
        "## Dependency-robust uncertainty vs current",
        "",
        markdown_table(envelope),
        "",
        "## Decision",
        "",
        f"Classification: **{conclusion}**.",
        "",
        f"Selected shadow candidate: `{best['candidate_key']}` ({best['model_family']}, `{int(best['complexity_parameters'])}` parameters).",
        f"Changed-team mean absolute Elo moved from `{current['changed_mean_abs_initial_delta']:.3f}` to `{best['changed_mean_abs_initial_delta']:.3f}`.",
        f"Brier delta `{best['delta_vs_current_brier_1x2']:+.9f}`; log-loss delta `{best['delta_vs_current_log_loss_1x2']:+.9f}`.",
        f"This retains only `{100.0 * brier_retention:.1f}%` of the global-strong Brier gain and `{100.0 * log_retention:.1f}%` of its log-loss gain.",
        f"Same-season Spearman delta `{best['delta_vs_current_same_season_spearman']:+.9f}`; pairwise delta `{best['delta_vs_current_same_season_pairwise_accuracy']:+.9f}`.",
        f"Forward Spearman delta `{best['delta_vs_current_forward_season_spearman']:+.9f}`; forward pairwise delta `{best['delta_vs_current_forward_season_pairwise_accuracy']:+.9f}`.",
        f"Adaptive cap hit rate is `{100.0 * best['total_cap_hit_rate']:.1f}%` versus current `{100.0 * current['total_cap_hit_rate']:.1f}%`; the cap is shaping the candidate rather than acting only as a rare guardrail.",
        "",
        "No candidate is promoted automatically. A development-window minimum is not an untouched prospective selection.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exposure-aware non-linear Domestic Surprise shadow experiment")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    args = parser.parse_args()

    production_path = PRODUCTION_CONTRACT.resolve()
    contract_hash_before = hashlib.sha256(production_path.read_bytes()).hexdigest()
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
    candidates = candidate_grid(production)
    candidates_by_key = {candidate.key: candidate for candidate in candidates}

    evaluations: dict[str, object] = {}
    distributions_all = []
    metric_rows = []
    fold_frames = []
    forward_frames = []
    competition_frames = []
    for index, candidate in enumerate(candidates, start=1):
        adjustments = build_exposure_adjustments(features, candidate, static_config, production)
        distribution, effect = enrich_distribution(adjustments, candidate, static_config)
        rating_map = {
            (str(row.season), int(row.team_id)): float(row.adjusted_ao_first_elo)
            for row in adjustments.itertuples(index=False)
        }
        if candidate.key == CURRENT_KEY:
            errors = [abs(value - production_domestic[key]) for key, value in rating_map.items()]
            if max(errors, default=0.0) > 1e-9:
                raise ValueError("Generated current ratings do not match production runtime input")
        evaluation = evaluate_arm(
            datasets,
            EvaluationArm(candidate.key, True, True, True, True, True),
            core=core,
            parameters=parameters,
            current_domestic=rating_map,
            baseline_domestic=rating_map,
            xg_map=xg_map,
            target=target,
        )
        evaluations[candidate.key] = evaluation
        metric_rows.append({**effect, **model_metrics(evaluation, evaluation_seasons, target, identity, seasons)})
        fold = fold_metrics(evaluation, folds)
        fold.insert(0, "candidate_key", candidate.key)
        fold_frames.append(fold)
        forward = forward_fold_metrics(evaluation, target, identity, seasons)
        forward.insert(0, "candidate_key", candidate.key)
        forward_frames.append(forward)
        competition = competition_metrics(evaluation, evaluation_seasons)
        competition.insert(0, "candidate_key", candidate.key)
        competition_frames.append(competition)
        distributions_all.append(distribution)
        print(f"  candidate {index}/{len(candidates)}: {candidate.key}", flush=True)

    surface = add_baseline_deltas(pd.DataFrame(metric_rows), CURRENT_KEY)
    folds_all = add_fold_deltas(pd.concat(fold_frames, ignore_index=True), CURRENT_KEY)
    surface = surface.merge(summarize_fold_stability(folds_all), on="candidate_key", validate="one_to_one")
    surface, forward_all = add_forward_stability(surface, pd.concat(forward_frames, ignore_index=True), CURRENT_KEY)
    surface = mark_pareto(surface)
    distributions = pd.concat(distributions_all, ignore_index=True)
    competition = pd.concat(competition_frames, ignore_index=True)
    current_competition = competition.loc[competition["candidate_key"].eq(CURRENT_KEY)].set_index("competition")
    for metric in ("brier_1x2", "log_loss_1x2", "accuracy_1x2", "same_season_spearman", "same_season_pairwise_accuracy"):
        competition[f"delta_vs_current_{metric}"] = competition.apply(
            lambda row: row[metric] - current_competition.loc[row["competition"], metric], axis=1
        )

    selected, current = selected_models(surface)
    best = selected["BEST BALANCED"]
    selected_keys = list(dict.fromkeys(str(row["candidate_key"]) for row in selected.values()))
    uncertainty = dependency_uncertainty(evaluations, selected_keys, evaluation_seasons, args.bootstrap_samples)
    global_strong = selected["GLOBAL STRONG"]
    conclusion = classify_result(best, current, global_strong, uncertainty)
    exposure = exposure_band_results(distributions, evaluations)
    double_counting = adaptive_achievement_double_counting(distributions)
    asymmetry = sign_asymmetry(distributions)
    failures = failure_analysis(distributions, str(best["candidate_key"]), target)
    contract_hash_after = hashlib.sha256(production_path.read_bytes()).hexdigest()
    safety = safety_audit(distributions, evaluations, candidates_by_key, contract_hash_before, contract_hash_after)
    if not safety["passed"].all():
        failed = safety.loc[~safety["passed"], ["candidate_key", "check"]]
        raise ValueError(f"Safety audit failed: {failed.to_dict(orient='records')}")

    selected_table = report_table(selected)
    family_rows = []
    for family in ("PIECEWISE", "LINEAR", "POWER", "LOGISTIC"):
        row = selected[f"BEST {family}"].copy()
        row["selected_label"] = f"BEST {family}"
        family_rows.append(row)
    family_results = pd.DataFrame(family_rows)
    adjustment_summary = surface[[column for column in surface.columns if column not in {"matches", "brier_1x2", "log_loss_1x2", "accuracy_1x2", "same_season_spearman", "same_season_pairwise_accuracy", "forward_season_spearman", "forward_season_pairwise_accuracy"}]].copy()
    top_positive = distributions.sort_values(["candidate_key", "initial_delta"], ascending=[True, False]).groupby("candidate_key", sort=False).head(15)
    top_negative = distributions.sort_values(["candidate_key", "initial_delta"], ascending=[True, True]).groupby("candidate_key", sort=False).head(15)

    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    family_results.to_csv(output / "model_family_results.csv", index=False)
    surface.to_csv(output / "parameter_grid_results.csv", index=False)
    surface.loc[surface["is_pareto_frontier"]].to_csv(output / "pareto_frontier.csv", index=False)
    forward_for_output = forward_all.rename(columns={"source_season": "test_season"})
    folds_all.merge(
        forward_for_output,
        on=["candidate_key", "fold", "test_season"],
        how="left",
        validate="one_to_one",
    ).to_csv(output / "fold_results.csv", index=False)
    exposure.to_csv(output / "exposure_band_results.csv", index=False)
    distributions.to_csv(output / "adjustment_distribution.csv", index=False)
    competition.to_csv(output / "competition_results.csv", index=False)
    uncertainty.to_csv(output / "dependency_uncertainty.csv", index=False)
    failures.to_csv(output / "failure_analysis.csv", index=False)
    top_positive.to_csv(output / "top_positive_adjustments.csv", index=False)
    top_negative.to_csv(output / "top_negative_adjustments.csv", index=False)
    asymmetry.to_csv(output / "positive_negative_asymmetry.csv", index=False)
    double_counting.to_csv(output / "achievement_double_counting.csv", index=False)
    safety.to_csv(output / "safety_audit.csv", index=False)
    selected_table.to_csv(output / "selected_model_comparison.csv", index=False)
    selected_payload = {
        "status": "SHADOW_RESEARCH_ONLY",
        "production_changed": False,
        "classification": conclusion,
        "candidate_key": str(best["candidate_key"]),
        "model_family": str(best["model_family"]),
        "complexity_parameters": int(best["complexity_parameters"]),
        "parameters": json.loads(str(best["parameter_json"])),
        "metrics": {column: float(best[column]) for column in ("brier_1x2", "log_loss_1x2", "accuracy_1x2", "same_season_spearman", "same_season_pairwise_accuracy", "forward_season_spearman", "forward_season_pairwise_accuracy")},
        "production_contract_sha256": contract_hash_after,
    }
    (output / "selected_shadow_candidate.json").write_text(json.dumps(selected_payload, indent=2), encoding="utf-8")
    write_report(
        output / "nonlinear_surprise_report.md",
        production,
        selected_table,
        best,
        current,
        global_strong,
        exposure,
        uncertainty,
        conclusion,
        len(candidates),
    )
    manifest = {
        "analysis": "DOMESTIC_SURPRISE_EXPOSURE_NONLINEAR",
        "evaluation_only": True,
        "production_changed": False,
        "production_revision": production["production_revision"],
        "production_contract_sha256": contract_hash_after,
        "candidate_count": len(candidates),
        "candidate_families": pd.Series([candidate.family for candidate in candidates]).value_counts().to_dict(),
        "evaluation_seasons": sorted(evaluation_seasons),
        "evaluation_matches": int(current["matches"]),
        "bootstrap_samples": args.bootstrap_samples,
        "best_balanced_candidate": str(best["candidate_key"]),
        "classification": conclusion,
    }
    (output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Classification: {conclusion}")
    print(f"Best balanced: {best['candidate_key']}")
    print(f"Report: {output / 'nonlinear_surprise_report.md'}")


if __name__ == "__main__":
    main()
