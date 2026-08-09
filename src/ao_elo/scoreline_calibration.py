from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import poisson, skellam

from ao_elo.evaluation import dependency_robust_loss_difference_ci
from ao_elo.scoreline import ScorelineModelConfig, fit_elo_poisson


LEVEL_STRENGTHS = (0.0, 0.25, 0.50, 0.75, 1.0)
SEASON_LOOKBACKS = (1, 2, 3)
COMPETITIONS = ("UCL", "UEL", "UECL")
EPSILON = 1e-15


@dataclass(frozen=True, order=True)
class GoalLevelCalibrationConfig:
    competition_strength: float
    season_strength: float
    season_lookback: int

    def validate(self) -> None:
        for name in ("competition_strength", "season_strength"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and within [0,1]")
        if self.season_lookback not in SEASON_LOOKBACKS:
            raise ValueError(f"season_lookback must be one of {SEASON_LOOKBACKS}")
        if self.season_strength == 0.0 and self.season_lookback != 1:
            raise ValueError("Disabled season calibration must use lookback=1")

    @property
    def key(self) -> str:
        return (
            f"comp_{self.competition_strength:g}_"
            f"season_{self.season_strength:g}_lb{self.season_lookback}"
        )

    @property
    def is_baseline(self) -> bool:
        return self.competition_strength == 0.0 and self.season_strength == 0.0


@dataclass(frozen=True)
class FittedGoalLevelCalibration:
    config: GoalLevelCalibrationConfig
    competition_log_offsets: dict[str, float]
    season_log_offset: float
    season_source: tuple[str, ...]

    def log_offset(self, competition: str) -> float:
        if competition not in COMPETITIONS:
            raise ValueError(f"Unknown competition: {competition}")
        return float(self.competition_log_offsets.get(competition, 0.0)) + float(
            self.season_log_offset
        )


@dataclass(frozen=True)
class GoalLevelBacktestResult:
    fold_selections: pd.DataFrame
    fold_results: pd.DataFrame
    inner_candidate_metrics: pd.DataFrame
    unseen_predictions: pd.DataFrame
    competition_summary: pd.DataFrame
    season_summary: pd.DataFrame
    dependency_uncertainty: pd.DataFrame
    selected_config: GoalLevelCalibrationConfig
    fitted_full_calibration: FittedGoalLevelCalibration
    full_scoreline_config: ScorelineModelConfig
    decision: str
    guardrails: dict[str, object]


def goal_level_candidates() -> tuple[GoalLevelCalibrationConfig, ...]:
    candidates = {
        GoalLevelCalibrationConfig(competition, 0.0, 1)
        for competition in LEVEL_STRENGTHS
    }
    candidates.update(
        GoalLevelCalibrationConfig(competition, season, lookback)
        for competition in LEVEL_STRENGTHS
        for season in LEVEL_STRENGTHS
        if season > 0.0
        for lookback in SEASON_LOOKBACKS
    )
    result = tuple(sorted(candidates))
    for candidate in result:
        candidate.validate()
    return result


def fit_goal_level_calibration(
    matches: pd.DataFrame,
    scoreline_config: ScorelineModelConfig,
    calibration_config: GoalLevelCalibrationConfig,
    *,
    elo_scale: float,
    home_advantage: float,
) -> FittedGoalLevelCalibration:
    data = _validated_matches(matches)
    scoreline_config.validate()
    calibration_config.validate()
    rates = _base_rates(data, scoreline_config, elo_scale, home_advantage)
    observed_total = data["home_goals"].to_numpy(float) + data["away_goals"].to_numpy(float)
    expected_total = rates[0] + rates[1]
    competition_offsets: dict[str, float] = {}
    for competition in COMPETITIONS:
        mask = data["competition"].eq(competition).to_numpy()
        if not mask.any() or calibration_config.competition_strength == 0.0:
            competition_offsets[competition] = 0.0
            continue
        raw_offset = math.log(float(observed_total[mask].sum() / expected_total[mask].sum()))
        competition_offsets[competition] = (
            calibration_config.competition_strength * raw_offset
        )

    ordered_seasons = tuple(
        data.groupby("season", sort=False)["kickoff_utc"].min().sort_values().index
    )
    source_seasons = ordered_seasons[-calibration_config.season_lookback :]
    season_offset = 0.0
    if calibration_config.season_strength > 0.0:
        mask = data["season"].isin(source_seasons).to_numpy()
        competition_adjustment = np.exp(
            data["competition"].map(competition_offsets).to_numpy(float)
        )
        adjusted_expected = expected_total * competition_adjustment
        raw_season_offset = math.log(
            float(observed_total[mask].sum() / adjusted_expected[mask].sum())
        )
        season_offset = calibration_config.season_strength * raw_season_offset
    fitted = FittedGoalLevelCalibration(
        calibration_config,
        competition_offsets,
        float(season_offset),
        tuple(str(value) for value in source_seasons),
    )
    _validate_fitted(fitted)
    return fitted


def predict_with_goal_levels(
    matches: pd.DataFrame,
    scoreline_config: ScorelineModelConfig,
    calibration: FittedGoalLevelCalibration,
    *,
    elo_scale: float,
    home_advantage: float,
) -> pd.DataFrame:
    data = _validated_matches(matches)
    base_home, base_away = _base_rates(
        data, scoreline_config, elo_scale, home_advantage
    )
    log_offsets = np.array(
        [calibration.log_offset(str(value)) for value in data["competition"]],
        dtype=float,
    )
    multiplier = np.exp(log_offsets)
    lambda_home = base_home * multiplier
    lambda_away = base_away * multiplier
    if not np.isfinite(lambda_home).all() or not np.isfinite(lambda_away).all():
        raise ValueError("Calibrated goal rates must be finite")
    if (lambda_home <= 0.0).any() or (lambda_away <= 0.0).any():
        raise ValueError("Calibrated goal rates must be positive")
    draw_probability = skellam.pmf(0, lambda_home, lambda_away)
    away_probability = skellam.cdf(-1, lambda_home, lambda_away)
    home_probability = 1.0 - skellam.cdf(0, lambda_home, lambda_away)
    probability_sum = home_probability + draw_probability + away_probability
    home_probability /= probability_sum
    draw_probability /= probability_sum
    away_probability /= probability_sum
    total_rate = lambda_home + lambda_away
    exact_probability = (
        poisson.pmf(data["home_goals"].to_numpy(int), lambda_home)
        * poisson.pmf(data["away_goals"].to_numpy(int), lambda_away)
    )
    predicted_home = np.floor(lambda_home).astype(int)
    predicted_away = np.floor(lambda_away).astype(int)
    result = data.copy()
    result["lambda_home"] = lambda_home
    result["lambda_away"] = lambda_away
    result["goal_level_log_offset"] = log_offsets
    result["goal_level_multiplier"] = multiplier
    result["home_probability"] = home_probability
    result["draw_probability"] = draw_probability
    result["away_probability"] = away_probability
    result["expected_total_goals"] = total_rate
    result["predicted_home_goals"] = predicted_home
    result["predicted_away_goals"] = predicted_away
    result["most_likely_score"] = [
        f"{home}-{away}" for home, away in zip(predicted_home, predicted_away)
    ]
    result["over_2_5_probability"] = 1.0 - poisson.cdf(2, total_rate)
    result["btts_probability"] = (
        1.0
        - np.exp(-lambda_home)
        - np.exp(-lambda_away)
        + np.exp(-total_rate)
    )
    result["exact_score_probability"] = exact_probability
    _add_losses(result)
    return result


def run_goal_level_walk_forward_backtest(
    matches: pd.DataFrame,
    folds: Sequence[tuple[Sequence[str], str]],
    *,
    elo_scale: float,
    home_advantage: float,
    bootstrap_samples: int = 4000,
    elo_identity_preserved: bool = True,
) -> GoalLevelBacktestResult:
    data = _validated_matches(matches, require_ao=True)
    if any(str(season).startswith("2026/") for season in data["season"]):
        raise ValueError("2026/27 untouched holdout cannot enter level calibration")
    candidates = goal_level_candidates()
    selection_rows: list[dict[str, object]] = []
    inner_rows: list[dict[str, object]] = []
    fold_result_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for fold_number, (train_seasons_value, test_season_value) in enumerate(
        folds, start=1
    ):
        train_seasons = tuple(str(value) for value in train_seasons_value)
        test_season = str(test_season_value)
        _validate_fold(data, train_seasons, test_season, fold_number)
        train = data.loc[data["season"].isin(train_seasons)].copy()
        test = data.loc[data["season"].eq(test_season)].copy()
        selected, inner_metrics = _select_candidate_inner(
            train,
            train_seasons,
            candidates,
            elo_scale,
            home_advantage,
        )
        inner_metrics.insert(0, "outer_fold", fold_number)
        inner_metrics.insert(1, "outer_test_season", test_season)
        inner_rows.extend(inner_metrics.to_dict("records"))
        scoreline = fit_elo_poisson(
            train,
            elo_scale=elo_scale,
            home_advantage=home_advantage,
            rho_grid=(0.0,),
        )
        calibration = fit_goal_level_calibration(
            train,
            scoreline,
            selected,
            elo_scale=elo_scale,
            home_advantage=home_advantage,
        )
        baseline_calibration = fit_goal_level_calibration(
            train,
            scoreline,
            GoalLevelCalibrationConfig(0.0, 0.0, 1),
            elo_scale=elo_scale,
            home_advantage=home_advantage,
        )
        candidate_predictions = predict_with_goal_levels(
            test,
            scoreline,
            calibration,
            elo_scale=elo_scale,
            home_advantage=home_advantage,
        )
        baseline_predictions = predict_with_goal_levels(
            test,
            scoreline,
            baseline_calibration,
            elo_scale=elo_scale,
            home_advantage=home_advantage,
        )
        _add_ao_losses(candidate_predictions)
        candidate_predictions["baseline_score_nll"] = baseline_predictions["score_nll"]
        candidate_predictions["baseline_brier_1x2"] = baseline_predictions["brier_1x2"]
        candidate_predictions["baseline_log_loss_1x2"] = baseline_predictions["log_loss_1x2"]
        candidate_predictions.insert(0, "fold", fold_number)
        prediction_frames.append(candidate_predictions)
        candidate_metrics = _metrics(candidate_predictions)
        baseline_metrics = _metrics(baseline_predictions)
        ao_metrics = _ao_metrics(candidate_predictions)
        fold_result_rows.extend(
            [
                {"fold": fold_number, "test_season": test_season, "model": "selected_level", **candidate_metrics},
                {"fold": fold_number, "test_season": test_season, "model": "elo_poisson", **baseline_metrics},
                {"fold": fold_number, "test_season": test_season, "model": "current_ao_1x2", **ao_metrics},
            ]
        )
        selection_rows.append(
            {
                "fold": fold_number,
                "train_seasons": "|".join(train_seasons),
                "test_season": test_season,
                "candidate": selected.key,
                "competition_strength": selected.competition_strength,
                "season_strength": selected.season_strength,
                "season_lookback": selected.season_lookback,
                "mu": scoreline.mu,
                "elo_slope": scoreline.elo_slope,
                "ucl_log_offset": calibration.competition_log_offsets["UCL"],
                "uel_log_offset": calibration.competition_log_offsets["UEL"],
                "uecl_log_offset": calibration.competition_log_offsets["UECL"],
                "season_log_offset": calibration.season_log_offset,
                "season_source": "|".join(calibration.season_source),
            }
        )

    unseen = pd.concat(prediction_frames, ignore_index=True)
    folds_frame = pd.DataFrame(fold_result_rows)
    competition_summary = _group_summary(unseen, "competition")
    season_summary = _group_summary(unseen, "season")
    uncertainty = _uncertainty(unseen, bootstrap_samples)
    selected_full, full_inner = _select_candidate_inner(
        data,
        tuple(
            data.groupby("season", sort=False)["kickoff_utc"].min().sort_values().index
        ),
        candidates,
        elo_scale,
        home_advantage,
    )
    full_inner.insert(0, "outer_fold", "FULL")
    full_inner.insert(1, "outer_test_season", "FULL")
    inner_rows.extend(full_inner.to_dict("records"))
    full_scoreline = fit_elo_poisson(
        data,
        elo_scale=elo_scale,
        home_advantage=home_advantage,
        rho_grid=(0.0,),
    )
    full_calibration = fit_goal_level_calibration(
        data,
        full_scoreline,
        selected_full,
        elo_scale=elo_scale,
        home_advantage=home_advantage,
    )
    decision, guardrails = _decision(
        folds_frame,
        unseen,
        competition_summary,
        uncertainty,
        elo_identity_preserved,
    )
    return GoalLevelBacktestResult(
        fold_selections=pd.DataFrame(selection_rows),
        fold_results=folds_frame,
        inner_candidate_metrics=pd.DataFrame(inner_rows),
        unseen_predictions=unseen,
        competition_summary=competition_summary,
        season_summary=season_summary,
        dependency_uncertainty=uncertainty,
        selected_config=selected_full,
        fitted_full_calibration=full_calibration,
        full_scoreline_config=full_scoreline,
        decision=decision,
        guardrails=guardrails,
    )


def _select_candidate_inner(
    data: pd.DataFrame,
    seasons: tuple[str, ...],
    candidates: tuple[GoalLevelCalibrationConfig, ...],
    elo_scale: float,
    home_advantage: float,
) -> tuple[GoalLevelCalibrationConfig, pd.DataFrame]:
    if len(seasons) < 2:
        raise ValueError("Inner selection requires at least two seasons")
    rows: list[dict[str, object]] = []
    prediction_cache: dict[str, list[pd.DataFrame]] = {
        candidate.key: [] for candidate in candidates
    }
    for index in range(1, len(seasons)):
        inner_train_seasons = seasons[:index]
        inner_test_season = seasons[index]
        train = data.loc[data["season"].isin(inner_train_seasons)].copy()
        test = data.loc[data["season"].eq(inner_test_season)].copy()
        scoreline = fit_elo_poisson(
            train,
            elo_scale=elo_scale,
            home_advantage=home_advantage,
            rho_grid=(0.0,),
        )
        for candidate in candidates:
            fitted = fit_goal_level_calibration(
                train,
                scoreline,
                candidate,
                elo_scale=elo_scale,
                home_advantage=home_advantage,
            )
            predicted = predict_with_goal_levels(
                test,
                scoreline,
                fitted,
                elo_scale=elo_scale,
                home_advantage=home_advantage,
            )
            prediction_cache[candidate.key].append(predicted)
    baseline_key = GoalLevelCalibrationConfig(0.0, 0.0, 1).key
    baseline = pd.concat(prediction_cache[baseline_key], ignore_index=True)
    baseline_metrics = _metrics(baseline)
    for candidate in candidates:
        predictions = pd.concat(prediction_cache[candidate.key], ignore_index=True)
        metrics = _metrics(predictions)
        eligible = bool(
            metrics["brier_1x2"] <= baseline_metrics["brier_1x2"] + 1e-12
            and metrics["log_loss_1x2"] <= baseline_metrics["log_loss_1x2"] + 1e-12
        )
        rows.append(
            {
                "candidate": candidate.key,
                "competition_strength": candidate.competition_strength,
                "season_strength": candidate.season_strength,
                "season_lookback": candidate.season_lookback,
                "inner_matches": metrics["matches"],
                "score_nll": metrics["score_nll"],
                "brier_1x2": metrics["brier_1x2"],
                "log_loss_1x2": metrics["log_loss_1x2"],
                "score_nll_difference": metrics["score_nll"] - baseline_metrics["score_nll"],
                "brier_difference": metrics["brier_1x2"] - baseline_metrics["brier_1x2"],
                "log_loss_difference": metrics["log_loss_1x2"] - baseline_metrics["log_loss_1x2"],
                "eligible": eligible,
                "complexity": int(candidate.competition_strength > 0.0)
                + int(candidate.season_strength > 0.0),
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        [
            "eligible",
            "score_nll",
            "brier_1x2",
            "log_loss_1x2",
            "complexity",
            "candidate",
        ],
        ascending=[False, True, True, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    selected_row = frame.loc[frame["eligible"]].iloc[0]
    selected = next(
        candidate for candidate in candidates if candidate.key == selected_row["candidate"]
    )
    return selected, frame


def _base_rates(
    data: pd.DataFrame,
    config: ScorelineModelConfig,
    elo_scale: float,
    home_advantage: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantage = np.where(data["is_neutral"].to_numpy(bool), 0.0, home_advantage)
    z = math.log(10.0) * (
        data["home_live_pre"].to_numpy(float)
        - data["away_live_pre"].to_numpy(float)
        + advantage
    ) / float(elo_scale)
    return (
        np.exp(config.mu + 0.5 * config.elo_slope * z),
        np.exp(config.mu - 0.5 * config.elo_slope * z),
    )


def _add_losses(data: pd.DataFrame) -> None:
    home_goals = data["home_goals"].to_numpy(int)
    away_goals = data["away_goals"].to_numpy(int)
    probabilities = data[
        ["home_probability", "draw_probability", "away_probability"]
    ].to_numpy(float)
    outcomes = np.where(home_goals > away_goals, 0, np.where(home_goals == away_goals, 1, 2))
    targets = np.eye(3)[outcomes]
    data["score_nll"] = -np.log(
        np.clip(data["exact_score_probability"].to_numpy(float), EPSILON, 1.0)
    )
    data["brier_1x2"] = np.square(probabilities - targets).sum(axis=1)
    data["log_loss_1x2"] = -np.log(
        np.clip(probabilities[np.arange(len(data)), outcomes], EPSILON, 1.0)
    )


def _add_ao_losses(data: pd.DataFrame) -> None:
    home_goals = data["home_goals"].to_numpy(int)
    away_goals = data["away_goals"].to_numpy(int)
    outcomes = np.where(home_goals > away_goals, 0, np.where(home_goals == away_goals, 1, 2))
    targets = np.eye(3)[outcomes]
    probabilities = data[
        ["ao_home_probability", "ao_draw_probability", "ao_away_probability"]
    ].to_numpy(float)
    data["ao_brier_1x2"] = np.square(probabilities - targets).sum(axis=1)
    data["ao_log_loss_1x2"] = -np.log(
        np.clip(probabilities[np.arange(len(data)), outcomes], EPSILON, 1.0)
    )


def _metrics(data: pd.DataFrame) -> dict[str, float | int]:
    observed_total = data["home_goals"] + data["away_goals"]
    return {
        "matches": int(len(data)),
        "score_nll": float(data["score_nll"].mean()),
        "brier_1x2": float(data["brier_1x2"].mean()),
        "log_loss_1x2": float(data["log_loss_1x2"].mean()),
        "total_goals_bias": float((data["expected_total_goals"] - observed_total).mean()),
        "total_goals_mae": float((data["expected_total_goals"] - observed_total).abs().mean()),
    }


def _ao_metrics(data: pd.DataFrame) -> dict[str, float | int]:
    return {
        "matches": int(len(data)),
        "score_nll": float("nan"),
        "brier_1x2": float(data["ao_brier_1x2"].mean()),
        "log_loss_1x2": float(data["ao_log_loss_1x2"].mean()),
        "total_goals_bias": float("nan"),
        "total_goals_mae": float("nan"),
    }


def _group_summary(data: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for value, frame in [("ALL", data), *data.groupby(column, sort=True)]:
        candidate = _metrics(frame)
        rows.append(
            {
                "segment": column,
                "value": value,
                "model": "selected_level",
                **candidate,
                "score_nll_difference_vs_elo_poisson": float((frame["score_nll"] - frame["baseline_score_nll"]).mean()),
                "brier_difference_vs_current_ao": float((frame["brier_1x2"] - frame["ao_brier_1x2"]).mean()),
                "log_loss_difference_vs_current_ao": float((frame["log_loss_1x2"] - frame["ao_log_loss_1x2"]).mean()),
            }
        )
    return pd.DataFrame(rows)


def _uncertainty(data: pd.DataFrame, bootstrap_samples: int) -> pd.DataFrame:
    comparisons = {
        "score_nll_vs_elo_poisson": data["score_nll"] - data["baseline_score_nll"],
        "brier_1x2_vs_current_ao": data["brier_1x2"] - data["ao_brier_1x2"],
        "log_loss_1x2_vs_current_ao": data["log_loss_1x2"] - data["ao_log_loss_1x2"],
    }
    rows = []
    for comparison, differences in comparisons.items():
        for competition, frame in [("ALL", data), *data.groupby("competition", sort=True)]:
            values = frame.copy()
            values["loss_difference"] = differences.loc[frame.index]
            ci = dependency_robust_loss_difference_ci(
                values,
                bootstrap_samples=bootstrap_samples,
            )
            ci.insert(0, "comparison", comparison)
            ci.insert(1, "competition", competition)
            rows.append(ci)
    return pd.concat(rows, ignore_index=True)


def _decision(
    fold_results: pd.DataFrame,
    unseen: pd.DataFrame,
    competition_summary: pd.DataFrame,
    uncertainty: pd.DataFrame,
    elo_identity_preserved: bool,
) -> tuple[str, dict[str, object]]:
    candidate = fold_results.loc[fold_results["model"].eq("selected_level")].set_index("fold")
    poisson_base = fold_results.loc[fold_results["model"].eq("elo_poisson")].set_index("fold")
    ao_base = fold_results.loc[fold_results["model"].eq("current_ao_1x2")].set_index("fold")
    score_wins = int((candidate["score_nll"] < poisson_base["score_nll"]).sum())
    brier_wins = int((candidate["brier_1x2"] < ao_base["brier_1x2"]).sum())
    log_wins = int((candidate["log_loss_1x2"] < ao_base["log_loss_1x2"]).sum())
    envelopes = uncertainty.loc[
        uncertainty["method"].eq("conservative_envelope")
        & uncertainty["competition"].eq("ALL")
    ].set_index("comparison")
    score_upper = float(envelopes.loc["score_nll_vs_elo_poisson", "ci_95_upper"])
    brier_upper = float(envelopes.loc["brier_1x2_vs_current_ao", "ci_95_upper"])
    log_upper = float(envelopes.loc["log_loss_1x2_vs_current_ao", "ci_95_upper"])
    competition = competition_summary.loc[competition_summary["value"].ne("ALL")]
    competition_safe = bool(
        competition["brier_difference_vs_current_ao"].le(0.0).all()
        and competition["log_loss_difference_vs_current_ao"].le(0.0).all()
    )
    pooled_bias = float(
        competition_summary.loc[competition_summary["value"].eq("ALL"), "total_goals_bias"].iloc[0]
    )
    competition_bias_safe = bool(competition["total_goals_bias"].abs().le(0.15).all())
    nonzero_folds = int(
        (~unseen.groupby("fold")["goal_level_log_offset"].apply(lambda values: np.allclose(values, 0.0))).sum()
    )
    guardrails: dict[str, object] = {
        "score_nll_fold_wins": score_wins,
        "score_nll_required_wins": 4,
        "score_nll_cluster_upper": score_upper,
        "brier_1x2_fold_wins": brier_wins,
        "log_loss_1x2_fold_wins": log_wins,
        "one_x_two_required_wins": 4,
        "brier_1x2_cluster_upper": brier_upper,
        "log_loss_1x2_cluster_upper": log_upper,
        "competition_no_harm": competition_safe,
        "pooled_goal_bias": pooled_bias,
        "pooled_goal_bias_safe": abs(pooled_bias) <= 0.05,
        "competition_goal_bias_safe": competition_bias_safe,
        "nonzero_calibration_folds": nonzero_folds,
        "elo_identity_preserved": bool(elo_identity_preserved),
    }
    passed = bool(
        score_wins >= 4
        and score_upper <= 0.0
        and brier_wins >= 4
        and log_wins >= 4
        and brier_upper <= 0.0
        and log_upper <= 0.0
        and competition_safe
        and abs(pooled_bias) <= 0.05
        and competition_bias_safe
        and nonzero_folds > 0
        and elo_identity_preserved
    )
    guardrails["all_passed"] = passed
    return ("PROMOTE_CANDIDATE" if passed else "KEEP_SHADOW"), guardrails


def _validated_matches(matches: pd.DataFrame, *, require_ao: bool = False) -> pd.DataFrame:
    required = {
        "match_id", "season", "competition", "kickoff_utc", "tie_id",
        "home_team_id", "away_team_id", "home_goals", "away_goals",
        "is_neutral", "home_live_pre", "away_live_pre",
    }
    if require_ao:
        required |= {"ao_home_probability", "ao_draw_probability", "ao_away_probability"}
    missing = sorted(required - set(matches.columns))
    if missing:
        raise ValueError(f"matches missing columns: {missing}")
    if matches.empty:
        raise ValueError("matches cannot be empty")
    data = matches.copy()
    if data["match_id"].isna().any() or data["match_id"].duplicated().any():
        raise ValueError("match_id must be non-null and unique")
    unknown = sorted(set(data["competition"]) - set(COMPETITIONS))
    if unknown:
        raise ValueError(f"Unknown competitions: {unknown}")
    for column in ("home_goals", "away_goals"):
        numeric = pd.to_numeric(data[column], errors="coerce")
        if numeric.isna().any() or (numeric < 0).any() or not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError(f"{column} must contain non-negative integers")
        data[column] = numeric.astype(int)
    for column in ("home_live_pre", "away_live_pre"):
        numeric = pd.to_numeric(data[column], errors="coerce")
        if numeric.isna().any() or not np.isfinite(numeric).all():
            raise ValueError(f"{column} must be finite numeric")
        data[column] = numeric.astype(float)
    data["kickoff_utc"] = pd.to_datetime(data["kickoff_utc"], utc=True, errors="raise")
    return data.sort_values(["kickoff_utc", "match_id"], kind="stable").reset_index(drop=True)


def _validate_fold(
    data: pd.DataFrame,
    train_seasons: tuple[str, ...],
    test_season: str,
    fold_number: int,
) -> None:
    season_order = {
        season: index
        for index, season in enumerate(
            data.groupby("season", sort=False)["kickoff_utc"].min().sort_values().index
        )
    }
    if test_season in train_seasons or test_season not in season_order:
        raise ValueError(f"Fold {fold_number} has an invalid test season")
    if any(
        season not in season_order or season_order[season] >= season_order[test_season]
        for season in train_seasons
    ):
        raise ValueError(f"Fold {fold_number} training seasons must precede test")


def _validate_fitted(calibration: FittedGoalLevelCalibration) -> None:
    calibration.config.validate()
    if set(calibration.competition_log_offsets) != set(COMPETITIONS):
        raise ValueError("Fitted calibration must contain all competitions")
    values = (*calibration.competition_log_offsets.values(), calibration.season_log_offset)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Goal-level offsets must be finite")
