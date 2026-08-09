from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln


SCORE_TIME_BASIS = "90/120-minute field score excluding penalty shootouts"
DEFAULT_RHO_GRID = (-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15)
PROBABILITY_EPSILON = 1e-15


@dataclass(frozen=True)
class ScorelineModelConfig:
    mu: float
    elo_slope: float
    rho: float = 0.0
    score_time_basis: str = SCORE_TIME_BASIS
    min_matrix_goals: int = 10
    max_matrix_goals: int = 25
    tail_tolerance: float = 1e-10
    model_version: str = "ao-scoreline-dixon-coles-v1-shadow"

    def validate(self) -> None:
        _require_finite("mu", self.mu)
        _require_positive_finite("elo_slope", self.elo_slope)
        _require_finite("rho", self.rho)
        if abs(float(self.rho)) >= 1.0:
            raise ValueError("rho must be strictly between -1 and 1")
        if not isinstance(self.score_time_basis, str) or not self.score_time_basis.strip():
            raise ValueError("score_time_basis must be a non-empty string")
        _require_non_negative_integer("min_matrix_goals", self.min_matrix_goals)
        _require_non_negative_integer("max_matrix_goals", self.max_matrix_goals)
        if self.max_matrix_goals < self.min_matrix_goals:
            raise ValueError("max_matrix_goals must be >= min_matrix_goals")
        _require_positive_finite("tail_tolerance", self.tail_tolerance)
        if self.tail_tolerance >= 1.0:
            raise ValueError("tail_tolerance must be below one")
        if not isinstance(self.model_version, str) or not self.model_version.strip():
            raise ValueError("model_version must be a non-empty string")


@dataclass(frozen=True)
class ScorelinePrediction:
    lambda_home: float
    lambda_away: float
    home_probability: float
    draw_probability: float
    away_probability: float
    expected_total_goals: float
    most_likely_score: tuple[int, int]
    over_2_5_probability: float
    btts_probability: float
    covered_probability_mass: float
    score_probability_matrix: np.ndarray


@dataclass(frozen=True)
class InterceptPoissonConfig:
    lambda_home: float
    lambda_away: float
    score_time_basis: str = SCORE_TIME_BASIS
    min_matrix_goals: int = 10
    max_matrix_goals: int = 25
    tail_tolerance: float = 1e-10

    def validate(self) -> None:
        _require_positive_finite("lambda_home", self.lambda_home)
        _require_positive_finite("lambda_away", self.lambda_away)
        ScorelineModelConfig(
            mu=0.0,
            elo_slope=1.0,
            score_time_basis=self.score_time_basis,
            min_matrix_goals=self.min_matrix_goals,
            max_matrix_goals=self.max_matrix_goals,
            tail_tolerance=self.tail_tolerance,
        ).validate()


@dataclass(frozen=True)
class ScorelineBacktestResult:
    fold_parameters: pd.DataFrame
    fold_results: pd.DataFrame
    unseen_predictions: pd.DataFrame
    competition_summary: pd.DataFrame
    calibration_summary: pd.DataFrame
    segment_summary: pd.DataFrame
    dependency_uncertainty: pd.DataFrame
    selected_config: ScorelineModelConfig
    decision: str
    guardrails: dict[str, object]


def elo_goal_expectations(
    home_rating: float,
    away_rating: float,
    *,
    is_neutral: bool,
    elo_scale: float,
    home_advantage: float,
    config: ScorelineModelConfig,
) -> tuple[float, float]:
    config.validate()
    _require_finite("home_rating", home_rating)
    _require_finite("away_rating", away_rating)
    _require_boolean("is_neutral", is_neutral)
    _require_positive_finite("elo_scale", elo_scale)
    _require_non_negative_finite("home_advantage", home_advantage)
    advantage = 0.0 if is_neutral else float(home_advantage)
    z = math.log(10.0) * (
        float(home_rating) - float(away_rating) + advantage
    ) / float(elo_scale)
    home_rate = math.exp(float(config.mu) + 0.5 * float(config.elo_slope) * z)
    away_rate = math.exp(float(config.mu) - 0.5 * float(config.elo_slope) * z)
    _require_positive_finite("lambda_home", home_rate)
    _require_positive_finite("lambda_away", away_rate)
    return home_rate, away_rate


def dixon_coles_tau(
    home_goals: int,
    away_goals: int,
    lambda_home: float,
    lambda_away: float,
    rho: float,
) -> float:
    _require_non_negative_integer("home_goals", home_goals)
    _require_non_negative_integer("away_goals", away_goals)
    _require_positive_finite("lambda_home", lambda_home)
    _require_positive_finite("lambda_away", lambda_away)
    _require_finite("rho", rho)
    if (home_goals, away_goals) == (0, 0):
        value = 1.0 - lambda_home * lambda_away * rho
    elif (home_goals, away_goals) == (0, 1):
        value = 1.0 + lambda_home * rho
    elif (home_goals, away_goals) == (1, 0):
        value = 1.0 + lambda_away * rho
    elif (home_goals, away_goals) == (1, 1):
        value = 1.0 - rho
    else:
        value = 1.0
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            "Dixon-Coles correction must be positive; "
            f"score={home_goals}-{away_goals}, rho={rho}"
        )
    return float(value)


def scoreline_matrix(
    lambda_home: float,
    lambda_away: float,
    config: ScorelineModelConfig,
) -> tuple[np.ndarray, float]:
    config.validate()
    _require_positive_finite("lambda_home", lambda_home)
    _require_positive_finite("lambda_away", lambda_away)
    for goal_limit in range(config.min_matrix_goals, config.max_matrix_goals + 1):
        goals = np.arange(goal_limit + 1, dtype=float)
        home_pmf = np.exp(-lambda_home + goals * math.log(lambda_home) - gammaln(goals + 1.0))
        away_pmf = np.exp(-lambda_away + goals * math.log(lambda_away) - gammaln(goals + 1.0))
        matrix = np.outer(home_pmf, away_pmf)
        for home_goals, away_goals in ((0, 0), (0, 1), (1, 0), (1, 1)):
            matrix[home_goals, away_goals] *= dixon_coles_tau(
                home_goals,
                away_goals,
                lambda_home,
                lambda_away,
                config.rho,
            )
        covered_mass = float(matrix.sum())
        if (
            np.isfinite(matrix).all()
            and float(matrix.min()) >= 0.0
            and abs(1.0 - covered_mass) <= config.tail_tolerance
        ):
            normalized = matrix / covered_mass
            if abs(float(normalized.sum()) - 1.0) > 1e-12:
                raise ValueError("Score probability matrix failed normalization")
            return normalized, covered_mass
    raise ValueError(
        "Score probability tail exceeds tolerance at max_matrix_goals; "
        f"lambdas=({lambda_home:.6g},{lambda_away:.6g}), "
        f"covered_mass={covered_mass:.12g}"
    )


def scoreline_to_1x2(matrix: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("score_probability_matrix must be square")
    if not np.isfinite(values).all() or float(values.min()) < 0.0:
        raise ValueError("score_probability_matrix must be finite and non-negative")
    if abs(float(values.sum()) - 1.0) > 1e-10:
        raise ValueError("score_probability_matrix must sum to one")
    home = float(np.tril(values, k=-1).sum())
    draw = float(np.trace(values))
    away = float(np.triu(values, k=1).sum())
    return home, draw, away


def predict_scoreline(
    home_rating: float,
    away_rating: float,
    *,
    is_neutral: bool,
    elo_scale: float,
    home_advantage: float,
    config: ScorelineModelConfig,
) -> ScorelinePrediction:
    lambda_home, lambda_away = elo_goal_expectations(
        home_rating,
        away_rating,
        is_neutral=is_neutral,
        elo_scale=elo_scale,
        home_advantage=home_advantage,
        config=config,
    )
    matrix, covered_mass = scoreline_matrix(lambda_home, lambda_away, config)
    home, draw, away = scoreline_to_1x2(matrix)
    home_goals, away_goals = np.unravel_index(int(np.argmax(matrix)), matrix.shape)
    total_indices = np.add.outer(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
    over_2_5 = float(matrix[total_indices >= 3].sum())
    btts = float(matrix[1:, 1:].sum())
    return ScorelinePrediction(
        lambda_home=lambda_home,
        lambda_away=lambda_away,
        home_probability=home,
        draw_probability=draw,
        away_probability=away,
        expected_total_goals=lambda_home + lambda_away,
        most_likely_score=(int(home_goals), int(away_goals)),
        over_2_5_probability=over_2_5,
        btts_probability=btts,
        covered_probability_mass=covered_mass,
        score_probability_matrix=matrix,
    )


def exact_score_probability(
    home_goals: int,
    away_goals: int,
    lambda_home: float,
    lambda_away: float,
    rho: float,
) -> float:
    _require_non_negative_integer("home_goals", home_goals)
    _require_non_negative_integer("away_goals", away_goals)
    log_probability = (
        -lambda_home
        + home_goals * math.log(lambda_home)
        - math.lgamma(home_goals + 1.0)
        - lambda_away
        + away_goals * math.log(lambda_away)
        - math.lgamma(away_goals + 1.0)
    )
    probability = math.exp(log_probability) * dixon_coles_tau(
        home_goals,
        away_goals,
        lambda_home,
        lambda_away,
        rho,
    )
    if not math.isfinite(probability) or probability <= 0.0:
        raise ValueError("Exact-score probability must be positive and finite")
    return probability


def fit_elo_poisson(
    matches: pd.DataFrame,
    *,
    elo_scale: float,
    home_advantage: float,
    rho_grid: Sequence[float] = DEFAULT_RHO_GRID,
) -> ScorelineModelConfig:
    data = _validated_matches(matches, require_baseline=False)
    _require_positive_finite("elo_scale", elo_scale)
    _require_non_negative_finite("home_advantage", home_advantage)
    z = _elo_z(data, elo_scale, home_advantage)
    home_goals = data["home_goals"].to_numpy(float)
    away_goals = data["away_goals"].to_numpy(float)
    initial_mu = math.log(float((home_goals.sum() + away_goals.sum()) / (2.0 * len(data))))

    def objective(parameters: np.ndarray) -> float:
        mu, slope = float(parameters[0]), float(parameters[1])
        home_log_rate = mu + 0.5 * slope * z
        away_log_rate = mu - 0.5 * slope * z
        home_rate = np.exp(home_log_rate)
        away_rate = np.exp(away_log_rate)
        nll = (
            home_rate - home_goals * home_log_rate + gammaln(home_goals + 1.0)
            + away_rate - away_goals * away_log_rate + gammaln(away_goals + 1.0)
        )
        return float(nll.sum())

    result = minimize(
        objective,
        np.array([initial_mu, 1.0], dtype=float),
        method="L-BFGS-B",
        bounds=((math.log(0.30), math.log(4.00)), (0.05, 3.00)),
        options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 1000},
    )
    if not result.success or not np.isfinite(result.fun):
        raise ValueError(f"Poisson optimizer failed: {result.message}")
    base = ScorelineModelConfig(mu=float(result.x[0]), elo_slope=float(result.x[1]))
    return _select_rho(data, base, elo_scale, home_advantage, rho_grid)


def fit_intercept_only_poisson(matches: pd.DataFrame) -> InterceptPoissonConfig:
    data = _validated_matches(matches, require_baseline=False)
    config = InterceptPoissonConfig(
        lambda_home=float(data["home_goals"].mean()),
        lambda_away=float(data["away_goals"].mean()),
    )
    config.validate()
    return config


def evaluate_predictions(predictions: pd.DataFrame) -> dict[str, float | int]:
    required = {
        "home_goals",
        "away_goals",
        "home_probability",
        "draw_probability",
        "away_probability",
        "exact_score_probability",
        "predicted_home_goals",
        "predicted_away_goals",
        "expected_total_goals",
        "over_2_5_probability",
        "btts_probability",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"predictions missing columns: {missing}")
    if predictions.empty:
        raise ValueError("predictions cannot be empty")
    home_goals = predictions["home_goals"].to_numpy(int)
    away_goals = predictions["away_goals"].to_numpy(int)
    probabilities = predictions[
        ["home_probability", "draw_probability", "away_probability"]
    ].to_numpy(float)
    outcomes = np.where(home_goals > away_goals, 0, np.where(home_goals == away_goals, 1, 2))
    targets = np.eye(3, dtype=float)[outcomes]
    observed = probabilities[np.arange(len(predictions)), outcomes]
    over_target = (home_goals + away_goals >= 3).astype(float)
    btts_target = ((home_goals > 0) & (away_goals > 0)).astype(float)
    exact_correct = (
        (predictions["predicted_home_goals"].to_numpy(int) == home_goals)
        & (predictions["predicted_away_goals"].to_numpy(int) == away_goals)
    )
    total_goals = home_goals + away_goals
    expected_total = predictions["expected_total_goals"].to_numpy(float)
    return {
        "matches": int(len(predictions)),
        "score_nll": float(-np.log(np.clip(predictions["exact_score_probability"], PROBABILITY_EPSILON, 1.0)).mean()),
        "brier_1x2": float(np.square(probabilities - targets).sum(axis=1).mean()),
        "log_loss_1x2": float(-np.log(np.clip(observed, PROBABILITY_EPSILON, 1.0)).mean()),
        "accuracy_1x2": float((np.argmax(probabilities, axis=1) == outcomes).mean()),
        "exact_score_accuracy": float(exact_correct.mean()),
        "total_goals_mae": float(np.abs(expected_total - total_goals).mean()),
        "total_goals_bias": float((expected_total - total_goals).mean()),
        "over_2_5_brier": _binary_brier(predictions["over_2_5_probability"], over_target),
        "over_2_5_log_loss": _binary_log_loss(predictions["over_2_5_probability"], over_target),
        "btts_brier": _binary_brier(predictions["btts_probability"], btts_target),
        "btts_log_loss": _binary_log_loss(predictions["btts_probability"], btts_target),
    }


def run_scoreline_walk_forward_backtest(
    matches: pd.DataFrame,
    folds: Sequence[tuple[Sequence[str], str]],
    *,
    elo_scale: float,
    home_advantage: float,
    rho_grid: Sequence[float] = DEFAULT_RHO_GRID,
    elo_identity_preserved: bool = True,
    bootstrap_samples: int = 4000,
) -> ScorelineBacktestResult:
    data = _validated_matches(matches, require_baseline=True)
    if any(str(season).startswith("2026/") for season in data["season"]):
        raise ValueError("2026/27 untouched holdout cannot enter scoreline backtest")
    season_order = {
        season: index
        for index, season in enumerate(
            data.groupby("season", sort=False)["kickoff_utc"].min().sort_values().index
        )
    }
    fold_parameter_rows: list[dict[str, object]] = []
    fold_result_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for fold_number, (train_seasons, test_season) in enumerate(folds, start=1):
        train_seasons = tuple(str(value) for value in train_seasons)
        test_season = str(test_season)
        if test_season in train_seasons:
            raise ValueError(f"Fold {fold_number} leaks its test season into training")
        if test_season not in season_order or any(
            season not in season_order or season_order[season] >= season_order[test_season]
            for season in train_seasons
        ):
            raise ValueError(f"Fold {fold_number} training seasons must precede the test season")
        train = data.loc[data["season"].isin(train_seasons)].copy()
        test = data.loc[data["season"].eq(test_season)].copy()
        if train.empty or test.empty:
            raise ValueError(f"Fold {fold_number} has empty train or test data")
        intercept = fit_intercept_only_poisson(train)
        selected = fit_elo_poisson(
            train,
            elo_scale=elo_scale,
            home_advantage=home_advantage,
            rho_grid=rho_grid,
        )
        poisson = ScorelineModelConfig(mu=selected.mu, elo_slope=selected.elo_slope, rho=0.0)
        fold_parameter_rows.append(
            {
                "fold": fold_number,
                "train_seasons": "|".join(train_seasons),
                "test_season": test_season,
                "train_matches": len(train),
                "test_matches": len(test),
                "intercept_lambda_home": intercept.lambda_home,
                "intercept_lambda_away": intercept.lambda_away,
                "poisson_mu": selected.mu,
                "elo_slope": selected.elo_slope,
                "selected_rho": selected.rho,
            }
        )
        model_predictions: dict[str, pd.DataFrame] = {}
        for model, config, use_elo in (
            ("intercept_poisson", intercept, False),
            ("elo_poisson", poisson, True),
            ("elo_dixon_coles", selected, True),
        ):
            predicted = _predict_frame(
                test,
                config,
                elo_scale=elo_scale,
                home_advantage=home_advantage,
                use_elo=use_elo,
            )
            model_predictions[model] = predicted
            fold_result_rows.append(
                {
                    "fold": fold_number,
                    "test_season": test_season,
                    "model": model,
                    **evaluate_predictions(predicted),
                }
            )
        selected_predictions = model_predictions["elo_dixon_coles"].copy()
        selected_predictions.insert(0, "fold", fold_number)
        intercept_predictions = model_predictions["intercept_poisson"]
        poisson_predictions = model_predictions["elo_poisson"]
        selected_predictions["intercept_score_nll"] = -np.log(
            np.clip(intercept_predictions["exact_score_probability"], PROBABILITY_EPSILON, 1.0)
        )
        selected_predictions["poisson_score_nll"] = -np.log(
            np.clip(poisson_predictions["exact_score_probability"], PROBABILITY_EPSILON, 1.0)
        )
        selected_predictions["score_nll"] = -np.log(
            np.clip(selected_predictions["exact_score_probability"], PROBABILITY_EPSILON, 1.0)
        )
        _add_current_ao_losses(selected_predictions)
        prediction_frames.append(selected_predictions)

    unseen = pd.concat(prediction_frames, ignore_index=True)
    fold_results = pd.DataFrame(fold_result_rows)
    fold_results = _append_current_ao_fold_results(fold_results, unseen)
    competition_summary = _summary_by_group(unseen, "competition")
    calibration_summary = _calibration_summary(unseen)
    segment_summary = _segment_summary(unseen)
    uncertainty = _dependency_uncertainty(unseen, bootstrap_samples)
    full_config = fit_elo_poisson(
        data,
        elo_scale=elo_scale,
        home_advantage=home_advantage,
        rho_grid=rho_grid,
    )
    decision, guardrails = _promotion_decision(
        fold_results,
        unseen,
        competition_summary,
        calibration_summary,
        uncertainty,
        elo_identity_preserved,
    )
    return ScorelineBacktestResult(
        fold_parameters=pd.DataFrame(fold_parameter_rows),
        fold_results=fold_results,
        unseen_predictions=unseen,
        competition_summary=competition_summary,
        calibration_summary=calibration_summary,
        segment_summary=segment_summary,
        dependency_uncertainty=uncertainty,
        selected_config=full_config,
        decision=decision,
        guardrails=guardrails,
    )


def _predict_frame(
    matches: pd.DataFrame,
    config: ScorelineModelConfig | InterceptPoissonConfig,
    *,
    elo_scale: float,
    home_advantage: float,
    use_elo: bool,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in matches.itertuples(index=False):
        if use_elo:
            prediction = predict_scoreline(
                float(row.home_live_pre),
                float(row.away_live_pre),
                is_neutral=bool(row.is_neutral),
                elo_scale=elo_scale,
                home_advantage=home_advantage,
                config=config,
            )
        else:
            if not isinstance(config, InterceptPoissonConfig):
                raise ValueError("Intercept prediction requires InterceptPoissonConfig")
            config.validate()
            matrix_config = ScorelineModelConfig(
                mu=0.0,
                elo_slope=1.0,
                rho=0.0,
                score_time_basis=config.score_time_basis,
                min_matrix_goals=config.min_matrix_goals,
                max_matrix_goals=config.max_matrix_goals,
                tail_tolerance=config.tail_tolerance,
            )
            matrix, covered = scoreline_matrix(
                config.lambda_home,
                config.lambda_away,
                matrix_config,
            )
            home, draw, away = scoreline_to_1x2(matrix)
            predicted_home, predicted_away = np.unravel_index(int(np.argmax(matrix)), matrix.shape)
            totals = np.add.outer(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
            prediction = ScorelinePrediction(
                config.lambda_home,
                config.lambda_away,
                home,
                draw,
                away,
                config.lambda_home + config.lambda_away,
                (int(predicted_home), int(predicted_away)),
                float(matrix[totals >= 3].sum()),
                float(matrix[1:, 1:].sum()),
                covered,
                matrix,
            )
        exact = exact_score_probability(
            int(row.home_goals),
            int(row.away_goals),
            prediction.lambda_home,
            prediction.lambda_away,
            config.rho if isinstance(config, ScorelineModelConfig) else 0.0,
        )
        rows.append(
            {
                **row._asdict(),
                "lambda_home": prediction.lambda_home,
                "lambda_away": prediction.lambda_away,
                "rho": config.rho if isinstance(config, ScorelineModelConfig) else 0.0,
                "home_probability": prediction.home_probability,
                "draw_probability": prediction.draw_probability,
                "away_probability": prediction.away_probability,
                "expected_total_goals": prediction.expected_total_goals,
                "predicted_home_goals": prediction.most_likely_score[0],
                "predicted_away_goals": prediction.most_likely_score[1],
                "most_likely_score": f"{prediction.most_likely_score[0]}-{prediction.most_likely_score[1]}",
                "over_2_5_probability": prediction.over_2_5_probability,
                "btts_probability": prediction.btts_probability,
                "covered_probability_mass": prediction.covered_probability_mass,
                "exact_score_probability": exact,
            }
        )
    return pd.DataFrame(rows)


def _select_rho(
    data: pd.DataFrame,
    base: ScorelineModelConfig,
    elo_scale: float,
    home_advantage: float,
    rho_grid: Sequence[float],
) -> ScorelineModelConfig:
    candidates: list[tuple[float, float]] = []
    z = _elo_z(data, elo_scale, home_advantage)
    home_rate = np.exp(base.mu + 0.5 * base.elo_slope * z)
    away_rate = np.exp(base.mu - 0.5 * base.elo_slope * z)
    home_goals = data["home_goals"].to_numpy(int)
    away_goals = data["away_goals"].to_numpy(int)
    for rho_value in rho_grid:
        rho = float(rho_value)
        try:
            for lambda_home, lambda_away in zip(home_rate, away_rate):
                for low_home, low_away in ((0, 0), (0, 1), (1, 0), (1, 1)):
                    dixon_coles_tau(
                        low_home,
                        low_away,
                        float(lambda_home),
                        float(lambda_away),
                        rho,
                    )
            corrections = np.array([
                dixon_coles_tau(int(hg), int(ag), float(lh), float(la), rho)
                for hg, ag, lh, la in zip(home_goals, away_goals, home_rate, away_rate)
            ])
        except ValueError:
            continue
        base_log = (
            -home_rate + home_goals * np.log(home_rate) - gammaln(home_goals + 1.0)
            -away_rate + away_goals * np.log(away_rate) - gammaln(away_goals + 1.0)
        )
        nll = float(-(base_log + np.log(corrections)).mean())
        candidates.append((nll, rho))
    if not candidates:
        raise ValueError("No valid Dixon-Coles rho candidate")
    _, selected_rho = min(candidates, key=lambda item: (item[0], abs(item[1]), item[1] != 0.0, item[1]))
    return ScorelineModelConfig(mu=base.mu, elo_slope=base.elo_slope, rho=selected_rho)


def _validated_matches(matches: pd.DataFrame, *, require_baseline: bool) -> pd.DataFrame:
    required = {
        "match_id", "season", "competition", "kickoff_utc", "tie_id", "stage",
        "home_team_id", "away_team_id", "home_goals", "away_goals", "is_neutral",
        "decided_on_penalties",
        "home_live_pre", "away_live_pre",
    }
    if require_baseline:
        required |= {"ao_home_probability", "ao_draw_probability", "ao_away_probability"}
    missing = sorted(required - set(matches.columns))
    if missing:
        raise ValueError(f"matches missing columns: {missing}")
    if matches.empty:
        raise ValueError("matches cannot be empty")
    data = matches.copy()
    if data["match_id"].isna().any() or data["match_id"].duplicated().any():
        raise ValueError("match_id must be non-null and unique")
    if data[["season", "competition", "home_team_id", "away_team_id"]].isna().any().any():
        raise ValueError("season, competition, and team identifiers cannot be missing")
    if data["home_team_id"].eq(data["away_team_id"]).any():
        raise ValueError("home and away teams must differ")
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
    if not data["is_neutral"].map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError("is_neutral must contain booleans")
    if not data["decided_on_penalties"].map(
        lambda value: isinstance(value, (bool, np.bool_))
    ).all():
        raise ValueError("decided_on_penalties must contain booleans")
    data["kickoff_utc"] = pd.to_datetime(data["kickoff_utc"], utc=True, errors="raise")
    return data.sort_values(["kickoff_utc", "match_id"], kind="stable").reset_index(drop=True)


def _elo_z(data: pd.DataFrame, elo_scale: float, home_advantage: float) -> np.ndarray:
    advantage = np.where(data["is_neutral"].to_numpy(bool), 0.0, float(home_advantage))
    return math.log(10.0) * (
        data["home_live_pre"].to_numpy(float)
        - data["away_live_pre"].to_numpy(float)
        + advantage
    ) / float(elo_scale)


def _add_current_ao_losses(data: pd.DataFrame) -> None:
    home_goals = data["home_goals"].to_numpy(int)
    away_goals = data["away_goals"].to_numpy(int)
    outcomes = np.where(home_goals > away_goals, 0, np.where(home_goals == away_goals, 1, 2))
    ao = data[["ao_home_probability", "ao_draw_probability", "ao_away_probability"]].to_numpy(float)
    selected = data[["home_probability", "draw_probability", "away_probability"]].to_numpy(float)
    targets = np.eye(3)[outcomes]
    data["ao_brier_1x2"] = np.square(ao - targets).sum(axis=1)
    data["ao_log_loss_1x2"] = -np.log(np.clip(ao[np.arange(len(data)), outcomes], PROBABILITY_EPSILON, 1.0))
    data["brier_1x2"] = np.square(selected - targets).sum(axis=1)
    data["log_loss_1x2"] = -np.log(np.clip(selected[np.arange(len(data)), outcomes], PROBABILITY_EPSILON, 1.0))


def _append_current_ao_fold_results(fold_results: pd.DataFrame, unseen: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fold, season), frame in unseen.groupby(["fold", "season"], sort=True):
        home_goals = frame["home_goals"].to_numpy(int)
        away_goals = frame["away_goals"].to_numpy(int)
        outcomes = np.where(home_goals > away_goals, 0, np.where(home_goals == away_goals, 1, 2))
        probabilities = frame[["ao_home_probability", "ao_draw_probability", "ao_away_probability"]].to_numpy(float)
        rows.append({
            "fold": fold,
            "test_season": season,
            "model": "current_ao_1x2",
            "matches": len(frame),
            "score_nll": np.nan,
            "brier_1x2": float(frame["ao_brier_1x2"].mean()),
            "log_loss_1x2": float(frame["ao_log_loss_1x2"].mean()),
            "accuracy_1x2": float((np.argmax(probabilities, axis=1) == outcomes).mean()),
            "exact_score_accuracy": np.nan,
            "total_goals_mae": np.nan,
            "total_goals_bias": np.nan,
            "over_2_5_brier": np.nan,
            "over_2_5_log_loss": np.nan,
            "btts_brier": np.nan,
            "btts_log_loss": np.nan,
        })
    return pd.concat([fold_results, pd.DataFrame(rows)], ignore_index=True)


def _summary_by_group(data: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    groups = [("ALL", data), *data.groupby(column, sort=True)]
    for value, frame in groups:
        selected = evaluate_predictions(frame)
        rows.append({"segment": column, "value": value, "model": "elo_dixon_coles", **selected})
        rows.append({
            "segment": column,
            "value": value,
            "model": "current_ao_1x2",
            "matches": len(frame),
            "score_nll": np.nan,
            "brier_1x2": float(frame["ao_brier_1x2"].mean()),
            "log_loss_1x2": float(frame["ao_log_loss_1x2"].mean()),
        })
    return pd.DataFrame(rows)


def _calibration_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for competition, frame in [("ALL", data), *data.groupby("competition", sort=True)]:
        rows.append({
            "calibration": "competition",
            "segment": competition,
            "matches": len(frame),
            "mean_expected_total_goals": float(frame["expected_total_goals"].mean()),
            "mean_observed_total_goals": float((frame["home_goals"] + frame["away_goals"]).mean()),
            "mean_expected_home_goals": float(frame["lambda_home"].mean()),
            "mean_observed_home_goals": float(frame["home_goals"].mean()),
            "mean_expected_away_goals": float(frame["lambda_away"].mean()),
            "mean_observed_away_goals": float(frame["away_goals"].mean()),
            "total_goals_bias": float((frame["expected_total_goals"] - frame["home_goals"] - frame["away_goals"]).mean()),
            "mean_predicted_over_2_5": float(frame["over_2_5_probability"].mean()),
            "observed_over_2_5": float(((frame["home_goals"] + frame["away_goals"]) >= 3).mean()),
            "mean_predicted_btts": float(frame["btts_probability"].mean()),
            "observed_btts": float(((frame["home_goals"] > 0) & (frame["away_goals"] > 0)).mean()),
        })
    bins = pd.qcut(data["expected_total_goals"], q=10, duplicates="drop")
    for interval, frame in data.groupby(bins, observed=True, sort=True):
        rows.append({
            "calibration": "expected_total_decile",
            "segment": str(interval),
            "matches": len(frame),
            "mean_expected_total_goals": float(frame["expected_total_goals"].mean()),
            "mean_observed_total_goals": float((frame["home_goals"] + frame["away_goals"]).mean()),
            "mean_expected_home_goals": float(frame["lambda_home"].mean()),
            "mean_observed_home_goals": float(frame["home_goals"].mean()),
            "mean_expected_away_goals": float(frame["lambda_away"].mean()),
            "mean_observed_away_goals": float(frame["away_goals"].mean()),
            "total_goals_bias": float((frame["expected_total_goals"] - frame["home_goals"] - frame["away_goals"]).mean()),
            "mean_predicted_over_2_5": float(frame["over_2_5_probability"].mean()),
            "observed_over_2_5": float(((frame["home_goals"] + frame["away_goals"]) >= 3).mean()),
            "mean_predicted_btts": float(frame["btts_probability"].mean()),
            "observed_btts": float(((frame["home_goals"] > 0) & (frame["away_goals"] > 0)).mean()),
        })
    return pd.DataFrame(rows)


def _segment_summary(data: pd.DataFrame) -> pd.DataFrame:
    work = data.copy()
    favorite = work[["ao_home_probability", "ao_away_probability"]].max(axis=1)
    work["match_band"] = np.where(favorite < 0.45, "BALANCED", np.where(favorite < 0.65, "FAVORITE", "HEAVY_FAVORITE"))
    work["venue"] = np.where(work["is_neutral"], "NEUTRAL", "HOME_AWAY")
    work["phase"] = np.where(work["is_knockout"], "KNOCKOUT", "NON_KNOCKOUT")
    rows = []
    for segment in ("match_band", "venue", "phase", "stage"):
        for value, frame in work.groupby(segment, sort=True):
            metrics = evaluate_predictions(frame)
            rows.append({
                "segment": segment,
                "value": value,
                **metrics,
                "brier_difference_vs_ao": float((frame["brier_1x2"] - frame["ao_brier_1x2"]).mean()),
                "log_loss_difference_vs_ao": float((frame["log_loss_1x2"] - frame["ao_log_loss_1x2"]).mean()),
            })
    return pd.DataFrame(rows)


def _dependency_uncertainty(data: pd.DataFrame, bootstrap_samples: int) -> pd.DataFrame:
    from ao_elo.evaluation import dependency_robust_loss_difference_ci

    rows = []
    comparisons = {
        "score_nll_vs_intercept": data["score_nll"] - data["intercept_score_nll"],
        "brier_1x2_vs_current_ao": data["brier_1x2"] - data["ao_brier_1x2"],
        "log_loss_1x2_vs_current_ao": data["log_loss_1x2"] - data["ao_log_loss_1x2"],
    }
    for comparison, differences in comparisons.items():
        for competition, frame in [("ALL", data), *data.groupby("competition", sort=True)]:
            values = frame.copy()
            values["loss_difference"] = differences.loc[frame.index]
            ci = dependency_robust_loss_difference_ci(values, bootstrap_samples=bootstrap_samples)
            ci.insert(0, "comparison", comparison)
            ci.insert(1, "competition", competition)
            rows.append(ci)
    return pd.concat(rows, ignore_index=True)


def _promotion_decision(
    fold_results: pd.DataFrame,
    unseen: pd.DataFrame,
    competition_summary: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    uncertainty: pd.DataFrame,
    elo_identity_preserved: bool,
) -> tuple[str, dict[str, object]]:
    selected = fold_results.loc[fold_results["model"].eq("elo_dixon_coles")].set_index("fold")
    poisson = fold_results.loc[fold_results["model"].eq("elo_poisson")].set_index("fold")
    intercept = fold_results.loc[fold_results["model"].eq("intercept_poisson")].set_index("fold")
    current = fold_results.loc[fold_results["model"].eq("current_ao_1x2")].set_index("fold")
    score_wins = int((selected["score_nll"] < intercept["score_nll"]).sum())
    dixon_coles_delta = float(
        np.average(
            selected["score_nll"] - poisson["score_nll"],
            weights=selected["matches"],
        )
    )
    brier_wins = int((selected["brier_1x2"] < current["brier_1x2"]).sum())
    log_wins = int((selected["log_loss_1x2"] < current["log_loss_1x2"]).sum())
    envelopes = uncertainty.loc[
        uncertainty["method"].eq("conservative_envelope")
        & uncertainty["competition"].eq("ALL")
    ].set_index("comparison")
    score_ci = float(envelopes.loc["score_nll_vs_intercept", "ci_95_upper"])
    brier_ci = float(envelopes.loc["brier_1x2_vs_current_ao", "ci_95_upper"])
    log_ci = float(envelopes.loc["log_loss_1x2_vs_current_ao", "ci_95_upper"])
    competition_rows = competition_summary.loc[
        competition_summary["model"].eq("elo_dixon_coles")
        & competition_summary["value"].ne("ALL")
    ].set_index("value")
    competition_safe = all(
        float((unseen.loc[unseen["competition"].eq(competition), "brier_1x2"] - unseen.loc[unseen["competition"].eq(competition), "ao_brier_1x2"]).mean()) <= 0.0
        and float((unseen.loc[unseen["competition"].eq(competition), "log_loss_1x2"] - unseen.loc[unseen["competition"].eq(competition), "ao_log_loss_1x2"]).mean()) <= 0.0
        for competition in competition_rows.index
    )
    calibration = calibration_summary.loc[calibration_summary["calibration"].eq("competition")].set_index("segment")
    pooled_bias_safe = abs(float(calibration.loc["ALL", "total_goals_bias"])) <= 0.05
    competition_bias_safe = bool(
        calibration.drop(index="ALL")["total_goals_bias"].abs().le(0.15).all()
    )
    matrix_safe = bool(
        np.isfinite(unseen["covered_probability_mass"]).all()
        and (unseen["covered_probability_mass"] - 1.0).abs().le(1e-10).all()
    )
    guardrails: dict[str, object] = {
        "score_nll_fold_wins": score_wins,
        "score_nll_required_wins": 5,
        "score_nll_cluster_upper": score_ci,
        "dixon_coles_delta_vs_elo_poisson": dixon_coles_delta,
        "dixon_coles_nonzero_rho_folds": int((selected["score_nll"] != poisson["score_nll"]).sum()),
        "brier_1x2_fold_wins": brier_wins,
        "log_loss_1x2_fold_wins": log_wins,
        "one_x_two_required_wins": 4,
        "brier_1x2_cluster_upper": brier_ci,
        "log_loss_1x2_cluster_upper": log_ci,
        "competition_no_harm": competition_safe,
        "pooled_goal_bias_safe": pooled_bias_safe,
        "competition_goal_bias_safe": competition_bias_safe,
        "probability_matrix_safe": matrix_safe,
        "elo_identity_preserved": bool(elo_identity_preserved),
    }
    passed = bool(
        score_wins >= 5
        and score_ci < 0.0
        and brier_wins >= 4
        and log_wins >= 4
        and brier_ci <= 0.0
        and log_ci <= 0.0
        and competition_safe
        and pooled_bias_safe
        and competition_bias_safe
        and matrix_safe
        and elo_identity_preserved
    )
    guardrails["all_passed"] = passed
    return ("PROMOTE_CANDIDATE" if passed else "KEEP_SHADOW"), guardrails


def _binary_brier(probabilities: Iterable[float], targets: np.ndarray) -> float:
    values = np.asarray(probabilities, dtype=float)
    return float(np.square(values - targets).mean())


def _binary_log_loss(probabilities: Iterable[float], targets: np.ndarray) -> float:
    values = np.clip(np.asarray(probabilities, dtype=float), PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    return float(-(targets * np.log(values) + (1.0 - targets) * np.log(1.0 - values)).mean())


def _require_boolean(name: str, value: bool) -> None:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be boolean")


def _require_finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


def _require_positive_finite(name: str, value: float) -> None:
    _require_finite(name, value)
    if float(value) <= 0.0:
        raise ValueError(f"{name} must be positive")


def _require_non_negative_finite(name: str, value: float) -> None:
    _require_finite(name, value)
    if float(value) < 0.0:
        raise ValueError(f"{name} must be non-negative")


def _require_non_negative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer")
