from __future__ import annotations

from collections.abc import Iterable, Sequence
from math import isfinite

import numpy as np
import pandas as pd


RANKING_MATCH_COLUMNS = {
    "season",
    "competition",
    "home_team_id",
    "away_team_id",
    "home_goals",
    "away_goals",
}
UNCERTAINTY_COLUMNS = {
    "season",
    "match_id",
    "home_team_id",
    "away_team_id",
    "kickoff_utc",
}
PROBABILITY_COLUMNS = ("home_probability", "draw_probability", "away_probability")


def schedule_adjusted_team_performance(
    matches: pd.DataFrame,
    *,
    home_edge_prior_matches: float = 20.0,
    opponent_prior_matches: float = 4.0,
    team_prior_matches: float = 2.0,
) -> pd.DataFrame:
    """Build a leave-team-out, venue- and schedule-adjusted ranking target.

    The target is deliberately independent of AO Elo. For each team, home edge is
    estimated without that team's matches and every opponent strength estimate
    excludes the direct meetings with the target team. The resulting score is a
    ranking index and is not clipped to the probability interval.
    """
    _validate_non_negative_finite(
        home_edge_prior_matches,
        "home_edge_prior_matches",
    )
    _validate_non_negative_finite(
        opponent_prior_matches,
        "opponent_prior_matches",
    )
    _validate_non_negative_finite(team_prior_matches, "team_prior_matches")
    data = _validated_match_frame(matches, RANKING_MATCH_COLUMNS)
    rows: list[dict[str, object]] = []

    for (season, competition), segment in data.groupby(
        ["season", "competition"],
        sort=True,
        dropna=False,
    ):
        long = _long_team_results(segment)
        teams = pd.unique(
            pd.concat(
                [segment["home_team_id"], segment["away_team_id"]],
                ignore_index=True,
            )
        )
        for team_id in teams:
            home_training = segment.loc[
                segment["home_team_id"].ne(team_id)
                & segment["away_team_id"].ne(team_id)
                & ~segment["is_neutral"]
            ]
            raw_home_edge = (
                float((home_training["actual_home_score"] - 0.5).mean())
                if len(home_training)
                else 0.0
            )
            home_reliability = _reliability(
                len(home_training),
                home_edge_prior_matches,
            )
            home_edge = home_reliability * raw_home_edge

            target = long.loc[long["team_id"].eq(team_id)].copy()
            target["venue_neutral_score"] = (
                target["score"] - target["venue_sign"] * home_edge
            )

            opponent_training = long.loc[long["opponent_id"].ne(team_id)].copy()
            opponent_training["venue_neutral_score"] = (
                opponent_training["score"]
                - opponent_training["venue_sign"] * home_edge
            )
            opponent_stats = opponent_training.groupby("team_id", sort=False)[
                "venue_neutral_score"
            ].agg(["sum", "count"])
            opponent_strength = (
                opponent_stats["sum"] + 0.5 * opponent_prior_matches
            ) / (opponent_stats["count"] + opponent_prior_matches)
            target["opponent_strength"] = target["opponent_id"].map(
                opponent_strength
            ).fillna(0.5)
            target["schedule_adjusted_match_score"] = (
                target["venue_neutral_score"]
                + target["opponent_strength"]
                - 0.5
            )

            unshrunk = float(target["schedule_adjusted_match_score"].mean())
            reliability = _reliability(len(target), team_prior_matches)
            adjusted = 0.5 + reliability * (unshrunk - 0.5)
            rows.append(
                {
                    "season": season,
                    "competition": competition,
                    "team_id": team_id,
                    "matches": int(len(target)),
                    "raw_score_rate": float(target["score"].mean()),
                    "home_edge_training_matches": int(len(home_training)),
                    "leave_team_out_home_edge": float(home_edge),
                    "venue_neutral_score_rate": float(
                        target["venue_neutral_score"].mean()
                    ),
                    "mean_opponent_strength": float(
                        target["opponent_strength"].mean()
                    ),
                    "schedule_adjusted_score_unshrunk": unshrunk,
                    "schedule_adjusted_score": float(adjusted),
                    "target_reliability": float(reliability),
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["season", "competition", "team_id"],
        kind="stable",
    ).reset_index(drop=True)


def score_preserving_1x2_probabilities(
    expected_home_score: Iterable[float] | float,
    *,
    draw_at_even: float,
    draw_shape: float,
) -> pd.DataFrame:
    """Convert Elo expected points into valid H/D/A probabilities.

    The construction preserves ``P(H) + 0.5 * P(D) == expected_home_score``.
    ``draw_at_even`` is therefore the draw probability when both teams are even.
    """
    _validate_probability_parameter(draw_at_even, "draw_at_even", maximum=0.5)
    if not isfinite(float(draw_shape)) or float(draw_shape) < 1.0:
        raise ValueError("draw_shape must be finite and >= 1")
    values, index = _one_dimensional_numeric(expected_home_score, "expected_home_score")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("expected_home_score must be within [0, 1]")

    draw = float(draw_at_even) * (4.0 * values * (1.0 - values)) ** float(
        draw_shape
    )
    home = values - 0.5 * draw
    away = 1.0 - values - 0.5 * draw
    probabilities = pd.DataFrame(
        {
            "home_probability": home,
            "draw_probability": draw,
            "away_probability": away,
        },
        index=index,
    )
    _validate_probability_frame(probabilities)
    return probabilities


def standard_1x2_losses(
    probabilities: pd.DataFrame,
    home_goals: Sequence[float] | pd.Series,
    away_goals: Sequence[float] | pd.Series,
    *,
    epsilon: float = 1e-15,
) -> pd.DataFrame:
    """Return standard three-class Brier and logarithmic loss per match."""
    _validate_probability_parameter(epsilon, "epsilon", maximum=0.5, allow_zero=False)
    _require_columns(probabilities, set(PROBABILITY_COLUMNS), "probabilities")
    probability_values = probabilities.loc[:, list(PROBABILITY_COLUMNS)].copy()
    _validate_probability_frame(probability_values)
    home = _validated_scores(home_goals, "home_goals", len(probability_values))
    away = _validated_scores(away_goals, "away_goals", len(probability_values))

    outcome_index = np.where(home > away, 0, np.where(home == away, 1, 2))
    one_hot = np.eye(3, dtype=float)[outcome_index]
    matrix = probability_values.to_numpy(dtype=float)
    observed_probability = matrix[np.arange(len(matrix)), outcome_index]
    brier = ((matrix - one_hot) ** 2).sum(axis=1)
    log_loss = -np.log(np.clip(observed_probability, epsilon, 1.0))
    outcome = np.asarray(("H", "D", "A"), dtype=object)[outcome_index]
    return pd.DataFrame(
        {
            "outcome_1x2": outcome,
            "brier_1x2": brier,
            "log_loss_1x2": log_loss,
        },
        index=probability_values.index,
    )


def evaluate_1x2_predictions(
    expected_home_score: Iterable[float] | float,
    home_goals: Sequence[float] | pd.Series,
    away_goals: Sequence[float] | pd.Series,
    *,
    draw_at_even: float,
    draw_shape: float,
) -> pd.DataFrame:
    """Create score-preserving probabilities and standard 1X2 losses."""
    probabilities = score_preserving_1x2_probabilities(
        expected_home_score,
        draw_at_even=draw_at_even,
        draw_shape=draw_shape,
    )
    losses = standard_1x2_losses(probabilities, home_goals, away_goals)
    return probabilities.join(losses)


def dependency_robust_loss_difference_ci(
    predictions: pd.DataFrame,
    *,
    difference_column: str = "loss_difference",
    tie_column: str = "tie_id",
    bootstrap_samples: int = 4000,
    seed: int = 20260720,
) -> pd.DataFrame:
    """Bootstrap paired loss differences under three dependency structures.

    The three rows are sensitivity analyses rather than a claim of formal
    multi-way clustering. The final envelope uses the widest lower and upper
    bounds, so a promotion is called reliable only when every view agrees.
    """
    required = set(UNCERTAINTY_COLUMNS) | {difference_column}
    _require_columns(predictions, required, "predictions")
    if not isinstance(bootstrap_samples, int) or bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be an integer >= 100")
    values = predictions.copy()
    difference = pd.to_numeric(values[difference_column], errors="coerce")
    if difference.isna().any() or not np.isfinite(difference).all():
        raise ValueError(f"predictions.{difference_column} must be finite numeric")
    values[difference_column] = difference.astype(float)
    if values["match_id"].isna().any() or values["match_id"].duplicated().any():
        raise ValueError("predictions.match_id must be non-null and unique")
    if values[["home_team_id", "away_team_id", "season"]].isna().any().any():
        raise ValueError("predictions team and season identifiers cannot be missing")
    if values["home_team_id"].eq(values["away_team_id"]).any():
        raise ValueError("predictions cannot contain the same home and away team")
    kickoff = pd.to_datetime(values["kickoff_utc"], utc=True, errors="coerce")
    if kickoff.isna().any():
        raise ValueError("predictions.kickoff_utc must contain valid UTC timestamps")

    if tie_column in values.columns:
        tie = values[tie_column].where(values[tie_column].notna(), values["match_id"])
    else:
        tie = values["match_id"]
    tie_clusters = values["season"].astype(str) + "|" + tie.astype(str)
    month_clusters = (
        values["season"].astype(str)
        + "|"
        + kickoff.dt.strftime("%Y-%m")
    )
    team_values = pd.concat(
        [
            pd.DataFrame(
                {
                    "cluster": values["season"].astype(str)
                    + "|"
                    + values["home_team_id"].astype(str),
                    "difference": difference,
                }
            ),
            pd.DataFrame(
                {
                    "cluster": values["season"].astype(str)
                    + "|"
                    + values["away_team_id"].astype(str),
                    "difference": difference,
                }
            ),
        ],
        ignore_index=True,
    )
    cluster_views = (
        (
            "tie_or_match",
            pd.DataFrame({"cluster": tie_clusters, "difference": difference}),
        ),
        ("team_season", team_values),
        (
            "calendar_month",
            pd.DataFrame({"cluster": month_clusters, "difference": difference}),
        ),
    )

    rows: list[dict[str, object]] = []
    for offset, (method, clustered) in enumerate(cluster_views):
        aggregates = clustered.groupby("cluster", sort=False)["difference"].agg(
            ["sum", "count"]
        )
        means = _cluster_bootstrap_means(
            aggregates,
            bootstrap_samples,
            seed + offset * 1009,
        )
        lower, upper = np.quantile(means, (0.025, 0.975))
        rows.append(
            {
                "method": method,
                "matches": int(len(values)),
                "clusters": int(len(aggregates)),
                "mean_difference": float(difference.mean()),
                "ci_95_lower": float(lower),
                "ci_95_upper": float(upper),
                "reliable_improvement": bool(upper < 0.0),
                "reliable_harm": bool(lower > 0.0),
            }
        )

    lower = min(float(row["ci_95_lower"]) for row in rows)
    upper = max(float(row["ci_95_upper"]) for row in rows)
    rows.append(
        {
            "method": "conservative_envelope",
            "matches": int(len(values)),
            "clusters": pd.NA,
            "mean_difference": float(difference.mean()),
            "ci_95_lower": lower,
            "ci_95_upper": upper,
            "reliable_improvement": bool(upper < 0.0),
            "reliable_harm": bool(lower > 0.0),
        }
    )
    return pd.DataFrame(rows)


def _validated_match_frame(
    matches: pd.DataFrame,
    required: set[str],
) -> pd.DataFrame:
    _require_columns(matches, required, "matches")
    data = matches.copy()
    identifiers = [
        "season",
        "competition",
        "home_team_id",
        "away_team_id",
    ]
    if data[identifiers].isna().any().any():
        raise ValueError("matches season, competition, and team identifiers cannot be missing")
    if data["home_team_id"].eq(data["away_team_id"]).any():
        raise ValueError("matches cannot contain the same home and away team")
    data["home_goals"] = _validated_scores(
        data["home_goals"], "home_goals", len(data)
    )
    data["away_goals"] = _validated_scores(
        data["away_goals"], "away_goals", len(data)
    )
    if "is_neutral" in data.columns:
        data["is_neutral"] = _coerce_boolean(data["is_neutral"], "is_neutral")
    else:
        data["is_neutral"] = False
    data["actual_home_score"] = np.where(
        data["home_goals"] > data["away_goals"],
        1.0,
        np.where(data["home_goals"] < data["away_goals"], 0.0, 0.5),
    )
    return data


def _long_team_results(segment: pd.DataFrame) -> pd.DataFrame:
    home = pd.DataFrame(
        {
            "team_id": segment["home_team_id"],
            "opponent_id": segment["away_team_id"],
            "score": segment["actual_home_score"],
            "venue_sign": np.where(segment["is_neutral"], 0.0, 1.0),
        }
    )
    away = pd.DataFrame(
        {
            "team_id": segment["away_team_id"],
            "opponent_id": segment["home_team_id"],
            "score": 1.0 - segment["actual_home_score"],
            "venue_sign": np.where(segment["is_neutral"], 0.0, -1.0),
        }
    )
    return pd.concat([home, away], ignore_index=True)


def _cluster_bootstrap_means(
    aggregates: pd.DataFrame,
    bootstrap_samples: int,
    seed: int,
) -> np.ndarray:
    if aggregates.empty:
        raise ValueError("At least one non-empty cluster is required")
    sums = aggregates["sum"].to_numpy(dtype=float)
    counts = aggregates["count"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    result = np.empty(bootstrap_samples, dtype=float)
    batch_size = min(256, bootstrap_samples)
    for start in range(0, bootstrap_samples, batch_size):
        stop = min(start + batch_size, bootstrap_samples)
        indices = rng.integers(0, len(aggregates), size=(stop - start, len(aggregates)))
        result[start:stop] = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    return result


def _one_dimensional_numeric(
    values: Iterable[float] | float,
    label: str,
) -> tuple[np.ndarray, pd.Index]:
    if np.isscalar(values):
        series = pd.Series([values])
    elif isinstance(values, pd.Series):
        series = values
    else:
        series = pd.Series(list(values))
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric).all():
        raise ValueError(f"{label} must contain finite numeric values")
    return numeric.to_numpy(dtype=float), series.index


def _validated_scores(
    values: Sequence[float] | pd.Series,
    label: str,
    expected_length: int,
) -> np.ndarray:
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    if len(series) != expected_length:
        raise ValueError(f"{label} length must match probabilities")
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric).all() or (numeric < 0).any():
        raise ValueError(f"{label} must contain finite non-negative values")
    if not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{label} must contain whole-number scores")
    return numeric.to_numpy(dtype=float)


def _coerce_boolean(values: pd.Series, label: str) -> pd.Series:
    mapping = {
        True: True,
        False: False,
        1: True,
        0: False,
        "1": True,
        "0": False,
        "true": True,
        "false": False,
    }
    normalized = values.map(
        lambda value: value.strip().lower() if isinstance(value, str) else value
    )
    converted = normalized.map(mapping)
    if converted.isna().any():
        bad = values.loc[converted.isna()].head(3).tolist()
        raise ValueError(f"matches.{label} contains invalid boolean values: {bad}")
    return converted.astype(bool)


def _validate_probability_frame(probabilities: pd.DataFrame) -> None:
    values = probabilities.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("probabilities must be finite")
    if (values < -1e-12).any() or (values > 1.0 + 1e-12).any():
        raise ValueError("probabilities must be within [0, 1]")
    totals = values.sum(axis=1)
    if not np.allclose(totals, 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("home, draw, and away probabilities must sum to 1")


def _validate_probability_parameter(
    value: float,
    label: str,
    *,
    maximum: float,
    allow_zero: bool = True,
) -> None:
    numeric = float(value)
    minimum_ok = numeric >= 0.0 if allow_zero else numeric > 0.0
    if not isfinite(numeric) or not minimum_ok or numeric > maximum:
        interval = "[0" if allow_zero else "(0"
        raise ValueError(f"{label} must be finite and within {interval}, {maximum}]")


def _validate_non_negative_finite(value: float, label: str) -> None:
    if not isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{label} must be finite and >= 0")


def _reliability(count: int, prior_matches: float) -> float:
    denominator = float(count) + float(prior_matches)
    return float(count) / denominator if denominator else 0.0


def _require_columns(data: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {', '.join(missing)}")
