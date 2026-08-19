from __future__ import annotations

"""External benchmark for the CURRENT production model.

Two independent external axes, both temporally clean:

1. `PREDICTION`  Served 1X2 quality against ClubElo on the paired exact-date
   sample. Standard three-class Brier and log-loss, dependency-robust intervals.
2. `RATING`      AO First Elo against the strictly pre-season Opta Power
   Rankings snapshot, where both are scored against the realized European
   season instead of against each other. The same axis carries a value-added
   floor: the UEFA club coefficient that AO First Elo is itself built from.

The UEFA coefficient arm is deliberately not called external. `club_points_t_*`
in `club_european_points.csv` are the five seasonal components of that
coefficient, so the comparison asks a different question: does the static
pipeline earn its complexity over the raw input it consumes? A model that only
matches its own input has no reason to exist. The published 2026 coefficient is
not used because its five-season window already contains the season being
predicted; the leakage-free pre-season value is the one the model was fed.

How this differs from the two existing external scripts:

- `run_external_elo_benchmark.py` scores a historical v1-scale candidate
  (`scale=225`, `H=40`, `K=28`, `carry=0.85`) with legacy expected-score Brier.
  This script scores the *current* contract and the served ML/Poisson ensemble
  with the standard three-class losses.
- `run_initial_elo_external_comparison_2025_26.py` measures *agreement* between
  AO First Elo and Opta and explicitly excludes match results. This script adds
  the missing question: which of the two better predicts what actually happened.

The script is read-only with respect to the model. It scores frozen prediction
and rating artifacts; it never fits or changes a production parameter.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "src"))

from ao_elo.draw_probability import score_preserving_1x2_scalar  # noqa: E402
from ao_elo.evaluation import (  # noqa: E402
    dependency_robust_loss_difference_ci,
    schedule_adjusted_team_performance,
)
from scripts.run_external_ranking_comparison_2025_26 import pairwise_accuracy  # noqa: E402


DEFAULT_EXTERNAL_ELO = (
    ROOT / "data" / "external_elo_benchmark_2018_2026"
    / "matches_with_dates_and_external_elo.csv"
)
DEFAULT_ENSEMBLE_PREDICTIONS = (
    ROOT / "output" / "final_prediction_ensemble_backtest_2018_2026"
    / "unseen_predictions.csv"
)
DEFAULT_OPTA_SNAPSHOT = (
    ROOT / "data" / "external_rankings_2025_26"
    / "opta_power_rankings_2025_07_03_ao_scope.csv"
)
DEFAULT_TEAM_IDENTITY = ROOT / "data" / "club_identity" / "team_season_identity.csv"
DEFAULT_CLUB_COEFFICIENTS = (
    ROOT / "data" / "backtest_stage_b_2018_2026" / "2025-26" / "club_european_points.csv"
)
DEFAULT_MODEL_RATINGS = (
    ROOT / "output" / "current_model_evaluation_2018_2026" / "model_end_ratings.csv"
)
DEFAULT_MATCHES = ROOT / "data" / "dynamic_backtest_2018_2026" / "matches.csv"
DEFAULT_OUTPUT = ROOT / "output" / "current_external_benchmark"

EVALUATION_SEASONS = (
    "2020/21",
    "2021/22",
    "2022/23",
    "2023/24",
    "2024/25",
    "2025/26",
)
RATING_SEASON = "2025/26"
MAX_SNAPSHOT_AGE_DAYS = 31
CLUBELO_PUBLISHED_SCALE = 400.0
DRAW_AT_EVEN = 0.24
SINGLE_MATCH_DRAW_AT_EVEN = 0.12
DRAW_SHAPE = 1.0
SERVED_POISSON_WEIGHT = 0.50
BOOTSTRAP_SAMPLES = 4000
BOOTSTRAP_SEED = 20260813

AO_ARMS = ("AO_RATING_CORE_1X2", "AO_SERVED_ENSEMBLE_50_50")
CLUBELO_ARMS = ("CLUBELO_PUBLISHED_SCALE_400", "CLUBELO_TUNED_SCALE_AND_H")


@dataclass(frozen=True)
class ClubEloFit:
    """Training-only ClubElo calibration for one outer test season."""

    season: str
    train_matches: int
    published_home_advantage: float
    tuned_scale: float
    tuned_home_advantage: float


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score the current AO contract against ClubElo and pre-season Opta "
            "on two independent external axes"
        )
    )
    parser.add_argument("--external-elo", type=Path, default=DEFAULT_EXTERNAL_ELO)
    parser.add_argument(
        "--ensemble-predictions", type=Path, default=DEFAULT_ENSEMBLE_PREDICTIONS
    )
    parser.add_argument("--opta-snapshot", type=Path, default=DEFAULT_OPTA_SNAPSHOT)
    parser.add_argument("--team-identity", type=Path, default=DEFAULT_TEAM_IDENTITY)
    parser.add_argument(
        "--club-coefficients", type=Path, default=DEFAULT_CLUB_COEFFICIENTS
    )
    parser.add_argument("--model-ratings", type=Path, default=DEFAULT_MODEL_RATINGS)
    parser.add_argument("--matches", type=Path, default=DEFAULT_MATCHES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-snapshot-age-days", type=int, default=MAX_SNAPSHOT_AGE_DAYS
    )
    arguments = parser.parse_args()

    output_root = arguments.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    prediction = run_prediction_axis(
        external_elo_path=arguments.external_elo,
        ensemble_predictions_path=arguments.ensemble_predictions,
        max_snapshot_age_days=int(arguments.max_snapshot_age_days),
    )
    rating = run_rating_axis(
        opta_snapshot_path=arguments.opta_snapshot,
        team_identity_path=arguments.team_identity,
        model_ratings_path=arguments.model_ratings,
        matches_path=arguments.matches,
        club_coefficients_path=arguments.club_coefficients,
    )

    prediction["model_summary"].to_csv(
        output_root / "prediction_model_summary.csv", index=False
    )
    prediction["competition_summary"].to_csv(
        output_root / "prediction_competition_summary.csv", index=False
    )
    prediction["uncertainty"].to_csv(
        output_root / "prediction_uncertainty.csv", index=False
    )
    prediction["clubelo_fits"].to_csv(
        output_root / "prediction_clubelo_fits.csv", index=False
    )
    prediction["coverage"].to_csv(output_root / "prediction_coverage.csv", index=False)
    rating["summary"].to_csv(output_root / "rating_model_summary.csv", index=False)
    rating["comparisons"].to_csv(output_root / "rating_comparisons.csv", index=False)
    rating["team_table"].to_csv(output_root / "rating_team_table.csv", index=False)

    manifest = build_manifest(prediction, rating, arguments)
    (output_root / "benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(output_root / "benchmark_report.md", prediction, rating, manifest)

    print(f"Wrote external benchmark to {output_root}")
    print(prediction["model_summary"].to_string(index=False))
    print()
    print(rating["summary"].to_string(index=False))
    print()
    print(rating["comparisons"].to_string(index=False))


# ---------------------------------------------------------------------------
# shared scoring helpers
# ---------------------------------------------------------------------------


def one_x_two_from_expected(
    expected_home_score: np.ndarray,
    single_match_tie: np.ndarray,
) -> np.ndarray:
    """Apply the active score-preserving draw model to any expected-score vector.

    Both AO and the external rating receive the identical draw treatment, so the
    comparison isolates rating quality rather than draw-model quality.
    """
    rows = [
        score_preserving_1x2_scalar(
            float(value),
            SINGLE_MATCH_DRAW_AT_EVEN if bool(is_single) else DRAW_AT_EVEN,
            DRAW_SHAPE,
        )
        for value, is_single in zip(expected_home_score, single_match_tie)
    ]
    return np.asarray(rows, dtype=float)


def elo_expected_home_score(
    rating_difference: np.ndarray,
    is_neutral: np.ndarray,
    home_advantage: float,
    scale: float,
) -> np.ndarray:
    effective = np.asarray(rating_difference, dtype=float) + np.where(
        np.asarray(is_neutral, dtype=bool), 0.0, float(home_advantage)
    )
    return 1.0 / (1.0 + 10.0 ** (-effective / float(scale)))


def vectorized_elo_expected_home_score(
    rating_difference: np.ndarray,
    is_neutral: np.ndarray,
    home_advantage: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """Row-wise expected score when home advantage and scale vary per season."""
    effective = np.asarray(rating_difference, dtype=float) + np.where(
        np.asarray(is_neutral, dtype=bool), 0.0, np.asarray(home_advantage, dtype=float)
    )
    return 1.0 / (1.0 + 10.0 ** (-effective / np.asarray(scale, dtype=float)))


def three_class_losses(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    one_hot = np.eye(3)[np.asarray(outcomes, dtype=int)]
    observed = probabilities[np.arange(len(outcomes)), np.asarray(outcomes, dtype=int)]
    brier = ((probabilities - one_hot) ** 2).sum(axis=1)
    log_loss = -np.log(np.clip(observed, 1e-15, 1.0))
    return brier, log_loss


def summarize_arm(name: str, probabilities: np.ndarray, outcomes: np.ndarray) -> dict:
    brier, log_loss = three_class_losses(probabilities, outcomes)
    return {
        "model_arm": name,
        "matches": int(len(outcomes)),
        "brier_1x2": float(brier.mean()),
        "log_loss_1x2": float(log_loss.mean()),
        "accuracy_1x2": float((probabilities.argmax(axis=1) == outcomes).mean()),
    }


def log_probability_blend(
    left: np.ndarray,
    right: np.ndarray,
    right_weight: float,
) -> np.ndarray:
    """Reproduce the served log-probability blend used by production."""
    logits = (1.0 - float(right_weight)) * np.log(np.maximum(left, 1e-15))
    logits += float(right_weight) * np.log(np.maximum(right, 1e-15))
    logits -= logits.max(axis=1, keepdims=True)
    result = np.exp(logits)
    return result / result.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Axis 1: prediction quality against ClubElo
# ---------------------------------------------------------------------------


def load_paired_sample(
    external_elo_path: Path,
    *,
    max_snapshot_age_days: int,
) -> pd.DataFrame:
    external = pd.read_csv(external_elo_path)
    both_rated = external["clubelo_home_elo"].notna() & external["clubelo_away_elo"].notna()
    fresh = external["clubelo_home_snapshot_age_days"].le(max_snapshot_age_days) & external[
        "clubelo_away_snapshot_age_days"
    ].le(max_snapshot_age_days)
    sample = external.loc[both_rated & fresh].copy()
    if sample.empty:
        raise ValueError("No paired ClubElo matches survived the snapshot age filter")

    tie_size = sample.groupby(["season", "tie_id"])["match_id"].transform("size")
    sample["single_match_tie"] = (
        sample["is_knockout"].fillna(False).astype(bool) & tie_size.eq(1)
    )
    sample["is_neutral"] = sample["is_neutral"].fillna(False).astype(bool)
    sample["actual_class"] = np.where(
        sample["home_goals"] > sample["away_goals"],
        0,
        np.where(sample["home_goals"] == sample["away_goals"], 1, 2),
    )
    sample["clubelo_rating_difference"] = (
        sample["clubelo_home_elo"] - sample["clubelo_away_elo"]
    )
    return sample.reset_index(drop=True)


def attach_model_arms(
    sample: pd.DataFrame,
    ensemble_predictions_path: Path,
) -> pd.DataFrame:
    raw = pd.read_csv(ensemble_predictions_path)
    columns = ["home_probability", "draw_probability", "away_probability"]
    wide = raw.pivot(index="match_id", columns="model", values=columns)
    wide.columns = [f"{model}__{field}" for field, model in wide.columns]
    joined = sample.merge(wide.reset_index(), on="match_id", how="inner")
    if joined.empty:
        raise ValueError("Paired ClubElo sample did not join to any model prediction")
    return joined


def fit_clubelo_published_scale(train: pd.DataFrame) -> float:
    """Fit only ClubElo's home advantage, keeping its published 400 scale."""
    if train.empty:
        return 0.0
    difference = train["clubelo_rating_difference"].to_numpy(float)
    neutral = train["is_neutral"].to_numpy(bool)
    single = train["single_match_tie"].to_numpy(bool)
    outcomes = train["actual_class"].to_numpy(int)

    def objective(home_advantage: float) -> float:
        expected = elo_expected_home_score(
            difference, neutral, float(home_advantage), CLUBELO_PUBLISHED_SCALE
        )
        return float(
            three_class_losses(one_x_two_from_expected(expected, single), outcomes)[1].mean()
        )

    return float(minimize_scalar(objective, bounds=(-20.0, 200.0), method="bounded").x)


def fit_clubelo_tuned(train: pd.DataFrame) -> tuple[float, float]:
    """Fit ClubElo's scale and home advantage together, a deliberately generous arm."""
    if train.empty:
        return CLUBELO_PUBLISHED_SCALE, 0.0
    difference = train["clubelo_rating_difference"].to_numpy(float)
    neutral = train["is_neutral"].to_numpy(bool)
    single = train["single_match_tie"].to_numpy(bool)
    outcomes = train["actual_class"].to_numpy(int)

    def objective(parameters: np.ndarray) -> float:
        scale, home_advantage = float(parameters[0]), float(parameters[1])
        if scale <= 50.0:
            return 1e9
        expected = elo_expected_home_score(difference, neutral, home_advantage, scale)
        return float(
            three_class_losses(one_x_two_from_expected(expected, single), outcomes)[1].mean()
        )

    result = minimize(
        objective,
        x0=np.array([CLUBELO_PUBLISHED_SCALE, 60.0]),
        method="Nelder-Mead",
        options={"maxiter": 2000, "fatol": 1e-10},
    )
    return float(result.x[0]), float(result.x[1])


def walk_forward_clubelo_fits(joined: pd.DataFrame) -> tuple[pd.DataFrame, list[ClubEloFit]]:
    """Calibrate ClubElo on earlier seasons only, then score each test season."""
    frames: list[pd.DataFrame] = []
    fits: list[ClubEloFit] = []
    for season in EVALUATION_SEASONS:
        test = joined.loc[joined["season"].eq(season)].copy()
        if test.empty:
            continue
        train = joined.loc[joined["season"].lt(season)]
        published_home = fit_clubelo_published_scale(train)
        tuned_scale, tuned_home = fit_clubelo_tuned(train)
        fits.append(
            ClubEloFit(
                season=season,
                train_matches=int(len(train)),
                published_home_advantage=published_home,
                tuned_scale=tuned_scale,
                tuned_home_advantage=tuned_home,
            )
        )
        test["clubelo_published_home_advantage"] = published_home
        test["clubelo_tuned_scale"] = tuned_scale
        test["clubelo_tuned_home_advantage"] = tuned_home
        if len(train):
            base_rate = (
                np.bincount(train["actual_class"].to_numpy(int), minlength=3) / len(train)
            )
        else:
            base_rate = np.full(3, 1.0 / 3.0)
        test[["climatology_home", "climatology_draw", "climatology_away"]] = np.tile(
            base_rate, (len(test), 1)
        )
        frames.append(test)
    if not frames:
        raise ValueError("No evaluation season produced a paired ClubElo sample")
    return pd.concat(frames, ignore_index=True), fits


def build_prediction_arms(data: pd.DataFrame) -> dict[str, np.ndarray]:
    difference = data["clubelo_rating_difference"].to_numpy(float)
    neutral = data["is_neutral"].to_numpy(bool)
    single = data["single_match_tie"].to_numpy(bool)

    published_home = data["clubelo_published_home_advantage"].to_numpy(float)
    tuned_home = data["clubelo_tuned_home_advantage"].to_numpy(float)
    tuned_scale = data["clubelo_tuned_scale"].to_numpy(float)

    published = one_x_two_from_expected(
        vectorized_elo_expected_home_score(
            difference,
            neutral,
            published_home,
            np.full(len(data), CLUBELO_PUBLISHED_SCALE),
        ),
        single,
    )
    tuned = one_x_two_from_expected(
        vectorized_elo_expected_home_score(
            difference,
            neutral,
            tuned_home,
            tuned_scale,
        ),
        single,
    )

    def arm(name: str) -> np.ndarray:
        return data[
            [
                f"{name}__home_probability",
                f"{name}__draw_probability",
                f"{name}__away_probability",
            ]
        ].to_numpy(float)

    return {
        "CLIMATOLOGY_WALK_FORWARD": data[
            ["climatology_home", "climatology_draw", "climatology_away"]
        ].to_numpy(float),
        "CLUBELO_PUBLISHED_SCALE_400": published,
        "CLUBELO_TUNED_SCALE_AND_H": tuned,
        "AO_RATING_CORE_1X2": arm("CURRENT_AO"),
        "AO_SERVED_ENSEMBLE_50_50": log_probability_blend(
            arm("CURRENT_ML_BLEND"),
            arm("AO_POISSON_RHO0_CONTROL"),
            SERVED_POISSON_WEIGHT,
        ),
    }


def prediction_uncertainty(
    data: pd.DataFrame,
    arms: dict[str, np.ndarray],
    outcomes: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for ao_name in AO_ARMS:
        for external_name in CLUBELO_ARMS:
            for metric, index in (("brier_1x2", 0), ("log_loss_1x2", 1)):
                ao_loss = three_class_losses(arms[ao_name], outcomes)[index]
                external_loss = three_class_losses(arms[external_name], outcomes)[index]
                frame = pd.DataFrame(
                    {
                        "season": data["season"].to_numpy(),
                        "match_id": data["match_id"].to_numpy(),
                        "home_team_id": data["home_team_id"].to_numpy(),
                        "away_team_id": data["away_team_id"].to_numpy(),
                        "kickoff_utc": data["kickoff_utc"].to_numpy(),
                        "tie_id": data["tie_id"].to_numpy(),
                        "loss_difference": ao_loss - external_loss,
                    }
                )
                envelope = dependency_robust_loss_difference_ci(frame)
                row = envelope.loc[envelope["method"].eq("conservative_envelope")].iloc[0]
                rows.append(
                    {
                        "ao_arm": ao_name,
                        "external_arm": external_name,
                        "metric": metric,
                        "matches": int(len(data)),
                        "mean_ao_minus_external": float(row["mean_difference"]),
                        "ci_95_lower": float(row["ci_95_lower"]),
                        "ci_95_upper": float(row["ci_95_upper"]),
                        "ao_reliably_better": bool(row["reliable_improvement"]),
                        "external_reliably_better": bool(row["reliable_harm"]),
                    }
                )
    return pd.DataFrame(rows)


def run_prediction_axis(
    *,
    external_elo_path: Path,
    ensemble_predictions_path: Path,
    max_snapshot_age_days: int,
) -> dict[str, object]:
    sample = load_paired_sample(
        external_elo_path, max_snapshot_age_days=max_snapshot_age_days
    )
    joined = attach_model_arms(sample, ensemble_predictions_path)
    data, fits = walk_forward_clubelo_fits(joined)
    outcomes = data["actual_class"].to_numpy(int)
    arms = build_prediction_arms(data)

    model_summary = pd.DataFrame(
        [summarize_arm(name, probabilities, outcomes) for name, probabilities in arms.items()]
    )
    climatology_brier = float(
        model_summary.loc[
            model_summary["model_arm"].eq("CLIMATOLOGY_WALK_FORWARD"), "brier_1x2"
        ].iloc[0]
    )
    climatology_log_loss = float(
        model_summary.loc[
            model_summary["model_arm"].eq("CLIMATOLOGY_WALK_FORWARD"), "log_loss_1x2"
        ].iloc[0]
    )
    model_summary["brier_skill_vs_climatology"] = (
        model_summary["brier_1x2"] - climatology_brier
    ) / climatology_brier
    model_summary["log_loss_skill_vs_climatology"] = (
        model_summary["log_loss_1x2"] - climatology_log_loss
    ) / climatology_log_loss

    competition_rows = []
    for name, probabilities in arms.items():
        for competition, index in data.groupby("competition").groups.items():
            position = data.index.get_indexer(index)
            competition_rows.append(
                {"competition": competition}
                | summarize_arm(name, probabilities[position], outcomes[position])
            )
    competition_summary = pd.DataFrame(competition_rows).sort_values(
        ["competition", "model_arm"], kind="stable"
    )

    coverage = (
        pd.read_csv(external_elo_path)
        .assign(
            paired=lambda frame: frame["clubelo_home_elo"].notna()
            & frame["clubelo_away_elo"].notna()
        )
        .groupby("season", as_index=False)
        .agg(source_matches=("match_id", "size"), clubelo_paired=("paired", "sum"))
    )
    scored = data.groupby("season", as_index=False).agg(scored_matches=("match_id", "size"))
    coverage = coverage.merge(scored, on="season", how="left").fillna({"scored_matches": 0})
    coverage["scored_matches"] = coverage["scored_matches"].astype(int)

    return {
        "data": data,
        "arms": arms,
        "model_summary": model_summary,
        "competition_summary": competition_summary,
        "uncertainty": prediction_uncertainty(data, arms, outcomes),
        "clubelo_fits": pd.DataFrame([fit.__dict__ for fit in fits]),
        "coverage": coverage,
        "scored_seasons": sorted(data["season"].unique().tolist()),
    }


# ---------------------------------------------------------------------------
# Axis 2: rating quality against the pre-season Opta snapshot
# ---------------------------------------------------------------------------


def realized_season_performance(matches_path: Path, season: str) -> pd.DataFrame:
    """Build the AO-independent, venue- and schedule-adjusted season target."""
    matches = pd.read_csv(matches_path)
    matches = matches.loc[matches["season"].eq(season)].copy()
    if matches.empty:
        raise ValueError(f"No matches found for rating season {season}")
    matches["home_team_id"] = matches["home_team_id"].astype(str)
    matches["away_team_id"] = matches["away_team_id"].astype(str)
    target = schedule_adjusted_team_performance(matches)
    aggregated = (
        target.assign(weighted=target["schedule_adjusted_score"] * target["matches"])
        .groupby("team_id", as_index=False)
        .agg(weighted=("weighted", "sum"), matches=("matches", "sum"))
    )
    aggregated["schedule_adjusted_score"] = aggregated["weighted"] / aggregated["matches"]
    return aggregated[["team_id", "matches", "schedule_adjusted_score"]]


def build_rating_table(
    *,
    opta_snapshot_path: Path,
    team_identity_path: Path,
    model_ratings_path: Path,
    matches_path: Path,
    club_coefficients_path: Path,
    season: str = RATING_SEASON,
) -> pd.DataFrame:
    opta = pd.read_csv(opta_snapshot_path)
    identity = pd.read_csv(team_identity_path)
    identity = identity.loc[identity["season"].eq(season)].copy()
    identity["local_team_id"] = identity["local_team_id"].astype(str)

    ratings = pd.read_csv(model_ratings_path)
    ratings = ratings.loc[
        ratings["model"].eq("CURRENT_PRODUCTION") & ratings["season"].eq(season)
    ].copy()
    ratings["team_id"] = ratings["team_id"].astype(str)

    coefficients = pd.read_csv(club_coefficients_path)
    coefficients = coefficients.loc[coefficients["season"].eq(season)].copy()
    coefficients["team_id"] = coefficients["team_id"].astype(str)

    target = realized_season_performance(matches_path, season)
    table = (
        target.merge(
            identity[["local_team_id", "club_id"]],
            left_on="team_id",
            right_on="local_team_id",
            how="inner",
        )
        .merge(ratings[["team_id", "initial_rating"]], on="team_id", how="inner")
        .merge(opta[["club_id", "opta_power_rating"]], on="club_id", how="inner")
        .merge(
            coefficients[["team_id", "official_club_coefficient"]],
            on="team_id",
            how="inner",
        )
    )
    if table.empty:
        raise ValueError("Rating axis produced no joined teams")
    return table.drop(columns=["local_team_id"]).reset_index(drop=True)


def paired_spearman_difference_ci(
    model_values: np.ndarray,
    benchmark_values: np.ndarray,
    realized: np.ndarray,
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Bootstrap the paired difference between two ratings' predictive Spearman."""
    generator = np.random.default_rng(seed)
    size = len(realized)
    differences = []
    for _ in range(samples):
        index = generator.integers(0, size, size)
        if len(np.unique(realized[index])) < 3:
            continue
        model_correlation = spearmanr(model_values[index], realized[index]).statistic
        benchmark_correlation = spearmanr(
            benchmark_values[index], realized[index]
        ).statistic
        if np.isnan(model_correlation) or np.isnan(benchmark_correlation):
            continue
        differences.append(model_correlation - benchmark_correlation)
    if not differences:
        raise ValueError("Bootstrap produced no valid resamples")
    values = np.asarray(differences, dtype=float)
    lower, upper = np.quantile(values, (0.025, 0.975))
    return float(values.mean()), float(lower), float(upper)


RATING_ARMS = (
    ("AO_FIRST_ELO", "initial_rating", "MODEL"),
    ("OPTA_PRE_SEASON_POWER_RANKING", "opta_power_rating", "EXTERNAL"),
    ("UEFA_CLUB_COEFFICIENT_PRE_SEASON", "official_club_coefficient", "OWN_INPUT"),
)


def run_rating_axis(
    *,
    opta_snapshot_path: Path,
    team_identity_path: Path,
    model_ratings_path: Path,
    matches_path: Path,
    club_coefficients_path: Path,
) -> dict[str, object]:
    table = build_rating_table(
        opta_snapshot_path=opta_snapshot_path,
        team_identity_path=team_identity_path,
        model_ratings_path=model_ratings_path,
        matches_path=matches_path,
        club_coefficients_path=club_coefficients_path,
    )
    realized = table["schedule_adjusted_score"].to_numpy(float)
    rows = []
    for label, column, reference_type in RATING_ARMS:
        values = table[column].to_numpy(float)
        rows.append(
            {
                "rating": label,
                "reference_type": reference_type,
                "teams": int(len(table)),
                "spearman_vs_realized": float(spearmanr(values, realized).statistic),
                "pearson_vs_realized": float(np.corrcoef(values, realized)[0, 1]),
                "pairwise_accuracy_vs_realized": pairwise_accuracy(values, realized),
            }
        )
    summary = pd.DataFrame(rows)

    ao_values = table["initial_rating"].to_numpy(float)
    comparison_rows = []
    for label, column, reference_type in RATING_ARMS[1:]:
        benchmark_values = table[column].to_numpy(float)
        mean_difference, lower, upper = paired_spearman_difference_ci(
            ao_values, benchmark_values, realized
        )
        comparison_rows.append(
            {
                "comparison": f"AO_FIRST_ELO_minus_{label}",
                "reference_type": reference_type,
                "teams": int(len(table)),
                "spearman_difference": mean_difference,
                "ci_95_lower": lower,
                "ci_95_upper": upper,
                "ao_reliably_better": bool(lower > 0.0),
                "benchmark_reliably_better": bool(upper < 0.0),
                "rank_agreement_spearman": float(
                    spearmanr(ao_values, benchmark_values).statistic
                ),
            }
        )
    return {
        "summary": summary,
        "comparisons": pd.DataFrame(comparison_rows),
        "team_table": table,
    }


# ---------------------------------------------------------------------------
# manifest and report
# ---------------------------------------------------------------------------


def build_manifest(
    prediction: dict[str, object],
    rating: dict[str, object],
    arguments: argparse.Namespace,
) -> dict[str, object]:
    summary = prediction["model_summary"].set_index("model_arm")
    return {
        "benchmark_version": "ao-current-external-benchmark-v1",
        "prediction_axis": {
            "paired_matches": int(len(prediction["data"])),
            "scored_seasons": prediction["scored_seasons"],
            "max_snapshot_age_days": int(arguments.max_snapshot_age_days),
            "clubelo_published_brier": float(
                summary.loc["CLUBELO_PUBLISHED_SCALE_400", "brier_1x2"]
            ),
            "ao_rating_core_brier": float(summary.loc["AO_RATING_CORE_1X2", "brier_1x2"]),
            "ao_served_ensemble_brier": float(
                summary.loc["AO_SERVED_ENSEMBLE_50_50", "brier_1x2"]
            ),
            "ao_served_brier_skill_vs_climatology": float(
                summary.loc["AO_SERVED_ENSEMBLE_50_50", "brier_skill_vs_climatology"]
            ),
            "any_reliable_difference": bool(
                prediction["uncertainty"][
                    ["ao_reliably_better", "external_reliably_better"]
                ].to_numpy().any()
            ),
        },
        "rating_axis": {
            "season": RATING_SEASON,
            "teams": int(rating["summary"]["teams"].iloc[0]),
            "spearman_vs_realized": {
                str(row["rating"]): float(row["spearman_vs_realized"])
                for _, row in rating["summary"].iterrows()
            },
            "comparisons": [
                {
                    "comparison": str(row["comparison"]),
                    "reference_type": str(row["reference_type"]),
                    "spearman_difference": float(row["spearman_difference"]),
                    "ci_95": [float(row["ci_95_lower"]), float(row["ci_95_upper"])],
                    "ao_reliably_better": bool(row["ao_reliably_better"]),
                    "benchmark_reliably_better": bool(row["benchmark_reliably_better"]),
                    "rank_agreement_spearman": float(row["rank_agreement_spearman"]),
                }
                for _, row in rating["comparisons"].iterrows()
            ],
        },
        "rating_feedback": False,
        "changes_production_parameters": False,
    }


def write_report(
    path: Path,
    prediction: dict[str, object],
    rating: dict[str, object],
    manifest: dict[str, object],
) -> None:
    lines = [
        "# Guncel Production Modeli Dis Benchmark",
        "",
        "Iki bagimsiz dis eksen. Hicbir production parametresi degistirilmez;",
        "dondurulmus tahmin ve rating artefact'lari puanlanir.",
        "",
        "## Eksen 1: ClubElo'ya karsi mac tahmini",
        "",
        f"- Eslesmis mac: `{manifest['prediction_axis']['paired_matches']}`.",
        f"- Puanlanan sezonlar: `{', '.join(prediction['scored_seasons'])}`.",
        f"- ClubElo snapshot yas siniri: `{manifest['prediction_axis']['max_snapshot_age_days']}` gun.",
        "- ClubElo ev sahibi avantaji yalniz onceki sezonlardan walk-forward fit edilir.",
        "- Iki tarafa da ayni score-preserving beraberlik modeli uygulanir; bu, karsilastirmayi",
        "  beraberlik modeli farkindan aritip rating kalitesine odaklar.",
        "- `CLUBELO_TUNED_SCALE_AND_H`, ClubElo'ya olcek ve H'yi birlikte fit eden cömert koldur.",
        "",
        "```text",
        prediction["model_summary"].to_string(index=False),
        "```",
        "",
        "### Bagimliliga dayanikli guven zarfi",
        "",
        "Pozitif fark ClubElo'nun onde oldugunu gosterir.",
        "",
        "```text",
        prediction["uncertainty"].to_string(index=False),
        "```",
        "",
        "### Turnuva kirilimi",
        "",
        "```text",
        prediction["competition_summary"].to_string(index=False),
        "```",
        "",
        "### ClubElo walk-forward kalibrasyonu",
        "",
        "```text",
        prediction["clubelo_fits"].to_string(index=False),
        "```",
        "",
        "### Kapsam",
        "",
        "```text",
        prediction["coverage"].to_string(index=False),
        "```",
        "",
        "## Eksen 2: Sezon basi rating",
        "",
        f"- Sezon: `{RATING_SEASON}`; takim: `{manifest['rating_axis']['teams']}`.",
        "- Opta snapshot `2025-07-03`, ilk 2025/26 macindan kesin olarak once alinmistir.",
        "- Hedef, hicbir ratingden etkilenmeyen leave-team-out schedule-adjusted",
        "  sezon performansidir.",
        "- `reference_type` kolonu kanit degerini belirler: `EXTERNAL` bagimsiz hakemdir,",
        "  `OWN_INPUT` modelin kendi girdisidir ve yalnizca value-added tabanini olcer.",
        "",
        "```text",
        rating["summary"].to_string(index=False),
        "```",
        "",
        "Eslesmis Spearman farklari:",
        "",
        "```text",
        rating["comparisons"].to_string(index=False),
        "```",
        "",
        "UEFA kulup katsayisi bagimsiz bir benchmark degildir: `club_points_t_*` girdileri",
        "o katsayinin bes sezonluk bilesenleridir. Bu satir yalniz su soruyu cevaplar:",
        "statik pipeline, tukettigi ham girdinin uzerine deger katiyor mu? Yayinlanmis 2026",
        "katsayisi kullanilmaz, cunku bes sezonluk penceresi tahmin edilen sezonu icerir.",
        "",
        "## Yorum siniri",
        "",
        "ClubElo arsivi agirlikla yerlesik kulupleri kapsar; eslesmis ornek tum eleme",
        "turu takimlarini temsil etmez. Eksen 1 bu nedenle yararli bir dis diagnostiktir,",
        "evrensel ustunluk iddiasi degildir. AO parametreleri ayni sezonlarda kalibre",
        "edildigi icin bu kosu da saf prospective holdout sayilmaz; 2026/27 kilitli",
        "ledger'i ayri kalir.",
        "",
        "Eksen 2 tek sezonluk bir olcumdur ve Opta snapshot'i ticari, kapali bir modeldir.",
        "Sonuc AO First Elo'nun yanlis oldugunu degil, ayni siralamayi daha gurultulu",
        "urettigini gosterir.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
