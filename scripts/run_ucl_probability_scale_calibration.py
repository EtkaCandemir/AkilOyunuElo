from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_achievement_carry_calibration import (  # noqa: E402
    AchievementCarryConfig,
    evaluate_sequence,
    load_carry_data,
)
from scripts.run_dynamic_core_calibration import DynamicCoreConfig, expanding_folds  # noqa: E402


STATIC_DATA_ROOT = ROOT / "data" / "backtest_stage_b_2018_2026"
EXACT_EVENTS_PATH = (
    ROOT / "data" / "external_elo_benchmark_2018_2026" / "exact_date_events.csv"
)
BENCHMARK_PATH = (
    ROOT
    / "data"
    / "external_elo_benchmark_2018_2026"
    / "matches_with_dates_and_external_elo.csv"
)
EXTERNAL_PREDICTIONS_PATH = (
    ROOT / "output" / "external_elo_benchmark_2018_2026" / "paired_predictions.csv"
)
OUTPUT_ROOT = ROOT / "output" / "ucl_probability_scale_calibration_2018_2026"
UPDATE_CORE = DynamicCoreConfig(225.0, 40.0, 28.0)
POWER_CARRY = AchievementCarryConfig(0.85, 0.0, 0.0)
BASELINE_SCALE = 225.0
SCALE_CANDIDATES = tuple(float(value) for value in range(200, 401, 25))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nested calibration of a UCL forecast scale without changing Elo updates"
    )
    parser.add_argument("--static-data-root", type=Path, default=STATIC_DATA_ROOT)
    parser.add_argument("--exact-events-path", type=Path, default=EXACT_EVENTS_PATH)
    parser.add_argument("--benchmark-path", type=Path, default=BENCHMARK_PATH)
    parser.add_argument(
        "--external-predictions-path",
        type=Path,
        default=EXTERNAL_PREDICTIONS_PATH,
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    datasets, _ = load_carry_data(
        args.static_data_root.resolve(),
        args.exact_events_path.resolve(),
    )
    _, live_predictions, _ = evaluate_sequence(
        datasets,
        UPDATE_CORE,
        POWER_CARRY,
        return_predictions=True,
    )
    assert live_predictions is not None
    ucl = load_ucl_pairs(args.benchmark_path.resolve(), live_predictions)
    external = pd.read_csv(args.external_predictions_path.resolve())
    seasons = tuple(sorted(ucl["season"].unique()))
    folds = expanding_folds(seasons)
    selections, predictions = run_nested_scale(ucl, external, folds)
    candidate_metrics = summarize_full_candidates(ucl)
    fold_results = summarize_fold_results(predictions)
    uncertainty = paired_uncertainty(predictions)
    calibration = calibration_bands(predictions)
    decision = calibration_decision(selections, uncertainty)

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    selections.to_csv(output_root / "fold_selections.csv", index=False)
    predictions.to_csv(output_root / "unseen_ucl_predictions.csv", index=False)
    candidate_metrics.to_csv(output_root / "full_candidate_metrics.csv", index=False)
    fold_results.to_csv(output_root / "fold_results.csv", index=False)
    uncertainty.to_csv(output_root / "paired_uncertainty.csv", index=False)
    calibration.to_csv(output_root / "probability_calibration_bands.csv", index=False)
    write_report(
        output_root / "calibration_report.md",
        ucl,
        selections,
        candidate_metrics,
        fold_results,
        uncertainty,
        calibration,
        decision,
    )

    selected = fold_results.loc[fold_results["model"].eq("selected_ucl_scale")]
    baseline = fold_results.loc[fold_results["model"].eq("baseline_scale_225")]
    external_rows = fold_results.loc[fold_results["model"].eq("clubelo_external")]
    print("UCL forecast probability scale calibration")
    print(f"UCL paired matches: {len(ucl)}")
    print(f"Unseen test matches: {int(selected['matches'].sum())}")
    print(f"Selected scale Brier: {weighted_brier(selected):.6f}")
    print(f"Scale 225 Brier: {weighted_brier(baseline):.6f}")
    print(f"ClubElo Brier: {weighted_brier(external_rows):.6f}")
    print(f"Decision: {decision}")
    print(f"Report: {output_root / 'calibration_report.md'}")


def load_ucl_pairs(path: Path, live_predictions: pd.DataFrame) -> pd.DataFrame:
    benchmark = pd.read_csv(path)
    paired = benchmark.loc[
        benchmark["external_elo_pair_available"].astype(bool)
        & benchmark["competition"].eq("UCL")
    ].copy()
    selected = live_predictions[
        ["match_id", "home_power", "away_power", "expected_home_score"]
    ].rename(columns={"expected_home_score": "update_scale_probability"})
    paired = paired.merge(selected, on="match_id", how="inner", validate="one_to_one")
    if len(paired) != int(
        (
            benchmark["external_elo_pair_available"].astype(bool)
            & benchmark["competition"].eq("UCL")
        ).sum()
    ):
        raise ValueError("An eligible UCL match lacks a live AO prediction")
    expected = forecast_probability(paired, BASELINE_SCALE)
    if not np.allclose(expected, paired["update_scale_probability"], atol=1e-12):
        raise ValueError("Scale 225 forecast does not reproduce the update-core probability")
    return paired.sort_values(["season", "kickoff_utc", "match_id"], kind="stable")


def forecast_probability(data: pd.DataFrame, scale: float) -> np.ndarray:
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Forecast scale must be positive and finite")
    advantage = np.where(data["is_neutral"].astype(bool), 0.0, UPDATE_CORE.home_advantage)
    difference = data["home_power"].to_numpy(float) - data["away_power"].to_numpy(float)
    return 1.0 / (1.0 + 10.0 ** (-((difference + advantage) / scale)))


def score_scale(data: pd.DataFrame, scale: float) -> tuple[float, float]:
    probability = np.clip(forecast_probability(data, scale), 1e-12, 1.0 - 1e-12)
    actual = data["actual_home_score"].to_numpy(float)
    brier = float(np.mean((probability - actual) ** 2))
    log_loss = float(
        np.mean(-(actual * np.log(probability) + (1.0 - actual) * np.log(1.0 - probability)))
    )
    return brier, log_loss


def select_scale(train: pd.DataFrame) -> tuple[float, float, float]:
    rows = []
    for scale in SCALE_CANDIDATES:
        brier, log_loss = score_scale(train, scale)
        rows.append((brier, log_loss, abs(scale - BASELINE_SCALE), scale))
    rows.sort()
    best = rows[0]
    return best[3], best[0], best[1]


def add_forecast_losses(data: pd.DataFrame, scale: float, prefix: str) -> pd.DataFrame:
    result = data.copy()
    probability = np.clip(forecast_probability(result, scale), 1e-12, 1.0 - 1e-12)
    actual = result["actual_home_score"].to_numpy(float)
    result[f"{prefix}_expected_home_score"] = probability
    result[f"{prefix}_brier_loss"] = (probability - actual) ** 2
    result[f"{prefix}_log_loss"] = -(
        actual * np.log(probability) + (1.0 - actual) * np.log(1.0 - probability)
    )
    return result


def run_nested_scale(
    ucl: pd.DataFrame,
    external: pd.DataFrame,
    folds: tuple[tuple[tuple[str, ...], str], ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    external_index = external.set_index("match_id")
    selection_rows = []
    prediction_frames = []
    for fold_number, (train_seasons, test_season) in enumerate(folds, start=1):
        train = ucl.loc[ucl["season"].isin(train_seasons)]
        test = ucl.loc[ucl["season"].eq(test_season)].copy()
        selected_scale, train_brier, train_log_loss = select_scale(train)
        selected = add_forecast_losses(test, selected_scale, "selected")
        selected = add_forecast_losses(selected, BASELINE_SCALE, "baseline")
        if not set(selected["match_id"]) <= set(external_index.index):
            raise ValueError(f"{test_season}: external predictions are incomplete")
        aligned_external = external_index.loc[selected["match_id"]]
        selected["clubelo_expected_home_score"] = aligned_external[
            "clubelo_expected_home_score"
        ].to_numpy(float)
        selected["clubelo_brier_loss"] = aligned_external["clubelo_brier_loss"].to_numpy(float)
        selected["clubelo_log_loss"] = aligned_external["clubelo_log_loss"].to_numpy(float)
        selected.insert(0, "fold", fold_number)
        selected.insert(1, "test_season", test_season)
        selected["selected_scale"] = selected_scale
        prediction_frames.append(selected)
        selection_rows.append(
            {
                "fold": fold_number,
                "train_seasons": ",".join(train_seasons),
                "test_season": test_season,
                "train_matches": len(train),
                "test_matches": len(test),
                "selected_scale": selected_scale,
                "train_brier": train_brier,
                "train_log_loss": train_log_loss,
            }
        )
    return pd.DataFrame(selection_rows), pd.concat(prediction_frames, ignore_index=True)


def summarize_full_candidates(ucl: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scale in SCALE_CANDIDATES:
        brier, log_loss = score_scale(ucl, scale)
        rows.append(
            {
                "forecast_scale": scale,
                "matches": len(ucl),
                "brier": brier,
                "log_loss": log_loss,
                "distance_from_225": abs(scale - BASELINE_SCALE),
            }
        )
    return pd.DataFrame(rows).sort_values(["brier", "log_loss", "distance_from_225"])


def summarize_fold_results(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fold, season), data in predictions.groupby(["fold", "test_season"], sort=True):
        for model, prefix in (
            ("selected_ucl_scale", "selected"),
            ("baseline_scale_225", "baseline"),
            ("clubelo_external", "clubelo"),
        ):
            rows.append(
                {
                    "fold": fold,
                    "test_season": season,
                    "model": model,
                    "matches": len(data),
                    "brier": data[f"{prefix}_brier_loss"].mean(),
                    "log_loss": data[f"{prefix}_log_loss"].mean(),
                }
            )
    return pd.DataFrame(rows)


def paired_uncertainty(
    predictions: pd.DataFrame,
    *,
    bootstrap_samples: int = 5000,
    seed: int = 20260715,
) -> pd.DataFrame:
    comparisons = (
        ("selected_vs_225", "selected", "baseline"),
        ("selected_vs_clubelo", "selected", "clubelo"),
        ("scale_225_vs_clubelo", "baseline", "clubelo"),
    )
    rng = np.random.default_rng(seed)
    rows = []
    for comparison, challenger, baseline in comparisons:
        differences = (
            predictions[f"{challenger}_brier_loss"]
            - predictions[f"{baseline}_brier_loss"]
        ).to_numpy(float)
        sampled = rng.choice(differences, size=(bootstrap_samples, len(differences)), replace=True)
        means = sampled.mean(axis=1)
        lower, upper = np.quantile(means, (0.025, 0.975))
        fold_means = (
            predictions.assign(difference=differences)
            .groupby("fold")["difference"]
            .mean()
        )
        rows.append(
            {
                "comparison": comparison,
                "matches": len(differences),
                "mean_brier_difference": differences.mean(),
                "ci_95_lower": lower,
                "ci_95_upper": upper,
                "folds_won": int(fold_means.lt(0).sum()),
                "folds": len(fold_means),
                "reliable_improvement": bool(upper < 0),
                "reliable_harm": bool(lower > 0),
            }
        )
    return pd.DataFrame(rows)


def calibration_bands(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    bins = np.linspace(0.0, 1.0, 6)
    for model, prefix in (
        ("selected_ucl_scale", "selected"),
        ("baseline_scale_225", "baseline"),
        ("clubelo_external", "clubelo"),
    ):
        probability = predictions[f"{prefix}_expected_home_score"]
        band = pd.cut(probability, bins=bins, include_lowest=True)
        for interval, data in predictions.assign(_band=band).groupby("_band", observed=True):
            rows.append(
                {
                    "model": model,
                    "probability_band": str(interval),
                    "matches": len(data),
                    "mean_prediction": data[f"{prefix}_expected_home_score"].mean(),
                    "mean_actual_score": data["actual_home_score"].mean(),
                    "calibration_gap": (
                        data[f"{prefix}_expected_home_score"].mean()
                        - data["actual_home_score"].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def calibration_decision(selections: pd.DataFrame, uncertainty: pd.DataFrame) -> str:
    comparison = uncertainty.loc[uncertainty["comparison"].eq("selected_vs_225")].iloc[0]
    stable = selections["selected_scale"].nunique() <= 3
    if (
        bool(comparison["reliable_improvement"])
        and int(comparison["folds_won"]) >= 4
        and stable
    ):
        return "PROVISIONAL_ACCEPT_UCL_FORECAST_SCALE_LAYER"
    return "KEEP_FORECAST_SCALE_225_NO_RELIABLE_UCL_GAIN"


def weighted_brier(rows: pd.DataFrame) -> float:
    return float(np.average(rows["brier"], weights=rows["matches"]))


def write_report(
    path: Path,
    ucl: pd.DataFrame,
    selections: pd.DataFrame,
    candidates: pd.DataFrame,
    fold_results: pd.DataFrame,
    uncertainty: pd.DataFrame,
    calibration: pd.DataFrame,
    decision: str,
) -> None:
    aggregate = (
        fold_results.groupby("model")
        .apply(
            lambda data: pd.Series(
                {
                    "matches": int(data["matches"].sum()),
                    "brier": weighted_brier(data),
                    "log_loss": float(np.average(data["log_loss"], weights=data["matches"])),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    lines = [
        "# UCL Forecast Probability Scale Calibration",
        "",
        "This layer changes only the mapping from pre-match AO rating difference to",
        "forecast probability. Elo updates, K, rating values and team ranking remain frozen.",
        "",
        f"Eligible UCL matches: {len(ucl)}",
        f"Decision: `{decision}`",
        "",
        "## Nested selections",
        "",
        "```text",
        selections.to_string(index=False),
        "```",
        "",
        "## Unseen aggregate",
        "",
        "```text",
        aggregate.to_string(index=False),
        "```",
        "",
        "## Paired uncertainty",
        "",
        "```text",
        uncertainty.to_string(index=False),
        "```",
        "",
        "## Full-data sensitivity",
        "",
        "```text",
        candidates.to_string(index=False),
        "```",
        "",
        "## Calibration bands",
        "",
        "```text",
        calibration.to_string(index=False),
        "```",
        "",
        "## Guardrail",
        "",
        "This diagnostic never changes AO Elo points or rank order. Acceptance requires",
        "at least four fold wins, a paired 95% improvement and stable scale selection.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
