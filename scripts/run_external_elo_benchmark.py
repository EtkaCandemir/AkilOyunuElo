from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
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
BENCHMARK_DATA_ROOT = ROOT / "data" / "external_elo_benchmark_2018_2026"
OUTPUT_ROOT = ROOT / "output" / "external_elo_benchmark_2018_2026"
AO_CORE = DynamicCoreConfig(225.0, 40.0, 28.0)
AO_CARRY = AchievementCarryConfig(0.85, 0.0, 0.0)
CLUBELO_SCALE = 400.0
CLUBELO_HOME_ADVANTAGES = tuple(float(value) for value in range(0, 121, 10))


@dataclass(frozen=True)
class BenchmarkMetrics:
    matches: int
    brier: float
    log_loss: float


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare AO live Elo with pre-match external ClubElo on identical matches"
    )
    parser.add_argument("--static-data-root", type=Path, default=STATIC_DATA_ROOT)
    parser.add_argument("--benchmark-data-root", type=Path, default=BENCHMARK_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    data_root = args.benchmark_data_root.resolve()
    benchmark = load_benchmark_data(data_root / "matches_with_dates_and_external_elo.csv")
    carry_data, _ = load_carry_data(
        args.static_data_root.resolve(),
        data_root / "exact_date_events.csv",
    )
    _, ao_predictions, _ = evaluate_sequence(
        carry_data,
        AO_CORE,
        AO_CARRY,
        return_predictions=True,
    )
    assert ao_predictions is not None
    paired = prepare_paired_data(benchmark, ao_predictions)
    seasons = tuple(sorted(paired["season"].unique()))
    folds = expanding_folds(seasons)
    selections, predictions = run_walk_forward(paired, folds)
    fold_results = summarize_folds(predictions)
    competition_summary = summarize_groups(predictions, "competition")
    season_summary = summarize_groups(predictions, "test_season")
    uncertainty = paired_uncertainty(predictions)
    coverage = coverage_table(benchmark)

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    selections.to_csv(output_root / "fold_selections.csv", index=False)
    predictions.to_csv(output_root / "paired_predictions.csv", index=False)
    fold_results.to_csv(output_root / "fold_results.csv", index=False)
    competition_summary.to_csv(output_root / "competition_summary.csv", index=False)
    season_summary.to_csv(output_root / "season_summary.csv", index=False)
    uncertainty.to_csv(output_root / "paired_uncertainty.csv", index=False)
    coverage.to_csv(output_root / "coverage.csv", index=False)
    write_report(
        output_root / "benchmark_report.md",
        benchmark,
        paired,
        selections,
        fold_results,
        competition_summary,
        uncertainty,
        coverage,
    )

    ao = predictions["ao_brier_loss"].mean()
    external = predictions["clubelo_brier_loss"].mean()
    print("AO vs external ClubElo exact-date benchmark")
    print(f"Eligible paired matches: {len(paired)}")
    print(f"Unseen test matches: {len(predictions)}")
    print(f"AO Brier: {ao:.6f}")
    print(f"ClubElo Brier: {external:.6f}")
    print(f"AO - ClubElo Brier: {ao - external:+.6f}")
    print(f"Report: {output_root / 'benchmark_report.md'}")


def load_benchmark_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {
        "match_id",
        "season",
        "competition",
        "round",
        "actual_home_score",
        "is_neutral",
        "kickoff_utc",
        "clubelo_home_elo",
        "clubelo_away_elo",
        "clubelo_home_snapshot_date",
        "clubelo_away_snapshot_date",
        "external_elo_pair_available",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Benchmark data missing columns: {missing}")
    if data["match_id"].duplicated().any():
        raise ValueError("Benchmark match_id must be unique")
    data["kickoff_utc"] = pd.to_datetime(data["kickoff_utc"], utc=True, errors="raise")
    data["clubelo_home_snapshot_date"] = pd.to_datetime(
        data["clubelo_home_snapshot_date"], errors="coerce"
    )
    data["clubelo_away_snapshot_date"] = pd.to_datetime(
        data["clubelo_away_snapshot_date"], errors="coerce"
    )
    eligible = data["external_elo_pair_available"].astype(bool)
    eligible_data = data.loc[eligible]
    match_dates = eligible_data["kickoff_utc"].dt.tz_localize(None).dt.normalize()
    if not eligible_data["clubelo_home_snapshot_date"].lt(match_dates).all():
        raise ValueError("Home ClubElo snapshot leakage detected")
    if not eligible_data["clubelo_away_snapshot_date"].lt(match_dates).all():
        raise ValueError("Away ClubElo snapshot leakage detected")
    return data


def prepare_paired_data(
    benchmark: pd.DataFrame,
    ao_predictions: pd.DataFrame,
) -> pd.DataFrame:
    ao_columns = ["match_id", "expected_home_score", "brier_loss", "log_loss"]
    if ao_predictions["match_id"].duplicated().any():
        raise ValueError("AO predictions contain duplicate match IDs")
    paired = benchmark.loc[benchmark["external_elo_pair_available"].astype(bool)].merge(
        ao_predictions[ao_columns].rename(
            columns={
                "expected_home_score": "ao_expected_home_score",
                "brier_loss": "ao_brier_loss",
                "log_loss": "ao_log_loss",
            }
        ),
        on="match_id",
        how="inner",
        validate="one_to_one",
    )
    if len(paired) != int(benchmark["external_elo_pair_available"].sum()):
        raise ValueError("An eligible external benchmark match lacks an AO prediction")
    return paired.sort_values(["kickoff_utc", "match_id"], kind="stable").reset_index(drop=True)


def clubelo_expected_home_score(
    home_elo: float,
    away_elo: float,
    home_advantage: float,
    *,
    neutral: bool,
) -> float:
    values = (home_elo, away_elo, home_advantage)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("ClubElo prediction inputs must be finite")
    if home_advantage < 0:
        raise ValueError("ClubElo home advantage must be non-negative")
    advantage = 0.0 if neutral else home_advantage
    exponent = -((home_elo - away_elo + advantage) / CLUBELO_SCALE)
    return 1.0 / (1.0 + 10.0 ** exponent)


def add_clubelo_predictions(data: pd.DataFrame, home_advantage: float) -> pd.DataFrame:
    result = data.copy()
    probabilities = np.array(
        [
            clubelo_expected_home_score(home, away, home_advantage, neutral=bool(neutral))
            for home, away, neutral in zip(
                result["clubelo_home_elo"],
                result["clubelo_away_elo"],
                result["is_neutral"],
            )
        ]
    )
    probabilities = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    actual = result["actual_home_score"].to_numpy(float)
    result["clubelo_expected_home_score"] = probabilities
    result["clubelo_brier_loss"] = (probabilities - actual) ** 2
    result["clubelo_log_loss"] = -(
        actual * np.log(probabilities) + (1.0 - actual) * np.log(1.0 - probabilities)
    )
    return result


def select_clubelo_home_advantage(train: pd.DataFrame) -> tuple[float, BenchmarkMetrics]:
    rows = []
    for home_advantage in CLUBELO_HOME_ADVANTAGES:
        predictions = add_clubelo_predictions(train, home_advantage)
        metrics = metrics_from_losses(
            predictions["clubelo_brier_loss"],
            predictions["clubelo_log_loss"],
        )
        rows.append((metrics.brier, metrics.log_loss, abs(home_advantage - 60.0), home_advantage, metrics))
    rows.sort(key=lambda row: row[:4])
    return rows[0][3], rows[0][4]


def run_walk_forward(
    paired: pd.DataFrame,
    folds: tuple[tuple[tuple[str, ...], str], ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selection_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for fold_number, (train_seasons, test_season) in enumerate(folds, start=1):
        train = paired.loc[paired["season"].isin(train_seasons)]
        test = paired.loc[paired["season"].eq(test_season)]
        if train.empty or test.empty:
            continue
        selected_h, train_metrics = select_clubelo_home_advantage(train)
        predicted = add_clubelo_predictions(test, selected_h)
        predicted.insert(0, "fold", fold_number)
        predicted.insert(1, "test_season", test_season)
        predicted["selected_clubelo_home_advantage"] = selected_h
        prediction_frames.append(predicted)
        selection_rows.append(
            {
                "fold": fold_number,
                "train_seasons": ",".join(train_seasons),
                "test_season": test_season,
                "train_matches": len(train),
                "test_matches": len(test),
                "selected_clubelo_home_advantage": selected_h,
                "train_clubelo_brier": train_metrics.brier,
                "train_clubelo_log_loss": train_metrics.log_loss,
            }
        )
    if not prediction_frames:
        raise ValueError("No external benchmark folds could be evaluated")
    return pd.DataFrame(selection_rows), pd.concat(prediction_frames, ignore_index=True)


def metrics_from_losses(brier: pd.Series, log_loss: pd.Series) -> BenchmarkMetrics:
    if len(brier) == 0:
        raise ValueError("Cannot compute metrics for an empty sample")
    return BenchmarkMetrics(len(brier), float(brier.mean()), float(log_loss.mean()))


def summarize_folds(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fold, season), data in predictions.groupby(["fold", "test_season"], sort=True):
        for model, prefix in (("AO_current", "ao"), ("ClubElo_external", "clubelo")):
            metrics = metrics_from_losses(data[f"{prefix}_brier_loss"], data[f"{prefix}_log_loss"])
            rows.append(
                {
                    "fold": fold,
                    "test_season": season,
                    "model": model,
                    "matches": metrics.matches,
                    "brier": metrics.brier,
                    "log_loss": metrics.log_loss,
                }
            )
    return pd.DataFrame(rows)


def summarize_groups(predictions: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows = []
    for group, data in predictions.groupby(group_column, sort=True):
        rows.append(
            {
                group_column: group,
                "matches": len(data),
                "ao_brier": data["ao_brier_loss"].mean(),
                "clubelo_brier": data["clubelo_brier_loss"].mean(),
                "ao_minus_clubelo_brier": (
                    data["ao_brier_loss"] - data["clubelo_brier_loss"]
                ).mean(),
                "ao_log_loss": data["ao_log_loss"].mean(),
                "clubelo_log_loss": data["clubelo_log_loss"].mean(),
            }
        )
    return pd.DataFrame(rows)


def paired_uncertainty(
    predictions: pd.DataFrame,
    *,
    bootstrap_samples: int = 5000,
    seed: int = 20260715,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    groups = [("ALL", predictions), *predictions.groupby("competition", sort=True)]
    for competition, data in groups:
        differences = (data["ao_brier_loss"] - data["clubelo_brier_loss"]).to_numpy(float)
        sampled = rng.choice(differences, size=(bootstrap_samples, len(differences)), replace=True)
        means = sampled.mean(axis=1)
        lower, upper = np.quantile(means, (0.025, 0.975))
        rows.append(
            {
                "competition": competition,
                "matches": len(differences),
                "mean_ao_minus_clubelo_brier": differences.mean(),
                "ci_95_lower": lower,
                "ci_95_upper": upper,
                "ao_reliably_better": bool(upper < 0),
                "clubelo_reliably_better": bool(lower > 0),
            }
        )
    return pd.DataFrame(rows)


def coverage_table(benchmark: pd.DataFrame) -> pd.DataFrame:
    return (
        benchmark.groupby(["season", "competition"], sort=True)
        .agg(
            matches=("match_id", "size"),
            exact_dates=("exact_date_available", "sum"),
            eligible_external_pairs=("external_elo_pair_available", "sum"),
        )
        .reset_index()
        .assign(
            external_pair_coverage=lambda frame: (
                frame["eligible_external_pairs"] / frame["matches"]
            )
        )
    )


def write_report(
    path: Path,
    benchmark: pd.DataFrame,
    paired: pd.DataFrame,
    selections: pd.DataFrame,
    fold_results: pd.DataFrame,
    competition_summary: pd.DataFrame,
    uncertainty: pd.DataFrame,
    coverage: pd.DataFrame,
) -> None:
    tested = fold_results.groupby("model").apply(
        lambda data: pd.Series(
            {
                "matches": int(data["matches"].sum()),
                "brier": np.average(data["brier"], weights=data["matches"]),
                "log_loss": np.average(data["log_loss"], weights=data["matches"]),
            }
        ),
        include_groups=False,
    ).reset_index()
    overall = uncertainty.loc[uncertainty["competition"].eq("ALL")].iloc[0]
    direction = (
        "AO daha iyi"
        if overall["mean_ao_minus_clubelo_brier"] < 0
        else "ClubElo daha iyi"
    )
    reliable = bool(overall["ao_reliably_better"] or overall["clubelo_reliably_better"])
    lines = [
        "# AO vs Historical External ClubElo Benchmark",
        "",
        "## Contract",
        "",
        "- Exact UTC match dates: official UEFA match service.",
        "- External ratings: latest ClubElo-derived snapshot strictly before the match.",
        "- Snapshot age limit: 31 days.",
        "- ClubElo probability scale: 400, matching ClubElo's published Elo equation.",
        "- ClubElo home advantage: selected only on earlier seasons in each fold.",
        f"- AO candidate: scale={AO_CORE.elo_scale:g}, H={AO_CORE.home_advantage:g}, "
        f"K={AO_CORE.k_factor:g}, season power carry={AO_CARRY.power_carry:g}.",
        "",
        "## Result",
        "",
        f"Full source matches: {len(benchmark)}",
        f"Eligible paired matches: {len(paired)}",
        f"Walk-forward unseen test matches: {int(fold_results.loc[fold_results['model'].eq('AO_current'), 'matches'].sum())}",
        f"Direction: {direction}",
        f"95% paired bootstrap conclusive: {reliable}",
        "",
        "```text",
        tested.to_string(index=False),
        "```",
        "",
        "## Competition detail",
        "",
        "```text",
        competition_summary.to_string(index=False),
        "```",
        "",
        "## Uncertainty",
        "",
        "```text",
        uncertainty.to_string(index=False),
        "```",
        "",
        "## Selected external H",
        "",
        "```text",
        selections.to_string(index=False),
        "```",
        "",
        "## Coverage",
        "",
        "```text",
        coverage.to_string(index=False),
        "```",
        "",
        "## Interpretation limit",
        "",
        "The ClubElo snapshot archive covers mostly stronger, established clubs. This paired",
        "sample is therefore not representative of all qualifying-round teams. It is a useful",
        "external diagnostic, not a claim that either model is universally superior.",
        "The AO parameters were previously calibrated on overlapping seasons, so this run is",
        "not a pristine final holdout. A later untouched-season test remains necessary.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
