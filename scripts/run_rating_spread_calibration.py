from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ao_elo.config import AOEuropeanEloConfig  # noqa: E402
from ao_elo.pipeline import compute_ao_first_elo_from_csv  # noqa: E402


DATA_ROOT = ROOT / "data" / "backtest_stage_b_2018_2026"
OUTPUT_ROOT = ROOT / "output" / "rating_spread_calibration_2018_2026"
BASELINE_SCALE = 400.0
BASELINE_HOME_ADVANTAGE = 70.0
ELO_SCALES = (100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 600, 800)
HOME_ADVANTAGES = (0, 25, 50, 70, 75, 100, 125, 150)
COMPETITIONS = ("UCL", "UEL", "UECL")


@dataclass(frozen=True)
class ProbabilityConfig:
    elo_scale: float
    home_advantage: float

    @property
    def spread_multiplier(self) -> float:
        return BASELINE_SCALE / self.elo_scale


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate AO First Elo rating-difference dispersion"
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    seasons = tuple(
        folder.name.replace("-", "/")
        for folder in sorted(data_root.glob("20??-??"))
    )
    if len(seasons) < 3:
        raise ValueError("Rating spread calibration requires at least three seasons")

    output_root.mkdir(parents=True, exist_ok=True)
    ratings, matches = load_backtest(data_root, seasons)
    distribution = rating_distribution(ratings)
    difference_bands = observed_difference_bands(matches)
    folds = expanding_folds(seasons)
    fold_selections, fold_results, paired_losses = run_nested_calibration(matches, folds)
    competition_summary = summarize_competitions(fold_results)
    uncertainty = summarize_paired_uncertainty(paired_losses)
    decision = calibration_decision(fold_selections, fold_results, competition_summary)

    ratings.to_csv(output_root / "all_ratings.csv", index=False)
    distribution.to_csv(output_root / "rating_distribution.csv", index=False)
    difference_bands.to_csv(output_root / "observed_difference_bands.csv", index=False)
    fold_selections.to_csv(output_root / "fold_selections.csv", index=False)
    fold_results.to_csv(output_root / "fold_results.csv", index=False)
    competition_summary.to_csv(output_root / "competition_summary.csv", index=False)
    uncertainty.to_csv(output_root / "paired_uncertainty.csv", index=False)
    write_report(
        output_root / "spread_calibration_report.md",
        distribution,
        difference_bands,
        fold_selections,
        fold_results,
        competition_summary,
        uncertainty,
        decision,
    )

    print("AO European Elo rating spread calibration")
    print(f"Seasons: {len(seasons)}")
    print(f"Team-seasons: {len(ratings)}")
    print(f"Matches: {len(matches)}")
    print(f"Outer folds: {len(folds)}")
    print(f"Decision: {decision['decision']}")
    print(f"Report: {output_root / 'spread_calibration_report.md'}")


def load_backtest(
    data_root: Path,
    seasons: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = AOEuropeanEloConfig.v1_1()
    rating_frames = []
    match_frames = []
    for season in seasons:
        folder = data_root / season.replace("/", "-")
        ratings = compute_ao_first_elo_from_csv(
            folder / "teams.csv",
            folder / "country_coefficients.csv",
            folder / "domestic_context.csv",
            folder / "club_european_points.csv",
            config,
        )
        rating_frames.append(
            ratings[
                [
                    "season",
                    "team_id",
                    "team_name",
                    "competition",
                    "european_exposure",
                    "ao_first_elo",
                    "ao_first_elo_rank",
                ]
            ]
        )
        match_frames.append(prepare_matches(ratings, pd.read_csv(folder / "matches.csv")))
    return (
        pd.concat(rating_frames, ignore_index=True),
        pd.concat(match_frames, ignore_index=True),
    )


def prepare_matches(ratings: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    home = ratings[["season", "team_id", "ao_first_elo"]].rename(
        columns={"team_id": "home_team_id", "ao_first_elo": "home_rating"}
    )
    away = ratings[["season", "team_id", "ao_first_elo"]].rename(
        columns={"team_id": "away_team_id", "ao_first_elo": "away_rating"}
    )
    data = matches.merge(
        home,
        on=["season", "home_team_id"],
        how="left",
        validate="many_to_one",
    ).merge(
        away,
        on=["season", "away_team_id"],
        how="left",
        validate="many_to_one",
    )
    if data[["home_rating", "away_rating"]].isna().any().any():
        raise ValueError("Match data contains a team without an AO First Elo rating")
    data["rating_difference"] = data["home_rating"] - data["away_rating"]
    data["actual_home_score"] = 0.5
    data.loc[data["home_goals"] > data["away_goals"], "actual_home_score"] = 1.0
    data.loc[data["home_goals"] < data["away_goals"], "actual_home_score"] = 0.0
    return data


def expected_home_score(
    rating_difference: pd.Series | np.ndarray,
    config: ProbabilityConfig,
) -> np.ndarray:
    if config.elo_scale <= 0:
        raise ValueError("elo_scale must be > 0")
    difference = np.asarray(rating_difference, dtype=float) + config.home_advantage
    return 1.0 / (1.0 + np.power(10.0, -difference / config.elo_scale))


def prediction_metrics(
    data: pd.DataFrame,
    config: ProbabilityConfig,
) -> dict[str, float | int]:
    if data.empty:
        return {
            "matches": 0,
            "log_loss": float("nan"),
            "brier": float("nan"),
            "mean_expected_home_score": float("nan"),
            "mean_actual_home_score": float("nan"),
        }
    probability = np.clip(
        expected_home_score(data["rating_difference"], config),
        1e-12,
        1 - 1e-12,
    )
    actual = data["actual_home_score"].to_numpy(dtype=float)
    log_loss = -np.mean(actual * np.log(probability) + (1 - actual) * np.log(1 - probability))
    return {
        "matches": int(len(data)),
        "log_loss": float(log_loss),
        "brier": float(np.mean(np.square(probability - actual))),
        "mean_expected_home_score": float(np.mean(probability)),
        "mean_actual_home_score": float(np.mean(actual)),
    }


def candidate_grid() -> tuple[ProbabilityConfig, ...]:
    values = {
        (float(scale), float(home_advantage)): ProbabilityConfig(
            float(scale),
            float(home_advantage),
        )
        for scale in ELO_SCALES
        for home_advantage in HOME_ADVANTAGES
    }
    values[(BASELINE_SCALE, BASELINE_HOME_ADVANTAGE)] = ProbabilityConfig(
        BASELINE_SCALE,
        BASELINE_HOME_ADVANTAGE,
    )
    return tuple(values[key] for key in sorted(values))


def select_candidate(data: pd.DataFrame) -> tuple[ProbabilityConfig, dict[str, float | int]]:
    rows = []
    for config in candidate_grid():
        metrics = prediction_metrics(data, config)
        rows.append(
            {
                "config": config,
                **metrics,
                "distance_from_baseline": abs(config.elo_scale - BASELINE_SCALE)
                + abs(config.home_advantage - BASELINE_HOME_ADVANTAGE),
            }
        )
    selected = min(
        rows,
        key=lambda row: (
            row["log_loss"],
            row["brier"],
            row["distance_from_baseline"],
        ),
    )
    return selected["config"], selected


def select_spread_only_candidate(
    data: pd.DataFrame,
) -> tuple[ProbabilityConfig, dict[str, float | int]]:
    rows = []
    for scale in ELO_SCALES:
        config = ProbabilityConfig(float(scale), BASELINE_HOME_ADVANTAGE)
        rows.append({"config": config, **prediction_metrics(data, config)})
    selected = min(
        rows,
        key=lambda row: (
            row["log_loss"],
            row["brier"],
            abs(row["config"].elo_scale - BASELINE_SCALE),
        ),
    )
    return selected["config"], selected


def expanding_folds(
    seasons: tuple[str, ...],
    minimum_train_seasons: int = 2,
) -> tuple[tuple[tuple[str, ...], str], ...]:
    if minimum_train_seasons < 1 or len(seasons) <= minimum_train_seasons:
        raise ValueError("Not enough seasons for expanding folds")
    return tuple(
        (seasons[:index], seasons[index])
        for index in range(minimum_train_seasons, len(seasons))
    )


def run_nested_calibration(
    matches: pd.DataFrame,
    folds: tuple[tuple[tuple[str, ...], str], ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline = ProbabilityConfig(BASELINE_SCALE, BASELINE_HOME_ADVANTAGE)
    selections = []
    results = []
    paired_frames = []
    for fold, (train_seasons, test_season) in enumerate(folds, start=1):
        train = matches.loc[matches["season"].isin(train_seasons)]
        test = matches.loc[matches["season"].eq(test_season)]
        selected, train_metrics = select_candidate(train)
        spread_only, spread_train_metrics = select_spread_only_candidate(train)
        selections.append(
            {
                "fold": fold,
                "train_seasons": "|".join(train_seasons),
                "test_season": test_season,
                "selected_elo_scale": selected.elo_scale,
                "selected_home_advantage": selected.home_advantage,
                "equivalent_spread_multiplier": selected.spread_multiplier,
                "train_log_loss": train_metrics["log_loss"],
                "train_brier": train_metrics["brier"],
                "spread_only_elo_scale": spread_only.elo_scale,
                "spread_only_multiplier": spread_only.spread_multiplier,
                "spread_only_train_log_loss": spread_train_metrics["log_loss"],
            }
        )
        paired = test[["competition"]].copy()
        paired.insert(0, "fold", fold)
        paired.insert(1, "test_season", test_season)
        paired["baseline_loss"] = match_log_losses(test, baseline)
        paired["joint_loss"] = match_log_losses(test, selected)
        paired["spread_only_loss"] = match_log_losses(test, spread_only)
        paired_frames.append(paired)
        for competition in ("ALL", *COMPETITIONS):
            subset = test if competition == "ALL" else test.loc[test["competition"].eq(competition)]
            for model, config in (
                ("baseline", baseline),
                ("joint", selected),
                ("spread_only", spread_only),
            ):
                results.append(
                    {
                        "fold": fold,
                        "test_season": test_season,
                        "competition": competition,
                        "model": model,
                        "elo_scale": config.elo_scale,
                        "home_advantage": config.home_advantage,
                        "spread_multiplier": config.spread_multiplier,
                        **prediction_metrics(subset, config),
                    }
                )
    return (
        pd.DataFrame(selections),
        pd.DataFrame(results),
        pd.concat(paired_frames, ignore_index=True),
    )


def match_log_losses(data: pd.DataFrame, config: ProbabilityConfig) -> np.ndarray:
    probability = np.clip(
        expected_home_score(data["rating_difference"], config),
        1e-12,
        1 - 1e-12,
    )
    actual = data["actual_home_score"].to_numpy(dtype=float)
    return -(actual * np.log(probability) + (1 - actual) * np.log(1 - probability))


def rating_distribution(ratings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, frame in [("ALL", ratings), *ratings.groupby("season", sort=True)]:
        values = frame["ao_first_elo"]
        rows.append(
            {
                "season": season,
                "teams": len(values),
                "minimum": values.min(),
                "p05": values.quantile(0.05),
                "p25": values.quantile(0.25),
                "median": values.median(),
                "p75": values.quantile(0.75),
                "p95": values.quantile(0.95),
                "maximum": values.max(),
                "standard_deviation": values.std(ddof=0),
                "iqr": values.quantile(0.75) - values.quantile(0.25),
                "range": values.max() - values.min(),
            }
        )
    return pd.DataFrame(rows)


def observed_difference_bands(matches: pd.DataFrame) -> pd.DataFrame:
    edges = [-np.inf, 25, 50, 75, 100, 150, 200, 300, np.inf]
    labels = ["0-25", "25-50", "50-75", "75-100", "100-150", "150-200", "200-300", "300+"]
    data = matches.copy()
    data["absolute_rating_difference"] = data["rating_difference"].abs()
    data["higher_rated_team_score"] = np.where(
        data["rating_difference"] >= 0,
        data["actual_home_score"],
        1.0 - data["actual_home_score"],
    )
    data["difference_band"] = pd.cut(
        data["absolute_rating_difference"],
        bins=edges,
        labels=labels,
        right=False,
    )
    return (
        data.groupby("difference_band", observed=True, sort=True)
        .agg(
            matches=("actual_home_score", "size"),
            mean_absolute_rating_difference=("absolute_rating_difference", "mean"),
            higher_rated_team_score=("higher_rated_team_score", "mean"),
            draw_rate=("actual_home_score", lambda values: values.eq(0.5).mean()),
        )
        .reset_index()
    )


def summarize_competitions(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (competition, model), group in results.groupby(["competition", "model"], sort=True):
        group = group.loc[group["matches"].gt(0) & group["log_loss"].notna()]
        weight = group["matches"]
        rows.append(
            {
                "competition": competition,
                "model": model,
                "matches": int(weight.sum()),
                "log_loss": float(np.average(group["log_loss"], weights=weight)),
                "brier": float(np.average(group["brier"], weights=weight)),
            }
        )
    summary = pd.DataFrame(rows)
    baseline = summary.loc[summary["model"].eq("baseline")].set_index("competition")
    summary["log_loss_delta"] = summary.apply(
        lambda row: row["log_loss"] - baseline.loc[row["competition"], "log_loss"],
        axis=1,
    )
    return summary


def summarize_paired_uncertainty(
    paired: pd.DataFrame,
    bootstrap_samples: int = 5000,
) -> pd.DataFrame:
    rng = np.random.default_rng(20260713)
    rows = []
    for competition in ("ALL", *COMPETITIONS):
        frame = paired if competition == "ALL" else paired.loc[paired["competition"].eq(competition)]
        for comparison, candidate_column in (
            ("joint_vs_baseline", "joint_loss"),
            ("spread_only_vs_baseline", "spread_only_loss"),
        ):
            differences = (frame[candidate_column] - frame["baseline_loss"]).to_numpy(dtype=float)
            if len(differences) == 0:
                continue
            bootstrap_means = np.empty(bootstrap_samples)
            for index in range(bootstrap_samples):
                sample = rng.integers(0, len(differences), len(differences))
                bootstrap_means[index] = differences[sample].mean()
            lower, upper = np.quantile(bootstrap_means, (0.025, 0.975))
            rows.append(
                {
                    "competition": competition,
                    "comparison": comparison,
                    "matches": len(differences),
                    "mean_log_loss_delta": differences.mean(),
                    "ci_95_lower": lower,
                    "ci_95_upper": upper,
                    "directionally_reliable_improvement": bool(upper < 0),
                    "directionally_reliable_harm": bool(lower > 0),
                }
            )
    return pd.DataFrame(rows)


def calibration_decision(
    selections: pd.DataFrame,
    results: pd.DataFrame,
    competition_summary: pd.DataFrame,
) -> dict[str, object]:
    overall = results.loc[results["competition"].eq("ALL")].pivot(
        index="fold",
        columns="model",
        values="log_loss",
    )
    improved_folds = int((overall["spread_only"] < overall["baseline"] - 1e-9).sum())
    selected_competitions = competition_summary.loc[
        competition_summary["model"].eq("spread_only")
        & competition_summary["competition"].isin(COMPETITIONS)
    ]
    non_worse_competitions = int((selected_competitions["log_loss_delta"] <= 1e-9).sum())
    scale_range = float(
        selections["spread_only_elo_scale"].max() - selections["spread_only_elo_scale"].min()
    )
    home_range = 0.0
    stable = scale_range <= 100
    promote = (
        improved_folds == len(overall)
        and non_worse_competitions == len(COMPETITIONS)
        and stable
    )
    return {
        "decision": "CALIBRATE_RATING_SPREAD" if promote else "KEEP_CURRENT_RATING_SPREAD",
        "improved_folds": improved_folds,
        "total_folds": len(overall),
        "non_worse_competitions": non_worse_competitions,
        "scale_range": scale_range,
        "home_advantage_range": home_range,
        "stable": stable,
    }


def write_report(
    path: Path,
    distribution: pd.DataFrame,
    difference_bands: pd.DataFrame,
    selections: pd.DataFrame,
    results: pd.DataFrame,
    competition_summary: pd.DataFrame,
    uncertainty: pd.DataFrame,
    decision: dict[str, object],
) -> None:
    overall_distribution = distribution.loc[distribution["season"].eq("ALL")].iloc[0]
    overall_results = results.loc[results["competition"].eq("ALL")].pivot(
        index=["fold", "test_season"],
        columns="model",
        values="log_loss",
    ).reset_index()
    overall_results["spread_only_delta"] = (
        overall_results["spread_only"] - overall_results["baseline"]
    )
    overall_results["joint_delta"] = overall_results["joint"] - overall_results["baseline"]
    fold_table = selections.merge(overall_results, on="fold", validate="one_to_one")

    lines = [
        "# AO European Elo Rating Spread Calibration",
        "",
        "This analysis calibrates only the conversion from pre-season rating differences",
        "to expected match score. It does not change team order or apply match-by-match Elo updates.",
        "",
        "Baseline probability conversion: `elo_scale=400`, `home_advantage=70`.",
        "A lower selected scale means the existing numerical rating gaps should have a",
        "stronger match-probability effect; it does not by itself require rewriting ratings.",
        "",
        "## Rating distribution",
        "",
        f"- Team-seasons: {int(overall_distribution['teams'])}",
        f"- Minimum / maximum: {overall_distribution['minimum']:.3f} / {overall_distribution['maximum']:.3f}",
        f"- Range: {overall_distribution['range']:.3f}",
        f"- Standard deviation: {overall_distribution['standard_deviation']:.3f}",
        f"- Interquartile range: {overall_distribution['iqr']:.3f}",
        "",
        "## Nested walk-forward selections",
        "",
        dataframe_to_markdown(
            fold_table[
                [
                    "fold",
                    "train_seasons",
                    "test_season_x",
                    "selected_elo_scale",
                    "selected_home_advantage",
                    "equivalent_spread_multiplier",
                    "spread_only_elo_scale",
                    "spread_only_multiplier",
                    "baseline",
                    "spread_only",
                    "spread_only_delta",
                    "joint",
                    "joint_delta",
                ]
            ].rename(columns={"test_season_x": "test_season"})
        ),
        "",
        "## Competition results",
        "",
        dataframe_to_markdown(competition_summary),
        "",
        "## Paired match-level uncertainty",
        "",
        dataframe_to_markdown(uncertainty),
        "",
        "## Observed score by raw rating difference",
        "",
        dataframe_to_markdown(difference_bands),
        "",
        "## Decision",
        "",
        f"- Decision: `{decision['decision']}`",
        f"- Unseen folds improved: `{decision['improved_folds']}/{decision['total_folds']}`",
        f"- Competitions not worse: `{decision['non_worse_competitions']}/{len(COMPETITIONS)}`",
        f"- Selected scale range: `{decision['scale_range']:.0f}`",
        f"- Selected home-advantage range: `{decision['home_advantage_range']:.0f}`",
        f"- Parameter stability gate: `{decision['stable']}`",
        "",
        "No visual preference for wider ratings can override failed unseen-season or",
        "competition stability gates.",
        "Match-level bootstrap intervals are diagnostic and do not model correlation",
        "between two-legged ties or repeated appearances by the same club.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include=["number"]).columns:
        if pd.api.types.is_integer_dtype(display[column]):
            display[column] = display[column].map(str)
        else:
            display[column] = display[column].map(lambda value: f"{value:.6f}")
    headers = [str(column) for column in display.columns]
    rows = [
        [str(value).replace("|", "\\|") for value in row]
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(row) + " |" for row in rows),
        ]
    )


if __name__ == "__main__":
    main()
