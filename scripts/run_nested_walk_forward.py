from __future__ import annotations

import sys
from dataclasses import asdict, replace
from math import log
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ao_elo.backtest import evaluate_match_predictions  # noqa: E402
from ao_elo.config import AOEuropeanEloConfig  # noqa: E402
from ao_elo.pipeline import compute_ao_first_elo_from_csv  # noqa: E402


DATA_ROOT = ROOT / "data" / "backtest_stage_b_2021_2026"
OUTPUT_ROOT = ROOT / "output" / "backtest_nested_walk_forward"
SEASONS = ("2021/22", "2022/23", "2023/24", "2024/25", "2025/26")
COMPETITIONS = ("ALL", "UCL", "UEL", "UECL")
HOME_ADVANTAGE = 70.0


class Evaluator:
    def __init__(self) -> None:
        self._ratings: dict[tuple[str, str], pd.DataFrame] = {}
        self._matches: dict[str, pd.DataFrame] = {}

    def metrics(
        self,
        candidate: str,
        config: AOEuropeanEloConfig,
        seasons: tuple[str, ...] = SEASONS,
    ) -> pd.DataFrame:
        rows = []
        for season in seasons:
            ratings = self.ratings(config, season)
            matches = self.matches(season)
            for competition in COMPETITIONS:
                subset = (
                    matches
                    if competition == "ALL"
                    else matches.loc[matches["competition"] == competition]
                )
                values = evaluate_match_predictions(
                    ratings,
                    subset,
                    home_advantage=HOME_ADVANTAGE,
                )
                rows.append(
                    {
                        "candidate": candidate,
                        "season": season,
                        "competition": competition,
                        **values,
                    }
                )
        return pd.DataFrame(rows)

    def ratings(
        self,
        config: AOEuropeanEloConfig,
        season: str,
    ) -> pd.DataFrame:
        key = (repr(config), season)
        if key not in self._ratings:
            folder = DATA_ROOT / season.replace("/", "-")
            self._ratings[key] = compute_ao_first_elo_from_csv(
                folder / "teams.csv",
                folder / "country_coefficients.csv",
                folder / "domestic_context.csv",
                folder / "club_european_points.csv",
                config,
            )
        return self._ratings[key]

    def matches(self, season: str) -> pd.DataFrame:
        if season not in self._matches:
            folder = DATA_ROOT / season.replace("/", "-")
            self._matches[season] = pd.read_csv(folder / "matches.csv")
        return self._matches[season]


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    baseline = AOEuropeanEloConfig.v1_1()
    country_candidate = AOEuropeanEloConfig.experimental_country_candidate()
    evaluator = Evaluator()

    baseline_metrics = evaluator.metrics("baseline_v1_1", baseline)
    country_candidate_metrics = evaluator.metrics(
        "country_candidate_fixed",
        country_candidate,
    )
    country_configs = country_candidates(baseline)
    country_metrics = pd.concat(
        [evaluator.metrics(name, config) for name, config in country_configs],
        ignore_index=True,
    )

    folds = expanding_folds(SEASONS)
    selections: list[dict[str, object]] = []
    fold_results: list[pd.DataFrame] = []
    domestic_search_results: list[pd.DataFrame] = []
    paired_results: list[pd.DataFrame] = []

    for fold_number, (train_seasons, test_season) in enumerate(folds, start=1):
        country_name = select_candidate(country_metrics, train_seasons)
        country_config = dict(country_configs)[country_name]
        country_train_loss = training_loss(
            country_metrics,
            country_name,
            train_seasons,
        )

        domestic_configs = domestic_candidates(country_config)
        domestic_metrics = pd.concat(
            [
                evaluator.metrics(name, config, train_seasons + (test_season,))
                for name, config in domestic_configs
            ],
            ignore_index=True,
        )
        domestic_metrics["fold"] = fold_number
        domestic_search_results.append(domestic_metrics)
        combined_name = select_candidate(domestic_metrics, train_seasons)
        combined_config = dict(domestic_configs)[combined_name]
        combined_train_loss = training_loss(
            domestic_metrics,
            combined_name,
            train_seasons,
        )

        fold_frame = pd.concat(
            [
                test_rows(baseline_metrics, "baseline_v1_1", test_season, "baseline"),
                test_rows(
                    country_candidate_metrics,
                    "country_candidate_fixed",
                    test_season,
                    "country_candidate_fixed",
                ),
                test_rows(country_metrics, country_name, test_season, "country_selected"),
                test_rows(domestic_metrics, combined_name, test_season, "combined_selected"),
            ],
            ignore_index=True,
        )
        fold_frame["fold"] = fold_number
        fold_frame = fold_frame[["fold", *[column for column in fold_frame if column != "fold"]]]
        fold_results.append(fold_frame)
        paired_results.append(
            paired_loss_frame(
                evaluator,
                fold_number,
                test_season,
                baseline,
                country_candidate,
                country_config,
                combined_config,
            )
        )

        selections.append(
            {
                "fold": fold_number,
                "train_seasons": "|".join(train_seasons),
                "test_season": test_season,
                "country_candidate": country_name,
                "country_train_log_loss": country_train_loss,
                "domestic_candidate": combined_name,
                "combined_train_log_loss": combined_train_loss,
                **selected_config_fields(combined_config),
            }
        )

    selections_frame = pd.DataFrame(selections)
    fold_frame = pd.concat(fold_results, ignore_index=True)
    competition_summary = summarize_competitions(fold_frame)
    stability = summarize_stability(selections_frame)
    uncertainty = summarize_paired_uncertainty(
        pd.concat(paired_results, ignore_index=True)
    )

    selections_frame.to_csv(OUTPUT_ROOT / "fold_selections.csv", index=False)
    fold_frame.to_csv(OUTPUT_ROOT / "fold_competition_metrics.csv", index=False)
    competition_summary.to_csv(
        OUTPUT_ROOT / "competition_summary.csv",
        index=False,
    )
    stability.to_csv(OUTPUT_ROOT / "parameter_stability.csv", index=False)
    uncertainty.to_csv(OUTPUT_ROOT / "paired_uncertainty.csv", index=False)
    country_metrics.to_csv(OUTPUT_ROOT / "country_candidate_metrics.csv", index=False)
    pd.concat(domestic_search_results, ignore_index=True).to_csv(
        OUTPUT_ROOT / "domestic_candidate_metrics.csv",
        index=False,
    )
    write_report(
        selections_frame,
        fold_frame,
        competition_summary,
        stability,
        uncertainty,
    )

    print("AO European Elo nested walk-forward")
    print(f"Outer folds: {len(folds)}")
    print(f"Country candidates: {len(country_configs)}")
    print(f"Domestic candidates per fold: {len(domestic_candidates(baseline))}")
    print(f"Report: {OUTPUT_ROOT / 'backtest_report.md'}")


def expanding_folds(
    seasons: tuple[str, ...],
    min_train_seasons: int = 2,
) -> list[tuple[tuple[str, ...], str]]:
    if len(seasons) <= min_train_seasons:
        raise ValueError("At least one test season is required")
    return [
        (seasons[:index], seasons[index])
        for index in range(min_train_seasons, len(seasons))
    ]


def country_candidates(
    baseline: AOEuropeanEloConfig,
) -> list[tuple[str, AOEuropeanEloConfig]]:
    values = [("country_current", baseline)]
    values.extend(
        (
            f"country_b{benchmark:g}_g{gamma:.1f}_c{component}",
            replace(
                baseline,
                country_strength_benchmark=float(benchmark),
                gamma=gamma,
                domestic_league_component=float(component),
            ),
        )
        for benchmark in (12.5, 15, 17.5, 20)
        for gamma in (1.8, 2.0, 2.2, 2.4, 2.6)
        for component in (280, 300, 320, 340, 360, 380, 400)
    )
    return values


def domestic_candidates(
    country_config: AOEuropeanEloConfig,
) -> list[tuple[str, AOEuropeanEloConfig]]:
    values = [("domestic_current", country_config)]
    values.extend(
        (
            f"domestic_c{component}_a{alpha:.1f}",
            replace(
                country_config,
                domestic_achievement_component=float(component),
                achievement_alpha=alpha,
            ),
        )
        for component in (80, 120, 160, 200, 240, 280, 320, 360, 400)
        for alpha in (0.0, 0.1, 0.2, 0.4)
    )
    return values


def select_candidate(metrics: pd.DataFrame, train_seasons: tuple[str, ...]) -> str:
    overall = metrics.loc[
        metrics["season"].isin(train_seasons) & metrics["competition"].eq("ALL")
    ]
    expected = len(train_seasons)
    counts = overall.groupby("candidate")["season"].nunique()
    if counts.empty or not counts.eq(expected).all():
        raise ValueError("Every candidate must cover every training season")
    summary = overall.groupby("candidate", as_index=False).agg(
        train_log_loss=("log_loss", "mean"),
        train_brier=("brier", "mean"),
        season_std=("log_loss", "std"),
    )
    return str(
        summary.sort_values(
            ["train_log_loss", "train_brier", "season_std", "candidate"]
        ).iloc[0]["candidate"]
    )


def training_loss(
    metrics: pd.DataFrame,
    candidate: str,
    train_seasons: tuple[str, ...],
) -> float:
    values = metrics.loc[
        metrics["candidate"].eq(candidate)
        & metrics["season"].isin(train_seasons)
        & metrics["competition"].eq("ALL"),
        "log_loss",
    ]
    return float(values.mean())


def test_rows(
    metrics: pd.DataFrame,
    candidate: str,
    test_season: str,
    model: str,
) -> pd.DataFrame:
    result = metrics.loc[
        metrics["candidate"].eq(candidate) & metrics["season"].eq(test_season)
    ].copy()
    result["selected_candidate"] = candidate
    result["model"] = model
    return result.drop(columns="candidate")


def summarize_competitions(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, competition), frame in results.groupby(["model", "competition"]):
        matches = int(frame["matches"].sum())
        rows.append(
            {
                "model": model,
                "competition": competition,
                "matches": matches,
                "weighted_log_loss": float(
                    (frame["log_loss"] * frame["matches"]).sum() / matches
                ),
                "weighted_brier": float(
                    (frame["brier"] * frame["matches"]).sum() / matches
                ),
                "folds_improved": 0,
            }
        )
    summary = pd.DataFrame(rows)
    baseline = summary.loc[summary["model"].eq("baseline")].set_index("competition")
    for index, row in summary.iterrows():
        competition = row["competition"]
        summary.loc[index, "log_loss_delta"] = (
            row["weighted_log_loss"]
            - baseline.loc[competition, "weighted_log_loss"]
        )
        if row["model"] != "baseline":
            candidate = results.loc[
                results["model"].eq(row["model"])
                & results["competition"].eq(competition)
            ].set_index("fold")
            base = results.loc[
                results["model"].eq("baseline")
                & results["competition"].eq(competition)
            ].set_index("fold")
            summary.loc[index, "folds_improved"] = int(
                (candidate["log_loss"] < base["log_loss"]).sum()
            )
    return summary.sort_values(["competition", "model"]).reset_index(drop=True)


def summarize_stability(selections: pd.DataFrame) -> pd.DataFrame:
    fields = (
        "country_strength_benchmark",
        "gamma",
        "domestic_league_component",
        "domestic_achievement_component",
        "achievement_alpha",
    )
    rows = []
    for field in fields:
        for value, count in selections[field].value_counts(dropna=False).items():
            rows.append(
                {
                    "parameter": field,
                    "value": value,
                    "folds_selected": int(count),
                    "selection_rate": float(count / len(selections)),
                }
            )
    return pd.DataFrame(rows)


def paired_loss_frame(
    evaluator: Evaluator,
    fold: int,
    season: str,
    baseline: AOEuropeanEloConfig,
    country_candidate: AOEuropeanEloConfig,
    country: AOEuropeanEloConfig,
    combined: AOEuropeanEloConfig,
) -> pd.DataFrame:
    matches = evaluator.matches(season).reset_index(drop=True)
    result = matches[["competition"]].copy()
    result.insert(0, "fold", fold)
    result.insert(1, "season", season)
    for label, config in (
        ("baseline", baseline),
        ("country_candidate_fixed", country_candidate),
        ("country_selected", country),
        ("combined_selected", combined),
    ):
        ratings = evaluator.ratings(config, season)
        result[f"{label}_loss"] = match_log_losses(ratings, matches)
    return result


def match_log_losses(ratings: pd.DataFrame, matches: pd.DataFrame) -> pd.Series:
    home = ratings[["team_id", "ao_first_elo"]].rename(
        columns={"team_id": "home_team_id", "ao_first_elo": "home_rating"}
    )
    away = ratings[["team_id", "ao_first_elo"]].rename(
        columns={"team_id": "away_team_id", "ao_first_elo": "away_rating"}
    )
    data = matches.merge(home, on="home_team_id", how="left", validate="many_to_one")
    data = data.merge(away, on="away_team_id", how="left", validate="many_to_one")
    if data[["home_rating", "away_rating"]].isna().any().any():
        raise ValueError("Cannot compute paired loss with missing ratings")
    difference = data["home_rating"] - data["away_rating"] + HOME_ADVANTAGE
    probability = 1.0 / (1.0 + 10.0 ** (-difference / 400.0))
    probability = probability.clip(1e-12, 1 - 1e-12)
    actual = pd.Series(0.5, index=data.index)
    actual.loc[data["home_goals"] > data["away_goals"]] = 1.0
    actual.loc[data["home_goals"] < data["away_goals"]] = 0.0
    return -(
        actual * probability.map(log)
        + (1.0 - actual) * (1.0 - probability).map(log)
    )


def summarize_paired_uncertainty(
    paired: pd.DataFrame,
    bootstrap_samples: int = 5000,
) -> pd.DataFrame:
    comparisons = (
        (
            "country_candidate_fixed_vs_baseline",
            "country_candidate_fixed_loss",
            "baseline_loss",
        ),
        ("country_selected_vs_baseline", "country_selected_loss", "baseline_loss"),
        ("combined_selected_vs_baseline", "combined_selected_loss", "baseline_loss"),
        (
            "combined_selected_vs_country_selected",
            "combined_selected_loss",
            "country_selected_loss",
        ),
    )
    rng = np.random.default_rng(20260713)
    rows = []
    for competition in COMPETITIONS:
        frame = paired if competition == "ALL" else paired.loc[paired["competition"].eq(competition)]
        for comparison, candidate_column, baseline_column in comparisons:
            differences = (
                frame[candidate_column] - frame[baseline_column]
            ).to_numpy(dtype=float)
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
                    "mean_log_loss_delta": float(differences.mean()),
                    "ci_95_lower": float(lower),
                    "ci_95_upper": float(upper),
                    "directionally_reliable": bool(upper < 0),
                }
            )
    return pd.DataFrame(rows)


def selected_config_fields(config: AOEuropeanEloConfig) -> dict[str, float]:
    values = asdict(config)
    return {
        field: float(values[field])
        for field in (
            "country_strength_benchmark",
            "gamma",
            "domestic_league_component",
            "domestic_achievement_component",
            "achievement_alpha",
        )
    }


def write_report(
    selections: pd.DataFrame,
    fold_results: pd.DataFrame,
    competition_summary: pd.DataFrame,
    stability: pd.DataFrame,
    uncertainty: pd.DataFrame,
) -> None:
    overall = competition_summary.loc[competition_summary["competition"].eq("ALL")]
    lines = [
        "# AO European Elo Nested Walk-Forward Backtest",
        "",
        "Each outer fold selects parameters using past seasons only and evaluates the",
        "selected configuration on the immediately following unseen season.",
        f"Home advantage remains fixed at `{HOME_ADVANTAGE:.0f}` Elo.",
        "Lower log loss and Brier are better.",
        "",
        "## Fold selections",
        "",
        "| Fold | Train | Test | Country candidate | Domestic candidate |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for row in selections.itertuples(index=False):
        lines.append(
            f"| {row.fold} | {row.train_seasons.replace('|', ', ')} | "
            f"{row.test_season} | {row.country_candidate} | {row.domestic_candidate} |"
        )
    lines.extend(
        [
            "",
            "## Pooled unseen-season results",
            "",
            "| Model | Matches | Log loss | Delta vs baseline | Brier | Folds improved |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in overall.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.matches} | {row.weighted_log_loss:.6f} | "
            f"{row.log_loss_delta:+.6f} | {row.weighted_brier:.6f} | "
            f"{row.folds_improved}/3 |"
        )
    lines.extend(
        [
            "",
            "## Competition split",
            "",
            "| Competition | Model | Matches | Log loss | Delta | Folds improved |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in competition_summary.loc[
        ~competition_summary["competition"].eq("ALL")
    ].itertuples(index=False):
        lines.append(
            f"| {row.competition} | {row.model} | {row.matches} | "
            f"{row.weighted_log_loss:.6f} | {row.log_loss_delta:+.6f} | "
            f"{row.folds_improved}/3 |"
        )
    lines.extend(
        [
            "",
            "## Parameter stability",
            "",
            "| Parameter | Value | Folds selected |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in stability.itertuples(index=False):
        lines.append(
            f"| {row.parameter} | {row.value:g} | {row.folds_selected}/3 |"
        )

    lines.extend(
        [
            "",
            "## Paired uncertainty",
            "",
            "Match-level paired bootstrap intervals condition on the evaluated configurations;",
            "they do not include parameter-selection uncertainty.",
            "",
            "| Competition | Comparison | Mean delta | 95% interval | Reliable direction |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for row in uncertainty.itertuples(index=False):
        lines.append(
            f"| {row.competition} | {row.comparison} | "
            f"{row.mean_log_loss_delta:+.6f} | "
            f"[{row.ci_95_lower:+.6f}, {row.ci_95_upper:+.6f}] | "
            f"{'YES' if row.directionally_reliable else 'NO'} |"
        )

    combined = overall.loc[overall["model"].eq("combined_selected")].iloc[0]
    fixed = overall.loc[overall["model"].eq("country_candidate_fixed")].iloc[0]
    tournament = competition_summary.loc[
        competition_summary["model"].eq("combined_selected")
        & ~competition_summary["competition"].eq("ALL")
    ]
    all_tournaments_improve = bool((tournament["log_loss_delta"] < 0).all())
    all_folds_improve = int(combined["folds_improved"]) == len(selections)
    country_mode = {
        field: selections[field].mode().iloc[0]
        for field in (
            "country_strength_benchmark",
            "gamma",
            "domestic_league_component",
        )
    }
    domestic_increment = uncertainty.loc[
        uncertainty["competition"].eq("ALL")
        & uncertainty["comparison"].eq(
            "combined_selected_vs_country_selected"
        )
    ].iloc[0]
    lines.extend(
        [
            "",
            "## Decision rule",
            "",
            f"- Combined model improves all unseen folds: `{'PASS' if all_folds_improve else 'FAIL'}`.",
            f"- Combined model improves pooled UCL, UEL, and UECL: `{'PASS' if all_tournaments_improve else 'FAIL'}`.",
            f"- Fixed country candidate improves `{int(fixed.folds_improved)}/3` unseen folds with "
            f"pooled delta `{fixed.log_loss_delta:+.6f}`.",
            f"- Stable country center: benchmark `{country_mode['country_strength_benchmark']:g}`, "
            f"gamma `{country_mode['gamma']:g}`, league component `{country_mode['domestic_league_component']:g}`.",
            f"- Domestic achievement incremental delta: `{domestic_increment.mean_log_loss_delta:+.6f}` "
            f"with 95% interval `[{domestic_increment.ci_95_lower:+.6f}, "
            f"{domestic_increment.ci_95_upper:+.6f}]`.",
            "- Reject production promotion despite aggregate improvement: the zero-exposure",
            "  segment can receive implausibly dominant ratings from the enlarged country prior.",
            "- Keep the v1.1 country and domestic defaults until exposure-segment guardrails pass.",
        ]
    )
    (OUTPUT_ROOT / "backtest_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
