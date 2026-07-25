from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from itertools import product
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ao_elo.config import (  # noqa: E402
    AO_MODEL_V2_VERSION,
    AOEuropeanEloConfig,
    V2_RATING_MULTIPLIER,
)
from ao_elo.pipeline import compute_ao_first_elo  # noqa: E402
from scripts.run_ranking_first_calibration import (  # noqa: E402
    pairwise_ranking_accuracy,
    ranking_metrics,
)


DATA_ROOT = ROOT / "data" / "backtest_stage_b_2018_2026"
PILOT_ROOT = ROOT / "data" / "real_pilot_10_teams"
EXTERNAL_PATH = (
    ROOT / "output" / "external_elo_benchmark_2018_2026" / "paired_predictions.csv"
)
OUTPUT_ROOT = ROOT / "output" / "v2_ranking_calibration_2018_2026"
COMPETITIONS = ("UCL", "UEL", "UECL")
COUNTRY_TAIL_BETAS = (0.0, 0.25, 0.50, 0.75, 1.0)
EUROPEAN_TAIL_BETAS = (0.0, 0.25, 0.50, 0.75, 1.0)
EXPOSURE_TAIL_BETAS = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
BASELINE_NAME = "c0_e0_x0"
RANK_TOLERANCE = 1e-12
ZERO_EXPOSURE_MAE_TOLERANCE = 0.02
V2_REFERENCE_SCALE = 225.0 * V2_RATING_MULTIPLIER
V2_REFERENCE_HOME_ADVANTAGE = 40.0 * V2_RATING_MULTIPLIER


class V2RankingEvaluator:
    def __init__(self, data_root: Path, seasons: tuple[str, ...]) -> None:
        self.seasons = seasons
        self.inputs: dict[str, dict[str, pd.DataFrame]] = {}
        self.matches: dict[str, pd.DataFrame] = {}
        self.performance: dict[tuple[str, str], pd.DataFrame] = {}
        for season in seasons:
            folder = data_root / season.replace("/", "-")
            self.inputs[season] = {
                "teams": pd.read_csv(folder / "teams.csv"),
                "country_coefficients": pd.read_csv(
                    folder / "country_coefficients.csv"
                ),
                "domestic_context": pd.read_csv(folder / "domestic_context.csv"),
                "club_european_points": pd.read_csv(
                    folder / "club_european_points.csv"
                ),
            }
            self.matches[season] = pd.read_csv(folder / "matches.csv")

    def evaluate(
        self,
        candidate: str,
        config: AOEuropeanEloConfig,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        ranking_rows: list[dict[str, object]] = []
        rating_frames: list[pd.DataFrame] = []
        for season in self.seasons:
            ratings = compute_ao_first_elo(config=config, **self.inputs[season])
            selected = ratings[
                [
                    "season",
                    "team_id",
                    "team_name",
                    "ao_first_elo",
                    "european_exposure",
                    "saturation_count",
                    "country_strength_saturated",
                    "achievement_saturated",
                    "european_history_saturated",
                    "exposure_saturated",
                ]
            ].copy()
            selected.insert(0, "candidate", candidate)
            rating_frames.append(selected)
            for competition in COMPETITIONS:
                table = self.ranking_table(ratings, season, competition)
                ranking_rows.append(
                    {
                        "candidate": candidate,
                        "season": season,
                        "competition": competition,
                        **ranking_metrics(table),
                    }
                )
        return pd.DataFrame(ranking_rows), pd.concat(rating_frames, ignore_index=True)

    def ranking_table(
        self,
        ratings: pd.DataFrame,
        season: str,
        competition: str,
    ) -> pd.DataFrame:
        performance = self.team_performance(season, competition)
        table = performance.merge(
            ratings[["team_id", "ao_first_elo", "european_exposure"]],
            on="team_id",
            how="inner",
            validate="one_to_one",
        )
        table["predicted_percentile"] = table["ao_first_elo"].rank(
            method="average", pct=True
        )
        table["actual_percentile"] = table["actual_score_rate"].rank(
            method="average", pct=True
        )
        table["percentile_error"] = (
            table["predicted_percentile"] - table["actual_percentile"]
        )
        return table

    def team_performance(self, season: str, competition: str) -> pd.DataFrame:
        key = (season, competition)
        if key not in self.performance:
            matches = self.matches[season]
            matches = matches.loc[matches["competition"].eq(competition)].copy()
            actual = pd.Series(0.5, index=matches.index)
            actual.loc[matches["home_goals"] > matches["away_goals"]] = 1.0
            actual.loc[matches["home_goals"] < matches["away_goals"]] = 0.0
            home = pd.DataFrame(
                {"team_id": matches["home_team_id"], "score": actual}
            )
            away = pd.DataFrame(
                {"team_id": matches["away_team_id"], "score": 1.0 - actual}
            )
            self.performance[key] = (
                pd.concat([home, away], ignore_index=True)
                .groupby("team_id", as_index=False)
                .agg(actual_score_rate=("score", "mean"), matches=("score", "size"))
            )
        return self.performance[key]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate AO European Elo v2 upper tails with ranking-first gates"
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--pilot-root", type=Path, default=PILOT_ROOT)
    parser.add_argument("--external-path", type=Path, default=EXTERNAL_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    pilot_root = args.pilot_root.resolve()
    output_root = args.output_root.resolve()
    seasons = discover_seasons(data_root)
    folds = expanding_folds(seasons)
    if len(folds) != 6:
        raise ValueError(f"Expected six outer folds, found {len(folds)}")
    output_root.mkdir(parents=True, exist_ok=True)

    candidates = tail_candidates()
    evaluator = V2RankingEvaluator(data_root, seasons)
    ranking_frames: list[pd.DataFrame] = []
    rating_frames: list[pd.DataFrame] = []
    pilot_rows: list[dict[str, object]] = []
    distribution_rows: list[dict[str, object]] = []

    for index, (name, config) in enumerate(candidates, start=1):
        ranking, ratings = evaluator.evaluate(name, config)
        ranking_frames.append(ranking)
        rating_frames.append(ratings)
        pilot_rows.append(evaluate_pilot_guardrails(name, config, pilot_root))
        distribution_rows.append(distribution_guardrails(name, ratings))
        if index % 10 == 0:
            print(f"Static tail candidates evaluated: {index}/{len(candidates)}")

    ranking = pd.concat(ranking_frames, ignore_index=True)
    ratings = pd.concat(rating_frames, ignore_index=True)
    pilot = pd.DataFrame(pilot_rows)
    distribution = pd.DataFrame(distribution_rows)
    static_guardrails = pilot.merge(distribution, on="candidate", validate="one_to_one")

    selections, fold_results = run_nested_walk_forward(
        ranking,
        static_guardrails,
        folds,
    )
    competition_summary = summarize_competitions(fold_results)
    outer_gate = evaluate_outer_gate(selections, fold_results, competition_summary)

    external = pd.read_csv(args.external_path.resolve())
    external_predictions, external_summary = evaluate_external_benchmark(
        external,
        ratings,
    )
    full_summary = summarize_training(
        ranking,
        static_guardrails,
        seasons,
    ).merge(external_summary, on="candidate", validate="one_to_one")
    baseline_external = external_summary.loc[
        external_summary["candidate"].eq(BASELINE_NAME)
    ].iloc[0]
    full_summary["external_gate_pass"] = (
        full_summary["ucl_brier_delta"] <= RANK_TOLERANCE
    ) & (
        full_summary["clubelo_rank_correlation"]
        >= baseline_external["clubelo_rank_correlation"] - RANK_TOLERANCE
    ) & (
        (full_summary["candidate"].eq(BASELINE_NAME))
        | (
            full_summary["elite_mean_rating_gap"]
            > baseline_external["elite_mean_rating_gap"] + RANK_TOLERANCE
        )
    )
    full_summary["final_eligible"] = (
        full_summary["eligible"] & full_summary["external_gate_pass"]
    )
    evaluated_candidate = select_candidate(full_summary, "final_eligible")
    uncertainty = clustered_candidate_vs_baseline_ci(
        external_predictions,
        evaluated_candidate,
    )
    decision = make_promotion_decision(
        outer_gate,
        full_summary,
        evaluated_candidate,
        uncertainty,
    )
    selected_candidate = (
        evaluated_candidate if decision == "PROMOTE" else BASELINE_NAME
    )
    selected_config = dict(candidates)[selected_candidate]

    cap_errors, cap_correlation = saturation_error_analysis(
        external_predictions,
        ratings,
    )
    selected_ratings = ratings.loc[ratings["candidate"].eq(selected_candidate)]
    selected_real_pilot = compute_pilot(selected_config, pilot_root)

    ranking.to_csv(output_root / "all_candidate_ranking_metrics.csv", index=False)
    static_guardrails.to_csv(
        output_root / "candidate_static_guardrails.csv", index=False
    )
    selections.to_csv(output_root / "fold_selections.csv", index=False)
    fold_results.to_csv(output_root / "unseen_fold_results.csv", index=False)
    competition_summary.to_csv(
        output_root / "unseen_competition_summary.csv", index=False
    )
    full_summary.to_csv(output_root / "full_candidate_summary.csv", index=False)
    external_summary.to_csv(output_root / "external_guardrails.csv", index=False)
    uncertainty.to_csv(output_root / "external_clustered_uncertainty.csv", index=False)
    cap_errors.to_csv(output_root / "ucl_saturation_error_groups.csv", index=False)
    cap_correlation.to_csv(
        output_root / "ucl_saturation_error_correlation.csv", index=False
    )
    selected_ratings.to_csv(output_root / "selected_historical_ratings.csv", index=False)
    selected_real_pilot.to_csv(output_root / "selected_real_pilot.csv", index=False)
    write_manifest(
        output_root / "selected_model.json",
        decision,
        evaluated_candidate,
        selected_candidate,
        selected_config,
        outer_gate,
        uncertainty,
    )
    write_report(
        output_root / "calibration_report.md",
        seasons,
        selections,
        competition_summary,
        full_summary,
        evaluated_candidate,
        selected_candidate,
        decision,
        outer_gate,
        uncertainty,
        cap_correlation,
    )

    print("AO European Elo v2 ranking-first calibration")
    print(f"Candidates: {len(candidates)}")
    print(f"Outer folds: {len(folds)}")
    print(f"Evaluated candidate: {evaluated_candidate}")
    print(f"Decision: {decision}")
    print(f"Active tail config: {selected_candidate}")
    print(f"Report: {output_root / 'calibration_report.md'}")


def discover_seasons(data_root: Path) -> tuple[str, ...]:
    seasons = tuple(
        folder.name.replace("-", "/")
        for folder in sorted(data_root.glob("20??-??"))
    )
    if len(seasons) < 3:
        raise ValueError("V2 ranking calibration needs at least three seasons")
    return seasons


def expanding_folds(
    seasons: tuple[str, ...],
    min_train_seasons: int = 2,
) -> tuple[tuple[tuple[str, ...], str], ...]:
    return tuple(
        (seasons[:index], seasons[index])
        for index in range(min_train_seasons, len(seasons))
    )


def tail_candidates() -> list[tuple[str, AOEuropeanEloConfig]]:
    values = []
    for country_beta, european_beta, exposure_beta in product(
        COUNTRY_TAIL_BETAS,
        EUROPEAN_TAIL_BETAS,
        EXPOSURE_TAIL_BETAS,
    ):
        name = candidate_name(country_beta, european_beta, exposure_beta)
        values.append(
            (
                name,
                AOEuropeanEloConfig.v2(
                    country_tail_beta=country_beta,
                    european_tail_beta=european_beta,
                    exposure_tail_beta=exposure_beta,
                ),
            )
        )
    return values


def candidate_name(
    country_beta: float,
    european_beta: float,
    exposure_beta: float,
) -> str:
    def token(value: float) -> str:
        return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", "p")

    return f"c{token(country_beta)}_e{token(european_beta)}_x{token(exposure_beta)}"


def compute_pilot(config: AOEuropeanEloConfig, pilot_root: Path) -> pd.DataFrame:
    return compute_ao_first_elo(
        teams=pd.read_csv(pilot_root / "teams.csv"),
        country_coefficients=pd.read_csv(pilot_root / "country_coefficients.csv"),
        domestic_context=pd.read_csv(pilot_root / "domestic_context.csv"),
        club_european_points=pd.read_csv(pilot_root / "club_european_points.csv"),
        config=config,
    )


def evaluate_pilot_guardrails(
    candidate: str,
    config: AOEuropeanEloConfig,
    pilot_root: Path,
) -> dict[str, object]:
    ratings = compute_pilot(config, pilot_root)
    values = ratings.set_index("team_name")["ao_first_elo"]
    guardrails = pd.read_csv(pilot_root / "ranking_guardrails.csv")
    violations = [
        f"{row.higher_team}<={row.lower_team}"
        for row in guardrails.itertuples(index=False)
        if values[row.higher_team] <= values[row.lower_team]
    ]
    return {
        "candidate": candidate,
        "pilot_guardrails_pass": not violations,
        "pilot_guardrail_violations": ";".join(violations),
        "pilot_top_team": str(values.idxmax()),
        "pilot_top_rating": float(values.max()),
    }


def distribution_guardrails(
    candidate: str,
    ratings: pd.DataFrame,
) -> dict[str, object]:
    values = ratings["ao_first_elo"].astype(float)
    top_count = max(1, ceil(len(values) * 0.01))
    top = values.nlargest(top_count)
    band_share = float(values.between(500.0, 2000.0).mean())
    median = float(values.median())
    p90 = float(values.quantile(0.90))
    exact_boundary_count = int(values.isin((500.0, 2000.0)).sum())
    duplicate_top_count = int(top.duplicated(keep=False).sum())
    passed = (
        band_share >= 0.95
        and 900.0 <= median <= 1200.0
        and 1600.0 <= p90 <= 1900.0
        and exact_boundary_count == 0
        and duplicate_top_count == 0
    )
    return {
        "candidate": candidate,
        "ratings": len(values),
        "reference_band_share": band_share,
        "median_rating": median,
        "p90_rating": p90,
        "minimum_rating": float(values.min()),
        "maximum_rating": float(values.max()),
        "exact_500_or_2000_count": exact_boundary_count,
        "elite_duplicate_rating_count": duplicate_top_count,
        "distribution_guardrails_pass": passed,
    }


def summarize_training(
    ranking: pd.DataFrame,
    static_guardrails: pd.DataFrame,
    seasons: tuple[str, ...],
) -> pd.DataFrame:
    train = ranking.loc[ranking["season"].isin(seasons)]
    summary = train.groupby("candidate", as_index=False).agg(
        ranking_score=("ranking_score", "mean"),
        pairwise_accuracy=("pairwise_accuracy", "mean"),
        top_quartile_precision=("top_quartile_precision", "mean"),
        rank_percentile_mae=("rank_percentile_mae", "mean"),
        zero_exposure_rank_mae=("zero_exposure_rank_mae", "mean"),
        zero_exposure_extreme_overrank_count=(
            "zero_exposure_extreme_overrank_count",
            "sum",
        ),
    )
    summary = summary.merge(static_guardrails, on="candidate", validate="one_to_one")
    baseline = summary.loc[summary["candidate"].eq(BASELINE_NAME)].iloc[0]
    summary["historical_zero_guardrail_pass"] = (
        summary["zero_exposure_rank_mae"]
        <= baseline["zero_exposure_rank_mae"] + ZERO_EXPOSURE_MAE_TOLERANCE
    ) & (
        summary["zero_exposure_extreme_overrank_count"]
        <= baseline["zero_exposure_extreme_overrank_count"]
    )

    competition = train.groupby(["candidate", "competition"], as_index=False).agg(
        ranking_score=("ranking_score", "mean"),
        pairwise_accuracy=("pairwise_accuracy", "mean"),
    )
    baseline_competition = competition.loc[
        competition["candidate"].eq(BASELINE_NAME)
    ].set_index("competition")
    competition["ranking_not_worse"] = competition.apply(
        lambda row: metric_not_worse(
            row["ranking_score"],
            baseline_competition.loc[row["competition"], "ranking_score"],
        ),
        axis=1,
    )
    competition["pairwise_not_worse"] = competition.apply(
        lambda row: metric_not_worse(
            row["pairwise_accuracy"],
            baseline_competition.loc[row["competition"], "pairwise_accuracy"],
        ),
        axis=1,
    )
    competition_gate = (
        competition.groupby("candidate", as_index=False)[
            ["ranking_not_worse", "pairwise_not_worse"]
        ]
        .all()
        .rename(
            columns={
                "ranking_not_worse": "all_competition_spearman_not_worse",
                "pairwise_not_worse": "all_competition_pairwise_not_worse",
            }
        )
    )
    summary = summary.merge(competition_gate, on="candidate", validate="one_to_one")
    summary["eligible"] = (
        summary["pilot_guardrails_pass"]
        & summary["distribution_guardrails_pass"]
        & summary["historical_zero_guardrail_pass"]
        & summary["all_competition_spearman_not_worse"]
        & summary["all_competition_pairwise_not_worse"]
    )
    return summary.sort_values(
        [
            "eligible",
            "ranking_score",
            "pairwise_accuracy",
            "top_quartile_precision",
            "rank_percentile_mae",
            "candidate",
        ],
        ascending=[False, False, False, False, True, True],
    ).reset_index(drop=True)


def metric_not_worse(value: float, baseline: float) -> bool:
    """Treat an unavailable historical segment as no evidence, not a veto."""
    if pd.isna(value) or pd.isna(baseline):
        return True
    return bool(value >= baseline - RANK_TOLERANCE)


def select_candidate(summary: pd.DataFrame, eligibility_column: str) -> str:
    eligible = summary.loc[summary[eligibility_column]]
    return str(eligible.iloc[0]["candidate"]) if not eligible.empty else BASELINE_NAME


def run_nested_walk_forward(
    ranking: pd.DataFrame,
    static_guardrails: pd.DataFrame,
    folds: tuple[tuple[tuple[str, ...], str], ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selection_rows: list[dict[str, object]] = []
    result_frames: list[pd.DataFrame] = []
    for fold, (train_seasons, test_season) in enumerate(folds, start=1):
        summary = summarize_training(ranking, static_guardrails, train_seasons)
        selected = select_candidate(summary, "eligible")
        row = summary.loc[summary["candidate"].eq(selected)].iloc[0]
        selection_rows.append(
            {
                "fold": fold,
                "train_seasons": "|".join(train_seasons),
                "test_season": test_season,
                "selected_candidate": selected,
                "train_spearman": row["ranking_score"],
                "train_pairwise": row["pairwise_accuracy"],
                "train_eligible": row["eligible"],
            }
        )
        result_frames.append(
            fold_comparison(ranking, fold, test_season, selected)
        )
    return pd.DataFrame(selection_rows), pd.concat(result_frames, ignore_index=True)


def fold_comparison(
    ranking: pd.DataFrame,
    fold: int,
    test_season: str,
    selected: str,
) -> pd.DataFrame:
    frames = []
    for model, candidate in (("baseline", BASELINE_NAME), ("selected", selected)):
        frame = ranking.loc[
            ranking["candidate"].eq(candidate)
            & ranking["season"].eq(test_season)
        ].copy()
        frame.insert(0, "fold", fold)
        frame.insert(1, "model", model)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def summarize_competitions(fold_results: pd.DataFrame) -> pd.DataFrame:
    summary = fold_results.groupby(["model", "competition"], as_index=False).agg(
        ranking_score=("ranking_score", "mean"),
        pairwise_accuracy=("pairwise_accuracy", "mean"),
        top_quartile_precision=("top_quartile_precision", "mean"),
        rank_percentile_mae=("rank_percentile_mae", "mean"),
    )
    baseline = summary.loc[summary["model"].eq("baseline")].set_index("competition")
    summary["spearman_delta"] = summary.apply(
        lambda row: row["ranking_score"]
        - baseline.loc[row["competition"], "ranking_score"],
        axis=1,
    )
    summary["pairwise_delta"] = summary.apply(
        lambda row: row["pairwise_accuracy"]
        - baseline.loc[row["competition"], "pairwise_accuracy"],
        axis=1,
    )
    return summary.sort_values(["competition", "model"]).reset_index(drop=True)


def evaluate_outer_gate(
    selections: pd.DataFrame,
    fold_results: pd.DataFrame,
    competition_summary: pd.DataFrame,
) -> dict[str, object]:
    fold_summary = fold_results.groupby(["fold", "model"], as_index=False).agg(
        ranking_score=("ranking_score", "mean"),
        pairwise_accuracy=("pairwise_accuracy", "mean"),
    )
    spearman = fold_summary.pivot(index="fold", columns="model", values="ranking_score")
    pairwise = fold_summary.pivot(
        index="fold", columns="model", values="pairwise_accuracy"
    )
    no_fold_regression = bool(
        (spearman["selected"] >= spearman["baseline"] - RANK_TOLERANCE).all()
        and (pairwise["selected"] >= pairwise["baseline"] - RANK_TOLERANCE).all()
    )
    folds_improved = int(
        (
            (spearman["selected"] > spearman["baseline"] + RANK_TOLERANCE)
            & (pairwise["selected"] > pairwise["baseline"] + RANK_TOLERANCE)
        ).sum()
    )
    selected_competitions = competition_summary.loc[
        competition_summary["model"].eq("selected")
    ]
    no_competition_regression = bool(
        (selected_competitions["spearman_delta"] >= -RANK_TOLERANCE).all()
        and (selected_competitions["pairwise_delta"] >= -RANK_TOLERANCE).all()
    )
    return {
        "outer_folds": len(selections),
        "no_unseen_fold_regression": no_fold_regression,
        "folds_improved_both_metrics": folds_improved,
        "at_least_four_folds_improved": folds_improved >= 4,
        "no_competition_regression": no_competition_regression,
    }


def evaluate_external_benchmark(
    external: pd.DataFrame,
    ratings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "match_id",
        "season",
        "competition",
        "home_team_id",
        "away_team_id",
        "actual_home_score",
        "is_neutral",
        "tie_id",
        "clubelo_home_elo",
        "clubelo_away_elo",
    }
    missing = sorted(required - set(external.columns))
    if missing:
        raise ValueError(f"External benchmark missing columns: {missing}")
    frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    clubelo = clubelo_team_seasons(external)
    for candidate, candidate_ratings in ratings.groupby("candidate", sort=False):
        home = candidate_ratings[
            ["season", "team_id", "ao_first_elo", "saturation_count"]
        ].rename(
            columns={
                "team_id": "home_team_id",
                "ao_first_elo": "home_rating",
                "saturation_count": "home_saturation_count",
            }
        )
        away = candidate_ratings[
            ["season", "team_id", "ao_first_elo", "saturation_count"]
        ].rename(
            columns={
                "team_id": "away_team_id",
                "ao_first_elo": "away_rating",
                "saturation_count": "away_saturation_count",
            }
        )
        predicted = external.merge(
            home, on=["season", "home_team_id"], how="inner", validate="many_to_one"
        ).merge(
            away, on=["season", "away_team_id"], how="inner", validate="many_to_one"
        )
        if len(predicted) != len(external):
            raise ValueError(f"Candidate {candidate} lacks an external benchmark rating")
        advantage = np.where(
            predicted["is_neutral"].astype(bool),
            0.0,
            V2_REFERENCE_HOME_ADVANTAGE,
        )
        difference = predicted["home_rating"] - predicted["away_rating"] + advantage
        predicted["expected_home_score"] = 1.0 / (
            1.0 + 10.0 ** (-difference / V2_REFERENCE_SCALE)
        )
        predicted["brier_loss"] = (
            predicted["expected_home_score"] - predicted["actual_home_score"]
        ) ** 2
        predicted.insert(0, "candidate", candidate)
        frames.append(
            predicted[
                [
                    "candidate",
                    "match_id",
                    "season",
                    "competition",
                    "tie_id",
                    "actual_home_score",
                    "home_rating",
                    "away_rating",
                    "home_saturation_count",
                    "away_saturation_count",
                    "expected_home_score",
                    "brier_loss",
                ]
            ]
        )

        ucl = predicted.loc[predicted["competition"].eq("UCL")]
        alignment = clubelo.merge(
            candidate_ratings[["season", "team_id", "ao_first_elo"]],
            on=["season", "team_id"],
            how="inner",
            validate="one_to_one",
        )
        correlation = alignment["ao_first_elo"].corr(
            alignment["clubelo_elo"], method="spearman"
        )
        elite = ucl.loc[
            (ucl["home_saturation_count"] >= 2)
            & (ucl["away_saturation_count"] >= 2)
        ]
        summary_rows.append(
            {
                "candidate": candidate,
                "ucl_matches": len(ucl),
                "ucl_brier": float(ucl["brier_loss"].mean()),
                "clubelo_team_seasons": len(alignment),
                "clubelo_rank_correlation": float(correlation),
                "elite_cap_matches": len(elite),
                "elite_mean_rating_gap": float(
                    (elite["home_rating"] - elite["away_rating"]).abs().mean()
                ),
            }
        )
    predictions = pd.concat(frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    baseline = summary.loc[summary["candidate"].eq(BASELINE_NAME)].iloc[0]
    summary["ucl_brier_delta"] = summary["ucl_brier"] - baseline["ucl_brier"]
    summary["clubelo_rank_correlation_delta"] = (
        summary["clubelo_rank_correlation"] - baseline["clubelo_rank_correlation"]
    )
    summary["elite_mean_rating_gap_delta"] = (
        summary["elite_mean_rating_gap"] - baseline["elite_mean_rating_gap"]
    )
    return predictions, summary


def clubelo_team_seasons(external: pd.DataFrame) -> pd.DataFrame:
    ucl = external.loc[external["competition"].eq("UCL")]
    home = ucl[["season", "home_team_id", "clubelo_home_elo"]].rename(
        columns={"home_team_id": "team_id", "clubelo_home_elo": "clubelo_elo"}
    )
    away = ucl[["season", "away_team_id", "clubelo_away_elo"]].rename(
        columns={"away_team_id": "team_id", "clubelo_away_elo": "clubelo_elo"}
    )
    return (
        pd.concat([home, away], ignore_index=True)
        .groupby(["season", "team_id"], as_index=False)
        .agg(clubelo_elo=("clubelo_elo", "median"))
    )


def clustered_candidate_vs_baseline_ci(
    predictions: pd.DataFrame,
    candidate: str,
    *,
    bootstrap_samples: int = 4000,
    seed: int = 20260717,
) -> pd.DataFrame:
    selected = predictions.loc[
        predictions["candidate"].eq(candidate)
        & predictions["competition"].eq("UCL")
    ].copy()
    baseline = predictions.loc[
        predictions["candidate"].eq(BASELINE_NAME)
        & predictions["competition"].eq("UCL"),
        ["match_id", "brier_loss"],
    ].rename(columns={"brier_loss": "baseline_brier_loss"})
    paired = selected.merge(baseline, on="match_id", validate="one_to_one")
    paired["difference"] = paired["brier_loss"] - paired["baseline_brier_loss"]
    paired["cluster"] = paired["season"].astype(str) + "|" + paired["tie_id"].fillna(
        paired["match_id"]
    ).astype(str)
    clusters = paired.groupby("cluster")["difference"].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    cluster_indices = rng.integers(
        0,
        len(clusters),
        size=(bootstrap_samples, len(clusters)),
    )
    sums = clusters["sum"].to_numpy()[cluster_indices].sum(axis=1)
    counts = clusters["count"].to_numpy()[cluster_indices].sum(axis=1)
    means = sums / counts
    lower, upper = np.quantile(means, (0.025, 0.975))
    return pd.DataFrame(
        [
            {
                "candidate": candidate,
                "baseline": BASELINE_NAME,
                "matches": len(paired),
                "clusters": len(clusters),
                "mean_brier_difference": float(paired["difference"].mean()),
                "ci_95_lower": float(lower),
                "ci_95_upper": float(upper),
                "reliable_improvement": bool(upper < 0.0),
                "reliable_harm": bool(lower > 0.0),
            }
        ]
    )


def saturation_error_analysis(
    predictions: pd.DataFrame,
    ratings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = predictions.loc[
        predictions["candidate"].eq(BASELINE_NAME)
        & predictions["competition"].eq("UCL")
    ].copy()
    baseline["mean_saturation_count"] = (
        baseline["home_saturation_count"] + baseline["away_saturation_count"]
    ) / 2.0
    baseline["max_saturation_count"] = baseline[
        ["home_saturation_count", "away_saturation_count"]
    ].max(axis=1)
    groups = baseline.groupby("max_saturation_count", as_index=False).agg(
        matches=("match_id", "size"),
        mean_brier=("brier_loss", "mean"),
        mean_absolute_probability_error=(
            "expected_home_score",
            lambda values: float(
                np.abs(
                    values.to_numpy()
                    - baseline.loc[values.index, "actual_home_score"].to_numpy()
                ).mean()
            ),
        ),
        mean_rating_gap=(
            "home_rating",
            lambda values: float(
                np.abs(
                    values.to_numpy()
                    - baseline.loc[values.index, "away_rating"].to_numpy()
                ).mean()
            ),
        ),
    )
    correlation = pd.DataFrame(
        [
            {
                "matches": len(baseline),
                "pearson_saturation_vs_brier": baseline[
                    "mean_saturation_count"
                ].corr(baseline["brier_loss"], method="pearson"),
                "spearman_saturation_vs_brier": baseline[
                    "mean_saturation_count"
                ].corr(baseline["brier_loss"], method="spearman"),
            }
        ]
    )
    return groups, correlation


def make_promotion_decision(
    outer_gate: dict[str, object],
    full_summary: pd.DataFrame,
    evaluated_candidate: str,
    uncertainty: pd.DataFrame,
) -> str:
    row = full_summary.loc[
        full_summary["candidate"].eq(evaluated_candidate)
    ].iloc[0]
    uncertainty_row = uncertainty.iloc[0]
    passes = (
        evaluated_candidate != BASELINE_NAME
        and bool(outer_gate["no_unseen_fold_regression"])
        and bool(outer_gate["at_least_four_folds_improved"])
        and bool(outer_gate["no_competition_regression"])
        and bool(row["final_eligible"])
        and not bool(uncertainty_row["reliable_harm"])
    )
    return "PROMOTE" if passes else "NO_PROMOTION"


def write_manifest(
    path: Path,
    decision: str,
    evaluated_candidate: str,
    selected_candidate: str,
    config: AOEuropeanEloConfig,
    outer_gate: dict[str, object],
    uncertainty: pd.DataFrame,
) -> None:
    payload = {
        "model_version": AO_MODEL_V2_VERSION,
        "decision": decision,
        "evaluated_candidate": evaluated_candidate,
        "selected_candidate": selected_candidate,
        "rating_multiplier": V2_RATING_MULTIPLIER,
        "config": asdict(config),
        "outer_gate": outer_gate,
        "external_uncertainty": uncertainty.iloc[0].to_dict(),
    }
    path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")


def json_default(value: object) -> object:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_report(
    path: Path,
    seasons: tuple[str, ...],
    selections: pd.DataFrame,
    competition_summary: pd.DataFrame,
    full_summary: pd.DataFrame,
    evaluated_candidate: str,
    selected_candidate: str,
    decision: str,
    outer_gate: dict[str, object],
    uncertainty: pd.DataFrame,
    cap_correlation: pd.DataFrame,
) -> None:
    evaluated = full_summary.loc[
        full_summary["candidate"].eq(evaluated_candidate)
    ].iloc[0]
    baseline = full_summary.loc[full_summary["candidate"].eq(BASELINE_NAME)].iloc[0]
    uncertainty_row = uncertainty.iloc[0]
    lines = [
        "# AO European Elo v2 Ranking-First Calibration",
        "",
        f"Development seasons: `{seasons[0]}` through `{seasons[-1]}`.",
        f"Candidates: `{len(tail_candidates())}`. Outer folds: `{len(selections)}`.",
        "The 500-2000 range is a reference band, not a clipping boundary.",
        "",
        "## Decision",
        "",
        f"- Result: `{decision}`",
        f"- Evaluated candidate: `{evaluated_candidate}`",
        f"- Active candidate: `{selected_candidate}`",
        f"- No unseen fold regression: `{outer_gate['no_unseen_fold_regression']}`",
        f"- Folds improving Spearman and pairwise: "
        f"`{outer_gate['folds_improved_both_metrics']}/{outer_gate['outer_folds']}`",
        f"- No competition regression: `{outer_gate['no_competition_regression']}`",
        "",
        "## Distribution",
        "",
        f"- Reference-band share: `{evaluated.reference_band_share:.3%}`",
        f"- Median / p90: `{evaluated.median_rating:.2f}` / `{evaluated.p90_rating:.2f}`",
        f"- Min / max: `{evaluated.minimum_rating:.2f}` / `{evaluated.maximum_rating:.2f}`",
        f"- Exact 500/2000 values: `{int(evaluated.exact_500_or_2000_count)}`",
        "",
        "## External UCL Guardrail",
        "",
        f"- Baseline static UCL Brier: `{baseline.ucl_brier:.6f}`",
        f"- Candidate static UCL Brier: `{evaluated.ucl_brier:.6f}`",
        f"- Candidate - baseline: `{evaluated.ucl_brier_delta:+.6f}`",
        f"- Clustered 95% CI: `[{uncertainty_row.ci_95_lower:+.6f}, "
        f"{uncertainty_row.ci_95_upper:+.6f}]`",
        f"- ClubElo rank correlation delta: "
        f"`{evaluated.clubelo_rank_correlation_delta:+.6f}`",
        f"- Elite cap-match mean rating-gap delta: "
        f"`{evaluated.elite_mean_rating_gap_delta:+.3f}`",
        "",
        "## Unseen Competition Ranking",
        "",
        "| Competition | Model | Spearman | Delta | Pairwise | Delta |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in competition_summary.itertuples(index=False):
        lines.append(
            f"| {row.competition} | {row.model} | {row.ranking_score:.4f} | "
            f"{row.spearman_delta:+.4f} | {row.pairwise_accuracy:.4f} | "
            f"{row.pairwise_delta:+.4f} |"
        )
    cap = cap_correlation.iloc[0]
    lines.extend(
        [
            "",
            "## Saturation Diagnostic",
            "",
            f"- UCL matches: `{int(cap.matches)}`",
            f"- Spearman saturation-count vs Brier: "
            f"`{cap.spearman_saturation_vs_brier:.4f}`",
            f"- Pearson saturation-count vs Brier: "
            f"`{cap.pearson_saturation_vs_brier:.4f}`",
            "",
            "A positive correlation is diagnostic evidence that heavily saturated "
            "matchups are harder for the static model; it is not by itself a promotion rule.",
            "",
            "The `2026/27` season remains outside parameter selection and is reserved "
            "for the future untouched holdout.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
