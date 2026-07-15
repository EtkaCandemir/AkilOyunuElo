from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from math import ceil
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
PILOT_ROOT = ROOT / "data" / "real_pilot_10_teams"
OUTPUT_ROOT = ROOT / "output" / "ranking_first_calibration"
SEASONS = ("2021/22", "2022/23", "2023/24", "2024/25", "2025/26")
COMPETITIONS = ("UCL", "UEL", "UECL")
HOME_ADVANTAGE = 70.0
ZERO_EXPOSURE_MAE_TOLERANCE = 0.02
EXTREME_OVERRANK_THRESHOLD = 0.35


class RankingEvaluator:
    def __init__(self) -> None:
        self._matches: dict[str, pd.DataFrame] = {}
        self._performance: dict[tuple[str, str], pd.DataFrame] = {}

    def evaluate(
        self,
        candidate: str,
        config: AOEuropeanEloConfig,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        ranking_rows = []
        log_loss_rows = []
        for season in SEASONS:
            ratings = self.ratings(config, season)
            matches = self.matches(season)
            overall = evaluate_match_predictions(
                ratings,
                matches,
                home_advantage=HOME_ADVANTAGE,
            )
            log_loss_rows.append(
                {
                    "candidate": candidate,
                    "season": season,
                    "log_loss": overall["log_loss"],
                    "brier": overall["brier"],
                }
            )
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
        return pd.DataFrame(ranking_rows), pd.DataFrame(log_loss_rows)

    def ratings(
        self,
        config: AOEuropeanEloConfig,
        season: str,
    ) -> pd.DataFrame:
        folder = DATA_ROOT / season.replace("/", "-")
        return compute_ao_first_elo_from_csv(
            folder / "teams.csv",
            folder / "country_coefficients.csv",
            folder / "domestic_context.csv",
            folder / "club_european_points.csv",
            config,
        )

    def matches(self, season: str) -> pd.DataFrame:
        if season not in self._matches:
            folder = DATA_ROOT / season.replace("/", "-")
            self._matches[season] = pd.read_csv(folder / "matches.csv")
        return self._matches[season]

    def ranking_table(
        self,
        ratings: pd.DataFrame,
        season: str,
        competition: str,
    ) -> pd.DataFrame:
        performance = self.performance(season, competition)
        columns = ["team_id", "team_name", "ao_first_elo", "european_exposure"]
        table = performance.merge(
            ratings[columns],
            on="team_id",
            how="inner",
            validate="one_to_one",
        )
        table["predicted_percentile"] = table["ao_first_elo"].rank(
            method="average",
            pct=True,
        )
        table["actual_percentile"] = table["actual_score_rate"].rank(
            method="average",
            pct=True,
        )
        table["percentile_error"] = (
            table["predicted_percentile"] - table["actual_percentile"]
        )
        return table

    def performance(self, season: str, competition: str) -> pd.DataFrame:
        key = (season, competition)
        if key not in self._performance:
            matches = self.matches(season)
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
            self._performance[key] = (
                pd.concat([home, away], ignore_index=True)
                .groupby("team_id", as_index=False)
                .agg(actual_score_rate=("score", "mean"), matches=("score", "size"))
            )
        return self._performance[key]


def main() -> None:
    global DATA_ROOT, OUTPUT_ROOT, SEASONS
    parser = argparse.ArgumentParser(description="Run ranking-first AO Elo calibration")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    DATA_ROOT = args.data_root.resolve()
    OUTPUT_ROOT = args.output_root.resolve()
    SEASONS = tuple(
        folder.name.replace("-", "/")
        for folder in sorted(DATA_ROOT.glob("20??-??"))
    )
    if len(SEASONS) < 3:
        raise ValueError("Ranking-first calibration requires at least three seasons")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    baseline = AOEuropeanEloConfig.v1_1()
    candidates = country_candidates(baseline)
    evaluator = RankingEvaluator()

    ranking_frames = []
    loss_frames = []
    pilot_rows = []
    for candidate, config in candidates:
        ranking, losses = evaluator.evaluate(candidate, config)
        ranking_frames.append(ranking)
        loss_frames.append(losses)
        pilot_rows.append(evaluate_pilot_guardrails(candidate, config))

    ranking = pd.concat(ranking_frames, ignore_index=True)
    losses = pd.concat(loss_frames, ignore_index=True)
    pilot = pd.DataFrame(pilot_rows)
    folds = expanding_folds(SEASONS)
    selections = []
    fold_results = []

    for fold, (train_seasons, test_season) in enumerate(folds, start=1):
        summary = summarize_training(
            ranking,
            losses,
            pilot,
            train_seasons,
        )
        summary.insert(0, "fold", fold)
        summary.to_csv(OUTPUT_ROOT / f"fold_{fold}_candidate_summary.csv", index=False)
        eligible = summary.loc[summary["eligible"]]
        selected = (
            str(eligible.iloc[0]["candidate"])
            if not eligible.empty
            else "country_current"
        )
        selection = summary.loc[summary["candidate"].eq(selected)].iloc[0]
        selections.append(
            {
                "fold": fold,
                "train_seasons": "|".join(train_seasons),
                "test_season": test_season,
                "selected_candidate": selected,
                "train_ranking_score": selection["ranking_score"],
                "train_pairwise_accuracy": selection["pairwise_accuracy"],
                "train_top_quartile_precision": selection["top_quartile_precision"],
                "pilot_guardrails_pass": selection["pilot_guardrails_pass"],
            }
        )
        fold_results.append(
            fold_comparison(ranking, losses, fold, test_season, selected)
        )

    selections_frame = pd.DataFrame(selections)
    results = pd.concat(fold_results, ignore_index=True)
    competition_summary = summarize_fold_results(results)
    decision = make_decision(selections_frame, results, competition_summary)

    ranking.to_csv(OUTPUT_ROOT / "all_candidate_ranking_metrics.csv", index=False)
    losses.to_csv(OUTPUT_ROOT / "all_candidate_prediction_metrics.csv", index=False)
    pilot.to_csv(OUTPUT_ROOT / "pilot_guardrail_results.csv", index=False)
    selections_frame.to_csv(OUTPUT_ROOT / "fold_selections.csv", index=False)
    results.to_csv(OUTPUT_ROOT / "fold_ranking_results.csv", index=False)
    competition_summary.to_csv(
        OUTPUT_ROOT / "competition_ranking_summary.csv",
        index=False,
    )
    write_report(selections_frame, results, competition_summary, decision)

    print("AO European Elo ranking-first calibration")
    print(f"Candidates: {len(candidates)}")
    print(f"Outer folds: {len(folds)}")
    print(f"Decision: {decision['decision']}")
    print(f"Report: {OUTPUT_ROOT / 'backtest_report.md'}")


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
                gamma=float(gamma),
                domestic_league_component=float(component),
            ),
        )
        for benchmark in (15, 20, 25, 30, 35)
        for gamma in (0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 2.0)
        for component in (100, 140, 180, 220, 260, 300, 360)
    )
    unique: dict[tuple[float, float, float], tuple[str, AOEuropeanEloConfig]] = {}
    for name, config in values:
        key = (
            float(config.country_strength_benchmark),
            config.gamma,
            config.domestic_league_component,
        )
        unique.setdefault(key, (name, config))
    return list(unique.values())


def ranking_metrics(table: pd.DataFrame) -> dict[str, float | int]:
    spearman = table["ao_first_elo"].corr(table["actual_score_rate"], method="spearman")
    pairwise = pairwise_ranking_accuracy(
        table["ao_first_elo"].to_numpy(dtype=float),
        table["actual_score_rate"].to_numpy(dtype=float),
    )
    top_count = max(1, ceil(len(table) * 0.25))
    predicted_top = set(table.nlargest(top_count, "ao_first_elo")["team_id"])
    actual_top = set(table.nlargest(top_count, "actual_score_rate")["team_id"])
    zero = table.loc[table["european_exposure"].eq(0)]
    zero_overrank = zero["percentile_error"].clip(lower=0)
    return {
        "teams": len(table),
        "ranking_score": float(spearman),
        "pairwise_accuracy": pairwise,
        "top_quartile_precision": len(predicted_top & actual_top) / top_count,
        "rank_percentile_mae": float(table["percentile_error"].abs().mean()),
        "zero_exposure_teams": len(zero),
        "zero_exposure_rank_mae": (
            float(zero["percentile_error"].abs().mean()) if len(zero) else 0.0
        ),
        "zero_exposure_mean_overrank": (
            float(zero_overrank.mean()) if len(zero) else 0.0
        ),
        "zero_exposure_extreme_overrank_count": int(
            (zero["percentile_error"] > EXTREME_OVERRANK_THRESHOLD).sum()
        ),
    }


def pairwise_ranking_accuracy(predicted: np.ndarray, actual: np.ndarray) -> float:
    correct = []
    for left in range(len(actual)):
        for right in range(left + 1, len(actual)):
            actual_direction = np.sign(actual[left] - actual[right])
            if actual_direction == 0:
                continue
            predicted_direction = np.sign(predicted[left] - predicted[right])
            correct.append(0.5 if predicted_direction == 0 else predicted_direction == actual_direction)
    return float(np.mean(correct)) if correct else float("nan")


def evaluate_pilot_guardrails(
    candidate: str,
    config: AOEuropeanEloConfig,
) -> dict[str, object]:
    ratings = compute_ao_first_elo_from_csv(
        PILOT_ROOT / "teams.csv",
        PILOT_ROOT / "country_coefficients.csv",
        PILOT_ROOT / "domestic_context.csv",
        PILOT_ROOT / "club_european_points.csv",
        config,
    )
    values = ratings.set_index("team_name")["ao_first_elo"]
    guardrails = pd.read_csv(PILOT_ROOT / "ranking_guardrails.csv")
    violations = []
    for row in guardrails.itertuples(index=False):
        if values[row.higher_team] <= values[row.lower_team]:
            violations.append(f"{row.higher_team}<={row.lower_team}")
    ranks = values.rank(method="min", ascending=False).astype(int)
    return {
        "candidate": candidate,
        "pilot_guardrails_pass": not violations,
        "pilot_guardrail_violations": ";".join(violations),
        "pilot_como_rating": float(values["Como"]),
        "pilot_como_rank": int(ranks["Como"]),
        "pilot_top_team": str(values.idxmax()),
    }


def expanding_folds(
    seasons: tuple[str, ...],
    min_train_seasons: int = 2,
) -> list[tuple[tuple[str, ...], str]]:
    return [
        (seasons[:index], seasons[index])
        for index in range(min_train_seasons, len(seasons))
    ]


def summarize_training(
    ranking: pd.DataFrame,
    losses: pd.DataFrame,
    pilot: pd.DataFrame,
    train_seasons: tuple[str, ...],
) -> pd.DataFrame:
    train = ranking.loc[ranking["season"].isin(train_seasons)]
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
    loss_summary = (
        losses.loc[losses["season"].isin(train_seasons)]
        .groupby("candidate", as_index=False)
        .agg(log_loss=("log_loss", "mean"))
    )
    summary = summary.merge(loss_summary, on="candidate", validate="one_to_one")
    summary = summary.merge(pilot, on="candidate", validate="one_to_one")
    baseline = summary.loc[summary["candidate"].eq("country_current")].iloc[0]
    summary["historical_zero_guardrail_pass"] = (
        summary["zero_exposure_rank_mae"]
        <= baseline["zero_exposure_rank_mae"] + ZERO_EXPOSURE_MAE_TOLERANCE
    ) & (
        summary["zero_exposure_extreme_overrank_count"]
        <= baseline["zero_exposure_extreme_overrank_count"]
    )
    summary["eligible"] = (
        summary["pilot_guardrails_pass"]
        & summary["historical_zero_guardrail_pass"]
    )
    return summary.sort_values(
        [
            "eligible",
            "ranking_score",
            "pairwise_accuracy",
            "top_quartile_precision",
            "rank_percentile_mae",
            "log_loss",
        ],
        ascending=[False, False, False, False, True, True],
    ).reset_index(drop=True)


def fold_comparison(
    ranking: pd.DataFrame,
    losses: pd.DataFrame,
    fold: int,
    test_season: str,
    selected: str,
) -> pd.DataFrame:
    rows = []
    for model, candidate in (("baseline", "country_current"), ("selected", selected)):
        frame = ranking.loc[
            ranking["candidate"].eq(candidate) & ranking["season"].eq(test_season)
        ]
        loss = losses.loc[
            losses["candidate"].eq(candidate) & losses["season"].eq(test_season)
        ].iloc[0]
        for row in frame.itertuples(index=False):
            rows.append(
                {
                    "fold": fold,
                    "test_season": test_season,
                    "model": model,
                    "candidate": candidate,
                    "competition": row.competition,
                    "ranking_score": row.ranking_score,
                    "pairwise_accuracy": row.pairwise_accuracy,
                    "top_quartile_precision": row.top_quartile_precision,
                    "rank_percentile_mae": row.rank_percentile_mae,
                    "zero_exposure_rank_mae": row.zero_exposure_rank_mae,
                    "zero_exposure_extreme_overrank_count": row.zero_exposure_extreme_overrank_count,
                    "log_loss": loss.log_loss,
                }
            )
    return pd.DataFrame(rows)


def summarize_fold_results(results: pd.DataFrame) -> pd.DataFrame:
    summary = results.groupby(["model", "competition"], as_index=False).agg(
        ranking_score=("ranking_score", "mean"),
        pairwise_accuracy=("pairwise_accuracy", "mean"),
        top_quartile_precision=("top_quartile_precision", "mean"),
        rank_percentile_mae=("rank_percentile_mae", "mean"),
        log_loss=("log_loss", "mean"),
    )
    baseline = summary.loc[summary["model"].eq("baseline")].set_index("competition")
    summary["ranking_score_delta"] = summary.apply(
        lambda row: row["ranking_score"]
        - baseline.loc[row["competition"], "ranking_score"],
        axis=1,
    )
    return summary.sort_values(["competition", "model"]).reset_index(drop=True)


def make_decision(
    selections: pd.DataFrame,
    results: pd.DataFrame,
    competition_summary: pd.DataFrame,
) -> dict[str, object]:
    pivot = results.pivot_table(
        index=["fold", "test_season"],
        columns="model",
        values="ranking_score",
        aggfunc="mean",
    )
    folds_improved = int((pivot["selected"] > pivot["baseline"]).sum())
    tournament = competition_summary.loc[
        competition_summary["model"].eq("selected")
    ]
    tournaments_not_worse = int((tournament["ranking_score_delta"] >= 0).sum())
    same_candidate = selections["selected_candidate"].nunique() == 1
    all_vetos_pass = bool(selections["pilot_guardrails_pass"].all())
    promote = (
        folds_improved == len(selections)
        and tournaments_not_worse == len(COMPETITIONS)
        and same_candidate
        and all_vetos_pass
        and selections.iloc[0]["selected_candidate"] != "country_current"
    )
    return {
        "decision": "PROMOTE" if promote else "KEEP_V1_1",
        "folds_improved": folds_improved,
        "tournaments_not_worse": tournaments_not_worse,
        "same_candidate_every_fold": same_candidate,
        "all_pilot_vetos_pass": all_vetos_pass,
    }


def write_report(
    selections: pd.DataFrame,
    results: pd.DataFrame,
    competition_summary: pd.DataFrame,
    decision: dict[str, object],
) -> None:
    lines = [
        "# AO European Elo Ranking-First Calibration",
        "",
        "Ranking is the primary objective. Log loss is used only after ranking metrics.",
        "Candidates must pass historical zero-exposure and real-pilot pairwise guardrails.",
        "A candidate is promoted only if it improves every unseen fold, does not worsen",
        "any competition, and the same configuration is selected in every fold.",
        "",
        "## Fold selections",
        "",
        "| Fold | Train | Test | Selected | Train Spearman | Pairwise | Top quartile |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in selections.itertuples(index=False):
        lines.append(
            f"| {row.fold} | {row.train_seasons.replace('|', ', ')} | {row.test_season} | "
            f"{row.selected_candidate} | {row.train_ranking_score:.4f} | "
            f"{row.train_pairwise_accuracy:.4f} | {row.train_top_quartile_precision:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Unseen competition ranking",
            "",
            "| Competition | Model | Spearman | Delta | Pairwise | Top quartile | Rank MAE |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in competition_summary.itertuples(index=False):
        lines.append(
            f"| {row.competition} | {row.model} | {row.ranking_score:.4f} | "
            f"{row.ranking_score_delta:+.4f} | {row.pairwise_accuracy:.4f} | "
            f"{row.top_quartile_precision:.4f} | {row.rank_percentile_mae:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Promotion decision",
            "",
            f"- Decision: `{decision['decision']}`",
            f"- Unseen folds improved: `{decision['folds_improved']}/{len(selections)}`",
            f"- Competitions not worse: `{decision['tournaments_not_worse']}/{len(COMPETITIONS)}`",
            f"- Same candidate selected in every fold: `{decision['same_candidate_every_fold']}`",
            f"- Pilot ranking vetos pass: `{decision['all_pilot_vetos_pass']}`",
            "",
            "No aggregate prediction improvement can override a failed ranking veto.",
        ]
    )
    (OUTPUT_ROOT / "backtest_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
