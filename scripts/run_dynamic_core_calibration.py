from __future__ import annotations

import argparse
import math
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


STATIC_DATA_ROOT = ROOT / "data" / "backtest_stage_b_2018_2026"
EVENTS_PATH = ROOT / "data" / "dynamic_backtest_2018_2026" / "matches.csv"
OUTPUT_ROOT = ROOT / "output" / "dynamic_core_calibration_2018_2026"
ELO_SCALES = (100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400)
HOME_ADVANTAGES = tuple(range(0, 81, 10))
K_FACTORS = (0, 8, 12, 16, 20, 24, 28, 32, 40)
LEGACY_SCALE = 400.0
LEGACY_HOME_ADVANTAGE = 70.0
RANK_CORRELATION_FLOOR = 0.85
MAX_RATING_MOVE_GUARDRAIL = 200.0


@dataclass(frozen=True, order=True)
class DynamicCoreConfig:
    elo_scale: float
    home_advantage: float
    k_factor: float

    def validate(self) -> None:
        if not math.isfinite(self.elo_scale) or self.elo_scale <= 0:
            raise ValueError("elo_scale must be positive and finite")
        if not math.isfinite(self.home_advantage) or self.home_advantage < 0:
            raise ValueError("home_advantage must be non-negative and finite")
        if not math.isfinite(self.k_factor) or self.k_factor < 0:
            raise ValueError("k_factor must be non-negative and finite")


@dataclass(frozen=True)
class SeasonData:
    season: str
    initial_ratings: np.ndarray
    active_team_ids: np.ndarray
    home_team_ids: np.ndarray
    away_team_ids: np.ndarray
    actual_home_scores: np.ndarray
    neutral_flags: np.ndarray
    competitions: np.ndarray
    match_ids: np.ndarray


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate AO dynamic Elo scale, home advantage and base K"
    )
    parser.add_argument("--static-data-root", type=Path, default=STATIC_DATA_ROOT)
    parser.add_argument("--events-path", type=Path, default=EVENTS_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    static_root = args.static_data_root.resolve()
    events_path = args.events_path.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    season_data = load_calibration_data(static_root, events_path)
    seasons = tuple(data.season for data in season_data)
    folds = expanding_folds(seasons)
    candidates = candidate_grid()

    selections: list[dict[str, object]] = []
    fold_metrics: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for fold_number, (train_seasons, test_season) in enumerate(folds, start=1):
        train_data = tuple(data for data in season_data if data.season in train_seasons)
        test_data = next(data for data in season_data if data.season == test_season)
        dynamic_config, dynamic_train = select_candidate(train_data, candidates)
        static_config, static_train = select_candidate(
            train_data,
            tuple(config for config in candidates if config.k_factor == 0),
        )
        legacy_config = DynamicCoreConfig(
            LEGACY_SCALE,
            LEGACY_HOME_ADVANTAGE,
            0.0,
        )
        selections.append(
            {
                "fold": fold_number,
                "train_seasons": ",".join(train_seasons),
                "test_season": test_season,
                "selected_scale": dynamic_config.elo_scale,
                "selected_home_advantage": dynamic_config.home_advantage,
                "selected_k": dynamic_config.k_factor,
                "train_brier": dynamic_train["brier"],
                "train_log_loss": dynamic_train["log_loss"],
                "static_scale": static_config.elo_scale,
                "static_home_advantage": static_config.home_advantage,
                "static_train_brier": static_train["brier"],
            }
        )
        model_predictions: dict[str, pd.DataFrame] = {}
        for model_name, config in (
            ("selected_dynamic", dynamic_config),
            ("selected_static", static_config),
            ("legacy_static", legacy_config),
        ):
            metrics, predictions = evaluate_seasons((test_data,), config, return_predictions=True)
            fold_metrics.append(
                {
                    "fold": fold_number,
                    "test_season": test_season,
                    "model": model_name,
                    "elo_scale": config.elo_scale,
                    "home_advantage": config.home_advantage,
                    "k_factor": config.k_factor,
                    **metrics,
                }
            )
            assert predictions is not None
            model_predictions[model_name] = predictions.rename(
                columns={
                    "expected_home_score": f"{model_name}_expected_home_score",
                    "brier_loss": f"{model_name}_brier_loss",
                    "log_loss": f"{model_name}_log_loss",
                }
            )
        joined = model_predictions["selected_dynamic"].merge(
            model_predictions["selected_static"][
                [
                    "match_id",
                    "selected_static_expected_home_score",
                    "selected_static_brier_loss",
                    "selected_static_log_loss",
                ]
            ],
            on="match_id",
            validate="one_to_one",
        ).merge(
            model_predictions["legacy_static"][
                [
                    "match_id",
                    "legacy_static_expected_home_score",
                    "legacy_static_brier_loss",
                    "legacy_static_log_loss",
                ]
            ],
            on="match_id",
            validate="one_to_one",
        )
        joined.insert(0, "fold", fold_number)
        prediction_frames.append(joined)

    selections_df = pd.DataFrame(selections)
    fold_metrics_df = pd.DataFrame(fold_metrics)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    competition_summary = summarize_competitions(predictions_df)
    uncertainty = paired_uncertainty(predictions_df)
    stability = parameter_stability(selections_df)

    final_config, final_metrics = select_candidate(season_data, candidates)
    full_candidate_metrics = candidate_metrics(season_data, candidates)
    final_season_metrics = evaluate_by_season(season_data, final_config)
    decision = calibration_decision(
        fold_metrics_df,
        competition_summary,
        uncertainty,
        stability,
        final_config,
    )

    selections_df.to_csv(output_root / "fold_selections.csv", index=False)
    fold_metrics_df.to_csv(output_root / "fold_results.csv", index=False)
    predictions_df.to_csv(output_root / "unseen_match_predictions.csv", index=False)
    competition_summary.to_csv(output_root / "competition_summary.csv", index=False)
    uncertainty.to_csv(output_root / "paired_uncertainty.csv", index=False)
    stability.to_csv(output_root / "parameter_stability.csv", index=False)
    full_candidate_metrics.to_csv(output_root / "full_candidate_metrics.csv", index=False)
    final_season_metrics.to_csv(output_root / "final_candidate_season_metrics.csv", index=False)
    write_report(
        output_root / "calibration_report.md",
        seasons,
        selections_df,
        fold_metrics_df,
        competition_summary,
        uncertainty,
        stability,
        final_config,
        final_metrics,
        decision,
    )

    print("AO dynamic Elo core calibration")
    print(f"Seasons: {len(seasons)}")
    print(f"Matches: {sum(len(data.match_ids) for data in season_data)}")
    print(f"Outer folds: {len(folds)}")
    print(
        "Full-data candidate: "
        f"Scale={final_config.elo_scale:g}, H={final_config.home_advantage:g}, "
        f"K={final_config.k_factor:g}"
    )
    print(f"Decision: {decision}")
    print(f"Report: {output_root / 'calibration_report.md'}")


def load_calibration_data(
    static_root: Path,
    events_path: Path,
) -> tuple[SeasonData, ...]:
    events = pd.read_csv(events_path).sort_values(["season", "event_order"])
    required = {
        "match_id", "season", "event_order", "competition", "home_team_id",
        "away_team_id", "actual_home_score", "is_neutral",
    }
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"Dynamic event data missing columns: {missing}")
    if events["match_id"].duplicated().any():
        raise ValueError("Dynamic event match_id must be unique")
    if not events["actual_home_score"].isin((0.0, 0.5, 1.0)).all():
        raise ValueError("actual_home_score must contain only 0, 0.5 or 1")

    config = AOEuropeanEloConfig.v1_1()
    result: list[SeasonData] = []
    for season, season_events in events.groupby("season", sort=True):
        folder = static_root / season.replace("/", "-")
        ratings = compute_ao_first_elo_from_csv(
            folder / "teams.csv",
            folder / "country_coefficients.csv",
            folder / "domestic_context.csv",
            folder / "club_european_points.csv",
            config,
        )
        max_team_id = int(
            max(
                ratings["team_id"].max(),
                season_events["home_team_id"].max(),
                season_events["away_team_id"].max(),
            )
        )
        initial = np.full(max_team_id + 1, np.nan, dtype=float)
        initial[ratings["team_id"].astype(int).to_numpy()] = ratings["ao_first_elo"].to_numpy(float)
        active_ids = np.unique(
            np.concatenate(
                (
                    season_events["home_team_id"].to_numpy(int),
                    season_events["away_team_id"].to_numpy(int),
                )
            )
        )
        if np.isnan(initial[active_ids]).any():
            raise ValueError(f"{season}: event team without an AO First Elo rating")
        expected_order = np.arange(1, len(season_events) + 1)
        if not np.array_equal(season_events["event_order"].to_numpy(int), expected_order):
            raise ValueError(f"{season}: event_order must be contiguous from one")
        result.append(
            SeasonData(
                season=season,
                initial_ratings=initial,
                active_team_ids=active_ids,
                home_team_ids=season_events["home_team_id"].to_numpy(int),
                away_team_ids=season_events["away_team_id"].to_numpy(int),
                actual_home_scores=season_events["actual_home_score"].to_numpy(float),
                neutral_flags=season_events["is_neutral"].to_numpy(bool),
                competitions=season_events["competition"].to_numpy(str),
                match_ids=season_events["match_id"].to_numpy(str),
            )
        )
    if len(result) < 3:
        raise ValueError("Dynamic calibration requires at least three seasons")
    return tuple(result)


def candidate_grid() -> tuple[DynamicCoreConfig, ...]:
    return tuple(
        DynamicCoreConfig(float(scale), float(home_advantage), float(k_factor))
        for scale in ELO_SCALES
        for home_advantage in HOME_ADVANTAGES
        for k_factor in K_FACTORS
    )


def expanding_folds(
    seasons: tuple[str, ...],
    minimum_train_seasons: int = 2,
) -> tuple[tuple[tuple[str, ...], str], ...]:
    if minimum_train_seasons < 1 or len(seasons) <= minimum_train_seasons:
        raise ValueError("Not enough seasons for expanding walk-forward folds")
    return tuple(
        (seasons[:index], seasons[index])
        for index in range(minimum_train_seasons, len(seasons))
    )


def expected_home_score(
    home_rating: float,
    away_rating: float,
    config: DynamicCoreConfig,
    *,
    neutral: bool,
) -> float:
    config.validate()
    home_advantage = 0.0 if neutral else config.home_advantage
    exponent = -((home_rating - away_rating + home_advantage) / config.elo_scale)
    return 1.0 / (1.0 + 10.0 ** exponent)


def run_season(
    data: SeasonData,
    config: DynamicCoreConfig,
    *,
    return_predictions: bool = False,
) -> tuple[dict[str, float | int], pd.DataFrame | None]:
    config.validate()
    ratings = data.initial_ratings.copy()
    brier_sum = 0.0
    log_loss_sum = 0.0
    prediction_rows: list[dict[str, object]] = []
    for index, (home_id, away_id, actual, neutral) in enumerate(
        zip(
            data.home_team_ids,
            data.away_team_ids,
            data.actual_home_scores,
            data.neutral_flags,
        )
    ):
        probability = float(
            np.clip(
                expected_home_score(
                    ratings[home_id],
                    ratings[away_id],
                    config,
                    neutral=bool(neutral),
                ),
                1e-12,
                1.0 - 1e-12,
            )
        )
        brier_loss = (probability - actual) ** 2
        log_loss = -(actual * math.log(probability) + (1.0 - actual) * math.log(1.0 - probability))
        brier_sum += brier_loss
        log_loss_sum += log_loss
        delta = config.k_factor * (actual - probability)
        ratings[home_id] += delta
        ratings[away_id] -= delta
        if return_predictions:
            prediction_rows.append(
                {
                    "match_id": data.match_ids[index],
                    "season": data.season,
                    "competition": data.competitions[index],
                    "actual_home_score": actual,
                    "is_neutral": bool(neutral),
                    "expected_home_score": probability,
                    "brier_loss": brier_loss,
                    "log_loss": log_loss,
                }
            )
    changes = ratings[data.active_team_ids] - data.initial_ratings[data.active_team_ids]
    start_ratings = data.initial_ratings[data.active_team_ids]
    end_ratings = ratings[data.active_team_ids]
    if np.allclose(start_ratings, end_ratings):
        rank_correlation = 1.0
    elif np.ptp(start_ratings) == 0 or np.ptp(end_ratings) == 0:
        rank_correlation = 0.0
    else:
        rank_correlation = pd.Series(end_ratings).corr(
            pd.Series(start_ratings),
            method="spearman",
        )
        if pd.isna(rank_correlation):
            rank_correlation = 0.0
    metrics: dict[str, float | int] = {
        "matches": len(data.match_ids),
        "brier": brier_sum / len(data.match_ids),
        "log_loss": log_loss_sum / len(data.match_ids),
        "mean_rating_change": float(np.mean(changes)),
        "rating_change_std": float(np.std(changes)),
        "max_abs_rating_change": float(np.max(np.abs(changes))),
        "start_end_rank_correlation": float(rank_correlation),
    }
    predictions = pd.DataFrame(prediction_rows) if return_predictions else None
    return metrics, predictions


def evaluate_seasons(
    datasets: tuple[SeasonData, ...],
    config: DynamicCoreConfig,
    *,
    return_predictions: bool = False,
) -> tuple[dict[str, float | int], pd.DataFrame | None]:
    season_metrics: list[dict[str, float | int]] = []
    frames: list[pd.DataFrame] = []
    for data in datasets:
        metrics, predictions = run_season(data, config, return_predictions=return_predictions)
        season_metrics.append(metrics)
        if predictions is not None:
            frames.append(predictions)
    total_matches = sum(int(row["matches"]) for row in season_metrics)
    aggregate: dict[str, float | int] = {
        "matches": total_matches,
        "brier": sum(float(row["brier"]) * int(row["matches"]) for row in season_metrics) / total_matches,
        "log_loss": sum(float(row["log_loss"]) * int(row["matches"]) for row in season_metrics) / total_matches,
        "mean_rating_change": float(np.mean([row["mean_rating_change"] for row in season_metrics])),
        "rating_change_std": float(np.mean([row["rating_change_std"] for row in season_metrics])),
        "max_abs_rating_change": float(max(row["max_abs_rating_change"] for row in season_metrics)),
        "start_end_rank_correlation": float(min(row["start_end_rank_correlation"] for row in season_metrics)),
    }
    predictions = pd.concat(frames, ignore_index=True) if frames else None
    return aggregate, predictions


def candidate_metrics(
    datasets: tuple[SeasonData, ...],
    candidates: tuple[DynamicCoreConfig, ...],
) -> pd.DataFrame:
    rows = []
    for config in candidates:
        metrics, _ = evaluate_seasons(datasets, config)
        rows.append(
            {
                "elo_scale": config.elo_scale,
                "home_advantage": config.home_advantage,
                "k_factor": config.k_factor,
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["brier", "log_loss", "k_factor", "elo_scale", "home_advantage"]
    ).reset_index(drop=True)


def select_candidate(
    datasets: tuple[SeasonData, ...],
    candidates: tuple[DynamicCoreConfig, ...],
) -> tuple[DynamicCoreConfig, dict[str, float | int]]:
    rows = candidate_metrics(datasets, candidates)
    selected = rows.iloc[0]
    config = DynamicCoreConfig(
        float(selected["elo_scale"]),
        float(selected["home_advantage"]),
        float(selected["k_factor"]),
    )
    metrics = {
        key: selected[key]
        for key in (
            "matches", "brier", "log_loss", "mean_rating_change",
            "rating_change_std", "max_abs_rating_change", "start_end_rank_correlation",
        )
    }
    return config, metrics


def evaluate_by_season(
    datasets: tuple[SeasonData, ...],
    config: DynamicCoreConfig,
) -> pd.DataFrame:
    rows = []
    for data in datasets:
        metrics, _ = run_season(data, config)
        rows.append({"season": data.season, **metrics})
    return pd.DataFrame(rows)


def summarize_competitions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for competition, data in predictions.groupby("competition"):
        dynamic_brier = data["selected_dynamic_brier_loss"].mean()
        static_brier = data["selected_static_brier_loss"].mean()
        dynamic_log = data["selected_dynamic_log_loss"].mean()
        static_log = data["selected_static_log_loss"].mean()
        rows.append(
            {
                "competition": competition,
                "matches": len(data),
                "dynamic_brier": dynamic_brier,
                "static_brier": static_brier,
                "brier_difference": dynamic_brier - static_brier,
                "dynamic_log_loss": dynamic_log,
                "static_log_loss": static_log,
                "log_loss_difference": dynamic_log - static_log,
            }
        )
    return pd.DataFrame(rows)


def paired_uncertainty(
    predictions: pd.DataFrame,
    *,
    bootstrap_samples: int = 4000,
    seed: int = 20260715,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    groups = [("ALL", predictions), *predictions.groupby("competition")]
    for competition, data in groups:
        differences = (
            data["selected_dynamic_brier_loss"] - data["selected_static_brier_loss"]
        ).to_numpy(float)
        sampled = rng.choice(differences, size=(bootstrap_samples, len(differences)), replace=True)
        means = sampled.mean(axis=1)
        lower, upper = np.quantile(means, (0.025, 0.975))
        rows.append(
            {
                "competition": competition,
                "matches": len(differences),
                "mean_brier_difference": differences.mean(),
                "ci_95_lower": lower,
                "ci_95_upper": upper,
                "reliable_improvement": bool(upper < 0),
                "reliable_harm": bool(lower > 0),
            }
        )
    return pd.DataFrame(rows)


def parameter_stability(selections: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in ("selected_scale", "selected_home_advantage", "selected_k"):
        counts = selections[column].value_counts().sort_values(ascending=False)
        rows.append(
            {
                "parameter": column,
                "mode": counts.index[0],
                "mode_count": int(counts.iloc[0]),
                "folds": len(selections),
                "mode_share": float(counts.iloc[0] / len(selections)),
                "min": float(selections[column].min()),
                "max": float(selections[column].max()),
            }
        )
    return pd.DataFrame(rows)


def calibration_decision(
    fold_metrics: pd.DataFrame,
    competition_summary: pd.DataFrame,
    uncertainty: pd.DataFrame,
    stability: pd.DataFrame,
    final_config: DynamicCoreConfig,
) -> str:
    pivot = fold_metrics.pivot(index="fold", columns="model", values="brier")
    fold_wins = int((pivot["selected_dynamic"] < pivot["selected_static"]).sum())
    overall = uncertainty.loc[uncertainty["competition"].eq("ALL")].iloc[0]
    no_competition_harm = not bool((competition_summary["brier_difference"] > 0).any())
    k_stability = float(
        stability.loc[stability["parameter"].eq("selected_k"), "mode_share"].iloc[0]
    )
    dynamic_rows = fold_metrics.loc[fold_metrics["model"].eq("selected_dynamic")]
    ranking_safe = bool(
        dynamic_rows["start_end_rank_correlation"].ge(RANK_CORRELATION_FLOOR).all()
        and dynamic_rows["max_abs_rating_change"].le(MAX_RATING_MOVE_GUARDRAIL).all()
    )
    evidence_passes = (
        fold_wins >= 5
        and bool(overall["reliable_improvement"])
        and no_competition_harm
        and k_stability >= 0.5
        and ranking_safe
    )
    grid_boundary_hit = (
        final_config.elo_scale in {min(ELO_SCALES), max(ELO_SCALES)}
        or final_config.home_advantage in {min(HOME_ADVANTAGES), max(HOME_ADVANTAGES)}
        or final_config.k_factor in {min(K_FACTORS), max(K_FACTORS)}
    )
    if evidence_passes and grid_boundary_hit:
        return "CORE_SIGNAL_CONFIRMED_PARAMETER_BOUNDARY"
    if evidence_passes:
        return "PROVISIONAL_ACCEPT_CORE"
    return "KEEP_AS_CANDIDATE_AND_CONTINUE_TESTING"


def write_report(
    path: Path,
    seasons: tuple[str, ...],
    selections: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    competition_summary: pd.DataFrame,
    uncertainty: pd.DataFrame,
    stability: pd.DataFrame,
    final_config: DynamicCoreConfig,
    final_metrics: dict[str, float | int],
    decision: str,
) -> None:
    pivot = fold_metrics.pivot(index="fold", columns="model", values="brier")
    fold_wins = int((pivot["selected_dynamic"] < pivot["selected_static"]).sum())
    overall = uncertainty.loc[uncertainty["competition"].eq("ALL")].iloc[0]
    selection_rows = [
        "| {fold} | {test_season} | {selected_scale:g} | {selected_home_advantage:g} | "
        "{selected_k:g} |".format(**row._asdict())
        for row in selections.itertuples(index=False)
    ]
    competition_rows = [
        f"| {row.competition} | {row.matches} | {row.brier_difference:.6f} | "
        f"{row.log_loss_difference:.6f} |"
        for row in competition_summary.itertuples(index=False)
    ]
    stability_rows = [
        f"| {row.parameter} | {row.mode:g} | {row.mode_count}/{row.folds} | "
        f"{row.min:g}-{row.max:g} |"
        for row in stability.itertuples(index=False)
    ]
    text = "\n".join(
        [
            "# AO Dynamic Elo Core Calibration",
            "",
            f"Decision: **{decision}**",
            "",
            "## Scope",
            "",
            f"Seasons: {seasons[0]} through {seasons[-1]}; outer folds: {len(selections)}.",
            "Only Elo scale, home advantage and base K are active. Competition, stage,",
            "goal-margin, progression and season-carry multipliers are fixed at neutral values.",
            "Every season starts from the frozen AO First Elo v1.1 rating.",
            "",
            "Brier score is the primary selection metric. The static comparator selects its",
            "own Scale/H on the same training seasons with K=0, so the comparison isolates",
            "the value of chronological match updates.",
            "",
            "## Walk-Forward Selections",
            "",
            "| Fold | Unseen season | Scale | H | K |",
            "| ---: | --- | ---: | ---: | ---: |",
            *selection_rows,
            "",
            f"Dynamic beat the tuned static comparator in **{fold_wins}/{len(selections)}** unseen folds.",
            f"Overall paired Brier difference: {overall.mean_brier_difference:.6f} ",
            f"(95% CI {overall.ci_95_lower:.6f} to {overall.ci_95_upper:.6f}).",
            "",
            "## Competition Guardrail",
            "",
            "Negative differences favor the dynamic model.",
            "",
            "| Competition | Matches | Brier difference | Log-loss difference |",
            "| --- | ---: | ---: | ---: |",
            *competition_rows,
            "",
            "## Parameter Stability",
            "",
            "| Parameter | Mode | Fold frequency | Range |",
            "| --- | ---: | ---: | ---: |",
            *stability_rows,
            "",
            "## Full-Data Research Candidate",
            "",
            f"`Scale={final_config.elo_scale:g}`, `H={final_config.home_advantage:g}`, "
            f"`K={final_config.k_factor:g}`; Brier={float(final_metrics['brier']):.6f}; ",
            f"log loss={float(final_metrics['log_loss']):.6f}.",
            "",
            "This is a research candidate, not a frozen production parameter set. The next",
            "tests must add one layer at a time: goal margin, competition, stage, progression,",
            "caps and season carry. Exact extra-time policy also remains outside this run.",
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
