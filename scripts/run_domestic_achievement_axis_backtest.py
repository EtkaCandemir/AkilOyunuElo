from __future__ import annotations

"""Walk-forward sweep of the three Domestic Prior axes, one axis at a time.

    Domestic Prior = base
                   + domestic_league_component      * L
                   + domestic_achievement_component * A * (alpha + (1 - alpha) * L)

    L = (ln(1 + weighted_country_score) / ln(1 + benchmark)) ** gamma

`gamma` compresses an already log-compressed quantity a second time: at 0.80 a
Georgian league lands at 0.32 where its raw normalization says 0.24, which
closes a quarter of the distance to England. The value was never selected on
evidence. `run_ranking_first_calibration.py` did sweep it, jointly with the
benchmark and the league component, and every fold preferred gamma >= 1.4 - but
the folds disagreed on the other two axes, the stability gate refused to promote
anything, and 0.80 stayed in place as the value inherited from v1.1.

This command asks the question that was never asked: what does each axis do on
its own, with everything else pinned to what production serves today?

Unlike the exposure studies, the seed cannot be transformed after the fact -
gamma feeds `league_strength`, which feeds `domestic_prior` upstream - so every
candidate recomputes AO First from the raw season inputs.

Research only: the active contract is hashed before and after, and nothing in
`contracts/` or `artifacts/` is written.
"""

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ao_elo.config import AOEuropeanEloConfig  # noqa: E402
from ao_elo.european_prior_recalibration import ranking_uncertainty_summary  # noqa: E402
from ao_elo.evaluation import schedule_adjusted_team_performance  # noqa: E402
from ao_elo.pipeline import compute_ao_first_elo_from_csv  # noqa: E402
from scripts.run_ao_first_seed_boost_backtest import aggregate_target  # noqa: E402
from scripts.run_current_model_evaluation import (  # noqa: E402
    EvaluationArm,
    evaluate_arm,
)
from scripts.run_dynamic_core_calibration import expanding_folds  # noqa: E402
from scripts.run_european_prior_recalibration_backtest import (  # noqa: E402
    competition_summary,
    model_summary,
    rank_metrics,
    sha256,
    uncertainty_summary,
)
from scripts.run_opponent_quintile_backtest import load_production_baseline  # noqa: E402
from scripts.run_stage_weighted_progression_backtest import (  # noqa: E402
    DOMESTIC_ADJUSTMENTS,
    EVENTS_PATH,
    PRODUCTION_CONTRACT,
    STATIC_DATA_ROOT,
    XG_DATA,
    load_xg_map,
)
from scripts.run_v2_evaluation_upgrade import read_events  # noqa: E402


OUTPUT_ROOT = ROOT / "output" / "domestic_achievement_axis_backtest_2018_2026"
BASELINE_KEY = "PRODUCTION"

ACTIVE = AOEuropeanEloConfig.active()
# Reaches 2.0, where the ranking-first sweep had already flattened, so the
# selection has an open edge and cannot be pinned there by the grid.
# `percentile_delta` bends the league-position curve the way gamma bends the
# league-strength curve: at the live 1.0 the gap between 1st and 2nd is worth
# exactly what the gap between 12th and 13th is worth. Above 1.0 a high finish
# counts for more, below it the table flattens.
PERCENTILE_DELTAS = (0.7, 1.3, 1.6, 2.0)
# `achievement_alpha` is the floor of the achievement scale, so it sets how much
# a strong domestic finish depends on the league it was earned in. It is also
# the channel gamma leaks through - lowering L lowers the scale - so it has
# never been measured apart from gamma.
ACHIEVEMENT_ALPHAS = (0.2, 0.3, 0.5, 0.7)


@dataclass(frozen=True)
class Candidate:
    key: str
    axis: str
    value: float
    config: AOEuropeanEloConfig


def candidate_grid() -> tuple[Candidate, ...]:
    return (
        Candidate(BASELINE_KEY, "BASELINE", 0.0, ACTIVE),
        *(
            Candidate(
                f"delta{value:g}",
                "PERCENTILE_DELTA",
                value,
                replace(ACTIVE, percentile_delta=value),
            )
            for value in PERCENTILE_DELTAS
        ),
        *(
            Candidate(
                f"alpha{value:g}",
                "ACHIEVEMENT_ALPHA",
                value,
                replace(ACTIVE, achievement_alpha=value),
            )
            for value in ACHIEVEMENT_ALPHAS
        ),
    )


def load_domestic_surprise() -> pd.DataFrame:
    """The frozen Domestic Surprise adjustment, one Elo delta per club-season.

    The pipeline's own `ao_first_elo` is the value *before* this layer, so a
    candidate built from it alone misses production by the layer's full cap of
    30 Elo. The adjustment comes from its own study and is held fixed here, and
    it rides along additively rather than being scaled - the same treatment
    `apply_european_prior_recalibration` gives it, and the only one that keeps
    an axis sweep measuring its own axis.
    """
    frame = pd.read_csv(DOMESTIC_ADJUSTMENTS, low_memory=False)
    return frame[["season", "team_id", "domestic_prior_adjustment"]]


def seed_frame(config: AOEuropeanEloConfig, surprise: pd.DataFrame) -> pd.DataFrame:
    """Recompute AO First for every historical season under one config."""
    rows = []
    for folder in sorted(STATIC_DATA_ROOT.glob("20??-??")):
        year = int(folder.name[:4])
        season = f"{year}/{str(year + 1)[-2:]}"
        rating = compute_ao_first_elo_from_csv(
            folder / "teams.csv",
            folder / "country_coefficients.csv",
            folder / "domestic_context.csv",
            folder / "club_european_points.csv",
            config,
        )
        rows.append(
            rating[
                [
                    "team_id",
                    "team_name",
                    "domestic_prior",
                    "european_prior",
                    "effective_european_exposure",
                ]
            ].assign(season=season)
        )
    frame = pd.concat(rows, ignore_index=True).merge(
        surprise, on=["season", "team_id"], validate="one_to_one"
    )
    adjusted_domestic = frame["domestic_prior"] + frame["domestic_prior_adjustment"]
    frame["ao_first_elo"] = adjusted_domestic + frame[
        "effective_european_exposure"
    ] * (frame["european_prior"] - adjusted_domestic)
    return frame


def build_unseen(selections, predictions, candidate_seeds, target, folds):
    """Score each fold's pick on its held-out season against the baseline.

    A local copy of the shared helper: that one resolves the baseline through
    `BASELINE_CONFIG.key`, which names a candidate from the exposure studies and
    does not exist in this grid.
    """
    unseen_rows = []
    fold_rows = []
    baseline_predictions = predictions[BASELINE_KEY]
    baseline_seeds = candidate_seeds[BASELINE_KEY]
    for fold, (_, test_season) in enumerate(folds, start=1):
        selected_key = str(
            selections.loc[selections["fold"].eq(fold), "selected_candidate_key"].iloc[0]
        )
        selected = predictions[selected_key].loc[
            predictions[selected_key]["season"].eq(test_season)
        ].copy()
        baseline = baseline_predictions.loc[
            baseline_predictions["season"].eq(test_season)
        ][["match_id", "brier_1x2", "log_loss_1x2"]].rename(
            columns={
                "brier_1x2": "baseline_brier_1x2",
                "log_loss_1x2": "baseline_log_loss_1x2",
            }
        )
        selected = selected.merge(baseline, on="match_id", validate="one_to_one")
        selected["fold"] = fold
        selected["selected_candidate_key"] = selected_key
        selected["brier_difference"] = (
            selected["brier_1x2"] - selected["baseline_brier_1x2"]
        )
        selected["log_loss_difference"] = (
            selected["log_loss_1x2"] - selected["baseline_log_loss_1x2"]
        )
        unseen_rows.append(selected)

        selected_rank = rank_metrics(candidate_seeds[selected_key], target, test_season)
        baseline_rank = rank_metrics(baseline_seeds, target, test_season)
        fold_rows.append(
            {
                "fold": fold,
                "test_season": test_season,
                "selected_candidate_key": selected_key,
                "matches": len(selected),
                "brier_1x2": selected["brier_1x2"].mean(),
                "baseline_brier_1x2": selected["baseline_brier_1x2"].mean(),
                "delta_brier_1x2": selected["brier_difference"].mean(),
                "log_loss_1x2": selected["log_loss_1x2"].mean(),
                "baseline_log_loss_1x2": selected["baseline_log_loss_1x2"].mean(),
                "delta_log_loss_1x2": selected["log_loss_difference"].mean(),
                "accuracy_1x2": (
                    selected["actual_class"] == selected["predicted_class"]
                ).mean(),
                "seed_spearman": selected_rank[0],
                "baseline_seed_spearman": baseline_rank[0],
                "delta_seed_spearman": selected_rank[0] - baseline_rank[0],
                "seed_pairwise_accuracy": selected_rank[1],
                "baseline_seed_pairwise_accuracy": baseline_rank[1],
                "delta_seed_pairwise_accuracy": selected_rank[1] - baseline_rank[1],
            }
        )
    return pd.concat(unseen_rows, ignore_index=True), pd.DataFrame(fold_rows)


def season_rows(
    candidate: Candidate,
    seeds: pd.DataFrame,
    predictions: pd.DataFrame,
    target: pd.DataFrame,
    seasons: tuple[str, ...],
) -> list[dict[str, object]]:
    rows = []
    frame = seeds.rename(columns={"ao_first_elo": "candidate_ao_first_elo"})
    for season in seasons:
        matches = predictions.loc[predictions["season"].eq(season)]
        spearman, pairwise = rank_metrics(frame, target, season)
        rows.append(
            {
                "candidate_key": candidate.key,
                "axis": candidate.axis,
                "value": candidate.value,
                "season": season,
                "matches": len(matches),
                "brier_1x2": float(matches["brier_1x2"].mean()),
                "log_loss_1x2": float(matches["log_loss_1x2"].mean()),
                "seed_spearman": spearman,
                "seed_pairwise_accuracy": pairwise,
            }
        )
    return rows


def summarise(surface: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        surface.groupby(["candidate_key", "axis", "value"], as_index=False)
        .agg(
            brier_1x2=("brier_1x2", "mean"),
            log_loss_1x2=("log_loss_1x2", "mean"),
            seed_spearman=("seed_spearman", "mean"),
            seed_pairwise_accuracy=("seed_pairwise_accuracy", "mean"),
        )
    )
    baseline = grouped.loc[grouped["candidate_key"].eq(BASELINE_KEY)].iloc[0]
    for column in (
        "brier_1x2",
        "log_loss_1x2",
        "seed_spearman",
        "seed_pairwise_accuracy",
    ):
        grouped[f"delta_{column}"] = grouped[column] - float(baseline[column])
    return grouped


def select(summary: pd.DataFrame) -> str:
    """Same rule the other seed studies use: ranking is a guard, Brier decides."""
    baseline = summary.loc[summary["candidate_key"].eq(BASELINE_KEY)].iloc[0]
    eligible = summary.loc[
        summary["seed_spearman"].ge(float(baseline["seed_spearman"]) - 0.002)
        & summary["seed_pairwise_accuracy"].ge(
            float(baseline["seed_pairwise_accuracy"]) - 0.002
        )
    ]
    if eligible.empty:
        return BASELINE_KEY
    return str(
        eligible.sort_values(
            ["brier_1x2", "log_loss_1x2", "candidate_key"], kind="stable"
        ).iloc[0]["candidate_key"]
    )


def nested_selection(surface: pd.DataFrame, folds) -> pd.DataFrame:
    rows = []
    for fold, (train_seasons, test_season) in enumerate(folds, start=1):
        train = surface.loc[surface["season"].isin(train_seasons)]
        chosen = select(summarise(train))
        rows.append(
            {
                "fold": fold,
                "train_seasons": "|".join(train_seasons),
                "test_season": test_season,
                "selected_candidate_key": chosen,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep the Domestic Prior axes one at a time"
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract_hash = sha256(PRODUCTION_CONTRACT)

    print("Loading frozen production replay", flush=True)
    _, datasets, core, parameters, production_seed_map = load_production_baseline()
    seasons = tuple(data.season for data in datasets)
    folds = expanding_folds(seasons)
    target_by_competition = schedule_adjusted_team_performance(read_events(EVENTS_PATH))
    target = aggregate_target(target_by_competition)
    xg_map = load_xg_map(XG_DATA, datasets)
    surprise = load_domestic_surprise()

    configs = candidate_grid()
    predictions: dict[str, pd.DataFrame] = {}
    candidate_seeds: dict[str, pd.DataFrame] = {}
    surface_rows: list[dict[str, object]] = []

    for index, candidate in enumerate(configs, start=1):
        seeds = seed_frame(candidate.config, surprise)
        rating_map = {
            (str(row.season), int(row.team_id)): float(row.ao_first_elo)
            for row in seeds.itertuples(index=False)
        }
        if candidate.key == BASELINE_KEY:
            # The baseline must reproduce the seeds the frozen replay was scored
            # on, or every delta below is measured against a model we do not
            # serve.
            error = max(
                abs(rating_map[key] - value)
                for key, value in production_seed_map.items()
            )
            if error > 1e-8:
                raise ValueError(f"Baseline does not reproduce production: {error}")
        evaluation = evaluate_arm(
            datasets,
            EvaluationArm(candidate.key, True, True, True, True, True),
            core=core,
            parameters=parameters,
            current_domestic=rating_map,
            baseline_domestic=production_seed_map,
            xg_map=xg_map,
            target=target_by_competition,
        )
        predictions[candidate.key] = evaluation.predictions
        candidate_seeds[candidate.key] = seeds.rename(
            columns={"ao_first_elo": "candidate_ao_first_elo"}
        )
        surface_rows.extend(
            season_rows(candidate, seeds, evaluation.predictions, target, seasons)
        )
        print(f"Replayed {index}/{len(configs)} candidates", flush=True)

    surface = pd.DataFrame(surface_rows)
    summary_frame = summarise(surface)
    selections = nested_selection(surface, folds)
    unseen, fold_results = build_unseen(
        selections, predictions, candidate_seeds, target, folds
    )
    comparison = model_summary(unseen, fold_results)
    competition = competition_summary(unseen)
    uncertainty = uncertainty_summary(unseen, int(args.bootstrap_samples))
    ranking_uncertainty = ranking_uncertainty_summary(
        fold_results, int(args.bootstrap_samples)
    )
    full_selected = select(summary_frame)

    selected = comparison.loc[comparison["model"].eq("NESTED_RECALIBRATION")].iloc[0]
    brier_wins = int((fold_results["delta_brier_1x2"] < 0.0).sum())
    log_wins = int((fold_results["delta_log_loss_1x2"] < 0.0).sum())
    payload = {
        "decision": (
            "PROMOTE_CANDIDATE"
            if brier_wins >= 4
            and log_wins >= 4
            and selected["delta_vs_baseline_brier_1x2"] < 0.0
            and selected["delta_vs_baseline_log_loss_1x2"] < 0.0
            and not bool(ranking_uncertainty["reliable_harm"].any())
            and not bool(
                (competition["delta_brier_1x2"] > 0.0).any()
                or (competition["delta_log_loss_1x2"] > 0.0).any()
            )
            else "KEEP_PRODUCTION"
        ),
        "production_changed": False,
        "candidate_count": len(configs),
        "fold_count": len(folds),
        "unseen_matches": int(comparison["matches"].max()),
        "full_history_candidate_key": full_selected,
        "nested_selected_keys": selections["selected_candidate_key"].tolist(),
        "brier_fold_wins": brier_wins,
        "log_loss_fold_wins": log_wins,
        "ranking_reliable_harm": bool(ranking_uncertainty["reliable_harm"].any()),
        "pooled_brier_delta": float(selected["delta_vs_baseline_brier_1x2"]),
        "pooled_seed_spearman_delta": float(
            selected["delta_vs_baseline_seed_spearman"]
        ),
        "production_contract_sha256": contract_hash,
    }

    # The per-match frame is what any segment question needs afterwards - a
    # competition-level point estimate cannot be given a confidence interval
    # from the summary alone.
    unseen.to_csv(output / "unseen_predictions.csv", index=False)
    surface.to_csv(output / "candidate_surface.csv", index=False)
    summary_frame.to_csv(output / "candidate_summary.csv", index=False)
    selections.to_csv(output / "fold_selections.csv", index=False)
    fold_results.to_csv(output / "fold_results.csv", index=False)
    comparison.to_csv(output / "model_comparison.csv", index=False)
    competition.to_csv(output / "competition_summary.csv", index=False)
    uncertainty.to_csv(output / "dependency_uncertainty.csv", index=False)
    ranking_uncertainty.to_csv(output / "ranking_uncertainty.csv", index=False)
    (output / "decision.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    if sha256(PRODUCTION_CONTRACT) != contract_hash:
        raise ValueError("Production contract changed during the sweep")
    print(f"Decision: {payload['decision']}")
    print(f"Full-history candidate: {full_selected}")


if __name__ == "__main__":
    main()
