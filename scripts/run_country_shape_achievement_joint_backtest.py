from __future__ import annotations

"""Joint sweep of the two Domestic Prior axes that share one channel.

    L  = ( ln(1 + score) / ln(1 + season_max) ) ** gamma
    DP = 500 + league_component * L
             + achievement_component * A * (alpha + (1 - alpha) * L)

`gamma` and `achievement_alpha` were each measured alone and each promoted:
gamma 1.6 on a season-relative anchor, and alpha 0.0 on the production anchor.
But they are not independent. Raising gamma lowers L, which lowers
`alpha + (1 - alpha) * L`, so gamma already reaches the achievement term
through the same multiplier alpha governs. Lowering alpha does the same thing
from the other side.

Measured apart, both say "let league strength speak louder." Applied together
the gains may add, overlap, or overshoot - the single-axis numbers cannot
distinguish those, and taking both on the strength of separate sweeps would be
the error this run exists to avoid.

Research only: the active contract is hashed before and after, and nothing in
`contracts/` or `artifacts/` is written.
"""

import argparse
from dataclasses import replace
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
from scripts.run_ao_first_seed_boost_backtest import aggregate_target  # noqa: E402
from scripts.run_current_model_evaluation import (  # noqa: E402
    EvaluationArm,
    evaluate_arm,
)
from scripts.run_domestic_prior_axis_backtest import (  # noqa: E402
    BASELINE_KEY,
    build_unseen,
    load_domestic_surprise,
    nested_selection,
    season_rows,
    select,
    summarise,
)
from scripts.run_dynamic_core_calibration import expanding_folds  # noqa: E402
from scripts.run_european_prior_recalibration_backtest import (  # noqa: E402
    competition_summary,
    model_summary,
    sha256,
    uncertainty_summary,
)
from scripts.run_opponent_quintile_backtest import load_production_baseline  # noqa: E402
# The anchor-aware copies: `run_relative_country_benchmark_backtest` predates
# the anchor axis and its Candidate has no `anchor` field.
from scripts.run_country_anchor_backtest import (  # noqa: E402
    Candidate,
    relative_seed_frame,
)
from scripts.run_stage_weighted_progression_backtest import (  # noqa: E402
    EVENTS_PATH,
    PRODUCTION_CONTRACT,
    XG_DATA,
    load_xg_map,
)
from scripts.run_v2_evaluation_upgrade import read_events  # noqa: E402


OUTPUT_ROOT = ROOT / "output" / "country_shape_achievement_joint_2018_2026"
ACTIVE = AOEuropeanEloConfig.active()
# Brackets each axis's own optimum on both sides so the joint surface can show
# whether the two pull apart once they are allowed to move together.
GAMMAS = (1.2, 1.6, 2.0)
ALPHAS = (0.0, 0.2, 0.4)


def candidate_grid() -> tuple[Candidate, ...]:
    return (
        Candidate(BASELINE_KEY, "BASELINE", 0.0, ACTIVE, False, "max"),
        *(
            Candidate(
                f"g{gamma:g}_a{alpha:g}",
                "JOINT",
                gamma,
                replace(ACTIVE, gamma=gamma, achievement_alpha=alpha),
                True,
                "max",
            )
            for gamma in GAMMAS
            for alpha in ALPHAS
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep gamma and achievement_alpha together"
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
        seeds = relative_seed_frame(
            candidate.config, surprise, candidate.relative, candidate.anchor
        )
        rating_map = {
            (str(row.season), int(row.team_id)): float(row.ao_first_elo)
            for row in seeds.itertuples(index=False)
        }
        if candidate.key == BASELINE_KEY:
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
        rows = season_rows(candidate, seeds, evaluation.predictions, target, seasons)
        for row in rows:
            row["gamma"] = float(candidate.config.gamma)
            row["alpha"] = float(candidate.config.achievement_alpha)
        surface_rows.extend(rows)
        print(f"Replayed {index}/{len(configs)} candidates", flush=True)

    surface = pd.DataFrame(surface_rows)
    summary_frame = summarise(surface)
    lookup = (
        surface[["candidate_key", "gamma", "alpha"]]
        .drop_duplicates("candidate_key")
        .set_index("candidate_key")
    )
    summary_frame = summary_frame.join(lookup, on="candidate_key")
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
    chosen = selections["selected_candidate_key"].tolist()
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
        "nested_selected_keys": chosen,
        "same_candidate_every_fold": len(set(chosen)) == 1,
        "brier_fold_wins": brier_wins,
        "log_loss_fold_wins": log_wins,
        "ranking_reliable_harm": bool(ranking_uncertainty["reliable_harm"].any()),
        "pooled_brier_delta": float(selected["delta_vs_baseline_brier_1x2"]),
        "pooled_seed_spearman_delta": float(
            selected["delta_vs_baseline_seed_spearman"]
        ),
        "production_contract_sha256": contract_hash,
    }

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
    print(f"Same candidate every fold: {payload['same_candidate_every_fold']}")


if __name__ == "__main__":
    main()
