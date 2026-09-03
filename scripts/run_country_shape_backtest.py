from __future__ import annotations

"""Joint walk-forward sweep of the two parameters that shape league strength.

    L = (ln(1 + weighted_country_score) / ln(1 + benchmark)) ** gamma

`benchmark` sets where the log curve saturates and `gamma` bends what comes out
of it, so neither is interpretable alone: lower the benchmark and every raw
normalization rises, which leaves a larger gamma doing the same amount of
compression. The single-axis sweep in `run_domestic_prior_axis_backtest.py`
found gamma 1.4 optimal, but only with the benchmark pinned at production's 25.

This asks whether that optimum survives when both move. The earlier joint study
(`run_ranking_first_calibration.py`) is the reason to doubt it: its folds agreed
gamma should rise but scattered across benchmarks 15 to 35, and its stability
gate refused to promote anything as a result.

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
    Candidate,
    build_unseen,
    load_domestic_surprise,
    nested_selection,
    season_rows,
    seed_frame,
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
from scripts.run_stage_weighted_progression_backtest import (  # noqa: E402
    EVENTS_PATH,
    PRODUCTION_CONTRACT,
    XG_DATA,
    load_xg_map,
)
from scripts.run_v2_evaluation_upgrade import read_events  # noqa: E402


OUTPUT_ROOT = ROOT / "output" / "country_shape_backtest_2018_2026"
ACTIVE = AOEuropeanEloConfig.active()
# The same benchmark range the earlier joint study used, so a disagreement
# between the two can be read as a disagreement about method rather than reach.
BENCHMARKS = (15.0, 20.0, 25.0, 30.0, 35.0)
# gamma 2.0 stays in: a lower benchmark lifts every raw normalization towards 1,
# where the exponent bites less, so the optimum should drift upward as the
# benchmark falls. Dropping it would hide exactly that drift.
GAMMAS = (1.0, 1.2, 1.4, 1.6, 2.0)


def candidate_grid() -> tuple[Candidate, ...]:
    return (
        Candidate(BASELINE_KEY, "BASELINE", 0.0, ACTIVE),
        *(
            Candidate(
                f"b{benchmark:g}_g{gamma:g}",
                "COUNTRY_SHAPE",
                gamma,
                replace(
                    ACTIVE,
                    country_strength_benchmark=benchmark,
                    gamma=gamma,
                ),
            )
            for benchmark in BENCHMARKS
            for gamma in GAMMAS
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep the country-strength benchmark and gamma jointly"
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
            row["benchmark"] = float(candidate.config.country_strength_benchmark)
            row["gamma"] = float(candidate.config.gamma)
        surface_rows.extend(rows)
        print(f"Replayed {index}/{len(configs)} candidates", flush=True)

    surface = pd.DataFrame(surface_rows)
    summary_frame = summarise(surface)
    # `summarise` groups on the axis columns, so carry the two real parameters
    # back onto it for reading; the key already encodes them uniquely.
    lookup = (
        surface[["candidate_key", "benchmark", "gamma"]]
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
        # The earlier joint study failed on exactly this: folds agreed on the
        # direction but not on the configuration, so record it explicitly.
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
