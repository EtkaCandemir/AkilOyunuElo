from __future__ import annotations

"""Walk-forward test of a match-volume definition of European exposure.

Production defines the blend weight as a recency-weighted share of the last
five seasons:

    exposure = 0.6 * sum(w_s * played_s) + 0.4 * sum(w_s * min(1, matches_s / cap_s))
    w = (0.07, 0.13, 0.20, 0.27, 0.33) for t-4 .. t

Two things about that are worth testing rather than assuming.

First, `season_weights` is the vector built for the *history* component, where
recency is obviously right: a recent result says more about a club than an old
one. Exposure reuses it verbatim for a different job - how confident we are -
and confidence comes from sample size. A match played four years ago is still
an observed match; it reduces uncertainty about as much as a recent one, even
though its result should count for less.

Second, the sub-cap region has never been studied. Every exposure sweep to date
moved `max_european_exposure`, which by construction only touches clubs above
it: of the 24 clubs below the cap in 2026/27, none are affected by any cap value
between 0.65 and 0.85, and only 13 by a cap as low as 0.40.

This command replaces the exposure definition with raw evidence volume,

    exposure = min(1, total_european_matches / benchmark)

leaving the 0.65 cap, the European Prior, the participation shrinkage and the
tail exactly as production has them. One axis, measured on its own.

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

from ao_elo.european_prior_recalibration import (  # noqa: E402
    EuropeanPriorRecalibrationConfig,
    apply_european_prior_recalibration,
    ranking_uncertainty_summary,
)
from ao_elo.evaluation import schedule_adjusted_team_performance  # noqa: E402
from ao_elo.features import SEASON_KEYS  # noqa: E402
from scripts.run_ao_first_seed_boost_backtest import aggregate_target  # noqa: E402
from scripts.run_current_model_evaluation import (  # noqa: E402
    EvaluationArm,
    evaluate_arm,
)
from scripts.run_dynamic_core_calibration import expanding_folds  # noqa: E402
from scripts.run_european_prior_recalibration_backtest import (  # noqa: E402
    BASELINE_CONFIG,
    aggregate_candidate_surface,
    build_unseen,
    competition_summary,
    load_seed_evidence,
    model_summary,
    nested_selection,
    season_surface,
    select_candidate,
    sha256,
    uncertainty_summary,
)
from scripts.run_opponent_quintile_backtest import load_production_baseline  # noqa: E402
from scripts.run_stage_weighted_progression_backtest import (  # noqa: E402
    EVENTS_PATH,
    PRODUCTION_CONTRACT,
    STATIC_DATA_ROOT,
    XG_DATA,
    load_xg_map,
)
from scripts.run_v2_evaluation_upgrade import read_events  # noqa: E402


OUTPUT_ROOT = ROOT / "output" / "match_volume_exposure_backtest_2018_2026"
# A club that plays every season accumulates roughly 50-64 European matches over
# the window, so the grid brackets that from well below to well above: at 30 a
# regular participant saturates early, at 70 even Bayern stays under the cap.
MATCH_BENCHMARKS = (30.0, 40.0, 50.0, 60.0, 70.0)
# The shared selection helper identifies the baseline by BASELINE_CONFIG.key,
# so the production candidate has to carry exactly that name.
PRODUCTION_KEY = BASELINE_CONFIG.key


@dataclass(frozen=True)
class Candidate:
    """A production config plus the exposure column it should be scored with.

    Delegates every config attribute so the shared surface helpers, which read
    `config.history_benchmark` and friends, keep working unchanged.
    """

    key: str
    benchmark: float | None
    config: EuropeanPriorRecalibrationConfig

    def __getattr__(self, item: str) -> object:
        return getattr(self.config, item)


def candidate_grid() -> tuple[Candidate, ...]:
    return (
        Candidate(PRODUCTION_KEY, None, BASELINE_CONFIG),
        *(
            Candidate(f"matches{benchmark:g}", benchmark, BASELINE_CONFIG)
            for benchmark in MATCH_BENCHMARKS
        ),
    )


def load_match_volume() -> pd.DataFrame:
    """Total European matches per club-season over the same five-season window."""
    rows = []
    for folder in sorted(STATIC_DATA_ROOT.glob("20??-??")):
        points = pd.read_csv(folder / "club_european_points.csv")
        total = sum(
            points[f"matches_{key}"].fillna(0.0).astype(float) for key in SEASON_KEYS
        )
        rows.append(
            pd.DataFrame(
                {
                    "season": points["season"].astype(str),
                    "team_id": points["team_id"].astype(int),
                    "total_european_matches": total,
                }
            )
        )
    frame = pd.concat(rows, ignore_index=True)
    if frame.duplicated(["season", "team_id"]).any():
        raise ValueError("Match volume is not unique per club-season")
    return frame


def seeds_for(
    seeds: pd.DataFrame, volume: pd.DataFrame, benchmark: float | None
) -> pd.DataFrame:
    if benchmark is None:
        return seeds
    merged = seeds.merge(volume, on=["season", "team_id"], how="left", validate="one_to_one")
    if merged["total_european_matches"].isna().any():
        raise ValueError("Match volume does not cover every seeded club-season")
    exposure = (merged["total_european_matches"] / float(benchmark)).clip(upper=1.0)
    # A club with no European matches must stay at zero exposure, exactly as
    # production does: its European Prior is the 500 base placeholder, not
    # evidence, and letting any weight reach it would be a silent corruption.
    exposure = exposure.where(merged["european_exposure"] > 0.0, 0.0)
    return merged.assign(european_exposure=exposure).drop(
        columns=["total_european_matches"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure a match-volume definition of European exposure"
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples must be at least 100")
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract_hash = sha256(PRODUCTION_CONTRACT)

    print("Loading frozen production replay and static seed evidence", flush=True)
    _, datasets, core, parameters, production_seed_map = load_production_baseline()
    seasons = tuple(data.season for data in datasets)
    folds = expanding_folds(seasons)
    events = read_events(EVENTS_PATH)
    target_by_competition = schedule_adjusted_team_performance(events)
    target = aggregate_target(target_by_competition)
    xg_map = load_xg_map(XG_DATA, datasets)
    seeds = load_seed_evidence(production_seed_map)
    volume = load_match_volume()

    configs = candidate_grid()
    predictions: dict[str, pd.DataFrame] = {}
    candidate_seeds: dict[str, pd.DataFrame] = {}
    surface_rows: list[dict[str, object]] = []
    reach_rows: list[dict[str, object]] = []
    base_frame = apply_european_prior_recalibration(seeds, BASELINE_CONFIG)

    for index, candidate in enumerate(configs, start=1):
        frame = seeds_for(seeds, volume, candidate.benchmark)
        scored = apply_european_prior_recalibration(frame, candidate.config)
        # The production baseline must reproduce the live seed exactly, or every
        # delta below is measured against something we do not serve.
        if candidate.benchmark is None:
            error = (
                scored["candidate_ao_first_elo"] - scored["adjusted_ao_first_elo"]
            ).abs().max()
            if float(error) > 1e-8:
                raise ValueError(f"Baseline does not reproduce production: {error}")
        candidate_seeds[candidate.key] = scored
        rating_map = {
            (str(row.season), int(row.team_id)): float(row.candidate_ao_first_elo)
            for row in scored.itertuples(index=False)
        }
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
        rows = season_surface(
            candidate, scored, evaluation.predictions, target, seasons
        )
        for row in rows:
            row["candidate_key"] = candidate.key
            row["match_benchmark"] = candidate.benchmark or 0.0
        surface_rows.extend(rows)
        moved = scored["candidate_ao_first_elo"] - base_frame["candidate_ao_first_elo"]
        reach_rows.append(
            {
                "candidate_key": candidate.key,
                "match_benchmark": candidate.benchmark or 0.0,
                "club_seasons_moved": int((moved.abs() > 1e-9).sum()),
                "club_seasons_total": int(len(moved)),
                "max_rise_elo": float(moved.max()),
                "max_fall_elo": float(moved.min()),
            }
        )
        print(f"Replayed {index}/{len(configs)} candidates", flush=True)

    surface = pd.DataFrame(surface_rows)
    selections = nested_selection(surface, configs, folds)
    unseen, fold_results = build_unseen(
        selections, predictions, candidate_seeds, target, folds
    )
    summary = model_summary(unseen, fold_results)
    competition = competition_summary(unseen)
    uncertainty = uncertainty_summary(unseen, int(args.bootstrap_samples))
    ranking_uncertainty = ranking_uncertainty_summary(
        fold_results, int(args.bootstrap_samples)
    )
    candidate_summary = aggregate_candidate_surface(surface)
    full_selected = select_candidate(candidate_summary)

    selected = summary.loc[summary["model"].eq("NESTED_RECALIBRATION")].iloc[0]
    brier_wins = int((fold_results["delta_brier_1x2"] < 0.0).sum())
    log_wins = int((fold_results["delta_log_loss_1x2"] < 0.0).sum())
    rank_harm = bool(ranking_uncertainty["reliable_harm"].any())
    segment_harm = bool(
        (competition["delta_brier_1x2"] > 0.0).any()
        or (competition["delta_log_loss_1x2"] > 0.0).any()
    )
    payload = {
        "decision": (
            "PROMOTE_CANDIDATE"
            if brier_wins >= 4
            and log_wins >= 4
            and selected["delta_vs_baseline_brier_1x2"] < 0.0
            and selected["delta_vs_baseline_log_loss_1x2"] < 0.0
            and not rank_harm
            and not segment_harm
            else "KEEP_PRODUCTION"
        ),
        "production_changed": False,
        "candidate_count": len(configs),
        "fold_count": len(folds),
        "unseen_matches": int(summary["matches"].max()),
        "full_history_candidate_key": full_selected,
        "nested_selected_keys": selections["selected_candidate_key"].tolist(),
        "brier_fold_wins": brier_wins,
        "log_loss_fold_wins": log_wins,
        "ranking_reliable_harm": rank_harm,
        "competition_loss_harm": segment_harm,
        "pooled_brier_delta": float(selected["delta_vs_baseline_brier_1x2"]),
        "pooled_log_loss_delta": float(selected["delta_vs_baseline_log_loss_1x2"]),
        "pooled_seed_spearman_delta": float(
            selected["delta_vs_baseline_seed_spearman"]
        ),
        "pooled_seed_pairwise_delta": float(
            selected["delta_vs_baseline_seed_pairwise_accuracy"]
        ),
        "production_contract_sha256": contract_hash,
    }

    surface.to_csv(output / "candidate_surface.csv", index=False)
    candidate_summary.to_csv(output / "candidate_summary.csv", index=False)
    pd.DataFrame(reach_rows).to_csv(output / "exposure_reach.csv", index=False)
    selections.to_csv(output / "fold_selections.csv", index=False)
    fold_results.to_csv(output / "fold_results.csv", index=False)
    summary.to_csv(output / "model_comparison.csv", index=False)
    competition.to_csv(output / "competition_summary.csv", index=False)
    uncertainty.to_csv(output / "dependency_uncertainty.csv", index=False)
    ranking_uncertainty.to_csv(output / "ranking_uncertainty.csv", index=False)
    (output / "decision.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    if sha256(PRODUCTION_CONTRACT) != contract_hash:
        raise ValueError("Production contract changed during research backtest")
    print(f"Decision: {payload['decision']}")
    print(f"Full-history candidate: {full_selected}")


if __name__ == "__main__":
    main()
