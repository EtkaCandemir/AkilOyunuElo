from __future__ import annotations

"""Does correcting the European history for who a club actually played help?

    weighted_european_history = sum_k season_weight[k] * club_points[k]

`club_points` is the club's raw UEFA coefficient haul. UEFA pays the same for a
win whoever it came against, so two clubs on the same points can have earned
them against very different opposition - and the European Prior, which is a
pure function of that history, cannot tell them apart.

This run scales each season's points by how hard that season's fixtures were:

    fixture_delta = schedule_adjusted_score_unshrunk - raw_score_rate
                  = (venue correction) + (mean_opponent_strength - 0.5)
    adjusted_points[k] = club_points[k] * max(0, 1 + kappa * fixture_delta[k])

`kappa = 0` is production exactly, and the sweep opens from there. The delta is
matches-weighted across competitions, because a club relegated from the
Champions League into the Europa League has one points total and two schedules.

Only `club_points_<slot>` is touched. `played_<slot>`, `matches_<slot>` and
`match_cap_<slot>` are left alone, so European Exposure - and therefore the
blend weight - is identical across every candidate and the history is the only
thing that moves. A guard checks that.

Two things to know before reading the numbers:

- **Spearman shares its form with the target.** The ranking metric is built by
  `schedule_adjusted_team_performance`, and so is this input. The seasons are
  disjoint (the seed for S reads S-5..S-1, the target is season S), so this is
  not leakage - but past-target predicting present-target enjoys a same-form
  advantage that raw UEFA points do not get. **Brier is the arbiter here**, and
  a Spearman gain on its own means nothing.
- **The delta partly rewards depth.** Opponent strength is measured within the
  season and competition, so a club that survives qualifying and meets the
  clubs who also survived scores a hard schedule. UEFA points already pay for
  going deep. If the adjustment mostly double-counts progression rather than
  correcting the draw, kappa > 0 will not help, and that is the answer.

Coverage is partial and deliberately so. Match data starts in `2018/19`, so a
seed season's early slots may have no fixtures to measure; those slots get
`delta = 0`, which is production behaviour. Rather than dropping the folds that
cannot be fully covered, every fold runs and the per-fold coverage is recorded,
so the later folds - which are fully covered - can be read on their own.

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
from ao_elo.pipeline import compute_ao_first_elo  # noqa: E402
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
    select,
    summarise,
)
from scripts.run_dynamic_core_calibration import expanding_folds  # noqa: E402
from scripts.run_european_prior_recalibration_backtest import (  # noqa: E402
    STATIC_DATA_ROOT,
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


OUTPUT_ROOT = ROOT / "output" / "fixture_adjusted_european_history_2018_2026"
ACTIVE = AOEuropeanEloConfig.active()
# How many seasons back each history slot sits from the seed season. Verified
# empirically against the match data: for every seed season and slot, the clubs
# `played_<slot>` marks as having played are a subset of the clubs that appear
# in the mapped season's fixtures. A one-year error breaks that containment.
SLOT_OFFSETS = {
    "t": 1,
    "t_minus_1": 2,
    "t_minus_2": 3,
    "t_minus_3": 4,
    "t_minus_4": 5,
}
# The observed delta spans about [-0.21, +0.25], so kappa 3.0 already reaches a
# +/-75% swing on a season's points. Wider is not worth measuring: it would
# scale points by a number that can approach zero.
KAPPAS = (0.5, 1.0, 1.5, 2.0, 3.0)


def slot_season(seed_year: int, slot: str) -> str:
    year = seed_year - SLOT_OFFSETS[slot]
    return f"{year}/{str(year + 1)[-2:]}"


def fixture_delta_table() -> pd.DataFrame:
    """Per club-season fixture difficulty, matches-weighted across competitions."""
    performance = schedule_adjusted_team_performance(read_events(EVENTS_PATH))
    performance["fixture_delta"] = (
        performance["schedule_adjusted_score_unshrunk"]
        - performance["raw_score_rate"]
    )
    weighted = performance.assign(
        weighted=performance["fixture_delta"] * performance["matches"]
    )
    table = weighted.groupby(["season", "team_id"], as_index=False).agg(
        weighted=("weighted", "sum"), matches=("matches", "sum")
    )
    table["fixture_delta"] = table["weighted"] / table["matches"]
    return table[["season", "team_id", "fixture_delta"]]


def adjust_points(
    points: pd.DataFrame,
    deltas: pd.DataFrame,
    seed_year: int,
    kappa: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Scale each slot's club points by that slot's fixture difficulty."""
    frame = points.copy()
    covered = {season for season in deltas["season"].unique()}
    lookup = {
        (str(row.season), int(row.team_id)): float(row.fixture_delta)
        for row in deltas.itertuples(index=False)
    }
    adjusted_slots = 0
    adjusted_rows = 0
    for slot in SLOT_OFFSETS:
        season = slot_season(seed_year, slot)
        column = f"club_points_{slot}"
        if season not in covered:
            continue
        adjusted_slots += 1
        delta = frame["team_id"].astype(int).map(
            lambda team_id, season=season: lookup.get((season, team_id), 0.0)
        )
        # A club with no fixtures that season has delta 0 and is left as it was.
        adjusted_rows += int((delta != 0.0).sum())
        multiplier = (1.0 + kappa * delta).clip(lower=0.0)
        frame[column] = frame[column].astype(float) * multiplier
    return frame, {
        "slots_adjusted": float(adjusted_slots),
        "slot_rows_adjusted": float(adjusted_rows),
    }


def seed_frame(
    kappa: float,
    deltas: pd.DataFrame,
    surprise: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Recompute AO First per season from fixture-adjusted European points."""
    rows = []
    coverage: list[dict[str, object]] = []
    for folder in sorted(STATIC_DATA_ROOT.glob("20??-??")):
        seed_year = int(folder.name[:4])
        season = f"{seed_year}/{str(seed_year + 1)[-2:]}"
        points = pd.read_csv(folder / "club_european_points.csv")
        adjusted, stats = adjust_points(points, deltas, seed_year, kappa)
        rating = compute_ao_first_elo(
            teams=pd.read_csv(folder / "teams.csv"),
            country_coefficients=pd.read_csv(folder / "country_coefficients.csv"),
            domestic_context=pd.read_csv(folder / "domestic_context.csv"),
            club_european_points=adjusted,
            config=ACTIVE,
        )
        coverage.append({"season": season, "kappa": kappa, **stats})
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
    return frame, coverage


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep a fixture-difficulty correction on the European history"
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
    deltas = fixture_delta_table()
    print(f"Fixture deltas for {len(deltas)} club-seasons", flush=True)

    configs = (
        Candidate(BASELINE_KEY, "BASELINE", 0.0, ACTIVE),
        *(
            Candidate(f"kappa{value:g}", "FIXTURE_KAPPA", value, ACTIVE)
            for value in KAPPAS
        ),
    )
    predictions: dict[str, pd.DataFrame] = {}
    candidate_seeds: dict[str, pd.DataFrame] = {}
    surface_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    baseline_exposure: pd.Series | None = None

    for index, candidate in enumerate(configs, start=1):
        seeds, coverage = seed_frame(candidate.value, deltas, surprise)
        coverage_rows.extend(coverage)
        exposure = seeds.set_index(["season", "team_id"])[
            "effective_european_exposure"
        ].sort_index()
        if candidate.key == BASELINE_KEY:
            baseline_exposure = exposure
            error = max(
                abs(
                    float(row.ao_first_elo)
                    - production_seed_map[(str(row.season), int(row.team_id))]
                )
                for row in seeds.itertuples(index=False)
            )
            if error > 1e-8:
                raise ValueError(f"kappa 0 does not reproduce production: {error}")
        else:
            # The whole design rests on the blend weight being untouched, so it
            # is checked rather than assumed.
            drift = float((exposure - baseline_exposure).abs().max())
            if drift > 0.0:
                raise ValueError(
                    f"{candidate.key} moved European Exposure by {drift}"
                )
        rating_map = {
            (str(row.season), int(row.team_id)): float(row.ao_first_elo)
            for row in seeds.itertuples(index=False)
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
    chosen = selections["selected_candidate_key"].tolist()
    coverage_frame = pd.DataFrame(coverage_rows)
    full_coverage = sorted(
        str(season)
        for season in coverage_frame.loc[
            coverage_frame["slots_adjusted"].eq(len(SLOT_OFFSETS)), "season"
        ].unique()
    )
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
        "fully_covered_seed_seasons": full_coverage,
        "exposure_held_fixed": True,
        "spearman_shares_form_with_target": (
            "The input and the ranking target are both built by "
            "schedule_adjusted_team_performance, on disjoint seasons. Brier is "
            "the arbiter; a Spearman gain alone is not evidence."
        ),
        "production_contract_sha256": contract_hash,
    }

    unseen.to_csv(output / "unseen_predictions.csv", index=False)
    surface.to_csv(output / "candidate_surface.csv", index=False)
    summary_frame.to_csv(output / "candidate_summary.csv", index=False)
    coverage_frame.to_csv(output / "slot_coverage.csv", index=False)
    deltas.to_csv(output / "fixture_deltas.csv", index=False)
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
