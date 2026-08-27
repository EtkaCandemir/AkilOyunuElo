from __future__ import annotations

"""Nested, ranking-first research of a narrow AO First Elo seed boost.

The script hashes the production contract before doing any work and verifies
the same bytes at the end.  It never writes a production model file.
"""

import argparse
import hashlib
import itertools
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ao_elo.ao_first_seed_boost import (  # noqa: E402
    BASELINE_KEY,
    BOOST_BLIND,
    BOOST_DOMESTIC_FORM,
    BOOST_LONG_HISTORY,
    SeedBoostConfig,
    apply_seed_boost,
    candidate_grid,
    season_relative_percentile,
)
from ao_elo.domestic_poisson import (  # noqa: E402
    DomesticPoissonConfig,
    DynamicDomesticPoisson,
    _domestic_records,
    _mapped_snapshot,
    _validate_domestic_matches,
    _validated_bridge,
)
from ao_elo.evaluation import (  # noqa: E402
    dependency_robust_loss_difference_ci,
    schedule_adjusted_team_performance,
)
from scripts.run_current_external_benchmark import (  # noqa: E402
    build_rating_table,
    paired_spearman_difference_ci,
)
from scripts.run_current_model_evaluation import (  # noqa: E402
    EvaluationArm,
    evaluate_arm,
)
from scripts.run_dynamic_core_calibration import expanding_folds  # noqa: E402
from scripts.run_external_ranking_comparison_2025_26 import pairwise_accuracy  # noqa: E402
from scripts.run_final_robustness import (  # noqa: E402
    load_team_season_identity,
    summarize_ranking,
)
from scripts.run_opponent_quintile_backtest import load_production_baseline  # noqa: E402
from scripts.run_stage_weighted_progression_backtest import (  # noqa: E402
    DOMESTIC_ADJUSTMENTS,
    EVENTS_PATH,
    PRODUCTION_CONTRACT,
    XG_DATA,
    load_xg_map,
)
from scripts.run_v2_evaluation_upgrade import read_events  # noqa: E402


SURPRISE_FEATURES = (
    ROOT
    / "output"
    / "domestic_surprise_variance_backtest_2018_2026"
    / "domestic_surprise_features.csv"
)
DOMESTIC_MATCHES = (
    ROOT / "data" / "domestic_league_matches_2013_2026" / "domestic_matches.csv"
)
DOMESTIC_BRIDGE = (
    ROOT
    / "data"
    / "domestic_league_matches_2013_2026"
    / "domestic_team_bridge.csv"
)
TEAM_IDENTITY = ROOT / "data" / "club_identity" / "team_season_identity.csv"
MODEL_RATINGS = (
    ROOT / "output" / "current_model_evaluation_2018_2026" / "model_end_ratings.csv"
)
OPTA_SNAPSHOT = (
    ROOT
    / "data"
    / "external_rankings_2025_26"
    / "opta_power_rankings_2025_07_03_ao_scope.csv"
)
CLUB_COEFFICIENTS = (
    ROOT
    / "data"
    / "backtest_stage_b_2018_2026"
    / "2025-26"
    / "club_european_points.csv"
)
OUTPUT_ROOT = ROOT / "output" / "ao_first_seed_boost_backtest_2018_2026"

EVALUATION_SEASONS = (
    "2020/21",
    "2021/22",
    "2022/23",
    "2023/24",
    "2024/25",
    "2025/26",
)
ARM_ORDER = (BASELINE_KEY, BOOST_BLIND, BOOST_DOMESTIC_FORM)
BOOTSTRAP_SEED = 20260820
DEFAULT_BOOTSTRAP_SAMPLES = 2000


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nested ranking-first test of a narrow AO First Elo seed boost"
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--production-contract", type=Path, default=PRODUCTION_CONTRACT)
    arguments = parser.parse_args()
    if arguments.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples must be at least 100")
    output = arguments.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract = arguments.production_contract.resolve()
    contract_hash = hashlib.sha256(contract.read_bytes()).hexdigest()

    print("Loading production replay and season-start seed evidence", flush=True)
    _, datasets, core, parameters, production_seed_map = load_production_baseline()
    seasons = tuple(data.season for data in datasets)
    folds = expanding_folds(seasons)
    if tuple(test for _, test in folds) != EVALUATION_SEASONS:
        raise ValueError("Unexpected outer fold seasons")
    events = read_events(EVENTS_PATH)
    target_by_competition = schedule_adjusted_team_performance(events)
    target = aggregate_target(target_by_competition)
    seeds = load_seed_features()
    form = build_domestic_form_snapshots(
        pd.read_csv(DOMESTIC_MATCHES, low_memory=False),
        pd.read_csv(DOMESTIC_BRIDGE, low_memory=False),
        pd.read_csv(TEAM_IDENTITY, low_memory=False),
        events,
    )
    seeds = seeds.merge(
        form,
        on=["season", "team_id", "club_id"],
        how="left",
        validate="one_to_one",
    )
    seeds["domestic_form_covered"] = seeds["domestic_form_covered"].fillna(False)
    seeds["domestic_form_percentile"] = seeds["domestic_form_percentile"].fillna(0.0)

    print("Selecting blind and domestic-form rules inside each outer fold", flush=True)
    surface, selections, nested_seed_audit = nested_selection(seeds, target, folds)
    rating_maps = build_nested_rating_maps(
        production_seed_map, nested_seed_audit, seasons
    )

    xg_map = load_xg_map(XG_DATA, datasets)
    evaluations = {}
    for arm in ARM_ORDER:
        print(f"Replaying {arm}", flush=True)
        evaluations[arm] = evaluate_arm(
            datasets,
            EvaluationArm(arm, True, True, True, True, True),
            core=core,
            parameters=parameters,
            current_domestic=rating_maps[arm],
            baseline_domestic=production_seed_map,
            xg_map=xg_map,
            target=target_by_competition,
        )

    fold_ranking = fold_seed_ranking(nested_seed_audit, target, folds)
    comparison = model_comparison(evaluations, fold_ranking)
    competition = competition_summary(evaluations, target_by_competition)
    exposure = exposure_summary(nested_seed_audit, target)
    impact = seed_impact_summary(nested_seed_audit)
    affected = affected_match_summary(evaluations, nested_seed_audit)
    forward = forward_ranking_summary(evaluations, target_by_competition, seasons)
    loss_uncertainty = loss_uncertainty_summary(
        evaluations, int(arguments.bootstrap_samples)
    )
    rank_uncertainty = ranking_uncertainty_summary(
        nested_seed_audit,
        target,
        samples=int(arguments.bootstrap_samples),
    )
    external = external_opta_summary(nested_seed_audit, int(arguments.bootstrap_samples))
    feasibility = long_history_feasibility()
    safety = safety_audit(
        nested_seed_audit,
        evaluations,
        contract,
        contract_hash,
    )
    decision = decide(
        comparison,
        rank_uncertainty,
        loss_uncertainty,
        external,
        selections,
        impact,
        forward,
        feasibility,
        contract_hash,
    )

    surface.to_csv(output / "candidate_surface.csv", index=False)
    selections.to_csv(output / "fold_selections.csv", index=False)
    nested_seed_audit.to_csv(output / "seed_impact.csv", index=False)
    form.to_csv(output / "domestic_form_snapshots.csv", index=False)
    fold_ranking.to_csv(output / "fold_ranking_results.csv", index=False)
    comparison.to_csv(output / "model_comparison.csv", index=False)
    competition.to_csv(output / "competition_summary.csv", index=False)
    exposure.to_csv(output / "exposure_segment_summary.csv", index=False)
    impact.to_csv(output / "seed_impact_summary.csv", index=False)
    affected.to_csv(output / "affected_match_summary.csv", index=False)
    forward.to_csv(output / "forward_ranking_summary.csv", index=False)
    loss_uncertainty.to_csv(output / "dependency_uncertainty.csv", index=False)
    rank_uncertainty.to_csv(output / "ranking_uncertainty.csv", index=False)
    external.to_csv(output / "external_opta_comparison.csv", index=False)
    feasibility.to_csv(output / "long_history_feasibility.csv", index=False)
    safety.to_csv(output / "safety_audit.csv", index=False)
    (output / "selected_candidate.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "backtest_report.md").write_text(
        build_report(
            decision,
            comparison,
            fold_ranking,
            selections,
            competition,
            exposure,
            impact,
            affected,
            forward,
            external,
            rank_uncertainty,
            loss_uncertainty,
            feasibility,
            safety,
        ),
        encoding="utf-8",
    )
    if hashlib.sha256(contract.read_bytes()).hexdigest() != contract_hash:
        raise ValueError("Production contract changed during research backtest")
    if not safety["passed"].all():
        raise ValueError("Seed boost safety audit failed")
    print(f"Decision: {decision['decision']}")
    print(f"Report: {output / 'backtest_report.md'}")


def aggregate_target(frame: pd.DataFrame) -> pd.DataFrame:
    result = (
        frame.assign(weighted=frame["schedule_adjusted_score"] * frame["matches"])
        .groupby(["season", "team_id"], as_index=False)
        .agg(weighted=("weighted", "sum"), matches=("matches", "sum"))
    )
    result["schedule_adjusted_score"] = result["weighted"] / result["matches"]
    return result[["season", "team_id", "matches", "schedule_adjusted_score"]]


def load_seed_features() -> pd.DataFrame:
    features = pd.read_csv(SURPRISE_FEATURES, low_memory=False)
    adjustments = pd.read_csv(DOMESTIC_ADJUSTMENTS, low_memory=False)
    columns = [
        "season",
        "team_id",
        "club_id",
        "current_direct_percentile",
        "domestic_prior",
        "european_prior",
        "effective_european_exposure",
    ]
    adjustment_columns = [
        "season",
        "team_id",
        "team_name",
        "country_code",
        "competition",
        "domestic_prior_adjustment",
        "adjusted_domestic_prior",
        "adjusted_ao_first_elo",
    ]
    result = features[columns].merge(
        adjustments[adjustment_columns],
        on=["season", "team_id"],
        validate="one_to_one",
    )
    if len(result) != 1887 or result.duplicated(["season", "team_id"]).any():
        raise ValueError("Expected 1,887 unique seed team-seasons")
    return result


def build_domestic_form_snapshots(
    domestic_matches: pd.DataFrame,
    bridge: pd.DataFrame,
    identity: pd.DataFrame,
    european_matches: pd.DataFrame,
) -> pd.DataFrame:
    """Freeze domestic state immediately before each season's first UEFA match."""

    domestic = _validate_domestic_matches(domestic_matches)
    mapping = _validated_bridge(bridge, domestic)
    config = DomesticPoissonConfig(0.02, 0.90, 10.0, False)
    engine = DynamicDomesticPoisson(config)
    cutoffs = (
        european_matches.assign(
            kickoff_utc=pd.to_datetime(european_matches["kickoff_utc"], utc=True)
        )
        .groupby("season")["kickoff_utc"]
        .min()
        .to_dict()
    )
    groups = [
        (kickoff, tuple(group))
        for kickoff, group in itertools.groupby(
            _domestic_records(domestic), key=lambda record: record[3]
        )
    ]
    domestic_index = 0
    rows = []
    for season in sorted(cutoffs, key=_season_start):
        cutoff = cutoffs[season]
        while domestic_index < len(groups) and groups[domestic_index][0] < cutoff:
            engine._update_records(groups[domestic_index][1])
            domestic_index += 1
        season_identity = identity.loc[identity["season"].eq(season)]
        for row in season_identity.itertuples(index=False):
            snapshot = _mapped_snapshot(engine, mapping, str(row.club_id))
            rows.append(
                {
                    "season": season,
                    "team_id": int(row.local_team_id),
                    "club_id": str(row.club_id),
                    "domestic_form_attack": snapshot.attack,
                    "domestic_form_defence": snapshot.defence,
                    "domestic_form_score": snapshot.attack + snapshot.defence,
                    "domestic_form_reliability": snapshot.reliability,
                    "domestic_form_effective_matches": snapshot.effective_matches,
                    "domestic_form_covered": snapshot.covered,
                    "domestic_form_cutoff_utc": pd.Timestamp(cutoff).isoformat(),
                }
            )
    result = pd.DataFrame(rows)
    result["domestic_form_percentile"] = season_relative_percentile(
        result, "domestic_form_score", "domestic_form_covered"
    )
    if result.duplicated(["season", "team_id"]).any():
        raise ValueError("domestic form snapshots contain duplicate team-season keys")
    return result


def nested_selection(
    seeds: pd.DataFrame,
    target: pd.DataFrame,
    folds,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    surface_rows = []
    selection_rows = []
    nested_rows = []
    for fold, (train_seasons, test_season) in enumerate(folds, start=1):
        train = seeds.loc[seeds["season"].isin(train_seasons)]
        test = seeds.loc[seeds["season"].eq(test_season)]
        baseline_metrics = seed_ranking_metrics(train, target)
        for arm, signal in ((BOOST_BLIND, "BLIND"), (BOOST_DOMESTIC_FORM, "DOMESTIC_FORM")):
            candidates: list[tuple[SeedBoostConfig | None, dict[str, float | int]]] = [
                (None, baseline_metrics)
            ]
            surface_rows.append(
                {
                    "fold": fold,
                    "test_season": test_season,
                    "arm": arm,
                    "candidate_key": BASELINE_KEY,
                    "affected_team_seasons": 0,
                    **baseline_metrics,
                }
            )
            for candidate in candidate_grid(signal):
                applied = apply_seed_boost(train, candidate)
                metrics = seed_ranking_metrics(
                    applied.rename(columns={"candidate_ao_first_elo": "ranking_rating"}),
                    target,
                    rating_column="ranking_rating",
                )
                affected_count = int(applied["seed_boost_eligible"].sum())
                row = {
                    "fold": fold,
                    "test_season": test_season,
                    "arm": arm,
                    "candidate_key": candidate.key,
                    "affected_team_seasons": affected_count,
                    **metrics,
                }
                surface_rows.append(row)
                if affected_count:
                    candidates.append((candidate, metrics))
            selected, selected_metrics = sorted(
                candidates,
                key=lambda item: (
                    -float(item[1]["spearman"]),
                    -float(item[1]["pairwise_accuracy"]),
                    0 if item[0] is None else 1,
                    BASELINE_KEY if item[0] is None else item[0].key,
                ),
            )[0]
            if selected is None:
                applied_test = baseline_audit(test, arm)
                selected_key = BASELINE_KEY
                selected_payload = None
            else:
                applied_test = apply_seed_boost(test, selected)
                applied_test["arm"] = arm
                selected_key = selected.key
                selected_payload = asdict(selected)
            applied_test["fold"] = fold
            applied_test["test_season"] = test_season
            nested_rows.append(applied_test)
            selection_rows.append(
                {
                    "fold": fold,
                    "test_season": test_season,
                    "arm": arm,
                    "selected_candidate_key": selected_key,
                    "selected_config": json.dumps(selected_payload, sort_keys=True),
                    "train_spearman": selected_metrics["spearman"],
                    "train_pairwise_accuracy": selected_metrics["pairwise_accuracy"],
                    "train_delta_spearman_vs_baseline": (
                        selected_metrics["spearman"] - baseline_metrics["spearman"]
                    ),
                    "train_delta_pairwise_vs_baseline": (
                        selected_metrics["pairwise_accuracy"]
                        - baseline_metrics["pairwise_accuracy"]
                    ),
                    "test_affected_teams": int(
                        applied_test["seed_boost_applied"].gt(0.0).sum()
                    ),
                }
            )
        base_test = baseline_audit(test, BASELINE_KEY)
        base_test["fold"] = fold
        base_test["test_season"] = test_season
        nested_rows.append(base_test)
    return (
        pd.DataFrame(surface_rows),
        pd.DataFrame(selection_rows),
        pd.concat(nested_rows, ignore_index=True),
    )


def baseline_audit(frame: pd.DataFrame, arm: str) -> pd.DataFrame:
    result = frame.copy()
    result["arm"] = arm
    result["european_prior_deficit"] = (
        result["adjusted_domestic_prior"] - result["european_prior"]
    )
    result["negative_european_drag"] = (
        result["adjusted_domestic_prior"] - result["adjusted_ao_first_elo"]
    )
    result["seed_boost_eligible"] = False
    result["seed_boost_requested"] = 0.0
    result["seed_boost_applied"] = 0.0
    result["seed_boost_cap_hit"] = False
    result["seed_boost_model"] = BASELINE_KEY
    result["seed_boost_candidate_key"] = BASELINE_KEY
    result["candidate_ao_first_elo"] = result["adjusted_ao_first_elo"]
    return result


def seed_ranking_metrics(
    seeds: pd.DataFrame,
    target: pd.DataFrame,
    *,
    rating_column: str = "adjusted_ao_first_elo",
) -> dict[str, float | int]:
    joined = seeds.merge(target, on=["season", "team_id"], validate="one_to_one")
    correlations = []
    pairwise = []
    weights = []
    for _, frame in joined.groupby("season", sort=True):
        correlations.append(float(spearmanr(frame[rating_column], frame["schedule_adjusted_score"]).statistic))
        pairwise.append(
            pairwise_accuracy(
                frame[rating_column].to_numpy(float),
                frame["schedule_adjusted_score"].to_numpy(float),
            )
        )
        weights.append(len(frame))
    return {
        "teams": int(len(joined)),
        "seasons": int(joined["season"].nunique()),
        "spearman": float(np.average(correlations, weights=weights)),
        "pairwise_accuracy": float(np.average(pairwise, weights=weights)),
    }


def build_nested_rating_maps(
    baseline: dict[tuple[str, int], float],
    audit: pd.DataFrame,
    seasons: tuple[str, ...],
) -> dict[str, dict[tuple[str, int], float]]:
    maps = {arm: dict(baseline) for arm in ARM_ORDER}
    for arm in (BOOST_BLIND, BOOST_DOMESTIC_FORM):
        rows = audit.loc[audit["arm"].eq(arm)]
        for row in rows.itertuples(index=False):
            maps[arm][(str(row.season), int(row.team_id))] = float(
                row.candidate_ao_first_elo
            )
    expected = {(season, team) for season, team in baseline}
    for arm, mapping in maps.items():
        if set(mapping) != expected or not all(math.isfinite(value) for value in mapping.values()):
            raise ValueError(f"invalid rating map for {arm}")
    return maps


def fold_seed_ranking(audit: pd.DataFrame, target: pd.DataFrame, folds) -> pd.DataFrame:
    rows = []
    for fold, (_, test_season) in enumerate(folds, start=1):
        baseline = None
        for arm in ARM_ORDER:
            frame = audit.loc[
                audit["arm"].eq(arm) & audit["season"].eq(test_season)
            ]
            metrics = seed_ranking_metrics(
                frame,
                target.loc[target["season"].eq(test_season)],
                rating_column="candidate_ao_first_elo",
            )
            if arm == BASELINE_KEY:
                baseline = metrics
            assert baseline is not None
            rows.append(
                {
                    "fold": fold,
                    "test_season": test_season,
                    "arm": arm,
                    **metrics,
                    "delta_spearman_vs_baseline": metrics["spearman"] - baseline["spearman"],
                    "delta_pairwise_vs_baseline": (
                        metrics["pairwise_accuracy"] - baseline["pairwise_accuracy"]
                    ),
                    "affected_teams": int(frame["seed_boost_eligible"].sum()),
                }
            )
    return pd.DataFrame(rows)


def model_comparison(evaluations, fold_ranking: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base_rank = fold_ranking.loc[fold_ranking["arm"].eq(BASELINE_KEY)]
    for arm in ARM_ORDER:
        predictions = evaluations[arm].predictions.loc[
            evaluations[arm].predictions["season"].isin(EVALUATION_SEASONS)
        ]
        ranking = fold_ranking.loc[fold_ranking["arm"].eq(arm)]
        rows.append(
            {
                "arm": arm,
                "matches": len(predictions),
                "brier_1x2": predictions["brier_1x2"].mean(),
                "log_loss_1x2": predictions["log_loss_1x2"].mean(),
                "accuracy_1x2": (predictions["actual_class"] == predictions["predicted_class"]).mean(),
                "seed_spearman": np.average(ranking["spearman"], weights=ranking["teams"]),
                "seed_pairwise_accuracy": np.average(ranking["pairwise_accuracy"], weights=ranking["teams"]),
                "spearman_fold_wins": int((ranking["delta_spearman_vs_baseline"] > 0).sum()),
                "pairwise_fold_wins": int((ranking["delta_pairwise_vs_baseline"] > 0).sum()),
            }
        )
    result = pd.DataFrame(rows)
    baseline = result.loc[result["arm"].eq(BASELINE_KEY)].iloc[0]
    for metric in ("brier_1x2", "log_loss_1x2", "accuracy_1x2", "seed_spearman", "seed_pairwise_accuracy"):
        result[f"delta_vs_baseline_{metric}"] = result[metric] - baseline[metric]
    return result


def competition_summary(evaluations, target_by_competition: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm in ARM_ORDER:
        predictions = evaluations[arm].predictions.loc[
            evaluations[arm].predictions["season"].isin(EVALUATION_SEASONS)
        ]
        initial = evaluations[arm].end_ratings.loc[
            evaluations[arm].end_ratings["season"].isin(EVALUATION_SEASONS)
        ][["season", "team_id", "initial_rating"]]
        for competition in ("UCL", "UEL", "UECL"):
            match_rows = predictions.loc[predictions["competition"].eq(competition)]
            rating_rows = target_by_competition.loc[
                target_by_competition["season"].isin(EVALUATION_SEASONS)
                & target_by_competition["competition"].eq(competition)
            ].merge(initial, on=["season", "team_id"], how="inner")
            rows.append(
                {
                    "arm": arm,
                    "competition": competition,
                    "matches": len(match_rows),
                    "teams": len(rating_rows),
                    "brier_1x2": match_rows["brier_1x2"].mean(),
                    "log_loss_1x2": match_rows["log_loss_1x2"].mean(),
                    "seed_spearman": spearmanr(
                        rating_rows["initial_rating"], rating_rows["schedule_adjusted_score"]
                    ).statistic,
                    "seed_pairwise_accuracy": pairwise_accuracy(
                        rating_rows["initial_rating"].to_numpy(float),
                        rating_rows["schedule_adjusted_score"].to_numpy(float),
                    ),
                }
            )
    result = pd.DataFrame(rows)
    baseline = result.loc[result["arm"].eq(BASELINE_KEY)].set_index("competition")
    for metric in ("brier_1x2", "log_loss_1x2", "seed_spearman", "seed_pairwise_accuracy"):
        result[f"delta_vs_baseline_{metric}"] = result.apply(
            lambda row: row[metric] - baseline.loc[row["competition"], metric], axis=1
        )
    return result


def exposure_summary(audit: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    data = audit.merge(target, on=["season", "team_id"], validate="many_to_one")
    data["exposure_band"] = pd.cut(
        data["effective_european_exposure"],
        [-1e-12, 0.0, 0.25, 0.50, 0.75, 1.0],
        labels=["0", "(0,0.25]", "(0.25,0.50]", "(0.50,0.75]", "(0.75,1.00]"],
        include_lowest=True,
    )
    rows = []
    for (arm, band), frame in data.groupby(["arm", "exposure_band"], observed=False):
        rows.append(
            {
                "arm": arm,
                "exposure_band": band,
                "teams": len(frame),
                "affected_teams": int(frame["seed_boost_eligible"].sum()),
                "mean_boost": frame["seed_boost_applied"].mean(),
                "maximum_boost": frame["seed_boost_applied"].max(),
                "seed_spearman": spearmanr(
                    frame["candidate_ao_first_elo"], frame["schedule_adjusted_score"]
                ).statistic if len(frame) >= 3 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def seed_impact_summary(audit: pd.DataFrame) -> pd.DataFrame:
    """Summarize intervention mass and overlap with active Domestic Surprise."""

    rows = []
    for arm in ARM_ORDER:
        frame = audit.loc[
            audit["arm"].eq(arm) & audit["season"].isin(EVALUATION_SEASONS)
        ].copy()
        affected = frame.loc[frame["seed_boost_applied"].gt(0.0)]
        boosts = frame["seed_boost_applied"].astype(float)
        rows.append(
            {
                "arm": arm,
                "team_seasons": len(frame),
                "affected_team_seasons": len(affected),
                "affected_share": len(affected) / len(frame) if len(frame) else 0.0,
                "mean_boost_all": boosts.mean(),
                "median_boost_all": boosts.median(),
                "mean_boost_affected": (
                    affected["seed_boost_applied"].mean() if len(affected) else 0.0
                ),
                "median_boost_affected": (
                    affected["seed_boost_applied"].median() if len(affected) else 0.0
                ),
                "p90_boost": boosts.quantile(0.90),
                "p95_boost": boosts.quantile(0.95),
                "maximum_boost": boosts.max(),
                "cap_hits": int(frame["seed_boost_cap_hit"].sum()),
                "negative_surprise_affected": int(
                    affected["domestic_prior_adjustment"].lt(0.0).sum()
                ),
                "minus_30_surprise_affected": int(
                    affected["domestic_prior_adjustment"].le(-30.0 + 1e-12).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def affected_match_summary(evaluations, audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm in ARM_ORDER:
        boosted = {
            (str(row.season), int(row.team_id))
            for row in audit.loc[
                audit["arm"].eq(arm) & audit["seed_boost_eligible"]
            ].itertuples(index=False)
        }
        predictions = evaluations[arm].predictions.loc[
            evaluations[arm].predictions["season"].isin(EVALUATION_SEASONS)
        ].copy()
        predictions["affected_segment"] = [
            "AFFECTED"
            if (str(season), int(home)) in boosted or (str(season), int(away)) in boosted
            else "UNAFFECTED"
            for season, home, away in zip(
                predictions["season"], predictions["home_team_id"], predictions["away_team_id"], strict=True
            )
        ]
        for segment, frame in predictions.groupby("affected_segment"):
            rows.append(
                {
                    "arm": arm,
                    "segment": segment,
                    "matches": len(frame),
                    "brier_1x2": frame["brier_1x2"].mean(),
                    "log_loss_1x2": frame["log_loss_1x2"].mean(),
                    "accuracy_1x2": (frame["actual_class"] == frame["predicted_class"]).mean(),
                }
            )
    return pd.DataFrame(rows)


def forward_ranking_summary(evaluations, target, seasons) -> pd.DataFrame:
    identity = load_team_season_identity()
    rows = []
    for arm in ARM_ORDER:
        ranking = summarize_ranking(
            evaluations[arm].end_ratings,
            target,
            allowed_target_seasons=set(seasons[3:]),
            identity=identity,
        )
        ranking["arm"] = arm
        rows.append(ranking)
    result = pd.concat(rows, ignore_index=True)
    baseline = result.loc[result["arm"].eq(BASELINE_KEY)].set_index("competition")
    for metric in ("ranking_score", "pairwise_accuracy"):
        result[f"delta_vs_baseline_{metric}"] = result.apply(
            lambda row: row[metric] - baseline.loc[row["competition"], metric], axis=1
        )
    return result


def loss_uncertainty_summary(evaluations, samples: int) -> pd.DataFrame:
    rows = []
    baseline = evaluations[BASELINE_KEY].predictions.loc[
        evaluations[BASELINE_KEY].predictions["season"].isin(EVALUATION_SEASONS)
    ]
    for arm in (BOOST_BLIND, BOOST_DOMESTIC_FORM):
        candidate = evaluations[arm].predictions.loc[
            evaluations[arm].predictions["season"].isin(EVALUATION_SEASONS)
        ]
        for competition in ("ALL", "UCL", "UEL", "UECL"):
            left = candidate if competition == "ALL" else candidate.loc[candidate["competition"].eq(competition)]
            right = baseline if competition == "ALL" else baseline.loc[baseline["competition"].eq(competition)]
            paired = left.merge(
                right[["match_id", "brier_1x2", "log_loss_1x2"]],
                on="match_id",
                suffixes=("_candidate", "_baseline"),
                validate="one_to_one",
            )
            for metric in ("brier_1x2", "log_loss_1x2"):
                sample = paired[["season", "match_id", "home_team_id", "away_team_id", "kickoff_utc", "tie_id"]].copy()
                sample["loss_difference"] = paired[f"{metric}_candidate"] - paired[f"{metric}_baseline"]
                ci = dependency_robust_loss_difference_ci(sample, bootstrap_samples=samples)
                ci.insert(0, "arm", arm)
                ci.insert(1, "competition", competition)
                ci.insert(2, "metric", metric)
                rows.append(ci)
    return pd.concat(rows, ignore_index=True)


def ranking_uncertainty_summary(
    audit: pd.DataFrame,
    target: pd.DataFrame,
    *,
    samples: int,
) -> pd.DataFrame:
    data = audit.loc[audit["season"].isin(EVALUATION_SEASONS)].merge(
        target, on=["season", "team_id"], validate="many_to_one"
    )
    baseline = data.loc[data["arm"].eq(BASELINE_KEY)][
        ["season", "team_id", "club_id", "adjusted_ao_first_elo", "schedule_adjusted_score"]
    ]
    rows = []
    for arm in (BOOST_BLIND, BOOST_DOMESTIC_FORM):
        candidate = data.loc[data["arm"].eq(arm)][
            ["season", "team_id", "candidate_ao_first_elo"]
        ]
        paired = baseline.merge(candidate, on=["season", "team_id"], validate="one_to_one")
        point = _weighted_season_spearman_difference(paired)
        method_intervals = []
        for method, groups in (
            ("team_season", [[index] for index in paired.index]),
            ("club", [list(index) for _, index in paired.groupby("club_id").groups.items()]),
            ("season", [list(index) for _, index in paired.groupby("season").groups.items()]),
        ):
            low, high = _cluster_bootstrap_spearman(
                paired, groups, samples=samples, seed=BOOTSTRAP_SEED
            )
            method_intervals.append((low, high))
            rows.append(
                {
                    "arm": arm,
                    "metric": "seed_spearman_difference",
                    "method": method,
                    "mean_difference": point,
                    "ci_95_lower": low,
                    "ci_95_upper": high,
                    "reliable_improvement": low > 0.0,
                    "reliable_harm": high < 0.0,
                }
            )
        low = min(value[0] for value in method_intervals)
        high = max(value[1] for value in method_intervals)
        rows.append(
            {
                "arm": arm,
                "metric": "seed_spearman_difference",
                "method": "conservative_envelope",
                "mean_difference": point,
                "ci_95_lower": low,
                "ci_95_upper": high,
                "reliable_improvement": low > 0.0,
                "reliable_harm": high < 0.0,
            }
        )
    return pd.DataFrame(rows)


def _cluster_bootstrap_spearman(frame, groups, *, samples: int, seed: int):
    generator = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        chosen = generator.integers(0, len(groups), len(groups))
        index = [row for group_index in chosen for row in groups[group_index]]
        sample = frame.loc[index]
        value = _weighted_season_spearman_difference(sample)
        if math.isfinite(value):
            values.append(value)
    if not values:
        raise ValueError("ranking bootstrap produced no finite sample")
    return tuple(float(value) for value in np.quantile(values, (0.025, 0.975)))


def _weighted_season_spearman_difference(frame: pd.DataFrame) -> float:
    """Match the season-weighted Spearman definition used in model_comparison."""

    differences = []
    weights = []
    for _, season in frame.groupby("season", sort=True):
        if len(season) < 3:
            continue
        realized = season["schedule_adjusted_score"]
        candidate = spearmanr(
            season["candidate_ao_first_elo"], realized
        ).statistic
        baseline = spearmanr(
            season["adjusted_ao_first_elo"], realized
        ).statistic
        if math.isfinite(candidate) and math.isfinite(baseline):
            differences.append(float(candidate - baseline))
            weights.append(len(season))
    if not differences:
        return float("nan")
    return float(np.average(differences, weights=weights))


def external_opta_summary(audit: pd.DataFrame, samples: int) -> pd.DataFrame:
    table = build_rating_table(
        opta_snapshot_path=OPTA_SNAPSHOT,
        team_identity_path=TEAM_IDENTITY,
        model_ratings_path=MODEL_RATINGS,
        matches_path=ROOT / "data" / "dynamic_backtest_2018_2026" / "matches.csv",
        club_coefficients_path=CLUB_COEFFICIENTS,
    )
    season_audit = audit.loc[audit["season"].eq("2025/26")]
    rows = []
    realized = table["schedule_adjusted_score"].to_numpy(float)
    opta = table["opta_power_rating"].to_numpy(float)
    for arm in ARM_ORDER:
        ratings = season_audit.loc[season_audit["arm"].eq(arm)][
            ["team_id", "candidate_ao_first_elo"]
        ].copy()
        ratings["team_id"] = ratings["team_id"].astype(str)
        joined = table.merge(ratings, on="team_id", validate="one_to_one")
        candidate = joined["candidate_ao_first_elo"].to_numpy(float)
        mean, lower, upper = paired_spearman_difference_ci(
            candidate,
            joined["opta_power_rating"].to_numpy(float),
            joined["schedule_adjusted_score"].to_numpy(float),
            samples=samples,
            seed=BOOTSTRAP_SEED,
        )
        rows.append(
            {
                "arm": arm,
                "teams": len(joined),
                "spearman_vs_realized": spearmanr(candidate, joined["schedule_adjusted_score"]).statistic,
                "pairwise_accuracy_vs_realized": pairwise_accuracy(candidate, joined["schedule_adjusted_score"].to_numpy(float)),
                "opta_spearman_vs_realized": spearmanr(joined["opta_power_rating"], joined["schedule_adjusted_score"]).statistic,
                "spearman_difference_vs_opta": mean,
                "ci_95_lower": lower,
                "ci_95_upper": upper,
                "opta_reliably_better": upper < 0.0,
            }
        )
    return pd.DataFrame(rows)


def long_history_feasibility() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "arm": BOOST_LONG_HISTORY,
                "status": "UNAVAILABLE_NOT_RUN",
                "required_signal": "pre-season club reputation older than the active five-season window",
                "available_local_source": "NONE",
                "reason": (
                    "The repository has no dated 10-20 year UEFA/Opta club-rating snapshots "
                    "or audited long-run title table. Reusing 2013-2026 domestic goals would "
                    "duplicate BOOST_DOMESTIC_FORM rather than create independent reputation."
                ),
                "minimum_future_contract": (
                    "dated pre-season snapshot, stable club_id crosswalk, source license, "
                    "coverage audit and at least two historical outer-fold training seasons"
                ),
            }
        ]
    )


def safety_audit(audit, evaluations, contract_path, contract_hash):
    boosted = audit.loc[audit["arm"].ne(BASELINE_KEY)]
    baseline = evaluations[BASELINE_KEY]
    rows = [
        {
            "check": "unique_seed_team_season_arm",
            "passed": not audit.duplicated(["arm", "season", "team_id"]).any(),
            "observed": int(audit.duplicated(["arm", "season", "team_id"]).sum()),
            "requirement": "zero duplicates",
        },
        {
            "check": "zero_exposure_never_boosted",
            "passed": not boosted.loc[boosted["effective_european_exposure"].eq(0.0), "seed_boost_applied"].gt(0).any(),
            "observed": int(boosted.loc[boosted["effective_european_exposure"].eq(0.0), "seed_boost_applied"].gt(0).sum()),
            "requirement": "zero",
        },
        {
            "check": "boost_non_negative_and_capped",
            "passed": boosted["seed_boost_applied"].between(0.0, 150.0).all(),
            "observed": float(boosted["seed_boost_applied"].max()),
            "requirement": "[0,150] Elo",
        },
        {
            "check": "ineligible_rows_unchanged",
            "passed": boosted.loc[~boosted["seed_boost_eligible"], "seed_boost_applied"].abs().le(1e-12).all(),
            "observed": float(boosted.loc[~boosted["seed_boost_eligible"], "seed_boost_applied"].abs().max()),
            "requirement": "0 Elo",
        },
        {
            "check": "production_contract_hash_unchanged",
            "passed": hashlib.sha256(contract_path.read_bytes()).hexdigest() == contract_hash,
            "observed": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
            "requirement": contract_hash,
        },
    ]
    for arm, evaluation in evaluations.items():
        rows.append(
            {
                "check": f"power_zero_sum_{arm}",
                "passed": evaluation.season_metrics["season_power_conservation_error"].le(1e-9).all(),
                "observed": float(evaluation.season_metrics["season_power_conservation_error"].max()),
                "requirement": "<=1e-9",
            }
        )
    return pd.DataFrame(rows)


def decide(
    comparison,
    rank_ci,
    loss_ci,
    external,
    selections,
    impact,
    forward,
    feasibility,
    contract_hash,
):
    rows = comparison.set_index("arm")
    ci = rank_ci.loc[rank_ci["method"].eq("conservative_envelope")].set_index("arm")
    loss_envelope = loss_ci.loc[
        loss_ci["method"].eq("conservative_envelope")
        & loss_ci["competition"].eq("ALL")
    ]
    external_index = external.set_index("arm")
    forward_index = forward.loc[forward["competition"].eq("ALL")].set_index("arm")
    arm_decisions = {}
    for arm in (BOOST_BLIND, BOOST_DOMESTIC_FORM):
        ranking_gain = float(rows.loc[arm, "delta_vs_baseline_seed_spearman"])
        reliable_ranking = bool(ci.loc[arm, "ci_95_lower"] > 0.0)
        reliable_loss_harm = bool(
            loss_envelope.loc[loss_envelope["arm"].eq(arm), "reliable_harm"].any()
        )
        opta_gap = abs(float(external_index.loc[arm, "spearman_difference_vs_opta"]))
        baseline_gap = abs(float(external_index.loc[BASELINE_KEY, "spearman_difference_vs_opta"]))
        affected = int(impact.set_index("arm").loc[arm, "affected_team_seasons"])
        if arm == BOOST_BLIND:
            status = "KEEP_DIAGNOSTIC_CONTROL"
        elif (
            ranking_gain > 0
            and reliable_ranking
            and not reliable_loss_harm
            and opta_gap < baseline_gap
            and affected > 0
        ):
            status = "PROMOTE_CANDIDATE"
        elif ranking_gain > 0 and not reliable_loss_harm and affected > 0:
            status = "KEEP_DIAGNOSTIC"
        else:
            status = "REJECTED"
        arm_decisions[arm] = {
            "status": status,
            "seed_spearman_delta": ranking_gain,
            "ranking_ci_95": [float(ci.loc[arm, "ci_95_lower"]), float(ci.loc[arm, "ci_95_upper"])],
            "reliable_loss_harm": reliable_loss_harm,
            "opta_gap": float(external_index.loc[arm, "spearman_difference_vs_opta"]),
            "selected_non_baseline_folds": int(
                selections.loc[
                    selections["arm"].eq(arm), "selected_candidate_key"
                ].ne(BASELINE_KEY).sum()
            ),
            "affected_team_seasons": affected,
            "effective_test_folds": int(
                selections.loc[selections["arm"].eq(arm), "test_affected_teams"]
                .gt(0)
                .sum()
            ),
            "forward_spearman_delta": float(
                forward_index.loc[arm, "delta_vs_baseline_ranking_score"]
            ),
            "forward_pairwise_delta": float(
                forward_index.loc[arm, "delta_vs_baseline_pairwise_accuracy"]
            ),
        }
    domestic = arm_decisions[BOOST_DOMESTIC_FORM]
    if domestic["status"] == "PROMOTE_CANDIDATE":
        decision = "PROMOTE_CANDIDATE"
        selected = BOOST_DOMESTIC_FORM
    elif domestic["status"] == "KEEP_DIAGNOSTIC":
        decision = "KEEP_DIAGNOSTIC"
        selected = BOOST_DOMESTIC_FORM
    else:
        decision = "REJECTED"
        selected = None
    return {
        "decision": decision,
        "selected_arm": selected,
        "production_activated": False,
        "production_contract_sha256": contract_hash,
        "arms": arm_decisions,
        "long_history_status": str(feasibility.iloc[0]["status"]),
        "interpretation": (
            "BOOST_BLIND is a diagnostic control and cannot be selected for production. "
            "Only BOOST_DOMESTIC_FORM is eligible as a product candidate. No production "
            "file or active parameter was changed."
        ),
    }


def build_report(
    decision,
    comparison,
    folds,
    selections,
    competition,
    exposure,
    impact,
    affected,
    forward,
    external,
    rank_ci,
    loss_ci,
    feasibility,
    safety,
):
    selected_view = selections[["fold", "test_season", "arm", "selected_candidate_key", "train_delta_spearman_vs_baseline"]]
    return "\n".join(
        [
            "# AO First Elo Seed Asimetrisi Boost Backtesti",
            "",
            f"Karar: **`{decision['decision']}`**. Production aktivasyonu: `False`.",
            "",
            "## Soru",
            "",
            "Ince fakat olumsuz Avrupa gecmisi, sifir Avrupa gecmisinden daha agir",
            "cezalandirilan ve son lig bitirisi zayif gorunen kulupte, bagimsiz bir",
            "'su anda iyi' sinyali seed ranking acigini kapatiyor mu?",
            "",
            "## Kollar",
            "",
            "- `BASELINE`: aktif AO First Elo.",
            "- `BOOST_BLIND`: yalniz zayif Avrupa + zayif son bitiris; kontrol.",
            "- `BOOST_DOMESTIC_FORM`: ayni kural + causal domestic Poisson attack/defence yuzdeligi.",
            "- `BOOST_LONG_HISTORY`: veri yoklugu nedeniyle `UNAVAILABLE_NOT_RUN`; sahte kol uretilmedi.",
            "",
            "Her aday additive, exposure modifier ve floor tasarimlarini; finish/deficit",
            "esiklerini ve boost buyuklugunu yalniz onceki sezonlarda nested olarak secer.",
            "Exposure `0` olan takim hicbir kosulda etkilenmez. Maksimum boost `150` Elo'dur.",
            "",
            "## Pooled sonuc",
            "",
            "```text",
            comparison.to_string(index=False),
            "```",
            "",
            "## Fold bazli seed ranking",
            "",
            "```text",
            folds.to_string(index=False),
            "```",
            "",
            "## Nested secimler",
            "",
            "```text",
            selected_view.to_string(index=False),
            "```",
            "",
            "## Opta dis benchmark ekseni (2025/26)",
            "",
            "```text",
            external.to_string(index=False),
            "```",
            "",
            "## Ranking belirsizligi",
            "",
            "```text",
            rank_ci.loc[rank_ci["method"].eq("conservative_envelope")].to_string(index=False),
            "```",
            "",
            "## Match-loss no-harm belirsizligi",
            "",
            "```text",
            loss_ci.loc[loss_ci["method"].eq("conservative_envelope")].to_string(index=False),
            "```",
            "",
            "## Turnuva segmentleri",
            "",
            "```text",
            competition.to_string(index=False),
            "```",
            "",
            "## Exposure ve etkilenen mac segmentleri",
            "",
            "```text",
            exposure.to_string(index=False),
            "```",
            "",
            "## Mudahale kutlesi ve Domestic Surprise etkilesimi",
            "",
            "```text",
            impact.to_string(index=False),
            "```",
            "",
            "```text",
            affected.to_string(index=False),
            "```",
            "",
            "## Forward ranking no-harm kontrolu",
            "",
            "Bu tablo ileri sezon siralamasini bir terfi hedefi olarak degil, olasi",
            "yan etki kontrolu olarak gosterir.",
            "",
            "```text",
            forward.to_string(index=False),
            "```",
            "",
            "## Long-history fizibilitesi",
            "",
            "```text",
            feasibility.to_string(index=False),
            "```",
            "",
            "## Safety",
            "",
            "```text",
            safety.to_string(index=False),
            "```",
            "",
            "Bu calisma development-window kanitidir; 2026/27 prospective sonuc yerine gecmez.",
            "Production contract yalniz hash'lendi ve degistirilmedi.",
            "",
        ]
    )


def _season_start(value: str) -> int:
    return int(str(value).split("/")[0])


if __name__ == "__main__":
    main()
