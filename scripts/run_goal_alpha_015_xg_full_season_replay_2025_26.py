from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ao_elo.evaluation import schedule_adjusted_team_performance  # noqa: E402
from scripts.run_bounded_xg_adjustment_walk_forward_2025_26 import (  # noqa: E402
    BASELINE_KEY,
    BoundedXGCandidate,
)
from scripts.run_controlled_goal_progression_backtest import (  # noqa: E402
    evaluate_predictions,
)
from scripts.run_fotmob_xg_backtest_2025_26 import (  # noqa: E402
    DATA_PATH,
    MANIFEST_PATH,
    PRODUCTION_MODEL_PATH,
    SEASON,
    STATIC_DATA_ROOT,
    STATIC_MANIFEST_PATH,
    markdown_table,
    read_fotmob_xg_dataset,
    validate_manifest,
)
from scripts.run_xg_goal_ablation_backtest import (  # noqa: E402
    load_initial_ratings,
    read_production_contract,
    read_static_config,
    same_season_ranking,
)
from scripts.run_xg_performance_bonus_walk_forward_2025_26 import (  # noqa: E402
    build_uncertainty as build_dependency_uncertainty,
    replay_candidate,
)


OUTPUT_ROOT = ROOT / "output" / "goal_alpha_015_xg_full_season_replay_2025_26"
PRODUCTION_ARM = "PRODUCTION_ALPHA_010_NO_XG"
CURRENT_XG_ARM = "ALPHA_010_XG_030_125"
CANDIDATE_ARM = "ALPHA_015_XG_030_125"
PRODUCTION_ALPHA = 0.10
CANDIDATE_ALPHA = 0.15
XG_RATIO = 0.30
XG_SCALE = 1.25


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay 2025/26 with approved goal_alpha=0.15 and bounded xG"
    )
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--static-data-root", type=Path, default=STATIC_DATA_ROOT)
    parser.add_argument("--static-manifest", type=Path, default=STATIC_MANIFEST_PATH)
    parser.add_argument("--production-model", type=Path, default=PRODUCTION_MODEL_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("bootstrap-samples must be at least 100")

    events = read_fotmob_xg_dataset(args.data.resolve(), strict_contract=True)
    manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    validate_manifest(manifest, events)
    production = read_production_contract(args.production_model.resolve())
    static_config = read_static_config(args.static_manifest.resolve())
    initial_ratings = load_initial_ratings(
        args.static_data_root.resolve(), events, static_config
    )[SEASON]

    replays = run_replays(events, initial_ratings, production)
    predictions = pd.concat(
        [
            replay.predictions.assign(model_arm=arm, goal_alpha=arm_alpha(arm))
            for arm, replay in replays.items()
        ],
        ignore_index=True,
    )
    comparison = build_model_comparison(predictions)
    competition = build_competition_summary(predictions)
    final_ratings = build_final_ratings(replays)
    ranking = build_ranking_summary(replays, events)
    match_updates = build_match_comparison(replays, production)
    trajectories = build_trajectories(replays, production)
    uncertainty = build_uncertainty(
        predictions, bootstrap_samples=args.bootstrap_samples
    )
    validate_replay(events, replays, match_updates, final_ratings)

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        output_root / "arm_match_updates.csv", index=False, float_format="%.9f"
    )
    match_updates.to_csv(
        output_root / "match_updates_comparison.csv",
        index=False,
        float_format="%.9f",
    )
    comparison.to_csv(output_root / "model_comparison.csv", index=False)
    competition.to_csv(output_root / "competition_summary.csv", index=False)
    ranking.to_csv(output_root / "ranking_summary.csv", index=False)
    uncertainty.to_csv(output_root / "dependency_uncertainty.csv", index=False)
    trajectories.to_csv(
        output_root / "team_rating_trajectories.csv",
        index=False,
        float_format="%.9f",
    )
    final_ratings.to_csv(
        output_root / "final_ratings_comparison.csv",
        index=False,
        float_format="%.9f",
    )
    payload = build_manifest(events, final_ratings, production, args.bootstrap_samples)
    (output_root / "replay_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_root / "replay_report.md").write_text(
        build_report(
            comparison,
            competition,
            ranking,
            uncertainty,
            final_ratings,
            match_updates,
        ),
        encoding="utf-8",
    )
    print(f"Matches per arm: {len(events)}")
    print(f"Teams: {len(final_ratings)}")
    print(f"xG eligible: {int(events['xg_analysis_eligible'].sum())}")
    print(f"Report: {output_root / 'replay_report.md'}")


def arm_alpha(arm: str) -> float:
    return CANDIDATE_ALPHA if arm == CANDIDATE_ARM else PRODUCTION_ALPHA


def candidate_contract(production: dict[str, object]) -> dict[str, object]:
    result = copy.deepcopy(production)
    goal = result["goal_margin"]
    if abs(float(goal["alpha"]) - PRODUCTION_ALPHA) > 1e-12:
        raise ValueError("Production comparator must use goal_alpha=0.10")
    goal["alpha"] = CANDIDATE_ALPHA
    return result


def run_replays(
    events: pd.DataFrame,
    initial_ratings: dict[int, float],
    production: dict[str, object],
) -> dict[str, object]:
    no_xg = BoundedXGCandidate(BASELINE_KEY, 0.0, 1.0)
    bounded_xg = BoundedXGCandidate("BOUNDED_XG_030_125", XG_RATIO, XG_SCALE)
    candidate = candidate_contract(production)
    return {
        PRODUCTION_ARM: replay_candidate(
            events, initial_ratings, production, no_xg, snapshot_times=()
        ),
        CURRENT_XG_ARM: replay_candidate(
            events, initial_ratings, production, bounded_xg, snapshot_times=()
        ),
        CANDIDATE_ARM: replay_candidate(
            events, initial_ratings, candidate, bounded_xg, snapshot_times=()
        ),
    }


def build_model_comparison(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for arm, frame in predictions.groupby("model_arm", sort=True):
        scopes = {
            "ALL_961": frame,
            "XG_ELIGIBLE_606": frame.loc[frame["xg_analysis_eligible"]],
            "NO_XG_355": frame.loc[~frame["xg_analysis_eligible"]],
        }
        for scope, values in scopes.items():
            decisive = values.loc[values["actual_class"].ne(1)]
            rows.append(
                {
                    "model_arm": arm,
                    "scope": scope,
                    **evaluate_predictions(values),
                    "mean_abs_match_delta": float(values["power_delta"].abs().mean()),
                    "max_abs_match_delta": float(values["power_delta"].abs().max()),
                    "minimum_winner_gain": float(decisive["winner_elo_gain"].min()),
                    "maximum_zero_sum_error": float(values["zero_sum_error"].max()),
                }
            )
    result = pd.DataFrame(rows)
    return add_metric_deltas(
        result,
        keys=["scope"],
        baselines=(PRODUCTION_ARM, CURRENT_XG_ARM),
    )


def build_competition_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "model_arm": arm,
            "competition": competition,
            **evaluate_predictions(frame),
        }
        for (arm, competition), frame in predictions.groupby(
            ["model_arm", "competition"], sort=True
        )
    ]
    return add_metric_deltas(
        pd.DataFrame(rows),
        keys=["competition"],
        baselines=(PRODUCTION_ARM, CURRENT_XG_ARM),
    )


def add_metric_deltas(
    values: pd.DataFrame,
    *,
    keys: list[str],
    baselines: tuple[str, ...],
) -> pd.DataFrame:
    result = values.copy()
    metrics = ("brier_1x2", "log_loss_1x2", "accuracy_1x2", "multiclass_ece")
    for baseline_arm in baselines:
        suffix = "production" if baseline_arm == PRODUCTION_ARM else "alpha_010_xg"
        baseline = result.loc[result["model_arm"].eq(baseline_arm)].set_index(keys)
        for metric in metrics:
            result[f"{metric}_delta_vs_{suffix}"] = result.apply(
                lambda row: row[metric]
                - baseline.loc[
                    row[keys[0]] if len(keys) == 1 else tuple(row[key] for key in keys),
                    metric,
                ],
                axis=1,
            )
    return result


def build_final_ratings(replays: dict[str, object]) -> pd.DataFrame:
    base = replays[PRODUCTION_ARM].final_ratings[
        ["team_id", "team_name", "start_rating"]
    ].copy()
    result = base
    for arm, replay in replays.items():
        prefix = {
            PRODUCTION_ARM: "production",
            CURRENT_XG_ARM: "alpha_010_xg",
            CANDIDATE_ARM: "alpha_015_xg",
        }[arm]
        frame = replay.final_ratings[
            ["team_id", "end_live_rating", "rating_change"]
        ].rename(
            columns={
                "end_live_rating": f"{prefix}_final_rating",
                "rating_change": f"{prefix}_season_change",
            }
        )
        result = result.merge(frame, on="team_id", validate="one_to_one")
        result[f"{prefix}_rank"] = result[f"{prefix}_final_rating"].rank(
            method="min", ascending=False
        ).astype(int)
    result["candidate_minus_production"] = (
        result["alpha_015_xg_final_rating"] - result["production_final_rating"]
    )
    result["candidate_minus_alpha_010_xg"] = (
        result["alpha_015_xg_final_rating"] - result["alpha_010_xg_final_rating"]
    )
    result["candidate_rank_change_vs_production"] = (
        result["production_rank"] - result["alpha_015_xg_rank"]
    )
    result["candidate_rank_change_vs_alpha_010_xg"] = (
        result["alpha_010_xg_rank"] - result["alpha_015_xg_rank"]
    )
    return result.sort_values(
        ["alpha_015_xg_final_rating", "team_id"], ascending=[False, True]
    ).reset_index(drop=True)


def build_ranking_summary(
    replays: dict[str, object], events: pd.DataFrame
) -> pd.DataFrame:
    target = schedule_adjusted_team_performance(events)
    frames = []
    for arm, replay in replays.items():
        ranking = same_season_ranking(replay.final_ratings, target, {SEASON}).copy()
        ranking.insert(0, "model_arm", arm)
        frames.append(ranking)
    result = pd.concat(frames, ignore_index=True)
    for baseline_arm in (PRODUCTION_ARM, CURRENT_XG_ARM):
        suffix = "production" if baseline_arm == PRODUCTION_ARM else "alpha_010_xg"
        baseline = result.loc[result["model_arm"].eq(baseline_arm)].set_index(
            "competition"
        )
        for metric in ("ranking_score", "pairwise_accuracy"):
            result[f"{metric}_delta_vs_{suffix}"] = result.apply(
                lambda row: row[metric] - baseline.loc[row["competition"], metric],
                axis=1,
            )
    return result


def build_match_comparison(
    replays: dict[str, object], production: dict[str, object]
) -> pd.DataFrame:
    metadata = [
        "match_id",
        "season",
        "competition",
        "round",
        "kickoff_utc",
        "home_team_id",
        "away_team_id",
        "home_team_name",
        "away_team_name",
        "home_goals",
        "away_goals",
        "decided_on_penalties",
        "xg_analysis_eligible",
        "xg_home",
        "xg_away",
        "xg_difference",
        "actual_class",
    ]
    fields = [
        "home_rating_pre",
        "away_rating_pre",
        "expected_home_score",
        "home_probability",
        "draw_probability",
        "away_probability",
        "brier_1x2",
        "log_loss_1x2",
        "goal_multiplier",
        "goal_bonus_residual",
        "xg_performance_adjustment",
        "power_delta",
        "winner_elo_gain",
        "home_rating_post",
        "away_rating_post",
        "zero_sum_error",
    ]
    prefixes = {
        PRODUCTION_ARM: "production",
        CURRENT_XG_ARM: "alpha_010_xg",
        CANDIDATE_ARM: "alpha_015_xg",
    }
    result = replays[PRODUCTION_ARM].predictions[metadata].copy()
    for arm, replay in replays.items():
        prefix = prefixes[arm]
        frame = replay.predictions[["match_id"] + fields].rename(
            columns={column: f"{prefix}_{column}" for column in fields}
        )
        result = result.merge(frame, on="match_id", validate="one_to_one")
    k_factor = float(production["dynamic_core"]["k_factor"])
    result["candidate_xg_adjustment_home_elo"] = (
        result["alpha_015_xg_xg_performance_adjustment"] * k_factor
    )
    result["candidate_power_delta_difference_vs_alpha_010_xg"] = (
        result["alpha_015_xg_power_delta"] - result["alpha_010_xg_power_delta"]
    )
    result["candidate_winner_gain_difference_vs_alpha_010_xg"] = (
        result["alpha_015_xg_winner_elo_gain"]
        - result["alpha_010_xg_winner_elo_gain"]
    )
    result["candidate_brier_difference_vs_alpha_010_xg"] = (
        result["alpha_015_xg_brier_1x2"] - result["alpha_010_xg_brier_1x2"]
    )
    result["candidate_log_loss_difference_vs_alpha_010_xg"] = (
        result["alpha_015_xg_log_loss_1x2"] - result["alpha_010_xg_log_loss_1x2"]
    )
    return result


def build_trajectories(
    replays: dict[str, object], production: dict[str, object]
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    k_factor = float(production["dynamic_core"]["k_factor"])
    for arm, replay in replays.items():
        values = replay.predictions
        for side, sign in (("HOME", 1.0), ("AWAY", -1.0)):
            lower = side.lower()
            frames.append(
                pd.DataFrame(
                    {
                        "model_arm": arm,
                        "match_id": values["match_id"],
                        "kickoff_utc": values["kickoff_utc"],
                        "competition": values["competition"],
                        "team_id": values[f"{lower}_team_id"],
                        "team_name": values[f"{lower}_team_name"],
                        "side": side,
                        "rating_pre": values[f"{lower}_rating_pre"],
                        "rating_post": values[f"{lower}_rating_post"],
                        "power_delta": sign * values["power_delta"],
                        "xg_adjustment_elo": sign
                        * values["xg_performance_adjustment"]
                        * k_factor,
                    }
                )
            )
    return pd.concat(frames, ignore_index=True).sort_values(
        ["kickoff_utc", "match_id", "model_arm", "side"]
    ).reset_index(drop=True)


def build_uncertainty(
    predictions: pd.DataFrame, *, bootstrap_samples: int
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    candidate_all = predictions.loc[predictions["model_arm"].eq(CANDIDATE_ARM)]
    for comparator in (PRODUCTION_ARM, CURRENT_XG_ARM):
        comparator_all = predictions.loc[predictions["model_arm"].eq(comparator)]
        for scope, eligible_only in (("ALL_961", False), ("XG_ELIGIBLE_606", True)):
            candidate = candidate_all
            baseline = comparator_all
            if eligible_only:
                candidate = candidate.loc[candidate["xg_analysis_eligible"]]
                baseline = baseline.loc[baseline["xg_analysis_eligible"]]
            paired = pd.concat(
                [
                    baseline.assign(model="GD_PRODUCTION"),
                    candidate.assign(model="NESTED_SELECTED_XG"),
                ],
                ignore_index=True,
            )
            result = build_dependency_uncertainty(
                paired, bootstrap_samples=bootstrap_samples
            )
            result.insert(0, "scope", scope)
            result.insert(1, "candidate_arm", CANDIDATE_ARM)
            result.insert(2, "comparator_arm", comparator)
            rows.append(result)
    return pd.concat(rows, ignore_index=True)


def validate_replay(
    events: pd.DataFrame,
    replays: dict[str, object],
    matches: pd.DataFrame,
    final_ratings: pd.DataFrame,
) -> None:
    if len(events) != 961 or int(events["xg_analysis_eligible"].sum()) != 606:
        raise ValueError("Replay requires 961 matches with 606 xG-eligible rows")
    if len(final_ratings) != 236:
        raise ValueError("Replay requires 236 teams")
    for arm, replay in replays.items():
        predictions = replay.predictions
        if len(predictions) != 961 or predictions["match_id"].duplicated().any():
            raise ValueError(f"{arm} did not process 961 unique matches")
        if predictions["zero_sum_error"].max() > 1e-9:
            raise ValueError(f"{arm} violated match-level zero-sum conservation")
        decisive = predictions.loc[predictions["actual_class"].ne(1)]
        if not decisive["winner_elo_gain"].gt(0.0).all():
            raise ValueError(f"{arm} produced a non-positive winner update")
        if abs(replay.final_ratings["rating_change"].sum()) > 1e-8:
            raise ValueError(f"{arm} violated season Elo conservation")
    no_xg = matches.loc[~matches["xg_analysis_eligible"]]
    if not no_xg["candidate_xg_adjustment_home_elo"].eq(0.0).all():
        raise ValueError("Missing-xG matches generated an xG adjustment")


def build_manifest(
    events: pd.DataFrame,
    final_ratings: pd.DataFrame,
    production: dict[str, object],
    bootstrap_samples: int,
) -> dict[str, object]:
    return {
        "analysis": "GOAL_ALPHA_015_XG_FULL_SEASON_REPLAY_2025_26",
        "evidence_class": "RETROSPECTIVE_COUNTERFACTUAL",
        "matches_per_arm": len(events),
        "teams": len(final_ratings),
        "xg_eligible_matches": int(events["xg_analysis_eligible"].sum()),
        "xg_missing_matches": int((~events["xg_analysis_eligible"]).sum()),
        "arms": {
            PRODUCTION_ARM: {"goal_alpha": PRODUCTION_ALPHA, "xg_enabled": False},
            CURRENT_XG_ARM: {
                "goal_alpha": PRODUCTION_ALPHA,
                "xg_ratio": XG_RATIO,
                "xg_scale": XG_SCALE,
            },
            CANDIDATE_ARM: {
                "goal_alpha": CANDIDATE_ALPHA,
                "xg_ratio": XG_RATIO,
                "xg_scale": XG_SCALE,
            },
        },
        "fixed_parameters": {
            "k_factor": production["dynamic_core"]["k_factor"],
            "elo_scale": production["dynamic_core"]["elo_scale"],
            "home_advantage": production["dynamic_core"]["home_advantage"],
            "goal_tau": production["goal_margin"]["tau"],
            "goal_cap": production["goal_margin"]["goal_difference_cap"],
            "max_match_delta": None,
        },
        "bootstrap_samples": bootstrap_samples,
        "production_changed": False,
    }


def build_report(
    comparison: pd.DataFrame,
    competition: pd.DataFrame,
    ranking: pd.DataFrame,
    uncertainty: pd.DataFrame,
    final_ratings: pd.DataFrame,
    matches: pd.DataFrame,
) -> str:
    all_scope = comparison.loc[comparison["scope"].eq("ALL_961")]
    xg_scope = comparison.loc[comparison["scope"].eq("XG_ELIGIBLE_606")]
    candidate_competition = competition.loc[
        competition["model_arm"].eq(CANDIDATE_ARM)
    ]
    ranking_view = ranking.loc[ranking["competition"].eq("ALL")]
    conservative_ci = uncertainty.loc[
        uncertainty["method"].eq("conservative_envelope")
    ]
    candidate_matches = matches
    covered = candidate_matches.loc[candidate_matches["xg_analysis_eligible"]]
    decisive = candidate_matches.loc[candidate_matches["actual_class"].ne(1)]
    final_delta = final_ratings["candidate_minus_alpha_010_xg"].abs()
    impact = pd.DataFrame(
        [
            {
                "xg_matches": len(covered),
                "mean_abs_xg_adjustment": covered[
                    "candidate_xg_adjustment_home_elo"
                ].abs().mean(),
                "mean_abs_alpha_delta_per_match": candidate_matches[
                    "candidate_power_delta_difference_vs_alpha_010_xg"
                ].abs().mean(),
                "max_abs_alpha_delta_per_match": candidate_matches[
                    "candidate_power_delta_difference_vs_alpha_010_xg"
                ].abs().max(),
                "minimum_candidate_winner_gain": decisive[
                    "alpha_015_xg_winner_elo_gain"
                ].min(),
                "median_final_abs_delta_vs_alpha_010": final_delta.median(),
                "p90_final_abs_delta_vs_alpha_010": final_delta.quantile(0.90),
                "max_final_abs_delta_vs_alpha_010": final_delta.max(),
            }
        ]
    )
    top = final_ratings.head(20)[
        [
            "alpha_015_xg_rank",
            "team_name",
            "start_rating",
            "production_final_rating",
            "alpha_010_xg_final_rating",
            "alpha_015_xg_final_rating",
            "candidate_minus_alpha_010_xg",
            "candidate_rank_change_vs_alpha_010_xg",
        ]
    ]
    increases = final_ratings.nlargest(12, "candidate_minus_alpha_010_xg")[
        [
            "team_name",
            "alpha_010_xg_final_rating",
            "alpha_015_xg_final_rating",
            "candidate_minus_alpha_010_xg",
            "candidate_rank_change_vs_alpha_010_xg",
        ]
    ]
    decreases = final_ratings.nsmallest(12, "candidate_minus_alpha_010_xg")[
        [
            "team_name",
            "alpha_010_xg_final_rating",
            "alpha_015_xg_final_rating",
            "candidate_minus_alpha_010_xg",
            "candidate_rank_change_vs_alpha_010_xg",
        ]
    ]
    return f"""# 2025/26 Goal Alpha 0.15 + xG Tam Sezon Replay

Bu calisma 961 maci uc paralel kolda kesin UTC sirasiyla isler. Her tahmin
sonuc gorulmeden uretilir, ardindan saha skoru ile Elo guncellenir.

```text
Kontrol 1: goal_alpha=0.10, xG kapali
Kontrol 2: goal_alpha=0.10, xG ratio=0.30, xG scale=1.25
Aday:      goal_alpha=0.15, xG ratio=0.30, xG scale=1.25
Sabit:     tau=300, GD cap=4, mac hareket tavani yok
xG kapsami: 606/961
```

## Tum 961 Mac

{markdown_table(all_scope)}

## xG Kapsamli 606 Mac

{markdown_table(xg_scope)}

## Adayin Turnuva Sonuclari

{markdown_table(candidate_competition)}

## Sezon Sonu Siralama Diagnostigi

{markdown_table(ranking_view)}

## Cluster Belirsizligi

{markdown_table(conservative_ci)}

## Elo Davranisi

{markdown_table(impact)}

## Aday Ilk 20

{markdown_table(top)}

## Alpha 0.10 + xG'ye Gore En Cok Yukselenler

{markdown_table(increases)}

## Alpha 0.10 + xG'ye Gore En Cok Dusenler

{markdown_table(decreases)}

Bu replay retrospective counterfactual'dir. Production aktivasyonu veya
prospective kanit yerine gecmez.
"""


if __name__ == "__main__":
    main()
