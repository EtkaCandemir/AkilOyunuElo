from __future__ import annotations

import argparse
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
    BASELINE_KEY,
    replay_candidate,
)


OUTPUT_ROOT = ROOT / "output" / "bounded_xg_full_season_replay_2025_26"
CANDIDATE_KEY = "BOUNDED_XG_ratio0.3_scale1.25"
MAX_XG_RATIO = 0.30
XG_SCALE = 1.25


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay the full 2025/26 UEFA season with production and bounded xG"
    )
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--static-data-root", type=Path, default=STATIC_DATA_ROOT)
    parser.add_argument("--static-manifest", type=Path, default=STATIC_MANIFEST_PATH)
    parser.add_argument("--production-model", type=Path, default=PRODUCTION_MODEL_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    events = read_fotmob_xg_dataset(args.data.resolve(), strict_contract=True)
    manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    validate_manifest(manifest, events)
    production = read_production_contract(args.production_model.resolve())
    static_config = read_static_config(args.static_manifest.resolve())
    initial_ratings = load_initial_ratings(
        args.static_data_root.resolve(), events, static_config
    )[SEASON]

    baseline_candidate = BoundedXGCandidate(BASELINE_KEY, 0.0, 1.0)
    bounded_candidate = BoundedXGCandidate(CANDIDATE_KEY, MAX_XG_RATIO, XG_SCALE)
    baseline = replay_candidate(
        events,
        initial_ratings,
        production,
        baseline_candidate,
        snapshot_times=(),
    )
    bounded = replay_candidate(
        events,
        initial_ratings,
        production,
        bounded_candidate,
        snapshot_times=(),
    )

    match_comparison = build_match_comparison(
        baseline.predictions,
        bounded.predictions,
        float(production["dynamic_core"]["k_factor"]),
    )
    long_predictions = pd.concat(
        [
            baseline.predictions.assign(model="GD_PRODUCTION"),
            bounded.predictions.assign(model="BOUNDED_XG_0.30_1.25"),
        ],
        ignore_index=True,
    )
    model_comparison = build_model_comparison(long_predictions)
    competition_summary = build_competition_summary(long_predictions)
    final_ratings = build_final_ratings(
        baseline.final_ratings,
        bounded.final_ratings,
    )
    trajectories = build_trajectories(
        baseline.predictions,
        bounded.predictions,
        float(production["dynamic_core"]["k_factor"]),
    )
    ranking_summary = build_ranking_summary(
        baseline.final_ratings,
        bounded.final_ratings,
        events,
    )
    validate_replay(
        events,
        baseline.predictions,
        bounded.predictions,
        baseline.final_ratings,
        bounded.final_ratings,
        match_comparison,
    )

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    match_comparison.to_csv(
        output_root / "match_updates_comparison.csv", index=False, float_format="%.9f"
    )
    long_predictions.to_csv(
        output_root / "arm_match_updates.csv", index=False, float_format="%.9f"
    )
    model_comparison.to_csv(output_root / "model_comparison.csv", index=False)
    competition_summary.to_csv(
        output_root / "competition_summary.csv", index=False
    )
    trajectories.to_csv(
        output_root / "team_rating_trajectories.csv", index=False, float_format="%.9f"
    )
    final_ratings.to_csv(
        output_root / "final_ratings_comparison.csv", index=False, float_format="%.9f"
    )
    ranking_summary.to_csv(output_root / "ranking_summary.csv", index=False)
    payload = {
        "analysis": "BOUNDED_XG_FULL_SEASON_REPLAY_2025_26",
        "evidence_class": "RETROSPECTIVE_COUNTERFACTUAL",
        "matches": len(events),
        "teams": len(final_ratings),
        "xg_eligible_matches": int(events["xg_analysis_eligible"].sum()),
        "xg_missing_matches": int((~events["xg_analysis_eligible"]).sum()),
        "production_arm": "GD_PRODUCTION",
        "bounded_xg_arm": {
            "key": CANDIDATE_KEY,
            "max_xg_ratio": MAX_XG_RATIO,
            "xg_scale": XG_SCALE,
            "minimum_winner_gain_ratio": 1.0 - MAX_XG_RATIO,
        },
        "production_changed": False,
    }
    (output_root / "replay_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_root / "replay_report.md").write_text(
        build_report(
            model_comparison,
            competition_summary,
            ranking_summary,
            final_ratings,
            match_comparison,
        ),
        encoding="utf-8",
    )
    print(f"Matches per arm: {len(events)}")
    print(f"Teams: {len(final_ratings)}")
    print(f"xG eligible: {int(events['xg_analysis_eligible'].sum())}")
    print(f"Report: {output_root / 'replay_report.md'}")


def build_match_comparison(
    production: pd.DataFrame,
    bounded: pd.DataFrame,
    k_factor: float,
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
    production_fields = [
        "home_rating_pre",
        "away_rating_pre",
        "expected_home_score",
        "home_probability",
        "draw_probability",
        "away_probability",
        "brier_1x2",
        "log_loss_1x2",
        "goal_multiplier",
        "power_delta",
        "winner_elo_gain",
        "home_rating_post",
        "away_rating_post",
    ]
    bounded_fields = production_fields + [
        "xg_performance_signal",
        "xg_performance_adjustment",
    ]
    left = production[metadata + production_fields].rename(
        columns={column: f"production_{column}" for column in production_fields}
    )
    right = bounded[["match_id"] + bounded_fields].rename(
        columns={column: f"bounded_{column}" for column in bounded_fields}
    )
    result = left.merge(right, on="match_id", how="inner", validate="one_to_one")
    result["bounded_xg_adjustment_home_elo"] = (
        result["bounded_xg_performance_adjustment"] * k_factor
    )
    winner_sign = result["actual_class"].map({0: 1.0, 1: 0.0, 2: -1.0})
    result["bounded_xg_adjustment_winner_elo"] = (
        winner_sign * result["bounded_xg_adjustment_home_elo"]
    )
    result["winner_gain_difference"] = (
        result["bounded_winner_elo_gain"] - result["production_winner_elo_gain"]
    )
    result["brier_difference"] = (
        result["bounded_brier_1x2"] - result["production_brier_1x2"]
    )
    result["log_loss_difference"] = (
        result["bounded_log_loss_1x2"] - result["production_log_loss_1x2"]
    )
    return result


def build_model_comparison(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scopes = {
        "ALL_961": pd.Series(True, index=predictions.index),
        "XG_ELIGIBLE_606": predictions["xg_analysis_eligible"],
        "NO_XG_355": ~predictions["xg_analysis_eligible"],
    }
    for model, model_frame in predictions.groupby("model", sort=True):
        for scope, mask in scopes.items():
            frame = model_frame.loc[mask.loc[model_frame.index]]
            rows.append({"model": model, "scope": scope, **evaluate_predictions(frame)})
    result = pd.DataFrame(rows)
    baseline = result.loc[result["model"].eq("GD_PRODUCTION")].set_index("scope")
    for metric in ("brier_1x2", "log_loss_1x2", "accuracy_1x2", "multiclass_ece"):
        result[f"{metric}_delta_vs_production"] = result.apply(
            lambda row: row[metric] - baseline.loc[row["scope"], metric], axis=1
        )
    return result


def build_competition_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "model": model,
            "competition": competition,
            **evaluate_predictions(frame),
        }
        for (model, competition), frame in predictions.groupby(
            ["model", "competition"], sort=True
        )
    ]
    result = pd.DataFrame(rows)
    baseline = result.loc[result["model"].eq("GD_PRODUCTION")].set_index(
        "competition"
    )
    for metric in ("brier_1x2", "log_loss_1x2", "accuracy_1x2"):
        result[f"{metric}_delta_vs_production"] = result.apply(
            lambda row: row[metric] - baseline.loc[row["competition"], metric], axis=1
        )
    return result


def build_final_ratings(
    production: pd.DataFrame,
    bounded: pd.DataFrame,
) -> pd.DataFrame:
    left = production[
        ["team_id", "team_name", "start_rating", "end_live_rating", "rating_change"]
    ].rename(
        columns={
            "end_live_rating": "production_final_rating",
            "rating_change": "production_season_change",
        }
    )
    right = bounded[["team_id", "end_live_rating", "rating_change"]].rename(
        columns={
            "end_live_rating": "bounded_xg_final_rating",
            "rating_change": "bounded_xg_season_change",
        }
    )
    result = left.merge(right, on="team_id", validate="one_to_one")
    result["bounded_minus_production"] = (
        result["bounded_xg_final_rating"] - result["production_final_rating"]
    )
    result["production_rank"] = result["production_final_rating"].rank(
        method="min", ascending=False
    ).astype(int)
    result["bounded_xg_rank"] = result["bounded_xg_final_rating"].rank(
        method="min", ascending=False
    ).astype(int)
    result["rank_change_vs_production"] = (
        result["production_rank"] - result["bounded_xg_rank"]
    )
    return result.sort_values(
        ["bounded_xg_final_rating", "team_id"], ascending=[False, True]
    ).reset_index(drop=True)


def build_trajectories(
    production: pd.DataFrame,
    bounded: pd.DataFrame,
    k_factor: float,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for model, values in (
        ("GD_PRODUCTION", production),
        ("BOUNDED_XG_0.30_1.25", bounded),
    ):
        home = pd.DataFrame(
            {
                "model": model,
                "match_id": values["match_id"],
                "kickoff_utc": values["kickoff_utc"],
                "competition": values["competition"],
                "team_id": values["home_team_id"],
                "team_name": values["home_team_name"],
                "side": "HOME",
                "rating_pre": values["home_rating_pre"],
                "rating_post": values["home_rating_post"],
                "power_delta": values["power_delta"],
                "xg_adjustment_elo": values["xg_performance_adjustment"] * k_factor,
            }
        )
        away = pd.DataFrame(
            {
                "model": model,
                "match_id": values["match_id"],
                "kickoff_utc": values["kickoff_utc"],
                "competition": values["competition"],
                "team_id": values["away_team_id"],
                "team_name": values["away_team_name"],
                "side": "AWAY",
                "rating_pre": values["away_rating_pre"],
                "rating_post": values["away_rating_post"],
                "power_delta": -values["power_delta"],
                "xg_adjustment_elo": -values["xg_performance_adjustment"] * k_factor,
            }
        )
        frames.extend([home, away])
    return pd.concat(frames, ignore_index=True).sort_values(
        ["kickoff_utc", "match_id", "model", "side"]
    ).reset_index(drop=True)


def build_ranking_summary(
    production: pd.DataFrame,
    bounded: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    target = schedule_adjusted_team_performance(events)
    frames = []
    for model, ratings in (
        ("GD_PRODUCTION", production),
        ("BOUNDED_XG_0.30_1.25", bounded),
    ):
        ranking = same_season_ranking(ratings, target, {SEASON}).copy()
        ranking.insert(0, "model", model)
        frames.append(ranking)
    result = pd.concat(frames, ignore_index=True)
    baseline = result.loc[result["model"].eq("GD_PRODUCTION")].set_index(
        "competition"
    )
    for metric in ("ranking_score", "pairwise_accuracy"):
        result[f"{metric}_delta_vs_production"] = result.apply(
            lambda row: row[metric] - baseline.loc[row["competition"], metric], axis=1
        )
    return result


def validate_replay(
    events: pd.DataFrame,
    production_predictions: pd.DataFrame,
    bounded_predictions: pd.DataFrame,
    production_final: pd.DataFrame,
    bounded_final: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    if len(events) != 961:
        raise ValueError(f"Expected 961 events, found {len(events)}")
    if len(production_predictions) != 961 or len(bounded_predictions) != 961:
        raise ValueError("Each replay arm must process all 961 matches")
    if production_predictions["match_id"].duplicated().any():
        raise ValueError("Production replay contains duplicate matches")
    if bounded_predictions["match_id"].duplicated().any():
        raise ValueError("Bounded replay contains duplicate matches")
    if len(production_final) != 236 or len(bounded_final) != 236:
        raise ValueError("Each arm must finish with 236 teams")
    if production_predictions["zero_sum_error"].max() > 1e-9:
        raise ValueError("Production replay violated zero-sum conservation")
    if bounded_predictions["zero_sum_error"].max() > 1e-9:
        raise ValueError("Bounded replay violated zero-sum conservation")
    decisive = bounded_predictions.loc[bounded_predictions["actual_class"].ne(1)]
    if not decisive["winner_elo_gain"].gt(0.0).all():
        raise ValueError("A decisive bounded-xG winner failed to gain Elo")
    no_xg = comparison.loc[~comparison["xg_analysis_eligible"]]
    if not no_xg["bounded_xg_adjustment_home_elo"].eq(0.0).all():
        raise ValueError("Missing-xG matches generated an xG adjustment")
    for final in (production_final, bounded_final):
        conservation = abs(final["rating_change"].sum())
        if conservation > 1e-8:
            raise ValueError("Season replay failed total Elo conservation")


def build_report(
    model_comparison: pd.DataFrame,
    competition: pd.DataFrame,
    ranking: pd.DataFrame,
    final_ratings: pd.DataFrame,
    matches: pd.DataFrame,
) -> str:
    all_scope = model_comparison.loc[model_comparison["scope"].eq("ALL_961")]
    xg_scope = model_comparison.loc[
        model_comparison["scope"].eq("XG_ELIGIBLE_606")
    ]
    ranking_view = ranking.loc[ranking["competition"].eq("ALL")]
    top_ratings = final_ratings.head(20)[
        [
            "bounded_xg_rank",
            "team_name",
            "start_rating",
            "production_final_rating",
            "bounded_xg_final_rating",
            "bounded_minus_production",
            "rank_change_vs_production",
        ]
    ]
    increases = final_ratings.nlargest(12, "bounded_minus_production")[
        ["team_name", "production_final_rating", "bounded_xg_final_rating", "bounded_minus_production"]
    ]
    decreases = final_ratings.nsmallest(12, "bounded_minus_production")[
        ["team_name", "production_final_rating", "bounded_xg_final_rating", "bounded_minus_production"]
    ]
    covered_matches = matches.loc[matches["xg_analysis_eligible"]]
    impact = pd.DataFrame(
        [
            {
                "xg_matches": len(covered_matches),
                "mean_abs_xg_adjustment": covered_matches[
                    "bounded_xg_adjustment_home_elo"
                ].abs().mean(),
                "max_positive_winner_adjustment": covered_matches[
                    "bounded_xg_adjustment_winner_elo"
                ].max(),
                "max_negative_winner_adjustment": covered_matches[
                    "bounded_xg_adjustment_winner_elo"
                ].min(),
                "max_final_rating_difference": final_ratings[
                    "bounded_minus_production"
                ].abs().max(),
            }
        ]
    )
    return f"""# 2025/26 Kontrollu xG Tam Sezon Replay

Bu replay 961 maci iki paralel kolda kesin tarih sirasiyla isler. Sonuclar
retrospective counterfactual'dir; production aktivasyon karari degildir.

```text
Production: goal alpha=0.10, tau=300, cap=4
Bounded xG: max_xg_ratio=0.30, xG_scale=1.25, minimum winner ratio=0.70
xG coverage: 606/961
```

## Tum 961 Mac

{markdown_table(all_scope)}

## xG Kapsamli 606 Mac

{markdown_table(xg_scope)}

## Turnuva Bazinda

{markdown_table(competition)}

## Siralama Diagnostigi

{markdown_table(ranking_view)}

## Mekanik Etki

{markdown_table(impact)}

## Bounded xG Ilk 20

{markdown_table(top_ratings)}

## Production'a Gore En Cok Yukselenler

{markdown_table(increases)}

## Production'a Gore En Cok Dusenler

{markdown_table(decreases)}
"""


if __name__ == "__main__":
    main()
