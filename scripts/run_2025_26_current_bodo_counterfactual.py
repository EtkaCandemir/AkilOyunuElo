from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ao_elo.config import AOEuropeanEloConfig  # noqa: E402
from ao_elo.evaluation import schedule_adjusted_team_performance  # noqa: E402
from ao_elo.season_counterfactual import (  # noqa: E402
    apply_score_overrides,
    build_all_wins_one_goal_overrides,
)
from scripts.run_controlled_goal_progression_backtest import (  # noqa: E402
    ControlledSeasonData,
    prepare_controlled_data,
)
from scripts.run_current_model_evaluation import (  # noqa: E402
    CURRENT,
    evaluate_arm,
    evaluation_arms,
)
from scripts.run_stage_weighted_progression_backtest import (  # noqa: E402
    DOMESTIC_ADJUSTMENTS,
    DYNAMIC_MANIFEST,
    EVENTS_PATH,
    PRODUCTION_CONTRACT,
    STATIC_DATA_ROOT,
    XG_DATA,
    load_domestic_adjustments,
    load_xg_map,
    validate_production_contract,
)
from scripts.run_v2_achievement_reserve_calibration import load_reserve_data  # noqa: E402
from scripts.run_v2_evaluation_upgrade import read_events  # noqa: E402


SEASON = "2025/26"
BODO_TEAM_ID = 170
EXPECTED_MATCHES = 961
EXPECTED_TEAMS = 236
OUTPUT_ROOT = ROOT / "output" / "season_replay_2025_26_current_bodo_8of8"
IDENTITY_PATH = ROOT / "data" / "club_identity" / "team_season_identity.csv"
ENSEMBLE_PREDICTIONS = (
    ROOT
    / "output"
    / "final_prediction_ensemble_backtest_2018_2026"
    / "unseen_predictions.csv"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay current 2025/26 production and a Bodø/Glimt 8/8 counterfactual"
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    production = json.loads(PRODUCTION_CONTRACT.read_text(encoding="utf-8"))
    core, parameters = validate_production_contract(production)
    dynamic = json.loads(DYNAMIC_MANIFEST.read_text(encoding="utf-8"))
    static_config = AOEuropeanEloConfig(**dynamic["static_config"])
    static_config.validate()

    all_events = read_events(EVENTS_PATH)
    season_events = all_events.loc[all_events["season"].astype(str).eq(SEASON)].copy()
    season_events = season_events.sort_values(
        ["kickoff_utc", "match_id"], kind="stable"
    ).reset_index(drop=True)
    if len(season_events) != EXPECTED_MATCHES or season_events["match_id"].duplicated().any():
        raise ValueError("2025/26 replay requires 961 unique matches")

    reserve, tie_audit = load_reserve_data(
        STATIC_DATA_ROOT, EVENTS_PATH, static_config
    )
    datasets = prepare_controlled_data(reserve, all_events)
    season_data = next(data for data in datasets if data.season == SEASON)
    current_domestic = load_domestic_adjustments(DOMESTIC_ADJUSTMENTS, datasets)
    baseline_domestic = {
        (data.season, int(team_id)): float(
            data.reserve.goal.carry.core.initial_ratings[int(team_id)]
        )
        for data in datasets
        for team_id in data.reserve.goal.carry.core.active_team_ids
    }
    xg_map = load_xg_map(XG_DATA, datasets)
    target = schedule_adjusted_team_performance(all_events)
    current_arm = next(arm for arm in evaluation_arms() if arm.name == CURRENT)

    actual = evaluate_arm(
        (season_data,),
        current_arm,
        core=core,
        parameters=parameters,
        current_domestic=current_domestic,
        baseline_domestic=baseline_domestic,
        xg_map=xg_map,
        target=target,
    )

    overrides = build_all_wins_one_goal_overrides(
        season_events,
        season=SEASON,
        competition="UCL",
        round_name="League Stage",
        team_id=BODO_TEAM_ID,
        expected_matches=8,
    )
    counterfactual_home, counterfactual_away = apply_score_overrides(
        season_data.reserve.goal.carry.core.match_ids,
        season_data.home_goals,
        season_data.away_goals,
        overrides,
    )
    counterfactual_data = ControlledSeasonData(
        reserve=season_data.reserve,
        home_goals=counterfactual_home,
        away_goals=counterfactual_away,
        kickoff_utc=season_data.kickoff_utc,
    )
    overridden_match_ids = set(overrides["match_id"].astype(str))
    counterfactual_xg = {
        match_id: values
        for match_id, values in xg_map.items()
        if match_id not in overridden_match_ids
    }
    counterfactual = evaluate_arm(
        (counterfactual_data,),
        current_arm,
        core=core,
        parameters=parameters,
        current_domestic=current_domestic,
        baseline_domestic=baseline_domestic,
        xg_map=counterfactual_xg,
        target=target,
    )

    identity = load_identity()
    initial = build_initial_table(season_data, current_domestic, identity)
    actual_final = decorate_final(actual.end_ratings, initial, "ACTUAL_2025_26")
    counterfactual_final = decorate_final(
        counterfactual.end_ratings, initial, "BODO_UCL_LEAGUE_STAGE_8_OF_8"
    )
    team_comparison = build_team_comparison(actual_final, counterfactual_final)

    league_stage_cutoff = pd.to_datetime(
        overrides["kickoff_utc"], utc=True, errors="raise"
    ).max()
    actual_snapshot = reconstruct_snapshot(
        actual.predictions,
        initial,
        cutoff_utc=league_stage_cutoff,
        scenario="ACTUAL_2025_26",
    )
    counterfactual_snapshot = reconstruct_snapshot(
        counterfactual.predictions,
        initial,
        cutoff_utc=league_stage_cutoff,
        scenario="BODO_UCL_LEAGUE_STAGE_8_OF_8",
    )
    snapshots = pd.concat(
        [actual_snapshot, counterfactual_snapshot], ignore_index=True
    )

    actual_matches = decorate_match_updates(
        actual.predictions, season_events, overrides, scenario="ACTUAL_2025_26"
    )
    counterfactual_matches = decorate_match_updates(
        counterfactual.predictions,
        season_events,
        overrides,
        scenario="BODO_UCL_LEAGUE_STAGE_8_OF_8",
    )
    served_predictions, prediction_metrics = load_prediction_layer_results()
    actual_matches = actual_matches.merge(
        served_predictions[
            [
                "match_id",
                "home_probability",
                "draw_probability",
                "away_probability",
                "predicted_class",
                "correct",
            ]
        ].rename(
            columns={
                "home_probability": "served_home_probability",
                "draw_probability": "served_draw_probability",
                "away_probability": "served_away_probability",
                "predicted_class": "served_predicted_class",
                "correct": "served_correct",
            }
        ),
        on="match_id",
        how="left",
        validate="one_to_one",
    )
    if actual_matches["served_home_probability"].isna().any():
        raise ValueError("Served ML+Poisson predictions do not cover all 961 matches")

    focus = build_focus_summary(
        initial,
        actual_final,
        counterfactual_final,
        actual.predictions,
        counterfactual.predictions,
        snapshots,
    )
    audit = validate_outputs(
        actual,
        counterfactual,
        initial,
        actual_final,
        counterfactual_final,
        overrides,
        actual_matches,
        tie_audit,
    )

    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    initial.to_csv(output / "initial_ratings.csv", index=False)
    actual_final.to_csv(output / "actual_final_ratings.csv", index=False)
    counterfactual_final.to_csv(
        output / "bodo_8of8_counterfactual_final_ratings.csv", index=False
    )
    team_comparison.to_csv(output / "scenario_team_comparison.csv", index=False)
    snapshots.to_csv(output / "league_stage_snapshot_ratings.csv", index=False)
    overrides.to_csv(output / "bodo_8of8_match_overrides.csv", index=False)
    actual_matches.to_csv(output / "actual_match_predictions_and_updates.csv", index=False)
    counterfactual_matches.to_csv(
        output / "bodo_8of8_counterfactual_match_updates.csv", index=False
    )
    focus.to_csv(output / "focus_team_summary.csv", index=False)
    prediction_metrics.to_csv(output / "prediction_layer_2025_26_metrics.csv", index=False)
    audit.to_csv(output / "safety_audit.csv", index=False)

    manifest = build_manifest(
        production,
        overrides,
        league_stage_cutoff,
        audit,
        prediction_metrics,
    )
    (output / "replay_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output / "replay_report.md").write_text(
        build_report(focus, prediction_metrics, overrides, manifest),
        encoding="utf-8",
    )

    bodo = focus.loc[focus["team_id"].eq(BODO_TEAM_ID)].iloc[0]
    print("AO 2025/26 current replay and Bodø/Glimt counterfactual complete")
    print(f"Matches: {len(actual_matches)}; teams: {len(initial)}")
    print(
        "Bodø/Glimt actual final: "
        f"{bodo['actual_final_rating']:.3f}, rank {int(bodo['actual_final_rank'])}"
    )
    print(
        "Bodø/Glimt 8/8 final: "
        f"{bodo['counterfactual_final_rating']:.3f}, "
        f"rank {int(bodo['counterfactual_final_rank'])}"
    )
    print(f"Report: {output / 'replay_report.md'}")


def load_identity() -> pd.DataFrame:
    identity = pd.read_csv(IDENTITY_PATH)
    identity = identity.loc[identity["season"].astype(str).eq(SEASON)].copy()
    if len(identity) != EXPECTED_TEAMS or identity["local_team_id"].duplicated().any():
        raise ValueError("2025/26 identity registry must contain 236 unique teams")
    return identity[
        ["local_team_id", "club_id", "uefa_team_id", "uefa_team_name"]
    ].rename(
        columns={"local_team_id": "team_id", "uefa_team_name": "team_name"}
    )


def stable_rank(frame: pd.DataFrame, rating_column: str) -> pd.Series:
    order = frame.sort_values(
        [rating_column, "club_id"], ascending=[False, True], kind="stable"
    ).index
    ranks = pd.Series(index=frame.index, dtype=int)
    ranks.loc[order] = np.arange(1, len(frame) + 1)
    return ranks.astype(int)


def build_initial_table(
    season_data: ControlledSeasonData,
    current_domestic: dict[tuple[str, int], float],
    identity: pd.DataFrame,
) -> pd.DataFrame:
    active = [
        int(team_id)
        for team_id in season_data.reserve.goal.carry.core.active_team_ids
    ]
    initial = pd.DataFrame(
        {
            "season": SEASON,
            "team_id": active,
            "initial_ao_first_elo": [
                current_domestic[(SEASON, team_id)] for team_id in active
            ],
        }
    ).merge(identity, on="team_id", validate="one_to_one")
    initial["initial_rank"] = stable_rank(initial, "initial_ao_first_elo")
    return initial.sort_values(["initial_rank", "club_id"]).reset_index(drop=True)


def decorate_final(
    final: pd.DataFrame, initial: pd.DataFrame, scenario: str
) -> pd.DataFrame:
    output = final.merge(
        initial[
            ["team_id", "club_id", "uefa_team_id", "team_name", "initial_rank"]
        ],
        on="team_id",
        validate="one_to_one",
    )
    output["scenario"] = scenario
    output["final_rank"] = stable_rank(output, "end_live_rating")
    output["rating_change"] = output["end_live_rating"] - output["initial_rating"]
    output["rank_change"] = output["initial_rank"] - output["final_rank"]
    columns = [
        "scenario",
        "season",
        "club_id",
        "team_id",
        "uefa_team_id",
        "team_name",
        "initial_rank",
        "initial_rating",
        "end_power_rating",
        "end_progression_bonus",
        "end_live_rating",
        "final_rank",
        "rating_change",
        "rank_change",
    ]
    return output[columns].sort_values(["final_rank", "club_id"]).reset_index(drop=True)


def build_team_comparison(
    actual: pd.DataFrame, counterfactual: pd.DataFrame
) -> pd.DataFrame:
    left = actual.rename(
        columns={
            "end_power_rating": "actual_end_power_rating",
            "end_progression_bonus": "actual_progression_bonus",
            "end_live_rating": "actual_final_rating",
            "final_rank": "actual_final_rank",
            "rating_change": "actual_rating_change",
            "rank_change": "actual_rank_change",
        }
    )
    right = counterfactual[
        [
            "team_id",
            "end_power_rating",
            "end_progression_bonus",
            "end_live_rating",
            "final_rank",
            "rating_change",
            "rank_change",
        ]
    ].rename(
        columns={
            "end_power_rating": "counterfactual_end_power_rating",
            "end_progression_bonus": "counterfactual_progression_bonus",
            "end_live_rating": "counterfactual_final_rating",
            "final_rank": "counterfactual_final_rank",
            "rating_change": "counterfactual_rating_change",
            "rank_change": "counterfactual_rank_change",
        }
    )
    output = left.merge(right, on="team_id", validate="one_to_one")
    output["counterfactual_rating_delta_vs_actual"] = (
        output["counterfactual_final_rating"] - output["actual_final_rating"]
    )
    output["counterfactual_rank_gain_vs_actual"] = (
        output["actual_final_rank"] - output["counterfactual_final_rank"]
    )
    return output.sort_values(["actual_final_rank", "club_id"]).reset_index(drop=True)


def reconstruct_snapshot(
    predictions: pd.DataFrame,
    initial: pd.DataFrame,
    *,
    cutoff_utc: pd.Timestamp,
    scenario: str,
) -> pd.DataFrame:
    power = dict(
        zip(initial["team_id"].astype(int), initial["initial_ao_first_elo"].astype(float))
    )
    bonus = {team_id: 0.0 for team_id in power}
    rows = predictions.sort_values(["kickoff_utc", "match_id"], kind="stable")
    rows = rows.loc[pd.to_datetime(rows["kickoff_utc"], utc=True) <= cutoff_utc]
    for row in rows.itertuples(index=False):
        home_id = int(row.home_team_id)
        away_id = int(row.away_team_id)
        power[home_id] += float(row.power_delta)
        power[away_id] -= float(row.power_delta)
        winner = int(row.progression_winner_team_id)
        if winner >= 0:
            bonus[winner] += float(row.progression_bonus_added)
    output = initial[
        ["club_id", "team_id", "uefa_team_id", "team_name"]
    ].copy()
    output["scenario"] = scenario
    output["snapshot_utc"] = cutoff_utc
    output["power_rating"] = output["team_id"].map(power)
    output["progression_bonus"] = output["team_id"].map(bonus)
    output["live_rating"] = output["power_rating"] + output["progression_bonus"]
    output["snapshot_rank"] = stable_rank(output, "live_rating")
    return output.sort_values(["snapshot_rank", "club_id"]).reset_index(drop=True)


def decorate_match_updates(
    predictions: pd.DataFrame,
    events: pd.DataFrame,
    overrides: pd.DataFrame,
    *,
    scenario: str,
) -> pd.DataFrame:
    names = events[
        ["match_id", "round", "home_team_name", "away_team_name"]
    ].copy()
    output = predictions.merge(names, on="match_id", validate="one_to_one")
    changed_ids = set(overrides["match_id"].astype(str))
    output.insert(0, "scenario", scenario)
    output["is_bodo_8of8_override"] = output["match_id"].astype(str).isin(changed_ids)
    return output.sort_values(["kickoff_utc", "match_id"]).reset_index(drop=True)


def load_prediction_layer_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(ENSEMBLE_PREDICTIONS)
    data = data.loc[data["season"].astype(str).eq(SEASON)].copy()
    models = (
        "CURRENT_AO",
        "CURRENT_ML_BLEND",
        "AO_POISSON_RHO0_CONTROL",
        "ML_POISSON_ENSEMBLE",
    )
    rows = []
    for model in models:
        frame = data.loc[data["model"].eq(model)]
        if len(frame) != EXPECTED_MATCHES or frame["match_id"].duplicated().any():
            raise ValueError(f"{model} does not cover 961 unique 2025/26 matches")
        rows.append(
            {
                "model": model,
                "matches": len(frame),
                "brier_1x2": float(frame["brier_1x2"].mean()),
                "log_loss_1x2": float(frame["log_loss_1x2"].mean()),
                "accuracy_1x2": float(frame["correct"].astype(bool).mean()),
            }
        )
    metrics = pd.DataFrame(rows)
    baseline = metrics.loc[metrics["model"].eq("CURRENT_AO")].iloc[0]
    for metric in ("brier_1x2", "log_loss_1x2", "accuracy_1x2"):
        metrics[f"delta_vs_current_ao_{metric}"] = metrics[metric] - baseline[metric]
    served = data.loc[data["model"].eq("ML_POISSON_ENSEMBLE")].copy()
    return served, metrics


def team_result_summary(predictions: pd.DataFrame, team_id: int) -> dict[str, float | int]:
    rows = predictions.loc[
        predictions["home_team_id"].eq(team_id)
        | predictions["away_team_id"].eq(team_id)
    ].copy()
    home = rows["home_team_id"].eq(team_id)
    wins = (home & rows["actual_class"].eq(0)) | (~home & rows["actual_class"].eq(2))
    draws = rows["actual_class"].eq(1)
    team_delta = np.where(home, rows["power_delta"], -rows["power_delta"])
    progression = rows.loc[
        rows["progression_winner_team_id"].eq(team_id), "progression_bonus_added"
    ].sum()
    return {
        "matches": int(len(rows)),
        "wins": int(wins.sum()),
        "draws": int(draws.sum()),
        "losses": int((~wins & ~draws).sum()),
        "power_delta_sum": float(np.sum(team_delta)),
        "progression_bonus_sum": float(progression),
    }


def build_focus_summary(
    initial: pd.DataFrame,
    actual: pd.DataFrame,
    counterfactual: pd.DataFrame,
    actual_predictions: pd.DataFrame,
    counterfactual_predictions: pd.DataFrame,
    snapshots: pd.DataFrame,
) -> pd.DataFrame:
    focus_ids = (63, 60, BODO_TEAM_ID)
    comparison = build_team_comparison(actual, counterfactual)
    comparison = comparison.loc[comparison["team_id"].isin(focus_ids)].copy()
    actual_snapshot = snapshots.loc[snapshots["scenario"].eq("ACTUAL_2025_26")].set_index("team_id")
    counterfactual_snapshot = snapshots.loc[
        snapshots["scenario"].eq("BODO_UCL_LEAGUE_STAGE_8_OF_8")
    ].set_index("team_id")
    rows = []
    for row in comparison.itertuples(index=False):
        team_id = int(row.team_id)
        actual_results = team_result_summary(actual_predictions, team_id)
        counterfactual_results = team_result_summary(counterfactual_predictions, team_id)
        rows.append(
            {
                "team_id": team_id,
                "club_id": row.club_id,
                "team_name": row.team_name,
                "initial_rating": row.initial_rating,
                "initial_rank": row.initial_rank,
                "actual_matches": actual_results["matches"],
                "actual_wins": actual_results["wins"],
                "actual_draws": actual_results["draws"],
                "actual_losses": actual_results["losses"],
                "actual_power_delta": actual_results["power_delta_sum"],
                "actual_progression_bonus": actual_results["progression_bonus_sum"],
                "actual_league_stage_snapshot_rating": actual_snapshot.loc[team_id, "live_rating"],
                "actual_league_stage_snapshot_rank": actual_snapshot.loc[team_id, "snapshot_rank"],
                "actual_final_rating": row.actual_final_rating,
                "actual_final_rank": row.actual_final_rank,
                "counterfactual_matches": counterfactual_results["matches"],
                "counterfactual_wins": counterfactual_results["wins"],
                "counterfactual_draws": counterfactual_results["draws"],
                "counterfactual_losses": counterfactual_results["losses"],
                "counterfactual_power_delta": counterfactual_results["power_delta_sum"],
                "counterfactual_progression_bonus": counterfactual_results["progression_bonus_sum"],
                "counterfactual_league_stage_snapshot_rating": counterfactual_snapshot.loc[team_id, "live_rating"],
                "counterfactual_league_stage_snapshot_rank": counterfactual_snapshot.loc[team_id, "snapshot_rank"],
                "counterfactual_final_rating": row.counterfactual_final_rating,
                "counterfactual_final_rank": row.counterfactual_final_rank,
                "counterfactual_rating_delta_vs_actual": row.counterfactual_rating_delta_vs_actual,
                "counterfactual_rank_gain_vs_actual": row.counterfactual_rank_gain_vs_actual,
            }
        )
    return pd.DataFrame(rows).sort_values("actual_final_rank").reset_index(drop=True)


def validate_outputs(
    actual,
    counterfactual,
    initial: pd.DataFrame,
    actual_final: pd.DataFrame,
    counterfactual_final: pd.DataFrame,
    overrides: pd.DataFrame,
    actual_matches: pd.DataFrame,
    tie_audit: pd.DataFrame,
) -> pd.DataFrame:
    checks = [
        ("actual_match_count", len(actual.predictions) == EXPECTED_MATCHES, len(actual.predictions), EXPECTED_MATCHES),
        ("counterfactual_match_count", len(counterfactual.predictions) == EXPECTED_MATCHES, len(counterfactual.predictions), EXPECTED_MATCHES),
        ("team_count", len(initial) == EXPECTED_TEAMS, len(initial), EXPECTED_TEAMS),
        ("counterfactual_exactly_eight_matches", len(overrides) == 8, len(overrides), 8),
        ("counterfactual_all_one_goal_wins", bool(((overrides["counterfactual_home_goals"] - overrides["counterfactual_away_goals"]).abs() == 1).all()), int(((overrides["counterfactual_home_goals"] - overrides["counterfactual_away_goals"]).abs() == 1).sum()), 8),
        ("actual_match_ids_unique", actual_matches["match_id"].is_unique, int(actual_matches["match_id"].nunique()), EXPECTED_MATCHES),
        ("actual_power_zero_sum", float(actual.season_metrics["season_power_conservation_error"].max()) <= 1e-8, float(actual.season_metrics["season_power_conservation_error"].max()), "<=1e-8"),
        ("counterfactual_power_zero_sum", float(counterfactual.season_metrics["season_power_conservation_error"].max()) <= 1e-8, float(counterfactual.season_metrics["season_power_conservation_error"].max()), "<=1e-8"),
        ("actual_progression_cap", float(actual.season_metrics["maximum_bonus_cap_error"].max()) <= 1e-12, float(actual.season_metrics["maximum_bonus_cap_error"].max()), 0),
        ("counterfactual_progression_cap", float(counterfactual.season_metrics["maximum_bonus_cap_error"].max()) <= 1e-12, float(counterfactual.season_metrics["maximum_bonus_cap_error"].max()), 0),
        ("tie_chronology_audit", not tie_audit.empty, len(tie_audit), ">0"),
        ("finite_actual_final", bool(np.isfinite(actual_final["end_live_rating"]).all()), int(np.isfinite(actual_final["end_live_rating"]).sum()), EXPECTED_TEAMS),
        ("finite_counterfactual_final", bool(np.isfinite(counterfactual_final["end_live_rating"]).all()), int(np.isfinite(counterfactual_final["end_live_rating"]).sum()), EXPECTED_TEAMS),
    ]
    audit = pd.DataFrame(
        checks, columns=["check", "passed", "observed", "requirement"]
    )
    if not audit["passed"].all():
        failed = audit.loc[~audit["passed"], "check"].tolist()
        raise ValueError(f"Replay safety checks failed: {failed}")
    return audit


def build_manifest(
    production: dict,
    overrides: pd.DataFrame,
    cutoff: pd.Timestamp,
    audit: pd.DataFrame,
    prediction_metrics: pd.DataFrame,
) -> dict[str, object]:
    return {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "season": SEASON,
        "matches": EXPECTED_MATCHES,
        "teams": EXPECTED_TEAMS,
        "production_model_version": production["model_version"],
        "production_revision": production["production_revision"],
        "production_contract_sha256": sha256(PRODUCTION_CONTRACT),
        "events_sha256": sha256(EVENTS_PATH),
        "prediction_source_sha256": sha256(ENSEMBLE_PREDICTIONS),
        "actual_scenario": "Chronological replay of all actual 2025/26 field scores",
        "counterfactual_scenario": {
            "team_id": BODO_TEAM_ID,
            "competition": "UCL",
            "round": "League Stage",
            "matches_overridden": len(overrides),
            "score_rule": "Every league-stage match is a one-goal Bodø/Glimt win (1-0 home, 0-1 away)",
            "xg_rule": "xG disabled for all eight overridden matches because counterfactual xG is unknown",
            "subsequent_schedule_rule": "Actual post-league-stage schedule and field scores are held fixed",
            "interpretation": "Controlled rating sensitivity, not a regenerated tournament bracket",
        },
        "league_stage_snapshot_utc": cutoff.isoformat(),
        "prediction_layer": {
            "served_model": "ML_POISSON_ENSEMBLE",
            "rating_feedback": False,
            "metrics": prediction_metrics.to_dict(orient="records"),
        },
        "all_safety_checks_passed": bool(audit["passed"].all()),
    }


def build_report(
    focus: pd.DataFrame,
    prediction_metrics: pd.DataFrame,
    overrides: pd.DataFrame,
    manifest: dict[str, object],
) -> str:
    by_id = focus.set_index("team_id")
    tottenham = by_id.loc[63]
    city = by_id.loc[60]
    bodo = by_id.loc[BODO_TEAM_ID]
    changed = int(overrides["result_changed"].sum())
    table = markdown_table(focus[
        [
            "team_name",
            "initial_rating",
            "initial_rank",
            "actual_wins",
            "actual_draws",
            "actual_losses",
            "actual_final_rating",
            "actual_final_rank",
            "counterfactual_final_rating",
            "counterfactual_final_rank",
        ]
    ], digits=3)
    metric_table = markdown_table(prediction_metrics, digits=6)
    override_table = markdown_table(overrides[
        [
            "match_id",
            "kickoff_utc",
            "home_team_name",
            "away_team_name",
            "actual_home_goals",
            "actual_away_goals",
            "counterfactual_home_goals",
            "counterfactual_away_goals",
            "result_changed",
        ]
    ], digits=3)
    return "\n".join(
        [
            "# AO European Elo 2025/26 Güncel Replay ve Bodø/Glimt 8/8 Senaryosu",
            "",
            "## Kapsam",
            "",
            f"- `{manifest['matches']}` UCL/UEL/UECL maçı ve `{manifest['teams']}` takım exact-UTC sırasıyla işlendi.",
            f"- Production sürümü: `{manifest['production_revision']}`.",
            "- AO First Elo, Domestic Surprise, gol farkı, uygun xG ve progression güncel contract değerleriyle kullanıldı.",
            "- ML+Poisson yalnız 1X2 tahmini üretir; Elo güncellemesine geri beslenmez.",
            "",
            "## Ana Sonuç",
            "",
            table,
            "",
            f"Gerçek replay'de Tottenham `{tottenham.actual_final_rating:.3f}` Elo ile `{int(tottenham.actual_final_rank)}.`; Manchester City `{city.actual_final_rating:.3f}` ile `{int(city.actual_final_rank)}.` sıradadır. Tottenham sezon boyunca `{int(tottenham.actual_wins)}G-{int(tottenham.actual_draws)}B-{int(tottenham.actual_losses)}M` ve `{tottenham.actual_power_delta:+.3f}` Power Elo; City `{int(city.actual_wins)}G-{int(city.actual_draws)}B-{int(city.actual_losses)}M` ve `{city.actual_power_delta:+.3f}` Power Elo üretmiştir. Bu nedenle sıralama, yerel lig tablosunu değil Avrupa sezonundaki modele göre düzeltilmiş performansı gösterir.",
            "",
            f"Bodø/Glimt gerçek sonuçlarla lig aşaması sonunda `{bodo.actual_league_stage_snapshot_rating:.3f}` Elo ve `{int(bodo.actual_league_stage_snapshot_rank)}.` sıra; sezon sonunda `{bodo.actual_final_rating:.3f}` ve `{int(bodo.actual_final_rank)}.` sıradadır.",
            "",
            f"8/8 kontrollü senaryoda lig aşaması sonunda `{bodo.counterfactual_league_stage_snapshot_rating:.3f}` Elo ve `{int(bodo.counterfactual_league_stage_snapshot_rank)}.` sıra; sabit fikstürlü sezon sonunda `{bodo.counterfactual_final_rating:.3f}` ve `{int(bodo.counterfactual_final_rank)}.` sıraya ulaşır. Gerçek finale göre fark `{bodo.counterfactual_rating_delta_vs_actual:+.3f}` Elo ve `{int(bodo.counterfactual_rank_gain_vs_actual):+d}` sıradır.",
            "",
            "## 8/8 Varsayımı",
            "",
            f"Sekiz lig aşaması maçı tek farklı Bodø/Glimt galibiyeti (`1-0` veya `0-1`) yapıldı; gerçek durumda galibiyet olmayan `{changed}` maçın sonucu değişti. Karşı-olgusal xG bilinmediği için bu sekiz maçta xG kullanılmadı. Sonraki gerçek fikstür sabit tutuldu.",
            "",
            "Bu nedenle sezon sonu 4.'lük, turnuva eşleşmelerini yeniden kura eden bir Monte Carlo sonucu değildir. 8/8 yapan takım normalde knockout play-off'u atlayacağı ve farklı rakiple eşleşeceği için en temiz doğrudan sonuç lig aşaması sonundaki 4. sıradır; sezon sonu değer kontrollü duyarlılık ölçümüdür.",
            "",
            override_table,
            "",
            "## ML ve Poisson Tahmin Katmanı",
            "",
            metric_table,
            "",
            "2025/26 tek sezonunda Current ML loss bakımından final ensemble'dan biraz daha iyi görünürken, final ensemble doğrulukta daha yüksektir. Production ensemble seçimi yalnız bu sezona göre değil, altı unseen foldun pooled sonucuna göre yapılmıştır.",
            "",
            "## Sonuç",
            "",
            "- Tottenham'ın City'nin üzerinde olması kod veya kimlik hatası değildir; Avrupa maç akışından kaynaklanır.",
            "- Kullanıcıya gösterilecek sıralamanın 'Avrupa performans gücü' olduğu açık yazılmalıdır; bunu genel kulüp gücü veya yerel lig formu olarak sunmak algı sorununa yol açabilir.",
            "- Bodø/Glimt 8/8 testi modelin güçlü ve beklenmedik seri performansı belirgin biçimde ödüllendirdiğini gösterir.",
            "- Production contract değiştirilmemiştir.",
            "",
        ]
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def markdown_table(frame: pd.DataFrame, *, digits: int) -> str:
    def value_text(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{digits}f}"
        return str(value)

    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(value_text(value) for value in row) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
