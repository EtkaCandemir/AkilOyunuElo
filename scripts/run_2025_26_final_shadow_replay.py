from __future__ import annotations

import argparse
import hashlib
import json
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ao_elo.club_identity import (  # noqa: E402
    attach_match_club_ids,
    permanent_club_id,
    validate_team_season_identity,
)
from ao_elo.dynamic import normalize_stage  # noqa: E402
from ao_elo.evaluation import schedule_adjusted_team_performance  # noqa: E402
from ao_elo.final_candidate import (  # noqa: E402
    FinalCandidateRuntime,
    load_final_candidate_runtime,
    update_final_candidate_match,
)
from ao_elo.pipeline import compute_ao_first_elo_from_csv  # noqa: E402
from ao_elo.robustness import one_x_two_probabilities_scalar  # noqa: E402
from ao_elo.tournament_bonus import (  # noqa: E402
    ELIGIBLE_PROGRESSION_STAGES,
    FixedTournamentBonusConfig,
    apply_tournament_progress_bonus,
)
from scripts.run_controlled_goal_progression_backtest import (  # noqa: E402
    evaluate_predictions,
)
from scripts.run_fotmob_xg_backtest_2025_26 import (  # noqa: E402
    DATA_PATH,
    MANIFEST_PATH,
    SEASON,
    STATIC_DATA_ROOT,
    STATIC_MANIFEST_PATH,
    read_fotmob_xg_dataset,
    validate_manifest,
)
from scripts.run_xg_goal_ablation_backtest import (  # noqa: E402
    elo_field_score,
    read_static_config,
    same_season_ranking,
)


FINAL_CONTRACT_PATH = ROOT / "contracts" / "ao_european_elo_v2_final_candidate.json"
IDENTITY_PATH = ROOT / "data" / "club_identity" / "team_season_identity.csv"
SURPRISE_PATH = (
    ROOT
    / "output"
    / "domestic_surprise_backtest_2018_2026"
    / "final_candidate_team_adjustments.csv"
)
SURPRISE_DECISION_PATH = (
    ROOT / "output" / "domestic_surprise_backtest_2018_2026" / "decision.json"
)
OUTPUT_ROOT = ROOT / "output" / "final_shadow_replay_2025_26"
REFERENCE_REPLAY_PATH = (
    ROOT
    / "output"
    / "goal_alpha_015_xg_full_season_replay_2025_26"
    / "final_ratings_comparison.csv"
)

MAIN_ARM = "MAIN_FINAL_CANDIDATE"
SURPRISE_ARM = "DOMESTIC_SURPRISE_SHADOW"
PROGRESSION_ARM = "PROGRESSION_12_8_4_SHADOW"
COMBINED_ARM = "SURPRISE_PLUS_PROGRESSION_SHADOW"
EXPECTED_MATCHES = 961
EXPECTED_TEAMS = 236
EXPECTED_XG_ELIGIBLE = 606
EXPECTED_BONUS_EVENTS = 69


@dataclass(frozen=True)
class ReplayArm:
    key: str
    label: str
    surprise_enabled: bool
    progression_enabled: bool
    status: str


@dataclass
class ArmReplayResult:
    predictions: pd.DataFrame
    final_ratings: pd.DataFrame
    trajectories: pd.DataFrame
    bonus_events: pd.DataFrame
    audit: dict[str, float | int]


def replay_arms() -> tuple[ReplayArm, ...]:
    return (
        ReplayArm(
            MAIN_ARM,
            "Ana model",
            surprise_enabled=False,
            progression_enabled=False,
            status="FINAL_MODEL_CANDIDATE",
        ),
        ReplayArm(
            SURPRISE_ARM,
            "Domestic Surprise shadow",
            surprise_enabled=True,
            progression_enabled=False,
            status="SHADOW_ONLY",
        ),
        ReplayArm(
            PROGRESSION_ARM,
            "Progression 12/8/4 shadow",
            surprise_enabled=False,
            progression_enabled=True,
            status="SHADOW_ONLY",
        ),
        ReplayArm(
            COMBINED_ARM,
            "Surprise + Progression shadow",
            surprise_enabled=True,
            progression_enabled=True,
            status="INTERACTION_DIAGNOSTIC_ONLY",
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the complete 2025/26 UEFA season with the AO final "
            "candidate and two isolated shadow features"
        )
    )
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--xg-manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--static-data-root", type=Path, default=STATIC_DATA_ROOT)
    parser.add_argument("--static-manifest", type=Path, default=STATIC_MANIFEST_PATH)
    parser.add_argument("--identity", type=Path, default=IDENTITY_PATH)
    parser.add_argument("--surprise", type=Path, default=SURPRISE_PATH)
    parser.add_argument("--surprise-decision", type=Path, default=SURPRISE_DECISION_PATH)
    parser.add_argument("--final-contract", type=Path, default=FINAL_CONTRACT_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    inputs = load_and_validate_inputs(args)
    events = inputs["events"]
    initial = inputs["initial"]
    runtime = inputs["runtime"]

    results = {
        arm.key: replay_arm(events, initial, runtime, arm)
        for arm in replay_arms()
    }
    outputs = build_outputs(events, initial, results)
    audits = validate_complete_replay(events, initial, results, outputs)

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    write_outputs(output_root, outputs, audits)
    manifest = build_output_manifest(args, inputs, results, outputs, audits)
    (output_root / "replay_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_root / "replay_report.md").write_text(
        build_markdown_report(outputs, manifest), encoding="utf-8"
    )

    print(f"Matches per arm: {len(events)}")
    print(f"Permanent clubs: {len(initial)}")
    print(f"xG eligible: {int(events['xg_analysis_eligible'].sum())}")
    print(f"Progression events per progression arm: {EXPECTED_BONUS_EVENTS}")
    print(f"Identity checks passed: {int(audits['passed'].sum())}/{len(audits)}")
    print(f"Comparison CSV: {output_root / 'initial_and_final_ratings.csv'}")
    print(f"Report: {output_root / 'replay_report.md'}")


def load_and_validate_inputs(args: argparse.Namespace) -> dict[str, object]:
    events = read_fotmob_xg_dataset(args.data.resolve(), strict_contract=True)
    xg_manifest = json.loads(args.xg_manifest.resolve().read_text(encoding="utf-8"))
    validate_manifest(xg_manifest, events)
    events = events.copy()
    for column in ("is_tie_decider", "is_knockout"):
        events[column] = coerce_bool(events[column], column)
    events["advanced_team_id"] = pd.to_numeric(
        events["advanced_team_id"], errors="coerce"
    ).astype("Int64")

    identity = pd.read_csv(args.identity.resolve())
    validate_team_season_identity(identity)
    identity = identity.loc[identity["season"].astype(str).eq(SEASON)].copy()
    if len(identity) != EXPECTED_TEAMS:
        raise ValueError(f"Expected {EXPECTED_TEAMS} team identities for {SEASON}")
    events = attach_match_club_ids(events, identity)
    validate_event_identity(events, identity)

    static_config = read_static_config(args.static_manifest.resolve())
    season_folder = args.static_data_root.resolve() / SEASON.replace("/", "-")
    initial = compute_ao_first_elo_from_csv(
        season_folder / "teams.csv",
        season_folder / "country_coefficients.csv",
        season_folder / "domestic_context.csv",
        season_folder / "club_european_points.csv",
        static_config,
    )
    initial = initial.merge(
        identity[
            [
                "local_team_id",
                "club_id",
                "uefa_team_id",
                "uefa_team_name",
                "identity_status",
                "event_identity_verified",
            ]
        ].rename(columns={"local_team_id": "team_id"}),
        on="team_id",
        how="left",
        validate="one_to_one",
    )
    surprise = pd.read_csv(args.surprise.resolve())
    surprise = surprise.loc[surprise["season"].astype(str).eq(SEASON)].copy()
    decision = json.loads(args.surprise_decision.resolve().read_text(encoding="utf-8"))
    if decision.get("decision") not in {
        "KEEP_BASELINE",
        "SHADOW_GAMMA_RECOMMENDATION",
    }:
        raise ValueError("Domestic Surprise must remain a non-production shadow")
    if bool(decision.get("production_change", False)):
        raise ValueError("Domestic Surprise replay input must not change production")
    surprise = normalize_surprise_input(surprise)
    initial = initial.merge(
        surprise[
            [
                "team_id",
                "club_id",
                "historical_finish_score",
                "history_seasons",
                "history_reliability",
                "surprise_score",
                "surprise_direction",
                "surprise_component",
                "ao_first_elo_adjustment",
                "adjusted_ao_first_elo",
            ]
        ].rename(columns={"club_id": "surprise_club_id"}),
        on="team_id",
        how="left",
        validate="one_to_one",
    )
    validate_initial_ratings(initial)

    participation = competition_participation(events)
    initial = initial.merge(participation, on="club_id", validate="one_to_one")
    initial["main_initial_rating"] = initial["ao_first_elo"].astype(float)
    initial["surprise_initial_rating"] = initial["adjusted_ao_first_elo"].astype(float)
    initial["main_initial_rank"] = stable_rank(
        initial, "main_initial_rating", tie_column="club_id"
    )
    initial["surprise_initial_rank"] = stable_rank(
        initial, "surprise_initial_rating", tie_column="club_id"
    )

    runtime = load_final_candidate_runtime(args.final_contract.resolve())
    return {
        "events": events,
        "identity": identity,
        "initial": initial,
        "runtime": runtime,
        "xg_manifest": xg_manifest,
        "surprise_decision": decision,
    }


def normalize_surprise_input(surprise: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "historical_mean": "historical_finish_score",
        "consistency_multiplier": "history_reliability",
        "raw_surprise": "surprise_score",
        "domestic_prior_adjustment": "surprise_component",
    }
    result = surprise.copy()
    for source, target in aliases.items():
        if target not in result.columns and source in result.columns:
            result[target] = result[source]
    required = {
        "team_id",
        "club_id",
        "historical_finish_score",
        "history_seasons",
        "history_reliability",
        "surprise_score",
        "surprise_direction",
        "surprise_component",
        "ao_first_elo_adjustment",
        "adjusted_ao_first_elo",
    }
    missing = sorted(required - set(result.columns))
    if missing:
        raise ValueError(f"Domestic Surprise input is missing columns: {missing}")
    return result


def validate_event_identity(events: pd.DataFrame, identity: pd.DataFrame) -> None:
    local_to_uefa = dict(
        zip(identity["local_team_id"].astype(int), identity["uefa_team_id"].astype(int))
    )
    expected_home = events["home_team_id"].map(local_to_uefa)
    expected_away = events["away_team_id"].map(local_to_uefa)
    if expected_home.isna().any() or expected_away.isna().any():
        raise ValueError("A replay team is missing from the permanent identity registry")
    if not expected_home.astype(int).eq(events["uefa_home_team_id"].astype(int)).all():
        raise ValueError("Home AO team_id and UEFA team ID disagree")
    if not expected_away.astype(int).eq(events["uefa_away_team_id"].astype(int)).all():
        raise ValueError("Away AO team_id and UEFA team ID disagree")
    expected_home_club = events["uefa_home_team_id"].map(permanent_club_id)
    expected_away_club = events["uefa_away_team_id"].map(permanent_club_id)
    if not expected_home_club.eq(events["home_club_id"]).all():
        raise ValueError("Home permanent club_id and UEFA team ID disagree")
    if not expected_away_club.eq(events["away_club_id"]).all():
        raise ValueError("Away permanent club_id and UEFA team ID disagree")


def validate_initial_ratings(initial: pd.DataFrame) -> None:
    if len(initial) != EXPECTED_TEAMS or initial["team_id"].duplicated().any():
        raise ValueError("Initial AO First Elo must contain 236 unique local teams")
    if initial["club_id"].isna().any() or initial["club_id"].duplicated().any():
        raise ValueError("Initial AO First Elo must contain 236 unique club_id values")
    if initial["surprise_club_id"].isna().any():
        raise ValueError("Domestic Surprise is missing permanent club identity")
    if not initial["club_id"].eq(initial["surprise_club_id"]).all():
        raise ValueError("Domestic Surprise club_id does not match the registry")
    if not np.allclose(
        initial["ao_first_elo"],
        initial["adjusted_ao_first_elo"] - initial["ao_first_elo_adjustment"],
        atol=1e-9,
    ):
        raise ValueError("Domestic Surprise adjustment does not reconcile to AO First Elo")
    numeric = initial[["ao_first_elo", "adjusted_ao_first_elo"]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("Initial ratings must be finite")


def replay_arm(
    events: pd.DataFrame,
    initial: pd.DataFrame,
    runtime: FinalCandidateRuntime,
    arm: ReplayArm,
) -> ArmReplayResult:
    start_column = "surprise_initial_rating" if arm.surprise_enabled else "main_initial_rating"
    power = dict(zip(initial["team_id"].astype(int), initial[start_column].astype(float)))
    start = dict(power)
    names = dict(zip(initial["team_id"].astype(int), initial["team_name"].astype(str)))
    club_ids = dict(zip(initial["team_id"].astype(int), initial["club_id"].astype(str)))
    bonus = {
        team_id: {competition: 0.0 for competition in ("UCL", "UEL", "UECL")}
        for team_id in power
    }
    processed_ties: set[str] = set()
    bonus_config = FixedTournamentBonusConfig(12.0)
    prediction_rows: list[dict[str, object]] = []
    trajectory_rows: list[dict[str, object]] = []
    bonus_rows: list[dict[str, object]] = []
    start_power_total = float(sum(power.values()))
    maximum_pair_error = 0.0

    for event_index, row in enumerate(events.itertuples(index=False), start=1):
        home_id = int(row.home_team_id)
        away_id = int(row.away_team_id)
        home_bonus_pre = sum(bonus[home_id].values())
        away_bonus_pre = sum(bonus[away_id].values())
        home_power_pre = power[home_id]
        away_power_pre = power[away_id]
        home_live_pre = home_power_pre + home_bonus_pre
        away_live_pre = away_power_pre + away_bonus_pre
        penalties = bool(row.decided_on_penalties)
        elo_home_goals, elo_away_goals = elo_field_score(
            int(row.home_goals), int(row.away_goals), penalties
        )
        eligible = bool(row.xg_analysis_eligible)
        xg_home = float(row.xg_home) if eligible else None
        xg_away = float(row.xg_away) if eligible else None
        update = update_final_candidate_match(
            runtime,
            home_live_pre,
            away_live_pre,
            elo_home_goals,
            elo_away_goals,
            is_neutral=bool(row.is_neutral),
            decided_on_penalties=penalties,
            xg_home=xg_home,
            xg_away=xg_away,
        )
        probabilities = one_x_two_probabilities_scalar(
            update.expected_home_score,
            runtime.dynamic_config.draw_at_even,
            runtime.dynamic_config.draw_shape,
        )
        actual_home = float(row.actual_home_score)
        observed = 0 if actual_home == 1.0 else 1 if actual_home == 0.5 else 2
        if observed == 0 and update.power_delta <= 0.0:
            raise ValueError(f"{row.match_id}: home winner did not gain Power Elo")
        if observed == 2 and update.power_delta >= 0.0:
            raise ValueError(f"{row.match_id}: away winner did not gain Power Elo")
        target = np.eye(3, dtype=float)[observed]
        brier = float(np.square(np.asarray(probabilities) - target).sum())
        log_loss = -math.log(max(probabilities[observed], 1e-15))

        power[home_id] = home_power_pre + update.power_delta
        power[away_id] = away_power_pre - update.power_delta
        maximum_pair_error = max(maximum_pair_error, update.zero_sum_error)

        competition = str(row.competition)
        stage = normalize_stage(str(row.round), competition, bool(row.is_knockout))
        applied_bonus = 0.0
        bonus_winner_id: int | None = None
        if (
            arm.progression_enabled
            and bool(row.is_tie_decider)
            and stage in ELIGIBLE_PROGRESSION_STAGES
        ):
            if pd.isna(row.tie_id) or pd.isna(row.advanced_team_id):
                raise ValueError(f"{row.match_id}: progression event is missing tie/winner")
            tie_id = str(row.tie_id)
            bonus_winner_id = int(row.advanced_team_id)
            if bonus_winner_id not in {home_id, away_id}:
                raise ValueError(f"{row.match_id}: advanced team is not a participant")
            bonus_update = apply_tournament_progress_bonus(
                bonus[bonus_winner_id][competition],
                competition,
                stage,
                tie_id,
                processed_ties,
                bonus_config,
            )
            bonus[bonus_winner_id][competition] = bonus_update.bonus_post
            applied_bonus = bonus_update.applied_bonus
            bonus_rows.append(
                {
                    "model_arm": arm.key,
                    "event_order": event_index,
                    "match_id": str(row.match_id),
                    "kickoff_utc": row.kickoff_utc,
                    "competition": competition,
                    "round": str(row.round),
                    "stage": stage,
                    "tie_id": tie_id,
                    "winner_team_id": bonus_winner_id,
                    "winner_club_id": club_ids[bonus_winner_id],
                    "winner_team_name": names[bonus_winner_id],
                    "applied_bonus": applied_bonus,
                    "competition_bonus_post": bonus_update.bonus_post,
                    "competition_cap": bonus_update.competition_cap,
                }
            )

        home_bonus_post = sum(bonus[home_id].values())
        away_bonus_post = sum(bonus[away_id].values())
        home_live_post = power[home_id] + home_bonus_post
        away_live_post = power[away_id] + away_bonus_post
        winner_gain = abs(update.power_delta) if observed != 1 else 0.0
        prediction_rows.append(
            {
                "model_arm": arm.key,
                "arm_status": arm.status,
                "event_order": event_index,
                "match_id": str(row.match_id),
                "season": SEASON,
                "kickoff_utc": row.kickoff_utc,
                "competition": competition,
                "round": str(row.round),
                "stage": stage,
                "tie_id": None if pd.isna(row.tie_id) else str(row.tie_id),
                "is_tie_decider": bool(row.is_tie_decider),
                "home_team_id": home_id,
                "away_team_id": away_id,
                "home_club_id": club_ids[home_id],
                "away_club_id": club_ids[away_id],
                "home_team_name": names[home_id],
                "away_team_name": names[away_id],
                "home_goals": int(row.home_goals),
                "away_goals": int(row.away_goals),
                "decided_on_penalties": penalties,
                "xg_analysis_eligible": eligible,
                "xg_home": xg_home,
                "xg_away": xg_away,
                "home_power_pre": home_power_pre,
                "away_power_pre": away_power_pre,
                "home_bonus_pre": home_bonus_pre,
                "away_bonus_pre": away_bonus_pre,
                "home_live_pre": home_live_pre,
                "away_live_pre": away_live_pre,
                "expected_home_score": update.expected_home_score,
                "home_probability": probabilities[0],
                "draw_probability": probabilities[1],
                "away_probability": probabilities[2],
                "actual_class": observed,
                "predicted_class": int(np.argmax(probabilities)),
                "brier_1x2": brier,
                "log_loss_1x2": log_loss,
                "goal_difference": update.goal_difference,
                "goal_multiplier": update.goal_difference_multiplier,
                "goal_bonus_residual": update.goal_bonus_residual,
                "xg_performance_signal": update.xg_performance_signal,
                "xg_performance_adjustment": update.xg_performance_adjustment,
                "power_delta": update.power_delta,
                "winner_elo_gain": winner_gain,
                "home_power_post": power[home_id],
                "away_power_post": power[away_id],
                "bonus_applied_after_match": applied_bonus,
                "bonus_winner_team_id": bonus_winner_id,
                "home_bonus_post": home_bonus_post,
                "away_bonus_post": away_bonus_post,
                "home_live_post": home_live_post,
                "away_live_post": away_live_post,
                "zero_sum_error": update.zero_sum_error,
            }
        )
        for side, team_id, power_pre, bonus_pre, power_post, bonus_post in (
            ("HOME", home_id, home_power_pre, home_bonus_pre, power[home_id], home_bonus_post),
            ("AWAY", away_id, away_power_pre, away_bonus_pre, power[away_id], away_bonus_post),
        ):
            trajectory_rows.append(
                {
                    "model_arm": arm.key,
                    "event_order": event_index,
                    "match_id": str(row.match_id),
                    "kickoff_utc": row.kickoff_utc,
                    "competition": competition,
                    "club_id": club_ids[team_id],
                    "team_id": team_id,
                    "team_name": names[team_id],
                    "side": side,
                    "power_pre": power_pre,
                    "bonus_pre": bonus_pre,
                    "live_pre": power_pre + bonus_pre,
                    "power_post": power_post,
                    "bonus_post": bonus_post,
                    "live_post": power_post + bonus_post,
                    "power_delta": power_post - power_pre,
                    "bonus_delta": bonus_post - bonus_pre,
                }
            )

    final_rows = []
    for row in initial.itertuples(index=False):
        team_id = int(row.team_id)
        end_bonus = sum(bonus[team_id].values())
        final_rows.append(
            {
                "model_arm": arm.key,
                "arm_label": arm.label,
                "arm_status": arm.status,
                "season": SEASON,
                "club_id": str(row.club_id),
                "team_id": team_id,
                "team_name": str(row.team_name),
                "country_code": str(row.country_code),
                "participating_competitions": str(row.participating_competitions),
                "start_rating": start[team_id],
                "end_power_rating": power[team_id],
                "end_progression_bonus": end_bonus,
                "end_live_rating": power[team_id] + end_bonus,
                "power_change": power[team_id] - start[team_id],
                "live_change": power[team_id] + end_bonus - start[team_id],
            }
        )
    final = pd.DataFrame(final_rows)
    final["start_rank"] = stable_rank(final, "start_rating", tie_column="club_id")
    final["end_rank"] = stable_rank(final, "end_live_rating", tie_column="club_id")
    final["rank_gain"] = final["start_rank"] - final["end_rank"]
    end_power_total = float(final["end_power_rating"].sum())
    total_bonus = float(final["end_progression_bonus"].sum())
    audit = {
        "matches": len(prediction_rows),
        "teams": len(final),
        "bonus_events": len(bonus_rows),
        "power_total_error": abs(end_power_total - start_power_total),
        "maximum_match_zero_sum_error": maximum_pair_error,
        "total_bonus_added": total_bonus,
        "live_accounting_error": abs(
            float(final["end_live_rating"].sum()) - end_power_total - total_bonus
        ),
        "maximum_team_bonus": float(final["end_progression_bonus"].max()),
    }
    return ArmReplayResult(
        predictions=pd.DataFrame(prediction_rows),
        final_ratings=final.sort_values(["end_rank", "club_id"]).reset_index(drop=True),
        trajectories=pd.DataFrame(trajectory_rows),
        bonus_events=pd.DataFrame(bonus_rows),
        audit=audit,
    )


def build_outputs(
    events: pd.DataFrame,
    initial: pd.DataFrame,
    results: dict[str, ArmReplayResult],
) -> dict[str, pd.DataFrame]:
    predictions = pd.concat(
        [results[arm.key].predictions for arm in replay_arms()], ignore_index=True
    )
    finals = pd.concat(
        [results[arm.key].final_ratings for arm in replay_arms()], ignore_index=True
    )
    trajectories = pd.concat(
        [results[arm.key].trajectories for arm in replay_arms()], ignore_index=True
    )
    bonus_frames = [
        results[arm.key].bonus_events
        for arm in replay_arms()
        if not results[arm.key].bonus_events.empty
    ]
    bonus_events = pd.concat(bonus_frames, ignore_index=True)
    comparison = build_initial_final_comparison(initial, results)
    model_comparison = build_model_comparison(results)
    competition_summary = build_competition_summary(predictions)
    ranking_summary = build_ranking_summary(events, results)
    shadow_summary = build_shadow_summary(comparison, model_comparison, results)
    return {
        "initial_ratings": initial,
        "main_initial_vs_final": comparison[
            [
                "season",
                "club_id",
                "team_id",
                "uefa_team_id",
                "team_name",
                "country_code",
                "participating_competitions",
                "main_start_rating",
                "main_start_rank",
                "main_end_power_rating",
                "main_end_live_rating",
                "main_end_rank",
                "main_live_change",
                "main_rank_gain",
            ]
        ],
        "initial_and_final_ratings": comparison,
        "final_ratings_long": finals,
        "match_predictions_and_updates": predictions,
        "team_rating_trajectories": trajectories,
        "bonus_events": bonus_events,
        "model_comparison": model_comparison,
        "competition_summary": competition_summary,
        "same_season_ranking_diagnostic": ranking_summary,
        "shadow_effect_summary": shadow_summary,
    }


def build_initial_final_comparison(
    initial: pd.DataFrame,
    results: dict[str, ArmReplayResult],
) -> pd.DataFrame:
    base_columns = [
        "season",
        "club_id",
        "team_id",
        "uefa_team_id",
        "team_name",
        "country_code",
        "country",
        "domestic_league",
        "participating_competitions",
        "competition",
        "entry_round",
        "domestic_position",
        "league_team_count",
        "domestic_prior",
        "european_prior",
        "european_exposure",
        "effective_european_exposure",
        "rating_source_type",
        "historical_finish_score",
        "history_seasons",
        "history_reliability",
        "surprise_score",
        "surprise_direction",
        "ao_first_elo_adjustment",
    ]
    output = initial[base_columns].copy()
    prefixes = {
        MAIN_ARM: "main",
        SURPRISE_ARM: "surprise",
        PROGRESSION_ARM: "progression",
        COMBINED_ARM: "combined",
    }
    for arm in replay_arms():
        prefix = prefixes[arm.key]
        frame = results[arm.key].final_ratings[
            [
                "club_id",
                "start_rating",
                "start_rank",
                "end_power_rating",
                "end_progression_bonus",
                "end_live_rating",
                "end_rank",
                "power_change",
                "live_change",
                "rank_gain",
            ]
        ].rename(
            columns={
                column: f"{prefix}_{column}"
                for column in (
                    "start_rating",
                    "start_rank",
                    "end_power_rating",
                    "end_progression_bonus",
                    "end_live_rating",
                    "end_rank",
                    "power_change",
                    "live_change",
                    "rank_gain",
                )
            }
        )
        output = output.merge(frame, on="club_id", validate="one_to_one")
    for prefix in ("surprise", "progression", "combined"):
        output[f"{prefix}_final_delta_vs_main"] = (
            output[f"{prefix}_end_live_rating"] - output["main_end_live_rating"]
        )
        output[f"{prefix}_rank_change_vs_main"] = (
            output["main_end_rank"] - output[f"{prefix}_end_rank"]
        )
    return output.sort_values(["main_end_rank", "club_id"]).reset_index(drop=True)


def build_model_comparison(results: dict[str, ArmReplayResult]) -> pd.DataFrame:
    rows = []
    for arm in replay_arms():
        replay = results[arm.key]
        metrics = evaluate_predictions(replay.predictions)
        final = replay.final_ratings
        rows.append(
            {
                "model_arm": arm.key,
                "arm_label": arm.label,
                "arm_status": arm.status,
                **metrics,
                "start_rating_min": float(final["start_rating"].min()),
                "start_rating_max": float(final["start_rating"].max()),
                "final_rating_min": float(final["end_live_rating"].min()),
                "final_rating_max": float(final["end_live_rating"].max()),
                "final_rating_mean": float(final["end_live_rating"].mean()),
                "final_rating_std": float(final["end_live_rating"].std(ddof=0)),
                "mean_abs_match_delta": float(replay.predictions["power_delta"].abs().mean()),
                "max_abs_match_delta": float(replay.predictions["power_delta"].abs().max()),
                "start_end_spearman": float(
                    final["start_rating"].corr(final["end_live_rating"], method="spearman")
                ),
                **replay.audit,
            }
        )
    result = pd.DataFrame(rows)
    baseline = result.loc[result["model_arm"].eq(MAIN_ARM)].iloc[0]
    for metric in ("brier_1x2", "log_loss_1x2", "accuracy_1x2", "multiclass_ece"):
        result[f"{metric}_delta_vs_main"] = result[metric] - baseline[metric]
    return result


def build_competition_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (arm, competition), frame in predictions.groupby(
        ["model_arm", "competition"], sort=True
    ):
        rows.append({"model_arm": arm, "competition": competition, **evaluate_predictions(frame)})
    result = pd.DataFrame(rows)
    baseline = result.loc[result["model_arm"].eq(MAIN_ARM)].set_index("competition")
    for metric in ("brier_1x2", "log_loss_1x2", "accuracy_1x2"):
        result[f"{metric}_delta_vs_main"] = result.apply(
            lambda row: row[metric] - baseline.loc[row["competition"], metric], axis=1
        )
    return result


def build_ranking_summary(
    events: pd.DataFrame, results: dict[str, ArmReplayResult]
) -> pd.DataFrame:
    target = schedule_adjusted_team_performance(events)
    rows = []
    for arm in replay_arms():
        final = results[arm.key].final_ratings[
            ["season", "team_id", "end_live_rating"]
        ]
        values = same_season_ranking(final, target, {SEASON}).copy()
        values.insert(0, "model_arm", arm.key)
        rows.append(values)
    result = pd.concat(rows, ignore_index=True)
    baseline = result.loc[result["model_arm"].eq(MAIN_ARM)].set_index("competition")
    for metric in ("ranking_score", "pairwise_accuracy"):
        result[f"{metric}_delta_vs_main"] = result.apply(
            lambda row: row[metric] - baseline.loc[row["competition"], metric], axis=1
        )
    return result


def build_shadow_summary(
    comparison: pd.DataFrame,
    metrics: pd.DataFrame,
    results: dict[str, ArmReplayResult],
) -> pd.DataFrame:
    rows = []
    main_metric = metrics.loc[metrics["model_arm"].eq(MAIN_ARM)].iloc[0]
    prefixes = {
        MAIN_ARM: "main",
        SURPRISE_ARM: "surprise",
        PROGRESSION_ARM: "progression",
        COMBINED_ARM: "combined",
    }
    for arm in replay_arms():
        prefix = prefixes[arm.key]
        final_delta = comparison[f"{prefix}_end_live_rating"] - comparison["main_end_live_rating"]
        rank_delta = comparison["main_end_rank"] - comparison[f"{prefix}_end_rank"]
        metric = metrics.loc[metrics["model_arm"].eq(arm.key)].iloc[0]
        rows.append(
            {
                "model_arm": arm.key,
                "arm_label": arm.label,
                "arm_status": arm.status,
                "teams": len(comparison),
                "teams_with_changed_initial_rating": int(
                    (comparison[f"{prefix}_start_rating"] - comparison["main_start_rating"])
                    .abs()
                    .gt(1e-9)
                    .sum()
                ),
                "mean_abs_initial_delta": float(
                    (comparison[f"{prefix}_start_rating"] - comparison["main_start_rating"])
                    .abs()
                    .mean()
                ),
                "teams_with_final_rating_change": int(final_delta.abs().gt(1e-9).sum()),
                "mean_abs_final_delta_vs_main": float(final_delta.abs().mean()),
                "max_final_gain_vs_main": float(final_delta.max()),
                "max_final_loss_vs_main": float(final_delta.min()),
                "teams_with_rank_change": int(rank_delta.ne(0).sum()),
                "maximum_rank_gain_vs_main": int(rank_delta.max()),
                "maximum_rank_loss_vs_main": int(rank_delta.min()),
                "total_progression_bonus": float(results[arm.key].audit["total_bonus_added"]),
                "bonus_events": int(results[arm.key].audit["bonus_events"]),
                "brier_1x2": float(metric["brier_1x2"]),
                "brier_delta_vs_main": float(metric["brier_1x2"] - main_metric["brier_1x2"]),
                "log_loss_1x2": float(metric["log_loss_1x2"]),
                "log_loss_delta_vs_main": float(
                    metric["log_loss_1x2"] - main_metric["log_loss_1x2"]
                ),
            }
        )
    return pd.DataFrame(rows)


def validate_complete_replay(
    events: pd.DataFrame,
    initial: pd.DataFrame,
    results: dict[str, ArmReplayResult],
    outputs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    checks: list[dict[str, object]] = []

    def add(name: str, passed: bool, observed: object, expected: object) -> None:
        checks.append(
            {"check": name, "passed": bool(passed), "observed": observed, "expected": expected}
        )
        if not passed:
            raise ValueError(f"Audit failed: {name}; observed={observed}, expected={expected}")

    add("match_count", len(events) == EXPECTED_MATCHES, len(events), EXPECTED_MATCHES)
    add("match_id_unique", events["match_id"].nunique() == EXPECTED_MATCHES, events["match_id"].nunique(), EXPECTED_MATCHES)
    chronological = events[["kickoff_utc", "match_id"]].reset_index(drop=True).equals(
        events.sort_values(["kickoff_utc", "match_id"], kind="stable")[["kickoff_utc", "match_id"]].reset_index(drop=True)
    )
    add("chronological_order", chronological, chronological, True)
    add("event_order_unique", events["event_order"].nunique() == EXPECTED_MATCHES, events["event_order"].nunique(), EXPECTED_MATCHES)
    add("team_count", len(initial) == EXPECTED_TEAMS, len(initial), EXPECTED_TEAMS)
    add("club_id_unique", initial["club_id"].nunique() == EXPECTED_TEAMS, initial["club_id"].nunique(), EXPECTED_TEAMS)
    add("uefa_team_id_unique", initial["uefa_team_id"].nunique() == EXPECTED_TEAMS, initial["uefa_team_id"].nunique(), EXPECTED_TEAMS)
    add("xg_eligible", int(events["xg_analysis_eligible"].sum()) == EXPECTED_XG_ELIGIBLE, int(events["xg_analysis_eligible"].sum()), EXPECTED_XG_ELIGIBLE)
    add("identity_status_verified", initial["identity_status"].eq("VERIFIED").all(), int(initial["identity_status"].eq("VERIFIED").sum()), EXPECTED_TEAMS)
    add("event_identity_verified", initial["event_identity_verified"].astype(bool).all(), int(initial["event_identity_verified"].astype(bool).sum()), EXPECTED_TEAMS)

    for arm in replay_arms():
        replay = results[arm.key]
        add(f"{arm.key}_matches", len(replay.predictions) == EXPECTED_MATCHES, len(replay.predictions), EXPECTED_MATCHES)
        add(f"{arm.key}_match_ids_unique", replay.predictions["match_id"].nunique() == EXPECTED_MATCHES, replay.predictions["match_id"].nunique(), EXPECTED_MATCHES)
        add(f"{arm.key}_teams", len(replay.final_ratings) == EXPECTED_TEAMS, len(replay.final_ratings), EXPECTED_TEAMS)
        add(f"{arm.key}_zero_sum", replay.audit["power_total_error"] <= 1e-8 and replay.audit["maximum_match_zero_sum_error"] <= 1e-9, max(float(replay.audit["power_total_error"]), float(replay.audit["maximum_match_zero_sum_error"])), "<=1e-8")
        expected_events = EXPECTED_BONUS_EVENTS if arm.progression_enabled else 0
        add(f"{arm.key}_bonus_events", int(replay.audit["bonus_events"]) == expected_events, int(replay.audit["bonus_events"]), expected_events)
        expected_cap = 60.0 if arm.progression_enabled else 0.0
        add(f"{arm.key}_bonus_cap", float(replay.audit["maximum_team_bonus"]) <= expected_cap + 1e-9, float(replay.audit["maximum_team_bonus"]), f"<={expected_cap}")
        decisive = replay.predictions.loc[replay.predictions["actual_class"].ne(1)]
        add(f"{arm.key}_winner_positive", decisive["winner_elo_gain"].gt(0.0).all(), int(decisive["winner_elo_gain"].gt(0.0).sum()), len(decisive))
        directions_ok = bool(
            replay.predictions.loc[replay.predictions["actual_class"].eq(0), "power_delta"].gt(0.0).all()
            and replay.predictions.loc[replay.predictions["actual_class"].eq(2), "power_delta"].lt(0.0).all()
        )
        add(f"{arm.key}_winner_direction", directions_ok, directions_ok, True)
        probabilities = replay.predictions[["home_probability", "draw_probability", "away_probability"]]
        probability_error = float((probabilities.sum(axis=1) - 1.0).abs().max())
        add(f"{arm.key}_probability_sum", probability_error <= 1e-12, probability_error, "<=1e-12")
        no_xg = replay.predictions.loc[~replay.predictions["xg_analysis_eligible"]]
        no_xg_error = float(no_xg["xg_performance_adjustment"].abs().max())
        add(f"{arm.key}_xg_fallback", no_xg_error <= 1e-12, no_xg_error, "<=1e-12")

    main_final = results[MAIN_ARM].final_ratings
    if REFERENCE_REPLAY_PATH.exists():
        reference = pd.read_csv(REFERENCE_REPLAY_PATH)[["team_id", "alpha_015_xg_final_rating"]]
        merged = main_final[["team_id", "end_live_rating"]].merge(reference, on="team_id", validate="one_to_one")
        max_diff = float((merged["end_live_rating"] - merged["alpha_015_xg_final_rating"]).abs().max())
        add("main_matches_existing_final_candidate_replay", max_diff <= 1e-8, max_diff, "<=1e-8")
    add("comparison_rows", len(outputs["initial_and_final_ratings"]) == EXPECTED_TEAMS, len(outputs["initial_and_final_ratings"]), EXPECTED_TEAMS)
    return pd.DataFrame(checks)


def write_outputs(
    output_root: Path,
    outputs: dict[str, pd.DataFrame],
    audits: pd.DataFrame,
) -> None:
    for name, frame in outputs.items():
        frame.to_csv(output_root / f"{name}.csv", index=False, float_format="%.9f")
    audits.to_csv(output_root / "identity_and_invariant_audit.csv", index=False)


def build_output_manifest(
    args: argparse.Namespace,
    inputs: dict[str, object],
    results: dict[str, ArmReplayResult],
    outputs: dict[str, pd.DataFrame],
    audits: pd.DataFrame,
) -> dict[str, object]:
    runtime = inputs["runtime"]
    assert isinstance(runtime, FinalCandidateRuntime)
    decision = inputs["surprise_decision"]
    assert isinstance(decision, dict)
    return {
        "analysis": "AO_2025_26_FINAL_AND_SHADOW_REPLAY",
        "season": SEASON,
        "evidence_class": "RETROSPECTIVE_COUNTERFACTUAL",
        "main_model_changed": False,
        "shadow_features_affect_main": False,
        "matches_per_arm": EXPECTED_MATCHES,
        "teams": EXPECTED_TEAMS,
        "xg_eligible_matches": EXPECTED_XG_ELIGIBLE,
        "identity_contract": "club_id=AO-UEFA-<UEFA team ID>",
        "identity_checks_passed": int(audits["passed"].sum()),
        "identity_checks_total": len(audits),
        "final_candidate": {
            "candidate_version": runtime.candidate_version,
            "contract_sha256": runtime.contract_sha256,
            "elo_scale": runtime.dynamic_config.elo_scale,
            "home_advantage": runtime.dynamic_config.home_advantage,
            "k_factor": runtime.dynamic_config.k_factor,
            "draw_at_even": runtime.dynamic_config.draw_at_even,
            "draw_shape": runtime.dynamic_config.draw_shape,
            "goal_alpha": runtime.dynamic_config.goal_alpha,
            "goal_tau": runtime.dynamic_config.goal_tau,
            "goal_difference_cap": runtime.dynamic_config.goal_difference_cap,
            "xg_ratio": runtime.xg_config.beta,
            "xg_scale": runtime.xg_config.xg_scale,
            "minimum_winner_gain_ratio": runtime.xg_config.minimum_winner_gain_ratio,
            "missing_xg_behavior": "GOAL_MARGIN_ONLY",
        },
        "domestic_surprise_shadow": {
            "decision": decision["decision"],
            "production_change": decision["production_change"],
            **surprise_manifest_parameters(decision),
        },
        "progression_shadow": {
            "base_bonus": 12.0,
            "ucl_increment": 12.0,
            "uel_increment": 8.0,
            "uecl_increment": 4.0,
            "ucl_cap": 60.0,
            "uel_cap": 40.0,
            "uecl_cap": 20.0,
            "winner_only": True,
            "production_change": False,
        },
        "arms": [arm.__dict__ for arm in replay_arms()],
        "arm_audits": {key: value.audit for key, value in results.items()},
        "output_rows": {key: len(frame) for key, frame in outputs.items()},
        "inputs": {
            "matches_sha256": sha256_file(args.data.resolve()),
            "xg_manifest_sha256": sha256_file(args.xg_manifest.resolve()),
            "identity_sha256": sha256_file(args.identity.resolve()),
            "surprise_adjustments_sha256": sha256_file(args.surprise.resolve()),
            "static_manifest_sha256": sha256_file(args.static_manifest.resolve()),
        },
        "interpretation": (
            "The main arm is the current final model candidate. Shadow arms are "
            "counterfactual diagnostics and do not change the active model."
        ),
    }


def surprise_manifest_parameters(decision: dict[str, object]) -> dict[str, object]:
    selected = decision.get("selected_full_candidate")
    if isinstance(selected, dict):
        return selected
    fixed = decision.get("fixed_parameters")
    if not isinstance(fixed, dict) or "selected_gamma" not in decision:
        raise ValueError("Domestic Surprise decision has no replayable parameters")
    return {
        "coefficient": fixed["coefficient"],
        "variance_penalty": decision["selected_gamma"],
        "max_abs_adjustment": fixed["max_abs_adjustment"],
        "minimum_history_seasons": fixed["minimum_history_seasons"],
        "volatility_normalization": fixed["volatility_normalization"],
    }


def build_markdown_report(
    outputs: dict[str, pd.DataFrame], manifest: dict[str, object]
) -> str:
    comparison = outputs["initial_and_final_ratings"]
    models = outputs["model_comparison"]
    shadows = outputs["shadow_effect_summary"]
    bonus = outputs["bonus_events"]
    main = models.loc[models["model_arm"].eq(MAIN_ARM)].iloc[0]
    top_final = comparison.nsmallest(15, "main_end_rank")
    risers = comparison.nlargest(10, "main_rank_gain")
    fallers = comparison.nsmallest(10, "main_rank_gain")
    lines = [
        "# AO European Elo 2025/26 Final Model ve Shadow Replay",
        "",
        "## Teknik sonuç",
        "",
        f"- {EXPECTED_MATCHES} maç ve {EXPECTED_TEAMS} kalıcı kulüp kimliği işlendi.",
        f"- {EXPECTED_XG_ELIGIBLE} maçta doğrulanmış xG kullanıldı; diğer maçlar gol farkı çekirdeğine döndü.",
        f"- Ana model pooled Brier `{main['brier_1x2']:.6f}`, log-loss `{main['log_loss_1x2']:.6f}` üretti.",
        "- Domestic Surprise ve progression yalnız shadow kollarında çalıştı; ana modele veri veya rating geri beslemesi olmadı.",
        "- Bu çalışma geçmiş sezonun counterfactual replay'idir; shadow terfisi için bağımsız prospective kanıt değildir.",
        "",
        "## Kimlik sözleşmesi",
        "",
        f"`{manifest['identity_contract']}` kullanıldı. 236 yerel takım kimliği, 236 UEFA kimliği ve 236 kalıcı club_id bire bir doğrulandı.",
        "",
        "## Model kolları",
        "",
        markdown_table(models[["model_arm", "brier_1x2", "log_loss_1x2", "accuracy_1x2", "total_bonus_added", "final_rating_std"]]),
        "",
        "## Shadow etkisi",
        "",
        markdown_table(shadows),
        "",
        "## Ana model sezon sonu ilk 15",
        "",
        markdown_table(top_final[["main_end_rank", "team_name", "participating_competitions", "main_start_rating", "main_end_live_rating", "main_live_change", "main_rank_gain"]]),
        "",
        "## Ana modelde en çok yükselenler",
        "",
        markdown_table(risers[["team_name", "main_start_rank", "main_end_rank", "main_live_change", "main_rank_gain"]]),
        "",
        "## Ana modelde en çok düşenler",
        "",
        markdown_table(fallers[["team_name", "main_start_rank", "main_end_rank", "main_live_change", "main_rank_gain"]]),
        "",
        "## Progression olayları",
        "",
        f"Her progression kolunda {len(bonus) // 2 if len(bonus) else 0} uygun eşleşme olayı işlendi. Birleşik dosyada progression ve combined kolları ayrı satırlardır.",
        "",
        "## Yorum sınırı",
        "",
        "Aynı sezonun sonuçları daha önce model geliştirme sürecinde görüldüğü için bu replay bağımsız holdout değildir. Ana model sonucu davranış ve denetim raporu; shadow sonuçları ise etki büyüklüğü ve yön diagnostiğidir.",
    ]
    return "\n".join(lines) + "\n"


def competition_participation(events: pd.DataFrame) -> pd.DataFrame:
    home = events[["home_club_id", "competition"]].rename(columns={"home_club_id": "club_id"})
    away = events[["away_club_id", "competition"]].rename(columns={"away_club_id": "club_id"})
    values = pd.concat([home, away], ignore_index=True).drop_duplicates()
    return (
        values.groupby("club_id", as_index=False)["competition"]
        .agg(lambda items: "/".join(sorted(set(items))))
        .rename(columns={"competition": "participating_competitions"})
    )


def stable_rank(frame: pd.DataFrame, rating_column: str, *, tie_column: str) -> pd.Series:
    ordered = frame[[rating_column, tie_column]].sort_values(
        [rating_column, tie_column], ascending=[False, True], kind="stable"
    )
    ranks = pd.Series(np.arange(1, len(ordered) + 1), index=ordered.index)
    return ranks.reindex(frame.index).astype(int)


def coerce_bool(values: pd.Series, name: str) -> pd.Series:
    mapping = {
        True: True,
        False: False,
        1: True,
        0: False,
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }
    normalized = values.map(
        lambda value: value.strip().lower() if isinstance(value, str) else value
    )
    result = normalized.map(mapping)
    if result.isna().any():
        raise ValueError(f"{name} contains invalid boolean values")
    return result.astype(bool)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def markdown_table(frame: pd.DataFrame) -> str:
    values = frame.copy()
    for column in values.select_dtypes(include=["float"]).columns:
        values[column] = values[column].map(lambda value: f"{value:.6f}")
    headers = [str(column) for column in values.columns]
    rows = [[str(value) for value in row] for row in values.itertuples(index=False, name=None)]
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(row) + " |" for row in rows),
        ]
    )


if __name__ == "__main__":
    main()
