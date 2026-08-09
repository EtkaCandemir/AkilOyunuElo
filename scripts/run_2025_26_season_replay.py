from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import asdict
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
from ao_elo.dynamic import normalize_stage  # noqa: E402
from ao_elo.evaluation import (  # noqa: E402
    dependency_robust_loss_difference_ci,
    schedule_adjusted_team_performance,
)
from ao_elo.match_context import DomesticRegressionConfig  # noqa: E402
from ao_elo.scoreline import (  # noqa: E402
    ScorelineModelConfig,
    exact_score_probability,
    predict_scoreline,
    scoreline_matrix,
    scoreline_to_1x2,
)
from ao_elo.season_replay import (  # noqa: E402
    ReplayArmSpec,
    classify_shadow_signal,
    prediction_metrics,
    same_season_ranking,
    stable_config_fingerprint,
)
from ao_elo.tournament_bonus import FixedTournamentBonusConfig  # noqa: E402
from scripts.run_controlled_goal_progression_backtest import (  # noqa: E402
    prepare_controlled_data,
)
from scripts.run_dynamic_core_calibration import DynamicCoreConfig  # noqa: E402
from scripts.run_fixed_tournament_bonus_backtest import (  # noqa: E402
    evaluate_sequence as evaluate_bonus_sequence,
)
from scripts.run_match_context_backtest import (  # noqa: E402
    ContextModelConfig,
    evaluate_context_sequence,
    load_context_data,
)
from scripts.run_v2_achievement_reserve_calibration import load_reserve_data  # noqa: E402
from scripts.run_v2_evaluation_upgrade import DrawModelConfig, read_events  # noqa: E402


SEASON = "2025/26"
STATIC_ROOT = ROOT / "data" / "backtest_stage_b_2018_2026"
EVENTS_PATH = ROOT / "data" / "external_elo_benchmark_2018_2026" / "exact_date_events.csv"
DYNAMIC_ROOT = ROOT / "output" / "v2_dynamic_calibration_2018_2026"
EVALUATION_ROOT = ROOT / "output" / "v2_evaluation_upgrade_2018_2026"
PRODUCTION_CONTRACT = ROOT / "contracts" / "ao_european_elo_v2_production.json"
OUTPUT_ROOT = ROOT / "output" / "season_replay_2025_26"
XG_CONVEX_PATH = ROOT / "output" / "xg_goal_ablation_backtest_2018_2026" / "matched_predictions.csv"
XG_LUCK_PATH = ROOT / "output" / "xg_luck_correction_backtest_2018_2026" / "holdout_predictions.csv"
PROGRESSION_PATH = ROOT / "output" / "progression_probability_2018_2026" / "unseen_tie_predictions.csv"
SCORELINE_MANIFEST = ROOT / "output" / "scoreline_backtest_2018_2026" / "selected_scoreline_model.json"
LEVEL_MANIFEST = ROOT / "output" / "scoreline_level_calibration_2018_2026" / "selected_level_model.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay all 2025/26 AO European Elo matches and shadow arms")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples must be at least 100")

    static_manifest = json.loads((DYNAMIC_ROOT / "selected_dynamic_model.json").read_text(encoding="utf-8"))
    static_config = AOEuropeanEloConfig(**static_manifest["static_config"])
    static_config.validate()
    production = json.loads(PRODUCTION_CONTRACT.read_text(encoding="utf-8"))
    validate_production(production)
    all_events = read_events(EVENTS_PATH)
    events = all_events.loc[all_events["season"].astype(str).eq(SEASON)].copy()
    events = events.sort_values(["kickoff_utc", "match_id"], kind="stable").reset_index(drop=True)
    events["stage"] = [
        normalize_stage(str(row.round), str(row.competition), bool(row.is_knockout))
        for row in events.itertuples(index=False)
    ]
    validate_events(events)
    reserve_data, tie_audit = load_reserve_data(STATIC_ROOT, EVENTS_PATH, static_config)
    controlled = prepare_controlled_data(reserve_data, all_events)
    season_data = next(data for data in controlled if data.season == SEASON)
    target = schedule_adjusted_team_performance(all_events)
    core = read_fold6_core()
    historical_draw = read_fold6_draw()
    current_draw = {key: DrawModelConfig(0.24, 1.0) for key in ("ALL", "UCL", "UEL", "UECL")}
    names = team_names(events)

    specs = (
        ReplayArmSpec("HISTORICAL_LOCKED", "OOS_FOLD6", 0.0, 300.0),
        ReplayArmSpec("HISTORICAL_BONUS_6_4_2", "OOS_FOLD6", 0.0, 300.0, 6.0, reference_arm="HISTORICAL_LOCKED"),
        ReplayArmSpec("PRODUCTION", "RETROSPECTIVE_COUNTERFACTUAL", 0.10, 300.0),
        ReplayArmSpec("NO_GD_CONTROL", "RETROSPECTIVE_COUNTERFACTUAL", 0.0, 300.0, reference_arm="PRODUCTION"),
        ReplayArmSpec("GD_PRIOR_GRID", "RETROSPECTIVE_COUNTERFACTUAL", 0.20, 400.0, reference_arm="PRODUCTION"),
        ReplayArmSpec("GD_EXTENDED", "RETROSPECTIVE_COUNTERFACTUAL", 0.125, 800.0, reference_arm="PRODUCTION"),
        ReplayArmSpec("FIXED_TOURNAMENT_BONUS", "RETROSPECTIVE_COUNTERFACTUAL", 0.10, 300.0, 12.0, reference_arm="PRODUCTION"),
    )
    predictions, finals, trajectories, bonus_events, season_metrics = run_rating_arms(
        season_data, target, core, historical_draw, current_draw, specs, names
    )
    domestic_predictions, domestic_final, domestic_audit = run_domestic_arm(
        static_config, all_events, target, core, events, names
    )
    predictions = pd.concat([predictions, domestic_predictions], ignore_index=True)
    finals = pd.concat([finals, domestic_final], ignore_index=True)
    season_metrics = pd.concat([season_metrics, domestic_audit], ignore_index=True)

    rankings = build_rankings(finals, target)
    uncertainty = build_uncertainty(predictions, specs, args.bootstrap_samples)
    segments = build_segments(predictions)
    comparison = build_model_comparison(predictions, rankings, segments, uncertainty)
    scoreline = build_scoreline_outputs(predictions, events, production)
    progression = build_progression_summary()
    xg_appendix = build_xg_appendix()
    initial = build_initial_ratings(season_data, names, static_config)
    validate_replay_outputs(
        initial, predictions, finals, scoreline, progression, xg_appendix, season_metrics
    )

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    initial.to_csv(output_root / "initial_ratings.csv", index=False)
    predictions.to_csv(output_root / "match_predictions_and_updates.csv", index=False)
    comparison.to_csv(output_root / "model_comparison.csv", index=False)
    segments.to_csv(output_root / "competition_stage_month_summary.csv", index=False)
    trajectories.to_csv(output_root / "team_rating_trajectories.csv", index=False)
    finals.to_csv(output_root / "final_ratings.csv", index=False)
    bonus_events.to_csv(output_root / "bonus_events.csv", index=False)
    scoreline.to_csv(output_root / "scoreline_predictions.csv", index=False)
    progression.to_csv(output_root / "progression_probability_summary.csv", index=False)
    xg_appendix.to_csv(output_root / "xg_appendix.csv", index=False)
    uncertainty.to_csv(output_root / "dependency_uncertainty.csv", index=False)
    rankings.to_csv(output_root / "same_season_ranking.csv", index=False)
    season_metrics.to_csv(output_root / "season_state_audit.csv", index=False)
    tie_audit.to_csv(output_root / "tie_chronology_audit.csv", index=False)

    manifest = build_manifest(initial, predictions, scoreline, progression, xg_appendix, specs, production)
    (output_root / "replay_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(output_root / "replay_report.md", comparison, rankings, scoreline, progression, xg_appendix, manifest)
    pdf_path = build_pdf_report(output_root)
    print("AO 2025/26 season replay complete")
    print(f"Full-scope matches per arm: {len(events)}")
    print(f"Output: {output_root}")
    print(f"PDF: {pdf_path}")


def validate_production(production: dict[str, object]) -> None:
    goal = production["goal_margin"]
    expected = (True, 0.1, 300.0, 4)
    actual = (bool(goal["active"]), float(goal["alpha"]), float(goal["tau"]), int(goal["goal_difference_cap"]))
    if actual != expected or float(production["active_power_carry"]) != 0.0:
        raise ValueError(f"Unexpected production contract: {actual}")


def validate_events(events: pd.DataFrame) -> None:
    if len(events) != 961 or events["match_id"].duplicated().any():
        raise ValueError("2025/26 replay requires exactly 961 unique matches")
    if set(events["competition"]) != {"UCL", "UEL", "UECL"}:
        raise ValueError("Competition coverage is incomplete")
    kickoff = pd.to_datetime(events["kickoff_utc"], utc=True, errors="raise")
    if not kickoff.is_monotonic_increasing:
        raise ValueError("Events must be in exact UTC chronology")


def read_fold6_core() -> DynamicCoreConfig:
    row = pd.read_csv(DYNAMIC_ROOT / "core_fold_selections.csv").loc[lambda frame: frame["fold"].eq(6)].iloc[0]
    config = DynamicCoreConfig(float(row["selected_scale"]), float(row["selected_home_advantage"]), float(row["selected_k"]))
    config.validate()
    return config


def read_fold6_draw() -> dict[str, DrawModelConfig]:
    rows = pd.read_csv(EVALUATION_ROOT / "draw_production_fold_selections.csv")
    rows = rows.loc[rows["fold"].eq(6)]
    result = {
        str(row.competition): DrawModelConfig(float(row.draw_at_even), float(row.draw_shape))
        for row in rows.itertuples(index=False)
    }
    if set(result) != {"ALL", "UCL", "UEL", "UECL"}:
        raise ValueError("Fold-6 draw map is incomplete")
    return result


def team_names(events: pd.DataFrame) -> dict[int, str]:
    home = events[["home_team_id", "home_team_name"]].rename(columns={"home_team_id": "team_id", "home_team_name": "team_name"})
    away = events[["away_team_id", "away_team_name"]].rename(columns={"away_team_id": "team_id", "away_team_name": "team_name"})
    values = pd.concat([home, away]).drop_duplicates()
    if values["team_id"].duplicated().any() or len(values) != 236:
        raise ValueError("Expected 236 stable team identities")
    return dict(zip(values["team_id"].astype(int), values["team_name"].astype(str)))


def run_rating_arms(season_data, target, core, historical_draw, current_draw, specs, names):
    prediction_frames = []
    final_frames = []
    trajectory_frames = []
    bonus_frames = []
    season_frames = []
    for spec in specs:
        spec.validate()
        draw = historical_draw if spec.evidence_class == "OOS_FOLD6" else current_draw
        evaluated = evaluate_bonus_sequence(
            (season_data,), core, draw, target,
            FixedTournamentBonusConfig(spec.tournament_bonus_base),
            alpha=spec.alpha, tau=spec.tau, goal_cap=4,
            evaluation_seasons={SEASON}, ranking_target_seasons={SEASON}, return_details=True,
        )
        predictions = add_arm_columns(evaluated.predictions, spec)
        predictions["draw_at_even"] = predictions["competition"].map(
            lambda competition: draw.get(str(competition), draw["ALL"]).draw_at_even
        )
        predictions["draw_shape"] = predictions["competition"].map(
            lambda competition: draw.get(str(competition), draw["ALL"]).draw_shape
        )
        if len(predictions) != 961 or not predictions["match_id"].is_unique:
            raise ValueError(f"{spec.model_arm}: replay coverage failed")
        prediction_frames.append(predictions)
        final = evaluated.end_ratings.copy()
        final["team_name"] = final["team_id"].map(names)
        final = add_arm_columns(final, spec)
        final_frames.append(final)
        trajectory_frames.append(build_trajectories(predictions, names))
        if not evaluated.bonus_events.empty:
            bonus_frames.append(add_arm_columns(evaluated.bonus_events.copy(), spec))
        season_frames.append(add_arm_columns(evaluated.season_metrics.copy(), spec))
    return (
        pd.concat(prediction_frames, ignore_index=True),
        pd.concat(final_frames, ignore_index=True),
        pd.concat(trajectory_frames, ignore_index=True),
        pd.concat(bonus_frames, ignore_index=True) if bonus_frames else pd.DataFrame(),
        pd.concat(season_frames, ignore_index=True),
    )


def add_arm_columns(frame: pd.DataFrame, spec: ReplayArmSpec) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, "config_fingerprint", spec.config_fingerprint)
    result.insert(0, "evidence_class", spec.evidence_class)
    result.insert(0, "model_arm", spec.model_arm)
    result["reference_arm"] = spec.reference_arm
    result["goal_alpha"] = spec.alpha
    result["goal_tau"] = spec.tau
    result["tournament_bonus_base"] = spec.tournament_bonus_base
    result["domestic_persistence"] = spec.domestic_persistence
    result["elo_scale"] = 835.561497
    result["home_advantage"] = 148.544266
    result["k_factor"] = 103.980986
    result["goal_difference_cap"] = 4
    result["active_power_carry"] = 0.0
    return result


def build_trajectories(predictions: pd.DataFrame, names: dict[int, str]) -> pd.DataFrame:
    rows = []
    for row in predictions.itertuples(index=False):
        bonus_home = float(row.bonus_applied_after_match) if int(row.bonus_winner_id) == int(row.home_team_id) else 0.0
        bonus_away = float(row.bonus_applied_after_match) if int(row.bonus_winner_id) == int(row.away_team_id) else 0.0
        for venue, team_id, pre, delta, added in (
            ("HOME", int(row.home_team_id), float(row.home_live_pre), float(row.power_delta), bonus_home),
            ("AWAY", int(row.away_team_id), float(row.away_live_pre), -float(row.power_delta), bonus_away),
        ):
            rows.append({
                "model_arm": row.model_arm, "evidence_class": row.evidence_class,
                "config_fingerprint": row.config_fingerprint, "match_id": row.match_id,
                "kickoff_utc": row.kickoff_utc, "competition": row.competition,
                "team_id": team_id, "team_name": names[team_id], "venue": venue,
                "live_rating_pre": pre, "power_delta": delta,
                "bonus_added": added, "live_rating_post": pre + delta + added,
            })
    return pd.DataFrame(rows)


def run_domestic_arm(static_config, all_events, target, core, events, names):
    datasets, _ = load_context_data(STATIC_ROOT, EVENTS_PATH, static_config)
    config = ContextModelConfig(domestic=DomesticRegressionConfig("DOMESTIC_ANCHORED", 0.75))
    evaluated = evaluate_context_sequence(
        datasets, core, config, target,
        evaluation_seasons={SEASON}, ranking_target_seasons={SEASON}, return_predictions=True,
    )
    spec = ReplayArmSpec(
        "DOMESTIC_ANCHORED", "RETROSPECTIVE_COUNTERFACTUAL", 0.10, 300.0,
        domestic_persistence=0.75, reference_arm="PRODUCTION",
    )
    metadata = events[["match_id", "kickoff_utc", "round", "tie_id", "home_goals", "away_goals", "decided_on_penalties", "is_neutral", "competition", "is_knockout"]].copy()
    metadata["stage"] = [
        normalize_stage(str(row.round), str(row.competition), bool(row.is_knockout))
        for row in metadata.itertuples(index=False)
    ]
    metadata = metadata.drop(columns=["competition", "is_knockout"])
    predictions = evaluated.predictions.merge(metadata, on="match_id", validate="one_to_one")
    predictions["actual_class"] = np.where(predictions["actual_home_score"].eq(1.0), 0, np.where(predictions["actual_home_score"].eq(0.5), 1, 2))
    predictions["predicted_class"] = predictions[["home_probability", "draw_probability", "away_probability"]].to_numpy().argmax(axis=1)
    predictions["home_live_pre"] = np.nan
    predictions["away_live_pre"] = np.nan
    predictions["bonus_applied_after_match"] = 0.0
    predictions["bonus_winner_id"] = -1
    predictions = add_arm_columns(predictions, spec)
    predictions["draw_at_even"] = 0.24
    predictions["draw_shape"] = 1.0
    final = evaluated.end_ratings.loc[evaluated.end_ratings["season"].eq(SEASON)].copy()
    final["team_name"] = final["team_id"].map(names)
    final = add_arm_columns(final, spec)
    audit = pd.DataFrame([{
        "model_arm": spec.model_arm,
        "evidence_class": spec.evidence_class,
        "config_fingerprint": spec.config_fingerprint,
        "season": SEASON,
        "matches": int(len(predictions)),
        "max_pair_sum_error": float(evaluated.metrics["max_pair_sum_error"]),
        "power_total_error": np.nan,
        "bonus_total": 0.0,
        "max_bonus_cap_excess": 0.0,
        "season_reset_error": 0.0,
    }])
    return predictions, final, audit


def build_initial_ratings(season_data, names, static_config):
    core = season_data.reserve.goal.carry.core
    fingerprint = stable_config_fingerprint(asdict(static_config))
    rows = [
        {
            "model_arm": "AO_FIRST_ELO_V2_SNAPSHOT",
            "evidence_class": "OOS_FOLD6",
            "config_fingerprint": fingerprint,
            "season": SEASON,
            "team_id": int(team_id),
            "team_name": names[int(team_id)],
            "ao_first_elo": float(core.initial_ratings[team_id]),
        }
        for team_id in core.active_team_ids
    ]
    result = pd.DataFrame(rows).sort_values("ao_first_elo", ascending=False).reset_index(drop=True)
    result.insert(0, "initial_rank", np.arange(1, len(result) + 1))
    if len(result) != 236 or result["team_id"].duplicated().any():
        raise ValueError("Initial rating snapshot failed validation")
    return result


def build_rankings(finals, target):
    frames = []
    for arm, frame in finals.groupby("model_arm", sort=False):
        ranking = same_season_ranking(frame, target, season=SEASON)
        ranking.insert(0, "model_arm", arm)
        ranking.insert(1, "evidence_class", frame["evidence_class"].iloc[0])
        ranking.insert(2, "config_fingerprint", frame["config_fingerprint"].iloc[0])
        frames.append(ranking)
    return pd.concat(frames, ignore_index=True)


def build_uncertainty(predictions, specs, bootstrap_samples):
    frames = []
    references = {spec.model_arm: spec.reference_arm for spec in specs if spec.reference_arm}
    references["DOMESTIC_ANCHORED"] = "PRODUCTION"
    for arm, reference in references.items():
        for metric in ("brier_1x2", "log_loss_1x2"):
            base = predictions.loc[predictions["model_arm"].eq(reference), ["match_id", "season", "kickoff_utc", "tie_id", "home_team_id", "away_team_id", metric]].rename(columns={metric: "base_loss"})
            candidate = predictions.loc[predictions["model_arm"].eq(arm), ["match_id", metric]].rename(columns={metric: "candidate_loss"})
            paired = base.merge(candidate, on="match_id", validate="one_to_one")
            paired["loss_difference"] = paired["candidate_loss"] - paired["base_loss"]
            result = dependency_robust_loss_difference_ci(
                paired, bootstrap_samples=bootstrap_samples,
                seed=20260803 + sum(ord(char) for char in arm + metric),
            )
            result.insert(0, "model_arm", arm)
            result.insert(1, "reference_arm", reference)
            result.insert(2, "metric", metric)
            result.insert(3, "evidence_class", predictions.loc[predictions["model_arm"].eq(arm), "evidence_class"].iloc[0])
            result.insert(4, "config_fingerprint", predictions.loc[predictions["model_arm"].eq(arm), "config_fingerprint"].iloc[0])
            frames.append(result)
    return pd.concat(frames, ignore_index=True)


def build_segments(predictions):
    values = predictions.copy()
    values["month"] = pd.to_datetime(values["kickoff_utc"], utc=True).dt.strftime("%Y-%m")
    production_expected = values.loc[values["model_arm"].eq("PRODUCTION"), ["match_id", "expected_home_score"]].rename(columns={"expected_home_score": "production_expected"})
    values = values.merge(production_expected, on="match_id", how="left", validate="many_to_one")
    values["match_band"] = np.select(
        [values["production_expected"].ge(0.60), values["production_expected"].le(0.40)],
        ["HOME_FAVORITE", "HOME_UNDERDOG"], default="BALANCED",
    )
    rows = []
    for segment_type, column in (("competition", "competition"), ("stage", "stage"), ("month", "month"), ("match_band", "match_band")):
        for (arm, segment), frame in values.groupby(["model_arm", column], sort=True, dropna=False):
            rows.append({
                "model_arm": arm,
                "evidence_class": frame["evidence_class"].iloc[0],
                "config_fingerprint": frame["config_fingerprint"].iloc[0],
                "segment_type": segment_type,
                "segment_value": str(segment),
                **prediction_metrics(frame),
            })
    return pd.DataFrame(rows)


def build_model_comparison(predictions, rankings, segments, uncertainty):
    references = {
        "HISTORICAL_LOCKED": "",
        "HISTORICAL_BONUS_6_4_2": "HISTORICAL_LOCKED",
        "PRODUCTION": "",
        "NO_GD_CONTROL": "PRODUCTION",
        "GD_PRIOR_GRID": "PRODUCTION",
        "GD_EXTENDED": "PRODUCTION",
        "FIXED_TOURNAMENT_BONUS": "PRODUCTION",
        "DOMESTIC_ANCHORED": "PRODUCTION",
    }
    metrics = {arm: prediction_metrics(frame) for arm, frame in predictions.groupby("model_arm", sort=False)}
    rows = []
    for arm, reference in references.items():
        frame = predictions.loc[predictions["model_arm"].eq(arm)]
        row = {
            "model_arm": arm,
            "evidence_class": frame["evidence_class"].iloc[0],
            "config_fingerprint": frame["config_fingerprint"].iloc[0],
            "reference_arm": reference,
            **metrics[arm],
        }
        rank = rankings.loc[(rankings["model_arm"].eq(arm)) & (rankings["competition"].eq("ALL"))]
        row["same_season_spearman"] = float(rank["ranking_score"].iloc[0])
        row["same_season_pairwise"] = float(rank["pairwise_accuracy"].iloc[0])
        if not reference:
            row.update({"brier_delta_vs_reference": 0.0, "log_loss_delta_vs_reference": 0.0, "classification": "REFERENCE"})
        else:
            brier_delta = row["brier_1x2"] - metrics[reference]["brier_1x2"]
            log_delta = row["log_loss_1x2"] - metrics[reference]["log_loss_1x2"]
            candidate_segments = segments.loc[(segments["model_arm"].eq(arm)) & (segments["segment_type"].eq("competition"))].set_index("segment_value")
            base_segments = segments.loc[(segments["model_arm"].eq(reference)) & (segments["segment_type"].eq("competition"))].set_index("segment_value")
            competition_no_harm = bool(
                ((candidate_segments["brier_1x2"] - base_segments["brier_1x2"]) <= 0.0).all()
                and ((candidate_segments["log_loss_1x2"] - base_segments["log_loss_1x2"]) <= 0.0).all()
            )
            envelope = uncertainty.loc[(uncertainty["model_arm"].eq(arm)) & (uncertainty["method"].eq("conservative_envelope"))]
            reliable_harm = bool((envelope["ci_95_lower"] > 0.0).any())
            row.update({
                "brier_delta_vs_reference": brier_delta,
                "log_loss_delta_vs_reference": log_delta,
                "competition_no_harm": competition_no_harm,
                "reliable_harm": reliable_harm,
                "classification": classify_shadow_signal(brier_delta, log_delta, competition_no_harm, reliable_harm),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def build_scoreline_outputs(predictions, events, production):
    selected = json.loads(SCORELINE_MANIFEST.read_text(encoding="utf-8"))
    level = json.loads(LEVEL_MANIFEST.read_text(encoding="utf-8"))
    current_config = ScorelineModelConfig(**selected["scoreline_model"])
    historical_config = ScorelineModelConfig(mu=0.29839042255184667, elo_slope=0.7107950091855719, rho=0.0)
    current_offsets = level["full_competition_log_offsets"]
    current_season_offset = float(level["full_season_log_offset"])
    definitions = (
        ("HISTORICAL_SCORELINE", "OOS_FOLD6", "HISTORICAL_LOCKED", historical_config, {key: 0.0 for key in ("UCL", "UEL", "UECL")}, 0.0),
        ("HISTORICAL_SCORELINE_LEVEL", "OOS_FOLD6", "HISTORICAL_LOCKED", historical_config, {key: 0.0 for key in ("UCL", "UEL", "UECL")}, 0.008981863056118367),
        ("SCORELINE_POISSON", "RETROSPECTIVE_COUNTERFACTUAL", "PRODUCTION", current_config, {key: 0.0 for key in ("UCL", "UEL", "UECL")}, 0.0),
        ("SCORELINE_LEVEL", "RETROSPECTIVE_COUNTERFACTUAL", "PRODUCTION", current_config, current_offsets, current_season_offset),
    )
    metadata = events.set_index("match_id")
    rows = []
    for arm, evidence, source_arm, config, offsets, season_offset in definitions:
        score_fingerprint = stable_config_fingerprint({
            "model_arm": arm,
            "rating_source_arm": source_arm,
            "scoreline_config": asdict(config),
            "competition_log_offsets": offsets,
            "season_log_offset": season_offset,
            "elo_scale": production["dynamic_core"]["elo_scale"],
            "home_advantage": production["dynamic_core"]["home_advantage"],
        })
        source = predictions.loc[predictions["model_arm"].eq(source_arm)].set_index("match_id")
        for match_id, event in metadata.iterrows():
            pre = source.loc[match_id]
            predicted = predict_scoreline(
                float(pre.home_live_pre), float(pre.away_live_pre), is_neutral=bool(event.is_neutral),
                elo_scale=float(production["dynamic_core"]["elo_scale"]),
                home_advantage=float(production["dynamic_core"]["home_advantage"]), config=config,
            )
            offset = float(offsets[str(event.competition)]) + float(season_offset)
            scale = math.exp(offset)
            lambda_home = predicted.lambda_home * scale
            lambda_away = predicted.lambda_away * scale
            matrix, covered = scoreline_matrix(lambda_home, lambda_away, config)
            home_p, draw_p, away_p = scoreline_to_1x2(matrix)
            most = np.unravel_index(int(np.argmax(matrix)), matrix.shape)
            home_goals, away_goals = int(event.home_goals), int(event.away_goals)
            exact = exact_score_probability(home_goals, away_goals, lambda_home, lambda_away, config.rho)
            total_index = np.add.outer(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
            actual_total = home_goals + away_goals
            over_actual = int(actual_total >= 3)
            btts_actual = int(home_goals > 0 and away_goals > 0)
            over_probability = float(matrix[total_index >= 3].sum())
            btts_probability = float(matrix[1:, 1:].sum())
            actual_class = 0 if home_goals > away_goals else (1 if home_goals == away_goals else 2)
            probabilities = np.array([home_p, draw_p, away_p], dtype=float)
            one_hot = np.eye(3, dtype=float)[actual_class]
            rows.append({
                "model_arm": arm, "evidence_class": evidence, "rating_source_arm": source_arm,
                "config_fingerprint": score_fingerprint,
                "match_id": match_id, "season": SEASON, "kickoff_utc": event.kickoff_utc,
                "competition": event.competition, "stage": event.stage, "tie_id": event.tie_id,
                "home_team_id": int(event.home_team_id), "away_team_id": int(event.away_team_id),
                "home_goals": home_goals, "away_goals": away_goals,
                "lambda_home": lambda_home, "lambda_away": lambda_away, "rho": config.rho,
                "goal_level_log_offset": offset, "home_probability": home_p,
                "draw_probability": draw_p, "away_probability": away_p,
                "expected_total_goals": lambda_home + lambda_away,
                "predicted_home_goals": int(most[0]), "predicted_away_goals": int(most[1]),
                "most_likely_score": f"{most[0]}-{most[1]}",
                "exact_score_correct": int((home_goals, away_goals) == (int(most[0]), int(most[1]))),
                "actual_class": actual_class,
                "brier_1x2": float(np.square(probabilities - one_hot).sum()),
                "log_loss_1x2": -math.log(max(float(probabilities[actual_class]), 1e-15)),
                "over_2_5_actual": over_actual,
                "over_2_5_probability": over_probability,
                "over_2_5_brier": float((over_probability - over_actual) ** 2),
                "over_2_5_log_loss": binary_log_loss(over_actual, over_probability),
                "btts_actual": btts_actual,
                "btts_probability": btts_probability,
                "btts_brier": float((btts_probability - btts_actual) ** 2),
                "btts_log_loss": binary_log_loss(btts_actual, btts_probability),
                "covered_probability_mass": covered,
                "normalized_probability_mass": float(matrix.sum()),
                "exact_score_probability": exact, "score_nll": -math.log(max(exact, 1e-15)),
                "total_goal_absolute_error": abs((lambda_home + lambda_away) - (home_goals + away_goals)),
            })
    result = pd.DataFrame(rows)
    if result.groupby("model_arm")["match_id"].nunique().ne(961).any():
        raise ValueError("Scoreline sidecar coverage failed")
    return result


def build_progression_summary():
    values = pd.read_csv(PROGRESSION_PATH)
    values = values.loc[values["season"].astype(str).eq(SEASON)].copy()
    selected = values.loc[values["model"].eq("CALIBRATED")].copy()
    identity = values.loc[values["model"].eq("IDENTITY")].copy()
    if selected.empty or identity.empty:
        raise ValueError("Progression probability fold-6 rows are missing")
    selected["model_arm"] = "FORMAT_P_ADVANCE"
    selected["evidence_class"] = "OOS_FOLD6"
    selected["config_fingerprint"] = stable_config_fingerprint({
        "model_arm": "FORMAT_P_ADVANCE", "slope": 1.5,
        "single_leg_home_bias": 0.1, "two_leg_bias": 0.0,
    })
    selected["identity_probability"] = identity.set_index("tie_id").loc[selected["tie_id"], "probability"].to_numpy()
    selected["identity_brier_loss"] = identity.set_index("tie_id").loc[selected["tie_id"], "brier_loss"].to_numpy()
    selected["identity_log_loss"] = identity.set_index("tie_id").loc[selected["tie_id"], "log_loss"].to_numpy()
    return selected.reset_index(drop=True)


def build_xg_appendix():
    convex = pd.read_csv(XG_CONVEX_PATH)
    convex = convex.loc[convex["evaluation_split"].eq("HOLDOUT") & convex["model"].isin(["GD", "GD_XG"])].copy()
    convex["model_arm"] = np.where(
        convex["model"].eq("GD"), "XG_GD_REFERENCE_CONVEX", "GD_XG_CONVEX"
    )
    luck = pd.read_csv(XG_LUCK_PATH).copy()
    luck["model_arm"] = np.where(
        luck["comparison_model"].eq("GD_BASELINE"), "XG_GD_REFERENCE_LUCK", "GD_XG_LUCK"
    )
    common = sorted(set(convex.columns) & set(luck.columns) | {"model_arm"})
    result = pd.concat([convex.reindex(columns=common), luck.reindex(columns=common)], ignore_index=True)
    result["evidence_class"] = "MATCHED_XG_APPENDIX"
    result["config_fingerprint"] = result["model_arm"].map(lambda arm: stable_config_fingerprint({
        "model_arm": arm,
        "rho": 0.05 if "CONVEX" in arm else 0.50 if "LUCK" in arm else 0.0,
        "xg_scale": 1.00 if "CONVEX" in arm else 0.75 if "LUCK" in arm else 0.0,
    }))
    counts = result.groupby("model_arm")["match_id"].nunique()
    if not {"GD_XG_CONVEX", "GD_XG_LUCK"}.issubset(counts.index) or counts.loc[["GD_XG_CONVEX", "GD_XG_LUCK"]].ne(180).any():
        raise ValueError(f"xG appendix must contain 180 matched rows per candidate: {counts.to_dict()}")
    if result["season"].astype(str).ne(SEASON).any():
        raise ValueError("xG appendix contains rows outside 2025/26")
    return result


def binary_log_loss(actual: int, probability: float) -> float:
    if actual not in (0, 1) or not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("Binary outcome/probability is invalid")
    selected = probability if actual == 1 else 1.0 - probability
    return -math.log(max(selected, 1e-15))


def validate_replay_outputs(initial, predictions, finals, scoreline, progression, xg, season_metrics):
    if len(initial) != 236 or initial["team_id"].duplicated().any():
        raise ValueError("Initial rating contract failed")
    expected_arms = set(predictions["model_arm"].unique())
    coverage = predictions.groupby("model_arm")["match_id"].agg(["size", "nunique"])
    if coverage["size"].ne(961).any() or coverage["nunique"].ne(961).any():
        raise ValueError(f"Full-arm coverage failed: {coverage.to_dict('index')}")
    final_arms = set(finals["model_arm"].unique())
    if final_arms != expected_arms:
        raise ValueError("Final-rating arms do not match prediction arms")
    probabilities = predictions[["home_probability", "draw_probability", "away_probability"]].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or (probabilities < 0.0).any():
        raise ValueError("Replay probabilities must be finite and non-negative")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("Replay 1X2 probabilities must sum to one")
    if scoreline.groupby("model_arm")["match_id"].nunique().ne(961).any():
        raise ValueError("Scoreline arm coverage failed")
    if not np.allclose(scoreline["normalized_probability_mass"], 1.0, atol=1e-12):
        raise ValueError("Scoreline probability matrices must sum to one")
    if progression["tie_id"].nunique() != 284:
        raise ValueError("Fold-6 progression coverage must contain 284 ties")
    xg_counts = xg.groupby("model_arm")["match_id"].nunique()
    if xg_counts.ne(180).any():
        raise ValueError(f"Every xG appendix arm must use the common 180 matches: {xg_counts.to_dict()}")
    pair_error = max(
        season_metrics.get("max_pair_sum_error", pd.Series([0.0])).fillna(0.0).abs().max(),
        season_metrics.get("match_pair_zero_sum_error", pd.Series([0.0])).fillna(0.0).abs().max(),
    )
    if pair_error > 1e-9:
        raise ValueError("Power update zero-sum invariant failed")
    bonus_error = max(
        season_metrics.get("max_bonus_cap_excess", pd.Series([0.0])).fillna(0.0).max(),
        season_metrics.get("bonus_cap_error", pd.Series([0.0])).fillna(0.0).max(),
    )
    if bonus_error > 1e-9:
        raise ValueError("Tournament bonus cap invariant failed")
    if season_metrics["season_reset_error"].fillna(0.0).max() > 1e-9:
        raise ValueError("Season reset invariant failed")
    if predictions["season"].astype(str).ne(SEASON).any() or scoreline["season"].astype(str).ne(SEASON).any():
        raise ValueError("Replay contains data outside 2025/26")


def build_manifest(initial, predictions, scoreline, progression, xg, specs, production):
    return {
        "season": SEASON, "replay_type": "chronological_actual-result_replay",
        "full_scope_matches": 961, "teams": 236,
        "competition_matches": {str(key): int(value) for key, value in predictions.loc[predictions["model_arm"].eq("PRODUCTION")].groupby("competition").size().items()},
        "rating_arms": [asdict(spec) | {"config_fingerprint": spec.config_fingerprint} for spec in specs] + [{"model_arm": "DOMESTIC_ANCHORED", "evidence_class": "RETROSPECTIVE_COUNTERFACTUAL", "domestic_persistence": 0.75}],
        "scoreline_arms": sorted(scoreline["model_arm"].unique()),
        "progression_ties": int(progression["tie_id"].nunique()),
        "xg_matched_matches": 180,
        "initial_rating_min": float(initial["ao_first_elo"].min()),
        "initial_rating_max": float(initial["ao_first_elo"].max()),
        "production_revision": production["production_revision"],
        "production_changed": False, "shadow_status_changed": False,
        "prospective_holdout": "2026/27 league phase and later",
        "interpretation": "Historical OOS plus retrospective counterfactual; no promotion decision",
    }


def write_report(path, comparison, rankings, scoreline, progression, xg, manifest):
    score_summary = scoreline.groupby("model_arm", as_index=False).agg(
        matches=("match_id", "size"), score_nll=("score_nll", "mean"),
        exact_score_top1=("exact_score_correct", "mean"),
        total_goal_mae=("total_goal_absolute_error", "mean"),
        brier_1x2=("brier_1x2", "mean"), log_loss_1x2=("log_loss_1x2", "mean"),
        over_2_5_brier=("over_2_5_brier", "mean"),
        over_2_5_log_loss=("over_2_5_log_loss", "mean"),
        btts_brier=("btts_brier", "mean"), btts_log_loss=("btts_log_loss", "mean"),
    )
    progression_summary = progression.groupby("model_arm", as_index=False).agg(ties=("tie_id", "nunique"), brier=("brier_loss", "mean"), log_loss=("log_loss", "mean"), identity_brier=("identity_brier_loss", "mean"), identity_log_loss=("identity_log_loss", "mean"))
    xg_summary = xg.groupby("model_arm", as_index=False).agg(matches=("match_id", "nunique"), brier=("brier_1x2", "mean"), log_loss=("log_loss_1x2", "mean"))
    path.write_text(f"""# AO European Elo 2025/26 Tam Sezon Replay

## Kanıt Statüsü

Bu çalışma 961 gerçek sonucu kesin kronolojiyle replay eder. `OOS_FOLD6` kolları yalnız 2024/25 sonuna kadar seçilmiş parametreleri; `RETROSPECTIVE_COUNTERFACTUAL` kolları bugünkü modeli geçmiş sezona uygular. Sonuçlar production veya shadow statüsünü değiştirmez.

## Model Karşılaştırması

```csv
{comparison.to_csv(index=False).strip()}
```

## Aynı Sezon Sıralama Diagnostiği

Bu tablo forward ranking değildir.

```csv
{rankings.to_csv(index=False).strip()}
```

## Skor Katmanı

```csv
{score_summary.to_csv(index=False).strip()}
```

## Format Duyarlı Tur Olasılığı

```csv
{progression_summary.to_csv(index=False).strip()}
```

## xG Eki

xG karşılaştırması yalnız aynı 180 maçta yapılmıştır ve veri kaynağı production-grade değildir.

```csv
{xg_summary.to_csv(index=False).strip()}
```

## Sözleşme

```json
{json.dumps(manifest, indent=2, ensure_ascii=False)}
```
""", encoding="utf-8")


def build_pdf_report(output_root: Path) -> Path:
    bundled_python = Path(
        "/Users/buycell/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
    )
    command = [
        str(bundled_python if bundled_python.exists() else sys.executable),
        str(ROOT / "scripts" / "build_2025_26_replay_pdf.py"),
        "--output-root",
        str(output_root),
    ]
    subprocess.run(command, check=True)
    pdf_path = output_root / "AO_2025_26_Sezon_Replay_Raporu.pdf"
    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        raise ValueError("Replay PDF was not created")
    return pdf_path


if __name__ == "__main__":
    main()
