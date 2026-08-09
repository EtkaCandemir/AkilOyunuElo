from __future__ import annotations

import argparse
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

from ao_elo.evaluation import (  # noqa: E402
    dependency_robust_loss_difference_ci,
    schedule_adjusted_team_performance,
)
from ao_elo.robustness import one_x_two_probabilities_scalar  # noqa: E402
from ao_elo.xg_live import XGBlendConfig, update_match_elo_with_xg  # noqa: E402
from scripts.run_controlled_goal_progression_backtest import (  # noqa: E402
    calibration_analysis,
    evaluate_predictions,
)
from scripts.run_xg_goal_ablation_backtest import (  # noqa: E402
    elo_field_score,
    load_initial_ratings,
    read_production_contract,
    read_static_config,
    safe_spearman,
    same_season_ranking,
)


DATA_PATH = ROOT / "data" / "xg_2025_26" / "uefa_2025_26_matches_with_xg.csv"
MANIFEST_PATH = ROOT / "data" / "xg_2025_26" / "source_manifest.json"
STATIC_DATA_ROOT = ROOT / "data" / "backtest_stage_b_2018_2026"
STATIC_MANIFEST_PATH = (
    ROOT
    / "output"
    / "v2_dynamic_calibration_2018_2026"
    / "selected_dynamic_model.json"
)
PRODUCTION_MODEL_PATH = ROOT / "contracts" / "ao_european_elo_v2_production.json"
OUTPUT_ROOT = ROOT / "output" / "fotmob_xg_backtest_2025_26"
SEASON = "2025/26"
BASELINE_ARM = "GD_PRODUCTION"
EXPECTED_COMPETITION_COUNTS = {"UCL": 281, "UEL": 271, "UECL": 409}
EXPECTED_MATCHES = 961
EXPECTED_XG_MATCHES = 606


@dataclass(frozen=True, order=True)
class ConfirmatoryXGArm:
    key: str
    rho: float
    xg_scale: float
    xg_mode: str
    evidence_basis: str

    def validate(self) -> None:
        if not self.key:
            raise ValueError("xG arm key cannot be empty")
        XGBlendConfig(self.rho, self.xg_scale, self.xg_mode).validate()
        if self.key == BASELINE_ARM and self.rho != 0.0:
            raise ValueError("Production baseline must use rho=0")
        if self.key != BASELINE_ARM and self.rho <= 0.0:
            raise ValueError("xG candidate arms must use rho>0")


@dataclass
class ReplayResult:
    predictions: pd.DataFrame
    end_ratings: pd.DataFrame
    rating_summary: dict[str, object]


def confirmatory_arms() -> tuple[ConfirmatoryXGArm, ...]:
    """Return parameters frozen before the FotMob 2025/26 analysis."""
    arms = (
        ConfirmatoryXGArm(
            BASELINE_ARM,
            0.0,
            1.0,
            "CONVEX_BLEND",
            "ACTIVE_PRODUCTION_CONTROL",
        ),
        ConfirmatoryXGArm(
            "GD_XG_CONVEX_PREREG",
            0.05,
            1.0,
            "CONVEX_BLEND",
            "PREVIOUS_COARSE_XG_DEVELOPMENT_SELECTION",
        ),
        ConfirmatoryXGArm(
            "GD_XG_ADDITIVE_PREREG",
            0.50,
            0.75,
            "ADDITIVE_LUCK_CORRECTION",
            "PREVIOUS_COARSE_XG_DEVELOPMENT_SELECTION",
        ),
        ConfirmatoryXGArm(
            "GD_XG_BOUNDED_PREREG",
            0.05,
            1.0,
            "DIRECTION_PRESERVING_LUCK_CORRECTION",
            "PREVIOUS_COARSE_XG_DEVELOPMENT_SELECTION",
        ),
    )
    for arm in arms:
        arm.validate()
    return arms


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a confirmatory chronological FotMob xG backtest over the "
            "frozen 2025/26 AO production Elo model"
        )
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
        raise ValueError("--bootstrap-samples must be at least 100")

    events = read_fotmob_xg_dataset(args.data.resolve(), strict_contract=True)
    manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    validate_manifest(manifest, events)
    static_config = read_static_config(args.static_manifest.resolve())
    production = read_production_contract(args.production_model.resolve())
    initial_by_season = load_initial_ratings(
        args.static_data_root.resolve(),
        events,
        static_config,
    )
    initial_ratings = initial_by_season[SEASON]
    target = schedule_adjusted_team_performance(events)
    arms = confirmatory_arms()

    print(
        f"FotMob xG confirmatory backtest: {len(events)} matches, "
        f"{int(events['xg_analysis_eligible'].sum())} eligible xG matches, "
        f"{len(arms)} fixed arms",
        flush=True,
    )
    evaluations = {
        arm.key: replay_arm(events, initial_ratings, production, arm)
        for arm in arms
    }
    predictions = pd.concat(
        [evaluations[arm.key].predictions for arm in arms],
        ignore_index=True,
    )
    predictions = add_common_bands(predictions)
    end_ratings = pd.concat(
        [evaluations[arm.key].end_ratings for arm in arms],
        ignore_index=True,
    )
    ranking = pd.concat(
        [
            same_season_ranking(
                evaluations[arm.key].end_ratings,
                target,
                {SEASON},
            ).assign(model_arm=arm.key)
            for arm in arms
        ],
        ignore_index=True,
    )
    comparison = build_model_comparison(evaluations, predictions, ranking, arms)
    competition = build_segment_summary(predictions, "competition")
    stage = build_segment_summary(predictions, "round")
    month = build_segment_summary(predictions, "calendar_month")
    match_band = build_segment_summary(predictions, "match_band")
    xg_band = build_segment_summary(
        predictions.loc[predictions["xg_analysis_eligible"]].copy(),
        "xg_difference_band",
        scopes=("XG_ELIGIBLE",),
    )
    calibration = build_calibration_summary(predictions)
    uncertainty = build_uncertainty(
        predictions,
        arms,
        bootstrap_samples=args.bootstrap_samples,
    )
    decisions = classify_shadow_signals(
        comparison,
        competition,
        uncertainty,
        ranking,
        arms,
    )

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_root / "prediction_updates.csv", index=False)
    comparison.to_csv(output_root / "confirmatory_model_comparison.csv", index=False)
    competition.to_csv(output_root / "competition_summary.csv", index=False)
    stage.to_csv(output_root / "stage_summary.csv", index=False)
    month.to_csv(output_root / "month_summary.csv", index=False)
    match_band.to_csv(output_root / "match_band_summary.csv", index=False)
    xg_band.to_csv(output_root / "xg_difference_band_summary.csv", index=False)
    calibration.to_csv(output_root / "calibration_summary.csv", index=False)
    uncertainty.to_csv(output_root / "dependency_uncertainty.csv", index=False)
    ranking.to_csv(output_root / "same_season_ranking_guardrail.csv", index=False)
    end_ratings.to_csv(output_root / "end_ratings.csv", index=False)

    decision_payload = {
        "analysis": "FOTMOB_XG_CONFIRMATORY_2025_26",
        "season": SEASON,
        "matches": len(events),
        "xg_eligible_matches": int(events["xg_analysis_eligible"].sum()),
        "xg_provider": "FotMob",
        "evidence_class": "RETROSPECTIVE_FIXED_PARAMETER_CONFIRMATION",
        "production_changed": False,
        "parameters_reselected_on_2025_26": False,
        "data_source_production_eligible": bool(
            manifest["contract"]["production_eligibility"]
        ),
        "arms": [arm.__dict__ for arm in arms],
        "shadow_decisions": decisions,
        "next_step": (
            "Review fixed-parameter evidence before running the wider full-season "
            "production and shadow replay."
        ),
    }
    (output_root / "selected_shadow_model.json").write_text(
        json.dumps(decision_payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    (output_root / "backtest_report.md").write_text(
        build_report(
            comparison,
            competition,
            uncertainty,
            ranking,
            decisions,
            manifest,
        ),
        encoding="utf-8",
    )
    print("Shadow classifications:")
    for key, value in decisions.items():
        print(f"  {key}: {value['classification']}")
    print(f"Report: {output_root / 'backtest_report.md'}")


def read_fotmob_xg_dataset(path: Path, *, strict_contract: bool) -> pd.DataFrame:
    events = pd.read_csv(path)
    required = {
        "match_id",
        "season",
        "competition",
        "round",
        "tie_id",
        "kickoff_utc",
        "event_order",
        "home_team_id",
        "away_team_id",
        "home_team_name",
        "away_team_name",
        "home_goals",
        "away_goals",
        "actual_home_score",
        "is_neutral",
        "decided_on_penalties",
        "xg_home",
        "xg_away",
        "xg_difference",
        "xg_analysis_eligible",
        "xg_provider",
        "xg_type",
        "score_verified",
        "chronology_verified",
    }
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"FotMob xG dataset missing columns: {missing}")
    events = events.copy()
    events["match_id"] = events["match_id"].astype(str)
    if events["match_id"].isna().any() or events["match_id"].duplicated().any():
        raise ValueError("match_id must be non-null and unique")
    events["kickoff_utc"] = pd.to_datetime(
        events["kickoff_utc"], utc=True, errors="coerce"
    )
    if events["kickoff_utc"].isna().any():
        raise ValueError("kickoff_utc contains invalid timestamps")
    for column in (
        "is_neutral",
        "decided_on_penalties",
        "xg_analysis_eligible",
        "score_verified",
        "chronology_verified",
    ):
        events[column] = coerce_boolean_series(events[column], column)
    for column in ("home_team_id", "away_team_id", "home_goals", "away_goals"):
        values = pd.to_numeric(events[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all():
            raise ValueError(f"{column} must be finite numeric")
        if not np.equal(values, np.floor(values)).all() or values.lt(0).any():
            raise ValueError(f"{column} must contain non-negative integers")
        events[column] = values.astype(int)
    if events["home_team_id"].eq(events["away_team_id"]).any():
        raise ValueError("A match cannot have the same home and away team")
    if not events["score_verified"].all() or not events["chronology_verified"].all():
        raise ValueError("All rows must have verified scores and chronology")
    eligible = events["xg_analysis_eligible"]
    for column in ("xg_home", "xg_away"):
        values = pd.to_numeric(events[column], errors="coerce")
        if values.loc[eligible].isna().any():
            raise ValueError(f"Eligible rows require {column}")
        if values.loc[eligible].lt(0.0).any() or not np.isfinite(values.loc[eligible]).all():
            raise ValueError(f"Eligible {column} must be finite and non-negative")
        if values.loc[~eligible].notna().any():
            raise ValueError(f"Ineligible rows cannot contain {column}")
        events[column] = values.astype(float)
    if events.loc[eligible, "xg_provider"].nunique() != 1:
        raise ValueError("Confirmatory sample must use one xG provider")
    if events.loc[eligible, "xg_type"].nunique() != 1:
        raise ValueError("Confirmatory sample must use one xG definition")
    if strict_contract:
        counts = events["competition"].value_counts().to_dict()
        if len(events) != EXPECTED_MATCHES or counts != EXPECTED_COMPETITION_COUNTS:
            raise ValueError(
                f"Expected 961 matches with {EXPECTED_COMPETITION_COUNTS}, "
                f"found {len(events)} with {counts}"
            )
        if int(eligible.sum()) != EXPECTED_XG_MATCHES:
            raise ValueError(
                f"Expected {EXPECTED_XG_MATCHES} eligible xG rows, found "
                f"{int(eligible.sum())}"
            )
        if set(events["season"]) != {SEASON}:
            raise ValueError(f"Confirmatory input must contain only {SEASON}")
    return events.sort_values(["kickoff_utc", "match_id"], kind="stable").reset_index(
        drop=True
    )


def validate_manifest(manifest: dict[str, object], events: pd.DataFrame) -> None:
    rows = manifest.get("rows")
    contract = manifest.get("contract")
    if not isinstance(rows, dict) or not isinstance(contract, dict):
        raise ValueError("xG source manifest is missing rows or contract")
    if int(rows.get("matches", -1)) != len(events):
        raise ValueError("xG manifest match count does not match dataset")
    if int(rows.get("primary_xg_analysis_eligible", -1)) != int(
        events["xg_analysis_eligible"].sum()
    ):
        raise ValueError("xG manifest eligible count does not match dataset")
    if contract.get("missing_xg") != "never imputed":
        raise ValueError("Confirmatory xG backtest requires non-imputed xG")


def coerce_boolean_series(values: pd.Series, name: str) -> pd.Series:
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
        raise ValueError(f"{name} must contain only true/false or 0/1")
    return result.astype(bool)


def replay_arm(
    events: pd.DataFrame,
    initial_ratings: dict[int, float],
    production: dict[str, object],
    arm: ConfirmatoryXGArm,
) -> ReplayResult:
    arm.validate()
    core = production["dynamic_core"]
    goal = production["goal_margin"]
    draw = production["one_x_two_probability"]
    power = {int(team_id): float(value) for team_id, value in initial_ratings.items()}
    active_ids = sorted(
        set(events["home_team_id"].astype(int))
        | set(events["away_team_id"].astype(int))
    )
    missing = sorted(set(active_ids) - set(power))
    if missing:
        raise ValueError(f"Initial ratings missing teams: {missing[:5]}")
    start = dict(power)
    rows: list[dict[str, object]] = []
    max_zero_sum_error = 0.0

    for row in events.itertuples(index=False):
        home_id = int(row.home_team_id)
        away_id = int(row.away_team_id)
        penalties = bool(row.decided_on_penalties)
        home_goals, away_goals = elo_field_score(
            int(row.home_goals), int(row.away_goals), penalties
        )
        eligible = bool(row.xg_analysis_eligible)
        xg_home = float(row.xg_home) if eligible else None
        xg_away = float(row.xg_away) if eligible else None
        update = update_match_elo_with_xg(
            power[home_id],
            power[away_id],
            home_goals,
            away_goals,
            k_factor=float(core["k_factor"]),
            elo_scale=float(core["elo_scale"]),
            home_advantage=float(core["home_advantage"]),
            is_neutral=bool(row.is_neutral),
            decided_on_penalties=penalties,
            goal_difference_enabled=True,
            goal_alpha=float(goal["alpha"]),
            goal_tau=float(goal["tau"]),
            goal_difference_cap=int(goal["goal_difference_cap"]),
            xg_config=XGBlendConfig(arm.rho, arm.xg_scale, arm.xg_mode),
            xg_home=xg_home,
            xg_away=xg_away,
        )
        probabilities = one_x_two_probabilities_scalar(
            update.expected_home_score,
            float(draw["draw_at_even"]),
            float(draw["draw_shape"]),
        )
        actual_score = float(row.actual_home_score)
        if actual_score not in {0.0, 0.5, 1.0}:
            raise ValueError(f"{row.match_id}: invalid actual_home_score")
        observed = 0 if actual_score == 1.0 else 1 if actual_score == 0.5 else 2
        target = tuple(1.0 if index == observed else 0.0 for index in range(3))
        brier = sum(
            (probability - actual) ** 2
            for probability, actual in zip(probabilities, target)
        )
        log_loss = -math.log(max(probabilities[observed], 1e-15))
        power[home_id] = update.home_rating_post
        power[away_id] = update.away_rating_post
        max_zero_sum_error = max(max_zero_sum_error, update.zero_sum_error)
        tie_id = None if pd.isna(row.tie_id) else str(row.tie_id)
        rows.append(
            {
                "model_arm": arm.key,
                "evidence_basis": arm.evidence_basis,
                "rho": arm.rho,
                "xg_scale": arm.xg_scale,
                "xg_mode": arm.xg_mode,
                "match_id": str(row.match_id),
                "season": str(row.season),
                "competition": str(row.competition),
                "round": str(row.round),
                "kickoff_utc": row.kickoff_utc,
                "calendar_month": row.kickoff_utc.strftime("%Y-%m"),
                "event_order": int(row.event_order),
                "tie_id": tie_id,
                "home_team_id": home_id,
                "away_team_id": away_id,
                "home_team_name": str(row.home_team_name),
                "away_team_name": str(row.away_team_name),
                "home_goals": int(row.home_goals),
                "away_goals": int(row.away_goals),
                "decided_on_penalties": penalties,
                "xg_analysis_eligible": eligible,
                "xg_home": xg_home,
                "xg_away": xg_away,
                "xg_difference": None if not eligible else xg_home - xg_away,
                "home_rating_pre": update.home_rating_pre,
                "away_rating_pre": update.away_rating_pre,
                "effective_rating_difference": update.effective_rating_difference,
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
                "xg_home_score": update.xg_home_score,
                "result_residual": update.result_residual,
                "xg_residual": update.xg_residual,
                "blended_residual": update.blended_residual,
                "power_delta": update.power_delta,
                "home_rating_post": update.home_rating_post,
                "away_rating_post": update.away_rating_post,
                "zero_sum_error": update.zero_sum_error,
            }
        )

    names = dict(
        zip(events["home_team_id"].astype(int), events["home_team_name"].astype(str))
    )
    names.update(
        zip(events["away_team_id"].astype(int), events["away_team_name"].astype(str))
    )
    end_ratings = pd.DataFrame(
        [
            {
                "model_arm": arm.key,
                "season": SEASON,
                "team_id": team_id,
                "team_name": names.get(team_id, str(team_id)),
                "start_rating": start[team_id],
                "end_live_rating": power[team_id],
                "rating_change": power[team_id] - start[team_id],
            }
            for team_id in active_ids
        ]
    )
    prediction_frame = pd.DataFrame(rows)
    start_values = np.array([start[team_id] for team_id in active_ids])
    end_values = np.array([power[team_id] for team_id in active_ids])
    summary = {
        "model_arm": arm.key,
        "matches": len(prediction_frame),
        "xg_matches": int(prediction_frame["xg_analysis_eligible"].sum()),
        "teams": len(active_ids),
        "rating_min": float(end_values.min()),
        "rating_max": float(end_values.max()),
        "rating_mean": float(end_values.mean()),
        "rating_std": float(end_values.std()),
        "mean_abs_match_delta": float(prediction_frame["power_delta"].abs().mean()),
        "max_abs_match_delta": float(prediction_frame["power_delta"].abs().max()),
        "max_abs_rating_change": float(np.max(np.abs(end_values - start_values))),
        "start_end_rank_correlation": safe_spearman(start_values, end_values),
        "total_elo_error": abs(float(end_values.sum() - start_values.sum())),
        "max_pair_sum_error": max_zero_sum_error,
    }
    return ReplayResult(prediction_frame, end_ratings, summary)


def add_common_bands(predictions: pd.DataFrame) -> pd.DataFrame:
    result = predictions.copy()
    baseline = result.loc[result["model_arm"].eq(BASELINE_ARM)].set_index("match_id")
    distance = (baseline["expected_home_score"] - 0.5).abs()
    match_band = pd.Series(
        np.select(
            [distance.lt(0.10), distance.lt(0.25)],
            ["BALANCED", "MODERATE_FAVORITE"],
            default="STRONG_FAVORITE",
        ),
        index=baseline.index,
    )
    xg_abs = baseline["xg_difference"].abs()
    xg_band = pd.Series(
        np.select(
            [xg_abs.lt(0.50), xg_abs.lt(1.00)],
            ["LOW", "MEDIUM"],
            default="HIGH",
        ),
        index=baseline.index,
    ).where(baseline["xg_analysis_eligible"], "NO_XG")
    result["match_band"] = result["match_id"].map(match_band)
    result["xg_difference_band"] = result["match_id"].map(xg_band)
    return result


def sample_frame(predictions: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "ALL_961":
        return predictions
    if scope == "XG_ELIGIBLE":
        return predictions.loc[predictions["xg_analysis_eligible"]]
    raise ValueError(f"Unknown sample scope: {scope}")


def build_model_comparison(
    evaluations: dict[str, ReplayResult],
    predictions: pd.DataFrame,
    ranking: pd.DataFrame,
    arms: tuple[ConfirmatoryXGArm, ...],
) -> pd.DataFrame:
    rank_all = ranking.loc[ranking["competition"].eq("ALL")].set_index("model_arm")
    rows: list[dict[str, object]] = []
    for scope in ("ALL_961", "XG_ELIGIBLE"):
        scoped = sample_frame(predictions, scope)
        for arm in arms:
            frame = scoped.loc[scoped["model_arm"].eq(arm.key)]
            rows.append(
                {
                    "sample_scope": scope,
                    "model_arm": arm.key,
                    "rho": arm.rho,
                    "xg_scale": arm.xg_scale,
                    "xg_mode": arm.xg_mode,
                    **evaluate_predictions(frame),
                    **{
                        key: value
                        for key, value in evaluations[arm.key].rating_summary.items()
                        if key not in {"model_arm", "matches", "xg_matches"}
                    },
                    "ranking_score": float(rank_all.loc[arm.key, "ranking_score"]),
                    "pairwise_accuracy": float(
                        rank_all.loc[arm.key, "pairwise_accuracy"]
                    ),
                }
            )
    result = pd.DataFrame(rows)
    for scope in result["sample_scope"].unique():
        mask = result["sample_scope"].eq(scope)
        baseline = result.loc[mask & result["model_arm"].eq(BASELINE_ARM)].iloc[0]
        for metric in (
            "brier_1x2",
            "log_loss_1x2",
            "accuracy_1x2",
            "multiclass_ece",
            "ranking_score",
            "pairwise_accuracy",
        ):
            result.loc[mask, f"{metric}_delta_vs_production"] = (
                result.loc[mask, metric] - float(baseline[metric])
            )
    return result


def build_segment_summary(
    predictions: pd.DataFrame,
    segment: str,
    *,
    scopes: tuple[str, ...] = ("ALL_961", "XG_ELIGIBLE"),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scope in scopes:
        scoped = sample_frame(predictions, scope)
        for (arm, value), frame in scoped.groupby(["model_arm", segment], sort=True):
            rows.append(
                {
                    "sample_scope": scope,
                    "model_arm": arm,
                    segment: value,
                    **evaluate_predictions(frame),
                }
            )
    result = pd.DataFrame(rows)
    for scope in result["sample_scope"].unique():
        mask = result["sample_scope"].eq(scope)
        baseline = result.loc[mask & result["model_arm"].eq(BASELINE_ARM)].set_index(
            segment
        )
        for index in result.index[mask]:
            value = result.at[index, segment]
            if value not in baseline.index:
                continue
            result.at[index, "brier_delta_vs_production"] = (
                float(result.at[index, "brier_1x2"])
                - float(baseline.loc[value, "brier_1x2"])
            )
            result.at[index, "log_loss_delta_vs_production"] = (
                float(result.at[index, "log_loss_1x2"])
                - float(baseline.loc[value, "log_loss_1x2"])
            )
    return result


def build_calibration_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for scope in ("ALL_961", "XG_ELIGIBLE"):
        frame = sample_frame(predictions, scope).rename(columns={"model_arm": "model"})
        calibrated = calibration_analysis(frame)
        calibrated.insert(0, "sample_scope", scope)
        calibrated = calibrated.rename(columns={"model": "model_arm"})
        rows.append(calibrated)
    return pd.concat(rows, ignore_index=True)


def build_uncertainty(
    predictions: pd.DataFrame,
    arms: tuple[ConfirmatoryXGArm, ...],
    *,
    bootstrap_samples: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for scope in ("ALL_961", "XG_ELIGIBLE"):
        scoped = sample_frame(predictions, scope)
        baseline = scoped.loc[scoped["model_arm"].eq(BASELINE_ARM)].set_index(
            "match_id"
        )
        for arm_index, arm in enumerate(arms[1:], start=1):
            candidate = scoped.loc[scoped["model_arm"].eq(arm.key)].set_index(
                "match_id"
            )
            if not candidate.index.equals(baseline.index):
                raise ValueError("xG uncertainty comparison requires paired matches")
            for loss_index, loss in enumerate(("brier_1x2", "log_loss_1x2")):
                paired = candidate[
                    [
                        "season",
                        "home_team_id",
                        "away_team_id",
                        "kickoff_utc",
                        "tie_id",
                    ]
                ].reset_index()
                paired["loss_difference"] = (
                    candidate[loss].to_numpy(float) - baseline[loss].to_numpy(float)
                )
                uncertainty = dependency_robust_loss_difference_ci(
                    paired,
                    bootstrap_samples=bootstrap_samples,
                    seed=20260804 + arm_index * 1009 + loss_index * 10007,
                )
                uncertainty.insert(0, "sample_scope", scope)
                uncertainty.insert(1, "candidate_arm", arm.key)
                uncertainty.insert(2, "baseline_arm", BASELINE_ARM)
                uncertainty.insert(3, "metric", loss)
                rows.append(uncertainty)
    return pd.concat(rows, ignore_index=True)


def classify_shadow_signals(
    comparison: pd.DataFrame,
    competition: pd.DataFrame,
    uncertainty: pd.DataFrame,
    ranking: pd.DataFrame,
    arms: tuple[ConfirmatoryXGArm, ...],
) -> dict[str, dict[str, object]]:
    decisions: dict[str, dict[str, object]] = {}
    pooled_rank = ranking.loc[ranking["competition"].eq("ALL")].set_index("model_arm")
    base_rank = pooled_rank.loc[BASELINE_ARM]
    for arm in arms[1:]:
        metrics = comparison.loc[comparison["model_arm"].eq(arm.key)].set_index(
            "sample_scope"
        )
        cis = uncertainty.loc[
            uncertainty["candidate_arm"].eq(arm.key)
            & uncertainty["method"].eq("conservative_envelope")
        ]
        reliable_harm = bool((cis["ci_95_lower"] > 0.0).any())
        reliable_improvement = bool(
            len(cis) == 4 and (cis["ci_95_upper"] < 0.0).all()
        )
        point_improvements = bool(
            (
                metrics[
                    [
                        "brier_1x2_delta_vs_production",
                        "log_loss_1x2_delta_vs_production",
                    ]
                ]
                < 0.0
            )
            .all()
            .all()
        )
        segments = competition.loc[
            competition["model_arm"].eq(arm.key)
            & competition["sample_scope"].eq("XG_ELIGIBLE")
        ]
        no_competition_regression = bool(
            len(segments) == 3
            and (segments["brier_delta_vs_production"] <= 0.0).all()
            and (segments["log_loss_delta_vs_production"] <= 0.0).all()
        )
        rank_delta = float(pooled_rank.loc[arm.key, "ranking_score"] - base_rank["ranking_score"])
        pairwise_delta = float(
            pooled_rank.loc[arm.key, "pairwise_accuracy"]
            - base_rank["pairwise_accuracy"]
        )
        ranking_not_worse = rank_delta >= 0.0 and pairwise_delta >= 0.0
        if reliable_harm:
            classification = "HARM_SIGNAL"
        elif (
            point_improvements
            and no_competition_regression
            and reliable_improvement
            and ranking_not_worse
        ):
            classification = "CONSISTENT_SHADOW_SIGNAL"
        else:
            classification = "MIXED_OR_INCONCLUSIVE"
        decisions[arm.key] = {
            "classification": classification,
            "point_improvement_in_both_scopes": point_improvements,
            "no_competition_regression_on_xg_sample": no_competition_regression,
            "cluster_reliable_improvement_all_metrics": reliable_improvement,
            "cluster_reliable_harm_any_metric": reliable_harm,
            "same_season_ranking_delta": rank_delta,
            "same_season_pairwise_delta": pairwise_delta,
            "same_season_ranking_guardrail_passed": ranking_not_worse,
            "production_promotion_allowed": False,
        }
    return decisions


def build_report(
    comparison: pd.DataFrame,
    competition: pd.DataFrame,
    uncertainty: pd.DataFrame,
    ranking: pd.DataFrame,
    decisions: dict[str, dict[str, object]],
    manifest: dict[str, object],
) -> str:
    view = comparison[
        [
            "sample_scope",
            "model_arm",
            "matches",
            "brier_1x2",
            "brier_1x2_delta_vs_production",
            "log_loss_1x2",
            "log_loss_1x2_delta_vs_production",
            "accuracy_1x2",
            "ranking_score",
            "pairwise_accuracy",
            "max_abs_match_delta",
        ]
    ].copy()
    comp_view = competition.loc[
        competition["sample_scope"].eq("XG_ELIGIBLE"),
        [
            "model_arm",
            "competition",
            "matches",
            "brier_delta_vs_production",
            "log_loss_delta_vs_production",
        ],
    ]
    ci_view = uncertainty.loc[
        uncertainty["method"].eq("conservative_envelope"),
        [
            "sample_scope",
            "candidate_arm",
            "metric",
            "mean_difference",
            "ci_95_lower",
            "ci_95_upper",
        ],
    ]
    rank_view = ranking.loc[
        ranking["competition"].eq("ALL"),
        ["model_arm", "ranking_score", "pairwise_accuracy"],
    ]
    decision_rows = pd.DataFrame(
        [{"model_arm": key, **value} for key, value in decisions.items()]
    )
    return "\n".join(
        [
            "# FotMob xG Katmanı 2025/26 Sabit Parametre Backtesti",
            "",
            "## Sözleşme",
            "",
            "- AO First Elo ve aktif gol farkı modeli değiştirilmedi.",
            "- 961 maç kesin UTC sırasıyla işlendi; 606 doğrulanmış FotMob xG satırı kullanıldı.",
            "- Her maçın tahmini sonuç ve o maçın xG'si görülmeden kaydedildi.",
            "- xG yalnız sonraki rating state'ini etkiledi; 355 eksik satırda production güncellemesine dönüldü.",
            "- Katsayılar 2025/26 üzerinde seçilmedi; önceki coarse-xG çalışmalarından donduruldu.",
            "- Bu tek sezon retrospektif doğrulaması production terfisi üretemez.",
            "",
            "## Ana Karşılaştırma",
            "",
            markdown_table(view),
            "",
            "## Turnuva Bazında Ortak 606 Maç",
            "",
            markdown_table(comp_view),
            "",
            "## Bağımlılık-Duyarlı Güven Aralığı",
            "",
            markdown_table(ci_view),
            "",
            "## Sıralama Guardrail'i",
            "",
            markdown_table(rank_view),
            "",
            "Bu ölçüm aynı sezon schedule-adjusted performansına göredir; forward ranking değildir.",
            "",
            "## Model Kararı",
            "",
            markdown_table(decision_rows),
            "",
            "## Veri Sınırı",
            "",
            f"Kaynak production uygunluğu: `{manifest['contract']['production_eligibility']}`. ",
            "FotMob kapsamı özellikle eleme turlarında eksiktir; eksikler doldurulmamıştır.",
        ]
    )


def markdown_table(frame: pd.DataFrame) -> str:
    def render(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6f}"
        return str(value).replace("|", "\\|")

    headers = [str(column).replace("|", "\\|") for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
