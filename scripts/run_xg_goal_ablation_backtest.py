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

from ao_elo.config import AOEuropeanEloConfig  # noqa: E402
from ao_elo.evaluation import (  # noqa: E402
    dependency_robust_loss_difference_ci,
    schedule_adjusted_team_performance,
)
from ao_elo.pipeline import compute_ao_first_elo_from_csv  # noqa: E402
from ao_elo.robustness import one_x_two_probabilities_scalar  # noqa: E402
from ao_elo.xg_live import XGBlendConfig, update_match_elo_with_xg  # noqa: E402
from scripts.run_controlled_goal_progression_backtest import (  # noqa: E402
    calibration_analysis,
    evaluate_predictions,
    segment_summary,
)


STATIC_DATA_ROOT = ROOT / "data" / "backtest_stage_b_2018_2026"
EVENTS_PATH = (
    ROOT / "data" / "external_elo_benchmark_2018_2026"
    / "exact_date_events.csv"
)
XG_PATH = ROOT / "data" / "xg_backtest_2018_2026" / "xg_matches.csv"
XG_MANIFEST_PATH = (
    ROOT / "data" / "xg_backtest_2018_2026" / "source_manifest.json"
)
STATIC_MANIFEST_PATH = (
    ROOT / "output" / "v2_dynamic_calibration_2018_2026"
    / "selected_dynamic_model.json"
)
PRODUCTION_MODEL_PATH = ROOT / "contracts" / "ao_european_elo_v2_production.json"
OUTPUT_ROOT = ROOT / "output" / "xg_goal_ablation_backtest_2018_2026"
HOLDOUT_SEASON = "2025/26"
RHO_GRID = (0.05, 0.10, 0.15, 0.20, 0.25)
XG_SCALE_GRID = (0.75, 1.00, 1.25, 1.50)
RANK_TOLERANCE = 1e-9


@dataclass(frozen=True, order=True)
class AblationCandidate:
    model: str
    goal_difference_enabled: bool
    rho: float
    xg_scale: float
    xg_mode: str = "CONVEX_BLEND"

    def validate(self) -> None:
        if self.model not in {"BASE", "GD", "XG", "GD_XG"}:
            raise ValueError(f"Unknown ablation model: {self.model}")
        XGBlendConfig(self.rho, self.xg_scale, self.xg_mode).validate()
        expects_goal = self.model in {"GD", "GD_XG"}
        expects_xg = self.model in {"XG", "GD_XG"}
        if self.goal_difference_enabled != expects_goal:
            raise ValueError(f"{self.model} goal-difference flag is inconsistent")
        if expects_xg != (self.rho > 0.0):
            raise ValueError(f"{self.model} rho is inconsistent")

    @property
    def key(self) -> str:
        if self.model in {"BASE", "GD"}:
            return self.model
        mode = {
            "CONVEX_BLEND": "",
            "ADDITIVE_LUCK_CORRECTION": "_luck",
            "DIRECTION_PRESERVING_LUCK_CORRECTION": "_bounded_luck",
        }[self.xg_mode]
        return (
            f"{self.model}{mode}_rho{self.rho:g}_cxg{self.xg_scale:g}"
        )


@dataclass
class SequenceEvaluation:
    predictions: pd.DataFrame
    end_ratings: pd.DataFrame
    season_metrics: pd.DataFrame


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the matched-sample BASE/GD/xG/GD+xG temporal ablation backtest"
        )
    )
    parser.add_argument("--static-data-root", type=Path, default=STATIC_DATA_ROOT)
    parser.add_argument("--events", type=Path, default=EVENTS_PATH)
    parser.add_argument("--xg", type=Path, default=XG_PATH)
    parser.add_argument("--xg-manifest", type=Path, default=XG_MANIFEST_PATH)
    parser.add_argument(
        "--static-manifest",
        type=Path,
        default=STATIC_MANIFEST_PATH,
    )
    parser.add_argument(
        "--production-model",
        type=Path,
        default=PRODUCTION_MODEL_PATH,
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples must be at least 100")

    static_config = read_static_config(args.static_manifest.resolve())
    production = read_production_contract(args.production_model.resolve())
    xg_manifest = json.loads(
        args.xg_manifest.resolve().read_text(encoding="utf-8")
    )
    events = read_events(args.events.resolve())
    xg = read_xg(args.xg.resolve(), events)
    initial_ratings = load_initial_ratings(
        args.static_data_root.resolve(),
        events,
        static_config,
    )
    target = schedule_adjusted_team_performance(events)
    candidates = candidate_grid()
    development_seasons = set(events["season"]) - {HOLDOUT_SEASON}
    holdout_seasons = {HOLDOUT_SEASON}

    print(
        f"xG ablation: {len(candidates)} candidates, "
        f"{len(xg)} eligible matched matches",
        flush=True,
    )
    evaluations: dict[str, SequenceEvaluation] = {}
    development_rows: list[dict[str, object]] = []
    for candidate in candidates:
        evaluation = evaluate_sequence(
            events,
            xg,
            initial_ratings,
            production,
            candidate,
        )
        evaluations[candidate.key] = evaluation
        development_rows.append(
            candidate_metric_row(
                candidate,
                evaluation,
                target,
                development_seasons,
                split="DEVELOPMENT",
            )
        )
    development_metrics = pd.DataFrame(development_rows)
    selected_xg = select_shadow_candidate(
        development_metrics,
        candidates,
        model="XG",
        baseline_key="BASE",
    )
    selected_full = select_shadow_candidate(
        development_metrics,
        candidates,
        model="GD_XG",
        baseline_key="GD",
    )
    selected = (
        candidate_by_key(candidates, "BASE"),
        candidate_by_key(candidates, "GD"),
        selected_xg,
        selected_full,
    )

    holdout_rows = [
        candidate_metric_row(
            candidate,
            evaluations[candidate.key],
            target,
            holdout_seasons,
            split="HOLDOUT",
        )
        for candidate in selected
    ]
    holdout_metrics = pd.DataFrame(holdout_rows)
    predictions = pd.concat(
        [
            add_split(
                evaluations[candidate.key].predictions,
                development_seasons,
                holdout_seasons,
            ).assign(model=candidate.model, candidate_key=candidate.key)
            for candidate in selected
        ],
        ignore_index=True,
    )
    predictions = add_match_bands(predictions)
    holdout_predictions = predictions.loc[
        predictions["evaluation_split"].eq("HOLDOUT")
    ].copy()
    competition_summary = segment_summary(
        holdout_predictions,
        "competition",
    )
    match_band_summary = segment_summary(
        holdout_predictions,
        "match_band",
    )
    xg_band_summary = segment_summary(
        holdout_predictions,
        "xg_difference_band",
    )
    calibration = calibration_analysis(holdout_predictions)
    uncertainty = build_uncertainty(
        holdout_predictions,
        bootstrap_samples=args.bootstrap_samples,
    )
    ranking_detail = pd.concat(
        [
            same_season_ranking(
                evaluations[candidate.key].end_ratings,
                target,
                holdout_seasons,
            ).assign(model=candidate.model, candidate_key=candidate.key)
            for candidate in selected
        ],
        ignore_index=True,
    )
    rating_distribution = pd.concat(
        [
            evaluations[candidate.key].season_metrics.assign(
                model=candidate.model,
                candidate_key=candidate.key,
            )
            for candidate in selected
        ],
        ignore_index=True,
    )
    decision, guardrails = model_decision(
        holdout_metrics,
        competition_summary,
        uncertainty,
        ranking_detail,
        xg_manifest,
    )

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    development_metrics.to_csv(
        output_root / "development_candidate_metrics.csv",
        index=False,
    )
    holdout_metrics.to_csv(
        output_root / "holdout_model_comparison.csv",
        index=False,
    )
    predictions.to_csv(output_root / "matched_predictions.csv", index=False)
    competition_summary.to_csv(
        output_root / "competition_summary.csv",
        index=False,
    )
    match_band_summary.to_csv(
        output_root / "match_band_summary.csv",
        index=False,
    )
    xg_band_summary.to_csv(
        output_root / "xg_difference_band_summary.csv",
        index=False,
    )
    calibration.to_csv(
        output_root / "calibration_analysis.csv",
        index=False,
    )
    uncertainty.to_csv(
        output_root / "dependency_uncertainty.csv",
        index=False,
    )
    ranking_detail.to_csv(
        output_root / "holdout_ranking.csv",
        index=False,
    )
    rating_distribution.to_csv(
        output_root / "season_rating_distribution.csv",
        index=False,
    )
    selected_payload = {
        "decision": decision,
        "production_changed": False,
        "data_source_production_eligible": bool(
            xg_manifest.get("production_eligibility", False)
        ),
        "development_selected": {
            "xg_only": candidate_payload(selected_xg),
            "gd_xg": candidate_payload(selected_full),
        },
        "active_production": {
            "goal_difference": {
                "active": True,
                "alpha": float(production["goal_margin"]["alpha"]),
                "tau": float(production["goal_margin"]["tau"]),
                "goal_difference_cap": int(
                    production["goal_margin"]["goal_difference_cap"]
                ),
            },
            "xg": {"active": False, "rho": 0.0},
        },
        "guardrails": guardrails,
    }
    (output_root / "selected_xg_model.json").write_text(
        json.dumps(selected_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_root / "backtest_report.md").write_text(
        build_report(
            xg_manifest,
            development_metrics,
            holdout_metrics,
            competition_summary,
            uncertainty,
            ranking_detail,
            rating_distribution,
            selected_xg,
            selected_full,
            decision,
            guardrails,
        ),
        encoding="utf-8",
    )
    print(f"Selected xG-only shadow: {selected_xg.key}")
    print(f"Selected GD+xG shadow: {selected_full.key}")
    print(f"Decision: {decision}")
    print(f"Report: {output_root / 'backtest_report.md'}")


def read_static_config(path: Path) -> AOEuropeanEloConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = AOEuropeanEloConfig(**payload["static_config"])
    config.validate()
    return config


def read_production_contract(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"dynamic_core", "one_x_two_probability", "goal_margin"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Production contract missing keys: {missing}")
    core = payload["dynamic_core"]
    goal = payload["goal_margin"]
    if not math.isclose(
        float(core["k_factor"]),
        103.98098633392752,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("xG ablation requires frozen production K")
    expected_goal = {
        "active": True,
        "alpha": 0.10,
        "tau": 300.0,
        "goal_difference_cap": 4,
    }
    for key, expected in expected_goal.items():
        actual = goal[key]
        if isinstance(expected, bool):
            if actual is not expected:
                raise ValueError(f"xG ablation requires goal_margin.{key}")
        elif not math.isclose(
            float(actual),
            float(expected),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"xG ablation requires goal_margin.{key}={expected}"
            )
    return payload


def read_events(path: Path) -> pd.DataFrame:
    events = pd.read_csv(path)
    required = {
        "match_id",
        "season",
        "competition",
        "kickoff_utc",
        "tie_id",
        "home_team_id",
        "away_team_id",
        "home_goals",
        "away_goals",
        "is_neutral",
        "decided_on_penalties",
    }
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"events missing columns: {missing}")
    events = events.copy()
    events["kickoff_utc"] = pd.to_datetime(
        events["kickoff_utc"],
        utc=True,
        errors="coerce",
    )
    if events["kickoff_utc"].isna().any():
        raise ValueError("events contains invalid exact UTC timestamps")
    if events["match_id"].isna().any() or events["match_id"].duplicated().any():
        raise ValueError("events.match_id must be non-null and unique")
    return events.sort_values(["kickoff_utc", "match_id"]).reset_index(drop=True)


def read_xg(path: Path, events: pd.DataFrame) -> pd.DataFrame:
    xg = pd.read_csv(path)
    required = {
        "match_id",
        "xg_home",
        "xg_away",
        "xg_type",
        "duration_scope",
        "provider",
        "snapshot_time",
        "penalty_shootout_excluded",
        "eligible_for_ablation",
    }
    missing = sorted(required - set(xg.columns))
    if missing:
        raise ValueError(f"xG data missing columns: {missing}")
    xg = xg.loc[xg["eligible_for_ablation"].eq(True)].copy()
    if xg.empty:
        raise ValueError("xG data has no eligible rows")
    if xg["match_id"].isna().any() or xg["match_id"].duplicated().any():
        raise ValueError("Eligible xG match IDs must be non-null and unique")
    if not set(xg["match_id"]).issubset(set(events["match_id"])):
        raise ValueError("Eligible xG contains unknown AO match IDs")
    for column in ("xg_home", "xg_away"):
        values = pd.to_numeric(xg[column], errors="coerce")
        if values.isna().any() or not values.map(math.isfinite).all():
            raise ValueError(f"xG {column} must be finite")
        if values.lt(0.0).any():
            raise ValueError(f"xG {column} must be non-negative")
        xg[column] = values.astype(float)
    if not xg["duration_scope"].eq("90_minutes_ft_only").all():
        raise ValueError("Ablation requires audited 90-minute FT xG")
    if not xg["penalty_shootout_excluded"].eq(True).all():
        raise ValueError("Ablation xG must exclude penalty shootouts")
    if xg["provider"].nunique() != 1 or xg["xg_type"].nunique() != 1:
        raise ValueError("Ablation requires one provider and one xG definition")
    return xg.sort_values(["kickoff_utc", "match_id"]).reset_index(drop=True)


def load_initial_ratings(
    static_root: Path,
    events: pd.DataFrame,
    static_config: AOEuropeanEloConfig,
) -> dict[str, dict[int, float]]:
    result: dict[str, dict[int, float]] = {}
    for season in events["season"].drop_duplicates():
        folder = static_root / str(season).replace("/", "-")
        ratings = compute_ao_first_elo_from_csv(
            folder / "teams.csv",
            folder / "country_coefficients.csv",
            folder / "domestic_context.csv",
            folder / "club_european_points.csv",
            static_config,
        )
        mapping = {
            int(row.team_id): float(row.ao_first_elo)
            for row in ratings[["team_id", "ao_first_elo"]].itertuples(
                index=False
            )
        }
        active = set(
            events.loc[events["season"].eq(season), "home_team_id"].astype(int)
        ) | set(
            events.loc[events["season"].eq(season), "away_team_id"].astype(int)
        )
        missing = sorted(active - set(mapping))
        if missing:
            raise ValueError(f"{season}: missing AO First Elo for {missing[:5]}")
        result[str(season)] = mapping
    return result


def candidate_grid() -> tuple[AblationCandidate, ...]:
    candidates = [
        AblationCandidate("BASE", False, 0.0, 1.0),
        AblationCandidate("GD", True, 0.0, 1.0),
    ]
    candidates.extend(
        AblationCandidate(model, model == "GD_XG", rho, xg_scale)
        for model in ("XG", "GD_XG")
        for rho in RHO_GRID
        for xg_scale in XG_SCALE_GRID
    )
    result = tuple(candidates)
    for candidate in result:
        candidate.validate()
    if len(result) != 42:
        raise ValueError(f"Expected 42 ablation candidates, found {len(result)}")
    return result


def evaluate_sequence(
    events: pd.DataFrame,
    xg: pd.DataFrame,
    initial_ratings: dict[str, dict[int, float]],
    production: dict[str, object],
    candidate: AblationCandidate,
) -> SequenceEvaluation:
    candidate.validate()
    core = production["dynamic_core"]
    goal = production["goal_margin"]
    draw = production["one_x_two_probability"]
    xg_by_match = xg.set_index("match_id")[["xg_home", "xg_away"]]
    prediction_rows: list[dict[str, object]] = []
    end_rows: list[dict[str, object]] = []
    season_rows: list[dict[str, object]] = []

    for season, season_events in events.groupby("season", sort=False):
        power = dict(initial_ratings[str(season)])
        start = dict(power)
        deltas: list[float] = []
        max_zero_sum_error = 0.0
        for row in season_events.itertuples(index=False):
            match_id = str(row.match_id)
            home_id = int(row.home_team_id)
            away_id = int(row.away_team_id)
            xg_values = xg_by_match.loc[match_id] if match_id in xg_by_match.index else None
            decided_on_penalties = bool(row.decided_on_penalties)
            home_goals, away_goals = elo_field_score(
                int(row.home_goals),
                int(row.away_goals),
                decided_on_penalties,
            )
            update = update_match_elo_with_xg(
                power[home_id],
                power[away_id],
                home_goals,
                away_goals,
                k_factor=float(core["k_factor"]),
                elo_scale=float(core["elo_scale"]),
                home_advantage=float(core["home_advantage"]),
                is_neutral=bool(row.is_neutral),
                decided_on_penalties=decided_on_penalties,
                goal_difference_enabled=candidate.goal_difference_enabled,
                goal_alpha=(
                    float(goal["alpha"])
                    if candidate.goal_difference_enabled
                    else 0.0
                ),
                goal_tau=float(goal["tau"]),
                goal_difference_cap=int(goal["goal_difference_cap"]),
                xg_config=XGBlendConfig(
                    candidate.rho,
                    candidate.xg_scale,
                    candidate.xg_mode,
                ),
                xg_home=(
                    None if xg_values is None else float(xg_values["xg_home"])
                ),
                xg_away=(
                    None if xg_values is None else float(xg_values["xg_away"])
                ),
            )
            probabilities = one_x_two_probabilities_scalar(
                update.expected_home_score,
                float(draw["draw_at_even"]),
                float(draw["draw_shape"]),
            )
            observed = (
                1
                if decided_on_penalties
                else 0
                if int(row.home_goals) > int(row.away_goals)
                else 1
                if int(row.home_goals) == int(row.away_goals)
                else 2
            )
            target_vector = tuple(
                1.0 if index == observed else 0.0 for index in range(3)
            )
            brier = sum(
                (probability - actual) ** 2
                for probability, actual in zip(probabilities, target_vector)
            )
            log_loss = -math.log(max(probabilities[observed], 1e-15))
            power[home_id] = update.home_rating_post
            power[away_id] = update.away_rating_post
            deltas.append(abs(update.power_delta))
            max_zero_sum_error = max(max_zero_sum_error, update.zero_sum_error)

            if xg_values is not None:
                tie_id = None if pd.isna(row.tie_id) else str(row.tie_id)
                prediction_rows.append(
                    {
                        "match_id": match_id,
                        "season": str(season),
                        "kickoff_utc": row.kickoff_utc,
                        "competition": str(row.competition),
                        "round": str(row.round),
                        "tie_id": tie_id,
                        "home_team_id": home_id,
                        "away_team_id": away_id,
                        "home_goals": int(row.home_goals),
                        "away_goals": int(row.away_goals),
                        "effective_rating_difference": (
                            update.effective_rating_difference
                        ),
                        "expected_home_score": update.expected_home_score,
                        "home_probability": probabilities[0],
                        "draw_probability": probabilities[1],
                        "away_probability": probabilities[2],
                        "actual_class": observed,
                        "predicted_class": int(np.argmax(probabilities)),
                        "brier_1x2": brier,
                        "log_loss_1x2": log_loss,
                        "goal_difference": update.goal_difference,
                        "goal_multiplier": (
                            update.goal_difference_multiplier
                        ),
                        "xg_home": float(xg_values["xg_home"]),
                        "xg_away": float(xg_values["xg_away"]),
                        "xg_difference": (
                            float(xg_values["xg_home"])
                            - float(xg_values["xg_away"])
                        ),
                        "xg_home_score": update.xg_home_score,
                        "result_residual": update.result_residual,
                        "xg_residual": update.xg_residual,
                        "blended_residual": update.blended_residual,
                        "power_delta": update.power_delta,
                    }
                )

        active_ids = sorted(
            set(season_events["home_team_id"].astype(int))
            | set(season_events["away_team_id"].astype(int))
        )
        initial_total = sum(start[team_id] for team_id in active_ids)
        final_total = sum(power[team_id] for team_id in active_ids)
        start_values = np.array([start[team_id] for team_id in active_ids])
        end_values = np.array([power[team_id] for team_id in active_ids])
        season_rows.append(
            {
                "season": str(season),
                "matches": len(season_events),
                "matched_xg_matches": int(
                    season_events["match_id"].isin(xg_by_match.index).sum()
                ),
                "teams": len(active_ids),
                "rating_min": float(end_values.min()),
                "rating_max": float(end_values.max()),
                "rating_mean": float(end_values.mean()),
                "rating_std": float(end_values.std()),
                "max_abs_match_delta": max(deltas),
                "max_abs_rating_change": float(
                    np.max(np.abs(end_values - start_values))
                ),
                "start_end_rank_correlation": safe_spearman(
                    start_values,
                    end_values,
                ),
                "total_elo_error": abs(final_total - initial_total),
                "max_pair_sum_error": max_zero_sum_error,
            }
        )
        end_rows.extend(
            {
                "season": str(season),
                "team_id": team_id,
                "start_rating": start[team_id],
                "end_live_rating": power[team_id],
            }
            for team_id in active_ids
        )
    return SequenceEvaluation(
        predictions=pd.DataFrame(prediction_rows),
        end_ratings=pd.DataFrame(end_rows),
        season_metrics=pd.DataFrame(season_rows),
    )


def elo_field_score(
    home_goals: int,
    away_goals: int,
    decided_on_penalties: bool,
) -> tuple[int, int]:
    del decided_on_penalties
    return home_goals, away_goals


def candidate_metric_row(
    candidate: AblationCandidate,
    evaluation: SequenceEvaluation,
    target: pd.DataFrame,
    seasons: set[str],
    *,
    split: str,
) -> dict[str, object]:
    predictions = evaluation.predictions.loc[
        evaluation.predictions["season"].isin(seasons)
    ]
    metrics = evaluate_predictions(predictions)
    ranking = same_season_ranking(evaluation.end_ratings, target, seasons)
    all_ranking = ranking.loc[ranking["competition"].eq("ALL")]
    season_metrics = evaluation.season_metrics.loc[
        evaluation.season_metrics["season"].isin(seasons)
    ]
    return {
        "split": split,
        "model": candidate.model,
        "candidate_key": candidate.key,
        "goal_difference_enabled": candidate.goal_difference_enabled,
        "rho": candidate.rho,
        "xg_scale": candidate.xg_scale,
        "xg_mode": candidate.xg_mode,
        **metrics,
        "ranking_score": (
            float(all_ranking.iloc[0]["ranking_score"])
            if len(all_ranking)
            else float("nan")
        ),
        "pairwise_accuracy": (
            float(all_ranking.iloc[0]["pairwise_accuracy"])
            if len(all_ranking)
            else float("nan")
        ),
        "max_abs_match_delta": float(
            season_metrics["max_abs_match_delta"].max()
        ),
        "max_abs_rating_change": float(
            season_metrics["max_abs_rating_change"].max()
        ),
        "minimum_start_end_rank_correlation": float(
            season_metrics["start_end_rank_correlation"].min()
        ),
        "maximum_total_elo_error": float(
            season_metrics["total_elo_error"].max()
        ),
        "maximum_pair_sum_error": float(
            season_metrics["max_pair_sum_error"].max()
        ),
    }


def same_season_ranking(
    end_ratings: pd.DataFrame,
    target: pd.DataFrame,
    seasons: set[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    eligible = target.loc[target["season"].isin(seasons)]
    for (season, competition), actual in eligible.groupby(
        ["season", "competition"],
        sort=True,
    ):
        predicted = end_ratings.loc[end_ratings["season"].eq(season)]
        table = actual[["team_id", "schedule_adjusted_score"]].merge(
            predicted[["team_id", "end_live_rating"]],
            on="team_id",
            validate="one_to_one",
        )
        if len(table) < 3:
            continue
        rows.append(
            {
                "season": str(season),
                "competition": str(competition),
                "teams": len(table),
                "team_weight": len(table),
                "pair_weight": len(table) * (len(table) - 1) / 2,
                "ranking_score": safe_spearman(
                    table["end_live_rating"].to_numpy(float),
                    table["schedule_adjusted_score"].to_numpy(float),
                ),
                "pairwise_accuracy": pairwise_ranking_accuracy(
                    table["end_live_rating"].to_numpy(float),
                    table["schedule_adjusted_score"].to_numpy(float),
                ),
            }
        )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return pd.DataFrame(
            columns=[
                "season",
                "competition",
                "teams",
                "team_weight",
                "pair_weight",
                "ranking_score",
                "pairwise_accuracy",
            ]
        )
    summary = {
        "season": "POOLED",
        "competition": "ALL",
        "teams": int(detail["teams"].sum()),
        "team_weight": float(detail["team_weight"].sum()),
        "pair_weight": float(detail["pair_weight"].sum()),
        "ranking_score": float(
            np.average(
                detail["ranking_score"],
                weights=detail["team_weight"],
            )
        ),
        "pairwise_accuracy": float(
            np.average(
                detail["pairwise_accuracy"],
                weights=detail["pair_weight"],
            )
        ),
    }
    return pd.concat([detail, pd.DataFrame([summary])], ignore_index=True)


def safe_spearman(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) != len(second) or len(first) < 2:
        raise ValueError("Spearman inputs must have the same length >= 2")
    if np.ptp(first) == 0.0 or np.ptp(second) == 0.0:
        return 0.0
    value = pd.Series(first).corr(pd.Series(second), method="spearman")
    return 0.0 if pd.isna(value) else float(value)


def pairwise_ranking_accuracy(
    predicted: np.ndarray,
    actual: np.ndarray,
) -> float:
    total = 0.0
    correct = 0.0
    for left in range(len(predicted)):
        for right in range(left + 1, len(predicted)):
            actual_sign = np.sign(actual[left] - actual[right])
            if actual_sign == 0:
                continue
            predicted_sign = np.sign(predicted[left] - predicted[right])
            total += 1.0
            if predicted_sign == actual_sign:
                correct += 1.0
            elif predicted_sign == 0:
                correct += 0.5
    return correct / total if total else 0.0


def select_shadow_candidate(
    metrics: pd.DataFrame,
    candidates: tuple[AblationCandidate, ...],
    *,
    model: str,
    baseline_key: str,
) -> AblationCandidate:
    baseline = metrics.loc[metrics["candidate_key"].eq(baseline_key)]
    if len(baseline) != 1:
        raise ValueError(f"Expected one baseline row for {baseline_key}")
    base = baseline.iloc[0]
    eligible = metrics.loc[metrics["model"].eq(model)].copy()
    eligible["ranking_safe"] = (
        eligible["ranking_score"].ge(
            float(base["ranking_score"]) - RANK_TOLERANCE
        )
        & eligible["pairwise_accuracy"].ge(
            float(base["pairwise_accuracy"]) - RANK_TOLERANCE
        )
    )
    pool = eligible.loc[eligible["ranking_safe"]]
    if pool.empty:
        pool = eligible
    selected = pool.sort_values(
        [
            "ranking_score",
            "pairwise_accuracy",
            "brier_1x2",
            "log_loss_1x2",
            "rho",
            "xg_scale",
        ],
        ascending=[False, False, True, True, True, True],
    ).iloc[0]
    return candidate_by_key(candidates, str(selected["candidate_key"]))


def candidate_by_key(
    candidates: tuple[AblationCandidate, ...],
    key: str,
) -> AblationCandidate:
    values = [candidate for candidate in candidates if candidate.key == key]
    if len(values) != 1:
        raise ValueError(f"Expected one candidate for {key}")
    return values[0]


def candidate_payload(candidate: AblationCandidate) -> dict[str, object]:
    return {
        "candidate_key": candidate.key,
        "model": candidate.model,
        "goal_difference_enabled": candidate.goal_difference_enabled,
        "rho": candidate.rho,
        "xg_scale": candidate.xg_scale,
        "xg_mode": candidate.xg_mode,
    }


def add_split(
    predictions: pd.DataFrame,
    development_seasons: set[str],
    holdout_seasons: set[str],
) -> pd.DataFrame:
    result = predictions.copy()
    result["evaluation_split"] = np.where(
        result["season"].isin(holdout_seasons),
        "HOLDOUT",
        np.where(
            result["season"].isin(development_seasons),
            "DEVELOPMENT",
            "UNASSIGNED",
        ),
    )
    return result


def add_match_bands(predictions: pd.DataFrame) -> pd.DataFrame:
    result = predictions.copy()
    distance = (result["expected_home_score"] - 0.5).abs()
    result["match_band"] = np.select(
        [distance.lt(0.10), distance.lt(0.25)],
        ["BALANCED", "MODERATE_FAVORITE"],
        default="STRONG_FAVORITE",
    )
    xg_abs = result["xg_difference"].abs()
    result["xg_difference_band"] = np.select(
        [xg_abs.lt(0.50), xg_abs.lt(1.00)],
        ["LOW", "MEDIUM"],
        default="HIGH",
    )
    return result


def build_uncertainty(
    predictions: pd.DataFrame,
    *,
    bootstrap_samples: int,
) -> pd.DataFrame:
    comparisons = (
        ("XG", "BASE"),
        ("GD_XG", "GD"),
        ("GD", "BASE"),
    )
    rows: list[pd.DataFrame] = []
    for candidate_model, baseline_model in comparisons:
        candidate = predictions.loc[
            predictions["model"].eq(candidate_model)
        ].set_index("match_id")
        baseline = predictions.loc[
            predictions["model"].eq(baseline_model)
        ].set_index("match_id")
        common = candidate.index.intersection(baseline.index)
        if len(common) != len(candidate) or len(common) != len(baseline):
            raise ValueError("Uncertainty comparison is not a paired sample")
        for loss in ("brier_1x2", "log_loss_1x2"):
            paired = candidate.loc[
                common,
                [
                    "season",
                    "home_team_id",
                    "away_team_id",
                    "kickoff_utc",
                    "tie_id",
                ],
            ].reset_index()
            paired["loss_difference"] = (
                candidate.loc[common, loss].to_numpy(float)
                - baseline.loc[common, loss].to_numpy(float)
            )
            uncertainty = dependency_robust_loss_difference_ci(
                paired,
                bootstrap_samples=bootstrap_samples,
            )
            uncertainty["candidate_model"] = candidate_model
            uncertainty["baseline_model"] = baseline_model
            uncertainty["loss"] = loss
            rows.append(uncertainty)
    return pd.concat(rows, ignore_index=True)


def model_decision(
    holdout: pd.DataFrame,
    competition: pd.DataFrame,
    uncertainty: pd.DataFrame,
    ranking: pd.DataFrame,
    xg_manifest: dict[str, object],
) -> tuple[str, dict[str, object]]:
    indexed = holdout.set_index("model")
    full = indexed.loc["GD_XG"]
    gd = indexed.loc["GD"]
    pooled_ranking = ranking.loc[ranking["competition"].eq("ALL")].set_index(
        "model"
    )
    ci = uncertainty.loc[
        uncertainty["candidate_model"].eq("GD_XG")
        & uncertainty["baseline_model"].eq("GD")
        & uncertainty["loss"].eq("brier_1x2")
        & uncertainty["method"].eq("conservative_envelope")
    ]
    reliable_improvement = bool(
        len(ci) == 1 and float(ci.iloc[0]["ci_95_upper"]) < 0.0
    )
    gd_segments = competition.loc[
        competition["model"].eq("GD"),
        ["competition", "brier_1x2", "log_loss_1x2"],
    ].set_index("competition")
    full_segments = competition.loc[
        competition["model"].eq("GD_XG"),
        ["competition", "brier_1x2", "log_loss_1x2"],
    ].set_index("competition")
    common_segments = gd_segments.index.intersection(full_segments.index)
    no_competition_loss_regression = bool(
        len(common_segments) == 3
        and (
            full_segments.loc[common_segments, "brier_1x2"]
            <= gd_segments.loc[common_segments, "brier_1x2"] + RANK_TOLERANCE
        ).all()
        and (
            full_segments.loc[common_segments, "log_loss_1x2"]
            <= gd_segments.loc[common_segments, "log_loss_1x2"]
            + RANK_TOLERANCE
        ).all()
    )
    guardrails = {
        "holdout_brier_delta_gd_xg_vs_gd": (
            float(full["brier_1x2"]) - float(gd["brier_1x2"])
        ),
        "holdout_log_loss_delta_gd_xg_vs_gd": (
            float(full["log_loss_1x2"]) - float(gd["log_loss_1x2"])
        ),
        "holdout_ranking_delta_gd_xg_vs_gd": (
            float(pooled_ranking.loc["GD_XG", "ranking_score"])
            - float(pooled_ranking.loc["GD", "ranking_score"])
        ),
        "holdout_pairwise_delta_gd_xg_vs_gd": (
            float(pooled_ranking.loc["GD_XG", "pairwise_accuracy"])
            - float(pooled_ranking.loc["GD", "pairwise_accuracy"])
        ),
        "no_competition_loss_regression": no_competition_loss_regression,
        "clustered_brier_reliable_improvement": reliable_improvement,
        "zero_sum_preserved": bool(
            holdout["maximum_total_elo_error"].max() <= 1e-9
            and holdout["maximum_pair_sum_error"].max() <= 1e-9
        ),
        "provider_production_eligible": bool(
            xg_manifest.get("production_eligibility", False)
        ),
    }
    decision = "KEEP_GD_PRODUCTION_XG_SHADOW"
    return decision, guardrails


def build_report(
    xg_manifest: dict[str, object],
    development: pd.DataFrame,
    holdout: pd.DataFrame,
    competition: pd.DataFrame,
    uncertainty: pd.DataFrame,
    ranking: pd.DataFrame,
    rating_distribution: pd.DataFrame,
    selected_xg: AblationCandidate,
    selected_full: AblationCandidate,
    decision: str,
    guardrails: dict[str, object],
) -> str:
    rows = xg_manifest["rows"]
    holdout_view = holdout[
        [
            "model",
            "candidate_key",
            "matches",
            "brier_1x2",
            "log_loss_1x2",
            "accuracy_1x2",
            "multiclass_ece",
            "ranking_score",
            "pairwise_accuracy",
            "max_abs_match_delta",
        ]
    ]
    competition_view = competition[
        [
            "model",
            "competition",
            "matches",
            "brier_1x2",
            "log_loss_1x2",
        ]
    ]
    development_selected = development.loc[
        development["candidate_key"].isin(
            [selected_xg.key, selected_full.key]
        ),
        [
            "model",
            "candidate_key",
            "matches",
            "brier_1x2",
            "log_loss_1x2",
            "ranking_score",
            "pairwise_accuracy",
        ],
    ]
    ci_view = uncertainty.loc[
        uncertainty["method"].eq("conservative_envelope"),
        [
            "candidate_model",
            "baseline_model",
            "loss",
            "mean_difference",
            "ci_95_lower",
            "ci_95_upper",
        ],
    ]
    ranking_view = ranking.loc[
        ranking["competition"].eq("ALL"),
        [
            "model",
            "ranking_score",
            "pairwise_accuracy",
            "teams",
        ],
    ]
    distribution = rating_distribution.loc[
        rating_distribution["season"].eq(HOLDOUT_SEASON),
        [
            "model",
            "rating_min",
            "rating_max",
            "rating_std",
            "max_abs_match_delta",
            "max_abs_rating_change",
            "total_elo_error",
        ],
    ]
    return f"""# AO Gol Farki ve xG Ablation Backtesti

## Model Karari

**{decision}**

Gol farki production'da aktif kalir. xG production'a alinmaz; secilen xG adaylari
yalnizca shadow mode'da tutulur. AO First Elo ve production sozlesmesi degismemistir.

## Veri Kapsami ve Siniri

- AO mac evreni: {rows["ao_matches"]}
- Kaynakta iki taraf xG bulunan mac: {rows["source_xg_matches"]}
- AO ile eslesen xG: {rows["matched_xg_matches"]}
- Katı FT/90 dakika sozlesmesini gecen mac: {rows["eligible_xg_matches"]}
- Development: 2018/19-2024/25 icindeki uygun maclar
- Temporal holdout: {HOLDOUT_SEASON}

Kaynak xG, per-shot Opta/StatsBomb xG degil; sut bolgelerinden turetilmis kaba bir
API-Football tahminidir. Kapsam sezonlar boyunca kesintilidir. Bu nedenle iyi bir
holdout sonucu bile production aktivasyonu icin yeterli kanit sayilmaz.

## Ablation Kollari

1. BASE: klasik Elo
2. GD: production gol farki (`alpha=0.10`, `tau=300`, `cap=4`)
3. XG: gol farki kapali, xG harmani aktif
4. GD_XG: production gol farki ve xG harmani birlikte

Formul:

```text
S_xG = 1 / (1 + exp(-(xG_home - xG_away) / c_xG))
Delta = K * [(1-rho) * (S-E) * M_GD + rho * (S_xG-E)]
```

xG olmayan maclarda XG kolu BASE'e, GD_XG kolu GD'ye doner. Eksik xG sifirla
doldurulmaz.

## Development Secimi

{markdown_table(development_selected)}

Secilen XG adayi: `{selected_xg.key}`

Secilen GD+xG adayi: `{selected_full.key}`

## 2025/26 Holdout Sonuclari

{markdown_table(holdout_view)}

## Turnuva Segmentleri

{markdown_table(competition_view)}

## Ranking Guardrail

{markdown_table(ranking_view)}

## Clustered Belirsizlik

Negatif fark aday model lehinedir.

{markdown_table(ci_view)}

## Rating Dagilimi ve Korunum

{markdown_table(distribution)}

## Guardrail Ozeti

```json
{json.dumps(guardrails, indent=2, ensure_ascii=False)}
```

## Sonuc

xG fikri reddedilmemistir; mevcut acik kaynak kaniti production karari icin yetersiz
ve metrik tanimi fazla kabadir. Bir sonraki production-grade test ayni kod yolunu,
tek bir per-shot xG/npxG saglayicisinin kesintisiz UCL/UEL/UECL verisiyle yeniden
calistirmalidir. O zamana kadar sade ve kanitlanmis GD modeli aktif kalir.
"""


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_Veri yok._"
    columns = list(frame.columns)
    rows = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for record in frame.itertuples(index=False, name=None):
        values = []
        for value in record:
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    main()
