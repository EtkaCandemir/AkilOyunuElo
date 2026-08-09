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
from ao_elo.xg_live import (  # noqa: E402
    UnsupportedMarginConfig,
    XGBlendConfig,
    update_match_elo_with_xg,
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
    elo_field_score,
    load_initial_ratings,
    read_production_contract,
    read_static_config,
    same_season_ranking,
)


OUTPUT_ROOT = ROOT / "output" / "unsupported_margin_backtest_2025_26"
VALIDATION_START = pd.Timestamp("2026-01-01T00:00:00Z")
BASELINE_KEY = "GD_PRODUCTION"
LEGACY_KEY = "LEGACY_BOUNDED_RHO005"
TOLERANCE_GRID = (0.50, 0.75, 1.00)
LAMBDA_GRID = (0.05, 0.10, 0.15, 0.20)
MINIMUM_MULTIPLIER_GRID = (0.70, 0.80, 0.90)


@dataclass(frozen=True, order=True)
class UnsupportedMarginCandidate:
    key: str
    kind: str
    tolerance: float = 0.0
    penalty_lambda: float = 0.0
    minimum_multiplier: float = 1.0

    def validate(self) -> None:
        if self.kind not in {
            "PRODUCTION",
            "LEGACY_XG",
            "UNSUPPORTED_MARGIN",
            "GOAL_BONUS_GUARD",
        }:
            raise ValueError(f"Unknown candidate kind: {self.kind}")
        if self.kind in {"UNSUPPORTED_MARGIN", "GOAL_BONUS_GUARD"}:
            UnsupportedMarginConfig(
                self.tolerance,
                self.penalty_lambda,
                self.minimum_multiplier,
                (
                    "GOAL_BONUS_ONLY"
                    if self.kind == "GOAL_BONUS_GUARD"
                    else "FULL_UPDATE"
                ),
            ).validate()
            if self.penalty_lambda <= 0.0:
                raise ValueError("Unsupported-margin candidate requires lambda>0")
        elif (
            self.tolerance != 0.0
            or self.penalty_lambda != 0.0
            or self.minimum_multiplier != 1.0
        ):
            raise ValueError("Control candidates cannot carry margin parameters")

    @property
    def config(self) -> UnsupportedMarginConfig | None:
        if self.kind not in {"UNSUPPORTED_MARGIN", "GOAL_BONUS_GUARD"}:
            return None
        return UnsupportedMarginConfig(
            self.tolerance,
            self.penalty_lambda,
            self.minimum_multiplier,
            (
                "GOAL_BONUS_ONLY"
                if self.kind == "GOAL_BONUS_GUARD"
                else "FULL_UPDATE"
            ),
        )


@dataclass
class CandidateReplay:
    predictions: pd.DataFrame
    development_ratings: pd.DataFrame
    final_ratings: pd.DataFrame


def candidate_grid() -> tuple[UnsupportedMarginCandidate, ...]:
    candidates = [
        UnsupportedMarginCandidate(BASELINE_KEY, "PRODUCTION"),
        UnsupportedMarginCandidate(LEGACY_KEY, "LEGACY_XG"),
    ]
    candidates.extend(
        UnsupportedMarginCandidate(
            (
                f"UNSUPPORTED_tol{tolerance:g}_lambda{penalty_lambda:g}"
                f"_min{minimum_multiplier:g}"
            ),
            "UNSUPPORTED_MARGIN",
            tolerance,
            penalty_lambda,
            minimum_multiplier,
        )
        for tolerance in TOLERANCE_GRID
        for penalty_lambda in LAMBDA_GRID
        for minimum_multiplier in MINIMUM_MULTIPLIER_GRID
    )
    result = tuple(candidates)
    for candidate in result:
        candidate.validate()
    if len(result) != 38:
        raise ValueError(f"Expected 38 candidates, found {len(result)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test an xG unsupported-score-margin penalty over the frozen "
            "production goal-difference Elo model"
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
    production = read_production_contract(args.production_model.resolve())
    static_config = read_static_config(args.static_manifest.resolve())
    initial_ratings = load_initial_ratings(
        args.static_data_root.resolve(), events, static_config
    )[SEASON]
    candidates = candidate_grid()

    development_events = events.loc[events["kickoff_utc"].lt(VALIDATION_START)]
    validation_events = events.loc[events["kickoff_utc"].ge(VALIDATION_START)]
    if int(development_events["xg_analysis_eligible"].sum()) != 399:
        raise ValueError("Expected 399 development xG matches")
    if len(validation_events) != 207 or not validation_events[
        "xg_analysis_eligible"
    ].all():
        raise ValueError("Expected 207 fully xG-covered validation matches")

    print(
        f"Unsupported-margin grid: {len(candidates)} candidates; "
        f"development={len(development_events)} matches/399 xG, "
        "validation=207 xG matches",
        flush=True,
    )
    evaluations = {
        candidate.key: replay_candidate(
            events,
            initial_ratings,
            production,
            candidate,
            validation_start=VALIDATION_START,
        )
        for candidate in candidates
    }
    development_target = schedule_adjusted_team_performance(development_events)
    full_target = schedule_adjusted_team_performance(events)
    development_metrics = build_metric_table(
        evaluations,
        candidates,
        development_target,
        split="DEVELOPMENT",
    )
    selected, selection = select_development_candidate(development_metrics)
    validation_metrics = build_metric_table(
        evaluations,
        candidates,
        full_target,
        split="VALIDATION",
    )
    comparison_keys = (BASELINE_KEY, LEGACY_KEY, selected.key)
    comparison = pd.concat(
        [
            development_metrics.loc[
                development_metrics["candidate_key"].isin(comparison_keys)
            ],
            validation_metrics.loc[
                validation_metrics["candidate_key"].isin(comparison_keys)
            ],
        ],
        ignore_index=True,
    )
    selected_predictions = pd.concat(
        [
            evaluations[key].predictions.assign(comparison_role=key)
            for key in comparison_keys
        ],
        ignore_index=True,
    )
    selected_predictions["evaluation_split"] = np.where(
        selected_predictions["kickoff_utc"].lt(VALIDATION_START),
        "DEVELOPMENT",
        "VALIDATION",
    )
    competition = build_competition_summary(
        selected_predictions.loc[
            selected_predictions["evaluation_split"].eq("VALIDATION")
        ]
    )
    uncertainty = build_uncertainty(
        selected_predictions,
        selected.key,
        bootstrap_samples=args.bootstrap_samples,
    )
    penalty_examples = build_penalty_examples(
        evaluations[selected.key].predictions.loc[
            lambda frame: frame["kickoff_utc"].ge(VALIDATION_START)
        ]
    )
    decision, guardrails = make_decision(
        comparison,
        competition,
        uncertainty,
        selected,
        selection,
    )

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    development_metrics.to_csv(
        output_root / "development_candidate_metrics.csv", index=False
    )
    validation_metrics.to_csv(
        output_root / "validation_candidate_metrics.csv", index=False
    )
    comparison.to_csv(output_root / "selected_model_comparison.csv", index=False)
    selected_predictions.to_csv(
        output_root / "selected_match_updates.csv", index=False
    )
    competition.to_csv(output_root / "competition_summary.csv", index=False)
    uncertainty.to_csv(output_root / "dependency_uncertainty.csv", index=False)
    penalty_examples.to_csv(output_root / "penalty_examples.csv", index=False)
    pd.concat(
        [evaluations[key].final_ratings for key in comparison_keys],
        ignore_index=True,
    ).to_csv(output_root / "final_ratings.csv", index=False)

    payload = {
        "analysis": "UNSUPPORTED_MARGIN_2025_26_TEMPORAL_SPLIT",
        "formula": (
            "U=max(0,min(abs(GD),4)-winner_xg_advantage-tolerance); "
            "P=max(minimum_multiplier,1-lambda*ln(1+U)); "
            "Delta_final=Delta_GD*P"
        ),
        "development_end_exclusive": VALIDATION_START.isoformat(),
        "development_matches": len(development_events),
        "development_xg_matches": int(
            development_events["xg_analysis_eligible"].sum()
        ),
        "validation_matches": len(validation_events),
        "validation_xg_matches": int(validation_events["xg_analysis_eligible"].sum()),
        "selected_candidate": selected.__dict__,
        "selection": selection,
        "decision": decision,
        "guardrails": guardrails,
        "production_changed": False,
        "evidence_class": "ONE_SEASON_TEMPORAL_SHADOW_VALIDATION",
    }
    (output_root / "selected_unsupported_margin_model.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_root / "backtest_report.md").write_text(
        build_report(
            comparison,
            competition,
            uncertainty,
            penalty_examples,
            selected,
            selection,
            decision,
            guardrails,
        ),
        encoding="utf-8",
    )
    print(f"Selected: {selected.key}")
    print(f"Decision: {decision}")
    print(f"Report: {output_root / 'backtest_report.md'}")


def replay_candidate(
    events: pd.DataFrame,
    initial_ratings: dict[int, float],
    production: dict[str, object],
    candidate: UnsupportedMarginCandidate,
    *,
    validation_start: pd.Timestamp,
) -> CandidateReplay:
    candidate.validate()
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
    development_snapshot: dict[int, float] | None = None
    rows: list[dict[str, object]] = []

    for row in events.itertuples(index=False):
        if development_snapshot is None and row.kickoff_utc >= validation_start:
            development_snapshot = dict(power)
        home_id = int(row.home_team_id)
        away_id = int(row.away_team_id)
        penalties = bool(row.decided_on_penalties)
        home_goals, away_goals = elo_field_score(
            int(row.home_goals), int(row.away_goals), penalties
        )
        eligible = bool(row.xg_analysis_eligible)
        xg_home = float(row.xg_home) if eligible else None
        xg_away = float(row.xg_away) if eligible else None
        xg_config = (
            XGBlendConfig(0.05, 1.0, "DIRECTION_PRESERVING_LUCK_CORRECTION")
            if candidate.kind == "LEGACY_XG"
            else XGBlendConfig(0.0, 1.0)
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
            decided_on_penalties=penalties,
            goal_difference_enabled=True,
            goal_alpha=float(goal["alpha"]),
            goal_tau=float(goal["tau"]),
            goal_difference_cap=int(goal["goal_difference_cap"]),
            xg_config=xg_config,
            xg_home=xg_home,
            xg_away=xg_away,
            unsupported_margin_config=candidate.config,
        )
        probabilities = one_x_two_probabilities_scalar(
            update.expected_home_score,
            float(draw["draw_at_even"]),
            float(draw["draw_shape"]),
        )
        actual_score = float(row.actual_home_score)
        observed = 0 if actual_score == 1.0 else 1 if actual_score == 0.5 else 2
        target = tuple(1.0 if index == observed else 0.0 for index in range(3))
        brier = sum(
            (probability - actual) ** 2
            for probability, actual in zip(probabilities, target)
        )
        log_loss = -math.log(max(probabilities[observed], 1e-15))
        power[home_id] = update.home_rating_post
        power[away_id] = update.away_rating_post
        rows.append(
            {
                "candidate_key": candidate.key,
                "candidate_kind": candidate.kind,
                "tolerance": candidate.tolerance,
                "penalty_lambda": candidate.penalty_lambda,
                "minimum_multiplier": candidate.minimum_multiplier,
                "match_id": str(row.match_id),
                "season": str(row.season),
                "competition": str(row.competition),
                "round": str(row.round),
                "kickoff_utc": row.kickoff_utc,
                "tie_id": None if pd.isna(row.tie_id) else str(row.tie_id),
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
                "winner_xg_advantage": update.winner_xg_advantage,
                "unsupported_margin": update.unsupported_margin,
                "xg_penalty_multiplier": update.xg_penalty_multiplier,
                "home_rating_pre": update.home_rating_pre,
                "away_rating_pre": update.away_rating_pre,
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
                "result_residual": update.result_residual,
                "base_result_residual": update.base_result_residual,
                "goal_bonus_residual": update.goal_bonus_residual,
                "adjusted_goal_bonus_residual": (
                    update.adjusted_goal_bonus_residual
                ),
                "blended_residual": update.blended_residual,
                "power_delta": update.power_delta,
                "home_rating_post": update.home_rating_post,
                "away_rating_post": update.away_rating_post,
                "zero_sum_error": update.zero_sum_error,
            }
        )
    if development_snapshot is None:
        raise ValueError("Validation cutoff is outside the event range")

    names = dict(
        zip(events["home_team_id"].astype(int), events["home_team_name"].astype(str))
    )
    names.update(
        zip(events["away_team_id"].astype(int), events["away_team_name"].astype(str))
    )

    def rating_frame(values: dict[int, float]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "candidate_key": candidate.key,
                    "season": SEASON,
                    "team_id": team_id,
                    "team_name": names.get(team_id, str(team_id)),
                    "start_rating": start[team_id],
                    "end_live_rating": values[team_id],
                    "rating_change": values[team_id] - start[team_id],
                }
                for team_id in active_ids
            ]
        )

    return CandidateReplay(
        predictions=pd.DataFrame(rows),
        development_ratings=rating_frame(development_snapshot),
        final_ratings=rating_frame(power),
    )


def build_metric_table(
    evaluations: dict[str, CandidateReplay],
    candidates: tuple[UnsupportedMarginCandidate, ...],
    ranking_target: pd.DataFrame,
    *,
    split: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    is_development = split == "DEVELOPMENT"
    for candidate in candidates:
        evaluation = evaluations[candidate.key]
        predictions = evaluation.predictions.loc[
            evaluation.predictions["kickoff_utc"].lt(VALIDATION_START)
            if is_development
            else evaluation.predictions["kickoff_utc"].ge(VALIDATION_START)
        ]
        ratings = (
            evaluation.development_ratings
            if is_development
            else evaluation.final_ratings
        )
        ranking = same_season_ranking(ratings, ranking_target, {SEASON})
        pooled = ranking.loc[ranking["competition"].eq("ALL")].iloc[0]
        penalized = predictions["xg_penalty_multiplier"].lt(1.0 - 1e-15)
        rows.append(
            {
                "split": split,
                "candidate_key": candidate.key,
                "candidate_kind": candidate.kind,
                "tolerance": candidate.tolerance,
                "penalty_lambda": candidate.penalty_lambda,
                "minimum_multiplier": candidate.minimum_multiplier,
                **evaluate_predictions(predictions),
                "ranking_score": float(pooled["ranking_score"]),
                "pairwise_accuracy": float(pooled["pairwise_accuracy"]),
                "penalized_matches": int(penalized.sum()),
                "penalized_rate": float(penalized.mean()),
                "mean_penalty_multiplier": float(
                    predictions.loc[penalized, "xg_penalty_multiplier"].mean()
                    if penalized.any()
                    else 1.0
                ),
                "minimum_observed_multiplier": float(
                    predictions["xg_penalty_multiplier"].min()
                ),
                "max_abs_match_delta": float(predictions["power_delta"].abs().max()),
                "maximum_zero_sum_error": float(predictions["zero_sum_error"].max()),
            }
        )
    result = pd.DataFrame(rows)
    baseline = result.loc[result["candidate_key"].eq(BASELINE_KEY)].iloc[0]
    for metric in (
        "brier_1x2",
        "log_loss_1x2",
        "accuracy_1x2",
        "multiclass_ece",
        "ranking_score",
        "pairwise_accuracy",
    ):
        result[f"{metric}_delta_vs_production"] = (
            result[metric] - float(baseline[metric])
        )
    return result


def select_development_candidate(
    development: pd.DataFrame,
    *,
    candidate_kind: str = "UNSUPPORTED_MARGIN",
) -> tuple[UnsupportedMarginCandidate, dict[str, object]]:
    baseline = development.loc[development["candidate_key"].eq(BASELINE_KEY)].iloc[0]
    candidates = development.loc[
        development["candidate_kind"].eq(candidate_kind)
    ].copy()
    candidates["ranking_safe"] = (
        candidates["ranking_score"].ge(float(baseline["ranking_score"]))
        & candidates["pairwise_accuracy"].ge(float(baseline["pairwise_accuracy"]))
    )
    ranking_safe = candidates.loc[candidates["ranking_safe"]].copy()
    pool = ranking_safe if not ranking_safe.empty else candidates
    selected_row = pool.sort_values(
        [
            "brier_1x2",
            "log_loss_1x2",
            "ranking_score",
            "pairwise_accuracy",
            "penalty_lambda",
            "tolerance",
            "minimum_multiplier",
        ],
        ascending=[True, True, False, False, True, False, False],
    ).iloc[0]
    selected = UnsupportedMarginCandidate(
        str(selected_row["candidate_key"]),
        candidate_kind,
        float(selected_row["tolerance"]),
        float(selected_row["penalty_lambda"]),
        float(selected_row["minimum_multiplier"]),
    )
    return selected, {
        "ranking_safe_candidates": int(len(ranking_safe)),
        "selection_pool": "RANKING_SAFE" if len(ranking_safe) else "LOSS_ONLY_FALLBACK",
        "selected_development_ranking_safe": bool(selected_row["ranking_safe"]),
        "validation_metrics_used_for_selection": False,
    }


def build_competition_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (candidate, competition), frame in predictions.groupby(
        ["candidate_key", "competition"], sort=True
    ):
        rows.append(
            {
                "candidate_key": candidate,
                "competition": competition,
                **evaluate_predictions(frame),
            }
        )
    result = pd.DataFrame(rows)
    baseline = result.loc[result["candidate_key"].eq(BASELINE_KEY)].set_index(
        "competition"
    )
    result["brier_delta_vs_production"] = result.apply(
        lambda row: row["brier_1x2"]
        - baseline.loc[row["competition"], "brier_1x2"],
        axis=1,
    )
    result["log_loss_delta_vs_production"] = result.apply(
        lambda row: row["log_loss_1x2"]
        - baseline.loc[row["competition"], "log_loss_1x2"],
        axis=1,
    )
    return result


def build_uncertainty(
    predictions: pd.DataFrame,
    selected_key: str,
    *,
    bootstrap_samples: int,
) -> pd.DataFrame:
    validation = predictions.loc[
        predictions["evaluation_split"].eq("VALIDATION")
    ]
    baseline = validation.loc[
        validation["candidate_key"].eq(BASELINE_KEY)
    ].set_index("match_id")
    rows: list[pd.DataFrame] = []
    for comparison_index, candidate_key in enumerate((LEGACY_KEY, selected_key)):
        candidate = validation.loc[
            validation["candidate_key"].eq(candidate_key)
        ].set_index("match_id")
        if not candidate.index.equals(baseline.index):
            raise ValueError("Validation uncertainty requires paired matches")
        for metric_index, metric in enumerate(("brier_1x2", "log_loss_1x2")):
            paired = candidate[
                ["season", "home_team_id", "away_team_id", "kickoff_utc", "tie_id"]
            ].reset_index()
            paired["loss_difference"] = (
                candidate[metric].to_numpy(float) - baseline[metric].to_numpy(float)
            )
            uncertainty = dependency_robust_loss_difference_ci(
                paired,
                bootstrap_samples=bootstrap_samples,
                seed=20260805 + comparison_index * 1009 + metric_index * 10007,
            )
            uncertainty.insert(0, "candidate_key", candidate_key)
            uncertainty.insert(1, "baseline_key", BASELINE_KEY)
            uncertainty.insert(2, "metric", metric)
            rows.append(uncertainty)
    return pd.concat(rows, ignore_index=True)


def build_penalty_examples(predictions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "match_id",
        "competition",
        "round",
        "kickoff_utc",
        "home_team_name",
        "away_team_name",
        "home_goals",
        "away_goals",
        "xg_home",
        "xg_away",
        "winner_xg_advantage",
        "unsupported_margin",
        "xg_penalty_multiplier",
        "result_residual",
        "blended_residual",
        "power_delta",
    ]
    return predictions.loc[
        predictions["xg_penalty_multiplier"].lt(1.0 - 1e-15), columns
    ].sort_values(
        ["xg_penalty_multiplier", "unsupported_margin"],
        ascending=[True, False],
    ).reset_index(drop=True)


def make_decision(
    comparison: pd.DataFrame,
    competition: pd.DataFrame,
    uncertainty: pd.DataFrame,
    selected: UnsupportedMarginCandidate,
    selection: dict[str, object],
) -> tuple[str, dict[str, object]]:
    validation = comparison.loc[comparison["split"].eq("VALIDATION")].set_index(
        "candidate_key"
    )
    row = validation.loc[selected.key]
    segments = competition.loc[competition["candidate_key"].eq(selected.key)]
    ci = uncertainty.loc[
        uncertainty["candidate_key"].eq(selected.key)
        & uncertainty["method"].eq("conservative_envelope")
    ]
    point_loss_improvement = bool(
        row["brier_1x2_delta_vs_production"] < 0.0
        and row["log_loss_1x2_delta_vs_production"] < 0.0
    )
    no_competition_regression = bool(
        len(segments) == 3
        and (segments["brier_delta_vs_production"] <= 0.0).all()
        and (segments["log_loss_delta_vs_production"] <= 0.0).all()
    )
    no_reliable_harm = bool(len(ci) == 2 and not (ci["ci_95_lower"] > 0.0).any())
    reliable_improvement = bool(
        len(ci) == 2 and (ci["ci_95_upper"] < 0.0).all()
    )
    ranking_guardrail = bool(
        row["ranking_score_delta_vs_production"] >= 0.0
        and row["pairwise_accuracy_delta_vs_production"] >= 0.0
    )
    guardrails = {
        "selected_on_development_only": True,
        "development_ranking_safe": bool(
            selection["selected_development_ranking_safe"]
        ),
        "validation_brier_delta": float(row["brier_1x2_delta_vs_production"]),
        "validation_log_loss_delta": float(
            row["log_loss_1x2_delta_vs_production"]
        ),
        "validation_ranking_delta": float(
            row["ranking_score_delta_vs_production"]
        ),
        "validation_pairwise_delta": float(
            row["pairwise_accuracy_delta_vs_production"]
        ),
        "point_loss_improvement": point_loss_improvement,
        "no_competition_regression": no_competition_regression,
        "no_cluster_reliable_harm": no_reliable_harm,
        "cluster_reliable_improvement": reliable_improvement,
        "ranking_guardrail": ranking_guardrail,
        "zero_sum_preserved": bool(row["maximum_zero_sum_error"] <= 1e-9),
    }
    passes = all(
        (
            guardrails["development_ranking_safe"],
            point_loss_improvement,
            no_competition_regression,
            no_reliable_harm,
            ranking_guardrail,
            guardrails["zero_sum_preserved"],
        )
    )
    return (
        "PROMISING_SHADOW_NOT_PRODUCTION"
        if passes
        else "MIXED_OR_INCONCLUSIVE_SHADOW"
    ), guardrails


def build_report(
    comparison: pd.DataFrame,
    competition: pd.DataFrame,
    uncertainty: pd.DataFrame,
    penalty_examples: pd.DataFrame,
    selected: UnsupportedMarginCandidate,
    selection: dict[str, object],
    decision: str,
    guardrails: dict[str, object],
) -> str:
    comparison_view = comparison[
        [
            "split",
            "candidate_key",
            "matches",
            "brier_1x2",
            "brier_1x2_delta_vs_production",
            "log_loss_1x2",
            "log_loss_1x2_delta_vs_production",
            "accuracy_1x2",
            "ranking_score",
            "ranking_score_delta_vs_production",
            "pairwise_accuracy",
            "pairwise_accuracy_delta_vs_production",
            "penalized_matches",
            "mean_penalty_multiplier",
        ]
    ]
    competition_view = competition[
        [
            "candidate_key",
            "competition",
            "matches",
            "brier_delta_vs_production",
            "log_loss_delta_vs_production",
        ]
    ]
    ci_view = uncertainty.loc[
        uncertainty["method"].eq("conservative_envelope"),
        [
            "candidate_key",
            "metric",
            "mean_difference",
            "ci_95_lower",
            "ci_95_upper",
        ],
    ]
    examples = penalty_examples.head(15)
    return "\n".join(
        [
            "# xG Unsupported Margin 2025/26 Backtesti",
            "",
            "## Formül",
            "",
            "```text",
            "winner_xg_advantage = result_direction * (xG_home - xG_away)",
            "U = max(0, min(abs(goal_difference), 4) - winner_xg_advantage - tolerance)",
            "P = max(minimum_multiplier, 1 - lambda * ln(1 + U))",
            "Delta_final = Delta_GD * P",
            "```",
            "",
            "Beraberlik, penaltı kararı veya eksik xG durumunda `P=1`. Desteklenen",
            "galibiyet ceza almaz; update yönü değişmez ve Elo sıfır-toplamlıdır.",
            "",
            "## Temporal Tasarım",
            "",
            "- Development: 1 Ocak 2026 öncesi 754 maç, 399 xG.",
            "- Validation: 1 Ocak 2026 ve sonrası 207 maç, tamamında xG.",
            "- Katsayı yalnız development bölümünden seçildi.",
            "- Bu tek sezon shadow doğrulamasıdır; production terfisi değildir.",
            "",
            "## Seçilen Aday",
            "",
            f"`{selected.key}`",
            "",
            markdown_table(pd.DataFrame([selection])),
            "",
            "## Ana Karşılaştırma",
            "",
            markdown_table(comparison_view),
            "",
            "## Validation Turnuva Segmentleri",
            "",
            markdown_table(competition_view),
            "",
            "## Cluster Güven Aralıkları",
            "",
            markdown_table(ci_view),
            "",
            "## En Güçlü Ceza Örnekleri",
            "",
            markdown_table(examples),
            "",
            "## Model Kararı",
            "",
            f"`{decision}`",
            "",
            markdown_table(pd.DataFrame([guardrails])),
        ]
    )


if __name__ == "__main__":
    main()
