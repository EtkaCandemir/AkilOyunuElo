from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ao_elo.evaluation import (  # noqa: E402
    dependency_robust_loss_difference_ci,
    score_preserving_1x2_probabilities,
    standard_1x2_losses,
)


SOURCE = ROOT / "output" / "current_model_evaluation_2018_2026" / "model_predictions.csv"
OUTPUT = ROOT / "output" / "draw_shape_backtest_2018_2026"
MODEL = "CURRENT_PRODUCTION"
BASELINE_DRAW = 0.24
BASELINE_SHAPE = 1.0
SEASONS = (
    "2018/19",
    "2019/20",
    "2020/21",
    "2021/22",
    "2022/23",
    "2023/24",
    "2024/25",
    "2025/26",
)
DRAW_CANDIDATES = (0.20, 0.21, 0.22, 0.23, 0.235, 0.24, 0.245, 0.25, 0.26, 0.27, 0.28)
SHAPE_CANDIDATES = (0.50, 0.60, 0.70, 0.75, 0.80, 0.84, 0.85, 0.90, 0.95, 1.00, 1.10, 1.25, 1.50)


@dataclass(frozen=True, order=True)
class DrawCandidate:
    draw_at_even: float
    draw_shape: float

    @property
    def key(self) -> str:
        return f"draw_{self.draw_at_even:g}_shape_{self.draw_shape:g}"


def candidate_grid() -> tuple[DrawCandidate, ...]:
    return tuple(
        DrawCandidate(float(draw), float(shape))
        for draw, shape in product(DRAW_CANDIDATES, SHAPE_CANDIDATES)
    )


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}; run scripts/run_current_model_evaluation.py first"
        )
    data = pd.read_csv(path)
    data = data.loc[data["model"].eq(MODEL)].copy()
    required = {
        "match_id",
        "season",
        "kickoff_utc",
        "competition",
        "tie_id",
        "home_team_id",
        "away_team_id",
        "home_goals",
        "away_goals",
        "expected_home_score",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Prediction source is missing columns: {missing}")
    if len(data) != 6340 or data["match_id"].duplicated().any():
        raise ValueError("Expected 6,340 unique current-model predictions")
    if tuple(data["season"].drop_duplicates()) != SEASONS:
        raise ValueError("Prediction seasons are not in the frozen chronological order")
    data["kickoff_utc"] = pd.to_datetime(data["kickoff_utc"], utc=True)
    return data.sort_values(["kickoff_utc", "match_id"], kind="stable").reset_index(drop=True)


def evaluate_candidate(frame: pd.DataFrame, candidate: DrawCandidate) -> pd.DataFrame:
    probabilities = score_preserving_1x2_probabilities(
        frame["expected_home_score"],
        draw_at_even=candidate.draw_at_even,
        draw_shape=candidate.draw_shape,
    ).reset_index(drop=True)
    losses = standard_1x2_losses(
        probabilities,
        frame["home_goals"].reset_index(drop=True),
        frame["away_goals"].reset_index(drop=True),
    ).reset_index(drop=True)
    result = pd.concat([probabilities, losses[["outcome_1x2", "brier_1x2", "log_loss_1x2"]]], axis=1)
    result["predicted_outcome"] = np.array(("H", "D", "A"))[np.argmax(
        result[["home_probability", "draw_probability", "away_probability"]].to_numpy(),
        axis=1,
    )]
    expected = frame["expected_home_score"].to_numpy(float)
    raw_draw = candidate.draw_at_even * (4.0 * expected * (1.0 - expected)) ** candidate.draw_shape
    limit = 2.0 * np.minimum(expected, 1.0 - expected)
    result["draw_envelope_active"] = raw_draw > limit + 1e-15
    return result


def summarize(frame: pd.DataFrame, evaluated: pd.DataFrame) -> dict[str, float | int]:
    return {
        "matches": int(len(frame)),
        "brier_1x2": float(evaluated["brier_1x2"].mean()),
        "log_loss_1x2": float(evaluated["log_loss_1x2"].mean()),
        "accuracy_1x2": float(evaluated["predicted_outcome"].eq(evaluated["outcome_1x2"]).mean()),
        "mean_predicted_draw": float(evaluated["draw_probability"].mean()),
        "observed_draw_rate": float(evaluated["outcome_1x2"].eq("D").mean()),
        "draw_envelope_hits": int(evaluated["draw_envelope_active"].sum()),
    }


def select_candidate(
    train: pd.DataFrame,
    *,
    draw_candidates: tuple[float, ...] = DRAW_CANDIDATES,
) -> tuple[DrawCandidate, pd.DataFrame]:
    rows = []
    candidates = tuple(
        DrawCandidate(float(draw), float(shape))
        for draw, shape in product(draw_candidates, SHAPE_CANDIDATES)
    )
    for candidate in candidates:
        metrics = summarize(train, evaluate_candidate(train, candidate))
        rows.append(
            {
                "candidate_key": candidate.key,
                "draw_at_even": candidate.draw_at_even,
                "draw_shape": candidate.draw_shape,
                "distance_from_baseline": abs(candidate.draw_at_even - BASELINE_DRAW)
                + 0.02 * abs(candidate.draw_shape - BASELINE_SHAPE),
                **metrics,
            }
        )
    surface = pd.DataFrame(rows).sort_values(
        ["log_loss_1x2", "brier_1x2", "distance_from_baseline", "draw_at_even", "draw_shape"],
        kind="stable",
    )
    winner = surface.iloc[0]
    return DrawCandidate(float(winner.draw_at_even), float(winner.draw_shape)), surface.reset_index(drop=True)


def continuous_fit(frame: pd.DataFrame) -> dict[str, object]:
    def objective(parameters: np.ndarray) -> float:
        candidate = DrawCandidate(float(parameters[0]), float(parameters[1]))
        return float(evaluate_candidate(frame, candidate)["log_loss_1x2"].mean())

    fitted = minimize(
        objective,
        x0=np.array([BASELINE_DRAW, 0.84]),
        method="L-BFGS-B",
        bounds=((0.05, 0.50), (0.05, 3.00)),
    )
    candidate = DrawCandidate(float(fitted.x[0]), float(fitted.x[1]))
    metrics = summarize(frame, evaluate_candidate(frame, candidate))
    return {
        "converged": bool(fitted.success),
        "message": str(fitted.message),
        "draw_at_even": candidate.draw_at_even,
        "draw_shape": candidate.draw_shape,
        **metrics,
    }


def run_walk_forward(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selection_rows = []
    result_rows = []
    prediction_frames = []
    baseline = DrawCandidate(BASELINE_DRAW, BASELINE_SHAPE)
    for fold, test_index in enumerate(range(2, len(SEASONS)), start=1):
        train_seasons = SEASONS[:test_index]
        test_season = SEASONS[test_index]
        train = data.loc[data["season"].isin(train_seasons)].reset_index(drop=True)
        test = data.loc[data["season"].eq(test_season)].reset_index(drop=True)
        selected, surface = select_candidate(
            train,
            draw_candidates=(BASELINE_DRAW,),
        )
        joint_selected, joint_surface = select_candidate(train)
        for selection_mode, candidate, candidate_surface in (
            ("FIXED_DRAW_SHAPE_ONLY", selected, surface),
            ("JOINT_DRAW_AND_SHAPE", joint_selected, joint_surface),
        ):
            selected_train = candidate_surface.iloc[0]
            selection_rows.append(
                {
                    "fold": fold,
                    "selection_mode": selection_mode,
                    "train_seasons": "|".join(train_seasons),
                    "test_season": test_season,
                    "train_matches": len(train),
                    "draw_at_even": candidate.draw_at_even,
                    "draw_shape": candidate.draw_shape,
                    "train_brier_1x2": selected_train.brier_1x2,
                    "train_log_loss_1x2": selected_train.log_loss_1x2,
                }
            )
        evaluated_by_arm = {
            "BASELINE_024_SHAPE_100": evaluate_candidate(test, baseline),
            "WALK_FORWARD_FIXED_DRAW_SUBUNIT_SHAPE": evaluate_candidate(test, selected),
            "WALK_FORWARD_JOINT_DRAW_SHAPE": evaluate_candidate(test, joint_selected),
            "PRESPECIFIED_024_SHAPE_084": evaluate_candidate(
                test,
                DrawCandidate(BASELINE_DRAW, 0.84),
            ),
        }
        for arm, evaluated in evaluated_by_arm.items():
            result_rows.append(
                {
                    "fold": fold,
                    "test_season": test_season,
                    "model_arm": arm,
                    "draw_at_even": (
                        baseline.draw_at_even
                        if arm.startswith("BASELINE")
                        else joint_selected.draw_at_even
                        if arm.startswith("WALK_FORWARD_JOINT")
                        else BASELINE_DRAW
                    ),
                    "draw_shape": (
                        baseline.draw_shape
                        if arm.startswith("BASELINE")
                        else joint_selected.draw_shape
                        if arm.startswith("WALK_FORWARD_JOINT")
                        else 0.84
                        if arm.startswith("PRESPECIFIED")
                        else selected.draw_shape
                    ),
                    **summarize(test, evaluated),
                }
            )
        joined = test.reset_index(drop=True).copy()
        for prefix, evaluated in (
            ("baseline", evaluated_by_arm["BASELINE_024_SHAPE_100"]),
            ("candidate", evaluated_by_arm["WALK_FORWARD_FIXED_DRAW_SUBUNIT_SHAPE"]),
            ("joint_candidate", evaluated_by_arm["WALK_FORWARD_JOINT_DRAW_SHAPE"]),
            ("prespecified_084", evaluated_by_arm["PRESPECIFIED_024_SHAPE_084"]),
        ):
            for column in (
                "home_probability",
                "draw_probability",
                "away_probability",
                "brier_1x2",
                "log_loss_1x2",
                "predicted_outcome",
                "draw_envelope_active",
            ):
                joined[f"{prefix}_{column}"] = evaluated[column].to_numpy()
        joined["fold"] = fold
        joined["candidate_draw_at_even"] = selected.draw_at_even
        joined["candidate_draw_shape"] = selected.draw_shape
        joined["joint_candidate_draw_at_even"] = joint_selected.draw_at_even
        joined["joint_candidate_draw_shape"] = joint_selected.draw_shape
        prediction_frames.append(joined)
    return pd.DataFrame(selection_rows), pd.DataFrame(result_rows), pd.concat(prediction_frames, ignore_index=True)


def competition_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for competition, frame in [("ALL", predictions), *predictions.groupby("competition", sort=True)]:
        for arm in ("baseline", "candidate"):
            rows.append(
                {
                    "competition": competition,
                    "model_arm": arm.upper(),
                    "matches": len(frame),
                    "brier_1x2": float(frame[f"{arm}_brier_1x2"].mean()),
                    "log_loss_1x2": float(frame[f"{arm}_log_loss_1x2"].mean()),
                    "accuracy_1x2": float(frame[f"{arm}_predicted_outcome"].eq(np.where(frame.home_goals > frame.away_goals, "H", np.where(frame.home_goals == frame.away_goals, "D", "A"))).mean()),
                    "mean_predicted_draw": float(frame[f"{arm}_draw_probability"].mean()),
                    "observed_draw_rate": float((frame.home_goals == frame.away_goals).mean()),
                }
            )
    result = pd.DataFrame(rows)
    baseline = result.loc[result.model_arm.eq("BASELINE")].set_index("competition")
    for metric in ("brier_1x2", "log_loss_1x2", "accuracy_1x2"):
        result[f"delta_vs_baseline_{metric}"] = result.apply(
            lambda row: float(row[metric] - baseline.loc[row.competition, metric]),
            axis=1,
        )
    return result


def arm_summary(fold_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm, frame in fold_results.groupby("model_arm", sort=False):
        weights = frame["matches"].to_numpy(float)
        rows.append(
            {
                "model_arm": arm,
                "matches": int(weights.sum()),
                "brier_1x2": float(np.average(frame["brier_1x2"], weights=weights)),
                "log_loss_1x2": float(np.average(frame["log_loss_1x2"], weights=weights)),
                "accuracy_1x2": float(np.average(frame["accuracy_1x2"], weights=weights)),
                "mean_predicted_draw": float(np.average(frame["mean_predicted_draw"], weights=weights)),
                "observed_draw_rate": float(np.average(frame["observed_draw_rate"], weights=weights)),
            }
        )
    result = pd.DataFrame(rows)
    baseline = result.loc[result.model_arm.eq("BASELINE_024_SHAPE_100")].iloc[0]
    for metric in ("brier_1x2", "log_loss_1x2", "accuracy_1x2"):
        result[f"delta_vs_baseline_{metric}"] = result[metric] - float(baseline[metric])
    return result


def calibration_deciles(predictions: pd.DataFrame) -> pd.DataFrame:
    data = predictions.copy()
    data["expected_score_decile"] = pd.qcut(
        data["expected_home_score"],
        10,
        labels=False,
        duplicates="drop",
    ) + 1
    data["actual_draw"] = data["home_goals"].eq(data["away_goals"]).astype(float)
    return data.groupby("expected_score_decile", as_index=False).agg(
        matches=("match_id", "size"),
        mean_expected_home_score=("expected_home_score", "mean"),
        observed_draw_rate=("actual_draw", "mean"),
        baseline_predicted_draw=("baseline_draw_probability", "mean"),
        candidate_predicted_draw=("candidate_draw_probability", "mean"),
    ).assign(
        baseline_error=lambda frame: frame.baseline_predicted_draw - frame.observed_draw_rate,
        candidate_error=lambda frame: frame.candidate_predicted_draw - frame.observed_draw_rate,
    )


def uncertainty(predictions: pd.DataFrame, bootstrap_samples: int) -> pd.DataFrame:
    rows = []
    for metric in ("brier_1x2", "log_loss_1x2"):
        source = predictions.copy()
        source["loss_difference"] = source[f"candidate_{metric}"] - source[f"baseline_{metric}"]
        result = dependency_robust_loss_difference_ci(
            source,
            bootstrap_samples=bootstrap_samples,
        )
        result.insert(0, "metric", metric)
        rows.append(result)
    return pd.concat(rows, ignore_index=True)


def decision_summary(
    folds: pd.DataFrame,
    competitions: pd.DataFrame,
    uncertainty_frame: pd.DataFrame,
) -> tuple[str, dict[str, object]]:
    pivot = folds.pivot(index="fold", columns="model_arm", values=["brier_1x2", "log_loss_1x2"])
    candidate_arm = "WALK_FORWARD_FIXED_DRAW_SUBUNIT_SHAPE"
    brier_wins = int((pivot[("brier_1x2", candidate_arm)] < pivot[("brier_1x2", "BASELINE_024_SHAPE_100")]).sum())
    log_wins = int((pivot[("log_loss_1x2", candidate_arm)] < pivot[("log_loss_1x2", "BASELINE_024_SHAPE_100")]).sum())
    candidate_comp = competitions.loc[competitions.model_arm.eq("CANDIDATE")]
    no_segment_harm = bool(
        candidate_comp["delta_vs_baseline_brier_1x2"].le(0.0).all()
        and candidate_comp["delta_vs_baseline_log_loss_1x2"].le(0.0).all()
    )
    envelope = uncertainty_frame.loc[uncertainty_frame.method.eq("conservative_envelope")]
    reliable = bool(envelope["reliable_improvement"].all())
    passed = brier_wins >= 4 and log_wins >= 4 and no_segment_harm and reliable
    return (
        "PROMOTE_CANDIDATE" if passed else "KEEP_SHADOW",
        {
            "brier_fold_wins": f"{brier_wins}/6",
            "log_loss_fold_wins": f"{log_wins}/6",
            "no_competition_harm": no_segment_harm,
            "dependency_reliable_improvement": reliable,
        },
    )


def markdown_table(frame: pd.DataFrame, digits: int = 7) -> str:
    shown = frame.copy()
    for column in shown.select_dtypes(include=["float"]).columns:
        shown[column] = shown[column].map(lambda value: f"{value:.{digits}f}")
    headers = list(shown.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in shown.itertuples(index=False, name=None))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest sub-unit score-preserving draw shapes")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    args = parser.parse_args()

    data = load_predictions(args.source.resolve())
    selections, fold_results, unseen = run_walk_forward(data)
    full_selected, fixed_surface = select_candidate(
        data,
        draw_candidates=(BASELINE_DRAW,),
    )
    full_joint_selected, surface = select_candidate(data)
    continuous = continuous_fit(data)
    competitions = competition_summary(unseen)
    arms = arm_summary(fold_results)
    calibration = calibration_deciles(unseen)
    uncertainty_frame = uncertainty(unseen, args.bootstrap_samples)
    decision, gates = decision_summary(fold_results, competitions, uncertainty_frame)

    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    selections.to_csv(output / "fold_selections.csv", index=False)
    fold_results.to_csv(output / "fold_results.csv", index=False)
    unseen.to_csv(output / "unseen_predictions.csv", index=False)
    competitions.to_csv(output / "competition_summary.csv", index=False)
    arms.to_csv(output / "arm_summary.csv", index=False)
    calibration.to_csv(output / "calibration_deciles.csv", index=False)
    uncertainty_frame.to_csv(output / "dependency_uncertainty.csv", index=False)
    surface.to_csv(output / "candidate_surface.csv", index=False)
    fixed_surface.to_csv(output / "fixed_draw_shape_surface.csv", index=False)
    selected = {
        "decision": decision,
        "baseline": {"draw_at_even": BASELINE_DRAW, "draw_shape": BASELINE_SHAPE},
        "full_grid_candidate": {
            "draw_at_even": full_selected.draw_at_even,
            "draw_shape": full_selected.draw_shape,
        },
        "full_joint_grid_candidate": {
            "draw_at_even": full_joint_selected.draw_at_even,
            "draw_shape": full_joint_selected.draw_shape,
        },
        "continuous_full_fit": continuous,
        "gates": gates,
        "rating_state_changed": False,
        "score_preserving_tail_envelope": "min(raw_draw, 2*min(E,1-E))",
    }
    (output / "selected_candidate.json").write_text(
        json.dumps(selected, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = "\n".join(
        [
            "# Draw Shape < 1 Walk-Forward Backtest",
            "",
            f"Decision: **{decision}**",
            "",
            f"- Source matches: `{len(data)}`; unseen matches: `{len(unseen)}`.",
            f"- Full grid candidate: draw-at-even `{full_selected.draw_at_even:g}`, shape `{full_selected.draw_shape:g}`.",
            f"- Full joint-grid candidate: draw-at-even `{full_joint_selected.draw_at_even:g}`, shape `{full_joint_selected.draw_shape:g}`.",
            f"- Continuous diagnostic fit: draw-at-even `{continuous['draw_at_even']:.6f}`, shape `{continuous['draw_shape']:.6f}`.",
            f"- Fold gates: `{gates}`.",
            "- This layer changes only the 1X2 decomposition. Expected score, Power Elo, AO Live Elo and ranking remain unchanged.",
            "- Shapes below one use a score-preserving tail envelope so no negative win probability can be produced.",
            "",
            "## Fold selections",
            "",
            markdown_table(selections),
            "",
            "## Fold results",
            "",
            markdown_table(fold_results),
            "",
            "## Pooled arm summary",
            "",
            markdown_table(arms),
            "",
            "## Competition summary",
            "",
            markdown_table(competitions),
            "",
            "## Expected-score decile calibration",
            "",
            markdown_table(calibration),
            "",
            "## Dependency-aware uncertainty",
            "",
            markdown_table(uncertainty_frame),
            "",
        ]
    )
    (output / "backtest_report.md").write_text(report, encoding="utf-8")
    print(f"Decision: {decision}")
    print(f"Full grid candidate: {full_selected}")
    print(f"Full joint-grid candidate: {full_joint_selected}")
    print(f"Continuous fit: {continuous['draw_at_even']:.6f}/{continuous['draw_shape']:.6f}")
    print(f"Report: {output / 'backtest_report.md'}")


if __name__ == "__main__":
    main()
