from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ao_elo.evaluation import (  # noqa: E402
    dependency_robust_loss_difference_ci,
    score_preserving_1x2_probabilities,
    standard_1x2_losses,
)


PREDICTIONS = ROOT / "output" / "current_model_evaluation_2018_2026" / "model_predictions.csv"
EVENTS = ROOT / "data" / "external_elo_benchmark_2018_2026" / "exact_date_events.csv"
OUTPUT = ROOT / "output" / "single_match_draw_backtest_2018_2026"
MODEL = "CURRENT_PRODUCTION"
BASELINE_DRAW = 0.24
BASELINE_SHAPE = 1.0
PRESPECIFIED_SINGLE_DRAW = 0.12
SINGLE_DRAW_CANDIDATES = tuple(float(value) for value in np.arange(0.04, 0.241, 0.01).round(2))
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


def load_data(prediction_path: Path, events_path: Path) -> pd.DataFrame:
    predictions = pd.read_csv(prediction_path)
    predictions = predictions.loc[predictions["model"].eq(MODEL)].copy()
    events = pd.read_csv(events_path)
    tie_counts = (
        events.loc[events["is_knockout"].astype(bool)]
        .groupby(["season", "competition", "tie_id"], sort=False)
        .size()
        .rename("tie_match_count")
        .reset_index()
    )
    event_format = events[
        ["match_id", "season", "competition", "tie_id", "is_knockout"]
    ].merge(
        tie_counts,
        on=["season", "competition", "tie_id"],
        how="left",
        validate="many_to_one",
    )
    data = predictions.merge(
        event_format,
        on=["match_id", "season", "competition", "tie_id"],
        suffixes=("", "_event"),
        validate="one_to_one",
    )
    if len(data) != 6340 or data["match_id"].duplicated().any():
        raise ValueError("Expected 6,340 unique current-model predictions")
    invalid_ties = data.loc[
        data["is_knockout"].astype(bool)
        & ~data["tie_match_count"].isin((1.0, 2.0))
    ]
    if not invalid_ties.empty:
        raise ValueError("Knockout ties must contain one or two field-score matches")
    data["is_single_match_tie"] = (
        data["is_knockout"].astype(bool) & data["tie_match_count"].eq(1.0)
    )
    data["is_two_legged_tie"] = (
        data["is_knockout"].astype(bool) & data["tie_match_count"].eq(2.0)
    )
    data["kickoff_utc"] = pd.to_datetime(data["kickoff_utc"], utc=True)
    return data.sort_values(["kickoff_utc", "match_id"], kind="stable").reset_index(drop=True)


def evaluate_format_model(
    frame: pd.DataFrame,
    single_match_draw_at_even: float,
) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    result["effective_draw_at_even"] = np.where(
        frame["is_single_match_tie"],
        single_match_draw_at_even,
        BASELINE_DRAW,
    )
    blocks = []
    for draw_at_even, indexes in result.groupby("effective_draw_at_even").groups.items():
        block = frame.loc[indexes]
        probabilities = score_preserving_1x2_probabilities(
            block["expected_home_score"],
            draw_at_even=float(draw_at_even),
            draw_shape=BASELINE_SHAPE,
        )
        losses = standard_1x2_losses(
            probabilities,
            block["home_goals"],
            block["away_goals"],
        )
        evaluated = probabilities.join(
            losses[["outcome_1x2", "brier_1x2", "log_loss_1x2"]]
        )
        blocks.append(evaluated)
    evaluated = pd.concat(blocks).sort_index()
    for column in evaluated.columns:
        result[column] = evaluated[column]
    probability_columns = ["home_probability", "draw_probability", "away_probability"]
    result["predicted_outcome"] = np.array(("H", "D", "A"))[
        np.argmax(result[probability_columns].to_numpy(float), axis=1)
    ]
    if result[probability_columns].isna().any().any():
        raise ValueError("Format-aware probabilities contain missing values")
    return result


def summarize(frame: pd.DataFrame, evaluated: pd.DataFrame) -> dict[str, float | int]:
    return {
        "matches": int(len(frame)),
        "single_match_ties": int(frame["is_single_match_tie"].sum()),
        "brier_1x2": float(evaluated["brier_1x2"].mean()),
        "log_loss_1x2": float(evaluated["log_loss_1x2"].mean()),
        "accuracy_1x2": float(evaluated["predicted_outcome"].eq(evaluated["outcome_1x2"]).mean()),
        "predicted_draw_rate": float(evaluated["draw_probability"].mean()),
        "observed_draw_rate": float(evaluated["outcome_1x2"].eq("D").mean()),
    }


def select_single_draw(train: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    rows = []
    for candidate in SINGLE_DRAW_CANDIDATES:
        evaluated = evaluate_format_model(train, candidate)
        metrics = summarize(train, evaluated)
        single = train["is_single_match_tie"]
        rows.append(
            {
                "single_match_draw_at_even": candidate,
                "distance_from_baseline": abs(candidate - BASELINE_DRAW),
                "training_single_matches": int(single.sum()),
                **metrics,
            }
        )
    surface = pd.DataFrame(rows).sort_values(
        ["log_loss_1x2", "brier_1x2", "distance_from_baseline"],
        kind="stable",
    )
    return float(surface.iloc[0]["single_match_draw_at_even"]), surface.reset_index(drop=True)


def continuous_single_fit(frame: pd.DataFrame) -> dict[str, object]:
    single = frame.loc[frame["is_single_match_tie"]].reset_index(drop=True)

    def objective(draw_at_even: float) -> float:
        return float(evaluate_format_model(single, draw_at_even)["log_loss_1x2"].mean())

    fitted = minimize_scalar(
        objective,
        bounds=(0.01, 0.40),
        method="bounded",
        options={"xatol": 1e-12},
    )
    return {
        "converged": bool(fitted.success),
        "single_match_draw_at_even": float(fitted.x),
        "single_matches": len(single),
        "log_loss_1x2": float(fitted.fun),
    }


def run_walk_forward(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selections = []
    results = []
    prediction_frames = []
    for fold, test_index in enumerate(range(2, len(SEASONS)), start=1):
        train_seasons = SEASONS[:test_index]
        test_season = SEASONS[test_index]
        train = data.loc[data["season"].isin(train_seasons)].reset_index(drop=True)
        test = data.loc[data["season"].eq(test_season)].reset_index(drop=True)
        selected, surface = select_single_draw(train)
        selections.append(
            {
                "fold": fold,
                "train_seasons": "|".join(train_seasons),
                "test_season": test_season,
                "train_matches": len(train),
                "training_single_matches": int(train["is_single_match_tie"].sum()),
                "selected_single_match_draw_at_even": selected,
                "train_brier_1x2": float(surface.iloc[0]["brier_1x2"]),
                "train_log_loss_1x2": float(surface.iloc[0]["log_loss_1x2"]),
            }
        )
        arms = {
            "BASELINE_GLOBAL_DRAW": BASELINE_DRAW,
            "WALK_FORWARD_FORMAT_DRAW": selected,
            "PRESPECIFIED_SINGLE_DRAW_012": PRESPECIFIED_SINGLE_DRAW,
        }
        evaluated = {
            arm: evaluate_format_model(test, draw_value)
            for arm, draw_value in arms.items()
        }
        for arm, values in evaluated.items():
            results.append(
                {
                    "fold": fold,
                    "test_season": test_season,
                    "model_arm": arm,
                    "single_match_draw_at_even": arms[arm],
                    **summarize(test, values),
                }
            )
        joined = test.copy()
        for prefix, values in (
            ("baseline", evaluated["BASELINE_GLOBAL_DRAW"]),
            ("candidate", evaluated["WALK_FORWARD_FORMAT_DRAW"]),
            ("prespecified_012", evaluated["PRESPECIFIED_SINGLE_DRAW_012"]),
        ):
            for column in (
                "home_probability",
                "draw_probability",
                "away_probability",
                "brier_1x2",
                "log_loss_1x2",
                "predicted_outcome",
            ):
                joined[f"{prefix}_{column}"] = values[column].to_numpy()
        joined["fold"] = fold
        joined["selected_single_match_draw_at_even"] = selected
        prediction_frames.append(joined)
    return pd.DataFrame(selections), pd.DataFrame(results), pd.concat(prediction_frames, ignore_index=True)


def arm_summary(fold_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm, frame in fold_results.groupby("model_arm", sort=False):
        weights = frame["matches"].to_numpy(float)
        rows.append(
            {
                "model_arm": arm,
                "matches": int(weights.sum()),
                "single_match_ties": int(frame["single_match_ties"].sum()),
                **{
                    metric: float(np.average(frame[metric], weights=weights))
                    for metric in (
                        "brier_1x2",
                        "log_loss_1x2",
                        "accuracy_1x2",
                    )
                },
            }
        )
    result = pd.DataFrame(rows)
    baseline = result.loc[result.model_arm.eq("BASELINE_GLOBAL_DRAW")].iloc[0]
    for metric in ("brier_1x2", "log_loss_1x2", "accuracy_1x2"):
        result[f"delta_vs_baseline_{metric}"] = result[metric] - float(baseline[metric])
    return result


def segment_summary(unseen: pd.DataFrame) -> pd.DataFrame:
    segments = {
        "ALL": pd.Series(True, index=unseen.index),
        "SINGLE_MATCH": unseen["is_single_match_tie"],
        "TWO_LEGGED": unseen["is_two_legged_tie"],
        "NON_KNOCKOUT": ~unseen["is_knockout"].astype(bool),
        "POST_COVID_SINGLE_2021_22_PLUS": unseen["is_single_match_tie"] & ~unseen["season"].eq("2020/21"),
    }
    rows = []
    for segment, mask in segments.items():
        frame = unseen.loc[mask]
        for competition, part in [("ALL", frame), *frame.groupby("competition", sort=True)]:
            if part.empty:
                continue
            for arm in ("baseline", "candidate", "prespecified_012"):
                observed = np.where(
                    part.home_goals > part.away_goals,
                    "H",
                    np.where(part.home_goals == part.away_goals, "D", "A"),
                )
                rows.append(
                    {
                        "segment": segment,
                        "competition": competition,
                        "model_arm": arm.upper(),
                        "matches": len(part),
                        "brier_1x2": float(part[f"{arm}_brier_1x2"].mean()),
                        "log_loss_1x2": float(part[f"{arm}_log_loss_1x2"].mean()),
                        "accuracy_1x2": float((part[f"{arm}_predicted_outcome"].to_numpy() == observed).mean()),
                        "predicted_draw_rate": float(part[f"{arm}_draw_probability"].mean()),
                        "observed_draw_rate": float((part.home_goals == part.away_goals).mean()),
                    }
                )
    result = pd.DataFrame(rows)
    baseline = result.loc[result.model_arm.eq("BASELINE")].set_index(["segment", "competition"])
    for metric in ("brier_1x2", "log_loss_1x2", "accuracy_1x2"):
        result[f"delta_vs_baseline_{metric}"] = result.apply(
            lambda row: float(row[metric] - baseline.loc[(row.segment, row.competition), metric]),
            axis=1,
        )
    return result


def uncertainty(unseen: pd.DataFrame, bootstrap_samples: int) -> pd.DataFrame:
    rows = []
    for model_arm, prefix in (
        ("WALK_FORWARD_FORMAT_DRAW", "candidate"),
        ("PRESPECIFIED_SINGLE_DRAW_012", "prespecified_012"),
    ):
        for metric in ("brier_1x2", "log_loss_1x2"):
            source = unseen.copy()
            source["loss_difference"] = (
                source[f"{prefix}_{metric}"] - source[f"baseline_{metric}"]
            )
            result = dependency_robust_loss_difference_ci(
                source,
                bootstrap_samples=bootstrap_samples,
            )
            result.insert(0, "metric", metric)
            result.insert(0, "model_arm", model_arm)
            rows.append(result)
    return pd.concat(rows, ignore_index=True)


def decision(
    fold_results: pd.DataFrame,
    arms: pd.DataFrame,
    segments: pd.DataFrame,
    uncertainty_frame: pd.DataFrame,
    *,
    candidate: str = "WALK_FORWARD_FORMAT_DRAW",
    segment_arm: str = "CANDIDATE",
) -> tuple[str, dict[str, object]]:
    pivot = fold_results.pivot(index="fold", columns="model_arm", values=["brier_1x2", "log_loss_1x2"])
    baseline = "BASELINE_GLOBAL_DRAW"
    brier_wins = int((pivot[("brier_1x2", candidate)] < pivot[("brier_1x2", baseline)]).sum())
    log_wins = int((pivot[("log_loss_1x2", candidate)] < pivot[("log_loss_1x2", baseline)]).sum())
    candidate_arm = arms.loc[arms.model_arm.eq(candidate)].iloc[0]
    competition = segments.loc[
        segments["segment"].eq("ALL") & segments["model_arm"].eq(segment_arm)
    ]
    no_competition_harm = bool(
        competition["delta_vs_baseline_brier_1x2"].le(0.0).all()
        and competition["delta_vs_baseline_log_loss_1x2"].le(0.0).all()
    )
    envelope = uncertainty_frame.loc[
        uncertainty_frame["model_arm"].eq(candidate)
        & uncertainty_frame.method.eq("conservative_envelope")
    ]
    reliable = bool(envelope["reliable_improvement"].all())
    gates = {
        "brier_fold_wins": f"{brier_wins}/6",
        "log_loss_fold_wins": f"{log_wins}/6",
        "pooled_brier_improved": bool(candidate_arm.delta_vs_baseline_brier_1x2 < 0.0),
        "pooled_log_loss_improved": bool(candidate_arm.delta_vs_baseline_log_loss_1x2 < 0.0),
        "no_competition_harm": no_competition_harm,
        "dependency_reliable_improvement": reliable,
    }
    passed = all(
        (
            brier_wins >= 4,
            log_wins >= 4,
            gates["pooled_brier_improved"],
            gates["pooled_log_loss_improved"],
            no_competition_harm,
            reliable,
        )
    )
    return ("PROMOTE_CANDIDATE" if passed else "KEEP_SHADOW", gates)


def markdown_table(frame: pd.DataFrame, digits: int = 7) -> str:
    shown = frame.copy()
    for column in shown.select_dtypes(include=["float"]).columns:
        shown[column] = shown[column].map(lambda value: f"{value:.{digits}f}")
    headers = list(shown.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in shown.itertuples(index=False, name=None))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest format-aware draw probabilities for single-match ties")
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS)
    parser.add_argument("--events", type=Path, default=EVENTS)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    args = parser.parse_args()

    data = load_data(args.predictions.resolve(), args.events.resolve())
    selections, fold_results, unseen = run_walk_forward(data)
    arms = arm_summary(fold_results)
    segments = segment_summary(unseen)
    uncertainty_frame = uncertainty(unseen, args.bootstrap_samples)
    selected_full, surface = select_single_draw(data)
    continuous = continuous_single_fit(data)
    model_decision, gates = decision(fold_results, arms, segments, uncertainty_frame)
    prespecified_decision, prespecified_gates = decision(
        fold_results,
        arms,
        segments,
        uncertainty_frame,
        candidate="PRESPECIFIED_SINGLE_DRAW_012",
        segment_arm="PRESPECIFIED_012",
    )

    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    selections.to_csv(output / "fold_selections.csv", index=False)
    fold_results.to_csv(output / "fold_results.csv", index=False)
    arms.to_csv(output / "arm_summary.csv", index=False)
    segments.to_csv(output / "segment_summary.csv", index=False)
    uncertainty_frame.to_csv(output / "dependency_uncertainty.csv", index=False)
    surface.to_csv(output / "candidate_surface.csv", index=False)
    unseen.to_csv(output / "unseen_predictions.csv", index=False)
    manifest = {
        "decision": model_decision,
        "baseline_draw_at_even": BASELINE_DRAW,
        "draw_shape": BASELINE_SHAPE,
        "full_grid_single_match_draw_at_even": selected_full,
        "continuous_full_fit": continuous,
        "gates": gates,
        "prespecified_012_decision": prespecified_decision,
        "prespecified_012_gates": prespecified_gates,
        "production_engineering_decision": "ACTIVATE_STRUCTURAL_FORMAT_CORRECTION_0_12",
        "production_decision_basis": "pre-match format semantics plus retrospective and post-COVID sensitivity; not a claim of untouched parameter selection",
        "single_match_definition": "knockout tie with one scheduled field-score match",
        "format_is_pre_match_metadata": True,
        "rating_state_changed": False,
    }
    (output / "selected_candidate.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = "\n".join(
        [
            "# Single-Match Format-Aware Draw Backtest",
            "",
            f"Decision: **{model_decision}**",
            "",
            f"- All-history single matches: `{int(data.is_single_match_tie.sum())}`; two-legged matches: `{int(data.is_two_legged_tie.sum())}`.",
            f"- Full grid single-match draw-at-even: `{selected_full:g}`.",
            f"- Continuous full fit: `{continuous['single_match_draw_at_even']:.6f}`.",
            f"- Gates: `{gates}`.",
            f"- Prespecified 0.12 gates: `{prespecified_gates}` ({prespecified_decision}).",
            "- Engineering decision: `ACTIVATE_STRUCTURAL_FORMAT_CORRECTION_0_12`; this is a format-contract correction, not an untouched-holdout promotion claim.",
            "- Classification uses scheduled tie format only; scores, extra time, penalties and advanced team are excluded.",
            "- Rating state and Elo expected score are unchanged; only the H/D/A decomposition changes.",
            "",
            "## Fold selections",
            "",
            markdown_table(selections),
            "",
            "## Pooled arms",
            "",
            markdown_table(arms),
            "",
            "## Fold results",
            "",
            markdown_table(fold_results),
            "",
            "## Segment and COVID sensitivity",
            "",
            markdown_table(segments),
            "",
            "## Dependency uncertainty",
            "",
            markdown_table(uncertainty_frame),
            "",
        ]
    )
    (output / "backtest_report.md").write_text(report, encoding="utf-8")
    print(f"Decision: {model_decision}")
    print(f"Full grid single draw: {selected_full:g}")
    print(f"Continuous fit: {continuous['single_match_draw_at_even']:.6f}")
    print(f"Report: {output / 'backtest_report.md'}")


if __name__ == "__main__":
    main()
