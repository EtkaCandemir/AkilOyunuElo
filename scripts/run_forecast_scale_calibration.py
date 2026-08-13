from __future__ import annotations

"""Test a prediction-only Elo spread calibration on frozen pre-match ratings.

The multiplier changes only the conversion from AO Live Elo difference to the
pre-match H/D/A probability. It never feeds back into the Power Elo update.
"""

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

from ao_elo.evaluation import dependency_robust_loss_difference_ci  # noqa: E402
from ao_elo.draw_probability import score_preserving_1x2_scalar  # noqa: E402


PREDICTIONS = ROOT / "output" / "current_model_evaluation_2018_2026" / "model_predictions.csv"
OUTPUT = ROOT / "output" / "forecast_scale_calibration_2018_2026"
MODEL = "CURRENT_PRODUCTION"
BASE_ELO_SCALE = 835.5614973262034
HOME_ADVANTAGE = 148.54426619132505
SCALE_MULTIPLIERS = tuple(float(value) for value in np.arange(0.80, 1.301, 0.025).round(3))
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


def load_predictions(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data = data.loc[data["model"].eq(MODEL)].copy()
    required = {
        "match_id", "season", "competition", "home_live_pre", "away_live_pre",
        "is_neutral", "home_goals", "away_goals", "effective_draw_at_even",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Current-model predictions missing columns: {missing}")
    if len(data) != 6340 or data["match_id"].duplicated().any():
        raise ValueError("Expected 6,340 unique current-model prediction rows")
    if set(data["season"].astype(str)) != set(SEASONS):
        raise ValueError("Current-model predictions do not contain the frozen 2018/19-2025/26 seasons")
    return data.sort_values(["season", "match_id"], kind="stable").reset_index(drop=True)


def evaluate_multiplier(frame: pd.DataFrame, multiplier: float) -> pd.DataFrame:
    if not np.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("scale multiplier must be positive and finite")
    result = frame.copy()
    advantage = np.where(result["is_neutral"].astype(bool), 0.0, HOME_ADVANTAGE)
    difference = (
        result["home_live_pre"].to_numpy(float)
        - result["away_live_pre"].to_numpy(float)
        + advantage
    )
    expected = 1.0 / (1.0 + 10.0 ** (-(difference / (BASE_ELO_SCALE * multiplier))))
    probabilities = np.array(
        [
            score_preserving_1x2_scalar(value, draw, 1.0)
            for value, draw in zip(
                expected,
                result["effective_draw_at_even"].to_numpy(float),
                strict=True,
            )
        ],
        dtype=float,
    )
    actual = np.where(
        result["home_goals"].to_numpy(int) > result["away_goals"].to_numpy(int),
        0,
        np.where(
            result["home_goals"].to_numpy(int) == result["away_goals"].to_numpy(int),
            1,
            2,
        ),
    )
    result["forecast_scale_multiplier"] = float(multiplier)
    result["forecast_elo_scale"] = BASE_ELO_SCALE * float(multiplier)
    result["forecast_expected_home_score"] = expected
    result[["home_probability", "draw_probability", "away_probability"]] = probabilities
    result["actual_class"] = actual
    result["predicted_class"] = probabilities.argmax(axis=1)
    result["brier_1x2"] = np.square(probabilities - np.eye(3)[actual]).sum(axis=1)
    result["log_loss_1x2"] = -np.log(np.clip(probabilities[np.arange(len(actual)), actual], 1e-15, 1.0))
    result["rating_state_changed"] = False
    return result


def metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    return {
        "matches": len(frame),
        "brier_1x2": float(frame["brier_1x2"].mean()),
        "log_loss_1x2": float(frame["log_loss_1x2"].mean()),
        "accuracy_1x2": float(frame["predicted_class"].eq(frame["actual_class"]).mean()),
        "predicted_home_rate": float(frame["home_probability"].mean()),
        "observed_home_rate": float(frame["actual_class"].eq(0).mean()),
    }


def select_multiplier(train: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    rows = []
    for multiplier in SCALE_MULTIPLIERS:
        evaluated = evaluate_multiplier(train, multiplier)
        rows.append(
            {
                "forecast_scale_multiplier": multiplier,
                "forecast_elo_scale": BASE_ELO_SCALE * multiplier,
                "distance_from_one": abs(multiplier - 1.0),
                **metrics(evaluated),
            }
        )
    surface = pd.DataFrame(rows).sort_values(
        ["log_loss_1x2", "brier_1x2", "distance_from_one"], kind="stable"
    ).reset_index(drop=True)
    return float(surface.iloc[0]["forecast_scale_multiplier"]), surface


def continuous_fit(frame: pd.DataFrame) -> float:
    fitted = minimize_scalar(
        lambda multiplier: float(evaluate_multiplier(frame, multiplier)["log_loss_1x2"].mean()),
        bounds=(0.50, 1.80),
        method="bounded",
        options={"xatol": 1e-10},
    )
    if not fitted.success:
        raise RuntimeError("Forecast-scale optimizer did not converge")
    return float(fitted.x)


def season_fits(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, frame in data.groupby("season", sort=False):
        fitted = continuous_fit(frame)
        baseline = evaluate_multiplier(frame, 1.0)
        selected = evaluate_multiplier(frame, fitted)
        rows.append(
            {
                "season": season,
                "matches": len(frame),
                "continuous_scale_multiplier": fitted,
                "continuous_elo_scale": BASE_ELO_SCALE * fitted,
                "baseline_brier_1x2": metrics(baseline)["brier_1x2"],
                "fitted_brier_1x2": metrics(selected)["brier_1x2"],
                "baseline_log_loss_1x2": metrics(baseline)["log_loss_1x2"],
                "fitted_log_loss_1x2": metrics(selected)["log_loss_1x2"],
            }
        )
    return pd.DataFrame(rows)


def walk_forward(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selections: list[dict[str, object]] = []
    results: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    for fold, test_index in enumerate(range(2, len(SEASONS)), start=1):
        train_seasons = SEASONS[:test_index]
        test_season = SEASONS[test_index]
        train = data.loc[data["season"].isin(train_seasons)]
        test = data.loc[data["season"].eq(test_season)]
        selected, surface = select_multiplier(train)
        selections.append(
            {
                "fold": fold,
                "train_seasons": "|".join(train_seasons),
                "test_season": test_season,
                "train_matches": len(train),
                "selected_scale_multiplier": selected,
                "selected_elo_scale": BASE_ELO_SCALE * selected,
                "train_brier_1x2": float(surface.iloc[0]["brier_1x2"]),
                "train_log_loss_1x2": float(surface.iloc[0]["log_loss_1x2"]),
            }
        )
        for arm, multiplier in (("BASELINE", 1.0), ("NESTED_FORECAST_SCALE", selected)):
            evaluated = evaluate_multiplier(test, multiplier)
            evaluated["model_arm"] = arm
            evaluated["fold"] = fold
            predictions.append(evaluated)
            results.append(
                {
                    "fold": fold,
                    "test_season": test_season,
                    "model_arm": arm,
                    "forecast_scale_multiplier": multiplier,
                    "forecast_elo_scale": BASE_ELO_SCALE * multiplier,
                    **metrics(evaluated),
                }
            )
    return pd.DataFrame(selections), pd.DataFrame(results), pd.concat(predictions, ignore_index=True)


def summarize_segments(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm, frame in predictions.groupby("model_arm", sort=False):
        for competition, part in [("ALL", frame), *frame.groupby("competition", sort=True)]:
            rows.append({"model_arm": arm, "competition": competition, **metrics(part)})
    result = pd.DataFrame(rows)
    baseline = result.loc[result["model_arm"].eq("BASELINE")].set_index("competition")
    for metric in ("brier_1x2", "log_loss_1x2", "accuracy_1x2"):
        result[f"delta_vs_baseline_{metric}"] = result.apply(
            lambda row: float(row[metric] - baseline.loc[row["competition"], metric]), axis=1
        )
    return result


def uncertainty(predictions: pd.DataFrame, samples: int) -> pd.DataFrame:
    baseline = predictions.loc[predictions["model_arm"].eq("BASELINE")].set_index("match_id")
    candidate = predictions.loc[predictions["model_arm"].eq("NESTED_FORECAST_SCALE")].set_index("match_id")
    if not baseline.index.equals(candidate.index):
        raise ValueError("Forecast-scale arms do not share the same matches")
    source = candidate.reset_index().copy()
    for metric in ("brier_1x2", "log_loss_1x2"):
        source["loss_difference"] = candidate[metric].to_numpy() - baseline[metric].to_numpy()
        result = dependency_robust_loss_difference_ci(source, bootstrap_samples=samples)
        result.insert(0, "metric", metric)
        yield result


def decision(results: pd.DataFrame, segments: pd.DataFrame, intervals: pd.DataFrame) -> tuple[str, dict[str, object]]:
    pivot = results.pivot(index="fold", columns="model_arm", values=["brier_1x2", "log_loss_1x2"])
    brier_wins = int((pivot[("brier_1x2", "NESTED_FORECAST_SCALE")] < pivot[("brier_1x2", "BASELINE")]).sum())
    log_wins = int((pivot[("log_loss_1x2", "NESTED_FORECAST_SCALE")] < pivot[("log_loss_1x2", "BASELINE")]).sum())
    candidate = segments.loc[segments["model_arm"].eq("NESTED_FORECAST_SCALE")]
    all_row = candidate.loc[candidate["competition"].eq("ALL")].iloc[0]
    no_competition_harm = bool(
        candidate.loc[candidate["competition"].isin(("UCL", "UEL", "UECL")), "delta_vs_baseline_brier_1x2"].le(0.0).all()
        and candidate.loc[candidate["competition"].isin(("UCL", "UEL", "UECL")), "delta_vs_baseline_log_loss_1x2"].le(0.0).all()
    )
    envelope = intervals.loc[intervals["method"].eq("conservative_envelope")]
    reliable_harm = bool(envelope["reliable_harm"].any())
    gates = {
        "brier_fold_wins": f"{brier_wins}/6",
        "log_loss_fold_wins": f"{log_wins}/6",
        "pooled_brier_improved": bool(all_row["delta_vs_baseline_brier_1x2"] < 0.0),
        "pooled_log_loss_improved": bool(all_row["delta_vs_baseline_log_loss_1x2"] < 0.0),
        "no_competition_harm": no_competition_harm,
        "no_reliable_harm": not reliable_harm,
    }
    passed = all((brier_wins >= 4, log_wins >= 4, gates["pooled_brier_improved"], gates["pooled_log_loss_improved"], no_competition_harm, not reliable_harm))
    return ("PROMOTE_PREDICTION_ONLY_CANDIDATE" if passed else "KEEP_SHADOW", gates)


def markdown_table(frame: pd.DataFrame, digits: int = 6) -> str:
    shown = frame.copy()
    for column in shown.select_dtypes(include=["float"]).columns:
        shown[column] = shown[column].map(lambda value: f"{value:.{digits}f}")
    lines = ["| " + " | ".join(map(str, shown.columns)) + " |", "| " + " | ".join("---" for _ in shown.columns) + " |"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in shown.itertuples(index=False, name=None))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward AO forecast-scale calibration")
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    args = parser.parse_args()

    data = load_predictions(args.predictions.resolve())
    fits = season_fits(data)
    full_grid, surface = select_multiplier(data)
    full_continuous = continuous_fit(data)
    selections, results, unseen = walk_forward(data)
    segments = summarize_segments(unseen)
    intervals = pd.concat(list(uncertainty(unseen, args.bootstrap_samples)), ignore_index=True)
    model_decision, gates = decision(results, segments, intervals)

    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    fits.to_csv(output / "season_scale_fits.csv", index=False)
    surface.to_csv(output / "full_candidate_surface.csv", index=False)
    selections.to_csv(output / "fold_selections.csv", index=False)
    results.to_csv(output / "fold_results.csv", index=False)
    unseen.to_csv(output / "unseen_predictions.csv", index=False)
    segments.to_csv(output / "competition_summary.csv", index=False)
    intervals.to_csv(output / "dependency_uncertainty.csv", index=False)
    manifest = {
        "decision": model_decision,
        "base_elo_scale": BASE_ELO_SCALE,
        "full_grid_multiplier": full_grid,
        "full_grid_elo_scale": BASE_ELO_SCALE * full_grid,
        "continuous_full_multiplier": full_continuous,
        "continuous_full_elo_scale": BASE_ELO_SCALE * full_continuous,
        "gates": gates,
        "rating_state_changed": False,
        "scope": "prediction-only; no feedback to Power Elo or ranking",
    }
    (output / "selected_candidate.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = "\n".join(
        [
            "# AO Forecast-Scale Calibration",
            "",
            f"Decision: **{model_decision}**",
            "",
            f"- Base dynamic Elo scale: `{BASE_ELO_SCALE:.6f}`.",
            f"- Full grid multiplier: `{full_grid:.3f}`; continuous multiplier: `{full_continuous:.6f}`.",
            "- The layer is prediction-only. AO First Elo, Power Elo, ranking, xG, goal-margin and progression values are unchanged.",
            f"- Gates: `{gates}`.",
            "",
            "## Season diagnostics",
            "",
            markdown_table(fits),
            "",
            "## Fold selections",
            "",
            markdown_table(selections),
            "",
            "## Fold results",
            "",
            markdown_table(results),
            "",
            "## Competition summary",
            "",
            markdown_table(segments),
            "",
            "## Dependency uncertainty",
            "",
            markdown_table(intervals),
            "",
        ]
    )
    (output / "backtest_report.md").write_text(report, encoding="utf-8")
    print(f"Decision: {model_decision}")
    print(f"Full continuous multiplier: {full_continuous:.6f}")
    print(f"Report: {output / 'backtest_report.md'}")


if __name__ == "__main__":
    main()
