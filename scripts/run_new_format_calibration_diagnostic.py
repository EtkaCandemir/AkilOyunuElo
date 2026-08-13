from __future__ import annotations

"""Evaluate whether the 2024/25 UEFA format needs distinct forecast calibration.

This is intentionally a prediction-only diagnostic.  It re-scores frozen
pre-match AO Live Elo values, so no result can alter AO First Elo, Power Elo,
goal-margin, xG or progression updates.  Two new-format seasons are too few
for a production branch; 2024/25 -> 2025/26 is the single available forward
check and 2026/27 remains the prospective holdout.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ao_elo.draw_probability import score_preserving_1x2_scalar  # noqa: E402
from ao_elo.evaluation import dependency_robust_loss_difference_ci  # noqa: E402


PREDICTIONS = ROOT / "output" / "current_model_evaluation_2018_2026" / "model_predictions.csv"
OUTPUT = ROOT / "output" / "new_format_calibration_diagnostic_2024_2026"
MODEL = "CURRENT_PRODUCTION"
BASE_ELO_SCALE = 835.5614973262034
BASE_HOME_ADVANTAGE = 148.54426619132505
PRE_FORMAT_SEASONS = ("2018/19", "2019/20", "2020/21", "2021/22", "2022/23", "2023/24")
NEW_FORMAT_SEASONS = ("2024/25", "2025/26")


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
    if set(NEW_FORMAT_SEASONS) - set(data["season"].astype(str)):
        raise ValueError("New-format seasons are missing from current-model predictions")
    return data.sort_values(["season", "match_id"], kind="stable").reset_index(drop=True)


def _validate_parameters(scale_multiplier: float, h_multiplier: float, draw_shape: float) -> None:
    values = (scale_multiplier, h_multiplier, draw_shape)
    if any(not np.isfinite(value) for value in values):
        raise ValueError("Format calibration parameters must be finite")
    if scale_multiplier <= 0.0 or h_multiplier < 0.0 or draw_shape <= 0.0:
        raise ValueError("Scale and shape must be positive; H multiplier cannot be negative")


def evaluate_format_parameters(
    frame: pd.DataFrame,
    scale_multiplier: float = 1.0,
    h_multiplier: float = 1.0,
    draw_shape: float = 1.0,
) -> pd.DataFrame:
    """Re-score frozen probabilities under a format-specific calibration."""
    _validate_parameters(scale_multiplier, h_multiplier, draw_shape)
    result = frame.copy()
    advantage = np.where(
        result["is_neutral"].astype(bool), 0.0, BASE_HOME_ADVANTAGE * h_multiplier
    )
    difference = (
        result["home_live_pre"].to_numpy(float)
        - result["away_live_pre"].to_numpy(float)
        + advantage
    )
    expected = 1.0 / (
        1.0 + 10.0 ** (-(difference / (BASE_ELO_SCALE * scale_multiplier)))
    )
    probabilities = np.array(
        [
            score_preserving_1x2_scalar(value, draw, draw_shape)
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
    result["format_scale_multiplier"] = float(scale_multiplier)
    result["format_h_multiplier"] = float(h_multiplier)
    result["format_draw_shape"] = float(draw_shape)
    result["forecast_elo_scale"] = BASE_ELO_SCALE * scale_multiplier
    result["effective_home_advantage"] = advantage
    result["forecast_expected_home_score"] = expected
    result[["home_probability", "draw_probability", "away_probability"]] = probabilities
    result["actual_class"] = actual
    result["predicted_class"] = probabilities.argmax(axis=1)
    result["brier_1x2"] = np.square(probabilities - np.eye(3)[actual]).sum(axis=1)
    result["log_loss_1x2"] = -np.log(
        np.clip(probabilities[np.arange(len(actual)), actual], 1e-15, 1.0)
    )
    result["rating_state_changed"] = False
    return result


def metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    return {
        "matches": int(len(frame)),
        "brier_1x2": float(frame["brier_1x2"].mean()),
        "log_loss_1x2": float(frame["log_loss_1x2"].mean()),
        "accuracy_1x2": float(frame["predicted_class"].eq(frame["actual_class"]).mean()),
        "predicted_home_rate": float(frame["home_probability"].mean()),
        "observed_home_rate": float(frame["actual_class"].eq(0).mean()),
        "predicted_draw_rate": float(frame["draw_probability"].mean()),
        "observed_draw_rate": float(frame["actual_class"].eq(1).mean()),
    }


def fit_parameters(frame: pd.DataFrame) -> dict[str, float | bool]:
    def objective(values: np.ndarray) -> float:
        return float(evaluate_format_parameters(frame, *values)["log_loss_1x2"].mean())

    result = minimize(
        objective,
        x0=np.array([1.0, 1.0, 1.0]),
        method="L-BFGS-B",
        bounds=((0.60, 1.80), (0.0, 2.00), (0.25, 2.00)),
        options={"ftol": 1e-13, "gtol": 1e-10, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(f"Format optimizer failed: {result.message}")
    scale_multiplier, h_multiplier, draw_shape = map(float, result.x)
    fitted = evaluate_format_parameters(frame, scale_multiplier, h_multiplier, draw_shape)
    baseline = evaluate_format_parameters(frame)
    return {
        "scale_multiplier": scale_multiplier,
        "h_multiplier": h_multiplier,
        "draw_shape": draw_shape,
        "elo_scale": BASE_ELO_SCALE * scale_multiplier,
        "home_advantage": BASE_HOME_ADVANTAGE * h_multiplier,
        "baseline_brier_1x2": metrics(baseline)["brier_1x2"],
        "fitted_brier_1x2": metrics(fitted)["brier_1x2"],
        "baseline_log_loss_1x2": metrics(baseline)["log_loss_1x2"],
        "fitted_log_loss_1x2": metrics(fitted)["log_loss_1x2"],
    }


def summarize_segments(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm, frame in predictions.groupby("model_arm", sort=False):
        for competition, part in [("ALL", frame), *frame.groupby("competition", sort=True)]:
            rows.append({"model_arm": arm, "competition": competition, **metrics(part)})
    result = pd.DataFrame(rows)
    baseline = result.loc[result["model_arm"].eq("BASELINE")].set_index("competition")
    for metric in ("brier_1x2", "log_loss_1x2", "accuracy_1x2"):
        result[f"delta_vs_baseline_{metric}"] = result.apply(
            lambda row: float(row[metric] - baseline.loc[row["competition"], metric]),
            axis=1,
        )
    return result


def uncertainty(predictions: pd.DataFrame, samples: int) -> pd.DataFrame:
    baseline = predictions.loc[predictions["model_arm"].eq("BASELINE")].set_index("match_id")
    candidate = predictions.loc[predictions["model_arm"].eq("TRAIN_2024_FORMAT_CANDIDATE")].set_index("match_id")
    if not baseline.index.equals(candidate.index):
        raise ValueError("Format-calibration arms do not share the same matches")
    source = candidate.reset_index().copy()
    for metric in ("brier_1x2", "log_loss_1x2"):
        source["loss_difference"] = candidate[metric].to_numpy() - baseline[metric].to_numpy()
        result = dependency_robust_loss_difference_ci(source, bootstrap_samples=samples)
        result.insert(0, "metric", metric)
        yield result


def markdown_table(frame: pd.DataFrame, digits: int = 6) -> str:
    shown = frame.copy()
    for column in shown.select_dtypes(include=["float"]).columns:
        shown[column] = shown[column].map(lambda value: f"{value:.{digits}f}")
    lines = [
        "| " + " | ".join(map(str, shown.columns)) + " |",
        "| " + " | ".join("---" for _ in shown.columns) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in shown.itertuples(index=False, name=None))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="AO new-format forecast calibration diagnostic")
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    args = parser.parse_args()

    data = load_predictions(args.predictions.resolve())
    pre_format = data.loc[data["season"].isin(PRE_FORMAT_SEASONS)]
    new_format = data.loc[data["season"].isin(NEW_FORMAT_SEASONS)]
    fit_rows = []
    for label, frame in (("PRE_FORMAT", pre_format), ("NEW_FORMAT", new_format)):
        fit_rows.append({"window": label, "matches": len(frame), **fit_parameters(frame)})
    for season in NEW_FORMAT_SEASONS:
        frame = data.loc[data["season"].eq(season)]
        fit_rows.append({"window": season, "matches": len(frame), **fit_parameters(frame)})
    fits = pd.DataFrame(fit_rows)

    train = data.loc[data["season"].eq("2024/25")]
    test = data.loc[data["season"].eq("2025/26")]
    selected = fit_parameters(train)
    baseline = evaluate_format_parameters(test).assign(model_arm="BASELINE")
    candidate = evaluate_format_parameters(
        test,
        float(selected["scale_multiplier"]),
        float(selected["h_multiplier"]),
        float(selected["draw_shape"]),
    ).assign(model_arm="TRAIN_2024_FORMAT_CANDIDATE")
    unseen = pd.concat((baseline, candidate), ignore_index=True)
    fold_results = pd.DataFrame(
        [
            {"model_arm": "BASELINE", **metrics(baseline)},
            {
                "model_arm": "TRAIN_2024_FORMAT_CANDIDATE",
                "scale_multiplier": selected["scale_multiplier"],
                "h_multiplier": selected["h_multiplier"],
                "draw_shape": selected["draw_shape"],
                **metrics(candidate),
            },
        ]
    )
    for metric in ("brier_1x2", "log_loss_1x2", "accuracy_1x2"):
        fold_results[f"delta_vs_baseline_{metric}"] = fold_results[metric] - fold_results.loc[0, metric]
    segments = summarize_segments(unseen)
    intervals = pd.concat(list(uncertainty(unseen, args.bootstrap_samples)), ignore_index=True)
    full_row = segments.loc[(segments["model_arm"].eq("TRAIN_2024_FORMAT_CANDIDATE")) & (segments["competition"].eq("ALL"))].iloc[0]
    decision = "KEEP_FORMAT_SHADOW"
    gates = {
        "available_forward_folds": 1,
        "minimum_required_forward_folds": 3,
        "oos_brier_improved": bool(full_row["delta_vs_baseline_brier_1x2"] < 0.0),
        "oos_log_loss_improved": bool(full_row["delta_vs_baseline_log_loss_1x2"] < 0.0),
        "future_prospective_holdout": "2026/27",
        "rating_state_changed": False,
    }

    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    fits.to_csv(output / "format_window_fits.csv", index=False)
    pd.DataFrame([selected]).to_csv(output / "fold_selection_2024_to_2025.csv", index=False)
    fold_results.to_csv(output / "fold_result_2025_26.csv", index=False)
    unseen.to_csv(output / "unseen_predictions_2025_26.csv", index=False)
    segments.to_csv(output / "competition_summary_2025_26.csv", index=False)
    intervals.to_csv(output / "dependency_uncertainty_2025_26.csv", index=False)
    manifest = {
        "decision": decision,
        "new_format_seasons": list(NEW_FORMAT_SEASONS),
        "new_format_matches": int(len(new_format)),
        "selection_train_season": "2024/25",
        "unseen_test_season": "2025/26",
        "next_prospective_holdout": "2026/27",
        "rating_state_changed": False,
        "scope": "prediction-only; format parameters are not active in the production dynamic core",
        "gates": gates,
    }
    (output / "selected_candidate.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = "\n".join(
        [
            "# AO New-Format Calibration Diagnostic",
            "",
            f"Decision: **{decision}**",
            "",
            "- 2024/25 and 2025/26 are the only new-format seasons, so there is one valid 2024/25 -> 2025/26 forward check.",
            "- These parameters re-score frozen predictions only; no AO Live Elo state is changed.",
            "- A format-specific production branch requires further prospective evidence from 2026/27.",
            f"- Gates: `{gates}`.",
            "",
            "## Window fits",
            "",
            markdown_table(fits),
            "",
            "## 2024/25 selection applied to 2025/26",
            "",
            markdown_table(fold_results),
            "",
            "## 2025/26 competition segments",
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
    print(f"Decision: {decision}")
    print(f"2024/25 fit: scale={selected['scale_multiplier']:.6f}, H={selected['h_multiplier']:.6f}, shape={selected['draw_shape']:.6f}")
    print(f"Report: {output / 'backtest_report.md'}")


if __name__ == "__main__":
    main()
