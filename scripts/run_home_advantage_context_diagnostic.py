from __future__ import annotations

"""Diagnose home-advantage calibration without changing Elo ratings.

The historical 2020/21 season is an exceptional closed-door period.  A season
label is not a production-safe proxy for attendance, so this script keeps the
season-specific result explicitly retrospective.  It selects a normal-venue H
only from non-COVID training seasons and never feeds any candidate back into
the Power Elo state.
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

from ao_elo.draw_probability import score_preserving_1x2_scalar  # noqa: E402
from ao_elo.evaluation import dependency_robust_loss_difference_ci  # noqa: E402


PREDICTIONS = ROOT / "output" / "current_model_evaluation_2018_2026" / "model_predictions.csv"
OUTPUT = ROOT / "output" / "home_advantage_context_diagnostic_2018_2026"
MODEL = "CURRENT_PRODUCTION"
BASE_ELO_SCALE = 835.5614973262034
BASE_HOME_ADVANTAGE = 148.54426619132505
COVID_SEASON = "2020/21"
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
H_MULTIPLIERS = tuple(float(value) for value in np.arange(0.50, 1.501, 0.025).round(3))


def load_predictions(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data = data.loc[data["model"].eq(MODEL)].copy()
    required = {
        "match_id",
        "season",
        "competition",
        "home_live_pre",
        "away_live_pre",
        "is_neutral",
        "home_goals",
        "away_goals",
        "effective_draw_at_even",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Current-model predictions missing columns: {missing}")
    if len(data) != 6340 or data["match_id"].duplicated().any():
        raise ValueError("Expected 6,340 unique current-model prediction rows")
    if set(data["season"].astype(str)) != set(SEASONS):
        raise ValueError("Current-model predictions do not contain frozen 2018/19-2025/26 seasons")
    return data.sort_values(["season", "match_id"], kind="stable").reset_index(drop=True)


def evaluate_home_advantage_multiplier(
    frame: pd.DataFrame,
    multiplier: float,
) -> pd.DataFrame:
    """Re-score frozen predictions with one global H multiplier.

    This is deliberately prediction-only: neither pre-match ratings nor the
    dynamic updates are rebuilt under the candidate value.
    """
    if not np.isfinite(multiplier) or multiplier < 0.0:
        raise ValueError("home-advantage multiplier must be finite and non-negative")
    result = frame.copy()
    advantage = np.where(
        result["is_neutral"].astype(bool),
        0.0,
        BASE_HOME_ADVANTAGE * float(multiplier),
    )
    difference = (
        result["home_live_pre"].to_numpy(float)
        - result["away_live_pre"].to_numpy(float)
        + advantage
    )
    expected = 1.0 / (1.0 + 10.0 ** (-(difference / BASE_ELO_SCALE)))
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
    result["home_advantage_multiplier"] = float(multiplier)
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
    }


def select_multiplier(train: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    rows = []
    for multiplier in H_MULTIPLIERS:
        evaluated = evaluate_home_advantage_multiplier(train, multiplier)
        rows.append(
            {
                "home_advantage_multiplier": multiplier,
                "effective_home_advantage": BASE_HOME_ADVANTAGE * multiplier,
                "distance_from_one": abs(multiplier - 1.0),
                **metrics(evaluated),
            }
        )
    surface = pd.DataFrame(rows).sort_values(
        ["log_loss_1x2", "brier_1x2", "distance_from_one"],
        kind="stable",
    ).reset_index(drop=True)
    return float(surface.iloc[0]["home_advantage_multiplier"]), surface


def continuous_fit(frame: pd.DataFrame) -> float:
    result = minimize_scalar(
        lambda multiplier: float(
            evaluate_home_advantage_multiplier(frame, multiplier)["log_loss_1x2"].mean()
        ),
        bounds=(0.0, 2.0),
        method="bounded",
        options={"xatol": 1e-10},
    )
    if not result.success:
        raise RuntimeError("Home-advantage optimizer did not converge")
    return float(result.x)


def season_diagnostics(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, frame in data.groupby("season", sort=False):
        fitted = continuous_fit(frame)
        baseline = evaluate_home_advantage_multiplier(frame, 1.0)
        selected = evaluate_home_advantage_multiplier(frame, fitted)
        rows.append(
            {
                "season": season,
                "matches": len(frame),
                "continuous_h_multiplier": fitted,
                "continuous_home_advantage": BASE_HOME_ADVANTAGE * fitted,
                "baseline_brier_1x2": metrics(baseline)["brier_1x2"],
                "fitted_brier_1x2": metrics(selected)["brier_1x2"],
                "baseline_log_loss_1x2": metrics(baseline)["log_loss_1x2"],
                "fitted_log_loss_1x2": metrics(selected)["log_loss_1x2"],
            }
        )
    return pd.DataFrame(rows)


def normal_venue_walk_forward(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select normal H from only past, non-COVID seasons.

    2020/21 is skipped here: absent date-level attendance metadata, a season
    proxy would not be deployable.  Its fitted multiplier stays diagnostic.
    """
    selections: list[dict[str, object]] = []
    results: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    for fold, test_index in enumerate(range(2, len(SEASONS)), start=1):
        test_season = SEASONS[test_index]
        if test_season == COVID_SEASON:
            continue
        train_seasons = tuple(
            season for season in SEASONS[:test_index] if season != COVID_SEASON
        )
        train = data.loc[data["season"].isin(train_seasons)]
        test = data.loc[data["season"].eq(test_season)]
        selected, surface = select_multiplier(train)
        selections.append(
            {
                "fold": fold,
                "train_seasons": "|".join(train_seasons),
                "excluded_season": COVID_SEASON,
                "test_season": test_season,
                "train_matches": len(train),
                "selected_normal_h_multiplier": selected,
                "selected_normal_home_advantage": BASE_HOME_ADVANTAGE * selected,
                "train_brier_1x2": float(surface.iloc[0]["brier_1x2"]),
                "train_log_loss_1x2": float(surface.iloc[0]["log_loss_1x2"]),
            }
        )
        for arm, multiplier in (("BASELINE", 1.0), ("NORMAL_H_NESTED", selected)):
            evaluated = evaluate_home_advantage_multiplier(test, multiplier)
            evaluated["model_arm"] = arm
            evaluated["fold"] = fold
            predictions.append(evaluated)
            results.append(
                {
                    "fold": fold,
                    "test_season": test_season,
                    "model_arm": arm,
                    "home_advantage_multiplier": multiplier,
                    "effective_home_advantage": BASE_HOME_ADVANTAGE * multiplier,
                    **metrics(evaluated),
                }
            )
    return pd.DataFrame(selections), pd.DataFrame(results), pd.concat(predictions, ignore_index=True)


def summarize_competitions(predictions: pd.DataFrame) -> pd.DataFrame:
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
    candidate = predictions.loc[predictions["model_arm"].eq("NORMAL_H_NESTED")].set_index("match_id")
    if not baseline.index.equals(candidate.index):
        raise ValueError("Home-advantage arms do not share the same matches")
    source = candidate.reset_index().copy()
    for metric in ("brier_1x2", "log_loss_1x2"):
        source["loss_difference"] = candidate[metric].to_numpy() - baseline[metric].to_numpy()
        result = dependency_robust_loss_difference_ci(source, bootstrap_samples=samples)
        result.insert(0, "metric", metric)
        yield result


def decision(results: pd.DataFrame, segments: pd.DataFrame, intervals: pd.DataFrame) -> tuple[str, dict[str, object]]:
    pivot = results.pivot(index="fold", columns="model_arm", values=["brier_1x2", "log_loss_1x2"])
    brier_wins = int((pivot[("brier_1x2", "NORMAL_H_NESTED")] < pivot[("brier_1x2", "BASELINE")]).sum())
    log_wins = int((pivot[("log_loss_1x2", "NORMAL_H_NESTED")] < pivot[("log_loss_1x2", "BASELINE")]).sum())
    candidate = segments.loc[segments["model_arm"].eq("NORMAL_H_NESTED")]
    all_row = candidate.loc[candidate["competition"].eq("ALL")].iloc[0]
    no_competition_harm = bool(
        candidate.loc[candidate["competition"].isin(("UCL", "UEL", "UECL")), "delta_vs_baseline_brier_1x2"].le(0.0).all()
        and candidate.loc[candidate["competition"].isin(("UCL", "UEL", "UECL")), "delta_vs_baseline_log_loss_1x2"].le(0.0).all()
    )
    reliable_harm = bool(intervals.loc[intervals["method"].eq("conservative_envelope"), "reliable_harm"].any())
    gates = {
        "evaluatable_folds": 5,
        "brier_fold_wins": f"{brier_wins}/5",
        "log_loss_fold_wins": f"{log_wins}/5",
        "pooled_brier_improved": bool(all_row["delta_vs_baseline_brier_1x2"] < 0.0),
        "pooled_log_loss_improved": bool(all_row["delta_vs_baseline_log_loss_1x2"] < 0.0),
        "no_competition_harm": no_competition_harm,
        "no_reliable_harm": not reliable_harm,
        "covid_status": "diagnostic only; no match-level closed-door metadata",
    }
    passed = all((brier_wins >= 4, log_wins >= 4, gates["pooled_brier_improved"], gates["pooled_log_loss_improved"], no_competition_harm, not reliable_harm))
    return ("PROMOTE_PREDICTION_ONLY_CANDIDATE" if passed else "KEEP_SHADOW_PENDING_VENUE_METADATA", gates)


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
    parser = argparse.ArgumentParser(description="AO home-advantage context diagnostic")
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    args = parser.parse_args()

    data = load_predictions(args.predictions.resolve())
    diagnostics = season_diagnostics(data)
    normal = data.loc[data["season"].ne(COVID_SEASON)]
    normal_grid, normal_surface = select_multiplier(normal)
    normal_continuous = continuous_fit(normal)
    covid = data.loc[data["season"].eq(COVID_SEASON)]
    covid_continuous = continuous_fit(covid)
    selections, results, unseen = normal_venue_walk_forward(data)
    segments = summarize_competitions(unseen)
    intervals = pd.concat(list(uncertainty(unseen, args.bootstrap_samples)), ignore_index=True)
    model_decision, gates = decision(results, segments, intervals)

    retrospective = pd.concat(
        [
            evaluate_home_advantage_multiplier(normal, normal_continuous).assign(model_arm="NORMAL_POOL_RESCORING"),
            evaluate_home_advantage_multiplier(covid, covid_continuous).assign(model_arm="COVID_SEASON_RETROSPECTIVE_ONLY"),
        ],
        ignore_index=True,
    )
    retrospective_summary = pd.DataFrame(
        [
            {"arm": "BASELINE", **metrics(evaluate_home_advantage_multiplier(data, 1.0))},
            {"arm": "RETROSPECTIVE_SEASON_PROXY", **metrics(retrospective)},
        ]
    )
    for metric in ("brier_1x2", "log_loss_1x2", "accuracy_1x2"):
        retrospective_summary[f"delta_vs_baseline_{metric}"] = retrospective_summary[metric] - retrospective_summary.loc[0, metric]

    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(output / "season_h_fits.csv", index=False)
    normal_surface.to_csv(output / "non_covid_candidate_surface.csv", index=False)
    selections.to_csv(output / "fold_selections.csv", index=False)
    results.to_csv(output / "fold_results.csv", index=False)
    unseen.to_csv(output / "unseen_predictions.csv", index=False)
    segments.to_csv(output / "competition_summary.csv", index=False)
    intervals.to_csv(output / "dependency_uncertainty.csv", index=False)
    retrospective_summary.to_csv(output / "retrospective_covid_proxy_summary.csv", index=False)
    manifest = {
        "decision": model_decision,
        "base_home_advantage": BASE_HOME_ADVANTAGE,
        "non_covid_grid_multiplier": normal_grid,
        "non_covid_continuous_multiplier": normal_continuous,
        "covid_continuous_multiplier": covid_continuous,
        "covid_season": COVID_SEASON,
        "season_proxy_is_production_safe": False,
        "required_production_input": "pre-match, match-level venue attendance status (NORMAL/CLOSED_DOORS)",
        "rating_state_changed": False,
        "scope": "prediction-only diagnostic; no feedback to AO Live Elo or ranking",
        "gates": gates,
    }
    (output / "selected_candidate.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = "\n".join(
        [
            "# AO Home-Advantage Context Diagnostic",
            "",
            f"Decision: **{model_decision}**",
            "",
            "- This study keeps 2020/21 as a closed-door-era diagnostic, not a deployable season rule.",
            "- Production needs a pre-match match-level `NORMAL` / `CLOSED_DOORS` flag; a season name must not stand in for venue status.",
            "- AO First Elo, AO Live Elo, Power Delta, goal margin, xG and progression stay unchanged.",
            f"- Base H: `{BASE_HOME_ADVANTAGE:.6f}`. Non-COVID pooled fit: `{normal_continuous:.6f}x`. COVID diagnostic fit: `{covid_continuous:.6f}x`.",
            f"- Gates: `{gates}`.",
            "",
            "## Season diagnostics",
            "",
            markdown_table(diagnostics),
            "",
            "## Normal-venue nested selections",
            "",
            markdown_table(selections),
            "",
            "## Normal-venue fold results",
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
            "## Retrospective season-proxy diagnostic",
            "",
            "This comparison is explanatory only and is not eligible for promotion.",
            "",
            markdown_table(retrospective_summary),
            "",
        ]
    )
    (output / "backtest_report.md").write_text(report, encoding="utf-8")
    print(f"Decision: {model_decision}")
    print(f"Non-COVID continuous H multiplier: {normal_continuous:.6f}")
    print(f"COVID diagnostic H multiplier: {covid_continuous:.6f}")
    print(f"Report: {output / 'backtest_report.md'}")


if __name__ == "__main__":
    main()
