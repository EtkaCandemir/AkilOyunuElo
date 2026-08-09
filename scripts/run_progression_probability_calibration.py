from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ao_elo.evaluation import dependency_robust_loss_difference_ci  # noqa: E402
from ao_elo.progression_probability import (  # noqa: E402
    ProgressionProbabilityConfig,
    calibrate_progression_probability,
    identity_progression_probability_config,
)
from scripts.run_v2_achievement_reserve_calibration import (  # noqa: E402
    normalize_exact_tie_deciders,
)


PROGRESSION_PREDICTIONS_PATH = (
    ROOT / "output" / "progression_prestige_calibration_2018_2026"
    / "unseen_match_predictions.csv"
)
EVENTS_PATH = (
    ROOT / "data" / "external_elo_benchmark_2018_2026"
    / "exact_date_events.csv"
)
OUTPUT_ROOT = ROOT / "output" / "progression_probability_2018_2026"
SLOPES = (0.50, 0.75, 1.00, 1.25, 1.50, 2.00)
SINGLE_HOME_BIASES = (0.0, 0.10, 0.20, 0.30, 0.40)
TWO_LEG_FIRST_HOME_BIASES = (-0.30, -0.15, 0.0, 0.15, 0.30)


def candidate_grid() -> tuple[ProgressionProbabilityConfig, ...]:
    candidates = {
        ProgressionProbabilityConfig(slope, single, two_leg)
        for slope in SLOPES
        for single in SINGLE_HOME_BIASES
        for two_leg in TWO_LEG_FIRST_HOME_BIASES
    }
    result = tuple(sorted(candidates))
    for candidate in result:
        candidate.validate()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate real single/two-leg UEFA progression probability"
    )
    parser.add_argument(
        "--progression-predictions",
        type=Path,
        default=PROGRESSION_PREDICTIONS_PATH,
    )
    parser.add_argument("--events", type=Path, default=EVENTS_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples must be at least 100")

    ties = load_tie_probability_data(
        args.progression_predictions.resolve(),
        args.events.resolve(),
    )
    candidates = candidate_grid()
    seasons = tuple(sorted(ties["season"].unique()))
    selections, predictions = run_walk_forward(ties, seasons, candidates)
    full_metrics = evaluate_candidates(ties, candidates)
    full_candidate = config_from_row(full_metrics.iloc[0])
    comparison = summarize_models(predictions)
    segments = segment_summary(predictions)
    uncertainty = build_uncertainty(
        predictions,
        bootstrap_samples=args.bootstrap_samples,
    )
    decision, guardrails = make_decision(
        selections,
        comparison,
        segments,
        uncertainty,
    )
    configs_by_test_season = {
        str(row.test_season): ProgressionProbabilityConfig(
            float(row.selected_logit_slope),
            float(row.selected_single_home_bias),
            float(row.selected_two_leg_first_home_bias),
        )
        for row in selections.itertuples(index=False)
    }

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    ties.to_csv(output_root / "tie_probability_dataset.csv", index=False)
    selections.to_csv(output_root / "fold_selections.csv", index=False)
    predictions.to_csv(output_root / "unseen_tie_predictions.csv", index=False)
    comparison.to_csv(output_root / "model_comparison.csv", index=False)
    segments.to_csv(output_root / "segment_summary.csv", index=False)
    uncertainty.to_csv(
        output_root / "dependency_uncertainty.csv",
        index=False,
    )
    full_metrics.to_csv(
        output_root / "full_candidate_metrics.csv",
        index=False,
    )
    payload = {
        "decision": decision,
        "probability_model": (
            "logit(P_advance)=slope*logit(P_raw)+format_bias"
        ),
        "full_data_candidate": config_payload(full_candidate),
        "identity_baseline": config_payload(
            identity_progression_probability_config()
        ),
        "configs_by_test_season": {
            season: config_payload(config)
            for season, config in configs_by_test_season.items()
        },
        "guardrails": guardrails,
        "reserve_retest_authorized": bool(
            guardrails["clustered_log_loss_reliable_improvement"]
            and guardrails["brier_delta"] < 0.0
            and guardrails["log_loss_delta"] < 0.0
        ),
    }
    (output_root / "selected_progression_probability.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "calibration_report.md").write_text(
        build_report(
            decision,
            ties,
            selections,
            comparison,
            segments,
            uncertainty,
            full_candidate,
            guardrails,
        ),
        encoding="utf-8",
    )
    print("AO progression probability calibration")
    print(f"Ties: {len(ties)}")
    print(f"Candidates: {len(candidates)}")
    print(f"Decision: {decision}")
    print(f"Full candidate: {full_candidate.key}")
    print(f"Report: {output_root / 'calibration_report.md'}")


def load_tie_probability_data(
    predictions_path: Path,
    events_path: Path,
) -> pd.DataFrame:
    predictions = pd.read_csv(predictions_path)
    required_predictions = {
        "fold",
        "match_id",
        "season",
        "competition",
        "tie_id",
        "expected_team_a_to_advance",
        "actual_team_a_advanced",
    }
    missing = sorted(required_predictions - set(predictions.columns))
    if missing:
        raise ValueError(f"Progression predictions missing columns: {missing}")
    ties = predictions.loc[
        predictions["expected_team_a_to_advance"].notna()
    ].copy()
    events = pd.read_csv(events_path)
    events, _ = normalize_exact_tie_deciders(events)
    knockout = events.loc[events["is_knockout"].astype(bool)].copy()
    knockout = knockout.sort_values(["season", "event_order"])
    metadata_rows = []
    for (season, tie_id), group in knockout.groupby(
        ["season", "tie_id"],
        sort=False,
    ):
        first = group.iloc[0]
        metadata_rows.append(
            {
                "season": str(season),
                "tie_id": str(tie_id),
                "tie_match_count": len(group),
                "first_match_neutral": bool(first["is_neutral"]),
                "team_a_id": int(first["home_team_id"]),
                "team_b_id": int(first["away_team_id"]),
                "first_kickoff_utc": str(first["kickoff_utc"]),
            }
        )
    metadata = pd.DataFrame(metadata_rows)
    ties["season"] = ties["season"].astype(str)
    ties["tie_id"] = ties["tie_id"].astype(str)
    ties = ties.merge(
        metadata,
        on=["season", "tie_id"],
        how="left",
        validate="one_to_one",
    )
    if ties[
        [
            "tie_match_count",
            "first_match_neutral",
            "team_a_id",
            "team_b_id",
        ]
    ].isna().any().any():
        raise ValueError("Tie probability metadata is incomplete")
    ties["tie_match_count"] = ties["tie_match_count"].astype(int)
    ties["first_match_neutral"] = ties["first_match_neutral"].astype(bool)
    ties["raw_probability"] = ties[
        "expected_team_a_to_advance"
    ].astype(float)
    ties["actual_advanced"] = ties["actual_team_a_advanced"].astype(float)
    if not ties["raw_probability"].between(0.0, 1.0).all():
        raise ValueError("Raw progression probability must be in [0,1]")
    if not ties["actual_advanced"].isin((0.0, 1.0)).all():
        raise ValueError("Tie outcome must be binary")
    if ties.duplicated(["season", "tie_id"]).any():
        raise ValueError("Tie probability dataset contains duplicate ties")
    return ties[
        [
            "fold",
            "match_id",
            "season",
            "competition",
            "tie_id",
            "tie_match_count",
            "first_match_neutral",
            "team_a_id",
            "team_b_id",
            "first_kickoff_utc",
            "raw_probability",
            "actual_advanced",
        ]
    ].sort_values(["season", "first_kickoff_utc", "tie_id"]).reset_index(
        drop=True
    )


def apply_config(
    data: pd.DataFrame,
    config: ProgressionProbabilityConfig,
) -> np.ndarray:
    config.validate()
    return np.array(
        [
            calibrate_progression_probability(
                float(probability),
                int(match_count),
                bool(neutral),
                config,
            )
            for probability, match_count, neutral in zip(
                data["raw_probability"],
                data["tie_match_count"],
                data["first_match_neutral"],
            )
        ],
        dtype=float,
    )


def evaluate_config(
    data: pd.DataFrame,
    config: ProgressionProbabilityConfig,
) -> dict[str, float | int | str]:
    probability = np.clip(apply_config(data, config), 1e-12, 1.0 - 1e-12)
    actual = data["actual_advanced"].to_numpy(float)
    return {
        "candidate_key": config.key,
        "logit_slope": config.logit_slope,
        "single_home_bias": config.single_home_bias,
        "two_leg_first_home_bias": config.two_leg_first_home_bias,
        "ties": len(data),
        "brier": float(np.mean((probability - actual) ** 2)),
        "log_loss": float(
            -np.mean(
                actual * np.log(probability)
                + (1.0 - actual) * np.log(1.0 - probability)
            )
        ),
        "accuracy": float(
            np.mean((probability >= 0.5).astype(float) == actual)
        ),
        "ece": expected_calibration_error(probability, actual),
    }


def evaluate_candidates(
    data: pd.DataFrame,
    candidates: tuple[ProgressionProbabilityConfig, ...],
) -> pd.DataFrame:
    return pd.DataFrame(
        [evaluate_config(data, candidate) for candidate in candidates]
    ).sort_values(
        ["log_loss", "brier", "ece", "candidate_key"]
    ).reset_index(drop=True)


def run_walk_forward(
    ties: pd.DataFrame,
    seasons: tuple[str, ...],
    candidates: tuple[ProgressionProbabilityConfig, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(seasons) < 2:
        raise ValueError("At least two seasons are required")
    identity = identity_progression_probability_config()
    selection_rows = []
    prediction_rows = []
    for index, test_season in enumerate(seasons):
        train_seasons = seasons[:index]
        train = ties.loc[ties["season"].isin(train_seasons)]
        selected = (
            identity
            if train.empty
            else config_from_row(evaluate_candidates(train, candidates).iloc[0])
        )
        selection_rows.append(
            {
                "fold": index + 1,
                "train_seasons": "|".join(train_seasons),
                "test_season": test_season,
                "training_ties": len(train),
                "selected_candidate": selected.key,
                "selected_logit_slope": selected.logit_slope,
                "selected_single_home_bias": selected.single_home_bias,
                "selected_two_leg_first_home_bias": (
                    selected.two_leg_first_home_bias
                ),
                "is_warmup_identity": train.empty,
            }
        )
        test = ties.loc[ties["season"].eq(test_season)].copy()
        for model, config in (
            ("CALIBRATED", selected),
            ("IDENTITY", identity),
        ):
            probability = np.clip(
                apply_config(test, config),
                1e-12,
                1.0 - 1e-12,
            )
            actual = test["actual_advanced"].to_numpy(float)
            model_rows = test.copy()
            model_rows["model"] = model
            model_rows["candidate_key"] = config.key
            model_rows["probability"] = probability
            model_rows["brier_loss"] = (probability - actual) ** 2
            model_rows["log_loss"] = -(
                actual * np.log(probability)
                + (1.0 - actual) * np.log(1.0 - probability)
            )
            prediction_rows.append(model_rows)
    return pd.DataFrame(selection_rows), pd.concat(
        prediction_rows,
        ignore_index=True,
    )


def summarize_models(predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        predictions.groupby("model", sort=True)
        .agg(
            ties=("tie_id", "size"),
            brier=("brier_loss", "mean"),
            log_loss=("log_loss", "mean"),
            accuracy=(
                "probability",
                lambda values: float(
                    np.mean(
                        (values.to_numpy(float) >= 0.5)
                        == predictions.loc[values.index, "actual_advanced"].to_numpy(
                            float
                        )
                    )
                ),
            ),
        )
        .reset_index()
    )


def segment_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    result = predictions.copy()
    result["format"] = np.where(
        result["tie_match_count"].eq(1),
        np.where(result["first_match_neutral"], "SINGLE_NEUTRAL", "SINGLE_HOME"),
        "TWO_LEG",
    )
    return (
        result.groupby(["model", "competition", "format"], sort=True)
        .agg(
            ties=("tie_id", "size"),
            brier=("brier_loss", "mean"),
            log_loss=("log_loss", "mean"),
        )
        .reset_index()
    )


def build_uncertainty(
    predictions: pd.DataFrame,
    *,
    bootstrap_samples: int,
) -> pd.DataFrame:
    calibrated = predictions.loc[
        predictions["model"].eq("CALIBRATED")
    ].set_index(["season", "tie_id"])
    identity = predictions.loc[
        predictions["model"].eq("IDENTITY")
    ].set_index(["season", "tie_id"])
    common = calibrated.index.intersection(identity.index)
    if len(common) != len(calibrated) or len(common) != len(identity):
        raise ValueError("Progression uncertainty sample must be paired")
    rows = []
    for loss in ("brier_loss", "log_loss"):
        paired = calibrated.loc[common].reset_index()[
            [
                "match_id",
                "season",
                "team_a_id",
                "team_b_id",
                "first_kickoff_utc",
                "tie_id",
            ]
        ]
        paired = paired.rename(
            columns={
                "team_a_id": "home_team_id",
                "team_b_id": "away_team_id",
                "first_kickoff_utc": "kickoff_utc",
            }
        )
        paired["loss_difference"] = (
            calibrated.loc[common, loss].to_numpy(float)
            - identity.loc[common, loss].to_numpy(float)
        )
        uncertainty = dependency_robust_loss_difference_ci(
            paired,
            bootstrap_samples=bootstrap_samples,
        )
        uncertainty["loss"] = loss
        rows.append(uncertainty)
    return pd.concat(rows, ignore_index=True)


def make_decision(
    selections: pd.DataFrame,
    comparison: pd.DataFrame,
    segments: pd.DataFrame,
    uncertainty: pd.DataFrame,
) -> tuple[str, dict[str, object]]:
    indexed = comparison.set_index("model")
    calibrated = indexed.loc["CALIBRATED"]
    identity = indexed.loc["IDENTITY"]
    evaluable = selections.loc[~selections["is_warmup_identity"]]
    nonidentity_share = float(
        evaluable["selected_candidate"].ne(
            identity_progression_probability_config().key
        ).mean()
    )
    ci = uncertainty.loc[
        uncertainty["method"].eq("conservative_envelope")
        & uncertainty["loss"].eq("log_loss")
    ]
    pivot = segments.pivot_table(
        index=["competition", "format"],
        columns="model",
        values=["brier", "log_loss"],
    )
    no_segment_harm = bool(
        (
            pivot[("brier", "CALIBRATED")]
            <= pivot[("brier", "IDENTITY")] + 1e-12
        ).all()
        and (
            pivot[("log_loss", "CALIBRATED")]
            <= pivot[("log_loss", "IDENTITY")] + 1e-12
        ).all()
    )
    guardrails = {
        "brier_delta": float(calibrated["brier"] - identity["brier"]),
        "log_loss_delta": float(
            calibrated["log_loss"] - identity["log_loss"]
        ),
        "accuracy_delta": float(
            calibrated["accuracy"] - identity["accuracy"]
        ),
        "nonidentity_fold_share": nonidentity_share,
        "clustered_log_loss_reliable_improvement": bool(
            len(ci) == 1 and float(ci.iloc[0]["ci_95_upper"]) < 0.0
        ),
        "no_competition_format_regression": no_segment_harm,
    }
    passed = (
        guardrails["brier_delta"] < 0.0
        and guardrails["log_loss_delta"] < 0.0
        and nonidentity_share >= 0.5
        and guardrails["clustered_log_loss_reliable_improvement"]
        and no_segment_harm
    )
    return (
        "PROMOTE_PROGRESSION_PROBABILITY"
        if passed
        else "KEEP_IDENTITY_PROGRESSION_PROXY"
    ), guardrails


def expected_calibration_error(
    probability: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
    value = 0.0
    for index in range(bins):
        mask = indices == index
        if mask.any():
            value += float(mask.mean()) * abs(
                float(probability[mask].mean()) - float(actual[mask].mean())
            )
    return value


def config_from_row(row: pd.Series) -> ProgressionProbabilityConfig:
    return ProgressionProbabilityConfig(
        float(row["logit_slope"]),
        float(row["single_home_bias"]),
        float(row["two_leg_first_home_bias"]),
    )


def config_payload(
    config: ProgressionProbabilityConfig,
) -> dict[str, float | str]:
    return {
        "key": config.key,
        "logit_slope": config.logit_slope,
        "single_home_bias": config.single_home_bias,
        "two_leg_first_home_bias": config.two_leg_first_home_bias,
    }


def build_report(
    decision: str,
    ties: pd.DataFrame,
    selections: pd.DataFrame,
    comparison: pd.DataFrame,
    segments: pd.DataFrame,
    uncertainty: pd.DataFrame,
    full_candidate: ProgressionProbabilityConfig,
    guardrails: dict[str, object],
) -> str:
    return f"""# AO Progression Probability Calibration

## Decision

**{decision}**

The current neutral single-match Elo proxy is compared with a calibrated,
format-aware progression probability:

```text
logit(P_advance) =
    slope x logit(P_raw)
    + single_home_bias
    + two_leg_first_home_bias
```

Only one applicable format bias is used. Neutral single matches receive no
home bias. All parameters are selected from earlier seasons only.

## Data

- Cross-fitted tie outcomes: {len(ties)}
- Seasons: {ties["season"].min()} through {ties["season"].max()}
- Single-match ties: {int(ties["tie_match_count"].eq(1).sum())}
- Two-leg ties: {int(ties["tie_match_count"].ge(2).sum())}

## Full-Data Diagnostic Candidate

`{full_candidate.key}`

## Walk-Forward Selections

```csv
{selections.to_csv(index=False).strip()}
```

## Model Comparison

```csv
{comparison.to_csv(index=False).strip()}
```

## Competition and Format

```csv
{segments.to_csv(index=False).strip()}
```

## Dependency Uncertainty

```csv
{uncertainty.to_csv(index=False).strip()}
```

## Guardrails

```json
{json.dumps(guardrails, indent=2)}
```

Achievement Reserve may be re-tested only if this probability layer is
promoted. Otherwise the previous reserve rejection remains binding.
"""


if __name__ == "__main__":
    main()
