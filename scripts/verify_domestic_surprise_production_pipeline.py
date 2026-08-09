from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ao_elo.config import AOEuropeanEloConfig  # noqa: E402
from ao_elo.pipeline import compute_ao_first_elo  # noqa: E402


SEASON = "2025/26"
STATIC_ROOT = ROOT / "data" / "backtest_stage_b_2018_2026" / "2025-26"
FEATURES_PATH = (
    ROOT
    / "output"
    / "domestic_surprise_variance_backtest_2018_2026"
    / "domestic_surprise_features.csv"
)
REFERENCE_PATH = (
    ROOT
    / "output"
    / "domestic_surprise_variance_backtest_2018_2026"
    / "gamma_sensitivity_cap_30"
    / "selected_candidate_team_adjustments.csv"
)
OUTPUT_ROOT = ROOT / "output" / "domestic_surprise_production_verification_2025_26"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify integrated Domestic Surprise against its frozen shadow snapshot"
    )
    parser.add_argument("--static-root", type=Path, default=STATIC_ROOT)
    parser.add_argument("--features", type=Path, default=FEATURES_PATH)
    parser.add_argument("--reference", type=Path, default=REFERENCE_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    static_root = args.static_root.resolve()
    domestic = pd.read_csv(static_root / "domestic_context.csv")
    features = pd.read_csv(args.features.resolve())
    features = features.loc[features["season"].astype(str).eq(SEASON)].copy()
    history_columns = [
        column
        for column in features.columns
        if column.startswith("history_position_")
        or column.startswith("history_team_count_")
    ]
    domestic = domestic.merge(
        features[["team_id", *history_columns]],
        on="team_id",
        how="left",
        validate="one_to_one",
    )
    for offset in range(5, 0, -1):
        position_column = f"history_position_t_minus_{offset}"
        count_column = f"history_team_count_t_minus_{offset}"
        incomplete = domestic[position_column].isna() | domestic[count_column].isna()
        domestic.loc[incomplete, [position_column, count_column]] = pd.NA
    output = compute_ao_first_elo(
        teams=pd.read_csv(static_root / "teams.csv"),
        country_coefficients=pd.read_csv(static_root / "country_coefficients.csv"),
        domestic_context=domestic,
        club_european_points=pd.read_csv(static_root / "club_european_points.csv"),
        config=AOEuropeanEloConfig.active(),
    )
    reference = pd.read_csv(args.reference.resolve())
    reference = reference.loc[reference["season"].astype(str).eq(SEASON)].copy()
    comparison = output.merge(
        reference[
            [
                "team_id",
                "adjusted_ao_first_elo",
                "domestic_prior_adjustment",
                "ao_first_elo_adjustment",
            ]
        ].rename(
            columns={
                "adjusted_ao_first_elo": "reference_ao_first_elo",
                "domestic_prior_adjustment": "reference_domestic_adjustment",
                "ao_first_elo_adjustment": "reference_ao_adjustment",
            }
        ),
        on="team_id",
        how="inner",
        validate="one_to_one",
    )
    if len(comparison) != 236:
        raise ValueError("Production verification requires 236 matched teams")
    comparison["ao_first_elo_error"] = (
        comparison["ao_first_elo"] - comparison["reference_ao_first_elo"]
    )
    comparison["domestic_adjustment_error"] = (
        comparison["domestic_surprise_domestic_adjustment"]
        - comparison["reference_domestic_adjustment"]
    )
    comparison["ao_adjustment_error"] = (
        comparison["domestic_surprise_ao_first_elo_adjustment"]
        - comparison["reference_ao_adjustment"]
    )
    max_error = float(
        comparison[
            [
                "ao_first_elo_error",
                "domestic_adjustment_error",
                "ao_adjustment_error",
            ]
        ].abs().to_numpy().max()
    )
    if max_error > 1e-9:
        raise ValueError(f"Integrated pipeline differs from shadow snapshot: {max_error}")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    domestic.to_csv(output_root / "domestic_context_with_history.csv", index=False)
    output.to_csv(output_root / "integrated_ao_first_elo.csv", index=False)
    comparison.to_csv(output_root / "pipeline_shadow_comparison.csv", index=False)
    summary = {
        "season": SEASON,
        "teams": len(comparison),
        "applied": int(output["domestic_surprise_status"].eq("APPLIED").sum()),
        "insufficient_history": int(
            output["domestic_surprise_status"].eq("INSUFFICIENT_HISTORY").sum()
        ),
        "max_absolute_pipeline_shadow_error": max_error,
        "mean_absolute_ao_adjustment": float(
            output["domestic_surprise_ao_first_elo_adjustment"].abs().mean()
        ),
        "max_positive_ao_adjustment": float(
            output["domestic_surprise_ao_first_elo_adjustment"].max()
        ),
        "max_negative_ao_adjustment": float(
            output["domestic_surprise_ao_first_elo_adjustment"].min()
        ),
    }
    (output_root / "verification.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Output: {output_root}")


if __name__ == "__main__":
    main()
