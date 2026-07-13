from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ao_elo import AOEuropeanEloConfig, compute_ao_first_elo_from_csv  # noqa: E402


DATA_DIR = ROOT / "data" / "pilot_10_teams"
OUTPUT_DIR = ROOT / "output" / "pilot_10_teams"
OUTPUT_CSV = OUTPUT_DIR / "ao_first_elo_pilot_output.csv"

EXPECTED_AO_FIRST_ELO = {
    "Metro Albion": 749.011464500652,
    "Midoria Champions": 752.2551037650705,
    "Smallia Kings": 706.4507259676069,
    "Cupmark Rangers": 679.8501411222367,
    "Few Match Wanderers": 763.010604894901,
    "Continental Giants": 901.1654490536773,
    "Low Score Veterans": 555.9349943673824,
    "Last Season Sparks": 712.4945092732662,
    "Distant History FC": 685.0553451963483,
    "Double Crown Athletic": 791.3687425343417,
}


def main() -> None:
    config = AOEuropeanEloConfig(
        country_strength_benchmark=25,
        european_history_benchmark=20,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = compute_ao_first_elo_from_csv(
        teams_csv=DATA_DIR / "teams.csv",
        country_coefficients_csv=DATA_DIR / "country_coefficients.csv",
        domestic_context_csv=DATA_DIR / "domestic_context.csv",
        club_european_points_csv=DATA_DIR / "club_european_points.csv",
        config=config,
        output_csv=OUTPUT_CSV,
    )

    run_smoke_checks(output)
    print_summary(output)
    print(f"Output CSV: {OUTPUT_CSV}")


def run_smoke_checks(output: pd.DataFrame) -> None:
    assert len(output) == 10, "pilot output must contain exactly 10 teams"

    expected_sources = {
        "Pure Domestic Projection",
        "Mixed Domestic-European Estimate",
        "European Evidence-Based Rating",
    }
    actual_sources = set(output["rating_source_type"])
    missing_sources = expected_sources - actual_sources
    assert not missing_sources, f"missing rating source type(s): {missing_sources}"

    no_europe = output.loc[output["team_name"] == "Metro Albion"].iloc[0]
    assert no_europe["european_exposure"] == 0
    assert no_europe["ao_first_elo"] == no_europe["domestic_prior"]

    few_matches = output.loc[output["team_name"] == "Few Match Wanderers"].iloc[0]
    assert abs(few_matches["weighted_match_exposure"] - 0.2833333333) < 1e-9
    assert abs(few_matches["european_exposure"] - 0.7133333333) < 1e-9

    full_exposure = output.loc[output["team_name"] == "Continental Giants"].iloc[0]
    assert full_exposure["european_exposure"] == 1.0

    unknown_cup = output.loc[output["team_name"] == "Cupmark Rangers"].iloc[0]
    assert unknown_cup["league_finish_score"] == 0.10
    assert unknown_cup["cup_double_bonus"] == 0

    actual_ratings = output.set_index("team_name")["ao_first_elo"].to_dict()
    for team_name, expected_rating in EXPECTED_AO_FIRST_ELO.items():
        actual_rating = float(actual_ratings[team_name])
        assert abs(actual_rating - expected_rating) < 1e-9, (
            f"pilot rating changed for {team_name}: "
            f"expected {expected_rating}, got {actual_rating}"
        )


def print_summary(output: pd.DataFrame) -> None:
    warning_count = int(output["validation_warnings"].fillna("").ne("").sum())
    print("AO European Elo pilot smoke test")
    print(f"Rows: {len(output)}")
    print("Rating source distribution:")
    for source_type, count in output["rating_source_type"].value_counts().items():
        print(f"  - {source_type}: {count}")
    print(f"AO First Elo min: {output['ao_first_elo'].min():.3f}")
    print(f"AO First Elo max: {output['ao_first_elo'].max():.3f}")
    print(f"Validation warning rows: {warning_count}")


if __name__ == "__main__":
    main()
