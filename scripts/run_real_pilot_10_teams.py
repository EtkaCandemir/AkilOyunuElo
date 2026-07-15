from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ao_elo import AOEuropeanEloConfig, compute_ao_first_elo_from_csv  # noqa: E402


DATA_DIR = ROOT / "data" / "real_pilot_10_teams"
OUTPUT_DIR = ROOT / "output" / "real_pilot_10_teams"
OUTPUT_CSV = OUTPUT_DIR / "ao_first_elo_real_pilot_output.csv"
REVIEW_CSV = OUTPUT_DIR / "real_pilot_review_table.csv"
REPORT_MD = OUTPUT_DIR / "real_pilot_results.md"

EXPECTED_AO_FIRST_ELO = {
    "Arsenal": 902.0,
    "Sporting CP": 884.1846401071498,
    "Benfica": 871.9657625081588,
    "Shakhtar Donetsk": 840.554170070006,
    "Galatasaray": 838.2916113102538,
    "AZ Alkmaar": 834.3930531190413,
    "Slavia Praha": 802.093360239539,
    "Pafos": 754.2098847131747,
    "Como": 748.1243760474089,
    "Omonia Nicosia": 741.9251186948296,
}


def main() -> None:
    config = AOEuropeanEloConfig.v1_1()
    candidate_config = AOEuropeanEloConfig.experimental_country_candidate()

    run_input_audit()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = compute_ao_first_elo_from_csv(
        teams_csv=DATA_DIR / "teams.csv",
        country_coefficients_csv=DATA_DIR / "country_coefficients.csv",
        domestic_context_csv=DATA_DIR / "domestic_context.csv",
        club_european_points_csv=DATA_DIR / "club_european_points.csv",
        config=config,
        output_csv=OUTPUT_CSV,
    )
    candidate_output = compute_ao_first_elo_from_csv(
        teams_csv=DATA_DIR / "teams.csv",
        country_coefficients_csv=DATA_DIR / "country_coefficients.csv",
        domestic_context_csv=DATA_DIR / "domestic_context.csv",
        club_european_points_csv=DATA_DIR / "club_european_points.csv",
        config=candidate_config,
    )

    run_smoke_checks(output)
    review = build_review_table(output, candidate_output)
    review.to_csv(REVIEW_CSV, index=False)
    REPORT_MD.write_text(build_report(output, review), encoding="utf-8")
    print_summary(output, review)


def run_input_audit() -> None:
    """Cross-check source totals before passing model-ready data to the engine."""
    country = pd.read_csv(DATA_DIR / "country_coefficients.csv")
    country_season_columns = [
        "points_t_minus_4",
        "points_t_minus_3",
        "points_t_minus_2",
        "points_t_minus_1",
        "points_t",
    ]
    calculated_country_totals = country[country_season_columns].sum(axis=1)
    assert (
        calculated_country_totals.sub(country["official_five_year_total"]).abs()
        < 0.0015
    ).all(), "country season points must match the official five-year total"

    club = pd.read_csv(DATA_DIR / "club_european_points.csv")
    club_season_columns = [
        "club_points_t_minus_4",
        "club_points_t_minus_3",
        "club_points_t_minus_2",
        "club_points_t_minus_1",
        "club_points_t",
    ]
    own_points = club[club_season_columns].sum(axis=1)
    expected_official = pd.concat([own_points, club["country_part"]], axis=1).max(axis=1)
    assert (
        expected_official.sub(club["official_club_coefficient"]).abs() < 0.0015
    ).all(), "official club coefficient must match own points or the country floor"


def run_smoke_checks(output: pd.DataFrame) -> None:
    assert len(output) == 10, "real pilot output must contain exactly 10 teams"
    assert output["validation_warnings"].fillna("").eq("").all()
    assert output["ao_first_elo"].is_monotonic_decreasing
    assert output["ao_first_elo_rank"].tolist() == list(range(1, 11))

    expected_sources = {
        "Pure Domestic Projection",
        "Mixed Domestic-European Estimate",
        "European Evidence-Based Rating",
    }
    assert expected_sources <= set(output["rating_source_type"])

    como = output.loc[output["team_name"] == "Como"].iloc[0]
    assert como["european_exposure"] == 0
    assert como["effective_european_exposure"] == 0
    assert como["ao_first_elo"] == como["domestic_prior"]
    arsenal = output.loc[output["team_name"] == "Arsenal"].iloc[0]
    assert como["ao_first_elo"] < arsenal["ao_first_elo"]
    assert output.loc[output["ao_first_elo"].idxmax(), "team_name"] == "Arsenal"

    pafos = output.loc[output["team_name"] == "Pafos"].iloc[0]
    assert abs(pafos["european_exposure"] - 0.60) < 1e-12
    assert pafos["rating_source_type"] == "Mixed Domestic-European Estimate"

    evidence = output.loc[output["team_name"] == "Benfica"].iloc[0]
    assert evidence["european_exposure"] == 1.0
    assert evidence["effective_european_exposure"] == 0.85

    for team_name in ("AZ Alkmaar", "Pafos"):
        cup_winner = output.loc[output["team_name"] == team_name].iloc[0]
        assert cup_winner["cup_base_score"] > 0
        assert cup_winner["cup_double_bonus"] == 0

    actual_ratings = output.set_index("team_name")["ao_first_elo"]
    for team_name, expected_rating in EXPECTED_AO_FIRST_ELO.items():
        assert abs(float(actual_ratings[team_name]) - expected_rating) < 1e-9


def build_review_table(
    output: pd.DataFrame,
    candidate_output: pd.DataFrame,
) -> pd.DataFrame:
    review = output[
        [
            "team_name",
            "country_code",
            "competition",
            "domestic_prior",
            "european_prior",
            "european_exposure",
            "effective_european_exposure",
            "ao_first_elo",
            "ao_first_elo_rank",
            "rating_source_type",
        ]
    ].copy()
    candidate_ratings = candidate_output.set_index("team_id")["ao_first_elo"]
    review["country_candidate_ao_first_elo"] = output["team_id"].map(
        candidate_ratings
    )
    review["candidate_delta"] = (
        review["country_candidate_ao_first_elo"] - review["ao_first_elo"]
    )
    review["candidate_rank"] = review["country_candidate_ao_first_elo"].rank(
        method="min",
        ascending=False,
    ).astype(int)
    review["european_adjustment"] = review["ao_first_elo"] - review["domestic_prior"]
    review["rank"] = review["ao_first_elo_rank"]
    review = review.sort_values(["rank", "team_name"])

    numeric_columns = [
        "domestic_prior",
        "european_prior",
        "european_exposure",
        "effective_european_exposure",
        "country_candidate_ao_first_elo",
        "candidate_delta",
        "ao_first_elo",
        "european_adjustment",
    ]
    review[numeric_columns] = review[numeric_columns].round(3)
    return review[
        [
            "rank",
            "team_name",
            "country_code",
            "competition",
            "domestic_prior",
            "european_prior",
            "european_exposure",
            "effective_european_exposure",
            "country_candidate_ao_first_elo",
            "candidate_delta",
            "candidate_rank",
            "european_adjustment",
            "ao_first_elo",
            "rating_source_type",
        ]
    ]


def build_report(output: pd.DataFrame, review: pd.DataFrame) -> str:
    source_counts = output["rating_source_type"].value_counts()
    lines = [
        "# AO European Elo v1.1 - Real 10-Team Pilot Results",
        "",
        "Data freeze: 2026-07-13",
        "",
        "Target season: `2026/27`",
        "",
        "Active model parameters: `Country_Strength_Benchmark=25`, `gamma=0.8`, "
        "`domestic_league_component=140`, `European_History_Benchmark=20`.",
        "",
        "> The `20 / 2.0 / 360` country candidate is rejected for production because "
        "it gives implausibly dominant ratings to zero-exposure clubs.",
        "",
        "## Summary",
        "",
        f"- Teams: {len(output)}",
        f"- Minimum AO First Elo: {output['ao_first_elo'].min():.3f}",
        f"- Maximum AO First Elo: {output['ao_first_elo'].max():.3f}",
        f"- Validation warning rows: "
        f"{int(output['validation_warnings'].fillna('').ne('').sum())}",
    ]
    for source_type, count in source_counts.items():
        lines.append(f"- {source_type}: {count}")

    lines.extend(
        [
            "",
            "## Ranking And Components",
            "",
            dataframe_to_markdown(review),
            "",
            "## Review Notes",
            "",
            "- Como is the no-history control: exposure is zero and the final "
            "rating equals the domestic prior.",
            "- Pafos is the recent-history control: only the latest two seasons "
            "carry evidence, so the European prior is partially blended.",
            "- Raw exposure measures evidence and still drives the source category; "
            "effective exposure is capped at `0.85` only for the final blend.",
            "- A raw exposure of `1.0` therefore preserves a `15%` Domestic Prior "
            "contribution instead of replacing it completely.",
            "- The active model retains the frozen v1.1 country values `25 / 0.8 / 140`; "
            "European History Benchmark `20` is frozen for the static v1.1 release "
            "after the eight-season ranking-first review.",
            "- `country_candidate_ao_first_elo` shows the rejected `20 / 2.0 / 360` "
            "candidate. Como rises to 978.316 under that candidate, which fails the "
            "zero-exposure plausibility guardrail.",
            "- Competition level is metadata today. Benfica can rank close to "
            "Arsenal despite entering the UEL because UCL/UEL/UECL do not alter "
            "the current formula.",
            "- Como ranks above Omonia because zero exposure preserves Como's "
            "strong Italian domestic prior, while Omonia's extensive European "
            "evidence pulls it toward a lower European prior.",
            "- Positive or negative `european_adjustment` shows the direction of "
            "shrinkage from domestic evidence toward European evidence.",
            "- AZ Alkmaar and Pafos receive the cup base score but no double bonus; "
            "the double bonus is reserved for league-and-cup champions.",
            "",
            "## Provenance",
            "",
            "See `data/real_pilot_10_teams/SOURCES.md`.",
            "",
        ]
    )
    return "\n".join(lines)


def dataframe_to_markdown(dataframe: pd.DataFrame) -> str:
    """Render a compact Markdown table without an optional tabulate dependency."""
    headers = [str(column) for column in dataframe.columns]
    rows = [
        [str(value).replace("|", "\\|") for value in row]
        for row in dataframe.itertuples(index=False, name=None)
    ]
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def print_summary(output: pd.DataFrame, review: pd.DataFrame) -> None:
    print("AO European Elo real 10-team pilot")
    print("Input source-total audit: passed")
    print(f"Rows: {len(output)}")
    print("Rating source distribution:")
    for source_type, count in output["rating_source_type"].value_counts().items():
        print(f"  - {source_type}: {count}")
    print(f"AO First Elo min/max: {output['ao_first_elo'].min():.3f} / "
          f"{output['ao_first_elo'].max():.3f}")
    print("Top 3:")
    for row in review.head(3).itertuples(index=False):
        print(f"  {row.rank}. {row.team_name}: {row.ao_first_elo:.3f}")
    print(f"Output CSV: {OUTPUT_CSV}")
    print(f"Review CSV: {REVIEW_CSV}")
    print(f"Report: {REPORT_MD}")


if __name__ == "__main__":
    main()
