from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output" / "pilot_10_teams"
SOURCE_CSV = OUTPUT_DIR / "ao_first_elo_pilot_output.csv"
INSPECTION_CSV = OUTPUT_DIR / "pilot_inspection_table.csv"
REPORT_MD = OUTPUT_DIR / "pilot_inspection_report.md"


SCENARIOS = {
    "Metro Albion": "Big league, no European history",
    "Midoria Champions": "Medium league champion, no European history",
    "Smallia Kings": "Small league champion, no European history",
    "Cupmark Rangers": "Unknown league finish plus cup winner",
    "Few Match Wanderers": "Five seasons with low match volume",
    "Continental Giants": "Five seasons with full evidence",
    "Low Score Veterans": "High exposure with weak European points",
    "Last Season Sparks": "Only latest season in Europe",
    "Distant History FC": "Older European history, no recent seasons",
    "Double Crown Athletic": "League and cup double winner",
}


INSPECTION_COLUMNS = [
    "scenario",
    "team_name",
    "competition",
    "entry_round",
    "domestic_prior",
    "european_prior",
    "european_exposure",
    "effective_european_exposure",
    "ao_first_elo",
    "rating_source_type",
    "weighted_match_exposure",
    "weighted_european_history",
    "domestic_achievement_score",
    "cup_double_bonus",
    "validation_warnings",
]


def main() -> None:
    output = pd.read_csv(SOURCE_CSV)
    inspection = build_inspection_table(output)
    inspection.to_csv(INSPECTION_CSV, index=False)
    REPORT_MD.write_text(build_markdown_report(output, inspection), encoding="utf-8")

    print(f"Inspection CSV: {INSPECTION_CSV}")
    print(f"Markdown report: {REPORT_MD}")


def build_inspection_table(output: pd.DataFrame) -> pd.DataFrame:
    table = output.copy()
    table["scenario"] = table["team_name"].map(SCENARIOS)
    missing = table.loc[table["scenario"].isna(), "team_name"].tolist()
    if missing:
        raise ValueError(f"Missing scenario labels for team(s): {missing}")

    table = table[INSPECTION_COLUMNS].copy()
    numeric_cols = table.select_dtypes(include=["number"]).columns
    table[numeric_cols] = table[numeric_cols].round(3)
    return table.sort_values("ao_first_elo", ascending=False).reset_index(drop=True)


def build_markdown_report(output: pd.DataFrame, inspection: pd.DataFrame) -> str:
    warning_rows = int(output["validation_warnings"].fillna("").ne("").sum())
    source_counts = output["rating_source_type"].value_counts().to_dict()
    highest = inspection.iloc[0]
    lowest = inspection.iloc[-1]

    checks = [
        acceptance_check(
            "10 teams produced",
            len(output) == 10,
            f"rows={len(output)}",
        ),
        acceptance_check(
            "All rating source categories present",
            {
                "Pure Domestic Projection",
                "Mixed Domestic-European Estimate",
                "European Evidence-Based Rating",
            }.issubset(set(output["rating_source_type"])),
            ", ".join(sorted(output["rating_source_type"].unique())),
        ),
        acceptance_check(
            "No Europe team stays at Domestic Prior",
            is_equal(
                output,
                "Metro Albion",
                "ao_first_elo",
                "domestic_prior",
            ),
            "Metro Albion",
        ),
        acceptance_check(
            "Low match volume exposure is about 0.713",
            value_close(output, "Few Match Wanderers", "european_exposure", 0.7133333333),
            metric_value(output, "Few Match Wanderers", "european_exposure"),
        ),
        acceptance_check(
            "Full evidence team exposure is 1.0",
            value_close(output, "Continental Giants", "european_exposure", 1.0),
            metric_value(output, "Continental Giants", "european_exposure"),
        ),
        acceptance_check(
            "Effective exposure is capped at 0.85",
            value_close(
                output,
                "Continental Giants",
                "effective_european_exposure",
                0.85,
            ),
            metric_value(
                output,
                "Continental Giants",
                "effective_european_exposure",
            ),
        ),
        acceptance_check(
            "Unknown finish plus cup gets no double bonus",
            value_close(output, "Cupmark Rangers", "cup_double_bonus", 0.0),
            metric_value(output, "Cupmark Rangers", "cup_double_bonus"),
        ),
    ]

    lines = [
        "# AO European Elo Pilot Inspection Report",
        "",
        "This report is generated from the synthetic 10-team pilot output. "
        "It is a smoke-test artifact, not a calibrated production result.",
        "",
        "## Summary",
        "",
        f"- Rows: {len(output)}",
        f"- AO First Elo min: {output['ao_first_elo'].min():.3f}",
        f"- AO First Elo max: {output['ao_first_elo'].max():.3f}",
        f"- Validation warning rows: {warning_rows}",
        "",
        "## Rating Source Distribution",
        "",
    ]
    for source_type, count in source_counts.items():
        lines.append(f"- {source_type}: {count}")

    lines.extend(
        [
            "",
            "## Acceptance Checks",
            "",
            *checks,
            "",
            "## Readout",
            "",
            f"- Highest AO First Elo: {highest['team_name']} ({highest['ao_first_elo']:.3f})",
            f"- Lowest AO First Elo: {lowest['team_name']} ({lowest['ao_first_elo']:.3f})",
            "- `Low Score Veterans` shows the intended downward pull while the "
            "effective exposure cap retains a 15% Domestic Prior contribution.",
            "- `Few Match Wanderers` shows limited trust in a five-season but low-match sample.",
            "- `Double Crown Athletic` confirms the cup double bonus path.",
            "",
            "## Inspection Table",
            "",
            dataframe_to_markdown(inspection),
            "",
        ]
    )
    return "\n".join(lines)


def acceptance_check(label: str, passed: bool, detail: str) -> str:
    status = "PASS" if passed else "FAIL"
    return f"- {status}: {label} ({detail})"


def team_row(output: pd.DataFrame, team_name: str) -> pd.Series:
    return output.loc[output["team_name"] == team_name].iloc[0]


def value_close(
    output: pd.DataFrame,
    team_name: str,
    column: str,
    expected: float,
    tolerance: float = 1e-9,
) -> bool:
    return abs(float(team_row(output, team_name)[column]) - expected) < tolerance


def is_equal(
    output: pd.DataFrame,
    team_name: str,
    left_column: str,
    right_column: str,
) -> bool:
    row = team_row(output, team_name)
    return float(row[left_column]) == float(row[right_column])


def metric_value(output: pd.DataFrame, team_name: str, column: str) -> str:
    return f"{team_name} {column}={float(team_row(output, team_name)[column]):.3f}"


def dataframe_to_markdown(dataframe: pd.DataFrame) -> str:
    columns = list(dataframe.columns)
    rows = [[format_cell(value) for value in row] for row in dataframe.to_numpy()]
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def format_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
