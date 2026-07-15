from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_achievement_carry_calibration import (  # noqa: E402
    AchievementCarryConfig,
    competition_phase,
    evaluate_sequence,
    load_carry_data,
)
from scripts.run_dynamic_core_calibration import DynamicCoreConfig  # noqa: E402


STATIC_DATA_ROOT = ROOT / "data" / "backtest_stage_b_2018_2026"
EXACT_EVENTS_PATH = (
    ROOT / "data" / "external_elo_benchmark_2018_2026" / "exact_date_events.csv"
)
EXTERNAL_PREDICTIONS_PATH = (
    ROOT / "output" / "external_elo_benchmark_2018_2026" / "paired_predictions.csv"
)
OUTPUT_ROOT = ROOT / "output" / "ucl_external_diagnostics_2018_2026"
CURRENT_CORE = DynamicCoreConfig(225.0, 40.0, 28.0)


@dataclass(frozen=True)
class AOVariant:
    name: str
    core: DynamicCoreConfig
    carry: AchievementCarryConfig


VARIANTS = (
    AOVariant(
        "static_start_only",
        DynamicCoreConfig(225.0, 40.0, 0.0),
        AchievementCarryConfig(0.0, 0.0, 0.0),
    ),
    AOVariant(
        "dynamic_season_reset",
        CURRENT_CORE,
        AchievementCarryConfig(0.0, 0.0, 0.0),
    ),
    AOVariant(
        "dynamic_carry_050",
        CURRENT_CORE,
        AchievementCarryConfig(0.50, 0.0, 0.0),
    ),
    AOVariant(
        "dynamic_carry_085_current",
        CURRENT_CORE,
        AchievementCarryConfig(0.85, 0.0, 0.0),
    ),
    AOVariant(
        "dynamic_carry_100",
        CURRENT_CORE,
        AchievementCarryConfig(1.0, 0.0, 0.0),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose the AO-vs-ClubElo UCL gap by isolated AO layer ablations"
    )
    parser.add_argument("--static-data-root", type=Path, default=STATIC_DATA_ROOT)
    parser.add_argument("--exact-events-path", type=Path, default=EXACT_EVENTS_PATH)
    parser.add_argument(
        "--external-predictions-path",
        type=Path,
        default=EXTERNAL_PREDICTIONS_PATH,
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    datasets, _ = load_carry_data(
        args.static_data_root.resolve(),
        args.exact_events_path.resolve(),
    )
    external = load_external_predictions(args.external_predictions_path.resolve())
    predictions, guardrails = evaluate_variants(datasets, external)
    model_summary = summarize_models(predictions)
    ucl = predictions.loc[predictions["competition"].eq("UCL")].copy()
    ucl_model_summary = summarize_models(ucl)
    ucl_season_summary = summarize_by_group(ucl, "test_season")
    ucl_phase_summary = summarize_by_group(ucl, "phase")
    uncertainty = incremental_uncertainty(ucl)
    rank_alignment = rating_alignment(ucl)
    club_rank_discrepancies = rank_discrepancies(ucl)
    disagreement = favorite_disagreement(ucl)
    decision = diagnostic_decision(ucl_model_summary, uncertainty, guardrails)

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_root / "paired_variant_predictions.csv", index=False)
    model_summary.to_csv(output_root / "all_competition_model_summary.csv", index=False)
    ucl_model_summary.to_csv(output_root / "ucl_model_summary.csv", index=False)
    ucl_season_summary.to_csv(output_root / "ucl_season_summary.csv", index=False)
    ucl_phase_summary.to_csv(output_root / "ucl_phase_summary.csv", index=False)
    uncertainty.to_csv(output_root / "ucl_incremental_uncertainty.csv", index=False)
    rank_alignment.to_csv(output_root / "ucl_rating_alignment.csv", index=False)
    club_rank_discrepancies.to_csv(
        output_root / "ucl_club_rank_discrepancies.csv", index=False
    )
    disagreement.to_csv(output_root / "ucl_favorite_disagreement.csv", index=False)
    guardrails.to_csv(output_root / "variant_guardrails.csv", index=False)
    write_report(
        output_root / "diagnostic_report.md",
        ucl,
        ucl_model_summary,
        ucl_season_summary,
        ucl_phase_summary,
        uncertainty,
        rank_alignment,
        club_rank_discrepancies,
        disagreement,
        guardrails,
        decision,
    )

    current = model_row(ucl_model_summary, "dynamic_carry_085_current")
    reset = model_row(ucl_model_summary, "dynamic_season_reset")
    static = model_row(ucl_model_summary, "static_start_only")
    external_row = model_row(ucl_model_summary, "clubelo_external")
    print("AO UCL external benchmark diagnostics")
    print(f"UCL unseen paired matches: {len(ucl)}")
    print(f"Static start Brier: {static['brier']:.6f}")
    print(f"Dynamic reset Brier: {reset['brier']:.6f}")
    print(f"Current 0.85 carry Brier: {current['brier']:.6f}")
    print(f"ClubElo Brier: {external_row['brier']:.6f}")
    print(f"Decision: {decision}")
    print(f"Report: {output_root / 'diagnostic_report.md'}")


def load_external_predictions(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {
        "match_id",
        "fold",
        "test_season",
        "competition",
        "round",
        "actual_home_score",
        "clubelo_home_elo",
        "clubelo_away_elo",
        "clubelo_expected_home_score",
        "clubelo_brier_loss",
        "clubelo_log_loss",
        "ao_expected_home_score",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"External paired predictions missing columns: {missing}")
    if data["match_id"].duplicated().any():
        raise ValueError("External paired match_id must be unique")
    if not data["actual_home_score"].isin((0.0, 0.5, 1.0)).all():
        raise ValueError("actual_home_score must contain 0, 0.5 or 1")
    data["phase"] = data["round"].map(competition_phase)
    return data


def evaluate_variants(
    datasets: tuple,
    external: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_columns = [
        "match_id",
        "fold",
        "test_season",
        "season",
        "competition",
        "round",
        "phase",
        "actual_home_score",
        "home_team_name",
        "away_team_name",
        "clubelo_home_elo",
        "clubelo_away_elo",
        "clubelo_expected_home_score",
        "clubelo_brier_loss",
        "clubelo_log_loss",
        "ao_expected_home_score",
    ]
    result = external[base_columns].copy()
    guardrail_rows: list[dict[str, object]] = []
    for variant in VARIANTS:
        metrics, variant_predictions, _ = evaluate_sequence(
            datasets,
            variant.core,
            variant.carry,
            return_predictions=True,
        )
        assert variant_predictions is not None
        selected = variant_predictions[
            [
                "match_id",
                "expected_home_score",
                "brier_loss",
                "log_loss",
                "home_power",
                "away_power",
            ]
        ].rename(
            columns={
                column: f"{variant.name}_{column}"
                for column in (
                    "expected_home_score",
                    "brier_loss",
                    "log_loss",
                    "home_power",
                    "away_power",
                )
            }
        )
        result = result.merge(selected, on="match_id", how="left", validate="one_to_one")
        guardrail_rows.append(
            {
                "model": variant.name,
                "power_carry": variant.carry.power_carry,
                "k_factor": variant.core.k_factor,
                "max_abs_rating_change": metrics["max_abs_rating_change"],
                "minimum_start_end_rank_correlation": metrics["start_end_rank_correlation"],
                "guardrail_pass": bool(
                    metrics["max_abs_rating_change"] <= 200
                    and metrics["start_end_rank_correlation"] >= 0.85
                ),
            }
        )
    if result.filter(like="_expected_home_score").isna().any().any():
        raise ValueError("At least one diagnostic model lacks a paired prediction")
    current_column = "dynamic_carry_085_current_expected_home_score"
    if not np.allclose(result[current_column], result["ao_expected_home_score"], atol=1e-12):
        raise ValueError("Current AO diagnostic variant does not reproduce external benchmark")
    return result, pd.DataFrame(guardrail_rows)


def model_names() -> tuple[str, ...]:
    return tuple(variant.name for variant in VARIANTS) + ("clubelo_external",)


def loss_columns(model: str) -> tuple[str, str]:
    if model == "clubelo_external":
        return "clubelo_brier_loss", "clubelo_log_loss"
    return f"{model}_brier_loss", f"{model}_log_loss"


def summarize_models(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in model_names():
        brier_column, log_column = loss_columns(model)
        rows.append(
            {
                "model": model,
                "matches": len(data),
                "brier": data[brier_column].mean(),
                "log_loss": data[log_column].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["brier", "log_loss", "model"]).reset_index(drop=True)


def summarize_by_group(data: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows = []
    for group, group_data in data.groupby(group_column, sort=True):
        for model in model_names():
            brier_column, log_column = loss_columns(model)
            rows.append(
                {
                    group_column: group,
                    "model": model,
                    "matches": len(group_data),
                    "brier": group_data[brier_column].mean(),
                    "log_loss": group_data[log_column].mean(),
                }
            )
    return pd.DataFrame(rows)


def incremental_uncertainty(
    ucl: pd.DataFrame,
    *,
    bootstrap_samples: int = 5000,
    seed: int = 20260715,
) -> pd.DataFrame:
    comparisons = (
        ("dynamic_update", "dynamic_season_reset", "static_start_only"),
        ("carry_050_vs_reset", "dynamic_carry_050", "dynamic_season_reset"),
        ("carry_085_vs_reset", "dynamic_carry_085_current", "dynamic_season_reset"),
        ("carry_100_vs_reset", "dynamic_carry_100", "dynamic_season_reset"),
        ("current_vs_clubelo", "dynamic_carry_085_current", "clubelo_external"),
    )
    rng = np.random.default_rng(seed)
    rows = []
    for comparison, challenger, baseline in comparisons:
        challenger_brier, _ = loss_columns(challenger)
        baseline_brier, _ = loss_columns(baseline)
        differences = (ucl[challenger_brier] - ucl[baseline_brier]).to_numpy(float)
        sampled = rng.choice(differences, size=(bootstrap_samples, len(differences)), replace=True)
        means = sampled.mean(axis=1)
        lower, upper = np.quantile(means, (0.025, 0.975))
        fold_differences = (
            ucl.assign(difference=differences)
            .groupby("fold")["difference"]
            .mean()
        )
        rows.append(
            {
                "comparison": comparison,
                "challenger": challenger,
                "baseline": baseline,
                "matches": len(differences),
                "mean_brier_difference": differences.mean(),
                "ci_95_lower": lower,
                "ci_95_upper": upper,
                "folds_won": int(fold_differences.lt(0).sum()),
                "folds": len(fold_differences),
                "reliable_improvement": bool(upper < 0),
                "reliable_harm": bool(lower > 0),
            }
        )
    return pd.DataFrame(rows)


def rating_alignment(ucl: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        home = pd.DataFrame(
            {
                "ao_rating": ucl[f"{variant.name}_home_power"],
                "external_rating": ucl["clubelo_home_elo"],
            }
        )
        away = pd.DataFrame(
            {
                "ao_rating": ucl[f"{variant.name}_away_power"],
                "external_rating": ucl["clubelo_away_elo"],
            }
        )
        observations = pd.concat([home, away], ignore_index=True)
        ao_differences = (
            ucl[f"{variant.name}_home_power"] - ucl[f"{variant.name}_away_power"]
        )
        external_differences = ucl["clubelo_home_elo"] - ucl["clubelo_away_elo"]
        rows.append(
            {
                "model": variant.name,
                "team_match_observations": len(observations),
                "rating_spearman_vs_clubelo": observations["ao_rating"].corr(
                    observations["external_rating"], method="spearman"
                ),
                "match_difference_spearman_vs_clubelo": ao_differences.corr(
                    external_differences, method="spearman"
                ),
            }
        )
    return pd.DataFrame(rows)


def favorite_disagreement(ucl: pd.DataFrame) -> pd.DataFrame:
    current_probability = ucl["dynamic_carry_085_current_expected_home_score"]
    external_probability = ucl["clubelo_expected_home_score"]
    current_side = np.sign(current_probability - 0.5)
    external_side = np.sign(external_probability - 0.5)
    labels = np.where(
        current_side == external_side,
        "same_favorite",
        "different_favorite",
    )
    rows = []
    for label in ("same_favorite", "different_favorite"):
        data = ucl.loc[labels == label]
        rows.append(
            {
                "favorite_relation": label,
                "matches": len(data),
                "ao_brier": data["dynamic_carry_085_current_brier_loss"].mean(),
                "clubelo_brier": data["clubelo_brier_loss"].mean(),
                "ao_minus_clubelo_brier": (
                    data["dynamic_carry_085_current_brier_loss"]
                    - data["clubelo_brier_loss"]
                ).mean(),
            }
        )
    return pd.DataFrame(rows)


def rank_discrepancies(ucl: pd.DataFrame) -> pd.DataFrame:
    home = pd.DataFrame(
        {
            "season": ucl["test_season"],
            "team_name": ucl["home_team_name"],
            "ao_live_rating": ucl["dynamic_carry_085_current_home_power"],
            "clubelo_rating": ucl["clubelo_home_elo"],
        }
    )
    away = pd.DataFrame(
        {
            "season": ucl["test_season"],
            "team_name": ucl["away_team_name"],
            "ao_live_rating": ucl["dynamic_carry_085_current_away_power"],
            "clubelo_rating": ucl["clubelo_away_elo"],
        }
    )
    observations = pd.concat([home, away], ignore_index=True)
    clubs = (
        observations.groupby(["season", "team_name"], sort=True)
        .agg(
            observations=("team_name", "size"),
            mean_ao_live_rating=("ao_live_rating", "mean"),
            mean_clubelo_rating=("clubelo_rating", "mean"),
        )
        .reset_index()
    )
    clubs["ao_rank_percentile"] = clubs.groupby("season")["mean_ao_live_rating"].rank(
        method="average", pct=True
    )
    clubs["clubelo_rank_percentile"] = clubs.groupby("season")["mean_clubelo_rating"].rank(
        method="average", pct=True
    )
    clubs["ao_minus_clubelo_rank_percentile"] = (
        clubs["ao_rank_percentile"] - clubs["clubelo_rank_percentile"]
    )
    clubs["absolute_rank_percentile_gap"] = clubs[
        "ao_minus_clubelo_rank_percentile"
    ].abs()
    return clubs.sort_values(
        ["absolute_rank_percentile_gap", "observations"],
        ascending=[False, False],
    ).reset_index(drop=True)


def model_row(summary: pd.DataFrame, name: str) -> pd.Series:
    rows = summary.loc[summary["model"].eq(name)]
    if len(rows) != 1:
        raise ValueError(f"Expected one summary row for {name}")
    return rows.iloc[0]


def diagnostic_decision(
    ucl_summary: pd.DataFrame,
    uncertainty: pd.DataFrame,
    guardrails: pd.DataFrame,
) -> str:
    if not guardrails["guardrail_pass"].all():
        return "BLOCK_MODEL_FREEZE_GUARDRAIL_FAILURE"
    dynamic = uncertainty.loc[uncertainty["comparison"].eq("dynamic_update")].iloc[0]
    carry = uncertainty.loc[uncertainty["comparison"].eq("carry_085_vs_reset")].iloc[0]
    current = model_row(ucl_summary, "dynamic_carry_085_current")
    external = model_row(ucl_summary, "clubelo_external")
    if bool(carry["reliable_harm"]):
        return "REVISE_OR_REMOVE_POWER_CARRY_FOR_UCL"
    if bool(dynamic["reliable_harm"]):
        return "REVISE_DYNAMIC_UPDATE_FOR_UCL"
    if current["brier"] > external["brier"]:
        return "KEEP_CORE_DIAGNOSE_INITIAL_RATING_AND_UCL_COVERAGE"
    return "KEEP_CURRENT_DYNAMIC_MODEL_NO_UCL_LAYER_HARM_FOUND"


def write_report(
    path: Path,
    ucl: pd.DataFrame,
    summary: pd.DataFrame,
    season_summary: pd.DataFrame,
    phase_summary: pd.DataFrame,
    uncertainty: pd.DataFrame,
    alignment: pd.DataFrame,
    club_rank_discrepancies: pd.DataFrame,
    disagreement: pd.DataFrame,
    guardrails: pd.DataFrame,
    decision: str,
) -> None:
    lines = [
        "# UCL External Elo Gap Diagnostics",
        "",
        "## Question",
        "",
        "Does the UCL gap against ClubElo come from AO First Elo, normal match updates,",
        "or cross-season power carry? All variants use identical exact-date test matches.",
        "",
        f"UCL unseen paired matches: {len(ucl)}",
        f"Decision: `{decision}`",
        "",
        "## Model summary",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "## Isolated layer differences",
        "",
        "Negative Brier difference favors the challenger.",
        "",
        "```text",
        uncertainty.to_string(index=False),
        "```",
        "",
        "## Season detail",
        "",
        "```text",
        season_summary.to_string(index=False),
        "```",
        "",
        "## Phase detail",
        "",
        "```text",
        phase_summary.to_string(index=False),
        "```",
        "",
        "## Rating alignment",
        "",
        "```text",
        alignment.to_string(index=False),
        "```",
        "",
        "## Largest club rank gaps",
        "",
        "Positive rank gap means AO ranks the club higher than ClubElo within the",
        "covered UCL season sample.",
        "",
        "```text",
        club_rank_discrepancies.head(20).to_string(index=False),
        "```",
        "",
        "## Favorite disagreement",
        "",
        "```text",
        disagreement.to_string(index=False),
        "```",
        "",
        "## Guardrails",
        "",
        "```text",
        guardrails.to_string(index=False),
        "```",
        "",
        "## Interpretation contract",
        "",
        "- ClubElo coverage is selective and mostly contains established clubs.",
        "- This run diagnoses AO layers; it does not tune them on ClubElo ratings.",
        "- A layer is revised only when paired match outcomes show reliable harm.",
        "- Final production proof still requires a future untouched season.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
