from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_competition_multiplier_calibration import (  # noqa: E402
    CompetitionMultiplierConfig,
    evaluate_competition_seasons,
)
from scripts.run_dynamic_core_calibration import (  # noqa: E402
    MAX_RATING_MOVE_GUARDRAIL,
    RANK_CORRELATION_FLOOR,
    DynamicCoreConfig,
    SeasonData,
    expanding_folds,
    load_calibration_data,
)
from scripts.run_goal_margin_calibration import (  # noqa: E402
    read_full_core_config,
    validate_core_fold_contract,
)


STATIC_DATA_ROOT = ROOT / "data" / "backtest_stage_b_2018_2026"
EVENTS_PATH = ROOT / "data" / "dynamic_backtest_2018_2026" / "matches.csv"
CORE_OUTPUT_ROOT = ROOT / "output" / "dynamic_core_calibration_2018_2026"
OUTPUT_ROOT = ROOT / "output" / "strict_competition_match_calibration_2018_2026"
UEL_CANDIDATES = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
UECL_CANDIDATES = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85)
MIN_GAP = 0.10
REFERENCE = CompetitionMultiplierConfig(0.65, 0.45)
BASELINE = CompetitionMultiplierConfig(1.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate strict UCL > UEL > UECL match Elo multipliers"
    )
    parser.add_argument("--static-data-root", type=Path, default=STATIC_DATA_ROOT)
    parser.add_argument("--events-path", type=Path, default=EVENTS_PATH)
    parser.add_argument("--core-output-root", type=Path, default=CORE_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    datasets = load_calibration_data(
        args.static_data_root.resolve(),
        args.events_path.resolve(),
    )
    seasons = tuple(data.season for data in datasets)
    folds = expanding_folds(seasons)
    core_output_root = args.core_output_root.resolve()
    core_selections = pd.read_csv(core_output_root / "fold_selections.csv")
    full_core = read_full_core_config(core_output_root / "full_candidate_metrics.csv")
    validate_core_fold_contract(core_selections, folds)
    strict_candidates = strict_candidate_grid()
    all_candidates = (BASELINE, *strict_candidates)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    selection_rows: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for fold_number, (train_seasons, test_season) in enumerate(folds, start=1):
        core_row = core_selections.loc[core_selections["fold"].eq(fold_number)].iloc[0]
        core_config = DynamicCoreConfig(
            float(core_row["selected_scale"]),
            float(core_row["selected_home_advantage"]),
            float(core_row["selected_k"]),
        )
        train_data = tuple(data for data in datasets if data.season in train_seasons)
        test_data = next(data for data in datasets if data.season == test_season)
        selected, selected_train = select_candidate(train_data, core_config, all_candidates)
        strict_challenger, strict_train = select_candidate(
            train_data,
            core_config,
            strict_candidates,
        )
        baseline_train, _ = evaluate_competition_seasons(
            train_data,
            core_config,
            BASELINE,
        )
        reference_train, _ = evaluate_competition_seasons(
            train_data,
            core_config,
            REFERENCE,
        )
        selection_rows.append(
            {
                "fold": fold_number,
                "train_seasons": ",".join(train_seasons),
                "test_season": test_season,
                "core_scale": core_config.elo_scale,
                "core_home_advantage": core_config.home_advantage,
                "core_k": core_config.k_factor,
                "selected_uel_multiplier": selected.uel_multiplier,
                "selected_uecl_multiplier": selected.uecl_multiplier,
                "selected_train_brier_difference": (
                    selected_train["brier"] - baseline_train["brier"]
                ),
                "strict_uel_multiplier": strict_challenger.uel_multiplier,
                "strict_uecl_multiplier": strict_challenger.uecl_multiplier,
                "strict_train_brier_difference": (
                    strict_train["brier"] - baseline_train["brier"]
                ),
                "reference_train_brier_difference": (
                    reference_train["brier"] - baseline_train["brier"]
                ),
            }
        )

        model_predictions: dict[str, pd.DataFrame] = {}
        for model_name, multiplier_config in (
            ("selected_layer", selected),
            ("strict_challenger", strict_challenger),
            ("reference_065_045", REFERENCE),
            ("core_baseline", BASELINE),
        ):
            metrics, predictions = evaluate_competition_seasons(
                (test_data,),
                core_config,
                multiplier_config,
                return_predictions=True,
            )
            result_rows.append(
                {
                    "fold": fold_number,
                    "test_season": test_season,
                    "model": model_name,
                    "core_scale": core_config.elo_scale,
                    "core_home_advantage": core_config.home_advantage,
                    "core_k": core_config.k_factor,
                    "ucl_multiplier": multiplier_config.ucl_multiplier,
                    "uel_multiplier": multiplier_config.uel_multiplier,
                    "uecl_multiplier": multiplier_config.uecl_multiplier,
                    **metrics,
                }
            )
            assert predictions is not None
            model_predictions[model_name] = predictions.rename(
                columns={
                    "expected_home_score": f"{model_name}_expected_home_score",
                    "brier_loss": f"{model_name}_brier_loss",
                    "log_loss": f"{model_name}_log_loss",
                }
            )
        joined = model_predictions["selected_layer"]
        for model_name in ("strict_challenger", "reference_065_045", "core_baseline"):
            joined = joined.merge(
                model_predictions[model_name]
                [[
                    "match_id",
                    f"{model_name}_expected_home_score",
                    f"{model_name}_brier_loss",
                    f"{model_name}_log_loss",
                ]],
                on="match_id",
                validate="one_to_one",
            )
        joined.insert(0, "fold", fold_number)
        prediction_frames.append(joined)

    selections = pd.DataFrame(selection_rows)
    fold_results = pd.DataFrame(result_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    summaries = {
        model: summarize_competitions(predictions, model)
        for model in ("selected_layer", "strict_challenger", "reference_065_045")
    }
    uncertainties = {
        model: paired_uncertainty(predictions, model)
        for model in ("selected_layer", "strict_challenger", "reference_065_045")
    }
    stability = parameter_stability(selections)
    final_config, final_metrics = select_candidate(datasets, full_core, all_candidates)
    final_strict, final_strict_metrics = select_candidate(
        datasets,
        full_core,
        strict_candidates,
    )
    full_metrics = candidate_metrics(datasets, full_core, all_candidates)
    effective_k = build_effective_k_table(full_core, final_strict, REFERENCE)
    decision = calibration_decision(
        selections,
        fold_results,
        uncertainties["selected_layer"],
        stability,
        final_config,
    )

    selections.to_csv(output_root / "fold_selections.csv", index=False)
    fold_results.to_csv(output_root / "fold_results.csv", index=False)
    predictions.to_csv(output_root / "unseen_match_predictions.csv", index=False)
    stability.to_csv(output_root / "parameter_stability.csv", index=False)
    full_metrics.to_csv(output_root / "full_candidate_metrics.csv", index=False)
    effective_k.to_csv(output_root / "effective_k_table.csv", index=False)
    for model, summary in summaries.items():
        summary.to_csv(output_root / f"{model}_competition_summary.csv", index=False)
    for model, uncertainty in uncertainties.items():
        uncertainty.to_csv(output_root / f"{model}_paired_uncertainty.csv", index=False)
    write_report(
        output_root / "calibration_report.md",
        seasons,
        selections,
        fold_results,
        summaries,
        uncertainties,
        stability,
        full_core,
        final_config,
        final_metrics,
        final_strict,
        final_strict_metrics,
        effective_k,
        decision,
    )

    print("AO strict competition match-multiplier calibration")
    print(f"Seasons: {len(seasons)}")
    print(f"Matches: {sum(len(data.match_ids) for data in datasets)}")
    print(f"Strict candidates: {len(strict_candidates)}")
    print(
        "Full strict candidate: "
        f"UCL=1, UEL={final_strict.uel_multiplier:g}, "
        f"UECL={final_strict.uecl_multiplier:g}"
    )
    print(f"Decision: {decision}")
    print(f"Report: {output_root / 'calibration_report.md'}")


def is_strict(config: CompetitionMultiplierConfig) -> bool:
    config.validate()
    return bool(
        config.ucl_multiplier > config.uel_multiplier > config.uecl_multiplier > 0
        and config.uel_multiplier - config.uecl_multiplier >= MIN_GAP - 1e-12
    )


def strict_candidate_grid() -> tuple[CompetitionMultiplierConfig, ...]:
    candidates = {
        CompetitionMultiplierConfig(uel, uecl)
        for uel in UEL_CANDIDATES
        for uecl in UECL_CANDIDATES
        if uel > uecl and uel - uecl >= MIN_GAP - 1e-12
    }
    if REFERENCE not in candidates:
        raise AssertionError("Reference 1.00/0.65/0.45 must be in strict grid")
    if not all(is_strict(candidate) for candidate in candidates):
        raise AssertionError("Strict grid contains an invalid hierarchy")
    return tuple(sorted(candidates))


def candidate_metrics(
    datasets: tuple[SeasonData, ...],
    core_config: DynamicCoreConfig,
    candidates: tuple[CompetitionMultiplierConfig, ...],
) -> pd.DataFrame:
    rows = []
    for config in candidates:
        metrics, _ = evaluate_competition_seasons(datasets, core_config, config)
        rows.append(
            {
                "ucl_multiplier": config.ucl_multiplier,
                "uel_multiplier": config.uel_multiplier,
                "uecl_multiplier": config.uecl_multiplier,
                "is_strict": is_strict(config) if config != BASELINE else False,
                "distance_from_neutral": (
                    abs(config.uel_multiplier - 1.0)
                    + abs(config.uecl_multiplier - 1.0)
                ),
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["brier", "log_loss", "distance_from_neutral", "uel_multiplier", "uecl_multiplier"]
    ).reset_index(drop=True)


def select_candidate(
    datasets: tuple[SeasonData, ...],
    core_config: DynamicCoreConfig,
    candidates: tuple[CompetitionMultiplierConfig, ...],
) -> tuple[CompetitionMultiplierConfig, dict[str, float | int]]:
    rows = candidate_metrics(datasets, core_config, candidates)
    selected = rows.iloc[0]
    config = CompetitionMultiplierConfig(
        float(selected["uel_multiplier"]),
        float(selected["uecl_multiplier"]),
    )
    excluded = {
        "ucl_multiplier", "uel_multiplier", "uecl_multiplier",
        "is_strict", "distance_from_neutral",
    }
    return config, {column: selected[column] for column in rows.columns if column not in excluded}


def summarize_competitions(predictions: pd.DataFrame, model: str) -> pd.DataFrame:
    rows = []
    for competition, data in predictions.groupby("competition"):
        model_brier = data[f"{model}_brier_loss"].mean()
        core_brier = data["core_baseline_brier_loss"].mean()
        model_log = data[f"{model}_log_loss"].mean()
        core_log = data["core_baseline_log_loss"].mean()
        rows.append(
            {
                "competition": competition,
                "matches": len(data),
                "brier_difference": model_brier - core_brier,
                "log_loss_difference": model_log - core_log,
            }
        )
    return pd.DataFrame(rows)


def paired_uncertainty(
    predictions: pd.DataFrame,
    model: str,
    *,
    bootstrap_samples: int = 4000,
    seed: int = 20260715,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    groups = [("ALL", predictions), *predictions.groupby("competition")]
    for competition, data in groups:
        differences = (
            data[f"{model}_brier_loss"] - data["core_baseline_brier_loss"]
        ).to_numpy(float)
        sampled = rng.choice(differences, size=(bootstrap_samples, len(differences)), replace=True)
        means = sampled.mean(axis=1)
        lower, upper = np.quantile(means, (0.025, 0.975))
        rows.append(
            {
                "competition": competition,
                "matches": len(differences),
                "mean_brier_difference": differences.mean(),
                "ci_95_lower": lower,
                "ci_95_upper": upper,
                "reliable_improvement": bool(upper < 0),
                "reliable_harm": bool(lower > 0),
            }
        )
    return pd.DataFrame(rows)


def parameter_stability(selections: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in ("selected_uel_multiplier", "selected_uecl_multiplier"):
        counts = selections[column].value_counts().sort_values(ascending=False)
        rows.append(
            {
                "parameter": column,
                "mode": float(counts.index[0]),
                "mode_count": int(counts.iloc[0]),
                "folds": len(selections),
                "mode_share": float(counts.iloc[0] / len(selections)),
                "min": float(selections[column].min()),
                "max": float(selections[column].max()),
            }
        )
    return pd.DataFrame(rows)


def build_effective_k_table(
    core_config: DynamicCoreConfig,
    strict_config: CompetitionMultiplierConfig,
    reference: CompetitionMultiplierConfig,
) -> pd.DataFrame:
    rows = []
    for competition in ("UCL", "UEL", "UECL"):
        rows.append(
            {
                "competition": competition,
                "core_k": core_config.k_factor,
                "strict_multiplier": strict_config.for_competition(competition),
                "strict_effective_k": (
                    core_config.k_factor * strict_config.for_competition(competition)
                ),
                "reference_multiplier": reference.for_competition(competition),
                "reference_effective_k": (
                    core_config.k_factor * reference.for_competition(competition)
                ),
            }
        )
    return pd.DataFrame(rows)


def calibration_decision(
    selections: pd.DataFrame,
    fold_results: pd.DataFrame,
    uncertainty: pd.DataFrame,
    stability: pd.DataFrame,
    final_config: CompetitionMultiplierConfig,
) -> str:
    pivot = fold_results.pivot(index="fold", columns="model", values="brier")
    fold_wins = int((pivot["selected_layer"] < pivot["core_baseline"]).sum())
    overall = uncertainty.loc[uncertainty["competition"].eq("ALL")].iloc[0]
    no_reliable_harm = not bool(
        uncertainty.loc[uncertainty["competition"].ne("ALL"), "reliable_harm"].any()
    )
    stable = bool(stability["mode_share"].ge(0.5).all())
    selected_rows = fold_results.loc[fold_results["model"].eq("selected_layer")]
    ranking_safe = bool(
        selected_rows["start_end_rank_correlation"].ge(RANK_CORRELATION_FLOOR).all()
        and selected_rows["max_abs_rating_change"].le(MAX_RATING_MOVE_GUARDRAIL).all()
    )
    if final_config == BASELINE:
        return "REJECT_STRICT_COMPETITION_MATCH_MULTIPLIERS_KEEP_CORE"
    if (
        fold_wins >= 5
        and bool(overall["reliable_improvement"])
        and no_reliable_harm
        and stable
        and ranking_safe
    ):
        return "PROVISIONAL_ACCEPT_STRICT_COMPETITION_MATCH_MULTIPLIERS"
    return "KEEP_STRICT_COMPETITION_MATCH_MULTIPLIERS_AS_CANDIDATE"


def write_report(
    path: Path,
    seasons: tuple[str, ...],
    selections: pd.DataFrame,
    fold_results: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
    uncertainties: dict[str, pd.DataFrame],
    stability: pd.DataFrame,
    core_config: DynamicCoreConfig,
    final_config: CompetitionMultiplierConfig,
    final_metrics: dict[str, float | int],
    final_strict: CompetitionMultiplierConfig,
    final_strict_metrics: dict[str, float | int],
    effective_k: pd.DataFrame,
    decision: str,
) -> None:
    pivot = fold_results.pivot(index="fold", columns="model", values="brier")
    wins = {
        model: int((pivot[model] < pivot["core_baseline"]).sum())
        for model in ("selected_layer", "strict_challenger", "reference_065_045")
    }
    selection_rows = [
        f"| {row.fold} | {row.test_season} | {row.selected_uel_multiplier:g} | "
        f"{row.selected_uecl_multiplier:g} | {row.strict_uel_multiplier:g} | "
        f"{row.strict_uecl_multiplier:g} | {row.strict_train_brier_difference:.6f} |"
        for row in selections.itertuples(index=False)
    ]
    comparison_rows = []
    for model, label in (
        ("selected_layer", "Nested selected"),
        ("strict_challenger", "Forced strict"),
        ("reference_065_045", "Reference 1/.65/.45"),
    ):
        overall = uncertainties[model].loc[
            uncertainties[model]["competition"].eq("ALL")
        ].iloc[0]
        comparison_rows.append(
            f"| {label} | {wins[model]}/6 | {overall.mean_brier_difference:.6f} | "
            f"{overall.ci_95_lower:.6f} | {overall.ci_95_upper:.6f} |"
        )
    segment_rows = []
    for model, label in (
        ("strict_challenger", "Forced strict"),
        ("reference_065_045", "Reference"),
    ):
        for row in summaries[model].itertuples(index=False):
            segment_rows.append(
                f"| {label} | {row.competition} | {row.matches} | "
                f"{row.brier_difference:.6f} | {row.log_loss_difference:.6f} |"
            )
    k_rows = [
        f"| {row.competition} | {row.strict_multiplier:g} | {row.strict_effective_k:g} | "
        f"{row.reference_multiplier:g} | {row.reference_effective_k:g} |"
        for row in effective_k.itertuples(index=False)
    ]
    stability_rows = [
        f"| {row.parameter} | {row.mode:g} | {row.mode_count}/{row.folds} | "
        f"{row.min:g}-{row.max:g} |"
        for row in stability.itertuples(index=False)
    ]
    text = "\n".join(
        [
            "# AO Strict Competition Match Multiplier Calibration",
            "",
            f"Decision: **{decision}**",
            "",
            "## Scope",
            "",
            f"Seasons: {seasons[0]} through {seasons[-1]}; outer folds: {len(selections)}.",
            "This run directly tests competition multipliers inside the normal zero-sum",
            "match Elo update. It does not add a second win or progression bonus.",
            "",
            "```text",
            "Delta_match = K_core * C_competition * (S - E)",
            "```",
            "",
            "The neutral 1/1/1 core is only the comparator. Every strict candidate satisfies",
            "UCL=1.00 > UEL > UECL with a minimum 0.10 UEL-UECL gap. The requested",
            "1.00/0.65/0.45 reference is reported independently in every unseen fold.",
            "",
            "## Fold Selections",
            "",
            "| Fold | Unseen season | Selected UEL | Selected UECL | Strict UEL | Strict UECL | Strict train diff |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
            *selection_rows,
            "",
            "## Unseen Comparison",
            "",
            "Negative Brier differences favor the multiplier model.",
            "",
            "| Model | Fold wins | Mean difference | CI lower | CI upper |",
            "| --- | ---: | ---: | ---: | ---: |",
            *comparison_rows,
            "",
            "## Competition Segments",
            "",
            "| Model | Competition | Matches | Brier difference | Log-loss difference |",
            "| --- | --- | ---: | ---: | ---: |",
            *segment_rows,
            "",
            "## Parameter Stability",
            "",
            "| Parameter | Mode | Fold frequency | Range |",
            "| --- | ---: | ---: | ---: |",
            *stability_rows,
            "",
            "## Full-Data Results",
            "",
            f"All-candidate selection: `UCL=1`, `UEL={final_config.uel_multiplier:g}`, ",
            f"`UECL={final_config.uecl_multiplier:g}`; Brier={float(final_metrics['brier']):.6f}.",
            f"Best forced strict: `UCL=1`, `UEL={final_strict.uel_multiplier:g}`, ",
            f"`UECL={final_strict.uecl_multiplier:g}`; ",
            f"Brier={float(final_strict_metrics['brier']):.6f}.",
            "",
            "| Competition | Strict multiplier | Strict effective K | Reference multiplier | Reference effective K |",
            "| --- | ---: | ---: | ---: | ---: |",
            *k_rows,
            "",
            "Promotion requires at least 5/6 nested unseen-fold wins, an overall paired",
            "95% interval below zero, no reliably harmed competition, stable parameters and",
            "ranking guardrails. Domain preference alone cannot override these checks.",
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
