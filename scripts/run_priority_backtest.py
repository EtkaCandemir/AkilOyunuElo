from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ao_elo.backtest import evaluate_match_predictions
from ao_elo.config import AOEuropeanEloConfig
from ao_elo.pipeline import compute_ao_first_elo_from_csv


DATA_ROOT = ROOT / "data" / "backtest_2021_2026"
OUTPUT_ROOT = ROOT / "output" / "backtest_2021_2026"
DEV_SEASONS = {"2021/22", "2022/23", "2023/24", "2024/25"}
HOLDOUT_SEASON = "2025/26"


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    baseline = AOEuropeanEloConfig.v1_1()
    selected = baseline
    all_results: list[pd.DataFrame] = []

    home_advantage_results = pd.concat(
        [
            evaluate_config(f"home_advantage_{value}", baseline, value).assign(
                stage="home_advantage"
            )
            for value in range(0, 101, 10)
        ],
        ignore_index=True,
    )
    home_summary = summarize(home_advantage_results)
    home_summary.to_csv(OUTPUT_ROOT / "home_advantage_summary.csv", index=False)
    home_advantage = float(str(home_summary.iloc[0]["candidate"]).rsplit("_", 1)[1])

    official_results = pd.concat(
        [
            evaluate_official_coefficient(f"official_multiplier_{value:.2f}", value, home_advantage).assign(
                stage="official_baseline"
            )
            for value in [step / 4 for step in range(2, 17)]
        ],
        ignore_index=True,
    )
    official_summary = summarize(official_results)
    official_summary.to_csv(OUTPUT_ROOT / "official_baseline_summary.csv", index=False)
    official_winner = str(official_summary.iloc[0]["candidate"])
    official_winner_metrics = official_results.loc[official_results["candidate"] == official_winner]
    all_results.extend([home_advantage_results, official_results])

    stages = [
        ("season_weights", season_weight_candidates),
        ("european_prior", european_prior_candidates),
        ("exposure_cap", exposure_cap_candidates),
        ("exposure_blend", exposure_blend_candidates),
        ("country_strength", country_strength_candidates),
    ]
    baseline_metrics = evaluate_config("baseline_v1_1", baseline, home_advantage)
    all_results.append(baseline_metrics.assign(stage="baseline"))

    selections: list[dict[str, object]] = []
    for stage, candidate_builder in stages:
        candidates = candidate_builder(selected)
        stage_results = pd.concat(
            [evaluate_config(name, config, home_advantage).assign(stage=stage) for name, config in candidates],
            ignore_index=True,
        )
        all_results.append(stage_results)
        summary = summarize(stage_results)
        summary.to_csv(OUTPUT_ROOT / f"{stage}_summary.csv", index=False)
        winner_name = str(summary.iloc[0]["candidate"])
        selected = dict(candidates)[winner_name]
        selections.append(
            {
                "stage": stage,
                "selected_candidate": winner_name,
                "dev_log_loss": summary.iloc[0]["dev_log_loss"],
                "validation_log_loss": summary.iloc[0]["validation_log_loss"],
            }
        )

    final_metrics = evaluate_config("selected_model", selected, home_advantage).assign(stage="selected")
    all_results.append(final_metrics)

    independent_candidates = [
        ("season_weights_decay_0.85", dict(season_weights=dict(zip(
            baseline.season_weights,
            [0.14075442253796464, 0.1655934382799584, 0.19481580974112753, 0.22919507028367947, 0.26964125915726994],
        ))), "KEEP_CURRENT", "Holdout log loss worsened."),
        ("european_history_benchmark_28", dict(european_history_benchmark=28.0), "STAGE_A_CANDIDATE", "Improved log loss in all five seasons."),
        ("max_european_exposure_0.80", dict(max_european_exposure=0.80), "KEEP_CURRENT", "Improvement is too small for promotion."),
        ("exposure_blend_0.70_0.30", dict(exposure_season_weight=0.70, exposure_match_weight=0.30), "KEEP_CURRENT", "Difference is practically neutral."),
        ("country_b15_g1.2_c180", dict(country_strength_benchmark=15.0, gamma=1.2, domestic_league_component=180.0), "BLOCKED", "Domestic positions and league sizes are incomplete."),
    ]
    independent_results = pd.concat(
        [
            evaluate_config(name, replace(baseline, **changes), home_advantage).assign(
                stage="independent_check"
            )
            for name, changes, _, _ in independent_candidates
        ],
        ignore_index=True,
    )
    all_results.append(independent_results)
    independent_summary = summarize_against_baseline(
        baseline_metrics,
        independent_results,
        independent_candidates,
    )
    independent_summary.to_csv(OUTPUT_ROOT / "independent_parameter_checks.csv", index=False)
    results = pd.concat(all_results, ignore_index=True)
    results.to_csv(OUTPUT_ROOT / "all_candidate_season_metrics.csv", index=False)
    pd.DataFrame(selections).to_csv(OUTPUT_ROOT / "stage_selections.csv", index=False)
    write_report(
        baseline,
        selected,
        results,
        selections,
        home_advantage,
        official_winner,
        official_winner_metrics,
        independent_summary,
    )

    baseline_holdout = metric_for(baseline_metrics, HOLDOUT_SEASON)
    selected_holdout = metric_for(final_metrics, HOLDOUT_SEASON)
    print("AO European Elo priority backtest")
    print(f"Development seasons: {', '.join(sorted(DEV_SEASONS))}")
    print(f"Holdout season: {HOLDOUT_SEASON}")
    print(f"Development-selected home advantage: {home_advantage:.0f}")
    print(f"Baseline holdout log loss: {baseline_holdout['log_loss']:.6f}")
    print(f"Selected holdout log loss: {selected_holdout['log_loss']:.6f}")
    print(f"Report: {OUTPUT_ROOT / 'backtest_report.md'}")


def evaluate_config(
    candidate: str,
    config: AOEuropeanEloConfig,
    home_advantage: float,
) -> pd.DataFrame:
    rows = []
    for folder in sorted(DATA_ROOT.glob("20??-??")):
        ratings = compute_ao_first_elo_from_csv(
            folder / "teams.csv", folder / "country_coefficients.csv",
            folder / "domestic_context.csv", folder / "club_european_points.csv", config,
        )
        matches = pd.read_csv(folder / "matches.csv")
        metrics = evaluate_match_predictions(ratings, matches, home_advantage=home_advantage)
        rows.append({"candidate": candidate, "season": ratings["season"].iloc[0], **metrics})
    return pd.DataFrame(rows)


def evaluate_official_coefficient(
    candidate: str,
    multiplier: float,
    home_advantage: float,
) -> pd.DataFrame:
    rows = []
    for folder in sorted(DATA_ROOT.glob("20??-??")):
        clubs = pd.read_csv(folder / "club_european_points.csv")
        ratings = clubs[["season", "team_id"]].copy()
        ratings["ao_first_elo"] = 500.0 + multiplier * clubs["official_club_coefficient"]
        matches = pd.read_csv(folder / "matches.csv")
        metrics = evaluate_match_predictions(ratings, matches, home_advantage=home_advantage)
        rows.append({"candidate": candidate, "season": ratings["season"].iloc[0], **metrics})
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    dev = results.loc[results["season"].isin(DEV_SEASONS)]
    validation = results.loc[results["season"] == "2024/25"]
    summary = dev.groupby("candidate", as_index=False).agg(
        dev_log_loss=("log_loss", "mean"),
        dev_brier=("brier", "mean"),
        dev_accuracy=("decisive_accuracy", "mean"),
        dev_spearman=("mean_season_spearman", "mean"),
        season_log_loss_std=("log_loss", "std"),
    )
    validation_values = validation[["candidate", "log_loss"]].rename(
        columns={"log_loss": "validation_log_loss"}
    )
    return summary.merge(validation_values, on="candidate").sort_values(
        ["dev_log_loss", "season_log_loss_std", "dev_brier"]
    ).reset_index(drop=True)


def season_weight_candidates(base: AOEuropeanEloConfig) -> list[tuple[str, AOEuropeanEloConfig]]:
    ratios = [0.45, 0.55, 0.65, 0.75, 0.85, 1.0]
    candidates = [("weights_current", base)]
    for ratio in ratios:
        raw = [ratio**power for power in (4, 3, 2, 1, 0)]
        total = sum(raw)
        weights = dict(zip(base.season_weights, [value / total for value in raw]))
        candidates.append((f"weights_decay_{ratio:.2f}", replace(base, season_weights=weights)))
    return candidates


def european_prior_candidates(base: AOEuropeanEloConfig) -> list[tuple[str, AOEuropeanEloConfig]]:
    return [
        (
            f"europe_b{benchmark}_boost{boost}",
            replace(base, european_history_benchmark=float(benchmark), european_prior_max_boost=float(boost)),
        )
        for benchmark in (12, 16, 20, 24, 28, 32)
        for boost in (300, 360, 420, 480, 540)
    ]


def exposure_cap_candidates(base: AOEuropeanEloConfig) -> list[tuple[str, AOEuropeanEloConfig]]:
    return [(f"exposure_cap_{cap:.2f}", replace(base, max_european_exposure=cap)) for cap in (.75, .80, .85, .90, .95, 1.0)]


def exposure_blend_candidates(base: AOEuropeanEloConfig) -> list[tuple[str, AOEuropeanEloConfig]]:
    return [
        (f"exposure_season_{season_weight:.2f}", replace(base, exposure_season_weight=season_weight, exposure_match_weight=1-season_weight))
        for season_weight in (.30, .40, .50, .60, .70)
    ]


def country_strength_candidates(base: AOEuropeanEloConfig) -> list[tuple[str, AOEuropeanEloConfig]]:
    return [
        (
            f"country_b{benchmark}_g{gamma:.1f}_c{component}",
            replace(base, country_strength_benchmark=float(benchmark), gamma=gamma, domestic_league_component=float(component)),
        )
        for benchmark in (15, 20, 25, 30, 35)
        for gamma in (.6, .8, 1.0, 1.2)
        for component in (100, 120, 140, 160, 180)
    ]


def metric_for(results: pd.DataFrame, season: str) -> pd.Series:
    return results.loc[results["season"] == season].iloc[0]


def summarize_against_baseline(
    baseline: pd.DataFrame,
    candidates: pd.DataFrame,
    definitions: list[tuple[str, dict[str, object], str, str]],
) -> pd.DataFrame:
    baseline_dev = baseline.loc[baseline["season"].isin(DEV_SEASONS), "log_loss"].mean()
    baseline_holdout = metric_for(baseline, HOLDOUT_SEASON)["log_loss"]
    statuses = {name: (status, reason) for name, _, status, reason in definitions}
    rows = []
    for candidate, frame in candidates.groupby("candidate", sort=False):
        dev = frame.loc[frame["season"].isin(DEV_SEASONS), "log_loss"].mean()
        holdout = metric_for(frame, HOLDOUT_SEASON)["log_loss"]
        status, reason = statuses[candidate]
        rows.append(
            {
                "candidate": candidate,
                "dev_log_loss_delta": dev - baseline_dev,
                "holdout_log_loss_delta": holdout - baseline_holdout,
                "seasons_improved": int(
                    sum(
                        metric_for(frame, season)["log_loss"] < metric_for(baseline, season)["log_loss"]
                        for season in sorted(DEV_SEASONS | {HOLDOUT_SEASON})
                    )
                ),
                "decision": status,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def write_report(
    baseline: AOEuropeanEloConfig,
    selected: AOEuropeanEloConfig,
    results: pd.DataFrame,
    selections: list[dict[str, object]],
    home_advantage: float,
    official_winner: str,
    official_winner_metrics: pd.DataFrame,
    independent_summary: pd.DataFrame,
) -> None:
    baseline_holdout = metric_for(results.loc[results["candidate"] == "baseline_v1_1"], HOLDOUT_SEASON)
    selected_holdout = metric_for(results.loc[results["candidate"] == "selected_model"], HOLDOUT_SEASON)
    official_holdout = metric_for(official_winner_metrics, HOLDOUT_SEASON)
    delta = selected_holdout["log_loss"] - baseline_holdout["log_loss"]
    lines = [
        "# AO European Elo Priority Backtest",
        "",
        "The 2025/26 season is a locked holdout and is not used for candidate selection.",
        f"A fixed `{home_advantage:.0f}` Elo home advantage was selected on development seasons using the v1.1 baseline and then held constant.",
        "Lower log loss and Brier are better; higher accuracy and Spearman are better.",
        "",
        "## Stage selections",
        "",
        "| Stage | Selected candidate | Development log loss | 2024/25 log loss |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in selections:
        lines.append(
            f"| {row['stage']} | {row['selected_candidate']} | "
            f"{row['dev_log_loss']:.6f} | {row['validation_log_loss']:.6f} |"
        )
    lines.extend(
        [
            "", "## Locked holdout", "",
            "| Model | Log loss | Brier | Decisive accuracy | Season Spearman |",
            "| --- | ---: | ---: | ---: | ---: |",
            f"| v1.1 baseline | {baseline_holdout['log_loss']:.6f} | {baseline_holdout['brier']:.6f} | {baseline_holdout['decisive_accuracy']:.3f} | {baseline_holdout['mean_season_spearman']:.3f} |",
            f"| UEFA official coefficient ({official_winner}) | {official_holdout['log_loss']:.6f} | {official_holdout['brier']:.6f} | {official_holdout['decisive_accuracy']:.3f} | {official_holdout['mean_season_spearman']:.3f} |",
            f"| Selected candidate | {selected_holdout['log_loss']:.6f} | {selected_holdout['brier']:.6f} | {selected_holdout['decisive_accuracy']:.3f} | {selected_holdout['mean_season_spearman']:.3f} |",
            "",
            f"Holdout log-loss delta (selected - baseline): {delta:+.6f}.",
            "",
            "## Selected config fields",
            "",
            f"- season_weights: `{selected.season_weights}`",
            f"- european_history_benchmark: `{selected.european_history_benchmark}`",
            f"- european_prior_max_boost: `{selected.european_prior_max_boost}`",
            f"- max_european_exposure: `{selected.max_european_exposure}`",
            f"- exposure season/match: `{selected.exposure_season_weight}/{selected.exposure_match_weight}`",
            f"- country_strength_benchmark: `{selected.country_strength_benchmark}`",
            f"- gamma: `{selected.gamma}`",
            f"- domestic_league_component: `{selected.domestic_league_component}`",
            "",
            "## Interpretation boundary",
            "",
            "Stage A lacks complete domestic position and league-size data for non-champions. "
            "Do not change domestic achievement parameters from this result. A parameter should "
            "be promoted into the production config only when development, validation, and locked "
            "holdout metrics improve consistently rather than from the development winner alone.",
        ]
    )
    lines.extend(
        [
            "", "## Independent parameter decisions", "",
            "These checks change one parameter group at a time from v1.1, avoiding compensation between unrelated parameters.",
            "",
            "| Candidate | Dev log-loss delta | Holdout delta | Seasons improved | Decision |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in independent_summary.itertuples(index=False):
        lines.append(
            f"| {row.candidate} | {row.dev_log_loss_delta:+.6f} | "
            f"{row.holdout_log_loss_delta:+.6f} | {row.seasons_improved}/5 | {row.decision} |"
        )
    lines.extend(
        [
            "",
            "Only `european_history_benchmark_28` is retained as a Stage A candidate. "
            "It is not promoted to the main config until the domestic-complete Stage B rerun.",
        ]
    )
    (OUTPUT_ROOT / "backtest_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
