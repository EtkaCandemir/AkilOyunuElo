from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_priority_backtest as stage_a  # noqa: E402
from ao_elo.config import AOEuropeanEloConfig  # noqa: E402


DATA_ROOT = ROOT / "data" / "backtest_stage_b_2021_2026"
OUTPUT_ROOT = ROOT / "output" / "backtest_stage_b_2021_2026"
HOLDOUT_SEASON = "2025/26"
DEV_SEASONS = {"2021/22", "2022/23", "2023/24", "2024/25"}


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stage_a.DATA_ROOT = DATA_ROOT
    stage_a.OUTPUT_ROOT = OUTPUT_ROOT

    baseline = AOEuropeanEloConfig.v1_1()
    home_advantage, home_results = select_home_advantage(baseline)
    baseline_metrics = stage_a.evaluate_config("baseline_v1_1", baseline, home_advantage)
    official_name, official_metrics, official_results = select_official_baseline(
        home_advantage
    )
    selected = baseline
    all_results = [
        home_results,
        official_results,
        baseline_metrics.assign(stage="baseline"),
    ]
    selections: list[dict[str, object]] = []

    stages = [
        ("season_weights", stage_a.season_weight_candidates),
        ("european_prior", stage_a.european_prior_candidates),
        ("exposure_cap", stage_a.exposure_cap_candidates),
        ("exposure_blend", stage_a.exposure_blend_candidates),
        ("country_strength", expanded_country_candidates),
        ("domestic_scale", domestic_scale_candidates),
        ("percentile_shape", percentile_candidates),
        ("cup_values", cup_candidates),
    ]

    winners: list[tuple[str, str, AOEuropeanEloConfig]] = []
    for stage, builder in stages:
        candidates = builder(selected)
        results = pd.concat(
            [
                stage_a.evaluate_config(name, config, home_advantage).assign(stage=stage)
                for name, config in candidates
            ],
            ignore_index=True,
        )
        all_results.append(results)
        summary = stage_a.summarize(results)
        summary.to_csv(OUTPUT_ROOT / f"{stage}_summary.csv", index=False)
        winner_name = str(summary.iloc[0]["candidate"])
        selected = dict(candidates)[winner_name]
        winners.append((stage, winner_name, isolate_stage_config(stage, selected, baseline)))
        selections.append(
            {
                "stage": stage,
                "selected_candidate": winner_name,
                "dev_log_loss": summary.iloc[0]["dev_log_loss"],
                "validation_log_loss": summary.iloc[0]["validation_log_loss"],
            }
        )

    selected_metrics = stage_a.evaluate_config(
        "stage_b_grid_winner", selected, home_advantage
    ).assign(stage="selected")
    all_results.append(selected_metrics)

    independent_results = pd.concat(
        [
            stage_a.evaluate_config(
                f"independent_{stage}", config, home_advantage
            ).assign(stage="independent")
            for stage, _, config in winners
        ],
        ignore_index=True,
    )
    all_results.append(independent_results)
    independent_summary = independent_decisions(
        baseline_metrics,
        independent_results,
        winners,
    )
    independent_summary.to_csv(
        OUTPUT_ROOT / "independent_parameter_checks.csv", index=False
    )

    focused_results = evaluate_focused_candidates(baseline, home_advantage)
    all_results.append(focused_results)
    focused_summary = focused_decisions(baseline_metrics, focused_results)
    focused_summary.to_csv(OUTPUT_ROOT / "focused_parameter_checks.csv", index=False)

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(OUTPUT_ROOT / "all_candidate_season_metrics.csv", index=False)
    pd.DataFrame(selections).to_csv(OUTPUT_ROOT / "stage_selections.csv", index=False)
    write_report(
        baseline_metrics,
        selected_metrics,
        selected,
        home_advantage,
        official_name,
        official_metrics,
        selections,
        independent_summary,
        focused_summary,
    )

    baseline_holdout = stage_a.metric_for(baseline_metrics, HOLDOUT_SEASON)
    selected_holdout = stage_a.metric_for(selected_metrics, HOLDOUT_SEASON)
    print("AO European Elo Stage B backtest")
    print(f"Development-selected home advantage: {home_advantage:.0f}")
    print(f"Baseline holdout log loss: {baseline_holdout['log_loss']:.6f}")
    print(f"Grid winner holdout log loss: {selected_holdout['log_loss']:.6f}")
    print(f"Report: {OUTPUT_ROOT / 'backtest_report.md'}")


def select_official_baseline(
    home_advantage: float,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    results = pd.concat(
        [
            stage_a.evaluate_official_coefficient(
                f"official_multiplier_{multiplier:.2f}", multiplier, home_advantage
            ).assign(stage="official_baseline")
            for multiplier in (step / 4 for step in range(2, 25))
        ],
        ignore_index=True,
    )
    summary = stage_a.summarize(results)
    summary.to_csv(OUTPUT_ROOT / "official_baseline_summary.csv", index=False)
    winner = str(summary.iloc[0]["candidate"])
    return winner, results.loc[results["candidate"] == winner], results


def select_home_advantage(
    baseline: AOEuropeanEloConfig,
) -> tuple[float, pd.DataFrame]:
    results = pd.concat(
        [
            stage_a.evaluate_config(f"home_advantage_{value}", baseline, value).assign(
                stage="home_advantage"
            )
            for value in range(0, 101, 10)
        ],
        ignore_index=True,
    )
    summary = stage_a.summarize(results)
    summary.to_csv(OUTPUT_ROOT / "home_advantage_summary.csv", index=False)
    value = float(str(summary.iloc[0]["candidate"]).rsplit("_", 1)[1])
    return value, results


def expanded_country_candidates(
    base: AOEuropeanEloConfig,
) -> list[tuple[str, AOEuropeanEloConfig]]:
    return [
        (
            f"country_b{benchmark}_g{gamma:.1f}_c{component}",
            replace(
                base,
                country_strength_benchmark=float(benchmark),
                gamma=gamma,
                domestic_league_component=float(component),
            ),
        )
        for benchmark in (10, 12.5, 15, 17.5, 20)
        for gamma in (1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6)
        for component in (220, 240, 260, 280, 300, 320, 340, 360, 400)
    ]


def domestic_scale_candidates(
    base: AOEuropeanEloConfig,
) -> list[tuple[str, AOEuropeanEloConfig]]:
    return [
        (
            f"domestic_component_{component}_alpha_{alpha:.1f}",
            replace(
                base,
                domestic_achievement_component=float(component),
                achievement_alpha=alpha,
            ),
        )
        for component in (80, 120, 160, 200, 240, 280, 320, 360, 400)
        for alpha in (0.0, .1, .2, .4, .6, .8, 1.0)
    ]


def evaluate_focused_candidates(
    baseline: AOEuropeanEloConfig,
    home_advantage: float,
) -> pd.DataFrame:
    candidates = [
        ("eu_b28_only", replace(baseline, european_history_benchmark=28.0)),
        ("eu_boost480_only", replace(baseline, european_prior_max_boost=480.0)),
        (
            "eu_b28_boost480",
            replace(
                baseline,
                european_history_benchmark=28.0,
                european_prior_max_boost=480.0,
            ),
        ),
        (
            "country_b15_only",
            replace(baseline, country_strength_benchmark=15.0),
        ),
        ("country_gamma1_8_only", replace(baseline, gamma=1.8)),
        (
            "league_component260_only",
            replace(baseline, domestic_league_component=260.0),
        ),
        (
            "league_component300_only",
            replace(baseline, domestic_league_component=300.0),
        ),
        (
            "country_b15_g1_8_c260",
            replace(
                baseline,
                country_strength_benchmark=15.0,
                gamma=1.8,
                domestic_league_component=260.0,
            ),
        ),
        ("achievement_alpha0_only", replace(baseline, achievement_alpha=0.0)),
        (
            "achievement_component280_only",
            replace(baseline, domestic_achievement_component=280.0),
        ),
        (
            "achievement_c280_alpha0",
            replace(
                baseline,
                domestic_achievement_component=280.0,
                achievement_alpha=0.0,
            ),
        ),
        (
            "achievement_c360_alpha0",
            replace(
                baseline,
                domestic_achievement_component=360.0,
                achievement_alpha=0.0,
            ),
        ),
    ]
    return pd.concat(
        [
            stage_a.evaluate_config(name, config, home_advantage).assign(
                stage="focused_check"
            )
            for name, config in candidates
        ],
        ignore_index=True,
    )


def focused_decisions(
    baseline: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    baseline_dev = baseline.loc[
        baseline["season"].isin(DEV_SEASONS), "log_loss"
    ].mean()
    baseline_holdout = stage_a.metric_for(baseline, HOLDOUT_SEASON)["log_loss"]
    rows = []
    for candidate, frame in candidates.groupby("candidate", sort=False):
        dev = frame.loc[frame["season"].isin(DEV_SEASONS), "log_loss"].mean()
        holdout = stage_a.metric_for(frame, HOLDOUT_SEASON)["log_loss"]
        seasons_improved = sum(
            stage_a.metric_for(frame, season)["log_loss"]
            < stage_a.metric_for(baseline, season)["log_loss"]
            for season in sorted(DEV_SEASONS | {HOLDOUT_SEASON})
        )
        rows.append(
            {
                "candidate": candidate,
                "dev_log_loss_delta": dev - baseline_dev,
                "holdout_log_loss_delta": holdout - baseline_holdout,
                "seasons_improved": seasons_improved,
            }
        )
    return pd.DataFrame(rows).sort_values("dev_log_loss_delta").reset_index(drop=True)


def percentile_candidates(
    base: AOEuropeanEloConfig,
) -> list[tuple[str, AOEuropeanEloConfig]]:
    return [
        (
            f"percentile_f{floor:.2f}_s{scale:.2f}_d{delta:.1f}",
            replace(
                base,
                percentile_floor=floor,
                percentile_scale=scale,
                percentile_delta=delta,
            ),
        )
        for floor in (.05, .10, .15, .20, .25, .30)
        for scale in (.50, .60, .70, .80, .90, 1.00)
        for delta in (.7, 1.0, 1.3)
    ]


def cup_candidates(
    base: AOEuropeanEloConfig,
) -> list[tuple[str, AOEuropeanEloConfig]]:
    return [
        (
            f"cup_base_{cup_base:.2f}_double_{double:.2f}",
            replace(
                base,
                cup_base_score=cup_base,
                cup_double_bonus_multiplier=double,
            ),
        )
        for cup_base in (.40, .50, .62, .70, .80)
        for double in (0.0, .04, .08, .12)
    ]


def isolate_stage_config(
    stage: str,
    winner: AOEuropeanEloConfig,
    baseline: AOEuropeanEloConfig,
) -> AOEuropeanEloConfig:
    fields = {
        "season_weights": ("season_weights",),
        "european_prior": ("european_history_benchmark", "european_prior_max_boost"),
        "exposure_cap": ("max_european_exposure",),
        "exposure_blend": ("exposure_season_weight", "exposure_match_weight"),
        "country_strength": ("country_strength_benchmark", "gamma", "domestic_league_component"),
        "domestic_scale": ("domestic_achievement_component", "achievement_alpha"),
        "percentile_shape": ("percentile_floor", "percentile_scale", "percentile_delta"),
        "cup_values": ("cup_base_score", "cup_double_bonus_multiplier"),
    }[stage]
    return replace(baseline, **{field: getattr(winner, field) for field in fields})


def independent_decisions(
    baseline: pd.DataFrame,
    candidates: pd.DataFrame,
    winners: list[tuple[str, str, AOEuropeanEloConfig]],
) -> pd.DataFrame:
    baseline_dev = baseline.loc[baseline["season"].isin(DEV_SEASONS), "log_loss"].mean()
    baseline_holdout = stage_a.metric_for(baseline, HOLDOUT_SEASON)["log_loss"]
    rows = []
    for stage, _, _ in winners:
        name = f"independent_{stage}"
        frame = candidates.loc[candidates["candidate"] == name]
        dev = frame.loc[frame["season"].isin(DEV_SEASONS), "log_loss"].mean()
        holdout = stage_a.metric_for(frame, HOLDOUT_SEASON)["log_loss"]
        seasons_improved = sum(
            stage_a.metric_for(frame, season)["log_loss"]
            < stage_a.metric_for(baseline, season)["log_loss"]
            for season in sorted(DEV_SEASONS | {HOLDOUT_SEASON})
        )
        if seasons_improved == 5 and holdout < baseline_holdout:
            decision = "CANDIDATE"
        elif holdout >= baseline_holdout:
            decision = "KEEP_CURRENT"
        else:
            decision = "WEAK_EVIDENCE"
        rows.append(
            {
                "stage": stage,
                "dev_log_loss_delta": dev - baseline_dev,
                "holdout_log_loss_delta": holdout - baseline_holdout,
                "seasons_improved": seasons_improved,
                "decision": decision,
            }
        )
    return pd.DataFrame(rows)


def write_report(
    baseline: pd.DataFrame,
    selected_metrics: pd.DataFrame,
    selected: AOEuropeanEloConfig,
    home_advantage: float,
    official_name: str,
    official_metrics: pd.DataFrame,
    selections: list[dict[str, object]],
    independent: pd.DataFrame,
    focused: pd.DataFrame,
) -> None:
    baseline_holdout = stage_a.metric_for(baseline, HOLDOUT_SEASON)
    selected_holdout = stage_a.metric_for(selected_metrics, HOLDOUT_SEASON)
    official_holdout = stage_a.metric_for(official_metrics, HOLDOUT_SEASON)
    lines = [
        "# AO European Elo Stage B Backtest",
        "",
        "Stage B uses domestic positions and league sizes for every required CH/N participant.",
        "Cup-only clubs outside the top division retain the model's explicit unknown-position path.",
        f"A fixed {home_advantage:.0f} Elo home advantage was selected on development data.",
        "The 2025/26 season remained locked until all stage selections were complete.",
        "",
        "## Stage selections",
        "",
        "| Stage | Winner | Dev log loss | 2024/25 latest-dev log loss |",
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
            "| Model | Log loss | Brier | Decisive accuracy | Spearman |",
            "| --- | ---: | ---: | ---: | ---: |",
            f"| v1.1 baseline | {baseline_holdout['log_loss']:.6f} | {baseline_holdout['brier']:.6f} | {baseline_holdout['decisive_accuracy']:.3f} | {baseline_holdout['mean_season_spearman']:.3f} |",
            f"| UEFA official coefficient ({official_name}) | {official_holdout['log_loss']:.6f} | {official_holdout['brier']:.6f} | {official_holdout['decisive_accuracy']:.3f} | {official_holdout['mean_season_spearman']:.3f} |",
            f"| Stage B grid winner | {selected_holdout['log_loss']:.6f} | {selected_holdout['brier']:.6f} | {selected_holdout['decisive_accuracy']:.3f} | {selected_holdout['mean_season_spearman']:.3f} |",
            "", "## Independent decisions", "",
            "| Parameter group | Dev delta | Holdout delta | Seasons improved | Decision |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in independent.itertuples(index=False):
        lines.append(
            f"| {row.stage} | {row.dev_log_loss_delta:+.6f} | "
            f"{row.holdout_log_loss_delta:+.6f} | {row.seasons_improved}/5 | {row.decision} |"
        )
    lines.extend(
        [
            "", "## Focused ablations", "",
            "Each row starts from the unchanged v1.1 baseline.",
            "Negative deltas improve log loss.",
            "",
            "| Candidate | Dev delta | Holdout delta | Seasons improved |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in focused.itertuples(index=False):
        lines.append(
            f"| {row.candidate} | {row.dev_log_loss_delta:+.6f} | "
            f"{row.holdout_log_loss_delta:+.6f} | {row.seasons_improved}/5 |"
        )
    lines.extend(
        [
            "", "## Grid winner config", "",
            f"- season_weights: `{selected.season_weights}`",
            f"- country benchmark/gamma/component: `{selected.country_strength_benchmark}/{selected.gamma}/{selected.domestic_league_component}`",
            f"- European benchmark/boost: `{selected.european_history_benchmark}/{selected.european_prior_max_boost}`",
            f"- exposure cap and blend: `{selected.max_european_exposure}` and `{selected.exposure_season_weight}/{selected.exposure_match_weight}`",
            f"- domestic component/alpha: `{selected.domestic_achievement_component}/{selected.achievement_alpha}`",
            f"- percentile floor/scale/delta: `{selected.percentile_floor}/{selected.percentile_scale}/{selected.percentile_delta}`",
            f"- cup base/double: `{selected.cup_base_score}/{selected.cup_double_bonus_multiplier}`",
            "",
            "## Decision",
            "",
            "- Keep the current season weights, exposure cap/blend, percentile shape, and cup values.",
            "- Retain `European benchmark 28 / boost 480` as a small but consistent candidate.",
            "- Retain country-strength and domestic-achievement scaling as strong Stage C candidates.",
            "- Do not promote the full grid winner: several country configurations are nearly tied,",
            "  and `achievement_alpha=0` is a boundary result whose exact value is not yet stable.",
            "- Run nested walk-forward folds, competition splits, and paired uncertainty checks before",
            "  changing production defaults.",
            "",
            "The UEFA row is a scaled official club-coefficient baseline, not UEFA's own match",
            "prediction model. The AO models are more predictive on this sample, but the result does",
            "not establish universal superiority over UEFA methodology.",
            "",
            "A grid winner is not automatically promoted. Production changes require consistent",
            "independent improvement, a non-boundary optimum, and a meaningful effect size.",
        ]
    )
    (OUTPUT_ROOT / "backtest_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
