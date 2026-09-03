from __future__ import annotations

"""Walk-forward ablation of proportional European exposure scaling.

Production uses ``min(raw_exposure, 0.65)``.  This research command compares
that fixed baseline with ``lambda * raw_exposure`` while pinning every other
active AO First and replay parameter.  It never writes contracts or artifacts.
"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ao_elo.european_prior_recalibration import (  # noqa: E402
    apply_european_prior_recalibration,
    ranking_uncertainty_summary,
)
from ao_elo.evaluation import schedule_adjusted_team_performance  # noqa: E402
from scripts.run_ao_first_seed_boost_backtest import aggregate_target  # noqa: E402
from scripts.run_current_model_evaluation import (  # noqa: E402
    EvaluationArm,
    evaluate_arm,
    markdown_table,
)
from scripts.run_dynamic_core_calibration import expanding_folds  # noqa: E402
from scripts.run_european_exposure_cap_backtest import (  # noqa: E402
    dependency_uncertainty,
)
from scripts.run_european_prior_recalibration_backtest import (  # noqa: E402
    BASELINE_CONFIG,
    aggregate_candidate_surface,
    build_unseen,
    competition_summary,
    current_snapshot_impact,
    historical_impact,
    load_seed_evidence,
    model_summary,
    nested_selection,
    season_surface,
    select_candidate,
)
from scripts.run_opponent_quintile_backtest import (  # noqa: E402
    load_production_baseline,
)
from scripts.run_stage_weighted_progression_backtest import (  # noqa: E402
    EVENTS_PATH,
    PRODUCTION_CONTRACT,
    XG_DATA,
    load_xg_map,
)
from scripts.run_v2_evaluation_upgrade import read_events  # noqa: E402


OUTPUT_ROOT = ROOT / "output" / "linear_european_exposure_backtest_2018_2026"
CURRENT_CUP_RATINGS = (
    ROOT
    / "output"
    / "season_2026_27_current_european_cups"
    / "ao_first_elo_current_european_teams_2026_27.csv"
)
LINEAR_SCALES = (
    0.40,
    0.50,
    0.60,
    0.625,
    0.65,
    0.675,
    0.70,
    0.725,
    0.75,
    0.775,
    0.80,
    0.825,
    0.85,
    0.90,
    0.95,
    1.00,
)
SOFT_POWERS = (0.5, 1.0, 1.5, 1.8, 0.65 / 0.35)
PRIMARY_SCALE = 0.65


def candidate_grid():
    linear = tuple(
        replace(
            BASELINE_CONFIG,
            exposure_family="LINEAR_SCALE",
            exposure_scale=scale,
        )
        for scale in LINEAR_SCALES
    )
    soft = tuple(
        replace(
            BASELINE_CONFIG,
            exposure_family="SOFT_POWER",
            exposure_scale=PRIMARY_SCALE,
            exposure_power=power,
        )
        for power in SOFT_POWERS
    )
    return (BASELINE_CONFIG, *linear, *soft)


def candidate_metadata(configs) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_key": config.key,
                "exposure_family": config.exposure_family,
                "exposure_scale": (
                    config.exposure_scale
                    if config.exposure_family == "LINEAR_SCALE"
                    else float("nan")
                ),
                "exposure_power": (
                    config.exposure_power
                    if config.exposure_family == "SOFT_POWER"
                    else float("nan")
                ),
            }
            for config in configs
        ]
    )


def fixed_unseen_summary(
    surface: pd.DataFrame,
    folds: tuple[tuple[tuple[str, ...], str], ...],
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    test_seasons = {test for _, test in folds}
    unseen = surface.loc[surface["season"].isin(test_seasons)].copy()
    summary = aggregate_candidate_surface(unseen)
    baseline = unseen.loc[unseen["candidate_key"].eq(BASELINE_CONFIG.key)][
        ["season", "brier_1x2", "log_loss_1x2"]
    ].rename(
        columns={
            "brier_1x2": "baseline_brier_1x2",
            "log_loss_1x2": "baseline_log_loss_1x2",
        }
    )
    wins = []
    for key, frame in unseen.groupby("candidate_key", sort=True):
        compared = frame.merge(baseline, on="season", validate="one_to_one")
        wins.append(
            {
                "candidate_key": key,
                "brier_fold_wins": int(
                    (compared["brier_1x2"] < compared["baseline_brier_1x2"]).sum()
                ),
                "log_loss_fold_wins": int(
                    (
                        compared["log_loss_1x2"]
                        < compared["baseline_log_loss_1x2"]
                    ).sum()
                ),
            }
        )
    return (
        summary.merge(pd.DataFrame(wins), on="candidate_key", validate="one_to_one")
        .merge(metadata, on="candidate_key", validate="one_to_one")
        .sort_values(
            ["exposure_family", "exposure_scale", "exposure_power"],
            na_position="first",
        )
        .reset_index(drop=True)
    )


def fixed_selection(candidate_key: str, folds) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fold": fold,
                "test_season": test_season,
                "selected_candidate_key": candidate_key,
            }
            for fold, (_, test_season) in enumerate(folds, start=1)
        ]
    )


def current_cup_impact(config) -> pd.DataFrame:
    current = pd.read_csv(CURRENT_CUP_RATINGS).rename(
        columns={
            "current_competition": "competition",
            "ao_first_elo": "adjusted_ao_first_elo",
        }
    )
    result = apply_european_prior_recalibration(current, config)
    result["current_rank"] = result["adjusted_ao_first_elo"].rank(
        ascending=False, method="min"
    ).astype(int)
    result["candidate_rank"] = result["candidate_ao_first_elo"].rank(
        ascending=False, method="min"
    ).astype(int)
    result["rank_change"] = result["current_rank"] - result["candidate_rank"]
    return result[
        [
            "season",
            "team_id",
            "team_name",
            "competition",
            "weighted_season_exposure",
            "european_exposure",
            "effective_european_exposure",
            "candidate_effective_exposure",
            "adjusted_domestic_prior",
            "candidate_european_prior",
            "adjusted_ao_first_elo",
            "candidate_ao_first_elo",
            "candidate_elo_delta",
            "current_rank",
            "candidate_rank",
            "rank_change",
        ]
    ].sort_values("candidate_rank")


def current_cup_candidate_summary(configs) -> pd.DataFrame:
    rows = []
    for config in configs:
        impact = current_cup_impact(config)
        delta = impact["candidate_elo_delta"]
        lyon = impact.loc[
            impact["team_name"].str.contains("Lyon", case=False)
        ].iloc[0]
        rows.append(
            {
                "candidate_key": config.key,
                "exposure_family": config.exposure_family,
                "exposure_scale": (
                    config.exposure_scale
                    if config.exposure_family != "HARD_CAP"
                    else float("nan")
                ),
                "exposure_power": (
                    config.exposure_power
                    if config.exposure_family == "SOFT_POWER"
                    else float("nan")
                ),
                "lyon_effective_exposure": float(
                    lyon["candidate_effective_exposure"]
                ),
                "lyon_elo": float(lyon["candidate_ao_first_elo"]),
                "lyon_rank": int(lyon["candidate_rank"]),
                "teams_down": int((delta < -1e-9).sum()),
                "teams_up": int((delta > 1e-9).sum()),
                "teams_unchanged": int((delta.abs() <= 1e-9).sum()),
                "teams_changed_over_10": int(delta.abs().gt(10.0).sum()),
                "teams_changed_over_50": int(delta.abs().gt(50.0).sum()),
                "mean_absolute_change": float(delta.abs().mean()),
                "maximum_absolute_change": float(delta.abs().max()),
            }
        )
    return pd.DataFrame(rows)


def report_text(
    decision: dict[str, object],
    fixed_curve: pd.DataFrame,
    fixed_model: pd.DataFrame,
    fixed_folds: pd.DataFrame,
    fixed_competition: pd.DataFrame,
    loss_uncertainty: pd.DataFrame,
    rank_uncertainty: pd.DataFrame,
    nested_model: pd.DataFrame,
    nested_selections: pd.DataFrame,
    impact: pd.DataFrame,
    impact_summary: pd.DataFrame,
    soft_models: pd.DataFrame,
    soft_loss_uncertainty: pd.DataFrame,
    soft_rank_uncertainty: pd.DataFrame,
) -> str:
    curve_columns = [
        "exposure_family",
        "exposure_scale",
        "exposure_power",
        "brier_1x2",
        "delta_vs_baseline_brier_1x2",
        "log_loss_1x2",
        "delta_vs_baseline_log_loss_1x2",
        "seed_spearman",
        "delta_vs_baseline_seed_spearman",
        "seed_pairwise_accuracy",
        "delta_vs_baseline_seed_pairwise_accuracy",
        "brier_fold_wins",
        "log_loss_fold_wins",
    ]
    lyon = impact.loc[impact["team_name"].str.contains("Lyon", case=False)].iloc[0]
    linear = fixed_curve.loc[fixed_curve["exposure_family"].eq("LINEAR_SCALE")]
    best_brier = linear.sort_values(["brier_1x2", "log_loss_1x2"]).iloc[0]
    best_log_loss = linear.sort_values(["log_loss_1x2", "brier_1x2"]).iloc[0]
    soft = fixed_curve.loc[fixed_curve["exposure_family"].eq("SOFT_POWER")]
    best_soft_brier = soft.sort_values(["brier_1x2", "log_loss_1x2"]).iloc[0]
    best_soft_log_loss = soft.sort_values(["log_loss_1x2", "brier_1x2"]).iloc[0]
    impact_delta = impact["candidate_elo_delta"]
    return "\n".join(
        [
            "# European Exposure Mapping Backtest",
            "",
            "## Sonuç",
            "",
            f"- Karar: `{decision['decision']}`",
            "- Sabit `lambda=0.65`, Brier ve log-loss'ta altı foldun "
            "hiçbirini kazanmadı.",
            f"- En iyi linear Brier katsayısı: `lambda={best_brier['exposure_scale']:g}`; "
            f"baseline farkı `{best_brier['delta_vs_baseline_brier_1x2']:+.6f}`.",
            f"- En iyi linear log-loss katsayısı: "
            f"`lambda={best_log_loss['exposure_scale']:g}`; baseline farkı "
            f"`{best_log_loss['delta_vs_baseline_log_loss_1x2']:+.6f}`.",
            "- Griddeki hiçbir linear katsayı production hard cap'i kayıp "
            "metriklerinde yenmedi.",
            f"- En iyi soft Brier adayı: `gamma={best_soft_brier['exposure_power']:g}`; "
            f"baseline farkı `{best_soft_brier['delta_vs_baseline_brier_1x2']:+.6f}`.",
            f"- En iyi soft log-loss adayı: "
            f"`gamma={best_soft_log_loss['exposure_power']:g}`; baseline farkı "
            f"`{best_soft_log_loss['delta_vs_baseline_log_loss_1x2']:+.6f}`.",
            "",
            "## Soru",
            "",
            "Production `min(raw_exposure, 0.65)` yerine doğrusal "
            "`lambda * raw_exposure` veya yumuşak "
            "`x * [1 - 0.35 * x^gamma]` kullanmak daha iyi mi?",
            "",
            "## Sabitler ve veri",
            "",
            f"- Pencere: `{decision['evidence_window']}`",
            f"- Expanding test fold'u: `{decision['fold_count']}`",
            f"- Görülmemiş maç: `{decision['unseen_matches']}`",
            "- Production baseline: `min(raw_exposure, 0.65)`",
            f"- Ana challenger: `{PRIMARY_SCALE:.2f} * raw_exposure`",
            f"- Linear grid: `{', '.join(f'{x:g}' for x in LINEAR_SCALES)}`",
            f"- Soft gamma grid: `{', '.join(f'{x:g}' for x in SOFT_POWERS)}`",
            "- Participation k, European tail, AO First ve replay parametreleri: "
            "aktif production değerlerine pinli",
            "",
            "## Sabit görülmemiş-sezon eğrisi",
            "",
            markdown_table(fixed_curve[curve_columns]),
            "",
            "### Soft aday pooled sonuçları",
            "",
            markdown_table(soft_models),
            "",
            "### Soft aday kayıp belirsizliği",
            "",
            markdown_table(soft_loss_uncertainty),
            "",
            "### Soft aday sıralama belirsizliği",
            "",
            markdown_table(soft_rank_uncertainty),
            "",
            "### 2026/27 aday etkileri",
            "",
            markdown_table(impact_summary),
            "",
            "Bu tablo test fold'larını kullanır; en iyi satırı buradan seçmek "
            "prospective seçim değildir. Katsayı seçimi aşağıdaki nested bölümde "
            "yalnız geçmiş train sezonlarından yapılır.",
            "",
            "## Sabit lambda=0.65",
            "",
            markdown_table(fixed_model),
            "",
            "### Fold sonuçları",
            "",
            markdown_table(fixed_folds),
            "",
            "### Kupa segmentleri",
            "",
            markdown_table(fixed_competition),
            "",
            "### Kayıp belirsizliği",
            "",
            markdown_table(loss_uncertainty),
            "",
            "### Sıralama belirsizliği",
            "",
            markdown_table(rank_uncertainty),
            "",
            "## Nested seçim",
            "",
            markdown_table(nested_selections),
            "",
            markdown_table(nested_model),
            "",
            "## 2026/27 snapshot etkisi",
            "",
            f"- Lyon: `{lyon['adjusted_ao_first_elo']:.3f}` -> "
            f"`{lyon['candidate_ao_first_elo']:.3f}`; sıra "
            f"`{int(lyon['current_rank'])}` -> `{int(lyon['candidate_rank'])}`",
            f"- 108 takımda düşen/yükselen/değişmeyen: "
            f"`{int((impact_delta < -1e-9).sum())}/"
            f"{int((impact_delta > 1e-9).sum())}/"
            f"{int((impact_delta.abs() <= 1e-9).sum())}`",
            f"- Mutlak değişim >10/>50/>100 Elo: "
            f"`{int(impact_delta.abs().gt(10).sum())}/"
            f"{int(impact_delta.abs().gt(50).sum())}/"
            f"{int(impact_delta.abs().gt(100).sum())}`",
            f"- Karar: `{decision['decision']}`",
            "- Production değişti: `hayır`",
            "",
            "## Kanıt sınırı",
            "",
            "Bu çalışma AO First seed değişikliğini historical replay üzerinde ölçer. "
            "2026/27 sonucu görülmemiş holdout kanıtı değildir. Linear katsayı, "
            "production contract veya artifact zincirine yazılmamıştır.",
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples must be at least 100")
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract_hash = hashlib.sha256(PRODUCTION_CONTRACT.read_bytes()).hexdigest()

    _, datasets, core, parameters, production_seed_map = load_production_baseline()
    seasons = tuple(data.season for data in datasets)
    folds = expanding_folds(seasons)
    events = read_events(EVENTS_PATH)
    target_by_competition = schedule_adjusted_team_performance(events)
    target = aggregate_target(target_by_competition)
    xg_map = load_xg_map(XG_DATA, datasets)
    seeds = load_seed_evidence(production_seed_map)
    configs = candidate_grid()
    metadata = candidate_metadata(configs)

    predictions: dict[str, pd.DataFrame] = {}
    candidate_seeds: dict[str, pd.DataFrame] = {}
    surface_rows: list[dict[str, object]] = []
    for index, config in enumerate(configs, start=1):
        candidate = apply_european_prior_recalibration(seeds, config)
        candidate_seeds[config.key] = candidate
        rating_map = {
            (str(row.season), int(row.team_id)): float(row.candidate_ao_first_elo)
            for row in candidate.itertuples(index=False)
        }
        evaluation = evaluate_arm(
            datasets,
            EvaluationArm(config.key, True, True, True, True, True),
            core=core,
            parameters=parameters,
            current_domestic=rating_map,
            baseline_domestic=production_seed_map,
            xg_map=xg_map,
            target=target_by_competition,
        )
        predictions[config.key] = evaluation.predictions
        rows = season_surface(config, candidate, evaluation.predictions, target, seasons)
        for row in rows:
            row["exposure_family"] = config.exposure_family
            row["exposure_scale"] = (
                config.exposure_scale
                if config.exposure_family == "LINEAR_SCALE"
                else float("nan")
            )
            row["exposure_power"] = (
                config.exposure_power
                if config.exposure_family == "SOFT_POWER"
                else float("nan")
            )
        surface_rows.extend(rows)
        print(f"Replayed {index}/{len(configs)}: {config.key}", flush=True)

    surface = pd.DataFrame(surface_rows)
    fixed_curve = fixed_unseen_summary(surface, folds, metadata)
    full_summary = aggregate_candidate_surface(surface).merge(
        metadata, on="candidate_key", validate="one_to_one"
    )
    primary = next(
        config
        for config in configs
        if config.exposure_family == "LINEAR_SCALE"
        and config.exposure_scale == PRIMARY_SCALE
    )

    primary_selections = fixed_selection(primary.key, folds)
    primary_unseen, primary_folds = build_unseen(
        primary_selections, predictions, candidate_seeds, target, folds
    )
    primary_model = model_summary(primary_unseen, primary_folds)
    primary_model.loc[
        primary_model["model"].eq("NESTED_RECALIBRATION"), "model"
    ] = "FIXED_LINEAR_0.65"
    primary_competition = competition_summary(primary_unseen)
    primary_loss_uncertainty = dependency_uncertainty(
        primary_unseen, int(args.bootstrap_samples)
    )
    primary_rank_uncertainty = ranking_uncertainty_summary(
        primary_folds, int(args.bootstrap_samples), seed=20260831
    )

    soft_model_rows = []
    soft_fold_rows = []
    soft_competition_rows = []
    soft_loss_rows = []
    soft_rank_rows = []
    for index, config in enumerate(
        (item for item in configs if item.exposure_family == "SOFT_POWER"),
        start=1,
    ):
        selections = fixed_selection(config.key, folds)
        unseen, fold_result = build_unseen(
            selections, predictions, candidate_seeds, target, folds
        )
        candidate_model = model_summary(unseen, fold_result).loc[
            lambda frame: frame["model"].eq("NESTED_RECALIBRATION")
        ].copy()
        candidate_model["exposure_power"] = config.exposure_power
        soft_model_rows.append(candidate_model)

        fold_result.insert(0, "exposure_power", config.exposure_power)
        soft_fold_rows.append(fold_result)
        competition = competition_summary(unseen)
        competition.insert(0, "exposure_power", config.exposure_power)
        soft_competition_rows.append(competition)
        loss = dependency_uncertainty(unseen, int(args.bootstrap_samples))
        loss.insert(0, "exposure_power", config.exposure_power)
        soft_loss_rows.append(loss)
        rank = ranking_uncertainty_summary(
            fold_result,
            int(args.bootstrap_samples),
            seed=20260831 + index,
        )
        rank.insert(0, "exposure_power", config.exposure_power)
        soft_rank_rows.append(rank)

    soft_models = pd.concat(soft_model_rows, ignore_index=True)
    soft_folds = pd.concat(soft_fold_rows, ignore_index=True)
    soft_competitions = pd.concat(soft_competition_rows, ignore_index=True)
    soft_loss_uncertainty = pd.concat(soft_loss_rows, ignore_index=True)
    soft_rank_uncertainty = pd.concat(soft_rank_rows, ignore_index=True)

    nested_selections = nested_selection(surface, configs, folds)
    nested_unseen, nested_folds = build_unseen(
        nested_selections, predictions, candidate_seeds, target, folds
    )
    nested_model = model_summary(nested_unseen, nested_folds)

    historical = historical_impact(candidate_seeds[primary.key])
    current = current_snapshot_impact(primary)
    current_cups = current_cup_impact(primary)
    cup_candidate_summary = current_cup_candidate_summary(configs)
    nested_candidate = nested_model.loc[
        nested_model["model"].eq("NESTED_RECALIBRATION")
    ].iloc[0]
    fixed_candidate = primary_model.loc[
        primary_model["model"].eq("FIXED_LINEAR_0.65")
    ].iloc[0]
    brier_wins = int((primary_folds["delta_brier_1x2"] < 0.0).sum())
    log_wins = int((primary_folds["delta_log_loss_1x2"] < 0.0).sum())
    reliable_rank_harm = bool(primary_rank_uncertainty["reliable_harm"].any())
    competition_harm = bool(
        (primary_competition["delta_brier_1x2"] > 0.0).any()
        or (primary_competition["delta_log_loss_1x2"] > 0.0).any()
    )
    supported = (
        brier_wins >= 4
        and log_wins >= 4
        and float(fixed_candidate["delta_vs_baseline_brier_1x2"]) < 0.0
        and float(fixed_candidate["delta_vs_baseline_log_loss_1x2"]) < 0.0
        and not reliable_rank_harm
        and not competition_harm
    )
    decision = {
        "decision": "SUPPORT_FIXED_LINEAR_0.65" if supported else "KEEP_PRODUCTION",
        "production_changed": False,
        "evidence_window": f"{seasons[0]}-{seasons[-1]}",
        "fold_count": len(folds),
        "unseen_matches": int(len(primary_unseen)),
        "candidate_count": len(configs),
        "primary_candidate_key": primary.key,
        "fixed_brier_fold_wins": brier_wins,
        "fixed_log_loss_fold_wins": log_wins,
        "fixed_brier_delta": float(
            fixed_candidate["delta_vs_baseline_brier_1x2"]
        ),
        "fixed_log_loss_delta": float(
            fixed_candidate["delta_vs_baseline_log_loss_1x2"]
        ),
        "fixed_seed_spearman_delta": float(
            fixed_candidate["delta_vs_baseline_seed_spearman"]
        ),
        "fixed_seed_pairwise_delta": float(
            fixed_candidate["delta_vs_baseline_seed_pairwise_accuracy"]
        ),
        "fixed_ranking_reliable_harm": reliable_rank_harm,
        "fixed_competition_harm": competition_harm,
        "full_history_diagnostic_selection": select_candidate(full_summary),
        "nested_selected_keys": nested_selections[
            "selected_candidate_key"
        ].tolist(),
        "nested_brier_delta": float(
            nested_candidate["delta_vs_baseline_brier_1x2"]
        ),
        "nested_log_loss_delta": float(
            nested_candidate["delta_vs_baseline_log_loss_1x2"]
        ),
        "soft_any_reliable_loss_improvement": bool(
            soft_loss_uncertainty["reliable_improvement"].any()
        ),
        "soft_any_reliable_loss_harm": bool(
            soft_loss_uncertainty["reliable_harm"].any()
        ),
        "production_contract_sha256": contract_hash,
    }

    surface.to_csv(output / "candidate_surface.csv", index=False)
    fixed_curve.to_csv(output / "fixed_unseen_curve.csv", index=False)
    full_summary.to_csv(output / "full_history_summary.csv", index=False)
    primary_unseen.to_csv(output / "fixed_linear_065_unseen_predictions.csv", index=False)
    primary_folds.to_csv(output / "fixed_linear_065_fold_results.csv", index=False)
    primary_model.to_csv(output / "fixed_linear_065_model_comparison.csv", index=False)
    primary_competition.to_csv(
        output / "fixed_linear_065_competition_summary.csv", index=False
    )
    primary_loss_uncertainty.to_csv(
        output / "fixed_linear_065_loss_uncertainty.csv", index=False
    )
    primary_rank_uncertainty.to_csv(
        output / "fixed_linear_065_ranking_uncertainty.csv", index=False
    )
    soft_models.to_csv(output / "soft_model_comparison.csv", index=False)
    soft_folds.to_csv(output / "soft_fold_results.csv", index=False)
    soft_competitions.to_csv(output / "soft_competition_summary.csv", index=False)
    soft_loss_uncertainty.to_csv(
        output / "soft_loss_uncertainty.csv", index=False
    )
    soft_rank_uncertainty.to_csv(
        output / "soft_ranking_uncertainty.csv", index=False
    )
    nested_selections.to_csv(output / "nested_selections.csv", index=False)
    nested_folds.to_csv(output / "nested_fold_results.csv", index=False)
    nested_model.to_csv(output / "nested_model_comparison.csv", index=False)
    historical.to_csv(output / "historical_seed_impact_by_exposure.csv", index=False)
    current.to_csv(output / "current_2026_27_impact.csv", index=False)
    current_cups.to_csv(output / "current_cup_impact.csv", index=False)
    cup_candidate_summary.to_csv(
        output / "current_cup_candidate_summary.csv", index=False
    )
    (output / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "REPORT.md").write_text(
        report_text(
            decision,
            fixed_curve,
            primary_model,
            primary_folds,
            primary_competition,
            primary_loss_uncertainty,
            primary_rank_uncertainty,
            nested_model,
            nested_selections,
            current_cups,
            cup_candidate_summary,
            soft_models,
            soft_loss_uncertainty,
            soft_rank_uncertainty,
        ),
        encoding="utf-8",
    )

    if hashlib.sha256(PRODUCTION_CONTRACT.read_bytes()).hexdigest() != contract_hash:
        raise ValueError("Production contract changed during linear exposure backtest")
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
