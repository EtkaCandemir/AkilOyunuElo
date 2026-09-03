from __future__ import annotations

"""Walk-forward ablation of a bounded bridge for European exposure.

Production uses ``min(raw_exposure, 0.65)``.  A bridge candidate leaves raw
exposure unchanged through a knee and maps the remaining evidence interval
continuously into the frozen 0.65 maximum:

``knee + (0.65-knee) * ((raw-knee)/(1-knee))**gamma``.

Every other AO First and replay parameter remains pinned to active production.
The command writes research output only; it never changes contracts or
artifacts.
"""

import argparse
from dataclasses import replace
import hashlib
import json
import math
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
from scripts.run_european_exposure_tail_backtest import (  # noqa: E402
    DEFAULT_BOOTSTRAP_SAMPLES,
    EXPECTED_FOLDS,
    EXPECTED_UNSEEN_MATCHES,
    add_fold_accuracy,
    add_model_accuracy,
    attach_baseline_classes,
    competition_with_accuracy,
    fixed_selection,
    metric_goal,
)
from scripts.run_european_prior_recalibration_backtest import (  # noqa: E402
    BASELINE_CONFIG,
    aggregate_candidate_surface,
    build_unseen,
    load_seed_evidence,
    model_summary,
    nested_selection,
    season_surface,
)
from scripts.run_linear_european_exposure_backtest import (  # noqa: E402
    current_cup_impact,
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


OUTPUT_ROOT = ROOT / "output" / "bounded_european_exposure_backtest_2018_2026"
EXPOSURE_KNEES = (0.40, 0.50, 0.55, 0.60, 0.625, 0.64)
EXPOSURE_POWERS = (1.0, 1.5, 2.0, 3.0)
# Declared before examining this surface.  It is the simplest local bridge:
# raw evidence is untouched through 0.60 and the last 0.40 is mapped linearly
# into the remaining 0.05 of the frozen European share.
PRIMARY_FIXED_KNEE = 0.60
PRIMARY_FIXED_POWER = 1.0


def candidate_grid():
    candidates = tuple(
        replace(
            BASELINE_CONFIG,
            exposure_family="CAPPED_BRIDGE",
            exposure_knee=knee,
            exposure_power=power,
        )
        for knee in EXPOSURE_KNEES
        for power in EXPOSURE_POWERS
    )
    configs = (BASELINE_CONFIG, *candidates)
    keys = [config.key for config in configs]
    if len(keys) != len(set(keys)):
        raise ValueError("CAPPED_BRIDGE candidate keys must be unique")
    return configs


def candidate_metadata(configs) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_key": config.key,
                "exposure_family": config.exposure_family,
                "exposure_knee": (
                    float("nan")
                    if config.exposure_family == "HARD_CAP"
                    else float(config.exposure_knee)
                ),
                "exposure_power": (
                    float("nan")
                    if config.exposure_family == "HARD_CAP"
                    else float(config.exposure_power)
                ),
                "maximum_exposure_weight": float(config.exposure_cap),
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
        ["season", "brier_1x2", "log_loss_1x2", "accuracy_1x2"]
    ].rename(
        columns={
            "brier_1x2": "baseline_brier_1x2",
            "log_loss_1x2": "baseline_log_loss_1x2",
            "accuracy_1x2": "baseline_accuracy_1x2",
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
                "accuracy_fold_wins": int(
                    (
                        compared["accuracy_1x2"]
                        > compared["baseline_accuracy_1x2"]
                    ).sum()
                ),
            }
        )
    return (
        summary.merge(pd.DataFrame(wins), on="candidate_key", validate="one_to_one")
        .merge(metadata, on="candidate_key", validate="one_to_one")
        .sort_values(
            ["exposure_knee", "exposure_power", "candidate_key"],
            na_position="first",
            kind="stable",
        )
        .reset_index(drop=True)
    )


def fixed_evidence(
    configs,
    folds,
    predictions,
    candidate_seeds,
    target,
    bootstrap_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_rows = []
    fold_rows = []
    competition_rows = []
    loss_rows = []
    ranking_rows = []
    baseline_predictions = predictions[BASELINE_CONFIG.key]
    for index, config in enumerate(configs, start=1):
        if config.key == BASELINE_CONFIG.key:
            continue
        selections = fixed_selection(config.key, folds)
        unseen, fold_result = build_unseen(
            selections, predictions, candidate_seeds, target, folds
        )
        unseen = attach_baseline_classes(unseen, baseline_predictions)
        fold_result = add_fold_accuracy(fold_result, unseen)
        model = add_model_accuracy(model_summary(unseen, fold_result), unseen).loc[
            lambda frame: frame["model"].eq("NESTED_RECALIBRATION")
        ]
        competition = competition_with_accuracy(unseen)
        loss = dependency_uncertainty(unseen, bootstrap_samples)
        ranking = ranking_uncertainty_summary(
            fold_result,
            bootstrap_samples,
            seed=20260902 + index,
        )
        for frame in (model, fold_result, competition, loss, ranking):
            frame.insert(0, "exposure_power", config.exposure_power)
            frame.insert(0, "exposure_knee", config.exposure_knee)
            frame.insert(0, "candidate_key", config.key)
        model_rows.append(model)
        fold_rows.append(fold_result)
        competition_rows.append(competition)
        loss_rows.append(loss)
        ranking_rows.append(ranking)
    return (
        pd.concat(model_rows, ignore_index=True),
        pd.concat(fold_rows, ignore_index=True),
        pd.concat(competition_rows, ignore_index=True),
        pd.concat(loss_rows, ignore_index=True),
        pd.concat(ranking_rows, ignore_index=True),
    )


def current_cup_evidence(configs) -> tuple[pd.DataFrame, pd.DataFrame]:
    impacts = []
    summaries = []
    for config in configs:
        impact = current_cup_impact(config)
        impact.insert(
            0,
            "exposure_power",
            float("nan")
            if config.exposure_family == "HARD_CAP"
            else config.exposure_power,
        )
        impact.insert(
            0,
            "exposure_knee",
            float("nan")
            if config.exposure_family == "HARD_CAP"
            else config.exposure_knee,
        )
        impact.insert(0, "candidate_key", config.key)
        impacts.append(impact)
        delta = impact["candidate_elo_delta"]
        summaries.append(
            {
                "candidate_key": config.key,
                "exposure_family": config.exposure_family,
                "exposure_knee": (
                    float("nan")
                    if config.exposure_family == "HARD_CAP"
                    else float(config.exposure_knee)
                ),
                "exposure_power": (
                    float("nan")
                    if config.exposure_family == "HARD_CAP"
                    else float(config.exposure_power)
                ),
                "maximum_exposure_weight": float(config.exposure_cap),
                "teams": len(impact),
                "teams_down": int((delta < -1e-9).sum()),
                "teams_up": int((delta > 1e-9).sum()),
                "teams_unchanged": int((delta.abs() <= 1e-9).sum()),
                "teams_changed_over_10": int(delta.abs().gt(10.0).sum()),
                "teams_changed_over_50": int(delta.abs().gt(50.0).sum()),
                "mean_absolute_change": float(delta.abs().mean()),
                "maximum_absolute_change": float(delta.abs().max()),
                "mean_rank_change": float(impact["rank_change"].mean()),
                "maximum_absolute_rank_change": int(
                    impact["rank_change"].abs().max()
                ),
            }
        )
    return pd.concat(impacts, ignore_index=True), pd.DataFrame(summaries)


def build_report(
    decision: dict[str, object],
    fixed_curve: pd.DataFrame,
    fixed_competition: pd.DataFrame,
    fixed_loss: pd.DataFrame,
    fixed_ranking: pd.DataFrame,
    nested_selections: pd.DataFrame,
    nested_model: pd.DataFrame,
    nested_folds: pd.DataFrame,
    nested_competition: pd.DataFrame,
    nested_loss: pd.DataFrame,
    nested_ranking: pd.DataFrame,
    current_summary: pd.DataFrame,
) -> str:
    curve_columns = [
        "exposure_family",
        "exposure_knee",
        "exposure_power",
        "maximum_exposure_weight",
        "brier_1x2",
        "delta_vs_baseline_brier_1x2",
        "log_loss_1x2",
        "delta_vs_baseline_log_loss_1x2",
        "accuracy_1x2",
        "delta_vs_baseline_accuracy_1x2",
        "seed_spearman",
        "delta_vs_baseline_seed_spearman",
        "seed_pairwise_accuracy",
        "delta_vs_baseline_seed_pairwise_accuracy",
        "brier_fold_wins",
        "log_loss_fold_wins",
        "accuracy_fold_wins",
    ]
    competition_columns = [
        "exposure_knee",
        "exposure_power",
        "competition",
        "matches",
        "delta_brier_1x2",
        "delta_log_loss_1x2",
        "delta_accuracy_1x2",
    ]
    uncertainty_columns = [
        "exposure_knee",
        "exposure_power",
        "metric",
        "method",
        "mean_difference",
        "ci_95_lower",
        "ci_95_upper",
        "reliable_improvement",
        "reliable_harm",
    ]
    return "\n".join(
        [
            "# Bounded European Exposure Bridge Backtest",
            "",
            "## Sonuç",
            "",
            f"- Karar: `{decision['decision']}`",
            f"- Ana sabit aday: `knee={PRIMARY_FIXED_KNEE:g}, "
            f"gamma={PRIMARY_FIXED_POWER:g}`; hedefi karşıladı: "
            f"`{decision['primary_fixed_meets_metric_goal']}`",
            "- Yalnız bridge gridinde nested train seçimi hedefi karşıladı: "
            f"`{decision['nested_meets_metric_goal']}`",
            f"- Nested Brier / log-loss / accuracy farkları: "
            f"`{decision['nested_brier_delta']:+.6f}` / "
            f"`{decision['nested_log_loss_delta']:+.6f}` / "
            f"`{decision['nested_accuracy_delta']:+.6f}`",
            "",
            "Promotion yalnız ana sabit aday veya geçmiş train sezonlarından seçilen "
            "nested politika pooled Brier ve log-loss'u düşürürken accuracy'yi "
            "artırırsa desteklenir. Test eğrisinden sonradan en iyi knee/gamma "
            "satırını seçmek holdout değildir ve promotion gerekçesi sayılmamıştır.",
            "",
            "## Formül ve sabitler",
            "",
            "- Production: `effective = min(raw, 0.65)`",
            "- Challenger, `raw <= knee` iken `effective = raw`.",
            "- Üstte: `effective = knee + (0.65-knee) * "
            "((raw-knee)/(1-knee))^gamma`.",
            f"- Knee grid: `{', '.join(f'{value:g}' for value in EXPOSURE_KNEES)}`",
            f"- Gamma grid: `{', '.join(f'{value:g}' for value in EXPOSURE_POWERS)}`",
            "- Her aday `raw=1` için tam olarak `0.65` üretir; frozen 65/35 "
            "maksimumu korunur.",
            "- Diğer AO First ve replay parametreleri aktif production değerlerine pinli.",
            "",
            "## Veri ve yöntem",
            "",
            f"- Sezon penceresi: `{decision['evidence_window']}`",
            f"- Expanding outer fold: `{decision['fold_count']}`",
            f"- Görülmemiş maç: `{decision['unseen_matches']}`",
            f"- Dependency bootstrap örneklemi: `{decision['bootstrap_samples']}`",
            "",
            "## Sabit bridge eğrisi (görülmemiş altı sezon)",
            "",
            markdown_table(fixed_curve[curve_columns]),
            "",
            "Bu tablonun en iyi satırı tanısaldır; test sezonlarını gördüğü için "
            "yeni bir sabit seçmekte kullanılamaz.",
            "",
            "## Sabit bridge kupa segmentleri",
            "",
            markdown_table(fixed_competition[competition_columns]),
            "",
            "## Sabit bridge kayıp belirsizliği",
            "",
            markdown_table(fixed_loss[uncertainty_columns]),
            "",
            "## Sabit bridge sıralama belirsizliği",
            "",
            markdown_table(fixed_ranking[uncertainty_columns]),
            "",
            "## Bridge-only nested train seçimi",
            "",
            markdown_table(nested_selections),
            "",
            markdown_table(nested_model),
            "",
            "### Nested foldlar",
            "",
            markdown_table(nested_folds),
            "",
            "### Nested kupa segmentleri",
            "",
            markdown_table(nested_competition),
            "",
            "### Nested kayıp ve sıralama belirsizliği",
            "",
            markdown_table(nested_loss),
            "",
            markdown_table(nested_ranking),
            "",
            "## 2026/27 güncel kupa etkisi",
            "",
            markdown_table(current_summary),
            "",
            "## Kanıt sınırı",
            "",
            "Bu çalışma 2018/19-2025/26 historical replay kanıtıdır; 2026/27 "
            "prospective sonucu değildir. Accuracy için ayrı güven aralığı "
            "hesaplanmaz. Grid taramasındaki post-hoc en iyi sabit aday promotion "
            "kanıtı değildir. Production contract, config ve artifact zinciri "
            "değiştirilmemiştir.",
            "",
            "## Yeniden üretme",
            "",
            "`python3 scripts/run_bounded_european_exposure_backtest.py`",
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES
    )
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples must be at least 100")
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract_hash = hashlib.sha256(PRODUCTION_CONTRACT.read_bytes()).hexdigest()

    _, datasets, core, parameters, production_seed_map = load_production_baseline()
    seasons = tuple(data.season for data in datasets)
    folds = expanding_folds(seasons)
    if len(folds) != EXPECTED_FOLDS:
        raise ValueError(f"Expected {EXPECTED_FOLDS} folds, observed {len(folds)}")
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
            row["exposure_knee"] = (
                float("nan")
                if config.exposure_family == "HARD_CAP"
                else config.exposure_knee
            )
            row["exposure_power"] = (
                float("nan")
                if config.exposure_family == "HARD_CAP"
                else config.exposure_power
            )
        surface_rows.extend(rows)
        print(f"Replayed {index}/{len(configs)}: {config.key}", flush=True)

    surface = pd.DataFrame(surface_rows)
    fixed_curve = fixed_unseen_summary(surface, folds, metadata)
    baseline_row = fixed_curve.loc[
        fixed_curve["candidate_key"].eq(BASELINE_CONFIG.key)
    ].iloc[0]
    if int(baseline_row["matches"]) != EXPECTED_UNSEEN_MATCHES:
        raise ValueError(
            f"Expected {EXPECTED_UNSEEN_MATCHES} unseen matches, observed "
            f"{int(baseline_row['matches'])}"
        )

    fixed_models, fixed_folds, fixed_competition, fixed_loss, fixed_ranking = (
        fixed_evidence(
            configs,
            folds,
            predictions,
            candidate_seeds,
            target,
            int(args.bootstrap_samples),
        )
    )

    nested_selections = nested_selection(surface, configs, folds)
    nested_unseen, nested_folds = build_unseen(
        nested_selections, predictions, candidate_seeds, target, folds
    )
    nested_unseen = attach_baseline_classes(
        nested_unseen, predictions[BASELINE_CONFIG.key]
    )
    nested_folds = add_fold_accuracy(nested_folds, nested_unseen)
    nested_model = add_model_accuracy(
        model_summary(nested_unseen, nested_folds), nested_unseen
    )
    nested_competition = competition_with_accuracy(nested_unseen)
    nested_loss = dependency_uncertainty(
        nested_unseen, int(args.bootstrap_samples)
    )
    nested_ranking = ranking_uncertainty_summary(
        nested_folds, int(args.bootstrap_samples), seed=20260903
    )

    current_impact, current_summary = current_cup_evidence(configs)
    primary = next(
        config
        for config in configs
        if config.exposure_family == "CAPPED_BRIDGE"
        and math.isclose(config.exposure_knee, PRIMARY_FIXED_KNEE, abs_tol=1e-12)
        and math.isclose(
            config.exposure_power, PRIMARY_FIXED_POWER, abs_tol=1e-12
        )
    )
    primary_row = fixed_curve.loc[
        fixed_curve["candidate_key"].eq(primary.key)
    ].iloc[0]
    nested_row = nested_model.loc[
        nested_model["model"].eq("NESTED_RECALIBRATION")
    ].iloc[0]
    primary_supported = metric_goal(primary_row)
    nested_supported = metric_goal(nested_row)
    posthoc_supported = fixed_curve.loc[
        fixed_curve.apply(metric_goal, axis=1), "candidate_key"
    ].tolist()
    decision = {
        "decision": (
            "SUPPORT_CAPPED_BRIDGE"
            if primary_supported or nested_supported
            else "KEEP_PRODUCTION"
        ),
        "production_changed": False,
        "evidence_window": f"{seasons[0]}-{seasons[-1]}",
        "fold_count": len(folds),
        "unseen_matches": int(len(nested_unseen)),
        "candidate_count_including_baseline": len(configs),
        "bootstrap_samples": int(args.bootstrap_samples),
        "primary_fixed_knee": PRIMARY_FIXED_KNEE,
        "primary_fixed_power": PRIMARY_FIXED_POWER,
        "primary_fixed_candidate_key": primary.key,
        "primary_fixed_meets_metric_goal": primary_supported,
        "primary_fixed_brier_delta": float(
            primary_row["delta_vs_baseline_brier_1x2"]
        ),
        "primary_fixed_log_loss_delta": float(
            primary_row["delta_vs_baseline_log_loss_1x2"]
        ),
        "primary_fixed_accuracy_delta": float(
            primary_row["delta_vs_baseline_accuracy_1x2"]
        ),
        "nested_selected_keys": nested_selections[
            "selected_candidate_key"
        ].tolist(),
        "nested_meets_metric_goal": nested_supported,
        "nested_brier_delta": float(
            nested_row["delta_vs_baseline_brier_1x2"]
        ),
        "nested_log_loss_delta": float(
            nested_row["delta_vs_baseline_log_loss_1x2"]
        ),
        "nested_accuracy_delta": float(
            nested_row["delta_vs_baseline_accuracy_1x2"]
        ),
        "posthoc_fixed_candidates_meeting_metric_goal_not_selection_evidence": (
            posthoc_supported
        ),
        "production_contract_sha256": contract_hash,
    }

    surface.to_csv(output / "candidate_surface.csv", index=False)
    fixed_curve.to_csv(output / "fixed_unseen_curve.csv", index=False)
    fixed_models.to_csv(output / "fixed_model_comparison.csv", index=False)
    fixed_folds.to_csv(output / "fixed_fold_results.csv", index=False)
    fixed_competition.to_csv(output / "fixed_competition_summary.csv", index=False)
    fixed_loss.to_csv(output / "fixed_loss_uncertainty.csv", index=False)
    fixed_ranking.to_csv(output / "fixed_ranking_uncertainty.csv", index=False)
    nested_selections.to_csv(output / "nested_selections.csv", index=False)
    nested_folds.to_csv(output / "nested_fold_results.csv", index=False)
    nested_model.to_csv(output / "nested_model_comparison.csv", index=False)
    nested_competition.to_csv(output / "nested_competition_summary.csv", index=False)
    nested_loss.to_csv(output / "nested_loss_uncertainty.csv", index=False)
    nested_ranking.to_csv(output / "nested_ranking_uncertainty.csv", index=False)
    nested_unseen.to_csv(output / "nested_unseen_predictions.csv", index=False)
    current_impact.to_csv(output / "current_cup_impact.csv", index=False)
    current_summary.to_csv(output / "current_cup_candidate_summary.csv", index=False)
    (output / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "REPORT.md").write_text(
        build_report(
            decision,
            fixed_curve,
            fixed_competition,
            fixed_loss,
            fixed_ranking,
            nested_selections,
            nested_model,
            nested_folds,
            nested_competition,
            nested_loss,
            nested_ranking,
            current_summary,
        ),
        encoding="utf-8",
    )

    final_contract_hash = hashlib.sha256(PRODUCTION_CONTRACT.read_bytes()).hexdigest()
    if final_contract_hash != contract_hash:
        raise ValueError("Production contract changed during bounded bridge backtest")
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)
    print(f"Report: {output / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
