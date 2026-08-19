from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ao_elo.config import AOEuropeanEloConfig  # noqa: E402
from ao_elo.evaluation import (  # noqa: E402
    dependency_robust_loss_difference_ci,
    schedule_adjusted_team_performance,
)
from ao_elo.qualification_stage_k import (  # noqa: E402
    QualificationStageKConfig,
    qualification_round_key,
    reference_config,
)
from ao_elo.qualification_transition import (  # noqa: E402
    QualificationTransitionConfig,
)
from scripts.run_controlled_goal_progression_backtest import prepare_controlled_data  # noqa: E402
from scripts.run_dynamic_core_calibration import expanding_folds  # noqa: E402
from scripts.run_final_robustness import load_team_season_identity, summarize_ranking  # noqa: E402
from scripts.run_qualification_stage_k_backtest import (  # noqa: E402
    build_league_entry_impact,
    build_qualifier_path_summary,
    evaluate_config,
    md_table,
    prediction_metrics,
)
from scripts.run_stage_weighted_progression_backtest import (  # noqa: E402
    DOMESTIC_ADJUSTMENTS,
    DYNAMIC_MANIFEST,
    EVENTS_PATH,
    PRODUCTION_CONTRACT,
    STATIC_DATA_ROOT,
    XG_DATA,
    aggregate_ranking,
    load_domestic_adjustments,
    load_xg_map,
    validate_production_contract,
)
from scripts.run_v2_achievement_reserve_calibration import load_reserve_data  # noqa: E402
from scripts.run_v2_evaluation_upgrade import read_events  # noqa: E402


OUTPUT_ROOT = ROOT / "output" / "qualification_stage_k_carry_backtest_2018_2026"
REFERENCE = "FULL_CARRY_100_REFERENCE"
NESTED = "NESTED_STAGE_K_CARRY"
RANK_TOLERANCE = 1e-9


def transition_grid() -> tuple[QualificationTransitionConfig, ...]:
    full = reference_config()
    mild = QualificationStageKConfig("MILD_STAGE_K", 0.70, 0.80, 0.90, 0.95)
    balanced = QualificationStageKConfig(
        "BALANCED_STAGE_K", 0.60, 0.75, 0.85, 0.95
    )
    configs = [
        QualificationTransitionConfig(REFERENCE, full, 1.0, selectable=False)
    ]
    for stage_name, stage in (("FULL", full), ("MILD", mild), ("BALANCED", balanced)):
        for carry in (0.50, 0.60, 0.75, 1.00):
            if stage_name == "FULL" and carry == 1.00:
                continue
            configs.append(
                QualificationTransitionConfig(
                    f"{stage_name}_CARRY_{int(carry * 100):03d}",
                    stage,
                    carry,
                    selectable=True,
                )
            )
    result = tuple(configs)
    for config in result:
        config.validate()
    if len(result) != 12 or sum(config.selectable for config in result) != 11:
        raise ValueError("Unexpected stage-K/carry candidate grid")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Main-stage evaluation of qualification stage-K and carry"
    )
    parser.add_argument("--static-data-root", type=Path, default=STATIC_DATA_ROOT)
    parser.add_argument("--events", type=Path, default=EVENTS_PATH)
    parser.add_argument("--dynamic-manifest", type=Path, default=DYNAMIC_MANIFEST)
    parser.add_argument("--production-contract", type=Path, default=PRODUCTION_CONTRACT)
    parser.add_argument("--domestic-adjustments", type=Path, default=DOMESTIC_ADJUSTMENTS)
    parser.add_argument("--xg-data", type=Path, default=XG_DATA)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples must be at least 100")

    contract_path = args.production_contract.resolve()
    contract_hash = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    core, parameters = validate_production_contract(contract)
    dynamic = json.loads(args.dynamic_manifest.resolve().read_text(encoding="utf-8"))
    static_config = AOEuropeanEloConfig(**dynamic["static_config"])
    static_config.validate()
    events = read_events(args.events.resolve())
    events["qualification_round_key"] = events["round"].map(qualification_round_key)
    main_events = events.loc[events["qualification_round_key"].eq("MAIN")].copy()
    main_target = schedule_adjusted_team_performance(main_events)
    reserve, tie_audit = load_reserve_data(
        args.static_data_root.resolve(), args.events.resolve(), static_config
    )
    datasets = prepare_controlled_data(reserve, events)
    seasons = tuple(data.season for data in datasets)
    folds = expanding_folds(seasons)
    if len(folds) != 6:
        raise ValueError(f"Expected six folds, found {len(folds)}")
    identity = load_team_season_identity()
    domestic = load_domestic_adjustments(args.domestic_adjustments.resolve(), datasets)
    xg = load_xg_map(args.xg_data.resolve(), datasets)
    event_rounds = events.set_index("match_id")["round"].astype(str).to_dict()
    configs = transition_grid()

    evaluations = {}
    for index, config in enumerate(configs, start=1):
        print(f"Evaluating {index}/{len(configs)}: {config.profile}", flush=True)
        evaluations[config.profile] = evaluate_config(
            datasets,
            main_target,
            core=core,
            parameters=parameters,
            domestic=domestic,
            xg=xg,
            event_rounds=event_rounds,
            config=config.stage_k,
            qualifier_carry=config.qualifier_carry,
            model_name=config.profile,
        )

    surface = candidate_surface(
        evaluations, configs, folds, main_target, identity, seasons
    )
    nested = nested_backtest(
        evaluations, configs, folds, main_target, identity, seasons
    )
    competition = competition_summary(nested["predictions"])
    forward = forward_ranking_summary(
        nested["end_ratings"], main_target, identity, seasons
    )
    entry = build_league_entry_impact(
        nested["league_entry"], identity, reference_model=REFERENCE
    )
    paths = build_qualifier_path_summary(entry)
    uncertainty = dependency_uncertainty(
        nested["predictions"], bootstrap_samples=args.bootstrap_samples
    )
    decision = decide(surface, nested["fold_results"], competition, forward, uncertainty, entry)
    safety = safety_audit(
        events,
        main_events,
        configs,
        nested,
        contract_hash,
        hashlib.sha256(contract_path.read_bytes()).hexdigest(),
    )
    if not safety["passed"].all():
        raise ValueError(
            f"Safety audit failed: {safety.loc[~safety['passed'], 'check'].tolist()}"
        )

    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    surface.to_csv(output / "candidate_surface.csv", index=False)
    nested["fold_selections"].to_csv(output / "fold_selections.csv", index=False)
    nested["fold_results"].to_csv(output / "fold_results.csv", index=False)
    nested["predictions"].to_csv(output / "unseen_main_stage_predictions.csv", index=False)
    competition.to_csv(output / "competition_summary.csv", index=False)
    forward.to_csv(output / "forward_ranking_summary.csv", index=False)
    entry.to_csv(output / "league_entry_rating_impact.csv", index=False)
    paths.to_csv(output / "qualifier_path_summary.csv", index=False)
    uncertainty.to_csv(output / "dependency_uncertainty.csv", index=False)
    safety.to_csv(output / "safety_audit.csv", index=False)
    tie_audit.to_csv(output / "tie_chronology_audit.csv", index=False)
    (output / "selected_candidate.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "backtest_report.md").write_text(
        report(
            contract,
            events,
            main_events,
            surface,
            nested["fold_selections"],
            nested["fold_results"],
            competition,
            forward,
            entry,
            paths,
            uncertainty,
            safety,
            decision,
        ),
        encoding="utf-8",
    )
    print(f"Decision: {decision['decision']}")
    print(f"Candidate: {decision['full_history_candidate']}")
    print(f"Report: {output / 'backtest_report.md'}")


def metrics(
    evaluation,
    seasons: set[str],
    target: pd.DataFrame,
    identity: pd.DataFrame,
) -> dict[str, float | int]:
    predictions = evaluation.predictions.loc[
        evaluation.predictions["season"].isin(seasons)
        & evaluation.predictions["qualification_round_key"].eq("MAIN")
    ]
    ranking = aggregate_ranking(
        evaluation.same_season_ranking.loc[
            evaluation.same_season_ranking["season"].isin(seasons)
        ]
    )
    all_rank = ranking.loc[ranking["competition"].eq("ALL")].iloc[0]
    source = evaluation.end_ratings.loc[
        evaluation.end_ratings["season"].isin(seasons)
        & evaluation.end_ratings["reached_main"].astype(bool)
    ]
    forward = summarize_ranking(
        source,
        target,
        allowed_target_seasons=set(seasons),
        identity=identity,
    )
    forward_all = forward.loc[forward["competition"].eq("ALL")]
    entry = evaluation.league_entry.loc[
        evaluation.league_entry["season"].isin(seasons)
        & evaluation.league_entry["entry_type"].eq("QUALIFIER")
    ]
    return {
        **prediction_metrics(predictions),
        "main_stage_spearman": float(all_rank["ranking_score"]),
        "main_stage_pairwise_accuracy": float(all_rank["pairwise_accuracy"]),
        "forward_spearman": float(forward_all.iloc[0]["ranking_score"])
        if len(forward_all) else np.nan,
        "forward_pairwise_accuracy": float(forward_all.iloc[0]["pairwise_accuracy"])
        if len(forward_all) else np.nan,
        "qualifier_entry_mean_abs_change": float(
            entry["qualifier_rating_change"].abs().mean()
        ),
        "qualifier_entry_p95_abs_change": float(
            entry["qualifier_rating_change"].abs().quantile(0.95)
        ),
        "qualifier_entry_max_abs_change": float(
            entry["qualifier_rating_change"].abs().max()
        ),
    }


def candidate_surface(evaluations, configs, folds, target, identity, seasons) -> pd.DataFrame:
    test_seasons = {test for _, test in folds}
    rows = []
    for config in configs:
        rows.append(
            {
                "candidate": config.profile,
                "config_id": config.config_id,
                "selectable": config.selectable,
                "stage_profile": config.stage_k.profile,
                "q1_multiplier": config.stage_k.q1_multiplier,
                "q2_multiplier": config.stage_k.q2_multiplier,
                "q3_multiplier": config.stage_k.q3_multiplier,
                "qualifying_playoff_multiplier": config.stage_k.qualifying_playoff_multiplier,
                "qualifier_carry": config.qualifier_carry,
                **metrics(evaluations[config.profile], test_seasons, target, identity),
            }
        )
    result = pd.DataFrame(rows)
    baseline = result.loc[result["candidate"].eq(REFERENCE)].iloc[0]
    for column in (
        "brier_1x2",
        "log_loss_1x2",
        "accuracy_1x2",
        "main_stage_spearman",
        "main_stage_pairwise_accuracy",
        "forward_spearman",
        "forward_pairwise_accuracy",
        "qualifier_entry_mean_abs_change",
        "qualifier_entry_p95_abs_change",
    ):
        result[f"delta_vs_reference_{column}"] = result[column] - float(baseline[column])
    return result


def select_config(evaluations, configs, train, target, identity):
    rows = []
    for config in configs:
        rows.append(
            {
                "candidate": config.profile,
                "selectable": config.selectable,
                **metrics(evaluations[config.profile], set(train), target, identity),
            }
        )
    frame = pd.DataFrame(rows)
    candidates = frame.loc[frame["selectable"]].copy()
    selected = candidates.sort_values(
        [
            "forward_pairwise_accuracy",
            "forward_spearman",
            "brier_1x2",
            "log_loss_1x2",
            "main_stage_pairwise_accuracy",
            "candidate",
        ],
        ascending=[False, False, True, True, False, True],
        na_position="last",
        kind="stable",
    ).iloc[0]
    return next(config for config in configs if config.profile == selected["candidate"]), selected


def nested_backtest(evaluations, configs, folds, target, identity, seasons):
    selection_rows, fold_rows = [], []
    predictions, end_ratings, entries = [], [], []
    for fold, (train, test) in enumerate(folds, start=1):
        selected, train_metrics = select_config(
            evaluations, configs, train, target, identity
        )
        selection_rows.append(
            {
                "fold": fold,
                "train_seasons": "|".join(train),
                "test_season": test,
                "selected_candidate": selected.profile,
                "selected_config_id": selected.config_id,
                "stage_profile": selected.stage_k.profile,
                "qualifier_carry": selected.qualifier_carry,
                "train_forward_spearman": train_metrics["forward_spearman"],
                "train_forward_pairwise_accuracy": train_metrics["forward_pairwise_accuracy"],
                "train_brier_1x2": train_metrics["brier_1x2"],
                "train_log_loss_1x2": train_metrics["log_loss_1x2"],
            }
        )
        for model, candidate in ((REFERENCE, REFERENCE), (NESTED, selected.profile)):
            evaluation = evaluations[candidate]
            row_metrics = metrics(evaluation, {test}, target, identity)
            fold_rows.append(
                {
                    "fold": fold,
                    "test_season": test,
                    "model": model,
                    "candidate": candidate,
                    **row_metrics,
                }
            )
            for source, destination in (
                (evaluation.predictions, predictions),
                (evaluation.end_ratings, end_ratings),
                (evaluation.league_entry, entries),
            ):
                part = source.loc[source["season"].eq(test)].copy()
                if source is evaluation.predictions:
                    part = part.loc[part["qualification_round_key"].eq("MAIN")]
                part["fold"] = fold
                part["model"] = model
                part["candidate"] = candidate
                destination.append(part)
        print(f"  fold {fold}/6 {test}: {selected.profile}", flush=True)
    fold_results = pd.DataFrame(fold_rows)
    baseline = fold_results.loc[fold_results["model"].eq(REFERENCE)].set_index("fold")
    for column in (
        "brier_1x2",
        "log_loss_1x2",
        "accuracy_1x2",
        "main_stage_spearman",
        "main_stage_pairwise_accuracy",
    ):
        fold_results[f"delta_vs_reference_{column}"] = fold_results.apply(
            lambda row: row[column] - baseline.loc[row["fold"], column], axis=1
        )
    return {
        "fold_selections": pd.DataFrame(selection_rows),
        "fold_results": fold_results,
        "predictions": pd.concat(predictions, ignore_index=True),
        "end_ratings": pd.concat(end_ratings, ignore_index=True),
        "league_entry": pd.concat(entries, ignore_index=True),
    }


def competition_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, competition), frame in predictions.groupby(["model", "competition"]):
        rows.append({"model": model, "competition": competition, **prediction_metrics(frame)})
    result = pd.DataFrame(rows)
    baseline = result.loc[result["model"].eq(REFERENCE)].set_index("competition")
    for column in ("brier_1x2", "log_loss_1x2", "accuracy_1x2"):
        result[f"delta_vs_reference_{column}"] = result.apply(
            lambda row: row[column] - baseline.loc[row["competition"], column], axis=1
        )
    return result


def forward_ranking_summary(end_ratings, target, identity, seasons):
    rows = []
    next_season = {seasons[i]: seasons[i + 1] for i in range(len(seasons) - 1)}
    for (model, fold), frame in end_ratings.groupby(["model", "fold"]):
        source = str(frame["season"].iloc[0])
        destination = next_season.get(source)
        if destination is None:
            continue
        eligible = frame.loc[frame["reached_main"].astype(bool)]
        ranking = summarize_ranking(
            eligible,
            target,
            allowed_target_seasons={destination},
            identity=identity,
        )
        for row in ranking.itertuples(index=False):
            rows.append(
                {
                    "model": model,
                    "fold": fold,
                    "source_season": source,
                    "target_season": destination,
                    "competition": row.competition,
                    "spearman": row.ranking_score,
                    "pairwise_accuracy": row.pairwise_accuracy,
                }
            )
    result = pd.DataFrame(rows)
    baseline = result.loc[result["model"].eq(REFERENCE)].set_index(
        ["fold", "competition"]
    )
    result["spearman_delta_vs_reference"] = result.apply(
        lambda row: row["spearman"] - baseline.loc[(row["fold"], row["competition"]), "spearman"],
        axis=1,
    )
    result["pairwise_delta_vs_reference"] = result.apply(
        lambda row: row["pairwise_accuracy"] - baseline.loc[(row["fold"], row["competition"]), "pairwise_accuracy"],
        axis=1,
    )
    return result


def dependency_uncertainty(predictions, *, bootstrap_samples):
    candidate = predictions.loc[predictions["model"].eq(NESTED)]
    baseline = predictions.loc[predictions["model"].eq(REFERENCE)]
    rows = []
    for competition in ("ALL", "UCL", "UEL", "UECL"):
        left = candidate if competition == "ALL" else candidate.loc[candidate["competition"].eq(competition)]
        right = baseline if competition == "ALL" else baseline.loc[baseline["competition"].eq(competition)]
        paired = left.merge(
            right[["match_id", "brier_1x2", "log_loss_1x2"]],
            on="match_id",
            suffixes=("_candidate", "_baseline"),
            validate="one_to_one",
        )
        for loss in ("brier_1x2", "log_loss_1x2"):
            sample = paired[["season", "match_id", "home_team_id", "away_team_id", "kickoff_utc", "tie_id"]].copy()
            sample["loss_difference"] = paired[f"{loss}_candidate"] - paired[f"{loss}_baseline"]
            result = dependency_robust_loss_difference_ci(sample, bootstrap_samples=bootstrap_samples)
            result.insert(0, "competition", competition)
            result.insert(1, "metric", loss)
            rows.append(result)
    return pd.concat(rows, ignore_index=True)


def decide(surface, folds, competition, forward, uncertainty, entry):
    baseline = surface.loc[surface["candidate"].eq(REFERENCE)].iloc[0]
    candidates = surface.loc[surface["selectable"]].copy()
    candidates["loss_safe"] = (
        candidates["brier_1x2"].le(float(baseline["brier_1x2"]) + 1e-12)
        & candidates["log_loss_1x2"].le(float(baseline["log_loss_1x2"]) + 1e-12)
    )
    pool = candidates.loc[candidates["loss_safe"]]
    if pool.empty:
        pool = candidates
    best = pool.sort_values(
        ["forward_pairwise_accuracy", "forward_spearman", "brier_1x2", "log_loss_1x2", "candidate"],
        ascending=[False, False, True, True, True],
        kind="stable",
    ).iloc[0]
    nested = folds.loc[folds["model"].eq(NESTED)].reset_index(drop=True)
    reference = folds.loc[folds["model"].eq(REFERENCE)].reset_index(drop=True)
    brier_wins = int((nested["brier_1x2"] < reference["brier_1x2"]).sum())
    log_wins = int((nested["log_loss_1x2"] < reference["log_loss_1x2"]).sum())
    weights = nested["matches"]
    pooled_brier_delta = float(((nested["brier_1x2"] - reference["brier_1x2"]) * weights).sum() / weights.sum())
    pooled_log_delta = float(((nested["log_loss_1x2"] - reference["log_loss_1x2"]) * weights).sum() / weights.sum())
    forward_all = forward.loc[forward["model"].eq(NESTED) & forward["competition"].eq("ALL")]
    forward_wins = int(
        ((forward_all["spearman_delta_vs_reference"] > RANK_TOLERANCE)
         | (forward_all["pairwise_delta_vs_reference"] > RANK_TOLERANCE)).sum()
    )
    conservative = uncertainty.loc[
        uncertainty["method"].eq("conservative_envelope")
    ]
    reliable_harm = bool(conservative["reliable_harm"].astype(bool).any())
    competition_harm = bool(
        conservative.loc[
            conservative["competition"].ne("ALL"), "reliable_harm"
        ].astype(bool).any()
    )
    qualifier = entry.loc[entry["model"].eq(NESTED) & entry["entry_type"].eq("QUALIFIER")]
    p95 = float(qualifier["qualifier_rating_change"].abs().quantile(0.95))
    reference_p95 = float(
        entry.loc[entry["model"].eq(REFERENCE) & entry["entry_type"].eq("QUALIFIER"), "qualifier_rating_change"].abs().quantile(0.95)
    )
    passed = bool(
        brier_wins >= 4
        and log_wins >= 4
        and pooled_brier_delta <= 0.0
        and pooled_log_delta <= 0.0
        and forward_wins >= 3
        and not reliable_harm
        and not competition_harm
        and p95 < reference_p95
    )
    return {
        "decision": "PROMOTE_RESEARCH_CANDIDATE" if passed else "KEEP_REFERENCE",
        "production_changed": False,
        "full_history_candidate": str(best["candidate"]),
        "full_history_config_id": str(best["config_id"]),
        "nested_selected_profiles": nested["candidate"].value_counts().to_dict(),
        "brier_fold_wins": brier_wins,
        "log_loss_fold_wins": log_wins,
        "forward_ranking_wins": forward_wins,
        "pooled_brier_delta": pooled_brier_delta,
        "pooled_log_loss_delta": pooled_log_delta,
        "competition_reliable_harm": competition_harm,
        "dependency_reliable_harm": reliable_harm,
        "qualifier_entry_p95": p95,
        "reference_entry_p95": reference_p95,
        "caveat": "Development-window replay; production requires explicit approval.",
    }


def safety_audit(events, main_events, configs, nested, before_hash, after_hash):
    predictions = nested["predictions"]
    entries = nested["league_entry"]
    candidate_entries = entries.loc[entries["model"].eq(NESTED) & entries["entry_type"].eq("QUALIFIER")]
    symmetric_error = (
        candidate_entries["qualifier_rating_change"]
        - candidate_entries["qualifier_carry"]
        * (candidate_entries["pre_carry_rating"] - candidate_entries["initial_rating"])
    ).abs().max()
    checks = [
        ("production_contract_unchanged", before_hash == after_hash, after_hash, before_hash),
        ("reference_not_selectable", not configs[0].selectable, configs[0].selectable, False),
        ("eleven_selectable_candidates", sum(c.selectable for c in configs) == 11, sum(c.selectable for c in configs), 11),
        ("main_match_count", predictions.loc[predictions["model"].eq(REFERENCE), "match_id"].nunique() == len(main_events.loc[main_events["season"].isin({"2020/21","2021/22","2022/23","2023/24","2024/25","2025/26"})]), predictions.loc[predictions["model"].eq(REFERENCE), "match_id"].nunique(), len(main_events.loc[main_events["season"].isin({"2020/21","2021/22","2022/23","2023/24","2024/25","2025/26"})])),
        ("main_stage_k_full", np.allclose(predictions["stage_k_multiplier"], 1.0), float(predictions["stage_k_multiplier"].min()), 1.0),
        ("match_zero_sum", float(predictions["zero_sum_error"].max()) <= 1e-9, float(predictions["zero_sum_error"].max()), 1e-9),
        ("carry_applied_once", not entries.duplicated(["fold", "model", "season", "team_id"]).any(), int(entries.duplicated(["fold", "model", "season", "team_id"]).sum()), 0),
        ("carry_symmetric", float(symmetric_error) <= 1e-9, float(symmetric_error), 1e-9),
        ("no_2026_27", not predictions["season"].eq("2026/27").any(), int(predictions["season"].eq("2026/27").sum()), 0),
    ]
    return pd.DataFrame([{"check": c, "passed": bool(p), "observed": o, "requirement": r} for c,p,o,r in checks])


def report(contract, events, main_events, surface, selections, folds, competition, forward, entry, paths, uncertainty, safety, decision):
    largest = entry.loc[entry["model"].eq(NESTED) & entry["entry_type"].eq("QUALIFIER")].copy()
    largest = largest.reindex(largest["league_entry_rating_delta_vs_reference"].abs().sort_values(ascending=False).index).head(15)
    return f"""# Qualification Stage-K + Carry Backtesti

## Karar

- Karar: **{decision['decision']}**
- Full-history aday: **{decision['full_history_candidate']}**
- Production değişti: **Hayır**
- Seçim yalnız qualifier sonrası ana aşama maçları ve forward ranking üzerinden yapıldı.

## Veri ve Sözleşme

- Toplam maç: `{len(events)}`; ana aşama maçları: `{len(main_events)}`
- Test sezonları: `2020/21-2025/26`; outer fold: `6`
- Base K: `{contract['dynamic_core']['k_factor']:.6f}`
- Referans: `FULL stage K + carry 1.00`, yalnız karşılaştırma kolu
- Carry AO First anchor'ına pozitif ve negatif yönde simetrik uygulanır.

## Sabit Aday Yüzeyi

{md_table(surface, ['candidate','selectable','stage_profile','q1_multiplier','q2_multiplier','q3_multiplier','qualifying_playoff_multiplier','qualifier_carry','brier_1x2','log_loss_1x2','main_stage_spearman','forward_spearman','forward_pairwise_accuracy','qualifier_entry_p95_abs_change'])}

## Fold Seçimleri

{md_table(selections)}

## Unseen Ana Aşama Sonuçları

{md_table(folds, ['fold','test_season','model','candidate','matches','brier_1x2','log_loss_1x2','accuracy_1x2','main_stage_spearman','main_stage_pairwise_accuracy','delta_vs_reference_brier_1x2','delta_vs_reference_log_loss_1x2'])}

## Turnuva Segmentleri

{md_table(competition)}

## Forward Ranking

{md_table(forward)}

## En Büyük Lig Giriş Farkları

{md_table(largest, ['season','team_name','country_code','club_id','candidate','qualifier_matches','qualifier_wins','qualifier_carry','pre_carry_rating','league_entry_rating','reference_league_entry_rating','league_entry_rating_delta_vs_reference','entry_rank_change_vs_reference'])}

## Yol Uzunluğu ve Doğrudan Katılım

{md_table(paths)}

## Belirsizlik

{md_table(uncertainty)}

## Güvenlik

{md_table(safety)}

## Sonuç Özeti

- Brier fold galibiyeti: `{decision['brier_fold_wins']}/6`
- Log-loss fold galibiyeti: `{decision['log_loss_fold_wins']}/6`
- Forward ranking galibiyeti: `{decision['forward_ranking_wins']}/5`
- Pooled Brier farkı: `{decision['pooled_brier_delta']:+.9f}`
- Pooled log-loss farkı: `{decision['pooled_log_loss_delta']:+.9f}`
- Qualifier giriş P95: `{decision['reference_entry_p95']:.3f} -> {decision['qualifier_entry_p95']:.3f}`

Bu çalışma qualifier carry ve stage-K'yi birlikte izole eder. Same-season qualifier ranking karar vetosu değildir. Production aktivasyonu ayrı onay gerektirir.
"""


if __name__ == "__main__":
    main()
