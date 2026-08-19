from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
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
    CURRENT_K_REFERENCE,
    QualificationStageKConfig,
    all_configs,
    candidate_configs,
    effective_match_k,
    qualification_round_key,
    stage_k_multiplier,
)
from ao_elo.qualification_transition import apply_qualifier_carry  # noqa: E402
from ao_elo.tournament_bonus import (  # noqa: E402
    ELIGIBLE_PROGRESSION_STAGES,
    FixedTournamentBonusConfig,
    apply_tournament_progress_bonus,
)
from ao_elo.xg_live import (  # noqa: E402
    XGBlendConfig,
    XGPerformanceBonusConfig,
    update_match_elo_with_xg,
)
from scripts.run_controlled_goal_progression_backtest import (  # noqa: E402
    ControlledSeasonData,
    prepare_controlled_data,
)
from scripts.run_dynamic_core_calibration import expanding_folds  # noqa: E402
from scripts.run_final_robustness import (  # noqa: E402
    load_team_season_identity,
    summarize_ranking,
)
from scripts.run_stage_weighted_progression_backtest import (  # noqa: E402
    COMPETITION_INDEX,
    DOMESTIC_ADJUSTMENTS,
    DYNAMIC_MANIFEST,
    EVENTS_PATH,
    PRODUCTION_CONTRACT,
    STATIC_DATA_ROOT,
    XG_DATA,
    aggregate_ranking,
    load_domestic_adjustments,
    load_xg_map,
    probability_vector,
    same_season_ranking,
    validate_production_contract,
)
from scripts.run_v2_achievement_reserve_calibration import load_reserve_data  # noqa: E402
from scripts.run_v2_evaluation_upgrade import read_events  # noqa: E402


OUTPUT_ROOT = ROOT / "output" / "qualification_stage_k_backtest_2018_2026"
MODEL_NESTED = "NESTED_STAGE_K"
RANK_TOLERANCE = 1e-9


@dataclass
class StageKEvaluation:
    predictions: pd.DataFrame
    end_ratings: pd.DataFrame
    same_season_ranking: pd.DataFrame
    season_metrics: pd.DataFrame
    league_entry: pd.DataFrame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nested walk-forward test of qualification-stage K multipliers"
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
    contract_hash_before = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    core, parameters = validate_production_contract(contract)
    if not math.isclose(core.k_factor, 103.98098633392752, abs_tol=1e-12):
        raise ValueError("Production K_base differs from the frozen research contract")

    dynamic = json.loads(args.dynamic_manifest.resolve().read_text(encoding="utf-8"))
    static_config = AOEuropeanEloConfig(**dynamic["static_config"])
    static_config.validate()
    events = read_events(args.events.resolve())
    reserve, tie_audit = load_reserve_data(
        args.static_data_root.resolve(), args.events.resolve(), static_config
    )
    datasets = prepare_controlled_data(reserve, events)
    seasons = tuple(data.season for data in datasets)
    folds = expanding_folds(seasons)
    if len(folds) != 6:
        raise ValueError(f"Expected six outer folds, found {len(folds)}")
    if any("2026/27" in values for fold in folds for values in (*fold[0], fold[1])):
        raise ValueError("2026/27 must remain outside parameter selection")

    target = schedule_adjusted_team_performance(events)
    identity = load_team_season_identity()
    domestic = load_domestic_adjustments(args.domestic_adjustments.resolve(), datasets)
    xg = load_xg_map(args.xg_data.resolve(), datasets)
    event_rounds = events.set_index("match_id")["round"].astype(str).to_dict()
    configs = all_configs()

    evaluations: dict[str, StageKEvaluation] = {}
    for index, config in enumerate(configs, start=1):
        print(f"Evaluating {index}/{len(configs)}: {config.profile}", flush=True)
        evaluations[config.profile] = evaluate_config(
            datasets,
            target,
            core=core,
            parameters=parameters,
            domestic=domestic,
            xg=xg,
            event_rounds=event_rounds,
            config=config,
        )

    nested = build_nested_results(
        evaluations, configs, folds, target=target, identity=identity, seasons=seasons
    )
    evaluation_seasons = {test for _, test in folds}
    surface = build_candidate_surface(
        evaluations,
        configs,
        evaluation_seasons=evaluation_seasons,
        target=target,
        identity=identity,
        seasons=seasons,
    )
    competition = build_competition_summary(nested["predictions"])
    stage_summary = build_stage_summary(nested["predictions"])
    ranking_summary = build_ranking_summary(
        nested["same_season_ranking"], nested["end_ratings"], target, identity, seasons
    )
    entry_impact = build_league_entry_impact(nested["league_entry"], identity)
    path_summary = build_qualifier_path_summary(entry_impact)
    uncertainty = build_dependency_uncertainty(
        nested["predictions"], bootstrap_samples=args.bootstrap_samples
    )
    selected = decide_candidate(
        surface,
        nested["fold_results"],
        competition,
        ranking_summary,
        uncertainty,
        entry_impact,
    )
    contract_hash_after = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    safety = build_safety_audit(
        events=events,
        configs=configs,
        nested=nested,
        contract_hash_before=contract_hash_before,
        contract_hash_after=contract_hash_after,
    )
    if not safety["passed"].all():
        failed = safety.loc[~safety["passed"], "check"].tolist()
        raise ValueError(f"Stage-K safety audit failed: {failed}")

    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    surface.to_csv(output / "candidate_surface.csv", index=False)
    nested["fold_selections"].to_csv(output / "fold_selections.csv", index=False)
    nested["fold_results"].to_csv(output / "fold_results.csv", index=False)
    nested["predictions"].to_csv(output / "unseen_predictions.csv", index=False)
    stage_summary.to_csv(output / "qualification_stage_summary.csv", index=False)
    entry_impact.to_csv(output / "league_entry_rating_impact.csv", index=False)
    path_summary.to_csv(output / "qualifier_path_summary.csv", index=False)
    competition.to_csv(output / "competition_summary.csv", index=False)
    ranking_summary.to_csv(output / "ranking_summary.csv", index=False)
    uncertainty.to_csv(output / "dependency_uncertainty.csv", index=False)
    safety.to_csv(output / "safety_audit.csv", index=False)
    tie_audit.to_csv(output / "tie_chronology_audit.csv", index=False)
    (output / "selected_candidate.json").write_text(
        json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = build_report(
        contract=contract,
        seasons=seasons,
        events=events,
        surface=surface,
        selections=nested["fold_selections"],
        fold_results=nested["fold_results"],
        competition=competition,
        stage_summary=stage_summary,
        ranking=ranking_summary,
        entry_impact=entry_impact,
        path_summary=path_summary,
        uncertainty=uncertainty,
        safety=safety,
        selected=selected,
    )
    (output / "backtest_report.md").write_text(report, encoding="utf-8")
    print(f"Decision: {selected['decision']}")
    print(f"Full-history candidate: {selected['full_history_candidate']}")
    print(f"Report: {output / 'backtest_report.md'}")


def evaluate_config(
    datasets: tuple[ControlledSeasonData, ...],
    target: pd.DataFrame,
    *,
    core,
    parameters: dict[str, float | int],
    domestic: dict[tuple[str, int], float],
    xg: dict[str, tuple[float, float]],
    event_rounds: dict[str, str],
    config: QualificationStageKConfig,
    qualifier_carry: float = 1.0,
    model_name: str | None = None,
) -> StageKEvaluation:
    if not math.isfinite(qualifier_carry) or not 0.0 <= qualifier_carry <= 1.0:
        raise ValueError("qualifier_carry must be finite and in [0, 1]")
    label = config.profile if model_name is None else model_name
    predictions: list[dict[str, object]] = []
    end_rows: list[dict[str, object]] = []
    season_rows: list[dict[str, object]] = []
    entry_rows: list[dict[str, object]] = []
    xg_blend = XGBlendConfig(0.0, 1.0)
    xg_bonus = XGPerformanceBonusConfig(
        float(parameters["xg_ratio"]),
        float(parameters["xg_scale"]),
        float(parameters["xg_floor"]),
    )
    fixed_bonus = FixedTournamentBonusConfig(12.0)

    for data in datasets:
        reserve = data.reserve
        goal = reserve.goal
        season_core = goal.carry.core
        active_ids = season_core.active_team_ids
        power = season_core.initial_ratings.copy()
        for team_id in active_ids:
            power[int(team_id)] = domestic[(data.season, int(team_id))]
        initial = power.copy()
        bonus = np.zeros((len(power), len(COMPETITION_INDEX)), dtype=float)
        processed_ties: set[str] = set()
        tie_counts = pd.Series(reserve.tie_ids, dtype="object").dropna().value_counts()
        qualifier_matches = {int(team_id): 0 for team_id in active_ids}
        qualifier_wins = {int(team_id): 0 for team_id in active_ids}
        first_main_pre: dict[int, float] = {}
        pre_carry_rating: dict[int, float] = {}
        carry_adjustment: dict[int, float] = {}
        carry_applied: set[int] = set()
        main_participants: set[int] = set()
        max_zero_sum = 0.0
        max_delta = 0.0
        initial_total = float(power[active_ids].sum())

        for index, (home_raw, away_raw, neutral_raw, competition_raw) in enumerate(
            zip(
                season_core.home_team_ids,
                season_core.away_team_ids,
                season_core.neutral_flags,
                season_core.competitions,
                strict=True,
            )
        ):
            home_id, away_id = int(home_raw), int(away_raw)
            competition = str(competition_raw)
            match_id = str(season_core.match_ids[index])
            round_name = event_rounds[match_id]
            round_key = qualification_round_key(round_name)
            multiplier = stage_k_multiplier(round_name, config)
            match_k = effective_match_k(core.k_factor, round_name, config)
            stage = str(reserve.stages[index])
            tie_value = reserve.tie_ids[index]
            tie_id = None if tie_value is None else str(tie_value)
            if round_key == "MAIN":
                main_participants.update((home_id, away_id))
                for team_id in (home_id, away_id):
                    if qualifier_matches[team_id] > 0 and team_id not in carry_applied:
                        carry_result = apply_qualifier_carry(
                            float(initial[team_id]),
                            float(power[team_id]),
                            qualifier_carry,
                        )
                        pre_carry_rating[team_id] = carry_result.pre_carry_rating
                        carry_adjustment[team_id] = carry_result.carry_adjustment
                        power[team_id] = carry_result.post_carry_rating
                        carry_applied.add(team_id)
                home_live_pre = float(power[home_id] + bonus[home_id].sum())
                away_live_pre = float(power[away_id] + bonus[away_id].sum())
                first_main_pre.setdefault(home_id, home_live_pre)
                first_main_pre.setdefault(away_id, away_live_pre)
            else:
                home_live_pre = float(power[home_id] + bonus[home_id].sum())
                away_live_pre = float(power[away_id] + bonus[away_id].sum())
                qualifier_matches[home_id] += 1
                qualifier_matches[away_id] += 1

            xg_values = xg.get(match_id)
            penalty = bool(goal.penalty_flags[index])
            update = update_match_elo_with_xg(
                home_live_pre,
                away_live_pre,
                int(data.home_goals[index]),
                int(data.away_goals[index]),
                k_factor=match_k,
                elo_scale=core.elo_scale,
                home_advantage=core.home_advantage,
                is_neutral=bool(neutral_raw),
                decided_on_penalties=penalty,
                goal_difference_enabled=True,
                goal_alpha=float(parameters["goal_alpha"]),
                goal_tau=float(parameters["goal_tau"]),
                goal_difference_cap=int(parameters["goal_cap"]),
                xg_config=xg_blend,
                xg_home=None if xg_values is None else xg_values[0],
                xg_away=None if xg_values is None else xg_values[1],
                xg_performance_bonus_config=xg_bonus,
            )
            observed = (
                0 if update.actual_home_score == 1.0
                else 1 if update.actual_home_score == 0.5
                else 2
            )
            if round_key != "MAIN":
                if observed == 0:
                    qualifier_wins[home_id] += 1
                elif observed == 2:
                    qualifier_wins[away_id] += 1
            single_match = bool(
                tie_id is not None
                and int(tie_counts.get(tie_id, 0)) == 1
                and bool(reserve.tie_decider_flags[index])
            )
            draw_at_even = (
                float(parameters["single_match_draw_at_even"])
                if single_match else float(parameters["draw_at_even"])
            )
            probabilities = probability_vector(
                update.expected_home_score, draw_at_even, float(parameters["draw_shape"])
            )
            target_vector = np.eye(3, dtype=float)[observed]
            brier = float(np.square(probabilities - target_vector).sum())
            log_loss = -math.log(max(float(probabilities[observed]), 1e-15))
            power[home_id] += update.power_delta
            power[away_id] -= update.power_delta
            max_zero_sum = max(max_zero_sum, abs(update.zero_sum_error))
            max_delta = max(max_delta, abs(update.power_delta))

            progression_added = 0.0
            if (
                stage in ELIGIBLE_PROGRESSION_STAGES
                and bool(reserve.tie_decider_flags[index])
            ):
                if tie_id is None:
                    raise ValueError(f"{data.season}/{match_id}: progression tie lacks id")
                winner_id = int(reserve.advanced_team_ids[index])
                component = COMPETITION_INDEX[competition]
                progress = apply_tournament_progress_bonus(
                    float(bonus[winner_id, component]),
                    competition,
                    stage,
                    tie_id,
                    processed_ties,
                    fixed_bonus,
                )
                bonus[winner_id, component] = progress.bonus_post
                progression_added = progress.applied_bonus

            predictions.append(
                {
                    "model": label,
                    "config_id": config.config_id,
                    "selectable": config.selectable,
                    "match_id": match_id,
                    "season": data.season,
                    "kickoff_utc": pd.Timestamp(data.kickoff_utc[index]),
                    "competition": competition,
                    "round": round_name,
                    "qualification_round_key": round_key,
                    "stage": stage,
                    "tie_id": tie_id,
                    "home_team_id": home_id,
                    "away_team_id": away_id,
                    "home_live_pre": home_live_pre,
                    "away_live_pre": away_live_pre,
                    "home_goals": int(data.home_goals[index]),
                    "away_goals": int(data.away_goals[index]),
                    "actual_class": observed,
                    "predicted_class": int(np.argmax(probabilities)),
                    "expected_home_score": update.expected_home_score,
                    "home_probability": probabilities[0],
                    "draw_probability": probabilities[1],
                    "away_probability": probabilities[2],
                    "brier_1x2": brier,
                    "log_loss_1x2": log_loss,
                    "stage_k_multiplier": multiplier,
                    "effective_k": match_k,
                    "goal_multiplier": update.goal_difference_multiplier,
                    "xg_applied": update.xg_performance_signal is not None,
                    "xg_performance_signal": update.xg_performance_signal,
                    "power_delta": update.power_delta,
                    "zero_sum_error": update.zero_sum_error,
                    "progression_bonus_added": progression_added,
                }
            )

        end_power = power[active_ids]
        end_live = end_power + bonus[active_ids].sum(axis=1)
        conservation = abs(float(end_power.sum()) - initial_total)
        season_rows.append(
            {
                "model": label,
                "season": data.season,
                "matches": len(season_core.match_ids),
                "teams": len(active_ids),
                "maximum_abs_match_delta": max_delta,
                "maximum_match_zero_sum_error": max_zero_sum,
                "season_power_conservation_error": conservation,
                "rating_mean_start": float(initial[active_ids].mean()),
                "rating_mean_end_power": float(end_power.mean()),
                "rating_std_start": float(initial[active_ids].std()),
                "rating_std_end_live": float(end_live.std()),
            }
        )
        for position, team_raw in enumerate(active_ids):
            team_id = int(team_raw)
            end_rows.append(
                {
                    "model": label,
                    "season": data.season,
                    "team_id": team_id,
                    "initial_rating": float(initial[team_id]),
                    "end_power_rating": float(power[team_id]),
                    "end_live_rating": float(end_live[position]),
                    "reached_main": team_id in main_participants,
                }
            )
            if team_id in main_participants:
                entry_rows.append(
                    {
                        "model": label,
                        "season": data.season,
                        "team_id": team_id,
                        "qualifier_matches": qualifier_matches[team_id],
                        "qualifier_wins": qualifier_wins[team_id],
                        "entry_type": (
                            "QUALIFIER" if qualifier_matches[team_id] > 0 else "DIRECT"
                        ),
                        "initial_rating": float(initial[team_id]),
                        "pre_carry_rating": pre_carry_rating.get(
                            team_id, float(initial[team_id])
                        ),
                        "qualifier_carry": qualifier_carry,
                        "carry_adjustment": carry_adjustment.get(team_id, 0.0),
                        "league_entry_rating": first_main_pre[team_id],
                        "qualifier_rating_change": (
                            first_main_pre[team_id] - float(initial[team_id])
                        ),
                    }
                )

    end_frame = pd.DataFrame(end_rows)
    ranking = same_season_ranking(
        end_frame, target, {data.season for data in datasets}
    )
    ranking["model"] = label
    return StageKEvaluation(
        predictions=pd.DataFrame(predictions),
        end_ratings=end_frame,
        same_season_ranking=ranking,
        season_metrics=pd.DataFrame(season_rows),
        league_entry=pd.DataFrame(entry_rows),
    )


def metrics_for_seasons(
    evaluation: StageKEvaluation,
    seasons: set[str],
    *,
    target: pd.DataFrame,
    identity: pd.DataFrame,
) -> dict[str, float | int]:
    predictions = evaluation.predictions.loc[evaluation.predictions["season"].isin(seasons)]
    rank = aggregate_ranking(
        evaluation.same_season_ranking.loc[
            evaluation.same_season_ranking["season"].isin(seasons)
        ]
    )
    all_rank = rank.loc[rank["competition"].eq("ALL")].iloc[0]
    source_end = evaluation.end_ratings.loc[evaluation.end_ratings["season"].isin(seasons)]
    forward = summarize_ranking(
        source_end,
        target,
        allowed_target_seasons=set(seasons),
        identity=identity,
    )
    forward_all = forward.loc[forward["competition"].eq("ALL")]
    invariants = evaluation.season_metrics.loc[
        evaluation.season_metrics["season"].isin(seasons)
    ]
    return {
        "matches": len(predictions),
        "brier_1x2": float(predictions["brier_1x2"].mean()),
        "log_loss_1x2": float(predictions["log_loss_1x2"].mean()),
        "accuracy_1x2": float(
            (predictions["actual_class"] == predictions["predicted_class"]).mean()
        ),
        "same_season_spearman": float(all_rank["ranking_score"]),
        "same_season_pairwise_accuracy": float(all_rank["pairwise_accuracy"]),
        "forward_spearman": (
            float(forward_all.iloc[0]["ranking_score"]) if len(forward_all) else np.nan
        ),
        "forward_pairwise_accuracy": (
            float(forward_all.iloc[0]["pairwise_accuracy"]) if len(forward_all) else np.nan
        ),
        "maximum_abs_match_delta": float(invariants["maximum_abs_match_delta"].max()),
        "maximum_zero_sum_error": float(
            invariants["maximum_match_zero_sum_error"].max()
        ),
        "maximum_conservation_error": float(
            invariants["season_power_conservation_error"].max()
        ),
    }


def select_training_config(
    evaluations: dict[str, StageKEvaluation],
    configs: tuple[QualificationStageKConfig, ...],
    train_seasons: tuple[str, ...],
    *,
    target: pd.DataFrame,
    identity: pd.DataFrame,
) -> tuple[QualificationStageKConfig, pd.DataFrame]:
    seasons = set(train_seasons)
    rows = []
    for config in configs:
        metrics = metrics_for_seasons(
            evaluations[config.profile], seasons, target=target, identity=identity
        )
        rows.append(
            {
                "candidate": config.profile,
                "selectable": config.selectable,
                **metrics,
            }
        )
    frame = pd.DataFrame(rows)
    baseline = frame.loc[frame["candidate"].eq(CURRENT_K_REFERENCE)].iloc[0]
    candidates = frame.loc[frame["selectable"]].copy()
    candidates["train_rank_safe"] = (
        candidates["same_season_spearman"].ge(
            float(baseline["same_season_spearman"]) - RANK_TOLERANCE
        )
        & candidates["same_season_pairwise_accuracy"].ge(
            float(baseline["same_season_pairwise_accuracy"]) - RANK_TOLERANCE
        )
        & (
            candidates["forward_spearman"].isna()
            | candidates["forward_spearman"].ge(
                float(baseline["forward_spearman"]) - RANK_TOLERANCE
            )
        )
        & (
            candidates["forward_pairwise_accuracy"].isna()
            | candidates["forward_pairwise_accuracy"].ge(
                float(baseline["forward_pairwise_accuracy"]) - RANK_TOLERANCE
            )
        )
    )
    pool = candidates.loc[candidates["train_rank_safe"]]
    if pool.empty:
        pool = candidates
    selected_row = pool.sort_values(
        [
            "forward_pairwise_accuracy",
            "forward_spearman",
            "same_season_pairwise_accuracy",
            "same_season_spearman",
            "brier_1x2",
            "log_loss_1x2",
            "candidate",
        ],
        ascending=[False, False, False, False, True, True, True],
        na_position="last",
        kind="stable",
    ).iloc[0]
    by_name = {config.profile: config for config in configs}
    return by_name[str(selected_row["candidate"])], candidates


def build_nested_results(
    evaluations: dict[str, StageKEvaluation],
    configs: tuple[QualificationStageKConfig, ...],
    folds,
    *,
    target: pd.DataFrame,
    identity: pd.DataFrame,
    seasons: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    selection_rows = []
    fold_rows = []
    prediction_frames = []
    end_frames = []
    ranking_frames = []
    entry_frames = []
    baseline = evaluations[CURRENT_K_REFERENCE]
    for fold, (train_seasons, test_season) in enumerate(folds, start=1):
        selected, train_surface = select_training_config(
            evaluations,
            configs,
            train_seasons,
            target=target,
            identity=identity,
        )
        selected_train = train_surface.loc[
            train_surface["candidate"].eq(selected.profile)
        ].iloc[0]
        selection_rows.append(
            {
                "fold": fold,
                "train_seasons": "|".join(train_seasons),
                "test_season": test_season,
                "selected_candidate": selected.profile,
                "selected_config_id": selected.config_id,
                "train_rank_safe": bool(selected_train["train_rank_safe"]),
                "q1_multiplier": selected.q1_multiplier,
                "q2_multiplier": selected.q2_multiplier,
                "q3_multiplier": selected.q3_multiplier,
                "qualifying_playoff_multiplier": selected.qualifying_playoff_multiplier,
            }
        )
        for model, source in (
            (CURRENT_K_REFERENCE, baseline),
            (MODEL_NESTED, evaluations[selected.profile]),
        ):
            metrics = metrics_for_seasons(
                source, {test_season}, target=target, identity=identity
            )
            fold_rows.append(
                {
                    "fold": fold,
                    "test_season": test_season,
                    "model": model,
                    "candidate": (
                        CURRENT_K_REFERENCE if model == CURRENT_K_REFERENCE else selected.profile
                    ),
                    **metrics,
                }
            )
            for collection, frames in (
                (source.predictions, prediction_frames),
                (source.end_ratings, end_frames),
                (source.same_season_ranking, ranking_frames),
                (source.league_entry, entry_frames),
            ):
                part = collection.loc[collection["season"].eq(test_season)].copy()
                part["fold"] = fold
                part["model"] = model
                part["candidate"] = (
                    CURRENT_K_REFERENCE if model == CURRENT_K_REFERENCE else selected.profile
                )
                frames.append(part)
        print(f"  fold {fold}/6 {test_season}: {selected.profile}", flush=True)
    fold_results = pd.DataFrame(fold_rows)
    all_end = pd.concat(end_frames, ignore_index=True)
    next_season = {
        seasons[index]: seasons[index + 1] for index in range(len(seasons) - 1)
    }
    for row_index, row in fold_results.iterrows():
        source_season = str(row["test_season"])
        destination = next_season.get(source_season)
        if destination is None:
            continue
        source = all_end.loc[
            all_end["fold"].eq(int(row["fold"]))
            & all_end["model"].eq(str(row["model"]))
            & all_end["season"].eq(source_season)
        ]
        forward = summarize_ranking(
            source,
            target,
            allowed_target_seasons={destination},
            identity=identity,
        )
        all_forward = forward.loc[forward["competition"].eq("ALL")]
        if len(all_forward):
            fold_results.loc[row_index, "forward_spearman"] = float(
                all_forward.iloc[0]["ranking_score"]
            )
            fold_results.loc[row_index, "forward_pairwise_accuracy"] = float(
                all_forward.iloc[0]["pairwise_accuracy"]
            )
    reference = fold_results.loc[
        fold_results["model"].eq(CURRENT_K_REFERENCE)
    ].set_index("fold")
    for metric in (
        "brier_1x2",
        "log_loss_1x2",
        "accuracy_1x2",
        "same_season_spearman",
        "same_season_pairwise_accuracy",
        "forward_spearman",
        "forward_pairwise_accuracy",
    ):
        fold_results[f"delta_vs_reference_{metric}"] = fold_results.apply(
            lambda row: row[metric] - reference.loc[row["fold"], metric], axis=1
        )
    return {
        "fold_selections": pd.DataFrame(selection_rows),
        "fold_results": fold_results,
        "predictions": pd.concat(prediction_frames, ignore_index=True),
        "end_ratings": all_end,
        "same_season_ranking": pd.concat(ranking_frames, ignore_index=True),
        "league_entry": pd.concat(entry_frames, ignore_index=True),
    }


def build_candidate_surface(
    evaluations: dict[str, StageKEvaluation],
    configs: tuple[QualificationStageKConfig, ...],
    *,
    evaluation_seasons: set[str],
    target: pd.DataFrame,
    identity: pd.DataFrame,
    seasons: tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    for config in configs:
        metrics = metrics_for_seasons(
            evaluations[config.profile],
            evaluation_seasons,
            target=target,
            identity=identity,
        )
        entry = evaluations[config.profile].league_entry
        qualified = entry.loc[
            entry["season"].isin(evaluation_seasons) & entry["entry_type"].eq("QUALIFIER")
        ]
        rows.append(
            {
                "candidate": config.profile,
                "config_id": config.config_id,
                "selectable": config.selectable,
                "q1_multiplier": config.q1_multiplier,
                "q2_multiplier": config.q2_multiplier,
                "q3_multiplier": config.q3_multiplier,
                "qualifying_playoff_multiplier": config.qualifying_playoff_multiplier,
                "main_multiplier": config.main_multiplier,
                **metrics,
                "qualifier_entry_mean_abs_change": float(
                    qualified["qualifier_rating_change"].abs().mean()
                ),
                "qualifier_entry_p95_abs_change": float(
                    qualified["qualifier_rating_change"].abs().quantile(0.95)
                ),
                "qualifier_entry_max_abs_change": float(
                    qualified["qualifier_rating_change"].abs().max()
                ),
            }
        )
    result = pd.DataFrame(rows)
    baseline = result.loc[result["candidate"].eq(CURRENT_K_REFERENCE)].iloc[0]
    for metric in (
        "brier_1x2",
        "log_loss_1x2",
        "accuracy_1x2",
        "same_season_spearman",
        "same_season_pairwise_accuracy",
        "forward_spearman",
        "forward_pairwise_accuracy",
        "qualifier_entry_mean_abs_change",
        "qualifier_entry_p95_abs_change",
        "qualifier_entry_max_abs_change",
    ):
        result[f"delta_vs_reference_{metric}"] = result[metric] - float(baseline[metric])
    return result


def prediction_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    return {
        "matches": len(frame),
        "brier_1x2": float(frame["brier_1x2"].mean()),
        "log_loss_1x2": float(frame["log_loss_1x2"].mean()),
        "accuracy_1x2": float(
            (frame["actual_class"] == frame["predicted_class"]).mean()
        ),
    }


def add_reference_deltas(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    result = frame.copy()
    reference = result.loc[result["model"].eq(CURRENT_K_REFERENCE)].set_index(keys)
    for metric in ("brier_1x2", "log_loss_1x2", "accuracy_1x2"):
        result[f"delta_vs_reference_{metric}"] = result.apply(
            lambda row: row[metric] - reference.loc[tuple(row[key] for key in keys), metric]
            if len(keys) > 1
            else row[metric] - reference.loc[row[keys[0]], metric],
            axis=1,
        )
    return result


def build_competition_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, competition), frame in predictions.groupby(
        ["model", "competition"], sort=True
    ):
        rows.append({"model": model, "competition": competition, **prediction_metrics(frame)})
    return add_reference_deltas(pd.DataFrame(rows), ["competition"])


def build_stage_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, round_key), frame in predictions.groupby(
        ["model", "qualification_round_key"], sort=True
    ):
        rows.append(
            {
                "model": model,
                "qualification_round_key": round_key,
                **prediction_metrics(frame),
                "mean_effective_k": float(frame["effective_k"].mean()),
                "mean_abs_power_delta": float(frame["power_delta"].abs().mean()),
                "p95_abs_power_delta": float(frame["power_delta"].abs().quantile(0.95)),
                "maximum_abs_power_delta": float(frame["power_delta"].abs().max()),
            }
        )
    return add_reference_deltas(pd.DataFrame(rows), ["qualification_round_key"])


def build_ranking_summary(
    same: pd.DataFrame,
    end_ratings: pd.DataFrame,
    target: pd.DataFrame,
    identity: pd.DataFrame,
    seasons: tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    for model, frame in same.groupby("model", sort=False):
        summary = aggregate_ranking(frame)
        for rank_row in summary.itertuples(index=False):
            rows.append(
                {
                    "model": model,
                    "scope": "SAME_SEASON_POOLED",
                    "source_season": "ALL",
                    "target_season": "ALL",
                    "competition": rank_row.competition,
                    "spearman": float(rank_row.ranking_score),
                    "pairwise_accuracy": float(rank_row.pairwise_accuracy),
                }
            )
    for model, frame in end_ratings.groupby("model", sort=False):
        for source_index in range(2, len(seasons) - 1):
            source = seasons[source_index]
            destination = seasons[source_index + 1]
            ranking = summarize_ranking(
                frame.loc[frame["season"].eq(source)],
                target,
                allowed_target_seasons={destination},
                identity=identity,
            )
            for rank_row in ranking.itertuples(index=False):
                rows.append(
                    {
                        "model": model,
                        "scope": "FORWARD_SEASON",
                        "source_season": source,
                        "target_season": destination,
                        "competition": rank_row.competition,
                        "spearman": float(rank_row.ranking_score),
                        "pairwise_accuracy": float(rank_row.pairwise_accuracy),
                    }
                )
    result = pd.DataFrame(rows)
    reference = result.loc[result["model"].eq(CURRENT_K_REFERENCE)].set_index(
        ["scope", "source_season", "target_season", "competition"]
    )
    result["spearman_delta_vs_reference"] = result.apply(
        lambda row: row["spearman"]
        - reference.loc[
            (row["scope"], row["source_season"], row["target_season"], row["competition"]),
            "spearman",
        ],
        axis=1,
    )
    result["pairwise_delta_vs_reference"] = result.apply(
        lambda row: row["pairwise_accuracy"]
        - reference.loc[
            (row["scope"], row["source_season"], row["target_season"], row["competition"]),
            "pairwise_accuracy",
        ],
        axis=1,
    )
    return result


def build_league_entry_impact(
    entry: pd.DataFrame,
    identity: pd.DataFrame,
    *,
    reference_model: str = CURRENT_K_REFERENCE,
) -> pd.DataFrame:
    result = entry.copy()
    result["entry_rank"] = result.groupby(["model", "season"])[
        "league_entry_rating"
    ].rank(method="min", ascending=False)
    baseline = result.loc[result["model"].eq(reference_model)][
        ["season", "team_id", "league_entry_rating", "entry_rank"]
    ].rename(
        columns={
            "league_entry_rating": "reference_league_entry_rating",
            "entry_rank": "reference_entry_rank",
        }
    )
    result = result.merge(baseline, on=["season", "team_id"], validate="many_to_one")
    names = identity[
        ["season", "local_team_id", "team_name", "country_code", "club_id"]
    ].rename(columns={"local_team_id": "team_id"})
    result = result.merge(
        names,
        on=["season", "team_id"],
        how="left",
        validate="many_to_one",
    )
    result["league_entry_rating_delta_vs_reference"] = (
        result["league_entry_rating"] - result["reference_league_entry_rating"]
    )
    result["entry_rank_change_vs_reference"] = (
        result["reference_entry_rank"] - result["entry_rank"]
    )
    result["entered_top_20_vs_reference"] = (
        result["entry_rank"].le(20) & result["reference_entry_rank"].gt(20)
    )
    result["left_top_20_vs_reference"] = (
        result["entry_rank"].gt(20) & result["reference_entry_rank"].le(20)
    )
    result["entered_top_50_vs_reference"] = (
        result["entry_rank"].le(50) & result["reference_entry_rank"].gt(50)
    )
    result["left_top_50_vs_reference"] = (
        result["entry_rank"].gt(50) & result["reference_entry_rank"].le(50)
    )
    return result.sort_values(["fold", "model", "season", "entry_rank", "team_id"])


def build_qualifier_path_summary(entry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, entry_type, matches), frame in entry.groupby(
        ["model", "entry_type", "qualifier_matches"], sort=True
    ):
        absolute = frame["qualifier_rating_change"].abs()
        rows.append(
            {
                "model": model,
                "entry_type": entry_type,
                "qualifier_matches": int(matches),
                "teams": len(frame),
                "mean_wins": float(frame["qualifier_wins"].mean()),
                "mean_rating_change": float(frame["qualifier_rating_change"].mean()),
                "mean_abs_rating_change": float(absolute.mean()),
                "median_abs_rating_change": float(absolute.median()),
                "p90_abs_rating_change": float(absolute.quantile(0.90)),
                "p95_abs_rating_change": float(absolute.quantile(0.95)),
                "maximum_abs_rating_change": float(absolute.max()),
                "mean_abs_entry_rank_change_vs_reference": float(
                    frame["entry_rank_change_vs_reference"].abs().mean()
                ),
                "maximum_entry_rank_gain_vs_reference": float(
                    frame["entry_rank_change_vs_reference"].max()
                ),
                "maximum_entry_rank_loss_vs_reference": float(
                    frame["entry_rank_change_vs_reference"].min()
                ),
                "entered_top_20_vs_reference": int(
                    frame["entered_top_20_vs_reference"].sum()
                ),
                "entered_top_50_vs_reference": int(
                    frame["entered_top_50_vs_reference"].sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_dependency_uncertainty(
    predictions: pd.DataFrame, *, bootstrap_samples: int
) -> pd.DataFrame:
    candidate = predictions.loc[predictions["model"].eq(MODEL_NESTED)]
    baseline = predictions.loc[predictions["model"].eq(CURRENT_K_REFERENCE)]
    rows = []
    for competition in ("ALL", "UCL", "UEL", "UECL"):
        left = candidate if competition == "ALL" else candidate.loc[
            candidate["competition"].eq(competition)
        ]
        right = baseline if competition == "ALL" else baseline.loc[
            baseline["competition"].eq(competition)
        ]
        paired = left.merge(
            right[["match_id", "brier_1x2", "log_loss_1x2"]],
            on="match_id",
            suffixes=("_candidate", "_baseline"),
            validate="one_to_one",
        )
        for metric in ("brier_1x2", "log_loss_1x2"):
            sample = paired[
                ["season", "match_id", "home_team_id", "away_team_id", "kickoff_utc", "tie_id"]
            ].copy()
            sample["loss_difference"] = (
                paired[f"{metric}_candidate"] - paired[f"{metric}_baseline"]
            )
            result = dependency_robust_loss_difference_ci(
                sample, bootstrap_samples=bootstrap_samples
            )
            result.insert(0, "competition", competition)
            result.insert(1, "metric", metric)
            rows.append(result)
    return pd.concat(rows, ignore_index=True)


def decide_candidate(
    surface: pd.DataFrame,
    fold_results: pd.DataFrame,
    competition: pd.DataFrame,
    ranking: pd.DataFrame,
    uncertainty: pd.DataFrame,
    entry: pd.DataFrame,
) -> dict[str, object]:
    selectable = surface.loc[surface["selectable"]].copy()
    baseline = surface.loc[surface["candidate"].eq(CURRENT_K_REFERENCE)].iloc[0]
    selectable["rank_safe"] = (
        selectable["forward_spearman"].ge(float(baseline["forward_spearman"]) - RANK_TOLERANCE)
        & selectable["forward_pairwise_accuracy"].ge(
            float(baseline["forward_pairwise_accuracy"]) - RANK_TOLERANCE
        )
    )
    pool = selectable.loc[selectable["rank_safe"]]
    if pool.empty:
        pool = selectable
    best = pool.sort_values(
        [
            "forward_pairwise_accuracy",
            "forward_spearman",
            "same_season_pairwise_accuracy",
            "same_season_spearman",
            "brier_1x2",
            "log_loss_1x2",
            "candidate",
        ],
        ascending=[False, False, False, False, True, True, True],
        kind="stable",
    ).iloc[0]
    nested = fold_results.loc[fold_results["model"].eq(MODEL_NESTED)]
    fold_rank_wins = int(
        (
            nested["delta_vs_reference_same_season_spearman"].gt(RANK_TOLERANCE)
            | nested["delta_vs_reference_same_season_pairwise_accuracy"].gt(RANK_TOLERANCE)
            | nested["delta_vs_reference_forward_spearman"].gt(RANK_TOLERANCE)
            | nested["delta_vs_reference_forward_pairwise_accuracy"].gt(RANK_TOLERANCE)
        ).sum()
    )
    pooled_predictions_safe = bool(
        nested["brier_1x2"].mul(nested["matches"]).sum() / nested["matches"].sum()
        <= fold_results.loc[fold_results["model"].eq(CURRENT_K_REFERENCE), "brier_1x2"].mul(
            fold_results.loc[fold_results["model"].eq(CURRENT_K_REFERENCE), "matches"]
        ).sum()
        / fold_results.loc[fold_results["model"].eq(CURRENT_K_REFERENCE), "matches"].sum()
        + 1e-12
        and nested["log_loss_1x2"].mul(nested["matches"]).sum() / nested["matches"].sum()
        <= fold_results.loc[fold_results["model"].eq(CURRENT_K_REFERENCE), "log_loss_1x2"].mul(
            fold_results.loc[fold_results["model"].eq(CURRENT_K_REFERENCE), "matches"]
        ).sum()
        / fold_results.loc[fold_results["model"].eq(CURRENT_K_REFERENCE), "matches"].sum()
        + 1e-12
    )
    reliable_harm = bool((uncertainty["ci_95_lower"] > 0.0).any())
    competition_harm = bool(
        uncertainty.loc[
            uncertainty["competition"].ne("ALL"), "reliable_harm"
        ].astype(bool).any()
    )
    forward = ranking.loc[
        ranking["model"].eq(MODEL_NESTED)
        & ranking["scope"].eq("FORWARD_SEASON")
        & ranking["competition"].eq("ALL")
    ]
    forward_safe = bool(
        (forward["spearman_delta_vs_reference"] >= -RANK_TOLERANCE).all()
        and (forward["pairwise_delta_vs_reference"] >= -RANK_TOLERANCE).all()
    )
    qualifier = entry.loc[
        entry["model"].eq(MODEL_NESTED) & entry["entry_type"].eq("QUALIFIER")
    ]
    entry_reduced = bool(
        qualifier["league_entry_rating_delta_vs_reference"].abs().mean() > 1e-9
    )
    passed = bool(
        fold_rank_wins >= 4
        and pooled_predictions_safe
        and not reliable_harm
        and not competition_harm
        and forward_safe
        and entry_reduced
    )
    return {
        "decision": "BEST_RESEARCH_CANDIDATE" if passed else "KEEP_CURRENT_K",
        "production_changed": False,
        "reference_was_selectable": False,
        "full_history_candidate": str(best["candidate"]),
        "full_history_config_id": str(best["config_id"]),
        "nested_selected_profiles": nested["candidate"].value_counts().to_dict(),
        "ranking_improvement_folds": fold_rank_wins,
        "pooled_brier_and_log_loss_non_worse": pooled_predictions_safe,
        "competition_reliable_harm": competition_harm,
        "dependency_reliable_harm": reliable_harm,
        "all_forward_transitions_non_worse": forward_safe,
        "qualifier_entry_movement_changed": entry_reduced,
        "caveat": "Development-window replay; not an untouched prospective holdout.",
    }


def build_safety_audit(
    *,
    events: pd.DataFrame,
    configs: tuple[QualificationStageKConfig, ...],
    nested: dict[str, pd.DataFrame],
    contract_hash_before: str,
    contract_hash_after: str,
) -> pd.DataFrame:
    predictions = nested["predictions"]
    qualifier_count = int(
        events["round"].isin(
            {
                "Preliminary Round",
                "1st Qualifying Round",
                "2nd Qualifying Round",
                "3rd Qualifying Round",
                "Qualifying Play-off Round",
            }
        ).sum()
    )
    candidate_list = candidate_configs()
    checks = [
        ("production_contract_unchanged", contract_hash_before == contract_hash_after, contract_hash_after, contract_hash_before),
        ("all_match_ids_unique", not events["match_id"].duplicated().any(), int(events["match_id"].duplicated().sum()), 0),
        ("qualifier_match_count", qualifier_count == 3083, qualifier_count, 3083),
        ("reference_not_selectable", not configs[0].selectable, configs[0].selectable, False),
        ("seven_selectable_candidates", len(candidate_list) == 7, len(candidate_list), 7),
        ("main_matches_full_k", np.allclose(predictions.loc[predictions["qualification_round_key"].eq("MAIN"), "stage_k_multiplier"], 1.0), float(predictions.loc[predictions["qualification_round_key"].eq("MAIN"), "stage_k_multiplier"].min()), 1.0),
        ("match_power_zero_sum", float(predictions["zero_sum_error"].max()) <= 1e-9, float(predictions["zero_sum_error"].max()), 1e-9),
        ("probabilities_normalized", np.allclose(predictions[["home_probability", "draw_probability", "away_probability"]].sum(axis=1), 1.0), float((predictions[["home_probability", "draw_probability", "away_probability"]].sum(axis=1) - 1.0).abs().max()), 1e-12),
        ("no_2026_27", not predictions["season"].eq("2026/27").any(), int(predictions["season"].eq("2026/27").sum()), 0),
        ("fold_reference_and_candidate_once", not predictions.duplicated(["fold", "model", "match_id"]).any(), int(predictions.duplicated(["fold", "model", "match_id"]).sum()), 0),
    ]
    return pd.DataFrame(
        [
            {"check": check, "passed": bool(passed), "observed": observed, "requirement": requirement}
            for check, passed, observed, requirement in checks
        ]
    )


def md_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    view = frame if columns is None else frame[columns]
    if view.empty:
        return "_Veri yok._"

    def render(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    headers = [str(column) for column in view.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in view.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def build_report(
    *,
    contract: dict,
    seasons: tuple[str, ...],
    events: pd.DataFrame,
    surface: pd.DataFrame,
    selections: pd.DataFrame,
    fold_results: pd.DataFrame,
    competition: pd.DataFrame,
    stage_summary: pd.DataFrame,
    ranking: pd.DataFrame,
    entry_impact: pd.DataFrame,
    path_summary: pd.DataFrame,
    uncertainty: pd.DataFrame,
    safety: pd.DataFrame,
    selected: dict[str, object],
) -> str:
    qualifier = entry_impact.loc[
        entry_impact["model"].eq(MODEL_NESTED) & entry_impact["entry_type"].eq("QUALIFIER")
    ]
    largest = qualifier.reindex(
        qualifier["league_entry_rating_delta_vs_reference"].abs().sort_values(ascending=False).index
    ).head(15)
    covid = fold_results.loc[fold_results["test_season"].eq("2020/21")]
    return f"""# Ön Eleme Aşamasına Göre K Çarpanı Backtesti

## Karar

- Karar: **{selected['decision']}**
- Full-history araştırma adayı: **{selected['full_history_candidate']}**
- Production değişti: **Hayır**
- Bütün turları `1.00` kullanan kol yalnız referanstır ve aday seçiminde kullanılmamıştır.

## Veri ve Aktif Çekirdek

- Sezonlar: `{seasons[0]}–{seasons[-1]}`
- Toplam maç: `{len(events)}`
- Ön eleme maçı: `{int(events['round'].isin(set(['Preliminary Round','1st Qualifying Round','2nd Qualifying Round','3rd Qualifying Round','Qualifying Play-off Round'])).sum())}`
- Outer fold: `6`; test sezonları `2020/21–2025/26`
- K base: `{contract['dynamic_core']['k_factor']:.6f}`
- Goal margin: `alpha={contract['goal_margin']['alpha']}`, `tau={contract['goal_margin']['tau']}`, cap `{contract['goal_margin']['goal_difference_cap']}`
- xG: ratio `{contract['xg_performance']['max_xg_ratio']}`, scale `{contract['xg_performance']['xg_scale']}`; eksikte GD fallback
- Ana turnuva maçları her adayda `K multiplier=1.00` kullanır.

## Sabit Aday Yüzeyi

{md_table(surface, ['candidate','selectable','q1_multiplier','q2_multiplier','q3_multiplier','qualifying_playoff_multiplier','brier_1x2','log_loss_1x2','same_season_spearman','same_season_pairwise_accuracy','forward_spearman','forward_pairwise_accuracy','qualifier_entry_p95_abs_change'])}

## Nested Fold Seçimleri

{md_table(selections)}

## Unseen Fold Sonuçları

{md_table(fold_results, ['fold','test_season','model','candidate','matches','brier_1x2','log_loss_1x2','accuracy_1x2','same_season_spearman','same_season_pairwise_accuracy','delta_vs_reference_brier_1x2','delta_vs_reference_log_loss_1x2'])}

## 2020/21 COVID Diagnostiği

Tek maçlı ön elemelerin yoğun olduğu sezon seçimden çıkarılmamış, ayrıca aşağıda görünür tutulmuştur.

{md_table(covid, ['model','candidate','matches','brier_1x2','log_loss_1x2','same_season_spearman','same_season_pairwise_accuracy'])}

## Turnuva Segmentleri

{md_table(competition)}

## Aşama Segmentleri

{md_table(stage_summary)}

## Ranking

{md_table(ranking)}

## Lig Aşamasına Giriş Etkisi

Nested adayın referansa göre en büyük giriş Elo farkları:

{md_table(largest, ['season','team_name','country_code','club_id','qualifier_matches','qualifier_wins','initial_rating','league_entry_rating','reference_league_entry_rating','league_entry_rating_delta_vs_reference','entry_rank_change_vs_reference'])}

Altı ve sekiz maç dahil qualifier yol uzunluğu özeti:

{md_table(path_summary)}

## Bağımlılığa Dayanıklı Belirsizlik

Negatif loss farkı aday lehinedir.

{md_table(uncertainty)}

## Güvenlik Denetimi

{md_table(safety)}

## Yorum

Bu çalışma yalnız qualifier maçlarındaki öğrenme hızını değiştirir. AO First Elo, beklenen skor, home advantage, Scale, progression, gol farkı ve xG sözleşmeleri değişmemiştir. Qualifier carry uygulanmamıştır. `{selected['decision']}` kararı geliştirme penceresindeki kanıta dayanır; `2026/27` prospective monitoring verisi değildir.
"""


if __name__ == "__main__":
    main()
