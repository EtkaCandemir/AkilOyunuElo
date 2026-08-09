from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ao_elo.config import AOEuropeanEloConfig  # noqa: E402
from ao_elo.domestic_surprise_symmetric import (  # noqa: E402
    SymmetricDomesticSurpriseConfig,
    calculate_symmetric_domestic_surprise_adjustment,
    normalized_finish_score,
)
from ao_elo.evaluation import schedule_adjusted_team_performance  # noqa: E402
from ao_elo.pipeline import compute_ao_first_elo_from_csv  # noqa: E402
from scripts.run_domestic_surprise_backtest import (  # noqa: E402
    Evaluation,
    LiveBacktestConfig,
    evaluate_live,
    fold_core,
    initial_ranking,
    json_default,
    live_config_from_contract,
    load_standings_cache,
    markdown_table,
    pair_predictions,
    resolve_identity_position,
    summarize_competitions,
)
from scripts.run_dynamic_core_calibration import (  # noqa: E402
    DynamicCoreConfig,
    expanding_folds,
)
from scripts.run_match_context_backtest import (  # noqa: E402
    comparison_uncertainty,
    read_context_events,
)
from scripts.run_v2_achievement_reserve_calibration import (  # noqa: E402
    ReserveSeasonData,
    load_reserve_data,
)
from scripts.build_backtest_stage_b import normalize_country  # noqa: E402


STATIC_ROOT = ROOT / "data" / "backtest_stage_b_2018_2026"
EVENTS_PATH = ROOT / "data" / "external_elo_benchmark_2018_2026" / "exact_date_events.csv"
DYNAMIC_ROOT = ROOT / "output" / "v2_dynamic_calibration_2018_2026"
FINAL_CONTRACT = ROOT / "contracts" / "ao_european_elo_v2_final_candidate.json"
CLUB_IDENTITY_PATH = ROOT / "data" / "club_identity" / "team_season_identity.csv"
OUTPUT_ROOT = ROOT / "output" / "domestic_surprise_5y_backtest_2018_2026"
COEFFICIENTS = (0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40)
ADJUSTMENT_CAPS = (15.0, 25.0, 50.0)
EXPOSURE_POWERS = (0.5, 1.0, 1.5)
NORMALIZATIONS = ("legacy_finish_curve", "direct_percentile")
MINIMUM_HISTORY_SEASONS = 5
RANK_TOLERANCE = 1e-9


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest symmetric five-season domestic-finish surprise"
    )
    parser.add_argument("--static-root", type=Path, default=STATIC_ROOT)
    parser.add_argument("--events-path", type=Path, default=EVENTS_PATH)
    parser.add_argument("--dynamic-root", type=Path, default=DYNAMIC_ROOT)
    parser.add_argument("--final-contract", type=Path, default=FINAL_CONTRACT)
    parser.add_argument("--club-identity", type=Path, default=CLUB_IDENTITY_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()

    static_root = args.static_root.resolve()
    events = read_context_events(args.events_path.resolve())
    dynamic_root = args.dynamic_root.resolve()
    dynamic_manifest = json.loads(
        (dynamic_root / "selected_dynamic_model.json").read_text(encoding="utf-8")
    )
    static_config = AOEuropeanEloConfig(**dynamic_manifest["static_config"])
    static_config.validate()
    final_contract = json.loads(
        args.final_contract.resolve().read_text(encoding="utf-8")
    )
    live = live_config_from_contract(final_contract)
    datasets, _ = load_reserve_data(static_root, args.events_path.resolve(), static_config)
    seasons = tuple(data.season for data in datasets)
    folds = expanding_folds(seasons)
    if len(folds) != 6:
        raise ValueError(f"Expected six outer folds, found {len(folds)}")
    core_selections = pd.read_csv(dynamic_root / "core_fold_selections.csv")
    target = schedule_adjusted_team_performance(events)

    print("Five-season history: parsing 2013-2025 full league tables", flush=True)
    features, history_long, coverage = build_domestic_history_features(
        static_root,
        static_config,
        seasons,
        args.club_identity.resolve(),
    )
    candidates = candidate_grid()
    print(
        f"Symmetric surprise: {len(candidates)} candidates, {len(folds)} outer folds",
        flush=True,
    )
    adjustments = {
        config: build_adjustments(features, config, static_config)
        for config in candidates
    }
    selections, fold_results, predictions, fold_adjustments = run_walk_forward_backtest(
        datasets,
        adjustments,
        events,
        target,
        folds,
        core_selections,
        candidates,
        live,
    )
    competition = summarize_competitions(predictions)
    uncertainty = comparison_uncertainty(predictions, args.bootstrap_samples)
    full_core = DynamicCoreConfig(**final_contract["dynamic_core"])
    final_candidate, full_metrics = select_candidate_from_adjustments(
        datasets,
        adjustments,
        target,
        set(seasons),
        full_core,
        candidates,
        live,
    )
    final_adjustments = adjustments[final_candidate]
    normalization_comparison = summarize_normalizations(full_metrics)
    decision, guardrails = promotion_decision(
        selections,
        fold_results,
        competition,
        uncertainty,
        final_candidate,
    )
    (
        family_selections,
        family_fold_results,
        family_predictions,
        family_competition,
        family_uncertainty,
        family_decisions,
    ) = run_family_backtests(
        datasets,
        adjustments,
        events,
        target,
        folds,
        core_selections,
        candidates,
        live,
        args.bootstrap_samples,
        full_core,
    )

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_root / "domestic_surprise_features.csv", index=False)
    history_long.to_csv(output_root / "domestic_history_long.csv", index=False)
    coverage.to_csv(output_root / "history_coverage.csv", index=False)
    selections.to_csv(output_root / "fold_selections.csv", index=False)
    fold_results.to_csv(output_root / "fold_results.csv", index=False)
    predictions.to_csv(output_root / "unseen_predictions.csv", index=False)
    fold_adjustments.to_csv(output_root / "unseen_team_adjustments.csv", index=False)
    competition.to_csv(output_root / "competition_summary.csv", index=False)
    uncertainty.to_csv(output_root / "dependency_uncertainty.csv", index=False)
    full_metrics.to_csv(output_root / "full_candidate_metrics.csv", index=False)
    normalization_comparison.to_csv(
        output_root / "normalization_comparison.csv", index=False
    )
    family_selections.to_csv(output_root / "family_fold_selections.csv", index=False)
    family_fold_results.to_csv(output_root / "family_fold_results.csv", index=False)
    family_predictions.to_csv(output_root / "family_unseen_predictions.csv", index=False)
    family_competition.to_csv(
        output_root / "family_competition_summary.csv", index=False
    )
    family_uncertainty.to_csv(
        output_root / "family_dependency_uncertainty.csv", index=False
    )
    family_decisions.to_csv(output_root / "family_decisions.csv", index=False)
    final_adjustments.to_csv(
        output_root / "final_candidate_team_adjustments.csv", index=False
    )
    manifest = {
        "feature": "SYMMETRIC_FIVE_SEASON_DOMESTIC_FINISH_SURPRISE",
        "decision": decision,
        "production_change": False,
        "seasons": list(seasons),
        "matches": int(len(events)),
        "candidate_count": len(candidates),
        "selected_full_candidate": asdict(final_candidate),
        "live_baseline": asdict(live),
        "history_contract": {
            "past_seasons_only": True,
            "weights_oldest_to_newest": [0.07, 0.13, 0.20, 0.27, 0.33],
            "minimum_history_seasons": MINIMUM_HISTORY_SEASONS,
            "missing_history_behavior": "ZERO_ADJUSTMENT",
            "identity": "persistent_club_id_plus_verified_name_aliases",
        },
        "formula_contract": {
            "symmetric_positive_negative_coefficient": True,
            "main_domestic_achievement_unchanged": True,
            "exposure_retention": "(1-effective_european_exposure)^exposure_power",
            "exposure_power_candidates": list(EXPOSURE_POWERS),
            "exposure_power_1_matches_before_blend_architecture": True,
        },
        "guardrails": guardrails,
    }
    (output_root / "decision.json").write_text(
        json.dumps(manifest, indent=2, default=json_default), encoding="utf-8"
    )
    write_report(
        output_root / "backtest_report.md",
        manifest,
        coverage,
        selections,
        fold_results,
        competition,
        uncertainty,
        normalization_comparison,
        family_decisions,
        family_fold_results,
        family_competition,
        final_adjustments,
    )
    print(f"Decision: {decision}")
    print(f"Full-data diagnostic candidate: {asdict(final_candidate)}")
    print(f"Report: {output_root / 'backtest_report.md'}")


def baseline_config() -> SymmetricDomesticSurpriseConfig:
    return SymmetricDomesticSurpriseConfig(
        coefficient=0.0,
        normalization="legacy_finish_curve",
        minimum_history_seasons=MINIMUM_HISTORY_SEASONS,
        max_abs_adjustment=50.0,
        exposure_power=1.0,
    )


def candidate_grid() -> tuple[SymmetricDomesticSurpriseConfig, ...]:
    candidates = {baseline_config()}
    candidates.update(
        SymmetricDomesticSurpriseConfig(
            coefficient=coefficient,
            normalization=normalization,
            minimum_history_seasons=MINIMUM_HISTORY_SEASONS,
            max_abs_adjustment=cap,
            exposure_power=exposure_power,
        )
        for normalization in NORMALIZATIONS
        for coefficient in COEFFICIENTS
        for cap in ADJUSTMENT_CAPS
        for exposure_power in EXPOSURE_POWERS
    )
    result = tuple(sorted(candidates))
    for candidate in result:
        candidate.validate()
    return result


def build_domestic_history_features(
    static_root: Path,
    static_config: AOEuropeanEloConfig,
    seasons: tuple[str, ...],
    club_identity_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    season_years = {season: int(season[:4]) for season in seasons}
    first_year = min(season_years.values()) - 5
    last_year = max(season_years.values())
    cache = load_standings_cache(static_root, tuple(range(first_year, last_year + 1)))
    identity = pd.read_csv(club_identity_path, dtype={"uefa_team_id": "string"})
    if identity.duplicated(["season", "local_team_id"]).any():
        raise ValueError("club identity contains duplicate season/local_team_id keys")
    name_variants = (
        identity.groupby("club_id")["team_name"]
        .agg(lambda values: tuple(sorted(set(map(str, values)))))
        .to_dict()
    )
    feature_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    for season in seasons:
        folder = static_root / season.replace("/", "-")
        teams = pd.read_csv(folder / "teams.csv")
        domestic = pd.read_csv(folder / "domestic_context.csv")
        ratings = compute_ao_first_elo_from_csv(
            folder / "teams.csv",
            folder / "country_coefficients.csv",
            folder / "domestic_context.csv",
            folder / "club_european_points.csv",
            static_config,
        )
        current = (
            teams.merge(domestic, on="team_id", validate="one_to_one")
            .merge(
                ratings[
                    [
                        "team_id",
                        "league_finish_score",
                        "league_strength",
                        "domestic_prior",
                        "european_prior",
                        "effective_european_exposure",
                        "ao_first_elo",
                    ]
                ],
                on="team_id",
                validate="one_to_one",
            )
            .merge(
                identity.loc[
                    identity["season"].eq(season), ["local_team_id", "club_id"]
                ],
                left_on="team_id",
                right_on="local_team_id",
                validate="one_to_one",
            )
        )
        target_year = season_years[season]
        for row in current.itertuples(index=False):
            country_key = normalize_country(str(row.country))
            history: dict[int, dict[str, object]] = {}
            for offset in range(5, 0, -1):
                source_year = target_year - offset
                table = cache.get((source_year, country_key))
                position: int | None = None
                team_count: int | None = None
                source_path: Path | None = None
                method = "NO_CACHED_SEASON"
                if table is not None:
                    lookup, team_count, source_path = table
                    position, method = resolve_identity_position(
                        name_variants[str(row.club_id)],
                        lookup,
                        current_team_name=str(row.team_name),
                    )
                    if position is None or not 1 <= position <= team_count:
                        position = None
                legacy_score = (
                    normalized_finish_score(position, team_count, "legacy_finish_curve")
                    if position is not None and team_count is not None
                    else None
                )
                direct_score = (
                    normalized_finish_score(position, team_count, "direct_percentile")
                    if position is not None and team_count is not None
                    else None
                )
                history[offset] = {
                    "position": position,
                    "team_count": team_count,
                    "legacy_score": legacy_score,
                    "direct_score": direct_score,
                    "method": method,
                }
                history_rows.append(
                    {
                        "season": season,
                        "team_id": int(row.team_id),
                        "team_name": row.team_name,
                        "club_id": row.club_id,
                        "country_code": row.country_code,
                        "history_offset": offset,
                        "source_year": source_year,
                        "domestic_position": position,
                        "league_team_count": team_count,
                        "legacy_finish_score": legacy_score,
                        "direct_percentile": direct_score,
                        "identity_match_method": method,
                        "source_cache": str(source_path) if source_path else "",
                    }
                )
            current_eligible = bool(
                pd.notna(row.domestic_position)
                and pd.notna(row.league_team_count)
                and int(row.league_team_count) > 1
            )
            current_direct = (
                normalized_finish_score(
                    int(row.domestic_position),
                    int(row.league_team_count),
                    "direct_percentile",
                )
                if current_eligible
                else np.nan
            )
            feature = {
                "season": season,
                "team_id": int(row.team_id),
                "team_name": row.team_name,
                "club_id": row.club_id,
                "country": row.country,
                "country_code": row.country_code,
                "competition": row.competition,
                "current_domestic_position": row.domestic_position,
                "current_league_team_count": row.league_team_count,
                "current_legacy_finish_score": float(row.league_finish_score),
                "current_direct_percentile": current_direct,
                "current_finish_eligible": current_eligible,
                "league_strength": float(row.league_strength),
                "achievement_scale": float(
                    static_config.achievement_alpha
                    + (1.0 - static_config.achievement_alpha)
                    * float(row.league_strength)
                ),
                "domestic_prior": float(row.domestic_prior),
                "european_prior": float(row.european_prior),
                "effective_european_exposure": float(
                    row.effective_european_exposure
                ),
                "baseline_ao_first_elo": float(row.ao_first_elo),
            }
            for offset in range(5, 0, -1):
                values = history[offset]
                feature[f"history_position_t_minus_{offset}"] = values["position"]
                feature[f"history_team_count_t_minus_{offset}"] = values["team_count"]
                feature[f"history_legacy_t_minus_{offset}"] = values["legacy_score"]
                feature[f"history_direct_t_minus_{offset}"] = values["direct_score"]
                feature[f"history_match_t_minus_{offset}"] = values["method"]
            feature["history_seasons_available"] = sum(
                history[offset]["position"] is not None for offset in range(5, 0, -1)
            )
            feature_rows.append(feature)
    features = pd.DataFrame(feature_rows).sort_values(["season", "team_id"])
    history_frame = pd.DataFrame(history_rows).sort_values(
        ["season", "team_id", "history_offset"], ascending=[True, True, False]
    )
    coverage = pd.DataFrame(
        [
            coverage_row(season, frame)
            for season, frame in features.groupby("season", sort=True)
        ]
        + [coverage_row("ALL", features)]
    )
    return features, history_frame, coverage


def coverage_row(season: str, frame: pd.DataFrame) -> dict[str, object]:
    eligible = frame["current_finish_eligible"].astype(bool)
    available = frame["history_seasons_available"]
    strict = eligible & available.eq(5)
    return {
        "season": season,
        "teams": len(frame),
        "current_finish_eligible": int(eligible.sum()),
        "history_3_plus": int((eligible & available.ge(3)).sum()),
        "history_4_plus": int((eligible & available.ge(4)).sum()),
        "history_5_complete": int(strict.sum()),
        "strict_eligible_rate": float(strict.mean()),
    }


def build_adjustments(
    features: pd.DataFrame,
    config: SymmetricDomesticSurpriseConfig,
    static_config: AOEuropeanEloConfig,
) -> pd.DataFrame:
    prefix = (
        "history_legacy_t_minus_"
        if config.normalization == "legacy_finish_curve"
        else "history_direct_t_minus_"
    )
    current_column = (
        "current_legacy_finish_score"
        if config.normalization == "legacy_finish_curve"
        else "current_direct_percentile"
    )
    rows = []
    for row in features.itertuples(index=False):
        current_value = getattr(row, current_column)
        history = [
            None
            if pd.isna(getattr(row, f"{prefix}{offset}"))
            else float(getattr(row, f"{prefix}{offset}"))
            for offset in range(5, 0, -1)
        ]
        if not bool(row.current_finish_eligible) or pd.isna(current_value):
            current_value = 0.0
            history = [None] * 5
        adjustment = calculate_symmetric_domestic_surprise_adjustment(
            current_finish_score=float(current_value),
            historical_finish_scores=history,
            domestic_prior=float(row.domestic_prior),
            european_prior=float(row.european_prior),
            effective_european_exposure=float(row.effective_european_exposure),
            domestic_achievement_component=static_config.domestic_achievement_component,
            achievement_scale=float(row.achievement_scale),
            config=config,
        )
        rows.append(
            {
                "season": row.season,
                "team_id": int(row.team_id),
                "team_name": row.team_name,
                "club_id": row.club_id,
                "country_code": row.country_code,
                "competition": row.competition,
                "normalization": config.normalization,
                "coefficient": config.coefficient,
                "max_abs_adjustment": config.max_abs_adjustment,
                "exposure_power": config.exposure_power,
                "current_domestic_position": row.current_domestic_position,
                "current_finish_score": current_value,
                "historical_finish_score": adjustment.historical_finish_score,
                "history_seasons": adjustment.history_seasons,
                "history_weight_coverage": adjustment.history_weight_coverage,
                "surprise_score": adjustment.surprise_score,
                "surprise_direction": (
                    "POSITIVE"
                    if adjustment.surprise_score > 0
                    else "NEGATIVE"
                    if adjustment.surprise_score < 0
                    else "NEUTRAL_OR_UNAVAILABLE"
                ),
                "surprise_component": adjustment.surprise_component,
                "domestic_prior_adjustment": adjustment.domestic_prior_adjustment,
                "ao_first_elo_adjustment": (
                    adjustment.adjusted_ao_first_elo - row.baseline_ao_first_elo
                ),
                "baseline_domestic_prior": row.domestic_prior,
                "adjusted_domestic_prior": adjustment.adjusted_domestic_prior,
                "baseline_ao_first_elo": row.baseline_ao_first_elo,
                "adjusted_ao_first_elo": adjustment.adjusted_ao_first_elo,
            }
        )
    return pd.DataFrame(rows)


def run_walk_forward_backtest(
    datasets: tuple[ReserveSeasonData, ...],
    adjustments: dict[SymmetricDomesticSurpriseConfig, pd.DataFrame],
    events: pd.DataFrame,
    target: pd.DataFrame,
    folds: tuple[tuple[tuple[str, ...], str], ...],
    core_selections: pd.DataFrame,
    candidates: tuple[SymmetricDomesticSurpriseConfig, ...],
    live: LiveBacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline = baseline_config()
    selection_rows = []
    result_rows = []
    prediction_frames = []
    adjustment_frames = []
    for fold, (train_seasons, test_season) in enumerate(folds, start=1):
        core = fold_core(core_selections, fold)
        selected, table = select_candidate_from_adjustments(
            datasets,
            adjustments,
            target,
            set(train_seasons),
            core,
            candidates,
            live,
        )
        selected_row = table.loc[
            table["candidate_key"].eq(config_key(selected))
        ].iloc[0]
        selection_rows.append(
            {
                "fold": fold,
                "train_seasons": "|".join(train_seasons),
                "test_season": test_season,
                "selected_candidate": config_key(selected),
                "selected_normalization": selected.normalization,
                "selected_coefficient": selected.coefficient,
                "selected_cap": selected.max_abs_adjustment,
                "selected_exposure_power": selected.exposure_power,
                "selected_is_baseline": selected == baseline,
                "train_brier_1x2": selected_row["brier_1x2"],
                "train_log_loss_1x2": selected_row["log_loss_1x2"],
                "train_ranking_score": selected_row["ranking_score"],
                "train_pairwise_accuracy": selected_row["pairwise_accuracy"],
            }
        )
        models: dict[str, pd.DataFrame] = {}
        for model, config in (("candidate", selected), ("baseline", baseline)):
            evaluated: Evaluation = evaluate_live(
                datasets,
                adjustments[config],
                {test_season},
                core,
                live,
                return_predictions=True,
            )
            ranking = initial_ranking(adjustments[config], target, {test_season})
            result_rows.append(
                {
                    "fold": fold,
                    "test_season": test_season,
                    "model": model,
                    "candidate_key": config_key(config),
                    **evaluated.metrics,
                    **ranking,
                }
            )
            models[model] = evaluated.predictions
        prediction_frames.append(
            pair_predictions(models["candidate"], models["baseline"], events, fold)
        )
        selected_adjustments = adjustments[selected].loc[
            adjustments[selected]["season"].eq(test_season)
        ].copy()
        selected_adjustments.insert(0, "fold", fold)
        adjustment_frames.append(selected_adjustments)
        print(
            f"  fold {fold}/{len(folds)} {test_season}: "
            f"method={selected.normalization}, theta={selected.coefficient:g}, "
            f"cap={selected.max_abs_adjustment:g}, q={selected.exposure_power:g}",
            flush=True,
        )
    return (
        pd.DataFrame(selection_rows),
        pd.DataFrame(result_rows),
        pd.concat(prediction_frames, ignore_index=True),
        pd.concat(adjustment_frames, ignore_index=True),
    )


def run_family_backtests(
    datasets: tuple[ReserveSeasonData, ...],
    adjustments: dict[SymmetricDomesticSurpriseConfig, pd.DataFrame],
    events: pd.DataFrame,
    target: pd.DataFrame,
    folds: tuple[tuple[tuple[str, ...], str], ...],
    core_selections: pd.DataFrame,
    candidates: tuple[SymmetricDomesticSurpriseConfig, ...],
    live: LiveBacktestConfig,
    bootstrap_samples: int,
    full_core: DynamicCoreConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selection_frames = []
    result_frames = []
    prediction_frames = []
    competition_frames = []
    uncertainty_frames = []
    decision_rows = []
    baseline = baseline_config()
    for normalization in NORMALIZATIONS:
        family_candidates = tuple(
            candidate
            for candidate in candidates
            if candidate == baseline or candidate.normalization == normalization
        )
        print(f"Family OOS comparison: {normalization}", flush=True)
        selections, results, predictions, _ = run_walk_forward_backtest(
            datasets,
            adjustments,
            events,
            target,
            folds,
            core_selections,
            family_candidates,
            live,
        )
        for frame in (selections, results, predictions):
            frame.insert(0, "candidate_family", normalization)
        competition = summarize_competitions(predictions)
        competition.insert(0, "candidate_family", normalization)
        uncertainty = comparison_uncertainty(predictions, bootstrap_samples)
        uncertainty.insert(0, "candidate_family", normalization)
        final_candidate, _ = select_candidate_from_adjustments(
            datasets,
            adjustments,
            target,
            {data.season for data in datasets},
            full_core,
            family_candidates,
            live,
        )
        decision, guards = promotion_decision(
            selections,
            results,
            competition.drop(columns="candidate_family"),
            uncertainty.drop(columns="candidate_family"),
            final_candidate,
        )
        decision_rows.append(
            {
                "candidate_family": normalization,
                "decision": decision,
                "selected_full_candidate": config_key(final_candidate),
                **guards,
            }
        )
        selection_frames.append(selections)
        result_frames.append(results)
        prediction_frames.append(predictions)
        competition_frames.append(competition)
        uncertainty_frames.append(uncertainty)
    return (
        pd.concat(selection_frames, ignore_index=True),
        pd.concat(result_frames, ignore_index=True),
        pd.concat(prediction_frames, ignore_index=True),
        pd.concat(competition_frames, ignore_index=True),
        pd.concat(uncertainty_frames, ignore_index=True),
        pd.DataFrame(decision_rows),
    )


def select_candidate_from_adjustments(
    datasets: tuple[ReserveSeasonData, ...],
    adjustments: dict[SymmetricDomesticSurpriseConfig, pd.DataFrame],
    target: pd.DataFrame,
    seasons: set[str],
    core: DynamicCoreConfig,
    candidates: tuple[SymmetricDomesticSurpriseConfig, ...],
    live: LiveBacktestConfig,
) -> tuple[SymmetricDomesticSurpriseConfig, pd.DataFrame]:
    baseline = baseline_config()
    baseline_rank = initial_ranking(adjustments[baseline], target, seasons)
    rows = []
    for candidate in candidates:
        ranking = initial_ranking(adjustments[candidate], target, seasons)
        rank_safe = bool(
            ranking["ranking_score"] >= baseline_rank["ranking_score"] - RANK_TOLERANCE
            and ranking["pairwise_accuracy"]
            >= baseline_rank["pairwise_accuracy"] - RANK_TOLERANCE
        )
        if rank_safe:
            metrics = evaluate_live(
                datasets, adjustments[candidate], seasons, core, live
            ).metrics
        else:
            metrics = {
                "matches": 0,
                "brier_1x2": float("inf"),
                "log_loss_1x2": float("inf"),
                "max_abs_match_delta": float("inf"),
                "max_pair_sum_error": float("inf"),
                "max_abs_rating_change": float("inf"),
            }
        rows.append(
            {
                "candidate_key": config_key(candidate),
                "normalization": candidate.normalization,
                "coefficient": candidate.coefficient,
                "max_abs_adjustment": candidate.max_abs_adjustment,
                "exposure_power": candidate.exposure_power,
                "is_baseline": candidate == baseline,
                "rank_safe": rank_safe,
                **metrics,
                **ranking,
                "distance": (
                    0.0
                    if candidate == baseline
                    else candidate.coefficient + candidate.max_abs_adjustment / 1000.0
                    + abs(candidate.exposure_power - 1.0) / 1000.0
                ),
                "config": json.dumps(asdict(candidate), sort_keys=True),
            }
        )
    table = pd.DataFrame(rows)
    selected_row = table.loc[table["rank_safe"]].sort_values(
        ["brier_1x2", "log_loss_1x2", "distance", "candidate_key"],
        kind="stable",
    ).iloc[0]
    selected = next(
        candidate
        for candidate in candidates
        if config_key(candidate) == selected_row.candidate_key
    )
    return selected, table


def summarize_normalizations(full_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    baseline = full_metrics.loc[full_metrics["is_baseline"]].iloc[0]
    rows.append({"candidate_family": "baseline", **baseline.to_dict()})
    for normalization in NORMALIZATIONS:
        pool = full_metrics.loc[
            full_metrics["normalization"].eq(normalization)
            & ~full_metrics["is_baseline"]
            & full_metrics["rank_safe"]
        ]
        if pool.empty:
            continue
        selected = pool.sort_values(
            ["brier_1x2", "log_loss_1x2", "distance", "candidate_key"],
            kind="stable",
        ).iloc[0]
        rows.append(
            {"candidate_family": normalization, **selected.to_dict()}
        )
    result = pd.DataFrame(rows)
    result["brier_difference_vs_baseline"] = (
        result["brier_1x2"] - float(baseline.brier_1x2)
    )
    result["log_loss_difference_vs_baseline"] = (
        result["log_loss_1x2"] - float(baseline.log_loss_1x2)
    )
    result["ranking_difference_vs_baseline"] = (
        result["ranking_score"] - float(baseline.ranking_score)
    )
    result["pairwise_difference_vs_baseline"] = (
        result["pairwise_accuracy"] - float(baseline.pairwise_accuracy)
    )
    return result


def promotion_decision(
    selections: pd.DataFrame,
    fold_results: pd.DataFrame,
    competition: pd.DataFrame,
    uncertainty: pd.DataFrame,
    final_candidate: SymmetricDomesticSurpriseConfig,
) -> tuple[str, dict[str, object]]:
    pivot = fold_results.pivot(index="fold", columns="model")
    brier_delta = pivot["brier_1x2"]["candidate"] - pivot["brier_1x2"]["baseline"]
    log_delta = (
        pivot["log_loss_1x2"]["candidate"]
        - pivot["log_loss_1x2"]["baseline"]
    )
    rank_delta = (
        pivot["ranking_score"]["candidate"] - pivot["ranking_score"]["baseline"]
    )
    pair_delta = (
        pivot["pairwise_accuracy"]["candidate"]
        - pivot["pairwise_accuracy"]["baseline"]
    )
    overall = competition.loc[competition["competition"].eq("ALL")].iloc[0]
    segments = competition.loc[competition["competition"].ne("ALL")]
    brier_ci = uncertainty.loc[
        (uncertainty["metric"].eq("brier_1x2"))
        & (uncertainty["method"].eq("conservative_envelope"))
    ].iloc[0]
    guards = {
        "selected_active_folds": int((~selections["selected_is_baseline"]).sum()),
        "brier_fold_wins": int((brier_delta < -1e-12).sum()),
        "log_loss_fold_wins": int((log_delta < -1e-12).sum()),
        "ranking_no_regression_folds": int((rank_delta >= -RANK_TOLERANCE).sum()),
        "pairwise_no_regression_folds": int((pair_delta >= -RANK_TOLERANCE).sum()),
        "ranking_improvement_folds": int((rank_delta > RANK_TOLERANCE).sum()),
        "pairwise_improvement_folds": int((pair_delta > RANK_TOLERANCE).sum()),
        "overall_brier_difference": float(overall.brier_difference),
        "overall_log_loss_difference": float(overall.log_loss_difference),
        "max_competition_brier_difference": float(segments.brier_difference.max()),
        "brier_ci_upper_95": float(brier_ci.ci_95_upper),
        "power_zero_sum": bool(fold_results["max_pair_sum_error"].max() <= 1e-9),
        "symmetric_contract": True,
        "strict_five_season_contract": bool(
            final_candidate.minimum_history_seasons == 5
        ),
    }
    passed = bool(
        guards["selected_active_folds"] >= 4
        and guards["brier_fold_wins"] >= 4
        and guards["log_loss_fold_wins"] >= 4
        and guards["ranking_no_regression_folds"] == 6
        and guards["pairwise_no_regression_folds"] == 6
        and guards["ranking_improvement_folds"] >= 4
        and guards["pairwise_improvement_folds"] >= 4
        and guards["overall_brier_difference"] <= 0
        and guards["overall_log_loss_difference"] <= 0
        and guards["max_competition_brier_difference"] <= 0
        and guards["brier_ci_upper_95"] <= 0
        and guards["power_zero_sum"]
    )
    return ("PROMOTE_CANDIDATE" if passed else "KEEP_BASELINE"), guards


def config_key(config: SymmetricDomesticSurpriseConfig) -> str:
    return json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))


def write_report(
    path: Path,
    manifest: dict[str, object],
    coverage: pd.DataFrame,
    selections: pd.DataFrame,
    fold_results: pd.DataFrame,
    competition: pd.DataFrame,
    uncertainty: pd.DataFrame,
    normalization_comparison: pd.DataFrame,
    family_decisions: pd.DataFrame,
    family_fold_results: pd.DataFrame,
    family_competition: pd.DataFrame,
    adjustments: pd.DataFrame,
) -> None:
    active = adjustments.loc[adjustments["ao_first_elo_adjustment"].ne(0)]
    positive = active.nlargest(12, "ao_first_elo_adjustment")[
        [
            "team_name",
            "season",
            "current_finish_score",
            "historical_finish_score",
            "surprise_score",
            "domestic_prior_adjustment",
            "ao_first_elo_adjustment",
        ]
    ]
    negative = active.nsmallest(12, "ao_first_elo_adjustment")[positive.columns]
    lines = [
        "# Symmetric Five-Season Domestic Surprise Backtest",
        "",
        f"Decision: **{manifest['decision']}**. Production was not changed.",
        "",
        "## Contract",
        "",
        "The main Domestic Achievement formula is unchanged. This isolated shadow layer",
        "compares the current completed domestic finish with five earlier completed seasons",
        "using 0.33/0.27/0.20/0.13/0.07 from newest to oldest. Positive and negative",
        "differences use the same coefficient. A club needs all five historical top-flight",
        "finishes; missing history produces zero adjustment rather than imputation.",
        "",
        "Both the existing finish curve and direct league-position percentile are tested",
        "inside the surprise calculation only. The selected domestic adjustment is measured",
        "against baseline AO First Elo. Exposure retention is tested as",
        "(1 - effective European exposure)^q for q in 0.5/1.0/1.5; q=1 reproduces",
        "the existing before-blend architecture.",
        "",
        f"Full-data diagnostic candidate: `{json.dumps(manifest['selected_full_candidate'], sort_keys=True)}`.",
        "",
        "## Coverage",
        "",
        markdown_table(coverage),
        "",
        "## Normalization comparison",
        "",
        markdown_table(
            normalization_comparison[
                [
                    "candidate_family",
                    "coefficient",
                    "max_abs_adjustment",
                    "exposure_power",
                    "brier_1x2",
                    "brier_difference_vs_baseline",
                    "log_loss_1x2",
                    "log_loss_difference_vs_baseline",
                    "ranking_difference_vs_baseline",
                    "pairwise_difference_vs_baseline",
                ]
            ]
        ),
        "",
        "## Separate normalization-family OOS decisions",
        "",
        markdown_table(family_decisions),
        "",
        "## Separate normalization-family fold results",
        "",
        markdown_table(family_fold_results),
        "",
        "## Separate normalization-family competition results",
        "",
        markdown_table(family_competition),
        "",
        "## Outer-fold selections",
        "",
        markdown_table(selections),
        "",
        "## Unseen fold results",
        "",
        markdown_table(fold_results),
        "",
        "## Competition results",
        "",
        markdown_table(competition),
        "",
        "## Dependency-aware uncertainty",
        "",
        markdown_table(uncertainty),
        "",
        "## Largest positive adjustments",
        "",
        markdown_table(positive),
        "",
        "## Largest negative adjustments",
        "",
        markdown_table(negative),
        "",
        "## Guardrails",
        "",
        "```json",
        json.dumps(manifest["guardrails"], indent=2),
        "```",
        "",
        "The full-data candidate is descriptive and is not independent evidence. Promotion",
        "depends on the six unseen outer folds and the ranking-first guardrails above.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
