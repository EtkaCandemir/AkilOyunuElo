from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
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
from ao_elo.controlled_live import calculate_goal_difference_multiplier  # noqa: E402
from ao_elo.domestic_surprise import (  # noqa: E402
    DomesticSurpriseConfig,
    calculate_domestic_surprise_adjustment,
    league_finish_score,
)
from ao_elo.evaluation import schedule_adjusted_team_performance  # noqa: E402
from ao_elo.pipeline import compute_ao_first_elo_from_csv  # noqa: E402
from ao_elo.robustness import one_x_two_probabilities_scalar  # noqa: E402
from scripts.build_backtest_stage_b import (  # noqa: E402
    build_position_lookup,
    normalize_country,
    read_standings,
    resolve_position,
)
from scripts.run_dynamic_core_calibration import (  # noqa: E402
    DynamicCoreConfig,
    expanding_folds,
)
from scripts.run_final_robustness import pairwise_ranking_accuracy  # noqa: E402
from scripts.run_match_context_backtest import (  # noqa: E402
    comparison_uncertainty,
    read_context_events,
)
from scripts.run_v2_achievement_reserve_calibration import (  # noqa: E402
    ReserveSeasonData,
    load_reserve_data,
)


STATIC_ROOT = ROOT / "data" / "backtest_stage_b_2018_2026"
EVENTS_PATH = ROOT / "data" / "external_elo_benchmark_2018_2026" / "exact_date_events.csv"
DYNAMIC_ROOT = ROOT / "output" / "v2_dynamic_calibration_2018_2026"
FINAL_CONTRACT = ROOT / "contracts" / "ao_european_elo_v2_final_candidate.json"
OUTPUT_ROOT = ROOT / "output" / "domestic_surprise_backtest_2018_2026"
CLUB_IDENTITY_PATH = ROOT / "data" / "club_identity" / "team_season_identity.csv"
POSITIVE_WEIGHTS = (0.0, 0.10, 0.20, 0.30, 0.40)
NEGATIVE_WEIGHTS = (0.0, 0.10, 0.20, 0.30, 0.40)
ADJUSTMENT_CAPS = (25.0, 50.0, 75.0)
LOOKBACKS = (2, 3, 4)
MINIMUM_HISTORY_SEASONS = 2
RANK_TOLERANCE = 1e-9


@dataclass(frozen=True)
class LiveBacktestConfig:
    elo_scale: float
    home_advantage: float
    k_factor: float
    draw_at_even: float
    draw_shape: float
    goal_alpha: float
    goal_tau: float
    goal_cap: int


@dataclass
class Evaluation:
    metrics: dict[str, float | int]
    predictions: pd.DataFrame
    end_ratings: pd.DataFrame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest positive and negative domestic-finish surprise adjustments"
    )
    parser.add_argument("--static-root", type=Path, default=STATIC_ROOT)
    parser.add_argument("--events-path", type=Path, default=EVENTS_PATH)
    parser.add_argument("--dynamic-root", type=Path, default=DYNAMIC_ROOT)
    parser.add_argument("--final-contract", type=Path, default=FINAL_CONTRACT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()

    static_root = args.static_root.resolve()
    events = read_context_events(args.events_path.resolve())
    dynamic_manifest = json.loads(
        (args.dynamic_root.resolve() / "selected_dynamic_model.json").read_text(
            encoding="utf-8"
        )
    )
    static_config = AOEuropeanEloConfig(**dynamic_manifest["static_config"])
    static_config.validate()
    final_contract = json.loads(args.final_contract.resolve().read_text(encoding="utf-8"))
    live = live_config_from_contract(final_contract)
    datasets, _ = load_reserve_data(static_root, args.events_path.resolve(), static_config)
    seasons = tuple(data.season for data in datasets)
    folds = expanding_folds(seasons)
    if len(folds) != 6:
        raise ValueError(f"Expected six outer folds, found {len(folds)}")
    core_selections = pd.read_csv(
        args.dynamic_root.resolve() / "core_fold_selections.csv"
    )
    target = schedule_adjusted_team_performance(events)

    print("Domestic history: parsing cached full league tables", flush=True)
    features, history_long, coverage = build_domestic_history_features(
        static_root, static_config, seasons
    )
    candidates = candidate_grid()
    print(
        f"Domestic surprise: {len(candidates)} candidates, {len(folds)} outer folds",
        flush=True,
    )
    selections, fold_results, predictions, fold_adjustments = run_walk_forward_backtest(
        datasets,
        features,
        events,
        target,
        folds,
        core_selections,
        candidates,
        live,
        static_config,
    )
    competition = summarize_competitions(predictions)
    uncertainty = comparison_uncertainty(predictions, args.bootstrap_samples)
    full_core = DynamicCoreConfig(**final_contract["dynamic_core"])
    final_candidate, full_metrics = select_candidate(
        datasets,
        features,
        target,
        set(seasons),
        full_core,
        candidates,
        live,
        static_config,
    )
    final_adjustments = build_adjustments(features, final_candidate, static_config)
    decision, guardrails = promotion_decision(
        selections, fold_results, competition, uncertainty, final_candidate
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
    final_adjustments.to_csv(output_root / "final_candidate_team_adjustments.csv", index=False)
    manifest = {
        "feature": "DOMESTIC_FINISH_SURPRISE",
        "decision": decision,
        "production_change": False,
        "seasons": list(seasons),
        "matches": int(len(events)),
        "candidate_count": len(candidates),
        "selected_full_candidate": asdict(final_candidate),
        "live_baseline": asdict(live),
        "history_contract": {
            "past_seasons_only": True,
            "minimum_history_seasons": MINIMUM_HISTORY_SEASONS,
            "missing_history_behavior": "ZERO_ADJUSTMENT",
            "identity": "country_plus_normalized_club_name",
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
        final_adjustments,
    )
    print(f"Decision: {decision}")
    print(f"Full candidate: {asdict(final_candidate)}")
    print(f"Report: {output_root / 'backtest_report.md'}")


def live_config_from_contract(payload: dict[str, object]) -> LiveBacktestConfig:
    core = payload["dynamic_core"]
    probability = payload["one_x_two_probability"]
    goal = payload["goal_margin"]
    config = LiveBacktestConfig(
        elo_scale=float(core["elo_scale"]),
        home_advantage=float(core["home_advantage"]),
        k_factor=float(core["k_factor"]),
        draw_at_even=float(probability["draw_at_even"]),
        draw_shape=float(probability["draw_shape"]),
        goal_alpha=float(goal["alpha"]),
        goal_tau=float(goal["tau"]),
        goal_cap=int(goal["goal_difference_cap"]),
    )
    if any(
        not math.isfinite(value)
        for value in (
            config.elo_scale,
            config.home_advantage,
            config.k_factor,
            config.draw_at_even,
            config.draw_shape,
            config.goal_alpha,
            config.goal_tau,
        )
    ):
        raise ValueError("Final candidate contract contains a non-finite value")
    if payload.get("active_power_carry") != 0.0:
        raise ValueError("Domestic surprise test expects the frozen no-carry candidate")
    return config


def candidate_grid() -> tuple[DomesticSurpriseConfig, ...]:
    baseline = DomesticSurpriseConfig()
    candidates = {baseline}
    candidates.update(
        DomesticSurpriseConfig(up, down, lookback, MINIMUM_HISTORY_SEASONS, cap)
        for up in POSITIVE_WEIGHTS
        for down in NEGATIVE_WEIGHTS
        if up > 0 and down > 0
        for lookback in LOOKBACKS
        for cap in ADJUSTMENT_CAPS
    )
    result = tuple(sorted(candidates))
    for candidate in result:
        candidate.validate()
    return result


def build_domestic_history_features(
    static_root: Path,
    static_config: AOEuropeanEloConfig,
    seasons: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    season_years = {season: int(season[:4]) for season in seasons}
    cache = load_standings_cache(static_root, tuple(season_years.values()))
    identity = pd.read_csv(CLUB_IDENTITY_PATH, dtype={"uefa_team_id": "string"})
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
                identity.loc[identity["season"].eq(season), ["local_team_id", "club_id"]],
                left_on="team_id",
                right_on="local_team_id",
                validate="one_to_one",
            )
        )
        target_year = season_years[season]
        for row in current.itertuples(index=False):
            country_key = normalize_country(str(row.country))
            history_scores: list[float | None] = []
            history_methods: list[str] = []
            for offset in range(4, 0, -1):
                source_year = target_year - offset
                table = cache.get((source_year, country_key))
                score: float | None = None
                position: int | None = None
                team_count: int | None = None
                method = "NO_CACHED_SEASON"
                if table is not None:
                    lookup, team_count, source_path = table
                    position, method = resolve_identity_position(
                        name_variants[str(row.club_id)],
                        lookup,
                        current_team_name=str(row.team_name),
                    )
                    if position is not None and 1 <= position <= team_count:
                        score = league_finish_score(position, team_count)
                    else:
                        position = None
                else:
                    source_path = None
                history_scores.append(score)
                history_methods.append(method)
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
                        "finish_score": score,
                        "identity_match_method": method,
                        "source_cache": str(source_path) if source_path else "",
                    }
                )
            current_eligible = bool(
                pd.notna(row.domestic_position)
                and pd.notna(row.league_team_count)
                and int(row.league_team_count) > 1
            )
            feature_rows.append(
                {
                    "season": season,
                    "team_id": int(row.team_id),
                    "team_name": row.team_name,
                    "club_id": row.club_id,
                    "country": row.country,
                    "country_code": row.country_code,
                    "competition": row.competition,
                    "current_domestic_position": row.domestic_position,
                    "current_league_team_count": row.league_team_count,
                    "current_finish_score": float(row.league_finish_score),
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
                    "history_t_minus_4": history_scores[0],
                    "history_t_minus_3": history_scores[1],
                    "history_t_minus_2": history_scores[2],
                    "history_t_minus_1": history_scores[3],
                    "history_match_t_minus_4": history_methods[0],
                    "history_match_t_minus_3": history_methods[1],
                    "history_match_t_minus_2": history_methods[2],
                    "history_match_t_minus_1": history_methods[3],
                    "history_seasons_available": int(
                        sum(value is not None for value in history_scores)
                    ),
                }
            )
    features = pd.DataFrame(feature_rows).sort_values(["season", "team_id"])
    history = pd.DataFrame(history_rows).sort_values(
        ["season", "team_id", "history_offset"], ascending=[True, True, False]
    )
    coverage_rows = []
    for season, frame in features.groupby("season", sort=True):
        coverage_rows.append(coverage_row(season, frame))
    coverage_rows.append(coverage_row("ALL", features))
    return features, history, pd.DataFrame(coverage_rows)


def load_standings_cache(
    static_root: Path,
    years: tuple[int, ...],
) -> dict[tuple[int, str], tuple[dict[str, int], int, Path]]:
    result: dict[tuple[int, str], tuple[dict[str, int], int, Path]] = {}
    cache_root = static_root / "_source_cache"
    for year in sorted(set(years)):
        for path in sorted(cache_root.glob(f"{year}_*.html")):
            country_key = path.stem[len(str(year)) + 1 :].rsplit("_", 1)[0]
            try:
                tables = read_standings(path)
            except ValueError:
                continue
            team_count = max(len(table) for table in tables)
            if not 4 <= team_count <= 30:
                continue
            key = (year, country_key)
            if key in result:
                raise ValueError(f"Duplicate cached standings for {year}/{country_key}")
            result[key] = (build_position_lookup(tables), team_count, path)
    return result


def resolve_identity_position(
    team_names: tuple[str, ...],
    lookup: dict[str, int],
    *,
    current_team_name: str,
) -> tuple[int | None, str]:
    matches = []
    method_priority = {
        "league_table_exact": 0,
        "league_table_substring": 1,
        "league_table_fuzzy": 2,
    }
    for team_name in team_names:
        position, method = resolve_position(team_name, lookup)
        if position is not None:
            matches.append((int(position), method, team_name))
    positions = {position for position, _, _ in matches}
    if not matches:
        return None, "UNRESOLVED"
    if len(positions) > 1:
        return None, "IDENTITY_VARIANT_CONFLICT"
    _, method, matched_name = min(
        matches, key=lambda value: (method_priority.get(value[1], 99), value[2])
    )
    suffix = "identity_alias" if matched_name != current_team_name else "current_name"
    return next(iter(positions)), f"{method}_{suffix}"


def coverage_row(season: str, frame: pd.DataFrame) -> dict[str, object]:
    eligible = frame["current_finish_eligible"].astype(bool)
    enough = frame["history_seasons_available"].ge(MINIMUM_HISTORY_SEASONS) & eligible
    return {
        "season": season,
        "teams": len(frame),
        "current_finish_eligible": int(eligible.sum()),
        "history_1_plus": int((frame["history_seasons_available"].ge(1) & eligible).sum()),
        "history_2_plus": int(enough.sum()),
        "history_3_plus": int((frame["history_seasons_available"].ge(3) & eligible).sum()),
        "history_4": int((frame["history_seasons_available"].eq(4) & eligible).sum()),
        "backtest_eligible_rate": float(enough.mean()),
    }


def build_adjustments(
    features: pd.DataFrame,
    config: DomesticSurpriseConfig,
    static_config: AOEuropeanEloConfig,
) -> pd.DataFrame:
    rows = []
    history_columns = [f"history_t_minus_{offset}" for offset in range(4, 0, -1)]
    for row in features.itertuples(index=False):
        history = [
            None if pd.isna(getattr(row, column)) else float(getattr(row, column))
            for column in history_columns
        ]
        if not bool(row.current_finish_eligible):
            history = [None] * 4
        adjustment = calculate_domestic_surprise_adjustment(
            current_finish_score=float(row.current_finish_score),
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
                "current_domestic_position": row.current_domestic_position,
                "current_finish_score": row.current_finish_score,
                "historical_finish_score": adjustment.historical_finish_score,
                "history_seasons": adjustment.history_seasons,
                "history_reliability": adjustment.history_reliability,
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
    features: pd.DataFrame,
    events: pd.DataFrame,
    target: pd.DataFrame,
    folds: tuple[tuple[tuple[str, ...], str], ...],
    core_selections: pd.DataFrame,
    candidates: tuple[DomesticSurpriseConfig, ...],
    live: LiveBacktestConfig,
    static_config: AOEuropeanEloConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selections = []
    fold_rows = []
    prediction_frames = []
    adjustment_frames = []
    baseline = DomesticSurpriseConfig()
    adjustments = {
        config: build_adjustments(features, config, static_config) for config in candidates
    }
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
        selected_row = table.loc[table["candidate_key"].eq(config_key(selected))].iloc[0]
        selections.append(
            {
                "fold": fold,
                "train_seasons": "|".join(train_seasons),
                "test_season": test_season,
                "selected_candidate": config_key(selected),
                "selected_config": json.dumps(asdict(selected), sort_keys=True),
                "selected_is_baseline": selected == baseline,
                "train_brier_1x2": selected_row["brier_1x2"],
                "train_log_loss_1x2": selected_row["log_loss_1x2"],
                "train_ranking_score": selected_row["ranking_score"],
                "train_pairwise_accuracy": selected_row["pairwise_accuracy"],
            }
        )
        models = {}
        for model, config in (("candidate", selected), ("baseline", baseline)):
            evaluated = evaluate_live(
                datasets,
                adjustments[config],
                {test_season},
                core,
                live,
                return_predictions=True,
            )
            ranking = initial_ranking(
                adjustments[config], target, {test_season}
            )
            fold_rows.append(
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
        paired = pair_predictions(models["candidate"], models["baseline"], events, fold)
        prediction_frames.append(paired)
        selected_adjustments = adjustments[selected].loc[
            adjustments[selected]["season"].eq(test_season)
        ].copy()
        selected_adjustments.insert(0, "fold", fold)
        adjustment_frames.append(selected_adjustments)
        print(
            f"  fold {fold}/{len(folds)} {test_season}: "
            f"up={selected.positive_weight:g}, down={selected.negative_weight:g}, "
            f"lookback={selected.lookback_seasons}, cap={selected.max_abs_adjustment:g}",
            flush=True,
        )
    return (
        pd.DataFrame(selections),
        pd.DataFrame(fold_rows),
        pd.concat(prediction_frames, ignore_index=True),
        pd.concat(adjustment_frames, ignore_index=True),
    )


def select_candidate(
    datasets: tuple[ReserveSeasonData, ...],
    features: pd.DataFrame,
    target: pd.DataFrame,
    seasons: set[str],
    core: DynamicCoreConfig,
    candidates: tuple[DomesticSurpriseConfig, ...],
    live: LiveBacktestConfig,
    static_config: AOEuropeanEloConfig,
) -> tuple[DomesticSurpriseConfig, pd.DataFrame]:
    adjustments = {
        config: build_adjustments(features, config, static_config) for config in candidates
    }
    return select_candidate_from_adjustments(
        datasets, adjustments, target, seasons, core, candidates, live
    )


def select_candidate_from_adjustments(
    datasets: tuple[ReserveSeasonData, ...],
    adjustments: dict[DomesticSurpriseConfig, pd.DataFrame],
    target: pd.DataFrame,
    seasons: set[str],
    core: DynamicCoreConfig,
    candidates: tuple[DomesticSurpriseConfig, ...],
    live: LiveBacktestConfig,
) -> tuple[DomesticSurpriseConfig, pd.DataFrame]:
    baseline = DomesticSurpriseConfig()
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
            evaluated = evaluate_live(
                datasets, adjustments[candidate], seasons, core, live
            )
            metrics = evaluated.metrics
        else:
            metrics = {
                "matches": 0,
                "brier_1x2": float("inf"),
                "log_loss_1x2": float("inf"),
                "max_abs_match_delta": float("inf"),
                "max_pair_sum_error": float("inf"),
            }
        rows.append(
            {
                "candidate_key": config_key(candidate),
                "is_baseline": candidate == baseline,
                "rank_safe": rank_safe,
                **metrics,
                **ranking,
                "distance": candidate.positive_weight
                + candidate.negative_weight
                + candidate.max_abs_adjustment / 1000.0
                + candidate.lookback_seasons / 1000.0,
                "config": json.dumps(asdict(candidate), sort_keys=True),
            }
        )
    table = pd.DataFrame(rows)
    selected_row = table.loc[table["rank_safe"]].sort_values(
        ["brier_1x2", "log_loss_1x2", "distance", "candidate_key"],
        kind="stable",
    ).iloc[0]
    selected = next(
        config for config in candidates if config_key(config) == selected_row.candidate_key
    )
    return selected, table


def evaluate_live(
    datasets: tuple[ReserveSeasonData, ...],
    adjustments: pd.DataFrame,
    seasons: set[str],
    core: DynamicCoreConfig,
    live: LiveBacktestConfig,
    *,
    return_predictions: bool = False,
) -> Evaluation:
    core.validate()
    adjustment_index = adjustments.set_index(["season", "team_id"])
    metric_rows = []
    prediction_rows = []
    end_rows = []
    for data in datasets:
        if data.season not in seasons:
            continue
        season_core = data.goal.carry.core
        power = season_core.initial_ratings.copy()
        for team_id in season_core.active_team_ids:
            power[team_id] = float(
                adjustment_index.loc[(data.season, int(team_id)), "adjusted_ao_first_elo"]
            )
        start = power.copy()
        brier_sum = 0.0
        log_sum = 0.0
        max_delta = 0.0
        max_pair_error = 0.0
        for index, (home_id, away_id, actual, neutral, competition) in enumerate(
            zip(
                season_core.home_team_ids,
                season_core.away_team_ids,
                season_core.actual_home_scores,
                season_core.neutral_flags,
                season_core.competitions,
            )
        ):
            edge = 0.0 if bool(neutral) else core.home_advantage
            difference = float(power[home_id] - power[away_id] + edge)
            expected = float(
                np.clip(
                    1.0 / (1.0 + 10.0 ** (-difference / core.elo_scale)),
                    1e-12,
                    1.0 - 1e-12,
                )
            )
            probabilities = one_x_two_probabilities_scalar(
                expected, live.draw_at_even, live.draw_shape
            )
            observed = 0 if actual == 1.0 else 1 if actual == 0.5 else 2
            target_vector = tuple(1.0 if value == observed else 0.0 for value in range(3))
            brier = sum(
                (probability - observed_value) ** 2
                for probability, observed_value in zip(probabilities, target_vector)
            )
            log_loss = -math.log(max(probabilities[observed], 1e-15))
            goal_multiplier = calculate_goal_difference_multiplier(
                int(data.goal.goal_differences[index]),
                difference,
                live.goal_alpha,
                live.goal_tau,
                decided_on_penalties=bool(data.goal.penalty_flags[index]),
                is_draw=float(actual) == 0.5,
                goal_cap=live.goal_cap,
            )
            delta = core.k_factor * goal_multiplier * (float(actual) - expected)
            pair_before = float(power[home_id] + power[away_id])
            power[home_id] += delta
            power[away_id] -= delta
            max_pair_error = max(
                max_pair_error,
                abs(float(power[home_id] + power[away_id]) - pair_before),
            )
            max_delta = max(max_delta, abs(float(delta)))
            brier_sum += brier
            log_sum += log_loss
            if return_predictions:
                prediction_rows.append(
                    {
                        "match_id": str(season_core.match_ids[index]),
                        "season": data.season,
                        "competition": str(competition),
                        "home_team_id": int(home_id),
                        "away_team_id": int(away_id),
                        "actual_home_score": float(actual),
                        "expected_home_score": expected,
                        "home_probability": probabilities[0],
                        "draw_probability": probabilities[1],
                        "away_probability": probabilities[2],
                        "brier_1x2": brier,
                        "log_loss_1x2": log_loss,
                        "goal_multiplier": goal_multiplier,
                        "power_delta": float(delta),
                    }
                )
        active = season_core.active_team_ids
        metric_rows.append(
            {
                "matches": len(season_core.match_ids),
                "brier_1x2": brier_sum / len(season_core.match_ids),
                "log_loss_1x2": log_sum / len(season_core.match_ids),
                "max_abs_match_delta": max_delta,
                "max_pair_sum_error": max_pair_error,
                "max_abs_rating_change": float(
                    np.max(np.abs(power[active] - start[active]))
                ),
            }
        )
        end_rows.extend(
            {
                "season": data.season,
                "team_id": int(team_id),
                "initial_rating": float(start[team_id]),
                "end_live_rating": float(power[team_id]),
            }
            for team_id in active
        )
    matches = sum(row["matches"] for row in metric_rows)
    metrics = {
        "matches": int(matches),
        "brier_1x2": sum(row["brier_1x2"] * row["matches"] for row in metric_rows)
        / matches,
        "log_loss_1x2": sum(
            row["log_loss_1x2"] * row["matches"] for row in metric_rows
        )
        / matches,
        "max_abs_match_delta": max(row["max_abs_match_delta"] for row in metric_rows),
        "max_pair_sum_error": max(row["max_pair_sum_error"] for row in metric_rows),
        "max_abs_rating_change": max(
            row["max_abs_rating_change"] for row in metric_rows
        ),
    }
    return Evaluation(metrics, pd.DataFrame(prediction_rows), pd.DataFrame(end_rows))


def initial_ranking(
    adjustments: pd.DataFrame,
    target: pd.DataFrame,
    seasons: set[str],
) -> dict[str, float | int]:
    table = target.loc[target["season"].isin(seasons)].merge(
        adjustments[
            ["season", "team_id", "baseline_ao_first_elo", "adjusted_ao_first_elo"]
        ],
        on=["season", "team_id"],
        validate="many_to_one",
    )
    rows = []
    for (_, _), frame in table.groupby(["season", "competition"], sort=True):
        if len(frame) < 3:
            continue
        spearman = frame["adjusted_ao_first_elo"].corr(
            frame["schedule_adjusted_score"], method="spearman"
        )
        if pd.isna(spearman):
            continue
        rows.append(
            {
                "teams": len(frame),
                "pairs": len(frame) * (len(frame) - 1) / 2,
                "spearman": float(spearman),
                "pairwise": pairwise_ranking_accuracy(
                    frame["adjusted_ao_first_elo"].to_numpy(float),
                    frame["schedule_adjusted_score"].to_numpy(float),
                ),
            }
        )
    if not rows:
        return {"ranking_groups": 0, "ranking_score": float("nan"), "pairwise_accuracy": float("nan")}
    return {
        "ranking_groups": len(rows),
        "ranking_score": float(
            np.average([row["spearman"] for row in rows], weights=[row["teams"] for row in rows])
        ),
        "pairwise_accuracy": float(
            np.average([row["pairwise"] for row in rows], weights=[row["pairs"] for row in rows])
        ),
    }


def fold_core(table: pd.DataFrame, fold: int) -> DynamicCoreConfig:
    row = table.loc[table["fold"].eq(fold)]
    if len(row) != 1:
        raise ValueError(f"Expected one dynamic core selection for fold {fold}")
    config = DynamicCoreConfig(
        float(row.iloc[0]["selected_scale"]),
        float(row.iloc[0]["selected_home_advantage"]),
        float(row.iloc[0]["selected_k"]),
    )
    config.validate()
    return config


def pair_predictions(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    events: pd.DataFrame,
    fold: int,
) -> pd.DataFrame:
    candidate = candidate.rename(
        columns={
            column: f"candidate_{column}"
            for column in (
                "expected_home_score",
                "home_probability",
                "draw_probability",
                "away_probability",
                "brier_1x2",
                "log_loss_1x2",
                "power_delta",
            )
        }
    )
    baseline = baseline[
        [
            "match_id",
            "expected_home_score",
            "home_probability",
            "draw_probability",
            "away_probability",
            "brier_1x2",
            "log_loss_1x2",
            "power_delta",
        ]
    ].rename(
        columns={
            column: f"baseline_{column}"
            for column in (
                "expected_home_score",
                "home_probability",
                "draw_probability",
                "away_probability",
                "brier_1x2",
                "log_loss_1x2",
                "power_delta",
            )
        }
    )
    paired = candidate.merge(baseline, on="match_id", validate="one_to_one")
    paired = paired.merge(
        events[["match_id", "tie_id", "kickoff_utc"]],
        on="match_id",
        validate="one_to_one",
    )
    paired.insert(0, "fold", fold)
    paired["brier_difference"] = (
        paired["candidate_brier_1x2"] - paired["baseline_brier_1x2"]
    )
    paired["log_loss_difference"] = (
        paired["candidate_log_loss_1x2"] - paired["baseline_log_loss_1x2"]
    )
    return paired


def summarize_competitions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for competition, frame in [
        ("ALL", predictions),
        *predictions.groupby("competition", sort=True),
    ]:
        rows.append(
            {
                "competition": competition,
                "matches": len(frame),
                "candidate_brier_1x2": frame["candidate_brier_1x2"].mean(),
                "baseline_brier_1x2": frame["baseline_brier_1x2"].mean(),
                "brier_difference": frame["brier_difference"].mean(),
                "candidate_log_loss_1x2": frame["candidate_log_loss_1x2"].mean(),
                "baseline_log_loss_1x2": frame["baseline_log_loss_1x2"].mean(),
                "log_loss_difference": frame["log_loss_difference"].mean(),
            }
        )
    return pd.DataFrame(rows)


def promotion_decision(
    selections: pd.DataFrame,
    fold_results: pd.DataFrame,
    competition: pd.DataFrame,
    uncertainty: pd.DataFrame,
    final_candidate: DomesticSurpriseConfig,
) -> tuple[str, dict[str, object]]:
    pivot = fold_results.pivot(index="fold", columns="model")
    brier_delta = pivot["brier_1x2"]["candidate"] - pivot["brier_1x2"]["baseline"]
    log_delta = pivot["log_loss_1x2"]["candidate"] - pivot["log_loss_1x2"]["baseline"]
    rank_delta = pivot["ranking_score"]["candidate"] - pivot["ranking_score"]["baseline"]
    pair_delta = pivot["pairwise_accuracy"]["candidate"] - pivot["pairwise_accuracy"]["baseline"]
    overall = competition.loc[competition["competition"].eq("ALL")].iloc[0]
    segments = competition.loc[competition["competition"].ne("ALL")]
    brier_ci = uncertainty.loc[uncertainty["metric"].eq("brier_1x2")]
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
        "brier_ci_upper_95": float(brier_ci["ci_95_upper"].max()),
        "zero_sum": bool(fold_results["max_pair_sum_error"].max() <= 1e-9),
        "full_candidate_has_both_directions": bool(
            final_candidate.positive_weight > 0 and final_candidate.negative_weight > 0
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
        and guards["zero_sum"]
        and guards["full_candidate_has_both_directions"]
    )
    return ("PROMOTE_CANDIDATE" if passed else "KEEP_BASELINE"), guards


def config_key(config: DomesticSurpriseConfig) -> str:
    return json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))


def write_report(
    path: Path,
    manifest: dict[str, object],
    coverage: pd.DataFrame,
    selections: pd.DataFrame,
    fold_results: pd.DataFrame,
    competition: pd.DataFrame,
    uncertainty: pd.DataFrame,
    adjustments: pd.DataFrame,
) -> None:
    selected = manifest["selected_full_candidate"]
    positive = adjustments.nlargest(10, "ao_first_elo_adjustment")[
        ["team_name", "season", "surprise_score", "ao_first_elo_adjustment"]
    ]
    negative = adjustments.nsmallest(10, "ao_first_elo_adjustment")[
        ["team_name", "season", "surprise_score", "ao_first_elo_adjustment"]
    ]
    lines = [
        "# Domestic Finish Surprise Backtest",
        "",
        f"Decision: **{manifest['decision']}**. Production was not changed.",
        "",
        "## Model",
        "",
        "The candidate compares the current normalized league finish with a recency-weighted",
        "history built only from earlier completed domestic seasons. Positive and negative",
        "surprises have separate coefficients. Missing or insufficient history produces zero",
        "adjustment. The adjustment is applied to Domestic Prior before European blending, so",
        "its final AO First Elo effect is naturally reduced by European exposure.",
        "",
        f"Full-data candidate: `{json.dumps(selected, sort_keys=True)}`.",
        "",
        "## Coverage",
        "",
        markdown_table(coverage),
        "",
        "## Outer-fold selections",
        "",
        markdown_table(selections[
            [
                "fold",
                "test_season",
                "selected_is_baseline",
                "train_brier_1x2",
                "train_log_loss_1x2",
                "train_ranking_score",
                "train_pairwise_accuracy",
            ]
        ]),
        "",
        "## Fold results",
        "",
        markdown_table(fold_results),
        "",
        "## Competition results",
        "",
        markdown_table(competition),
        "",
        "## Uncertainty",
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
        "The 2018/19 and 2019/20 snapshots have limited pre-history because the local full-table",
        "cache begins in 2018. Reliability shrinkage prevents this from becoming a fabricated",
        "signal. Identity-ambiguous or lower-division history is not imputed.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    rows = [columns, ["---"] * len(columns)]
    for record in frame.itertuples(index=False, name=None):
        rows.append([markdown_value(value) for value in record])
    return "\n".join(
        "| " + " | ".join(str(value) for value in row) + " |" for row in rows
    )


def markdown_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.8f}"
    return str(value).replace("|", "\\|")


def json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


if __name__ == "__main__":
    main()
