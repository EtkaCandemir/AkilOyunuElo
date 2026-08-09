from __future__ import annotations

import argparse
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

from ao_elo.club_identity import attach_match_club_ids  # noqa: E402
from ao_elo.config import AOEuropeanEloConfig  # noqa: E402
from ao_elo.evaluation import dependency_robust_loss_difference_ci  # noqa: E402
from ao_elo.team_venue_context import (  # noqa: E402
    TeamVenueContextConfig,
    contextual_home_expectation,
    estimate_team_venue_effect,
)
from scripts.run_controlled_goal_progression_backtest import (  # noqa: E402
    prepare_controlled_data,
)
from scripts.run_dynamic_core_calibration import expanding_folds  # noqa: E402
from scripts.run_final_robustness import load_team_season_identity  # noqa: E402
from scripts.run_stage_weighted_progression_backtest import (  # noqa: E402
    DOMESTIC_ADJUSTMENTS,
    DYNAMIC_MANIFEST,
    EVENTS_PATH,
    MODEL_CURRENT,
    PRODUCTION_CONTRACT,
    STATIC_DATA_ROOT,
    XG_DATA,
    BonusCandidate,
    evaluate_candidate,
    load_domestic_adjustments,
    load_xg_map,
    markdown_table,
    probability_vector,
    validate_production_contract,
)
from scripts.run_v2_achievement_reserve_calibration import (  # noqa: E402
    load_reserve_data,
)
from scripts.run_v2_evaluation_upgrade import read_events  # noqa: E402


OUTPUT_ROOT = ROOT / "output" / "team_venue_context_backtest_2018_2026"
MODEL_BASELINE = "GLOBAL_HOME_ADVANTAGE"
MODEL_NESTED = "NESTED_TEAM_VENUE"
BASELINE_KEY = "global_h_148_544266"
RANK_TOLERANCE = 1e-12


@dataclass(frozen=True)
class VenueCandidate:
    key: str
    config: TeamVenueContextConfig | None

    @property
    def complexity(self) -> int:
        return int(self.config is not None)


@dataclass
class VenuePredictionSet:
    predictions: pd.DataFrame
    profiles: pd.DataFrame


def candidate_grid() -> tuple[VenueCandidate, ...]:
    candidates = [VenueCandidate(BASELINE_KEY, None)]
    for window in (3, 5):
        for decay in (0.75, 1.0):
            for shrinkage in (6.0, 10.0, 15.0, 20.0):
                for cap in (25.0, 35.0, 50.0, 75.0, 100.0, 150.0):
                    config = TeamVenueContextConfig(
                        window,
                        decay,
                        shrinkage,
                        cap,
                        cap,
                        0.0,
                        300.0,
                    )
                    config.validate()
                    candidates.append(VenueCandidate(config.key, config))
    if len(candidates) != 97:
        raise ValueError("Unexpected team venue candidate grid")
    return tuple(candidates)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prediction-only nested backtest of team home and away effects"
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

    contract = json.loads(
        args.production_contract.resolve().read_text(encoding="utf-8")
    )
    core, parameters = validate_production_contract(contract)
    dynamic = json.loads(
        args.dynamic_manifest.resolve().read_text(encoding="utf-8")
    )
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
    identity = load_team_season_identity()
    domestic = load_domestic_adjustments(args.domestic_adjustments.resolve(), datasets)
    xg = load_xg_map(args.xg_data.resolve(), datasets)
    baseline = build_production_baseline(
        datasets,
        core,
        parameters,
        domestic,
        xg,
        identity,
        seasons,
    )
    candidates = candidate_grid()

    print(
        f"Team venue context: building {len(candidates) - 1} causal profile candidates",
        flush=True,
    )
    prediction_sets: dict[str, VenuePredictionSet] = {
        BASELINE_KEY: VenuePredictionSet(baseline.copy(), pd.DataFrame())
    }
    for index, candidate in enumerate(candidates[1:], start=1):
        assert candidate.config is not None
        prediction_sets[candidate.key] = build_context_predictions(
            baseline,
            seasons,
            core.elo_scale,
            core.home_advantage,
            float(parameters["draw_at_even"]),
            float(parameters["draw_shape"]),
            candidate.config,
        )
        if index % 8 == 0 or index == len(candidates) - 1:
            print(f"  candidate {index}/{len(candidates) - 1}", flush=True)

    nested = run_nested_walk_forward(
        prediction_sets,
        candidates,
        folds,
        seasons,
    )
    surface = build_candidate_surface(prediction_sets, candidates, folds)
    competition = build_competition_summary(nested["predictions"])
    uncertainty = build_venue_uncertainty(
        nested["predictions"],
        baseline_model=MODEL_BASELINE,
        candidate_model=MODEL_NESTED,
        bootstrap_samples=args.bootstrap_samples,
    )
    best_fixed_key = str(
        surface.loc[surface["candidate_key"].ne(BASELINE_KEY)]
        .sort_values(["brier_1x2", "log_loss_1x2", "candidate_key"], kind="stable")
        .iloc[0]["candidate_key"]
    )
    test_seasons = {test_season for _, test_season in folds}
    fixed_uncertainty = build_fixed_uncertainty(
        prediction_sets[BASELINE_KEY].predictions.loc[
            prediction_sets[BASELINE_KEY].predictions["season"].isin(test_seasons)
        ],
        prediction_sets[best_fixed_key].predictions.loc[
            prediction_sets[best_fixed_key].predictions["season"].isin(test_seasons)
        ],
        best_fixed_key,
        bootstrap_samples=args.bootstrap_samples,
    )
    full_selection = select_candidate(
        prediction_sets,
        candidates,
        set(seasons),
    )
    full_selected_key = str(full_selection["candidate_key"])
    full_selected_profiles = (
        prediction_sets[full_selected_key].profiles.copy()
        if full_selected_key != BASELINE_KEY
        else pd.DataFrame(columns=["season", "club_id", "candidate_key"])
    )
    profile_summary = build_profile_summary(
        nested["selected_profiles"],
        nested["predictions"],
        full_selected_profiles,
        prediction_sets[full_selected_key].predictions,
    )
    decision = decide_model(
        nested["fold_results"],
        competition,
        uncertainty,
        fixed_uncertainty,
        full_selection,
        nested["selected_profiles"],
        nested["predictions"],
    )

    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    surface.to_csv(output / "candidate_surface.csv", index=False)
    nested["fold_selections"].to_csv(output / "fold_selections.csv", index=False)
    nested["fold_results"].to_csv(output / "fold_results.csv", index=False)
    nested["predictions"].to_csv(output / "unseen_predictions.csv", index=False)
    competition.to_csv(output / "competition_summary.csv", index=False)
    nested["selected_profiles"].to_csv(output / "team_venue_profiles.csv", index=False)
    full_selected_profiles.to_csv(
        output / "full_selected_team_venue_profiles.csv", index=False
    )
    profile_summary.to_csv(output / "profile_summary.csv", index=False)
    uncertainty.to_csv(output / "dependency_uncertainty.csv", index=False)
    fixed_uncertainty.to_csv(
        output / "retrospective_best_fixed_uncertainty.csv", index=False
    )
    (output / "selected_candidate.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(
        output / "backtest_report.md",
        contract,
        seasons,
        nested["fold_selections"],
        nested["fold_results"],
        surface,
        competition,
        uncertainty,
        fixed_uncertainty,
        profile_summary,
        decision,
        len(tie_audit),
    )
    print(f"Decision: {decision['decision']}")
    print(f"Full-history selection: {decision['full_history_candidate_key']}")
    print(f"Report: {output / 'backtest_report.md'}")


def build_production_baseline(
    datasets,
    core,
    parameters,
    domestic,
    xg,
    identity: pd.DataFrame,
    seasons: tuple[str, ...],
) -> pd.DataFrame:
    current = BonusCandidate(
        MODEL_CURRENT,
        "FIXED_FIVE",
        "EQUAL_FIVE",
        60.0,
        True,
    )
    evaluation = evaluate_candidate(
        datasets,
        pd.DataFrame(
            columns=[
                "season",
                "competition",
                "team_id",
                "schedule_adjusted_score",
            ]
        ),
        core,
        parameters,
        domestic,
        xg,
        current,
        evaluation_seasons=set(seasons),
        return_details=True,
    )
    if evaluation.predictions.empty:
        raise ValueError("Production baseline did not produce predictions")
    frame = attach_match_club_ids(evaluation.predictions, identity)
    frame["actual_home_score"] = np.where(
        frame["home_goals"] > frame["away_goals"],
        1.0,
        np.where(frame["home_goals"] == frame["away_goals"], 0.5, 0.0),
    )
    frame["candidate_key"] = BASELINE_KEY
    frame["home_team_effect"] = 0.0
    frame["away_team_effect"] = 0.0
    frame["raw_context_offset"] = 0.0
    frame["applied_context_offset"] = 0.0
    frame["effective_home_advantage"] = np.where(
        frame["is_neutral"].astype(bool), 0.0, core.home_advantage
    )
    frame["context_expected_home_score"] = frame["expected_home_score"]
    return frame.sort_values(["kickoff_utc", "match_id"], kind="stable").reset_index(drop=True)


def build_context_predictions(
    baseline: pd.DataFrame,
    seasons: tuple[str, ...],
    elo_scale: float,
    global_home_advantage: float,
    draw_at_even: float,
    draw_shape: float,
    config: TeamVenueContextConfig,
) -> VenuePredictionSet:
    config.validate()
    season_index = {season: index for index, season in enumerate(seasons)}
    result_frames = []
    profile_frames = []
    for season in seasons:
        current_index = season_index[season]
        current = baseline.loc[baseline["season"].eq(season)].copy()
        history = baseline.loc[
            baseline["season"].map(season_index).between(
                max(0, current_index - config.lookback_seasons),
                current_index - 1,
                inclusive="both",
            )
        ].copy()
        clubs = sorted(
            set(current["home_club_id"].astype(str))
            | set(current["away_club_id"].astype(str))
        )
        profiles = build_season_profiles(
            history,
            clubs,
            season,
            current_index,
            season_index,
            elo_scale,
            config,
        )
        profile_frames.append(profiles)
        home_map = profiles.set_index("club_id")["home_effect_centered"]
        away_map = profiles.set_index("club_id")["away_effect_centered"]
        current["home_team_effect"] = current["home_club_id"].map(home_map).fillna(0.0)
        current["away_team_effect"] = current["away_club_id"].map(away_map).fillna(0.0)
        contextual = [
            contextual_home_expectation(
                float(base),
                float(home_effect),
                float(away_effect),
                global_home_advantage=global_home_advantage,
                elo_scale=elo_scale,
                is_neutral=bool(neutral),
                config=config,
            )
            for base, home_effect, away_effect, neutral in zip(
                current["expected_home_score"],
                current["home_team_effect"],
                current["away_team_effect"],
                current["is_neutral"],
                strict=True,
            )
        ]
        current["raw_context_offset"] = [item.raw_context_offset for item in contextual]
        current["applied_context_offset"] = [item.applied_context_offset for item in contextual]
        current["effective_home_advantage"] = [item.effective_home_advantage for item in contextual]
        current["context_expected_home_score"] = [item.expected_home_score for item in contextual]
        probabilities = np.vstack(
            [
                probability_vector(value, draw_at_even, draw_shape)
                for value in current["context_expected_home_score"]
            ]
        )
        current[["home_probability", "draw_probability", "away_probability"]] = probabilities
        observed = current["actual_class"].to_numpy(int)
        current["brier_1x2"] = np.square(probabilities - np.eye(3)[observed]).sum(axis=1)
        current["log_loss_1x2"] = -np.log(
            np.clip(probabilities[np.arange(len(current)), observed], 1e-15, 1.0)
        )
        current["predicted_class"] = probabilities.argmax(axis=1)
        current["candidate_key"] = config.key
        result_frames.append(current)
    predictions = pd.concat(result_frames, ignore_index=True).sort_values(
        ["kickoff_utc", "match_id"], kind="stable"
    )
    profiles = pd.concat(profile_frames, ignore_index=True)
    validate_context_output(predictions, profiles, config)
    return VenuePredictionSet(predictions.reset_index(drop=True), profiles)


def build_season_profiles(
    history: pd.DataFrame,
    clubs: list[str],
    target_season: str,
    target_index: int,
    season_index: dict[str, int],
    elo_scale: float,
    config: TeamVenueContextConfig,
) -> pd.DataFrame:
    history_seasons = set(history["season"].astype(str))
    unknown_seasons = history_seasons.difference(season_index)
    if unknown_seasons:
        raise ValueError(
            f"Unknown seasons in venue history: {sorted(unknown_seasons)}"
        )
    if any(season_index[season] >= target_index for season in history_seasons):
        raise ValueError(
            f"Venue history for {target_season} contains current or future season"
        )
    rows = []
    for club_id in clubs:
        home = history.loc[history["home_club_id"].eq(club_id)]
        away = history.loc[history["away_club_id"].eq(club_id)]
        home_weights = [
            config.season_decay
            ** (target_index - season_index[str(season)] - 1)
            for season in home["season"]
        ]
        away_weights = [
            config.season_decay
            ** (target_index - season_index[str(season)] - 1)
            for season in away["season"]
        ]
        home_estimate = estimate_team_venue_effect(
            home["expected_home_score"].tolist(),
            home["actual_home_score"].tolist(),
            home_weights,
            elo_scale=elo_scale,
            shrinkage_matches=config.shrinkage_matches,
            max_team_effect=config.max_team_effect,
        )
        away_estimate = estimate_team_venue_effect(
            (1.0 - away["expected_home_score"]).tolist(),
            (1.0 - away["actual_home_score"]).tolist(),
            away_weights,
            elo_scale=elo_scale,
            shrinkage_matches=config.shrinkage_matches,
            max_team_effect=config.max_team_effect,
        )
        rows.append(
            {
                "season": target_season,
                "club_id": club_id,
                "history_max_season": club_history_max_season(home, away, season_index),
                "home_observations": home_estimate.observations,
                "home_effective_matches": home_estimate.effective_matches,
                "home_raw_effect": home_estimate.raw_effect,
                "home_effect_shrunk": home_estimate.shrunk_effect,
                "away_observations": away_estimate.observations,
                "away_effective_matches": away_estimate.effective_matches,
                "away_raw_effect": away_estimate.raw_effect,
                "away_effect_shrunk": away_estimate.shrunk_effect,
            }
        )
    profiles = pd.DataFrame(rows)
    profiles["home_center"] = weighted_center(
        profiles["home_effect_shrunk"], profiles["home_effective_matches"]
    )
    profiles["away_center"] = weighted_center(
        profiles["away_effect_shrunk"], profiles["away_effective_matches"]
    )
    profiles["home_effect_centered"] = np.where(
        profiles["home_observations"].gt(0),
        np.clip(
            profiles["home_effect_shrunk"] - profiles["home_center"],
            -config.max_team_effect,
            config.max_team_effect,
        ),
        0.0,
    )
    profiles["away_effect_centered"] = np.where(
        profiles["away_observations"].gt(0),
        np.clip(
            profiles["away_effect_shrunk"] - profiles["away_center"],
            -config.max_team_effect,
            config.max_team_effect,
        ),
        0.0,
    )
    profiles["candidate_key"] = config.key
    return profiles


def weighted_center(values: pd.Series, weights: pd.Series) -> float:
    mask = weights.gt(0.0)
    if not mask.any():
        return 0.0
    return float(np.average(values[mask], weights=weights[mask]))


def club_history_max_season(
    home: pd.DataFrame,
    away: pd.DataFrame,
    season_index: dict[str, int],
) -> str | object:
    seasons = set(home["season"].astype(str)) | set(away["season"].astype(str))
    if not seasons:
        return pd.NA
    return max(seasons, key=season_index.__getitem__)


def validate_context_output(
    predictions: pd.DataFrame,
    profiles: pd.DataFrame,
    config: TeamVenueContextConfig,
) -> None:
    probabilities = predictions[["home_probability", "draw_probability", "away_probability"]].to_numpy(float)
    if not np.isfinite(probabilities).all() or (probabilities < 0.0).any():
        raise ValueError("Context probabilities must be finite and non-negative")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12):
        raise ValueError("Context probabilities must sum to one")
    non_neutral = predictions.loc[~predictions["is_neutral"].astype(bool)]
    if not non_neutral["effective_home_advantage"].between(
        config.minimum_home_advantage, config.maximum_home_advantage
    ).all():
        raise ValueError("Effective home advantage violated configured bounds")
    if non_neutral["applied_context_offset"].abs().max() > config.max_context_offset + 1e-9:
        raise ValueError("Context offset exceeded configured cap")
    neutral = predictions.loc[predictions["is_neutral"].astype(bool)]
    if not neutral.empty and (
        not np.allclose(neutral["applied_context_offset"], 0.0)
        or not np.allclose(neutral["effective_home_advantage"], 0.0)
    ):
        raise ValueError("Neutral matches must ignore venue context")
    season_order = {
        season: index
        for index, season in enumerate(sorted(predictions["season"].unique()))
    }
    with_history = profiles.loc[profiles["history_max_season"].notna()]
    if not with_history.empty and not all(
        season_order[str(history)] < season_order[str(target)]
        for history, target in zip(
            with_history["history_max_season"], with_history["season"], strict=True
        )
    ):
        raise ValueError("Venue profiles contain current or future season leakage")


def prediction_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    return {
        "matches": len(frame),
        "brier_1x2": float(frame["brier_1x2"].mean()),
        "log_loss_1x2": float(frame["log_loss_1x2"].mean()),
        "accuracy_1x2": float((frame["actual_class"] == frame["predicted_class"]).mean()),
        "mean_abs_context_offset": float(frame["applied_context_offset"].abs().mean()),
        "maximum_abs_context_offset": float(frame["applied_context_offset"].abs().max()),
        "minimum_effective_home_advantage": float(frame["effective_home_advantage"].min()),
        "maximum_effective_home_advantage": float(frame["effective_home_advantage"].max()),
    }


def select_candidate(
    prediction_sets: dict[str, VenuePredictionSet],
    candidates: tuple[VenueCandidate, ...],
    seasons: set[str],
) -> dict[str, object]:
    rows = []
    for candidate in candidates:
        frame = prediction_sets[candidate.key].predictions
        metrics = prediction_metrics(frame.loc[frame["season"].isin(seasons)])
        rows.append(
            {
                "candidate_key": candidate.key,
                "complexity": candidate.complexity,
                **metrics,
            }
        )
    metrics = pd.DataFrame(rows)
    baseline = metrics.loc[metrics["candidate_key"].eq(BASELINE_KEY)].iloc[0]
    safe = metrics.loc[
        (metrics["brier_1x2"] <= baseline["brier_1x2"] + RANK_TOLERANCE)
        & (metrics["log_loss_1x2"] <= baseline["log_loss_1x2"] + RANK_TOLERANCE)
    ]
    return safe.sort_values(
        ["brier_1x2", "log_loss_1x2", "complexity", "candidate_key"],
        ascending=[True, True, True, True],
        kind="stable",
    ).iloc[0].to_dict()


def run_nested_walk_forward(
    prediction_sets: dict[str, VenuePredictionSet],
    candidates: tuple[VenueCandidate, ...],
    folds,
    seasons: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    selection_rows = []
    result_rows = []
    prediction_frames = []
    profile_frames = []
    for fold, (train_seasons, test_season) in enumerate(folds, start=1):
        selected = select_candidate(
            prediction_sets, candidates, set(train_seasons)
        )
        selected_key = str(selected["candidate_key"])
        selection_rows.append(
            {
                "fold": fold,
                "train_seasons": ", ".join(train_seasons),
                "test_season": test_season,
                "selected_candidate_key": selected_key,
                "train_brier_1x2": selected["brier_1x2"],
                "train_log_loss_1x2": selected["log_loss_1x2"],
                "train_mean_abs_context_offset": selected["mean_abs_context_offset"],
            }
        )
        for model, key in ((MODEL_BASELINE, BASELINE_KEY), (MODEL_NESTED, selected_key)):
            test = prediction_sets[key].predictions.loc[
                prediction_sets[key].predictions["season"].eq(test_season)
            ].copy()
            test["fold"] = fold
            test["model"] = model
            test["selected_candidate_key"] = selected_key
            prediction_frames.append(test)
            result_rows.append(
                {
                    "fold": fold,
                    "test_season": test_season,
                    "model": model,
                    "candidate_key": key,
                    **prediction_metrics(test),
                }
            )
        if selected_key != BASELINE_KEY:
            profiles = prediction_sets[selected_key].profiles.loc[
                prediction_sets[selected_key].profiles["season"].eq(test_season)
            ].copy()
            profiles["fold"] = fold
            profile_frames.append(profiles)
        print(f"  fold {fold}/6 -> {test_season}: {selected_key}", flush=True)
    return {
        "fold_selections": pd.DataFrame(selection_rows),
        "fold_results": pd.DataFrame(result_rows),
        "predictions": pd.concat(prediction_frames, ignore_index=True),
        "selected_profiles": pd.concat(profile_frames, ignore_index=True) if profile_frames else pd.DataFrame(
            columns=["fold", "season", "club_id", "candidate_key"]
        ),
    }


def build_candidate_surface(
    prediction_sets: dict[str, VenuePredictionSet],
    candidates: tuple[VenueCandidate, ...],
    folds,
) -> pd.DataFrame:
    test_seasons = {test for _, test in folds}
    rows = []
    for candidate in candidates:
        frame = prediction_sets[candidate.key].predictions
        metrics = prediction_metrics(frame.loc[frame["season"].isin(test_seasons)])
        config = candidate.config
        rows.append(
            {
                "candidate_key": candidate.key,
                "lookback_seasons": 0 if config is None else config.lookback_seasons,
                "season_decay": 0.0 if config is None else config.season_decay,
                "shrinkage_matches": 0.0 if config is None else config.shrinkage_matches,
                "max_team_effect": 0.0 if config is None else config.max_team_effect,
                "max_context_offset": 0.0 if config is None else config.max_context_offset,
                **metrics,
            }
        )
    surface = pd.DataFrame(rows)
    baseline = surface.loc[surface["candidate_key"].eq(BASELINE_KEY)].iloc[0]
    surface["brier_delta_vs_global"] = surface["brier_1x2"] - baseline["brier_1x2"]
    surface["log_loss_delta_vs_global"] = surface["log_loss_1x2"] - baseline["log_loss_1x2"]
    return surface


def build_competition_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, competition), frame in predictions.groupby(["model", "competition"], sort=True):
        rows.append({"model": model, "competition": competition, **prediction_metrics(frame)})
    result = pd.DataFrame(rows)
    baseline = result.loc[result["model"].eq(MODEL_BASELINE)].set_index("competition")
    result["brier_delta_vs_global"] = result.apply(
        lambda row: row["brier_1x2"] - baseline.loc[row["competition"], "brier_1x2"], axis=1
    )
    result["log_loss_delta_vs_global"] = result.apply(
        lambda row: row["log_loss_1x2"] - baseline.loc[row["competition"], "log_loss_1x2"], axis=1
    )
    return result


def prediction_only_state_invariant(predictions: pd.DataFrame) -> bool:
    baseline = predictions.loc[predictions["model"].eq(MODEL_BASELINE)]
    candidate = predictions.loc[predictions["model"].eq(MODEL_NESTED)]
    if len(baseline) != len(candidate) or baseline["match_id"].duplicated().any():
        return False
    numeric_columns = [
        "home_live_pre",
        "away_live_pre",
        "expected_home_score",
        "power_delta",
        "goal_multiplier",
        "xg_performance_adjustment",
        "bonus_applied_after_match",
    ]
    paired = candidate.merge(
        baseline[["match_id", *numeric_columns, "bonus_winner_id"]],
        on="match_id",
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    )
    if len(paired) != len(baseline):
        return False
    for column in numeric_columns:
        if not np.allclose(
            paired[f"{column}_candidate"],
            paired[f"{column}_baseline"],
            atol=1e-12,
            rtol=0.0,
            equal_nan=True,
        ):
            return False
    candidate_bonus = paired["bonus_winner_id_candidate"].fillna("").astype(str)
    baseline_bonus = paired["bonus_winner_id_baseline"].fillna("").astype(str)
    return bool(candidate_bonus.eq(baseline_bonus).all())


def build_profile_summary(
    nested_profiles: pd.DataFrame,
    nested_predictions: pd.DataFrame,
    full_profiles: pd.DataFrame,
    full_predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for evidence, profiles, predictions in (
        ("NESTED_OOS", nested_profiles, nested_predictions),
        ("RETROSPECTIVE_FULL_SELECTION", full_profiles, full_predictions),
    ):
        if profiles.empty:
            continue
        candidate_predictions = predictions.loc[
            predictions.get("model", pd.Series(index=predictions.index, dtype=object))
            .fillna(MODEL_NESTED)
            .eq(MODEL_NESTED)
        ].copy()
        if candidate_predictions.empty:
            candidate_predictions = predictions.copy()
        for season, season_profiles in profiles.groupby("season", sort=True):
            season_predictions = candidate_predictions.loc[
                candidate_predictions["season"].eq(season)
            ]
            non_neutral = season_predictions.loc[
                ~season_predictions["is_neutral"].astype(bool)
            ]
            cap = float(
                season_profiles["candidate_key"]
                .str.extract(r"_ctx([0-9.]+)$", expand=False)
                .astype(float)
                .iloc[0]
            )
            rows.append(
                {
                    "evidence_class": evidence,
                    "season": season,
                    "candidate_key": season_profiles["candidate_key"].iloc[0],
                    "clubs": len(season_profiles),
                    "clubs_without_home_history": int(
                        season_profiles["home_observations"].eq(0).sum()
                    ),
                    "clubs_without_away_history": int(
                        season_profiles["away_observations"].eq(0).sum()
                    ),
                    "median_home_observations": float(
                        season_profiles["home_observations"].median()
                    ),
                    "median_away_observations": float(
                        season_profiles["away_observations"].median()
                    ),
                    "mean_abs_home_effect": float(
                        season_profiles["home_effect_centered"].abs().mean()
                    ),
                    "mean_abs_away_effect": float(
                        season_profiles["away_effect_centered"].abs().mean()
                    ),
                    "matches": len(season_predictions),
                    "mean_abs_context_offset": float(
                        non_neutral["applied_context_offset"].abs().mean()
                    ),
                    "context_cap_hit_rate": float(
                        np.isclose(
                            non_neutral["applied_context_offset"].abs(), cap, atol=1e-9
                        ).mean()
                    ),
                    "minimum_non_neutral_home_advantage": float(
                        non_neutral["effective_home_advantage"].min()
                    ),
                    "maximum_non_neutral_home_advantage": float(
                        non_neutral["effective_home_advantage"].max()
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_fixed_uncertainty(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    candidate_key: str,
    *,
    bootstrap_samples: int,
) -> pd.DataFrame:
    left = baseline.copy()
    left["model"] = MODEL_BASELINE
    right = candidate.copy()
    right["model"] = MODEL_NESTED
    result = build_venue_uncertainty(
        pd.concat([left, right], ignore_index=True),
        baseline_model=MODEL_BASELINE,
        candidate_model=MODEL_NESTED,
        bootstrap_samples=bootstrap_samples,
    )
    result["candidate_key"] = candidate_key
    return result


def build_venue_uncertainty(
    predictions: pd.DataFrame,
    *,
    baseline_model: str,
    candidate_model: str,
    bootstrap_samples: int,
) -> pd.DataFrame:
    baseline = predictions.loc[predictions["model"].eq(baseline_model)]
    candidate = predictions.loc[predictions["model"].eq(candidate_model)]
    rows = []
    for competition_name in ("ALL", "UCL", "UEL", "UECL"):
        left = candidate if competition_name == "ALL" else candidate.loc[
            candidate["competition"].eq(competition_name)
        ]
        right = baseline if competition_name == "ALL" else baseline.loc[
            baseline["competition"].eq(competition_name)
        ]
        paired = left.merge(
            right[["match_id", "brier_1x2", "log_loss_1x2"]],
            on="match_id",
            suffixes=("_candidate", "_baseline"),
            validate="one_to_one",
        )
        if paired.empty:
            raise ValueError(
                f"No paired venue predictions for {competition_name}"
            )
        for metric in ("brier_1x2", "log_loss_1x2"):
            sample = paired[
                [
                    "season",
                    "match_id",
                    "home_team_id",
                    "away_team_id",
                    "kickoff_utc",
                    "tie_id",
                ]
            ].copy()
            sample["loss_difference"] = (
                paired[f"{metric}_candidate"] - paired[f"{metric}_baseline"]
            )
            ci = dependency_robust_loss_difference_ci(
                sample,
                bootstrap_samples=bootstrap_samples,
            )
            ci["competition"] = competition_name
            ci["metric"] = metric
            rows.append(ci)
    return pd.concat(rows, ignore_index=True)


def decide_model(
    folds: pd.DataFrame,
    competition: pd.DataFrame,
    uncertainty: pd.DataFrame,
    fixed_uncertainty: pd.DataFrame,
    full_selection: dict[str, object],
    selected_profiles: pd.DataFrame,
    nested_predictions: pd.DataFrame,
) -> dict[str, object]:
    baseline = folds.loc[folds["model"].eq(MODEL_BASELINE)].set_index("fold")
    nested = folds.loc[folds["model"].eq(MODEL_NESTED)].set_index("fold")
    brier_delta = nested["brier_1x2"] - baseline["brier_1x2"]
    log_delta = nested["log_loss_1x2"] - baseline["log_loss_1x2"]
    fold_wins = int(((brier_delta < 0.0) & (log_delta < 0.0)).sum())
    pooled_brier = float(
        np.average(nested["brier_1x2"], weights=nested["matches"])
        - np.average(baseline["brier_1x2"], weights=baseline["matches"])
    )
    pooled_log = float(
        np.average(nested["log_loss_1x2"], weights=nested["matches"])
        - np.average(baseline["log_loss_1x2"], weights=baseline["matches"])
    )
    nested_comp = competition.loc[competition["model"].eq(MODEL_NESTED)]
    envelopes = uncertainty.loc[uncertainty["method"].eq("conservative_envelope")]
    all_envelopes = envelopes.loc[envelopes["competition"].eq("ALL")]
    reliable_both = bool(
        len(all_envelopes) == 2 and all_envelopes["reliable_improvement"].all()
    )
    no_comp_harm = bool(
        (nested_comp["brier_delta_vs_global"] <= 0.0).all()
        and (nested_comp["log_loss_delta_vs_global"] <= 0.0).all()
        and not envelopes["reliable_harm"].any()
    )
    profiles_valid = bool(
        selected_profiles.empty
        or (
            selected_profiles["home_effect_centered"].abs().max() <= 150.0 + 1e-9
            and selected_profiles["away_effect_centered"].abs().max() <= 150.0 + 1e-9
            and selected_profiles.loc[
                selected_profiles["history_max_season"].isna(),
                ["home_observations", "away_observations"],
            ]
            .eq(0)
            .all()
            .all()
        )
    )
    state_invariant = prediction_only_state_invariant(nested_predictions)
    gates = {
        "fold_wins_at_least_4_of_6": fold_wins >= 4,
        "pooled_brier_improves": pooled_brier < 0.0,
        "pooled_log_loss_improves": pooled_log < 0.0,
        "pooled_dependency_ci_reliably_improves": reliable_both,
        "no_competition_harm": no_comp_harm,
        "full_history_selects_team_venue": str(full_selection["candidate_key"]) != BASELINE_KEY,
        "prediction_only_state_invariant": state_invariant,
        "profile_guardrails": profiles_valid,
    }
    promoted = all(gates.values())
    shadow_signal = all(
        value
        for key, value in gates.items()
        if key != "pooled_dependency_ci_reliably_improves"
    )
    fixed_envelope = fixed_uncertainty.loc[
        fixed_uncertainty["method"].eq("conservative_envelope")
        & fixed_uncertainty["competition"].eq("ALL")
    ]
    return {
        "decision": (
            "PROMOTE_CANDIDATE"
            if promoted
            else "KEEP_SHADOW_CANDIDATE"
            if shadow_signal
            else "KEEP_GLOBAL_HOME_ADVANTAGE"
        ),
        "production_changed": False,
        "layer_mode": "PREDICTION_ONLY",
        "full_history_candidate_key": full_selection["candidate_key"],
        "full_history_candidate_metrics": full_selection,
        "nested_fold_wins": f"{fold_wins}/6",
        "pooled_brier_delta_vs_global": pooled_brier,
        "pooled_log_loss_delta_vs_global": pooled_log,
        "retrospective_best_fixed_candidate_key": fixed_uncertainty["candidate_key"].iloc[0],
        "retrospective_best_fixed_candidate_pooled_envelope": fixed_envelope.to_dict(orient="records"),
        "gates": gates,
        "interpretation": "Prediction-only test; AO ratings and production contract are unchanged.",
    }


def write_report(
    path: Path,
    contract: dict[str, object],
    seasons: tuple[str, ...],
    selections: pd.DataFrame,
    folds: pd.DataFrame,
    surface: pd.DataFrame,
    competition: pd.DataFrame,
    uncertainty: pd.DataFrame,
    fixed_uncertainty: pd.DataFrame,
    profile_summary: pd.DataFrame,
    decision: dict[str, object],
    tie_audit_rows: int,
) -> None:
    best = surface.sort_values(["brier_1x2", "log_loss_1x2", "candidate_key"], kind="stable").head(10)
    nested_comp = competition.loc[competition["model"].eq(MODEL_NESTED)]
    envelopes = uncertainty.loc[uncertainty["method"].eq("conservative_envelope")]
    fixed_envelopes = fixed_uncertainty.loc[
        fixed_uncertainty["method"].eq("conservative_envelope")
    ]
    nested_profile_summary = profile_summary.loc[
        profile_summary["evidence_class"].eq("NESTED_OOS")
    ]
    lines = [
        "# Takım Bazlı Home/Away Venue Context Backtesti",
        "",
        "## Sözleşme",
        "",
        f"- Dönem: `{seasons[0]}`–`{seasons[-1]}`; altı expanding outer fold.",
        "- Katman prediction-only shadow olarak test edildi; Power Elo güncellemesi değişmedi.",
        "- Home effect yalnız geçmiş iç saha residual'ından, away strength yalnız geçmiş deplasman residual'ından üretildi.",
        "- Her sezon profili yalnız önceki tamamlanmış sezonları kullandı; aynı sezon sonucu profile girmedi.",
        "- Takım etkileri global H çevresinde ağırlıklı merkezlendi; az verili takım sıfıra shrink edildi.",
        "- Aday grid: window `3/5`, decay `0.75/1`, shrinkage `6/10/15/20`, cap `25/35/50/75/100/150`.",
        f"- Global H `{contract['dynamic_core']['home_advantage']:.6f}`, Scale `{contract['dynamic_core']['elo_scale']:.6f}`.",
        "",
        "## Fold Seçimleri",
        "",
        markdown_table(selections),
        "",
        "## Fold Sonuçları",
        "",
        markdown_table(folds, float_digits=9),
        "",
        "## Turnuva Sonuçları",
        "",
        markdown_table(nested_comp[["competition", "brier_delta_vs_global", "log_loss_delta_vs_global", "mean_abs_context_offset"]], float_digits=9),
        "",
        "## Nested Dependency Belirsizliği",
        "",
        markdown_table(envelopes[["competition", "metric", "mean_difference", "ci_95_lower", "ci_95_upper", "reliable_improvement", "reliable_harm"]], float_digits=9),
        "",
        "## Retrospektif En İyi Sabit Aday Belirsizliği",
        "",
        "Bu tablo aynı OOS yüzeyinde aday seçildikten sonra hesaplanan bir duyarlılık analizidir; terfi kanıtı değildir.",
        "",
        markdown_table(fixed_envelopes[["candidate_key", "competition", "metric", "mean_difference", "ci_95_lower", "ci_95_upper", "reliable_improvement", "reliable_harm"]], float_digits=9),
        "",
        "## Profil Kapsamı ve Sınır Kullanımı",
        "",
        markdown_table(
            nested_profile_summary[
                [
                    "season",
                    "candidate_key",
                    "clubs_without_home_history",
                    "clubs_without_away_history",
                    "median_home_observations",
                    "median_away_observations",
                    "mean_abs_context_offset",
                    "context_cap_hit_rate",
                ]
            ],
            float_digits=6,
        ),
        "",
        "Yüksek context-cap kullanım oranı, seçilen etkinin sıkça güvenlik sınırı tarafından belirlendiğini gösterir ve production terfisi öncesinde ayrıca izlenmelidir.",
        "",
        "## Aday Yüzeyi: İlk 10",
        "",
        markdown_table(best[["candidate_key", "brier_1x2", "log_loss_1x2", "accuracy_1x2", "mean_abs_context_offset", "maximum_abs_context_offset", "brier_delta_vs_global", "log_loss_delta_vs_global"]], float_digits=9),
        "",
        "## Karar",
        "",
        f"**{decision['decision']}**",
        "",
        f"Tam geçmiş seçimi: `{decision['full_history_candidate_key']}`.",
        f"Nested fold win: `{decision['nested_fold_wins']}`.",
        f"Pooled Brier farkı: `{decision['pooled_brier_delta_vs_global']:+.9f}`.",
        f"Pooled log-loss farkı: `{decision['pooled_log_loss_delta_vs_global']:+.9f}`.",
        "",
        *[f"- `{key}`: `{value}`" for key, value in decision["gates"].items()],
        "",
        "Production JSON'u ve AO Live Elo state'i değiştirilmedi.",
        "",
        "## Audit",
        "",
        f"- Tie kronoloji audit satırı: `{tie_audit_rows}`.",
        "- Nötr saha context offset'i sıfırdır.",
        "- Tüm olasılıklar sonlu, negatif olmayan ve toplamı birdir.",
        "- Takım profillerinde current/future season leakage yasaktır.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
