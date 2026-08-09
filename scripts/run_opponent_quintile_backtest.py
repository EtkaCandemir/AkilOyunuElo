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
from ao_elo.evaluation import dependency_robust_loss_difference_ci  # noqa: E402
from ao_elo.opponent_quintile_context import (  # noqa: E402
    CAUSAL_ONLINE,
    CATEGORICAL_1X2_OUTCOME,
    DYNAMIC_QUINTILES,
    EXPECTED_SCORE_OUTCOME,
    FIXED_THRESHOLDS,
    QUINTILES,
    SEASON_LOCKED,
    OpponentQuintileContextConfig,
    QuintileProfile,
    assign_fixed_threshold_membership,
    assign_quintiles,
    contextual_matchup_expectation,
    estimate_quintile_profile,
    profile_to_record,
)
from scripts.run_controlled_goal_progression_backtest import prepare_controlled_data  # noqa: E402
from scripts.run_dynamic_core_calibration import expanding_folds  # noqa: E402
from scripts.run_final_robustness import load_team_season_identity  # noqa: E402
from scripts.run_stage_weighted_progression_backtest import (  # noqa: E402
    DOMESTIC_ADJUSTMENTS,
    DYNAMIC_MANIFEST,
    EVENTS_PATH,
    PRODUCTION_CONTRACT,
    STATIC_DATA_ROOT,
    XG_DATA,
    load_domestic_adjustments,
    load_xg_map,
    probability_vector,
    validate_production_contract,
)
from scripts.run_team_venue_context_backtest import build_production_baseline  # noqa: E402
from scripts.run_v2_achievement_reserve_calibration import load_reserve_data  # noqa: E402
from scripts.run_v2_evaluation_upgrade import read_events  # noqa: E402


OUTPUT_ROOT = ROOT / "output" / "opponent_quintile_context_backtest_2018_2026"
FOTMOB_XG_PATH = ROOT / "data" / "xg_2025_26" / "uefa_2025_26_matches_with_xg.csv"
SPORTSDB_STATS_PATH = ROOT / "data" / "thesportsdb_2025_26" / "event_stats_wide.csv"
BASELINE_KEY = "BASELINE"
MODEL_BASELINE = "BASELINE"
MODEL_CONTEXT = "OPPONENT_QUINTILE_CONTEXT"
EPSILON = 1e-12
ACTIVE_OUTCOME_MODEL = EXPECTED_SCORE_OUTCOME
ACTIVE_DRAW_AT_EVEN = 0.24
ACTIVE_DRAW_SHAPE = 1.0


@dataclass(frozen=True)
class Candidate:
    key: str
    config: OpponentQuintileContextConfig | None

    @property
    def complexity(self) -> int:
        return int(self.config is not None)


@dataclass(frozen=True)
class RawBaseConfig:
    band_mode: str
    profile_update_mode: str
    lookback_seasons: int
    season_decay: float
    outcome_model: str

    @property
    def key(self) -> str:
        mode = "dynamic" if self.band_mode == DYNAMIC_QUINTILES else "fixed"
        update = "online" if self.profile_update_mode == CAUSAL_ONLINE else "locked"
        outcome = "categorical_" if self.outcome_model == CATEGORICAL_1X2_OUTCOME else ""
        return f"{outcome}{mode}_{update}_w{self.lookback_seasons}_d{self.season_decay:g}"


def config_fingerprint(candidate_key: str) -> str:
    return hashlib.sha256(candidate_key.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    global ACTIVE_DRAW_AT_EVEN, ACTIVE_DRAW_SHAPE, ACTIVE_OUTCOME_MODEL, QUINTILES

    parser = argparse.ArgumentParser(
        description="Nested walk-forward backtest for dynamic opponent-band profiles"
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    parser.add_argument("--band-count", type=int, choices=(3, 5), default=5)
    parser.add_argument(
        "--outcome-model",
        choices=(EXPECTED_SCORE_OUTCOME, CATEGORICAL_1X2_OUTCOME),
        default=EXPECTED_SCORE_OUTCOME,
    )
    parser.add_argument("--skip-explanation-appendix", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples must be at least 100")
    QUINTILES = tuple(range(1, args.band_count + 1))
    ACTIVE_OUTCOME_MODEL = args.outcome_model

    baseline, datasets, core, parameters, domestic = load_production_baseline()
    ACTIVE_DRAW_AT_EVEN = float(parameters["draw_at_even"])
    ACTIVE_DRAW_SHAPE = float(parameters["draw_shape"])
    seasons = tuple(sorted(baseline["season"].astype(str).unique()))
    folds = expanding_folds(seasons)
    if len(folds) != 6:
        raise ValueError(f"Expected six outer folds, found {len(folds)}")
    assignments, thresholds = build_quintile_assignments(baseline, datasets, domestic)
    baseline = baseline.merge(
        assignments[
            [
                "match_id",
                "snapshot_id",
                "home_quintile_dynamic",
                "away_quintile_dynamic",
                "home_quintile_fixed",
                "away_quintile_fixed",
            ]
        ],
        on="match_id",
        validate="one_to_one",
    )
    perspectives = build_team_perspectives(baseline)
    candidates = candidate_grid()
    raw_bases = raw_base_configs(candidates)
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(output / "match_quintile_assignments.csv", index=False)
    thresholds.to_csv(output / "quintile_thresholds.csv", index=False)

    raw_features: dict[str, pd.DataFrame] = {}
    for index, raw in enumerate(raw_bases, start=1):
        print(f"Raw profile base {index}/{len(raw_bases)}: {raw.key}", flush=True)
        raw_features[raw.key] = build_raw_profile_features(
            baseline, perspectives, seasons, core.elo_scale, raw
        )

    metrics_by_candidate: dict[str, pd.DataFrame] = {
        BASELINE_KEY: metrics_by_season(baseline)
    }
    for index, candidate in enumerate(candidates[1:], start=1):
        assert candidate.config is not None
        features = raw_features[raw_key(candidate.config)]
        predictions = apply_candidate(features, candidate.config, parameters, core.elo_scale)
        metrics_by_candidate[candidate.key] = metrics_by_season(predictions)
        if index % 16 == 0 or index == len(candidates) - 1:
            print(f"Candidate metrics {index}/{len(candidates) - 1}", flush=True)

    nested = nested_selection(metrics_by_candidate, candidates, folds)
    selected_keys = set(nested["fold_selections"]["selected_candidate_key"].astype(str))
    full_selected = select_candidate(metrics_by_candidate, candidates, set(seasons))
    selected_keys.add(str(full_selected["candidate_key"]))
    selected_predictions = {BASELINE_KEY: baseline.copy()}
    selected_profiles: dict[str, pd.DataFrame] = {}
    for key in sorted(selected_keys - {BASELINE_KEY}):
        candidate = candidate_by_key(candidates, key)
        assert candidate.config is not None
        features = raw_features[raw_key(candidate.config)]
        selected_predictions[key] = apply_candidate(
            features, candidate.config, parameters, core.elo_scale
        )
        selected_profiles[key] = selected_profiles_from_features(
            features, candidate.config
        )

    unseen, fold_results, profile_rows = build_nested_outputs(
        baseline,
        selected_predictions,
        selected_profiles,
        nested,
        folds,
    )
    surface = candidate_surface(metrics_by_candidate, candidates, folds)
    competition = comparison_summary(unseen, ["competition"])
    quintile_summary = comparison_summary(
        unseen, ["home_quintile_dynamic", "away_quintile_dynamic"]
    )
    matrix = build_quintile_matrix(baseline)
    uncertainty = build_uncertainty(unseen, args.bootstrap_samples)
    persistence = build_profile_persistence(
        profile_rows, unseen, core.elo_scale, args.bootstrap_samples
    )
    decision = decide_model(
        nested["fold_selections"],
        fold_results,
        competition,
        quintile_summary,
        uncertainty,
        persistence,
        full_selected,
        candidates,
        unseen,
    )

    surface.to_csv(output / "candidate_surface.csv", index=False)
    nested["fold_selections"].to_csv(output / "fold_selections.csv", index=False)
    fold_results.to_csv(output / "fold_results.csv", index=False)
    unseen.to_csv(output / "unseen_predictions.csv", index=False)
    competition.to_csv(output / "competition_summary.csv", index=False)
    quintile_summary.to_csv(output / "quintile_summary.csv", index=False)
    matrix.to_csv(output / "quintile_matrix.csv", index=False)
    profile_rows.to_csv(output / "team_quintile_profiles.csv", index=False)
    persistence.to_csv(output / "profile_persistence.csv", index=False)
    uncertainty.to_csv(output / "dependency_uncertainty.csv", index=False)
    if not args.skip_explanation_appendix:
        build_2025_explanation_appendix(baseline, output)
    (output / "selected_candidate.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "backtest_report.md").write_text(
        build_report(decision, surface, nested["fold_selections"], fold_results, competition, persistence),
        encoding="utf-8",
    )
    validate_outputs(baseline, assignments, thresholds, unseen, profile_rows)
    print(f"Decision: {decision['decision']}")
    print(f"Full-history selection: {decision['full_history_candidate_key']}")
    print(f"Output: {output}")


def load_production_baseline():
    contract = json.loads(PRODUCTION_CONTRACT.read_text(encoding="utf-8"))
    core, parameters = validate_production_contract(contract)
    dynamic = json.loads(DYNAMIC_MANIFEST.read_text(encoding="utf-8"))
    static_config = AOEuropeanEloConfig(**dynamic["static_config"])
    static_config.validate()
    events = read_events(EVENTS_PATH)
    reserve, _ = load_reserve_data(STATIC_DATA_ROOT, EVENTS_PATH, static_config)
    datasets = prepare_controlled_data(reserve, events)
    identity = load_team_season_identity()
    domestic = load_domestic_adjustments(DOMESTIC_ADJUSTMENTS, datasets)
    xg = load_xg_map(XG_DATA, datasets)
    baseline = build_production_baseline(
        datasets,
        core,
        parameters,
        domestic,
        xg,
        identity,
        tuple(data.season for data in datasets),
    )
    if len(baseline) != 6340 or baseline["match_id"].duplicated().any():
        raise ValueError("Production baseline must contain 6,340 unique matches")
    return baseline, datasets, core, parameters, domestic


def candidate_grid() -> tuple[Candidate, ...]:
    candidates = [Candidate(BASELINE_KEY, None)]
    for band_mode in (FIXED_THRESHOLDS, DYNAMIC_QUINTILES):
        for update in (SEASON_LOCKED, CAUSAL_ONLINE):
            for window in (3, 5):
                for decay in (0.75, 1.0):
                    for shrinkage in (6.0, 10.0, 15.0, 20.0):
                        for cap in (25.0, 50.0, 75.0):
                            config = OpponentQuintileContextConfig(
                                band_mode,
                                update,
                                window,
                                decay,
                                shrinkage,
                                cap,
                                len(QUINTILES),
                                ACTIVE_OUTCOME_MODEL,
                            )
                            config.validate()
                            candidates.append(Candidate(config.key, config))
    keys = [candidate.key for candidate in candidates]
    if len(candidates) != 193 or len(keys) != len(set(keys)):
        raise ValueError("Expected baseline plus 192 quintile candidates")
    return tuple(candidates)


def raw_base_configs(candidates: tuple[Candidate, ...]) -> tuple[RawBaseConfig, ...]:
    values = {
        RawBaseConfig(
            candidate.config.band_mode,
            candidate.config.profile_update_mode,
            candidate.config.lookback_seasons,
            candidate.config.season_decay,
            candidate.config.outcome_model,
        )
        for candidate in candidates
        if candidate.config is not None
    }
    return tuple(sorted(values, key=lambda value: value.key))


def raw_key(config: OpponentQuintileContextConfig) -> str:
    return RawBaseConfig(
        config.band_mode,
        config.profile_update_mode,
        config.lookback_seasons,
        config.season_decay,
        config.outcome_model,
    ).key


def build_quintile_assignments(
    baseline: pd.DataFrame,
    datasets,
    domestic: dict[tuple[str, int], float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay the visible AO Live state for every season and timestamp batch."""

    baseline = baseline.sort_values(["season", "kickoff_utc", "match_id"], kind="stable")
    assignments: list[dict[str, object]] = []
    thresholds: list[dict[str, object]] = []
    for data in datasets:
        season = data.season
        season_rows = baseline.loc[baseline["season"].eq(season)].copy()
        core = data.reserve.goal.carry.core
        active_ids = np.asarray(core.active_team_ids, dtype=int)
        clubs = club_id_map(season_rows, active_ids)
        state = {
            int(team_id): float(domestic[(season, int(team_id))])
            for team_id in active_ids
        }
        fixed_threshold_values: tuple[float, ...] | None = None
        for snapshot_number, (kickoff, batch) in enumerate(
            season_rows.groupby("kickoff_utc", sort=True), start=1
        ):
            team_ids = active_ids.tolist()
            club_ids = [clubs[team_id] for team_id in team_ids]
            ratings = [state[team_id] for team_id in team_ids]
            dynamic = assign_quintiles(
                team_ids, club_ids, ratings, band_count=len(QUINTILES)
            )
            if fixed_threshold_values is None:
                fixed_threshold_values = dynamic.thresholds
            fixed = assign_fixed_threshold_membership(
                team_ids, club_ids, ratings, fixed_threshold_values
            )
            dynamic_map = dynamic.assignments.set_index("team_id")
            fixed_map = fixed.set_index("team_id")
            snapshot_id = f"{season.replace('/', '-')}-{snapshot_number:03d}"
            dynamic_counts = dynamic.assignments["quintile"].value_counts().sort_index()
            threshold_audit = {
                **{
                    f"dynamic_threshold_q{lower}_q{upper}": dynamic.thresholds[lower - 1]
                    for lower, upper in zip(QUINTILES[:-1], QUINTILES[1:], strict=True)
                },
                **{
                    f"fixed_threshold_q{lower}_q{upper}": fixed_threshold_values[lower - 1]
                    for lower, upper in zip(QUINTILES[:-1], QUINTILES[1:], strict=True)
                },
                **{
                    f"dynamic_boundary_tie_q{lower}_q{upper}": dynamic.boundary_ties[lower - 1]
                    for lower, upper in zip(QUINTILES[:-1], QUINTILES[1:], strict=True)
                },
            }
            thresholds.append(
                {
                    "season": season,
                    "snapshot_id": snapshot_id,
                    "kickoff_utc": kickoff,
                    "team_universe": len(team_ids),
                    **threshold_audit,
                    **{f"dynamic_q{quintile}_teams": int(dynamic_counts[quintile]) for quintile in QUINTILES},
                    "batch_matches": len(batch),
                }
            )
            for row in batch.itertuples(index=False):
                home_id, away_id = int(row.home_team_id), int(row.away_team_id)
                _assert_state_matches_baseline(state, home_id, float(row.home_live_pre), row.match_id)
                _assert_state_matches_baseline(state, away_id, float(row.away_live_pre), row.match_id)
                assignments.append(
                    {
                        "match_id": row.match_id,
                        "season": season,
                        "kickoff_utc": kickoff,
                        "snapshot_id": snapshot_id,
                        "home_team_id": home_id,
                        "away_team_id": away_id,
                        "home_club_id": str(row.home_club_id),
                        "away_club_id": str(row.away_club_id),
                        "home_live_pre": float(row.home_live_pre),
                        "away_live_pre": float(row.away_live_pre),
                        "home_quintile_dynamic": int(dynamic_map.loc[home_id, "quintile"]),
                        "away_quintile_dynamic": int(dynamic_map.loc[away_id, "quintile"]),
                        "home_quintile_fixed": int(fixed_map.loc[home_id, "quintile"]),
                        "away_quintile_fixed": int(fixed_map.loc[away_id, "quintile"]),
                        **threshold_audit,
                    }
                )
            # Every prediction in this batch is frozen before any result enters the state.
            for row in batch.sort_values("match_id", kind="stable").itertuples(index=False):
                home_id, away_id = int(row.home_team_id), int(row.away_team_id)
                state[home_id] += float(row.power_delta)
                state[away_id] -= float(row.power_delta)
                winner = int(row.bonus_winner_id)
                if winner >= 0:
                    if winner not in state:
                        raise ValueError(f"{row.match_id}: bonus winner is outside season universe")
                    state[winner] += float(row.bonus_applied_after_match)
    assignment_frame = pd.DataFrame(assignments).sort_values(
        ["kickoff_utc", "match_id"], kind="stable"
    )
    threshold_frame = pd.DataFrame(thresholds).sort_values(
        ["season", "kickoff_utc"], kind="stable"
    )
    if len(assignment_frame) != len(baseline):
        raise ValueError("Every production baseline match needs one quintile assignment")
    return assignment_frame.reset_index(drop=True), threshold_frame.reset_index(drop=True)


def club_id_map(rows: pd.DataFrame, active_ids: np.ndarray) -> dict[int, str]:
    long = pd.concat(
        [
            rows[["home_team_id", "home_club_id"]].rename(
                columns={"home_team_id": "team_id", "home_club_id": "club_id"}
            ),
            rows[["away_team_id", "away_club_id"]].rename(
                columns={"away_team_id": "team_id", "away_club_id": "club_id"}
            ),
        ],
        ignore_index=True,
    ).drop_duplicates()
    if long.duplicated("team_id").any():
        raise ValueError("One season team_id maps to multiple permanent club IDs")
    mapping = {int(row.team_id): str(row.club_id) for row in long.itertuples(index=False)}
    missing = sorted(set(active_ids).difference(mapping))
    if missing:
        raise ValueError(f"Season universe has unmapped team IDs: {missing[:5]}")
    return mapping


def _assert_state_matches_baseline(
    state: dict[int, float], team_id: int, reported: float, match_id: str
) -> None:
    if not math.isclose(state[team_id], reported, abs_tol=1e-8, rel_tol=0.0):
        raise ValueError(
            f"{match_id}: reconstructed AO Live state diverged for team {team_id}: "
            f"{state[team_id]:.12f} != {reported:.12f}"
        )


def build_team_perspectives(baseline: pd.DataFrame) -> pd.DataFrame:
    home = baseline[
        [
            "match_id", "season", "kickoff_utc", "snapshot_id", "home_club_id",
            "away_club_id", "expected_home_score", "actual_home_score",
            "home_quintile_dynamic", "away_quintile_dynamic",
            "home_quintile_fixed", "away_quintile_fixed",
        ]
    ].rename(
        columns={
            "home_club_id": "club_id",
            "away_club_id": "opponent_club_id",
            "expected_home_score": "expected_score",
            "actual_home_score": "actual_score",
            "away_quintile_dynamic": "opponent_quintile_dynamic",
            "away_quintile_fixed": "opponent_quintile_fixed",
        }
    )
    away = baseline[
        [
            "match_id", "season", "kickoff_utc", "snapshot_id", "away_club_id",
            "home_club_id", "expected_home_score", "actual_home_score",
            "home_quintile_dynamic", "away_quintile_dynamic",
            "home_quintile_fixed", "away_quintile_fixed",
        ]
    ].rename(
        columns={
            "away_club_id": "club_id",
            "home_club_id": "opponent_club_id",
            "home_quintile_dynamic": "opponent_quintile_dynamic",
            "home_quintile_fixed": "opponent_quintile_fixed",
        }
    )
    away["expected_score"] = 1.0 - away.pop("expected_home_score")
    away["actual_score"] = 1.0 - away.pop("actual_home_score")
    home = home.drop(columns=["home_quintile_dynamic", "home_quintile_fixed"])
    away = away.drop(columns=["away_quintile_dynamic", "away_quintile_fixed"])
    result = pd.concat([home, away], ignore_index=True)
    result["residual"] = result["actual_score"] - result["expected_score"]
    return result.sort_values(["kickoff_utc", "match_id", "club_id"], kind="stable").reset_index(drop=True)


def build_raw_profile_features(
    baseline: pd.DataFrame,
    perspectives: pd.DataFrame,
    seasons: tuple[str, ...],
    elo_scale: float,
    config: RawBaseConfig,
) -> pd.DataFrame:
    season_index = {season: index for index, season in enumerate(seasons)}
    quintile_column = (
        "opponent_quintile_dynamic"
        if config.band_mode == DYNAMIC_QUINTILES
        else "opponent_quintile_fixed"
    )
    result = baseline.copy()
    profile_records: list[dict[str, object]] = []
    for season in seasons:
        target_index = season_index[season]
        current = result.loc[result["season"].eq(season)].copy()
        prior = prior_history(
            perspectives,
            target_index,
            season_index,
            config.lookback_seasons,
            config.season_decay,
        )
        if config.profile_update_mode == SEASON_LOCKED:
            features, records = season_locked_features(
                current, prior, quintile_column, elo_scale, season, config.key
            )
        else:
            features, records = causal_online_features(
                current, perspectives.loc[perspectives["season"].eq(season)].copy(),
                prior,
                quintile_column,
                elo_scale,
                season,
                config.key,
            )
        result.loc[features.index, features.columns] = features
        profile_records.extend(records)
    required = raw_feature_columns("home") + raw_feature_columns("away")
    if result[required].isna().any().any():
        raise ValueError(f"{config.key}: raw quintile profile feature is missing")
    result["raw_profile_base_key"] = config.key
    result.attrs["profile_records"] = pd.DataFrame(profile_records)
    return result


def prior_history(
    perspectives: pd.DataFrame,
    target_index: int,
    season_index: dict[str, int],
    window: int,
    decay: float,
) -> pd.DataFrame:
    selected = perspectives.loc[
        perspectives["season"].map(season_index).between(
            max(0, target_index - window), target_index - 1, inclusive="both"
        )
    ].copy()
    if selected.empty:
        selected["weight"] = pd.Series(dtype=float)
        return selected
    selected["weight"] = [
        decay ** (target_index - season_index[str(season)] - 1)
        for season in selected["season"]
    ]
    return selected


def season_locked_features(
    current: pd.DataFrame,
    history: pd.DataFrame,
    quintile_column: str,
    elo_scale: float,
    season: str,
    base_key: str,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    clubs = sorted(
        set(current["home_club_id"].astype(str)) | set(current["away_club_id"].astype(str))
    )
    profiles = {
        club_id: raw_profile(
            club_id, history.loc[history["club_id"].eq(club_id)], quintile_column, elo_scale
        )
        for club_id in clubs
    }
    features = current.loc[:, []].copy()
    for side, club_column in (("home", "home_club_id"), ("away", "away_club_id")):
        values = [profiles[str(club_id)] for club_id in current[club_column]]
        append_raw_profile_columns(features, side, values)
    records = [
        {
            "season": season,
            "snapshot_id": "SEASON_LOCKED",
            "raw_profile_base_key": base_key,
            **profile_to_record(profile),
        }
        for profile in profiles.values()
    ]
    return features, records


def causal_online_features(
    current: pd.DataFrame,
    current_perspectives: pd.DataFrame,
    prior: pd.DataFrame,
    quintile_column: str,
    elo_scale: float,
    season: str,
    base_key: str,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    current = current.sort_values(["kickoff_utc", "match_id"], kind="stable")
    perspective_by_match = {
        match_id: frame.copy()
        for match_id, frame in current_perspectives.groupby("match_id", sort=False)
    }
    histories = {
        str(club_id): frame.copy()
        for club_id, frame in prior.groupby("club_id", sort=False)
    }
    cache: dict[str, QuintileProfile] = {}
    feature_rows: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    for snapshot_id, batch in current.groupby("snapshot_id", sort=False):
        clubs = sorted(
            set(batch["home_club_id"].astype(str)) | set(batch["away_club_id"].astype(str))
        )
        profiles = {}
        for club_id in clubs:
            if club_id not in cache:
                profiles[club_id] = raw_profile(
                    club_id,
                    histories.get(club_id, pd.DataFrame(columns=prior.columns)),
                    quintile_column,
                    elo_scale,
                )
                cache[club_id] = profiles[club_id]
            else:
                profiles[club_id] = cache[club_id]
            records.append(
                {
                    "season": season,
                    "snapshot_id": snapshot_id,
                    "raw_profile_base_key": base_key,
                    **profile_to_record(profiles[club_id]),
                }
            )
        for row in batch.itertuples():
            feature = {"_index": row.Index}
            feature.update(raw_profile_record("home", profiles[str(row.home_club_id)]))
            feature.update(raw_profile_record("away", profiles[str(row.away_club_id)]))
            feature_rows.append(feature)
        for match_id in batch["match_id"]:
            additions = perspective_by_match[str(match_id)].copy()
            additions["weight"] = 1.0
            for club_id, club_rows in additions.groupby("club_id", sort=False):
                club_key = str(club_id)
                histories[club_key] = pd.concat(
                    [histories.get(club_key, pd.DataFrame(columns=prior.columns)), club_rows],
                    ignore_index=True,
                )
                cache.pop(club_key, None)
    features = pd.DataFrame(feature_rows).set_index("_index").sort_index()
    return features, records


def raw_profile(
    club_id: str,
    history: pd.DataFrame,
    quintile_column: str,
    elo_scale: float,
) -> QuintileProfile:
    selected = history.rename(columns={quintile_column: "opponent_quintile"})
    return estimate_quintile_profile(
        club_id,
        selected[["opponent_quintile", "expected_score", "actual_score", "weight"]],
        elo_scale=elo_scale,
        shrinkage_matches=0.0,
        effect_cap=2000.0,
        band_count=len(QUINTILES),
        outcome_model=ACTIVE_OUTCOME_MODEL,
        draw_at_even=ACTIVE_DRAW_AT_EVEN,
        draw_shape=ACTIVE_DRAW_SHAPE,
    )


def raw_feature_columns(side: str) -> list[str]:
    columns = []
    for quintile in QUINTILES:
        columns.extend(
            [
                f"{side}_raw_specific_q{quintile}",
                f"{side}_effective_matches_q{quintile}",
                f"{side}_observations_q{quintile}",
            ]
        )
    return columns


def raw_profile_record(side: str, profile: QuintileProfile) -> dict[str, object]:
    record: dict[str, object] = {}
    for index, quintile in enumerate(QUINTILES):
        record[f"{side}_raw_specific_q{quintile}"] = profile.raw_specific_effects[index]
        record[f"{side}_effective_matches_q{quintile}"] = profile.effective_matches[index]
        record[f"{side}_observations_q{quintile}"] = profile.observations[index]
    return record


def append_raw_profile_columns(
    frame: pd.DataFrame, side: str, profiles: list[QuintileProfile]
) -> None:
    for quintile in QUINTILES:
        index = quintile - 1
        frame[f"{side}_raw_specific_q{quintile}"] = [
            profile.raw_specific_effects[index] for profile in profiles
        ]
        frame[f"{side}_effective_matches_q{quintile}"] = [
            profile.effective_matches[index] for profile in profiles
        ]
        frame[f"{side}_observations_q{quintile}"] = [
            profile.observations[index] for profile in profiles
        ]


def apply_candidate(
    features: pd.DataFrame,
    config: OpponentQuintileContextConfig,
    parameters: dict[str, float | int],
    elo_scale: float,
) -> pd.DataFrame:
    config.validate()
    result = features.copy()
    home_effects = materialize_effects(result, "home", config)
    away_effects = materialize_effects(result, "away", config)
    for quintile in QUINTILES:
        result[f"home_effect_q{quintile}"] = home_effects[:, quintile - 1]
        result[f"away_effect_q{quintile}"] = away_effects[:, quintile - 1]
    home_band = result[
        "away_quintile_dynamic"
        if config.band_mode == DYNAMIC_QUINTILES
        else "away_quintile_fixed"
    ].to_numpy(int)
    away_band = result[
        "home_quintile_dynamic"
        if config.band_mode == DYNAMIC_QUINTILES
        else "home_quintile_fixed"
    ].to_numpy(int)
    index = np.arange(len(result))
    home_selected = home_effects[index, home_band - 1]
    away_selected = away_effects[index, away_band - 1]
    raw_offset = home_selected - away_selected
    applied_offset = np.clip(
        raw_offset,
        -config.effect_and_context_cap,
        config.effect_and_context_cap,
    )
    base = result["expected_home_score"].to_numpy(float)
    logit = np.log(base / (1.0 - base))
    context_expected = logistic_array(
        logit + math.log(10.0) * applied_offset / elo_scale
    )
    probabilities = np.vstack(
        [
            probability_vector(
                float(value),
                float(parameters["draw_at_even"]),
                float(parameters["draw_shape"]),
            )
            for value in context_expected
        ]
    )
    observed = result["actual_class"].to_numpy(int)
    result["home_team_selected_effect"] = home_selected
    result["away_team_selected_effect"] = away_selected
    result["raw_matchup_offset"] = raw_offset
    result["applied_matchup_offset"] = applied_offset
    result["context_cap_hit"] = np.isclose(
        np.abs(applied_offset), config.effect_and_context_cap, atol=1e-12
    )
    result["context_expected_home_score"] = context_expected
    result[["home_probability", "draw_probability", "away_probability"]] = probabilities
    result["predicted_class"] = probabilities.argmax(axis=1)
    result["brier_1x2"] = np.square(probabilities - np.eye(3)[observed]).sum(axis=1)
    result["log_loss_1x2"] = -np.log(
        np.clip(probabilities[np.arange(len(result)), observed], 1e-15, 1.0)
    )
    result["candidate_key"] = config.key
    result["config_fingerprint"] = config_fingerprint(config.key)
    result["band_mode"] = config.band_mode
    result["profile_update_mode"] = config.profile_update_mode
    result["lookback_seasons"] = config.lookback_seasons
    result["season_decay"] = config.season_decay
    result["shrinkage_matches"] = config.shrinkage_matches
    result["effect_and_context_cap"] = config.effect_and_context_cap
    result["outcome_model"] = config.outcome_model
    return result


def materialize_effects(
    frame: pd.DataFrame,
    side: str,
    config: OpponentQuintileContextConfig,
) -> np.ndarray:
    raw = frame[
        [f"{side}_raw_specific_q{quintile}" for quintile in QUINTILES]
    ].to_numpy(float)
    effective = frame[
        [f"{side}_effective_matches_q{quintile}" for quintile in QUINTILES]
    ].to_numpy(float)
    observed = effective > 0.0
    reliability = np.divide(
        effective,
        effective + config.shrinkage_matches,
        out=np.zeros_like(effective),
        where=effective > 0.0,
    )
    preliminary = np.clip(
        raw * reliability,
        -config.effect_and_context_cap,
        config.effect_and_context_cap,
    )
    denominator = (effective * observed).sum(axis=1)
    center = np.divide(
        (preliminary * effective * observed).sum(axis=1),
        denominator,
        out=np.zeros(len(frame), dtype=float),
        where=denominator > 0.0,
    )
    effects = np.where(
        observed,
        np.clip(
            preliminary - center[:, None],
            -config.effect_and_context_cap,
            config.effect_and_context_cap,
        ),
        0.0,
    )
    effects[observed.sum(axis=1) < 2] = 0.0
    return effects


def logistic_array(values: np.ndarray) -> np.ndarray:
    positive = values >= 0.0
    result = np.empty_like(values, dtype=float)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


def metrics_by_season(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, frame in predictions.groupby("season", sort=True):
        rows.append({"season": season, **prediction_metrics(frame)})
    return pd.DataFrame(rows)


def prediction_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    return {
        "matches": int(len(frame)),
        "brier_1x2": float(frame["brier_1x2"].mean()),
        "log_loss_1x2": float(frame["log_loss_1x2"].mean()),
        "accuracy_1x2": float(
            frame["actual_class"].eq(frame["predicted_class"]).mean()
        ),
        "mean_abs_context_offset": float(
            frame.get("applied_matchup_offset", pd.Series(0.0, index=frame.index))
            .abs()
            .mean()
        ),
        "context_cap_hit_rate": float(
            frame.get("context_cap_hit", pd.Series(False, index=frame.index)).mean()
        ),
    }


def select_candidate(
    metrics: dict[str, pd.DataFrame],
    candidates: tuple[Candidate, ...],
    seasons: set[str],
) -> dict[str, object]:
    rows = []
    for candidate in candidates:
        values = metrics[candidate.key].loc[metrics[candidate.key]["season"].isin(seasons)]
        rows.append(
            {
                "candidate_key": candidate.key,
                "config_fingerprint": config_fingerprint(candidate.key),
                "complexity": candidate.complexity,
                "brier_1x2": float(np.average(values["brier_1x2"], weights=values["matches"])),
                "log_loss_1x2": float(np.average(values["log_loss_1x2"], weights=values["matches"])),
                "accuracy_1x2": float(np.average(values["accuracy_1x2"], weights=values["matches"])),
                "mean_abs_context_offset": float(
                    np.average(values["mean_abs_context_offset"], weights=values["matches"])
                ),
                "context_cap_hit_rate": float(
                    np.average(values["context_cap_hit_rate"], weights=values["matches"])
                ),
            }
        )
    table = pd.DataFrame(rows)
    baseline = table.loc[table["candidate_key"].eq(BASELINE_KEY)].iloc[0]
    safe = table.loc[
        table["brier_1x2"].le(float(baseline["brier_1x2"]) + EPSILON)
        & table["log_loss_1x2"].le(float(baseline["log_loss_1x2"]) + EPSILON)
    ]
    return safe.sort_values(
        ["brier_1x2", "log_loss_1x2", "complexity", "candidate_key"],
        kind="stable",
    ).iloc[0].to_dict()


def nested_selection(metrics, candidates, folds) -> dict[str, pd.DataFrame]:
    rows = []
    for fold, (train, test) in enumerate(folds, start=1):
        selected = select_candidate(metrics, candidates, set(train))
        rows.append(
            {
                "fold": fold,
                "train_seasons": ", ".join(train),
                "test_season": test,
                "selected_candidate_key": selected["candidate_key"],
                "selected_config_fingerprint": config_fingerprint(
                    str(selected["candidate_key"])
                ),
                "train_brier_1x2": selected["brier_1x2"],
                "train_log_loss_1x2": selected["log_loss_1x2"],
                "train_mean_abs_context_offset": selected["mean_abs_context_offset"],
                "train_context_cap_hit_rate": selected["context_cap_hit_rate"],
            }
        )
        print(f"Fold {fold}/6 -> {test}: {selected['candidate_key']}", flush=True)
    return {"fold_selections": pd.DataFrame(rows)}


def candidate_by_key(candidates: tuple[Candidate, ...], key: str) -> Candidate:
    for candidate in candidates:
        if candidate.key == key:
            return candidate
    raise ValueError(f"Unknown candidate key: {key}")


def selected_profiles_from_features(
    features: pd.DataFrame,
    config: OpponentQuintileContextConfig,
) -> pd.DataFrame:
    profiles = features.attrs.get("profile_records", pd.DataFrame()).copy()
    if profiles.empty:
        return profiles
    raw = profiles[
        [f"raw_specific_effect_q{quintile}" for quintile in QUINTILES]
    ].to_numpy(float)
    effective = profiles[
        [f"effective_matches_q{quintile}" for quintile in QUINTILES]
    ].to_numpy(float)
    observed = effective > 0.0
    reliability = np.divide(
        effective,
        effective + config.shrinkage_matches,
        out=np.zeros_like(effective),
        where=effective > 0.0,
    )
    preliminary = np.clip(
        raw * reliability,
        -config.effect_and_context_cap,
        config.effect_and_context_cap,
    )
    denominator = (effective * observed).sum(axis=1)
    center = np.divide(
        (preliminary * effective * observed).sum(axis=1),
        denominator,
        out=np.zeros(len(profiles), dtype=float),
        where=denominator > 0.0,
    )
    effects = np.where(
        observed,
        np.clip(
            preliminary - center[:, None],
            -config.effect_and_context_cap,
            config.effect_and_context_cap,
        ),
        0.0,
    )
    effects[observed.sum(axis=1) < 2] = 0.0
    for quintile in QUINTILES:
        profiles[f"effect_q{quintile}"] = effects[:, quintile - 1]
        profiles[f"reliability_q{quintile}"] = reliability[:, quintile - 1]
    profiles["candidate_key"] = config.key
    profiles["band_mode"] = config.band_mode
    profiles["profile_update_mode"] = config.profile_update_mode
    profiles["lookback_seasons"] = config.lookback_seasons
    profiles["season_decay"] = config.season_decay
    profiles["shrinkage_matches"] = config.shrinkage_matches
    profiles["effect_and_context_cap"] = config.effect_and_context_cap
    profiles["outcome_model"] = config.outcome_model
    return profiles.drop_duplicates(
        ["season", "snapshot_id", "club_id", "candidate_key"], keep="last"
    ).reset_index(drop=True)


def build_nested_outputs(
    baseline: pd.DataFrame,
    prediction_sets: dict[str, pd.DataFrame],
    profiles: dict[str, pd.DataFrame],
    nested: dict[str, pd.DataFrame],
    folds,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    metrics = []
    profile_rows = []
    for selection in nested["fold_selections"].itertuples(index=False):
        test_season = str(selection.test_season)
        selected_key = str(selection.selected_candidate_key)
        for model, frame in (
            (MODEL_BASELINE, baseline),
            (MODEL_CONTEXT, prediction_sets[selected_key]),
        ):
            current = frame.loc[frame["season"].eq(test_season)].copy()
            current["fold"] = int(selection.fold)
            current["model"] = model
            current["selected_candidate_key"] = selected_key
            current["config_fingerprint"] = config_fingerprint(
                BASELINE_KEY if model == MODEL_BASELINE else selected_key
            )
            if model == MODEL_BASELINE:
                current["outcome_model"] = "BASELINE"
                current["raw_matchup_offset"] = 0.0
                current["applied_matchup_offset"] = 0.0
                current["context_cap_hit"] = False
                current["context_expected_home_score"] = current["expected_home_score"]
                current["home_team_selected_effect"] = 0.0
                current["away_team_selected_effect"] = 0.0
                for quintile in QUINTILES:
                    current[f"home_effect_q{quintile}"] = 0.0
                    current[f"away_effect_q{quintile}"] = 0.0
            rows.append(current)
            metrics.append(
                {
                    "fold": int(selection.fold),
                    "test_season": test_season,
                    "model": model,
                    "candidate_key": BASELINE_KEY if model == MODEL_BASELINE else selected_key,
                    **prediction_metrics(current),
                }
            )
        if selected_key != BASELINE_KEY:
            profile = profiles[selected_key].loc[
                profiles[selected_key]["season"].eq(test_season)
            ].copy()
            profile["fold"] = int(selection.fold)
            profile_rows.append(profile)
    return (
        pd.concat(rows, ignore_index=True),
        pd.DataFrame(metrics),
        pd.concat(profile_rows, ignore_index=True)
        if profile_rows
        else pd.DataFrame(columns=["fold", "season", "club_id"]),
    )


def candidate_surface(metrics, candidates, folds) -> pd.DataFrame:
    test_seasons = {test for _, test in folds}
    rows = []
    baseline_values = metrics[BASELINE_KEY].loc[
        metrics[BASELINE_KEY]["season"].isin(test_seasons)
    ]
    baseline_brier = float(
        np.average(baseline_values["brier_1x2"], weights=baseline_values["matches"])
    )
    baseline_log = float(
        np.average(baseline_values["log_loss_1x2"], weights=baseline_values["matches"])
    )
    for candidate in candidates:
        values = metrics[candidate.key].loc[metrics[candidate.key]["season"].isin(test_seasons)]
        config = candidate.config
        brier = float(np.average(values["brier_1x2"], weights=values["matches"]))
        log_loss = float(np.average(values["log_loss_1x2"], weights=values["matches"]))
        rows.append(
            {
                "candidate_key": candidate.key,
                "config_fingerprint": config_fingerprint(candidate.key),
                "outcome_model": (
                    EXPECTED_SCORE_OUTCOME if config is None else config.outcome_model
                ),
                "band_mode": "BASELINE" if config is None else config.band_mode,
                "profile_update_mode": "BASELINE" if config is None else config.profile_update_mode,
                "lookback_seasons": 0 if config is None else config.lookback_seasons,
                "season_decay": 0.0 if config is None else config.season_decay,
                "shrinkage_matches": 0.0 if config is None else config.shrinkage_matches,
                "effect_and_context_cap": 0.0 if config is None else config.effect_and_context_cap,
                "brier_1x2": brier,
                "log_loss_1x2": log_loss,
                "accuracy_1x2": float(np.average(values["accuracy_1x2"], weights=values["matches"])),
                "mean_abs_context_offset": float(np.average(values["mean_abs_context_offset"], weights=values["matches"])),
                "context_cap_hit_rate": float(np.average(values["context_cap_hit_rate"], weights=values["matches"])),
                "brier_delta_vs_baseline": brier - baseline_brier,
                "log_loss_delta_vs_baseline": log_loss - baseline_log,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["brier_1x2", "log_loss_1x2", "candidate_key"], kind="stable"
    ).reset_index(drop=True)


def comparison_summary(unseen: pd.DataFrame, dimensions: list[str]) -> pd.DataFrame:
    rows = []
    baseline = unseen.loc[unseen["model"].eq(MODEL_BASELINE)].copy()
    candidate = unseen.loc[unseen["model"].eq(MODEL_CONTEXT)].copy()
    merge_columns = ["match_id", *dimensions]
    paired = candidate.merge(
        baseline[["match_id", "brier_1x2", "log_loss_1x2"]],
        on="match_id",
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    )
    for values, frame in paired.groupby(dimensions, sort=True, dropna=False):
        values = (values,) if not isinstance(values, tuple) else values
        rows.append(
            {
                **dict(zip(dimensions, values, strict=True)),
                "matches": len(frame),
                "brier_candidate": float(frame["brier_1x2_candidate"].mean()),
                "log_loss_candidate": float(frame["log_loss_1x2_candidate"].mean()),
                "brier_delta_vs_baseline": float(
                    (frame["brier_1x2_candidate"] - frame["brier_1x2_baseline"]).mean()
                ),
                "log_loss_delta_vs_baseline": float(
                    (frame["log_loss_1x2_candidate"] - frame["log_loss_1x2_baseline"]).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_quintile_matrix(baseline: pd.DataFrame) -> pd.DataFrame:
    home = pd.DataFrame(
        {
            "own_quintile": baseline["home_quintile_dynamic"],
            "opponent_quintile": baseline["away_quintile_dynamic"],
            "expected_score": baseline["expected_home_score"],
            "actual_score": baseline["actual_home_score"],
            "competition": baseline["competition"],
        }
    )
    away = pd.DataFrame(
        {
            "own_quintile": baseline["away_quintile_dynamic"],
            "opponent_quintile": baseline["home_quintile_dynamic"],
            "expected_score": 1.0 - baseline["expected_home_score"],
            "actual_score": 1.0 - baseline["actual_home_score"],
            "competition": baseline["competition"],
        }
    )
    long = pd.concat([home, away], ignore_index=True)
    long["residual"] = long["actual_score"] - long["expected_score"]
    rows = []
    for (own, opponent), frame in long.groupby(
        ["own_quintile", "opponent_quintile"], sort=True
    ):
        rows.append(
            {
                "own_quintile": int(own),
                "opponent_quintile": int(opponent),
                "team_match_observations": len(frame),
                "expected_score": float(frame["expected_score"].mean()),
                "actual_score": float(frame["actual_score"].mean()),
                "residual": float(frame["residual"].mean()),
                "wins": int(frame["actual_score"].eq(1.0).sum()),
                "draws": int(frame["actual_score"].eq(0.5).sum()),
                "losses": int(frame["actual_score"].eq(0.0).sum()),
            }
        )
    return pd.DataFrame(rows)


def build_uncertainty(unseen: pd.DataFrame, bootstrap_samples: int) -> pd.DataFrame:
    baseline = unseen.loc[unseen["model"].eq(MODEL_BASELINE)]
    candidate = unseen.loc[unseen["model"].eq(MODEL_CONTEXT)]
    paired = candidate.merge(
        baseline[["match_id", "brier_1x2", "log_loss_1x2"]],
        on="match_id",
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    )
    segments: list[tuple[str, str, str, pd.DataFrame]] = [
        ("all", "ALL", "ALL", paired),
    ]
    segments.extend(
        ("competition", str(competition), str(competition), frame)
        for competition, frame in paired.groupby("competition", sort=True)
    )
    segments.extend(
        (
            "dynamic_quintile_pair",
            f"Q{int(home_band)}_vs_Q{int(away_band)}",
            f"Q{int(home_band)}_vs_Q{int(away_band)}",
            frame,
        )
        for (home_band, away_band), frame in paired.groupby(
            ["home_quintile_dynamic", "away_quintile_dynamic"], sort=True
        )
    )
    rows = []
    for segment_type, segment, competition, frame in segments:
        for metric in ("brier_1x2", "log_loss_1x2"):
            audit = frame[
                ["season", "match_id", "home_team_id", "away_team_id", "kickoff_utc", "tie_id"]
            ].copy()
            audit["loss_difference"] = (
                frame[f"{metric}_candidate"] - frame[f"{metric}_baseline"]
            )
            uncertainty = dependency_robust_loss_difference_ci(
                audit,
                bootstrap_samples=bootstrap_samples,
            )
            uncertainty["competition"] = competition
            uncertainty["segment_type"] = segment_type
            uncertainty["segment"] = segment
            uncertainty["metric"] = metric
            rows.append(uncertainty)
    return pd.concat(rows, ignore_index=True)


def build_profile_persistence(
    profiles: pd.DataFrame,
    unseen: pd.DataFrame,
    elo_scale: float,
    bootstrap_samples: int,
) -> pd.DataFrame:
    candidate = unseen.loc[unseen["model"].eq(MODEL_CONTEXT)].copy()
    if candidate.empty:
        return pd.DataFrame()
    home = pd.DataFrame(
        {
            "fold": candidate["fold"],
            "season": candidate["season"],
            "club_id": candidate["home_club_id"],
            "effect": candidate["home_team_selected_effect"],
            "residual": candidate["actual_home_score"] - candidate["expected_home_score"],
        }
    )
    away = pd.DataFrame(
        {
            "fold": candidate["fold"],
            "season": candidate["season"],
            "club_id": candidate["away_club_id"],
            "effect": candidate["away_team_selected_effect"],
            "residual": (1.0 - candidate["actual_home_score"])
            - (1.0 - candidate["expected_home_score"]),
        }
    )
    values = pd.concat([home, away], ignore_index=True)
    values = values.loc[values["effect"].abs().gt(1e-12)].copy()
    rows = []
    for fold, frame in values.groupby("fold", sort=True):
        rows.append(persistence_row(str(fold), frame, bootstrap_samples))
    rows.append(persistence_row("ALL", values, bootstrap_samples))
    return pd.DataFrame(rows)


def persistence_row(label: str, frame: pd.DataFrame, bootstrap_samples: int) -> dict[str, object]:
    if len(frame) < 3 or frame["effect"].nunique() < 2 or frame["residual"].nunique() < 2:
        spearman = math.nan
    else:
        spearman = float(frame["effect"].corr(frame["residual"], method="spearman"))
    sign_accuracy = float(
        ((frame["effect"] * frame["residual"]) > 0.0).mean()
    ) if len(frame) else math.nan
    lower, upper = bootstrap_spearman(frame, bootstrap_samples)
    return {
        "fold": label,
        "team_match_observations": len(frame),
        "clubs": int(frame["club_id"].nunique()),
        "spearman": spearman,
        "sign_accuracy": sign_accuracy,
        "team_season_ci_95_lower": lower,
        "team_season_ci_95_upper": upper,
        "reliable_negative": bool(np.isfinite(upper) and upper < 0.0),
    }


def bootstrap_spearman(frame: pd.DataFrame, samples: int) -> tuple[float, float]:
    if len(frame) < 3:
        return math.nan, math.nan
    groups = [group for _, group in frame.groupby(["season", "club_id"], sort=False)]
    if len(groups) < 2:
        return math.nan, math.nan
    generator = np.random.default_rng(20260807)
    values = []
    for _ in range(samples):
        sample = pd.concat(
            [groups[index] for index in generator.integers(0, len(groups), len(groups))],
            ignore_index=True,
        )
        if sample["effect"].nunique() > 1 and sample["residual"].nunique() > 1:
            values.append(float(sample["effect"].corr(sample["residual"], method="spearman")))
    if not values:
        return math.nan, math.nan
    return tuple(float(value) for value in np.quantile(values, (0.025, 0.975)))


def decide_model(
    selections: pd.DataFrame,
    fold_results: pd.DataFrame,
    competition: pd.DataFrame,
    quintiles: pd.DataFrame,
    uncertainty: pd.DataFrame,
    persistence: pd.DataFrame,
    full_selected: dict[str, object],
    candidates: tuple[Candidate, ...],
    unseen: pd.DataFrame,
) -> dict[str, object]:
    pivot = fold_results.pivot(index="fold", columns="model", values=["brier_1x2", "log_loss_1x2"])
    brier_wins = int(
        (pivot["brier_1x2"][MODEL_CONTEXT] <= pivot["brier_1x2"][MODEL_BASELINE]).sum()
    )
    log_wins = int(
        (pivot["log_loss_1x2"][MODEL_CONTEXT] <= pivot["log_loss_1x2"][MODEL_BASELINE]).sum()
    )
    all_uncertainty = uncertainty.loc[uncertainty["segment_type"].eq("all")]
    pooled_brier = all_uncertainty.loc[all_uncertainty["metric"].eq("brier_1x2")]
    pooled_log = all_uncertainty.loc[all_uncertainty["metric"].eq("log_loss_1x2")]
    envelope_brier = pooled_brier.loc[pooled_brier["method"].eq("conservative_envelope")].iloc[0]
    envelope_log = pooled_log.loc[pooled_log["method"].eq("conservative_envelope")].iloc[0]
    no_segment_harm = not bool(
        uncertainty.loc[uncertainty["segment_type"].ne("all"), "reliable_harm"].any()
    )
    persistence_all = persistence.loc[persistence["fold"].eq("ALL")]
    persistence_row = persistence_all.iloc[0] if len(persistence_all) else None
    non_negative_persistence = int(
        persistence.loc[persistence["fold"].ne("ALL"), "spearman"].ge(0.0).sum()
    )
    persistence_ok = bool(
        persistence_row is not None
        and float(persistence_row["spearman"]) > 0.0
        and non_negative_persistence >= 4
        and not bool(persistence_row["reliable_negative"])
    )
    selected_candidate = candidate_by_key(candidates, str(full_selected["candidate_key"]))
    dynamic_wins_full_history = bool(
        selected_candidate.config is not None
        and selected_candidate.config.band_mode == DYNAMIC_QUINTILES
    )
    cap_ok = float(full_selected["context_cap_hit_rate"]) <= 0.25
    state_invariant = prediction_state_invariant(unseen)
    gates = {
        "brier_fold_wins_at_least_4_of_6": brier_wins >= 4,
        "log_loss_fold_wins_at_least_4_of_6": log_wins >= 4,
        "pooled_brier_not_worse": float(envelope_brier["mean_difference"]) <= EPSILON,
        "pooled_log_loss_not_worse": float(envelope_log["mean_difference"]) <= EPSILON,
        "no_competition_or_strength_band_reliable_harm": no_segment_harm,
        "profile_persistence": persistence_ok,
        "full_history_selects_dynamic": dynamic_wins_full_history,
        "context_cap_hit_rate_at_most_25pct": cap_ok,
        "prediction_only_state_invariant": state_invariant,
    }
    if all(gates.values()):
        outcome = "PROMOTE_DYNAMIC_SHADOW_CANDIDATE"
    elif any(
        (
            bool(envelope_brier["reliable_harm"]),
            bool(envelope_log["reliable_harm"]),
            not state_invariant,
        )
    ):
        outcome = "REJECT"
    else:
        outcome = "KEEP_DIAGNOSTIC"
    return {
        "decision": outcome,
        "production_changed": False,
        "layer_mode": "PREDICTION_ONLY_SHADOW",
        "band_count": len(QUINTILES),
        "outcome_model": ACTIVE_OUTCOME_MODEL,
        "full_history_candidate_key": full_selected["candidate_key"],
        "full_history_config_fingerprint": config_fingerprint(
            str(full_selected["candidate_key"])
        ),
        "full_history_candidate_metrics": full_selected,
        "nested_brier_fold_wins": f"{brier_wins}/6",
        "nested_log_loss_fold_wins": f"{log_wins}/6",
        "pooled_brier_delta_vs_baseline": float(envelope_brier["mean_difference"]),
        "pooled_log_loss_delta_vs_baseline": float(envelope_log["mean_difference"]),
        "profile_persistence_spearman": (
            math.nan if persistence_row is None else float(persistence_row["spearman"])
        ),
        "gates": gates,
    }


def prediction_state_invariant(unseen: pd.DataFrame) -> bool:
    baseline = unseen.loc[unseen["model"].eq(MODEL_BASELINE)]
    candidate = unseen.loc[unseen["model"].eq(MODEL_CONTEXT)]
    if len(baseline) != len(candidate) or baseline["match_id"].duplicated().any():
        return False
    columns = [
        "home_live_pre", "away_live_pre", "expected_home_score", "power_delta",
        "goal_multiplier", "xg_performance_adjustment", "bonus_applied_after_match",
    ]
    paired = candidate.merge(
        baseline[["match_id", *columns, "bonus_winner_id"]],
        on="match_id",
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    )
    for column in columns:
        if not np.allclose(
            paired[f"{column}_candidate"],
            paired[f"{column}_baseline"],
            atol=1e-12,
            rtol=0.0,
            equal_nan=True,
        ):
            return False
    return bool(
        paired["bonus_winner_id_candidate"].fillna(-1).eq(
            paired["bonus_winner_id_baseline"].fillna(-1)
        ).all()
    )


def build_2025_explanation_appendix(baseline: pd.DataFrame, output: Path) -> None:
    xg = pd.read_csv(FOTMOB_XG_PATH)
    stats = pd.read_csv(SPORTSDB_STATS_PATH)
    season = baseline.loc[baseline["season"].eq("2025/26")].copy()
    data = season.merge(
        xg[["match_id", "xg_home", "xg_away", "xg_analysis_eligible"]],
        on="match_id",
        how="left",
        validate="one_to_one",
    ).merge(
        stats[[
            "match_id", "home_total_shots", "away_total_shots", "home_ball_possession",
            "away_ball_possession", "stats_covered",
        ]],
        on="match_id",
        how="left",
        validate="one_to_one",
    )
    data["xg_difference"] = data["xg_home"] - data["xg_away"]
    rows = []
    for bands, frame in data.groupby(
        ["home_quintile_dynamic", "away_quintile_dynamic"], sort=True
    ):
        rows.append(
            {
                "home_quintile": int(bands[0]),
                "away_quintile": int(bands[1]),
                "matches": len(frame),
                "xg_eligible_matches": int(frame["xg_analysis_eligible"].fillna(False).sum()),
                "mean_home_xg_difference": float(frame["xg_difference"].mean()),
                "stats_covered_matches": int(frame["stats_covered"].fillna(False).sum()),
                "mean_home_shot_difference": float(
                    (frame["home_total_shots"] - frame["away_total_shots"]).mean()
                ),
                "mean_home_possession_difference": float(
                    (frame["home_ball_possession"] - frame["away_ball_possession"]).mean()
                ),
            }
        )
    pd.DataFrame(rows).to_csv(output / "xg_stats_explanation_2025_26.csv", index=False)


def validate_outputs(
    baseline: pd.DataFrame,
    assignments: pd.DataFrame,
    thresholds: pd.DataFrame,
    unseen: pd.DataFrame,
    profiles: pd.DataFrame,
) -> None:
    if len(assignments) != len(baseline) or assignments["match_id"].duplicated().any():
        raise ValueError("Every baseline match requires exactly one quintile assignment")
    counts = thresholds[[f"dynamic_q{quintile}_teams" for quintile in QUINTILES]]
    if (counts.max(axis=1) - counts.min(axis=1) > 1).any():
        raise ValueError("Dynamic quintile group sizes differ by more than one team")
    probabilities = unseen[["home_probability", "draw_probability", "away_probability"]].to_numpy(float)
    if not np.isfinite(probabilities).all() or (probabilities < 0.0).any():
        raise ValueError("Quintile probabilities must be finite and non-negative")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12):
        raise ValueError("Quintile probabilities must sum to one")
    candidate = unseen.loc[unseen["model"].eq(MODEL_CONTEXT)]
    if not candidate.empty and candidate["applied_matchup_offset"].abs().gt(
        candidate["effect_and_context_cap"] + 1e-9
    ).any():
        raise ValueError("Applied matchup offset exceeded candidate cap")
    if not profiles.empty:
        effect_columns = [f"effect_q{quintile}" for quintile in QUINTILES]
        values = profiles[effect_columns].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError("Profile effects must be finite")


def build_report(
    decision: dict[str, object],
    surface: pd.DataFrame,
    selections: pd.DataFrame,
    fold_results: pd.DataFrame,
    competition: pd.DataFrame,
    persistence: pd.DataFrame,
) -> str:
    def csv_table(frame: pd.DataFrame) -> str:
        return "```csv\n" + frame.to_csv(index=False).rstrip() + "\n```"

    top = surface.head(12)
    band_count = int(decision.get("band_count", len(QUINTILES)))
    band_label = "T1-T3" if band_count == 3 else "Q1-Q5"
    return f"""# Dinamik {band_label} Rakip Profili Backtesti

## Karar

**{decision['decision']}**

- Full-history secimi: `{decision['full_history_candidate_key']}`
- Dinamik band sayisi: `{band_count}`
- Profil outcome modeli: `{decision.get('outcome_model', EXPECTED_SCORE_OUTCOME)}`
- Nested Brier fold win: `{decision['nested_brier_fold_wins']}`
- Nested log-loss fold win: `{decision['nested_log_loss_fold_wins']}`
- Pooled Brier farki: `{decision['pooled_brier_delta_vs_baseline']:+.9f}`
- Pooled log-loss farki: `{decision['pooled_log_loss_delta_vs_baseline']:+.9f}`
- Profil devamlıligi Spearman: `{decision['profile_persistence_spearman']:.6f}`

Production AO Live Elo, Power Delta, gol farki, xG ve progression bonusu bu
calismada degistirilmemistir. Katman yalniz prediction-only shadow olarak test edilmistir.

## Fold Secimleri

{csv_table(selections)}

## Fold Sonuclari

{csv_table(fold_results)}

## Turnuva Ozeti

{csv_table(competition)}

## Profil Devamliligi

{csv_table(persistence)}

## OOS Aday Yuzeyi Ilk 12

{csv_table(top)}
"""


if __name__ == "__main__":
    main()
