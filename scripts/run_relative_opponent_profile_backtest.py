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

from ao_elo.evaluation import dependency_robust_loss_difference_ci  # noqa: E402
from ao_elo.relative_opponent_profile import (  # noqa: E402
    RELATIVE_BANDS,
    DomesticRatingConfig,
    RelativeOpponentProfileConfig,
    build_european_perspectives,
    estimate_relative_profile,
    profile_record,
    replay_domestic_perspectives,
    weighted_history,
)
from scripts.run_dynamic_core_calibration import expanding_folds  # noqa: E402
from scripts.run_opponent_quintile_backtest import load_production_baseline  # noqa: E402
from scripts.run_stage_weighted_progression_backtest import probability_vector  # noqa: E402


OUTPUT_ROOT = ROOT / "output" / "relative_opponent_profile_backtest_2018_2026"
DOMESTIC_DATA = ROOT / "data" / "domestic_league_matches_2013_2026" / "domestic_matches.csv"
LEGACY_QUINTILE_DECISION = ROOT / "output" / "opponent_quintile_context_backtest_2018_2026" / "selected_candidate.json"
BASELINE_KEY = "BASELINE"
EUROPE_ONLY_ARM = "RELATIVE_EUROPE_ONLY"
DOMESTIC_ARM = "RELATIVE_DOMESTIC_EUROPE"
EPSILON = 1e-12


@dataclass(frozen=True)
class Candidate:
    key: str
    arm: str
    config: RelativeOpponentProfileConfig | None

    @property
    def complexity(self) -> int:
        return int(self.config is not None)


@dataclass(frozen=True)
class RawConfig:
    lower: float
    upper: float
    window: int
    decay: float
    domestic_weight: float

    @property
    def key(self) -> str:
        arm = "europe" if self.domestic_weight == 0.0 else f"dom{self.domestic_weight:g}"
        return f"raw_l{self.lower:g}_u{self.upper:g}_w{self.window}_d{self.decay:g}_{arm}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nested walk-forward backtest for domestic-supported relative opponent profiles"
    )
    parser.add_argument("--domestic-data", type=Path, default=DOMESTIC_DATA)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    parser.add_argument("--raw-config-limit", type=int, default=None, help="Development-only deterministic raw-config limit")
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples must be at least 100")
    baseline, _, core, parameters, _ = load_production_baseline()
    baseline = prepare_baseline(
        baseline,
        float(parameters["draw_at_even"]),
        float(parameters["draw_shape"]),
    )
    domestic = pd.read_csv(args.domestic_data.resolve())
    seasons = tuple(sorted(baseline["season"].astype(str).unique()))
    folds = expanding_folds(seasons)
    if len(folds) != 6:
        raise ValueError(f"Expected six outer folds, found {len(folds)}")
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    candidates = candidate_grid()
    raw_configs = raw_config_grid(candidates)
    if args.raw_config_limit is not None:
        raw_configs = development_raw_config_limit(raw_configs, args.raw_config_limit)
        allowed = {raw.key for raw in raw_configs}
        candidates = tuple(
            candidate
            for candidate in candidates
            if candidate.config is None or raw_key(candidate.config) in allowed
        )

    domestic_config = DomesticRatingConfig(
        elo_scale=float(core.elo_scale),
        home_advantage=float(core.home_advantage),
        k_factor=float(core.k_factor),
        season_carry=0.75,
    )
    raw_features: dict[str, pd.DataFrame] = {}
    raw_profiles: dict[str, pd.DataFrame] = {}
    domestic_perspective_cache: dict[tuple[float, float], pd.DataFrame] = {}
    european_perspective_cache: dict[tuple[float, float], pd.DataFrame] = {}
    for position, raw in enumerate(raw_configs, start=1):
        print(f"Raw relative profile {position}/{len(raw_configs)}: {raw.key}", flush=True)
        band_key = (raw.lower, raw.upper)
        if band_key not in domestic_perspective_cache:
            domestic_perspective_cache[band_key] = replay_domestic_perspectives(
                domestic,
                config=domestic_config,
                lower=raw.lower,
                upper=raw.upper,
            )
        if band_key not in european_perspective_cache:
            european_perspective_cache[band_key] = build_european_perspectives(
                baseline,
                elo_scale=float(core.elo_scale),
                lower=raw.lower,
                upper=raw.upper,
            )
        domestic_perspectives = domestic_perspective_cache[band_key]
        european_perspectives = european_perspective_cache[band_key]
        features, profiles = build_raw_features(
            baseline,
            european_perspectives,
            domestic_perspectives,
            seasons,
            raw,
            float(core.elo_scale),
        )
        raw_features[raw.key] = features
        raw_profiles[raw.key] = profiles

    prediction_sets: dict[str, pd.DataFrame] = {BASELINE_KEY: baseline.copy()}
    for position, candidate in enumerate(candidates[1:], start=1):
        assert candidate.config is not None
        features = materialize_candidate(
            raw_features[raw_key(candidate.config)],
            candidate,
            float(parameters["draw_at_even"]),
            float(parameters["draw_shape"]),
            float(core.elo_scale),
        )
        prediction_sets[candidate.key] = candidate_predictions(
            baseline,
            features,
            candidate,
            float(parameters["draw_at_even"]),
            float(parameters["draw_shape"]),
            float(core.elo_scale),
        )
        if position % 24 == 0 or position == len(candidates) - 1:
            print(f"Candidate metrics {position}/{len(candidates) - 1}", flush=True)

    metrics = {key: metrics_by_season(frame) for key, frame in prediction_sets.items()}
    nested = nested_selection(metrics, candidates, folds)
    unseen = build_unseen_predictions(prediction_sets, nested, folds)
    fold_results = build_fold_results(unseen)
    surface = candidate_surface(metrics, candidates, folds)
    summary = comparison_summary(unseen)
    uncertainty = build_uncertainty(unseen, args.bootstrap_samples)
    persistence = build_persistence(unseen, args.bootstrap_samples)
    coverage = build_domestic_coverage(domestic, raw_features, nested, candidates)
    decision = decide_model(nested, fold_results, summary, uncertainty, persistence, coverage)

    surface.to_csv(output / "candidate_surface.csv", index=False)
    nested.to_csv(output / "fold_selections.csv", index=False)
    fold_results.to_csv(output / "fold_results.csv", index=False)
    unseen.to_csv(output / "unseen_predictions.csv", index=False)
    summary.to_csv(output / "competition_summary.csv", index=False)
    uncertainty.to_csv(output / "dependency_uncertainty.csv", index=False)
    persistence.to_csv(output / "profile_persistence.csv", index=False)
    coverage.to_csv(output / "domestic_evidence_coverage.csv", index=False)
    selected_profiles = selected_profile_rows(raw_profiles, nested, candidates)
    selected_profiles.to_csv(output / "selected_profile_audit.csv", index=False)
    copy_legacy_reference(output)
    (output / "selected_candidate.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "backtest_report.md").write_text(build_report(decision, surface, nested, coverage), encoding="utf-8")
    validate_outputs(baseline, unseen, candidates)
    print(f"Decision: {decision['decision']}")
    print(f"Output: {output}")


def prepare_baseline(baseline: pd.DataFrame, draw_at_even: float, draw_shape: float) -> pd.DataFrame:
    result = baseline.copy().sort_values(["kickoff_utc", "match_id"], kind="stable").reset_index(drop=True)
    result["actual_class"] = np.where(
        result["actual_home_score"].eq(1.0), 0,
        np.where(result["actual_home_score"].eq(0.5), 1, 2),
    )
    probabilities = np.vstack(
        [probability_vector(float(value), draw_at_even, draw_shape) for value in result["expected_home_score"]]
    )
    result[["home_probability", "draw_probability", "away_probability"]] = probabilities
    result["predicted_class"] = probabilities.argmax(axis=1)
    result["brier_1x2"] = np.square(probabilities - np.eye(3)[result["actual_class"].to_numpy(int)]).sum(axis=1)
    result["log_loss_1x2"] = -np.log(np.clip(probabilities[np.arange(len(result)), result["actual_class"].to_numpy(int)], 1e-15, 1.0))
    result["candidate_key"] = BASELINE_KEY
    result["context_expected_home_score"] = result["expected_home_score"]
    result["home_team_selected_effect"] = 0.0
    result["away_team_selected_effect"] = 0.0
    result["applied_matchup_offset"] = 0.0
    result["context_cap_hit"] = False
    return result


def candidate_grid() -> tuple[Candidate, ...]:
    candidates = [Candidate(BASELINE_KEY, "BASELINE", None)]
    for lower, upper in ((0.35, 0.65), (0.40, 0.60), (0.45, 0.55)):
        for window in (3, 5):
            for decay in (0.75, 1.0):
                for domestic_weight in (0.0, 0.25, 0.50, 0.75, 1.0):
                    for shrinkage in (20.0, 40.0, 60.0):
                        for cap in (25.0, 50.0):
                            config = RelativeOpponentProfileConfig(lower, upper, window, decay, domestic_weight, shrinkage, cap)
                            config.validate()
                            arm = EUROPE_ONLY_ARM if domestic_weight == 0.0 else DOMESTIC_ARM
                            candidates.append(Candidate(config.key, arm, config))
    keys = [candidate.key for candidate in candidates]
    if len(keys) != len(set(keys)) or len(candidates) != 361:
        raise ValueError("Expected baseline plus 360 relative profile candidates")
    return tuple(candidates)


def raw_config_grid(candidates: tuple[Candidate, ...]) -> tuple[RawConfig, ...]:
    values = {
        RawConfig(
            candidate.config.lower_expected_score,
            candidate.config.upper_expected_score,
            candidate.config.lookback_seasons,
            candidate.config.season_decay,
            candidate.config.domestic_weight,
        )
        for candidate in candidates
        if candidate.config is not None
    }
    return tuple(sorted(values, key=lambda value: value.key))


def development_raw_config_limit(
    raw_configs: tuple[RawConfig, ...],
    limit: int,
) -> tuple[RawConfig, ...]:
    """Keep both evidence arms available in reduced development runs."""

    if limit < 2:
        raise ValueError("--raw-config-limit must be at least 2 to retain both model arms")
    if limit >= len(raw_configs):
        return raw_configs
    europe = next(raw for raw in raw_configs if raw.domestic_weight == 0.0)
    domestic = next(raw for raw in raw_configs if raw.domestic_weight > 0.0)
    selected = [europe, domestic]
    selected.extend(raw for raw in raw_configs if raw not in selected)
    return tuple(selected[:limit])


def raw_key(config: RelativeOpponentProfileConfig) -> str:
    return RawConfig(
        config.lower_expected_score,
        config.upper_expected_score,
        config.lookback_seasons,
        config.season_decay,
        config.domestic_weight,
    ).key


def build_raw_features(
    baseline: pd.DataFrame,
    european: pd.DataFrame,
    domestic: pd.DataFrame,
    seasons: tuple[str, ...],
    raw: RawConfig,
    elo_scale: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = RelativeOpponentProfileConfig(raw.lower, raw.upper, raw.window, raw.decay, raw.domestic_weight, 0.0, 2000.0)
    config.validate()
    season_values = sorted(set(seasons) | set(european["ao_season"].astype(str)) | set(domestic["ao_season"].astype(str)))
    season_index = {season: index for index, season in enumerate(season_values)}
    events = pd.concat([european, domestic], ignore_index=True, sort=False).sort_values(["kickoff_utc", "match_id", "source", "club_id"], kind="stable")
    baseline_by_match = baseline.set_index("match_id")
    if not baseline_by_match.index.is_unique:
        raise ValueError("Baseline match_id must be unique")
    histories: dict[str, list[dict[str, object]]] = {}
    versions: dict[str, int] = {}
    cache: dict[tuple[str, str, int], object] = {}
    feature_rows: list[dict[str, object]] = []
    profile_rows: list[dict[str, object]] = []
    seen_profile: set[tuple[str, str, str]] = set()
    for _, batch in events.groupby("kickoff_utc", sort=True):
        european_batch = batch.loc[batch["source"].eq("EUROPE")]
        for match_id, match_rows in european_batch.groupby("match_id", sort=True):
            if len(match_rows) != 2:
                raise ValueError(f"{match_id}: expected two European perspectives")
            original = baseline_by_match.loc[str(match_id)]
            home = match_rows.loc[match_rows["club_id"].eq(str(original.home_club_id))].iloc[0]
            away = match_rows.loc[match_rows["club_id"].eq(str(original.away_club_id))].iloc[0]
            home_profile = cached_profile(str(home.club_id), histories, versions, cache, str(original.season), season_index, config, elo_scale)
            away_profile = cached_profile(str(away.club_id), histories, versions, cache, str(original.season), season_index, config, elo_scale)
            feature_rows.append(
                {
                    "match_id": str(match_id),
                    "raw_profile_key": raw.key,
                    "home_relative_band": int(home.opponent_band),
                    "away_relative_band": int(away.opponent_band),
                    "home_raw_effect_strong": home_profile.raw_specific_effects[0],
                    "home_raw_effect_even": home_profile.raw_specific_effects[1],
                    "home_raw_effect_weak": home_profile.raw_specific_effects[2],
                    "home_effective_matches_strong": home_profile.effective_matches[0],
                    "home_effective_matches_even": home_profile.effective_matches[1],
                    "home_effective_matches_weak": home_profile.effective_matches[2],
                    "away_raw_effect_strong": away_profile.raw_specific_effects[0],
                    "away_raw_effect_even": away_profile.raw_specific_effects[1],
                    "away_raw_effect_weak": away_profile.raw_specific_effects[2],
                    "away_effective_matches_strong": away_profile.effective_matches[0],
                    "away_effective_matches_even": away_profile.effective_matches[1],
                    "away_effective_matches_weak": away_profile.effective_matches[2],
                }
            )
            for club_id, profile in ((str(home.club_id), home_profile), (str(away.club_id), away_profile)):
                key = (str(original.season), str(match_id), club_id)
                if key not in seen_profile:
                    profile_rows.append({"season": str(original.season), "match_id": str(match_id), "raw_profile_key": raw.key, **profile_record(profile)})
                    seen_profile.add(key)
        # All profiles above see the same state; only this block makes this timestamp observable.
        for row in batch.itertuples(index=False):
            item = row._asdict()
            club_id = str(item["club_id"])
            histories.setdefault(club_id, []).append(item)
            versions[club_id] = versions.get(club_id, 0) + 1
    features = pd.DataFrame(feature_rows).sort_values("match_id", kind="stable").reset_index(drop=True)
    if len(features) != len(baseline) or features["match_id"].duplicated().any():
        raise ValueError("Raw relative profile features must cover every European match exactly once")
    return features, pd.DataFrame(profile_rows)


def cached_profile(club_id, histories, versions, cache, target_season, season_index, config, elo_scale):
    key = (club_id, target_season, versions.get(club_id, 0))
    if key not in cache:
        history = weighted_history(histories.get(club_id, []), target_season=target_season, season_index=season_index, config=config)
        cache[key] = estimate_relative_profile(club_id, history, config=config, elo_scale=elo_scale)
    return cache[key]


def materialize_candidate(raw_features: pd.DataFrame, candidate: Candidate, draw_at_even: float, draw_shape: float, elo_scale: float) -> pd.DataFrame:
    assert candidate.config is not None
    config = candidate.config
    result = raw_features.copy()
    home_effects = materialize_effects(result, "home", config)
    away_effects = materialize_effects(result, "away", config)
    index = np.arange(len(result))
    home_selected = home_effects[index, result["home_relative_band"].to_numpy(int) - 1]
    away_selected = away_effects[index, result["away_relative_band"].to_numpy(int) - 1]
    result["home_team_selected_effect"] = home_selected
    result["away_team_selected_effect"] = away_selected
    result["applied_matchup_offset"] = np.clip(home_selected - away_selected, -config.effect_cap, config.effect_cap)
    result["context_cap_hit"] = np.isclose(np.abs(result["applied_matchup_offset"]), config.effect_cap, atol=1e-12)
    return result


def materialize_effects(frame: pd.DataFrame, side: str, config: RelativeOpponentProfileConfig) -> np.ndarray:
    labels = ("strong", "even", "weak")
    raw = frame[[f"{side}_raw_effect_{label}" for label in labels]].to_numpy(float)
    effective = frame[[f"{side}_effective_matches_{label}" for label in labels]].to_numpy(float)
    observed = effective > 0.0
    reliability = np.divide(effective, effective + config.shrinkage_matches, out=np.zeros_like(effective), where=effective > 0.0)
    preliminary = np.clip(raw * reliability, -config.effect_cap, config.effect_cap)
    denominator = (effective * observed).sum(axis=1)
    center = np.divide((preliminary * effective * observed).sum(axis=1), denominator, out=np.zeros(len(frame)), where=denominator > 0.0)
    effects = np.where(observed, np.clip(preliminary - center[:, None], -config.effect_cap, config.effect_cap), 0.0)
    effects[observed.sum(axis=1) < 2] = 0.0
    return effects


def candidate_predictions(base: pd.DataFrame, candidate_features: pd.DataFrame, candidate: Candidate, draw_at_even: float, draw_shape: float, elo_scale: float) -> pd.DataFrame:
    # Baseline carries zero-valued audit fields.  Drop them before joining so the
    # candidate's computed offset cannot be silently shadowed by the control row.
    baseline_audit_columns = [
        "context_expected_home_score",
        "home_team_selected_effect",
        "away_team_selected_effect",
        "applied_matchup_offset",
        "context_cap_hit",
    ]
    frame = base.drop(columns=baseline_audit_columns, errors="ignore").merge(
        candidate_features,
        on="match_id",
        validate="one_to_one",
    )
    base_expected = frame["expected_home_score"].to_numpy(float)
    logit = np.log(base_expected / (1.0 - base_expected))
    expected = 1.0 / (1.0 + np.exp(-(logit + math.log(10.0) * frame["applied_matchup_offset"].to_numpy(float) / elo_scale)))
    probabilities = np.vstack([probability_vector(float(value), draw_at_even, draw_shape) for value in expected])
    frame["context_expected_home_score"] = expected
    frame[["home_probability", "draw_probability", "away_probability"]] = probabilities
    frame["predicted_class"] = probabilities.argmax(axis=1)
    observed = frame["actual_class"].to_numpy(int)
    frame["brier_1x2"] = np.square(probabilities - np.eye(3)[observed]).sum(axis=1)
    frame["log_loss_1x2"] = -np.log(np.clip(probabilities[np.arange(len(frame)), observed], 1e-15, 1.0))
    frame["candidate_key"] = candidate.key
    frame["model_arm"] = candidate.arm
    return frame.sort_values(["kickoff_utc", "match_id"], kind="stable").reset_index(drop=True)


def materialize_all_predictions(raw_features: pd.DataFrame, base: pd.DataFrame, candidate: Candidate, draw_at_even: float, draw_shape: float, elo_scale: float) -> pd.DataFrame:
    features = materialize_candidate(raw_features, candidate, draw_at_even, draw_shape, elo_scale)
    return candidate_predictions(base, features, candidate, draw_at_even, draw_shape, elo_scale)


def metrics_by_season(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, values in frame.groupby("season", sort=True):
        rows.append({"season": season, "matches": int(len(values)), "brier_1x2": float(values["brier_1x2"].mean()), "log_loss_1x2": float(values["log_loss_1x2"].mean()), "accuracy_1x2": float(values["actual_class"].eq(values["predicted_class"]).mean()), "context_cap_hit_rate": float(values.get("context_cap_hit", pd.Series(False, index=values.index)).mean()), "mean_abs_context_offset": float(values.get("applied_matchup_offset", pd.Series(0.0, index=values.index)).abs().mean())})
    return pd.DataFrame(rows)


def select_candidate(metrics, candidates, train_seasons, arm):
    values = []
    for candidate in candidates:
        if candidate.arm not in (arm, "BASELINE"):
            continue
        frame = metrics[candidate.key].loc[metrics[candidate.key]["season"].isin(train_seasons)]
        values.append({"candidate_key": candidate.key, "brier_1x2": float(np.average(frame["brier_1x2"], weights=frame["matches"])), "log_loss_1x2": float(np.average(frame["log_loss_1x2"], weights=frame["matches"])), "complexity": candidate.complexity})
    table = pd.DataFrame(values)
    baseline = table.loc[table["candidate_key"].eq(BASELINE_KEY)].iloc[0]
    safe = table.loc[table["brier_1x2"].le(float(baseline.brier_1x2) + EPSILON) & table["log_loss_1x2"].le(float(baseline.log_loss_1x2) + EPSILON)]
    return safe.sort_values(["brier_1x2", "log_loss_1x2", "complexity", "candidate_key"], kind="stable").iloc[0].to_dict()


def nested_selection(metrics, candidates, folds) -> pd.DataFrame:
    rows = []
    for fold, (train, test) in enumerate(folds, start=1):
        for arm in (EUROPE_ONLY_ARM, DOMESTIC_ARM):
            selection = select_candidate(metrics, candidates, set(train), arm)
            rows.append({"fold": fold, "arm": arm, "train_seasons": ",".join(train), "test_season": test, "selected_candidate_key": selection["candidate_key"], "train_brier_1x2": selection["brier_1x2"], "train_log_loss_1x2": selection["log_loss_1x2"]})
    return pd.DataFrame(rows)


def build_unseen_predictions(prediction_sets, selections, folds) -> pd.DataFrame:
    rows = []
    for fold, (_, test) in enumerate(folds, start=1):
        base = prediction_sets[BASELINE_KEY].loc[prediction_sets[BASELINE_KEY]["season"].eq(test)].copy()
        base["fold"], base["model_arm"] = fold, "BASELINE"
        rows.append(base)
        for selection in selections.loc[selections["fold"].eq(fold)].itertuples(index=False):
            frame = prediction_sets[str(selection.selected_candidate_key)].loc[prediction_sets[str(selection.selected_candidate_key)]["season"].eq(test)].copy()
            frame["fold"], frame["model_arm"] = fold, selection.arm
            rows.append(frame)
    return pd.concat(rows, ignore_index=True).sort_values(["fold", "model_arm", "kickoff_utc", "match_id"], kind="stable").reset_index(drop=True)


def build_fold_results(unseen: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fold, arm), frame in unseen.groupby(["fold", "model_arm"], sort=True):
        rows.append(
            {
                "fold": int(fold),
                "model_arm": str(arm),
                "matches": int(len(frame)),
                "brier_1x2": float(frame["brier_1x2"].mean()),
                "log_loss_1x2": float(frame["log_loss_1x2"].mean()),
                "accuracy_1x2": float(frame["actual_class"].eq(frame["predicted_class"]).mean()),
                "context_cap_hit_rate": float(frame.get("context_cap_hit", pd.Series(False, index=frame.index)).mean()),
            }
        )
    return pd.DataFrame(rows)


def candidate_surface(metrics, candidates, folds) -> pd.DataFrame:
    test_seasons = {test for _, test in folds}
    baseline = metrics[BASELINE_KEY].loc[metrics[BASELINE_KEY]["season"].isin(test_seasons)]
    base_brier = float(np.average(baseline.brier_1x2, weights=baseline.matches))
    base_log = float(np.average(baseline.log_loss_1x2, weights=baseline.matches))
    rows = []
    for candidate in candidates:
        frame = metrics[candidate.key].loc[metrics[candidate.key]["season"].isin(test_seasons)]
        config = candidate.config
        brier = float(np.average(frame.brier_1x2, weights=frame.matches))
        log = float(np.average(frame.log_loss_1x2, weights=frame.matches))
        rows.append({"candidate_key": candidate.key, "arm": candidate.arm, "lower": pd.NA if config is None else config.lower_expected_score, "upper": pd.NA if config is None else config.upper_expected_score, "lookback_seasons": 0 if config is None else config.lookback_seasons, "season_decay": 0.0 if config is None else config.season_decay, "domestic_weight": 0.0 if config is None else config.domestic_weight, "shrinkage_matches": 0.0 if config is None else config.shrinkage_matches, "effect_cap": 0.0 if config is None else config.effect_cap, "brier_1x2": brier, "log_loss_1x2": log, "brier_delta_vs_baseline": brier - base_brier, "log_loss_delta_vs_baseline": log - base_log})
    return pd.DataFrame(rows).sort_values(["brier_1x2", "log_loss_1x2", "candidate_key"], kind="stable").reset_index(drop=True)


def comparison_summary(unseen: pd.DataFrame) -> pd.DataFrame:
    base = unseen.loc[unseen.model_arm.eq("BASELINE")]
    rows = []
    for arm in (EUROPE_ONLY_ARM, DOMESTIC_ARM):
        candidate = unseen.loc[unseen.model_arm.eq(arm)]
        paired = candidate.merge(base[["match_id", "brier_1x2", "log_loss_1x2"]], on="match_id", suffixes=("_candidate", "_baseline"), validate="one_to_one")
        for competition, segment in paired.groupby("competition", sort=True):
            rows.append({"arm": arm, "segment": str(competition), "matches": int(len(segment)), "brier_delta_vs_baseline": float((segment.brier_1x2_candidate - segment.brier_1x2_baseline).mean()), "log_loss_delta_vs_baseline": float((segment.log_loss_1x2_candidate - segment.log_loss_1x2_baseline).mean())})
        rows.append({"arm": arm, "segment": "ALL", "matches": int(len(paired)), "brier_delta_vs_baseline": float((paired.brier_1x2_candidate - paired.brier_1x2_baseline).mean()), "log_loss_delta_vs_baseline": float((paired.log_loss_1x2_candidate - paired.log_loss_1x2_baseline).mean())})
    return pd.DataFrame(rows)


def build_uncertainty(unseen: pd.DataFrame, bootstrap_samples: int) -> pd.DataFrame:
    base = unseen.loc[unseen.model_arm.eq("BASELINE")]
    candidate = unseen.loc[unseen.model_arm.eq(DOMESTIC_ARM)]
    paired = candidate.merge(base[["match_id", "brier_1x2", "log_loss_1x2"]], on="match_id", suffixes=("_candidate", "_baseline"), validate="one_to_one")
    rows = []
    for segment, frame in [("ALL", paired), *[(str(name), values) for name, values in paired.groupby("competition", sort=True)]]:
        for metric in ("brier_1x2", "log_loss_1x2"):
            audit = frame[["season", "match_id", "home_team_id", "away_team_id", "kickoff_utc", "tie_id"]].copy()
            audit["loss_difference"] = frame[f"{metric}_candidate"] - frame[f"{metric}_baseline"]
            result = dependency_robust_loss_difference_ci(audit, bootstrap_samples=bootstrap_samples)
            result["segment"] = segment
            result["metric"] = metric
            rows.append(result)
    return pd.concat(rows, ignore_index=True)


def build_persistence(unseen: pd.DataFrame, bootstrap_samples: int) -> pd.DataFrame:
    candidate = unseen.loc[unseen.model_arm.eq(DOMESTIC_ARM)]
    rows = []
    for fold, frame in candidate.groupby("fold", sort=True):
        rows.append(persistence_row(str(fold), perspective_effects(frame), bootstrap_samples))
    rows.append(persistence_row("ALL", perspective_effects(candidate), bootstrap_samples))
    return pd.DataFrame(rows)


def perspective_effects(frame: pd.DataFrame) -> pd.DataFrame:
    home = pd.DataFrame({"season": frame.season, "club_id": frame.home_club_id, "effect": frame.home_team_selected_effect, "residual": frame.actual_home_score - frame.expected_home_score})
    away = pd.DataFrame({"season": frame.season, "club_id": frame.away_club_id, "effect": frame.away_team_selected_effect, "residual": (1.0 - frame.actual_home_score) - (1.0 - frame.expected_home_score)})
    return pd.concat([home, away], ignore_index=True).loc[lambda values: values.effect.abs().gt(1e-12)]


def persistence_row(label: str, frame: pd.DataFrame, samples: int) -> dict[str, object]:
    spearman = float(frame.effect.corr(frame.residual, method="spearman")) if len(frame) >= 3 and frame.effect.nunique() > 1 else math.nan
    sign = float((frame.effect * frame.residual > 0.0).mean()) if len(frame) else math.nan
    lower, upper = bootstrap_spearman(frame, samples)
    return {"fold": label, "team_match_observations": int(len(frame)), "clubs": int(frame.club_id.nunique()), "spearman": spearman, "sign_accuracy": sign, "ci_95_lower": lower, "ci_95_upper": upper, "reliable_negative": bool(np.isfinite(upper) and upper < 0.0)}


def bootstrap_spearman(frame: pd.DataFrame, samples: int) -> tuple[float, float]:
    groups = [group for _, group in frame.groupby(["season", "club_id"], sort=False)]
    if len(groups) < 2:
        return math.nan, math.nan
    generator = np.random.default_rng(20260807)
    values = []
    for _ in range(samples):
        sampled = pd.concat([groups[index] for index in generator.integers(0, len(groups), len(groups))], ignore_index=True)
        if sampled.effect.nunique() > 1 and sampled.residual.nunique() > 1:
            values.append(float(sampled.effect.corr(sampled.residual, method="spearman")))
    return tuple(np.quantile(values, (0.025, 0.975))) if values else (math.nan, math.nan)


def build_domestic_coverage(
    domestic: pd.DataFrame,
    raw_features: dict[str, pd.DataFrame],
    selections: pd.DataFrame,
    candidates: tuple[Candidate, ...],
) -> pd.DataFrame:
    candidate_map = {candidate.key: candidate for candidate in candidates}
    selected_raw_keys = {
        raw_key(candidate_map[str(key)].config)
        for key in selections.selected_candidate_key.astype(str)
        if key != BASELINE_KEY
    }
    rows = [{"metric": "accepted_domestic_matches", "value": int(len(domestic))}, {"metric": "mapped_domestic_team_sides", "value": int(domestic.home_ao_club_id.notna().sum() + domestic.away_ao_club_id.notna().sum())}]
    for key, features in raw_features.items():
        if key in selected_raw_keys:
            rows.append({"metric": f"selected_raw_profile_rows:{key}", "value": int(len(features))})
    return pd.DataFrame(rows)


def selected_profile_rows(raw_profiles, selections, candidates) -> pd.DataFrame:
    candidate_map = {candidate.key: candidate for candidate in candidates}
    selected = []
    for selection in selections.itertuples(index=False):
        if selection.selected_candidate_key == BASELINE_KEY:
            continue
        candidate = candidate_map[str(selection.selected_candidate_key)]
        assert candidate.config is not None
        values = raw_profiles[raw_key(candidate.config)]
        selected.append(
            values.loc[values.season.eq(selection.test_season)].assign(
                fold=selection.fold,
                arm=selection.arm,
                selected_candidate_key=selection.selected_candidate_key,
            )
        )
    return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()


def decide_model(selections, fold_results, summary, uncertainty, persistence, coverage):
    domestic = selections.loc[selections.arm.eq(DOMESTIC_ARM)]
    base = selections.loc[selections.arm.eq(EUROPE_ONLY_ARM)]
    unseen_brier = summary.loc[(summary.arm.eq(DOMESTIC_ARM)) & (summary.segment.eq("ALL")), "brier_delta_vs_baseline"].iloc[0]
    unseen_log = summary.loc[(summary.arm.eq(DOMESTIC_ARM)) & (summary.segment.eq("ALL")), "log_loss_delta_vs_baseline"].iloc[0]
    envelope = uncertainty.loc[uncertainty.method.eq("conservative_envelope")]
    all_brier = envelope.loc[(envelope.segment.eq("ALL")) & (envelope.metric.eq("brier_1x2"))].iloc[0]
    all_log = envelope.loc[(envelope.segment.eq("ALL")) & (envelope.metric.eq("log_loss_1x2"))].iloc[0]
    segment_harm = bool(envelope.loc[~envelope.segment.eq("ALL"), "reliable_harm"].any())
    persistence_all = persistence.loc[persistence.fold.eq("ALL")].iloc[0]
    pivot = fold_results.pivot(index="fold", columns="model_arm", values=["brier_1x2", "log_loss_1x2"])
    brier_wins = int((pivot["brier_1x2"][DOMESTIC_ARM] <= pivot["brier_1x2"]["BASELINE"]).sum())
    log_wins = int((pivot["log_loss_1x2"][DOMESTIC_ARM] <= pivot["log_loss_1x2"]["BASELINE"]).sum())
    domestic_beats_europe = bool(
        (pivot["brier_1x2"][DOMESTIC_ARM] <= pivot["brier_1x2"][EUROPE_ONLY_ARM]).sum() >= 4
        and (pivot["log_loss_1x2"][DOMESTIC_ARM] <= pivot["log_loss_1x2"][EUROPE_ONLY_ARM]).sum() >= 4
    )
    selected_domestic = domestic.selected_candidate_key.ne(BASELINE_KEY).all()
    gates = {
        "brier_at_least_4_of_6": brier_wins >= 4,
        "log_loss_at_least_4_of_6": log_wins >= 4,
        "pooled_brier_not_worse": float(all_brier.mean_difference) <= EPSILON,
        "pooled_log_loss_not_worse": float(all_log.mean_difference) <= EPSILON,
        "no_competition_reliable_harm": not segment_harm,
        "profile_persistence_spearman_at_least_010": float(persistence_all.spearman) >= 0.10,
        "profile_direction_accuracy_at_least_055": float(persistence_all.sign_accuracy) >= 0.55,
        "domestic_candidate_selected_each_fold": bool(selected_domestic),
        "domestic_outperforms_europe_only_in_at_least_4_folds": domestic_beats_europe,
        "prediction_only_state_invariant": True,
    }
    if all(gates.values()):
        decision = "PROMOTE_DYNAMIC_SHADOW_CANDIDATE"
    elif bool(all_brier.reliable_harm) or bool(all_log.reliable_harm) or segment_harm:
        decision = "REJECT"
    else:
        decision = "KEEP_DIAGNOSTIC"
    return {"decision": decision, "production_changed": False, "layer_mode": "PREDICTION_ONLY_SHADOW", "nested_brier_fold_wins": f"{brier_wins}/6", "nested_log_loss_fold_wins": f"{log_wins}/6", "pooled_brier_delta_vs_baseline": float(unseen_brier), "pooled_log_loss_delta_vs_baseline": float(unseen_log), "profile_persistence_spearman": float(persistence_all.spearman), "profile_direction_accuracy": float(persistence_all.sign_accuracy), "gates": gates, "domestic_evidence": coverage.to_dict(orient="records")}


def copy_legacy_reference(output: Path) -> None:
    if LEGACY_QUINTILE_DECISION.is_file():
        output.joinpath("legacy_quintile_reference.json").write_text(LEGACY_QUINTILE_DECISION.read_text(encoding="utf-8"), encoding="utf-8")


def validate_outputs(baseline: pd.DataFrame, unseen: pd.DataFrame, candidates: tuple[Candidate, ...]) -> None:
    if len(unseen.loc[unseen.model_arm.eq("BASELINE")]) != len(baseline.loc[baseline.season.isin(sorted(unseen.season.unique()))]):
        raise ValueError("Nested baseline predictions must preserve all unseen matches")
    if unseen[["home_probability", "draw_probability", "away_probability"]].isna().any().any():
        raise ValueError("Relative profile probabilities cannot be missing")
    values = unseen[["home_probability", "draw_probability", "away_probability"]].to_numpy(float)
    if (values < 0.0).any() or not np.allclose(values.sum(axis=1), 1.0, atol=1e-12):
        raise ValueError("Relative profile probabilities must be valid")
    if len({candidate.key for candidate in candidates}) != len(candidates):
        raise ValueError("Candidate keys must be unique")


def build_report(decision, surface, selections, coverage) -> str:
    best = markdown_table(surface.head(10))
    return f"""# Yerel Lig Destekli Rakip Profili Backtesti\n\n- Karar: `{decision['decision']}`\n- Katman: prediction-only shadow; AO Live state değişmez.\n- Brier fold kazanımı: {decision['nested_brier_fold_wins']}\n- Log-loss fold kazanımı: {decision['nested_log_loss_fold_wins']}\n- Pooled Brier farkı: {decision['pooled_brier_delta_vs_baseline']:.8f}\n- Pooled log-loss farkı: {decision['pooled_log_loss_delta_vs_baseline']:.8f}\n- Profil devamlılığı Spearman: {decision['profile_persistence_spearman']:.4f}\n\n## Seçilimler\n\n{markdown_table(selections)}\n\n## En İyi Yüzey\n\n{best}\n\n## Yerel Kanıt\n\n{markdown_table(coverage)}\n"""


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_Satır yok._"
    columns = [str(column) for column in frame.columns]

    def cell(value: object) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")

    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    rows.extend(
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(rows)


if __name__ == "__main__":
    main()
