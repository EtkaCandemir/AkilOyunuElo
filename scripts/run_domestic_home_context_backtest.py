from __future__ import annotations

"""Prediction-only test of club-specific home context learned from domestic play."""

import argparse
import gc
import json
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
    DomesticRatingConfig,
    replay_domestic_perspectives,
)
from ao_elo.team_venue_context import (  # noqa: E402
    TeamVenueContextConfig,
    contextual_home_expectation,
    estimate_team_venue_effect,
)
from scripts.run_dynamic_core_calibration import expanding_folds  # noqa: E402
from scripts.run_opponent_quintile_backtest import load_production_baseline  # noqa: E402
from scripts.run_stage_weighted_progression_backtest import probability_vector  # noqa: E402


DOMESTIC_MATCHES = ROOT / "data" / "domestic_league_matches_2013_2026" / "domestic_matches.csv"
OUTPUT_ROOT = ROOT / "output" / "domestic_home_context_backtest_2018_2026"
BASELINE_KEY = "global_home_advantage"
BASELINE_MODEL = "PRODUCTION_GLOBAL_H"
CONTEXT_MODEL = "DOMESTIC_HOME_CONTEXT"
TOLERANCE = 1e-12


@dataclass(frozen=True)
class Candidate:
    key: str
    config: TeamVenueContextConfig | None

    @property
    def complexity(self) -> int:
        return int(self.config is not None)


@dataclass
class PredictionSet:
    predictions: pd.DataFrame
    profiles: pd.DataFrame


def candidate_grid() -> tuple[Candidate, ...]:
    candidates = [Candidate(BASELINE_KEY, None)]
    for window in (3, 5):
        for decay in (0.75, 1.0):
            for shrinkage in (6.0, 10.0, 15.0, 20.0):
                for cap in (25.0, 35.0, 50.0, 75.0):
                    config = TeamVenueContextConfig(window, decay, shrinkage, cap, cap)
                    config.validate()
                    candidates.append(Candidate(f"domestic_home_{config.key}", config))
    if len(candidates) != 65:
        raise ValueError("Expected baseline plus 64 domestic home candidates")
    return tuple(candidates)


def load_domestic_home_perspectives(path: Path, *, elo_scale: float, home_advantage: float, k_factor: float) -> pd.DataFrame:
    matches = pd.read_csv(path)
    perspectives = replay_domestic_perspectives(
        matches,
        config=DomesticRatingConfig(elo_scale, home_advantage, k_factor),
        lower=0.35,
        upper=0.65,
    )
    home = perspectives.loc[perspectives["venue"].eq("HOME")].copy()
    if home.empty:
        raise ValueError("Domestic source has no mapped home perspectives")
    return home.sort_values(["kickoff_utc", "match_id", "club_id"], kind="stable").reset_index(drop=True)


def season_profiles(
    home_history: pd.DataFrame,
    clubs: list[str],
    *,
    target_season: str,
    season_index: dict[str, int],
    elo_scale: float,
    config: TeamVenueContextConfig,
) -> pd.DataFrame:
    target_index = season_index[target_season]
    history = home_history.copy()
    history["season_index"] = history["ao_season"].astype(str).map(season_index)
    history = history.loc[
        history["season_index"].notna()
        & history["season_index"].lt(target_index)
        & history["season_index"].ge(target_index - config.lookback_seasons)
    ].copy()
    history["weight"] = np.power(
        config.season_decay,
        target_index - history["season_index"].astype(int) - 1,
    )
    by_club = {club: frame for club, frame in history.groupby("club_id", sort=False)}
    rows: list[dict[str, object]] = []
    for club_id in clubs:
        frame = by_club.get(club_id, pd.DataFrame(columns=history.columns))
        estimate = estimate_team_venue_effect(
            frame["expected_score"].tolist(),
            frame["actual_score"].tolist(),
            frame["weight"].tolist(),
            elo_scale=elo_scale,
            shrinkage_matches=config.shrinkage_matches,
            max_team_effect=config.max_team_effect,
        )
        rows.append(
            {
                "season": target_season,
                "club_id": club_id,
                "home_observations": estimate.observations,
                "home_effective_matches": estimate.effective_matches,
                "home_raw_effect": estimate.raw_effect,
                "home_reliability": estimate.reliability,
                "home_effect_shrunk": estimate.shrunk_effect,
                "history_max_season": (
                    pd.NA
                    if frame.empty
                    else max(frame["ao_season"].astype(str), key=season_index.__getitem__)
                ),
            }
        )
    result = pd.DataFrame(rows)
    weighted = result.loc[result["home_effective_matches"].gt(0.0)]
    center = (
        float(np.average(weighted["home_effect_shrunk"], weights=weighted["home_effective_matches"]))
        if not weighted.empty
        else 0.0
    )
    result["home_center"] = center
    result["home_effect_centered"] = np.where(
        result["home_observations"].gt(0),
        np.clip(result["home_effect_shrunk"] - center, -config.max_team_effect, config.max_team_effect),
        0.0,
    )
    result["away_effect"] = 0.0
    result["candidate_key"] = f"domestic_home_{config.key}"
    return result


def build_context_predictions(
    baseline: pd.DataFrame,
    home_history: pd.DataFrame,
    seasons: tuple[str, ...],
    *,
    elo_scale: float,
    global_home_advantage: float,
    draw_at_even: float,
    draw_shape: float,
    config: TeamVenueContextConfig,
) -> PredictionSet:
    config.validate()
    season_index = {season: index for index, season in enumerate(seasons)}
    prediction_frames: list[pd.DataFrame] = []
    profile_frames: list[pd.DataFrame] = []
    for season in seasons:
        current = baseline.loc[baseline["season"].eq(season)].copy()
        clubs = sorted(set(current["home_club_id"].astype(str)) | set(current["away_club_id"].astype(str)))
        profiles = season_profiles(
            home_history,
            clubs,
            target_season=season,
            season_index=season_index,
            elo_scale=elo_scale,
            config=config,
        )
        profile_frames.append(profiles)
        effect_map = profiles.set_index("club_id")["home_effect_centered"]
        current["home_team_effect"] = current["home_club_id"].map(effect_map).fillna(0.0)
        current["away_team_effect"] = 0.0
        contexts = [
            contextual_home_expectation(
                float(base),
                float(effect),
                0.0,
                global_home_advantage=global_home_advantage,
                elo_scale=elo_scale,
                is_neutral=bool(neutral),
                config=config,
            )
            for base, effect, neutral in zip(
                current["expected_home_score"], current["home_team_effect"], current["is_neutral"], strict=True
            )
        ]
        current["raw_context_offset"] = [item.raw_context_offset for item in contexts]
        current["applied_context_offset"] = [item.applied_context_offset for item in contexts]
        current["effective_home_advantage"] = [item.effective_home_advantage for item in contexts]
        current["context_expected_home_score"] = [item.expected_home_score for item in contexts]
        probabilities = np.vstack([
            probability_vector(value, draw_at_even, draw_shape)
            for value in current["context_expected_home_score"]
        ])
        current[["home_probability", "draw_probability", "away_probability"]] = probabilities
        observed = current["actual_class"].to_numpy(int)
        current["brier_1x2"] = np.square(probabilities - np.eye(3)[observed]).sum(axis=1)
        current["log_loss_1x2"] = -np.log(np.clip(probabilities[np.arange(len(current)), observed], 1e-15, 1.0))
        current["predicted_class"] = probabilities.argmax(axis=1)
        current["candidate_key"] = f"domestic_home_{config.key}"
        current["configured_context_cap"] = config.max_context_offset
        prediction_frames.append(current)
    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(["kickoff_utc", "match_id"], kind="stable")
    profiles = pd.concat(profile_frames, ignore_index=True)
    validate_predictions(predictions, profiles, config)
    return PredictionSet(predictions.reset_index(drop=True), profiles.reset_index(drop=True))


def metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    non_neutral = frame.loc[~frame["is_neutral"].astype(bool)]
    capped = non_neutral.loc[non_neutral["configured_context_cap"].gt(0.0)]
    return {
        "matches": int(len(frame)),
        "brier_1x2": float(frame["brier_1x2"].mean()),
        "log_loss_1x2": float(frame["log_loss_1x2"].mean()),
        "accuracy_1x2": float((frame["actual_class"] == frame["predicted_class"]).mean()),
        "mean_abs_context_offset": float(frame["applied_context_offset"].abs().mean()),
        "maximum_abs_context_offset": float(frame["applied_context_offset"].abs().max()),
        "context_cap_hit_rate": float(
            np.isclose(
                capped["applied_context_offset"].abs(),
                capped["configured_context_cap"],
                atol=1e-9,
            ).mean()
        ) if not capped.empty else 0.0,
    }


def select_candidate(sets: dict[str, PredictionSet], candidates: tuple[Candidate, ...], seasons: set[str]) -> dict[str, object]:
    rows = []
    for candidate in candidates:
        value = metrics(sets[candidate.key].predictions.loc[sets[candidate.key].predictions["season"].isin(seasons)])
        rows.append({"candidate_key": candidate.key, "complexity": candidate.complexity, **value})
    surface = pd.DataFrame(rows)
    baseline = surface.loc[surface["candidate_key"].eq(BASELINE_KEY)].iloc[0]
    safe = surface.loc[(surface["brier_1x2"] <= baseline["brier_1x2"] + TOLERANCE) & (surface["log_loss_1x2"] <= baseline["log_loss_1x2"] + TOLERANCE)]
    return safe.sort_values(["brier_1x2", "log_loss_1x2", "complexity", "candidate_key"], kind="stable").iloc[0].to_dict()


def aggregate_candidate_metrics(table: pd.DataFrame, seasons: set[str]) -> pd.DataFrame:
    selected = table.loc[table["season"].isin(seasons)].copy()
    rows = []
    for candidate_key, frame in selected.groupby("candidate_key", sort=True):
        weights = frame["matches"].to_numpy(float)
        row = {
            "candidate_key": candidate_key,
            "complexity": int(frame["complexity"].iloc[0]),
            "matches": int(frame["matches"].sum()),
        }
        for column in (
            "brier_1x2",
            "log_loss_1x2",
            "accuracy_1x2",
            "mean_abs_context_offset",
            "maximum_abs_context_offset",
            "context_cap_hit_rate",
        ):
            row[column] = float(np.average(frame[column], weights=weights))
        for column in ("lookback_seasons", "season_decay", "shrinkage_matches", "effect_cap"):
            row[column] = frame[column].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def select_candidate_from_metrics(table: pd.DataFrame, seasons: set[str]) -> dict[str, object]:
    aggregate = aggregate_candidate_metrics(table, seasons)
    baseline = aggregate.loc[aggregate["candidate_key"].eq(BASELINE_KEY)].iloc[0]
    safe = aggregate.loc[
        (aggregate["brier_1x2"] <= baseline["brier_1x2"] + TOLERANCE)
        & (aggregate["log_loss_1x2"] <= baseline["log_loss_1x2"] + TOLERANCE)
    ]
    return safe.sort_values(
        ["brier_1x2", "log_loss_1x2", "complexity", "candidate_key"],
        kind="stable",
    ).iloc[0].to_dict()


def run_nested(sets: dict[str, PredictionSet], candidates: tuple[Candidate, ...], folds) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selections: list[dict[str, object]] = []
    results: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    for fold, (train_seasons, test_season) in enumerate(folds, start=1):
        selected = select_candidate(sets, candidates, set(train_seasons))
        key = str(selected["candidate_key"])
        selections.append({"fold": fold, "train_seasons": ", ".join(train_seasons), "test_season": test_season, "selected_candidate_key": key, "train_brier_1x2": selected["brier_1x2"], "train_log_loss_1x2": selected["log_loss_1x2"]})
        for model, candidate_key in ((BASELINE_MODEL, BASELINE_KEY), (CONTEXT_MODEL, key)):
            test = sets[candidate_key].predictions.loc[sets[candidate_key].predictions["season"].eq(test_season)].copy()
            test["fold"] = fold
            test["model"] = model
            test["selected_candidate_key"] = key
            predictions.append(test)
            results.append({"fold": fold, "test_season": test_season, "model": model, "candidate_key": candidate_key, **metrics(test)})
        print(f"  fold {fold}/6 -> {test_season}: {key}", flush=True)
    return pd.DataFrame(selections), pd.DataFrame(results), pd.concat(predictions, ignore_index=True)


def uncertainty(predictions: pd.DataFrame, *, bootstrap_samples: int) -> pd.DataFrame:
    base = predictions.loc[predictions["model"].eq(BASELINE_MODEL)]
    candidate = predictions.loc[predictions["model"].eq(CONTEXT_MODEL)]
    rows = []
    for competition in ("ALL", "UCL", "UEL", "UECL"):
        left = candidate if competition == "ALL" else candidate.loc[candidate["competition"].eq(competition)]
        right = base if competition == "ALL" else base.loc[base["competition"].eq(competition)]
        paired = left.merge(right[["match_id", "brier_1x2", "log_loss_1x2"]], on="match_id", suffixes=("_candidate", "_baseline"), validate="one_to_one")
        for metric in ("brier_1x2", "log_loss_1x2"):
            sample = paired[["season", "match_id", "home_team_id", "away_team_id", "kickoff_utc", "tie_id"]].copy()
            sample["loss_difference"] = paired[f"{metric}_candidate"] - paired[f"{metric}_baseline"]
            result = dependency_robust_loss_difference_ci(sample, bootstrap_samples=bootstrap_samples)
            result["competition"] = competition
            result["metric"] = metric
            rows.append(result)
    return pd.concat(rows, ignore_index=True)


def validate_predictions(predictions: pd.DataFrame, profiles: pd.DataFrame, config: TeamVenueContextConfig) -> None:
    probabilities = predictions[["home_probability", "draw_probability", "away_probability"]].to_numpy(float)
    if not np.isfinite(probabilities).all() or (probabilities < 0.0).any() or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12):
        raise ValueError("Home context probabilities must be valid")
    if not np.allclose(predictions["away_team_effect"], 0.0):
        raise ValueError("Home-only context may not use away effects")
    neutral = predictions.loc[predictions["is_neutral"].astype(bool)]
    if not neutral.empty and (not np.allclose(neutral["applied_context_offset"], 0.0) or not np.allclose(neutral["effective_home_advantage"], 0.0)):
        raise ValueError("Neutral matches must ignore home context")
    non_neutral = predictions.loc[~predictions["is_neutral"].astype(bool)]
    if non_neutral["applied_context_offset"].abs().max() > config.max_context_offset + 1e-9:
        raise ValueError("Home context exceeded cap")
    if profiles["home_effect_centered"].abs().max() > config.max_team_effect + 1e-9:
        raise ValueError("Home effect exceeded cap")


def state_invariant(predictions: pd.DataFrame) -> bool:
    base = predictions.loc[predictions["model"].eq(BASELINE_MODEL)]
    candidate = predictions.loc[predictions["model"].eq(CONTEXT_MODEL)]
    fields = ["home_live_pre", "away_live_pre", "expected_home_score", "power_delta", "goal_multiplier", "xg_performance_adjustment", "bonus_applied_after_match"]
    merged = candidate.merge(base[["match_id", *fields]], on="match_id", suffixes=("_candidate", "_baseline"), validate="one_to_one")
    return len(merged) == len(base) and all(np.allclose(merged[f"{field}_candidate"], merged[f"{field}_baseline"], atol=1e-12, rtol=0.0, equal_nan=True) for field in fields)


def markdown_table(frame: pd.DataFrame, digits: int = 7) -> str:
    if frame.empty:
        return "_Yok._"
    headers = [str(column) for column in frame.columns]
    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for record in frame.itertuples(index=False, name=None):
        values = [
            f"{float(value):.{digits}f}"
            if isinstance(value, (float, np.floating))
            else str(value)
            for value in record
        ]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Nested prediction-only domestic home-context backtest")
    parser.add_argument("--domestic-matches", type=Path, default=DOMESTIC_MATCHES)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples must be at least 100")

    baseline, datasets, core, parameters, _ = load_production_baseline()
    seasons = tuple(data.season for data in datasets)
    folds = expanding_folds(seasons)
    if len(folds) != 6:
        raise ValueError("Expected six expanding folds")
    domestic_home = load_domestic_home_perspectives(args.domestic_matches.resolve(), elo_scale=core.elo_scale, home_advantage=core.home_advantage, k_factor=core.k_factor)
    candidates = candidate_grid()
    baseline = baseline.copy()
    baseline["configured_context_cap"] = 0.0
    base = PredictionSet(baseline, pd.DataFrame())
    metric_rows = []
    for season in seasons:
        metric_rows.append({
            "candidate_key": BASELINE_KEY,
            "season": season,
            "complexity": 0,
            "lookback_seasons": 0,
            "season_decay": 0.0,
            "shrinkage_matches": 0.0,
            "effect_cap": 0.0,
            **metrics(base.predictions.loc[base.predictions["season"].eq(season)]),
        })
    print(f"Domestic home context: {len(domestic_home):,} mapped home observations; evaluating {len(candidates) - 1} candidates", flush=True)
    for number, candidate in enumerate(candidates[1:], start=1):
        assert candidate.config is not None
        prediction_set = build_context_predictions(baseline, domestic_home, seasons, elo_scale=core.elo_scale, global_home_advantage=core.home_advantage, draw_at_even=float(parameters["draw_at_even"]), draw_shape=float(parameters["draw_shape"]), config=candidate.config)
        for season in seasons:
            metric_rows.append({
                "candidate_key": candidate.key,
                "season": season,
                "complexity": candidate.complexity,
                "lookback_seasons": candidate.config.lookback_seasons,
                "season_decay": candidate.config.season_decay,
                "shrinkage_matches": candidate.config.shrinkage_matches,
                "effect_cap": candidate.config.max_team_effect,
                **metrics(prediction_set.predictions.loc[prediction_set.predictions["season"].eq(season)]),
            })
        del prediction_set
        gc.collect()
        if number % 8 == 0 or number == len(candidates) - 1:
            print(f"  candidate {number}/{len(candidates) - 1}", flush=True)

    candidate_metrics = pd.DataFrame(metric_rows)
    fold_selection_rows = []
    for fold, (train_seasons, test_season) in enumerate(folds, start=1):
        selected = select_candidate_from_metrics(candidate_metrics, set(train_seasons))
        fold_selection_rows.append({
            "fold": fold,
            "train_seasons": ", ".join(train_seasons),
            "test_season": test_season,
            "selected_candidate_key": selected["candidate_key"],
            "train_brier_1x2": selected["brier_1x2"],
            "train_log_loss_1x2": selected["log_loss_1x2"],
        })
    selections = pd.DataFrame(fold_selection_rows)
    selected_keys = set(selections["selected_candidate_key"].astype(str))
    sets: dict[str, PredictionSet] = {BASELINE_KEY: base}
    candidates_by_key = {candidate.key: candidate for candidate in candidates}
    for key in sorted(selected_keys.difference({BASELINE_KEY})):
        candidate = candidates_by_key[key]
        assert candidate.config is not None
        sets[key] = build_context_predictions(baseline, domestic_home, seasons, elo_scale=core.elo_scale, global_home_advantage=core.home_advantage, draw_at_even=float(parameters["draw_at_even"]), draw_shape=float(parameters["draw_shape"]), config=candidate.config)
    result_rows = []
    unseen_frames = []
    for selection in selections.itertuples(index=False):
        for model, key in ((BASELINE_MODEL, BASELINE_KEY), (CONTEXT_MODEL, selection.selected_candidate_key)):
            test = sets[key].predictions.loc[sets[key].predictions["season"].eq(selection.test_season)].copy()
            test["fold"] = selection.fold
            test["model"] = model
            test["selected_candidate_key"] = selection.selected_candidate_key
            unseen_frames.append(test)
            result_rows.append({"fold": selection.fold, "test_season": selection.test_season, "model": model, "candidate_key": key, **metrics(test)})
    results = pd.DataFrame(result_rows)
    unseen = pd.concat(unseen_frames, ignore_index=True)
    test_seasons = {test for _, test in folds}
    surface = aggregate_candidate_metrics(candidate_metrics, test_seasons)
    base_surface = surface.loc[surface["candidate_key"].eq(BASELINE_KEY)].iloc[0]
    surface["brier_delta_vs_baseline"] = surface["brier_1x2"] - base_surface["brier_1x2"]
    surface["log_loss_delta_vs_baseline"] = surface["log_loss_1x2"] - base_surface["log_loss_1x2"]
    comp_rows = []
    for (model, competition), frame in unseen.groupby(["model", "competition"], sort=True):
        comp_rows.append({"model": model, "competition": competition, **metrics(frame)})
    competition = pd.DataFrame(comp_rows)
    base_comp = competition.loc[competition["model"].eq(BASELINE_MODEL)].set_index("competition")
    competition["brier_delta_vs_baseline"] = competition.apply(lambda row: row["brier_1x2"] - base_comp.loc[row["competition"], "brier_1x2"], axis=1)
    competition["log_loss_delta_vs_baseline"] = competition.apply(lambda row: row["log_loss_1x2"] - base_comp.loc[row["competition"], "log_loss_1x2"], axis=1)
    ci = uncertainty(unseen, bootstrap_samples=args.bootstrap_samples)
    full_selection = select_candidate_from_metrics(candidate_metrics, set(seasons))
    base_results = results.loc[results["model"].eq(BASELINE_MODEL)].set_index("fold")
    ctx_results = results.loc[results["model"].eq(CONTEXT_MODEL)].set_index("fold")
    brier_delta = ctx_results["brier_1x2"] - base_results["brier_1x2"]
    log_delta = ctx_results["log_loss_1x2"] - base_results["log_loss_1x2"]
    all_envelope = ci.loc[(ci["competition"].eq("ALL")) & (ci["method"].eq("conservative_envelope"))]
    context_comp = competition.loc[competition["model"].eq(CONTEXT_MODEL)]
    decision = {
        "decision": "PROMOTE_HOME_CONTEXT_SHADOW_CANDIDATE" if int(((brier_delta < 0) & (log_delta < 0)).sum()) >= 4 and float(np.average(ctx_results["brier_1x2"], weights=ctx_results["matches"]) - np.average(base_results["brier_1x2"], weights=base_results["matches"])) <= 0 and float(np.average(ctx_results["log_loss_1x2"], weights=ctx_results["matches"]) - np.average(base_results["log_loss_1x2"], weights=base_results["matches"])) <= 0 and not ci.loc[ci["method"].eq("conservative_envelope"), "reliable_harm"].any() and state_invariant(unseen) else "KEEP_DIAGNOSTIC",
        "production_changed": False,
        "layer_mode": "PREDICTION_ONLY_HOME_ONLY",
        "full_history_candidate_key": str(full_selection["candidate_key"]),
        "nested_fold_both_loss_wins": f"{int(((brier_delta < 0) & (log_delta < 0)).sum())}/6",
        "pooled_brier_delta": float(np.average(ctx_results["brier_1x2"], weights=ctx_results["matches"]) - np.average(base_results["brier_1x2"], weights=base_results["matches"])),
        "pooled_log_loss_delta": float(np.average(ctx_results["log_loss_1x2"], weights=ctx_results["matches"]) - np.average(base_results["log_loss_1x2"], weights=base_results["matches"])),
        "no_segment_reliable_harm": bool(not ci.loc[ci["method"].eq("conservative_envelope"), "reliable_harm"].any()),
        "prediction_only_state_invariant": state_invariant(unseen),
        "interpretation": "Only club home context is estimated from prior domestic home results. Away context remains zero; AO Live state is unchanged.",
    }
    selected_profiles = []
    for selection in selections.itertuples(index=False):
        if selection.selected_candidate_key != BASELINE_KEY:
            selected_profiles.append(sets[selection.selected_candidate_key].profiles.loc[sets[selection.selected_candidate_key].profiles["season"].eq(selection.test_season)].assign(fold=selection.fold))
    profiles = pd.concat(selected_profiles, ignore_index=True) if selected_profiles else pd.DataFrame()
    coverage = pd.DataFrame([{
        "mapped_domestic_club_perspectives": int(len(domestic_home)),
        "mapped_domestic_home_observations": int(len(domestic_home)),
        "european_matches": int(len(baseline)),
        "european_teams": int(
            len(
                set(baseline["home_club_id"].astype(str))
                | set(baseline["away_club_id"].astype(str))
            )
        ),
        "source": str(args.domestic_matches.resolve()),
    }])
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    surface.to_csv(output / "candidate_surface.csv", index=False)
    selections.to_csv(output / "fold_selections.csv", index=False)
    results.to_csv(output / "fold_results.csv", index=False)
    unseen.to_csv(output / "unseen_predictions.csv", index=False)
    competition.to_csv(output / "competition_summary.csv", index=False)
    profiles.to_csv(output / "home_context_profiles.csv", index=False)
    ci.to_csv(output / "dependency_uncertainty.csv", index=False)
    coverage.to_csv(output / "data_coverage.csv", index=False)
    (output / "selected_candidate.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")
    report = [
        "# Yerel Lig Destekli Takım Bazlı HomeContext Backtesti",
        "",
        "## Sözleşme",
        "",
        "- Katman yalnız geçmiş, tamamlanmış yerel **ev sahibi** maçlarından tahmin edilir.",
        "- Deplasman etkisi bu koşuda sabit `0`dır; rakip profili, Power Delta ve AO Live state değiştirilmez.",
        f"- Avrupa değerlendirmesi: `{seasons[0]}`–`{seasons[-1]}`, altı expanding fold.",
        f"- Global H: `{core.home_advantage:.6f}`, Elo Scale: `{core.elo_scale:.6f}`.",
        "- Adaylar: lookback `3/5`, season decay `0.75/1.00`, shrinkage `6/10/15/20`, effect cap `25/35/50/75`.",
        "",
        "## Veri Kapsamı",
        "",
        markdown_table(coverage),
        "",
        "## Fold Seçimleri",
        "",
        markdown_table(selections),
        "",
        "## Fold Sonuçları",
        "",
        markdown_table(results),
        "",
        "## Turnuva Segmentleri",
        "",
        markdown_table(competition[["model", "competition", "brier_delta_vs_baseline", "log_loss_delta_vs_baseline", "mean_abs_context_offset"]]),
        "",
        "## Belirsizlik",
        "",
        markdown_table(ci.loc[ci["method"].eq("conservative_envelope"), ["competition", "metric", "mean_difference", "ci_95_lower", "ci_95_upper", "reliable_improvement", "reliable_harm"]]),
        "",
        "## Karar",
        "",
        f"**{decision['decision']}**",
        "",
        f"Tam geçmiş seçimi: `{decision['full_history_candidate_key']}`. Bu sonuç production aktivasyonu değildir.",
    ]
    (output / "backtest_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Decision: {decision['decision']}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
