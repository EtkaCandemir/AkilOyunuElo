from __future__ import annotations

"""Nested research backtest for stronger Domestic Surprise effects.

Production inputs are loaded read-only.  Candidate seeds are rebased from the
active production seed map, which guarantees that the documented
theta=.40/cap=30/linear control is byte-identical to the served rating seed.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ao_elo.config import AOEuropeanEloConfig  # noqa: E402
from ao_elo.domestic_surprise_amplification import (  # noqa: E402
    EXPOSURE_FAMILIES,
    DomesticSurpriseAmplificationConfig,
    calculate_domestic_surprise_amplification,
    production_control_config,
)
from ao_elo.evaluation import dependency_robust_loss_difference_ci, schedule_adjusted_team_performance  # noqa: E402
from scripts.run_controlled_goal_progression_backtest import prepare_controlled_data  # noqa: E402
from scripts.run_current_model_evaluation import EvaluationArm, evaluate_arm, prediction_summary  # noqa: E402
from scripts.run_domestic_surprise_5y_backtest import build_domestic_history_features  # noqa: E402
from scripts.run_dynamic_core_calibration import expanding_folds  # noqa: E402
from scripts.run_final_robustness import load_team_season_identity, summarize_ranking  # noqa: E402
from scripts.run_stage_weighted_progression_backtest import (  # noqa: E402
    DOMESTIC_ADJUSTMENTS,
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


OUTPUT_ROOT = ROOT / "output" / "domestic_surprise_amplification_backtest_2018_2026"
CLUB_IDENTITY = ROOT / "data" / "club_identity" / "team_season_identity.csv"
THETAS = (0.40, 0.60, 0.80, 1.00, 1.25, 1.50, 1.75, 2.00)
DOMESTIC_CAPS = (30.0, 45.0, 60.0, 75.0, 100.0, 150.0)
VARIANCE_PENALTIES = (0.00, 0.25, 0.50, 0.75)
EPSILON = 1e-12


def candidate_key(config: DomesticSurpriseAmplificationConfig) -> str:
    return json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))


def config_fingerprint(config: DomesticSurpriseAmplificationConfig) -> str:
    return hashlib.sha256(candidate_key(config).encode("utf-8")).hexdigest()[:16]


def stage_one_candidates() -> tuple[DomesticSurpriseAmplificationConfig, ...]:
    candidates = tuple(
        DomesticSurpriseAmplificationConfig(theta=theta, domestic_cap=cap)
        for theta in THETAS
        for cap in DOMESTIC_CAPS
    )
    if len(candidates) != 48:
        raise AssertionError("Stage one must contain exactly 48 candidates")
    return candidates


def relative_loss_score(metrics: pd.Series, baseline: pd.Series) -> float:
    return 0.5 * (metrics["brier_1x2"] / baseline["brier_1x2"] - 1.0) + 0.5 * (
        metrics["log_loss_1x2"] / baseline["log_loss_1x2"] - 1.0
    )


def candidate_sort_key(row: pd.Series) -> tuple[float, float, float, int, float, str]:
    family_order = {family: index for index, family in enumerate(EXPOSURE_FAMILIES)}
    return (
        float(row["relative_loss_score"]),
        float(row["log_loss_1x2"]),
        float(row["mean_abs_effect"]),
        family_order[str(row["exposure_family"])],
        float(row["theta"]),
        str(row["candidate_key"]),
    )


def build_candidate_effects(
    features: pd.DataFrame,
    config: DomesticSurpriseAmplificationConfig,
    static_config: AOEuropeanEloConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in features.itertuples(index=False):
        history = [
            None
            if pd.isna(getattr(row, f"history_direct_t_minus_{offset}"))
            else float(getattr(row, f"history_direct_t_minus_{offset}"))
            for offset in range(5, 0, -1)
        ]
        current = float(row.current_direct_percentile) if bool(row.current_finish_eligible) and pd.notna(row.current_direct_percentile) else 0.0
        if not bool(row.current_finish_eligible):
            history = [None] * 5
        result = calculate_domestic_surprise_amplification(
            current_finish_score=current,
            historical_finish_scores=history,
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
                "candidate_key": candidate_key(config),
                "config_fingerprint": config_fingerprint(config),
                **asdict(config),
                "current_finish_score": current,
                "ao_first_elo_without_rebuilt_surprise": float(
                    row.baseline_ao_first_elo
                ),
                "effective_european_exposure": float(row.effective_european_exposure),
                "historical_mean": result.historical_mean,
                "historical_variance": result.historical_variance,
                "historical_volatility": result.historical_volatility,
                "normalized_volatility": result.normalized_volatility,
                "consistency_multiplier": result.consistency_multiplier,
                "history_seasons": result.history_seasons,
                "raw_surprise": result.raw_surprise,
                "effective_surprise": result.effective_surprise,
                "domestic_adjustment": result.domestic_adjustment,
                "exposure_weight": result.exposure_weight,
                "ao_first_elo_adjustment": result.ao_first_elo_adjustment,
                "final_effect_capped": result.final_effect_capped,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.duplicated(["season", "team_id"]).any():
        raise ValueError("Candidate effects contain duplicate season/team keys")
    return frame


def make_seed_map(
    production_seed: dict[tuple[str, int], float],
    production_effects: pd.DataFrame,
    candidate_effects: pd.DataFrame,
) -> dict[tuple[str, int], float]:
    control = production_effects.set_index(["season", "team_id"])["ao_first_elo_adjustment"]
    candidate = candidate_effects.set_index(["season", "team_id"])["ao_first_elo_adjustment"]
    if set(control.index) != set(production_seed) or set(candidate.index) != set(production_seed):
        raise ValueError("Candidate effects and production seed map must have identical keys")
    return {
        key: float(production_seed[key] - control.loc[key] + candidate.loc[key])
        for key in production_seed
    }


def metric_table(predictions: pd.DataFrame, evaluation, target: pd.DataFrame, identity: pd.DataFrame, seasons: set[str]) -> dict[str, float | int]:
    frame = predictions.loc[predictions["season"].isin(seasons)]
    ranking = aggregate_ranking(
        evaluation.same_season_ranking.loc[evaluation.same_season_ranking["season"].isin(seasons)]
    )
    all_ranking = ranking.loc[ranking["competition"].eq("ALL")].iloc[0]
    forward = summarize_ranking(
        evaluation.end_ratings,
        target,
        allowed_target_seasons=seasons,
        identity=identity,
    )
    forward_rows = forward.loc[forward["competition"].eq("ALL")]
    # A same-season replay has no later target season available.  Keep the
    # forward fields explicit rather than treating absent forward evidence as
    # a zero or silently leaking a future season into selection.
    forward_all = None if forward_rows.empty else forward_rows.iloc[0]
    return {
        **prediction_summary(frame),
        "same_season_spearman": float(all_ranking["ranking_score"]),
        "same_season_pairwise_accuracy": float(all_ranking["pairwise_accuracy"]),
        "forward_spearman": (
            float("nan") if forward_all is None else float(forward_all["ranking_score"])
        ),
        "forward_pairwise_accuracy": (
            float("nan") if forward_all is None else float(forward_all["pairwise_accuracy"])
        ),
    }


def effect_summary(effects: pd.DataFrame, seasons: set[str]) -> dict[str, float]:
    values = effects.loc[effects["season"].isin(seasons), "ao_first_elo_adjustment"].abs()
    return {
        "mean_abs_effect": float(values.mean()),
        "median_abs_effect": float(values.median()),
        "p90_abs_effect": float(values.quantile(0.90)),
        "p95_abs_effect": float(values.quantile(0.95)),
        "cap_hit_rate": float(effects.loc[effects["season"].isin(seasons), "final_effect_capped"].mean()),
    }


MINIMUM_MODAL_FOLD_SHARE = 0.50


def selection_stability(selections: pd.DataFrame) -> dict[str, object]:
    """How often did the folds agree on the same candidate?

    A surface this wide can always produce a per-fold winner. If the winners
    disagree, the pooled loss difference is a property of the search rather
    than of any parameter set that could be shipped, so the number that
    decides whether the run means anything is how often the folds land on the
    same configuration.
    """

    required = {"selected_candidate", "theta", "domestic_cap", "variance_penalty", "exposure_family"}
    missing = sorted(required - set(selections.columns))
    if missing:
        raise ValueError(f"selection stability needs at least one fold; missing {missing}")
    keys = selections["selected_candidate"].astype(str).tolist()
    if not keys:
        raise ValueError("selection stability needs at least one fold")
    modal = max(set(keys), key=keys.count)
    spans: dict[str, object] = {}
    for column in ("theta", "domestic_cap", "variance_penalty"):
        values = selections[column].astype(float)
        spans[f"{column}_min"] = float(values.min())
        spans[f"{column}_max"] = float(values.max())
    return {
        "folds": len(keys),
        "distinct_candidates": len(set(keys)),
        "modal_fold_share": keys.count(modal) / len(keys),
        "distinct_exposure_families": int(selections["exposure_family"].nunique()),
        **spans,
    }


def season_block_ranking_uncertainty(ranking: pd.DataFrame, *, samples: int = 10000) -> pd.DataFrame:
    """Season-block bootstrap for ranking deltas used as a harm veto.

    Every metric here is a ranking score where higher is better, so the
    candidate harms ranking when the difference is **negative**. That is the
    opposite of the loss convention in
    ``dependency_robust_loss_difference_ci``, where a positive difference is
    the harm. Applying the loss test to a ranking metric reports a reliable
    degradation as safe, which is what this function did before.
    """
    rng = np.random.default_rng(20260821)
    rows: list[dict[str, object]] = []
    metric_columns = [
        "same_season_spearman",
        "same_season_pairwise_accuracy",
        "forward_spearman",
        "forward_pairwise_accuracy",
    ]
    for metric in metric_columns:
        values = ranking[metric].dropna().to_numpy(dtype=float)
        if not len(values):
            rows.append({"metric": metric, "seasons": 0, "mean_difference": np.nan, "ci_95_lower": np.nan, "ci_95_upper": np.nan, "reliable_improvement": False, "reliable_harm": False})
            continue
        draws = rng.integers(0, len(values), size=(samples, len(values)))
        means = values[draws].mean(axis=1)
        lower, upper = np.quantile(means, (0.025, 0.975))
        rows.append({"metric": metric, "seasons": len(values), "mean_difference": float(values.mean()), "ci_95_lower": float(lower), "ci_95_upper": float(upper), "reliable_improvement": bool(lower > 0.0), "reliable_harm": bool(upper < 0.0)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Research-only Domestic Surprise amplification backtest")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument(
        "--baseline-source",
        choices=("stored", "rebuilt_current_contract"),
        default="stored",
        help="Use the frozen stored seed or a clean 0.65-contract rebuilt seed.",
    )
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples must be at least 100")

    contract_path = PRODUCTION_CONTRACT.resolve()
    contract_bytes = contract_path.read_bytes()
    contract = json.loads(contract_bytes)
    core, parameters = validate_production_contract(contract)
    static_config = AOEuropeanEloConfig.active()
    static_config.validate()
    if static_config.max_european_exposure != 0.65:
        raise AssertionError("Research must use active effective exposure cap 0.65")
    events = read_events(EVENTS_PATH)
    reserve, _ = load_reserve_data(STATIC_DATA_ROOT, EVENTS_PATH, static_config)
    datasets = prepare_controlled_data(reserve, events)
    seasons = tuple(data.season for data in datasets)
    folds = expanding_folds(seasons)
    if len(folds) != 6:
        raise ValueError(f"Expected six expanding folds, found {len(folds)}")
    target = schedule_adjusted_team_performance(events)
    identity = load_team_season_identity()
    stored_production_seed = load_domestic_adjustments(DOMESTIC_ADJUSTMENTS, datasets)
    xg_map = load_xg_map(XG_DATA, datasets)

    print("Rebuilding current-contract five-season Domestic Surprise features", flush=True)
    features, history_long, coverage = build_domestic_history_features(
        STATIC_DATA_ROOT, static_config, seasons, CLUB_IDENTITY
    )
    control_config = production_control_config()
    control_effects = build_candidate_effects(features, control_config, static_config)
    stored_effects = pd.read_csv(DOMESTIC_ADJUSTMENTS)[
        ["season", "team_id", "ao_first_elo_adjustment"]
    ].rename(columns={"ao_first_elo_adjustment": "stored_production_effect"})
    control_effects = control_effects.merge(
        stored_effects,
        on=["season", "team_id"],
        how="left",
        validate="one_to_one",
    )
    control_effects["stored_effect_difference"] = (
        control_effects["ao_first_elo_adjustment"]
        - control_effects["stored_production_effect"]
    )
    if control_effects["stored_production_effect"].isna().any():
        raise ValueError("Stored production Domestic Surprise artifact has missing keys")
    control_artifact_max_difference = float(
        control_effects["stored_effect_difference"].abs().max()
    )
    # The stored artifact was written under the previous European exposure cap.
    # Lowering the cap raises the domestic weight `1 - exposure` for every club
    # that sits at it, so the rebuilt control cannot equal the stored artifact
    # by construction and a bare equality check fails for a reason that has
    # nothing to do with this study. Split it: below the cap the weight is
    # untouched and the rebuild must reproduce the artifact exactly, and every
    # non-zero difference must be explained by a club-season at the cap.
    at_exposure_cap = control_effects["effective_european_exposure"].ge(
        static_config.max_european_exposure - 1e-9
    )
    below_cap_difference = control_effects.loc[
        ~at_exposure_cap, "stored_effect_difference"
    ].abs()
    control_artifact_max_difference_below_cap = (
        float(below_cap_difference.max()) if len(below_cap_difference) else 0.0
    )
    control_artifact_matches = control_artifact_max_difference_below_cap <= 1e-9
    migration_delta_confined_to_cap = bool(
        at_exposure_cap[
            control_effects["stored_effect_difference"].abs() > 1e-9
        ].all()
    )
    rebuilt_control_seed = {
        (str(row.season), int(row.team_id)): float(
            row.ao_first_elo_without_rebuilt_surprise
            + row.ao_first_elo_adjustment
        )
        for row in control_effects.itertuples(index=False)
    }
    if set(rebuilt_control_seed) != set(stored_production_seed):
        raise ValueError("Rebuilt and stored Domestic Surprise seed keys differ")
    seed_difference = pd.DataFrame(
        [
            {
                "season": season,
                "team_id": team_id,
                "stored_production_ao_first_elo": stored_production_seed[(season, team_id)],
                "rebuilt_current_contract_ao_first_elo": rebuilt_control_seed[(season, team_id)],
                "seed_difference": rebuilt_control_seed[(season, team_id)]
                - stored_production_seed[(season, team_id)],
            }
            for season, team_id in sorted(rebuilt_control_seed)
        ]
    )
    evaluation_seed = (
        stored_production_seed
        if args.baseline_source == "stored"
        else rebuilt_control_seed
    )
    control_seed = make_seed_map(evaluation_seed, control_effects, control_effects)
    if any(abs(control_seed[key] - evaluation_seed[key]) > 1e-12 for key in evaluation_seed):
        raise AssertionError("Control must reproduce the selected evaluation seed map")

    arm = EvaluationArm("DOMESTIC_SURPRISE_AMPLIFICATION", True, True, True, True, True)
    cache: dict[str, tuple[DomesticSurpriseAmplificationConfig, pd.DataFrame, object]] = {}

    def evaluate(config: DomesticSurpriseAmplificationConfig):
        key = candidate_key(config)
        if key not in cache:
            effects = build_candidate_effects(features, config, static_config)
            seed_map = make_seed_map(evaluation_seed, control_effects, effects)
            evaluation = evaluate_arm(
                datasets,
                arm,
                core=core,
                parameters=parameters,
                current_domestic=seed_map,
                baseline_domestic=seed_map,
                xg_map=xg_map,
                target=target,
            )
            cache[key] = (config, effects, evaluation)
        return cache[key]

    print("Evaluating production control", flush=True)
    evaluate(control_config)
    selection_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    selected_effects: list[pd.DataFrame] = []
    selected_predictions: list[pd.DataFrame] = []

    for fold, (train_tuple, test_season) in enumerate(folds, start=1):
        train = set(train_tuple)
        test = {test_season}
        control_eval = evaluate(control_config)[2]
        control_train = pd.Series(metric_table(control_eval.predictions, control_eval, target, identity, train))

        stage1_rows: list[dict[str, object]] = []
        for config in stage_one_candidates():
            _, effects, evaluation = evaluate(config)
            metrics = pd.Series(metric_table(evaluation.predictions, evaluation, target, identity, train))
            stage1_rows.append({**asdict(config), "candidate_key": candidate_key(config), **metrics.to_dict(), **effect_summary(effects, train)})
        stage1 = pd.DataFrame(stage1_rows)
        stage1["relative_loss_score"] = stage1.apply(lambda row: relative_loss_score(row, control_train), axis=1)
        stage1 = stage1.sort_values(by=list(stage1.columns), key=None) if False else stage1
        top8 = sorted(stage1.to_dict("records"), key=lambda row: candidate_sort_key(pd.Series(row)))[:8]

        stage2_configs = [
            DomesticSurpriseAmplificationConfig(
                theta=float(row["theta"]), domestic_cap=float(row["domestic_cap"]),
                variance_penalty=0.50, exposure_family=family
            )
            for row in top8 for family in EXPOSURE_FAMILIES
        ]
        stage2_rows: list[dict[str, object]] = []
        for config in stage2_configs:
            _, effects, evaluation = evaluate(config)
            metrics = pd.Series(metric_table(evaluation.predictions, evaluation, target, identity, train))
            stage2_rows.append({**asdict(config), "candidate_key": candidate_key(config), **metrics.to_dict(), **effect_summary(effects, train)})
        stage2 = pd.DataFrame(stage2_rows)
        stage2["relative_loss_score"] = stage2.apply(lambda row: relative_loss_score(row, control_train), axis=1)
        top5 = sorted(stage2.to_dict("records"), key=lambda row: candidate_sort_key(pd.Series(row)))[:5]

        stage3_configs = [
            DomesticSurpriseAmplificationConfig(
                theta=float(row["theta"]), domestic_cap=float(row["domestic_cap"]),
                variance_penalty=variance, exposure_family=str(row["exposure_family"])
            )
            for row in top5 for variance in VARIANCE_PENALTIES
        ]
        stage3_rows: list[dict[str, object]] = []
        for config in stage3_configs:
            _, effects, evaluation = evaluate(config)
            metrics = pd.Series(metric_table(evaluation.predictions, evaluation, target, identity, train))
            stage3_rows.append({**asdict(config), "candidate_key": candidate_key(config), **metrics.to_dict(), **effect_summary(effects, train)})
        stage3 = pd.DataFrame(stage3_rows)
        stage3["relative_loss_score"] = stage3.apply(lambda row: relative_loss_score(row, control_train), axis=1)
        selected_record = sorted(stage3.to_dict("records"), key=lambda row: candidate_sort_key(pd.Series(row)))[0]
        selected = DomesticSurpriseAmplificationConfig(
            theta=float(selected_record["theta"]), domestic_cap=float(selected_record["domestic_cap"]),
            variance_penalty=float(selected_record["variance_penalty"]),
            exposure_family=str(selected_record["exposure_family"]),
        )
        _, effects, evaluation = evaluate(selected)
        _, _, control_evaluation = evaluate(control_config)
        candidate_test = metric_table(evaluation.predictions, evaluation, target, identity, test)
        control_test = metric_table(control_evaluation.predictions, control_evaluation, target, identity, test)
        selection_rows.append({
            "fold": fold, "train_seasons": "|".join(train_tuple), "test_season": test_season,
            "stage1_candidates": len(stage1), "stage2_candidates": len(stage2), "stage3_candidates": len(stage3),
            "selected_candidate": candidate_key(selected), **asdict(selected),
            "train_relative_loss_score": float(selected_record["relative_loss_score"]),
        })
        for model, metric, effect, arm_evaluation in (
            ("CURRENT_PRODUCTION", control_test, control_effects, control_evaluation),
            ("AMPLIFIED_CANDIDATE", candidate_test, effects, evaluation),
        ):
            season_safety = arm_evaluation.season_metrics.loc[
                arm_evaluation.season_metrics["season"].eq(test_season)
            ].iloc[0]
            fold_rows.append(
                {
                    "fold": fold,
                    "test_season": test_season,
                    "model": model,
                    **metric,
                    **effect_summary(effect, test),
                    "maximum_match_zero_sum_error": float(
                        season_safety["maximum_match_zero_sum_error"]
                    ),
                    "season_power_conservation_error": float(
                        season_safety["season_power_conservation_error"]
                    ),
                }
            )
        test_effects = effects.loc[effects["season"].eq(test_season)].copy()
        test_effects.insert(0, "fold", fold)
        selected_effects.append(test_effects)
        candidate_predictions = evaluation.predictions.loc[evaluation.predictions["season"].eq(test_season)].copy()
        base_predictions = control_evaluation.predictions.loc[control_evaluation.predictions["season"].eq(test_season)].copy()
        paired = candidate_predictions.merge(
            base_predictions[["match_id", "brier_1x2", "log_loss_1x2"]], on="match_id", suffixes=("_candidate", "_baseline"), validate="one_to_one"
        )
        paired["fold"] = fold
        paired["candidate_key"] = candidate_key(selected)
        selected_predictions.append(paired)
        print(f"Fold {fold}/6 {test_season}: theta={selected.theta:g}, cap={selected.domestic_cap:g}, exposure={selected.exposure_family}, variance={selected.variance_penalty:g}", flush=True)

    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    folds_frame = pd.DataFrame(fold_rows)
    selections = pd.DataFrame(selection_rows)
    paired = pd.concat(selected_predictions, ignore_index=True)
    effects = pd.concat(selected_effects, ignore_index=True)
    candidates_out = pd.concat([
        pd.DataFrame({"fold": fold, "stage": "stage1", **stage1}) if False else pd.DataFrame()
        for fold in ()
    ]) if False else pd.DataFrame()

    # A compact full-history diagnostic uses the most frequent selected config.
    selected_configs = [
        DomesticSurpriseAmplificationConfig(
            theta=float(row.theta), domestic_cap=float(row.domestic_cap), variance_penalty=float(row.variance_penalty), exposure_family=str(row.exposure_family)
        ) for row in selections.itertuples(index=False)
    ]
    full_selected = max(set(selected_configs), key=selected_configs.count)
    _, full_effects, full_eval = evaluate(full_selected)
    _, _, control_eval = evaluate(control_config)
    all_unseen = set(fold[1] for fold in folds)
    full_candidate_metrics = metric_table(full_eval.predictions, full_eval, target, identity, all_unseen)
    full_control_metrics = metric_table(control_eval.predictions, control_eval, target, identity, all_unseen)

    uncertainty_rows: list[pd.DataFrame] = []
    for metric in ("brier_1x2", "log_loss_1x2"):
        sample = paired[["season", "match_id", "home_team_id", "away_team_id", "kickoff_utc", "tie_id"]].copy()
        sample["loss_difference"] = paired[f"{metric}_candidate"] - paired[f"{metric}_baseline"]
        result = dependency_robust_loss_difference_ci(sample, bootstrap_samples=args.bootstrap_samples)
        result.insert(0, "metric", metric)
        uncertainty_rows.append(result)
    uncertainty = pd.concat(uncertainty_rows, ignore_index=True)

    by_competition: list[dict[str, object]] = []
    for competition, frame in paired.groupby("competition", sort=True):
        for metric in ("brier_1x2", "log_loss_1x2"):
            by_competition.append({"competition": competition, "metric": metric, "matches": len(frame), "mean_difference": float((frame[f"{metric}_candidate"] - frame[f"{metric}_baseline"]).mean())})
    competition = pd.DataFrame(by_competition)
    current_rows = folds_frame.loc[folds_frame.model.eq("CURRENT_PRODUCTION")].set_index("fold")
    candidate_rows = folds_frame.loc[folds_frame.model.eq("AMPLIFIED_CANDIDATE")].set_index("fold")
    ranking = candidate_rows[["same_season_spearman", "same_season_pairwise_accuracy", "forward_spearman", "forward_pairwise_accuracy"]].subtract(current_rows[["same_season_spearman", "same_season_pairwise_accuracy", "forward_spearman", "forward_pairwise_accuracy"]]).reset_index()
    ranking.insert(1, "test_season", selections.set_index("fold").loc[ranking.fold, "test_season"].to_numpy())
    ranking_uncertainty = season_block_ranking_uncertainty(ranking)

    brier_ci = uncertainty.loc[uncertainty.metric.eq("brier_1x2")]
    log_ci = uncertainty.loc[uncertainty.metric.eq("log_loss_1x2")]
    brier_wins = int((candidate_rows.brier_1x2 < current_rows.brier_1x2 - EPSILON).sum())
    log_wins = int((candidate_rows.log_loss_1x2 < current_rows.log_loss_1x2 - EPSILON).sum())
    pooled_brier = float((paired.brier_1x2_candidate - paired.brier_1x2_baseline).mean())
    pooled_log = float((paired.log_loss_1x2_candidate - paired.log_loss_1x2_baseline).mean())
    conservative = uncertainty.loc[uncertainty["method"].eq("conservative_envelope")]
    reliable_improvement = bool(conservative["reliable_improvement"].any())
    reliable_harm = bool(conservative["reliable_harm"].any())
    reliable_ranking_harm = bool(ranking_uncertainty["reliable_harm"].any())
    segment_harm = bool((competition.groupby("competition").mean(numeric_only=True)["mean_difference"] > 0).any())
    stability = selection_stability(selections)
    selection_is_stable = bool(
        stability["modal_fold_share"] >= MINIMUM_MODAL_FOLD_SHARE
    )
    # Ranking harm and selection instability veto both promotion tiers. A layer
    # cannot be amplified "with monitoring" when the folds never agreed on what
    # to amplify, and it cannot be shipped while it reliably degrades the very
    # ranking the seed exists to produce.
    decision = (
        "KEEP_CURRENT"
        if args.baseline_source == "stored" and not control_artifact_matches
        else "PROMOTE_CANDIDATE" if pooled_brier < 0 and pooled_log < 0 and brier_wins >= 4 and log_wins >= 4 and reliable_improvement and not reliable_harm and not segment_harm and not reliable_ranking_harm and selection_is_stable
        else "AMPLIFY_WITH_MONITORING" if pooled_brier < 0 and pooled_log < 0 and not reliable_harm and not reliable_ranking_harm and selection_is_stable
        else "KEEP_CURRENT"
    )
    safety = pd.DataFrame([
        {"check": "production_contract_sha256_unchanged", "passed": hashlib.sha256(contract_path.read_bytes()).hexdigest() == hashlib.sha256(contract_bytes).hexdigest(), "observed": hashlib.sha256(contract_path.read_bytes()).hexdigest()},
        {"check": "rebuilt_control_matches_stored_artifact_below_exposure_cap", "passed": control_artifact_matches, "observed": control_artifact_max_difference_below_cap},
        {"check": "contract_migration_delta_confined_to_capped_exposure", "passed": migration_delta_confined_to_cap, "observed": control_artifact_max_difference},
        {"check": "fold_selection_modal_share_at_least_half", "passed": selection_is_stable, "observed": float(stability["modal_fold_share"])},
        {"check": "ranking_not_reliably_harmed", "passed": not reliable_ranking_harm, "observed": float(ranking_uncertainty["mean_difference"].min())},
        {"check": "evaluation_seed_matches_selected_baseline_source", "passed": True, "observed": args.baseline_source},
        {"check": "effects_within_plus_minus_75", "passed": bool(full_effects.ao_first_elo_adjustment.abs().max() <= 75 + EPSILON), "observed": float(full_effects.ao_first_elo_adjustment.abs().max())},
        {"check": "cap_hit_rate_at_most_10_percent", "passed": bool(full_effects.final_effect_capped.mean() <= .10 + EPSILON), "observed": float(full_effects.final_effect_capped.mean())},
        {"check": "no_sign_reversal", "passed": bool(((full_effects.raw_surprise * full_effects.ao_first_elo_adjustment) >= -EPSILON).all()), "observed": 0},
        {"check": "five_season_missing_history_zero", "passed": bool((full_effects.loc[full_effects.history_seasons.lt(5), "ao_first_elo_adjustment"].abs() <= EPSILON).all()), "observed": int(full_effects.history_seasons.lt(5).sum())},
        {"check": "power_zero_sum", "passed": bool(candidate_rows.maximum_match_zero_sum_error.max() <= 1e-9), "observed": float(candidate_rows.maximum_match_zero_sum_error.max())},
    ])
    distribution = full_effects.copy()
    distribution["abs_ao_first_elo_adjustment"] = distribution.ao_first_elo_adjustment.abs()
    distribution["exposure_band"] = pd.cut(distribution.effective_european_exposure, [-0.001,.20,.40,.65,1.001], labels=["0-.20", ".20-.40", ".40-.65", ">.65"])
    distribution_summary = distribution.groupby("exposure_band", observed=False).agg(team_seasons=("team_id", "size"), mean_abs_effect=("abs_ao_first_elo_adjustment", "mean"), p95_abs_effect=("abs_ao_first_elo_adjustment", lambda values: values.quantile(.95)), max_abs_effect=("abs_ao_first_elo_adjustment", "max")).reset_index()

    control_effects.to_csv(output / "baseline_effect_audit.csv", index=False)
    seed_difference.to_csv(output / "baseline_seed_reconciliation.csv", index=False)
    # The surface lists all evaluated unique candidates, with full six-season AO-core diagnostics.
    surface_rows = []
    for config, candidate_effects, evaluation in cache.values():
        metrics = metric_table(evaluation.predictions, evaluation, target, identity, all_unseen)
        surface_rows.append({**asdict(config), "candidate_key": candidate_key(config), **metrics, **effect_summary(candidate_effects, all_unseen)})
    surface = pd.DataFrame(surface_rows)
    surface["relative_loss_score"] = surface.apply(lambda row: relative_loss_score(row, pd.Series(full_control_metrics)), axis=1)
    surface.sort_values(["relative_loss_score", "log_loss_1x2", "mean_abs_effect"], inplace=True)
    surface.to_csv(output / "candidate_surface.csv", index=False)
    selections.to_csv(output / "fold_selections.csv", index=False)
    folds_frame.to_csv(output / "fold_results.csv", index=False)
    distribution.to_csv(output / "effect_distribution.csv", index=False)
    distribution_summary.to_csv(output / "exposure_ablation.csv", index=False)
    surface.groupby("variance_penalty", as_index=False).agg(candidates=("candidate_key", "count"), best_relative_loss_score=("relative_loss_score", "min"), minimum_brier=("brier_1x2", "min"), minimum_log_loss=("log_loss_1x2", "min")).to_csv(output / "variance_ablation.csv", index=False)
    competition.to_csv(output / "competition_summary.csv", index=False)
    ranking.to_csv(output / "ranking_summary.csv", index=False)
    ranking_uncertainty.to_csv(output / "ranking_uncertainty.csv", index=False)
    uncertainty.to_csv(output / "dependency_uncertainty.csv", index=False)
    safety.to_csv(output / "safety_audit.csv", index=False)
    selected = {
        "decision": decision, "production_change": False,
        "baseline_source": args.baseline_source,
        "most_frequent_fold_candidate": asdict(full_selected),
        "fold_brier_wins": brier_wins, "fold_log_loss_wins": log_wins,
        "pooled_brier_difference": pooled_brier, "pooled_log_loss_difference": pooled_log,
        "loss_conservative_ci_reliable_improvement": reliable_improvement,
        "loss_conservative_ci_reliable_harm": reliable_harm,
        "ranking_reliable_harm": reliable_ranking_harm,
        "ranking_harmed_metrics": sorted(
            ranking_uncertainty.loc[ranking_uncertainty["reliable_harm"], "metric"]
        ),
        "selection_stability": stability,
        "selection_is_stable": selection_is_stable,
        "rebuilt_control_matches_stored_artifact_below_exposure_cap": control_artifact_matches,
        "rebuilt_control_max_effect_difference_below_cap": control_artifact_max_difference_below_cap,
        "contract_migration_delta_confined_to_capped_exposure": migration_delta_confined_to_cap,
        "contract_migration_max_effect_difference": control_artifact_max_difference,
        "current_ao_core": full_control_metrics, "candidate_ao_core": full_candidate_metrics,
        "served_ml_poisson": "NOT_REOPTIMIZED: candidate selection and promotion evidence use AO_CORE_1X2; served ML/Poisson requires a separate frozen-artifact replay.",
        "contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
    }
    (output / "selected_candidate.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    report = [
        "# Domestic Surprise Amplification Backtest",
        "", f"Decision: **{decision}**. Production contract was not changed.", "",
        "## Research contract", "",
        "All team-season inputs were rebuilt with `AOEuropeanEloConfig.active()` and effective European exposure cap `0.65`. The active control is theta `0.40`, variance penalty `0.50`, domestic cap `30`, linear exposure, and final effect cap `±75`.", "",
        "## Control-artifact audit", "",
        "The stored production Domestic Surprise artifact was written under the previous European exposure cap. Lowering the cap raises the domestic weight `1 - exposure` for every club-season sitting at it, so the rebuilt control cannot equal the stored artifact by construction. The audit therefore separates the two questions.", "",
        f"- Below the cap, where the domestic weight is untouched, the rebuild reproduces the stored artifact to `{control_artifact_max_difference_below_cap:.2e}` Elo.",
        f"- Every non-zero difference sits at the cap: `{migration_delta_confined_to_cap}`. The largest is `{control_artifact_max_difference:.6f}` Elo, the arithmetic consequence of the weight moving from `1 - 0.85` to `1 - {static_config.max_european_exposure:g}` on a club at the `±30` domestic cap.", "",
        f"This run uses `{args.baseline_source}` as its seed baseline. `baseline_seed_reconciliation.csv` records the full migration delta without overwriting production.", "",
        "## AO Core pooled result", "",
        f"- Brier difference: `{pooled_brier:+.6f}`", f"- Log-loss difference: `{pooled_log:+.6f}`", f"- Fold wins: Brier `{brier_wins}/6`, log-loss `{log_wins}/6`.", "",
        "## Ranking", "",
        "Ranking scores are higher-is-better, so the candidate harms ranking when the difference is negative. This is the opposite of the loss convention and the veto is evaluated accordingly.", "",
        f"- Reliably harmed metrics: `{sorted(ranking_uncertainty.loc[ranking_uncertainty['reliable_harm'], 'metric']) or 'none'}`", "",
        "## Selection stability", "",
        "A surface this wide can always produce a per-fold winner. If the winners disagree there is no parameter set to ship, whatever the pooled loss says.", "",
        f"- Folds: `{stability['folds']}`, distinct selected candidates: `{stability['distinct_candidates']}`, modal fold share: `{stability['modal_fold_share']:.3f}` (gate `>= {MINIMUM_MODAL_FOLD_SHARE:.2f}`).",
        f"- theta spanned `{stability['theta_min']:g}`-`{stability['theta_max']:g}`, domestic cap `{stability['domestic_cap_min']:g}`-`{stability['domestic_cap_max']:g}`, variance penalty `{stability['variance_penalty_min']:g}`-`{stability['variance_penalty_max']:g}`, across `{stability['distinct_exposure_families']}` exposure families.", "",
        "## Served 1X2 boundary", "",
        "The served ML/Poisson model is intentionally not re-trained or re-selected in this study. Its historical feature/artifact replay must be candidate-seed-aware before this change can be promoted as a served-prediction change; this script therefore treats AO Core results as the valid evidence line rather than inventing a proxy.", "",
        "## Outputs", "",
        "See `candidate_surface.csv`, `fold_selections.csv`, `fold_results.csv`, `effect_distribution.csv`, `competition_summary.csv`, `ranking_summary.csv`, `ranking_uncertainty.csv`, `dependency_uncertainty.csv`, and `safety_audit.csv`.",
    ]
    (output / "backtest_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Decision: {decision}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
