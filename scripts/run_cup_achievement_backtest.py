from __future__ import annotations

"""Walk-forward test of a generalized domestic cup contribution.

The active model combines the league finish and the domestic cup with a
maximum, so the cup acts as a floor. Any cup winner whose league finish
already scores above the cup base receives nothing for the trophy, and the
double bonus fires only for the champion-and-cup pair. This script scores the
one-parameter generalization

```text
Achievement = min(cap, max(L, C) + weight * min(L, C))
```

against the active model on both axes the repository gates on:

1. `RATING`      the season-start seed against the realized European season,
   measured with the leave-team-out schedule-adjusted target.
2. `PREDICTION`  the full dynamic replay through the production kernel, with
   standard three-class Brier and log-loss.

Selection is nested: the weight for each test season is chosen from the
training seasons only. Prespecified fixed-weight arms are reported alongside
so a plateau is visible rather than a single tuned point.

This script changes no production parameter. It writes evidence only.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (str(SRC), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from ao_elo import compute_ao_first_elo_from_csv  # noqa: E402
from ao_elo.config import AOEuropeanEloConfig  # noqa: E402
from ao_elo.cup_achievement import (  # noqa: E402
    CupContributionConfig,
    achievement_delta_to_ao_first_elo,
    candidate_weights,
    champion_equivalent_weight,
    generalized_domestic_achievement,
    load_static_achievement_inputs,
)
from ao_elo.evaluation import (  # noqa: E402
    dependency_robust_loss_difference_ci,
    schedule_adjusted_team_performance,
)
from scripts.run_controlled_goal_progression_backtest import (  # noqa: E402
    prepare_controlled_data,
)
from scripts.run_current_model_evaluation import (  # noqa: E402
    ArmEvaluation,
    EvaluationArm,
    evaluate_arm,
)
from scripts.run_dynamic_core_calibration import expanding_folds  # noqa: E402
from scripts.run_external_ranking_comparison_2025_26 import (  # noqa: E402
    pairwise_accuracy,
)
from scripts.run_stage_weighted_progression_backtest import (  # noqa: E402
    DOMESTIC_ADJUSTMENTS,
    DYNAMIC_MANIFEST,
    EVENTS_PATH,
    PRODUCTION_CONTRACT,
    STATIC_DATA_ROOT,
    XG_DATA,
    load_domestic_adjustments,
    load_xg_map,
    validate_production_contract,
)
from scripts.run_v2_achievement_reserve_calibration import load_reserve_data  # noqa: E402
from scripts.run_v2_evaluation_upgrade import read_events  # noqa: E402


DEFAULT_OUTPUT = ROOT / "output" / "cup_achievement_backtest_2018_2026"
SEASON_DIRECTORY = re.compile(r"^\d{4}-\d{2}$")
BASELINE_ARM = "CURRENT_PRODUCTION"
NESTED_ARM = "NESTED_WALK_FORWARD"
BOOTSTRAP_SAMPLES = 4000
BOOTSTRAP_SEED = 20260818
SEED_TOLERANCE = 1e-9


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score a weighted domestic cup contribution against the active "
            "max-based achievement rule on two walk-forward axes"
        )
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--static-data-root", type=Path, default=STATIC_DATA_ROOT)
    arguments = parser.parse_args()

    output_root = arguments.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    production = json.loads(PRODUCTION_CONTRACT.read_text(encoding="utf-8"))
    core, parameters = validate_production_contract(production)
    dynamic = json.loads(DYNAMIC_MANIFEST.read_text(encoding="utf-8"))
    static_config = AOEuropeanEloConfig(**dynamic["static_config"])
    static_config.validate()

    events = read_events(EVENTS_PATH)
    reserve, _ = load_reserve_data(STATIC_DATA_ROOT, EVENTS_PATH, static_config)
    datasets = prepare_controlled_data(reserve, events)
    seasons = tuple(data.season for data in datasets)
    folds = expanding_folds(seasons)
    evaluation_seasons = {test for _, test in folds}
    target = schedule_adjusted_team_performance(events)
    xg_map = load_xg_map(XG_DATA, datasets)

    production_seeds = load_domestic_adjustments(DOMESTIC_ADJUSTMENTS, datasets)
    statics = load_static_achievement_inputs(
        arguments.static_data_root, static_config, seasons
    )
    weights = candidate_weights()
    equivalent = champion_equivalent_weight(static_config)

    seed_frame = build_seed_frame(statics, production_seeds, static_config, weights)
    gates = seed_validation_gates(seed_frame, production_seeds, weights)

    arms = {BASELINE_ARM: production_seeds}
    for weight in weights:
        arms[weight_arm_name(weight)] = seed_map_for_weight(seed_frame, weight)

    evaluations: dict[str, ArmEvaluation] = {}
    for name, seed_map in arms.items():
        print(f"Replaying {name}", flush=True)
        evaluations[name] = evaluate_arm(
            datasets,
            EvaluationArm(name, True, True, True, True, True),
            core=core,
            parameters=parameters,
            current_domestic=seed_map,
            baseline_domestic=seed_map,
            xg_map=xg_map,
            target=target,
        )

    rating = rating_axis(
        seed_frame,
        production_seeds,
        target,
        weights,
        evaluation_seasons,
        samples=int(arguments.bootstrap_samples),
    )
    prediction = prediction_axis(
        evaluations,
        weights,
        evaluation_seasons,
        samples=int(arguments.bootstrap_samples),
    )
    nested = nested_selection(prediction["season_metrics"], folds, weights)
    affected = affected_population(seed_frame, weights)

    rating["summary"].to_csv(output_root / "rating_summary.csv", index=False)
    rating["comparisons"].to_csv(output_root / "rating_comparisons.csv", index=False)
    rating["season_detail"].to_csv(output_root / "rating_season_detail.csv", index=False)
    prediction["summary"].to_csv(
        output_root / "prediction_summary.csv", index=False
    )
    prediction["season_metrics"].to_csv(
        output_root / "prediction_season_metrics.csv", index=False
    )
    prediction["fold_summary"].to_csv(
        output_root / "prediction_fold_summary.csv", index=False
    )
    prediction["uncertainty"].to_csv(
        output_root / "prediction_uncertainty.csv", index=False
    )
    nested.to_csv(output_root / "nested_fold_selections.csv", index=False)
    affected.to_csv(output_root / "affected_population.csv", index=False)
    seed_frame.to_csv(output_root / "seed_impact.csv", index=False)
    gates.to_csv(output_root / "validation_gates.csv", index=False)

    manifest = build_manifest(
        rating,
        prediction,
        nested,
        affected,
        gates,
        weights,
        equivalent,
        seasons,
        evaluation_seasons,
    )
    (output_root / "backtest_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(
        output_root / "backtest_report.md",
        rating,
        prediction,
        nested,
        affected,
        gates,
        manifest,
    )

    print(f"Wrote cup achievement backtest to {output_root}")
    print(gates.to_string(index=False))
    print()
    print(rating["summary"].to_string(index=False))
    print()
    print(prediction["summary"].to_string(index=False))


# ---------------------------------------------------------------------------
# static inputs and candidate seeds
# ---------------------------------------------------------------------------


def build_seed_frame(
    statics: tuple[tuple[str, pd.DataFrame], ...],
    production_seeds: dict[tuple[str, int], float],
    config: AOEuropeanEloConfig,
    weights: tuple[float, ...],
) -> pd.DataFrame:
    """Return one row per team-season with the seed under every weight."""

    rows: list[dict[str, object]] = []
    for season, frame in statics:
        for row in frame.itertuples(index=False):
            key = (season, int(row.team_id))
            if key not in production_seeds:
                continue
            baseline_score = float(row.domestic_achievement_score)
            record: dict[str, object] = {
                "season": season,
                "team_id": int(row.team_id),
                "team_name": str(row.team_name),
                "is_cup_winner": bool(row.is_cup_winner),
                "is_league_champion": bool(row.is_league_champion),
                "league_strength": float(row.league_strength),
                "effective_european_exposure": float(
                    row.effective_european_exposure
                ),
                "production_seed": float(production_seeds[key]),
                "baseline_achievement": baseline_score,
            }
            for weight in weights:
                candidate = generalized_domestic_achievement(
                    row.domestic_position,
                    row.league_team_count,
                    row.is_league_champion,
                    row.is_cup_winner,
                    config,
                    CupContributionConfig(weight),
                )
                delta = (
                    float(candidate.domestic_achievement_score) - baseline_score
                )
                seed_delta = achievement_delta_to_ao_first_elo(
                    delta,
                    float(row.league_strength),
                    float(row.effective_european_exposure),
                    config,
                )
                record[f"achievement_w{weight_key(weight)}"] = float(
                    candidate.domestic_achievement_score
                )
                record[f"seed_delta_w{weight_key(weight)}"] = seed_delta
                record[f"seed_w{weight_key(weight)}"] = (
                    float(production_seeds[key]) + seed_delta
                )
            rows.append(record)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("Seed frame is empty")
    return frame


def weight_key(weight: float) -> str:
    """Dot-free suffix so `itertuples` keeps the column name addressable."""

    return f"{weight:g}".replace(".", "_")


def weight_arm_name(weight: float) -> str:
    return f"CUP_WEIGHT_{weight_key(weight)}"


def seed_map_for_weight(
    seed_frame: pd.DataFrame, weight: float
) -> dict[tuple[str, int], float]:
    column = f"seed_w{weight_key(weight)}"
    return {
        (str(row.season), int(row.team_id)): float(getattr(row, column))
        for row in seed_frame.itertuples(index=False)
    }


def seed_validation_gates(
    seed_frame: pd.DataFrame,
    production_seeds: dict[tuple[str, int], float],
    weights: tuple[float, ...],
) -> pd.DataFrame:
    """Fail loudly when the candidate seeds break a structural guarantee."""

    rows = []
    covered = {
        (str(row.season), int(row.team_id))
        for row in seed_frame.itertuples(index=False)
    }
    rows.append(
        {
            "gate": "seed_coverage",
            "passed": bool(covered == set(production_seeds)),
            "observed": len(covered),
            "requirement": "Every active team-season has a candidate seed",
        }
    )

    non_winner = seed_frame.loc[~seed_frame["is_cup_winner"]]
    worst_inert = 0.0
    for weight in weights:
        column = f"seed_delta_w{weight_key(weight)}"
        worst_inert = max(worst_inert, float(non_winner[column].abs().max()))
    rows.append(
        {
            "gate": "inert_outside_cup_winners",
            "passed": bool(worst_inert <= SEED_TOLERANCE),
            "observed": worst_inert,
            "requirement": "Teams without a domestic cup keep the production seed",
        }
    )

    champion_gap = 0.0
    champion_cup = seed_frame.loc[
        seed_frame["is_cup_winner"] & seed_frame["is_league_champion"]
    ]
    if not champion_cup.empty:
        champion_gap = float(champion_cup["seed_delta_w0"].max())
    rows.append(
        {
            "gate": "zero_weight_removes_double_bonus",
            "passed": bool(champion_gap <= SEED_TOLERANCE),
            "observed": champion_gap,
            "requirement": "weight=0 cannot raise a champion-and-cup seed",
        }
    )

    monotone = True
    winners = seed_frame.loc[seed_frame["is_cup_winner"]]
    for left, right in zip(weights, weights[1:]):
        difference = (
            winners[f"seed_w{weight_key(right)}"] - winners[f"seed_w{weight_key(left)}"]
        ).min()
        if difference < -SEED_TOLERANCE:
            monotone = False
    rows.append(
        {
            "gate": "monotone_in_weight",
            "passed": bool(monotone),
            "observed": bool(monotone),
            "requirement": "Cup-winner seeds never fall as the weight rises",
        }
    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# axis 1: season-start rating
# ---------------------------------------------------------------------------


def rating_axis(
    seed_frame: pd.DataFrame,
    production_seeds: dict[tuple[str, int], float],
    target: pd.DataFrame,
    weights: tuple[float, ...],
    evaluation_seasons: set[str],
    *,
    samples: int,
) -> dict[str, object]:
    """Score every candidate seed against realized season performance."""

    scored = target.copy()
    scored["team_id"] = scored["team_id"].astype(int)
    joined = seed_frame.merge(scored, on=["season", "team_id"], how="inner")
    joined = joined.loc[joined["season"].isin(evaluation_seasons)].copy()
    if joined.empty:
        raise ValueError("Rating axis produced no scored teams")

    columns = {BASELINE_ARM: "production_seed"}
    columns.update({weight_arm_name(w): f"seed_w{weight_key(w)}" for w in weights})

    detail_rows = []
    summary_rows = []
    for name, column in columns.items():
        season_stats = []
        for season, group in joined.groupby("season"):
            values = group[column].to_numpy(float)
            realized = group["schedule_adjusted_score"].to_numpy(float)
            statistic = float(spearmanr(values, realized).statistic)
            accuracy = pairwise_accuracy(values, realized)
            season_stats.append((len(group), statistic, accuracy))
            detail_rows.append(
                {
                    "arm": name,
                    "season": season,
                    "teams": int(len(group)),
                    "spearman": statistic,
                    "pairwise_accuracy": accuracy,
                }
            )
        weight_total = sum(count for count, _, _ in season_stats)
        summary_rows.append(
            {
                "arm": name,
                "seasons": int(len(season_stats)),
                "team_seasons": int(weight_total),
                "weighted_spearman": float(
                    sum(count * value for count, value, _ in season_stats)
                    / weight_total
                ),
                "weighted_pairwise_accuracy": float(
                    sum(count * value for count, _, value in season_stats)
                    / weight_total
                ),
            }
        )

    comparison_rows = []
    for name, column in columns.items():
        if name == BASELINE_ARM:
            continue
        difference, lower, upper = pooled_paired_spearman_ci(
            joined,
            column,
            "production_seed",
            samples=samples,
        )
        comparison_rows.append(
            {
                "comparison": f"{name}_minus_{BASELINE_ARM}",
                "team_seasons": int(len(joined)),
                "spearman_difference": difference,
                "ci_95_lower": lower,
                "ci_95_upper": upper,
                "reliable_improvement": bool(lower > 0.0),
                "reliable_harm": bool(upper < 0.0),
            }
        )

    return {
        "summary": pd.DataFrame(summary_rows),
        "comparisons": pd.DataFrame(comparison_rows),
        "season_detail": pd.DataFrame(detail_rows),
    }


def pooled_paired_spearman_ci(
    joined: pd.DataFrame,
    candidate_column: str,
    baseline_column: str,
    *,
    samples: int,
) -> tuple[float, float, float]:
    """Bootstrap the season-weighted Spearman difference by resampling teams.

    Teams are resampled inside their own season so the season structure and
    the paired comparison are both preserved.
    """

    seasons = sorted(joined["season"].unique())
    blocks = [joined.loc[joined["season"].eq(season)] for season in seasons]
    observed = weighted_spearman_difference(
        blocks, candidate_column, baseline_column
    )
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(samples, dtype=float)
    for index in range(samples):
        resampled = []
        for block in blocks:
            positions = generator.integers(0, len(block), len(block))
            resampled.append(block.iloc[positions])
        draws[index] = weighted_spearman_difference(
            resampled, candidate_column, baseline_column
        )
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return observed, float("nan"), float("nan")
    lower, upper = np.percentile(finite, [2.5, 97.5])
    return observed, float(lower), float(upper)


def weighted_spearman_difference(
    blocks: list[pd.DataFrame],
    candidate_column: str,
    baseline_column: str,
) -> float:
    total = 0
    accumulated = 0.0
    for block in blocks:
        realized = block["schedule_adjusted_score"].to_numpy(float)
        if len(block) < 3 or np.all(realized == realized[0]):
            continue
        candidate = spearmanr(
            block[candidate_column].to_numpy(float), realized
        ).statistic
        baseline = spearmanr(
            block[baseline_column].to_numpy(float), realized
        ).statistic
        if not (np.isfinite(candidate) and np.isfinite(baseline)):
            continue
        accumulated += len(block) * (float(candidate) - float(baseline))
        total += len(block)
    if total == 0:
        return float("nan")
    return accumulated / total


# ---------------------------------------------------------------------------
# axis 2: dynamic replay prediction quality
# ---------------------------------------------------------------------------


def prediction_axis(
    evaluations: dict[str, ArmEvaluation],
    weights: tuple[float, ...],
    evaluation_seasons: set[str],
    *,
    samples: int,
) -> dict[str, pd.DataFrame]:
    summary_rows = []
    season_rows = []
    for name, evaluation in evaluations.items():
        frame = evaluation.predictions
        scored = frame.loc[frame["season"].isin(evaluation_seasons)]
        summary_rows.append(
            {
                "arm": name,
                "matches": int(len(scored)),
                "brier_1x2": float(scored["brier_1x2"].mean()),
                "log_loss_1x2": float(scored["log_loss_1x2"].mean()),
                "accuracy_1x2": float(
                    scored["predicted_class"].eq(scored["actual_class"]).mean()
                ),
            }
        )
        # Selection needs every season, including the early ones that are only
        # ever training data, so this loop deliberately walks the full frame
        # while the pooled summary above stays on the evaluation seasons.
        for season, group in frame.groupby("season"):
            season_rows.append(
                {
                    "arm": name,
                    "season": season,
                    "is_evaluation_season": bool(season in evaluation_seasons),
                    "matches": int(len(group)),
                    "brier_1x2": float(group["brier_1x2"].mean()),
                    "log_loss_1x2": float(group["log_loss_1x2"].mean()),
                    "accuracy_1x2": float(
                        group["predicted_class"].eq(group["actual_class"]).mean()
                    ),
                }
            )
    summary = pd.DataFrame(summary_rows)
    season_metrics = pd.DataFrame(season_rows)

    evaluated = season_metrics.loc[season_metrics["is_evaluation_season"]]
    baseline_seasons = evaluated.loc[
        evaluated["arm"].eq(BASELINE_ARM)
    ].set_index("season")
    fold_rows = []
    for name in evaluations:
        if name == BASELINE_ARM:
            continue
        arm_seasons = evaluated.loc[evaluated["arm"].eq(name)].set_index("season")
        fold_rows.append(
            {
                "arm": name,
                "brier_fold_wins": int(
                    sum(
                        arm_seasons.loc[season, "brier_1x2"]
                        < baseline_seasons.loc[season, "brier_1x2"]
                        for season in arm_seasons.index
                    )
                ),
                "log_loss_fold_wins": int(
                    sum(
                        arm_seasons.loc[season, "log_loss_1x2"]
                        < baseline_seasons.loc[season, "log_loss_1x2"]
                        for season in arm_seasons.index
                    )
                ),
                "folds": int(len(arm_seasons)),
            }
        )
    fold_summary = pd.DataFrame(fold_rows)

    baseline_frame = evaluations[BASELINE_ARM].predictions
    baseline_scored = baseline_frame.loc[
        baseline_frame["season"].isin(evaluation_seasons)
    ]
    uncertainty_blocks = []
    for name, evaluation in evaluations.items():
        if name == BASELINE_ARM:
            continue
        candidate = evaluation.predictions
        candidate_scored = candidate.loc[
            candidate["season"].isin(evaluation_seasons)
        ]
        paired = candidate_scored.merge(
            baseline_scored[["match_id", "brier_1x2", "log_loss_1x2"]],
            on="match_id",
            suffixes=("_candidate", "_baseline"),
            validate="one_to_one",
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
            block = dependency_robust_loss_difference_ci(
                sample,
                bootstrap_samples=samples,
                seed=BOOTSTRAP_SEED,
            )
            block.insert(0, "arm", name)
            block.insert(1, "metric", metric)
            uncertainty_blocks.append(block)

    return {
        "summary": summary,
        "season_metrics": season_metrics,
        "fold_summary": fold_summary,
        "uncertainty": pd.concat(uncertainty_blocks, ignore_index=True),
    }


def nested_selection(
    season_metrics: pd.DataFrame,
    folds: tuple[tuple[tuple[str, ...], str], ...],
    weights: tuple[float, ...],
) -> pd.DataFrame:
    """Select the weight on training seasons only, then read the test season."""

    rows = []
    for index, (train_seasons, test_season) in enumerate(folds, start=1):
        best_weight = None
        best_score = float("inf")
        for weight in weights:
            arm = weight_arm_name(weight)
            train = season_metrics.loc[
                season_metrics["arm"].eq(arm)
                & season_metrics["season"].isin(train_seasons)
            ]
            if train.empty:
                continue
            score = float(
                np.average(train["brier_1x2"], weights=train["matches"])
            )
            if score < best_score - 1e-15:
                best_score = score
                best_weight = weight
        if best_weight is None:
            raise ValueError(
                f"Fold {index} has no training metrics for any candidate weight"
            )
        selected_arm = weight_arm_name(best_weight)
        test = season_metrics.loc[
            season_metrics["arm"].eq(selected_arm)
            & season_metrics["season"].eq(test_season)
        ]
        baseline = season_metrics.loc[
            season_metrics["arm"].eq(BASELINE_ARM)
            & season_metrics["season"].eq(test_season)
        ]
        rows.append(
            {
                "fold": index,
                "train_seasons": "|".join(train_seasons),
                "test_season": test_season,
                "selected_weight": best_weight,
                "train_brier_1x2": best_score,
                "test_matches": int(test["matches"].iloc[0]),
                "test_brier_1x2": float(test["brier_1x2"].iloc[0]),
                "baseline_brier_1x2": float(baseline["brier_1x2"].iloc[0]),
                "test_brier_delta": float(
                    test["brier_1x2"].iloc[0] - baseline["brier_1x2"].iloc[0]
                ),
                "test_log_loss_delta": float(
                    test["log_loss_1x2"].iloc[0]
                    - baseline["log_loss_1x2"].iloc[0]
                ),
            }
        )
    return pd.DataFrame(rows)


def affected_population(
    seed_frame: pd.DataFrame, weights: tuple[float, ...]
) -> pd.DataFrame:
    """Report how many teams the layer can actually move."""

    rows = []
    for weight in weights:
        column = f"seed_delta_w{weight_key(weight)}"
        moved = seed_frame.loc[seed_frame[column].abs() > SEED_TOLERANCE]
        rows.append(
            {
                "weight": weight,
                "team_seasons": int(len(seed_frame)),
                "cup_winners": int(seed_frame["is_cup_winner"].sum()),
                "moved_team_seasons": int(len(moved)),
                "moved_share": float(len(moved) / len(seed_frame)),
                "mean_abs_seed_delta": float(moved[column].abs().mean())
                if len(moved)
                else 0.0,
                "max_abs_seed_delta": float(moved[column].abs().max())
                if len(moved)
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def build_manifest(
    rating: dict[str, object],
    prediction: dict[str, pd.DataFrame],
    nested: pd.DataFrame,
    affected: pd.DataFrame,
    gates: pd.DataFrame,
    weights: tuple[float, ...],
    equivalent_weight: float,
    seasons: tuple[str, ...],
    evaluation_seasons: set[str],
) -> dict[str, object]:
    summary = prediction["summary"].set_index("arm")
    rating_summary = rating["summary"].set_index("arm")
    envelope = prediction["uncertainty"].loc[
        prediction["uncertainty"]["method"].eq("conservative_envelope")
    ]
    return {
        "layer": "GENERALIZED_DOMESTIC_CUP_CONTRIBUTION",
        "family": "min(cap, max(L,C) + weight * min(L,C))",
        "changes_production_parameters": False,
        "rating_feedback": False,
        "development_window": f"{seasons[0]}-{seasons[-1]}",
        "evaluation_seasons": sorted(evaluation_seasons),
        "candidate_weights": list(weights),
        "champion_equivalent_weight": float(equivalent_weight),
        "all_validation_gates_passed": bool(gates["passed"].all()),
        "baseline_brier_1x2": float(summary.loc[BASELINE_ARM, "brier_1x2"]),
        "baseline_weighted_spearman": float(
            rating_summary.loc[BASELINE_ARM, "weighted_spearman"]
        ),
        "best_prediction_arm": str(summary["brier_1x2"].idxmin()),
        "best_rating_arm": str(rating_summary["weighted_spearman"].idxmax()),
        "nested_selected_weights": [
            float(value) for value in nested["selected_weight"]
        ],
        "nested_pooled_brier_delta": float(
            np.average(nested["test_brier_delta"], weights=nested["test_matches"])
        )
        if not nested.empty
        else None,
        # The repository gates on the conservative envelope, so a single
        # optimistic clustering view must never be reported as the verdict.
        "conservative_envelope_reliable_improvement": bool(
            envelope["reliable_improvement"].any()
        ),
        "conservative_envelope_reliable_harm": bool(
            envelope["reliable_harm"].any()
        ),
        "any_clustering_view_reliable_improvement": bool(
            prediction["uncertainty"]["reliable_improvement"].any()
        ),
        "best_fold_gate_arms": sorted(
            str(row.arm)
            for row in prediction["fold_summary"].itertuples(index=False)
            if int(row.brier_fold_wins) == int(row.folds)
            and int(row.log_loss_fold_wins) == int(row.folds)
        ),
        "any_reliable_rating_improvement": bool(
            rating["comparisons"]["reliable_improvement"].any()
        ),
        "any_reliable_rating_harm": bool(
            rating["comparisons"]["reliable_harm"].any()
        ),
        "max_moved_share": float(affected["moved_share"].max()),
    }


def write_report(
    path: Path,
    rating: dict[str, object],
    prediction: dict[str, pd.DataFrame],
    nested: pd.DataFrame,
    affected: pd.DataFrame,
    gates: pd.DataFrame,
    manifest: dict[str, object],
) -> None:
    lines = [
        "# Genellestirilmis Yerel Kupa Katkisi Backtest",
        "",
        "Aktif model lig ve kupa basarisini `max` ile birlestirir; bu kupayi bir",
        "*taban* yapar, katki yapmaz. Lig siralamasi kupa tabaninin uzerinde olan",
        "her kupa sahibi kupasindan sifir kredi alir ve duble bonusu yalniz",
        "sampiyon-ve-kupa ciftinde calisir. Bu kosu su tek parametreli",
        "genellestirmeyi olcer:",
        "",
        "```text",
        "Achievement = min(cap, max(L, C) + weight * min(L, C))",
        "```",
        "",
        f"- Sampiyon-esdeger agirlik: `{manifest['champion_equivalent_weight']:.6f}`",
        "  (mevcut duble bonus buyuklugunu korur, ayni mantigi herkese genisletir).",
        f"- Aday agirliklar: `{manifest['candidate_weights']}`.",
        f"- Gelistirme penceresi: `{manifest['development_window']}`.",
        f"- Degerlendirme sezonlari: `{', '.join(manifest['evaluation_seasons'])}`.",
        "- Bu kosu production parametresi degistirmez.",
        "",
        "## Yapisal dogrulama kapilari",
        "",
        "```text",
        gates.to_string(index=False),
        "```",
        "",
        "## Etkilenen kutle",
        "",
        "Katman kupa sahibi olmayan takimda tanim geregi inerttir. Gercekten",
        "hareket eden takim-sezon sayisi:",
        "",
        "```text",
        affected.to_string(index=False),
        "```",
        "",
        "## Eksen 1: Sezon basi rating",
        "",
        "Hedef, hicbir ratingden etkilenmeyen leave-team-out schedule-adjusted",
        "sezon performansidir.",
        "",
        "```text",
        rating["summary"].to_string(index=False),
        "```",
        "",
        "Eslesmis Spearman farklari (sezon icinde takim yeniden orneklemesi):",
        "",
        "```text",
        rating["comparisons"].to_string(index=False),
        "```",
        "",
        "## Eksen 2: Dynamic replay tahmin kalitesi",
        "",
        "Production kernel'i ile tam sezon replay; standart uc sinifli kayiplar.",
        "",
        "```text",
        prediction["summary"].to_string(index=False),
        "```",
        "",
        "Fold kazanimlari:",
        "",
        "```text",
        prediction["fold_summary"].to_string(index=False),
        "```",
        "",
        "Dependency-robust belirsizlik:",
        "",
        "```text",
        prediction["uncertainty"].to_string(index=False),
        "```",
        "",
        "## Nested walk-forward secim",
        "",
        "Agirlik yalniz training sezonlarindan secilir; test sezonu gorulmez.",
        "",
        "```text",
        nested.to_string(index=False),
        "```",
        "",
        "## Karar girdisi",
        "",
        f"- Conservative envelope guvenilir iyilesme: `{manifest['conservative_envelope_reliable_improvement']}`.",
        f"- Conservative envelope guvenilir zarar: `{manifest['conservative_envelope_reliable_harm']}`.",
        f"- Tek bir kumeleme gorusunde iyilesme: `{manifest['any_clustering_view_reliable_improvement']}` (kapi degildir).",
        f"- Tum fold'lari kazanan kollar: `{manifest['best_fold_gate_arms']}`.",
        f"- Guvenilir rating iyilesmesi: `{manifest['any_reliable_rating_improvement']}`.",
        f"- Guvenilir rating zarari: `{manifest['any_reliable_rating_harm']}`.",
        f"- En fazla hareket eden kutle orani: `{manifest['max_moved_share']:.4f}`.",
        "",
        "Karar urun tarafina aittir. Bu belge yalniz kanit uretir.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
