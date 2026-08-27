from __future__ import annotations

"""Does removing non-participation from the European Prior improve the seed?

The external benchmark puts the model's measurable gap in one place: the
season-start seed. AO First Elo scores `0.4180` Spearman against realized
2025/26 performance where Opta scores `0.4866`, a reliable gap.

Inside the seed, the European Prior carries almost all of the discrimination
(sd `487` against `180` for the domestic prior). And it is built from a
fixed-weight sum in which a season the club did not enter contributes zero
points - indistinguishable from a season it entered and lost every match. The
prior therefore answers a mixture of "how good were you in Europe" and "did you
qualify at all", when only the first belongs to it. The second is a domestic
achievement that the Domestic Prior already owns, so it is charged twice: once
through the depressed history and once through the reduced exposure weight.

This run removes the first charge and leaves the second alone:

```text
BASELINE                   production calendar normalization
PARTICIPATION_BLIND_LIFT   control - same population, same average movement,
                           driven by exposure alone
PARTICIPATION_NORMALIZED   candidate - history renormalized over participation
```

The control is not optional. Without it a gain could not be told apart from the
plain fact of lifting low-exposure clubs, and the run would be uninterpretable.

The decision axis is the seed's Spearman against a realized performance target
that no rating touches. Match loss and forward ranking are carried as no-harm
checks, and the single-season Opta axis as a diagnostic.

The production contract is read only to hash it, and the hash is re-checked on
exit. This script activates nothing.
"""

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (str(SRC), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from ao_elo.config import AOEuropeanEloConfig  # noqa: E402
from ao_elo.european_participation import (  # noqa: E402
    BASELINE,
    BLIND_LIFT,
    NORMALIZED,
    SHRINKAGE_GRID,
    ParticipationNormalizationConfig,
    apply_participation_normalization,
    blind_control_config,
    calibrate_blind_lift,
    production_control_config,
)
from ao_elo.european_prior_recalibration import (  # noqa: E402
    ranking_uncertainty_summary,
)
from ao_elo.evaluation import (  # noqa: E402
    dependency_robust_loss_difference_ci,
    schedule_adjusted_team_performance,
)
from ao_elo.pipeline import compute_ao_first_elo_from_csv  # noqa: E402
from scripts.run_ao_first_seed_boost_backtest import (  # noqa: E402
    aggregate_target,
    load_seed_features,
    seed_ranking_metrics,
)
from scripts.run_current_model_evaluation import (  # noqa: E402
    EvaluationArm,
    evaluate_arm,
)
from scripts.run_dynamic_core_calibration import expanding_folds  # noqa: E402
from scripts.run_opponent_quintile_backtest import load_production_baseline  # noqa: E402
from scripts.run_stage_weighted_progression_backtest import load_xg_map  # noqa: E402
from scripts.run_v2_evaluation_upgrade import read_events  # noqa: E402


STATIC_ROOT = ROOT / "data" / "backtest_stage_b_2018_2026"
EVENTS_PATH = ROOT / "data" / "external_elo_benchmark_2018_2026" / "exact_date_events.csv"
XG_DATA = ROOT / "data" / "xg_2020_2026" / "uefa_2020_2026_matches_with_xg.csv"
PRODUCTION_CONTRACT = ROOT / "contracts" / "ao_european_elo_v2_production.json"
OUTPUT_ROOT = ROOT / "output" / "european_participation_backtest_2018_2026"
SEASON_DIRECTORY = re.compile(r"^\d{4}-\d{2}$")
ARM_ORDER = (BASELINE, BLIND_LIFT, NORMALIZED)
EVALUATION_SEASONS = ("2020/21", "2021/22", "2022/23", "2023/24", "2024/25", "2025/26")
EXPECTED_TEAM_SEASONS = 1887
SEED_MOVEMENT_P95_LIMIT = 100.0
MINIMUM_MODAL_FOLD_SHARE = 0.50
BASELINE_TOLERANCE = 1e-8
# reports/external_benchmark/rating_model_summary.csv, 2025/26, 236 clubs.
OPTA_PUBLISHED_SPEARMAN = 0.4866178392787856


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Renormalize the European Prior over participated seasons and test "
            "it against a blind control on the season-start seed axis"
        )
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument(
        "--skip-loss-axis",
        action="store_true",
        help="Skip the match-loss replay and report the seed axis only.",
    )
    arguments = parser.parse_args()
    if arguments.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples must be at least 100")
    output = arguments.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract_hash = hashlib.sha256(PRODUCTION_CONTRACT.read_bytes()).hexdigest()

    print("Loading production replay and seed evidence", flush=True)
    _, datasets, core, parameters, production_seed_map = load_production_baseline()
    seasons = tuple(data.season for data in datasets)
    folds = expanding_folds(seasons)
    if tuple(test for _, test in folds) != EVALUATION_SEASONS:
        raise ValueError("Unexpected outer fold seasons")

    events = read_events(EVENTS_PATH)
    target_by_competition = schedule_adjusted_team_performance(events)
    target = aggregate_target(target_by_competition)
    seeds, baseline_error = load_seed_evidence()
    print(
        f"Seed evidence: {len(seeds)} team-seasons; baseline reproduces "
        f"production to {baseline_error:.3e} Elo",
        flush=True,
    )

    print("Selecting the shrinkage inside each outer fold", flush=True)
    surface = candidate_surface(seeds, target)
    selections, audit = nested_selection(seeds, target, folds)

    arm_summary = summarize_arms(audit, target)
    fold_ranking = fold_seed_ranking(audit, target, folds)
    ranking_uncertainty = ranking_uncertainty_summary(
        fold_ranking.loc[fold_ranking["arm"].eq(NORMALIZED)],
        arguments.bootstrap_samples,
    )
    control_uncertainty = ranking_uncertainty_summary(
        candidate_minus_control(fold_ranking), arguments.bootstrap_samples
    )
    impact = seed_impact_summary(audit)
    opta = opta_diagnostic(audit, target)

    loss_summary = pd.DataFrame()
    loss_uncertainty = pd.DataFrame()
    if not arguments.skip_loss_axis:
        print("Replaying the match-loss no-harm axis", flush=True)
        loss_summary, loss_uncertainty = loss_axis(
            audit,
            datasets,
            core,
            parameters,
            production_seed_map,
            target_by_competition,
            arguments.bootstrap_samples,
        )

    gates = validation_gates(
        seeds, audit, selections, fold_ranking, ranking_uncertainty,
        control_uncertainty, impact, baseline_error, contract_hash,
    )
    if hashlib.sha256(PRODUCTION_CONTRACT.read_bytes()).hexdigest() != contract_hash:
        raise ValueError("Production contract changed during a research backtest")

    decision = decide(gates, arm_summary, ranking_uncertainty, control_uncertainty)
    write_outputs(
        output, surface, selections, audit, arm_summary, fold_ranking,
        ranking_uncertainty, control_uncertainty, impact, opta, loss_summary,
        loss_uncertainty, gates, decision, contract_hash,
    )
    print(f"\nDecision: {decision['decision']}")
    print(gates.to_string(index=False))
    print()
    print(arm_summary.to_string(index=False))


# ---------------------------------------------------------------------------
# evidence
# ---------------------------------------------------------------------------


def load_seed_evidence() -> tuple[pd.DataFrame, float]:
    """Return the production seed frame plus the two participation aggregates.

    `weighted_season_exposure` is the participation weight this study needs and
    the static pipeline already publishes it, so nothing upstream of the blend
    has to be recomputed. The baseline arm is then required to reproduce the
    served seed, which is the check that catches a stale artifact.
    """

    seeds = load_seed_features()
    frames = []
    config = AOEuropeanEloConfig.active()
    for directory in sorted(
        path for path in STATIC_ROOT.iterdir()
        if path.is_dir() and SEASON_DIRECTORY.match(path.name)
    ):
        frame = compute_ao_first_elo_from_csv(
            directory / "teams.csv",
            directory / "country_coefficients.csv",
            directory / "domestic_context.csv",
            directory / "club_european_points.csv",
            config,
        )
        frame["season"] = directory.name.replace("-", "/")
        frames.append(
            frame[
                [
                    "season", "team_id", "weighted_european_history",
                    "weighted_season_exposure", "european_exposure",
                ]
            ]
        )
    aggregates = pd.concat(frames, ignore_index=True)
    # `load_seed_features` and the realized-performance target agree on a
    # numeric team_id; the static pipeline emits it as a string, so align to
    # the seed frame rather than forcing either side.
    aggregates["team_id"] = aggregates["team_id"].astype(seeds["team_id"].dtype)
    result = seeds.merge(
        aggregates, on=["season", "team_id"], validate="one_to_one"
    )
    if len(result) != EXPECTED_TEAM_SEASONS:
        raise ValueError(f"Expected {EXPECTED_TEAM_SEASONS} seed team-seasons")

    control = apply_participation_normalization(result, production_control_config())
    error = float(
        (control["candidate_ao_first_elo"] - control["adjusted_ao_first_elo"])
        .abs()
        .max()
    )
    if error > BASELINE_TOLERANCE:
        raise ValueError(
            "Research baseline does not reproduce the served seed "
            f"({error:.6f} Elo). The frozen seed artifact is stale relative to "
            "the active contract; rebuild it before trusting this run."
        )
    return result, error


# ---------------------------------------------------------------------------
# arms and nested selection
# ---------------------------------------------------------------------------


def arm_configs(
    seeds: pd.DataFrame, shrinkage: float
) -> dict[str, ParticipationNormalizationConfig]:
    """The three arms for one shrinkage, with the control calibrated to match."""

    candidate = ParticipationNormalizationConfig(arm=NORMALIZED, shrinkage=shrinkage)
    return {
        BASELINE: production_control_config(),
        BLIND_LIFT: blind_control_config(calibrate_blind_lift(seeds, candidate)),
        NORMALIZED: candidate,
    }


def candidate_surface(seeds: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    """Full-history diagnostics for every shrinkage, for the report only."""

    rows = []
    for shrinkage in SHRINKAGE_GRID:
        for arm, config in arm_configs(seeds, shrinkage).items():
            applied = apply_participation_normalization(seeds, config)
            metrics = seed_ranking_metrics(
                applied, target, rating_column="candidate_ao_first_elo"
            )
            rows.append(
                {
                    "arm": arm,
                    "shrinkage": shrinkage,
                    "candidate_key": config.key,
                    **metrics,
                    "mean_abs_elo_delta": float(
                        applied["candidate_elo_delta"].abs().mean()
                    ),
                    "p95_abs_elo_delta": float(
                        applied["candidate_elo_delta"].abs().quantile(0.95)
                    ),
                }
            )
    return pd.DataFrame(rows)


def nested_selection(
    seeds: pd.DataFrame,
    target: pd.DataFrame,
    folds,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pick the shrinkage on training seasons, then score the held-out season."""

    selection_rows: list[dict[str, object]] = []
    audit_frames: list[pd.DataFrame] = []
    for fold, (train_seasons, test_season) in enumerate(folds, start=1):
        train = seeds.loc[seeds["season"].isin(train_seasons)]
        train_target = target.loc[target["season"].isin(train_seasons)]
        scored = []
        for shrinkage in SHRINKAGE_GRID:
            config = ParticipationNormalizationConfig(
                arm=NORMALIZED, shrinkage=shrinkage
            )
            applied = apply_participation_normalization(train, config)
            metrics = seed_ranking_metrics(
                applied, train_target, rating_column="candidate_ao_first_elo"
            )
            scored.append({"shrinkage": shrinkage, **metrics})
        table = pd.DataFrame(scored).sort_values(
            ["spearman", "pairwise_accuracy", "shrinkage"],
            ascending=[False, False, True],
        )
        selected = float(table.iloc[0]["shrinkage"])
        configs = arm_configs(train, selected)
        test = seeds.loc[seeds["season"].eq(test_season)]
        for arm, config in configs.items():
            applied = apply_participation_normalization(test, config)
            applied.insert(0, "fold", fold)
            applied.insert(1, "arm", arm)
            audit_frames.append(applied)
        selection_rows.append(
            {
                "fold": fold,
                "test_season": test_season,
                "train_seasons": "|".join(train_seasons),
                "selected_shrinkage": selected,
                "blind_lift_coefficient": configs[BLIND_LIFT].blind_lift_coefficient,
                "train_surface": json.dumps(
                    table.to_dict(orient="records"), sort_keys=True
                ),
            }
        )
        print(
            f"  fold {fold} {test_season}: k={selected:g}  "
            f"blind c={configs[BLIND_LIFT].blind_lift_coefficient:.3f}",
            flush=True,
        )
    return pd.DataFrame(selection_rows), pd.concat(audit_frames, ignore_index=True)


# ---------------------------------------------------------------------------
# seed axis
# ---------------------------------------------------------------------------


def summarize_arms(audit: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm in ARM_ORDER:
        frame = audit.loc[audit["arm"].eq(arm)]
        metrics = seed_ranking_metrics(
            frame, target, rating_column="candidate_ao_first_elo"
        )
        rows.append(
            {
                "arm": arm,
                **metrics,
                "mean_abs_elo_delta": float(frame["candidate_elo_delta"].abs().mean()),
                "p95_abs_elo_delta": float(
                    frame["candidate_elo_delta"].abs().quantile(0.95)
                ),
            }
        )
    result = pd.DataFrame(rows)
    baseline = result.loc[result["arm"].eq(BASELINE)].iloc[0]
    for metric in ("spearman", "pairwise_accuracy"):
        result[f"{metric}_vs_baseline"] = result[metric] - baseline[metric]
    return result


def fold_seed_ranking(
    audit: pd.DataFrame, target: pd.DataFrame, folds
) -> pd.DataFrame:
    rows = []
    for fold, (_, test_season) in enumerate(folds, start=1):
        season_target = target.loc[target["season"].eq(test_season)]
        metrics = {}
        for arm in ARM_ORDER:
            frame = audit.loc[audit["arm"].eq(arm) & audit["season"].eq(test_season)]
            metrics[arm] = seed_ranking_metrics(
                frame, season_target, rating_column="candidate_ao_first_elo"
            )
        for arm in ARM_ORDER:
            rows.append(
                {
                    "fold": fold,
                    "test_season": test_season,
                    "arm": arm,
                    "teams": metrics[arm]["teams"],
                    "seed_spearman": metrics[arm]["spearman"],
                    "seed_pairwise_accuracy": metrics[arm]["pairwise_accuracy"],
                    "delta_seed_spearman": metrics[arm]["spearman"]
                    - metrics[BASELINE]["spearman"],
                    "delta_seed_pairwise_accuracy": metrics[arm]["pairwise_accuracy"]
                    - metrics[BASELINE]["pairwise_accuracy"],
                }
            )
    return pd.DataFrame(rows)


def candidate_minus_control(fold_ranking: pd.DataFrame) -> pd.DataFrame:
    """The comparison the control exists for: candidate against blind lift.

    Both arms move the same population by the same average amount, so a gap
    between them can only come from reading participation.
    """

    indexed = fold_ranking.set_index(["fold", "arm"])
    rows = []
    for fold in sorted(fold_ranking["fold"].unique()):
        candidate = indexed.loc[(fold, NORMALIZED)]
        control = indexed.loc[(fold, BLIND_LIFT)]
        rows.append(
            {
                "fold": fold,
                "delta_seed_spearman": float(
                    candidate["seed_spearman"] - control["seed_spearman"]
                ),
                "delta_seed_pairwise_accuracy": float(
                    candidate["seed_pairwise_accuracy"]
                    - control["seed_pairwise_accuracy"]
                ),
            }
        )
    return pd.DataFrame(rows)


def opta_diagnostic(audit: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    """Where each arm sits on the axis the external benchmark reports.

    Opta only publishes a `2025/26` pre-season snapshot, so this is a single
    season of `236` clubs and cannot be a decision gate. It is here because it
    is the axis the documented gap is quoted on, and a reader should be able to
    see the movement without re-deriving it.
    """

    season = EVALUATION_SEASONS[-1]
    season_target = target.loc[target["season"].eq(season)]
    rows = []
    for arm in ARM_ORDER:
        frame = audit.loc[audit["arm"].eq(arm) & audit["season"].eq(season)]
        metrics = seed_ranking_metrics(
            frame, season_target, rating_column="candidate_ao_first_elo"
        )
        rows.append(
            {
                "season": season,
                "arm": arm,
                "teams": metrics["teams"],
                "seed_spearman": metrics["spearman"],
                "seed_pairwise_accuracy": metrics["pairwise_accuracy"],
                "opta_published_spearman": OPTA_PUBLISHED_SPEARMAN,
                "gap_to_opta": metrics["spearman"] - OPTA_PUBLISHED_SPEARMAN,
            }
        )
    return pd.DataFrame(rows)


def seed_impact_summary(audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm in ARM_ORDER:
        frame = audit.loc[audit["arm"].eq(arm)]
        delta = frame["candidate_elo_delta"]
        full = frame["candidate_played_weight"].ge(1.0 - 1e-12)
        zero = frame["candidate_played_weight"].le(1e-12)
        rows.append(
            {
                "arm": arm,
                "team_seasons": int(len(frame)),
                "moved_team_seasons": int((delta.abs() > 1e-9).sum()),
                "mean_abs_delta": float(delta.abs().mean()),
                "p95_abs_delta": float(delta.abs().quantile(0.95)),
                "max_abs_delta": float(delta.abs().max()),
                "full_participation_max_abs_delta": float(
                    delta[full].abs().max() if full.any() else 0.0
                ),
                "zero_participation_max_abs_delta": float(
                    delta[zero].abs().max() if zero.any() else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# loss no-harm axis
# ---------------------------------------------------------------------------


def loss_axis(
    audit: pd.DataFrame,
    datasets,
    core,
    parameters,
    production_seed_map,
    target_by_competition: pd.DataFrame,
    samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    xg_map = load_xg_map(XG_DATA, datasets)
    evaluations = {}
    for arm in ARM_ORDER:
        mapping = dict(production_seed_map)
        for row in audit.loc[audit["arm"].eq(arm)].itertuples(index=False):
            mapping[(str(row.season), int(row.team_id))] = float(
                row.candidate_ao_first_elo
            )
        if set(mapping) != set(production_seed_map) or not all(
            math.isfinite(value) for value in mapping.values()
        ):
            raise ValueError(f"invalid rating map for {arm}")
        evaluations[arm] = evaluate_arm(
            datasets,
            EvaluationArm(arm, True, True, True, True, True),
            core=core,
            parameters=parameters,
            current_domestic=mapping,
            baseline_domestic=production_seed_map,
            xg_map=xg_map,
            target=target_by_competition,
        )
    rows = []
    for arm in ARM_ORDER:
        predictions = evaluations[arm].predictions
        rows.append(
            {
                "arm": arm,
                "matches": int(len(predictions)),
                "brier_1x2": float(predictions["brier_1x2"].mean()),
                "log_loss_1x2": float(predictions["log_loss_1x2"].mean()),
            }
        )
    summary = pd.DataFrame(rows)
    baseline = summary.loc[summary["arm"].eq(BASELINE)].iloc[0]
    for metric in ("brier_1x2", "log_loss_1x2"):
        summary[f"{metric}_vs_baseline"] = summary[metric] - baseline[metric]

    base_predictions = evaluations[BASELINE].predictions
    blocks = []
    for arm in (BLIND_LIFT, NORMALIZED):
        merged = evaluations[arm].predictions.merge(
            base_predictions[["match_id", "brier_1x2", "log_loss_1x2"]],
            on="match_id",
            suffixes=("", "_baseline"),
            validate="one_to_one",
        )
        for metric in ("brier_1x2", "log_loss_1x2"):
            sample = merged[
                ["season", "match_id", "home_team_id", "away_team_id", "kickoff_utc"]
            ].copy()
            sample["loss_difference"] = (
                merged[metric] - merged[f"{metric}_baseline"]
            ).to_numpy(float)
            result = dependency_robust_loss_difference_ci(
                sample, bootstrap_samples=samples
            )
            result.insert(0, "arm", arm)
            result.insert(1, "metric", metric)
            blocks.append(result)
    return summary, pd.concat(blocks, ignore_index=True)


# ---------------------------------------------------------------------------
# gates and decision
# ---------------------------------------------------------------------------


def validation_gates(
    seeds: pd.DataFrame,
    audit: pd.DataFrame,
    selections: pd.DataFrame,
    fold_ranking: pd.DataFrame,
    ranking_uncertainty: pd.DataFrame,
    control_uncertainty: pd.DataFrame,
    impact: pd.DataFrame,
    baseline_error: float,
    contract_hash: str,
) -> pd.DataFrame:
    indexed = impact.set_index("arm")
    keys = selections["selected_shrinkage"].tolist()
    modal_share = keys.count(max(set(keys), key=keys.count)) / len(keys)
    candidate_folds = fold_ranking.loc[fold_ranking["arm"].eq(NORMALIZED)]
    control_folds = fold_ranking.loc[fold_ranking["arm"].eq(BLIND_LIFT)]
    beats_control = float(
        (
            candidate_folds.set_index("fold")["seed_spearman"]
            > control_folds.set_index("fold")["seed_spearman"]
        ).mean()
    )
    harmed = ranking_uncertainty.loc[ranking_uncertainty["reliable_harm"], "metric"]
    return pd.DataFrame(
        [
            {
                "gate": "baseline_reproduces_production",
                "passed": bool(baseline_error <= BASELINE_TOLERANCE),
                "observed": baseline_error,
                "requirement": "The baseline arm must reproduce the served seed",
            },
            {
                "gate": "full_participation_unchanged",
                "passed": bool(
                    indexed.loc[NORMALIZED, "full_participation_max_abs_delta"] <= 1e-9
                ),
                "observed": float(
                    indexed.loc[NORMALIZED, "full_participation_max_abs_delta"]
                ),
                "requirement": "Complete five-season evidence must never move",
            },
            {
                "gate": "zero_participation_unchanged",
                "passed": bool(
                    indexed.loc[NORMALIZED, "zero_participation_max_abs_delta"] <= 1e-9
                ),
                "observed": float(
                    indexed.loc[NORMALIZED, "zero_participation_max_abs_delta"]
                ),
                "requirement": "A club that never entered must keep its seed",
            },
            {
                "gate": "control_arm_present",
                "passed": bool(set(audit["arm"]) == set(ARM_ORDER)),
                "observed": int(audit["arm"].nunique()),
                "requirement": "The blind control cannot be dropped",
            },
            {
                "gate": "candidate_beats_control",
                "passed": bool(beats_control > 0.5),
                "observed": beats_control,
                "requirement": "The candidate must beat the blind control on folds",
            },
            {
                "gate": "ranking_not_reliably_harmed",
                "passed": bool(harmed.empty),
                "observed": float(ranking_uncertainty["mean_difference"].min()),
                "requirement": "Higher is better, so harm is an upper bound below zero",
            },
            {
                "gate": "selection_stability",
                "passed": bool(modal_share >= MINIMUM_MODAL_FOLD_SHARE),
                "observed": modal_share,
                "requirement": f"Modal fold share at least {MINIMUM_MODAL_FOLD_SHARE}",
            },
            {
                "gate": "seed_movement_bounded",
                "passed": bool(
                    indexed.loc[NORMALIZED, "p95_abs_delta"] <= SEED_MOVEMENT_P95_LIMIT
                ),
                "observed": float(indexed.loc[NORMALIZED, "p95_abs_delta"]),
                "requirement": f"p95 seed movement at most {SEED_MOVEMENT_P95_LIMIT} Elo",
            },
            {
                "gate": "contract_sha256_unchanged",
                "passed": bool(
                    hashlib.sha256(PRODUCTION_CONTRACT.read_bytes()).hexdigest()
                    == contract_hash
                ),
                "observed": contract_hash,
                "requirement": "No production file may change during a research run",
            },
        ]
    )


def decide(
    gates: pd.DataFrame,
    arm_summary: pd.DataFrame,
    ranking_uncertainty: pd.DataFrame,
    control_uncertainty: pd.DataFrame,
) -> dict[str, object]:
    indexed = arm_summary.set_index("arm")
    reliable_gain = bool(
        ranking_uncertainty.loc[
            ranking_uncertainty["metric"].eq("seed_spearman"), "reliable_improvement"
        ].any()
    )
    reliable_over_control = bool(
        control_uncertainty.loc[
            control_uncertainty["metric"].eq("seed_spearman"), "reliable_improvement"
        ].any()
    )
    structural = gates.loc[
        gates["gate"].isin(
            (
                "baseline_reproduces_production",
                "full_participation_unchanged",
                "zero_participation_unchanged",
                "control_arm_present",
                "contract_sha256_unchanged",
            )
        ),
        "passed",
    ].all()
    promotion = gates["passed"].all() and reliable_gain and reliable_over_control
    decision = (
        "KEEP_CURRENT"
        if not structural
        else "PROMOTE_CANDIDATE" if promotion
        else "KEEP_SHADOW_CANDIDATE" if reliable_over_control
        else "KEEP_SHADOW"
    )
    return {
        "decision": decision,
        "production_change": False,
        "candidate_seed_spearman": float(indexed.loc[NORMALIZED, "spearman"]),
        "control_seed_spearman": float(indexed.loc[BLIND_LIFT, "spearman"]),
        "baseline_seed_spearman": float(indexed.loc[BASELINE, "spearman"]),
        "candidate_vs_baseline": float(indexed.loc[NORMALIZED, "spearman_vs_baseline"]),
        "control_vs_baseline": float(indexed.loc[BLIND_LIFT, "spearman_vs_baseline"]),
        "ranking_reliable_improvement_vs_baseline": reliable_gain,
        "ranking_reliable_improvement_vs_control": reliable_over_control,
        "all_gates_passed": bool(gates["passed"].all()),
    }


def write_outputs(
    output: Path, surface, selections, audit, arm_summary, fold_ranking,
    ranking_uncertainty, control_uncertainty, impact, opta, loss_summary,
    loss_uncertainty, gates, decision, contract_hash,
) -> None:
    surface.to_csv(output / "candidate_surface.csv", index=False)
    selections.to_csv(output / "fold_selections.csv", index=False)
    audit.to_csv(output / "seed_audit.csv", index=False)
    arm_summary.to_csv(output / "arm_summary.csv", index=False)
    fold_ranking.to_csv(output / "fold_ranking.csv", index=False)
    ranking_uncertainty.to_csv(output / "ranking_uncertainty.csv", index=False)
    control_uncertainty.to_csv(output / "control_uncertainty.csv", index=False)
    impact.to_csv(output / "seed_impact_summary.csv", index=False)
    opta.to_csv(output / "opta_diagnostic.csv", index=False)
    gates.to_csv(output / "safety_audit.csv", index=False)
    if not loss_summary.empty:
        loss_summary.to_csv(output / "loss_summary.csv", index=False)
        loss_uncertainty.to_csv(output / "loss_uncertainty.csv", index=False)
    payload = {**decision, "production_contract_sha256": contract_hash}
    (output / "selected_candidate.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "backtest_report.md").write_text(
        build_report(
            arm_summary, fold_ranking, ranking_uncertainty, control_uncertainty,
            impact, opta, selections, loss_summary, gates, decision,
        ),
        encoding="utf-8",
    )


def build_report(
    arm_summary, fold_ranking, ranking_uncertainty, control_uncertainty,
    impact, opta, selections, loss_summary, gates, decision,
) -> str:
    loss_block = (
        "```text\n" + loss_summary.to_string(index=False) + "\n```"
        if not loss_summary.empty
        else "_Loss ekseni bu kosuda atlandi._"
    )
    return f"""# European Prior Katilim Normalizasyonu

Karar: **{decision['decision']}**. Production contract degistirilmedi.

## Soru

Production European Prior, bes sezonluk UEFA puanlarinin sabit agirlikli
toplamidir ve **girilmeyen sezon sifir puan katkisi yapar** - girilip hic puan
alinamayan sezondan ayirt edilemez. Boylece "Avrupa'da ne kadar iyiydin" ile
"katilabildin mi" karisir; ikincisi domestic prior'in zaten sahiplendigi bir
olgudur ve iki kez faturalandirilir.

Bu kosu ilk faturayi kaldirir, ikincisine dokunmaz:

```text
rate = weighted_european_history * (1 + k) / (weighted_season_exposure + k)
```

Tam katilimda `rate` yayinlanmis history'nin aynisidir, yani tam kanita sahip
kulup tanim geregi hic hareket etmez.

## Dogrulama kapilari

```text
{gates.to_string(index=False)}
```

## Kol ozeti (sezon basi seed Spearman)

```text
{arm_summary.to_string(index=False)}
```

## Fold bazinda

```text
{fold_ranking.to_string(index=False)}
```

## Nested secimler

```text
{selections[['fold','test_season','selected_shrinkage','blind_lift_coefficient']].to_string(index=False)}
```

## Belirsizlik: tabana karsi

```text
{ranking_uncertainty.to_string(index=False)}
```

## Belirsizlik: kor kontrole karsi

Iki kol da ayni kutleyi ayni ortalama buyuklukte hareket ettirir. Aralarindaki
guvenilir bir fark yalniz katilim yapisini okumaktan gelebilir.

```text
{control_uncertainty.to_string(index=False)}
```

## Seed hareketi

```text
{impact.to_string(index=False)}
```

## Mac loss no-harm ekseni

{loss_block}

## Opta ekseni (diagnostik, tek sezon)

Opta yalniz `2025/26` icin pre-season snapshot yayinlar; `236` kulup, karar
kapisi degil. Dokumanlarda alintilanan acik bu eksendedir.

```text
{opta.to_string(index=False)}
```

## Karar girdisi

- Aday tabana karsi guvenilir mi: `{decision['ranking_reliable_improvement_vs_baseline']}`
- Aday kontrole karsi guvenilir mi: `{decision['ranking_reliable_improvement_vs_control']}`
- Tum kapilar gecti mi: `{decision['all_gates_passed']}`

Karar urun tarafina aittir; bu belge yalniz kanit uretir.
"""


if __name__ == "__main__":
    main()
