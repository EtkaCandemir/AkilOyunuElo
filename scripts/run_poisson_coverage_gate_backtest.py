from __future__ import annotations

"""Test coverage-aware production Poisson weights without changing production.

The active served layer blends Current ML with AO Domestic Poisson (rho=0) in
log-probability space. This diagnostic keeps BOTH at the production weight,
forces NONE to Current ML, and selects the ONE weight using past unseen folds
only. A full-history ONE surface is also reported for the next prospective
season, but never used to score an earlier fold.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ao_elo.evaluation import dependency_robust_loss_difference_ci  # noqa: E402
from ao_elo.ml_prediction import blend_probabilities, multiclass_losses  # noqa: E402
from ao_elo.prediction_ensemble import (  # noqa: E402
    AO_POISSON_RHO0_CONTROL,
    CURRENT_ML_BLEND,
)


SOURCE = (
    ROOT
    / "output"
    / "final_prediction_ensemble_backtest_2018_2026"
    / "unseen_predictions.csv"
)
CONTRACT = ROOT / "contracts" / "ao_european_elo_v2_production.json"
OUTPUT = ROOT / "output" / "poisson_coverage_gate_backtest_2018_2026"
PROBABILITY_COLUMNS = ("home_probability", "draw_probability", "away_probability")
WEIGHTS = tuple(round(value, 2) for value in np.linspace(0.0, 1.0, 21))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest a coverage-aware ML/Poisson production gate"
    )
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("bootstrap-samples must be at least 100")

    contract = args.contract.resolve()
    contract_before = hashlib.sha256(contract.read_bytes()).hexdigest()
    source = load_source(args.source.resolve())
    aligned = align_components(source)

    one = aligned.loc[aligned["domestic_poisson_coverage"].eq("ONE")].copy()
    surface = weight_surface(one)
    prospective = select_weight(surface)
    selections, walk_forward_one = walk_forward_one_predictions(one)
    prospective_one = prediction_block(
        one,
        blend_probabilities(
            probabilities(one, "ml"),
            probabilities(one, "poisson"),
            float(prospective["poisson_weight"]),
        ),
        "PROSPECTIVE_ONE",
    )
    prospective_one["selected_poisson_weight"] = float(
        prospective["poisson_weight"]
    )
    policy = build_walk_forward_policy(
        aligned,
        selections,
        prospective_one_weight=float(prospective["poisson_weight"]),
    )

    fold_results = fold_summary(walk_forward_one, reference_weight=0.50)
    competition = segment_summary(prospective_one, "competition")
    policy_summary = model_summary(policy)
    uncertainty = uncertainty_summary(
        walk_forward_one,
        bootstrap_samples=args.bootstrap_samples,
    )

    contract_after = hashlib.sha256(contract.read_bytes()).hexdigest()
    if contract_after != contract_before:
        raise ValueError("Production contract changed during research backtest")

    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    surface.to_csv(output / "one_weight_surface.csv", index=False)
    selections.to_csv(output / "walk_forward_selections.csv", index=False)
    fold_results.to_csv(output / "one_fold_results.csv", index=False)
    competition.to_csv(output / "one_competition_summary.csv", index=False)
    policy_summary.to_csv(output / "conditional_policy_summary.csv", index=False)
    uncertainty.to_csv(output / "dependency_uncertainty.csv", index=False)

    decision = decision_payload(
        one,
        prospective,
        selections,
        fold_results,
        policy_summary,
        uncertainty,
        contract_after,
    )
    (output / "selected_candidate.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output / "backtest_report.md").write_text(
        report(decision, surface, selections, fold_results, competition, policy_summary),
        encoding="utf-8",
    )

    print(f"ONE matches: {len(one)}")
    print(f"Prospective ONE weight: {prospective['poisson_weight']:.2f}")
    print(f"Decision: {decision['decision']}")
    print(f"Output: {output}")


def load_source(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    required = {
        "fold", "model", "match_id", "season", "kickoff_utc", "competition",
        "tie_id", "home_team_id", "away_team_id", "home_club_id",
        "away_club_id", "actual_class",
        "domestic_poisson_coverage", *PROBABILITY_COLUMNS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Source predictions missing columns: {missing}")
    selected = frame.loc[
        frame["model"].isin((CURRENT_ML_BLEND, AO_POISSON_RHO0_CONTROL))
    ].copy()
    if selected["season"].astype(str).eq("2026/27").any():
        raise ValueError("2026/27 must not enter coverage-gate selection")
    expected = selected.groupby("model")["match_id"].nunique()
    if set(expected.index) != {CURRENT_ML_BLEND, AO_POISSON_RHO0_CONTROL}:
        raise ValueError("Both production blend components are required")
    if expected.nunique() != 1 or int(expected.iloc[0]) != 4884:
        raise ValueError("Each source component must contain 4,884 unseen matches")
    if selected.duplicated(["model", "match_id"]).any():
        raise ValueError("Source model/match keys must be unique")
    probabilities = selected[list(PROBABILITY_COLUMNS)].to_numpy(float)
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0.0).any()
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-10)
    ):
        raise ValueError("Source probabilities must be finite and normalized")
    return selected


def align_components(source: pd.DataFrame) -> pd.DataFrame:
    identity = [
        "fold", "match_id", "season", "kickoff_utc", "competition", "tie_id",
        "home_team_id", "away_team_id", "home_club_id", "away_club_id",
        "actual_class",
        "domestic_poisson_coverage",
    ]
    ml = source.loc[source["model"].eq(CURRENT_ML_BLEND), identity + list(PROBABILITY_COLUMNS)]
    poisson = source.loc[
        source["model"].eq(AO_POISSON_RHO0_CONTROL),
        ["match_id", *PROBABILITY_COLUMNS],
    ]
    result = ml.merge(
        poisson,
        on="match_id",
        suffixes=("_ml", "_poisson"),
        validate="one_to_one",
    )
    return result.sort_values(["fold", "kickoff_utc", "match_id"], kind="stable")


def weight_surface(frame: pd.DataFrame) -> pd.DataFrame:
    baseline = metrics(probabilities(frame, "ml"), frame["actual_class"].to_numpy(int))
    rows = []
    for weight in WEIGHTS:
        candidate = blend_probabilities(
            probabilities(frame, "ml"), probabilities(frame, "poisson"), weight
        )
        observed = metrics(candidate, frame["actual_class"].to_numpy(int))
        objective = 0.5 * (
            observed["brier_1x2"] / baseline["brier_1x2"]
            + observed["log_loss_1x2"] / baseline["log_loss_1x2"]
        )
        rows.append(
            {
                "poisson_weight": weight,
                "ml_weight": 1.0 - weight,
                "matches": len(frame),
                "objective": objective,
                **observed,
                "delta_brier_vs_ml": observed["brier_1x2"] - baseline["brier_1x2"],
                "delta_log_loss_vs_ml": observed["log_loss_1x2"] - baseline["log_loss_1x2"],
            }
        )
    return pd.DataFrame(rows)


def select_weight(surface: pd.DataFrame) -> pd.Series:
    return surface.sort_values(
        ["objective", "log_loss_1x2", "brier_1x2", "poisson_weight"],
        kind="stable",
    ).iloc[0]


def walk_forward_one_predictions(one: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selection_rows = []
    prediction_rows = []
    folds = sorted(one["fold"].unique())
    for fold in folds:
        train = one.loc[one["fold"].lt(fold)]
        if train.empty:
            weight = 0.50
            source = "FROZEN_PRODUCTION_DEFAULT_NO_PRIOR_OOS"
        else:
            weight = float(select_weight(weight_surface(train))["poisson_weight"])
            source = "PRIOR_UNSEEN_FOLDS_ONLY"
        test = one.loc[one["fold"].eq(fold)].copy()
        candidate = blend_probabilities(
            probabilities(test, "ml"), probabilities(test, "poisson"), weight
        )
        block = prediction_block(test, candidate, "WALK_FORWARD_ONE")
        block["selected_poisson_weight"] = weight
        prediction_rows.append(block)
        selection_rows.append(
            {
                "fold": int(fold),
                "test_season": str(test["season"].iloc[0]),
                "training_one_matches": int(len(train)),
                "selection_source": source,
                "poisson_weight": weight,
                "ml_weight": 1.0 - weight,
            }
        )
    return pd.DataFrame(selection_rows), pd.concat(prediction_rows, ignore_index=True)


def build_walk_forward_policy(
    aligned: pd.DataFrame,
    selections: pd.DataFrame,
    *,
    prospective_one_weight: float,
) -> pd.DataFrame:
    rows = []
    selected = dict(zip(selections["fold"], selections["poisson_weight"]))
    for fold, frame in aligned.groupby("fold", sort=True):
        weights = np.where(
            frame["domestic_poisson_coverage"].eq("NONE"),
            0.0,
            np.where(
                frame["domestic_poisson_coverage"].eq("ONE"),
                float(selected[fold]),
                0.50,
            ),
        )
        candidate = rowwise_log_blend(
            probabilities(frame, "ml"), probabilities(frame, "poisson"), weights
        )
        block = prediction_block(frame, candidate, "COVERAGE_GATED_WALK_FORWARD")
        block["selected_poisson_weight"] = weights
        rows.append(block)
    gated = pd.concat(rows, ignore_index=True)
    prospective_weights = np.where(
        aligned["domestic_poisson_coverage"].eq("NONE"),
        0.0,
        np.where(
            aligned["domestic_poisson_coverage"].eq("ONE"),
            prospective_one_weight,
            0.50,
        ),
    )
    prospective = prediction_block(
        aligned,
        rowwise_log_blend(
            probabilities(aligned, "ml"),
            probabilities(aligned, "poisson"),
            prospective_weights,
        ),
        "PROSPECTIVE_COVERAGE_POLICY",
    )
    prospective["selected_poisson_weight"] = prospective_weights
    fixed = prediction_block(
        aligned,
        blend_probabilities(probabilities(aligned, "ml"), probabilities(aligned, "poisson"), 0.50),
        "FIXED_WEIGHT_050",
    )
    ml = prediction_block(aligned, probabilities(aligned, "ml"), "CURRENT_ML")
    return pd.concat([ml, fixed, gated, prospective], ignore_index=True)


def fold_summary(candidate: pd.DataFrame, *, reference_weight: float) -> pd.DataFrame:
    rows = []
    for fold, frame in candidate.groupby("fold", sort=True):
        reference = blend_probabilities(
            frame[[f"{name}_ml" for name in PROBABILITY_COLUMNS]].to_numpy(float),
            frame[[f"{name}_poisson" for name in PROBABILITY_COLUMNS]].to_numpy(float),
            reference_weight,
        )
        ref = metrics(reference, frame["actual_class"].to_numpy(int))
        ml = metrics(
            frame[[f"{name}_ml" for name in PROBABILITY_COLUMNS]].to_numpy(float),
            frame["actual_class"].to_numpy(int),
        )
        observed = metrics(
            frame[list(PROBABILITY_COLUMNS)].to_numpy(float),
            frame["actual_class"].to_numpy(int),
        )
        rows.append(
            {
                "fold": int(fold),
                "test_season": str(frame["season"].iloc[0]),
                "matches": len(frame),
                "poisson_weight": float(frame["selected_poisson_weight"].iloc[0]),
                **observed,
                "delta_brier_vs_fixed_050": observed["brier_1x2"] - ref["brier_1x2"],
                "delta_log_loss_vs_fixed_050": observed["log_loss_1x2"] - ref["log_loss_1x2"],
                "delta_brier_vs_ml": observed["brier_1x2"] - ml["brier_1x2"],
                "delta_log_loss_vs_ml": observed["log_loss_1x2"] - ml["log_loss_1x2"],
            }
        )
    return pd.DataFrame(rows)


def segment_summary(candidate: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for value, frame in candidate.groupby(column, sort=True):
        reference = blend_probabilities(
            frame[[f"{name}_ml" for name in PROBABILITY_COLUMNS]].to_numpy(float),
            frame[[f"{name}_poisson" for name in PROBABILITY_COLUMNS]].to_numpy(float),
            0.50,
        )
        ref = metrics(reference, frame["actual_class"].to_numpy(int))
        ml = metrics(
            frame[[f"{name}_ml" for name in PROBABILITY_COLUMNS]].to_numpy(float),
            frame["actual_class"].to_numpy(int),
        )
        observed = metrics(
            frame[list(PROBABILITY_COLUMNS)].to_numpy(float),
            frame["actual_class"].to_numpy(int),
        )
        rows.append(
            {
                column: value,
                "matches": len(frame),
                **observed,
                "delta_brier_vs_fixed_050": observed["brier_1x2"] - ref["brier_1x2"],
                "delta_log_loss_vs_fixed_050": observed["log_loss_1x2"] - ref["log_loss_1x2"],
                "delta_brier_vs_ml": observed["brier_1x2"] - ml["brier_1x2"],
                "delta_log_loss_vs_ml": observed["log_loss_1x2"] - ml["log_loss_1x2"],
            }
        )
    return pd.DataFrame(rows)


def model_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, block in frame.groupby("model", sort=False):
        rows.append(
            {
                "model": model,
                "matches": len(block),
                **metrics(
                    block[list(PROBABILITY_COLUMNS)].to_numpy(float),
                    block["actual_class"].to_numpy(int),
                ),
            }
        )
    result = pd.DataFrame(rows)
    reference = result.loc[result["model"].eq("FIXED_WEIGHT_050")].iloc[0]
    result["delta_brier_vs_fixed_050"] = result["brier_1x2"] - reference["brier_1x2"]
    result["delta_log_loss_vs_fixed_050"] = result["log_loss_1x2"] - reference["log_loss_1x2"]
    return result


def uncertainty_summary(candidate: pd.DataFrame, *, bootstrap_samples: int) -> pd.DataFrame:
    rows = []
    for baseline_name, weight in (("CURRENT_ML", 0.0), ("FIXED_WEIGHT_050", 0.50)):
        reference = rowwise_log_blend(
            candidate[[f"{name}_ml" for name in PROBABILITY_COLUMNS]].to_numpy(float),
            candidate[[f"{name}_poisson" for name in PROBABILITY_COLUMNS]].to_numpy(float),
            np.full(len(candidate), weight),
        )
        reference_loss = multiclass_losses(
            reference, candidate["actual_class"].to_numpy(int)
        )
        for metric in ("brier_1x2", "log_loss_1x2"):
            audit = candidate.copy()
            audit["loss_difference"] = (
                audit[metric].to_numpy(float) - reference_loss[metric].to_numpy(float)
            )
            ci = dependency_robust_loss_difference_ci(
                audit,
                bootstrap_samples=bootstrap_samples,
                seed=20260820 + len(rows) * 1009,
            )
            ci.insert(0, "candidate", "WALK_FORWARD_ONE")
            ci.insert(1, "baseline", baseline_name)
            ci.insert(2, "metric", metric)
            rows.append(ci)
    return pd.concat(rows, ignore_index=True)


def decision_payload(
    one: pd.DataFrame,
    prospective: pd.Series,
    selections: pd.DataFrame,
    folds: pd.DataFrame,
    policy: pd.DataFrame,
    uncertainty: pd.DataFrame,
    contract_sha256: str,
) -> dict[str, object]:
    fixed = metrics(
        blend_probabilities(probabilities(one, "ml"), probabilities(one, "poisson"), 0.50),
        one["actual_class"].to_numpy(int),
    )
    selected_probabilities = blend_probabilities(
        probabilities(one, "ml"),
        probabilities(one, "poisson"),
        float(prospective["poisson_weight"]),
    )
    selected = metrics(selected_probabilities, one["actual_class"].to_numpy(int))
    fold_brier_wins = int(folds["delta_brier_vs_fixed_050"].lt(0.0).sum())
    fold_log_wins = int(folds["delta_log_loss_vs_fixed_050"].lt(0.0).sum())
    envelope = uncertainty.loc[
        uncertainty["method"].eq("conservative_envelope")
        & uncertainty["baseline"].eq("FIXED_WEIGHT_050")
    ]
    reliable_harm = bool(envelope["reliable_harm"].any())
    policy_row = policy.loc[
        policy["model"].eq("PROSPECTIVE_COVERAGE_POLICY")
    ].iloc[0]
    walk_forward_policy_row = policy.loc[
        policy["model"].eq("COVERAGE_GATED_WALK_FORWARD")
    ].iloc[0]
    fixed_row = policy.loc[policy["model"].eq("FIXED_WEIGHT_050")].iloc[0]
    improves_one = bool(
        selected["brier_1x2"] < fixed["brier_1x2"]
        and selected["log_loss_1x2"] < fixed["log_loss_1x2"]
    )
    decision = (
        "ONE_WEIGHT_CANDIDATE"
        if improves_one and not reliable_harm
        else "KEEP_ONE_WEIGHT_050"
    )
    return {
        "decision": decision,
        "production_changed": False,
        "coverage_contract_under_test": {
            "BOTH": 0.50,
            "ONE": float(prospective["poisson_weight"]),
            "NONE": 0.0,
        },
        "one_matches": int(len(one)),
        "prospective_one_weight_selection": {
            "selection_window": "2020/21-2025/26 unseen matches",
            "poisson_weight": float(prospective["poisson_weight"]),
            "ml_weight": float(prospective["ml_weight"]),
            "brier_1x2": float(selected["brier_1x2"]),
            "log_loss_1x2": float(selected["log_loss_1x2"]),
            "delta_brier_vs_fixed_050": float(selected["brier_1x2"] - fixed["brier_1x2"]),
            "delta_log_loss_vs_fixed_050": float(selected["log_loss_1x2"] - fixed["log_loss_1x2"]),
        },
        "walk_forward": {
            "fold_brier_wins_vs_fixed_050": fold_brier_wins,
            "fold_log_loss_wins_vs_fixed_050": fold_log_wins,
            "weights": selections[["fold", "test_season", "poisson_weight"]].to_dict(orient="records"),
            "reliable_harm_vs_fixed_050": reliable_harm,
        },
        "full_policy_delta_vs_fixed_050": {
            "brier_1x2": float(policy_row["brier_1x2"] - fixed_row["brier_1x2"]),
            "log_loss_1x2": float(policy_row["log_loss_1x2"] - fixed_row["log_loss_1x2"]),
            "accuracy_1x2": float(policy_row["accuracy_1x2"] - fixed_row["accuracy_1x2"]),
        },
        "walk_forward_policy_delta_vs_fixed_050": {
            "brier_1x2": float(
                walk_forward_policy_row["brier_1x2"] - fixed_row["brier_1x2"]
            ),
            "log_loss_1x2": float(
                walk_forward_policy_row["log_loss_1x2"]
                - fixed_row["log_loss_1x2"]
            ),
            "accuracy_1x2": float(
                walk_forward_policy_row["accuracy_1x2"]
                - fixed_row["accuracy_1x2"]
            ),
        },
        "production_contract_sha256": contract_sha256,
        "rating_feedback": False,
        "untouched_holdout": "2026/27",
    }


def probabilities(frame: pd.DataFrame, suffix: str) -> np.ndarray:
    return frame[[f"{name}_{suffix}" for name in PROBABILITY_COLUMNS]].to_numpy(float)


def rowwise_log_blend(left: np.ndarray, right: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float).reshape(-1, 1)
    logits = (1.0 - weights) * np.log(np.clip(left, 1e-15, 1.0)) + weights * np.log(
        np.clip(right, 1e-15, 1.0)
    )
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=1, keepdims=True)


def prediction_block(frame: pd.DataFrame, probability: np.ndarray, model: str) -> pd.DataFrame:
    result = frame.copy()
    result[list(PROBABILITY_COLUMNS)] = probability
    loss = multiclass_losses(probability, result["actual_class"].to_numpy(int))
    result["brier_1x2"] = loss["brier_1x2"].to_numpy(float)
    result["log_loss_1x2"] = loss["log_loss_1x2"].to_numpy(float)
    result["model"] = model
    return result


def metrics(probability: np.ndarray, outcomes: np.ndarray) -> dict[str, float]:
    loss = multiclass_losses(probability, outcomes)
    return {
        "brier_1x2": float(loss["brier_1x2"].mean()),
        "log_loss_1x2": float(loss["log_loss_1x2"].mean()),
        "accuracy_1x2": float((probability.argmax(axis=1) == outcomes).mean()),
    }


def report(decision, surface, selections, folds, competition, policy) -> str:
    return f"""# Poisson Coverage Gate Backtesti

## Karar

**{decision['decision']}**. Production degistirilmedi. `NONE` Poisson agirligi
`0`, `BOTH` production agirligi `0.50` ve `ONE` agirligi gecmis unseen
foldlardan secilen ayri bir guven kapisi olarak test edildi.

## Veri

- Unseen pencere: `2020/21-2025/26`
- Tum maclar: `4.884`
- `ONE` coverage maclari: `{decision['one_matches']}`
- `2026/27`: secim disinda

## ONE sabit agirlik yuzeyi

{markdown(surface)}

## Walk-forward secimler

{markdown(selections)}

## ONE fold sonuclari

{markdown(folds)}

## ONE turnuva segmentleri

{markdown(competition)}

## Tam coverage politikasi

{markdown(policy)}

## Not

Tam-gecmis ONE secimi yalniz `2026/27` prospective adayidir. Onceki foldlarin
testinde yalniz daha eski unseen foldlar kullanilmistir. Bu rapor AO Live Elo'ya
geri besleme yapmaz ve production contract'i degistirmez.
"""


def markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for values in frame.itertuples(index=False, name=None):
        lines.append(
            "| "
            + " | ".join(f"{value:.6f}" if isinstance(value, float) else str(value) for value in values)
            + " |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
