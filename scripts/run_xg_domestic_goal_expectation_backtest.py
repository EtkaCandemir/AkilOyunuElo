from __future__ import annotations

"""Does the xG form term still pay on top of the repository's best score arm?

The earlier run (`run_xg_goal_expectation_backtest.py`) settled a narrower
question. It put one form term per side on an **Elo-only** goal expectation and
found that feeding that term xG beats feeding it goals: `exact_score_nll`
`3.027008` against `3.029642`, on the identical `3528` matches. What it could
not settle is whether the same term survives once the model already knows
something about attacking and defending. Its baseline, `ELO_ONLY` at
`3.029843`, is simpler than the repository's best score arm,
`DOMESTIC_ATTACK_DEFENCE_POISSON`, which reaches `3.021437` on those same
`3528` matches.

So this run rebuilds the comparison one level up. The baseline is the domestic
attack/defence transfer itself and the form term is added **on top of** it,
never in place of it:

```text
DOMESTIC_AD                 the repository's best score arm, the baseline
DOMESTIC_AD_GOALS_FORM      the same arm plus a goals form term, the control
DOMESTIC_AD_XG_FORM         the same arm plus an xG form term, the candidate
```

The goals control stays. Without it a gain in the xG arm could not be told
apart from the gain of simply granting the model two more parameters, and the
whole run would be uninterpretable.

Protocol: expanding walk-forward, every selection made on an inner validation
season that closes before the test season, dependency-robust bootstrap, and the
conservative envelope as the decision gate. Segments are reported because xG
coverage is `98.7%` in the main stage and `11%` in qualifying, so a pooled
number alone would hide where the layer works.

The script reads the production contract only to hash it, and asserts the hash
is unchanged on exit. It activates nothing.
"""

import argparse
import hashlib
import json
import math
import platform
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (str(SRC), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from ao_elo.domestic_poisson import (  # noqa: E402
    build_domestic_poisson_feature_store,
    domestic_candidate_grid,
)
from ao_elo.domestic_poisson_backtest import L2_GRID, select_domestic_config  # noqa: E402
from ao_elo.evaluation import dependency_robust_loss_difference_ci  # noqa: E402
from ao_elo.scoreline import ScorelineModelConfig, scoreline_matrix  # noqa: E402
from ao_elo.xg_domestic_goal_model import (  # noqa: E402
    ARM_BY_SOURCE,
    DOMESTIC_AD,
    DOMESTIC_AD_GOALS_FORM,
    DOMESTIC_AD_XG_FORM,
    attach_form_features,
    candidate_form_sources,
    fit_domestic_form_expectation,
    independent_poisson_nll,
    predict_domestic_form_expectation,
)


DEFAULT_FEATURES = (
    ROOT / "output" / "ml_1x2_backtest_2018_2026" / "pre_match_feature_store.csv"
)
DEFAULT_XG = ROOT / "data" / "xg_2020_2026" / "uefa_2020_2026_matches_with_xg.csv"
DEFAULT_DOMESTIC = (
    ROOT / "data" / "domestic_league_matches_2013_2026" / "domestic_matches.csv"
)
DEFAULT_BRIDGE = (
    ROOT / "data" / "domestic_league_matches_2013_2026" / "domestic_team_bridge.csv"
)
DEFAULT_SURFACE = (
    ROOT
    / "output"
    / "domestic_poisson_backtest_2018_2026"
    / "domestic_prequential_results.csv"
)
DEFAULT_REPO_ARM = (
    ROOT / "output" / "domestic_poisson_backtest_2018_2026" / "unseen_predictions.csv"
)
DEFAULT_CONTRACT = ROOT / "contracts" / "ao_european_elo_v2_production.json"
DEFAULT_OUTPUT = ROOT / "output" / "xg_domestic_goal_expectation_backtest_2020_2026"

QUALIFYING = frozenset({
    "Preliminary Round", "1st Qualifying Round", "2nd Qualifying Round",
    "3rd Qualifying Round", "Qualifying Play-off Round",
})
XG_SEASONS = ("2020/21", "2021/22", "2022/23", "2023/24", "2024/25", "2025/26")
MINIMUM_XG_TRAIN_SEASONS = 2
REPO_ARM = "DOMESTIC_ATTACK_DEFENCE_POISSON"
REPO_ARM_FULL_SAMPLE_NLL = 3.016594456946233
METRICS = ("exact_score_nll", "total_goals_error", "over_2_5_brier", "btts_brier")
BOOTSTRAP_SAMPLES = 4000
BOOTSTRAP_SEED = 20260819
ARM_ORDER = (DOMESTIC_AD, DOMESTIC_AD_GOALS_FORM, DOMESTIC_AD_XG_FORM)
CANDIDATE_ARMS = (DOMESTIC_AD_GOALS_FORM, DOMESTIC_AD_XG_FORM)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Add a goals-form and an xG-form term on top of the domestic "
            "attack/defence Poisson arm and compare the three on one sample"
        )
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--xg-data", type=Path, default=DEFAULT_XG)
    parser.add_argument("--domestic-matches", type=Path, default=DEFAULT_DOMESTIC)
    parser.add_argument("--domestic-bridge", type=Path, default=DEFAULT_BRIDGE)
    parser.add_argument("--domestic-surface", type=Path, default=DEFAULT_SURFACE)
    parser.add_argument("--repo-arm-predictions", type=Path, default=DEFAULT_REPO_ARM)
    parser.add_argument("--production-contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--training-window",
        choices=("full", "xg"),
        default="full",
        help=(
            "full: train on every season before the test season, which keeps "
            "the baseline at the strength the repository reports. "
            "xg: train only on xG-covered seasons, matching the earlier "
            "Elo-only run so the goals and xG arms see identical rows."
        ),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    arguments = parser.parse_args()

    output_root = arguments.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    contract_path = arguments.production_contract.resolve()
    contract_before = hashlib.sha256(contract_path.read_bytes()).hexdigest()

    data = load_matches(arguments.features, arguments.xg_data)
    domestic = pd.read_csv(arguments.domestic_matches.resolve(), low_memory=False)
    bridge = pd.read_csv(arguments.domestic_bridge.resolve(), low_memory=False)
    surface = pd.read_csv(arguments.domestic_surface.resolve())
    folds = build_folds(data, arguments.training_window)

    print(
        f"Training window '{arguments.training_window}': "
        f"{len(folds)} folds, test seasons "
        f"{', '.join(test for _, _, test in folds)}",
        flush=True,
    )
    predictions, selections = walk_forward(data, domestic, bridge, surface, folds)
    predictions = add_losses(predictions)

    summary = arm_summary(predictions)
    segments = segment_summary(predictions)
    folds_frame = fold_summary(predictions)
    uncertainty = uncertainty_summary(
        predictions, samples=int(arguments.bootstrap_samples)
    )
    wins = fold_wins(folds_frame)
    reconciliation = baseline_reconciliation(
        predictions, arguments.repo_arm_predictions
    )
    gates = validation_gates(predictions, selections, reconciliation, folds)

    contract_after = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    if contract_after != contract_before:
        raise ValueError("Production contract changed during a prediction-only run")

    predictions.to_csv(output_root / "unseen_predictions.csv", index=False)
    summary.to_csv(output_root / "arm_summary.csv", index=False)
    segments.to_csv(output_root / "segment_summary.csv", index=False)
    folds_frame.to_csv(output_root / "fold_summary.csv", index=False)
    uncertainty.to_csv(output_root / "uncertainty.csv", index=False)
    selections.to_csv(output_root / "fold_selections.csv", index=False)
    wins.to_csv(output_root / "fold_wins.csv", index=False)
    reconciliation.to_csv(output_root / "baseline_reconciliation.csv", index=False)
    gates.to_csv(output_root / "validation_gates.csv", index=False)

    manifest = build_manifest(
        summary,
        segments,
        uncertainty,
        gates,
        wins,
        reconciliation,
        folds,
        arguments.training_window,
        contract_after,
    )
    (output_root / "backtest_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_report(
        output_root / "backtest_report.md",
        summary,
        segments,
        folds_frame,
        uncertainty,
        selections,
        wins,
        reconciliation,
        gates,
        manifest,
    )

    print(f"\nWrote the domestic + form backtest to {output_root}")
    print(gates.to_string(index=False))
    print()
    print(summary.to_string(index=False))
    print()
    print(segments.to_string(index=False))
    print()
    print(wins.to_string(index=False))


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def load_matches(features_path: Path, xg_path: Path) -> pd.DataFrame:
    """Return the European feature store with xG columns attached.

    Seasons before `2020/21` carry no xG at all: FotMob never returns the field
    for them. Those rows stay in the frame because the baseline is entitled to
    train on them; they are marked ineligible so the xG history simply does not
    advance across them.
    """

    frame = pd.read_csv(features_path.resolve(), low_memory=False)
    xg = pd.read_csv(xg_path.resolve(), low_memory=False)[
        ["match_id", "xg_home", "xg_away", "xg_analysis_eligible"]
    ]
    for column in ("match_id",):
        frame[column] = frame[column].astype(str)
        xg[column] = xg[column].astype(str)
    merged = frame.merge(xg, on="match_id", how="left", validate="one_to_one")
    merged["xg_analysis_eligible"] = (
        merged["xg_analysis_eligible"].fillna(False).astype(bool)
    )
    for column in ("xg_home", "xg_away"):
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)
    merged.loc[~merged["xg_analysis_eligible"], ["xg_home", "xg_away"]] = 0.0
    merged["phase"] = np.where(merged["round"].isin(QUALIFYING), "QUALIFYING", "MAIN")
    merged["kickoff_utc"] = pd.to_datetime(
        merged["kickoff_utc"], utc=True, format="ISO8601"
    )
    for side in ("home", "away"):
        merged[f"{side}_club_id"] = merged[f"{side}_club_id"].astype(str)
        merged[f"{side}_team_id"] = merged[f"{side}_team_id"].astype(str)
    covered = merged.loc[merged["season"].isin(XG_SEASONS)]
    if not covered["xg_analysis_eligible"].any():
        raise ValueError("xG columns did not attach to any covered season")
    if merged.loc[~merged["season"].isin(XG_SEASONS), "xg_analysis_eligible"].any():
        raise ValueError("xG appeared in a season the dataset does not cover")
    return merged.sort_values(["kickoff_utc", "match_id"], kind="stable").reset_index(
        drop=True
    )


def build_folds(
    data: pd.DataFrame, training_window: str
) -> list[tuple[tuple[str, ...], str, str]]:
    """Return `(train_seasons, inner_validation_season, test_season)` per fold.

    A test season qualifies only when at least `MINIMUM_XG_TRAIN_SEASONS`
    xG-covered seasons close before it. A fold whose training half carries no
    xG cannot fit an xG coefficient, and scoring the test season with a
    coefficient that was never fitted would be worse than dropping the fold.
    """

    seasons = tuple(dict.fromkeys(data["season"].astype(str)))
    folds: list[tuple[tuple[str, ...], str, str]] = []
    for index, test_season in enumerate(seasons):
        prior = seasons[:index]
        covered = tuple(season for season in prior if season in XG_SEASONS)
        if len(covered) < MINIMUM_XG_TRAIN_SEASONS:
            continue
        train = covered if training_window == "xg" else prior
        if len(train) < 2:
            continue
        folds.append((tuple(train), train[-1], test_season))
    if not folds:
        raise ValueError("no fold satisfies the xG training requirement")
    return folds


# ---------------------------------------------------------------------------
# walk-forward
# ---------------------------------------------------------------------------


def walk_forward(
    data: pd.DataFrame,
    domestic: pd.DataFrame,
    bridge: pd.DataFrame,
    surface: pd.DataFrame,
    folds: list[tuple[tuple[str, ...], str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit on training seasons only, then score the held-out season."""

    candidates = {candidate.key: candidate for candidate in domestic_candidate_grid()}
    feature_cache: dict[str, pd.DataFrame] = {}
    blocks: list[pd.DataFrame] = []
    selection_rows: list[dict[str, object]] = []

    for fold, (train_seasons, validation_season, test_season) in enumerate(
        folds, start=1
    ):
        domestic_config = select_domestic_config(surface, candidates, validation_season)
        enriched = _features_for_config(
            feature_cache, data, domestic, bridge, domestic_config
        )
        visible = enriched.loc[
            enriched["season"].isin((*train_seasons, test_season))
        ].copy()
        inner_train_seasons = train_seasons[:-1]

        for source in candidate_form_sources():
            featured, prior = attach_form_features(
                visible,
                source,
                training_frame=visible.loc[visible["season"].isin(train_seasons)],
            )
            inner_train = featured.loc[featured["season"].isin(inner_train_seasons)]
            inner_valid = featured.loc[featured["season"].eq(validation_season)]
            outer_train = featured.loc[featured["season"].isin(train_seasons)]
            outer_test = featured.loc[featured["season"].eq(test_season)]
            if min(len(inner_train), len(inner_valid), len(outer_test)) == 0:
                raise ValueError(f"fold {fold} contains an empty temporal split")

            l2_surface = _select_l2(
                inner_train, inner_valid, source, domestic_config.venue_context, prior
            )
            best_l2 = float(
                l2_surface.sort_values(
                    ["validation_goal_nll", "l2_strength"], kind="stable"
                ).iloc[0]["l2_strength"]
            )
            config = fit_domestic_form_expectation(
                outer_train,
                source,
                l2_strength=best_l2,
                use_reliability=True,
                use_venue=domestic_config.venue_context,
                form_prior_value=prior,
            )
            rates = predict_domestic_form_expectation(outer_test, config)
            block = outer_test[
                [
                    "match_id", "season", "competition", "round", "phase", "tie_id",
                    "home_team_id", "away_team_id", "kickoff_utc", "home_goals",
                    "away_goals", "xg_analysis_eligible", "domestic_poisson_coverage",
                ]
            ].merge(rates, on="match_id", validate="one_to_one")
            block.insert(0, "fold", fold)
            block["arm"] = ARM_BY_SOURCE[source]
            block["form_attack"] = config.form_attack
            block["form_defence"] = config.form_defence
            blocks.append(block)
            selection_rows.append(
                {
                    "fold": fold,
                    "test_season": test_season,
                    "inner_validation_season": validation_season,
                    "train_seasons": "|".join(train_seasons),
                    "arm": ARM_BY_SOURCE[source],
                    "domestic_candidate_key": domestic_config.key,
                    "selected_l2_strength": best_l2,
                    "l2_surface": json.dumps(
                        l2_surface.to_dict(orient="records"), sort_keys=True
                    ),
                    **asdict(config),
                }
            )
            print(
                f"  fold {fold} {test_season} {ARM_BY_SOURCE[source]:<24} "
                f"l2={best_l2:<5g} form=({config.form_attack:.6f},"
                f"{config.form_defence:.6f})",
                flush=True,
            )

    predictions = pd.concat(blocks, ignore_index=True)
    return predictions, pd.DataFrame(selection_rows)


def _features_for_config(
    cache: dict[str, pd.DataFrame],
    base: pd.DataFrame,
    domestic: pd.DataFrame,
    bridge: pd.DataFrame,
    config,
) -> pd.DataFrame:
    if config.key not in cache:
        print(f"Building domestic Poisson features for {config.key}", flush=True)
        features = build_domestic_poisson_feature_store(domestic, base, bridge, config)
        cache[config.key] = base.merge(features, on="match_id", validate="one_to_one")
    return cache[config.key]


def _select_l2(
    inner_train: pd.DataFrame,
    inner_valid: pd.DataFrame,
    source: str,
    use_venue: bool,
    prior: float,
) -> pd.DataFrame:
    """Score the regularization grid on the inner validation season.

    The objective is the two-sided Poisson goal NLL, the family this run is
    judged on. Every arm is selected by the same rule, so no arm is handed a
    regularization advantage the others did not get.
    """

    rows: list[dict[str, object]] = []
    for l2_strength in L2_GRID:
        config = fit_domestic_form_expectation(
            inner_train,
            source,
            l2_strength=float(l2_strength),
            use_reliability=True,
            use_venue=use_venue,
            form_prior_value=prior,
        )
        rates = predict_domestic_form_expectation(inner_valid, config)
        rows.append(
            {
                "l2_strength": float(l2_strength),
                "validation_goal_nll": independent_poisson_nll(
                    inner_valid["home_goals"].to_numpy(float),
                    inner_valid["away_goals"].to_numpy(float),
                    rates["lambda_home"].to_numpy(float),
                    rates["lambda_away"].to_numpy(float),
                ),
                "form_attack": config.form_attack,
                "form_defence": config.form_defence,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------


def add_losses(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the four reported metrics, all read off one score matrix."""

    matrix_config = ScorelineModelConfig(mu=0.0, elo_slope=1.0, rho=0.0)
    exact, over, btts, total = [], [], [], []
    for row in frame.itertuples(index=False):
        matrix, _ = scoreline_matrix(
            float(row.lambda_home), float(row.lambda_away), matrix_config
        )
        home, away = int(row.home_goals), int(row.away_goals)
        probability = (
            float(matrix[home, away])
            if home < matrix.shape[0] and away < matrix.shape[1]
            else 1e-12
        )
        exact.append(-math.log(max(probability, 1e-12)))
        grid = np.add.outer(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
        over.append(float(matrix[grid >= 3].sum()))
        btts.append(float(matrix[1:, 1:].sum()))
        total.append(float(row.lambda_home + row.lambda_away))
    result = frame.copy()
    result["exact_score_nll"] = exact
    result["expected_total_goals"] = total
    actual_total = result["home_goals"] + result["away_goals"]
    result["total_goals_error"] = (result["expected_total_goals"] - actual_total).abs()
    result["over_2_5_actual"] = (actual_total >= 3).astype(float)
    result["over_2_5_probability"] = over
    result["over_2_5_brier"] = (
        result["over_2_5_probability"] - result["over_2_5_actual"]
    ) ** 2
    result["btts_actual"] = (
        (result["home_goals"] > 0) & (result["away_goals"] > 0)
    ).astype(float)
    result["btts_probability"] = btts
    result["btts_brier"] = (result["btts_probability"] - result["btts_actual"]) ** 2
    return result


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------


def arm_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm, block in predictions.groupby("arm"):
        record = {"arm": arm, "matches": int(len(block))}
        record.update({metric: float(block[metric].mean()) for metric in METRICS})
        rows.append(record)
    frame = pd.DataFrame(rows)
    baseline = frame.loc[frame["arm"].eq(DOMESTIC_AD)].iloc[0]
    for metric in METRICS:
        frame[f"{metric}_vs_baseline"] = frame[metric] - baseline[metric]
    frame["arm_order"] = frame["arm"].map({arm: i for i, arm in enumerate(ARM_ORDER)})
    return frame.sort_values("arm_order").drop(columns="arm_order").reset_index(drop=True)


def _paired(
    predictions: pd.DataFrame, arm: str, reference: str = DOMESTIC_AD
) -> pd.DataFrame:
    base = predictions.loc[predictions["arm"].eq(reference)].set_index("match_id")
    block = predictions.loc[predictions["arm"].eq(arm)].set_index("match_id")
    joined = block.join(base[list(METRICS)], rsuffix="_base")
    joined["has_xg"] = joined["xg_analysis_eligible"].astype(bool)
    return joined


def _segments(joined: pd.DataFrame) -> tuple[tuple[str, pd.DataFrame], ...]:
    return (
        ("ALL", joined),
        ("PHASE:MAIN", joined.loc[joined["phase"].eq("MAIN")]),
        ("PHASE:QUALIFYING", joined.loc[joined["phase"].eq("QUALIFYING")]),
        ("XG_PRESENT", joined.loc[joined["has_xg"]]),
        ("XG_ABSENT", joined.loc[~joined["has_xg"]]),
    )


def segment_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm in CANDIDATE_ARMS:
        joined = _paired(predictions, arm)
        for label, subset in _segments(joined):
            if subset.empty:
                continue
            record = {"arm": arm, "segment": label, "matches": int(len(subset))}
            for metric in METRICS:
                record[f"{metric}_vs_baseline"] = float(
                    (subset[metric] - subset[f"{metric}_base"]).mean()
                )
            rows.append(record)
    return pd.DataFrame(rows)


def fold_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (arm, season), block in predictions.groupby(["arm", "season"]):
        record = {"arm": arm, "test_season": season, "matches": int(len(block))}
        record.update({metric: float(block[metric].mean()) for metric in METRICS})
        rows.append(record)
    frame = pd.DataFrame(rows)
    baseline = frame.loc[frame["arm"].eq(DOMESTIC_AD)].set_index("test_season")
    for metric in METRICS:
        frame[f"{metric}_vs_baseline"] = frame.apply(
            lambda row, m=metric: float(row[m] - baseline.loc[row["test_season"], m]),
            axis=1,
        )
    return frame.sort_values(["test_season", "arm"]).reset_index(drop=True)


def uncertainty_summary(predictions: pd.DataFrame, *, samples: int) -> pd.DataFrame:
    """Bootstrap every arm against the baseline, and xG against its control.

    The head-to-head row is the one the control exists for. Both candidate arms
    carry the same two extra parameters, so a reliable gap between them can
    only come from the source the form term is read out of.
    """

    comparisons = [(arm, DOMESTIC_AD) for arm in CANDIDATE_ARMS]
    comparisons.append((DOMESTIC_AD_XG_FORM, DOMESTIC_AD_GOALS_FORM))
    blocks = []
    for arm, reference in comparisons:
        joined = _paired(predictions, arm, reference).reset_index()
        for label, subset in _segments(joined):
            if subset.empty:
                continue
            for metric in ("exact_score_nll", "over_2_5_brier"):
                sample = subset[
                    [
                        "season", "match_id", "home_team_id", "away_team_id",
                        "kickoff_utc", "tie_id",
                    ]
                ].copy()
                sample["loss_difference"] = (
                    subset[metric] - subset[f"{metric}_base"]
                ).to_numpy(float)
                result = dependency_robust_loss_difference_ci(
                    sample, bootstrap_samples=samples, seed=BOOTSTRAP_SEED
                )
                result.insert(0, "arm", arm)
                result.insert(1, "reference", reference)
                result.insert(2, "segment", label)
                result.insert(3, "metric", metric)
                blocks.append(result)
    return pd.concat(blocks, ignore_index=True)


def fold_wins(folds_frame: pd.DataFrame) -> pd.DataFrame:
    """Count, per metric, how often a candidate beats what it is compared to."""

    pivot = folds_frame.pivot(index="test_season", columns="arm", values=list(METRICS))
    rows = []
    for arm, reference in (
        (DOMESTIC_AD_GOALS_FORM, DOMESTIC_AD),
        (DOMESTIC_AD_XG_FORM, DOMESTIC_AD),
        (DOMESTIC_AD_XG_FORM, DOMESTIC_AD_GOALS_FORM),
    ):
        for metric in METRICS:
            better = pivot[(metric, arm)] < pivot[(metric, reference)]
            rows.append(
                {
                    "arm": arm,
                    "reference": reference,
                    "metric": metric,
                    "folds": int(len(better)),
                    "wins": int(better.sum()),
                    "record": f"{int(better.sum())}/{len(better)}",
                }
            )
    return pd.DataFrame(rows)


def baseline_reconciliation(
    predictions: pd.DataFrame, repo_arm_path: Path
) -> pd.DataFrame:
    """Line this run's baseline up against the published domestic arm.

    The published arm is not reproduced bit for bit: it selects its
    regularization on a `1X2` objective and this run selects on goal NLL, and
    it trains its folds from `2018/19` regardless of xG coverage. What the
    check has to establish is narrower and more important - that the baseline
    here is the same model at roughly the same strength, not a weakened stand-in
    that the candidate arms would beat for free.
    """

    if not repo_arm_path.is_file():
        return pd.DataFrame(
            [{"scope": "repo_arm_unavailable", "matches": 0, "exact_score_nll": math.nan}]
        )
    repo = pd.read_csv(
        repo_arm_path.resolve(),
        low_memory=False,
        usecols=["match_id", "season", "model", "exact_score_probability"],
    )
    repo = repo.loc[repo["model"].eq(REPO_ARM)].copy()
    repo["match_id"] = repo["match_id"].astype(str)
    baseline = predictions.loc[predictions["arm"].eq(DOMESTIC_AD)]
    shared = sorted(set(baseline["match_id"]) & set(repo["match_id"]))
    repo_shared = repo.loc[repo["match_id"].isin(shared)]
    own_shared = baseline.loc[baseline["match_id"].isin(shared)]
    repo_nll = float(
        -np.log(repo_shared["exact_score_probability"].clip(1e-15, 1.0)).mean()
    )
    own_nll = float(own_shared["exact_score_nll"].mean())
    return pd.DataFrame(
        [
            {
                "scope": "repo_arm_full_sample",
                "matches": int(len(repo)),
                "exact_score_nll": REPO_ARM_FULL_SAMPLE_NLL,
                "difference_vs_this_baseline": math.nan,
            },
            {
                "scope": "repo_arm_shared_sample",
                "matches": int(len(repo_shared)),
                "exact_score_nll": repo_nll,
                "difference_vs_this_baseline": own_nll - repo_nll,
            },
            {
                "scope": "this_baseline_shared_sample",
                "matches": int(len(own_shared)),
                "exact_score_nll": own_nll,
                "difference_vs_this_baseline": 0.0,
            },
        ]
    )


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------


def validation_gates(
    predictions: pd.DataFrame,
    selections: pd.DataFrame,
    reconciliation: pd.DataFrame,
    folds: list[tuple[tuple[str, ...], str, str]],
) -> pd.DataFrame:
    arms = set(predictions["arm"])
    per_arm = predictions.groupby("arm")["match_id"].nunique()
    xg_arm = predictions.loc[predictions["arm"].eq(DOMESTIC_AD_XG_FORM)]
    per_fold_matches = (
        predictions.groupby(["fold", "arm"])["match_id"].nunique().unstack()
    )
    shared = reconciliation.loc[
        reconciliation["scope"].eq("repo_arm_shared_sample"), "difference_vs_this_baseline"
    ]
    drift = float(shared.iloc[0]) if len(shared) else math.nan
    selection_is_causal = all(
        max(_season_start(season) for season in train) < _season_start(test)
        and _season_start(validation) < _season_start(test)
        for train, validation, test in folds
    )
    return pd.DataFrame(
        [
            {
                "gate": "three_arms_scored",
                "passed": arms == set(ARM_ORDER),
                "observed": len(arms),
                "requirement": "Domestic baseline, goals control and xG candidate",
            },
            {
                "gate": "arms_share_one_sample",
                "passed": bool(per_arm.nunique() == 1),
                "observed": int(per_arm.iloc[0]),
                "requirement": "Every arm scores the identical held-out matches",
            },
            {
                "gate": "arms_share_every_fold",
                "passed": bool(per_fold_matches.nunique(axis=1).eq(1).all()),
                "observed": int(per_fold_matches.shape[0]),
                "requirement": "No arm may skip or add a fold",
            },
            {
                "gate": "baseline_carries_no_form",
                "passed": bool(
                    predictions.loc[
                        predictions["arm"].eq(DOMESTIC_AD),
                        ["form_attack", "form_defence"],
                    ]
                    .eq(0.0)
                    .all()
                    .all()
                ),
                "observed": 0.0,
                "requirement": "The baseline stays the published domestic arm",
            },
            {
                "gate": "selection_never_sees_test_season",
                "passed": bool(selection_is_causal),
                "observed": len(folds),
                "requirement": "Training and inner validation close before the test season",
            },
            {
                "gate": "arms_share_domestic_configuration",
                "passed": bool(
                    selections.groupby("fold")["domestic_candidate_key"].nunique().eq(1).all()
                ),
                "observed": int(selections["domestic_candidate_key"].nunique()),
                "requirement": "One domestic state per fold, shared by all three arms",
            },
            {
                "gate": "rates_within_bounds",
                "passed": bool(
                    predictions["lambda_home"].between(0.2, 4.5).all()
                    and predictions["lambda_away"].between(0.2, 4.5).all()
                ),
                "observed": float(predictions["lambda_home"].max()),
                "requirement": "Rates stay inside the production lambda window",
            },
            {
                "gate": "xg_form_is_sparser_than_goals",
                "passed": bool(
                    xg_arm["xg_analysis_eligible"].astype(bool).mean() < 0.95
                ),
                "observed": float(xg_arm["xg_analysis_eligible"].astype(bool).mean()),
                "requirement": "Documented coverage gap must be visible in the sample",
            },
            {
                "gate": "baseline_matches_published_arm",
                "passed": bool(math.isfinite(drift) and abs(drift) < 0.01),
                "observed": drift,
                "requirement": (
                    "Baseline within 0.01 nats of the published domestic arm on "
                    "the shared matches, so the candidate is not beating a "
                    "weakened stand-in"
                ),
            },
        ]
    )


def _season_start(season: str) -> int:
    return int(str(season).split("/")[0])


# ---------------------------------------------------------------------------
# manifest and report
# ---------------------------------------------------------------------------


def build_manifest(
    summary: pd.DataFrame,
    segments: pd.DataFrame,
    uncertainty: pd.DataFrame,
    gates: pd.DataFrame,
    wins: pd.DataFrame,
    reconciliation: pd.DataFrame,
    folds: list[tuple[tuple[str, ...], str, str]],
    training_window: str,
    contract_sha256: str,
) -> dict[str, object]:
    indexed = summary.set_index("arm")
    envelope = uncertainty.loc[uncertainty["method"].eq("conservative_envelope")]
    reliable = envelope.loc[envelope["reliable_improvement"]]
    harmful = envelope.loc[envelope["reliable_harm"]]
    exact_wins = wins.loc[wins["metric"].eq("exact_score_nll")].set_index(
        ["arm", "reference"]
    )["record"]
    return {
        "layer": "XG_FORM_ON_DOMESTIC_ATTACK_DEFENCE",
        "question": (
            "Does the xG form term still improve goal expectation once the "
            "domestic attack/defence transfer is already in the model?"
        ),
        "changes_production_parameters": False,
        "production_contract_sha256": contract_sha256,
        "training_window": training_window,
        "folds": len(folds),
        "test_seasons": [test for _, _, test in folds],
        "matches_per_arm": int(indexed["matches"].iloc[0]),
        "all_gates_passed": bool(gates["passed"].all()),
        "exact_score_nll": {
            str(arm): float(indexed.loc[arm, "exact_score_nll"]) for arm in indexed.index
        },
        "over_2_5_brier": {
            str(arm): float(indexed.loc[arm, "over_2_5_brier"]) for arm in indexed.index
        },
        "xg_beats_baseline_on_exact_score": bool(
            indexed.loc[DOMESTIC_AD_XG_FORM, "exact_score_nll"]
            < indexed.loc[DOMESTIC_AD, "exact_score_nll"]
        ),
        "xg_beats_goals_on_exact_score": bool(
            indexed.loc[DOMESTIC_AD_XG_FORM, "exact_score_nll"]
            < indexed.loc[DOMESTIC_AD_GOALS_FORM, "exact_score_nll"]
        ),
        "exact_score_fold_record": {
            f"{arm}_vs_{reference}": str(record)
            for (arm, reference), record in exact_wins.items()
        },
        "reliable_improvements": [
            {
                "arm": str(row.arm),
                "reference": str(row.reference),
                "segment": str(row.segment),
                "metric": str(row.metric),
                "mean_difference": float(row.mean_difference),
                "ci_95_lower": float(row.ci_95_lower),
                "ci_95_upper": float(row.ci_95_upper),
            }
            for row in reliable.itertuples(index=False)
        ],
        "xg_vs_goals_envelope": [
            {
                "segment": str(row.segment),
                "metric": str(row.metric),
                "mean_difference": float(row.mean_difference),
                "ci_95_lower": float(row.ci_95_lower),
                "ci_95_upper": float(row.ci_95_upper),
                "reliable_improvement": bool(row.reliable_improvement),
            }
            for row in envelope.loc[
                envelope["reference"].eq(DOMESTIC_AD_GOALS_FORM)
            ].itertuples(index=False)
        ],
        "reliable_harms": [
            {
                "arm": str(row.arm),
                "reference": str(row.reference),
                "segment": str(row.segment),
                "metric": str(row.metric),
                "mean_difference": float(row.mean_difference),
                "ci_95_lower": float(row.ci_95_lower),
                "ci_95_upper": float(row.ci_95_upper),
            }
            for row in harmful.itertuples(index=False)
        ],
        "baseline_reconciliation": [
            {
                key: None if pd.isna(value) else value
                for key, value in row.items()
            }
            for row in reconciliation.to_dict(orient="records")
        ],
        "python_version": platform.python_version(),
        "scipy_version": scipy.__version__,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def write_report(
    path: Path,
    summary: pd.DataFrame,
    segments: pd.DataFrame,
    folds_frame: pd.DataFrame,
    uncertainty: pd.DataFrame,
    selections: pd.DataFrame,
    wins: pd.DataFrame,
    reconciliation: pd.DataFrame,
    gates: pd.DataFrame,
    manifest: dict[str, object],
) -> None:
    envelope = uncertainty.loc[uncertainty["method"].eq("conservative_envelope")]
    against_baseline = envelope.loc[envelope["reference"].eq(DOMESTIC_AD)]
    head_to_head = envelope.loc[envelope["reference"].eq(DOMESTIC_AD_GOALS_FORM)]
    selection_view = selections[
        [
            "fold", "test_season", "inner_validation_season", "arm",
            "domestic_candidate_key", "selected_l2_strength", "attack_coefficient",
            "defence_coefficient", "venue_coefficient", "form_attack", "form_defence",
        ]
    ]
    lines = [
        "# xG Form Terimi Domestic Attack/Defence Uzerinde",
        "",
        "Onceki kosu xG form terimini **Elo-only** bir tabana ekledi ve xG'nin gol",
        "kontrolunu gectigini gosterdi. Ama o taban reponun mevcut en iyi skor",
        "kolundan basitti. Bu kosu ayni terimi",
        "`DOMESTIC_ATTACK_DEFENCE_POISSON` kolunun **uzerine** koyar, yerine degil.",
        "",
        "`DOMESTIC_AD_GOALS_FORM` kontroldur: onsuz xG kolundaki bir kazanc,",
        "xG'nin mi yoksa iki ek parametrenin mi katkisi oldugu ayirt edilemezdi.",
        "",
        f"Egitim penceresi: `{manifest['training_window']}`. "
        f"Fold sayisi: `{manifest['folds']}`. "
        f"Kol basina mac: `{manifest['matches_per_arm']}`.",
        "",
        "## Dogrulama kapilari",
        "",
        "```text",
        gates.to_string(index=False),
        "```",
        "",
        "## Taban uzlastirmasi",
        "",
        "Bu kosunun tabani yayinlanmis domestic kolunun zayiflatilmis bir",
        "kopyasi olmamalidir; asagidaki tablo ikisini ayni maclarda karsilastirir.",
        "",
        "```text",
        reconciliation.to_string(index=False, na_rep="n/a"),
        "```",
        "",
        "## Kol ozeti",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "## Segment kirilimi (domestic tabana karsi)",
        "",
        "```text",
        segments.to_string(index=False),
        "```",
        "",
        "## Fold bazinda",
        "",
        "```text",
        folds_frame.to_string(index=False),
        "```",
        "",
        "## Nested secimler",
        "",
        "```text",
        selection_view.to_string(index=False),
        "```",
        "",
        "## Fold kazanimlari",
        "",
        "```text",
        wins.to_string(index=False),
        "```",
        "",
        "## Dependency-robust belirsizlik: tabana karsi (conservative envelope)",
        "",
        "```text",
        against_baseline[
            [
                "arm", "segment", "metric", "mean_difference", "ci_95_lower",
                "ci_95_upper", "reliable_improvement", "reliable_harm",
            ]
        ].to_string(index=False),
        "```",
        "",
        "## Dependency-robust belirsizlik: xG kolu gol kontrolune karsi",
        "",
        "Iki kol da ayni iki ek parametreyi tasir. Aralarindaki guvenilir bir",
        "fark, yalniz form teriminin okundugu kaynaktan gelebilir.",
        "",
        "```text",
        head_to_head[
            [
                "segment", "metric", "mean_difference", "ci_95_lower",
                "ci_95_upper", "reliable_improvement", "reliable_harm",
            ]
        ].to_string(index=False),
        "```",
        "",
        "## Karar girdisi",
        "",
        f"- xG kolu tabani exact-score'da geciyor mu: "
        f"`{manifest['xg_beats_baseline_on_exact_score']}`.",
        f"- xG kolu gol kontrolunu geciyor mu: "
        f"`{manifest['xg_beats_goals_on_exact_score']}`.",
        f"- Exact-score fold kaydi: "
        f"`{manifest['exact_score_fold_record']}`.",
        f"- Guvenilir iyilesme sayisi: `{len(manifest['reliable_improvements'])}`.",
        f"- Guvenilir zarar sayisi: `{len(manifest['reliable_harms'])}`.",
        f"- Production parametresi degisti mi: "
        f"`{manifest['changes_production_parameters']}`.",
        "",
        "Karar urun tarafina aittir; bu belge yalniz kanit uretir.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
