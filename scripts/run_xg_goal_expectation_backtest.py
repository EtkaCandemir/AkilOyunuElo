from __future__ import annotations

"""Does recent xG predict goals better than recent goals do?

The scoreline layer is `Diagnostic` because it barely clears climatology: on
`4884` matches its best arm beats a constant base-rate forecast by `0.0018`
Brier on over-2.5 and `0.0003` on both-teams-to-score. The layer also derives
both rates from the Elo difference alone, with no team form at all.

This run adds one symmetric form term per side and feeds it from two sources.
The goals arm is the control that matters: without it a gain from the xG arm
could not be told apart from the gain of simply having a form term.

Walk-forward, expanding, selection never touches the test season. The script
changes no production parameter.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (str(SRC), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from ao_elo.dynamic import DynamicEloConfig  # noqa: E402
from ao_elo.evaluation import dependency_robust_loss_difference_ci  # noqa: E402
from ao_elo.scoreline import ScorelineModelConfig, scoreline_matrix  # noqa: E402
from ao_elo.xg_goal_model import (  # noqa: E402
    build_form_features,
    candidate_sources,
    fit_goal_expectation,
    predict_goal_expectations,
)


DEFAULT_XG = ROOT / "data" / "xg_2020_2026" / "uefa_2020_2026_matches_with_xg.csv"
DEFAULT_PREDICTIONS = (
    ROOT / "output" / "current_model_evaluation_2018_2026" / "model_predictions.csv"
)
DEFAULT_IDENTITY = ROOT / "data" / "club_identity" / "team_season_identity.csv"
DEFAULT_OUTPUT = ROOT / "output" / "xg_goal_expectation_backtest_2020_2026"
QUALIFYING = frozenset({
    "Preliminary Round", "1st Qualifying Round", "2nd Qualifying Round",
    "3rd Qualifying Round", "Qualifying Play-off Round",
})
BOOTSTRAP_SAMPLES = 4000
BOOTSTRAP_SEED = 20260819


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Elo-only, goals-form and xG-form goal expectations"
    )
    parser.add_argument("--xg-data", type=Path, default=DEFAULT_XG)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--identity", type=Path, default=DEFAULT_IDENTITY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    arguments = parser.parse_args()

    output_root = arguments.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    core = DynamicEloConfig.calibrated_v2()

    data = load_matches(arguments.xg_data, arguments.predictions, arguments.identity)
    seasons = sorted(data["season"].unique())
    folds = [(tuple(seasons[:index]), seasons[index]) for index in range(2, len(seasons))]

    predictions = walk_forward(data, folds, core)
    predictions.to_csv(output_root / "unseen_predictions.csv", index=False)

    summary = arm_summary(predictions)
    segments = segment_summary(predictions)
    folds_frame = fold_summary(predictions)
    uncertainty = uncertainty_summary(
        predictions, samples=int(arguments.bootstrap_samples)
    )
    gates = validation_gates(data, predictions)

    summary.to_csv(output_root / "arm_summary.csv", index=False)
    segments.to_csv(output_root / "segment_summary.csv", index=False)
    folds_frame.to_csv(output_root / "fold_summary.csv", index=False)
    uncertainty.to_csv(output_root / "uncertainty.csv", index=False)
    gates.to_csv(output_root / "validation_gates.csv", index=False)

    manifest = build_manifest(summary, segments, uncertainty, gates, folds)
    (output_root / "backtest_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(output_root / "backtest_report.md", summary, segments, folds_frame, uncertainty, gates, manifest)

    print(f"Wrote xG goal-expectation backtest to {output_root}")
    print(gates.to_string(index=False))
    print()
    print(summary.to_string(index=False))
    print()
    print(segments.to_string(index=False))


def load_matches(xg_path: Path, predictions_path: Path, identity_path: Path) -> pd.DataFrame:
    xg = pd.read_csv(xg_path)
    baseline = pd.read_csv(predictions_path)
    baseline = baseline.loc[baseline["model"].eq("CURRENT_PRODUCTION")]
    frame = xg[
        [
            "match_id", "season", "kickoff_utc", "competition", "round",
            "home_team_id", "away_team_id", "home_goals", "away_goals",
            "xg_home", "xg_away", "xg_analysis_eligible",
        ]
    ].merge(
        baseline[["match_id", "home_live_pre", "away_live_pre", "is_neutral"]],
        on="match_id", how="inner", validate="one_to_one",
    )
    identity = pd.read_csv(identity_path)[["season", "local_team_id", "club_id"]]
    identity["local_team_id"] = identity["local_team_id"].astype(str)
    for side in ("home", "away"):
        frame[f"{side}_team_id"] = frame[f"{side}_team_id"].astype(str)
        frame = frame.merge(
            identity.rename(
                columns={"local_team_id": f"{side}_team_id", "club_id": f"{side}_club_id"}
            ),
            on=["season", f"{side}_team_id"], how="left", validate="many_to_one",
        )
    if frame[["home_club_id", "away_club_id"]].isna().any().any():
        raise ValueError("club identity is incomplete")
    frame["phase"] = np.where(frame["round"].isin(QUALIFYING), "QUALIFYING", "MAIN")
    frame["kickoff_utc"] = pd.to_datetime(frame["kickoff_utc"], utc=True, format="ISO8601")
    return frame.sort_values(["kickoff_utc", "match_id"], kind="stable").reset_index(drop=True)


def walk_forward(
    data: pd.DataFrame,
    folds: list[tuple[tuple[str, ...], str]],
    core: DynamicEloConfig,
) -> pd.DataFrame:
    """Fit on training seasons only, then score the held-out season."""

    rows = []
    for source in candidate_sources():
        featured = build_form_features(data, source)
        for train_seasons, test_season in folds:
            train = featured.loc[featured["season"].isin(train_seasons)]
            test = featured.loc[featured["season"].eq(test_season)]
            config = fit_goal_expectation(
                data.loc[data["season"].isin(train_seasons)],
                source,
                elo_scale=core.elo_scale,
                home_advantage=core.home_advantage,
            )
            rates = predict_goal_expectations(
                test, config, elo_scale=core.elo_scale, home_advantage=core.home_advantage
            )
            block = test[
                ["match_id", "season", "competition", "phase", "home_team_id",
                 "away_team_id", "kickoff_utc", "home_goals", "away_goals",
                 "xg_analysis_eligible"]
            ].merge(rates, on="match_id", validate="one_to_one")
            block["arm"] = source.upper() if source != "none" else "ELO_ONLY"
            block["fold_train_seasons"] = "|".join(train_seasons)
            block["attack"] = config.attack
            block["defence"] = config.defence
            rows.append(block)
    predictions = pd.concat(rows, ignore_index=True)
    predictions["tie_id"] = predictions["match_id"]
    return add_losses(predictions)


def add_losses(frame: pd.DataFrame) -> pd.DataFrame:
    matrix_config = ScorelineModelConfig(mu=0.0, elo_slope=1.0, rho=0.0)
    exact, over, btts, total = [], [], [], []
    for row in frame.itertuples(index=False):
        matrix, _ = scoreline_matrix(float(row.lambda_home), float(row.lambda_away), matrix_config)
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
    result["over_2_5_brier"] = (result["over_2_5_probability"] - result["over_2_5_actual"]) ** 2
    result["btts_actual"] = (
        (result["home_goals"] > 0) & (result["away_goals"] > 0)
    ).astype(float)
    result["btts_probability"] = btts
    result["btts_brier"] = (result["btts_probability"] - result["btts_actual"]) ** 2
    return result


METRICS = ("exact_score_nll", "total_goals_error", "over_2_5_brier", "btts_brier")


def arm_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm, block in predictions.groupby("arm"):
        record = {"arm": arm, "matches": int(len(block))}
        record.update({metric: float(block[metric].mean()) for metric in METRICS})
        rows.append(record)
    frame = pd.DataFrame(rows)
    baseline = frame.loc[frame["arm"].eq("ELO_ONLY")].iloc[0]
    for metric in METRICS:
        frame[f"{metric}_vs_elo"] = frame[metric] - baseline[metric]
    return frame.sort_values("exact_score_nll").reset_index(drop=True)


def segment_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    baseline = predictions.loc[predictions["arm"].eq("ELO_ONLY")].set_index("match_id")
    rows = []
    for arm in ("GOALS", "XG"):
        block = predictions.loc[predictions["arm"].eq(arm)].set_index("match_id")
        joined = block.join(baseline[list(METRICS)], rsuffix="_base")
        joined["has_xg"] = joined["xg_analysis_eligible"].astype(bool)
        for label, subset in (
            ("ALL", joined),
            ("PHASE:MAIN", joined.loc[joined["phase"].eq("MAIN")]),
            ("PHASE:QUALIFYING", joined.loc[joined["phase"].eq("QUALIFYING")]),
            ("XG_PRESENT", joined.loc[joined["has_xg"]]),
            ("XG_ABSENT", joined.loc[~joined["has_xg"]]),
        ):
            if subset.empty:
                continue
            record = {"arm": arm, "segment": label, "matches": int(len(subset))}
            for metric in METRICS:
                record[f"{metric}_vs_elo"] = float(
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
    return pd.DataFrame(rows).sort_values(["test_season", "arm"]).reset_index(drop=True)


def uncertainty_summary(predictions: pd.DataFrame, *, samples: int) -> pd.DataFrame:
    baseline = predictions.loc[predictions["arm"].eq("ELO_ONLY")].set_index("match_id")
    blocks = []
    for arm in ("GOALS", "XG"):
        block = predictions.loc[predictions["arm"].eq(arm)].set_index("match_id")
        joined = block.join(baseline[list(METRICS)], rsuffix="_base").reset_index()
        joined["has_xg"] = joined["xg_analysis_eligible"].astype(bool)
        for label, subset in (
            ("ALL", joined),
            ("XG_PRESENT", joined.loc[joined["has_xg"]]),
        ):
            for metric in ("exact_score_nll", "over_2_5_brier"):
                sample = subset[
                    ["season", "match_id", "home_team_id", "away_team_id", "kickoff_utc", "tie_id"]
                ].copy()
                sample["loss_difference"] = (
                    subset[metric] - subset[f"{metric}_base"]
                ).to_numpy(float)
                result = dependency_robust_loss_difference_ci(
                    sample, bootstrap_samples=samples, seed=BOOTSTRAP_SEED
                )
                result.insert(0, "arm", arm)
                result.insert(1, "segment", label)
                result.insert(2, "metric", metric)
                blocks.append(result)
    return pd.concat(blocks, ignore_index=True)


def validation_gates(data: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    arms = set(predictions["arm"])
    per_arm = predictions.groupby("arm")["match_id"].nunique()
    xg_arm = predictions.loc[predictions["arm"].eq("XG")]
    return pd.DataFrame(
        [
            {
                "gate": "three_arms_scored",
                "passed": arms == {"ELO_ONLY", "GOALS", "XG"},
                "observed": len(arms),
                "requirement": "Elo-only baseline, goals control and xG candidate",
            },
            {
                "gate": "arms_share_one_sample",
                "passed": bool(per_arm.nunique() == 1),
                "observed": int(per_arm.iloc[0]),
                "requirement": "Every arm scores the identical held-out matches",
            },
            {
                "gate": "elo_only_carries_no_form",
                "passed": bool(
                    predictions.loc[predictions["arm"].eq("ELO_ONLY"), "attack"].eq(0).all()
                ),
                "observed": 0.0,
                "requirement": "The baseline must stay a two-parameter model",
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
        ]
    )


def build_manifest(summary, segments, uncertainty, gates, folds) -> dict[str, object]:
    indexed = summary.set_index("arm")
    envelope = uncertainty.loc[uncertainty["method"].eq("conservative_envelope")]
    reliable = envelope.loc[envelope["reliable_improvement"]]
    return {
        "layer": "XG_INFORMED_GOAL_EXPECTATION",
        "changes_production_parameters": False,
        "folds": len(folds),
        "all_gates_passed": bool(gates["passed"].all()),
        "exact_score_nll": {
            str(arm): float(indexed.loc[arm, "exact_score_nll"]) for arm in indexed.index
        },
        "over_2_5_brier": {
            str(arm): float(indexed.loc[arm, "over_2_5_brier"]) for arm in indexed.index
        },
        "reliable_improvements": [
            {"arm": str(row.arm), "segment": str(row.segment), "metric": str(row.metric),
             "mean_difference": float(row.mean_difference)}
            for row in reliable.itertuples(index=False)
        ],
        "xg_beats_goals_on_exact_score": bool(
            indexed.loc["XG", "exact_score_nll"] < indexed.loc["GOALS", "exact_score_nll"]
        ),
    }


def write_report(path, summary, segments, folds_frame, uncertainty, gates, manifest) -> None:
    envelope = uncertainty.loc[uncertainty["method"].eq("conservative_envelope")]
    lines = [
        "# xG Bilgili Gol Beklentisi Backtest",
        "",
        "Aktif skor katmani her iki orani yalniz Elo farkindan turetir ve",
        "`Diagnostic` statusundedir; climatology'yi ust 2.5'te `0.0018` Brier",
        "gecer. Bu kosu her iki tarafa birer form terimi ekler ve ayni yapiyi iki",
        "farkli kaynakla besler.",
        "",
        "`GOALS` kolu kontroldur: onsuz `XG` kolundaki bir kazanc, xG'nin mi yoksa",
        "form teriminin mi katkisi oldugu ayirt edilemezdi.",
        "",
        "## Dogrulama kapilari",
        "",
        "```text",
        gates.to_string(index=False),
        "```",
        "",
        "## Kol ozeti",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "## Segment kirilimi (Elo-only'ye karsi)",
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
        "## Dependency-robust belirsizlik (conservative envelope)",
        "",
        "```text",
        envelope[["arm", "segment", "metric", "mean_difference", "ci_95_lower",
                  "ci_95_upper", "reliable_improvement", "reliable_harm"]].to_string(index=False),
        "```",
        "",
        "## Karar girdisi",
        "",
        f"- xG exact-score'da gol formunu geciyor mu: `{manifest['xg_beats_goals_on_exact_score']}`.",
        f"- Guvenilir iyilesme sayisi: `{len(manifest['reliable_improvements'])}`.",
        "",
        "Karar urun tarafina aittir; bu belge yalniz kanit uretir.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
