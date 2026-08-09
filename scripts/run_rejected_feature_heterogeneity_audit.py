from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "output" / "rejected_feature_heterogeneity_2018_2026"
EVENTS_PATH = (
    ROOT / "data" / "external_elo_benchmark_2018_2026" / "exact_date_events.csv"
)


@dataclass(frozen=True)
class FeatureSource:
    feature: str
    path: Path
    layout: str
    candidate_model: str | None = None
    baseline_model: str | None = None


SOURCES = (
    FeatureSource(
        "contextual_draw",
        ROOT
        / "output/match_context_backtest_2018_2026/contextual_draw/unseen_predictions.csv",
        "paired",
    ),
    FeatureSource(
        "dynamic_k",
        ROOT / "output/dynamic_k_backtest_2018_2026/unseen_predictions.csv",
        "long",
        "DYNAMIC_K",
        "BASE",
    ),
    FeatureSource(
        "progression_bonus",
        ROOT
        / "output/controlled_goal_progression_backtest_2018_2026/unseen_predictions.csv",
        "long",
        "PROGRESSION_ONLY",
        "BASE",
    ),
    FeatureSource(
        "competition_k",
        ROOT
        / "output/final_robustness_2018_2026/competition_k/unseen_predictions.csv",
        "paired",
    ),
    FeatureSource(
        "achievement_reserve",
        ROOT
        / "output/progression_reserve_final_backtest_2018_2026/unseen_predictions.csv",
        "paired",
    ),
    FeatureSource(
        "domestic_regression_shadow",
        ROOT
        / "output/match_context_backtest_2018_2026/domestic_regression/unseen_predictions.csv",
        "paired",
    ),
)


NO_SELECTION_SOURCES = {
    "aggregate_state": ROOT
    / "output/match_context_backtest_2018_2026/aggregate_state/fold_selections.csv",
    "dynamic_home": ROOT
    / "output/match_context_backtest_2018_2026/dynamic_home/fold_selections.csv",
}


def main() -> None:
    events = pd.read_csv(EVENTS_PATH)[
        [
            "match_id",
            "home_team_name",
            "away_team_name",
        ]
    ]
    summaries = []
    fold_frames = []
    competition_frames = []
    season_frames = []
    leave_one_out_frames = []

    for source in SOURCES:
        predictions = load_source(source)
        predictions = predictions.merge(events, on="match_id", validate="one_to_one")
        overall = metric_summary(predictions)
        fold = grouped_summary(predictions, "fold")
        competition = grouped_summary(predictions, "competition")
        season = grouped_summary(predictions, "season")
        leave_one_out = build_leave_one_out(predictions)
        classification = classify_feature(
            overall,
            fold,
            competition,
            leave_one_out,
        )
        summaries.append(
            {
                "feature": source.feature,
                **overall,
                "harmful_folds": int(fold["both_metrics_harm"].sum()),
                "improving_folds": int(fold["both_metrics_improve"].sum()),
                "harmful_competitions": int(
                    competition["both_metrics_harm"].sum()
                ),
                "improving_competitions": int(
                    competition["both_metrics_improve"].sum()
                ),
                "team_sign_flips": int(
                    leave_one_out.loc[
                        leave_one_out["cluster_type"].eq("team"),
                        "both_metrics_sign_flip",
                    ].sum()
                ),
                "season_sign_flips": int(
                    leave_one_out.loc[
                        leave_one_out["cluster_type"].eq("season"),
                        "both_metrics_sign_flip",
                    ].sum()
                ),
                "competition_sign_flips": int(
                    leave_one_out.loc[
                        leave_one_out["cluster_type"].eq("competition"),
                        "both_metrics_sign_flip",
                    ].sum()
                ),
                **classification,
            }
        )
        fold.insert(0, "feature", source.feature)
        competition.insert(0, "feature", source.feature)
        season.insert(0, "feature", source.feature)
        leave_one_out.insert(0, "feature", source.feature)
        fold_frames.append(fold)
        competition_frames.append(competition)
        season_frames.append(season)
        leave_one_out_frames.append(leave_one_out)

    for feature, path in NO_SELECTION_SOURCES.items():
        selections = pd.read_csv(path)
        all_baseline = bool(selections["selected_is_baseline"].all())
        summaries.append(
            {
                "feature": feature,
                "matches": 4884,
                "brier_difference": 0.0,
                "log_loss_difference": 0.0,
                "harm_match_share": 0.0,
                "harmful_folds": 0,
                "improving_folds": 0,
                "harmful_competitions": 0,
                "improving_competitions": 0,
                "team_sign_flips": 0,
                "season_sign_flips": 0,
                "competition_sign_flips": 0,
                "classification": (
                    "NO_NONBASELINE_CANDIDATE_SELECTED"
                    if all_baseline
                    else "MIXED_SELECTION"
                ),
                "outlier_driven": False,
                "segment_candidate": False,
            }
        )

    summary = pd.DataFrame(summaries).sort_values("feature").reset_index(drop=True)
    folds = pd.concat(fold_frames, ignore_index=True)
    competitions = pd.concat(competition_frames, ignore_index=True)
    seasons = pd.concat(season_frames, ignore_index=True)
    leave_one_out = pd.concat(leave_one_out_frames, ignore_index=True)
    output_root = OUTPUT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_root / "feature_classification.csv", index=False)
    folds.to_csv(output_root / "fold_summary.csv", index=False)
    competitions.to_csv(output_root / "competition_summary.csv", index=False)
    seasons.to_csv(output_root / "season_summary.csv", index=False)
    leave_one_out.to_csv(output_root / "leave_one_cluster_out.csv", index=False)
    manifest = {
        "data_window": "2020/21-2025/26 unseen folds",
        "matches_per_feature": 4884,
        "classification": summary.to_dict(orient="records"),
        "interpretation_contract": {
            "broad_harm": "harm is stable across folds and competitions",
            "heterogeneous_harm": "global harm exists but differs by segment",
            "outlier_sensitive": "removing one pre-defined cluster reverses both losses",
            "no_candidate_selected": "ranking-first training chose baseline in every fold",
            "shadow_benefit": "loss improves but another promotion guardrail fails",
        },
        "post_hoc_segment_activation_allowed": False,
        "production_change": False,
    }
    (output_root / "audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=json_default), encoding="utf-8"
    )
    write_report(
        output_root / "audit_report.md",
        summary,
        folds,
        competitions,
        leave_one_out,
    )
    print("Rejected-feature heterogeneity audit complete")
    print(summary[["feature", "classification", "outlier_driven"]].to_string(index=False))
    print(f"Report: {output_root / 'audit_report.md'}")


def load_source(source: FeatureSource) -> pd.DataFrame:
    data = pd.read_csv(source.path)
    if source.layout == "paired":
        required = {
            "fold",
            "match_id",
            "season",
            "competition",
            "home_team_id",
            "away_team_id",
            "candidate_brier_1x2",
            "baseline_brier_1x2",
            "candidate_log_loss_1x2",
            "baseline_log_loss_1x2",
        }
        missing = sorted(required - set(data.columns))
        if missing:
            raise ValueError(f"{source.feature} paired data missing: {missing}")
        result = data.copy()
    elif source.layout == "long":
        required = {
            "fold",
            "model",
            "match_id",
            "season",
            "competition",
            "home_team_id",
            "away_team_id",
            "brier_1x2",
            "log_loss_1x2",
        }
        missing = sorted(required - set(data.columns))
        if missing:
            raise ValueError(f"{source.feature} long data missing: {missing}")
        candidate = data.loc[data["model"].eq(source.candidate_model)].copy()
        baseline = data.loc[data["model"].eq(source.baseline_model)].copy()
        if len(candidate) != len(baseline):
            raise ValueError(f"{source.feature} candidate/baseline row mismatch")
        candidate = candidate.rename(
            columns={
                "brier_1x2": "candidate_brier_1x2",
                "log_loss_1x2": "candidate_log_loss_1x2",
            }
        )
        baseline = baseline[["match_id", "brier_1x2", "log_loss_1x2"]].rename(
            columns={
                "brier_1x2": "baseline_brier_1x2",
                "log_loss_1x2": "baseline_log_loss_1x2",
            }
        )
        result = candidate.merge(baseline, on="match_id", validate="one_to_one")
    else:
        raise ValueError(f"Unknown source layout: {source.layout}")
    if result["match_id"].duplicated().any():
        raise ValueError(f"{source.feature} contains duplicate match_id")
    result["brier_difference"] = (
        result["candidate_brier_1x2"] - result["baseline_brier_1x2"]
    )
    result["log_loss_difference"] = (
        result["candidate_log_loss_1x2"] - result["baseline_log_loss_1x2"]
    )
    return result[[
        "fold",
        "match_id",
        "season",
        "competition",
        "home_team_id",
        "away_team_id",
        "brier_difference",
        "log_loss_difference",
    ]]


def metric_summary(data: pd.DataFrame) -> dict[str, float | int]:
    return {
        "matches": len(data),
        "brier_difference": float(data["brier_difference"].mean()),
        "log_loss_difference": float(data["log_loss_difference"].mean()),
        "harm_match_share": float(
            (
                data["brier_difference"].gt(0.0)
                & data["log_loss_difference"].gt(0.0)
            ).mean()
        ),
    }


def grouped_summary(data: pd.DataFrame, column: str) -> pd.DataFrame:
    summary = (
        data.groupby(column, sort=True)
        .agg(
            matches=("match_id", "size"),
            brier_difference=("brier_difference", "mean"),
            log_loss_difference=("log_loss_difference", "mean"),
        )
        .reset_index()
    )
    summary["both_metrics_harm"] = (
        summary["brier_difference"].gt(0.0)
        & summary["log_loss_difference"].gt(0.0)
    )
    summary["both_metrics_improve"] = (
        summary["brier_difference"].lt(0.0)
        & summary["log_loss_difference"].lt(0.0)
    )
    return summary


def build_leave_one_out(data: pd.DataFrame) -> pd.DataFrame:
    total_count = len(data)
    total_brier = float(data["brier_difference"].sum())
    total_log = float(data["log_loss_difference"].sum())
    overall_brier = total_brier / total_count
    overall_log = total_log / total_count
    rows = []
    for cluster_type, column in (
        ("fold", "fold"),
        ("season", "season"),
        ("competition", "competition"),
    ):
        grouped = data.groupby(column, sort=True).agg(
            removed_matches=("match_id", "size"),
            removed_brier_sum=("brier_difference", "sum"),
            removed_log_sum=("log_loss_difference", "sum"),
        )
        for key, row in grouped.iterrows():
            rows.append(
                leave_one_out_row(
                    cluster_type,
                    str(key),
                    None,
                    int(row["removed_matches"]),
                    float(row["removed_brier_sum"]),
                    float(row["removed_log_sum"]),
                    total_count,
                    total_brier,
                    total_log,
                    overall_brier,
                    overall_log,
                )
            )

    long = pd.concat(
        [
            data[[
                "match_id",
                "home_team_id",
                "home_team_name",
                "brier_difference",
                "log_loss_difference",
            ]].rename(
                columns={
                    "home_team_id": "team_id",
                    "home_team_name": "team_name",
                }
            ),
            data[[
                "match_id",
                "away_team_id",
                "away_team_name",
                "brier_difference",
                "log_loss_difference",
            ]].rename(
                columns={
                    "away_team_id": "team_id",
                    "away_team_name": "team_name",
                }
            ),
        ],
        ignore_index=True,
    )
    grouped_team = long.groupby(["team_id", "team_name"], sort=True).agg(
        removed_matches=("match_id", "size"),
        removed_brier_sum=("brier_difference", "sum"),
        removed_log_sum=("log_loss_difference", "sum"),
    )
    for (team_id, team_name), row in grouped_team.iterrows():
        rows.append(
            leave_one_out_row(
                "team",
                str(int(team_id)),
                str(team_name),
                int(row["removed_matches"]),
                float(row["removed_brier_sum"]),
                float(row["removed_log_sum"]),
                total_count,
                total_brier,
                total_log,
                overall_brier,
                overall_log,
            )
        )
    return pd.DataFrame(rows)


def leave_one_out_row(
    cluster_type: str,
    cluster_id: str,
    cluster_name: str | None,
    removed_matches: int,
    removed_brier_sum: float,
    removed_log_sum: float,
    total_count: int,
    total_brier: float,
    total_log: float,
    overall_brier: float,
    overall_log: float,
) -> dict[str, object]:
    remaining = total_count - removed_matches
    if remaining <= 0:
        raise ValueError("Leave-one-out cluster removes every match")
    brier = (total_brier - removed_brier_sum) / remaining
    log_loss = (total_log - removed_log_sum) / remaining
    if overall_brier > 0.0 and overall_log > 0.0:
        sign_flip = brier <= 0.0 and log_loss <= 0.0
    elif overall_brier < 0.0 and overall_log < 0.0:
        sign_flip = brier >= 0.0 and log_loss >= 0.0
    else:
        sign_flip = False
    return {
        "cluster_type": cluster_type,
        "cluster_id": cluster_id,
        "cluster_name": cluster_name,
        "removed_matches": removed_matches,
        "overall_brier_difference": overall_brier,
        "remaining_brier_difference": brier,
        "overall_log_loss_difference": overall_log,
        "remaining_log_loss_difference": log_loss,
        "both_metrics_sign_flip": sign_flip,
    }


def classify_feature(
    overall: dict[str, float | int],
    folds: pd.DataFrame,
    competitions: pd.DataFrame,
    leave_one_out: pd.DataFrame,
) -> dict[str, object]:
    brier = float(overall["brier_difference"])
    log_loss = float(overall["log_loss_difference"])
    team_flips = int(
        leave_one_out.loc[
            leave_one_out["cluster_type"].eq("team"),
            "both_metrics_sign_flip",
        ].sum()
    )
    season_flips = int(
        leave_one_out.loc[
            leave_one_out["cluster_type"].eq("season"),
            "both_metrics_sign_flip",
        ].sum()
    )
    competition_flips = int(
        leave_one_out.loc[
            leave_one_out["cluster_type"].eq("competition"),
            "both_metrics_sign_flip",
        ].sum()
    )
    outlier_driven = team_flips > 0
    segment_sensitive = season_flips > 0 or competition_flips > 0
    harmful_folds = int(folds["both_metrics_harm"].sum())
    harmful_competitions = int(competitions["both_metrics_harm"].sum())
    if brier < 0.0 and log_loss < 0.0:
        classification = "SHADOW_BENEFIT_OTHER_GUARDRAIL_FAILED"
    elif brier > 0.0 and log_loss > 0.0:
        if outlier_driven:
            classification = "OUTLIER_SENSITIVE_GLOBAL_HARM"
        elif segment_sensitive:
            classification = "SEGMENT_SENSITIVE_GLOBAL_HARM"
        elif harmful_folds >= 4 and harmful_competitions == len(competitions):
            classification = "BROAD_SYSTEMATIC_HARM"
        else:
            classification = "HETEROGENEOUS_GLOBAL_HARM"
    else:
        classification = "MIXED_LOSS_DIRECTION"
    segment_candidate = bool(
        classification in (
            "OUTLIER_SENSITIVE_GLOBAL_HARM",
            "SEGMENT_SENSITIVE_GLOBAL_HARM",
            "HETEROGENEOUS_GLOBAL_HARM",
        )
        and competitions["both_metrics_improve"].any()
    )
    return {
        "classification": classification,
        "outlier_driven": outlier_driven,
        "segment_candidate": segment_candidate,
    }


def write_report(
    path: Path,
    summary: pd.DataFrame,
    folds: pd.DataFrame,
    competitions: pd.DataFrame,
    leave_one_out: pd.DataFrame,
) -> None:
    lines = [
        "# Rejected Feature Heterogeneity Audit",
        "",
        "## Question",
        "",
        "Were globally rejected features actually harmed by only a few teams or segments?",
        "",
        "## Classification",
        "",
        markdown_table(summary, float_digits=6),
        "",
        "## Competition Direction",
        "",
        markdown_table(competitions, float_digits=6),
        "",
        "## Fold Direction",
        "",
        markdown_table(folds, float_digits=6),
        "",
        "## Sign-Flipping Clusters",
        "",
    ]
    flips = leave_one_out.loc[leave_one_out["both_metrics_sign_flip"]].copy()
    lines.append(
        markdown_table(flips, float_digits=6)
        if len(flips)
        else "No single team, season, fold, or competition reverses both loss metrics."
    )
    lines.extend(
        [
            "",
            "Segment benefits are diagnostic only. A segment cannot be activated from this "
            "post-hoc audit; its rule must be pre-registered and pass a new nested walk-forward "
            "test against the unchanged production baseline.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame, *, float_digits: int) -> str:
    if frame.empty:
        return "No rows."
    columns = list(frame.columns)

    def format_value(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{float_digits}f}"
        return str(value).replace("|", "\\|")

    rows = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    rows.extend(
        "| " + " | ".join(format_value(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(rows)


def json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


if __name__ == "__main__":
    main()
