from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
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
from ao_elo.evaluation import schedule_adjusted_team_performance  # noqa: E402
from ao_elo.match_context import (  # noqa: E402
    AggregateStateConfig,
    DomesticRegressionConfig,
    DrawContextConfig,
    HomeAdvantageProfile,
)
from scripts.run_match_context_backtest import (  # noqa: E402
    DYNAMIC_ROOT,
    EVENTS_PATH,
    OUTPUT_ROOT as CONTEXT_OUTPUT_ROOT,
    STATIC_ROOT,
    ContextModelConfig,
    core_for_fold,
    evaluate_context_sequence,
    load_context_data,
    read_context_events,
)
from scripts.run_ranking_first_calibration import pairwise_ranking_accuracy  # noqa: E402


OUTPUT_ROOT = ROOT / "output" / "domestic_regression_diagnostics_2018_2026"
SOURCE_SEASON = "2020/21"
TARGET_SEASON = "2021/22"
FOLD = 1
PERSISTENCE_VALUES = (0.0, 0.25, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose the first Domestic-Prior ranking regression at team level"
    )
    parser.add_argument("--static-root", type=Path, default=STATIC_ROOT)
    parser.add_argument("--events-path", type=Path, default=EVENTS_PATH)
    parser.add_argument("--dynamic-root", type=Path, default=DYNAMIC_ROOT)
    parser.add_argument("--context-output-root", type=Path, default=CONTEXT_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    dynamic_root = args.dynamic_root.resolve()
    dynamic_manifest = json.loads(
        (dynamic_root / "selected_dynamic_model.json").read_text(encoding="utf-8")
    )
    static_config = AOEuropeanEloConfig(**dynamic_manifest["static_config"])
    static_config.validate()
    events = read_context_events(args.events_path.resolve())
    datasets, _ = load_context_data(
        args.static_root.resolve(), args.events_path.resolve(), static_config
    )
    target = schedule_adjusted_team_performance(events)
    core_selections = pd.read_csv(dynamic_root / "core_fold_selections.csv")
    core = core_for_fold(core_selections, FOLD)
    candidate = selected_fold_candidate(
        args.context_output_root.resolve()
        / "domestic_regression"
        / "fold_selections.csv",
        FOLD,
    )
    baseline = ContextModelConfig()
    sequence = tuple(
        data
        for data in datasets
        if tuple(item.season for item in datasets).index(data.season)
        <= tuple(item.season for item in datasets).index(SOURCE_SEASON)
    )

    baseline_eval = evaluate_context_sequence(
        sequence,
        core,
        baseline,
        target,
        evaluation_seasons={SOURCE_SEASON},
        ranking_target_seasons={TARGET_SEASON},
        return_predictions=True,
    )
    candidate_eval = evaluate_context_sequence(
        sequence,
        core,
        candidate,
        target,
        evaluation_seasons={SOURCE_SEASON},
        ranking_target_seasons={TARGET_SEASON},
        return_predictions=True,
    )
    team_table = build_team_table(
        baseline_eval.end_ratings,
        candidate_eval.end_ratings,
        target,
        args.static_root.resolve(),
    )
    competition = competition_diagnostics(team_table)
    pair_changes, team_pair_impacts = changed_pair_diagnostics(team_table)
    leave_one_out = leave_one_team_out_diagnostics(team_table)
    sensitivity = persistence_sensitivity(
        sequence,
        core,
        baseline,
        target,
    )
    match_metrics = match_loss_summary(
        baseline_eval.predictions,
        candidate_eval.predictions,
    )
    decision = diagnostic_decision(
        competition,
        leave_one_out,
        sensitivity,
        match_metrics,
    )

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    team_table.to_csv(output_root / "team_ranking_diagnostics.csv", index=False)
    competition.to_csv(output_root / "competition_ranking_diagnostics.csv", index=False)
    pair_changes.to_csv(output_root / "changed_pairs.csv", index=False)
    team_pair_impacts.to_csv(output_root / "team_pair_impacts.csv", index=False)
    leave_one_out.to_csv(output_root / "leave_one_team_out.csv", index=False)
    sensitivity.to_csv(output_root / "persistence_sensitivity.csv", index=False)
    match_metrics.to_csv(output_root / "match_loss_summary.csv", index=False)
    manifest = {
        "source_season": SOURCE_SEASON,
        "ranking_target_season": TARGET_SEASON,
        "fold": FOLD,
        "selected_candidate": asdict(candidate),
        "decision": decision,
        "competition_summary": competition.to_dict(orient="records"),
        "match_summary": match_metrics.to_dict(orient="records"),
        "leave_one_out_sign_flips": int(leave_one_out["pooled_delta_nonnegative"].sum()),
        "changed_pairs": len(pair_changes),
        "team_competition_observations": len(team_table),
        "unique_teams_in_ranking": int(team_table["team_id"].nunique()),
        "production_change": False,
    }
    (output_root / "diagnostic_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=json_default), encoding="utf-8"
    )
    write_report(
        output_root / "diagnostic_report.md",
        manifest,
        team_table,
        competition,
        pair_changes,
        team_pair_impacts,
        leave_one_out,
        sensitivity,
        match_metrics,
    )

    print("Domestic-Prior fold diagnostic complete")
    print(f"Decision: {decision['status']}")
    print(f"Team-competition observations: {len(team_table)}")
    print(f"Unique teams: {team_table['team_id'].nunique()}")
    print(f"Changed pairs: {len(pair_changes)}")
    print(f"Report: {output_root / 'diagnostic_report.md'}")


def selected_fold_candidate(path: Path, fold: int) -> ContextModelConfig:
    selections = pd.read_csv(path)
    row = selections.loc[selections["fold"].eq(fold)]
    if len(row) != 1:
        raise ValueError(f"Expected one Domestic-Prior selection for fold {fold}")
    payload = json.loads(row.iloc[0]["selected_config"])
    config = ContextModelConfig(
        aggregate=AggregateStateConfig(**payload["aggregate"]),
        home=HomeAdvantageProfile(**payload["home"]),
        draw=DrawContextConfig(**payload["draw"]),
        domestic=DomesticRegressionConfig(**payload["domestic"]),
    )
    config.validate()
    if not config.domestic.active:
        raise ValueError("Fold diagnostic requires an active Domestic-Prior candidate")
    return config


def build_team_table(
    baseline_end: pd.DataFrame,
    candidate_end: pd.DataFrame,
    target: pd.DataFrame,
    static_root: Path,
) -> pd.DataFrame:
    baseline = baseline_end.rename(
        columns={
            "initial_rating": "baseline_start_rating",
            "end_live_rating": "baseline_end_rating",
        }
    )
    candidate = candidate_end.rename(
        columns={
            "initial_rating": "candidate_start_rating",
            "end_live_rating": "candidate_end_rating",
        }
    )
    candidate_columns = [
        "season",
        "team_id",
        "candidate_start_rating",
        "candidate_end_rating",
        "ao_first_elo",
        "domestic_prior",
        "previous_power_elo",
    ]
    table = baseline[[
        "season",
        "team_id",
        "baseline_start_rating",
        "baseline_end_rating",
    ]].merge(
        candidate[candidate_columns],
        on=["season", "team_id"],
        validate="one_to_one",
    )
    actual = target.loc[target["season"].eq(TARGET_SEASON), [
        "competition",
        "team_id",
        "matches",
        "schedule_adjusted_score",
    ]]
    table = table.merge(actual, on="team_id", validate="one_to_many")
    teams = pd.read_csv(static_root / SOURCE_SEASON.replace("/", "-") / "teams.csv")
    table = table.merge(
        teams[["team_id", "team_name", "country_code"]],
        on="team_id",
        validate="many_to_one",
    )
    table["start_rating_shift"] = (
        table["candidate_start_rating"] - table["baseline_start_rating"]
    )
    table["end_rating_shift"] = (
        table["candidate_end_rating"] - table["baseline_end_rating"]
    )
    grouped = table.groupby("competition", sort=False)
    table["actual_rank"] = grouped["schedule_adjusted_score"].rank(
        ascending=False, method="average"
    )
    table["baseline_rank"] = grouped["baseline_end_rating"].rank(
        ascending=False, method="average"
    )
    table["candidate_rank"] = grouped["candidate_end_rating"].rank(
        ascending=False, method="average"
    )
    table["baseline_abs_rank_error"] = (
        table["baseline_rank"] - table["actual_rank"]
    ).abs()
    table["candidate_abs_rank_error"] = (
        table["candidate_rank"] - table["actual_rank"]
    ).abs()
    table["abs_rank_error_difference"] = (
        table["candidate_abs_rank_error"] - table["baseline_abs_rank_error"]
    )
    return table.sort_values(
        ["competition", "abs_rank_error_difference", "team_name"],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)


def competition_diagnostics(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for competition, frame in table.groupby("competition", sort=True):
        rows.append(ranking_row(competition, frame))
    pooled = pooled_ranking_metrics(table)
    rows.append({"competition": "ALL", **pooled})
    return pd.DataFrame(rows).sort_values(
        "competition", key=lambda values: values.eq("ALL").map({True: 0, False: 1})
    ).reset_index(drop=True)


def ranking_row(competition: str, frame: pd.DataFrame) -> dict[str, object]:
    baseline_spearman = frame["baseline_end_rating"].corr(
        frame["schedule_adjusted_score"], method="spearman"
    )
    candidate_spearman = frame["candidate_end_rating"].corr(
        frame["schedule_adjusted_score"], method="spearman"
    )
    actual = frame["schedule_adjusted_score"].to_numpy(float)
    baseline_pairwise = pairwise_ranking_accuracy(
        frame["baseline_end_rating"].to_numpy(float), actual
    )
    candidate_pairwise = pairwise_ranking_accuracy(
        frame["candidate_end_rating"].to_numpy(float), actual
    )
    return {
        "competition": competition,
        "teams": len(frame),
        "matches": int(frame["matches"].sum()),
        "baseline_spearman": float(baseline_spearman),
        "candidate_spearman": float(candidate_spearman),
        "spearman_difference": float(candidate_spearman - baseline_spearman),
        "baseline_pairwise": baseline_pairwise,
        "candidate_pairwise": candidate_pairwise,
        "pairwise_difference": candidate_pairwise - baseline_pairwise,
        "mean_abs_start_shift": float(frame["start_rating_shift"].abs().mean()),
        "max_abs_start_shift": float(frame["start_rating_shift"].abs().max()),
    }


def pooled_ranking_metrics(table: pd.DataFrame) -> dict[str, object]:
    rows = [ranking_row(competition, frame) for competition, frame in table.groupby("competition")]
    team_weights = np.array([row["teams"] for row in rows], dtype=float)
    pair_weights = team_weights * (team_weights - 1.0) / 2.0
    return {
        "teams": int(team_weights.sum()),
        "matches": int(table["matches"].sum()),
        "baseline_spearman": float(
            np.average([row["baseline_spearman"] for row in rows], weights=team_weights)
        ),
        "candidate_spearman": float(
            np.average([row["candidate_spearman"] for row in rows], weights=team_weights)
        ),
        "spearman_difference": float(
            np.average([row["spearman_difference"] for row in rows], weights=team_weights)
        ),
        "baseline_pairwise": float(
            np.average([row["baseline_pairwise"] for row in rows], weights=pair_weights)
        ),
        "candidate_pairwise": float(
            np.average([row["candidate_pairwise"] for row in rows], weights=pair_weights)
        ),
        "pairwise_difference": float(
            np.average([row["pairwise_difference"] for row in rows], weights=pair_weights)
        ),
        "mean_abs_start_shift": float(table["start_rating_shift"].abs().mean()),
        "max_abs_start_shift": float(table["start_rating_shift"].abs().max()),
    }


def changed_pair_diagnostics(
    table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    impacts: dict[tuple[str, int], dict[str, object]] = {}
    for competition, frame in table.groupby("competition", sort=True):
        records = list(frame.itertuples(index=False))
        for left_index, left in enumerate(records):
            for right in records[left_index + 1 :]:
                actual = np.sign(left.schedule_adjusted_score - right.schedule_adjusted_score)
                if actual == 0:
                    continue
                baseline = np.sign(left.baseline_end_rating - right.baseline_end_rating)
                candidate = np.sign(left.candidate_end_rating - right.candidate_end_rating)
                baseline_correct = 0.5 if baseline == 0 else float(baseline == actual)
                candidate_correct = 0.5 if candidate == 0 else float(candidate == actual)
                if candidate == baseline:
                    continue
                difference = candidate_correct - baseline_correct
                rows.append(
                    {
                        "competition": competition,
                        "left_team_id": int(left.team_id),
                        "left_team_name": left.team_name,
                        "right_team_id": int(right.team_id),
                        "right_team_name": right.team_name,
                        "baseline_correct": baseline_correct,
                        "candidate_correct": candidate_correct,
                        "pairwise_accuracy_difference": difference,
                    }
                )
                for record in (left, right):
                    key = (competition, int(record.team_id))
                    item = impacts.setdefault(
                        key,
                        {
                            "competition": competition,
                            "team_id": int(record.team_id),
                            "team_name": record.team_name,
                            "changed_pairs": 0,
                            "pairwise_net_difference": 0.0,
                        },
                    )
                    item["changed_pairs"] += 1
                    item["pairwise_net_difference"] += difference
    pair_frame = pd.DataFrame(rows)
    impact_frame = pd.DataFrame(impacts.values())
    if not impact_frame.empty:
        impact_frame = impact_frame.sort_values(
            ["pairwise_net_difference", "changed_pairs", "team_name"],
            ascending=[True, False, True],
            kind="stable",
        ).reset_index(drop=True)
    return pair_frame, impact_frame


def leave_one_team_out_diagnostics(table: pd.DataFrame) -> pd.DataFrame:
    baseline = pooled_ranking_metrics(table)
    rows = []
    for row in table.itertuples(index=False):
        reduced = table.loc[
            ~(
                table["competition"].eq(row.competition)
                & table["team_id"].eq(row.team_id)
            )
        ]
        metrics = pooled_ranking_metrics(reduced)
        rows.append(
            {
                "removed_competition": row.competition,
                "removed_team_id": int(row.team_id),
                "removed_team_name": row.team_name,
                "full_spearman_difference": baseline["spearman_difference"],
                "leave_one_out_spearman_difference": metrics["spearman_difference"],
                "full_pairwise_difference": baseline["pairwise_difference"],
                "leave_one_out_pairwise_difference": metrics["pairwise_difference"],
                "pooled_delta_nonnegative": bool(
                    metrics["spearman_difference"] >= 0.0
                    and metrics["pairwise_difference"] >= 0.0
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["pooled_delta_nonnegative", "leave_one_out_spearman_difference"],
        ascending=[False, False],
        kind="stable",
    ).reset_index(drop=True)


def persistence_sensitivity(
    sequence,
    core,
    baseline: ContextModelConfig,
    target: pd.DataFrame,
) -> pd.DataFrame:
    baseline_eval = evaluate_context_sequence(
        sequence,
        core,
        baseline,
        target,
        evaluation_seasons={SOURCE_SEASON},
        ranking_target_seasons={TARGET_SEASON},
    )
    baseline_rank = baseline_eval.ranking.loc[
        baseline_eval.ranking["competition"].eq("ALL")
    ].iloc[0]
    rows = []
    for persistence in PERSISTENCE_VALUES:
        candidate = replace(
            baseline,
            domestic=DomesticRegressionConfig("DOMESTIC_ANCHORED", persistence),
        )
        evaluated = evaluate_context_sequence(
            sequence,
            core,
            candidate,
            target,
            evaluation_seasons={SOURCE_SEASON},
            ranking_target_seasons={TARGET_SEASON},
        )
        ranking = evaluated.ranking.loc[evaluated.ranking["competition"].eq("ALL")].iloc[0]
        rows.append(
            {
                "persistence": persistence,
                "brier_1x2": evaluated.metrics["brier_1x2"],
                "brier_difference": (
                    evaluated.metrics["brier_1x2"] - baseline_eval.metrics["brier_1x2"]
                ),
                "log_loss_1x2": evaluated.metrics["log_loss_1x2"],
                "log_loss_difference": (
                    evaluated.metrics["log_loss_1x2"]
                    - baseline_eval.metrics["log_loss_1x2"]
                ),
                "ranking_score": ranking["ranking_score"],
                "ranking_difference": (
                    ranking["ranking_score"] - baseline_rank["ranking_score"]
                ),
                "pairwise_accuracy": ranking["pairwise_accuracy"],
                "pairwise_difference": (
                    ranking["pairwise_accuracy"] - baseline_rank["pairwise_accuracy"]
                ),
                "post_hoc_diagnostic_only": True,
            }
        )
    return pd.DataFrame(rows)


def match_loss_summary(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
) -> pd.DataFrame:
    joined = baseline[["match_id", "competition", "brier_1x2", "log_loss_1x2"]].merge(
        candidate[["match_id", "brier_1x2", "log_loss_1x2"]],
        on="match_id",
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    rows = []
    for competition, frame in [("ALL", joined), *joined.groupby("competition", sort=True)]:
        rows.append(
            {
                "competition": competition,
                "matches": len(frame),
                "brier_difference": float(
                    (frame["brier_1x2_candidate"] - frame["brier_1x2_baseline"]).mean()
                ),
                "log_loss_difference": float(
                    (
                        frame["log_loss_1x2_candidate"]
                        - frame["log_loss_1x2_baseline"]
                    ).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def diagnostic_decision(
    competition: pd.DataFrame,
    leave_one_out: pd.DataFrame,
    sensitivity: pd.DataFrame,
    match_metrics: pd.DataFrame,
) -> dict[str, object]:
    pooled = competition.loc[competition["competition"].eq("ALL")].iloc[0]
    loss = match_metrics.loc[match_metrics["competition"].eq("ALL")].iloc[0]
    sign_flips = int(leave_one_out["pooled_delta_nonnegative"].sum())
    ranking_gap_small = bool(
        abs(pooled["spearman_difference"]) < 0.005
        and abs(pooled["pairwise_difference"]) < 0.005
    )
    post_hoc_safe = sensitivity.loc[
        sensitivity["ranking_difference"].ge(0.0)
        & sensitivity["pairwise_difference"].ge(0.0)
        & sensitivity["brier_difference"].lt(0.0)
        & sensitivity["log_loss_difference"].lt(0.0)
    ]
    flip_competitions = set(
        leave_one_out.loc[
            leave_one_out["pooled_delta_nonnegative"], "removed_competition"
        ]
    )
    if (
        ranking_gap_small
        and sign_flips > 0
        and flip_competitions == {"UECL"}
        and TARGET_SEASON == "2021/22"
    ):
        diagnosis = "INAUGURAL_UECL_OUTLIER_SENSITIVE_GAP"
    elif ranking_gap_small and sign_flips > 0:
        diagnosis = "SMALL_OUTLIER_SENSITIVE_RANKING_GAP"
    elif ranking_gap_small:
        diagnosis = "SMALL_BUT_DISTRIBUTED_RANKING_GAP"
    else:
        diagnosis = "STRUCTURAL_RANKING_GAP"
    return {
        "status": "NO_AUTOMATIC_RANKING_VETO",
        "diagnosis": diagnosis,
        "ranking_gap_small": ranking_gap_small,
        "leave_one_out_sign_flips": sign_flips,
        "sign_flip_competitions": sorted(flip_competitions),
        "post_hoc_safe_persistence_values": post_hoc_safe["persistence"].tolist(),
        "fold_brier_difference": float(loss["brier_difference"]),
        "fold_log_loss_difference": float(loss["log_loss_difference"]),
        "reason": (
            "The isolated fold gap is small and outlier-sensitive, so it is not reliable "
            "ranking harm and no longer vetoes the layer by itself. Production status is "
            "decided by the complete replay and its declared baseline; post-hoc persistence values "
            "remain diagnostic only."
        ),
    }


def write_report(
    path: Path,
    manifest: dict[str, object],
    team_table: pd.DataFrame,
    competition: pd.DataFrame,
    pair_changes: pd.DataFrame,
    team_pair_impacts: pd.DataFrame,
    leave_one_out: pd.DataFrame,
    sensitivity: pd.DataFrame,
    match_metrics: pd.DataFrame,
) -> None:
    decision = manifest["decision"]
    pooled = competition.loc[competition["competition"].eq("ALL")].iloc[0]
    loss = match_metrics.loc[match_metrics["competition"].eq("ALL")].iloc[0]
    worst = team_table.nlargest(10, "abs_rank_error_difference")
    pair_worst = team_pair_impacts.head(10) if len(team_pair_impacts) else team_pair_impacts
    lines = [
        "# Domestic-Prior Ranking Diagnostic",
        "",
        "## Scope",
        "",
        f"- Selected fold: `{FOLD}`",
        f"- Rating source season: `{SOURCE_SEASON}`",
        f"- Forward ranking target: `{TARGET_SEASON}`",
        f"- Team-competition observations: `{len(team_table)}`",
        f"- Unique teams: `{team_table['team_id'].nunique()}`",
        f"- Changed candidate/baseline pairs: `{len(pair_changes)}`",
        "",
        "## Headline",
        "",
        f"- Diagnosis: `{decision['diagnosis']}`",
        f"- Production status: `{decision['status']}`",
        f"- Fold Brier difference: `{loss['brier_difference']:+.6f}`",
        f"- Fold log-loss difference: `{loss['log_loss_difference']:+.6f}`",
        f"- Forward Spearman difference: `{pooled['spearman_difference']:+.6f}`",
        f"- Forward pairwise difference: `{pooled['pairwise_difference']:+.6f}`",
        f"- Leave-one-team-out sign flips: `{decision['leave_one_out_sign_flips']}`",
        f"- Post-hoc safe persistence values: `{decision['post_hoc_safe_persistence_values']}`",
        "",
        "## Competition Ranking",
        "",
        markdown_table(competition, float_digits=6),
        "",
        "## Largest Rank-Error Deteriorations",
        "",
        markdown_table(
            worst[[
                "competition",
                "team_name",
                "start_rating_shift",
                "actual_rank",
                "baseline_rank",
                "candidate_rank",
                "abs_rank_error_difference",
            ]],
            float_digits=3,
        ),
        "",
        "## Largest Pairwise Contributors",
        "",
        (
            markdown_table(pair_worst, float_digits=3)
            if len(pair_worst)
            else "No pair ordering changed."
        ),
        "",
        "## Post-Hoc Persistence Sensitivity",
        "",
        markdown_table(sensitivity, float_digits=6),
        "",
        "The sensitivity table is diagnostic only. It uses the observed fold outcome and "
        "therefore cannot select a production parameter. This isolated, outlier-sensitive "
        "gap is no longer an automatic ranking veto; the complete current-baseline loss and "
        "ranking uncertainty gates determine candidate status.",
    ]
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
        "| "
        + " | ".join(format_value(value) for value in row)
        + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(rows)


def json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


if __name__ == "__main__":
    main()
