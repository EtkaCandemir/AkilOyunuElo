from __future__ import annotations

"""Re-test the shipped bounded xG rating layer across six seasons.

The active contract enables `xg_performance` on evidence from a single season:
`961` matches with `606` eligible, activated by explicit product decision. Only
2025/26 xG existed when that call was made, so every earlier season replayed
with the goal-margin-only fallback and could not speak to the layer at all.

FotMob xG reaches back to 2020/21, which raises the eligible sample from `606`
to `2827` across `4884` matches. This script replays the production kernel with
and without the xG adjustment over that wider window and reports the difference
on the axes the repository gates on, including the split that matters most
here: main-stage matches carry near-total coverage while qualifiers are almost
empty, so a pooled number alone would hide where the layer actually acts.

The script changes no production parameter. It writes evidence only.
"""

import argparse
import json
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
from ao_elo.evaluation import (  # noqa: E402
    dependency_robust_loss_difference_ci,
    schedule_adjusted_team_performance,
)
from scripts.run_controlled_goal_progression_backtest import (  # noqa: E402
    prepare_controlled_data,
)
from scripts.run_current_model_evaluation import (  # noqa: E402
    CURRENT,
    NO_XG,
    ArmEvaluation,
    EvaluationArm,
    evaluate_arm,
)
from scripts.run_stage_weighted_progression_backtest import (  # noqa: E402
    DOMESTIC_ADJUSTMENTS,
    DYNAMIC_MANIFEST,
    EVENTS_PATH,
    PRODUCTION_CONTRACT,
    STATIC_DATA_ROOT,
    load_domestic_adjustments,
    load_xg_map,
    validate_production_contract,
)
from scripts.run_v2_achievement_reserve_calibration import load_reserve_data  # noqa: E402
from scripts.run_v2_evaluation_upgrade import read_events  # noqa: E402


DEFAULT_XG = ROOT / "data" / "xg_2020_2026" / "uefa_2020_2026_matches_with_xg.csv"
LEGACY_XG = ROOT / "data" / "xg_2025_26" / "uefa_2025_26_matches_with_xg.csv"
DEFAULT_OUTPUT = ROOT / "output" / "xg_multiseason_backtest_2020_2026"
BOOTSTRAP_SAMPLES = 4000
BOOTSTRAP_SEED = 20260819
QUALIFYING_ROUNDS = frozenset(
    {
        "Preliminary Round",
        "1st Qualifying Round",
        "2nd Qualifying Round",
        "3rd Qualifying Round",
        "Qualifying Play-off Round",
    }
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score the shipped bounded xG rating layer on the six seasons that "
            "have FotMob xG coverage"
        )
    )
    parser.add_argument("--xg-data", type=Path, default=DEFAULT_XG)
    parser.add_argument("--legacy-xg-data", type=Path, default=LEGACY_XG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
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
    target = schedule_adjusted_team_performance(events)
    seeds = load_domestic_adjustments(DOMESTIC_ADJUSTMENTS, datasets)

    wide = load_xg_map(arguments.xg_data, datasets)
    legacy = load_xg_map(arguments.legacy_xg_data, datasets)
    context = match_context(events)
    gates = validation_gates(wide, legacy, arguments.xg_data, context)
    if not gates["passed"].all():
        raise ValueError(
            f"xG backtest gates failed: {gates.loc[~gates['passed'], 'gate'].tolist()}"
        )

    evaluations: dict[str, ArmEvaluation] = {}
    for name, arm, xg_map in (
        ("XG_SIX_SEASON", EvaluationArm(CURRENT, True, True, True, True, True), wide),
        ("XG_2025_26_ONLY", EvaluationArm(CURRENT, True, True, True, True, True), legacy),
        ("NO_XG", EvaluationArm(NO_XG, True, True, False, True, True), {}),
    ):
        print(f"Replaying {name}", flush=True)
        evaluations[name] = evaluate_arm(
            datasets,
            arm,
            core=core,
            parameters=parameters,
            current_domestic=seeds,
            baseline_domestic=seeds,
            xg_map=xg_map,
            target=target,
        )

    summary = arm_summary(evaluations, context)
    segments = segment_summary(evaluations, context, wide)
    seasons = season_summary(evaluations, context)
    uncertainty = uncertainty_summary(
        evaluations, context, wide, samples=int(arguments.bootstrap_samples)
    )

    summary.to_csv(output_root / "arm_summary.csv", index=False)
    segments.to_csv(output_root / "segment_summary.csv", index=False)
    seasons.to_csv(output_root / "season_summary.csv", index=False)
    uncertainty.to_csv(output_root / "uncertainty.csv", index=False)
    gates.to_csv(output_root / "validation_gates.csv", index=False)

    manifest = build_manifest(summary, uncertainty, gates, wide, legacy, production)
    (output_root / "backtest_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(output_root / "backtest_report.md", summary, segments, seasons, uncertainty, gates, manifest)

    print(f"Wrote xG multiseason backtest to {output_root}")
    print(gates.to_string(index=False))
    print()
    print(summary.to_string(index=False))
    print()
    print(segments.to_string(index=False))


def match_context(events: pd.DataFrame) -> pd.DataFrame:
    frame = events[["match_id", "season", "competition", "round"]].copy()
    frame["match_id"] = frame["match_id"].astype(str)
    frame["phase"] = np.where(
        frame["round"].isin(QUALIFYING_ROUNDS), "QUALIFYING", "MAIN"
    )
    return frame


def scored(evaluation: ArmEvaluation, context: pd.DataFrame) -> pd.DataFrame:
    frame = evaluation.predictions.copy()
    frame["match_id"] = frame["match_id"].astype(str)
    return frame.merge(
        context[["match_id", "phase"]], on="match_id", how="left", validate="one_to_one"
    )


def arm_summary(
    evaluations: dict[str, ArmEvaluation], context: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for name, evaluation in evaluations.items():
        frame = scored(evaluation, context)
        rows.append(
            {
                "arm": name,
                "matches": int(len(frame)),
                "brier_1x2": float(frame["brier_1x2"].mean()),
                "log_loss_1x2": float(frame["log_loss_1x2"].mean()),
                "accuracy_1x2": float(
                    frame["predicted_class"].eq(frame["actual_class"]).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def paired(
    evaluations: dict[str, ArmEvaluation], context: pd.DataFrame, candidate: str
) -> pd.DataFrame:
    left = scored(evaluations[candidate], context)
    right = scored(evaluations["NO_XG"], context)
    return left.merge(
        right[["match_id", "brier_1x2", "log_loss_1x2"]],
        on="match_id",
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    )


def segment_summary(
    evaluations: dict[str, ArmEvaluation],
    context: pd.DataFrame,
    xg_map: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    rows = []
    for candidate in ("XG_SIX_SEASON", "XG_2025_26_ONLY"):
        frame = paired(evaluations, context, candidate)
        frame["has_xg"] = frame["match_id"].isin(xg_map)
        frame["brier_delta"] = frame["brier_1x2_candidate"] - frame["brier_1x2_baseline"]
        for label, block in (
            ("ALL", frame),
            ("PHASE:MAIN", frame.loc[frame["phase"].eq("MAIN")]),
            ("PHASE:QUALIFYING", frame.loc[frame["phase"].eq("QUALIFYING")]),
            ("XG_PRESENT", frame.loc[frame["has_xg"]]),
            ("XG_ABSENT", frame.loc[~frame["has_xg"]]),
        ):
            if block.empty:
                continue
            rows.append(
                {
                    "arm": candidate,
                    "segment": label,
                    "matches": int(len(block)),
                    "brier_delta_vs_no_xg": float(block["brier_delta"].mean()),
                }
            )
    return pd.DataFrame(rows)


def season_summary(
    evaluations: dict[str, ArmEvaluation], context: pd.DataFrame
) -> pd.DataFrame:
    frame = paired(evaluations, context, "XG_SIX_SEASON")
    frame["brier_delta"] = frame["brier_1x2_candidate"] - frame["brier_1x2_baseline"]
    frame["log_loss_delta"] = (
        frame["log_loss_1x2_candidate"] - frame["log_loss_1x2_baseline"]
    )
    return (
        frame.groupby("season")
        .agg(
            matches=("brier_delta", "size"),
            brier_delta=("brier_delta", "mean"),
            log_loss_delta=("log_loss_delta", "mean"),
        )
        .reset_index()
    )


def uncertainty_summary(
    evaluations: dict[str, ArmEvaluation],
    context: pd.DataFrame,
    xg_map: dict[str, tuple[float, float]],
    *,
    samples: int,
) -> pd.DataFrame:
    blocks = []
    for candidate in ("XG_SIX_SEASON", "XG_2025_26_ONLY"):
        frame = paired(evaluations, context, candidate)
        frame["has_xg"] = frame["match_id"].isin(xg_map)
        for label, block in (
            ("ALL", frame),
            ("PHASE:MAIN", frame.loc[frame["phase"].eq("MAIN")]),
            ("XG_PRESENT", frame.loc[frame["has_xg"]]),
        ):
            if block.empty:
                continue
            for metric in ("brier_1x2", "log_loss_1x2"):
                sample = block[
                    ["season", "match_id", "home_team_id", "away_team_id", "kickoff_utc", "tie_id"]
                ].copy()
                sample["loss_difference"] = (
                    block[f"{metric}_candidate"] - block[f"{metric}_baseline"]
                )
                result = dependency_robust_loss_difference_ci(
                    sample, bootstrap_samples=samples, seed=BOOTSTRAP_SEED
                )
                result.insert(0, "arm", candidate)
                result.insert(1, "segment", label)
                result.insert(2, "metric", metric)
                blocks.append(result)
    return pd.concat(blocks, ignore_index=True)


def validation_gates(
    wide: dict[str, tuple[float, float]],
    legacy: dict[str, tuple[float, float]],
    xg_path: Path,
    context: pd.DataFrame,
) -> pd.DataFrame:
    frame = pd.read_csv(xg_path)
    eligible = frame.loc[frame["xg_analysis_eligible"].astype(bool)]
    shared = set(legacy) & set(wide)
    agree = all(
        np.isclose(legacy[key][0], wide[key][0], atol=1e-9)
        and np.isclose(legacy[key][1], wide[key][1], atol=1e-9)
        for key in shared
    )
    return pd.DataFrame(
        [
            {
                "gate": "wide_superset_of_legacy",
                "passed": bool(set(legacy) <= set(wide)),
                "observed": len(wide),
                "requirement": "Six-season map must contain every 2025/26 eligible match",
            },
            {
                "gate": "shared_values_identical",
                "passed": bool(agree),
                "observed": len(shared),
                "requirement": "Re-fetched 2025/26 xG must match the frozen dataset",
            },
            {
                "gate": "eligible_sample_grew",
                "passed": bool(len(wide) > 4 * len(legacy)),
                "observed": len(wide),
                "requirement": "Evidence base must be materially wider than the shipped one",
            },
            {
                "gate": "coverage_concentrated_in_main_stage",
                "passed": bool(
                    eligible.merge(context, on="match_id", how="left")["phase"]
                    .eq("MAIN")
                    .mean()
                    > 0.80
                ),
                "observed": float(
                    eligible.merge(context, on="match_id", how="left")["phase"]
                    .eq("MAIN")
                    .mean()
                ),
                "requirement": "Documented coverage shape must hold in the loaded map",
            },
        ]
    )


def build_manifest(
    summary: pd.DataFrame,
    uncertainty: pd.DataFrame,
    gates: pd.DataFrame,
    wide: dict[str, tuple[float, float]],
    legacy: dict[str, tuple[float, float]],
    production: dict,
) -> dict[str, object]:
    indexed = summary.set_index("arm")
    envelope = uncertainty.loc[uncertainty["method"].eq("conservative_envelope")]
    return {
        "layer": "BOUNDED_XG_PERFORMANCE_MULTISEASON_REVALIDATION",
        "changes_production_parameters": False,
        "shipped_evidence": production["xg_performance_evidence"],
        "eligible_matches_shipped": len(legacy),
        "eligible_matches_now": len(wide),
        "all_gates_passed": bool(gates["passed"].all()),
        "brier": {
            str(arm): float(indexed.loc[arm, "brier_1x2"]) for arm in indexed.index
        },
        "conservative_envelope_reliable_improvement": bool(
            envelope["reliable_improvement"].any()
        ),
        "conservative_envelope_reliable_harm": bool(envelope["reliable_harm"].any()),
    }


def write_report(
    path: Path,
    summary: pd.DataFrame,
    segments: pd.DataFrame,
    seasons: pd.DataFrame,
    uncertainty: pd.DataFrame,
    gates: pd.DataFrame,
    manifest: dict[str, object],
) -> None:
    envelope = uncertainty.loc[uncertainty["method"].eq("conservative_envelope")]
    lines = [
        "# Bounded xG Katmani: Alti Sezonluk Yeniden Dogrulama",
        "",
        "Aktif contract `xg_performance` katmanini tek sezonluk kanitla acti:",
        f"`{manifest['shipped_evidence']['full_season_matches']}` mac, "
        f"`{manifest['shipped_evidence']['xg_eligible_matches']}` uygun, "
        "`manual_product_decision`. FotMob xG 2020/21'e kadar uzandigi icin uygun",
        f"orneklem `{manifest['eligible_matches_shipped']}` -> "
        f"`{manifest['eligible_matches_now']}` seviyesine cikti.",
        "",
        "Bu kosu production parametresi degistirmez.",
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
        "## Segment kirilimi (NO_XG'ye karsi Brier farki)",
        "",
        "Negatif deger xG katmaninin faydali oldugunu gosterir. Kapsam ana asamada",
        "yogunlastigi icin `PHASE:MAIN` ve `XG_PRESENT` satirlari karar icin",
        "havuzlanmis satirdan daha bilgilendiricidir.",
        "",
        "```text",
        segments.to_string(index=False),
        "```",
        "",
        "## Sezon bazinda",
        "",
        "```text",
        seasons.to_string(index=False),
        "```",
        "",
        "## Dependency-robust belirsizlik (conservative envelope)",
        "",
        "```text",
        envelope[
            ["arm", "segment", "metric", "mean_difference", "ci_95_lower", "ci_95_upper", "reliable_improvement", "reliable_harm"]
        ].to_string(index=False),
        "```",
        "",
        "## Karar girdisi",
        "",
        f"- Guvenilir iyilesme: `{manifest['conservative_envelope_reliable_improvement']}`.",
        f"- Guvenilir zarar: `{manifest['conservative_envelope_reliable_harm']}`.",
        "",
        "Karar urun tarafina aittir; bu belge yalniz kanit uretir.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
