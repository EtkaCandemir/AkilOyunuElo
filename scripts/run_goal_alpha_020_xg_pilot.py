from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_bounded_xg_adjustment_pilot import (  # noqa: E402
    run_bounded_pilot,
    validate_bounded_pilot,
)
from scripts.run_xg_performance_bonus_pilot import (  # noqa: E402
    DEFAULT_INPUT,
    GOAL_CAP,
    GOAL_TAU,
    markdown_table,
    read_scenarios,
)


DEFAULT_OUTPUT = ROOT / "output" / "goal_alpha_020_xg_pilot"
BASELINE_ALPHA = 0.10
CANDIDATE_ALPHA = 0.20
XG_RATIO = 0.30
XG_SCALE = 1.25


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare goal_alpha=0.10 and 0.20 with bounded xG"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    scenarios = read_scenarios(args.input.resolve())
    results = run_comparison_pilot(scenarios)
    validate_comparison_pilot(results)

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results.to_csv(
        output_root / "pilot_results.csv", index=False, float_format="%.9f"
    )
    build_group_summary(results).to_csv(
        output_root / "scenario_group_summary.csv",
        index=False,
        float_format="%.9f",
    )
    (output_root / "pilot_report.md").write_text(
        build_report(results), encoding="utf-8"
    )

    decisive = results.loc[results["is_decisive"]]
    print(f"Scenarios: {len(results)}")
    print(
        "Alpha 0.20 + xG winner gain range: "
        f"{decisive['alpha_020_xg_winner_gain'].min():.3f} to "
        f"{decisive['alpha_020_xg_winner_gain'].max():.3f}"
    )
    print(
        "Alpha 0.20 uplift versus alpha 0.10 + xG: "
        f"{decisive['combined_gain_difference'].min():.3f} to "
        f"{decisive['combined_gain_difference'].max():.3f}"
    )
    print(f"Report: {output_root / 'pilot_report.md'}")


def run_comparison_pilot(scenarios: pd.DataFrame) -> pd.DataFrame:
    baseline = run_bounded_pilot(scenarios, goal_alpha=BASELINE_ALPHA)
    candidate = run_bounded_pilot(scenarios, goal_alpha=CANDIDATE_ALPHA)
    validate_bounded_pilot(baseline)
    validate_bounded_pilot(candidate)

    identity_columns = [
        "scenario_id",
        "scenario_group",
        "description",
        "home_rating",
        "away_rating",
        "home_goals",
        "away_goals",
        "xg_home",
        "xg_away",
        "winner",
        "is_decisive",
        "expected_home_score",
        "classic_winner_gain",
    ]
    results = baseline[identity_columns].copy()
    for prefix, frame in (("alpha_010", baseline), ("alpha_020", candidate)):
        results[f"{prefix}_goal_multiplier"] = frame["goal_multiplier"]
        results[f"{prefix}_gd_winner_gain"] = frame["production_gd_winner_gain"]
        results[f"{prefix}_goal_bonus"] = frame["goal_bonus_on_winner"]
        results[f"{prefix}_xg_adjustment"] = frame["xg_adjustment_on_winner"]
        results[f"{prefix}_xg_winner_gain"] = frame["candidate_winner_gain"]
        results[f"{prefix}_gain_ratio_vs_classic"] = frame[
            "candidate_gain_ratio_vs_classic"
        ]
        results[f"{prefix}_minimum_bound_active"] = frame[
            "minimum_bound_active"
        ]
        results[f"{prefix}_home_delta"] = frame["candidate_home_delta"]
        results[f"{prefix}_zero_sum_error"] = frame["zero_sum_error"]

    results["combined_gain_difference"] = (
        results["alpha_020_xg_winner_gain"]
        - results["alpha_010_xg_winner_gain"]
    )
    results["goal_bonus_difference"] = (
        results["alpha_020_goal_bonus"] - results["alpha_010_goal_bonus"]
    )
    return results


def validate_comparison_pilot(results: pd.DataFrame) -> None:
    if len(results) != 21:
        raise ValueError(f"Expected 21 scenarios, found {len(results)}")
    decisive = results.loc[results["is_decisive"]]
    if not decisive["alpha_020_xg_winner_gain"].gt(0.0).all():
        raise ValueError("Every alpha=0.20 winner must gain Elo")
    for prefix in ("alpha_010", "alpha_020"):
        if results[f"{prefix}_zero_sum_error"].max() > 1e-9:
            raise ValueError(f"{prefix} violated zero-sum conservation")
    multi_goal = decisive.loc[
        (decisive["home_goals"] - decisive["away_goals"]).abs().gt(1)
    ]
    if not multi_goal["alpha_020_goal_multiplier"].ge(
        multi_goal["alpha_010_goal_multiplier"] - 1e-12
    ).all():
        raise ValueError("Higher alpha reduced a multi-goal multiplier")
    missing = results.loc[results["scenario_id"].eq("MISSING_XG_3_0")].iloc[0]
    if abs(
        missing["alpha_020_xg_winner_gain"]
        - missing["alpha_020_gd_winner_gain"]
    ) > 1e-9:
        raise ValueError("Missing xG must fall back to alpha=0.20 GD")


def build_group_summary(results: pd.DataFrame) -> pd.DataFrame:
    decisive = results.loc[results["is_decisive"]].copy()
    return (
        decisive.groupby("scenario_group", sort=True)
        .agg(
            scenarios=("scenario_id", "count"),
            mean_classic_gain=("classic_winner_gain", "mean"),
            mean_alpha_010_xg_gain=("alpha_010_xg_winner_gain", "mean"),
            mean_alpha_020_xg_gain=("alpha_020_xg_winner_gain", "mean"),
            mean_alpha_uplift=("combined_gain_difference", "mean"),
            max_alpha_uplift=("combined_gain_difference", "max"),
        )
        .reset_index()
    )


def build_report(results: pd.DataFrame) -> str:
    selected = [
        "BAL_1_0_EQUAL_XG",
        "BAL_1_0_SUPPORTED",
        "BAL_1_0_LUCKY",
        "BAL_1_0_EXTREME_LUCKY",
        "BAL_2_0_EQUAL_XG",
        "BAL_2_0_SUPPORTED",
        "BAL_2_0_LUCKY",
        "BAL_4_0_SUPPORTED",
        "BAL_4_0_LUCKY",
        "BAL_3_2_SUPPORTED",
        "BAL_3_2_LUCKY",
        "FAV_1_0_LUCKY",
        "DOG_1_0_SUPPORTED",
        "DOG_1_0_LUCKY",
        "DRAW_2_2_XG_HOME",
        "PENALTY_DRAW",
        "MISSING_XG_3_0",
    ]
    columns = [
        "scenario_id",
        "home_goals",
        "away_goals",
        "xg_home",
        "xg_away",
        "classic_winner_gain",
        "alpha_010_xg_winner_gain",
        "alpha_020_gd_winner_gain",
        "alpha_020_xg_adjustment",
        "alpha_020_xg_winner_gain",
        "combined_gain_difference",
    ]
    view = results.set_index("scenario_id").loc[selected].reset_index()[columns]
    groups = build_group_summary(results)
    decisive = results.loc[results["is_decisive"]]
    return f"""# Goal Alpha 0.20 ve Kontrollu xG Pilot Raporu

```text
Karsilastirma alpha = {BASELINE_ALPHA:.2f} vs {CANDIDATE_ALPHA:.2f}
xG ratio = {XG_RATIO:.2f}
xG scale = {XG_SCALE:.2f}
goal_tau = {GOAL_TAU:.0f}
goal_cap = {GOAL_CAP}
mac hareket tavani = KAPALI
```

## Kontrollu Senaryolar

{markdown_table(view)}

`alpha_020_gd_winner_gain`, xG olmadan yalniz gol farkli Elo kazancidir.
`alpha_020_xg_adjustment`, xG'nin bu kazanca ekledigi veya cikardigi puandir.
`alpha_020_xg_winner_gain` ikisinin toplamidir. Tek farkli galibiyette gol
bonusu sifirdir; xG yine temel kazanci kontrollu bicimde guclendirir veya azaltir.

## Senaryo Gruplari

{markdown_table(groups)}

## Guvenlik Kontrolleri

```text
minimum winner gain = {decisive['alpha_020_xg_winner_gain'].min():.6f}
maximum winner gain = {decisive['alpha_020_xg_winner_gain'].max():.6f}
maximum zero-sum error = {max(results['alpha_020_zero_sum_error'].max(), results['alpha_010_zero_sum_error'].max()):.12f}
alpha 0.20 uplift range = {decisive['combined_gain_difference'].min():.6f} / {decisive['combined_gain_difference'].max():.6f}
```

Bu pilot davranis testidir; tarihsel performans veya production terfi kaniti
degildir.
"""


if __name__ == "__main__":
    main()
