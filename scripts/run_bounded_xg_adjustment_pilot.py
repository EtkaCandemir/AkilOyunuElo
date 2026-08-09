from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ao_elo.xg_live import (  # noqa: E402
    XGBlendConfig,
    XGPerformanceBonusConfig,
    update_match_elo_with_xg,
)
from scripts.run_xg_performance_bonus_pilot import (  # noqa: E402
    DEFAULT_INPUT,
    ELO_SCALE,
    GOAL_ALPHA,
    GOAL_CAP,
    GOAL_TAU,
    HOME_ADVANTAGE,
    K_FACTOR,
    markdown_table,
    read_scenarios,
)


DEFAULT_OUTPUT = ROOT / "output" / "bounded_xg_adjustment_pilot"
MAX_XG_RATIO = 0.30
XG_SCALE = 1.25
MINIMUM_WINNER_RATIO = 1.0 - MAX_XG_RATIO


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show the controlled impact of the selected bounded xG model"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--goal-alpha", type=float, default=GOAL_ALPHA)
    args = parser.parse_args()

    results = run_bounded_pilot(
        read_scenarios(args.input.resolve()), goal_alpha=args.goal_alpha
    )
    validate_bounded_pilot(results)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_root / "pilot_results.csv", index=False, float_format="%.9f")
    (output_root / "pilot_report.md").write_text(
        build_report(results, goal_alpha=args.goal_alpha), encoding="utf-8"
    )
    decisive = results.loc[results["is_decisive"]]
    print(f"Scenarios: {len(results)}")
    print(
        "Candidate winner gain range: "
        f"{decisive['candidate_winner_gain'].min():.3f} to "
        f"{decisive['candidate_winner_gain'].max():.3f}"
    )
    print(f"Report: {output_root / 'pilot_report.md'}")


def run_bounded_pilot(
    scenarios: pd.DataFrame,
    *,
    goal_alpha: float = GOAL_ALPHA,
) -> pd.DataFrame:
    if not math.isfinite(goal_alpha) or goal_alpha < 0.0:
        raise ValueError("goal_alpha must be finite and non-negative")
    rows: list[dict[str, object]] = []
    config = XGPerformanceBonusConfig(
        MAX_XG_RATIO,
        XG_SCALE,
        MINIMUM_WINNER_RATIO,
    )
    for scenario in scenarios.itertuples(index=False):
        common = {
            "home_rating": float(scenario.home_rating),
            "away_rating": float(scenario.away_rating),
            "home_goals": int(scenario.home_goals),
            "away_goals": int(scenario.away_goals),
            "k_factor": K_FACTOR,
            "elo_scale": ELO_SCALE,
            "home_advantage": HOME_ADVANTAGE,
            "is_neutral": bool(scenario.is_neutral),
            "decided_on_penalties": bool(scenario.decided_on_penalties),
            "goal_tau": GOAL_TAU,
            "goal_difference_cap": GOAL_CAP,
            "xg_config": XGBlendConfig(0.0, 1.0),
        }
        xg_home = None if pd.isna(scenario.xg_home) else float(scenario.xg_home)
        xg_away = None if pd.isna(scenario.xg_away) else float(scenario.xg_away)
        classic = update_match_elo_with_xg(
            **common,
            goal_difference_enabled=False,
            goal_alpha=0.0,
        )
        production = update_match_elo_with_xg(
            **common,
            goal_difference_enabled=True,
            goal_alpha=goal_alpha,
        )
        candidate = update_match_elo_with_xg(
            **common,
            goal_difference_enabled=True,
            goal_alpha=goal_alpha,
            xg_home=xg_home,
            xg_away=xg_away,
            xg_performance_bonus_config=config,
        )
        winner_sign = 0.0
        winner = "DRAW"
        if candidate.actual_home_score == 1.0:
            winner_sign = 1.0
            winner = "HOME"
        elif candidate.actual_home_score == 0.0:
            winner_sign = -1.0
            winner = "AWAY"
        is_decisive = winner_sign != 0.0
        classic_gain = winner_sign * classic.power_delta
        production_gain = winner_sign * production.power_delta
        candidate_gain = winner_sign * candidate.power_delta
        xg_adjustment = candidate_gain - production_gain
        minimum_gain = (
            MINIMUM_WINNER_RATIO * classic_gain
            if is_decisive and xg_home is not None
            else None
        )
        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "goal_alpha": goal_alpha,
                "scenario_group": scenario.scenario_group,
                "description": scenario.description,
                "home_rating": float(scenario.home_rating),
                "away_rating": float(scenario.away_rating),
                "home_goals": int(scenario.home_goals),
                "away_goals": int(scenario.away_goals),
                "xg_home": xg_home,
                "xg_away": xg_away,
                "winner": winner,
                "is_decisive": is_decisive,
                "expected_home_score": candidate.expected_home_score,
                "classic_winner_gain": classic_gain,
                "goal_multiplier": production.goal_difference_multiplier,
                "production_gd_winner_gain": production_gain,
                "goal_bonus_on_winner": production_gain - classic_gain,
                "xg_signal_home": candidate.xg_performance_signal,
                "xg_adjustment_on_winner": xg_adjustment,
                "xg_adjustment_pct_of_classic": (
                    100.0 * xg_adjustment / classic_gain
                    if is_decisive and classic_gain > 0.0
                    else None
                ),
                "minimum_winner_gain": minimum_gain,
                "minimum_bound_active": bool(
                    minimum_gain is not None
                    and math.isclose(candidate_gain, minimum_gain, abs_tol=1e-9)
                ),
                "candidate_winner_gain": candidate_gain,
                "candidate_gain_ratio_vs_classic": (
                    candidate_gain / classic_gain
                    if is_decisive and classic_gain > 0.0
                    else None
                ),
                "candidate_home_delta": candidate.power_delta,
                "zero_sum_error": candidate.zero_sum_error,
            }
        )
    return pd.DataFrame(rows)


def validate_bounded_pilot(results: pd.DataFrame) -> None:
    if len(results) != 21:
        raise ValueError(f"Expected 21 scenarios, found {len(results)}")
    decisive = results.loc[results["is_decisive"]]
    if not decisive["candidate_winner_gain"].gt(0.0).all():
        raise ValueError("Every winner must gain Elo")
    covered = decisive.loc[decisive["xg_signal_home"].notna()]
    if not covered["candidate_gain_ratio_vs_classic"].ge(0.70 - 1e-12).all():
        raise ValueError("Bounded xG fell below the 70% winner guardrail")
    if results["zero_sum_error"].max() > 1e-9:
        raise ValueError("Pilot violated zero-sum conservation")
    fallback = results.loc[results["scenario_id"].eq("MISSING_XG_3_0")].iloc[0]
    if not math.isclose(
        fallback["candidate_winner_gain"],
        fallback["production_gd_winner_gain"],
        abs_tol=1e-9,
    ):
        raise ValueError("Missing xG must use production GD behavior")


def build_report(
    results: pd.DataFrame,
    *,
    goal_alpha: float = GOAL_ALPHA,
) -> str:
    selected = [
        "BAL_1_0_EQUAL_XG",
        "BAL_1_0_SUPPORTED",
        "BAL_1_0_LUCKY",
        "BAL_1_0_EXTREME_LUCKY",
        "BAL_2_0_SUPPORTED",
        "BAL_2_0_LUCKY",
        "BAL_3_2_SUPPORTED",
        "BAL_3_2_LUCKY",
        "FAV_1_0_LUCKY",
        "DOG_1_0_SUPPORTED",
        "DOG_1_0_LUCKY",
        "DRAW_2_2_XG_HOME",
        "PENALTY_DRAW",
        "MISSING_XG_3_0",
    ]
    view = results.set_index("scenario_id").loc[selected].reset_index()
    view = view[
        [
            "scenario_id",
            "home_goals",
            "away_goals",
            "xg_home",
            "xg_away",
            "classic_winner_gain",
            "production_gd_winner_gain",
            "xg_adjustment_on_winner",
            "candidate_winner_gain",
            "candidate_gain_ratio_vs_classic",
        ]
    ]
    return f"""# Kontrollu xG Katsayisi Pilot Raporu

```text
goal_alpha = {goal_alpha:.3f}
max_xg_ratio = {MAX_XG_RATIO:.2f}
xG_scale = {XG_SCALE:.2f}
minimum_winner_ratio = {MINIMUM_WINNER_RATIO:.2f}
```

{markdown_table(view)}

Kazananin klasik sonuc Elo'su xG nedeniyle `%70`in altina inemez. Gol farki
bonusu ayri kalir. Beraberlik, penalti karari ve eksik xG bonus uretmez.
"""


if __name__ == "__main__":
    main()
