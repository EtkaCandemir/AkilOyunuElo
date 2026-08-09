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

from ao_elo.xg_live import (  # noqa: E402
    XGBlendConfig,
    XGPerformanceBonusConfig,
    update_match_elo_with_xg,
)


DEFAULT_INPUT = ROOT / "data" / "xg_performance_bonus_pilot" / "scenarios.csv"
DEFAULT_OUTPUT = ROOT / "output" / "xg_performance_bonus_pilot"

K_FACTOR = 103.98098633392752
ELO_SCALE = 835.5614973262034
HOME_ADVANTAGE = 148.54426619132505
GOAL_ALPHA = 0.10
GOAL_TAU = 300.0
GOAL_CAP = 4
XG_BETA = 1.50
XG_SCALE = 3.00
WINNER_FLOOR = 0.05


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run controlled examples for the two-sided xG performance bonus"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    scenarios = read_scenarios(args.input.resolve())
    results = run_pilot(scenarios)
    validate_results(results)

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_root / "pilot_results.csv", index=False, float_format="%.9f")
    (output_root / "pilot_report.md").write_text(
        build_report(results), encoding="utf-8"
    )

    decisive = results.loc[results["is_decisive"]]
    print(f"Scenarios: {len(results)}")
    print(f"Decisive scenarios: {len(decisive)}")
    print(
        "Candidate winner gain range: "
        f"{decisive['candidate_winner_gain'].min():.3f} to "
        f"{decisive['candidate_winner_gain'].max():.3f}"
    )
    print(
        "xG effect versus production range: "
        f"{decisive['xg_effect_on_winner_vs_gd'].min():.3f} to "
        f"{decisive['xg_effect_on_winner_vs_gd'].max():.3f}"
    )
    print(f"Report: {output_root / 'pilot_report.md'}")


def read_scenarios(path: Path) -> pd.DataFrame:
    values = pd.read_csv(path)
    required = {
        "scenario_id",
        "scenario_group",
        "description",
        "home_rating",
        "away_rating",
        "home_goals",
        "away_goals",
        "xg_home",
        "xg_away",
        "is_neutral",
        "decided_on_penalties",
    }
    missing = sorted(required - set(values.columns))
    if missing:
        raise ValueError(f"Missing pilot columns: {missing}")
    if values["scenario_id"].duplicated().any():
        raise ValueError("scenario_id must be unique")
    for column in ("is_neutral", "decided_on_penalties"):
        values[column] = values[column].map(parse_bool)
    if values[["xg_home", "xg_away"]].isna().any(axis=1).ne(
        values[["xg_home", "xg_away"]].isna().all(axis=1)
    ).any():
        raise ValueError("xG values must be both present or both missing")
    return values


def parse_bool(value: object) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def run_pilot(scenarios: pd.DataFrame) -> pd.DataFrame:
    rows = []
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
            goal_alpha=GOAL_ALPHA,
        )
        candidate = update_match_elo_with_xg(
            **common,
            goal_difference_enabled=True,
            goal_alpha=GOAL_ALPHA,
            xg_home=xg_home,
            xg_away=xg_away,
            xg_performance_bonus_config=XGPerformanceBonusConfig(
                XG_BETA,
                XG_SCALE,
                WINNER_FLOOR,
            ),
        )
        is_decisive = candidate.actual_home_score != 0.5
        winner_sign = 0.0
        winner = "DRAW"
        if candidate.actual_home_score == 1.0:
            winner_sign = 1.0
            winner = "HOME"
        elif candidate.actual_home_score == 0.0:
            winner_sign = -1.0
            winner = "AWAY"
        classic_winner_gain = winner_sign * classic.power_delta
        production_winner_gain = winner_sign * production.power_delta
        candidate_winner_gain = winner_sign * candidate.power_delta
        xg_effect = candidate_winner_gain - production_winner_gain
        floor_gain = (
            WINNER_FLOOR * classic_winner_gain if is_decisive and xg_home is not None else None
        )
        clamp_active = bool(
            is_decisive
            and floor_gain is not None
            and math.isclose(candidate_winner_gain, floor_gain, abs_tol=1e-9)
        )
        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "scenario_group": scenario.scenario_group,
                "description": scenario.description,
                "home_rating": float(scenario.home_rating),
                "away_rating": float(scenario.away_rating),
                "home_goals": int(scenario.home_goals),
                "away_goals": int(scenario.away_goals),
                "xg_home": xg_home,
                "xg_away": xg_away,
                "is_neutral": bool(scenario.is_neutral),
                "decided_on_penalties": bool(scenario.decided_on_penalties),
                "winner": winner,
                "is_decisive": is_decisive,
                "expected_home_score": candidate.expected_home_score,
                "classic_home_delta": classic.power_delta,
                "classic_winner_gain": classic_winner_gain,
                "goal_difference_multiplier": production.goal_difference_multiplier,
                "production_gd_home_delta": production.power_delta,
                "production_gd_winner_gain": production_winner_gain,
                "goal_difference_bonus_on_winner": (
                    production_winner_gain - classic_winner_gain
                ),
                "xg_signal_home": candidate.xg_performance_signal,
                "xg_bonus_home": candidate.xg_performance_adjustment * K_FACTOR,
                "xg_bonus_winner_raw": (
                    winner_sign * candidate.xg_performance_adjustment * K_FACTOR
                ),
                "minimum_winner_gain_floor": floor_gain,
                "winner_floor_activated": clamp_active,
                "candidate_home_delta": candidate.power_delta,
                "candidate_winner_gain": candidate_winner_gain,
                "xg_effect_on_winner_vs_gd": xg_effect,
                "xg_effect_pct_of_classic_gain": (
                    100.0 * xg_effect / classic_winner_gain
                    if is_decisive and classic_winner_gain > 0.0
                    else None
                ),
                "total_effect_on_winner_vs_classic": (
                    candidate_winner_gain - classic_winner_gain
                ),
                "candidate_gain_ratio_vs_classic": (
                    candidate_winner_gain / classic_winner_gain
                    if is_decisive and classic_winner_gain > 0.0
                    else None
                ),
                "zero_sum_error": candidate.zero_sum_error,
            }
        )
    return pd.DataFrame(rows)


def validate_results(results: pd.DataFrame) -> None:
    if len(results) != 21:
        raise ValueError(f"Expected 21 pilot scenarios, found {len(results)}")
    decisive = results.loc[results["is_decisive"]]
    if not decisive["candidate_winner_gain"].gt(0.0).all():
        raise ValueError("Every decisive match must give the winner positive Elo")
    if results["zero_sum_error"].max() > 1e-9:
        raise ValueError("Pilot violated zero-sum Elo conservation")
    missing = results.loc[results["scenario_id"].eq("MISSING_XG_3_0")].iloc[0]
    if not math.isclose(
        missing["candidate_home_delta"],
        missing["production_gd_home_delta"],
        abs_tol=1e-9,
    ):
        raise ValueError("Missing xG must fall back to production GD update")
    neutral_xg = results.loc[
        results["scenario_id"].isin({"DRAW_2_2_XG_HOME", "PENALTY_DRAW"})
    ]
    if not neutral_xg["xg_effect_on_winner_vs_gd"].eq(0.0).all():
        raise ValueError("Draws and penalty decisions cannot produce an xG bonus")


def build_report(results: pd.DataFrame) -> str:
    selected_ids = [
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
        "AWAY_0_2_SUPPORTED",
        "DRAW_2_2_XG_HOME",
        "PENALTY_DRAW",
        "MISSING_XG_3_0",
    ]
    view = results.set_index("scenario_id").loc[selected_ids].reset_index()
    view = view[
        [
            "scenario_id",
            "home_goals",
            "away_goals",
            "xg_home",
            "xg_away",
            "expected_home_score",
            "classic_winner_gain",
            "production_gd_winner_gain",
            "xg_effect_on_winner_vs_gd",
            "xg_effect_pct_of_classic_gain",
            "candidate_winner_gain",
            "winner_floor_activated",
        ]
    ]
    table = markdown_table(view)
    return f"""# xG Performans Bonusu Kontrollu Pilot

## Aktif Katsayilar

```text
K = {K_FACTOR:.6f}
Scale = {ELO_SCALE:.6f}
Home advantage = {HOME_ADVANTAGE:.6f}
Goal alpha = {GOAL_ALPHA:.2f}
Goal tau = {GOAL_TAU:.0f}
Goal cap = {GOAL_CAP}
xG beta = {XG_BETA:.2f}
xG scale = {XG_SCALE:.2f}
Minimum winner gain = %{WINNER_FLOOR * 100:.0f} of classic result Elo
```

```text
Delta_classic = K * (S-E)
Delta_GD = Delta_classic * M_GD
Q_xG = tanh((xG_home-xG_away)/3.00)
Delta_xG = 1.50 * abs(Delta_classic) * Q_xG
Delta_candidate = positive-winner-clamp(Delta_GD + Delta_xG)
```

## Secili Senaryolar

{table}

## Okuma Rehberi

- `classic_winner_gain`: yalniz mac sonucundan gelen Elo.
- `production_gd_winner_gain`: gol farki carpanindan sonraki mevcut production Elo.
- `xg_effect_on_winner_vs_gd`: xG katmaninin production uzerine ekledigi veya cikardigi puan.
- `candidate_winner_gain`: onerilen katsayilarla kazananin nihai Elo kazanci.
- Ters xG cok gucluyse `%5` taban devreye girer; kazanan eksi Elo almaz.
- Beraberlik, penalti karari ve eksik xG xG bonusu uretmez.

Bu dosya bir tahmin backtesti degil, katsayilarin mekanik etkisini izole eden
kontrollu pilot testidir. Production aktivasyonu hakkindaki karar walk-forward
backtest ve prospective shadow kanitina baglidir.
"""


def markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows = [headers, ["---"] * len(headers)]
    for values in frame.itertuples(index=False, name=None):
        rows.append([format_markdown_cell(value) for value in values])
    return "\n".join(
        "| " + " | ".join(str(value) for value in row) + " |" for row in rows
    )


def format_markdown_cell(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value).replace("|", "\\|")


if __name__ == "__main__":
    main()
