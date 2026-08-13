from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ao_elo.dynamic_csv import (  # noqa: E402
    MATCH_INPUT_COLUMNS,
    load_selected_v2_config,
    run_batch,
)
from ao_elo.tournament_bonus import ELIGIBLE_PROGRESSION_STAGES  # noqa: E402


SEASON = "2025/26"
EXPECTED_MATCHES = 961
EXPECTED_XG_ELIGIBLE = 606
EXPECTED_XG_FALLBACK = EXPECTED_MATCHES - EXPECTED_XG_ELIGIBLE
STATIC_RATINGS = (
    ROOT
    / "output"
    / "domestic_surprise_production_verification_2025_26"
    / "integrated_ao_first_elo.csv"
)
EVENTS = (
    ROOT
    / "data"
    / "xg_2025_26"
    / "uefa_2025_26_matches_with_xg.csv"
)
MANIFEST = ROOT / "contracts" / "ao_european_elo_v2_production.json"
OUTPUT = ROOT / "output" / "progression_bonus_production_verification_2025_26"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    initial = build_initial_ratings()
    matches = build_matches()
    initial_path = OUTPUT / "initial_ratings.csv"
    matches_path = OUTPUT / "matches.csv"
    initial.to_csv(initial_path, index=False, lineterminator="\n")
    matches.to_csv(matches_path, index=False, lineterminator="\n")

    config = load_selected_v2_config(MANIFEST)
    state, updates = run_batch(
        initial_path,
        matches_path,
        OUTPUT / "production_replay",
        config,
    )
    update_frame = pd.read_csv(OUTPUT / "production_replay" / "match_updates.csv")
    bonus_events = update_frame.loc[update_frame["progression_bonus_added"].gt(0)].copy()
    bonus_events.to_csv(OUTPUT / "bonus_events.csv", index=False, lineterminator="\n")

    verification = verify(initial, state, update_frame, bonus_events, config)
    (OUTPUT / "verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("AO production progression verification complete")
    for key, value in verification.items():
        print(f"{key}: {value}")


def build_initial_ratings() -> pd.DataFrame:
    data = pd.read_csv(STATIC_RATINGS)
    data = data.loc[data["season"].astype(str).eq(SEASON)].copy()
    result = data[["season", "team_id", "team_name", "ao_first_elo"]].copy()
    if len(result) != 236 or result["team_id"].duplicated().any():
        raise ValueError("Expected 236 unique 2025/26 initial ratings")
    return result.sort_values("team_id", kind="mergesort").reset_index(drop=True)


def build_matches() -> pd.DataFrame:
    events = pd.read_csv(EVENTS)
    events = events.loc[events["season"].astype(str).eq(SEASON)].copy()
    events = events.sort_values(["kickoff_utc", "match_id"], kind="mergesort")
    tie_counts = events.loc[events["is_knockout"]].groupby("tie_id")["match_id"].transform("size")
    events["is_single_match_tie"] = False
    events.loc[tie_counts.index, "is_single_match_tie"] = tie_counts.eq(1)
    result = pd.DataFrame(
        {
            "match_id": events["match_id"],
            "season": events["season"],
            "kickoff_utc": events["kickoff_utc"],
            "competition": events["competition"],
            "round": events["round"],
            "tie_id": events["tie_id"],
            "is_knockout": events["is_knockout"],
            "is_tie_decider": events["is_tie_decider"],
            "is_single_match_tie": events["is_single_match_tie"],
            "stage": "",
            "home_team_id": events["home_team_id"],
            "away_team_id": events["away_team_id"],
            "home_goals": events["home_goals"],
            "away_goals": events["away_goals"],
            "xg_home": events["xg_home"].where(events["xg_analysis_eligible"]),
            "xg_away": events["xg_away"].where(events["xg_analysis_eligible"]),
            "xg_analysis_eligible": events["xg_analysis_eligible"],
            "is_neutral": events["is_neutral"],
            "decided_on_penalties": events["decided_on_penalties"],
            "advanced_team_id": events["advanced_team_id"],
        },
        columns=MATCH_INPUT_COLUMNS,
    )
    if len(result) != EXPECTED_MATCHES or result["match_id"].duplicated().any():
        raise ValueError(f"Expected {EXPECTED_MATCHES} unique 2025/26 matches")
    return result.reset_index(drop=True)


def verify(initial, state, updates, bonus_events, config) -> dict[str, object]:
    if len(updates) != EXPECTED_MATCHES:
        raise ValueError("Production replay match count is incorrect")
    eligible_deciders = updates.loc[
        updates["is_tie_decider"]
        & updates["stage"].isin(ELIGIBLE_PROGRESSION_STAGES)
    ]
    if len(bonus_events) != len(eligible_deciders):
        raise ValueError("Progression bonus event count is incorrect")
    if updates.loc[
        updates["stage"].eq("KNOCKOUT_PLAYOFF"),
        "progression_bonus_added",
    ].gt(0.0).any():
        raise ValueError("Knockout play-off cannot generate progression bonus")
    xg_eligible = int(updates["xg_analysis_eligible"].sum())
    xg_fallback = int(updates["xg_fallback_used"].sum())
    if xg_eligible != EXPECTED_XG_ELIGIBLE:
        raise ValueError("Production replay xG eligible count is incorrect")
    if xg_fallback != EXPECTED_XG_FALLBACK:
        raise ValueError("Production replay xG fallback count is incorrect")
    if updates.loc[updates["decided_on_penalties"], "xg_applied"].any():
        raise ValueError("Penalty-decided matches cannot apply xG adjustment")
    observed_values = sorted(bonus_events["progression_bonus_added"].unique().tolist())
    if observed_values != [4.0, 8.0, 12.0]:
        raise ValueError(f"Unexpected progression increments: {observed_values}")
    total_bonus = float(bonus_events["progression_bonus_added"].sum())
    expected_total_bonus = sum(
        config.fixed_progression_config.increment(competition)
        for competition in eligible_deciders["competition"]
    )
    if not math.isclose(total_bonus, expected_total_bonus, abs_tol=1e-9):
        raise ValueError("Total progression bonus is incorrect")
    maximum_cap = max(
        config.fixed_progression_config.cap(competition)
        for competition in ("UCL", "UEL", "UECL")
    )
    if float(updates["progression_bonus_competition_post"].max()) > maximum_cap + 1e-9:
        raise ValueError("Progression cap invariant failed")
    initial_power_total = float(initial["ao_first_elo"].sum())
    final_power_total = float(sum(rating.power_elo for rating in state.ratings.values()))
    power_total_error = abs(final_power_total - initial_power_total)
    if power_total_error > 1e-8:
        raise ValueError("Power Elo zero-sum invariant failed")
    live_accounting_error = max(
        abs(
            rating.ao_live_elo
            - rating.power_elo
            - rating.achievement_reserve
            - rating.progression_bonus_total
        )
        for rating in state.ratings.values()
    )
    if live_accounting_error > 1e-9:
        raise ValueError("AO Live Elo accounting invariant failed")
    maximum_team_bonus = max(
        rating.progression_bonus_total for rating in state.ratings.values()
    )
    return {
        "season": SEASON,
        "matches": len(updates),
        "teams": len(state.ratings),
        "config_id": config.config_id,
        "xg_eligible_matches": xg_eligible,
        "xg_applied_matches": int(updates["xg_applied"].sum()),
        "xg_fallback_matches": xg_fallback,
        "penalty_decisions": int(updates["decided_on_penalties"].sum()),
        "bonus_events": len(bonus_events),
        "knockout_playoff_bonus_events": 0,
        "bonus_values": observed_values,
        "total_bonus_added": total_bonus,
        "maximum_team_bonus": maximum_team_bonus,
        "processed_ties": len(state.processed_tie_ids),
        "open_ties_after_replay": len(state.open_ties),
        "power_total_error": power_total_error,
        "live_accounting_error": live_accounting_error,
    }


if __name__ == "__main__":
    main()
