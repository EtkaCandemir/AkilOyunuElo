from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ao_elo.config import AOEuropeanEloConfig
from ao_elo.dynamic_csv import (
    load_selected_v2_config,
    run_batch,
    state_to_frame,
    updates_to_frame,
)
from ao_elo.pipeline import compute_ao_first_elo_from_csv
from scripts.build_2026_27_preproduction_inputs import (
    entry_competition,
    stable_match_order,
)
from scripts.run_2026_27_preproduction_replay import (
    build_phase_summary,
    markdown_table,
)


DATA_ROOT = ROOT / "data" / "season_2026_27_preproduction"
CONTRACT = ROOT / "contracts" / "ao_european_elo_v2_production.json"


def test_entry_competition_maps_supported_routes() -> None:
    assert entry_competition("CL-Q1") == "UCL"
    assert entry_competition("EL-LS") == "UEL"
    assert entry_competition("CO-Q2") == "UECL"
    with pytest.raises(ValueError, match="Unknown European entry round"):
        entry_competition("UNKNOWN")


def test_stable_match_order_uses_kickoff_then_match_id() -> None:
    frame = pd.DataFrame(
        {
            "match_id": ["m3", "m2", "m1"],
            "kickoff_utc": [
                "2026-07-02T18:00:00Z",
                "2026-07-01T18:00:00Z",
                "2026-07-01T18:00:00Z",
            ],
        }
    )
    ordered = stable_match_order(frame)
    assert ordered["match_id"].tolist() == ["m1", "m2", "m3"]
    assert ordered["event_order"].tolist() == [1, 2, 3]


def test_curated_2026_27_input_snapshot_contract() -> None:
    teams = pd.read_csv(DATA_ROOT / "teams.csv")
    identity = pd.read_csv(DATA_ROOT / "team_identity_audit.csv", dtype=str)
    completed = pd.read_csv(DATA_ROOT / "matches_completed.csv")
    upcoming = pd.read_csv(DATA_ROOT / "fixtures_upcoming.csv")

    assert len(teams) == 237
    assert len(completed) == 342
    assert len(upcoming) == 86
    assert teams["team_id"].is_unique
    assert identity["uefa_team_id"].is_unique
    assert set(teams["team_id"]) == set(identity["team_id"])
    assert identity["team_id"].map(
        lambda value: bool(re.fullmatch(r"AO-UEFA-\d+", value))
    ).all()
    assert completed["match_id"].is_unique
    assert upcoming["match_id"].is_unique
    assert completed["event_order"].tolist() == list(range(1, 343))
    assert upcoming["event_order"].tolist() == list(range(1, 87))
    # xG is present from 2026/27 onward wherever the audited source has it.
    # The invariant is the acceptance contract, not the count: a row either
    # carries both sides or none, and a missing value is never imputed.
    eligible = completed["xg_analysis_eligible"].astype(bool)
    assert completed.loc[eligible, ["xg_home", "xg_away"]].notna().all().all()
    assert completed.loc[~eligible, ["xg_home", "xg_away"]].isna().all().all()
    assert completed.loc[eligible, "xg_fallback"].eq("FOTMOB_PRIMARY").all()
    assert completed.loc[~eligible, "xg_fallback"].eq("GOAL_MARGIN_ONLY").all()


def test_preproduction_q1_q3_replay_obeys_active_contract(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ao_first_elo.csv"
    compute_ao_first_elo_from_csv(
        DATA_ROOT / "teams.csv",
        DATA_ROOT / "country_coefficients.csv",
        DATA_ROOT / "domestic_context.csv",
        DATA_ROOT / "club_european_points.csv",
        AOEuropeanEloConfig.active(),
        ratings_path,
    )
    config = load_selected_v2_config(CONTRACT)
    final_state, updates = run_batch(
        ratings_path,
        DATA_ROOT / "matches_completed.csv",
        tmp_path / "replay",
        config,
    )
    frame = updates_to_frame(updates)
    initial = pd.read_csv(ratings_path)
    final = state_to_frame(final_state).merge(
        initial[["team_id", "ao_first_elo_rank"]],
        on="team_id",
        how="left",
        validate="one_to_one",
    )
    final["q3_end_rank"] = final["ao_live_elo"].rank(
        method="min", ascending=False
    ).astype(int)
    final["elo_change"] = final["ao_live_elo"] - final["ao_first_elo"]
    final["rank_change"] = final["ao_first_elo_rank"] - final["q3_end_rank"]
    phase_summary = build_phase_summary(
        initial,
        final,
        frame,
        pd.read_csv(DATA_ROOT / "fixtures_upcoming.csv"),
    )

    assert len(state_to_frame(final_state)) == 237
    assert len(frame) == 342
    assert (
        frame.groupby("qualification_round_key")["stage_k_multiplier"].first().to_dict()
        == {"Q1": 0.20, "Q2": 0.275, "Q3": 0.35}
    )
    assert frame["qualification_round_key"].value_counts().to_dict() == {
        "Q2": 144,
        "Q3": 106,
        "Q1": 92,
    }
    assert not frame["home_qualifier_carry_applied"].any()
    assert not frame["away_qualifier_carry_applied"].any()
    # The kernel suppresses xG on draws, so an applied adjustment implies both
    # an eligible row and a decisive field result. Anything else would mean the
    # settlement path stopped honouring its own eligibility gate.
    results = pd.read_csv(DATA_ROOT / "matches_completed.csv")
    eligible_ids = set(
        results.loc[results["xg_analysis_eligible"].astype(bool), "match_id"]
    )
    decisive_ids = set(
        results.loc[results["home_goals"].ne(results["away_goals"]), "match_id"]
    )
    applied = set(frame.loc[frame["xg_applied"], "match_id"])
    assert applied <= eligible_ids & decisive_ids
    assert not frame.loc[
        frame["xg_applied"] & frame["home_goals"].eq(frame["away_goals"])
    ].shape[0]
    zero_sum_error = (
        frame["home_power_post"]
        - frame["home_power_pre"]
        + frame["away_power_post"]
        - frame["away_power_pre"]
    ).abs()
    assert zero_sum_error.max() < 1e-9
    assert len(phase_summary) == 237
    assert int(phase_summary["upcoming_playoff_participant"].sum()) == 86
    assert not phase_summary["main_entry_reset_applied_at_this_cutoff"].any()
    assert phase_summary["q3_end_live_elo"].equals(
        phase_summary["pre_playoff_live_elo"]
    )
    qualifier = phase_summary.loc[
        phase_summary["qualification_transition_rule"].eq(
            "CONTINUOUS_MATCH_UPDATE_NO_MAIN_ENTRY_RESET"
        )
    ]
    assert qualifier["qualifier_delta_retention_rate"].eq(0.50).all()
    assert qualifier["non_match_main_entry_adjustment"].eq(0.0).all()
    assert qualifier[
        "main_entry_elo_if_no_further_qualifier_change"
    ].to_numpy() == pytest.approx(qualifier["pre_playoff_live_elo"].to_numpy())


def test_markdown_table_has_no_optional_dependency() -> None:
    rendered = markdown_table(
        pd.DataFrame({"team": ["A|B"], "rating": [1234.567]}),
        float_digits=2,
    )
    assert "A\\|B" in rendered
    assert "1234.57" in rendered


# ---------------------------------------------------------------------------
# the Domestic Surprise history window must never contain its own current
# season: a season scored against a window holding itself is damped by exactly
# that season's weight, which silently mutes the layer for the affected teams
# ---------------------------------------------------------------------------


# Only the qualification route and the champion flag carry the season that
# decided 2026/27 entry. Everything else falls back to the newest cached table,
# which is one season older, so its history window has to shift with it.
CURRENT_SEASON_SOURCES = {"QUALIFICATION_CHAMPION", "QUALIFICATION_ROUTE"}


@pytest.fixture(name="history_audit")
def fixture_history_audit() -> pd.DataFrame:
    return pd.read_csv(DATA_ROOT / "domestic_history_audit.csv")


def test_history_window_ends_before_the_current_position_vintage(
    history_audit: pd.DataFrame,
) -> None:
    """Every team's five-season window must stop one year short of its own."""
    for row in history_audit.itertuples(index=False):
        start, end = (int(part) for part in str(row.history_window).split("-"))

        assert end == int(row.current_position_vintage) - 1
        assert end - start == 4


def test_vintage_follows_the_position_source(
    history_audit: pd.DataFrame,
) -> None:
    """Route and champion positions are current; every fallback is older."""
    for row in history_audit.itertuples(index=False):
        current = row.current_position_source in CURRENT_SEASON_SOURCES
        expected = 2026 if current else 2025

        assert int(row.current_position_vintage) == expected

    assert history_audit["current_position_vintage"].isin({2025, 2026}).all()


def test_cache_fallback_teams_are_not_scored_against_themselves(
    history_audit: pd.DataFrame,
) -> None:
    """The regression signature was every fallback team matching t_minus_1."""
    context = pd.read_csv(DATA_ROOT / "domestic_context.csv")
    merged = history_audit.merge(context, on="team_id", suffixes=("", "_context"))
    fallback = merged.loc[
        ~merged["current_position_source"].isin(CURRENT_SEASON_SOURCES)
    ].dropna(subset=["domestic_position", "history_position_t_minus_1"])

    assert not fallback.empty
    repeated = fallback["domestic_position"].eq(
        fallback["history_position_t_minus_1"]
    )

    # Consecutive seasons repeat a finishing position often enough that a low
    # rate is expected; a perfect match across the group is the defect.
    assert repeated.mean() < 0.5


def test_self_reference_rate_stays_in_the_backtest_range() -> None:
    """The 2018-2026 backtest seasons sit between 20% and 31%."""
    context = pd.read_csv(DATA_ROOT / "domestic_context.csv").dropna(
        subset=["domestic_position", "history_position_t_minus_1"]
    )
    rate = (
        context["domestic_position"]
        .eq(context["history_position_t_minus_1"])
        .mean()
    )

    assert rate < 0.31
