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
    preserve_existing_xg,
    select_primary_standings,
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


def test_primary_standings_rejects_position_by_round_table() -> None:
    position_by_round = pd.DataFrame(
        {
            "position": [3, 1, 2, 4],
            "team_name": ["A", "B", "C", "D"],
            "team_key": ["a", "b", "c", "d"],
        }
    )
    final_table = pd.DataFrame(
        {
            "position": [1, 2, 3, 4],
            "team_name": ["B", "C", "A", "D"],
            "team_key": ["b", "c", "a", "d"],
        }
    )

    selected = select_primary_standings(
        [position_by_round, final_table], country="Testland"
    )

    assert selected["team_name"].tolist() == ["B", "C", "A", "D"]


def test_static_rebuild_preserves_only_verified_existing_xg() -> None:
    rebuilt = pd.DataFrame(
        {
            "match_id": ["m1", "m2"],
            "home_team_id": ["a", "c"],
            "away_team_id": ["b", "d"],
            "home_goals": [2, 0],
            "away_goals": [1, 0],
            "xg_home": [pd.NA, pd.NA],
            "xg_away": [pd.NA, pd.NA],
            "xg_analysis_eligible": [False, False],
            "xg_fallback": ["GOAL_MARGIN_ONLY", "GOAL_MARGIN_ONLY"],
        }
    )
    existing = rebuilt.copy()
    existing.loc[0, ["xg_home", "xg_away"]] = [1.8, 0.7]
    existing.loc[0, "xg_analysis_eligible"] = True
    existing.loc[0, "xg_fallback"] = "FOTMOB_PRIMARY"

    result = preserve_existing_xg(rebuilt, existing)

    assert result.loc[0, "xg_home"] == pytest.approx(1.8)
    assert result.loc[0, "xg_away"] == pytest.approx(0.7)
    assert bool(result.loc[0, "xg_analysis_eligible"])
    assert result.loc[0, "xg_fallback"] == "FOTMOB_PRIMARY"
    assert not bool(result.loc[1, "xg_analysis_eligible"])


def test_static_rebuild_rejects_stale_xg_identity() -> None:
    rebuilt = pd.DataFrame(
        {
            "match_id": ["m1"],
            "home_team_id": ["a"],
            "away_team_id": ["b"],
            "home_goals": [2],
            "away_goals": [1],
            "xg_home": [pd.NA],
            "xg_away": [pd.NA],
            "xg_analysis_eligible": [False],
            "xg_fallback": ["GOAL_MARGIN_ONLY"],
        }
    )
    existing = rebuilt.copy()
    existing.loc[0, "home_goals"] = 3
    existing.loc[0, ["xg_home", "xg_away"]] = [1.8, 0.7]
    existing.loc[0, "xg_analysis_eligible"] = True

    with pytest.raises(ValueError, match="identity/score mismatch"):
        preserve_existing_xg(rebuilt, existing)


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
    assert initial["effective_european_exposure"].max() == pytest.approx(0.65)
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
    """Every route uses the completed season that decided 2026/27 entry."""
    assert history_audit["current_position_vintage"].eq(2026).all()
    assert history_audit["history_window"].eq("2021-2025").all()


def test_cup_routes_never_use_a_stale_domestic_fallback(
    history_audit: pd.DataFrame,
) -> None:
    """Cup winners must use current evidence or an explicit unavailable state."""
    context = pd.read_csv(DATA_ROOT / "domestic_context.csv")
    merged = history_audit.merge(context, on="team_id", suffixes=("", "_context"))
    cup = merged.loc[merged["route"].eq("CW")]

    assert len(cup) == 36
    assert cup["current_position_vintage"].eq(2026).all()
    assert cup["current_source_url"].fillna("").ne("").all()
    assert not cup["current_position_source"].str.contains("FALLBACK").any()
    assert set(cup.loc[cup["domestic_position"].isna(), "team_name"]) == {
        "FC Vaduz",
        "Lillestrøm SK",
        "União Torreense",
    }

    dedicated_audit = pd.read_csv(DATA_ROOT / "cw_domestic_evidence_audit.csv")
    assert len(dedicated_audit) == 36
    assert dedicated_audit["team_id"].is_unique
    assert set(dedicated_audit["team_id"]) == set(cup["team_id"])


def test_trabzonspor_uses_the_completed_2025_26_final_table() -> None:
    context = pd.read_csv(DATA_ROOT / "domestic_context.csv")
    audit = pd.read_csv(DATA_ROOT / "domestic_history_audit.csv")
    team_id = "AO-UEFA-52731"
    row = context.loc[context["team_id"].eq(team_id)].iloc[0]
    evidence = audit.loc[audit["team_id"].eq(team_id)].iloc[0]

    assert float(row["domestic_position"]) == 3.0
    assert float(row["league_team_count"]) == 18.0
    assert int(evidence["current_position_vintage"]) == 2026
    assert "2025" in str(evidence["current_source_url"])


def test_all_cup_winner_domestic_positions_match_the_frozen_snapshot() -> None:
    expected = {
        "Dinamo Tirana": (4, 10),
        "Atlètic Club d'Escaldes": (5, 10),
        "FC Noah": (2, 10),
        "BATE Borisov": (10, 16),
        "Zrinjski Mostar": (2, 10),
        "CSKA Sofia": (4, 16),
        "Pafos FC": (4, 14),
        "FC Midtjylland": (2, 12),
        "HJK Helsinki": (3, 12),
        "Dila Gori": (2, 10),
        "OFI Heraklion": (6, 14),
        "Ferencváros": (2, 12),
        "IF Vestri": (9, 12),
        "Maccabi Tel-Aviv": (3, 14),
        "Tobol Kustanai": (3, 14),
        "KF Dukagjini": (4, 10),
        "FK Auda": (5, 10),
        "FC Vaduz": (None, None),
        "FK Panevezys": (6, 10),
        "Differdange 03": (2, 16),
        "Valletta FC": (3, 12),
        "Sheriff Tiraspol": (3, 8),
        "Mornar Bar": (2, 10),
        "AZ Alkmaar": (7, 18),
        "Sileks Kratovo": (4, 12),
        "Coleraine": (3, 12),
        "Lillestrøm SK": (None, None),
        "União Torreense": (None, None),
        "La Fiorita": (4, 16),
        "MSK Zilina": (3, 12),
        "NK Aluminij": (7, 10),
        "Real Sociedad": (10, 20),
        "FC Sankt Gallen": (2, 12),
        "Trabzonspor": (3, 18),
        "Dynamo Kyiv": (4, 16),
        "Caernarfon Town": (4, 12),
    }
    audit = pd.read_csv(DATA_ROOT / "cw_domestic_evidence_audit.csv")

    assert set(audit["team_name"]) == set(expected)
    for row in audit.itertuples(index=False):
        actual_position = None if pd.isna(row.current_position) else int(row.current_position)
        actual_count = None if pd.isna(row.current_team_count) else int(row.current_team_count)
        assert (actual_position, actual_count) == expected[row.team_name]


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
