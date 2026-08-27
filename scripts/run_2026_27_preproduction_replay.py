from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ao_elo.dynamic_csv import (  # noqa: E402
    load_selected_v2_config,
    run_batch,
    state_to_frame,
    updates_to_frame,
)
from ao_elo.config import AOEuropeanEloConfig  # noqa: E402
from ao_elo.pipeline import compute_ao_first_elo_from_csv  # noqa: E402


DEFAULT_DATA_ROOT = ROOT / "data" / "season_2026_27_preproduction"
DEFAULT_OUTPUT_ROOT = ROOT / "output" / "season_2026_27_preproduction"
DEFAULT_CONTRACT = ROOT / "contracts" / "ao_european_elo_v2_production.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay completed 2026/27 qualifiers without changing production"
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    replay_root = output_root / "q3_completed_replay"
    if replay_root.exists():
        for path in replay_root.iterdir():
            if path.is_file():
                path.unlink()
    replay_root.mkdir(parents=True, exist_ok=True)

    initial_path = output_root / "ao_first_elo_2026_27.csv"
    compute_ao_first_elo_from_csv(
        data_root / "teams.csv",
        data_root / "country_coefficients.csv",
        data_root / "domestic_context.csv",
        data_root / "club_european_points.csv",
        AOEuropeanEloConfig.active(),
        initial_path,
    )
    matches_path = data_root / "matches_completed.csv"
    fixtures_path = data_root / "fixtures_upcoming.csv"
    config = load_selected_v2_config(args.contract.resolve())
    final_state, updates = run_batch(
        initial_path,
        matches_path,
        replay_root,
        config,
    )

    initial = pd.read_csv(initial_path)
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
    final = final.sort_values(
        ["q3_end_rank", "team_id"], kind="stable"
    ).reset_index(drop=True)
    final.to_csv(
        output_root / "q3_end_live_ratings.csv",
        index=False,
        lineterminator="\n",
    )

    update_frame = updates_to_frame(updates)
    upcoming = pd.read_csv(fixtures_path)
    trajectories = build_trajectories(initial, update_frame)
    trajectories.to_csv(
        output_root / "q1_q3_team_rating_trajectories.csv",
        index=False,
        lineterminator="\n",
    )
    phase_summary = build_phase_summary(initial, final, update_frame, upcoming)
    phase_summary.to_csv(
        output_root / "team_rating_phase_summary.csv",
        index=False,
        lineterminator="\n",
    )
    phase_summary.loc[phase_summary["upcoming_playoff_participant"]].to_csv(
        output_root / "playoff_pre_match_team_ratings.csv",
        index=False,
        lineterminator="\n",
    )
    summary = build_summary(initial, final, update_frame, config.config_id)
    (output_root / "q3_replay_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_root / "q3_replay_report.md").write_text(
        build_report(final, update_frame, summary),
        encoding="utf-8",
    )

    print("AO 2026/27 preproduction Q1-Q3 replay")
    print(f"Matches: {len(update_frame)}")
    print(f"Teams: {len(final)}")
    print(f"Average absolute movement: {final['elo_change'].abs().mean():.3f}")
    print(f"Maximum gain: {final['elo_change'].max():.3f}")
    print(f"Maximum loss: {final['elo_change'].min():.3f}")
    print(f"Output: {output_root / 'q3_end_live_ratings.csv'}")


def build_trajectories(initial: pd.DataFrame, updates: pd.DataFrame) -> pd.DataFrame:
    names = initial.set_index("team_id")["team_name"].to_dict()
    rows = []
    for team in initial.itertuples(index=False):
        rows.append(
            {
                "event_order": 0,
                "match_id": "SEASON_START",
                "kickoff_utc": "",
                "team_id": team.team_id,
                "team_name": team.team_name,
                "opponent_team_id": "",
                "competition": "",
                "round": "",
                "stage_k_multiplier": "",
                "power_delta": 0.0,
                "ao_live_elo": float(team.ao_first_elo),
            }
        )
    for event_order, update in enumerate(updates.itertuples(index=False), start=1):
        rows.extend(
            [
                {
                    "event_order": event_order,
                    "match_id": update.match_id,
                    "kickoff_utc": update.kickoff_utc,
                    "team_id": update.home_team_id,
                    "team_name": names[update.home_team_id],
                    "opponent_team_id": update.away_team_id,
                    "competition": update.competition,
                    "round": update.round,
                    "stage_k_multiplier": update.stage_k_multiplier,
                    "power_delta": update.power_delta,
                    "ao_live_elo": update.home_live_post,
                },
                {
                    "event_order": event_order,
                    "match_id": update.match_id,
                    "kickoff_utc": update.kickoff_utc,
                    "team_id": update.away_team_id,
                    "team_name": names[update.away_team_id],
                    "opponent_team_id": update.home_team_id,
                    "competition": update.competition,
                    "round": update.round,
                    "stage_k_multiplier": update.stage_k_multiplier,
                    "power_delta": -update.power_delta,
                    "ao_live_elo": update.away_live_post,
                },
            ]
        )
    return pd.DataFrame(rows).sort_values(
        ["event_order", "team_id"], kind="stable"
    )


def build_phase_summary(
    initial: pd.DataFrame,
    final: pd.DataFrame,
    updates: pd.DataFrame,
    upcoming: pd.DataFrame,
) -> pd.DataFrame:
    ordered = updates.assign(
        _kickoff=pd.to_datetime(updates["kickoff_utc"], utc=True, errors="raise")
    ).sort_values(["_kickoff", "match_id"], kind="stable")
    phase_cutoffs = {
        phase: ordered.loc[ordered["qualification_round_key"].eq(phase), "_kickoff"].max()
        for phase in ("Q1", "Q2", "Q3")
    }
    initial_rating = initial.set_index("team_id")["ao_first_elo"].astype(float)
    snapshots: dict[str, dict[str, float]] = {}
    for phase, cutoff in phase_cutoffs.items():
        ratings = initial_rating.to_dict()
        for update in ordered.loc[ordered["_kickoff"].le(cutoff)].itertuples(index=False):
            ratings[update.home_team_id] = float(update.home_live_post)
            ratings[update.away_team_id] = float(update.away_live_post)
        snapshots[phase] = ratings

    match_counts: dict[str, dict[str, int]] = {}
    for phase in ("Q1", "Q2", "Q3"):
        phase_matches = ordered.loc[ordered["qualification_round_key"].eq(phase)]
        participants = pd.concat(
            [phase_matches["home_team_id"], phase_matches["away_team_id"]],
            ignore_index=True,
        )
        match_counts[phase] = participants.value_counts().astype(int).to_dict()

    completed_participants = set(ordered["home_team_id"]) | set(ordered["away_team_id"])
    upcoming_participants = set(upcoming["home_team_id"]) | set(upcoming["away_team_id"])
    final_lookup = final.set_index("team_id")
    rows: list[dict[str, object]] = []
    for team in initial.itertuples(index=False):
        current = float(final_lookup.loc[team.team_id, "ao_live_elo"])
        raw_change = current - float(team.ao_first_elo)
        is_qualifier = team.team_id in completed_participants or team.team_id in upcoming_participants
        row: dict[str, object] = {
            "season": team.season,
            "team_id": team.team_id,
            "team_name": team.team_name,
            "country": team.country,
            "country_code": team.country_code,
            "entry_competition": team.competition,
            "entry_round": team.entry_round,
            "ao_first_elo": float(team.ao_first_elo),
            "ao_first_rank": int(team.ao_first_elo_rank),
        }
        for phase in ("Q1", "Q2", "Q3"):
            phase_rating = float(snapshots[phase][team.team_id])
            row[f"{phase.lower()}_matches"] = match_counts[phase].get(team.team_id, 0)
            row[f"{phase.lower()}_end_live_elo"] = phase_rating
            row[f"{phase.lower()}_cumulative_change"] = (
                phase_rating - float(team.ao_first_elo)
            )
        row.update(
            {
                "completed_qualifier_matches": sum(
                    match_counts[phase].get(team.team_id, 0)
                    for phase in ("Q1", "Q2", "Q3")
                ),
                "played_qualifier_to_date": team.team_id in completed_participants,
                "upcoming_playoff_participant": team.team_id in upcoming_participants,
                "pre_playoff_live_elo": current,
                "pre_playoff_rank": int(final_lookup.loc[team.team_id, "q3_end_rank"]),
                "pre_playoff_live_elo_change": raw_change,
                "pre_playoff_rank_change": int(
                    final_lookup.loc[team.team_id, "rank_change"]
                ),
                "qualifier_delta_retention_rate": 0.50 if is_qualifier else 0.0,
                "main_entry_reset_applied_at_this_cutoff": False,
                "main_entry_elo_if_no_further_qualifier_change": current,
                "non_match_main_entry_adjustment": 0.0,
                "qualification_transition_rule": (
                    "CONTINUOUS_MATCH_UPDATE_NO_MAIN_ENTRY_RESET"
                    if is_qualifier
                    else "DIRECT_ENTRANT_MAIN_K_1_NO_RESET"
                ),
                "projection_status": (
                    "CURRENT_LIVE_ELO_CONTINUES_PLAYOFF_RESULTS_NOT_INCLUDED"
                    if is_qualifier
                    else "NOT_APPLICABLE"
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["pre_playoff_rank", "team_id"], kind="stable"
    ).reset_index(drop=True)


def build_summary(
    initial: pd.DataFrame,
    final: pd.DataFrame,
    updates: pd.DataFrame,
    config_id: str,
) -> dict[str, object]:
    zero_sum_error = (
        (updates["home_power_post"] - updates["home_power_pre"])
        + (updates["away_power_post"] - updates["away_power_pre"])
    ).abs()
    multiplier_counts = (
        updates.groupby(["qualification_round_key", "stage_k_multiplier"])
        .size()
        .rename("matches")
        .reset_index()
        .to_dict(orient="records")
    )
    return {
        "evidence_class": "RETROSPECTIVE_PREPRODUCTION_REPLAY",
        "prospective_holdout_evidence": False,
        "season": "2026/27",
        "cutoff": "AFTER_COMPLETED_Q3_BEFORE_QUALIFYING_PLAYOFF",
        "matches": len(updates),
        "teams": len(final),
        "config_id": config_id,
        "xg_eligible_matches": int(updates["xg_applied"].sum()),
        "goal_margin_fallback_matches": int((~updates["xg_applied"]).sum()),
        "max_zero_sum_error": float(zero_sum_error.max()),
        "total_power_change": float(final["elo_change"].sum()),
        "average_absolute_elo_change": float(final["elo_change"].abs().mean()),
        "median_absolute_elo_change": float(final["elo_change"].abs().median()),
        "p95_absolute_elo_change": float(final["elo_change"].abs().quantile(0.95)),
        "maximum_gain": float(final["elo_change"].max()),
        "maximum_loss": float(final["elo_change"].min()),
        "maximum_rank_gain": int(final["rank_change"].max()),
        "maximum_rank_loss": int(final["rank_change"].min()),
        "stage_k_usage": multiplier_counts,
        "qualification_carry_applied": False,
        "qualification_carry_explanation": (
            "Qualifier delta retention is embedded in effective Q1/Q2/Q3/QPO "
            "multipliers; MAIN entry never changes rating without a match."
        ),
        "initial_rating_min": float(initial["ao_first_elo"].min()),
        "initial_rating_max": float(initial["ao_first_elo"].max()),
        "q3_end_rating_min": float(final["ao_live_elo"].min()),
        "q3_end_rating_max": float(final["ao_live_elo"].max()),
    }


def build_report(
    final: pd.DataFrame,
    updates: pd.DataFrame,
    summary: dict[str, object],
) -> str:
    movers = final.sort_values("elo_change", ascending=False)
    top = markdown_table(
        movers.head(15)[
            ["team_name", "ao_first_elo", "ao_live_elo", "elo_change", "rank_change"]
        ],
        float_digits=2,
    )
    bottom = markdown_table(
        movers.tail(15).sort_values("elo_change")[
            ["team_name", "ao_first_elo", "ao_live_elo", "elo_change", "rank_change"]
        ],
        float_digits=2,
    )
    stage_frame = (
        updates.groupby(["qualification_round_key", "stage_k_multiplier"])
        .agg(
            matches=("match_id", "size"),
            mean_abs_delta=("power_delta", lambda x: x.abs().mean()),
            max_abs_delta=("power_delta", lambda x: x.abs().max()),
        )
        .reset_index()
    )
    stage = markdown_table(stage_frame, float_digits=3)
    return f"""# AO 2026/27 Q1-Q3 Preproduction Replay

Bu calisma production degisikligi degildir. {summary['matches']} tamamlanmis
UEFA eleme maci, aktif production stage-K sozlesmesiyle kronolojik olarak replay
edilmistir. Play-off maclari henuz sonuclanmadigi icin bu snapshot Q3 sonunu
gosterir. `%50` qualifier delta retention efektif stage-K degerlerine gomuludur;
ana asamaya geciste ayri carry veya mekanik rating degisimi yoktur.

## Veri Davranisi

- xG uygun mac: {summary['xg_eligible_matches']}
- Goal-margin fallback: {summary['goal_margin_fallback_matches']}
- Ortalama mutlak Elo hareketi: {summary['average_absolute_elo_change']:.3f}
- P95 mutlak Elo hareketi: {summary['p95_absolute_elo_change']:.3f}
- En buyuk artis: {summary['maximum_gain']:.3f}
- En buyuk dusus: {summary['maximum_loss']:.3f}
- Maksimum zero-sum hatasi: {summary['max_zero_sum_error']:.3e}

## Stage-K Kullanimi

{stage}

## En Cok Yukselenler

{top}

## En Cok Dusenler

{bottom}

## Sinir

Bu replay gercek pre-match ledger degildir ve prospective holdout kaniti olarak
kullanilamaz. 2026/27 xG henuz iki tarafli ve zaman kapsami dogrulanmis bicimde
toplanmadigi icin tum maclar goal-margin fallback'iyle islenmistir.
"""


def markdown_table(frame: pd.DataFrame, *, float_digits: int) -> str:
    def render(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.{float_digits}f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
