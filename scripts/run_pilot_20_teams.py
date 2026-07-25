from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ao_elo.dynamic_csv import (  # noqa: E402
    load_selected_v2_config,
    run_batch,
)


DATA_ROOT = ROOT / "data" / "pilot_20_teams"
OUTPUT_ROOT = ROOT / "output" / "pilot_20_teams"
PRODUCTION_REPLAY = OUTPUT_ROOT / "production_replay"
CLASSIC_REPLAY = OUTPUT_ROOT / "classic_no_goal_difference_replay"
MANIFEST = ROOT / "contracts" / "ao_european_elo_v2_production.json"


def main() -> None:
    results = run_pilot(DATA_ROOT, OUTPUT_ROOT)
    print("AO European Elo 20-team synthetic pilot complete")
    print(f"Teams: {len(results['teams'])}")
    print(f"Matches: {len(results['matches'])}")
    print(
        "Goal multiplier range: "
        f"{results['matches']['goal_multiplier'].min():.6f} - "
        f"{results['matches']['goal_multiplier'].max():.6f}"
    )
    print(
        "AO Live Elo range: "
        f"{results['teams']['final_live_elo'].min():.3f} - "
        f"{results['teams']['final_live_elo'].max():.3f}"
    )
    print(
        "Maximum |production - classic| final Elo: "
        f"{results['teams']['goal_difference_net_effect_vs_classic'].abs().max():.3f}"
    )
    print(f"Output: {OUTPUT_ROOT}")


def run_pilot(
    data_root: Path = DATA_ROOT,
    output_root: Path = OUTPUT_ROOT,
) -> dict[str, pd.DataFrame]:
    output_root.mkdir(parents=True, exist_ok=True)
    initial_path = data_root / "initial_ratings.csv"
    matches_path = data_root / "matches.csv"
    initial = pd.read_csv(initial_path)
    match_metadata = pd.read_csv(matches_path)
    _validate_inputs(initial, match_metadata)

    production_config = load_selected_v2_config(MANIFEST)
    classic_config = replace(
        production_config,
        goal_difference_enabled=False,
        goal_alpha=0.0,
    )
    production_state, production_updates = run_batch(
        initial_path,
        matches_path,
        output_root / "production_replay",
        production_config,
    )
    classic_state, classic_updates = run_batch(
        initial_path,
        matches_path,
        output_root / "classic_no_goal_difference_replay",
        classic_config,
    )

    production = pd.read_csv(
        output_root / "production_replay" / "match_updates.csv"
    )
    classic = pd.read_csv(
        output_root
        / "classic_no_goal_difference_replay"
        / "match_updates.csv"
    )
    detailed_matches = build_match_detail(
        initial,
        match_metadata,
        production,
        classic,
        production_config.k_factor,
    )
    team_summary = build_team_summary(
        initial,
        detailed_matches,
        production_state,
        classic_state,
    )
    scenario_summary = build_scenario_summary(detailed_matches)
    competition_summary = build_competition_summary(team_summary, detailed_matches)
    parameters = pd.DataFrame(
        [
            ("model_version", production_config.model_version),
            ("config_id", production_config.config_id),
            ("elo_scale", production_config.elo_scale),
            ("home_advantage", production_config.home_advantage),
            ("k_factor", production_config.k_factor),
            ("power_carry", production_config.power_carry),
            ("draw_at_even", production_config.draw_at_even),
            ("draw_shape", production_config.draw_shape),
            ("goal_difference_enabled", production_config.goal_difference_enabled),
            ("goal_alpha", production_config.goal_alpha),
            ("goal_tau", production_config.goal_tau),
            ("goal_difference_cap", production_config.goal_difference_cap),
            ("progression_bonus_enabled", False),
            ("achievement_reserve_base", production_config.reserve_base),
            ("competition_k_enabled", False),
        ],
        columns=["parameter", "value"],
    )

    team_summary.to_csv(output_root / "team_start_end_summary.csv", index=False)
    detailed_matches.to_csv(output_root / "match_updates_detailed.csv", index=False)
    scenario_summary.to_csv(output_root / "scenario_summary.csv", index=False)
    competition_summary.to_csv(
        output_root / "competition_summary.csv",
        index=False,
    )
    parameters.to_csv(output_root / "model_parameters.csv", index=False)
    write_markdown_report(
        output_root / "pilot_report.md",
        team_summary,
        detailed_matches,
        competition_summary,
        production_config.config_id,
    )
    _validate_results(
        team_summary,
        detailed_matches,
        production_updates,
        classic_updates,
        production_config.config_id,
    )
    return {
        "teams": team_summary,
        "matches": detailed_matches,
        "scenarios": scenario_summary,
        "competitions": competition_summary,
        "parameters": parameters,
    }


def build_match_detail(
    initial: pd.DataFrame,
    metadata: pd.DataFrame,
    production: pd.DataFrame,
    classic: pd.DataFrame,
    k_factor: float,
) -> pd.DataFrame:
    names = initial.set_index("team_id")["team_name"].to_dict()
    metadata_columns = [
        "match_id",
        "scenario_group",
        "scenario_note",
    ]
    detail = production.merge(
        metadata[metadata_columns],
        on="match_id",
        validate="one_to_one",
    )
    classic_view = classic[
        [
            "match_id",
            "home_power_pre",
            "away_power_pre",
            "expected_home_score",
            "power_delta",
            "home_power_post",
            "away_power_post",
        ]
    ].rename(
        columns={
            "home_power_pre": "classic_home_power_pre",
            "away_power_pre": "classic_away_power_pre",
            "expected_home_score": "classic_expected_home_score",
            "power_delta": "classic_replay_power_delta",
            "home_power_post": "classic_home_power_post",
            "away_power_post": "classic_away_power_post",
        }
    )
    detail = detail.merge(classic_view, on="match_id", validate="one_to_one")
    detail["home_team_name"] = detail["home_team_id"].map(names)
    detail["away_team_name"] = detail["away_team_id"].map(names)
    detail["classic_delta_same_state"] = (
        float(k_factor)
        * (detail["actual_home_score"] - detail["expected_home_score"])
    )
    detail["gd_extra_delta_same_state"] = (
        detail["power_delta"] - detail["classic_delta_same_state"]
    )
    detail["absolute_gd_extra_same_state"] = detail[
        "gd_extra_delta_same_state"
    ].abs()
    detail["production_vs_classic_delta"] = (
        detail["power_delta"] - detail["classic_replay_power_delta"]
    )
    detail["result"] = detail.apply(_result_label, axis=1)
    detail["winner_team_id"] = detail.apply(_winner_team_id, axis=1)
    detail["winner_team_name"] = detail["winner_team_id"].map(names).fillna("Draw")
    detail["winner_rating_gain"] = detail["power_delta"].abs().where(
        detail["actual_home_score"].ne(0.5),
        0.0,
    )
    detail["total_power_zero_sum_error"] = (
        detail["home_power_post"]
        + detail["away_power_post"]
        - detail["home_power_pre"]
        - detail["away_power_pre"]
    ).abs()
    ordered = [
        "match_id",
        "kickoff_utc",
        "competition",
        "round",
        "scenario_group",
        "scenario_note",
        "home_team_id",
        "home_team_name",
        "away_team_id",
        "away_team_name",
        "home_goals",
        "away_goals",
        "result",
        "is_neutral",
        "decided_on_penalties",
        "effective_rating_difference",
        "expected_home_score",
        "home_win_probability",
        "draw_probability",
        "away_win_probability",
        "actual_home_score",
        "goal_difference",
        "goal_multiplier",
        "classic_delta_same_state",
        "gd_extra_delta_same_state",
        "absolute_gd_extra_same_state",
        "power_delta",
        "home_power_pre",
        "away_power_pre",
        "home_power_post",
        "away_power_post",
        "classic_replay_power_delta",
        "production_vs_classic_delta",
        "winner_team_id",
        "winner_team_name",
        "winner_rating_gain",
        "progression_reserve_added",
        "trophy_reserve_added",
        "total_power_zero_sum_error",
        "config_id",
    ]
    return detail.loc[:, ordered]


def build_team_summary(
    initial: pd.DataFrame,
    matches: pd.DataFrame,
    production_state,
    classic_state,
) -> pd.DataFrame:
    summary = initial.copy()
    production_final = {
        team_id: rating.power_elo
        for team_id, rating in production_state.ratings.items()
    }
    classic_final = {
        team_id: rating.power_elo
        for team_id, rating in classic_state.ratings.items()
    }
    summary["start_elo"] = summary["ao_first_elo"].astype(float)
    summary["final_power_elo"] = summary["team_id"].map(production_final)
    summary["final_live_elo"] = summary["final_power_elo"]
    summary["classic_final_elo"] = summary["team_id"].map(classic_final)
    summary["total_elo_change"] = summary["final_live_elo"] - summary["start_elo"]
    summary["goal_difference_net_effect_vs_classic"] = (
        summary["final_live_elo"] - summary["classic_final_elo"]
    )

    perspectives = []
    for row in matches.itertuples(index=False):
        perspectives.append(
            {
                "team_id": row.home_team_id,
                "goals_for": row.home_goals,
                "goals_against": row.away_goals,
                "result": (
                    "W"
                    if row.actual_home_score == 1.0
                    else "D" if row.actual_home_score == 0.5 else "L"
                ),
                "goal_multiplier": row.goal_multiplier,
                "absolute_gd_extra_same_state": row.absolute_gd_extra_same_state,
            }
        )
        perspectives.append(
            {
                "team_id": row.away_team_id,
                "goals_for": row.away_goals,
                "goals_against": row.home_goals,
                "result": (
                    "L"
                    if row.actual_home_score == 1.0
                    else "D" if row.actual_home_score == 0.5 else "W"
                ),
                "goal_multiplier": row.goal_multiplier,
                "absolute_gd_extra_same_state": row.absolute_gd_extra_same_state,
            }
        )
    perspective = pd.DataFrame(perspectives)
    aggregates = (
        perspective.groupby("team_id", as_index=False)
        .agg(
            matches=("result", "size"),
            wins=("result", lambda values: int((values == "W").sum())),
            draws=("result", lambda values: int((values == "D").sum())),
            losses=("result", lambda values: int((values == "L").sum())),
            goals_for=("goals_for", "sum"),
            goals_against=("goals_against", "sum"),
            average_goal_multiplier=("goal_multiplier", "mean"),
            maximum_goal_multiplier=("goal_multiplier", "max"),
            cumulative_absolute_gd_extra=(
                "absolute_gd_extra_same_state",
                "sum",
            ),
        )
    )
    summary = summary.merge(aggregates, on="team_id", validate="one_to_one")
    summary["start_rank"] = (
        summary["start_elo"].rank(method="min", ascending=False).astype(int)
    )
    summary["final_rank"] = (
        summary["final_live_elo"].rank(method="min", ascending=False).astype(int)
    )
    summary["rank_change"] = summary["start_rank"] - summary["final_rank"]
    ordered = [
        "team_id",
        "team_name",
        "competition_track",
        "strength_band",
        "synthetic_profile",
        "start_rank",
        "start_elo",
        "final_rank",
        "final_live_elo",
        "total_elo_change",
        "rank_change",
        "classic_final_elo",
        "goal_difference_net_effect_vs_classic",
        "matches",
        "wins",
        "draws",
        "losses",
        "goals_for",
        "goals_against",
        "average_goal_multiplier",
        "maximum_goal_multiplier",
        "cumulative_absolute_gd_extra",
    ]
    return summary.loc[:, ordered].sort_values("final_rank").reset_index(drop=True)


def build_scenario_summary(matches: pd.DataFrame) -> pd.DataFrame:
    return (
        matches.groupby("scenario_group", as_index=False)
        .agg(
            matches=("match_id", "size"),
            average_abs_D=("effective_rating_difference", lambda x: x.abs().mean()),
            average_goal_difference=("goal_difference", "mean"),
            average_multiplier=("goal_multiplier", "mean"),
            maximum_multiplier=("goal_multiplier", "max"),
            average_abs_power_delta=("power_delta", lambda x: x.abs().mean()),
            average_abs_gd_extra=(
                "gd_extra_delta_same_state",
                lambda x: x.abs().mean(),
            ),
        )
        .sort_values(["maximum_multiplier", "scenario_group"], ascending=[False, True])
        .reset_index(drop=True)
    )


def build_competition_summary(
    teams: pd.DataFrame,
    matches: pd.DataFrame,
) -> pd.DataFrame:
    team_view = (
        teams.groupby("competition_track", as_index=False)
        .agg(
            teams=("team_id", "size"),
            start_elo_sum=("start_elo", "sum"),
            final_elo_sum=("final_live_elo", "sum"),
            start_elo_mean=("start_elo", "mean"),
            final_elo_mean=("final_live_elo", "mean"),
            largest_gain=("total_elo_change", "max"),
            largest_loss=("total_elo_change", "min"),
            gd_net_effect_sum=("goal_difference_net_effect_vs_classic", "sum"),
        )
        .rename(columns={"competition_track": "competition"})
    )
    match_view = (
        matches.groupby("competition", as_index=False)
        .agg(
            matches=("match_id", "size"),
            average_multiplier=("goal_multiplier", "mean"),
            maximum_multiplier=("goal_multiplier", "max"),
            total_abs_gd_extra=(
                "absolute_gd_extra_same_state",
                "sum",
            ),
        )
    )
    result = team_view.merge(match_view, on="competition", validate="one_to_one")
    result["elo_conservation_error"] = (
        result["final_elo_sum"] - result["start_elo_sum"]
    ).abs()
    return result.sort_values("competition").reset_index(drop=True)


def write_markdown_report(
    path: Path,
    teams: pd.DataFrame,
    matches: pd.DataFrame,
    competitions: pd.DataFrame,
    config_id: str,
) -> None:
    biggest_gain = teams.sort_values("total_elo_change", ascending=False).iloc[0]
    biggest_loss = teams.sort_values("total_elo_change").iloc[0]
    strongest_gd = matches.sort_values("goal_multiplier", ascending=False).iloc[0]
    lines = [
        "# AO European Elo 20 Takimli Sentetik Pilot",
        "",
        f"- Production config id: `{config_id}`",
        f"- Takim: `{len(teams)}`",
        f"- Mac: `{len(matches)}`",
        "- Dagilim: `8 UCL / 6 UEL / 6 UECL`",
        "- Baslangic Elo araligi: `940-1940`",
        "- Progression, Achievement Reserve ve competition K kapali.",
        "",
        "## Ana Bulgular",
        "",
        f"- En buyuk Elo kazanci: {biggest_gain.team_name} "
        f"`{biggest_gain.total_elo_change:+.3f}`.",
        f"- En buyuk Elo kaybi: {biggest_loss.team_name} "
        f"`{biggest_loss.total_elo_change:+.3f}`.",
        f"- En yuksek M_GD: {strongest_gd.match_id} "
        f"`{strongest_gd.goal_multiplier:.6f}`.",
        f"- Maksimum final gol-farki net etkisi: "
        f"`{teams['goal_difference_net_effect_vs_classic'].abs().max():.3f}`.",
        f"- Maksimum power zero-sum hatasi: "
        f"`{matches['total_power_zero_sum_error'].max():.3e}`.",
        "",
        "## Turnuva Korunumu",
        "",
        dataframe_to_markdown(
            competitions[
                [
                    "competition",
                    "teams",
                    "matches",
                    "start_elo_sum",
                    "final_elo_sum",
                    "elo_conservation_error",
                ]
            ]
        ),
        "",
        "Bu veri tamamen sentetiktir; gercek takim gucu veya turnuva sonucu iddiasi "
        "tasimaz. Amac production matematik davranisini kontrollu senaryolarda gostermektir.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows = [
        [str(value).replace("|", "\\|") for value in row]
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


def _validate_inputs(initial: pd.DataFrame, matches: pd.DataFrame) -> None:
    if len(initial) != 20 or initial["team_id"].nunique() != 20:
        raise ValueError("Pilot requires exactly 20 unique teams")
    if set(initial["competition_track"]) != {"UCL", "UEL", "UECL"}:
        raise ValueError("Pilot must include UCL, UEL and UECL teams")
    if initial["ao_first_elo"].min() != 940 or initial["ao_first_elo"].max() != 1940:
        raise ValueError("Pilot starting Elo range must be 940-1940")
    if len(matches) != 33 or matches["match_id"].nunique() != 33:
        raise ValueError("Pilot requires exactly 33 unique matches")
    if set(matches["competition"]) != {"UCL", "UEL", "UECL"}:
        raise ValueError("Pilot matches must cover UCL, UEL and UECL")


def _validate_results(
    teams: pd.DataFrame,
    matches: pd.DataFrame,
    production_updates,
    classic_updates,
    config_id: str,
) -> None:
    if len(production_updates) != 33 or len(classic_updates) != 33:
        raise ValueError("Both replay arms must process 33 matches")
    if not matches["config_id"].eq(config_id).all():
        raise ValueError("Match audit contains a config mismatch")
    if not matches["progression_reserve_added"].eq(0.0).all():
        raise ValueError("Progression reserve must stay disabled")
    if not matches["trophy_reserve_added"].eq(0.0).all():
        raise ValueError("Trophy reserve must stay disabled")
    if matches["goal_multiplier"].min() < 1.0:
        raise ValueError("Goal multiplier cannot fall below one")
    if not matches.loc[matches["goal_difference"].le(1), "goal_multiplier"].eq(1.0).all():
        raise ValueError("Draw and one-goal results must keep multiplier one")
    penalty = matches.loc[matches["decided_on_penalties"]]
    if len(penalty) != 1 or not penalty["goal_multiplier"].eq(1.0).all():
        raise ValueError("Penalty result must keep multiplier one")
    if not matches["goal_multiplier"].gt(1.0).any():
        raise ValueError("Pilot must exercise the active goal-difference layer")
    if matches["total_power_zero_sum_error"].max() > 1e-9:
        raise ValueError("Power updates violated zero-sum conservation")
    if abs(teams["total_elo_change"].sum()) > 1e-9:
        raise ValueError("Final total Elo must equal starting total Elo")
    if abs(teams["goal_difference_net_effect_vs_classic"].sum()) > 1e-9:
        raise ValueError("Goal-difference counterfactual must remain zero-sum")
    numeric = teams.select_dtypes(include="number")
    if not numeric.map(math.isfinite).all().all():
        raise ValueError("Team summary contains a non-finite value")


def _result_label(row: pd.Series) -> str:
    if row["actual_home_score"] == 0.5:
        return "DRAW_PEN" if bool(row["decided_on_penalties"]) else "DRAW"
    return "HOME_WIN" if row["actual_home_score"] == 1.0 else "AWAY_WIN"


def _winner_team_id(row: pd.Series) -> str | None:
    if row["actual_home_score"] == 0.5:
        return None
    return row["home_team_id"] if row["actual_home_score"] == 1.0 else row["away_team_id"]


if __name__ == "__main__":
    main()
