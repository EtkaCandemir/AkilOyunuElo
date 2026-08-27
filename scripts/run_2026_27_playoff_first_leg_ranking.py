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

from ao_elo.config import AOEuropeanEloConfig  # noqa: E402
from ao_elo.dynamic_csv import (  # noqa: E402
    load_selected_v2_config,
    run_batch,
    state_to_frame,
    updates_to_frame,
)
from ao_elo.pipeline import compute_ao_first_elo_from_csv  # noqa: E402


DEFAULT_DATA_ROOT = ROOT / "data" / "season_2026_27_preproduction"
DEFAULT_PLAYOFF_RESULTS = (
    ROOT
    / "output"
    / "playoff_first_legs_2026_27"
    / "match_predictions_vs_results.csv"
)
DEFAULT_PLAYOFF_FIXTURES = (
    ROOT / "output" / "playoff_first_legs_2026_27" / "fixtures_played.csv"
)
DEFAULT_OUTPUT_ROOT = ROOT / "output" / "season_2026_27_playoff_first_legs"
DEFAULT_CONTRACT = ROOT / "contracts" / "ao_european_elo_v2_production.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay 2026/27 Q1-Q3 plus completed qualifying play-off first legs "
            "to produce an AO Live Elo ranking."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--playoff-fixtures", type=Path, default=DEFAULT_PLAYOFF_FIXTURES)
    parser.add_argument("--playoff-results", type=Path, default=DEFAULT_PLAYOFF_RESULTS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    return parser.parse_args()


def build_playoff_matches(fixtures_path: Path, results_path: Path) -> pd.DataFrame:
    fixtures = pd.read_csv(fixtures_path)
    results = pd.read_csv(results_path)
    required_fixture = {
        "match_id",
        "season",
        "kickoff_utc",
        "competition",
        "round",
        "tie_id",
        "is_knockout",
        "is_tie_decider",
        "is_single_match_tie",
        "stage",
        "home_team_id",
        "away_team_id",
        "is_neutral",
    }
    required_results = {"match_id", "home_goals", "away_goals"}
    if missing := required_fixture.difference(fixtures.columns):
        raise ValueError(f"Play-off fixtures are missing columns: {sorted(missing)}")
    if missing := required_results.difference(results.columns):
        raise ValueError(f"Play-off results are missing columns: {sorted(missing)}")
    if fixtures["match_id"].duplicated().any() or results["match_id"].duplicated().any():
        raise ValueError("Play-off inputs must have unique match_id values")

    merged = fixtures.merge(
        results[["match_id", "home_goals", "away_goals"]],
        on="match_id",
        how="left",
        validate="one_to_one",
    )
    if merged[["home_goals", "away_goals"]].isna().any().any():
        missing_ids = merged.loc[
            merged["home_goals"].isna() | merged["away_goals"].isna(), "match_id"
        ].tolist()
        raise ValueError(f"Missing official result for play-off matches: {missing_ids}")

    # These are first legs. No progression event is settled and no verified 90/120 xG
    # snapshot is available in this input, so the production goal-margin fallback applies.
    merged["decided_on_penalties"] = False
    merged["advanced_team_id"] = ""
    merged["xg_home"] = ""
    merged["xg_away"] = ""
    merged["xg_analysis_eligible"] = False
    merged["xg_fallback"] = "GOAL_MARGIN_ONLY"
    return merged.sort_values(["kickoff_utc", "match_id"], kind="stable").reset_index(
        drop=True
    )


def build_ranking(
    initial: pd.DataFrame,
    before_playoff: pd.DataFrame,
    after_playoff: pd.DataFrame,
    playoff_updates: pd.DataFrame,
) -> pd.DataFrame:
    before = before_playoff[["team_id", "ao_live_elo", "q3_end_rank"]].rename(
        columns={
            "ao_live_elo": "pre_playoff_live_elo",
            "q3_end_rank": "pre_playoff_rank",
        }
    )
    after = after_playoff.copy()
    after["after_first_leg_rank"] = after["ao_live_elo"].rank(
        method="min", ascending=False
    ).astype(int)
    played_counts = pd.concat(
        [playoff_updates["home_team_id"], playoff_updates["away_team_id"]],
        ignore_index=True,
    ).value_counts()
    ranking = (
        after.merge(
            initial[["team_id", "team_name", "country", "country_code", "ao_first_elo"]],
            on=["team_id", "team_name", "ao_first_elo"],
            how="left",
            validate="one_to_one",
        )
        .merge(before, on="team_id", how="left", validate="one_to_one")
        .sort_values(["after_first_leg_rank", "team_id"], kind="stable")
        .reset_index(drop=True)
    )
    ranking["playoff_first_leg_matches"] = ranking["team_id"].map(played_counts).fillna(0).astype(int)
    ranking["playoff_first_leg_elo_change"] = (
        ranking["ao_live_elo"] - ranking["pre_playoff_live_elo"]
    )
    ranking["playoff_first_leg_rank_change"] = (
        ranking["pre_playoff_rank"] - ranking["after_first_leg_rank"]
    )
    ranking["season_elo_change"] = ranking["ao_live_elo"] - ranking["ao_first_elo"]
    ranking["season_rank_change"] = (
        ranking["ao_first_elo"].rank(method="min", ascending=False).astype(int)
        - ranking["after_first_leg_rank"]
    )
    columns = [
        "after_first_leg_rank",
        "team_id",
        "team_name",
        "country_code",
        "ao_first_elo",
        "pre_playoff_live_elo",
        "pre_playoff_rank",
        "playoff_first_leg_matches",
        "playoff_first_leg_elo_change",
        "playoff_first_leg_rank_change",
        "ao_live_elo",
        "season_elo_change",
        "season_rank_change",
        "power_elo",
        "progression_bonus_total",
        "last_event_utc",
        "last_match_id",
        "model_version",
        "config_id",
    ]
    return ranking[columns]


def write_report(
    ranking: pd.DataFrame,
    playoff_updates: pd.DataFrame,
    output_root: Path,
) -> None:
    moved = ranking.loc[ranking["playoff_first_leg_matches"].gt(0)].copy()
    largest_gains = moved.nlargest(15, "playoff_first_leg_elo_change")
    largest_losses = moved.nsmallest(15, "playoff_first_leg_elo_change")
    report_columns = [
        "team_name",
        "pre_playoff_rank",
        "after_first_leg_rank",
        "playoff_first_leg_elo_change",
        "ao_live_elo",
    ]
    lines = [
        "# 2026/27 Qualifying Play-off İlk Maçları Sonrası AO Live Elo",
        "",
        "Bu rapor, Q1-Q3 replay'i ile 43 tamamlanmış qualifying play-off ilk maçını exact UTC kronolojisinde birleştirir.",
        "ML/Poisson tahmin katmanı rating state'ine geri beslenmez; sıralama yalnız AO Live Elo güncellemelerinden oluşur.",
        "",
        "## Kapsam",
        "",
        f"- Önceki tamamlanmış ön eleme maçı: {len(playoff_updates) - 43}",
        "- Eklenen play-off ilk maçı: 43",
        f"- Güncellenen takım sayısı: {len(moved)}",
        f"- Ortalama mutlak ilk-maç Elo hareketi: {moved['playoff_first_leg_elo_change'].abs().mean():.3f}",
        f"- En büyük ilk-maç kazancı: {moved['playoff_first_leg_elo_change'].max():.3f}",
        f"- En büyük ilk-maç kaybı: {moved['playoff_first_leg_elo_change'].min():.3f}",
        "- İlk maçlar tie decider olmadığı için progression bonusu uygulanmadı.",
        "- Bu 43 maçta doğrulanmış xG inputu olmadığından, production sözleşmesindeki goal-margin fallback kullanıldı.",
        "",
        "## En Çok Yükselenler",
        "",
        markdown_table(largest_gains[report_columns]),
        "",
        "## En Çok Düşenler",
        "",
        markdown_table(largest_losses[report_columns]),
        "",
        "Tam takım tablosu `team_ranking_after_playoff_first_legs.csv` dosyasındadır.",
    ]
    (output_root / "ranking_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact deterministic Markdown table without an optional dependency."""
    headers = [str(column) for column in frame.columns]
    body = []
    for row in frame.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append(f"{value:.2f}")
            else:
                cells.append(str(value))
        body.append(cells)
    widths = [len(header) for header in headers]
    for row in body:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]

    def render(values: list[str]) -> str:
        return "| " + " | ".join(
            value.ljust(width) for value, width in zip(values, widths)
        ) + " |"

    return "\n".join(
        [
            render(headers),
            "| " + " | ".join("-" * width for width in widths) + " |",
            *(render(row) for row in body),
        ]
    )


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    replay_root = output_root / "replay"
    output_root.mkdir(parents=True, exist_ok=True)

    initial_path = output_root / "ao_first_elo_2026_27.csv"
    compute_ao_first_elo_from_csv(
        data_root / "teams.csv",
        data_root / "country_coefficients.csv",
        data_root / "domestic_context.csv",
        data_root / "club_european_points.csv",
        AOEuropeanEloConfig.active(),
        initial_path,
    )
    initial = pd.read_csv(initial_path)
    initial["ao_first_rank"] = initial["ao_first_elo"].rank(
        method="min", ascending=False
    ).astype(int)

    q1_q3_path = data_root / "matches_completed.csv"
    q1_q3 = pd.read_csv(q1_q3_path)
    playoff = build_playoff_matches(args.playoff_fixtures.resolve(), args.playoff_results.resolve())
    overlap = set(q1_q3["match_id"]).intersection(playoff["match_id"])
    if overlap:
        raise ValueError(f"Play-off matches already exist in completed input: {sorted(overlap)}")
    combined = pd.concat([q1_q3, playoff], ignore_index=True, sort=False).sort_values(
        ["kickoff_utc", "match_id"], kind="stable"
    )
    combined.to_csv(output_root / "replay_matches_q1_to_playoff_first_legs.csv", index=False, lineterminator="\n")

    config = load_selected_v2_config(args.contract.resolve())
    final_state, all_updates = run_batch(initial_path, output_root / "replay_matches_q1_to_playoff_first_legs.csv", replay_root, config)
    all_updates_frame = updates_to_frame(all_updates)
    playoff_updates = all_updates_frame.loc[
        all_updates_frame["match_id"].isin(set(playoff["match_id"]))
    ].copy()
    if len(playoff_updates) != len(playoff):
        raise AssertionError("Every play-off first leg must be settled exactly once")
    if not (playoff_updates["home_progression_bonus_delta"].abs() < 1e-12).all():
        raise AssertionError("A non-deciding first leg cannot apply a progression bonus")

    q3_state, _ = run_batch(
        initial_path,
        q1_q3_path,
        output_root / "q1_q3_reference_replay",
        config,
    )
    before = state_to_frame(q3_state)
    before["q3_end_rank"] = before["ao_live_elo"].rank(method="min", ascending=False).astype(int)
    after = state_to_frame(final_state)
    ranking = build_ranking(initial, before, after, playoff_updates)
    ranking.to_csv(output_root / "team_ranking_after_playoff_first_legs.csv", index=False, lineterminator="\n")
    playoff_updates.to_csv(output_root / "playoff_first_leg_match_updates.csv", index=False, lineterminator="\n")
    write_report(ranking, all_updates_frame, output_root)
    manifest = {
        "scope": "2026/27 Q1-Q3 plus completed qualifying play-off first legs",
        "q1_q3_matches": int(len(q1_q3)),
        "playoff_first_leg_matches": int(len(playoff)),
        "total_matches": int(len(combined)),
        "teams": int(len(ranking)),
        "model_version": config.model_version,
        "config_id": config.config_id,
        "rating_feedback_from_prediction_layer": False,
        "playoff_xg_behavior": "GOAL_MARGIN_ONLY_FALLBACK",
    }
    (output_root / "replay_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(output_root / "team_ranking_after_playoff_first_legs.csv")


if __name__ == "__main__":
    main()
