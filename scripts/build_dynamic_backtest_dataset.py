from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_backtest_dataset import (  # noqa: E402
    normalize_competition,
    normalize_name,
    parse_html,
    read_cached_text,
)


DATA_ROOT = ROOT / "data" / "backtest_stage_b_2018_2026"
SOURCE_CACHE = ROOT / "data" / "backtest_2018_2026" / "_source_cache"
OUTPUT_ROOT = ROOT / "data" / "dynamic_backtest_2018_2026"
COMPETITION_ORDER = {"UCL": 0, "UEL": 1, "UECL": 2}
NON_KNOCKOUT_ROUNDS = {"Group Stage", "League Stage"}
PENALTY_PREFIX = "Penalty shootout:"
PENALTY_SCORE_RE = re.compile(r"^(\d+)\s*-\s*(\d+)$")


@dataclass(frozen=True)
class PenaltyRecord:
    competition: str
    round_name: str
    left_key: str
    right_key: str
    left_goals: int
    right_goals: int

    @property
    def winner_key(self) -> str:
        return self.left_key if self.left_goals > self.right_goals else self.right_key

    @property
    def lookup_key(self) -> tuple[str, str, frozenset[str]]:
        return self.competition, self.round_name, frozenset((self.left_key, self.right_key))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build chronology-safe match events for dynamic AO Elo calibration"
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--source-cache", type=Path, default=SOURCE_CACHE)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    source_cache = args.source_cache.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    season_folders = sorted(data_root.glob("20??-??"))
    if not season_folders:
        raise ValueError(f"No season folders found in {data_root}")

    for folder in season_folders:
        season = folder.name.replace("-", "/")
        source_path = source_cache / f"match{int(folder.name[:4]) + 1}.html"
        if not source_path.exists():
            raise FileNotFoundError(f"Missing Kassiesa match source: {source_path}")
        source_matches = pd.read_csv(folder / "matches.csv")
        teams = pd.read_csv(folder / "teams.csv")
        penalties = parse_penalty_records(read_cached_text(source_path))
        events, season_audit = build_season_events(source_matches, teams, penalties)
        frames.append(events)
        audit_rows.extend(
            {"season": season, **row}
            for row in season_audit
        )

    all_events = pd.concat(frames, ignore_index=True)
    all_events.to_csv(output_root / "matches.csv", index=False)
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(output_root / "build_audit.csv", index=False)
    write_readme(output_root / "README.md", all_events, audit, data_root)

    failures = audit.loc[audit["status"].eq("FAIL")]
    print("AO dynamic backtest event dataset")
    print(f"Seasons: {all_events['season'].nunique()}")
    print(f"Matches: {len(all_events)}")
    print(f"Ties: {all_events['tie_id'].nunique(dropna=True)}")
    print(f"Penalty shootouts: {int(all_events['decided_on_penalties'].sum())}")
    print(f"Neutral matches: {int(all_events['is_neutral'].sum())}")
    print(f"Audit: {len(audit) - len(failures)}/{len(audit)} passed")
    print(f"Output: {output_root / 'matches.csv'}")
    if not failures.empty:
        raise ValueError(f"Dynamic dataset audit failed with {len(failures)} issue(s)")


def parse_penalty_records(source: str) -> tuple[PenaltyRecord, ...]:
    records: list[PenaltyRecord] = []
    for row in parse_html(source).rows:
        if not row.cells or not row.cells[0].startswith(PENALTY_PREFIX):
            continue
        names = row.cells[0][len(PENALTY_PREFIX):].strip().split(" - ", maxsplit=1)
        score_match = PENALTY_SCORE_RE.match(row.cells[-1].strip())
        if len(names) != 2 or score_match is None:
            raise ValueError(f"Cannot parse penalty row: {row.cells}")
        left_goals, right_goals = map(int, score_match.groups())
        if left_goals == right_goals:
            raise ValueError(f"Penalty shootout cannot finish level: {row.cells}")
        records.append(
            PenaltyRecord(
                competition=normalize_competition(row.cup),
                round_name=row.round_name,
                left_key=normalize_name(names[0]),
                right_key=normalize_name(names[1]),
                left_goals=left_goals,
                right_goals=right_goals,
            )
        )
    keys = [record.lookup_key for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate penalty shootout lookup key in source")
    return tuple(records)


def build_season_events(
    source_matches: pd.DataFrame,
    teams: pd.DataFrame,
    penalties: tuple[PenaltyRecord, ...],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    required = {
        "season", "competition", "round", "home_team_id", "away_team_id",
        "home_team_name", "away_team_name", "home_goals", "away_goals",
    }
    missing = sorted(required - set(source_matches.columns))
    if missing:
        raise ValueError(f"matches.csv missing required columns: {missing}")
    if source_matches.empty:
        raise ValueError("matches.csv cannot be empty")
    season_values = source_matches["season"].drop_duplicates().tolist()
    if len(season_values) != 1:
        raise ValueError(f"Expected one season per file, got {season_values}")
    season = str(season_values[0])

    team_ids = set(teams["team_id"].astype(int))
    match_team_ids = set(source_matches["home_team_id"].astype(int)) | set(
        source_matches["away_team_id"].astype(int)
    )
    if not match_team_ids <= team_ids:
        raise ValueError(f"{season}: matches reference unknown team IDs")

    penalty_lookup = {record.lookup_key: record for record in penalties}
    matched_penalties: set[tuple[str, str, frozenset[str]]] = set()
    round_frames: list[pd.DataFrame] = []
    round_sequence = 0
    for competition in sorted(
        source_matches["competition"].unique(),
        key=lambda value: COMPETITION_ORDER[str(value)],
    ):
        competition_matches = source_matches.loc[
            source_matches["competition"].eq(competition)
        ].copy()
        for round_name in competition_matches["round"].drop_duplicates():
            round_data = competition_matches.loc[
                competition_matches["round"].eq(round_name)
            ].copy()
            if round_name == "Group Stage":
                annotated = annotate_group_stage(round_data)
            elif round_name == "League Stage":
                annotated = annotate_league_stage(round_data)
            else:
                annotated, used = annotate_knockout_round(
                    round_data,
                    penalty_lookup,
                )
                matched_penalties.update(used)
            annotated["round_sequence"] = round_sequence
            round_sequence += 1
            round_frames.append(annotated)

    events = pd.concat(round_frames, ignore_index=True)
    events = events.sort_values(
        ["round_sequence", "chronology_sort", "_source_index"],
        kind="stable",
    ).reset_index(drop=True)
    events["event_order"] = range(1, len(events) + 1)
    season_key = season.replace("/", "")
    events["match_id"] = [
        f"{season_key}-{index:04d}" for index in events["event_order"]
    ]
    events["actual_home_score"] = 0.5
    events.loc[events["home_goals"].gt(events["away_goals"]), "actual_home_score"] = 1.0
    events.loc[events["home_goals"].lt(events["away_goals"]), "actual_home_score"] = 0.0
    events["goal_difference"] = (
        events["home_goals"] - events["away_goals"]
    ).abs()
    events["result_basis"] = "displayed_score_excluding_shootout"

    output_columns = [
        "match_id", "season", "event_order", "competition", "round",
        "round_sequence", "matchday", "tie_id", "leg_number",
        "is_tie_decider", "is_knockout", "is_neutral",
        "chronology_method", "home_team_id", "away_team_id",
        "home_team_name", "away_team_name", "home_goals", "away_goals",
        "actual_home_score", "goal_difference", "result_basis",
        "decided_on_penalties", "home_penalty_goals", "away_penalty_goals",
        "advanced_team_id",
    ]
    events = events[output_columns]
    audit = audit_season(
        source_matches,
        events,
        len(penalties),
        len(matched_penalties),
    )
    return events, audit


def annotate_group_stage(data: pd.DataFrame) -> pd.DataFrame:
    data = data.reset_index(names="_source_index")
    if len(data) % 12 != 0:
        raise ValueError(f"Group Stage row count must be divisible by 12, got {len(data)}")
    rows: list[dict[str, object]] = []
    second_leg_matchday = (5, 5, 6, 6, 4, 4)
    for group_index, start in enumerate(range(0, len(data), 12), start=1):
        block = data.iloc[start:start + 12]
        participants = set(block["home_team_id"]) | set(block["away_team_id"])
        if len(participants) != 4:
            raise ValueError(
                f"Group Stage block {group_index} has {len(participants)} teams, expected 4"
            )
        for local_pair in range(6):
            first = block.iloc[local_pair * 2]
            second = block.iloc[local_pair * 2 + 1]
            if not is_reverse_fixture(first, second):
                raise ValueError(
                    f"Group Stage block {group_index}, pair {local_pair + 1} is not reversible"
                )
            rows.append(
                standard_event(
                    first,
                    matchday=(local_pair // 2) + 1,
                    chronology_sort=((local_pair // 2) + 1) * 100 + group_index,
                    chronology_method="group_matchday_reconstruction",
                    is_knockout=False,
                )
            )
            rows.append(
                standard_event(
                    second,
                    matchday=second_leg_matchday[local_pair],
                    chronology_sort=second_leg_matchday[local_pair] * 100 + group_index,
                    chronology_method="group_matchday_reconstruction",
                    is_knockout=False,
                )
            )
    return pd.DataFrame(rows)


def annotate_league_stage(data: pd.DataFrame) -> pd.DataFrame:
    data = data.reset_index(names="_source_index")
    participants = set(data["home_team_id"]) | set(data["away_team_id"])
    if len(participants) % 2:
        raise ValueError("League Stage participant count must be even")
    matches_per_day = len(participants) // 2
    if len(data) % matches_per_day:
        raise ValueError("League Stage rows do not form complete matchday blocks")
    rows: list[dict[str, object]] = []
    for start in range(0, len(data), matches_per_day):
        block = data.iloc[start:start + matches_per_day]
        matchday = (start // matches_per_day) + 1
        ids = list(block["home_team_id"]) + list(block["away_team_id"])
        if len(ids) != len(set(ids)) or set(ids) != participants:
            raise ValueError(f"League Stage matchday {matchday} does not cover each team once")
        for local_index, (_, row) in enumerate(block.iterrows()):
            rows.append(
                standard_event(
                    row,
                    matchday=matchday,
                    chronology_sort=matchday * 100 + local_index,
                    chronology_method="league_stage_source_block",
                    is_knockout=False,
                )
            )
    return pd.DataFrame(rows)


def annotate_knockout_round(
    data: pd.DataFrame,
    penalty_lookup: dict[tuple[str, str, frozenset[str]], PenaltyRecord],
) -> tuple[pd.DataFrame, set[tuple[str, str, frozenset[str]]]]:
    data = data.reset_index(names="_source_index")
    season = str(data.iloc[0]["season"])
    competition = str(data.iloc[0]["competition"])
    round_name = str(data.iloc[0]["round"])
    rows: list[dict[str, object]] = []
    matched_penalties: set[tuple[str, str, frozenset[str]]] = set()
    tie_number = 0
    index = 0
    while index < len(data):
        tie_number += 1
        first = data.iloc[index]
        tie_rows = [first]
        if index + 1 < len(data) and is_reverse_fixture(first, data.iloc[index + 1]):
            tie_rows.append(data.iloc[index + 1])
            index += 2
        else:
            index += 1
        tie_id = (
            f"{season.replace('/', '')}-{competition}-"
            f"{slug(round_name)}-{tie_number:03d}"
        )
        team_key = frozenset(
            (normalize_name(first["home_team_name"]), normalize_name(first["away_team_name"]))
        )
        lookup_key = (competition, round_name, team_key)
        penalty = penalty_lookup.get(lookup_key)
        if penalty is not None:
            matched_penalties.add(lookup_key)
        winner_id = infer_tie_winner(tie_rows, penalty)
        neutral = is_neutral_match(season, round_name)
        for leg_index, row in enumerate(tie_rows, start=1):
            is_decider = leg_index == len(tie_rows)
            event = standard_event(
                row,
                matchday=pd.NA,
                chronology_sort=tie_number * 10 + leg_index,
                chronology_method="source_round_leg_order",
                is_knockout=True,
            )
            event.update(
                {
                    "tie_id": tie_id,
                    "leg_number": leg_index,
                    "is_tie_decider": is_decider,
                    "is_neutral": neutral,
                    "decided_on_penalties": bool(is_decider and penalty is not None),
                    "advanced_team_id": winner_id if is_decider else pd.NA,
                }
            )
            if is_decider and penalty is not None:
                home_key = normalize_name(row["home_team_name"])
                if home_key == penalty.left_key:
                    event["home_penalty_goals"] = penalty.left_goals
                    event["away_penalty_goals"] = penalty.right_goals
                else:
                    event["home_penalty_goals"] = penalty.right_goals
                    event["away_penalty_goals"] = penalty.left_goals
            rows.append(event)
    return pd.DataFrame(rows), matched_penalties


def standard_event(
    row: pd.Series,
    *,
    matchday: int | pd._libs.missing.NAType,
    chronology_sort: int,
    chronology_method: str,
    is_knockout: bool,
) -> dict[str, object]:
    event = row.to_dict()
    event.update(
        {
            "matchday": matchday,
            "chronology_sort": chronology_sort,
            "chronology_method": chronology_method,
            "tie_id": pd.NA,
            "leg_number": pd.NA,
            "is_tie_decider": False,
            "is_knockout": is_knockout,
            "is_neutral": False,
            "decided_on_penalties": False,
            "home_penalty_goals": pd.NA,
            "away_penalty_goals": pd.NA,
            "advanced_team_id": pd.NA,
        }
    )
    return event


def infer_tie_winner(
    tie_rows: list[pd.Series],
    penalty: PenaltyRecord | None,
) -> int:
    first = tie_rows[0]
    home_id = int(first["home_team_id"])
    away_id = int(first["away_team_id"])
    if penalty is not None:
        home_key = normalize_name(first["home_team_name"])
        return home_id if penalty.winner_key == home_key else away_id

    home_total = int(first["home_goals"])
    away_total = int(first["away_goals"])
    if len(tie_rows) == 2:
        second = tie_rows[1]
        home_total += int(second["away_goals"])
        away_total += int(second["home_goals"])
    if home_total != away_total:
        return home_id if home_total > away_total else away_id
    if len(tie_rows) == 2:
        home_away_goals = int(tie_rows[1]["away_goals"])
        away_away_goals = int(first["away_goals"])
        if home_away_goals != away_away_goals:
            return home_id if home_away_goals > away_away_goals else away_id
    raise ValueError(
        f"Cannot infer tie winner for {first['season']} {first['competition']} "
        f"{first['round']}: {first['home_team_name']} - {first['away_team_name']}"
    )


def is_reverse_fixture(first: pd.Series, second: pd.Series) -> bool:
    return (
        int(first["home_team_id"]) == int(second["away_team_id"])
        and int(first["away_team_id"]) == int(second["home_team_id"])
    )


def is_neutral_match(season: str, round_name: str) -> bool:
    if round_name == "Final":
        return True
    return season == "2019/20" and round_name in {"Quarter Finals", "Semi Finals"}


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def audit_season(
    source: pd.DataFrame,
    events: pd.DataFrame,
    penalty_count: int,
    matched_penalty_count: int,
) -> list[dict[str, object]]:
    def row(check: str, passed: bool, detail: object) -> dict[str, object]:
        return {"check": check, "status": "PASS" if passed else "FAIL", "detail": detail}

    source_scores = source[
        ["competition", "round", "home_team_id", "away_team_id", "home_goals", "away_goals"]
    ].value_counts().sort_index()
    event_scores = events[
        ["competition", "round", "home_team_id", "away_team_id", "home_goals", "away_goals"]
    ].value_counts().sort_index()
    group_valid = True
    for (_, _, matchday), group in events.loc[
        events["round"].isin(NON_KNOCKOUT_ROUNDS)
    ].groupby(["competition", "round", "matchday"]):
        ids = list(group["home_team_id"]) + list(group["away_team_id"])
        group_valid &= len(ids) == len(set(ids))
    tie_deciders = events.loc[events["is_tie_decider"]]
    return [
        row("row_count_preserved", len(source) == len(events), f"{len(events)}/{len(source)}"),
        row("score_rows_preserved", source_scores.equals(event_scores), "exact multiset match"),
        row("match_id_unique", events["match_id"].is_unique, len(events)),
        row("event_order_unique", events["event_order"].is_unique, len(events)),
        row("group_matchday_team_uniqueness", group_valid, "one match per team per matchday"),
        row(
            "penalty_rows_matched",
            penalty_count == matched_penalty_count,
            f"{matched_penalty_count}/{penalty_count}",
        ),
        row(
            "knockout_winner_complete",
            tie_deciders["advanced_team_id"].notna().all(),
            f"{tie_deciders['advanced_team_id'].notna().sum()}/{len(tie_deciders)}",
        ),
        row(
            "finals_neutral",
            events.loc[events["round"].eq("Final"), "is_neutral"].all(),
            int(events.loc[events["round"].eq("Final"), "is_neutral"].sum()),
        ),
        row(
            "actual_score_valid",
            events["actual_home_score"].isin((0.0, 0.5, 1.0)).all(),
            sorted(events["actual_home_score"].unique()),
        ),
    ]


def write_readme(
    path: Path,
    events: pd.DataFrame,
    audit: pd.DataFrame,
    source_root: Path,
) -> None:
    season_summary = (
        events.groupby("season")
        .agg(matches=("match_id", "size"), ties=("tie_id", "nunique"), penalties=("decided_on_penalties", "sum"))
        .reset_index()
    )
    table_rows = [
        f"| {row.season} | {row.matches} | {row.ties} | {int(row.penalties)} |"
        for row in season_summary.itertuples(index=False)
    ]
    text = "\n".join(
        [
            "# AO Dynamic Elo Calibration Events",
            "",
            "This is a research dataset for chronological dynamic-rating calibration.",
            f"Its score and team universe is preserved exactly from `{source_root}`.",
            "",
            "The key grain is one played match (`match_id`). `event_order` is the",
            "chronological order required by the Elo update. Exact calendar timestamps are",
            "not claimed: knockout legs use source round/leg order, group stages are rebuilt",
            "to the six official matchdays, and league stages use validated source blocks.",
            "This is sufficient because updates for matches involving disjoint teams commute.",
            "",
            "Penalty shootouts do not change the match score: a level match remains 0.5.",
            "The advancing club is stored separately in `advanced_team_id` for the later",
            "progression calibration layer. Displayed goals may include extra time; exact",
            "90/120-minute splits are intentionally not asserted in this first dataset.",
            "",
            "| Season | Matches | Knockout ties | Shootouts |",
            "| --- | ---: | ---: | ---: |",
            *table_rows,
            "",
            f"Build audit: {int(audit['status'].eq('PASS').sum())}/{len(audit)} checks passed.",
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
