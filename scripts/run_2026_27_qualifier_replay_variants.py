from __future__ import annotations

"""Replay the completed 2026/27 qualifiers from two different season-start seeds.

The question is what today's ratings look like after the 428 matches between the
1st qualifying round and the qualifying play-off, starting from:

    URETIM      ao_first_elo, the blended seed production actually serves
    HARMANSIZ   the European Prior before the 0.65 blend is applied, with the
                Domestic Prior standing in for the clubs that have no European
                evidence at all - which is what the domestic side is there for

`read_team_seeds` reads only `season`, `team_id`, `team_name` and
`ao_first_elo`, and `initialize_season` fills both `power_elo` and
`ao_first_elo` from that single column, so an alternative seed has to be written
into `ao_first_elo` of a staged CSV. `run_positive_bridge_shadow_2026_27.py`
established that pattern and this command follows it.

Research output, not a candidate: no backtest gate is run, nothing is proposed
for production, and the active contract is hashed before and after to prove it
was not touched.
"""

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ao_elo.dynamic_csv import (  # noqa: E402
    load_selected_v2_config,
    run_batch,
    state_to_frame,
)


DEFAULT_SEED = (
    ROOT / "output" / "season_2026_27_preproduction" / "ao_first_elo_2026_27.csv"
)
DEFAULT_MATCHES = (
    ROOT / "data" / "season_2026_27_preproduction" / "matches_completed.csv"
)
DEFAULT_FIXTURES = (
    ROOT / "data" / "season_2026_27_preproduction" / "fixtures_upcoming.csv"
)
DEFAULT_CONTRACT = ROOT / "contracts" / "ao_european_elo_v2_production.json"
REFERENCE_RATINGS = (
    ROOT
    / "output"
    / "season_2026_27_preproduction"
    / "playoff_pre_match_team_ratings.csv"
)
OUTPUT_ROOT = ROOT / "output" / "season_2026_27_qualifier_replay_variants"

EXPECTED_MATCHES = 428
EXPECTED_ROSTER = 108
TOLERANCE = 1e-9

VARIANTS = (
    ("uretim_0.65_harmanli", "URETIM"),
    ("harmansiz_european_prior", "HARMANSIZ"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def unblended_seed(production: pd.DataFrame) -> pd.DataFrame:
    """Replace the blended seed with the European Prior that produced it.

    A club with no European evidence keeps its Domestic Prior. Its European
    Prior is the 500 base placeholder rather than a rating, so carrying it into
    the replay would start ten clubs a thousand points below where any evidence
    puts them. Production already treats them this way - their exposure is 0, so
    the blend returns the domestic side untouched.
    """
    required = {
        "european_prior",
        "adjusted_domestic_prior",
        "european_exposure",
        "ao_first_elo",
        "ao_first_elo_rank",
    }
    missing = sorted(required.difference(production.columns))
    if missing:
        raise ValueError(f"Production seed is missing columns {missing}")

    frame = production.copy()
    has_evidence = frame["european_exposure"].astype(float) > 0.0
    frame["production_ao_first_elo"] = frame["ao_first_elo"].astype(float)
    frame["production_ao_first_elo_rank"] = frame["ao_first_elo_rank"]
    frame["ao_first_elo"] = frame["european_prior"].astype(float).where(
        has_evidence, frame["adjusted_domestic_prior"].astype(float)
    )
    # The rank column is not read by the replay but it is read by the shared
    # summary helpers, and leaving the production ranking on a different rating
    # would make any later join quietly wrong.
    frame["ao_first_elo_rank"] = (
        frame["ao_first_elo"].rank(method="min", ascending=False).astype(int)
    )
    return frame


def check_unblended_seed(frame: pd.DataFrame) -> None:
    has_evidence = frame["european_exposure"].astype(float) > 0.0
    expected = frame["european_prior"].astype(float).where(
        has_evidence, frame["adjusted_domestic_prior"].astype(float)
    )
    error = (frame["ao_first_elo"].astype(float) - expected).abs().max()
    if float(error) > TOLERANCE:
        raise ValueError(f"Unblended seed does not match its definition: {error}")


def league_stage_roster(fixtures_csv: Path) -> set[str]:
    fixtures = pd.read_csv(fixtures_csv)
    roster = set(fixtures["home_team_id"].astype(str)) | set(
        fixtures["away_team_id"].astype(str)
    )
    if len(roster) != EXPECTED_ROSTER:
        raise ValueError(
            f"League-stage roster is {len(roster)} clubs, expected {EXPECTED_ROSTER}"
        )
    return roster


def check_zero_sum(replay_dir: Path) -> float:
    """Every match must move the two sides by equal and opposite amounts.

    Qualifier rounds award no progression bonus, so the power total is conserved
    exactly. A violation would mean the replay is inventing or destroying rating.
    """
    updates = pd.read_csv(replay_dir / "match_updates.csv")
    drift = (
        updates["home_power_post"]
        - updates["home_power_pre"]
        + updates["away_power_post"]
        - updates["away_power_pre"]
    ).abs()
    worst = float(drift.max())
    if worst > 1e-6:
        raise ValueError(f"Power is not conserved; worst match drifts {worst}")
    return worst


def replay(
    seed_frame: pd.DataFrame,
    matches_csv: Path,
    variant_root: Path,
    config,
) -> pd.DataFrame:
    if variant_root.exists():
        shutil.rmtree(variant_root)
    variant_root.mkdir(parents=True)
    seed_path = variant_root / "seed.csv"
    seed_frame.to_csv(seed_path, index=False, lineterminator="\n")

    replay_dir = variant_root / "qualifier_replay"
    state, updates = run_batch(seed_path, matches_csv, replay_dir, config)
    if len(updates) != EXPECTED_MATCHES:
        raise ValueError(
            f"Replayed {len(updates)} matches, expected {EXPECTED_MATCHES}"
        )
    check_zero_sum(replay_dir)
    return state_to_frame(state)


def build_table(
    seed_frame: pd.DataFrame, final: pd.DataFrame, roster: set[str]
) -> pd.DataFrame:
    start = seed_frame[["team_id", "team_name", "ao_first_elo"]].copy()
    start["team_id"] = start["team_id"].astype(str)
    end = final[["team_id", "ao_live_elo"]].copy()
    end["team_id"] = end["team_id"].astype(str)

    table = start.merge(end, on="team_id", how="inner", validate="one_to_one")
    table = table.loc[table["team_id"].isin(roster)].copy()
    if len(table) != EXPECTED_ROSTER:
        raise ValueError(f"Table has {len(table)} rows, expected {EXPECTED_ROSTER}")

    table = table.rename(
        columns={
            "team_name": "kulup",
            "ao_first_elo": "baslangic_elo",
            "ao_live_elo": "mac_sonrasi_elo",
        }
    )
    table["degisim"] = table["mac_sonrasi_elo"] - table["baslangic_elo"]
    table = table.sort_values("mac_sonrasi_elo", ascending=False)
    return table[["kulup", "baslangic_elo", "mac_sonrasi_elo", "degisim"]].round(2)


def check_against_reference(table: pd.DataFrame, seed_frame: pd.DataFrame) -> float:
    """The production variant must reproduce the existing replay exactly.

    If it does not, the harness is wired differently from the frozen run and the
    unblended numbers cannot be trusted either.
    """
    reference = pd.read_csv(REFERENCE_RATINGS)
    merged = table.merge(
        reference[["team_name", "pre_playoff_live_elo"]],
        left_on="kulup",
        right_on="team_name",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != EXPECTED_ROSTER:
        raise ValueError(
            f"Only {len(merged)} of {EXPECTED_ROSTER} clubs matched the reference"
        )
    error = (merged["mac_sonrasi_elo"] - merged["pre_playoff_live_elo"]).abs().max()
    # The table is rounded to two decimals for reading, so compare at that scale.
    if float(error) > 0.005:
        raise ValueError(f"Production variant diverges from the frozen replay: {error}")
    return float(error)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay the 2026/27 qualifiers from two season-start seeds"
    )
    parser.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    parser.add_argument("--matches", type=Path, default=DEFAULT_MATCHES)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    contract_hash = sha256(args.contract)
    config = load_selected_v2_config(args.contract)
    roster = league_stage_roster(args.fixtures)
    production_seed = pd.read_csv(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)

    seeds = {
        "URETIM": production_seed.copy(),
        "HARMANSIZ": unblended_seed(production_seed),
    }
    check_unblended_seed(seeds["HARMANSIZ"])

    for filename, variant in VARIANTS:
        print(f"Replaying {variant}", flush=True)
        seed_frame = seeds[variant]
        final = replay(
            seed_frame, args.matches, args.output_root / filename, config
        )
        table = build_table(seed_frame, final, roster)
        if variant == "URETIM":
            error = check_against_reference(table, seed_frame)
            print(f"  frozen replay reproduced, max difference {error:.6f}")
        destination = args.output_root / f"{filename}.csv"
        table.to_csv(destination, index=False, lineterminator="\n")
        print(f"  {destination}")

    if sha256(args.contract) != contract_hash:
        raise ValueError("Production contract changed during the replay")
    print("Contract unchanged")


if __name__ == "__main__":
    main()
