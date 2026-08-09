from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd  # noqa: E402

from ao_elo.opta_league_snapshot import (  # noqa: E402
    compare_snapshot_to_ao_country_strength,
    read_and_validate_snapshot,
)


DEFAULT_SNAPSHOT = ROOT / "data" / "opta_league_strength" / "opta_2026_07_27_snapshot.csv"
DEFAULT_COUNTRY = ROOT / "data" / "backtest_stage_b_2018_2026" / "2025-26" / "country_coefficients.csv"
DEFAULT_OUTPUT = ROOT / "output" / "opta_league_strength_2026_07_27"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a manually frozen Opta league-strength snapshot")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--country-coefficients", type=Path, default=DEFAULT_COUNTRY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    snapshot = read_and_validate_snapshot(args.snapshot.resolve())
    country = pd.read_csv(args.country_coefficients.resolve())
    comparison, summary = compare_snapshot_to_ao_country_strength(snapshot, country)
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output / "opta_vs_ao_league_strength.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary.iloc[0].to_dict(), indent=2) + "\n", encoding="utf-8")
    print(f"Leagues compared: {int(summary.iloc[0]['leagues_compared'])}")
    print(f"Spearman: {float(summary.iloc[0]['spearman_league_average_vs_ao_rank']):.4f}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
