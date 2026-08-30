"""Prospective prediction ledger'ini yazar ve dogrular.

Kullanim:
    python3 scripts/append_prediction_ledger.py record   --served <csv>
    python3 scripts/append_prediction_ledger.py settle   --results <csv>
    python3 scripts/append_prediction_ledger.py verify
    python3 scripts/append_prediction_ledger.py head
"""

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

from ao_elo.prediction_ledger import (  # noqa: E402
    append_predictions,
    append_settlements,
    ledger_head,
    verify_ledger,
)

DEFAULT_LEDGER = ROOT / "data" / "prediction_ledger" / "prediction_ledger_2026_27.jsonl"
DEFAULT_SERVED = (
    ROOT / "output" / "season_2026_27_prediction_features" / "served_predictions.csv"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("record", "settle", "verify", "head"))
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--served", type=Path, default=DEFAULT_SERVED)
    parser.add_argument("--results", type=Path)
    parser.add_argument(
        "--recorded-at",
        help="UTC damgasi; verilmezse su an kullanilir",
    )
    parser.add_argument(
        "--only-new",
        action="store_true",
        help="Ledger'da zaten bulunan match_id'leri atla, kalanini yaz",
    )
    args = parser.parse_args()

    if args.command == "verify":
        report = verify_ledger(args.ledger)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        raise SystemExit(0 if report["valid"] else 1)

    if args.command == "head":
        head = ledger_head(args.ledger)
        print(json.dumps(head.__dict__, indent=2))
        raise SystemExit(0)

    recorded_at = args.recorded_at or pd.Timestamp.now(tz="UTC").isoformat()

    if args.command == "record":
        rows = pd.read_csv(args.served)
        if args.only_new:
            from ao_elo.prediction_ledger import PREDICTION, read_entries

            known = {
                str(e.payload["match_id"])
                for e in read_entries(args.ledger)
                if e.kind == PREDICTION
            }
            before = len(rows)
            rows = rows[~rows["match_id"].astype(str).isin(known)]
            print(f"{before - len(rows)} satir zaten ledger'da, atlandi")
        if rows.empty:
            print("Yazilacak yeni satir yok")
            raise SystemExit(0)
        report = append_predictions(args.ledger, rows, recorded_at=recorded_at)
    else:
        if args.results is None:
            parser.error("settle icin --results gerekli")
        report = append_settlements(
            args.ledger, pd.read_csv(args.results), recorded_at=recorded_at
        )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    head = ledger_head(args.ledger)
    print()
    print("Head hash (bu degeri disarida sabitleyin):")
    print(f"  {head.entry_hash}")
    print(f"  entries={head.entries} predictions={head.predictions} settlements={head.settlements}")


if __name__ == "__main__":
    main()
