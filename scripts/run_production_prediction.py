from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ao_elo.production_prediction import (  # noqa: E402
    ProductionPredictionService,
)


DEFAULT_CONTRACT = ROOT / "contracts" / "ao_european_elo_v2_production.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Produce locked AO ML + Domestic Poisson pre-match probabilities"
    )
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generated-at-utc", required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--strict-artifacts",
        action="store_true",
        help="Fail instead of using CURRENT_AO_1X2 when artifacts cannot load",
    )
    args = parser.parse_args()

    features = pd.read_csv(args.features, low_memory=False)
    service = ProductionPredictionService.from_contract(
        args.contract,
        repository_root=ROOT,
        allow_degraded_fallback=not args.strict_artifacts,
    )
    predictions = service.predict(
        features,
        generated_at_utc=pd.Timestamp(args.generated_at_utc),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False, lineterminator="\n")

    counts = predictions["prediction_status"].value_counts().to_dict()
    print(f"Rows: {len(predictions)}")
    print(f"Status: {counts}")
    print(f"Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
