from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import joblib
import pandas as pd
import scipy
import sklearn


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ao_elo.domestic_poisson import (  # noqa: E402
    build_domestic_club_mapping,
    replay_domestic_poisson_state,
)
from ao_elo.ml_prediction import TrainedML1X2  # noqa: E402
from ao_elo.production_prediction import (  # noqa: E402
    PRODUCTION_PREDICTION_FAMILY,
    PRODUCTION_PREDICTION_MODEL_VERSION,
    PRODUCTION_PREDICTION_STATUS,
    selected_production_prediction_config,
)


SOURCE_ML_ARTIFACT = (
    ROOT / "output" / "ml_1x2_backtest_2018_2026" / "selected_ml_pipeline.joblib"
)
ENSEMBLE_DECISION = (
    ROOT
    / "output"
    / "final_prediction_ensemble_backtest_2018_2026"
    / "selected_candidate.json"
)
DOMESTIC_MATCHES = (
    ROOT / "data" / "domestic_league_matches_2013_2026" / "domestic_matches.csv"
)
DOMESTIC_BRIDGE = (
    ROOT
    / "data"
    / "domestic_league_matches_2013_2026"
    / "domestic_team_bridge.csv"
)
OUTPUT_ROOT = ROOT / "artifacts" / "production_prediction"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build frozen ML and Domestic Poisson production artifacts"
    )
    parser.add_argument("--source-ml-artifact", type=Path, default=SOURCE_ML_ARTIFACT)
    parser.add_argument("--ensemble-decision", type=Path, default=ENSEMBLE_DECISION)
    parser.add_argument("--domestic-matches", type=Path, default=DOMESTIC_MATCHES)
    parser.add_argument("--domestic-bridge", type=Path, default=DOMESTIC_BRIDGE)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    source_ml = args.source_ml_artifact.resolve()
    decision_path = args.ensemble_decision.resolve()
    domestic_path = args.domestic_matches.resolve()
    bridge_path = args.domestic_bridge.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)

    config = selected_production_prediction_config()
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    prospective = decision.get("prospective_2026_27_selection")
    if not isinstance(prospective, dict):
        raise ValueError("Ensemble decision lacks prospective selection")
    expected_selection = {
        "poisson_source": "AO_POISSON_RHO0_CONTROL",
        "poisson_weight": 0.5,
        "ml_weight": 0.5,
    }
    for key, expected in expected_selection.items():
        if prospective.get(key) != expected:
            raise ValueError(f"Unexpected prospective ensemble selection: {key}")

    source_package = joblib.load(source_ml)
    if not isinstance(source_package, dict):
        raise ValueError("Selected ML source artifact is invalid")
    model = source_package.get("model")
    if not isinstance(model, TrainedML1X2):
        raise ValueError("Selected ML source does not contain TrainedML1X2")
    if model.arm_name != "STRUCTURAL_LOGISTIC" or model.family != "LOGISTIC":
        raise ValueError("Selected ML source is not the frozen structural logistic")
    if float(source_package.get("blend_weight")) != 0.9:
        raise ValueError("Selected current ML blend weight must equal 0.9")
    feature_schema = {
        "numeric": list(model.schema.numeric),
        "categorical": list(model.schema.categorical),
    }
    feature_schema_sha = _payload_sha256(feature_schema)

    ml_output = output / "structural_logistic_v1.joblib"
    joblib.dump(
        {
            "artifact_version": "ao-structural-logistic-v1-production",
            "production_status": PRODUCTION_PREDICTION_STATUS,
            "model": model,
            "model_fingerprint": model.fingerprint,
            "feature_schema": feature_schema,
            "feature_schema_sha256": feature_schema_sha,
            "ao_blend_weight": 0.9,
            "source_artifact_sha256": _sha256(source_ml),
            "training_window": "2018/19-2025/26",
            "rating_feedback": False,
        },
        ml_output,
        compress=3,
    )

    domestic = pd.read_csv(domestic_path, low_memory=False)
    bridge = pd.read_csv(bridge_path, low_memory=False)
    engine = replay_domestic_poisson_state(domestic, config.domestic_config)
    identity_map = build_domestic_club_mapping(domestic, bridge)
    cutoff = pd.to_datetime(domestic["kickoff_utc"], utc=True).max().isoformat()
    state_output = output / "domestic_poisson_state_2025_26.json"
    state_payload = {
        "artifact_version": "ao-domestic-poisson-state-v1",
        "state_cutoff_utc": cutoff,
        "source_matches": int(len(domestic)),
        "source_data_sha256": _sha256(domestic_path),
        "identity_bridge_sha256": _sha256(bridge_path),
        "identity_map": {
            club_id: {
                "league_id": identity[0],
                "source_team_id": identity[1],
            }
            for club_id, identity in sorted(identity_map.items())
        },
        "engine_state": engine.to_payload(),
    }
    _write_json(state_output, state_payload)

    manifest_output = output / "manifest.json"
    manifest = {
        "schema_version": "1.0",
        "model_version": PRODUCTION_PREDICTION_MODEL_VERSION,
        "decision": PRODUCTION_PREDICTION_STATUS,
        "family": PRODUCTION_PREDICTION_FAMILY,
        "activation_date": "2026-08-13",
        "config_id": config.config_id,
        "rating_feedback": False,
        "fallback": "CURRENT_AO_1X2",
        "monitoring_season": "2026/27",
        "blend_contract": {
            "space": "LOG_PROBABILITY",
            "current_ml_weight": 0.5,
            "ao_domestic_poisson_weight": 0.5,
            "current_ml_internal": {"ao_weight": 0.1, "ml_weight": 0.9},
            "ao_poisson_internal": {"ao_weight": 0.5, "poisson_weight": 0.5},
        },
        "domestic_config": config.domestic_config.__dict__,
        "transfer_config": config.transfer_config.__dict__,
        "ml_artifact": {
            "path": str(ml_output.relative_to(ROOT)),
            "sha256": _sha256(ml_output),
            "model_fingerprint": model.fingerprint,
            "feature_schema_sha256": feature_schema_sha,
        },
        "domestic_state_artifact": {
            "path": str(state_output.relative_to(ROOT)),
            "sha256": _sha256(state_output),
            "state_cutoff_utc": cutoff,
            "mapped_ao_clubs": len(identity_map),
        },
        "evidence": {
            "source": "output/final_prediction_ensemble_backtest_2018_2026/selected_candidate.json",
            "development_matches": 6340,
            "unseen_matches": 4884,
            "pooled_brier": 0.5680932199564515,
            "pooled_log_loss": 0.9592418968185772,
            "pooled_accuracy": 0.5538493038493039,
            "delta_brier_vs_ao": -0.00399944549914677,
            "delta_log_loss_vs_ao": -0.00512894042292622,
            "manual_operational_decision": True,
            "historical_gate_decision": "KEEP_SHADOW",
        },
        "runtime": {
            "python": platform.python_version(),
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    _write_json(manifest_output, manifest)
    print(f"ML artifact: {ml_output} ({_sha256(ml_output)})")
    print(f"Domestic state: {state_output} ({_sha256(state_output)})")
    print(f"Manifest: {manifest_output} ({_sha256(manifest_output)})")
    print(f"Config ID: {config.config_id}")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload_sha256(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
