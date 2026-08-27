from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np
import pandas as pd

from ao_elo.domestic_poisson import (
    DomesticPoissonConfig,
    DomesticTeamSnapshot,
    DynamicDomesticPoisson,
    EuropeanPoissonTransferConfig,
    empty_domestic_snapshot,
    predict_european_poisson_transfer,
)
from ao_elo.ml_prediction import (
    TrainedML1X2,
    blend_probabilities,
    predict_ml_1x2,
)


PRODUCTION_PREDICTION_STATUS = "PROMOTE_WITH_MONITORING"
PRODUCTION_PREDICTION_FAMILY = "ML_DOMESTIC_POISSON_LOG_PROBABILITY_BLEND"
PRODUCTION_PREDICTION_MODEL_VERSION = "ao-ml-poisson-ensemble-v1-production"
PRODUCTION_PREDICTION_LOG_COLUMNS = (
    "match_id",
    "season",
    "kickoff_utc",
    "generated_at_utc",
    "competition",
    "round",
    "stage",
    "format_type",
    "is_neutral",
    "home_club_id",
    "away_club_id",
    "ao_home_probability",
    "ao_draw_probability",
    "ao_away_probability",
    "ml_home_probability",
    "ml_draw_probability",
    "ml_away_probability",
    "current_ml_home_probability",
    "current_ml_draw_probability",
    "current_ml_away_probability",
    "poisson_home_probability",
    "poisson_draw_probability",
    "poisson_away_probability",
    "ao_poisson_home_probability",
    "ao_poisson_draw_probability",
    "ao_poisson_away_probability",
    "home_probability",
    "draw_probability",
    "away_probability",
    "domestic_poisson_coverage",
    "poisson_component_fallback",
    "prediction_status",
    "fallback_reason",
    "rating_feedback_applied",
    "prediction_model_version",
    "config_id",
    "contract_sha256",
    "artifact_manifest_sha256",
    "ml_artifact_sha256",
    "domestic_state_sha256",
)


@dataclass(frozen=True)
class ProductionPredictionConfig:
    model_version: str
    decision: str
    family: str
    current_ml_weight: float
    ao_poisson_weight: float
    ao_within_ml_weight: float
    raw_ml_within_component_weight: float
    ao_within_poisson_weight: float
    raw_poisson_within_component_weight: float
    fallback: str
    rating_feedback: bool
    monitoring_season: str
    domestic_config: DomesticPoissonConfig
    transfer_config: EuropeanPoissonTransferConfig

    def validate(self) -> None:
        if self.model_version != PRODUCTION_PREDICTION_MODEL_VERSION:
            raise ValueError("Production prediction model_version is invalid")
        if self.decision != PRODUCTION_PREDICTION_STATUS:
            raise ValueError("Production prediction decision must enable monitoring")
        if self.family != PRODUCTION_PREDICTION_FAMILY:
            raise ValueError("Production prediction family is invalid")
        for left_name, left, right_name, right in (
            (
                "current_ml_weight",
                self.current_ml_weight,
                "ao_poisson_weight",
                self.ao_poisson_weight,
            ),
            (
                "ao_within_ml_weight",
                self.ao_within_ml_weight,
                "raw_ml_within_component_weight",
                self.raw_ml_within_component_weight,
            ),
            (
                "ao_within_poisson_weight",
                self.ao_within_poisson_weight,
                "raw_poisson_within_component_weight",
                self.raw_poisson_within_component_weight,
            ),
        ):
            _probability(left_name, left)
            _probability(right_name, right)
            if not math.isclose(float(left) + float(right), 1.0, abs_tol=1e-12):
                raise ValueError(f"{left_name} and {right_name} must sum to one")
        if not math.isclose(self.current_ml_weight, 0.50, abs_tol=1e-12):
            raise ValueError("Production current ML weight must equal 0.50")
        if not math.isclose(self.ao_poisson_weight, 0.50, abs_tol=1e-12):
            raise ValueError("Production AO Poisson weight must equal 0.50")
        if not math.isclose(self.raw_ml_within_component_weight, 0.90, abs_tol=1e-12):
            raise ValueError("Current ML component must preserve its 0.90 ML weight")
        if not math.isclose(
            self.raw_poisson_within_component_weight, 0.50, abs_tol=1e-12
        ):
            raise ValueError("AO Poisson component must preserve its 0.50 weight")
        if not math.isclose(self.transfer_config.rho, 0.0, abs_tol=1e-12):
            raise ValueError("Production Domestic Poisson rho must equal zero")
        if self.fallback != "CURRENT_AO_1X2":
            raise ValueError("Production prediction fallback must be CURRENT_AO_1X2")
        if self.rating_feedback is not False:
            raise ValueError("Production prediction cannot feed back into AO Live Elo")
        if self.monitoring_season != "2026/27":
            raise ValueError("Production prediction monitoring season must be 2026/27")
        self.domestic_config.validate()
        self.transfer_config.validate()

    @property
    def config_id(self) -> str:
        payload = {
            "model_version": self.model_version,
            "decision": self.decision,
            "family": self.family,
            "current_ml_weight": self.current_ml_weight,
            "ao_poisson_weight": self.ao_poisson_weight,
            "ao_within_ml_weight": self.ao_within_ml_weight,
            "raw_ml_within_component_weight": self.raw_ml_within_component_weight,
            "ao_within_poisson_weight": self.ao_within_poisson_weight,
            "raw_poisson_within_component_weight": self.raw_poisson_within_component_weight,
            "fallback": self.fallback,
            "rating_feedback": self.rating_feedback,
            "monitoring_season": self.monitoring_season,
            "domestic_config": self.domestic_config.__dict__,
            "transfer_config": self.transfer_config.__dict__,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class ProductionPredictionRuntime:
    config: ProductionPredictionConfig
    ml_model: TrainedML1X2
    domestic_engine: DynamicDomesticPoisson
    domestic_identity_map: dict[str, tuple[str, str]]
    contract_sha256: str
    artifact_manifest_sha256: str
    ml_artifact_sha256: str
    domestic_state_sha256: str


class ProductionPredictionService:
    """Prediction-only ML/Poisson service with a strict AO fallback."""

    def __init__(
        self,
        runtime: ProductionPredictionRuntime | None,
        *,
        load_error: str | None = None,
        contract_sha256: str = "UNAVAILABLE",
    ):
        self.runtime = runtime
        self.load_error = load_error
        self.contract_sha256 = (
            runtime.contract_sha256 if runtime is not None else contract_sha256
        )

    @classmethod
    def from_contract(
        cls,
        contract_path: str | Path,
        *,
        repository_root: str | Path | None = None,
        allow_degraded_fallback: bool = True,
    ) -> "ProductionPredictionService":
        path = Path(contract_path).resolve()
        contract_sha = _sha256(path)
        try:
            runtime = load_production_prediction_runtime(
                path, repository_root=repository_root
            )
        except (OSError, ValueError, TypeError, KeyError) as exc:
            if not allow_degraded_fallback:
                raise
            return cls(
                None,
                load_error=f"ARTIFACT_OR_CONTRACT_UNAVAILABLE:{type(exc).__name__}",
                contract_sha256=contract_sha,
            )
        return cls(runtime)

    def predict(
        self,
        frame: pd.DataFrame,
        *,
        generated_at_utc: datetime | pd.Timestamp,
    ) -> pd.DataFrame:
        data = _validate_base_prediction_frame(frame, generated_at_utc)
        rows = []
        for row in data.itertuples(index=False):
            row_frame = pd.DataFrame([row._asdict()])
            if self.runtime is None:
                rows.append(
                    self._fallback_row(
                        row,
                        pd.Timestamp(generated_at_utc),
                        self.load_error or "RUNTIME_UNAVAILABLE",
                    )
                )
                continue
            try:
                rows.append(
                    self._predict_row(row, row_frame, pd.Timestamp(generated_at_utc))
                )
            except Exception as exc:  # noqa: BLE001 - the contract is total
                # The contract promises Current AO 1X2 for any artifact,
                # feature or state problem, so a narrower catch would let an
                # unexpected exception type take down the whole batch instead
                # of degrading one row. The type name is kept in the reason so
                # the breadth costs no diagnostic detail, and the fallback rate
                # is an operational alarm. BaseException is deliberately not
                # caught: interrupts and exits must still stop the process.
                rows.append(
                    self._fallback_row(
                        row,
                        pd.Timestamp(generated_at_utc),
                        f"FEATURE_OR_STATE_INVALID:{type(exc).__name__}",
                    )
                )
        result = pd.DataFrame(rows, columns=PRODUCTION_PREDICTION_LOG_COLUMNS)
        _validate_prediction_log(result, expected_rows=len(data))
        return result

    def apply_domestic_results(self, matches: pd.DataFrame) -> pd.DataFrame:
        if self.runtime is None:
            raise ValueError("Cannot update domestic state while runtime is degraded")
        return self.runtime.domestic_engine.update_batch(matches)

    def domestic_state_payload(self) -> dict[str, object]:
        if self.runtime is None:
            raise ValueError("Cannot checkpoint domestic state while runtime is degraded")
        return self.runtime.domestic_engine.to_payload()

    def _predict_row(
        self,
        row: object,
        row_frame: pd.DataFrame,
        generated_at: pd.Timestamp,
    ) -> dict[str, object]:
        assert self.runtime is not None
        runtime = self.runtime
        ao = _row_ao_probabilities(row)
        raw_ml = predict_ml_1x2(runtime.ml_model, row_frame)
        current_ml = blend_probabilities(
            ao,
            raw_ml,
            runtime.config.raw_ml_within_component_weight,
        )
        transfer = _build_poisson_transfer_row(runtime, row, ao)
        poisson_prediction = predict_european_poisson_transfer(
            transfer,
            runtime.config.transfer_config,
            fallback_to_ao_without_history=True,
        )
        raw_poisson = poisson_prediction[
            ["home_probability", "draw_probability", "away_probability"]
        ].to_numpy(float)
        ao_poisson = blend_probabilities(
            ao,
            raw_poisson,
            runtime.config.raw_poisson_within_component_weight,
        )
        final = blend_probabilities(
            current_ml,
            ao_poisson,
            runtime.config.ao_poisson_weight,
        )
        coverage = str(transfer.loc[0, "domestic_poisson_coverage"])
        return _prediction_log_row(
            row,
            generated_at,
            ao=ao[0],
            raw_ml=raw_ml[0],
            current_ml=current_ml[0],
            raw_poisson=raw_poisson[0],
            ao_poisson=ao_poisson[0],
            final=final[0],
            coverage=coverage,
            poisson_component_fallback=bool(
                poisson_prediction.loc[0, "ao_fallback"]
            ),
            status="ACTIVE_ENSEMBLE",
            fallback_reason="",
            runtime=runtime,
        )

    def _fallback_row(
        self,
        row: object,
        generated_at: pd.Timestamp,
        reason: str,
    ) -> dict[str, object]:
        ao = _row_ao_probabilities(row)[0]
        runtime = self.runtime
        return _prediction_log_row(
            row,
            generated_at,
            ao=ao,
            raw_ml=np.full(3, np.nan),
            current_ml=np.full(3, np.nan),
            raw_poisson=np.full(3, np.nan),
            ao_poisson=np.full(3, np.nan),
            final=ao,
            coverage="UNAVAILABLE",
            poisson_component_fallback=True,
            status="FALLBACK_CURRENT_AO",
            fallback_reason=reason,
            runtime=runtime,
            contract_sha256=self.contract_sha256,
        )


def load_production_prediction_runtime(
    contract_path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> ProductionPredictionRuntime:
    path = Path(contract_path).resolve()
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else path.parent.parent
    )
    contract = json.loads(path.read_text(encoding="utf-8"))
    layer = contract.get("prediction_layer")
    if not isinstance(layer, Mapping) or layer.get("active") is not True:
        raise ValueError("Production contract must activate prediction_layer")
    artifact_reference = layer.get("artifact_manifest")
    if not isinstance(artifact_reference, Mapping):
        raise ValueError("prediction_layer.artifact_manifest is required")
    manifest_path = _repository_path(root, artifact_reference.get("path"))
    expected_manifest_sha = _required_sha(
        artifact_reference.get("sha256"), "artifact manifest sha256"
    )
    actual_manifest_sha = _sha256(manifest_path)
    if actual_manifest_sha != expected_manifest_sha:
        raise ValueError("Production prediction artifact manifest checksum mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = _config_from_contract(layer)
    config.validate()
    _validate_manifest_matches_config(manifest, config)

    ml_reference = manifest.get("ml_artifact")
    state_reference = manifest.get("domestic_state_artifact")
    if not isinstance(ml_reference, Mapping) or not isinstance(
        state_reference, Mapping
    ):
        raise ValueError("Production prediction artifacts are incomplete")
    ml_path = _repository_path(root, ml_reference.get("path"))
    state_path = _repository_path(root, state_reference.get("path"))
    ml_sha = _required_sha(ml_reference.get("sha256"), "ML artifact sha256")
    state_sha = _required_sha(
        state_reference.get("sha256"), "domestic state sha256"
    )
    if _sha256(ml_path) != ml_sha:
        raise ValueError("Production ML artifact checksum mismatch")
    if _sha256(state_path) != state_sha:
        raise ValueError("Production domestic state checksum mismatch")

    package = joblib.load(ml_path)
    if not isinstance(package, Mapping):
        raise ValueError("Production ML artifact package is invalid")
    model = package.get("model")
    if not isinstance(model, TrainedML1X2):
        raise ValueError("Production ML artifact does not contain TrainedML1X2")
    if model.arm_name != "STRUCTURAL_LOGISTIC" or model.family != "LOGISTIC":
        raise ValueError("Production ML artifact uses an unexpected model family")
    if package.get("production_status") != PRODUCTION_PREDICTION_STATUS:
        raise ValueError("Production ML artifact status is not activated")
    if not math.isclose(
        float(package.get("ao_blend_weight")),
        config.raw_ml_within_component_weight,
        abs_tol=1e-12,
    ):
        raise ValueError("Production ML artifact blend weight mismatch")
    if package.get("model_fingerprint") != model.fingerprint:
        raise ValueError("Production ML model fingerprint mismatch")
    schema_payload = _feature_schema_payload(model)
    if package.get("feature_schema") != schema_payload:
        raise ValueError("Production ML feature schema mismatch")
    schema_sha = _payload_sha256(schema_payload)
    if package.get("feature_schema_sha256") != schema_sha:
        raise ValueError("Production ML feature schema checksum mismatch")
    if ml_reference.get("feature_schema_sha256") != schema_sha:
        raise ValueError("Artifact manifest feature schema checksum mismatch")

    state_artifact = json.loads(state_path.read_text(encoding="utf-8"))
    engine_payload = state_artifact.get("engine_state")
    identity_payload = state_artifact.get("identity_map")
    if not isinstance(engine_payload, Mapping) or not isinstance(
        identity_payload, Mapping
    ):
        raise ValueError("Production domestic state artifact is incomplete")
    engine = DynamicDomesticPoisson.from_payload(engine_payload)
    if engine.config != config.domestic_config:
        raise ValueError("Production domestic state config mismatch")
    identity_map: dict[str, tuple[str, str]] = {}
    excluded_club_ids = state_artifact.get("excluded_ao_club_ids", [])
    if not isinstance(excluded_club_ids, list) or not all(
        isinstance(club_id, str) and club_id for club_id in excluded_club_ids
    ):
        raise ValueError("Production domestic state excluded_ao_club_ids is invalid")
    excluded_club_ids = set(excluded_club_ids)
    for club_id, identity in identity_payload.items():
        if not isinstance(identity, Mapping):
            raise ValueError("Domestic identity map entry is invalid")
        league_id = identity.get("league_id")
        source_team_id = identity.get("source_team_id")
        if not all(isinstance(value, str) and value for value in (league_id, source_team_id)):
            raise ValueError("Domestic identity map values must be non-empty strings")
        if str(club_id) not in excluded_club_ids:
            identity_map[str(club_id)] = (league_id, source_team_id)
    return ProductionPredictionRuntime(
        config=config,
        ml_model=model,
        domestic_engine=engine,
        domestic_identity_map=identity_map,
        contract_sha256=_sha256(path),
        artifact_manifest_sha256=actual_manifest_sha,
        ml_artifact_sha256=ml_sha,
        domestic_state_sha256=state_sha,
    )


def selected_production_prediction_config() -> ProductionPredictionConfig:
    """Return the manually approved, frozen 2026/27 monitoring configuration."""
    config = ProductionPredictionConfig(
        model_version=PRODUCTION_PREDICTION_MODEL_VERSION,
        decision=PRODUCTION_PREDICTION_STATUS,
        family=PRODUCTION_PREDICTION_FAMILY,
        current_ml_weight=0.50,
        ao_poisson_weight=0.50,
        ao_within_ml_weight=0.10,
        raw_ml_within_component_weight=0.90,
        ao_within_poisson_weight=0.50,
        raw_poisson_within_component_weight=0.50,
        fallback="CURRENT_AO_1X2",
        rating_feedback=False,
        monitoring_season="2026/27",
        domestic_config=DomesticPoissonConfig(
            team_learning_rate=0.02,
            season_carry=0.90,
            shrinkage_matches=10.0,
            venue_context=False,
            model_version="ao-domestic-dynamic-poisson-v1-production-state",
        ),
        transfer_config=EuropeanPoissonTransferConfig(
            mu=0.28040079791632944,
            elo_slope=0.6663036964859297,
            attack_coefficient=0.09209745423072817,
            defence_coefficient=0.06065648173017676,
            venue_coefficient=0.0,
            l2_strength=10.0,
            rho=0.0,
            use_reliability=True,
            use_venue=False,
            model_version="ao-domestic-poisson-transfer-v1-production-rho0",
        ),
    )
    config.validate()
    return config


def _config_from_contract(layer: Mapping[str, object]) -> ProductionPredictionConfig:
    top = _mapping(layer.get("top_level_blend"), "top_level_blend")
    ml = _mapping(layer.get("current_ml_component"), "current_ml_component")
    poisson = _mapping(
        layer.get("ao_domestic_poisson_component"),
        "ao_domestic_poisson_component",
    )
    monitoring = _mapping(layer.get("monitoring"), "monitoring")
    if top.get("space") != "LOG_PROBABILITY":
        raise ValueError("Production prediction blend space must be LOG_PROBABILITY")
    if ml.get("source") != "STRUCTURAL_LOGISTIC":
        raise ValueError("Production current ML source must be STRUCTURAL_LOGISTIC")
    if poisson.get("source") != "AO_POISSON_RHO0_CONTROL":
        raise ValueError("Production Poisson source must be AO_POISSON_RHO0_CONTROL")
    if layer.get("rating_feedback") is not False:
        raise ValueError("Production prediction must explicitly disable rating feedback")
    if monitoring.get("active") is not True:
        raise ValueError("Production prediction monitoring must be active")
    if monitoring.get("log_every_prediction") is not True:
        raise ValueError("Production prediction must log every pre-match prediction")
    if monitoring.get("compare_against_current_ao") is not True:
        raise ValueError("Production monitoring must retain the current AO comparator")
    domestic = DomesticPoissonConfig(
        **dict(_mapping(poisson.get("domestic_config"), "domestic_config"))
    )
    transfer = EuropeanPoissonTransferConfig(
        **dict(_mapping(poisson.get("transfer_config"), "transfer_config"))
    )
    return ProductionPredictionConfig(
        model_version=str(layer.get("model_version")),
        decision=str(layer.get("decision")),
        family=str(layer.get("family")),
        current_ml_weight=float(top.get("current_ml_weight")),
        ao_poisson_weight=float(top.get("ao_domestic_poisson_weight")),
        ao_within_ml_weight=float(ml.get("ao_weight")),
        raw_ml_within_component_weight=float(ml.get("ml_weight")),
        ao_within_poisson_weight=float(poisson.get("ao_weight")),
        raw_poisson_within_component_weight=float(poisson.get("poisson_weight")),
        fallback=str(layer.get("fallback")),
        rating_feedback=False,
        monitoring_season=str(monitoring.get("season")),
        domestic_config=domestic,
        transfer_config=transfer,
    )


def _build_poisson_transfer_row(
    runtime: ProductionPredictionRuntime,
    row: object,
    ao: np.ndarray,
) -> pd.DataFrame:
    home = _domestic_snapshot(runtime, str(row.home_club_id))
    away = _domestic_snapshot(runtime, str(row.away_club_id))
    coverage = (
        "BOTH"
        if home.covered and away.covered
        else "ONE"
        if home.covered or away.covered
        else "NONE"
    )
    values: dict[str, object] = {
        "match_id": str(row.match_id),
        "expected_home_score": float(row.expected_home_score),
        "ao_home_probability": float(ao[0, 0]),
        "ao_draw_probability": float(ao[0, 1]),
        "ao_away_probability": float(ao[0, 2]),
        "domestic_poisson_coverage": coverage,
        "domestic_poisson_venue_edge": home.home_venue - away.away_venue,
        "domestic_poisson_venue_edge_raw": (
            home.home_venue_raw_z - away.away_venue_raw_z
        ),
    }
    for side, snapshot in (("home", home), ("away", away)):
        for name in ("attack", "defence"):
            values[f"{side}_domestic_poisson_{name}"] = getattr(snapshot, name)
            values[f"{side}_domestic_poisson_{name}_raw_z"] = getattr(
                snapshot, f"{name}_raw_z"
            )
    return pd.DataFrame([values])


def _domestic_snapshot(
    runtime: ProductionPredictionRuntime,
    club_id: str,
) -> DomesticTeamSnapshot:
    identity = runtime.domestic_identity_map.get(club_id)
    if identity is None:
        return empty_domestic_snapshot()
    league = runtime.domestic_engine.leagues.get(identity[0])
    if league is None or league.current_season is None:
        return empty_domestic_snapshot()
    return runtime.domestic_engine.snapshot(
        identity[0],
        league.current_season,
        identity[1],
    )


def _validate_base_prediction_frame(
    frame: pd.DataFrame,
    generated_at_utc: datetime | pd.Timestamp,
) -> pd.DataFrame:
    required = {
        "match_id",
        "season",
        "kickoff_utc",
        "competition",
        "round",
        "stage",
        "format_type",
        "is_neutral",
        "home_club_id",
        "away_club_id",
        "ao_home_probability",
        "ao_draw_probability",
        "ao_away_probability",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Production prediction frame missing columns: {missing}")
    if frame.empty:
        raise ValueError("Production prediction frame cannot be empty")
    result = frame.copy()
    if result["match_id"].isna().any() or result["match_id"].duplicated().any():
        raise ValueError("Prediction match_id must be non-null and unique")
    identity_columns = [
        "season",
        "competition",
        "round",
        "stage",
        "format_type",
        "home_club_id",
        "away_club_id",
    ]
    if result[identity_columns].isna().any().any():
        raise ValueError("Prediction identity fields cannot be missing")
    if result["home_club_id"].astype(str).eq(result["away_club_id"].astype(str)).any():
        raise ValueError("Home and away club IDs must differ")
    try:
        result["is_neutral"] = result["is_neutral"].map(_coerce_boolean)
    except ValueError as exc:
        raise ValueError("is_neutral must be boolean or 0/1") from exc
    kickoff = pd.to_datetime(result["kickoff_utc"], utc=True, errors="coerce")
    generated = pd.Timestamp(generated_at_utc)
    if generated.tzinfo is None:
        raise ValueError("generated_at_utc must be timezone-aware")
    generated = generated.tz_convert("UTC")
    if kickoff.isna().any() or (kickoff <= generated).any():
        raise ValueError("Every production prediction must be locked before kickoff")
    result["kickoff_utc"] = kickoff
    probabilities = result[
        ["ao_home_probability", "ao_draw_probability", "ao_away_probability"]
    ].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0.0).any()
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-10, rtol=0.0)
    ):
        raise ValueError("Current AO fallback probabilities must be normalized")
    return result.reset_index(drop=True)


def _prediction_log_row(
    row: object,
    generated_at: pd.Timestamp,
    *,
    ao: np.ndarray,
    raw_ml: np.ndarray,
    current_ml: np.ndarray,
    raw_poisson: np.ndarray,
    ao_poisson: np.ndarray,
    final: np.ndarray,
    coverage: str,
    poisson_component_fallback: bool,
    status: str,
    fallback_reason: str,
    runtime: ProductionPredictionRuntime | None,
    contract_sha256: str | None = None,
) -> dict[str, object]:
    values = {
        "match_id": str(row.match_id),
        "season": str(row.season),
        "kickoff_utc": pd.Timestamp(row.kickoff_utc).isoformat(),
        "generated_at_utc": generated_at.tz_convert("UTC").isoformat(),
        "competition": str(row.competition),
        "round": str(row.round),
        "stage": str(row.stage),
        "format_type": str(row.format_type),
        "is_neutral": bool(row.is_neutral),
        "home_club_id": str(row.home_club_id),
        "away_club_id": str(row.away_club_id),
        "ao_home_probability": float(ao[0]),
        "ao_draw_probability": float(ao[1]),
        "ao_away_probability": float(ao[2]),
        "ml_home_probability": _optional_float(raw_ml[0]),
        "ml_draw_probability": _optional_float(raw_ml[1]),
        "ml_away_probability": _optional_float(raw_ml[2]),
        "current_ml_home_probability": _optional_float(current_ml[0]),
        "current_ml_draw_probability": _optional_float(current_ml[1]),
        "current_ml_away_probability": _optional_float(current_ml[2]),
        "poisson_home_probability": _optional_float(raw_poisson[0]),
        "poisson_draw_probability": _optional_float(raw_poisson[1]),
        "poisson_away_probability": _optional_float(raw_poisson[2]),
        "ao_poisson_home_probability": _optional_float(ao_poisson[0]),
        "ao_poisson_draw_probability": _optional_float(ao_poisson[1]),
        "ao_poisson_away_probability": _optional_float(ao_poisson[2]),
        "home_probability": float(final[0]),
        "draw_probability": float(final[1]),
        "away_probability": float(final[2]),
        "domestic_poisson_coverage": coverage,
        "poisson_component_fallback": bool(poisson_component_fallback),
        "prediction_status": status,
        "fallback_reason": fallback_reason,
        "rating_feedback_applied": False,
        "prediction_model_version": (
            runtime.config.model_version
            if runtime is not None
            else "CURRENT_AO_FALLBACK"
        ),
        "config_id": runtime.config.config_id if runtime is not None else "FALLBACK",
        "contract_sha256": (
            runtime.contract_sha256
            if runtime is not None
            else contract_sha256 or "UNAVAILABLE"
        ),
        "artifact_manifest_sha256": (
            runtime.artifact_manifest_sha256 if runtime is not None else "UNAVAILABLE"
        ),
        "ml_artifact_sha256": (
            runtime.ml_artifact_sha256 if runtime is not None else "UNAVAILABLE"
        ),
        "domestic_state_sha256": (
            runtime.domestic_state_sha256 if runtime is not None else "UNAVAILABLE"
        ),
    }
    return values


def _validate_prediction_log(frame: pd.DataFrame, *, expected_rows: int) -> None:
    if tuple(frame.columns) != PRODUCTION_PREDICTION_LOG_COLUMNS:
        raise ValueError("Production prediction log schema mismatch")
    if len(frame) != expected_rows or frame["match_id"].duplicated().any():
        raise ValueError("Production prediction log must contain one row per match")
    probabilities = frame[
        ["home_probability", "draw_probability", "away_probability"]
    ].to_numpy(float)
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0.0).any()
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12, rtol=0.0)
    ):
        raise ValueError("Production probabilities must be finite and normalized")
    if frame["rating_feedback_applied"].any():
        raise ValueError("Production prediction log cannot report rating feedback")


def _validate_manifest_matches_config(
    manifest: Mapping[str, object],
    config: ProductionPredictionConfig,
) -> None:
    if manifest.get("model_version") != config.model_version:
        raise ValueError("Artifact manifest model_version mismatch")
    if manifest.get("decision") != config.decision:
        raise ValueError("Artifact manifest decision mismatch")
    if manifest.get("family") != config.family:
        raise ValueError("Artifact manifest family mismatch")
    if manifest.get("rating_feedback") is not False:
        raise ValueError("Artifact manifest must disable rating feedback")
    if manifest.get("fallback") != config.fallback:
        raise ValueError("Artifact manifest fallback mismatch")
    if manifest.get("config_id") != config.config_id:
        raise ValueError("Artifact manifest config_id mismatch")


def _row_ao_probabilities(row: object) -> np.ndarray:
    values = np.array(
        [[
            float(row.ao_home_probability),
            float(row.ao_draw_probability),
            float(row.ao_away_probability),
        ]],
        dtype=float,
    )
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("AO probabilities must sum to one")
    return values


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _repository_path(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Artifact path must be a non-empty string")
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Artifact path must remain inside the repository") from exc
    return path


def _required_sha(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a SHA-256 hex digest") from exc
    return value.lower()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _feature_schema_payload(model: TrainedML1X2) -> dict[str, list[str]]:
    return {
        "numeric": list(model.schema.numeric),
        "categorical": list(model.schema.categorical),
    }


def _payload_sha256(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _probability(name: str, value: float) -> None:
    if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be in [0,1]")


def _coerce_boolean(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    if isinstance(value, (float, np.floating)) and float(value) in {0.0, 1.0}:
        return bool(int(value))
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    raise ValueError("invalid boolean")


def _optional_float(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None
