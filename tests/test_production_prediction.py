from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from ao_elo.domestic_poisson import DynamicDomesticPoisson
from ao_elo.ml_features import FEATURE_SCHEMAS
from ao_elo.ml_prediction import blend_probabilities
from ao_elo.production_prediction import (
    PRODUCTION_PREDICTION_LOG_COLUMNS,
    ProductionPredictionService,
    load_production_prediction_runtime,
)


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "contracts" / "ao_european_elo_v2_production.json"


def test_production_runtime_loads_frozen_artifacts() -> None:
    runtime = load_production_prediction_runtime(PRODUCTION)

    assert runtime.config.decision == "PROMOTE_WITH_MONITORING"
    assert runtime.config.current_ml_weight == pytest.approx(0.5)
    assert runtime.config.ao_poisson_weight == pytest.approx(0.5)
    assert runtime.config.transfer_config.rho == pytest.approx(0.0)
    assert runtime.config.rating_feedback is False
    assert runtime.ml_model.arm_name == "STRUCTURAL_LOGISTIC"
    assert len(runtime.domestic_identity_map) == 311
    assert "AO-UEFA-52298" in runtime.domestic_identity_map  # Ferencvaros
    assert "AO-UEFA-2605907" in runtime.domestic_identity_map  # Kauno Zalgiris
    assert "AO-UEFA-2603107" not in runtime.domestic_identity_map  # Torreense


def test_production_prediction_is_normalized_and_prediction_only() -> None:
    service = ProductionPredictionService.from_contract(
        PRODUCTION, allow_degraded_fallback=False
    )
    result = service.predict(
        _feature_frame(),
        generated_at_utc=pd.Timestamp("2026-08-13T10:00:00Z"),
    )

    assert tuple(result.columns) == PRODUCTION_PREDICTION_LOG_COLUMNS
    assert result.loc[0, "prediction_status"] == "ACTIVE_ENSEMBLE"
    assert result.loc[0, "domestic_poisson_coverage"] == "BOTH"
    assert not bool(result.loc[0, "rating_feedback_applied"])
    assert result.loc[0, "home_probability"] + result.loc[0, "draw_probability"] + result.loc[0, "away_probability"] == pytest.approx(1.0)
    assert result.loc[0, "home_probability"] != pytest.approx(
        result.loc[0, "ao_home_probability"]
    )


def test_production_coverage_gate_disables_insufficient_team_profile() -> None:
    service = ProductionPredictionService.from_contract(
        PRODUCTION, allow_degraded_fallback=False
    )
    frame = _feature_frame()
    frame["home_club_id"] = "AO-UEFA-2603107"  # Uniao Torreense: below gate
    frame["away_club_id"] = "AO-UEFA-2605907"  # Kauno Zalgiris: eligible

    result = service.predict(
        frame,
        generated_at_utc=pd.Timestamp("2026-08-13T10:00:00Z"),
    )

    assert result.loc[0, "prediction_status"] == "ACTIVE_ENSEMBLE"
    assert result.loc[0, "domestic_poisson_coverage"] == "ONE"


def test_production_prediction_uses_frozen_nested_log_blends_without_state_mutation(
) -> None:
    service = ProductionPredictionService.from_contract(
        PRODUCTION, allow_degraded_fallback=False
    )
    assert service.runtime is not None
    state_before = service.domestic_state_payload()
    result = service.predict(
        _feature_frame(),
        generated_at_utc=pd.Timestamp("2026-08-13T10:00:00Z"),
    )
    ao = result[
        ["ao_home_probability", "ao_draw_probability", "ao_away_probability"]
    ].to_numpy(float)
    raw_ml = result[
        ["ml_home_probability", "ml_draw_probability", "ml_away_probability"]
    ].to_numpy(float)
    raw_poisson = result[
        [
            "poisson_home_probability",
            "poisson_draw_probability",
            "poisson_away_probability",
        ]
    ].to_numpy(float)
    current_ml = blend_probabilities(ao, raw_ml, 0.90)
    ao_poisson = blend_probabilities(ao, raw_poisson, 0.50)
    expected = blend_probabilities(current_ml, ao_poisson, 0.50)

    assert result[
        [
            "current_ml_home_probability",
            "current_ml_draw_probability",
            "current_ml_away_probability",
        ]
    ].to_numpy(float) == pytest.approx(current_ml)
    assert result[
        [
            "ao_poisson_home_probability",
            "ao_poisson_draw_probability",
            "ao_poisson_away_probability",
        ]
    ].to_numpy(float) == pytest.approx(ao_poisson)
    assert result[
        ["home_probability", "draw_probability", "away_probability"]
    ].to_numpy(float) == pytest.approx(expected)
    assert service.domestic_state_payload() == state_before


def test_missing_model_feature_falls_back_exactly_to_current_ao() -> None:
    service = ProductionPredictionService.from_contract(
        PRODUCTION, allow_degraded_fallback=False
    )
    frame = _feature_frame().drop(columns=["home_euro_residual_h3"])
    result = service.predict(
        frame,
        generated_at_utc=pd.Timestamp("2026-08-13T10:00:00Z"),
    )

    assert result.loc[0, "prediction_status"] == "FALLBACK_CURRENT_AO"
    assert result.loc[0, "home_probability"] == pytest.approx(0.38)
    assert result.loc[0, "draw_probability"] == pytest.approx(0.24)
    assert result.loc[0, "away_probability"] == pytest.approx(0.38)
    assert result.loc[0, "fallback_reason"].startswith("FEATURE_OR_STATE_INVALID")


def test_invalid_artifact_checksum_enters_degraded_ao_fallback(tmp_path: Path) -> None:
    payload = json.loads(PRODUCTION.read_text(encoding="utf-8"))
    payload["prediction_layer"]["artifact_manifest"]["sha256"] = "0" * 64
    contract = tmp_path / "production.json"
    contract.write_text(json.dumps(payload), encoding="utf-8")
    service = ProductionPredictionService.from_contract(
        contract,
        repository_root=ROOT,
        allow_degraded_fallback=True,
    )
    result = service.predict(
        _feature_frame(),
        generated_at_utc=pd.Timestamp("2026-08-13T10:00:00Z"),
    )

    assert service.runtime is None
    assert result.loc[0, "prediction_status"] == "FALLBACK_CURRENT_AO"
    assert result.loc[0, "home_probability"] == pytest.approx(0.38)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("prediction_layer", "top_level_blend", "space"), "LINEAR"),
        (("prediction_layer", "current_ml_component", "source"), "OTHER_ML"),
        (
            ("prediction_layer", "ao_domestic_poisson_component", "source"),
            "AO_POISSON_BLEND",
        ),
        (("prediction_layer", "rating_feedback"), True),
        (("prediction_layer", "monitoring", "active"), False),
    ],
)
def test_runtime_rejects_contract_drift(
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
) -> None:
    payload = json.loads(PRODUCTION.read_text(encoding="utf-8"))
    cursor = payload
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    contract = tmp_path / "production.json"
    contract.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        load_production_prediction_runtime(contract, repository_root=ROOT)


def test_prediction_must_be_generated_before_kickoff() -> None:
    service = ProductionPredictionService.from_contract(
        PRODUCTION, allow_degraded_fallback=False
    )

    with pytest.raises(ValueError, match="locked before kickoff"):
        service.predict(
            _feature_frame(),
            generated_at_utc=pd.Timestamp("2026-09-01T20:00:00Z"),
        )


def test_same_team_match_is_rejected() -> None:
    service = ProductionPredictionService.from_contract(
        PRODUCTION, allow_degraded_fallback=False
    )
    frame = _feature_frame()
    frame["away_club_id"] = frame["home_club_id"]

    with pytest.raises(ValueError, match="must differ"):
        service.predict(
            frame,
            generated_at_utc=pd.Timestamp("2026-08-13T10:00:00Z"),
        )


def test_invalid_neutral_flag_is_rejected() -> None:
    service = ProductionPredictionService.from_contract(
        PRODUCTION, allow_degraded_fallback=False
    )
    frame = _feature_frame()
    frame["is_neutral"] = "sometimes"

    with pytest.raises(ValueError, match="boolean or 0/1"):
        service.predict(
            frame,
            generated_at_utc=pd.Timestamp("2026-08-13T10:00:00Z"),
        )


def test_domestic_state_serialization_round_trip() -> None:
    runtime = load_production_prediction_runtime(PRODUCTION)
    payload = runtime.domestic_engine.to_payload()
    restored = DynamicDomesticPoisson.from_payload(payload)

    assert restored.config == runtime.domestic_engine.config
    assert restored.to_payload() == payload


def _feature_frame() -> pd.DataFrame:
    numeric = {
        column: 0.0
        for column in FEATURE_SCHEMAS["STRUCTURAL_LOGISTIC"].numeric
    }
    categorical = {
        "competition": "UCL",
        "stage": "LEAGUE",
        "round": "League Stage",
        "format_type": "LEAGUE_OR_GROUP",
    }
    ao = (0.38, 0.24, 0.38)
    numeric.update(
        {
            "ao_log_home_draw": math.log(ao[0] / ao[1]),
            "ao_log_away_draw": math.log(ao[2] / ao[1]),
            "expected_home_score": 0.5,
            "month": 9.0,
            "home_days_since_any_match": 10.0,
            "away_days_since_any_match": 10.0,
        }
    )
    return pd.DataFrame(
        [
            {
                "match_id": "production-test-1",
                "season": "2026/27",
                "kickoff_utc": "2026-09-01T19:00:00Z",
                "home_club_id": "AO-UEFA-50030",
                "away_club_id": "AO-UEFA-50062",
                "is_neutral": False,
                "ao_home_probability": ao[0],
                "ao_draw_probability": ao[1],
                "ao_away_probability": ao[2],
                **numeric,
                **categorical,
            }
        ]
    )


# ---------------------------------------------------------------------------
# the served fallback must be total: the contract promises Current AO 1X2 for
# any artifact, feature or state problem, so an unexpected exception type must
# degrade one row rather than take down the whole batch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "failure",
    [
        AttributeError("model lost an attribute"),
        IndexError("positional lookup escaped the frame"),
        ZeroDivisionError("degenerate scaling term"),
        RuntimeError("third-party library failure"),
        FloatingPointError("numpy raised under errstate"),
    ],
)
def test_unexpected_exception_types_fall_back_instead_of_propagating(
    monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    """None of these inherit from the four types the catch used to list."""
    service = ProductionPredictionService.from_contract(
        PRODUCTION, allow_degraded_fallback=False
    )

    def explode(*args: object, **kwargs: object) -> dict[str, object]:
        raise failure

    monkeypatch.setattr(service, "_predict_row", explode)
    result = service.predict(
        _feature_frame(),
        generated_at_utc=pd.Timestamp("2026-08-13T10:00:00Z"),
    )

    assert result.loc[0, "prediction_status"] == "FALLBACK_CURRENT_AO"
    assert result.loc[0, "fallback_reason"] == (
        f"FEATURE_OR_STATE_INVALID:{type(failure).__name__}"
    )
    assert result.loc[0, "home_probability"] == pytest.approx(0.38)
    assert result.loc[0, "draw_probability"] == pytest.approx(0.24)
    assert result.loc[0, "away_probability"] == pytest.approx(0.38)


def test_one_bad_row_does_not_lose_the_healthy_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A batch must survive a single poisoned row."""
    service = ProductionPredictionService.from_contract(
        PRODUCTION, allow_degraded_fallback=False
    )
    frame = pd.concat([_feature_frame(), _feature_frame()], ignore_index=True)
    frame.loc[1, "match_id"] = f"{frame.loc[0, 'match_id']}-second"

    original = service._predict_row
    calls = {"count": 0}

    def flaky(*args: object, **kwargs: object) -> dict[str, object]:
        calls["count"] += 1
        if calls["count"] == 1:
            raise AttributeError("first row explodes")
        return original(*args, **kwargs)

    monkeypatch.setattr(service, "_predict_row", flaky)
    result = service.predict(
        frame,
        generated_at_utc=pd.Timestamp("2026-08-13T10:00:00Z"),
    )

    assert len(result) == 2
    assert result.loc[0, "prediction_status"] == "FALLBACK_CURRENT_AO"
    assert result.loc[1, "prediction_status"] == "ACTIVE_ENSEMBLE"


def test_keyboard_interrupt_still_stops_the_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BaseException must not be swallowed by the broadened catch."""
    service = ProductionPredictionService.from_contract(
        PRODUCTION, allow_degraded_fallback=False
    )

    def interrupt(*args: object, **kwargs: object) -> dict[str, object]:
        raise KeyboardInterrupt

    monkeypatch.setattr(service, "_predict_row", interrupt)

    with pytest.raises(KeyboardInterrupt):
        service.predict(
            _feature_frame(),
            generated_at_utc=pd.Timestamp("2026-08-13T10:00:00Z"),
        )
