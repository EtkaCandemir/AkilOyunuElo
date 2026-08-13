from __future__ import annotations

import numpy as np
import pytest

from ao_elo.ml_prediction import blend_probabilities
from ao_elo.prediction_ensemble import (
    AO_POISSON_BLEND,
    AO_POISSON_RHO0_CONTROL,
    select_prediction_ensemble,
)


def test_prediction_ensemble_selects_complementary_poisson_source() -> None:
    outcomes = np.array([0, 1, 2, 0, 1, 2])
    current_ml = np.array(
        [
            [0.45, 0.30, 0.25],
            [0.30, 0.40, 0.30],
            [0.25, 0.30, 0.45],
            [0.45, 0.30, 0.25],
            [0.30, 0.40, 0.30],
            [0.25, 0.30, 0.45],
        ]
    )
    rho0 = np.eye(3)[outcomes] * 0.70 + 0.10
    rho0 /= rho0.sum(axis=1, keepdims=True)

    selected, surface = select_prediction_ensemble(
        current_ml,
        {
            AO_POISSON_RHO0_CONTROL: rho0,
            AO_POISSON_BLEND: current_ml,
        },
        outcomes,
    )

    assert len(surface) == 22
    assert selected["poisson_source"] == AO_POISSON_RHO0_CONTROL
    assert selected["poisson_weight"] == 1.0
    assert selected["ml_weight"] == 0.0


def test_prediction_ensemble_uses_zero_weight_when_no_source_adds_value() -> None:
    outcomes = np.array([0, 1, 2])
    current_ml = np.array(
        [
            [0.70, 0.20, 0.10],
            [0.15, 0.70, 0.15],
            [0.10, 0.20, 0.70],
        ]
    )

    selected, _ = select_prediction_ensemble(
        current_ml,
        {
            AO_POISSON_RHO0_CONTROL: current_ml,
            AO_POISSON_BLEND: current_ml,
        },
        outcomes,
    )

    assert selected["poisson_weight"] == 0.0
    assert selected["poisson_source"] == AO_POISSON_RHO0_CONTROL


def test_prediction_ensemble_is_log_probability_blend() -> None:
    outcomes = np.array([0, 2])
    current_ml = np.array([[0.60, 0.25, 0.15], [0.20, 0.25, 0.55]])
    poisson = np.array([[0.50, 0.30, 0.20], [0.15, 0.30, 0.55]])

    selected, surface = select_prediction_ensemble(
        current_ml,
        {
            AO_POISSON_RHO0_CONTROL: poisson,
            AO_POISSON_BLEND: poisson,
        },
        outcomes,
        weights=(0.5,),
    )
    expected = blend_probabilities(current_ml, poisson, 0.5)
    observed = next(
        row
        for row in surface
        if row["poisson_source"] == selected["poisson_source"]
    )

    assert selected["poisson_weight"] == 0.5
    assert np.isclose(expected.sum(axis=1), 1.0).all()
    assert observed["brier_1x2"] >= 0.0


def test_prediction_ensemble_rejects_invalid_source_contract() -> None:
    current_ml = np.array([[0.50, 0.30, 0.20]])

    with pytest.raises(ValueError, match="Poisson sources"):
        select_prediction_ensemble(
            current_ml,
            {AO_POISSON_BLEND: current_ml},
            np.array([0]),
        )


@pytest.mark.parametrize("weight", (-0.1, 1.1, float("nan")))
def test_prediction_ensemble_rejects_invalid_weights(weight: float) -> None:
    current_ml = np.array([[0.50, 0.30, 0.20]])

    with pytest.raises(ValueError, match="weights"):
        select_prediction_ensemble(
            current_ml,
            {
                AO_POISSON_RHO0_CONTROL: current_ml,
                AO_POISSON_BLEND: current_ml,
            },
            np.array([0]),
            weights=(weight,),
        )
