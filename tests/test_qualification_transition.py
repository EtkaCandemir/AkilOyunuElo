from __future__ import annotations

import pytest

from ao_elo.qualification_stage_k import QualificationStageKConfig, reference_config
from ao_elo.qualification_transition import (
    QualificationTransitionConfig,
    apply_qualifier_carry,
)


def test_half_carry_is_symmetric_for_positive_and_negative_changes() -> None:
    gain = apply_qualifier_carry(1200.0, 1400.0, 0.50)
    loss = apply_qualifier_carry(1200.0, 1050.0, 0.50)
    assert gain.post_carry_rating == pytest.approx(1300.0)
    assert gain.carried_qualifier_change == pytest.approx(100.0)
    assert loss.post_carry_rating == pytest.approx(1125.0)
    assert loss.carried_qualifier_change == pytest.approx(-75.0)


def test_full_carry_is_identity_and_zero_carry_returns_anchor() -> None:
    full = apply_qualifier_carry(1200.0, 1375.0, 1.0)
    zero = apply_qualifier_carry(1200.0, 1375.0, 0.0)
    assert full.post_carry_rating == pytest.approx(1375.0)
    assert full.carry_adjustment == pytest.approx(0.0)
    assert zero.post_carry_rating == pytest.approx(1200.0)


def test_transition_reference_must_be_full_k_and_full_carry() -> None:
    valid = QualificationTransitionConfig(
        "REFERENCE", reference_config(), 1.0, selectable=False
    )
    valid.validate()
    with pytest.raises(ValueError):
        QualificationTransitionConfig(
            "BAD", reference_config(), 0.5, selectable=False
        ).validate()


def test_transition_accepts_selectable_mild_stage_profile() -> None:
    stage = QualificationStageKConfig("MILD", 0.7, 0.8, 0.9, 0.95)
    config = QualificationTransitionConfig("MILD_CARRY_050", stage, 0.5)
    config.validate()
    assert len(config.config_id) == 16
