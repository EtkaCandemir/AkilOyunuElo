from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ao_elo.controlled_live import update_match_elo  # noqa: E402
from ao_elo.dynamic_k import (  # noqa: E402
    DynamicKConfig,
    aggregate_match_k,
    baseline_dynamic_k_config,
    calculate_dynamic_match_k,
    calculate_team_k,
    calculate_team_uncertainty,
)
from scripts.run_dynamic_k_backtest import (  # noqa: E402
    add_evidence_bands,
    candidate_grid,
    elo_field_score,
    select_candidate,
)


BASE_K = 103.98098633392752


def dynamic_config(
    *,
    lambda_factor: float = 0.50,
    max_k_multiplier: float = 1.50,
    aggregation: str = "ARITHMETIC",
    inactivity_days: float = 270.0,
    match_evidence_scale: float = 10.0,
) -> DynamicKConfig:
    return DynamicKConfig(
        lambda_factor,
        max_k_multiplier,
        aggregation,
        inactivity_days,
        match_evidence_scale,
    )


def test_candidate_grid_has_one_baseline_and_162_dynamic_candidates() -> None:
    candidates = candidate_grid()

    assert len(candidates) == 163
    assert sum(candidate.lambda_factor == 0.0 for candidate in candidates) == 1
    assert baseline_dynamic_k_config() in candidates


def test_uncertainty_decreases_with_exposure_and_match_evidence() -> None:
    config = dynamic_config()
    uncertain = calculate_team_uncertainty(0.10, 1.0, 270.0, config)
    established = calculate_team_uncertainty(1.00, 40.0, 10.0, config)

    assert uncertain.combined > established.combined
    assert uncertain.exposure > established.exposure
    assert uncertain.match_evidence > established.match_evidence
    assert uncertain.inactivity > established.inactivity


def test_unknown_previous_date_has_maximum_inactivity_uncertainty() -> None:
    config = dynamic_config()

    uncertainty = calculate_team_uncertainty(0.5, 10.0, None, config)

    assert uncertainty.inactivity == pytest.approx(1.0)
    assert 0.0 <= uncertainty.combined <= 1.0


def test_lambda_zero_reproduces_fixed_k_exactly() -> None:
    config = baseline_dynamic_k_config()
    result = calculate_dynamic_match_k(
        base_k=BASE_K,
        home_exposure=0.0,
        away_exposure=1.0,
        home_prior_matches=0.0,
        away_prior_matches=75.0,
        home_days_since_last_match=None,
        away_days_since_last_match=1.0,
        config=config,
    )

    assert result.home_k == pytest.approx(BASE_K)
    assert result.away_k == pytest.approx(BASE_K)
    assert result.match_k == pytest.approx(BASE_K)


def test_team_k_respects_configured_cap() -> None:
    config = dynamic_config(lambda_factor=0.50, max_k_multiplier=1.25)

    value = calculate_team_k(BASE_K, 1.0, config)

    assert value == pytest.approx(BASE_K * 1.25)


def test_arithmetic_and_geometric_aggregation_are_deterministic() -> None:
    arithmetic = aggregate_match_k(100.0, 144.0, "ARITHMETIC")
    geometric = aggregate_match_k(100.0, 144.0, "GEOMETRIC")

    assert arithmetic == pytest.approx(122.0)
    assert geometric == pytest.approx(120.0)
    assert geometric < arithmetic


def test_dynamic_match_k_keeps_the_power_update_zero_sum() -> None:
    config = dynamic_config()
    dynamic = calculate_dynamic_match_k(
        base_k=BASE_K,
        home_exposure=0.2,
        away_exposure=0.9,
        home_prior_matches=2.0,
        away_prior_matches=30.0,
        home_days_since_last_match=300.0,
        away_days_since_last_match=20.0,
        config=config,
    )
    update = update_match_elo(
        1500.0,
        1400.0,
        3,
        0,
        k_factor=dynamic.match_k,
        elo_scale=835.5614973262034,
        home_advantage=148.54426619132505,
        is_neutral=False,
        decided_on_penalties=False,
        alpha=0.10,
        tau=300.0,
    )

    assert update.zero_sum_error <= 1e-9
    assert update.home_rating_post + update.away_rating_post == pytest.approx(
        2900.0
    )
    assert dynamic.match_k > BASE_K


def test_penalty_decision_preserves_the_elo_field_score() -> None:
    assert elo_field_score(1, 0, True) == (1, 0)
    assert elo_field_score(3, 1, False) == (3, 1)


def test_evidence_bands_use_mean_pre_match_exposure() -> None:
    predictions = pd.DataFrame(
        {
            "home_exposure": [0.0, 0.6, 1.0],
            "away_exposure": [0.4, 0.8, 0.8],
        }
    )

    result = add_evidence_bands(predictions)

    assert result["evidence_band"].tolist() == ["LOW", "MEDIUM", "HIGH"]


def test_candidate_selection_rejects_ranking_regression() -> None:
    baseline = baseline_dynamic_k_config()
    challenger = dynamic_config()
    common = {
        "brier_1x2": 0.16,
        "log_loss_1x2": 0.60,
        "pairwise_accuracy": 0.60,
        "minimum_start_end_rank_correlation": 0.90,
        "maximum_abs_rating_change": 100.0,
        "maximum_total_elo_error": 0.0,
        "matches": 100,
        "accuracy_1x2": 0.50,
        "maximum_abs_match_delta": 50.0,
        "maximum_match_k": BASE_K,
        "mean_match_k": BASE_K,
        "mean_team_uncertainty": 0.3,
    }
    frame = pd.DataFrame(
        [
            {
                "candidate_key": baseline.key,
                "lambda_factor": baseline.lambda_factor,
                "max_k_multiplier": baseline.max_k_multiplier,
                "aggregation": baseline.aggregation,
                "inactivity_days": baseline.inactivity_days,
                "match_evidence_scale": baseline.match_evidence_scale,
                "complexity": baseline.complexity,
                "ranking_score": 0.50,
                **common,
            },
            {
                "candidate_key": challenger.key,
                "lambda_factor": challenger.lambda_factor,
                "max_k_multiplier": challenger.max_k_multiplier,
                "aggregation": challenger.aggregation,
                "inactivity_days": challenger.inactivity_days,
                "match_evidence_scale": challenger.match_evidence_scale,
                "complexity": challenger.complexity,
                "ranking_score": 0.49,
                "brier_1x2": 0.15,
                **{
                    key: value
                    for key, value in common.items()
                    if key != "brier_1x2"
                },
            },
        ]
    )

    selected = select_candidate(frame)

    assert selected["candidate_key"] == baseline.key


@pytest.mark.parametrize(
    "config",
    [
        DynamicKConfig(-0.1, 1.25, "ARITHMETIC", 180.0, 6.0),
        DynamicKConfig(0.1, 0.99, "ARITHMETIC", 180.0, 6.0),
        DynamicKConfig(0.1, 1.25, "UNKNOWN", 180.0, 6.0),
        DynamicKConfig(0.1, 1.25, "ARITHMETIC", 0.0, 6.0),
        DynamicKConfig(0.1, 1.25, "ARITHMETIC", 180.0, math.inf),
    ],
)
def test_invalid_dynamic_k_configs_are_rejected(config: DynamicKConfig) -> None:
    with pytest.raises(ValueError):
        config.validate()
