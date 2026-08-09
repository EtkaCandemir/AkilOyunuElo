from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from scripts.run_relative_opponent_profile_backtest import (
    BASELINE_KEY,
    Candidate,
    DOMESTIC_ARM,
    EUROPE_ONLY_ARM,
    candidate_grid,
    candidate_predictions,
    development_raw_config_limit,
    raw_config_grid,
)


def test_relative_profile_grid_covers_both_evidence_arms() -> None:
    candidates = candidate_grid()
    assert len(candidates) == 361
    assert candidates[0].key == BASELINE_KEY
    assert sum(candidate.arm == EUROPE_ONLY_ARM for candidate in candidates) == 72
    assert sum(candidate.arm == DOMESTIC_ARM for candidate in candidates) == 288


def test_development_limit_keeps_europe_and_domestic_configs() -> None:
    limited = development_raw_config_limit(raw_config_grid(candidate_grid()), 2)
    assert len(limited) == 2
    assert limited[0].domestic_weight == 0.0
    assert limited[1].domestic_weight > 0.0


def test_candidate_prediction_uses_candidate_offset_not_baseline_audit_offset() -> None:
    import pandas as pd

    from ao_elo.relative_opponent_profile import RelativeOpponentProfileConfig

    base = pd.DataFrame(
        {
            "match_id": ["M-1"],
            "kickoff_utc": ["2025-01-01T20:00:00Z"],
            "expected_home_score": [0.5],
            "context_expected_home_score": [0.5],
            "home_team_selected_effect": [0.0],
            "away_team_selected_effect": [0.0],
            "applied_matchup_offset": [0.0],
            "context_cap_hit": [False],
            "actual_class": [0],
        }
    )
    features = pd.DataFrame(
        {
            "match_id": ["M-1"],
            "home_team_selected_effect": [20.0],
            "away_team_selected_effect": [0.0],
            "applied_matchup_offset": [20.0],
            "context_cap_hit": [False],
        }
    )
    config = RelativeOpponentProfileConfig(0.35, 0.65, 3, 0.75, 0.25, 20.0, 25.0)
    prediction = candidate_predictions(
        base,
        features,
        Candidate(config.key, DOMESTIC_ARM, config),
        draw_at_even=0.24,
        draw_shape=1.0,
        elo_scale=835.561497,
    )
    assert prediction.loc[0, "context_expected_home_score"] > 0.5
