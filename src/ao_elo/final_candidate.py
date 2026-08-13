from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from ao_elo.dynamic import DynamicEloConfig
from ao_elo.dynamic_csv import load_selected_v2_config
from ao_elo.xg_live import (
    XGBlendConfig,
    XGMatchEloUpdate,
    XGPerformanceBonusConfig,
    update_match_elo_with_xg,
)


FINAL_CANDIDATE_STATUS = "FINAL_MODEL_CANDIDATE"
FINAL_CANDIDATE_XG_FAMILY = "BOUNDED_TWO_SIDED_PERFORMANCE_ADJUSTMENT"


@dataclass(frozen=True)
class FinalCandidateRuntime:
    candidate_version: str
    contract_sha256: str
    dynamic_config: DynamicEloConfig
    xg_config: XGPerformanceBonusConfig


def load_final_candidate_runtime(path: str | Path) -> FinalCandidateRuntime:
    """Load and validate the selected goal-margin plus bounded-xG candidate."""
    contract_path = Path(path)
    raw = contract_path.read_bytes()
    payload = json.loads(raw)
    if payload.get("candidate_status") != FINAL_CANDIDATE_STATUS:
        raise ValueError("Candidate contract must have FINAL_MODEL_CANDIDATE status")
    if payload.get("production_activation") is not True:
        raise ValueError("Promoted final candidate must claim production activation")
    candidate_version = payload.get("candidate_version")
    if not isinstance(candidate_version, str) or not candidate_version.strip():
        raise ValueError("candidate_version must be a non-empty string")

    dynamic_config = load_selected_v2_config(contract_path)
    if not math.isclose(dynamic_config.goal_alpha, 0.15, abs_tol=1e-12):
        raise ValueError("Final candidate goal_alpha must equal 0.15")
    if not math.isclose(dynamic_config.goal_tau, 300.0, abs_tol=1e-12):
        raise ValueError("Final candidate goal_tau must equal 300")
    if dynamic_config.goal_difference_cap != 4:
        raise ValueError("Final candidate goal_difference_cap must equal 4")

    xg = payload.get("xg_performance")
    if not isinstance(xg, dict):
        raise ValueError("Final candidate contract requires xg_performance")
    if xg.get("active") is not True:
        raise ValueError("Final candidate xG performance must be active")
    if xg.get("family") != FINAL_CANDIDATE_XG_FAMILY:
        raise ValueError("Final candidate xG family is invalid")
    if xg.get("missing_xg_behavior") != "FALL_BACK_TO_GOAL_MARGIN_ONLY":
        raise ValueError("Final candidate must fall back to goal-margin-only Elo")
    if xg.get("draw_behavior") != "NO_XG_ADJUSTMENT":
        raise ValueError("Final candidate draws must not receive xG adjustment")
    if xg.get("penalty_shootout_behavior") != "NO_XG_ADJUSTMENT":
        raise ValueError("Final candidate shoot-outs must not receive xG adjustment")
    if xg.get("zero_sum") is not True:
        raise ValueError("Final candidate xG performance must remain zero-sum")
    winner_gain_bound = xg.get("winner_gain_bound")
    if not isinstance(winner_gain_bound, dict):
        raise ValueError("Final candidate requires winner_gain_bound metadata")
    if winner_gain_bound.get("source") != "ANALYTIC_XG_RATIO_BOUND":
        raise ValueError("Final candidate winner gain bound must be analytic")
    if winner_gain_bound.get("runtime_floor_expected_to_bind") is not False:
        raise ValueError("Final candidate floor must be analytically non-binding")

    max_ratio = _finite_number(xg.get("max_xg_ratio"), "max_xg_ratio")
    xg_scale = _finite_number(xg.get("xg_scale"), "xg_scale")
    minimum_ratio = _finite_number(
        xg.get("minimum_winner_gain_ratio"), "minimum_winner_gain_ratio"
    )
    if not math.isclose(max_ratio, 0.30, abs_tol=1e-12):
        raise ValueError("Final candidate max_xg_ratio must equal 0.30")
    if not math.isclose(xg_scale, 1.25, abs_tol=1e-12):
        raise ValueError("Final candidate xg_scale must equal 1.25")
    if not math.isclose(minimum_ratio, 1.0 - max_ratio, abs_tol=1e-12):
        raise ValueError("minimum_winner_gain_ratio must equal 1-max_xg_ratio")
    if not math.isclose(
        _finite_number(winner_gain_bound.get("minimum_ratio"), "winner_gain_bound.minimum_ratio"),
        minimum_ratio,
        abs_tol=1e-12,
    ):
        raise ValueError("winner_gain_bound.minimum_ratio must equal minimum_winner_gain_ratio")
    xg_config = XGPerformanceBonusConfig(max_ratio, xg_scale, minimum_ratio)
    xg_config.validate()
    return FinalCandidateRuntime(
        candidate_version=candidate_version,
        contract_sha256=hashlib.sha256(raw).hexdigest(),
        dynamic_config=dynamic_config,
        xg_config=xg_config,
    )


def update_final_candidate_match(
    runtime: FinalCandidateRuntime,
    home_rating: float,
    away_rating: float,
    home_goals: int,
    away_goals: int,
    *,
    is_neutral: bool,
    decided_on_penalties: bool,
    xg_home: float | None = None,
    xg_away: float | None = None,
) -> XGMatchEloUpdate:
    """Apply the frozen final-candidate update, with deterministic xG fallback."""
    config = runtime.dynamic_config
    return update_match_elo_with_xg(
        home_rating,
        away_rating,
        home_goals,
        away_goals,
        k_factor=config.k_factor,
        elo_scale=config.elo_scale,
        home_advantage=config.home_advantage,
        is_neutral=is_neutral,
        decided_on_penalties=decided_on_penalties,
        goal_difference_enabled=config.goal_difference_enabled,
        goal_alpha=config.goal_alpha,
        goal_tau=config.goal_tau,
        goal_difference_cap=config.goal_difference_cap,
        xg_config=XGBlendConfig(0.0, 1.0),
        xg_home=xg_home,
        xg_away=xg_away,
        xg_performance_bonus_config=runtime.xg_config,
    )


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result
