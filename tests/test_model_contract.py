from __future__ import annotations

import json
from pathlib import Path

import pytest

from ao_elo.config import AO_MODEL_V2_VERSION, AOEuropeanEloConfig
from ao_elo.dynamic import DynamicEloConfig
from ao_elo.dynamic_csv import (
    FIXTURE_INPUT_COLUMNS,
    MATCH_INPUT_COLUMNS,
    MATCH_UPDATE_COLUMNS,
    PRE_MATCH_LOG_COLUMNS,
    RATINGS_STATE_COLUMNS,
    REPLAY_PREDICTION_COLUMNS,
)
from ao_elo.production_prediction import PRODUCTION_PREDICTION_LOG_COLUMNS


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts" / "ao_european_elo_v2.json"
MANIFEST_ROOTS = {
    "ranking": ROOT
    / "output"
    / "v2_ranking_calibration_2018_2026"
    / "selected_model.json",
    "dynamic": ROOT
    / "output"
    / "v2_dynamic_calibration_2018_2026"
    / "selected_dynamic_model.json",
    "goal": ROOT
    / "output"
    / "v2_goal_margin_calibration_2018_2026"
    / "selected_goal_model.json",
    "reserve": ROOT
    / "output"
    / "v2_achievement_reserve_calibration_2018_2026"
    / "selected_reserve_model.json",
}
EVALUATION_MANIFEST = (
    ROOT
    / "output"
    / "v2_evaluation_upgrade_2018_2026"
    / "evaluation_manifest.json"
)
PRODUCTION_MANIFEST = (
    ROOT / "contracts" / "ao_european_elo_v2_production.json"
)
ROBUSTNESS_MANIFEST = (
    ROOT
    / "output"
    / "final_robustness_2018_2026"
    / "robustness_manifest.json"
)


def test_frozen_contract_matches_active_static_and_dynamic_config() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    static = AOEuropeanEloConfig.active()
    dynamic = DynamicEloConfig.calibrated_v2()

    assert payload["model_version"] == AO_MODEL_V2_VERSION
    assert static.model_version == payload["model_version"]
    assert dynamic.model_version == payload["model_version"]
    for key, expected in payload["static"].items():
        if key == "tail_decision":
            continue
        assert getattr(static, key) == pytest.approx(expected) if isinstance(
            expected, float
        ) else getattr(static, key) == expected
    for key in ("elo_scale", "home_advantage", "k_factor", "power_carry"):
        assert getattr(dynamic, key) == pytest.approx(payload["dynamic"][key])
    assert dynamic.draw_at_even == pytest.approx(
        payload["dynamic"]["one_x_two_probability"]["draw_at_even"]
    )
    assert dynamic.draw_shape == pytest.approx(
        payload["dynamic"]["one_x_two_probability"]["draw_shape"]
    )
    assert dynamic.goal_difference_enabled is True
    assert dynamic.goal_alpha == pytest.approx(
        payload["dynamic"]["goal_margin"]["alpha"]
    )
    assert dynamic.goal_tau == pytest.approx(
        payload["dynamic"]["goal_margin"]["tau"]
    )
    assert dynamic.goal_difference_cap == (
        payload["dynamic"]["goal_margin"]["goal_difference_cap"]
    )
    qualification = payload["dynamic"]["qualification_transition"]
    assert dynamic.qualification_q1_multiplier == pytest.approx(
        qualification["q1_multiplier"]
    )
    assert dynamic.qualification_q2_multiplier == pytest.approx(
        qualification["q2_multiplier"]
    )
    assert dynamic.qualification_q3_multiplier == pytest.approx(
        qualification["q3_multiplier"]
    )
    assert dynamic.qualification_playoff_multiplier == pytest.approx(
        qualification["qualifying_playoff_multiplier"]
    )
    assert dynamic.qualifier_to_main_carry == pytest.approx(
        qualification["qualifier_to_main_carry"]
    )
    assert dynamic.achievement_reserve is None


def test_frozen_contract_schemas_match_csv_engine() -> None:
    schemas = json.loads(CONTRACT.read_text(encoding="utf-8"))["schemas"]

    assert schemas["fixtures_csv"] == FIXTURE_INPUT_COLUMNS
    assert schemas["matches_csv"] == MATCH_INPUT_COLUMNS
    assert schemas["ratings_state_csv"] == RATINGS_STATE_COLUMNS
    assert schemas["match_updates_csv"] == MATCH_UPDATE_COLUMNS
    assert schemas["replay_predictions_csv"] == REPLAY_PREDICTION_COLUMNS
    assert schemas["pre_match_log_csv"] == PRE_MATCH_LOG_COLUMNS


def test_holdout_contract_forbids_parameter_selection() -> None:
    holdout = json.loads(CONTRACT.read_text(encoding="utf-8"))["holdout"]

    assert holdout["season"] == "2026/27"
    assert holdout["parameter_selection_allowed"] is False
    assert holdout["eligible_scope"] == "LEAGUE_PHASE_AND_LATER"
    assert holdout["qualifying_and_playoffs_included"] is False


def test_research_manifests_declare_the_active_exposure_cap() -> None:
    """Every static config a research arm can be seeded from must use 0.65.

    The canonical evaluation seeds its reference and no-surprise arms from the
    dynamic manifest while the production arm reads the rebuilt Domestic
    Surprise artifact. When the manifest lagged at the superseded 0.85 cap,
    949 of 1887 team-seasons differed between the two arms and the exposure
    decision was scored as a feature contribution.
    """

    production = json.loads(PRODUCTION_MANIFEST.read_text(encoding="utf-8"))
    contract_cap = production["static_initial_elo"]["max_european_exposure"]
    active_cap = AOEuropeanEloConfig.active().max_european_exposure

    assert contract_cap == pytest.approx(0.65)
    assert active_cap == pytest.approx(contract_cap)
    assert production["static_initial_elo"]["minimum_domestic_prior_weight"] == (
        pytest.approx(0.35)
    )

    for name in ("ranking", "dynamic"):
        manifest = json.loads(MANIFEST_ROOTS[name].read_text(encoding="utf-8"))
        static_payload = manifest.get("static_config") or manifest.get("config")
        assert static_payload is not None, name
        assert static_payload["max_european_exposure"] == pytest.approx(
            contract_cap
        ), name


def test_calibration_manifests_match_frozen_layer_decisions() -> None:
    manifests = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in MANIFEST_ROOTS.items()
    }

    assert all(
        manifest["model_version"] == AO_MODEL_V2_VERSION
        for manifest in manifests.values()
    )
    assert manifests["ranking"]["decision"] == "NO_PROMOTION"
    assert manifests["ranking"]["selected_candidate"] == "c0_e0_x0"
    assert manifests["dynamic"]["dynamic_core_decision"] == (
        "PROMOTE_DYNAMIC_CORE"
    )
    assert manifests["dynamic"]["carry_decision"] == "PROMOTE_CARRY"
    assert manifests["goal"]["decision"] == "DISABLE_GOAL_MARGIN"
    assert manifests["goal"]["goal_margin"]["goal_weight"] == 0.0
    assert manifests["reserve"]["decision"] == (
        "DISABLE_ACHIEVEMENT_RESERVE"
    )
    assert manifests["reserve"]["achievement_reserve"]["reserve_base"] == 0.0

    evaluation = json.loads(EVALUATION_MANIFEST.read_text(encoding="utf-8"))
    production = json.loads(PRODUCTION_MANIFEST.read_text(encoding="utf-8"))
    static = AOEuropeanEloConfig.active()
    assert evaluation["ranking_target"]["static_tail_decision"] == "NO_PROMOTION"
    assert evaluation["probability_contract"]["decision"] == "PROMOTE_1X2_OUTPUT"
    assert evaluation["layer_revalidation"]["dynamic_core"]["decision"] == (
        "CONFIRMED_1X2"
    )
    assert evaluation["layer_revalidation"]["season_carry"]["decision"] == (
        "DISABLE_CARRY_1X2"
    )
    assert production["model_version"] == AO_MODEL_V2_VERSION
    assert production["static_initial_elo"]["max_european_exposure"] == pytest.approx(
        static.max_european_exposure
    )
    assert production["static_initial_elo"]["minimum_domestic_prior_weight"] == (
        pytest.approx(1.0 - static.max_european_exposure)
    )
    assert production["active_power_carry"] == 0.0
    assert production["one_x_two_probability"]["draw_at_even"] == pytest.approx(
        0.24
    )
    assert production["one_x_two_probability"]["single_match_draw_enabled"] is True
    assert production["one_x_two_probability"][
        "single_match_draw_at_even"
    ] == pytest.approx(0.12)
    assert production["goal_margin"]["active"] is True
    assert production["goal_margin"]["alpha"] == pytest.approx(0.15)
    assert production["goal_margin"]["tau"] == pytest.approx(300.0)
    assert production["goal_margin"]["goal_difference_cap"] == 4
    assert production["xg_performance"]["active"] is True
    assert production["xg_performance"]["max_xg_ratio"] == pytest.approx(0.30)
    assert production["domestic_surprise"]["active"] is True
    assert production["domestic_surprise"]["coefficient"] == pytest.approx(0.40)
    assert production["domestic_surprise"]["variance_penalty"] == pytest.approx(0.50)
    assert production["domestic_surprise"]["max_abs_adjustment"] == pytest.approx(30.0)
    assert production["progression_bonus"]["active"] is True
    assert production["progression_bonus"]["base_bonus"] == pytest.approx(12.0)
    assert production["progression_bonus"]["increments"] == {
        "UCL": 12.0,
        "UEL": 8.0,
        "UECL": 4.0,
    }
    assert production["progression_bonus"]["season_caps"] == {
        "UCL": 48.0,
        "UEL": 32.0,
        "UECL": 16.0,
    }
    assert production["progression_bonus"]["stages_per_competition"] == 4
    assert production["progression_bonus"]["eligible_stages"] == [
        "ROUND_OF_16",
        "QUARTERFINAL",
        "SEMIFINAL",
        "FINAL",
    ]
    qualification = production["qualification_transition"]
    assert qualification["family"] == (
        "CONTINUOUS_STAGE_WEIGHTED_K_NO_MAIN_ENTRY_RESET"
    )
    assert qualification["qualifier_delta_retention"] == pytest.approx(0.50)
    assert qualification["stage_k_multipliers"] == {
        "Q1": 0.20,
        "Q2": 0.275,
        "Q3": 0.35,
        "QUALIFYING_PLAYOFF": 0.425,
        "MAIN": 1.0,
    }
    assert qualification["qualifier_to_main_carry"] == pytest.approx(1.0)
    assert qualification["non_match_rating_change"] is False


def test_historical_robustness_remains_immutable_after_manual_goal_decision() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    robustness = json.loads(ROBUSTNESS_MANIFEST.read_text(encoding="utf-8"))
    recommended = robustness["recommended_production_model"]

    assert contract["contract_version"] == "1.10.0"
    assert contract["prediction_layer"]["active"] is True
    assert contract["prediction_layer"]["decision"] == "PROMOTE_WITH_MONITORING"
    assert contract["prediction_layer"]["rating_feedback"] is False
    assert tuple(contract["schemas"]["production_prediction_log_csv"]) == (
        PRODUCTION_PREDICTION_LOG_COLUMNS
    )
    assert contract["evaluation"]["dynamic_ranking_same_season_reuse_allowed"] is False
    assert contract["evaluation"]["dynamic_forward_ranking_folds"] == 5
    assert robustness["decisions"] == {
        "goal_margin": "DISABLE_GOAL_MARGIN",
        "competition_k": "DISABLE_COMPETITION_K",
        "achievement_reserve": "DISABLE_ACHIEVEMENT_RESERVE",
        "external_clubelo": "UCL_CLUBELO_POINT_ESTIMATE_BETTER_INCONCLUSIVE",
    }
    assert recommended["active_power_carry"] == 0.0
    assert recommended["goal_margin"]["active"] is False
    assert recommended["competition_k"]["active"] is False
    assert recommended["achievement_reserve"]["active"] is False
    assert contract["dynamic"]["goal_margin"]["active"] is True
    assert contract["dynamic"]["progression_bonus"]["active"] is True
