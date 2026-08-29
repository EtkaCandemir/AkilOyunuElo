from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ao_elo.ml_features import (
    FEATURE_SCHEMAS,
    build_pre_match_feature_store,
    build_domestic_snapshots,
    validate_feature_store,
)
from ao_elo.ml_prediction import (
    blend_probabilities,
    fit_ml_1x2,
    multiclass_losses,
    predict_ml_1x2,
)
from scripts.run_thesportsdb_stats_quality_pilot import stratified_sample


def test_overlapping_metadata_is_coalesced_without_feature_loss() -> None:
    baseline = _baseline_fixture().assign(round="Round of 16", round_sequence=6, leg_number=2, is_knockout=True)
    metadata = baseline[["match_id", "round", "round_sequence", "leg_number", "is_knockout"]].copy()
    expected = build_pre_match_feature_store(baseline, _empty_domestic())
    result = build_pre_match_feature_store(baseline, _empty_domestic(), match_metadata=metadata)
    pd.testing.assert_frame_equal(result, expected)
    baseline["round"] = None
    pd.testing.assert_frame_equal(build_pre_match_feature_store(baseline, _empty_domestic(), match_metadata=metadata), expected)


@pytest.mark.parametrize("column,value", [("round", "Final"), ("round_sequence", 7), ("leg_number", 1), ("is_knockout", False)])
def test_conflicting_metadata_is_rejected(column, value) -> None:
    baseline = _baseline_fixture().assign(round="Round of 16", round_sequence=6, leg_number=2, is_knockout=True)
    metadata = baseline[["match_id", column]].copy()
    metadata[column] = value
    with pytest.raises(ValueError, match="Conflicting match metadata"):
        build_pre_match_feature_store(baseline, _empty_domestic(), match_metadata=metadata)


@pytest.mark.parametrize("flag", [False, True])
def test_string_boolean_flags_match_boolean_features(flag) -> None:
    baseline = _baseline_fixture().assign(is_single_match_tie=flag, is_neutral=flag, is_knockout=flag)
    expected = build_pre_match_feature_store(baseline, _empty_domestic())
    strings = baseline.assign(is_single_match_tie=str(flag).lower(), is_neutral=str(flag).lower(), is_knockout=str(flag).lower())
    pd.testing.assert_frame_equal(build_pre_match_feature_store(strings, _empty_domestic()), expected)


@pytest.mark.parametrize("column", ["is_single_match_tie", "is_neutral", "is_knockout"])
def test_invalid_model_boolean_is_rejected(column) -> None:
    baseline = _baseline_fixture()
    baseline[column] = "sometimes"
    with pytest.raises(ValueError, match="boolean"):
        build_pre_match_feature_store(baseline, _empty_domestic())


def test_feature_inputs_and_store_reject_naive_kickoff() -> None:
    baseline = _baseline_fixture()
    valid = build_pre_match_feature_store(baseline, _empty_domestic())
    baseline["kickoff_utc"] = "2020-01-01 12:00:00"
    with pytest.raises(ValueError, match="timezone-aware"):
        build_pre_match_feature_store(baseline, _empty_domestic())
    valid["kickoff_utc"] = "2020-01-01 12:00:00"
    with pytest.raises(ValueError, match="timezone-aware"):
        validate_feature_store(valid, expected_rows=3)
    domestic = _empty_domestic()
    domestic.loc[0] = ["d1", "L", "2020-01-01 12:00:00", "A", "B", 1, 0, "CLUB-A", "CLUB-B"]
    with pytest.raises(ValueError, match="timezone-aware"):
        build_domestic_snapshots(domestic)


def _empty_domestic() -> pd.DataFrame:
    return pd.DataFrame(columns=["source_event_id", "sportsdb_league_id", "kickoff_utc", "home_source_team_id", "away_source_team_id", "home_goals", "away_goals", "home_ao_club_id", "away_ao_club_id"])


def test_feature_store_is_causal_with_same_kickoff_batch() -> None:
    baseline = _baseline_fixture()
    original = copy.deepcopy(baseline)
    domestic = pd.DataFrame(
        [
            {
                "source_event_id": "d1",
                "sportsdb_league_id": "100",
                "kickoff_utc": "2019-12-31T12:00:00Z",
                "home_source_team_id": "10",
                "away_source_team_id": "11",
                "home_goals": 2,
                "away_goals": 0,
                "home_ao_club_id": "CLUB-A",
                "away_ao_club_id": pd.NA,
            }
        ]
    )
    metadata = pd.DataFrame(
        {
            "match_id": ["m1", "m2", "m3"],
            "round": ["R", "R", "R"],
            "round_sequence": [1, 1, 2],
            "leg_number": [1, 1, 2],
            "is_knockout": [True, True, True],
        }
    )
    features = build_pre_match_feature_store(
        baseline,
        domestic,
        match_metadata=metadata,
    )
    validate_feature_store(features, expected_rows=3)
    first = features.set_index("match_id")
    assert first.loc["m1", "home_euro_matches_pre"] == 0
    assert first.loc["m2", "home_euro_matches_pre"] == 0
    assert first.loc["m3", "home_euro_matches_pre"] == 2
    assert first.loc["m1", "home_domestic_covered"] == 1
    assert first.loc["m1", "away_domestic_covered"] == 0
    pd.testing.assert_frame_equal(baseline, original)


def test_domestic_match_at_exact_kickoff_is_not_visible() -> None:
    baseline = _baseline_fixture().iloc[[0]].copy()
    domestic = pd.DataFrame(
        [
            {
                "source_event_id": "d1",
                "sportsdb_league_id": "100",
                "kickoff_utc": "2020-01-01T12:00:00Z",
                "home_source_team_id": "10",
                "away_source_team_id": "11",
                "home_goals": 2,
                "away_goals": 0,
                "home_ao_club_id": "CLUB-A",
                "away_ao_club_id": pd.NA,
            }
        ]
    )
    features = build_pre_match_feature_store(baseline, domestic)
    assert features.loc[0, "home_domestic_covered"] == 0


def test_blend_and_losses_are_normalized_and_direction_safe() -> None:
    ao = np.array([[0.60, 0.25, 0.15], [0.20, 0.30, 0.50]])
    ml = np.array([[0.50, 0.30, 0.20], [0.25, 0.25, 0.50]])
    np.testing.assert_allclose(blend_probabilities(ao, ml, 0.0), ao)
    np.testing.assert_allclose(blend_probabilities(ao, ml, 1.0), ml)
    mixed = blend_probabilities(ao, ml, 0.5)
    np.testing.assert_allclose(mixed.sum(axis=1), 1.0)
    losses = multiclass_losses(mixed, np.array([0, 2]))
    assert (losses[["brier_1x2", "log_loss_1x2"]].to_numpy() >= 0.0).all()


def test_logistic_and_histogram_models_return_three_probabilities() -> None:
    frame = _synthetic_feature_frame(120)
    for family, parameters in (
        ("LOGISTIC", {"C": 0.1, "l1_ratio": 0.25}),
        (
            "HIST_GRADIENT_BOOSTING",
            {
                "learning_rate": 0.03,
                "max_leaf_nodes": 7,
                "min_samples_leaf": 20,
                "l2_regularization": 1.0,
            },
        ),
    ):
        model = fit_ml_1x2(
            frame,
            arm_name="TEST",
            family=family,
            schema=FEATURE_SCHEMAS["HIST_GRADIENT_BOOSTING"],
            parameters=parameters,
        )
        probabilities = predict_ml_1x2(model, frame)
        assert probabilities.shape == (120, 3)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

        repeated_model = fit_ml_1x2(
            frame,
            arm_name="TEST",
            family=family,
            schema=FEATURE_SCHEMAS["HIST_GRADIENT_BOOSTING"],
            parameters=parameters,
        )
        np.testing.assert_allclose(
            probabilities,
            predict_ml_1x2(repeated_model, frame),
            atol=1e-12,
        )


def test_stats_pilot_sampling_is_deterministic_and_stratified() -> None:
    rows = []
    for country in ("ENG", "TUR"):
        for season in ("2014/15", "2019/20", "2024/25"):
            for index in range(20):
                rows.append(
                    {
                        "source_event_id": f"{country}-{season}-{index}",
                        "country_code": country,
                        "ao_season": season,
                        "sportsdb_league_id": country,
                    }
                )
    data = pd.DataFrame(rows)
    first = stratified_sample(data, 60)
    second = stratified_sample(data, 60)
    assert len(first) == 60
    assert first["source_event_id"].tolist() == second["source_event_id"].tolist()
    assert first.groupby(["country_code", "era"]).size().gt(0).all()


def _baseline_fixture() -> pd.DataFrame:
    common = {
        "season": "2020/21",
        "competition": "UCL",
        "stage": "QUALIFYING",
        "tie_id": "tie-1",
        "is_single_match_tie": False,
        "home_team_id": 1,
        "away_team_id": 2,
        "home_club_id": "CLUB-A",
        "away_club_id": "CLUB-B",
        "home_goals": 1,
        "away_goals": 0,
        "actual_class": 0,
        "actual_home_score": 1.0,
        "is_neutral": False,
        "home_live_pre": 1200.0,
        "away_live_pre": 1100.0,
        "expected_home_score": 0.60,
        "home_probability": 0.50,
        "draw_probability": 0.20,
        "away_probability": 0.30,
        "power_delta": 40.0,
    }
    rows = []
    for match_id, kickoff in (
        ("m1", "2020-01-01T12:00:00Z"),
        ("m2", "2020-01-01T12:00:00Z"),
        ("m3", "2020-01-02T12:00:00Z"),
    ):
        rows.append({"match_id": match_id, "kickoff_utc": kickoff, **common})
    return pd.DataFrame(rows)


def _synthetic_feature_frame(rows: int) -> pd.DataFrame:
    rng = np.random.default_rng(20260812)
    schema = FEATURE_SCHEMAS["HIST_GRADIENT_BOOSTING"]
    data: dict[str, object] = {}
    for column in schema.numeric:
        data[column] = rng.normal(size=rows)
    for column in schema.categorical:
        data[column] = rng.choice(["A", "B", "C"], size=rows)
    data["actual_class"] = np.arange(rows) % 3
    data["kickoff_utc"] = pd.date_range("2020-01-01", periods=rows, freq="D", tz="UTC")
    data["match_id"] = [f"m-{index}" for index in range(rows)]
    return pd.DataFrame(data)
