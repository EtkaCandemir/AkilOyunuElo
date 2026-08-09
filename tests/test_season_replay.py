import math

import pandas as pd
import pytest

from ao_elo.season_replay import (
    ReplayArmSpec,
    classify_shadow_signal,
    prediction_metrics,
    same_season_ranking,
    stable_config_fingerprint,
)


def test_replay_arm_fingerprint_is_deterministic_and_parameter_sensitive() -> None:
    arm = ReplayArmSpec("PRODUCTION", "RETROSPECTIVE_COUNTERFACTUAL", 0.1, 300.0)
    assert arm.config_fingerprint == arm.config_fingerprint
    changed = ReplayArmSpec("PRODUCTION", "RETROSPECTIVE_COUNTERFACTUAL", 0.2, 300.0)
    assert changed.config_fingerprint != arm.config_fingerprint


def test_stable_config_fingerprint_ignores_mapping_order() -> None:
    assert stable_config_fingerprint({"alpha": 0.1, "tau": 300}) == stable_config_fingerprint(
        {"tau": 300, "alpha": 0.1}
    )


def test_prediction_metrics_uses_standard_multiclass_losses() -> None:
    frame = pd.DataFrame(
        {
            "home_probability": [0.7],
            "draw_probability": [0.2],
            "away_probability": [0.1],
            "actual_class": [0],
        }
    )
    metrics = prediction_metrics(frame)
    assert metrics["brier_1x2"] == pytest.approx(0.14)
    assert metrics["log_loss_1x2"] == pytest.approx(-math.log(0.7))


def test_same_season_ranking_reports_all_and_competition() -> None:
    ratings = pd.DataFrame({"team_id": [1, 2, 3], "end_live_rating": [3.0, 2.0, 1.0]})
    target = pd.DataFrame(
        {
            "season": ["2025/26"] * 3,
            "competition": ["UCL"] * 3,
            "team_id": [1, 2, 3],
            "schedule_adjusted_score": [0.8, 0.5, 0.2],
        }
    )
    result = same_season_ranking(ratings, target, season="2025/26")
    assert set(result["competition"]) == {"ALL", "UCL"}
    assert result.loc[result["competition"].eq("ALL"), "teams"].iloc[0] == 3
    assert result["ranking_score"].eq(1.0).all()
    assert result["pairwise_accuracy"].eq(1.0).all()


@pytest.mark.parametrize(
    ("brier", "log_loss", "no_harm", "reliable_harm", "expected"),
    [
        (-0.001, -0.002, True, False, "CONSISTENT_SHADOW_SIGNAL"),
        (-0.001, 0.001, True, False, "MIXED_OR_INCONCLUSIVE"),
        (0.001, 0.002, True, False, "HARM_SIGNAL"),
        (-0.001, -0.002, True, True, "HARM_SIGNAL"),
    ],
)
def test_shadow_classification(brier, log_loss, no_harm, reliable_harm, expected) -> None:
    assert classify_shadow_signal(brier, log_loss, no_harm, reliable_harm) == expected
