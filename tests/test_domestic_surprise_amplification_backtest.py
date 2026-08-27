from __future__ import annotations

"""Guard the gates that decide whether the amplification may be promoted.

Three of these gates were wrong or missing when the study first ran, and each
failure pointed the same way - toward promoting a layer that the evidence does
not support:

1. The ranking veto used the loss sign convention. Ranking scores are
   higher-is-better, so a reliable degradation was reported as safe.
2. Nothing checked whether the folds agreed on a candidate. A surface of 206
   configurations always yields a per-fold winner; if the winners differ there
   is no parameter set to ship.
3. The control-artifact check compared a rebuild against an artifact written
   under the previous European exposure cap, so it failed for a reason that
   has nothing to do with the candidate.

The evidence tests at the end pin the corrected outcome so a later edit cannot
quietly restore the old verdict.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_domestic_surprise_amplification_backtest import (
    MINIMUM_MODAL_FOLD_SHARE,
    season_block_ranking_uncertainty,
    selection_stability,
)


RUN = ROOT / "output" / "domestic_surprise_amplification_backtest_2018_2026_rebuilt_065"
RANKING_METRICS = (
    "same_season_spearman",
    "same_season_pairwise_accuracy",
    "forward_spearman",
    "forward_pairwise_accuracy",
)

requires_run = pytest.mark.skipif(
    not (RUN / "selected_candidate.json").is_file(),
    reason="the rebuilt-baseline amplification run is not present in this checkout",
)


def ranking_frame(**columns: float) -> pd.DataFrame:
    """Six season rows, one constant delta per metric unless overridden."""
    data = {metric: np.full(6, columns.get(metric, 0.0)) for metric in RANKING_METRICS}
    frame = pd.DataFrame(data)
    frame.insert(0, "fold", range(1, 7))
    return frame


def selection_frame(candidates: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for index, candidate in enumerate(candidates, start=1):
        rows.append(
            {
                "fold": index,
                "selected_candidate": json.dumps(candidate, sort_keys=True),
                **candidate,
            }
        )
    return pd.DataFrame(rows)


def candidate(theta: float, cap: float, penalty: float, family: str) -> dict[str, object]:
    return {
        "theta": theta,
        "domestic_cap": cap,
        "variance_penalty": penalty,
        "exposure_family": family,
    }


# ---------------------------------------------------------------------------
# 1. the ranking veto must read the ranking sign convention
# ---------------------------------------------------------------------------


def test_a_consistently_negative_ranking_delta_is_reliable_harm() -> None:
    """Higher Spearman is better, so a negative delta is the harm.

    Reading this with the loss convention (`lower > 0`) reports the exact case
    the veto exists to catch as safe.
    """
    result = season_block_ranking_uncertainty(
        ranking_frame(same_season_spearman=-0.002), samples=2000
    ).set_index("metric")
    row = result.loc["same_season_spearman"]

    assert row["ci_95_upper"] < 0.0
    assert bool(row["reliable_harm"]) is True
    assert bool(row["reliable_improvement"]) is False


def test_a_consistently_positive_ranking_delta_is_reliable_improvement() -> None:
    result = season_block_ranking_uncertainty(
        ranking_frame(forward_spearman=0.002), samples=2000
    ).set_index("metric")
    row = result.loc["forward_spearman"]

    assert row["ci_95_lower"] > 0.0
    assert bool(row["reliable_improvement"]) is True
    assert bool(row["reliable_harm"]) is False


def test_a_delta_that_straddles_zero_is_neither() -> None:
    frame = ranking_frame()
    frame["same_season_spearman"] = [0.01, -0.01, 0.008, -0.009, 0.002, -0.003]
    row = season_block_ranking_uncertainty(frame, samples=2000).set_index("metric").loc[
        "same_season_spearman"
    ]

    assert row["ci_95_lower"] < 0.0 < row["ci_95_upper"]
    assert bool(row["reliable_harm"]) is False
    assert bool(row["reliable_improvement"]) is False


def test_every_ranking_metric_is_evaluated() -> None:
    result = season_block_ranking_uncertainty(ranking_frame(), samples=500)

    assert list(result["metric"]) == list(RANKING_METRICS)


# ---------------------------------------------------------------------------
# 2. selection stability
# ---------------------------------------------------------------------------


def test_folds_that_all_agree_are_perfectly_stable() -> None:
    agreed = candidate(0.6, 150.0, 0.0, "LINEAR")
    stability = selection_stability(selection_frame([agreed] * 6))

    assert stability["distinct_candidates"] == 1
    assert stability["modal_fold_share"] == 1.0
    assert stability["modal_fold_share"] >= MINIMUM_MODAL_FOLD_SHARE


def test_folds_that_all_disagree_fail_the_gate() -> None:
    """This is the observed shape of the real run, not a hypothetical."""
    stability = selection_stability(
        selection_frame(
            [
                candidate(0.8, 100.0, 0.50, "FLOOR_50"),
                candidate(1.75, 30.0, 0.75, "POWER_050"),
                candidate(0.4, 60.0, 0.00, "LINEAR"),
                candidate(0.6, 75.0, 0.25, "LINEAR"),
                candidate(0.6, 150.0, 0.00, "LINEAR"),
                candidate(0.6, 150.0, 0.00, "LINEAR"),
            ]
        )
    )

    assert stability["distinct_candidates"] == 5
    assert stability["modal_fold_share"] == pytest.approx(2 / 6)
    assert stability["modal_fold_share"] < MINIMUM_MODAL_FOLD_SHARE
    assert stability["theta_min"] == 0.4
    assert stability["theta_max"] == 1.75
    assert stability["distinct_exposure_families"] == 3


def test_stability_reports_the_span_of_every_tuned_parameter() -> None:
    stability = selection_stability(
        selection_frame(
            [candidate(0.4, 30.0, 0.0, "LINEAR"), candidate(2.0, 150.0, 0.75, "LINEAR")]
        )
    )

    assert (stability["theta_min"], stability["theta_max"]) == (0.4, 2.0)
    assert (stability["domestic_cap_min"], stability["domestic_cap_max"]) == (30.0, 150.0)
    assert (
        stability["variance_penalty_min"],
        stability["variance_penalty_max"],
    ) == (0.0, 0.75)


def test_stability_needs_at_least_one_fold() -> None:
    with pytest.raises(ValueError, match="at least one fold"):
        selection_stability(selection_frame([]))


# ---------------------------------------------------------------------------
# 3. the corrected verdict of the real run
# ---------------------------------------------------------------------------


@pytest.fixture(name="decision", scope="module")
def fixture_decision() -> dict:
    return json.loads((RUN / "selected_candidate.json").read_text(encoding="utf-8"))


@requires_run
def test_the_run_changed_no_production_parameter(decision: dict) -> None:
    assert decision["production_change"] is False


@requires_run
def test_ranking_harm_is_reported_and_vetoes_promotion(decision: dict) -> None:
    """Both same-season metrics degrade with a CI that excludes zero."""
    assert decision["ranking_reliable_harm"] is True
    assert decision["ranking_harmed_metrics"] == [
        "same_season_pairwise_accuracy",
        "same_season_spearman",
    ]
    assert decision["decision"] == "KEEP_CURRENT"


@requires_run
def test_the_folds_never_agreed_on_a_candidate(decision: dict) -> None:
    stability = decision["selection_stability"]

    assert decision["selection_is_stable"] is False
    assert stability["distinct_candidates"] > stability["folds"] / 2
    assert stability["theta_max"] / stability["theta_min"] > 4.0


@requires_run
def test_the_pooled_gain_is_indistinguishable_from_zero(decision: dict) -> None:
    """Roughly two hundred times smaller than the exposure-cap change it sits on."""
    assert abs(decision["pooled_brier_difference"]) < 1e-4
    assert decision["loss_conservative_ci_reliable_improvement"] is False


@requires_run
def test_the_control_reconciles_where_the_contract_did_not_move(decision: dict) -> None:
    """Below the exposure cap the domestic weight is untouched by the migration."""
    assert decision["rebuilt_control_matches_stored_artifact_below_exposure_cap"] is True
    assert decision["rebuilt_control_max_effect_difference_below_cap"] < 1e-9


@requires_run
def test_the_migration_delta_is_confined_to_capped_exposure(decision: dict) -> None:
    """`(1-0.65) - (1-0.85) = 0.20`, and `0.20 * 30 = 6.0` at the domestic cap."""
    assert decision["contract_migration_delta_confined_to_capped_exposure"] is True
    assert decision["contract_migration_max_effect_difference"] == pytest.approx(
        (0.35 - 0.15) * 30.0, abs=1e-6
    )


@requires_run
def test_the_safety_audit_records_both_failing_gates() -> None:
    audit = pd.read_csv(RUN / "safety_audit.csv").set_index("check")

    assert audit.loc["ranking_not_reliably_harmed", "passed"] == False  # noqa: E712
    assert audit.loc["fold_selection_modal_share_at_least_half", "passed"] == False  # noqa: E712
    assert audit.loc["rebuilt_control_matches_stored_artifact_below_exposure_cap", "passed"] == True  # noqa: E712
    assert audit.loc["contract_migration_delta_confined_to_capped_exposure", "passed"] == True  # noqa: E712
