from __future__ import annotations

"""Guard the walk-forward that answers the critical scoreline question.

Two kinds of test live here. The first kind pins the run's construction: which
folds exist, what the arms are allowed to see, and which gates have to hold for
the output to mean anything. The second kind pins the evidence the report
quotes, so a later change to the model or the data cannot leave the written
conclusion standing on numbers that no longer reproduce.

The evidence tests skip when the run has not been executed in this checkout.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_xg_domestic_goal_expectation_backtest import (
    MINIMUM_XG_TRAIN_SEASONS,
    XG_SEASONS,
    build_folds,
    fold_wins,
    load_matches,
    validation_gates,
)
from ao_elo.xg_domestic_goal_model import (
    DOMESTIC_AD,
    DOMESTIC_AD_GOALS_FORM,
    DOMESTIC_AD_XG_FORM,
)


OUTPUT = ROOT / "output" / "xg_domestic_goal_expectation_backtest_2020_2026"
SENSITIVITY = OUTPUT / "sensitivity_xg_training_window"
FEATURES = ROOT / "output" / "ml_1x2_backtest_2018_2026" / "pre_match_feature_store.csv"
XG_DATA = ROOT / "data" / "xg_2020_2026" / "uefa_2020_2026_matches_with_xg.csv"
ALL_SEASONS = (
    "2018/19", "2019/20", "2020/21", "2021/22",
    "2022/23", "2023/24", "2024/25", "2025/26",
)
EXPECTED_TEST_SEASONS = ["2022/23", "2023/24", "2024/25", "2025/26"]
EXPECTED_MATCHES = 3528

requires_run = pytest.mark.skipif(
    not (OUTPUT / "backtest_manifest.json").is_file(),
    reason="the domestic + form backtest has not been run in this checkout",
)
requires_inputs = pytest.mark.skipif(
    not (FEATURES.is_file() and XG_DATA.is_file()),
    reason="the feature store or the xG dataset is not built in this checkout",
)


def season_frame(seasons=ALL_SEASONS) -> pd.DataFrame:
    return pd.DataFrame({"season": list(seasons)})


@pytest.fixture(name="manifest", scope="module")
def fixture_manifest() -> dict:
    return json.loads((OUTPUT / "backtest_manifest.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# fold construction: never fit a coefficient the training half cannot support
# ---------------------------------------------------------------------------


def test_folds_start_only_once_two_xg_seasons_have_closed() -> None:
    folds = build_folds(season_frame(), "full")

    assert [test for _, _, test in folds] == EXPECTED_TEST_SEASONS
    for train, _, test in folds:
        covered = [season for season in train if season in XG_SEASONS]
        assert len(covered) >= MINIMUM_XG_TRAIN_SEASONS
        assert test not in train


def test_the_full_window_trains_on_every_earlier_season() -> None:
    folds = build_folds(season_frame(), "full")
    train, _, test = folds[0]

    assert train == ("2018/19", "2019/20", "2020/21", "2021/22")
    assert test == "2022/23"


def test_the_xg_window_trains_only_on_covered_seasons() -> None:
    folds = build_folds(season_frame(), "xg")
    train, _, test = folds[0]

    assert train == ("2020/21", "2021/22")
    assert test == "2022/23"
    assert all(
        season in XG_SEASONS for train, _, _ in folds for season in train
    )


def test_both_windows_score_the_identical_test_seasons() -> None:
    """The sensitivity run must differ in training only, never in what it scores."""
    full = [test for _, _, test in build_folds(season_frame(), "full")]
    covered = [test for _, _, test in build_folds(season_frame(), "xg")]

    assert full == covered == EXPECTED_TEST_SEASONS


def test_inner_validation_is_the_last_training_season() -> None:
    for train, validation, test in build_folds(season_frame(), "full"):
        assert validation == train[-1]
        assert validation < test


def test_a_history_too_short_for_xg_raises() -> None:
    with pytest.raises(ValueError, match="no fold satisfies"):
        build_folds(season_frame(("2018/19", "2019/20", "2020/21")), "full")


# ---------------------------------------------------------------------------
# input handling
# ---------------------------------------------------------------------------


@requires_inputs
def test_seasons_without_xg_are_kept_but_marked_ineligible() -> None:
    """Pre-2020/21 rows stay so the baseline can train on them.

    FotMob returns no xG field at all for those seasons. Marking them
    ineligible rather than dropping them keeps the baseline at full strength
    while stopping the xG history from advancing across rows it cannot read.
    """
    data = load_matches(FEATURES, XG_DATA)
    early = data.loc[~data["season"].isin(XG_SEASONS)]

    assert len(early) > 0
    assert not early["xg_analysis_eligible"].any()
    assert (early[["xg_home", "xg_away"]] == 0.0).all().all()
    assert data.loc[data["season"].isin(XG_SEASONS), "xg_analysis_eligible"].any()


@requires_inputs
def test_phase_splits_the_documented_coverage_gap() -> None:
    data = load_matches(FEATURES, XG_DATA)
    covered = data.loc[data["season"].isin(XG_SEASONS)]
    rates = covered.groupby("phase")["xg_analysis_eligible"].mean()

    assert rates["MAIN"] > 0.95
    assert rates["QUALIFYING"] < 0.25


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------


def gate_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    selections = pd.DataFrame(
        {"fold": [1, 1, 1], "domestic_candidate_key": ["k", "k", "k"]}
    )
    reconciliation = pd.DataFrame(
        [{"scope": "repo_arm_shared_sample", "difference_vs_this_baseline": 0.0001}]
    )
    folds = [(("2020/21", "2021/22"), "2021/22", "2022/23")]
    return validation_gates(predictions, selections, reconciliation, folds).set_index(
        "gate"
    )


def minimal_predictions() -> pd.DataFrame:
    rows = []
    for arm in (DOMESTIC_AD, DOMESTIC_AD_GOALS_FORM, DOMESTIC_AD_XG_FORM):
        for index in range(4):
            rows.append(
                {
                    "fold": 1,
                    "arm": arm,
                    "match_id": f"m{index}",
                    "lambda_home": 1.4,
                    "lambda_away": 1.1,
                    "xg_analysis_eligible": index < 2,
                    "form_attack": 0.0 if arm == DOMESTIC_AD else 0.12,
                    "form_defence": 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_gates_pass_on_a_well_formed_run() -> None:
    gates = gate_frame(minimal_predictions())

    assert gates["passed"].all()


def test_a_baseline_that_carries_form_fails_its_gate() -> None:
    predictions = minimal_predictions()
    predictions.loc[predictions["arm"].eq(DOMESTIC_AD), "form_attack"] = 0.2

    assert not gate_frame(predictions).loc["baseline_carries_no_form", "passed"]


def test_dropping_the_goals_control_fails_the_run() -> None:
    """Without the control the result is uninterpretable, so the gate must fail."""
    predictions = minimal_predictions()
    predictions = predictions.loc[~predictions["arm"].eq(DOMESTIC_AD_GOALS_FORM)]

    assert not gate_frame(predictions).loc["three_arms_scored", "passed"]


def test_a_baseline_that_drifts_from_the_published_arm_fails_its_gate() -> None:
    predictions = minimal_predictions()
    selections = pd.DataFrame({"fold": [1], "domestic_candidate_key": ["k"]})
    reconciliation = pd.DataFrame(
        [{"scope": "repo_arm_shared_sample", "difference_vs_this_baseline": -0.05}]
    )
    folds = [(("2020/21", "2021/22"), "2021/22", "2022/23")]
    gates = validation_gates(
        predictions, selections, reconciliation, folds
    ).set_index("gate")

    assert not gates.loc["baseline_matches_published_arm", "passed"]


def test_selection_that_reaches_into_the_test_season_fails_its_gate() -> None:
    leaky = [(("2020/21", "2022/23"), "2022/23", "2022/23")]
    gates = validation_gates(
        minimal_predictions(),
        pd.DataFrame({"fold": [1], "domestic_candidate_key": ["k"]}),
        pd.DataFrame(
            [{"scope": "repo_arm_shared_sample", "difference_vs_this_baseline": 0.0}]
        ),
        leaky,
    ).set_index("gate")

    assert not gates.loc["selection_never_sees_test_season", "passed"]


# ---------------------------------------------------------------------------
# fold accounting
# ---------------------------------------------------------------------------


def test_fold_wins_counts_both_the_baseline_and_the_control_comparison() -> None:
    folds_frame = pd.DataFrame(
        [
            {"arm": arm, "test_season": season, "exact_score_nll": value,
             "total_goals_error": value, "over_2_5_brier": value, "btts_brier": value}
            for season, values in (
                ("2022/23", (3.0, 2.9, 2.8)),
                ("2023/24", (3.0, 3.1, 3.2)),
            )
            for arm, value in zip(
                (DOMESTIC_AD, DOMESTIC_AD_GOALS_FORM, DOMESTIC_AD_XG_FORM), values
            )
        ]
    )
    wins = fold_wins(folds_frame).set_index(["arm", "reference", "metric"])

    assert wins.loc[
        (DOMESTIC_AD_XG_FORM, DOMESTIC_AD, "exact_score_nll"), "record"
    ] == "1/2"
    assert wins.loc[
        (DOMESTIC_AD_XG_FORM, DOMESTIC_AD_GOALS_FORM, "exact_score_nll"), "record"
    ] == "1/2"


# ---------------------------------------------------------------------------
# evidence the report quotes
# ---------------------------------------------------------------------------


@requires_run
def test_the_run_changed_no_production_parameter(manifest: dict) -> None:
    assert manifest["changes_production_parameters"] is False
    assert len(manifest["production_contract_sha256"]) == 64
    assert set(manifest["production_contract_sha256"]) <= set("0123456789abcdef")


@requires_run
def test_every_gate_passed(manifest: dict) -> None:
    assert manifest["all_gates_passed"] is True


@requires_run
def test_the_sample_is_the_one_the_earlier_run_used(manifest: dict) -> None:
    """Both runs score the same 3528 matches, so their arms line up directly."""
    assert manifest["matches_per_arm"] == EXPECTED_MATCHES
    assert manifest["test_seasons"] == EXPECTED_TEST_SEASONS
    assert manifest["folds"] == 4


@requires_run
def test_the_baseline_reproduces_the_published_domestic_arm(manifest: dict) -> None:
    shared = next(
        row
        for row in manifest["baseline_reconciliation"]
        if row["scope"] == "repo_arm_shared_sample"
    )

    assert abs(shared["difference_vs_this_baseline"]) < 0.001
    assert shared["matches"] == EXPECTED_MATCHES


@requires_run
def test_manifest_is_strict_json_without_non_finite_constants(manifest: dict) -> None:
    raw = (OUTPUT / "backtest_manifest.json").read_text(encoding="utf-8")
    full_sample = next(
        row
        for row in manifest["baseline_reconciliation"]
        if row["scope"] == "repo_arm_full_sample"
    )

    assert "NaN" not in raw
    assert "Infinity" not in raw
    assert full_sample["difference_vs_this_baseline"] is None


@requires_run
def test_xg_beats_both_the_baseline_and_the_control(manifest: dict) -> None:
    assert manifest["xg_beats_baseline_on_exact_score"] is True
    assert manifest["xg_beats_goals_on_exact_score"] is True


@requires_run
def test_the_reliable_gain_is_confined_to_where_xg_exists(manifest: dict) -> None:
    """Pooled `ALL` must not be claimed as reliable; only the xG segments are."""
    reliable = {
        (row["arm"], row["segment"], row["metric"])
        for row in manifest["reliable_improvements"]
    }

    assert ("DOMESTIC_AD_XG_FORM", "PHASE:MAIN", "exact_score_nll") in reliable
    assert ("DOMESTIC_AD_XG_FORM", "XG_PRESENT", "exact_score_nll") in reliable
    assert ("DOMESTIC_AD_XG_FORM", "ALL", "exact_score_nll") not in reliable
    assert not any(arm == "DOMESTIC_AD_GOALS_FORM" for arm, _, _ in reliable)


@requires_run
def test_no_segment_is_reliably_harmed(manifest: dict) -> None:
    assert manifest["reliable_harms"] == []


@requires_run
def test_the_head_to_head_isolates_the_source(manifest: dict) -> None:
    """Both candidate arms carry the same two parameters.

    A reliable gap between them therefore cannot be the parameters; it can only
    be the source the form term reads.
    """
    envelope = {row["segment"]: row for row in manifest["xg_vs_goals_envelope"]
                if row["metric"] == "exact_score_nll"}

    assert envelope["XG_PRESENT"]["reliable_improvement"] is True
    assert envelope["PHASE:MAIN"]["reliable_improvement"] is True
    assert envelope["ALL"]["reliable_improvement"] is False
    assert envelope["XG_ABSENT"]["reliable_improvement"] is False


@requires_run
def test_derivative_markets_still_do_not_clear(manifest: dict) -> None:
    """Over/under and BTTS remain at climatology; only the score shape moves."""
    assert not any(
        row["metric"] == "over_2_5_brier" for row in manifest["reliable_improvements"]
    )


@requires_run
def test_the_qualifying_segment_still_costs_the_layer(manifest: dict) -> None:
    """xG is 11% covered in qualifying, so the term must stay conditional."""
    segments = pd.read_csv(OUTPUT / "segment_summary.csv").set_index(
        ["arm", "segment"]
    )
    qualifying = segments.loc[
        (DOMESTIC_AD_XG_FORM, "PHASE:QUALIFYING"), "exact_score_nll_vs_baseline"
    ]
    main = segments.loc[
        (DOMESTIC_AD_XG_FORM, "PHASE:MAIN"), "exact_score_nll_vs_baseline"
    ]

    assert qualifying > 0.0
    assert main < 0.0


@pytest.mark.skipif(
    not (SENSITIVITY / "backtest_manifest.json").is_file(),
    reason="the sensitivity run has not been executed in this checkout",
)
def test_the_training_window_does_not_change_the_conclusion() -> None:
    sensitivity = json.loads(
        (SENSITIVITY / "backtest_manifest.json").read_text(encoding="utf-8")
    )

    assert sensitivity["training_window"] == "xg"
    assert sensitivity["matches_per_arm"] == EXPECTED_MATCHES
    assert sensitivity["xg_beats_baseline_on_exact_score"] is True
    assert sensitivity["xg_beats_goals_on_exact_score"] is True
    reliable = {
        (row["arm"], row["segment"]) for row in sensitivity["reliable_improvements"]
    }
    assert ("DOMESTIC_AD_XG_FORM", "XG_PRESENT") in reliable
    assert not any(arm == "DOMESTIC_AD_GOALS_FORM" for arm, _ in reliable)
