from __future__ import annotations

"""Guard the walk-forward that tests the participation normalization.

Two kinds of test. The first pins the run's construction: the arms, the fold
shape, and the gates that have to hold for the output to mean anything. Each
gate is exercised twice - once on a well-formed run and once on a deliberately
broken one - because a gate that cannot fail is not a gate.

The second pins the verdict the report quotes, so a later change cannot leave
the written conclusion standing on numbers that no longer reproduce.
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

from ao_elo.european_participation import (
    BASELINE,
    BLIND_LIFT,
    NORMALIZED,
    SHRINKAGE_GRID,
)
from scripts.run_european_participation_backtest import (
    ARM_ORDER,
    EVALUATION_SEASONS,
    EXPECTED_TEAM_SEASONS,
    MINIMUM_MODAL_FOLD_SHARE,
    SEED_MOVEMENT_P95_LIMIT,
    candidate_minus_control,
    decide,
    seed_impact_summary,
    validation_gates,
)


RUN = ROOT / "output" / "european_participation_backtest_2018_2026"
requires_run = pytest.mark.skipif(
    not (RUN / "selected_candidate.json").is_file(),
    reason="the participation backtest has not been run in this checkout",
)


# ---------------------------------------------------------------------------
# fixtures: a well-formed run, plus knobs to break one property at a time
# ---------------------------------------------------------------------------


def audit_frame(
    *,
    full_participation_delta: float = 0.0,
    zero_participation_delta: float = 0.0,
    arms: tuple[str, ...] = ARM_ORDER,
    p95_delta: float = 40.0,
) -> pd.DataFrame:
    rows = []
    for arm in arms:
        moving = arm != BASELINE
        for season in EVALUATION_SEASONS:
            for index, played in enumerate((1.0, 0.47, 0.0)):
                if played >= 1.0:
                    delta = full_participation_delta if moving else 0.0
                elif played <= 0.0:
                    delta = zero_participation_delta if moving else 0.0
                else:
                    delta = p95_delta if moving else 0.0
                rows.append(
                    {
                        "fold": EVALUATION_SEASONS.index(season) + 1,
                        "arm": arm,
                        "season": season,
                        "team_id": index,
                        "candidate_played_weight": played,
                        "candidate_elo_delta": delta,
                        "candidate_ao_first_elo": 1200.0 + delta,
                    }
                )
    return pd.DataFrame(rows)


def fold_ranking_frame(
    *, candidate_delta: float = 0.01, control_delta: float = -0.002
) -> pd.DataFrame:
    rows = []
    for fold, season in enumerate(EVALUATION_SEASONS, start=1):
        for arm, delta in (
            (BASELINE, 0.0),
            (BLIND_LIFT, control_delta),
            (NORMALIZED, candidate_delta),
        ):
            rows.append(
                {
                    "fold": fold,
                    "test_season": season,
                    "arm": arm,
                    "teams": 236,
                    "seed_spearman": 0.46 + delta,
                    "seed_pairwise_accuracy": 0.66 + delta / 2,
                    "delta_seed_spearman": delta,
                    "delta_seed_pairwise_accuracy": delta / 2,
                }
            )
    return pd.DataFrame(rows)


def selection_frame(shrinkages: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold": range(1, len(shrinkages) + 1),
            "selected_shrinkage": shrinkages,
            "blind_lift_coefficient": [40.0] * len(shrinkages),
        }
    )


def uncertainty_frame(*, improvement: bool, harm: bool = False) -> pd.DataFrame:
    lower, upper = (0.004, 0.014) if improvement else (-0.004, 0.010)
    if harm:
        lower, upper = -0.014, -0.004
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "method": "season_block_bootstrap",
                "folds": 6,
                "mean_difference": (lower + upper) / 2,
                "ci_95_lower": lower,
                "ci_95_upper": upper,
                "reliable_improvement": lower > 0.0,
                "reliable_harm": upper < 0.0,
            }
            for metric in ("seed_spearman", "seed_pairwise_accuracy")
        ]
    )


def gates_for(
    *,
    audit: pd.DataFrame | None = None,
    selections: pd.DataFrame | None = None,
    fold_ranking: pd.DataFrame | None = None,
    ranking: pd.DataFrame | None = None,
    baseline_error: float = 4.5e-13,
) -> pd.DataFrame:
    audit = audit if audit is not None else audit_frame()
    fold_ranking = fold_ranking if fold_ranking is not None else fold_ranking_frame()
    ranking = ranking if ranking is not None else uncertainty_frame(improvement=True)
    selections = (
        selections if selections is not None else selection_frame([0.2] * 6)
    )
    contract = ROOT / "contracts" / "ao_european_elo_v2_production.json"
    import hashlib

    return validation_gates(
        pd.DataFrame(),
        audit,
        selections,
        fold_ranking,
        ranking,
        candidate_minus_control(fold_ranking),
        seed_impact_summary(audit),
        baseline_error,
        hashlib.sha256(contract.read_bytes()).hexdigest(),
    ).set_index("gate")


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


def test_three_arms_are_ordered_baseline_control_candidate() -> None:
    assert ARM_ORDER == (BASELINE, BLIND_LIFT, NORMALIZED)


def test_the_evaluation_window_is_the_six_unseen_seasons() -> None:
    assert EVALUATION_SEASONS == (
        "2020/21", "2021/22", "2022/23", "2023/24", "2024/25", "2025/26",
    )
    assert EXPECTED_TEAM_SEASONS == 1887


def test_the_shrinkage_grid_includes_the_unshrunk_end() -> None:
    """`k = 0` has to be on the surface so its small-denominator risk is
    measured rather than assumed away."""
    assert 0.0 in SHRINKAGE_GRID
    assert SHRINKAGE_GRID == tuple(sorted(SHRINKAGE_GRID))


def test_candidate_minus_control_pairs_folds() -> None:
    frame = candidate_minus_control(
        fold_ranking_frame(candidate_delta=0.01, control_delta=-0.002)
    )

    assert list(frame["fold"]) == list(range(1, 7))
    assert frame["delta_seed_spearman"].round(6).eq(0.012).all()


# ---------------------------------------------------------------------------
# every gate must be able to fail
# ---------------------------------------------------------------------------


def test_a_well_formed_run_passes_every_gate() -> None:
    assert gates_for()["passed"].all()


def test_a_stale_baseline_fails_its_gate() -> None:
    """This is the gate that caught the stale seed artifact in practice."""
    gates = gates_for(baseline_error=162.49)

    assert not gates.loc["baseline_reproduces_production", "passed"]


def test_moving_a_full_participation_club_fails_its_gate() -> None:
    gates = gates_for(audit=audit_frame(full_participation_delta=5.0))

    assert not gates.loc["full_participation_unchanged", "passed"]


def test_moving_a_club_that_never_entered_fails_its_gate() -> None:
    gates = gates_for(audit=audit_frame(zero_participation_delta=5.0))

    assert not gates.loc["zero_participation_unchanged", "passed"]


def test_dropping_the_blind_control_fails_the_run() -> None:
    gates = gates_for(audit=audit_frame(arms=(BASELINE, NORMALIZED)))

    assert not gates.loc["control_arm_present", "passed"]


def test_losing_to_the_control_fails_its_gate() -> None:
    """A candidate that cannot beat a blind lift has shown nothing."""
    gates = gates_for(
        fold_ranking=fold_ranking_frame(candidate_delta=-0.002, control_delta=0.01)
    )

    assert not gates.loc["candidate_beats_control", "passed"]


def test_reliable_ranking_harm_fails_its_gate() -> None:
    gates = gates_for(ranking=uncertainty_frame(improvement=False, harm=True))

    assert not gates.loc["ranking_not_reliably_harmed", "passed"]


def test_unstable_fold_selection_fails_its_gate() -> None:
    gates = gates_for(selections=selection_frame([0.0, 0.1, 0.2, 0.35, 0.5, 0.75]))

    assert not gates.loc["selection_stability", "passed"]
    assert gates.loc["selection_stability", "observed"] < MINIMUM_MODAL_FOLD_SHARE


def test_unbounded_seed_movement_fails_its_gate() -> None:
    gates = gates_for(audit=audit_frame(p95_delta=SEED_MOVEMENT_P95_LIMIT + 50.0))

    assert not gates.loc["seed_movement_bounded", "passed"]


# ---------------------------------------------------------------------------
# the decision ladder
# ---------------------------------------------------------------------------


def arm_summary_frame(candidate: float = 0.469, control: float = 0.457) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"arm": BASELINE, "spearman": 0.460, "spearman_vs_baseline": 0.0},
            {"arm": BLIND_LIFT, "spearman": control, "spearman_vs_baseline": control - 0.460},
            {"arm": NORMALIZED, "spearman": candidate, "spearman_vs_baseline": candidate - 0.460},
        ]
    )


def test_promotion_needs_a_reliable_gain_over_both_baseline_and_control() -> None:
    decision = decide(
        gates_for().reset_index(),
        arm_summary_frame(),
        uncertainty_frame(improvement=True),
        uncertainty_frame(improvement=True),
    )

    assert decision["decision"] == "PROMOTE_CANDIDATE"
    assert decision["production_change"] is False


def test_beating_only_the_control_stops_at_shadow_candidate() -> None:
    decision = decide(
        gates_for().reset_index(),
        arm_summary_frame(),
        uncertainty_frame(improvement=False),
        uncertainty_frame(improvement=True),
    )

    assert decision["decision"] == "KEEP_SHADOW_CANDIDATE"


def test_beating_neither_stops_at_shadow() -> None:
    decision = decide(
        gates_for().reset_index(),
        arm_summary_frame(),
        uncertainty_frame(improvement=False),
        uncertainty_frame(improvement=False),
    )

    assert decision["decision"] == "KEEP_SHADOW"


def test_a_structural_failure_forces_keep_current() -> None:
    """A broken baseline invalidates every comparison built on it."""
    decision = decide(
        gates_for(baseline_error=162.49).reset_index(),
        arm_summary_frame(),
        uncertainty_frame(improvement=True),
        uncertainty_frame(improvement=True),
    )

    assert decision["decision"] == "KEEP_CURRENT"


# ---------------------------------------------------------------------------
# the verdict of the real run
# ---------------------------------------------------------------------------


@pytest.fixture(name="verdict", scope="module")
def fixture_verdict() -> dict:
    return json.loads((RUN / "selected_candidate.json").read_text(encoding="utf-8"))


@requires_run
def test_the_run_changed_no_production_parameter(verdict: dict) -> None:
    """The run itself must not mutate production.

    The script hashes the contract on entry and on exit and records the result
    in `safety_audit.csv`; that is the property this test guards. It
    deliberately does not require the recorded hash to equal today's contract:
    this study is the evidence that activated the layer, so the contract has
    moved on by design. Requiring equality would make a successful activation
    look like a failed research run.
    """

    import hashlib

    assert verdict["production_change"] is False
    recorded = verdict["production_contract_sha256"]
    assert isinstance(recorded, str) and len(recorded) == 64
    int(recorded, 16)

    audit = pd.read_csv(RUN / "safety_audit.csv")
    row = audit[audit["gate"].eq("contract_sha256_unchanged")]
    assert not row.empty, "the in-run contract guard must be recorded"
    assert bool(row["passed"].iloc[0])
    assert str(row["observed"].iloc[0]) == recorded

    contract = ROOT / "contracts" / "ao_european_elo_v2_production.json"
    live = hashlib.sha256(contract.read_bytes()).hexdigest()
    if live != recorded:
        # The only sanctioned reason for divergence is that this study was
        # promoted. Anything else means the evidence is stale.
        payload = json.loads(contract.read_text(encoding="utf-8"))
        assert payload["european_participation"]["active"] is True
        assert verdict["decision"] == "PROMOTE_CANDIDATE"
        # The activated shrinkage must be the value the folds actually chose,
        # not a hand-picked one.
        selections = pd.read_csv(RUN / "fold_selections.csv")
        modal = selections["selected_shrinkage"].mode().iloc[0]
        assert payload["european_participation"]["shrinkage"] == pytest.approx(modal)


@requires_run
def test_the_candidate_beats_the_blind_control(verdict: dict) -> None:
    """The control is what makes the gain attributable to participation."""
    assert verdict["candidate_vs_baseline"] > verdict["control_vs_baseline"]
    assert verdict["candidate_seed_spearman"] > verdict["control_seed_spearman"]


@requires_run
def test_every_gate_passed(verdict: dict) -> None:
    audit = pd.read_csv(RUN / "safety_audit.csv").set_index("gate")

    assert bool(audit["passed"].all()), sorted(
        audit.loc[~audit["passed"].astype(bool)].index
    )


@requires_run
def test_complete_evidence_never_moved() -> None:
    impact = pd.read_csv(RUN / "seed_impact_summary.csv").set_index("arm")

    assert impact.loc[NORMALIZED, "full_participation_max_abs_delta"] <= 1e-9
    assert impact.loc[NORMALIZED, "zero_participation_max_abs_delta"] <= 1e-9
