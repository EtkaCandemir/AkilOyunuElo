from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_v2_ranking_calibration import (  # noqa: E402
    BASELINE_NAME,
    candidate_name,
    clustered_candidate_vs_baseline_ci,
    distribution_guardrails,
    evaluate_outer_gate,
    expanding_folds,
    metric_not_worse,
    tail_candidates,
)


def test_tail_grid_contains_exactly_one_hundred_candidates() -> None:
    candidates = tail_candidates()
    names = [name for name, _ in candidates]

    assert len(candidates) == 100
    assert len(set(names)) == 100
    assert names[0] == BASELINE_NAME
    assert candidate_name(0.25, 0.5, 2 / 3) == "c0p25_e0p5_x0p666667"


def test_eight_seasons_create_six_outer_folds() -> None:
    seasons = tuple(f"20{year:02d}/{year + 1:02d}" for year in range(18, 26))
    folds = expanding_folds(seasons)

    assert len(folds) == 6
    assert folds[0] == (seasons[:2], seasons[2])
    assert folds[-1] == (seasons[:7], seasons[7])


def test_unavailable_competition_is_not_an_automatic_veto() -> None:
    assert metric_not_worse(float("nan"), float("nan"))
    assert metric_not_worse(0.4, float("nan"))
    assert not metric_not_worse(0.39, 0.40)


def test_distribution_guardrails_reject_clipping_boundary() -> None:
    healthy = pd.DataFrame(
        {"ao_first_elo": [950.0] * 80 + [1700.0] * 19 + [1850.0]}
    )
    clipped = healthy.copy()
    clipped.loc[0, "ao_first_elo"] = 500.0

    assert distribution_guardrails("healthy", healthy)[
        "distribution_guardrails_pass"
    ]
    assert not distribution_guardrails("clipped", clipped)[
        "distribution_guardrails_pass"
    ]


def test_outer_gate_requires_both_ranking_metrics() -> None:
    selections = pd.DataFrame(
        [{"fold": fold, "selected_candidate": "candidate"} for fold in range(1, 7)]
    )
    rows = []
    for fold in range(1, 7):
        rows.extend(
            [
                {
                    "fold": fold,
                    "model": "baseline",
                    "ranking_score": 0.40,
                    "pairwise_accuracy": 0.60,
                },
                {
                    "fold": fold,
                    "model": "selected",
                    "ranking_score": 0.41,
                    "pairwise_accuracy": 0.59 if fold == 6 else 0.61,
                },
            ]
        )
    competition = pd.DataFrame(
        [
            {
                "model": "selected",
                "competition": competition,
                "spearman_delta": 0.01,
                "pairwise_delta": 0.01,
            }
            for competition in ("UCL", "UEL", "UECL")
        ]
    )

    gate = evaluate_outer_gate(selections, pd.DataFrame(rows), competition)

    assert not gate["no_unseen_fold_regression"]
    assert gate["folds_improved_both_metrics"] == 5


def test_clustered_uncertainty_detects_reliable_harm() -> None:
    rows = []
    for index in range(20):
        common = {
            "match_id": f"m{index}",
            "season": "2024/25",
            "competition": "UCL",
            "tie_id": f"t{index}",
        }
        rows.append({**common, "candidate": BASELINE_NAME, "brier_loss": 0.10})
        rows.append({**common, "candidate": "candidate", "brier_loss": 0.15})

    result = clustered_candidate_vs_baseline_ci(
        pd.DataFrame(rows),
        "candidate",
        bootstrap_samples=500,
        seed=7,
    ).iloc[0]

    assert result["mean_brier_difference"] == pytest.approx(0.05)
    assert result["reliable_harm"]
    assert not result["reliable_improvement"]
