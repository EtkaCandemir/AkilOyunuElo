from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_external_ranking_comparison_2025_26 import (
    bootstrap_spearman_delta_ci,
    normalize_name,
    pairwise_accuracy,
    ranking_metrics,
)


def test_normalize_name_handles_accents_and_punctuation() -> None:
    assert normalize_name("Bayern Munchen") == "bayernmunchen"
    assert normalize_name("Bayern Muenchen") == "bayernmuenchen"
    assert normalize_name("Bodo/Glimt") == "bodoglimt"


def test_ranking_metrics_recognize_perfect_and_reverse_order() -> None:
    benchmark = pd.Series([4.0, 3.0, 2.0, 1.0])

    perfect = ranking_metrics(pd.Series([40.0, 30.0, 20.0, 10.0]), benchmark)
    reverse = ranking_metrics(pd.Series([10.0, 20.0, 30.0, 40.0]), benchmark)

    assert perfect.spearman == pytest.approx(1.0)
    assert perfect.pairwise_accuracy == pytest.approx(1.0)
    assert perfect.rank_mae == pytest.approx(0.0)
    assert reverse.spearman == pytest.approx(-1.0)
    assert reverse.pairwise_accuracy == pytest.approx(0.0)


def test_pairwise_accuracy_excludes_benchmark_ties() -> None:
    model = pd.Series([3.0, 2.0, 1.0]).to_numpy()
    benchmark = pd.Series([2.0, 2.0, 1.0]).to_numpy()

    assert pairwise_accuracy(model, benchmark) == pytest.approx(1.0)


def test_paired_bootstrap_delta_is_deterministic() -> None:
    start = pd.Series([1.0, 4.0, 2.0, 3.0, 5.0])
    final = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    benchmark = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

    first = bootstrap_spearman_delta_ci(
        start, final, benchmark, seed=202526, samples=200
    )
    second = bootstrap_spearman_delta_ci(
        start, final, benchmark, seed=202526, samples=200
    )

    assert first == second
    assert first[0] >= 0.0
