"""Tur 2 bulgu #3: holdout guard'i gercek tarih/sezon dogrulamasi olmali."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ao_elo.holdout_window import (
    DEVELOPMENT_SEASONS,
    HOLDOUT_OPENS_UTC,
    HOLDOUT_SEASON,
    untouched_holdout_label,
    validate_development_window,
)

ROOT = Path(__file__).resolve().parents[1]
FEATURE_STORE = ROOT / "output" / "ml_1x2_backtest_2018_2026" / "pre_match_feature_store.csv"


def _window(seasons: tuple[str, ...], *, last_kickoff: str = "2026-05-01T12:00:00+00:00") -> pd.DataFrame:
    rows = [
        {"season": season, "kickoff_utc": f"{2018 + index}-09-01T12:00:00+00:00"}
        for index, season in enumerate(seasons)
    ]
    rows[-1]["kickoff_utc"] = last_kickoff
    return pd.DataFrame(rows)


def test_frozen_development_window_is_accepted() -> None:
    assert validate_development_window(_window(DEVELOPMENT_SEASONS), label="t") == DEVELOPMENT_SEASONS


def test_eight_seasons_ending_in_the_holdout_are_rejected() -> None:
    # Tur 2'nin gosterdigi tam senaryo: sekiz sezon sayilir ama pencere kaymistir.
    shifted = DEVELOPMENT_SEASONS[1:] + (HOLDOUT_SEASON,)
    assert len(shifted) == 8
    with pytest.raises(ValueError, match="frozen development window"):
        validate_development_window(_window(shifted), label="t")


def test_correct_labels_with_a_holdout_kickoff_are_rejected() -> None:
    # Etiketler dogru ama bir satirin kickoff'u holdout'a dusuyor.
    frame = _window(DEVELOPMENT_SEASONS, last_kickoff="2026-07-02T12:00:00+00:00")
    with pytest.raises(ValueError, match="holdout opening"):
        validate_development_window(frame, label="t")


def test_holdout_boundary_is_exclusive_at_the_opening_instant() -> None:
    frame = _window(DEVELOPMENT_SEASONS, last_kickoff=HOLDOUT_OPENS_UTC.isoformat())
    with pytest.raises(ValueError, match="holdout opening"):
        validate_development_window(frame, label="t")


def test_timezone_naive_kickoff_is_refused_not_assumed_utc() -> None:
    # Naive bir deger UTC sayilirsa, holdout'un bir saat icindeki yerel zaman
    # 30 Haziran gibi okunur ve pencereden gecer.
    frame = _window(DEVELOPMENT_SEASONS, last_kickoff="2026-06-30 23:30:00")
    with pytest.raises(ValueError, match="timezone-aware"):
        validate_development_window(frame, label="t")


def test_the_same_wall_clock_with_an_offset_is_judged_on_real_utc() -> None:
    frame = _window(DEVELOPMENT_SEASONS, last_kickoff="2026-06-30T23:30:00-04:00")
    with pytest.raises(ValueError, match="holdout opening"):
        validate_development_window(frame, label="t")


def test_untouched_label_is_derived_and_cannot_outlive_its_condition() -> None:
    assert untouched_holdout_label(_window(DEVELOPMENT_SEASONS), label="t") == HOLDOUT_SEASON
    shifted = DEVELOPMENT_SEASONS[1:] + (HOLDOUT_SEASON,)
    with pytest.raises(ValueError):
        untouched_holdout_label(_window(shifted), label="t")


def test_the_real_feature_store_satisfies_the_guard() -> None:
    if not FEATURE_STORE.exists():  # pragma: no cover - artifact kosusuna bagli
        pytest.skip("feature store not built")
    store = pd.read_csv(FEATURE_STORE)
    data = store.sort_values(["kickoff_utc", "match_id"], kind="stable")
    assert validate_development_window(data, label="real") == DEVELOPMENT_SEASONS
