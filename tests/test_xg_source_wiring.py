from __future__ import annotations

"""Guard how xG reaches the model: the evaluation map and the live season.

Two paths carry xG. The shared `XG_DATA` constant feeds every replay-based
backtest, and the 2026/27 preproduction results feed the live season replay.
Both were pointed at a single season while the contract documents six, so these
tests pin the wiring rather than the numbers it produces.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ao_elo.xg_dataset import validate_master_dataset
from scripts.run_stage_weighted_progression_backtest import (
    LEGACY_SINGLE_SEASON_XG_DATA,
    XG_DATA,
)


LIVE_MATCHES = ROOT / "data" / "season_2026_27_preproduction" / "matches_completed.csv"
LIVE_XG = ROOT / "data" / "xg_2026_27" / "uefa_2026_27_matches_with_xg.csv"


# ---------------------------------------------------------------------------
# the shared evaluation map
# ---------------------------------------------------------------------------


def test_shared_constant_points_at_the_six_season_map() -> None:
    """A single-season map would replay seven of eight seasons without xG."""
    assert XG_DATA.name == "uefa_2020_2026_matches_with_xg.csv"
    assert XG_DATA.exists()


def test_frozen_single_season_dataset_is_still_reachable() -> None:
    """The original evidence file stays addressable for provenance."""
    assert LEGACY_SINGLE_SEASON_XG_DATA.exists()
    assert LEGACY_SINGLE_SEASON_XG_DATA != XG_DATA


@pytest.mark.skipif(not XG_DATA.exists(), reason="six-season map is not built")
def test_shared_map_covers_more_than_one_season() -> None:
    frame = pd.read_csv(XG_DATA)

    assert frame["season"].nunique() >= 6


# ---------------------------------------------------------------------------
# the live season
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not LIVE_MATCHES.exists(), reason="live season is not built")
def test_live_results_carry_xg_where_the_source_has_it() -> None:
    matches = pd.read_csv(LIVE_MATCHES)
    eligible = matches["xg_analysis_eligible"].astype(bool)

    assert eligible.sum() > 0, "the live season must not replay xG-blind"
    assert matches.loc[eligible, ["xg_home", "xg_away"]].notna().all().all()


@pytest.mark.skipif(not LIVE_MATCHES.exists(), reason="live season is not built")
def test_live_results_never_impute_missing_xg(  ) -> None:
    matches = pd.read_csv(LIVE_MATCHES)
    blind = ~matches["xg_analysis_eligible"].astype(bool)

    assert matches.loc[blind, ["xg_home", "xg_away"]].isna().all().all()
    assert matches.loc[blind, "xg_fallback"].eq("GOAL_MARGIN_ONLY").all()


@pytest.mark.skipif(
    not (LIVE_MATCHES.exists() and LIVE_XG.exists()),
    reason="live season or its xG source is not built",
)
def test_live_xg_values_match_their_source() -> None:
    """The wiring must copy the audited value, never a re-derived one."""
    matches = pd.read_csv(LIVE_MATCHES)
    source = pd.read_csv(LIVE_XG)
    merged = matches[["match_id", "xg_home", "xg_away", "xg_analysis_eligible"]].merge(
        source[["match_id", "xg_home", "xg_away"]],
        on="match_id",
        suffixes=("_live", "_source"),
        validate="one_to_one",
    )
    eligible = merged["xg_analysis_eligible"].astype(bool)

    for side in ("home", "away"):
        assert merged.loc[eligible, f"xg_{side}_live"].equals(
            merged.loc[eligible, f"xg_{side}_source"]
        )


# ---------------------------------------------------------------------------
# the validator that had to open up for multi-season use
# ---------------------------------------------------------------------------


WIDE_MASTER = ROOT / "data" / "xg_2020_2026" / "uefa_2020_2026_matches_with_xg.csv"
requires_master = pytest.mark.skipif(
    not WIDE_MASTER.exists(), reason="six-season master is not built"
)


@pytest.fixture(name="master")
def fixture_master() -> pd.DataFrame:
    """The real six-season master, which carries a small unresolved tail."""
    return pd.read_csv(WIDE_MASTER)


@requires_master
def test_real_master_has_an_unresolved_tail(master: pd.DataFrame) -> None:
    blank = master["xg_source_match_id"].astype("string").str.strip()
    unresolved = int((blank.isna() | blank.eq("")).sum())

    assert 0 < unresolved < len(master) * 0.05, "expected a small unresolved tail"


@requires_master
def test_unresolved_rows_are_allowed_outside_the_frozen_dataset(
    master: pd.DataFrame,
) -> None:
    """A small unresolved tail is expected once the window widens."""
    validate_master_dataset(master, strict_counts=False)


@requires_master
def test_frozen_mode_rejects_the_wider_master(master: pd.DataFrame) -> None:
    """Strict mode is the frozen 2025/26 guarantee, not a general rule."""
    with pytest.raises(ValueError):
        validate_master_dataset(master, strict_counts=True)


@requires_master
def test_duplicate_source_ids_are_still_rejected(master: pd.DataFrame) -> None:
    corrupted = master.copy()
    resolved = corrupted.index[
        corrupted["xg_source_match_id"].astype("string").str.strip().fillna("").ne("")
    ][:2]
    corrupted.loc[resolved[1], "xg_source_match_id"] = corrupted.loc[
        resolved[0], "xg_source_match_id"
    ]

    with pytest.raises(ValueError, match="must be unique"):
        validate_master_dataset(corrupted, strict_counts=False)


@requires_master
def test_blank_source_ids_do_not_count_as_duplicates(master: pd.DataFrame) -> None:
    """Two unresolved rows share an empty string; that is not a collision."""
    blanked = master.copy()
    blanked["xg_source_match_id"] = blanked["xg_source_match_id"].astype(object)
    blanked.loc[blanked.index[:2], "xg_source_match_id"] = ""

    validate_master_dataset(blanked, strict_counts=False)
