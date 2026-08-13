from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ao_elo.model_based_partitioning import MOBConfig
from scripts.run_domestic_surprise_mob_backtest import (
    EARLY_UECL_FALLBACK_SEASONS,
    PRE_UECL,
    UECL_ERA,
    build_walk_forward_mob_adjustments,
    era_for_season,
)


def mob_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(11)
    rows = []
    current_rows = []
    seasons = ("2018/19", "2019/20", "2020/21", "2021/22", "2022/23", "2023/24")
    for season_index, season in enumerate(seasons):
        for team_id in range(30):
            rating = 900.0 + team_id * 15.0 + rng.normal(0.0, 5.0)
            current_adjustment = (-1.0 if team_id % 2 else 1.0) * 8.0
            strong_adjustment = (-1.0 if team_id % 2 else 1.0) * 30.0
            common = {
                "season": season,
                "team_id": team_id,
                "team_name": f"Team {team_id}",
                "club_id": f"CLUB-{team_id}",
                "country_code": "TST",
                "competition": ("UCL", "UEL", "UECL")[team_id % 3] if season_index >= 3 else ("UCL", "UEL")[team_id % 2],
                "league_strength": (team_id + 1.0) / 30.0,
                "rating_percentile": (team_id + 1.0) / 30.0,
                "effective_european_exposure": (team_id % 10) / 10.0,
                "raw_surprise": np.sign(current_adjustment),
            }
            rows.append({
                **common,
                "target_score": 0.25 + rating * 0.00035 + strong_adjustment * 0.0002 + rng.normal(0.0, 0.01),
                "target_matches": 8,
                "current_initial_rating": rating,
                "current_adjustment": current_adjustment,
                "strong_adjustment": strong_adjustment,
                "strong_increment": strong_adjustment - current_adjustment,
                "history_seasons_available": 5,
                "format_era": era_for_season(season),
            })
            current_common = {key: value for key, value in common.items() if key != "rating_percentile"}
            current_rows.append({
                **current_common,
                "history_seasons": 5,
                "ao_first_elo_adjustment": current_adjustment,
                "baseline_ao_first_elo": rating - current_adjustment,
                "adjusted_ao_first_elo": rating,
                "domestic_prior_adjustment": current_adjustment,
            })
    return pd.DataFrame(rows), pd.DataFrame(current_rows)


def test_uecl_era_contract_is_explicit() -> None:
    assert era_for_season("2020/21") == PRE_UECL
    assert era_for_season("2021/22") == UECL_ERA


def test_first_two_uecl_era_seasons_fall_back_to_current() -> None:
    learning, current = mob_frames()
    config = MOBConfig(
        partition_variables=("league_strength",),
        permutations=49,
        max_depth=0,
        max_leaves=1,
        minimum_leaf_team_seasons=10,
        minimum_nonzero_increment=10,
    )
    adjustments, assignments, _, _ = build_walk_forward_mob_adjustments(
        learning, current, config, "TEST_MOB"
    )
    fallback = assignments.loc[assignments["season"].isin(EARLY_UECL_FALLBACK_SEASONS)]
    assert fallback["mob_fallback"].all()
    assert fallback["mob_theta"].eq(0.0).all()
    assert np.allclose(fallback["ao_first_elo_adjustment"], fallback["current_adjustment"])
    active = assignments.loc[assignments["season"].eq("2023/24")]
    assert not active["mob_fallback"].any()
    assert adjustments["ao_first_elo_adjustment"].abs().le(60.0).all()


def test_walk_forward_training_never_crosses_era_or_target_season() -> None:
    learning, current = mob_frames()
    config = MOBConfig(
        partition_variables=("league_strength",),
        permutations=49,
        max_depth=0,
        max_leaves=1,
        minimum_leaf_team_seasons=10,
        minimum_nonzero_increment=10,
    )
    _, assignments, _, _ = build_walk_forward_mob_adjustments(
        learning, current, config, "TEST_MOB"
    )
    for row in assignments.itertuples(index=False):
        history = [value for value in str(row.mob_training_seasons).split(",") if value]
        assert all(era_for_season(value) == row.format_era for value in history)
        assert all(int(value[:4]) < int(row.season[:4]) for value in history)
