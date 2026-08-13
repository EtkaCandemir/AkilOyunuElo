from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ao_elo.model_based_partitioning import (
    MOBConfig,
    fit_mob_tree,
    predict_multiplier,
    serialize_tree,
)


def synthetic_frame(*, heterogeneous: bool, seed: int = 17) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for season_index, season in enumerate(("2018/19", "2019/20", "2020/21")):
        for team_id in range(120):
            league = (team_id + 0.5) / 120.0
            rating = 900.0 + 650.0 * league + rng.normal(0.0, 20.0)
            increment = rng.choice((-1.0, 1.0)) * rng.uniform(20.0, 80.0)
            theta = 0.05 if heterogeneous and league <= 0.5 else 1.15 if heterogeneous else 0.55
            target = 0.25 + 0.00035 * rating + theta * 0.00035 * increment
            target += rng.normal(0.0, 0.012)
            rows.append({
                "season": season,
                "team_id": season_index * 1000 + team_id,
                "target_score": target,
                "target_matches": 8 + team_id % 5,
                "current_initial_rating": rating,
                "strong_increment": increment,
                "league_strength": league,
                "rating_percentile": league,
                "effective_european_exposure": (team_id % 10) / 10.0,
                "competition": ("UCL", "UEL", "UECL")[team_id % 3],
            })
    return pd.DataFrame(rows)


def test_mob_recovers_known_numeric_parameter_instability() -> None:
    config = MOBConfig(
        partition_variables=("league_strength",),
        permutations=499,
        max_depth=1,
        minimum_leaf_team_seasons=60,
        minimum_leaf_seasons=2,
        minimum_nonzero_increment=40,
    )
    tree = fit_mob_tree(synthetic_frame(heterogeneous=True), config)
    assert not tree.root.is_leaf
    assert tree.root.split_variable == "league_strength"
    assert tree.root.split_threshold == pytest.approx(0.5, abs=0.08)
    assert tree.root.left is not None and tree.root.right is not None
    assert tree.root.left.theta < 0.25
    assert tree.root.right.theta > 0.90


def test_homogeneous_data_does_not_create_spurious_split() -> None:
    config = MOBConfig(
        partition_variables=("league_strength", "competition"),
        permutations=499,
        max_depth=2,
        minimum_leaf_team_seasons=60,
        minimum_leaf_seasons=2,
        minimum_nonzero_increment=40,
    )
    tree = fit_mob_tree(synthetic_frame(heterogeneous=False), config)
    assert tree.root.is_leaf
    assert tree.root.stop_reason == "NO_SIGNIFICANT_PARAMETER_INSTABILITY"


def test_unseen_competition_falls_back_to_current() -> None:
    frame = synthetic_frame(heterogeneous=False)
    frame = frame.loc[~frame["competition"].eq("UECL")]
    tree = fit_mob_tree(
        frame,
        MOBConfig(
            partition_variables=("competition",),
            permutations=99,
            minimum_leaf_team_seasons=30,
            minimum_nonzero_increment=20,
        ),
    )
    prediction = predict_multiplier(tree, {"competition": "UECL"})
    assert prediction.fallback
    assert prediction.theta == 0.0
    assert prediction.fallback_reason == "UNSEEN_COMPETITION"


def test_tree_serialization_is_deterministic_and_valid_json() -> None:
    config = MOBConfig(
        partition_variables=("league_strength",),
        permutations=99,
        max_depth=1,
        minimum_leaf_team_seasons=60,
        minimum_nonzero_increment=40,
    )
    first = serialize_tree(fit_mob_tree(synthetic_frame(heterogeneous=True), config))
    second = serialize_tree(fit_mob_tree(synthetic_frame(heterogeneous=True), config))
    assert first == second
    assert json.loads(first)["model"] == "DOMESTIC_SURPRISE_MOB"


def test_duplicate_team_season_is_rejected() -> None:
    frame = synthetic_frame(heterogeneous=False)
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate season/team_id"):
        fit_mob_tree(frame, MOBConfig(permutations=9))


def test_config_rejects_unknown_partition_variable() -> None:
    with pytest.raises(ValueError, match="Unsupported MOB partition"):
        MOBConfig(partition_variables=("future_result",)).validate()

