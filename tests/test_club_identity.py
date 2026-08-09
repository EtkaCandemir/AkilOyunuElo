from __future__ import annotations

import pandas as pd
import pytest

from ao_elo.club_identity import (
    attach_match_club_ids,
    identity_safe_forward_ranking,
    permanent_club_id,
    validate_club_registry,
    validate_team_season_identity,
)


def identity_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"season": "2022/23", "local_team_id": 1, "team_name": "A", "country_code": "AAA", "club_id": "AO-UEFA-10", "uefa_team_id": "10"},
            {"season": "2022/23", "local_team_id": 2, "team_name": "B", "country_code": "BBB", "club_id": "AO-UEFA-20", "uefa_team_id": "20"},
            {"season": "2022/23", "local_team_id": 3, "team_name": "C", "country_code": "CCC", "club_id": "AO-UEFA-30", "uefa_team_id": "30"},
            # Local IDs are intentionally permuted in the following season.
            {"season": "2023/24", "local_team_id": 1, "team_name": "B", "country_code": "BBB", "club_id": "AO-UEFA-20", "uefa_team_id": "20"},
            {"season": "2023/24", "local_team_id": 2, "team_name": "C", "country_code": "CCC", "club_id": "AO-UEFA-30", "uefa_team_id": "30"},
            {"season": "2023/24", "local_team_id": 3, "team_name": "A renamed", "country_code": "AAA", "club_id": "AO-UEFA-10", "uefa_team_id": "10"},
        ]
    )


def test_permanent_club_id_is_deterministic() -> None:
    assert permanent_club_id("52723") == "AO-UEFA-52723"
    assert permanent_club_id(52723) == "AO-UEFA-52723"
    with pytest.raises(ValueError):
        permanent_club_id(0)


def test_team_season_identity_allows_local_id_changes() -> None:
    identity = identity_frame()
    validate_team_season_identity(identity)
    assert identity.loc[identity["club_id"].eq("AO-UEFA-10"), "local_team_id"].tolist() == [1, 3]


def test_duplicate_club_in_same_season_is_rejected() -> None:
    identity = identity_frame()
    identity.loc[1, "club_id"] = "AO-UEFA-10"
    identity.loc[1, "uefa_team_id"] = "10"
    with pytest.raises(ValueError, match="two local teams"):
        validate_team_season_identity(identity)


def test_match_identity_uses_season_bridge() -> None:
    matches = pd.DataFrame(
        {
            "match_id": ["m1"],
            "season": ["2023/24"],
            "home_team_id": [3],
            "away_team_id": [1],
        }
    )
    attached = attach_match_club_ids(matches, identity_frame())
    assert attached.iloc[0]["home_club_id"] == "AO-UEFA-10"
    assert attached.iloc[0]["away_club_id"] == "AO-UEFA-20"


def test_forward_ranking_joins_permanent_clubs_not_local_ids() -> None:
    end_ratings = pd.DataFrame(
        {
            "season": ["2022/23"] * 3,
            "team_id": [1, 2, 3],
            "end_live_rating": [1900.0, 1700.0, 1500.0],
        }
    )
    target = pd.DataFrame(
        {
            "season": ["2023/24"] * 3,
            "competition": ["UCL"] * 3,
            "team_id": [3, 1, 2],
            "schedule_adjusted_score": [0.9, 0.5, 0.1],
        }
    )
    ranking = identity_safe_forward_ranking(
        end_ratings,
        target,
        identity_frame(),
        allowed_target_seasons={"2023/24"},
    )
    pooled = ranking.loc[ranking["competition"].eq("ALL")].iloc[0]
    assert pooled["ranking_score"] == pytest.approx(1.0)
    assert pooled["pairwise_accuracy"] == pytest.approx(1.0)


def test_registry_rejects_duplicate_provider_id() -> None:
    registry = pd.DataFrame(
        [
            {"club_id": "AO-UEFA-10", "uefa_team_id": "10", "canonical_name": "A", "country_code": "AAA", "first_season": "2022/23", "last_season": "2023/24"},
            {"club_id": "AO-UEFA-10", "uefa_team_id": "10", "canonical_name": "A2", "country_code": "AAA", "first_season": "2023/24", "last_season": "2023/24"},
        ]
    )
    with pytest.raises(ValueError, match="duplicate club_id"):
        validate_club_registry(registry)
