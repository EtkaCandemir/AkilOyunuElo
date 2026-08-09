from __future__ import annotations

import pandas as pd
import pytest

from ao_elo.opta_league_snapshot import compare_snapshot_to_ao_country_strength, validate_snapshot


def snapshot() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"snapshot_date": "2026-07-27", "source_url": "https://example.com", "country_code": "ENG", "league_name": "Premier League", "league_average_rating": 90.0, "top_10_average_rating": 92.0, "top_5_average_rating": 94.0, "source_sha256": "a" * 64},
            {"snapshot_date": "2026-07-27", "source_url": "https://example.com", "country_code": "TUR", "league_name": "Super Lig", "league_average_rating": 70.0, "top_10_average_rating": 75.0, "top_5_average_rating": 80.0, "source_sha256": "a" * 64},
        ]
    )


def test_snapshot_validation_and_ao_comparison() -> None:
    values = snapshot()
    validate_snapshot(values)
    comparison, summary = compare_snapshot_to_ao_country_strength(
        values,
        pd.DataFrame([{"country_code": "ENG", "official_country_rank": 1}, {"country_code": "TUR", "official_country_rank": 10}]),
    )
    assert list(comparison["country_code"]) == ["ENG", "TUR"]
    assert summary.iloc[0]["leagues_compared"] == 2


def test_snapshot_rejects_invalid_depth_order() -> None:
    values = snapshot()
    values.loc[0, "top_5_average_rating"] = 80.0
    with pytest.raises(ValueError, match="top-five"):
        validate_snapshot(values)
