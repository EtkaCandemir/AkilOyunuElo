from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_initial_elo_external_comparison_2025_26 import (  # noqa: E402
    exposure_segment,
    load_archived_opta,
)


def test_load_archived_opta_accepts_gzip_and_orders_rank(tmp_path: Path) -> None:
    rows = [
        {
            "contestantId": "away",
            "currentRating": 80.0,
            "rank": 2,
            "contestantName": "Away",
            "contestantShortName": "Away",
            "contestantClubName": "Away FC",
        },
        {
            "contestantId": "home",
            "currentRating": 90.0,
            "rank": 1,
            "contestantName": "Home",
            "contestantShortName": "Home",
            "contestantClubName": "Home FC",
        },
    ]
    path = tmp_path / "opta.json.gz"
    path.write_bytes(gzip.compress(json.dumps(rows).encode("utf-8")))

    result = load_archived_opta(path)

    assert result["opta_team_id"].tolist() == ["home", "away"]
    assert result["opta_power_rating"].tolist() == [90.0, 80.0]


def test_load_archived_opta_rejects_duplicate_team_ids(tmp_path: Path) -> None:
    row = {
        "contestantId": "same",
        "currentRating": 80.0,
        "rank": 1,
        "contestantName": "Same",
        "contestantShortName": "Same",
        "contestantClubName": "Same FC",
    }
    path = tmp_path / "opta.json"
    path.write_text(json.dumps([row, row]), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate team IDs"):
        load_archived_opta(path)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, "EXPOSURE_ZERO"),
        (0.49, "EXPOSURE_LOW"),
        (0.50, "EXPOSURE_MEDIUM"),
        (0.849, "EXPOSURE_MEDIUM"),
        (0.85, "EXPOSURE_HIGH"),
        (1.0, "EXPOSURE_HIGH"),
    ],
)
def test_exposure_segment_boundaries(value: float, expected: str) -> None:
    assert exposure_segment(value) == expected
