"""Frozen development window shared by the offline artifact producers.

`docs/HOLDOUT_PROTOCOL_2026_27.md` keeps 2026/27 out of every fit and every
selection surface.  Before this module each producer guarded that on its own and
none of the guards actually reached the data:

* `ml_backtest` only counted eight distinct seasons.  A 2019/20-2026/27 window
  passed, trained the final model on 2026/27 rows, and still wrote
  `"holdout": "2026/27_UNTOUCHED"` into the artifact metadata.
* `domestic_poisson_backtest` and `prediction_ensemble` additionally checked
  `seasons[-1] != "2026/27"`, which a mislabelled or out-of-order row defeats.

The window is therefore validated by identity *and* by the kickoff timestamps
that back the labels, and the "untouched" claim is only produced by a function
that re-runs that validation.
"""

from __future__ import annotations

import pandas as pd

from ao_elo.validators import require_utc_timestamp

DEVELOPMENT_SEASONS: tuple[str, ...] = (
    "2018/19",
    "2019/20",
    "2020/21",
    "2021/22",
    "2022/23",
    "2023/24",
    "2024/25",
    "2025/26",
)

HOLDOUT_SEASON = "2026/27"

# AO seasons roll over on 1 July, so this is the first instant belonging to the
# holdout no matter how a row is labelled.
HOLDOUT_OPENS_UTC = pd.Timestamp("2026-07-01T00:00:00+00:00")


def validate_development_window(data: pd.DataFrame, *, label: str) -> tuple[str, ...]:
    """Return the development seasons after proving the holdout is absent.

    `label` names the caller so a failure says which producer was fed the bad
    window.  Raises rather than filtering: silently dropping holdout rows would
    let a mislabelled input produce a quietly different training set.
    """

    missing = sorted({"season", "kickoff_utc"}.difference(data.columns))
    if missing:
        raise ValueError(f"{label}: development window needs columns {missing}")

    seasons = tuple(dict.fromkeys(data["season"].astype(str)))
    if seasons != DEVELOPMENT_SEASONS:
        raise ValueError(
            f"{label}: expected the frozen development window "
            f"{DEVELOPMENT_SEASONS}, found {seasons}"
        )

    # `pd.to_datetime(..., utc=True)` stamps UTC onto a naive value instead of
    # refusing it, so a local-time row an hour inside the holdout would read as
    # 30 June and pass.  DATA_CONTRACTS section 1 requires an explicit offset;
    # validate each value rather than coercing the column.
    latest: pd.Timestamp | None = None
    for position, value in enumerate(data["kickoff_utc"]):
        kickoff = require_utc_timestamp(value, f"{label}: kickoff_utc row {position}")
        if latest is None or kickoff > latest:
            latest = kickoff
    if latest is None:
        raise ValueError(f"{label}: development window has no kickoff_utc rows")
    if latest >= HOLDOUT_OPENS_UTC:
        raise ValueError(
            f"{label}: development window reaches {latest.isoformat()}, at or after "
            f"the {HOLDOUT_SEASON} holdout opening {HOLDOUT_OPENS_UTC.isoformat()}"
        )
    return seasons


def untouched_holdout_label(data: pd.DataFrame, *, label: str) -> str:
    """Return the holdout season only when `data` demonstrably excludes it.

    Artifact metadata must never assert an untouched holdout as a literal; the
    claim is re-derived here so it cannot outlive the condition it describes.
    """

    validate_development_window(data, label=label)
    return HOLDOUT_SEASON
