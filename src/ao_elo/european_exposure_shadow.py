"""Pure seed transform for the frozen Positive Bridge exposure shadow.

This module does not import or change the production pipeline.  It accepts an
already-audited AO First output frame, verifies that the active rating in every
row still has the production blend semantics, and returns an independent frame
whose ``ao_first_elo`` column is suitable for an isolated shadow replay.

The candidate was frozen as a post-hoc shadow hypothesis.  Its constants are
therefore intentionally not configurable here:

``w_shadow = w + 0.20 * (0.65 - w)`` only when
``0 < w < 0.65 and european_prior > adjusted_domestic_prior``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


POSITIVE_BRIDGE_ARM_ID = "POSITIVE_BRIDGE_020"
POSITIVE_BRIDGE_ETA = 0.20
POSITIVE_BRIDGE_EXPOSURE_CAP = 0.65
PRODUCTION_AO_CONSISTENCY_TOLERANCE = 1e-9

REQUIRED_COLUMNS = frozenset(
    {
        "season",
        "team_id",
        "team_name",
        "adjusted_domestic_prior",
        "european_prior",
        "effective_european_exposure",
        "ao_first_elo",
    }
)
NUMERIC_COLUMNS = (
    "adjusted_domestic_prior",
    "european_prior",
    "effective_european_exposure",
    "ao_first_elo",
)


def apply_positive_bridge_seed_transform(production: pd.DataFrame) -> pd.DataFrame:
    """Return an isolated Positive Bridge seed frame.

    ``production`` is never mutated.  Ineligible rows retain their original
    ``ao_first_elo`` bit-for-bit; eligible rows receive the locally frozen
    Positive Bridge rating.  Audit columns preserve the production value,
    eligibility decision, effective shadow weight, and Elo adjustment.
    """

    numeric = _validate_production_seed_frame(production)
    domestic = numeric["adjusted_domestic_prior"]
    european = numeric["european_prior"]
    exposure = numeric["effective_european_exposure"]
    production_ao = numeric["ao_first_elo"]

    expected_production = domestic + exposure * (european - domestic)
    difference = (production_ao - expected_production).abs()
    inconsistent = difference.gt(PRODUCTION_AO_CONSISTENCY_TOLERANCE)
    if inconsistent.any():
        team_ids = production.loc[inconsistent, "team_id"].astype(str).tolist()
        raise ValueError(
            "ao_first_elo is inconsistent with the active AO blend for team_id "
            f"values {team_ids[:5]}; maximum absolute difference="
            f"{float(difference.max()):.12g}"
        )

    eligible = (
        exposure.gt(0.0)
        & exposure.lt(POSITIVE_BRIDGE_EXPOSURE_CAP)
        & european.gt(domestic)
    )
    shadow_exposure = exposure.copy()
    shadow_exposure.loc[eligible] = exposure.loc[eligible] + (
        POSITIVE_BRIDGE_ETA
        * (POSITIVE_BRIDGE_EXPOSURE_CAP - exposure.loc[eligible])
    )

    shadow_ao = production_ao.copy()
    shadow_ao.loc[eligible] = domestic.loc[eligible] + shadow_exposure.loc[
        eligible
    ] * (european.loc[eligible] - domestic.loc[eligible])

    result = production.copy(deep=True)
    result["production_ao_first_elo"] = production_ao
    result["positive_bridge_applied"] = eligible.astype(bool)
    result["shadow_effective_european_exposure"] = shadow_exposure
    result["positive_bridge_elo_delta"] = shadow_ao - production_ao
    result["positive_bridge_arm_id"] = POSITIVE_BRIDGE_ARM_ID
    result["positive_bridge_eta"] = POSITIVE_BRIDGE_ETA
    result["positive_bridge_exposure_cap"] = POSITIVE_BRIDGE_EXPOSURE_CAP
    result["ao_first_elo"] = shadow_ao

    _validate_shadow_result(result, production, eligible)
    return result


def _validate_production_seed_frame(production: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(production, pd.DataFrame):
        raise TypeError("production must be a pandas DataFrame")
    if production.empty:
        raise ValueError("Production seed frame cannot be empty")
    if production.columns.duplicated().any():
        duplicates = production.columns[production.columns.duplicated()].tolist()
        raise ValueError(f"Production seed frame has duplicate columns: {duplicates}")
    missing = sorted(REQUIRED_COLUMNS - set(production.columns))
    if missing:
        raise ValueError(f"Production seed frame is missing columns: {missing}")

    team_ids = production["team_id"].astype("string").str.strip()
    if team_ids.isna().any() or team_ids.eq("").any():
        raise ValueError("team_id values must be non-empty")
    if team_ids.duplicated().any():
        duplicates = team_ids.loc[team_ids.duplicated(keep=False)].unique().tolist()
        raise ValueError(f"Production seed frame has duplicate team_id values: {duplicates}")

    numeric = production.loc[:, NUMERIC_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    finite = np.isfinite(numeric.to_numpy(dtype=float))
    if not finite.all():
        row_number, column_number = np.argwhere(~finite)[0]
        column = NUMERIC_COLUMNS[int(column_number)]
        team_id = str(production.iloc[int(row_number)]["team_id"])
        raise ValueError(f"{column} must be finite for team_id: {team_id}")

    exposure = numeric["effective_european_exposure"]
    outside_range = exposure.lt(0.0) | exposure.gt(POSITIVE_BRIDGE_EXPOSURE_CAP)
    if outside_range.any():
        team_ids = production.loc[outside_range, "team_id"].astype(str).tolist()
        raise ValueError(
            "effective_european_exposure must be within [0, 0.65] for team_id "
            f"values: {team_ids[:5]}"
        )
    return numeric


def _validate_shadow_result(
    result: pd.DataFrame,
    production: pd.DataFrame,
    eligible: pd.Series,
) -> None:
    inactive = ~eligible
    if not result.loc[inactive, "ao_first_elo"].equals(
        production.loc[inactive, "ao_first_elo"]
    ):
        raise AssertionError("Positive Bridge changed an ineligible AO First row")
    if not result.loc[inactive, "shadow_effective_european_exposure"].equals(
        pd.to_numeric(
            production.loc[inactive, "effective_european_exposure"],
            errors="raise",
        )
    ):
        raise AssertionError("Positive Bridge changed an ineligible exposure row")

    deltas = result["positive_bridge_elo_delta"].to_numpy(dtype=float)
    if not np.isfinite(deltas).all():
        raise AssertionError("Positive Bridge produced a non-finite Elo adjustment")
    if (result.loc[eligible, "positive_bridge_elo_delta"] <= 0.0).any():
        raise AssertionError("An eligible Positive Bridge row did not increase")
    if (result["shadow_effective_european_exposure"] > POSITIVE_BRIDGE_EXPOSURE_CAP).any():
        raise AssertionError("Positive Bridge exposure exceeded the frozen cap")
