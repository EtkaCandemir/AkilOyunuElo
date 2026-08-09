from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "snapshot_date",
    "source_url",
    "country_code",
    "league_name",
    "league_average_rating",
    "top_10_average_rating",
    "top_5_average_rating",
    "source_sha256",
}


def read_and_validate_snapshot(path: Path) -> pd.DataFrame:
    snapshot = pd.read_csv(path)
    validate_snapshot(snapshot)
    return snapshot.sort_values(["country_code", "league_name"], kind="stable").reset_index(drop=True)


def validate_snapshot(snapshot: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS.difference(snapshot.columns))
    if missing:
        raise ValueError(f"Opta snapshot missing columns: {missing}")
    if snapshot.empty:
        raise ValueError("Opta snapshot cannot be empty")
    if snapshot.duplicated(["snapshot_date", "country_code", "league_name"]).any():
        raise ValueError("Opta snapshot has duplicate league rows")
    dates = pd.to_datetime(snapshot["snapshot_date"], utc=True, errors="coerce")
    if dates.isna().any():
        raise ValueError("Opta snapshot_date must be valid ISO date values")
    if snapshot["source_url"].isna().any() or snapshot["source_sha256"].isna().any():
        raise ValueError("Opta snapshot requires source URL and checksum")
    if snapshot["country_code"].astype(str).str.fullmatch(r"[A-Z]{3}").eq(False).any():
        raise ValueError("Opta country_code must be uppercase ISO-style three-letter code")
    numeric = snapshot[["league_average_rating", "top_10_average_rating", "top_5_average_rating"]].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Opta rating values must be finite numeric values")
    if ((numeric < 0.0) | (numeric > 100.0)).any().any():
        raise ValueError("Opta ratings must be in [0,100]")
    if numeric["top_5_average_rating"].lt(numeric["top_10_average_rating"]).any():
        raise ValueError("Opta top-five average cannot be below top-ten average")
    if numeric["top_10_average_rating"].lt(numeric["league_average_rating"]).any():
        raise ValueError("Opta top-ten average cannot be below league average")


def compare_snapshot_to_ao_country_strength(
    snapshot: pd.DataFrame,
    country_coefficients: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_snapshot(snapshot)
    required = {"country_code", "official_country_rank"}
    missing = sorted(required.difference(country_coefficients.columns))
    if missing:
        raise ValueError(f"country coefficients missing columns: {missing}")
    ao = country_coefficients[["country_code", "official_country_rank"]].copy()
    ao["official_country_rank"] = pd.to_numeric(ao["official_country_rank"], errors="coerce")
    if ao["official_country_rank"].isna().any() or ao["official_country_rank"].le(0).any():
        raise ValueError("official_country_rank must be positive numeric")
    values = snapshot.merge(ao, on="country_code", how="inner", validate="many_to_one")
    if values.empty:
        raise ValueError("Opta snapshot has no country overlap with AO country strengths")
    values["opta_league_rank"] = values["league_average_rating"].rank(method="min", ascending=False).astype(int)
    values["rank_difference_opta_minus_ao"] = values["opta_league_rank"] - values["official_country_rank"].astype(int)
    values = values.sort_values(["opta_league_rank", "country_code"], kind="stable").reset_index(drop=True)
    summary = pd.DataFrame(
        [
            {
                "leagues_compared": int(len(values)),
                "spearman_league_average_vs_ao_rank": float(
                    values["league_average_rating"].corr(-values["official_country_rank"], method="spearman")
                ),
                "mean_absolute_rank_difference": float(values["rank_difference_opta_minus_ao"].abs().mean()),
                "snapshot_date_min": str(values["snapshot_date"].min()),
                "snapshot_date_max": str(values["snapshot_date"].max()),
            }
        ]
    )
    return values, summary
