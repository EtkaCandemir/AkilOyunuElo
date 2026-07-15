from __future__ import annotations

from math import log

import pandas as pd


MATCH_COLUMNS = {
    "season",
    "home_team_id",
    "away_team_id",
    "home_goals",
    "away_goals",
}
RATING_COLUMNS = {"season", "team_id", "ao_first_elo"}


def evaluate_match_predictions(
    ratings: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    home_advantage: float = 0.0,
    elo_scale: float = 400.0,
) -> dict[str, float | int]:
    """Evaluate fixed pre-season ratings against completed match outcomes."""
    _require_columns(ratings, RATING_COLUMNS, "ratings")
    _require_columns(matches, MATCH_COLUMNS, "matches")
    if elo_scale <= 0:
        raise ValueError("elo_scale must be > 0")
    if ratings.duplicated(["season", "team_id"]).any():
        raise ValueError("ratings must be unique by season + team_id")

    data = matches.copy()
    if (data["home_team_id"] == data["away_team_id"]).any():
        raise ValueError("matches cannot contain the same home and away team")
    for column in ("home_goals", "away_goals"):
        numeric = pd.to_numeric(data[column], errors="coerce")
        if numeric.isna().any() or (numeric < 0).any():
            raise ValueError(f"matches.{column} must contain non-negative numbers")
        data[column] = numeric

    home = ratings[["season", "team_id", "ao_first_elo"]].rename(
        columns={"team_id": "home_team_id", "ao_first_elo": "home_rating"}
    )
    away = ratings[["season", "team_id", "ao_first_elo"]].rename(
        columns={"team_id": "away_team_id", "ao_first_elo": "away_rating"}
    )
    data = data.merge(home, on=["season", "home_team_id"], how="left", validate="many_to_one")
    data = data.merge(away, on=["season", "away_team_id"], how="left", validate="many_to_one")
    missing = data["home_rating"].isna() | data["away_rating"].isna()
    if missing.any():
        keys = data.loc[missing, ["season", "home_team_id", "away_team_id"]].head(5)
        raise ValueError(f"matches contain teams without ratings: {keys.to_dict('records')}")

    difference = data["home_rating"] - data["away_rating"] + float(home_advantage)
    data["expected_home_score"] = 1.0 / (1.0 + 10.0 ** (-difference / elo_scale))
    data["actual_home_score"] = 0.5
    data.loc[data["home_goals"] > data["away_goals"], "actual_home_score"] = 1.0
    data.loc[data["home_goals"] < data["away_goals"], "actual_home_score"] = 0.0

    probability = data["expected_home_score"].clip(1e-12, 1 - 1e-12)
    actual = data["actual_home_score"]
    log_loss = -(
        actual * probability.map(log) + (1.0 - actual) * (1.0 - probability).map(log)
    ).mean()
    brier = ((probability - actual) ** 2).mean()

    decisive = data["home_goals"] != data["away_goals"]
    predicted_home_win = probability > 0.5
    actual_home_win = data["home_goals"] > data["away_goals"]
    decisive_accuracy = (predicted_home_win[decisive] == actual_home_win[decisive]).mean()

    team_performance = _team_performance(data)
    rating_values = ratings[["season", "team_id", "ao_first_elo"]]
    rank_data = rating_values.merge(team_performance, on=["season", "team_id"], validate="one_to_one")
    spearman_by_season = rank_data.groupby("season", sort=False).apply(
        _safe_spearman,
        include_groups=False,
    )

    return {
        "matches": int(len(data)),
        "log_loss": float(log_loss),
        "brier": float(brier),
        "decisive_accuracy": float(decisive_accuracy),
        "mean_season_spearman": float(spearman_by_season.mean()),
    }


def _team_performance(matches: pd.DataFrame) -> pd.DataFrame:
    home = matches[["season", "home_team_id", "actual_home_score"]].rename(
        columns={"home_team_id": "team_id", "actual_home_score": "score"}
    )
    away = matches[["season", "away_team_id", "actual_home_score"]].rename(
        columns={"away_team_id": "team_id"}
    )
    away["score"] = 1.0 - away["actual_home_score"]
    values = pd.concat([home[["season", "team_id", "score"]], away[["season", "team_id", "score"]]])
    return (
        values.groupby(["season", "team_id"], as_index=False)["score"]
        .mean()
        .rename(columns={"score": "actual_score_rate"})
    )


def _safe_spearman(frame: pd.DataFrame) -> float:
    if frame["ao_first_elo"].nunique() < 2 or frame["actual_score_rate"].nunique() < 2:
        return float("nan")
    return float(frame["ao_first_elo"].corr(frame["actual_score_rate"], method="spearman"))


def _require_columns(data: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {', '.join(missing)}")
