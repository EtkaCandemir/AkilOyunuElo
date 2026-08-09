from __future__ import annotations

import re

import pandas as pd


CLUB_ID_PATTERN = re.compile(r"^AO-UEFA-[1-9]\d*$")
TEAM_SEASON_KEY = ("season", "local_team_id")


def permanent_club_id(uefa_team_id: object) -> str:
    if isinstance(uefa_team_id, bool):
        raise ValueError("uefa_team_id must be a positive integer")
    text = str(uefa_team_id).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if not text.isdigit() or int(text) <= 0:
        raise ValueError("uefa_team_id must be a positive integer")
    return f"AO-UEFA-{int(text)}"


def validate_team_season_identity(identity: pd.DataFrame) -> None:
    required = {
        "season",
        "local_team_id",
        "team_name",
        "country_code",
        "club_id",
        "uefa_team_id",
    }
    missing = sorted(required - set(identity.columns))
    if missing:
        raise ValueError(f"Team-season identity missing columns: {missing}")
    if identity[list(TEAM_SEASON_KEY)].isna().any().any():
        raise ValueError("Team-season identity key cannot be null")
    if identity.duplicated(list(TEAM_SEASON_KEY)).any():
        raise ValueError("Duplicate season + local_team_id identity")
    if identity.duplicated(["season", "club_id"]).any():
        raise ValueError("One permanent club_id cannot map to two local teams in a season")
    if identity["club_id"].isna().any() or not identity["club_id"].map(
        lambda value: bool(CLUB_ID_PATTERN.fullmatch(str(value)))
    ).all():
        raise ValueError("club_id must use the AO-UEFA-<positive integer> contract")
    expected = identity["uefa_team_id"].map(permanent_club_id)
    if not expected.equals(identity["club_id"].astype(str)):
        raise ValueError("club_id and uefa_team_id disagree")
    if identity.groupby("club_id")["country_code"].nunique().gt(1).any():
        raise ValueError("A permanent club_id maps to multiple countries")


def validate_club_registry(registry: pd.DataFrame) -> None:
    required = {
        "club_id",
        "uefa_team_id",
        "canonical_name",
        "country_code",
        "first_season",
        "last_season",
    }
    missing = sorted(required - set(registry.columns))
    if missing:
        raise ValueError(f"Club registry missing columns: {missing}")
    if registry["club_id"].duplicated().any():
        raise ValueError("Club registry contains duplicate club_id")
    if registry["uefa_team_id"].astype(str).duplicated().any():
        raise ValueError("Club registry contains duplicate UEFA team ID")
    expected = registry["uefa_team_id"].map(permanent_club_id)
    if not expected.equals(registry["club_id"].astype(str)):
        raise ValueError("Club registry club_id and UEFA team ID disagree")
    if registry[["canonical_name", "country_code"]].isna().any().any():
        raise ValueError("Club registry canonical identity cannot be null")


def attach_permanent_club_id(
    frame: pd.DataFrame,
    identity: pd.DataFrame,
    *,
    team_id_column: str = "team_id",
    output_column: str = "club_id",
) -> pd.DataFrame:
    validate_team_season_identity(identity)
    required = {"season", team_id_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Frame missing identity key columns: {missing}")
    mapping = identity[["season", "local_team_id", "club_id"]].rename(
        columns={"local_team_id": team_id_column, "club_id": output_column}
    )
    result = frame.merge(
        mapping,
        on=["season", team_id_column],
        how="left",
        validate="many_to_one",
    )
    if result[output_column].isna().any():
        sample = result.loc[result[output_column].isna(), ["season", team_id_column]]
        raise ValueError(
            "Permanent club identity is missing for rows: "
            f"{sample.head(5).to_dict(orient='records')}"
        )
    return result


def attach_match_club_ids(
    matches: pd.DataFrame,
    identity: pd.DataFrame,
) -> pd.DataFrame:
    result = attach_permanent_club_id(
        matches,
        identity,
        team_id_column="home_team_id",
        output_column="home_club_id",
    )
    result = attach_permanent_club_id(
        result,
        identity,
        team_id_column="away_team_id",
        output_column="away_club_id",
    )
    if result["home_club_id"].eq(result["away_club_id"]).any():
        raise ValueError("A match cannot contain the same permanent club on both sides")
    return result


def identity_safe_forward_ranking(
    end_ratings: pd.DataFrame,
    target: pd.DataFrame,
    identity: pd.DataFrame,
    *,
    allowed_target_seasons: set[str],
) -> pd.DataFrame:
    from scripts.run_final_robustness import pairwise_ranking_accuracy

    predicted = attach_permanent_club_id(end_ratings, identity)
    actual = attach_permanent_club_id(target, identity)
    target_seasons = tuple(
        sorted(set(actual["season"].astype(str)) | set(predicted["season"].astype(str)))
    )
    previous_season = {
        target_seasons[index]: target_seasons[index - 1]
        for index in range(1, len(target_seasons))
    }
    rows: list[dict[str, object]] = []
    eligible = actual.loc[actual["season"].isin(allowed_target_seasons)]
    for (season, competition), actual_group in eligible.groupby(
        ["season", "competition"], sort=True
    ):
        source_season = previous_season.get(str(season))
        if source_season is None:
            continue
        predicted_group = predicted.loc[predicted["season"].eq(source_season)]
        table = actual_group[["club_id", "schedule_adjusted_score"]].merge(
            predicted_group[["club_id", "end_live_rating"]],
            on="club_id",
            validate="one_to_one",
        )
        if len(table) < 3:
            continue
        spearman = table["end_live_rating"].corr(
            table["schedule_adjusted_score"], method="spearman"
        )
        rows.append(
            {
                "source_season": source_season,
                "target_season": season,
                "competition": competition,
                "teams": len(table),
                "pair_weight": len(table) * (len(table) - 1) / 2,
                "ranking_score": float(spearman),
                "pairwise_accuracy": pairwise_ranking_accuracy(
                    table["end_live_rating"].to_numpy(float),
                    table["schedule_adjusted_score"].to_numpy(float),
                ),
            }
        )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return pd.DataFrame(
            columns=[
                "competition",
                "groups",
                "team_weight",
                "ranking_score",
                "pairwise_accuracy",
            ]
        )
    summaries = []
    for competition, group in [
        ("ALL", detail),
        *detail.groupby("competition", sort=True),
    ]:
        summaries.append(
            {
                "competition": competition,
                "groups": len(group),
                "team_weight": int(group["teams"].sum()),
                "ranking_score": float(
                    (group["ranking_score"] * group["teams"]).sum()
                    / group["teams"].sum()
                ),
                "pairwise_accuracy": float(
                    (group["pairwise_accuracy"] * group["pair_weight"]).sum()
                    / group["pair_weight"].sum()
                ),
            }
        )
    return pd.DataFrame(summaries)
