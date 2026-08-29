from __future__ import annotations

import bisect
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from ao_elo.validators import require_utc_timestamp


PROBABILITY_EPSILON = 1e-12

AO_CALIBRATION_FEATURES = (
    "ao_log_home_draw",
    "ao_log_away_draw",
    "is_single_match_tie",
    "is_neutral",
)

STRUCTURAL_NUMERIC_FEATURES = AO_CALIBRATION_FEATURES + (
    "expected_home_score",
    "live_rating_difference",
    "initial_rating_difference",
    "exposure_difference",
    "home_live_change",
    "away_live_change",
    "home_euro_matches_pre",
    "away_euro_matches_pre",
    "home_euro_residual_h3",
    "away_euro_residual_h3",
    "home_euro_residual_h8",
    "away_euro_residual_h8",
    "home_euro_goals_for_5",
    "away_euro_goals_for_5",
    "home_euro_goals_against_5",
    "away_euro_goals_against_5",
    "home_euro_home_residual",
    "away_euro_away_residual",
    "home_days_since_any_match",
    "away_days_since_any_match",
    "home_matches_14d",
    "away_matches_14d",
    "home_matches_30d",
    "away_matches_30d",
    "tie_matches_played_pre",
    "aggregate_home_lead_pre",
    "leg_number",
    "round_sequence",
    "month",
)

STRUCTURAL_CATEGORICAL_FEATURES = (
    "competition",
    "stage",
    "round",
    "format_type",
)

DOMESTIC_NUMERIC_FEATURES = (
    "home_domestic_rating_percentile",
    "away_domestic_rating_percentile",
    "domestic_rating_percentile_difference",
    "home_domestic_residual_5",
    "away_domestic_residual_5",
    "home_domestic_residual_10",
    "away_domestic_residual_10",
    "home_domestic_goals_for_5",
    "away_domestic_goals_for_5",
    "home_domestic_goals_against_5",
    "away_domestic_goals_against_5",
    "home_domestic_home_residual",
    "away_domestic_away_residual",
    "home_domestic_trend",
    "away_domestic_trend",
    "home_domestic_volatility",
    "away_domestic_volatility",
    "home_domestic_match_count",
    "away_domestic_match_count",
    "home_domestic_reliability",
    "away_domestic_reliability",
    "home_domestic_days_since",
    "away_domestic_days_since",
    "home_domestic_matches_14d",
    "away_domestic_matches_14d",
    "home_domestic_matches_30d",
    "away_domestic_matches_30d",
    "home_domestic_covered",
    "away_domestic_covered",
    "both_domestic_covered",
)


@dataclass(frozen=True)
class FeatureSchema:
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]

    @property
    def columns(self) -> tuple[str, ...]:
        return self.numeric + self.categorical


FEATURE_SCHEMAS: Mapping[str, FeatureSchema] = {
    "AO_CALIBRATION": FeatureSchema(AO_CALIBRATION_FEATURES, ()),
    "STRUCTURAL_LOGISTIC": FeatureSchema(
        STRUCTURAL_NUMERIC_FEATURES,
        STRUCTURAL_CATEGORICAL_FEATURES,
    ),
    "DOMESTIC_LOGISTIC": FeatureSchema(
        STRUCTURAL_NUMERIC_FEATURES + DOMESTIC_NUMERIC_FEATURES,
        STRUCTURAL_CATEGORICAL_FEATURES,
    ),
    "HIST_GRADIENT_BOOSTING": FeatureSchema(
        STRUCTURAL_NUMERIC_FEATURES + DOMESTIC_NUMERIC_FEATURES,
        STRUCTURAL_CATEGORICAL_FEATURES,
    ),
}


def build_pre_match_feature_store(
    production_predictions: pd.DataFrame,
    domestic_matches: pd.DataFrame,
    *,
    match_metadata: pd.DataFrame | None = None,
    initial_context: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one leakage-safe row per European match.

    Production predictions already contain the exact pre-match AO Live state. This
    function only derives prediction features; it never replays or mutates ratings.
    """
    baseline = _validated_baseline(production_predictions)
    metadata = _validated_metadata(match_metadata)
    if metadata is not None:
        # Align explicitly: pandas' implicit _x/_y suffixes would hide both
        # supplied values from the defaults below.
        metadata = metadata.set_index("match_id")
        for column in metadata.columns:
            supplied = baseline["match_id"].map(metadata[column])
            if column not in baseline:
                baseline[column] = supplied
                continue
            current = baseline[column]
            both = current.notna() & supplied.notna()
            left, right = current[both], supplied[both]
            if column == "is_knockout":
                left, right = left.map(_as_bool), right.map(_as_bool)
            elif column in {"round_sequence", "leg_number"}:
                left = pd.to_numeric(left, errors="raise")
                right = pd.to_numeric(right, errors="raise")
            if not left.eq(right).all():
                raise ValueError(f"Conflicting match metadata: {column}")
            baseline[column] = current.where(current.notna(), supplied)
    for column, default in (
        ("round", "UNKNOWN"),
        ("round_sequence", -1.0),
        ("leg_number", 0.0),
        ("is_knockout", False),
    ):
        if column not in baseline:
            baseline[column] = default
        baseline[column] = baseline[column].fillna(default)
    for column in ("is_single_match_tie", "is_neutral", "is_knockout"):
        baseline[column] = baseline[column].map(_as_bool)

    initial = _build_initial_context(baseline, initial_context)
    domestic_snapshots = build_domestic_snapshots(domestic_matches)
    rows: list[dict[str, object]] = []
    european_history: dict[str, list[dict[str, object]]] = defaultdict(list)
    tie_goals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    tie_matches: dict[str, int] = defaultdict(int)

    ordered = baseline.sort_values(["kickoff_utc", "match_id"], kind="stable")
    for kickoff, batch in ordered.groupby("kickoff_utc", sort=True):
        pending_updates: list[pd.Series] = []
        timestamp = pd.Timestamp(kickoff)
        for match in batch.sort_values("match_id", kind="stable").itertuples(index=False):
            home_id = str(match.home_club_id)
            away_id = str(match.away_club_id)
            home_initial = initial[(str(match.season), home_id)]
            away_initial = initial[(str(match.season), away_id)]
            home_euro = _european_history_features(european_history[home_id], timestamp)
            away_euro = _european_history_features(european_history[away_id], timestamp)
            home_domestic = _domestic_features_at(domestic_snapshots.get(home_id, []), timestamp)
            away_domestic = _domestic_features_at(domestic_snapshots.get(away_id, []), timestamp)
            home_days_any = _minimum_finite(
                home_euro["days_since"], home_domestic["days_since"]
            )
            away_days_any = _minimum_finite(
                away_euro["days_since"], away_domestic["days_since"]
            )

            tie_id = None if pd.isna(match.tie_id) else str(match.tie_id)
            prior_tie_matches = 0 if tie_id is None else tie_matches[tie_id]
            aggregate_lead = 0
            if tie_id is not None:
                aggregate_lead = tie_goals[tie_id][home_id] - tie_goals[tie_id][away_id]
            is_single = bool(match.is_single_match_tie)
            is_knockout = _as_bool(match.is_knockout)
            format_type = (
                "SINGLE_MATCH"
                if is_single
                else "TWO_LEG"
                if is_knockout and tie_id is not None
                else "LEAGUE_OR_GROUP"
            )
            probabilities = np.array(
                [match.home_probability, match.draw_probability, match.away_probability],
                dtype=float,
            )
            probabilities = np.maximum(probabilities, PROBABILITY_EPSILON)
            row: dict[str, object] = {
                "match_id": str(match.match_id),
                "season": str(match.season),
                "kickoff_utc": timestamp,
                "competition": str(match.competition),
                "stage": str(match.stage),
                "round": str(match.round),
                "format_type": format_type,
                "home_team_id": int(match.home_team_id),
                "away_team_id": int(match.away_team_id),
                "home_club_id": home_id,
                "away_club_id": away_id,
                "home_goals": int(match.home_goals),
                "away_goals": int(match.away_goals),
                "actual_class": int(match.actual_class),
                "tie_id": tie_id,
                "is_single_match_tie": int(is_single),
                "is_neutral": int(bool(match.is_neutral)),
                "round_sequence": _finite_or_default(match.round_sequence, -1.0),
                "leg_number": _finite_or_default(match.leg_number, 0.0),
                "month": int(timestamp.month),
                "ao_home_probability": float(probabilities[0]),
                "ao_draw_probability": float(probabilities[1]),
                "ao_away_probability": float(probabilities[2]),
                "ao_log_home_draw": float(math.log(probabilities[0] / probabilities[1])),
                "ao_log_away_draw": float(math.log(probabilities[2] / probabilities[1])),
                "expected_home_score": float(match.expected_home_score),
                "home_live_pre": float(match.home_live_pre),
                "away_live_pre": float(match.away_live_pre),
                "live_rating_difference": float(match.home_live_pre - match.away_live_pre),
                "home_initial_rating": home_initial["initial_rating"],
                "away_initial_rating": away_initial["initial_rating"],
                "initial_rating_difference": home_initial["initial_rating"]
                - away_initial["initial_rating"],
                "home_effective_exposure": home_initial["exposure"],
                "away_effective_exposure": away_initial["exposure"],
                "exposure_difference": home_initial["exposure"] - away_initial["exposure"],
                "home_live_change": float(match.home_live_pre) - home_initial["initial_rating"],
                "away_live_change": float(match.away_live_pre) - away_initial["initial_rating"],
                "tie_matches_played_pre": prior_tie_matches,
                "aggregate_home_lead_pre": aggregate_lead,
                "home_days_since_any_match": home_days_any,
                "away_days_since_any_match": away_days_any,
                "home_matches_14d": home_euro["matches_14d"] + home_domestic["matches_14d"],
                "away_matches_14d": away_euro["matches_14d"] + away_domestic["matches_14d"],
                "home_matches_30d": home_euro["matches_30d"] + home_domestic["matches_30d"],
                "away_matches_30d": away_euro["matches_30d"] + away_domestic["matches_30d"],
                "baseline_power_delta": float(match.power_delta),
            }
            _add_european_side(row, "home", home_euro)
            _add_european_side(row, "away", away_euro)
            _add_domestic_side(row, "home", home_domestic)
            _add_domestic_side(row, "away", away_domestic)
            row["domestic_rating_percentile_difference"] = (
                row["home_domestic_rating_percentile"]
                - row["away_domestic_rating_percentile"]
                if row["home_domestic_covered"] and row["away_domestic_covered"]
                else np.nan
            )
            row["both_domestic_covered"] = int(
                bool(row["home_domestic_covered"] and row["away_domestic_covered"])
            )
            rows.append(row)
            pending_updates.append(pd.Series(match._asdict()))

        for match in pending_updates:
            home_id = str(match["home_club_id"])
            away_id = str(match["away_club_id"])
            actual_home = float(match["actual_home_score"])
            expected_home = float(match["expected_home_score"])
            european_history[home_id].append(
                _history_record(timestamp, actual_home - expected_home, int(match["home_goals"]), int(match["away_goals"]), 1)
            )
            european_history[away_id].append(
                _history_record(timestamp, expected_home - actual_home, int(match["away_goals"]), int(match["home_goals"]), -1)
            )
            if not pd.isna(match["tie_id"]):
                tie_id = str(match["tie_id"])
                tie_goals[tie_id][home_id] += int(match["home_goals"])
                tie_goals[tie_id][away_id] += int(match["away_goals"])
                tie_matches[tie_id] += 1

    result = pd.DataFrame(rows).sort_values(["kickoff_utc", "match_id"], kind="stable")
    validate_feature_store(result, expected_rows=len(baseline))
    return result.reset_index(drop=True)


def build_domestic_snapshots(
    domestic_matches: pd.DataFrame,
) -> dict[str, list[dict[str, object]]]:
    required = {
        "source_event_id",
        "sportsdb_league_id",
        "kickoff_utc",
        "home_source_team_id",
        "away_source_team_id",
        "home_goals",
        "away_goals",
        "home_ao_club_id",
        "away_ao_club_id",
    }
    missing = sorted(required - set(domestic_matches.columns))
    if missing:
        raise ValueError(f"domestic_matches missing columns: {missing}")
    data = domestic_matches.copy()
    data["kickoff_utc"] = data["kickoff_utc"].map(
        lambda value: require_utc_timestamp(value, "domestic_matches.kickoff_utc")
    )
    if data["kickoff_utc"].isna().any():
        raise ValueError("domestic_matches.kickoff_utc contains invalid timestamps")
    if data["source_event_id"].isna().any() or data["source_event_id"].duplicated().any():
        raise ValueError("domestic_matches.source_event_id must be non-null and unique")
    for column in ("home_goals", "away_goals"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
        if data[column].isna().any() or (data[column] < 0).any():
            raise ValueError(f"domestic_matches.{column} must be non-negative numeric")

    ratings: dict[tuple[str, str], float] = defaultdict(float)
    histories: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    league_teams: dict[str, set[tuple[str, str]]] = defaultdict(set)
    snapshots: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in data.sort_values(["kickoff_utc", "source_event_id"], kind="stable").itertuples(index=False):
        league = str(row.sportsdb_league_id)
        home_key = (league, str(row.home_source_team_id))
        away_key = (league, str(row.away_source_team_id))
        league_teams[league].update((home_key, away_key))
        home_rating, away_rating = ratings[home_key], ratings[away_key]
        expected_home = 1.0 / (1.0 + 10.0 ** (-(home_rating - away_rating + 60.0) / 400.0))
        actual_home = 1.0 if row.home_goals > row.away_goals else 0.0 if row.home_goals < row.away_goals else 0.5
        delta = 20.0 * (actual_home - expected_home)
        ratings[home_key] += delta
        ratings[away_key] -= delta
        timestamp = pd.Timestamp(row.kickoff_utc)
        histories[home_key].append(
            _history_record(timestamp, actual_home - expected_home, int(row.home_goals), int(row.away_goals), 1)
        )
        histories[away_key].append(
            _history_record(timestamp, expected_home - actual_home, int(row.away_goals), int(row.home_goals), -1)
        )
        league_values = np.array([ratings[key] for key in league_teams[league]], dtype=float)
        for key, club_id in (
            (home_key, row.home_ao_club_id),
            (away_key, row.away_ao_club_id),
        ):
            if pd.isna(club_id) or not str(club_id).strip():
                continue
            summary = _rolling_history_summary(histories[key])
            summary.update(
                {
                    "timestamp": timestamp,
                    "rating_percentile": float(np.mean(league_values <= ratings[key])),
                }
            )
            snapshots[str(club_id)].append(summary)
    for club_id in snapshots:
        snapshots[club_id].sort(key=lambda value: value["timestamp"])
    return dict(snapshots)


def validate_feature_store(frame: pd.DataFrame, *, expected_rows: int | None = None) -> None:
    required = {
        "match_id",
        "season",
        "kickoff_utc",
        "home_club_id",
        "away_club_id",
        "actual_class",
        "ao_home_probability",
        "ao_draw_probability",
        "ao_away_probability",
    }
    for schema in FEATURE_SCHEMAS.values():
        required.update(schema.columns)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"feature_store missing columns: {missing}")
    if expected_rows is not None and len(frame) != expected_rows:
        raise ValueError(f"feature_store rows={len(frame)} expected={expected_rows}")
    if frame["match_id"].isna().any() or frame["match_id"].duplicated().any():
        raise ValueError("feature_store.match_id must be non-null and unique")
    kickoff = frame["kickoff_utc"].map(
        lambda value: require_utc_timestamp(value, "feature_store.kickoff_utc")
    )
    if kickoff.isna().any() or not kickoff.is_monotonic_increasing:
        raise ValueError("feature_store must have valid monotonic kickoff_utc")
    probabilities = frame[
        ["ao_home_probability", "ao_draw_probability", "ao_away_probability"]
    ].to_numpy(float)
    if not np.isfinite(probabilities).all() or (probabilities < 0.0).any():
        raise ValueError("feature_store AO probabilities must be finite and non-negative")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("feature_store AO probabilities must sum to one")
    if not frame["actual_class"].isin((0, 1, 2)).all():
        raise ValueError("feature_store.actual_class must be 0, 1, or 2")
    forbidden = {
        "current_match_xg_home",
        "current_match_xg_away",
        "post_match_rating",
        "advanced_team_id",
    }
    if forbidden.intersection(FEATURE_SCHEMAS["HIST_GRADIENT_BOOSTING"].columns):
        raise ValueError("Post-outcome fields cannot be ML features")


def _validated_baseline(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "match_id",
        "season",
        "kickoff_utc",
        "competition",
        "stage",
        "tie_id",
        "is_single_match_tie",
        "home_team_id",
        "away_team_id",
        "home_club_id",
        "away_club_id",
        "home_goals",
        "away_goals",
        "actual_class",
        "actual_home_score",
        "is_neutral",
        "home_live_pre",
        "away_live_pre",
        "expected_home_score",
        "home_probability",
        "draw_probability",
        "away_probability",
        "power_delta",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"production_predictions missing columns: {missing}")
    result = frame.copy()
    result["kickoff_utc"] = result["kickoff_utc"].map(
        lambda value: require_utc_timestamp(value, "production_predictions.kickoff_utc")
    )
    if result["kickoff_utc"].isna().any():
        raise ValueError("production_predictions.kickoff_utc contains invalid values")
    if result["match_id"].isna().any() or result["match_id"].duplicated().any():
        raise ValueError("production_predictions.match_id must be non-null and unique")
    if result[["home_club_id", "away_club_id"]].isna().any().any():
        raise ValueError("production_predictions requires permanent club IDs")
    return result


def _validated_metadata(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    if frame is None:
        return None
    if "match_id" not in frame or frame["match_id"].isna().any() or frame["match_id"].duplicated().any():
        raise ValueError("match_metadata.match_id must be non-null and unique")
    columns = [
        column
        for column in ("match_id", "round", "round_sequence", "leg_number", "is_knockout")
        if column in frame
    ]
    return frame[columns].copy()


def _build_initial_context(
    baseline: pd.DataFrame,
    supplied: pd.DataFrame | None,
) -> dict[tuple[str, str], dict[str, float]]:
    appearances = pd.concat(
        [
            baseline[["season", "kickoff_utc", "home_club_id", "home_live_pre"]].rename(
                columns={"home_club_id": "club_id", "home_live_pre": "initial_rating"}
            ),
            baseline[["season", "kickoff_utc", "away_club_id", "away_live_pre"]].rename(
                columns={"away_club_id": "club_id", "away_live_pre": "initial_rating"}
            ),
        ],
        ignore_index=True,
    )
    first = (
        appearances.sort_values(["season", "club_id", "kickoff_utc"], kind="stable")
        .groupby(["season", "club_id"], as_index=False)
        .first()
    )
    first["exposure"] = 0.0
    if supplied is not None:
        required = {"season", "club_id", "effective_european_exposure"}
        missing = sorted(required - set(supplied.columns))
        if missing:
            raise ValueError(f"initial_context missing columns: {missing}")
        if supplied.duplicated(["season", "club_id"]).any():
            raise ValueError("initial_context season+club_id must be unique")
        context = supplied.copy()
        rating_column = (
            "adjusted_ao_first_elo"
            if "adjusted_ao_first_elo" in context
            else "ao_first_elo"
            if "ao_first_elo" in context
            else None
        )
        selected = context[["season", "club_id", "effective_european_exposure"]].rename(
            columns={"effective_european_exposure": "supplied_exposure"}
        )
        if rating_column:
            selected["supplied_initial_rating"] = pd.to_numeric(
                context[rating_column], errors="coerce"
            )
        first = first.merge(selected, on=["season", "club_id"], how="left", validate="one_to_one")
        first["exposure"] = first["supplied_exposure"].fillna(0.0)
        if "supplied_initial_rating" in first:
            first["initial_rating"] = first["supplied_initial_rating"].fillna(first["initial_rating"])
    if not np.isfinite(first[["initial_rating", "exposure"]].to_numpy(float)).all():
        raise ValueError("Initial rating context must be finite")
    if ((first["exposure"] < 0.0) | (first["exposure"] > 1.0)).any():
        raise ValueError("Initial exposure must be in [0,1]")
    return {
        (str(row.season), str(row.club_id)): {
            "initial_rating": float(row.initial_rating),
            "exposure": float(row.exposure),
        }
        for row in first.itertuples(index=False)
    }


def _history_record(
    timestamp: pd.Timestamp,
    residual: float,
    goals_for: int,
    goals_against: int,
    venue: int,
) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "residual": float(residual),
        "goals_for": int(goals_for),
        "goals_against": int(goals_against),
        "venue": int(venue),
    }


def _rolling_history_summary(history: list[dict[str, object]]) -> dict[str, float]:
    residuals = np.array([float(row["residual"]) for row in history], dtype=float)
    goals_for = np.array([float(row["goals_for"]) for row in history], dtype=float)
    goals_against = np.array([float(row["goals_against"]) for row in history], dtype=float)
    venues = np.array([int(row["venue"]) for row in history], dtype=int)
    last5 = residuals[-5:]
    prior5 = residuals[-10:-5]
    return {
        "residual_5": _mean_or_zero(last5),
        "residual_10": _mean_or_zero(residuals[-10:]),
        "goals_for_5": _mean_or_zero(goals_for[-5:]),
        "goals_against_5": _mean_or_zero(goals_against[-5:]),
        "home_residual": _mean_or_zero(residuals[venues == 1][-10:]),
        "away_residual": _mean_or_zero(residuals[venues == -1][-10:]),
        "trend": _mean_or_zero(last5) - _mean_or_zero(prior5),
        "volatility": float(np.std(residuals[-10:])) if len(residuals) >= 2 else 0.0,
        "match_count": float(len(history)),
        "reliability": float(len(history) / (len(history) + 10.0)),
    }


def _european_history_features(
    history: list[dict[str, object]],
    kickoff: pd.Timestamp,
) -> dict[str, float]:
    if not history:
        return {
            "matches": 0.0,
            "residual_h3": 0.0,
            "residual_h8": 0.0,
            "goals_for_5": 0.0,
            "goals_against_5": 0.0,
            "home_residual": 0.0,
            "away_residual": 0.0,
            "days_since": np.nan,
            "matches_14d": 0.0,
            "matches_30d": 0.0,
        }
    residuals = np.array([float(row["residual"]) for row in history], dtype=float)
    goals_for = np.array([float(row["goals_for"]) for row in history], dtype=float)
    goals_against = np.array([float(row["goals_against"]) for row in history], dtype=float)
    venues = np.array([int(row["venue"]) for row in history], dtype=int)
    timestamps = [pd.Timestamp(row["timestamp"]) for row in history]
    return {
        "matches": float(len(history)),
        "residual_h3": _ewma_by_match(residuals, 3.0),
        "residual_h8": _ewma_by_match(residuals, 8.0),
        "goals_for_5": _mean_or_zero(goals_for[-5:]),
        "goals_against_5": _mean_or_zero(goals_against[-5:]),
        "home_residual": _mean_or_zero(residuals[venues == 1][-8:]),
        "away_residual": _mean_or_zero(residuals[venues == -1][-8:]),
        "days_since": max(0.0, (kickoff - timestamps[-1]).total_seconds() / 86400.0),
        "matches_14d": float(sum((kickoff - value).total_seconds() <= 14 * 86400 for value in timestamps)),
        "matches_30d": float(sum((kickoff - value).total_seconds() <= 30 * 86400 for value in timestamps)),
    }


def _domestic_features_at(
    snapshots: list[dict[str, object]],
    kickoff: pd.Timestamp,
) -> dict[str, float]:
    if not snapshots:
        return _empty_domestic_features()
    timestamps = [pd.Timestamp(row["timestamp"]) for row in snapshots]
    index = bisect.bisect_left(timestamps, kickoff) - 1
    if index < 0:
        return _empty_domestic_features()
    snapshot = snapshots[index]
    recent = timestamps[: index + 1]
    result = {key: float(value) for key, value in snapshot.items() if key != "timestamp"}
    result.update(
        {
            "covered": 1.0,
            "days_since": max(0.0, (kickoff - timestamps[index]).total_seconds() / 86400.0),
            "matches_14d": float(sum((kickoff - value).total_seconds() <= 14 * 86400 for value in recent)),
            "matches_30d": float(sum((kickoff - value).total_seconds() <= 30 * 86400 for value in recent)),
        }
    )
    return result


def _empty_domestic_features() -> dict[str, float]:
    return {
        "rating_percentile": np.nan,
        "residual_5": np.nan,
        "residual_10": np.nan,
        "goals_for_5": np.nan,
        "goals_against_5": np.nan,
        "home_residual": np.nan,
        "away_residual": np.nan,
        "trend": np.nan,
        "volatility": np.nan,
        "match_count": 0.0,
        "reliability": 0.0,
        "covered": 0.0,
        "days_since": np.nan,
        "matches_14d": 0.0,
        "matches_30d": 0.0,
    }


def _add_european_side(row: dict[str, object], side: str, values: Mapping[str, float]) -> None:
    row[f"{side}_euro_matches_pre"] = values["matches"]
    row[f"{side}_euro_residual_h3"] = values["residual_h3"]
    row[f"{side}_euro_residual_h8"] = values["residual_h8"]
    row[f"{side}_euro_goals_for_5"] = values["goals_for_5"]
    row[f"{side}_euro_goals_against_5"] = values["goals_against_5"]
    row[f"{side}_euro_home_residual"] = values["home_residual"]
    row[f"{side}_euro_away_residual"] = values["away_residual"]


def _add_domestic_side(row: dict[str, object], side: str, values: Mapping[str, float]) -> None:
    for key in (
        "rating_percentile",
        "residual_5",
        "residual_10",
        "goals_for_5",
        "goals_against_5",
        "home_residual",
        "away_residual",
        "trend",
        "volatility",
        "match_count",
        "reliability",
        "days_since",
        "matches_14d",
        "matches_30d",
        "covered",
    ):
        row[f"{side}_domestic_{key}"] = values[key]


def _ewma_by_match(values: np.ndarray, half_life: float) -> float:
    if len(values) == 0:
        return 0.0
    ages = np.arange(len(values) - 1, -1, -1, dtype=float)
    weights = np.power(0.5, ages / half_life)
    return float(np.average(values, weights=weights))


def _mean_or_zero(values: Iterable[float]) -> float:
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    return float(array.mean()) if len(array) else 0.0


def _minimum_finite(left: float, right: float) -> float:
    finite = [float(value) for value in (left, right) if math.isfinite(float(value))]
    return min(finite) if finite else np.nan


def _finite_or_default(value: object, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0", ""}:
            return False
    if value in (0, 1):
        return bool(value)
    raise ValueError(f"Cannot coerce boolean value: {value!r}")
