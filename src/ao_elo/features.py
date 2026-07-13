from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ao_elo.config import AOEuropeanEloConfig, SEASON_KEYS
from ao_elo.scoring import normalize_log_score


@dataclass(frozen=True)
class DomesticAchievement:
    domestic_position_percentile: float | None
    league_finish_score: float
    cup_base_score: float
    cup_double_bonus: float
    domestic_achievement_score: float


def is_missing(value: Any) -> bool:
    return pd.isna(value)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if is_missing(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "evet"}
    return bool(value)


def weighted_sum(row: pd.Series, prefix: str, config: AOEuropeanEloConfig) -> float:
    total = 0.0
    for key in SEASON_KEYS:
        value = row.get(f"{prefix}_{key}", 0.0)
        if is_missing(value):
            value = 0.0
        total += config.season_weights[key] * float(value)
    return total


def compute_weighted_country_score(
    country_row: pd.Series,
    config: AOEuropeanEloConfig,
) -> float:
    """Weighted Country Score model component."""
    return weighted_sum(country_row, "points", config)


def compute_league_strength(
    weighted_country_score: float,
    config: AOEuropeanEloConfig,
) -> tuple[float, float]:
    """League Strength normalization and gamma transform."""
    norm = normalize_log_score(
        weighted_country_score,
        float(config.country_strength_benchmark),
    )
    return norm, norm**config.gamma


def compute_domestic_achievement(
    domestic_position: Any,
    league_team_count: Any,
    is_league_champion: Any,
    is_cup_winner: Any,
    config: AOEuropeanEloConfig,
) -> DomesticAchievement:
    """Domestic Achievement Score from final league standing and cup result."""
    champion = as_bool(is_league_champion)
    cup_winner = as_bool(is_cup_winner)
    position_known = not is_missing(domestic_position) and not is_missing(
        league_team_count
    )

    percentile: float | None = None
    if not position_known:
        league_finish_score = config.unknown_league_finish_score
    else:
        position = float(domestic_position)
        team_count = float(league_team_count)
        if team_count <= 1:
            raise ValueError("league_team_count must be > 1 when position is known")
        if position < 1 or position > team_count:
            raise ValueError("domestic_position must be between 1 and league_team_count")
        percentile = (team_count - position) / (team_count - 1.0)
        percentile_score = config.percentile_floor + config.percentile_scale * (
            percentile**config.percentile_delta
        )
        champion_base = config.champion_base_score if champion else 0.0
        league_finish_score = max(percentile_score, champion_base)

    cup_base_score = config.cup_base_score if cup_winner else 0.0
    cup_double_bonus = (
        config.cup_double_bonus_multiplier * league_finish_score
        if cup_winner and position_known
        else 0.0
    )
    achievement_score = min(
        config.achievement_cap,
        max(league_finish_score, cup_base_score) + cup_double_bonus,
    )

    return DomesticAchievement(
        domestic_position_percentile=percentile,
        league_finish_score=league_finish_score,
        cup_base_score=cup_base_score,
        cup_double_bonus=cup_double_bonus,
        domestic_achievement_score=achievement_score,
    )


def compute_weighted_european_history(
    club_row: pd.Series,
    config: AOEuropeanEloConfig,
) -> float:
    """Weighted European History from club's own season points."""
    return weighted_sum(club_row, "club_points", config)


def compute_weighted_season_exposure(
    club_row: pd.Series,
    config: AOEuropeanEloConfig,
) -> float:
    """European Exposure season-count component."""
    return weighted_sum(club_row, "played", config)


def compute_weighted_match_exposure(
    club_row: pd.Series,
    config: AOEuropeanEloConfig,
) -> float:
    """European Exposure match-count component with season-specific caps."""
    total = 0.0
    for key in SEASON_KEYS:
        matches = club_row.get(f"matches_{key}", 0.0)
        cap = club_row.get(f"match_cap_{key}", 0.0)
        if is_missing(matches):
            matches = 0.0
        if is_missing(cap):
            cap = 0.0
        matches_float = float(matches)
        cap_float = float(cap)
        exposure = 0.0 if matches_float == 0 else min(1.0, matches_float / cap_float)
        total += config.season_weights[key] * exposure
    return total
