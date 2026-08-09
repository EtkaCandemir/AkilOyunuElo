from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from ao_elo.opponent_quintile_context import (
    QuintileProfile,
    contextual_matchup_expectation,
    estimate_quintile_profile,
    profile_to_record,
)


STRONG_OPPONENT = 1
EVEN_OPPONENT = 2
WEAK_OPPONENT = 3
RELATIVE_BANDS = (STRONG_OPPONENT, EVEN_OPPONENT, WEAK_OPPONENT)
RELATIVE_BAND_LABELS = {
    STRONG_OPPONENT: "STRONG",
    EVEN_OPPONENT: "EVEN",
    WEAK_OPPONENT: "WEAK",
}


@dataclass(frozen=True, order=True)
class RelativeOpponentProfileConfig:
    lower_expected_score: float
    upper_expected_score: float
    lookback_seasons: int
    season_decay: float
    domestic_weight: float
    shrinkage_matches: float
    effect_cap: float

    def validate(self) -> None:
        _probability("lower_expected_score", self.lower_expected_score)
        _probability("upper_expected_score", self.upper_expected_score)
        if self.lower_expected_score >= self.upper_expected_score:
            raise ValueError("relative profile lower bound must be below upper bound")
        if not isinstance(self.lookback_seasons, int) or isinstance(self.lookback_seasons, bool):
            raise ValueError("lookback_seasons must be an integer")
        if self.lookback_seasons not in (3, 5):
            raise ValueError("lookback_seasons must be either 3 or 5")
        _positive("season_decay", self.season_decay)
        if self.season_decay > 1.0:
            raise ValueError("season_decay must be <= 1")
        _non_negative("domestic_weight", self.domestic_weight)
        _non_negative("shrinkage_matches", self.shrinkage_matches)
        _non_negative("effect_cap", self.effect_cap)

    @property
    def key(self) -> str:
        return (
            f"relative_l{self.lower_expected_score:g}_u{self.upper_expected_score:g}"
            f"_w{self.lookback_seasons}_d{self.season_decay:g}"
            f"_dom{self.domestic_weight:g}_k{self.shrinkage_matches:g}_cap{self.effect_cap:g}"
        )


@dataclass(frozen=True)
class DomesticRatingConfig:
    elo_scale: float
    home_advantage: float
    k_factor: float
    season_carry: float = 0.75
    league_mean: float = 1500.0

    def validate(self) -> None:
        _positive("elo_scale", self.elo_scale)
        _non_negative("home_advantage", self.home_advantage)
        _non_negative("k_factor", self.k_factor)
        _probability("season_carry", self.season_carry, allow_zero=True)
        if not math.isfinite(self.league_mean):
            raise ValueError("league_mean must be finite")


def classify_relative_opponent(
    neutral_expected_score: float,
    *,
    lower: float,
    upper: float,
) -> int:
    _probability("neutral_expected_score", neutral_expected_score, allow_zero=False)
    _probability("lower", lower)
    _probability("upper", upper)
    if lower >= upper:
        raise ValueError("lower must be less than upper")
    if neutral_expected_score < lower:
        return STRONG_OPPONENT
    if neutral_expected_score > upper:
        return WEAK_OPPONENT
    return EVEN_OPPONENT


def elo_expected_score(
    own_rating: float,
    opponent_rating: float,
    *,
    elo_scale: float,
    home_advantage: float = 0.0,
) -> float:
    for name, value in (("own_rating", own_rating), ("opponent_rating", opponent_rating), ("home_advantage", home_advantage)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    _positive("elo_scale", elo_scale)
    difference = own_rating - opponent_rating + home_advantage
    return 1.0 / (1.0 + 10.0 ** (-difference / elo_scale))


def replay_domestic_perspectives(
    matches: pd.DataFrame,
    *,
    config: DomesticRatingConfig,
    lower: float,
    upper: float,
) -> pd.DataFrame:
    """Create causal domestic match perspectives without touching AO Live state."""

    config.validate()
    required = {
        "match_id", "sportsdb_league_id", "provider_season", "ao_season", "kickoff_utc",
        "home_source_team_id", "away_source_team_id", "home_ao_club_id", "away_ao_club_id",
        "actual_home_score",
    }
    missing = sorted(required.difference(matches.columns))
    if missing:
        raise ValueError(f"domestic matches missing columns: {missing}")
    data = matches.copy()
    data["kickoff_utc"] = pd.to_datetime(data["kickoff_utc"], utc=True, errors="coerce")
    if data["kickoff_utc"].isna().any():
        raise ValueError("domestic matches must have valid UTC timestamps")
    if data["match_id"].duplicated().any():
        raise ValueError("domestic match_id must be unique")
    if not data["actual_home_score"].isin((0.0, 0.5, 1.0)).all():
        raise ValueError("domestic actual_home_score must be 0, 0.5 or 1")
    rows: list[dict[str, object]] = []
    for league_id, league in data.groupby("sportsdb_league_id", sort=True):
        state: dict[str, float] = {}
        seen_season: str | None = None
        league = league.sort_values(["kickoff_utc", "match_id"], kind="stable")
        for provider_season, season_rows in league.groupby("provider_season", sort=False):
            teams = sorted(
                set(season_rows["home_source_team_id"].astype(str))
                | set(season_rows["away_source_team_id"].astype(str))
            )
            if seen_season is not None:
                for team_id in teams:
                    if team_id in state:
                        state[team_id] = config.league_mean + config.season_carry * (
                            state[team_id] - config.league_mean
                        )
            for team_id in teams:
                state.setdefault(team_id, config.league_mean)
            seen_season = str(provider_season)
            for _, batch in season_rows.groupby("kickoff_utc", sort=True):
                prepared: list[dict[str, object]] = []
                for match in batch.sort_values("match_id", kind="stable").itertuples(index=False):
                    home_key, away_key = str(match.home_source_team_id), str(match.away_source_team_id)
                    home_pre, away_pre = state[home_key], state[away_key]
                    home_expected = elo_expected_score(
                        home_pre,
                        away_pre,
                        elo_scale=config.elo_scale,
                        home_advantage=config.home_advantage,
                    )
                    home_neutral = elo_expected_score(home_pre, away_pre, elo_scale=config.elo_scale)
                    prepared.append(
                        {
                            "match": match,
                            "home_pre": home_pre,
                            "away_pre": away_pre,
                            "home_expected": home_expected,
                            "home_neutral": home_neutral,
                        }
                    )
                for item in prepared:
                    match = item["match"]
                    home_expected = float(item["home_expected"])
                    home_neutral = float(item["home_neutral"])
                    home_score = float(match.actual_home_score)
                    common = {
                        "match_id": str(match.match_id),
                        "source": "DOMESTIC",
                        "country_code": str(match.country_code),
                        "sportsdb_league_id": str(league_id),
                        "provider_season": str(match.provider_season),
                        "ao_season": str(match.ao_season),
                        "kickoff_utc": match.kickoff_utc,
                    }
                    if pd.notna(match.home_ao_club_id):
                        rows.append(
                            {
                                **common,
                                "club_id": str(match.home_ao_club_id),
                                "venue": "HOME",
                                "opponent_source_club_key": f"THESPORTSDB:{match.away_source_team_id}",
                                "expected_score": home_expected,
                                "neutral_expected_score": home_neutral,
                                "actual_score": home_score,
                                "opponent_band": classify_relative_opponent(home_neutral, lower=lower, upper=upper),
                            }
                        )
                    if pd.notna(match.away_ao_club_id):
                        away_expected, away_neutral = 1.0 - home_expected, 1.0 - home_neutral
                        rows.append(
                            {
                                **common,
                                "club_id": str(match.away_ao_club_id),
                                "venue": "AWAY",
                                "opponent_source_club_key": f"THESPORTSDB:{match.home_source_team_id}",
                                "expected_score": away_expected,
                                "neutral_expected_score": away_neutral,
                                "actual_score": 1.0 - home_score,
                                "opponent_band": classify_relative_opponent(away_neutral, lower=lower, upper=upper),
                            }
                        )
                for item in prepared:
                    match = item["match"]
                    delta = config.k_factor * (float(match.actual_home_score) - float(item["home_expected"]))
                    state[str(match.home_source_team_id)] += delta
                    state[str(match.away_source_team_id)] -= delta
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=_PERSPECTIVE_COLUMNS)
    return result.sort_values(["kickoff_utc", "match_id", "club_id"], kind="stable").reset_index(drop=True)


def build_european_perspectives(
    baseline: pd.DataFrame,
    *,
    elo_scale: float,
    lower: float,
    upper: float,
) -> pd.DataFrame:
    required = {
        "match_id", "season", "competition", "kickoff_utc", "home_club_id", "away_club_id",
        "home_live_pre", "away_live_pre", "expected_home_score", "actual_home_score",
    }
    missing = sorted(required.difference(baseline.columns))
    if missing:
        raise ValueError(f"baseline missing columns: {missing}")
    frame = baseline.copy()
    frame["kickoff_utc"] = pd.to_datetime(frame["kickoff_utc"], utc=True, errors="coerce")
    if frame["kickoff_utc"].isna().any() or frame["match_id"].duplicated().any():
        raise ValueError("European baseline requires unique match IDs and valid UTC times")
    rows: list[dict[str, object]] = []
    for match in frame.itertuples(index=False):
        home_neutral = elo_expected_score(float(match.home_live_pre), float(match.away_live_pre), elo_scale=elo_scale)
        home_expected = float(match.expected_home_score)
        score = float(match.actual_home_score)
        common = {
            "match_id": str(match.match_id),
            "source": "EUROPE",
            "country_code": pd.NA,
            "sportsdb_league_id": pd.NA,
            "provider_season": pd.NA,
            "ao_season": str(match.season),
            "competition": str(match.competition),
            "kickoff_utc": match.kickoff_utc,
        }
        rows.extend(
            (
                {
                    **common,
                    "club_id": str(match.home_club_id),
                    "venue": "HOME",
                    "opponent_source_club_key": str(match.away_club_id),
                    "expected_score": home_expected,
                    "neutral_expected_score": home_neutral,
                    "actual_score": score,
                    "opponent_band": classify_relative_opponent(home_neutral, lower=lower, upper=upper),
                },
                {
                    **common,
                    "club_id": str(match.away_club_id),
                    "venue": "AWAY",
                    "opponent_source_club_key": str(match.home_club_id),
                    "expected_score": 1.0 - home_expected,
                    "neutral_expected_score": 1.0 - home_neutral,
                    "actual_score": 1.0 - score,
                    "opponent_band": classify_relative_opponent(1.0 - home_neutral, lower=lower, upper=upper),
                },
            )
        )
    return pd.DataFrame(rows).sort_values(["kickoff_utc", "match_id", "club_id"], kind="stable").reset_index(drop=True)


def estimate_relative_profile(
    club_id: str,
    history: pd.DataFrame,
    *,
    config: RelativeOpponentProfileConfig,
    elo_scale: float,
) -> QuintileProfile:
    config.validate()
    if history.empty:
        history = pd.DataFrame(
            columns=["opponent_band", "expected_score", "actual_score", "weight"]
        )
    selected = history.loc[:, ["opponent_band", "expected_score", "actual_score", "weight"]].rename(columns={"opponent_band": "opponent_quintile"})
    return estimate_quintile_profile(
        club_id,
        selected,
        elo_scale=elo_scale,
        shrinkage_matches=config.shrinkage_matches,
        effect_cap=config.effect_cap,
        band_count=3,
    )


def profile_effect_for_band(profile: QuintileProfile, band: int) -> float:
    if band not in RELATIVE_BANDS:
        raise ValueError("relative opponent band must be 1, 2 or 3")
    return float(profile.effect_for(band))


def profile_record(profile: QuintileProfile) -> dict[str, object]:
    result = profile_to_record(profile)
    for band, label in RELATIVE_BAND_LABELS.items():
        index = band - 1
        result[f"effect_{label.lower()}"] = profile.effects[index]
        result[f"effective_matches_{label.lower()}"] = profile.effective_matches[index]
        result[f"reliability_{label.lower()}"] = profile.reliabilities[index]
    return result


def contextual_expected_score(
    base_expected_home_score: float,
    home_profile: QuintileProfile,
    away_profile: QuintileProfile,
    *,
    home_band: int,
    away_band: int,
    elo_scale: float,
    context_cap: float,
) -> tuple[float, float, float, float]:
    result = contextual_matchup_expectation(
        base_expected_home_score,
        profile_effect_for_band(home_profile, home_band),
        profile_effect_for_band(away_profile, away_band),
        elo_scale=elo_scale,
        context_cap=context_cap,
    )
    return result.expected_home_score, result.home_effect, result.away_effect, result.applied_offset


def weighted_history(
    rows: Iterable[dict[str, object]] | pd.DataFrame,
    *,
    target_season: str,
    season_index: dict[str, int],
    config: RelativeOpponentProfileConfig,
) -> pd.DataFrame:
    history = pd.DataFrame(rows).copy()
    if history.empty:
        history["weight"] = pd.Series(dtype=float)
        return history
    target = season_index[target_season]
    values = history["ao_season"].astype(str).map(season_index)
    if values.isna().any():
        invalid = history.loc[values.isna(), ["ao_season", "kickoff_utc"]]
        raise ValueError(
            "relative profile history contains an unknown season: "
            f"{invalid.head(3).to_dict(orient='records')}"
        )
    # The caller supplies only earlier kickoff timestamps.  A small number of
    # COVID-delayed domestic matches carry the following season label despite
    # preceding an August European tie, so treat their season distance as zero.
    distance = np.maximum(target - values.astype(int), 0)
    selected = history.loc[distance.le(config.lookback_seasons)].copy()
    selected_distance = distance.loc[selected.index].astype(int)
    source_weight = np.where(selected["source"].eq("DOMESTIC"), config.domestic_weight, 1.0)
    selected["weight"] = source_weight * np.power(config.season_decay, np.maximum(selected_distance - 1, 0))
    return selected.loc[selected["weight"].gt(0.0)].copy()


_PERSPECTIVE_COLUMNS = (
    "match_id", "source", "country_code", "sportsdb_league_id", "provider_season", "ao_season",
    "competition", "kickoff_utc", "club_id", "venue", "opponent_source_club_key", "expected_score",
    "neutral_expected_score", "actual_score", "opponent_band",
)


def _positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")


def _non_negative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")


def _probability(name: str, value: float, *, allow_zero: bool = False) -> None:
    if not math.isfinite(value) or value > 1.0 or value < 0.0 or (not allow_zero and value == 0.0):
        raise ValueError(f"{name} must be within (0,1]")
