from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from ao_elo.scoreline import (
    DEFAULT_RHO_GRID,
    ScorelineModelConfig,
    exact_score_probability,
    scoreline_matrix,
    scoreline_to_1x2,
)


DEFAULT_GOAL_LEVEL = math.log(1.35)
DEFAULT_HOME_ADVANTAGE = math.log(1.12)
PROBABILITY_EPSILON = 1e-15
DOMESTIC_STATE_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class DomesticPoissonConfig:
    team_learning_rate: float
    season_carry: float
    shrinkage_matches: float
    venue_context: bool
    lambda_min: float = 0.20
    lambda_max: float = 4.50
    team_parameter_cap: float = 1.00
    venue_parameter_cap: float = 0.75
    league_home_min: float = -0.30
    league_home_max: float = 0.50
    league_learning_ratio: float = 0.10
    model_version: str = "ao-domestic-dynamic-poisson-v1-shadow"

    def validate(self) -> None:
        _positive("team_learning_rate", self.team_learning_rate)
        if not 0.0 <= float(self.season_carry) <= 1.0:
            raise ValueError("season_carry must be in [0,1]")
        _positive("shrinkage_matches", self.shrinkage_matches)
        if not isinstance(self.venue_context, (bool, np.bool_)):
            raise ValueError("venue_context must be boolean")
        _positive("lambda_min", self.lambda_min)
        _positive("lambda_max", self.lambda_max)
        if self.lambda_max <= self.lambda_min:
            raise ValueError("lambda_max must exceed lambda_min")
        _positive("team_parameter_cap", self.team_parameter_cap)
        _positive("venue_parameter_cap", self.venue_parameter_cap)
        _finite("league_home_min", self.league_home_min)
        _finite("league_home_max", self.league_home_max)
        if self.league_home_max <= self.league_home_min:
            raise ValueError("league_home_max must exceed league_home_min")
        if not 0.0 < float(self.league_learning_ratio) <= 1.0:
            raise ValueError("league_learning_ratio must be in (0,1]")
        if not isinstance(self.model_version, str) or not self.model_version.strip():
            raise ValueError("model_version must be a non-empty string")

    @property
    def key(self) -> str:
        return (
            f"lr{self.team_learning_rate:g}_carry{self.season_carry:g}_"
            f"shrink{self.shrinkage_matches:g}_venue{int(self.venue_context)}"
        )


@dataclass
class DomesticTeamState:
    attack: float = 0.0
    defence: float = 0.0
    home_strength: float = 0.0
    away_strength: float = 0.0
    effective_matches: float = 0.0


@dataclass
class DomesticLeagueState:
    goal_level: float = DEFAULT_GOAL_LEVEL
    home_advantage: float = DEFAULT_HOME_ADVANTAGE
    current_season: str | None = None
    teams: dict[str, DomesticTeamState] = field(default_factory=dict)


@dataclass(frozen=True)
class DomesticTeamSnapshot:
    attack_raw_z: float
    defence_raw_z: float
    home_venue_raw_z: float
    away_venue_raw_z: float
    attack: float
    defence: float
    home_venue: float
    away_venue: float
    reliability: float
    effective_matches: float
    covered: bool


@dataclass(frozen=True)
class EuropeanPoissonTransferConfig:
    mu: float
    elo_slope: float
    attack_coefficient: float
    defence_coefficient: float
    venue_coefficient: float
    l2_strength: float
    rho: float = 0.0
    use_reliability: bool = True
    use_venue: bool = True
    model_version: str = "ao-domestic-poisson-transfer-v1-shadow"

    def validate(self) -> None:
        if not math.log(0.30) <= float(self.mu) <= math.log(4.00):
            raise ValueError("mu must be within the configured goal-level bounds")
        if not 0.05 <= float(self.elo_slope) <= 3.00:
            raise ValueError("elo_slope must be in [0.05,3.00]")
        for name, value in (
            ("attack_coefficient", self.attack_coefficient),
            ("defence_coefficient", self.defence_coefficient),
            ("venue_coefficient", self.venue_coefficient),
        ):
            if not 0.0 <= float(value) <= 1.50:
                raise ValueError(f"{name} must be in [0,1.50]")
        _non_negative("l2_strength", self.l2_strength)
        if not -0.15 <= float(self.rho) <= 0.15:
            raise ValueError("rho must be in [-0.15,0.15]")
        if not isinstance(self.use_reliability, (bool, np.bool_)):
            raise ValueError("use_reliability must be boolean")
        if not isinstance(self.use_venue, (bool, np.bool_)):
            raise ValueError("use_venue must be boolean")

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.__dict__, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class DynamicDomesticPoisson:
    """Causal league-relative attack/defence state updated in kickoff batches."""

    def __init__(self, config: DomesticPoissonConfig):
        config.validate()
        self.config = config
        self.leagues: dict[str, DomesticLeagueState] = {}

    def ensure_season(self, league_id: str, season: str) -> None:
        league = self.leagues.setdefault(str(league_id), DomesticLeagueState())
        season = str(season)
        if league.current_season is None:
            league.current_season = season
            return
        current = _season_start(league.current_season)
        target = _season_start(season)
        if target < current:
            raise ValueError(
                f"Domestic chronology regressed for league={league_id}: "
                f"{league.current_season} -> {season}"
            )
        while current < target:
            self._carry_league(league)
            current += 1
        league.current_season = season

    def predict_match(
        self,
        league_id: str,
        season: str,
        home_team_id: str,
        away_team_id: str,
    ) -> tuple[float, float]:
        self.ensure_season(league_id, season)
        league = self.leagues[str(league_id)]
        home = league.teams.setdefault(str(home_team_id), DomesticTeamState())
        away = league.teams.setdefault(str(away_team_id), DomesticTeamState())
        venue_edge = (
            home.home_strength - away.away_strength
            if self.config.venue_context
            else 0.0
        )
        home_log = (
            league.goal_level
            + league.home_advantage
            + home.attack
            - away.defence
            + 0.5 * venue_edge
        )
        away_log = (
            league.goal_level
            + away.attack
            - home.defence
            - 0.5 * venue_edge
        )
        return self._bounded_rate(home_log), self._bounded_rate(away_log)

    def update_batch(self, matches: pd.DataFrame) -> pd.DataFrame:
        data = _validate_domestic_matches(matches)
        if data["kickoff_utc"].nunique() != 1:
            raise ValueError("update_batch requires one exact kickoff_utc snapshot")
        return pd.DataFrame(self._update_records(_domestic_records(data)))

    def _update_records(
        self,
        records: Sequence[tuple[str, str, str, pd.Timestamp, str, str, int, int]],
    ) -> list[dict[str, object]]:
        if not records:
            raise ValueError("Domestic update records cannot be empty")
        if len({record[3] for record in records}) != 1:
            raise ValueError("Domestic update records must share one kickoff timestamp")
        predictions: list[dict[str, object]] = []
        gradients: dict[tuple[str, str, str], float] = defaultdict(float)
        league_gradients: dict[tuple[str, str], float] = defaultdict(float)
        appearances: dict[tuple[str, str], int] = defaultdict(int)

        for event_id, league_id, season, _, home_id, away_id, home_goals, away_goals in records:
            lambda_home, lambda_away = self.predict_match(
                league_id, season, home_id, away_id
            )
            home_residual = float(home_goals) - lambda_home
            away_residual = float(away_goals) - lambda_away
            predictions.append(
                {
                    "source_event_id": event_id,
                    "ao_season": season,
                    "sportsdb_league_id": league_id,
                    "lambda_home": lambda_home,
                    "lambda_away": lambda_away,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "goal_nll": _independent_poisson_nll(
                        home_goals, away_goals, lambda_home, lambda_away
                    ),
                }
            )
            gradients[(league_id, home_id, "attack")] += home_residual
            gradients[(league_id, away_id, "defence")] -= home_residual
            gradients[(league_id, away_id, "attack")] += away_residual
            gradients[(league_id, home_id, "defence")] -= away_residual
            if self.config.venue_context:
                venue_gradient = 0.5 * (home_residual - away_residual)
                gradients[(league_id, home_id, "home_strength")] += venue_gradient
                gradients[(league_id, away_id, "away_strength")] -= venue_gradient
            league_gradients[(league_id, "goal_level")] += home_residual + away_residual
            league_gradients[(league_id, "home_advantage")] += home_residual
            appearances[(league_id, home_id)] += 1
            appearances[(league_id, away_id)] += 1

        learning_rate = float(self.config.team_learning_rate)
        for (league_id, team_id, parameter), gradient in gradients.items():
            team = self.leagues[league_id].teams[team_id]
            value = getattr(team, parameter) + learning_rate * gradient
            cap = (
                self.config.venue_parameter_cap
                if parameter in {"home_strength", "away_strength"}
                else self.config.team_parameter_cap
            )
            setattr(team, parameter, float(np.clip(value, -cap, cap)))
        for (league_id, team_id), count in appearances.items():
            self.leagues[league_id].teams[team_id].effective_matches += float(count)
        league_rate = learning_rate * float(self.config.league_learning_ratio)
        for (league_id, parameter), gradient in league_gradients.items():
            league = self.leagues[league_id]
            value = getattr(league, parameter) + league_rate * gradient
            if parameter == "goal_level":
                value = float(np.clip(value, math.log(self.config.lambda_min), math.log(self.config.lambda_max)))
            else:
                value = float(np.clip(value, self.config.league_home_min, self.config.league_home_max))
            setattr(league, parameter, value)
        for league_id in {record[1] for record in records}:
            self._center_league(self.leagues[league_id])
        return predictions

    def snapshot(
        self,
        league_id: str,
        season: str,
        team_id: str,
    ) -> DomesticTeamSnapshot:
        self.ensure_season(league_id, season)
        league = self.leagues[str(league_id)]
        team = league.teams.get(str(team_id))
        if team is None or team.effective_matches <= 0.0:
            return empty_domestic_snapshot()
        reliability = team.effective_matches / (
            team.effective_matches + float(self.config.shrinkage_matches)
        )
        attack_z = self._z_score(league, "attack", team.attack)
        defence_z = self._z_score(league, "defence", team.defence)
        home_z = self._z_score(league, "home_strength", team.home_strength)
        away_z = self._z_score(league, "away_strength", team.away_strength)
        return DomesticTeamSnapshot(
            attack_raw_z=attack_z,
            defence_raw_z=defence_z,
            home_venue_raw_z=home_z,
            away_venue_raw_z=away_z,
            attack=attack_z * reliability,
            defence=defence_z * reliability,
            home_venue=home_z * reliability,
            away_venue=away_z * reliability,
            reliability=float(np.clip(reliability, 0.0, 1.0)),
            effective_matches=float(team.effective_matches),
            covered=True,
        )

    def validate_state(self) -> None:
        for league_id, league in self.leagues.items():
            if not self.config.league_home_min <= league.home_advantage <= self.config.league_home_max:
                raise ValueError(f"league home effect cap violated: {league_id}")
            if not math.isfinite(league.goal_level):
                raise ValueError(f"league goal level is non-finite: {league_id}")
            for parameter, cap in (
                ("attack", self.config.team_parameter_cap),
                ("defence", self.config.team_parameter_cap),
                ("home_strength", self.config.venue_parameter_cap),
                ("away_strength", self.config.venue_parameter_cap),
            ):
                values = np.array(
                    [getattr(team, parameter) for team in league.teams.values()], dtype=float
                )
                if len(values) and (
                    not np.isfinite(values).all()
                    or np.max(np.abs(values)) > float(cap) + 1e-12
                    or abs(float(values.mean())) > 1e-12
                ):
                    raise ValueError(
                        f"league-relative {parameter} invariant failed: {league_id}"
                    )

    def to_payload(self) -> dict[str, object]:
        """Serialize the causal state without provider responses or match outcomes."""
        self.validate_state()
        return {
            "schema_version": DOMESTIC_STATE_SCHEMA_VERSION,
            "config": asdict(self.config),
            "leagues": {
                league_id: {
                    "goal_level": league.goal_level,
                    "home_advantage": league.home_advantage,
                    "current_season": league.current_season,
                    "teams": {
                        team_id: asdict(team)
                        for team_id, team in sorted(league.teams.items())
                    },
                }
                for league_id, league in sorted(self.leagues.items())
            },
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "DynamicDomesticPoisson":
        """Restore a trusted, versioned state checkpoint and re-run invariants."""
        if payload.get("schema_version") != DOMESTIC_STATE_SCHEMA_VERSION:
            raise ValueError("Unsupported domestic Poisson state schema")
        config_payload = payload.get("config")
        leagues_payload = payload.get("leagues")
        if not isinstance(config_payload, Mapping) or not isinstance(
            leagues_payload, Mapping
        ):
            raise ValueError("Domestic Poisson state payload is incomplete")
        try:
            config = DomesticPoissonConfig(**dict(config_payload))
        except TypeError as exc:
            raise ValueError("Domestic Poisson state config is invalid") from exc
        engine = cls(config)
        for league_id, raw_league in sorted(leagues_payload.items()):
            if not isinstance(raw_league, Mapping):
                raise ValueError("Domestic Poisson league state must be an object")
            season = raw_league.get("current_season")
            if season is not None and (not isinstance(season, str) or not season):
                raise ValueError("Domestic Poisson current_season is invalid")
            teams_payload = raw_league.get("teams")
            if not isinstance(teams_payload, Mapping):
                raise ValueError("Domestic Poisson team state must be an object")
            league = DomesticLeagueState(
                goal_level=_finite_value(
                    "domestic state goal_level", raw_league.get("goal_level")
                ),
                home_advantage=_finite_value(
                    "domestic state home_advantage",
                    raw_league.get("home_advantage"),
                ),
                current_season=season,
            )
            for team_id, raw_team in sorted(teams_payload.items()):
                if not isinstance(raw_team, Mapping):
                    raise ValueError("Domestic Poisson team payload must be an object")
                team = DomesticTeamState(
                    attack=_finite_value("domestic state attack", raw_team.get("attack")),
                    defence=_finite_value(
                        "domestic state defence", raw_team.get("defence")
                    ),
                    home_strength=_finite_value(
                        "domestic state home_strength",
                        raw_team.get("home_strength"),
                    ),
                    away_strength=_finite_value(
                        "domestic state away_strength",
                        raw_team.get("away_strength"),
                    ),
                    effective_matches=_finite_value(
                        "domestic state effective_matches",
                        raw_team.get("effective_matches"),
                    ),
                )
                if team.effective_matches < 0.0:
                    raise ValueError("Domestic effective_matches cannot be negative")
                league.teams[str(team_id)] = team
            engine.leagues[str(league_id)] = league
        engine.validate_state()
        return engine

    def _carry_league(self, league: DomesticLeagueState) -> None:
        carry = float(self.config.season_carry)
        for team in league.teams.values():
            team.attack *= carry
            team.defence *= carry
            team.home_strength *= carry
            team.away_strength *= carry
            team.effective_matches *= carry
        league.goal_level = DEFAULT_GOAL_LEVEL + carry * (
            league.goal_level - DEFAULT_GOAL_LEVEL
        )
        league.home_advantage = DEFAULT_HOME_ADVANTAGE + carry * (
            league.home_advantage - DEFAULT_HOME_ADVANTAGE
        )
        self._center_league(league)

    def _center_league(self, league: DomesticLeagueState) -> None:
        if not league.teams:
            return
        for parameter in ("attack", "defence", "home_strength", "away_strength"):
            teams = list(league.teams.values())
            values = np.array([getattr(team, parameter) for team in teams], dtype=float)
            values -= float(values.mean())
            cap = (
                float(self.config.venue_parameter_cap)
                if parameter in {"home_strength", "away_strength"}
                else float(self.config.team_parameter_cap)
            )
            maximum = float(np.max(np.abs(values)))
            if maximum > cap:
                values *= cap / maximum
            for team, value in zip(teams, values, strict=True):
                setattr(team, parameter, float(value))

    @staticmethod
    def _z_score(league: DomesticLeagueState, parameter: str, value: float) -> float:
        values = np.array([getattr(team, parameter) for team in league.teams.values()], dtype=float)
        standard_deviation = float(values.std()) if len(values) > 1 else 0.0
        if standard_deviation <= 1e-9:
            return 0.0
        return float(np.clip(float(value) / standard_deviation, -3.0, 3.0))

    def _bounded_rate(self, log_rate: float) -> float:
        return float(
            np.clip(math.exp(float(log_rate)), self.config.lambda_min, self.config.lambda_max)
        )


def domestic_candidate_grid() -> tuple[DomesticPoissonConfig, ...]:
    return tuple(
        DomesticPoissonConfig(rate, carry, shrinkage, venue)
        for rate in (0.02, 0.05, 0.08)
        for carry in (0.50, 0.75, 0.90)
        for shrinkage in (10.0, 20.0, 40.0)
        for venue in (False, True)
    )


def evaluate_domestic_candidates(
    matches: pd.DataFrame,
    candidates: Sequence[DomesticPoissonConfig] | None = None,
) -> pd.DataFrame:
    data = _validate_domestic_matches(matches)
    candidates = tuple(candidates or domestic_candidate_grid())
    if len(candidates) != 54 or len({candidate.key for candidate in candidates}) != 54:
        raise ValueError("Expected 54 unique domestic Poisson candidates")
    rows: list[dict[str, object]] = []
    records = _domestic_records(data)
    grouped = tuple(
        tuple(group)
        for _, group in itertools.groupby(records, key=lambda record: record[3])
    )
    for config in candidates:
        engine = DynamicDomesticPoisson(config)
        predictions: list[dict[str, object]] = []
        for batch in grouped:
            predictions.extend(engine._update_records(batch))
        result = pd.DataFrame(predictions)
        engine.validate_state()
        for season, season_rows in result.groupby("ao_season", sort=False):
            rows.append(
                {
                    "candidate_key": config.key,
                    "team_learning_rate": config.team_learning_rate,
                    "season_carry": config.season_carry,
                    "shrinkage_matches": config.shrinkage_matches,
                    "venue_context": config.venue_context,
                    "season": str(season),
                    "matches": int(len(season_rows)),
                    "goal_nll": float(season_rows["goal_nll"].mean()),
                    "home_goal_bias": float(
                        (season_rows["lambda_home"] - season_rows["home_goals"]).mean()
                    ),
                    "away_goal_bias": float(
                        (season_rows["lambda_away"] - season_rows["away_goals"]).mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_domestic_poisson_feature_store(
    domestic_matches: pd.DataFrame,
    european_matches: pd.DataFrame,
    bridge: pd.DataFrame,
    config: DomesticPoissonConfig,
) -> pd.DataFrame:
    domestic = _validate_domestic_matches(domestic_matches)
    european = _validate_european_matches(european_matches)
    mapping = _validated_bridge(bridge, domestic)
    engine = DynamicDomesticPoisson(config)
    domestic_groups = [
        (kickoff, tuple(group))
        for kickoff, group in itertools.groupby(
            _domestic_records(domestic), key=lambda record: record[3]
        )
    ]
    domestic_index = 0
    rows: list[dict[str, object]] = []

    for kickoff, batch in european.groupby("kickoff_utc", sort=True):
        while (
            domestic_index < len(domestic_groups)
            and domestic_groups[domestic_index][0] < kickoff
        ):
            engine._update_records(domestic_groups[domestic_index][1])
            domestic_index += 1
        for match in batch.sort_values("match_id", kind="stable").itertuples(index=False):
            home = _mapped_snapshot(engine, mapping, str(match.home_club_id), str(match.season))
            away = _mapped_snapshot(engine, mapping, str(match.away_club_id), str(match.season))
            row: dict[str, object] = {
                "match_id": str(match.match_id),
                "domestic_poisson_config": config.key,
            }
            _add_snapshot(row, "home", home)
            _add_snapshot(row, "away", away)
            row["domestic_poisson_coverage"] = (
                "BOTH"
                if home.covered and away.covered
                else "ONE"
                if home.covered or away.covered
                else "NONE"
            )
            row["domestic_poisson_venue_edge"] = home.home_venue - away.away_venue
            row["domestic_poisson_venue_edge_raw"] = (
                home.home_venue_raw_z - away.away_venue_raw_z
            )
            rows.append(row)
    result = pd.DataFrame(rows)
    if len(result) != len(european) or result["match_id"].duplicated().any():
        raise ValueError("Domestic Poisson feature store must contain one row per European match")
    return result


def replay_domestic_poisson_state(
    matches: pd.DataFrame,
    config: DomesticPoissonConfig,
) -> DynamicDomesticPoisson:
    """Build the deterministic end-of-data state used to bootstrap production."""
    data = _validate_domestic_matches(matches)
    engine = DynamicDomesticPoisson(config)
    records = _domestic_records(data)
    for _, group in itertools.groupby(records, key=lambda record: record[3]):
        engine._update_records(tuple(group))
    engine.validate_state()
    return engine


def build_domestic_club_mapping(
    domestic_matches: pd.DataFrame,
    bridge: pd.DataFrame,
) -> dict[str, tuple[str, str]]:
    """Return the audited AO club -> provider league/team identity mapping."""
    domestic = _validate_domestic_matches(domestic_matches)
    return _validated_bridge(bridge, domestic)


def fit_european_poisson_transfer(
    matches: pd.DataFrame,
    *,
    l2_strength: float,
    use_reliability: bool = True,
    use_venue: bool = True,
) -> EuropeanPoissonTransferConfig:
    data = _validate_transfer_frame(matches)
    _non_negative("l2_strength", l2_strength)
    feature_columns = _transfer_columns(use_reliability)
    z_ao = _ao_logit(data)
    attack_home = data[feature_columns["home_attack"]].to_numpy(float)
    attack_away = data[feature_columns["away_attack"]].to_numpy(float)
    defence_home = data[feature_columns["home_defence"]].to_numpy(float)
    defence_away = data[feature_columns["away_defence"]].to_numpy(float)
    venue = data[
        "domestic_poisson_venue_edge"
        if use_reliability
        else "domestic_poisson_venue_edge_raw"
    ].to_numpy(float)
    home_goals = data["home_goals"].to_numpy(float)
    away_goals = data["away_goals"].to_numpy(float)
    initial_mu = math.log(float((home_goals.sum() + away_goals.sum()) / (2.0 * len(data))))

    def objective(parameters: np.ndarray) -> float:
        mu, beta, attack_coefficient, defence_coefficient = parameters[:4]
        venue_coefficient = float(parameters[4]) if use_venue else 0.0
        home_log = (
            mu
            + 0.5 * beta * z_ao
            + attack_coefficient * attack_home
            - defence_coefficient * defence_away
            + 0.5 * venue_coefficient * venue
        )
        away_log = (
            mu
            - 0.5 * beta * z_ao
            + attack_coefficient * attack_away
            - defence_coefficient * defence_home
            - 0.5 * venue_coefficient * venue
        )
        home_rate = np.exp(np.clip(home_log, math.log(0.20), math.log(4.50)))
        away_rate = np.exp(np.clip(away_log, math.log(0.20), math.log(4.50)))
        nll = (
            home_rate - home_goals * np.log(home_rate) + gammaln(home_goals + 1.0)
            + away_rate - away_goals * np.log(away_rate) + gammaln(away_goals + 1.0)
        ).sum()
        penalty = float(l2_strength) * float(np.square(parameters[1:]).sum())
        return float(nll + penalty)

    initial = [initial_mu, 0.75, 0.05, 0.05]
    bounds = [
        (math.log(0.30), math.log(4.00)),
        (0.05, 3.00),
        (0.0, 1.50),
        (0.0, 1.50),
    ]
    if use_venue:
        initial.append(0.02)
        bounds.append((0.0, 1.50))
    fitted = minimize(
        objective,
        np.array(initial, dtype=float),
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-11, "gtol": 1e-7, "maxiter": 1000},
    )
    if not fitted.success or not np.isfinite(fitted.fun):
        raise ValueError(f"European Poisson transfer optimizer failed: {fitted.message}")
    values = fitted.x
    config = EuropeanPoissonTransferConfig(
        mu=float(values[0]),
        elo_slope=float(values[1]),
        attack_coefficient=float(values[2]),
        defence_coefficient=float(values[3]),
        venue_coefficient=float(values[4]) if use_venue else 0.0,
        l2_strength=float(l2_strength),
        use_reliability=use_reliability,
        use_venue=use_venue,
    )
    config.validate()
    return config


def predict_european_poisson_transfer(
    matches: pd.DataFrame,
    config: EuropeanPoissonTransferConfig,
    *,
    fallback_to_ao_without_history: bool = True,
) -> pd.DataFrame:
    config.validate()
    data = _validate_transfer_prediction_frame(matches)
    columns = _transfer_columns(config.use_reliability)
    z_ao = _ao_logit(data)
    venue = data[
        "domestic_poisson_venue_edge"
        if config.use_reliability
        else "domestic_poisson_venue_edge_raw"
    ].to_numpy(float)
    home_log = (
        config.mu
        + 0.5 * config.elo_slope * z_ao
        + config.attack_coefficient * data[columns["home_attack"]].to_numpy(float)
        - config.defence_coefficient * data[columns["away_defence"]].to_numpy(float)
        + 0.5 * config.venue_coefficient * venue
    )
    away_log = (
        config.mu
        - 0.5 * config.elo_slope * z_ao
        + config.attack_coefficient * data[columns["away_attack"]].to_numpy(float)
        - config.defence_coefficient * data[columns["home_defence"]].to_numpy(float)
        - 0.5 * config.venue_coefficient * venue
    )
    lambda_home = np.exp(np.clip(home_log, math.log(0.20), math.log(4.50)))
    lambda_away = np.exp(np.clip(away_log, math.log(0.20), math.log(4.50)))
    rows: list[dict[str, object]] = []
    matrix_config = ScorelineModelConfig(mu=0.0, elo_slope=1.0, rho=config.rho)
    for index, row in enumerate(data.itertuples(index=False)):
        matrix, covered_mass = scoreline_matrix(
            float(lambda_home[index]), float(lambda_away[index]), matrix_config
        )
        home_probability, draw_probability, away_probability = scoreline_to_1x2(matrix)
        fallback = bool(
            fallback_to_ao_without_history
            and row.domestic_poisson_coverage == "NONE"
        )
        if fallback:
            home_probability = float(row.ao_home_probability)
            draw_probability = float(row.ao_draw_probability)
            away_probability = float(row.ao_away_probability)
        exact = (
            exact_score_probability(
                int(row.home_goals),
                int(row.away_goals),
                float(lambda_home[index]),
                float(lambda_away[index]),
                config.rho,
            )
            if hasattr(row, "home_goals") and hasattr(row, "away_goals")
            else math.nan
        )
        rows.append(
            {
                "match_id": str(row.match_id),
                "lambda_home": float(lambda_home[index]),
                "lambda_away": float(lambda_away[index]),
                "home_probability": home_probability,
                "draw_probability": draw_probability,
                "away_probability": away_probability,
                "exact_score_probability": exact,
                "expected_total_goals": float(lambda_home[index] + lambda_away[index]),
                "over_2_5_probability": float(
                    matrix[
                        np.add.outer(np.arange(matrix.shape[0]), np.arange(matrix.shape[1])) >= 3
                    ].sum()
                ),
                "btts_probability": float(matrix[1:, 1:].sum()),
                "covered_probability_mass": covered_mass,
                "ao_fallback": fallback,
            }
        )
    result = pd.DataFrame(rows)
    probabilities = result[
        ["home_probability", "draw_probability", "away_probability"]
    ].to_numpy(float)
    if not np.isfinite(probabilities).all() or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-10
    ):
        raise ValueError("European Poisson probabilities must be finite and normalized")
    return result


def select_dixon_coles_rho(
    validation_matches: pd.DataFrame,
    base_config: EuropeanPoissonTransferConfig,
    rho_grid: Sequence[float] = DEFAULT_RHO_GRID,
) -> tuple[float, pd.DataFrame]:
    rows = []
    for rho in rho_grid:
        try:
            candidate = EuropeanPoissonTransferConfig(
                **{**base_config.__dict__, "rho": float(rho)}
            )
            prediction = predict_european_poisson_transfer(
                validation_matches, candidate
            )
            metrics = _one_x_two_metrics(validation_matches, prediction)
            rows.append({"rho": float(rho), "valid": True, **metrics})
        except ValueError:
            rows.append(
                {
                    "rho": float(rho),
                    "valid": False,
                    "brier_1x2": np.nan,
                    "log_loss_1x2": np.nan,
                    "combined_loss": np.inf,
                }
            )
    surface = pd.DataFrame(rows)
    valid = surface[surface["valid"]].copy()
    if valid.empty:
        raise ValueError("No valid Dixon-Coles rho candidate")
    valid["absolute_rho"] = valid["rho"].abs()
    selected = valid.sort_values(
        ["combined_loss", "absolute_rho", "rho"], kind="stable"
    ).iloc[0]
    return float(selected["rho"]), surface


def empty_domestic_snapshot() -> DomesticTeamSnapshot:
    return DomesticTeamSnapshot(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False)


def _mapped_snapshot(
    engine: DynamicDomesticPoisson,
    mapping: Mapping[str, tuple[str, str]],
    club_id: str,
    season: str,
) -> DomesticTeamSnapshot:
    identity = mapping.get(club_id)
    if identity is None:
        return empty_domestic_snapshot()
    league = engine.leagues.get(identity[0])
    if league is None or league.current_season is None:
        return empty_domestic_snapshot()
    return engine.snapshot(identity[0], league.current_season, identity[1])


def _add_snapshot(
    row: dict[str, object], side: str, snapshot: DomesticTeamSnapshot
) -> None:
    for name in (
        "attack_raw_z",
        "defence_raw_z",
        "home_venue_raw_z",
        "away_venue_raw_z",
        "attack",
        "defence",
        "home_venue",
        "away_venue",
        "reliability",
        "effective_matches",
        "covered",
    ):
        row[f"{side}_domestic_poisson_{name}"] = getattr(snapshot, name)


def _transfer_columns(use_reliability: bool) -> dict[str, str]:
    suffix = "" if use_reliability else "_raw_z"
    return {
        "home_attack": f"home_domestic_poisson_attack{suffix}",
        "away_attack": f"away_domestic_poisson_attack{suffix}",
        "home_defence": f"home_domestic_poisson_defence{suffix}",
        "away_defence": f"away_domestic_poisson_defence{suffix}",
    }


def _ao_logit(frame: pd.DataFrame) -> np.ndarray:
    expected = np.clip(frame["expected_home_score"].to_numpy(float), 1e-8, 1.0 - 1e-8)
    return np.log(expected / (1.0 - expected))


def _one_x_two_metrics(
    matches: pd.DataFrame, predictions: pd.DataFrame
) -> dict[str, float]:
    probabilities = predictions[
        ["home_probability", "draw_probability", "away_probability"]
    ].to_numpy(float)
    outcomes = matches["actual_class"].to_numpy(int)
    target = np.eye(3)[outcomes]
    observed = probabilities[np.arange(len(matches)), outcomes]
    brier = float(np.square(probabilities - target).sum(axis=1).mean())
    log_loss = float(-np.log(np.clip(observed, PROBABILITY_EPSILON, 1.0)).mean())
    return {
        "brier_1x2": brier,
        "log_loss_1x2": log_loss,
        "combined_loss": 0.5 * (brier + log_loss),
    }


def _validated_bridge(
    bridge: pd.DataFrame, domestic: pd.DataFrame
) -> dict[str, tuple[str, str]]:
    required = {"source_team_id", "ao_club_id", "identity_ambiguous"}
    missing = sorted(required - set(bridge.columns))
    if missing:
        raise ValueError(f"domestic bridge missing columns: {missing}")
    accepted = bridge[
        bridge["ao_club_id"].notna() & ~bridge["identity_ambiguous"].astype(bool)
    ].copy()
    if accepted["ao_club_id"].duplicated().any():
        raise ValueError("Each AO club must map to at most one domestic source team")
    source_leagues = pd.concat(
        [
            domestic[["home_source_team_id", "sportsdb_league_id"]].rename(
                columns={"home_source_team_id": "source_team_id"}
            ),
            domestic[["away_source_team_id", "sportsdb_league_id"]].rename(
                columns={"away_source_team_id": "source_team_id"}
            ),
        ],
        ignore_index=True,
    )
    league_map = (
        source_leagues.astype(str)
        .groupby("source_team_id")["sportsdb_league_id"]
        .agg(lambda values: values.value_counts().sort_index().idxmax())
        .to_dict()
    )
    result: dict[str, tuple[str, str]] = {}
    for row in accepted.itertuples(index=False):
        source_id = str(row.source_team_id)
        if source_id in league_map:
            result[str(row.ao_club_id)] = (str(league_map[source_id]), source_id)
    return result


def _validate_domestic_matches(matches: pd.DataFrame) -> pd.DataFrame:
    required = {
        "source_event_id",
        "sportsdb_league_id",
        "ao_season",
        "kickoff_utc",
        "home_source_team_id",
        "away_source_team_id",
        "home_goals",
        "away_goals",
    }
    missing = sorted(required - set(matches.columns))
    if missing:
        raise ValueError(f"domestic_matches missing columns: {missing}")
    result = matches.copy()
    result["kickoff_utc"] = pd.to_datetime(result["kickoff_utc"], utc=True, errors="coerce")
    if result["kickoff_utc"].isna().any():
        raise ValueError("domestic_matches.kickoff_utc contains invalid timestamps")
    if result["source_event_id"].isna().any() or result["source_event_id"].duplicated().any():
        raise ValueError("domestic_matches.source_event_id must be non-null and unique")
    if result[["sportsdb_league_id", "ao_season", "home_source_team_id", "away_source_team_id"]].isna().any().any():
        raise ValueError("domestic match identity fields cannot be missing")
    if result["home_source_team_id"].astype(str).eq(result["away_source_team_id"].astype(str)).any():
        raise ValueError("Domestic home and away teams must differ")
    for column in ("home_goals", "away_goals"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if result[column].isna().any() or (result[column] < 0).any():
            raise ValueError(f"domestic_matches.{column} must be non-negative numeric")
        if not np.allclose(result[column], np.round(result[column])):
            raise ValueError(f"domestic_matches.{column} must contain integers")
        result[column] = result[column].astype(int)
    return result.sort_values(["kickoff_utc", "source_event_id"], kind="stable").reset_index(drop=True)


def _validate_european_matches(matches: pd.DataFrame) -> pd.DataFrame:
    required = {"match_id", "season", "kickoff_utc", "home_club_id", "away_club_id"}
    missing = sorted(required - set(matches.columns))
    if missing:
        raise ValueError(f"european_matches missing columns: {missing}")
    result = matches.copy()
    result["kickoff_utc"] = pd.to_datetime(result["kickoff_utc"], utc=True, errors="coerce")
    if result["kickoff_utc"].isna().any():
        raise ValueError("european_matches.kickoff_utc contains invalid timestamps")
    if result["match_id"].isna().any() or result["match_id"].duplicated().any():
        raise ValueError("european_matches.match_id must be non-null and unique")
    return result.sort_values(["kickoff_utc", "match_id"], kind="stable").reset_index(drop=True)


def _validate_transfer_frame(matches: pd.DataFrame) -> pd.DataFrame:
    result = _validate_transfer_prediction_frame(matches)
    required = {"home_goals", "away_goals", "actual_class"}
    missing = sorted(required - set(result.columns))
    if missing:
        raise ValueError(f"European transfer frame missing columns: {missing}")
    values = result[list(required - {"actual_class"})].apply(
        pd.to_numeric, errors="coerce"
    )
    if values.isna().any().any() or not np.isfinite(values.to_numpy(float)).all():
        raise ValueError("European transfer outcome fields must be finite")
    if not result["actual_class"].isin((0, 1, 2)).all():
        raise ValueError("actual_class must be 0, 1, or 2")
    return result


def _validate_transfer_prediction_frame(matches: pd.DataFrame) -> pd.DataFrame:
    required = {
        "match_id",
        "expected_home_score",
        "ao_home_probability",
        "ao_draw_probability",
        "ao_away_probability",
        "domestic_poisson_coverage",
        "domestic_poisson_venue_edge",
        "domestic_poisson_venue_edge_raw",
    }
    required.update(_transfer_columns(True).values())
    required.update(_transfer_columns(False).values())
    missing = sorted(required - set(matches.columns))
    if missing:
        raise ValueError(f"European transfer frame missing columns: {missing}")
    if matches.empty:
        raise ValueError("European transfer frame cannot be empty")
    values = matches[list(required - {"match_id", "domestic_poisson_coverage"})].apply(
        pd.to_numeric, errors="coerce"
    )
    if values.isna().any().any() or not np.isfinite(values.to_numpy(float)).all():
        raise ValueError("European transfer numeric fields must be finite")
    result = matches.reset_index(drop=True).copy()
    if result["match_id"].isna().any() or result["match_id"].duplicated().any():
        raise ValueError("European transfer match_id must be non-null and unique")
    if not result["domestic_poisson_coverage"].isin(("NONE", "ONE", "BOTH")).all():
        raise ValueError("domestic_poisson_coverage must be NONE, ONE, or BOTH")
    probabilities = result[
        ["ao_home_probability", "ao_draw_probability", "ao_away_probability"]
    ].to_numpy(float)
    if (probabilities < 0.0).any() or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-10, rtol=0.0
    ):
        raise ValueError("AO probabilities must be non-negative and normalized")
    expected = result["expected_home_score"].to_numpy(float)
    if ((expected < 0.0) | (expected > 1.0)).any():
        raise ValueError("expected_home_score must be in [0,1]")
    return result


def _independent_poisson_nll(
    home_goals: int, away_goals: int, lambda_home: float, lambda_away: float
) -> float:
    return float(
        lambda_home
        - home_goals * math.log(lambda_home)
        + math.lgamma(home_goals + 1.0)
        + lambda_away
        - away_goals * math.log(lambda_away)
        + math.lgamma(away_goals + 1.0)
    )


def _domestic_records(
    matches: pd.DataFrame,
) -> list[tuple[str, str, str, pd.Timestamp, str, str, int, int]]:
    columns = [
        "source_event_id",
        "sportsdb_league_id",
        "ao_season",
        "kickoff_utc",
        "home_source_team_id",
        "away_source_team_id",
        "home_goals",
        "away_goals",
    ]
    return [
        (
            str(event_id),
            str(league_id),
            str(season),
            pd.Timestamp(kickoff),
            str(home_id),
            str(away_id),
            int(home_goals),
            int(away_goals),
        )
        for event_id, league_id, season, kickoff, home_id, away_id, home_goals, away_goals
        in matches[columns].itertuples(index=False, name=None)
    ]


def _season_start(season: str) -> int:
    value = str(season).strip()
    if "/" in value:
        value = value.split("/", 1)[0]
    elif "-" in value and len(value) >= 4:
        value = value.split("-", 1)[0]
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid season value: {season!r}") from exc


def _positive(name: str, value: float) -> None:
    _finite(name, value)
    if float(value) <= 0.0:
        raise ValueError(f"{name} must be positive")


def _non_negative(name: str, value: float) -> None:
    _finite(name, value)
    if float(value) < 0.0:
        raise ValueError(f"{name} must be non-negative")


def _finite(name: str, value: float) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


def _finite_value(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result
