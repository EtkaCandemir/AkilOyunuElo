from __future__ import annotations

"""Goal-expectation models that can read recent xG instead of recent goals.

The active scoreline layer derives both rates from the Elo difference alone:

```text
z          = ln(10) * (R_home - R_away + H) / scale
lambda_home = exp(mu + 0.5 * slope * z)
lambda_away = exp(mu - 0.5 * slope * z)
```

Two parameters, no team-specific form. This module adds one symmetric form
term per side so the question "does xG predict goals better than goals do"
can be answered against a like-for-like control:

```text
lambda_home = exp(mu + 0.5 * slope * z + attack * f_home - defence * a_away)
lambda_away = exp(mu - 0.5 * slope * z + attack * f_away - defence * a_home)
```

`f` is a team's recent scoring rate and `a` its recent conceding rate, both
centered so that a team with no history contributes zero and the model falls
back to the Elo-only form. Feeding the same structure with goals or with xG
isolates the signal rather than the extra parameters.

History is causal by construction: a match only ever sees rows that kicked off
strictly earlier, and a source that carries no xG for a match simply does not
update the xG history.
"""

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln


FormSource = Literal["none", "goals", "xg"]
DEFAULT_WINDOW = 5
DEFAULT_SHRINKAGE = 3.0
LAMBDA_MIN = 0.20
LAMBDA_MAX = 4.50


@dataclass(frozen=True)
class GoalExpectationConfig:
    """Fitted rates for one form source."""

    source: FormSource
    mu: float
    elo_slope: float
    attack: float = 0.0
    defence: float = 0.0
    window: int = DEFAULT_WINDOW
    shrinkage: float = DEFAULT_SHRINKAGE

    def validate(self) -> None:
        for name in ("mu", "elo_slope", "attack", "defence"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.source not in ("none", "goals", "xg"):
            raise ValueError(f"unknown form source: {self.source}")
        if self.window < 1:
            raise ValueError("window must be positive")
        if self.shrinkage < 0.0:
            raise ValueError("shrinkage must be non-negative")
        if self.source == "none" and (self.attack or self.defence):
            raise ValueError("the Elo-only arm cannot carry form coefficients")

    @property
    def key(self) -> str:
        self.validate()
        return f"{self.source}_w{self.window}_s{self.shrinkage:g}"


def elo_z(
    frame: pd.DataFrame, *, elo_scale: float, home_advantage: float
) -> np.ndarray:
    advantage = np.where(
        frame["is_neutral"].to_numpy(bool), 0.0, float(home_advantage)
    )
    difference = (
        frame["home_live_pre"].to_numpy(float)
        - frame["away_live_pre"].to_numpy(float)
        + advantage
    )
    return math.log(10.0) * difference / float(elo_scale)


def form_prior(frame: pd.DataFrame, source: FormSource) -> float:
    """Return the centering level `build_form_features` would derive itself.

    Exposed so a walk-forward caller can compute it from training rows only
    and hand it back in, keeping the one global constant out of the test
    season.
    """

    if source == "none":
        return 0.0
    if source == "goals":
        if frame.empty:
            raise ValueError("goal form requires at least one match")
        return float(
            (frame["home_goals"].sum() + frame["away_goals"].sum()) / (2.0 * len(frame))
        )
    if source != "xg":
        raise ValueError(f"unknown form source: {source}")
    eligible = frame.loc[frame["xg_analysis_eligible"].astype(bool)]
    if eligible.empty:
        raise ValueError("xG form requires at least one eligible match")
    return float(
        (eligible["xg_home"].sum() + eligible["xg_away"].sum()) / (2.0 * len(eligible))
    )


def build_form_features(
    frame: pd.DataFrame,
    source: FormSource,
    *,
    window: int = DEFAULT_WINDOW,
    shrinkage: float = DEFAULT_SHRINKAGE,
    prior: float | None = None,
) -> pd.DataFrame:
    """Return causal, centered scoring and conceding rates per side.

    Rows are processed in kickoff order and a match is scored from history
    that closes strictly before it. Shrinkage pulls a short history toward the
    competition mean so an early-season team is not judged on one result.

    `prior` is the centering level a team with no history is pulled toward.
    Left unset it is the mean of `frame`, which is convenient but reads rows
    the caller may not have been allowed to see yet. A walk-forward caller
    should pass the mean of its training rows instead.
    """

    if source not in ("none", "goals", "xg"):
        raise ValueError(f"unknown form source: {source}")
    result = frame.copy()
    for column in ("home_form_for", "home_form_against", "away_form_for", "away_form_against"):
        result[column] = 0.0
    if source == "none":
        return result

    ordered = result.sort_values(["kickoff_utc", "match_id"], kind="stable")
    scored: dict[str, list[tuple[float, float]]] = {}
    if prior is not None:
        prior = float(prior)
        if not math.isfinite(prior):
            raise ValueError("prior must be finite")
    elif source == "goals":
        prior = float(
            (ordered["home_goals"].sum() + ordered["away_goals"].sum())
            / (2.0 * len(ordered))
        )
    else:
        eligible = ordered.loc[ordered["xg_analysis_eligible"].astype(bool)]
        if eligible.empty:
            raise ValueError("xG form requires at least one eligible match")
        prior = float(
            (eligible["xg_home"].sum() + eligible["xg_away"].sum())
            / (2.0 * len(eligible))
        )

    def summarize(club: str) -> tuple[float, float]:
        history = scored.get(club, [])
        recent = history[-window:]
        count = len(recent)
        if count == 0:
            return 0.0, 0.0
        weight = count / (count + shrinkage)
        made = float(np.mean([value for value, _ in recent]))
        let = float(np.mean([value for _, value in recent]))
        return weight * (made - prior), weight * (let - prior)

    rows: dict[str, list[float]] = {
        "home_form_for": [], "home_form_against": [],
        "away_form_for": [], "away_form_against": [],
    }
    index: list[int] = []
    for position, match in zip(ordered.index, ordered.itertuples(index=False)):
        home, away = str(match.home_club_id), str(match.away_club_id)
        home_for, home_against = summarize(home)
        away_for, away_against = summarize(away)
        rows["home_form_for"].append(home_for)
        rows["home_form_against"].append(home_against)
        rows["away_form_for"].append(away_for)
        rows["away_form_against"].append(away_against)
        index.append(position)

        if source == "goals":
            made_home, made_away = float(match.home_goals), float(match.away_goals)
        elif bool(match.xg_analysis_eligible):
            made_home, made_away = float(match.xg_home), float(match.xg_away)
        else:
            continue
        scored.setdefault(home, []).append((made_home, made_away))
        scored.setdefault(away, []).append((made_away, made_home))

    for column, values in rows.items():
        result.loc[index, column] = values
    return result


def fit_goal_expectation(
    frame: pd.DataFrame,
    source: FormSource,
    *,
    elo_scale: float,
    home_advantage: float,
    window: int = DEFAULT_WINDOW,
    shrinkage: float = DEFAULT_SHRINKAGE,
) -> GoalExpectationConfig:
    """Fit the rates by independent Poisson negative log-likelihood."""

    data = build_form_features(frame, source, window=window, shrinkage=shrinkage)
    z = elo_z(data, elo_scale=elo_scale, home_advantage=home_advantage)
    home_goals = data["home_goals"].to_numpy(float)
    away_goals = data["away_goals"].to_numpy(float)
    hf = data["home_form_for"].to_numpy(float)
    ha = data["home_form_against"].to_numpy(float)
    af = data["away_form_for"].to_numpy(float)
    aa = data["away_form_against"].to_numpy(float)
    start = math.log(float((home_goals.sum() + away_goals.sum()) / (2.0 * len(data))))

    def negative_log_likelihood(parameters: np.ndarray) -> float:
        mu, slope, attack, defence = (float(value) for value in parameters)
        home_log = mu + 0.5 * slope * z + attack * hf - defence * aa
        away_log = mu - 0.5 * slope * z + attack * af - defence * ha
        home_rate = np.exp(np.clip(home_log, math.log(LAMBDA_MIN), math.log(LAMBDA_MAX)))
        away_rate = np.exp(np.clip(away_log, math.log(LAMBDA_MIN), math.log(LAMBDA_MAX)))
        return float(
            np.sum(home_rate - home_goals * np.log(home_rate) + gammaln(home_goals + 1.0))
            + np.sum(away_rate - away_goals * np.log(away_rate) + gammaln(away_goals + 1.0))
        )

    free = source != "none"
    bounds = [(-2.0, 2.0), (0.0, 3.0), (0.0, 2.0) if free else (0.0, 0.0),
              (0.0, 2.0) if free else (0.0, 0.0)]
    solution = minimize(
        negative_log_likelihood,
        np.array([start, 0.5, 0.2 if free else 0.0, 0.2 if free else 0.0]),
        method="L-BFGS-B",
        bounds=bounds,
    )
    if not solution.success and not np.isfinite(solution.fun):
        raise ValueError(f"goal expectation fit failed: {solution.message}")
    mu, slope, attack, defence = (float(value) for value in solution.x)
    config = GoalExpectationConfig(
        source=source, mu=mu, elo_slope=slope,
        attack=attack if free else 0.0, defence=defence if free else 0.0,
        window=window, shrinkage=shrinkage,
    )
    config.validate()
    return config


def predict_goal_expectations(
    frame: pd.DataFrame,
    config: GoalExpectationConfig,
    *,
    elo_scale: float,
    home_advantage: float,
) -> pd.DataFrame:
    """Return bounded home and away rates for already-built form features."""

    config.validate()
    missing = sorted(
        {"home_form_for", "home_form_against", "away_form_for", "away_form_against"}
        - set(frame.columns)
    )
    if missing:
        raise ValueError(f"form features missing: {missing}")
    z = elo_z(frame, elo_scale=elo_scale, home_advantage=home_advantage)
    home_log = (
        config.mu
        + 0.5 * config.elo_slope * z
        + config.attack * frame["home_form_for"].to_numpy(float)
        - config.defence * frame["away_form_against"].to_numpy(float)
    )
    away_log = (
        config.mu
        - 0.5 * config.elo_slope * z
        + config.attack * frame["away_form_for"].to_numpy(float)
        - config.defence * frame["home_form_against"].to_numpy(float)
    )
    bounds = (math.log(LAMBDA_MIN), math.log(LAMBDA_MAX))
    result = frame[["match_id"]].copy()
    result["lambda_home"] = np.exp(np.clip(home_log, *bounds))
    result["lambda_away"] = np.exp(np.clip(away_log, *bounds))
    return result


def candidate_sources() -> tuple[FormSource, ...]:
    """Elo-only baseline, a goals control, and the xG candidate."""

    return ("none", "goals", "xg")
