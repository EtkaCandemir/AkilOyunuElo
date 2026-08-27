from __future__ import annotations

"""Does an xG form term still pay once the domestic attack/defence model is there?

The previous study (`xg_goal_model`) put one form term per side on top of an
Elo-only goal expectation and showed that feeding it xG beats feeding it goals.
That baseline is simpler than the repository's best scoreline arm,
`DOMESTIC_ATTACK_DEFENCE_POISSON`, so the result licensed only the narrow claim
"xG helps a simple model" - not "xG helps the best model".

This module closes that gap. It keeps the domestic transfer intact and adds the
same form term on top of it, so the two rate models differ by exactly the form
pair and nothing else:

```text
z          = logit(expected_home_score)                       # AO Live Elo
A_x, D_x   = league-relative domestic attack / defence z-scores
f_x, a_x   = recent scoring / conceding rate, centered        # the form term

lambda_home = exp(mu + 0.5*b*z + a*A_home - d*D_away + 0.5*v*edge + fa*f_home - fd*a_away)
lambda_away = exp(mu - 0.5*b*z + a*A_away - d*D_home - 0.5*v*edge + fa*f_away - fd*a_home)
```

Setting the form source to `none` reproduces the production-shadow transfer
exactly, which is what makes the three arms a controlled experiment:

| arm | form source | role |
| --- | --- | --- |
| `DOMESTIC_AD` | `none` | the repository's best scoreline arm, the baseline |
| `DOMESTIC_AD_GOALS_FORM` | `goals` | the control that isolates the source |
| `DOMESTIC_AD_XG_FORM` | `xg` | the candidate |

The goals control is not optional. Without it a gain in the xG arm cannot be
told apart from the gain of simply granting the model two more parameters.

Nothing here reads or writes a production parameter. The domestic features and
the form history are both causal: a match only ever sees rows that kicked off
strictly earlier.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

# The private helpers are imported rather than restated so the baseline arm
# cannot silently drift away from the transfer model it is meant to reproduce.
from ao_elo.domestic_poisson import (
    _ao_logit,
    _transfer_columns,
    _validate_transfer_frame,
    _validate_transfer_prediction_frame,
)
from ao_elo.xg_goal_model import (
    DEFAULT_SHRINKAGE,
    DEFAULT_WINDOW,
    LAMBDA_MAX,
    LAMBDA_MIN,
    FormSource,
    build_form_features,
    form_prior,
)


DOMESTIC_AD = "DOMESTIC_AD"
DOMESTIC_AD_GOALS_FORM = "DOMESTIC_AD_GOALS_FORM"
DOMESTIC_AD_XG_FORM = "DOMESTIC_AD_XG_FORM"
ARM_BY_SOURCE: dict[FormSource, str] = {
    "none": DOMESTIC_AD,
    "goals": DOMESTIC_AD_GOALS_FORM,
    "xg": DOMESTIC_AD_XG_FORM,
}
FORM_COLUMNS = (
    "home_form_for",
    "home_form_against",
    "away_form_for",
    "away_form_against",
)
COEFFICIENT_MAX = 1.50
MODEL_VERSION = "ao-domestic-form-goal-expectation-v1-research"


@dataclass(frozen=True)
class DomesticFormExpectationConfig:
    """Fitted goal rates for one form source on top of the domestic transfer."""

    form_source: FormSource
    mu: float
    elo_slope: float
    attack_coefficient: float
    defence_coefficient: float
    venue_coefficient: float
    form_attack: float
    form_defence: float
    l2_strength: float
    use_reliability: bool = True
    use_venue: bool = True
    window: int = DEFAULT_WINDOW
    shrinkage: float = DEFAULT_SHRINKAGE
    form_prior: float = 0.0
    rho: float = 0.0
    model_version: str = MODEL_VERSION

    def validate(self) -> None:
        if self.form_source not in ARM_BY_SOURCE:
            raise ValueError(f"unknown form source: {self.form_source}")
        if not math.log(0.30) <= float(self.mu) <= math.log(4.00):
            raise ValueError("mu must be within the configured goal-level bounds")
        if not 0.05 <= float(self.elo_slope) <= 3.00:
            raise ValueError("elo_slope must be in [0.05,3.00]")
        for name in (
            "attack_coefficient",
            "defence_coefficient",
            "venue_coefficient",
            "form_attack",
            "form_defence",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= COEFFICIENT_MAX:
                raise ValueError(f"{name} must be in [0,{COEFFICIENT_MAX}]")
        if float(self.l2_strength) < 0.0:
            raise ValueError("l2_strength must be non-negative")
        if not -0.15 <= float(self.rho) <= 0.15:
            raise ValueError("rho must be in [-0.15,0.15]")
        if int(self.window) < 1:
            raise ValueError("window must be positive")
        if float(self.shrinkage) < 0.0:
            raise ValueError("shrinkage must be non-negative")
        if not math.isfinite(float(self.form_prior)):
            raise ValueError("form_prior must be finite")
        if self.form_source == "none" and (self.form_attack or self.form_defence):
            raise ValueError("the domestic baseline arm cannot carry form coefficients")

    @property
    def arm(self) -> str:
        self.validate()
        return ARM_BY_SOURCE[self.form_source]

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def attach_form_features(
    frame: pd.DataFrame,
    source: FormSource,
    *,
    training_frame: pd.DataFrame | None = None,
    window: int = DEFAULT_WINDOW,
    shrinkage: float = DEFAULT_SHRINKAGE,
) -> tuple[pd.DataFrame, float]:
    """Attach the causal form pair, centered on training rows only.

    `build_form_features` derives its centering level from whatever frame it is
    handed. In a walk-forward that frame has to carry the test season - the
    history a test match reads lives in it - so the level is computed here from
    `training_frame` and passed back in. The returned prior is recorded on the
    fitted config so the choice is auditable.
    """

    prior = (
        0.0
        if source == "none"
        else form_prior(frame if training_frame is None else training_frame, source)
    )
    featured = build_form_features(
        frame, source, window=window, shrinkage=shrinkage, prior=prior
    )
    return featured, prior


def fit_domestic_form_expectation(
    frame: pd.DataFrame,
    source: FormSource,
    *,
    l2_strength: float,
    use_reliability: bool = True,
    use_venue: bool = True,
    window: int = DEFAULT_WINDOW,
    shrinkage: float = DEFAULT_SHRINKAGE,
    form_prior_value: float = 0.0,
) -> DomesticFormExpectationConfig:
    """Fit the rates by independent Poisson negative log-likelihood.

    The penalty covers every coefficient except the intercept, form terms
    included, so no arm is regularized on different terms than another.
    """

    if source not in ARM_BY_SOURCE:
        raise ValueError(f"unknown form source: {source}")
    if float(l2_strength) < 0.0:
        raise ValueError("l2_strength must be non-negative")
    data = _validate_form_frame(_validate_transfer_frame(frame), source)
    columns = _transfer_columns(use_reliability)
    z_ao = _ao_logit(data)
    attack_home = data[columns["home_attack"]].to_numpy(float)
    attack_away = data[columns["away_attack"]].to_numpy(float)
    defence_home = data[columns["home_defence"]].to_numpy(float)
    defence_away = data[columns["away_defence"]].to_numpy(float)
    venue = data[
        "domestic_poisson_venue_edge"
        if use_reliability
        else "domestic_poisson_venue_edge_raw"
    ].to_numpy(float)
    home_for = data["home_form_for"].to_numpy(float)
    home_against = data["home_form_against"].to_numpy(float)
    away_for = data["away_form_for"].to_numpy(float)
    away_against = data["away_form_against"].to_numpy(float)
    home_goals = data["home_goals"].to_numpy(float)
    away_goals = data["away_goals"].to_numpy(float)
    free_form = source != "none"
    if free_form and not np.any(np.abs(home_for) + np.abs(away_for) > 0.0):
        raise ValueError(
            f"{source} form carries no signal in the training frame; "
            "a coefficient with no evidence must not be fitted"
        )
    bounds_log = (math.log(LAMBDA_MIN), math.log(LAMBDA_MAX))
    initial_mu = math.log(
        float((home_goals.sum() + away_goals.sum()) / (2.0 * len(data)))
    )

    def unpack(parameters: np.ndarray) -> tuple[float, ...]:
        mu, beta, attack, defence = (float(value) for value in parameters[:4])
        offset = 4
        venue_coefficient = float(parameters[offset]) if use_venue else 0.0
        offset += 1 if use_venue else 0
        if free_form:
            form_attack = float(parameters[offset])
            form_defence = float(parameters[offset + 1])
        else:
            form_attack = form_defence = 0.0
        return mu, beta, attack, defence, venue_coefficient, form_attack, form_defence

    def objective(parameters: np.ndarray) -> float:
        mu, beta, attack, defence, venue_c, form_a, form_d = unpack(parameters)
        home_log = (
            mu
            + 0.5 * beta * z_ao
            + attack * attack_home
            - defence * defence_away
            + 0.5 * venue_c * venue
            + form_a * home_for
            - form_d * away_against
        )
        away_log = (
            mu
            - 0.5 * beta * z_ao
            + attack * attack_away
            - defence * defence_home
            - 0.5 * venue_c * venue
            + form_a * away_for
            - form_d * home_against
        )
        home_rate = np.exp(np.clip(home_log, *bounds_log))
        away_rate = np.exp(np.clip(away_log, *bounds_log))
        nll = (
            home_rate
            - home_goals * np.log(home_rate)
            + gammaln(home_goals + 1.0)
            + away_rate
            - away_goals * np.log(away_rate)
            + gammaln(away_goals + 1.0)
        ).sum()
        penalty = float(l2_strength) * float(np.square(parameters[1:]).sum())
        return float(nll + penalty)

    initial = [initial_mu, 0.75, 0.05, 0.05]
    bounds = [
        (math.log(0.30), math.log(4.00)),
        (0.05, 3.00),
        (0.0, COEFFICIENT_MAX),
        (0.0, COEFFICIENT_MAX),
    ]
    if use_venue:
        initial.append(0.02)
        bounds.append((0.0, COEFFICIENT_MAX))
    if free_form:
        initial.extend([0.10, 0.10])
        bounds.extend([(0.0, COEFFICIENT_MAX), (0.0, COEFFICIENT_MAX)])
    fitted = minimize(
        objective,
        np.array(initial, dtype=float),
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-11, "gtol": 1e-7, "maxiter": 1000},
    )
    if not fitted.success or not np.isfinite(fitted.fun):
        raise ValueError(f"domestic form expectation fit failed: {fitted.message}")
    mu, beta, attack, defence, venue_c, form_a, form_d = unpack(fitted.x)
    config = DomesticFormExpectationConfig(
        form_source=source,
        mu=mu,
        elo_slope=beta,
        attack_coefficient=attack,
        defence_coefficient=defence,
        venue_coefficient=venue_c,
        form_attack=form_a,
        form_defence=form_d,
        l2_strength=float(l2_strength),
        use_reliability=bool(use_reliability),
        use_venue=bool(use_venue),
        window=int(window),
        shrinkage=float(shrinkage),
        form_prior=float(form_prior_value),
    )
    config.validate()
    return config


def predict_domestic_form_expectation(
    frame: pd.DataFrame,
    config: DomesticFormExpectationConfig,
) -> pd.DataFrame:
    """Return bounded home and away rates for already-built form features."""

    config.validate()
    data = _validate_form_frame(
        _validate_transfer_prediction_frame(frame), config.form_source
    )
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
        + config.form_attack * data["home_form_for"].to_numpy(float)
        - config.form_defence * data["away_form_against"].to_numpy(float)
    )
    away_log = (
        config.mu
        - 0.5 * config.elo_slope * z_ao
        + config.attack_coefficient * data[columns["away_attack"]].to_numpy(float)
        - config.defence_coefficient * data[columns["home_defence"]].to_numpy(float)
        - 0.5 * config.venue_coefficient * venue
        + config.form_attack * data["away_form_for"].to_numpy(float)
        - config.form_defence * data["home_form_against"].to_numpy(float)
    )
    bounds_log = (math.log(LAMBDA_MIN), math.log(LAMBDA_MAX))
    result = data[["match_id"]].copy()
    result["lambda_home"] = np.exp(np.clip(home_log, *bounds_log))
    result["lambda_away"] = np.exp(np.clip(away_log, *bounds_log))
    return result


def candidate_form_sources() -> tuple[FormSource, ...]:
    """The domestic baseline, the goals control, and the xG candidate."""

    return ("none", "goals", "xg")


def _validate_form_frame(frame: pd.DataFrame, source: FormSource) -> pd.DataFrame:
    missing = sorted(set(FORM_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"form features missing: {missing}")
    values = frame[list(FORM_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any() or not np.isfinite(values.to_numpy(float)).all():
        raise ValueError("form features must be finite")
    if source == "none" and values.to_numpy(float).any():
        raise ValueError("the domestic baseline arm must be handed zeroed form")
    return frame


def independent_poisson_nll(
    home_goals: Sequence[float],
    away_goals: Sequence[float],
    lambda_home: Sequence[float],
    lambda_away: Sequence[float],
) -> float:
    """Mean two-sided Poisson NLL, the objective the inner loop selects on."""

    home_goals = np.asarray(home_goals, dtype=float)
    away_goals = np.asarray(away_goals, dtype=float)
    lambda_home = np.asarray(lambda_home, dtype=float)
    lambda_away = np.asarray(lambda_away, dtype=float)
    if not len(home_goals):
        raise ValueError("cannot score an empty frame")
    return float(
        np.mean(
            lambda_home
            - home_goals * np.log(lambda_home)
            + gammaln(home_goals + 1.0)
            + lambda_away
            - away_goals * np.log(lambda_away)
            + gammaln(away_goals + 1.0)
        )
    )
