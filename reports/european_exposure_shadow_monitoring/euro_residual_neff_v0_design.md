# EURO_RESIDUAL_NEFF_V0 — Design Specification

Status: design-only, not numerically frozen, not a prospective prediction arm.

This challenger keeps incumbent AO First and adds only a posterior estimate of
systematic historical European residual:

```text
R_shadow(i,T) = R_production(i,T) + beta * posterior_residual(i,T), beta >= 0
```

For each completed historical match, use the incumbent AO pre-match expectation:

```text
residual_home = actual_home_score - expected_home_score_production
residual_away = -residual_home
```

`actual_home_score` is `1 / 0.5 / 0` from the 90/120-minute field result;
shootout goals do not change it. The expectation comes from the locked AO-core
snapshot, not served ML/Poisson probabilities, and preserves
`P_home + 0.5 * P_draw = E_home`.

## Evidence units and hierarchy

- A non-empty `tie_id` is one evidence unit; league and single-match fixtures use
  `match_id`.
- Canonical stage changes residual variance/precision only. It never gives a
  direct rating bonus or penalty.
- Club-season and club effects are partially pooled:

```text
z(i,s,u) = theta(i) + b(i,s) + epsilon(i,s,u)
b(i,s) ~ Normal(0, sigma_season^2)
epsilon(i,s,u) ~ Normal(0, sigma_stage^2)
theta(i) ~ Normal(0, sigma_club^2)
```

The challenger correction uses the posterior mean of `theta`. Missing seasons
add no zero-result observation; they add no precision, so the posterior remains
closer to zero. With no evidence, the correction is exactly zero.

Audit both unit and season effective sample sizes:

```text
n_eff_unit   = sum(precision_u)^2 / sum(precision_u^2)
n_eff_season = sum(season_precision_s)^2 / sum(season_precision_s^2)
```

This distinguishes many observations concentrated in one season from evidence
repeated across several seasons.

## Timing and frozen decisions

For target season `T`, only completed fixtures whose kickoff and result precede
the seed cutoff may enter. Every expected score must come from a pre-match
snapshot locked before any result in the same exact-UTC batch was processed.

Structural decisions to freeze before numerical fitting:

- five completed-season window;
- production recency weights `0.07 / 0.13 / 0.20 / 0.27 / 0.33`;
- source production contract and revision hashes;
- residual outcome and AO expectation semantics;
- tie-unit definition and canonical stage map;
- hierarchical model family and fit rule;
- non-negative `beta` constraint.

Variance components and `beta` are estimated only inside training data. Every
outer fold fits variance components on earlier seasons and selects `beta`, with
`beta=0` included, using inner expanding walk-forward log-loss. No 2026/27
result may define or select the model.

## Invariants

- Match residuals are zero-sum within `1e-12`.
- A positive posterior cannot create a negative correction.
- Zero evidence produces zero correction.
- Stage alone cannot move a rating.
- Duplicate, identity, timing or stage errors fail closed; no silent drop or
  imputation.
- Production contract, artifacts, ledger and state are never mutated.

## Evaluation

Compare factorially:

1. production baseline;
2. `POSITIVE_BRIDGE_020`;
3. `EURO_RESIDUAL_NEFF_V0`;
4. bridge plus structural residual.

Each arm requires candidate seed generation, exact-UTC replay, feature rebuild,
Structural Logistic retraining, Poisson transfer refitting and final predictions
under the frozen production blend policy. Report paired Brier, log-loss,
accuracy, calibration, seed ranking and UCL/UEL/UECL-stage-ESS segments with
tie/match, team-season and calendar-month dependency views.

A full five-season history for every outer fold may require 2015/16–2017/18
pre-match replay data. If unavailable, early folds are left-truncated and the
result is feasibility/shadow evidence, not promotion evidence.
