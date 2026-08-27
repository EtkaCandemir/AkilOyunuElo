# Curated Reports

This directory contains the small, review-ready evidence packages that belong
in Git. Full replay tables, prediction ledgers, caches, and temporary outputs
are generated under `output/` and intentionally ignored.

## Current model

`current_model/` is the canonical evaluation package for the active production
contract. Start with `current_model_evaluation_report.md`; use the JSON and CSV
files for parameter and metric auditing.

The current report's historical replay metrics predate the 18 August 2026
continuous qualifier-retention activation. Active qualifier behavior is defined
by the production contract: effective Q1/Q2/Q3/QPO multipliers
`0.20/0.275/0.35/0.425`, MAIN `1.00`, and no non-match MAIN-entry reset.

## External benchmark

`external_benchmark/` is the only package that scores the active contract
against references outside this repository: ClubElo for match prediction and the
pre-season Opta Power Rankings snapshot for the season-start rating. Every other
report compares the model with its own ablation arms, so this package is the
reference point for "is the model actually good", not just "did the layer help".

## Cup achievement challenger

`cup_achievement/` records the walk-forward test of a generalized domestic cup
contribution. The active model combines the league finish and the cup with
`max`, which makes the cup a floor rather than a contribution: in `2025/26`,
`35` of `51` cup winners received no credit for the trophy. The generalization
wins `6/6` folds at `w=0.05` and `w=0.08` but never clears the conservative
dependency envelope, so the decision is `KEEP_SHADOW`. The package is kept
because the negative result is the answer to a real design question, not an
absence of one.

## xG multiseason revalidation

`xg_multiseason/` re-tests the shipped bounded xG rating layer on the six
seasons that have FotMob coverage. The layer was activated on `606` eligible
matches from a single season; the wider sample carries `2827`. It is the one
layer in this repository that clears the conservative dependency envelope on
every segment, and the gain sits exactly where xG exists. The package records
evidence only: the contract was not changed.

## xG-informed goal expectation

`xg_goal_expectation/` tests whether recent xG predicts goals better than
recent goals do. It clears the conservative envelope only on the segment where
xG exists, and its baseline is simpler than the repository's best existing
scoreline arm, so the decision is `KEEP_SHADOW`. The package documents what the
next test has to be rather than claiming the layer is ready.

## European prior participation normalization

`european_participation/` removes a double charge from the European Prior. A
season the club did not enter contributes zero UEFA points, which is
indistinguishable from a season it entered and lost every match, so the prior
answers "did you qualify" as well as "how good were you" - and qualifying is a
domestic achievement the Domestic Prior already owns. The layer renormalizes
the history over participated seasons only, leaving the exposure weight
untouched, and is neutral by construction for the `644` club-seasons with
complete five-season evidence.

It improves the season-start seed by `+0.009434` Spearman and beats a blind
control that lifts the same population by the same average amount - the control
actually loses `0.002465`, which is what makes the gain attributable to the
participation structure rather than to the lift. Match loss improves as well
(`-0.002051` Brier), and all four intervals exclude zero. Nine gates pass.

**Activated on 2026-08-27** at `k = 0.20`, the modal fold choice, after two
further gates cleared: the exposure cap sweep below and the served-layer replay
in `participation_served_ensemble/`.

## Exposure cap x participation interaction

`participation_exposure_interaction/` asks whether the exposure cap should move
once the European Prior is cleaner. It sweeps the same fourteen cap values on
both priors, re-selecting the participation shrinkage inside every fold at
every cap. Two findings: the layers are not substitutes - the participation
gain is roughly constant across the whole cap range - and the ranking benefit
that raising the cap used to buy is absorbed by the cleaner prior, so `0.65` to
`0.70` now buys nothing. Most decisively, **no cap in `0.40`-`0.85` is reliably
different from the activated `0.65`**: every season-block interval crosses
zero. The decision is `KEEP_CURRENT_CAP`, taken on loss, shrinkage stability
and absence of evidence rather than on a peak read off a flat curve.

## Served layer under the participation seed

`participation_served_ensemble/` is the gate the participation study asked for
before activation. The served `%50 ML + %50 Poisson` blend chose its weights
while every arm read the production seed, and the Structural Logistic features
read the AO log-odds and the AO rating differences, so a seed that moves `1008`
of `1887` club-seasons cannot be assumed safe. The full chain - feature store,
ML backtest, Poisson backtest, ensemble - was rebuilt on both seeds. The served
layer's edge over its own AO base is `-0.004450` on the control seed and
`-0.004360` on the candidate, and no segment shows reliable harm in either arm.

## Domestic Surprise amplification

`domestic_surprise_amplification/` tests whether widening the active Domestic
Surprise coefficients improves the season-start seed. It does not: the pooled
gain is indistinguishable from zero, same-season ranking degrades with a
confidence interval that excludes zero, and six folds select five different
configurations. The decisive finding is that the gain was already taken by the
European exposure cap activation - rerunning against the current contract
instead of the pre-cap one shrinks the pooled Brier difference by a factor of
`53`. Decision `REJECTED`. The package also records three measurement faults
that were corrected before the verdict, all of which had pointed toward
promoting the layer.

## xG form on the domestic attack/defence arm

`xg_domestic_goal_expectation/` runs the test the package above asked for. It
puts the same form term **on top of** `DOMESTIC_ATTACK_DEFENCE_POISSON` rather
than in place of it, on the identical `3528` matches, and keeps the goals
control so the gain can be attributed to the source rather than to the two
extra parameters. About `76%` of the earlier gain survives the stronger
baseline, and the xG arm clears the conservative envelope against both the
baseline and the control on `PHASE:MAIN` and `XG_PRESENT`. Pooled `ALL` still
crosses zero and over/under and BTTS are still at climatology, so the decision
is `KEEP_SHADOW_CANDIDATE` and the score layer stays `Diagnostic`. A
`sensitivity_xg_training_window/` sub-package repeats the run on the narrower
training window and reaches the same conclusion.

## AO First seed asymmetry

`ao_first_seed_boost/` tests whether a club with thin negative European history
needs a narrow season-start correction when a causal domestic-form signal is
strong. The blind rule is retained only as a diagnostic control. The independent
domestic-form arm affects just `1/1413` unseen team-seasons and does not close
the 2025/26 Opta ranking gap, so the decision is `KEEP_DIAGNOSTIC`. No production
contract or active rating parameter changed.
