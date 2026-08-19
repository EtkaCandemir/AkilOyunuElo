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
