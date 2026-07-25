# AO European Elo v2 Evaluation Upgrade

Development window: `2018/19` through `2025/26`. The `2026/27` season remains untouched.

## Adjusted Ranking Target

The target removes a leave-team-out home-edge estimate and adds opponent strength estimated without the direct target-opponent meetings. It never uses AO or ClubElo ratings.

- Static tail decision: `NO_PROMOTION`
- Evaluated candidate: `c0_e0_x0`
- Active candidate: `c0_e0_x0`
- No unseen fold regression: `True`
- Both rank metrics improved: `0/6` folds
- No competition regression: `True`

## Standard 1X2 Output

Brier is the sum of the three squared class errors (range 0-2). Log loss uses the observed H/D/A probability. The draw mapping preserves the Elo expected-points identity exactly.

- Probability decision: `PROMOTE_1X2_OUTPUT`
- Full-data carry candidate: `0.85`
- Gate-approved carry: `0`

| Competition | Draw at even | Shape |
| --- | ---: | ---: |
| ALL | 0.24 | 1.00 |
| UCL | 0.24 | 1.00 |
| UECL | 0.24 | 1.00 |
| UEL | 0.24 | 1.00 |

| Segment | Matches | AO Brier | Base Brier | Difference | AO Log | Base Log | Difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ALL | 4884 | 0.575337 | 0.628739 | -0.053402 | 0.970015 | 1.042101 | -0.072086 |
| UCL | 1384 | 0.557583 | 0.627914 | -0.070331 | 0.944045 | 1.039621 | -0.095576 |
| UECL | 2073 | 0.583028 | 0.629335 | -0.046307 | 0.981064 | 1.043672 | -0.062607 |
| UEL | 1427 | 0.581384 | 0.628673 | -0.047289 | 0.979151 | 1.042223 | -0.063073 |

## Layer Revalidation

- Dynamic core: `CONFIRMED_1X2`
- Season carry: `DISABLE_CARRY_1X2`

The reported confidence interval is the conservative envelope of tie/match, team-season, and calendar-month bootstrap views. These are dependency sensitivity analyses, not a formal multi-way clustered standard-error claim.

| Layer | Competition | Loss | Mean difference | Envelope 95% CI |
| --- | --- | --- | ---: | --- |
| dynamic_core_vs_tuned_static | ALL | brier_1x2 | -0.008929 | [-0.012823, -0.004990] |
| dynamic_core_vs_tuned_static | ALL | log_loss_1x2 | -0.012527 | [-0.017869, -0.007109] |
| dynamic_core_vs_tuned_static | UCL | brier_1x2 | -0.010540 | [-0.016401, -0.004486] |
| dynamic_core_vs_tuned_static | UCL | log_loss_1x2 | -0.015141 | [-0.023558, -0.006437] |
| dynamic_core_vs_tuned_static | UECL | brier_1x2 | -0.006905 | [-0.014002, -0.000123] |
| dynamic_core_vs_tuned_static | UECL | log_loss_1x2 | -0.009729 | [-0.019584, -0.000477] |
| dynamic_core_vs_tuned_static | UEL | brier_1x2 | -0.010306 | [-0.017919, -0.002903] |
| dynamic_core_vs_tuned_static | UEL | log_loss_1x2 | -0.014054 | [-0.024680, -0.003944] |
| carry_vs_no_carry | ALL | brier_1x2 | -0.005895 | [-0.009464, -0.002653] |
| carry_vs_no_carry | ALL | log_loss_1x2 | -0.008550 | [-0.013651, -0.003838] |
| carry_vs_no_carry | UCL | brier_1x2 | -0.008087 | [-0.014882, -0.001095] |
| carry_vs_no_carry | UCL | log_loss_1x2 | -0.010796 | [-0.020943, -0.000303] |
| carry_vs_no_carry | UECL | brier_1x2 | -0.001806 | [-0.005730, +0.002136] |
| carry_vs_no_carry | UECL | log_loss_1x2 | -0.003207 | [-0.008777, +0.002393] |
| carry_vs_no_carry | UEL | brier_1x2 | -0.009708 | [-0.013955, -0.005277] |
| carry_vs_no_carry | UEL | log_loss_1x2 | -0.014134 | [-0.020214, -0.007880] |

Static rating parameters remain unchanged. The production season carry is set to zero because the recalibrated carry process won only five of six unseen folds. Any future static-tail promotion must still be followed by a fresh dynamic calibration.
