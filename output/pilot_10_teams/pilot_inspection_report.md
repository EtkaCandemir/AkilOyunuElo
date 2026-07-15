# AO European Elo Pilot Inspection Report

This report is generated from the synthetic 10-team pilot output. It is a smoke-test artifact, not a calibrated production result.

## Summary

- Rows: 10
- AO First Elo min: 591.963
- AO First Elo max: 880.969
- Validation warning rows: 0

## Rating Source Distribution

- Pure Domestic Projection: 4
- European Evidence-Based Rating: 3
- Mixed Domestic-European Estimate: 3

## Acceptance Checks

- PASS: 10 teams produced (rows=10)
- PASS: All rating source categories present (European Evidence-Based Rating, Mixed Domestic-European Estimate, Pure Domestic Projection)
- PASS: No Europe team stays at Domestic Prior (Metro Albion)
- PASS: Low match volume exposure is about 0.713 (Few Match Wanderers european_exposure=0.713)
- PASS: Full evidence team exposure is 1.0 (Continental Giants european_exposure=1.000)
- PASS: Effective exposure is capped at 0.85 (Continental Giants effective_european_exposure=0.850)
- PASS: Unknown finish plus cup gets no double bonus (Cupmark Rangers cup_double_bonus=0.000)

## Readout

- Highest AO First Elo: Continental Giants (880.969)
- Lowest AO First Elo: Low Score Veterans (591.963)
- `Low Score Veterans` shows the intended downward pull while the effective exposure cap retains a 15% Domestic Prior contribution.
- `Few Match Wanderers` shows limited trust in a five-season but low-match sample.
- `Double Crown Athletic` confirms the cup double bonus path.

## Inspection Table

| scenario | team_name | competition | entry_round | domestic_prior | european_prior | european_exposure | effective_european_exposure | ao_first_elo | rating_source_type | weighted_match_exposure | weighted_european_history | domestic_achievement_score | cup_double_bonus | validation_warnings |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Five seasons with full evidence | Continental Giants | UCL | League Phase | 766.521 | 901.165 | 1.0 | 0.85 | 880.969 | European Evidence-Based Rating | 1.0 | 17.32 | 0.813 | 0.0 |  |
| League and cup double winner | Double Crown Athletic | UCL | Playoff Round | 763.501 | 792.275 | 0.968 | 0.85 | 787.959 | European Evidence-Based Rating | 0.921 | 7.32 | 1.08 | 0.08 |  |
| Five seasons with low match volume | Few Match Wanderers | UECL | League Phase | 743.175 | 770.982 | 0.713 | 0.713 | 763.011 | Mixed Domestic-European Estimate | 0.283 | 6.13 | 0.666 | 0.0 |  |
| Medium league champion, no European history | Midoria Champions | UCL | Second Qualifying Round | 752.255 | 500.0 | 0.0 | 0.0 | 752.255 | Pure Domestic Projection | 0.0 | 0.0 | 1.0 | 0.0 |  |
| Big league, no European history | Metro Albion | UEL | League Phase | 749.011 | 500.0 | 0.0 | 0.0 | 749.011 | Pure Domestic Projection | 0.0 | 0.0 | 0.703 | 0.0 |  |
| Only latest season in Europe | Last Season Sparks | UECL | Third Qualifying Round | 718.048 | 701.22 | 0.33 | 0.33 | 712.495 | Mixed Domestic-European Estimate | 0.33 | 3.3 | 0.757 | 0.0 |  |
| Small league champion, no European history | Smallia Kings | UCL | First Qualifying Round | 706.451 | 500.0 | 0.0 | 0.0 | 706.451 | Pure Domestic Projection | 0.0 | 0.0 | 1.0 | 0.0 |  |
| Older European history, no recent seasons | Distant History FC | UECL | Second Qualifying Round | 688.849 | 679.365 | 0.4 | 0.4 | 685.055 | Mixed Domestic-European Estimate | 0.4 | 2.67 | 0.688 | 0.0 |  |
| Unknown league finish plus cup winner | Cupmark Rangers | UEL | Playoff Round | 679.85 | 500.0 | 0.0 | 0.0 | 679.85 | Pure Domestic Projection | 0.0 | 0.0 | 0.62 | 0.0 |  |
| High exposure with weak European points | Low Score Veterans | UCL | League Phase | 796.121 | 555.935 | 1.0 | 0.85 | 591.963 | European Evidence-Based Rating | 1.0 | 0.5 | 1.0 | 0.0 |  |
