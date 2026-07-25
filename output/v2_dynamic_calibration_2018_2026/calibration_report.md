# AO European Elo v2 Dynamic Core And Carry Calibration

Exact-date development seasons: `2018/19` through `2025/26`.
All matches are processed in exact UTC order. Penalty shootouts do not alter the field score contract.

## Dynamic Core

- Decision: `PROMOTE_DYNAMIC_CORE`
- Scale: `835.561497`
- Home advantage: `148.544266`
- K: `103.980986`
- Unseen overall Brier difference: `-0.003955`
- Clustered 95% CI: `[-0.005491, -0.002370]`

| Competition | Matches | Brier difference | Log-loss difference |
| --- | ---: | ---: | ---: |
| UCL | 1384 | -0.004570 | -0.010609 |
| UECL | 2073 | -0.002631 | -0.006285 |
| UEL | 1427 | -0.005282 | -0.010637 |

## Season Carry

- Decision: `PROMOTE_CARRY`
- Full-data carry candidate: `0.85`
- Active carry: `0.85`
- Unseen overall Brier difference: `-0.002597`
- Clustered 95% CI: `[-0.003932, -0.001197]`

| Competition | Matches | Brier difference | Log-loss difference |
| --- | ---: | ---: | ---: |
| UCL | 1384 | -0.002963 | -0.007522 |
| UECL | 2073 | -0.001161 | -0.002827 |
| UEL | 1427 | -0.004327 | -0.010312 |

The core and carry are promoted independently. Goal margin and European Achievement Reserve remain disabled until their incremental tests pass.

The `2026/27` season remains untouched for future holdout evaluation.
