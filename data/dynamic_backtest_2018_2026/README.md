# AO Dynamic Elo Calibration Events

This is a research dataset for chronological dynamic-rating calibration.
Its score and team universe is preserved exactly from `/Users/buycell/Desktop/Akil Oyunu Elo/data/backtest_stage_b_2018_2026`.

The key grain is one played match (`match_id`). `event_order` is the
chronological order required by the Elo update. Exact calendar timestamps are
not claimed: knockout legs use source round/leg order, group stages are rebuilt
to the six official matchdays, and league stages use validated source blocks.
This is sufficient because updates for matches involving disjoint teams commute.

Penalty shootouts do not change the match score: a level match remains 0.5.
The advancing club is stored separately in `advanced_team_id` for the later
progression calibration layer. Displayed goals may include extra time; exact
90/120-minute splits are intentionally not asserted in this first dataset.

| Season | Matches | Knockout ties | Shootouts |
| --- | ---: | ---: | ---: |
| 2018/19 | 735 | 250 | 6 |
| 2019/20 | 721 | 250 | 7 |
| 2020/21 | 540 | 250 | 21 |
| 2021/22 | 816 | 267 | 19 |
| 2022/23 | 804 | 261 | 31 |
| 2023/24 | 806 | 262 | 25 |
| 2024/25 | 957 | 282 | 29 |
| 2025/26 | 961 | 284 | 17 |

Build audit: 72/72 checks passed.
