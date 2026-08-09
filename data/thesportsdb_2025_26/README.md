# TheSportsDB 2025/26 UEFA Match Dataset

This directory contains the complete premium API extraction for 2025/26 UCL,
UEL and UECL matches. The grain of `events.csv` and `event_stats_wide.csv` is
one row per AO match. Stats, timeline and lineup tables are long-form child
tables joined by `match_id` and `source_event_id`.

## Coverage

- Matches: 961
- Match identities: 961
- Stats: 648/961
- Raw xG fields: 648/961
- Analysis-eligible TheSportsDB xG: 437/961
- Suspected xG placeholders: 211
- Common eligible TheSportsDB/FotMob xG: 437
- Timeline: 938/961
- Lineup: 845/961

`expected_goals` is preserved exactly as returned by TheSportsDB. Its provider
model, penalty inclusion and 90/120-minute scope are not documented, so it must
not replace the audited production xG source without a separate comparison.
Provider rows where xG is zero despite recorded shots, and rows where both xG
values are zero, remain in the raw columns but are excluded by
`xg_analysis_eligible=false`. `xg_source_comparison.csv` contains the match-level
FotMob comparison and `xg_source_comparison_summary.csv` contains aggregate
agreement metrics.

Raw API payloads are stored under ignored `_source_cache/`. API credentials are
read from `.env.local` and are never written to any output.
