# AO Permanent Club Identity Registry

`club_id` is the stable cross-season identifier. Local `team_id` values remain valid only
inside their own season and must never be used for cross-season joins.

Contract: `AO-UEFA-<positive UEFA team ID>`.

- Verified clubs: 504
- Verified team-seasons: 1887
- Verified matches: 6340
- Audit checks passed: 10/10

`team_season_identity.csv` is the mandatory bridge from `season + local_team_id` to
`club_id`. `club_registry.csv` stores one canonical row per club. `match_identity.csv`
proves both event sides agree with UEFA's provider IDs.
