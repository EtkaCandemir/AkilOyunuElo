# Exact-Date External Elo Benchmark Data

This dataset attaches official UEFA kickoff timestamps and a historical external
ClubElo benchmark to the AO dynamic match history.

## Leakage contract

- UEFA kickoff timestamps come from the official UEFA match service.
- ClubElo values use the latest archived snapshot strictly before the match date.
- Missing provider coverage stays missing; values are never imputed from the future.
- Snapshots older than 31 days are excluded from paired coverage.
- ClubElo archive snapshots are available on the 1st and 15th, so ratings can be stale
  by up to roughly two weeks. This limitation must accompany benchmark results.

## Files

- `matches_with_dates_and_external_elo.csv`: full row-level benchmark contract.
- `exact_date_events.csv`: dynamic events reordered by exact UTC kickoff.
- `uefa_team_identity_audit.csv`: AO-to-UEFA club identity decisions.
- `uefa_match_identity_audit.csv`: one-to-one fixture reconciliation.
- `clubelo_identity_audit.csv`: provider name coverage.
- `build_audit.csv`: required checks and descriptive coverage checks.
- `source_manifest.json`: source endpoints, retrieval time and checksum.

## Coverage

```text
 season competition  matches  exact_dates  external_pairs
2018/19         UCL      216          216              24
2018/19         UEL      519          519              47
2019/20         UCL      210          210              28
2019/20         UEL      511          511              30
2020/21         UCL      178          178              32
2020/21         UEL      362          362              21
2021/22         UCL      218          218              35
2021/22        UECL      423          423              22
2021/22         UEL      175          175              14
2022/23         UCL      214          214              28
2022/23        UECL      415          415              20
2022/23         UEL      175          175              32
2023/24         UCL      214          214              36
2023/24        UECL      417          417              24
2023/24         UEL      175          175              14
2024/25         UCL      279          279              40
2024/25        UECL      409          409              24
2024/25         UEL      269          269              21
2025/26         UCL      281          281               0
2025/26        UECL      409          409               0
2025/26         UEL      271          271               0
```

Required audit failures: 0

ClubElo archive URL:
https://raw.githubusercontent.com/xgabora/Club-Football-Match-Data-2000-2025/main/data/EloRatings.csv
