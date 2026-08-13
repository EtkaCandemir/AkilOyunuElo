# 2025/26 External Ranking Snapshots

This directory stores the frozen external inputs used to compare the 2025/26
AO season-end ranking with two different benchmarks:

- UEFA 2026 sporting club coefficients: a five-season European achievement and
  seeding measure.
- Opta Power Rankings dated 10 August 2026: a current-strength measure that may
  include early 2026/27 evidence.

The two rankings are intentionally not merged into one target. UEFA is matched
by the permanent UEFA team ID. Opta is matched with UEFA-assisted exact names
and a small audited alias table. Source URLs, dates, checksums and the temporal
caveat are recorded in `source_manifest.json`.

The directory also contains a separate pre-season Opta snapshot archived at
`2025-07-03T23:11:50Z`, strictly before the first 2025/26 UEFA match on 8 July.
It is the temporally correct external ranking check for AO First Elo. The
normalized 236-team extract and its independent source manifest are:

- `opta_power_rankings_2025_07_03_ao_scope.csv`
- `opta_power_rankings_2025_07_03_manifest.json`

Rebuild the normalized inputs and comparison from frozen raw downloads:

```bash
python3 scripts/run_external_ranking_comparison_2025_26.py \
  --uefa-raw-json /path/to/uefa-club-coefficients-2026.json \
  --opta-raw-json /path/to/opta-power-rankings-2026-08-10.json
```

Once the normalized inputs exist, the comparison is reproducible without
network access:

```bash
python3 scripts/run_external_ranking_comparison_2025_26.py
```

Rebuild the pre-season comparison from the frozen Wayback response:

```bash
python3 scripts/run_initial_elo_external_comparison_2025_26.py \
  --opta-raw-json /path/to/opta-power-rankings-2025-07-03.json.gz
```

Once normalized, it can also be reproduced offline:

```bash
python3 scripts/run_initial_elo_external_comparison_2025_26.py
```
