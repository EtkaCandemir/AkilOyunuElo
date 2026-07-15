# External Historical Elo Benchmark Requirements

An external Elo comparator can validate absolute rating quality, but it cannot
replace the paired ablation that isolates one AO model layer.

Required before a leakage-safe comparison:

- exact match date or timestamp for every event;
- a historical pre-match club Elo snapshot from one consistent provider;
- a durable provider-club identifier mapped to AO global club_key;
- documented treatment of neutral venues, extra time and promoted/new clubs;
- snapshots captured before kickoff, never end-of-season or current ratings.

The current event set intentionally does not assert exact dates, so joining a
daily external Elo history now would risk look-ahead leakage. The benchmark must
wait until exact dates and historical snapshots are added and audited.
