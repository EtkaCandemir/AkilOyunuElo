# AO European Elo v2 Achievement Reserve Calibration

Exact-date seasons: `2018/19` through `2025/26`.
The comparator is the selected Power core, season carry and active goal layer.
Normal match updates remain zero-sum; reserve is a separate non-zero-sum state.

## Formula

```text
Advance Reserve = Base * Competition * Stage * (1 - P_advance)
Trophy Reserve = Base * Competition * (1 - P_advance)
AO Live Elo = Power Elo + Achievement Reserve
```

The expectation is frozen immediately before the tie's first match. Reserve is added only after the tie is decided and is visible from the next match onward.

## Decision

- Result: `DISABLE_ACHIEVEMENT_RESERVE`
- Candidate base: `0.000000`
- Candidate UEL / UECL: `0.65` / `0.45`
- Candidate profile / decay: `FLAT` / `0`
- Active base: `0.000000`
- Unseen fold wins: `0/6`
- Overall Brier difference: `+0.000000`
- Clustered 95% CI: `[+0.000000, +0.000000]`
- Ranking safe: `True`
- Reserve cap safe: `True`

| Competition | Matches | Brier difference | Log-loss difference |
| --- | ---: | ---: | ---: |
| UCL | 1384 | +0.000000 | +0.000000 |
| UECL | 2073 | +0.000000 | +0.000000 |
| UEL | 1427 | +0.000000 | +0.000000 |

If disabled, competition and stage multipliers remain documented research candidates but add exactly zero reserve in production.
