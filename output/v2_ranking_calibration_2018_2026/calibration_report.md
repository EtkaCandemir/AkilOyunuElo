# AO European Elo v2 Ranking-First Calibration

Development seasons: `2018/19` through `2025/26`.
Candidates: `100`. Outer folds: `6`.
The 500-2000 range is a reference band, not a clipping boundary.

## Decision

- Result: `NO_PROMOTION`
- Evaluated candidate: `c0_e0_x0`
- Active candidate: `c0_e0_x0`
- No unseen fold regression: `True`
- Folds improving Spearman and pairwise: `0/6`
- No competition regression: `True`

## Distribution

- Reference-band share: `100.000%`
- Median / p90: `1075.83` / `1747.65`
- Min / max: `651.49` / `1996.09`
- Exact 500/2000 values: `0`

## External UCL Guardrail

- Baseline static UCL Brier: `0.175531`
- Candidate static UCL Brier: `0.175531`
- Candidate - baseline: `+0.000000`
- Clustered 95% CI: `[+0.000000, +0.000000]`
- ClubElo rank correlation delta: `+0.000000`
- Elite cap-match mean rating-gap delta: `+0.000`

## Unseen Competition Ranking

| Competition | Model | Spearman | Delta | Pairwise | Delta |
| --- | --- | ---: | ---: | ---: | ---: |
| UCL | baseline | 0.4367 | +0.0000 | 0.6691 | +0.0000 |
| UCL | selected | 0.4367 | +0.0000 | 0.6691 | +0.0000 |
| UECL | baseline | 0.3484 | +0.0000 | 0.6351 | +0.0000 |
| UECL | selected | 0.3484 | +0.0000 | 0.6351 | +0.0000 |
| UEL | baseline | 0.4521 | +0.0000 | 0.6756 | +0.0000 |
| UEL | selected | 0.4521 | +0.0000 | 0.6756 | +0.0000 |

## Saturation Diagnostic

- UCL matches: `171`
- Spearman saturation-count vs Brier: `0.0671`
- Pearson saturation-count vs Brier: `-0.0050`

A positive correlation is diagnostic evidence that heavily saturated matchups are harder for the static model; it is not by itself a promotion rule.

The `2026/27` season remains outside parameter selection and is reserved for the future untouched holdout.
