# AO European Elo Historical Backtest Data

This dataset is generated from pre-season and completed-season Kassiesa UEFA tables.
Every target season uses only information available before that season starts.

| Target season | Teams | Matches |
| --- | ---: | ---: |
| 2021/22 | 237 | 816 |
| 2022/23 | 233 | 804 |
| 2023/24 | 234 | 806 |
| 2024/25 | 236 | 957 |
| 2025/26 | 236 | 961 |

## Stage A limitation

Domestic qualification routes identify league champions and cup entrants, but this
first layer intentionally leaves domestic position and league team count empty for
non-champions. It is suitable for calibrating European-history and exposure parameters,
not the domestic percentile/component parameters.

## Sources

- https://kassiesa.net/uefa/data/index.html
- https://kassiesa.net/uefa/history/qual2024.html (year changes by season)

Build audit: 20/20 checks passed.
