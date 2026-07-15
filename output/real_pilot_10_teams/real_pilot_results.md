# AO European Elo v1.1 - Real 10-Team Pilot Results

Data freeze: 2026-07-13

Target season: `2026/27`

Active model parameters: `Country_Strength_Benchmark=25`, `gamma=0.8`, `domestic_league_component=140`, `European_History_Benchmark=20`.

> The `20 / 2.0 / 360` country candidate is rejected for production because it gives implausibly dominant ratings to zero-exposure clubs.

## Summary

- Teams: 10
- Minimum AO First Elo: 741.925
- Maximum AO First Elo: 902.000
- Validation warning rows: 0
- European Evidence-Based Rating: 8
- Mixed Domestic-European Estimate: 1
- Pure Domestic Projection: 1

## Ranking And Components

| rank | team_name | country_code | competition | domestic_prior | european_prior | european_exposure | effective_european_exposure | country_candidate_ao_first_elo | candidate_delta | candidate_rank | european_adjustment | ao_first_elo | rating_source_type |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Arsenal | ENG | UCL | 800.0 | 920.0 | 0.93 | 0.85 | 935.0 | 33.0 | 2 | 102.0 | 902.0 | European Evidence-Based Rating |
| 2 | Sporting CP | POR | UCL | 746.008 | 908.569 | 1.0 | 0.85 | 911.619 | 27.434 | 3 | 138.176 | 884.185 | European Evidence-Based Rating |
| 3 | Benfica | POR | UEL | 739.845 | 895.281 | 1.0 | 0.85 | 899.418 | 27.452 | 4 | 132.12 | 871.966 | European Evidence-Based Rating |
| 4 | Shakhtar Donetsk | UKR | UCL | 716.17 | 862.504 | 1.0 | 0.85 | 843.88 | 3.326 | 7 | 124.384 | 840.554 | European Evidence-Based Rating |
| 5 | Galatasaray | TUR | UCL | 753.187 | 853.31 | 0.87 | 0.85 | 854.984 | 16.693 | 6 | 85.104 | 838.292 | European Evidence-Based Rating |
| 6 | AZ Alkmaar | NED | UEL | 706.423 | 856.976 | 1.0 | 0.85 | 855.395 | 21.002 | 5 | 127.97 | 834.393 | European Evidence-Based Rating |
| 7 | Slavia Praha | CZE | UCL | 751.585 | 811.007 | 1.0 | 0.85 | 818.078 | 15.985 | 8 | 50.509 | 802.093 | European Evidence-Based Rating |
| 8 | Pafos | CYP | UEL | 697.766 | 791.839 | 0.6 | 0.6 | 786.487 | 32.278 | 9 | 56.444 | 754.21 | Mixed Domestic-European Estimate |
| 9 | Como | ITA | UCL | 748.124 | 500.0 | 0.0 | 0.0 | 978.316 | 230.191 | 1 | 0.0 | 748.124 | Pure Domestic Projection |
| 10 | Omonia Nicosia | CYP | UCL | 740.01 | 742.263 | 0.973 | 0.85 | 753.151 | 11.226 | 10 | 1.915 | 741.925 | European Evidence-Based Rating |

## Review Notes

- Como is the no-history control: exposure is zero and the final rating equals the domestic prior.
- Pafos is the recent-history control: only the latest two seasons carry evidence, so the European prior is partially blended.
- Raw exposure measures evidence and still drives the source category; effective exposure is capped at `0.85` only for the final blend.
- A raw exposure of `1.0` therefore preserves a `15%` Domestic Prior contribution instead of replacing it completely.
- The active model retains the frozen v1.1 country values `25 / 0.8 / 140`; European History Benchmark `20` is frozen for the static v1.1 release after the eight-season ranking-first review.
- `country_candidate_ao_first_elo` shows the rejected `20 / 2.0 / 360` candidate. Como rises to 978.316 under that candidate, which fails the zero-exposure plausibility guardrail.
- Competition level is metadata today. Benfica can rank close to Arsenal despite entering the UEL because UCL/UEL/UECL do not alter the current formula.
- Como ranks above Omonia because zero exposure preserves Como's strong Italian domestic prior, while Omonia's extensive European evidence pulls it toward a lower European prior.
- Positive or negative `european_adjustment` shows the direction of shrinkage from domestic evidence toward European evidence.
- AZ Alkmaar and Pafos receive the cup base score but no double bonus; the double bonus is reserved for league-and-cup champions.

## Provenance

See `data/real_pilot_10_teams/SOURCES.md`.
