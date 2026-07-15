# Real Pilot 10 Teams - Data Provenance

Data freeze: 2026-07-13

Target rating season: `2026/27`. Therefore `t` is the completed `2025/26`
season and `t_minus_4` is `2021/22`.

## Scope

This is a model-ready real-data pilot, not a production-calibrated rating set.
It contains ten clubs selected to cover large and small associations, domestic
champions, cup winners, clubs with full European history, a recent entrant and
a club with no European history in the five-season window.

The 2026/27 UEFA participant lists were provisional when the data was frozen.

## Sources

- UEFA 2026/27 Champions League teams and entry rounds:
  https://www.uefa.com/uefachampionsleague/clubs/
- UEFA 2026/27 Europa League teams and entry rounds:
  https://www.uefa.com/uefaeuropaleague/clubs/
- UEFA association rankings:
  https://www.uefa.com/nationalassociations/uefarankings/?year=2026
- UEFA club coefficient calculation and 2026/27 reference window:
  https://www.uefa.com/news-media/news/0252-0cda38714c0d-0874ab234eb6-1000--how-the-club-coefficients-are-calculated/
- Five-season association coefficient table:
  https://www.fworks.jp/eurofoot/ucl/2026-2027/ranking-country.html
- Five-season club coefficient table:
  https://www.fworks.jp/eurofoot/ucl/2026-2027/ranking-club.html
- Arsenal final league status:
  https://www.premierleague.com/en/tables/premier-league/2025-26
- Como final league status:
  https://www.legaseriea.it/serie-a/news/numeri-e-statistiche-della-serie-a-enilive-2025-2026
- Turkish final league context:
  https://www.aa.com.tr/tr/spor/trendyol-super-ligde-2025-2026-sezonu-sona-erdi/3940393
- Portugal final league context:
  https://www.slbenfica.pt/en-us/futebol/classificacao?tournamentId=sr%3Atournament%3A238
- Shakhtar final league status:
  https://upl.ua/en/tournaments/championship/428
- Slavia Praha final league status:
  https://statistiky1ligy.fotbal.cz/sezona/2025-2026/tabulky
- AZ Alkmaar final league status:
  https://www.rsssf.org/tablesn/ned2026.html
- Omonia championship confirmation:
  https://www.cfa.com.cy/En/news/53319
- Pafos cup win:
  https://www.cna.org.cy/en/article/10377404/pafos-fc-wins-cyprus-cup-with-a-2-0-victory-against-apollon
- Pafos final league position control:
  https://www.espn.com/soccer/table?league=cyp.1
- AZ Alkmaar cup result control:
  https://en.wikipedia.org/wiki/2026_KNVB_Cup_final

## Transformations

- `club_points_*` stores each club's own season coefficient points. It does not
  spread the five-year total across seasons and does not substitute the country
  floor for club performance.
- `official_club_coefficient` is an audit field. Como has zero own club points
  in the window but a non-zero official coefficient from Italy's country floor;
  the model correctly uses the explicit zero season history instead.
- `country_part` is `20%` of the association's five-season total and is audit
  metadata only.
- `matches_*` is model-ready capped match evidence: `min(verified UEFA matches,
  match_cap)`. The model cannot distinguish values above the cap, so this keeps
  the exposure input reproducible without pretending the field is a raw match
  archive. Zero and below-cap cases are retained explicitly.
- Match caps follow the current model contract: `6, 6, 6, 8, 8`.
- Club point values displayed by the source at two decimals were retained at
  known quarter-point precision where needed (`14.375`, `6.875`, and similar).

## Known Limitations

- Active v1.1 country values remain `Country_Strength_Benchmark=25`,
  `gamma=0.8`, and `domestic_league_component=140`. The `20 / 2.0 / 360`
  candidate was rejected after failing the zero-exposure plausibility guardrail.
  `European_History_Benchmark=20` remains provisional.
- Competition and entry-round fields are metadata and do not change the rating.
- `ranking_guardrails.csv` contains explicit domain-order constraints used as hard
  promotion vetos; aggregate prediction metrics cannot override these constraints.
- AO European Elo v1.1 applies `cup_double_bonus` only when a club is both the
  league champion and the cup winner. Cup-only winners retain `cup_base_score`
  without a double bonus.
- This ten-team snapshot is not suitable for calibrating the season/match
  exposure blend. Most played seasons are already at `match_cap`, so the two
  exposure components carry little independent variation. Historical raw match
  counts, including below-cap observations, are required for that backtest.
