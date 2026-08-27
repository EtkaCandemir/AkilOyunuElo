from __future__ import annotations

"""Generalized domestic cup contribution for the Domestic Achievement Score.

The active model combines the league finish and the domestic cup with a
maximum:

```text
Achievement = min(cap, max(L, C) + 0.08 * L   if champion and cup winner)
```

`max` makes the cup a *floor* rather than a contribution. A cup winner whose
league finish already scores above the cup base receives nothing for the
trophy, and the double bonus fires only for the champion-and-cup pair. This
module implements the one-parameter generalization that treats the weaker of
the two achievements as a genuine contribution:

```text
Achievement = min(cap, max(L, C) + weight * min(L, C))
```

Properties that make the family safe to test against production:

- A team that did not win the cup has `min(L, C) = 0`, so its score is
  identical to the active model. The layer is inert outside cup winners.
- `weight = 0` removes the champion-and-cup double bonus entirely and is the
  lower anchor of the family, not the active model.
- `weight = cup_double_bonus_multiplier / cup_base_score` reproduces the
  active champion-and-cup magnitude while extending the same logic to every
  other league-and-cup combination.

The combination rule itself now lives in `features.py` because it is part of
the active production pipeline. This module re-exports it and keeps the
backtest-only helpers, so the research scripts and tests here are unchanged.
"""

import math
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

from ao_elo.config import AOEuropeanEloConfig
from ao_elo.features import (
    CupContributionConfig,
    DomesticAchievement,
    champion_equivalent_weight,
    compute_domestic_achievement,
    generalized_domestic_achievement,
)
from ao_elo.pipeline import compute_ao_first_elo_from_csv


PRODUCTION_FAMILY = "MAX_WITH_CHAMPION_ONLY_DOUBLE_BONUS"
GENERALIZED_FAMILY = "MAX_PLUS_WEIGHTED_MINIMUM"


def achievement_delta_to_ao_first_elo(
    achievement_delta: float,
    league_strength: float,
    effective_european_exposure: float,
    config: AOEuropeanEloConfig,
) -> float:
    """Convert an achievement-score change into an AO First Elo change.

    Only the Domestic Prior depends on the achievement score, and the blend
    toward the European Prior is linear, so the transfer is exact:

    ```text
    DomesticPrior = base + C_L * L + C_A * A * scale
    AOFirst_pre   = (1 - e_eff) * DomesticPrior + e_eff * EuropeanPrior
    dAOFirst_pre  = (1 - e_eff) * C_A * scale * dA
    ```

    The Domestic Surprise adjustment is unaffected because its own scale
    depends on league strength, not on the achievement score.
    """

    for name, value in (
        ("achievement_delta", achievement_delta),
        ("league_strength", league_strength),
        ("effective_european_exposure", effective_european_exposure),
    ):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    exposure = float(effective_european_exposure)
    if not 0.0 <= exposure <= 1.0:
        raise ValueError("effective_european_exposure must be in [0,1]")
    achievement_scale = float(config.achievement_alpha) + (
        (1.0 - float(config.achievement_alpha)) * float(league_strength)
    )
    return (
        (1.0 - exposure)
        * float(config.domestic_achievement_component)
        * achievement_scale
        * float(achievement_delta)
    )


def candidate_weights() -> tuple[float, ...]:
    """Prespecified search grid, frozen before any test season is scored."""

    return (0.0, 0.05, 0.08, 0.10, 0.13, 0.15, 0.20, 0.25, 0.30)


SEASON_DIRECTORY = re.compile(r"^\d{4}-\d{2}$")


def load_static_achievement_inputs(
    static_root: str | Path,
    config: AOEuropeanEloConfig,
    seasons: Sequence[str],
) -> tuple[tuple[str, pd.DataFrame], ...]:
    """Run the production static pipeline once per requested season.

    The returned frame carries the pipeline components needed to re-price the
    achievement score plus the two domestic result flags, which the pipeline
    output does not itself expose.
    """

    wanted = set(seasons)
    results: list[tuple[str, pd.DataFrame]] = []
    for directory in sorted(Path(static_root).iterdir()):
        if not directory.is_dir() or not SEASON_DIRECTORY.match(directory.name):
            continue
        season = f"{directory.name[:4]}/{directory.name[5:]}"
        if season not in wanted:
            continue
        frame = compute_ao_first_elo_from_csv(
            directory / "teams.csv",
            directory / "country_coefficients.csv",
            directory / "domestic_context.csv",
            directory / "club_european_points.csv",
            config,
        )
        context = pd.read_csv(directory / "domestic_context.csv")
        # The pipeline frame already carries position and team count, so only
        # the two result flags are merged; taking more would rename the
        # achievement inputs and silently detach them.
        merged = frame.merge(
            context[["team_id", "is_league_champion", "is_cup_winner"]],
            on="team_id",
            how="left",
            validate="one_to_one",
        )
        for column in ("is_league_champion", "is_cup_winner"):
            merged[column] = merged[column].fillna(False).astype(bool)
        results.append((season, merged))
    missing = wanted - {season for season, _ in results}
    if missing:
        raise ValueError(f"Missing static inputs for seasons: {sorted(missing)}")
    return tuple(results)


def cup_variant_seed_map(
    statics: Iterable[tuple[str, pd.DataFrame]],
    production_seeds: Mapping[tuple[str, int], float],
    config: AOEuropeanEloConfig,
    weight: float,
) -> dict[tuple[str, int], float]:
    """Return production seeds re-priced under one cup-contribution weight.

    The Domestic Surprise adjustment already inside `production_seeds` is
    carried through untouched, because its own scale depends on league
    strength rather than on the achievement score.
    """

    cup_config = CupContributionConfig(weight)
    cup_config.validate()
    seeds: dict[tuple[str, int], float] = {}
    for season, frame in statics:
        for row in frame.itertuples(index=False):
            key = (season, int(row.team_id))
            if key not in production_seeds:
                continue
            candidate = generalized_domestic_achievement(
                row.domestic_position,
                row.league_team_count,
                row.is_league_champion,
                row.is_cup_winner,
                config,
                cup_config,
            )
            delta = achievement_delta_to_ao_first_elo(
                float(candidate.domestic_achievement_score)
                - float(row.domestic_achievement_score),
                float(row.league_strength),
                float(row.effective_european_exposure),
                config,
            )
            seeds[key] = float(production_seeds[key]) + delta
    missing = set(production_seeds) - set(seeds)
    if missing:
        raise ValueError(f"Cup variant seeds missing {len(missing)} team-seasons")
    return seeds
