from __future__ import annotations

"""Walk-forward test of a season-relative country-strength benchmark.

Production normalizes league strength against a fixed constant:

    L = (ln(1 + weighted_country_score) / ln(1 + 25)) ** gamma      clipped at 1

The 25 was set above the strongest league of its day, and for eight seasons no
country ever reached it - the clip is dead code from 2018/19 to 2025/26. In
2026/27 England's weighted score is 25.35 and the clip binds for the first time.
As coefficients inflate, more leagues will pile onto that ceiling, which is the
same collapse the European history norm had before its tail was opened.

The candidate here replaces the constant with the strongest league of each
season:

    L = (ln(1 + weighted_country_score) / ln(1 + max_score_that_season)) ** gamma

The strongest league is 1.0 by construction, so no clip can ever bind and
`country_tail_beta` stops mattering. Within a season this is a uniform rescale
of the normalization - every country is multiplied by the same factor, and the
ordering is untouched - so the change is not about who leads. It is about the
anchor: league strength stops being absolute and becomes relative to the best
league of that year, which is what a coefficient-based measure arguably always
meant.

The rescale is not neutral on the Domestic Prior, because the achievement term
carries its own dependence on L:

    DP_new - DP_old = (c - 1) * L * (league_component + 0.6 * achievement_component * A)

so clubs with a strong league and a strong finish move more than the rest, and
the effect has to be measured rather than reasoned about.

Research only: the active contract is hashed before and after, and nothing in
`contracts/` or `artifacts/` is written.
"""

import argparse
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ao_elo.config import AOEuropeanEloConfig  # noqa: E402
from ao_elo.european_prior_recalibration import ranking_uncertainty_summary  # noqa: E402
from ao_elo.evaluation import schedule_adjusted_team_performance  # noqa: E402
from ao_elo.features import compute_weighted_country_score  # noqa: E402
from ao_elo.pipeline import compute_ao_first_elo_from_csv  # noqa: E402
from scripts.run_ao_first_seed_boost_backtest import aggregate_target  # noqa: E402
from scripts.run_current_model_evaluation import (  # noqa: E402
    EvaluationArm,
    evaluate_arm,
)
from scripts.run_domestic_prior_axis_backtest import (  # noqa: E402
    BASELINE_KEY,
    build_unseen,
    load_domestic_surprise,
    nested_selection,
    season_rows,
    select,
    summarise,
)
from scripts.run_dynamic_core_calibration import expanding_folds  # noqa: E402
from scripts.run_european_prior_recalibration_backtest import (  # noqa: E402
    competition_summary,
    model_summary,
    sha256,
    uncertainty_summary,
)
from scripts.run_opponent_quintile_backtest import load_production_baseline  # noqa: E402
from scripts.run_stage_weighted_progression_backtest import (  # noqa: E402
    EVENTS_PATH,
    PRODUCTION_CONTRACT,
    STATIC_DATA_ROOT,
    XG_DATA,
    load_xg_map,
)
from scripts.run_v2_evaluation_upgrade import read_events  # noqa: E402


OUTPUT_ROOT = ROOT / "output" / "country_anchor_high_gamma_2018_2026"
ACTIVE = AOEuropeanEloConfig.active()
# The relative anchor lifts every normalization, and gamma acts on the lifted
# value, so the two cannot be judged apart: the sweep pairs the anchor with the
# same gamma range the fixed-benchmark study used.
GAMMAS = (2.0, 2.6, 3.2)
# The anchor is what the whole scale hangs on, so a single-point choice is a
# liability: one league having a freak season would move every other league's
# normalization. These three say how much that matters. TOP3 and P95 both stay
# at or below the maximum, so the clip still cannot bind.
ANCHORS = ("max",)


@dataclass(frozen=True)
class Candidate:
    key: str
    axis: str
    value: float
    config: AOEuropeanEloConfig
    relative: bool
    anchor: str = "max"


def candidate_grid() -> tuple[Candidate, ...]:
    return (
        Candidate(BASELINE_KEY, "BASELINE", 0.0, ACTIVE, False),
        *(
            Candidate(
                f"{anchor}_g{gamma:g}",
                "RELATIVE_BENCHMARK",
                gamma,
                replace(ACTIVE, gamma=gamma),
                True,
                anchor,
            )
            for anchor in ANCHORS
            for gamma in GAMMAS
        ),
    )


def season_benchmark(
    folder: Path, config: AOEuropeanEloConfig, anchor: str = "max"
) -> float:
    """The season's reference league strength, under one of three anchors.

    `max` is the literal strongest league and is the most exposed to a single
    freak season. `top3` averages the three strongest and `p95` takes the 95th
    percentile, both of which move less when one league swings. All three sit at
    or below the maximum, so a club can still reach or pass 1.0 under the
    steadier anchors - the caller checks for that rather than assuming it.
    """
    coefficients = pd.read_csv(folder / "country_coefficients.csv")
    scores = coefficients.apply(
        lambda row: compute_weighted_country_score(row, config), axis=1
    )
    if anchor == "max":
        value = float(scores.max())
    elif anchor == "top3":
        value = float(scores.nlargest(3).mean())
    elif anchor == "p95":
        value = float(scores.quantile(0.95))
    else:
        raise ValueError(f"Unknown anchor: {anchor}")
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{folder.name}: no positive country score to anchor on")
    return value


def relative_seed_frame(
    config: AOEuropeanEloConfig,
    surprise: pd.DataFrame,
    relative: bool,
    anchor: str = "max",
) -> pd.DataFrame:
    """Recompute AO First per season, optionally re-anchoring the benchmark.

    The benchmark has to be resolved inside the season loop: the strongest
    league changes from year to year, so a single config cannot express it.
    """
    rows = []
    anchors = []
    for folder in sorted(STATIC_DATA_ROOT.glob("20??-??")):
        year = int(folder.name[:4])
        season = f"{year}/{str(year + 1)[-2:]}"
        seasonal = config
        if relative:
            value = season_benchmark(folder, config, anchor)
            seasonal = replace(config, country_strength_benchmark=value)
            anchors.append({"season": season, "anchor": anchor, "benchmark": value})
        rating = compute_ao_first_elo_from_csv(
            folder / "teams.csv",
            folder / "country_coefficients.csv",
            folder / "domestic_context.csv",
            folder / "club_european_points.csv",
            seasonal,
        )
        if relative and anchor == "max":
            # Only the maximum guarantees nothing exceeds the anchor. The
            # steadier anchors sit below it on purpose, so clipping there is
            # expected and is measured rather than forbidden.
            saturated = int((rating["country_strength_uncapped_norm"] > 1.0 + 1e-9).sum())
            if saturated:
                raise ValueError(
                    f"{season}: {saturated} clubs exceed a max-derived benchmark"
                )
        rows.append(
            rating[
                [
                    "team_id",
                    "team_name",
                    "domestic_prior",
                    "european_prior",
                    "effective_european_exposure",
                ]
            ].assign(season=season)
        )
    frame = pd.concat(rows, ignore_index=True).merge(
        surprise, on=["season", "team_id"], validate="one_to_one"
    )
    adjusted_domestic = frame["domestic_prior"] + frame["domestic_prior_adjustment"]
    frame["ao_first_elo"] = adjusted_domestic + frame[
        "effective_european_exposure"
    ] * (frame["european_prior"] - adjusted_domestic)
    frame.attrs["anchors"] = anchors
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure a season-relative country-strength benchmark"
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract_hash = sha256(PRODUCTION_CONTRACT)

    print("Loading frozen production replay", flush=True)
    _, datasets, core, parameters, production_seed_map = load_production_baseline()
    seasons = tuple(data.season for data in datasets)
    folds = expanding_folds(seasons)
    target_by_competition = schedule_adjusted_team_performance(read_events(EVENTS_PATH))
    target = aggregate_target(target_by_competition)
    xg_map = load_xg_map(XG_DATA, datasets)
    surprise = load_domestic_surprise()

    configs = candidate_grid()
    predictions: dict[str, pd.DataFrame] = {}
    candidate_seeds: dict[str, pd.DataFrame] = {}
    surface_rows: list[dict[str, object]] = []
    anchors: list[dict[str, object]] = []

    for index, candidate in enumerate(configs, start=1):
        seeds = relative_seed_frame(
            candidate.config, surprise, candidate.relative, candidate.anchor
        )
        if candidate.relative:
            anchors.extend(seeds.attrs["anchors"])
        rating_map = {
            (str(row.season), int(row.team_id)): float(row.ao_first_elo)
            for row in seeds.itertuples(index=False)
        }
        if candidate.key == BASELINE_KEY:
            error = max(
                abs(rating_map[key] - value)
                for key, value in production_seed_map.items()
            )
            if error > 1e-8:
                raise ValueError(f"Baseline does not reproduce production: {error}")
        evaluation = evaluate_arm(
            datasets,
            EvaluationArm(candidate.key, True, True, True, True, True),
            core=core,
            parameters=parameters,
            current_domestic=rating_map,
            baseline_domestic=production_seed_map,
            xg_map=xg_map,
            target=target_by_competition,
        )
        predictions[candidate.key] = evaluation.predictions
        candidate_seeds[candidate.key] = seeds.rename(
            columns={"ao_first_elo": "candidate_ao_first_elo"}
        )
        surface_rows.extend(
            season_rows(candidate, seeds, evaluation.predictions, target, seasons)
        )
        print(f"Replayed {index}/{len(configs)} candidates", flush=True)

    surface = pd.DataFrame(surface_rows)
    summary_frame = summarise(surface)
    selections = nested_selection(surface, folds)
    unseen, fold_results = build_unseen(
        selections, predictions, candidate_seeds, target, folds
    )
    comparison = model_summary(unseen, fold_results)
    competition = competition_summary(unseen)
    uncertainty = uncertainty_summary(unseen, int(args.bootstrap_samples))
    ranking_uncertainty = ranking_uncertainty_summary(
        fold_results, int(args.bootstrap_samples)
    )
    full_selected = select(summary_frame)

    selected = comparison.loc[comparison["model"].eq("NESTED_RECALIBRATION")].iloc[0]
    brier_wins = int((fold_results["delta_brier_1x2"] < 0.0).sum())
    log_wins = int((fold_results["delta_log_loss_1x2"] < 0.0).sum())
    chosen = selections["selected_candidate_key"].tolist()
    payload = {
        "decision": (
            "PROMOTE_CANDIDATE"
            if brier_wins >= 4
            and log_wins >= 4
            and selected["delta_vs_baseline_brier_1x2"] < 0.0
            and selected["delta_vs_baseline_log_loss_1x2"] < 0.0
            and not bool(ranking_uncertainty["reliable_harm"].any())
            and not bool(
                (competition["delta_brier_1x2"] > 0.0).any()
                or (competition["delta_log_loss_1x2"] > 0.0).any()
            )
            else "KEEP_PRODUCTION"
        ),
        "production_changed": False,
        "candidate_count": len(configs),
        "fold_count": len(folds),
        "unseen_matches": int(comparison["matches"].max()),
        "full_history_candidate_key": full_selected,
        "nested_selected_keys": chosen,
        "same_candidate_every_fold": len(set(chosen)) == 1,
        "brier_fold_wins": brier_wins,
        "log_loss_fold_wins": log_wins,
        "ranking_reliable_harm": bool(ranking_uncertainty["reliable_harm"].any()),
        "pooled_brier_delta": float(selected["delta_vs_baseline_brier_1x2"]),
        "pooled_seed_spearman_delta": float(
            selected["delta_vs_baseline_seed_spearman"]
        ),
        "season_anchors": anchors,
        "production_contract_sha256": contract_hash,
    }

    unseen.to_csv(output / "unseen_predictions.csv", index=False)
    surface.to_csv(output / "candidate_surface.csv", index=False)
    summary_frame.to_csv(output / "candidate_summary.csv", index=False)
    selections.to_csv(output / "fold_selections.csv", index=False)
    fold_results.to_csv(output / "fold_results.csv", index=False)
    comparison.to_csv(output / "model_comparison.csv", index=False)
    competition.to_csv(output / "competition_summary.csv", index=False)
    uncertainty.to_csv(output / "dependency_uncertainty.csv", index=False)
    ranking_uncertainty.to_csv(output / "ranking_uncertainty.csv", index=False)
    (output / "decision.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    if sha256(PRODUCTION_CONTRACT) != contract_hash:
        raise ValueError("Production contract changed during the sweep")
    print(f"Decision: {payload['decision']}")
    print(f"Full-history candidate: {full_selected}")


if __name__ == "__main__":
    main()
