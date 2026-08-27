from __future__ import annotations

"""Where should the exposure cap sit once the European Prior is cleaner?

The cap exists because some clubs' European Prior is unreliable - they have
played too little for the five-season points sum to mean much - so it limits
how far the blend may lean on that number. The participation normalization
repairs part of exactly that unreliability: a club that missed seasons no
longer has its prior depressed by the gap.

That makes the cap's tuning a question again. If the European number is now
cleaner for the clubs the cap was protecting, the optimum could move up. Or the
two could be substitutes and it could move down. The sweep that chose `0.65`
ran against the untouched prior, so it cannot answer this.

This run sweeps the same cap grid twice - once on the production prior and once
on the participation-normalized prior - and reports where each curve peaks on
the season-start seed axis. `k` is re-selected inside every fold at every cap,
so the two curves are compared at their own best settings rather than at one
borrowed from the other.

Match loss is replayed only for a shortlist, because each replay is minutes and
the shape of the curve is what the decision needs.

The production contract is read only to hash it. This script activates nothing.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (str(SRC), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from ao_elo.european_participation import (  # noqa: E402
    BASELINE,
    NORMALIZED,
    SHRINKAGE_GRID,
    ParticipationNormalizationConfig,
    apply_participation_normalization,
)
from ao_elo.european_prior_recalibration import (  # noqa: E402
    ranking_uncertainty_summary,
)
from ao_elo.evaluation import schedule_adjusted_team_performance  # noqa: E402
from scripts.run_ao_first_seed_boost_backtest import (  # noqa: E402
    aggregate_target,
    seed_ranking_metrics,
)
from scripts.run_current_model_evaluation import (  # noqa: E402
    EvaluationArm,
    evaluate_arm,
)
from scripts.run_dynamic_core_calibration import expanding_folds  # noqa: E402
from scripts.run_european_participation_backtest import (  # noqa: E402
    EVALUATION_SEASONS,
    EVENTS_PATH,
    XG_DATA,
    load_seed_evidence,
)
from scripts.run_opponent_quintile_backtest import load_production_baseline  # noqa: E402
from scripts.run_stage_weighted_progression_backtest import load_xg_map  # noqa: E402
from scripts.run_v2_evaluation_upgrade import read_events  # noqa: E402


PRODUCTION_CONTRACT = ROOT / "contracts" / "ao_european_elo_v2_production.json"
OUTPUT_ROOT = ROOT / "output" / "participation_exposure_interaction_2018_2026"
# The grid the original exposure-cap study swept, so the two are comparable.
CAP_GRID = (
    0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.675,
    0.70, 0.725, 0.75, 0.775, 0.80, 0.825, 0.85,
)
ACTIVE_CAP = 0.65


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep the European exposure cap on the production prior and on "
            "the participation-normalized prior, and compare the two curves"
        )
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--loss-caps",
        type=float,
        nargs="*",
        default=None,
        help=(
            "Caps to replay for match loss. Defaults to the active cap, each "
            "curve's own peak, and any cap given on the command line."
        ),
    )
    parser.add_argument("--skip-loss-axis", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    arguments = parser.parse_args()
    output = arguments.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract_hash = hashlib.sha256(PRODUCTION_CONTRACT.read_bytes()).hexdigest()

    print("Loading production replay and seed evidence", flush=True)
    _, datasets, core, parameters, production_seed_map = load_production_baseline()
    seasons = tuple(data.season for data in datasets)
    folds = expanding_folds(seasons)
    events = read_events(EVENTS_PATH)
    target_by_competition = schedule_adjusted_team_performance(events)
    target = aggregate_target(target_by_competition)
    seeds, baseline_error = load_seed_evidence()
    print(f"Baseline reproduces production to {baseline_error:.3e} Elo", flush=True)

    print(f"Sweeping {len(CAP_GRID)} caps on both priors", flush=True)
    curve, fold_metrics, audits = sweep(seeds, target, folds)
    peaks = curve_peaks(curve)
    uncertainty = cap_uncertainty(fold_metrics, arguments.bootstrap_samples)
    print()
    print(peaks.to_string(index=False))
    print()
    print(
        uncertainty.loc[uncertainty["metric"].eq("seed_spearman")][
            ["cap", "versus", "mean_difference", "ci_95_lower", "ci_95_upper",
             "reliable_improvement", "reliable_harm"]
        ].to_string(index=False)
    )

    loss = pd.DataFrame()
    if not arguments.skip_loss_axis:
        shortlist = sorted(
            {ACTIVE_CAP, *peaks["cap"].tolist(), *(arguments.loss_caps or [])}
        )
        print(f"\nReplaying match loss for caps {shortlist}", flush=True)
        loss = loss_shortlist(
            audits, shortlist, datasets, core, parameters,
            production_seed_map, target_by_competition, xg_data=XG_DATA,
        )
        print()
        print(loss.to_string(index=False))

    if hashlib.sha256(PRODUCTION_CONTRACT.read_bytes()).hexdigest() != contract_hash:
        raise ValueError("Production contract changed during a research backtest")

    curve.to_csv(output / "exposure_curve.csv", index=False)
    fold_metrics.to_csv(output / "fold_metrics.csv", index=False)
    uncertainty.to_csv(output / "cap_uncertainty.csv", index=False)
    peaks.to_csv(output / "curve_peaks.csv", index=False)
    if not loss.empty:
        loss.to_csv(output / "loss_shortlist.csv", index=False)
    summary = build_summary(curve, peaks, loss, uncertainty, contract_hash)
    (output / "interaction_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "interaction_report.md").write_text(
        build_report(curve, peaks, uncertainty, loss, summary), encoding="utf-8"
    )
    print(f"\nOutput: {output}")


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------


def nested_seeds(
    seeds: pd.DataFrame,
    target: pd.DataFrame,
    folds,
    cap: float,
    arm: str,
) -> tuple[pd.DataFrame, list[float]]:
    """Score the unseen window at one cap, re-selecting `k` inside each fold."""

    chosen: list[float] = []
    blocks = []
    for train_seasons, test_season in folds:
        test = seeds.loc[seeds["season"].eq(test_season)]
        if arm == BASELINE:
            config = ParticipationNormalizationConfig(arm=BASELINE, exposure_cap=cap)
            chosen.append(0.0)
        else:
            train = seeds.loc[seeds["season"].isin(train_seasons)]
            train_target = target.loc[target["season"].isin(train_seasons)]
            scored = []
            for shrinkage in SHRINKAGE_GRID:
                candidate = ParticipationNormalizationConfig(
                    arm=NORMALIZED, shrinkage=shrinkage, exposure_cap=cap
                )
                metrics = seed_ranking_metrics(
                    apply_participation_normalization(train, candidate),
                    train_target,
                    rating_column="candidate_ao_first_elo",
                )
                scored.append({"shrinkage": shrinkage, **metrics})
            table = pd.DataFrame(scored).sort_values(
                ["spearman", "pairwise_accuracy", "shrinkage"],
                ascending=[False, False, True],
            )
            selected = float(table.iloc[0]["shrinkage"])
            chosen.append(selected)
            config = ParticipationNormalizationConfig(
                arm=NORMALIZED, shrinkage=selected, exposure_cap=cap
            )
        blocks.append(apply_participation_normalization(test, config))
    return pd.concat(blocks, ignore_index=True), chosen


def sweep(
    seeds: pd.DataFrame, target: pd.DataFrame, folds
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[float, str], pd.DataFrame]]:
    rows = []
    fold_rows = []
    audits: dict[tuple[float, str], pd.DataFrame] = {}
    for cap in CAP_GRID:
        for arm in (BASELINE, NORMALIZED):
            audit, chosen = nested_seeds(seeds, target, folds, cap, arm)
            audits[(cap, arm)] = audit
            for fold, (_, test_season) in enumerate(folds, start=1):
                season_metrics = seed_ranking_metrics(
                    audit.loc[audit["season"].eq(test_season)],
                    target.loc[target["season"].eq(test_season)],
                    rating_column="candidate_ao_first_elo",
                )
                fold_rows.append(
                    {
                        "cap": cap,
                        "arm": arm,
                        "fold": fold,
                        "test_season": test_season,
                        "seed_spearman": season_metrics["spearman"],
                        "seed_pairwise_accuracy": season_metrics["pairwise_accuracy"],
                    }
                )
            metrics = seed_ranking_metrics(
                audit, target, rating_column="candidate_ao_first_elo"
            )
            rows.append(
                {
                    "cap": cap,
                    "arm": arm,
                    "minimum_domestic_weight": round(1.0 - cap, 4),
                    "teams": metrics["teams"],
                    "seed_spearman": metrics["spearman"],
                    "seed_pairwise_accuracy": metrics["pairwise_accuracy"],
                    "selected_shrinkages": "|".join(f"{value:g}" for value in chosen),
                    "modal_shrinkage": max(set(chosen), key=chosen.count),
                }
            )
        print(f"  cap {cap:<6g} done", flush=True)
    curve = pd.DataFrame(rows)
    wide = curve.pivot(index="cap", columns="arm", values="seed_spearman")
    gain = (wide[NORMALIZED] - wide[BASELINE]).rename("participation_gain")
    return curve.merge(gain, on="cap", how="left"), pd.DataFrame(fold_rows), audits


def cap_uncertainty(fold_metrics: pd.DataFrame, samples: int) -> pd.DataFrame:
    """Season-block bootstrap of every cap against the activated one.

    The curve is a set of point estimates on one six-fold window, so reading a
    peak off it is a preference, not a measurement. This turns each cap into a
    paired difference against the incumbent and asks whether the six folds
    agree, which is the same veto the rest of the repository uses.
    """

    frame = fold_metrics.loc[fold_metrics["arm"].eq(NORMALIZED)]
    incumbent = frame.loc[frame["cap"].eq(ACTIVE_CAP)].set_index("fold")
    blocks = []
    for cap in sorted(frame["cap"].unique()):
        if cap == ACTIVE_CAP:
            continue
        candidate = frame.loc[frame["cap"].eq(cap)].set_index("fold")
        deltas = pd.DataFrame(
            {
                "delta_seed_spearman": candidate["seed_spearman"]
                - incumbent["seed_spearman"],
                "delta_seed_pairwise_accuracy": candidate["seed_pairwise_accuracy"]
                - incumbent["seed_pairwise_accuracy"],
            }
        ).reset_index()
        result = ranking_uncertainty_summary(deltas, samples)
        result.insert(0, "cap", cap)
        result.insert(1, "versus", ACTIVE_CAP)
        blocks.append(result)
    return pd.concat(blocks, ignore_index=True)


def curve_peaks(curve: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm in (BASELINE, NORMALIZED):
        frame = curve.loc[curve["arm"].eq(arm)]
        best = frame.loc[frame["seed_spearman"].idxmax()]
        active = frame.loc[frame["cap"].eq(ACTIVE_CAP)].iloc[0]
        rows.append(
            {
                "arm": arm,
                "cap": float(best["cap"]),
                "peak_seed_spearman": float(best["seed_spearman"]),
                "active_cap_seed_spearman": float(active["seed_spearman"]),
                "peak_minus_active": float(
                    best["seed_spearman"] - active["seed_spearman"]
                ),
                "modal_shrinkage": best["modal_shrinkage"],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# loss shortlist
# ---------------------------------------------------------------------------


def loss_shortlist(
    audits,
    caps,
    datasets,
    core,
    parameters,
    production_seed_map,
    target_by_competition,
    *,
    xg_data,
) -> pd.DataFrame:
    xg_map = load_xg_map(xg_data, datasets)
    rows = []
    for cap in caps:
        for arm in (BASELINE, NORMALIZED):
            audit = audits.get((cap, arm))
            if audit is None:
                continue
            mapping = dict(production_seed_map)
            for row in audit.itertuples(index=False):
                mapping[(str(row.season), int(row.team_id))] = float(
                    row.candidate_ao_first_elo
                )
            if not all(math.isfinite(value) for value in mapping.values()):
                raise ValueError(f"invalid rating map for cap {cap} arm {arm}")
            evaluation = evaluate_arm(
                datasets,
                EvaluationArm(f"{arm}_cap{cap:g}", True, True, True, True, True),
                core=core,
                parameters=parameters,
                current_domestic=mapping,
                baseline_domestic=production_seed_map,
                xg_map=xg_map,
                target=target_by_competition,
            )
            predictions = evaluation.predictions
            rows.append(
                {
                    "cap": cap,
                    "arm": arm,
                    "matches": int(len(predictions)),
                    "brier_1x2": float(predictions["brier_1x2"].mean()),
                    "log_loss_1x2": float(predictions["log_loss_1x2"].mean()),
                }
            )
            print(f"    cap {cap:g} {arm} replayed", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def build_summary(curve, peaks, loss, uncertainty, contract_hash: str) -> dict[str, object]:
    indexed = peaks.set_index("arm")
    moved = float(indexed.loc[NORMALIZED, "cap"]) - float(indexed.loc[BASELINE, "cap"])
    gains = curve.loc[curve["arm"].eq(NORMALIZED), ["cap", "participation_gain"]]
    return {
        "production_change": False,
        "production_contract_sha256": contract_hash,
        "active_cap": ACTIVE_CAP,
        "baseline_peak_cap": float(indexed.loc[BASELINE, "cap"]),
        "participation_peak_cap": float(indexed.loc[NORMALIZED, "cap"]),
        "peak_shift": moved,
        "participation_gain_at_active_cap": float(
            gains.loc[gains["cap"].eq(ACTIVE_CAP), "participation_gain"].iloc[0]
        ),
        "participation_gain_min": float(gains["participation_gain"].min()),
        "participation_gain_max": float(gains["participation_gain"].max()),
        "loss_replayed_caps": sorted(loss["cap"].unique().tolist())
        if not loss.empty
        else [],
        "caps_reliably_better_than_active": sorted(
            uncertainty.loc[
                uncertainty["metric"].eq("seed_spearman")
                & uncertainty["reliable_improvement"],
                "cap",
            ].tolist()
        ),
        "caps_reliably_worse_than_active": sorted(
            uncertainty.loc[
                uncertainty["metric"].eq("seed_spearman")
                & uncertainty["reliable_harm"],
                "cap",
            ].tolist()
        ),
    }


def build_report(curve, peaks, uncertainty, loss, summary) -> str:
    loss_block = (
        "```text\n" + loss.to_string(index=False) + "\n```"
        if not loss.empty
        else "_Loss ekseni bu kosuda atlandi._"
    )
    shift = summary["peak_shift"]
    direction = (
        "yukari" if shift > 0 else "asagi" if shift < 0 else "ayni yerde"
    )
    return f"""# Exposure Cap x Katilim Normalizasyonu Etkilesimi

Production contract degistirilmedi.

## Soru

Cap, bazi kuluplerin Avrupa prior'i guvenilmez oldugu icin vardir: blend'in o
sayiya ne kadar yaslanabilecegini sinirlar. Katilim normalizasyonu tam o
guvenilmezligin bir kismini onarir - sezon kaciran kulubun prior'i artik
bosluk yuzunden bastirilmaz.

Bu, cap'in ayarini yeniden acar. Avrupa sayisi artik daha temizse optimum
yukari kayabilir; yoksa ikisi birbirinin yerine geciyorsa asagi. `0.65`i secen
tarama dokunulmamis prior uzerinde kosmustu ve bu soruyu cevaplayamaz.

## Egri

`k` her cap'te, her fold'un icinde yeniden secilir; iki egri kendi en iyi
ayarinda karsilastirilir.

```text
{curve.to_string(index=False)}
```

## Tepe noktalari

```text
{peaks.to_string(index=False)}
```

Aktif cap `{summary['active_cap']}`. Taban egrisinin tepesi
`{summary['baseline_peak_cap']}`, katilim egrisinin tepesi
`{summary['participation_peak_cap']}` - yani optimum **{direction}**
kaydi (`{shift:+g}`).

## Cap boyutunda belirsizlik

Her cap, aktive edilmis `{summary['active_cap']}`e karsi sezon-blogu bootstrap
ile karsilastirilir. Egriden tepe okumak bir tercihtir; guvenilir fark ise
olcumdur.

```text
{uncertainty.loc[uncertainty['metric'].eq('seed_spearman')][['cap','versus','mean_difference','ci_95_lower','ci_95_upper','reliable_improvement','reliable_harm']].to_string(index=False)}
```

Aktif cap'ten **guvenilir sekilde iyi** olan cap'ler:
`{summary['caps_reliably_better_than_active'] or 'YOK'}`.
Guvenilir sekilde kotu olanlar:
`{summary['caps_reliably_worse_than_active'] or 'YOK'}`.

## Mac loss (kisa liste)

{loss_block}

## Karar girdisi

- Aktif cap'te katilim kazanci: `{summary['participation_gain_at_active_cap']:+.6f}`
- Kazancin cap boyunca araligi:
  `{summary['participation_gain_min']:+.6f}` … `{summary['participation_gain_max']:+.6f}`

Karar urun tarafina aittir; bu belge yalniz kanit uretir.
"""


if __name__ == "__main__":
    main()
