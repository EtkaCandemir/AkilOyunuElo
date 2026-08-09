from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ao_elo.evaluation import (  # noqa: E402
    dependency_robust_loss_difference_ci,
    schedule_adjusted_team_performance,
)
from scripts.run_controlled_goal_progression_backtest import (  # noqa: E402
    evaluate_predictions,
)
from scripts.run_xg_goal_ablation_backtest import (  # noqa: E402
    AblationCandidate,
    EVENTS_PATH,
    HOLDOUT_SEASON,
    PRODUCTION_MODEL_PATH,
    STATIC_DATA_ROOT,
    STATIC_MANIFEST_PATH,
    XG_MANIFEST_PATH,
    XG_PATH,
    candidate_metric_row,
    evaluate_sequence,
    load_initial_ratings,
    read_events,
    read_production_contract,
    read_static_config,
    read_xg,
    same_season_ranking,
)


OUTPUT_ROOT = (
    ROOT / "output" / "xg_direction_preserving_backtest_2018_2026"
)
RHO_GRID = (
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
)
XG_SCALE_GRID = (0.75, 1.00, 1.25, 1.50)
RANK_TOLERANCE = 1e-9


def candidate_grid() -> tuple[AblationCandidate, ...]:
    candidates = [AblationCandidate("GD", True, 0.0, 1.0)]
    candidates.extend(
        AblationCandidate(
            "GD_XG",
            True,
            rho,
            xg_scale,
            "DIRECTION_PRESERVING_LUCK_CORRECTION",
        )
        for rho in RHO_GRID
        for xg_scale in XG_SCALE_GRID
    )
    result = tuple(candidates)
    for candidate in result:
        candidate.validate()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Backtest direction-preserving xG luck correction over the "
            "frozen production goal-difference model"
        )
    )
    parser.add_argument("--static-data-root", type=Path, default=STATIC_DATA_ROOT)
    parser.add_argument("--events", type=Path, default=EVENTS_PATH)
    parser.add_argument("--xg", type=Path, default=XG_PATH)
    parser.add_argument("--xg-manifest", type=Path, default=XG_MANIFEST_PATH)
    parser.add_argument(
        "--static-manifest",
        type=Path,
        default=STATIC_MANIFEST_PATH,
    )
    parser.add_argument(
        "--production-model",
        type=Path,
        default=PRODUCTION_MODEL_PATH,
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples must be at least 100")

    static_config = read_static_config(args.static_manifest.resolve())
    production = read_production_contract(args.production_model.resolve())
    manifest = json.loads(
        args.xg_manifest.resolve().read_text(encoding="utf-8")
    )
    events = read_events(args.events.resolve())
    xg = read_xg(args.xg.resolve(), events)
    initial_ratings = load_initial_ratings(
        args.static_data_root.resolve(),
        events,
        static_config,
    )
    target = schedule_adjusted_team_performance(events)
    development_seasons = set(events["season"]) - {HOLDOUT_SEASON}
    holdout_seasons = {HOLDOUT_SEASON}
    candidates = candidate_grid()

    print(
        f"xG luck correction: {len(candidates)} candidates, "
        f"{len(xg)} eligible xG matches",
        flush=True,
    )
    evaluations = {
        candidate.key: evaluate_sequence(
            events,
            xg,
            initial_ratings,
            production,
            candidate,
        )
        for candidate in candidates
    }
    development = pd.DataFrame(
        [
            candidate_metric_row(
                candidate,
                evaluations[candidate.key],
                target,
                development_seasons,
                split="DEVELOPMENT",
            )
            for candidate in candidates
        ]
    )
    selected, ranking_safe = select_development_candidate(
        development,
        candidates,
    )
    comparison_candidates = (candidates[0], selected)
    holdout_candidates = pd.DataFrame(
        [
            candidate_metric_row(
                candidate,
                evaluations[candidate.key],
                target,
                holdout_seasons,
                split="HOLDOUT",
            )
            for candidate in candidates
        ]
    )
    holdout = holdout_candidates.loc[
        holdout_candidates["candidate_key"].isin(
            [candidate.key for candidate in comparison_candidates]
        )
    ].copy()
    predictions = pd.concat(
        [
            evaluations[candidate.key].predictions.loc[
                lambda frame: frame["season"].eq(HOLDOUT_SEASON)
            ].assign(
                comparison_model=(
                    "GD_BASELINE"
                    if candidate.model == "GD"
                    else "GD_XG_LUCK"
                ),
                candidate_key=candidate.key,
            )
            for candidate in comparison_candidates
        ],
        ignore_index=True,
    )
    competition = build_competition_summary(predictions)
    uncertainty = build_uncertainty(
        predictions,
        bootstrap_samples=args.bootstrap_samples,
    )
    ranking = pd.concat(
        [
            same_season_ranking(
                evaluations[candidate.key].end_ratings,
                target,
                holdout_seasons,
            ).assign(
                comparison_model=(
                    "GD_BASELINE"
                    if candidate.model == "GD"
                    else "GD_XG_LUCK"
                ),
                candidate_key=candidate.key,
            )
            for candidate in comparison_candidates
        ],
        ignore_index=True,
    )
    decision, guardrails = make_decision(
        holdout,
        competition,
        uncertainty,
        ranking,
        ranking_safe=ranking_safe,
        provider_eligible=bool(manifest["production_eligibility"]),
    )

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    development.to_csv(
        output_root / "development_candidate_metrics.csv",
        index=False,
    )
    holdout_candidates.to_csv(
        output_root / "holdout_candidate_metrics.csv",
        index=False,
    )
    holdout.to_csv(output_root / "holdout_model_comparison.csv", index=False)
    predictions.to_csv(output_root / "holdout_predictions.csv", index=False)
    competition.to_csv(output_root / "competition_summary.csv", index=False)
    uncertainty.to_csv(
        output_root / "dependency_uncertainty.csv",
        index=False,
    )
    ranking.to_csv(output_root / "holdout_ranking.csv", index=False)
    payload = {
        "decision": decision,
        "production_changed": False,
        "formula": (
            "Raw = K * ((S-E)*M_GD + rho*(S_xG-S)); "
            "Delta = clamp_to_result_direction(Raw); "
            "S_xG = logistic((xG_home-xG_away)/xg_scale)"
        ),
        "selected_shadow_candidate": candidate_payload(selected),
        "development_ranking_safe": ranking_safe,
        "provider_production_eligible": bool(
            manifest["production_eligibility"]
        ),
        "guardrails": guardrails,
    }
    (output_root / "selected_luck_correction_model.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "backtest_report.md").write_text(
        build_report(
            decision,
            selected,
            development,
            holdout,
            competition,
            uncertainty,
            ranking,
            ranking_safe,
            manifest,
            guardrails,
        ),
        encoding="utf-8",
    )
    print(f"Decision: {decision}")
    print(f"Selected shadow candidate: {selected.key}")
    print(f"Output: {output_root}")


def select_development_candidate(
    metrics: pd.DataFrame,
    candidates: tuple[AblationCandidate, ...],
) -> tuple[AblationCandidate, bool]:
    baseline = metrics.loc[metrics["candidate_key"].eq("GD")]
    if len(baseline) != 1:
        raise ValueError("Expected one GD development baseline")
    base = baseline.iloc[0]
    eligible = metrics.loc[metrics["model"].eq("GD_XG")].copy()
    eligible["ranking_safe"] = (
        eligible["ranking_score"].ge(
            float(base["ranking_score"]) - RANK_TOLERANCE
        )
        & eligible["pairwise_accuracy"].ge(
            float(base["pairwise_accuracy"]) - RANK_TOLERANCE
        )
    )
    pool = eligible.loc[eligible["ranking_safe"]]
    ranking_safe = not pool.empty
    if pool.empty:
        selected = eligible.sort_values(
            [
                "ranking_score",
                "pairwise_accuracy",
                "brier_1x2",
                "log_loss_1x2",
                "rho",
                "xg_scale",
            ],
            ascending=[False, False, True, True, True, True],
        ).iloc[0]
    else:
        selected = pool.sort_values(
            [
                "brier_1x2",
                "log_loss_1x2",
                "ranking_score",
                "pairwise_accuracy",
                "rho",
                "xg_scale",
            ],
            ascending=[True, True, False, False, True, True],
        ).iloc[0]
    key = str(selected["candidate_key"])
    matches = [candidate for candidate in candidates if candidate.key == key]
    if len(matches) != 1:
        raise ValueError(f"Expected one candidate for {key}")
    return matches[0], ranking_safe


def build_competition_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, competition), group in predictions.groupby(
        ["comparison_model", "competition"],
        sort=True,
    ):
        metrics = evaluate_predictions(group)
        rows.append(
            {
                "comparison_model": model,
                "competition": competition,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def build_uncertainty(
    predictions: pd.DataFrame,
    *,
    bootstrap_samples: int,
) -> pd.DataFrame:
    baseline = predictions.loc[
        predictions["comparison_model"].eq("GD_BASELINE")
    ].set_index("match_id")
    candidate = predictions.loc[
        predictions["comparison_model"].eq("GD_XG_LUCK")
    ].set_index("match_id")
    common = baseline.index.intersection(candidate.index)
    if len(common) != len(baseline) or len(common) != len(candidate):
        raise ValueError("Luck-correction uncertainty sample must be paired")
    rows: list[pd.DataFrame] = []
    for loss in ("brier_1x2", "log_loss_1x2"):
        paired = candidate.loc[
            common,
            [
                "season",
                "home_team_id",
                "away_team_id",
                "kickoff_utc",
                "tie_id",
            ],
        ].reset_index()
        paired["loss_difference"] = (
            candidate.loc[common, loss].to_numpy(float)
            - baseline.loc[common, loss].to_numpy(float)
        )
        result = dependency_robust_loss_difference_ci(
            paired,
            bootstrap_samples=bootstrap_samples,
        )
        result["candidate_model"] = "GD_XG_LUCK"
        result["baseline_model"] = "GD_BASELINE"
        result["loss"] = loss
        rows.append(result)
    return pd.concat(rows, ignore_index=True)


def make_decision(
    holdout: pd.DataFrame,
    competition: pd.DataFrame,
    uncertainty: pd.DataFrame,
    ranking: pd.DataFrame,
    *,
    ranking_safe: bool,
    provider_eligible: bool,
) -> tuple[str, dict[str, object]]:
    indexed = holdout.set_index("model")
    baseline = indexed.loc["GD"]
    candidate = indexed.loc["GD_XG"]
    pooled = ranking.loc[ranking["competition"].eq("ALL")].set_index(
        "comparison_model"
    )
    ci = uncertainty.loc[
        uncertainty["method"].eq("conservative_envelope")
        & uncertainty["loss"].eq("brier_1x2")
    ]
    candidate_segments = competition.loc[
        competition["comparison_model"].eq("GD_XG_LUCK")
    ].set_index("competition")
    baseline_segments = competition.loc[
        competition["comparison_model"].eq("GD_BASELINE")
    ].set_index("competition")
    common = candidate_segments.index.intersection(baseline_segments.index)
    no_segment_harm = bool(
        len(common) == 3
        and (
            candidate_segments.loc[common, "brier_1x2"]
            <= baseline_segments.loc[common, "brier_1x2"]
        ).all()
        and (
            candidate_segments.loc[common, "log_loss_1x2"]
            <= baseline_segments.loc[common, "log_loss_1x2"]
        ).all()
    )
    guardrails = {
        "holdout_brier_delta": (
            float(candidate["brier_1x2"]) - float(baseline["brier_1x2"])
        ),
        "holdout_log_loss_delta": (
            float(candidate["log_loss_1x2"])
            - float(baseline["log_loss_1x2"])
        ),
        "holdout_ranking_delta": (
            float(pooled.loc["GD_XG_LUCK", "ranking_score"])
            - float(pooled.loc["GD_BASELINE", "ranking_score"])
        ),
        "holdout_pairwise_delta": (
            float(pooled.loc["GD_XG_LUCK", "pairwise_accuracy"])
            - float(pooled.loc["GD_BASELINE", "pairwise_accuracy"])
        ),
        "development_ranking_safe": ranking_safe,
        "clustered_brier_reliable_improvement": bool(
            len(ci) == 1 and float(ci.iloc[0]["ci_95_upper"]) < 0.0
        ),
        "no_competition_loss_regression": no_segment_harm,
        "provider_production_eligible": provider_eligible,
        "zero_sum_preserved": bool(
            float(candidate["maximum_pair_sum_error"]) <= 1e-9
            and float(candidate["maximum_total_elo_error"]) <= 1e-9
        ),
    }
    predictive_gain = (
        guardrails["holdout_brier_delta"] < 0.0
        and guardrails["holdout_log_loss_delta"] < 0.0
    )
    rank_guardrail = (
        guardrails["holdout_ranking_delta"] >= 0.0
        and guardrails["holdout_pairwise_delta"] >= 0.0
    )
    if not ranking_safe:
        return "NO_XG_DIRECTION_PRESERVING_PROMOTION", guardrails
    if (
        predictive_gain
        and rank_guardrail
        and no_segment_harm
        and provider_eligible
    ):
        return "PROMOTE_XG_DIRECTION_PRESERVING", guardrails
    if predictive_gain:
        return "KEEP_XG_DIRECTION_PRESERVING_SHADOW", guardrails
    return "NO_XG_DIRECTION_PRESERVING_PROMOTION", guardrails


def candidate_payload(candidate: AblationCandidate) -> dict[str, object]:
    return {
        "candidate_key": candidate.key,
        "rho": candidate.rho,
        "xg_scale": candidate.xg_scale,
        "xg_mode": candidate.xg_mode,
    }


def build_report(
    decision: str,
    selected: AblationCandidate,
    development: pd.DataFrame,
    holdout: pd.DataFrame,
    competition: pd.DataFrame,
    uncertainty: pd.DataFrame,
    ranking: pd.DataFrame,
    ranking_safe: bool,
    manifest: dict[str, object],
    guardrails: dict[str, object],
) -> str:
    selected_development = development.loc[
        development["candidate_key"].isin(["GD", selected.key])
    ]
    pooled_ranking = ranking.loc[ranking["competition"].eq("ALL")]
    return f"""# AO xG Sans Duzeltmesi Backtesti

## Model Karari

**{decision}**

Production modeli degistirilmemistir. Bu calisma, aktif gol farki cekirdeginin
uzerine xG'yi ayri bir sans duzeltmesi olarak test eder.

```text
Raw Delta = K x [(S-E) x M_GD + rho x (S_xG-S)]
Final Delta = Raw Delta, sonuc sinyalinin yonunu koruyacak sekilde clamp edilir
S_xG = 1 / (1 + exp(-(xG_home-xG_away)/c_xG))
```

## Veri

- Uygun xG maci: {manifest["rows"]["eligible_xg_matches"]}
- Temporal holdout: {HOLDOUT_SEASON}
- Provider production uygunlugu: {manifest["production_eligibility"]}
- Metrik: {manifest["contract"]["xg_type"]}

## Development Secimi

Secilen aday: `{selected.key}`

- `rho={selected.rho}`
- `c_xG={selected.xg_scale}`
- Development ranking guardrail gecildi: `{ranking_safe}`

```csv
{selected_development.to_csv(index=False).strip()}
```

## Holdout Karsilastirmasi

```csv
{holdout.to_csv(index=False).strip()}
```

## Turnuva Segmentleri

```csv
{competition.to_csv(index=False).strip()}
```

## Ranking

```csv
{pooled_ranking.to_csv(index=False).strip()}
```

## Clustered Belirsizlik

```csv
{uncertainty.to_csv(index=False).strip()}
```

## Guardrail

```json
{json.dumps(guardrails, indent=2)}
```

## Yorum

Bu veri kesintili ve kaba zone-derived xG oldugu icin, olumlu sonuc production
katsayisini kesinlestirmez. Penaltillar dahil toplam xG tanimi belgelenmis,
tek-saglayici ve kesintisiz veriyle ayni test tekrar edilmelidir.
"""


if __name__ == "__main__":
    main()
