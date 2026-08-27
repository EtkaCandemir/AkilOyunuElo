from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ao_elo.domestic_poisson import (  # noqa: E402
    DomesticPoissonConfig,
    domestic_candidate_grid,
    evaluate_domestic_candidates,
)
from ao_elo.domestic_poisson_backtest import (  # noqa: E402
    AO_ML_POISSON_BLEND,
    AO_POISSON_BLEND,
    CURRENT_AO,
    run_domestic_poisson_walk_forward_backtest,
)


DATA_ROOT = ROOT / "data" / "domestic_league_expansion_ucl_uel"
BASELINE_ROOT = ROOT / "output" / "domestic_poisson_backtest_2018_2026"
FEATURE_STORE = ROOT / "output" / "ml_1x2_backtest_2018_2026" / "pre_match_feature_store.csv"
CURRENT_ML = ROOT / "output" / "ml_1x2_backtest_2018_2026" / "unseen_predictions.csv"
CONTRACT = ROOT / "contracts" / "ao_european_elo_v2_production.json"
REPORT_ROOT = ROOT / "reports" / "domestic_poisson_ucl_uel_expansion"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the UCL/UEL domestic-Poisson expansion without production activation"
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
    parser.add_argument("--feature-store", type=Path, default=FEATURE_STORE)
    parser.add_argument("--current-ml-predictions", type=Path, default=CURRENT_ML)
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--reuse-surface", action="store_true")
    parser.add_argument(
        "--full-grid",
        action="store_true",
        help="Research-only: recompute all 54 domestic state candidates instead of the frozen production config.",
    )
    args = parser.parse_args()

    output = args.report_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract = args.contract.resolve()
    contract_before = _sha256(contract)
    domestic = pd.read_csv(args.data_root.resolve() / "domestic_matches_candidate.csv", low_memory=False)
    bridge = pd.read_csv(args.data_root.resolve() / "domestic_team_bridge_candidate.csv", low_memory=False)
    features = pd.read_csv(args.feature_store.resolve(), low_memory=False)
    current_ml = pd.read_csv(args.current_ml_predictions.resolve(), low_memory=False)

    surface_path = output / "domestic_prequential_results.csv"
    if args.reuse_surface and surface_path.is_file():
        surface = pd.read_csv(surface_path, low_memory=False)
    elif args.full_grid:
        print("Evaluating research 54-candidate domestic prequential surface", flush=True)
        surface = evaluate_domestic_candidates(domestic)
        surface.to_csv(surface_path, index=False)
    else:
        print("Evaluating frozen production domestic config prequentially", flush=True)
        fixed = DomesticPoissonConfig(0.02, 0.90, 10.0, False)
        fixed_surface = evaluate_domestic_candidates(
            domestic,
            candidates=(fixed,),
            require_full_grid=False,
        )
        surface = _frozen_production_surface(fixed_surface, fixed)
        surface.to_csv(surface_path, index=False)
    print("Running six-fold nested UCL/UEL expansion evaluation", flush=True)
    result = run_domestic_poisson_walk_forward_backtest(
        features,
        domestic,
        bridge,
        surface,
        current_ml,
        bootstrap_samples=args.bootstrap_samples,
    )
    if _sha256(contract) != contract_before:
        raise ValueError("Production contract changed during candidate evaluation")

    _write_result_files(output, result)
    coverage = _coverage_before_after(args.data_root.resolve())
    coverage.to_csv(output / "coverage_before_after.csv", index=False)
    profile = _team_profile_audit(args.data_root.resolve(), bridge, domestic)
    profile.to_csv(output / "team_profile_audit.csv", index=False)
    comparison = _prediction_comparison(args.baseline_root.resolve(), result.unseen_predictions)
    comparison.to_csv(output / "prediction_comparison.csv", index=False)
    report = _build_report(result, coverage, profile, comparison, contract_before)
    (output / "evaluation_report.md").write_text(report, encoding="utf-8")
    selected = dict(result.selected_candidate)
    selected.update(
        {
            "status": "CANDIDATE_EVALUATION_ONLY",
            "production_contract_sha256": contract_before,
            "candidate_domestic_matches": int(len(domestic)),
            "candidate_mapped_clubs": int(bridge["ao_club_id"].notna().sum()),
            "target_coverage": {
                "covered": int(coverage["candidate_state_eligible"].sum()),
                "total": int(len(coverage)),
            },
        }
    )
    (output / "selected_candidate.json").write_text(
        json.dumps(selected, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Decision: {selected['decision']}")
    print(f"Output: {output}")


def _write_result_files(output: Path, result: object) -> None:
    files = {
        "fold_domestic_selections.csv": result.fold_domestic_selections,
        "domestic_poisson_feature_store.csv": result.domestic_poisson_feature_store,
        "fold_transfer_parameters.csv": result.fold_transfer_parameters,
        "fold_results.csv": result.fold_results,
        "unseen_predictions.csv": result.unseen_predictions,
        "model_comparison.csv": result.model_comparison,
        "feature_ablation.csv": result.feature_ablation,
        "competition_summary.csv": result.competition_coverage_summary,
        "scoreline_diagnostics.csv": result.scoreline_diagnostics,
        "dependency_uncertainty.csv": result.dependency_uncertainty,
    }
    for filename, frame in files.items():
        frame.to_csv(output / filename, index=False)


def _coverage_before_after(data_root: Path) -> pd.DataFrame:
    return pd.read_csv(data_root / "target_coverage_audit.csv", low_memory=False)


def _frozen_production_surface(
    fixed_surface: pd.DataFrame,
    fixed: DomesticPoissonConfig,
) -> pd.DataFrame:
    """Satisfy the generic selector while pinning its choice to production config."""

    rows = [fixed_surface]
    for config in domestic_candidate_grid():
        if config.key == fixed.key:
            continue
        blocked = fixed_surface.copy()
        blocked["candidate_key"] = config.key
        blocked["goal_nll"] = 1_000_000.0
        blocked["home_goal_bias"] = 1_000_000.0
        blocked["away_goal_bias"] = 1_000_000.0
        rows.append(blocked)
    return pd.concat(rows, ignore_index=True, sort=False)


def _team_profile_audit(data_root: Path, bridge: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    targets = pd.read_csv(data_root / "target_team_audit.csv", low_memory=False)
    mapped = bridge.loc[bridge["ao_club_id"].notna()].copy()
    lookup = {
        (str(row.country_code), str(row.source_team_id)): str(row.ao_club_id)
        for row in mapped.itertuples(index=False)
    }
    rows: list[dict[str, object]] = []
    for target in targets.itertuples(index=False):
        club_id = str(target.ao_club_id)
        played = matches.loc[
            matches.apply(
                lambda row: lookup.get((str(row.country_code), str(row.home_source_team_id))) == club_id
                or lookup.get((str(row.country_code), str(row.away_source_team_id))) == club_id,
                axis=1,
            )
        ]
        rows.append(
            {
                "ao_club_id": club_id,
                "team_name": target.team_name,
                "country_code": target.country_code,
                "domestic_matches": int(len(played)),
                "domestic_seasons": int(played["ao_season"].nunique()),
                "provider_count": int(played.get("source_provider", pd.Series(dtype=str)).nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values(["country_code", "team_name"], kind="stable")


def _prediction_comparison(baseline_root: Path, candidate_unseen: pd.DataFrame) -> pd.DataFrame:
    baseline = pd.read_csv(baseline_root / "unseen_predictions.csv", low_memory=False)
    models = (CURRENT_AO, AO_POISSON_BLEND, AO_ML_POISSON_BLEND)
    left = baseline[baseline["model"].isin(models)].copy()
    right = candidate_unseen[candidate_unseen["model"].isin(models)].copy()
    merged = left.merge(
        right,
        on=["match_id", "model"],
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    rows = []
    for model, group in merged.groupby("model", sort=True):
        rows.append(
            {
                "model": model,
                "matches": int(len(group)),
                "mean_absolute_probability_shift": float(
                    group[["home_probability_baseline", "draw_probability_baseline", "away_probability_baseline"]]
                    .to_numpy()
                    .__sub__(group[["home_probability_candidate", "draw_probability_candidate", "away_probability_candidate"]].to_numpy())
                    .__abs__()
                    .mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _build_report(result: object, coverage: pd.DataFrame, profile: pd.DataFrame, comparison: pd.DataFrame, contract_hash: str) -> str:
    models = result.model_comparison
    focus = models[models["model"].isin((CURRENT_AO, AO_POISSON_BLEND, AO_ML_POISSON_BLEND))]
    folds = result.fold_results[result.fold_results["model"].eq(AO_ML_POISSON_BLEND)]
    return """# UCL/UEL Yerel Lig Poisson Kapsama Aday Değerlendirmesi

## Statü

Bu çalışma **candidate-only** statüsündedir. Production contract, ML artifact ve
aktif domestic Poisson state değiştirilmemiştir. Causal 2026/27 state ayrıca
üretilmiştir; yalnız Avrupa maçı öncesindeki yerel sonuçları içerir.

## Kapsam

""" + f"- Hedef UCL/UEL kulübü: **{len(coverage)}**\n- Candidate state uygun: **{int(coverage['candidate_state_eligible'].sum())}**\n- Açıklanmış fallback: **{int((~coverage['candidate_state_eligible']).sum())}**\n- Candidate domestic fixture: **{int(profile['domestic_matches'].sum())}** hedef-kulüp görünümü\n- Production contract SHA-256: `{contract_hash}`\n\n" + "## Pooled model karşılaştırması\n\n" + _markdown_csv(focus) + "\n\n## Fold sonuçları\n\n" + _markdown_csv(folds) + "\n\n## Olasılık farkı\n\n" + _markdown_csv(comparison) + "\n\n## Not\n\nYeni veri, mevcut 19 ligdeki kaynak satırlarını değiştirmez. Primary/secondary seçimleri league-season seviyesinde tek kaynaktan yapılır; kaynak değişen hedef liglerde state anahtarları yalnız doğrulanmış AO kulüpleri için kanonikleştirilir.\n"


def _markdown_csv(frame: pd.DataFrame) -> str:
    return "```csv\n" + frame.to_csv(index=False).strip() + "\n```"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - persist failure details for long-running candidate jobs
        import traceback

        REPORT_ROOT.mkdir(parents=True, exist_ok=True)
        (REPORT_ROOT / "evaluation_failure.log").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        raise
