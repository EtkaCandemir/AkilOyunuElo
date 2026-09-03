from __future__ import annotations

"""Post-hoc discovery study of known-entry-path exposure tails.

WARNING: this hypothesis was created after pooled competition-segment results
were inspected.  Neither a fixed result nor the nested diagnostic produced by
this command is promotion evidence.  A positive result can only motivate a
new, prospectively specified validation window.

The historical ``competition`` column is outcome-conditioned, so this command
does not use it as a candidate switch.  It derives the competition a club was
known to be entering from ``entry_round`` (CL/ch=UCL, EL/eu/nc=UEL,
ECL/CO=UECL).  Each candidate applies CAP_TAIL only to one such pre-match entry
path; every other seed row uses the production rating.
"""

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ao_elo.european_prior_recalibration import (  # noqa: E402
    apply_european_prior_recalibration,
    ranking_uncertainty_summary,
)
from ao_elo.evaluation import schedule_adjusted_team_performance  # noqa: E402
from scripts.run_ao_first_seed_boost_backtest import aggregate_target  # noqa: E402
from scripts.run_current_model_evaluation import (  # noqa: E402
    EvaluationArm,
    evaluate_arm,
    markdown_table,
)
from scripts.run_dynamic_core_calibration import expanding_folds  # noqa: E402
from scripts.run_european_exposure_cap_backtest import (  # noqa: E402
    dependency_uncertainty,
)
from scripts.run_european_exposure_tail_backtest import (  # noqa: E402
    DEFAULT_BOOTSTRAP_SAMPLES,
    EXPECTED_FOLDS,
    EXPECTED_UNSEEN_MATCHES,
    add_fold_accuracy,
    add_model_accuracy,
    attach_baseline_classes,
    competition_with_accuracy,
    fixed_selection,
    metric_goal,
)
from scripts.run_european_prior_recalibration_backtest import (  # noqa: E402
    BASELINE_CONFIG,
    aggregate_candidate_surface,
    build_unseen,
    load_seed_evidence,
    model_summary,
    nested_selection,
    season_surface,
)
from scripts.run_opponent_quintile_backtest import (  # noqa: E402
    load_production_baseline,
)
from scripts.run_stage_weighted_progression_backtest import (  # noqa: E402
    EVENTS_PATH,
    PRODUCTION_CONTRACT,
    STATIC_DATA_ROOT,
    XG_DATA,
    load_xg_map,
)
from scripts.run_v2_evaluation_upgrade import read_events  # noqa: E402


OUTPUT_ROOT = ROOT / "output" / "entry_path_exposure_tail_discovery_2018_2026"
TARGET_COMPETITIONS = ("UCL", "UEL", "UECL")
TAIL_BETAS = (
    0.025,
    0.05,
    0.075,
    0.10,
    0.15,
    0.20,
    0.25,
    1.0 / 3.0,
    0.40,
    0.50,
    2.0 / 3.0,
    0.75,
    1.00,
)
PRIMARY_COMPETITION = "UCL"
PRIMARY_BETA = 0.10


@dataclass(frozen=True)
class CompetitionTailCandidate:
    target_competition: str | None
    config: object

    @property
    def key(self) -> str:
        if self.target_competition is None:
            return BASELINE_CONFIG.key
        return f"{self.config.key}_only_{self.target_competition.lower()}"

    @property
    def beta(self) -> float:
        return 0.0 if self.target_competition is None else self.config.exposure_tail_beta


def candidate_grid() -> tuple[CompetitionTailCandidate, ...]:
    baseline = CompetitionTailCandidate(None, BASELINE_CONFIG)
    challengers = tuple(
        CompetitionTailCandidate(
            competition,
            replace(
                BASELINE_CONFIG,
                exposure_family="CAP_TAIL",
                exposure_tail_beta=beta,
            ),
        )
        for competition in TARGET_COMPETITIONS
        for beta in TAIL_BETAS
    )
    candidates = (baseline, *challengers)
    keys = [candidate.key for candidate in candidates]
    if len(keys) != len(set(keys)):
        raise ValueError("Competition-tail candidate keys must be unique")
    return candidates


def outcome_conditioned_competition_audit() -> pd.DataFrame:
    """Find qualifier entrants labelled by the competition they later reached.

    A Champions-path qualifier is known to be entering UCL at its season-start
    prediction time.  Historical ``domestic_context.csv`` instead labels most
    such rows UEL/UECL after elimination.  That makes ``competition`` unsafe as
    a season-start candidate switch: it contains a later qualification result.
    """

    rows = []
    for folder in sorted(STATIC_DATA_ROOT.glob("20??-??")):
        teams = pd.read_csv(folder / "teams.csv")
        context = pd.read_csv(folder / "domestic_context.csv")
        frame = teams[["team_id", "team_name"]].merge(
            context,
            on="team_id",
            validate="one_to_one",
        )
        entry_round = frame["entry_round"].astype(str)
        entered_ucl_qualifying = entry_round.str.startswith(("ch/", "CL-"))
        leaked = frame.loc[
            entered_ucl_qualifying & frame["competition"].ne("UCL"),
            [
                "season",
                "team_id",
                "team_name",
                "european_entry_type",
                "entry_round",
                "competition",
            ],
        ].copy()
        rows.append(leaked)
    result = pd.concat(rows, ignore_index=True)
    if result.empty:
        raise ValueError(
            "Expected historical competition timing audit to expose "
            "outcome-conditioned qualifier labels"
        )
    return result


def entry_path_map() -> pd.DataFrame:
    """Build the competition known before the club's first European match."""

    rows = []
    for folder in sorted(STATIC_DATA_ROOT.glob("20??-??")):
        context = pd.read_csv(folder / "domestic_context.csv")
        token = context["entry_round"].astype(str)

        def classify(value: str) -> str:
            if value.startswith(("ch/", "CL-")) or value == "UCL":
                return "UCL"
            if value.startswith(("eu/", "eu//", "nc/", "EL-")) or value == "UEL":
                return "UEL"
            if value.startswith(("ECL-", "CO-")) or value == "UECL":
                return "UECL"
            raise ValueError(f"Unsupported historical entry_round token: {value!r}")

        mapped = context[["season", "team_id", "entry_round"]].copy()
        mapped["entry_competition"] = token.map(classify)
        rows.append(mapped)
    result = pd.concat(rows, ignore_index=True)
    if result.duplicated(["season", "team_id"]).any():
        raise ValueError("Entry-path map contains duplicate team-season keys")
    return result


def candidate_metadata(candidates) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_key": candidate.key,
                "target_competition": (
                    "ALL_PRODUCTION"
                    if candidate.target_competition is None
                    else candidate.target_competition
                ),
                "exposure_tail_beta": candidate.beta,
                "full_exposure_weight_for_target": (
                    candidate.config.exposure_cap
                    + candidate.beta * (1.0 - candidate.config.exposure_cap)
                ),
            }
            for candidate in candidates
        ]
    )


def competition_specific_seed_frame(
    source: pd.DataFrame,
    production: pd.DataFrame,
    candidate: CompetitionTailCandidate,
) -> pd.DataFrame:
    """Apply the tail only to the candidate's known entry-path seed rows."""

    if candidate.target_competition is None:
        result = production.copy()
        result["candidate_key"] = candidate.key
        return result

    tail = apply_european_prior_recalibration(source, candidate.config)
    if not tail[["season", "team_id", "competition", "entry_competition"]].equals(
        production[["season", "team_id", "competition", "entry_competition"]]
    ):
        raise ValueError("Tail and production seed frames are not row-aligned")
    mask = production["entry_competition"].eq(candidate.target_competition)
    result = production.copy()
    candidate_columns = [
        column for column in result.columns if column.startswith("candidate_")
    ]
    for column in candidate_columns:
        result.loc[mask, column] = tail.loc[mask, column]
    result["candidate_key"] = candidate.key
    result["candidate_elo_delta"] = (
        result["candidate_ao_first_elo"] - result["adjusted_ao_first_elo"]
    )
    if not result.loc[~mask, "candidate_elo_delta"].abs().le(1e-8).all():
        raise ValueError("Competition-tail candidate changed a non-target seed row")
    if not result.loc[mask, "entry_competition"].eq(
        candidate.target_competition
    ).all():
        raise ValueError("Entry-path tail target mask is inconsistent")
    return result


def fixed_unseen_summary(
    surface: pd.DataFrame,
    folds: tuple[tuple[tuple[str, ...], str], ...],
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    test_seasons = {test for _, test in folds}
    unseen = surface.loc[surface["season"].isin(test_seasons)].copy()
    summary = aggregate_candidate_surface(unseen)
    baseline = unseen.loc[unseen["candidate_key"].eq(BASELINE_CONFIG.key)][
        ["season", "brier_1x2", "log_loss_1x2", "accuracy_1x2"]
    ].rename(
        columns={
            "brier_1x2": "baseline_brier_1x2",
            "log_loss_1x2": "baseline_log_loss_1x2",
            "accuracy_1x2": "baseline_accuracy_1x2",
        }
    )
    wins = []
    for key, frame in unseen.groupby("candidate_key", sort=True):
        compared = frame.merge(baseline, on="season", validate="one_to_one")
        wins.append(
            {
                "candidate_key": key,
                "brier_fold_wins": int(
                    (compared["brier_1x2"] < compared["baseline_brier_1x2"]).sum()
                ),
                "log_loss_fold_wins": int(
                    (
                        compared["log_loss_1x2"]
                        < compared["baseline_log_loss_1x2"]
                    ).sum()
                ),
                "accuracy_fold_wins": int(
                    (
                        compared["accuracy_1x2"]
                        > compared["baseline_accuracy_1x2"]
                    ).sum()
                ),
            }
        )
    return (
        summary.merge(pd.DataFrame(wins), on="candidate_key", validate="one_to_one")
        .merge(metadata, on="candidate_key", validate="one_to_one")
        .sort_values(
            ["target_competition", "exposure_tail_beta", "candidate_key"],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def fixed_evidence(
    candidates,
    folds,
    predictions,
    candidate_seeds,
    target,
    bootstrap_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_rows = []
    fold_rows = []
    competition_rows = []
    loss_rows = []
    ranking_rows = []
    baseline_predictions = predictions[BASELINE_CONFIG.key]
    for index, candidate in enumerate(candidates, start=1):
        if candidate.target_competition is None:
            continue
        selections = fixed_selection(candidate.key, folds)
        unseen, fold_result = build_unseen(
            selections, predictions, candidate_seeds, target, folds
        )
        unseen = attach_baseline_classes(unseen, baseline_predictions)
        fold_result = add_fold_accuracy(fold_result, unseen)
        model = add_model_accuracy(model_summary(unseen, fold_result), unseen).loc[
            lambda frame: frame["model"].eq("NESTED_RECALIBRATION")
        ]
        competition = competition_with_accuracy(unseen)
        loss = dependency_uncertainty(unseen, bootstrap_samples)
        ranking = ranking_uncertainty_summary(
            fold_result,
            bootstrap_samples,
            seed=20260904 + index,
        )
        for frame in (model, fold_result, competition, loss, ranking):
            frame.insert(0, "exposure_tail_beta", candidate.beta)
            frame.insert(0, "target_competition", candidate.target_competition)
            frame.insert(0, "candidate_key", candidate.key)
        model_rows.append(model)
        fold_rows.append(fold_result)
        competition_rows.append(competition)
        loss_rows.append(loss)
        ranking_rows.append(ranking)
    return (
        pd.concat(model_rows, ignore_index=True),
        pd.concat(fold_rows, ignore_index=True),
        pd.concat(competition_rows, ignore_index=True),
        pd.concat(loss_rows, ignore_index=True),
        pd.concat(ranking_rows, ignore_index=True),
    )


def targeted_competition_summary(fixed_competition: pd.DataFrame) -> pd.DataFrame:
    result = fixed_competition.loc[
        fixed_competition["competition"].eq(
            fixed_competition["target_competition"]
        )
    ].copy()
    if len(result) != len(TARGET_COMPETITIONS) * len(TAIL_BETAS):
        raise ValueError("Every competition-tail candidate needs one target segment")
    return result.reset_index(drop=True)


def build_report(
    decision: dict[str, object],
    timing_audit: pd.DataFrame,
    seed_scope_audit: pd.DataFrame,
    fixed_curve: pd.DataFrame,
    targeted_segments: pd.DataFrame,
    fixed_competition: pd.DataFrame,
    fixed_loss: pd.DataFrame,
    fixed_ranking: pd.DataFrame,
    nested_selections: pd.DataFrame,
    nested_model: pd.DataFrame,
    nested_folds: pd.DataFrame,
    nested_competition: pd.DataFrame,
    nested_loss: pd.DataFrame,
    nested_ranking: pd.DataFrame,
) -> str:
    curve_columns = [
        "target_competition",
        "exposure_tail_beta",
        "full_exposure_weight_for_target",
        "brier_1x2",
        "delta_vs_baseline_brier_1x2",
        "log_loss_1x2",
        "delta_vs_baseline_log_loss_1x2",
        "accuracy_1x2",
        "delta_vs_baseline_accuracy_1x2",
        "seed_spearman",
        "delta_vs_baseline_seed_spearman",
        "seed_pairwise_accuracy",
        "delta_vs_baseline_seed_pairwise_accuracy",
        "brier_fold_wins",
        "log_loss_fold_wins",
        "accuracy_fold_wins",
    ]
    competition_columns = [
        "target_competition",
        "exposure_tail_beta",
        "competition",
        "matches",
        "delta_brier_1x2",
        "delta_log_loss_1x2",
        "delta_accuracy_1x2",
    ]
    uncertainty_columns = [
        "target_competition",
        "exposure_tail_beta",
        "metric",
        "method",
        "mean_difference",
        "ci_95_lower",
        "ci_95_upper",
        "reliable_improvement",
        "reliable_harm",
    ]
    return "\n".join(
        [
            "# Known Entry-Path Exposure Tail Discovery",
            "",
            "> **EXPLORATORY / POST-HOC / PROMOTION KANITI DEĞİL.** Aday switch'i "
            "tahmin öncesi bilinen `entry_round` yolundan türetilir; tarihsel "
            "sonuç-koşullu `competition` etiketi kullanılmaz.",
            "",
            "## Sonuç",
            "",
            f"- Karar etiketi: `{decision['decision']}`",
            "- Promotion için uygun: `False`",
            f"- Kaynakta saptanan sonuç-koşullu qualifier etiketi: "
            f"`{decision['outcome_conditioned_qualifier_rows']}` satır",
            f"- Ana exploratory aday: `{PRIMARY_COMPETITION}`, "
            f"`beta={PRIMARY_BETA:g}`; üçlü hedef: "
            f"`{decision['primary_exploratory_meets_metric_goal']}`",
            "- Nested diagnostic üçlü hedefi: "
            f"`{decision['nested_meets_metric_goal']}`",
            f"- Nested Brier / log-loss / accuracy farkları: "
            f"`{decision['nested_brier_delta']:+.6f}` / "
            f"`{decision['nested_log_loss_delta']:+.6f}` / "
            f"`{decision['nested_accuracy_delta']:+.6f}`",
            "- UCL ile UEL/UECL kayıp işaretleri ayrıştı: "
            f"`{decision['competition_sign_split_observed']}`",
            "- Nested seçim her foldda beta gridinin üst sınırında: "
            f"`{decision['nested_selected_grid_edge_every_fold']}`",
            "- Nested sıralama proxy'sinde güvenilir zarar: "
            f"`{decision['nested_ranking_reliable_harm']}`",
            "",
            "Nested seçim yalnız geçmiş train sezonlarını kullanır; ancak araştırma "
            "sorusu aynı veri üzerindeki önceki segment sonuçlarından doğdu. Entry "
            "path zaman açısından geçerli olsa da bu post-hoc araştırma seçimini "
            "prospective kanıta dönüştürmez.",
            "",
            "## Zaman sızıntısı kanıtı",
            "",
            "`entry_round=ch/Q*` veya `CL-Q*` takımın UCL qualifying'e girdiğini "
            "gösterir. Aynı satırdaki sonuç-koşullu `competition` etiketi aday "
            "switch'inde kullanılmaz; aşağıdaki tablo bu dışlamanın nedenini gösterir.",
            "",
            markdown_table(timing_audit.head(100)),
            "",
            "## Hipotez ve uygulama",
            "",
            "- Production baseline aynen korunur.",
            "- Her challenger yalnız UCL, UEL veya UECL entry-path seed "
            "satırlarında CAP_TAIL uygular.",
            "- Diğer iki kupanın team-season seed ratingleri birebir production'dır.",
            "- Tail: `raw <= 0.65` ise `raw`; üstte "
            "`0.65 + beta * (raw - 0.65)`.",
            f"- Beta grid: `{', '.join(f'{value:g}' for value in TAIL_BETAS)}`",
            "",
            "## Veri ve yöntem",
            "",
            f"- Sezon penceresi: `{decision['evidence_window']}`",
            f"- Expanding outer fold: `{decision['fold_count']}`",
            f"- Görülmemiş maç: `{decision['unseen_matches']}`",
            f"- Dependency bootstrap örneklemi: `{decision['bootstrap_samples']}`",
            "",
            "## Seed kapsam denetimi",
            "",
            markdown_table(seed_scope_audit),
            "",
            "## Sabit candidate pooled sonuçları",
            "",
            markdown_table(fixed_curve[curve_columns]),
            "",
            "Bu tablodan en iyi kupa/beta seçmek post-hoc'tur; promotion için "
            "kullanılamaz.",
            "",
            "## Yalnız hedef kupa segmenti",
            "",
            markdown_table(targeted_segments[competition_columns]),
            "",
            "## Bütün candidate x kupa segmentleri",
            "",
            markdown_table(fixed_competition[competition_columns]),
            "",
            "## Sabit candidate kayıp belirsizliği",
            "",
            markdown_table(fixed_loss[uncertainty_columns]),
            "",
            "## Sabit candidate sıralama belirsizliği",
            "",
            markdown_table(fixed_ranking[uncertainty_columns]),
            "",
            "## Candidate-grid nested diagnostic",
            "",
            markdown_table(nested_selections),
            "",
            markdown_table(nested_model),
            "",
            "### Nested foldlar",
            "",
            markdown_table(nested_folds),
            "",
            "### Nested kupa segmentleri",
            "",
            markdown_table(nested_competition),
            "",
            "### Nested kayıp ve sıralama belirsizliği",
            "",
            markdown_table(nested_loss),
            "",
            markdown_table(nested_ranking),
            "",
            "## Kanıt sınırı",
            "",
            "Bu sonuç aynı tarihsel penceredeki segment keşfinden sonra tasarlandığı "
            "için production seçimi sayılamaz. Switch'in girdisi entry_round olduğu "
            "için zaman açısından kullanılabilirdir. Accuracy için ayrı güven aralığı "
            "hesaplanmaz. Production contract, config ve artifact zinciri "
            "değiştirilmemiştir.",
            "",
            "## Yeniden üretme",
            "",
            "`python3 scripts/run_entry_path_exposure_tail_backtest.py`",
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES
    )
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples must be at least 100")
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract_hash = hashlib.sha256(PRODUCTION_CONTRACT.read_bytes()).hexdigest()

    _, datasets, core, parameters, production_seed_map = load_production_baseline()
    seasons = tuple(data.season for data in datasets)
    folds = expanding_folds(seasons)
    if len(folds) != EXPECTED_FOLDS:
        raise ValueError(f"Expected {EXPECTED_FOLDS} folds, observed {len(folds)}")
    events = read_events(EVENTS_PATH)
    target_by_competition = schedule_adjusted_team_performance(events)
    target = aggregate_target(target_by_competition)
    xg_map = load_xg_map(XG_DATA, datasets)
    source_seeds = load_seed_evidence(production_seed_map)
    timing_audit = outcome_conditioned_competition_audit()
    entry_paths = entry_path_map()
    source_seeds = source_seeds.merge(
        entry_paths,
        on=["season", "team_id"],
        validate="one_to_one",
    )
    production_seeds = apply_european_prior_recalibration(
        source_seeds, BASELINE_CONFIG
    )
    candidates = candidate_grid()
    metadata = candidate_metadata(candidates)

    predictions: dict[str, pd.DataFrame] = {}
    candidate_seeds: dict[str, pd.DataFrame] = {}
    seed_scope_rows: list[dict[str, object]] = []
    surface_rows: list[dict[str, object]] = []
    for index, candidate in enumerate(candidates, start=1):
        seed_frame = competition_specific_seed_frame(
            source_seeds, production_seeds, candidate
        )
        candidate_seeds[candidate.key] = seed_frame
        if candidate.target_competition is None:
            target_mask = pd.Series(False, index=seed_frame.index)
        else:
            target_mask = seed_frame["competition"].eq(candidate.target_competition)
        seed_delta = seed_frame["candidate_elo_delta"].abs()
        non_target_delta = seed_delta.loc[~target_mask]
        seed_scope_rows.append(
            {
                "candidate_key": candidate.key,
                "target_competition": (
                    "ALL_PRODUCTION"
                    if candidate.target_competition is None
                    else candidate.target_competition
                ),
                "exposure_tail_beta": candidate.beta,
                "team_seasons": len(seed_frame),
                "target_team_seasons": int(target_mask.sum()),
                "target_rows_changed": int(seed_delta.loc[target_mask].gt(1e-8).sum()),
                "non_target_team_seasons": int((~target_mask).sum()),
                "non_target_rows_changed": int(non_target_delta.gt(1e-8).sum()),
                "maximum_non_target_absolute_delta": (
                    float(non_target_delta.max()) if len(non_target_delta) else 0.0
                ),
            }
        )
        rating_map = {
            (str(row.season), int(row.team_id)): float(row.candidate_ao_first_elo)
            for row in seed_frame.itertuples(index=False)
        }
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
        rows = season_surface(
            candidate.config,
            seed_frame,
            evaluation.predictions,
            target,
            seasons,
        )
        for row in rows:
            row["candidate_key"] = candidate.key
            row["target_competition"] = (
                "ALL_PRODUCTION"
                if candidate.target_competition is None
                else candidate.target_competition
            )
            row["exposure_tail_beta"] = candidate.beta
        surface_rows.extend(rows)
        print(f"Replayed {index}/{len(candidates)}: {candidate.key}", flush=True)

    surface = pd.DataFrame(surface_rows)
    seed_scope_audit = pd.DataFrame(seed_scope_rows)
    fixed_curve = fixed_unseen_summary(surface, folds, metadata)
    baseline_row = fixed_curve.loc[
        fixed_curve["candidate_key"].eq(BASELINE_CONFIG.key)
    ].iloc[0]
    if int(baseline_row["matches"]) != EXPECTED_UNSEEN_MATCHES:
        raise ValueError(
            f"Expected {EXPECTED_UNSEEN_MATCHES} unseen matches, observed "
            f"{int(baseline_row['matches'])}"
        )

    fixed_models, fixed_folds, fixed_competition, fixed_loss, fixed_ranking = (
        fixed_evidence(
            candidates,
            folds,
            predictions,
            candidate_seeds,
            target,
            int(args.bootstrap_samples),
        )
    )
    targeted_segments = targeted_competition_summary(fixed_competition)

    nested_selections = nested_selection(surface, candidates, folds)
    nested_unseen, nested_folds = build_unseen(
        nested_selections, predictions, candidate_seeds, target, folds
    )
    nested_unseen = attach_baseline_classes(
        nested_unseen, predictions[BASELINE_CONFIG.key]
    )
    nested_folds = add_fold_accuracy(nested_folds, nested_unseen)
    nested_model = add_model_accuracy(
        model_summary(nested_unseen, nested_folds), nested_unseen
    )
    nested_competition = competition_with_accuracy(nested_unseen)
    nested_loss = dependency_uncertainty(
        nested_unseen, int(args.bootstrap_samples)
    )
    nested_ranking = ranking_uncertainty_summary(
        nested_folds, int(args.bootstrap_samples), seed=20260905
    )

    primary = next(
        candidate
        for candidate in candidates
        if candidate.target_competition == PRIMARY_COMPETITION
        and math.isclose(candidate.beta, PRIMARY_BETA, abs_tol=1e-12)
    )
    primary_row = fixed_curve.loc[
        fixed_curve["candidate_key"].eq(primary.key)
    ].iloc[0]
    primary_target_row = targeted_segments.loc[
        targeted_segments["candidate_key"].eq(primary.key)
    ].iloc[0]
    nested_row = nested_model.loc[
        nested_model["model"].eq("NESTED_RECALIBRATION")
    ].iloc[0]
    challenger_curve = fixed_curve.loc[
        fixed_curve["target_competition"].isin(TARGET_COMPETITIONS)
    ]
    ucl_curve = challenger_curve.loc[
        challenger_curve["target_competition"].eq("UCL")
    ]
    lower_cup_curve = challenger_curve.loc[
        challenger_curve["target_competition"].isin(("UEL", "UECL"))
    ]
    sign_split = bool(
        ucl_curve["delta_vs_baseline_brier_1x2"].lt(0.0).all()
        and ucl_curve["delta_vs_baseline_log_loss_1x2"].lt(0.0).all()
        and lower_cup_curve["delta_vs_baseline_brier_1x2"].gt(0.0).all()
        and lower_cup_curve["delta_vs_baseline_log_loss_1x2"].gt(0.0).all()
    )
    edge_keys = {
        candidate.key
        for candidate in candidates
        if candidate.target_competition is not None
        and math.isclose(candidate.beta, max(TAIL_BETAS), abs_tol=1e-12)
    }
    nested_edge_every_fold = bool(
        nested_selections["selected_candidate_key"].isin(edge_keys).all()
    )
    nested_loss_envelope = nested_loss.loc[
        nested_loss["method"].eq("conservative_envelope")
    ]
    decision = {
        "decision": "EXPLORATORY_ENTRY_PATH_ONLY_NO_PROMOTION",
        "eligible_for_promotion": False,
        "candidate_switch_uses_pre_match_entry_path": True,
        "outcome_conditioned_competition_excluded_from_switch": True,
        "outcome_conditioned_qualifier_rows": int(len(timing_audit)),
        "posthoc_hypothesis": True,
        "posthoc_reason": (
            "Competition-specific hypothesis was created after pooled segment "
            "results from the global exposure mapping had been inspected."
        ),
        "production_changed": False,
        "evidence_window": f"{seasons[0]}-{seasons[-1]}",
        "fold_count": len(folds),
        "unseen_matches": int(len(nested_unseen)),
        "candidate_count_including_baseline": len(candidates),
        "bootstrap_samples": int(args.bootstrap_samples),
        "primary_exploratory_competition": PRIMARY_COMPETITION,
        "primary_exploratory_beta": PRIMARY_BETA,
        "primary_exploratory_candidate_key": primary.key,
        "primary_exploratory_meets_metric_goal": metric_goal(primary_row),
        "primary_pooled_brier_delta": float(
            primary_row["delta_vs_baseline_brier_1x2"]
        ),
        "primary_pooled_log_loss_delta": float(
            primary_row["delta_vs_baseline_log_loss_1x2"]
        ),
        "primary_pooled_accuracy_delta": float(
            primary_row["delta_vs_baseline_accuracy_1x2"]
        ),
        "primary_target_segment_brier_delta": float(
            primary_target_row["delta_brier_1x2"]
        ),
        "primary_target_segment_log_loss_delta": float(
            primary_target_row["delta_log_loss_1x2"]
        ),
        "primary_target_segment_accuracy_delta": float(
            primary_target_row["delta_accuracy_1x2"]
        ),
        "nested_selected_keys": nested_selections[
            "selected_candidate_key"
        ].tolist(),
        "nested_meets_metric_goal": metric_goal(nested_row),
        "nested_brier_delta": float(
            nested_row["delta_vs_baseline_brier_1x2"]
        ),
        "nested_log_loss_delta": float(
            nested_row["delta_vs_baseline_log_loss_1x2"]
        ),
        "nested_accuracy_delta": float(
            nested_row["delta_vs_baseline_accuracy_1x2"]
        ),
        "nested_loss_reliable_improvement": bool(
            len(nested_loss_envelope) == 2
            and nested_loss_envelope["reliable_improvement"].all()
        ),
        "nested_ranking_reliable_harm": bool(
            nested_ranking["reliable_harm"].any()
        ),
        "nested_selected_grid_edge_every_fold": nested_edge_every_fold,
        "competition_sign_split_observed": sign_split,
        "production_contract_sha256": contract_hash,
    }

    surface.to_csv(output / "candidate_surface.csv", index=False)
    timing_audit.to_csv(output / "outcome_conditioned_competition_audit.csv", index=False)
    fixed_curve.to_csv(output / "fixed_unseen_curve.csv", index=False)
    fixed_models.to_csv(output / "fixed_model_comparison.csv", index=False)
    fixed_folds.to_csv(output / "fixed_fold_results.csv", index=False)
    fixed_competition.to_csv(output / "fixed_competition_summary.csv", index=False)
    targeted_segments.to_csv(output / "targeted_competition_summary.csv", index=False)
    fixed_loss.to_csv(output / "fixed_loss_uncertainty.csv", index=False)
    fixed_ranking.to_csv(output / "fixed_ranking_uncertainty.csv", index=False)
    seed_scope_audit.to_csv(output / "seed_scope_audit.csv", index=False)
    nested_selections.to_csv(output / "nested_selections.csv", index=False)
    nested_folds.to_csv(output / "nested_fold_results.csv", index=False)
    nested_model.to_csv(output / "nested_model_comparison.csv", index=False)
    nested_competition.to_csv(output / "nested_competition_summary.csv", index=False)
    nested_loss.to_csv(output / "nested_loss_uncertainty.csv", index=False)
    nested_ranking.to_csv(output / "nested_ranking_uncertainty.csv", index=False)
    nested_unseen.to_csv(output / "nested_unseen_predictions.csv", index=False)
    (output / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "REPORT.md").write_text(
        build_report(
            decision,
            timing_audit,
            seed_scope_audit,
            fixed_curve,
            targeted_segments,
            fixed_competition,
            fixed_loss,
            fixed_ranking,
            nested_selections,
            nested_model,
            nested_folds,
            nested_competition,
            nested_loss,
            nested_ranking,
        ),
        encoding="utf-8",
    )

    final_contract_hash = hashlib.sha256(PRODUCTION_CONTRACT.read_bytes()).hexdigest()
    if final_contract_hash != contract_hash:
        raise ValueError("Production contract changed during discovery backtest")
    print("EXPLORATORY ENTRY-PATH RESULT — NOT PROMOTION EVIDENCE", flush=True)
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)
    print(f"Report: {output / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
