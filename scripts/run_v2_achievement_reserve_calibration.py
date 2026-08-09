from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ao_elo.config import AO_MODEL_V2_VERSION, V2_RATING_MULTIPLIER  # noqa: E402
from ao_elo.progression_probability import (  # noqa: E402
    ProgressionProbabilityConfig,
    calibrate_progression_probability,
    identity_progression_probability_config,
)
from scripts.run_dynamic_core_calibration import (  # noqa: E402
    DynamicCoreConfig,
    expanding_folds,
    expected_home_score,
)
from scripts.run_goal_margin_calibration import (  # noqa: E402
    GoalMarginConfig,
    goal_margin_multiplier,
)
from scripts.run_progression_prestige_calibration import (  # noqa: E402
    TieExpectation,
    validate_tie_contract,
)
from scripts.run_stage_progression_calibration import (  # noqa: E402
    PROFILE_BY_NAME,
    STAGE_PROFILES,
    normalize_stage,
)
from scripts.run_v2_dynamic_calibration import (  # noqa: E402
    MAX_RATING_MOVE_GUARDRAIL,
    RANK_CORRELATION_FLOOR,
    add_event_clusters,
    clustered_uncertainty,
    prefix_losses,
    read_event_metadata,
    read_static_config,
    summarize_loss_differences,
)
from scripts.run_v2_goal_margin_calibration import (  # noqa: E402
    GoalCarrySeasonData,
    load_goal_data,
    read_dynamic_manifest,
    safe_rank_correlation,
)


STATIC_DATA_ROOT = ROOT / "data" / "backtest_stage_b_2018_2026"
EXACT_EVENTS_PATH = (
    ROOT / "data" / "external_elo_benchmark_2018_2026" / "exact_date_events.csv"
)
STATIC_MANIFEST_PATH = (
    ROOT / "output" / "v2_ranking_calibration_2018_2026" / "selected_model.json"
)
DYNAMIC_OUTPUT_ROOT = ROOT / "output" / "v2_dynamic_calibration_2018_2026"
GOAL_OUTPUT_ROOT = ROOT / "output" / "v2_goal_margin_calibration_2018_2026"
OUTPUT_ROOT = ROOT / "output" / "v2_achievement_reserve_calibration_2018_2026"
RESERVE_BASES = tuple(value * V2_RATING_MULTIPLIER for value in (0, 20, 30, 40, 50, 60))
UEL_MULTIPLIERS = (0.50, 0.65, 0.80)
UECL_MULTIPLIERS = (0.25, 0.45, 0.60)
RESERVE_DECAYS = (0.25, 0.50, 0.75, 1.0)
RESERVE_CAP = 80.0 * V2_RATING_MULTIPLIER
MODEL_NAME = "achievement_reserve"
BASELINE_NAME = "matched_power_goal"


@dataclass(frozen=True, order=True)
class AchievementReserveConfig:
    reserve_base: float
    uel_multiplier: float
    uecl_multiplier: float
    stage_profile: str
    reserve_decay: float
    reserve_cap: float = RESERVE_CAP

    def validate(self) -> None:
        values = {
            "reserve_base": self.reserve_base,
            "uel_multiplier": self.uel_multiplier,
            "uecl_multiplier": self.uecl_multiplier,
            "reserve_decay": self.reserve_decay,
            "reserve_cap": self.reserve_cap,
        }
        if any(not math.isfinite(value) or value < 0 for value in values.values()):
            raise ValueError("Reserve config values must be non-negative and finite")
        if self.reserve_cap <= 0:
            raise ValueError("reserve_cap must be positive")
        if self.reserve_base > self.reserve_cap:
            raise ValueError("reserve_base cannot exceed reserve_cap")
        if not 1.0 > self.uel_multiplier > self.uecl_multiplier > 0.0:
            raise ValueError("Competition hierarchy must satisfy UCL > UEL > UECL > 0")
        if self.stage_profile not in PROFILE_BY_NAME:
            raise ValueError(f"Unknown stage profile: {self.stage_profile}")
        if not 0.0 <= self.reserve_decay <= 1.0:
            raise ValueError("reserve_decay must be in [0,1]")
        if self.reserve_base == 0.0 and self.reserve_decay != 0.0:
            raise ValueError("reserve_decay must be zero when reserve is disabled")
        PROFILE_BY_NAME[self.stage_profile].validate()

    def competition_multiplier(self, competition: str) -> float:
        self.validate()
        try:
            return {
                "UCL": 1.0,
                "UEL": self.uel_multiplier,
                "UECL": self.uecl_multiplier,
            }[competition]
        except KeyError as error:
            raise ValueError(f"Unknown competition: {competition}") from error

    @property
    def profile(self):
        return PROFILE_BY_NAME[self.stage_profile]


@dataclass(frozen=True)
class ReserveSeasonData:
    goal: GoalCarrySeasonData
    tie_ids: np.ndarray
    tie_match_counts: np.ndarray
    knockout_flags: np.ndarray
    tie_decider_flags: np.ndarray
    advanced_team_ids: np.ndarray
    stages: np.ndarray

    @property
    def season(self) -> str:
        return self.goal.season


def baseline_config() -> AchievementReserveConfig:
    return AchievementReserveConfig(0.0, 0.65, 0.45, "FLAT", 0.0)


def candidate_grid() -> tuple[AchievementReserveConfig, ...]:
    candidates = {baseline_config()}
    candidates.update(
        AchievementReserveConfig(base, uel, uecl, profile.name, decay)
        for base in RESERVE_BASES
        if base > 0
        for uel in UEL_MULTIPLIERS
        for uecl in UECL_MULTIPLIERS
        if uel > uecl
        for profile in STAGE_PROFILES
        for decay in RESERVE_DECAYS
    )
    result = tuple(sorted(candidates))
    for candidate in result:
        candidate.validate()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate separate AO Elo v2 European Achievement Reserve"
    )
    parser.add_argument("--static-data-root", type=Path, default=STATIC_DATA_ROOT)
    parser.add_argument("--events-path", type=Path, default=EXACT_EVENTS_PATH)
    parser.add_argument("--static-manifest", type=Path, default=STATIC_MANIFEST_PATH)
    parser.add_argument("--dynamic-output-root", type=Path, default=DYNAMIC_OUTPUT_ROOT)
    parser.add_argument("--goal-output-root", type=Path, default=GOAL_OUTPUT_ROOT)
    parser.add_argument(
        "--progression-probability-manifest",
        type=Path,
        default=None,
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    static_config = read_static_config(args.static_manifest.resolve())
    dynamic_root = args.dynamic_output_root.resolve()
    dynamic_manifest = read_dynamic_manifest(
        dynamic_root / "selected_dynamic_model.json"
    )
    goal_manifest = read_goal_manifest(
        args.goal_output_root.resolve() / "selected_goal_model.json"
    )
    events_path = args.events_path.resolve()
    event_metadata = read_event_metadata(events_path)
    datasets, tie_audit = load_reserve_data(
        args.static_data_root.resolve(),
        events_path,
        static_config,
    )
    seasons = tuple(data.season for data in datasets)
    folds = expanding_folds(seasons)
    core_selections = pd.read_csv(dynamic_root / "core_fold_selections.csv")
    carry_selections = pd.read_csv(dynamic_root / "carry_fold_selections.csv")
    goal_config = GoalMarginConfig(
        float(goal_manifest["goal_margin"]["goal_weight"]),
        float(goal_manifest["goal_margin"]["goal_cap"]),
    )
    advance_configs = None
    full_advance_config = identity_progression_probability_config()
    if args.progression_probability_manifest is not None:
        probability_manifest = json.loads(
            args.progression_probability_manifest.resolve().read_text(
                encoding="utf-8"
            )
        )
        if not bool(probability_manifest.get("reserve_retest_authorized")):
            raise ValueError(
                "Progression probability manifest does not authorize reserve retest"
            )
        advance_configs = {
            str(season): progression_config_from_payload(payload)
            for season, payload in probability_manifest[
                "configs_by_test_season"
            ].items()
        }
        full_advance_config = progression_config_from_payload(
            probability_manifest["full_data_candidate"]
        )

    selections, fold_results, predictions = run_walk_forward(
        datasets,
        folds,
        core_selections,
        carry_selections,
        goal_config,
        event_metadata,
        advance_configs_by_test_season=advance_configs,
    )
    competition_summary = summarize_loss_differences(
        predictions, MODEL_NAME, BASELINE_NAME
    )
    uncertainty = clustered_uncertainty(predictions, MODEL_NAME, BASELINE_NAME)
    full_core = DynamicCoreConfig(**dynamic_manifest["dynamic_core"])
    full_carry = float(dynamic_manifest["active_power_carry"])
    final_candidate, full_candidate_metrics = select_candidate(
        datasets,
        full_core,
        full_carry,
        goal_config,
        candidate_grid(),
        progress_label="full-data",
        advance_probability_config=full_advance_config,
    )
    decision, guardrails = promotion_decision(
        selections,
        fold_results,
        uncertainty,
        final_candidate,
    )
    active = final_candidate if decision == "PROMOTE_ACHIEVEMENT_RESERVE" else baseline_config()
    stage_table = build_stage_table(active)

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    selections.to_csv(output_root / "fold_selections.csv", index=False)
    fold_results.to_csv(output_root / "fold_results.csv", index=False)
    predictions.to_csv(output_root / "unseen_predictions.csv", index=False)
    competition_summary.to_csv(
        output_root / "competition_summary.csv", index=False
    )
    uncertainty.to_csv(output_root / "clustered_uncertainty.csv", index=False)
    full_candidate_metrics.to_csv(
        output_root / "full_candidate_metrics.csv", index=False
    )
    stage_table.to_csv(output_root / "active_reserve_table.csv", index=False)
    tie_audit.to_csv(output_root / "tie_chronology_corrections.csv", index=False)
    write_manifest(
        output_root / "selected_reserve_model.json",
        dynamic_manifest,
        goal_manifest,
        final_candidate,
        active,
        decision,
        guardrails,
    )
    write_report(
        output_root / "calibration_report.md",
        seasons,
        selections,
        competition_summary,
        uncertainty,
        final_candidate,
        active,
        decision,
        guardrails,
    )

    print("AO European Elo v2 achievement-reserve calibration")
    print(f"Candidates: {len(candidate_grid())}")
    print(
        "Candidate: "
        f"base={final_candidate.reserve_base:.6f}, "
        f"UEL={final_candidate.uel_multiplier:g}, "
        f"UECL={final_candidate.uecl_multiplier:g}, "
        f"profile={final_candidate.stage_profile}, "
        f"decay={final_candidate.reserve_decay:g}"
    )
    print(f"Decision: {decision}")
    print(f"Active reserve base: {active.reserve_base:.6f}")
    print(f"Report: {output_root / 'calibration_report.md'}")


def progression_config_from_payload(
    payload: dict[str, object],
) -> ProgressionProbabilityConfig:
    config = ProgressionProbabilityConfig(
        float(payload["logit_slope"]),
        float(payload["single_home_bias"]),
        float(payload["two_leg_first_home_bias"]),
    )
    config.validate()
    return config


def read_goal_manifest(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "goal_margin" not in payload or "dynamic_core" not in payload:
        raise ValueError("Goal manifest lacks matched baseline parameters")
    return payload


def load_reserve_data(
    static_root: Path,
    events_path: Path,
    static_config,
) -> tuple[tuple[ReserveSeasonData, ...], pd.DataFrame]:
    goal_data = load_goal_data(static_root, events_path, static_config)
    events = pd.read_csv(events_path).sort_values(["season", "event_order"])
    required = {
        "match_id",
        "season",
        "competition",
        "round",
        "tie_id",
        "is_knockout",
        "is_tie_decider",
        "advanced_team_id",
        "home_team_id",
        "away_team_id",
    }
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"Reserve event data missing columns: {missing}")
    events, tie_audit = normalize_exact_tie_deciders(events)
    validate_tie_contract(events)
    tie_counts = (
        events.loc[events["is_knockout"].astype(bool)]
        .groupby(["season", "tie_id"])
        .size()
        .to_dict()
    )
    event_index = events.set_index("match_id")
    result = []
    for goal in goal_data:
        aligned = event_index.loc[goal.carry.core.match_ids]
        stages = np.array(
            [
                normalize_stage(str(round_name), str(competition), bool(knockout))
                for round_name, competition, knockout in zip(
                    aligned["round"],
                    aligned["competition"],
                    aligned["is_knockout"],
                )
            ]
        )
        result.append(
            ReserveSeasonData(
                goal=goal,
                tie_ids=aligned["tie_id"].where(
                    aligned["tie_id"].notna(), None
                ).to_numpy(object),
                tie_match_counts=np.array(
                    [
                        int(tie_counts[(str(season), str(tie_id))])
                        if pd.notna(tie_id)
                        else 0
                        for season, tie_id in zip(
                            aligned["season"],
                            aligned["tie_id"],
                        )
                    ],
                    dtype=int,
                ),
                knockout_flags=aligned["is_knockout"].to_numpy(bool),
                tie_decider_flags=aligned["is_tie_decider"].to_numpy(bool),
                advanced_team_ids=aligned["advanced_team_id"].fillna(-1).to_numpy(int),
                stages=stages,
            )
        )
    return tuple(result), tie_audit


def normalize_exact_tie_deciders(
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Move each tie decision to its chronologically last match."""
    result = events.sort_values(["season", "event_order"]).copy()
    audit_rows: list[dict[str, object]] = []
    knockout = result.loc[result["is_knockout"].astype(bool)]
    for (season, tie_id), tie in knockout.groupby(["season", "tie_id"], sort=False):
        teams = set(tie["home_team_id"].astype(int)) | set(
            tie["away_team_id"].astype(int)
        )
        if len(teams) != 2:
            raise ValueError(f"{season}/{tie_id}: tie must contain exactly two teams")
        winners = tie["advanced_team_id"].dropna().astype(int).unique()
        if len(winners) != 1 or int(winners[0]) not in teams:
            raise ValueError(
                f"{season}/{tie_id}: tie must have one valid advanced team"
            )
        old_deciders = tie.loc[tie["is_tie_decider"].astype(bool), "match_id"].tolist()
        final_index = tie.index[-1]
        final_match_id = str(result.loc[final_index, "match_id"])
        if old_deciders != [final_match_id]:
            audit_rows.append(
                {
                    "season": season,
                    "tie_id": tie_id,
                    "old_decider_match_ids": "|".join(map(str, old_deciders)),
                    "new_decider_match_id": final_match_id,
                    "advanced_team_id": int(winners[0]),
                    "reason": "exact_utc_chronological_last_match",
                }
            )
        result.loc[tie.index, "is_tie_decider"] = False
        result.loc[tie.index, "advanced_team_id"] = np.nan
        result.loc[final_index, "is_tie_decider"] = True
        result.loc[final_index, "advanced_team_id"] = int(winners[0])
    return result, pd.DataFrame(
        audit_rows,
        columns=[
            "season",
            "tie_id",
            "old_decider_match_ids",
            "new_decider_match_id",
            "advanced_team_id",
            "reason",
        ],
    )


def reserve_addition(
    winner_probability: float,
    competition: str,
    stage: str,
    config: AchievementReserveConfig,
    *,
    trophy: bool = False,
) -> float:
    config.validate()
    if not 0.0 <= winner_probability <= 1.0:
        raise ValueError("winner_probability must be in [0,1]")
    stage_multiplier = 1.0 if trophy else config.profile.for_stage(stage)
    return (
        config.reserve_base
        * config.competition_multiplier(competition)
        * stage_multiplier
        * (1.0 - winner_probability)
    )


def evaluate_sequence(
    datasets: tuple[ReserveSeasonData, ...],
    core_config: DynamicCoreConfig,
    power_carry: float,
    goal_config: GoalMarginConfig,
    reserve_config: AchievementReserveConfig,
    *,
    evaluation_seasons: set[str] | None = None,
    return_predictions: bool = False,
    advance_probability_config: ProgressionProbabilityConfig | None = None,
) -> tuple[dict[str, float | int], pd.DataFrame | None]:
    core_config.validate()
    goal_config.validate()
    reserve_config.validate()
    probability_config = (
        identity_progression_probability_config()
        if advance_probability_config is None
        else advance_probability_config
    )
    probability_config.validate()
    if not 0.0 <= power_carry <= 1.0:
        raise ValueError("power_carry must be in [0,1]")
    evaluation = evaluation_seasons or {data.season for data in datasets}
    previous_power: dict[str, float] = {}
    previous_reserve: dict[str, float] = {}
    metric_rows: list[dict[str, float | int]] = []
    prediction_rows: list[dict[str, object]] = []
    for data in datasets:
        goal = data.goal
        carry = goal.carry
        core = carry.core
        power = core.initial_ratings.copy()
        reserve = np.zeros_like(power)
        for team_id in core.active_team_ids:
            key = str(carry.club_keys[team_id])
            if key in previous_power:
                power[team_id] = (
                    (1.0 - power_carry) * core.initial_ratings[team_id]
                    + power_carry * previous_power[key]
                )
                reserve[team_id] = min(
                    reserve_config.reserve_cap,
                    reserve_config.reserve_decay * previous_reserve.get(key, 0.0),
                )
        open_ties: dict[str, TieExpectation] = {}
        brier_sum = 0.0
        log_loss_sum = 0.0
        reserve_total_added = 0.0
        trophy_total_added = 0.0
        max_pair_sum_error = 0.0
        for index, (home_id, away_id, actual, neutral, competition) in enumerate(
            zip(
                core.home_team_ids,
                core.away_team_ids,
                core.actual_home_scores,
                core.neutral_flags,
                core.competitions,
            )
        ):
            tie_value = data.tie_ids[index]
            tie_id = None if tie_value is None else str(tie_value)
            if data.knockout_flags[index] and tie_id not in open_ties:
                if tie_id is None:
                    raise ValueError(f"{data.season}/{core.match_ids[index]}: missing tie_id")
                raw_probability = expected_home_score(
                    power[home_id] + reserve[home_id],
                    power[away_id] + reserve[away_id],
                    core_config,
                    neutral=True,
                )
                open_ties[tie_id] = TieExpectation(
                    int(home_id),
                    int(away_id),
                    calibrate_progression_probability(
                        raw_probability,
                        int(data.tie_match_counts[index]),
                        bool(neutral),
                        probability_config,
                    ),
                )

            home_live = power[home_id] + reserve[home_id]
            away_live = power[away_id] + reserve[away_id]
            probability = float(
                np.clip(
                    expected_home_score(
                        home_live,
                        away_live,
                        core_config,
                        neutral=bool(neutral),
                    ),
                    1e-12,
                    1.0 - 1e-12,
                )
            )
            multiplier = goal_margin_multiplier(
                int(goal.goal_differences[index]), goal_config
            )
            delta = core_config.k_factor * multiplier * (actual - probability)
            pair_before = power[home_id] + power[away_id]
            power[home_id] += delta
            power[away_id] -= delta
            max_pair_sum_error = max(
                max_pair_sum_error,
                abs((power[home_id] + power[away_id]) - pair_before),
            )

            advance_added = 0.0
            trophy_added = 0.0
            winner_probability = np.nan
            winner_id = -1
            if data.tie_decider_flags[index]:
                if tie_id is None or tie_id not in open_ties:
                    raise ValueError(
                        f"{data.season}/{core.match_ids[index]}: unknown deciding tie"
                    )
                expectation = open_ties.pop(tie_id)
                winner_id = int(data.advanced_team_ids[index])
                if winner_id not in (expectation.team_a, expectation.team_b):
                    raise ValueError(f"{data.season}/{tie_id}: invalid advanced team")
                winner_probability = (
                    expectation.expected_a_to_advance
                    if winner_id == expectation.team_a
                    else 1.0 - expectation.expected_a_to_advance
                )
                stage = str(data.stages[index])
                advance_added = reserve_addition(
                    winner_probability,
                    str(competition),
                    stage,
                    reserve_config,
                )
                if stage == "FINAL":
                    trophy_added = reserve_addition(
                        winner_probability,
                        str(competition),
                        stage,
                        reserve_config,
                        trophy=True,
                    )
                available = max(0.0, reserve_config.reserve_cap - reserve[winner_id])
                total_added = min(available, advance_added + trophy_added)
                if advance_added + trophy_added > 0:
                    advance_share = advance_added / (advance_added + trophy_added)
                    advance_added = total_added * advance_share
                    trophy_added = total_added - advance_added
                reserve[winner_id] += total_added
                reserve_total_added += advance_added
                trophy_total_added += trophy_added

            brier_loss = (probability - actual) ** 2
            log_loss = -(
                actual * math.log(probability)
                + (1.0 - actual) * math.log(1.0 - probability)
            )
            if data.season in evaluation:
                brier_sum += brier_loss
                log_loss_sum += log_loss
                if return_predictions:
                    prediction_rows.append(
                        {
                            "match_id": core.match_ids[index],
                            "season": data.season,
                            "competition": competition,
                            "round": carry.rounds[index],
                            "stage": data.stages[index],
                            "actual_home_score": actual,
                            "home_power": power[home_id] - delta,
                            "away_power": power[away_id] + delta,
                            "home_reserve": reserve[home_id]
                            - (advance_added + trophy_added if winner_id == home_id else 0.0),
                            "away_reserve": reserve[away_id]
                            - (advance_added + trophy_added if winner_id == away_id else 0.0),
                            "expected_home_score": probability,
                            "power_delta": delta,
                            "winner_probability": winner_probability,
                            "advance_reserve_added_after_match": advance_added,
                            "trophy_reserve_added_after_match": trophy_added,
                            "brier_loss": brier_loss,
                            "log_loss": log_loss,
                        }
                    )
        if open_ties:
            raise ValueError(f"{data.season}: undecided knockout ties remain open")
        active_ids = core.active_team_ids
        live_end = power[active_ids] + reserve[active_ids]
        start = core.initial_ratings[active_ids]
        if data.season in evaluation:
            metric_rows.append(
                {
                    "matches": len(core.match_ids),
                    "brier": brier_sum / len(core.match_ids),
                    "log_loss": log_loss_sum / len(core.match_ids),
                    "advance_reserve_added": reserve_total_added,
                    "trophy_reserve_added": trophy_total_added,
                    "max_reserve": float(np.max(reserve[active_ids])),
                    "max_pair_sum_error": max_pair_sum_error,
                    "max_abs_rating_change": float(np.max(np.abs(live_end - start))),
                    "start_end_rank_correlation": safe_rank_correlation(start, live_end),
                }
            )
        previous_power = {
            str(carry.club_keys[team_id]): float(power[team_id])
            for team_id in active_ids
        }
        previous_reserve = {
            str(carry.club_keys[team_id]): float(reserve[team_id])
            for team_id in active_ids
        }
    if not metric_rows:
        raise ValueError("No reserve evaluation seasons were processed")
    matches = sum(int(row["matches"]) for row in metric_rows)
    metrics: dict[str, float | int] = {
        "matches": matches,
        "brier": sum(float(row["brier"]) * int(row["matches"]) for row in metric_rows)
        / matches,
        "log_loss": sum(
            float(row["log_loss"]) * int(row["matches"]) for row in metric_rows
        )
        / matches,
        "advance_reserve_added": sum(
            row["advance_reserve_added"] for row in metric_rows
        ),
        "trophy_reserve_added": sum(
            row["trophy_reserve_added"] for row in metric_rows
        ),
        "max_reserve": max(row["max_reserve"] for row in metric_rows),
        "max_pair_sum_error": max(row["max_pair_sum_error"] for row in metric_rows),
        "max_abs_rating_change": max(
            row["max_abs_rating_change"] for row in metric_rows
        ),
        "start_end_rank_correlation": min(
            row["start_end_rank_correlation"] for row in metric_rows
        ),
    }
    predictions = pd.DataFrame(prediction_rows) if return_predictions else None
    return metrics, predictions


def candidate_metrics(
    datasets: tuple[ReserveSeasonData, ...],
    core: DynamicCoreConfig,
    power_carry: float,
    goal: GoalMarginConfig,
    candidates: tuple[AchievementReserveConfig, ...],
    *,
    progress_label: str | None = None,
    advance_probability_config: ProgressionProbabilityConfig | None = None,
) -> pd.DataFrame:
    rows = []
    for index, config in enumerate(candidates, start=1):
        metrics, _ = evaluate_sequence(
            datasets,
            core,
            power_carry,
            goal,
            config,
            advance_probability_config=advance_probability_config,
        )
        rows.append(
            {
                "reserve_base": config.reserve_base,
                "uel_multiplier": config.uel_multiplier,
                "uecl_multiplier": config.uecl_multiplier,
                "stage_profile": config.stage_profile,
                "reserve_decay": config.reserve_decay,
                **metrics,
            }
        )
        if progress_label and index % 200 == 0:
            print(
                f"Reserve candidates ({progress_label}): {index}/{len(candidates)}",
                flush=True,
            )
    return pd.DataFrame(rows).sort_values(
        [
            "brier",
            "log_loss",
            "reserve_base",
            "reserve_decay",
            "stage_profile",
            "uel_multiplier",
            "uecl_multiplier",
        ]
    ).reset_index(drop=True)


def select_candidate(
    datasets: tuple[ReserveSeasonData, ...],
    core: DynamicCoreConfig,
    power_carry: float,
    goal: GoalMarginConfig,
    candidates: tuple[AchievementReserveConfig, ...],
    *,
    progress_label: str | None = None,
    advance_probability_config: ProgressionProbabilityConfig | None = None,
) -> tuple[AchievementReserveConfig, pd.DataFrame]:
    metrics = candidate_metrics(
        datasets,
        core,
        power_carry,
        goal,
        candidates,
        progress_label=progress_label,
        advance_probability_config=advance_probability_config,
    )
    eligible = metrics.loc[
        metrics["start_end_rank_correlation"].ge(RANK_CORRELATION_FLOOR)
        & metrics["max_abs_rating_change"].le(MAX_RATING_MOVE_GUARDRAIL)
        & metrics["max_reserve"].le(RESERVE_CAP + 1e-9)
        & metrics["max_pair_sum_error"].le(1e-9)
    ]
    if eligible.empty:
        raise ValueError("No achievement-reserve candidate passes guardrails")
    row = eligible.iloc[0]
    config = AchievementReserveConfig(
        float(row["reserve_base"]),
        float(row["uel_multiplier"]),
        float(row["uecl_multiplier"]),
        str(row["stage_profile"]),
        float(row["reserve_decay"]),
    )
    return config, metrics


def run_walk_forward(
    datasets: tuple[ReserveSeasonData, ...],
    folds: tuple[tuple[tuple[str, ...], str], ...],
    core_selections: pd.DataFrame,
    carry_selections: pd.DataFrame,
    goal_config: GoalMarginConfig,
    events: pd.DataFrame,
    *,
    advance_configs_by_test_season: (
        dict[str, ProgressionProbabilityConfig] | None
    ) = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidates = candidate_grid()
    baseline = baseline_config()
    selections: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for fold, (train_seasons, test_season) in enumerate(folds, start=1):
        advance_config = (
            identity_progression_probability_config()
            if advance_configs_by_test_season is None
            else advance_configs_by_test_season[test_season]
        )
        advance_config.validate()
        core_row = core_selections.loc[core_selections["fold"].eq(fold)].iloc[0]
        carry_row = carry_selections.loc[carry_selections["fold"].eq(fold)].iloc[0]
        core = DynamicCoreConfig(
            float(core_row["selected_scale"]),
            float(core_row["selected_home_advantage"]),
            float(core_row["selected_k"]),
        )
        power_carry = float(carry_row["selected_power_carry"])
        train = tuple(data for data in datasets if data.season in train_seasons)
        sequence = tuple(
            data for data in datasets if data.season in (*train_seasons, test_season)
        )
        selected, train_metrics = select_candidate(
            train,
            core,
            power_carry,
            goal_config,
            candidates,
            progress_label=f"fold {fold}",
            advance_probability_config=advance_config,
        )
        baseline_metrics, _ = evaluate_sequence(
            train,
            core,
            power_carry,
            goal_config,
            baseline,
            advance_probability_config=advance_config,
        )
        selected_row = train_metrics.loc[
            train_metrics["reserve_base"].eq(selected.reserve_base)
            & train_metrics["uel_multiplier"].eq(selected.uel_multiplier)
            & train_metrics["uecl_multiplier"].eq(selected.uecl_multiplier)
            & train_metrics["stage_profile"].eq(selected.stage_profile)
            & train_metrics["reserve_decay"].eq(selected.reserve_decay)
        ].iloc[0]
        selections.append(
            {
                "fold": fold,
                "train_seasons": "|".join(train_seasons),
                "test_season": test_season,
                "core_scale": core.elo_scale,
                "core_home_advantage": core.home_advantage,
                "core_k": core.k_factor,
                "power_carry": power_carry,
                "advance_probability_config": advance_config.key,
                "advance_logit_slope": advance_config.logit_slope,
                "advance_single_home_bias": advance_config.single_home_bias,
                "advance_two_leg_first_home_bias": (
                    advance_config.two_leg_first_home_bias
                ),
                "selected_reserve_base": selected.reserve_base,
                "selected_uel_multiplier": selected.uel_multiplier,
                "selected_uecl_multiplier": selected.uecl_multiplier,
                "selected_stage_profile": selected.stage_profile,
                "selected_reserve_decay": selected.reserve_decay,
                "train_brier_difference": (
                    selected_row["brier"] - baseline_metrics["brier"]
                ),
            }
        )
        models: dict[str, pd.DataFrame] = {}
        for model, config in ((MODEL_NAME, selected), (BASELINE_NAME, baseline)):
            metrics, predictions = evaluate_sequence(
                sequence,
                core,
                power_carry,
                goal_config,
                config,
                evaluation_seasons={test_season},
                return_predictions=True,
                advance_probability_config=advance_config,
            )
            result_rows.append(
                {
                    "fold": fold,
                    "test_season": test_season,
                    "model": model,
                    "reserve_base": config.reserve_base,
                    "uel_multiplier": config.uel_multiplier,
                    "uecl_multiplier": config.uecl_multiplier,
                    "stage_profile": config.stage_profile,
                    "reserve_decay": config.reserve_decay,
                    "advance_probability_config": advance_config.key,
                    **metrics,
                }
            )
            assert predictions is not None
            models[model] = prefix_losses(predictions, model)
        joined = models[MODEL_NAME].merge(
            models[BASELINE_NAME][
                [
                    "match_id",
                    f"{BASELINE_NAME}_expected_home_score",
                    f"{BASELINE_NAME}_brier_loss",
                    f"{BASELINE_NAME}_log_loss",
                ]
            ],
            on="match_id",
            validate="one_to_one",
        )
        joined.insert(0, "fold", fold)
        prediction_frames.append(add_event_clusters(joined, events))
        print(f"Reserve outer fold complete: {fold}/{len(folds)}", flush=True)
    return (
        pd.DataFrame(selections),
        pd.DataFrame(result_rows),
        pd.concat(prediction_frames, ignore_index=True),
    )


def promotion_decision(
    selections: pd.DataFrame,
    fold_results: pd.DataFrame,
    uncertainty: pd.DataFrame,
    final_candidate: AchievementReserveConfig,
) -> tuple[str, dict[str, object]]:
    pivot = fold_results.pivot(index="fold", columns="model", values="brier")
    fold_wins = int((pivot[MODEL_NAME] < pivot[BASELINE_NAME]).sum())
    selected = fold_results.loc[fold_results["model"].eq(MODEL_NAME)]
    overall = uncertainty.loc[uncertainty["competition"].eq("ALL")].iloc[0]
    segments = uncertainty.loc[uncertainty["competition"].ne("ALL")]
    nonzero_share = float(selections["selected_reserve_base"].gt(0).mean())
    ranking_safe = bool(
        selected["start_end_rank_correlation"].ge(RANK_CORRELATION_FLOOR).all()
    )
    movement_safe = bool(
        selected["max_abs_rating_change"].le(MAX_RATING_MOVE_GUARDRAIL).all()
    )
    reserve_safe = bool(selected["max_reserve"].le(RESERVE_CAP + 1e-9).all())
    power_zero_sum = bool(selected["max_pair_sum_error"].le(1e-9).all())
    no_segment_harm = not bool(segments["reliable_harm"].any())
    hierarchy_valid = bool(
        final_candidate.uel_multiplier > final_candidate.uecl_multiplier
    )
    passed = (
        final_candidate.reserve_base > 0
        and nonzero_share >= 0.5
        and fold_wins >= 5
        and bool(overall["reliable_improvement"])
        and ranking_safe
        and movement_safe
        and reserve_safe
        and power_zero_sum
        and no_segment_harm
        and hierarchy_valid
    )
    guardrails = {
        "fold_wins": fold_wins,
        "nonzero_fold_share": nonzero_share,
        "overall_reliable_improvement": bool(overall["reliable_improvement"]),
        "ranking_safe": ranking_safe,
        "movement_safe": movement_safe,
        "reserve_cap_safe": reserve_safe,
        "power_updates_zero_sum": power_zero_sum,
        "no_competition_reliable_harm": no_segment_harm,
        "strict_competition_hierarchy": hierarchy_valid,
    }
    decision = "PROMOTE_ACHIEVEMENT_RESERVE" if passed else "DISABLE_ACHIEVEMENT_RESERVE"
    return decision, guardrails


def build_stage_table(config: AchievementReserveConfig) -> pd.DataFrame:
    rows = []
    for competition in ("UCL", "UEL", "UECL"):
        for stage in (
            "QUALIFYING",
            "KNOCKOUT_PLAYOFF",
            "ROUND_OF_16",
            "QUARTERFINAL",
            "SEMIFINAL",
            "FINAL",
        ):
            rows.append(
                {
                    "competition": competition,
                    "stage": stage,
                    "competition_multiplier": config.competition_multiplier(competition),
                    "stage_multiplier": (
                        1.0 if stage == "FINAL" else config.profile.for_stage(stage)
                    ),
                    "reserve_at_winner_probability_0_50": reserve_addition(
                        0.5,
                        competition,
                        stage,
                        config,
                        trophy=stage == "FINAL",
                    ),
                    "is_trophy_reserve": stage == "FINAL",
                }
            )
    return pd.DataFrame(rows)


def write_manifest(
    path: Path,
    dynamic_manifest: dict[str, object],
    goal_manifest: dict[str, object],
    candidate: AchievementReserveConfig,
    active: AchievementReserveConfig,
    decision: str,
    guardrails: dict[str, object],
) -> None:
    payload = {
        "model_version": AO_MODEL_V2_VERSION,
        "dynamic_core": dynamic_manifest["dynamic_core"],
        "active_power_carry": dynamic_manifest["active_power_carry"],
        "goal_margin": goal_manifest["goal_margin"],
        "achievement_reserve_candidate": config_dict(candidate),
        "achievement_reserve": {
            "active": active.reserve_base > 0,
            **config_dict(active),
        },
        "decision": decision,
        "guardrails": guardrails,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def config_dict(config: AchievementReserveConfig) -> dict[str, object]:
    return {
        "reserve_base": config.reserve_base,
        "ucl_multiplier": 1.0,
        "uel_multiplier": config.uel_multiplier,
        "uecl_multiplier": config.uecl_multiplier,
        "stage_profile": config.stage_profile,
        "reserve_decay": config.reserve_decay,
        "reserve_cap": config.reserve_cap,
        "trophy_uses_same_base": True,
    }


def write_report(
    path: Path,
    seasons: tuple[str, ...],
    selections: pd.DataFrame,
    competitions: pd.DataFrame,
    uncertainty: pd.DataFrame,
    candidate: AchievementReserveConfig,
    active: AchievementReserveConfig,
    decision: str,
    guardrails: dict[str, object],
) -> None:
    overall = uncertainty.loc[uncertainty["competition"].eq("ALL")].iloc[0]
    lines = [
        "# AO European Elo v2 Achievement Reserve Calibration",
        "",
        f"Exact-date seasons: `{seasons[0]}` through `{seasons[-1]}`.",
        "The comparator is the selected Power core, season carry and active goal layer.",
        "Normal match updates remain zero-sum; reserve is a separate non-zero-sum state.",
        "",
        "## Formula",
        "",
        "```text",
        "Advance Reserve = Base * Competition * Stage * (1 - P_advance)",
        "Trophy Reserve = Base * Competition * (1 - P_advance)",
        "AO Live Elo = Power Elo + Achievement Reserve",
        "```",
        "",
        "The expectation is frozen immediately before the tie's first match. Reserve is "
        "added only after the tie is decided and is visible from the next match onward.",
        "",
        "## Decision",
        "",
        f"- Result: `{decision}`",
        f"- Candidate base: `{candidate.reserve_base:.6f}`",
        f"- Candidate UEL / UECL: `{candidate.uel_multiplier:g}` / "
        f"`{candidate.uecl_multiplier:g}`",
        f"- Candidate profile / decay: `{candidate.stage_profile}` / "
        f"`{candidate.reserve_decay:g}`",
        f"- Active base: `{active.reserve_base:.6f}`",
        f"- Unseen fold wins: `{guardrails['fold_wins']}/6`",
        f"- Overall Brier difference: `{overall.mean_brier_difference:+.6f}`",
        f"- Clustered 95% CI: `[{overall.ci_95_lower:+.6f}, "
        f"{overall.ci_95_upper:+.6f}]`",
        f"- Ranking safe: `{guardrails['ranking_safe']}`",
        f"- Reserve cap safe: `{guardrails['reserve_cap_safe']}`",
        "",
        "| Competition | Matches | Brier difference | Log-loss difference |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in competitions.itertuples(index=False):
        lines.append(
            f"| {row.competition} | {row.matches} | {row.brier_difference:+.6f} | "
            f"{row.log_loss_difference:+.6f} |"
        )
    lines.extend(
        [
            "",
            "If disabled, competition and stage multipliers remain documented research "
            "candidates but add exactly zero reserve in production.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
