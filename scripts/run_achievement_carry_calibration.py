from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_backtest_dataset import normalize_name  # noqa: E402
from ao_elo.config import AOEuropeanEloConfig  # noqa: E402
from scripts.run_dynamic_core_calibration import (  # noqa: E402
    MAX_RATING_MOVE_GUARDRAIL,
    RANK_CORRELATION_FLOOR,
    DynamicCoreConfig,
    SeasonData,
    expanding_folds,
    expected_home_score,
    load_calibration_data,
)
from scripts.run_goal_margin_calibration import (  # noqa: E402
    read_full_core_config,
    validate_core_fold_contract,
)


STATIC_DATA_ROOT = ROOT / "data" / "backtest_stage_b_2018_2026"
EVENTS_PATH = ROOT / "data" / "dynamic_backtest_2018_2026" / "matches.csv"
CORE_OUTPUT_ROOT = ROOT / "output" / "dynamic_core_calibration_2018_2026"
OUTPUT_ROOT = ROOT / "output" / "achievement_carry_calibration_2018_2026"
POWER_CARRY_CANDIDATES = (0.0, 0.25, 0.50, 0.75, 0.85, 0.90, 1.0)
RESERVE_DECAY_CANDIDATES = (0.25, 0.50, 0.75, 1.0)
TROPHY_BASE_CANDIDATES = (20.0, 30.0, 40.0, 50.0, 60.0)
RESERVE_CAP = 80.0
COMPETITION_PRESTIGE = {"UCL": 1.0, "UEL": 0.65, "UECL": 0.45}


@dataclass(frozen=True, order=True)
class AchievementCarryConfig:
    power_carry: float
    reserve_decay: float
    ucl_trophy_base: float
    reserve_cap: float = RESERVE_CAP

    def validate(self) -> None:
        values = {
            "power_carry": self.power_carry,
            "reserve_decay": self.reserve_decay,
            "ucl_trophy_base": self.ucl_trophy_base,
            "reserve_cap": self.reserve_cap,
        }
        for name, value in values.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be non-negative and finite")
        if self.power_carry > 1 or self.reserve_decay > 1:
            raise ValueError("Carry and decay values must be in [0,1]")
        if self.reserve_cap <= 0:
            raise ValueError("reserve_cap must be positive")
        if self.ucl_trophy_base > self.reserve_cap:
            raise ValueError("UCL trophy base cannot exceed reserve cap")
        if self.ucl_trophy_base == 0 and self.reserve_decay != 0:
            raise ValueError("reserve_decay must be zero when trophy reserve is disabled")

    def trophy_bonus(self, competition: str) -> float:
        self.validate()
        try:
            return self.ucl_trophy_base * COMPETITION_PRESTIGE[competition]
        except KeyError as error:
            raise ValueError(f"Unknown competition: {competition}") from error


BASELINE = AchievementCarryConfig(0.0, 0.0, 0.0)
REFERENCE = AchievementCarryConfig(0.50, 0.50, 40.0)


@dataclass(frozen=True)
class CarrySeasonData:
    core: SeasonData
    club_keys: np.ndarray
    rounds: np.ndarray
    tie_decider_flags: np.ndarray
    advanced_team_ids: np.ndarray

    @property
    def season(self) -> str:
        return self.core.season


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate cross-season power carry and European trophy reserve"
    )
    parser.add_argument("--static-data-root", type=Path, default=STATIC_DATA_ROOT)
    parser.add_argument("--events-path", type=Path, default=EVENTS_PATH)
    parser.add_argument("--core-output-root", type=Path, default=CORE_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    datasets, identity_audit = load_carry_data(
        args.static_data_root.resolve(),
        args.events_path.resolve(),
    )
    seasons = tuple(data.season for data in datasets)
    folds = expanding_folds(seasons)
    core_output_root = args.core_output_root.resolve()
    core_selections = pd.read_csv(core_output_root / "fold_selections.csv")
    full_core = read_full_core_config(core_output_root / "full_candidate_metrics.csv")
    validate_core_fold_contract(core_selections, folds)
    candidates = candidate_grid()
    achievement_candidates = tuple(
        candidate for candidate in candidates if candidate.ucl_trophy_base > 0
    )
    carry_only_candidates = tuple(
        candidate
        for candidate in candidates
        if candidate.ucl_trophy_base == 0 and candidate.power_carry > 0
    )
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    selection_rows: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for fold_number, (train_seasons, test_season) in enumerate(folds, start=1):
        core_row = core_selections.loc[core_selections["fold"].eq(fold_number)].iloc[0]
        core_config = DynamicCoreConfig(
            float(core_row["selected_scale"]),
            float(core_row["selected_home_advantage"]),
            float(core_row["selected_k"]),
        )
        sequence_data = tuple(
            data for data in datasets if data.season in (*train_seasons, test_season)
        )
        train_data = tuple(data for data in datasets if data.season in train_seasons)
        selected, selected_train = select_candidate(train_data, core_config, candidates)
        achievement, achievement_train = select_candidate(
            train_data,
            core_config,
            achievement_candidates,
        )
        carry_only, carry_train = select_candidate(
            train_data,
            core_config,
            carry_only_candidates,
        )
        baseline_train, _, _ = evaluate_sequence(train_data, core_config, BASELINE)
        reference_train, _, _ = evaluate_sequence(train_data, core_config, REFERENCE)
        selection_rows.append(
            {
                "fold": fold_number,
                "train_seasons": ",".join(train_seasons),
                "test_season": test_season,
                "core_scale": core_config.elo_scale,
                "core_home_advantage": core_config.home_advantage,
                "core_k": core_config.k_factor,
                "selected_power_carry": selected.power_carry,
                "selected_reserve_decay": selected.reserve_decay,
                "selected_ucl_trophy_base": selected.ucl_trophy_base,
                "selected_train_brier_difference": (
                    selected_train["brier"] - baseline_train["brier"]
                ),
                "achievement_power_carry": achievement.power_carry,
                "achievement_reserve_decay": achievement.reserve_decay,
                "achievement_ucl_trophy_base": achievement.ucl_trophy_base,
                "achievement_train_brier_difference": (
                    achievement_train["brier"] - baseline_train["brier"]
                ),
                "carry_only_power_carry": carry_only.power_carry,
                "carry_only_train_brier_difference": (
                    carry_train["brier"] - baseline_train["brier"]
                ),
                "reference_train_brier_difference": (
                    reference_train["brier"] - baseline_train["brier"]
                ),
            }
        )

        model_predictions: dict[str, pd.DataFrame] = {}
        for model_name, config in (
            ("selected_layer", selected),
            ("achievement_challenger", achievement),
            ("carry_only_challenger", carry_only),
            ("reference_carry_trophy", REFERENCE),
            ("core_baseline", BASELINE),
        ):
            metrics, predictions, _ = evaluate_sequence(
                sequence_data,
                core_config,
                config,
                evaluation_seasons={test_season},
                return_predictions=True,
            )
            result_rows.append(
                {
                    "fold": fold_number,
                    "test_season": test_season,
                    "model": model_name,
                    "core_scale": core_config.elo_scale,
                    "core_home_advantage": core_config.home_advantage,
                    "core_k": core_config.k_factor,
                    "power_carry": config.power_carry,
                    "reserve_decay": config.reserve_decay,
                    "ucl_trophy_base": config.ucl_trophy_base,
                    **metrics,
                }
            )
            assert predictions is not None
            model_predictions[model_name] = predictions.rename(
                columns={
                    "expected_home_score": f"{model_name}_expected_home_score",
                    "brier_loss": f"{model_name}_brier_loss",
                    "log_loss": f"{model_name}_log_loss",
                }
            )
        joined = model_predictions["selected_layer"]
        for model_name in (
            "achievement_challenger",
            "carry_only_challenger",
            "reference_carry_trophy",
            "core_baseline",
        ):
            joined = joined.merge(
                model_predictions[model_name]
                [[
                    "match_id",
                    f"{model_name}_expected_home_score",
                    f"{model_name}_brier_loss",
                    f"{model_name}_log_loss",
                ]],
                on="match_id",
                validate="one_to_one",
            )
        joined.insert(0, "fold", fold_number)
        prediction_frames.append(joined)

    selections = pd.DataFrame(selection_rows)
    fold_results = pd.DataFrame(result_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    comparison_models = (
        "selected_layer",
        "achievement_challenger",
        "carry_only_challenger",
        "reference_carry_trophy",
    )
    summaries = {
        model: summarize_competitions(predictions, model)
        for model in comparison_models
    }
    uncertainties = {
        model: paired_uncertainty(predictions, model)
        for model in comparison_models
    }
    trophy_incremental_summary = summarize_competitions(
        predictions,
        "achievement_challenger",
        baseline_model="carry_only_challenger",
    )
    trophy_incremental_uncertainty = paired_uncertainty(
        predictions,
        "achievement_challenger",
        baseline_model="carry_only_challenger",
    )
    stability = parameter_stability(selections)

    final_config, final_metrics = select_candidate(datasets, full_core, candidates)
    final_achievement, final_achievement_metrics = select_candidate(
        datasets,
        full_core,
        achievement_candidates,
    )
    full_metrics = candidate_metrics(datasets, full_core, candidates)
    _, final_predictions, final_team_ratings = evaluate_sequence(
        datasets,
        full_core,
        final_config,
        return_predictions=True,
        return_team_ratings=True,
    )
    assert final_predictions is not None
    strength_profile = build_competition_strength_profile(final_predictions)
    bonus_table = build_bonus_table(final_achievement)
    decision = calibration_decision(
        selections,
        fold_results,
        uncertainties["carry_only_challenger"],
        trophy_incremental_uncertainty,
        final_config,
    )

    selections.to_csv(output_root / "fold_selections.csv", index=False)
    fold_results.to_csv(output_root / "fold_results.csv", index=False)
    predictions.to_csv(output_root / "unseen_match_predictions.csv", index=False)
    identity_audit.to_csv(output_root / "club_identity_audit.csv", index=False)
    stability.to_csv(output_root / "parameter_stability.csv", index=False)
    full_metrics.to_csv(output_root / "full_candidate_metrics.csv", index=False)
    final_team_ratings.to_csv(output_root / "final_candidate_team_ratings.csv", index=False)
    strength_profile.to_csv(output_root / "competition_strength_profile.csv", index=False)
    bonus_table.to_csv(output_root / "trophy_bonus_table.csv", index=False)
    for model, summary in summaries.items():
        summary.to_csv(output_root / f"{model}_competition_summary.csv", index=False)
    for model, uncertainty in uncertainties.items():
        uncertainty.to_csv(output_root / f"{model}_paired_uncertainty.csv", index=False)
    trophy_incremental_summary.to_csv(
        output_root / "trophy_incremental_competition_summary.csv", index=False
    )
    trophy_incremental_uncertainty.to_csv(
        output_root / "trophy_incremental_paired_uncertainty.csv", index=False
    )
    write_external_benchmark_requirements(
        output_root / "external_elo_benchmark_requirements.md"
    )
    write_report(
        output_root / "calibration_report.md",
        seasons,
        selections,
        fold_results,
        summaries,
        uncertainties,
        trophy_incremental_summary,
        trophy_incremental_uncertainty,
        stability,
        final_config,
        final_metrics,
        final_achievement,
        final_achievement_metrics,
        bonus_table,
        strength_profile,
        identity_audit,
        decision,
    )

    print("AO achievement reserve and season-carry calibration")
    print(f"Seasons: {len(seasons)}")
    print(f"Matches: {sum(len(data.core.match_ids) for data in datasets)}")
    print(f"Global clubs: {identity_audit['club_key'].nunique()}")
    print(f"Candidates: {len(candidates)}")
    print(
        "Full-data candidate: "
        f"carry={final_config.power_carry:g}, decay={final_config.reserve_decay:g}, "
        f"UCL trophy={final_config.ucl_trophy_base:g}"
    )
    print(f"Decision: {decision}")
    print(f"Report: {output_root / 'calibration_report.md'}")


def club_key(team_name: str, country_code: str) -> str:
    normalized = normalize_name(team_name)
    country = str(country_code).strip().upper()
    if not normalized or not country:
        raise ValueError("Club identity requires team_name and country_code")
    return f"{country}::{normalized}"


def load_carry_data(
    static_root: Path,
    events_path: Path,
    static_config: AOEuropeanEloConfig | None = None,
    *,
    require_exact_utc: bool = False,
) -> tuple[tuple[CarrySeasonData, ...], pd.DataFrame]:
    core_datasets = load_calibration_data(
        static_root,
        events_path,
        static_config,
        require_exact_utc=require_exact_utc,
    )
    events = pd.read_csv(events_path).sort_values(["season", "event_order"])
    required = {
        "match_id", "season", "round", "is_tie_decider", "advanced_team_id",
        "home_team_id", "away_team_id", "home_team_name", "away_team_name",
    }
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"Carry event data missing columns: {missing}")
    event_index = events.set_index("match_id")
    if not event_index.index.is_unique:
        raise ValueError("Carry event match_id must be unique")
    result: list[CarrySeasonData] = []
    audit_rows: list[dict[str, object]] = []
    for core in core_datasets:
        folder = static_root / core.season.replace("/", "-")
        teams = pd.read_csv(folder / "teams.csv")
        if teams["team_id"].duplicated().any():
            raise ValueError(f"{core.season}: duplicate team_id in teams.csv")
        teams["club_key"] = [
            club_key(name, country)
            for name, country in zip(teams["team_name"], teams["country_code"])
        ]
        if teams["club_key"].duplicated().any():
            duplicated = teams.loc[teams["club_key"].duplicated(False), "club_key"].tolist()
            raise ValueError(f"{core.season}: duplicate global club keys: {duplicated[:3]}")
        keys = np.full(len(core.initial_ratings), None, dtype=object)
        for row in teams.itertuples(index=False):
            team_id = int(row.team_id)
            if team_id >= len(keys):
                raise ValueError(f"{core.season}: team_id exceeds rating array")
            keys[team_id] = row.club_key
            audit_rows.append(
                {
                    "season": core.season,
                    "local_team_id": team_id,
                    "team_name": row.team_name,
                    "country_code": row.country_code,
                    "club_key": row.club_key,
                }
            )
        if any(keys[team_id] is None for team_id in core.active_team_ids):
            raise ValueError(f"{core.season}: active event team lacks a global club key")
        aligned = event_index.loc[core.match_ids]
        for row in aligned.itertuples(index=False):
            home_key = keys[int(row.home_team_id)]
            away_key = keys[int(row.away_team_id)]
            if normalize_name(row.home_team_name) not in home_key:
                raise ValueError(f"{core.season}: home club identity mismatch")
            if normalize_name(row.away_team_name) not in away_key:
                raise ValueError(f"{core.season}: away club identity mismatch")
        final_flags = aligned["round"].eq("Final") & aligned["is_tie_decider"].astype(bool)
        if final_flags.any() and aligned.loc[final_flags, "advanced_team_id"].isna().any():
            raise ValueError(f"{core.season}: final winner is missing")
        result.append(
            CarrySeasonData(
                core=core,
                club_keys=keys,
                rounds=aligned["round"].to_numpy(str),
                tie_decider_flags=aligned["is_tie_decider"].to_numpy(bool),
                advanced_team_ids=aligned["advanced_team_id"].fillna(-1).to_numpy(int),
            )
        )
    audit = pd.DataFrame(audit_rows)
    country_counts = audit.groupby("club_key")["country_code"].nunique()
    if country_counts.gt(1).any():
        raise ValueError("A global club key maps to multiple countries")
    return tuple(result), audit


def candidate_grid() -> tuple[AchievementCarryConfig, ...]:
    candidates = {BASELINE}
    candidates.update(
        AchievementCarryConfig(carry, 0.0, 0.0)
        for carry in POWER_CARRY_CANDIDATES
        if carry > 0
    )
    candidates.update(
        AchievementCarryConfig(carry, decay, trophy)
        for carry in POWER_CARRY_CANDIDATES
        for decay in RESERVE_DECAY_CANDIDATES
        for trophy in TROPHY_BASE_CANDIDATES
    )
    if REFERENCE not in candidates:
        raise AssertionError("Reference carry/trophy candidate is missing")
    for candidate in candidates:
        candidate.validate()
    return tuple(sorted(candidates))


def evaluate_sequence(
    datasets: tuple[CarrySeasonData, ...],
    core_config: DynamicCoreConfig,
    config: AchievementCarryConfig,
    *,
    evaluation_seasons: set[str] | None = None,
    return_predictions: bool = False,
    return_team_ratings: bool = False,
) -> tuple[dict[str, float | int], pd.DataFrame | None, pd.DataFrame]:
    core_config.validate()
    config.validate()
    evaluation = evaluation_seasons or {data.season for data in datasets}
    previous_power: dict[str, float] = {}
    previous_reserve: dict[str, float] = {}
    metric_rows: list[dict[str, float | int]] = []
    prediction_rows: list[dict[str, object]] = []
    team_rows: list[dict[str, object]] = []
    for data in datasets:
        core = data.core
        power = core.initial_ratings.copy()
        reserve = np.zeros_like(power)
        carried_clubs = 0
        for team_id in core.active_team_ids:
            key = str(data.club_keys[team_id])
            if key in previous_power:
                power[team_id] = (
                    (1.0 - config.power_carry) * core.initial_ratings[team_id]
                    + config.power_carry * previous_power[key]
                )
                reserve[team_id] = min(
                    config.reserve_cap,
                    config.reserve_decay * previous_reserve.get(key, 0.0),
                )
                carried_clubs += 1
        brier_sum = 0.0
        log_loss_sum = 0.0
        trophy_total = 0.0
        for index, (home_id, away_id, actual, neutral, competition) in enumerate(
            zip(
                core.home_team_ids,
                core.away_team_ids,
                core.actual_home_scores,
                core.neutral_flags,
                core.competitions,
            )
        ):
            probability = float(
                np.clip(
                    expected_home_score(
                        power[home_id] + reserve[home_id],
                        power[away_id] + reserve[away_id],
                        core_config,
                        neutral=bool(neutral),
                    ),
                    1e-12,
                    1.0 - 1e-12,
                )
            )
            brier_loss = (probability - actual) ** 2
            log_loss = -(
                actual * math.log(probability)
                + (1.0 - actual) * math.log(1.0 - probability)
            )
            delta = core_config.k_factor * (actual - probability)
            power[home_id] += delta
            power[away_id] -= delta
            trophy_added = 0.0
            if data.rounds[index] == "Final" and data.tie_decider_flags[index]:
                winner_id = int(data.advanced_team_ids[index])
                if winner_id not in (home_id, away_id):
                    raise ValueError(f"{data.season}/{core.match_ids[index]}: invalid final winner")
                available = config.reserve_cap - reserve[winner_id]
                trophy_added = max(
                    0.0,
                    min(available, config.trophy_bonus(str(competition))),
                )
                reserve[winner_id] += trophy_added
                trophy_total += trophy_added
            if data.season in evaluation:
                brier_sum += brier_loss
                log_loss_sum += log_loss
                if return_predictions:
                    prediction_rows.append(
                        {
                            "match_id": core.match_ids[index],
                            "season": data.season,
                            "competition": competition,
                            "round": data.rounds[index],
                            "actual_home_score": actual,
                            "home_power": power[home_id] - delta,
                            "away_power": power[away_id] + delta,
                            "home_reserve": reserve[home_id] - (
                                trophy_added if winner_id_if_home(
                                    data, index, int(home_id)
                                ) else 0.0
                            ),
                            "away_reserve": reserve[away_id] - (
                                trophy_added if winner_id_if_home(
                                    data, index, int(away_id)
                                ) else 0.0
                            ),
                            "expected_home_score": probability,
                            "brier_loss": brier_loss,
                            "log_loss": log_loss,
                            "power_delta": delta,
                            "trophy_bonus_added_after_match": trophy_added,
                        }
                    )
        active = core.active_team_ids
        total_end = power[active] + reserve[active]
        reference_start = core.initial_ratings[active]
        changes = total_end - reference_start
        if np.allclose(reference_start, total_end):
            rank_correlation = 1.0
        elif np.ptp(reference_start) == 0 or np.ptp(total_end) == 0:
            rank_correlation = 0.0
        else:
            rank_correlation = pd.Series(total_end).corr(
                pd.Series(reference_start), method="spearman"
            )
            if pd.isna(rank_correlation):
                rank_correlation = 0.0
        if data.season in evaluation:
            metric_rows.append(
                {
                    "matches": len(core.match_ids),
                    "brier": brier_sum / len(core.match_ids),
                    "log_loss": log_loss_sum / len(core.match_ids),
                    "carried_clubs": carried_clubs,
                    "trophy_bonus_total": trophy_total,
                    "mean_reserve": float(np.mean(reserve[active])),
                    "max_reserve": float(np.max(reserve[active])),
                    "mean_rating_change": float(np.mean(changes)),
                    "rating_change_std": float(np.std(changes)),
                    "max_abs_rating_change": float(np.max(np.abs(changes))),
                    "start_end_rank_correlation": float(rank_correlation),
                }
            )
            if return_team_ratings:
                for team_id in active:
                    team_rows.append(
                        {
                            "season": data.season,
                            "club_key": data.club_keys[team_id],
                            "local_team_id": int(team_id),
                            "ao_first_elo": core.initial_ratings[team_id],
                            "power_elo_end": power[team_id],
                            "achievement_reserve_end": reserve[team_id],
                            "ao_live_rating_end": power[team_id] + reserve[team_id],
                        }
                    )
        previous_power = {
            str(data.club_keys[team_id]): float(power[team_id]) for team_id in active
        }
        previous_reserve = {
            str(data.club_keys[team_id]): float(reserve[team_id]) for team_id in active
        }
    if not metric_rows:
        raise ValueError("No evaluation seasons were processed")
    matches = sum(int(row["matches"]) for row in metric_rows)
    aggregate: dict[str, float | int] = {
        "matches": matches,
        "brier": sum(float(row["brier"]) * int(row["matches"]) for row in metric_rows) / matches,
        "log_loss": sum(float(row["log_loss"]) * int(row["matches"]) for row in metric_rows) / matches,
        "carried_clubs": sum(int(row["carried_clubs"]) for row in metric_rows),
        "trophy_bonus_total": sum(float(row["trophy_bonus_total"]) for row in metric_rows),
        "mean_reserve": float(np.mean([row["mean_reserve"] for row in metric_rows])),
        "max_reserve": float(max(row["max_reserve"] for row in metric_rows)),
        "mean_rating_change": float(np.mean([row["mean_rating_change"] for row in metric_rows])),
        "rating_change_std": float(np.mean([row["rating_change_std"] for row in metric_rows])),
        "max_abs_rating_change": float(max(row["max_abs_rating_change"] for row in metric_rows)),
        "start_end_rank_correlation": float(min(row["start_end_rank_correlation"] for row in metric_rows)),
    }
    predictions = pd.DataFrame(prediction_rows) if return_predictions else None
    return aggregate, predictions, pd.DataFrame(team_rows)


def winner_id_if_home(data: CarrySeasonData, index: int, team_id: int) -> bool:
    return bool(
        data.rounds[index] == "Final"
        and data.tie_decider_flags[index]
        and int(data.advanced_team_ids[index]) == team_id
    )


def candidate_metrics(
    datasets: tuple[CarrySeasonData, ...],
    core_config: DynamicCoreConfig,
    candidates: tuple[AchievementCarryConfig, ...],
) -> pd.DataFrame:
    rows = []
    for config in candidates:
        metrics, _, _ = evaluate_sequence(datasets, core_config, config)
        rows.append(
            {
                "power_carry": config.power_carry,
                "reserve_decay": config.reserve_decay,
                "ucl_trophy_base": config.ucl_trophy_base,
                "distance_from_reference": (
                    abs(config.power_carry - REFERENCE.power_carry)
                    + abs(config.reserve_decay - REFERENCE.reserve_decay)
                    + abs(config.ucl_trophy_base - REFERENCE.ucl_trophy_base) / 40.0
                ),
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["brier", "log_loss", "ucl_trophy_base", "distance_from_reference"]
    ).reset_index(drop=True)


def select_candidate(
    datasets: tuple[CarrySeasonData, ...],
    core_config: DynamicCoreConfig,
    candidates: tuple[AchievementCarryConfig, ...],
) -> tuple[AchievementCarryConfig, dict[str, float | int]]:
    rows = candidate_metrics(datasets, core_config, candidates)
    selected = rows.iloc[0]
    config = AchievementCarryConfig(
        float(selected["power_carry"]),
        float(selected["reserve_decay"]),
        float(selected["ucl_trophy_base"]),
    )
    excluded = {
        "power_carry", "reserve_decay", "ucl_trophy_base", "distance_from_reference",
    }
    return config, {column: selected[column] for column in rows.columns if column not in excluded}


def summarize_competitions(
    predictions: pd.DataFrame,
    model: str,
    *,
    baseline_model: str = "core_baseline",
) -> pd.DataFrame:
    rows = []
    for competition, data in predictions.groupby("competition"):
        rows.append(
            {
                "competition": competition,
                "matches": len(data),
                "brier_difference": (
                    data[f"{model}_brier_loss"].mean()
                    - data[f"{baseline_model}_brier_loss"].mean()
                ),
                "log_loss_difference": (
                    data[f"{model}_log_loss"].mean()
                    - data[f"{baseline_model}_log_loss"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def paired_uncertainty(
    predictions: pd.DataFrame,
    model: str,
    *,
    baseline_model: str = "core_baseline",
    bootstrap_samples: int = 4000,
    seed: int = 20260715,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    groups = [("ALL", predictions), *predictions.groupby("competition")]
    for competition, data in groups:
        differences = (
            data[f"{model}_brier_loss"] - data[f"{baseline_model}_brier_loss"]
        ).to_numpy(float)
        sampled = rng.choice(differences, size=(bootstrap_samples, len(differences)), replace=True)
        means = sampled.mean(axis=1)
        lower, upper = np.quantile(means, (0.025, 0.975))
        rows.append(
            {
                "competition": competition,
                "matches": len(differences),
                "mean_brier_difference": differences.mean(),
                "ci_95_lower": lower,
                "ci_95_upper": upper,
                "reliable_improvement": bool(upper < 0),
                "reliable_harm": bool(lower > 0),
            }
        )
    return pd.DataFrame(rows)


def parameter_stability(selections: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in (
        "selected_power_carry",
        "selected_reserve_decay",
        "selected_ucl_trophy_base",
        "carry_only_power_carry",
        "achievement_power_carry",
        "achievement_reserve_decay",
        "achievement_ucl_trophy_base",
    ):
        counts = selections[column].value_counts().sort_values(ascending=False)
        rows.append(
            {
                "parameter": column,
                "mode": float(counts.index[0]),
                "mode_count": int(counts.iloc[0]),
                "folds": len(selections),
                "mode_share": float(counts.iloc[0] / len(selections)),
                "min": float(selections[column].min()),
                "max": float(selections[column].max()),
            }
        )
    return pd.DataFrame(rows)


def build_bonus_table(config: AchievementCarryConfig) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "competition": ("UCL", "UEL", "UECL"),
            "prestige_reference": (1.0, 0.65, 0.45),
            "trophy_bonus": (
                config.trophy_bonus("UCL"),
                config.trophy_bonus("UEL"),
                config.trophy_bonus("UECL"),
            ),
            "reserve_decay": config.reserve_decay,
            "reserve_cap": config.reserve_cap,
        }
    )


def build_competition_strength_profile(predictions: pd.DataFrame) -> pd.DataFrame:
    data = predictions.copy()
    data["phase"] = data["round"].map(competition_phase)
    home = data[["competition", "phase", "home_power", "home_reserve"]].rename(
        columns={"home_power": "power", "home_reserve": "reserve"}
    )
    away = data[["competition", "phase", "away_power", "away_reserve"]].rename(
        columns={"away_power": "power", "away_reserve": "reserve"}
    )
    participants = pd.concat((home, away), ignore_index=True)
    participants["live_rating"] = participants["power"] + participants["reserve"]
    rows = []
    for (competition, phase), group in participants.groupby(["competition", "phase"]):
        values = group["live_rating"].to_numpy(float)
        rows.append(
            {
                "competition": competition,
                "phase": phase,
                "team_match_observations": len(values),
                "mean_live_rating": float(np.mean(values)),
                "p25_live_rating": float(np.quantile(values, 0.25)),
                "median_live_rating": float(np.median(values)),
                "p75_live_rating": float(np.quantile(values, 0.75)),
            }
        )
    return pd.DataFrame(rows).sort_values(["phase", "mean_live_rating"], ascending=[True, False])


def competition_phase(round_name: str) -> str:
    if round_name in {
        "Preliminary Round",
        "1st Qualifying Round",
        "2nd Qualifying Round",
        "3rd Qualifying Round",
        "Qualifying Play-off Round",
    }:
        return "QUALIFYING"
    if round_name in {"Group Stage", "League Stage"}:
        return "MAIN_STAGE"
    return "KNOCKOUT"


def calibration_decision(
    selections: pd.DataFrame,
    fold_results: pd.DataFrame,
    carry_uncertainty: pd.DataFrame,
    trophy_incremental_uncertainty: pd.DataFrame,
    final_config: AchievementCarryConfig,
) -> str:
    pivot = fold_results.pivot(index="fold", columns="model", values="brier")
    carry_wins = int((pivot["carry_only_challenger"] < pivot["core_baseline"]).sum())
    trophy_wins = int(
        (pivot["achievement_challenger"] < pivot["carry_only_challenger"]).sum()
    )
    carry_overall = carry_uncertainty.loc[
        carry_uncertainty["competition"].eq("ALL")
    ].iloc[0]
    trophy_overall = trophy_incremental_uncertainty.loc[
        trophy_incremental_uncertainty["competition"].eq("ALL")
    ].iloc[0]
    no_trophy_harm = not bool(
        trophy_incremental_uncertainty.loc[
            trophy_incremental_uncertainty["competition"].ne("ALL"),
            "reliable_harm",
        ].any()
    )
    selected_rows = fold_results.loc[
        fold_results["model"].eq("achievement_challenger")
    ]
    ranking_safe = bool(
        selected_rows["start_end_rank_correlation"].ge(RANK_CORRELATION_FLOOR).all()
        and selected_rows["max_abs_rating_change"].le(MAX_RATING_MOVE_GUARDRAIL).all()
        and selected_rows["max_reserve"].le(RESERVE_CAP).all()
    )
    carry_confirmed = carry_wins >= 5 and bool(carry_overall["reliable_improvement"])
    trophy_confirmed = (
        trophy_wins >= 5
        and bool(trophy_overall["reliable_improvement"])
        and no_trophy_harm
        and ranking_safe
    )
    if carry_confirmed and trophy_confirmed:
        return "PROVISIONAL_ACCEPT_ACHIEVEMENT_RESERVE_AND_CARRY"
    if carry_confirmed:
        if final_config.power_carry == max(POWER_CARRY_CANDIDATES):
            return "POWER_CARRY_CONFIRMED_AT_LOGICAL_BOUND_TROPHY_REJECTED"
        return "PROVISIONAL_ACCEPT_POWER_CARRY_TROPHY_REJECTED"
    return "REJECT_ACHIEVEMENT_RESERVE_AND_CARRY_KEEP_RESET_CORE"


def write_external_benchmark_requirements(path: Path) -> None:
    text = "\n".join(
        [
            "# External Historical Elo Benchmark Requirements",
            "",
            "An external Elo comparator can validate absolute rating quality, but it cannot",
            "replace the paired ablation that isolates one AO model layer.",
            "",
            "Required before a leakage-safe comparison:",
            "",
            "- exact match date or timestamp for every event;",
            "- a historical pre-match club Elo snapshot from one consistent provider;",
            "- a durable provider-club identifier mapped to AO global club_key;",
            "- documented treatment of neutral venues, extra time and promoted/new clubs;",
            "- snapshots captured before kickoff, never end-of-season or current ratings.",
            "",
            "The current event set intentionally does not assert exact dates, so joining a",
            "daily external Elo history now would risk look-ahead leakage. The benchmark must",
            "wait until exact dates and historical snapshots are added and audited.",
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")


def write_report(
    path: Path,
    seasons: tuple[str, ...],
    selections: pd.DataFrame,
    fold_results: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
    uncertainties: dict[str, pd.DataFrame],
    trophy_incremental_summary: pd.DataFrame,
    trophy_incremental_uncertainty: pd.DataFrame,
    stability: pd.DataFrame,
    final_config: AchievementCarryConfig,
    final_metrics: dict[str, float | int],
    final_achievement: AchievementCarryConfig,
    final_achievement_metrics: dict[str, float | int],
    bonus_table: pd.DataFrame,
    strength_profile: pd.DataFrame,
    identity_audit: pd.DataFrame,
    decision: str,
) -> None:
    pivot = fold_results.pivot(index="fold", columns="model", values="brier")
    models = (
        "selected_layer",
        "achievement_challenger",
        "carry_only_challenger",
        "reference_carry_trophy",
    )
    wins = {model: int((pivot[model] < pivot["core_baseline"]).sum()) for model in models}
    trophy_incremental_wins = int(
        (pivot["achievement_challenger"] < pivot["carry_only_challenger"]).sum()
    )
    trophy_incremental_overall = trophy_incremental_uncertainty.loc[
        trophy_incremental_uncertainty["competition"].eq("ALL")
    ].iloc[0]
    selection_rows = [
        f"| {row.fold} | {row.test_season} | {row.selected_power_carry:g} | "
        f"{row.selected_reserve_decay:g} | {row.selected_ucl_trophy_base:g} | "
        f"{row.achievement_power_carry:g} | {row.achievement_reserve_decay:g} | "
        f"{row.achievement_ucl_trophy_base:g} |"
        for row in selections.itertuples(index=False)
    ]
    comparison_rows = []
    for model, label in (
        ("selected_layer", "Nested selected"),
        ("achievement_challenger", "Forced achievement"),
        ("carry_only_challenger", "Carry only"),
        ("reference_carry_trophy", "Reference .5/.5/40"),
    ):
        overall = uncertainties[model].loc[
            uncertainties[model]["competition"].eq("ALL")
        ].iloc[0]
        comparison_rows.append(
            f"| {label} | {wins[model]}/6 | {overall.mean_brier_difference:.6f} | "
            f"{overall.ci_95_lower:.6f} | {overall.ci_95_upper:.6f} |"
        )
    segment_rows = []
    for model, label in (
        ("achievement_challenger", "Forced achievement"),
        ("reference_carry_trophy", "Reference"),
    ):
        for row in summaries[model].itertuples(index=False):
            segment_rows.append(
                f"| {label} | {row.competition} | {row.matches} | "
                f"{row.brier_difference:.6f} | {row.log_loss_difference:.6f} |"
            )
    trophy_incremental_rows = [
        f"| {row.competition} | {row.matches} | {row.brier_difference:.6f} | "
        f"{row.log_loss_difference:.6f} |"
        for row in trophy_incremental_summary.itertuples(index=False)
    ]
    stability_rows = [
        f"| {row.parameter} | {row.mode:g} | {row.mode_count}/{row.folds} | "
        f"{row.min:g}-{row.max:g} |"
        for row in stability.itertuples(index=False)
    ]
    bonus_rows = [
        f"| {row.competition} | {row.prestige_reference:g} | {row.trophy_bonus:g} | "
        f"{row.reserve_decay:g} | {row.reserve_cap:g} |"
        for row in bonus_table.itertuples(index=False)
    ]
    strength_rows = [
        f"| {row.phase} | {row.competition} | {row.team_match_observations} | "
        f"{row.mean_live_rating:.1f} | {row.median_live_rating:.1f} | "
        f"{row.p25_live_rating:.1f}-{row.p75_live_rating:.1f} |"
        for row in strength_profile.itertuples(index=False)
    ]
    text = "\n".join(
        [
            "# AO Achievement Reserve and Season Carry Calibration",
            "",
            f"Decision: **{decision}**",
            "",
            "## Scope",
            "",
            f"Seasons: {seasons[0]} through {seasons[-1]}; outer folds: {len(selections)}.",
            f"Global identity audit: {len(identity_audit)} team-season rows and ",
            f"{identity_audit['club_key'].nunique()} persistent clubs.",
            "Match results update only zero-sum Power Elo. Final winners receive a separate",
            "non-zero-sum Achievement Reserve. Power and reserve can influence the next",
            "season only through independently controlled carry and decay parameters.",
            "",
            "```text",
            "Power_start = (1-carry) * AO_First_current + carry * Power_end_previous",
            "Reserve_start = decay * Reserve_end_previous",
            "AO_live = Power + Reserve",
            "Trophy_bonus = UCL_base * competition_reference  # capped reserve",
            "```",
            "",
            "Competition references are fixed at UCL=1.00, UEL=0.65 and UECL=0.45 in this",
            "run. They are not calibrated unless the reserve layer first proves useful.",
            "",
            "## Fold Selections",
            "",
            "| Fold | Unseen | Selected carry | Selected decay | Selected trophy | Achievement carry | Achievement decay | Achievement trophy |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            *selection_rows,
            "",
            "## Unseen Comparison",
            "",
            "| Model | Fold wins | Mean Brier difference | CI lower | CI upper |",
            "| --- | ---: | ---: | ---: | ---: |",
            *comparison_rows,
            "",
            "## Isolated Trophy Contribution",
            "",
            "The achievement challenger is compared directly with the fold-selected",
            "carry-only challenger so carry improvement cannot be attributed to trophies.",
            "",
            f"Trophy beat matched carry-only in **{trophy_incremental_wins}/{len(selections)}** folds.",
            f"Incremental Brier difference: {trophy_incremental_overall.mean_brier_difference:.6f} ",
            f"(95% CI {trophy_incremental_overall.ci_95_lower:.6f} to "
            f"{trophy_incremental_overall.ci_95_upper:.6f}).",
            "",
            "| Competition | Matches | Trophy vs carry Brier difference | Log-loss difference |",
            "| --- | ---: | ---: | ---: |",
            *trophy_incremental_rows,
            "",
            "## Competition Segments",
            "",
            "| Model | Competition | Matches | Brier difference | Log-loss difference |",
            "| --- | --- | ---: | ---: | ---: |",
            *segment_rows,
            "",
            "## Parameter Stability",
            "",
            "| Parameter | Mode | Fold frequency | Range |",
            "| --- | ---: | ---: | ---: |",
            *stability_rows,
            "",
            "## Full-Data Results",
            "",
            f"All-candidate selection: `carry={final_config.power_carry:g}`, ",
            f"`decay={final_config.reserve_decay:g}`, ",
            f"`UCL trophy={final_config.ucl_trophy_base:g}`; ",
            f"Brier={float(final_metrics['brier']):.6f}.",
            f"Best forced achievement: `carry={final_achievement.power_carry:g}`, ",
            f"`decay={final_achievement.reserve_decay:g}`, ",
            f"`UCL trophy={final_achievement.ucl_trophy_base:g}`; ",
            f"Brier={float(final_achievement_metrics['brier']):.6f}.",
            "",
            "| Competition | Reference | Trophy bonus | Decay | Reserve cap |",
            "| --- | ---: | ---: | ---: | ---: |",
            *bonus_rows,
            "",
            "## Observed Competition Strength",
            "",
            "This table describes the pre-match AO Live Rating distribution of actual",
            "participants. It measures the strength of the field, not a prestige bonus.",
            "",
            "| Phase | Competition | Team-match observations | Mean | Median | IQR |",
            "| --- | --- | ---: | ---: | ---: | --- |",
            *strength_rows,
            "",
            "## External Elo Benchmark",
            "",
            "The paired AO ablation remains valid without an external Elo because every",
            "candidate uses the same matches and initial ratings. An external historical Elo",
            "would still improve absolute validation and expose initial-rating bias. It cannot",
            "be joined safely until exact match dates and pre-match historical snapshots are",
            "available; using current or end-of-season Elo would introduce look-ahead leakage.",
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
