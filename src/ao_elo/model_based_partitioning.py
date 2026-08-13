from __future__ import annotations

"""Model-based recursive partitioning for Domestic Surprise research.

The fitted leaf parameter has a deliberately narrow interpretation: zero keeps
the current Domestic Surprise adjustment, while one releases the full increment
from the current adjustment to the global-strong research adjustment.
"""

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


NUMERIC_PARTITION_VARIABLES = (
    "league_strength",
    "rating_percentile",
    "effective_european_exposure",
)
CATEGORICAL_PARTITION_VARIABLES = ("competition",)
DEFAULT_PARTITION_VARIABLES = NUMERIC_PARTITION_VARIABLES + CATEGORICAL_PARTITION_VARIABLES


@dataclass(frozen=True)
class MOBConfig:
    partition_variables: tuple[str, ...] = DEFAULT_PARTITION_VARIABLES
    significance_level: float = 0.05
    permutations: int = 2000
    random_seed: int = 20260812
    max_depth: int = 2
    max_leaves: int = 4
    minimum_leaf_team_seasons: int = 80
    minimum_leaf_seasons: int = 2
    minimum_nonzero_increment: int = 30
    theta_lower: float = 0.0
    theta_upper: float = 1.25
    target_match_shrinkage: float = 6.0
    final_adjustment_cap: float = 60.0
    minimum_era_history_seasons: int = 2

    def validate(self) -> None:
        unknown = set(self.partition_variables) - set(DEFAULT_PARTITION_VARIABLES)
        if unknown:
            raise ValueError(f"Unsupported MOB partition variables: {sorted(unknown)}")
        if len(set(self.partition_variables)) != len(self.partition_variables):
            raise ValueError("MOB partition variables must be unique")
        if not 0.0 < self.significance_level < 1.0:
            raise ValueError("significance_level must be in (0,1)")
        if self.permutations < 1:
            raise ValueError("permutations must be positive")
        if self.max_depth < 0 or self.max_leaves < 1:
            raise ValueError("MOB tree limits are invalid")
        if self.minimum_leaf_team_seasons < 2:
            raise ValueError("minimum_leaf_team_seasons must be at least two")
        if self.minimum_leaf_seasons < 1 or self.minimum_nonzero_increment < 1:
            raise ValueError("MOB child support limits must be positive")
        if not 0.0 <= self.theta_lower < self.theta_upper:
            raise ValueError("theta bounds are invalid")
        if self.target_match_shrinkage <= 0.0 or not math.isfinite(self.target_match_shrinkage):
            raise ValueError("target_match_shrinkage must be positive and finite")
        if self.final_adjustment_cap <= 0.0 or not math.isfinite(self.final_adjustment_cap):
            raise ValueError("final_adjustment_cap must be positive and finite")
        if self.minimum_era_history_seasons < 1:
            raise ValueError("minimum_era_history_seasons must be positive")


@dataclass(frozen=True)
class ParameterInstabilityResult:
    node_id: str
    variable: str
    variable_type: str
    statistic: float
    raw_p_value: float
    adjusted_p_value: float
    threshold: float | None = None
    left_categories: tuple[str, ...] = ()
    right_categories: tuple[str, ...] = ()
    left_observations: int = 0
    right_observations: int = 0
    weighted_rss_after_split: float | None = None
    valid: bool = True
    reason: str = "OK"

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["left_categories"] = "|".join(self.left_categories)
        payload["right_categories"] = "|".join(self.right_categories)
        return payload


@dataclass
class MOBNode:
    node_id: str
    depth: int
    observations: int
    seasons: tuple[str, ...]
    theta: float
    theta_standard_error: float
    diagnostic_intercept: float
    weighted_rss: float
    nonzero_increment_observations: int
    split_variable: str | None = None
    split_type: str | None = None
    split_threshold: float | None = None
    left_categories: tuple[str, ...] = ()
    raw_p_value: float | None = None
    adjusted_p_value: float | None = None
    stop_reason: str = "LEAF"
    left: MOBNode | None = None
    right: MOBNode | None = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "depth": self.depth,
            "observations": self.observations,
            "seasons": list(self.seasons),
            "theta": self.theta,
            "theta_standard_error": self.theta_standard_error,
            "diagnostic_intercept": self.diagnostic_intercept,
            "weighted_rss": self.weighted_rss,
            "nonzero_increment_observations": self.nonzero_increment_observations,
            "split_variable": self.split_variable,
            "split_type": self.split_type,
            "split_threshold": self.split_threshold,
            "left_categories": list(self.left_categories),
            "raw_p_value": self.raw_p_value,
            "adjusted_p_value": self.adjusted_p_value,
            "stop_reason": self.stop_reason,
            "left": None if self.left is None else self.left.as_dict(),
            "right": None if self.right is None else self.right.as_dict(),
        }


@dataclass(frozen=True)
class MOBPrediction:
    theta: float
    leaf_id: str
    fallback: bool
    fallback_reason: str


@dataclass
class MOBTree:
    config: MOBConfig
    root: MOBNode
    calibration_intercept: float
    calibration_slope: float
    rating_mean: float
    rating_scale: float
    training_seasons: tuple[str, ...]
    training_categories: dict[str, tuple[str, ...]]
    instability_tests: list[ParameterInstabilityResult] = field(default_factory=list)
    fit_status: str = "FITTED"

    def predict(self, row: Mapping[str, Any]) -> MOBPrediction:
        for variable, levels in self.training_categories.items():
            value = str(row.get(variable, ""))
            if value not in levels:
                return MOBPrediction(0.0, "CURRENT_FALLBACK", True, f"UNSEEN_{variable.upper()}")

        node = self.root
        while not node.is_leaf:
            variable = str(node.split_variable)
            value = row.get(variable)
            if node.split_type == "numeric":
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    return MOBPrediction(0.0, "CURRENT_FALLBACK", True, f"INVALID_{variable.upper()}")
                if not math.isfinite(number):
                    return MOBPrediction(0.0, "CURRENT_FALLBACK", True, f"INVALID_{variable.upper()}")
                node = node.left if number <= float(node.split_threshold) else node.right
            else:
                category = str(value)
                node = node.left if category in node.left_categories else node.right
            if node is None:
                return MOBPrediction(0.0, "CURRENT_FALLBACK", True, "INVALID_TREE_PATH")
        return MOBPrediction(float(node.theta), node.node_id, False, "")

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": "DOMESTIC_SURPRISE_MOB",
            "fit_status": self.fit_status,
            "config": asdict(self.config),
            "calibration": {
                "intercept": self.calibration_intercept,
                "slope": self.calibration_slope,
                "rating_mean": self.rating_mean,
                "rating_scale": self.rating_scale,
            },
            "training_seasons": list(self.training_seasons),
            "training_categories": {
                key: list(value) for key, value in sorted(self.training_categories.items())
            },
            "root": self.root.as_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True, allow_nan=False)


@dataclass(frozen=True)
class _SplitCandidate:
    variable: str
    variable_type: str
    statistic: float
    raw_p_value: float
    threshold: float | None
    left_categories: tuple[str, ...]
    right_categories: tuple[str, ...]
    left_indices: np.ndarray
    right_indices: np.ndarray
    weighted_rss_after_split: float


def _stable_seed(base: int, *parts: str) -> int:
    digest = hashlib.sha256("|".join((str(base), *parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[name] = running
    return adjusted


def _validate_training_frame(frame: pd.DataFrame, config: MOBConfig) -> pd.DataFrame:
    required = {
        "season",
        "team_id",
        "target_score",
        "target_matches",
        "current_initial_rating",
        "strong_increment",
        *config.partition_variables,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"MOB training frame is missing columns: {missing}")
    if frame.duplicated(["season", "team_id"]).any():
        raise ValueError("MOB training frame contains duplicate season/team_id rows")
    result = frame.sort_values(["season", "team_id"], kind="stable").reset_index(drop=True).copy()
    numeric = [
        "target_score",
        "target_matches",
        "current_initial_rating",
        "strong_increment",
        *(variable for variable in config.partition_variables if variable in NUMERIC_PARTITION_VARIABLES),
    ]
    for column in numeric:
        values = pd.to_numeric(result[column], errors="coerce")
        if not np.isfinite(values).all():
            raise ValueError(f"MOB training column {column} must contain finite numbers")
        result[column] = values.astype(float)
    if result["target_matches"].le(0.0).any():
        raise ValueError("MOB target_matches must be positive")
    for variable in config.partition_variables:
        if variable in CATEGORICAL_PARTITION_VARIABLES:
            if result[variable].isna().any() or result[variable].astype(str).str.strip().eq("").any():
                raise ValueError(f"MOB categorical column {variable} cannot be empty")
            result[variable] = result[variable].astype(str)
    return result


def _weighted_calibration(frame: pd.DataFrame, config: MOBConfig) -> tuple[float, float, float, float]:
    ratings = frame["current_initial_rating"].to_numpy(float)
    matches = frame["target_matches"].to_numpy(float)
    weights = matches / (matches + config.target_match_shrinkage)
    mean = float(np.average(ratings, weights=weights))
    variance = float(np.average(np.square(ratings - mean), weights=weights))
    scale = math.sqrt(max(variance, 0.0))
    if scale <= 1e-12:
        raise ValueError("MOB current_initial_rating has zero weighted variance")
    z = (ratings - mean) / scale
    design = np.column_stack([np.ones(len(frame)), z])
    root_w = np.sqrt(weights)[:, None]
    coefficients, *_ = np.linalg.lstsq(design * root_w, frame["target_score"].to_numpy(float) * root_w[:, 0], rcond=None)
    intercept, slope = map(float, coefficients)
    if not math.isfinite(intercept) or not math.isfinite(slope) or slope <= 1e-12:
        raise ValueError("MOB WLS calibration slope must be positive and finite")
    return intercept, slope, mean, scale


def _fit_theta(
    residual: np.ndarray,
    effect: np.ndarray,
    weights: np.ndarray,
    lower: float,
    upper: float,
) -> tuple[float, float, float, float, np.ndarray]:
    denominator = float(np.sum(weights * np.square(effect)))
    if denominator <= 1e-18:
        error = residual.copy()
        rss = float(np.sum(weights * np.square(error)))
        intercept = float(np.average(error, weights=weights))
        return 0.0, float("nan"), intercept, rss, np.zeros_like(error)

    objective = lambda theta: float(np.sum(weights * np.square(residual - theta * effect)))
    result = minimize_scalar(objective, bounds=(lower, upper), method="bounded", options={"xatol": 1e-12})
    if not result.success or not math.isfinite(float(result.x)):
        theta = float(np.clip(np.sum(weights * effect * residual) / denominator, lower, upper))
    else:
        theta = float(np.clip(result.x, lower, upper))
    error = residual - theta * effect
    rss = float(np.sum(weights * np.square(error)))
    degrees = max(len(error) - 1, 1)
    standard_error = math.sqrt(max(rss / degrees, 0.0) / denominator)
    intercept = float(np.average(error, weights=weights))
    scores = weights * effect * error
    scores = scores - float(scores.mean())
    return theta, standard_error, intercept, rss, scores


def _valid_child(mask: np.ndarray, seasons: np.ndarray, nonzero: np.ndarray, config: MOBConfig) -> bool:
    return bool(
        int(mask.sum()) >= config.minimum_leaf_team_seasons
        and len(np.unique(seasons[mask])) >= config.minimum_leaf_seasons
        and int(nonzero[mask].sum()) >= config.minimum_nonzero_increment
    )


def _permuted_scores(
    scores: np.ndarray,
    seasons: np.ndarray,
    permutations: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.empty((permutations, len(scores)), dtype=float)
    for season in sorted(np.unique(seasons)):
        indices = np.flatnonzero(seasons == season)
        random_order = np.argsort(rng.random((permutations, len(indices))), axis=1)
        result[:, indices] = scores[indices][random_order]
    return result


def _candidate_rss(
    left: np.ndarray,
    right: np.ndarray,
    residual: np.ndarray,
    effect: np.ndarray,
    weights: np.ndarray,
    config: MOBConfig,
) -> float:
    return sum(
        _fit_theta(
            residual[indices], effect[indices], weights[indices], config.theta_lower, config.theta_upper
        )[3]
        for indices in (left, right)
    )


def _numeric_instability(
    node_id: str,
    variable: str,
    values: np.ndarray,
    indices: np.ndarray,
    scores: np.ndarray,
    seasons: np.ndarray,
    nonzero: np.ndarray,
    residual: np.ndarray,
    effect: np.ndarray,
    weights: np.ndarray,
    config: MOBConfig,
) -> _SplitCandidate | None:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    valid_positions: list[int] = []
    for position in range(config.minimum_leaf_team_seasons, len(order) - config.minimum_leaf_team_seasons + 1):
        if position >= len(order) or sorted_values[position - 1] >= sorted_values[position]:
            continue
        left_local = order[:position]
        right_local = order[position:]
        if _valid_child(np.isin(np.arange(len(order)), left_local), seasons, nonzero, config) and _valid_child(
            np.isin(np.arange(len(order)), right_local), seasons, nonzero, config
        ):
            valid_positions.append(position)
    if not valid_positions:
        return None

    score_energy = float(np.sum(np.square(scores)))
    if score_energy <= 1e-20:
        return None
    positions = np.asarray(valid_positions, dtype=int)
    fractions = positions / len(order)
    denominators = np.sqrt(score_energy * fractions * (1.0 - fractions))
    ordered_scores = scores[order]
    observed_values = np.abs(np.cumsum(ordered_scores)[positions - 1] / denominators)
    maximum = float(observed_values.max())
    best_positions = positions[np.isclose(observed_values, maximum, rtol=0.0, atol=1e-12)]

    permutation_scores = _permuted_scores(
        scores, seasons, config.permutations, _stable_seed(config.random_seed, node_id, variable)
    )
    permutation_ordered = permutation_scores[:, order]
    permutation_statistics = np.max(
        np.abs(np.cumsum(permutation_ordered, axis=1)[:, positions - 1] / denominators), axis=1
    )
    p_value = float((1 + np.count_nonzero(permutation_statistics >= maximum - 1e-12)) / (config.permutations + 1))

    options = []
    for position in best_positions:
        left_local, right_local = order[:position], order[position:]
        rss = _candidate_rss(left_local, right_local, residual, effect, weights, config)
        threshold = float((sorted_values[position - 1] + sorted_values[position]) / 2.0)
        options.append((rss, threshold, left_local, right_local))
    rss, threshold, left_local, right_local = min(options, key=lambda item: (item[0], item[1]))
    return _SplitCandidate(
        variable,
        "numeric",
        maximum,
        p_value,
        threshold,
        (),
        (),
        indices[left_local],
        indices[right_local],
        rss,
    )


def _category_subsets(categories: tuple[str, ...]) -> Iterable[tuple[str, ...]]:
    from itertools import combinations

    first = categories[0]
    for size in range(1, len(categories)):
        for subset in combinations(categories, size):
            complement = tuple(value for value in categories if value not in subset)
            if first not in subset or not complement:
                continue
            yield subset


def _categorical_instability(
    node_id: str,
    variable: str,
    values: np.ndarray,
    indices: np.ndarray,
    scores: np.ndarray,
    seasons: np.ndarray,
    nonzero: np.ndarray,
    residual: np.ndarray,
    effect: np.ndarray,
    weights: np.ndarray,
    config: MOBConfig,
) -> _SplitCandidate | None:
    categories = tuple(sorted(np.unique(values.astype(str))))
    if len(categories) < 2:
        return None
    masks: list[np.ndarray] = []
    subsets: list[tuple[str, ...]] = []
    for subset in _category_subsets(categories):
        mask = np.isin(values, subset)
        if _valid_child(mask, seasons, nonzero, config) and _valid_child(~mask, seasons, nonzero, config):
            masks.append(mask)
            subsets.append(subset)
    if not masks:
        return None
    matrix = np.column_stack(masks).astype(float)
    score_energy = float(np.sum(np.square(scores)))
    sizes = matrix.sum(axis=0)
    fractions = sizes / len(values)
    denominators = np.sqrt(score_energy * fractions * (1.0 - fractions))
    observed_values = np.abs(scores @ matrix / denominators)
    maximum = float(observed_values.max())
    best = np.flatnonzero(np.isclose(observed_values, maximum, rtol=0.0, atol=1e-12))

    permutation_scores = _permuted_scores(
        scores, seasons, config.permutations, _stable_seed(config.random_seed, node_id, variable)
    )
    permutation_statistics = np.max(np.abs(permutation_scores @ matrix / denominators), axis=1)
    p_value = float((1 + np.count_nonzero(permutation_statistics >= maximum - 1e-12)) / (config.permutations + 1))

    options = []
    for candidate_index in best:
        mask = masks[int(candidate_index)]
        left_local, right_local = np.flatnonzero(mask), np.flatnonzero(~mask)
        rss = _candidate_rss(left_local, right_local, residual, effect, weights, config)
        subset = subsets[int(candidate_index)]
        options.append((rss, subset, left_local, right_local))
    rss, subset, left_local, right_local = min(options, key=lambda item: (item[0], item[1]))
    complement = tuple(value for value in categories if value not in subset)
    return _SplitCandidate(
        variable,
        "categorical",
        maximum,
        p_value,
        None,
        subset,
        complement,
        indices[left_local],
        indices[right_local],
        rss,
    )


def fit_mob_tree(frame: pd.DataFrame, config: MOBConfig | None = None) -> MOBTree:
    config = config or MOBConfig()
    config.validate()
    data = _validate_training_frame(frame, config)
    intercept, slope, rating_mean, rating_scale = _weighted_calibration(data, config)
    ratings = data["current_initial_rating"].to_numpy(float)
    weights = data["target_matches"].to_numpy(float)
    weights = weights / (weights + config.target_match_shrinkage)
    residual = data["target_score"].to_numpy(float) - (
        intercept + slope * ((ratings - rating_mean) / rating_scale)
    )
    effect = slope * data["strong_increment"].to_numpy(float) / rating_scale
    seasons = data["season"].astype(str).to_numpy()
    nonzero = np.abs(data["strong_increment"].to_numpy(float)) > 1e-12
    tests: list[ParameterInstabilityResult] = []
    leaf_count = 1

    def build(indices: np.ndarray, depth: int, node_id: str) -> MOBNode:
        nonlocal leaf_count
        theta, standard_error, diagnostic_intercept, rss, scores = _fit_theta(
            residual[indices], effect[indices], weights[indices], config.theta_lower, config.theta_upper
        )
        node = MOBNode(
            node_id=node_id,
            depth=depth,
            observations=len(indices),
            seasons=tuple(sorted(np.unique(seasons[indices]))),
            theta=theta,
            theta_standard_error=standard_error,
            diagnostic_intercept=diagnostic_intercept,
            weighted_rss=rss,
            nonzero_increment_observations=int(nonzero[indices].sum()),
        )
        if depth >= config.max_depth:
            node.stop_reason = "MAX_DEPTH"
            return node
        if leaf_count >= config.max_leaves:
            node.stop_reason = "MAX_LEAVES"
            return node
        if len(indices) < 2 * config.minimum_leaf_team_seasons:
            node.stop_reason = "INSUFFICIENT_OBSERVATIONS_FOR_SPLIT"
            return node
        if len(np.unique(seasons[indices])) < config.minimum_leaf_seasons:
            node.stop_reason = "INSUFFICIENT_SEASONS"
            return node
        if int(nonzero[indices].sum()) < 2 * config.minimum_nonzero_increment:
            node.stop_reason = "INSUFFICIENT_NONZERO_INCREMENT"
            return node

        split_candidates: dict[str, _SplitCandidate] = {}
        for variable in config.partition_variables:
            local_values = data.loc[indices, variable].to_numpy()
            if variable in NUMERIC_PARTITION_VARIABLES:
                candidate = _numeric_instability(
                    node_id,
                    variable,
                    local_values.astype(float),
                    indices,
                    scores,
                    seasons[indices],
                    nonzero[indices],
                    residual[indices],
                    effect[indices],
                    weights[indices],
                    config,
                )
                variable_type = "numeric"
            else:
                candidate = _categorical_instability(
                    node_id,
                    variable,
                    local_values.astype(str),
                    indices,
                    scores,
                    seasons[indices],
                    nonzero[indices],
                    residual[indices],
                    effect[indices],
                    weights[indices],
                    config,
                )
                variable_type = "categorical"
            if candidate is None:
                tests.append(
                    ParameterInstabilityResult(
                        node_id,
                        variable,
                        variable_type,
                        0.0,
                        1.0,
                        1.0,
                        valid=False,
                        reason="NO_VALID_SPLIT",
                    )
                )
            else:
                split_candidates[variable] = candidate

        if not split_candidates:
            node.stop_reason = "NO_VALID_SPLIT"
            return node
        adjusted = _holm_adjust({name: candidate.raw_p_value for name, candidate in split_candidates.items()})
        for variable, candidate in split_candidates.items():
            tests.append(
                ParameterInstabilityResult(
                    node_id,
                    variable,
                    candidate.variable_type,
                    candidate.statistic,
                    candidate.raw_p_value,
                    adjusted[variable],
                    candidate.threshold,
                    candidate.left_categories,
                    candidate.right_categories,
                    len(candidate.left_indices),
                    len(candidate.right_indices),
                    candidate.weighted_rss_after_split,
                )
            )
        selected = min(
            split_candidates.values(),
            key=lambda candidate: (
                adjusted[candidate.variable],
                -candidate.statistic,
                candidate.weighted_rss_after_split,
                candidate.variable,
            ),
        )
        if adjusted[selected.variable] >= config.significance_level:
            node.stop_reason = "NO_SIGNIFICANT_PARAMETER_INSTABILITY"
            return node

        node.split_variable = selected.variable
        node.split_type = selected.variable_type
        node.split_threshold = selected.threshold
        node.left_categories = selected.left_categories
        node.raw_p_value = selected.raw_p_value
        node.adjusted_p_value = adjusted[selected.variable]
        node.stop_reason = "SPLIT"
        leaf_count += 1
        node.left = build(selected.left_indices, depth + 1, f"{node_id}L")
        node.right = build(selected.right_indices, depth + 1, f"{node_id}R")
        return node

    all_indices = np.arange(len(data), dtype=int)
    root = build(all_indices, 0, "N")
    categories = {
        variable: tuple(sorted(data[variable].astype(str).unique()))
        for variable in config.partition_variables
        if variable in CATEGORICAL_PARTITION_VARIABLES
    }
    return MOBTree(
        config=config,
        root=root,
        calibration_intercept=intercept,
        calibration_slope=slope,
        rating_mean=rating_mean,
        rating_scale=rating_scale,
        training_seasons=tuple(sorted(data["season"].astype(str).unique())),
        training_categories=categories,
        instability_tests=tests,
    )


def predict_multiplier(tree: MOBTree, row: Mapping[str, Any]) -> MOBPrediction:
    return tree.predict(row)


def serialize_tree(tree: MOBTree) -> str:
    return tree.to_json()

