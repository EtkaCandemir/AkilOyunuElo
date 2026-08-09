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

from ao_elo.config import AOEuropeanEloConfig  # noqa: E402
from ao_elo.evaluation import schedule_adjusted_team_performance  # noqa: E402
from ao_elo.progression_probability import (  # noqa: E402
    ProgressionProbabilityConfig,
)
from ao_elo.robustness import (  # noqa: E402
    baseline_competition_k,
    baseline_goal_margin,
)
from scripts.run_dynamic_core_calibration import expanding_folds  # noqa: E402
from scripts.run_final_robustness import (  # noqa: E402
    ControlledGoalConfig,
    read_final_production_contract,
    run_reserve_calibration,
    validate_fold_inputs,
    write_layer_outputs,
)
from scripts.run_v2_achievement_reserve_calibration import (  # noqa: E402
    load_reserve_data,
)
from scripts.run_v2_evaluation_upgrade import read_events  # noqa: E402


STATIC_DATA_ROOT = ROOT / "data" / "backtest_stage_b_2018_2026"
EVENTS_PATH = (
    ROOT / "data" / "external_elo_benchmark_2018_2026"
    / "exact_date_events.csv"
)
DYNAMIC_ROOT = ROOT / "output" / "v2_dynamic_calibration_2018_2026"
EVALUATION_ROOT = ROOT / "output" / "v2_evaluation_upgrade_2018_2026"
PRODUCTION_CONTRACT = ROOT / "contracts" / "ao_european_elo_v2_production.json"
PROBABILITY_MANIFEST = (
    ROOT / "output" / "progression_probability_2018_2026"
    / "selected_progression_probability.json"
)
OUTPUT_ROOT = (
    ROOT / "output"
    / "progression_reserve_final_backtest_2018_2026"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Re-test Achievement Reserve on carry=0 standard 1X2, controlled "
            "goal difference, and calibrated progression probability"
        )
    )
    parser.add_argument("--static-data-root", type=Path, default=STATIC_DATA_ROOT)
    parser.add_argument("--events", type=Path, default=EVENTS_PATH)
    parser.add_argument("--dynamic-root", type=Path, default=DYNAMIC_ROOT)
    parser.add_argument("--evaluation-root", type=Path, default=EVALUATION_ROOT)
    parser.add_argument(
        "--production-contract",
        type=Path,
        default=PRODUCTION_CONTRACT,
    )
    parser.add_argument(
        "--probability-manifest",
        type=Path,
        default=PROBABILITY_MANIFEST,
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples must be at least 100")

    dynamic_root = args.dynamic_root.resolve()
    evaluation_root = args.evaluation_root.resolve()
    dynamic_manifest = json.loads(
        (dynamic_root / "selected_dynamic_model.json").read_text(
            encoding="utf-8"
        )
    )
    static_config = AOEuropeanEloConfig(**dynamic_manifest["static_config"])
    static_config.validate()
    production = read_final_production_contract(
        evaluation_root / "selected_production_model.json"
    )
    controlled_contract = json.loads(
        args.production_contract.resolve().read_text(encoding="utf-8")
    )
    goal = controlled_contract["goal_margin"]
    if not bool(goal["active"]):
        raise ValueError("Controlled goal-difference production layer is inactive")
    controlled_goal = ControlledGoalConfig(
        float(goal["alpha"]),
        float(goal["tau"]),
        int(goal["goal_difference_cap"]),
    )
    controlled_goal.validate()
    probability_manifest = json.loads(
        args.probability_manifest.resolve().read_text(encoding="utf-8")
    )
    if not bool(probability_manifest["reserve_retest_authorized"]):
        raise ValueError("Progression probability does not authorize reserve retest")
    advance_configs = {
        str(season): config_from_payload(payload)
        for season, payload in probability_manifest[
            "configs_by_test_season"
        ].items()
    }
    full_advance = config_from_payload(
        probability_manifest["full_data_candidate"]
    )

    events = read_events(args.events.resolve())
    datasets, _ = load_reserve_data(
        args.static_data_root.resolve(),
        args.events.resolve(),
        static_config,
    )
    seasons = tuple(data.season for data in datasets)
    folds = expanding_folds(seasons)
    core_selections = pd.read_csv(
        dynamic_root / "core_fold_selections.csv"
    )
    draw_selections = pd.read_csv(
        evaluation_root / "draw_production_fold_selections.csv"
    )
    validate_fold_inputs(core_selections, draw_selections, folds)
    target = schedule_adjusted_team_performance(events)

    print(
        "Final reserve backtest: calibrated P_advance + controlled GD",
        flush=True,
    )
    result = run_reserve_calibration(
        datasets,
        events,
        target,
        folds,
        core_selections,
        draw_selections,
        production,
        baseline_goal_margin(),
        baseline_competition_k(),
        bootstrap_samples=args.bootstrap_samples,
        advance_configs_by_test_season=advance_configs,
        full_advance_config=full_advance,
        controlled_goal_config=controlled_goal,
    )
    output_root = args.output_root.resolve()
    write_layer_outputs(output_root, result)
    (output_root / "backtest_report.md").write_text(
        build_report(result, controlled_goal, full_advance),
        encoding="utf-8",
    )
    print(f"Decision: {result['decision']}")
    print(f"Full candidate: {result['full_candidate']}")
    print(f"Report: {output_root / 'backtest_report.md'}")


def config_from_payload(
    payload: dict[str, object],
) -> ProgressionProbabilityConfig:
    config = ProgressionProbabilityConfig(
        float(payload["logit_slope"]),
        float(payload["single_home_bias"]),
        float(payload["two_leg_first_home_bias"]),
    )
    config.validate()
    return config


def build_report(
    result: dict[str, object],
    controlled_goal: ControlledGoalConfig,
    full_advance: ProgressionProbabilityConfig,
) -> str:
    summary = result["summary"]
    uncertainty = result["uncertainty"]
    selections = result["selections"]
    assert isinstance(summary, pd.DataFrame)
    assert isinstance(uncertainty, pd.DataFrame)
    assert isinstance(selections, pd.DataFrame)
    return f"""# AO Progression Reserve Final Backtest

## Decision

**{result["decision"]}**

This is the incremental Achievement Reserve test on the current comparator:

- carry = 0
- standard multiclass 1X2
- fixed K
- controlled goal difference alpha={controlled_goal.alpha},
  tau={controlled_goal.tau}, cap={controlled_goal.goal_cap}
- calibrated single/two-leg P_advance
- competition K disabled

Full-data progression diagnostic config:
`{full_advance.key}`

Full reserve candidate:
`{result["full_candidate"]}`

Active reserve candidate:
`{result["active_candidate"]}`

## Fold Selections

```csv
{selections.to_csv(index=False).strip()}
```

## Competition Summary

```csv
{summary.to_csv(index=False).strip()}
```

## Dependency Uncertainty

```csv
{uncertainty.to_csv(index=False).strip()}
```

## Guardrails

```json
{json.dumps(result["guardrails"], indent=2)}
```
"""


if __name__ == "__main__":
    main()
