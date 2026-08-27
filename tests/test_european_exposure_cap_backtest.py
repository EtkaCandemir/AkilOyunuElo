from __future__ import annotations

from ao_elo.european_prior_recalibration import (
    EuropeanPriorRecalibrationConfig,
    exposure_refinement_grid,
)


def test_focused_exposure_grid_is_ordered_and_contains_baseline() -> None:
    configs = exposure_refinement_grid()
    assert EuropeanPriorRecalibrationConfig() in configs
    assert len({config.key for config in configs}) == len(configs)
    for benchmark in {config.history_benchmark for config in configs}:
        exposures = [
            config.exposure_cap
            for config in configs
            if config.history_benchmark == benchmark
        ]
        assert exposures == sorted(exposures)
        assert exposures[0] == 0.40
        assert exposures[-1] == 0.85
