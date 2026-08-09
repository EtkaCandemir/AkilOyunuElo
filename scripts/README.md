# Script Guide

Scripts remain in one directory because several backtest runners import shared
helpers from one another. Moving them into subdirectories would change Python
import paths without improving model behavior.

Naming conventions:

- `build_*`: create datasets, audits, or documents.
- `run_*_backtest.py`: historical model evaluation.
- `run_*_pilot.py`: controlled smoke and scenario tests.
- `run_*_replay.py`: chronological season or match replay.
- `verify_*` and `validate_*`: contract and data-quality checks.

The current full evaluation entry point is:

```bash
python3 scripts/run_current_model_evaluation.py
```
