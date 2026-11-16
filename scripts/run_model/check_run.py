"""run.py on a small slice, to check that a model runs end to end.

    RUN_PY=scripts/run_model/check_run.py sbatch scripts/run_model/<model>.sh mmmlu,polymath

Each task is cut to its first CHECK_SUBSETS subsets (default 2) and CHECK_LIMIT
examples per subset (default 8), and results go to results_analysis/_check/ so
that real results are not overwritten. Everything else is run.py unchanged.
"""
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import run  # noqa: E402

_load_config = run.load_config


def load_small_config(path):
    config = _load_config(path)
    for task in config.get('tasks', {}).values():
        if isinstance(task.get('subsets'), list):
            task['subsets'] = task['subsets'][:int(os.getenv('CHECK_SUBSETS', 2))]
        task.setdefault('defaults', {})['limit_per_subset'] = int(os.getenv('CHECK_LIMIT', 8))
    return config


run.load_config = load_small_config
run.get_output_dir = lambda: 'results_analysis/_check'
asyncio.run(run.main())
