import sys
import os
import time

# Add 'model' to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))

from em12_stoch_ec_mvl import ECMVL
from vates import StochExecutor

def main():
    model_args = {
        "model_name": "stoch_ec_mvl",
        "start_year": 2024,
        "start_month": 12,
        "end_year": 2124,
        "scenario": 'demo',
        "workspace_directory": "examples",
        "input_directories": ["input/12_stoch_ec_mvl"],
        "results_directory": "results/script_12_stoch_ec_mvl",
        "simulations": "1-9",
        # "simulation": 1,
        "max_workers": 3,
    }

    model = StochExecutor(ECMVL, **model_args)
    # model = ECMVL(**model_args)
    model.run()


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Complete! Time elapsed: {end_time - start_time:.1f} seconds.")