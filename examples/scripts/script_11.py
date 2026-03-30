import sys
import os
import time

# Add 'model' to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))

from em11_stoch_monte_carlo import PortMonteCarlo, PortMonteCarloStoch

def main():
    model_args = {
        "model_name": "stoch_monte_carlo",
        "start_year": 2024,
        "start_month": 12,
        "end_year": 2034,
        "scenario": 'demo',
        "workspace_directory": "examples",
        "input_directories": ["input/11_stoch_monte_carlo"],
        "results_directory": "results/script_11_stoch_monte_carlo",
        "simulations": "1-1000",
        "max_workers": 4,
    }

    model = PortMonteCarloStoch(PortMonteCarlo, **model_args)
    model.run()


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Complete! Time elapsed: {end_time - start_time:.1f} seconds.")