import sys
import os
import time

# Add 'model' to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))

from em02_fund_proj import FundModel

def main():
    model_args = {
        "name": "fund_proj",
        "start_year": 2024,
        "start_month": 12,
        "end_year": 2026,
        "scenario": "demo",
        "workspace_directory": "examples",
        "input_directories": ["input/02_fund_proj"],
        "results_directory": "results/script_02_fund_proj"
    }

    model = FundModel(**model_args)
    model.run()


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Complete! Time elapsed: {end_time - start_time:.1f} seconds.")