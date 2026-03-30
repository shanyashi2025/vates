import sys
import os
import time

# Add 'model' to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))

from em01_asset_proj import AssetModel

def main():
    model_args = {
        "model_name": "asset_proj",
        "start_year": 2024,
        "start_month": 12,
        "end_year": 2026,
        "scenario": "demo",
        "workspace_directory": "examples",
        "input_directories": ["input/01_asset_proj"],
        "results_directory": "results/script_01_asset_proj"
    }

    model = AssetModel(**model_args)
    model.run()


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Complete! Time elapsed: {end_time - start_time:.1f} seconds.")