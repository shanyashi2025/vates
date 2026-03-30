import sys
import os
import time

# Add 'model' to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))

import em22_efficient_frontier

def main():
    model_args = {
        "workspace_directory": "examples",
        "input_directories": ["input/22_efficient_frontier"],
    }
    em22_efficient_frontier.main(**model_args)


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Complete! Time elapsed: {end_time - start_time:.1f} seconds.")