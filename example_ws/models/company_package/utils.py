import json
import sys

def run_with_json_config(func, config_path: str | None = None):
    """
    used for command `python {script.py} {config.json}`, JSON config file will be parsed by `sys.argv[1]`
    """
    if config_path is None:
        if len(sys.argv) < 2:
            raise SystemExit("Usage: python script.py config.json")
        config_path = sys.argv[1]

    with open(config_path, "r", encoding="utf-8") as file:
        kwargs = json.load(file)

    return func(**kwargs)
