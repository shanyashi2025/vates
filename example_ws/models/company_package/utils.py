import json
import pandas as pd
import sys
from typing import Callable

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


def load_file_df(read_func: Callable, filename_dict: dict[str, str], read_args: dict[str, dict] | None = None,
                 include: list[str] | None = None, exclude: list[str] | None = None) -> dict[str, pd.DataFrame]:
    if include and exclude:
        raise ValueError(f"Can specify either 'include' or 'exclude', but not both.")

    file_df_dict: dict = {}
    read_args = read_args or {}

    for key, name in filename_dict.items():
        if (include and key not in include) or (exclude and key in exclude):
            continue

        kwargs = read_args.get(key) or {}
        df = read_func(name, **kwargs)
        if df is not None:
            df = df.set_flags(allows_duplicate_labels=False)
        file_df_dict[key] = df

    return file_df_dict
