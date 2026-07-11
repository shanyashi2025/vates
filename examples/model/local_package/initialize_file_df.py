from typing import Callable
import pandas as pd


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
        if (df is not None) and (not df.index.is_unique):
            raise ValueError(f"Table {name}: duplicate indexes.")
        file_df_dict[key] = df

    return file_df_dict
