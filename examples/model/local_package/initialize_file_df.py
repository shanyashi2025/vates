from typing import Callable
import pandas as pd


def load_file_df(read_func: Callable, filename_dict: dict[str, str], read_config: dict[str, dict] | None = None
                 ) -> dict[str, pd.DataFrame]:
    file_df_dict: dict = {}
    if read_config is None:
        read_config = {}

    for key, name in filename_dict.items():
        kwargs = read_config.get(key, None)
        if kwargs is None:
            file_df_dict[key] = read_func(name)
        else:
            file_df_dict[key] = read_func(name, **kwargs)

        if (file_df_dict[key] is not None) and (not file_df_dict[key].index.is_unique):
            raise ValueError(f"Table {name}: duplicate indexes.")

    return file_df_dict
