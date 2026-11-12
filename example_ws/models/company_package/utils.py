import json
import pandas as pd
import sys
from typing import Callable, Literal

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


def parse_int_list_from_str(str_in: str, /, *, separator: str = ',', joiner: str = '-',
                            sort_list: Literal[None, 'ascending', 'asc', 'descending', 'desc'] = None,
                            on_duplicate: Literal['keep', 'remove', 'error'] = 'error') -> list[int]:
    """Parse string to a list of non-negative integers"""
    if not isinstance(str_in, str):
        raise TypeError(f"{str_in}: type {type(str_in)} is not allowed, expected 'str'.")

    import re
    int_lst = []
    parts = str_in.replace(' ', '').split(separator)

    for part in parts:
        if not part:
            pass
        elif re.fullmatch(rf'\d+\s*{joiner}\s*\d+', part):  # range like "1-5"
            s1, s2 = re.split(rf'\s*{joiner}\s*', part)
            n1, n2 = int(s1), int(s2)
            n1, n2 = min(n1, n2), max(n1, n2)
            int_lst.extend(range(n1, n2 + 1))
        elif re.fullmatch(r'\d+', part):  # single integer
            n = int(part)
            int_lst.append(n)
        else:
            raise ValueError(f"{part} is not allowed, expected non-negative integer, separated by '{separator}' "
                             f"and/or a range joined by '{joiner}'.")

    if sort_list is None:
        pass
    elif sort_list.lower() in ('ascending', 'asc'):
        int_lst.sort()
    elif sort_list.lower() in ('descending', 'desc'):
        int_lst.sort(reverse=True)

    int_set = set(int_lst)
    if len(int_lst) != len(int_set):
        on_duplicate = on_duplicate.lower()
        if on_duplicate == 'keep':
            pass
        elif on_duplicate == 'remove':
            int_lst = list(int_set)
        elif on_duplicate == 'error':
            dup_lst = [x for x in int_set if int_lst.count(x) > 1]
            raise ValueError(f"Duplicate entries in '{str_in}': e.g. {dup_lst[:min(5, len(dup_lst))]}. "
                             f"Revise the input, or specify duplicate_handler='keep' or 'remove'.")

    return int_lst
