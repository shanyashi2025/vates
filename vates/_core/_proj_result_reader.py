import pandas as pd
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path


def proj_result(
    *,
    results_directory: str | Path,
    slug: str,
    group: str | None = None,
    owner: str | None = None,
    variable: str | None = None,
    date: str | int | pd.Period | None = None,
) -> pd.DataFrame | float:

    df = ProjResultFileReader.load_proj_result(Path(results_directory) / f"{slug}.proj.csv")

    if all(item is None for item in [group, owner, variable, date]):
        return df

    required = {"group": group, "owner": owner, "variable": variable,}
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise IndexError(f"Missing arguments: {missing}")

    row_label = (group, owner, variable)
    if date is None:
        col_label = "constant"
    elif isinstance(date, pd.Period):
        col_label = str(date.year * 100 + date.month)
    else:
        col_label = str(date)

    try:
        value = df.at[row_label, col_label]
    except KeyError:
        raise LookupError(f"row: '{row_label}' + column: '{col_label}' not found in file '{slug}.proj.csv'.")

    if pd.isna(value) and col_label != "constant":
        value = df.at[row_label, "constant"]  # use 'constant' as fallback

    return value


@dataclass(slots=True, frozen=True)
class CacheEntry:
    df: pd.DataFrame
    mtime: int
    size: int


class ProjResultFileReader:

    _cache: OrderedDict[Path, CacheEntry] = OrderedDict()
    MAX_CACHE_FILES = 32

    @classmethod
    def load_proj_result(cls, file_path: str | Path, /, force_reload: bool = False) -> pd.DataFrame:
        try:
            file_path = Path(file_path)
            stat = file_path.stat()
            mtime, size = stat.st_mtime_ns, stat.st_size
            entry = cls._cache.get(file_path)
            if force_reload or entry is None or entry.mtime != mtime or entry.size != size:
                df = (
                    pd.read_csv(
                        file_path,
                        encoding='utf-8',
                        dtype={"group": str, "owner": str, "variable": str}).
                    set_index(["group", "owner", "variable"]).
                    sort_index().
                    set_flags(allows_duplicate_labels=False)
                )
                cls._cache[file_path] = CacheEntry(df=df, mtime=mtime, size=size)
                cls._clear_cache_if_needed()
            cls._cache.move_to_end(file_path)
            return cls._cache[file_path].df.copy()
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{file_path}' does not exist.")

    @classmethod
    def _clear_cache_if_needed(cls):
        if len(cls._cache) > cls.MAX_CACHE_FILES:
            cls._cache.popitem(last=False)
