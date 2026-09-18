import pandas as pd
import numpy as np
from typing import overload, Callable

from vates._core import TDimVariable
from vates.alm.assets import Asset
from vates.alm.liabs import Liab


class AssetLiabConnector:
    """Asset liability connector
    """
    __slots__ = ('_assets', '_liabs', '_free_estate')

    def __init__(self, assets: list[Asset] | None = None, liabs: list[Liab] | None = None, free_estate: float = 0.0):
        self._assets: list[Asset] = assets or []
        self._liabs: list[Liab] = liabs or []
        self._free_estate: float = free_estate

    @property
    def assets(self) -> list[Asset]:
        return self._assets

    @property
    def liabs(self) -> list[Liab]:
        return self._liabs

    @property
    def free_estate(self) -> float:
        return self._free_estate

    def accumulate_free_estate(self, amount: float) -> None:
        self._free_estate += amount

    def dispose_free_estate(self) -> float:
        amount = self._free_estate
        self._free_estate = 0.0
        return amount

    @overload
    def sum_asset(self, key: str, /, *, treat_missing_as_0: bool = False) -> float:
        ...

    @overload
    def sum_asset(self, name: list[str], /, *, treat_missing_as_0: bool = False) -> np.ndarray:
        ...

    def sum_asset(self, name: str | list[str], /, *, treat_missing_as_0: bool = False) -> float | np.ndarray:
        return self._sum(self._assets, name=name, treat_missing_as_0=treat_missing_as_0)

    @overload
    def sum_liab(self, name: str, /, *, treat_missing_as_0: bool = False) -> float:
        ...

    @overload
    def sum_liab(self, name: list[str], /, *, treat_missing_as_0: bool = False) -> np.ndarray:
        ...

    def sum_liab(self, name: str | list[str], /, *, treat_missing_as_0: bool = False) -> float | np.ndarray:
        return self._sum(self._liabs, name=name, treat_missing_as_0=treat_missing_as_0)

    @classmethod
    def _sum(cls, obj_list: list, /, *, name: str | list[str], treat_missing_as_0: bool = False) -> float | np.ndarray:
        default = 0.0 if treat_missing_as_0 else None
        if isinstance(name, str):
            return sum(getattr(obj, name, default) for obj in obj_list)
        elif isinstance(name, list):
            n_name: int = len(name)
            total: list[float] = [0.0] * n_name
            for i in range(n_name):
                total[i] = sum(getattr(obj, name[i], default) for obj in obj_list)
            return np.array(total, dtype=float)
        else:
            raise TypeError(f"Invalid {type(name)=}, expected 'str' or 'list[str]'.")

    @overload
    def groupby_sum_asset(self, name: str, /, *, groupby: str, in_list: str, treat_missing_as_0: bool = False) -> float:
        ...

    @overload
    def groupby_sum_asset(self, name: str, /, *, groupby: str, in_list: list[str], treat_missing_as_0: bool = False
                          ) -> list[float]:
        ...

    @overload
    def groupby_sum_asset(self, name: str, /, *, groupby: str, in_list: None = None, treat_missing_as_0: bool = False
                          ) -> dict[str, float]:
        ...

    @overload
    def groupby_sum_asset(self, name: list[str], /, *, groupby: str, in_list: str, treat_missing_as_0: bool = False
                          ) -> np.ndarray:
        ...

    @overload
    def groupby_sum_asset(self, name: list[str], /, *, groupby: str, in_list: list[str], treat_missing_as_0: bool = False
                          ) -> list[np.ndarray]:
        ...

    @overload
    def groupby_sum_asset(self, name: list[str], /, *, groupby: str, in_list: None = None, treat_missing_as_0: bool = False
                          ) -> dict[str, np.ndarray]:
        ...

    def groupby_sum_asset(
        self,
        name: str | list[str],
        /,
        *,
        groupby: str,
        in_list: str | list[str] | None = None,
        treat_missing_as_0: bool = False,
    ) -> float | np.ndarray | list[float | np.ndarray] | dict[str, float | np.ndarray]:
        return self._groupby_sum(self._assets, name=name, groupby=groupby, in_list=in_list,
                                 treat_missing_as_0=treat_missing_as_0)

    @overload
    def groupby_sum_liab(self, name: str, /, *, groupby: str, in_list: str, treat_missing_as_0: bool = False) -> float:
        ...

    @overload
    def groupby_sum_liab(self, name: str, /, *, groupby: str, in_list: list[str], treat_missing_as_0: bool = False
                         ) -> list[float]:
        ...

    @overload
    def groupby_sum_liab(self, name: str, /, *, groupby: str, in_list: None = None, treat_missing_as_0: bool = False
                         ) -> dict[str, float]:
        ...

    @overload
    def groupby_sum_liab(self, name: list[str], /, *, groupby: str, in_list: str, treat_missing_as_0: bool = False
                         ) -> np.ndarray:
        ...

    @overload
    def groupby_sum_liab(self, name: list[str], /, *, groupby: str, in_list: list[str], treat_missing_as_0: bool = False
                         ) -> list[np.ndarray]:
        ...

    @overload
    def groupby_sum_liab(self, name: list[str], /, *, groupby: str, in_list: None = None, treat_missing_as_0: bool = False
                         ) -> dict[str, np.ndarray]:
        ...

    def groupby_sum_liab(
        self,
        name: str | list[str],
        /,
        *,
        groupby: str,
        in_list: str | list[str] | None = None,
        treat_missing_as_0: bool = False,
    ) -> float | np.ndarray | list[float | np.ndarray] | dict[str, float | np.ndarray]:
        return self._groupby_sum(self._liabs, name=name, groupby=groupby, in_list=in_list,
                                 treat_missing_as_0=treat_missing_as_0)

    @classmethod
    def _groupby_sum(
            cls,
            obj_list: list,
            /,
            *,
            name: str | list[str],
            groupby: str,
            in_list: str | list[str] | None = None,
            treat_missing_as_0: bool = False,
    ) -> float | np.ndarray | list[float | np.ndarray] | dict[str, float | np.ndarray]:
        """
        path 1:
            (name: str, groupby: str, in_list: str): -> float
        path 2:
            (name: str, groupby: str, in_list: list[str]): -> list[float]
        path 3:
            (name: str, groupby: str, in_list: None): -> dict[str, float]
        path 4:
            (name: list[str], groupby: str, in_list: str): -> np.ndarray
        path 5:
            (name: list[str], groupby: str, in_list: list[str]): -> list[np.ndarray]
        path 6:
            (name: list[str], groupby: str, in_list: None): -> dict[str, np.ndarray]
        """
        if not isinstance(groupby, str):
            raise TypeError(f"Invalid type of 'groupby': {type(groupby)}, expected 'str'.")
        if not isinstance(name, (str, list)):
            raise TypeError(f"Invalid type of name {type(name)}, expected 'str' or 'list[str]'.")
        if in_list is not None and not isinstance(in_list, (str, list)):
            raise TypeError(f"Invalid type of 'in_list': {type(in_list)}, expected 'str' or 'list[str]'.")
        default = 0.0 if treat_missing_as_0 else None

        if isinstance(name, str):
            # --- path 1 ---
            if isinstance(in_list, str):
                return sum(getattr(obj, name, default) for obj in obj_list if getattr(obj, groupby) == in_list)  # float
            # --- path 2 ---
            if isinstance(in_list, list):
                total: list[float] = [0.0] * len(in_list)
                key_to_index: dict[str, int] = {item: index for index, item in enumerate(in_list)}
                if len(total) != len(key_to_index):
                    raise ValueError(f"'in_list' constains duplicate values: {in_list}.")
                for obj in obj_list:
                    key = getattr(obj, groupby)
                    if key in key_to_index:
                        val = getattr(obj, name, default)
                        total[key_to_index[key]] += val
                return total  # list[float]
            # --- path 3 ---
            if in_list is None:
                total: dict[str, float] = {}
                for obj in obj_list:
                    key = getattr(obj, groupby)
                    val = getattr(obj, name, default)
                    if key in total:
                        total[key] += val
                    else:
                        total[key] = val
                return total  # dict[str, float]
            raise ValueError(f"Calculation not defined: {groupby=}, {in_list=}")
        else:  # isinstance(basis, list)
            n_name = len(name)
            # --- path 4 ---
            if isinstance(in_list, str):
                total: list[float] = [0.0] * n_name
                for obj in obj_list:
                    key = getattr(obj, groupby)
                    if key == in_list:
                        for i in range(n_name):
                            val = getattr(obj, name[i], default)
                            total[i] += val
                return np.array(total, dtype=float)  # np.ndarray
            # --- path 5 ---
            if isinstance(in_list, list):
                total: list[list[float]] = [[0.0] * n_name for _ in in_list]
                key_to_index: dict[str, int] = {item: index for index, item in enumerate(in_list)}
                for obj in obj_list:
                    key = getattr(obj, groupby)
                    idx = key_to_index[key]
                    if key in key_to_index:
                        for i in range(n_name):
                            val = getattr(obj, name[i], default)
                            total[idx][i] += val
                return [np.array(x, dtype=float) for x in total]  # list[np.ndarray]
            # --- path 6 ---
            if in_list is None:
                total: dict[str, list[float]] = {}
                for obj in obj_list:
                    key = getattr(obj, groupby)
                    if key in total:
                        for i in range(n_name):
                            val = getattr(obj, name[i], default)
                            total[key][i] += val
                    else:
                        total[key] = [getattr(obj, bas) for bas in name]
                return {key: np.array(val, dtype=float) for key, val in total.items()}  # dict[str, np.ndarray]

            raise ValueError(f"Calculation not defined: {groupby=}, {in_list=}")

    def get_size(self, *, func: Callable | None = None, key: str | tuple[str, ...] | dict[str, str],
                 treat_missing_as_0: bool = False) -> float:
        """
        path 1:
            (func: None, key: str)
        path 2:
            (func: Callable, key: str)
        path 3:
            (func: Callable, key: tuple[str])
        path 4:
            (func: Callable, key: dict[str, str])
        """
        if func is None:
            return self._get_size_measure(key, treat_missing_as_0)
        if isinstance(key, str):
            return func(self._get_size_measure(key, treat_missing_as_0))
        elif isinstance(key, tuple):
            args = tuple([self._get_size_measure(x, treat_missing_as_0) for x in key])
            return func(*args)
        elif isinstance(key, dict):
            kwargs = {k: self._get_size_measure(v, treat_missing_as_0) for k, v in key.items()}
            return func(**kwargs)
        raise TypeError(f"Invalid {type(key)=}, expected 'str', 'tuple', 'dict'.")

    def _get_size_measure(self, key: str, treat_missing_as_0: bool = False) -> float:
        if not isinstance(key, str):
            raise TypeError(f"Invalid {type(key)=}, expected 'str'.")

        if "." not in key:
            return getattr(self, key, 0.0 if treat_missing_as_0 else None)

        owner, name, *_ = key.split(".")
        if owner == "asset":
            return self._sum(self._assets, name=name, treat_missing_as_0=treat_missing_as_0)
        elif owner in ("liab", "liability"):
            return self._sum(self._liabs, name=name, treat_missing_as_0=treat_missing_as_0)
        else:
            raise ValueError(f"Invalid '{key}', expected 'asset.foo' or 'liab.foo'.")


class _RateOfReturnIndexer:

    __slots__ = ('_tdv', '_divby')

    def __init__(self, tdv: TDimVariable, /, divby: float = 1):
        self._tdv: TDimVariable = tdv
        self._divby: float = divby

    def __getitem__(self, keys) -> float:
        if isinstance(keys, tuple):
            t, *dims = keys
        else:
            t, dims = keys, None

        if self._tdv.ndim != len(dims):
            raise ValueError(f"'{self._tdv.name}' ndim = {self._tdv.ndim}, but {dims} is provided.")

        dim_index = tuple([self._tdv.dims[i].index(dim_name) for i, dim_name in enumerate(dims)]) if dims else None
        if isinstance(t, pd.PeriodIndex):
            val = 1
            for tt in t:
                val *= (1 + self._ror(tt, dim_index))
            return val - 1
        return self._ror(t, dim_index)

    def _ror(self, t: int | pd.Period, dim_index: tuple[int] | None) -> float:
        return (self._tdv[t][dim_index] if dim_index else self._tdv[t]) / self._divby
