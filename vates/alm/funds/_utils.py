import pandas as pd

from vates._core import TDimVariable


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