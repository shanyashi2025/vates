import pandas as pd
import numpy as np
from typing import overload
from enum import Enum, unique

from vates._core import TDimVariable
from vates.alm.assets import Asset
from vates.alm.liabs import Liab

@unique
class FundSizeType(Enum):
    """Enum for fund size types."""
    FUND = "FUND"
    SURR_VALUE = "SURR_VALUE"
    MATH_RES = "MATH_RES"
    ACCT_VALUE = "ACCT_VALUE"
    ASSET_SHARE = "ASSET_SHARE"
    MAX_AS_MATH = "MAX_AS_MATH"
    MAX_AS_CSV = "MAX_AS_CSV"


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
    def groupby_sum_asset_cash_flow(self, *, groupby: None = None, in_list: None = None,) -> float:
        ...

    @overload
    def groupby_sum_asset_cash_flow(self, *, groupby: str, in_list: str,) -> float:
        ...

    @overload
    def groupby_sum_asset_cash_flow(self, *, groupby: str, in_list: list[str],) -> list[float]:
        ...

    @overload
    def groupby_sum_asset_cash_flow(self, *, groupby: str, in_list: None = None,) -> dict[str, float]:
        ...

    def groupby_sum_asset_cash_flow(
        self,
        *,
        groupby: str | None = None,
        in_list: str | list[str] | None = None,
    ) -> float | list[float] | dict[str, float]:
        """
        path 1:
            (groupby: None, in_list: None): -> float
        path 2:
            (groupby: str, in_list: str): -> float
        path 3:
            (groupby: str, in_list: list[str]): -> list[float]
        path 4:
            (groupby: str, in_list: None): -> dict[str, float]
        """
        if groupby is not None and not isinstance(groupby, str):
            raise TypeError(f"Invalid type of 'groupby': {type(groupby)}, expected 'str'.")
        if in_list is not None and not isinstance(in_list, (str, list)):
            raise TypeError(f"Invalid type of 'in_list': {type(in_list)}, expected 'str' or 'list[str]'.")
        # --- path 1 ---
        if groupby is None and in_list is None:
            return sum(x.cash_flow for x in self._assets)
        # --- path 2 ---
        if groupby is not None and isinstance(in_list, str):
            return sum(x.cash_flow for x in self._assets if getattr(x, groupby) == in_list)
        # --- path 3 ---
        if groupby is not None and isinstance(in_list, list):
            total: list[float] = [0.0] * len(in_list)
            key_to_index: dict[str, int] = {item: index for index, item in enumerate(in_list)}
            if len(total) != len(key_to_index):
                raise ValueError(f"'in_list' constains duplicate values: {in_list}.")
            for asset in self._assets:
                key = getattr(asset, groupby)
                idx = key_to_index[key]
                if key in key_to_index:
                    val = asset.cash_flow
                    total[idx] += val
            return total  # list[float]
        # --- path 4 ---
        if groupby is not None and in_list is None:
            total: dict[str, float] = {}
            for asset in self._assets:
                key = getattr(asset, groupby)
                val = asset.cash_flow
                if key in total:
                    total[key] += val
                else:
                    total[key] = val
            return total  # dict[str, float]

        raise ValueError(f"Calculation not defined: {groupby=}, {in_list=}")

    @overload
    def groupby_sum_asset_report_value(self, *, basis: str, groupby: None = None, in_list: None = None) -> float:
        ...

    @overload
    def groupby_sum_asset_report_value(self, *, basis: str, groupby: str, in_list: str) -> float:
        ...

    @overload
    def groupby_sum_asset_report_value(self, *, basis: str, groupby: str, in_list: list[str]) -> list[float]:
        ...

    @overload
    def groupby_sum_asset_report_value(self, *, basis: str, groupby: str, in_list: None = None) -> dict[str, float]:
        ...

    @overload
    def groupby_sum_asset_report_value(self, *, basis: list[str], groupby: None = None, in_list: None = None) -> np.ndarray:
        ...

    @overload
    def groupby_sum_asset_report_value(self, *, basis: list[str], groupby: str, in_list: str) -> np.ndarray:
        ...

    @overload
    def groupby_sum_asset_report_value(self, *, basis: list[str], groupby: str, in_list: list[str]) -> list[np.ndarray]:
        ...

    @overload
    def groupby_sum_asset_report_value(self, *, basis: list[str], groupby: str, in_list: None = None) -> dict[str, np.ndarray]:
        ...

    def groupby_sum_asset_report_value(
        self,
        *,
        basis: str | list[str],
        groupby: str | None = None ,
        in_list: str | list[str] | None = None,
    ) -> float | np.ndarray | list[float | np.ndarray] | dict[str, float | np.ndarray]:
        """
        path 1:
            (basis: str, groupby: None, in_list: None): -> float
        path 2:
            (basis: str, groupby: str, in_list: str): -> float
        path 3:
            (basis: str, groupby: str, in_list: list[str]): -> list[float]
        path 4:
            (basis: str, groupby: str, in_list: None): -> dict[str, float]
        path 5:
            (basis: list[str], groupby: None, in_list: None): -> np.ndarray
        path 6:
            (basis: list[str], groupby: str, in_list: str): -> np.ndarray
        path 7:
            (basis: list[str], groupby: str, in_list: list[str]): -> list[np.ndarray]
        path 8:
            (basis: list[str], groupby: str, in_list: None): -> dict[str, np.ndarray]
        """
        if not isinstance(basis, (str, list)):
            raise TypeError(f"Invalid type of basis {type(basis)}, expected 'str' or 'list[str]'.")
        if groupby is not None and not isinstance(groupby, str):
            raise TypeError(f"Invalid type of 'groupby': {type(groupby)}, expected 'str'.")
        if in_list is not None and not isinstance(in_list, (str, list)):
            raise TypeError(f"Invalid type of 'in_list': {type(in_list)}, expected 'str' or 'list[str]'.")

        if isinstance(basis, str):
            # --- path 1 ---
            if groupby is None and in_list is None:
                return sum(getattr(x, basis) for x in self._assets)  # float
            # --- path 2 ---
            if groupby is not None and isinstance(in_list, str):
                return sum(getattr(x, basis) for x in self._assets if getattr(x, groupby) == in_list)  # float
            # --- path 3 ---
            if groupby is not None and isinstance(in_list, list):
                total: list[float] = [0.0] * len(in_list)
                key_to_index: dict[str, int] = {item: index for index, item in enumerate(in_list)}
                if len(total) != len(key_to_index):
                    raise ValueError(f"'in_list' constains duplicate values: {in_list}.")
                for asset in self._assets:
                    key = getattr(asset, groupby)
                    if key in key_to_index:
                        val = getattr(asset, basis)
                        total[key_to_index[key]] += val
                return total  # list[float]
            # --- path 4 ---
            if groupby is not None and in_list is None:
                total: dict[str, float] = {}
                for asset in self._assets:
                    key = getattr(asset, groupby)
                    val = getattr(asset, basis)
                    if key in total:
                        total[key] += val
                    else:
                        total[key] = val
                return total  # dict[str, float]
            raise ValueError(f"Calculation not defined: {groupby=}, {in_list=}")
        else: # isinstance(basis, list)
            n_basis = len(basis)
            # --- path 5 ---
            if groupby is None and in_list is None:
                total: list[float] = [0.0] * n_basis
                for asset in self._assets:
                    for i in range(n_basis):
                        val = getattr(asset, basis[i])
                        total[i] += val
                return np.array(total, dtype=float)  # np.ndarray
            # --- path 6 ---
            if groupby is not None and isinstance(in_list, str):
                total: list[float] = [0.0] * n_basis
                for asset in self._assets:
                    key = getattr(asset, groupby)
                    if key == in_list:
                        for i in range(n_basis):
                            val = getattr(asset, basis[i])
                            total[i] += val
                return np.array(total, dtype=float)  # np.ndarray
            # --- path 7 ---
            if groupby is not None and isinstance(in_list, list):
                total: list[list[float]] = [[0.0] * n_basis for _ in in_list]
                key_to_index: dict[str, int] = {item: index for index, item in enumerate(in_list)}
                for asset in self._assets:
                    key = getattr(asset, groupby)
                    idx = key_to_index[key]
                    if key in key_to_index:
                        for i in range(n_basis):
                            val = getattr(asset, basis[i])
                            total[idx][i] += val
                return [np.array(x, dtype=float) for x in total]  # list[np.ndarray]
            # --- path 8 ---
            if groupby is not None and in_list is None:
                total: dict[str, list[float]] = {}
                for asset in self._assets:
                    key = getattr(asset, groupby)
                    if key in total:
                        for i in range(n_basis):
                            val = getattr(asset, basis[i])
                            total[key][i] += val
                    else:
                        total[key] = [getattr(asset, bas) for bas in basis]
                return {key: np.array(val, dtype=float) for key, val in total.items()}  # dict[str, np.ndarray]

            raise ValueError(f"Calculation not defined: {groupby=}, {in_list=}")

    @property
    def totliab_surr_value(self) -> float:
        return sum(x.surr_val_if for x in self._liabs)

    @property
    def totliab_math_res(self) -> float:
        return sum(x.math_res_if for x in self._liabs)

    @property
    def totliab_acct_value(self) -> float:
        return sum(x.acct_value_if for x in self._liabs)

    @property
    def totliab_asset_share(self) -> float:
        return sum(x.asset_share_if_bd for x in self._liabs)

    def get_totliab_attr(self, attr: str, /, *, treat_missing_as_0: bool = False) -> float:
        if treat_missing_as_0:
            return sum(getattr(x, attr, 0.0) for x in self._liabs)
        return sum(getattr(x, attr) for x in self._liabs)

    def get_size(self, *, size_type: FundSizeType, asset_size_basis: str = "MV") -> float:
        """Get the fund size based on the fund size type and basis.

        Args:
            size_type (str): Fund size type (FUND, MATH_RES, ASSET_SHARE, etc.).
            asset_size_basis (str): Asset reporting basis use for rebalance, defaults to "MV"

        Returns:
            float: Computed fund size on the requested basis.

        Raises:
            ValueError: If fund size type is invalid.
        """
        if size_type == FundSizeType.FUND:
            return self.groupby_sum_asset_report_value(basis=asset_size_basis) + self.free_estate
            # # need to include free_estate
        elif size_type == FundSizeType.SURR_VALUE:
            return self.totliab_surr_value
        elif size_type == FundSizeType.MATH_RES:
            return self.totliab_math_res
        elif size_type == FundSizeType.ACCT_VALUE:
            return self.totliab_acct_value
        elif size_type == FundSizeType.ASSET_SHARE:
            return self.totliab_asset_share
        elif size_type == FundSizeType.MAX_AS_MATH:
            return max(self.totliab_asset_share, self.totliab_math_res)
        elif size_type == FundSizeType.MAX_AS_CSV:
            return max(self.totliab_asset_share, self.totliab_surr_value)
        raise ValueError(f"Unknown fund size type: {size_type}.")


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
