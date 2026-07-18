import pandas as pd
from dataclasses import dataclass, field

from vates._core import TDimVariable
from vates.alm.assets import Asset
from vates.alm.liabs import Liab
from vates.alm.enums import AssetRepBasis


@dataclass(slots=True)
class ALContainer:
    """Asset liability container

    """
    name: str
    _assets: list[Asset] = field(default_factory=list)
    _liabs: list[Liab] = field(default_factory=list)
    _free_estate: float = 0.0

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

    def get_totass_value(self, basis: AssetRepBasis, /, *, include_free_estate: bool = False):
        if basis == AssetRepBasis.MV:
            val = self.totass_mv
        elif basis == AssetRepBasis.FAV:
            val = self.totass_fav
        elif basis == AssetRepBasis.BSV:
            val = self.totass_bsv
        else:
            raise ValueError(f"Invalid '{basis=}'")
        return val + (self._free_estate if include_free_estate else 0)

    @property
    def totass_mv(self) -> float:
        return sum(x.mv for x in self._assets)

    @property
    def totass_fav(self) -> float:
        return sum(x.fav for x in self._assets)

    @property
    def totass_bsv(self) -> float:
        return sum(x.bsv for x in self._assets)

    @property
    def totliab_surr_value(self) -> float:
        return sum(x.surr_val for x in self._liabs)

    @property
    def totliab_math_res(self) -> float:
        return sum(x.math_res for x in self._liabs)

    @property
    def totliab_acct_value(self) -> float:
        return sum(x.acct_value for x in self._liabs)

    @property
    def totliab_asset_share(self) -> float:
        return sum(x.asset_share for x in self._liabs)


class _RateOfReturnIndexer:

    __slots__ = ('_tdv', '_arr_index', '_divby')

    def __init__(self, tdv: TDimVariable, /, arr_index: int | list[int] | None = None, divby: float = 1):
        self._tdv: TDimVariable = tdv
        self._arr_index: int | None = arr_index
        self._divby: float = divby

    def __getitem__(self, t: int | pd.Period | pd.PeriodIndex, /) -> float:
        if isinstance(t, pd.PeriodIndex):
            val = 1
            for tt in t:
                val *= (1 + (self._tdv[tt][self._arr_index] if self._arr_index else self._tdv[tt]) / self._divby)
            return val - 1
        return (self._tdv[t][self._arr_index] if self._arr_index else self._tdv[t]) / self._divby
