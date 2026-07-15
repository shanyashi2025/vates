"""
Defines the abstract Liab class.
"""
from abc import ABC, abstractmethod
import pandas as pd

from vates._core import ProjModelEngine, add_projection_time_synchronizer
from vates.alm.econs import Currency

@add_projection_time_synchronizer
class Liab(ABC):
    """
    Abstract base class for all liability types.

    Attributes:
        _liab_id (str): Liability identifier.
        _fund_id (str): Fund identifier.
        _currency (Currency): Currency of the liability.
        _entry_date (pd.Period): Entry date of the liability.
        _num_pols (float): Number of policies in force.
        _surr_val (float): Surrender value in force.
        _math_res (float): Mathematical reserve in force.
        _acct_value (float): Account value in force.
        _asset_share (float): Asset share in force.
    """
    time: int           # for type hint only, will be injected by decorator `has_time_synchronizer`
    period: pd.Period   # for type hint only, will be injected by decorator `has_time_synchronizer`

    __slots__ = ('__dict__', '__weakref__', '_time_synchronizer', '_tt_dict', '_liab_id', '_fund_id', '_currency',
                 '_entry_date', '_num_pols', '_surr_val', '_math_res', '_acct_value', '_asset_share', '_cash_flow',
                 '_prem_inc')

    def __init__(
        self,
        *,
        model_engine: ProjModelEngine | None = None,  # will be referenced by decorator `has_time_synchronizer`
        liab_id: str,
        fund_id: str,
        currency: Currency,
        entry_date: pd.Period,
        no_pols_if: float,
        surr_val_if: float,
        math_res_if: float,
        acct_value_if: float,
        asset_share_if: float
    ):
        """
        Initialize a liability object.

        Args:
            model_engine: Model engine object.
            liab_id (str): Liability identifier.
            fund_id (str): Fund identifier.
            currency (Currency): Currency of the liability.
            entry_date (pd.Period): Entry date of the liability.
            no_pols_if (float): Number of policies in force.
            surr_val_if (float): Surrender value in force.
            math_res_if (float): Mathematical reserve in force.
            acct_value_if (float): Account value in force.
            asset_share_if (float): Asset share in force.
        """
        self._liab_id: str = liab_id
        self._fund_id: str = fund_id
        self._currency: Currency = currency
        self._entry_date: pd.Period = entry_date
        self._num_pols: float = no_pols_if
        self._surr_val: float = surr_val_if
        self._math_res: float = math_res_if
        self._acct_value: float = acct_value_if
        self._asset_share: float = asset_share_if
        self._cash_flow: float = 0.0
        self._prem_inc: float = 0.0
        self._tt_dict: dict[str, int] = {"roll_forward": self.time, "update_ad": self.time}

    @property
    def liab_id(self) -> str:
        return self._liab_id

    @property
    def fund_id(self) -> str:
        return self._fund_id

    @property
    def currency(self) -> Currency:
        return self._currency

    @property
    def entry_date(self) -> pd.Period:
        return self._entry_date

    @property
    def last_roll_forward(self) -> int | None:
        """int | None: Last roll forward time index."""
        return self._tt_dict['roll_forward']

    @property
    def last_update_ad(self) -> int | None:
        """int | None: Last update after dealing time index."""
        return self._tt_dict['update_ad']

    @abstractmethod
    def roll_forward(self, *args, **kwargs):
        """
        Abstract method to roll the liability forward in time.
        """
        pass

    @abstractmethod
    def update_ad(self, *args, **kwargs):
        """
        Abstract method to update the liability after dealing.
        """
        pass

    @property
    def cash_flow(self) -> float:
        """float: Cash flow in period"""
        return self._cash_flow

    @property
    def prem_inc(self) -> float:
        """float: Premium income in period"""
        return self._prem_inc

    @property
    def num_pols(self) -> float:
        """float: Number of policies in force."""
        return self._num_pols

    @property
    def surr_val(self) -> float:
        """float: Surrender value in force."""
        return self._surr_val

    @property
    def math_res(self) -> float:
        """float: Mathematical reserve in force."""
        return self._math_res

    @property
    def acct_value(self) -> float:
        """float: Account value."""
        return self._acct_value

    @property
    def asset_share(self) -> float:
        """float: Asset share."""
        return self._asset_share

    @property
    @abstractmethod
    def arr_cash_flow(self):
        """
        Abstract method to get the cash flow for the liability.
        """
        pass

    @property
    @abstractmethod
    def arr_prem_inc(self):
        """
        Abstract method to get the premium income for the liability.
        """
        pass

    @property
    @abstractmethod
    def arr_surr_val(self):
        """
        Abstract property for surrender value. as at a date.
        """
        pass

    @property
    @abstractmethod
    def arr_math_res(self):
        """
        Abstract property for mathematical reserve as at a date.
        """
        pass

    @property
    @abstractmethod
    def arr_acct_value_bd(self):
        """
        Abstract property for account value before dealing as at a date.
        """
        pass

    @property
    @abstractmethod
    def arr_acct_value_ad(self):
        """
        Abstract property for account value after dealing as at a date.
        """
        pass

    @property
    @abstractmethod
    def arr_asset_share_bd(self):
        """
        Abstract property for asset share before dealing as at a date.
        """
        pass

    @property
    @abstractmethod
    def arr_asset_share_ad(self):
        """
        Abstract property for asset share after dealing as at a date.
        """
        pass

    def __str__(self) -> str:
        return f"{type(self).__name__} - '{self._liab_id}'"
