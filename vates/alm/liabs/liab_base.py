"""
Defines the abstract Liab class.
"""
import pandas as pd
import uuid
from abc import ABC, abstractmethod

from vates._core import ProjModelEngine, add_projection_time_synchronizer
from vates.alm.econs import Currency

@add_projection_time_synchronizer
class Liab(ABC):
    """
    Abstract base class for all liability types.

    Attributes:
        _liab_id (str): Liability identifier.
        _currency (Currency): Currency of the liability.
        _entry_date (pd.Period): Entry date of the liability.
    """
    time: int           # for type hint only, will be injected by decorator `add_projection_time_synchronizer`
    period: pd.Period   # for type hint only, will be injected by decorator `add_projection_time_synchronizer`

    __slots__ = ('__dict__', '__weakref__', '_time_synchronizer', '_state', '_liab_id', '_currency',
                 '_entry_date', '_flex_attr_map',)

    def __init__(
        self,
        *,
        model_engine: ProjModelEngine | None = None,  # will be referenced by decorator `add_projection_time_synchronizer`
        liab_id: str,
        currency: Currency | None,
        entry_date: pd.Period | None,
        flex_attr_map: dict[str, str] | None,
    ):
        """
        Initialize a liability object.

        Args:
            model_engine: Model engine object.
            liab_id (str): Liability identifier.
            currency (Currency): Currency of the liability.
            entry_date (pd.Period): Entry date of the liability.
            flex_attr_map (dict[str, str]): Dict of string to named attribute, {"MATH_RES": "math_reserve"}
        """
        self._liab_id: str = liab_id or str(uuid.uuid4())
        self._currency: Currency = currency
        self._entry_date: pd.Period = entry_date
        self._flex_attr_map: dict[str, str] | None = flex_attr_map
        self._state: tuple[str, int] = ("initialized", self.time or 0)

    @property
    def liab_id(self) -> str:
        return self._liab_id

    @property
    def currency(self) -> Currency:
        return self._currency

    @property
    def entry_date(self) -> pd.Period:
        return self._entry_date

    @abstractmethod
    def roll_forward(self, *args, **kwargs):
        """
        Abstract method to roll the liability forward in time.
        """
        pass

    @abstractmethod
    def close_dealing(self, *args, **kwargs):
        """
        Abstract method to update the liability after dealing.
        """
        pass

    @property
    @abstractmethod
    def cash_flow(self) -> float:
        """
        Abstract method to get the cash flow in period.
        """
        pass

    @property
    @abstractmethod
    def prem_inc(self) -> float:
        """
        Abstract method to get the premium income in period.
        """
        pass

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

    def __getattr__(self, item):
        try:
            flex_map = object.__getattribute__(self, "_flex_attr_map")
        except AttributeError:
            raise AttributeError(item) from None

        if flex_map is None or item not in flex_map:
            raise AttributeError(item)

        target = flex_map[item]

        if target == item:
            raise AttributeError(item)

        try:
            return object.__getattribute__(self, target)
        except AttributeError:
            raise AttributeError(item) from None

    def __str__(self) -> str:
        return f"{type(self).__name__} - '{self._liab_id}'"
