import pandas as pd
import uuid
from abc import ABC, abstractmethod

from vates._core import ProjModelEngine, add_projection_time_synchronizer
from vates.alm.econs import Currency

@add_projection_time_synchronizer
class Asset(ABC):
    """
    Abstract base class for all financial assets.

    Attributes:
        _asset_id (str): Unique identifier for the asset.
        _is_profile (bool): Ture if profile asset, False if existing asset.
        _units (float): Number of assets
        _purchase_date (pd.Period): Purchase date.
        _currency (Currency): Currency of the asset.
        _report_basis_to_attr (dict[str, str]): Dict of asset reporting basis to named attribute, {"MV": "market_value"}
    """
    time: int           # for type hint only, will be injected by decorator `add_projection_time_synchronizer`
    period: pd.Period   # for type hint only, will be injected by decorator `add_projection_time_synchronizer`

    __slots__ = ('__dict__', '__weakref__', '_time_synchronizer', '_state', '_asset_id', '_is_profile', '_units',
                 '_purchase_date', '_currency', '_report_basis_to_attr',)

    def __init__(
        self,
        *,
        model_engine: ProjModelEngine = None,  # will be referenced by decorator `add_projection_time_synchronizer`
        asset_id: str,
        is_profile: bool,
        units: float,
        purchase_date: pd.Period | None,
        currency: Currency | None,
        report_basis_to_attr: dict[str, str],
    ):
        """
        Initialize the Asset.

        Args:
            model_engine: Model engine object.
            asset_id (str): Asset identifier.
            is_profile (bool): Ture if profile asset, False if existing asset.
            units (float): Number of assets
            purchase_date (pd.Period): Purchase date. Set to initilization date if input is None.
            currency (Currency): Asset currency.
            report_basis_to_attr (dict[str, str]): Dict of asset reporting basis to named attribute, {"MV": "market_value"}
        """
        self._asset_id: str = asset_id or str(uuid.uuid4())
        self._is_profile: bool = is_profile
        self._units: float = units
        self._purchase_date: pd.Period = purchase_date or (self.period if self._is_profile else None)
        self._currency: Currency | None = currency
        self._report_basis_to_attr: dict[str, str] = report_basis_to_attr | {"MV": "market_value"}  # "MV" is always required
        self._state: tuple[str, int] = ("initialized", self.time or 0)

    @property
    def state(self) -> tuple[str, int]:
        return self._state

    @property
    def asset_id(self) -> str:
        return self._asset_id

    @property
    def is_profile(self) -> bool:
        return self._is_profile

    @property
    def units(self) -> float:
        return self._units

    @property
    def currency(self) -> Currency:
        return self._currency

    @property
    def purchase_date(self) -> pd.Period:
        return self._purchase_date

    @property
    @abstractmethod
    def is_alive(self) -> bool:
        """
        Abstract property for whether asset is alive.

        Returns:
            bool: True if asset is alive, False otherwise.
        """
        pass

    @property
    @abstractmethod
    def market_value(self):
        """
        Abstract property for market value.

        Returns:
            float: Market value (to be implemented by subclasses).
        """
        pass

    def get_report_value(self, basis: str | list[str] | None = None, /) -> float | list[float] | dict[str, float]:
        """
        Get reported value(s).

        Returns:
            float | list[float] | dict[str, float]: The reported value of for a given basis, list of reported values
                corresponding to the given list of bases, or all reported values as a dict.
        """
        if basis is None:
            return {key: getattr(self, val) for key, val in self._report_basis_to_attr.items()}
        elif isinstance(basis, str):
            return getattr(self, self._report_basis_to_attr[basis])
        elif isinstance(basis, list):
            return [getattr(self, self._report_basis_to_attr[x]) for x in basis]
        raise TypeError(f"Invalid type of basis {type(basis)}, expected 'str' or 'list[str]'.")

    @abstractmethod
    def roll_forward(self, *args, **kwargs):
        """
        Abstract method to roll the asset forward in time.
        """
        pass

    @property
    @abstractmethod
    def cash_flow(self) -> float:
        """
        Abstract method to get the cash flow for the asset in period.
        """
        pass

    @property
    @abstractmethod
    def arr_cash_flow(self):
        """
        Abstract method to get the cash flow for the asset.
        """
        pass

    @abstractmethod
    def buy_propn(self, *args, **kwargs):
        """
        Abstract method to buy a proportion of the asset.
        """
        pass

    @abstractmethod
    def sell_propn(self, *args, **kwargs):
        """
        Abstract method to sell a proportion of the asset.
        """
        pass

    @abstractmethod
    def buy_profile_scale(self, *args, **kwargs):
        """
        Abstract method to scale the asset profile by a factor.
        """
        pass

    @abstractmethod
    def close_dealing(self, *args, **kwargs):
        """
        Abstract method to update the asset after dealing.
        """
        pass

    def __str__(self) -> str:
        return f"{type(self).__name__} - '{self.asset_id}'"
