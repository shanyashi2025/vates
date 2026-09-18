import pandas as pd
import uuid
import warnings
import weakref
from abc import ABC, abstractmethod
from typing import Self

from vates._core import ProjModelEngine, add_projection_time_synchronizer
from vates.alm.econs import Currency

@add_projection_time_synchronizer
class Asset(ABC):
    """
    Abstract base class for all financial assets.

    Attributes:
        _model_ref (weakref.ref[ProjModelEngine]): Model engine reference.
        _asset_id (str): Unique identifier for the asset.
        _is_profile (bool): Ture if profile asset, False if existing asset.
        _units (float): Number of assets
        _purchase_date (pd.Period): Purchase date.
        _currency (Currency): Currency of the asset.
        _flex_attr_map (dict[str, str]): Dict of string to named attribute, {"MV": "market_value"}
    """
    time: int           # for type hint only, will be injected by decorator `add_projection_time_synchronizer`
    period: pd.Period   # for type hint only, will be injected by decorator `add_projection_time_synchronizer`

    __slots__ = ('__dict__', '__weakref__', '_model_ref', '_time_synchronizer', '_state', '_asset_id', '_is_profile', '_units',
                 '_purchase_date', '_currency', '_flex_attr_map',)

    def __init__(
        self,
        *,
        model_engine: ProjModelEngine = None,  # will be referenced by decorator `add_projection_time_synchronizer`
        asset_id: str,
        is_profile: bool,
        units: float,
        purchase_date: pd.Period | None,
        currency: Currency | None,
        flex_attr_map: dict[str, str] | None,
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
            flex_attr_map (dict[str, str]): Dict of string to named attribute, {"MV": "market_value"}
        """
        self._model_ref: weakref.ref[ProjModelEngine] = weakref.ref(model_engine) if model_engine is not None else (lambda: None)
        self._asset_id: str = asset_id or str(uuid.uuid4())
        self._is_profile: bool = is_profile
        self._units: float = units
        self._purchase_date: pd.Period = purchase_date
        self._currency: Currency | None = currency
        self._flex_attr_map: dict[str, str] | None = flex_attr_map
        self._state: tuple[str, int] = ("initialized", self.time or 0)

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
    def scale_profile(self, *args, **kwargs) -> Self:
        """Return a new, non-profile asset scaled by `scale`; leave self unchanged."""
        pass

    def _copy_flex_attrs_to(self, clone) -> None:
        if clone is self:
            warnings.warn("Ignore copying attributes to self.")
            return

        for key, val in self.__dict__.items():
            if not isinstance(val, (type(None), bool, str, int, float, tuple, frozenset, bytes)):
                warnings.warn(f"Shallow-copying non-immutable attribute '{key}'; profile and clone will share this object.")
            setattr(clone, key, val)

        #  Shallow copy aliases mutable values by design. The warning acknowledges it but does not prevent it: after clone, profile.__dict__[k] is
        #  clone.__dict__[k] for a list/dict. Since there’s no immutability guarantee, a clone mutating that list would corrupt the reusable profile. Given the small
        #  set of expected metadata, I’d rather copy selectively/deep rather than warn-and-share. Two options: copy.deepcopy(val) for non-immutable values (only if
        #  those are guaranteed copyable), or better, move dynamic metadata into one explicit mapping (_flex_attrs) and deep-copy that. A whitelist based on
        #  _flex_attr_map values + known metadata keys is the most predictable.

    @abstractmethod
    def close_dealing(self, *args, **kwargs):
        """
        Abstract method to update the asset after dealing.
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
        return f"{type(self).__name__} - '{self.asset_id}'"
