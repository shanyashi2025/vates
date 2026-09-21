import pandas as pd
import uuid
import warnings
import weakref
from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import Enum, auto
from typing import Self

from vates._core import ProjModelEngine, add_projection_time_synchronizer
from vates.utils import Lifecycle, maybe_raise_if_ne
from vates.global_conf import CheckLevel
from vates.alm.econs import Currency

_IMMUTABLE = (type(None), bool, int, float, str, bytes, tuple, frozenset)

class AssetPhase(Enum):
    PROFILE = auto()
    ROLLED = auto()
    CLOSED = auto()


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
        _attr_aliases (dict[str, str]): Dict of alias to named attribute, {"MV": "market_value"}
    """
    time: int           # for type hint only, will be injected by decorator `add_projection_time_synchronizer`
    period: pd.Period   # for type hint only, will be injected by decorator `add_projection_time_synchronizer`
    _mutable_attr_check: CheckLevel = CheckLevel.ERROR

    __slots__ = ('__dict__', '__weakref__', '_model_ref', '_time_synchronizer', 'time', 'period', '_lc', '_asset_id', '_is_profile', '_units',
                 '_purchase_date', '_currency', '_attr_aliases',)

    def __init__(
        self,
        *,
        model_engine: ProjModelEngine = None,  # will be referenced by decorator `add_projection_time_synchronizer`
        asset_id: str,
        is_profile: bool,
        units: float,
        purchase_date: pd.Period | None,
        currency: Currency | None,
        attr_aliases: Mapping[str, str] | None,
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
            attr_aliases (Mapping[str, str]): Mapping of alias to named attribute, {"MV": "market_value"}
        """
        self._model_ref: weakref.ref[ProjModelEngine] = weakref.ref(model_engine) if model_engine is not None else (lambda: None)
        self._asset_id: str = asset_id or str(uuid.uuid4())
        self._is_profile: bool = is_profile
        self._units: float = units
        self._purchase_date: pd.Period = purchase_date
        self._currency: Currency | None = currency
        self._attr_aliases: Mapping[str, str] | None = attr_aliases
        self._lc: Lifecycle[AssetPhase] = Lifecycle[AssetPhase]()
        if self._is_profile:
            self._lc.mark(AssetPhase.PROFILE, self.time or 0)
        else:
            self._lc.mark(AssetPhase.ROLLED, self.time or 0)
            self._lc.mark(AssetPhase.CLOSED, self.time or 0)

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

    def _copy_dynamic_attrs_to(self, clone: Self, *, check_level: CheckLevel | None = None) -> None:
        if clone is self:
            warnings.warn("Ignore copying attributes to self.")
            return

        if check_level is None:
            check_level = self._mutable_attr_check

        for key, val in self.__dict__.items():
            if check_level == CheckLevel.BYPASS:
                pass
            elif not isinstance(val, _IMMUTABLE):
                msg = (f"Dynamic attribute '{key}' on profile '{self._asset_id}' is {type(val).__name__}, which is mutable. "
                       f"Profiles must stay immutable; store mutable metadata elsewhere or register it as shared.")
                if check_level == CheckLevel.ERROR:
                    raise TypeError(msg)
                if check_level == CheckLevel.WARN:
                    warnings.warn(msg)
            setattr(clone, key, val)

    @abstractmethod
    def close_dealing(self, *args, **kwargs):
        """
        Abstract method to update the asset after dealing.
        """
        pass

    def require_lifecycle(self, phase: AssetPhase, t: int | None = None):
        t = t if t is not None else self.time
        maybe_raise_if_ne(self._lc[phase], t)

    def __getattr__(self, name):
        try:
            aliases = object.__getattribute__(self, "_attr_aliases")
        except AttributeError:
            raise AttributeError(name) from None

        target = aliases.get(name)

        if target is None or target == name:
            raise AttributeError(name)

        try:
            return object.__getattribute__(self, target)
        except AttributeError:
            raise AttributeError(name) from None

    def __str__(self) -> str:
        return f"{type(self).__name__} - '{self.asset_id}'"
