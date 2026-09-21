import pandas as pd
import uuid
import weakref
from abc import ABC, abstractmethod
from enum import Enum, auto

from vates._core import ProjModelEngine, add_projection_time_synchronizer
from vates.utils import Lifecycle, maybe_raise_if_ne
from vates.alm.econs import Currency

class LiabPhase(Enum):
    ROLLED = auto()
    CLOSED = auto()


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

    __slots__ = ('__dict__', '__weakref__', '_model_ref', '_time_synchronizer', '_lc', '_liab_id', '_currency',
                 '_entry_date', '_attr_aliases',)

    def __init__(
        self,
        *,
        model_engine: ProjModelEngine | None = None,  # will be referenced by decorator `add_projection_time_synchronizer`
        liab_id: str,
        currency: Currency | None,
        entry_date: pd.Period | None,
        attr_aliases: dict[str, str] | None,
    ):
        """
        Initialize a liability object.

        Args:
            model_engine: Model engine object.
            liab_id (str): Liability identifier.
            currency (Currency): Currency of the liability.
            entry_date (pd.Period): Entry date of the liability.
            attr_aliases (dict[str, str]): Dict of alias to named attribute, {"MATH_RES": "math_reserve"}
        """
        self._model_ref: weakref.ref[ProjModelEngine] = weakref.ref(model_engine) if model_engine is not None else (lambda: None)
        self._liab_id: str = liab_id or str(uuid.uuid4())
        self._currency: Currency = currency
        self._entry_date: pd.Period = entry_date
        self._attr_aliases: dict[str, str] | None = attr_aliases
        self._lc: Lifecycle[LiabPhase] = Lifecycle[LiabPhase](created_phase=LiabPhase.CLOSED, created_at=self.time or 0)

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
    def arr_cash_flow(self):
        """
        Abstract method to get the cash flow for the liability.
        """
        pass

    def require_lifecycle(self, phase: LiabPhase, t: int | None = None):
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
        return f"{type(self).__name__} - '{self._liab_id}'"
