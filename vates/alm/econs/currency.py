import pandas as pd
import weakref

from ..._core import TDepVariable


class Currency:
    """
    Represents a currency and its FX rate time series.

    Attributes:
        currency_id (str): Currency identifier.
        tdv_fx_rate (TDepVariable): Current FX rate.
    """
    def __init__(self, model, currency_id: str) -> None:
        """
        Initialize a Currency object.

        Args:
            currency_id (str): Currency identifier.
        """
        self._model_ref: weakref.ref = weakref.ref(model)
        self.currency_id: str = currency_id
        self._last_update: int | None = None
        self.tdv_fx_rate: TDepVariable = TDepVariable(model, "fx_rate", currency_id, 'currency')

    @property
    def time(self) -> int | None:
        return self._model_ref().time
    
    @property
    def period(self) -> pd.Period | None:
        return self._model_ref().period

    @property
    def last_update(self) -> int | None:
        """int: Last update time index."""
        return self._last_update

    @property
    def arr_fx_rate(self) -> TDepVariable:
        """Optional[float]: Current FX rate."""
        return self.tdv_fx_rate

    def update(self, fx_rate: float) -> None:
        """
        Update the FX rate for the current time step.

        Args:
            fx_rate (float): New FX rate.
        """
        t = self.time
        self.tdv_fx_rate[t] = fx_rate
        self._last_update = t

    def __str__(self) -> str:
        return self.currency_id
