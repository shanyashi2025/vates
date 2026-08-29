import pandas as pd

from vates._core import ProjModelEngine, add_projection_time_synchronizer, TDimVariable


@add_projection_time_synchronizer
class Currency:
    """
    Represents a currency and its FX rate time series.

    Attributes:
        currency_id (str): Currency identifier.
        tdv_fx_rate (TDimVariable): Current FX rate.
    """
    time: int           # for type hint only, will be injected by decorator `add_projection_time_synchronizer`
    period: pd.Period   # for type hint only, will be injected by decorator `add_projection_time_synchronizer`
    
    __slots__ = ('__dict__', '__weakref__', '_time_synchronizer', '_last_update',
                 'currency_id', 'tdv_fx_rate', )

    def __init__(
        self,
        currency_id: str,
        *,
        model_engine: ProjModelEngine | None = None,
    ) -> None:
        """
        Initialize a Currency object.

        Args:
            currency_id (str): Currency identifier.
        """
        self.currency_id: str = currency_id
        self._last_update: int | None = None

        self.tdv_fx_rate: TDimVariable = TDimVariable("fx_rate", model_engine=model_engine, owner=currency_id, group='currency')

    @property
    def last_update(self) -> int | None:
        """int: Last update time index."""
        return self._last_update

    @property
    def arr_fx_rate(self) -> TDimVariable:
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
        return f"{type(self).__name__} - '{self.currency_id}'"
