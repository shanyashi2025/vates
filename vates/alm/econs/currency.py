import pandas as pd

from vates._core import ProjModelEngine, TDepVariable


class Currency:
    """
    Represents a currency and its FX rate time series.

    Attributes:
        currency_id (str): Currency identifier.
        tdv_fx_rate (TDepVariable): Current FX rate.
    """
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

        if model_engine is not None:
            model_engine.attach_time_observer(self)
            self.time: int = model_engine.time
            self._start_date: pd.Period = model_engine.START_DATE

        self.tdv_fx_rate: TDepVariable = TDepVariable("fx_rate", model_engine=model_engine, owner=currency_id, group='currency')

    def sync_time(self, subject) -> None:
        self.time = subject.time

    @property
    def period(self) -> pd.Period:
        return self._start_date + self.time

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
        return f"{type(self).__name__} - '{self.currency_id}'"
