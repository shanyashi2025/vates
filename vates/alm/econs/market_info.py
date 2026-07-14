import pandas as pd

from vates._core import ProjModelEngine

class MarketInfo:
    """
    Represents the market infomration.
    """
    def __init__(
        self,
        info_id: str = 'untitled',
        *,
        model_engine: ProjModelEngine | None = None,
    ) -> None:
        """
        Initialize a MarketInfo object.

        Args:
            info_id (str): Market infomration identifier.
        """
        self.info_id: str = info_id
        self._data: dict[str, ...] = {}
        self._last_update: dict[str, int] = {}

        if model_engine is not None:
            model_engine.attach_time_observer(self)
            self.time: int = model_engine.time
            self._start_date: pd.Period = model_engine.START_DATE

    def sync_time(self, subject) -> None:
        self.time = subject.time

    @property
    def period(self) -> pd.Period:
        return self._start_date + self.time

    @property
    def last_update(self) -> dict[str, int]:
        """int: Last update time index."""
        return self._last_update

    def get(self, key, default=None, /):
        return self._data.get(key, default)

    def __setitem__(self, key, value):
        self._data[key] = value
        self._last_update[key] = self.time

    def __getitem__(self, item):
        return self._data[item]

    def __str__(self) -> str:
        return f"{type(self).__name__} - '{self.info_id}'"
