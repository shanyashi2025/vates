import pandas as pd

from vates._core import ProjModelEngine

class MarketInfo:
    """
    Represents the market infomration.
    """
    def __init__(self, model_engine: ProjModelEngine, info_id: str) -> None:
        """
        Initialize a MarketInfo object.

        Args:
            info_id (str): Market infomration identifier.
        """
        model_engine.attach_time_observer(self)
        self.time: int = model_engine.time
        self.period: pd.Period = model_engine.period

        self.info_id: str = info_id
        self._last_update: int | None = None
        self._data: dict[str, ...] = {}

    def sync_time(self, subject: ProjModelEngine) -> None:
        self.time = subject.time
        self.period = subject.period

    @property
    def last_update(self) -> int | None:
        """int: Last update time index."""
        return self._last_update

    @property
    def data(self) -> dict | None:
        """float: Market information data."""
        return self._data

    def update(self, data: dict[str, ...]) -> None:
        """
        Update the market information data for the current time step.

        Args:
            data (dict[str, ...]): Dictionary of market information data.
        """
        self._data = data
        self._last_update = self.time

    def __str__(self) -> str:
        return self.info_id
