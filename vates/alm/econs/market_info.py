import pandas as pd
import weakref

class MarketInfo:
    """
    Represents the market infomration.
    """
    def __init__(self, model, info_id: str) -> None:
        """
        Initialize a MarketInfo object.

        Args:
            info_id (str): Market infomration identifier.
        """
        self._model_ref: weakref.ref = weakref.ref(model)
        self.info_id: str = info_id
        self._last_update: int | None = None
        self._data: dict[str, ...] = {}

    @property
    def model_proxy(self):
        return self._model_ref()

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
