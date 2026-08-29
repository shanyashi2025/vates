import pandas as pd

from vates._core import ProjModelEngine, add_projection_time_synchronizer


@add_projection_time_synchronizer
class MarketInfo:
    """
    Represents the market infomration.
    """
    time: int           # for type hint only, will be injected by decorator `add_projection_time_synchronizer`
    period: pd.Period   # for type hint only, will be injected by decorator `add_projection_time_synchronizer`
    
    __slots__ = ('__dict__', '__weakref__', '_time_synchronizer', '_last_update', 'info_id', '_data', )

    def __init__(
        self,
        info_id: str = 'untitled',
        *,
        model_engine: ProjModelEngine | None = None,  # will be referenced by decorator `add_projection_time_synchronizer`
    ) -> None:
        """
        Initialize a MarketInfo object.

        Args:
            info_id (str): Market infomration identifier.
        """
        self.info_id: str = info_id
        self._data: dict[str, ...] = {}
        self._last_update: dict[str, int] = {}

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
