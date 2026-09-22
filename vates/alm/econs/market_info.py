import pandas as pd

from vates._core import ProjModelEngine, time_synchronized


@time_synchronized
class MarketInfo:
    """
    Represents the market infomration.
    """
    time: int           # for type hint only, will be injected by decorator `add_projection_time_synchronizer`
    period: pd.Period   # for type hint only, will be injected by decorator `add_projection_time_synchronizer`
    
    __slots__ = ('__dict__', '__weakref__', 'time', 'period', 'info_id', '_data', )

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

    def get(self, key, default=None, /):
        return self._data.get(key, default)

    def __setitem__(self, key, value):
        self._data[key] = value

    def __getitem__(self, item):
        return self._data[item]

    def __str__(self) -> str:
        return f"{type(self).__name__} - '{self.info_id}'"
