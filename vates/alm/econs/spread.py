import numpy as np
import numpy.typing as npt
import pandas as pd

from vates import TDimVariable
from vates._core import ProjModelEngine, time_synchronized, make_proj_variable
from vates.utils import maybe_raise_if_ne


@time_synchronized("model_engine.time_synchronizer")
class Spread:
    """
    Represents spread information for a financial instrument.

    Attributes:
        spread_id (str): spread identifier.
        _additive (float | npt.NDArray[np.float64]): Additive spread.
        _proportional (float | npt.NDArray[np.float64]): Proportional spread (multiplied by risk-free rates).
    """
    time: int           # for type hint only, will be injected by decorator `time_synchronized`
    period: pd.Period   # for type hint only, will be injected by decorator `time_synchronized`
    _additive: float | npt.NDArray[np.float64]
    _proportional: float | npt.NDArray[np.float64]

    __slots__ = ('__dict__', '__weakref__', 'time', 'period', '_last_update', 'spread_id',
                 '_additive', '_proportional', 'tdv_add', 'tdv_propn',)

    def __init__(
        self,
        spread_id: str,
        *,
        model_engine: ProjModelEngine | None = None,
        tdv_additive_term_dim: list[int] | None = None,
        tdv_proportional_term_dim: list[int] | None = None,
    ) -> None:
        """
        Initialize a Spread object.

        Args:
            spread_id (str): spread identifier.
        """
        self.spread_id: str = spread_id
        tdv_kwargs = {"model_engine": model_engine, "owner": spread_id, "group": 'spread'}
        if tdv_additive_term_dim is None:
            self.tdv_add = make_proj_variable("additive_spread", **tdv_kwargs)
        else:  # store term structure
            self.tdv_add = make_proj_variable("additive_spread", dims=[tdv_additive_term_dim], **tdv_kwargs)
        if tdv_proportional_term_dim is None:
            self.tdv_propn = make_proj_variable("proportional_spread", **tdv_kwargs)
        else: # store term structure
            self.tdv_propn = make_proj_variable("proportional_spread", dims=[tdv_proportional_term_dim], **tdv_kwargs)
        self._last_update: int = self.time or 0

    @property
    def additive(self) -> float | npt.NDArray[np.float64]:
        """npt.NDArray[np.float64]: Additive spread as at period end."""
        maybe_raise_if_ne(self._last_update, self.time)
        return self._additive

    @property
    def proportional(self) -> float | npt.NDArray[np.float64]:
        """npt.NDArray[np.float64]: Proportional spread (multiplied by risk-free rates) as at period end."""
        maybe_raise_if_ne(self._last_update, self.time)
        return self._proportional

    def update(self, *, additive: float | npt.NDArray[np.float64] | None = None,
               proportional: float | npt.NDArray[np.float64] | None = None) -> None:
        """
        Update the spread parameters for the current time step.

        Args:
            additive (float | npt.NDArray[np.float64]): Additive spread.
            proportional (float | npt.NDArray[np.float64]): Proportional spread (multiplied by risk-free rates).
        """
        if additive is None and proportional is None:
           pass
        elif additive is None or proportional is None:
            raise ValueError(f"One or more parameters are None.")
        else:
            self._additive = additive
            self._proportional = proportional
        # self._on_exit_update()
        self._store_tdv(tdv=self.tdv_add, var=self._additive)
        self._store_tdv(tdv=self.tdv_propn, var=self._proportional)
        self._last_update = self.time

    def _store_tdv(self, tdv: TDimVariable, var: float | npt.NDArray[np.float64]) -> None:
        t = self.time
        arr_len = len(var) if isinstance(var, np.ndarray) else 0
        tdv_term_dim = [int(i) for i in tdv.dims[0]] if tdv.dims else None
        if arr_len == 0:  # scalar
            if tdv_term_dim is None:
                tdv[t] = var  # most often seen case, store a scalar value
            else:
                tdv[t] = np.full(len(tdv_term_dim), var)
        else:  # store term structure
            if tdv_term_dim is None:
                tdv[t] = var[min(arr_len, 12)]  # store first year value only
            else:
                tdv[t] = np.array([0 if i > arr_len else var[i] for i in tdv_term_dim])

    def __str__(self) -> str:
        return f"{type(self).__name__} - '{self.spread_id}'"
