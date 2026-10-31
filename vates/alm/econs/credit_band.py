import numpy as np
import numpy.typing as npt
import pandas as pd

from vates._core import ProjModelEngine, time_synchronized, make_proj_variable
from vates.utils import maybe_raise_if_ne


@time_synchronized("model_engine.time_synchronizer")
class CreditBand:
    """
    Represents credit information for a financial instrument.

    Attributes:
        band_id (str): credit band identifier.
        _spread_add (npt.NDArray[np.float64] | None): Additive spread.
        _spread_mult (npt.NDArray[np.float64] | None): Multiplicative spread (proportional to risk-free spot rates).
        tdv_mpd (float): Marginal Probability of Default (MPD).
        tdv_lgd (float): Loss Given Default (LGD).
    """
    time: int           # for type hint only, will be injected by decorator `time_synchronized`
    period: pd.Period   # for type hint only, will be injected by decorator `time_synchronized`
    _spread_add: float | npt.NDArray[np.float64]
    _spread_mult: float | npt.NDArray[np.float64]
    _mpd: float
    _lgd: float

    __slots__ = ('__dict__', '__weakref__', 'time', 'period', '_last_update',
                 'band_id', '_spread_add', '_spread_mult', '_mpd', '_lgd',
                 'tdv_mpd', 'tdv_lgd', 'tdv_spread_add', 'tdv_spread_mult',)

    def __init__(
        self,
        band_id: str,
        *,
        model_engine: ProjModelEngine | None = None,
        tdv_spread_add_term_dim: list[int] | None = None,
        tdv_spread_mult_term_dim: list[int] | None = None,
    ) -> None:
        """
        Initialize a CreditRisk object.

        Args:
            band_id (str): credit band identifier.
        """
        self.band_id: str = band_id
        tdv_kwargs = {"model_engine": model_engine, "owner": band_id, "group": 'credit'}
        self.tdv_mpd = make_proj_variable("MPD", **tdv_kwargs)
        self.tdv_lgd = make_proj_variable("LGD", **tdv_kwargs)
        if tdv_spread_add_term_dim is None:
            self.tdv_spread_add = make_proj_variable("spread_add", **tdv_kwargs)
        else:  # has term structure
            self.tdv_spread_add = make_proj_variable("spread_add", dims=[tdv_spread_add_term_dim], **tdv_kwargs)
        if tdv_spread_mult_term_dim is None:
            self.tdv_spread_mult = make_proj_variable("spread_mult", **tdv_kwargs)
        else: # has term structure
            self.tdv_spread_mult = make_proj_variable("spread_mult", dims=[tdv_spread_mult_term_dim], **tdv_kwargs)
        self._last_update: int = self.time or 0

    @property
    def spread_add(self) -> float | npt.NDArray[np.float64]:
        """npt.NDArray[np.float64]: Additive spread as at period end."""
        maybe_raise_if_ne(self._last_update, self.time)
        return self._spread_add

    @property
    def spread_mult(self) -> float | npt.NDArray[np.float64]:
        """npt.NDArray[np.float64]: Multiplicative spread (proportional to risk-free spot rates) as at period end."""
        maybe_raise_if_ne(self._last_update, self.time)
        return self._spread_mult

    @property
    def marginal_prob_of_default(self) -> float:
        """float: Marginal Probability of Default (MPD)."""
        maybe_raise_if_ne(self._last_update, self.time)
        return self._mpd

    @property
    def loss_given_default(self) -> float:
        """float: Loss Given Default (LGD)."""
        maybe_raise_if_ne(self._last_update, self.time)
        return self._lgd

    @property
    def recovery_rate(self) -> float:
        """float: Recovery Rate (RR)."""
        maybe_raise_if_ne(self._last_update, self.time)
        return 1 - self._lgd

    def update(self, *, mpd: float | None = None, lgd: float| None = None,
               spread_add: float | npt.NDArray[np.float64] | None = None,
               spread_mult: float | npt.NDArray[np.float64] | None = None,
               is_unchange: bool = False) -> None:
        """
        Update the credit parameters for the current time step.

        Args:
            mpd (float): Marginal Probability of Default (MPD).
            lgd (float): Loss Given Default (LGD).
            spread_add (float | npt.NDArray[np.float64]): Additive spread.
            spread_mult (float | npt.NDArray[np.float64]): Multiplicative spread (proportional to risk-free spot rates).
            is_unchange (bool): True if kept unchanged, defaults to False.
        """
        if is_unchange:
            if any(x is not None for x in (mpd, lgd, spread_mult, spread_add)):
                raise ValueError(f"is_unchange=True but parameters are provided.")
        else:
            if any(x is None for x in (mpd, lgd, spread_mult, spread_add)):
                raise ValueError(f"One or more parameters are None.")
            self._mpd = mpd
            self._lgd = lgd
            self._spread_mult = spread_mult
            self._spread_add = spread_add
        self._on_exit_update()
        self._last_update = self.time

    def _on_exit_update(self) -> None:
        t = self.time
        self.tdv_mpd[t] = self._mpd
        self.tdv_lgd[t] = self._lgd

        # tdv_spread_add
        arr_len = len(self._spread_add) if isinstance(self._spread_add, np.ndarray) else 0
        tdv_term_dim = [int(i) for i in self.tdv_spread_add.dims[0]] if self.tdv_spread_add.dims else None
        if arr_len == 0: # scalar
            if tdv_term_dim is None:
                self.tdv_spread_add[t] = self._spread_add  # most often seen case, store a scalar value
            else:
                self.tdv_spread_add[t] = np.full(len(tdv_term_dim), self._spread_add)
        else: # has term structure
            if tdv_term_dim is None:
                self.tdv_spread_add[t] = self._spread_add[min(arr_len, 12)]  # store first year value only
            else:
                self.tdv_spread_add[t] = np.array([0 if i > arr_len else self._spread_add[i] for i in tdv_term_dim])

        # tdv_spread_mult
        arr_len = len(self._spread_mult) if isinstance(self._spread_mult, np.ndarray) else 0
        tdv_term_dim = [int(i) for i in self.tdv_spread_mult.dims[0]] if self.tdv_spread_mult.dims else None
        if arr_len == 0: # scalar
            if tdv_term_dim is None:
                self.tdv_spread_mult[t] = self._spread_mult  # most often seen case, store a scalar value
            else:
                self.tdv_spread_mult[t] = np.full(len(tdv_term_dim), self._spread_mult)
        else: # has term structure
            if tdv_term_dim is None:
                self.tdv_spread_mult[t] = self._spread_mult[min(arr_len, 12)]  # store first year value only
            else:
                self.tdv_spread_mult[t] = np.array([0 if i > arr_len else self._spread_mult[i] for i in tdv_term_dim])

    def __str__(self) -> str:
        return f"{type(self).__name__} - '{self.band_id}'"