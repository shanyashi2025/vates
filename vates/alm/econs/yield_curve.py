import numpy as np
import numpy.typing as npt
import pandas as pd
import warnings
from typing import Literal

from vates._core import ProjModelEngine, time_synchronized, TDimVariable
from vates.finmath import InterestRateTermStructure
from vates.utils import maybe_raise_if_ne


@time_synchronized
class YieldCurve:
    """
    Represents a yield curve and its derived rates.

    Attributes:
        curve_id (str): Yield curve identifier.
        _curve (InterestRateTermStructure): Yield curve (interest rate term structure)
    """
    time: int           # for type hint only, will be injected by decorator `add_projection_time_synchronizer`
    period: pd.Period   # for type hint only, will be injected by decorator `add_projection_time_synchronizer`
    _curve: InterestRateTermStructure | None

    __slots__ = ('__dict__', '__weakref__', 'time', 'period', '_last_update',
                 'curve_id', '_curve', 'tdv_spot_rates',)

    def __init__(
        self,
        curve_id: str,
        *,
        model_engine: ProjModelEngine | None = None,
        tdv_term_dim: list[int] | None = None,
    ) -> None:
        """
        Initialize a YieldCurve object.

        Args:
            curve_id (str): Unique identifier for this yield curve.
            tdv_term_dim (list[int] | None): List of terms (in months) to be output.
        """
        self.curve_id = curve_id
        self._curve: InterestRateTermStructure | None = None

        if tdv_term_dim is not None:
            for value in tdv_term_dim:
                if not isinstance(value, int):
                    tdv_term_dim = None
                    warnings.warn(f"'tdv_term_dim': type {type(value)} is not allowed, expected 'int'. Default is used.")
                    break
                if value < 0:
                    tdv_term_dim = None
                    warnings.warn(f"'tdv_term_dim': value {value} is not allowed, expected positive. Default is used.")
                    break
        tdv_term_dim = tdv_term_dim or [*range(12, 61, 12), *range(120, 601, 120)] # default: 1/2/3/4/5/10/20/30/40/50Y

        self.tdv_spot_rates: TDimVariable = TDimVariable("spot_rate", dims=[tdv_term_dim],
                                                         model_engine=model_engine, owner=curve_id, group='yield_curve')
        self._last_update: int = self.time or 0

    def update(self, *, from_what: Literal["spot_rates", "forward_rates", "discount_factors"] = None,
               value: npt.NDArray[np.float64] = None, is_unchange: bool = False) -> None:
        """
        Update the yield curve for the current time step.

        Args:
            from_what (Literal["spot_rates", "forward_rates", "discount_factors"]): Update from.
            value (npt.NDArray[np.float64]): Updated value.
            is_unchange (bool): True if kept unchanged, defaults to False.
        """
        if is_unchange:
            if any(x is not None for x in (from_what, value)):
                raise ValueError(f"is_unchange=True but other arguments are given.")
        elif any(x is None for x in (from_what, value)):
            raise ValueError(f"`from_what` or `value` is None.")
        elif from_what == "spot_rates":
            self._curve = InterestRateTermStructure.from_zeroac(value, interval_unit="M")
        elif from_what == "forward_rates":
            self._curve = InterestRateTermStructure.from_forwardac(value, interval_unit="M")
        elif from_what == "discount_factors":
            self._curve = InterestRateTermStructure.from_discount(value, interval_unit="M")
        else:
            raise ValueError(f"Invalid {from_what=}, expected: 'spot_rates', 'forward_rates' or 'discount_factors'.")

        self._on_exit_update()
        self._last_update = self.time

    @property
    def spot_rates(self) -> npt.NDArray[np.float64]:
        """npt.NDArray[np.float64] | None: Spot rates, or None if the curve has not been initialized."""
        maybe_raise_if_ne(self._last_update, self.time)
        return self._curve.spotac

    @property
    def discount_factors(self) -> npt.NDArray[np.float64]:
        """npt.NDArray[np.float64] | None: Discount factors, or None if the curve has not been initialized."""
        maybe_raise_if_ne(self._last_update, self.time)
        return self._curve.discount

    @property
    def forward_rates(self) -> npt.NDArray[np.float64]:
        """npt.NDArray[np.float64] | None: Forward rates, or None if the curve has not been initialized."""
        maybe_raise_if_ne(self._last_update, self.time)
        return self._curve.forwardac

    @property
    def par_yields(self) -> dict[int, npt.NDArray[np.float64]]:
        """dict[int, npt.NDArray[np.float64]] | None: Par yields, or None if the curve has not been initialized."""
        maybe_raise_if_ne(self._last_update, self.time)
        return self._curve.parac

    def _on_exit_update(self) -> None:
        if self._curve is None:
            raise ValueError("Yield curve has not been initialized; call `update(...)` before `is_unchange=True`.")
        t = self.time
        spots = self._curve.spotac
        n = len(spots)
        tdv_term_dim = (int(i) for i in self.tdv_spot_rates.dims[0])
        self.tdv_spot_rates[t] = np.array([0.0 if i >= n else spots[i] for i in tdv_term_dim])

    def __str__(self) -> str:
        return f"{type(self).__name__} - '{self.curve_id}'"