import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Literal

from vates._core import ProjModelEngine, time_synchronized, TDimArrayVariable, make_proj_variable
from vates.finmath import InterestRateTermStructure
from vates.utils import maybe_raise_on_compare


@time_synchronized("model_engine")
class YieldCurve:
    """
    Represents a yield curve and its derived rates.

    Attributes:
        curve_id (str): Yield curve identifier.
        _curve (InterestRateTermStructure): Yield curve (interest rate term structure)
    """
    time: int           # for type hint only, will be injected by decorator `time_synchronized`
    period: pd.Period   # for type hint only, will be injected by decorator `time_synchronized`
    _curve: InterestRateTermStructure
    _default_output_terms: tuple[int, ...] = tuple((
        *range(12, 61, 12), *range(120, 601, 120)))  # default: 1/2/3/4/5/10/20/30/40/50Y

    __slots__ = ('__dict__', '__weakref__', 'time', 'period', '_last_update',
                 'curve_id', '_curve', '_output_terms', 'tdv_spot_rates',)

    def __init__(
        self,
        curve_id: str,
        *,
        model_engine: ProjModelEngine | None = None,
        output_terms: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        """
        Initialize a YieldCurve object.

        Args:
            curve_id (str): Unique identifier for this yield curve.
            output_terms (list[int] | None): List of terms (in months) to be output.
        """
        self.curve_id: str = curve_id
        self._curve: InterestRateTermStructure = InterestRateTermStructure.empty(1)
        self._output_terms: list[int] | tuple[int, ...] = (output_terms if output_terms is not None else
                                                           self._default_output_terms)

        self.tdv_spot_rates: TDimArrayVariable = make_proj_variable(
            "spot_rate", owner=curve_id, group='yield_curve', dims=[self._output_terms], model_engine=model_engine)
        self._last_update: int = self.time or 0

    def update(self, *,
               kind: Literal["spot_rates", "spot", "spotac", "zeroac", "forward_rates", "forward", "forwardac",
               "discount_factors", "discount"] | None = None,
               value: npt.NDArray[np.float64] | None = None) -> None:
        """
        Update the yield curve for the current time step.

        Args:
            kind (Literal["spot_rates", "forward_rates", "discount_factors"]): Update from.
            value (npt.NDArray[np.float64]): Updated value; if None, the previous value is retained.
        """
        if value is not None:
            if kind in ("spot_rates", "spot", "spotac", "zeroac"):
                self._curve = InterestRateTermStructure.from_zeroac(value, interval_unit="M")
            elif kind in ("forward_rates", "forward", "forwardac"):
                self._curve = InterestRateTermStructure.from_forwardac(value, interval_unit="M")
            elif kind in ("discount_factors", "discount"):
                self._curve = InterestRateTermStructure.from_discount(value, interval_unit="M")
            else:
                raise ValueError(f"Invalid {kind=}, expected: ('spot', 'forward', 'discount').")

        spots = self._curve.spotac
        n = len(spots)
        self.tdv_spot_rates[self.time] = np.array([0.0 if i >= n else spots[i] for i in self._output_terms])

        self._last_update = self.time

    @property
    def spot_rates(self) -> npt.NDArray[np.float64]:
        """npt.NDArray[np.float64] | None: Spot rates, or None if the curve has not been initialized."""
        maybe_raise_on_compare(self._last_update, self.time)
        return self._curve.spotac

    @property
    def discount_factors(self) -> npt.NDArray[np.float64]:
        """npt.NDArray[np.float64] | None: Discount factors, or None if the curve has not been initialized."""
        maybe_raise_on_compare(self._last_update, self.time)
        return self._curve.discount

    @property
    def forward_rates(self) -> npt.NDArray[np.float64]:
        """npt.NDArray[np.float64] | None: Forward rates, or None if the curve has not been initialized."""
        maybe_raise_on_compare(self._last_update, self.time)
        return self._curve.forwardac

    @property
    def par_yields(self) -> dict[int, npt.NDArray[np.float64]]:
        """dict[int, npt.NDArray[np.float64]] | None: Par yields, or None if the curve has not been initialized."""
        maybe_raise_on_compare(self._last_update, self.time)
        return self._curve.parac

    def __str__(self) -> str:
        return f"{type(self).__name__} - '{self.curve_id}'"