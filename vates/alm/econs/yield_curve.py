from typing import Optional
from enum import Enum, unique
import numpy as np
import numpy.typing as npt
import pandas as pd
import weakref

from ..._core import TDepVariable
from ...utils import (
    convert_spot_to_disc,
    convert_disc_to_spot,
    convert_disc_to_fwrd,
    convert_disc_to_par,
)

@unique
class YieldCurveSpecifiedBy(Enum):
    """
    Enum for source types by which yield curve is specified.

    Attributes:
    SPOT: Yield curve is specified using spot rates, then derive other rates.
    ZCB: Yield curve is specified using ZCB prices (equivalent to DISC), then derive other rates.
    DISC: Yield curve is specified using discount factors (equivalent to ZCB), then derive other rates.
    """
    SPOT = "SPOT"
    ZCB = "ZCB"
    DISC = "DISC"


class YieldCurve:
    """
    Represents a yield curve and its derived rates.

    Attributes:
        curve_id (str): Yield curve identifier.
        _spot_rates: Spot rates.
        _disc_factors: Discount factors.
        _forward_rates: Forward rates.
        _par_yields: Par yields.
        _short_rate: Short rate
    """
    def __init__(self, model, curve_id: str, output_terms: list[int] | None = None) -> None:
        """
        Initialize a YieldCurve object.

        Args:
            curve_id (str): Unique identifier for this yield curve.
            output_terms (list[int] | None): List of terms (in months) to be output.
        """
        self._model_ref: weakref.ref = weakref.ref(model)
        self.curve_id = curve_id
        self._spot_rates: npt.NDArray[np.float64] | None = None
        self._disc_factors: npt.NDArray[np.float64] | None = None
        self._forward_rates: npt.NDArray[np.float64] | None = None
        self._par_yields: dict[int, Optional[npt.NDArray[np.float64]]] = {1: None, 2: None, 4: None, 12: None}
        self._short_rate: float = 0.0
        self._last_update: int | None = None

        if output_terms is not None:
            for value in output_terms:
                if type(value) != int:
                    raise TypeError(f"Argument 'output_terms': type {type(value)} is not allowed, expected 'int'.")
                if value < 0:
                    raise ValueError(f"Argument 'output_terms': value {value} is not allowed, expected positive.")
            self._output_terms = output_terms
        else:
            self._output_terms = [*range(12, 61, 12), *range(120, 601, 120)]

        self._tdv_spot_rates: TDepVariable = TDepVariable(model, "spot_rate", curve_id, 'yield_curve',
                                                          dims=[self._output_terms])

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

    def skip_update(self) -> None:
        self._last_update = self.time

    def update(self, rate_spec: YieldCurveSpecifiedBy, rates: npt.NDArray[np.float64], **kwargs) -> None:
        """
        Update the yield curve for the current time step.

        Args:
            rate_spec (YieldCurveSpecifiedBy): Using which the yield curve is specified (SPOT or ZCB / DISC).
            rates (npt.NDArray[np.float64]): Array of rates.

        Raises:
            ValueError: If rates are empty or rate_spec is invalid.
        """
        if kwargs.get('skip_update', False):
            self._last_update = self.time
            return

        len_rates = len(rates)
        if len_rates == 0: raise ValueError("Yield curve input cannot be empty")

        if self._disc_factors is not None:
            self._short_rate = 1 / self._disc_factors[1] - 1

        if rate_spec == YieldCurveSpecifiedBy.SPOT:
            self.spot_rates = rates
        elif rate_spec in (YieldCurveSpecifiedBy.ZCB, YieldCurveSpecifiedBy.DISC):
            self.disc_factors = rates
        else:
            raise ValueError(f"Yield curve can not be specified using {rate_spec=}")

        t = self.time
        self._tdv_spot_rates[t] = np.array([0 if i > len_rates else self._spot_rates[i] for i in self._output_terms])

        self._last_update = t

    @property
    def spot_rates(self) -> npt.NDArray[np.float64]:
        return self._spot_rates

    @spot_rates.setter
    def spot_rates(self, value: npt.NDArray[np.float64]) -> None:
        """
        Specify the yield curve using spot rates.

        Args:
            value (npt.NDArray[np.float64]): Spot rates.
        """
        self._spot_rates = value.copy()
        self._disc_factors = convert_spot_to_disc(self._spot_rates, "M")
        self._forward_rates = convert_disc_to_fwrd(self._disc_factors, "M")
        for n in self._par_yields:
            self._par_yields[n] = convert_disc_to_par(self._disc_factors, n, "M")

    @property
    def disc_factors(self) -> npt.NDArray[np.float64]:
        return self._spot_rates

    @disc_factors.setter
    def disc_factors(self, value: npt.NDArray[np.float64]) -> None:
        """
        Specify the yield curve using discount factors.

        Args:
            value (npt.NDArray[np.float64]): Discount factors.
        """
        self._disc_factors = value.copy()
        self._spot_rates = convert_disc_to_spot(self._disc_factors, "M")
        self._forward_rates = convert_disc_to_fwrd(self._disc_factors, "M")
        for n in self._par_yields:
            self._par_yields[n] = convert_disc_to_par(self._disc_factors, n, "M")

    @property
    def short_rate(self) -> float:
        """float: Short rate (1M forward rate) in period"""
        return self._short_rate

    @property
    def curve_data(self) -> dict[str, npt.NDArray[np.float64]]:
        """dict[str, npt.NDArray[np.float64]]: Yield curve as at period end."""
        return  {
            'spot': self._spot_rates,
            'discount': self._disc_factors,
            'forward': self._forward_rates,
            'par1': self._par_yields[1],
            'par2': self._par_yields[2],
            'par4': self._par_yields[4],
            'par12': self._par_yields[12],
        }

    def __str__(self) -> str:
        return self.curve_id