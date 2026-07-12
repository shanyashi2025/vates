from typing import Optional
import numpy as np
import numpy.typing as npt
import pandas as pd

from vates._core import ProjModelEngine, TDepVariable
from vates.utils import convert_spot_to_disc, convert_disc_to_spot, convert_disc_to_fwrd, convert_disc_to_par, convert_fwrd_to_disc


class YieldCurve:
    """
    Represents a yield curve and its derived rates.

    Attributes:
        curve_id (str): Yield curve identifier.
        _spot_rates: Spot rates.
        _disc_factors: Discount factors.
        _forward_rates: Forward rates.
        _par_yields: Par yields.
    """
    def __init__(self, model_engine: ProjModelEngine, curve_id: str, output_terms: list[int] | None = None) -> None:
        """
        Initialize a YieldCurve object.

        Args:
            curve_id (str): Unique identifier for this yield curve.
            output_terms (list[int] | None): List of terms (in months) to be output.
        """
        model_engine.attach_time_observer(self)
        self.time: int = model_engine.time
        self._start_date: pd.Period = model_engine.START_DATE

        self.curve_id = curve_id
        self._spot_rates: npt.NDArray[np.float64] | None = None
        self._disc_factors: npt.NDArray[np.float64] | None = None
        self._forward_rates: npt.NDArray[np.float64] | None = None
        self._par_yields: dict[int, Optional[npt.NDArray[np.float64]]] = {1: None, 2: None, 4: None, 12: None}
        self._last_update: int | None = None

        if output_terms is not None:
            for value in output_terms:
                if not isinstance(value, int):
                    raise TypeError(f"Argument 'output_terms': type {type(value)} is not allowed, expected 'int'.")
                if value < 0:
                    raise ValueError(f"Argument 'output_terms': value {value} is not allowed, expected positive.")
            self._output_terms = output_terms
        else:
            self._output_terms = [*range(12, 61, 12), *range(120, 601, 120)]

        self._tdv_spot_rates: TDepVariable = TDepVariable(model_engine, "spot_rate", curve_id, 'yield_curve',
                                                          dims=[self._output_terms])

    def sync_time(self, subject: ProjModelEngine) -> None:
        self.time = subject.time

    @property
    def period(self) -> pd.Period:
        return self._start_date + self.time

    @property
    def last_update(self) -> int | None:
        """int: Last update time index."""
        return self._last_update

    def skip_update(self) -> None:
        self._last_update = self.time

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
        self._complete_update()

    @property
    def disc_factors(self) -> npt.NDArray[np.float64]:
        return self._disc_factors

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
        self._complete_update()

    @property
    def forward_rates(self) -> npt.NDArray[np.float64]:
        return self._forward_rates

    @forward_rates.setter
    def forward_rates(self, value: npt.NDArray[np.float64]) -> None:
        """
        Specify the yield curve using forward rates.

        Args:
            value (npt.NDArray[np.float64]): Forward rates.
        """
        self._forward_rates = value.copy()
        self._disc_factors = convert_fwrd_to_disc(self._forward_rates, "M")
        self._spot_rates = convert_disc_to_spot(self._disc_factors, "M")
        for n in self._par_yields:
            self._par_yields[n] = convert_disc_to_par(self._disc_factors, n, "M")
        self._complete_update()

    @property
    def par_yields(self) -> dict[int, npt.NDArray[np.float64]]:
        return self._par_yields

    def _complete_update(self) -> None:
        t = self.time
        len_rates = len(self._spot_rates)
        self._tdv_spot_rates[t] = np.array([0 if i > len_rates else self._spot_rates[i] for i in self._output_terms])
        self._last_update = t

    def __str__(self) -> str:
        return self.curve_id