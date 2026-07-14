import numpy as np
import numpy.typing as npt
import pandas as pd
import warnings
from typing import Optional

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
        self._spot_rates: npt.NDArray[np.float64] | None = None
        self._disc_factors: npt.NDArray[np.float64] | None = None
        self._forward_rates: npt.NDArray[np.float64] | None = None
        self._par_yields: dict[int, Optional[npt.NDArray[np.float64]]] = {1: None, 2: None, 4: None, 12: None}
        self._last_update: int | None = None

        if model_engine is not None:
            model_engine.attach_time_observer(self)
            self.time: int = model_engine.time
            self._start_date: pd.Period = model_engine.START_DATE

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
        self._tdv_term_dim = tdv_term_dim or [*range(12, 61, 12), *range(120, 601, 120)] # default: 1/2/3/4/5/10/20/30/40/50Y

        self._tdv_spot_rates: TDepVariable = TDepVariable("spot_rate", dims=[self._tdv_term_dim],
                                                          model_engine=model_engine, owner=curve_id, group='yield_curve')

    def sync_time(self, subject) -> None:
        self.time = subject.time

    @property
    def period(self) -> pd.Period:
        return self._start_date + self.time

    @property
    def last_update(self) -> int | None:
        """int: Last update time index."""
        return self._last_update

    def no_change_on_update(self) -> None:
        self._on_exit_update()

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
        self._on_exit_update()

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
        self._on_exit_update()

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
        self._on_exit_update()

    @property
    def par_yields(self) -> dict[int, npt.NDArray[np.float64]]:
        return self._par_yields

    def _on_exit_update(self) -> None:
        t = self.time
        len_rates = len(self._spot_rates)
        self._tdv_spot_rates[t] = np.array([0 if i > len_rates else self._spot_rates[i] for i in self._tdv_term_dim])
        self._last_update = t

    def __str__(self) -> str:
        return f"{type(self).__name__} - '{self.curve_id}'"