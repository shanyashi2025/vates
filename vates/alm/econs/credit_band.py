import numpy as np
import numpy.typing as npt
import pandas as pd

from vates._core import ProjModelEngine, TDepVariable


class CreditBand:
    """
    Represents credit information for a financial instrument.

    Attributes:
        band_id (str): credit band identifier.
        _spread (npt.NDArray[np.float64] | None): Credit spread(s).
        _spotmult (npt.NDArray[np.float64] | None): Spot rate multipliers.
        tdv_prob_of_default_ac (float): Probability of default (annual compounding).
        tdv_recovery_rate (float): Recovery rate.
    """

    def __init__(
        self,
        band_id: str,
        *,
        model_engine: ProjModelEngine | None = None,
    ) -> None:
        """
        Initialize a CreditRisk object.

        Args:
            band_id (str): credit band identifier.
        """
        self.band_id: str = band_id
        self._spread: npt.NDArray[np.float64] | None = None
        self._spotmult: npt.NDArray[np.float64] | None = None
        self._prob_of_default_ac: float | None = None
        self._recovery_rate: float | None = None
        self._last_update: int | None = None

        if model_engine is not None:
            model_engine.attach_time_observer(self)
            self.time: int = model_engine.time
            self._start_date: pd.Period = model_engine.START_DATE

        create_tdv = lambda name: TDepVariable(name, model_engine=model_engine, owner=band_id, group='credit')
        self.tdv_prob_of_default_ac: TDepVariable = create_tdv("prob_of_default_ac",)
        self.tdv_recovery_rate: TDepVariable = create_tdv("recovery_rate")

    def sync_time(self, subject) -> None:
        self.time = subject.time

    @property
    def period(self) -> pd.Period:
        return self._start_date + self.time

    @property
    def last_update(self) -> int | None:
        """int: Last update time index."""
        return self._last_update

    @property
    def credit_spread(self) -> npt.NDArray[np.float64]:
        """npt.NDArray[np.float64]: Credit spread(s) as at period end."""
        return self._spread

    @property
    def credit_spotmult(self) -> npt.NDArray[np.float64]:
        """npt.NDArray[np.float64]: Spot rate multipliers as at period end."""
        return self._spotmult

    @property
    def prob_of_default_ac(self) -> float | None:
        """float: Probability of default (annual compounding) in period."""
        return self._prob_of_default_ac

    @property
    def recovery_rate(self) -> float | None:
        """float: Recovery rate in period."""
        return self._recovery_rate

    def skip_update(self) -> None:
        self._last_update = self.period

    def update(self, prop_of_default_ac: float, recovery_rate: float, credit_spotmult: float | npt.NDArray[np.float64],
               credit_spread: float | npt.NDArray[np.float64]) -> None:
        """
        Update the credit parameters for the current time step.

        Args:
            prop_of_default_ac (float): Probability of default (annual compounding).
            recovery_rate (float): Recovery rate.
            credit_spotmult (float | npt.NDArray[np.float64]): Spot rate multipliers.
            credit_spread (float | npt.NDArray[np.float64]): Credit spread(s).
        """
        t = self.time
        self._prob_of_default_ac = prop_of_default_ac
        self._recovery_rate = recovery_rate
        self._spotmult = credit_spotmult
        self._spread = credit_spread
        self.tdv_prob_of_default_ac[t] = prop_of_default_ac
        self.tdv_recovery_rate[t] = recovery_rate
        self._last_update = t

    def __str__(self) -> str:
        return self.band_id