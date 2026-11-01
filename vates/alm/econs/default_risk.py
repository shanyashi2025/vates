import pandas as pd
from dataclasses import dataclass

from vates._core import ProjModelEngine, time_synchronized, make_proj_variable
from vates.utils import maybe_raise_if_ne


@dataclass(slots=True)
class DefaultRiskParams:
    """
    Represents default risk parameters.

    Attributes:
        default_risk_id (str): default risk identifier.
        marginal_prob_of_default (float): Marginal Probability of Default (MPD).
        pd_affects_decrement (bool): True if PD affects decrement.
        loss_given_default (float): Loss Given Default (LGD).
    """

    default_risk_id: str = ""
    marginal_prob_of_default: float = 0.0
    loss_given_default: float = 1.0
    pd_affects_decrement: bool = False

    @property
    def recovery_rate(self) -> float:
        """float: Recovery Rate (RR)."""
        return 1 - self.loss_given_default


@time_synchronized("model_engine.time_synchronizer")
class DefaultRisk:
    """
    Represents default risk information for a financial instrument.

    Attributes:
        default_risk_id (str): default risk identifier.
        _params (DefaultRiskParams): default risk parameters.
        tdv_mpd (float): Marginal Probability of Default (MPD).
        tdv_lgd (float): Loss Given Default (LGD).
    """
    time: int           # for type hint only, will be injected by decorator `time_synchronized`
    period: pd.Period   # for type hint only, will be injected by decorator `time_synchronized`
    _params: DefaultRiskParams

    __slots__ = ('__dict__', '__weakref__', 'time', 'period', '_last_update', 'default_risk_id', '_params',
                 'tdv_mpd', 'tdv_lgd',)

    def __init__(
        self,
        default_risk_id: str,
        *,
        model_engine: ProjModelEngine | None = None,
    ) -> None:
        """
        Initialize a DefaultRisk object.

        Args:
            default_risk_id (str): default risk identifier.
        """
        self.default_risk_id: str = default_risk_id
        self._params: DefaultRiskParams = DefaultRiskParams()
        tdv_kwargs = {"model_engine": model_engine, "owner": default_risk_id, "group": 'default_risk'}
        self.tdv_mpd = make_proj_variable("MPD", **tdv_kwargs)
        self.tdv_lgd = make_proj_variable("LGD", **tdv_kwargs)
        self._last_update: int = self.time or 0

    @property
    def marginal_prob_of_default(self) -> float:
        """float: Marginal Probability of Default (MPD)."""
        maybe_raise_if_ne(self._last_update, self.time)
        return self._params.marginal_prob_of_default

    @property
    def pd_affects_decrement(self) -> bool:
        """bool: True if PD affects decrement."""
        return self._params.pd_affects_decrement

    @property
    def loss_given_default(self) -> float:
        """float: Loss Given Default (LGD)."""
        maybe_raise_if_ne(self._last_update, self.time)
        return self._params.loss_given_default

    @property
    def recovery_rate(self) -> float:
        """float: Recovery Rate (RR)."""
        maybe_raise_if_ne(self._last_update, self.time)
        return self._params.recovery_rate

    def update(self, *, marginal_prob_of_default: float | None = None, loss_given_default: float | None = None,
               pd_affects_decrement: bool | None = None,) -> None:
        """
        Update the default risk parameters for the current time step.

        Args:
            marginal_prob_of_default (float): Marginal Probability of Default (MPD).
            pd_affects_decrement (bool): True if PD affects decrement.
            loss_given_default (float): Loss Given Default (LGD).
        """
        tup = (marginal_prob_of_default, loss_given_default, pd_affects_decrement)
        if all(x is None for x in tup):
            pass
        elif any(x is None for x in tup):
            raise ValueError(f"One or more parameters are None.")
        else:
            self._params.marginal_prob_of_default = marginal_prob_of_default
            self._params.loss_given_default = loss_given_default
            self._params.pd_affects_decrement = pd_affects_decrement

        t = self.time
        self.tdv_mpd[t] = self._params.marginal_prob_of_default
        self.tdv_lgd[t] = self._params.loss_given_default
        self._last_update = t

    def __str__(self) -> str:
        return f"{type(self).__name__} - '{self.default_risk_id}'"
