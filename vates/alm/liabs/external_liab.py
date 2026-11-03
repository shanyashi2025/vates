import pandas as pd
from typing import Callable, Self

from vates._core import ProjModelEngine, TDimVariable, make_proj_variable
from vates.utils import transition
from vates.alm.econs import Currency
from vates.alm.liabs.liab_base import Liab, LiabPhase

_MISSING: object = object()


class ExternalLiab(Liab):
    """
    Externally calculated liability.
    """

    __slots__ = ('_cash_flow', 'on_time_change_hook', 'roll_forward_hook', 'close_dealing_hook',
                 'tdv_cash_flow', 'tdv_list_bd', 'tdv_list_ad')

    def __init__(
        self,
        *,
        model_engine: ProjModelEngine | None = None,
        liab_id: str = "",
        currency: Currency | None = None,
        entry_date: pd.Period | None = None,
        attr_aliases: dict[str, str] | None = None,
        output_attrs_bd: list[str] | tuple[str, ...] = (),
        output_attrs_ad: list[str] | tuple[str, ...] = (),
        **kwargs
    ):
        """
        Initialize a ExternalLiab object.

        Args:
            liab_id (str): Liability identifier.
            fund_id (str): Fund identifier.
            currency (Currency): Currency of the liability.
            entry_date (pd.Period): Entry date of the liability.
            attr_aliases (dict[str, str]): Dict of alias to named attribute.
            output_attrs_bd (list[str] | tuple[str, ...]): List or tuple of attributes to output.
            output_attrs_ad (list[str] | tuple[str, ...]): List or tuple of attributes to output.
        """
        super().__init__(model_engine=model_engine, liab_id=liab_id, currency=currency,
                         entry_date=entry_date, attr_aliases=attr_aliases)
        for key, val in kwargs.items():
            setattr(self, key, val)

        self._cash_flow: float = 0.0
        self.on_time_change_hook: list[Callable[[Self], None]] = []
        self.roll_forward_hook: list[Callable[[Self], None]] = []
        self.close_dealing_hook: list[Callable[[Self], None]] = []

        create_tdv = lambda name: make_proj_variable(name, model_engine=model_engine, owner=liab_id, group='liability')
        self.tdv_cash_flow: TDimVariable = create_tdv("cash_flow")
        self.tdv_list_bd: list[TDimVariable] = [create_tdv(name) for name in output_attrs_bd]
        self.tdv_list_ad: list[TDimVariable] = [create_tdv(name) for name in output_attrs_ad]
        self._set_tdv(self.tdv_list_bd + self.tdv_list_ad)

    def _set_tdv(self, tdv_list: list[TDimVariable]) -> None:
        t = self.time
        for item in tdv_list:
            val = getattr(self, item.name, _MISSING)
            if val is not _MISSING:
                item[t] = val

    def register_on_time_change_hook(self, hook_func: Callable[[Self], None], /) -> None:
        self.on_time_change_hook.append(hook_func)

    def register_roll_forward_hook(self, hook_func: Callable[[Self], None], /) -> None:
        self.roll_forward_hook.append(hook_func)

    def register_close_dealing_hook(self, hook_func: Callable[[Self], None], /) -> None:
        self.close_dealing_hook.append(hook_func)

    def _on_time_change(self) -> None:
        self._cash_flow = 0.0
        for hook in self.on_time_change_hook:
            hook(self)

    @transition(require_all=LiabPhase.CLOSED, require_offset=-1, require_not=LiabPhase.ROLLED, mark=LiabPhase.ROLLED)
    def roll_forward(self, **kwargs):
        """
        Roll the liability forward one period, updating variables and calculating cash flow.
        """
        for key, val in kwargs.items():
            setattr(self, key, val)
        for hook in self.roll_forward_hook:
            hook(self)
        self.tdv_cash_flow[self.time] = self._cash_flow
        self._set_tdv(self.tdv_list_bd)

    @transition(require_all=LiabPhase.ROLLED, require_not=LiabPhase.CLOSED, mark=LiabPhase.CLOSED)
    def close_dealing(self, **kwargs) -> None:
        """
        Update the liability after dealing, adjusting asset share.
        """
        for key, val in kwargs.items():
            setattr(self, key, val)
        for hook in self.close_dealing_hook:
            hook(self)
        self._set_tdv(self.tdv_list_ad)

    @property
    @transition(require_all=LiabPhase.ROLLED)
    def cash_flow(self) -> float:
        return self._cash_flow

    @cash_flow.setter
    def cash_flow(self, value: float) -> None:
        self._cash_flow = value

    @property
    def arr_cash_flow(self) -> TDimVariable:
        """TDepVariable: Cash flow array"""
        return self.tdv_cash_flow
