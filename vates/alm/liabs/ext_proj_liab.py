import pandas as pd

from vates._core import ProjModelEngine, TDimVariable
from vates.alm.econs import Currency
from vates.alm.liabs.liab_base import Liab
from vates.alm.liabs._utils import maybe_check_liab_state_roll, maybe_check_liab_state_close


class ExtProjLiab(Liab):
    """
    Externally projected liability.
    """

    __slots__ = ('_cash_flow', 'tdv_cash_flow', 'output_attrs_bd', 'output_attrs_ad')


    def __init__(
        self,
        *,
        model_engine: ProjModelEngine | None = None,
        liab_id: str | None = None,
        currency: Currency | None = None,
        entry_date: pd.Period | None = None,
        flex_attr_map: dict[str, str] | None = None,
        output_attrs_bd: list[str] | None = None,
        output_attrs_ad: list[str] | None = None,
        **kwargs
    ):
        """
        Initialize a ExtProjLiab object.

        Args:
            liab_id (str): Liability identifier.
            fund_id (str): Fund identifier.
            currency (Currency): Currency of the liability.
            entry_date (pd.Period): Entry date of the liability.
            flex_attr_map (dict[str, str]): Dict of string to named attribute.
            output_attrs_bd (list[str]): List of attributes to output.
            output_attrs_ad (list[str]): List of attributes to output.
        """
        super().__init__(model_engine=model_engine, liab_id=liab_id, currency=currency,
                         entry_date=entry_date, flex_attr_map=flex_attr_map)
        for key, val in kwargs.items():
            setattr(self, key, val)

        self._cash_flow: float = 0.0

        create_tdv = lambda name: TDimVariable(name, model_engine=model_engine, owner=liab_id, group='liability')
        self.tdv_cash_flow: TDimVariable = create_tdv("cash_flow")
        self.output_attrs_bd: list[TDimVariable] = [create_tdv(name)for name in (output_attrs_bd or [])]
        self.output_attrs_ad: list[TDimVariable] = [create_tdv(name) for name in (output_attrs_ad or [])]
        self._set_output_attrs_bd()
        self._set_output_attrs_ad()

    def _set_output_attrs_bd(self) -> None:
        t = self.time
        for item in self.output_attrs_bd:
            if not hasattr(self, item.name):
                setattr(self, item.name, 0.0)
            item[t] = getattr(self, item.name)

    def _set_output_attrs_ad(self) -> None:
        t = self.time
        for item in self.output_attrs_ad:
            if not hasattr(self, item.name):
                setattr(self, item.name, 0.0)
            item[t] = getattr(self, item.name)

    @maybe_check_liab_state_roll
    def roll_forward(self, cash_flow: float, **kwargs):
        """
        Roll the liability forward one period, updating variables and calculating cash flow.
        """
        self._cash_flow = cash_flow
        for key, val in kwargs.items():
            setattr(self, key, val)
        self.tdv_cash_flow[self.time] = self._cash_flow
        self._set_output_attrs_bd()

    @maybe_check_liab_state_close
    def close_dealing(self, **kwargs) -> None:
        """
        Update the liability after dealing, adjusting asset share.
        """
        for key, val in kwargs.items():
            setattr(self, key, val)
        self._set_output_attrs_ad()

    @property
    def cash_flow(self) -> float:
        return self._cash_flow

    @property
    def arr_cash_flow(self) -> TDimVariable:
        """TDepVariable: Cash flow array"""
        return self.tdv_cash_flow
