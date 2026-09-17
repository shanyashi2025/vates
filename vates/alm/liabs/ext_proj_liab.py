import pandas as pd

from vates._core import ProjModelEngine, TDimVariable
from vates.alm.econs import Currency
from vates.alm.liabs.liab_base import Liab
from vates.alm.liabs._utils import maybe_check_asset_state_roll, maybe_check_asset_state_close


class ExtProjLiab(Liab):
    """
    Externally projected liability.
    """

    __slots__ = ('_cash_flow', '_prem_inc', 'tdv_cash_flow', 'tdv_prem_inc',)

    _cash_flow: float
    _prem_inc: float

    def __init__(
        self,
        *,
        model_engine: ProjModelEngine | None = None,
        liab_id: str | None = None,
        currency: Currency | None = None,
        entry_date: pd.Period | None = None,
        flex_attr_map: dict[str, str] | None = None,
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
        """
        super().__init__(model_engine=model_engine, liab_id=liab_id, currency=currency,
                         entry_date=entry_date, flex_attr_map=flex_attr_map)
        for key, val in kwargs.items():
            setattr(self, key, val)

        create_tdv = lambda name: TDimVariable(name, model_engine=model_engine, owner=liab_id, group='liability')
        self.tdv_cash_flow: TDimVariable = create_tdv("cash_flow")
        self.tdv_prem_inc: TDimVariable = create_tdv("prem_inc")

    @maybe_check_asset_state_roll
    def roll_forward(self, cash_flow: float, prem_inc: float, **kwargs):
        """
        Roll the liability forward one period, updating variables and calculating cash flow.
        """
        self._cash_flow = cash_flow
        self._prem_inc = prem_inc
        for key, val in kwargs.items():
            setattr(self, key, val)

        self.tdv_cash_flow[self.time] = self._cash_flow
        self.tdv_prem_inc[self.time] = self._prem_inc

    @maybe_check_asset_state_close
    def close_dealing(self, **kwargs) -> None:
        """
        Update the liability after dealing, adjusting asset share.
        """
        for key, val in kwargs.items():
            setattr(self, key, val)

    @property
    def cash_flow(self) -> float:
        return self._cash_flow

    @property
    def prem_inc(self) -> float:
        return self._prem_inc

    @property
    def arr_cash_flow(self) -> TDimVariable:
        """TDepVariable: Cash flow array"""
        return self.tdv_cash_flow

    @property
    def arr_prem_inc(self) -> TDimVariable:
        """TDepVariable: Premium income array"""
        return self.tdv_prem_inc
