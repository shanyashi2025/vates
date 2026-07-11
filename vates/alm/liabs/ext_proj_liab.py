import pandas as pd

from vates._core import TDepVariable
from vates.utils import t_checker
from vates.alm.econs import Currency
from vates.alm.liabs.liab_base import Liab


class ExtProjLiab(Liab):
    """
    Externally projected liability.
    """
    __slots__ = ('tdv_cash_flow', 'tdv_prem_inc', 'tdv_num_pols', 'tdv_surr_val', 'tdv_math_res', 'tdv_acct_value_bd',
                 'tdv_acct_value_ad', 'tdv_asset_share_bd', 'tdv_asset_share_ad',)

    def __init__(self, model, liab_id: str, fund_id: str, currency: Currency,
                 entry_date: pd.Period, no_pols_if: float, surr_val_if: float, math_res_if: float,
                 acct_value_if: float=0, asset_share_if: float=0):
        """
        Initialize a ExtProjLiab object.

        Args:
            liab_id (str): Liability identifier.
            fund_id (str): Fund identifier.
            currency (Currency): Currency of the liability.
            entry_date (pd.Period): Entry date of the liability.
            no_pols_if (float): Number of policies in force.
            surr_val_if (float): Surrender value in force.
            math_res_if (float): Mathematical reserve in force.
            surr_val_if (float): Surrender value in force.
            math_res_if (float): Mathematical reserve in force.
            acct_value_if (float): Account value in force.
            asset_share_if (float): Asset share in force.
        """
        super().__init__(model, liab_id, fund_id, currency, entry_date, no_pols_if, surr_val_if, math_res_if,
                         acct_value_if, asset_share_if)
        self.tdv_cash_flow: TDepVariable = TDepVariable(model, "cash_flow", liab_id, 'liability')
        self.tdv_prem_inc: TDepVariable = TDepVariable(model, "prem_inc", liab_id, 'liability')
        self.tdv_num_pols: TDepVariable = TDepVariable(model, "no_pols_if", liab_id, 'liability')
        self.tdv_surr_val: TDepVariable = TDepVariable(model, "surr_val_if", liab_id, 'liability')
        self.tdv_math_res: TDepVariable = TDepVariable(model, "math_res_if", liab_id, 'liability')
        self.tdv_acct_value_bd: TDepVariable = TDepVariable(model, "acct_value_if_bd", liab_id, 'liability')
        self.tdv_acct_value_ad: TDepVariable = TDepVariable(model, "acct_value_if_ad", liab_id, 'liability')
        self.tdv_asset_share_bd: TDepVariable = TDepVariable(model, "asset_share_if_bd", liab_id, 'liability')
        self.tdv_asset_share_ad: TDepVariable = TDepVariable(model, "asset_share_if_ad", liab_id, 'liability')

        t = self.time
        self.tdv_num_pols[t] = self._num_pols
        self.tdv_surr_val[t] = self._surr_val
        self.tdv_math_res[t] = self._math_res
        self.tdv_acct_value_bd[t] = self._acct_value
        self.tdv_acct_value_ad[t] = self._acct_value
        self.tdv_asset_share_bd[t] = self._asset_share
        self.tdv_asset_share_ad[t] = self._asset_share

    @t_checker({"roll_forward": -1, "update_ad": -1}, "roll_forward")
    def roll_forward(self, **kwargs):
        """
        Roll the liability forward one period, updating variables and calculating cash flow.
        """
        t = self.time

        self._cash_flow = kwargs["cash_flow"]
        self._prem_inc = kwargs["prem_inc"]
        self._num_pols = kwargs["no_pols_if"]
        self._surr_val = kwargs["surr_val_if"]
        self._math_res = kwargs["math_res_if"]
        self._acct_value = kwargs.get("acct_value_if", 0.0)
        self._asset_share = kwargs.get("asset_share_if", 0.0)

        self.tdv_cash_flow[t] = self._cash_flow
        self.tdv_prem_inc[t] = self._prem_inc
        self.tdv_num_pols[t] = self._num_pols
        self.tdv_surr_val[t] = self._surr_val
        self.tdv_math_res[t] = self._math_res
        self.tdv_acct_value_bd[t] = self._acct_value
        self.tdv_asset_share_bd[t] = self._asset_share

    @t_checker({"update_ad": -1, "roll_forward": 0}, "update_ad")
    def update_ad(self, **kwargs) -> None:
        """
        Update the liability after dealing, adjusting asset share.
        """
        t = self.time

        self._acct_value = kwargs.get("acct_value_if", 0.0)
        self._asset_share = kwargs.get("asset_share_if", 0.0)

        self.tdv_acct_value_ad[t] = self._acct_value
        self.tdv_asset_share_ad[t] = self._asset_share

    @property
    def arr_cash_flow(self) -> TDepVariable:
        """TDepVariable: Cash flow array"""
        return self.tdv_cash_flow

    @property
    def arr_prem_inc(self) -> TDepVariable:
        """TDepVariable: Premium income array"""
        return self.tdv_prem_inc

    @property
    def arr_math_res(self) -> TDepVariable:
        """float: Mathematical reserve in force as at a date."""
        return self.tdv_math_res

    @property
    def arr_surr_val(self) -> TDepVariable:
        """float: Surrender value in force as at a date."""
        return self.tdv_surr_val

    @property
    def arr_acct_value_bd(self) -> TDepVariable:
        """float: Separate account value before dealing as at a date."""
        return self.tdv_acct_value_bd

    @property
    def arr_acct_value_ad(self) -> TDepVariable:
        """float: Separate account value after dealing as at a date."""
        return self.tdv_acct_value_bd

    @property
    def arr_asset_share_bd(self) -> TDepVariable:
        """float: Asset share before dealing as at a date."""
        return self.tdv_asset_share_bd

    @property
    def arr_asset_share_ad(self) -> TDepVariable:
        """float: Asset share after dealing as at a date."""
        return self.tdv_asset_share_ad
