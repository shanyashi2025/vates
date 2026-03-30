import pandas as pd

from ..._core import TDepVariable
from ...utils import check_calc_time
from ..enums import AssetClassification
from ..econs import Currency, MarketInfo
from .asset_base import Asset


class Cash(Asset):
    """
    Represents a cash asset in the portfolio.

    Attributes:
        _nominal (float): Nominal value of the cash asset.
        _market_info (MarketInfo): Market information.
        _ret_id (str): Identifier of cash return.
        _ret_id_short_pos (str): Identifier of cash return on short cash positions.

    """
    __slots__ = ('_nominal', '_market_info', '_ret_id', '_ret_id_short_pos', 'tdv_cash_flow', 'tdv_mv_bd', 'tdv_mv_ad',)

    def __init__(self, model, asset_id: str, currency: Currency | None, asset_class: str, fund_id: str,
                 allocation_group: str, nominal: float, market_info: MarketInfo, ret_id: str, ret_id_short_pos: str,
                 classification: AssetClassification = AssetClassification.FVTPL, purchase_date: pd.Period | None=None):
        """
        Initialize a Cash asset.

        Args:
            asset_id (str): Asset identifier.
            currency (Currency): Asset currency.
            fund_id (str): Fund identifier.
            allocation_group (str): Allocation group.
            nominal (float): Nominal value.
            market_info (MarketInfo): Market information.
            ret_id (str): Identifier of cash return.
            ret_id_short_pos (str): Identifier of cash return on short cash positions.
            classification (AssetClassification): Asset classification. Defaults to FVTPL.
            purchase_date (pd.Period | None): Purchase date, default to initilization date.
        """
        super().__init__(model, asset_id, False, 1, purchase_date, currency, classification, asset_class,
                         fund_id, allocation_group)
        self._nominal: float = nominal
        self._market_info: MarketInfo = market_info
        self._ret_id: str = ret_id
        self._ret_id_short_pos: str = ret_id_short_pos

        self.tdv_cash_flow: TDepVariable = TDepVariable(model, "cash_flow", asset_id, 'cash')
        self.tdv_mv_bd: TDepVariable = TDepVariable(model, "mv_bd", asset_id, 'cash')
        self.tdv_mv_ad: TDepVariable = TDepVariable(model, "mv_ad", asset_id, 'cash')

        self.tdv_mv_ad[self.time] = self.mv

    @property
    def is_alive(self) -> bool:
        return True

    @property
    def nominal(self) -> float:
        return self._nominal

    @property
    def ret_id(self) -> str:
        return self._ret_id

    @property
    def ret_id_short_pos(self) -> str:
        return self._ret_id_short_pos

    @check_calc_time({"roll_forward": -1, "complete_dealing": -1}, "roll_forward")
    def roll_forward(self, **kwargs) -> None:
        """
        Roll the cash asset forward one period.
        """
        t = self.time
        if self._market_info.last_update != t:
            raise ValueError(f"{self._market_info.info_id} is not updated on {t} ({self.period}).")
        ret = self._market_info.data[self._ret_id if self._nominal >=0 else self._ret_id_short_pos]
        self._nominal = self._nominal * (1 + ret) ** (1 / 12)
        self.tdv_cash_flow[t] = 0.0
        self.tdv_mv_bd[t] = self.mv

    def invest_new_money(self, amount: float) -> None:
        """
        Invest new money into the cash asset.

        Args:
            amount (float): Amount to invest.
        """
        self._nominal += amount

    def buy_propn(self, propn: float) -> None:
        """
        Buy a proportion of the cash asset.

        Args:
            propn (float): Proportion to buy.
        """
        self._nominal += self._nominal * propn

    def sell_propn(self, propn: float) -> None:
        """
        Sell a proportion of the cash asset.

        Args:
            propn (float): Proportion to sell.
        """
        self._nominal -= self._nominal * propn

    def buy_profile_scale(self, *args, **kwargs) -> None:
        """
        Not applicable for cash assets.
        """
        # should never get here
        pass

    @check_calc_time({"complete_dealing": -1, "roll_forward": 0}, "complete_dealing")
    def complete_dealing(self, **kwargs) -> None:
        """
        Update the cash asset after dealing.
        """
        self.tdv_mv_ad[self.time] = self.mv

    @property
    def mv(self) -> float:
        """float: Market value of the cash asset."""
        return self._nominal

    @property
    def fav(self) -> float:
        """float: Funa accounting value of the cash asset."""
        return self._nominal

    @property
    def bsv(self) -> float:
        """float: Balance sheet value of the cash asset."""
        return self._nominal

    @property
    def cash_flow(self) -> float:
        """float: Cash flow in period"""
        return 0.0

    @property
    def arr_cash_flow(self) -> TDepVariable:
        """TDepVariable: Cash flow array"""
        return self.tdv_cash_flow
