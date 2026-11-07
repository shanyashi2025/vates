import pandas as pd
from collections.abc import Mapping
from typing import Literal

from vates._core import ProjModelEngine, TDimVariable, make_proj_variable
from vates.utils import transition
from vates.alm.econs import Currency
from vates.alm.assets.asset_base import Asset, AssetPhase


class Cash(Asset):
    """
    Represents a cash asset in the portfolio.

    Attributes:
        _nominal (float): Nominal value of the cash asset.

    """
    __slots__ = ('_nominal', '_purchase_proceeds', '_disposal_proceeds',
                 'tdv_cash_flow', 'tdv_mv_bd', 'tdv_mv_ad', 'tdv_purchase_proceeds', 'tdv_disposal_proceeds',)

    def __init__(
        self,
        *,
        nominal: float,
        model_engine: ProjModelEngine | None = None,
        asset_id: str = "",
        currency: Currency | None = None,
        attr_aliases: Mapping[str, str] | None = None,
        purchase_date: pd.Period | None = None
    ):
        """
        Initialize a Cash asset.

        Args:
            asset_id (str): Asset identifier.
            currency (Currency): Asset currency.
            nominal (float): Nominal value.
            attr_aliases (Mapping[str, str]): Mapping of alias to named attribute.
            purchase_date (pd.Period | None): Purchase date, default to initialization date.
        """
        super().__init__(model_engine=model_engine, asset_id=asset_id, is_profile=False,
                         purchase_date=purchase_date, currency=currency, attr_aliases=attr_aliases)
        self._nominal: float = self._ensure_non_zero(nominal)
        self._purchase_proceeds: float = 0.0
        self._disposal_proceeds: float = 0.0

        create_tdv = lambda name: make_proj_variable(name, model_engine=model_engine, owner=asset_id, group='cash')
        self.tdv_cash_flow: TDimVariable = create_tdv("cash_flow")
        self.tdv_mv_bd: TDimVariable = create_tdv("mv_bd")
        self.tdv_mv_ad: TDimVariable = create_tdv("mv_ad")
        self.tdv_purchase_proceeds: TDimVariable = create_tdv("purchase_proceeds")
        self.tdv_disposal_proceeds: TDimVariable = create_tdv("disposal_proceeds")

        self.tdv_mv_ad[self.time] = self._nominal

    def is_alive(self, at: Literal["begin", "end"], /) -> bool:
        return True

    @property
    def nominal(self) -> float:
        return self._nominal

    def _on_time_change(self) -> None:
        self._purchase_proceeds = 0.0
        self._disposal_proceeds = 0.0

    @transition(require_all=AssetPhase.CLOSED, require_offset=-1, require_not=AssetPhase.ROLLED, mark=AssetPhase.ROLLED)
    def roll_forward(self, *, rate_of_return: float, **kwargs) -> None:
        """
        Roll the cash asset forward one period.
        """
        t = self.time
        self._nominal = self._nominal * (1 + rate_of_return)
        self.tdv_cash_flow[t] = 0.0
        self.tdv_mv_bd[t] = self._nominal

    def buy_amount(self, amount: float) -> None:
        """
        Invest new money into the cash asset.

        Args:
            amount (float): Amount to invest.
        """
        self._nominal += amount
        self._nominal = self._ensure_non_zero(self._nominal)
        if amount > 0.0:
            self._purchase_proceeds += amount
        else:
            self._disposal_proceeds += -amount

    @transition(require_all=AssetPhase.ROLLED, require_not=AssetPhase.CLOSED)
    def buy_propn(self, propn: float) -> None:
        """
        Buy a proportion of the cash asset.

        Args:
            propn (float): Proportion to buy.
        """
        self.buy_amount(self._nominal * propn)

    @transition(require_all=AssetPhase.ROLLED, require_not=AssetPhase.CLOSED)
    def sell_propn(self, propn: float) -> None:
        """
        Sell a proportion of the cash asset.

        Args:
            propn (float): Proportion to sell.
        """
        self.buy_amount(self._nominal * -propn)

    def buy_profile(self, *args, **kwargs):
        """
        Not applicable for cash assets.
        """
        raise ValueError(f"`scale_profile` is not applicable for cash.")

    @transition(require_all=AssetPhase.ROLLED, mark=AssetPhase.CLOSED)
    def close_dealing(self, **kwargs) -> None:
        """
        Update the cash asset after dealing.
        """
        t = self.time
        self.tdv_mv_ad[t] = self.market_value
        self.tdv_purchase_proceeds[t] = self._purchase_proceeds
        self.tdv_disposal_proceeds[t] = self._disposal_proceeds

    @property
    def market_value(self) -> float:
        """float: Market value of the cash asset."""
        return self._nominal

    @property
    def cash_flow(self) -> float:
        """float: Cash flow in period"""
        return 0.0

    @property
    def purchase_proceeds(self) -> float:
        return self._purchase_proceeds

    @property
    def disposal_proceeds(self) -> float:
        return self._disposal_proceeds

    @property
    def arr_cash_flow(self) -> TDimVariable:
        """TDepVariable: Cash flow array"""
        return self.tdv_cash_flow
