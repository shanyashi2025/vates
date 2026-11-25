import pandas as pd
from collections.abc import Mapping
from typing import Literal

from vates._core import ProjModelEngine, TDimVariable, make_proj_variable
from vates.utils import transition
from vates.alm.econs import Currency
from vates.alm.assets.asset_base import BaseAsset, AssetPhase


class Cash(BaseAsset):
    """
    Represents a cash asset in the portfolio.

    Attributes:
        _nominal (float): Nominal value of the cash asset.

    """
    __slots__ = ('_nominal', '_purchase_amount', '_sale_proceeds',
                 'tdv_mv_bd', 'tdv_mv_ad', 'tdv_purchase_amount', 'tdv_sale_proceeds',)

    def __init__(
        self,
        *,
        nominal: float,
        model_engine: ProjModelEngine | None = None,
        asset_id: str = "",
        currency: Currency | None = None,
        attr_aliases: Mapping[str, str] | None = None,
        entry_date: pd.Period | None = None,
        **kwargs
    ):
        """
        Initialize a Cash asset.

        Args:
            asset_id (str): Asset identifier.
            currency (Currency): Asset currency.
            nominal (float): Nominal value.
            attr_aliases (Mapping[str, str]): Mapping of alias to named attribute.
            entry_date (pd.Period | None): Entry date; defaults to the period at initialization.
        """
        super().__init__(model_engine=model_engine, asset_id=asset_id, is_profile=False,
                         entry_date=entry_date, currency=currency, attr_aliases=attr_aliases, **kwargs)
        self._nominal: float = self._ensure_non_zero(nominal)
        self._purchase_amount: float = 0.0
        self._sale_proceeds: float = 0.0

        create_tdv = lambda name: make_proj_variable(name, model_engine=model_engine, owner=asset_id, group='cash')
        self.tdv_mv_bd: TDimVariable = create_tdv("mv_bd")
        self.tdv_mv_ad: TDimVariable = create_tdv("mv_ad")
        self.tdv_purchase_amount: TDimVariable = create_tdv("purchase_amount")
        self.tdv_sale_proceeds: TDimVariable = create_tdv("sale_proceeds")

        self.tdv_mv_ad[self.time] = self._nominal

    def is_alive(self, at: Literal["bop", "eop"], /) -> bool:
        return True

    @property
    def nominal(self) -> float:
        return self._nominal

    def _on_time_change(self) -> None:
        self._purchase_amount = 0.0
        self._sale_proceeds = 0.0

    @transition(require_all=AssetPhase.CLOSED, require_offset=-1, require_not=AssetPhase.ROLLED, mark=AssetPhase.ROLLED)
    def roll_forward(self, *, rate_of_return: float, **kwargs) -> None:
        """
        Roll the cash asset forward one period.
        """
        t = self.time
        self._nominal = self._nominal * (1 + rate_of_return)
        self.tdv_mv_bd[t] = self._nominal

    def adjust_nominal(self, amount: float) -> None:
        """
        Adjust the nominal of the cash asset.

        Args:
            amount (float): Amount to adjust.
        """
        self._nominal += amount
        self._nominal = self._ensure_non_zero(self._nominal)
        if amount > 0.0:
            self._purchase_amount += amount
        else:
            self._sale_proceeds += -amount

    @transition(require_all=AssetPhase.ROLLED, require_not=AssetPhase.CLOSED)
    def buy_propn(self, propn: float) -> None:
        """
        Buy a proportion of the cash asset.

        Args:
            propn (float): Proportion to buy.
        """
        self.adjust_nominal(self._nominal * propn)

    @transition(require_all=AssetPhase.ROLLED, require_not=AssetPhase.CLOSED)
    def sell_propn(self, propn: float) -> None:
        """
        Sell a proportion of the cash asset.

        Args:
            propn (float): Proportion to sell.
        """
        self.adjust_nominal(self._nominal * -propn)

    def buy_profile(self, *args, **kwargs):
        """
        Not allowed for cash. Raises NotImplementedError.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Not applicable for 'Cash' instance.")

    @transition(require_all=AssetPhase.ROLLED, mark=AssetPhase.CLOSED)
    def close_dealing(self, **kwargs) -> None:
        """
        Update the cash asset after dealing.
        """
        t = self.time
        self.tdv_mv_ad[t] = self.market_value
        self.tdv_purchase_amount[t] = self._purchase_amount
        self.tdv_sale_proceeds[t] = self._sale_proceeds

    @property
    def market_value(self) -> float:
        """float: Market value of the cash asset."""
        return self._nominal

    @property
    def cash_flow(self) -> float:
        """float: Cash flow in period"""
        return 0.0

    @property
    def purchase_amount(self) -> float:
        """float: Cash paid to increase the cash asset during the period."""
        return self._purchase_amount

    @property
    def sale_proceeds(self) -> float:
        """float: Cash received from reducing the cash asset during the period."""
        return self._sale_proceeds

    def retrieve_cash_flow(self, *args) -> float:
        return 0.0
