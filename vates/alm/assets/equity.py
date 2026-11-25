import pandas as pd
import warnings
from collections.abc import Mapping
from typing import Literal, Self

from vates._core import ProjModelEngine, TDimVariable, make_proj_variable
from vates.utils import transition, maybe_raise_on_compare
from vates.alm.econs import Currency, EquityIndex
from vates.alm.assets.asset_base import BaseAsset, AssetPhase


class Equity(BaseAsset):
    """
    Represents an equity asset in the portfolio.

    Attributes:
        _equity_index (EquityIndex): Associated equity index.
        _mv (float): Market value of the equity asset.
        _hist_cost (float): Cost basis of the equity asset.
        tdv_dividend (float): Dividend for the current period.
    """
    __slots__ = ('_equity_index', '_mv', '_hist_cost', '_cash_flow', '_dividend',
                 '_purchase_amount', '_sale_proceeds', '_n_clone',
                 'tdv_cash_flow', 'tdv_dividend', 'tdv_mv_bd', 'tdv_mv_ad', 'tdv_hist_cost_bd',
                 'tdv_hist_cost_ad', 'tdv_purchase_amount', 'tdv_sale_proceeds',)

    def __init__(
        self,
        *,
        market_value: float,
        equity_index: EquityIndex,
        historical_cost: float | None = None,
        attr_aliases: Mapping[str, str] | None = None,
        model_engine: ProjModelEngine | None = None,
        asset_id: str = "",
        is_profile: bool = False,
        currency: Currency | None = None,
        entry_date: pd.Period | None = None,
        **kwargs
    ):
        """
        Initialize an Equity asset.

        Args:
            asset_id (str): Asset identifier.
            is_profile (bool): Ture if profile asset, False if existing asset.
            currency (Currency): Asset currency.
            market_value (float): Market value.
            historical_cost (float | None): Historical cost.
            equity_index (EquityIndex): Associated equity index.
            attr_aliases (Mapping[str, str]): Mapping of alias to named attribute.
            entry_date (pd.Period | None): Entry date; defaults to the period at initialization.
        """
        super().__init__(model_engine=model_engine, asset_id=asset_id, is_profile=is_profile,
                         entry_date=entry_date, currency=currency, attr_aliases=attr_aliases, **kwargs)
        self._equity_index: EquityIndex = equity_index
        self._mv: float = self._ensure_non_zero(market_value)
        self._hist_cost: float = self._ensure_non_zero(historical_cost) if historical_cost is not None else self._mv
        self._n_clone: int = 0
        self._dividend: float = 0.0
        self._cash_flow: float = 0.0
        self._purchase_amount: float = 0.0
        self._sale_proceeds: float = 0.0

        create_tdv = lambda name: make_proj_variable(name, model_engine=model_engine, owner=asset_id, group='equity')
        self.tdv_cash_flow: TDimVariable = create_tdv("cash_flow")
        self.tdv_dividend: TDimVariable = create_tdv("dividend")
        self.tdv_mv_bd: TDimVariable = create_tdv("mv_bd")
        self.tdv_mv_ad: TDimVariable = create_tdv("mv_ad")
        self.tdv_hist_cost_bd: TDimVariable = create_tdv("historical_cost_bd")
        self.tdv_hist_cost_ad: TDimVariable = create_tdv("historical_cost_ad")
        self.tdv_purchase_amount: TDimVariable = create_tdv("purchase_amount")
        self.tdv_sale_proceeds: TDimVariable = create_tdv("sale_proceeds")

        t = self.time
        self.tdv_mv_ad[t] = self._mv
        self.tdv_hist_cost_ad[t] = self._hist_cost

    def is_alive(self, at: Literal["bop", "eop"], /) -> bool:
        return True

    def _on_time_change(self) -> None:
        self._dividend = 0.0
        self._cash_flow = 0.0
        self._purchase_amount = 0.0
        self._sale_proceeds = 0.0

    @transition(require_all=AssetPhase.CLOSED, require_offset=-1, require_not=AssetPhase.ROLLED, mark=AssetPhase.ROLLED)
    def roll_forward(self, **kwargs) -> None:
        """
        Roll the equity asset forward one period, updating value and dividend.
        """
        self._dividend = self._mv * self._equity_index.dividend_yield
        self._cash_flow = self._dividend
        self._mv = self._mv * (1 + self._equity_index.price_return)  # total return = price return + dividend yield

        t = self.time
        self.tdv_dividend[t] = self._dividend
        self.tdv_cash_flow[t] = self._cash_flow
        self.tdv_mv_bd[t] = self._mv
        self.tdv_hist_cost_bd[t] = self._hist_cost

    @transition(require_all=AssetPhase.ROLLED, require_not=AssetPhase.CLOSED)
    def buy_propn(self, propn: float) -> None:
        """
        Buy a proportion of the equity asset.

        Args:
            propn (float): Proportion to buy.
        """
        if propn < 0:
            warnings.warn(f"Buying equity '{self._asset_id}' with a negative proportion: {propn:.4f}.")
        amount = self._mv * propn
        self._purchase_amount += amount  # record the amount of money spent to buy (proportion of) the equity
        self._mv += amount
        self._mv = self._ensure_non_zero(self._mv)  # ensure that the value doesn't mathematically drop to zero
        self._hist_cost += amount
        self._hist_cost = self._ensure_non_zero(self._hist_cost)

    @transition(require_all=AssetPhase.ROLLED, require_not=AssetPhase.CLOSED)
    def sell_propn(self, propn: float) -> None:
        """
        Sell a proportion of the equity asset.
        Caveat: Historical cost is calculated using the Pro Rata method; for equities classified as FVOCI, opting for a
        FIFO or LIFO method instead of the Pro Rata method may lead to unexpected results regarding the investment
        returns and/or rate of return. For this case, you should consider using "profile" instance + `buy_profile`.

        Args:
            propn (float): Proportion to sell.
        """
        if not (0 < propn <=1):
            warnings.warn(f"Selling equity '{self._asset_id}' with a proportion: {propn:.4f}, "
                          f"normally expected: 0 < proportion <=1.")
        amount = self._mv * propn
        self._sale_proceeds += amount  # record the amount of money received from selling (proportion of) the equity
        self._mv -= amount
        self._mv = self._ensure_non_zero(self._mv)  # ensure that the value doesn't mathematically drop to zero
        self._hist_cost *= (1 - propn)  # Pro Rata
        self._hist_cost = self._ensure_non_zero(self._hist_cost)

    @property
    def purchase_amount(self) -> float:
        return self._purchase_amount

    @property
    def sale_proceeds(self) -> float:
        return self._sale_proceeds

    @transition(require_all=AssetPhase.PROFILE)
    def buy_profile(self, scale: float, *, new_asset_id: str | None = None) -> Self:
        """
        Scale the equity profile by a factor.

        Args:
            scale (float): Scaling factor.
            new_asset_id (str): Asset id for new asset.
        """
        if not self._is_profile:
            raise ValueError("This equity object is not a profile.")
        if scale < 0:
            warnings.warn(f"Buying equity profile '{self._asset_id}' with a negative scale: {scale:.4f}.")

        if new_asset_id is None:
            new_asset_id = self.asset_id + ("" if self._n_clone == 0 else f"_{self._n_clone}")

        amount = self._mv * scale

        clone = self.__class__(
            market_value=amount,
            historical_cost=amount,
            equity_index=self._equity_index,
            attr_aliases=self._attr_aliases,
            model_engine=self._model_ref(),
            asset_id=new_asset_id,
            is_profile=False,
            currency=self._currency,
            entry_date=self.period,
        )
        self._copy_dynamic_attrs_to(clone)
        clone._purchase_amount = amount
        self._n_clone += 1
        return clone

    @transition(require_all=AssetPhase.ROLLED, mark=AssetPhase.CLOSED)
    def close_dealing(self, **kwargs) -> None:
        """
        Update the equity asset after dealing.
        """
        t = self.time
        self.tdv_mv_ad[t] = self._mv
        self.tdv_hist_cost_ad[t] = self._hist_cost
        self.tdv_purchase_amount[t] = self._purchase_amount
        self.tdv_sale_proceeds[t] = self._sale_proceeds

    @property
    def market_value(self) -> float:
        """float: Market value of the equity asset."""
        return self._mv

    @property
    def historical_cost(self) -> float:
        """float: Historical cost of the equity asset."""
        return self._hist_cost

    @property
    @transition(require_all=AssetPhase.ROLLED)
    def cash_flow(self) -> float:
        """float: Cash flow in period"""
        return self._cash_flow

    @property
    @transition(require_all=AssetPhase.ROLLED)
    def dividend(self) -> float:
        """float: Dividend in period"""
        return self._dividend

    def retrieve_cash_flow(self, t: int) -> float:
        maybe_raise_on_compare(t, must_ge=0, must_le=self.time)
        return self.tdv_cash_flow[t]