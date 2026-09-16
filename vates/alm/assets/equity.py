import pandas as pd
import warnings

from vates._core import ProjModelEngine, TDimVariable
from vates.alm.econs import Currency, EquityIndex
from vates.alm.assets.asset_base import Asset
from vates.alm.assets._utils import maybe_check_asset_state_roll, maybe_check_asset_state_close


class Equity(Asset):
    """
    Represents an equity asset in the portfolio.

    Attributes:
        _equity_index (EquityIndex): Associated equity index.
        _mv (float): Market value of the equity asset.
        _purchase_cost (float | None): Purchase cost of the equity asset.
        tdv_dividend (float): Dividend for the current period.
    """
    __slots__ = ('_equity_index', '_mv', '_purchase_cost', '_cash_flow', '_disposal_proceeds',
                 'tdv_cash_flow', 'tdv_dividend', 'tdv_mv_bd', 'tdv_mv_ad', 'tdv_purch_cost_bd', 'tdv_purch_cost_ad',)

    def __init__(
        self,
        *,
        market_value: float,
        equity_index: EquityIndex,
        purchase_cost: float | None = None,
        report_basis_to_attr: dict[str, str],
        model_engine: ProjModelEngine | None = None,
        asset_id: str = "",
        is_profile: bool = False,
        currency: Currency | None = None,
        asset_category: str = "",
        fund_id: str = "",
        allocation_group: str = "",
        purchase_date: pd.Period | None = None
    ):
        """
        Initialize an Equity asset.

        Args:
            asset_id (str): Asset identifier.
            is_profile (bool): Ture if profile asset, False if existing asset.
            currency (Currency): Asset currency.
            asset_category (str): Asset category.
            fund_id (str): Fund identifier.
            allocation_group (str): Allocation group.
            market_value (float): Market value.
            purchase_cost (float | None): Purchase cost.
            equity_index (EquityIndex): Associated equity index.
            report_basis_to_attr (dict[str, str]): Dict of asset reporting basis to named attribute.
            purchase_date (pd.Period | None): Purchase date, default to initilization date.
        """
        super().__init__(model_engine=model_engine, asset_id=asset_id, is_profile=is_profile, units=1,
                         purchase_date=purchase_date, currency=currency, report_basis_to_attr=report_basis_to_attr,
                         asset_category=asset_category, fund_id=fund_id, allocation_group=allocation_group)
        self._equity_index: EquityIndex = equity_index
        self._mv: float = market_value
        self._purchase_cost: float | None = purchase_cost

        if abs(self._mv) < 1e-8:
            self._mv = 1e-8  # to prevent crash when proportionally buy new asset
        if self._purchase_cost is not None and abs(self._purchase_cost) < 1e-8:
            self._purchase_cost = 1e-8

        self._cash_flow: float = 0.0
        self._disposal_proceeds: float = 0.0

        create_tdv = lambda name: TDimVariable(name, model_engine=model_engine, owner=asset_id, group='equity')
        self.tdv_cash_flow: TDimVariable = create_tdv("cash_flow")
        self.tdv_dividend: TDimVariable = create_tdv("dividend")
        self.tdv_mv_bd: TDimVariable = create_tdv("mv_bd")
        self.tdv_mv_ad: TDimVariable = create_tdv("mv_ad")
        self.tdv_purch_cost_bd: TDimVariable = create_tdv("purch_cost_bd")
        self.tdv_purch_cost_ad: TDimVariable = create_tdv("purch_cost_ad")

        if not is_profile:
            t = self.time
            self.tdv_mv_ad[t] = self._mv
            self.tdv_purch_cost_ad[t] = self._purchase_cost

    @property
    def is_alive(self) -> bool:
        return True

    @maybe_check_asset_state_roll
    def roll_forward(self, **kwargs) -> None:
        """
        Roll the equity asset forward one period, updating value and dividend.
        """
        self._disposal_proceeds = 0  # reset

        dividend = self._mv * self._equity_index.dividend_yield
        self._cash_flow = dividend
        self._mv = self._mv * (1 + self._equity_index.capital_growth)  # total return = capital growth + dividend yield

        t = self.time
        self.tdv_dividend[t] = dividend
        self.tdv_cash_flow[t] = self._cash_flow
        self.tdv_mv_bd[t] = self._mv
        self.tdv_purch_cost_bd[t] = self._purchase_cost
        self._state = ("rolled", t)

    def buy_propn(self, propn: float) -> None:
        """
        Buy a proportion of the equity asset.

        Args:
            propn (float): Proportion to buy.
        """
        if propn < 0:
            warnings.warn(f"Buying negative proportion ({propn:.4f}) of an exsiting equity '{self._asset_id}'.")
        amount = self._mv * propn
        self._mv += amount
        if self._purchase_cost is not None:
            self._purchase_cost += amount  # the difference between market value and purchase cost doesn't change (in dollar amount)

    def sell_propn(self, propn: float) -> None:
        """
        Sell a proportion of the equity asset.

        Args:
            propn (float): Proportion to sell.
        """
        if not (0 < propn <=1):
            warnings.warn(f"Buying proportion {propn:.4f} of an exsiting equity '{self._asset_id}', "
                          f"normally expected: 0 < proportion <=1.")
        amount = self._mv * propn
        self._mv -= amount
        if self._purchase_cost is not None:
            self._purchase_cost -= amount  # the difference between market value and purchase cost doesn't change (in dollar amount)
        self._disposal_proceeds += amount  # record the amount of money received when selling (proportion of) the equity

    @property
    def disposal_proceeds(self) -> float:
        return self._disposal_proceeds

    def buy_profile_scale(self, scale: float) -> None:
        """
        Scale the equity profile by a factor.

        Args:
            scale (float): Scaling factor.
        """
        if not self._is_profile:
            raise ValueError("This equity object is not a profile.")
        if scale < 0:
            warnings.warn(f"Scaling equity profile '{self._asset_id}' by a negative number ({scale:.4f}).")
        self._mv = self._mv * scale
        self._purchase_cost = self._mv  # purchased cost is determined as the initial carrying amount
        self._is_profile = False

    @maybe_check_asset_state_close
    def close_dealing(self, **kwargs) -> None:
        """
        Update the equity asset after dealing.
        """
        t = self.time
        self.tdv_mv_ad[t] = self._mv
        self.tdv_purch_cost_ad[t] = self._purchase_cost

    @property
    def market_value(self) -> float:
        """float: Market value of the equity asset."""
        return self._mv

    @property
    def purchase_cost(self) -> float | None:
        """float: Purchase cost of the equity asset."""
        return self._purchase_cost

    @property
    def cash_flow(self) -> float:
        """float: Cash flow in period"""
        return self._cash_flow

    @property
    def arr_cash_flow(self) -> TDimVariable:
        """TDepVariable: Cash flow array"""
        return self.tdv_cash_flow
