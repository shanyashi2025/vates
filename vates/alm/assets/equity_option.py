import pandas as pd
import math

from ..._core import TDepVariable
from ...utils import check_calc_time
from ..enums import AssetClassification
from ..econs import Currency, EquityIndex, YieldCurve
from .asset_base import Asset
from ._derivative_pricer import CallOrPut, BlackScholesCalculator

class EquityOption(Asset):
    """
    Represents an equity option asset.

    Attributes:
        _call_or_put (CallOrPut): Call or put option.
        _exercise_date (pd.Period): Option exercise date.
        _price (float): Option price (market value).
        _stock_price (float): Underlying stock price.
        _strike_price (float): Strike price.
        _equity_index (EquityIndex): Associated equity index.
        _rf_curve (YieldCurve): Risk-free curve.
        _std_dev (float): Standard deviation, i.e. volatility.
        _is_pay_dividend (bool): True if paying dividend, otherwise False.
    """
    __slots__ = ('_call_or_put', '_exercise_date', '_price', '_stock_price', '_strike_price', '_equity_index',
                 '_rf_curve', '_std_dev', '_is_pay_dividend', '_cash_flow', 'tdv_units_bd', 'tdv_units_ad',
                 'tdv_cash_flow', 'tdv_stock_price', 'tdv_price', 'tdv_mv_bd', 'tdv_mv_ad',)

    def __init__(self, model, asset_id: str, is_profile: bool, units: float, currency: Currency | None, asset_class: str, 
                 fund_id: str, allocation_group: str, call_or_put: CallOrPut, exercise_date: pd.Period, price: float,
                 stock_price: float, strike_price: float, equity_index: EquityIndex, rf_curve: YieldCurve,
                 std_dev: float, is_pay_dividend: bool, classification: AssetClassification = AssetClassification.FVTPL,
                 purchase_date: pd.Period | None=None):
        """
        Initialize an EquityOption asset.

        Args:
            asset_id (str): Asset identifier.
            is_profile (bool): Ture if profile asset, False if existing asset.
            units (float): Number of equity option units, +ve/-ve means long/short position.
            currency (Currency): Asset currency.
            fund_id (str): Fund identifier.
            allocation_group (str): Allocation group.
            call_or_put (CallOrPut): Call or put option.
            exercise_date (pd.Period): Option exercise date.
            price (float): Option price (market value).
            stock_price (float): Underlying stock price.
            strike_price (float): Strike price.
            equity_index (EquityIndex): Associated equity index.
            rf_curve (YieldCurve): Risk-free curve.
            std_dev (float): Standard deviation, i.e. volatility.
            is_pay_dividend (bool): True if paying dividend, otherwise False.
            classification (AssetClassification): Asset classification. Defaults to FVTPL.
            purchase_date (pd.Period | None): Purchase date, default to initilization date.
        """
        super().__init__(model, asset_id, is_profile, units, purchase_date, currency, classification, asset_class,
                         fund_id, allocation_group)
        t = self.time
        self._call_or_put: CallOrPut = call_or_put
        self._exercise_date: pd.Period = exercise_date
        self._price: float = price
        self._stock_price: float = stock_price
        self._strike_price: float = strike_price
        self._equity_index: EquityIndex = equity_index
        self._rf_curve: YieldCurve = rf_curve
        self._std_dev: float = std_dev
        self._is_pay_dividend: bool = is_pay_dividend
        self._cash_flow: float = 0.0

        # validate initial price
        calc_price = BlackScholesCalculator.price(
            call_or_put=self._call_or_put, s=self._stock_price, k=self._strike_price,
            r=math.log(1 + self._rf_curve.curve_data['spot'][self.os_term_m]),
            q=math.log(1 + self._equity_index.dividend_yield_ac) if self._is_pay_dividend else 0.0,
            sigma=self._std_dev, tau=self.os_term_m / 12
        )
        tolerance = max(abs(price) * 1e-6, 1e-8)
        if abs(price - calc_price) > tolerance:
            raise ValueError(f"Equity option {self.asset_id}: price is calculated as {calc_price: .4f} "
                             f"based on std_dev, but input price is {price: .4f}.")

        # create array variables
        self.tdv_units_bd: TDepVariable = TDepVariable(model, "units_bd", asset_id, 'equity_option')
        self.tdv_units_ad: TDepVariable = TDepVariable(model, "units_ad", asset_id, 'equity_option')
        self.tdv_cash_flow: TDepVariable = TDepVariable(model, "cash_flow", asset_id, 'equity_option')
        self.tdv_stock_price: TDepVariable = TDepVariable(model, "stock_price", asset_id, 'equity_option')
        self.tdv_price: TDepVariable = TDepVariable(model, "price", asset_id, 'equity_option')
        self.tdv_mv_bd: TDepVariable = TDepVariable(model, "mv_bd", asset_id, 'equity_option')
        self.tdv_mv_ad: TDepVariable = TDepVariable(model, "mv_ad", asset_id, 'equity_option')

        if not is_profile:
            self.tdv_units_ad[t] = self._units
            self.tdv_stock_price[t] = self._stock_price
            self.tdv_price[t] = self._price
            self.tdv_mv_ad[t] = self.mv

    @property
    def std_dev(self) -> float:
        return self._std_dev

    @std_dev.setter
    def std_dev(self, value: float) -> None:
        self._std_dev = value

    @property
    def os_term_m(self) -> int:
        return max((self._exercise_date - self.period).n, 0)

    @property
    def is_alive(self) -> bool:
        return self.period < self._exercise_date

    @property
    def is_alive_beg(self) -> bool:
        return self.period <= self._exercise_date

    @check_calc_time({"roll_forward": -1, "complete_dealing": -1}, "roll_forward")
    def roll_forward(self, **kwargs) -> None:
        """
        Roll the equity option asset forward one period.
        """
        t = self.time

        if not self.is_alive_beg:
            self._units = 0
            self._price = 0
            self._cash_flow = 0
            return

        if self._equity_index.last_update != t:
            raise ValueError(f"{self._equity_index.index_id} not updated on {t} ({self.period}).")
        if self._rf_curve.last_update != t:
            raise ValueError(f"{self._rf_curve.curve_id} not updated on {t} ({self.period}).")

        if self._is_pay_dividend:
            self._stock_price = self._stock_price * (1 + self._equity_index.capital_growth)
            q = math.log(1 + self._equity_index.dividend_yield_ac) # obtain continuously compounded dividend assumption
        else:
            self._stock_price = self._stock_price * (1 + self._equity_index.total_return)
            q = 0.0

        if self.is_alive:
            self._cash_flow = 0.0
            self._price = BlackScholesCalculator.price(
                call_or_put=self._call_or_put, s=self._stock_price, k=self._strike_price,
                r=math.log(1 + self._rf_curve.curve_data['spot'][self.os_term_m]),
                q=q, sigma=self._std_dev, tau=self.os_term_m / 12
            )
        else:  # exercise at this month
            payoff = max(self._stock_price - self._strike_price, 0.0) if self._call_or_put == CallOrPut.CALL else \
                     max(self._strike_price - self._stock_price, 0.0)
            self._cash_flow = payoff * self._units
            self._price = 0.0

        self.tdv_units_bd[t] = self._units
        self.tdv_stock_price[t] = self._stock_price
        self.tdv_price[t] = self._price
        self.tdv_mv_bd[t] = self.mv
        self.tdv_cash_flow[t] = self._cash_flow

    def get_greeks(self) -> dict:
        """
        Get greeks for the equity option.

        Returns:
            dict[str, float]: dictionary of greeks.
        """
        return BlackScholesCalculator.greeks(
            call_or_put=self._call_or_put, s=self._stock_price, k=self._strike_price,
            r=math.log(1 + self._rf_curve.curve_data['spot'][self.os_term_m]),
            q=math.log(1 + self._equity_index.dividend_yield_ac) if self._is_pay_dividend else 0.0,
            sigma=self._std_dev, tau=self.os_term_m / 12
        )

    def buy_propn(self, *args, **kwargs) -> None:
        """
        Buy a proportion of the equity option asset.

        Raises:
            ValueError: Always.
        """
        raise ValueError("Can not buy equity option by scaling exsiting segment.")

    def sell_propn(self, propn: float) -> None:
        """
        Sell a proportion of the equity option asset.

        Args:
            propn (float): Proportion to sell.

        Raises:
            ValueError: If propn is negative.
        """
        if propn < 0: raise ValueError("Can not sell negative proportion of an exsiting equity option.")
        if propn > 1: raise ValueError("Can not sell >100% proportion of an exsiting equity option.")
        self._units -= self._units * propn

    def buy_profile_scale(self, scale: float, list_to_append: list | None=None) -> None:
        """
        Scale the equity option profile by a factor, positive/negative scale represents long/short.

        Args:
            scale (float): Scaling factor.
            list_to_append (list): Append to list.
        """
        if not self._is_profile: raise ValueError("This equity option object is not a profile.")
        self._units = self._units * scale
        self._is_profile = False
        if list_to_append is not None: list_to_append.append(self)

    @check_calc_time({"complete_dealing": -1, "roll_forward": 0}, "complete_dealing")
    def complete_dealing(self, **kwargs) -> None:
        """
        Update the equity option asset after dealing.
        """
        t = self.time
        self.tdv_units_ad[t] = self._units
        self.tdv_mv_ad[t] = self.mv

    @property
    def mv_price(self) -> float:
        return self._price

    @property
    def mv(self) -> float:
        """float: Market value of the equity option asset."""
        return self.mv_price * self._units

    @property
    def fav(self) -> float:
        """float: Fund accouting value of the equity option asset."""
        return self.mv_price * self._units

    @property
    def bsv(self) -> float:
        """float: Balance sheet value of the equity option asset."""
        return self.mv_price * self._units

    @property
    def cash_flow(self) -> float:
        """float: Cash flow in period"""
        return self._cash_flow

    @property
    def arr_cash_flow(self) -> TDepVariable:
        """TDepVariable: Cash flow array"""
        return self.tdv_cash_flow
