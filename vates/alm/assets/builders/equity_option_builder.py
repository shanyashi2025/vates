import math
import pandas as pd
import warnings
from typing import Self

from vates._core import ProjModelEngine
from vates.finmath import CallOrPut, BlackScholesCalculator
from vates.alm.econs import Currency, EquityIndex, YieldCurve
from vates.alm.assets.derivatives import EquityOption


class EquityOptionBuilder:
    """
    Builder for creating and initializing EquityOption objects.
    """
    def __init__(
        self,
        model_engine: ProjModelEngine,
        asset_id: str,
        asset_category: str,
        fund_id: str,
        allocation_group: str,
        currency: Currency,
        is_profile: bool,
        call_or_put: CallOrPut,
        exercise_date: pd.Period,
        units: float,
        stock_price: float,
        strike_price: float,
        equity_index: EquityIndex,
        rf_curve: YieldCurve,
        is_pay_dividend: bool,
        price: float | None = None,
        std_dev: float | None = None,
        *args,
        **kwargs
    ):
        """
        Initialize a EquityOptionBuilder with all required parameters.

        Args:
            model: Model object.
            asset_id (str): Asset identifier.
            asset_category (str): Asset category.
            fund_id (str): Fund identifier.
            allocation_group (str): Allocation group.
            currency (Currency): Asset currency.
            is_profile (bool): Ture if profile asset, False if existing asset.
            call_or_put (CallOrPut): Call or put option.
            exercise_date (pd.Period): Option exercise date.
            units (float): Number of equity option units.
            price (float): Option price (market value).
            stock_price (float): Underlying stock price.
            strike_price (float): Strike price.
            equity_index (EquityIndex): Associated equity index.
            rf_curve (YieldCurve): Risk-free curve.
            std_dev (float): Standard deviation, i.e. volatility.
            is_pay_dividend (bool): True if paying dividend, otherwise False.
        """
        self.model_engine: ProjModelEngine = model_engine
        self.asset_id: str = asset_id
        self.asset_category: str = asset_category
        self.fund_id: str = fund_id
        self.allocation_group: str = allocation_group
        self.currency: Currency = currency
        self.is_profile: bool = is_profile
        self.call_or_put: CallOrPut = call_or_put
        self.exercise_date: pd.Period = exercise_date
        self.units: float = units
        self.stock_price: float = stock_price
        self.strike_price: float = strike_price
        self.equity_index: EquityIndex = equity_index
        self.rf_curve: YieldCurve = rf_curve
        self.is_pay_dividend: bool = is_pay_dividend
        if self.os_term_m <= 0: raise ValueError(f'Equity option {self.asset_id} has alreay expired, can not be built.')

        self.price: float | None = price
        self.std_dev: float | None = std_dev

    @property
    def p(self) -> pd.Period:
        return self.model_engine.period

    @property
    def t(self) -> int:
        return self.model_engine.time

    @property
    def os_term_m(self) -> int:
        return (self.exercise_date - self.p).n

    def calculate_market_price(self) -> Self:
        """
        Calculate the market price using the volatility (standard deviation).
        """
        if self.is_pay_dividend and self.equity_index.last_update != self.t:
            raise ValueError(f"{self.equity_index.index_id} not updated on {self.t} ({self.p}).")
        if self.rf_curve.last_update != self.t:
            raise ValueError(f"{self.rf_curve.curve_id} not updated on {self.t} ({self.p}).")

        self.price = BlackScholesCalculator.price(
            call_or_put=self.call_or_put, s=self.stock_price, k=self.strike_price,
            r=math.log(1 + self.rf_curve.spot_rates[self.os_term_m]),
            q=math.log(1 + self.equity_index.dividend_yield_ac) if self.is_pay_dividend else 0.0, # continuously compounded
            sigma=self.std_dev, tau=self.os_term_m / 12
        )

        return self

    def calibrate_implied_volatility(self) -> Self:
        """
        Calculate the implied volatility.
        """
        if self.equity_index.last_update != self.t:
            raise ValueError(f"{self.equity_index.index_id} not updated on {self.t} ({self.p}).")
        if self.rf_curve.last_update != self.t:
            raise ValueError(f"{self.rf_curve.curve_id} not updated on {self.t} ({self.p}).")

        self.std_dev = BlackScholesCalculator.implied_volatility(
            call_or_put=self.call_or_put, price=self.price, s=self.stock_price, k=self.strike_price,
            r=math.log(1 + self.rf_curve.spot_rates[self.os_term_m]),
            q=math.log(1 + self.equity_index.dividend_yield_ac) if self.is_pay_dividend else 0.0, # continuously compounded
            tau=self.os_term_m / 12
        )

        return self

    def risk_neutralization(self) -> Self:
        return self.calibrate_implied_volatility()

    def build(self, pipeline: str | list[str] | None = None, pipe_operator: str = "|>") -> EquityOption:
        if pipeline:
            if isinstance(pipeline, str):
                pipeline = pipeline.split(pipe_operator)
            if not isinstance(pipeline, list):
                raise TypeError(f"Invalid type of pipeline {type(pipeline)}, expected 'list' or 'str'")

            for step in pipeline:
                step = step.lower()
                if step in ('calculate_market_price', 'market_price'):
                    self.calculate_market_price()
                elif step in ('calibrate_implied_volatility', 'calculate_implied_volatility', 'implied_volatility'):
                    self.calibrate_implied_volatility()
                elif step in ('risk_neutralization', 'risk_neutralize'):
                    self.risk_neutralization()
                else:
                    warnings.warn(f"{type(self).__name__}: '{step}' is ignored as no method matches.")

        if self.price is None: raise RuntimeError("price is not yet set.")
        if self.std_dev is None: raise RuntimeError("std_dev is not yet set.")

        return EquityOption(
            model_engine=self.model_engine,
            asset_id=self.asset_id,
            asset_category=self.asset_category,
            fund_id=self.fund_id,
            allocation_group=self.allocation_group,
            currency=self.currency,
            is_profile=self.is_profile,
            call_or_put=self.call_or_put,
            exercise_date=self.exercise_date,
            units=self.units,
            price=self.price,
            stock_price=self.stock_price,
            strike_price=self.strike_price,
            equity_index=self.equity_index,
            rf_curve=self.rf_curve,
            std_dev=self.std_dev,
            is_pay_dividend=self.is_pay_dividend
        )