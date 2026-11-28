import math
import pandas as pd
import warnings
from collections.abc import Mapping
from typing import Self, Literal

from vates._core import ProjModelEngine, time_synchronized
from vates.finmath import CallPut, BlackScholesCalculator
from vates.alm.econs import Currency, EquityIndex, YieldCurve
from vates.alm.assets.derivatives import EquityOption


@time_synchronized("model_engine")
class EquityOptionBuilder:
    """
    Builder for creating and initializing EquityOption objects.
    """

    time: int           # for type hint only, will be injected by decorator `time_synchronized`
    period: pd.Period   # for type hint only, will be injected by decorator `time_synchronized`

    def __init__(
        self,
        model_engine: ProjModelEngine,
        asset_id: str,
        currency: Currency,
        is_profile: bool,
        call_put: CallPut,
        exercise_date: pd.Period,
        qty: float,
        stock_price: float,
        strike_price: float,
        equity_index: EquityIndex,
        index_return_kind: Literal["total", "price"],
        rf_curve: YieldCurve,
        entry_date: pd.Period | None = None,
        attr_aliases: Mapping[str, str] | None = None,
        price: float  = float("nan"),
        volatility: float  = float("nan"),
        q: float = 0.0,
        **kwargs
    ):
        """
        Initialize a EquityOptionBuilder with all required parameters.

        Args:
            model: Model object.
            asset_id (str): Asset identifier.
            attr_aliases (Mapping[str, str]): Mapping of alias to named attribute.
            currency (Currency): Asset currency.
            is_profile (bool): Ture if profile asset, False if existing asset.
            entry_date (pd.Period | None): Entry date; defaults to the period at initialization.
            call_put (CallPut): Call or put option.
            exercise_date (pd.Period): Option exercise date.
            qty (float): Number of units, +ve/-ve represents long/short position.
            price (float): Option price (market value).
            stock_price (float): Underlying stock price.
            strike_price (float): Strike price.
            equity_index (EquityIndex): Associated equity index.
            index_return_kind (Literal["total", "price"]): Return series used to update the underlying stock price:
                "total" (total return) or "price" (price return).
            rf_curve (YieldCurve): Risk-free curve.
            volatility (float): Annualized volatility of the underlying (standard deviation of log returns).
            q (float): Continuous annual dividend yield, defaults to 0.0.
        """
        self.model_engine: ProjModelEngine = model_engine
        self.asset_id: str = asset_id
        self.attr_aliases: Mapping[str, str] | None = attr_aliases
        self.currency: Currency = currency
        self.is_profile: bool = is_profile
        self.entry_date: pd.Period | None = entry_date
        self.call_put: CallPut = call_put
        self.exercise_date: pd.Period = exercise_date
        self.qty: float = qty
        self.stock_price: float = stock_price
        self.strike_price: float = strike_price
        self.equity_index: EquityIndex = equity_index
        self.index_return_kind: Literal["total", "price"] = index_return_kind
        self.rf_curve: YieldCurve = rf_curve
        self.q: float = q
        if self.os_term_m <= 0: raise ValueError(f'Equity option {self.asset_id} has already expired, can not be built.')

        self.price: float = price
        self.volatility: float = volatility
        self.extra_kwargs = kwargs

    @property
    def os_term_m(self) -> int:
        return (self.exercise_date - self.period).n

    def calculate_market_price(self) -> Self:
        """
        Calculate the market price using the annualized volatility.
        """
        if math.isnan(self.volatility):
            raise ValueError('volatility is not yet set.')
        self.price = BlackScholesCalculator.price(
            call_put=self.call_put, s=self.stock_price, k=self.strike_price,
            r=math.log(1 + self.rf_curve.spot_rates[self.os_term_m]),
            q=self.q, sigma=self.volatility, tau=self.os_term_m / 12
        )

        return self

    def calibrate_implied_volatility(self) -> Self:
        """
        Calculate the implied annualized volatility.
        """
        if math.isnan(self.price):
            raise ValueError('price is not yet set.')
        self.volatility = BlackScholesCalculator.implied_volatility(
            call_put=self.call_put, price=self.price, s=self.stock_price, k=self.strike_price,
            r=math.log(1 + self.rf_curve.spot_rates[self.os_term_m]),
            q=self.q, tau=self.os_term_m / 12
        )

        return self

    def risk_neutralize(self) -> Self:
        return self.calibrate_implied_volatility()

    def build(self, pipeline: list[str] | None = None) -> EquityOption:
        pipeline = pipeline or []
        if not isinstance(pipeline, list):
            raise TypeError(f"Invalid type of pipeline {type(pipeline)}, expected 'list'")

        for step in pipeline:
            name = step.lower()
            if name in ('calculate_market_price', 'market_price'):
                self.calculate_market_price()
            elif name in ('calibrate_implied_volatility', 'calculate_implied_volatility', 'implied_volatility'):
                self.calibrate_implied_volatility()
            elif name in ('risk_neutralization', 'risk_neutralize'):
                self.risk_neutralize()
            else:
                warnings.warn(f"{type(self).__name__}: '{name}' is ignored as no method matches.")

        if math.isnan(self.price): raise ValueError("price is not yet set.")
        if math.isnan(self.volatility): raise ValueError("volatility is not yet set.")

        return EquityOption(
            model_engine=self.model_engine,
            asset_id=self.asset_id,
            attr_aliases=self.attr_aliases,
            currency=self.currency,
            is_profile=self.is_profile,
            entry_date=self.entry_date,
            call_put=self.call_put,
            exercise_date=self.exercise_date,
            qty=self.qty,
            price=self.price,
            stock_price=self.stock_price,
            strike_price=self.strike_price,
            equity_index=self.equity_index,
            index_return_kind=self.index_return_kind,
            rf_curve=self.rf_curve,
            volatility=self.volatility,
            q=self.q,
            **self.extra_kwargs
        )