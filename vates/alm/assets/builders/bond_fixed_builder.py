import math
import numpy as np
import pandas as pd
import warnings
from collections.abc import Mapping
from typing import Self

from vates._core import ProjModelEngine, time_synchronized
from vates.finmath import convert_interest_rates, solve_ytm, solve_z_spread
from vates.alm.econs import Currency, YieldCurve, CreditBand
from vates.alm.assets.bond_fixed import BondFixed
from vates.alm.assets._bond_fixed_component import (
    BondFixedParameters, BondFixedCashFlow, BondFixedCashFlowGenerator, BondFixedPricer
)
from vates.alm.assets._utils import calculate_risk_adjusted_spot_rates


_MISSING: object = object()

@time_synchronized("model_engine.time_synchronizer")
class BondFixedBuilder:
    """
    Builder for creating and initializing BondFixed objects.
    """

    time: int           # for type hint only, will be injected by decorator `time_synchronized`
    period: pd.Period   # for type hint only, will be injected by decorator `time_synchronized`

    def __init__(
        self,
        model_engine: ProjModelEngine,
        asset_id: str,
        currency: Currency,
        units: float,
        issue_date: pd.Period,
        maturity_date: pd.Period,
        coupon_freq: int,
        face_value: float,
        rf_curve: YieldCurve,
        credit_band: CreditBand | None,
        is_profile: bool,
        attr_aliases: Mapping[str, str] | None = None,
        fixed_cash_flow: BondFixedCashFlow | dict[str, np.ndarray] | None = None,
        coupon_rate: float | None = None,
        abv_price: float | None = None,
        amort_rate: float | None = None,
        mv_price: float | None = None,
        market_spread: float | None = None,
        *args,
        **kwargs
    ):
        """
        Initialize a BondBuilder with all required bond and market parameters.

        Args:
            model_engine: Model engine object.
            is_profile (bool): Ture if profile asset, False if existing asset.
            asset_id (str): Asset identifier.
            attr_aliases (Mapping[str, str]): Mapping of alias to named attribute.
            currency (Currency): Currency.
            units (float): Number of units.
            rf_curve (YieldCurve): Risk-free yield curve
            credit_band (CreditBand | None): Credit band (to provide assumptions, e.g. default and spread).
            issue_date (pd.Period): Issue date of the bond.
            maturity_date (pd.Period): Maturity date of the bond.
            coupon_freq (int): Coupon frequency per year, [0, 1, 2, 4, 12].
            face_value (float | None): Face value of the bond.
            fixed_cash_flow (BondFixedCashFlow | dict[str, np.ndarray] | None): Fixed cash flow of the bond.
            coupon_rate (float | None): Coupon rate.
            mv_price (float | None): Market value price (dirty value).
            market_spread (float | None): Market spread.
            abv_price (float | None): Amortized book value price (dirty value).
            amort_rate (float | None): Amortization rate.
        """
        self.model_engine: ProjModelEngine = model_engine
        self.asset_id: str = asset_id
        self.attr_aliases: Mapping[str, str] | None = attr_aliases
        self.currency: Currency = currency
        self.units: float = units
        self.issue_date: pd.Period = issue_date
        self.maturity_date: pd.Period = maturity_date

        self.coupon_freq: int = coupon_freq
        self.face_value: float = face_value

        if fixed_cash_flow is None or isinstance(fixed_cash_flow, BondFixedCashFlow):
            self.fixed_cash_flow: BondFixedCashFlow | None = fixed_cash_flow
        elif isinstance(fixed_cash_flow, dict):
            self.fixed_cash_flow: BondFixedCashFlow | None = BondFixedCashFlow(**fixed_cash_flow)
        else:
            raise TypeError(f"Invalid type of 'fixed_cash_flow', expected ('BondFixedCashFlow', 'dict').")

        self.rf_curve: YieldCurve = rf_curve
        self.credit_band: CreditBand | None = credit_band
        self.is_profile: bool = is_profile

        self.coupon_rate: float | None = coupon_rate
        self.abv_price: float | None = abv_price
        self.amort_rate: float | None = amort_rate
        self.mv_price: float | None = mv_price
        self.market_spread: float | None = market_spread

    @property
    def _params(self) -> BondFixedParameters:
        if self.coupon_rate is None:
            raise ValueError(f'{self.asset_id}: coupon_rate not specified.')
        return BondFixedParameters(
            issue_date=self.issue_date,
            maturity_date=self.maturity_date,
            coupon_rate=self.coupon_rate,
            coupon_freq=self.coupon_freq,
            face_value=self.face_value,
        )

    def _make_cash_flow_gen(self, params: BondFixedParameters | object = _MISSING) -> BondFixedCashFlowGenerator:
        if params is _MISSING:
            params = self._params
        return BondFixedCashFlowGenerator(bond_params=params, fixed_cash_flows=self.fixed_cash_flow)

    def _make_cash_flow(self, params: BondFixedParameters | object = _MISSING) -> np.ndarray:
        if params is _MISSING:
            params = self._params
        cash_flow_gen = self._make_cash_flow_gen(params=params)
        cash_flows = cash_flow_gen.get_future_cash_flows(self.period)
        if cash_flows is None:
            raise ValueError(f'{self.asset_id}: cash_flows (future) is None, bond has already matured.')
        return cash_flows

    def _make_pricer(self, params: BondFixedParameters | object = _MISSING) -> BondFixedPricer:
        if params is _MISSING:
            params = self._params
        return BondFixedPricer(cash_flow_generator=self._make_cash_flow_gen(params=params))

    @property
    def _rf_spot_rates(self) -> np.ndarray:
        return self.rf_curve.spot_rates

    def _make_ra_spot_rates(self, credit_band: CreditBand | object = _MISSING,
                            market_spread: float | object = _MISSING) -> np.ndarray:
        if credit_band is _MISSING:
            credit_band = self.credit_band
        if market_spread is _MISSING:
            market_spread = self.market_spread
        if market_spread is None:
            raise ValueError(f'{self.asset_id}: market_spread not specified.')
        if credit_band:
            return calculate_risk_adjusted_spot_rates(rf_spots=self._rf_spot_rates, mult=credit_band.spread_mult,
                                                      add=credit_band.spread_add + market_spread)
        else:
            return self._rf_spot_rates + market_spread

    def derive_coupon_rate(self) -> Self:
        """
        Derive coupon rate based on yields at the point of purchase, assuming bond is purchased at par.
        """
        if self.coupon_freq == 0:
            self.coupon_rate = 0.0
            return self

        if self.market_spread is None:
            self.market_spread = 0.0
            warnings.warn(f'{self.asset_id}: market_spread not specified, set to 0.')

        spots = self._make_ra_spot_rates()
        n_months = (self.maturity_date - self.issue_date).n
        par_yield = convert_interest_rates(
            spots, interval_unit="M", from_kind="spot", to_kind="par", coupon_frequency=self.coupon_freq)[n_months]
        self.coupon_rate = float(par_yield)

        return self

    def calculate_amort_rate(self) -> Self:
        """
        Calibrate the amortized rate using the set amortized book value price.
        """
        if self.abv_price is None:
            raise ValueError('abv_price need to be set first.')
        self.amort_rate = solve_ytm(
            target_pv=self.abv_price,
            cash_flows=self._make_cash_flow(),
            freq=1 if self.coupon_freq == 0 else self.coupon_freq,  # 1 for zero coupon bond
            initial_guess=self.coupon_rate or 0.0
        )
        return self

    def calibrate_market_spread(self) -> Self:
        """
        Calculate the market spread using the set market price.
        """
        if self.mv_price is None:
            raise ValueError('mv_price need to be set first.')
        self.market_spread = solve_z_spread(
            target_pv=self.mv_price,
            cash_flows=self._make_cash_flow(),
            spots=self._make_ra_spot_rates(market_spread=0.0)
        )
        return self

    def calculate_market_price(self) -> Self:
        """
        Calculate the market price using the set market spread.
        """
        pricer = self._make_pricer()
        self.mv_price = pricer.calculate_market_price(self.period, spots=self._make_ra_spot_rates())
        return self

    def risk_neutralize(self) -> Self:
        """
        Set market spread to zero and goal seek the face value that gives market price.
        """
        if self.mv_price is None:
            raise ValueError('mv_price need to be set first.')
        self.market_spread = 0.0
        pricer = self._make_pricer()
        calc_price = pricer.calculate_market_price(self.period, spots=self._make_ra_spot_rates())
        scale = self.mv_price / calc_price
        self.face_value *= scale  # scale face_value that makes mv_price
        if self.fixed_cash_flow is not None:
            self.fixed_cash_flow *= scale
        return self

    def build(self, pipeline: str | list[str] | None = None, pipe_operator: str = "|>") -> BondFixed:
        """
        Build and return a fully initialized BondFixed object.

        Returns:
            BondFixed: The constructed BondFixed object.

        Raises:
            ValueError: If required data is missing or timing constraints are violated.
        """
        if pipeline:
            if isinstance(pipeline, str):
                pipeline = pipeline.split(pipe_operator)
            if not isinstance(pipeline, list):
                raise TypeError(f"Invalid type of pipeline {type(pipeline)}, expected 'list' or 'str'")

            for step in pipeline:
                step = step.lower()
                if step in ('calibrate_market_spread', 'calculate_market_spread', 'market_spread'):
                    self.calibrate_market_spread()
                elif step in ('calculate_market_price', 'market_price'):
                    self.calculate_market_price()
                elif step in ('calculate_amort_rate', 'amort_rate'):
                    self.calculate_amort_rate()
                elif step in ('derive_coupon_rate', 'calculate_coupon_rate', 'coupon_rate'):
                    self.derive_coupon_rate()
                elif step in ('risk_neutralization', 'risk_neutralize'):
                    self.risk_neutralize()
                else:
                    warnings.warn(f"{type(self).__name__}: '{step}' is ignored as no method matches.")

        # Validate required data
        if self.coupon_rate is None: raise ValueError("coupon_rate is not yet set.")
        if self.mv_price is None: raise ValueError("mv_price is not yet set.")
        if self.market_spread is None: raise ValueError("market_spread is not yet set.")
        if self.abv_price is None: raise ValueError("abv_price is not yet set.")
        if self.amort_rate is None: self.calculate_amort_rate()

        # Validate initial price
        if self.is_profile and not math.isclose(self.abv_price, self.mv_price, rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError(f'Bond {self.asset_id} at {self.period}: abv_price != mv_price at purchase, '
                             f'abv={self.abv_price}, mv={self.mv_price}')

        # Validate timing constraints
        if not self.is_profile and self.issue_date > self.period:
            raise ValueError(f"Issue date of existing bond should not be later than {self.period}.")
        if self.is_profile and self.issue_date != self.period:
            raise ValueError(f"New bond must be initialized at purchase {self.issue_date}.")

        # Create the bond
        return BondFixed(
            model_engine=self.model_engine,
            asset_id=self.asset_id,
            attr_aliases=self.attr_aliases,
            currency=self.currency,
            units=self.units,
            issue_date=self.issue_date,
            maturity_date=self.maturity_date,
            coupon_rate=self.coupon_rate,
            coupon_freq=self.coupon_freq,
            face_value=self.face_value,
            fixed_cash_flow=self.fixed_cash_flow,
            mv_price=self.mv_price,
            market_spread=self.market_spread,
            abv_price=self.abv_price,
            amort_rate=self.amort_rate,
            rf_curve=self.rf_curve,
            credit_band=self.credit_band,
            is_profile=self.is_profile
        )