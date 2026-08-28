import numpy as np
import pandas as pd
import warnings
from typing import Self

from vates._core import ProjModelEngine
from vates.finmath import solve_ytm, solve_z_spread, InterestRateConvertor
from vates.alm.econs import Currency, YieldCurve, CreditBand
from vates.alm.enums import AssetClassification
from vates.alm.assets.bond_fixed import BondFixed
from vates.alm.assets._bond_fixed_component import (
    BondFixedParameters, BondFixedCashFlowGenerator, BondFixedCashFlowProvider, BondFixedPricer
)
from vates.alm.assets._utils import calculate_risk_adj_spot


class BondFixedBuilder:
    """
    Builder for creating and initializing BondFixed objects.
    """
    def __init__(
        self,
        model_engine: ProjModelEngine,
        asset_id: str,
        asset_category: str,
        fund_id: str,
        allocation_group: str,
        classification: AssetClassification | str,
        currency: Currency,
        units: float,
        issue_date: pd.Period,
        maturity_date: pd.Period,
        coupon_freq: int,
        face_value: float,
        rf_curve: YieldCurve | None,
        credit_band: CreditBand | None,
        is_profile: bool,
        provided_cash_flow_dict: dict[str, np.ndarray] | None = None,
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
            asset_category (str): Asset category.
            fund_id (str): Fund identifier.
            allocation_group (str): Allocation group for the bond.
            classification (AssetClassification): Asset classification.
            currency (Currency): Currency.
            units (float): Number of units.
            rf_curve (YieldCurve): Risk-free yield curve
            credit_band (CreditBand | None): Credit band (to provide assumptions, e.g. default and spread).
            issue_date (pd.Period): Issue date of the bond.
            maturity_date (pd.Period): Maturity date of the bond.
            coupon_freq (int): Coupon frequency per year, [0, 1, 2, 4, 12].
            redemp_sched (np.ndarray | None): Redemption schedule.
            face_value (float | None): Face value of the bond.
            coupon_rate (float | None): Coupon rate.
            mv_price (float | None): Market value price (dirty value).
            market_spread (float | None): Market spread.
            abv_price (float | None): Amortized book value price (dirty value).
            amort_rate (float | None): Amortization rate.
        """
        self.model_engine: ProjModelEngine = model_engine
        self.asset_id: str = asset_id
        self.asset_category: str = asset_category
        self.fund_id: str = fund_id
        self.allocation_group: str = allocation_group
        self.classification: AssetClassification = classification
        self.currency: Currency = currency
        self.units: float = units
        self.issue_date: pd.Period = issue_date
        self.maturity_date: pd.Period = maturity_date

        self.coupon_freq: int = coupon_freq
        self.face_value: float = face_value
        self.provided_cash_flow_dict: dict[str, np.ndarray] | None = provided_cash_flow_dict

        self.rf_curve: YieldCurve = rf_curve
        self.credit_band: CreditBand | None = credit_band
        self.is_profile: bool = is_profile

        self.coupon_rate: float | None = coupon_rate
        self.abv_price: float | None = abv_price
        self.amort_rate: float | None = amort_rate
        self.mv_price: float | None = mv_price
        self.market_spread: float | None = market_spread

    @property
    def p(self) -> pd.Period:
        return self.model_engine.period

    @property
    def t(self) -> int:
        return self.model_engine.time

    def derive_coupon_rate(self) -> Self:
        """
        Derive coupon rate based on yields at the point of purchase, assuming bond is purchased at par.
        """
        if self.coupon_freq != 0:
            if self.rf_curve.last_update != self.t:
                raise ValueError(f"Risk free curve is not updated on {self.t} ({self.p}).")
            if self.market_spread is None:
                self.market_spread = 0
                warnings.warn(f'{self.asset_id}: market_spread not specified, set to 0.')

            rf_spots = self.rf_curve.spot_rates

            if self.credit_band:
                if self.credit_band.last_update != self.t:
                    raise ValueError(f"Credit band is not updated on {self.t} ({self.p}).")
                spots = calculate_risk_adj_spot(
                    rf_spots=rf_spots,
                    mult=self.credit_band.credit_spotmult,
                    add=self.credit_band.credit_spread + self.market_spread
                )
            else:
                spots = rf_spots + self.market_spread

            n_months = (self.maturity_date - self.issue_date).n
            par_yield = InterestRateConvertor.spot_to_par(spots, freq=self.coupon_freq, time_interval=1/12)[n_months]

            self.coupon_rate = float(par_yield)

        else:  # coupon_freq == 0
            self.coupon_rate = 0.0

        return self

    def calculate_amort_rate(self) -> Self:
        """
        Calibrate the amortized rate using the set amortized book value price.
        """
        if self.abv_price is None:
            raise ValueError('abv_price need to be set first.')

        bond_params = BondFixedParameters(
            issue_date=self.issue_date,
            maturity_date=self.maturity_date,
            coupon_rate=self.coupon_rate,
            coupon_freq=self.coupon_freq,
            face_value=self.face_value,
        )

        if self.provided_cash_flow_dict is None:
            cash_flow_gen = BondFixedCashFlowGenerator(bond_params)
        else:
            cash_flow_gen = BondFixedCashFlowProvider(bond_params, self.provided_cash_flow_dict)

        freq = 1 if self.coupon_freq == 0 else self.coupon_freq  # 1 for zero coupon bond
        self.amort_rate = solve_ytm(
            target_pv=self.abv_price,
            cash_flows=cash_flow_gen.get_future_cash_flows(self.p),
            freq=freq,
            initial_guess=self.coupon_rate
        )

        return self

    def calibrate_market_spread(self) -> Self:
        """
        Calculate the market spread using the set market price.
        """
        if self.mv_price is None:
            raise ValueError('mv_price need to be set first.')
        if self.rf_curve.last_update != self.t:
            raise ValueError(f"Risk free curve not updated on {self.t} ({self.p}).")
        rf_spots = self.rf_curve.spot_rates

        if self.credit_band:
            spots = calculate_risk_adj_spot(
                rf_spots=rf_spots,
                mult=self.credit_band.credit_spotmult,
                add=self.credit_band.credit_spread
            )
        else:
            spots = rf_spots

        bond_params = BondFixedParameters(
            issue_date=self.issue_date,
            maturity_date=self.maturity_date,
            coupon_rate=self.coupon_rate,
            coupon_freq=self.coupon_freq,
            face_value=self.face_value,
        )

        if self.provided_cash_flow_dict is None:
            cash_flow_gen = BondFixedCashFlowGenerator(bond_params)
        else:
            cash_flow_gen = BondFixedCashFlowProvider(bond_params, self.provided_cash_flow_dict)

        self.market_spread = solve_z_spread(
            target_pv=self.mv_price,
            cash_flows=cash_flow_gen.get_future_cash_flows(self.p),
            spots=spots
        )

        return self

    def calculate_market_price(self) -> Self:
        """
        Calculate the market price using the set market spread.
        """
        if self.market_spread is None: raise ValueError('market_spread need to be set first.')
        if self.rf_curve.last_update != self.t: raise ValueError(f"Risk free curve not updated on {self.t} ({self.p}).")

        rf_spots = self.rf_curve.spot_rates
        if self.credit_band:
            spots = calculate_risk_adj_spot(
                rf_spots=rf_spots,
                mult=self.credit_band.credit_spotmult,
                add=self.credit_band.credit_spread + self.market_spread
            )
        else:
            spots = rf_spots + self.market_spread

        bond_params = BondFixedParameters(
            issue_date=self.issue_date,
            maturity_date=self.maturity_date,
            coupon_rate=self.coupon_rate,
            coupon_freq=self.coupon_freq,
            face_value=self.face_value,
        )
        if self.provided_cash_flow_dict is None:
            cash_flow_gen = BondFixedCashFlowGenerator(bond_params)
        else:
            cash_flow_gen = BondFixedCashFlowProvider(bond_params, self.provided_cash_flow_dict)
        pricer = BondFixedPricer(cash_flow_gen)

        self.mv_price = pricer.calculate_market_price(self.p, spots)

        return self

    def risk_neutralization(self) -> Self:
        """
        Set market spread to zero and goal seek the face value that gives market price.
        """
        if self.mv_price is None: raise ValueError('mv_price need to be set first.')
        if self.rf_curve.last_update != self.t: raise ValueError(f"Risk free curve not updated on {self.t} ({self.p}).")

        self.market_spread = 0
        rf_spots = self.rf_curve.spot_rates
        if self.credit_band:
            spots = calculate_risk_adj_spot(
                rf_spots=rf_spots,
                mult=self.credit_band.credit_spotmult,
                add=self.credit_band.credit_spread
            )
        else:
            spots = rf_spots

        bond_params = BondFixedParameters(
            issue_date=self.issue_date,
            maturity_date=self.maturity_date,
            coupon_rate=self.coupon_rate,
            coupon_freq=self.coupon_freq,
            face_value=self.face_value,
        )
        if self.provided_cash_flow_dict is None:
            cash_flow_gen = BondFixedCashFlowGenerator(bond_params)
        else:
            cash_flow_gen = BondFixedCashFlowProvider(bond_params, self.provided_cash_flow_dict)
        pricer = BondFixedPricer(cash_flow_gen)

        calc_price = pricer.calculate_market_price(self.p, spots)  # typically > mv_price if market_spread > 0
        self.face_value *= self.mv_price / calc_price  # scale face_value that gives mv_price

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
                    self.risk_neutralization()
                else:
                    warnings.warn(f"{type(self).__name__}: '{step}' is ignored as no method matches.")

        # Validate required data
        if self.mv_price is None: raise ValueError("mv_price is not yet set.")
        if self.market_spread is None: raise ValueError("market_spread is not yet set.")
        if self.abv_price is None: raise ValueError("abv_price is not yet set.")
        if self.amort_rate is None: self.calculate_amort_rate()

        # Validate initial price
        tolerance = max(self.mv_price * 1e-6, self.abv_price * 1e-6, 1e-8)
        if self.is_profile and abs(self.abv_price - self.mv_price) > tolerance:
            raise ValueError(f'Bond {self.asset_id} at {self.p}: abv_price != mv_price at purchase, '
                             f'abv={self.abv_price:.4f}, mv={self.mv_price:.4f}')

        # Validate timing constraints
        if not self.is_profile and self.issue_date > self.p:
            raise ValueError(f"Issue date of existing bond should not be later than {self.p}.")
        if self.is_profile and self.issue_date != self.p:
            raise ValueError(f"New bond must be initialized at purchase {self.issue_date}.")

        # Create the bond
        return BondFixed(
            model_engine=self.model_engine,
            asset_id=self.asset_id,
            asset_category=self.asset_category,
            fund_id=self.fund_id,
            allocation_group=self.allocation_group,
            classification=self.classification,
            currency=self.currency,
            units=self.units,
            issue_date=self.issue_date,
            maturity_date=self.maturity_date,
            coupon_rate=self.coupon_rate,
            coupon_freq=self.coupon_freq,
            face_value=self.face_value,
            provided_cash_flow_dict=self.provided_cash_flow_dict,
            mv_price=self.mv_price,
            market_spread=self.market_spread,
            abv_price=self.abv_price,
            amort_rate=self.amort_rate,
            rf_curve=self.rf_curve,
            credit_band=self.credit_band,
            is_profile=self.is_profile
        )