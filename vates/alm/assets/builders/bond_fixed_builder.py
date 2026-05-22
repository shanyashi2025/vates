import numpy as np
import pandas as pd
import warnings


from vates.utils import newton_raphson_ytm, newton_raphson_z_spread, calculate_risk_adj_spot, convert_spot_to_par
from vates.alm.econs import Currency, YieldCurve, CreditBand
from vates.alm.enums import AssetClassification
from vates.alm.assets.bond_fixed import BondFixed
from vates.alm.assets._bond_fixed_component import BondFixedParameters, BondFixedCashFlowGenerator, BondFixedPricer


class BondFixedBuilder:
    """
    Builder for creating and initializing BondFixed objects.
    """
    def __init__(self, **kwargs):
        """
        Initialize a BondBuilder with all required bond and market parameters.

        Args:
            model: Model object.
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
        self.model = kwargs['model']
        self.asset_id: str = kwargs['asset_id']
        self.asset_category: str = kwargs['asset_category']
        self.fund_id: str = kwargs['fund_id']
        self.allocation_group: str = kwargs['allocation_group']
        self.classification: AssetClassification = kwargs['classification']
        self.currency: Currency = kwargs['currency']
        self.units: float = kwargs['units']
        self.params = BondFixedParameters(
            issue_date=kwargs['issue_date'],
            maturity_date=kwargs['maturity_date'],
            coupon_rate=kwargs.get('coupon_rate', -9999),
            coupon_freq=kwargs['coupon_freq'],
            face_value=kwargs['face_value'],
            redemp_sched=kwargs['redemp_sched']
        )
        self.rf_curve: YieldCurve | None = kwargs['rf_curve']
        self.credit_band: CreditBand | None = kwargs['credit_band']
        self.is_profile: bool = kwargs['is_profile']

        self.abv_price: float | None = kwargs.get('abv_price', None)
        self.amort_rate: float | None = kwargs.get('amort_rate', None)
        self.mv_price: float | None = kwargs.get('mv_price', None)
        self.market_spread: float | None = kwargs.get('market_spread', None)

    @property
    def p(self) -> pd.Period:
        return self.model.period

    @property
    def t(self) -> int:
        return self.model.time

    def derive_coupon_rate(self) -> None:
        """
        Derive coupon rate based on yields at the point of purchase, assuming bond is purchased at par.
        """
        if self.params.coupon_freq != 0:
            if self.rf_curve.last_update != self.t:
                raise RuntimeError(f"Risk free curve is not updated on {self.t} ({self.p}).")
            if self.market_spread is None:
                self.market_spread = 0
                warnings.warn(f'{self.asset_id}: market_spread not specified, set to 0.')

            rf_spots = self.rf_curve.spot_rates

            if self.credit_band:
                if self.credit_band.last_update != self.t:
                    raise RuntimeError(f"Credit band is not updated on {self.t} ({self.p}).")
                spots = calculate_risk_adj_spot(
                    rf_spots=rf_spots,
                    mult=self.credit_band.credit_spotmult,
                    add=self.credit_band.credit_spread + self.market_spread
                )
            else:
                spots = rf_spots + self.market_spread

            n_months = (self.params.maturity_date - self.params.issue_date).n
            par_yield = convert_spot_to_par(spots=spots, payment_freq=self.params.coupon_freq, term_type='M')[n_months]

            self.params.coupon_rate = float(par_yield)

        else:  # coupon_freq == 0
            self.params.coupon_rate = 0.0

    def calculate_amort_rate(self) -> None:
        """
        Calibrate the amortized rate using the set amortized book value price.
        """
        if self.params.coupon_rate == -9999: raise ValueError('coupon_rate need to be set first.')
        if self.abv_price is None: raise ValueError('abv_price need to be set first.')

        cash_flow_gen = BondFixedCashFlowGenerator(self.params)

        freq = 1 if self.params.coupon_freq == 0 else self.params.coupon_freq  # 1 for zero coupon bond
        self.amort_rate = newton_raphson_ytm(
            target_pv=self.abv_price,
            cash_flows=cash_flow_gen.get_future_cash_flows(self.p),
            freq=freq,
            initial_guess=self.params.coupon_rate
        )

    def calibrate_market_spread(self) -> None:
        """
        Calculate the market spread using the set market price.
        """
        if self.params.coupon_rate == -9999: raise ValueError('coupon_rate need to be set first.')
        if self.mv_price is None: raise ValueError('mv_price need to be set first.')
        if self.rf_curve.last_update != self.t: raise RuntimeError(f"Risk free curve not updated on {self.t} ({self.p}).")
        rf_spots = self.rf_curve.spot_rates

        if self.credit_band:
            spots = calculate_risk_adj_spot(
                rf_spots=rf_spots,
                mult=self.credit_band.credit_spotmult,
                add=self.credit_band.credit_spread
            )
        else:
            spots = rf_spots

        cash_flow_gen = BondFixedCashFlowGenerator(self.params)

        self.market_spread = newton_raphson_z_spread(
            target_pv=self.mv_price,
            cash_flows=cash_flow_gen.get_future_cash_flows(self.p),
            spots=spots
        )

    def calculate_market_price(self) -> None:
        """
        Calculate the market price using the set market spread.
        """
        if self.params.coupon_rate == -9999: raise ValueError('coupon_rate need to be set first.')
        if self.market_spread is None: raise ValueError('market_spread need to be set first.')
        if self.rf_curve.last_update != self.t: raise RuntimeError(f"Risk free curve not updated on {self.t} ({self.p}).")

        rf_spots = self.rf_curve.spot_rates
        if self.credit_band:
            spots = calculate_risk_adj_spot(
                rf_spots=rf_spots,
                mult=self.credit_band.credit_spotmult,
                add=self.credit_band.credit_spread + self.market_spread
            )
        else:
            spots = rf_spots + self.market_spread

        cash_flow_gen = BondFixedCashFlowGenerator(self.params)
        pricer = BondFixedPricer(self.params, cash_flow_gen)

        self.mv_price = pricer.calculate_market_price(self.p, spots)

    def risk_neutralization(self) -> None:
        """
        Set market spread to zero and goal seek the face value that gives market price.
        """
        if self.mv_price is None: raise ValueError('mv_price need to be set first.')
        if self.params.coupon_rate == -9999: raise ValueError('coupon_rate need to be set first.')
        if self.rf_curve.last_update != self.t: raise RuntimeError(f"Risk free curve not updated on {self.t} ({self.p}).")

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

        cash_flow_gen = BondFixedCashFlowGenerator(self.params)
        pricer = BondFixedPricer(self.params, cash_flow_gen)

        calc_price = pricer.calculate_market_price(self.p, spots)  # typically > mv_price if market_spread > 0
        self.params.face_value *= self.mv_price / calc_price  # scale face_value that gives mv_price

    def build(self) -> BondFixed:
        """
        Build and return a fully initialized BondFixed object.

        Returns:
            BondFixed: The constructed BondFixed object.

        Raises:
            RuntimeError: If required data is missing or timing constraints are violated.
        """
        # Validate required data
        if self.params.coupon_rate == -9999: raise RuntimeError("coupon_rate is not yet set.")
        if self.mv_price is None: raise RuntimeError("mv_price is not yet set.")
        if self.market_spread is None: raise RuntimeError("market_spread is not yet set.")
        if self.abv_price is None: raise RuntimeError("abv_price is not yet set.")
        if self.amort_rate is None: self.calculate_amort_rate()

        # Validate initial price
        tolerance = max(self.mv_price * 1e-6, self.abv_price * 1e-6, 1e-8)
        if self.is_profile and abs(self.abv_price - self.mv_price) > tolerance:
            raise ValueError(f'Bond {self.asset_id} at {self.p}: abv_price != mv_price at purchase, '
                             f'abv={self.abv_price:.4f}, mv={self.mv_price:.4f}')

        # Validate timing constraints
        if not self.is_profile and self.params.issue_date > self.p:
            raise RuntimeError(f"Issue date of existing bond should not be later than {self.p}.")
        if self.is_profile and self.params.issue_date != self.p:
            raise RuntimeError(f"New bond must be initialized at purchase {self.params.issue_date}.")

        # Create the bond
        return BondFixed(
            model=self.model,
            asset_id=self.asset_id,
            asset_category=self.asset_category,
            fund_id=self.fund_id,
            allocation_group=self.allocation_group,
            classification=self.classification,
            currency=self.currency,
            units=self.units,
            issue_date=self.params.issue_date,
            maturity_date=self.params.maturity_date,
            coupon_rate=self.params.coupon_rate,
            coupon_freq=self.params.coupon_freq,
            face_value=self.params.face_value,
            redemp_sched=self.params.redemp_sched,
            mv_price=self.mv_price,
            market_spread=self.market_spread,
            abv_price=self.abv_price,
            amort_rate=self.amort_rate,
            rf_curve=self.rf_curve,
            credit_band=self.credit_band,
            is_profile=self.is_profile
        )