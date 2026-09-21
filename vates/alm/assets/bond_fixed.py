import math
import numpy as np
import numpy.typing as npt
import pandas as pd
import warnings
from collections.abc import Mapping
from typing import Self

from vates._core import ProjModelEngine, TDimVariable
from vates.utils import transition
from vates.alm.econs import Currency, YieldCurve, CreditBand
from vates.alm.assets.asset_base import Asset, AssetPhase
from vates.alm.assets._bond_fixed_component import (
    BondFixedParameters,
    BondFixedCashFlowGenerator,
    BondFixedCashFlowProvider,
    BondFixedPricer,
    BondFixedRiskCalculator,
)
from vates.alm.assets._utils import calculate_risk_adj_spot


class BondFixed(Asset):
    """
    Represents a bond asset in the portfolio.

    Attributes:
        _params (BondFixedParameters): Bond parameters.
        _mv_price_dirty (float): Market value price (dirty).
        _market_spread (float): Market spread.
        _abv_price_dirty (float): Amortized book value price (dirty).
        _amort_rate (float): Amortization rate.
        _rf_curve (YieldCurve): Risk-free yield curve.
        _credit_band (CreditBand | None): Credit band (to provide assumptions, e.g. default and spread).
        cash_flow_gen (BondFixedCashFlowGenerator): Cash flow generator.
        pricer (BondFixedPricer): Bond pricer.
        risk_calc (BondFixedRiskCalculator): Risk calculator.
    """
    __slots__ = ('_params', '_mv_price_dirty', '_market_spread', '_abv_price_dirty', '_amort_rate', '_rf_curve',
                 '_credit_band', '_cash_flow', 'cash_flow_gen', 'pricer', 'risk_calc', '_n_clone',
                 'tdv_units_default', 'tdv_units_maturity', 'tdv_units_bd', 'tdv_units_ad', 'tdv_cash_flow',
                 'tdv_interest', 'tdv_principal', 'tdv_default_recovery', 'tdv_mv_price', 'tdv_abv_price',
                 'tdv_mv_bd', 'tdv_abv_bd', 'tdv_mv_ad', 'tdv_abv_ad',)

    
    def __init__(
        self,
        *,
        issue_date: pd.Period,
        maturity_date: pd.Period,
        coupon_rate: float,
        coupon_freq: int,
        face_value: float,
        mv_price: float,
        market_spread: float,
        abv_price: float,
        amort_rate: float,
        rf_curve: YieldCurve,
        attr_aliases: Mapping[str, str] | None = None,
        model_engine: ProjModelEngine | None = None,
        asset_id: str | None = None,
        is_profile: bool = False,
        units: float = 1.0,
        currency: Currency | None = None,
        provided_cash_flow_dict: dict[str, np.ndarray] | None = None,
        credit_band: CreditBand | None = None,
        purchase_date: pd.Period | None = None,
        _bypass_init_validation: bool = False,
    ):
        """
        Initialize a Bond asset.

        Args:
            asset_id (str): Asset identifier.
            is_profile (bool): Ture if profile asset, False if existing asset.
            units (float): Number of bond units.
            currency (Currency): Asset currency.
            attr_aliases (Mapping[str, str]): Mapping of alias to named attribute.
            issue_date (pd.Period): Issue date of the bond.
            maturity_date (pd.Period): Maturity date of the bond.
            coupon_rate (float): Coupon rate.
            coupon_freq (int): Coupon frequency per year, [0, 1, 2, 4, 12].
            face_value (float): Face value of the bond.
            provided_cash_flow_dict (np.ndarray | None): Dict of provided bond cash flows.
            mv_price (float): Market value price (dirty value).
            market_spread (float): Market spread.
            abv_price (float): Amortized book value price (dirty value).
            amort_rate (float): Amortization rate.
            rf_curve (YieldCurve): Risk-free yield curve.
            credit_band (CreditBand | None): Credit band (to provide assumptions, e.g. default and spread).
            purchase_date (pd.Period | None): Purchase date, default to initilization date.
            _bypass_init_validation (bool): True to bypass initial validation. Defaults to False.
        """
        super().__init__(model_engine=model_engine, asset_id=asset_id, is_profile=is_profile, units=units,
                         purchase_date=purchase_date, currency=currency, attr_aliases=attr_aliases)

        self._params = BondFixedParameters(
            issue_date=issue_date,
            maturity_date=maturity_date,
            coupon_rate=coupon_rate,
            coupon_freq=coupon_freq,
            face_value=face_value,
        )
        self._mv_price_dirty: float = mv_price
        self._market_spread: float = market_spread
        self._abv_price_dirty: float = abv_price
        self._amort_rate: float = amort_rate
        self._rf_curve: YieldCurve = rf_curve
        self._credit_band: CreditBand | None = credit_band
        self._cash_flow: float = 0.0
        self._n_clone: int = 0

        # Compose with specialized components
        if provided_cash_flow_dict is None:
            self.cash_flow_gen = BondFixedCashFlowGenerator(self._params)
        else:
            self.cash_flow_gen = BondFixedCashFlowProvider(self._params, provided_cash_flow_dict)
        self.pricer = BondFixedPricer(self.cash_flow_gen)
        self.risk_calc = BondFixedRiskCalculator(self.pricer)

        # validate profile price
        if is_profile and not math.isclose(self._abv_price_dirty, self._mv_price_dirty, rel_tol=1e-6, abs_tol=1e-8):
            ValueError(f"Profile bond {self.asset_id}: abv_price={self._abv_price_dirty:.4f} != "
                       f"mv_price={self._mv_price_dirty:.4f}")

        if not _bypass_init_validation:
            if self.time is None:
                warnings.warn(f"'time' is None, can\'t perform init validation.")
            calc_price = self.pricer.calculate_market_price(self.period, self.ra_spots)
            if not math.isclose(self._mv_price_dirty, calc_price, rel_tol=1e-6, abs_tol=1e-8):
                raise ValueError(f"Bond {asset_id} mv_price {self._mv_price_dirty:.4f} != calculated {calc_price:.4f}.")
            calc_price = self.pricer.calculate_amortized_price(self.period, self._amort_rate)
            if not math.isclose(self._abv_price_dirty, calc_price, rel_tol=1e-6, abs_tol=1e-8):
                raise ValueError(f"Bond {asset_id} abv_price {self._abv_price_dirty:.4f} != calculated {calc_price:.4f}.")

        # Initialize TDepVariable
        create_tdv = lambda name: TDimVariable(name, model_engine=model_engine, owner=asset_id, group='bond')
        self.tdv_units_default: TDimVariable = create_tdv("units_default")
        self.tdv_units_maturity: TDimVariable = create_tdv("units_maturity")
        self.tdv_units_bd: TDimVariable = create_tdv("units_bd")
        self.tdv_units_ad: TDimVariable = create_tdv("units_ad")
        self.tdv_cash_flow: TDimVariable = create_tdv("cash_flow")
        self.tdv_interest: TDimVariable = create_tdv("interest")
        self.tdv_principal: TDimVariable = create_tdv("principal")
        self.tdv_default_recovery: TDimVariable = create_tdv("default_recovery")
        self.tdv_mv_price: TDimVariable = create_tdv("mv_price")
        self.tdv_abv_price: TDimVariable = create_tdv("abv_price")
        self.tdv_mv_bd: TDimVariable = create_tdv("mv_bd")
        self.tdv_abv_bd: TDimVariable = create_tdv("abv_bd")
        self.tdv_mv_ad: TDimVariable = create_tdv("mv_ad")
        self.tdv_abv_ad: TDimVariable = create_tdv("abv_ad")

        t = self.time
        self.tdv_mv_price[t] = self._mv_price_dirty
        self.tdv_abv_price[t] = self._abv_price_dirty
        if not is_profile:
            self.tdv_units_ad[t] = self._units
            self.tdv_mv_ad[t] = self.mv_price * self._units
            self.tdv_abv_ad[t] = self.abv_price * self._units

    @property
    def mv_price(self) -> float:
        return self._mv_price_dirty

    @property
    def abv_price(self) -> float:
        return self._abv_price_dirty

    @property
    def market_value(self) -> float:
        """float: Market value of the bond asset."""
        return self._mv_price_dirty * self._units

    @property
    def amortized_book_value(self) -> float:
        """float: Amortized book value of the bond asset."""
        return self._abv_price_dirty * self._units

    @property
    def market_spread(self) -> float:
        return self._market_spread

    @property
    def amort_rate(self) -> float:
        return self._amort_rate

    @property
    def ra_spots(self) -> npt.NDArray[np.float64]:
        """
        Get risk-adjusted spot rates for the bond.

        Returns:
            npt.NDArray[np.float64]: Risk-adjusted spot rates.
        """
        rf_spots = self._rf_curve.spot_rates

        if rf_spots is None:
            raise ValueError("rf_curve.spot_rates is None. Yield curve must be initialized.")
        if self._credit_band:
            return calculate_risk_adj_spot(
                rf_spots=rf_spots,
                mult=self._credit_band.credit_spotmult,
                add=self._credit_band.credit_spread + self._market_spread
            )
        else:
            return rf_spots + self._market_spread

    @property
    def os_term_m(self) -> int | None:
        if self._params.issue_date <= self.period <= self._params.maturity_date:
            return (self._params.maturity_date - self.period).n  # gives: n, n-1, ..., 1, 0 during lifespan
        return None

    @property
    def is_alive(self) -> bool:
        return self._params.issue_date <= self.period < self._params.maturity_date

    @property
    def is_alive_beg(self) -> bool:
        return self._params.issue_date < self.period <= self._params.maturity_date

    def calculate_risk_metrics(self, eff_dur_delta: float = 0.001):
        """
        Calculate all risk metrics for the bond.

        Args:
            eff_dur_delta (float): Delta yiled curve for effective duration calculation.
        """
        return self.risk_calc.calculate_all_risk_metrics(
            valn_date=self.period, market_price=self._mv_price_dirty, spots=self.ra_spots, eff_dur_delta=eff_dur_delta
        )

    @transition(require_all=AssetPhase.CLOSED, require_offset=-1, require_not=AssetPhase.ROLLED, mark=AssetPhase.ROLLED)
    def roll_forward(self, *, is_update_mv_price: bool = True, **kwargs) -> None:
        """
        Roll the bond forward one period, updating units, prices, and cash flows.
        """
        t, p = self.time, self.period

        if not self.is_alive_beg:
            self._units = 0
            self._mv_price_dirty = 0
            self._abv_price_dirty = 0
            self._cash_flow = 0
            return

        if self._credit_band is None:
            prob_of_default = 0.0
            recovery_rate = 0.0
        else:
            prob_of_default = 1 - (1 - self._credit_band.prob_of_default_ac) ** (1/12)
            recovery_rate = self._credit_band.recovery_rate

        units_st = self._units
        units_default = units_st * prob_of_default
        units_survive = units_st - units_default
        units_maturity = units_survive if (p == self._params.maturity_date) else 0
        self._units = units_survive - units_maturity

        abv_price_st = self._abv_price_dirty
        coupon_paid = self.cash_flow_gen.get_interest_payment_at_month(p)
        principal_paid = self.cash_flow_gen.get_principal_payment_at_month(p)

        interest = coupon_paid * units_survive
        principal = principal_paid * units_survive
        default_recovery = (abv_price_st - coupon_paid) * units_default * recovery_rate
        self._cash_flow = interest + principal + default_recovery

        freq = 1 if self._params.coupon_freq == 0 else self._params.coupon_freq  # 1 for zero coupon bond
        self._abv_price_dirty = abv_price_st * (1 + self._amort_rate / freq) ** (freq / 12) - coupon_paid - principal_paid
        # valid, price = self._validate_current_price("abv", self._abv_price_dirty)

        if is_update_mv_price:
            self._mv_price_dirty = self.pricer.calculate_market_price(p, self.ra_spots)

        # Update arrays
        self.tdv_mv_price[t] = self._mv_price_dirty
        self.tdv_abv_price[t] = self._abv_price_dirty
        self.tdv_units_bd[t] = self._units
        self.tdv_units_default[t] = units_default
        self.tdv_units_maturity[t] = units_maturity
        self.tdv_mv_bd[t] = self._mv_price_dirty * self._units
        self.tdv_abv_bd[t] = self._abv_price_dirty * self._units
        self.tdv_interest[t] = interest
        self.tdv_principal[t] = principal
        self.tdv_default_recovery[t] = default_recovery
        self.tdv_cash_flow[t] = self._cash_flow

    @property
    @transition(require_all=AssetPhase.ROLLED)
    def cash_flow(self) -> float:
        """float: Cash flow in period"""
        return self._cash_flow

    @property
    def arr_cash_flow(self) -> TDimVariable:
        """TDepVariable: Cash flow array"""
        return self.tdv_cash_flow

    def buy_propn(self, *args, **kwargs) -> None:
        """
        Not allowed for bonds. Raises ValueError.

        Raises:
            ValueError: Always.
        """
        raise ValueError("It's not allowed to buy bonds by scaling exsiting segments.")

    @transition(require_all=AssetPhase.ROLLED, require_not=AssetPhase.CLOSED)
    def sell_propn(self, propn: float) -> None:
        """
        Sell a proportion of existing bonds.

        Args:
            propn (float): Proportion to sell.

        Raises:
            ValueError: If propn is negative.
        """
        if propn < 0: raise ValueError(f"Can not sell negative proportion of exsiting bonds.")
        if propn > 1: raise ValueError(f"Can not sell >100% proportion of exsiting bonds.")
        self._units -= self._units * propn

    def scale_profile(self, scale: float, *, new_asset_id: str | None = None) -> Self:
        """
        Scale the bond profile by a factor.

        Args:
            scale (float): Scaling factor.
            new_asset_id (str): Asset id for new asset.

        Raises:
            ValueError: If scale is negative.
        """
        if not self._is_profile:
            raise ValueError("This bond object is not a profile.")
        if scale < 0:
            raise ValueError("Can not scale bond profile by a negative number.")

        if new_asset_id is None:
            new_asset_id = self.asset_id + ("" if self._n_clone == 0 else f"_{self._n_clone}")

        clone = BondFixed(
            units=self._units * scale,
            issue_date=self._params.issue_date,
            maturity_date=self._params.maturity_date,
            coupon_rate=self._params.coupon_rate,
            coupon_freq=self._params.coupon_freq,
            face_value=self._params.face_value,
            mv_price=self._mv_price_dirty,
            market_spread=self._market_spread,
            abv_price=self._abv_price_dirty,
            amort_rate=self._amort_rate,
            rf_curve=self._rf_curve,
            attr_aliases=self._attr_aliases,
            model_engine=self._model_ref(),
            asset_id=new_asset_id,
            is_profile=False,
            currency=self._currency,
            provided_cash_flow_dict=self.cash_flow_gen.to_dict() if isinstance(self.cash_flow_gen, BondFixedCashFlowProvider) else None,
            credit_band=self._credit_band,
            purchase_date=self.period,
        )
        self._copy_dynamic_attrs_to(clone)
        self._n_clone += 1
        return clone

    @transition(require_all=AssetPhase.ROLLED, mark=AssetPhase.CLOSED)
    def close_dealing(self) -> None:
        """
        Update the bond after dealing, storing units and values.
        """
        t = self.time
        self.tdv_units_ad[t] = self._units
        self.tdv_mv_ad[t] = self.market_value
        self.tdv_abv_ad[t] = self.amortized_book_value
