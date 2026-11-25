import math
import numpy as np
import numpy.typing as npt
import pandas as pd
from collections.abc import Mapping
from typing import Literal, Self

from vates._core import ProjModelEngine, TDimVariable, make_proj_variable
from vates.utils import transition, maybe_raise_on_compare
from vates.alm.econs import Currency, YieldCurve, Spread, DefaultRisk, DefaultRiskParams
from vates.alm.assets.asset_base import BaseAsset, AssetPhase
from vates.alm.assets._bond_fixed_component import (
    BondFixedParameters,
    BondFixedCashFlow,
    BondFixedCashFlowGenerator,
    BondFixedPricer,
    BondFixedRiskCalculator,
)
from vates.alm.assets._utils import make_risk_adjusted_rates


class BondFixed(BaseAsset):
    """
    Represents a bond asset in the portfolio.

    Attributes:
        _params (BondFixedParameters): Bond parameters.
        _mv_price_dirty (float): Market value price (dirty).
        _market_spread (float): Market spread.
        _abv_price_dirty (float): Amortized book value price (dirty).
        _amort_rate (float): Amortization rate.
        _rf_curve (YieldCurve): Risk-free yield curve.
        _spread (Spread | None): Spread.
        _default_risk (DefaultRisk | DefaultRiskParams | None): Default risk.
        cash_flow_gen (BondFixedCashFlowGenerator): Cash flow generator.
        pricer (BondFixedPricer): Bond pricer.
        risk_calc (BondFixedRiskCalculator): Risk calculator.
    """
    __slots__ = ('_qty', '_params', '_mv_price_dirty', '_market_spread', '_abv_price_dirty', '_amort_rate', '_rf_curve',
                 '_spread', '_default_risk', '_cash_flow', '_coupon', '_principal', '_recovery', '_credit_loss', '_interest'
                 '_purchase_amount', '_sale_proceeds', 'cash_flow_gen', 'pricer', 'risk_calc', '_n_clone',
                 'tdv_qty_default', 'tdv_qty_maturity', 'tdv_qty_bd', 'tdv_qty_ad', 'tdv_cash_flow',
                 'tdv_coupon', 'tdv_principal', 'tdv_recovery', 'tdv_credit_loss', 'tdv_interest', 'tdv_mv_price', 'tdv_abv_price',
                 'tdv_mv_bd', 'tdv_abv_bd', 'tdv_mv_ad', 'tdv_abv_ad', 'tdv_purchase_amount',
                 'tdv_sale_proceeds',)

    
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
        asset_id: str = "",
        is_profile: bool = False,
        qty: float = 1.0,
        currency: Currency | None = None,
        fixed_cash_flow: BondFixedCashFlow | None = None,
        spread: Spread | None = None,
        default_risk: DefaultRisk | DefaultRiskParams | None = None,
        entry_date: pd.Period | None = None,
        _bypass_init_validation: bool = False,
        **kwargs
    ):
        """
        Initialize a Bond asset.

        Args:
            asset_id (str): Asset identifier.
            is_profile (bool): Ture if profile asset, False if existing asset.
            qty (float): Number of units.
            currency (Currency): Asset currency.
            attr_aliases (Mapping[str, str]): Mapping of alias to named attribute.
            issue_date (pd.Period): Issue date of the bond.
            maturity_date (pd.Period): Maturity date of the bond.
            coupon_rate (float): Coupon rate.
            coupon_freq (int): Coupon frequency per year, [0, 1, 2, 4, 12].
            face_value (float): Face value of the bond.
            fixed_cash_flow (BondFixedCashFlow | None): Fixed cash flow of the bond.
            mv_price (float): Market value price (dirty value).
            market_spread (float): Market spread.
            abv_price (float): Amortized book value price (dirty value).
            amort_rate (float): Amortization rate.
            rf_curve (YieldCurve): Risk-free yield curve.
            spread (Spread | None): Spread.
            default_risk (DefaultRisk | DefaultRiskParams | None): Default risk.
            entry_date (pd.Period | None): Entry date; defaults to the period at initialization.
            _bypass_init_validation (bool): True to bypass initial validation. Defaults to False.
        """
        super().__init__(model_engine=model_engine, asset_id=asset_id, is_profile=is_profile,
                         entry_date=entry_date, currency=currency, attr_aliases=attr_aliases, **kwargs)
        self._qty: float = qty
        self._params: BondFixedParameters = BondFixedParameters(
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
        self._spread: Spread | None = spread
        self._default_risk: DefaultRisk | DefaultRiskParams | None = default_risk
        self._cash_flow: float = 0.0
        self._coupon: float = 0.0
        self._principal: float = 0.0
        self._recovery: float = 0.0
        self._credit_loss: float = 0.0
        self._interest: float = 0.0
        self._purchase_amount: float = 0.0
        self._sale_proceeds: float = 0.0
        self._n_clone: int = 0

        # Compose with specialized components
        self.cash_flow_gen = BondFixedCashFlowGenerator(bond_params=self._params, fixed_cash_flows=fixed_cash_flow)
        self.pricer = BondFixedPricer(self.cash_flow_gen)
        self.risk_calc = BondFixedRiskCalculator(self.pricer)
        self.pricer.lock_amortized_price(amort_rate=self.amort_rate, since_date=self.period, init_price=self._abv_price_dirty)

        # validate profile price
        if is_profile and not math.isclose(self._abv_price_dirty, self._mv_price_dirty, rel_tol=1e-6, abs_tol=1e-8):
            ValueError(f"Profile bond {self.asset_id}: abv_price={self._abv_price_dirty:.4f} != "
                       f"mv_price={self._mv_price_dirty:.4f}")

        if not _bypass_init_validation:
            calc_price = self.pricer.calculate_market_price(self.period, self.ra_spots)
            if not math.isclose(self._mv_price_dirty, calc_price, rel_tol=1e-6, abs_tol=1e-8):
                raise ValueError(f"Bond {asset_id} mv_price {self._mv_price_dirty:.4f} != calculated {calc_price:.4f}.")
            calc_price = self.pricer.get_amortized_price(self.period)
            if not math.isclose(self._abv_price_dirty, calc_price, rel_tol=1e-6, abs_tol=1e-8):
                raise ValueError(f"Bond {asset_id} abv_price {self._abv_price_dirty:.4f} != calculated {calc_price:.4f}.")

        # Initialize TDepVariable
        create_tdv = lambda name: make_proj_variable(name, model_engine=model_engine, owner=asset_id, group='bond')
        self.tdv_qty_default: TDimVariable = create_tdv("qty_default")
        self.tdv_qty_maturity: TDimVariable = create_tdv("qty_maturity")
        self.tdv_qty_bd: TDimVariable = create_tdv("qty_bd")
        self.tdv_qty_ad: TDimVariable = create_tdv("qty_ad")
        self.tdv_cash_flow: TDimVariable = create_tdv("cash_flow")
        self.tdv_coupon: TDimVariable = create_tdv("coupon")
        self.tdv_principal: TDimVariable = create_tdv("principal")
        self.tdv_recovery: TDimVariable = create_tdv("recovery")
        self.tdv_credit_loss: TDimVariable = create_tdv("credit_loss")
        self.tdv_interest: TDimVariable = create_tdv("interest")
        self.tdv_mv_price: TDimVariable = create_tdv("mv_price")
        self.tdv_abv_price: TDimVariable = create_tdv("abv_price")
        self.tdv_mv_bd: TDimVariable = create_tdv("mv_bd")
        self.tdv_abv_bd: TDimVariable = create_tdv("abv_bd")
        self.tdv_mv_ad: TDimVariable = create_tdv("mv_ad")
        self.tdv_abv_ad: TDimVariable = create_tdv("abv_ad")
        self.tdv_purchase_amount: TDimVariable = create_tdv("purchase_amount")
        self.tdv_sale_proceeds: TDimVariable = create_tdv("sale_proceeds")

        t = self.time
        self.tdv_mv_price[t] = self._mv_price_dirty
        self.tdv_abv_price[t] = self._abv_price_dirty
        self.tdv_qty_ad[t] = self._qty
        self.tdv_mv_ad[t] = self.mv_price * self._qty
        self.tdv_abv_ad[t] = self.abv_price * self._qty

    @property
    def qty(self) -> float:
        return self._qty

    @property
    def mv_price(self) -> float:
        return self._mv_price_dirty

    @property
    def abv_price(self) -> float:
        return self._abv_price_dirty

    @property
    def market_value(self) -> float:
        """float: Market value of the bond asset."""
        return self._mv_price_dirty * self._qty

    @property
    def amortized_book_value(self) -> float:
        """float: Amortized book value of the bond asset."""
        return self._abv_price_dirty * self._qty

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
        if self._spread:
            return make_risk_adjusted_rates(rf_rates=rf_spots, proportional_spread=self._spread.proportional,
                                            additive_spread=self._spread.additive + self._market_spread)
        else:
            return rf_spots + self._market_spread

    @property
    def os_term_m(self) -> int | None:
        if self._params.issue_date <= self.period <= self._params.maturity_date:
            return (self._params.maturity_date - self.period).n  # gives: n, n-1, ..., 1, 0 during lifespan
        return None

    def is_alive(self, at: Literal["bop", "eop"], /) -> bool:
        maybe_raise_on_compare(at, must_in=("bop", "eop"))
        return (self._params.issue_date <= self.period < self._params.maturity_date) if at == "eop" else (
                self._params.issue_date < self.period <= self._params.maturity_date)

    def calculate_risk_metrics(self, eff_dur_delta: float = 0.001):
        """
        Calculate all risk metrics for the bond.

        Args:
            eff_dur_delta (float): Delta yield curve for effective duration calculation.
        """
        return self.risk_calc.calculate_all_risk_metrics(
            valn_date=self.period, market_price=self._mv_price_dirty, spots=self.ra_spots, eff_dur_delta=eff_dur_delta
        )

    def _on_time_change(self) -> None:
        self._cash_flow = 0.0
        self._coupon: float = 0.0
        self._principal: float = 0.0
        self._recovery: float = 0.0
        self._credit_loss: float = 0.0
        self._interest: float = 0.0
        self._purchase_amount = 0.0
        self._sale_proceeds = 0.0

    @transition(require_all=AssetPhase.CLOSED, require_offset=-1, require_not=AssetPhase.ROLLED, mark=AssetPhase.ROLLED)
    def roll_forward(self, *, is_update_mv_price: bool = True, **kwargs) -> None:
        """
        Roll the bond forward one period, updating quantity, prices, and cash flows.
        """
        t, p = self.time, self.period

        if not self.is_alive("bop"):
            self._qty = 0
            self._mv_price_dirty = 0
            self._abv_price_dirty = 0
            return

        qty_st = self._qty
        coupon = self.cash_flow_gen.get_coupon(p)
        principal = self.cash_flow_gen.get_principal(p)
        interest = self.pricer.get_amortized_interest(p)
        abv_price = self.pricer.get_amortized_price(p)

        if self._default_risk is None:
            mpd = 0
            lgd = 0
            qty_default = 0
            ead = 0
        else:
            mpd = self._default_risk.marginal_prob_of_default
            lgd = self._default_risk.loss_given_default
            if self._default_risk.pd_affects_decrement:
                qty_default = qty_st * mpd
                ead = abv_price + coupon + principal  # EAD (Exposure at Default), also equals abv_price_st + interest
            else:
                qty_default = 0
                ead = coupon + principal

        # 1. MPD affects decrement
        #   +-------------+----------------------------+------------+
        #   | probability |         cash flow          | book value |
        #   +-------------+----------------------------+------------+
        #   |     1 - MPD |         coupon + principal |         bv |
        #   |         MPD | recovery = (1 - LGD) * EAD |         0  |
        #   +-------------+----------------------------+------------+
        #   expected cash flow = (1 - MPD) * (coupon + principal) + MPD * recovery
        #  expected book value = (1 - MPD) * bv
        # expected credit loss = contractual cash flow + contractual book value - expected cash flow - expected book value
        #                      = MPD * (coupon + principal + bv) - MPD * recovery
        #                      = MPD * EAD - MPD * (1 - LGD) * EAD  <- given EAD = coupon + principal + bv
        #                      = MPD * LGD * EAD  <- the classic formula
        # 2. MPD doesn't affect decrement
        #   +-------------+----------------------------+
        #   | probability |         cash flow          |
        #   +-------------+----------------------------+
        #   |     1 - MPD |         coupon + principal |
        #   |         MPD | recovery = (1 - LGD) * EAD |
        #   +-------------+----------------------------+
        #   expected cash flow = (1 - MPD) * (coupon + principal) + MPD * recovery
        # expected credit loss = contractual cash flow - expected cash flow  <- no impact to book value
        #                      = MPD * (coupon + principal) - MPD * recovery
        #                      = MPD * EAD - MPD * (1 - LGD) * EAD  <- given EAD = coupon + principal
        #                      = MPD * LGD * EAD  <- the classic formula

        self._coupon = (1 - mpd) * coupon * qty_st
        self._principal = (1 - mpd) * principal * qty_st
        self._recovery = mpd * (1 - lgd) * ead * qty_st
        self._credit_loss = mpd * lgd * ead * qty_st
        self._interest = interest * qty_st

        # decrement
        qty = qty_st - qty_default  # survival
        qty, qty_maturity = (0, qty) if (p == self._params.maturity_date) else (qty, 0)

        # store calculated results
        self._qty = qty
        self._cash_flow = self._coupon + self._principal + self._recovery
        self._abv_price_dirty = abv_price
        if is_update_mv_price:
            self._mv_price_dirty = self.pricer.calculate_market_price(p, self.ra_spots)

        # Update arrays
        self.tdv_mv_price[t] = self._mv_price_dirty
        self.tdv_abv_price[t] = self._abv_price_dirty
        self.tdv_qty_bd[t] = self._qty
        self.tdv_qty_default[t] = qty_default
        self.tdv_qty_maturity[t] = qty_maturity
        self.tdv_mv_bd[t] = self._mv_price_dirty * self._qty
        self.tdv_abv_bd[t] = self._abv_price_dirty * self._qty
        self.tdv_coupon[t] = self._coupon
        self.tdv_principal[t] = self._principal
        self.tdv_recovery[t] = self._recovery
        self.tdv_credit_loss[t] = self._credit_loss
        self.tdv_cash_flow[t] = self._cash_flow
        self.tdv_interest[t] = self._interest

    @property
    @transition(require_all=AssetPhase.ROLLED)
    def cash_flow(self) -> float:
        """float: Cash flow in period"""
        return self._cash_flow

    @property
    @transition(require_all=AssetPhase.ROLLED)
    def coupon(self) -> float:
        """float: Coupon cash flow in period"""
        return self._coupon

    @property
    @transition(require_all=AssetPhase.ROLLED)
    def principal(self) -> float:
        """float: Principal cash flow in period"""
        return self._principal

    @property
    @transition(require_all=AssetPhase.ROLLED)
    def recovery(self) -> float:
        """float: Recovery cash flow (on default) in period"""
        return self._recovery

    @property
    @transition(require_all=AssetPhase.ROLLED)
    def credit_loss(self) -> float:
        """float: Credit loss (amortized basis) in period, abv_bd(t) = abv_ad(t-1) + interest - cash_flow - credit_loss"""
        return self._credit_loss

    @property
    @transition(require_all=AssetPhase.ROLLED)
    def interest(self) -> float:
        """float: Interest (amortized basis) in period"""
        return self._interest

    def buy_propn(self, *args, **kwargs) -> None:
        """
        Not allowed for bonds. Raises NotImplementedError.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Cannot buy bond by scaling existing instance.")

    @transition(require_all=AssetPhase.ROLLED, require_not=AssetPhase.CLOSED)
    def sell_propn(self, propn: float) -> None:
        """
        Sell a proportion of existing bonds.

        Args:
            propn (float): Proportion to sell.

        Raises:
            ValueError: If propn is negative.
        """
        if propn < 0: raise ValueError(f"{propn=:.6f}: can not sell negative proportion of existing bonds.")
        if propn > 1: raise ValueError(f"{propn=:.6f}: can not sell >100% proportion of existing bonds.")
        qty = self._qty * propn
        self._qty -= qty
        self._sale_proceeds += self._mv_price_dirty * qty

    @transition(require_all=AssetPhase.PROFILE)
    def buy_profile(self, scale: float, *, new_asset_id: str | None = None) -> Self:
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

        qty = self._qty * scale

        clone = self.__class__(
            qty=qty,
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
            fixed_cash_flow=self.cash_flow_gen.fixed_cash_flow,
            spread=self._spread,
            default_risk=self._default_risk,
            entry_date=self.period,
        )
        self._copy_dynamic_attrs_to(clone)
        clone._purchase_amount = clone._mv_price_dirty * qty
        self._n_clone += 1
        return clone

    @property
    def purchase_amount(self) -> float:
        """float: Cash paid to purchase the bond position during the period."""
        return self._purchase_amount

    @property
    def sale_proceeds(self) -> float:
        """float: Cash received from selling the bond position during the period."""
        return self._sale_proceeds

    @transition(require_all=AssetPhase.ROLLED, mark=AssetPhase.CLOSED)
    def close_dealing(self) -> None:
        """
        Update the bond after dealing, storing quantity and values.
        """
        t = self.time
        self.tdv_qty_ad[t] = self._qty
        self.tdv_mv_ad[t] = self.market_value
        self.tdv_abv_ad[t] = self.amortized_book_value
        self.tdv_purchase_amount[t] = self._purchase_amount
        self.tdv_sale_proceeds[t] = self._sale_proceeds

    def retrieve_cash_flow(self, t: int) -> float:
        maybe_raise_on_compare(t, must_ge=0, must_le=self.time)
        return self.tdv_cash_flow[t]
