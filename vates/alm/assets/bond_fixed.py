import numpy as np
import numpy.typing as npt
import pandas as pd

from ..._core import TDepVariable
from ...utils import calculate_risk_adj_spot, check_calc_time
from ..enums import AssetClassification
from ..econs import Currency, YieldCurve, CreditBand
from .asset_base import Asset
from ._bond_fixed_component import BondFixedParameters, BondFixedCashFlowGenerator, BondFixedPricer, BondFixedRiskCalculator


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
                 '_credit_band', '_cash_flow', 'cash_flow_gen', 'pricer', 'risk_calc',
                 'tdv_units_default', 'tdv_units_maturity', 'tdv_units_bd', 'tdv_units_ad', 'tdv_cash_flow',
                 'tdv_interest', 'tdv_principal', 'tdv_default_recovery', 'tdv_mv_price', 'tdv_abv_price',
                 'tdv_mv_bd', 'tdv_abv_bd', 'tdv_mv_ad', 'tdv_abv_ad',)

    
    def __init__(self, model, asset_id: str, is_profile: bool, units: float, currency: Currency | None, asset_class: str,
                 fund_id: str, allocation_group: str, classification: AssetClassification, issue_date: pd.Period,
                 maturity_date: pd.Period, redemp_sched: np.ndarray | None, coupon_rate: float, coupon_freq: int,
                 face_value: float, mv_price: float, market_spread: float, abv_price: float, amort_rate: float,
                 rf_curve: YieldCurve, credit_band: CreditBand | None, purchase_date: pd.Period | None=None):
        """
        Initialize a Bond asset.

        Args:
            asset_id (str): Asset identifier.
            is_profile (bool): Ture if profile asset, False if existing asset.
            units (float): Number of bond units.
            currency (Currency): Asset currency.
            asset_class (str): Asset class.
            fund_id (str): Fund identifier.
            allocation_group (str): Allocation group.
            classification (AssetClassification): Asset classification.
            issue_date (pd.Period): Issue date of the bond.
            maturity_date (pd.Period): Maturity date of the bond.
            coupon_rate (float): Coupon rate.
            coupon_freq (int): Coupon frequency per year, [0, 1, 2, 4, 12].
            face_value (float): Face value of the bond.
            redemp_sched (np.ndarray | None): Redemption schedule.
            mv_price (float): Market value price (dirty value).
            market_spread (float): Market spread.
            abv_price (float): Amortized book value price (dirty value).
            amort_rate (float): Amortization rate.
            rf_curve (YieldCurve): Risk-free yield curve.
            credit_band (CreditBand | None): Credit band (to provide assumptions, e.g. default and spread).
            purchase_date (pd.Period | None): Purchase date, default to initilization date.
        """
        super().__init__(model, asset_id, is_profile, units, purchase_date, currency, classification, asset_class,
                         fund_id, allocation_group)
        t, p = self.time, self.period
        self._params = BondFixedParameters(
            issue_date=issue_date,
            maturity_date=maturity_date,
            coupon_rate=coupon_rate,
            coupon_freq=coupon_freq,
            face_value=face_value,
            redemp_sched=redemp_sched
        )
        self._mv_price_dirty: float = mv_price
        self._market_spread: float = market_spread
        self._abv_price_dirty: float = abv_price
        self._amort_rate: float = amort_rate
        self._rf_curve: YieldCurve = rf_curve
        self._credit_band: CreditBand | None = credit_band
        self._cash_flow: float = 0.0

        # Compose with specialized components
        self.cash_flow_gen = BondFixedCashFlowGenerator(self._params)
        self.pricer = BondFixedPricer(self._params, self.cash_flow_gen)
        self.risk_calc = BondFixedRiskCalculator(self._params, self.cash_flow_gen, self.pricer)

        # Validate initial arguments
        self._validate_mv_price(self._mv_price_dirty, p)
        self._validate_abv_price(self._abv_price_dirty, p)

        # Initialize TDepVariable
        self.tdv_units_default: TDepVariable = TDepVariable(model, "units_default", asset_id, 'bond')
        self.tdv_units_maturity: TDepVariable = TDepVariable(model, "units_maturity", asset_id, 'bond')
        self.tdv_units_bd: TDepVariable = TDepVariable(model, "units_bd", asset_id, 'bond')
        self.tdv_units_ad: TDepVariable = TDepVariable(model, "units_ad", asset_id, 'bond')
        self.tdv_cash_flow: TDepVariable = TDepVariable(model, "cash_flow", asset_id, 'bond')
        self.tdv_interest: TDepVariable = TDepVariable(model, "interest", asset_id, 'bond')
        self.tdv_principal: TDepVariable = TDepVariable(model, "principal", asset_id, 'bond')
        self.tdv_default_recovery: TDepVariable = TDepVariable(model, "default_recovery", asset_id, 'bond')
        self.tdv_mv_price: TDepVariable = TDepVariable(model, "mv_price", asset_id, 'bond')
        self.tdv_abv_price: TDepVariable = TDepVariable(model, "abv_price", asset_id, 'bond')
        self.tdv_mv_bd: TDepVariable = TDepVariable(model, "mv_bd", asset_id, 'bond')
        self.tdv_abv_bd: TDepVariable = TDepVariable(model, "abv_bd", asset_id, 'bond')
        self.tdv_mv_ad: TDepVariable = TDepVariable(model, "mv_ad", asset_id, 'bond')
        self.tdv_abv_ad: TDepVariable = TDepVariable(model, "abv_ad", asset_id, 'bond')

        self.tdv_mv_price[t] = self._mv_price_dirty
        self.tdv_abv_price[t] = self._abv_price_dirty

        if is_profile:
            tolerance = max(max(abs(self._mv_price_dirty), abs(self._abv_price_dirty) * 1e-6), 1e-8)
            if abs(self._abv_price_dirty - self._mv_price_dirty) > tolerance:
                raise ValueError(f'Bond {self.asset_id} at {p}: abv_price != mv_price at purchase, '
                                 f'abv={self._abv_price_dirty:.4f}, mv={self._mv_price_dirty:.4f}')
        else: # not is_profile
            self.tdv_units_ad[t] = self._units
            self.tdv_mv_ad[t] = self.mv
            self.tdv_abv_ad[t] = self.abv

    @property
    def mv_price(self) -> float:
        return self._mv_price_dirty

    @property
    def abv_price(self) -> float:
        return self._abv_price_dirty

    @property
    def mv(self) -> float:
        """float: Market value of the bond asset."""
        return self.mv_price * self._units

    @property
    def abv(self) -> float:
        """float: Amortized book value of the bond asset."""
        return self.abv_price * self._units

    @property
    def fav(self) -> float:
        if self.classification == AssetClassification.FVTPL:
            return self.mv
        elif self.classification == AssetClassification.FVOCI:
            return self.abv
        elif self.classification == AssetClassification.AC:
            return self.abv
        else:
            raise ValueError(f"Bond {self.asset_id}: invalid asset classification {self.classification}.")

    @property
    def bsv(self) -> float:
        if self.classification == AssetClassification.FVTPL:
            return self.mv
        elif self.classification == AssetClassification.FVOCI:
            return self.mv
        elif self.classification == AssetClassification.AC:
            return self.abv
        else:
            raise ValueError(f"Bond {self.asset_id}: invalid asset classification {self.classification}.")

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
        rf_spots = self._rf_curve.curve_data["spot"]

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
    def os_term_m(self) -> int:
        return max((self._params.maturity_date - self.period).n, 0)

    @property
    def is_alive(self) -> bool:
        return self.period < self._params.maturity_date

    @property
    def is_alive_beg(self) -> bool:
        return self.period <= self._params.maturity_date

    @check_calc_time({"roll_forward": 0})
    def calculate_risk_metrics(self, eff_dur_delta: float=0.001):
        """
        Calculate all risk metrics for the bond.

        Args:
            eff_dur_delta (float): Delta yiled curve for effective duration calculation.
        """
        self.risk_calc.calculate_all_risk_metrics(self.period, self._mv_price_dirty, self.ra_spots, eff_dur_delta)

    def get_risk_metrics(self, var_name: str=None) -> dict[str, float] | float | None:
        """
        Get all calculated risk metrics for the bond.

        Returns:
            dict[str, float]: dictionary of risk metrics.
        """
        if var_name:
            return None if var_name not in self.risk_calc.risk_metrics else self.risk_calc.risk_metrics[var_name]
        else:
            return self.risk_calc.risk_metrics

    @check_calc_time({"roll_forward": -1, "complete_dealing": -1}, "roll_forward")
    def roll_forward(self, **kwargs) -> None:
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
        # self._validate_abv_price(self._abv_price_dirty, p)

        if not kwargs.get('skip_dcf', False):
            if self._rf_curve.last_update != t:
                raise ValueError(f"{self._rf_curve.curve_id} is not updated on {t} ({p}).")
            if self._credit_band is not None and self._credit_band.last_update != t:
                raise ValueError(f"{self._credit_band.band_id} is not updated on {t} ({p}).")
            self._mv_price_dirty = self.pricer.calculate_market_price(p, self.ra_spots)

        # Update arrays
        self.tdv_mv_price[t] = self._mv_price_dirty
        self.tdv_abv_price[t] = self._abv_price_dirty
        self.tdv_units_bd[t] = self._units
        self.tdv_units_default[t] = units_default
        self.tdv_units_maturity[t] = units_maturity
        self.tdv_mv_bd[t] = self.mv
        self.tdv_abv_bd[t] = self.abv
        self.tdv_interest[t] = interest
        self.tdv_principal[t] = principal
        self.tdv_default_recovery[t] = default_recovery
        self.tdv_cash_flow[t] = self._cash_flow

    @property
    def cash_flow(self) -> float:
        """float: Cash flow in period"""
        return self._cash_flow

    @property
    def arr_cash_flow(self) -> TDepVariable:
        """TDepVariable: Cash flow array"""
        return self.tdv_cash_flow

    def buy_propn(self, *args, **kwargs) -> None:
        """
        Not allowed for bonds. Raises ValueError.

        Raises:
            ValueError: Always.
        """
        raise ValueError("It's not allowed to buy bonds by scaling exsiting segments.")

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

    def buy_profile_scale(self, scale: float, list_to_append: list | None=None) -> None:
        """
        Scale the bond profile by a factor.

        Args:
            scale (float): Scaling factor.
            list_to_append (list): Append to list.

        Raises:
            ValueError: If scale is negative.
        """
        if not self._is_profile: raise ValueError("This bond object is not a profile.")
        if scale < 0: raise ValueError("Can not scale bond profile by a negative number.")
        self._units = self._units * scale
        self._is_profile = False
        if list_to_append is not None: list_to_append.append(self)

    @check_calc_time({"complete_dealing": -1, "roll_forward": 0}, "complete_dealing")
    def complete_dealing(self) -> None:
        """
        Update the bond after dealing, storing units and values.
        """
        t = self.time
        self.tdv_units_ad[t] = self._units
        self.tdv_mv_ad[t] = self.mv
        self.tdv_abv_ad[t] = self.abv

    def _validate_mv_price(self, price: float, valn_date: pd.Period) -> None:
        """
        Validate the market value price.
        """
        if price <= 0 and abs(price) > 1e-8:
            raise ValueError(f"Bond {self.asset_id} mv price={price: .4f}, expect >0.")
        calc_price = self.pricer.calculate_market_price(valn_date, self.ra_spots)
        tolerance = max(price * 1e-6, 1e-8)
        if abs(price - calc_price) > tolerance:
            raise ValueError(
                f"{valn_date} bond {self.asset_id}: mv price is calculated as {calc_price: .4f} "
                f"based on risk free rate, credit spread and market spread, "
                f"but input mv price is {price: .4f}."
            )

    def _validate_abv_price(self, price: float, valn_date: pd.Period) -> None:
        """
        Validate the amortized book value price.
        """
        if price <= 0 and abs(price) > 1e-8:
            raise ValueError(f"Bond {self.asset_id} abv price={price: .4f}, expect >0.")
        calc_price = self.pricer.calculate_amortized_price(valn_date, self._amort_rate)
        tolerance = max(price * 1e-6, 1e-8)
        if abs(price - calc_price) > tolerance:
            raise ValueError(
                f"{valn_date} bond {self.asset_id}: abv price is calculated as {calc_price: .4f} "
                f"based on amortized rate {self._amort_rate: .4%}, "
                f"but input abv price is {price: .4f}."
            )
