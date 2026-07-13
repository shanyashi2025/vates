import numpy as np
import pandas as pd
import numpy.typing as npt
import warnings
from dataclasses import dataclass

from vates.utils import convert_spot_to_disc, solve_ytm


@dataclass(frozen=True, slots=True)
class BondFixedParameters:
    """
    Data class to hold bond parameters.

    Attributes:
        issue_date (pd.Period): Issue date of the bond.
        maturity_date (pd.Period): Maturity date of the bond.
        coupon_rate (float): Coupon rate.
        coupon_freq (int): Coupon frequency per year, [0, 1, 2, 4, 12].
        face_value (float): Face value of the bond.
    """
    issue_date: pd.Period
    maturity_date: pd.Period
    coupon_rate: float
    coupon_freq: int
    face_value: float

    def __post_init__(self):
        if self.issue_date >= self.maturity_date:
            raise ValueError(f'issue_date {self.issue_date} > maturity_date {self.maturity_date}.')
        if self.face_value <= 0:
            raise ValueError(f'face_value={self.face_value} should be positive.')
        if self.coupon_freq not in (0, 1, 2, 4, 12):
            raise ValueError(f'Invalid coupon_freq={self.coupon_freq}, expected [0, 1, 2, 4, 12].')
        if self.coupon_interval and (self.maturity_date - self.issue_date).n % self.coupon_interval != 0:
            warnings.warn(f"Months between maturity date ({self.maturity_date}) and issue date {self.issue_date} is not "
                          f"divisible by coupon interval ({self.coupon_interval}). Maturity date will be used in "
                          f"determining months to pay coupon.")
        if not -0.05 < self.coupon_rate < 0.2:
            warnings.warn(f'coupon_rate={self.coupon_rate} not in normal range, recommend to check.')

    @property
    def coupon_interval(self) -> int | None:
        if self.coupon_freq != 0:
            return 12 // self.coupon_freq
        else:
            return None

class BondFixedCashFlowGenerator:
    """
    Handles cash flow generation for bonds.

    Attributes:
        params (BondFixedParameters): Bond parameters.
        _cash_flow_from_issue (np.ndarray): Bond cash flows from issue, index `0, 1, ..., n-1` represents month `1, 2, ..., n`
    """

    __slots__ = ('params', '_cash_flow_from_issue',)

    def __init__(self, bond_params: BondFixedParameters):
        """
        Initialize the cash flow generator.

        Args:
            bond_params (BondFixedParameters): Parameters of the bond.
        """
        self.params = bond_params
        self._cash_flow_from_issue: npt.NDArray[np.float64] | None = None

    def get_interest_payment_at_month(self, ref_date: pd.Period) -> float:
        """
        Get the interest payment at a specific month.

        Args:
            ref_date (pd.Period): The date for which to calculate interest.

        Returns:
            float: Interest payment for the month.
        """
        if self.params.coupon_rate == 0.0:
            return 0.0

        coupon_interval = self.params.coupon_interval
        if coupon_interval is not None:  # i.e. coupon_freq in (1, 2, 4, 12)
            ost_in_month = (self.params.maturity_date - ref_date).n
            if ost_in_month % coupon_interval != 0:
                return 0.0
            else:
                return self.params.face_value * self.params.coupon_rate / self.params.coupon_freq
        else:  # coupon_freq == 0:
            return self.params.face_value * self.params.coupon_rate if ref_date == self.params.maturity_date else 0.0

    def get_principal_payment_at_month(self, ref_date: pd.Period) -> float:
        """
        Get the principal payment at a specific month.

        Args:
            ref_date (pd.Period): The date for which to calculate principal.

        Returns:
            float: Principal payment for the month (at the maturity date).
        """
        return self.params.face_value if ref_date == self.params.maturity_date else 0.0

    @property
    def cash_flow_from_issue(self) -> npt.NDArray[np.float64]:
        """
        Generate the cash flow array from bond start date for each month.
        """
        if self._cash_flow_from_issue is not None:
            return self._cash_flow_from_issue

        n_months = (self.params.maturity_date - self.params.issue_date).n
        arr = np.zeros(n_months)
        coupon_interval = self.params.coupon_interval

        if self.params.coupon_rate == 0.0:  # zero coupon bond
            arr[n_months - 1] = self.params.face_value
        elif coupon_interval is not None:  # i.e. coupon_freq in (1, 2, 4, 12)
            coupon_amt = self.params.face_value * self.params.coupon_rate / self.params.coupon_freq
            arr[::-coupon_interval] = coupon_amt  # traverse the array backward
            arr[n_months - 1] += self.params.face_value
        else:  # one-off payment at maturity
            arr[n_months - 1] = self.params.face_value * (1 + self.params.coupon_rate)

        self._cash_flow_from_issue = arr
        return self._cash_flow_from_issue

    def get_future_cash_flows(self, valn_date: pd.Period) -> npt.NDArray[np.float64] | None:
        """
        Get cash flows from a specific valuation date forward.

        Args:
            valn_date (pd.Period): The valuation date (t).

        Returns:
            npt.NDArray[np.float64] | None: Array of future cash flows (t+1, t+2, ...), None if already matured.

        Raises:
            RuntimeError: If the bond has already matured.
        """
        if valn_date < self.params.maturity_date:
            age_in_month = (valn_date - self.params.issue_date).n
            return self.cash_flow_from_issue[age_in_month:]
        else:
            return None


class BondFixedPricer:
    """
    Handles bond pricing calculations.

    Attributes:
        cash_flow_gen (BondFixedCashFlowGenerator): Cash flow generator.
    """

    __slots__ = ('cash_flow_gen',)

    def __init__(self, cash_flow_generator: BondFixedCashFlowGenerator):
        """
        Initialize the bond pricer.

        Args:
            cash_flow_generator (BondFixedCashFlowGenerator): Cash flow generator.
        """
        self.cash_flow_gen = cash_flow_generator

    def calculate_market_price(self, valn_date: pd.Period, spots: npt.NDArray[np.float64]) -> float:
        """
        Calculate market price using risk-adjusted spot rates.

        Args:
            valn_date (pd.Period): Valuation date.
            spots (npt.NDArray[np.float64]): Spot rates.

        Returns:
            float: Market price.
        """
        cash_flows = self.cash_flow_gen.get_future_cash_flows(valn_date)
        if cash_flows is not None:  # return None if matured
            disc_factors = convert_spot_to_disc(spots, "M")
            return np.dot(cash_flows, disc_factors[1:len(cash_flows) + 1])
        else:
            return 0.0

    def calculate_amortized_price(self, valn_date: pd.Period, amort_rate: float) -> float:
        """
        Calculate amortized price using amortized rate.

        Args:
            valn_date (pd.Period): Valuation date.
            amort_rate (float): Amortization rate.

        Returns:
            float: Amortized price.
        """
        cash_flows = self.cash_flow_gen.get_future_cash_flows(valn_date)
        if cash_flows is not None:  # return None if matured
            periods = np.arange(1, len(cash_flows) + 1)
            coupon_freq = self.cash_flow_gen.params.coupon_freq
            freq = 1 if coupon_freq == 0 else coupon_freq # 1 for zero coupon bond
            disc_fac = (1 / (1 + amort_rate / freq)) ** (1 / 12 * freq)
            disc_factors = disc_fac ** periods
            return np.dot(cash_flows, disc_factors)
        else:
            return 0.0


class BondFixedRiskCalculator:
    """
    Handles risk metric calculations (duration, convexity, etc.).

    Attributes:
        _pricer (BondFixedPricer): Bond pricer.
    """
    __slots__ = ('_pricer', )

    def __init__(self, pricer: BondFixedPricer):
        """
        Initialize the risk calculator.

        Args:
            pricer (BondFixedPricer): Bond pricer.
        """
        self._pricer = pricer

    def calculate_all_risk_metrics(self, *, valn_date: pd.Period, market_price: float,
                                   spots: npt.NDArray[np.float64], eff_dur_delta: float = 0.001):
        """
        Calculate all risk metrics for the bond.

        Args:
            valn_date (pd.Period): Valuation date.
            market_price (float): Market price of the bond.
            spots (npt.NDArray[np.float64]): Spot rates for effective duration calculation.
            eff_dur_delta (float): Delta yiled curve for effective duration calculation.
        """
        ytm = self.calculate_yield_to_maturity(valn_date=valn_date, market_price=market_price)
        mac_dur, mod_dur, conv = self.calculate_duration_convexity(
            valn_date=valn_date, ytm=ytm, market_price=market_price
        )
        eff_dur = self.calculate_effective_duration(
            valn_date=valn_date, spots=spots, market_price=market_price, delta=eff_dur_delta
        )

        return {
            "yield_to_maturity": ytm,
            "macaulay_duration": mac_dur,
            "modified_duration": mod_dur,
            "convexity": conv,
            "effective_duration": eff_dur,
        }

    def calculate_yield_to_maturity(self, *, valn_date: pd.Period, market_price: float) -> float:
        """
        Calculate yield to maturity.

        Args:
            valn_date (pd.Period): Valuation date.
            market_price (float): Market price of the bond.

        Returns:
            float: Yield to maturity.
        """
        coupon_freq = self._pricer.cash_flow_gen.params.coupon_freq
        freq = 1 if coupon_freq == 0 else coupon_freq  # 1 for zero coupon bond
        cash_flows = self._pricer.cash_flow_gen.get_future_cash_flows(valn_date)
        return solve_ytm(
            target_pv=market_price,
            cash_flows=cash_flows,
            freq=freq,
            initial_guess=0.0
        )

    def calculate_duration_convexity(self, *, valn_date: pd.Period, ytm: float, market_price: float | None = None
                                     ) -> tuple[float, float, float]:
        """
        Calculate Macaulay Duration, Modified Duration and Convexity

        Args:
            valn_date (pd.Period): Valuation date.
            ytm (float): Yield to maturity.
            market_price (float|None): Market price for validation, if None skipped.

        Returns:
            tuple[float, float, float]: (mac_duration, mod_duration, convexity).
        """
        coupon_freq = self._pricer.cash_flow_gen.params.coupon_freq
        freq = 1 if coupon_freq == 0 else coupon_freq  # 1 for zero coupon bond
        disc_fac = (1 / (1 + ytm / freq)) ** (1 / 12 * freq)  # monthly discount factor
        cash_flows = self._pricer.cash_flow_gen.get_future_cash_flows(valn_date)

        present_value = 0.0
        weighted_time = 0.0
        convexity_sum = 0.0

        for i in range(len(cash_flows)):
            if cash_flows[i] > 0:
                t = (i + 1) / 12  # t in years
                pv_cf = cash_flows[i] * disc_fac ** (i + 1)
                present_value += pv_cf
                weighted_time += t * pv_cf
                convexity_sum += pv_cf * t * (t + 1 / freq)

        if market_price:
            tolerance = max(market_price * 1e-6, 1e-8)
            if abs(present_value - market_price) > tolerance:
                raise ValueError(f'Present value {present_value: .4f} does not equal market price {market_price: .4f}')

        mac_duration = weighted_time / present_value
        mod_duration = mac_duration / (1 + ytm / freq)

        convexity_denominator = present_value * (1 + ytm / freq) ** 2
        convexity = convexity_sum / convexity_denominator

        return mac_duration, mod_duration, convexity

    def calculate_effective_duration(self, *, valn_date: pd.Period, spots: npt.NDArray[np.float64],
                                     market_price: float | None = None, delta: float = 0.001) -> float:
        """
        Calculate effective duration

        Args:
            valn_date (pd.Period): Valuation date.
            spots (npt.NDArray[np.float64]): Spot rates.
            market_price (float|None): Market price for validation, if None skipped.
            delta (float): Delta yiled curve for effective duration calculation.

        Returns:
            float: Effective duration
        """
        pv0 = self._pricer.calculate_market_price(valn_date, spots)

        if market_price:
            tolerance = max(market_price * 1e-6, 1e-8)
            if abs(pv0 - market_price) > tolerance:
                raise ValueError(f'Present value {pv0: .4f} does not equal market price {market_price: .4f}')

        pvu = self._pricer.calculate_market_price(valn_date, spots + delta)
        pvd = self._pricer.calculate_market_price(valn_date, spots - delta)
        return (pvd - pvu) / (2 * delta * pv0)


class BondFixedCashFlowProvider:
    """
    Provide cash flow for bonds.

    Attributes:
        params (BondFixedParameters): Bond parameters.
        _interest_from_issue (np.ndarray): Bond interest cash flows from issue, index `0, 1, ..., n-1` represents month `1, 2, ..., n`
        _principal_from_issue (np.ndarray): Bond principal cash flows from issue, index ...
        _total_from_issue (np.ndarray): Bond total cash flows from issue, index ...
    """

    __slots__ = ('params', '_interest_from_issue', '_principal_from_issue', '_total_from_issue',)

    def __init__(self, bond_params: BondFixedParameters, provided_cash_flows_dict: dict[str, np.ndarray]):
        """
        Initialize the cash flow generator.

        Args:
            bond_params (BondFixedParameters): Parameters of the bond.
        """
        self.params = bond_params
        self._interest_from_issue: npt.NDArray[np.float64] = provided_cash_flows_dict["interest"]
        self._principal_from_issue: npt.NDArray[np.float64] = provided_cash_flows_dict["principal"]
        n_months = (self.params.maturity_date - self.params.issue_date).n
        if len(self._interest_from_issue) != n_months:
            raise ValueError(f"Length of interest cash flow array ({len(self._interest_from_issue)}) inconsistent with "
                             f"maturity_date - issue_date ({n_months}).")
        if len(self._principal_from_issue) != n_months:
            raise ValueError(f"Length of principal cash flow array ({len(self._principal_from_issue)}) inconsistent with "
                             f"maturity_date - issue_date ({n_months}).")
        self._total_from_issue: npt.NDArray[np.float64] = self._interest_from_issue + self._principal_from_issue

    def get_interest_payment_at_month(self, ref_date: pd.Period) -> float:
        """
        Get the interest payment at a specific month.

        Args:
            ref_date (pd.Period): The date for which to calculate interest.

        Returns:
            float: Interest payment for the month.
        """
        if self.params.issue_date < ref_date <= self.params.maturity_date:
            age_in_month = (ref_date - self.params.issue_date).n
            return float(self._interest_from_issue[age_in_month - 1])
        else:
            return 0.0

    def get_principal_payment_at_month(self, ref_date: pd.Period) -> float:
        """
        Get the principal payment at a specific month.

        Args:
            ref_date (pd.Period): The date for which to calculate principal.

        Returns:
            float: Principal payment for the month. Usually the principal payment is at the maturity date, unless there is a repayment schedule.
        """
        if self.params.issue_date < ref_date <= self.params.maturity_date:
            age_in_month = (ref_date - self.params.issue_date).n
            return float(self._principal_from_issue[age_in_month - 1])
        else:
            return 0.0

    def get_future_cash_flows(self, valn_date: pd.Period) -> npt.NDArray[np.float64] | None:
        """
        Get cash flows from a specific valuation date forward.

        Args:
            valn_date (pd.Period): The valuation date (t).

        Returns:
            npt.NDArray[np.float64] | None: Array of future cash flows (t+1, t+2, ...), None if already matured.

        Raises:
            RuntimeError: If the bond has already matured.
        """
        if valn_date <= self.params.maturity_date:
            age_in_month = (valn_date - self.params.issue_date).n
            return self._total_from_issue[age_in_month:]
        else:
            return None
