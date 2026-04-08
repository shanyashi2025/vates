import numpy as np
import pandas as pd
import numpy.typing as npt
import warnings
from dataclasses import dataclass

from vates.utils import convert_spot_to_disc, newton_raphson_ytm


@dataclass
class BondFixedParameters:
    """
    Data class to hold bond parameters.

    Attributes:
        issue_date (pd.Period): Issue date of the bond.
        maturity_date (pd.Period): Maturity date of the bond.
        coupon_rate (float): Coupon rate.
        coupon_freq (int): Coupon frequency per year, [0, 1, 2, 4, 12].
        face_value (float): Face value of the bond.
        redemp_sched (npt.NDArray[np.float64] | None): Redemption schedule.
    """
    issue_date: pd.Period
    maturity_date: pd.Period
    coupon_rate: float
    coupon_freq: int
    face_value: float
    redemp_sched: npt.NDArray[np.float64] | None = None


class BondFixedCashFlowGenerator:
    """
    Handles cash flow generation for bonds.

    Attributes:
        _params (BondFixedParameters): Bond parameters.
    """

    def __init__(self, bond_params: BondFixedParameters):
        """
        Initialize the cash flow generator.

        Args:
            bond_params (BondFixedParameters): Parameters of the bond.
        """
        self._params = bond_params
        self._outstanding_principal_from_issue: npt.NDArray[np.float64] | None = None
        self._cash_flow_from_issue: npt.NDArray[np.float64] | None = None

        if self._params.issue_date >= self._params.maturity_date:
            raise ValueError(f'issue_date {self._params.issue_date} is later than '
                             f'maturity_date {self._params.maturity_date}.')
        if self._params.face_value <= 0:
            raise ValueError(f'face_value={self._params.face_value} should be positive.')
        if self._params.coupon_freq not in (0, 1, 2, 4, 12):
            raise ValueError(f'Invalid coupon_freq={self._params.coupon_freq}, expected [0, 1, 2, 4, 12].')
        if not -0.05 < self._params.coupon_rate < 0.2:
            warnings.warn(f'coupon_rate={self._params.coupon_rate} not in normal range, recommend to check.')

    def get_interest_payment_at_month(self, ref_date: pd.Period) -> float:
        """
        Get the interest payment at a specific month.

        Args:
            ref_date (pd.Period): The date for which to calculate interest.

        Returns:
            float: Interest payment for the month.
        """
        if self._params.coupon_rate == 0.0:
            return 0.0
        elif self._params.coupon_freq in (1, 2, 4, 12):
            coupon_interval = 12 // self._params.coupon_freq
            age_in_month = (ref_date - self._params.issue_date).n
            if age_in_month % coupon_interval == 0:
                beg_principal = float(self._outstanding_principal_from_issue[age_in_month - 1])
                return beg_principal * self._params.coupon_rate / self._params.coupon_freq
            else:
                return 0.0
        elif self._params.coupon_freq == 0:
            if ref_date == self._params.maturity_date:
                age_in_month = (ref_date - self._params.issue_date).n
                beg_principal = float(self._outstanding_principal_from_issue[age_in_month - 1])
                return beg_principal * self._params.coupon_rate
            else:
                return 0.0
        else:  # coupon_freq has been validated in `__init__`, should never get here
            raise ValueError(f'Invalid coupon_freq={self._params.coupon_freq}, expected [0, 1, 2, 4, 12].')

    def get_principal_payment_at_month(self, ref_date: pd.Period) -> float:
        """
        Get the principal payment at a specific month.

        Args:
            ref_date (pd.Period): The date for which to calculate principal.

        Returns:
            float: Principal payment for the month. Usually the principal payment is at the maturity date, unless there is a repayment schedule.
        """
        if self._outstanding_principal_from_issue is None:
            self._calculate_outstanding_principal_from_issue()

        age_in_month = (ref_date - self._params.issue_date).n
        principal = float(self._outstanding_principal_from_issue[age_in_month])
        principal_prev = float(self._outstanding_principal_from_issue[age_in_month - 1])

        return principal_prev - principal

    def _calculate_outstanding_principal_from_issue(self) -> None:
        """
        Calculate the outstanding principal array from the issue date for each month (end).
        """
        if self._outstanding_principal_from_issue: return

        n_months = (self._params.maturity_date - self._params.issue_date).n
        os_principal = np.zeros(n_months + 1)

        if self._params.redemp_sched is None:
            os_principal[:n_months] = self._params.face_value
            os_principal[n_months] = 0.0
        else:
            os_principal[0] = self._params.face_value
            os_principal[n_months] = 0.0  # force zero at maturity date
            for i in range(1, n_months):
                os_principal[i] = (os_principal[i - 1] - self._params.face_value * self._params.redemp_sched[i])

        self._outstanding_principal_from_issue = os_principal

    def _calculate_cash_flows_from_issue(self) -> None:
        """
        Generate the cash flow array from bond start date for each month.
        """
        if self._cash_flow_from_issue: return

        self._calculate_outstanding_principal_from_issue()

        n_months = (self._params.maturity_date - self._params.issue_date).n
        cash_flows = np.zeros(n_months)

        for i in range(n_months):
            ref_date = self._params.issue_date + i + 1
            cash_flows[i] = self.get_interest_payment_at_month(ref_date) + self.get_principal_payment_at_month(ref_date)

        self._cash_flow_from_issue = cash_flows

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
        if self._cash_flow_from_issue is None:
            self._calculate_cash_flows_from_issue()

        if valn_date < self._params.maturity_date:
            age_in_month = (valn_date - self._params.issue_date).n
            return self._cash_flow_from_issue[age_in_month:]
        else:
            return None


class BondFixedPricer:
    """
    Handles bond pricing calculations.

    Attributes:
        _params (BondFixedParameters): Bond parameters.
        _cash_flow_gen (BondFixedCashFlowGenerator): Cash flow generator.
    """

    def __init__(self, bond_params: BondFixedParameters, cash_flow_generator: BondFixedCashFlowGenerator):
        """
        Initialize the bond pricer.

        Args:
            bond_params (BondFixedParameters): Parameters of the bond.
            cash_flow_generator (BondFixedCashFlowGenerator): Cash flow generator.
        """
        self._params = bond_params
        self._cash_flow_gen = cash_flow_generator

    def calculate_market_price(self, valn_date: pd.Period, spots: npt.NDArray[np.float64]) -> float:
        """
        Calculate market price using risk-adjusted spot rates.

        Args:
            valn_date (pd.Period): Valuation date.
            spots (npt.NDArray[np.float64]): Spot rates.

        Returns:
            float: Market price.
        """
        cash_flows = self._cash_flow_gen.get_future_cash_flows(valn_date)
        if cash_flows is not None:  # return None if matured
            dfs = convert_spot_to_disc(spots, "M")
            return np.dot(cash_flows, dfs[1:len(cash_flows) + 1])
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
        cash_flows = self._cash_flow_gen.get_future_cash_flows(valn_date)
        if cash_flows is not None:  # return None if matured
            periods = np.arange(1, len(cash_flows) + 1)
            freq = 1 if self._params.coupon_freq == 0 else self._params.coupon_freq # 1 for zero coupon bond
            df = (1 / (1 + amort_rate / freq)) ** (1 / 12 * freq)
            dfs = df ** periods
            return np.dot(cash_flows, dfs)
        else:
            return 0.0


class BondFixedRiskCalculator:
    """
    Handles risk metric calculations (duration, convexity, etc.).

    Attributes:
        _params (BondFixedParameters): Bond parameters.
        _cash_flow_gen (BondFixedCashFlowGenerator): Cash flow generator.
        _pricer (BondFixedPricer): Bond pricer.
    """

    def __init__(self, bond_params: BondFixedParameters, cash_flow_generator: BondFixedCashFlowGenerator,
                 pricer: BondFixedPricer):
        """
        Initialize the risk calculator.

        Args:
            bond_params (BondFixedParameters): Parameters of the bond.
            cash_flow_generator (BondFixedCashFlowGenerator): Cash flow generator.
            pricer (BondFixedPricer): Bond pricer.
        """
        self._params = bond_params
        self._cash_flow_gen = cash_flow_generator
        self._pricer = pricer
        self._yield_to_maturity: float | None = None
        self._macaulay_duration: float | None = None
        self._modified_duration: float | None = None
        self._convexity: float | None = None
        self._effective_duration: float | None = None

    @property
    def risk_metrics(self) -> dict[str, float | None]:
        """
        Return all calculated risk metrics.

        Returns:
            dict[str, float]: Dictionary of risk metrics.
        """
        return {
            'yield_to_maturity': self._yield_to_maturity,
            'macaulay_duration': self._macaulay_duration,
            'modified_duration': self._modified_duration,
            'convexity': self._convexity,
            'effective_duration': self._effective_duration,
        }

    def calculate_all_risk_metrics(self, valn_date: pd.Period, market_price: float,
                                   spots: npt.NDArray[np.float64], eff_dur_delta: float=0.001):
        """
        Calculate all risk metrics for the bond.

        Args:
            valn_date (pd.Period): Valuation date.
            market_price (float): Market price of the bond.
            spots (npt.NDArray[np.float64]): Spot rates for effective duration calculation.
            eff_dur_delta (float): Delta yiled curve for effective duration calculation.
        """
        self._yield_to_maturity = self.calculate_yield_to_maturity(valn_date, market_price)
        self._macaulay_duration, self._modified_duration, self._convexity = self.calculate_duration_convexity(
            valn_date, self._yield_to_maturity, market_price
        )
        self._effective_duration = self.calculate_effective_duration(valn_date, spots, market_price, eff_dur_delta)

    def calculate_yield_to_maturity(self, valn_date: pd.Period, market_price: float) -> float:
        """
        Calculate yield to maturity.

        Args:
            valn_date (pd.Period): Valuation date.
            market_price (float): Market price of the bond.

        Returns:
            float: Yield to maturity.
        """
        freq = 1 if self._params.coupon_freq == 0 else self._params.coupon_freq  # 1 for zero coupon bond
        cash_flows = self._cash_flow_gen.get_future_cash_flows(valn_date)
        return newton_raphson_ytm(
            target_pv=market_price,
            cash_flows=cash_flows,
            freq=freq,
            initial_guess=0.0
        )

    def calculate_duration_convexity(self, valn_date: pd.Period, ytm: float, market_price: float|None = None) \
            -> tuple[float, float, float]:
        """
        Calculate Macaulay Duration, Modified Duration and Convexity

        Args:
            valn_date (pd.Period): Valuation date.
            ytm (float): Yield to maturity.
            market_price (float|None): Market price for validation, if None skipped.

        Returns:
            tuple[float, float, float]: (mac_duration, mod_duration, convexity).
        """
        freq = 1 if self._params.coupon_freq == 0 else self._params.coupon_freq  # 1 for zero coupon bond
        df = (1 / (1 + ytm / freq)) ** (1 / 12 * freq)  # monthly discount factor
        cash_flows = self._cash_flow_gen.get_future_cash_flows(valn_date)

        present_value = 0.0
        weighted_time = 0.0
        convexity_sum = 0.0

        for i in range(len(cash_flows)):
            if cash_flows[i] > 0:
                t = (i + 1) / 12  # t in years
                pv_cf = cash_flows[i] * df ** (i + 1)
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

    def calculate_effective_duration(self, valn_date: pd.Period, spots: npt.NDArray[np.float64],
                                     market_price: float|None=None, delta: float=0.001) -> float:
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
