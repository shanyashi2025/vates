import numpy as np
import numpy.typing as npt
import warnings
from enum import Enum
from typing import Callable, Self
from dataclasses import dataclass

class InterestRateAlias(Enum):
    spot = "spot"
    zero = "spot"
    spot_rate = "spot"
    spot_rates = "spot"
    forward = "forward"
    forward_rate = "forward"
    forward_rates = "forward"
    discount = "discount"
    discount_factor = "discount"
    discount_factors = "discount"
    par = "par"
    par_yield = "par"
    par_yields = "par"
    par_rate = "par"
    par_rates = "par"


def convert_interest_rates(rates: np.ndarray, /, *, time_interval: float = 1, from_type: str, to_type: str, **kwargs
                          ) -> npt.NDArray[np.float64]:
    from_type = InterestRateAlias[from_type.lower()].value
    to_type = InterestRateAlias[to_type.lower()].value
    func = InterestRateConvertor.get_func(from_type=from_type, to_type=to_type)
    return func(rates, time_interval=time_interval, **kwargs)


class InterestRateConvertor:

    @classmethod
    def spot_to_discount(cls, spot: npt.NDArray[np.float64], /, *, time_interval: float = 1) -> npt.NDArray[np.float64]:
        """
        Convert spot rates to discount factors.

        Args:
            spot (npt.NDArray[np.float64]): Array of spot rates.
            time_interval: Time interval in years, defaults to 1.

        Returns:
            npt.NDArray[np.float64]: Array of discount factors.
        """
        time = np.arange(len(spot)) * time_interval
        return (1 + spot) ** (-time)

    @classmethod
    def forward_to_discount(cls, forward: npt.NDArray[np.float64], /, *, time_interval: float = 1
                            ) -> npt.NDArray[np.float64]:
        """
        Convert forward rates to discount factors.

        Args:
            forward (npt.NDArray[np.float64]): Array of forward rates.
            time_interval: Time interval in years, defaults to 1.

        Returns:
            npt.NDArray[np.float64]: Array of discount factors.
        """
        factors = (1 + forward) ** (-time_interval)
        return np.cumprod(factors, dtype=float)

    @classmethod
    def discount_to_spot(cls, discount: npt.NDArray[np.float64], /, *, time_interval: float = 1
                         ) -> npt.NDArray[np.float64]:
        """
        Convert discount factors to spot rates.

        Args:
            discount (npt.NDArray[np.float64]): Array of discount factors.
            time_interval: Time interval in years, defaults to 1.

        Returns:
            npt.NDArray[np.float64]: Array of spot rates.
        """
        time = np.arange(len(discount)) * time_interval
        spot = np.zeros(len(discount))
        spot[1:] = discount[1:] ** (-1 / time[1:]) - 1
        return spot

    @classmethod
    def discount_to_forward(cls, discount: npt.NDArray[np.float64], /, *, time_interval: float = 1
                            ) -> npt.NDArray[np.float64]:
        """
        Convert discount factors to forward rates.

        Args:
            discount (npt.NDArray[np.float64]): Array of discount factors.
            time_interval: Time interval in years, defaults to 1.

        Returns:
            npt.NDArray[np.float64]: Array of forward rates.
        """
        forward = np.zeros(len(discount))
        forward[1:] = (discount[:-1] / discount[1:]) ** (1 / time_interval) - 1
        return forward

    @classmethod
    def discount_to_par(cls, discount: npt.NDArray[np.float64], /, *,time_interval: float = 1,  freq: int = 1,
                        ) -> npt.NDArray[np.float64]:
        """
        Convert discount factors to par yields.

        Args:
            discount (npt.NDArray[np.float64]): Array of discount factors.
            freq (int): Coupon frequency (1, 2, 4, or 12).
            time_interval: Time interval in years, defaults to 1.

        Returns:
            npt.NDArray[np.float64]: Array of par yields.

        Raises:
            ValueError: If payment_in_year_or_month.
        """
        if freq not in (1, 2, 4, 12):
            raise ValueError(f"Invalid payment frequency: {freq}. Must be 1, 2, 4, or 12")

        if time_interval == 1 and freq != 1:
            warnings.warn(f"Cannot calculate {freq=} with yearly rates provided, 'freq' is set to 1.")
            freq = 1

        step = 1 if (time_interval == 1 and freq == 1) else 12 // freq

        par = np.zeros(len(discount))
        ann_factor = 0.0
        for i in range(step, len(discount), step):
            ann_factor += discount[i]
            par[i] = (1 - discount[i]) / ann_factor * freq

        return par

    @classmethod
    def spot_to_forward(cls, spot: npt.NDArray[np.float64], /, *, time_interval: float = 1) -> npt.NDArray[np.float64]:
        """
        Convert spot rates to forward rates.

        Args:
            spot (npt.NDArray[np.float64]): Array of spot rates.
            time_interval: Time interval in years, defaults to 1.

        Returns:
            npt.NDArray[np.float64]: Array of forward rates.
        """
        discount = cls.spot_to_discount(spot, time_interval=time_interval)
        return cls.discount_to_forward(discount, time_interval=time_interval)

    @classmethod
    def spot_to_par(cls, spot: npt.NDArray[np.float64], /, *, time_interval: float = 1, freq: int,
                    ) -> npt.NDArray[np.float64]:
        """
        Convert spot rates to par yields.

        Args:
            spot (npt.NDArray[np.float64]): Array of spot rates.
            freq (int): Coupon frequency (1, 2, 4, or 12).
            time_interval: Time interval in years, defaults to 1.

        Returns:
            npt.NDArray[np.float64]: Array of par yields.
        """
        discount = cls.spot_to_discount(spot, time_interval=time_interval)
        return cls.discount_to_par(discount, freq=freq, time_interval=time_interval)

    @classmethod
    def forward_to_spot(cls, forward: npt.NDArray[np.float64], /, *, time_interval: float = 1) -> npt.NDArray[np.float64]:
        """
        Convert forward rates to spot rates.

        Args:
            forward (npt.NDArray[np.float64]): Array of forward rates.
            time_interval: Time interval in years, defaults to 1.

        Returns:
            npt.NDArray[np.float64]: Array of forward rates.
        """
        discount = cls.forward_to_discount(forward, time_interval=time_interval)
        return cls.discount_to_spot(discount, time_interval=time_interval)

    @classmethod
    def forward_to_par(cls, forward: npt.NDArray[np.float64], /, *, time_interval: float = 1, freq: int,
                       ) -> npt.NDArray[np.float64]:
        """
        Convert spot rates to par yields.

        Args:
            forward (npt.NDArray[np.float64]): Array of forward rates.
            freq (int): Coupon frequency (1, 2, 4, or 12).
            time_interval: Time interval in years, defaults to 1.

        Returns:
            npt.NDArray[np.float64]: Array of par yields.
        """
        discount = cls.forward_to_discount(forward, time_interval=time_interval)
        return cls.discount_to_par(discount, freq=freq, time_interval=time_interval)

    @classmethod
    def get_func(cls, *, from_type: str, to_type: str) -> Callable:
        if from_type == "discount" and to_type == "forward":
            return cls.discount_to_forward
        if from_type == "discount" and to_type == "par":
            return cls.discount_to_par
        if from_type == "discount" and to_type == "spot":
            return cls.discount_to_spot
        if from_type == "forward" and to_type == "discount":
            return cls.forward_to_discount
        if from_type == "forward" and to_type == "par":
            return cls.forward_to_par
        if from_type == "forward" and to_type == "spot":
            return cls.forward_to_spot
        if from_type == "spot" and to_type == "discount":
            return cls.spot_to_discount
        if from_type == "spot" and to_type == "forward":
            return cls.spot_to_forward
        if from_type == "par" and to_type == "spot_to_par":
            return cls.spot_to_discount
        raise ValueError(f"Invalid from '{from_type}' to '{to_type}' combination.")


def solve_z_spread(*, target_pv: float, cash_flows: npt.NDArray[np.float64], spots: npt.NDArray[np.float64]) -> float:
    """
    Solve z-spread using the Newton-Raphson method.

    Args:
        target_pv (float): Target present value.
        cash_flows (npt.NDArray[np.float64]): Array of cash flows, `0, 1, ..., n-1` represents month `1, 2, ..., n`
        spots (npt.NDArray[np.float64]): Array of spot rates, `1, 2, ..., n` represents month `1, 2, ..., n`

    Returns:
        float: Solved z-spread.

    Raises:
        ValueError: If the method does not converge or input is invalid.
    """
    tolerance = 1e-10  # Numerical tolerance for calculations
    max_iterations = 100  # Maximum iterations for iterative methods
    epsilon = 0.0001  # Small increment for numerical derivative

    if abs(target_pv) < tolerance:
        raise ValueError("Target present value cannot be zero")

    min_spot_val = spots.min()
    n_months = len(cash_flows)
    z = 0.0  # initial guess

    for _ in range(max_iterations):  # max iterations
        spots_plus_z = spots + z
        discount = InterestRateConvertor.spot_to_discount(spots_plus_z, time_interval=1/12)
        pv = np.dot(cash_flows, discount[1: n_months + 1])

        if abs(pv / target_pv - 1) < tolerance:
            return z

        # Newton-Raphson approximation
        if pv > target_pv:
            delta = epsilon
        else:
            delta = max(-epsilon, tolerance - 1 - min_spot_val - z)  # ensure (1 + min_spot_val + z + delta) > 0
        discount = InterestRateConvertor.spot_to_discount(spots_plus_z + delta, time_interval=1/12)
        pv_delta = np.dot(cash_flows, discount[1: n_months + 1])
        derivative = (pv_delta - pv) / delta

        if abs(derivative) < tolerance:
            return z

        z = z - (pv - target_pv) / derivative
        z = max(z, tolerance - 1 - min_spot_val)  # ensure (1 + min_spot_val + z) > 0 for edge case

    raise ValueError(f"Newton-Raphson method did not converge after {max_iterations} iterations")


def solve_ytm(*, target_pv: float, cash_flows: npt.NDArray[np.float64], freq: int = 1,
              initial_guess: float = 0.0) -> float:
    """
    Solve yield to maturity using the Newton-Raphson method.

    Args:
        target_pv (float): Target present value.
        cash_flows (npt.NDArray[np.float64]): Array of cash flows, `0, 1, ..., n-1` represents month `1, 2, ..., n`
        freq (int): Coupon frequency, [1, 2, 4, 12]
        initial_guess (float, optional): Initial guess for yield to maturity. Defaults to 0.0.

    Returns:
        float: Internal rate of return.

    Raises:
        ValueError: If the method does not converge or input is invalid.
    """
    if freq not in (1, 2, 4, 12):
        raise ValueError(f'Invalid {freq=}, expected [1, 2, 4, 12].')

    tolerance = 1e-10  # Numerical tolerance for calculations
    max_iterations = 100  # Maximum iterations for iterative methods
    epsilon = 0.0001  # Small increment for numerical derivative

    if abs(target_pv) < tolerance:
        raise ValueError("Target present value cannot be zero")

    time = np.arange(1, len(cash_flows) + 1)  # time in month
    ytm = initial_guess

    for _ in range(max_iterations):
        factor = (1 + ytm / freq) ** (-freq / 12)  # monthly discount factor
        discount = factor ** time
        pv = np.dot(cash_flows, discount)

        # Check convergence
        if abs(pv / target_pv - 1) < tolerance:
            return ytm

        # Calculate numerical derivative
        factor = (1 + (ytm + epsilon) / freq) ** (-freq / 12)
        discount = factor ** time
        pv_up = np.dot(cash_flows, discount)

        derivative = (pv_up - pv) / epsilon

        # Check if derivative is too small
        if abs(derivative) < tolerance:
            return ytm

        # Newton-Raphson update
        ytm = ytm - (pv - target_pv) / derivative
        ytm = max(ytm, (tolerance - 1) * freq)  # ensure (1 + ytm / in_year_or_month) > 0 for edge case

    raise ValueError(f"Newton-Raphson method did not converge after {max_iterations} iterations")


@dataclass(frozen=True, slots=True)
class InterestRateTermStructure:
    """Interest rate term structure"""
    discount: np.ndarray
    forwardac: np.ndarray
    forwardcc: np.ndarray
    zeroac: np.ndarray
    zerocc: np.ndarray

    @property
    def spotac(self) -> np.ndarray:
        return self.zeroac

    @property
    def spotcc(self) -> np.ndarray:
        return self.zerocc

    @property
    def max_maturity(self) -> int:
        return len(self.discount) - 1

    @classmethod
    def from_discount(cls, discount, /) -> Self:
        max_maturity = len(discount) - 1
        forwardac = np.zeros(max_maturity + 1)
        forwardcc = np.zeros(max_maturity + 1)
        zeroac = np.zeros(max_maturity + 1)
        zerocc = np.zeros(max_maturity + 1)
        zeroac[1:] = discount[1:] ** (-1 / np.arange(1, max_maturity + 1)) - 1
        forwardac[1:] = discount[:-1] / discount[1:] - 1
        forwardcc[1:] = np.log1p(forwardac[1:])
        zerocc[1:] = np.log1p(zeroac[1:])
        return InterestRateTermStructure(
            discount=discount,
            forwardac=forwardac,
            forwardcc=forwardcc,
            zeroac=zeroac,
            zerocc=zerocc,
        )

    @classmethod
    def from_zeroac(cls, zeroac, /) -> Self:
        max_maturity = len(zeroac) - 1
        discount = np.zeros(max_maturity + 1)
        forwardac = np.zeros(max_maturity + 1)
        forwardcc = np.zeros(max_maturity + 1)
        zerocc = np.zeros(max_maturity + 1)
        discount[0] = 1.0
        discount[1:] = (1.0 + zeroac[1:]) ** (1 / np.arange(1, max_maturity + 1))
        forwardac[1:] = discount[:-1] / discount[1:] - 1
        forwardcc[1:] = np.log1p(forwardac[1:])
        zerocc[1:] = np.log1p(zeroac[1:])
        return InterestRateTermStructure(
            discount=discount,
            zeroac=zeroac,
            zerocc=zerocc,
            forwardac=forwardac,
            forwardcc=forwardcc,
        )

    @classmethod
    def from_forwardac(cls, forwardac, /) -> Self:
        max_maturity = len(forwardac) - 1
        discount = np.zeros(max_maturity + 1)
        forwardcc = np.zeros(max_maturity + 1)
        zeroac = np.zeros(max_maturity + 1)
        zerocc = np.zeros(max_maturity + 1)
        forwardcc[1:] = np.log1p(forwardac[1:])
        discount[0] = 1.0
        discount[1:] = np.cumprod(1.0 / (1.0 + forwardac[1:]))
        zeroac[1:] = discount[1:] ** (-1 / np.arange(1, max_maturity + 1)) - 1
        zerocc[1:] = np.log1p(zeroac[1:])
        return InterestRateTermStructure(
            discount=discount,
            zeroac=zeroac,
            zerocc=zerocc,
            forwardac=forwardac,
            forwardcc=forwardcc,
        )
