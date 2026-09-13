import numpy as np
import numpy.typing as npt
from enum import Enum
from typing import Callable, Literal, Self
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


def _normalize_rate_alias(alias: str, *, name: str) -> str:
    """Map a user-provided rate alias to its canonical form.

    Args:
        alias: Rate alias, e.g. ``"spot_rates"``, ``"zero"``, ``"discount_factors"``.
        name: Argument name used in the error message.

    Returns:
        str: Canonical rate type, one of ``"spot"``, ``"forward"``, ``"discount"``, ``"par"``.

    Raises:
        ValueError: If `alias` is not a known rate alias.
    """
    if not isinstance(alias, str):
        raise ValueError(f"Invalid {name}={alias!r}, expected str.")
    try:
        return InterestRateAlias[alias.lower()].value
    except KeyError:
        valid = sorted({a.name for a in InterestRateAlias} | {a.value for a in InterestRateAlias})
        raise ValueError(f"Invalid {name}={alias!r}, expected one of {valid}.") from None


def frequency_from_interval_unit(interval_unit: Literal["Y", "M"] = "Y", /) -> int:
    """Return the number of grid steps per year for the given interval unit.

    Args:
        interval_unit (Literal["Y", "M"]): Grid interval unit, year or month.

    Returns:
        int: 1 for yearly, 12 for monthly.

    Raises:
        ValueError: If `interval_unit` is not `"Y"` or `"M"`.
    """
    if interval_unit == "Y":
        return 1
    elif interval_unit == "M":
        return 12
    else:
        raise ValueError(f"Invalid {interval_unit=}, expected 'Y' or 'M'.")


def convert_interest_rates(rates: np.ndarray, /, *, interval_unit: Literal["Y", "M"] = "Y", from_what: str, to_what: str,
                           **kwargs) -> npt.NDArray[np.float64]:
    """
    Convert interest rates from one representation to another.

    Args:
        rates (np.ndarray): Rates to convert, index `0, 1, ..., n` corresponds to term `0, 1, ..., n`.
        interval_unit (Literal["Y", "M"]): Grid interval unit, year or month. Defaults to year.
        from_what (str): Source representation, one of `"spot"`, `"forward"`, `"discount"`, `"par"`
            (common aliases such as `"zero"` or `"spot_rates"` are accepted).
        to_what (str): Target representation, same accepted values as `from_what`.
        **kwargs: Extra keyword arguments forwarded to the underlying converter (e.g. `freq` for par yields).

    Returns:
        npt.NDArray[np.float64]: Converted rates.

    Raises:
        ValueError: If `from_what`/`to_what` is unknown or the conversion is not supported.
    """
    from_what = _normalize_rate_alias(from_what, name="from_what")
    to_what = _normalize_rate_alias(to_what, name="to_what")
    func = InterestRateConvertor.get_func(from_what=from_what, to_what=to_what)
    return func(rates, interval_unit=interval_unit, **kwargs)


class InterestRateConvertor:

    @classmethod
    def spot_to_discount(cls, spot: npt.NDArray[np.float64], /, *, interval_unit: Literal["Y", "M"] = "Y"
                         ) -> npt.NDArray[np.float64]:
        """
        Convert spot rates to discount factors.

        Args:
            spot (npt.NDArray[np.float64]): Array of spot rates.
            interval_unit (Literal["Y", "M"]): Grid interval unit, year or month. Defaults to year.

        Returns:
            npt.NDArray[np.float64]: Array of discount factors.
        """
        interval_in_years = 1 / frequency_from_interval_unit(interval_unit)
        time = np.arange(len(spot)) * interval_in_years
        return (1 + spot) ** (-time)

    @classmethod
    def forward_to_discount(cls, forward: npt.NDArray[np.float64], /, *, interval_unit: Literal["Y", "M"] = "Y"
                            ) -> npt.NDArray[np.float64]:
        """
        Convert forward rates to discount factors.

        Args:
            forward (npt.NDArray[np.float64]): Array of forward rates.
            interval_unit (Literal["Y", "M"]): Grid interval unit, year or month. Defaults to year.

        Returns:
            npt.NDArray[np.float64]: Array of discount factors.
        """
        interval_in_years = 1 / frequency_from_interval_unit(interval_unit)
        factors = (1 + forward) ** (-interval_in_years)
        return np.cumprod(factors, dtype=float)

    @classmethod
    def discount_to_spot(cls, discount: npt.NDArray[np.float64], /, *, interval_unit: Literal["Y", "M"] = "Y"
                         ) -> npt.NDArray[np.float64]:
        """
        Convert discount factors to spot rates.

        Args:
            discount (npt.NDArray[np.float64]): Array of discount factors.
            interval_unit (Literal["Y", "M"]): Grid interval unit, year or month. Defaults to year.

        Returns:
            npt.NDArray[np.float64]: Array of spot rates.
        """
        interval_in_years = 1 / frequency_from_interval_unit(interval_unit)
        time = np.arange(len(discount)) * interval_in_years
        spot = np.zeros(len(discount))
        spot[1:] = discount[1:] ** (-1 / time[1:]) - 1
        return spot

    @classmethod
    def discount_to_forward(cls, discount: npt.NDArray[np.float64], /, *, interval_unit: Literal["Y", "M"] = "Y"
                            ) -> npt.NDArray[np.float64]:
        """
        Convert discount factors to forward rates.

        Args:
            discount (npt.NDArray[np.float64]): Array of discount factors.
            interval_unit (Literal["Y", "M"]): Grid interval unit, year or month. Defaults to year.

        Returns:
            npt.NDArray[np.float64]: Array of forward rates.
        """
        frequency = frequency_from_interval_unit(interval_unit)
        forward = np.zeros(len(discount))
        forward[1:] = (discount[:-1] / discount[1:]) ** frequency - 1
        return forward

    @classmethod
    def discount_to_par(cls, discount: npt.NDArray[np.float64], /, *, interval_unit: Literal["Y", "M"] = "Y",
                        coupon_frequency: int = 1, ) -> npt.NDArray[np.float64]:
        """
        Convert discount factors to par yields.

        Par yields are only populated on coupon dates (multiples of `_coupon_step`); all other entries are zero.

        Args:
            discount (npt.NDArray[np.float64]): Array of discount factors.
            coupon_frequency (int): Coupon frequency (1, 2, 4, or 12).
            interval_unit (Literal["Y", "M"]): Grid interval unit, year or month. Defaults to year.

        Returns:
            npt.NDArray[np.float64]: Array of par yields (zeros on non-coupon dates).

        Raises:
            ValueError: If `freq` is invalid or cannot be represented on the grid.
        """
        grid_frequency = frequency_from_interval_unit(interval_unit)
        if coupon_frequency not in (1, 2, 4, 12):
            raise ValueError(f"Invalid payment frequency: {coupon_frequency}. Must be 1, 2, 4, or 12.")
        if grid_frequency < coupon_frequency:
            raise ValueError(
                f"Cannot represent coupon frequency {coupon_frequency} on a grid of {interval_unit} interval unit: "
                f"a coupon period spans {grid_frequency / coupon_frequency:.4g} grid step(s)."
            )
        step = round(grid_frequency / coupon_frequency)

        par = np.zeros(len(discount))
        ann_factor = 0.0
        for i in range(step, len(discount), step):
            ann_factor += discount[i]
            par[i] = (1 - discount[i]) / ann_factor * coupon_frequency

        return par

    @classmethod
    def spot_to_forward(cls, spot: npt.NDArray[np.float64], /, *, interval_unit: Literal["Y", "M"] = "Y") -> npt.NDArray[np.float64]:
        """
        Convert spot rates to forward rates.

        Args:
            spot (npt.NDArray[np.float64]): Array of spot rates.
            interval_unit (Literal["Y", "M"]): Grid interval unit, year or month. Defaults to year.

        Returns:
            npt.NDArray[np.float64]: Array of forward rates.
        """
        discount = cls.spot_to_discount(spot, interval_unit=interval_unit)
        return cls.discount_to_forward(discount, interval_unit=interval_unit)

    @classmethod
    def spot_to_par(cls, spot: npt.NDArray[np.float64], /, *, interval_unit: Literal["Y", "M"] = "Y",
                    coupon_frequency: int, ) -> npt.NDArray[np.float64]:
        """
        Convert spot rates to par yields.

        Args:
            spot (npt.NDArray[np.float64]): Array of spot rates.
            coupon_frequency (int): Coupon frequency (1, 2, 4, or 12).
            interval_unit (Literal["Y", "M"]): Grid interval unit, year or month. Defaults to year.

        Returns:
            npt.NDArray[np.float64]: Array of par yields.
        """
        discount = cls.spot_to_discount(spot, interval_unit=interval_unit)
        return cls.discount_to_par(discount, coupon_frequency=coupon_frequency, interval_unit=interval_unit)

    @classmethod
    def forward_to_spot(cls, forward: npt.NDArray[np.float64], /, *, interval_unit: Literal["Y", "M"] = "Y"
                        ) -> npt.NDArray[np.float64]:
        """
        Convert forward rates to spot rates.

        Args:
            forward (npt.NDArray[np.float64]): Array of forward rates.
            interval_unit (Literal["Y", "M"]): Grid interval unit, year or month. Defaults to year.

        Returns:
            npt.NDArray[np.float64]: Array of spot rates.
        """
        discount = cls.forward_to_discount(forward, interval_unit=interval_unit)
        return cls.discount_to_spot(discount, interval_unit=interval_unit)

    @classmethod
    def forward_to_par(cls, forward: npt.NDArray[np.float64], /, *, interval_unit: Literal["Y", "M"] = "Y",
                       coupon_frequency: int, ) -> npt.NDArray[np.float64]:
        """
        Convert forward rates to par yields.

        Args:
            forward (npt.NDArray[np.float64]): Array of forward rates.
            coupon_frequency (int): Coupon frequency (1, 2, 4, or 12).
            interval_unit (Literal["Y", "M"]): Grid interval unit, year or month. Defaults to year.

        Returns:
            npt.NDArray[np.float64]: Array of par yields.
        """
        discount = cls.forward_to_discount(forward, interval_unit=interval_unit)
        return cls.discount_to_par(discount, coupon_frequency=coupon_frequency, interval_unit=interval_unit)

    @classmethod
    def get_func(cls, *, from_what: str, to_what: str) -> Callable:
        """Return the converter method for the requested rate-type pair.

        Args:
            from_what: Canonical source rate type (`"spot"`, `"forward"`, `"discount"`, `"par"`).
            to_what: Canonical target rate type.

        Returns:
            Callable: Converter method of this class.

        Raises:
            ValueError: If the pair is not supported.
        """
        if from_what == "discount" and to_what == "forward":
            return cls.discount_to_forward
        if from_what == "discount" and to_what == "par":
            return cls.discount_to_par
        if from_what == "discount" and to_what == "spot":
            return cls.discount_to_spot
        if from_what == "forward" and to_what == "discount":
            return cls.forward_to_discount
        if from_what == "forward" and to_what == "par":
            return cls.forward_to_par
        if from_what == "forward" and to_what == "spot":
            return cls.forward_to_spot
        if from_what == "spot" and to_what == "discount":
            return cls.spot_to_discount
        if from_what == "spot" and to_what == "forward":
            return cls.spot_to_forward
        if from_what == "spot" and to_what == "par":
            return cls.spot_to_par
        raise ValueError(f"Invalid from '{from_what}' to '{to_what}' combination.")


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
        discount = InterestRateConvertor.spot_to_discount(spots_plus_z, interval_unit="M")
        pv = np.dot(cash_flows, discount[1: n_months + 1])

        if abs(pv / target_pv - 1) < tolerance:
            return z

        # Newton-Raphson approximation
        if pv > target_pv:
            delta = epsilon
        else:
            delta = max(-epsilon, tolerance - 1 - min_spot_val - z)  # ensure (1 + min_spot_val + z + delta) > 0
        discount = InterestRateConvertor.spot_to_discount(spots_plus_z + delta, interval_unit="M")
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


@dataclass(frozen=True, slots=True, eq=False)
class InterestRateTermStructure:
    """Interest rate term structure.

    All rate arrays are indexed by term: entry `i` corresponds to term `i` grid steps, where a grid step is
    one year when `interval_unit == "Y"` and one month when `interval_unit == "M"`.

    Attributes:
        discount (np.ndarray): Discount factors.
        forwardac (np.ndarray): Annually compounded forward rates.
        forwardcc (np.ndarray): Continuously compounded forward rates.
        zeroac (np.ndarray): Annually compounded zero (spot) rates.
        zerocc (np.ndarray): Continuously compounded zero (spot) rates.
        parac (dict[int, np.ndarray]): Par yields keyed by coupon frequency.
        interval_unit (Literal["Y", "M"]): Grid interval unit, yearly or monthly.
    """
    discount: np.ndarray
    forwardac: np.ndarray
    forwardcc: np.ndarray
    zeroac: np.ndarray
    zerocc: np.ndarray
    parac: dict[int, np.ndarray]
    interval_unit: Literal["Y", "M"]

    def __post_init__(self) -> None:
        if self.interval_unit not in ("Y", "M"):
            raise ValueError(f"Invalid interval_unit={self.interval_unit!r}, expected 'Y' or 'M'.")
        lengths = {len(self.discount), len(self.forwardac), len(self.forwardcc), len(self.zeroac), len(self.zerocc)}
        if len(lengths) != 1:
            raise ValueError(f"Inconsistent rate array lengths: {sorted(lengths)}.")
        if any(len(v) != len(self.discount) for v in self.parac.values()):
            raise ValueError("Par-yield arrays must have the same length as the discount-factor array.")

    @property
    def spotac(self) -> np.ndarray:
        """np.ndarray: Annually compounded spot (zero) rates."""
        return self.zeroac

    @property
    def spotcc(self) -> np.ndarray:
        """np.ndarray: Continuously compounded spot (zero) rates."""
        return self.zerocc

    @property
    def max_maturity(self) -> int:
        """int: Maximum term index (number of grid points minus one)."""
        return len(self.discount) - 1

    @classmethod
    def compute_parac(cls, discount: np.ndarray, interval_unit: Literal["Y", "M"]) -> dict[int, np.ndarray]:
        """Compute par yields for every supported coupon frequency on the given grid.

        Args:
            discount (np.ndarray): Discount factors.
            interval_unit (Literal["Y", "M"]): Grid interval unit, yearly or monthly.

        Returns:
            dict[int, np.ndarray]: Par yields keyed by coupon frequency.
        """
        freqs = (1,) if interval_unit == "Y" else (1, 2, 4, 12)
        return {n: InterestRateConvertor.discount_to_par(discount, coupon_frequency=n, interval_unit=interval_unit)
                for n in freqs}

    @classmethod
    def from_discount(cls, discount: npt.NDArray[np.float64], /, interval_unit: Literal["Y", "M"] = "Y") -> Self:
        """Build a term structure from discount factors.

        Args:
            discount (npt.NDArray[np.float64]): Discount factors, index `0, 1, ..., n` corresponds to term
                `0, 1, ..., n` grid steps.
            interval_unit (Literal["Y", "M"]): Grid interval unit, year or month. Defaults to year.

        Returns:
            InterestRateTermStructure: The corresponding term structure.
        """
        freq = frequency_from_interval_unit(interval_unit)
        discount = np.array(discount, dtype=float, copy=True)
        n = len(discount)
        time = np.arange(n)
        forwardac = np.zeros(n)
        forwardcc = np.zeros(n)
        zeroac = np.zeros(n)
        zerocc = np.zeros(n)
        zeroac[1:] = discount[1:] ** (-freq / time[1:]) - 1.0
        forwardac[1:] = (discount[:-1] / discount[1:]) ** freq - 1.0
        forwardcc[1:] = np.log1p(forwardac[1:])
        zerocc[1:] = np.log1p(zeroac[1:])
        return InterestRateTermStructure(
            discount=discount,
            forwardac=forwardac,
            forwardcc=forwardcc,
            zeroac=zeroac,
            zerocc=zerocc,
            parac=cls.compute_parac(discount, interval_unit),
            interval_unit=interval_unit,
        )

    @classmethod
    def from_zeroac(cls, zeroac: npt.NDArray[np.float64], /, interval_unit: Literal["Y", "M"] = "Y") -> Self:
        """Build a term structure from annually compounded zero (spot) rates.

        Args:
            zeroac (npt.NDArray[np.float64]): Annually compounded zero rates, index `0, 1, ..., n` corresponds
                to term `0, 1, ..., n` grid steps.
            interval_unit (Literal["Y", "M"]): Grid interval unit, year or month. Defaults to year.

        Returns:
            InterestRateTermStructure: The corresponding term structure.
        """
        freq = frequency_from_interval_unit(interval_unit)
        zeroac = np.array(zeroac, dtype=float, copy=True)
        n = len(zeroac)
        time = np.arange(n)
        discount = np.zeros(n)
        forwardac = np.zeros(n)
        forwardcc = np.zeros(n)
        zerocc = np.zeros(n)
        discount[0] = 1.0
        discount[1:] = (1.0 + zeroac[1:]) ** (-time[1:] / freq)
        forwardac[1:] = (discount[:-1] / discount[1:]) ** freq - 1.0
        forwardcc[1:] = np.log1p(forwardac[1:])
        zerocc[1:] = np.log1p(zeroac[1:])
        return InterestRateTermStructure(
            discount=discount,
            zeroac=zeroac,
            zerocc=zerocc,
            forwardac=forwardac,
            forwardcc=forwardcc,
            parac=cls.compute_parac(discount, interval_unit),
            interval_unit=interval_unit,
        )

    @classmethod
    def from_forwardac(cls, forwardac: npt.NDArray[np.float64], /, interval_unit: Literal["Y", "M"] = "Y") -> Self:
        """Build a term structure from annually compounded forward rates.

        Args:
            forwardac (npt.NDArray[np.float64]): Annually compounded forward rates, index `0, 1, ..., n`
                corresponds to term `0, 1, ..., n` grid steps.
            interval_unit (Literal["Y", "M"]): Grid interval unit, year or month. Defaults to year.

        Returns:
            InterestRateTermStructure: The corresponding term structure.
        """
        freq = frequency_from_interval_unit(interval_unit)
        forwardac = np.array(forwardac, dtype=float, copy=True)
        n = len(forwardac)
        time = np.arange(n)
        discount = np.zeros(n)
        forwardcc = np.zeros(n)
        zeroac = np.zeros(n)
        zerocc = np.zeros(n)
        forwardcc[1:] = np.log1p(forwardac[1:])
        discount[0] = 1.0
        discount[1:] = np.cumprod((1.0 + forwardac[1:]) ** (-1 / freq))
        zeroac[1:] = discount[1:] ** (-freq / time[1:]) - 1.0
        zerocc[1:] = np.log1p(zeroac[1:])
        return InterestRateTermStructure(
            discount=discount,
            zeroac=zeroac,
            zerocc=zerocc,
            forwardac=forwardac,
            forwardcc=forwardcc,
            parac=cls.compute_parac(discount, interval_unit),
            interval_unit=interval_unit,
        )

    def __len__(self) -> int:
        """int: Number of grid points in the term structure."""
        return len(self.discount)
