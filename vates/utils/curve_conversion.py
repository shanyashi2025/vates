import numpy as np
import numpy.typing as npt


def _time_interval(term_type: str) -> float:
    """
    Get the time interval in years for a given term type.

    Args:
        term_type (str): 'M' for monthly, 'A' for annual.

    Returns:
        float: Time interval in years.

    Raises:
        ValueError: If term_type is not 'M' or 'A'.
    """
    time_intervals: dict[str, float] = {"M": 1 / 12, "A": 1}
    if term_type.upper() not in time_intervals: raise ValueError(f"Invalid term type: {term_type}. Must be 'M' or 'A'")
    return time_intervals[term_type.upper()]

def convert_spot_to_disc(spots: npt.NDArray[np.float64], term_type: str) -> npt.NDArray[np.float64]:
    """
    Convert spot rates to discount factors.

    Args:
        spots (npt.NDArray[np.float64]): Array of spot rates.
        term_type (str): 'M' for monthly, 'A' for annual.

    Returns:
        npt.NDArray[np.float64]: Array of discount factors.
    """
    periods = np.arange(len(spots)) * _time_interval(term_type)
    return (1 + spots) ** (-periods)


def convert_fwrd_to_disc(fwrds: npt.NDArray[np.float64], term_type: str) -> npt.NDArray[np.float64]:
    """
    Convert forward rates to discount factors.

    Args:
        fwrds (npt.NDArray[np.float64]): Array of forward rates.
        term_type (str): 'M' for monthly, 'A' for annual.

    Returns:
        npt.NDArray[np.float64]: Array of discount factors.
    """
    factors = (1 + fwrds) ** (-_time_interval(term_type))
    return np.cumprod(factors, dtype=float)


def convert_disc_to_spot(discs: npt.NDArray[np.float64], term_type: str) -> npt.NDArray[np.float64]:
    """
    Convert discount factors to spot rates.

    Args:
        discs (npt.NDArray[np.float64]): Array of discount factors.
        term_type (str): 'M' for monthly, 'A' for annual.

    Returns:
        npt.NDArray[np.float64]: Array of spot rates.
    """
    periods = np.arange(len(discs)) * _time_interval(term_type)
    spots = np.zeros(len(discs))
    spots[1:] = discs[1:] ** (-1 / periods[1:]) - 1
    return spots


def convert_disc_to_fwrd(discs: npt.NDArray[np.float64], term_type: str) -> npt.NDArray[np.float64]:
    """
    Convert discount factors to forward rates.

    Args:
        discs (npt.NDArray[np.float64]): Array of discount factors.
        term_type (str): 'M' for monthly, 'A' for annual.

    Returns:
        npt.NDArray[np.float64]: Array of forward rates.
    """
    fwrds = np.zeros(len(discs))
    fwrds[1:] = (discs[:-1] / discs[1:]) ** (1 / _time_interval(term_type)) - 1
    return fwrds


def convert_disc_to_par(discs: npt.NDArray[np.float64], coupon_freq: int, term_type) -> npt.NDArray[np.float64]:
    """
    Convert discount factors to par yields.

    Args:
        discs (npt.NDArray[np.float64]): Array of discount factors.
        coupon_freq (int): Coupon frequency (1, 2, 4, or 12).
        term_type (str): 'M' for monthly, 'A' for annual.

    Returns:
        npt.NDArray[np.float64]: Array of par yields.

    Raises:
        ValueError: If payment_freq or term_type is invalid.
    """
    if coupon_freq not in (1, 2, 4, 12):
        raise ValueError(f"Invalid payment frequency: {coupon_freq}. Must be 1, 2, 4, or 12")

    if term_type not in ("M", "A"):
        raise ValueError(f"Invalid term type: {term_type}. Must be 'M' or 'A'")

    if coupon_freq != 1 and term_type == "A":
        raise ValueError(f"Cannot calculate for payment frequency {coupon_freq} with annual term type")

    step = 1 if (term_type == "A" and coupon_freq == 1) else 12 // coupon_freq

    pars = np.zeros(len(discs))
    ann_factor = 0.0
    for i in range(step, len(discs), step):
        ann_factor += discs[i]
        pars[i] = (1 - discs[i]) / ann_factor * coupon_freq

    return pars


def convert_spot_to_fwrd(spots: npt.NDArray[np.float64], term_type: str) -> npt.NDArray[np.float64]:
    """
    Convert spot rates to forward rates.

    Args:
        spots (npt.NDArray[np.float64]): Array of spot rates.
        term_type (str): 'M' for monthly, 'A' for annual.

    Returns:
        npt.NDArray[np.float64]: Array of forward rates.
    """
    discs = convert_spot_to_disc(spots, term_type)
    return convert_disc_to_fwrd(discs, term_type)


def convert_spot_to_par(spots: npt.NDArray[np.float64], payment_freq: int, term_type: str) -> \
        npt.NDArray[np.float64]:
    """
    Convert spot rates to par yields.

    Args:
        spots (npt.NDArray[np.float64]): Array of spot rates.
        payment_freq (int): Payment frequency (1, 2, 4, or 12).
        term_type (str): 'M' for monthly, 'A' for annual.

    Returns:
        npt.NDArray[np.float64]: Array of par yields.
    """
    discs = convert_spot_to_disc(spots, term_type)
    return convert_disc_to_par(discs, payment_freq, term_type)


def convert_fwrd_to_spot(fwrds: npt.NDArray[np.float64], term_type: str) -> npt.NDArray[np.float64]:
    """
    Convert forward rates to spot rates.

    Args:
        fwrds (npt.NDArray[np.float64]): Array of spot rates.
        term_type (str): 'M' for monthly, 'A' for annual.

    Returns:
        npt.NDArray[np.float64]: Array of forward rates.
    """
    discs = convert_fwrd_to_disc(fwrds, term_type)
    return convert_disc_to_spot(discs, term_type)


def newton_raphson_z_spread(target_pv: float, cash_flows: npt.NDArray[np.float64],
                            spots: npt.NDArray[np.float64]) -> float:
    """
    Calculate z-spread using the Newton-Raphson method.

    Args:
        target_pv (float): Target present value.
        cash_flows (npt.NDArray[np.float64]): Array of cash flows.
        spots (npt.NDArray[np.float64]): Array of spot rates.

    Returns:
        float: Calculated z-spread.

    Raises:
        ValueError: If the method does not converge or input is invalid.
    """
    tolerance = 1e-10  # Numerical tolerance for calculations
    max_iterations = 100  # Maximum iterations for iterative methods
    epsilon = 0.0001  # Small increment for numerical derivative

    if abs(target_pv) < tolerance:
        raise ValueError("Target present value cannot be zero")

    z = 0.0  # initial guess

    for _ in range(max_iterations):  # max iterations
        spots_plus_z = spots + z
        dfs = convert_spot_to_disc(spots_plus_z, "M")
        pv = np.dot(cash_flows, dfs[1:len(cash_flows) + 1])

        if abs(pv / target_pv - 1) < tolerance:
            return z

        # Newton-Raphson approximation
        delta = epsilon if pv > target_pv else (- epsilon)
        dfs_delta = convert_spot_to_disc(spots_plus_z + delta, "M")
        pv_delta = np.dot(cash_flows, dfs_delta[1:len(cash_flows) + 1])
        derivative = (pv_delta - pv) / delta

        if abs(derivative) < tolerance:
            return z

        z = z - (pv - target_pv) / derivative

    raise ValueError(f"Newton-Raphson method did not converge after {max_iterations} iterations")


def newton_raphson_ytm(target_pv: float, cash_flows: npt.NDArray[np.float64],
                       freq: int, initial_guess: float=0.0) -> float:
    """
    Calculate yield to maturity using the Newton-Raphson method.

    Args:
        target_pv (float): Target present value.
        cash_flows (npt.NDArray[np.float64]): Array of cash flows.
        freq (int): Payment frequency, [1, 2, 4, 12]
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

    periods = np.arange(1, len(cash_flows) + 1)
    ytm = initial_guess

    for _ in range(max_iterations):
        df = (1 / (1 + ytm / freq)) ** (1 / 12 * freq)
        dfs = df ** periods
        pv = np.dot(cash_flows, dfs)

        # Check convergence
        if abs(pv / target_pv - 1) < tolerance:
            return ytm

        # Calculate numerical derivative
        df_up = (1 / (1 + (ytm + epsilon) / freq)) ** (1 / 12 * freq)
        dfs_up = df_up ** periods
        pv_up = np.dot(cash_flows, dfs_up)

        derivative = (pv_up - pv) / epsilon

        # Check if derivative is too small
        if abs(derivative) < tolerance:
            return ytm

        # Newton-Raphson update
        ytm = ytm - (pv - target_pv) / derivative

    raise ValueError(f"Newton-Raphson method did not converge after {max_iterations} iterations")


def calculate_risk_adj_spot(rf_spots: npt.NDArray[np.float64], mult: npt.NDArray[np.float64],
                            add: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Calculate risk-adjusted spot rates.

    Args:
        rf_spots (npt.NDArray[np.float64]): Risk-free spot rates.
        mult (npt.NDArray[np.float64]): Multiplicative adjustment factors.
        add (npt.NDArray[np.float64]): Additive spread adjustments.

    Returns:
        npt.NDArray[np.float64]: Risk-adjusted spot rates.
    """
    len_spot = len(rf_spots)

    def align_len(arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        len_arr = len(arr)
        if len_arr == len_spot:
            return arr
        elif len_arr > len_spot:
            return arr[len_spot]
        else:  # len_arr < len_spot
            new_arr = np.full(len_spot, arr[-1])
            new_arr[:len_arr] = arr
            return new_arr

    mult, add = align_len(mult), align_len(add)
    return rf_spots * (1 + mult) + add
