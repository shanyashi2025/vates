import numpy as np

from vates.finmath.rate_conversion import InterestRateTermStructure
from vates.finmath._smith_wilson import SmithWilsonExtrapolator


def extrapolate_interest_rates(rates: np.ndarray, /, has_t0: bool, *args, method: str, **kwargs
                               ) -> InterestRateTermStructure:
    if method.lower() in ("smith_wilson", "smithwilson"):
        if has_t0:
            rates = rates[1:]  # index `0, 1, ..., n` should correspond to term `1, 2, ..., n-1`.
        return smith_wilson_extrapolation(rates, *args, **kwargs)
    if method.lower() in ("eiopa_alternative", "eiopa_alt"):
        if not has_t0:
            rates = np.insert(arr=rates, obj=0, values=0)  # index `0, 1, ..., n` should correspond to term `0, 1, ..., n`.
        return eiopa_alternative_extrapolation(rates, *args, **kwargs)
    raise ValueError(f"Invalid {method=}.")


def eiopa_alternative_extrapolation(
    rates: np.ndarray,
    *,
    instrument: str = "zero",
    ufr: float,
    fsp: int,
    alpha: float,
    llfr_weight: dict[tuple[int, int], float],
    max_maturity: int = 120,
) -> InterestRateTermStructure:
    """
    Args:
        rates: Interest rates as decimals, index `0, 1, ..., n` should correspond to term `0, 1, ..., n`.
        instrument: Instrument type: "zero".
        ufr: Ultimate forward rate as an annually compounded rate.
        fsp: First smoothing point.
        alpha: Convergence factor, determines the speed of post-FSP convergence to the UFR.
        llfr_weight: Weightings of LLFR (last liquid forward rate).
        max_maturity: Max maturity of the extrapolation.

    Returns: InterestRateTermStructure

    References:
        - EIOPA-BoS-20/750 (17 December 2020): Background Document on the Opinion on the 2020 Review of Solvency II,
            Annex 2.6 – Alternative method to derive the risk free rate term structure

    """
    if instrument.lower() not in ('zero', 'spot'):
        raise ValueError(f"{instrument=} is not supported.")

    if abs(sum(llfr_weight.values()) - 1) > 1e-8:
        raise ValueError(f"Sum of llfr weights = {sum(llfr_weight.values()):.4f}, expected 100%.")

    ln_ufr = np.log(1 + ufr)
    fv = np.array([1.0 if i == 0 else (1 + rates[i]) ** i for i in range(len(rates))])
    llfr = 0.0
    for (x, y), w in llfr_weight.items():
        if abs(w) > 1e-8:
            fxy = np.log(fv[y] / fv[x]) / (y - x)
            llfr += fxy * w

    bah = np.zeros(max_maturity + 1 - fsp)
    fhh = np.zeros(max_maturity + 1 - fsp)
    fvh = np.zeros(max_maturity + 1 - fsp)

    h = np.arange(1, max_maturity + 1 - fsp)
    bah[1:] = (1 - np.exp(-alpha * h)) / (alpha * h)
    fhh[1:] = ln_ufr + (llfr - ln_ufr) * bah[1:]
    fvh[0] = 1
    fvh[1:] = np.exp(fhh[1:] * h)


    forwardac = np.zeros(max_maturity + 1)
    forwardac[1: fsp + 1] = fv[1: fsp + 1] / fv[0: fsp] - 1
    forwardac[fsp + 1: max_maturity + 1] = fvh[1: max_maturity + 1 - fsp] / fvh[0: max_maturity - fsp] - 1

    return InterestRateTermStructure.from_forwardac(forwardac)


def smith_wilson_extrapolation(
    rates: np.ndarray,
    /,
    *,
    llp: int,
    ufr: float,
    convergence_point: int,
    min_alpha: float,
    instrument: str = "zero",
    coupon_frequency: int = 1,
    convergence_tolerance_bp: float = 0.1,
    max_maturity: int = 150
) -> InterestRateTermStructure:
    """Smith-Wilson Risk-Free Interest Rate Extrapolation Tool.

    Args:
        rates: Observed market rates as decimals. For bonds/swaps, use coupon rates or par rates per instrument,
            index `0, 1, ..., n` should correspond to term `1, 2, ..., n-1`.
        instrument: Instrument type: "zero", "bond", or "swap" (case-insensitive).
        coupon_frequency: Number of coupon payments per year for coupon instruments (1, 2, 4, or 12). Ignored for zeros.
        llp: Last liquid point.
        ufr: Ultimate forward rate as an annually compounded rate (decimal), e.g., 0.035 for 3.5%.
        convergence_point: Convergence point (in years) by which the forward rate is within the specified tolerance of
            the UFR.
        min_alpha: Minimum value of the alpha parameter used as a lower bound in calibration.
        convergence_tolerance_bp: Convergence tolerance in basis points (bp) used as tau in calibration. Default is 0.1 bp.
        max_maturity: Maximum maturity in whole years (inclusive) for which the outputs are returned. Default is 150 (year).

    Returns: InterestRateTermStructure
    """

    sw = SmithWilsonExtrapolator(
        instrument=instrument,
        coupon_frequency=coupon_frequency,
        rates=rates[: llp],
        maturities=np.arange(1, llp + 1),
        ufr=ufr,
        convergence_point=convergence_point,
        min_alpha=min_alpha,
        convergence_tolerance_bp=convergence_tolerance_bp
    )
    return sw.extrapolate(max_maturity)
