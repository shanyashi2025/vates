import math
import warnings
import numpy as np
from enum import Enum, unique

@unique
class AccountType(Enum):
    """Enum for C-ROSS Account Type."""
    TRAD = "TRAD"  # Traditional
    PAR = "PAR"  # Participating
    UNIV = "UNIV"  # Universal
    ILNK = "ILNK"  # Investment_Linked
    SH = "SH"  # Shareholder

# --- Article 18, Rule No.2 ---
# Correlation matrix (4 x 4) for minimum capital aggregation, indexed by:
#   (1) life
#   (2) non-life
#   (3) market
#   (4) credit
MC_CORR_MATRIX = np.array([[1.00, 0.20, 0.30, 0.15],
                           [0.20, 1.00, 0.10, 0.10],
                           [0.30, 0.10, 1.00, 0.35],
                           [0.15, 0.10, 0.35, 1.00]
                           ])

# --- Article 19, Rule No.2 ---
LA_RHO = 0.35  # Correlation coefficient between market risk and credit risk
LA_K = 0  # Note: K is published by regulator


def calculate_loss_absorbency(mc_market: float, mc_credit: float, pv_base: float, pv_lower_limit: float) -> float:
    la_upper_limit = max(pv_base - pv_lower_limit, 0)
    mc_market_credit = aggregate_market_credit_risk(mc_market, mc_credit)
    beta = calculate_beta(mc_market_credit, la_upper_limit)
    return min(mc_market_credit * beta, la_upper_limit)


def aggregate_market_credit_risk(mc_market: float, mc_credit: float) -> float:
    return math.sqrt(mc_market ** 2 + 2 * LA_RHO * mc_market * mc_credit + mc_credit ** 2)


def calculate_beta(mc_market_credit: float, la_upper_limit: float) -> float:
    if mc_market_credit < 0:
        warnings.warn(f'MC of market and credit={mc_market_credit:.4f}, expected > 0, '
                      f'beta is assgined as zero to prevent crash.')
        return 0
    return (1 + LA_K) * min(0.5, 0.22 * la_upper_limit / mc_market_credit + 0.02)


# --- Article 12, Rule No.5 ---
# Correlation matrix (2 x 2) for minimum morbidity risk capital aggregation, indexed by:
#   (1) morbidity incidence
#   (2) morbidity trend
MORB_MC_CORR_MATRIX = np.array([[1.00, 0.25],
                                [0.25, 1.00]
                                ])

# --- Article 16, Rule No.5 ---
# Correlation matrix (6 x 6) for minimum loss rate risk capital aggregation, indexed by:
#   (1) death
#   (2) catastrophe
#   (3) longevity
#   (4) morbidity
#   (5) medical and health indemnity loss rate
#   (6) others
LOSS_MC_CORR_MATRIX = np.array([[1.00, 0.25, -.25, 0.25, 0.25, 0.25],
                                [0.25, 1.00, 0.00, 0.25, 0.25, 0.25],
                                [-.25, 0.00, 1.00, 0.00, 0.00, 0.00],
                                [0.25, 0.25, 0.00, 1.00, 0.25, 0.25],
                                [0.25, 0.25, 0.00, 0.25, 1.00, 0.25],
                                [0.25, 0.25, 0.00, 0.25, 0.25, 1.00]
                                ])

# --- Article 25, Rule No.5 ---
# Correlation matrix (3 x 3) for minimum life insurance risk capital aggregation, indexed by:
#   (1) loss rate
#   (2) cost / expense
#   (3) surrender
LIFE_MC_CORR_MATRIX = np.array([[1.00, 0.40, 0.00],
                                [0.40, 1.00, 0.50],
                                [0.00, 0.50, 1.00]
                                ])

# --- Article 70, Rule No.8 ---
# Correlation matrix (6 x6) for minimum market risk capital aggregation, indexed by:
#   (1) interest rate
#   (2) equity price
#   (3) real estate
#   (4) overseas fixed-income
#   (5) overseas equity
#   (6) exchange rate
MARKET_MC_CORR_MATRIX = np.array([[1.00, -.14, -.18, 0.00, -.16, 0.07],
                                  [-.14, 1.00, 0.22, 0.06, 0.50, 0.04],
                                  [-.18, 0.22, 1.00, 0.18, 0.19, -.14],
                                  [0.00, 0.06, 0.18, 1.00, 0.04, -.01],
                                  [-.16, 0.50, 0.19, 0.04, 1.00, -.19],
                                  [0.07, 0.04, -.14, -.01, -.19, 1.00]
                                  ])

# --- Article 43, Rule No.9 ---
# Correlation matrix (2 x 2) for minimum credit risk capital aggregation, indexed by:
#   (1) spread
#   (2) counterparty default
CREDIT_MC_CORR_MATRIX = np.array([[1.00, 0.25],
                                  [0.25, 1.00]
                                  ])

# --- Article 121, Rule No.12 ---
# Risk factor to be applied on minimum control risk capital.
def calculate_control_risk_factor(score: float) -> float:
    if 0 <= score <= 70:
        return -0.01 * score + 0.75
    elif 70 < score <= 90:
        return -0.005 * score + 0.4
    elif 90 < score <= 100:
        return -0.01 * score + 0.85
    else:
        warnings.warn(f'Invalid range of {score=:.2f}, expected 0 to 100, factor=0 is returned.')
        return 0


# --- Interest Rate Risk ---
_USR = 0.045
_INT_UP_FACTOR = np.array([
    1.00,
    1.97, 1.76, 1.68, 1.65, 1.66, 1.61, 1.55, 1.53, 1.52, 1.50,
    1.49, 1.47, 1.45, 1.42, 1.41, 1.39, 1.38, 1.38, 1.38, 1.37,
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.17,
])
_INT_DN_FACTOR = np.array([
    1.00,
    0.29, 0.34, 0.39, 0.46, 0.52, 0.55, 0.58, 0.61, 0.64, 0.66,
    0.68, 0.70, 0.72, 0.73, 0.75, 0.76, 0.77, 0.77, 0.77, 0.77,
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.89,
])

def interest_risk_discount_curve(gby_60d_ma: np.ndarray, max_maturity: int=50
                                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_in = gby_60d_ma[:41]
    intba_curve = base_curve_quadratic_interpolation(
        arr_rate_in=gby_60d_ma[:41], ult_spot_rate=_USR,
        start=20, end=40, max_maturity=max_maturity
    )
    intup_curve = base_curve_quadratic_interpolation(
        arr_rate_in=base_in * _INT_UP_FACTOR, ult_spot_rate=float(_USR * _INT_UP_FACTOR[40]),
        start=20, end=40, max_maturity=max_maturity
    )
    intdn_curve = base_curve_quadratic_interpolation(
        arr_rate_in=base_in * _INT_DN_FACTOR, ult_spot_rate=float(_USR * _INT_DN_FACTOR[40]),
        start=20, end=40, max_maturity=max_maturity
    )
    spread_curve = spread_interpolation(0.045, 0, 20, 40, max_maturity)

    return intba_curve + spread_curve, intup_curve + spread_curve, intdn_curve + spread_curve


def base_curve_quadratic_interpolation(arr_rate_in: np.ndarray, ult_spot_rate: float,
                                       start: int = 20, end: int = 40, max_maturity: int = 50) -> np.ndarray:
    # --- first interpolation ---
    arr_rate_1: np.ndarray = np.zeros(max_maturity + 1)
    arr_rate_1[1:start + 1] = arr_rate_in[1:start + 1]
    arr_rate_1[end + 1:] = ult_spot_rate
    k = (ult_spot_rate - arr_rate_in[start]) / (end - start)
    for t in range(start + 1, end + 1):
        arr_rate_1[t] = arr_rate_in[start] + k * (t - start)

    # --- second interpolation ---
    arr_rate_out: np.ndarray = arr_rate_1.copy()
    for t in range(start + 1, end + 1):
        w = (t - start) / (end - start)
        arr_rate_out[t] = arr_rate_1[t] * w + arr_rate_in[t] * (1 - w)

    return arr_rate_out

def spread_interpolation(spread_start: float, spread_end: float = 0,
                         start: int = 20, end: int = 40, max_maturity: int = 50) -> np.ndarray:
    arr_spread: np.ndarray = np.zeros(max_maturity + 1)
    arr_spread[1:start + 1] = spread_start
    arr_spread[end + 1:] = spread_end
    for t in range(start + 1, end + 1):
        w = (t - start) / (end - start)
        arr_spread[t] = spread_end * w + arr_spread[start] * (1 - w)
    return arr_spread
