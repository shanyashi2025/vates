import numpy as np

# Insurance (Valuation and Capital) Rules (Cap. 41 sub. leg. R)

# --- Schedule 7 ---

# 1. Correlation matrix for margin over current estimate for long term insurance liabilities

# Table 1 - Correlation Matrix for Mortality, Longevity, Morbidity, Expense and Lapse Risks
# (1) mortality
# (2) longevity
# (3) morbidity
# (4) expense
# (5) lapse
CORR_MATRIX_MOCE = np.array([
    [1.00, -.25, 0.25, 0.25, 0.00],
    [-.25, 1.00, 0.00, 0.25, 0.25],
    [0.25, 0.00, 1.00, 0.50, 0.00],
    [0.25, 0.25, 0.50, 1.00, 0.50],
    [0.00, 0.25, 0.00, 0.50, 1.00]
])


# 2. Correlation matrix for prescribed capital amount and risk capital amounts

# Table 2 - Correlation Matrix for Prescribed Capital Amount
# (1) market
# (2) life insurance
# (3) general insurance
# (4) counterparty default and other
CORR_MATRIX_PCR = np.array([
    [1.00, 0.25, 0.25, 0.25],
    [0.25, 1.00, 0.00, 0.25],
    [0.25, 0.00, 1.00, 0.25],
    [0.25, 0.25, 0.25, 1.00]
])

# Table 3A/3B - Correlation Matrix for Market Risk—Interest Rate
# (1) interest rate
# (2) credit spread
# (3) equity
# (4) property
# (5) currency
def CORR_MATRIX_MARKET(which_interest_rate_risk_bites: str) -> np.ndarray:
    if which_interest_rate_risk_bites.lower() in ("increase", "upward", "up"):
        return np.array([
            [1.00, 0.00, 0.00, 0.00, 0.25],
            [0.00, 1.00, 0.75, 0.50, 0.25],
            [0.00, 0.75, 1.00, 0.50, 0.25],
            [0.00, 0.50, 0.50, 1.00, 0.25],
            [0.25, 0.25, 0.25, 0.25, 1.00]
        ])
    elif which_interest_rate_risk_bites.lower() in ("decrease", "downward", "down"):
        return np.array([
            [1.00, 0.50, 0.50, 0.25, 0.25],
            [0.50, 1.00, 0.75, 0.50, 0.25],
            [0.50, 0.75, 1.00, 0.50, 0.25],
            [0.25, 0.50, 0.50, 1.00, 0.25],
            [0.25, 0.25, 0.25, 0.25, 1.00]
        ])
    else:
        raise ValueError(f"Undefined {which_interest_rate_risk_bites=}.")

# Table 4 - Correlation Matrix for Life Insurance Risk
# (1) mortality
# (2) longevity
# (3) life catastrophe
# (4) morbidity
# (5) expense
# (6) lapse

CORR_MATRIX_LIFE = np.array([
    [1.00, -.25, 0.25, 0.25, 0.25, 0.00],
    [-.25, 1.00, 0.00, 0.00, 0.25, 0.25],
    [0.25, 0.00, 1.00, 0.25, 0.25, 0.25],
    [0.25, 0.00, 0.25, 1.00, 0.50, 0.00],
    [0.25, 0.25, 0.25, 0.50, 1.00, 0.50],
    [0.00, 0.25, 0.25, 0.00, 0.50, 1.00]
])

# Table 5 - Correlation Matrix for General Insurance Risk
# (1) general insurance (other than mortgage insurance)
# (2) mortgage insurance
CORR_MATRIX_GI = np.array([
    [1.00, 0.25],
    [0.25, 1.00]
])

# Table 6 - Correlation Matrix for General Insurance (other than Mortgage Insurance) Risk
# (1) reserve and premium
# (2) catastrophe
CORR_MATRIX_GI_EX_MI = np.array([
    [1.00, 0.25],
    [0.25, 1.00]
])

# Table 7 - Correlation Matrix for Reserve and Premium Risk
# (1) reserve
# (2) premium
CORR_MATRIX_GI_RESPREM = np.array([
    [1.00, 0.50],
    [0.50, 1.00]
])
