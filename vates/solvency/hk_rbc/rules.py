import math
import warnings
import numpy as np


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
CORR_MATRIX_PCA = np.array([
    [1.00, 0.25, 0.25, 0.25],
    [0.25, 1.00, 0.00, 0.25],
    [0.25, 0.00, 1.00, 0.25],
    [0.25, 0.25, 0.25, 1.00]
])

# Table 3A - Correlation Matrix for Market Risk—Interest Rate Upward
# (1) interest rate
# (2) credit spread
# (3) equity
# (4) property
# (5) currency
CORR_MATRIX_MARKET_IR_UP = np.array([
    [1.00, 0.00, 0.00, 0.00, 0.25],
    [0.00, 1.00, 0.75, 0.50, 0.25],
    [0.00, 0.75, 1.00, 0.50, 0.25],
    [0.00, 0.50, 0.50, 1.00, 0.25],
    [0.25, 0.25, 0.25, 0.25, 1.00]
])

# Table 3B - Correlation Matrix for Market Risk—Interest Rate Downward
# (1) interest rate
# (2) credit spread
# (3) equity
# (4) property
# (5) currency
CORR_MATRIX_MARKET_IR_DOWN = np.array([
    [1.00, 0.50, 0.50, 0.25, 0.25],
    [0.50, 1.00, 0.75, 0.50, 0.25],
    [0.50, 0.75, 1.00, 0.50, 0.25],
    [0.25, 0.50, 0.50, 1.00, 0.25],
    [0.25, 0.25, 0.25, 0.25, 1.00]
])

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

# Table 8 - Correlation Matrix for Reserve and Premium Risk—General Insurance Lines of Business
# pass

# Table 9 - Correlation Matrix for Catastrophe Risk
# (1) natural catastrophe
# (2) man-made non-systemic catastrophe
# (3) man-made systemic catastrophe
CORR_MATRIX_GI_CAT = np.array([
    [1.00, 0.00, 0.00],
    [0.00, 1.00, 0.00],
    [0.00, 0.00, 1.00],
])


# Table 10 - Correlation Matrix for Net Loss for Windstorm for Natural Catastrophe Risk
# pass

# Table 11 - Correlation Matrix for Net Loss for Earthquake for Natural Catastrophe Risk
# pass

# Table 12 - Correlation Matrix for Mortgage Insurance Risk
# (1) onshore mortgage insurance
# (2) offshore mortgage insurance
CORR_MATRIX_GI_MI = np.array([
    [1.00, 0.75],
    [0.75, 1.00]
])


# Table 13 - Correlation Matrix for Onshore Standard Mortgage Insurance
# pass

# Table 14 - Correlation Matrix for Offshore Mortgage Insurance
# pass

# Table 15 - Correlation Matrix for Reserve and Premium Risk for Offshore Mortgage Insurance
# pass

