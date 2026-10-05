import numpy as np

# (I)  COMMISSION DELEGATED REGULATION (EU) 2015/35
# (II) DIRECTIVE 2009/138/EC OF THE EUROPEAN PARLIAMENT AND OF THE COUNCIL


# 1. Directive 2009/138/EC, Annex IV, point (1)
# Correlation Matrix for Basic SCR
# (1) market
# (2) default
# (3) life
# (4) health
# (5) non-life
CORR_MATRIX_SCR = np.array([
    [1.00, 0.25, 0.25, 0.25, 0.25],
    [0.25, 1.00, 0.25, 0.25, 0.50],
    [0.25, 0.25, 1.00, 0.25, 0.00],
    [0.25, 0.25, 0.25, 1.00, 0.00],
    [0.25, 0.50, 0.00, 0.00, 1.00]
])

# 2. SECTION 2, Article 114, point (3)
# Correlation Matrix for Non-life SCR
# (1) non-life premium and reserve
# (2) non-life catastrophe
# (3) non-life lapse
CORR_MATRIX_NONLIFE = np.array([
    [1.00, 0.25, 0.00],
    [0.25, 1.00, 0.00],
    [0.00, 0.00, 1.00]
])

# 3. SECTION 3, Article 136, point (3)
# Correlation Matrix for Life SCR
# (1) mortality
# (2) longevity
# (3) disability
# (4) life expense
# (5) revision
# (6) lapse
# (7) life catastrophe
CORR_MATRIX_LIFE = np.array([
    [1.00, -.25, 0.25, 0.25, 0.00, 0.00, 0.25],
    [-.25, 1.00, 0.00, 0.25, 0.25, 0.25, 0.00],
    [0.25, 0.00, 1.00, 0.50, 0.00, 0.00, 0.25],
    [0.25, 0.25, 0.50, 1.00, 0.50, 0.50, 0.25],
    [0.00, 0.25, 0.00, 0.50, 1.00, 0.00, 0.00],
    [0.00, 0.25, 0.00, 0.50, 0.00, 1.00, 0.25],
    [0.25, 0.00, 0.25, 0.25, 0.00, 0.25, 1.00]
])

# 4. SECTION 4, Article 144, point (3)
# Correlation Matrix for Health SCR
# (1) NSLT health underwriting
# (2) SLT health underwriting
# (3) Health catastrophe
CORR_MATRIX_HEALTH = np.array([
    [1.00, 0.50, 0.25],
    [0.50, 1.00, 0.25],
    [0.25, 0.25, 1.00]
])

# 5. SECTION 4, Article 151, point (3)
# Correlation Matrix for SLT Health SCR
# (1) health mortality
# (2) health longevity
# (3) health disability-morbidity
# (4) health expense
# (5) health revision
# (6) SLT health lapse
CORR_MATRIX_SLTH = np.array([
    [1.00, -.25, 0.25, 0.25, 0.00, 0.00],
    [-.25, 1.00, 0.00, 0.25, 0.25, 0.25],
    [0.25, 0.00, 1.00, 0.50, 0.00, 0.00],
    [0.25, 0.25, 0.50, 1.00, 0.50, 0.50],
    [0.00, 0.25, 0.00, 0.50, 1.00, 0.00],
    [0.00, 0.25, 0.00, 0.50, 0.00, 1.00]
])

# 6. SECTION 5, Article 164, point (3)
# Correlation Matrix for Market SCR
# (1) interest rate
# (2) equity
# (3) property
# (4) spread
# (5) concentration
# (6) currency
def CORR_MATRIX_MARKET(which_interest_rate_risk_bites: str) -> np.ndarray:
    if which_interest_rate_risk_bites.lower() in ("increase", "upward", "up"):
        a = 0.00
    elif which_interest_rate_risk_bites.lower() in ("decrease", "downward", "down"):
        a = 0.50
    else:
        raise ValueError(f"Undefined {which_interest_rate_risk_bites=}.")
    return np.array([
        [1.00,    a,    a,    a, 0.00, 0.25],
        [   a, 1.00, 0.75, 0.75, 0.00, 0.25],
        [   a, 0.75, 1.00, 0.50, 0.00, 0.25],
        [   a, 0.75, 0.50, 1.00, 0.00, 0.25],
        [0.00, 0.00, 0.00, 0.00, 1.00, 0.00],
        [0.25, 0.25, 0.25, 0.25, 0.00, 1.00],
    ])
