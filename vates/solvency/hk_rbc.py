import numpy as np
from vates.solvency.risk_tree import RiskTree, risk_aggregation

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


def _max_at_zero(**capitals: float) -> float:
    return max(max(capitals.values()), 0.0)

def _floor_at_zero(*args: float) -> tuple[float, ...]:
    return tuple([max(x, 0.0) for x in args])

def _pcr_agg(market: float, life_insurance: float, general_insurance: float, counterparty_default: float,
             ) -> float:
    return risk_aggregation(market, life_insurance, general_insurance, counterparty_default,
                            corr_matrix=CORR_MATRIX_PCR)

def _market_risk_agg(interest_rate_upward: float, interest_rate_downward: float, credit_spread: float, equity: float,
                     property_: float, currency: float, **kwargs) -> float:
    if interest_rate_upward > interest_rate_downward:
        interest_rate = interest_rate_upward
        corr_matrix = CORR_MATRIX_MARKET("upward")
    else:
        interest_rate = interest_rate_downward
        corr_matrix = CORR_MATRIX_MARKET("downward")
    return risk_aggregation(interest_rate, credit_spread, equity, property_, currency, corr_matrix=corr_matrix)


def _life_risk_agg(mortality: float, longevity: float, catastrophe: float, morbidity: float, expense: float,
                   lapse: float) -> float:
    return risk_aggregation(*_floor_at_zero(mortality, longevity, catastrophe, morbidity, expense, lapse),
                            corr_matrix=CORR_MATRIX_LIFE)

def _gi_risk_agg(reserve_premium: float, catastrophe: float, mortgage_insurance: float) -> float:
    gi_ex_mi = risk_aggregation(reserve_premium, catastrophe, corr_matrix=CORR_MATRIX_GI_EX_MI)
    return risk_aggregation(gi_ex_mi, mortgage_insurance, corr_matrix=CORR_MATRIX_GI)


def make_pcr_module(*, submodule: str | None = None) -> RiskTree:
    structure = {
        "HKRBC PCR": {
            "children": ("Market", "Life Insurance", "General Insurance", "Counterparty Default"),
            "agg_func": _pcr_agg},
        "HKRBC PCR/Market": {
            "children": ("Interest Rate", "Credit Spread", "Equity", "Property", "Currency"),
            "agg_func": _market_risk_agg, "agg_scope": "descendants"},
        "HKRBC PCR/Market/Interest Rate": {
            "children": ("Interest Rate Upward", "Interest Rate Downward"),
            "agg_func": _max_at_zero},
        "HKRBC PCR/Market/Property": {"identifier": "property_"},
        "HKRBC PCR/Life Insurance": {
            "children": ("Mortality", "Longevity", "Catastrophe", "Morbidity", "Expense", "Lapse"),
            "agg_func": _life_risk_agg},
        "HKRBC PCR/Life Insurance/Lapse": {"children": ("Level & Trend", "Mass"), "agg_func": _max_at_zero},
        "HKRBC PCR/Life Insurance/Lapse/Level & Trend": {
            "identifier": "level",
            "children": ("Lapse Upward", "Lapse Downward"),
            "agg_func": _max_at_zero},
        "HKRBC PCR/General Insurance": {
            "children": ("Reserve Premium", "Catastrophe", "Mortgage Insurance"),
            "agg_func": _gi_risk_agg},
    }

    tree = RiskTree.from_structure(structure=structure, is_lock_structure=True,
                                   is_zerofill=True)

    if submodule is None:
        return tree
    else:
        return tree.get_subtree(submodule).duplicate()

