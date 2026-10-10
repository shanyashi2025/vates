from vates.solvency.risk_tree import RiskTree, risk_aggregation
from vates.solvency.hk_rbc.rules import (
    CORR_MATRIX_PCR,
    CORR_MATRIX_MARKET,
    CORR_MATRIX_LIFE,
    CORR_MATRIX_GI,
    CORR_MATRIX_GI_EX_MI,
)


def _max_at_zero(**capitals: float) -> float:
    return max(max(capitals.values()), 0.0)

def _floor_at_zero(*args: float) -> tuple[float, ...]:
    return tuple([max(x, 0.0) for x in args])

def _pcr_agg(market: float, life_insurance: float, general_insurance: float, counterparty_default: float,
             operational: float) -> float:
    return risk_aggregation(market, life_insurance, general_insurance, counterparty_default,
                            corr_matrix=CORR_MATRIX_PCR) + operational

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


def make_hkrbc_pcr_module(*, submodule: str | None = None) -> RiskTree:
    structure = {
        "HKRBC PCR": {
            "children": ("Market", "Life Insurance", "General Insurance", "Counterparty Default", "Operational"),
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
                                   is_zeroize=True)

    if submodule is None:
        return tree
    else:
        return tree.get_subtree(submodule).duplicate()
