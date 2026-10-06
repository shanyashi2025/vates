from vates.solvency.risk_tree import RiskNode, RiskTree, risk_aggregation
from vates.solvency.hk_rbc.rules import (
    CORR_MATRIX_PCR,
    CORR_MATRIX_MARKET,
    CORR_MATRIX_LIFE,
    CORR_MATRIX_GI,
    CORR_MATRIX_GI_EX_MI,
)


def _max_at_zero(**capitals: float) -> float:
    return max(max(capitals.values()), 0.0)

def _pcr_agg(market: float, life_insurance: float, general_insurance: float, counterparty_default: float,
             operational: float) -> float:
    return risk_aggregation(market, life_insurance, general_insurance, counterparty_default,
                            corr_matrix=CORR_MATRIX_PCR) + operational

def _market_risk_agg(interest_rate_upward: float, interest_rate_downward: float, credit_spread: float, equity: float,
                     property_: float, currency: float) -> float:
    if interest_rate_upward > interest_rate_downward:
        interest_rate = interest_rate_upward
        corr_matrix = CORR_MATRIX_MARKET("upward")
    else:
        interest_rate = interest_rate_downward
        corr_matrix = CORR_MATRIX_MARKET("downward")
    return risk_aggregation(interest_rate, credit_spread, equity, property_, currency, corr_matrix=corr_matrix)


def _life_risk_agg(mortality: float, longevity: float, catastrophe: float, morbidity: float, expense: float,
                   lapse: float) -> float:
    return risk_aggregation(mortality, longevity, catastrophe, morbidity, expense, lapse,
                            corr_matrix=CORR_MATRIX_LIFE)

def _gi_risk_agg(reserve_premium: float, catastrophe: float, mortgage_insurance: float) -> float:
    gi_ex_mi = risk_aggregation(reserve_premium, catastrophe, corr_matrix=CORR_MATRIX_GI_EX_MI)
    return risk_aggregation(gi_ex_mi, mortgage_insurance, corr_matrix=CORR_MATRIX_GI)


def make_hkrbc_pcr_module(*, submodule: str | None = None, is_zeroize: bool = True) -> RiskTree:
    tree = RiskTree(root="HKRBC PCR")

    # (root)
    tree.set_up_node("",
                     children=("Market", "Life Insurance", "General Insurance", "Counterparty Default", "Operational"),
                     agg_func=_pcr_agg)

    # Market
    tree.set_up_node("Market",
                     children=("Interest Rate Upward", "Interest Rate Downward", "Credit Spread", "Equity",
                               RiskNode("Property", identifier="property_"), "Currency"),
                     agg_func=_market_risk_agg)

    # Life Insurance
    tree.set_up_node("Life Insurance",
                     children=("Mortality", "Longevity", "Catastrophe", "Morbidity", "Expense", "Lapse"),
                     agg_func=_life_risk_agg)

    # Life Insurance/Lapse
    tree.set_up_node("Life Insurance/Lapse", children=("Level", "Mass"), agg_func=_max_at_zero)

    # General Insurance
    tree.set_up_node("General Insurance", children=("Reserve Premium", "Catastrophe", "Mortgage Insurance"),
                     agg_func=_gi_risk_agg)

    if is_zeroize:
        tree.zeroize()

    if submodule is None:
        return tree
    else:
        return tree.get_subtree(submodule).deepcopy()
