from vates.solvency.risk_node import RiskNode, RiskTree, risk_aggregation
from vates.solvency.hk_rbc.rules import (
    CORR_MATRIX_PCA,
    CORR_MATRIX_MARKET_IR_UP,
    CORR_MATRIX_MARKET_IR_DOWN,
    CORR_MATRIX_LIFE,
    CORR_MATRIX_GI,
    CORR_MATRIX_GI_EX_MI,
    CORR_MATRIX_GI_CAT,
    CORR_MATRIX_GI_MI,
)


def _pcr_agg(market: float, life_insurance: float, general_insurance: float, counterparty_default: float,
             operational: float) -> float:
    return risk_aggregation(market, life_insurance, general_insurance, counterparty_default,
                            corr_matrix=CORR_MATRIX_PCA) + operational

def _market_risk_agg(interest_rate_up: float, interest_rate_down: float, credit_spread: float, equity: float,
                     property_: float, currency: float) -> float:
    if interest_rate_up > interest_rate_down:
        interest_rate = interest_rate_up
        corr_matrix = CORR_MATRIX_MARKET_IR_UP
    else:
        interest_rate = interest_rate_down
        corr_matrix = CORR_MATRIX_MARKET_IR_DOWN
    return risk_aggregation(interest_rate, credit_spread, equity, property_, currency, corr_matrix=corr_matrix)


def _life_risk_agg(mortality: float, longevity: float, life_catastrophe: float, morbidity: float,
                   expense: float, lapse: float) -> float:
    return risk_aggregation(mortality, longevity, life_catastrophe, morbidity, expense, lapse,
                            corr_matrix=CORR_MATRIX_LIFE)

def _gi_risk_agg(gi_ex_mi: float, mortgage_insurance: float) -> float:
    return risk_aggregation(gi_ex_mi, mortgage_insurance, corr_matrix=CORR_MATRIX_GI)

def _gi_ex_mi_risk_agg(gi_reserve_premium: float, gi_catastrophe: float) -> float:
    return risk_aggregation(gi_reserve_premium, gi_catastrophe, corr_matrix=CORR_MATRIX_GI_EX_MI)

def _gi_cat_risk_agg(gi_cat_nature: float, gi_cat_man_nonsys: float, gi_cat_man_sys: float) -> float:
    return risk_aggregation(gi_cat_nature, gi_cat_man_nonsys, gi_cat_man_sys, corr_matrix=CORR_MATRIX_GI_CAT)

def _gi_mi_risk_agg(onshore_mi: float, offshore_mi) -> float:
    return risk_aggregation(onshore_mi, offshore_mi, corr_matrix=CORR_MATRIX_GI_MI)


def make_hkrbc_pcr_module(name: str, /, simplify_gi: bool = True) -> RiskTree:
    tree = RiskTree(any_node=RiskNode("HKRBC"), name=name)

    # (root)
    node = tree.root
    node.attach_sub_risk(RiskNode("Market"), RiskNode("Life Insurance"), RiskNode("General Insurance"),
                         RiskNode("Counterparty Default"), RiskNode("Operational"))
    node.set_agg_func(_pcr_agg)

    # Market
    node = tree.select("Market")
    node.attach_sub_risk(RiskNode("Interest Rate Up"), RiskNode("Interest Rate Down"), RiskNode("Credit Spread"),
                         RiskNode("Equity"), RiskNode("Property", slug="property_"), RiskNode("Currency"))
    node.set_agg_func(_market_risk_agg)

    # Life Insurance
    node = tree.select("Life Insurance")
    node.attach_sub_risk(RiskNode("Mortality"), RiskNode("Longevity"), RiskNode("Life Catastrophe"),
                         RiskNode("Morbidity"), RiskNode("Expense"), RiskNode("Lapse"))
    node.set_agg_func(_life_risk_agg)

    # Life Insurance/Lapse
    node = tree.select("Life Insurance/Lapse")
    node.attach_sub_risk(RiskNode("Level & Trend", slug="lapse_level"), RiskNode("Mass Lapse"))
    node.set_agg_func(lambda lapse_level, mass_lapse: max(lapse_level, mass_lapse, 0.0))

    if not simplify_gi:
        # General Insurance
        node = tree.select("General Insurance")
        node.attach_sub_risk(RiskNode("Other than mortgage insurance", slug="gi_ex_mi"),
                             RiskNode("Mortgage Insurance"))
        node.set_agg_func(_gi_risk_agg)

        # General Insurance/Other than mortgage insurance
        node = tree.select("General Insurance/Other than mortgage insurance")
        node.attach_sub_risk(RiskNode("Reserve & Premium", slug="gi_reserve_premium"),
                             RiskNode("Catastrophe", slug="gi_catastrophe"))
        node.set_agg_func(_gi_ex_mi_risk_agg)

        # General Insurance/Other than mortgage insurance/Catastrophe
        node = tree.select("General Insurance/Other than mortgage insurance/Catastrophe")
        node.attach_sub_risk(RiskNode("nature", slug="gi_cat_nature"),
                             RiskNode("man-made non-systemic", slug="gi_cat_man_nonsys"),
                             RiskNode("man-made systemic", slug="gi_cat_man_sys"))
        node.set_agg_func(_gi_cat_risk_agg)

        # General Insurance/Mortgage Insurance
        node = tree.select("General Insurance/Mortgage Insurance")
        node.attach_sub_risk(RiskNode("onshore", slug="onshore_mi"), RiskNode("offshore", slug="offshore_mi"))
        node.set_agg_func(_gi_mi_risk_agg)

    return tree
