import math

from vates.solvency.risk_tree import RiskNode, RiskTree, risk_aggregation
from vates.solvency.eu_solvency2.rules import (
    CORR_MATRIX_SCR,
    CORR_MATRIX_NONLIFE,
    CORR_MATRIX_LIFE,
    CORR_MATRIX_HEALTH,
    CORR_MATRIX_SLTH,
    CORR_MATRIX_MARKET,
)

def _scr_agg(market: float, counterparty_default: float, life: float, health: float, non_life: float,
             intangibles: float) -> float:
    return risk_aggregation(market, counterparty_default, life, health, non_life,
                            corr_matrix=CORR_MATRIX_SCR) + intangibles

def _nonlife_risk_agg(premium_reserve: float, catastrophe: float, lapse: float) -> float:
    return risk_aggregation(premium_reserve, catastrophe, lapse, corr_matrix=CORR_MATRIX_NONLIFE)

def _life_risk_agg(mortality: float, longevity: float, disability: float, expense: float, revision: float, lapse: float,
                   catastrophe: float) -> float:
    return risk_aggregation(mortality, longevity, disability, expense, revision, lapse, catastrophe,
                            corr_matrix=CORR_MATRIX_LIFE)

def _health_risk_agg(nslt: float, slt: float, catastrophe: float) -> float:
    return risk_aggregation(nslt, slt, catastrophe, corr_matrix=CORR_MATRIX_HEALTH)

def _slth_risk_agg(mortality: float, longevity: float, disability_morbidity: float, expense: float, revision: float,
                   lapse: float) -> float:
    return risk_aggregation(mortality, longevity, disability_morbidity, expense, revision, lapse,
                            corr_matrix=CORR_MATRIX_SLTH)

def _market_risk_agg(interest_rate_increase: float, interest_rate_decrease: float, equity: float, property_: float,
                     spread: float, concentration: float, currency: float) -> float:
    if interest_rate_increase > interest_rate_decrease:
        interest_rate = interest_rate_increase
        corr_matrix = CORR_MATRIX_MARKET("increase")
    else:
        interest_rate = interest_rate_decrease
        corr_matrix = CORR_MATRIX_MARKET("decrease")
    return risk_aggregation(interest_rate, equity, property_, spread, concentration, currency,
                            corr_matrix=corr_matrix)

def _counterparty_default_risk_agg(type_1: float, type_2: float) -> float:
    return math.sqrt(type_1 ** 2 + 1.5 * type_1 * type_2 + type_2 ** 2)


def make_solvency2_scr_module(name: str = "Solvency II", /, submodule: str | None = None, is_zeroize: bool = True
                              ) -> RiskTree:
    tree = RiskTree(name=name, root=RiskNode("Solvency II"))

    # (root)
    node = tree.root
    node.attach_sub_risk(RiskNode("Market"), RiskNode("Counterparty Default"), RiskNode("Life"),
                         RiskNode("Health"), RiskNode("Non-life"), RiskNode("Intangibles"))
    node.set_agg_func(_scr_agg)

    # Non-life
    node = tree.get_node("Non-life")
    node.attach_sub_risk(RiskNode("Premium & Reserve", identifier="premium_reserve"),
                         RiskNode("Catastrophe"), RiskNode("Lapse"))
    node.set_agg_func(_nonlife_risk_agg)

    # Life
    node = tree.get_node("Life")
    node.attach_sub_risk(RiskNode("Mortality"), RiskNode("Longevity"), RiskNode("Disability"), RiskNode("Expense"),
                         RiskNode("Revision"), RiskNode("Lapse"), RiskNode("Catastrophe"))
    node.set_agg_func(_life_risk_agg)

    # Life/Lapse
    node = tree.get_node("Life/Lapse")
    node.attach_sub_risk(RiskNode("Increase"), RiskNode("Decrease"), RiskNode("Mass"))
    node.set_agg_func(lambda increase, decrease, mass: max(increase, decrease, mass, 0.0))

    # Health
    node = tree.get_node("Health")
    node.attach_sub_risk(RiskNode("NSLT"), RiskNode("SLT"), RiskNode("Catastrophe"))
    node.set_agg_func(_health_risk_agg)

    # Health/SLT
    node = tree.get_node("Health/SLT")
    node.attach_sub_risk(RiskNode("Mortality"), RiskNode("Longevity"), RiskNode("Disability-Morbidity"),
                         RiskNode("Expense"), RiskNode("Revision"), RiskNode("Lapse"))
    node.set_agg_func(_slth_risk_agg)

    # Market
    node = tree.get_node("Market")
    node.attach_sub_risk(RiskNode("Interest Rate Increase"), RiskNode("Interest Rate Decrease"), RiskNode("Equity"),
                         RiskNode("Property", identifier="property_"), RiskNode("Spread"),
                         RiskNode("Concentration"), RiskNode("Currency"))
    node.set_agg_func(_market_risk_agg)

    # Counterparty Default
    node = tree.get_node("Counterparty Default")
    node.attach_sub_risk(RiskNode("Type 1"), RiskNode("Type 2"))
    node.set_agg_func(_counterparty_default_risk_agg)

    if is_zeroize:
        tree.zeroize()

    if submodule is None:
        return tree
    else:
        return tree.get_subtree(submodule, name=name).deepcopy()
