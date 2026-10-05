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


def make_solvency2_scr_module(*, submodule: str | None = None, is_zeroize: bool = True
                              ) -> RiskTree:
    tree = RiskTree(root="Solvency II SCR")

    # (root)
    tree.root.add_sub_risk(
        "Market", "Counterparty Default", "Life", "Health", "Non-life", "Intangibles", agg_func=_scr_agg)

    # Non-life
    tree.get_node("Non-life").add_sub_risk("Premium Reserve", "Catastrophe", "Lapse", agg_func=_nonlife_risk_agg)

    # Life
    tree.get_node("Life").add_sub_risk(
        "Mortality", "Longevity", "Disability", "Expense", "Revision", "Lapse", "Catastrophe",
        agg_func=_life_risk_agg)

    # Life/Lapse
    tree.get_node("Life/Lapse").add_sub_risk(
        "Increase", "Decrease", "Mass", agg_func=lambda increase, decrease, mass: max(increase, decrease, mass, 0.0))

    # Health
    tree.get_node("Health").add_sub_risk("NSLT", "SLT", "Catastrophe", agg_func=_health_risk_agg)

    # Health/SLT
    tree.get_node("Health/SLT").add_sub_risk(
        "Mortality", "Longevity", "Disability-Morbidity", "Expense", "Revision", "Lapse", agg_func=_slth_risk_agg)

    # Health/SLT/Disability-Morbidity
    tree.get_node("Health/SLT/Disability-Morbidity").add_sub_risk(
        "Medical Payment Increase", "Medical Payment Decrease", "Income Protection",
        agg_func=lambda medical_payment_increase, medical_payment_decrease, income_protection: max(
            medical_payment_increase, medical_payment_decrease) + income_protection)

    # Health/SLT/Lapse
    tree.get_node("Health/SLT/Lapse").add_sub_risk(
        "Increase", "Decrease", "Mass", agg_func=lambda increase, decrease, mass: max(increase, decrease, mass, 0.0))

    # Market
    tree.get_node("Market").add_sub_risk(
        "Interest Rate Increase", "Interest Rate Decrease", "Equity",
        RiskNode("Property", identifier="property_"), "Spread", "Concentration", "Currency",
        agg_func=_market_risk_agg)

    # Counterparty Default
    tree.get_node("Counterparty Default").add_sub_risk("Type 1", "Type 2", agg_func=_counterparty_default_risk_agg)

    if is_zeroize:
        tree.zeroize()

    if submodule is None:
        return tree
    else:
        return tree.get_subtree(submodule).deepcopy()
